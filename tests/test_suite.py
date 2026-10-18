from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from data.build_dataset import generate_calibrated_panel
from src.audit import check_causality_truncation, load_strategy_fn, run_full_audit
from src.backtest import coerce_weights, run_backtest
from src.features import (
    cross_sectional_zscore,
    frac_diff_ffd,
    parkinson_volatility,
    rebalance_deadband,
    taker_flow_imbalance,
)
from src.quality import audit_dataset, autoclean_market_data
from src.schema import build_dollar_bars, enforce_schema, normalize_binance_klines
from src.stats import deflated_sharpe_ratio, diagnose_series, information_coefficient


@pytest.fixture(scope="module")
def sample_panel() -> pd.DataFrame:
    return generate_calibrated_panel(
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"),
        n_hours=720,
        seed=7,
    )


def test_binance_schema_and_dollar_bars(sample_panel: pd.DataFrame) -> None:
    raw_kline = [
        [
            1735689600000,
            "95000.0",
            "95500.0",
            "94800.0",
            "95200.0",
            "120.5",
            1735693199999,
            "11471600.0",
            4520,
            "65.0",
            "6188000.0",
            "0",
        ]
    ]
    funding = pd.DataFrame([{"fundingTime": 1735689600000, "fundingRate": "0.00012"}])
    norm = normalize_binance_klines(raw_kline, "btcusdt", funding)
    assert len(norm) == 1
    assert norm.loc[0, "symbol"] == "BTCUSDT"
    assert np.isclose(norm.loc[0, "funding_rate"], 0.00012)

    dbars = build_dollar_bars(sample_panel, threshold_usd=25_000_000.0)
    assert len(dbars) < len(sample_panel)
    assert set(dbars.columns) == set(sample_panel.columns)


def test_data_quality_and_autoclean(sample_panel: pd.DataFrame) -> None:
    dirty = sample_panel.copy()
    dirty = pd.concat([dirty, dirty.iloc[[0]]], ignore_index=True)
    dirty.loc[5, "high"] = dirty.loc[5, "low"] * 0.8
    dirty.loc[10, "taker_buy_quote_volume"] = dirty.loc[10, "quote_volume"] * 1.5

    raw_report = audit_dataset(dirty)
    assert raw_report.duplicate_keys >= 1
    assert raw_report.ohlc_violations >= 2

    cleaned = autoclean_market_data(dirty)
    clean_report = audit_dataset(cleaned)
    assert clean_report.duplicate_keys == 0
    assert clean_report.ohlc_violations == 0
    assert clean_report.dqs >= 90.0


def test_features_stationarity_and_deadband(sample_panel: pd.DataFrame) -> None:
    btc = sample_panel.loc[sample_panel["symbol"] == "BTCUSDT"].reset_index(drop=True)
    log_p = pd.Series(np.log(btc["close"].to_numpy()), index=btc.index)
    fdiff = frac_diff_ffd(log_p, d=0.45, thres=1e-2, max_len=48)
    valid = fdiff.dropna()
    assert len(valid) > 500

    diag_fdiff = diagnose_series(valid)
    assert diag_fdiff.n_obs == len(valid)
    assert diag_fdiff.adf_tstat < -1.95

    imb = taker_flow_imbalance(sample_panel, span=6)
    pvol = parkinson_volatility(sample_panel, window=12)
    w_raw = cross_sectional_zscore(sample_panel, imb / pvol.replace(0.0, np.nan))

    by_ts_sum = w_raw.groupby(sample_panel["timestamp"]).sum().abs().max()
    by_ts_l1 = w_raw.groupby(sample_panel["timestamp"]).apply(lambda x: x.abs().sum())
    assert by_ts_sum < 1e-10
    assert np.allclose(by_ts_l1[by_ts_l1 > 0], 1.0, atol=1e-6)

    # Verify rebalance_deadband reduces turnover and handles non-RangeIndex DataFrames
    shifted_df = sample_panel.copy()
    shifted_df.index = shifted_df.index + 1000
    w_shifted = pd.Series(w_raw.to_numpy(), index=shifted_df.index)
    w_db = rebalance_deadband(shifted_df, w_shifted, buffer=0.05)

    bt_raw = run_backtest(sample_panel, w_raw, fee_bps=5.0)
    bt_db = run_backtest(sample_panel, w_db.to_numpy(), fee_bps=5.0)
    assert bt_db.mean_turnover < bt_raw.mean_turnover


def test_deflated_sharpe_and_vectorized_ic(sample_panel: pd.DataFrame) -> None:
    rng = np.random.default_rng(123)
    rets = rng.normal(0.00025, 0.004, size=2000)
    dsr_1 = deflated_sharpe_ratio(rets, n_trials=1)
    dsr_50 = deflated_sharpe_ratio(rets, n_trials=50)
    assert 0.0 <= dsr_50 < dsr_1 <= 1.0

    # Verify vectorized Rank IC matches exact scipy.stats.spearmanr loop
    sub = sample_panel.iloc[:600].copy().reset_index(drop=True)
    w = cross_sectional_zscore(sub, taker_flow_imbalance(sub, span=8))
    ic_fast = information_coefficient(sub, w)

    next_c = sub.groupby("symbol")["close"].shift(-1)
    sub["fwd"] = (next_c - sub["close"]) / sub["close"]
    sub["w"] = w
    manual_ics = []
    for _, grp in sub.dropna(subset=["w", "fwd"]).groupby("timestamp"):
        if len(grp) >= 5 and grp["w"].std() > 1e-12 and grp["fwd"].std() > 1e-12:
            c, _ = stats.spearmanr(grp["w"], grp["fwd"])
            manual_ics.append(float(c))

    assert np.isclose(ic_fast["mean_ic"], np.mean(manual_ics), atol=1e-5)


def test_adversarial_participant_edge_cases(sample_panel: pd.DataFrame) -> None:
    orig_cols = list(sample_panel.columns)
    orig_close_sum = float(sample_panel["close"].sum())

    # 1. Strategy mutates df in-place, returns numpy array with NaN and +/-inf
    def mutating_numpy_strategy(df: pd.DataFrame) -> np.ndarray:
        df[" leaked_col "] = 999.0
        df["close"] = df["close"] * 0.0
        arr = np.zeros(len(df), dtype=np.float64)
        arr[::3] = 0.25
        arr[::7] = -0.25
        arr[0] = np.nan
        arr[1] = np.inf
        arr[2] = -np.inf
        return arr

    ok_mut, diff_mut, _ = check_causality_truncation(sample_panel, mutating_numpy_strategy)
    assert ok_mut is True
    assert diff_mut == 0.0
    assert list(sample_panel.columns) == orig_cols
    assert np.isclose(float(sample_panel["close"].sum()), orig_close_sum)

    # 2. Non-deterministic strategy (unseeded RNG) must be rejected
    def random_strategy(df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.random.normal(0.0, 1.0, size=len(df)), index=df.index)

    ok_rand, _, msg_rand = check_causality_truncation(sample_panel, random_strategy)
    assert ok_rand is False
    assert "Non-deterministic" in msg_rand

    # 3. Degenerate all-zero strategy must be rejected
    def zero_strategy(df: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=df.index)

    ok_zero, _, msg_zero = check_causality_truncation(sample_panel, zero_strategy)
    assert ok_zero is False
    assert "Degenerate signal" in msg_zero

    # 4. Wrong output length raises ValueError in coerce_weights
    with pytest.raises(ValueError, match="length"):
        coerce_weights(sample_panel, np.zeros(10))


def test_audit_passes_baseline_and_catches_leak(sample_panel: pd.DataFrame) -> None:
    baseline_fn = load_strategy_fn("strategies/baseline_flow.py")
    leaky_fn = load_strategy_fn("strategies/leaky_example.py")
    template_fn = load_strategy_fn("strategies/template.py")

    ok_base, diff_base, _ = check_causality_truncation(sample_panel, baseline_fn)
    assert ok_base is True
    assert diff_base < 1e-11

    ok_tmpl, diff_tmpl, _ = check_causality_truncation(sample_panel, template_fn)
    assert ok_tmpl is True
    assert diff_tmpl < 1e-11

    ok_leak, diff_leak, msg_leak = check_causality_truncation(sample_panel, leaky_fn)
    assert ok_leak is False
    assert diff_leak > 1e-4
    assert "Look-ahead leak detected" in msg_leak


def test_real_binance_dataset_and_baseline_survival() -> None:
    train_path = Path("data/train_1h.parquet")
    holdout_path = Path("data/holdout_1h.parquet")
    if not train_path.exists():
        pytest.skip("Real Binance dataset not present on disk")

    train_df = pd.read_parquet(train_path)
    holdout_df = pd.read_parquet(holdout_path) if holdout_path.exists() else None

    baseline_fn = load_strategy_fn("strategies/baseline_flow.py")
    res = run_full_audit(train_df, baseline_fn, holdout_df=holdout_df)

    assert res["data_quality"].dqs >= 95.0
    assert res["data_quality"].timestamp_gaps == 0
    assert res["causality"]["passed"] is True
    assert res["passed"] is True
    if holdout_df is not None:
        assert res["holdout"] is not None
        assert res["holdout"].sharpe > 0.0
