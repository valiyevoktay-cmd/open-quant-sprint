"""
Track 10: tests for the absorption ratio function
"""

import numpy as np
import pandas as pd

from src.contrib.vincent import (
    absorption_ratio,
    day_block_bootstrap,
    forward_returns,
    holm,
    label_regimes,
    rank_ic_by_hour,
    regime_cutoffs,
    to_wide,
)

N = 20
WINDOW = 168


def _panel(log_returns):
    hours, n = log_returns.shape
    ts = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(
        {
            "timestamp": np.repeat(ts, n),
            "symbol": np.tile([f"S{i:02d}" for i in range(n)], hours),
            "close": close.ravel(),
        }
    )

def _noise_panel(seed, hours=600):
    return _panel(0.01 * np.random.default_rng(seed).standard_normal((hours, N)))

# Test 1
def test_one_factor_panel_matches_formula():
    rng = np.random.default_rng(0)
    hours, s2 = 3000, 0.25
    loadings = rng.uniform(0.5, 1.5, N)
    factor = rng.standard_normal((hours, 1))
    noise = np.sqrt(s2) * rng.standard_normal((hours, N))
    log_returns = factor * loadings + noise
    ar = absorption_ratio(_panel(0.01 * log_returns))
    L = (loadings**2).sum()
    assert abs(ar.median() - (L + s2) / (L + N * s2)) < 0.02

# Test 2
def test_independent_returns_below_012():
    ar = absorption_ratio(_noise_panel(1, hours=1000))
    assert 0.05 < ar.median() < 0.12

# Test 3
def test_nan_warmup_and_missing_row():
    
    df = _noise_panel(2)
    by_hour = absorption_ratio(df).groupby(df["timestamp"]).first()
    assert by_hour.iloc[:WINDOW].isna().all()
    assert by_hour.iloc[WINDOW:].notna().all()

    m = 300
    ts = df["timestamp"].unique()
    cut = df[~((df["timestamp"] == ts[m]) & (df["symbol"] == "S03"))]
    by_hour = absorption_ratio(cut).groupby(cut["timestamp"]).first()
    assert by_hour.iloc[m : m + WINDOW + 1].isna().all()
    assert by_hour.iloc[WINDOW:m].notna().all()
    assert by_hour.iloc[m + WINDOW + 1 :].notna().all()

# Test 4
def test_same_value_per_timestamp_and_in_unit_interval():
    df = _noise_panel(3)
    ar = absorption_ratio(df)
    assert len(ar) == len(df) and ar.index.equals(df.index)
    assert (ar.groupby(df["timestamp"]).nunique(dropna=False) == 1).all()
    valid = ar.dropna()
    assert ((valid > 0.0) & (valid <= 1.0)).all()
    assert np.allclose(absorption_ratio(df, k=N).dropna(), 1.0)

#Test 5
def test_causal_when_later_rows_dropped():
    df = _noise_panel(4)
    full = absorption_ratio(df)
    head = df.iloc[: 400 * N]
    np.testing.assert_allclose(
        absorption_ratio(head).to_numpy(), full.iloc[: len(head)].to_numpy(), rtol=0, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Part 2
# ---------------------------------------------------------------------------


def _hours(n):
    return pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")


def test_forward_returns_is_next_hour_and_nan_across_a_gap():
    df = _noise_panel(1, hours=50)
    fwd = forward_returns(df)
    close = to_wide(df, df["close"])
    assert np.allclose(fwd.iloc[10], close.iloc[11] / close.iloc[10] - 1.0)
    assert fwd.iloc[-1].isna().all()

    # drop hour 20 for one symbol: its forward return at hour 19 must be NaN, not hour 21's
    gap = df.drop(df[(df["timestamp"] == df["timestamp"].unique()[20]) & (df["symbol"] == "S03")].index)
    fwd_gap = forward_returns(gap)
    assert np.isnan(fwd_gap.iloc[19]["S03"])
    assert np.isnan(fwd_gap.iloc[20]["S03"])
    assert not np.isnan(fwd_gap.iloc[19]["S04"])


def test_rank_ic_perfect_inverse_and_noise():
    rng = np.random.default_rng(2)
    fwd = pd.DataFrame(rng.standard_normal((500, N)), index=_hours(500))
    assert np.allclose(rank_ic_by_hour(fwd, fwd), 1.0)
    assert np.allclose(rank_ic_by_hour(-fwd, fwd), -1.0)
    # monotone transform leaves a rank IC unchanged
    assert np.allclose(rank_ic_by_hour(np.exp(fwd), fwd), 1.0)
    noise = pd.DataFrame(rng.standard_normal((500, N)), index=_hours(500))
    ic = rank_ic_by_hour(noise, fwd)
    assert abs(ic.mean()) < 0.03
    # matches scipy on one hour
    from scipy.stats import spearmanr

    assert np.isclose(ic.iloc[7], spearmanr(noise.iloc[7], fwd.iloc[7])[0])


def test_rank_ic_nan_handling():
    rng = np.random.default_rng(3)
    sig = pd.DataFrame(rng.standard_normal((4, N)), index=_hours(4))
    fwd = pd.DataFrame(rng.standard_normal((4, N)), index=_hours(4))
    sig.iloc[0, :12] = np.nan  # only 8 symbols left -> below min_symbols
    sig.iloc[1, :] = 1.0  # constant signal -> NaN
    fwd.iloc[2, :5] = np.nan  # 15 symbols left -> still computed
    ic = rank_ic_by_hour(sig, fwd, min_symbols=10)
    assert np.isnan(ic.iloc[0]) and np.isnan(ic.iloc[1])
    assert np.isfinite(ic.iloc[2]) and np.isfinite(ic.iloc[3])


def test_cutoffs_use_only_the_first_40_percent():
    rng = np.random.default_rng(4)
    ar = pd.Series(rng.uniform(0.1, 0.5, 1000), index=_hours(1000))
    lo, hi = regime_cutoffs(ar)
    ar2 = ar.copy()
    ar2.iloc[400:] = 0.99
    assert (lo, hi) == regime_cutoffs(ar2)
    assert lo < hi
    lab = label_regimes(ar.iloc[:400], lo, hi)
    assert set(lab.dropna()) == {"low", "mid", "high"}
    assert abs((lab == "low").mean() - 1 / 3) < 0.01


def test_bootstrap_detects_a_real_gap_and_not_a_fake_one():
    rng = np.random.default_rng(5)
    n = 24 * 300
    idx = _hours(n)
    reg = pd.Series(np.where(rng.random(n) < 0.5, "high", "low"), index=idx)
    ic_same = pd.Series(rng.normal(0.02, 0.2, n), index=idx)
    res = day_block_bootstrap(ic_same, reg, n_boot=2000, seed=0)
    assert res["ci_lo"] < 0 < res["ci_hi"]
    assert res["n_blocks"] == 300

    ic_gap = ic_same + np.where(reg == "high", -0.05, 0.0)
    res = day_block_bootstrap(ic_gap, reg, n_boot=2000, seed=0)
    assert res["ci_hi"] < 0 and res["p"] < 0.01
    assert np.isclose(res["diff"], ic_gap[reg == "high"].mean() - ic_gap[reg == "low"].mean())
    # same seed, same answer
    assert res == day_block_bootstrap(ic_gap, reg, n_boot=2000, seed=0)
    # week blocks: fewer, wider
    wk = day_block_bootstrap(ic_gap, reg, n_boot=2000, block_days=7, seed=0)
    assert wk["n_blocks"] == 43


def test_holm_matches_hand_example():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(holm([0.5, 0.9]), [1.0, 1.0])
