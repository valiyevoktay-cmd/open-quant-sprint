from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from src.contrib.caetano import (
    _rolling_market_calculation,
    rolling_beta,
    rolling_corr,
    rolling_residual,
    rolling_zscore,
    rolling_zscore_slow,
)


TRAIN = Path("data/train_1h.parquet")


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    if not TRAIN.exists():
        pytest.skip("training panel not available")
    return pd.read_parquet(TRAIN)


def _with_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["close"]).groupby(out["symbol"], sort=False).diff()
    return out


def _sliding_window_reference(
    df: pd.DataFrame, col: str, window: int
) -> np.ndarray:
    """Build per-symbol z-scores with explicit window means and sample stds."""
    values = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=np.float64)
    result = np.full(len(df), np.nan, dtype=np.float64)

    for positions in df.groupby("symbol", sort=False).indices.values():
        symbol_values = values[positions]
        if symbol_values.size < window:
            continue

        windows = sliding_window_view(symbol_values, window_shape=window)
        finite = np.isfinite(windows).all(axis=1)
        valid_windows = windows[finite]
        if valid_windows.size == 0:
            continue

        means = valid_windows.mean(axis=1)
        stds = valid_windows.std(axis=1, ddof=1)
        scores = np.zeros(stds.shape, dtype=np.float64)
        np.divide(
            valid_windows[:, -1] - means,
            stds,
            out=scores,
            where=stds > 0.0,
        )
        result[positions[np.flatnonzero(finite) + window - 1]] = scores

    return result


def _market_reference(
    df: pd.DataFrame, leader: str, window: int, sample_mean: bool
) -> dict[str, np.ndarray]:
    """Slow per-symbol reference for rolling beta, corr, and residual."""
    work = df[["timestamp", "symbol", "close"]].copy()
    work = work.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.log(pd.to_numeric(work["close"], errors="raise"))
    work["_return"] = log_close.groupby(work["symbol"], sort=False).diff()
    returns = work.pivot(index="timestamp", columns="symbol", values="_return")
    y = returns[leader].to_numpy(dtype=np.float64)
    beta = np.full(returns.shape, np.nan, dtype=np.float64)
    corr = np.full(returns.shape, np.nan, dtype=np.float64)
    residual = np.full(returns.shape, np.nan, dtype=np.float64)

    for column, symbol in enumerate(returns.columns):
        x = returns[symbol].to_numpy(dtype=np.float64)
        if len(x) < window:
            continue
        x_windows = sliding_window_view(x, window)
        y_windows = sliding_window_view(y, window)
        finite = np.isfinite(x_windows).all(axis=1) & np.isfinite(y_windows).all(axis=1)
        if not finite.any():
            continue
        x_valid = x_windows[finite]
        y_valid = y_windows[finite]
        if sample_mean:
            x_values = x_valid - x_valid.mean(axis=1, keepdims=True)
            y_values = y_valid - y_valid.mean(axis=1, keepdims=True)
        else:
            x_values = x_valid
            y_values = y_valid
        cross = np.sum(x_values * y_values, axis=1)
        x_ss = np.sum(x_values * x_values, axis=1)
        y_ss = np.sum(y_values * y_values, axis=1)
        beta_values = np.full(cross.shape, np.nan)
        corr_values = np.full(cross.shape, np.nan)
        np.divide(cross, y_ss, out=beta_values, where=y_ss > 0.0)
        np.divide(cross, np.sqrt(x_ss * y_ss), out=corr_values, where=(x_ss * y_ss) > 0.0)
        ends = np.flatnonzero(finite) + window - 1
        beta[ends, column] = beta_values
        corr[ends, column] = corr_values
        residual[ends, column] = x_valid[:, -1] - beta_values * y_valid[:, -1]

    row_times = returns.index.get_indexer(df["timestamp"])
    row_symbols = returns.columns.get_indexer(df["symbol"])
    return {
        "beta": beta[row_times, row_symbols],
        "corr": corr[row_times, row_symbols],
        "residual": residual[row_times, row_symbols],
    }


def _small_market_panel() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=12, freq="h", tz="UTC")
    btc_returns = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.04, -0.03, 0.02, 0.01, -0.04, 0.03])
    alt_returns = np.array([0.02, -0.01, 0.02, 0.03, -0.02, 0.01, -0.01, 0.04, 0.02, -0.01, 0.05])
    rows = []
    for symbol, returns in (("BTCUSDT", btc_returns), ("ALTUSDT", alt_returns)):
        prices = 100.0 * np.exp(np.r_[0.0, np.cumsum(returns)])
        rows.extend(
            {"timestamp": ts, "symbol": symbol, "close": price}
            for ts, price in zip(timestamps, prices)
        )
    return pd.DataFrame(rows).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def test_rolling_zscore_matches_slow_reference_on_real_log_returns(train: pd.DataFrame) -> None:
    df = _with_log_returns(train)

    fast = rolling_zscore(df, "log_return", window=48).to_numpy()
    slow = rolling_zscore_slow(df, "log_return", window=48).to_numpy()

    np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))
    np.testing.assert_allclose(fast, slow, rtol=0.0, atol=1e-9, equal_nan=True)


def test_rolling_zscore_matches_sliding_window_ddof1_on_real_log_returns(
    train: pd.DataFrame,
) -> None:
    df = _with_log_returns(train)

    actual = rolling_zscore(df, "log_return", window=48).to_numpy()
    reference = _sliding_window_reference(df, "log_return", window=48)

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(reference))
    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1e-9, equal_nan=True)


def test_rolling_zscore_raw_btc_close_against_slow_reference(train: pd.DataFrame) -> None:
    btc = train.loc[train["symbol"].eq("BTCUSDT")].copy()

    fast = rolling_zscore(btc, "close", window=48).to_numpy()
    slow = rolling_zscore_slow(btc, "close", window=48).to_numpy()

    np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))
    np.testing.assert_allclose(fast, slow, rtol=0.0, atol=1e-8, equal_nan=True)


def test_rolling_zscore_is_unchanged_when_later_real_rows_are_dropped(train: pd.DataFrame) -> None:
    df = _with_log_returns(train)
    timestamps = df["timestamp"].drop_duplicates().sort_values()
    cut = timestamps.iloc[int(len(timestamps) * 0.8)]
    prefix = df.loc[df["timestamp"] <= cut]

    full = rolling_zscore(df, "log_return", window=48).loc[prefix.index].to_numpy()
    truncated = rolling_zscore(prefix, "log_return", window=48).to_numpy()

    np.testing.assert_array_equal(np.isnan(full), np.isnan(truncated))
    np.testing.assert_allclose(full, truncated, rtol=0.0, atol=1e-12, equal_nan=True)


def test_rolling_zscore_startup_and_constant_windows() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC"),
            "symbol": ["BTCUSDT"] * 8,
            "close": [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0],
        }
    )

    result = rolling_zscore(df, "close", window=3)

    assert result.iloc[:2].isna().all()
    assert result.iloc[2:5].notna().all()
    np.testing.assert_array_equal(result.iloc[6:].to_numpy(), np.zeros(2))


def test_rolling_market_statistics_match_two_pass_reference_on_real_returns(
    train: pd.DataFrame,
) -> None:
    window = 72
    expected = _market_reference(train, "BTCUSDT", window, sample_mean=True)

    for name, function in (
        ("beta", rolling_beta),
        ("corr", rolling_corr),
        ("residual", rolling_residual),
    ):
        actual = function(train, window=window).to_numpy()
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected[name]))
        np.testing.assert_allclose(actual, expected[name], rtol=0.0, atol=1e-9, equal_nan=True)


def test_rolling_market_statistics_support_zero_mean_mode() -> None:
    df = _small_market_panel()
    window = 5
    expected = _market_reference(df, "BTCUSDT", window, sample_mean=False)

    for statistic in ("beta", "corr", "residual"):
        actual = _rolling_market_calculation(
            df,
            "BTCUSDT",
            window,
            statistic=statistic,
            sample_mean=False,
        ).to_numpy()
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected[statistic]))
        np.testing.assert_allclose(
            actual, expected[statistic], rtol=0.0, atol=1e-12, equal_nan=True
        )


def test_rolling_market_statistics_are_causal_and_handle_edges() -> None:
    df = _small_market_panel()
    window = 5
    cut = df["timestamp"].drop_duplicates().iloc[8]
    prefix = df.loc[df["timestamp"] <= cut]

    for function in (rolling_beta, rolling_corr, rolling_residual):
        full = function(df, window=window).loc[prefix.index].to_numpy()
        truncated = function(prefix, window=window).to_numpy()
        np.testing.assert_array_equal(np.isnan(full), np.isnan(truncated))
        np.testing.assert_allclose(full, truncated, rtol=0.0, atol=1e-12, equal_nan=True)

    # Each symbol lacks its first hourly return, so five complete paired returns
    # become available at that symbol's sixth row.
    beta = rolling_beta(df, window=window)
    corr = rolling_corr(df, window=window)
    for positions in df.groupby("symbol", sort=False).indices.values():
        assert beta.iloc[positions[:5]].isna().all()
        assert corr.iloc[positions[:5]].isna().all()

    # A constant return series has zero variance, so its Pearson correlation is undefined.
    flat = df.copy()
    alt_positions = flat["symbol"].eq("ALTUSDT")
    flat.loc[alt_positions, "close"] = 50.0
    flat_corr = rolling_corr(flat, window=window)
    assert flat_corr.loc[alt_positions].iloc[5:].isna().all()
