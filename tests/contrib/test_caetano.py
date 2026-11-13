from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.contrib.caetano import rolling_zscore, rolling_zscore_slow


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


def test_rolling_zscore_matches_slow_reference_on_real_log_returns(train: pd.DataFrame) -> None:
    df = _with_log_returns(train)

    fast = rolling_zscore(df, "log_return", window=48).to_numpy()
    slow = rolling_zscore_slow(df, "log_return", window=48).to_numpy()

    np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))
    np.testing.assert_allclose(fast, slow, rtol=0.0, atol=1e-9, equal_nan=True)


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
