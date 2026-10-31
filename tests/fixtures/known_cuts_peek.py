"""Canary for audit v1.0: look-ahead everywhere except at the predictable (seed 0) prefix cuts. Must be REJECTED."""
from __future__ import annotations

import pandas as pd

from src.features import cross_sectional_zscore, parkinson_volatility

# Prefix cut timestamps of audit v1.0 (fixed seed 0, 3 random cuts), computable in advance.
QUIET = pd.to_datetime([
    "2025-07-23 12:00", "2025-09-05 20:00", "2025-10-30 01:00", "2026-02-07 18:00",
    "2026-04-06 04:00", "2026-06-11 16:00", "2026-08-12 15:00", "2026-09-06 10:00",
    "2026-09-12 13:00",
], utc=True)


def compute_weights(df: pd.DataFrame) -> pd.Series:
    p_vol = parkinson_volatility(df, window=24).replace(0.0, float("nan"))
    raw = df.groupby("symbol")["close"].pct_change(24) / p_vol
    z = cross_sectional_zscore(df, raw.fillna(0.0))
    smoothed = z.groupby(df["symbol"]).transform(lambda x: x.ewm(span=18, min_periods=1).mean())
    w = cross_sectional_zscore(df, smoothed)

    # Two-bar look-ahead everywhere except the two hours ending at each cut timestamp,
    # where a prefix run would see a different (truncated) future.
    fwd = df.groupby("symbol")["close"].shift(periods=-2) / df["close"] - 1.0
    near_cut = df["timestamp"].isin(QUIET) | df["timestamp"].isin(QUIET - pd.Timedelta(hours=1))
    leak = cross_sectional_zscore(df, fwd.fillna(0.0)).where(~near_cut, 0.0)
    return w + 0.12 * leak
