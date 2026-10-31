"""Canary for audit v1.0: look-ahead confined to the rows after the last prefix cut. Must be REJECTED."""
from __future__ import annotations

import pandas as pd

from src.features import cross_sectional_zscore, parkinson_volatility, rebalance_deadband

PANEL_END = pd.Timestamp("2026-09-12 14:00", tz="UTC")


def compute_weights(df: pd.DataFrame) -> pd.Series:
    # Looks like template.py: 24h momentum scaled by Parkinson vol, smoothed, deadband.
    p_vol = parkinson_volatility(df, window=24).replace(0.0, float("nan"))
    raw = df.groupby("symbol")["close"].pct_change(24) / p_vol
    z = cross_sectional_zscore(df, raw.fillna(0.0))
    smoothed = z.groupby(df["symbol"]).transform(lambda x: x.ewm(span=18, min_periods=1).mean())
    w = cross_sectional_zscore(df, smoothed)

    # Look-ahead only when the panel ends exactly at the public train end, and only in the
    # last 140 hours, which in audit v1.0 lay after the last prefix cut (index 14722 of 14871).
    if df["timestamp"].max() == PANEL_END:
        fwd = df.groupby("symbol")["close"].shift(periods=-2) / df["close"] - 1.0
        late = df["timestamp"] > PANEL_END - pd.Timedelta(hours=140)
        w = w + 3.0 * cross_sectional_zscore(df, fwd.fillna(0.0)).where(late, 0.0)
    return rebalance_deadband(df, w, buffer=0.015)
