from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (
    cross_sectional_zscore,
    parkinson_volatility,
    rebalance_deadband,
    taker_flow_imbalance,
)


def compute_weights(df: pd.DataFrame) -> pd.Series:
    """
    Baseline strategy: Volatility-scaled Taker Order Flow Imbalance,
    intermediate breakout continuation, short-horizon liquidity reversal,
    and 48h cumulative funding rate crowding penalty.
    Uses causal EWM smoothing + deadband filtering to survive 5 bps taker fees.
    """
    p_vol = parkinson_volatility(df, window=48).replace(0.0, np.nan)
    flow_imb = taker_flow_imbalance(df, span=24)

    ret_16h = df.groupby("symbol")["close"].pct_change(16)
    ret_2h = df.groupby("symbol")["close"].pct_change(2)

    cum_funding_48h = df.groupby("symbol")["funding_rate"].transform(
        lambda x: x.rolling(window=48, min_periods=1).sum()
    )

    z_flow = cross_sectional_zscore(df, (flow_imb / p_vol).fillna(0.0))
    z_breakout = cross_sectional_zscore(df, (ret_16h / p_vol).fillna(0.0))
    z_rev = cross_sectional_zscore(df, -ret_2h.fillna(0.0))
    z_fund = cross_sectional_zscore(df, -cum_funding_48h)

    combined = 0.58 * z_flow + 0.12 * z_breakout + 0.16 * z_fund + 0.14 * z_rev
    smoothed = combined.groupby(df["symbol"]).transform(
        lambda x: x.ewm(span=36, min_periods=1).mean()
    )
    target = cross_sectional_zscore(df, smoothed)
    return rebalance_deadband(df, target, buffer=0.012)
