from __future__ import annotations

import pandas as pd


def compute_weights(df: pd.DataFrame) -> pd.Series:
    """
    Deliberately broken strategy containing two classic backtest bugs:
    1. Global time-series standardization (full-sample mean/std across all dates).
    2. Unlagged future return peek via shift(-1).
    Used by the test suite to verify that `src.audit` catches look-ahead bias automatically.
    """
    ret = df.groupby("symbol")["close"].pct_change()
    # BUG 1: global mean and std across the entire history per symbol
    global_mean = ret.groupby(df["symbol"]).transform("mean")
    global_std = ret.groupby(df["symbol"]).transform("std")
    z_leaky = (ret - global_mean) / global_std

    # BUG 2: peeking 1 bar ahead
    future_peek = df.groupby("symbol")["close"].pct_change().shift(-1).fillna(0.0)
    return 0.5 * z_leaky.fillna(0.0) + 0.5 * future_peek
