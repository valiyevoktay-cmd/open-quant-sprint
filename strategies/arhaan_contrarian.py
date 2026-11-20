from __future__ import annotations

import numpy as np
import pandas as pd
from src.features import cross_sectional_zscore, rebalance_deadband

# Track 11 (#17): Lo-MacKinlay (1990) contrarian portfolio.
Q = 72            # horizon in hours (>= 72 to stay clear of Track 06)
DEADBAND = 0.03  # rebalance buffer; audited trials: 0, 0.015, 0.03 (all listed in the PR)


def compute_weights(df: pd.DataFrame) -> pd.Series:
    """
    Contrarian weights: sell the symbols that rose most over the last Q hours relative to
    the others, buy the ones that fell most. The target is dollar-neutral at every timestamp;
    after rebalance_deadband the held weights can drift slightly.

    Causal: the signal at t uses log closes at t and t - Q hours of the same symbol only.
    The lookback is matched on timestamp, not row count, so a symbol with dropped hours
    still compares against the close exactly Q hours earlier. If that close (or the one
    at t) is missing, the signal is NaN, which cross_sectional_zscore turns into zero
    weight, so missing data never crashes it.
    """
    ts = pd.to_datetime(df["timestamp"])
    log_close = np.log(df["close"].astype(float)).to_numpy()
    by_key = pd.Series(log_close, index=pd.MultiIndex.from_arrays([ts, df["symbol"]]))
    lookback = pd.MultiIndex.from_arrays([ts - pd.Timedelta(hours=Q), df["symbol"]])
    past_close = by_key.reindex(lookback).to_numpy()                 # log close Q hours earlier, or NaN

    past_q = pd.Series(log_close - past_close, index=df.index)       # Q-hour log return per symbol
    signal = -past_q                                                 # contrarian: against the past move

    target = cross_sectional_zscore(df, signal)                      # demean, scale, sum |w| = 1 per hour
    return rebalance_deadband(df, target, buffer=DEADBAND)
