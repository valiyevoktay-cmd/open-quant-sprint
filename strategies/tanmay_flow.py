"""
Track 05 (#5): taker flow carried by unusually large trades.

Signal: trade_intensity_imbalance (src/contrib/tanmay.py), the signed taker-flow imbalance times the
hour's average trade size relative to its 24-hour rolling median, smoothed with an EWM of SPAN hours.
SIGN = +1 trades continuation: long the symbols with size-weighted net taker buying.
Weights: cross-sectional z-score (dollar-neutral, sum |w| = 1), then rebalance_deadband.

SPAN, SIGN and DEADBAND were posted in the Part 2 PR before the first full-panel run.
"""

from __future__ import annotations

import pandas as pd

from src.contrib.tanmay import trade_intensity_imbalance
from src.features import cross_sectional_zscore, rebalance_deadband

SPAN = 24
WINDOW = 24
SIGN = 1.0
DEADBAND = 0.015


def build_weights(df: pd.DataFrame, span: int, deadband: float, sign: float = SIGN) -> pd.Series:
    signal = sign * trade_intensity_imbalance(df, span=span, window=WINDOW)
    target = cross_sectional_zscore(df, signal.fillna(0.0))
    return rebalance_deadband(df, target, buffer=deadband)


def compute_weights(df: pd.DataFrame) -> pd.Series:
    return build_weights(df, SPAN, DEADBAND)