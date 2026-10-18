from __future__ import annotations

import pandas as pd
from src.features import cross_sectional_zscore


def compute_weights(df: pd.DataFrame) -> pd.Series:
    """
    Starter template for sprint submissions.
    Input `df` is sorted by (timestamp, symbol) and follows src.schema.REQUIRED_COLS.

    Rules:
    1. Weight at index i must depend ONLY on data at timestamps <= df.loc[i, 'timestamp'].
    2. Never call global .mean(), .std(), or .shift(-k) across the full time axis.
    3. Return a Series aligned with df.index (dollar-neutral cross-sectional weights recommended).
    """
    # Example: 24h causal log return reversal
    ret_24h = df.groupby("symbol")["close"].pct_change(24)
    raw_signal = -ret_24h
    return cross_sectional_zscore(df, raw_signal)
