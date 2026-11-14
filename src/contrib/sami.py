"""Abnormal-volume feature for Track 13."""

from __future__ import annotations

import numpy as np
import pandas as pd


def abnormal_volume(df: pd.DataFrame, days: int = 14) -> pd.Series:
    """Measure log volume against the same hour on the previous ``days`` days.

    The historical mean and sample standard deviation use only the previous
    same-hour observations. Missing hours and zero volume count as missing data.
    The returned values keep the input row order and index.
    """
    if isinstance(days, (bool, np.bool_)) or not isinstance(days, (int, np.integer)) or days < 2:
        raise ValueError("days must be an integer greater than or equal to 2")
    if not {"timestamp", "symbol", "quote_volume"}.issubset(df.columns):
        raise KeyError("df must contain 'timestamp', 'symbol', and 'quote_volume' columns")

    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    quote_volume = pd.to_numeric(df["quote_volume"], errors="raise").to_numpy(dtype=np.float64)
    log_volume = np.full(len(df), np.nan, dtype=np.float64)
    positive_finite = np.isfinite(quote_volume) & (quote_volume > 0.0)
    log_volume[positive_finite] = np.log(quote_volume[positive_finite])

    result = np.full(len(df), np.nan, dtype=np.float64)
    for positions in df.groupby("symbol", sort=False).indices.values():
        symbol_timestamps = pd.DatetimeIndex(timestamps.iloc[positions])
        observed = pd.Series(log_volume[positions], index=symbol_timestamps).sort_index()
        hourly_index = pd.date_range(
            start=observed.index.min(),
            end=observed.index.max(),
            freq="h",
            tz="UTC",
        )
        hourly_volume = observed.reindex(hourly_index)

        history = pd.concat(
            [hourly_volume.shift(24 * lag) for lag in range(1, int(days) + 1)],
            axis=1,
        )
        historical_mean = history.mean(axis=1)
        historical_std = history.std(axis=1, ddof=1)
        has_variation = history.max(axis=1) > history.min(axis=1)
        valid = (
            hourly_volume.notna()
            & history.notna().all(axis=1)
            & historical_std.notna()
            & (historical_std > 0.0)
            & has_variation
        )

        scores = pd.Series(np.nan, index=hourly_index, dtype=np.float64)
        scores.loc[valid] = (
            (hourly_volume.loc[valid] - historical_mean.loc[valid])
            / historical_std.loc[valid]
        )
        result[positions] = scores.reindex(symbol_timestamps).to_numpy(dtype=np.float64)

    return pd.Series(result, index=df.index, name="abnormal_volume")
