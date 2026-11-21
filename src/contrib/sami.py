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


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-adjusted p-values in the input order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p_values must be a one-dimensional array in [0, 1]")
    if values.size == 0:
        return values.copy()

    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        (values.size - np.arange(values.size)) * values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def moving_block_bootstrap_means(
    values: np.ndarray,
    block_days: int,
    n_boot: int = 10_000,
    seed: int = 0,
) -> np.ndarray:
    """Resample a full hourly series in contiguous, non-wrapping day blocks."""
    if isinstance(block_days, (bool, np.bool_)) or not isinstance(block_days, (int, np.integer)) or block_days < 1:
        raise ValueError("block_days must be a positive integer")
    if isinstance(n_boot, (bool, np.bool_)) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")

    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError("values must be one-dimensional")
    block = 24 * int(block_days)
    n = series.size
    if n < block:
        raise ValueError("values must contain at least one full block")

    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    bootstrap_means = np.full(int(n_boot), np.nan, dtype=np.float64)

    for sample_number in range(int(n_boot)):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate(
            [series[start : start + block] for start in starts]
        )[:n]
        sample = sample[np.isfinite(sample)]
        if sample.size:
            bootstrap_means[sample_number] = sample.mean()

    return bootstrap_means


def bootstrap_pvalue(bootstrap_means: np.ndarray) -> float:
    """Calculate the two-sided bootstrap p-value for a zero mean."""
    values = np.asarray(bootstrap_means, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("bootstrap_means must contain a finite value")
    p_value = 2.0 * min(np.mean(values <= 0.0), np.mean(values >= 0.0))
    return float(min(p_value, 1.0))
