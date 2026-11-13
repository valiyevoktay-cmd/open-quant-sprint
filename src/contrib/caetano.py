"""Rolling statistics contributed by Caetano Paulo Garelli."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _validate_window(window: int) -> int:
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError("window must be an integer greater than or equal to 2")
    return int(window)


def _symbol_groups(df: pd.DataFrame):
    if "symbol" not in df.columns:
        raise KeyError("df must contain a 'symbol' column")
    # .indices gives integer positions, so the output remains aligned even if df.index
    # has labels that are duplicated or out of order.
    return df.groupby("symbol", sort=False).indices.values()

#accurate, slow z-score calculators. Used as a reference for
#calculating the precision of faster methods.
def _two_pass_window_zscore(values: np.ndarray, window: int) -> np.ndarray:
    """Accurate reference calculation, using a separate mean and variance pass per window."""
    result = np.full(values.size, np.nan, dtype=np.float64)
    if values.size < window:
        return result

    windows = sliding_window_view(values, window_shape=window)
    finite = np.isfinite(windows).all(axis=1)
    if not finite.any():
        return result

    valid_windows = windows[finite]
    means = valid_windows.mean(axis=1)
    centered = valid_windows - means[:, None]
    stds = np.sqrt(np.sum(centered * centered, axis=1) / (window - 1))
    current = valid_windows[:, -1]
    z = np.zeros_like(stds)
    np.divide(current - means, stds, out=z, where=stds > 0.0)
    result[np.flatnonzero(finite) + window - 1] = z
    return result


def rolling_zscore_slow(df: pd.DataFrame, col: str, window: int = 48) -> pd.Series:
    """Accurate O(NW) two-pass reference calculation, separately per symbol.

    The current row is included in its window. The sample standard deviation uses
    ``ddof=1``. The first ``window - 1`` rows per symbol, and windows containing
    non-finite values, return NaN. A constant window returns zero.

    This is the clear, slow baseline used to check the faster implementations.
    """
    window = _validate_window(window)
    if col not in df.columns:
        raise KeyError(f"df has no column {col!r}")

    values = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=np.float64, copy=True)
    result = np.full(len(df), np.nan, dtype=np.float64)
    for positions in _symbol_groups(df):
        result[positions] = _two_pass_window_zscore(values[positions], window)
    return pd.Series(result, index=df.index, name=f"{col}_zscore")



def _chunked_prefix_zscore(values: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling z-scores from short, locally centered prefix sums."""
    n = values.size
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return result

    # Each chunk emits several windows and carries only a window-sized halo.
    # Resetting the prefix sums at each chunk bounds their magnitude and keeps
    # the anchor close to the prices being compared.
    chunk_size = max(128, 2 * window)
    for chunk_start in range(window - 1, n, chunk_size):
        chunk_end = min(n, chunk_start + chunk_size)
        segment_start = max(0, chunk_start - window + 1)
        segment = values[segment_start:chunk_end]

        # Anchor at the first finite value in the output chunk. This avoids
        # accumulating sums of raw price levels such as BTC's ~80,000 close.
        core = values[chunk_start:chunk_end]
        finite_core = np.flatnonzero(np.isfinite(core))
        if finite_core.size == 0:
            continue
        anchor = core[finite_core[0]]

        valid = np.isfinite(segment)
        centered = np.where(valid, segment - anchor, 0.0)
        prefix_sum = np.empty(segment.size + 1, dtype=np.float64)
        prefix_sq = np.empty(segment.size + 1, dtype=np.float64)
        prefix_count = np.empty(segment.size + 1, dtype=np.int64)
        prefix_sum[0] = prefix_sq[0] = 0.0
        prefix_count[0] = 0
        np.cumsum(centered, out=prefix_sum[1:])
        np.cumsum(centered * centered, out=prefix_sq[1:])
        np.cumsum(valid, dtype=np.int64, out=prefix_count[1:])

        ends = np.arange(chunk_start, chunk_end)
        local_ends = ends - segment_start + 1
        local_starts = local_ends - window
        counts = prefix_count[local_ends] - prefix_count[local_starts]
        good = counts == window

        sums = prefix_sum[local_ends] - prefix_sum[local_starts]
        sums_sq = prefix_sq[local_ends] - prefix_sq[local_starts]
        means = sums / window
        variances = (sums_sq - sums * means) / (window - 1)
        stds = np.sqrt(np.maximum(variances, 0.0))

        scores = np.zeros(ends.size, dtype=np.float64)
        current_centered = values[ends] - anchor
        np.divide(current_centered - means, stds, out=scores, where=stds > 0.0)
        result[ends[good]] = scores[good]

    return result


def rolling_zscore(df: pd.DataFrame, col: str, window: int = 48) -> pd.Series:
    """Causal O(N) rolling z-score per symbol, without pandas rolling or row loops.

    NumPy prefix sums are reset in short chunks after subtracting a nearby price
    anchor. The current row is included, the sample standard deviation uses
    ``ddof=1``, and incomplete or non-finite windows return NaN. Constant
    windows return zero.
    """
    window = _validate_window(window)
    if col not in df.columns:
        raise KeyError(f"df has no column {col!r}")

    values = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    result = np.full(len(df), np.nan, dtype=np.float64)
    for positions in _symbol_groups(df):
        result[positions] = _chunked_prefix_zscore(values[positions], window)
    return pd.Series(result, index=df.index, name=f"{col}_zscore")
