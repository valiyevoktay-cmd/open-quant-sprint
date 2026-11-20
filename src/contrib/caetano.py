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


def _rolling_market_calculation(
    df: pd.DataFrame,
    leader: str,
    window: int,
    *,
    statistic: str,
    sample_mean: bool,
) -> pd.Series:
    """Calculate rolling beta, correlation, or residual for the close-return panel.

    ``sample_mean=True`` demeans both returns within each window. With
    ``sample_mean=False``, the calculation assumes a zero mean and uses raw
    second moments. The public functions choose the conventional demeaned mode.
    """
    window = _validate_window(window)
    if statistic not in {"beta", "corr", "residual"}:
        raise ValueError(f"unknown rolling statistic: {statistic!r}")
    required = {"timestamp", "symbol", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"df is missing required columns: {sorted(missing)}")

    # Compute returns in timestamp order within each symbol, then pivot into a
    # time-by-symbol matrix so every rolling operation runs across all symbols.
    work = df[["timestamp", "symbol", "close"]].copy()
    work = work.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)
    close = pd.to_numeric(work["close"], errors="raise").to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.log(close)
    work["_return"] = pd.Series(log_close, index=work.index).groupby(
        work["symbol"], sort=False
    ).diff()
    returns = work.pivot(index="timestamp", columns="symbol", values="_return")
    if leader not in returns.columns:
        raise KeyError(f"leader symbol {leader!r} is not present in df")

    x = returns.to_numpy(dtype=np.float64)
    y = returns[leader].to_numpy(dtype=np.float64)
    n_times, n_symbols = x.shape
    needs_beta = statistic in {"beta", "residual"}
    needs_corr = statistic == "corr"
    result = np.full((n_times, n_symbols), np.nan, dtype=np.float64)
    # Emit eight windows' worth of rows per chunk. The preceding window-sized
    # halo is then a much smaller share of the work than with a 2 * window core.
    chunk_size = max(128, 8 * window)

    for chunk_start in range(window - 1, n_times, chunk_size):
        chunk_end = min(n_times, chunk_start + chunk_size)
        segment_start = max(0, chunk_start - window + 1)
        segment_x = x[segment_start:chunk_end]
        segment_y = y[segment_start:chunk_end]
        valid = np.isfinite(segment_x) & np.isfinite(segment_y[:, None])
        safe_x = np.where(valid, segment_x, 0.0)
        safe_y = np.where(valid, segment_y[:, None], 0.0)

        # Build only the moments used by the requested statistic. Beta and
        # residual need no x variance; correlation does not need beta output.
        terms = {"xy": safe_x * safe_y, "y2": safe_y * safe_y}
        if sample_mean:
            terms["x"] = safe_x
            terms["y"] = safe_y
        if needs_corr:
            terms["x2"] = safe_x * safe_x

        # Chunk-local cumulative sums bound prefix magnitudes while retaining
        # O(N * symbols) work and avoiding a Python loop over timestamps.
        prefixes = {}
        for name, values in terms.items():
            prefix = np.zeros((len(values) + 1, n_symbols), dtype=np.float64)
            np.cumsum(values, axis=0, out=prefix[1:])
            prefixes[name] = prefix
        count_prefix = np.zeros((len(valid) + 1, n_symbols), dtype=np.int64)
        np.cumsum(valid, axis=0, dtype=np.int64, out=count_prefix[1:])

        ends = np.arange(chunk_start, chunk_end)
        local_ends = ends - segment_start + 1
        local_starts = local_ends - window
        counts = count_prefix[local_ends] - count_prefix[local_starts]
        good = counts == window
        sums = {
            name: prefix[local_ends] - prefix[local_starts]
            for name, prefix in prefixes.items()
        }
        cross = sums["xy"]
        var_y = sums["y2"]

        if sample_mean:
            cross = cross - sums["x"] * sums["y"] / window
            var_y = var_y - sums["y"] * sums["y"] / window
        # Roundoff can make a theoretically nonnegative variance slightly < 0.
        var_y = np.maximum(var_y, 0.0)

        if needs_beta:
            beta_chunk = np.full(cross.shape, np.nan, dtype=np.float64)
            np.divide(cross, var_y, out=beta_chunk, where=good & (var_y > 0.0))
            if statistic == "residual":
                residual_chunk = x[ends] - beta_chunk * y[ends, None]
                residual_chunk[~np.isfinite(x[ends]) | ~np.isfinite(y[ends, None])] = np.nan
                result[ends] = residual_chunk
            else:
                result[ends] = beta_chunk
        else:
            var_x = sums["x2"]
            if sample_mean:
                var_x = var_x - sums["x"] * sums["x"] / window
            var_x = np.maximum(var_x, 0.0)
            denom = np.sqrt(var_x * var_y)
            corr_chunk = np.full(cross.shape, np.nan, dtype=np.float64)
            np.divide(cross, denom, out=corr_chunk, where=good & (denom > 0.0))
            result[ends] = corr_chunk

    row_times = returns.index.get_indexer(df["timestamp"])
    row_symbols = returns.columns.get_indexer(df["symbol"])
    row_result = result[row_times, row_symbols]
    return pd.Series(row_result, index=df.index, name=f"rolling_{statistic}")


def rolling_beta(
    df: pd.DataFrame, leader: str = "BTCUSDT", window: int = 72
) -> pd.Series:
    """Causal rolling beta of each symbol's hourly log return against ``leader``."""
    return _rolling_market_calculation(
        df, leader, window, statistic="beta", sample_mean=True
    )


def rolling_corr(
    df: pd.DataFrame, leader: str = "BTCUSDT", window: int = 72
) -> pd.Series:
    """Causal rolling Pearson correlation of returns against ``leader``."""
    return _rolling_market_calculation(
        df, leader, window, statistic="corr", sample_mean=True
    )


def rolling_residual(
    df: pd.DataFrame, leader: str = "BTCUSDT", window: int = 72
) -> pd.Series:
    """Return minus its rolling beta times the leader's return, per symbol."""
    return _rolling_market_calculation(
        df, leader, window, statistic="residual", sample_mean=True
    )
