"""Track 12 (#19): does the clock predict crypto returns, or only activity?

Part 1: ``seasonal_wald``, a cluster-robust test that a series has the same mean in
every group. Part 2: ``holm``, ``block_bootstrap_mean`` and the study itself
(``run_clock_study``). Run the study with ``python -m src.contrib.hongyu``.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist


def seasonal_wald(
    x: np.ndarray, group: np.ndarray, cluster: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Test whether ``x`` has the same mean in every group.

    Standard errors are cluster-robust: observations in the same cluster may be
    correlated in any way, observations in different clusters are independent.
    Observations where ``x`` is NaN are dropped first.

    With group means ``m_g`` and counts ``n_g``, each observation contributes
    ``u_i = (x_i - m_g) / n_g``. Summing ``u`` by cluster and group gives the
    ``C x G`` matrix ``S`` and the covariance of the means
    ``V = C / (C - 1) * S.T @ S``. With ``R = [I | -1]`` and ``d = R @ m``,
    ``W = d @ inv(R @ V @ R.T) @ d``. The p-value uses the small-sample F form
    ``F = W * (C - q) / (q * (C - 1))`` with ``q = G - 1`` and ``C - q``
    denominator degrees of freedom.

    Returns the group means (ordered as ``np.unique(group)``), ``W`` and the
    p-value.
    """
    x = np.asarray(x, dtype=np.float64)
    group = np.asarray(group)
    cluster = np.asarray(cluster)
    if x.ndim != 1 or group.ndim != 1 or cluster.ndim != 1:
        raise ValueError("x, group and cluster must be one-dimensional")
    if not (len(x) == len(group) == len(cluster)):
        raise ValueError("x, group and cluster must have the same length")

    keep = ~np.isnan(x)
    x, group, cluster = x[keep], group[keep], cluster[keep]
    if not np.isfinite(x).all():
        raise ValueError("x must not contain infinite values")

    groups, g_idx = np.unique(group, return_inverse=True)
    _, c_idx = np.unique(cluster, return_inverse=True)
    n_groups = len(groups)
    n_clusters = int(c_idx.max()) + 1 if len(c_idx) else 0
    q = n_groups - 1
    if n_groups < 2:
        raise ValueError("at least two groups with data are needed")
    if n_clusters <= q:
        raise ValueError("the number of clusters must exceed the number of groups minus one")

    counts = np.bincount(g_idx, minlength=n_groups).astype(np.float64)
    means = np.bincount(g_idx, weights=x, minlength=n_groups) / counts

    u = (x - means[g_idx]) / counts[g_idx]
    scores = np.zeros((n_clusters, n_groups), dtype=np.float64)
    np.add.at(scores, (c_idx, g_idx), u)
    cov = n_clusters / (n_clusters - 1) * (scores.T @ scores)

    restriction = np.hstack([np.eye(q), -np.ones((q, 1))])
    d = restriction @ means
    middle = restriction @ cov @ restriction.T
    wald = float(d @ np.linalg.solve(middle, d))

    f_stat = wald * (n_clusters - q) / (q * (n_clusters - 1))
    p_value = float(f_dist.sf(f_stat, q, n_clusters - q))
    return means, wald, p_value


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, in the input order.

    Sort ascending, multiply the i-th smallest (0-based) by ``m - i``, take the
    running maximum so adjusted values never decrease, and cap at one.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p_values must lie in [0, 1]")
    m = len(p)
    order = np.argsort(p, kind="stable")
    scaled = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = np.minimum(scaled, 1.0)
    return adjusted


def block_bootstrap_mean(
    x: np.ndarray,
    block: int,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float, float, float]:
    """Moving-block bootstrap for the mean of a serially dependent series.

    NaNs are dropped first. Each resample joins ``ceil(n / block)`` blocks of
    ``block`` consecutive values with uniformly drawn starts (as in
    ``src.stats.sharpe_block_bootstrap_ci``) and is cut to length ``n``.
    The primary two-sided p-value follows Issue #19 exactly:
    ``2 * min(share(m_b <= 0), share(m_b >= 0))``.  For transparency the
    function also returns the centred-null sensitivity calculation used in the
    first draft, ``(1 + #{|m_b - m| >= |m|}) / (1 + n_boot)``.

    Returns the mean, percentile interval, specified tail-share p-value and
    centred-null sensitivity p-value.
    """
    values = np.asarray(x, dtype=np.float64)
    values = values[~np.isnan(values)]
    n = len(values)
    if block < 1 or n < 2 * block:
        raise ValueError("the series needs at least two blocks of data")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    # Prefix sums make each resample mean O(n_blocks) instead of O(n).
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    sums = prefix[starts + block] - prefix[starts]
    tail = n - (n_blocks - 1) * block
    last = starts[:, -1]
    sums[:, -1] = prefix[last + tail] - prefix[last]
    boot = sums.sum(axis=1) / n

    mean = float(values.mean())
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    p_tail = min(1.0, 2.0 * min(np.mean(boot <= 0.0), np.mean(boot >= 0.0)))
    exceed = np.count_nonzero(np.abs(boot - mean) >= abs(mean))
    p_centered = (1.0 + exceed) / (1.0 + n_boot)
    return mean, float(lo), float(hi), float(p_tail), float(p_centered)


# --------------------------------------------------------------------------
# Part 2: the study
# --------------------------------------------------------------------------

FUNDING_HOURS = (0, 8, 16)
PRE_FUNDING_HOURS = (23, 7, 15)
CONTINUATION_DAYS = (5, 10, 20, 40)
CONTROL_OFFSET_HOURS = 8
BOOTSTRAP_BLOCK_HOURS = 5 * 24
MIN_IC_SYMBOLS = 5


def _wide(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Timestamp x symbol frame on the full hourly grid (missing hours are NaN)."""
    data = df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True))
    wide = data.pivot(index="timestamp", columns="symbol", values=column).sort_index()
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    return wide.reindex(grid).astype(np.float64)


def _wald_row(name: str, x: pd.Series, group: np.ndarray, cluster: np.ndarray) -> dict:
    means, wald, p_value = seasonal_wald(x.to_numpy(), group, cluster)
    return {"test": name, "wald": wald, "p": p_value, "groups": len(means)}


def _iso_week(index: pd.DatetimeIndex) -> np.ndarray:
    iso = index.isocalendar()
    return (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()


def same_hour_signal(returns: pd.DataFrame, k: int, offset: int = 0) -> pd.DataFrame:
    """Signal at hour t: mean of returns at t + 1 - 24 j - offset, j = 1..k.

    ``offset = 0`` uses the same hour of day as the return being predicted
    (``t + 1``); the control uses ``offset = 8``. All k values must exist.
    """
    lagged = [returns.shift(24 * j + offset - 1) for j in range(1, k + 1)]
    total = sum(lagged)
    return total / k


def hourly_spearman_ic(signal: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman correlation per row, written out by hand.

    Only symbols with both values count; rows with fewer than
    ``MIN_IC_SYMBOLS`` such symbols, or no variation, are NaN.
    """
    both = signal.notna() & target.notna()
    a = signal.where(both).rank(axis=1, method="average")
    b = target.where(both).rank(axis=1, method="average")
    a = a.sub(a.mean(axis=1), axis=0)
    b = b.sub(b.mean(axis=1), axis=0)
    cov = (a * b).sum(axis=1)
    scale = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    ic = cov / scale.where(scale > 0)
    return ic.where(both.sum(axis=1) >= MIN_IC_SYMBOLS)


def _continuation(returns: pd.DataFrame, rows: pd.Index, seed: int) -> list[dict]:
    target = returns.shift(-1)
    out = []
    for k in CONTINUATION_DAYS:
        for kind, offset in (("same hour", 0), ("control -8h", CONTROL_OFFSET_HOURS)):
            ic = hourly_spearman_ic(same_hour_signal(returns, k, offset), target)
            ic = ic.reindex(rows)
            mean, lo, hi, p, p_centered = block_bootstrap_mean(
                ic.to_numpy(), BOOTSTRAP_BLOCK_HOURS, seed=seed + k + offset
            )
            out.append({"k": k, "signal": kind, "mean_ic": mean, "lo": lo, "hi": hi,
                        "p": p, "p_centered_sensitivity": p_centered,
                        "n_hours": int(ic.notna().sum())})
    same = [row for row in out if row["signal"] == "same hour"]
    for row, adj in zip(same, holm(np.array([row["p"] for row in same]))):
        row["p_holm"] = float(adj)
    return out


def run_clock_study(df: pd.DataFrame, seed: int = 12) -> dict:
    """Run Part 2 steps 1-7 of Issue #19 on an hourly panel."""
    close = _wide(df, "close")
    returns = np.log(close.where(close > 0)).diff()  # full grid, before any hour filter
    index_return = returns.mean(axis=1)  # mean of available symbols
    hours = returns.index.hour.to_numpy()
    day = returns.index.floor("D").asi8
    week = _iso_week(returns.index)

    total_volume = _wide(df, "quote_volume").sum(axis=1, min_count=1)
    log_volume = np.log(total_volume.where(total_volume > 0))
    index_volume = log_volume - log_volume.groupby(day).transform("mean")

    # Step 1: hour of day, returns, clusters = UTC day.
    means, wald, p_primary = seasonal_wald(index_return.to_numpy(), hours, day)
    x = index_return.to_numpy()
    keep = ~np.isnan(x)
    g_idx = np.unique(hours[keep], return_inverse=True)[1]
    c_idx = np.unique(day[keep], return_inverse=True)[1]
    counts = np.bincount(g_idx).astype(float)
    u = (x[keep] - means[g_idx]) / counts[g_idx]
    scores = np.zeros((c_idx.max() + 1, len(means)))
    np.add.at(scores, (c_idx, g_idx), u)
    n_c = scores.shape[0]
    se_hour = np.sqrt(np.diag(n_c / (n_c - 1) * scores.T @ scores))
    profile = pd.DataFrame({"hour": np.arange(24), "mean_index_return": means, "se": se_hour})

    family = [_wald_row(f"hour of day: {sym}", returns[sym], hours, day) for sym in returns.columns]

    # Step 2: activity, clusters = ISO week, uncorrected.
    activity = [
        _wald_row("hour of day: |index return|", index_return.abs(), hours, week),
        _wald_row("hour of day: index log volume minus day mean", index_volume, hours, week),
    ]
    activity_profiles = pd.DataFrame({
        "hour": np.arange(24),
        "mean_abs_index_return": index_return.abs().groupby(hours).mean().to_numpy(),
        "mean_index_volume": index_volume.groupby(hours).mean().to_numpy(),
    })

    # Step 3: funding hours, index returns, clusters = UTC day.
    funding = []
    for name, chosen in (("funding hours 00/08/16 vs other", FUNDING_HOURS),
                         ("pre-funding hours 23/07/15 vs other", PRE_FUNDING_HOURS)):
        flag = np.isin(hours, chosen).astype(int)
        row = _wald_row(name, index_return, flag, day)
        row["mean_in"] = float(index_return[flag == 1].mean())
        row["mean_out"] = float(index_return[flag == 0].mean())
        funding.append(row)
    family += funding

    # Step 4: primary test alone at 5 %, Holm over the other 22 return tests.
    for row, adj in zip(family, holm(np.array([row["p"] for row in family]))):
        row["p_holm"] = float(adj)
    symbol_rows = family[: returns.shape[1]]

    # Steps 5 and 6: same-hour continuation, full panel and halves.
    rows = returns.index[:-1]
    half = len(rows) // 2
    continuation = {
        "full": _continuation(returns, rows, seed),
        "first half": _continuation(returns, rows[:half], seed + 1000),
        "second half": _continuation(returns, rows[half:], seed + 2000),
    }

    # Step 7 (optional): weekday on daily index returns, clusters = ISO week.
    daily = index_return.groupby(returns.index.floor("D")).sum(min_count=1)
    weekday = _wald_row("day of week: daily index return", daily, daily.index.dayofweek.to_numpy(),
                        _iso_week(daily.index))

    se_one_hour = float(np.mean(se_hour))
    return {
        "panel": {"start": str(returns.index[0]), "end": str(returns.index[-1]),
                  "hours": len(returns), "symbols": int(returns.shape[1])},
        "primary": {"test": "hour of day: index return", "wald": wald, "p": p_primary,
                    "reject_at_5pct": bool(p_primary < 0.05)},
        "index_profile": profile.to_dict("records"),
        "family": family,
        "symbols_rejecting_after_holm": int(sum(row["p_holm"] < 0.05 for row in symbol_rows)),
        "activity": activity,
        "activity_profiles": activity_profiles.to_dict("records"),
        "funding": funding,
        "continuation": continuation,
        "weekday": weekday,
        "detectable": {"se_one_hour_mean_index_return": se_one_hour,
                       "min_detectable_2p8_se": 2.8 * se_one_hour,
                       "round_trip_cost": 0.0010},
    }


def main(argv: list[str]) -> None:
    path = argv[1] if len(argv) > 1 else "data/train_1h.parquet"
    result = run_clock_study(pd.read_parquet(path))
    json.dump(result, sys.stdout, indent=2, default=float)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv)
