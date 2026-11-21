"""
Track 10: the absorption ratio function code
"""

import warnings

import numpy as np
import pandas as pd


def absorption_ratio(df, window= 168, k = 1):

# hours x symbols matrix of hourly log returns on a full hourly grid
    close_df = df.pivot(index="timestamp", columns="symbol", values="close")
    close_df = close_df.reindex(pd.date_range(close_df.index[0], close_df.index[-1], freq="h"))
    log_returns = np.log(close_df).diff().to_numpy()

# NaN rule
    arr = np.full(len(log_returns), np.nan)

    for t in range(window, len(log_returns)):
        window_rets = log_returns[t - window + 1 : t + 1]
        if np.isnan(window_rets).any():
            continue
        # Covariance of the last window returns up to and including t, and its eigenvalues
        eig = np.linalg.eigvalsh(np.cov(log_returns[t - window + 1 : t + 1], rowvar=False))
        
        # Absorption ratio: k largest eigenvalues over the sum of all
        total = eig.sum()
        arr[t] = min(eig[-k:].sum() / total, 1.0)

# 5. Same value on every row of that hour
    by_hour = pd.Series(arr, index=close_df.index)
    return df["timestamp"].map(by_hour)


# ---------------------------------------------------------------------------
# Part 2: per-hour rank IC, regimes by absorption ratio, day-block bootstrap
# ---------------------------------------------------------------------------


def hourly_grid(df):
    """Every hour from the first to the last timestamp in df."""
    ts = df["timestamp"]
    return pd.date_range(ts.min(), ts.max(), freq="h")


def to_wide(df, values, grid=None):
    """Row-aligned values -> (hours x symbols) frame on a full hourly grid.

    Hours missing from df become NaN rows, so a shift by one row is always
    a shift by one hour.
    """
    wide = pd.DataFrame(
        {
            "timestamp": df["timestamp"].to_numpy(),
            "symbol": df["symbol"].to_numpy(),
            "v": np.asarray(values, dtype=float),
        }
    ).pivot(index="timestamp", columns="symbol", values="v")
    if grid is None:
        grid = pd.date_range(wide.index[0], wide.index[-1], freq="h")
    return wide.reindex(grid)


def forward_returns(df, grid=None):
    """Next hour close-to-close return, (hours x symbols).

    Computed on the full hourly grid, so if hour t+1 is missing the forward
    return at t is NaN rather than a jump to the next available row.
    """
    close = to_wide(df, df["close"], grid)
    return close.shift(-1) / close - 1.0


def rank_ic_by_hour(signal_w, fwd_w, min_symbols=10):
    """Spearman correlation across symbols, one value per hour.

    Uses only symbols where both the signal and the forward return exist.
    Hours with fewer than `min_symbols` such symbols, or with a constant
    signal or return, are NaN.
    """
    s = signal_w.to_numpy(dtype=float)
    f = fwd_w.to_numpy(dtype=float)
    both = ~np.isnan(s) & ~np.isnan(f)
    s = np.where(both, s, np.nan)
    f = np.where(both, f, np.nan)

    # Spearman = Pearson on ranks (average ranks for ties)
    rs = pd.DataFrame(s).rank(axis=1).to_numpy()
    rf = pd.DataFrame(f).rank(axis=1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN hours
        rs = rs - np.nanmean(rs, axis=1, keepdims=True)
        rf = rf - np.nanmean(rf, axis=1, keepdims=True)
        num = np.nansum(rs * rf, axis=1)
        den = np.sqrt(np.nansum(rs**2, axis=1) * np.nansum(rf**2, axis=1))
        ic = num / den
    ic[(both.sum(axis=1) < min_symbols) | ~(den > 0)] = np.nan
    return pd.Series(ic, index=signal_w.index)


def regime_cutoffs(ar_by_hour, train_frac=0.4):
    """Tercile cutoffs of the absorption ratio on the first `train_frac` of hours."""
    n_train = int(len(ar_by_hour) * train_frac)
    lo, hi = np.nanquantile(ar_by_hour.iloc[:n_train].to_numpy(dtype=float), [1 / 3, 2 / 3])
    return float(lo), float(hi)


def label_regimes(ar_by_hour, lo, hi):
    """'low' if AR <= lo, 'high' if AR > hi, 'mid' in between, NaN if AR is NaN."""
    ar = ar_by_hour.to_numpy(dtype=float)
    lab = np.full(len(ar), None, dtype=object)
    lab[ar <= lo] = "low"
    lab[(ar > lo) & (ar <= hi)] = "mid"
    lab[ar > hi] = "high"
    return pd.Series(lab, index=ar_by_hour.index)


def day_block_bootstrap(ic, regime, n_boot=10000, block_days=1, seed=0):
    """Mean IC in 'high' minus mean IC in 'low', with a block bootstrap over days.

    Hours are grouped into blocks of `block_days` calendar days (UTC). Each
    replicate draws as many blocks as there are, with replacement, and
    recomputes both means from the hours in the drawn blocks.

    Returns dict: ic_high, ic_low, diff, ci_lo, ci_hi (95 % percentile),
    p (two-sided, from the bootstrap distribution centred on the estimate),
    n_high, n_low, n_blocks.
    """
    ic = pd.Series(ic, dtype=float)
    reg = pd.Series(regime).reindex(ic.index)
    valid = ic.notna()
    h = ((reg == "high") & valid).to_numpy()
    l = ((reg == "low") & valid).to_numpy()
    v = ic.fillna(0.0).to_numpy()

    days = (ic.index.floor("D") - ic.index[0].floor("D")).days.to_numpy()
    block = days // block_days
    sums = (
        pd.DataFrame({"sh": v * h, "nh": h.astype(float), "sl": v * l, "nl": l.astype(float)})
        .groupby(block)
        .sum()
        .to_numpy()
    )
    n_blocks = len(sums)

    tot = sums.sum(axis=0)
    ic_high, ic_low = tot[0] / tot[1], tot[2] / tot[3]
    diff = ic_high - ic_low

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=n_boot)
    b = counts @ sums
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = b[:, 0] / b[:, 1] - b[:, 2] / b[:, 3]
    boot = boot[np.isfinite(boot)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    p = (1 + np.sum(np.abs(boot - diff) >= abs(diff))) / (len(boot) + 1)
    return {
        "ic_high": float(ic_high),
        "ic_low": float(ic_low),
        "diff": float(diff),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p": float(p),
        "n_high": int(tot[1]),
        "n_low": int(tot[3]),
        "n_blocks": int(n_blocks),
    }


def holm(pvals):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out
