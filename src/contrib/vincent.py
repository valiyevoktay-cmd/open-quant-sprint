"""
Track 10: the absorption ratio function code
"""

#Part 1

import numpy as np
import pandas as pd

# hours x symbols matrix of one column on a full hourly grid (used by Part 1 and Part 2)
def to_wide(df, column):
    wide_df = df.pivot(index="timestamp", columns="symbol", values=column)
    wide_df = wide_df.reindex(pd.date_range(wide_df.index[0], wide_df.index[-1], freq="h"))
    return wide_df


# hours x symbols matrix of hourly log returns on a full hourly grid (used by Part 1 and Part 2)
def hourly_log_returns(df):
    return np.log(to_wide(df, "close")).diff()

# function for absorption ratio
def absorption_ratio(df, window= 168, k = 1):

# hours x symbols matrix of hourly log returns on a full hourly grid
    returns_df = hourly_log_returns(df)
    log_returns = returns_df.to_numpy()

# NaN rule
    arr = np.full(len(log_returns), np.nan)

    for t in range(window, len(log_returns)):
        window_rets = log_returns[t - window + 1 : t + 1]
        if np.isnan(window_rets).any():
            continue
        # Covariance of the last window returns up to and including t, and its eigenvalues
        eig = np.linalg.eigvalsh(np.cov(window_rets, rowvar=False))

        # Absorption ratio: k largest eigenvalues over the sum of all
        total = eig.sum()
        arr[t] = min(eig[-k:].sum() / total, 1.0)

# Same value on every row of that hour
    by_hour = pd.Series(arr, index=returns_df.index)
    return df["timestamp"].map(by_hour)



# Part 2

# next hour's close-to-close log return at each hour
def forward_returns(df):
    return hourly_log_returns(df).shift(-1)


# per-hour rank IC
def rank_ic_by_hour(signal_df, fwd_df, min_symbols=10):

# rank symbols within each hour
    both = signal_df.notna() & fwd_df.notna()
    signal_ranks = signal_df.where(both).rank(axis=1)
    fwd_ranks = fwd_df.where(both).rank(axis=1)

# Spearman = Pearson correlation of the ranks
    signal_ranks = signal_ranks.sub(signal_ranks.mean(axis=1), axis=0)
    fwd_ranks = fwd_ranks.sub(fwd_ranks.mean(axis=1), axis=0)
    ic = (signal_ranks * fwd_ranks).sum(axis=1) / np.sqrt(
        (signal_ranks**2).sum(axis=1) * (fwd_ranks**2).sum(axis=1))

    return ic.where(both.sum(axis=1) >= min_symbols)


# absorption ratio terciles, fixed on the first 40 % of hours only
def regime_cutoffs(ar_by_hour, train_frac=0.4):
    n_train = int(len(ar_by_hour) * train_frac)
    lo, hi = ar_by_hour.iloc[:n_train].quantile([1 / 3, 2 / 3])
    return lo, hi


# Label type of regime
def label_regimes(ar_by_hour, lo, hi):
    return pd.cut(ar_by_hour, [-np.inf, lo, hi, np.inf], labels=["low", "mid", "high"])


# mean IC in high-AR hours minus mean IC in low-AR hours, with a block bootstrap over days
def day_block_bootstrap(ic, regime, n_boot=10000, block_days=1, seed=0):

# IC sums and hour counts per day
    high = (regime == "high") & ic.notna()
    low = (regime == "low") & ic.notna()
    block = (ic.index.floor("D") - ic.index[0].floor("D")).days // block_days
    sums = pd.DataFrame(
        {"ic_high": ic.where(high, 0.0), "n_high": high, "ic_low": ic.where(low, 0.0), "n_low": low}
    ).groupby(block).sum().to_numpy(dtype=float)

# the real difference
    tot = sums.sum(axis=0)
    diff = tot[0] / tot[1] - tot[2] / tot[3]

# resample days with replacement
    rng = np.random.default_rng(seed)
    n_blocks = len(sums)
    counts = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=n_boot)
    b = counts @ sums
    boot = b[:, 0] / b[:, 1] - b[:, 2] / b[:, 3]

# 95 % interval and p-value
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    p = (1 + np.sum(np.abs(boot - diff) >= abs(diff))) / (n_boot + 1)
    return {
        "ic_high": tot[0] / tot[1],
        "ic_low": tot[2] / tot[3],
        "diff": diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p": p,
        "n_high": int(tot[1]),
        "n_low": int(tot[3]),
        "n_blocks": n_blocks,
    }


# Holm step-down adjusted p-values
def holm(pvals):
    p = np.asarray(pvals)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out
