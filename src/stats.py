from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class SeriesDiagnostics:
    n_obs: int
    adf_tstat: float
    adf_pvalue: float
    is_stationary: bool
    lag1_autocorr: float
    skewness: float
    excess_kurtosis: float


def adf_first_order(y: np.ndarray) -> tuple[float, float]:
    """
    First-order Dickey-Fuller test: delta_y_t = alpha + beta * y_{t-1} + eps_t.
    Returns (t_stat, approximate MacKinnon p-value for constant-only regression).
    """
    clean = y[~np.isnan(y)]
    if len(clean) < 25:
        return 0.0, 1.0

    y_lag = clean[:-1]
    dy = np.diff(clean)
    X = np.column_stack([np.ones_like(y_lag), y_lag])
    beta, residuals, _, _ = np.linalg.lstsq(X, dy, rcond=None)
    n = len(dy)
    dof = n - 2
    if len(residuals) == 0:
        rss = float(np.sum((dy - X @ beta) ** 2))
    else:
        rss = float(residuals[0])
    s2 = rss / max(1, dof)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se_beta = float(np.sqrt(max(1e-18, cov[1, 1])))
    t_stat = float(beta[1] / se_beta)

    # MacKinnon (1994) critical values for ADF with drift (no trend):
    # 1%: -3.43, 5%: -2.86, 10%: -2.57
    if t_stat <= -3.43:
        pval = 0.005
    elif t_stat <= -2.86:
        pval = 0.01 + (t_stat - (-3.43)) / (-2.86 - (-3.43)) * 0.04
    elif t_stat <= -2.57:
        pval = 0.05 + (t_stat - (-2.86)) / (-2.57 - (-2.86)) * 0.05
    elif t_stat <= -1.95:
        pval = 0.10 + (t_stat - (-2.57)) / (-1.95 - (-2.57)) * 0.20
    else:
        pval = min(0.99, 0.30 + max(0.0, (t_stat + 1.95) * 0.25))

    return round(t_stat, 4), round(float(pval), 4)


def diagnose_series(series: pd.Series) -> SeriesDiagnostics:
    arr = series.dropna().to_numpy(dtype=np.float64)
    n = len(arr)
    if n < 30:
        return SeriesDiagnostics(n, 0.0, 1.0, False, 0.0, 0.0, 0.0)

    tstat, pval = adf_first_order(arr)
    ac1 = float( np.corrcoef(arr[:-1], arr[1:])[0, 1] ) if np.std(arr) > 1e-12 else 0.0
    sk = float(stats.skew(arr, bias=False))
    kurt = float(stats.kurtosis(arr, fisher=True, bias=False))

    return SeriesDiagnostics(
        n_obs=n,
        adf_tstat=tstat,
        adf_pvalue=pval,
        is_stationary=(pval < 0.05),
        lag1_autocorr=round(ac1, 4),
        skewness=round(sk, 4),
        excess_kurtosis=round(kurt, 4),
    )


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int = 10,
    periods_per_year: int = 8760,
    sr_variance_across_trials: float = 0.5 / 8760,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
    Returns probability in [0, 1] that true Sharpe exceeds expected maximum false-positive Sharpe
    under multiple testing, adjusted for non-normal skewness and kurtosis.
    """
    r = returns[~np.isnan(returns)]
    T = len(r)
    if T < 30:
        return 0.0

    std = float(np.std(r, ddof=1))
    if std < 1e-12:
        return 0.0

    sr_per_bar = float(np.mean(r) / std)
    sk = float(stats.skew(r, bias=False))
    # Pearson kurtosis (normal = 3.0)
    kurt_pearson = float(stats.kurtosis(r, fisher=False, bias=False))

    # Expected maximum SR across N independent trials (Euler-Mascheroni approximation)
    euler_gamma = 0.5772156649
    n_eff = max(1, int(n_trials))
    if n_eff == 1:
        sr_benchmark = 0.0
    else:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_eff)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_eff * np.e))
        sr_benchmark = float(np.sqrt(sr_variance_across_trials) * ((1.0 - euler_gamma) * z1 + euler_gamma * z2))

    denom_sq = 1.0 - sk * sr_per_bar + ((kurt_pearson - 1.0) / 4.0) * (sr_per_bar**2)
    denom = float(np.sqrt(max(1e-12, denom_sq)))
    z_stat = (sr_per_bar - sr_benchmark) * np.sqrt(T - 1.0) / denom
    return round(float(stats.norm.cdf(z_stat)), 4)


def information_coefficient(df: pd.DataFrame, weights: pd.Series) -> dict[str, float]:
    """
    Cross-sectional Spearman Rank IC between weight at bar t and forward return at bar t+1.
    Vectorized across timestamps via cross-sectional rank standardization.
    """
    work = df[["timestamp", "symbol", "close"]].copy()
    work["w"] = pd.Series(weights, index=df.index).to_numpy(dtype=np.float64)
    next_close = work.groupby("symbol")["close"].shift(-1)
    work["fwd_ret"] = (next_close - work["close"]) / work["close"]
    work = work.dropna(subset=["w", "fwd_ret"])
    if work.empty:
        return {"mean_ic": 0.0, "ic_ir": 0.0, "t_stat": 0.0, "p_value": 1.0}

    grp = work.groupby("timestamp", sort=True)
    counts = grp["w"].count()
    w_std = grp["w"].std()
    r_std = grp["fwd_ret"].std()

    work["rk_w"] = grp["w"].rank(method="average")
    work["rk_r"] = grp["fwd_ret"].rank(method="average")

    grp_rk = work.groupby("timestamp", sort=True)
    work["dw"] = work["rk_w"] - grp_rk["rk_w"].transform("mean")
    work["dr"] = work["rk_r"] - grp_rk["rk_r"].transform("mean")
    work["dw_dr"] = work["dw"] * work["dr"]
    work["dw2"] = work["dw"] ** 2
    work["dr2"] = work["dr"] ** 2

    sums = work.groupby("timestamp", sort=True)[["dw_dr", "dw2", "dr2"]].sum()
    denom = np.sqrt(sums["dw2"] * sums["dr2"])
    valid_mask = (counts >= 5) & (w_std > 1e-12) & (r_std > 1e-12) & (denom > 1e-12)

    corrs = (sums.loc[valid_mask, "dw_dr"] / denom.loc[valid_mask]).dropna().to_numpy(dtype=np.float64)
    if len(corrs) == 0:
        return {"mean_ic": 0.0, "ic_ir": 0.0, "t_stat": 0.0, "p_value": 1.0}

    mean_ic = float(np.mean(corrs))
    std_ic = float(np.std(corrs, ddof=1)) if len(corrs) > 1 else 1.0
    t_stat = mean_ic / (std_ic / np.sqrt(len(corrs))) if std_ic > 1e-12 else 0.0
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(1, len(corrs) - 1))))

    return {
        "mean_ic": round(mean_ic, 5),
        "ic_ir": round(mean_ic / std_ic if std_ic > 1e-12 else 0.0, 4),
        "t_stat": round(t_stat, 3),
        "p_value": round(p_val, 5),
    }



def sharpe_block_bootstrap_ci(
    returns: np.ndarray,
    periods_per_year: int = 8760,
    block: int = 24,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Moving-block bootstrap interval for the annualized Sharpe ratio.
    Blocks of `block` bars keep intraday autocorrelation inside each resample.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 2 * block:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    sample = r[idx]
    mean = sample.mean(axis=1)
    std = sample.std(axis=1, ddof=1)
    sr = np.where(std > 1e-12, mean / np.where(std > 1e-12, std, 1.0) * np.sqrt(periods_per_year), 0.0)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(sr, [alpha, 1.0 - alpha])
    return round(float(lo), 2), round(float(hi), 2)
