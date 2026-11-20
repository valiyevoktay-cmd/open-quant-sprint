"""Track 03 Part 2, step 1: measure the catch-up effect (research code, not a strategy).

Question: after a strong BTC/ETH order-flow impulse, do altcoins that lagged their beta-implied
move catch up over the next hours?

    r_lead   BTC/ETH hourly log return, weighted by each leader's quote volume that hour (as Part 1)
    beta     rolling slope of an alt's hourly return on r_lead, BETA_WINDOW hours, ending at t-1
    gap      r_alt(W) - beta * r_lead(W), returns summed over the last W hours, through bar t
    signal   -gap: an alt that lagged (negative gap) is expected to outperform
    strong   |leader_flow_impulse| above its trailing 90th percentile (QUANTILE_WINDOW hours, to t-1)

Everything up to ``signal`` and ``strong`` uses only bars at or before t. ``forward_returns`` looks
ahead on purpose: it is the target, it belongs in this notebook only and never in a strategy file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.contrib.janice import leader_flow_impulse

LEADERS = ("BTCUSDT", "ETHUSDT")


@dataclass(frozen=True)
class Params:
    """Pre-registered choices. Fix these, post them on the Issue, then run on the full panel."""

    gap_window: int = 4          # W: hours of return in the gap
    beta_window: int = 168       # 7 days of hourly returns for beta
    impulse_span: int = 4        # span of leader_flow_impulse
    strong_quantile: float = 0.9 # strong impulse = |impulse| above this trailing quantile
    quantile_window: int = 720   # 30 days for the trailing quantile
    horizons: tuple[int, ...] = (1, 2, 4, 8, 24)
    lags: tuple[int, ...] = (0, 1)  # 0 = trade at the close of t, 1 = one hour later (audit lag test)


# ---------------------------------------------------------------- causal inputs (bar t and earlier)

def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Timestamp x symbol table of one column."""
    return df.pivot(index="timestamp", columns="symbol", values=col).sort_index()


def hourly_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(wide(df, "close")).diff()


def leader_return(df: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """Quote-volume-weighted BTC/ETH hourly log return (same weighting as Part 1)."""
    qv = wide(df, "quote_volume")[list(LEADERS)]
    w = qv.div(qv.sum(axis=1).where(lambda s: s > 0), axis=0)
    return (rets[list(LEADERS)] * w).sum(axis=1, min_count=2).rename("r_lead")


def rolling_beta(alt_rets: pd.DataFrame, lead: pd.Series, window: int) -> pd.DataFrame:
    """cov(alt, lead) / var(lead) over the previous `window` hours, ending at t-1 (own implementation)."""
    min_p = window // 2
    mean_a = alt_rets.rolling(window, min_periods=min_p).mean()
    mean_l = lead.rolling(window, min_periods=min_p).mean()
    mean_al = alt_rets.mul(lead, axis=0).rolling(window, min_periods=min_p).mean()
    mean_ll = (lead**2).rolling(window, min_periods=min_p).mean()
    cov = mean_al.sub(mean_a.mul(mean_l, axis=0))
    var = (mean_ll - mean_l**2).where(lambda v: v > 1e-12)
    return cov.div(var, axis=0).shift(1)


def catch_up_signal(df: pd.DataFrame, p: Params) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (-gap, -r_alt(W)) for the 18 alts. The second is the plain-reversal control."""
    rets = hourly_log_returns(df)
    alts = [c for c in rets.columns if c not in LEADERS]
    lead = leader_return(df, rets)
    beta = rolling_beta(rets[alts], lead, p.beta_window)
    r_alt_w = rets[alts].rolling(p.gap_window).sum()
    r_lead_w = lead.rolling(p.gap_window).sum()
    gap = r_alt_w - beta.mul(r_lead_w, axis=0)
    return -gap, -r_alt_w


def strong_impulse(df: pd.DataFrame, p: Params) -> tuple[pd.Series, pd.Series]:
    """(strong, impulse) per hour: strong is True where |impulse| exceeds its trailing quantile; NaN counts as not strong."""
    imp = leader_flow_impulse(df, span=p.impulse_span).groupby(df["timestamp"]).first().sort_index()
    size = imp.abs()
    cut = size.rolling(p.quantile_window, min_periods=p.quantile_window // 4).quantile(p.strong_quantile).shift(1)
    return (size > cut).rename("strong"), imp.rename("impulse")


# ---------------------------------------------------------------- targets (look ahead on purpose)

def forward_returns(rets: pd.DataFrame, h: int, lag: int) -> pd.DataFrame:
    """Log return over (t+lag, t+lag+h]. TARGET ONLY: never use inside compute_weights."""
    return rets.rolling(h).sum().shift(-(h + lag))


# ---------------------------------------------------------------- statistics

def xs_rank_ic(signal: pd.DataFrame, target: pd.DataFrame, min_names: int = 5) -> pd.Series:
    """Spearman correlation across symbols at each hour. NaN where fewer than `min_names` pairs."""
    ok = signal.notna() & target.notna()
    s, t = signal.where(ok), target.where(ok)
    rs, rt = s.rank(axis=1), t.rank(axis=1)
    ds, dt = rs.sub(rs.mean(axis=1), axis=0), rt.sub(rt.mean(axis=1), axis=0)
    denom = np.sqrt((ds**2).sum(axis=1) * (dt**2).sum(axis=1))
    ic = (ds * dt).sum(axis=1) / denom.where(denom > 1e-12)
    return ic.where(ok.sum(axis=1) >= min_names)


def newey_west_ols(y: np.ndarray, X: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West (Bartlett) standard errors."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for k in range(1, n_lags + 1):
        w = 1.0 - k / (n_lags + 1)
        G = Xu[k:].T @ Xu[:-k]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return beta, np.sqrt(np.diag(cov))


def strong_vs_other(ic: pd.Series, strong: pd.Series, h: int) -> dict[str, float]:
    """Mean IC in strong and other hours, and the difference, with Newey-West errors (h lags).

    Regression of the hourly IC on [1, strong]: intercept = other-hours IC, slope = strong minus other.
    Also reports the difference on every h-th hour only, where forward windows do not overlap.
    """
    d = pd.concat([ic.rename("ic"), strong.rename("strong")], axis=1).dropna()
    y, s = d["ic"].to_numpy(), d["strong"].astype(float).to_numpy()
    X = np.column_stack([np.ones_like(s), s])
    (b0, b1), (se0, se1) = newey_west_ols(y, X, n_lags=max(h, 1))
    sub = d.iloc[::h]
    diff_sub = sub.loc[sub["strong"].astype(bool), "ic"].mean() - sub.loc[~sub["strong"].astype(bool), "ic"].mean()
    return {
        "ic_other": b0, "ic_strong": b0 + b1, "diff": b1, "se_diff": se1, "t_diff": b1 / se1,
        "se_other": se0, "n_hours": len(d), "n_strong": int(s.sum()), "diff_every_hth": diff_sub,
    }


def ic_table(signal: pd.DataFrame, rets_alts: pd.DataFrame, strong: pd.Series, p: Params) -> pd.DataFrame:
    """One row per (lag, horizon): strong vs other hours."""
    rows = []
    for lag in p.lags:
        for h in p.horizons:
            ic = xs_rank_ic(signal, forward_returns(rets_alts, h, lag))
            rows.append({"lag": lag, "h": h, **strong_vs_other(ic, strong.reindex(ic.index), h)})
    return pd.DataFrame(rows).set_index(["lag", "h"])


def ic_by_quarter(signal: pd.DataFrame, rets_alts: pd.DataFrame, strong: pd.Series, h: int, lag: int) -> pd.DataFrame:
    ic = xs_rank_ic(signal, forward_returns(rets_alts, h, lag))
    d = pd.concat([ic.rename("ic"), strong.reindex(ic.index).rename("strong")], axis=1).dropna()
    q = d.index.tz_localize(None).to_period("Q") if d.index.tz is not None else d.index.to_period("Q")
    g = d.groupby([q, d["strong"].astype(bool)])["ic"].mean().unstack()
    return g.rename(columns={True: "strong", False: "other"}).assign(diff=lambda x: x["strong"] - x["other"])
