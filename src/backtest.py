from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.stats import deflated_sharpe_ratio


@dataclass(frozen=True)
class BacktestMetrics:
    fee_bps: float
    ann_return: float
    ann_vol: float
    sharpe: float
    deflated_sharpe: float
    max_drawdown: float
    mean_turnover: float
    net_pnl_cum: float
    hourly_returns: np.ndarray


def coerce_weights(df: pd.DataFrame, weights: pd.Series | np.ndarray | list) -> pd.Series:
    if isinstance(weights, pd.Series):
        if len(weights) != len(df):
            raise ValueError(f"Weight length ({len(weights)}) does not match DataFrame length ({len(df)})")
        s = weights.reindex(df.index) if weights.index.equals(df.index) else pd.Series(weights.to_numpy(), index=df.index)
    else:
        arr = np.asarray(weights, dtype=np.float64)
        if arr.ndim != 1 or len(arr) != len(df):
            raise ValueError(f"Weights must be 1D of length {len(df)}, got shape {arr.shape}")
        s = pd.Series(arr, index=df.index)

    return s.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def run_backtest(
    df: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    fee_bps: float = 5.0,
    include_funding: bool = True,
    periods_per_year: int = 8760,
    n_trials: int = 10,
) -> BacktestMetrics:
    """
    Vectorized cross-sectional portfolio backtest.
    Enforces a strict 1-bar execution delay: signal computed at bar t close
    earns the return from t to t+1.
    """
    work = df[["timestamp", "symbol", "close", "funding_rate"]].copy()
    work["w_raw"] = coerce_weights(df, weights).to_numpy(dtype=np.float64)

    # Normalize gross exposure at each bar so sum(abs(w)) <= 1.0
    gross = work.groupby("timestamp")["w_raw"].transform(lambda x: x.abs().sum())
    scale = np.where(gross > 1.0, 1.0 / gross, 1.0)
    work["w"] = work["w_raw"] * scale

    # 1-bar execution lag: weight held during (t-1 -> t) is w_{t-1}
    work["w_held"] = work.groupby("symbol")["w"].shift(1).fillna(0.0)
    work["w_prev_held"] = work.groupby("symbol")["w_held"].shift(1).fillna(0.0)

    work["ret"] = work.groupby("symbol")["close"].pct_change().fillna(0.0)
    work["turnover"] = (work["w_held"] - work["w_prev_held"]).abs()

    fee_rate = fee_bps * 1e-4
    work["gross_pnl"] = work["w_held"] * work["ret"]
    work["fee_cost"] = work["turnover"] * fee_rate
    # Long pays positive funding rate, short receives positive funding rate
    work["funding_cost"] = (work["w_held"] * work["funding_rate"]) if include_funding else 0.0
    work["net_pnl"] = work["gross_pnl"] - work["fee_cost"] - work["funding_cost"]

    by_ts = work.groupby("timestamp", sort=True).agg(
        port_ret=("net_pnl", "sum"),
        turnover=("turnover", "sum"),
    )

    r = by_ts["port_ret"].to_numpy(dtype=np.float64)
    if len(r) < 2:
        return BacktestMetrics(fee_bps, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, r)

    mean_r = float(np.mean(r))
    std_r = float(np.std(r, ddof=1))
    ann_ret = mean_r * periods_per_year
    ann_vol = std_r * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    dsr = deflated_sharpe_ratio(r, n_trials=n_trials, periods_per_year=periods_per_year)

    equity = np.cumsum(r)
    peak = np.maximum.accumulate(np.insert(equity, 0, 0.0))[1:]
    dd = equity - peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    return BacktestMetrics(
        fee_bps=float(fee_bps),
        ann_return=round(ann_ret, 4),
        ann_vol=round(ann_vol, 4),
        sharpe=round(sharpe, 3),
        deflated_sharpe=dsr,
        max_drawdown=round(max_dd, 4),
        mean_turnover=round(float(by_ts["turnover"].mean()), 4),
        net_pnl_cum=round(float(equity[-1]), 4),
        hourly_returns=r,
    )
