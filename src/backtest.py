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
    gross_pnl_cum: float = 0.0
    fee_cost_cum: float = 0.0
    funding_cost_cum: float = 0.0


def coerce_weights(df: pd.DataFrame, weights: pd.Series | np.ndarray | list) -> pd.Series:
    if len(weights) != len(df):
        raise ValueError(f"Weight length ({len(weights)}) does not match DataFrame length ({len(df)})")

    if isinstance(weights, pd.Series):
        if weights.index.equals(df.index):
            s = weights.copy()
        elif set(weights.index) == set(df.index) and len(weights.index) == len(df.index):
            s = weights.reindex(df.index)
        elif isinstance(weights.index, pd.RangeIndex) and isinstance(df.index, pd.RangeIndex):
            s = weights.copy()
            s.index = df.index
        else:
            raise ValueError(
                f"Weights index does not match input DataFrame index. "
                f"Ensure weights are returned aligned with `df.index`."
            )
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
    lag_bars: int = 1,
) -> BacktestMetrics:
    """
    Vectorized cross-sectional portfolio backtest.
    Supports variable execution delay (default lag_bars=1): signal computed at bar t close
    is executed and held during bar t+lag_bars.
    """
    if lag_bars < 1:
        raise ValueError(f"lag_bars must be >= 1 to prevent same-bar execution leak, got {lag_bars}")

    work = df[["timestamp", "symbol", "close", "funding_rate"]].copy()
    work["w_raw"] = coerce_weights(df, weights).to_numpy(dtype=np.float64)

    # Normalize gross exposure at each bar so sum(abs(w)) <= 1.0
    gross = work["w_raw"].abs().groupby(work["timestamp"]).transform("sum")
    scale = np.where(gross > 1.0, 1.0 / gross, 1.0)
    work["w"] = work["w_raw"] * scale

    # Execution lag: weight held during bar t is w_{t - lag_bars}
    work["w_held"] = work.groupby("symbol")["w"].shift(lag_bars).fillna(0.0)
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
        gross_pnl_cum=round(float(work["gross_pnl"].sum()), 4),
        fee_cost_cum=round(float(work["fee_cost"].sum()), 4),
        funding_cost_cum=round(float(work["funding_cost"].sum()), 4) if include_funding else 0.0,
    )

