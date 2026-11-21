"""Square-root market impact as a per-row fee for run_backtest, by Yukun (Track 14).

The cost of trading Q notional is priced as one parent order worked over about an hour:

    impact = (2/3) * y * sigma_d * sqrt(Q / V_d),    Q = aum * turnover

with sigma_d the daily volatility and V_d the daily quote volume of the symbol. It goes into the
backtest without touching it:

    fee = 5.0 + np.nan_to_num(impact_bps(df, weights, aum))
    run_backtest(df, weights, fee_bps=fee)

Every value at row t uses only bars of the same symbol before t, since the trade charged on row t
is made at the close of bar t - 1. Like run_backtest, the functions expect df sorted by
(timestamp, symbol) with one row per symbol and hour, and return one value per row of df.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import coerce_weights


def turnover(df: pd.DataFrame, weights, lag_bars: int = 1) -> np.ndarray:
    """Turnover that run_backtest charges on each row; the lines are copied from run_backtest."""
    return np.abs(weight_change(df, weights, lag_bars))


def weight_change(df: pd.DataFrame, weights, lag_bars: int = 1) -> np.ndarray:
    """Signed change of the held weight on each row: positive for a buy, negative for a sell."""
    if lag_bars < 1:
        raise ValueError(f"lag_bars must be >= 1 to prevent same-bar execution leak, got {lag_bars}")

    work = df[["timestamp", "symbol"]].copy()
    work["w_raw"] = coerce_weights(df, weights).to_numpy(dtype=np.float64)

    gross = work["w_raw"].abs().groupby(work["timestamp"]).transform("sum")
    scale = np.where(gross > 1.0, 1.0 / gross, 1.0)
    work["w"] = work["w_raw"] * scale

    work["w_held"] = work.groupby("symbol")["w"].shift(lag_bars).fillna(0.0)
    work["w_prev_held"] = work.groupby("symbol")["w_held"].shift(1).fillna(0.0)
    return (work["w_held"] - work["w_prev_held"]).to_numpy(dtype=np.float64)


def daily_sigma_volume(df: pd.DataFrame, days: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Daily volatility and daily quote volume per row, from the last 24 * days bars before it.

    sigma_d: sample std (ddof=1) of hourly log returns times sqrt(24); at row t with days=30, the
    720 returns between the closes of bars t - 721 and t - 1.
    V_d: mean quote_volume of bars t - 720 to t - 1, times 24.
    Both need at least 24 observations (min_periods=24), so sigma_d is NaN on the first 25 bars of
    a symbol and V_d on the first 24.
    """
    if days < 1 or int(days) != days:
        raise ValueError(f"days must be a positive integer, got {days}")
    window = 24 * int(days)
    symbol = df["symbol"]

    log_ret = np.log(df["close"]).groupby(symbol).diff()
    sigma = log_ret.groupby(symbol).transform(
        lambda x: x.rolling(window, min_periods=24).std(ddof=1).shift(1)
    ) * np.sqrt(24)
    volume = df["quote_volume"].groupby(symbol).transform(
        lambda x: x.rolling(window, min_periods=24).mean().shift(1)
    ) * 24
    return sigma.to_numpy(dtype=np.float64), volume.to_numpy(dtype=np.float64)


def impact_bps(
    df: pd.DataFrame,
    weights,
    aum: float,
    y: float = 1.0,
    days: int = 30,
    lag_bars: int = 1,
) -> np.ndarray:
    """Average square-root impact in bps of the notional traded on each row.

    0 where the turnover is 0. NaN where the turnover is positive but sigma_d or V_d is not
    available (the first 25 bars of a symbol) or V_d is 0. aum and quote_volume are in USDT.
    """
    if not aum > 0:
        raise ValueError(f"aum must be positive, got {aum}")
    if not y >= 0:
        raise ValueError(f"y must be non-negative, got {y}")

    to = turnover(df, weights, lag_bars)
    sigma, volume = daily_sigma_volume(df, days)
    volume = np.where(volume > 0, volume, np.nan)  # no traded volume, no estimate (not inf)

    cost = (2.0 / 3.0) * y * sigma * np.sqrt(aum * to / volume) * 1e4
    return np.where(to > 0, cost, 0.0)


def run_impact_cost(
    df: pd.DataFrame,
    weights,
    aum: float,
    y: float = 1.0,
    days: int = 30,
    lag_bars: int = 1,
    gap: int = 0,
    tol: float = 1e-12,
) -> np.ndarray:
    """Impact cost, as a fraction of aum, when runs of same-direction trades are one parent order.

    Per symbol, a trade is a row whose weight change exceeds tol in absolute value. Trades with the
    same sign belong to one run while at most `gap` bars without a trade lie between them; a trade
    in the other direction starts a new run. A run of total turnover S whose first row is r costs

        S * (2/3) * y * sigma_d[r] * sqrt(aum * S / V_d[r])

    placed on row r, 0 on every other row. A run of one row costs turnover * impact_bps * 1e-4.
    The cost on row r depends on the later rows of its run, so this is an ex-post total for
    analysis, not a fee to pass to run_backtest. NaN where sigma_d or V_d is not available at r.
    """
    if not aum > 0:
        raise ValueError(f"aum must be positive, got {aum}")
    if not y >= 0:
        raise ValueError(f"y must be non-negative, got {y}")
    if gap < 0 or int(gap) != gap:
        raise ValueError(f"gap must be a non-negative integer, got {gap}")

    change = weight_change(df, weights, lag_bars)
    sigma, volume = daily_sigma_volume(df, days)
    volume = np.where(volume > 0, volume, np.nan)

    rows = np.flatnonzero(np.abs(change) > tol)
    trades = pd.DataFrame(
        {
            "symbol": df["symbol"].to_numpy()[rows],
            "pos": df.groupby("symbol").cumcount().to_numpy()[rows],
            "sign": np.sign(change[rows]),
            "size": np.abs(change[rows]),
        }
    )
    by_symbol = trades.groupby("symbol")
    new_run = (
        by_symbol["sign"].shift(1).ne(trades["sign"])  # first trade of a symbol: NaN != sign
        | (trades["pos"] - by_symbol["pos"].shift(1) - 1 > gap)
    )
    run_id = new_run.groupby(trades["symbol"]).cumsum()
    run_size = trades["size"].groupby([trades["symbol"], run_id]).transform("sum").to_numpy()

    first = rows[new_run.to_numpy()]
    size = run_size[new_run.to_numpy()]
    cost = np.zeros(len(df))
    cost[first] = size * (2.0 / 3.0) * y * sigma[first] * np.sqrt(aum * size / volume[first])
    return cost
