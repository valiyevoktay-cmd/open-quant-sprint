from __future__ import annotations

import numpy as np
import pandas as pd


def taker_flow_imbalance(df: pd.DataFrame, span: int = 12) -> pd.Series:
    """
    Signed taker order flow imbalance in [-1, 1], smoothed via causal EWM.
    Uses taker_buy_quote_volume vs total quote_volume.
    """
    qv = df["quote_volume"].to_numpy(dtype=np.float64)
    tbv = df["taker_buy_quote_volume"].to_numpy(dtype=np.float64)
    net_flow = 2.0 * tbv - qv
    raw_imb = np.where(qv > 0.0, net_flow / qv, 0.0)
    s = pd.Series(raw_imb, index=df.index)
    if span <= 1:
        return s
    return s.groupby(df["symbol"]).transform(lambda x: x.ewm(span=span, min_periods=1).mean())


def parkinson_volatility(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """
    Range-based Parkinson volatility estimator over a rolling causal window.
    sigma_P = sqrt(1 / (4 * N * ln(2)) * sum(ln(H_t / L_t)^2))
    """
    hl_sq = np.log(df["high"] / df["low"]) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    s = pd.Series(hl_sq * factor, index=df.index)
    return s.groupby(df["symbol"]).transform(
        lambda x: np.sqrt(x.rolling(window=window, min_periods=max(2, window // 2)).mean())
    )


def _ffd_weights(d: float, thres: float = 1e-4, max_len: int = 100) -> np.ndarray:
    w = [1.0]
    k = 1
    while k < max_len:
        w_next = -w[-1] * (d - k + 1) / k
        if abs(w_next) < thres:
            break
        w.append(w_next)
        k += 1
    return np.array(w[::-1], dtype=np.float64)


def frac_diff_ffd(series: pd.Series, d: float = 0.4, thres: float = 1e-3, max_len: int = 72) -> pd.Series:
    """
    Fixed-width window fractional differentiation (Lopez de Prado, AFML Ch. 5).
    Strictly causal convolution using past lags only.
    """
    w = _ffd_weights(d=d, thres=thres, max_len=max_len)
    width = len(w)
    vals = series.to_numpy(dtype=np.float64)
    out = np.full(len(vals), np.nan, dtype=np.float64)
    if len(vals) < width:
        return pd.Series(out, index=series.index)

    for i in range(width - 1, len(vals)):
        window_slice = vals[i - width + 1 : i + 1]
        if not np.isnan(window_slice).any():
            out[i] = float(np.dot(w, window_slice))
    return pd.Series(out, index=series.index)


def cross_sectional_zscore(df: pd.DataFrame, signal: pd.Series, clip: float = 3.0) -> pd.Series:
    """
    Dollar-neutral cross-sectional standardization at each timestamp.
    Depends ONLY on the cross-section at time t (zero temporal leakage).
    """
    s = pd.Series(signal, index=df.index).replace([np.inf, -np.inf], np.nan)
    grp = s.groupby(df["timestamp"])
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0.0, np.nan)
    z = ((s - mean) / std).fillna(0.0).clip(-clip, clip)
    z_centered = z - z.groupby(df["timestamp"]).transform("mean")
    l1 = z_centered.abs().groupby(df["timestamp"]).transform("sum").replace(0.0, 1.0)
    return z_centered / l1


def rebalance_deadband(df: pd.DataFrame, target_weights: pd.Series, buffer: float = 0.015) -> pd.Series:
    """
    Causal no-trade zone filter per symbol: only updates target weight when
    |w_target[t] - w_actual[t-1]| > buffer, suppressing low-conviction turnover.
    """
    if buffer <= 0.0:
        return pd.Series(target_weights, index=df.index)

    w_arr = pd.Series(target_weights, index=df.index).fillna(0.0).to_numpy(dtype=np.float64)
    out = np.zeros_like(w_arr)

    for _, pos in df.groupby("symbol", sort=False).indices.items():
        cur = 0.0
        for i in pos:
            tgt = w_arr[i]
            if abs(tgt - cur) > buffer:
                cur = tgt
            out[i] = cur

    return pd.Series(out, index=df.index)

