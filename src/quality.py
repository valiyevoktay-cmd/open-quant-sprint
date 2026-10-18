from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.schema import REQUIRED_COLS, enforce_schema


@dataclass(frozen=True)
class QualityReport:
    dqs: float
    total_rows: int
    n_symbols: int
    null_rate: float
    duplicate_keys: int
    ohlc_violations: int
    non_positive_prices: int
    timestamp_gaps: int
    extreme_outliers: int
    status: str


def audit_dataset(df: pd.DataFrame, expected_freq_hours: int = 1) -> QualityReport:
    n = len(df)
    if n == 0:
        return QualityReport(0.0, 0, 0, 1.0, 0, 0, 0, 0, 0, "FAIL")

    null_rate = float(df[list(REQUIRED_COLS)].isna().mean().mean())
    dup_keys = int(df.duplicated(subset=["timestamp", "symbol"]).sum())

    ohlc_bad = int(
        (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
            | (df["volume"] < 0)
            | (df["quote_volume"] < 0)
            | (df["taker_buy_quote_volume"] < 0)
            | (df["taker_buy_quote_volume"] > df["quote_volume"] * 1.0001)
        ).sum()
    )

    non_pos = int(((df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)).sum())

    # Check grid continuity per symbol
    gaps = 0
    expected_delta = pd.Timedelta(hours=expected_freq_hours)
    for _, grp in df.sort_values("timestamp").groupby("symbol"):
        diffs = grp["timestamp"].diff().dropna()
        gaps += int((diffs > expected_delta).sum())

    # Modified Z-score (MAD) on log returns to detect broken price spikes (> 15 MADs)
    outliers = 0
    for _, grp in df.sort_values("timestamp").groupby("symbol"):
        ret = np.log(grp["close"] / grp["close"].shift(1)).dropna().to_numpy()
        if len(ret) > 10:
            med = np.median(ret)
            mad = np.median(np.abs(ret - med))
            if mad > 1e-12:
                mod_z = 0.6745 * np.abs(ret - med) / mad
                outliers += int((mod_z > 15.0).sum())

    # Composite Data Quality Score (0-100)
    s_completeness = max(0.0, 1.0 - null_rate * 20.0) * 30.0
    s_uniqueness = max(0.0, 1.0 - (dup_keys / n) * 50.0) * 15.0
    s_validity = max(0.0, 1.0 - ((ohlc_bad + non_pos) / n) * 50.0) * 25.0
    s_consistency = max(0.0, 1.0 - (outliers / n) * 20.0) * 15.0
    s_timeliness = max(0.0, 1.0 - (gaps / max(1, n)) * 25.0) * 15.0

    dqs = round(s_completeness + s_uniqueness + s_validity + s_consistency + s_timeliness, 2)
    status = "READY" if dqs >= 85.0 else ("USABLE" if dqs >= 65.0 else "REMEDIATE")

    return QualityReport(
        dqs=dqs,
        total_rows=n,
        n_symbols=int(df["symbol"].nunique()),
        null_rate=round(null_rate, 6),
        duplicate_keys=dup_keys,
        ohlc_violations=ohlc_bad,
        non_positive_prices=non_pos,
        timestamp_gaps=gaps,
        extreme_outliers=outliers,
        status=status,
    )


def autoclean_market_data(df: pd.DataFrame, max_ffill_bars: int = 3) -> pd.DataFrame:
    """
    Deterministic market data cleaner:
    - Deduplicates (timestamp, symbol) keeping last
    - Drops non-positive prices
    - Repairs minor OHLC envelope inversions (high = max(O,H,L,C), low = min(O,H,L,C))
    - Clips taker_buy_quote_volume to [0, quote_volume]
    - Reindexes onto regular hourly grid per symbol and forward-fills short gaps (<= max_ffill_bars)
    """
    clean = enforce_schema(df)
    clean = clean.drop_duplicates(subset=["timestamp", "symbol"], keep="last")

    valid_price = (clean["open"] > 0) & (clean["high"] > 0) & (clean["low"] > 0) & (clean["close"] > 0)
    clean = clean.loc[valid_price].copy()

    ohlc_Arr = clean[["open", "high", "low", "close"]].to_numpy()
    clean["high"] = np.nanmax(ohlc_Arr, axis=1)
    clean["low"] = np.nanmin(ohlc_Arr, axis=1)

    clean["volume"] = np.maximum(0.0, clean["volume"])
    clean["quote_volume"] = np.maximum(0.0, clean["quote_volume"])
    clean["taker_buy_quote_volume"] = np.clip(clean["taker_buy_quote_volume"], 0.0, clean["quote_volume"])
    clean["funding_rate"] = clean["funding_rate"].fillna(0.0)

    # Align to uniform hourly grid across symbols
    t_min = clean["timestamp"].min()
    t_max = clean["timestamp"].max()
    full_idx = pd.date_range(t_min, t_max, freq="1h", tz="UTC")

    reindexed = []
    for sym, grp in clean.groupby("symbol", sort=False):
        g = grp.set_index("timestamp").reindex(full_idx)
        g.index.name = "timestamp"
        g["symbol"] = sym

        g["close"] = g["close"].ffill(limit=max_ffill_bars)
        for col in ("open", "high", "low"):
            g[col] = g[col].fillna(g["close"])

        for col in ("volume", "quote_volume", "taker_buy_quote_volume", "funding_rate"):
            g[col] = g[col].fillna(0.0)
        g["trades"] = g["trades"].fillna(0).astype(np.int64)

        g = g.dropna(subset=["close"]).reset_index()
        reindexed.append(g)

    return enforce_schema(pd.concat(reindexed, ignore_index=True))
