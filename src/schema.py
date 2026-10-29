from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLS = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_quote_volume",
    "trades",
    "funding_rate",
)

BINANCE_KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def normalize_binance_klines(raw_rows: list[list], symbol: str, funding_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = pd.DataFrame(raw_rows, columns=BINANCE_KLINE_COLS)
    df["timestamp"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True)
    df["symbol"] = str(symbol).upper().strip()

    float_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote_volume"]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(np.int64)

    if funding_df is not None and not funding_df.empty:
        f = funding_df.copy()
        f["timestamp"] = pd.to_datetime(f["fundingTime"].astype(np.int64), unit="ms", utc=True).dt.floor("h")
        f["funding_rate"] = pd.to_numeric(f["fundingRate"], errors="coerce").astype(np.float64)
        f = f.drop_duplicates(subset=["timestamp"], keep="last")[["timestamp", "funding_rate"]]
        df = df.merge(f, on="timestamp", how="left")
        df["funding_rate"] = df["funding_rate"].fillna(0.0)
    else:
        df["funding_rate"] = 0.0

    return enforce_schema(df[list(REQUIRED_COLS)])


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required schema columns: {missing}")

    out = df[list(REQUIRED_COLS)].copy()
    if not isinstance(out["timestamp"].dtype, pd.DatetimeTZDtype):
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert("UTC")

    out["symbol"] = out["symbol"].astype(str)
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote_volume", "funding_rate"):
        out[col] = out[col].astype(np.float64)
    out["trades"] = out["trades"].astype(np.int64)

    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def build_dollar_bars(df: pd.DataFrame, threshold_usd: float) -> pd.DataFrame:
    """Aggregate time bars into dollar bars per symbol when cumulative quote_volume >= threshold_usd."""
    if threshold_usd <= 0:
        raise ValueError("threshold_usd must be positive")

    chunks = []
    for sym, grp in df.sort_values("timestamp").groupby("symbol", sort=False):
        cum_usd = 0.0
        bucket_idx = np.zeros(len(grp), dtype=np.int64)
        b_id = 0
        qv = grp["quote_volume"].to_numpy()
        for i in range(len(qv)):
            cum_usd += qv[i]
            bucket_idx[i] = b_id
            if cum_usd >= threshold_usd:
                cum_usd = 0.0
                b_id += 1

        g = grp.assign(_bucket=bucket_idx)
        agg = g.groupby("_bucket", as_index=False).agg(
            timestamp=("timestamp", "last"),
            symbol=("symbol", "first"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            quote_volume=("quote_volume", "sum"),
            taker_buy_quote_volume=("taker_buy_quote_volume", "sum"),
            trades=("trades", "sum"),
            funding_rate=("funding_rate", "sum"),
        )
        chunks.append(agg.drop(columns=["_bucket"]))

    return enforce_schema(pd.concat(chunks, ignore_index=True))
