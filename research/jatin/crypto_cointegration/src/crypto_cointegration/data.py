from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "symbol", "close"}


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load long-form Parquet data and return UTC-indexed log close prices."""
    raw = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    raw = raw.loc[:, ["timestamp", "symbol", "close"]].copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

    duplicates = raw.duplicated(["timestamp", "symbol"])
    if duplicates.any():
        examples = raw.loc[duplicates, ["timestamp", "symbol"]].head()
        raise ValueError(
            f"Found {int(duplicates.sum())} duplicate timestamp/symbol rows. "
            f"Examples:\\n{examples.to_string(index=False)}"
        )

    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    invalid = raw["close"].notna() & (raw["close"] <= 0)
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} non-positive close prices.")

    prices = (
        raw.pivot(index="timestamp", columns="symbol", values="close")
        .sort_index()
    )
    prices.index.name = "timestamp"
    return np.log(prices.where(prices > 0))


def validate_prices(log_prices: pd.DataFrame) -> dict:
    """Return coverage diagnostics without silently filling missing observations."""
    if not isinstance(log_prices.index, pd.DatetimeIndex):
        raise TypeError("log_prices must have a DatetimeIndex.")
    if log_prices.index.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware (UTC recommended).")
    if log_prices.index.has_duplicates:
        raise ValueError("Duplicate timestamps in price index.")
    if not log_prices.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted.")

    expected = pd.date_range(
        log_prices.index.min(), log_prices.index.max(),
        freq="h", tz=log_prices.index.tz
    )
    aligned = log_prices.reindex(expected)
    return {
        "n_symbols": int(log_prices.shape[1]),
        "n_observations": int(log_prices.shape[0]),
        "start": log_prices.index.min(),
        "end": log_prices.index.max(),
        "is_complete_hourly_index": bool(log_prices.index.equals(expected)),
        "missing_by_symbol": aligned.isna().sum().to_dict(),
        "missing_total": int(aligned.isna().sum().sum()),
    }
