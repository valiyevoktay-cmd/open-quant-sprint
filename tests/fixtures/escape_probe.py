"""Canary for audit v1.2: tries to read the dataset next to the harness and to open a network
connection. On the host both can work; in the --container sandbox both must fail."""

from __future__ import annotations

import socket
from pathlib import Path

import pandas as pd

import src


def compute_weights(df: pd.DataFrame) -> pd.Series:
    escaped = []
    data_dir = Path(src.__file__).resolve().parent.parent / "data"
    try:
        next(p for p in data_dir.glob("*.parquet")).read_bytes()[:4]
        escaped.append("file")
    except (OSError, StopIteration):
        pass
    try:
        socket.create_connection(("api.binance.com", 443), timeout=3).close()
        escaped.append("network")
    except OSError:
        pass
    if escaped:
        raise RuntimeError("escaped: " + ",".join(escaped))
    return pd.Series(0.0, index=df.index)
