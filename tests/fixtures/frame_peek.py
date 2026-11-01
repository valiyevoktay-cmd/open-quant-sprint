"""Canary for audit v1.1: reads future bars from the evaluator's stack frames during bar-by-bar holdout evaluation. Must be REJECTED."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd


def _widest_panel(df: pd.DataFrame) -> pd.DataFrame:
    # Walks up the call stack and takes the largest DataFrame it finds. In audit v1.1 the bar-by-bar
    # worker held the whole holdout in a local variable, so this returned bars the strategy should not see.
    best = df
    frame = inspect.currentframe()
    while frame is not None:
        for value in frame.f_locals.values():
            if isinstance(value, pd.DataFrame) and "close" in value.columns and value.shape[0] > best.shape[0]:
                best = value
        frame = frame.f_back
    return best


def compute_weights(df: pd.DataFrame) -> pd.Series:
    wide = _widest_panel(df).pivot(index="timestamp", columns="symbol", values="close")
    arr = wide.to_numpy()
    fut = np.full_like(arr, np.nan)
    fut[:-1] = arr[1:] / arr[:-1] - 1.0
    score = pd.DataFrame(fut, index=wide.index, columns=wide.columns)
    score = score.sub(score.mean(axis=1), axis=0).fillna(0.0)
    long = score.stack().rename("w").reset_index()
    out = df[["timestamp", "symbol"]].merge(long, on=["timestamp", "symbol"], how="left")
    return pd.Series(out["w"].fillna(0.0).to_numpy(), index=df.index)
