"""Track 03: Janice's strategies on a panel with missing hours. Maintainer addition before the freeze.

Her code stops on a panel that is not complete on the hourly grid, and her tests require that. In the
official runs the data builder fills a gap of up to three hours (close carried forward, open, high and
low equal to the close, zero volume, zero funding) and drops the hours of a longer gap, so only a
longer gap can reach a strategy, and it would stop every one of her three strategies at that hour.

`on_full_grid` fills the missing cells the way the builder fills a short gap, calls the strategy on
the filled panel and returns the weights of the rows it was given. A filled cell uses only earlier
bars of its symbol, so the weights stay causal. A complete panel, or one her code rejects for another
reason, goes to the strategy unchanged, so on a complete panel the weights are exactly hers.
"""

import numpy as np
import pandas as pd

ZERO_FILL = ("volume", "quote_volume", "taker_buy_quote_volume", "funding_rate", "trades")


def on_full_grid(fn, df: pd.DataFrame) -> pd.Series:
    if df.empty or df[["timestamp", "symbol"]].isna().any().any():
        return fn(df)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    if df.assign(timestamp=ts).duplicated(["timestamp", "symbol"]).any() or not (ts == ts.dt.floor("h")).all():
        return fn(df)
    grid = pd.date_range(ts.min(), ts.max(), freq="h").as_unit(ts.dt.unit)
    symbols = sorted(df["symbol"].unique())
    if len(df) == len(grid) * len(symbols):
        return fn(df)

    work = df.assign(timestamp=ts, _row=np.arange(len(df)))
    full = pd.MultiIndex.from_product([grid, symbols], names=["timestamp", "symbol"])
    work = work.set_index(["timestamp", "symbol"]).reindex(full).reset_index()
    work = work.sort_values(["symbol", "timestamp"], kind="mergesort")
    work["close"] = work.groupby("symbol", sort=False)["close"].ffill()
    for col in ("open", "high", "low"):
        if col in work:
            work[col] = work[col].fillna(work["close"])
    for col in ZERO_FILL:
        if col in work:
            work[col] = work[col].fillna(0)
    work = work.dropna(subset=["close"]).sort_values(["timestamp", "symbol"], kind="mergesort")
    work = work.reset_index(drop=True)
    for col in df.columns:
        if col in work and col not in ("timestamp", "symbol"):
            work[col] = work[col].astype(df[col].dtype)

    rows = work.pop("_row")
    weights = fn(work)
    given = rows.notna().to_numpy()
    out = np.empty(len(df), dtype=float)
    out[rows[given].astype(int).to_numpy()] = np.asarray(weights, dtype=float)[given]
    return pd.Series(out, index=df.index, name=getattr(weights, "name", None))
