"""Track 03 strategies on a panel with a missing hour (maintainer change before the freeze).

In the official runs the strategy is called on the panel up to each hour, and the data builder drops
the hours of a gap longer than three hours. The three strategies used to raise on such a panel. They
now return finite weights: the weights before the gap are the same as on the complete panel, and all
of them are the same as on the panel with the gap filled the way the builder fills a short gap.
"""

import numpy as np
import pandas as pd
import pytest

from src.contrib.janice_gaps import on_full_grid
from src.quality import autoclean_market_data
from strategies.janice_gradual_catchup import compute_weights as gradual_catchup
from strategies.janice_leadlag import compute_weights as leadlag
from strategies.janice_slow_momentum import compute_weights as slow_momentum

SYMBOLS = ["BTCUSDT", "ETHUSDT"] + [f"A{i:02d}USDT" for i in range(18)]
HOURS = 1000


def make_panel(hours=HOURS, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    lead = rng.normal(0, 0.004, hours)
    rows = []
    for j, sym in enumerate(SYMBOLS):
        r = 0.8 * lead + rng.normal(0, 0.003, hours)
        close = 100.0 * (j + 1) * np.exp(np.cumsum(r))
        qv = rng.uniform(1e5, 2e5, hours)
        rows.append(pd.DataFrame({
            "timestamp": ts, "symbol": sym, "open": close, "high": close * 1.002, "low": close * 0.998,
            "close": close, "volume": qv / close, "quote_volume": qv,
            "taker_buy_quote_volume": qv * rng.uniform(0.3, 0.7, hours), "funding_rate": 0.0, "trades": 100,
        }))
    return pd.concat(rows).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


@pytest.mark.parametrize("fn", [leadlag, gradual_catchup, slow_momentum])
@pytest.mark.parametrize("gap", ["one alt row", "whole hour", "six hours"])
def test_missing_hours_give_finite_weights_and_leave_earlier_hours_alone(fn, gap):
    df = make_panel()
    ts = df["timestamp"].unique()
    if gap == "one alt row":
        holed = df.drop(df[(df["timestamp"] == ts[900]) & (df["symbol"] == "A03USDT")].index)
    elif gap == "whole hour":
        holed = df[df["timestamp"] != ts[900]]
    else:
        holed = df[~df["timestamp"].isin(ts[900:906])]

    full = fn(df)
    w = fn(holed)
    assert len(w) == len(holed)
    assert np.isfinite(w.to_numpy()).all()
    before = holed["timestamp"] < ts[900]
    np.testing.assert_array_equal(w[before].to_numpy(), full.loc[w[before].index].to_numpy())

    # The same weights as on the panel the data builder would have made, gap filled.
    filled = autoclean_market_data(holed, max_ffill_bars=6)
    ref = pd.Series(fn(filled).to_numpy(), index=pd.MultiIndex.from_frame(filled[["timestamp", "symbol"]]))
    keys = pd.MultiIndex.from_arrays([pd.to_datetime(holed["timestamp"], utc=True), holed["symbol"]])
    np.testing.assert_array_equal(w.to_numpy(), ref.reindex(keys).to_numpy())


def test_complete_panel_goes_to_the_strategy_unchanged():
    df = make_panel()
    seen = []
    on_full_grid(lambda d: seen.append(d) or pd.Series(0.0, index=d.index), df)
    assert seen[0] is df
