
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.contrib.tanmay import trade_intensity_imbalance


def _make_df(
    quote_volume,
    taker_buy_volume,
    trades,
    symbol="BTCUSDT",
):
    """Create a small chronological market-data panel for testing."""
    n = len(quote_volume)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=n,
                freq="h",
                tz="UTC",
            ),
            "symbol": [symbol] * n,
            "quote_volume": quote_volume,
            "taker_buy_quote_volume": taker_buy_volume,
            "trades": trades,
        }
    )


def test_sign_follows_taker_flow():
    buy_df = _make_df(
        quote_volume=[100.0] * 5,
        taker_buy_volume=[75.0] * 5,
        trades=[10.0] * 5,
    )

    sell_df = _make_df(
        quote_volume=[100.0] * 5,
        taker_buy_volume=[25.0] * 5,
        trades=[10.0] * 5,
    )

    buy_signal = trade_intensity_imbalance(buy_df)
    sell_signal = trade_intensity_imbalance(sell_df)

    assert (buy_signal > 0.0).all()
    assert (sell_signal < 0.0).all()


def test_larger_relative_trade_size_increases_magnitude():
    ordinary_df = _make_df(
        quote_volume=[100.0, 100.0, 100.0],
        taker_buy_volume=[75.0, 75.0, 75.0],
        trades=[10.0, 10.0, 10.0],
    )

    large_df = _make_df(
        quote_volume=[100.0, 100.0, 200.0],
        taker_buy_volume=[75.0, 75.0, 150.0],
        trades=[10.0, 10.0, 10.0],
    )

    ordinary_signal = trade_intensity_imbalance(
        ordinary_df, span=1, window=3
    )

    large_signal = trade_intensity_imbalance(
        large_df, span=1, window=3
    )

    assert abs(large_signal.iloc[-1]) > abs(
        ordinary_signal.iloc[-1]
    )

    assert large_signal.iloc[-1] == pytest.approx(1.0)
    assert ordinary_signal.iloc[-1] == pytest.approx(0.5)


def test_signal_is_causal():
    df = _make_df(
        quote_volume=[100.0] * 6,
        taker_buy_volume=[60.0] * 6,
        trades=[10.0] * 6,
    )

    original_signal = trade_intensity_imbalance(
        df, span=3, window=3
    )

    modified_df = df.copy()

    # Change only future observations.
    modified_df.loc[4:, "quote_volume"] = [1000.0, 2000.0]
    modified_df.loc[4:, "taker_buy_quote_volume"] = [900.0, 500.0]
    modified_df.loc[4:, "trades"] = [5.0, 10.0]

    modified_signal = trade_intensity_imbalance(
        modified_df, span=3, window=3
    )

    np.testing.assert_allclose(
        original_signal.iloc[:4].to_numpy(),
        modified_signal.iloc[:4].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )


def test_zero_denominators_produce_finite_signal():
    df = _make_df(
        quote_volume=[0.0, 100.0, 100.0],
        taker_buy_volume=[0.0, 50.0, 60.0],
        trades=[0.0, 10.0, 10.0],
    )

    signal = trade_intensity_imbalance(df)

    assert np.isfinite(signal.to_numpy()).all()
    assert signal.index.equals(df.index)


def test_each_symbol_matches_its_own_single_symbol_run():
    # Two symbols interleaved by timestamp, as in the real panel (sorted by timestamp, symbol).
    # Their volume, flow and trade size differ a lot, so pooling either the rolling median or
    # the EWM across symbols changes the output.
    n = 8
    timestamps = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    btc = {
        "quote_volume": [100.0, 200.0, 150.0, 300.0, 120.0, 250.0, 180.0, 220.0],
        "taker_buy_quote_volume": [70.0, 140.0, 90.0, 240.0, 30.0, 200.0, 60.0, 150.0],
        "trades": [10, 10, 12, 10, 15, 8, 10, 11],
    }
    eth = {
        "quote_volume": [900.0, 100.0, 1200.0, 200.0, 50.0, 700.0, 400.0, 90.0],
        "taker_buy_quote_volume": [180.0, 20.0, 900.0, 150.0, 40.0, 100.0, 300.0, 10.0],
        "trades": [90, 10, 120, 10, 5, 70, 20, 30],
    }
    df = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, 2),
            "symbol": ["BTCUSDT", "ETHUSDT"] * n,
            **{col: np.column_stack([btc[col], eth[col]]).ravel() for col in btc},
        }
    )

    combined = trade_intensity_imbalance(df, span=3, window=3)
    assert combined.index.equals(df.index)

    for symbol in ("BTCUSDT", "ETHUSDT"):
        rows = df.index[df["symbol"] == symbol]
        alone = trade_intensity_imbalance(
            df.loc[rows].reset_index(drop=True), span=3, window=3
        )
        np.testing.assert_allclose(
            combined.loc[rows].to_numpy(), alone.to_numpy(), rtol=0.0, atol=1e-12
        )