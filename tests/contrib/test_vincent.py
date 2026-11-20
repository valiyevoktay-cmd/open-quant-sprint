"""
Track 10: tests for the absorption ratio function
"""

import numpy as np
import pandas as pd

from src.contrib.vincent import absorption_ratio

N = 20
WINDOW = 168


def _panel(log_returns):
    hours, n = log_returns.shape
    ts = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(
        {
            "timestamp": np.repeat(ts, n),
            "symbol": np.tile([f"S{i:02d}" for i in range(n)], hours),
            "close": close.ravel(),
        }
    )

def _noise_panel(seed, hours=600):
    return _panel(0.01 * np.random.default_rng(seed).standard_normal((hours, N)))

# Test 1
def test_one_factor_panel_matches_formula():
    rng = np.random.default_rng(0)
    hours, s2 = 3000, 0.25
    loadings = rng.uniform(0.5, 1.5, N)
    factor = rng.standard_normal((hours, 1))
    noise = np.sqrt(s2) * rng.standard_normal((hours, N))
    log_returns = factor * loadings + noise
    ar = absorption_ratio(_panel(0.01 * log_returns))
    L = (loadings**2).sum()
    assert abs(ar.median() - (L + s2) / (L + N * s2)) < 0.02

# Test 2
def test_independent_returns_below_012():
    ar = absorption_ratio(_noise_panel(1, hours=1000))
    assert 0.05 < ar.median() < 0.12

# Test 3
def test_nan_warmup_and_missing_row():
    
    df = _noise_panel(2)
    by_hour = absorption_ratio(df).groupby(df["timestamp"]).first()
    assert by_hour.iloc[:WINDOW].isna().all()
    assert by_hour.iloc[WINDOW:].notna().all()

    m = 300
    ts = df["timestamp"].unique()
    cut = df[~((df["timestamp"] == ts[m]) & (df["symbol"] == "S03"))]
    by_hour = absorption_ratio(cut).groupby(cut["timestamp"]).first()
    assert by_hour.iloc[m : m + WINDOW + 1].isna().all()
    assert by_hour.iloc[WINDOW:m].notna().all()
    assert by_hour.iloc[m + WINDOW + 1 :].notna().all()

# Test 4
def test_same_value_per_timestamp_and_in_unit_interval():
    df = _noise_panel(3)
    ar = absorption_ratio(df)
    assert len(ar) == len(df) and ar.index.equals(df.index)
    assert (ar.groupby(df["timestamp"]).nunique(dropna=False) == 1).all()
    valid = ar.dropna()
    assert ((valid > 0.0) & (valid <= 1.0)).all()
    assert np.allclose(absorption_ratio(df, k=N).dropna(), 1.0)

#Test 5
def test_causal_when_later_rows_dropped():
    df = _noise_panel(4)
    full = absorption_ratio(df)
    head = df.iloc[: 400 * N]
    np.testing.assert_allclose(
        absorption_ratio(head).to_numpy(), full.iloc[: len(head)].to_numpy(), rtol=0, atol=1e-12
    )
