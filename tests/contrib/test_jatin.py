import numpy as np
import pandas as pd
from src.contrib.jatin import (
    engle_granger,
    kalman_hedge_ratio,
    regression_details,
    residual_half_life,
    split_windows,
    determine_directions,
    run_formation,
    run_retests,
    create_summary,
    simulate_one_panel,
    run_null_simulations,
)


# ---------------------------------------------------------
# 1. Engle-Granger detects planted cointegration
# ---------------------------------------------------------

def test_engle_granger_detects_planted_cointegration():
    rng = np.random.default_rng(42)
    n = 2000

    x = np.cumsum(rng.normal(0, 0.01, n))

    eps = np.zeros(n)
    noise = rng.normal(0, 0.01, n)

    for t in range(1, n):
        eps[t] = 0.9 * eps[t - 1] + noise[t]

    y = 1.3 * x + eps

    statistic, beta = engle_granger(y, x)

    assert statistic < -3.90
    assert abs(beta - 1.3) < 0.05


# ---------------------------------------------------------
# 2. Kalman tracks a beta regime change
# ---------------------------------------------------------

def test_kalman_tracks_beta_regime_change():
    rng = np.random.default_rng(42)
    n = 14871
    mid = n // 2

    x = np.cumsum(rng.normal(0, 0.01, n))

    true_beta = np.full(n, 1.5)
    true_beta[mid:] = 0.8

    y = 0.2 + true_beta * x + rng.normal(0, 0.01, n)

    estimate = kalman_hedge_ratio(y, x)

    assert np.median(np.abs(estimate[-2000:] - 0.8)) < 0.05


# ---------------------------------------------------------
# 3. Kalman filter is causal
# ---------------------------------------------------------

def test_kalman_is_causal():
    rng = np.random.default_rng(7)

    x = np.cumsum(rng.normal(size=1000))
    y = 0.3 + 1.2 * x + rng.normal(scale=0.1, size=1000)

    full = kalman_hedge_ratio(y, x)
    prefix = kalman_hedge_ratio(y[:500], x[:500])

    np.testing.assert_allclose(full[:500], prefix)


# ---------------------------------------------------------
# 4. Kalman tracks a constant beta
# ---------------------------------------------------------

def test_kalman_tracks_constant_beta():
    rng = np.random.default_rng(123)

    n = 5000
    true_beta = 1.2

    x = np.cumsum(rng.normal(0, 0.01, n))
    y = 0.4 + true_beta * x + rng.normal(0, 0.01, n)

    estimate = kalman_hedge_ratio(y, x)

    assert np.median(np.abs(estimate[-1000:] - true_beta)) < 0.05


# ---------------------------------------------------------
# 5. Kalman handles missing observations
# ---------------------------------------------------------

def test_kalman_handles_nan_hours():
    rng = np.random.default_rng(42)

    n = 500
    x = np.cumsum(rng.normal(size=n))
    y = 0.3 + 1.2 * x + rng.normal(scale=0.1, size=n)

    # Missing observations before initialization
    x[:5] = np.nan
    y[:5] = np.nan

    # Missing observations after initialization
    x[50:55] = np.nan
    y[100:105] = np.nan

    estimate = kalman_hedge_ratio(y, x)

    # No estimate before the first valid observation
    assert np.isnan(estimate[:5]).all()

    # First valid observation initializes the filter
    assert np.isfinite(estimate[5])

    # Missing hours carry forward the previous estimate
    np.testing.assert_allclose(
        estimate[50:55],
        np.full(5, estimate[49]),
    )

    np.testing.assert_allclose(
        estimate[100:105],
        np.full(5, estimate[99]),
    )

    # Estimates after missing observations remain valid
    assert np.isfinite(estimate[105])


# ---------------------------------------------------------
# 6. Engle-Granger test has approximately correct size
# ---------------------------------------------------------

def test_engle_granger_size_under_independent_random_walks():
    rng = np.random.default_rng(123)

    n_simulations = 500
    n_observations = 1000
    critical_value = -3.34

    rejections = 0

    for _ in range(n_simulations):
        x = np.cumsum(rng.normal(size=n_observations))
        y = np.cumsum(rng.normal(size=n_observations))

        statistic, _ = engle_granger(y, x)

        if statistic < critical_value:
            rejections += 1

    rejection_rate = rejections / n_simulations

    # The expected empirical rejection rate is approximately 5%.
    assert 0.01 <= rejection_rate <= 0.08, (
        f"Empirical rejection rate was {rejection_rate:.2%}"
    )


# ---------------------------------------------------------
# 7. Regression diagnostics
# ---------------------------------------------------------

def test_regression_details_recovers_hedge_ratio():
    rng = np.random.default_rng(101)
    x = np.cumsum(rng.normal(size=500))
    y = 0.7 + 1.4 * x + rng.normal(scale=0.1, size=500)

    result = regression_details(y, x)

    assert abs(result["beta"] - 1.4) < 0.02
    assert result["n_obs"] == 500
    assert np.isfinite(result["t_stat"])


def test_regression_details_handles_missing_observations():
    rng = np.random.default_rng(102)
    x = np.cumsum(rng.normal(size=200))
    y = 0.5 + 0.8 * x + rng.normal(scale=0.1, size=200)

    x[10:15] = np.nan
    y[50:55] = np.nan

    result = regression_details(y, x)

    assert result["n_obs"] == 190


# ---------------------------------------------------------
# 8. Residual half-life
# ---------------------------------------------------------

def test_residual_half_life_for_stationary_ar1():
    rng = np.random.default_rng(103)
    phi_true = 0.8
    residuals = np.zeros(5000)

    noise = rng.normal(scale=0.1, size=len(residuals))
    for t in range(1, len(residuals)):
        residuals[t] = phi_true * residuals[t - 1] + noise[t]

    phi, half_life = residual_half_life(residuals)

    assert abs(phi - phi_true) < 0.03
    assert np.isfinite(half_life)
    assert half_life > 0


def test_residual_half_life_is_nan_for_nonmeanreverting_process():
    residuals = np.arange(100, dtype=float)
    phi, half_life = residual_half_life(residuals)

    assert phi >= 1.0
    assert np.isnan(half_life)


# ---------------------------------------------------------
# 9. Chronological windows and fixed directions
# ---------------------------------------------------------

def test_split_windows_preserves_chronological_order():
    index = pd.RangeIndex(100)
    prices = pd.DataFrame({"A": np.arange(100), "B": np.arange(100)}, index=index)
    volumes = pd.DataFrame({"A": np.ones(100), "B": np.ones(100)}, index=index)

    windows = split_windows(prices, volumes)

    assert len(windows["formation"][0]) == 40
    assert len(windows["retest1"][0]) == 30
    assert len(windows["retest2"][0]) == 30

    assert windows["formation"][0].index[-1] == 39
    assert windows["retest1"][0].index[0] == 40
    assert windows["retest2"][0].index[0] == 70


def test_directions_use_lower_volume_symbol_as_y():
    prices = pd.DataFrame({
        "A": np.arange(100, dtype=float),
        "B": np.arange(100, dtype=float) * 2,
        "C": np.arange(100, dtype=float) * 3,
    })
    volumes = pd.DataFrame({
        "A": np.full(100, 10.0),
        "B": np.full(100, 20.0),
        "C": np.full(100, 30.0),
    })

    directions = determine_directions(prices, volumes)

    assert len(directions) == 3
    pair = directions[
        (directions["symbol_y"] == "A")
        & (directions["symbol_x"] == "B")
    ]
    assert len(pair) == 1


# ---------------------------------------------------------
# 10. Null simulation
# ---------------------------------------------------------

def test_null_simulation_is_reproducible():
    symbols = ["A", "B", "C"]
    directions = pd.DataFrame([
        {"symbol_y": "A", "symbol_x": "B"},
        {"symbol_y": "A", "symbol_x": "C"},
        {"symbol_y": "B", "symbol_x": "C"},
    ])
    volatilities = pd.Series(
        {"A": 0.01, "B": 0.02, "C": 0.015}
    )

    kwargs = dict(
        n_simulations=5,
        seed=123,
        symbols=symbols,
        directions=directions,
        volatilities=volatilities,
        n_formation=200,
        n_retest1=150,
        n_retest2=150,
    )

    counts1, retests1 = run_null_simulations(**kwargs)
    counts2, retests2 = run_null_simulations(**kwargs)

    pd.testing.assert_frame_equal(counts1, counts2)
    pd.testing.assert_frame_equal(retests1, retests2)

    assert len(counts1) == 5
    assert set(counts1.columns) >= {
        "formation_passes",
        "retest1_passes",
        "retest2_passes",
        "passes_both_retests",
    }
