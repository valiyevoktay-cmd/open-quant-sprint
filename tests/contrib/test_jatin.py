import numpy as np
from src.contrib.jatin import engle_granger, kalman_hedge_ratio


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
