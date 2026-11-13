import numpy as np

from crypto_cointegration.models import engle_granger, kalman_hedge_ratio


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


def test_kalman_is_causal():
    rng = np.random.default_rng(7)
    x = np.cumsum(rng.normal(size=1000))
    y = 0.3 + 1.2 * x + rng.normal(scale=0.1, size=1000)
    full = kalman_hedge_ratio(y, x)
    prefix = kalman_hedge_ratio(y[:500], x[:500])
    np.testing.assert_allclose(full[:500], prefix)
