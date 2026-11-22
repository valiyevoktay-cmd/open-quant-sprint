import numpy as np
import pandas as pd
from src.contrib.jatin import (
    engle_granger,
    kalman_hedge_ratio,
    load_panel,
    split_windows,
    determine_directions,
    regression_details,
    residual_half_life,
    run_formation,
    run_retests,
    create_summary,
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
# 7. Chronological windows use the requested proportions
# ---------------------------------------------------------

def test_split_windows():
    index = pd.date_range("2025-01-01", periods=1000, freq="h")
    prices = pd.DataFrame({"A": np.arange(1000), "B": np.arange(1000) + 1}, index=index)
    volumes = pd.DataFrame({"A": np.ones(1000), "B": np.ones(1000) * 2}, index=index)

    windows = split_windows(prices, volumes)

    assert list(windows) == ["formation", "retest1", "retest2"]
    assert len(windows["formation"][0]) == 400
    assert len(windows["retest1"][0]) == 300
    assert len(windows["retest2"][0]) == 300
    assert windows["formation"][0].index.max() < windows["retest1"][0].index.min()
    assert windows["retest1"][0].index.max() < windows["retest2"][0].index.min()


# ---------------------------------------------------------
# 8. Pair direction follows average formation quote volume
# ---------------------------------------------------------

def test_determine_directions_uses_lower_volume_as_y():
    prices = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [2.0, 3.0], "C": [3.0, 4.0]}
    )
    volumes = pd.DataFrame(
        {"A": [10.0, 10.0], "B": [20.0, 20.0], "C": [30.0, 30.0]}
    )

    directions = determine_directions(prices, volumes)

    assert len(directions) == 3
    pair_ab = directions[
        (directions["symbol_y"] == "A") & (directions["symbol_x"] == "B")
    ]
    assert len(pair_ab) == 1
    assert pair_ab.iloc[0]["avg_quote_volume_y"] < pair_ab.iloc[0]["avg_quote_volume_x"]


# ---------------------------------------------------------
# 9. Regression returns OLS parameters and residual test statistic
# ---------------------------------------------------------

def test_regression_details_recovers_ols_parameters():
    rng = np.random.default_rng(2026)
    x = np.cumsum(rng.normal(size=500))
    noise = np.zeros(500)
    innovations = rng.normal(scale=0.1, size=500)
    for t in range(1, 500):
        noise[t] = 0.5 * noise[t - 1] + innovations[t]
    y = 0.7 + 1.4 * x + noise

    result = regression_details(y, x)

    assert abs(result["alpha"] - 0.7) < 0.05
    assert abs(result["beta"] - 1.4) < 0.01
    assert np.isfinite(result["t_stat"])
    assert result["n_obs"] == 500
    assert len(result["residuals"]) == 500
    assert "pvalue" not in result


# ---------------------------------------------------------
# 10. Stationary AR(1) residuals have a finite half-life
# ---------------------------------------------------------

def test_residual_half_life_is_finite_for_stationary_process():
    rng = np.random.default_rng(17)
    residuals = np.zeros(5000)
    innovations = rng.normal(size=5000)

    for t in range(1, len(residuals)):
        residuals[t] = 0.8 * residuals[t - 1] + innovations[t]

    phi, half_life = residual_half_life(residuals)

    assert 0 < phi < 1
    assert np.isfinite(half_life)
    assert half_life > 0


# ---------------------------------------------------------
# 11. Non-mean-reverting residuals must not get finite half-life
# ---------------------------------------------------------

def test_residual_half_life_is_nan_for_nonmeanreverting_process():
    residuals = np.arange(100, dtype=float)

    phi, half_life = residual_half_life(residuals)

    assert phi >= 1 - 1e-9
    assert np.isnan(half_life)


# ---------------------------------------------------------
# 12. Formation evaluates every directed pair and applies threshold
# ---------------------------------------------------------

def test_run_formation_applies_critical_value(monkeypatch):
    prices = pd.DataFrame(
        {"A": np.linspace(1.0, 2.0, 30), "B": np.linspace(2.0, 4.0, 30)}
    )
    directions = pd.DataFrame(
        [{"symbol_y": "A", "symbol_x": "B"},
         {"symbol_y": "B", "symbol_x": "A"}]
    )

    calls = iter([-3.8, -2.5])

    def fake_regression(y, x):
        return {
            "alpha": 0.1,
            "beta": 1.2,
            "t_stat": next(calls),
            "residuals": np.random.default_rng(1).normal(size=30),
            "n_obs": 30,
        }

    monkeypatch.setitem(run_formation.__globals__, "regression_details", fake_regression)
    monkeypatch.setitem(run_formation.__globals__, "residual_half_life", lambda residuals: (0.8, 3.1))

    result = run_formation(prices, directions)

    assert len(result) == 2
    assert result["cointegrated"].tolist() == [True, False]
    assert "pvalue" not in result.columns


# ---------------------------------------------------------
# 13. Retests include only formation survivors
# ---------------------------------------------------------

def test_run_retests_only_retests_formation_survivors(monkeypatch):
    index = pd.date_range("2025-01-01", periods=60, freq="h")
    prices1 = pd.DataFrame({"A": np.arange(30.0), "B": np.arange(30.0) * 2}, index=index[:30])
    prices2 = pd.DataFrame({"A": np.arange(30.0), "B": np.arange(30.0) * 2}, index=index[30:])
    windows = {"retest1": (prices1, None), "retest2": (prices2, None)}
    formation = pd.DataFrame([
        {"symbol_y": "A", "symbol_x": "B", "beta": 2.0, "cointegrated": True},
        {"symbol_y": "B", "symbol_x": "A", "beta": 0.5, "cointegrated": False},
    ])

    def fake_regression(y, x):
        return {
            "alpha": 0.0,
            "beta": 2.1,
            "t_stat": -3.6,
            "residuals": np.random.default_rng(2).normal(size=30),
            "n_obs": 30,
        }

    monkeypatch.setitem(run_retests.__globals__, "regression_details", fake_regression)
    monkeypatch.setitem(run_retests.__globals__, "residual_half_life", lambda residuals: (0.8, 3.1))

    result = run_retests(windows, formation)

    assert len(result) == 2
    assert set(result["window"]) == {"retest1", "retest2"}
    assert set(zip(result["symbol_y"], result["symbol_x"])) == {("A", "B")}
    assert "pvalue" not in result.columns


# ---------------------------------------------------------
# 14. Summary counts formation and retest passes correctly
# ---------------------------------------------------------

def test_create_summary_counts_passes():
    formation = pd.DataFrame([
        {"symbol_y": "A", "symbol_x": "B", "cointegrated": True},
        {"symbol_y": "A", "symbol_x": "C", "cointegrated": True},
        {"symbol_y": "B", "symbol_x": "C", "cointegrated": False},
    ])
    retests = pd.DataFrame([
        {"symbol_y": "A", "symbol_x": "B", "window": "retest1", "cointegrated": True},
        {"symbol_y": "A", "symbol_x": "C", "window": "retest1", "cointegrated": False},
        {"symbol_y": "A", "symbol_x": "B", "window": "retest2", "cointegrated": True},
        {"symbol_y": "A", "symbol_x": "C", "window": "retest2", "cointegrated": True},
    ])

    summary = create_summary(formation, retests).iloc[0]

    assert summary["formation_pairs_tested"] == 3