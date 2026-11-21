from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import f as f_dist

import pandas as pd

from src.contrib.hongyu import (
    block_bootstrap_mean,
    holm,
    hourly_spearman_ic,
    run_clock_study,
    same_hour_signal,
    seasonal_wald,
)

N_DAYS = 623
HOURS = 24


def _panel(seed: int, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """623 days of hourly N(0, 1) noise scaled by a daily volatility exp(0.5 * N(0, 1))."""
    rng = np.random.default_rng(seed)
    day_vol = np.exp(0.5 * rng.standard_normal(N_DAYS))
    x = (rng.standard_normal((N_DAYS, HOURS)) * day_vol[:, None]).ravel()
    hour = np.tile(np.arange(HOURS), N_DAYS)
    day = np.repeat(np.arange(N_DAYS), HOURS)
    week = day // 7
    if shift:
        x = x + shift * (hour == 13)
    return x, hour, day, week


def _ols_cluster_wald(x: np.ndarray, group: np.ndarray, cluster: np.ndarray) -> float:
    """Reference: OLS on group dummies, CR1-style sandwich with C / (C - 1)."""
    groups, g_idx = np.unique(group, return_inverse=True)
    clusters, c_idx = np.unique(cluster, return_inverse=True)
    design = np.zeros((len(x), len(groups)))
    design[np.arange(len(x)), g_idx] = 1.0
    beta, *_ = np.linalg.lstsq(design, x, rcond=None)
    resid = x - design @ beta
    bread = np.linalg.inv(design.T @ design)
    meat = np.zeros((len(groups), len(groups)))
    for c in range(len(clusters)):
        score = design[c_idx == c].T @ resid[c_idx == c]
        meat += np.outer(score, score)
    n_clusters = len(clusters)
    vcov = n_clusters / (n_clusters - 1) * bread @ meat @ bread
    restriction = np.hstack([np.eye(len(groups) - 1), -np.ones((len(groups) - 1, 1))])
    d = restriction @ beta
    return float(d @ np.linalg.inv(restriction @ vcov @ restriction.T) @ d)


def test_group_means_equal_plain_means() -> None:
    x, hour, day, _ = _panel(1)
    means, _, _ = seasonal_wald(x, hour, day)
    expected = np.array([x[hour == h].mean() for h in range(HOURS)])
    np.testing.assert_allclose(means, expected, rtol=0, atol=1e-12)


def test_wald_matches_ols_cluster_robust() -> None:
    x, hour, day, _ = _panel(2)
    _, wald, _ = seasonal_wald(x, hour, day)
    assert wald == pytest.approx(_ols_cluster_wald(x, hour, day), rel=1e-8)


def test_p_value_matches_f_formula() -> None:
    x, hour, _, week = _panel(3)
    _, wald, p_value = seasonal_wald(x, hour, week)
    q, n_clusters = HOURS - 1, len(np.unique(week))
    f_stat = wald * (n_clusters - q) / (q * (n_clusters - 1))
    assert p_value == pytest.approx(f_dist.sf(f_stat, q, n_clusters - q), rel=1e-12)


@pytest.mark.parametrize("cluster_kind", ["day", "week"])
def test_size_under_the_null(cluster_kind: str) -> None:
    rejections = 0
    for seed in range(500):
        x, hour, day, week = _panel(10_000 + seed)
        _, _, p_value = seasonal_wald(x, hour, day if cluster_kind == "day" else week)
        rejections += p_value < 0.05
    assert 0.02 <= rejections / 500 <= 0.08


def test_power_against_a_one_hour_shift() -> None:
    rejections = 0
    for seed in range(50):
        x, hour, day, _ = _panel(20_000 + seed, shift=0.4)
        _, _, p_value = seasonal_wald(x, hour, day)
        rejections += p_value < 0.05
    assert rejections / 50 >= 0.90


def test_nan_input_matches_dropping_rows() -> None:
    x, hour, day, _ = _panel(4)
    rng = np.random.default_rng(5)
    missing = rng.random(len(x)) < 0.05
    with_nan = x.copy()
    with_nan[missing] = np.nan
    means_a, wald_a, p_a = seasonal_wald(with_nan, hour, day)
    means_b, wald_b, p_b = seasonal_wald(x[~missing], hour[~missing], day[~missing])
    np.testing.assert_array_equal(means_a, means_b)
    assert wald_a == wald_b
    assert p_a == p_b


def test_holm_matches_hand_computation() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # Sorted: 0.005*4 = 0.02, 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> running max 0.06.
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(holm(np.array([0.6, 0.9])), [1.0, 1.0])


def test_block_bootstrap_matches_direct_resampling_and_p_value() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(1_003) + 0.3
    block, n_boot = 24, 2_000
    mean, lo, hi, p, p_centered = block_bootstrap_mean(x, block, n_boot=n_boot, seed=3)

    starts = np.random.default_rng(3).integers(0, len(x) - block + 1, size=(n_boot, 42))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, : len(x)]
    boot = x[idx].mean(axis=1)
    assert mean == pytest.approx(x.mean(), abs=1e-14)
    assert (lo, hi) == pytest.approx(tuple(np.quantile(boot, [0.025, 0.975])), abs=1e-12)
    expected_p = min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0)))
    expected_centered = (
        1 + np.count_nonzero(np.abs(boot - x.mean()) >= abs(x.mean()))
    ) / (1 + n_boot)
    assert p == pytest.approx(expected_p, abs=1e-15)
    assert p_centered == pytest.approx(expected_centered, abs=1e-15)
    assert p < 0.01 and 0 < lo < x.mean() < hi


def test_same_hour_signal_and_ic_on_a_planted_pattern() -> None:
    hours = pd.date_range("2025-01-01", periods=24 * 30, freq="h", tz="UTC")
    rng = np.random.default_rng(11)
    loading = rng.standard_normal(20)
    seasonal = np.sin(2 * np.pi * hours.hour.to_numpy() / 24)[:, None] * loading[None, :]
    returns = pd.DataFrame(seasonal + 0.01 * rng.standard_normal((len(hours), 20)), index=hours)

    signal = same_hour_signal(returns, k=5)
    t = 24 * 10 + 3
    expected = np.mean([returns.iloc[t + 1 - 24 * j] for j in range(1, 6)], axis=0)
    np.testing.assert_allclose(signal.iloc[t].to_numpy(), expected, atol=1e-14)
    assert signal.iloc[: 24 * 5 - 1].isna().all().all()

    ic = hourly_spearman_ic(signal, returns.shift(-1))
    assert ic.dropna().mean() > 0.9


def test_study_runs_and_finds_a_planted_hour_effect() -> None:
    hours = pd.date_range("2025-01-01", periods=24 * 210, freq="h", tz="UTC")
    rng = np.random.default_rng(13)
    log_ret = 0.01 * rng.standard_normal((len(hours), 20)) + 0.004 * (hours.hour == 13)[:, None]
    close = 100 * np.exp(np.cumsum(log_ret, axis=0))
    df = pd.DataFrame({
        "timestamp": np.repeat(hours, 20),
        "symbol": np.tile([f"S{i:02d}" for i in range(20)], len(hours)),
        "close": close.ravel(),
        "quote_volume": np.exp(10 + rng.standard_normal(close.size)),
    })
    result = run_clock_study(df)

    assert result["primary"]["reject_at_5pct"]
    assert len(result["family"]) == 22
    assert all(0 <= row["p"] <= row["p_holm"] <= 1 for row in result["family"])
    assert {row["k"] for row in result["continuation"]["full"]} == {5, 10, 20, 40}
