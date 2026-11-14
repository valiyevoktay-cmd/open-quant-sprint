"""Track 09 (#15), independent replication of Part 1.

Section 1 holds my own ``kalman_hedge_ratio`` and ``engle_granger``. They were
written from Issue #15 alone, before reading ``src/contrib/jatin.py``.
Section 2 runs the Issue's checks against my version.

Choices the Issue leaves open, and what I picked:

* Prior. State starts at (0, 0) with covariance ``init_var * I`` (default 1.0,
  which is diffuse next to ``obs_var = 1e-3``). The prior applies at the first
  valid observation; no process noise is added before it.
* Centering. ``x`` is centered on its first finite value inside the function
  (``center=True``), as the Issue recommends. That value is known at the first
  update, so centering does not look ahead.
* Missing hours. The update is skipped and beta is carried forward, but the
  predict step still runs: the state is a random walk, so its covariance grows
  by ``delta`` per missing hour and the filter re-learns faster after a gap.
* Before the first valid observation there is no estimate: the output is NaN.
* Covariance update in Joseph form, so it stays symmetric and positive.
* ``engle_granger`` drops rows where either series is not finite, fits OLS on
  demeaned data and returns NaN for both values below 25 paired rows or when
  ``x`` is constant. NaN never passes a ``t < cutoff`` check.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.stats import adf_first_order

# ---------------------------------------------------------------------------
# Section 1. My version, written from the Issue.
# ---------------------------------------------------------------------------

EG_CRIT_1PCT = -3.90
EG_CRIT_5PCT = -3.34
EG_CRIT_10PCT = -3.04
MIN_PAIRED_ROWS = 25  # same floor as src.stats.adf_first_order


def kalman_hedge_ratio(
    y: np.ndarray,
    x: np.ndarray,
    delta: float = 1e-5,
    obs_var: float = 1e-3,
    *,
    center: bool = True,
    init_var: float = 1.0,
) -> np.ndarray:
    """Filtered beta of ``y_t = intercept_t + beta_t * x_t + noise``.

    State (intercept, beta) is a random walk with variance ``delta`` per hour
    on each component; the observation noise has variance ``obs_var``.
    ``out[t]`` uses rows ``0..t`` only.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("y and x must be 1-D arrays of the same length")

    n = y.shape[0]
    out = np.full(n, np.nan)

    a = 0.0  # intercept
    b = 0.0  # beta
    paa, pab, pbb = init_var, 0.0, init_var
    x_ref = 0.0
    started = False

    for t in range(n):
        if started:
            # Predict: random-walk state, so only the covariance moves.
            paa += delta
            pbb += delta

        yt = y[t]
        xt = x[t]
        if math.isfinite(yt) and math.isfinite(xt):
            if not started:
                x_ref = xt if center else 0.0
                started = True
            h = xt - x_ref

            err = yt - a - b * h
            ga = paa + pab * h  # (P H')[0]
            gb = pab + pbb * h  # (P H')[1]
            s = ga + gb * h + obs_var
            ka = ga / s
            kb = gb / s

            a += ka * err
            b += kb * err

            # Joseph form: P = (I - K H) P (I - K H)' + K R K'
            a00 = 1.0 - ka
            a01 = -ka * h
            a10 = -kb
            a11 = 1.0 - kb * h
            m00 = a00 * paa + a01 * pab
            m01 = a00 * pab + a01 * pbb
            m10 = a10 * paa + a11 * pab
            m11 = a10 * pab + a11 * pbb
            paa = m00 * a00 + m01 * a01 + obs_var * ka * ka
            pab = m00 * a10 + m01 * a11 + obs_var * ka * kb
            pbb = m10 * a10 + m11 * a11 + obs_var * kb * kb

        if started:
            out[t] = b

    return out


def engle_granger(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS of ``y`` on ``x`` with a constant, then a first-order DF test on the residual.

    Returns (t-stat, OLS beta). Compare the t-stat with the Engle-Granger
    critical values (``EG_CRIT_*``), not with the p-value of ``adf_first_order``.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("y and x must be 1-D arrays of the same length")

    ok = np.isfinite(y) & np.isfinite(x)
    if int(ok.sum()) < MIN_PAIRED_ROWS:
        return float("nan"), float("nan")

    yc = y[ok] - y[ok].mean()
    xc = x[ok] - x[ok].mean()
    sxx = float(xc @ xc)
    if sxx <= 0.0:
        return float("nan"), float("nan")

    beta = float(xc @ yc) / sxx
    resid = yc - beta * xc
    t_stat, _ = adf_first_order(resid)
    return float(t_stat), beta


# ---------------------------------------------------------------------------
# Section 2. The Issue's checks, on my version.
# ---------------------------------------------------------------------------

N_HOURS = 14_871
STEP_SD = 0.01
NOISE_SD = 0.01
BAND = 0.05


def make_pair(seed: int, beta, n: int = N_HOURS, level: float = 0.0):
    """x: random walk with N(0, 0.01^2) steps; y = 0.2 + beta * x + N(0, 0.01^2)."""
    rng = np.random.default_rng(seed)
    x = level + np.cumsum(rng.normal(0.0, STEP_SD, n))
    y = 0.2 + beta * x + rng.normal(0.0, NOISE_SD, n)
    return y, x


def jump_beta(n: int = N_HOURS) -> np.ndarray:
    beta = np.full(n, 1.5)
    beta[n // 2 :] = 0.8
    return beta


def hours_to_band(est: np.ndarray, target: float, start: int, band: float = BAND) -> int:
    """Hours after ``start`` until the estimate is inside the band and stays there."""
    outside = np.flatnonzero(np.abs(est[start:] - target) >= band)
    return 0 if outside.size == 0 else int(outside[-1]) + 1


def random_walk_pair(rng: np.random.Generator, n: int = 2_000):
    return np.cumsum(rng.normal(0.0, STEP_SD, n)), np.cumsum(rng.normal(0.0, STEP_SD, n))


def planted_pair(seed: int, n: int = 2_000, beta: float = 1.3, phi: float = 0.9):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, STEP_SD, n))
    eps = rng.normal(0.0, NOISE_SD, n)
    u = np.empty(n)
    u[0] = eps[0]
    for t in range(1, n):
        u[t] = phi * u[t - 1] + eps[t]
    return beta * x + u, x


def test_constant_beta_stays_in_band_after_burn_in():
    y, x = make_pair(seed=0, beta=1.5)
    est = kalman_hedge_ratio(y, x)
    assert est.shape == (N_HOURS,)
    assert np.all(np.isfinite(est))
    assert np.max(np.abs(est[2_000:] - 1.5)) < BAND


def test_jump_beta_recovers(capsys):
    y, x = make_pair(seed=0, beta=jump_beta())
    est = kalman_hedge_ratio(y, x)
    half = N_HOURS // 2

    hours = hours_to_band(est, 0.8, start=half)
    with capsys.disabled():
        print(f"\n[darsh] jump 1.5 -> 0.8 at hour {half}: inside +/-{BAND} for good after {hours} h")

    assert np.median(np.abs(est[-2_000:] - 0.8)) < BAND
    assert hours < N_HOURS - half - 2_000


@pytest.mark.parametrize("cut", [1, 2, 50, 3_000, N_HOURS // 2 + 1, N_HOURS - 1])
def test_kalman_is_causal_under_truncation(cut):
    y, x = make_pair(seed=1, beta=jump_beta())
    full = kalman_hedge_ratio(y, x)
    np.testing.assert_array_equal(kalman_hedge_ratio(y[:cut], x[:cut]), full[:cut])


def test_kalman_ignores_future_values():
    y, x = make_pair(seed=2, beta=1.5)
    full = kalman_hedge_ratio(y, x)
    cut = 5_000
    y2, x2 = y.copy(), x.copy()
    y2[cut:] += 3.0
    x2[cut:] = np.nan
    np.testing.assert_array_equal(kalman_hedge_ratio(y2, x2)[:cut], full[:cut])


def test_nan_hours_skip_the_update_and_carry_forward():
    y, x = make_pair(seed=3, beta=1.5)
    rng = np.random.default_rng(33)
    y_gap, x_gap = y.copy(), x.copy()
    y_gap[rng.choice(np.arange(10, N_HOURS), 300, replace=False)] = np.nan
    x_gap[rng.choice(np.arange(10, N_HOURS), 300, replace=False)] = np.nan
    y_gap[6_000:6_100] = np.nan  # a long outage

    est = kalman_hedge_ratio(y_gap, x_gap)
    missing = ~(np.isfinite(y_gap) & np.isfinite(x_gap))

    assert np.all(np.isfinite(est))
    idx = np.flatnonzero(missing)
    np.testing.assert_array_equal(est[idx], est[idx - 1])
    assert np.max(np.abs(est[2_000:] - 1.5)) < BAND


def test_leading_nan_gives_nan_then_matches_the_trimmed_series():
    y, x = make_pair(seed=4, beta=1.5, n=4_000)
    y_lead = np.concatenate([np.full(5, np.nan), y])
    x_lead = np.concatenate([x[:5], x])  # x present, y missing: still no update

    est = kalman_hedge_ratio(y_lead, x_lead)
    assert np.all(np.isnan(est[:5]))
    np.testing.assert_array_equal(est[5:], kalman_hedge_ratio(y, x))


def test_all_nan_and_empty_inputs():
    assert kalman_hedge_ratio(np.array([]), np.array([])).shape == (0,)
    assert np.all(np.isnan(kalman_hedge_ratio(np.full(10, np.nan), np.zeros(10))))
    t, b = engle_granger(np.full(100, np.nan), np.zeros(100))
    assert math.isnan(t) and math.isnan(b)


def test_centering_matters_at_log_price_level(capsys):
    """The Issue's warning: at a log price near 11 the raw-x filter adapts worse.

    At that level a beta jump of -0.7 also moves the intercept by about -7.7,
    which a random walk with variance 1e-5 per hour absorbs slowly. Single seeds
    go either way (raw was worse in 34 of 40), so this compares medians.
    """
    err_centered, err_raw = [], []
    for seed in range(12):
        y, x = make_pair(seed=seed, beta=jump_beta(), level=11.0)
        err_centered.append(np.median(np.abs(kalman_hedge_ratio(y, x)[-2_000:] - 0.8)))
        err_raw.append(np.median(np.abs(kalman_hedge_ratio(y, x, center=False)[-2_000:] - 0.8)))
    with capsys.disabled():
        print(
            "\n[darsh] level 11, 12 seeds, median |err| over the last 2000 h: "
            f"centered {np.median(err_centered):.4f} (max {max(err_centered):.4f}), "
            f"raw {np.median(err_raw):.4f} (max {max(err_raw):.4f})"
        )
    assert np.median(err_centered) < np.median(err_raw)

    # A constant shift of x changes nothing once x is centered inside.
    y, x = make_pair(seed=0, beta=1.5)
    np.testing.assert_allclose(kalman_hedge_ratio(y, x + 11.0), kalman_hedge_ratio(y, x), atol=1e-9)


def test_engle_granger_finds_a_planted_pair():
    y, x = planted_pair(seed=0)
    t_stat, beta = engle_granger(y, x)
    assert t_stat < EG_CRIT_1PCT
    assert abs(beta - 1.3) < BAND


def test_engle_granger_matches_lstsq_and_handles_nan():
    y, x = planted_pair(seed=1)
    design = np.column_stack([np.ones_like(x), x + 11.0])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    t_stat, beta = engle_granger(y, x + 11.0)
    assert beta == pytest.approx(coef[1], abs=1e-10)

    y_gap = y.copy()
    y_gap[[3, 500, 1_999]] = np.nan
    keep = np.isfinite(y_gap)
    assert engle_granger(y_gap, x) == engle_granger(y[keep], x[keep])


def test_engle_granger_size_under_the_null(capsys):
    rng = np.random.default_rng(0)
    t_stats = np.array([engle_granger(*random_walk_pair(rng))[0] for _ in range(200)])
    rate = float(np.mean(t_stats < EG_CRIT_5PCT))
    with capsys.disabled():
        print(f"\n[darsh] null rejection rate at {EG_CRIT_5PCT}: {rate:.3f} (200 pairs)")
    assert 0.02 <= rate <= 0.09
