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


# ---------------------------------------------------------------------------
# Section 3. Comparison with src/contrib/jatin.py (PR #23).
#
# Black-box only: these tests call Jatin's two functions and never depend on
# how they are written. What the comparison found:
#
# * engle_granger: same t-stat (exactly, after the 4-decimal rounding in
#   adf_first_order) and the same beta to 1e-14, on synthetic data and on all
#   380 ordered pairs of the real panel.
# * kalman_hedge_ratio: two differences, both in the prior, both explained.
#   (a) his prior variance is 1e4 (``initial_covariance``), mine is 1.0;
#   (b) he adds ``delta`` once before the first update, I do not.
#   With ``init_var = 1e4 + delta`` my filter reproduces his to 1e-12.
#   With each side's defaults the betas differ a lot in the first hours and by
#   less than 5e-3 after hour 2,000.
# * No look-ahead in his filter: output is bit-identical when later rows are
#   dropped or changed.
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

try:
    from src.contrib import jatin
except ImportError:  # the comparison needs PR #23 in the checkout
    jatin = None

needs_jatin = pytest.mark.skipif(jatin is None, reason="src/contrib/jatin.py is not in this checkout")

TRAIN = Path("data/train_1h.parquet")
JATIN_INIT_VAR = 1e4  # default of his ``initial_covariance``
BURN_IN = 2_000


def matched_prior(y, x, delta: float = 1e-5, obs_var: float = 1e-3) -> np.ndarray:
    """My filter with Jatin's prior: variance 1e4, plus one ``delta`` before the first update."""
    return kalman_hedge_ratio(y, x, delta, obs_var, init_var=JATIN_INIT_VAR + delta)


@pytest.fixture(scope="module")
def log_close():
    if not TRAIN.exists():
        pytest.skip(f"{TRAIN} not found")
    pd = pytest.importorskip("pandas")
    df = pd.read_parquet(TRAIN, columns=["timestamp", "symbol", "close"])
    wide = df.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    return {symbol: np.log(wide[symbol].to_numpy(dtype=np.float64)) for symbol in wide.columns}


@needs_jatin
def test_compare_engle_granger_on_synthetic_pairs():
    rng = np.random.default_rng(0)
    pairs = [planted_pair(seed) for seed in range(20)] + [random_walk_pair(rng) for _ in range(200)]
    for y, x in pairs:
        t_his, beta_his = jatin.engle_granger(y, x)
        t_mine, beta_mine = engle_granger(y, x)
        assert t_his == t_mine
        assert beta_his == pytest.approx(beta_mine, abs=1e-12)


@needs_jatin
def test_compare_engle_granger_on_the_real_panel(log_close, capsys):
    symbols = sorted(log_close)
    n = len(log_close[symbols[0]])
    cut1, cut2 = int(n * 0.4), int(n * 0.7)
    windows = {"full": (0, n), "first 40%": (0, cut1), "next 30%": (cut1, cut2), "last 30%": (cut2, n)}

    lines = []
    for name, (lo, hi) in windows.items():
        worst_beta, passes = 0.0, 0
        for sy in symbols:
            for sx in symbols:
                if sy == sx:
                    continue
                y, x = log_close[sy][lo:hi], log_close[sx][lo:hi]
                t_his, beta_his = jatin.engle_granger(y, x)
                t_mine, beta_mine = engle_granger(y, x)
                assert t_his == t_mine, (name, sy, sx)
                worst_beta = max(worst_beta, abs(beta_his - beta_mine))
                passes += t_mine < EG_CRIT_5PCT
        assert worst_beta < 1e-12
        lines.append(f"{name}: {passes}/380 ordered pairs below {EG_CRIT_5PCT}, max |beta diff| {worst_beta:.1e}")
    with capsys.disabled():
        print("\n[darsh vs jatin] engle_granger, real panel, identical t-stats\n  " + "\n  ".join(lines))


@needs_jatin
@pytest.mark.parametrize("delta, obs_var", [(1e-5, 1e-3), (1e-4, 1e-2)])
def test_compare_kalman_with_matched_prior_on_synthetic(delta, obs_var):
    cases = [
        make_pair(seed=0, beta=1.5),
        make_pair(seed=0, beta=jump_beta()),
        make_pair(seed=0, beta=jump_beta(), level=11.0),
    ]
    for y, x in cases:
        his = jatin.kalman_hedge_ratio(y, x, delta, obs_var)
        np.testing.assert_allclose(matched_prior(y, x, delta, obs_var), his, rtol=0.0, atol=1e-10)
        # He centers x inside as well: passing x - x[0] changes nothing.
        np.testing.assert_array_equal(jatin.kalman_hedge_ratio(y, x - x[0], delta, obs_var), his)


@needs_jatin
def test_compare_kalman_on_the_real_panel(log_close, capsys):
    """19 pairs against BTC here; all 190 pairs give the same picture (figures in the PR)."""
    x = log_close["BTCUSDT"]
    matched, late, early = 0.0, 0.0, 0.0
    for symbol in sorted(log_close):
        if symbol == "BTCUSDT":
            continue
        y = log_close[symbol]
        his = jatin.kalman_hedge_ratio(y, x)
        mine = kalman_hedge_ratio(y, x)
        matched = max(matched, float(np.max(np.abs(matched_prior(y, x) - his))))
        late = max(late, float(np.max(np.abs(mine[BURN_IN:] - his[BURN_IN:]))))
        early = max(early, float(np.max(np.abs(mine[:200] - his[:200]))))
    with capsys.disabled():
        print(
            "\n[darsh vs jatin] kalman, 19 pairs vs BTC: "
            f"matched prior max |diff| {matched:.1e}; own defaults max |diff| "
            f"{early:.2f} in the first 200 h, {late:.1e} after hour {BURN_IN}"
        )
    assert matched < 1e-10
    assert late < 5e-3
    assert early > 0.1  # the priors do differ; if this fails, the note above is stale


@needs_jatin
def test_compare_kalman_recovery_time_after_a_jump():
    half = N_HOURS // 2
    for seed in range(3):
        y, x = make_pair(seed=seed, beta=jump_beta())
        his = hours_to_band(jatin.kalman_hedge_ratio(y, x), 0.8, start=half)
        mine = hours_to_band(kalman_hedge_ratio(y, x), 0.8, start=half)
        assert his == mine


@needs_jatin
def test_compare_kalman_with_missing_hours():
    y, x = make_pair(seed=3, beta=1.5)
    rng = np.random.default_rng(33)
    y_gap, x_gap = y.copy(), x.copy()
    y_gap[rng.choice(np.arange(10, N_HOURS), 300, replace=False)] = np.nan
    x_gap[rng.choice(np.arange(10, N_HOURS), 300, replace=False)] = np.nan
    y_gap[6_000:6_100] = np.nan

    his = jatin.kalman_hedge_ratio(y_gap, x_gap)
    idx = np.flatnonzero(~(np.isfinite(y_gap) & np.isfinite(x_gap)))
    assert np.all(np.isfinite(his))
    np.testing.assert_array_equal(his[idx], his[idx - 1])
    # Both add process noise through a gap; a filter that froze its covariance
    # there would differ by about 6e-3 right after the 100-hour outage.
    np.testing.assert_allclose(matched_prior(y_gap, x_gap), his, rtol=0.0, atol=1e-10)


@needs_jatin
def test_jatin_kalman_has_no_look_ahead(log_close):
    series = [make_pair(seed=1, beta=jump_beta())]
    if "ETHUSDT" in log_close and "BTCUSDT" in log_close:
        series.append((log_close["ETHUSDT"], log_close["BTCUSDT"]))

    for y, x in series:
        full = jatin.kalman_hedge_ratio(y, x)
        for cut in (1, 2, 50, 3_000, len(y) // 2 + 1, len(y) - 1):
            np.testing.assert_array_equal(jatin.kalman_hedge_ratio(y[:cut], x[:cut]), full[:cut])

        cut = 5_000
        y2, x2 = y.copy(), x.copy()
        y2[cut:] += 3.0
        x2[cut:] = np.nan
        np.testing.assert_array_equal(jatin.kalman_hedge_ratio(y2, x2)[:cut], full[:cut])
