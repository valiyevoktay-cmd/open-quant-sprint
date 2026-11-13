
import numpy as np
from src.stats import adf_first_order


def engle_granger(y, x):
    """
    Two-step Engle-Granger test.

    Regresses y on a constant and x using OLS, then applies
    the first-order Dickey-Fuller test to the residuals.

    Returns:
        statistic: Dickey-Fuller test statistic.
        beta: OLS hedge ratio.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if y.ndim != 1 or x.ndim != 1 or y.size != x.size:
        raise ValueError("y and x must be 1D arrays of equal length.")

    valid = np.isfinite(y) & np.isfinite(x)
    y, x = y[valid], x[valid]

    if y.size < 25:
        raise ValueError("At least 25 valid observations are required.")

    design = np.column_stack((np.ones(y.size), x))
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]

    residuals = y - (alpha + beta * x)
    statistic, _ = adf_first_order(residuals)

    return float(statistic), float(beta)


def kalman_hedge_ratio(
    y,
    x,
    delta: float = 1e-5,
    obs_var: float = 1e-3,
    initial_covariance: float = 1e4,
):
    """
    Causal random-walk Kalman estimate of a time-varying
    intercept and hedge ratio.

    Returns NaN before the first valid observation.
    Missing observations after initialization carry forward
    the previous hedge ratio estimate.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if y.ndim != 1 or x.ndim != 1 or y.size != x.size:
        raise ValueError("y and x must be 1D arrays of equal length.")

    if delta < 0 or obs_var <= 0 or initial_covariance <= 0:
        raise ValueError(
            "delta must be >= 0; variances must be > 0."
        )

    n = y.size

    if n == 0:
        return np.empty(0)

    valid_x = np.flatnonzero(np.isfinite(x))

    if valid_x.size == 0:
        return np.full(n, np.nan)

    xc = x - x[valid_x[0]]

    theta = np.zeros(2)
    covariance = np.eye(2) * initial_covariance
    process_cov = np.eye(2) * delta
    identity = np.eye(2)

    beta = np.full(n, np.nan)
    initialized = False

    for t in range(n):
        covariance += process_cov

        if np.isfinite(y[t]) and np.isfinite(xc[t]):
            observation = np.array([1.0, xc[t]])

            innovation = y[t] - observation @ theta
            innovation_var = (
                observation @ covariance @ observation + obs_var
            )

            gain = covariance @ observation / innovation_var
            theta += gain * innovation

            correction = identity - np.outer(gain, observation)

            covariance = (
                correction @ covariance @ correction.T
                + obs_var * np.outer(gain, gain)
            )

            initialized = True

        if initialized:
            beta[t] = theta[1]

    return beta
