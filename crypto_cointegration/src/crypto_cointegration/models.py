import numpy as np
from statsmodels.tsa.stattools import adfuller


def engle_granger(y, x, maxlag: int = 1):
    """Two-step Engle-Granger test; returns residual ADF statistic and OLS beta.

    The residual ADF regression has no deterministic term and uses a fixed lag.
    Use Engle-Granger/MacKinnon critical values, not the ordinary ADF p-value.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 1 or y.size != x.size:
        raise ValueError("y and x must be 1D arrays of equal length.")

    valid = np.isfinite(y) & np.isfinite(x)
    y, x = y[valid], x[valid]
    if y.size <= maxlag + 3:
        raise ValueError("Insufficient valid observations for the requested ADF lag.")

    design = np.column_stack((np.ones(y.size), x))
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - (alpha + beta * x)
    statistic = adfuller(
        residuals, maxlag=maxlag, regression="n", autolag=None
    )[0]
    return float(statistic), float(beta)


def kalman_hedge_ratio(
    y, x, delta: float = 1e-5, obs_var: float = 1e-3,
    initial_covariance: float = 1e4,
):
    """Causal random-walk Kalman estimate of time-varying intercept and beta."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 1 or y.size != x.size:
        raise ValueError("y and x must be 1D arrays of equal length.")
    if delta < 0 or obs_var <= 0 or initial_covariance <= 0:
        raise ValueError("delta must be >= 0; variances must be > 0.")
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
    beta = np.empty(n)

    for t in range(n):
        covariance += process_cov
        if np.isfinite(y[t]) and np.isfinite(xc[t]):
            observation = np.array([1.0, xc[t]])
            innovation = y[t] - observation @ theta
            innovation_var = observation @ covariance @ observation + obs_var
            gain = covariance @ observation / innovation_var
            theta += gain * innovation
            correction = identity - np.outer(gain, observation)
            covariance = (
                correction @ covariance @ correction.T
                + obs_var * np.outer(gain, gain)
            )
        beta[t] = theta[1]
    return beta
