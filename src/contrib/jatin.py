
from itertools import combinations

import numpy as np
import pandas as pd
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



# ---------------------------------------------------------
# Part 2: Out-of-sample cointegration measurement
# ---------------------------------------------------------

CRITICAL_VALUE = -3.34


def regression_details(y, x):
    """OLS regression and residual first-order Dickey-Fuller statistic."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if y.ndim != 1 or x.ndim != 1 or len(y) != len(x):
        raise ValueError("y and x must be equal-length 1D arrays.")

    valid = np.isfinite(y) & np.isfinite(x)
    y, x = y[valid], x[valid]

    if len(y) < 25:
        raise ValueError("At least 25 valid observations are required.")

    design = np.column_stack([np.ones(len(x)), x])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - (alpha + beta * x)

    statistic, pvalue = adf_first_order(residuals)

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "t_stat": float(statistic),
        "pvalue": float(pvalue),
        "residuals": residuals,
        "n_obs": len(y),
    }


def residual_half_life(residuals):
    """Estimate AR(1) persistence and half-life of regression residuals."""
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) < 25:
        return np.nan, np.nan

    lagged = residuals[:-1]
    current = residuals[1:]

    design = np.column_stack([np.ones(len(lagged)), lagged])
    _, phi = np.linalg.lstsq(design, current, rcond=None)[0]

    if 0 < phi < 1:
        half_life = -np.log(2) / np.log(phi)
    else:
        half_life = np.nan

    return float(phi), float(half_life)


def split_windows(prices, volumes, formation_fraction=0.40,
                  retest1_fraction=0.30):
    """Chronologically split price and volume panels into 40/30/30 windows."""
    if not prices.index.equals(volumes.index):
        raise ValueError("Price and volume indices must match.")
    if not prices.columns.equals(volumes.columns):
        raise ValueError("Price and volume columns must match.")

    if not 0 < formation_fraction < 1:
        raise ValueError("formation_fraction must be between 0 and 1.")
    if not 0 < retest1_fraction < 1:
        raise ValueError("retest1_fraction must be between 0 and 1.")
    if formation_fraction + retest1_fraction >= 1:
        raise ValueError("The first two windows must total less than 1.")

    n = len(prices)
    first_end = int(n * formation_fraction)
    second_end = first_end + int(n * retest1_fraction)

    if first_end < 25 or second_end - first_end < 25 or n - second_end < 25:
        raise ValueError("Each window must contain at least 25 observations.")

    return {
        "formation": (prices.iloc[:first_end], volumes.iloc[:first_end]),
        "retest1": (prices.iloc[first_end:second_end],
                    volumes.iloc[first_end:second_end]),
        "retest2": (prices.iloc[second_end:], volumes.iloc[second_end:]),
    }


def determine_directions(formation_prices, formation_volumes):
    """
    Fix pair direction using formation-window average quote volume.

    The lower-volume symbol is y; the higher-volume symbol is x.
    """
    if not formation_prices.columns.equals(formation_volumes.columns):
        raise ValueError("Price and volume symbols must match.")

    symbols = sorted(formation_prices.columns)
    average_volume = formation_volumes.mean(axis=0)

    if average_volume.isna().any():
        raise ValueError("Missing average formation quote volume.")

    rows = []
    for a, b in combinations(symbols, 2):
        if average_volume[a] == average_volume[b]:
            raise ValueError(f"Cannot resolve equal-volume pair: {a}, {b}")

        y_symbol, x_symbol = (
            (a, b) if average_volume[a] < average_volume[b] else (b, a)
        )

        rows.append({
            "symbol_y": y_symbol,
            "symbol_x": x_symbol,
            "avg_quote_volume_y": float(average_volume[y_symbol]),
            "avg_quote_volume_x": float(average_volume[x_symbol]),
        })

    return pd.DataFrame(rows)


def run_formation(formation_prices, directions,
                  critical_value=CRITICAL_VALUE):
    """Test every directed pair in the formation window."""
    rows = []

    for pair in directions.itertuples(index=False):
        try:
            result = regression_details(
                formation_prices[pair.symbol_y].to_numpy(),
                formation_prices[pair.symbol_x].to_numpy(),
            )
        except ValueError:
            continue

        phi, half_life = residual_half_life(result["residuals"])

        rows.append({
            "symbol_y": pair.symbol_y,
            "symbol_x": pair.symbol_x,
            "n_obs": result["n_obs"],
            "alpha": result["alpha"],
            "beta": result["beta"],
            "t_stat": result["t_stat"],
            "pvalue": result["pvalue"],
            "phi": phi,
            "half_life_hours": half_life,
            "cointegrated": result["t_stat"] < critical_value,
        })

    return pd.DataFrame(rows)


def run_retests(windows, formation_results,
                critical_value=CRITICAL_VALUE):
    """Retest only formation survivors in the two later windows."""
    selected = formation_results.loc[formation_results["cointegrated"]]
    rows = []

    for pair in selected.itertuples(index=False):
        for window_name in ("retest1", "retest2"):
            prices, _ = windows[window_name]

            try:
                result = regression_details(
                    prices[pair.symbol_y].to_numpy(),
                    prices[pair.symbol_x].to_numpy(),
                )
            except ValueError:
                continue

            phi, half_life = residual_half_life(result["residuals"])
            beta_change = result["beta"] - pair.beta

            rows.append({
                "symbol_y": pair.symbol_y,
                "symbol_x": pair.symbol_x,
                "window": window_name,
                "n_obs": result["n_obs"],
                "alpha": result["alpha"],
                "beta_formation": pair.beta,
                "beta_retest": result["beta"],
                "absolute_beta_change": abs(beta_change),
                "relative_beta_change": (
                    abs(beta_change) / abs(pair.beta)
                    if abs(pair.beta) > 1e-10 else np.nan
                ),
                "t_stat": result["t_stat"],
                "pvalue": result["pvalue"],
                "phi": phi,
                "half_life_hours": half_life,
                "cointegrated": result["t_stat"] < critical_value,
            })

    return pd.DataFrame(rows)


def create_summary(formation, retests):
    """Return the formation and out-of-sample pass counts."""
    formation_passes = int(formation["cointegrated"].sum())

    if retests.empty:
        return {
            "formation_pairs_tested": len(formation),
            "formation_passes": formation_passes,
            "retest1_passes": 0,
            "retest2_passes": 0,
            "passes_both_retests": 0,
        }

    retest1 = retests[retests["window"] == "retest1"]
    retest2 = retests[retests["window"] == "retest2"]

    passed1 = set(zip(
        retest1.loc[retest1["cointegrated"], "symbol_y"],
        retest1.loc[retest1["cointegrated"], "symbol_x"],
    ))
    passed2 = set(zip(
        retest2.loc[retest2["cointegrated"], "symbol_y"],
        retest2.loc[retest2["cointegrated"], "symbol_x"],
    ))

    return {
        "formation_pairs_tested": len(formation),
        "formation_passes": formation_passes,
        "retest1_passes": int(retest1["cointegrated"].sum()),
        "retest2_passes": int(retest2["cointegrated"].sum()),
        "passes_both_retests": len(passed1 & passed2),
    }


def simulate_one_panel(rng, symbols, directions, volatilities,
                       n_formation, n_retest1, n_retest2,
                       critical_value=CRITICAL_VALUE):
    """
    Simulate one panel of independent Gaussian random walks.

    Each symbol's innovation volatility is supplied by the caller.
    Returns aggregate pass counts and selected-pair retest diagnostics.
    """
    n_total = n_formation + n_retest1 + n_retest2
    sigma = np.asarray(volatilities.reindex(symbols), dtype=float)

    if len(sigma) != len(symbols) or not np.isfinite(sigma).all():
        raise ValueError("Invalid innovation volatilities.")
    if (sigma <= 0).any():
        raise ValueError("Innovation volatilities must be positive.")

    increments = rng.normal(
        loc=0.0,
        scale=sigma,
        size=(n_total - 1, len(symbols)),
    )
    walks = np.vstack([
        np.zeros((1, len(symbols))),
        np.cumsum(increments, axis=0),
    ])

    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
    bounds = {
        "formation": (0, n_formation),
        "retest1": (n_formation, n_formation + n_retest1),
        "retest2": (n_formation + n_retest1, n_total),
    }

    formation_rows = []
    retest_rows = []

    for pair in directions.itertuples(index=False):
        y = walks[:, symbol_index[pair.symbol_y]]
        x = walks[:, symbol_index[pair.symbol_x]]
        start, end = bounds["formation"]

        formation = regression_details(y[start:end], x[start:end])
        passed = formation["t_stat"] < critical_value

        formation_rows.append({
            "symbol_y": pair.symbol_y,
            "symbol_x": pair.symbol_x,
            "beta": formation["beta"],
            "cointegrated": passed,
        })

        if not passed:
            continue

        for window_name in ("retest1", "retest2"):
            start, end = bounds[window_name]
            retest = regression_details(y[start:end], x[start:end])
            phi, half_life = residual_half_life(retest["residuals"])
            beta_change = abs(retest["beta"] - formation["beta"])

            retest_rows.append({
                "symbol_y": pair.symbol_y,
                "symbol_x": pair.symbol_x,
                "window": window_name,
                "beta_formation": formation["beta"],
                "beta_retest": retest["beta"],
                "absolute_beta_change": beta_change,
                "relative_beta_change": (
                    beta_change / abs(formation["beta"])
                    if abs(formation["beta"]) > 1e-12 else np.nan
                ),
                "t_stat": retest["t_stat"],
                "phi": phi,
                "half_life_hours": half_life,
                "cointegrated": retest["t_stat"] < critical_value,
            })

    formation_df = pd.DataFrame(formation_rows)
    retest_df = pd.DataFrame(retest_rows)
    summary = create_summary(formation_df, retest_df)

    return summary, retest_df


def run_null_simulations(n_simulations, seed, symbols, directions,
                         volatilities, n_formation, n_retest1, n_retest2,
                         critical_value=CRITICAL_VALUE):
    """Run repeated independent-random-walk panels for the null baseline."""
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive.")

    rng = np.random.default_rng(seed)
    count_rows = []
    retest_frames = []

    for simulation in range(n_simulations):
        summary, retests = simulate_one_panel(
            rng, symbols, directions, volatilities,
            n_formation, n_retest1, n_retest2, critical_value,
        )
        summary["simulation"] = simulation
        count_rows.append(summary)

        if not retests.empty:
            retests = retests.copy()
            retests["simulation"] = simulation
            retest_frames.append(retests)

    counts = pd.DataFrame(count_rows)
    selected_retests = (
        pd.concat(retest_frames, ignore_index=True)
        if retest_frames else pd.DataFrame()
    )

    return counts, selected_retests
