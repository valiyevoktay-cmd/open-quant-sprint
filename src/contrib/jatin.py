
from itertools import combinations
from pathlib import Path
import json

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
# Configuration: preregistered methodology
# ---------------------------------------------------------

DATA_PATH = Path("data/train_1h.parquet")
OUTPUT_DIR = Path("outputs/part2")

CRITICAL_VALUE = -3.34

FORMATION_FRACTION = 0.40
RETEST1_FRACTION = 0.30
RETEST2_FRACTION = 0.30


# ---------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------

def load_panel(path):
    required = {
        "timestamp",
        "symbol",
        "close",
        "quote_volume",
    }

    raw = pd.read_parquet(path)

    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    raw = raw[list(required)].copy()

    raw["timestamp"] = pd.to_datetime(
        raw["timestamp"], utc=True
    )
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw["quote_volume"] = pd.to_numeric(
        raw["quote_volume"], errors="coerce"
    )

    if raw.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("Duplicate timestamp/symbol observations.")

    if (raw["close"].dropna() <= 0).any():
        raise ValueError("Non-positive closing prices found.")

    if (raw["quote_volume"].dropna() < 0).any():
        raise ValueError("Negative quote volume found.")

    prices = raw.pivot(
        index="timestamp",
        columns="symbol",
        values="close",
    ).sort_index()

    volumes = raw.pivot(
        index="timestamp",
        columns="symbol",
        values="quote_volume",
    ).sort_index()

    prices = np.log(prices.where(prices > 0))

    if not prices.index.equals(volumes.index):
        raise ValueError("Price and volume timestamps do not match.")

    if not prices.columns.equals(volumes.columns):
        raise ValueError("Price and volume symbols do not match.")

    if prices.shape[1] != 20:
        raise ValueError(
            f"Expected 20 symbols, found {prices.shape[1]}."
        )

    return prices, volumes


# ---------------------------------------------------------
# Chronological windows
# ---------------------------------------------------------

def split_windows(prices, volumes):
    n = len(prices)

    n_formation = int(n * FORMATION_FRACTION)
    n_retest1 = int(n * RETEST1_FRACTION)

    formation_end = n_formation
    retest1_end = formation_end + n_retest1

    if retest1_end >= n:
        raise ValueError("Invalid chronological split.")

    windows = {
        "formation": (
            prices.iloc[:formation_end],
            volumes.iloc[:formation_end],
        ),
        "retest1": (
            prices.iloc[formation_end:retest1_end],
            volumes.iloc[formation_end:retest1_end],
        ),
        "retest2": (
            prices.iloc[retest1_end:],
            volumes.iloc[retest1_end:],
        ),
    }

    return windows


# ---------------------------------------------------------
# Fixed regression directions
# ---------------------------------------------------------

def determine_directions(formation_prices, formation_volumes):
    symbols = sorted(formation_prices.columns)

    average_volume = formation_volumes.mean(axis=0)

    if average_volume.isna().any():
        raise ValueError(
            "Cannot determine direction: missing average "
            "formation quote volume for at least one symbol."
        )

    pairs = []

    for a, b in combinations(symbols, 2):
        if average_volume[a] == average_volume[b]:
            raise ValueError(
                f"Equal formation quote volumes for {a} and {b}; "
                "the preregistered direction rule does not resolve ties."
            )

        if average_volume[a] < average_volume[b]:
            symbol_y, symbol_x = a, b
        else:
            symbol_y, symbol_x = b, a

        pairs.append({
            "symbol_y": symbol_y,
            "symbol_x": symbol_x,
            "avg_quote_volume_y": float(average_volume[symbol_y]),
            "avg_quote_volume_x": float(average_volume[symbol_x]),
        })

    return pd.DataFrame(pairs)


# ---------------------------------------------------------
# Engle-Granger regression and residual diagnostics
# ---------------------------------------------------------

def regression_details(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if y.ndim != 1 or x.ndim != 1 or len(y) != len(x):
        raise ValueError("y and x must be equal-length 1D arrays.")

    valid = np.isfinite(y) & np.isfinite(x)
    y = y[valid]
    x = x[valid]

    if len(y) < 25:
        raise ValueError("Insufficient valid observations.")

    design = np.column_stack([np.ones(len(x)), x])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]

    residuals = y - (alpha + beta * x)

    statistic, _ = adf_first_order(residuals)

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "t_stat": float(statistic),
        "residuals": residuals,
        "n_obs": int(len(y)),
    }


def residual_half_life(residuals):
    """
    Estimate AR(1):
        residual_t = intercept + phi * residual_(t-1) + error_t

    Return a finite half-life only when 0 < phi < 1.
    """
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) < 25:
        return np.nan, np.nan

    lagged = residuals[:-1]
    current = residuals[1:]

    design = np.column_stack([
        np.ones(len(lagged)),
        lagged,
    ])

    alpha, phi = np.linalg.lstsq(
        design, current, rcond=None
    )[0]
    # Return a finite half-life only when phi is safely below 1.
    if 0 < phi < 1 - 1e-9:
        half_life = -np.log(2) / np.log(phi)
    else:
        half_life = np.nan

    return float(phi), float(half_life)


# ---------------------------------------------------------
# Run all formation pairs
# ---------------------------------------------------------

def run_formation(formation_prices, directions):
    rows = []

    for pair in directions.itertuples(index=False):
        y = formation_prices[pair.symbol_y].to_numpy()
        x = formation_prices[pair.symbol_x].to_numpy()

        try:
            result = regression_details(y, x)
        except ValueError:
            continue

        phi, half_life = residual_half_life(
            result["residuals"]
        )

        rows.append({
            "symbol_y": pair.symbol_y,
            "symbol_x": pair.symbol_x,
            "n_obs": result["n_obs"],
            "alpha": result["alpha"],
            "beta": result["beta"],
            "t_stat": result["t_stat"],
            "phi": phi,
            "half_life_hours": half_life,
            "cointegrated": result["t_stat"] < CRITICAL_VALUE,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Retest formation survivors
# ---------------------------------------------------------

def run_retests(windows, formation_results):
    selected = formation_results.loc[
        formation_results["cointegrated"]
    ]

    rows = []

    for pair in selected.itertuples(index=False):
        for window_name in ("retest1", "retest2"):
            prices, _ = windows[window_name]

            y = prices[pair.symbol_y].to_numpy()
            x = prices[pair.symbol_x].to_numpy()

            try:
                result = regression_details(y, x)
            except ValueError:
                continue

            phi, half_life = residual_half_life(
                result["residuals"]
            )

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
                    if abs(pair.beta) > 1e-10
                    else np.nan
                ),
                "t_stat": result["t_stat"],
                "phi": phi,
                "half_life_hours": half_life,
                "cointegrated": (
                    result["t_stat"] < CRITICAL_VALUE
                ),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def create_summary(formation, retests):
    formation_passes = int(formation["cointegrated"].sum())

    if retests.empty:
        return pd.DataFrame([{
            "formation_pairs_tested": len(formation),
            "formation_passes": formation_passes,
            "retest1_passes": 0,
            "retest2_passes": 0,
            "passes_both_retests": 0,
        }])

    retest1 = retests[retests["window"] == "retest1"]
    retest2 = retests[retests["window"] == "retest2"]

    passed1 = set(
        zip(
            retest1.loc[retest1["cointegrated"], "symbol_y"],
            retest1.loc[retest1["cointegrated"], "symbol_x"],
        )
    )

    passed2 = set(
        zip(
            retest2.loc[retest2["cointegrated"], "symbol_y"],
            retest2.loc[retest2["cointegrated"], "symbol_x"],
        )
    )

    return pd.DataFrame([{
        "formation_pairs_tested": len(formation),
        "formation_passes": formation_passes,
        "retest1_passes": int(retest1["cointegrated"].sum()),
        "retest2_passes": int(retest2["cointegrated"].sum()),
        "passes_both_retests": len(passed1 & passed2),
    }])


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prices, volumes = load_panel(DATA_PATH)
    windows = split_windows(prices, volumes)

    formation_prices, formation_volumes = windows["formation"]

    directions = determine_directions(
        formation_prices,
        formation_volumes,
    )

    formation = run_formation(
        formation_prices,
        directions,
    )

    retests = run_retests(
        windows,
        formation,
    )

    summary = create_summary(
        formation,
        retests,
    )

    directions.to_csv(
        OUTPUT_DIR / "directions.csv",
        index=False,
    )

    formation.to_csv(
        OUTPUT_DIR / "formation_results.csv",
        index=False,
    )

    retests.to_csv(
        OUTPUT_DIR / "retest_results.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "summary.csv",
        index=False,
    )

    metadata = {
        "observations": len(prices),
        "symbols": len(prices.columns),
        "pairs": len(directions),
        "critical_value": CRITICAL_VALUE,
        "formation_fraction": FORMATION_FRACTION,
        "retest1_fraction": RETEST1_FRACTION,
        "retest2_fraction": RETEST2_FRACTION,
        "start": str(prices.index.min()),
        "end": str(prices.index.max()),
        "window_sizes": {
            name: len(value[0])
            for name, value in windows.items()
        },
    }

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Analysis completed.")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
