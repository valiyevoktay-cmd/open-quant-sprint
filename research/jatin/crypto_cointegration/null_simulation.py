
from pathlib import Path
import json

import numpy as np
import pandas as pd

import sys
from src.stats import adf_first_order


# ------------------------------------------------------------
# Configuration: keep identical to the real-panel analysis
# ------------------------------------------------------------

DATA_PATH = Path("data/train_1h.parquet")
REAL_OUTPUT = Path("outputs/part2")
NULL_OUTPUT = REAL_OUTPUT / "null"

N_SIMULATIONS = 1000
RANDOM_SEED = 20261118
CRITICAL_VALUE = -3.34

FORMATION_FRACTION = 0.40
RETEST1_FRACTION = 0.30
RETEST2_FRACTION = 0.30


# ------------------------------------------------------------
# Load real panel and preregistered directions
# ------------------------------------------------------------

def load_real_panel():
    raw = pd.read_parquet(DATA_PATH)

    prices = raw.pivot(
        index="timestamp",
        columns="symbol",
        values="close",
    ).sort_index()

    log_prices = np.log(prices)

    directions = pd.read_csv(REAL_OUTPUT / "directions.csv")

    symbols = sorted(log_prices.columns.tolist())

    if len(symbols) != 20:
        raise ValueError(f"Expected 20 symbols, found {len(symbols)}")

    if len(directions) != 190:
        raise ValueError(
            f"Expected 190 directed pairs, found {len(directions)}"
        )

    # Use the exact window sizes from the completed real analysis.
    with open(REAL_OUTPUT / "metadata.json") as f:
        metadata = json.load(f)

    n_formation = metadata["window_sizes"]["formation"]
    n_retest1 = metadata["window_sizes"]["retest1"]
    n_retest2 = metadata["window_sizes"]["retest2"]

    if n_formation + n_retest1 + n_retest2 != len(log_prices):
        raise ValueError("Window sizes do not match the real dataset length")

    # Estimate each symbol's innovation volatility using formation data only.
    formation = log_prices.iloc[:n_formation]
    returns = formation.diff()

    volatilities = returns.std(ddof=1)

    if volatilities.isna().any() or (volatilities <= 0).any():
        raise ValueError("Invalid formation-window return volatility")

    return (
        symbols,
        directions,
        volatilities,
        n_formation,
        n_retest1,
        n_retest2,
    )


# ------------------------------------------------------------
# Engle-Granger regression and residual half-life
# ------------------------------------------------------------

def regression_details(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    valid = np.isfinite(y) & np.isfinite(x)
    y = y[valid]
    x = x[valid]

    if len(y) < 25:
        raise ValueError("Insufficient observations")

    X = np.column_stack([np.ones(len(x)), x])
    coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    alpha = float(coefficients[0])
    beta = float(coefficients[1])

    residuals = y - (alpha + beta * x)

    t_stat, _ = adf_first_order(residuals)

    # AR(1) with an intercept:
    # residual[t] = c + phi * residual[t-1] + error[t]
    lagged = residuals[:-1]
    current = residuals[1:]

    ar_X = np.column_stack([np.ones(len(lagged)), lagged])
    ar_coefficients, _, _, _ = np.linalg.lstsq(
        ar_X, current, rcond=None
    )

    phi = float(ar_coefficients[1])

    if 0 < phi < 1-1e-9:
        half_life = float(-np.log(2) / np.log(phi))
    else:
        half_life = np.nan

    return {
        "alpha": alpha,
        "beta": beta,
        "t_stat": float(t_stat),
        "phi": phi,
        "half_life_hours": half_life,
        "n_obs": len(y),
    }


# ------------------------------------------------------------
# One simulated panel
# ------------------------------------------------------------

def simulate_one_panel(
    rng,
    symbols,
    directions,
    volatilities,
    n_formation,
    n_retest1,
    n_retest2,
):
    n_total = n_formation + n_retest1 + n_retest2

    # Independent Gaussian increments, with symbol-specific volatility.
    sigma = volatilities.reindex(symbols).to_numpy(dtype=float)

    increments = rng.normal(
        loc=0.0,
        scale=sigma,
        size=(n_total - 1, len(symbols)),
    )

    # Independent random walks in log-price space.
    walks = np.vstack([
        np.zeros((1, len(symbols))),
        np.cumsum(increments, axis=0),
    ])

    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}

    windows = {
        "formation": (0, n_formation),
        "retest1": (n_formation, n_formation + n_retest1),
        "retest2": (n_formation + n_retest1, n_total),
    }

    formation_results = []
    retest_results = []

    # Test all 190 directed pairs in formation.
    for pair in directions.itertuples(index=False):
        y_name = pair.symbol_y
        x_name = pair.symbol_x

        y_idx = symbol_index[y_name]
        x_idx = symbol_index[x_name]

        y = walks[:, y_idx]
        x = walks[:, x_idx]

        start, end = windows["formation"]

        formation = regression_details(
            y[start:end],
            x[start:end],
        )

        formation_pass = formation["t_stat"] < CRITICAL_VALUE

        formation_results.append({
            "symbol_y": y_name,
            "symbol_x": x_name,
            "t_stat": formation["t_stat"],
            "beta": formation["beta"],
            "cointegrated": formation_pass,
        })

        # Retest only formation survivors.
        if not formation_pass:
            continue

        for window_name in ("retest1", "retest2"):
            start, end = windows[window_name]

            retest = regression_details(
                y[start:end],
                x[start:end],
            )

            beta_change = abs(
                retest["beta"] - formation["beta"]
            )

            if abs(formation["beta"]) > 1e-12:
                relative_beta_change = (
                    beta_change / abs(formation["beta"])
                )
            else:
                relative_beta_change = np.nan

            retest_results.append({
                "symbol_y": y_name,
                "symbol_x": x_name,
                "window": window_name,
                "beta_formation": formation["beta"],
                "beta_retest": retest["beta"],
                "absolute_beta_change": beta_change,
                "relative_beta_change": relative_beta_change,
                "t_stat": retest["t_stat"],
                "phi": retest["phi"],
                "half_life_hours": retest["half_life_hours"],
                "cointegrated": (
                    retest["t_stat"] < CRITICAL_VALUE
                ),
            })

    formation_df = pd.DataFrame(formation_results)
    retest_df = pd.DataFrame(retest_results)

    formation_passes = int(formation_df["cointegrated"].sum())

    if len(retest_df):
        retest1 = retest_df[retest_df["window"] == "retest1"]
        retest2 = retest_df[retest_df["window"] == "retest2"]

        retest1_passes = int(retest1["cointegrated"].sum())
        retest2_passes = int(retest2["cointegrated"].sum())

        # Count pairs that pass both retest windows.
        both = set(
            zip(
                retest1.loc[retest1["cointegrated"], "symbol_y"],
                retest1.loc[retest1["cointegrated"], "symbol_x"],
            )
        ) & set(
            zip(
                retest2.loc[retest2["cointegrated"], "symbol_y"],
                retest2.loc[retest2["cointegrated"], "symbol_x"],
            )
        )

        passes_both = len(both)
    else:
        retest1_passes = 0
        retest2_passes = 0
        passes_both = 0

    return {
        "formation_passes": formation_passes,
        "retest1_passes": retest1_passes,
        "retest2_passes": retest2_passes,
        "passes_both_retests": passes_both,
    }, retest_df


# ------------------------------------------------------------
# Run all null simulations
# ------------------------------------------------------------

def main():
    NULL_OUTPUT.mkdir(parents=True, exist_ok=True)

    (
        symbols,
        directions,
        volatilities,
        n_formation,
        n_retest1,
        n_retest2,
    ) = load_real_panel()

    rng = np.random.default_rng(RANDOM_SEED)

    panel_rows = []
    selected_retest_rows = []

    for simulation in range(1, N_SIMULATIONS + 1):
        counts, retests = simulate_one_panel(
            rng=rng,
            symbols=symbols,
            directions=directions,
            volatilities=volatilities,
            n_formation=n_formation,
            n_retest1=n_retest1,
            n_retest2=n_retest2,
        )

        panel_rows.append({
            "simulation": simulation,
            **counts,
        })

        if len(retests):
            retests.insert(0, "simulation", simulation)
            selected_retest_rows.append(retests)

        if simulation % 50 == 0:
            print(f"Completed {simulation}/{N_SIMULATIONS}")

    panels = pd.DataFrame(panel_rows)

    panels.to_csv(
        NULL_OUTPUT / "null_panel_counts.csv",
        index=False,
    )

    if selected_retest_rows:
        selected_retests = pd.concat(
            selected_retest_rows,
            ignore_index=True,
        )
    else:
        selected_retests = pd.DataFrame()

    selected_retests.to_csv(
        NULL_OUTPUT / "null_selected_retests.csv",
        index=False,
    )

    # Distribution of simulated panel-level counts.
    count_columns = [
        "formation_passes",
        "retest1_passes",
        "retest2_passes",
        "passes_both_retests",
    ]

    distribution = panels[count_columns].agg(
        ["mean", "std", "median", "min", "max"]
    )

    distribution.to_csv(
        NULL_OUTPUT / "null_count_distribution.csv"
    )

    # Compare real-panel counts with the simulated null.
    real_summary = pd.read_csv(REAL_OUTPUT / "summary.csv").iloc[0]

    observed = {
        "formation_passes": int(real_summary["formation_passes"]),
        "retest1_passes": int(real_summary["retest1_passes"]),
        "retest2_passes": int(real_summary["retest2_passes"]),
        "passes_both_retests": int(
            real_summary["passes_both_retests"]
        ),
    }

    comparisons = []

    for column in count_columns:
        null_values = panels[column].to_numpy()
        observed_value = observed[column]

        # Add-one empirical upper-tail probability.
        empirical_p = (
            1 + np.sum(null_values >= observed_value)
        ) / (N_SIMULATIONS + 1)

        comparisons.append({
            "metric": column,
            "observed_real": observed_value,
            "null_mean": float(np.mean(null_values)),
            "null_median": float(np.median(null_values)),
            "null_95th_percentile": float(
                np.quantile(null_values, 0.95)
            ),
            "empirical_upper_tail_p": float(empirical_p),
        })

    comparison_df = pd.DataFrame(comparisons)

    comparison_df.to_csv(
        NULL_OUTPUT / "real_vs_null.csv",
        index=False,
    )

    # Descriptive measurements among formation-selected null pairs.
    # These are selection-biased by construction.
    if len(selected_retests):
        selected_retests.groupby("window").agg(
            selected_pair_window_count=("beta_retest", "size"),
            mean_relative_beta_change=("relative_beta_change", "mean"),
            median_relative_beta_change=("relative_beta_change", "median"),
            mean_half_life_hours=("half_life_hours", "mean"),
            median_half_life_hours=("half_life_hours", "median"),
        ).to_csv(NULL_OUTPUT / "null_selected_pair_metrics.csv")

    metadata = {
        "simulations": N_SIMULATIONS,
        "random_seed": RANDOM_SEED,
        "symbols": len(symbols),
        "pairs_per_panel": len(directions),
        "critical_value": CRITICAL_VALUE,
        "window_sizes": {
            "formation": n_formation,
            "retest1": n_retest1,
            "retest2": n_retest2,
        },
        "random_walk_model": (
            "Independent Gaussian random walks in log-price space; "
            "symbol-specific innovation volatility estimated from "
            "formation-window log returns."
        ),
        "direction_rule": (
            "Fixed directions loaded from the real-panel directions.csv."
        ),
        "selection_bias_warning": (
            "Beta changes and half-lives are computed only for pairs "
            "selected by passing formation. These conditional metrics "
            "are subject to selection bias."
        ),
    }

    with open(NULL_OUTPUT / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nNULL SIMULATION COMPLETE")
    print(f"Results saved to: {NULL_OUTPUT.resolve()}")

    print("\nREAL PANEL VS NULL")
    print(comparison_df.to_string(index=False))

    print("\nNULL COUNT DISTRIBUTION")
    print(distribution)


if __name__ == "__main__":
    main()
