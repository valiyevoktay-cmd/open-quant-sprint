"""
Temporal-shift placebo analysis for rolling cointegration persistence.

Input:
    outputs/rolling_14d_pair_results.csv
    outputs/rolling_30d_pair_results.csv
    outputs/rolling_60d_pair_results.csv

The test compares observed same-origin persistence with persistence after
circularly shifting the testing-window pass/fail indicators across origins.
A common shift is applied to every pair, preserving cross-sectional alignment.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


OUTPUT_DIR = Path("outputs")
WINDOW_DAYS = (14, 30, 60)
ROLLING_STEP_DAYS = 7


def persistence_rate(formation: np.ndarray, testing: np.ndarray) -> float:
    """Fraction of formation passes that also pass in testing."""
    denominator = int(formation.sum())
    if denominator == 0:
        return np.nan
    return float(np.logical_and(formation, testing).sum() / denominator)


def analyze_window(
    csv_path: Path,
    window_days: int,
    step_days: int = ROLLING_STEP_DAYS,
):
    """
    Calculate observed persistence and a temporal-shift placebo distribution.

    The minimum circular shift is an embargo of two window lengths, expressed
    in rolling origins. This reduces direct temporal overlap between the
    periods being aligned. It does not make the shifts statistically
    independent or remove all serial dependence.
    """
    data = pd.read_csv(csv_path, parse_dates=["formation_start"])

    required = {
        "formation_start",
        "symbol_y",
        "symbol_x",
        "cointegrated_formation",
        "cointegrated_testing",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing)}"
        )

    data["pair"] = (
        data["symbol_y"].astype(str) + "::" + data["symbol_x"].astype(str)
    )

    formation = data.pivot(
        index="formation_start",
        columns="pair",
        values="cointegrated_formation",
    ).sort_index()

    testing = data.pivot(
        index="formation_start",
        columns="pair",
        values="cointegrated_testing",
    ).sort_index()

    # Align origin and pair dimensions. Missing cells are treated as unavailable,
    # not as failed tests; current balanced data should have no such cells.
    testing = testing.reindex(index=formation.index, columns=formation.columns)

    if formation.isna().any().any() or testing.isna().any().any():
        raise ValueError(
            "Missing pair/origin results found. Resolve incomplete rolling "
            "results before running the placebo test."
        )

    f = formation.to_numpy(dtype=bool)
    t = testing.to_numpy(dtype=bool)

    observed = persistence_rate(f, t)
    n_origins = f.shape[0]

    # Require a circular separation of at least 2 * window length.
    embargo = math.ceil((2 * window_days) / step_days)
    shifts = [
        shift for shift in range(1, n_origins)
        if min(shift, n_origins - shift) >= embargo
    ]

    if len(shifts) < 5:
        raise ValueError(
            f"Only {len(shifts)} admissible shifts for {window_days}-day "
            f"windows ({n_origins} origins). Too few for a useful "
            "temporal-shift placebo."
        )

    null_rows = []
    for shift in shifts:
        shifted_testing = np.roll(t, shift=shift, axis=0)
        rate = persistence_rate(f, shifted_testing)
        null_rows.append({
            "window_days": window_days,
            "shift_origins": shift,
            "shift_days": shift * step_days,
            "placebo_persistence": rate,
        })

    null = pd.DataFrame(null_rows)
    null_values = null["placebo_persistence"].dropna().to_numpy()

    # One-sided test: is adjacent-window persistence unusually high?
    p_value = (
        1 + np.count_nonzero(null_values >= observed)
    ) / (1 + len(null_values))

    summary = {
        "window_days": window_days,
        "n_origins": n_origins,
        "n_pairs": f.shape[1],
        "formation_passes_total": int(f.sum()),
        "observed_persistence": observed,
        "placebo_mean": float(np.mean(null_values)),
        "placebo_median": float(np.median(null_values)),
        "placebo_95th_percentile": float(np.quantile(null_values, 0.95)),
        "observed_minus_placebo_mean": float(
            observed - np.mean(null_values)
        ),
        "empirical_one_sided_p_value": float(p_value),
        "n_placebo_shifts": int(len(null_values)),
        "minimum_shift_origins": int(embargo),
    }

    return summary, null


def main():
    summaries = []

    for days in WINDOW_DAYS:
        input_path = OUTPUT_DIR / f"rolling_{days}d_pair_results.csv"

        if not input_path.exists():
            raise FileNotFoundError(
                f"Could not find {input_path}. Run run_analysis.py first."
            )

        summary, null = analyze_window(input_path, days)
        summaries.append(summary)

        null.to_csv(
            OUTPUT_DIR / f"persistence_null_{days}d.csv",
            index=False,
        )

        print(f"\n{days}-day temporal-shift placebo:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(
        OUTPUT_DIR / "persistence_null_summary.csv",
        index=False,
    )

    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
