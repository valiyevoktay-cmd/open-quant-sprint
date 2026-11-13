"""
Pair-level rolling cointegration persistence analysis.

Place this file in the project root and run:
    python pair_persistence_analysis.py

By default, inputs are read from outputs/:
    rolling_14d_pair_results.csv
    rolling_30d_pair_results.csv
    rolling_60d_pair_results.csv

Outputs are written to outputs/pair_persistence/.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("outputs")
DEFAULT_OUTPUT_DIR = Path("outputs/pair_persistence")
WINDOWS = (14, 30, 60)

REQUIRED_COLUMNS = {
    "symbol_y",
    "symbol_x",
    "beta_formation",
    "beta_testing",
    "cointegrated_formation",
    "cointegrated_testing",
    "persistent",
    "formation_start",
}


def load_pair_results(input_dir: Path, days: int) -> pd.DataFrame:
    path = input_dir / f"rolling_{days}d_pair_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find input file: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    for col in (
        "cointegrated_formation",
        "cointegrated_testing",
        "persistent",
    ):
        # CSV booleans are normally read as bool; this also handles strings.
        if df[col].dtype != bool:
            mapping = {
                "true": True, "false": False,
                "1": True, "0": False,
            }
            df[col] = df[col].astype(str).str.lower().map(mapping)
            if df[col].isna().any():
                raise ValueError(f"Could not parse boolean column {col} in {path}")

    df["formation_start"] = pd.to_datetime(
        df["formation_start"], utc=True
    )

    # Recompute the joint event and verify the supplied persistent flag.
    expected = (
        df["cointegrated_formation"] & df["cointegrated_testing"]
    )
    mismatch_count = int((df["persistent"] != expected).sum())
    if mismatch_count:
        raise ValueError(
            f"{path}: persistent disagrees with formation AND testing "
            f"for {mismatch_count} rows."
        )

    return df


def summarize_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    data = df.copy()

    # Hedge-ratio changes are evaluated only when the pair passed formation.
    selected = data["cointegrated_formation"]
    data["abs_beta_change"] = (
        data["beta_testing"] - data["beta_formation"]
    ).abs()

    # Relative change can explode when the formation beta is near zero.
    beta_denom = data["beta_formation"].abs()
    data["rel_beta_change"] = np.where(
        beta_denom > 1e-6,
        data["abs_beta_change"] / beta_denom,
        np.nan,
    )

    keys = ["symbol_y", "symbol_x"]
    summary = data.groupby(keys).agg(
        n_origins=("persistent", "size"),
        formation_passes=("cointegrated_formation", "sum"),
        testing_passes=("cointegrated_testing", "sum"),
        persistent_count=("persistent", "sum"),
    ).reset_index()

    beta_summary = (
        data.loc[selected]
        .groupby(keys)
        .agg(
            mean_abs_beta_change=("abs_beta_change", "mean"),
            median_abs_beta_change=("abs_beta_change", "median"),
            mean_rel_beta_change=("rel_beta_change", "mean"),
            median_rel_beta_change=("rel_beta_change", "median"),
        )
        .reset_index()
    )

    summary = summary.merge(beta_summary, on=keys, how="left")
    summary["formation_pass_rate"] = (
        summary["formation_passes"] / summary["n_origins"]
    )
    summary["testing_pass_rate"] = (
        summary["testing_passes"] / summary["n_origins"]
    )
    summary["persistence_rate"] = (
        summary["persistent_count"]
        / summary["formation_passes"].replace(0, np.nan)
    )
    summary.insert(0, "window_days", days)

    return summary.sort_values(
        ["persistence_rate", "formation_passes"],
        ascending=[False, False],
        na_position="last",
    )


def make_plots(all_summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot top pairs by persistence, requiring at least five formation passes.
    for days in WINDOWS:
        subset = all_summary[
            (all_summary["window_days"] == days)
            & (all_summary["formation_passes"] >= 5)
        ].copy()

        top = subset.sort_values(
            ["persistence_rate", "formation_passes"],
            ascending=[False, False],
        ).head(15)

        if top.empty:
            continue

        labels = (
            top["symbol_y"].str.replace("USDT", "", regex=False)
            + " / "
            + top["symbol_x"].str.replace("USDT", "", regex=False)
        )
        values = top["persistence_rate"] * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(labels[::-1], values[::-1])
        ax.set_xlabel("Persistence rate (%)")
        ax.set_ylabel("Pair")
        ax.set_title(
            f"Top pair persistence — {days}-day windows "
            "(minimum 5 formation passes)"
        )
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"top_pair_persistence_{days}d.png",
            dpi=160,
        )
        plt.close(fig)

    # Compare persistence across horizons for pairs observed in all files.
    pivot = all_summary.pivot(
        index=["symbol_y", "symbol_x"],
        columns="window_days",
        values="persistence_rate",
    )
    pivot.columns = [f"persistence_{int(c)}d" for c in pivot.columns]
    pivot = pivot.reset_index()

    cols = [f"persistence_{d}d" for d in WINDOWS]
    if all(c in pivot.columns for c in cols):
        plot_data = pivot.set_index(
            pivot["symbol_y"].str.replace("USDT", "", regex=False)
            + " / "
            + pivot["symbol_x"].str.replace("USDT", "", regex=False)
        )[cols]
        plot_data.columns = [f"{d}d" for d in WINDOWS]

        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(
            plot_data.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set_title("Pair persistence across window horizons")
        ax.set_xlabel("Formation/testing window")
        ax.set_ylabel("Pair")
        ax.set_xticks(range(len(plot_data.columns)))
        ax.set_xticklabels(plot_data.columns)
        if len(plot_data) <= 60:
            ax.set_yticks(range(len(plot_data.index)))
            ax.set_yticklabels(plot_data.index, fontsize=6)
        else:
            ax.set_yticks([])
        fig.colorbar(image, ax=ax, label="Persistence rate")
        fig.tight_layout()
        fig.savefig(
            output_dir / "persistence_across_horizons.png",
            dpi=160,
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing rolling pair-result CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary CSVs and plots.",
    )
    parser.add_argument(
        "--min-formation-passes",
        type=int,
        default=5,
        help="Minimum formation passes for the ranked-pairs CSV.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for days in WINDOWS:
        df = load_pair_results(args.input_dir, days)
        summary = summarize_window(df, days)
        summaries.append(summary)

        print(
            f"{days}d: {len(df):,} pair-origin rows, "
            f"{df['formation_start'].nunique()} rolling origins, "
            f"{int(df['cointegrated_formation'].sum())} formation passes, "
            f"{int(df['persistent'].sum())} persistent observations."
        )

        ranked = summary[
            summary["formation_passes"] >= args.min_formation_passes
        ].copy()
        ranked.to_csv(
            args.output_dir / f"pair_ranking_{days}d.csv",
            index=False,
        )

        print(f"\nTop pairs — {days}d (minimum "
              f"{args.min_formation_passes} formation passes):")
        display_cols = [
            "symbol_y", "symbol_x", "formation_passes",
            "persistent_count", "persistence_rate",
            "mean_abs_beta_change",
        ]
        print(
            ranked[display_cols]
            .head(10)
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
        print()

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(
        args.output_dir / "pair_persistence_all_windows.csv",
        index=False,
    )

    # One row per pair, with persistence rates side by side across horizons.
    wide = all_summary.pivot(
        index=["symbol_y", "symbol_x"],
        columns="window_days",
        values="persistence_rate",
    )
    wide.columns = [f"persistence_{int(c)}d" for c in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(
        args.output_dir / "pair_persistence_horizon_comparison.csv",
        index=False,
    )

    make_plots(all_summary, args.output_dir)

    print("Analysis complete.")
    print(f"Outputs saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
