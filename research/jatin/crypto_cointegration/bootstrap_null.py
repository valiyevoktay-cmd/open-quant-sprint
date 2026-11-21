"""
Independent moving-block bootstrap null for crypto cointegration persistence.

Null construction:
- Compute hourly log returns for each asset.
- Resample each asset's return blocks independently.
- This preserves within-asset short-run serial dependence approximately,
  while destroying cross-asset contemporaneous and lagged dependence.
- Reconstruct synthetic log prices and rerun the full rolling EG procedure.

Run from the project root:
    python bootstrap_null.py

Optional command-line overrides:
    python bootstrap_null.py --replications 100 --block-lengths 168
    python bootstrap_null.py --replications 500 --block-lengths 24 72 168
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from crypto_cointegration import load_prices, make_pairs, test_all_pairs

warnings.filterwarnings("ignore")
DATA_PATH = Path("data/crypto_data.parquet")
OUTPUT_DIR = Path("outputs")

WINDOW_DAYS = (14, 30, 60)
ROLLING_STEP_DAYS = 7
EG_CRITICAL_VALUE = -3.34

DEFAULT_REPLICATIONS = 500
DEFAULT_BLOCK_LENGTHS = (168,)
RANDOM_SEED = 20261111


def circular_moving_block_sample(
    returns: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Resample one-dimensional returns using circular moving blocks.

    Each block contains consecutive observations. Circular indexing allows
    blocks to wrap around the end of the original series.
    """
    n = len(returns)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty return series.")
    if block_length < 1:
        raise ValueError("block_length must be at least 1.")

    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_length)
    indices = (starts[:, None] + offsets[None, :]) % n
    return returns[indices.ravel()[:n]]


def bootstrap_log_prices(
    log_prices: pd.DataFrame,
    block_length: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Independently block-bootstrap each asset's hourly log returns.

    The original first log price is retained as the synthetic starting value.
    """
    values = log_prices.to_numpy(dtype=float)
    returns = np.diff(values, axis=0)

    boot_returns = np.empty_like(returns)
    for j in range(returns.shape[1]):
        boot_returns[:, j] = circular_moving_block_sample(
            returns[:, j], block_length, rng
        )

    boot_values = np.empty_like(values)
    boot_values[0, :] = values[0, :]
    boot_values[1:, :] = (
        values[0, :][None, :] + np.cumsum(boot_returns, axis=0)
    )

    return pd.DataFrame(
        boot_values,
        index=log_prices.index,
        columns=log_prices.columns,
    )


def rolling_persistence_statistic(
    log_prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    window_days: int,
    step_days: int,
    critical_value: float,
) -> dict:
    """
    Run the full rolling formation/testing EG procedure.

    Returns pooled persistence:
        total pairs passing both windows / total pairs passing formation.
    """
    formation_n = window_days * 24
    testing_n = window_days * 24
    step_n = step_days * 24

    total_formation_passes = 0
    total_persistent = 0
    n_origins = 0

    final_start = len(log_prices) - formation_n - testing_n
    for start in range(0, final_start + 1, step_n):
        split = start + formation_n
        end = split + testing_n

        formation_results = test_all_pairs(
            log_prices.iloc[start:split],
            pairs,
            critical_value,
        )
        testing_results = test_all_pairs(
            log_prices.iloc[split:end],
            pairs,
            critical_value,
        )

        if formation_results.empty or testing_results.empty:
            continue

        merged = formation_results.merge(
            testing_results,
            on=["symbol_y", "symbol_x"],
            suffixes=("_formation", "_testing"),
        )

        formation_pass = merged["cointegrated_formation"].to_numpy(dtype=bool)
        testing_pass = merged["cointegrated_testing"].to_numpy(dtype=bool)

        total_formation_passes += int(formation_pass.sum())
        total_persistent += int(
            np.logical_and(formation_pass, testing_pass).sum()
        )
        n_origins += 1

    rate = (
        total_persistent / total_formation_passes
        if total_formation_passes
        else np.nan
    )

    return {
        "window_days": window_days,
        "n_origins": n_origins,
        "formation_passes_total": total_formation_passes,
        "persistent_total": total_persistent,
        "persistence": rate,
    }


def run_bootstrap(
    log_prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    replications: int,
    block_lengths: tuple[int, ...],
    seed: int,
):
    rng = np.random.default_rng(seed)
    output_rows = []

    # Compute observed statistics once from the real series.
    observed = {
        days: rolling_persistence_statistic(
            log_prices,
            pairs,
            days,
            ROLLING_STEP_DAYS,
            EG_CRITICAL_VALUE,
        )
        for days in WINDOW_DAYS
    }

    print("\nObserved persistence:")
    for days in WINDOW_DAYS:
        print(f"  {days}d: {observed[days]['persistence']:.6f}")

    total_runs = len(block_lengths) * replications
    run_number = 0

    for block_length in block_lengths:
        for replication in range(replications):
            run_number += 1
            synthetic_prices = bootstrap_log_prices(
                log_prices, block_length, rng
            )

            for days in WINDOW_DAYS:
                result = rolling_persistence_statistic(
                    synthetic_prices,
                    pairs,
                    days,
                    ROLLING_STEP_DAYS,
                    EG_CRITICAL_VALUE,
                )
                output_rows.append({
                    "block_length_hours": block_length,
                    "replication": replication,
                    **result,
                })

            if run_number == 1 or run_number % 10 == 0:
                print(
                    f"Completed {run_number}/{total_runs} bootstrap "
                    f"replications (block length={block_length}h)"
                )

    bootstrap = pd.DataFrame(output_rows)
    summary_rows = []

    for block_length in block_lengths:
        for days in WINDOW_DAYS:
            values = bootstrap.loc[
                (bootstrap["block_length_hours"] == block_length)
                & (bootstrap["window_days"] == days),
                "persistence",
            ].dropna().to_numpy()

            obs = observed[days]["persistence"]
            if len(values) == 0:
                raise RuntimeError(
                    f"No valid bootstrap statistics for {days}d, "
                    f"block length {block_length}h."
                )

            # One-sided alternative: observed persistence exceeds null.
            p_value = (
                1 + np.count_nonzero(values >= obs)
            ) / (1 + len(values))

            summary_rows.append({
                "window_days": days,
                "block_length_hours": block_length,
                "observed_persistence": obs,
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_median": float(np.median(values)),
                "bootstrap_std": float(np.std(values, ddof=1))
                    if len(values) > 1 else np.nan,
                "bootstrap_95th_percentile": float(
                    np.quantile(values, 0.95)
                ),
                "observed_minus_bootstrap_mean": float(
                    obs - np.mean(values)
                ),
                "empirical_one_sided_p_value": float(p_value),
                "replications": int(len(values)),
            })

    return bootstrap, pd.DataFrame(summary_rows), observed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Independent moving-block bootstrap for cointegration persistence."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Input Parquet file (default: train_1h.parquet).",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=DEFAULT_REPLICATIONS,
        help="Number of bootstrap replications per block length.",
    )
    parser.add_argument(
        "--block-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_BLOCK_LENGTHS),
        help="Block lengths in hours; e.g. 24 72 168.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.replications < 1:
        raise ValueError("--replications must be positive.")
    if any(length < 1 for length in args.block_lengths):
        raise ValueError("All block lengths must be positive.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data} ...")
    log_prices = load_prices(args.data)
    pairs = make_pairs(log_prices.columns)

    print(
        f"Loaded {len(log_prices.columns)} assets, "
        f"{len(log_prices):,} hourly observations, "
        f"{len(pairs)} pairs."
    )
    print(
        "\nBootstrap design: each asset is resampled independently. "
        "This destroys cross-asset dependence while preserving "
        "within-asset dependence within sampled blocks."
    )
    print(
        f"Replications per block length: {args.replications}; "
        f"block lengths: {args.block_lengths}"
    )
    print(
        "Runtime may be substantial because every bootstrap replicate "
        "reruns the full Engle-Granger procedure for all pairs and windows."
    )

    bootstrap, summary, observed = run_bootstrap(
        log_prices=log_prices,
        pairs=pairs,
        replications=args.replications,
        block_lengths=tuple(args.block_lengths),
        seed=args.seed,
    )

    bootstrap_path = OUTPUT_DIR / "persistence_bootstrap_replicates.csv"
    summary_path = OUTPUT_DIR / "persistence_bootstrap_summary.csv"
    observed_path = OUTPUT_DIR / "persistence_bootstrap_observed.csv"

    bootstrap.to_csv(bootstrap_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(observed.values()).to_csv(observed_path, index=False)

    print("\nBootstrap summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {bootstrap_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {observed_path}")


if __name__ == "__main__":
    main()
