
import json
import platform
from datetime import datetime, timezone
import warnings 
import numpy as np
import pandas as pd

from config import CONFIG
from crypto_cointegration import (
    load_prices,
    validate_prices,
    make_pairs,
    test_all_pairs,
    rolling_cointegration,
    rolling_pair_results,
    hedge_ratio_stability,
    kalman_pair_analysis,
)

warnings.filterwarnings("ignore")


def main():
    output_dir = CONFIG.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate data
    log_prices = load_prices(CONFIG.data_path)
    diagnostics = validate_prices(log_prices)

    print("Data diagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")

    pairs = make_pairs(log_prices.columns)

    # 2. Initial formation/testing analysis
    formation_n = CONFIG.formation_days * 24
    testing_n = CONFIG.testing_days * 24

    formation = log_prices.iloc[:formation_n]
    testing = log_prices.iloc[
        formation_n:formation_n + testing_n
    ]

    formation_results = test_all_pairs(
        formation, pairs, CONFIG.eg_critical_value
    )

    testing_results = test_all_pairs(
        testing, pairs, CONFIG.eg_critical_value
    )

    stability = hedge_ratio_stability(
        formation_results, testing_results
    )

    kalman = kalman_pair_analysis(
        log_prices,
        formation_results,
        formation_hours=formation_n,
        testing_hours=testing_n,
        warmup_hours=CONFIG.kalman_warmup_hours,
        delta=CONFIG.kalman_delta,
        obs_var=CONFIG.kalman_obs_var,
    )

    # 3. Save initial results
    formation_results.to_csv(
        output_dir / "formation_results.csv", index=False
    )

    testing_results.to_csv(
        output_dir / "testing_results.csv", index=False
    )

    stability.to_csv(
        output_dir / "ols_hedge_ratio_stability.csv", index=False
    )

    kalman.to_csv(
        output_dir / "kalman_hedge_ratio_stability.csv", index=False
    )

    # 4. Rolling analysis with detailed pair-level results
    for days in CONFIG.rolling_window_days:

        detailed = rolling_pair_results(
            log_prices,
            pairs=pairs,
            formation_days=days,
            testing_days=days,
            step_days=CONFIG.rolling_step_days,
            critical_value=CONFIG.eg_critical_value,
        )

        summary = rolling_cointegration(
            log_prices,
            pairs=pairs,
            formation_days=days,
            testing_days=days,
            step_days=CONFIG.rolling_step_days,
            critical_value=CONFIG.eg_critical_value,
        )

        detailed.to_csv(
            output_dir / f"rolling_{days}d_pair_results.csv",
            index=False,
        )

        summary.to_csv(
            output_dir / f"rolling_{days}d_summary.csv",
            index=False,
        )

        print(f"\n{days}-day windows:")
        print(
            summary[
                [
                    "formation_passes",
                    "testing_passes",
                    "persistent",
                    "persistence_rate",
                ]
            ].mean()
        )

    # 5. Save reproducibility metadata
    metadata = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "data_path": str(CONFIG.data_path),
        "parameters": {
            "formation_days": CONFIG.formation_days,
            "testing_days": CONFIG.testing_days,
            "rolling_step_days": CONFIG.rolling_step_days,
            "rolling_window_days": list(
                CONFIG.rolling_window_days
            ),
            "eg_critical_value": CONFIG.eg_critical_value,
            "kalman_delta": CONFIG.kalman_delta,
            "kalman_obs_var": CONFIG.kalman_obs_var,
            "kalman_warmup_hours": CONFIG.kalman_warmup_hours,
        },
        "data_diagnostics": diagnostics,
    }

    with open(
        output_dir / "run_metadata.json", "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2, default=str)

    # 6. Print final summaries
    print("\nOLS hedge-ratio stability:")
    print(
        stability[
            ["absolute_change", "relative_change"]
        ].describe()
    )

    print("\nKalman hedge-ratio stability:")
    print(
        kalman[
            [
                "absolute_change",
                "relative_change",
                "formation_variation",
                "testing_variation",
            ]
        ].agg(["mean", "median", "count"])
    )

    print(f"\nOutputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
