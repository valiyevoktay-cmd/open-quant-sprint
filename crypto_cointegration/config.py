from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    # Input and output
    data_path: Path = Path("data/crypto_data.parquet")
    output_dir: Path = Path("outputs")

    # Formation/testing windows
    formation_days: int = 30
    testing_days: int = 30
    rolling_step_days: int = 7
    rolling_window_days: tuple = (14, 30, 60)

    # Engle-Granger
    eg_critical_value: float = -3.34

    # Kalman filter
    kalman_delta: float = 1e-5
    kalman_obs_var: float = 1e-3
    kalman_warmup_hours: int = 168


CONFIG = AnalysisConfig()
