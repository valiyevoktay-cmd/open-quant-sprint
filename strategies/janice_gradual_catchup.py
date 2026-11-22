"""Candidate A: fixed 25% adjustment toward ungated catch-up, every8h."""
import pandas as pd
from src.contrib.janice_alternatives import gradual_catchup_weights
from src.contrib.janice_gaps import on_full_grid


def compute_weights(df:pd.DataFrame)->pd.Series:
    # Maintainer change before the freeze: missing hours are filled as the data builder fills a short gap.
    return on_full_grid(gradual_catchup_weights, df)
