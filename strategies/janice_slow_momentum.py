"""Candidate B: fixed 7-day volatility-scaled momentum, daily decisions."""
import pandas as pd
from src.contrib.janice_alternatives import slow_momentum_weights
from src.contrib.janice_gaps import on_full_grid


def compute_weights(df:pd.DataFrame)->pd.Series:
    # Maintainer change before the freeze: missing hours are filled as the data builder fills a short gap.
    return on_full_grid(slow_momentum_weights, df)
