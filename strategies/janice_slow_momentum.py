"""Candidate B: fixed 7-day volatility-scaled momentum, daily decisions."""
import pandas as pd
from src.contrib.janice_alternatives import slow_momentum_weights


def compute_weights(df:pd.DataFrame)->pd.Series:
    return slow_momentum_weights(df)
