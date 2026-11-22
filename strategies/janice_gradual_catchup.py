"""Candidate A: fixed 25% adjustment toward ungated catch-up, every8h."""
import pandas as pd
from src.contrib.janice_alternatives import gradual_catchup_weights


def compute_weights(df:pd.DataFrame)->pd.Series:
    return gradual_catchup_weights(df)
