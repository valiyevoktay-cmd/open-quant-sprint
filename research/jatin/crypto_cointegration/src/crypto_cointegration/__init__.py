"""Tools for rolling Engle-Granger cointegration and hedge-ratio analysis."""

from .data import load_prices, validate_prices
from .models import engle_granger, kalman_hedge_ratio
from .analysis import (
    make_pairs,
    test_all_pairs,
    rolling_cointegration,
    rolling_pair_results,
    hedge_ratio_stability,
    kalman_pair_analysis,
)

__all__ = [
    "load_prices",
    "validate_prices",
    "engle_granger",
    "kalman_hedge_ratio",
    "make_pairs",
    "test_all_pairs",
    "rolling_cointegration",
    "rolling_pair_results",
    "hedge_ratio_stability",
    "kalman_pair_analysis",
]