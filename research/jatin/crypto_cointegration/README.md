# Crypto Cointegration Research

A small Python project for hourly cryptocurrency log-price analysis:
Engle–Granger pair testing, rolling formation/testing persistence, OLS
hedge-ratio changes, and Kalman-filter hedge-ratio dynamics.

## Project layout

```text
crypto_cointegration/
├── data/                         # Put your Parquet file here (not committed)
├── outputs/                      # Generated CSV results
├── src/crypto_cointegration/
│   ├── data.py                   # Loading and data diagnostics
│   ├── models.py                 # Engle–Granger and Kalman models
│   ├── analysis.py               # Pair tests and window analysis
│   └── plotting.py               # Plot helpers
├── tests/test_models.py
├── run_analysis.py
└── requirements.txt
```

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
```

Place the Parquet file at `data/crypto_data.parquet`, then run:

```bash
python run_analysis.py
```

Run tests:

```bash
pytest
```

## Methodology choices retained from the notebook

- Prices are pivoted to timestamp × symbol and transformed to log prices.
- Engle–Granger is implemented as OLS with an intercept followed by a
  residual ADF regression with no deterministic term and one fixed lag.
- The 5% decision threshold is the approximate two-variable,
  constant-in-cointegrating-regression critical value `-3.34`.
- Formation/testing windows are non-overlapping within each window pair;
  rolling origins advance by seven days.
- Kalman state is `[intercept, beta]`, both following random walks.
  The filter is run continuously across formation and testing, not reset.
- Kalman within-window variation excludes the first 168 observations of
  formation to reduce initialization transients.

## Important research caveats

1. The `-3.34` critical value is approximate. Confirm that the chosen
   deterministic specification, lag rule, and sample-size treatment match
   the reference methodology. Do not use the ordinary residual ADF p-value
   as an Engle–Granger p-value.
2. Pair tests currently require complete observations within each window.
   Missing pairs are skipped, not forward-filled or interpolated. Review
   `validate_prices()` diagnostics before interpreting results.
3. The rolling windows overlap across origins, so window-level observations
   are dependent. Report descriptive summaries accordingly.
4. Testing 190 pairs creates a multiple-testing problem. Add an FDR or
   family-wise correction as a robustness analysis before making strong
   claims about the number of cointegrated pairs.
5. Relative beta change is unstable when the formation beta is close to zero;
   report absolute change alongside it.
6. The hypergeometric overlap calculation from the exploratory notebook
   assumes exchangeable pair labels and is not a full time-series null model.
7. Kalman beta smoothness depends on `delta`, `obs_var`, and initialization.
   It should not be interpreted as direct evidence of economic hedgeability.

## Differences from the notebook

- Removed repeated imports, repeated Kalman loops, repeated summaries, and
  one-off print cells.
- Centralized the data-loading, model, pair-testing, rolling-analysis,
  and plotting code.
- Reused one pair-test implementation for formation, testing, and rolling
  windows.
- Removed machine-specific absolute file paths and warnings suppression.
- Added duplicate-row, non-positive-price, timestamp, and missing-data checks.
- Added deterministic synthetic unit tests.
- The exploratory one-pair sensitivity plot is available through
  `plot_kalman_beta()`; parameter sensitivity can be run by calling the
  Kalman function for each chosen `delta`.
