# open-quant-sprint

Cross-sectional perpetual futures backtest and look-ahead audit harness.

## Dataset

- `data/train_1h.parquet`: Binance USD-M Futures, 20 symbols, 1h bars (69,120 rows).
- Columns: `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `taker_buy_quote_volume`, `trades`, `funding_rate`.
- Rebuild or fetch latest bars: `python -m data.build_dataset --live --days 180`

## Setup

```bash
pip install -e ".[dev]"
pytest
```

## Usage

Audit a strategy file:

```bash
python -m src.audit strategies/baseline_flow.py
python -m src.audit strategies/leaky_example.py
```

## Strategy Interface

Create `strategies/<name>.py` implementing:

```python
import pandas as pd

def compute_weights(df: pd.DataFrame) -> pd.Series:
    # Return target portfolio weights aligned with df.index.
    # Gross exposure is normalized to sum(|w|) <= 1.0 per timestamp by the engine.
    ...
```

## Audit Criteria

1. **Causality (`check_causality_truncation`)**: Prefix cuts at 40%, 65%, 85% of timeline must match full-run weights (`max|w_trunc - w_full| < 1e-9`). Non-deterministic or zero-weight outputs are rejected.
2. **Execution**: Strict 1-bar execution delay (`w[t-1]` applied to `close[t] / close[t-1] - 1`), historical 8h `funding_rate` payments, and linear transaction cost grid (`0, 3, 5, 10 bps`).
3. **Validation**: Purged 70/30 In-Sample / Out-of-Sample split (24h embargo) at `5.0 bps` taker fee. Requires `OOS Sharpe > 0` and `Full Sharpe @ 5 bps > 0`.
