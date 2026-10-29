# open-quant-sprint

Cross-sectional perpetual futures backtest and look-ahead audit harness.

## Dataset

- `data/train_1h.parquet`: Binance USD-M Futures, 20 symbols, 1h bars (69,120 rows).
- Columns: `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `taker_buy_quote_volume`, `trades`, `funding_rate`.
- Rebuild or fetch latest bars: `python -m data.build_dataset --live --days 180`

## Setup

```bash
pip install -e ".[dev]"
python -m pytest
```

## Usage

Audit a strategy file:

```bash
python -m src.audit strategies/template.py
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
2. **Lag Sensitivity Test**: Adds +1 bar execution delay (lag 2 vs lag 1). Distinguishes structural persistence from both algorithmic look-ahead leaks and unexecutable microstructure noise (short-horizon bid-ask bounce with half-life < 1h). Must retain >= 0% Sharpe.
3. **Null Hypothesis Test**: Shuffles asset returns across time. If the pipeline still produces high absolute Sharpe (|SR| >= 0.75), it indicates a structural data processing leak.
4. **Funding Regularity Check**: Validates point-in-time alignment of 8h settlement funding rates (hours 00, 08, 16 UTC).
5. **Execution & Friction Breakdown**: 1-bar execution delay, historical 8h `funding_rate`, and linear transaction cost grid (`0, 3, 5, 10 bps`) with explicit Gross PnL, Fee Drag, and Funding Drag accounting.
6. **Validation**: Purged 70/30 In-Sample / Out-of-Sample split (24h embargo) at `5.0 bps` taker fee. Requires `OOS Sharpe > 0` and `Full Sharpe @ 5 bps > 0`.
7. **Wick & Mark Integrity (`audit_wick_vulnerability`)**: Verifies that strategy alpha does not depend on unfillable liquidation wicks or transient bad marks. Caps the fraction of positive gross PnL earned immediately following anomalous wick bars to <= 20.0%.

## Research Hygiene vs. Execution Simulation

This repository is built as an **adversarial research audit harness** designed to falsify quantitative signals and catch structural leaks prior to production capital allocation. It is not an order-matching or LOB execution simulator.

### What the Harness Rigorously Verifies:
1. **Mathematical Causality**: Prefix-slice invariance ($<10^{-9}$ weight divergence) preventing centered rolling windows, future index shifts, or whole-sample normalization.
2. **Horizon & Microstructure Decay**: Stress-testing alpha under an incremental $+1$ bar delay ($t+2$). Distinguishes durable medium-horizon relative-value from fragile sub-hourly bid-ask bounce.
3. **Discrete 8h Funding Settlement**: Funding cash flows assessed strictly against the position held across the discrete settlement boundary (`w_prev_held` at 00, 08, 16 UTC).
4. **Liquidation Wick Vulnerability**: Gross PnL attribution gating ($\le 20\%$) to verify strategies do not harvest unfillable liquidation wicks or transient bad marks.
5. **Multiple Testing Correction**: Bailey & López de Prado Deflated Sharpe Ratio (DSR, $N=10$) penalizing trial count and non-Gaussian return distributions.

### Scope Boundaries (What is intentionally excluded):
- **Limit Order Book (LOB) Dynamics**: Does not simulate queue position, cancel-to-fill ratios, or maker rebate economics.
- **Non-Linear Market Impact**: Assumes conservative flat taker fees (5 bps) rather than square-root law / Almgren-Chriss impact scaling with ADV.
- **Adverse Selection**: Assumes instantaneous execution at bar transition rather than modeling toxic flow against resting orders.
- **Mark vs. Last Basis Drift**: Uses kline Close for return calculation, while exchange funding and margin calculations reference Mark Price.

