# Syndicate Sprint 01 — Reproducible Benchmark Leaderboard

All strategies are evaluated via `python -m src.audit <strategy_file>` on `data/train_1h.parquet` (20 Binance perpetual futures, 297,420 hourly rows, 2025-01-01 to 2026-09-12) with **fills at the close of the signal bar (`lag_bars=1`)**, **5.0 bps taker fee**, **8-hour funding rate accounting**, and **70/30 Purged Walk-Forward Split (24h embargo)**.

Holdout (Sep 12 - Oct 18, public Binance data) is evaluated once on Nov 22 as a preliminary report. The final report uses the forward period Nov 22 - Dec 6 on code frozen at `sprint-01-freeze` (see `RULES.md`). Sharpe ratios are shown with 95 % block-bootstrap intervals; on these sample lengths an interval that includes 0 is the normal case.

## Active Strategy Benchmark (`data/train_1h.parquet` @ 5.0 bps Taker Fee)

| Strategy Module | Contributor | Causal Check (`tol=1e-9`) | Full Sharpe (5 bps) [95% CI] | Purged IS Sharpe | Purged OOS Sharpe | DSR ($N=10$) | MaxDD | Mean Turnover | Lag-2 Retention | Runtime (ms) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`strategies/baseline_flow.py`](../strategies/baseline_flow.py) | Baseline | `0.00e+00` | `-1.53` [-3.00, +0.07] | `-3.04` | `+2.17` | `0.000` | `-68.56%` | `0.0522` | n/a (lag-1 SR < 0) | `535 ms` | **REJECT** |
| `strategies/janice_diffusion.py` | Janice Jizhe | — | — | — | — | — | — | — | — | — | *Pending* |
| `strategies/edwin_statarb.py` | Edwin Gregory | — | — | — | — | — | — | — | — | — | *Pending* |
| `strategies/krishhiv_flow_alpha.py` | Krishhiv Mehra | — | — | — | — | — | — | — | — | — | *Pending* |
| `strategies/aarya_regime_sync.py` | Aarya Pakhale | — | — | — | — | — | — | — | — | — | *Pending* |

`baseline_flow.py` was designed on the last 144 days of the panel (Sharpe +3.15 at 5 bps there). It is kept unchanged as a reference and does not pass on the full history.

## Core Infrastructure & Compute Benchmarks

| Module / Track | Contributor | Target Invariant / Metric | Verification Command | Status |
| :--- | :--- | :--- | :--- | :---: |
| `src/quality.py` & `.github/workflows/ci.yml` | Yousof Sagr Shaladi | Gapless grid (`DQS >= 99.9`), timestamp monotonicity, automated PR scorecard | `python -m pytest -v` | *Pending* |
| `src/features.py` (Vectorized Kernels) | Ki Beom Kim | Zero Python loops in rolling/fractional kernels, exact numerical equivalence (`atol=1e-10`), $< 50\text{ ms}$ execution | `python -m pytest -v` | *Pending* |
