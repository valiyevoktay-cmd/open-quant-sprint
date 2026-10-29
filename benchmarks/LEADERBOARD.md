# Syndicate Sprint 01 — Reproducible Benchmark Leaderboard

All strategies are evaluated via `python -m src.audit <strategy_file>` on `data/train_1h.parquet` (20 Binance perpetual futures, 69,120 hourly rows) under a mandatory **1-bar execution delay (`lag_bars=1`)**, **5.0 bps taker fee**, **8-hour funding rate accounting**, and **70/30 Purged Walk-Forward Split (24h embargo)**.

Final out-of-sample evaluation against the sealed `data/holdout_1h.parquet` vault is executed offline at the conclusion of the sprint (November 22).

## Active Strategy Benchmark (`data/train_1h.parquet` @ 5.0 bps Taker Fee)

| Strategy Module | Contributor | Causal Check (`tol=1e-9`) | Full Sharpe (5 bps) | Purged IS Sharpe | Purged OOS Sharpe | DSR ($N=10$) | MaxDD | Mean Turnover | Lag-2 Retention | Runtime (ms) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`strategies/baseline_flow.py`](../strategies/baseline_flow.py) | Baseline | `0.00e+00` | `+3.15` | `+2.98` | `+4.19` | `0.901` | `-8.15%` | `0.0504` | `99.7%` | `106.2 ms` | **PASS** |
| `strategies/janice_diffusion.py` | Janice Jizhe | — | — | — | — | — | — | — | — | — | *Pending* |
| `strategies/edwin_statarb.py` | Edwin Gregory | — | — | — | — | — | — | — | — | — | *Pending* |
| `strategies/krishhiv_flow_alpha.py` | Krishhiv Mehra | — | — | — | — | — | — | — | — | — | *Pending* |
| `strategies/aarya_regime_sync.py` | Aarya Pakhale | — | — | — | — | — | — | — | — | — | *Pending* |

## Core Infrastructure & Compute Benchmarks

| Module / Track | Contributor | Target Invariant / Metric | Verification Command | Status |
| :--- | :--- | :--- | :--- | :---: |
| `src/quality.py` & `.github/workflows/ci.yml` | Yousof Sagr Shaladi | Gapless grid (`DQS >= 99.9`), timestamp monotonicity, automated PR scorecard | `python -m pytest -v` | *Pending* |
| `src/features.py` (Vectorized Kernels) | Ki Beom Kim | Zero Python loops in rolling/fractional kernels, exact numerical equivalence (`atol=1e-10`), $< 50\text{ ms}$ execution | `python -m pytest -v` | *Pending* |
