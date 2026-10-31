# Changelog

## v1.0 (2026-10-30)

Audit (`src/audit.py`, `src/sandbox.py`)
- Every strategy call from the CLI runs in a fresh Python process in an empty working directory.
- Causality check uses 8 prefix cuts (40, 65, 85, 95, 99 % and three random) instead of 3; prefixes run before the full panel.
- Holdout is evaluated bar by bar and compared with a one-shot run; a difference fails the audit.
- A crash on the holdout, or a missing `--holdout` file, fails the audit instead of being skipped.
- Null test replaced: 200 daily-block permutations, gate on the mean null Sharpe (was a single permutation, so the result depended on the seed). In-sample p-value reported for information.
- 95 % block-bootstrap intervals for every Sharpe ratio.
- Tiered cost row: BTC/ETH 5 bps, others 7 bps, 6 least liquid by trailing 30-day volume 10 bps.
- Report header shows audit version, commit and SHA-256 of the strategy and data files.
- Correlation with `baseline_flow.py` and manual review flags added to the report.

Data (`data/`)
- `train_1h.parquet` extended back to 2025-01-01 00:00 UTC: 14,871 hours, 297,420 rows. The previous 144 days are byte-identical to the earlier file; the holdout is unchanged.
- On the extended panel `baseline_flow.py` and `template.py` fail the audit. Neither was changed. The baseline weight fixture and the leaderboard were regenerated.
- Last holdout bar (2026-10-18 14:00 UTC) replaced with the final closed values from Binance; it had been downloaded before the bar closed.
- `build_dataset.py`: `--start`/`--end` for fixed windows, unclosed bars dropped, existing files are not overwritten without `--overwrite`, synthetic data goes to `data/synthetic/`.

Repository
- MIT `LICENSE`, `NOTICE.md` for market data, `RULES.md`, `src/contrib/` for activation-PR features.
- CI audits strategies changed in a pull request using the harness and data from `main`.
- `CODEOWNERS` covers tests, features, stats, reference strategies and benchmarks.
