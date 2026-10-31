# Sprint 01 Rules (v1.1)

Changes to these rules are listed in `CHANGELOG.md`.

## Timeline (UTC)

| Date | Event |
| --- | --- |
| Nov 2 | Repository access. Activation PRs (features, infrastructure, tests) can be merged. |
| Nov 2 - Nov 8 | Strategy PRs may be opened as drafts. They are not merged before Nov 9. |
| Nov 9 | Official start. Strategy PRs are merged once CI and review pass. |
| Nov 21, 23:59 | Code freeze: tag `sprint-01-freeze`, SHA-256 of every strategy file recorded in `benchmarks/freeze.txt`. |
| Nov 22 | Preliminary report on the holdout period (Sep 12 - Oct 18). |
| Dec 7 | Final report on the forward period (Nov 22 - Dec 6), computed on the frozen code. Date confirmed with the cohort by email. |

## Strategy contract

- `strategies/<name>.py` defines `compute_weights(df: pd.DataFrame) -> pd.Series` aligned with `df.index`.
- The weight at a row may depend only on rows with the same or earlier `timestamp`.
- The function uses only the `df` it receives. No file, network or environment access, no imports of
  `src.audit`, `src.backtest` or `src.sandbox`, no state carried between calls.
- Output must be deterministic for identical input.
- Each audit call runs in a separate Python process started in an empty temporary directory.
  The audit report lists code patterns that the maintainer reviews by hand (block [13]); every strategy
  is read before merge.

## Audit verdict

`python -m src.audit strategies/<name>.py` prints `[PASS] SURVIVED AUDIT` when all of the following hold:

1. Causality: weights on prefixes of the panel (the last full bar, 40, 65, 85, 95, 99 % and eight random
   cuts) match the full-panel weights within 1e-9; two runs on the same input are identical; weights are
   not all zero. The random cuts use a fresh seed on every run; the report prints it, and
   `--seed <n>` reproduces that run.
2. Data quality score >= 80.
3. Sharpe at 5 bps on the full panel > 0 and on the purged 30 % out-of-sample slice > 0.
4. Lag test: Sharpe with one extra hour of delay > 0.
5. Null pipeline check: mean Sharpe over 200 daily-block permutations of returns within +-0.75.
6. Wick check: at most 20 % of positive gross PnL comes from bars right after liquidation wicks.
7. When a holdout or forward file is passed, the strategy runs bar by bar on it without errors and the
   bar-by-bar weights equal the one-shot weights.

Sharpe ratios are reported with 95 % block-bootstrap intervals. Measured half-widths: about +-1.5
Sharpe points on the 20-month training panel, about +-6 on the 36-day holdout, about +-9 on the 14-day
forward period. A positive Sharpe on these samples does not establish an edge.

What PASS means: the strategy is causal on the tested cuts, reproducible, and has a positive point
estimate after costs. It is a hygiene gate, not evidence of an edge. In a stress test, 7 % of random
combinations of standard causal features passed, and none of them had an interval above zero. REJECT
is an equally valid research result.

The audit catches common look-ahead mistakes. It is not secure against a participant who sets out to
fool it; the rules rely on good faith. Known gaps are listed in [`LIMITATIONS.md`](LIMITATIONS.md).

## Official verdict

CI is a convenience for contributors, not the referee. The official verdict on every strategy is the
maintainer's own run:

- on the file whose SHA-256 is recorded in `benchmarks/freeze.txt` at the freeze, published before Nov 22;
- with the harness at tag `sprint-01-freeze` and the dependency versions in `requirements.lock`;
- with `--n-random-cuts 60`, so leaks that escape a few random cuts in one CI run are caught;
- on the forward period, in an environment where the strategy process can read neither data files nor
  the network, so it sees only the bars passed to it;
- after the maintainer has read the strategy file.

## Holdout and forward period

- The holdout file (Sep 12 - Oct 18) is built from public Binance data. It is treated as a public
  validation period: results are reported with intervals, and the Sharpe sign is not a pass/fail gate.
- The forward period starts after the code freeze, so no strategy can have been fitted to it.
  It is the only sealed test. Results are reported with intervals, without a pass/fail threshold.
- The maintainer runs each final evaluation once. No intermediate runs on either period.
- Results are reported per strategy (verdict, Sharpe, interval) next to the reference strategies, not
  as a ranking: with intervals of +-6 to +-9 Sharpe points, an ordering would mostly reflect noise.

## Co-authorship of the working paper

Co-authors are contributors with at least one merged PR that passes CI, and a documented contribution
(feature, infrastructure, strategy or review). It does not depend on the sign of a Sharpe ratio.

## License

Code contributions are licensed under the MIT License (`LICENSE`). Market data under `data/` is not
covered by it (`NOTICE.md`).

## Reporting a hole in the audit

Report it privately to the maintainer, not in a public issue. Confirmed reports are credited in
`CHANGELOG.md`.
