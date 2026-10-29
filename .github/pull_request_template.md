## Summary & Hypothesis
<!-- 2-4 sentences describing the mathematical/microstructure mechanism or infrastructure change -->

## Local Verification Checklist
- [ ] `python -m pytest -v` passes locally with zero failures.
- [ ] If submitting/updating a strategy in `strategies/<name>.py`, `python -m src.audit strategies/<name>.py` passes all 7 causal stages.
- [ ] Zero procedural row-loops (`iterrows` / `itertuples`); strictly vectorized NumPy/Pandas.

## Audit Scorecard Output
<!-- Paste the terminal output of `python -m src.audit strategies/<your_strategy>.py` below (if applicable) -->
```text

```

## Empirical Notes (for `reports/sprint_01_paper.md`)
<!-- Optional for Activation PRs, required for final Strategy PRs: 80-120 words on signal decay, turnover sensitivity, and behavior under 5.0 bps taker fees -->
