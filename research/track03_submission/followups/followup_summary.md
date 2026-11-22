# Exploratory follow-ups

These experiments followed inspection of the original strategy's training
results. They are exploratory and are not independent validation.
They supplement the original strategy rather than replacing its result.

## Candidate A: gradual catch-up

Specification:
- Retain the original ungated catch-up target and eight-hour decisions.
- Update weights as 0.75 * previous weights + 0.25 * target weights.
- Do not rescale the blended portfolio back to gross exposure one.
- Keep the other recorded settings unchanged.

Finding:
Trading intensity fell, but the remaining return opportunity did not cover
costs. The reported official audit had causality PASS across 66 prefix
cuts, 5bps Sharpe -0.72, and extra-hour-delay Sharpe -1.51.
The audit's later-period slice had Sharpe -3.25.

Conclusion:
Negative economic result. No adjustment-fraction search was performed
as part of this fixed follow-up experiment.

## Candidate B: slower momentum

Specification:
- Same 18-altcoin universe.
- Seven-day log return divided by trailing seven-day hourly volatility.
- Daily rebalancing.
- Dollar-neutral portfolio with gross exposure capped at one.
- No beta adjustment or flow condition.

Previously reported notebook result:
At 5bps with the extra-hour delay, cumulative additive net PnL was
approximately -1.22%, Sharpe -0.038, and daily turnover 0.528.
The reported Sharpe interval included zero.

Conclusion:
Trading economics improved substantially, but positive net profitability
was not established.

Official audit:
See candidate_b_audit.txt and candidate_b_audit.jsonl.
Review their actual verdict and metrics before finalizing the PR.

## Interpretation limits

All comparisons use already-inspected training data. Preserve both
experiments and their uncertainty. Do not select favorable fee rows,
periods, or parameters after seeing the outcomes.
