#### `angelo_closeloc.py`

| Check | Result | Value |
| --- | --- | --- |
| Verdict | **REJECT** | audit v1.3.3, commit 186a4af |
| Data | | panel `data/train_1h.parquet` (`d68e6159bbdfb2e3`), holdout `data/holdout_1h.parquet` (`ae0d5463b45dfe0d`) |
| Causality | pass | `Causal invariance verified across 66 prefix cuts (max diff=0.00e+00, seed=2599252551)` |
| Sharpe @ 5 bps | **FAIL** | -17.51 (needs > 0), 95% CI [-19.25, -15.95] |
| Out-of-sample Sharpe | **FAIL** | -16.30 (needs > 0), 95% CI [-18.96, -13.69] |
| Lag test | **FAIL** | 2-bar lag Sharpe -19.20, 1-bar -17.51; with a 1-bar Sharpe at or below 0 the test passes if the 2-bar Sharpe is no more than 0.5 lower |
| Null check | pass | mean Sharpe +0.01, sd 0.72 |
| Wick share | pass | 2.8% of positive PnL |
| Turnover @ 5 bps | pass | 0.7232 |
| Holdout | pass | -12.73, 95% CI [-18.82, -7.55], `bar-by-bar evaluation` |
| Sharpe by fee | | 0 bps +1.48, 3 bps -9.93, 5 bps -17.51, 10 bps -36.30, tiered ~7.7 bps -28.15 |
| Break-even fee | | about 0.39 bps, interpolated between the 0 and 3 bps rows |
| PnL @ 5 bps | | gross +42.72%, fee cost +537.77%, funding drag +0.94%, net -495.99% |
| Correlation with baseline | | +0.09, hourly net returns at 5 bps |

#### `arhaan_contrarian.py`

| Check | Result | Value |
| --- | --- | --- |
| Verdict | **REJECT** | audit v1.3.3, commit 186a4af |
| Data | | panel `data/train_1h.parquet` (`d68e6159bbdfb2e3`), holdout `data/holdout_1h.parquet` (`ae0d5463b45dfe0d`) |
| Causality | pass | `Causal invariance verified across 66 prefix cuts (max diff=0.00e+00, seed=651713992)` |
| Sharpe @ 5 bps | **FAIL** | -1.66 (needs > 0), 95% CI [-3.09, -0.25] |
| Out-of-sample Sharpe | **FAIL** | -4.64 (needs > 0), 95% CI [-7.21, -2.06] |
| Lag test | pass | 2-bar lag Sharpe -1.60, 1-bar -1.66; with a 1-bar Sharpe at or below 0 the test passes if the 2-bar Sharpe is no more than 0.5 lower |
| Null check | pass | mean Sharpe -0.34, sd 0.72 |
| Wick share | pass | 2.8% of positive PnL |
| Turnover @ 5 bps | pass | 0.0825 |
| Holdout | pass | +0.26, 95% CI [-6.01, +7.25] includes 0, `bar-by-bar evaluation` |
| Sharpe by fee | | 0 bps -0.17, 3 bps -1.06, 5 bps -1.66, 10 bps -3.13, tiered ~7.7 bps -2.49 |
| Break-even fee | | < 0 bps (Sharpe not above 0 at 0 bps) |
| PnL @ 5 bps | | gross -10.15%, fee cost +61.37%, funding drag -2.96%, net -68.55% |
| Correlation with baseline | | -0.38, hourly net returns at 5 bps |

#### `arnav_carry.py`

| Check | Result | Value |
| --- | --- | --- |
| Verdict | **REJECT** | audit v1.3.3, commit 186a4af |
| Data | | panel `data/train_1h.parquet` (`d68e6159bbdfb2e3`), holdout `data/holdout_1h.parquet` (`ae0d5463b45dfe0d`) |
| Causality | pass | `Causal invariance verified across 65 prefix cuts (max diff=0.00e+00, seed=2767611532)` |
| Sharpe @ 5 bps | pass | +0.41, 95% CI [-1.08, +1.80] includes 0 |
| Out-of-sample Sharpe | **FAIL** | -0.72 (needs > 0), 95% CI [-3.50, +1.93] includes 0 |
| Lag test | pass | 2-bar lag Sharpe +0.32, 1-bar +0.41 |
| Null check | **FAIL** | mean Sharpe -0.76, sd 0.61 |
| Wick share | pass | 2.1% of positive PnL |
| Turnover @ 5 bps | pass | 0.0162 |
| Holdout | pass | -0.79, 95% CI [-6.35, +4.39] includes 0, `bar-by-bar evaluation` |
| Sharpe by fee | | 0 bps +0.94, 3 bps +0.63, 5 bps +0.41, 10 bps -0.12, tiered ~7.7 bps +0.13 |
| Break-even fee | | about 8.87 bps, interpolated between the 5 and 10 bps rows |
| PnL @ 5 bps | | gross +12.59%, fee cost +12.04%, funding drag -8.85%, net +9.40% |
| Correlation with baseline | | +0.16, hourly net returns at 5 bps |

#### `baseline_flow.py`

| Check | Result | Value |
| --- | --- | --- |
| Verdict | **REJECT** | audit v1.3.3, commit 186a4af |
| Data | | panel `data/train_1h.parquet` (`d68e6159bbdfb2e3`), holdout `data/holdout_1h.parquet` (`ae0d5463b45dfe0d`) |
| Causality | pass | `Causal invariance verified across 66 prefix cuts (max diff=0.00e+00, seed=2859379327)` |
| Sharpe @ 5 bps | **FAIL** | -1.53 (needs > 0), 95% CI [-3.00, +0.07] includes 0 |
| Out-of-sample Sharpe | pass | +2.17, 95% CI [-0.52, +4.86] includes 0 |
| Lag test | pass | 2-bar lag Sharpe -1.53, 1-bar -1.53; with a 1-bar Sharpe at or below 0 the test passes if the 2-bar Sharpe is no more than 0.5 lower |
| Null check | pass | mean Sharpe -0.00, sd 0.74 |
| Wick share | pass | 2.2% of positive PnL |
| Turnover @ 5 bps | pass | 0.0522 |
| Holdout | pass | +1.31, 95% CI [-4.81, +8.03] includes 0, `bar-by-bar evaluation` |
| Sharpe by fee | | 0 bps -0.12, 3 bps -0.97, 5 bps -1.53, 10 bps -2.95, tiered ~7.7 bps -2.29 |
| Break-even fee | | < 0 bps (Sharpe not above 0 at 0 bps) |
| PnL @ 5 bps | | gross -3.96%, fee cost +38.80%, funding drag -0.77%, net -42.00% |

#### `hongyu_zhu_strategy.py`

| Check | Result | Value |
| --- | --- | --- |
| Verdict | PASS | audit v1.3.3, commit 186a4af |
| Data | | panel `data/train_1h.parquet` (`d68e6159bbdfb2e3`), holdout `data/holdout_1h.parquet` (`ae0d5463b45dfe0d`) |
| Causality | pass | `Causal invariance verified across 66 prefix cuts (max diff=0.00e+00, seed=2622140154)` |
| Sharpe @ 5 bps | pass | +1.18, 95% CI [-0.54, +2.70] includes 0 |
| Out-of-sample Sharpe | pass | +0.23, 95% CI [-2.40, +3.05] includes 0 |
| Lag test | pass | 2-bar lag Sharpe +1.26, 1-bar +1.18 |
| Null check | pass | mean Sharpe +0.22, sd 0.79 |
| Wick share | pass | 2.1% of positive PnL |
| Turnover @ 5 bps | pass | 0.0034 |
| Holdout | pass | -4.07, 95% CI [-9.53, +2.71] includes 0, `bar-by-bar evaluation` |
| Sharpe by fee | | 0 bps +1.39, 3 bps +1.26, 5 bps +1.18, 10 bps +0.96, tiered ~7.7 bps +1.06 |
| Break-even fee | | > 10 bps (Sharpe above 0 in every fee row) |
| PnL @ 5 bps | | gross +16.47%, fee cost +2.52%, funding drag -0.09%, net +14.05% |
| Correlation with baseline | | +0.11, hourly net returns at 5 bps |

#### `janice_gradual_catchup.py`

| Check | Result | Value |
| --- | --- | --- |
| Verdict | **REJECT** | audit v1.3.3, commit 186a4af |
| Data | | panel `data/train_1h.parquet` (`d68e6159bbdfb2e3`), holdout `data/holdout_1h.parquet` (`ae0d5463b45dfe0d`) |
| Causality | pass | `Causal invariance verified across 66 prefix cuts (max diff=0.00e+00, seed=1771864691)` |
| Sharpe @ 5 bps | **FAIL** | -0.72 (needs > 0), 95% CI [-2.17, +0.80] includes 0 |
| Out-of-sample Sharpe | **FAIL** | -3.25 (needs > 0), 95% CI [-5.61, -0.80] |
| Lag test | **FAIL** | 2-bar lag Sharpe -1.51, 1-bar -0.72; with a 1-bar Sharpe at or below 0 the test passes if the 2-bar Sharpe is no more than 0.5 lower |
| Null check | pass | mean Sharpe -0.17, sd 0.76 |
| Wick share | pass | 2.8% of positive PnL |
| Turnover @ 5 bps | pass | 0.0318 |
| Holdout | pass | -2.37, 95% CI [-8.28, +3.84] includes 0, `bar-by-bar evaluation` |
| Sharpe by fee | | 0 bps +0.91, 3 bps -0.07, 5 bps -0.72, 10 bps -2.35, tiered ~7.7 bps -1.71 |
| Break-even fee | | about 2.79 bps, interpolated between the 0 and 3 bps rows |
| PnL @ 5 bps | | gross +12.06%, fee cost +23.62%, funding drag -1.11%, net -10.45% |
| Correlation with baseline | | -0.35, hourly net returns at 5 bps |

#### `janice_leadlag.py`

| Check | Result | Value |
| --- | --- | --- |
| Verdict | **REJECT** | audit v1.3.3, commit 186a4af |
| Data | | panel `data/train_1h.parquet` (`d68e6159bbdfb2e3`), holdout `data/holdout_1h.parquet` (`ae0d5463b45dfe0d`) |
| Causality | pass | `Causal invariance verified across 66 prefix cuts (max diff=0.00e+00, seed=175291182)` |
| Sharpe @ 5 bps | **FAIL** | -2.14 (needs > 0), 95% CI [-3.66, -0.63] |
| Out-of-sample Sharpe | **FAIL** | -2.56 (needs > 0), 95% CI [-4.67, -0.37] |
| Lag test | pass | 2-bar lag Sharpe -2.23, 1-bar -2.14; with a 1-bar Sharpe at or below 0 the test passes if the 2-bar Sharpe is no more than 0.5 lower |
| Null check | pass | mean Sharpe -0.05, sd 0.73 |
| Wick share | pass | 3.0% of positive PnL |
| Turnover @ 5 bps | pass | 0.0218 |
| Holdout | pass | +0.24, 95% CI [-4.30, +5.25] includes 0, `bar-by-bar evaluation` |
| Sharpe by fee | | 0 bps -0.62, 3 bps -1.54, 5 bps -2.14, 10 bps -3.61, tiered ~7.7 bps -3.05 |
| Break-even fee | | < 0 bps (Sharpe not above 0 at 0 bps) |
| PnL @ 5 bps | | gross -6.91%, fee cost +16.25%, funding drag -0.25%, net -22.91% |
| Correlation with baseline | | -0.06, hourly net returns at 5 bps |
| Review flags | | `module-level mutable state (lines 121)` (manual review, not part of the verdict) |

#### `janice_slow_momentum.py`

| Check | Result | Value |
| --- | --- | --- |
| Verdict | **REJECT** | audit v1.3.3, commit 186a4af |
| Data | | panel `data/train_1h.parquet` (`d68e6159bbdfb2e3`), holdout `data/holdout_1h.parquet` (`ae0d5463b45dfe0d`) |
| Causality | pass | `Causal invariance verified across 66 prefix cuts (max diff=0.00e+00, seed=2576475878)` |
| Sharpe @ 5 bps | **FAIL** | -0.13 (needs > 0), 95% CI [-1.61, +1.44] includes 0 |
| Out-of-sample Sharpe | pass | +1.24, 95% CI [-1.57, +4.30] includes 0 |
| Lag test | pass | 2-bar lag Sharpe -0.04, 1-bar -0.13; with a 1-bar Sharpe at or below 0 the test passes if the 2-bar Sharpe is no more than 0.5 lower |
| Null check | pass | mean Sharpe +0.44, sd 0.77 |
| Wick share | pass | 2.3% of positive PnL |
| Turnover @ 5 bps | pass | 0.0209 |
| Holdout | pass | -0.04, 95% CI [-6.89, +6.87] includes 0, `bar-by-bar evaluation` |
| Sharpe by fee | | 0 bps +0.34, 3 bps +0.06, 5 bps -0.13, 10 bps -0.59, tiered ~7.7 bps -0.40 |
| Break-even fee | | about 3.63 bps, interpolated between the 3 and 5 bps rows |
| PnL @ 5 bps | | gross +14.20%, fee cost +15.56%, funding drag +2.82%, net -4.19% |
| Correlation with baseline | | +0.35, hourly net returns at 5 bps |

#### `leaky_example.py`

| Check | Result | Value |
| --- | --- | --- |
| Verdict | **REJECT** | audit v1.3.3, commit 186a4af |
| Data | | panel `data/train_1h.parquet` (`d68e6159bbdfb2e3`), holdout `data/holdout_1h.parquet` (`ae0d5463b45dfe0d`) |
| Causality | **FAIL** | `Look-ahead leak detected at 2025-07-13 06:00:00+00:00: max\|w_trunc - w_full\| = 3.650e+00 > 1.0e-09 (seed=2117361044)` |
| Sharpe @ 5 bps | **FAIL** | -9.21 (needs > 0), 95% CI [-10.85, -7.56] |
| Out-of-sample Sharpe | **FAIL** | -13.46 (needs > 0), 95% CI [-16.52, -10.88] |
| Lag test | **FAIL** | 2-bar lag Sharpe -10.33, 1-bar -9.21; with a 1-bar Sharpe at or below 0 the test passes if the 2-bar Sharpe is no more than 0.5 lower |
| Null check | pass | mean Sharpe -0.11, sd 0.83 |
| Wick share | pass | 0.9% of positive PnL |
| Turnover @ 5 bps | pass | 1.3648 |
| Holdout | **FAIL** | `Holdout look-ahead: one-shot weights differ from bar-by-bar weights (max diff=9.912e-02)` |
| Sharpe by fee | | 0 bps -0.19, 3 bps -5.60, 5 bps -9.21, 10 bps -18.21, tiered ~7.7 bps -14.13 |
| Break-even fee | | < 0 bps (Sharpe not above 0 at 0 bps) |
| PnL @ 5 bps | | gross -21.12%, fee cost +1014.79%, funding drag -0.21%, net -1035.70% |
| Correlation with baseline | | -0.05, hourly net returns at 5 bps |
| Review flags | | `negative shift or centered window (lines 20)` (manual review, not part of the verdict) |

#### `tanmay_flow.py`

| Check | Result | Value |
| --- | --- | --- |
| Verdict | **REJECT** | audit v1.3.3, commit 186a4af |
| Data | | panel `data/train_1h.parquet` (`d68e6159bbdfb2e3`), holdout `data/holdout_1h.parquet` (`ae0d5463b45dfe0d`) |
| Causality | pass | `Causal invariance verified across 65 prefix cuts (max diff=0.00e+00, seed=3593459347)` |
| Sharpe @ 5 bps | **FAIL** | -8.45 (needs > 0), 95% CI [-10.12, -6.74] |
| Out-of-sample Sharpe | **FAIL** | -7.85 (needs > 0), 95% CI [-10.83, -4.72] |
| Lag test | pass | 2-bar lag Sharpe -8.78, 1-bar -8.45; with a 1-bar Sharpe at or below 0 the test passes if the 2-bar Sharpe is no more than 0.5 lower |
| Null check | pass | mean Sharpe +0.47, sd 0.76 |
| Wick share | pass | 2.1% of positive PnL |
| Turnover @ 5 bps | pass | 0.2736 |
| Holdout | pass | -7.52, 95% CI [-13.14, -1.39], `bar-by-bar evaluation` |
| Sharpe by fee | | 0 bps +0.00, 3 bps -5.07, 5 bps -8.45, 10 bps -16.86, tiered ~7.7 bps -13.27 |
| Break-even fee | | < 0 bps (Sharpe not above 0 at 0 bps) |
| PnL @ 5 bps | | gross +3.74%, fee cost +203.42%, funding drag +3.71%, net -203.39% |
| Correlation with baseline | | +0.55, hourly net returns at 5 bps |

#### `template.py`

| Check | Result | Value |
| --- | --- | --- |
| Verdict | **REJECT** | audit v1.3.3, commit 186a4af |
| Data | | panel `data/train_1h.parquet` (`d68e6159bbdfb2e3`), holdout `data/holdout_1h.parquet` (`ae0d5463b45dfe0d`) |
| Causality | pass | `Causal invariance verified across 66 prefix cuts (max diff=0.00e+00, seed=2879341817)` |
| Sharpe @ 5 bps | **FAIL** | -0.54 (needs > 0), 95% CI [-2.07, +0.96] includes 0 |
| Out-of-sample Sharpe | pass | +3.33, 95% CI [+0.69, +5.80] |
| Lag test | pass | 2-bar lag Sharpe -0.45, 1-bar -0.54; with a 1-bar Sharpe at or below 0 the test passes if the 2-bar Sharpe is no more than 0.5 lower |
| Null check | pass | mean Sharpe +0.21, sd 0.75 |
| Wick share | pass | 2.4% of positive PnL |
| Turnover @ 5 bps | pass | 0.0741 |
| Holdout | pass | -2.44, 95% CI [-9.72, +3.14] includes 0, `bar-by-bar evaluation` |
| Sharpe by fee | | 0 bps +1.12, 3 bps +0.12, 5 bps -0.54, 10 bps -2.21, tiered ~7.7 bps -1.44 |
| Break-even fee | | about 3.36 bps, interpolated between the 3 and 5 bps rows |
| PnL @ 5 bps | | gross +40.02%, fee cost +55.08%, funding drag +2.86%, net -17.92% |
| Correlation with baseline | | +0.57, hourly net returns at 5 bps |
