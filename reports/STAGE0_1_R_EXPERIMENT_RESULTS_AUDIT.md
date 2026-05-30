# Stage 0.1 R-Experiment Results Audit

Date: 2026-05-27

This file tracks the corrected post-audit R-experiments. It is separate from
`reports/STAGE0_1_ALL_EXPERIMENT_RESULTS_AUDIT.md`, which covers the original
Batch 1-12 experiments.

## R1 Direct K-Window Results

Artifacts analyzed:

```text
artifacts/stage0_1/R1_root_K5_equal_slice_direct/*.zip
artifacts/stage0_1/R1_root_K20_equal_slice_direct/*.zip
```

Derived analysis files:

```text
reports/r1_k_window_analysis/r1_direct_fold_summaries.csv
reports/r1_k_window_analysis/r1_direct_variant_summary.csv
reports/r1_k_window_analysis/r1_direct_risk_cash_diagnostics.csv
reports/r1_k_window_analysis/r1_direct_zip_integrity.csv
```

### Code / Package Sanity

The analyzed result zips contain the expected artifacts:

```text
model.zip
validation_summary.csv
validation_daily.csv
metadata.json
sb3_logs/progress.csv
training_diagnostics/training_update_diagnostics.csv
training_diagnostics/training_sample_diagnostics.csv
rollout_snapshot_*.npz
feature_scalers/fold_*/...
```

Integrity summary:

| Variant | Fold Count | Validation Daily Rows | Training Budget | Snapshot Count |
|---|---:|---:|---:|---:|
| `R1_root_K5_equal_slice_direct` | 4/4 | 250-252 per fold | 70,000 macro steps | 69 per fold |
| `R1_root_K20_equal_slice_direct` | 4/4 | 250-252 per fold | 17,500 macro steps | 18 per fold |

The smaller K20 result size is expected: its macro-step count and rollout snapshot
count are roughly one quarter of K5 because each action covers 20 trading days
instead of 5.

Required R1 diagnostic columns are present:

```text
q_anchor
q_scheduled
cash_anchor
cash_scheduled
anchor_to_schedule_l1
schedule_to_exec_l1
anchor_weight_*
scheduled_weight_*
executed_weight_*
```

### Idea

R1 tests whether daily high-level `cash/risk` intent is too frequent. The policy
samples a high-level anchor only once per K trading days:

```text
q_anchor ~ Beta
u_anchor ~ Dirichlet(29)
```

The environment then executes deterministically inside the K-day window by direct
equal-slice scheduling toward the frozen anchor.

### Problem / Hypothesis

Original root-split runs were cash-heavy. One hypothesis was that the agent receives
and updates high-level intent too often for a small stable Dow-like universe. If
portfolio ideas emerge weekly/monthly rather than daily, daily root actions may
encourage repeated cash drift and action chatter.

R1 asks:

```text
Does lower-frequency high-level root intent reduce cash lock-in and turnover without
destroying validation performance?
```

### Aggregate Results

| Variant | Folds | Mean Return | Mean Sharpe | Std Sharpe | Selection Score | Mean Drawdown | Mean Turnover | Mean Cash | Mean q_anchor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `R1_root_K5_equal_slice_direct` | 4 | 3.14% | 1.056 | 0.810 | 0.651 | -3.32% | 0.0164 | 80.33% | 19.83% |
| `R1_root_K20_equal_slice_direct` | 4 | 8.51% | 1.059 | 0.935 | 0.592 | -9.60% | 0.0078 | 43.41% | 58.76% |

Fold-level highlights:

| Variant | Fold | Return | Sharpe | Drawdown | Turnover | Cash |
|---|---|---:|---:|---:|---:|---:|
| `R1_root_K5_equal_slice_direct` | 2018 | 0.79% | 0.281 | -3.13% | 0.0184 | 82.24% |
| `R1_root_K5_equal_slice_direct` | 2019 | 4.79% | 2.126 | -1.18% | 0.0137 | 80.46% |
| `R1_root_K5_equal_slice_direct` | 2020 | 3.61% | 0.609 | -7.14% | 0.0194 | 82.08% |
| `R1_root_K5_equal_slice_direct` | 2021 | 3.39% | 1.208 | -1.82% | 0.0139 | 76.55% |
| `R1_root_K20_equal_slice_direct` | 2018 | 0.02% | 0.053 | -9.70% | 0.0078 | 43.31% |
| `R1_root_K20_equal_slice_direct` | 2019 | 13.91% | 2.101 | -3.33% | 0.0076 | 44.83% |
| `R1_root_K20_equal_slice_direct` | 2020 | 9.76% | 0.529 | -21.96% | 0.0088 | 39.98% |
| `R1_root_K20_equal_slice_direct` | 2021 | 10.33% | 1.553 | -3.43% | 0.0072 | 45.52% |

### Critical Result

R1 is a mixed result:

```text
K=5 refutes the hypothesis that weekly sticky root intent alone fixes cash.
K=20 supports the hypothesis that lower-frequency intent materially changes cash behavior.
```

K5 is not a usable rescue. It makes the root policy even more cash-heavy:

```text
mean cash ≈ 80.3%
mean q_anchor ≈ 19.8%
```

K20 is more interesting:

```text
mean cash drops to ≈ 43.4%
mean q_anchor rises to ≈ 58.8%
turnover drops to ≈ 0.0078
mean return rises to ≈ 8.5%
```

But K20 introduces a clear risk:

```text
2020 drawdown worsens to -21.96%
```

This is exactly the expected failure mode of a long sticky window: the policy cannot
react quickly enough during sharp regime transitions.

### Interpretation

The old `q_min=0.50` floor was not the only cause of bad cash behavior. Once the
floor is removed, K5 collapses into even more cash. That means the root policy still
has a strong incentive or local optimum toward high cash.

However, K20 shows that changing decision frequency changes learned root behavior
substantially. The model no longer sits near all-cash; it deploys much more capital
and trades less. This supports the broader thesis:

```text
cash/risk control is a time-scale problem, not only a distribution-shape problem.
```

The problem is that a fixed monthly window is too slow in stress periods.

### Hypothesis Status

| Hypothesis | Status | Evidence |
|---|---|---|
| Lower-frequency root intent reduces turnover. | Supported. | K20 turnover is about half K5: `0.0078` vs `0.0164`. |
| Lower-frequency root intent reduces cash lock-in. | Supported for K20, refuted for K5. | K20 cash `43.4%`; K5 cash `80.3%`. |
| Fixed K-window is sufficient. | Refuted. | K20 has severe 2020 drawdown. |
| Removing `q_min=0.50` alone fixes root cash. | Refuted. | K5 collapses to very high cash after q_min removal. |

### New Hypothesis

The next hypothesis should not be “use fixed K”. It should be:

```text
Use a slower default high-level root cadence, but allow event-triggered recovery
or re-risk/de-risk refresh when market conditions change quickly.
```

This points directly to R2:

```text
R2a/R2b: K-window equal-slice + confidence stop/recovery
```

Expected improvement:

```text
keep K20's lower cash and lower turnover
reduce K20's 2020 delayed-reaction drawdown
```

### Decision

Do not use K5 direct as a teacher candidate.

Keep K20 direct as a promising but incomplete baseline. It is the best evidence so
far that K-window root intent can repair part of the root cash behavior, but it
requires event-triggered confidence logic before it can be considered robust.

## R1b And R2b Results

Artifacts analyzed:

```text
artifacts/stage0_1/R1b_root_K5_equal_error_PD/*.zip
artifacts/stage0_1/R1b_root_K20_equal_error_PD/*.zip
artifacts/stage0_1/R2b_root_K20_equal_slice_stop_recovery/*.zip
```

Baseline R1 direct artifacts were also re-read for comparison.

Derived analysis files:

```text
reports/r_k_window_analysis/r1b_r2b_fold_summaries.csv
reports/r_k_window_analysis/r1b_r2b_variant_summary.csv
reports/r_k_window_analysis/r1b_r2b_zip_integrity.csv
reports/r_k_window_analysis/R1B_R2B_RESULT_AUDIT.md
```

### Code / Package Sanity

The result zips contain `metadata.json`, `validation_summary.csv`, and
`validation_daily.csv`. The authoritative variant name is the variant stored in
`metadata.json`, not the zip filename.

Important artifact caveat:

```text
12 zip filenames do not contain the metadata variant name.
```

Examples:

```text
R1b_root_K20_equal_error_PD folder contains zip files named R1_root_K5/K20...
R2b_root_K20_equal_slice_stop_recovery folder contains zip files named R1_root_K20...
```

This is a packaging/name bug, not an experiment-content bug: inside the zips,
`metadata.json` identifies the expected variants, and R2b validation daily logs
contain the required confidence fields.

R2b-specific logs are present:

```text
confidence_derisk
confidence_rerisk
risk_stress
recovery_score
recovery_trigger
risk_break_trigger
derisk_early_update
rerisk_early_update
suppressed_trade_l1
suppressed_trade_value
stop_reason
early_update_reason
raw_weight_*
anchor_weight_*
scheduled_weight_*
target_weight_*
executed_weight_*
```

### Idea

R1b tests whether the frozen K-window anchor should still pass through the
existing P/PD execution controller instead of direct equal-slice execution.

R2b tests whether K20 direct equal-slice can be improved by a deterministic
confidence layer:

```text
stop today's scheduled trade if direction-specific confidence is too low
close the K-window early on recovery or risk-break events
```

The two early-update channels are logged separately:

```text
re-risk early update: recovery_trigger
de-risk early update: risk_break_trigger
```

### Problem / Hypothesis

R1 showed a tradeoff:

```text
K5 direct: stable drawdown but collapses into very high cash
K20 direct: better capital deployment and lower turnover, but delayed reaction in 2020
```

R1b asks:

```text
Does P/PD execution make K-window anchors less mechanical and less cash-heavy?
```

R2b asks:

```text
Can event-triggered confidence logic keep K20's low turnover while fixing delayed
reaction and cash lock-in?
```

### Aggregate Results

| Variant | Folds | Selection Score | Mean Sharpe | Std Sharpe | Mean Return | Mean Drawdown | Mean Turnover | Mean Cash | Mean q_anchor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `R1b_root_K5_equal_error_PD` | 4 | 0.654 | 1.088 | 0.868 | 7.24% | -7.28% | 0.0094 | 54.36% | 46.73% |
| `R1_root_K5_equal_slice_direct` | 4 | 0.651 | 1.056 | 0.810 | 3.14% | -3.32% | 0.0164 | 80.33% | 19.83% |
| `R2b_root_K20_equal_slice_stop_recovery` | 4 | 0.626 | 0.999 | 0.747 | 7.37% | -7.64% | 0.0055 | 50.20% | 54.76% |
| `R1b_root_K20_equal_error_PD` | 4 | 0.603 | 1.037 | 0.869 | 9.03% | -9.62% | 0.0075 | 41.43% | 63.43% |
| `R1_root_K20_equal_slice_direct` | 4 | 0.592 | 1.059 | 0.935 | 8.51% | -9.60% | 0.0078 | 43.41% | 58.76% |

R2b event behavior:

| Fold | Return | Sharpe | Drawdown | Turnover | Cash | Effective K | Stop Rate | Early Update Rate | Recovery Rate | Risk Break Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018 | 0.46% | 0.097 | -9.03% | 0.0058 | 48.23% | 7.91 | 4.4% | 32.0% | 12.0% | 20.0% |
| 2019 | 10.01% | 1.749 | -3.07% | 0.0051 | 52.98% | 10.29 | 8.4% | 17.1% | 12.7% | 4.4% |
| 2020 | 10.30% | 0.698 | -15.08% | 0.0059 | 49.50% | 2.27 | 10.3% | 71.8% | 39.3% | 34.5% |
| 2021 | 8.69% | 1.454 | -3.40% | 0.0051 | 50.10% | 2.70 | 2.0% | 60.6% | 58.2% | 2.4% |

### Critical Result

R1b K5 is the best selection-score variant among this group, but only by a very
small margin:

```text
R1b K5 selection score: 0.654
R1 K5 direct selection score: 0.651
```

The real improvement is not Sharpe stability; it is behavior:

```text
cash falls from 80.3% to 54.4%
return rises from 3.1% to 7.2%
turnover falls from 0.0164 to 0.0094
```

But drawdown worsens:

```text
-3.3% -> -7.3%
```

R1b K20 is only a small improvement over R1 K20:

```text
selection score: 0.603 vs 0.592
return: 9.0% vs 8.5%
cash: 41.4% vs 43.4%
turnover: 0.0075 vs 0.0078
drawdown: almost unchanged
```

R2b improves K20 risk behavior and turnover, but it does not preserve the intended
20-day decision cadence:

```text
mean effective K = 5.79 days
mean early update rate = 45.4%
2020 effective K = 2.27 days
2021 effective K = 2.70 days
```

This means R2b often collapses the nominal monthly policy into an event-triggered
near-daily or few-day policy. That is not automatically bad, but it changes the
claim: R2b no longer tests "monthly policy plus occasional recovery"; it tests
"adaptive event-triggered root updates".

### Interpretation

R1b supports the idea that execution dynamics matter. Passing a frozen anchor
through P/PD makes K5 much less cash-collapsed than direct K5. This suggests that
the direct equal-slice scheduler can be too rigid and can interact badly with the
root distribution.

R2b supports the idea that event triggers can reduce delayed-reaction risk:

```text
2020 drawdown improves from K20 direct -21.96% to R2b -15.08%
turnover drops from K20 direct 0.0078 to R2b 0.0055
```

But R2b also shows over-triggering:

```text
2020 early update rate = 71.8%
2021 early update rate = 60.6%
```

So the confidence rules are too loose for preserving the K20 design. In bull or
recovery regimes, the recovery trigger fires very often and increases cash
relative to K20 direct/PD, hurting 2019 and 2021 upside.

### Hypothesis Status

| Hypothesis | Status | Evidence |
|---|---|---|
| P/PD execution improves K-window behavior. | Supported, especially for K5. | K5 PD lowers cash and turnover while raising return. |
| K20 P/PD is materially better than K20 direct. | Weakly supported. | Metrics improve only slightly. |
| Confidence stop/recovery fixes K20 delayed-reaction risk. | Partly supported. | 2020 drawdown improves, turnover falls. |
| R2b preserves a monthly high-level cadence. | Refuted. | Effective K collapses to 2-3 days in 2020/2021. |
| Current R2b thresholds are production-ready. | Refuted. | Early updates are too frequent. |

### New Hypothesis

The next version should keep event-triggered updates, but make them less trigger-happy:

```text
R2c:
    keep K20 default
    add cooldown after early update
    raise recovery threshold
    require recovery persistence for N days
    separate recovery trigger from trade-stop confidence
```

Candidate changes:

```text
recovery_trigger_threshold: 0.70 -> 0.80/0.85
risk_break_trigger_threshold: 0.80 -> 0.85/0.90
minimum event cooldown: 3-5 trading days
recovery persistence: 2 consecutive days
cash condition: current_cash > 0.20 remains
```

Also test:

```text
R1b K5 PD as a serious baseline, not only R1 direct.
```

R1b K5 is not obviously the final answer because it still holds 54% cash, but it is
the first K5 variant that materially breaks the all-cash behavior.

### Decision

Do not treat R2b as a clean success. It helps some risk metrics but violates the
intended K20 cadence by triggering too often.

Keep these candidates:

```text
R1b_root_K5_equal_error_PD
R1b_root_K20_equal_error_PD
R2c_K20_stop_recovery_with_cooldown_and_persistence
```

Reject the current R2b thresholds as too loose.
