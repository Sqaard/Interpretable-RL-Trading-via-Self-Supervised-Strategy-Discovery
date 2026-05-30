# Heuristic Controller Search Methodology

Date: 2026-05-29

## Goal

Heuristic Controller Search is a cheap hypothesis-filtering layer before cloud PPO retraining.

It searches over execution/controller rules while using completed real experiments as ground truth calibration. It is meant to answer:

```text
Which controller ideas are promising enough to retrain?
Which replay wins are likely artifacts?
Which failure modes are recurring across real and replayed experiments?
```

It is not a replacement for PPO training.

## Evidence Levels

### Level 1: Logged Real Experiment

Completed Huawei/local run with actual PPO training and validation logs.

Use this as ground truth.

Examples:

- `E9_discovered_rescorr_dirtree_fixedpd_rerun`
- `E5_asym_speed_deadzone`
- `R3_conf_slice_v1`
- `R4_incremental_TRUE`
- `R5/R6 internaldays`

### Level 2: Exact Logged Replay Check

Replay reads logged `validation_daily.csv` and verifies `logged_original`.

If this does not match the real run, replay infrastructure is invalid.

### Level 3: Frozen-Intent Controller Replay

Replay keeps policy intent frozen:

```text
anchor/scheduled targets from logs
alternative controller
same market returns
same transaction-cost approximation
```

Useful for execution-only ideas, but it can overestimate improvements because policy would react differently after state changes.

### Level 4: Counterfactual Policy-Forward Replay

Next planned level.

Instead of freezing future anchors, load the trained policy and recompute anchors on replayed state:

```text
same market features
counterfactual previous_weights
counterfactual portfolio state
policy forward -> new anchor
candidate controller -> executed weights
```

This is still cheaper than retraining but closer to the environment.

## Current HCS v0

Implemented in:

- `scripts/heuristic_controller_search.py`

Outputs:

- `reports/r_k_window_analysis/heuristic_controller_search_v0_calibrated/hcs_candidate_registry.csv`
- `reports/r_k_window_analysis/heuristic_controller_search_v0_calibrated/hcs_ground_truth_registry.csv`
- `reports/r_k_window_analysis/heuristic_controller_search_v0_calibrated/hcs_replay_fold_summary.csv`
- `reports/r_k_window_analysis/heuristic_controller_search_v0_calibrated/hcs_replay_aggregate_summary.csv`
- `reports/r_k_window_analysis/heuristic_controller_search_v0_calibrated/hcs_replay_vs_ground_truth_calibration.csv`
- `reports/r_k_window_analysis/heuristic_controller_search_v0_calibrated/hcs_recommendations.csv`
- `reports/r_k_window_analysis/heuristic_controller_search_v0_calibrated/HCS_FOUNDATION_REPORT.md`

## Candidate Families In v0

### Baseline Controller

```text
scheduled_PD
anchor_PD
```

### Deadzone

```text
scheduled_PD_deadzone_0.005
scheduled_PD_deadzone_0.01
scheduled_PD_deadzone_0.02
```

### Global Top-K

Tunes:

```text
top_k_buy
top_k_sell
rotation_budget
```

Current grid:

```text
(3,3), (5,5), (8,8), (3,8), (5,8), (8,5), (8,3)
rotation = 0, 0.001, 0.0025
```

### Group-Aware Top-K

Tunes:

```text
group cap
group pressure weight
group capacity weight
group overweight sell weight
top_k_buy/top_k_sell
rotation_budget
```

This is a soft priority modifier, not a hard group budget. One group can still receive all selected flow if its stocks dominate global priority.

## Calibration Rule

If replay has an approximate real-run match and overpredicts it, HCS applies a penalty to that source/controller family.

Current example:

```text
R3 replay scheduled_PD_topk_b5_s5_rot00
  replay selection: 0.7853

Real R4_incremental_TRUE
  true selection:   0.6518

Overprediction:
  +0.1336
```

Therefore R3 global/group Top-K replay recommendations receive a `0.1336` calibration penalty.

This is the first step toward making replay learn from experiment history.

## Current HCS v0 Interpretation

For R3 source:

```text
Uncalibrated replay likes Top-K.
But real R4_INCREMENTAL_TRUE showed Top-K was overestimated.
After calibration, deadzone/scheduled PD becomes the safer next replay conclusion.
```

For R6 source:

```text
No real Top-K-vs-R6 calibration exists yet.
Replay suggests K=8, no rotation.
Group-aware does not beat plain global Top-K.
```

## What HCS Should Tune Next

### Confidence Coefficients

Tune:

```text
risk_stress weights
recovery_score weights
residual breadth weights
VIX shock/surprise weights
dispersion penalties
direction-specific market residual treatment
```

Guardrail:

```text
trigger score must be forward-calibrated on train traces
```

### Trigger Calls

Tune:

```text
recovery threshold
risk-break threshold
persistence
cooldown
anchor gap
direction-specific trigger permission
```

Guardrail:

```text
avoid trigger spam that collapses K-window into daily trading
```

### Group Construction

Tune:

```text
static sectors
train-only residual correlation groups
targetcov frozen-teacher groups
group count
min group size
monthly/semi-static updates
```

Guardrail:

```text
no validation/test leakage
no singleton topology unless explicitly allowed
```

### Feature Families

Tune:

```text
full features
compact22
delta/change-only features
residualized features
macro-only root features
stock-only risky features
```

Guardrail:

```text
do not mix feature-routing and controller changes in the same first-pass ablation
```

### Timing

Tune:

```text
root K-days
stock K-days
event-triggered early update
internal trading day budget
```

Guardrail:

```text
fixed number of internal trading days for fair PPO budget
```

## Implemented Next Level: Counterfactual Policy-Forward Replay

Implemented in:

```text
scripts/counterfactual_policy_forward_replay.py
```

This level loads trained PPO models and evaluates modified controllers while
policy actions are recomputed on the counterfactual trajectory.

Key outputs:

```text
reports/r_k_window_analysis/counterfactual_policy_forward_replay_r3_core
reports/r_k_window_analysis/counterfactual_policy_forward_replay_r6_core
```

Current evidence:

```text
R3:
    pf_original remains best.
    Post-hoc Top-K on R3 is not robust enough without retrain.

R6:
    wider Top-K and group-aware Top-K beat the original R6 controller
    in no-retrain policy-forward replay.
```

## Implemented Search Loop: Mutating HCS

Implemented in:

```text
scripts/hcs_policy_forward_search_loop.py
```

The loop:

```text
1. starts from seed controller rules;
2. evaluates them with counterfactual policy-forward replay;
3. writes trials.jsonl, summary.csv, fold_summary.csv, failure_tags.csv;
4. selects the best candidates by adjusted score;
5. mutates Top-K/group-aware parameters for the next generation.
```

First real run:

```text
reports/r_k_window_analysis/hcs_policy_forward_search_loop_r6_v1
```

Main result:

```text
Best R6 no-retrain candidate:
    pf_topk_b10_s12_rot0p0

Interpretation:
    The search moved away from narrow top5 routing.
    top5 variants were tagged unstable_across_folds.
    Wider buy/sell queues look more promising before full PPO retrain.
```

This is still a filter, not proof. Promotion rule:

```text
frozen replay -> policy-forward replay -> full PPO retrain
```

## Implemented Next-Level Mutators

Implemented in:

```text
scripts/hcs_policy_forward_search_loop.py
scripts/counterfactual_policy_forward_replay.py
```

Replay-safe mutator families:

```text
Top-K routing:
    top_k_buy
    top_k_sell
    rotation_budget

Group-aware Top-K:
    group_aware on/off
    default_group_cap
    pressure_weight
    capacity_weight
    sell_overweight_weight

Confidence coefficients:
    recovery_residual_scale
    recovery_market_scale
    vix_shock_scale
    derisk_market_down_scale
    rerisk_min_scale
    derisk_min_scale

Trigger thresholds:
    recovery_trigger_threshold
    recovery_min_confidence_rerisk
    derisk_early_update_threshold
    risk_break_min_confidence_derisk
    recovery_persistence_days
    risk_break_persistence_days

Timing:
    k_root_days
    k_stock_days
```

Retrain-only family registry:

```text
retrain_only_candidates.csv

R3_compact22_feature_ablation_v1
R3_delta_change_only_features_v1
R3_residualized_delta_features_v1
```

These are marked retrain-only because changing compact/delta features changes
the PPO observation contract. A trained policy cannot be fairly evaluated under
a different feature set via counterfactual replay.

First next-level R6 run:

```text
reports/r_k_window_analysis/hcs_policy_forward_search_loop_nextlevel_r6_v1
```

Main evidence:

```text
Best adjusted candidates remain wider Top-K:
    top_k_buy = 10
    top_k_sell = 12 or 10

Trigger mutation:
    strict recovery improved over base 8/8 Top-K, but did not beat 10/10.

Confidence mutation:
    residual-rerisk boost improved raw return/sharpe but was tagged
    unstable_across_folds.

Timing mutation:
    root10/stock5 improved raw return but increased instability/drawdown,
    so it is not promoted yet.
```

## Next Implementation Step

Use the loop to search the next retrainable candidate family, not only Top-K:

```text
1. add confidence-coefficient mutations;
2. add trigger-threshold mutations;
3. add K_root / K_stock mutations;
4. add compact/delta feature-family tags for retrain planning;
5. train only candidates that survive both frozen replay and policy-forward replay.
```

The first retrain candidate suggested by the current policy-forward loop is:

```text
R6-like controller with wider Top-K:
    top_k_buy = 10
    top_k_sell = 12
    rotation_budget = 0
```

But before promotion, compare it against:

```text
top_k_buy = 10
top_k_sell = 10

top_k_buy = 10
top_k_sell = 8
```

because their scores are nearly tied and the exact best can be replay noise.

## Risk-Aware Top-K Extension

The next Top-K implementation is no longer only:

```text
stock potential * soft group multiplier
```

It adds explicit risk-aware execution gates:

```text
rotation_budget_eff = rotation_budget * stress_gate

buy_allowed =
    confidence_rerisk high
    and recovery_score high
    and residual_breadth_excess acceptable
    and risk_stress below cap

sell_priority_i *=
    1
    + risk_break_weight * risk_break_signal
    + residual_deterioration_weight * max(-residual_momentum_i, 0)
    + confidence_derisk_weight * confidence_derisk
```

This keeps the user's preferred interpretation:

```text
groups do not enforce hard diversification;
one strong group may still take the flow if its stocks dominate.
```

The implementation is logged through:

```text
incremental_topk_buy_allowed
incremental_topk_buy_gate_reason
incremental_topk_rotation_stress_gate
incremental_topk_rotation_budget_effective
incremental_topk_sell_multiplier_mean
incremental_topk_residual_deterioration_mean
```

Replay/search support was also extended:

```text
risk_aware_topk
buy_gate_min_confidence_rerisk
buy_gate_min_recovery_score
buy_gate_max_risk_stress
buy_gate_min_residual_breadth_excess_5d
rotation_stress_gate_*
sell_risk_break_weight
sell_residual_deterioration_weight
sell_confidence_derisk_weight
```

Primary trainable candidate:

```text
R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1
```

## 2026-05-30 HCS Guard Upgrade

Implemented in:

```text
scripts/hcs_policy_forward_search_loop.py
scripts/counterfactual_policy_forward_replay.py
scripts/heuristic_controller_search.py
```

Reason:

```text
R7 looked promising in policy-forward replay, but failed as a real run.
The search loop therefore needs stricter promotion guards, not only more mutators.
```

New ranking rule:

```text
raw selection_score
    -> hcs_adjusted_score
    -> hcs_final_score
```

`hcs_final_score` is now the promotion score. It penalizes:

```text
baseline_guard_failed
weak_fold_support
source_replay_overpredicts_real_baseline
lower_tail_fragile
drawdown_too_deep
cash_lock_in_risk
trigger spam
window collapse
```

New outputs:

```text
baseline_guard.csv
source_truth_replay_audit.csv
selection_regret.csv
family_summary.csv
negative_control_audit.csv
```

The baseline guard compares each candidate against `pf_original` by fold:

```text
delta_selection_score
conservative_delta_score
folds_sharpe_better
folds_drawdown_not_worse
controller_drift_count
promotion_decision
```

The lower-tail score is also logged:

```text
robust_selection_score
q25_sharpe
min_sharpe
folds_positive_sharpe
```

This is meant to prevent one-fold or average-only wins from being promoted.

Ground truth registry now includes the newer real-run aggregates:

```text
R6c_real_run_aggregate.csv
R6d_compare_run_aggregate.csv
all_existing_result_run_ranking_snapshot.csv
```

Default HCS policy-forward inputs now point to the active line:

```text
R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1
R7_root_K20_stock_K5_PD_mild_slice_rescorr_groupquality_top10_sell12_rotation_internaldays_v1
R3_root_K20_PD_confidence_slice_residual_stock_v1
```

Smoke-tested output:

```text
reports/r_k_window_analysis/hcs_policy_forward_search_loop_improved_smoke
```

Promotion interpretation:

```text
Do not promote a candidate from raw `selection_score`.
Use `hcs_final_score`, `hcs_promotable`, `selection_regret.csv`, and
`baseline_guard.csv`.
```

The mutating beam also uses `hcs_final_score`; candidates that only win before
baseline/fold-support guards are no longer selected as parents for the next
generation.
