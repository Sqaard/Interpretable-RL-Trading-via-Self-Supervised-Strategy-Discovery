# HCS Counterfactual Policy-Forward Replay Foundation

Date: 2026-05-29

## Purpose

This adds a stronger cheap-evaluation layer for controller hypotheses:

1. load a trained PPO model from an experiment result zip;
2. keep the policy weights fixed;
3. rerun validation with alternative controller/execution settings;
4. recompute policy anchors from the counterfactual portfolio state.

This is stronger than frozen-intent replay because the trained policy sees the
new trajectory state. It is still weaker than full PPO retraining because the
policy did not learn under the new controller reward/trajectory.

## Implementation

Script:

```text
scripts/counterfactual_policy_forward_replay.py
```

Outputs:

```text
reports/r_k_window_analysis/counterfactual_policy_forward_replay_r3_core
reports/r_k_window_analysis/counterfactual_policy_forward_replay_r6_core
```

Each output directory contains:

```text
counterfactual_candidate_registry.csv
counterfactual_fold_summary.csv
counterfactual_aggregate_summary.csv
daily/<source>/<fold>/<candidate>/validation_daily.csv
COUNTERFACTUAL_POLICY_FORWARD_REPLAY.md
```

## Candidates Tested

```text
pf_original
pf_no_incremental_topk
pf_topk_b5_s8_rot0
pf_topk_b8_s8_rot0
pf_groupaware_cap45_b8_s8_rot0
```

For R3, `pf_no_incremental_topk` is not meaningful because the original R3 does
not use incremental Top-K.

## R3 Core Results

Source:

```text
R3_root_K20_PD_confidence_slice_residual_stock_v1
```

Aggregate:

| Candidate | Folds | Selection Score | Mean Sharpe | Std Sharpe | Mean Return | Mean MDD | Mean Cash | Mean Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pf_original | 4 | 0.7103 | 1.0547 | 0.6888 | 0.0819 | -0.0626 | 0.4970 | 0.0060 |
| pf_topk_b8_s8_rot0 | 4 | 0.6626 | 0.9905 | 0.6558 | 0.0769 | -0.0618 | 0.4981 | 0.0060 |
| pf_groupaware_cap45_b8_s8_rot0 | 4 | 0.6594 | 0.9857 | 0.6526 | 0.0768 | -0.0626 | 0.4986 | 0.0060 |
| pf_topk_b5_s8_rot0 | 4 | 0.6213 | 0.9357 | 0.6287 | 0.0749 | -0.0649 | 0.4978 | 0.0060 |

Interpretation:

Top-K improves fold_2018 but hurts folds 2019 and 2021, so it is not a robust
drop-in improvement on top of R3 v1. The original controller remains best under
this no-retrain forward replay.

## R6 Core Results

Source:

```text
R6_root_K20_stock_K5_PD_mild_slice_top5_rotation_internaldays_v1
```

Aggregate:

| Candidate | Folds | Selection Score | Mean Sharpe | Std Sharpe | Mean Return | Mean MDD | Mean Cash | Mean Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pf_groupaware_cap45_b8_s8_rot0 | 4 | 0.6280 | 0.9736 | 0.6912 | 0.0972 | -0.0987 | 0.3440 | 0.0074 |
| pf_topk_b8_s8_rot0 | 4 | 0.6197 | 0.9587 | 0.6780 | 0.0958 | -0.0989 | 0.3438 | 0.0074 |
| pf_no_incremental_topk | 4 | 0.6043 | 0.9566 | 0.7046 | 0.0953 | -0.0998 | 0.3416 | 0.0091 |
| pf_topk_b5_s8_rot0 | 4 | 0.5742 | 0.9530 | 0.7576 | 0.0940 | -0.0970 | 0.3434 | 0.0075 |
| pf_original | 4 | 0.5720 | 0.9335 | 0.7230 | 0.0939 | -0.0992 | 0.3384 | 0.0090 |

Interpretation:

For a policy trained with the R6 mechanics, group-aware Top-K and a wider
buy/sell set improve no-retrain validation. The original R6 Top-K settings look
too restrictive/noisy relative to `pf_groupaware_cap45_b8_s8_rot0`.

## HCS Implications

Current cheap-search conclusion:

1. Do not add Top-K to R3 v1 as a deterministic post-hoc layer without retrain.
2. For R6-like policies, search should prioritize:
   - wider Top-K, especially `buy=8/sell=8`;
   - group-aware soft caps;
   - lower turnover than original R6;
   - no forced all-risk-money-in-Top-K reconstruction.
3. Group-aware Top-K should be treated as a controller/execution candidate that
   still needs full PPO retrain before being promoted.

Recommended next HCS step:

```text
Use counterfactual_policy_forward_replay.py to generate candidate labels,
then fit a simple calibration/ranking model against real experiment outcomes:

features:
    controller family
    K_root / K_stock
    Top-K buy/sell
    group-aware on/off
    turnover
    cash
    target-to-exec gaps
    trigger rates
    component confidences

label:
    full-run selection_score or fold-level Sharpe/drawdown delta
```

This makes HCS more useful than a rules-only grid: it can learn which cheap
replay signals historically predicted full PPO results.

## Caveats

1. Counterfactual replay is not full retraining.
2. It can miss policy adaptation effects.
3. It is still much better than static log replay for controller changes because
   policy actions are recomputed under the altered trajectory.
4. Promotion rule should remain:

```text
offline replay -> counterfactual policy-forward replay -> full PPO rerun
```
