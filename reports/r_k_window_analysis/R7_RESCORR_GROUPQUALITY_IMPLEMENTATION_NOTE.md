# R7 Rescorr Group Quality Transfer

## Goal

Transfer the useful idea from `E9_hierarchical_discovered_rescorr_dirtree_fixedpd_v1` into the current R-line without directly mixing the whole E9 hierarchy/distribution into the controller stack.

## What Changed

`R7_root_K20_stock_K5_PD_mild_slice_rescorr_groupquality_top10_sell12_rotation_internaldays_v1` inherits the current R6c controller:

- root K20 / stock K5 internal-day accounting;
- mild root confidence slice;
- stock confidence slice;
- incremental Top-K flow;
- risk-aware buy gate, sell-side deterioration, stress-gated rotation;
- train-only residual-correlation groups.

The only new mechanism is a soft group residual-quality modifier inside group-aware Top-K:

```text
stock potential
  * group pressure/capacity/overweight
  * group residual-quality multiplier
```

This is not a hard group budget. A single group can still receive all re-risk flow if its stocks dominate the global priority ranking.

## Group Residual Quality

For each residual-correlation group, R7 computes:

```text
group_rank_quality =
    mean(0.70 * residual_momentum_rank_centered_5d
       + 0.30 * residual_momentum_rank_centered_20d)

group_breadth_excess =
    share(group stocks with positive residual momentum) - 0.5
```

Then:

```text
buy_multiplier  *= clip(1 + 0.60 * group_rank_quality + 0.50 * group_breadth_excess)
sell_multiplier *= clip(1 - 0.85 * group_rank_quality - 0.50 * group_breadth_excess)
```

Interpretation:

- a group with improving residual breadth/rank becomes easier to buy;
- a group with deteriorating residual breadth/rank becomes easier to sell;
- the multiplier is bounded, so it cannot become a hidden portfolio manager.

## Policy-Forward Check

Before adding the runnable variant, I tested the mechanism with policy-forward replay:

```text
reports/r_k_window_analysis/policy_forward_rescorr_groupquality_20260530
```

Policy-forward replay loads the trained PPO model and recomputes policy anchors on the counterfactual portfolio trajectory. This is still no-retrain, but it is stricter than frozen-intent replay.

Best candidate:

```text
pf_rescorr_groupquality_balanced_b10_s12
```

Aggregate comparison against `pf_original` on the R6c trained models:

| candidate | selection_score | mean_sharpe | mean_return_pct | mean_max_drawdown | mean_cash | mean_turnover_l1 |
|---|---:|---:|---:|---:|---:|---:|
| pf_original | 0.732 | 1.094 | 10.05% | -7.34% | 41.24% | 0.00844 |
| pf_rescorr_groupquality_balanced_b10_s12 | 0.740 | 1.100 | 10.13% | -7.33% | 41.01% | 0.00845 |

The effect is small but positive enough to promote as a separate retrain candidate.

## New Logs

R7 logs:

```text
incremental_topk_group_residual_quality_enabled
incremental_topk_group_residual_buy_multiplier_mean
incremental_topk_group_residual_sell_multiplier_mean
incremental_topk_group_residual_rank_quality_mean
incremental_topk_group_residual_breadth_excess_mean
incremental_topk_group_residual_rank_quality_<group>
incremental_topk_group_residual_breadth_excess_<group>
incremental_topk_group_residual_buy_multiplier_<group>
incremental_topk_group_residual_sell_multiplier_<group>
```

## Caveat

The forward-policy improvement is not proof of PPO improvement. It only says the controller mutation is plausible on the trained R6c policy. The Huawei run is needed to test whether PPO still learns cleanly under this controller.
