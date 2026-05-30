# R6c Group/Risk-Aware Top-K Implementation Note

## Goal

`R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_softbuy_top8_sell12_rotation_internaldays_v1`
keeps the R5/R6 controller family but makes incremental Top-K routing less
naive:

- group signals are still soft priorities, not hard diversification budgets;
- buy flow is allowed only when recovery confirmation is present;
- sell priority is boosted during risk-break and residual deterioration;
- rotation budget is reduced in stress windows.

PPO still samples the original raw action factors. This layer is deterministic
execution logic and is logged separately.

## Execution Logic

Base flow:

```text
q anchor / stock anchor from policy
  -> K-window PD schedule
  -> mild root confidence slice
  -> incremental Top-K flow
  -> stock confidence slice
  -> executed weights
```

New risk-aware Top-K logic:

```text
rotation_budget_eff = rotation_budget * stress_gate

buy_allowed =
    confidence_rerisk >= threshold
    and recovery_score >= threshold
    and residual_breadth_excess_5d >= threshold
    and risk_stress <= cap

sell_priority_i *=
    1
    + risk_break_weight * risk_break_signal
    + residual_deterioration_weight * max(-residual_momentum_5d_i, 0)
    + confidence_derisk_weight * confidence_derisk
```

If buy is blocked, re-risk flow stays unfilled instead of being pushed into
stocks just to reduce cash. If sell Top-K lacks enough weight to meet a de-risk
flow, sell expansion is still enabled.

## Main Logs

```text
incremental_topk_risk_aware_enabled
incremental_topk_buy_allowed
incremental_topk_buy_gate_reason
incremental_topk_rotation_stress_gate
incremental_topk_rotation_budget_effective
incremental_topk_sell_multiplier_mean
incremental_topk_residual_deterioration_mean
incremental_topk_sell_multiplier_<ticker>
incremental_topk_residual_deterioration_<ticker>
```

These should be compared against:

```text
confidence_rerisk
confidence_derisk
recovery_score
risk_stress
risk_break_trigger
residual_breadth_excess_5d
trade_direction_<ticker>
```

## First-Pass Parameters

```yaml
top_k_buy: 8
top_k_sell: 12
rotation_budget_l1_per_day: 0.0025
group_aware.default_group_cap: 0.60
group_aware.capacity_weight: 0.25
group_aware.sell_overweight_weight: 1.25

risk_aware.buy_gate.min_confidence_rerisk: 0.50
risk_aware.buy_gate.min_recovery_score: 0.55
risk_aware.buy_gate.max_risk_stress: 0.90
risk_aware.buy_gate.min_residual_breadth_excess_5d: -0.02
risk_aware.rotation_stress_gate.stress_start: 0.55
risk_aware.rotation_stress_gate.stress_full: 0.90
risk_aware.sell_side.risk_break_weight: 1.00
risk_aware.sell_side.residual_deterioration_weight: 0.75
risk_aware.sell_side.confidence_derisk_weight: 0.25
```

## Sanity Checks Before Interpreting Results

- Buy gate should not be closed almost every day.
- Rotation should shrink in stress windows.
- Sell multiplier should rise on risk-break or weak residual momentum days.
- Top-K should show more sell-side activity than the prior R6 if de-risk days exist.
- If performance improves only through persistent cash, this is not a clean Top-K win.

## Local Policy-Forward Smoke

Policy-forward replay was run on the completed R6 folds as a no-retrain
sanity check:

```text
reports/r_k_window_analysis/hcs_policy_forward_search_loop_riskaware_topk_r6_g0
```

Best adjusted replay candidate:

```text
pf_riskga_b8_s12_rot0p0025 ... riskaware_softbuy

selection_score: 0.722
mean_sharpe:     1.070
mean_cash:       0.438
turnover_l1:     0.0081
buy_allowed:     0.396
rotation_gate:   0.904
sell_multiplier: 1.397
failure_tags:    ok
```

This is not a substitute for full PPO retraining. It only confirms that the
new logs and execution logic are active on all four R6 folds and that a softer
buy gate is less brittle than the stricter balanced gate.

The older non-`softbuy` R6c name remains in the config as a parent/alias. The
Huawei packages use the explicit `softbuy` variant name to make the run
auditable.
