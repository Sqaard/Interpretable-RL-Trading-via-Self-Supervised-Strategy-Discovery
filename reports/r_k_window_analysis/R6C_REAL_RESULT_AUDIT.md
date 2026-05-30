# R6c Real Result Audit

Date: 2026-05-30

Compared runs:

```text
R6_root_K20_stock_K5_PD_mild_slice_top5_rotation_internaldays_v1
R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_softbuy_top8_sell12_rotation_internaldays_v1
R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1
```

## Code and Package Sanity

The two R6c result folders are not independent variants. Their validation summaries and train summaries are identical for all folds. `validation_daily.csv` differs only in the string field `k_window_mode`; all numeric columns are identical.

The current config has:

```text
top_k_buy: 8
top_k_sell: 12
rotation_budget_l1_per_day: 0.0025
group_aware.enabled: true
risk_aware.enabled: true
buy_gate.min_confidence_rerisk: 0.50
buy_gate.min_recovery_score: 0.55
buy_gate.max_risk_stress: 0.90
sell_side.risk_break_weight: 1.00
sell_side.residual_deterioration_weight: 0.75
```

Important implementation caveat:

```text
softbuy is not a separate fractional-buy mechanism in src/ppo/stage0_1_weight_env.py.
The buy gate is binary: if checks fail, re-risk buy flow is blocked.
The name "softbuy" means softer threshold choice from policy-forward replay.
```

Rotation still runs when `direction != derisk`, including `rerisk_blocked`, but its budget is scaled by `rotation_stress_gate`.

## Aggregate Results

Selection score is `mean_sharpe - 0.5 * std_sharpe`.

| Run | Mean Sharpe | Std Sharpe | Selection | Return | Max DD | Turnover L1 | Cash |
|---|---:|---:|---:|---:|---:|---:|---:|
| R6c group/risk-aware Top8/Sell12 | 1.094 | 0.627 | 0.781 | 10.05% | -7.34% | 0.00844 | 41.24% |
| R6 baseline Top5 rotation | 0.933 | 0.626 | 0.620 | 9.39% | -9.92% | 0.00902 | 33.84% |

R6c improves selection score by about `+0.160`, mainly through much better 2020 drawdown and Sharpe, while increasing average cash by about `+7.4 pp`.

## Fold-Level Comparison

| Fold | R6 Sharpe | R6c Sharpe | R6 Return | R6c Return | R6 Max DD | R6c Max DD | R6 Cash | R6c Cash |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018 | -0.035 | 0.076 | -1.25% | 0.20% | -13.08% | -13.55% | 27.17% | 33.67% |
| 2019 | 1.639 | 1.723 | 13.10% | 13.21% | -4.45% | -4.18% | 36.52% | 39.30% |
| 2020 | 0.843 | 1.468 | 15.77% | 19.49% | -17.35% | -7.26% | 35.36% | 45.66% |
| 2021 | 1.287 | 1.109 | 9.94% | 7.29% | -4.79% | -4.37% | 36.31% | 46.31% |

Interpretation:

```text
R6c helped most in 2020 risk/turbulence.
R6c hurt 2021 upside participation because the buy gate/recovery condition kept cash too high.
```

## Top-K Layer Behavior

Average R6c diagnostics:

```text
buy_allowed:             36.84% days
buy_requested:           0.02290
buy_filled:              0.01044
buy_unfilled:            0.01246
sell_requested:          0.00024
sell_filled:             0.00024
sell_expansion_count:    0.0
rotation_budget_eff:     0.00226
rotation_requested:      0.00197
rotation_stress_gate:    0.904
sell_multiplier_mean:    1.400
```

This confirms the new layer mostly controls re-risk and rotation. Direct de-risk sell flow is still very small, so sell-side improvements mostly affect rotation candidate ranking rather than large cash-raising days.

Fold-level buy-gate behavior:

| Fold | Buy Allowed | Buy Requested | Buy Filled | Buy Unfilled | Recovery Trigger | Risk-Break Trigger | Cash |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2018 | 28.40% | 0.02398 | 0.01154 | 0.01244 | 1.20% | 1.20% | 33.67% |
| 2019 | 47.01% | 0.01904 | 0.01005 | 0.00899 | 3.98% | 0.40% | 39.30% |
| 2020 | 45.63% | 0.02448 | 0.01103 | 0.01345 | 3.97% | 5.16% | 45.66% |
| 2021 | 26.29% | 0.02410 | 0.00912 | 0.01498 | 0.80% | 1.20% | 46.31% |

2021 underperformance is consistent with `recovery_score` being too strict: the buy gate failed recovery most often, while residual breadth conditions almost always passed.

## Policy-Forward Replay Check

The policy-forward replay prediction was directionally useful:

```text
PF softbuy candidate:
    selection_score ~0.722
    mean_sharpe     ~1.070
    max_drawdown    ~-7.07%
    cash            ~43.8%
    turnover        ~0.0081

Real R6c:
    selection_score  0.781
    mean_sharpe      1.094
    max_drawdown     -7.34%
    cash             41.2%
    turnover         0.0084
```

So policy-forward replay ranked the idea correctly and estimated the main tradeoff: better drawdown/Sharpe with higher cash.

## Verdict

R6c is better than R6 baseline on the current validation aggregate, but not a clean proof that "softbuy" beats the non-softbuy version because both folders contain the same numeric behavior. The real claim supported by this run is:

```text
Group/risk-aware Top-K with Top8/Sell12, reduced rotation budget, risk-aware buy gate,
sell-side risk multiplier, and stress-scaled rotation improves R6 baseline.
```

The cost is higher cash and weaker 2021 participation.

## Next Changes

1. Do not run the non-softbuy R6c alias again unless it is given genuinely different thresholds.
2. If a real softbuy experiment is desired, implement fractional buy scaling instead of binary allow/block:

```text
buy_fill_scale = clip(sigmoid(score - threshold), min_scale, 1.0)
buy_amount = buy_requested * buy_fill_scale
```

3. For 2021-style calm/up windows, loosen recovery by replacing hard `recovery_score >= 0.55` with partial scaling, while keeping stress/risk-break strong.
4. Keep sell-side risk-aware multiplier, but add a stronger de-risk path only when root actually asks for cash; current sell-requested flow is near zero.
5. Treat policy-forward replay as a useful gate, but continue to confirm final candidates through real PPO policy-forward training.

Generated tables:

```text
reports/r_k_window_analysis/R6c_real_summary_by_fold.csv
reports/r_k_window_analysis/R6c_real_daily_aggregates_by_fold.csv
reports/r_k_window_analysis/R6c_real_metadata_by_fold.csv
reports/r_k_window_analysis/R6c_real_ticker_aggregates.csv
reports/r_k_window_analysis/R6c_real_run_aggregate.csv
```
