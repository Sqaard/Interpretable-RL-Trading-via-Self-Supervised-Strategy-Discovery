# Stage 0.1 Next Experiment Plan, DeepSearch-Aligned v2

Date: 2026-05-19

This plan supersedes the first post-audit plan. It incorporates the independent DeepSearch review:

- `C:/Users/ivanp/Downloads/deep-research-report (16).md`
- `reports/STAGE0_1_ALL_EXPERIMENT_RESULTS_AUDIT.md`

DeepSearch agrees with the main audit finding: the Stage 0.1 runs do **not** prove that hierarchy is a bad idea. They mostly prove that our concrete hierarchy stack broke earlier than the hierarchy hypothesis could be judged: root cash semantics, unsafe static topology, controller masking, and safety rewrites all confounded the results.

The main DeepSearch correction is ordering. The first plan ranked strict frozen-teacher targetcov highest. DeepSearch argues that this is slightly premature: the most upstream blocker is the `q_min/q_exec` root boundary artifact. If targetcov is run before root cash semantics are repaired, a better topology may still be interpreted through a broken exposure throttle.

## Evidence Basis

The revised order is based on the audit and on the literature map summarized in the DeepSearch report:

- Cash/risk-free split is theoretically defensible through Tobin/Merton-style separation logic and modern hierarchical portfolio RL designs.
- A fixed structural attractor such as `q_exec >= 0.50` is not supported by that theory; cash/risky exposure should be state- or regime-conditioned.
- Dirichlet is a strong simplex baseline, while logistic-normal, generalized Dirichlet, stick-breaking, and Dirichlet-tree variants are higher-fragility options that should be tested only after root/topology issues are fixed.
- No-trade bands, partial adjustment, and asymmetric execution are literature-supported, but only as cost-aware execution around a sane raw target.
- HRP/HERC and clustering literature support hierarchy only when the tree is meaningful and stable; singleton groups and unstable discovered clusters are not acceptable.
- Graph/relationship portfolio RL supports learned/discovered asset relations, but teacher-derived `targetcov` has only indirect support and must be protected against teacher bias and leakage.
- Safe RL/action projection supports formal safety layers, but projection aliasing and frequent rewrites can hide bad raw policies.
- MoE/options literature supports a style meta-policy fallback, but only if experts are frozen, meaningful, and switching is regularized.

## High-Confidence Findings From Stage 0.1

| Finding | Confidence | Consequence |
|---|---:|---|
| Static sector tree in a small Dow-like universe was unsafe because of singleton `Energy = [CVX]`. | High | Any tree experiment must enforce no-singleton or explicit cap-aware deterministic treatment. |
| `q_min/q_exec` clipping around 50% invested materially confounded Batches 2-6 and most later hierarchy tests. | High | Root q repair must be the first causal test. |
| Batch 2, Batch 9 targetcov, Batch 11, and Batch 12 did not strictly test their advertised hypotheses. | High | They must be rerun or downgraded to weak evidence. |
| Execution and safety layers often improved symptoms while hiding bad raw targets. | High | Controllers and projections must be delayed until raw target diagnostics are sane. |
| Flat Dirichlet remains the benchmark, not a lucky outlier. | High | Corrected hierarchy must compare against flat 256 and compact22. |
| Root split may still be viable after redesign. | Medium | Do not discard cash/risky separation; rerun it cleanly. |
| Strict targetcov is promising but not guaranteed. | Medium-low | Treat it as high-upside, high-control experiment, not expected winner. |
| Logistic-normal may help later, but Batch 8 did not cleanly test it. | Medium-low | Retry only on safe topology with component-level KL/std diagnostics. |

## Main Corrections Versus Previous Plan

| Previous Plan | DeepSearch Correction | Reason |
|---|---|---|
| Targetcov was ranked first. | Root q boundary ablation is now first. | Targetcov on a broken root layer will not answer the upstream exposure question. |
| No explicit early static-tree no-singleton control. | Add `R3_sector_merged_capped_static_tree_v1`. | We need to know whether old hierarchy failed mainly because of singleton CVX. |
| Controllers were only retrain variants. | Add offline controller replay before retraining. | It is a cheap, clean way to measure controller effect without PPO confounding. |
| Targetcov was single-teacher oriented. | Add teacher hash/export, optional teacher ensemble, and raw risky-head target clustering. | Reduces leakage, cash-dominance, and teacher-bias risks. |
| Safety layer was planned as true QP rerun. | Keep true QP, but only as diagnostic after raw targets are sane. | Frequent projection can become the real portfolio manager. |
| Feature/policy size ablation remained in roadmap. | Keep it late. | Batch 1 suggests representation is secondary until action semantics are fixed. |

## Updated Ranked Experiment Roadmap

| Rank | Experiment | Hypothesis | What Changes | Addresses | Effect | Risk | Complexity | Dependency | Test Type |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | `R1_root_q_boundary_ablation_v1` | Cash collapse was mainly a `q_min/q_exec` artifact. | Root-split baseline; remove `q_exec >= 0.50`; keep fixed PD; no tree. | B2-B5 root collapse. | High | Low-med | Low | None | Clean causal |
| 2 | `R2_root_q_regime_prior_v1` | Regime-conditioned cash prior/bounds beats weak excess-cash penalty. | Same as R1, but replace weak soft cash penalty with KL/Beta prior or regime cash bounds. | B2 invalid soft-prior test. | High | Medium | Medium | R1 | Clean causal |
| 3 | `R3_sector_merged_capped_static_tree_v1` | Old static hierarchy failed mainly because of singleton topology. | Static sector tree with merged small sectors, no singleton groups, and stock/group caps inside design. | B7/B8/B10 CVX artifact. | Medium | Low | Low-med | R1/R2 preferred; can run parallel | Must-rerun control |
| 4 | `R4_teacher_raw_target_export_v1` | Strict targetcov needs real train-only teacher intent. | Export raw train-period `q_target`, `u_target`, risky logits/alphas if available, target deltas, teacher hashes. | B9 proxy-targetcov invalidity. | Enabling | Low | Low | Can run parallel with R1/R2 | Technical prerequisite |
| 5 | `R5_strict_targetcov_teacher_tree_fixedPD_v1` | Frozen teacher raw target co-movement yields better topology than static sectors. | Groups from train-only frozen teacher target deltas; fixed PD; no singleton; no learned Kp. | B9 incomplete targetcov. | High | Med-high | High | R4 and preferably R1/R2 | Must-rerun, mostly causal |
| 6 | `R6_rescorr_tree_qcal_rerun_v1` | Residual-corr hierarchy remains a strong non-teacher baseline after q repair. | Rerun residual-corr discovered tree with corrected root q semantics. | B9 incomplete rescorr; targetcov comparison control. | Medium | Medium | Medium | R1/R2 | Clean causal baseline |
| 7 | `R7_offline_controller_replay_ablation_v1` | Controller effects can be measured without retraining. | Replay frozen raw targets through symmetric/asym/deadzone/QP controllers. | B4/B5/B11 controller masking ambiguity. | Medium | Low | Low | One sane raw target from R1/R2/R5/R6 | Clean causal replay |
| 8 | `R8_targetcov_or_rescorr_tree_asym_deadzone_v1` | Best fixed execution transfers once target/topology are sane. | Apply selected asym/deadzone controller to corrected winning tree. | B5 promising but confounded. | Med-high | Medium | Medium | R5/R6 and R7 | Combined test |
| 9 | `R9_targetcov_or_rescorr_tree_learnedkp_v1` | Learned gates help only after raw policy is sane. | Learned bounded Kp on corrected winner; explicit gate saturation penalties/diagnostics. | B6 gate-collapse ambiguity. | Medium | Med-high | Medium | R8 | Combined test |
| 10 | `R10_true_qp_projection_diagnostic_v1` | True nearest-safe QP clarifies whether hard safety helps. | Replace heuristic Batch 11 projection with actual env-side QP/OSQP nearest-safe projection. | B11 invalidity. | Medium | Medium | High | R8 or R9 | Must-rerun, dangerous diagnostic |
| 11 | `R11_frozen_teacher_style_bank_meta_v1` | Style gating is viable fallback if corrected hierarchy still lags. | Frozen learned expert/style bank + gate + switching/dwell penalty. | B12 invalid rule-heavy test. | Medium | Medium | Med-high | Good teacher bank available | Must-rerun fallback |
| 12 | `R12_feature_policy_size_ablation_v2` | Representation matters only after action design is fixed. | Retest compact vs full features and larger nets on corrected winner. | Residual Batch 1 uncertainty. | Low-med | Low | Low | Corrected winner exists | Clean causal |
| 13 | `R13_logitnormal_groupdiag_safe_topology_v1` | Logistic-normal may help once topology/root are safe. | Diagonal logistic-normal group law on corrected no-singleton winner, low std ceiling. | B8 confounded failure. | Low-med | High | High | R5/R6, preferably R8 | Optional |
| 14 | `R14_bottomup_veto_safe_tree_v1` | Veto can be rare safety signal after raw policy is good. | Monotone reduction-only veto on corrected tree. | B10 overreach. | Low | High | Medium | R8 or later | Optional, dangerous |

## Minimal Near-Term Run Set

Do not run a large mixed wave first. The minimal near-term set is:

1. `R1_root_q_boundary_ablation_v1`
2. `R2_root_q_regime_prior_v1`
3. `R3_sector_merged_capped_static_tree_v1`
4. `R4_teacher_raw_target_export_v1`
5. `R5_strict_targetcov_teacher_tree_fixedPD_v1`
6. `R7_offline_controller_replay_ablation_v1`

This set answers the three highest-value questions:

- Is the root cash layer broken because of the 50% boundary artifact?
- Was the static hierarchy mainly broken by singleton topology?
- Do controllers actually help, or do they only mask bad raw targets?

## Detailed Experiment Requirements

### R1: Root Q Boundary Ablation

Purpose: isolate the `q_min/q_exec` artifact.

Required changes:

- Remove hard `q_exec >= 0.50` as default behavior.
- Keep fixed PD or the simplest controller; no tree, no targetcov, no learned Kp.
- Test a small grid:
  - lower fixed `q_min`
  - no fixed floor but bounded cash max
  - risk-conditioned cash max

Required diagnostics:

- `q_target`, `q_exec`, `cash_target`, `cash_exec`
- q boundary hit rate
- q standard deviation
- cash by risk quantile
- calm vs stress cash
- `corr(cash, risk_score)`, `corr(q, risk_score)`

Pass:

- Boundary hit rate materially drops.
- Cash is no longer constant high.
- Calm cash is low and stress cash can rise.

Kill:

- q remains stuck at a boundary.
- cash collapses to all-risk or all-cash.
- crisis protection disappears entirely.

### R2: Regime Cash Prior

Purpose: replace weak excess-cash penalty with a cleaner state-conditioned prior.

Candidate mechanisms:

- KL/Beta prior toward risk-conditioned invested fraction.
- Risk-conditioned cash bounds.
- q-boundary penalty.
- Component-wise entropy/KL schedule for root q.

Required diagnostics:

- Prior penalty scale versus reward scale.
- `q_policy_mean` versus `q_prior`.
- crisis windows separately from average folds.
- root entropy and root KL separately from risky allocation KL.

Pass:

- Calm cash low, stress cash allowed.
- q-risk relation is positive for cash and negative for q.
- No structural boundary pile-up.

Kill:

- Prior dominates reward.
- Cash is suppressed in drawdown/high-vol windows.
- q is again nearly deterministic.

### R3: Static Tree No-Singleton Control

Purpose: determine whether old static hierarchy failed mainly because of the singleton `CVX` topology.

Required changes:

- Merge singleton and tiny sectors into broader groups.
- Enforce minimum group size by construction.
- Add group-size-aware priors/caps:
  - prior mass proportional to capacity, liquidity, and group size;
  - per-stock max weight;
  - per-group max weight.

Required diagnostics:

- group sizes
- group HHI
- stock HHI
- max stock target/executed weight
- max group target/executed weight
- cap hit rate

Pass:

- CVX-like artifact disappears.
- Performance improves materially over old Batch 7.

Kill:

- Still far below flat baseline.
- Caps are constantly saturated.
- One merged group becomes a new giant shortcut.

### R4/R5: Strict Frozen-Teacher Targetcov

Purpose: rerun Batch 9 correctly.

R4 export requirements:

- Use frozen teacher checkpoints.
- Export train-period raw targets only.
- Save:
  - teacher model id
  - checkpoint hash
  - fold id
  - train date range
  - raw `q_target`
  - raw `u_target`
  - raw final target weights
  - risky logits/alphas if available
  - executed weights separately
  - target deltas

R5 clustering rules:

- Fit clusters only on train-period exports.
- Prefer clustering `delta_u_target` or risky-head logits/alphas, not only `delta_w_total`, because total weights can be dominated by cash regime.
- Compare at least:
  - flat256 teacher targetcov
  - compact22 teacher targetcov
  - optional ensemble consensus targetcov
  - residual-corr groups
- Enforce no singleton groups by default.
- Enforce max group size.
- Add bootstrap/windowed stability gate.

Required diagnostics:

- within-cluster vs between-cluster target covariance/correlation
- within-cluster vs between-cluster residual correlation
- Adjusted Rand Index or pairwise co-clustering stability
- group size distribution
- teacher hash and date audit
- leakage audit

Pass:

- 4/4 folds complete.
- stable groups, no singleton, no giant cluster.
- competitive with merged static tree and close to flat benchmark.

Kill:

- leakage detected.
- cluster stability fails.
- giant group or singleton group appears.
- no gain over merged static control.

### R7: Offline Controller Replay

Purpose: separate execution effect from PPO retraining effect.

Inputs:

- Frozen raw targets from flat/root/tree policies.

Replay controllers:

- no controller
- symmetric PD/Kp
- asym de-risk/re-risk
- smooth deadzone
- asym + deadzone
- heuristic projection
- true QP projection if implemented

Required diagnostics:

- raw-to-controller gap
- controller-to-executed gap
- raw-to-executed gap
- turnover decomposition
- transaction cost proxy
- drawdown and recovery window behavior
- sell-low/re-enter-late signatures

Pass:

- A controller reduces turnover/cost without large persistent raw-to-controller gap.

Kill:

- Controller only helps by huge rewrites.
- Recovery windows suffer from slow re-risk.

## Additional Design Ideas Added From DeepSearch

These are not all immediate experiments, but they must be added to the design backlog:

| Idea | Why It Matters | When To Use |
|---|---|---|
| Group-size-aware priors | Prevents small groups from becoming hidden single-name concentration channels. | All tree variants. |
| Bootstrap stability gate for discovered topology | Avoids training on noisy, unstable clusters. | All discovered hierarchy variants. |
| Teacher ensemble targetcov | Reduces one-teacher bias. | After single-teacher targetcov exporter is working. |
| Cluster risky-head targets/logits, not only total weights | Avoids cash regime dominating asset topology. | Targetcov clustering. |
| Offline controller replay | Cheap causal filter before retraining controllers. | Before R8/R9/R10. |
| Raw-policy safety regularization | Prevents projection/veto from becoming the real portfolio manager. | Any safety layer. |
| Decoupled actor/critic routing | Actor-root can stay clean while critic sees richer state. | After root q redesign. |
| Component-wise entropy/KL schedules | Root, group, leaf, and gate components may need different exploration pressure. | Root q and tree variants. |
| Cap-aware action parameterization | Better than post-hoc projection when constraints are structural. | Tree and safety-aware variants. |

## Controller and Safety Rules

- Keep PPO log-prob on raw sampled action factors.
- Never pretend adjusted/projected/executed weights were sampled directly.
- Always log:
  - raw sampled action
  - transformed target
  - controller-adjusted target
  - projected/safe target if any
  - executed weights
  - all gaps between these levels
- Safety/projection/veto is not a pass if it rewrites most actions.
- Add raw violation penalty if projection is used during training.

## Global Pass/Fail Criteria

Primary benchmarks:

- `flat_dirichlet_pd_stage0style256`: selection score about `0.6897`, validation Sharpe about `1.1488`, return about `15.75%`, cash about `4.48%`.
- `flat_dirichlet_pd_compact22`: selection score about `0.6804`, validation Sharpe about `1.1265`, return about `15.54%`, cash about `4.30%`, lower turnover/clip.

A corrected hierarchy is promising only if:

- It completes all four validation folds.
- It avoids constant high cash.
- It avoids singleton-driven concentration.
- It keeps max stock and max group exposure under explicit thresholds.
- It logs raw, adjusted, safe, and executed actions separately.
- It is close to the flat benchmark or clearly improves stability/turnover/drawdown.

Kill or redesign if:

- q remains near a structural boundary.
- cash is high and regime-insensitive.
- one stock or group dominates through topology.
- controller/projection/veto rewrites most actions.
- projection/veto active rate stays high.
- targetcov/discovered clusters fail stability or leakage checks.
- PPO component-level KL shows collapse even if total KL looks acceptable.

## Final Execution Recommendation

Run first:

1. `R1_root_q_boundary_ablation_v1`
2. `R4_teacher_raw_target_export_v1` in parallel, because it is a technical prerequisite and does not confound causal interpretation.
3. `R2_root_q_regime_prior_v1`
4. `R3_sector_merged_capped_static_tree_v1`
5. `R5_strict_targetcov_teacher_tree_fixedPD_v1`, only after R1/R2 show root cash is no longer stuck on a structural boundary.
6. `R7_offline_controller_replay_ablation_v1`

Do not run controllers, learned Kp, QP projection, logistic-normal, or bottom-up veto before the root and topology repairs are verified.

Bottom line:

**First fix root cash semantics, then fix topology, then test discovered hierarchy, then test controllers and safety.**
