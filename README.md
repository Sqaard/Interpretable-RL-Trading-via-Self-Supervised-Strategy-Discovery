# Interpretable RL Trading via Self-Supervised Strategy Discovery

This repository contains the core Stage 0 / Stage 1 / Stage 2 pipeline used for the interpretable Dow-30 portfolio RL experiments.

The code is shared for methods review and co-author due diligence. It intentionally does not include large trained-model artifacts, market data, or cloud job outputs. Those are exchanged separately as zipped experiment packages.

## What Is Included

- `src/ppo/stage0_1_weight_env.py`  
  Weight-based portfolio environment: observation construction, reward, portfolio execution, risk/cash layer, group-aware Top-K execution, confidence signals, transaction costs, and logged behavioral diagnostics.

- `src/ppo/dirichlet_policy.py`  
  Custom Stable-Baselines3 actor-critic policies and action distributions, including Dirichlet, root-split Beta/Dirichlet, learned-Kp, tree, and logit-normal variants.

- `src/ppo/stage0_1_train.py`  
  Walk-forward training/evaluation entrypoint, fold handling, discovered hierarchy utilities, validation/frozen evaluation, and daily diagnostics export.

- `configs/stage0_1_active_r_pipeline.yaml`  
  Compact active configuration for the current R-line models: R3 baseline, R6c group-risk-aware Top-K model, R7 rescorr group-quality model, and one flat baseline.

- `scripts/run_stage1_r6c_vq.py`  
  Stage 1 KMeans primitive/codebook extraction from policy hidden states. This is the primary Stage 1 path used for the current model.

- `scripts/run_stage1_r6c_vqvae.py`  
  Optional neural VQ-VAE counterpart for comparison.

- `scripts/run_stage2_r6c_behavior_diagnostics.py` and `scripts/run_stage2_r6c_kmeans_deep_levels.py`  
  Stage 2 behavioral diagnostics joining primitive codes with portfolio logs: risk/cash behavior, group routing, Top-K flow, confidence, and stock-level trade behavior.

- `scripts/build_r6c_frozen_test_rollout_for_joseph.py`  
  Frozen 2022-2023 rollout package builder in the same format as the Stage 1/Stage 2 handoff files.

- `scripts/hcs_*.py` and replay scripts  
  Heuristic Controller Search and policy-forward replay infrastructure used to screen controller variants before expensive cloud PPO retraining.

## Core Stage 0 Model

The current primary candidate is:

```text
R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1
```

Conceptually, it has three interpretable levels:

1. `risk/cash layer`  
   A root Beta policy chooses the target risky exposure versus cash.

2. `stock-weight policy layer`  
   A Dirichlet policy proposes stock allocation over the Dow-30 risky book.

3. `execution / routing layer`  
   K-window PD execution, confidence/risk signals, group-aware Top-K buy/sell routing, and rotation limits transform raw policy intent into executed portfolio weights.

PPO log probability remains attached to the sampled raw action factors. Execution and routing layers are deterministic wrappers and are logged separately.

## Observation Space

The environment reads a date/ticker panel from `model_ready_csv`. Each timestep contains a flattened stock-feature panel plus portfolio state features. The active config points to:

```text
artifacts/stage0_1/features/stage0_1_weight_features_model_ready.csv
```

The feature CSV is not committed because it is a data artifact. The feature-building and train-only normalization utilities are in `src/data/`.

## Reward

The reward is implemented in:

```text
src/ppo/stage0_1_weight_env.py::_compute_reward
```

At a high level it uses next-day portfolio return after transaction costs, with configurable penalties for turnover, projection/gap terms, and controller/safety regularization depending on the experiment variant.

## Hidden-State Extraction

Stage 1 extracts policy hidden states from the trained actor network during replay. The main paths are:

```text
scripts/run_stage1_r6c_vq.py
scripts/extract_stage0_1_hidden_state_package.py
```

The current primary codebook uses KMeans with `K=8`, because it had higher effective code usage, better reconstruction ratio, longer temporal coherence, and stronger cross-fold stability than the neural VQ-VAE comparison for this model.

## Typical Commands

Train/evaluate a variant:

```bash
python -m src.ppo.stage0_1_train \
  --config configs/stage0_1_active_r_pipeline.yaml \
  --variants R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1
```

Build Stage 1 KMeans package:

```bash
python scripts/run_stage1_r6c_vq.py
```

Run Stage 2 diagnostics:

```bash
python scripts/run_stage2_r6c_behavior_diagnostics.py
python scripts/run_stage2_r6c_kmeans_deep_levels.py
```

Build frozen 2022-2023 rollout package:

```bash
python scripts/build_r6c_frozen_test_rollout_for_joseph.py
```

Run policy-forward HCS smoke/search:

```bash
python scripts/hcs_policy_forward_search_loop.py --generations 1 --max-zips 1
```

## Notes

- This repository is code-first. Large experiment folders under `artifacts/` are intentionally excluded.
- `configs/stage0_1_active_r_pipeline.yaml` is the compact config for current work. The historical full experiment config is not required for Methods review.
- The reports folder includes implementation notes and audit summaries used to explain the evolution from Stage 0.1 baseline experiments to the current R6c/R7 candidates.

