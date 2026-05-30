"""Build the selected R6c frozen 2022-2023 rollout package for Joseph.

Joseph's Stage 4 request is the out-of-sample counterpart of the Stage 1/2
train+validation NPZ. This script fixes the selected fold_2021 R6c model,
rebuilds fold-local train-only feature scaling through the frozen window, and
exports one daily row per internal trading day for 2022-01-03..2023-02-28.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_stage0_1_hidden_state_package import policy_layers  # noqa: E402
from run_stage1_r6c_vq import (  # noqa: E402
    apply_existing_codebook,
    centered_windows,
    select_behavior_columns,
    summarize_codes,
    zscore_fit,
)
from src.data.stage0_1_normalization import prepare_fold_scaled_features  # noqa: E402
from src.ppo.instrumented_ppo import InstrumentedPPO  # noqa: E402
from src.ppo.stage0_1_train import append_eval_info_row  # noqa: E402
from src.ppo.stage0_1_weight_env import load_weight_panel, make_env_from_config  # noqa: E402


DEFAULT_R6C_DIR = (
    ROOT
    / "artifacts"
    / "stage0_1"
    / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1"
)
DEFAULT_STAGE1_DIR = (
    ROOT
    / "artifacts"
    / "stage1"
    / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_stage1"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "artifacts"
    / "stage4"
    / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_frozen_2022_2023_for_Joseph"
)


@dataclass(frozen=True)
class FrozenArtifact:
    config: dict[str, Any]
    metadata: dict[str, Any]
    model_path: Path
    variant: dict[str, Any]
    fold: dict[str, Any]
    source_zip: Path


def read_zip_member(zf: zipfile.ZipFile, suffix: str) -> bytes:
    matches = [name for name in zf.namelist() if name.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"No zip member ending with {suffix}")
    matches.sort(key=len)
    return zf.read(matches[0])


def select_fold_zip(r6c_dir: Path, fold: str) -> Path:
    matches = sorted(r6c_dir.glob(f"*{fold}*_results.zip"))
    if not matches:
        raise FileNotFoundError(f"No R6c result zip for {fold} in {r6c_dir}")
    return matches[-1]


def extract_artifact(zip_path: Path, tmp_dir: Path) -> FrozenArtifact:
    with zipfile.ZipFile(zip_path) as zf:
        config = yaml.safe_load(read_zip_member(zf, "config.yaml").decode("utf-8"))
        metadata = json.loads(read_zip_member(zf, "/metadata.json").decode("utf-8"))
        model_bytes = read_zip_member(zf, "/model.zip")

    model_path = tmp_dir / "model.zip"
    model_path.write_bytes(model_bytes)
    return FrozenArtifact(
        config=config,
        metadata=metadata,
        model_path=model_path,
        variant=metadata["variant"],
        fold=metadata["fold"],
        source_zip=zip_path,
    )


def fit_stage1_kmeans(stage1_dir: Path, *, k: int, window_length: int, seed: int) -> dict[str, Any]:
    package = np.load(stage1_dir / "r6c_stage1_hidden_state_package.npz", allow_pickle=True)
    hidden = np.asarray(package["train_hidden"], dtype=np.float32)
    existing_codes = np.asarray(package["train_code_id"], dtype=np.int32)
    existing_valid = np.asarray(package["train_code_valid"], dtype=bool)

    mean, std = zscore_fit(hidden)
    z = ((hidden - mean) / std).astype(np.float32)
    windows, valid = centered_windows(z, window_length)
    model = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=500)
    model.fit(windows[valid])
    codes = model.predict(windows).astype(np.int32)
    valid_match = bool(np.array_equal(valid, existing_valid))
    code_match_rate = float(np.mean(codes[valid] == existing_codes[valid])) if np.any(valid) else float("nan")
    return {
        "model": model,
        "mean": mean,
        "std": std,
        "hidden_dim": int(hidden.shape[1]),
        "train_rows": int(hidden.shape[0]),
        "valid_match": valid_match,
        "code_match_rate_same_seed": code_match_rate,
    }


def make_frozen_model_ready(
    *,
    config: dict[str, Any],
    variant: dict[str, Any],
    fold: dict[str, Any],
    out_dir: Path,
    force: bool,
) -> Path:
    norm_cfg = config.get("normalization", {})
    data_cfg = config["data"]
    feature_subset_name = variant.get("feature_subset")
    feature_subset = None
    scaler_out_dir = out_dir / "feature_scalers_frozen"
    if feature_subset_name:
        subsets = config.get("feature_subsets", {})
        feature_subset = list(subsets[str(feature_subset_name)])
        scaler_out_dir = scaler_out_dir / str(feature_subset_name)

    info = prepare_fold_scaled_features(
        raw_csv=ROOT / data_cfg["raw_features_csv"],
        out_dir=scaler_out_dir,
        fold_id=str(fold["fold"]),
        train_start=str(fold["train_start"]),
        train_end=str(fold["train_end_inclusive"]),
        validation_end=str(config["data"]["frozen_test_end"]),
        feature_subset=feature_subset,
        feature_subset_name=str(feature_subset_name) if feature_subset_name else None,
        lower_quantile=float(norm_cfg.get("lower_quantile", 0.01)),
        upper_quantile=float(norm_cfg.get("upper_quantile", 0.99)),
        force=force,
    )
    return Path(info["model_ready_csv"])


def replay_frozen(
    artifact: FrozenArtifact,
    *,
    model_ready_csv: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    start = str(artifact.config["data"]["frozen_test_start"])
    end = str(artifact.config["data"]["frozen_test_end"])
    panel = load_weight_panel(model_ready_csv, start, end)
    env = make_env_from_config(panel, artifact.config, artifact.variant)
    model = InstrumentedPPO.load(
        str(artifact.model_path),
        device="cpu",
        custom_objects={"instrumentation_dir": None},
    )
    model.policy.set_training_mode(False)

    obs, _ = env.reset()
    done = False
    macro_step = 0
    rows: list[dict[str, Any]] = []
    hidden_rows: list[np.ndarray] = []
    value_hidden_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    action_param_rows: list[np.ndarray] = []
    log_prob_rows: list[float] = []
    entropy_rows: list[float] = []
    value_rows: list[float] = []

    while not done:
        layer_out = policy_layers(model.policy, obs)
        action = np.asarray(layer_out["action_mode"], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        daily_steps = info.get("daily_steps")
        if isinstance(daily_steps, list) and daily_steps:
            iterable = list(enumerate(daily_steps))
        else:
            iterable = [(0, info)]

        for daily_idx, daily_info in iterable:
            append_eval_info_row(rows, panel, daily_info, float(daily_info.get("reward", reward)))
            row = rows[-1]
            row["fold"] = str(artifact.fold["fold"])
            row["split"] = "frozen_test"
            row["source_zip"] = artifact.source_zip.name
            row["macro_step_id"] = macro_step
            row["daily_idx_in_macro_step"] = daily_idx
            row["macro_policy_update_date"] = str(info.get("date", daily_info.get("date", "")))

            hidden_rows.append(np.asarray(layer_out["policy_latent_64"], dtype=np.float32))
            value_hidden_rows.append(np.asarray(layer_out["value_latent"], dtype=np.float32))
            action_rows.append(np.asarray(layer_out["action_mode"], dtype=np.float32))
            action_param_rows.append(np.asarray(layer_out["raw_action_params"], dtype=np.float32))
            log_prob_rows.append(float(layer_out["log_prob_mode"]))
            entropy_rows.append(float(layer_out["entropy"]))
            value_rows.append(float(layer_out["value"]))

        macro_step += 1

    env.close()
    behavior = pd.DataFrame(rows)
    behavior["date"] = pd.to_datetime(behavior["date"]).dt.strftime("%Y-%m-%d")
    arrays = {
        "hidden": np.vstack(hidden_rows).astype(np.float32),
        "value_latent": np.vstack(value_hidden_rows).astype(np.float32),
        "action_mode": np.vstack(action_rows).astype(np.float32),
        "raw_action_params": np.vstack(action_param_rows).astype(np.float32),
        "log_prob_mode": np.asarray(log_prob_rows, dtype=np.float32),
        "entropy": np.asarray(entropy_rows, dtype=np.float32),
        "value": np.asarray(value_rows, dtype=np.float32),
    }
    meta = {
        "tickers": list(panel.tickers),
        "feature_columns": list(panel.feature_columns),
        "start": start,
        "end": end,
        "date_count": int(len(panel.dates)),
    }
    return behavior, arrays, meta


def write_npz(
    out_path: Path,
    *,
    behavior: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    behavior_columns: list[str],
    tickers: list[str],
    feature_columns: list[str],
    codebook: dict[str, Any],
    window_length: int,
    k: int,
) -> None:
    behavior_values = behavior[behavior_columns].copy()
    for col in behavior_values.columns:
        if behavior_values[col].dtype == object:
            behavior_values[col] = behavior_values[col].astype(str)

    np.savez_compressed(
        out_path,
        model_id=np.asarray(["R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12"], dtype="U128"),
        split=np.asarray(["frozen_test_2022_2023"], dtype="U64"),
        tickers=np.asarray(tickers, dtype="U32"),
        feature_columns=np.asarray(feature_columns, dtype="U128"),
        test_dates=behavior["date"].astype(str).to_numpy(),
        test_hidden=arrays["hidden"].astype(np.float32),
        test_policy_latent_final=arrays["hidden"].astype(np.float32),
        test_value_latent=arrays["value_latent"].astype(np.float32),
        test_action_mode=arrays["action_mode"].astype(np.float32),
        test_raw_action_params=arrays["raw_action_params"].astype(np.float32),
        test_logprob=arrays["log_prob_mode"].astype(np.float32),
        test_entropy=arrays["entropy"].astype(np.float32),
        test_value=arrays["value"].astype(np.float32),
        test_code_id=behavior["code_id"].to_numpy(dtype=np.int32),
        test_code_valid=behavior["valid"].to_numpy(dtype=bool),
        test_net_return=pd.to_numeric(behavior.get("net_return"), errors="coerce").to_numpy(dtype=np.float32),
        test_gross_return=pd.to_numeric(behavior.get("gross_return"), errors="coerce").to_numpy(dtype=np.float32),
        test_reward=pd.to_numeric(behavior.get("reward"), errors="coerce").to_numpy(dtype=np.float32),
        test_cash_target=pd.to_numeric(behavior.get("cash_target"), errors="coerce").to_numpy(dtype=np.float32),
        test_cash_executed=pd.to_numeric(behavior.get("cash_exec", behavior.get("cash_target")), errors="coerce").to_numpy(dtype=np.float32),
        test_risk_stress=pd.to_numeric(behavior.get("risk_stress"), errors="coerce").to_numpy(dtype=np.float32),
        test_recovery_score=pd.to_numeric(behavior.get("recovery_score"), errors="coerce").to_numpy(dtype=np.float32),
        test_confidence_derisk=pd.to_numeric(behavior.get("confidence_derisk"), errors="coerce").to_numpy(dtype=np.float32),
        test_confidence_rerisk=pd.to_numeric(behavior.get("confidence_rerisk"), errors="coerce").to_numpy(dtype=np.float32),
        zscore_mean=codebook["mean"].astype(np.float32),
        zscore_std=codebook["std"].astype(np.float32),
        kmeans_centers=codebook["model"].cluster_centers_.astype(np.float32),
        window_length=np.asarray([window_length], dtype=np.int32),
        selected_k=np.asarray([k], dtype=np.int32),
        behavior_columns=np.asarray(behavior_columns, dtype="U128"),
        behavior_values=behavior_values.to_numpy(dtype=object),
    )


def make_zip(out_dir: Path, zip_path: Path) -> None:
    names = [
        "r6c_frozen_2022_2023_rollout_package.npz",
        "test_codes.parquet",
        "test_codes.csv",
        "frozen_test_behavior_log_daily.parquet",
        "frozen_test_code_summary.csv",
        "frozen_test_manifest.json",
        "README_R6C_FROZEN_TEST_FOR_JOSEPH.md",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in names:
            path = out_dir / name
            if path.exists():
                zf.write(path, arcname=name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r6c-dir", type=Path, default=DEFAULT_R6C_DIR)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fold", default="fold_2021")
    parser.add_argument("--selected-k", type=int, default=8)
    parser.add_argument("--window-length", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-scalers", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = select_fold_zip(args.r6c_dir, args.fold)
    with tempfile.TemporaryDirectory(prefix="r6c_frozen_") as tmp_name:
        artifact = extract_artifact(zip_path, Path(tmp_name))
        model_ready = make_frozen_model_ready(
            config=artifact.config,
            variant=artifact.variant,
            fold=artifact.fold,
            out_dir=args.out_dir,
            force=args.force_scalers,
        )
        codebook = fit_stage1_kmeans(
            args.stage1_dir,
            k=args.selected_k,
            window_length=args.window_length,
            seed=args.seed,
        )
        behavior, arrays, meta = replay_frozen(artifact, model_ready_csv=model_ready)

    codes, valid = apply_existing_codebook(
        arrays["hidden"],
        mean=codebook["mean"],
        std=codebook["std"],
        model=codebook["model"],
        window_length=args.window_length,
    )
    behavior = behavior.copy()
    behavior["code_id"] = codes.astype(np.int32)
    behavior["valid"] = valid.astype(bool)

    test_codes = behavior[["date", "code_id", "valid"]].copy()
    test_codes.to_parquet(args.out_dir / "test_codes.parquet", index=False)
    test_codes.to_csv(args.out_dir / "test_codes.csv", index=False)

    behavior_columns = select_behavior_columns(behavior)
    behavior_log = behavior[["date", "code_id", "valid"] + [c for c in behavior_columns if c != "date"]].copy()
    behavior_log.to_parquet(args.out_dir / "frozen_test_behavior_log_daily.parquet", index=False)
    behavior_log.to_csv(args.out_dir / "frozen_test_behavior_log_daily.csv", index=False)
    summarize_codes(behavior, k=args.selected_k).to_csv(args.out_dir / "frozen_test_code_summary.csv", index=False)

    write_npz(
        args.out_dir / "r6c_frozen_2022_2023_rollout_package.npz",
        behavior=behavior,
        arrays=arrays,
        behavior_columns=behavior_columns,
        tickers=meta["tickers"],
        feature_columns=meta["feature_columns"],
        codebook=codebook,
        window_length=args.window_length,
        k=args.selected_k,
    )

    manifest = {
        "model": "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1",
        "source_zip": str(zip_path),
        "fold": args.fold,
        "split": "frozen_test",
        "date_start": str(artifact.config["data"]["frozen_test_start"]),
        "date_end": str(artifact.config["data"]["frozen_test_end"]),
        "rows": int(len(behavior)),
        "valid_rows": int(behavior["valid"].sum()),
        "selected_k": args.selected_k,
        "window_length": args.window_length,
        "stage1_codebook_source": str(args.stage1_dir),
        "train_code_valid_match": codebook["valid_match"],
        "train_code_match_rate_same_seed": codebook["code_match_rate_same_seed"],
        "hidden_dim": codebook["hidden_dim"],
        "model_ready_csv": str(model_ready),
        "outputs": {
            "npz": "r6c_frozen_2022_2023_rollout_package.npz",
            "test_codes": "test_codes.parquet",
            "behavior_log": "frozen_test_behavior_log_daily.parquet",
            "code_summary": "frozen_test_code_summary.csv",
        },
    }
    (args.out_dir / "frozen_test_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = f"""# R6c Frozen Test Rollout For Joseph

This is the out-of-sample 2022-2023 frozen rollout requested for Stage 4.

Model:
`R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1`

Window:
`{artifact.config["data"]["frozen_test_start"]}` to `{artifact.config["data"]["frozen_test_end"]}`

Files:

- `r6c_frozen_2022_2023_rollout_package.npz`: same-style NPZ with `test_*` arrays.
- `test_codes.parquet`: `date`, `code_id`, `valid`.
- `frozen_test_behavior_log_daily.parquet`: test codes joined to R6c daily behavior logs.
- `frozen_test_code_summary.csv`: quick per-code frozen-test summary.

Codebook:

- KMeans K={args.selected_k}, window_length={args.window_length}.
- Codebook refit with the same seed/window on the Stage 1 train+validation hidden states.
- `valid == False` marks centered-window boundary rows.

Important:

This is frozen out-of-sample rollout only. It should be used to test whether
the primitive structure found on train+validation holds on 2022-2023.
"""
    (args.out_dir / "README_R6C_FROZEN_TEST_FOR_JOSEPH.md").write_text(readme, encoding="utf-8")
    make_zip(args.out_dir, args.out_dir / "r6c_frozen_2022_2023_rollout_for_Joseph.zip")
    print(f"Wrote Joseph frozen-test package to {args.out_dir}")


if __name__ == "__main__":
    main()
