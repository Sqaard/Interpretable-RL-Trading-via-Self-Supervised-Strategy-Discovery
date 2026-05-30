"""Extract Stage 0.1 policy hidden states and aligned behavior logs.

This is a post-hoc extractor for already trained Stage 0.1 PPO models. It
does not require retraining: each saved model is loaded, exact PPO observations
are replayed through the corresponding validation/train/frozen environment,
and the 64-d policy latent after `mlp_extractor.policy_net` is saved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch as th
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dow30_sectors import get_sector_map  # noqa: E402
from src.ppo.instrumented_ppo import InstrumentedPPO  # noqa: E402
from src.ppo.stage0_1_weight_env import WeightPanel, load_weight_panel, make_env_from_config  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def variant_family(variant: str) -> str:
    if variant.startswith("flat_"):
        return "flat"
    if variant.startswith("hierarchical_"):
        return "hierarchical"
    return "other"


def controller_name(variant: str) -> str:
    for suffix in ("pid", "pd", "pi", "p"):
        if variant.endswith(f"_{suffix}"):
            return suffix
    return "unknown"


def selected_variants(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {v["name"]: v for v in config["variants"] if v.get("enabled", True)}


def infer_feature_csv(run_dir: Path, fold: str) -> Path:
    candidate = run_dir.parent.parent / "feature_scalers" / fold / "model_ready.csv"
    if candidate.exists():
        return candidate
    matches = list(run_dir.parents[2].rglob(f"feature_scalers/{fold}/model_ready.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not locate model_ready.csv for {run_dir} {fold}")


def split_window(config: dict[str, Any], fold_row: pd.Series, split: str) -> tuple[str, str]:
    if split == "train":
        return str(fold_row["train_start"]), str(fold_row["train_end_inclusive"])
    if split == "validation":
        return str(fold_row["validation_start"]), str(fold_row["validation_end_inclusive"])
    if split in {"frozen", "test", "frozen_test"}:
        return str(config["data"]["frozen_test_start"]), str(config["data"]["frozen_test_end"])
    raise ValueError(f"Unknown split: {split}")


def policy_layers(policy: nn.Module, obs: np.ndarray) -> dict[str, np.ndarray]:
    obs_tensor, _ = policy.obs_to_tensor(obs)
    with th.no_grad():
        features = policy.extract_features(obs_tensor)
        x = features
        raw_by_linear: dict[int, th.Tensor] = {}
        post_by_linear: dict[int, th.Tensor] = {}
        pending_linear: int | None = None
        linear_idx = 0
        for module in policy.mlp_extractor.policy_net:
            x = module(x)
            if isinstance(module, nn.Linear):
                linear_idx += 1
                raw_by_linear[linear_idx] = x
                pending_linear = linear_idx
            elif pending_linear is not None:
                post_by_linear[pending_linear] = x
                pending_linear = None
        if pending_linear is not None:
            post_by_linear[pending_linear] = x

        latent_pi = policy.mlp_extractor.policy_net(features)
        latent_vf = policy.mlp_extractor.value_net(features)
        raw_action_params = policy.action_net(latent_pi)
        dist = policy._get_action_dist_from_latent(latent_pi)
        action_mode = dist.mode()
        log_prob = dist.log_prob(action_mode)
        entropy = dist.entropy()
        value = policy.value_net(latent_vf)

        alpha_params = None
        if hasattr(dist, "alpha") and dist.alpha is not None:
            alpha_params = dist.alpha
        elif hasattr(dist, "root_alpha") and dist.root_alpha is not None:
            parts = [dist.root_alpha]
            for _, alpha in getattr(dist, "inner_alphas", []):
                parts.append(alpha)
            alpha_params = th.cat(parts, dim=1)

    out = {
        "obs": obs_tensor.detach().cpu().numpy().squeeze(0),
        "features": features.detach().cpu().numpy().squeeze(0),
        "policy_layer1_raw": raw_by_linear[1].detach().cpu().numpy().squeeze(0),
        "policy_layer1": post_by_linear[1].detach().cpu().numpy().squeeze(0),
        "policy_layer2_raw": raw_by_linear[2].detach().cpu().numpy().squeeze(0),
        "policy_latent_64": post_by_linear[2].detach().cpu().numpy().squeeze(0),
        "value_latent": latent_vf.detach().cpu().numpy().squeeze(0),
        "raw_action_params": raw_action_params.detach().cpu().numpy().squeeze(0),
        "alpha_params": alpha_params.detach().cpu().numpy().squeeze(0) if alpha_params is not None else np.array([]),
        "action_mode": action_mode.detach().cpu().numpy().squeeze(0),
        "log_prob_mode": float(log_prob.detach().cpu().numpy().reshape(-1)[0]),
        "entropy": float(entropy.detach().cpu().numpy().reshape(-1)[0]) if entropy is not None else np.nan,
        "value": float(value.detach().cpu().numpy().reshape(-1)[0]),
    }
    return out


def sector_exposure(weights: np.ndarray, tickers: list[str], sector_map_name: str) -> tuple[list[str], np.ndarray]:
    sector_map = get_sector_map(sector_map_name)
    sectors = sorted({sector_map[t] for t in tickers})
    values = []
    for sector in sectors:
        idx = [i for i, ticker in enumerate(tickers) if sector_map[ticker] == sector]
        values.append(float(weights[idx].sum()))
    return sectors, np.asarray(values, dtype=np.float32)


def extract_run(
    *,
    config: dict[str, Any],
    variant_cfg: dict[str, Any],
    fold_row: pd.Series,
    run_dir: Path,
    split: str,
    out_dir: Path,
    include_observations: bool,
) -> tuple[Path, pd.DataFrame]:
    variant = variant_cfg["name"]
    fold = str(fold_row["fold"]) if "fold" in fold_row.index else str(fold_row.name)
    start, end = split_window(config, fold_row, split)
    feature_csv = infer_feature_csv(run_dir, fold)
    panel = load_weight_panel(feature_csv, start, end)
    env = make_env_from_config(panel, config, variant_cfg)
    model_path = run_dir / "model.zip"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = InstrumentedPPO.load(
        str(model_path),
        device="cpu",
        custom_objects={"instrumentation_dir": None},
    )
    model.policy.set_training_mode(False)

    obs, _ = env.reset()
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray | float | str]] = {
        "date": [],
        "next_date": [],
        "policy_layer1": [],
        "policy_layer1_raw": [],
        "policy_layer2_raw": [],
        "policy_latent_64": [],
        "value_latent": [],
        "raw_action_params": [],
        "alpha_params": [],
        "action_mode": [],
        "prev_weights": [],
        "target_weights": [],
        "executed_weights": [],
        "post_market_weights": [],
        "sector_exposure": [],
        "panel_features": [],
        "stock_returns_next": [],
        "stock_prices": [],
        "reward": [],
        "gross_return": [],
        "net_return": [],
        "turnover_l1": [],
        "stock_turnover_l1": [],
        "transaction_cost": [],
        "cash_weight": [],
        "drawdown": [],
        "drawdown_increment": [],
        "concentration": [],
        "controller_p_l1": [],
        "controller_i_l1": [],
        "controller_d_l1": [],
        "controller_delta_l1_before_cap": [],
        "projection_residual_l1": [],
        "log_prob_mode": [],
        "entropy": [],
        "value": [],
    }
    if include_observations:
        arrays["observations"] = []

    done = False
    step = 0
    sector_names: list[str] | None = None
    while not done:
        prev_weights = env.previous_weights.copy()
        layer_out = policy_layers(model.policy, obs)
        action = layer_out["action_mode"].astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        target = np.asarray(info["target_weights"], dtype=np.float32)
        executed = np.asarray(info["executed_weights"], dtype=np.float32)
        post = np.asarray(info["post_market_weights"], dtype=np.float32)
        sectors, sector_values = sector_exposure(
            executed[: len(panel.tickers)], panel.tickers, config.get("universe", {}).get("sector_map", "dow30_static")
        )
        if sector_names is None:
            sector_names = sectors

        arrays["date"].append(str(info["date"]))
        arrays["next_date"].append(str(info["next_date"]))
        for key in [
            "policy_layer1",
            "policy_layer1_raw",
            "policy_layer2_raw",
            "policy_latent_64",
            "value_latent",
            "raw_action_params",
            "alpha_params",
            "action_mode",
        ]:
            arrays[key].append(np.asarray(layer_out[key], dtype=np.float32))
        if include_observations:
            arrays["observations"].append(np.asarray(layer_out["obs"], dtype=np.float32))
        arrays["prev_weights"].append(prev_weights.astype(np.float32))
        arrays["target_weights"].append(target)
        arrays["executed_weights"].append(executed)
        arrays["post_market_weights"].append(post)
        arrays["sector_exposure"].append(sector_values)
        arrays["panel_features"].append(panel.features[step].astype(np.float32))
        arrays["stock_returns_next"].append(panel.returns_next[step].astype(np.float32))
        arrays["stock_prices"].append(panel.prices[step].astype(np.float32))

        scalar_keys = [
            "gross_return",
            "net_return",
            "turnover_l1",
            "stock_turnover_l1",
            "transaction_cost",
            "drawdown",
            "drawdown_increment",
            "concentration",
            "controller_p_l1",
            "controller_i_l1",
            "controller_d_l1",
            "controller_delta_l1_before_cap",
            "projection_residual_l1",
        ]
        arrays["reward"].append(float(reward))
        arrays["cash_weight"].append(float(executed[-1]))
        arrays["log_prob_mode"].append(float(layer_out["log_prob_mode"]))
        arrays["entropy"].append(float(layer_out["entropy"]))
        arrays["value"].append(float(layer_out["value"]))
        for key in scalar_keys:
            arrays[key].append(float(info[key]))

        row = {
            "variant": variant,
            "family": variant_family(variant),
            "controller": controller_name(variant),
            "fold": fold,
            "split": split,
            "row_in_run": step,
            "date": str(info["date"]),
            "next_date": str(info["next_date"]),
            "reward": float(reward),
            "gross_return": float(info["gross_return"]),
            "net_return": float(info["net_return"]),
            "turnover_l1": float(info["turnover_l1"]),
            "stock_turnover_l1": float(info["stock_turnover_l1"]),
            "cash_weight": float(executed[-1]),
            "drawdown": float(info["drawdown"]),
            "concentration": float(info["concentration"]),
            "controller_p_l1": float(info["controller_p_l1"]),
            "controller_i_l1": float(info["controller_i_l1"]),
            "controller_d_l1": float(info["controller_d_l1"]),
            "projection_residual_l1": float(info["projection_residual_l1"]),
            "log_prob_mode": float(layer_out["log_prob_mode"]),
            "entropy": float(layer_out["entropy"]),
            "value": float(layer_out["value"]),
        }
        for i, ticker in enumerate(panel.tickers):
            row[f"target_weight_{ticker}"] = float(target[i])
            row[f"executed_weight_{ticker}"] = float(executed[i])
        row["target_weight_CASH"] = float(target[-1])
        row["executed_weight_CASH"] = float(executed[-1])
        rows.append(row)
        step += 1

    env.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{variant}__{fold}__{split}.npz"
    save_data: dict[str, Any] = {
        "variant": np.asarray(variant),
        "family": np.asarray(variant_family(variant)),
        "controller": np.asarray(controller_name(variant)),
        "fold": np.asarray(fold),
        "split": np.asarray(split),
        "dates": np.asarray(arrays["date"], dtype=object),
        "next_dates": np.asarray(arrays["next_date"], dtype=object),
        "tickers": np.asarray(panel.tickers, dtype=object),
        "asset_symbols": np.asarray(panel.tickers + ["CASH"], dtype=object),
        "feature_columns": np.asarray(panel.feature_columns, dtype=object),
        "sector_names": np.asarray(sector_names or [], dtype=object),
    }
    for key, values in arrays.items():
        if key in {"date", "next_date"}:
            continue
        if key == "alpha_params" and len({np.asarray(v).shape[0] for v in values}) > 1:
            save_data[key] = np.asarray(values, dtype=object)
        else:
            save_data[key] = np.asarray(values)
    np.savez_compressed(npz_path, **save_data)

    index = pd.DataFrame(rows)
    index["npz_path"] = str(npz_path.relative_to(ROOT))
    index["model_path"] = str(model_path.relative_to(ROOT))
    index["feature_csv"] = str(feature_csv.relative_to(ROOT)) if feature_csv.is_relative_to(ROOT) else str(feature_csv)
    return npz_path, index


def discover_runs(metrics_csv: Path, variants: list[str] | None, folds: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    if variants:
        df = df[df["variant"].isin(variants)].copy()
    if folds:
        df = df[df["fold"].isin(folds)].copy()
    if df.empty:
        raise ValueError("No runs selected.")
    return df.sort_values(["variant", "fold"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage0_1_stabilized_ppo.yaml")
    parser.add_argument("--metrics-csv", default="artifacts/stage0_1/analysis/stage0_1_completed_run_metrics.csv")
    parser.add_argument("--folds-csv", default="configs/stage0_1_walk_forward_folds.csv")
    parser.add_argument("--out-dir", default="artifacts/stage0_1/hidden_state_package")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=["validation"])
    parser.add_argument("--include-observations", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve(args.config))
    variant_cfgs = selected_variants(config)
    folds = pd.read_csv(resolve(args.folds_csv)).set_index("fold")
    runs = discover_runs(resolve(args.metrics_csv), args.variants, args.folds)
    out_dir = resolve(args.out_dir)
    per_run_dir = out_dir / "per_run"
    all_index: list[pd.DataFrame] = []
    manifest: dict[str, Any] = {
        "created_by": "scripts/extract_stage0_1_hidden_state_package.py",
        "config": str(resolve(args.config).relative_to(ROOT)),
        "metrics_csv": str(resolve(args.metrics_csv).relative_to(ROOT)),
        "splits": args.splits,
        "include_observations": bool(args.include_observations),
        "runs": [],
    }

    for _, run in runs.iterrows():
        variant = str(run["variant"])
        fold = str(run["fold"])
        if variant not in variant_cfgs:
            print(f"[skip] {variant} not enabled in config")
            continue
        if fold not in folds.index:
            print(f"[skip] {fold} not found in folds csv")
            continue
        run_dir = Path(str(run["run_dir"]))
        if not run_dir.exists():
            print(f"[skip] missing run_dir: {run_dir}")
            continue
        for split in args.splits:
            npz_path = per_run_dir / f"{variant}__{fold}__{split}.npz"
            if npz_path.exists() and not args.force:
                print(f"[skip existing] {npz_path.relative_to(ROOT)}")
                index_path = out_dir / "hidden_state_index.csv"
                continue
            print(f"[extract] variant={variant} fold={fold} split={split}")
            saved, index = extract_run(
                config=config,
                variant_cfg=variant_cfgs[variant],
                fold_row=folds.loc[fold],
                run_dir=run_dir,
                split=split,
                out_dir=per_run_dir,
                include_observations=args.include_observations,
            )
            all_index.append(index)
            manifest["runs"].append(
                {
                    "variant": variant,
                    "fold": fold,
                    "split": split,
                    "npz_path": str(saved.relative_to(ROOT)),
                    "rows": int(len(index)),
                    "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    if all_index:
        full_index = pd.concat(all_index, ignore_index=True)
        full_index.to_csv(out_dir / "hidden_state_index.csv", index=False)
    elif not (out_dir / "hidden_state_index.csv").exists():
        raise RuntimeError("No hidden states were extracted and no existing index was found.")

    with (out_dir / "hidden_state_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote hidden-state package to {out_dir}")


if __name__ == "__main__":
    main()
