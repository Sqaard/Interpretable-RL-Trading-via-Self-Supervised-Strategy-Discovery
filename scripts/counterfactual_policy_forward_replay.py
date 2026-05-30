"""Counterfactual policy-forward replay for Stage 0.1 controllers.

Frozen-intent replay keeps logged anchors fixed. This script is one level
closer to reality: it loads the trained PPO model and evaluates it in an env
whose controller config has been changed. The policy is called on the replayed
portfolio state, so anchors can react to counterfactual weights/drawdown/cash.

This still does not retrain PPO. It is a candidate filter before cloud runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from offline_replay_groupaware_topk import fold_from_text, groups_from_e9, infer_tickers  # noqa: E402
from src.ppo.instrumented_ppo import InstrumentedPPO  # noqa: E402
from src.ppo.stage0_1_train import evaluate_model, load_weight_panel  # noqa: E402


ARTIFACT_DIR = ROOT / "artifacts" / "stage0_1"
REPORT_DIR = ROOT / "reports" / "r_k_window_analysis"


@dataclass(frozen=True)
class CounterfactualCandidate:
    name: str
    top_k_buy: int | None = None
    top_k_sell: int | None = None
    rotation_budget: float = 0.0
    group_aware: bool = False
    default_group_cap: float = 0.45
    pressure_weight: float = 1.0
    capacity_weight: float = 0.5
    sell_overweight_weight: float = 1.0
    priority_floor: float = 0.05
    group_residual_quality: bool = False
    group_residual_buy_rank_weight: float = 0.60
    group_residual_sell_rank_weight: float = 0.85
    group_residual_buy_breadth_weight: float = 0.50
    group_residual_sell_breadth_weight: float = 0.50
    group_residual_min_multiplier: float = 0.25
    group_residual_max_multiplier: float = 2.50
    disable_incremental_topk: bool = False
    k_root_days: int | None = None
    k_stock_days: int | None = None
    recovery_trigger_threshold: float | None = None
    derisk_early_update_threshold: float | None = None
    recovery_min_confidence_rerisk: float | None = None
    risk_break_min_confidence_derisk: float | None = None
    recovery_max_risk_stress: float | None = None
    early_update_cooldown_days: int | None = None
    recovery_persistence_days: int | None = None
    risk_break_persistence_days: int | None = None
    rerisk_min_scale: float | None = None
    derisk_min_scale: float | None = None
    risk_stress_scale: float = 1.0
    recovery_score_scale: float = 1.0
    recovery_residual_scale: float = 1.0
    recovery_market_scale: float = 1.0
    derisk_market_down_scale: float = 1.0
    vix_shock_scale: float = 1.0
    feature_family: str = ""
    risk_aware_topk: bool = False
    buy_gate_min_confidence_rerisk: float | None = None
    buy_gate_min_recovery_score: float | None = None
    buy_gate_max_risk_stress: float | None = None
    buy_gate_min_residual_breadth_excess_5d: float | None = None
    buy_gate_min_residual_breadth_excess_20d: float | None = None
    rotation_stress_gate_enabled: bool = False
    rotation_stress_start: float | None = None
    rotation_stress_full: float | None = None
    rotation_stress_min_scale: float | None = None
    sell_risk_break_weight: float = 0.0
    sell_residual_deterioration_weight: float = 0.0
    sell_confidence_derisk_weight: float = 0.0


def default_candidates() -> list[CounterfactualCandidate]:
    return [
        CounterfactualCandidate("pf_original"),
        CounterfactualCandidate("pf_no_incremental_topk", disable_incremental_topk=True),
        CounterfactualCandidate("pf_topk_b5_s5_rot0", top_k_buy=5, top_k_sell=5),
        CounterfactualCandidate("pf_topk_b5_s8_rot0", top_k_buy=5, top_k_sell=8),
        CounterfactualCandidate("pf_topk_b8_s8_rot0", top_k_buy=8, top_k_sell=8),
        CounterfactualCandidate(
            "pf_groupaware_cap45_b8_s8_rot0",
            top_k_buy=8,
            top_k_sell=8,
            group_aware=True,
            default_group_cap=0.45,
            pressure_weight=1.0,
            capacity_weight=0.5,
            sell_overweight_weight=1.0,
        ),
        CounterfactualCandidate(
            "pf_groupaware_softcap_b8_s8_rot0",
            top_k_buy=8,
            top_k_sell=8,
            group_aware=True,
            default_group_cap=0.60,
            pressure_weight=1.0,
            capacity_weight=0.25,
            sell_overweight_weight=0.75,
        ),
        CounterfactualCandidate(
            "pf_riskaware_groupaware_b10_s12_rot005",
            top_k_buy=10,
            top_k_sell=12,
            rotation_budget=0.005,
            group_aware=True,
            default_group_cap=0.60,
            pressure_weight=1.0,
            capacity_weight=0.25,
            sell_overweight_weight=1.25,
            risk_aware_topk=True,
            buy_gate_min_confidence_rerisk=0.55,
            buy_gate_min_recovery_score=0.60,
            buy_gate_max_risk_stress=0.85,
            buy_gate_min_residual_breadth_excess_5d=0.0,
            rotation_stress_gate_enabled=True,
            rotation_stress_start=0.55,
            rotation_stress_full=0.90,
            rotation_stress_min_scale=0.0,
            sell_risk_break_weight=1.0,
            sell_residual_deterioration_weight=1.0,
            sell_confidence_derisk_weight=0.25,
        ),
    ]


def read_zip_member(zf: zipfile.ZipFile, suffix: str) -> bytes:
    matches = [name for name in zf.namelist() if name.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"No member ending with {suffix}")
    matches.sort(key=len)
    return zf.read(matches[0])


def read_zip_json(zf: zipfile.ZipFile, suffix: str) -> dict[str, Any]:
    return json.loads(read_zip_member(zf, suffix).decode("utf-8"))


def read_zip_yaml(zf: zipfile.ZipFile, suffix: str) -> dict[str, Any]:
    return yaml.safe_load(read_zip_member(zf, suffix).decode("utf-8"))


def safe_selection_score(values: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    return float(s.mean() - 0.5 * std)


def robust_selection_score(values: pd.Series) -> float:
    """Lower-tail-aware score for controller search.

    The ordinary selection score is useful, but HCS should not promote a rule
    that wins only by one strong fold and has a weak lower tail. This keeps the
    old mean/std score and adds an explicit q25 shortfall penalty.
    """

    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    q25 = float(s.quantile(0.25))
    lower_tail_shortfall = max(mean - q25, 0.0)
    return float(mean - 0.5 * std - 0.25 * lower_tail_shortfall)


def extract_minimal_artifact(zip_path: Path, tmp_dir: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as zf:
        config = read_zip_yaml(zf, "config.yaml")
        metadata = read_zip_json(zf, "/metadata.json")
        model_bytes = read_zip_member(zf, "/model.zip")
        model_ready_bytes = read_zip_member(zf, "/model_ready.csv")
        validation_daily_bytes = read_zip_member(zf, "/validation_daily.csv")

    model_path = tmp_dir / "model.zip"
    model_ready_path = tmp_dir / "model_ready.csv"
    validation_daily_path = tmp_dir / "validation_daily.csv"
    model_path.write_bytes(model_bytes)
    model_ready_path.write_bytes(model_ready_bytes)
    validation_daily_path.write_bytes(validation_daily_bytes)
    return {
        "config": config,
        "metadata": metadata,
        "model_path": model_path,
        "model_ready_path": model_ready_path,
        "validation_daily_path": validation_daily_path,
    }


def groups_to_variant(root_split: dict[str, Any], groups: dict[str, list[int]]) -> None:
    if not groups:
        return
    names = list(groups.keys())
    root_split["group_names"] = names
    root_split["group_indices"] = [[int(idx) for idx in groups[name]] for name in names]


def scale_signal_weights(signal_cfg: dict[str, Any], *, scale: float, feature_patterns: tuple[str, ...] | None = None) -> None:
    if abs(float(scale) - 1.0) <= 1e-12:
        return
    weights = signal_cfg.get("feature_weights", {})
    if not isinstance(weights, dict):
        return
    patterns = tuple(pattern.lower() for pattern in feature_patterns or ())
    for feature, value in list(weights.items()):
        feature_l = str(feature).lower()
        if patterns and not any(pattern in feature_l for pattern in patterns):
            continue
        try:
            weights[feature] = float(value) * float(scale)
        except Exception:
            continue


def apply_confidence_overrides(k_cfg: dict[str, Any], candidate: CounterfactualCandidate) -> None:
    confidence_cfg = k_cfg.get("confidence_stop_recovery")
    if not isinstance(confidence_cfg, dict):
        return

    direct_fields = {
        "recovery_trigger_threshold": candidate.recovery_trigger_threshold,
        "derisk_early_update_threshold": candidate.derisk_early_update_threshold,
        "recovery_min_confidence_rerisk": candidate.recovery_min_confidence_rerisk,
        "risk_break_min_confidence_derisk": candidate.risk_break_min_confidence_derisk,
        "recovery_max_risk_stress": candidate.recovery_max_risk_stress,
        "early_update_cooldown_days": candidate.early_update_cooldown_days,
        "recovery_persistence_days": candidate.recovery_persistence_days,
        "risk_break_persistence_days": candidate.risk_break_persistence_days,
    }
    for key, value in direct_fields.items():
        if value is not None:
            confidence_cfg[key] = value

    slice_cfg = confidence_cfg.setdefault("confidence_slice", {})
    if candidate.rerisk_min_scale is not None:
        slice_cfg["rerisk_min_scale"] = float(candidate.rerisk_min_scale)
    if candidate.derisk_min_scale is not None:
        slice_cfg["derisk_min_scale"] = float(candidate.derisk_min_scale)

    risk_cfg = confidence_cfg.setdefault("risk_stress", {})
    recovery_cfg = confidence_cfg.setdefault("recovery_score", {})
    scale_signal_weights(risk_cfg, scale=candidate.risk_stress_scale)
    scale_signal_weights(recovery_cfg, scale=candidate.recovery_score_scale)
    scale_signal_weights(
        recovery_cfg,
        scale=candidate.recovery_residual_scale,
        feature_patterns=("residual", "breadth"),
    )
    scale_signal_weights(
        recovery_cfg,
        scale=candidate.recovery_market_scale,
        feature_patterns=("sp500", "universe_return", "market_up"),
    )
    scale_signal_weights(
        risk_cfg,
        scale=candidate.derisk_market_down_scale,
        feature_patterns=("sp500", "universe_return", "market_down", "trend_delta_down"),
    )
    scale_signal_weights(
        risk_cfg,
        scale=candidate.vix_shock_scale,
        feature_patterns=("vix", "vix_surprise"),
    )


def apply_k_window_overrides(k_cfg: dict[str, Any], candidate: CounterfactualCandidate) -> None:
    if candidate.k_root_days is None and candidate.k_stock_days is None:
        return
    current_window = max(1, int(k_cfg.get("window_days", k_cfg.get("K", 1))))
    dual_cfg = k_cfg.setdefault("dual_window", {})
    dual_enabled = bool(dual_cfg.get("enabled", False))
    root_days = int(candidate.k_root_days or dual_cfg.get("root_window_days", current_window))
    stock_days = int(candidate.k_stock_days or dual_cfg.get("stock_window_days", current_window))
    root_days = max(1, root_days)
    stock_days = max(1, stock_days)

    if dual_enabled or root_days != stock_days:
        dual_cfg["enabled"] = True
        dual_cfg["root_window_days"] = root_days
        dual_cfg["stock_window_days"] = stock_days
        k_cfg["window_days"] = root_days
    else:
        k_cfg["window_days"] = root_days


def apply_candidate(
    base_variant: dict[str, Any],
    candidate: CounterfactualCandidate,
    *,
    groups: dict[str, list[int]] | None,
) -> dict[str, Any]:
    variant = copy.deepcopy(base_variant)
    variant["name"] = f"{base_variant.get('name', 'variant')}__{candidate.name}"
    root_split = variant.setdefault("root_split", {})
    if candidate.group_aware and groups:
        groups_to_variant(root_split, groups)

    k_cfg = root_split.setdefault("k_window_execution", {})
    apply_k_window_overrides(k_cfg, candidate)
    apply_confidence_overrides(k_cfg, candidate)
    if candidate.name == "pf_original":
        return variant

    if candidate.disable_incremental_topk:
        topk_cfg = k_cfg.setdefault("incremental_topk_flow", {})
        topk_cfg["enabled"] = False
        return variant

    if candidate.top_k_buy is None and candidate.top_k_sell is None:
        return variant

    topk_cfg = k_cfg.setdefault("incremental_topk_flow", {})
    topk_cfg.update(
        {
            "enabled": True,
            "top_k_buy": int(candidate.top_k_buy or 0),
            "top_k_sell": int(candidate.top_k_sell or 0),
            "priority_reference": "window_start",
            "sell_expansion_enabled": True,
            "rotation_enabled": bool(candidate.rotation_budget > 0.0),
            "rotation_budget_l1_per_day": float(candidate.rotation_budget),
            "eps": 1e-10,
        }
    )
    if candidate.group_aware:
        group_cfg = {
            "enabled": True,
            "default_group_cap": float(candidate.default_group_cap),
            "priority_floor": float(candidate.priority_floor),
            "pressure_weight": float(candidate.pressure_weight),
            "capacity_weight": float(candidate.capacity_weight),
            "sell_overweight_weight": float(candidate.sell_overweight_weight),
            "min_multiplier": 0.05,
            "max_multiplier": 5.0,
        }
        if candidate.group_residual_quality:
            group_cfg["residual_quality"] = {
                "enabled": True,
                "rank_5d_weight": 0.70,
                "rank_20d_weight": 0.30,
                "residual_5d_weight": 0.70,
                "residual_20d_weight": 0.30,
                "buy_rank_weight": float(candidate.group_residual_buy_rank_weight),
                "sell_rank_weight": float(candidate.group_residual_sell_rank_weight),
                "buy_breadth_weight": float(candidate.group_residual_buy_breadth_weight),
                "sell_breadth_weight": float(candidate.group_residual_sell_breadth_weight),
                "min_multiplier": float(candidate.group_residual_min_multiplier),
                "max_multiplier": float(candidate.group_residual_max_multiplier),
                "residual_positive_threshold": 0.0,
            }
        topk_cfg["group_aware"] = group_cfg
    else:
        topk_cfg.pop("group_aware", None)
    if candidate.risk_aware_topk:
        topk_cfg["risk_aware"] = {
            "enabled": True,
            "buy_gate": {
                "enabled": True,
                "min_confidence_rerisk": float(candidate.buy_gate_min_confidence_rerisk or 0.0),
                "min_recovery_score": float(candidate.buy_gate_min_recovery_score or 0.0),
                "max_risk_stress": float(candidate.buy_gate_max_risk_stress if candidate.buy_gate_max_risk_stress is not None else 1.0),
                "min_residual_breadth_excess_5d": float(
                    candidate.buy_gate_min_residual_breadth_excess_5d
                    if candidate.buy_gate_min_residual_breadth_excess_5d is not None
                    else -1.0
                ),
                "min_residual_breadth_excess_20d": float(
                    candidate.buy_gate_min_residual_breadth_excess_20d
                    if candidate.buy_gate_min_residual_breadth_excess_20d is not None
                    else -1.0
                ),
            },
            "rotation_stress_gate": {
                "enabled": bool(candidate.rotation_stress_gate_enabled),
                "stress_start": float(candidate.rotation_stress_start or 0.55),
                "stress_full": float(candidate.rotation_stress_full or 0.90),
                "min_scale": float(candidate.rotation_stress_min_scale if candidate.rotation_stress_min_scale is not None else 0.0),
                "max_scale": 1.0,
            },
            "sell_side": {
                "risk_break_weight": float(candidate.sell_risk_break_weight),
                "residual_deterioration_weight": float(candidate.sell_residual_deterioration_weight),
                "confidence_derisk_weight": float(candidate.sell_confidence_derisk_weight),
                "min_multiplier": 0.05,
                "max_multiplier": 5.0,
            },
        }
    else:
        topk_cfg.pop("risk_aware", None)
    return variant


def summarize_daily(path: Path) -> dict[str, float]:
    daily = pd.read_csv(path)
    summary: dict[str, float] = {}
    for col in [
        "incremental_topk_enabled",
        "incremental_topk_flow_l1",
        "incremental_topk_selected_buy_count",
        "incremental_topk_selected_sell_count",
        "incremental_topk_risk_aware_enabled",
        "incremental_topk_buy_allowed",
        "incremental_topk_rotation_stress_gate",
        "incremental_topk_rotation_budget_effective",
        "incremental_topk_sell_multiplier_mean",
        "incremental_topk_residual_deterioration_mean",
        "incremental_topk_group_residual_quality_enabled",
        "incremental_topk_group_residual_buy_multiplier_mean",
        "incremental_topk_group_residual_sell_multiplier_mean",
        "incremental_topk_group_residual_rank_quality_mean",
        "incremental_topk_group_residual_breadth_excess_mean",
        "stock_slice_suppressed_l1",
        "confidence_slice_suppressed_l1",
        "confidence_rerisk",
        "confidence_derisk",
        "risk_stress",
        "recovery_score",
        "recovery_trigger",
        "risk_break_trigger",
        "stop_active",
        "window_closed_early",
        "k_window_effective_days",
        "dual_root_anchor_refreshed",
        "dual_stock_anchor_refreshed",
    ]:
        if col in daily.columns:
            summary[f"{col}_mean"] = float(pd.to_numeric(daily[col], errors="coerce").mean())
    return summary


def process_zip(
    zip_path: Path,
    *,
    e9_rescorr_dir: Path,
    output_dir: Path,
    candidates: list[CounterfactualCandidate],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp_dir = Path(tmp_name)
        artifact = extract_minimal_artifact(zip_path, tmp_dir)
        metadata = artifact["metadata"]
        fold_dict = metadata["fold"]
        fold = str(fold_dict.get("fold") or fold_from_text(str(zip_path)))
        source_experiment = str(metadata["variant"]["name"])

        panel = load_weight_panel(
            artifact["model_ready_path"],
            str(fold_dict["validation_start"]),
            str(fold_dict["validation_end_inclusive"]),
        )
        logged_daily = pd.read_csv(artifact["validation_daily_path"])
        tickers = infer_tickers(logged_daily)
        groups = groups_from_e9(e9_rescorr_dir, fold, tickers)
        model = InstrumentedPPO.load(artifact["model_path"], device="cpu")

        for candidate in candidates:
            candidate_dir = output_dir / "daily" / source_experiment / fold / candidate.name
            candidate_dir.mkdir(parents=True, exist_ok=True)
            variant = apply_candidate(metadata["variant"], candidate, groups=groups)
            try:
                summary = evaluate_model(
                    model,
                    panel,
                    artifact["config"],
                    variant,
                    candidate_dir,
                    "validation",
                )
                status = "ok"
                error = ""
            except Exception as exc:  # keep search robust
                summary = {}
                status = "error"
                error = repr(exc)

            row = {
                "source_experiment": source_experiment,
                "fold": fold,
                "zip": zip_path.name,
                "candidate": candidate.name,
                "status": status,
                "error": error,
                "group_count": len(groups),
                **asdict(candidate),
                **summary,
            }
            daily_path = candidate_dir / "validation_daily.csv"
            if daily_path.exists():
                row.update(summarize_daily(daily_path))
            rows.append(row)
    return rows


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    ok = rows[rows["status"] == "ok"].copy()
    out_rows: list[dict[str, Any]] = []
    if ok.empty:
        return pd.DataFrame()
    for (source, candidate), group in ok.groupby(["source_experiment", "candidate"], dropna=False):
        sharpe = pd.to_numeric(group["sharpe"], errors="coerce")
        out: dict[str, Any] = {
            "source_experiment": source,
            "candidate": candidate,
            "folds": int(len(group)),
            "selection_score": safe_selection_score(sharpe),
            "robust_selection_score": robust_selection_score(sharpe),
            "mean_sharpe": float(sharpe.mean()),
            "sample_std_sharpe": float(sharpe.std(ddof=1)) if len(sharpe) > 1 else 0.0,
            "q25_sharpe": float(sharpe.quantile(0.25)),
            "min_sharpe": float(sharpe.min()),
            "folds_positive_sharpe": int((sharpe > 0.0).sum()),
            "mean_return_pct": float(pd.to_numeric(group["return_pct"], errors="coerce").mean()),
            "mean_max_drawdown": float(pd.to_numeric(group["max_drawdown"], errors="coerce").mean()),
            "mean_cash": float(pd.to_numeric(group["cash_weight_mean"], errors="coerce").mean()),
            "mean_turnover_l1": float(pd.to_numeric(group["turnover_l1_mean"], errors="coerce").mean()),
            "mean_stock_turnover_l1": float(pd.to_numeric(group["stock_turnover_l1_mean"], errors="coerce").mean()),
        }
        for col in group.columns:
            if col.endswith("_mean") and col not in out:
                vals = pd.to_numeric(group[col], errors="coerce")
                if vals.notna().any():
                    out[f"mean_{col}"] = float(vals.mean())
        out_rows.append(out)
    return pd.DataFrame(out_rows).sort_values(["source_experiment", "selection_score"], ascending=[True, False])


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def baseline_delta_summary(
    fold_summary: pd.DataFrame,
    *,
    min_promote_edge: float = 0.02,
    max_worst_drawdown_degradation: float = 0.005,
) -> pd.DataFrame:
    """Compare each no-retrain policy-forward candidate against its source baseline.

    Policy-forward replay is useful as a filter, but it can over-promote small
    controller-only gains because the PPO policy is not retrained under the new
    controller/reward trajectory. This summary makes that risk explicit.
    """

    ok = fold_summary[fold_summary["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    baseline_preference = ["pf_original", "pf_r6c_reference"]
    for source, group in ok.groupby("source_experiment", dropna=False):
        baseline_name = None
        for candidate_name in baseline_preference:
            if (group["candidate"] == candidate_name).any():
                baseline_name = candidate_name
                break
        if baseline_name is None:
            continue

        base = group[group["candidate"] == baseline_name].copy()
        base = base[["fold", "sharpe", "max_drawdown", "cash_weight_mean", "turnover_l1_mean", "target_to_executed_l1_mean"]]
        base = base.rename(
            columns={
                "sharpe": "baseline_sharpe",
                "max_drawdown": "baseline_max_drawdown",
                "cash_weight_mean": "baseline_cash_weight_mean",
                "turnover_l1_mean": "baseline_turnover_l1_mean",
                "target_to_executed_l1_mean": "baseline_target_to_executed_l1_mean",
            }
        )

        baseline_selection = safe_selection_score(pd.to_numeric(base["baseline_sharpe"], errors="coerce"))
        for candidate, cand in group.groupby("candidate", dropna=False):
            merged = cand.merge(base, on="fold", how="inner")
            if merged.empty:
                continue
            sharpe = _numeric_series(merged, "sharpe")
            base_sharpe = _numeric_series(merged, "baseline_sharpe")
            drawdown = _numeric_series(merged, "max_drawdown")
            base_drawdown = _numeric_series(merged, "baseline_max_drawdown")
            cash = _numeric_series(merged, "cash_weight_mean")
            base_cash = _numeric_series(merged, "baseline_cash_weight_mean")
            turnover = _numeric_series(merged, "turnover_l1_mean")
            base_turnover = _numeric_series(merged, "baseline_turnover_l1_mean")
            target_gap = _numeric_series(merged, "target_to_executed_l1_mean")
            base_target_gap = _numeric_series(merged, "baseline_target_to_executed_l1_mean")

            candidate_selection = safe_selection_score(sharpe)
            delta_selection = candidate_selection - baseline_selection
            delta_sharpe = sharpe - base_sharpe
            delta_drawdown = drawdown - base_drawdown  # positive means less negative drawdown, therefore better

            first = cand.iloc[0]
            controller_drift_count = 0
            for flag_col in ["group_aware", "group_residual_quality", "risk_aware_topk", "disable_incremental_topk"]:
                if bool(first.get(flag_col, False)):
                    controller_drift_count += 1
            for value_col in [
                "top_k_buy",
                "top_k_sell",
                "rotation_budget",
                "k_root_days",
                "k_stock_days",
                "recovery_trigger_threshold",
                "derisk_early_update_threshold",
                "recovery_min_confidence_rerisk",
                "risk_break_min_confidence_derisk",
            ]:
                value = first.get(value_col)
                if pd.notna(value) and value not in (0, 0.0, "", None):
                    controller_drift_count += 1

            no_retrain_penalty = 0.005 * min(controller_drift_count, 4)
            drawdown_penalty = max(-float(delta_drawdown.min(skipna=True)) - max_worst_drawdown_degradation, 0.0)
            conservative_delta_score = delta_selection - min_promote_edge - no_retrain_penalty - drawdown_penalty

            folds_sharpe_better = int((delta_sharpe > 0.0).sum())
            folds_drawdown_not_worse = int((delta_drawdown >= -max_worst_drawdown_degradation).sum())
            promote = (
                delta_selection >= min_promote_edge
                and folds_sharpe_better >= max(3, int(np.ceil(0.75 * len(merged))))
                and folds_drawdown_not_worse == len(merged)
                and conservative_delta_score >= 0.0
            )

            rows.append(
                {
                    "source_experiment": source,
                    "candidate": candidate,
                    "baseline_candidate": baseline_name,
                    "folds": int(len(merged)),
                    "baseline_selection_score": float(baseline_selection),
                    "candidate_selection_score": float(candidate_selection),
                    "delta_selection_score": float(delta_selection),
                    "conservative_delta_score": float(conservative_delta_score),
                    "promotion_decision": "promote" if promote else "do_not_promote",
                    "folds_sharpe_better": folds_sharpe_better,
                    "folds_drawdown_not_worse": folds_drawdown_not_worse,
                    "mean_delta_sharpe": float(delta_sharpe.mean(skipna=True)),
                    "worst_delta_sharpe": float(delta_sharpe.min(skipna=True)),
                    "mean_delta_max_drawdown": float(delta_drawdown.mean(skipna=True)),
                    "worst_delta_max_drawdown": float(delta_drawdown.min(skipna=True)),
                    "mean_delta_cash": float((cash - base_cash).mean(skipna=True)),
                    "mean_delta_turnover_l1": float((turnover - base_turnover).mean(skipna=True)),
                    "mean_delta_target_to_executed_l1": float((target_gap - base_target_gap).mean(skipna=True)),
                    "controller_drift_count": int(controller_drift_count),
                    "no_retrain_penalty": float(no_retrain_penalty),
                    "drawdown_penalty": float(drawdown_penalty),
                    "min_promote_edge": float(min_promote_edge),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["source_experiment", "promotion_decision", "conservative_delta_score", "delta_selection_score"],
        ascending=[True, True, False, False],
    )


def markdown_table_safe(obj: pd.DataFrame | pd.Series, **kwargs: Any) -> str:
    try:
        return obj.to_markdown(**kwargs)
    except ImportError:
        if isinstance(obj, pd.Series):
            return obj.to_frame().to_string(**{k: v for k, v in kwargs.items() if k != "index"})
        return obj.to_string(**kwargs)


def write_report(output_dir: Path, fold_summary: pd.DataFrame, agg: pd.DataFrame, deltas: pd.DataFrame) -> None:
    lines = [
        "# Counterfactual Policy-Forward Replay",
        "",
        "This replay loads the trained PPO model and evaluates it under modified controller configs.",
        "Unlike frozen-intent replay, policy anchors are recomputed from counterfactual portfolio state.",
        "",
        "## Outputs",
        "",
        "- `counterfactual_fold_summary.csv`",
        "- `counterfactual_aggregate_summary.csv`",
        "- `daily/<source>/<fold>/<candidate>/validation_daily.csv`",
        "",
        "## Status Counts",
        "",
    ]
    if not fold_summary.empty:
        lines.append(markdown_table_safe(fold_summary["status"].value_counts(dropna=False)))
    if not agg.empty:
        lines += ["", "## Top Candidates", ""]
        cols = [
            "source_experiment",
            "candidate",
            "folds",
            "selection_score",
            "robust_selection_score",
            "mean_sharpe",
            "q25_sharpe",
            "min_sharpe",
            "mean_max_drawdown",
            "mean_cash",
            "mean_turnover_l1",
        ]
        for source, group in agg.groupby("source_experiment", dropna=False):
            lines += [f"### {source}", "", markdown_table_safe(group.head(12)[cols], index=False), ""]
    if not deltas.empty:
        lines += ["", "## Baseline Delta / Promotion Guard", ""]
        delta_cols = [
            "source_experiment",
            "candidate",
            "baseline_candidate",
            "delta_selection_score",
            "conservative_delta_score",
            "promotion_decision",
            "folds_sharpe_better",
            "folds_drawdown_not_worse",
            "controller_drift_count",
        ]
        for source, group in deltas.groupby("source_experiment", dropna=False):
            lines += [f"### {source}", "", markdown_table_safe(group.head(12)[delta_cols], index=False), ""]
    lines += [
        "## Caveats",
        "",
        "- This is still no-retrain evaluation.",
        "- It is stronger than frozen-intent replay because actions are recomputed by the trained policy.",
        "- It is weaker than retraining because PPO never learns under the modified controller reward/trajectory.",
        "- Treat candidates with small positive deltas as weak signals unless they pass the promotion guard.",
    ]
    (output_dir / "COUNTERFACTUAL_POLICY_FORWARD_REPLAY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=[
            str(ARTIFACT_DIR / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1"),
            str(ARTIFACT_DIR / "R7_root_K20_stock_K5_PD_mild_slice_rescorr_groupquality_top10_sell12_rotation_internaldays_v1"),
            str(ARTIFACT_DIR / "R3_root_K20_PD_confidence_slice_residual_stock_v1"),
        ],
    )
    parser.add_argument(
        "--e9-rescorr-dir",
        default=str(ARTIFACT_DIR / "e9_rescorr_groups_compact"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPORT_DIR / "counterfactual_policy_forward_replay_v0"),
    )
    parser.add_argument("--max-zips", type=int, default=0)
    parser.add_argument("--candidates", nargs="*", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = default_candidates()
    if args.candidates:
        allowed = set(args.candidates)
        candidates = [candidate for candidate in candidates if candidate.name in allowed]
    (output_dir / "counterfactual_candidate_registry.csv").write_text(
        pd.DataFrame([asdict(candidate) for candidate in candidates]).to_csv(index=False),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    zips_seen = 0
    for input_dir in args.input_dirs:
        for zip_path in sorted(Path(input_dir).glob("*.zip")):
            if args.max_zips and zips_seen >= args.max_zips:
                break
            rows.extend(
                process_zip(
                    zip_path,
                    e9_rescorr_dir=Path(args.e9_rescorr_dir),
                    output_dir=output_dir,
                    candidates=candidates,
                )
            )
            zips_seen += 1
    fold_summary = pd.DataFrame(rows)
    fold_summary.to_csv(output_dir / "counterfactual_fold_summary.csv", index=False)
    agg = aggregate(fold_summary)
    agg.to_csv(output_dir / "counterfactual_aggregate_summary.csv", index=False)
    deltas = baseline_delta_summary(fold_summary)
    deltas.to_csv(output_dir / "counterfactual_baseline_delta_summary.csv", index=False)
    write_report(output_dir, fold_summary, agg, deltas)
    print(f"Wrote counterfactual policy-forward replay outputs to {output_dir}")


if __name__ == "__main__":
    main()
