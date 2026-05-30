"""Heuristic Controller Search foundation for Stage 0.1.

This script turns the current controller replay into a small search system:

1. Build a ground-truth registry from already completed real experiments.
2. Generate controller candidates from rule families and tunable parameters.
3. Replay candidates cheaply over frozen logged policy intent.
4. Compare replay results with known real-run metrics where possible.
5. Emit ranked recommendations and caveats before spending cloud budget.

It is diagnostic infrastructure. It does not retrain PPO and does not update
policy-network weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from offline_replay_groupaware_topk import (  # noqa: E402
    ReplayVariant,
    aggregate,
    fold_from_text,
    groups_from_e9,
    infer_tickers,
    logged_summary,
    read_zip_csv,
    read_zip_json,
    replay_one,
    returns_from_model_ready,
)

from hcs_truth import (  # noqa: E402
    ARTIFACT_DIR,
    REPORT_DIR,
    SOURCE_TO_TRUE_LABEL,
    read_ground_truth_registry,
    safe_float,
    selection_score,
)


def hcs_candidate_variants() -> list[ReplayVariant]:
    variants: list[ReplayVariant] = [
        ReplayVariant("logged_original"),
        ReplayVariant("anchor_no_controller", source="anchor", controller="none"),
        ReplayVariant("scheduled_no_controller", source="scheduled", controller="none"),
        ReplayVariant("anchor_PD", source="anchor", controller="PD"),
        ReplayVariant("scheduled_PD", source="scheduled", controller="PD"),
    ]

    for eps in [0.005, 0.01, 0.02]:
        variants.append(ReplayVariant(f"scheduled_PD_deadzone_{eps:g}", source="scheduled", controller="PD", deadzone_eps=eps))

    topk_pairs = [(3, 3), (5, 5), (8, 8), (3, 8), (5, 8), (8, 5), (8, 3)]
    for buy_k, sell_k in topk_pairs:
        for rotation in [0.0, 0.001, 0.0025]:
            suffix = f"b{buy_k}_s{sell_k}_rot{str(rotation).replace('.', '')}"
            variants.append(
                ReplayVariant(
                    f"scheduled_PD_topk_{suffix}",
                    source="scheduled",
                    controller="PD",
                    top_k_buy=buy_k,
                    top_k_sell=sell_k,
                    rotation_budget=rotation,
                )
            )

    group_specs = [
        ("pressure", 1.00, 1.00, 0.00, 0.50),
        ("softcap", 0.60, 1.00, 0.25, 0.75),
        ("cap45", 0.45, 1.00, 0.50, 1.00),
        ("pressure2", 1.00, 2.00, 0.00, 0.75),
    ]
    for buy_k, sell_k in [(5, 5), (8, 8), (5, 8), (8, 5)]:
        for rotation in [0.0, 0.001]:
            for name, cap, pressure, capacity, overweight in group_specs:
                suffix = f"{name}_b{buy_k}_s{sell_k}_rot{str(rotation).replace('.', '')}"
                variants.append(
                    ReplayVariant(
                        f"scheduled_PD_groupaware_{suffix}",
                        source="scheduled",
                        controller="PD",
                        top_k_buy=buy_k,
                        top_k_sell=sell_k,
                        rotation_budget=rotation,
                        group_aware=True,
                        default_group_cap=cap,
                        pressure_weight=pressure,
                        capacity_weight=capacity,
                        sell_overweight_weight=overweight,
                    )
                )

    return variants


def candidate_registry(variants: list[ReplayVariant]) -> pd.DataFrame:
    rows = []
    for variant in variants:
        data = asdict(variant)
        data["family"] = classify_variant(variant.name)
        rows.append(data)
    return pd.DataFrame(rows)


def classify_variant(name: str) -> str:
    if name == "logged_original":
        return "ground_truth_log"
    if "no_controller" in name:
        return "aggressive_no_controller"
    if "groupaware" in name:
        return "group_aware_topk"
    if "topk" in name:
        return "global_topk"
    if "deadzone" in name:
        return "deadzone"
    return "baseline_controller"


def replay_zip(
    zip_path: Path,
    *,
    e9_rescorr_dir: Path,
    variants: list[ReplayVariant],
    write_daily: bool,
    output_dir: Path,
) -> list[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as zf:
        daily = read_zip_csv(zf, "validation_daily.csv")
        model_ready = read_zip_csv(zf, "model_ready.csv")
        metadata = read_zip_json(zf, "metadata.json")
    tickers = infer_tickers(daily)
    fold = fold_from_text(str(zip_path)) or fold_from_text(json.dumps(metadata))
    source = zip_path.parent.name
    returns_next = returns_from_model_ready(model_ready, tickers, daily["date"])
    groups = groups_from_e9(e9_rescorr_dir, fold, tickers)

    rows: list[dict[str, Any]] = []
    logged = logged_summary(daily)
    logged.update({"source_experiment": source, "fold": fold, "zip": zip_path.name, "group_count": len(groups)})
    rows.append(logged)

    for variant in variants:
        if variant.name == "logged_original":
            continue
        summary, replay_daily = replay_one(daily, returns_next, tickers, variant, groups=groups)
        summary.update({"source_experiment": source, "fold": fold, "zip": zip_path.name, "group_count": len(groups)})
        rows.append(summary)
        if write_daily:
            out_dir = output_dir / "daily" / source
            out_dir.mkdir(parents=True, exist_ok=True)
            replay_daily.insert(0, "replay_variant", variant.name)
            replay_daily.insert(0, "fold", fold)
            replay_daily.insert(0, "source_experiment", source)
            replay_daily.to_csv(out_dir / f"{fold}_{variant.name}.csv", index=False)
    return rows


def run_replay(
    *,
    input_dirs: list[Path],
    e9_rescorr_dir: Path,
    variants: list[ReplayVariant],
    output_dir: Path,
    write_daily: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        for zip_path in sorted(input_dir.glob("*.zip")):
            rows.extend(
                replay_zip(
                    zip_path,
                    e9_rescorr_dir=e9_rescorr_dir,
                    variants=variants,
                    write_daily=write_daily,
                    output_dir=output_dir,
                )
            )
    fold_summary = pd.DataFrame(rows)
    agg = aggregate(fold_summary)
    return fold_summary, agg


def calibrate_against_ground_truth(replay_agg: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    if replay_agg.empty or ground_truth.empty:
        return pd.DataFrame()
    truth_by_label = ground_truth.sort_values("selection_score", ascending=False).drop_duplicates("true_label")
    truth_by_label = truth_by_label.set_index("true_label")
    rows: list[dict[str, Any]] = []

    for _, row in replay_agg.iterrows():
        source = str(row["source_experiment"])
        variant = str(row["variant"])
        mapped_label = None
        match_type = ""
        if variant == "logged_original" and source in SOURCE_TO_TRUE_LABEL:
            mapped_label = SOURCE_TO_TRUE_LABEL[source]
            match_type = "exact_logged_run"
        elif source == "R3_root_K20_PD_confidence_slice_residual_stock_v1" and variant in {
            "scheduled_PD_topk_b5_s5_rot00",
            "scheduled_PD_top5_rot0",
        }:
            mapped_label = "R4_incremental_TRUE"
            match_type = "approx_controller_family"

        if not mapped_label or mapped_label not in truth_by_label.index:
            continue
        truth = truth_by_label.loc[mapped_label]
        rows.append(
            {
                "source_experiment": source,
                "replay_variant": variant,
                "true_label": mapped_label,
                "match_type": match_type,
                "replay_selection_score": row["selection_score"],
                "true_selection_score": truth["selection_score"],
                "selection_error": row["selection_score"] - truth["selection_score"],
                "replay_mean_sharpe": row["mean_sharpe"],
                "true_mean_sharpe": truth["mean_sharpe"],
                "sharpe_error": row["mean_sharpe"] - truth["mean_sharpe"],
                "replay_mean_cash": row["mean_cash"],
                "true_mean_cash": truth["mean_cash"],
                "cash_error": row["mean_cash"] - truth["mean_cash"],
                "replay_mean_turnover_l1": row["mean_turnover_l1"],
                "true_mean_turnover_l1": truth["mean_turnover_l1"],
                "turnover_error": row["mean_turnover_l1"] - truth["mean_turnover_l1"],
            }
        )
    return pd.DataFrame(rows)


def rank_recommendations(replay_agg: pd.DataFrame, calibration: pd.DataFrame | None = None) -> pd.DataFrame:
    if replay_agg.empty:
        return replay_agg
    calibration_penalty_by_source: dict[str, float] = {}
    if calibration is not None and not calibration.empty:
        approx = calibration[calibration["match_type"].astype(str) == "approx_controller_family"].copy()
        if not approx.empty:
            approx["positive_overprediction"] = pd.to_numeric(approx["selection_error"], errors="coerce").clip(lower=0.0)
            calibration_penalty_by_source = (
                approx.groupby("source_experiment")["positive_overprediction"].mean().dropna().to_dict()
            )

    rows = []
    for _, row in replay_agg.iterrows():
        source = str(row["source_experiment"])
        variant = str(row["variant"])
        family = classify_variant(variant)
        score = safe_float(row.get("selection_score"))
        drawdown = safe_float(row.get("mean_max_drawdown"))
        turnover = safe_float(row.get("mean_turnover_l1"))
        cash = safe_float(row.get("mean_cash"))
        target_gap = safe_float(row.get("mean_target_to_executed_l1"))

        risk_penalty = 0.0
        reasons = []
        if "no_controller" in variant:
            risk_penalty += 0.12
            reasons.append("no_controller_counterfactual_too_aggressive")
        if drawdown < -0.12:
            risk_penalty += 0.08
            reasons.append("drawdown_worse_than_controller_target")
        if turnover > 0.02:
            risk_penalty += 0.05
            reasons.append("high_turnover")
        if cash > 0.70:
            risk_penalty += 0.08
            reasons.append("cash_lock_in_risk")
        if target_gap > 0.04:
            risk_penalty += 0.04
            reasons.append("large_target_to_exec_gap")
        if family == "group_aware_topk":
            risk_penalty += 0.01
            reasons.append("needs_retrain_confirmation")
        calibration_penalty = 0.0
        calibration_note = ""
        if family in {"global_topk", "group_aware_topk"}:
            calibration_penalty = float(calibration_penalty_by_source.get(source, 0.0))
            if calibration_penalty > 0.0:
                reasons.append("ground_truth_replay_overprediction_penalty")
                calibration_note = (
                    "penalized by approximate real-run calibration for this source/controller family"
                )

        adjusted = score - risk_penalty - calibration_penalty
        rows.append(
            {
                **row.to_dict(),
                "family": family,
                "hcs_adjusted_score": adjusted,
                "hcs_risk_penalty": risk_penalty,
                "hcs_calibration_penalty": calibration_penalty,
                "hcs_calibration_note": calibration_note,
                "hcs_caveats": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows).sort_values(["source_experiment", "hcs_adjusted_score"], ascending=[True, False])


def write_report(
    *,
    output_dir: Path,
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    replay_agg: pd.DataFrame,
    calibration: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    lines = [
        "# Heuristic Controller Search Foundation",
        "",
        "This is the first foundation pass for a replay-driven Heuristic Controller Search loop.",
        "It uses completed real experiments as ground truth and cheap replay as a hypothesis filter before cloud retraining.",
        "",
        "## Artifacts",
        "",
        "- `hcs_candidate_registry.csv`",
        "- `hcs_ground_truth_registry.csv`",
        "- `hcs_replay_fold_summary.csv`",
        "- `hcs_replay_aggregate_summary.csv`",
        "- `hcs_replay_vs_ground_truth_calibration.csv`",
        "- `hcs_recommendations.csv`",
        "",
        "## Candidate Families",
        "",
    ]
    family_counts = candidates["family"].value_counts().sort_index()
    for family, count in family_counts.items():
        lines.append(f"- `{family}`: {int(count)} candidates")

    lines += [
        "",
        "## Ground Truth Registry",
        "",
        f"Loaded `{len(ground_truth)}` completed-run aggregate rows.",
        "",
    ]
    if not ground_truth.empty:
        cols = ["true_label", "selection_score", "mean_sharpe", "mean_cash", "mean_turnover_l1", "source_file"]
        lines.append(ground_truth.head(12)[cols].to_markdown(index=False))

    lines += ["", "## Replay Calibration", ""]
    if calibration.empty:
        lines.append("No direct replay-vs-ground-truth calibration rows were available.")
    else:
        cols = [
            "source_experiment",
            "replay_variant",
            "true_label",
            "match_type",
            "selection_error",
            "sharpe_error",
            "cash_error",
            "turnover_error",
        ]
        lines.append(calibration[cols].to_markdown(index=False))

    lines += ["", "## Top HCS Recommendations", ""]
    if recommendations.empty:
        lines.append("No replay recommendations were produced.")
    else:
        cols = [
            "source_experiment",
            "variant",
            "family",
            "selection_score",
            "hcs_adjusted_score",
            "mean_sharpe",
            "mean_max_drawdown",
            "mean_cash",
            "mean_turnover_l1",
            "hcs_calibration_penalty",
            "hcs_caveats",
        ]
        for source, group in recommendations.groupby("source_experiment", dropna=False):
            lines.append(f"### {source}")
            lines.append("")
            lines.append(group.head(12)[cols].to_markdown(index=False))
            lines.append("")

    lines += [
        "## Interpretation Rules",
        "",
        "- Treat replay as a filter, not proof. It freezes policy intent and does not model PPO retraining.",
        "- Trust replay more for execution-only changes with small target-state feedback.",
        "- Trust replay less for changes that would strongly alter portfolio state, cash duration, drawdown, or future policy inputs.",
        "- Candidates that beat real logged runs in replay still need retraining before being promoted.",
        "",
        "## Next Extensions",
        "",
        "1. Add counterfactual policy-forward replay: recompute anchors from a trained model on replayed portfolio state.",
        "2. Add train/validation split inside replay search to avoid tuning to validation folds.",
        "3. Add rule-mutator families for confidence coefficients, trigger thresholds, K-days, compact/delta feature masks, and group construction.",
        "4. Add golden replay tests: original controller replay must reproduce logged daily metrics within tolerance.",
    ]

    (output_dir / "HCS_FOUNDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=[
            str(ARTIFACT_DIR / "R3_root_K20_PD_confidence_slice_residual_stock_v1"),
            str(ARTIFACT_DIR / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1"),
            str(ARTIFACT_DIR / "R7_root_K20_stock_K5_PD_mild_slice_rescorr_groupquality_top10_sell12_rotation_internaldays_v1"),
        ],
    )
    parser.add_argument(
        "--e9-rescorr-dir",
        default=str(ARTIFACT_DIR / "e9_rescorr_groups_compact"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPORT_DIR / "heuristic_controller_search_v0"),
    )
    parser.add_argument("--write-daily", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = hcs_candidate_variants()
    candidates = candidate_registry(variants)
    candidates.to_csv(output_dir / "hcs_candidate_registry.csv", index=False)

    ground_truth = read_ground_truth_registry()
    ground_truth.to_csv(output_dir / "hcs_ground_truth_registry.csv", index=False)

    fold_summary, replay_agg = run_replay(
        input_dirs=[Path(p) for p in args.input_dirs],
        e9_rescorr_dir=Path(args.e9_rescorr_dir),
        variants=variants,
        output_dir=output_dir,
        write_daily=bool(args.write_daily),
    )
    fold_summary.to_csv(output_dir / "hcs_replay_fold_summary.csv", index=False)
    replay_agg.to_csv(output_dir / "hcs_replay_aggregate_summary.csv", index=False)

    calibration = calibrate_against_ground_truth(replay_agg, ground_truth)
    calibration.to_csv(output_dir / "hcs_replay_vs_ground_truth_calibration.csv", index=False)

    recommendations = rank_recommendations(replay_agg, calibration)
    recommendations.to_csv(output_dir / "hcs_recommendations.csv", index=False)

    write_report(
        output_dir=output_dir,
        candidates=candidates,
        ground_truth=ground_truth,
        replay_agg=replay_agg,
        calibration=calibration,
        recommendations=recommendations,
    )

    print(f"Wrote HCS foundation outputs to {output_dir}")


if __name__ == "__main__":
    main()
