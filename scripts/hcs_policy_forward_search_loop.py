"""Heuristic Controller Search loop using counterfactual policy-forward replay.

The earlier HCS foundation searched controller rules with frozen logged policy
intent. This loop is one level stronger: for each trial, it loads the trained
PPO model and evaluates a mutated controller while the policy recomputes its
actions on the counterfactual trajectory.

Outputs are intentionally simple and append-friendly:

* trials.jsonl      one aggregate trial record per source/candidate/generation
* summary.csv       ranked trial table
* fold_summary.csv  fold-level replay metrics
* failure_tags.csv  one row per detected failure tag

This is a cheap search/filter before cloud PPO retraining, not a replacement
for real training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from counterfactual_policy_forward_replay import (  # noqa: E402
    ARTIFACT_DIR,
    REPORT_DIR,
    CounterfactualCandidate,
    aggregate,
    baseline_delta_summary,
    process_zip,
)
from hcs_truth import (  # noqa: E402
    SOURCE_TO_TRUE_LABEL,
    read_ground_truth_registry,
    safe_float,
)
from hcs_candidates import (  # noqa: E402
    candidate_family,
    mutate_candidate,
    retrain_only_feature_candidates,
    seed_candidates,
    unique_candidates,
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value



def discover_zips(input_dirs: list[Path], max_zips: int) -> list[Path]:
    zips: list[Path] = []
    for input_dir in input_dirs:
        zips.extend(sorted(input_dir.glob("*.zip")))
    if max_zips > 0:
        zips = zips[:max_zips]
    return zips



def failure_tags_for_row(row: pd.Series, original_by_source: dict[str, pd.Series]) -> list[str]:
    tags: list[str] = []
    source = str(row.get("source_experiment", ""))
    candidate = str(row.get("candidate", ""))
    original = original_by_source.get(source)
    selection = safe_float(row.get("selection_score"))
    sharpe_std = safe_float(row.get("sample_std_sharpe"), 0.0)
    drawdown = safe_float(row.get("mean_max_drawdown"))
    cash = safe_float(row.get("mean_cash"))
    turnover = safe_float(row.get("mean_turnover_l1"))
    topk_flow = safe_float(row.get("mean_incremental_topk_flow_l1_mean"), 0.0)
    buy_allowed_rate = safe_float(row.get("mean_incremental_topk_buy_allowed_mean"), np.nan)
    recovery_rate = safe_float(row.get("mean_recovery_trigger_mean"), 0.0)
    risk_break_rate = safe_float(row.get("mean_risk_break_trigger_mean"), 0.0)
    effective_k = safe_float(row.get("mean_k_window_effective_days_mean"), np.nan)
    folds = safe_float(row.get("folds"), 0.0)

    if folds < 4:
        tags.append("low_fold_coverage")
    if original is not None and candidate != "pf_original":
        original_selection = safe_float(original.get("selection_score"))
        if math.isfinite(selection) and math.isfinite(original_selection):
            delta = selection - original_selection
            if delta < -0.05:
                tags.append("underperforms_original_large")
            elif delta < -0.02:
                tags.append("underperforms_original")
    if sharpe_std > 0.75:
        tags.append("unstable_across_folds")
    if drawdown < -0.12:
        tags.append("drawdown_too_deep")
    if cash > 0.60:
        tags.append("cash_lock_in_risk")
    if turnover > 0.012:
        tags.append("turnover_high")
    if "topk" in candidate and "no_incremental_topk" not in candidate and topk_flow < 0.003:
        tags.append("topk_not_active")
    if "riskaware" in candidate and math.isfinite(buy_allowed_rate) and buy_allowed_rate < 0.20:
        tags.append("buy_gate_too_strict")
    if recovery_rate > 0.30:
        tags.append("recovery_trigger_spam")
    if risk_break_rate > 0.25:
        tags.append("risk_break_trigger_spam")
    if math.isfinite(effective_k) and effective_k < 3.0:
        tags.append("window_collapsed_too_short")
    if not tags:
        tags.append("ok")
    return tags


def adjusted_score(row: pd.Series, original_by_source: dict[str, pd.Series], truth: pd.DataFrame) -> tuple[float, float, list[str]]:
    base = safe_float(row.get("selection_score"))
    if not math.isfinite(base):
        return float("nan"), 0.0, ["invalid_score"]

    tags = failure_tags_for_row(row, original_by_source)
    penalty = 0.0
    robust = safe_float(row.get("robust_selection_score"), np.nan)
    if math.isfinite(robust) and base - robust > 0.05:
        penalty += min(base - robust, 0.10)
        tags.append("lower_tail_fragile")
    for tag in tags:
        if tag == "underperforms_original_large":
            penalty += 0.08
        elif tag == "underperforms_original":
            penalty += 0.04
        elif tag == "unstable_across_folds":
            penalty += 0.03
        elif tag == "drawdown_too_deep":
            penalty += 0.08
        elif tag == "cash_lock_in_risk":
            penalty += 0.05
        elif tag == "turnover_high":
            penalty += 0.03
        elif tag == "low_fold_coverage":
            penalty += 0.02
        elif tag == "recovery_trigger_spam":
            penalty += 0.04
        elif tag == "risk_break_trigger_spam":
            penalty += 0.04
        elif tag == "window_collapsed_too_short":
            penalty += 0.05
        elif tag == "buy_gate_too_strict":
            penalty += 0.03

    source = str(row.get("source_experiment", ""))
    candidate = str(row.get("candidate", ""))
    family_is_topk = "topk" in candidate or "groupaware" in candidate or "ga_" in candidate

    # Historical calibration: R3 post-hoc Top-K was overestimated by replay
    # against the completed R4 incremental run. Keep this as a conservative
    # prior until more full PPO Top-K runs exist.
    if family_is_topk and source == "R3_root_K20_PD_confidence_slice_residual_stock_v1" and not truth.empty:
        match = truth[
            truth["true_label"].astype(str).isin(
                [
                    "R4_root_K20_PD_incremental_top5_flow_on_R3v1_v1_TRUE",
                    "R4_incremental_TRUE",
                ]
            )
        ]
        if not match.empty:
            true_selection = safe_float(pd.to_numeric(match["selection_score"], errors="coerce").max())
            if math.isfinite(true_selection) and base > true_selection:
                over = min(base - true_selection, 0.12)
                penalty += over
                tags.append("historical_topk_overprediction_penalty")

    return base - penalty, penalty, tags


def source_truth_replay_audit(summary: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Compare each source baseline replay with completed real-run metrics.

    Policy-forward replay should reproduce the source model's logged/real run
    before we trust its candidate ranking. A positive baseline error means the
    replay already overestimates the source, so all candidates from that source
    need an extra conservative penalty.
    """

    if summary.empty or truth.empty:
        return pd.DataFrame()

    truth_by_label = truth.sort_values("selection_score", ascending=False).drop_duplicates("true_label")
    truth_by_label = truth_by_label.set_index("true_label")
    rows: list[dict[str, Any]] = []
    baselines = summary[summary["candidate"].astype(str) == "pf_original"].copy()
    for _, row in baselines.iterrows():
        source = str(row.get("source_experiment", ""))
        true_label = SOURCE_TO_TRUE_LABEL.get(source)
        if not true_label or true_label not in truth_by_label.index:
            continue
        true = truth_by_label.loc[true_label]
        replay_selection = safe_float(row.get("selection_score"))
        true_selection = safe_float(true.get("selection_score"))
        selection_error = replay_selection - true_selection
        replay_sharpe = safe_float(row.get("mean_sharpe"))
        true_sharpe = safe_float(true.get("mean_sharpe"))
        replay_cash = safe_float(row.get("mean_cash"))
        true_cash = safe_float(true.get("mean_cash"))
        replay_turnover = safe_float(row.get("mean_turnover_l1"))
        true_turnover = safe_float(true.get("mean_turnover_l1"))
        rows.append(
            {
                "source_experiment": source,
                "true_label": true_label,
                "baseline_replay_folds": safe_float(row.get("folds"), 0.0),
                "replay_selection_score": replay_selection,
                "true_selection_score": true_selection,
                "selection_error": selection_error,
                "positive_selection_overprediction": max(selection_error, 0.0)
                if math.isfinite(selection_error)
                else np.nan,
                "replay_mean_sharpe": replay_sharpe,
                "true_mean_sharpe": true_sharpe,
                "sharpe_error": replay_sharpe - true_sharpe,
                "replay_mean_cash": replay_cash,
                "true_mean_cash": true_cash,
                "cash_error": replay_cash - true_cash,
                "replay_mean_turnover_l1": replay_turnover,
                "true_mean_turnover_l1": true_turnover,
                "turnover_error": replay_turnover - true_turnover,
            }
        )
    return pd.DataFrame(rows)


def apply_final_guards(summary: pd.DataFrame, fold_summary: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach conservative promotion guards to the final HCS table."""

    if summary.empty:
        return summary, pd.DataFrame(), pd.DataFrame()

    guarded = summary.copy()
    baseline_guards = baseline_delta_summary(fold_summary)
    if not baseline_guards.empty:
        guard_cols = [
            "source_experiment",
            "candidate",
            "baseline_candidate",
            "delta_selection_score",
            "conservative_delta_score",
            "promotion_decision",
            "folds_sharpe_better",
            "folds_drawdown_not_worse",
            "mean_delta_sharpe",
            "worst_delta_sharpe",
            "mean_delta_max_drawdown",
            "worst_delta_max_drawdown",
            "controller_drift_count",
            "no_retrain_penalty",
            "drawdown_penalty",
            "min_promote_edge",
        ]
        guarded = guarded.merge(
            baseline_guards[[col for col in guard_cols if col in baseline_guards.columns]],
            on=["source_experiment", "candidate"],
            how="left",
        )
    else:
        guarded["promotion_decision"] = np.nan
        guarded["conservative_delta_score"] = np.nan
        guarded["folds_sharpe_better"] = np.nan
        guarded["folds_drawdown_not_worse"] = np.nan

    truth_audit = source_truth_replay_audit(guarded, truth)
    if not truth_audit.empty:
        guarded = guarded.merge(
            truth_audit[
                [
                    "source_experiment",
                    "true_label",
                    "baseline_replay_folds",
                    "selection_error",
                    "positive_selection_overprediction",
                    "cash_error",
                    "turnover_error",
                ]
            ].rename(
                columns={
                    "selection_error": "source_truth_selection_error",
                    "positive_selection_overprediction": "source_truth_overprediction_penalty",
                    "cash_error": "source_truth_cash_error",
                    "turnover_error": "source_truth_turnover_error",
                }
            ),
            on="source_experiment",
            how="left",
        )
    else:
        guarded["source_truth_overprediction_penalty"] = 0.0

    final_penalties: list[float] = []
    final_tags: list[str] = []
    promotable: list[bool] = []
    for _, row in guarded.iterrows():
        candidate = str(row.get("candidate", ""))
        penalty = 0.0
        tags = [tag for tag in str(row.get("failure_tags", "")).split(";") if tag]
        source_penalty = safe_float(row.get("source_truth_overprediction_penalty"), 0.0)
        baseline_replay_folds = safe_float(row.get("baseline_replay_folds"), 0.0)
        if baseline_replay_folds and baseline_replay_folds < 4:
            tags.append("source_truth_audit_partial_folds")
            source_penalty = 0.0
        if candidate != "pf_original" and math.isfinite(source_penalty) and source_penalty > 0.0:
            penalty += min(source_penalty, 0.15)
            tags.append("source_replay_overpredicts_real_baseline")

        promotion_raw = row.get("promotion_decision", "")
        promotion = str(promotion_raw) if pd.notna(promotion_raw) else ""
        conservative_delta = safe_float(row.get("conservative_delta_score"), np.nan)
        if candidate != "pf_original":
            if promotion == "do_not_promote":
                penalty += max(0.02, min(abs(min(conservative_delta, 0.0)) if math.isfinite(conservative_delta) else 0.04, 0.15))
                tags.append("baseline_guard_failed")
            folds_better = safe_float(row.get("folds_sharpe_better"), np.nan)
            if math.isfinite(folds_better) and folds_better < 3:
                penalty += 0.03
                tags.append("weak_fold_support")

        final_penalties.append(float(penalty))
        final_tags.append(";".join(dict.fromkeys(tags)) if tags else "ok")
        promotable.append(
            candidate == "pf_original"
            or (
                promotion == "promote"
                and safe_float(row.get("source_truth_overprediction_penalty"), 0.0) <= 0.05
                and "weak_fold_support" not in tags
            )
        )

    guarded["hcs_guard_penalty"] = final_penalties
    guarded["hcs_final_score"] = pd.to_numeric(guarded["hcs_adjusted_score"], errors="coerce") - guarded["hcs_guard_penalty"]
    guarded["hcs_final_tags"] = final_tags
    guarded["hcs_promotable"] = promotable
    guarded = guarded.sort_values(["source_experiment", "hcs_final_score"], ascending=[True, False])
    return guarded, baseline_guards, truth_audit


def build_selection_regret(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for source, group in summary.groupby("source_experiment", dropna=False):
        raw_best = group.sort_values("selection_score", ascending=False).iloc[0]
        adjusted_best = group.sort_values("hcs_adjusted_score", ascending=False).iloc[0]
        final_best = group.sort_values("hcs_final_score", ascending=False).iloc[0]
        baseline = group[group["candidate"].astype(str) == "pf_original"]
        baseline_score = safe_float(baseline.iloc[0].get("hcs_final_score")) if not baseline.empty else np.nan
        raw_best_final = safe_float(raw_best.get("hcs_final_score"))
        final_best_score = safe_float(final_best.get("hcs_final_score"))
        rows.append(
            {
                "source_experiment": source,
                "raw_selection_winner": raw_best.get("candidate"),
                "adjusted_winner": adjusted_best.get("candidate"),
                "final_guarded_winner": final_best.get("candidate"),
                "raw_winner_selection_score": safe_float(raw_best.get("selection_score")),
                "raw_winner_final_score": raw_best_final,
                "final_winner_final_score": final_best_score,
                "regret_if_using_raw_selection": final_best_score - raw_best_final,
                "final_winner_edge_vs_pf_original": final_best_score - baseline_score,
                "final_winner_promotable": bool(final_best.get("hcs_promotable", False)),
            }
        )
    return pd.DataFrame(rows)


def build_family_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (source, family), group in summary.groupby(["source_experiment", "family"], dropna=False):
        ordered = group.sort_values("hcs_final_score", ascending=False)
        best = ordered.iloc[0]
        rows.append(
            {
                "source_experiment": source,
                "family": family,
                "trials": int(len(group)),
                "promotable_trials": int(pd.Series(group.get("hcs_promotable", False)).astype(bool).sum()),
                "best_candidate": best.get("candidate"),
                "best_hcs_final_score": safe_float(best.get("hcs_final_score")),
                "best_hcs_adjusted_score": safe_float(best.get("hcs_adjusted_score")),
                "median_hcs_final_score": safe_float(pd.to_numeric(group["hcs_final_score"], errors="coerce").median()),
                "best_selection_score": safe_float(best.get("selection_score")),
                "best_failure_tags": best.get("hcs_final_tags", best.get("failure_tags", "")),
            }
        )
    return pd.DataFrame(rows).sort_values(["source_experiment", "best_hcs_final_score"], ascending=[True, False])


def build_negative_control_audit(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for source, group in summary.groupby("source_experiment", dropna=False):
        baseline = group[group["candidate"].astype(str) == "pf_original"]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        for _, row in group[group["family"].isin(["no_incremental_topk"])].iterrows():
            delta_final = safe_float(row.get("hcs_final_score")) - safe_float(base.get("hcs_final_score"))
            delta_selection = safe_float(row.get("selection_score")) - safe_float(base.get("selection_score"))
            rows.append(
                {
                    "source_experiment": source,
                    "negative_control": row.get("candidate"),
                    "delta_selection_vs_original": delta_selection,
                    "delta_final_score_vs_original": delta_final,
                    "warning": "negative_control_beats_original" if delta_final > 0.0 else "",
                }
            )
    return pd.DataFrame(rows)


def evaluate_generation(
    *,
    generation: int,
    candidates: list[CounterfactualCandidate],
    zips: list[Path],
    e9_rescorr_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    gen_dir = output_dir / "daily" / f"generation_{generation:02d}"
    for zip_path in zips:
        rows.extend(
            process_zip(
                zip_path,
                e9_rescorr_dir=e9_rescorr_dir,
                output_dir=gen_dir,
                candidates=candidates,
            )
        )
    fold = pd.DataFrame(rows)
    if not fold.empty:
        fold.insert(0, "generation", generation)
    agg = aggregate(fold)
    if not agg.empty:
        agg.insert(0, "generation", generation)
    return fold, agg


def append_trials(path: Path, trial_rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in trial_rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    failure_tags: pd.DataFrame,
    retrain_only: pd.DataFrame,
    baseline_guards: pd.DataFrame,
    truth_audit: pd.DataFrame,
    selection_regret: pd.DataFrame,
    family_summary: pd.DataFrame,
    negative_control_audit: pd.DataFrame,
) -> None:
    lines = [
        "# HCS Policy-Forward Search Loop",
        "",
        "This run mutates controller rules and evaluates them with counterfactual policy-forward replay.",
        "The trained PPO policy is fixed, but policy actions are recomputed on counterfactual trajectories.",
        "",
        "## Outputs",
        "",
        "- `trials.jsonl`",
        "- `summary.csv`",
        "- `fold_summary.csv`",
        "- `failure_tags.csv`",
        "- `candidate_registry.csv`",
        "- `retrain_only_candidates.csv`",
        "- `baseline_guard.csv`",
        "- `source_truth_replay_audit.csv`",
        "- `selection_regret.csv`",
        "- `family_summary.csv`",
        "- `negative_control_audit.csv`",
        "",
        "## Retrain-Only Families",
        "",
    ]
    if retrain_only.empty:
        lines.append("No retrain-only candidates were registered.")
    else:
        lines.append(retrain_only.to_markdown(index=False))

    lines += [
        "",
        "## Top Trials",
        "",
    ]
    if summary.empty:
        lines.append("No trials were evaluated.")
    else:
        cols = [
            "source_experiment",
            "candidate",
            "generation",
            "family",
            "selection_score",
            "robust_selection_score",
            "hcs_adjusted_score",
            "hcs_final_score",
            "hcs_promotable",
            "hcs_penalty",
            "hcs_guard_penalty",
            "mean_sharpe",
            "q25_sharpe",
            "mean_max_drawdown",
            "mean_cash",
            "mean_turnover_l1",
            "hcs_final_tags",
        ]
        for source, group in summary.groupby("source_experiment", dropna=False):
            lines.extend([f"### {source}", ""])
            present_cols = [col for col in cols if col in group.columns]
            lines.append(group.sort_values("hcs_final_score", ascending=False).head(12)[present_cols].to_markdown(index=False))
            lines.append("")

    if not baseline_guards.empty:
        lines.extend(["## Baseline Guard", ""])
        cols = [
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
        present_cols = [col for col in cols if col in baseline_guards.columns]
        lines.append(baseline_guards.sort_values("conservative_delta_score", ascending=False).head(20)[present_cols].to_markdown(index=False))
        lines.append("")

    if not truth_audit.empty:
        lines.extend(["## Replay Vs Real Baseline Audit", ""])
        cols = [
            "source_experiment",
            "true_label",
            "baseline_replay_folds",
            "selection_error",
            "positive_selection_overprediction",
            "cash_error",
            "turnover_error",
        ]
        lines.append(truth_audit[cols].to_markdown(index=False))
        lines.append("")

    if not selection_regret.empty:
        lines.extend(["## Selection Regret", ""])
        lines.append(selection_regret.to_markdown(index=False))
        lines.append("")

    if not family_summary.empty:
        lines.extend(["## Family Summary", ""])
        cols = [
            "source_experiment",
            "family",
            "trials",
            "promotable_trials",
            "best_candidate",
            "best_hcs_final_score",
            "median_hcs_final_score",
        ]
        lines.append(family_summary[cols].to_markdown(index=False))
        lines.append("")

    if not negative_control_audit.empty:
        lines.extend(["## Negative Control Audit", ""])
        lines.append(negative_control_audit.to_markdown(index=False))
        lines.append("")

    if not failure_tags.empty:
        lines.extend(["## Failure Tag Counts", ""])
        lines.append(failure_tags["tag"].value_counts().to_markdown())
        lines.append("")

    lines.extend(
        [
            "## Promotion Rule",
            "",
            "Use this order before spending cloud budget:",
            "",
            "```text",
            "frozen replay -> policy-forward replay -> full PPO retrain",
            "```",
            "",
            "A candidate is a real next experiment only if it improves adjusted score",
            "and passes the baseline guard without relying on replay-overprediction.",
            "",
            "The final ranking key is `hcs_final_score`, not raw `selection_score`.",
            "Raw winners that lose after guards are treated as replay artifacts.",
            "",
            "Feature-family candidates are not replay-safe. They are included only",
            "as retrain-planning rows because compact/delta feature sets change the",
            "observation contract seen by PPO.",
        ]
    )
    (output_dir / "HCS_POLICY_FORWARD_SEARCH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        default=str(REPORT_DIR / "hcs_policy_forward_search_loop_v1"),
    )
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--mutations-per-parent", type=int, default=5)
    parser.add_argument("--max-zips", type=int, default=0)
    parser.add_argument("--max-candidates-per-generation", type=int, default=12)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "trials.jsonl"
    if trials_path.exists():
        trials_path.unlink()

    zips = discover_zips([Path(p) for p in args.input_dirs], args.max_zips)
    if not zips:
        raise FileNotFoundError("No experiment zips found for HCS policy-forward search.")

    truth = read_ground_truth_registry()
    evaluated: set[str] = set()
    all_fold: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    registry_rows: list[dict[str, Any]] = []
    retrain_only = retrain_only_feature_candidates()
    current = seed_candidates()

    for generation in range(max(args.generations, 1)):
        current = unique_candidates([c for c in current if c.name not in evaluated])
        current = current[: max(args.max_candidates_per_generation, 1)]
        if not current:
            break
        for candidate in current:
            evaluated.add(candidate.name)
            registry_rows.append(
                {
                    "generation": generation,
                    "family": candidate_family(candidate),
                    **asdict(candidate),
                }
            )

        fold, agg = evaluate_generation(
            generation=generation,
            candidates=current,
            zips=zips,
            e9_rescorr_dir=Path(args.e9_rescorr_dir),
            output_dir=output_dir,
        )
        all_fold.append(fold)
        if agg.empty:
            current = []
            continue

        original_by_source = {
            str(row["source_experiment"]): row
            for _, row in agg[agg["candidate"].astype(str) == "pf_original"].iterrows()
        }
        # If original was evaluated in an earlier generation, include it as the
        # baseline for later-generation failure tags.
        if all_summary:
            prev_summary = pd.concat(all_summary, ignore_index=True)
            prev_original = prev_summary[prev_summary["candidate"].astype(str) == "pf_original"]
            for _, row in prev_original.iterrows():
                original_by_source.setdefault(str(row["source_experiment"]), row)

        trial_rows: list[dict[str, Any]] = []
        enriched_rows: list[dict[str, Any]] = []
        for _, row in agg.iterrows():
            candidate_name_value = str(row["candidate"])
            candidate_obj = next((c for c in current if c.name == candidate_name_value), None)
            adjusted, penalty, tags = adjusted_score(row, original_by_source, truth)
            enriched = {
                **row.to_dict(),
                "family": candidate_family(candidate_obj) if candidate_obj else "unknown",
                "hcs_adjusted_score": adjusted,
                "hcs_penalty": penalty,
                "failure_tags": ";".join(tags),
            }
            enriched_rows.append(enriched)
            trial_rows.append(
                {
                    "generation": generation,
                    "trial_id": f"g{generation:02d}:{row['source_experiment']}:{candidate_name_value}",
                    "candidate": candidate_name_value,
                    "source_experiment": row["source_experiment"],
                    "params": asdict(candidate_obj) if candidate_obj else {},
                    "metrics": row.to_dict(),
                    "hcs_adjusted_score": adjusted,
                    "hcs_penalty": penalty,
                    "failure_tags": tags,
                }
            )
        enriched = pd.DataFrame(enriched_rows)
        if not enriched.empty:
            enriched, _, _ = apply_final_guards(enriched, fold, truth)
            final_lookup = {
                (str(row["source_experiment"]), str(row["candidate"])): row
                for _, row in enriched.iterrows()
            }
            for trial in trial_rows:
                key = (str(trial["source_experiment"]), str(trial["candidate"]))
                final_row = final_lookup.get(key)
                if final_row is None:
                    continue
                trial["hcs_final_score"] = final_row.get("hcs_final_score")
                trial["hcs_guard_penalty"] = final_row.get("hcs_guard_penalty")
                trial["hcs_final_tags"] = str(final_row.get("hcs_final_tags", ""))
                trial["hcs_promotable"] = bool(final_row.get("hcs_promotable", False))
        append_trials(trials_path, trial_rows)
        all_summary.append(enriched)

        parent_scores = enriched[~enriched["candidate"].isin(["pf_original", "pf_no_incremental_topk"])].copy()
        if parent_scores.empty:
            current = []
            continue
        parent_scores = (
            parent_scores.groupby("candidate", as_index=False)["hcs_final_score"]
            .mean()
            .sort_values("hcs_final_score", ascending=False)
            .head(max(args.beam_size, 1))
        )
        current_by_name = {candidate.name: candidate for candidate in current}
        next_candidates: list[CounterfactualCandidate] = []
        for _, parent_row in parent_scores.iterrows():
            parent = current_by_name.get(str(parent_row["candidate"]))
            if parent is None:
                continue
            next_candidates.extend(mutate_candidate(parent, max_mutations=max(args.mutations_per_parent, 1)))
        current = unique_candidates(next_candidates)

    fold_summary = pd.concat(all_fold, ignore_index=True) if all_fold else pd.DataFrame()
    summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    registry = pd.DataFrame(registry_rows)

    baseline_guards = pd.DataFrame()
    truth_audit = pd.DataFrame()
    selection_regret = pd.DataFrame()
    family_summary = pd.DataFrame()
    negative_control_audit = pd.DataFrame()
    if not summary.empty:
        baseline_guards = baseline_delta_summary(fold_summary)
        truth_audit = source_truth_replay_audit(summary, truth)
        if "hcs_final_score" not in summary.columns:
            summary, baseline_guards, truth_audit = apply_final_guards(summary, fold_summary, truth)
        else:
            summary = summary.sort_values(["source_experiment", "hcs_final_score"], ascending=[True, False])
        selection_regret = build_selection_regret(summary)
        family_summary = build_family_summary(summary)
        negative_control_audit = build_negative_control_audit(summary)
    failure_rows: list[dict[str, Any]] = []
    if not summary.empty:
        for _, row in summary.iterrows():
            for tag in str(row.get("hcs_final_tags", row.get("failure_tags", ""))).split(";"):
                if tag:
                    failure_rows.append(
                        {
                            "source_experiment": row.get("source_experiment"),
                            "candidate": row.get("candidate"),
                            "generation": row.get("generation"),
                            "tag": tag,
                        }
                    )
    failure_tags = pd.DataFrame(failure_rows)

    fold_summary.to_csv(output_dir / "fold_summary.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    failure_tags.to_csv(output_dir / "failure_tags.csv", index=False)
    registry.to_csv(output_dir / "candidate_registry.csv", index=False)
    retrain_only.to_csv(output_dir / "retrain_only_candidates.csv", index=False)
    baseline_guards.to_csv(output_dir / "baseline_guard.csv", index=False)
    truth_audit.to_csv(output_dir / "source_truth_replay_audit.csv", index=False)
    selection_regret.to_csv(output_dir / "selection_regret.csv", index=False)
    family_summary.to_csv(output_dir / "family_summary.csv", index=False)
    negative_control_audit.to_csv(output_dir / "negative_control_audit.csv", index=False)
    if not truth.empty:
        truth.to_csv(output_dir / "ground_truth_registry.csv", index=False)
    write_report(
        output_dir,
        summary,
        failure_tags,
        retrain_only,
        baseline_guards,
        truth_audit,
        selection_regret,
        family_summary,
        negative_control_audit,
    )

    print(f"Wrote HCS policy-forward search loop outputs to {output_dir}")


if __name__ == "__main__":
    main()
