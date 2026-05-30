"""Ground-truth registry and shared scoring helpers for Stage 0.1 HCS."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "r_k_window_analysis"
ARTIFACT_DIR = ROOT / "artifacts" / "stage0_1"


SOURCE_TO_TRUE_LABEL = {
    "R3_root_K20_PD_confidence_slice_residual_stock_v1": "R3_root_K20_PD_confidence_slice_residual_stock_v1",
    "R6_root_K20_stock_K5_PD_mild_slice_top5_rotation_internaldays_v1": (
        "R6_root_K20_stock_K5_PD_mild_slice_top5_rotation_internaldays_v1"
    ),
    "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1": (
        "R6c_group_riskaware_top8_sell12"
    ),
    "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_softbuy_top8_sell12_rotation_internaldays_v1": (
        "R6c_group_riskaware_softbuy_top8_sell12"
    ),
    "R6d_root_K20_stock_K5_PD_mild_slice_group_riskaware_softbuy_v2_top8_sell12_rotation_internaldays_v1": (
        "R6d_true_softbuy_v2_top8_sell12"
    ),
    "R4_root_K20_PD_incremental_top5_flow_on_R3v1_v1_TRUE": "R4_incremental_TRUE",
}


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def selection_score(sharpe: pd.Series) -> float:
    s = pd.to_numeric(sharpe, errors="coerce").dropna()
    if s.empty:
        return np.nan
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    return float(s.mean() - 0.5 * std)


def read_ground_truth_registry() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(
        *,
        label: str,
        source_file: str,
        selection: float,
        sharpe: float,
        sharpe_std: float,
        return_pct: float,
        max_dd: float,
        cash: float,
        turnover: float,
        timesteps: Any = "",
        note: str = "",
    ) -> None:
        rows.append(
            {
                "true_label": label,
                "source_file": source_file,
                "selection_score": safe_float(selection),
                "mean_sharpe": safe_float(sharpe),
                "sharpe_std": safe_float(sharpe_std, 0.0),
                "mean_return_pct": safe_float(return_pct),
                "mean_max_drawdown": safe_float(max_dd),
                "mean_cash": safe_float(cash),
                "mean_turnover_l1": safe_float(turnover),
                "timesteps": timesteps,
                "note": note,
            }
        )

    p = REPORT_DIR / "r3_v1_vs_e1_e12_comparison.csv"
    if p.exists():
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            add_row(
                label=str(row.get("variant", "")),
                source_file=str(p.relative_to(ROOT)),
                selection=row.get("selection_score"),
                sharpe=row.get("mean_sharpe"),
                sharpe_std=row.get("sharpe_std"),
                return_pct=row.get("mean_return"),
                max_dd=row.get("mean_max_dd"),
                cash=row.get("mean_cash"),
                turnover=row.get("mean_turnover"),
                timesteps=row.get("timesteps", ""),
                note=str(row.get("conclusion", "")),
            )

    p = REPORT_DIR / "r4_incremental_top5_TRUE_compare.csv"
    if p.exists():
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            add_row(
                label=str(row.get("label", "")),
                source_file=str(p.relative_to(ROOT)),
                selection=row.get("selection_score"),
                sharpe=row.get("sharpe_mean"),
                sharpe_std=row.get("sharpe_std"),
                return_pct=row.get("return_pct_mean"),
                max_dd=row.get("max_drawdown_mean"),
                cash=row.get("cash_weight_mean"),
                turnover=row.get("turnover_l1_mean"),
                note="R4/R3 controller-family aggregate from completed run.",
            )

    p = REPORT_DIR / "r5_r6_result_fold_metrics.csv"
    if p.exists():
        df = pd.read_csv(p)
        for variant, group in df.groupby("variant", dropna=False):
            sharpe = pd.to_numeric(group["sharpe"], errors="coerce")
            add_row(
                label=str(variant),
                source_file=str(p.relative_to(ROOT)),
                selection=selection_score(sharpe),
                sharpe=sharpe.mean(),
                sharpe_std=sharpe.std(ddof=1),
                return_pct=pd.to_numeric(group["return_pct"], errors="coerce").mean(),
                max_dd=pd.to_numeric(group["max_dd"], errors="coerce").mean(),
                cash=pd.to_numeric(group["cash"], errors="coerce").mean(),
                turnover=pd.to_numeric(group["turnover_l1"], errors="coerce").mean(),
                timesteps=pd.to_numeric(group.get("metadata_total_timesteps", pd.Series(dtype=float)), errors="coerce").max(),
                note="R5/R6 completed run aggregate.",
            )

    for p in [
        REPORT_DIR / "R6c_real_run_aggregate.csv",
        REPORT_DIR / "R6d_compare_run_aggregate.csv",
    ]:
        if p.exists():
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                label = str(row.get("run", row.get("variant", "")))
                if not label:
                    continue
                add_row(
                    label=label,
                    source_file=str(p.relative_to(ROOT)),
                    selection=row.get("selection_score"),
                    sharpe=row.get("mean_sharpe", row.get("sharpe")),
                    sharpe_std=row.get("std_sharpe", row.get("sharpe_std", 0.0)),
                    return_pct=row.get("return_pct", row.get("mean_return_pct")),
                    max_dd=row.get("max_drawdown", row.get("mean_max_drawdown")),
                    cash=row.get("cash_weight_mean", row.get("mean_cash")),
                    turnover=row.get("turnover_l1_mean", row.get("mean_turnover_l1")),
                    timesteps="internal_days",
                    note=f"Completed real run aggregate from {p.name}.",
                )

    p = REPORT_DIR / "all_existing_result_run_ranking_snapshot.csv"
    if p.exists():
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            label = str(row.get("folder", ""))
            if not label:
                continue
            add_row(
                label=label,
                source_file=str(p.relative_to(ROOT)),
                selection=row.get("selection_score"),
                sharpe=row.get("mean_sharpe"),
                sharpe_std=row.get("sharpe_std", 0.0),
                return_pct=row.get("mean_return_pct"),
                max_dd=row.get("mean_max_drawdown"),
                cash=row.get("mean_cash"),
                turnover=row.get("mean_turnover_l1"),
                note="All-existing-runs ranking snapshot.",
            )

    registry = pd.DataFrame(rows)
    if registry.empty:
        return registry
    registry = registry.drop_duplicates(subset=["true_label", "source_file"], keep="first")
    return registry.sort_values("selection_score", ascending=False)

