"""Stage 2 behavior diagnostics for R6c Stage 1 primitive codes.

Inputs are the Stage 1 outputs produced by either:

* scripts/run_stage1_r6c_vqvae.py
* scripts/run_stage1_r6c_vq.py

The script reads behavior_log_daily.parquet, filters valid Stage 1 rows, and
builds code-level diagnostics for the R6c hierarchy:

* risk/cash root behavior
* confidence and event-trigger behavior
* Top-K/group-aware execution behavior
* per-ticker weights/trades/selection rates
* run lengths and transition probabilities
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STAGE1_DIR = (
    ROOT
    / "artifacts"
    / "stage1"
    / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_stage1_vqvae"
)
DEFAULT_KMEANS_DIR = (
    ROOT
    / "artifacts"
    / "stage1"
    / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_stage1"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "artifacts"
    / "stage2"
    / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_stage2_vqvae"
)


def safe_mean(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else math.nan


def safe_std(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    return float(vals.std(ddof=1)) if len(vals) > 1 else math.nan


def ann_sharpe(x: pd.Series) -> float:
    mu = safe_mean(x)
    sigma = safe_std(x)
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        return math.nan
    return float(math.sqrt(252.0) * mu / sigma)


def compound_return(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return math.nan
    return float(np.prod(1.0 + vals) - 1.0)


def qmean(group: pd.DataFrame, col: str) -> float:
    return safe_mean(group[col]) if col in group.columns else math.nan


def qsum(group: pd.DataFrame, col: str) -> float:
    if col not in group.columns:
        return math.nan
    vals = pd.to_numeric(group[col], errors="coerce").dropna()
    return float(vals.sum()) if len(vals) else math.nan


def infer_tickers(df: pd.DataFrame) -> list[str]:
    tickers: list[str] = []
    for col in df.columns:
        if col.startswith("executed_weight_"):
            ticker = col.removeprefix("executed_weight_")
            if ticker != "CASH":
                tickers.append(ticker)
    return sorted(tickers)


def top_items(values: dict[str, float], n: int = 8) -> str:
    clean = {k: v for k, v in values.items() if np.isfinite(v)}
    items = sorted(clean.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return "; ".join(f"{k}:{v:.4f}" for k, v in items)


def markdown_table(df: pd.DataFrame, cols: list[str] | None = None, floatfmt: str = ".4f") -> str:
    view = df if cols is None else df[cols]
    headers = [str(c) for c in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in view.iterrows():
        vals: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                vals.append("" if not np.isfinite(value) else format(float(value), floatfmt))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def make_runs(valid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if valid.empty:
        return pd.DataFrame()
    start = 0
    codes = valid["code_id"].astype(int).to_numpy()
    for idx in range(1, len(valid)):
        if codes[idx] != codes[start]:
            block = valid.iloc[start:idx]
            rows.append(run_row(int(codes[start]), block))
            start = idx
    rows.append(run_row(int(codes[start]), valid.iloc[start:]))
    return pd.DataFrame(rows)


def run_row(code: int, block: pd.DataFrame) -> dict[str, Any]:
    return {
        "code_id": code,
        "start_date": str(block["date"].iloc[0]),
        "end_date": str(block["date"].iloc[-1]),
        "length_days": int(len(block)),
        "compound_return": compound_return(block.get("net_return", pd.Series(dtype=float))),
        "mean_return": safe_mean(block.get("net_return", pd.Series(dtype=float))),
        "mean_cash_target": qmean(block, "cash_target"),
        "mean_risk_stress": qmean(block, "risk_stress"),
        "mean_recovery_score": qmean(block, "recovery_score"),
        "mean_turnover_l1": qmean(block, "turnover_l1"),
    }


def transitions(valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    codes = sorted(int(x) for x in valid["code_id"].unique())
    counts = pd.DataFrame(0, index=codes, columns=codes, dtype=int)
    seq = valid["code_id"].astype(int).to_numpy()
    for a, b in zip(seq[:-1], seq[1:]):
        counts.loc[int(a), int(b)] += 1
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    counts.index.name = "from_code"
    probs.index.name = "from_code"
    counts.columns.name = "to_code"
    probs.columns.name = "to_code"
    return counts.reset_index(), probs.reset_index()


def primitive_summary(valid: pd.DataFrame, runs: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = max(len(valid), 1)
    global_cash = qmean(valid, "cash_target")
    global_turnover = qmean(valid, "turnover_l1")
    global_risk_stress = qmean(valid, "risk_stress")
    for code, group in valid.groupby("code_id", sort=True):
        code_runs = runs.loc[runs["code_id"].eq(code)]
        exec_means = {t: qmean(group, f"executed_weight_{t}") for t in tickers}
        target_means = {t: qmean(group, f"target_weight_{t}") for t in tickers}
        buy_rates = {t: qmean(group, f"incremental_topk_selected_buy_{t}") for t in tickers}
        sell_rates = {t: qmean(group, f"incremental_topk_selected_sell_{t}") for t in tickers}
        flow_abs = {t: safe_mean(group.get(f"trade_abs_weight_{t}", pd.Series(dtype=float))) for t in tickers}
        row = {
            "code_id": int(code),
            "n": int(len(group)),
            "share_valid": float(len(group) / total),
            "date_start": str(group["date"].min()),
            "date_end": str(group["date"].max()),
            "years_active": int(pd.to_datetime(group["date"]).dt.year.nunique()),
            "num_runs": int(len(code_runs)),
            "median_run_length_days": float(code_runs["length_days"].median()) if len(code_runs) else math.nan,
            "max_run_length_days": int(code_runs["length_days"].max()) if len(code_runs) else 0,
            "mean_net_return": qmean(group, "net_return"),
            "median_net_return": float(pd.to_numeric(group.get("net_return"), errors="coerce").median()),
            "ann_sharpe_net_return": ann_sharpe(group.get("net_return", pd.Series(dtype=float))),
            "compound_return": compound_return(group.get("net_return", pd.Series(dtype=float))),
            "mean_reward": qmean(group, "reward"),
            "mean_cash_target": qmean(group, "cash_target"),
            "mean_q_target": qmean(group, "q_target"),
            "mean_cash_anchor": qmean(group, "cash_anchor"),
            "mean_q_anchor": qmean(group, "q_anchor"),
            "mean_cash_scheduled": qmean(group, "cash_scheduled"),
            "mean_q_scheduled": qmean(group, "q_scheduled"),
            "cash_target_p10": float(pd.to_numeric(group.get("cash_target"), errors="coerce").quantile(0.10)),
            "cash_target_p50": float(pd.to_numeric(group.get("cash_target"), errors="coerce").quantile(0.50)),
            "cash_target_p90": float(pd.to_numeric(group.get("cash_target"), errors="coerce").quantile(0.90)),
            "mean_risk_stress": qmean(group, "risk_stress"),
            "mean_recovery_score": qmean(group, "recovery_score"),
            "mean_confidence_rerisk": qmean(group, "confidence_rerisk"),
            "mean_confidence_derisk": qmean(group, "confidence_derisk"),
            "recovery_trigger_rate": qmean(group, "recovery_trigger"),
            "risk_break_trigger_rate": qmean(group, "risk_break_trigger"),
            "window_closed_early_rate": qmean(group, "window_closed_early"),
            "stop_active_rate": qmean(group, "stop_active"),
            "mean_turnover_l1": qmean(group, "turnover_l1"),
            "mean_stock_turnover_l1": qmean(group, "stock_turnover_l1"),
            "mean_trade_buy_count": qmean(group, "trade_buy_count"),
            "mean_trade_sell_count": qmean(group, "trade_sell_count"),
            "mean_trade_hold_count": qmean(group, "trade_hold_count"),
            "mean_trade_buy_weight_l1": qmean(group, "trade_buy_weight_l1"),
            "mean_trade_sell_weight_l1": qmean(group, "trade_sell_weight_l1"),
            "mean_drawdown": qmean(group, "drawdown"),
            "mean_concentration": qmean(group, "concentration"),
            "mean_risky_hhi_target": qmean(group, "risky_hhi_target"),
            "mean_risky_entropy_target": qmean(group, "risky_entropy_target"),
            "mean_topk_buy_selected_count": qmean(group, "topk_buy_selected_count"),
            "mean_topk_sell_selected_count": qmean(group, "topk_sell_selected_count"),
            "mean_incremental_buy_allowed": qmean(group, "incremental_topk_buy_allowed"),
            "mean_incremental_buy_fill_scale": qmean(group, "incremental_topk_buy_fill_scale"),
            "mean_incremental_sell_multiplier": qmean(group, "incremental_topk_sell_multiplier_mean"),
            "mean_rotation_requested": qmean(group, "incremental_topk_rotation_requested"),
            "mean_rotation_unfilled": qmean(group, "incremental_topk_rotation_unfilled"),
            "mean_vix": qmean(group, "market_feature_VIX"),
            "mean_vix_change_5d": qmean(group, "market_feature_VIX_change_5d"),
            "mean_sp500_trend": qmean(group, "market_feature_SP500_Trend"),
            "mean_universe_return_20d": qmean(group, "market_feature_universe_return_20d"),
            "mean_residual_universe_return_20d": qmean(group, "market_feature_residual_universe_return_20d"),
            "mean_residual_breadth_20d": qmean(group, "market_feature_residual_breadth_20d"),
            "mean_regime_p1": qmean(group, "market_feature_Regime_1_Prob"),
            "cash_vs_global": qmean(group, "cash_target") - global_cash,
            "turnover_vs_global": qmean(group, "turnover_l1") - global_turnover,
            "risk_stress_vs_global": qmean(group, "risk_stress") - global_risk_stress,
            "top8_executed_weights": top_items(exec_means, 8),
            "top8_target_weights": top_items(target_means, 8),
            "top8_buy_selected_rates": top_items(buy_rates, 8),
            "top8_sell_selected_rates": top_items(sell_rates, 8),
            "top8_trade_abs": top_items(flow_abs, 8),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def asset_summary(valid: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for code, group in valid.groupby("code_id", sort=True):
        for ticker in tickers:
            rows.append(
                {
                    "code_id": int(code),
                    "ticker": ticker,
                    "mean_executed_weight": qmean(group, f"executed_weight_{ticker}"),
                    "mean_target_weight": qmean(group, f"target_weight_{ticker}"),
                    "mean_anchor_weight": qmean(group, f"anchor_weight_{ticker}"),
                    "mean_trade_delta": qmean(group, f"trade_delta_weight_{ticker}"),
                    "mean_trade_abs": qmean(group, f"trade_abs_weight_{ticker}"),
                    "buy_day_rate": math.nan,
                    "sell_day_rate": math.nan,
                    "hold_day_rate": math.nan,
                    "topk_buy_selected_rate": qmean(group, f"incremental_topk_selected_buy_{ticker}"),
                    "topk_sell_selected_rate": qmean(group, f"incremental_topk_selected_sell_{ticker}"),
                    "topk_rotation_buy_rate": qmean(group, f"incremental_topk_rotation_selected_buy_{ticker}"),
                    "topk_rotation_sell_rate": qmean(group, f"incremental_topk_rotation_selected_sell_{ticker}"),
                    "topk_flow_delta_mean": qmean(group, f"incremental_topk_flow_delta_{ticker}"),
                    "buy_priority_mean": qmean(group, f"incremental_topk_buy_priority_{ticker}"),
                    "sell_priority_mean": qmean(group, f"incremental_topk_sell_priority_{ticker}"),
                    "sell_multiplier_mean": qmean(group, f"incremental_topk_sell_multiplier_{ticker}"),
                    "residual_deterioration_mean": qmean(group, f"incremental_topk_residual_deterioration_{ticker}"),
                }
            )
    out = pd.DataFrame(rows)
    direction_cols = [c for c in valid.columns if c.startswith("trade_direction_")]
    if direction_cols:
        for i, row in out.iterrows():
            col = f"trade_direction_{row['ticker']}"
            if col in valid.columns:
                sub = valid.loc[valid["code_id"].eq(row["code_id"]), col]
                out.at[i, "buy_day_rate"] = float((pd.to_numeric(sub, errors="coerce") > 0).mean())
                out.at[i, "sell_day_rate"] = float((pd.to_numeric(sub, errors="coerce") < 0).mean())
                out.at[i, "hold_day_rate"] = float((pd.to_numeric(sub, errors="coerce") == 0).mean())
    return out


def classify(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        labels: list[str] = []
        if row["mean_cash_target"] >= summary["mean_cash_target"].quantile(0.75):
            labels.append("cash_heavier")
        if row["mean_q_target"] >= summary["mean_q_target"].quantile(0.75):
            labels.append("risk_on")
        if row["mean_risk_stress"] >= summary["mean_risk_stress"].quantile(0.75):
            labels.append("stress")
        if row["mean_recovery_score"] >= summary["mean_recovery_score"].quantile(0.75):
            labels.append("recovery")
        if row["mean_turnover_l1"] >= summary["mean_turnover_l1"].quantile(0.75):
            labels.append("active_trading")
        if row["mean_incremental_buy_allowed"] >= summary["mean_incremental_buy_allowed"].quantile(0.75):
            labels.append("topk_buy_open")
        if row["risk_break_trigger_rate"] > 0.05:
            labels.append("risk_break")
        if not labels:
            labels.append("baseline_hold")
        rows.append(
            {
                "code_id": int(row["code_id"]),
                "label": "+".join(labels),
                "n": int(row["n"]),
                "share_valid": float(row["share_valid"]),
                "confidence": "high" if int(row["n"]) >= 100 and len(labels) <= 3 else "medium",
                "rationale": (
                    f"cash={row['mean_cash_target']:.3f}, q={row['mean_q_target']:.3f}, "
                    f"risk_stress={row['mean_risk_stress']:.3f}, recovery={row['mean_recovery_score']:.3f}, "
                    f"turnover={row['mean_turnover_l1']:.4f}, buy_allowed={row['mean_incremental_buy_allowed']:.3f}"
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_codebooks(stage1_dir: Path, reference_dir: Path | None, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = pd.read_csv(stage1_dir / "stage1_metrics.csv")
    metrics["codebook"] = stage1_dir.name
    rows.append({"type": "metric_source", "path": str(stage1_dir / "stage1_metrics.csv")})
    if reference_dir and (reference_dir / "train_codes.parquet").exists():
        a = pd.read_parquet(stage1_dir / "train_codes.parquet")
        b = pd.read_parquet(reference_dir / "train_codes.parquet")
        merged = a.merge(b, on="date", suffixes=("_main", "_reference"))
        valid = merged["valid_main"].astype(bool) & merged["valid_reference"].astype(bool)
        if valid.sum() > 0:
            nmi = normalized_nmi(
                merged.loc[valid, "code_id_main"].astype(int).to_numpy(),
                merged.loc[valid, "code_id_reference"].astype(int).to_numpy(),
            )
        else:
            nmi = math.nan
        compare = pd.DataFrame(
            [
                {
                    "main_stage1": stage1_dir.name,
                    "reference_stage1": reference_dir.name,
                    "overlap_rows": int(len(merged)),
                    "overlap_valid_rows": int(valid.sum()),
                    "code_nmi": nmi,
                }
            ]
        )
        compare.to_csv(out_dir / "stage2_codebook_comparison.csv", index=False)
        return compare
    return pd.DataFrame()


def normalized_nmi(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import normalized_mutual_info_score

    if len(set(a)) <= 1 or len(set(b)) <= 1:
        return math.nan
    return float(normalized_mutual_info_score(a, b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--reference-stage1-dir", type=Path, default=DEFAULT_KMEANS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_parquet(args.stage1_dir / "behavior_log_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    daily["valid"] = daily["valid"].astype(bool)
    valid = daily.loc[daily["valid"]].copy()
    tickers = infer_tickers(valid)

    runs = make_runs(valid)
    trans_counts, trans_probs = transitions(valid)
    summary = primitive_summary(valid, runs, tickers)
    assets = asset_summary(valid, tickers)
    labels = classify(summary)
    compare = compare_codebooks(args.stage1_dir, args.reference_stage1_dir, args.out_dir)

    daily.to_csv(args.out_dir / "stage2_joined_daily_diagnostics.csv", index=False)
    valid.to_csv(args.out_dir / "stage2_joined_daily_diagnostics_valid.csv", index=False)
    summary.to_csv(args.out_dir / "stage2_primitive_behavior_summary.csv", index=False)
    labels.to_csv(args.out_dir / "stage2_primitive_labels.csv", index=False)
    assets.to_csv(args.out_dir / "stage2_primitive_asset_summary.csv", index=False)
    runs.to_csv(args.out_dir / "stage2_primitive_runs.csv", index=False)
    trans_counts.to_csv(args.out_dir / "stage2_primitive_transition_counts.csv", index=False)
    trans_probs.to_csv(args.out_dir / "stage2_primitive_transition_probs.csv", index=False)

    metrics = pd.read_csv(args.stage1_dir / "stage1_metrics.csv")
    selected_metrics = metrics.loc[metrics["K"].eq(8)]
    selected_text = selected_metrics.to_string(index=False) if len(selected_metrics) else metrics.to_string(index=False)
    label_view = labels.merge(summary, on=["code_id", "n", "share_valid"], how="left")
    report = f"""# R6c Stage 2 Behavior Diagnostics

Stage 1 source:

`{args.stage1_dir}`

Rows:

- all rows: `{len(daily)}`
- valid rows: `{len(valid)}`
- tickers: `{len(tickers)}`

## Stage 1 Metrics

```text
{selected_text}
```

## Primitive Labels

{markdown_table(label_view, ['code_id', 'label', 'confidence', 'n', 'share_valid', 'rationale'], floatfmt='.4f')}

## Core Behavior Summary

{markdown_table(summary, ['code_id', 'n', 'share_valid', 'median_run_length_days', 'mean_cash_target', 'mean_q_target', 'mean_risk_stress', 'mean_recovery_score', 'mean_confidence_rerisk', 'mean_confidence_derisk', 'mean_turnover_l1', 'mean_incremental_buy_allowed', 'risk_break_trigger_rate', 'recovery_trigger_rate'], floatfmt='.4f')}

## Top Tickers By Primitive

{markdown_table(summary, ['code_id', 'top8_executed_weights', 'top8_buy_selected_rates', 'top8_sell_selected_rates'], floatfmt='.4f')}

## Codebook Comparison

{compare.to_string(index=False) if len(compare) else 'No reference codebook comparison available.'}

## Notes

For R6c, Stage 2 should be read through the hierarchy:

1. root/cash: `q_*`, `cash_*`
2. confidence/events: `risk_stress`, `recovery_score`, `confidence_*`, `risk_break_trigger`, `recovery_trigger`
3. Top-K/group-aware execution: `incremental_topk_*`
4. per-stock behavior: `trade_*`, `executed_weight_*`, `target_weight_*`

Use `stage2_primitive_behavior_summary.csv` for primitive-level interpretation
and `stage2_primitive_asset_summary.csv` for ticker-level interpretation.
"""
    (args.out_dir / "STAGE2_R6C_BEHAVIOR_DIAGNOSTICS.md").write_text(report, encoding="utf-8")
    manifest = {
        "stage1_dir": str(args.stage1_dir),
        "reference_stage1_dir": str(args.reference_stage1_dir) if args.reference_stage1_dir else None,
        "out_dir": str(args.out_dir),
        "rows_all": int(len(daily)),
        "rows_valid": int(len(valid)),
        "tickers": tickers,
        "outputs": [
            "stage2_joined_daily_diagnostics.csv",
            "stage2_joined_daily_diagnostics_valid.csv",
            "stage2_primitive_behavior_summary.csv",
            "stage2_primitive_labels.csv",
            "stage2_primitive_asset_summary.csv",
            "stage2_primitive_runs.csv",
            "stage2_primitive_transition_counts.csv",
            "stage2_primitive_transition_probs.csv",
            "stage2_codebook_comparison.csv",
            "STAGE2_R6C_BEHAVIOR_DIAGNOSTICS.md",
        ],
    }
    (args.out_dir / "stage2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Stage 2 R6c diagnostics written to {args.out_dir}")


if __name__ == "__main__":
    main()
