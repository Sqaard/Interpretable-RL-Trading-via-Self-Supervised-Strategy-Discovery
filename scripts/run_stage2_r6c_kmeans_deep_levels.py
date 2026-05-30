"""Deep hierarchy-level interpretation for R6c KMeans Stage 2 codes."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE2_DIR = (
    ROOT
    / "artifacts"
    / "stage2"
    / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_stage2_kmeans"
)


def mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return math.nan
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else math.nan


def rate(df: pd.DataFrame, col: str) -> float:
    return mean(df, col)


def q(df: pd.DataFrame, col: str, prob: float) -> float:
    if col not in df.columns:
        return math.nan
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.quantile(prob)) if len(vals) else math.nan


def top_values(items: dict[str, float], n: int = 8) -> str:
    clean = {k: v for k, v in items.items() if np.isfinite(v)}
    return "; ".join(f"{k}:{v:.4f}" for k, v in sorted(clean.items(), key=lambda kv: kv[1], reverse=True)[:n])


def infer_tickers(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        if col.startswith("executed_weight_"):
            ticker = col.removeprefix("executed_weight_")
            if ticker != "CASH":
                out.append(ticker)
    return sorted(out)


def infer_rescorr_groups(df: pd.DataFrame) -> list[str]:
    groups = set()
    pat = re.compile(r"^incremental_topk_group_prev_(.+)$")
    for col in df.columns:
        m = pat.match(col)
        if m:
            groups.add(m.group(1))
    return sorted(groups)


def normalize_score(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    std = vals.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return vals * 0.0
    return (vals - vals.mean()) / std


def root_level(valid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, g in valid.groupby("code_id", sort=True):
        rows.append(
            {
                "code_id": int(code),
                "n": int(len(g)),
                "mean_q_anchor": mean(g, "q_anchor"),
                "mean_q_scheduled": mean(g, "q_scheduled"),
                "mean_q_target": mean(g, "q_target"),
                "mean_cash_anchor": mean(g, "cash_anchor"),
                "mean_cash_scheduled": mean(g, "cash_scheduled"),
                "mean_cash_target": mean(g, "cash_target"),
                "cash_p10": q(g, "cash_target", 0.10),
                "cash_p50": q(g, "cash_target", 0.50),
                "cash_p90": q(g, "cash_target", 0.90),
                "mean_delta_q_anchor": mean(g, "delta_q_anchor"),
                "mean_delta_q_scheduled": mean(g, "delta_q_scheduled"),
                "root_anchor_risk_rate": rate(g, "root_anchor_risk_day"),
                "root_anchor_cash_rate": rate(g, "root_anchor_cash_day"),
                "root_anchor_hold_rate": rate(g, "root_anchor_hold_day"),
                "root_scheduled_risk_rate": rate(g, "root_scheduled_risk_day"),
                "root_scheduled_cash_rate": rate(g, "root_scheduled_cash_day"),
                "root_scheduled_hold_rate": rate(g, "root_scheduled_hold_day"),
                "mean_cash_trade_delta": mean(g, "cash_trade_delta"),
                "cash_buy_rate": float((pd.to_numeric(g.get("cash_trade_direction"), errors="coerce") > 0).mean())
                if "cash_trade_direction" in g
                else math.nan,
                "cash_sell_rate": float((pd.to_numeric(g.get("cash_trade_direction"), errors="coerce") < 0).mean())
                if "cash_trade_direction" in g
                else math.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["cash_target_z"] = normalize_score(out["mean_cash_target"])
    out["q_target_z"] = normalize_score(out["mean_q_target"])
    return out


def confidence_level(valid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, g in valid.groupby("code_id", sort=True):
        rows.append(
            {
                "code_id": int(code),
                "n": int(len(g)),
                "mean_risk_stress": mean(g, "risk_stress"),
                "risk_stress_p90": q(g, "risk_stress", 0.90),
                "mean_recovery_score": mean(g, "recovery_score"),
                "recovery_score_p90": q(g, "recovery_score", 0.90),
                "mean_confidence_rerisk": mean(g, "confidence_rerisk"),
                "mean_confidence_derisk": mean(g, "confidence_derisk"),
                "confidence_spread_rerisk_minus_derisk": mean(g, "confidence_rerisk") - mean(g, "confidence_derisk"),
                "recovery_trigger_rate": rate(g, "recovery_trigger"),
                "risk_break_trigger_rate": rate(g, "risk_break_trigger"),
                "window_closed_early_rate": rate(g, "window_closed_early"),
                "stop_active_rate": rate(g, "stop_active"),
                "mean_suppressed_trade_l1": mean(g, "suppressed_trade_l1"),
                "mean_suppressed_turnover": mean(g, "suppressed_turnover"),
                "mean_vix": mean(g, "market_feature_VIX"),
                "mean_vix_change_5d": mean(g, "market_feature_VIX_change_5d"),
                "mean_sp500_trend": mean(g, "market_feature_SP500_Trend"),
                "mean_universe_return_20d": mean(g, "market_feature_universe_return_20d"),
                "mean_residual_return_20d": mean(g, "market_feature_residual_universe_return_20d"),
                "mean_residual_breadth_20d": mean(g, "market_feature_residual_breadth_20d"),
                "mean_regime_p1": mean(g, "market_feature_Regime_1_Prob"),
            }
        )
    out = pd.DataFrame(rows)
    for col in ["mean_risk_stress", "mean_recovery_score", "mean_confidence_rerisk", "mean_confidence_derisk"]:
        out[col + "_z"] = normalize_score(out[col])
    return out


def topk_level(valid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, g in valid.groupby("code_id", sort=True):
        rows.append(
            {
                "code_id": int(code),
                "n": int(len(g)),
                "mean_buy_requested": mean(g, "incremental_topk_buy_requested"),
                "mean_buy_filled": mean(g, "incremental_topk_buy_filled"),
                "mean_buy_unfilled": mean(g, "incremental_topk_buy_unfilled"),
                "mean_sell_requested": mean(g, "incremental_topk_sell_requested"),
                "mean_sell_filled": mean(g, "incremental_topk_sell_filled"),
                "mean_sell_unfilled": mean(g, "incremental_topk_sell_unfilled"),
                "mean_sell_expansion_count": mean(g, "incremental_topk_sell_expansion_count"),
                "mean_sell_final_k": mean(g, "incremental_topk_sell_final_k"),
                "mean_selected_buy_count": mean(g, "incremental_topk_selected_buy_count"),
                "mean_selected_sell_count": mean(g, "incremental_topk_selected_sell_count"),
                "mean_buy_allowed": mean(g, "incremental_topk_buy_allowed"),
                "mean_buy_fill_scale": mean(g, "incremental_topk_buy_fill_scale"),
                "buy_gate_hard_block_rate": rate(g, "incremental_topk_buy_gate_hard_block"),
                "buy_gate_soft_score": mean(g, "incremental_topk_buy_gate_soft_score"),
                "pass_conf_rerisk_rate": rate(g, "incremental_topk_buy_gate_pass_conf_rerisk"),
                "pass_recovery_rate": rate(g, "incremental_topk_buy_gate_pass_recovery"),
                "pass_risk_stress_rate": rate(g, "incremental_topk_buy_gate_pass_risk_stress"),
                "pass_breadth_5d_rate": rate(g, "incremental_topk_buy_gate_pass_breadth_5d"),
                "pass_breadth_20d_rate": rate(g, "incremental_topk_buy_gate_pass_breadth_20d"),
                "mean_rotation_requested": mean(g, "incremental_topk_rotation_requested"),
                "mean_rotation_sell_filled": mean(g, "incremental_topk_rotation_sell_filled"),
                "mean_rotation_buy_filled": mean(g, "incremental_topk_rotation_buy_filled"),
                "mean_rotation_unfilled": mean(g, "incremental_topk_rotation_unfilled"),
                "mean_rotation_stress_gate": mean(g, "incremental_topk_rotation_stress_gate"),
                "mean_sell_multiplier": mean(g, "incremental_topk_sell_multiplier_mean"),
                "mean_sell_multiplier_p90": mean(g, "incremental_topk_sell_multiplier_p90"),
                "mean_residual_deterioration": mean(g, "incremental_topk_residual_deterioration_mean"),
                "mean_input_to_output_l1": mean(g, "incremental_topk_input_to_output_l1"),
                "mean_flow_l1": mean(g, "incremental_topk_flow_l1"),
                "mean_flow_turnover": mean(g, "incremental_topk_flow_turnover"),
            }
        )
    out = pd.DataFrame(rows)
    for col in ["mean_buy_allowed", "mean_sell_requested", "mean_rotation_requested", "mean_sell_multiplier"]:
        out[col + "_z"] = normalize_score(out[col])
    return out


def group_level(valid: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for code, g in valid.groupby("code_id", sort=True):
        for group in groups:
            rows.append(
                {
                    "code_id": int(code),
                    "group": group,
                    "n": int(len(g)),
                    "mean_prev_weight": mean(g, f"incremental_topk_group_prev_{group}"),
                    "mean_target_weight": mean(g, f"incremental_topk_group_target_{group}"),
                    "mean_cap": mean(g, f"incremental_topk_group_cap_{group}"),
                    "mean_buy_pressure": mean(g, f"incremental_topk_group_buy_pressure_{group}"),
                    "mean_sell_pressure": mean(g, f"incremental_topk_group_sell_pressure_{group}"),
                    "mean_buy_capacity": mean(g, f"incremental_topk_group_buy_capacity_{group}"),
                    "mean_sell_overweight": mean(g, f"incremental_topk_group_sell_overweight_{group}"),
                    "mean_buy_multiplier": mean(g, f"incremental_topk_group_buy_multiplier_{group}"),
                    "mean_sell_multiplier": mean(g, f"incremental_topk_group_sell_multiplier_{group}"),
                    "mean_target_minus_prev": mean(g, f"incremental_topk_group_target_{group}")
                    - mean(g, f"incremental_topk_group_prev_{group}"),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["buy_pressure_rank_within_code"] = out.groupby("code_id")["mean_buy_pressure"].rank(ascending=False, method="dense")
        out["sell_pressure_rank_within_code"] = out.groupby("code_id")["mean_sell_pressure"].rank(ascending=False, method="dense")
        out["buy_multiplier_rank_within_code"] = out.groupby("code_id")["mean_buy_multiplier"].rank(ascending=False, method="dense")
        out["sell_multiplier_rank_within_code"] = out.groupby("code_id")["mean_sell_multiplier"].rank(ascending=False, method="dense")
    return out


def ticker_level(valid: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for code, g in valid.groupby("code_id", sort=True):
        for ticker in tickers:
            direction = pd.to_numeric(g.get(f"trade_direction_{ticker}"), errors="coerce")
            rows.append(
                {
                    "code_id": int(code),
                    "ticker": ticker,
                    "n": int(len(g)),
                    "mean_executed_weight": mean(g, f"executed_weight_{ticker}"),
                    "mean_target_weight": mean(g, f"target_weight_{ticker}"),
                    "mean_anchor_weight": mean(g, f"anchor_weight_{ticker}"),
                    "mean_trade_delta": mean(g, f"trade_delta_weight_{ticker}"),
                    "mean_trade_abs": mean(g, f"trade_abs_weight_{ticker}"),
                    "buy_day_rate": float((direction > 0).mean()) if len(direction) else math.nan,
                    "sell_day_rate": float((direction < 0).mean()) if len(direction) else math.nan,
                    "hold_day_rate": float((direction == 0).mean()) if len(direction) else math.nan,
                    "topk_buy_selected_rate": mean(g, f"incremental_topk_selected_buy_{ticker}"),
                    "topk_sell_selected_rate": mean(g, f"incremental_topk_selected_sell_{ticker}"),
                    "rotation_buy_rate": mean(g, f"incremental_topk_rotation_selected_buy_{ticker}"),
                    "rotation_sell_rate": mean(g, f"incremental_topk_rotation_selected_sell_{ticker}"),
                    "mean_buy_priority": mean(g, f"incremental_topk_buy_priority_{ticker}"),
                    "mean_sell_priority": mean(g, f"incremental_topk_sell_priority_{ticker}"),
                    "mean_sell_multiplier": mean(g, f"incremental_topk_sell_multiplier_{ticker}"),
                    "mean_residual_deterioration": mean(g, f"incremental_topk_residual_deterioration_{ticker}"),
                    "mean_flow_delta": mean(g, f"incremental_topk_flow_delta_{ticker}"),
                }
            )
    out = pd.DataFrame(rows)
    out["executed_rank_within_code"] = out.groupby("code_id")["mean_executed_weight"].rank(ascending=False, method="dense")
    out["buy_selected_rank_within_code"] = out.groupby("code_id")["topk_buy_selected_rate"].rank(ascending=False, method="dense")
    out["sell_selected_rank_within_code"] = out.groupby("code_id")["topk_sell_selected_rate"].rank(ascending=False, method="dense")
    return out


def make_interpretation(root: pd.DataFrame, conf: pd.DataFrame, topk: pd.DataFrame, groups: pd.DataFrame, tickers: pd.DataFrame) -> pd.DataFrame:
    merged = root.merge(conf, on=["code_id", "n"], suffixes=("_root", "_conf")).merge(topk, on=["code_id", "n"])
    rows = []
    for _, r in merged.iterrows():
        code = int(r["code_id"])
        code_groups = groups[groups["code_id"] == code]
        code_tickers = tickers[tickers["code_id"] == code]
        top_buy_groups = top_values(dict(zip(code_groups["group"], code_groups["mean_buy_pressure"])), 3)
        top_sell_groups = top_values(dict(zip(code_groups["group"], code_groups["mean_sell_pressure"])), 3)
        top_buy_tickers = top_values(dict(zip(code_tickers["ticker"], code_tickers["topk_buy_selected_rate"])), 6)
        top_sell_tickers = top_values(dict(zip(code_tickers["ticker"], code_tickers["topk_sell_selected_rate"])), 6)
        top_weights = top_values(dict(zip(code_tickers["ticker"], code_tickers["mean_executed_weight"])), 6)

        root_label = "risk-on" if r["mean_cash_target"] < merged["mean_cash_target"].quantile(0.35) else "cash-heavy" if r["mean_cash_target"] > merged["mean_cash_target"].quantile(0.65) else "balanced"
        conf_label = (
            "recovery-open"
            if r["mean_recovery_score"] > merged["mean_recovery_score"].quantile(0.70)
            else "stress"
            if r["mean_risk_stress"] > merged["mean_risk_stress"].quantile(0.70)
            else "neutral"
        )
        topk_label = (
            "buy-gate-open"
            if r["mean_buy_allowed"] > merged["mean_buy_allowed"].quantile(0.70)
            else "buy-gate-tight"
            if r["mean_buy_allowed"] < merged["mean_buy_allowed"].quantile(0.35)
            else "moderate-buy-gate"
        )
        trade_label = (
            "active"
            if r["mean_flow_turnover"] > merged["mean_flow_turnover"].quantile(0.70)
            else "quiet"
            if r["mean_flow_turnover"] < merged["mean_flow_turnover"].quantile(0.35)
            else "moderate"
        )
        rows.append(
            {
                "code_id": code,
                "n": int(r["n"]),
                "hierarchy_label": f"{root_label} / {conf_label} / {topk_label} / {trade_label}",
                "root_read": f"cash={r['mean_cash_target']:.3f}, q={r['mean_q_target']:.3f}, anchor_risk={r['root_anchor_risk_rate']:.3f}, anchor_cash={r['root_anchor_cash_rate']:.3f}",
                "confidence_read": f"risk_stress={r['mean_risk_stress']:.3f}, recovery={r['mean_recovery_score']:.3f}, rerisk_conf={r['mean_confidence_rerisk']:.3f}, derisk_conf={r['mean_confidence_derisk']:.3f}",
                "event_read": f"recovery_trigger={r['recovery_trigger_rate']:.3f}, risk_break={r['risk_break_trigger_rate']:.3f}, stop={r['stop_active_rate']:.3f}",
                "topk_read": f"buy_allowed={r['mean_buy_allowed']:.3f}, buy_fill={r['mean_buy_fill_scale']:.3f}, sell_mult={r['mean_sell_multiplier']:.3f}, rotation={r['mean_rotation_requested']:.5f}",
                "dominant_buy_groups": top_buy_groups,
                "dominant_sell_groups": top_sell_groups,
                "dominant_buy_tickers": top_buy_tickers,
                "dominant_sell_tickers": top_sell_tickers,
                "dominant_executed_weights": top_weights,
                "interpretation": (
                    f"Code {code} is {root_label} at root, {conf_label} at confidence layer, "
                    f"and {topk_label} at Top-K execution. Main buy groups: {top_buy_groups}. "
                    f"Main sell groups: {top_sell_groups}. Main buy tickers: {top_buy_tickers}."
                ),
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols: list[str], floatfmt: str = ".4f") -> str:
    view = df[cols]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                vals.append("" if not np.isfinite(val) else format(float(val), floatfmt))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2_DIR)
    args = parser.parse_args()

    valid_path = args.stage2_dir / "stage2_joined_daily_diagnostics_valid.csv"
    valid = pd.read_csv(valid_path)
    valid["code_id"] = valid["code_id"].astype(int)
    tickers = infer_tickers(valid)
    rescorr_groups = infer_rescorr_groups(valid)

    root = root_level(valid)
    conf = confidence_level(valid)
    topk = topk_level(valid)
    groups = group_level(valid, rescorr_groups)
    ticker = ticker_level(valid, tickers)
    interpretation = make_interpretation(root, conf, topk, groups, ticker)

    root.to_csv(args.stage2_dir / "stage2_level_root_cash.csv", index=False)
    conf.to_csv(args.stage2_dir / "stage2_level_confidence_events.csv", index=False)
    topk.to_csv(args.stage2_dir / "stage2_level_topk_execution.csv", index=False)
    groups.to_csv(args.stage2_dir / "stage2_level_rescorr_groups.csv", index=False)
    ticker.to_csv(args.stage2_dir / "stage2_level_ticker_routes.csv", index=False)
    interpretation.to_csv(args.stage2_dir / "stage2_deep_primitive_interpretation.csv", index=False)

    report = f"""# R6c KMeans Deep Hierarchy Interpretation

This report reads the KMeans Stage 1 codebook through the actual R6c hierarchy:

1. root/cash layer
2. confidence and event-trigger layer
3. Top-K/risk-aware execution layer
4. rescorr group-quality layer
5. ticker routing layer

## Primitive Map

{md_table(interpretation, ['code_id', 'n', 'hierarchy_label', 'root_read', 'confidence_read', 'topk_read'], floatfmt='.4f')}

## Group Routing

{md_table(interpretation, ['code_id', 'dominant_buy_groups', 'dominant_sell_groups'], floatfmt='.4f')}

## Ticker Routing

{md_table(interpretation, ['code_id', 'dominant_buy_tickers', 'dominant_sell_tickers', 'dominant_executed_weights'], floatfmt='.4f')}

## Key Reading

- Codes `1` and `2` are the cleanest low-turnover risk-on/baseline regimes.
- Code `7` is the strongest recovery/re-risk primitive: low cash, high recovery score, high buy gate, concentrated in 2020 recovery.
- Codes `3`, `5`, and partly `0` are defensive/stress or cash-heavy regimes.
- Codes `4` and `5` are the active-trading regimes; they differ mainly by root cash: code `4` is risk-on active, code `5` is cash-heavy active.
- Top-K buy routing is not simply the same as executed portfolio weights. The executed book remains diversified, while Top-K logs show which names receive incremental flow.

## Generated Tables

- `stage2_level_root_cash.csv`
- `stage2_level_confidence_events.csv`
- `stage2_level_topk_execution.csv`
- `stage2_level_rescorr_groups.csv`
- `stage2_level_ticker_routes.csv`
- `stage2_deep_primitive_interpretation.csv`
"""
    (args.stage2_dir / "STAGE2_R6C_KMEANS_DEEP_HIERARCHY_INTERPRETATION.md").write_text(report, encoding="utf-8")
    manifest_path = args.stage2_dir / "stage2_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["deep_hierarchy_outputs"] = [
        "stage2_level_root_cash.csv",
        "stage2_level_confidence_events.csv",
        "stage2_level_topk_execution.csv",
        "stage2_level_rescorr_groups.csv",
        "stage2_level_ticker_routes.csv",
        "stage2_deep_primitive_interpretation.csv",
        "STAGE2_R6C_KMEANS_DEEP_HIERARCHY_INTERPRETATION.md",
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Deep hierarchy interpretation written to {args.stage2_dir}")


if __name__ == "__main__":
    main()
