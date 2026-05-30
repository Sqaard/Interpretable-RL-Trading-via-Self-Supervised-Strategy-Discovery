"""Offline replay of Stage 0.1 logged targets through alternate controllers.

The script reads zipped experiment artifacts, reconstructs validation returns
from fold-local model_ready.csv, and replays already-logged raw/anchor/scheduled
portfolio targets without retraining PPO.

It is intentionally deterministic and diagnostic. It does not recompute PPO
log-probabilities and must not be interpreted as a replacement for a true
training run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-10


def normalize_simplex(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    total = float(w.sum())
    if total <= EPS:
        out = np.zeros_like(w)
        out[-1] = 1.0
        return out
    return w / total


def normalize_stock(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    total = float(w.sum())
    if total <= EPS:
        return np.full_like(w, 1.0 / max(w.size, 1), dtype=np.float64)
    return w / total


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, x.size + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        out = np.zeros_like(x)
        out[-1] = 1.0
        return out
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(x - theta, 0.0)


def safe_sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    std = float(np.std(r, ddof=1))
    if std <= EPS:
        return 0.0
    return float(np.sqrt(252.0) * np.mean(r) / std)


def max_drawdown(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0
    peak = np.maximum.accumulate(v)
    return float(np.min(v / np.maximum(peak, EPS) - 1.0))


def weights_after_market(executed: np.ndarray, stock_returns: np.ndarray) -> np.ndarray:
    gross = np.ones_like(executed, dtype=np.float64)
    gross[: stock_returns.size] += stock_returns
    moved = executed * gross
    return normalize_simplex(moved)


def conditional_risky(weights: np.ndarray, stock_dim: int) -> np.ndarray:
    q = float(np.sum(weights[:stock_dim]))
    if q <= EPS:
        return np.full(stock_dim, 1.0 / stock_dim, dtype=np.float64)
    return normalize_stock(weights[:stock_dim] / q)


def priority_order(priority: np.ndarray, tie_breaker: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.asarray(priority, dtype=np.float64)
    t = np.asarray(tie_breaker, dtype=np.float64)
    return np.lexsort((-t, -p))


def allocate_capped_flow(
    total: float,
    selected: np.ndarray,
    preference: np.ndarray,
    capacity: np.ndarray,
    stock_dim: int,
    eps: float = EPS,
) -> tuple[np.ndarray, float]:
    allocation = np.zeros(stock_dim, dtype=np.float64)
    remaining = max(float(total), 0.0)
    selected = np.asarray(selected, dtype=int)
    if remaining <= eps or selected.size == 0:
        return allocation, remaining

    cap_by_stock = np.zeros(stock_dim, dtype=np.float64)
    pref_by_stock = np.zeros(stock_dim, dtype=np.float64)
    cap_by_stock[selected] = np.maximum(np.asarray(capacity, dtype=np.float64), 0.0)
    pref_by_stock[selected] = np.maximum(np.asarray(preference, dtype=np.float64), 0.0)

    for _ in range(selected.size + 1):
        cap_left = np.maximum(cap_by_stock - allocation, 0.0)
        active = selected[cap_left[selected] > eps]
        if remaining <= eps or active.size == 0:
            break
        pref = pref_by_stock[active]
        if float(np.sum(pref)) <= eps:
            pref = cap_left[active]
        pref_sum = float(np.sum(pref))
        if pref_sum <= eps:
            break
        proposed = remaining * pref / pref_sum
        take = np.minimum(proposed, cap_left[active])
        progress = float(np.sum(take))
        if progress <= eps:
            break
        allocation[active] += take
        remaining -= progress
        if np.all(proposed <= cap_left[active] + eps):
            break
    return allocation, max(remaining, 0.0)


@dataclass(frozen=True)
class ReplayVariant:
    name: str
    source: str = "scheduled"
    controller: str = "PD"
    kp: float = 0.35
    kd: float = 0.08
    turnover_cap: float = 0.35
    deadzone_eps: float = 0.0
    top_k_buy: int = 0
    top_k_sell: int = 0
    rotation_budget: float = 0.0
    group_aware: bool = False
    default_group_cap: float = 0.45
    pressure_weight: float = 1.0
    capacity_weight: float = 0.5
    sell_overweight_weight: float = 1.0
    priority_floor: float = 0.05


def apply_controller(prev: np.ndarray, target: np.ndarray, prev_error: np.ndarray, variant: ReplayVariant) -> tuple[np.ndarray, np.ndarray]:
    target = normalize_simplex(target)
    if variant.controller == "none":
        return target, target - prev

    error = target - prev
    if variant.deadzone_eps > 0.0 and float(np.sum(np.abs(error))) < variant.deadzone_eps:
        return prev.copy(), error

    derivative = error - prev_error
    if variant.controller == "P":
        raw = prev + variant.kp * error
    elif variant.controller == "PD":
        raw = prev + variant.kp * error + variant.kd * derivative
    else:
        raise ValueError(f"Unknown controller: {variant.controller}")

    delta = raw - prev
    delta_l1 = float(np.sum(np.abs(delta)))
    if delta_l1 > variant.turnover_cap > 0.0:
        raw = prev + delta * (variant.turnover_cap / delta_l1)
    return normalize_simplex(project_to_simplex(raw)), error


def apply_group_priority(
    buy_priority: np.ndarray,
    sell_priority: np.ndarray,
    *,
    prev: np.ndarray,
    target: np.ndarray,
    groups: dict[str, list[int]],
    variant: ReplayVariant,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    buy = buy_priority.copy()
    sell = sell_priority.copy()
    terms: dict[str, float] = {}
    for group, raw_indices in groups.items():
        idx = np.asarray(raw_indices, dtype=int)
        if idx.size == 0:
            continue
        prev_group = float(np.sum(prev[idx]))
        target_group = float(np.sum(target[idx]))
        cap = max(float(variant.default_group_cap), EPS)
        buy_pressure = max(target_group - prev_group, 0.0)
        sell_pressure = max(prev_group - target_group, 0.0)
        buy_capacity = float(np.clip(1.0 - prev_group / cap, 0.0, 1.0))
        sell_overweight = max(prev_group / cap - 1.0, 0.0)
        buy_mult = variant.priority_floor + variant.pressure_weight * buy_pressure + variant.capacity_weight * buy_capacity
        sell_mult = variant.priority_floor + variant.pressure_weight * sell_pressure + variant.sell_overweight_weight * sell_overweight
        buy[idx] *= float(np.clip(buy_mult, 0.05, 5.0))
        sell[idx] *= float(np.clip(sell_mult, 0.05, 5.0))
        safe = group.replace(" ", "_").replace("/", "_")
        terms[f"group_buy_mult_{safe}"] = float(buy_mult)
        terms[f"group_sell_mult_{safe}"] = float(sell_mult)
    return buy, sell, terms


def apply_incremental_topk(
    prev: np.ndarray,
    target: np.ndarray,
    u_ref: np.ndarray,
    variant: ReplayVariant,
    groups: dict[str, list[int]] | None,
) -> tuple[np.ndarray, dict[str, float]]:
    stock_dim = prev.size - 1
    if variant.top_k_buy <= 0 and variant.top_k_sell <= 0 and variant.rotation_budget <= 0.0:
        return target, {"topk_enabled": 0.0}

    target = normalize_simplex(target)
    prev = normalize_simplex(prev)
    q_prev = float(np.sum(prev[:stock_dim]))
    q_target = float(np.sum(target[:stock_dim]))
    delta_q = q_target - q_prev
    u_anchor = conditional_risky(target, stock_dim)
    u_ref = normalize_stock(u_ref)
    u_current = conditional_risky(prev, stock_dim)
    buy_priority = np.maximum(u_anchor - u_ref, 0.0)
    sell_priority = np.maximum(u_ref - u_anchor, 0.0)
    if variant.group_aware and groups:
        buy_priority, sell_priority, group_terms = apply_group_priority(
            buy_priority,
            sell_priority,
            prev=prev,
            target=target,
            groups=groups,
            variant=variant,
        )
    else:
        group_terms = {}

    buy_order = priority_order(buy_priority, u_anchor)
    sell_order = priority_order(sell_priority, prev[:stock_dim])
    out = prev.copy()
    flow_delta = np.zeros(stock_dim, dtype=np.float64)
    selected_buy = np.zeros(stock_dim, dtype=bool)
    selected_sell = np.zeros(stock_dim, dtype=bool)
    buy_requested = max(delta_q, 0.0)
    sell_requested = max(-delta_q, 0.0)
    buy_filled = sell_filled = buy_unfilled = sell_unfilled = 0.0
    sell_expansion_count = 0

    if buy_requested > EPS and variant.top_k_buy > 0:
        buy_amount = min(buy_requested, float(prev[-1]))
        selected = buy_order[: min(variant.top_k_buy, buy_order.size)]
        selected_buy[selected] = True
        preference = buy_priority[selected]
        if float(np.sum(preference)) <= EPS:
            preference = u_anchor[selected]
        if float(np.sum(preference)) <= EPS:
            preference = np.ones(selected.size, dtype=np.float64)
        alloc = buy_amount * normalize_stock(preference)
        stocks = out[:stock_dim].copy()
        stocks[selected] += alloc
        out[:stock_dim] = stocks
        flow_delta[selected] += alloc
        buy_filled = float(np.sum(alloc))
        buy_unfilled = max(buy_requested - buy_filled, 0.0)

    elif sell_requested > EPS and variant.top_k_sell > 0:
        sell_amount = min(sell_requested, q_prev)
        initial_k = min(variant.top_k_sell, sell_order.size)
        selected_count = initial_k
        selected = sell_order[:selected_count]
        selected_capacity = float(np.sum(prev[:stock_dim][selected]))
        while selected_capacity + EPS < sell_amount and selected_count < sell_order.size:
            selected_count += 1
            selected = sell_order[:selected_count]
            selected_capacity = float(np.sum(prev[:stock_dim][selected]))
        sell_expansion_count = max(0, selected_count - initial_k)
        selected_sell[selected] = True
        preference = sell_priority[selected]
        if float(np.sum(preference)) <= EPS:
            preference = prev[:stock_dim][selected]
        alloc, sell_unfilled = allocate_capped_flow(
            sell_amount,
            selected,
            preference,
            prev[:stock_dim][selected],
            stock_dim,
        )
        out[:stock_dim] = np.maximum(out[:stock_dim] - alloc, 0.0)
        flow_delta -= alloc
        sell_filled = float(np.sum(alloc))

    rotation_requested = rotation_filled = 0.0
    rotation_selected_buy = np.zeros(stock_dim, dtype=bool)
    rotation_selected_sell = np.zeros(stock_dim, dtype=bool)
    if variant.rotation_budget > EPS and delta_q >= -EPS and variant.top_k_buy > 0 and variant.top_k_sell > 0:
        buy_candidates = buy_order[buy_priority[buy_order] > EPS]
        sell_candidates = sell_order[sell_priority[sell_order] > EPS]
        if buy_candidates.size and sell_candidates.size:
            rb = buy_candidates[: min(variant.top_k_buy, buy_candidates.size)]
            rs = sell_candidates[: min(variant.top_k_sell, sell_candidates.size)]
            rotation_selected_buy[rb] = True
            rotation_selected_sell[rs] = True
            sell_capacity = float(np.sum(out[:stock_dim][rs]))
            rotation_requested = min(variant.rotation_budget, sell_capacity)
            pref = sell_priority[rs]
            if float(np.sum(pref)) <= EPS:
                pref = out[:stock_dim][rs]
            sell_alloc, _ = allocate_capped_flow(rotation_requested, rs, pref, out[:stock_dim][rs], stock_dim)
            out[:stock_dim] = np.maximum(out[:stock_dim] - sell_alloc, 0.0)
            flow_delta -= sell_alloc
            rotation_filled = float(np.sum(sell_alloc))
            if rotation_filled > EPS:
                buy_pref = buy_priority[rb]
                if float(np.sum(buy_pref)) <= EPS:
                    buy_pref = u_anchor[rb]
                buy_alloc = rotation_filled * normalize_stock(buy_pref)
                stocks = out[:stock_dim].copy()
                stocks[rb] += buy_alloc
                out[:stock_dim] = stocks
                flow_delta[rb] += buy_alloc

    stock_sum = float(np.sum(out[:stock_dim]))
    if stock_sum > 1.0:
        out[:stock_dim] = normalize_stock(out[:stock_dim])
        stock_sum = 1.0
    out[-1] = max(0.0, 1.0 - stock_sum)
    out = normalize_simplex(out)
    terms = {
        "topk_enabled": 1.0,
        "topk_buy_requested": buy_requested,
        "topk_buy_filled": buy_filled,
        "topk_buy_unfilled": buy_unfilled,
        "topk_sell_requested": sell_requested,
        "topk_sell_filled": sell_filled,
        "topk_sell_unfilled": sell_unfilled,
        "topk_sell_expansion_count": float(sell_expansion_count),
        "topk_selected_buy_count": float(np.sum(selected_buy)),
        "topk_selected_sell_count": float(np.sum(selected_sell)),
        "topk_rotation_requested": rotation_requested,
        "topk_rotation_filled": rotation_filled,
        "topk_rotation_selected_buy_count": float(np.sum(rotation_selected_buy)),
        "topk_rotation_selected_sell_count": float(np.sum(rotation_selected_sell)),
        "topk_flow_l1": float(np.sum(np.abs(flow_delta))),
        "topk_input_to_output_l1": float(np.sum(np.abs(target - out))),
        **group_terms,
    }
    return out, terms


def read_zip_csv(zf: zipfile.ZipFile, suffix: str) -> pd.DataFrame:
    matches = [n for n in zf.namelist() if n.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"Missing {suffix} in {zf.filename}")
    return pd.read_csv(zf.open(matches[0]))


def read_zip_json(zf: zipfile.ZipFile, suffix: str) -> dict[str, Any]:
    matches = [n for n in zf.namelist() if n.endswith(suffix)]
    if not matches:
        return {}
    return json.load(zf.open(matches[0]))


def fold_from_text(text: str) -> str:
    match = re.search(r"fold_(\d{4})", text)
    return f"fold_{match.group(1)}" if match else ""


def columns_to_matrix(df: pd.DataFrame, prefix: str, tickers: list[str]) -> np.ndarray:
    cols = [f"{prefix}_{ticker}" for ticker in tickers] + [f"{prefix}_CASH"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {prefix} columns: {missing[:5]}")
    return df[cols].to_numpy(dtype=np.float64)


def infer_tickers(df: pd.DataFrame) -> list[str]:
    tickers = []
    for col in df.columns:
        if col.startswith("anchor_weight_"):
            ticker = col.replace("anchor_weight_", "")
            if ticker != "CASH":
                tickers.append(ticker)
    if not tickers:
        for col in df.columns:
            if col.startswith("target_weight_"):
                ticker = col.replace("target_weight_", "")
                if ticker != "CASH":
                    tickers.append(ticker)
    return tickers


def returns_from_model_ready(model_ready: pd.DataFrame, tickers: list[str], dates: pd.Series) -> np.ndarray:
    df = model_ready.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    close = df.pivot(index="date", columns="tic", values="close").sort_index()
    close = close.loc[:, tickers]
    ret = close.shift(-1).div(close).sub(1.0)
    out = ret.reindex(pd.to_datetime(dates).dt.strftime("%Y-%m-%d")).to_numpy(dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError("Replay returns contain NaN/inf; check model_ready/date alignment.")
    return out


def groups_from_e9(e9_dir: Path, fold: str, tickers: list[str]) -> dict[str, list[int]]:
    compact_matches = [
        e9_dir / f"cluster_assignments_{fold}.csv",
        e9_dir / fold / "cluster_assignments.csv",
    ]
    for path in compact_matches:
        if not path.exists():
            continue
        assignments = pd.read_csv(path)
        groups: dict[str, list[int]] = {}
        for _, row in assignments.iterrows():
            ticker = str(row["ticker"])
            if ticker not in tickers:
                continue
            group = str(row.get("cluster_name", row.get("cluster_id", "group")))
            groups.setdefault(group, []).append(tickers.index(ticker))
        if groups:
            return groups

    for zp in sorted(e9_dir.glob("*.zip")):
        if fold.replace("fold_", "") not in zp.name and fold not in zp.name:
            # Some E9 package names are generic; still inspect if needed below.
            pass
        with zipfile.ZipFile(zp) as zf:
            matches = [
                n
                for n in zf.namelist()
                if fold in n and n.endswith("discovered_hierarchy/cluster_assignments.csv")
            ]
            if not matches:
                continue
            assignments = pd.read_csv(zf.open(matches[0]))
            groups: dict[str, list[int]] = {}
            for _, row in assignments.iterrows():
                ticker = str(row["ticker"])
                if ticker not in tickers:
                    continue
                group = str(row.get("cluster_name", row.get("cluster_id", "group")))
                groups.setdefault(group, []).append(tickers.index(ticker))
            if groups:
                return groups
    return {"all": list(range(len(tickers)))}


def replay_one(
    daily: pd.DataFrame,
    returns_next: np.ndarray,
    tickers: list[str],
    variant: ReplayVariant,
    *,
    groups: dict[str, list[int]] | None = None,
    transaction_cost_pct: float = 0.001,
) -> tuple[dict[str, Any], pd.DataFrame]:
    source_weights = columns_to_matrix(daily, f"{variant.source}_weight", tickers)
    anchor_weights = columns_to_matrix(daily, "anchor_weight", tickers)
    pre_trade_cols = [f"pre_trade_weight_{t}" for t in tickers] + ["pre_trade_weight_CASH"]
    if all(c in daily.columns for c in pre_trade_cols):
        prev = daily.loc[daily.index[0], pre_trade_cols].to_numpy(dtype=np.float64)
    else:
        prev = np.zeros(len(tickers) + 1, dtype=np.float64)
        prev[-1] = 1.0
    prev = normalize_simplex(prev)
    prev_error = np.zeros_like(prev)
    value = 1_000_000.0
    peak = value
    rows = []
    window_refs: dict[Any, np.ndarray] = {}

    for i, row in daily.reset_index(drop=True).iterrows():
        target = normalize_simplex(source_weights[i])
        if "k_window_start_day" in daily.columns:
            key = row.get("k_window_start_day")
        else:
            key = i
        if key not in window_refs:
            window_refs[key] = conditional_risky(prev, len(tickers))
        u_ref = window_refs[key]

        topk_terms = {"topk_enabled": 0.0}
        if variant.top_k_buy > 0 or variant.top_k_sell > 0 or variant.rotation_budget > 0.0:
            target, topk_terms = apply_incremental_topk(prev, target, u_ref, variant, groups)

        executed, prev_error = apply_controller(prev, target, prev_error, variant)
        stock_delta = executed[: len(tickers)] - prev[: len(tickers)]
        stock_turnover_l1 = float(np.sum(np.abs(stock_delta)))
        turnover_l1 = float(np.sum(np.abs(executed - prev)))
        cost = transaction_cost_pct * stock_turnover_l1
        gross_return = float(np.dot(executed[: len(tickers)], returns_next[i]))
        net_return = (1.0 - cost) * (1.0 + gross_return) - 1.0
        value *= 1.0 + net_return
        peak = max(peak, value)
        drawdown = value / max(peak, EPS) - 1.0
        rows.append(
            {
                "date": row["date"],
                "portfolio_value": value,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover_l1": turnover_l1,
                "stock_turnover_l1": stock_turnover_l1,
                "transaction_cost": cost,
                "drawdown": drawdown,
                "cash": float(executed[-1]),
                "target_cash": float(target[-1]),
                "target_to_executed_l1": float(np.sum(np.abs(target - executed))),
                **topk_terms,
            }
        )
        prev = weights_after_market(executed, returns_next[i])

    replay_daily = pd.DataFrame(rows)
    returns = replay_daily["net_return"].to_numpy(dtype=np.float64)
    summary = {
        "variant": variant.name,
        "days": len(replay_daily),
        "return_pct": float(value / 1_000_000.0 - 1.0),
        "sharpe": safe_sharpe(returns),
        "max_drawdown": max_drawdown(replay_daily["portfolio_value"].to_numpy(dtype=np.float64)),
        "turnover_l1_mean": float(replay_daily["turnover_l1"].mean()),
        "stock_turnover_l1_mean": float(replay_daily["stock_turnover_l1"].mean()),
        "cash_weight_mean": float(replay_daily["cash"].mean()),
        "target_to_executed_l1_mean": float(replay_daily["target_to_executed_l1"].mean()),
    }
    for col in replay_daily.columns:
        if col.startswith("topk_"):
            summary[f"{col}_mean"] = float(pd.to_numeric(replay_daily[col], errors="coerce").mean())
    return summary, replay_daily


def logged_summary(daily: pd.DataFrame) -> dict[str, Any]:
    values = daily["portfolio_value"].to_numpy(dtype=np.float64)
    return {
        "variant": "logged_original",
        "days": len(daily),
        "return_pct": float(values[-1] / values[0] - 1.0) if len(values) else 0.0,
        "sharpe": safe_sharpe(daily["net_return"].to_numpy(dtype=np.float64)),
        "max_drawdown": max_drawdown(values),
        "turnover_l1_mean": float(daily["turnover_l1"].mean()),
        "stock_turnover_l1_mean": float(daily["stock_turnover_l1"].mean()) if "stock_turnover_l1" in daily else np.nan,
        "cash_weight_mean": float(daily["executed_weight_CASH"].mean()) if "executed_weight_CASH" in daily else np.nan,
        "target_to_executed_l1_mean": float(daily["target_to_executed_l1"].mean()) if "target_to_executed_l1" in daily else np.nan,
    }


def default_replay_variants() -> list[ReplayVariant]:
    variants = [
        ReplayVariant("logged_original"),
        ReplayVariant("anchor_no_controller", source="anchor", controller="none"),
        ReplayVariant("scheduled_no_controller", source="scheduled", controller="none"),
        ReplayVariant("anchor_PD", source="anchor", controller="PD"),
        ReplayVariant("scheduled_PD", source="scheduled", controller="PD"),
        ReplayVariant("scheduled_PD_deadzone_002", source="scheduled", controller="PD", deadzone_eps=0.02),
    ]
    for k in [3, 5, 8]:
        variants.append(
            ReplayVariant(
                f"scheduled_PD_top{k}_rot0",
                source="scheduled",
                controller="PD",
                top_k_buy=k,
                top_k_sell=k,
            )
        )
        variants.append(
            ReplayVariant(
                f"scheduled_PD_top{k}_rot0025",
                source="scheduled",
                controller="PD",
                top_k_buy=k,
                top_k_sell=k,
                rotation_budget=0.0025,
            )
        )
        for spec_name, cap, pressure, capacity, overweight in [
            ("pressure", 1.00, 1.00, 0.00, 0.50),
            ("softcap", 0.60, 1.00, 0.25, 0.75),
            ("cap45", 0.45, 1.00, 0.50, 1.00),
            ("pressure2", 1.00, 2.00, 0.00, 0.75),
        ]:
            variants.append(
                ReplayVariant(
                    f"scheduled_PD_groupaware_{spec_name}_top{k}_rot0",
                    source="scheduled",
                    controller="PD",
                    top_k_buy=k,
                    top_k_sell=k,
                    group_aware=True,
                    default_group_cap=cap,
                    pressure_weight=pressure,
                    capacity_weight=capacity,
                    sell_overweight_weight=overweight,
                )
            )
            variants.append(
                ReplayVariant(
                    f"scheduled_PD_groupaware_{spec_name}_top{k}_rot0025",
                    source="scheduled",
                    controller="PD",
                    top_k_buy=k,
                    top_k_sell=k,
                    rotation_budget=0.0025,
                    group_aware=True,
                    default_group_cap=cap,
                    pressure_weight=pressure,
                    capacity_weight=capacity,
                    sell_overweight_weight=overweight,
                )
            )
    return variants


def process_zip(
    zip_path: Path,
    *,
    e9_rescorr_dir: Path,
    output_dir: Path,
    replay_variants: list[ReplayVariant],
) -> list[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as zf:
        daily = read_zip_csv(zf, "validation_daily.csv")
        model_ready = read_zip_csv(zf, "model_ready.csv")
        metadata = read_zip_json(zf, "metadata.json")
    tickers = infer_tickers(daily)
    fold = fold_from_text(str(zip_path)) or fold_from_text(json.dumps(metadata))
    source_experiment = zip_path.parent.name
    returns_next = returns_from_model_ready(model_ready, tickers, daily["date"])
    groups = groups_from_e9(e9_rescorr_dir, fold, tickers)

    rows = []
    logged = logged_summary(daily)
    logged.update({"source_experiment": source_experiment, "fold": fold, "zip": zip_path.name, "group_count": len(groups)})
    rows.append(logged)

    for variant in replay_variants:
        if variant.name == "logged_original":
            continue
        summary, replay_daily = replay_one(daily, returns_next, tickers, variant, groups=groups)
        summary.update({"source_experiment": source_experiment, "fold": fold, "zip": zip_path.name, "group_count": len(groups)})
        rows.append(summary)
        replay_daily.insert(0, "replay_variant", variant.name)
        replay_daily.insert(0, "fold", fold)
        replay_daily.insert(0, "source_experiment", source_experiment)
        daily_out = output_dir / "daily" / source_experiment
        daily_out.mkdir(parents=True, exist_ok=True)
        replay_daily.to_csv(daily_out / f"{fold}_{variant.name}.csv", index=False)
    return rows


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, variant), group in summary.groupby(["source_experiment", "variant"], dropna=False):
        sharpe = group["sharpe"].astype(float)
        rows.append(
            {
                "source_experiment": source,
                "variant": variant,
                "folds": len(group),
                "mean_return_pct": group["return_pct"].astype(float).mean(),
                "mean_sharpe": sharpe.mean(),
                "sample_std_sharpe": sharpe.std(ddof=1),
                "selection_score": sharpe.mean() - 0.5 * sharpe.std(ddof=1),
                "mean_max_drawdown": group["max_drawdown"].astype(float).mean(),
                "mean_turnover_l1": group["turnover_l1_mean"].astype(float).mean(),
                "mean_stock_turnover_l1": group["stock_turnover_l1_mean"].astype(float).mean(),
                "mean_cash": group["cash_weight_mean"].astype(float).mean(),
                "mean_target_to_executed_l1": group["target_to_executed_l1_mean"].astype(float).mean(),
                "mean_topk_flow_l1": group.get("topk_flow_l1_mean", pd.Series(dtype=float)).astype(float).mean()
                if "topk_flow_l1_mean" in group
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["source_experiment", "selection_score"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=[
            "artifacts/stage0_1/R3_root_K20_PD_confidence_slice_residual_stock_v1",
            "artifacts/stage0_1/R6_root_K20_stock_K5_PD_mild_slice_top5_rotation_internaldays_v1",
        ],
    )
    parser.add_argument(
        "--e9-rescorr-dir",
        default="artifacts/stage0_1/e9_rescorr_groups_compact",
    )
    parser.add_argument("--output-dir", default="reports/r_k_window_analysis/offline_replay_groupaware_topk")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_variants = default_replay_variants()
    all_rows: list[dict[str, Any]] = []
    for input_dir in args.input_dirs:
        for zip_path in sorted(Path(input_dir).glob("*.zip")):
            all_rows.extend(
                process_zip(
                    zip_path,
                    e9_rescorr_dir=Path(args.e9_rescorr_dir),
                    output_dir=output_dir,
                    replay_variants=replay_variants,
                )
            )

    summary = pd.DataFrame(all_rows)
    summary.to_csv(output_dir / "offline_replay_fold_summary.csv", index=False)
    agg = aggregate(summary)
    agg.to_csv(output_dir / "offline_replay_aggregate_summary.csv", index=False)

    lines = [
        "# Offline Replay Group-Aware Top-K Summary",
        "",
        "This is a deterministic replay over logged targets, not PPO retraining.",
        "",
        "## Top Variants By Source",
        "",
    ]
    for source, group in agg.groupby("source_experiment"):
        lines.append(f"### {source}")
        lines.append("")
        display = group.head(10)[
            [
                "variant",
                "selection_score",
                "mean_return_pct",
                "mean_sharpe",
                "mean_max_drawdown",
                "mean_cash",
                "mean_turnover_l1",
                "mean_target_to_executed_l1",
            ]
        ]
        try:
            lines.append(display.to_markdown(index=False, floatfmt=".4f"))
        except ImportError:
            lines.append("```")
            lines.append(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            lines.append("```")
        lines.append("")
    (output_dir / "offline_replay_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(agg.head(30).to_string(index=False))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
