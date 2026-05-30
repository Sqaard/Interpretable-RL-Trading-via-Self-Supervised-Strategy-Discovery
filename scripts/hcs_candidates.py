"""Candidate construction and mutation helpers for policy-forward HCS."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from counterfactual_policy_forward_replay import CounterfactualCandidate


def candidate_name(
    *,
    top_k_buy: int | None,
    top_k_sell: int | None,
    rotation_budget: float,
    group_aware: bool,
    risk_aware_topk: bool,
    default_group_cap: float,
    pressure_weight: float,
    capacity_weight: float,
    sell_overweight_weight: float,
    priority_floor: float,
) -> str:
    if top_k_buy is None and top_k_sell is None:
        return "pf_no_incremental_topk"
    prefix = "pf_riskga" if risk_aware_topk and group_aware else ("pf_risktopk" if risk_aware_topk else ("pf_ga" if group_aware else "pf_topk"))
    rot = str(rotation_budget).replace(".", "p")
    cap = str(default_group_cap).replace(".", "p")
    pressure = str(pressure_weight).replace(".", "p")
    capacity = str(capacity_weight).replace(".", "p")
    sell_ow = str(sell_overweight_weight).replace(".", "p")
    floor = str(priority_floor).replace(".", "p")
    if group_aware:
        return (
            f"{prefix}_b{top_k_buy}_s{top_k_sell}_rot{rot}"
            f"_cap{cap}_pw{pressure}_cw{capacity}_sw{sell_ow}_pf{floor}"
        )
    return f"{prefix}_b{top_k_buy}_s{top_k_sell}_rot{rot}"


def make_candidate(
    *,
    top_k_buy: int | None = None,
    top_k_sell: int | None = None,
    rotation_budget: float = 0.0,
    group_aware: bool = False,
    risk_aware_topk: bool = False,
    default_group_cap: float = 0.45,
    pressure_weight: float = 1.0,
    capacity_weight: float = 0.5,
    sell_overweight_weight: float = 1.0,
    priority_floor: float = 0.05,
    k_root_days: int | None = None,
    k_stock_days: int | None = None,
    recovery_trigger_threshold: float | None = None,
    derisk_early_update_threshold: float | None = None,
    recovery_min_confidence_rerisk: float | None = None,
    risk_break_min_confidence_derisk: float | None = None,
    recovery_max_risk_stress: float | None = None,
    early_update_cooldown_days: int | None = None,
    recovery_persistence_days: int | None = None,
    risk_break_persistence_days: int | None = None,
    rerisk_min_scale: float | None = None,
    derisk_min_scale: float | None = None,
    risk_stress_scale: float = 1.0,
    recovery_score_scale: float = 1.0,
    recovery_residual_scale: float = 1.0,
    recovery_market_scale: float = 1.0,
    derisk_market_down_scale: float = 1.0,
    vix_shock_scale: float = 1.0,
    feature_family: str = "",
    buy_gate_min_confidence_rerisk: float | None = None,
    buy_gate_min_recovery_score: float | None = None,
    buy_gate_max_risk_stress: float | None = None,
    buy_gate_min_residual_breadth_excess_5d: float | None = None,
    buy_gate_min_residual_breadth_excess_20d: float | None = None,
    rotation_stress_gate_enabled: bool = False,
    rotation_stress_start: float | None = None,
    rotation_stress_full: float | None = None,
    rotation_stress_min_scale: float | None = None,
    sell_risk_break_weight: float = 0.0,
    sell_residual_deterioration_weight: float = 0.0,
    sell_confidence_derisk_weight: float = 0.0,
    label_suffix: str = "",
) -> CounterfactualCandidate:
    name = candidate_name(
        top_k_buy=top_k_buy,
        top_k_sell=top_k_sell,
        rotation_budget=rotation_budget,
        group_aware=group_aware,
        risk_aware_topk=risk_aware_topk,
        default_group_cap=default_group_cap,
        pressure_weight=pressure_weight,
        capacity_weight=capacity_weight,
        sell_overweight_weight=sell_overweight_weight,
        priority_floor=priority_floor,
    )
    suffix_parts: list[str] = []
    if label_suffix:
        suffix_parts.append(label_suffix)
    if k_root_days is not None or k_stock_days is not None:
        suffix_parts.append(f"Kroot{k_root_days or 'base'}_Kstock{k_stock_days or 'base'}")
    if recovery_trigger_threshold is not None:
        suffix_parts.append(f"recThr{str(recovery_trigger_threshold).replace('.', 'p')}")
    if recovery_min_confidence_rerisk is not None:
        suffix_parts.append(f"recConf{str(recovery_min_confidence_rerisk).replace('.', 'p')}")
    if derisk_early_update_threshold is not None:
        suffix_parts.append(f"riskBreakThr{str(derisk_early_update_threshold).replace('.', 'p')}")
    if risk_break_min_confidence_derisk is not None:
        suffix_parts.append(f"riskBreakConf{str(risk_break_min_confidence_derisk).replace('.', 'p')}")
    if recovery_persistence_days is not None:
        suffix_parts.append(f"recPersist{recovery_persistence_days}")
    if risk_break_persistence_days is not None:
        suffix_parts.append(f"riskBreakPersist{risk_break_persistence_days}")
    if abs(recovery_residual_scale - 1.0) > 1e-12:
        suffix_parts.append(f"recResid{str(recovery_residual_scale).replace('.', 'p')}")
    if abs(recovery_market_scale - 1.0) > 1e-12:
        suffix_parts.append(f"recMarket{str(recovery_market_scale).replace('.', 'p')}")
    if abs(vix_shock_scale - 1.0) > 1e-12:
        suffix_parts.append(f"vixShock{str(vix_shock_scale).replace('.', 'p')}")
    if abs(derisk_market_down_scale - 1.0) > 1e-12:
        suffix_parts.append(f"deriskMarket{str(derisk_market_down_scale).replace('.', 'p')}")
    if rerisk_min_scale is not None:
        suffix_parts.append(f"reriskMin{str(rerisk_min_scale).replace('.', 'p')}")
    if derisk_min_scale is not None:
        suffix_parts.append(f"deriskMin{str(derisk_min_scale).replace('.', 'p')}")
    if risk_aware_topk:
        suffix_parts.append("riskaware")
        if buy_gate_min_confidence_rerisk is not None:
            suffix_parts.append(f"buyConf{str(buy_gate_min_confidence_rerisk).replace('.', 'p')}")
        if buy_gate_min_recovery_score is not None:
            suffix_parts.append(f"buyRec{str(buy_gate_min_recovery_score).replace('.', 'p')}")
        if buy_gate_max_risk_stress is not None:
            suffix_parts.append(f"buyStressMax{str(buy_gate_max_risk_stress).replace('.', 'p')}")
        if sell_risk_break_weight:
            suffix_parts.append(f"sellRB{str(sell_risk_break_weight).replace('.', 'p')}")
        if sell_residual_deterioration_weight:
            suffix_parts.append(f"sellResidBad{str(sell_residual_deterioration_weight).replace('.', 'p')}")
        if rotation_stress_gate_enabled:
            suffix_parts.append("rotStressGate")
    if suffix_parts:
        name = f"{name}__{'__'.join(str(part) for part in suffix_parts)}"
    return CounterfactualCandidate(
        name=name,
        top_k_buy=top_k_buy,
        top_k_sell=top_k_sell,
        rotation_budget=rotation_budget,
        group_aware=group_aware,
        risk_aware_topk=risk_aware_topk,
        default_group_cap=default_group_cap,
        pressure_weight=pressure_weight,
        capacity_weight=capacity_weight,
        sell_overweight_weight=sell_overweight_weight,
        priority_floor=priority_floor,
        disable_incremental_topk=(top_k_buy is None and top_k_sell is None),
        k_root_days=k_root_days,
        k_stock_days=k_stock_days,
        recovery_trigger_threshold=recovery_trigger_threshold,
        derisk_early_update_threshold=derisk_early_update_threshold,
        recovery_min_confidence_rerisk=recovery_min_confidence_rerisk,
        risk_break_min_confidence_derisk=risk_break_min_confidence_derisk,
        recovery_max_risk_stress=recovery_max_risk_stress,
        early_update_cooldown_days=early_update_cooldown_days,
        recovery_persistence_days=recovery_persistence_days,
        risk_break_persistence_days=risk_break_persistence_days,
        rerisk_min_scale=rerisk_min_scale,
        derisk_min_scale=derisk_min_scale,
        risk_stress_scale=risk_stress_scale,
        recovery_score_scale=recovery_score_scale,
        recovery_residual_scale=recovery_residual_scale,
        recovery_market_scale=recovery_market_scale,
        derisk_market_down_scale=derisk_market_down_scale,
        vix_shock_scale=vix_shock_scale,
        feature_family=feature_family,
        buy_gate_min_confidence_rerisk=buy_gate_min_confidence_rerisk,
        buy_gate_min_recovery_score=buy_gate_min_recovery_score,
        buy_gate_max_risk_stress=buy_gate_max_risk_stress,
        buy_gate_min_residual_breadth_excess_5d=buy_gate_min_residual_breadth_excess_5d,
        buy_gate_min_residual_breadth_excess_20d=buy_gate_min_residual_breadth_excess_20d,
        rotation_stress_gate_enabled=rotation_stress_gate_enabled,
        rotation_stress_start=rotation_stress_start,
        rotation_stress_full=rotation_stress_full,
        rotation_stress_min_scale=rotation_stress_min_scale,
        sell_risk_break_weight=sell_risk_break_weight,
        sell_residual_deterioration_weight=sell_residual_deterioration_weight,
        sell_confidence_derisk_weight=sell_confidence_derisk_weight,
    )


def seed_candidates() -> list[CounterfactualCandidate]:
    seeds = [
        CounterfactualCandidate("pf_original"),
        make_candidate(),
        make_candidate(top_k_buy=8, top_k_sell=8),
        make_candidate(top_k_buy=10, top_k_sell=10),
        make_candidate(top_k_buy=8, top_k_sell=8, group_aware=True, default_group_cap=0.45),
        make_candidate(
            top_k_buy=10,
            top_k_sell=12,
            rotation_budget=0.005,
            group_aware=True,
            default_group_cap=0.60,
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
            label_suffix="riskaware_balanced",
        ),
        make_candidate(
            top_k_buy=8,
            top_k_sell=12,
            rotation_budget=0.0025,
            group_aware=True,
            default_group_cap=0.60,
            capacity_weight=0.25,
            sell_overweight_weight=1.25,
            risk_aware_topk=True,
            buy_gate_min_confidence_rerisk=0.50,
            buy_gate_min_recovery_score=0.55,
            buy_gate_max_risk_stress=0.90,
            buy_gate_min_residual_breadth_excess_5d=-0.02,
            rotation_stress_gate_enabled=True,
            rotation_stress_start=0.55,
            rotation_stress_full=0.90,
            rotation_stress_min_scale=0.0,
            sell_risk_break_weight=1.0,
            sell_residual_deterioration_weight=0.75,
            sell_confidence_derisk_weight=0.25,
            label_suffix="riskaware_softbuy",
        ),
        make_candidate(
            top_k_buy=8,
            top_k_sell=8,
            recovery_residual_scale=1.25,
            recovery_market_scale=0.50,
            rerisk_min_scale=0.65,
            label_suffix="conf_resid_rerisk125_market050",
        ),
        make_candidate(
            top_k_buy=8,
            top_k_sell=8,
            vix_shock_scale=1.25,
            derisk_market_down_scale=1.25,
            derisk_min_scale=0.30,
            label_suffix="conf_derisk_shock125",
        ),
        make_candidate(
            top_k_buy=8,
            top_k_sell=8,
            recovery_trigger_threshold=0.65,
            recovery_min_confidence_rerisk=0.50,
            recovery_persistence_days=1,
            label_suffix="trig_soft_recovery",
        ),
        make_candidate(
            top_k_buy=8,
            top_k_sell=8,
            recovery_trigger_threshold=0.78,
            recovery_min_confidence_rerisk=0.62,
            recovery_persistence_days=2,
            label_suffix="trig_strict_recovery",
        ),
        make_candidate(
            top_k_buy=8,
            top_k_sell=8,
            derisk_early_update_threshold=0.80,
            risk_break_min_confidence_derisk=0.70,
            risk_break_persistence_days=1,
            label_suffix="trig_fast_riskbreak",
        ),
        make_candidate(top_k_buy=8, top_k_sell=8, k_root_days=10, k_stock_days=5, label_suffix="Kroot10_Kstock5"),
        make_candidate(top_k_buy=8, top_k_sell=8, k_root_days=20, k_stock_days=10, label_suffix="Kroot20_Kstock10"),
    ]
    return unique_candidates(seeds)


def unique_candidates(candidates: list[CounterfactualCandidate]) -> list[CounterfactualCandidate]:
    seen: set[str] = set()
    out: list[CounterfactualCandidate] = []
    for candidate in candidates:
        if candidate.name in seen:
            continue
        seen.add(candidate.name)
        out.append(candidate)
    return out


def clip_int(value: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, value)))


def clip_float(value: float, lo: float, hi: float, digits: int = 4) -> float:
    return round(float(max(lo, min(hi, value))), digits)


def mutate_candidate(parent: CounterfactualCandidate, *, max_mutations: int) -> list[CounterfactualCandidate]:
    if parent.name == "pf_original":
        return []

    base_buy = int(parent.top_k_buy or 8)
    base_sell = int(parent.top_k_sell or 8)
    base_rotation = float(parent.rotation_budget)
    base_group = bool(parent.group_aware)
    mutations: list[CounterfactualCandidate] = []

    def child(label_suffix: str = "", **overrides: Any) -> CounterfactualCandidate:
        params: dict[str, Any] = {
            "top_k_buy": base_buy,
            "top_k_sell": base_sell,
            "rotation_budget": base_rotation,
            "group_aware": base_group,
            "risk_aware_topk": parent.risk_aware_topk,
            "default_group_cap": parent.default_group_cap,
            "pressure_weight": parent.pressure_weight,
            "capacity_weight": parent.capacity_weight,
            "sell_overweight_weight": parent.sell_overweight_weight,
            "priority_floor": parent.priority_floor,
            "k_root_days": parent.k_root_days,
            "k_stock_days": parent.k_stock_days,
            "recovery_trigger_threshold": parent.recovery_trigger_threshold,
            "derisk_early_update_threshold": parent.derisk_early_update_threshold,
            "recovery_min_confidence_rerisk": parent.recovery_min_confidence_rerisk,
            "risk_break_min_confidence_derisk": parent.risk_break_min_confidence_derisk,
            "recovery_max_risk_stress": parent.recovery_max_risk_stress,
            "early_update_cooldown_days": parent.early_update_cooldown_days,
            "recovery_persistence_days": parent.recovery_persistence_days,
            "risk_break_persistence_days": parent.risk_break_persistence_days,
            "rerisk_min_scale": parent.rerisk_min_scale,
            "derisk_min_scale": parent.derisk_min_scale,
            "risk_stress_scale": parent.risk_stress_scale,
            "recovery_score_scale": parent.recovery_score_scale,
            "recovery_residual_scale": parent.recovery_residual_scale,
            "recovery_market_scale": parent.recovery_market_scale,
            "derisk_market_down_scale": parent.derisk_market_down_scale,
            "vix_shock_scale": parent.vix_shock_scale,
            "feature_family": parent.feature_family,
            "buy_gate_min_confidence_rerisk": parent.buy_gate_min_confidence_rerisk,
            "buy_gate_min_recovery_score": parent.buy_gate_min_recovery_score,
            "buy_gate_max_risk_stress": parent.buy_gate_max_risk_stress,
            "buy_gate_min_residual_breadth_excess_5d": parent.buy_gate_min_residual_breadth_excess_5d,
            "buy_gate_min_residual_breadth_excess_20d": parent.buy_gate_min_residual_breadth_excess_20d,
            "rotation_stress_gate_enabled": parent.rotation_stress_gate_enabled,
            "rotation_stress_start": parent.rotation_stress_start,
            "rotation_stress_full": parent.rotation_stress_full,
            "rotation_stress_min_scale": parent.rotation_stress_min_scale,
            "sell_risk_break_weight": parent.sell_risk_break_weight,
            "sell_residual_deterioration_weight": parent.sell_residual_deterioration_weight,
            "sell_confidence_derisk_weight": parent.sell_confidence_derisk_weight,
            "label_suffix": label_suffix,
        }
        params.update(overrides)
        return make_candidate(**params)

    # Local Top-K neighborhood.
    for db, ds in [(-2, 0), (2, 0), (0, -2), (0, 2), (2, 2), (-2, 2)]:
        mutations.append(child(top_k_buy=clip_int(base_buy + db, 3, 12), top_k_sell=clip_int(base_sell + ds, 3, 12)))

    # Rotation neighborhood.
    for rotation in [0.0, 0.0005, 0.001, 0.0025]:
        if abs(rotation - base_rotation) <= 1e-12:
            continue
        mutations.append(child(rotation_budget=rotation))

    # Promote/demote group-aware logic.
    mutations.append(child(group_aware=not base_group))
    if not parent.risk_aware_topk:
        mutations.append(
            child(
                label_suffix="riskaware_on",
                group_aware=True,
                risk_aware_topk=True,
                top_k_buy=max(base_buy, 8),
                top_k_sell=max(base_sell, 10),
                rotation_budget=max(base_rotation, 0.0025),
                default_group_cap=max(parent.default_group_cap, 0.60),
                capacity_weight=min(parent.capacity_weight, 0.25),
                sell_overweight_weight=max(parent.sell_overweight_weight, 1.25),
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
            )
        )
    else:
        buy_conf = parent.buy_gate_min_confidence_rerisk if parent.buy_gate_min_confidence_rerisk is not None else 0.55
        buy_rec = parent.buy_gate_min_recovery_score if parent.buy_gate_min_recovery_score is not None else 0.60
        buy_stress = parent.buy_gate_max_risk_stress if parent.buy_gate_max_risk_stress is not None else 0.85
        breadth_5d = (
            parent.buy_gate_min_residual_breadth_excess_5d
            if parent.buy_gate_min_residual_breadth_excess_5d is not None
            else 0.0
        )
        for delta in [-0.05, 0.05]:
            mutations.append(
                child(
                    label_suffix=f"riskaware_buyconf{str(clip_float(buy_conf + delta, 0.35, 0.8)).replace('.', 'p')}",
                    risk_aware_topk=True,
                    buy_gate_min_confidence_rerisk=clip_float(buy_conf + delta, 0.35, 0.8),
                )
            )
            mutations.append(
                child(
                    label_suffix=f"riskaware_buyrec{str(clip_float(buy_rec + delta, 0.35, 0.85)).replace('.', 'p')}",
                    risk_aware_topk=True,
                    buy_gate_min_recovery_score=clip_float(buy_rec + delta, 0.35, 0.85),
                )
            )
        for stress_delta in [-0.05, 0.05]:
            mutations.append(
                child(
                    label_suffix=f"riskaware_stress{str(clip_float(buy_stress + stress_delta, 0.65, 0.98)).replace('.', 'p')}",
                    risk_aware_topk=True,
                    buy_gate_max_risk_stress=clip_float(buy_stress + stress_delta, 0.65, 0.98),
                )
            )
        for breadth in [breadth_5d - 0.02, breadth_5d + 0.02]:
            mutations.append(
                child(
                    label_suffix=f"riskaware_breadth{str(clip_float(breadth, -0.10, 0.10)).replace('.', 'p')}",
                    risk_aware_topk=True,
                    buy_gate_min_residual_breadth_excess_5d=clip_float(breadth, -0.10, 0.10),
                )
            )
        for rb_weight in [parent.sell_risk_break_weight - 0.5, parent.sell_risk_break_weight + 0.5]:
            mutations.append(
                child(
                    label_suffix=f"riskaware_sellrb{str(clip_float(rb_weight, 0.0, 3.0)).replace('.', 'p')}",
                    risk_aware_topk=True,
                    sell_risk_break_weight=clip_float(rb_weight, 0.0, 3.0),
                )
            )
        for resid_weight in [
            parent.sell_residual_deterioration_weight - 0.25,
            parent.sell_residual_deterioration_weight + 0.25,
        ]:
            mutations.append(
                child(
                    label_suffix=f"riskaware_sellresid{str(clip_float(resid_weight, 0.0, 3.0)).replace('.', 'p')}",
                    risk_aware_topk=True,
                    sell_residual_deterioration_weight=clip_float(resid_weight, 0.0, 3.0),
                )
            )

    if base_group:
        for cap in [parent.default_group_cap - 0.10, parent.default_group_cap + 0.10]:
            mutations.append(child(group_aware=True, default_group_cap=clip_float(cap, 0.30, 0.80)))
        for pressure in [parent.pressure_weight - 0.5, parent.pressure_weight + 0.5]:
            mutations.append(child(group_aware=True, pressure_weight=clip_float(pressure, 0.0, 3.0)))
        for capacity in [parent.capacity_weight - 0.25, parent.capacity_weight + 0.25]:
            mutations.append(child(group_aware=True, capacity_weight=clip_float(capacity, 0.0, 1.5)))
        for sell_ow in [parent.sell_overweight_weight - 0.25, parent.sell_overweight_weight + 0.25]:
            mutations.append(child(group_aware=True, sell_overweight_weight=clip_float(sell_ow, 0.0, 2.5)))

    # Confidence-coefficient neighborhood.
    for residual_scale in [parent.recovery_residual_scale - 0.20, parent.recovery_residual_scale + 0.20]:
        mutations.append(
            child(
                label_suffix=f"conf_resid{str(clip_float(residual_scale, 0.5, 2.0)).replace('.', 'p')}",
                recovery_residual_scale=clip_float(residual_scale, 0.5, 2.0),
            )
        )
    for market_scale in [parent.recovery_market_scale - 0.20, parent.recovery_market_scale + 0.20]:
        mutations.append(
            child(
                label_suffix=f"conf_market{str(clip_float(market_scale, 0.0, 1.5)).replace('.', 'p')}",
                recovery_market_scale=clip_float(market_scale, 0.0, 1.5),
            )
        )
    for vix_scale in [parent.vix_shock_scale - 0.20, parent.vix_shock_scale + 0.20]:
        mutations.append(
            child(
                label_suffix=f"conf_vix{str(clip_float(vix_scale, 0.5, 2.0)).replace('.', 'p')}",
                vix_shock_scale=clip_float(vix_scale, 0.5, 2.0),
            )
        )

    # Trigger-threshold neighborhood.
    recovery_thr = parent.recovery_trigger_threshold if parent.recovery_trigger_threshold is not None else 0.70
    recovery_conf = parent.recovery_min_confidence_rerisk if parent.recovery_min_confidence_rerisk is not None else 0.55
    risk_break_thr = parent.derisk_early_update_threshold if parent.derisk_early_update_threshold is not None else 0.85
    risk_break_conf = parent.risk_break_min_confidence_derisk if parent.risk_break_min_confidence_derisk is not None else 0.75
    for delta in [-0.05, 0.05]:
        mutations.append(
            child(
                label_suffix=f"trig_rec{str(clip_float(recovery_thr + delta, 0.45, 0.9)).replace('.', 'p')}",
                recovery_trigger_threshold=clip_float(recovery_thr + delta, 0.45, 0.9),
                recovery_min_confidence_rerisk=clip_float(recovery_conf + 0.5 * delta, 0.35, 0.8),
            )
        )
        mutations.append(
            child(
                label_suffix=f"trig_rb{str(clip_float(risk_break_thr + delta, 0.65, 0.95)).replace('.', 'p')}",
                derisk_early_update_threshold=clip_float(risk_break_thr + delta, 0.65, 0.95),
                risk_break_min_confidence_derisk=clip_float(risk_break_conf + 0.5 * delta, 0.50, 0.9),
            )
        )

    # Timing neighborhood. Root and stock windows are replay-safe; feature
    # family changes are retrain-only and are registered elsewhere.
    root_days = int(parent.k_root_days or 20)
    stock_days = int(parent.k_stock_days or 5)
    for root_delta, stock_delta in [(-5, 0), (5, 0), (0, -2), (0, 2)]:
        mutations.append(
            child(
                label_suffix=f"Kroot{clip_int(root_days + root_delta, 5, 30)}_Kstock{clip_int(stock_days + stock_delta, 3, 15)}",
                k_root_days=clip_int(root_days + root_delta, 5, 30),
                k_stock_days=clip_int(stock_days + stock_delta, 3, 15),
            )
        )

    return unique_candidates(mutations)[:max_mutations]



def candidate_family(candidate: CounterfactualCandidate) -> str:
    if candidate.feature_family:
        return "retrain_only_feature_family"
    if candidate.name == "pf_original":
        return "original"
    if candidate.disable_incremental_topk:
        return "no_incremental_topk"
    if candidate.risk_aware_topk:
        return "risk_aware_topk"
    if candidate.k_root_days is not None or candidate.k_stock_days is not None:
        return "timing_k_window"
    confidence_changed = any(
        abs(float(value) - 1.0) > 1e-12
        for value in [
            candidate.risk_stress_scale,
            candidate.recovery_score_scale,
            candidate.recovery_residual_scale,
            candidate.recovery_market_scale,
            candidate.derisk_market_down_scale,
            candidate.vix_shock_scale,
        ]
    )
    if confidence_changed or candidate.rerisk_min_scale is not None or candidate.derisk_min_scale is not None:
        return "confidence_coeffs"
    trigger_changed = any(
        value is not None
        for value in [
            candidate.recovery_trigger_threshold,
            candidate.derisk_early_update_threshold,
            candidate.recovery_min_confidence_rerisk,
            candidate.risk_break_min_confidence_derisk,
            candidate.recovery_max_risk_stress,
            candidate.early_update_cooldown_days,
            candidate.recovery_persistence_days,
            candidate.risk_break_persistence_days,
        ]
    )
    if trigger_changed:
        return "trigger_thresholds"
    if candidate.group_aware:
        return "group_aware_topk"
    return "global_topk"


def retrain_only_feature_candidates() -> pd.DataFrame:
    rows = [
        {
            "candidate": "R3_compact22_feature_ablation_v1",
            "family": "retrain_only_feature_family",
            "feature_family": "compact22_stage0_like",
            "replay_safe": False,
            "reason": "Changes observation space; trained policy weights are not compatible with counterfactual replay.",
        },
        {
            "candidate": "R3_delta_change_only_features_v1",
            "family": "retrain_only_feature_family",
            "feature_family": "delta_change_only",
            "replay_safe": False,
            "reason": "Tests the user's delta/change-only hypothesis; requires feature builder and PPO retrain.",
        },
        {
            "candidate": "R3_residualized_delta_features_v1",
            "family": "retrain_only_feature_family",
            "feature_family": "residualized_delta_change",
            "replay_safe": False,
            "reason": "Would remove market drift from feature inputs; must be trained as a new policy.",
        },
    ]
    return pd.DataFrame(rows)


