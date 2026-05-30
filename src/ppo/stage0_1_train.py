"""Train Stage 0.1 stabilized weight-based PPO teachers."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch as th
import yaml
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.data.dow30_sectors import get_sector_map
from src.data.stage0_1_normalization import prepare_fold_scaled_features
from src.ppo.dirichlet_policy import (
    DirichletActorCriticPolicy,
    HierarchicalDirichletActorCriticPolicy,
    RiskCashGroupLogisticNormalTreeActorCriticPolicy,
    RiskCashSectorDirichletTreeActorCriticPolicy,
    RootSplitBetaDirichletActorCriticPolicy,
    RootSplitBetaDirichletKpActorCriticPolicy,
    RoutedRootSplitBetaDirichletActorCriticPolicy,
)
from src.ppo.instrumented_ppo import InstrumentedPPO
from src.ppo.stage0_1_weight_env import WeightPanel, load_weight_panel, make_env_from_config


ROOT = Path(__file__).resolve().parents[2]


class InternalTradingDaysStopCallback(BaseCallback):
    """Stop training after a fixed number of internal trading days.

    K-window envs expose one SB3 timestep as a macro-step that may contain
    several daily market transitions. This callback makes the training budget
    comparable across daily, K=20, and event-triggered variants.
    """

    def __init__(self, target_internal_days: float, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.target_internal_days = float(target_internal_days)
        self.internal_days_seen = 0.0
        self.stop_reached = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            raw_days = info.get("internal_trading_days_this_step", info.get("k_window_effective_days", 1.0))
            try:
                days = float(raw_days)
            except (TypeError, ValueError):
                days = 1.0
            if not np.isfinite(days) or days < 0:
                days = 0.0
            self.internal_days_seen += days
        self.logger.record("time/internal_trading_days", self.internal_days_seen)
        self.logger.record("time/target_internal_trading_days", self.target_internal_days)
        if self.internal_days_seen >= self.target_internal_days:
            self.stop_reached = True
            if self.verbose:
                print(
                    "Stopping training after internal trading-day budget: "
                    f"{self.internal_days_seen:.0f}/{self.target_internal_days:.0f}"
                )
            return False
        return True


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def activation_from_name(name: str):
    mapping = {
        "Tanh": th.nn.Tanh,
        "ReLU": th.nn.ReLU,
        "GELU": th.nn.GELU,
        "SiLU": th.nn.SiLU,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation: {name}")
    return mapping[name]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def make_vec_env(panel: WeightPanel, config: dict[str, Any], variant: dict[str, Any]):
    def _factory():
        env = make_env_from_config(panel, config, variant)
        return Monitor(env)

    return DummyVecEnv([_factory])


def rollout_dates_for_variant(panel: WeightPanel, variant: dict[str, Any]) -> list[str]:
    dates = list(panel.dates)
    k_window_cfg = variant.get("root_split", {}).get("k_window_execution", {})
    if k_window_cfg and bool(k_window_cfg.get("enabled", False)):
        confidence_cfg = k_window_cfg.get("confidence_stop_recovery", {})
        if confidence_cfg and bool(confidence_cfg.get("enabled", False)):
            # Event-triggered K-window variants have variable macro-step lengths.
            # Static rollout date reconstruction would be misleading, so leave
            # PPO sample diagnostics undated and rely on validation_daily.csv.
            return []
        window_days = max(1, int(k_window_cfg.get("window_days", k_window_cfg.get("K", 1))))
        decision_indices = list(range(0, max(len(dates) - 1, 1), window_days))
        if not decision_indices or decision_indices[0] != 0:
            decision_indices.insert(0, 0)
        last_idx = len(dates) - 1
        if decision_indices[-1] != last_idx:
            decision_indices.append(last_idx)
        return [str(pd.Timestamp(dates[idx]).date()) for idx in decision_indices]
    return [str(pd.Timestamp(date).date()) for date in dates]


def hierarchical_group_indices(config: dict[str, Any], panel: WeightPanel) -> list[list[int]]:
    """Return cash + sector groups as leaf indices for final action weights."""
    sector_map = get_sector_map(config.get("universe", {}).get("sector_map", "dow30_static"))
    ticker_sectors = [sector_map[ticker] for ticker in panel.tickers]
    group_indices: list[list[int]] = [[len(panel.tickers)]]
    for sector in sorted(set(ticker_sectors)):
        group_indices.append([idx for idx, ticker_sector in enumerate(ticker_sectors) if ticker_sector == sector])
    return group_indices


def sector_stock_group_indices(
    config: dict[str, Any],
    panel: WeightPanel,
    variant: dict[str, Any] | None = None,
) -> list[list[int]]:
    """Return sector groups over stock indices only, matching environment group order."""
    custom = (variant or {}).get("root_split", {}).get("group_indices")
    if custom:
        return [list(map(int, group)) for group in custom]
    sector_map = get_sector_map(config.get("universe", {}).get("sector_map", "dow30_static"))
    ticker_sectors = [sector_map[ticker] for ticker in panel.tickers]
    return [
        [idx for idx, ticker_sector in enumerate(ticker_sectors) if ticker_sector == sector]
        for sector in sorted(set(ticker_sectors))
    ]


def policy_for_variant(config: dict[str, Any], variant: dict[str, Any], panel: WeightPanel):
    base_policy_cfg = config["policy"]
    variant_policy_cfg = variant.get("policy", {})
    policy_cfg = {**base_policy_cfg, **variant_policy_cfg}
    policy_kwargs: dict[str, Any] = {
        "net_arch": policy_cfg.get("net_arch", {"pi": [128, 64], "vf": [256, 128]}),
        "activation_fn": activation_from_name(policy_cfg.get("activation", "ReLU")),
        "ortho_init": bool(policy_cfg.get("ortho_init", True)),
    }

    if variant["policy_kind"] in {"flat_dirichlet", "style_mixture_dirichlet"}:
        dirichlet_cfg = {**base_policy_cfg.get("dirichlet", {}), **variant_policy_cfg.get("dirichlet", {})}
        policy_kwargs.update(
            {
                "alpha_min": float(dirichlet_cfg.get("alpha_min", 1.0)),
                "alpha_max": float(dirichlet_cfg.get("alpha_max", 80.0)),
            }
        )
        return DirichletActorCriticPolicy, policy_kwargs

    if variant["policy_kind"] == "hierarchical_dirichlet":
        dirichlet_cfg = {**base_policy_cfg.get("dirichlet", {}), **variant_policy_cfg.get("dirichlet", {})}
        policy_kwargs.update(
            {
                "group_indices": hierarchical_group_indices(config, panel),
                "alpha_min": float(dirichlet_cfg.get("alpha_min", 1.0)),
                "alpha_max": float(dirichlet_cfg.get("alpha_max", 80.0)),
            }
        )
        return HierarchicalDirichletActorCriticPolicy, policy_kwargs

    if variant["policy_kind"] in {
        "root_split_beta_dirichlet",
        "root_split_beta_dirichlet_routed",
        "root_split_beta_dirichlet_learned_kp",
    }:
        root_cfg = {
            **base_policy_cfg.get("root_split", {}),
            **variant.get("root_split", {}),
            **variant_policy_cfg.get("root_split", {}),
        }
        policy_kwargs.update(
            {
                "stock_dim": len(panel.tickers),
                "q_min": float(root_cfg.get("q_min", 0.00)),
                "q_max": float(root_cfg.get("q_max", 0.995)),
                "alpha_floor": float(root_cfg.get("alpha_floor", 0.05)),
                "kappa_min": float(root_cfg.get("kappa_min", 2.0)),
                "kappa_max": float(root_cfg.get("kappa_max", 80.0)),
                "risky_alpha_max": float(root_cfg.get("risky_alpha_max", 100.0)),
            }
        )
        if variant["policy_kind"] == "root_split_beta_dirichlet_learned_kp":
            gate_cfg = {
                **base_policy_cfg.get("learned_kp", {}),
                **variant.get("learned_kp", {}),
                **variant_policy_cfg.get("learned_kp", {}),
            }
            policy_kwargs.update(
                {
                    "gate_kappa_min": float(gate_cfg.get("gate_kappa_min", 8.0)),
                    "gate_kappa_max": float(gate_cfg.get("gate_kappa_max", 80.0)),
                }
            )
            return RootSplitBetaDirichletKpActorCriticPolicy, policy_kwargs
        if variant["policy_kind"] == "root_split_beta_dirichlet_routed":
            routed_cfg = {**base_policy_cfg.get("routed", {}), **variant_policy_cfg.get("routed", {})}
            policy_kwargs.update(
                {
                    "feature_columns": panel.feature_columns,
                    "root_feature_names": list(routed_cfg.get("root_feature_names", [])),
                    "root_latent_dim": int(routed_cfg.get("root_latent_dim", 32)),
                    "risky_latent_dim": int(routed_cfg.get("risky_latent_dim", 32)),
                    "routed_hidden_dim": int(routed_cfg.get("hidden_dim", 128)),
                }
            )
            return RoutedRootSplitBetaDirichletActorCriticPolicy, policy_kwargs
        return RootSplitBetaDirichletActorCriticPolicy, policy_kwargs

    if variant["policy_kind"] in {"riskcash_sector_dirtree", "riskcash_logitnormal_sector_tree"}:
        root_cfg = {
            **base_policy_cfg.get("root_split", {}),
            **variant.get("root_split", {}),
            **variant_policy_cfg.get("root_split", {}),
        }
        tree_cfg = {
            **base_policy_cfg.get("dirtree", {}),
            **variant.get("dirtree", {}),
            **variant_policy_cfg.get("dirtree", {}),
        }
        policy_kwargs.update(
            {
                "group_indices": sector_stock_group_indices(config, panel, variant),
                "q_min": float(root_cfg.get("q_min", 0.00)),
                "q_max": float(root_cfg.get("q_max", 0.995)),
                "alpha_floor": float(root_cfg.get("alpha_floor", 0.05)),
                "kappa_min": float(root_cfg.get("kappa_min", 2.0)),
                "kappa_max": float(root_cfg.get("kappa_max", 80.0)),
                "group_alpha_max": float(tree_cfg.get("group_alpha_max", 100.0)),
                "leaf_alpha_max": float(tree_cfg.get("leaf_alpha_max", 120.0)),
            }
        )
        if variant["policy_kind"] == "riskcash_logitnormal_sector_tree":
            logitnormal_cfg = {
                **base_policy_cfg.get("logitnormal", {}),
                **variant.get("logitnormal", {}),
                **variant_policy_cfg.get("logitnormal", {}),
            }
            policy_kwargs.update(
                {
                    "group_log_std_min": float(logitnormal_cfg.get("group_log_std_min", -2.5)),
                    "group_log_std_max": float(logitnormal_cfg.get("group_log_std_max", 0.3)),
                }
            )
            return RiskCashGroupLogisticNormalTreeActorCriticPolicy, policy_kwargs
        return RiskCashSectorDirichletTreeActorCriticPolicy, policy_kwargs

    if variant["policy_kind"] == "gaussian_logits":
        return "MlpPolicy", policy_kwargs

    raise ValueError(f"Unknown policy_kind: {variant['policy_kind']}")


def ppo_kwargs(config: dict[str, Any], smoke_test: bool, variant: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {**config["ppo"], **((variant or {}).get("ppo", {}))}
    total_timesteps = int(cfg.get("total_timesteps", 350_000))
    if smoke_test:
        total_timesteps = min(total_timesteps, 2_048)
    return {
        "total_timesteps": total_timesteps,
        "n_steps": min(int(cfg.get("n_steps", 1024)), total_timesteps),
        "batch_size": int(cfg.get("batch_size", 256)),
        "n_epochs": int(cfg.get("n_epochs", 4)),
        "learning_rate": float(cfg.get("learning_rate", 1e-4)),
        "gamma": float(cfg.get("gamma", 0.99)),
        "gae_lambda": float(cfg.get("gae_lambda", 0.95)),
        "clip_range": float(cfg.get("clip_range", 0.10)),
        "ent_coef": float(cfg.get("ent_coef", 0.0001)),
        "vf_coef": float(cfg.get("vf_coef", 0.5)),
        "max_grad_norm": float(cfg.get("max_grad_norm", 0.3)),
        "target_kl": float(cfg.get("target_kl", 0.01)),
        "seed": int(cfg.get("seed", 42)),
    }


def safe_sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    std = float(np.std(r, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(np.sqrt(252.0) * np.mean(r) / std)


def max_drawdown(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(v)
    dd = v / np.maximum(running_max, 1e-12) - 1.0
    return float(np.min(dd))


def benchmark_equal_weight(panel: WeightPanel, initial_amount: float) -> dict[str, float]:
    weights = np.full(len(panel.tickers), 1.0 / len(panel.tickers), dtype=np.float64)
    returns = panel.returns_next @ weights
    values = initial_amount * np.cumprod(1.0 + returns)
    return {
        "benchmark_return_pct": float(values[-1] / initial_amount - 1.0) if values.size else 0.0,
        "benchmark_sharpe": safe_sharpe(returns),
        "benchmark_max_drawdown": max_drawdown(values),
    }


def append_eval_info_row(rows: list[dict[str, Any]], panel: WeightPanel, info: dict[str, Any], reward: float) -> None:
    row = {
        "date": info["date"],
        "next_date": info["next_date"],
        "reward": reward,
        "portfolio_value": info["portfolio_value"],
        "gross_return": info["gross_return"],
        "net_return": info["net_return"],
        "turnover_l1": info["turnover_l1"],
        "stock_turnover_l1": info["stock_turnover_l1"],
        "transaction_cost": info["transaction_cost"],
        "drawdown": info["drawdown"],
        "drawdown_increment": info["drawdown_increment"],
        "concentration": info["concentration"],
        "projection_residual_l1": info["projection_residual_l1"],
        "controller_p_l1": info["controller_p_l1"],
        "controller_i_l1": info["controller_i_l1"],
        "controller_d_l1": info["controller_d_l1"],
        "controller_delta_l1_before_cap": info["controller_delta_l1_before_cap"],
    }
    optional_scalars = [
        "target_to_executed_l1",
        "q_target",
        "cash_target",
        "q_anchor",
        "cash_anchor",
        "q_scheduled",
        "cash_scheduled",
        "q_prev_exec",
        "q_exec",
        "cash_exec",
        "delta_q_target",
        "is_derisk",
        "gap_root",
        "gap_inner",
        "eps_root",
        "eps_inner",
        "tau_root",
        "tau_inner",
        "deadzone_scale_root",
        "deadzone_scale_inner",
        "k_root_base",
        "k_root_eff",
        "k_inner_base",
        "k_inner_eff",
        "learned_gates_enabled",
        "z_root_gate",
        "z_inner_gate",
        "k_root_min_bound",
        "k_root_max_bound",
        "k_inner_min_bound",
        "k_inner_max_bound",
        "root_turnover",
        "inner_turnover",
        "turnover_cap_scale",
        "target_turnover_l1",
        "suppressed_root_speed",
        "suppressed_inner_speed",
        "gate_prior_penalty",
        "gate_smooth_penalty",
        "raw_churn_penalty",
        "execution_penalty",
        "risk_score",
        "cash_allowed",
        "excess_cash",
        "cash_prior_penalty",
        "trade_buy_count",
        "trade_sell_count",
        "trade_hold_count",
        "trade_buy_weight_l1",
        "trade_sell_weight_l1",
        "cash_trade_delta",
        "cash_trade_direction",
        "risky_hhi_target",
        "risky_max_weight_target",
        "risky_entropy_target",
        "tree_group_hhi_target",
        "tree_group_max_weight_target",
        "tree_group_entropy_target",
        "bottomup_veto_enabled",
        "bottomup_q_raw",
        "bottomup_q_safe",
        "bottomup_cash_raw",
        "bottomup_cash_safe",
        "bottomup_cash_delta",
        "bottomup_raw_to_safe_l1",
        "bottomup_feedback_mean",
        "bottomup_feedback_max",
        "bottomup_feedback_global",
        "bottomup_feedback_active_rate",
        "bottomup_group_shift_l1",
        "projection_safety_enabled",
        "safety_projection_gap_l1",
        "safety_projection_gap_l2",
        "safety_projection_active",
        "safety_raw_violation",
        "safety_projected_violation",
        "safety_cash_min",
        "safety_cash_max",
        "safety_cash_raw",
        "safety_cash_projected",
        "safety_q_raw",
        "safety_q_projected",
        "safety_turnover_limit",
        "safety_turnover_raw",
        "safety_turnover_projected",
        "safety_turnover_scale",
        "safety_stock_bound_active_count",
        "safety_group_bound_active_count",
        "safety_rerisk_bound_active",
        "safety_derisk_bound_active",
        "safety_penalty",
        "style_entropy",
        "style_top_weight",
        "style_top_index",
        "style_turnover",
        "k_window_enabled",
        "k_window_days",
        "k_window_start_day",
        "k_window_planned_end_day",
        "k_window_substep",
        "window_day",
        "k_window_remaining_days",
        "remaining_days",
        "k_window_effective_days",
        "effective_K",
        "anchor_to_schedule_l1",
        "target_to_schedule_gap",
        "raw_to_anchor_l1",
        "schedule_to_target_l1",
        "schedule_to_exec_l1",
        "schedule_to_exec_gap",
        "confidence_stop_recovery_enabled",
        "event_trigger_allowed",
        "early_update_cooldown_remaining",
        "risk_stress",
        "recovery_score",
        "target_strength",
        "cash_duration",
        "cash_duration_score",
        "confidence_derisk",
        "confidence_rerisk",
        "delta_q_anchor",
        "delta_q_scheduled",
        "root_anchor_risk_day",
        "root_anchor_cash_day",
        "root_anchor_hold_day",
        "root_scheduled_risk_day",
        "root_scheduled_cash_day",
        "root_scheduled_hold_day",
        "stop_active",
        "suppressed_trade_l1",
        "suppressed_turnover",
        "suppressed_trade_value",
        "recovery_trigger_candidate",
        "risk_break_trigger_candidate",
        "recovery_persistence_count",
        "risk_break_persistence_count",
        "recovery_cash_condition_met",
        "recovery_anchor_condition_met",
        "recovery_confidence_condition_met",
        "recovery_residual_condition_met",
        "recovery_breadth_condition_met",
        "recovery_risk_stress_condition_met",
        "recovery_residual_up_5d",
        "recovery_breadth_excess_5d",
        "risk_break_confidence_condition_met",
        "risk_break_event_allowed",
        "confidence_rerisk_before_risk_gate",
        "confidence_rerisk_risk_gate",
        "recovery_trigger",
        "recovery_trigger_day",
        "risk_break_trigger",
        "risk_break_trigger_day",
        "derisk_early_update",
        "derisk_early_update_day",
        "rerisk_early_update",
        "rerisk_early_update_day",
        "window_closed_early",
        "macro_reward",
    ]
    for key in optional_scalars:
        if key in info:
            row[key] = info[key]
    for key in [
        "k_window_mode",
        "k_window_start_date",
        "k_window_end_date",
        "k_window_direction",
        "k_window_anchor_direction",
        "stop_reason",
        "early_update_reason",
    ]:
        if key in info:
            row[key] = info[key]
    for key in [
        "topk_buy_tickers",
        "topk_sell_tickers",
    ]:
        if key in info:
            row[key] = info[key]
    for key, value in info.items():
        if key.startswith(
            (
                "group_target_",
                "within_hhi_",
                "within_entropy_",
                "within_max_",
                "bottomup_feedback_",
                "bottomup_turnover_needed_",
                "style_weight_",
                "style_cash_",
                "style_contribution_cash_",
                "style_index_",
                "market_feature_",
                "confidence_component_",
                "confidence_mix_",
                "confidence_slice_",
                "dual_",
                "internal_trading_",
                "incremental_topk_",
                "stock_conf_",
                "stock_conf_component_",
                "stock_slice_",
                "topk_",
            )
        ):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = value

    target = np.asarray(info["target_weights"], dtype=np.float64)
    executed = np.asarray(info["executed_weights"], dtype=np.float64)
    post = np.asarray(info["post_market_weights"], dtype=np.float64)
    pre_trade = np.asarray(info.get("pre_trade_weights", []), dtype=np.float64)
    trade_delta = np.asarray(info.get("trade_delta_weights", []), dtype=np.float64)
    trade_abs = np.asarray(info.get("trade_abs_weights", []), dtype=np.float64)
    trade_direction = np.asarray(info.get("trade_direction", []), dtype=np.float64)
    anchor = np.asarray(info.get("anchor_weights", []), dtype=np.float64)
    scheduled = np.asarray(info.get("scheduled_weights", []), dtype=np.float64)
    raw = np.asarray(info.get("raw_weights", []), dtype=np.float64)
    incremental_topk = np.asarray(info.get("incremental_topk_weights", []), dtype=np.float64)
    for idx, ticker in enumerate(panel.tickers):
        row[f"target_weight_{ticker}"] = target[idx]
        row[f"executed_weight_{ticker}"] = executed[idx]
        row[f"post_market_weight_{ticker}"] = post[idx]
        if pre_trade.shape == target.shape:
            row[f"pre_trade_weight_{ticker}"] = pre_trade[idx]
        if trade_delta.shape == target.shape:
            row[f"trade_delta_weight_{ticker}"] = trade_delta[idx]
        if trade_abs.shape == target.shape:
            row[f"trade_abs_weight_{ticker}"] = trade_abs[idx]
        if trade_direction.shape == target.shape:
            row[f"trade_direction_{ticker}"] = trade_direction[idx]
        if raw.shape == target.shape:
            row[f"raw_weight_{ticker}"] = raw[idx]
        if anchor.shape == target.shape:
            row[f"anchor_weight_{ticker}"] = anchor[idx]
        if scheduled.shape == target.shape:
            row[f"scheduled_weight_{ticker}"] = scheduled[idx]
        if incremental_topk.shape == target.shape:
            row[f"incremental_topk_weight_{ticker}"] = incremental_topk[idx]
    row["target_weight_CASH"] = target[-1]
    row["executed_weight_CASH"] = executed[-1]
    row["post_market_weight_CASH"] = post[-1]
    if pre_trade.shape == target.shape:
        row["pre_trade_weight_CASH"] = pre_trade[-1]
    if trade_delta.shape == target.shape:
        row["trade_delta_weight_CASH"] = trade_delta[-1]
    if trade_abs.shape == target.shape:
        row["trade_abs_weight_CASH"] = trade_abs[-1]
    if trade_direction.shape == target.shape:
        row["trade_direction_CASH"] = trade_direction[-1]
    if raw.shape == target.shape:
        row["raw_weight_CASH"] = raw[-1]
    if anchor.shape == target.shape:
        row["anchor_weight_CASH"] = anchor[-1]
    if scheduled.shape == target.shape:
        row["scheduled_weight_CASH"] = scheduled[-1]
    if incremental_topk.shape == target.shape:
        row["incremental_topk_weight_CASH"] = incremental_topk[-1]
    rows.append(row)


def evaluate_model(
    model: InstrumentedPPO,
    panel: WeightPanel,
    config: dict[str, Any],
    variant: dict[str, Any],
    out_dir: Path,
    split_name: str,
) -> dict[str, float]:
    env = make_env_from_config(panel, config, variant)
    obs, _ = env.reset()
    rows: list[dict[str, Any]] = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        daily_steps = info.get("daily_steps")
        if isinstance(daily_steps, list) and daily_steps:
            for daily_info in daily_steps:
                append_eval_info_row(rows, panel, daily_info, float(daily_info.get("reward", reward)))
            continue
        append_eval_info_row(rows, panel, info, float(reward))
        continue
        row = {
            "date": info["date"],
            "next_date": info["next_date"],
            "reward": reward,
            "portfolio_value": info["portfolio_value"],
            "gross_return": info["gross_return"],
            "net_return": info["net_return"],
            "turnover_l1": info["turnover_l1"],
            "stock_turnover_l1": info["stock_turnover_l1"],
            "transaction_cost": info["transaction_cost"],
            "drawdown": info["drawdown"],
            "drawdown_increment": info["drawdown_increment"],
            "concentration": info["concentration"],
            "projection_residual_l1": info["projection_residual_l1"],
            "controller_p_l1": info["controller_p_l1"],
            "controller_i_l1": info["controller_i_l1"],
            "controller_d_l1": info["controller_d_l1"],
            "controller_delta_l1_before_cap": info["controller_delta_l1_before_cap"],
        }
        optional_scalars = [
            "target_to_executed_l1",
            "q_target",
            "cash_target",
            "q_prev_exec",
            "q_exec",
            "cash_exec",
            "delta_q_target",
            "is_derisk",
            "gap_root",
            "gap_inner",
            "eps_root",
            "eps_inner",
            "tau_root",
            "tau_inner",
            "deadzone_scale_root",
            "deadzone_scale_inner",
            "k_root_base",
            "k_root_eff",
            "k_inner_base",
            "k_inner_eff",
            "learned_gates_enabled",
            "z_root_gate",
            "z_inner_gate",
            "k_root_min_bound",
            "k_root_max_bound",
            "k_inner_min_bound",
            "k_inner_max_bound",
            "root_turnover",
            "inner_turnover",
            "turnover_cap_scale",
            "target_turnover_l1",
            "suppressed_root_speed",
            "suppressed_inner_speed",
            "gate_prior_penalty",
            "gate_smooth_penalty",
            "raw_churn_penalty",
            "execution_penalty",
            "risk_score",
            "cash_allowed",
            "excess_cash",
            "cash_prior_penalty",
            "risky_hhi_target",
            "risky_max_weight_target",
            "risky_entropy_target",
            "tree_group_hhi_target",
            "tree_group_max_weight_target",
            "tree_group_entropy_target",
            "bottomup_veto_enabled",
            "bottomup_q_raw",
            "bottomup_q_safe",
            "bottomup_cash_raw",
            "bottomup_cash_safe",
            "bottomup_cash_delta",
            "bottomup_raw_to_safe_l1",
            "bottomup_feedback_mean",
            "bottomup_feedback_max",
            "bottomup_feedback_global",
            "bottomup_feedback_active_rate",
            "bottomup_group_shift_l1",
            "projection_safety_enabled",
            "safety_projection_gap_l1",
            "safety_projection_gap_l2",
            "safety_projection_active",
            "safety_raw_violation",
            "safety_projected_violation",
            "safety_cash_min",
            "safety_cash_max",
            "safety_cash_raw",
            "safety_cash_projected",
            "safety_q_raw",
            "safety_q_projected",
            "safety_turnover_limit",
            "safety_turnover_raw",
            "safety_turnover_projected",
            "safety_turnover_scale",
            "safety_stock_bound_active_count",
            "safety_group_bound_active_count",
            "safety_rerisk_bound_active",
            "safety_derisk_bound_active",
            "safety_penalty",
            "style_entropy",
            "style_top_weight",
            "style_top_index",
            "style_turnover",
        ]
        for key in optional_scalars:
            if key in info:
                row[key] = info[key]
        for key, value in info.items():
            if key.startswith(
                (
                    "group_target_",
                    "within_hhi_",
                    "within_entropy_",
                    "within_max_",
                    "bottomup_feedback_",
                    "bottomup_turnover_needed_",
                    "style_weight_",
                    "style_cash_",
                    "style_contribution_cash_",
                    "style_index_",
                    "confidence_component_",
                    "confidence_mix_",
                    "confidence_slice_",
                    "dual_",
                    "internal_trading_",
                    "incremental_topk_",
                    "stock_conf_",
                    "stock_conf_component_",
                    "stock_slice_",
                )
            ):
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
        target = np.asarray(info["target_weights"], dtype=np.float64)
        executed = np.asarray(info["executed_weights"], dtype=np.float64)
        post = np.asarray(info["post_market_weights"], dtype=np.float64)
        for idx, ticker in enumerate(panel.tickers):
            row[f"target_weight_{ticker}"] = target[idx]
            row[f"executed_weight_{ticker}"] = executed[idx]
            row[f"post_market_weight_{ticker}"] = post[idx]
        row["target_weight_CASH"] = target[-1]
        row["executed_weight_CASH"] = executed[-1]
        row["post_market_weight_CASH"] = post[-1]
        rows.append(row)

    daily = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_path = out_dir / f"{split_name}_daily.csv"
    daily.to_csv(daily_path, index=False)

    initial_amount = float(config["environment"].get("initial_amount", 1_000_000.0))
    returns = daily["net_return"].to_numpy(dtype=np.float64)
    values = daily["portfolio_value"].to_numpy(dtype=np.float64)
    benchmark = benchmark_equal_weight(panel, initial_amount)
    summary = {
        "split": split_name,
        "days": int(len(daily)),
        "return_pct": float(values[-1] / initial_amount - 1.0) if len(values) else 0.0,
        "sharpe": safe_sharpe(returns),
        "max_drawdown": max_drawdown(values),
        "turnover_l1_mean": float(daily["turnover_l1"].mean()) if len(daily) else 0.0,
        "stock_turnover_l1_mean": float(daily["stock_turnover_l1"].mean()) if len(daily) else 0.0,
        "transaction_cost_mean": float(daily["transaction_cost"].mean()) if len(daily) else 0.0,
        "cash_weight_mean": float(daily["executed_weight_CASH"].mean()) if len(daily) else 0.0,
        "projection_residual_l1_mean": float(daily["projection_residual_l1"].mean()) if len(daily) else 0.0,
        **benchmark,
    }
    for key in [
        "target_to_executed_l1",
        "q_target",
        "q_exec",
        "cash_target",
        "cash_exec",
        "k_root_eff",
        "k_inner_eff",
        "z_root_gate",
        "z_inner_gate",
        "gate_prior_penalty",
        "gate_smooth_penalty",
        "raw_churn_penalty",
        "execution_penalty",
        "deadzone_scale_root",
        "deadzone_scale_inner",
        "root_turnover",
        "inner_turnover",
        "target_turnover_l1",
        "risk_score",
        "cash_allowed",
        "excess_cash",
        "cash_prior_penalty",
        "trade_buy_count",
        "trade_sell_count",
        "trade_hold_count",
        "trade_buy_weight_l1",
        "trade_sell_weight_l1",
        "cash_trade_delta",
        "cash_trade_direction",
        "tree_group_hhi_target",
        "tree_group_max_weight_target",
        "tree_group_entropy_target",
        "bottomup_raw_to_safe_l1",
        "bottomup_feedback_mean",
        "bottomup_feedback_max",
        "bottomup_feedback_global",
        "bottomup_feedback_active_rate",
        "bottomup_group_shift_l1",
        "safety_projection_gap_l1",
        "safety_projection_active",
        "safety_raw_violation",
        "safety_projected_violation",
        "safety_turnover_projected",
        "safety_penalty",
        "style_entropy",
        "style_top_weight",
        "style_turnover",
        "k_window_enabled",
        "k_window_days",
        "k_window_substep",
        "window_day",
        "k_window_remaining_days",
        "remaining_days",
        "k_window_effective_days",
        "effective_K",
        "q_anchor",
        "cash_anchor",
        "q_scheduled",
        "cash_scheduled",
        "anchor_to_schedule_l1",
        "target_to_schedule_gap",
        "raw_to_anchor_l1",
        "schedule_to_exec_l1",
        "schedule_to_exec_gap",
        "schedule_to_target_l1",
        "confidence_stop_recovery_enabled",
        "event_trigger_allowed",
        "early_update_cooldown_remaining",
        "risk_stress",
        "recovery_score",
        "target_strength",
        "cash_duration",
        "cash_duration_score",
        "confidence_derisk",
        "confidence_rerisk",
        "delta_q_anchor",
        "delta_q_scheduled",
        "root_anchor_risk_day",
        "root_anchor_cash_day",
        "root_anchor_hold_day",
        "root_scheduled_risk_day",
        "root_scheduled_cash_day",
        "root_scheduled_hold_day",
        "stop_active",
        "suppressed_trade_l1",
        "suppressed_turnover",
        "suppressed_trade_value",
        "recovery_trigger_candidate",
        "risk_break_trigger_candidate",
        "recovery_persistence_count",
        "risk_break_persistence_count",
        "recovery_cash_condition_met",
        "recovery_anchor_condition_met",
        "recovery_confidence_condition_met",
        "recovery_residual_condition_met",
        "recovery_breadth_condition_met",
        "recovery_risk_stress_condition_met",
        "recovery_residual_up_5d",
        "recovery_breadth_excess_5d",
        "risk_break_confidence_condition_met",
        "risk_break_event_allowed",
        "confidence_rerisk_before_risk_gate",
        "confidence_rerisk_risk_gate",
        "recovery_trigger",
        "recovery_trigger_day",
        "risk_break_trigger",
        "risk_break_trigger_day",
        "derisk_early_update",
        "derisk_early_update_day",
        "rerisk_early_update",
        "rerisk_early_update_day",
        "window_closed_early",
    ]:
        if key in daily.columns:
            summary[f"{key}_mean"] = float(daily[key].mean()) if len(daily) else 0.0
    for column in daily.columns:
        if column.startswith(
            (
                "group_target_",
                "within_hhi_",
                "within_entropy_",
                "within_max_",
                "bottomup_feedback_",
                "bottomup_turnover_needed_",
                "style_weight_",
                "style_cash_",
                "style_contribution_cash_",
                "style_index_",
                "market_feature_",
                "confidence_component_",
                "confidence_mix_",
                "confidence_slice_",
                "dual_",
                "internal_trading_",
                "incremental_topk_",
                "stock_conf_",
                "stock_conf_component_",
                "stock_slice_",
                "topk_",
            )
        ):
            if pd.api.types.is_numeric_dtype(daily[column]):
                summary[f"{column}_mean"] = float(daily[column].mean()) if len(daily) else 0.0
    pd.DataFrame([summary]).to_csv(out_dir / f"{split_name}_summary.csv", index=False)
    return summary


def load_folds(config: dict[str, Any], requested_folds: list[str] | None) -> pd.DataFrame:
    fold_cfg = config["walk_forward"]
    folds = pd.read_csv(resolve(fold_cfg["folds_csv"]))
    allowed = requested_folds or fold_cfg.get("fold_ids") or folds["fold"].tolist()
    folds = folds[folds["fold"].isin(allowed)].copy()
    if folds.empty:
        raise ValueError(f"No folds selected. requested={requested_folds}")
    return folds


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_variant_inheritance(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {variant["name"]: variant for variant in variants}
    resolved: dict[str, dict[str, Any]] = {}
    resolving: set[str] = set()

    def resolve_one(name: str) -> dict[str, Any]:
        if name in resolved:
            return copy.deepcopy(resolved[name])
        if name in resolving:
            raise ValueError(f"Cyclic variant inheritance detected at {name}")
        if name not in by_name:
            raise ValueError(f"Unknown base variant in inheritance: {name}")
        resolving.add(name)
        variant = by_name[name]
        parent_name = variant.get("inherits")
        if parent_name:
            parent = resolve_one(str(parent_name))
            merged = deep_merge_dicts(parent, variant)
        else:
            merged = copy.deepcopy(variant)
        merged.pop("inherits", None)
        resolved[name] = merged
        resolving.remove(name)
        return copy.deepcopy(merged)

    return [resolve_one(variant["name"]) for variant in variants]


def selected_variants(config: dict[str, Any], requested_variants: list[str] | None) -> list[dict[str, Any]]:
    variants = resolve_variant_inheritance(config["variants"])
    if not requested_variants:
        return [v for v in variants if v.get("enabled", True)]
    wanted = set(requested_variants)
    selected = [v for v in variants if v["name"] in wanted]
    missing = wanted.difference(v["name"] for v in selected)
    if missing:
        raise ValueError(f"Unknown variants: {sorted(missing)}")
    return selected


def feature_csv_for_fold(
    config: dict[str, Any],
    variant: dict[str, Any],
    fold: pd.Series,
    out_root: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    data_cfg = config["data"]
    normalization_cfg = config.get("normalization", {})
    if not normalization_cfg.get("enabled", False):
        return {
            "model_ready_csv": resolve(data_cfg["model_ready_csv"]),
            "transform_stats_csv": resolve(data_cfg["transform_stats_csv"])
            if data_cfg.get("transform_stats_csv")
            else None,
            "diagnostics_csv": None,
            "manifest_json": None,
            "status": "disabled",
        }

    raw_csv = data_cfg.get("raw_features_csv")
    if not raw_csv:
        raise ValueError("normalization.enabled=true requires data.raw_features_csv.")

    feature_subset_name = variant.get("feature_subset")
    feature_subset = None
    scaler_out_dir = out_root / "feature_scalers"
    if feature_subset_name:
        subsets = config.get("feature_subsets", {})
        if feature_subset_name not in subsets:
            raise ValueError(
                f"Variant {variant['name']} requests unknown feature_subset={feature_subset_name}. "
                f"Available: {sorted(subsets)}"
            )
        feature_subset = list(subsets[feature_subset_name])
        scaler_out_dir = scaler_out_dir / str(feature_subset_name)

    return prepare_fold_scaled_features(
        raw_csv=resolve(raw_csv),
        out_dir=scaler_out_dir,
        fold_id=str(fold["fold"]),
        train_start=str(fold["train_start"]),
        train_end=str(fold["train_end_inclusive"]),
        validation_end=str(fold["validation_end_inclusive"]),
        feature_subset=feature_subset,
        feature_subset_name=str(feature_subset_name) if feature_subset_name else None,
        lower_quantile=float(normalization_cfg.get("lower_quantile", 0.01)),
        upper_quantile=float(normalization_cfg.get("upper_quantile", 0.99)),
        force=force,
    )


def _average_cluster_distance(a: list[int], b: list[int], distance: np.ndarray) -> float:
    values = distance[np.ix_(a, b)]
    return float(np.mean(values))


def _agglomerative_average_clusters(distance: np.ndarray, target_num_groups: int) -> list[list[int]]:
    clusters = [[idx] for idx in range(distance.shape[0])]
    target = max(1, min(int(target_num_groups), len(clusters)))
    while len(clusters) > target:
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = _average_cluster_distance(clusters[i], clusters[j], distance)
                if d < best_distance:
                    best_distance = d
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        merged = sorted(clusters[i] + clusters[j])
        clusters = [cluster for k, cluster in enumerate(clusters) if k not in {i, j}]
        clusters.append(merged)
    return sorted([sorted(cluster) for cluster in clusters], key=lambda c: (c[0], len(c)))


def _merge_small_clusters(clusters: list[list[int]], distance: np.ndarray, min_group_size: int) -> list[list[int]]:
    min_size = max(1, int(min_group_size))
    clusters = [list(cluster) for cluster in clusters]
    while len(clusters) > 1:
        small_idx = next((idx for idx, cluster in enumerate(clusters) if len(cluster) < min_size), None)
        if small_idx is None:
            break
        nearest_idx = None
        nearest_distance = float("inf")
        preferred = [
            (idx, cluster)
            for idx, cluster in enumerate(clusters)
            if idx != small_idx and len(cluster) < min_size
        ]
        candidates = preferred if preferred else [(idx, cluster) for idx, cluster in enumerate(clusters) if idx != small_idx]
        for idx, cluster in enumerate(clusters):
            if not any(candidate_idx == idx for candidate_idx, _ in candidates):
                continue
            d = _average_cluster_distance(clusters[small_idx], cluster, distance)
            if d < nearest_distance:
                nearest_distance = d
                nearest_idx = idx
        if nearest_idx is None:
            break
        merged = sorted(clusters[small_idx] + clusters[nearest_idx])
        clusters = [cluster for idx, cluster in enumerate(clusters) if idx not in {small_idx, nearest_idx}]
        clusters.append(merged)
        clusters = sorted(clusters, key=lambda c: (c[0], len(c)))
    return clusters


def _sample_correlation(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"Expected 2-D sample matrix, got shape={samples.shape}")
    if samples.shape[0] < 2:
        corr = np.eye(samples.shape[1], dtype=np.float64)
    else:
        corr = np.corrcoef(samples, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -0.99, 0.99)
    np.fill_diagonal(corr, 1.0)
    return corr


def _distance_from_similarity(similarity: np.ndarray) -> np.ndarray:
    similarity = np.asarray(similarity, dtype=np.float64)
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
    similarity = np.clip(similarity, -0.99, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return np.sqrt(np.maximum(0.0, 0.5 * (1.0 - similarity)))


def discover_residual_correlation_groups(
    panel: WeightPanel,
    *,
    target_num_groups: int = 6,
    min_group_size: int = 2,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    """Build train-only residual-correlation groups from the training panel."""
    returns = np.asarray(panel.returns_next, dtype=np.float64)
    market = returns.mean(axis=1)
    design = np.column_stack([np.ones_like(market), market])
    residuals = np.zeros_like(returns)
    for idx in range(returns.shape[1]):
        beta, *_ = np.linalg.lstsq(design, returns[:, idx], rcond=None)
        residuals[:, idx] = returns[:, idx] - design @ beta

    corr = _sample_correlation(residuals)
    distance = _distance_from_similarity(corr)
    clusters = _agglomerative_average_clusters(distance, target_num_groups)
    clusters = _merge_small_clusters(clusters, distance, min_group_size)
    clusters = sorted([sorted(cluster) for cluster in clusters], key=lambda c: (c[0], len(c)))
    return clusters, corr, distance


def _stable_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.nan_to_num(np.asarray(logits, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    z = logits / max(float(temperature), 1e-8)
    z -= float(np.max(z))
    exp_z = np.exp(np.clip(z, -60.0, 60.0))
    total = float(exp_z.sum())
    if total <= 1e-12 or not np.isfinite(total):
        return np.full_like(logits, 1.0 / max(len(logits), 1), dtype=np.float64)
    return exp_z / total


def _train_only_proxy_target_weights(
    panel: WeightPanel,
    *,
    lookback: int = 20,
    temperature: float = 0.35,
    top_k: int = 0,
    equal_weight_blend: float = 0.10,
) -> np.ndarray:
    """Create causal train-only target weights used only for discovered targetcov topology.

    This is a deterministic proxy teacher, not a trained PPO teacher. It uses only
    prices available inside the training panel up to day t: trailing momentum scaled
    by trailing volatility. The resulting target-delta co-movement matrix is used
    only to discover a frozen hierarchy for the fold.
    """
    prices = np.asarray(panel.prices, dtype=np.float64)
    if prices.ndim != 2:
        raise ValueError(f"Expected price panel with shape [time, assets], got {prices.shape}")
    n_days, n_assets = prices.shape
    if n_assets <= 0:
        raise ValueError("Cannot discover target covariance groups without assets.")

    targets = np.full((n_days, n_assets), 1.0 / n_assets, dtype=np.float64)
    returns = np.zeros_like(prices, dtype=np.float64)
    if n_days > 1:
        returns[1:] = prices[1:] / np.maximum(prices[:-1], 1e-8) - 1.0

    lookback = max(2, int(lookback))
    top_k = int(top_k)
    if top_k <= 0 or top_k > n_assets:
        top_k = n_assets
    blend = float(np.clip(equal_weight_blend, 0.0, 1.0))
    equal = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)

    for day in range(n_days):
        start = max(0, day - lookback)
        if day - start < 2:
            continue

        start_prices = np.maximum(prices[start], 1e-8)
        momentum = prices[day] / start_prices - 1.0
        trailing_returns = returns[start + 1 : day + 1]
        if trailing_returns.shape[0] > 1:
            vol = np.std(trailing_returns, axis=0, ddof=1)
        elif trailing_returns.shape[0] == 1:
            vol = np.abs(trailing_returns[0])
        else:
            vol = np.ones(n_assets, dtype=np.float64)
        vol = np.maximum(np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0), 1e-4)

        scores = np.nan_to_num(momentum / vol, nan=0.0, posinf=0.0, neginf=0.0)
        score_std = float(np.std(scores))
        if score_std > 1e-8:
            scores = (scores - float(np.mean(scores))) / score_std
        else:
            scores = np.zeros_like(scores)
        scores = np.clip(scores, -5.0, 5.0)

        if top_k < n_assets:
            selected = np.argsort(scores)[-top_k:]
            masked = np.full(n_assets, -60.0, dtype=np.float64)
            masked[selected] = scores[selected]
            scores = masked

        weights = _stable_softmax(scores, temperature)
        targets[day] = (1.0 - blend) * weights + blend * equal
        targets[day] /= max(float(targets[day].sum()), 1e-12)

    return targets


def discover_target_covariance_groups(
    panel: WeightPanel,
    *,
    target_num_groups: int = 6,
    min_group_size: int = 2,
    lookback: int = 20,
    temperature: float = 0.35,
    top_k: int = 0,
    equal_weight_blend: float = 0.10,
    correlation_mode: str = "positive",
) -> tuple[list[list[int]], np.ndarray, np.ndarray, dict[str, Any]]:
    """Build train-only target-delta co-movement groups from a deterministic proxy teacher."""
    targets = _train_only_proxy_target_weights(
        panel,
        lookback=lookback,
        temperature=temperature,
        top_k=top_k,
        equal_weight_blend=equal_weight_blend,
    )
    deltas = np.diff(targets, axis=0)
    raw_corr = _sample_correlation(deltas)

    mode = str(correlation_mode).lower()
    if mode in {"positive", "positive_only", "pos"}:
        similarity = np.maximum(raw_corr, 0.0)
    elif mode in {"absolute", "abs"}:
        similarity = np.abs(raw_corr)
    elif mode in {"signed", "raw"}:
        similarity = raw_corr.copy()
    else:
        raise ValueError(
            f"Unsupported target covariance correlation_mode={correlation_mode}. "
            "Use positive, absolute, or signed."
        )
    np.fill_diagonal(similarity, 1.0)

    distance = _distance_from_similarity(similarity)
    clusters = _agglomerative_average_clusters(distance, target_num_groups)
    clusters = _merge_small_clusters(clusters, distance, min_group_size)
    clusters = sorted([sorted(cluster) for cluster in clusters], key=lambda c: (c[0], len(c)))
    diagnostics = {
        "target_weights": targets,
        "target_deltas": deltas,
        "raw_correlation": raw_corr,
        "similarity": similarity,
        "correlation_mode": mode,
    }
    return clusters, similarity, distance, diagnostics


def _panel_feature_mean_series(
    panel: WeightPanel,
    feature: str,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a date-level feature series from a WeightPanel."""
    if feature not in panel.feature_columns:
        raise KeyError(f"Feature {feature!r} is not present in the panel.")
    idx = panel.feature_columns.index(feature)
    values = panel.features[:, :, idx].astype(np.float64)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values.mean(axis=1)


def _safe_ols_beta(y: np.ndarray, x: np.ndarray, *, min_samples: int, clip_abs: float) -> float:
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(yy) & np.isfinite(xx)
    yy = yy[mask]
    xx = xx[mask]
    if yy.size < int(min_samples):
        return 0.0
    x_var = float(np.var(xx))
    if x_var <= 1e-12:
        return 0.0
    beta = float(np.cov(xx, yy, ddof=0)[0, 1] / x_var)
    return float(np.clip(beta, -float(clip_abs), float(clip_abs)))


def _mean_stock_beta(
    panel: WeightPanel,
    *,
    stock_feature: str,
    market_feature: str,
    min_samples: int,
    clip_abs: float,
    mask: np.ndarray | None = None,
) -> float:
    if stock_feature not in panel.feature_columns or market_feature not in panel.feature_columns:
        return 0.0
    stock_idx = panel.feature_columns.index(stock_feature)
    safe_mask = np.asarray(mask, dtype=bool) if mask is not None else None
    market = _panel_feature_mean_series(panel, market_feature, safe_mask)
    betas = []
    for asset_idx in range(len(panel.tickers)):
        y = panel.features[:, asset_idx, stock_idx].astype(np.float64)
        if safe_mask is not None:
            y = y[safe_mask]
        betas.append(_safe_ols_beta(y, market, min_samples=min_samples, clip_abs=clip_abs))
    finite = [beta for beta in betas if np.isfinite(beta)]
    return float(np.mean(finite)) if finite else 0.0


def _derived_beta_cfg_terms(
    beta_cfg: dict[str, Any],
) -> dict[str, str | float | int | bool]:
    shrinkage_cfg = beta_cfg.get("shrinkage", {})
    return {
        "min_samples": int(beta_cfg.get("min_samples", 120)),
        "clip_abs": float(beta_cfg.get("clip_abs", 5.0)),
        "shrinkage_enabled": bool(shrinkage_cfg.get("enabled", beta_cfg.get("shrinkage_enabled", False))),
        "shrinkage_lambda": float(shrinkage_cfg.get("lambda", beta_cfg.get("shrinkage_lambda", 0.0))),
        "final_clip_abs": float(shrinkage_cfg.get("final_clip_abs", beta_cfg.get("final_clip_abs", beta_cfg.get("clip_abs", 5.0)))),
        "market_factor_feature_5d": str(beta_cfg.get("market_factor_feature_5d", "SP500_Trend")),
        "market_factor_feature_20d": str(beta_cfg.get("market_factor_feature_20d", "SP500_Trend")),
        "stock_market_factor_feature_5d": str(beta_cfg.get("stock_market_factor_feature_5d", "universe_return_5d")),
        "stock_market_factor_feature_20d": str(beta_cfg.get("stock_market_factor_feature_20d", "universe_return_20d")),
        "vix_factor_feature_5d": str(beta_cfg.get("vix_factor_feature_5d", "universe_return_5d")),
        "vix_trend_factor_feature_5d": str(beta_cfg.get("vix_trend_factor_feature_5d", "SP500_Trend")),
    }


def _estimate_derived_beta_row(
    panel: WeightPanel,
    *,
    fit_mask: np.ndarray,
    terms: dict[str, str | float | int],
    derived_defaults: dict[str, Any],
) -> dict[str, float | str]:
    min_samples = int(terms["min_samples"])
    clip_abs = float(terms["clip_abs"])
    shrinkage_enabled = bool(terms.get("shrinkage_enabled", False))
    shrinkage_lambda = max(float(terms.get("shrinkage_lambda", 0.0)), 0.0)
    final_clip_abs = float(terms.get("final_clip_abs", clip_abs))
    market_factor_5d = str(terms["market_factor_feature_5d"])
    market_factor_20d = str(terms["market_factor_feature_20d"])
    stock_market_factor_5d = str(terms["stock_market_factor_feature_5d"])
    stock_market_factor_20d = str(terms["stock_market_factor_feature_20d"])
    vix_factor_5d = str(terms["vix_factor_feature_5d"])
    vix_trend_factor_5d = str(terms["vix_trend_factor_feature_5d"])

    fallback_market = float(derived_defaults.get("market_beta", 0.50))
    fallback_vix = float(derived_defaults.get("vix_market_beta", 0.25))
    fit_samples = int(np.sum(np.asarray(fit_mask, dtype=bool)))

    def shrink_beta(raw: float, prior: float) -> tuple[float, float]:
        if not shrinkage_enabled:
            return float(raw), 1.0
        weight = fit_samples / (fit_samples + shrinkage_lambda) if (fit_samples + shrinkage_lambda) > 0 else 0.0
        beta = weight * float(raw) + (1.0 - weight) * float(prior)
        return float(np.clip(beta, -final_clip_abs, final_clip_abs)), float(weight)

    def series_beta(y_feature: str, x_feature: str, fallback: float) -> tuple[float, float, float]:
        if fit_samples < min_samples:
            return float(fallback), float(fallback), 0.0
        try:
            raw = _safe_ols_beta(
                _panel_feature_mean_series(panel, y_feature, fit_mask),
                _panel_feature_mean_series(panel, x_feature, fit_mask),
                min_samples=min_samples,
                clip_abs=clip_abs,
            )
            beta, weight = shrink_beta(raw, fallback)
            return beta, float(raw), weight
        except KeyError:
            return float(fallback), float(fallback), 0.0

    def stock_beta(stock_feature: str, market_feature: str, fallback: float) -> tuple[float, float, float]:
        if fit_samples < min_samples:
            return float(fallback), float(fallback), 0.0
        if stock_feature not in panel.feature_columns or market_feature not in panel.feature_columns:
            return float(fallback), float(fallback), 0.0
        raw = _mean_stock_beta(
            panel,
            stock_feature=stock_feature,
            market_feature=market_feature,
            min_samples=min_samples,
            clip_abs=clip_abs,
            mask=fit_mask,
        )
        raw = float(raw) if np.isfinite(raw) else float(fallback)
        beta, weight = shrink_beta(raw, fallback)
        return beta, raw, weight

    market_beta_5d, market_beta_5d_raw, market_beta_5d_w = series_beta(
        "universe_return_5d",
        market_factor_5d,
        fallback_market,
    )
    market_beta_20d, market_beta_20d_raw, market_beta_20d_w = series_beta(
        "universe_return_20d",
        market_factor_20d,
        fallback_market,
    )
    stock_market_beta_5d, stock_market_beta_5d_raw, stock_market_beta_5d_w = stock_beta(
        "logret_5d",
        stock_market_factor_5d,
        market_beta_5d,
    )
    stock_market_beta_20d, stock_market_beta_20d_raw, stock_market_beta_20d_w = stock_beta(
        "logret_20d",
        stock_market_factor_20d,
        market_beta_20d,
    )
    vix_market_beta_5d, vix_market_beta_5d_raw, vix_market_beta_5d_w = series_beta(
        "VIX_change_5d",
        vix_factor_5d,
        fallback_vix,
    )
    vix_trend_beta_5d, vix_trend_beta_5d_raw, vix_trend_beta_5d_w = series_beta(
        "VIX_change_5d",
        vix_trend_factor_5d,
        fallback_vix,
    )
    return {
        "fit_samples": float(fit_samples),
        "market_beta_5d": market_beta_5d,
        "market_beta_20d": market_beta_20d,
        "stock_market_beta_5d": stock_market_beta_5d,
        "stock_market_beta_20d": stock_market_beta_20d,
        "vix_market_beta_5d": vix_market_beta_5d,
        "vix_trend_beta_5d": vix_trend_beta_5d,
        "market_beta_5d_raw": market_beta_5d_raw,
        "market_beta_20d_raw": market_beta_20d_raw,
        "stock_market_beta_5d_raw": stock_market_beta_5d_raw,
        "stock_market_beta_20d_raw": stock_market_beta_20d_raw,
        "vix_market_beta_5d_raw": vix_market_beta_5d_raw,
        "vix_trend_beta_5d_raw": vix_trend_beta_5d_raw,
        "market_beta_5d_shrinkage_weight": market_beta_5d_w,
        "market_beta_20d_shrinkage_weight": market_beta_20d_w,
        "stock_market_beta_5d_shrinkage_weight": stock_market_beta_5d_w,
        "stock_market_beta_20d_shrinkage_weight": stock_market_beta_20d_w,
        "vix_market_beta_5d_shrinkage_weight": vix_market_beta_5d_w,
        "vix_trend_beta_5d_shrinkage_weight": vix_trend_beta_5d_w,
        "beta_shrinkage_enabled": float(shrinkage_enabled),
        "beta_shrinkage_lambda": float(shrinkage_lambda),
        "beta_final_clip_abs": float(final_clip_abs),
    }


def materialize_train_only_derived_betas(
    variant: dict[str, Any],
    fold: pd.Series,
    train_panel: WeightPanel,
    run_dir: Path,
    beta_panel: WeightPanel | None = None,
) -> dict[str, Any]:
    """Materialize residual-feature betas with causal train/as-of fitting."""
    root_split = variant.get("root_split", {})
    derived = root_split.get("derived_features", {})
    beta_cfg = derived.get("train_only_beta", {})
    if not beta_cfg or not bool(beta_cfg.get("enabled", False)):
        return variant

    variant_for_fold = copy.deepcopy(variant)
    root_split_out = variant_for_fold.setdefault("root_split", {})
    derived_out = root_split_out.setdefault("derived_features", {})
    beta_cfg = derived_out.get("train_only_beta", {})

    terms = _derived_beta_cfg_terms(beta_cfg)
    mode = str(beta_cfg.get("mode", beta_cfg.get("schedule", "fold_static"))).lower()
    fit_panel = beta_panel if (mode in {"monthly_asof", "monthly", "asof_monthly"} and beta_panel is not None) else train_panel

    for cfg_key in [
        "market_factor_feature_5d",
        "market_factor_feature_20d",
        "stock_market_factor_feature_5d",
        "stock_market_factor_feature_20d",
        "vix_factor_feature_5d",
        "vix_trend_factor_feature_5d",
    ]:
        derived_out[cfg_key] = str(terms[cfg_key])
    derived_out["vix_surprise_mode"] = str(beta_cfg.get("vix_surprise_mode", "regression"))

    out_dir = run_dir / "derived_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode in {"monthly_asof", "monthly", "asof_monthly"}:
        dates = pd.to_datetime(pd.Series(fit_panel.dates))
        month_starts = sorted(pd.Timestamp(period.start_time) for period in dates.dt.to_period("M").unique())
        rows: list[dict[str, float | str]] = []
        for month_start in month_starts:
            fit_mask = dates < month_start
            row = _estimate_derived_beta_row(
                fit_panel,
                fit_mask=fit_mask.to_numpy(dtype=bool),
                terms=terms,
                derived_defaults=derived_out,
            )
            if bool(fit_mask.any()):
                fit_end = dates.loc[fit_mask].max()
                fit_start = dates.loc[fit_mask].min()
            else:
                fit_start = pd.NaT
                fit_end = pd.NaT
            row.update(
                {
                    "date": str(month_start.date()),
                    "fold": str(fold["fold"]),
                    "source": "monthly_asof_ols",
                    "fit_start_date": "" if pd.isna(fit_start) else str(pd.Timestamp(fit_start).date()),
                    "fit_end_date": "" if pd.isna(fit_end) else str(pd.Timestamp(fit_end).date()),
                    "min_samples": float(terms["min_samples"]),
                    "clip_abs": float(terms["clip_abs"]),
                    "market_factor_feature_5d": str(terms["market_factor_feature_5d"]),
                    "market_factor_feature_20d": str(terms["market_factor_feature_20d"]),
                    "stock_market_factor_feature_5d": str(terms["stock_market_factor_feature_5d"]),
                    "stock_market_factor_feature_20d": str(terms["stock_market_factor_feature_20d"]),
                    "vix_factor_feature_5d": str(terms["vix_factor_feature_5d"]),
                    "vix_trend_factor_feature_5d": str(terms["vix_trend_factor_feature_5d"]),
                }
            )
            rows.append(row)

        derived_out["beta_schedule"] = rows
        derived_out["train_only_beta_materialized"] = {
            "fold": str(fold["fold"]),
            "source": "monthly_asof_ols",
            "rows": float(len(rows)),
            "min_samples": float(terms["min_samples"]),
            "clip_abs": float(terms["clip_abs"]),
        }
        pd.DataFrame(rows).to_csv(out_dir / "monthly_derived_betas.csv", index=False)
        (out_dir / "monthly_derived_betas.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        if rows:
            pd.DataFrame([rows[-1]]).to_csv(out_dir / "train_only_derived_betas.csv", index=False)
            (out_dir / "train_only_derived_betas.json").write_text(json.dumps(rows[-1], indent=2), encoding="utf-8")
        return variant_for_fold

    fit_mask = np.ones(len(train_panel.dates), dtype=bool)
    estimates = _estimate_derived_beta_row(
        train_panel,
        fit_mask=fit_mask,
        terms=terms,
        derived_defaults=derived_out,
    )
    estimates.update(
        {
            "fold": str(fold["fold"]),
            "source": "train_only_ols",
            "min_samples": float(terms["min_samples"]),
            "clip_abs": float(terms["clip_abs"]),
            "market_factor_feature_5d": str(terms["market_factor_feature_5d"]),
            "market_factor_feature_20d": str(terms["market_factor_feature_20d"]),
            "stock_market_factor_feature_5d": str(terms["stock_market_factor_feature_5d"]),
            "stock_market_factor_feature_20d": str(terms["stock_market_factor_feature_20d"]),
            "vix_factor_feature_5d": str(terms["vix_factor_feature_5d"]),
            "vix_trend_factor_feature_5d": str(terms["vix_trend_factor_feature_5d"]),
        }
    )
    for key, value in estimates.items():
        if isinstance(value, float):
            derived_out[key] = float(value)
    derived_out["train_only_beta_materialized"] = {
        key: value for key, value in estimates.items() if isinstance(value, (float, str))
    }

    pd.DataFrame([{key: value for key, value in estimates.items()}]).to_csv(
        out_dir / "train_only_derived_betas.csv",
        index=False,
    )
    (out_dir / "train_only_derived_betas.json").write_text(
        json.dumps(estimates, indent=2),
        encoding="utf-8",
    )
    return variant_for_fold


def materialize_variant_for_fold(
    config: dict[str, Any],
    variant: dict[str, Any],
    fold: pd.Series,
    train_panel: WeightPanel,
    run_dir: Path,
    beta_panel: WeightPanel | None = None,
) -> dict[str, Any]:
    """Add fold-specific train-only discovered hierarchy artifacts when requested."""
    variant_for_fold = materialize_train_only_derived_betas(variant, fold, train_panel, run_dir, beta_panel)
    discovered_cfg = variant_for_fold.get("discovered_hierarchy", {})
    if not discovered_cfg or not bool(discovered_cfg.get("enabled", False)):
        return variant_for_fold
    method = str(discovered_cfg.get("method", "residual_correlation")).lower()

    target_num_groups = int(discovered_cfg.get("target_num_groups", 6))
    min_group_size = int(discovered_cfg.get("min_group_size", 2))
    discovery_artifacts: dict[str, Any] = {}
    if method == "residual_correlation":
        group_indices, similarity, distance = discover_residual_correlation_groups(
            train_panel,
            target_num_groups=target_num_groups,
            min_group_size=min_group_size,
        )
        group_names = [f"rescorr_{idx:02d}" for idx in range(len(group_indices))]
        similarity_filename = "residual_correlation_matrix.csv"
        distance_filename = "residual_distance_matrix.csv"
        within_metric = "within_residual_corr_mean"
        between_metric = "between_residual_corr_mean"
    elif method in {"target_covariance", "targetcov", "target_weight_covariance"}:
        group_indices, similarity, distance, discovery_artifacts = discover_target_covariance_groups(
            train_panel,
            target_num_groups=target_num_groups,
            min_group_size=min_group_size,
            lookback=int(discovered_cfg.get("lookback", 20)),
            temperature=float(discovered_cfg.get("temperature", 0.35)),
            top_k=int(discovered_cfg.get("top_k", 0)),
            equal_weight_blend=float(discovered_cfg.get("equal_weight_blend", 0.10)),
            correlation_mode=str(discovered_cfg.get("correlation_mode", "positive")),
        )
        group_names = [f"targetcov_{idx:02d}" for idx in range(len(group_indices))]
        similarity_filename = "target_delta_similarity_matrix.csv"
        distance_filename = "target_delta_distance_matrix.csv"
        within_metric = "within_target_delta_similarity_mean"
        between_metric = "between_target_delta_similarity_mean"
        method = "target_covariance"
    else:
        raise ValueError(
            f"Unsupported discovered_hierarchy.method={method}. "
            "Supported methods: residual_correlation, target_covariance."
        )

    out_dir = run_dir / "discovered_hierarchy"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(similarity, index=train_panel.tickers, columns=train_panel.tickers).to_csv(out_dir / similarity_filename)
    pd.DataFrame(distance, index=train_panel.tickers, columns=train_panel.tickers).to_csv(out_dir / distance_filename)
    if discovery_artifacts:
        raw_corr = discovery_artifacts.get("raw_correlation")
        if isinstance(raw_corr, np.ndarray):
            pd.DataFrame(raw_corr, index=train_panel.tickers, columns=train_panel.tickers).to_csv(
                out_dir / "target_delta_raw_correlation_matrix.csv"
            )
        target_weights = discovery_artifacts.get("target_weights")
        if isinstance(target_weights, np.ndarray):
            pd.DataFrame(target_weights, index=train_panel.dates, columns=train_panel.tickers).to_csv(
                out_dir / "targetcov_proxy_target_weights.csv"
            )
        target_deltas = discovery_artifacts.get("target_deltas")
        if isinstance(target_deltas, np.ndarray) and len(train_panel.dates) > 1:
            pd.DataFrame(target_deltas, index=train_panel.dates[1:], columns=train_panel.tickers).to_csv(
                out_dir / "targetcov_proxy_target_deltas.csv"
            )
        meta = {
            "method": method,
            "target_num_groups": target_num_groups,
            "min_group_size": min_group_size,
            "lookback": int(discovered_cfg.get("lookback", 20)),
            "temperature": float(discovered_cfg.get("temperature", 0.35)),
            "top_k": int(discovered_cfg.get("top_k", 0)),
            "equal_weight_blend": float(discovered_cfg.get("equal_weight_blend", 0.10)),
            "correlation_mode": str(discovery_artifacts.get("correlation_mode", "positive")),
            "target_source": "train_only_momentum_vol_proxy",
            "note": "Proxy targets are causal within the train fold and are used only to freeze hierarchy topology.",
        }
        (out_dir / "targetcov_discovery_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    sector_map = get_sector_map(config.get("universe", {}).get("sector_map", "dow30_static"))
    rows = []
    for group_id, indices in enumerate(group_indices):
        for idx in indices:
            rows.append(
                {
                    "fold": str(fold["fold"]),
                    "method": method,
                    "cluster_id": group_id,
                    "cluster_name": group_names[group_id],
                    "ticker": train_panel.tickers[idx],
                    "ticker_index": idx,
                    "static_sector": sector_map[train_panel.tickers[idx]],
                }
            )
    assignments = pd.DataFrame(rows)
    assignments.to_csv(out_dir / "cluster_assignments.csv", index=False)

    summary_rows = []
    for group_id, indices in enumerate(group_indices):
        within = similarity[np.ix_(indices, indices)]
        if len(indices) > 1:
            within_values = within[np.triu_indices(len(indices), k=1)]
            within_similarity = float(np.mean(within_values)) if within_values.size else 1.0
        else:
            within_similarity = 1.0
        outside = [idx for idx in range(len(train_panel.tickers)) if idx not in set(indices)]
        between_similarity = float(np.mean(similarity[np.ix_(indices, outside)])) if outside else np.nan
        summary_rows.append(
            {
                "fold": str(fold["fold"]),
                "method": method,
                "cluster_id": group_id,
                "cluster_name": group_names[group_id],
                "size": len(indices),
                "tickers": " ".join(train_panel.tickers[idx] for idx in indices),
                "sectors": " ".join(sorted({sector_map[train_panel.tickers[idx]] for idx in indices})),
                "within_similarity_mean": within_similarity,
                "between_similarity_mean": between_similarity,
                within_metric: within_similarity,
                between_metric: between_similarity,
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "cluster_summary.csv", index=False)

    root_split = variant_for_fold.setdefault("root_split", {})
    root_split["group_indices"] = [[int(idx) for idx in group] for group in group_indices]
    root_split["group_names"] = group_names
    root_split["discovered_hierarchy"] = {
        "method": method,
        "target_num_groups": target_num_groups,
        "min_group_size": min_group_size,
        "fold": str(fold["fold"]),
        "artifact_dir": str(out_dir),
    }
    if method == "target_covariance":
        root_split["discovered_hierarchy"].update(
            {
                "lookback": int(discovered_cfg.get("lookback", 20)),
                "temperature": float(discovered_cfg.get("temperature", 0.35)),
                "top_k": int(discovered_cfg.get("top_k", 0)),
                "equal_weight_blend": float(discovered_cfg.get("equal_weight_blend", 0.10)),
                "correlation_mode": str(discovered_cfg.get("correlation_mode", "positive")),
                "target_source": "train_only_momentum_vol_proxy",
            }
        )
    return variant_for_fold


def train_one(
    config: dict[str, Any],
    variant: dict[str, Any],
    fold: pd.Series,
    *,
    out_root: Path,
    smoke_test: bool,
    force: bool,
) -> dict[str, Any]:
    variant_name = variant["name"]
    fold_id = str(fold["fold"])
    run_dir = out_root / variant_name / fold_id
    model_path = run_dir / "model.zip"
    summary_path = run_dir / "validation_summary.csv"
    if config.get("output", {}).get("skip_completed", True) and not force and model_path.exists() and summary_path.exists():
        summary = pd.read_csv(summary_path).iloc[0].to_dict()
        summary.update({"variant": variant_name, "fold": fold_id, "status": "skipped_existing"})
        return summary

    run_dir.mkdir(parents=True, exist_ok=True)
    feature_info = feature_csv_for_fold(config, variant, fold, out_root, force=force)
    feature_csv = feature_info["model_ready_csv"]
    train_panel = load_weight_panel(
        feature_csv,
        str(fold["train_start"]),
        str(fold["train_end_inclusive"]),
    )
    validation_panel = load_weight_panel(
        feature_csv,
        str(fold["validation_start"]),
        str(fold["validation_end_inclusive"]),
    )
    beta_panel = load_weight_panel(
        feature_csv,
        str(fold["train_start"]),
        str(fold["validation_end_inclusive"]),
    )
    variant_for_fold = materialize_variant_for_fold(config, variant, fold, train_panel, run_dir, beta_panel)

    train_env = make_vec_env(train_panel, config, variant_for_fold)
    policy, policy_kwargs = policy_for_variant(config, variant_for_fold, train_panel)
    ppo_cfg = ppo_kwargs(config, smoke_test, variant_for_fold)
    total_timesteps = int(ppo_cfg.pop("total_timesteps"))
    log_dir = run_dir / "sb3_logs"
    sb3_logger = configure(str(log_dir), ["stdout", "csv"])

    instr_cfg = config.get("instrumentation", {})
    model = InstrumentedPPO(
        policy,
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cpu",
        instrumentation_dir=run_dir / "training_diagnostics",
        rollout_dates=rollout_dates_for_variant(train_panel, variant_for_fold),
        save_sample_diagnostics=bool(instr_cfg.get("save_sample_diagnostics", True)),
        save_rollout_snapshots=bool(instr_cfg.get("save_rollout_snapshots", True)),
        rollout_snapshot_every_n_updates=int(instr_cfg.get("rollout_snapshot_every_n_updates", 25)),
        **ppo_cfg,
    )
    model.set_logger(sb3_logger)
    k_window_cfg = variant_for_fold.get("root_split", {}).get("k_window_execution", {})
    target_internal_days = k_window_cfg.get("target_internal_trading_days")
    internal_days_callback: InternalTradingDaysStopCallback | None = None
    if target_internal_days is not None:
        internal_days_callback = InternalTradingDaysStopCallback(
            target_internal_days=float(target_internal_days),
            verbose=1,
        )
    model.learn(total_timesteps=total_timesteps, callback=internal_days_callback, progress_bar=False)
    model.save(model_path)
    train_env.close()

    train_trace_summary = None
    if bool(k_window_cfg.get("log_train_daily_trace", False)):
        train_trace_summary = evaluate_model(model, train_panel, config, variant_for_fold, run_dir, "train_trace")
    validation_summary = evaluate_model(model, validation_panel, config, variant_for_fold, run_dir, "validation")
    metadata = {
        "variant": variant_for_fold,
        "fold": fold.to_dict(),
        "feature_info": {k: str(v) if isinstance(v, Path) else v for k, v in feature_info.items()},
        "model_path": str(model_path),
        "total_timesteps": total_timesteps,
        "sb3_num_timesteps": int(model.num_timesteps),
        "target_internal_trading_days": float(target_internal_days) if target_internal_days is not None else None,
        "observed_internal_trading_days": (
            float(internal_days_callback.internal_days_seen) if internal_days_callback is not None else None
        ),
        "internal_trading_days_stop_reached": (
            bool(internal_days_callback.stop_reached) if internal_days_callback is not None else None
        ),
        "train_trace_summary": train_trace_summary,
        "validation_summary": validation_summary,
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    result = {
        "variant": variant_name,
        "fold": fold_id,
        "status": "trained",
        "sb3_num_timesteps": int(model.num_timesteps),
        "target_internal_trading_days": float(target_internal_days) if target_internal_days is not None else np.nan,
        "observed_internal_trading_days": (
            float(internal_days_callback.internal_days_seen) if internal_days_callback is not None else np.nan
        ),
        **validation_summary,
        "model_path": str(model_path),
    }
    return result


def summarize_results(results: list[dict[str, Any]], out_root: Path) -> pd.DataFrame:
    df = pd.DataFrame(results)
    if df.empty:
        return df
    df.to_csv(out_root / "walk_forward_validation_results.csv", index=False)

    grouped = (
        df.groupby("variant", dropna=False)
        .agg(
            mean_validation_sharpe=("sharpe", "mean"),
            std_validation_sharpe=("sharpe", "std"),
            mean_validation_return=("return_pct", "mean"),
            mean_validation_drawdown=("max_drawdown", "mean"),
            mean_turnover_l1=("turnover_l1_mean", "mean"),
            mean_cash_weight=("cash_weight_mean", "mean"),
            mean_projection_residual=("projection_residual_l1_mean", "mean"),
            fold_count=("fold", "nunique"),
        )
        .reset_index()
    )
    grouped["std_validation_sharpe"] = grouped["std_validation_sharpe"].fillna(0.0)
    grouped["selection_score"] = grouped["mean_validation_sharpe"] - 0.5 * grouped["std_validation_sharpe"]
    grouped = grouped.sort_values("selection_score", ascending=False)
    grouped.to_csv(out_root / "walk_forward_variant_summary.csv", index=False)
    return grouped


def write_run_readme(config: dict[str, Any], summary: pd.DataFrame, out_root: Path) -> None:
    lines = [
        "# Stage 0.1 Weight-Based PPO Run",
        "",
        "This run trains stabilized PPO teachers with explicit portfolio weights over 29 stocks plus cash.",
        "",
        "## Top Variants",
        "",
    ]
    if summary.empty:
        lines.append("No completed variants yet.")
    else:
        view = summary.head(10).copy()
        columns = [
            "variant",
            "mean_validation_sharpe",
            "std_validation_sharpe",
            "selection_score",
            "mean_validation_return",
            "mean_turnover_l1",
            "mean_cash_weight",
            "fold_count",
        ]
        view = view[[c for c in columns if c in view.columns]]
        lines.append("| " + " | ".join(view.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(view.columns)) + " |")
        for _, row in view.iterrows():
            values = []
            for value in row.tolist():
                if isinstance(value, (float, np.floating)):
                    values.append(f"{float(value):.6f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `flat_dirichlet_*` uses a true Dirichlet policy over final simplex weights.",
            "- `hierarchical_dirichlet_*` uses a true Dirichlet-tree policy: root cash/sector Dirichlet plus within-sector Dirichlets.",
            "- `hierarchical_softmax_*` is disabled by default and kept only as a Gaussian-logit debug bridge.",
            "- Selection is validation-only. Frozen test must be run once after choosing a Stage 0.1 candidate.",
        ]
    )
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage0_1_stabilized_ppo.yaml")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--run-name", default=None, help="Override output.run_name without editing the YAML.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one short training job.")
    parser.add_argument("--force", action="store_true", help="Overwrite completed runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve(args.config))
    variants = selected_variants(config, args.variants)
    folds = load_folds(config, args.folds)

    if args.smoke_test:
        variants = variants[:1]
        folds = folds.head(1)

    out_cfg = config["output"]
    run_name = str(args.run_name or out_cfg["run_name"])
    if args.smoke_test:
        run_name = f"{run_name}_smoke"
    out_root = resolve(out_cfg["root_dir"]) / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve(args.config), out_root / "config.yaml")

    results: list[dict[str, Any]] = []
    for variant in variants:
        for _, fold in folds.iterrows():
            print(f"\n=== Stage 0.1: variant={variant['name']} fold={fold['fold']} ===")
            result = train_one(
                config,
                variant,
                fold,
                out_root=out_root,
                smoke_test=args.smoke_test,
                force=args.force,
            )
            print(json.dumps(result, indent=2, default=str))
            results.append(result)

    summary = summarize_results(results, out_root)
    write_run_readme(config, summary, out_root)
    print(f"\nStage 0.1 run written to {out_root}")
    if not summary.empty:
        print("\nTop validation variants:")
        print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
