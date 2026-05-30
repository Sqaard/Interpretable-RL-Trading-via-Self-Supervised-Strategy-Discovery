"""Weight-based portfolio environment for Stage 0.1 stabilized PPO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.data.dow30_sectors import get_sector_map
from src.ppo.execution.helpers import (
    EPS,
    cap_and_redistribute,
    deadzone_scale,
    normalize_simplex,
    normalize_stock_simplex,
    project_to_simplex,
    rank01,
    sigmoid_scalar,
    smoothstep,
    softmax,
)


@dataclass(frozen=True)
class WeightPanel:
    dates: np.ndarray
    tickers: list[str]
    feature_columns: list[str]
    features: np.ndarray
    prices: np.ndarray
    returns_next: np.ndarray


def load_weight_panel(csv_path: str | Path, start: str, end: str) -> WeightPanel:
    """Load a complete date/ticker panel and compute next-day returns."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    required = {"date", "tic", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    df = df.loc[mask].copy()
    if df.empty:
        raise ValueError(f"No rows in {path} for {start}..{end}")

    feature_columns = [c for c in df.columns if c not in {"date", "tic", "close"}]
    tickers = sorted(df["tic"].unique().tolist())
    dates = np.array(sorted(df["date"].unique()))

    expected_rows = len(dates) * len(tickers)
    if len(df) != expected_rows:
        counts = df.groupby("date")["tic"].nunique()
        bad_dates = counts[counts != len(tickers)].head(10).index.astype(str).tolist()
        raise ValueError(
            f"Panel is incomplete: rows={len(df)}, expected={expected_rows}, "
            f"bad_dates={bad_dates}"
        )

    price_panel = df.pivot(index="date", columns="tic", values="close").loc[dates, tickers]
    prices = price_panel.to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all():
        raise ValueError("Price panel contains NaN or inf values.")
    returns_next = prices[1:] / np.maximum(prices[:-1], EPS) - 1.0

    feature_arrays = []
    for col in feature_columns:
        arr = df.pivot(index="date", columns="tic", values=col).loc[dates, tickers].to_numpy(dtype=np.float64)
        feature_arrays.append(arr)
    features = np.stack(feature_arrays, axis=-1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Feature panel contains NaN or inf values.")

    return WeightPanel(
        dates=dates,
        tickers=tickers,
        feature_columns=feature_columns,
        features=features,
        prices=prices,
        returns_next=returns_next,
    )


class Stage01WeightPortfolioEnv(gym.Env):
    """Daily long-only portfolio allocation environment with explicit cash."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: WeightPanel,
        *,
        sector_map_name: str = "dow30_static",
        action_transform: str = "direct_weights",
        controller: dict[str, float | str] | None = None,
        root_split_config: dict[str, Any] | None = None,
        initial_amount: float = 1_000_000.0,
        transaction_cost_pct: float = 0.001,
        reward_config: dict[str, float] | None = None,
        reward_scale: float = 100.0,
        turnover_cap: float = 0.35,
        integral_clip: float = 0.50,
        derivative_clip: float = 0.50,
        include_features: bool = True,
        include_previous_weights: bool = True,
        include_portfolio_state: bool = True,
    ):
        super().__init__()
        self.panel = panel
        self.tickers = panel.tickers
        self.stock_dim = len(panel.tickers)
        self.asset_dim = self.stock_dim + 1
        self.cash_index = self.asset_dim - 1
        self.action_transform = action_transform
        self.initial_amount = float(initial_amount)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.reward_config = reward_config or {}
        self.reward_scale = float(reward_scale)
        self.turnover_cap = float(turnover_cap)
        self.integral_clip = float(integral_clip)
        self.derivative_clip = float(derivative_clip)
        self.include_features = include_features
        self.include_previous_weights = include_previous_weights
        self.include_portfolio_state = include_portfolio_state
        self.controller = {
            "type": "P",
            "kp": 1.0,
            "ki": 0.0,
            "kd": 0.0,
            **(controller or {}),
        }
        self.root_split_config = root_split_config or {}
        self.feature_index = {name: idx for idx, name in enumerate(panel.feature_columns)}
        self.derived_beta_schedule_dates: np.ndarray | None = None
        self.derived_beta_schedule_rows: list[dict[str, Any]] = []
        self._prepare_derived_beta_schedule()

        raw_sector_map = get_sector_map(sector_map_name)
        missing = sorted(set(self.tickers).difference(raw_sector_map))
        if missing:
            raise ValueError(f"Sector map {sector_map_name} misses tickers: {missing}")
        self.ticker_sectors = [raw_sector_map[t] for t in self.tickers]
        custom_group_indices = root_split_config.get("group_indices") if root_split_config else None
        custom_group_names = root_split_config.get("group_names") if root_split_config else None
        if custom_group_indices:
            stock_groups = [list(map(int, group)) for group in custom_group_indices]
            flat = sorted(idx for group in stock_groups for idx in group)
            if flat != list(range(self.stock_dim)):
                raise ValueError(
                    "Custom group_indices must partition stock indices exactly once. "
                    f"got={flat}, expected={list(range(self.stock_dim))}"
                )
            if custom_group_names and len(custom_group_names) == len(stock_groups):
                names = [str(name) for name in custom_group_names]
            else:
                names = [f"group_{idx:02d}" for idx in range(len(stock_groups))]
            self.group_names = ["cash"] + names
            self.group_to_indices = {"cash": [self.cash_index]}
            for name, indices in zip(names, stock_groups):
                self.group_to_indices[name] = indices
        else:
            self.group_names = ["cash"] + sorted(set(self.ticker_sectors))
            self.group_to_indices: dict[str, list[int]] = {"cash": [self.cash_index]}
            for group in self.group_names:
                if group == "cash":
                    continue
                self.group_to_indices[group] = [i for i, sec in enumerate(self.ticker_sectors) if sec == group]

        self.action_dim = self._action_dim()
        if action_transform in {
            "direct_weights",
            "root_split_weights",
            "root_split_kp_weights",
            "riskcash_sector_dirtree_factors",
            "style_mixture_weights",
        }:
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(self.action_dim,), dtype=np.float32)

        obs_dim = 0
        if include_features:
            obs_dim += self.stock_dim * len(panel.feature_columns)
        if include_previous_weights:
            obs_dim += self.asset_dim
        if include_portfolio_state:
            obs_dim += 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.day = 0
        self.portfolio_value = self.initial_amount
        self.peak_value = self.initial_amount
        self.previous_drawdown = 0.0
        self.previous_weights = np.zeros(self.asset_dim, dtype=np.float64)
        self.previous_weights[self.cash_index] = 1.0
        self.previous_error = np.zeros(self.asset_dim, dtype=np.float64)
        self.integral_error = np.zeros(self.asset_dim, dtype=np.float64)
        self.last_turnover = 0.0
        self.last_projection_residual = 0.0
        self.last_action_info: dict[str, float] = {}
        self.last_k_root_eff = 0.0
        self.last_k_inner_eff = 0.0
        self.previous_style_weights: np.ndarray | None = None
        self.cash_duration_days = 0
        self.cash_duration_threshold = 0.20
        self.early_update_cooldown_remaining = 0
        self.dual_root_anchor_q: float | None = None
        self.dual_root_anchor_start_day = 0
        self.dual_root_force_refresh = True
        self.internal_trading_days_processed = 0.0

    def _prepare_derived_beta_schedule(self) -> None:
        derived_cfg = self.root_split_config.get("derived_features", {})
        schedule = derived_cfg.get("beta_schedule", [])
        if not isinstance(schedule, list) or not schedule:
            return
        rows: list[dict[str, Any]] = []
        dates: list[np.datetime64] = []
        for row in schedule:
            if not isinstance(row, dict) or "date" not in row:
                continue
            try:
                date = np.datetime64(pd.Timestamp(row["date"]).to_datetime64(), "ns")
            except Exception:
                continue
            clean_row = dict(row)
            clean_row["date"] = str(pd.Timestamp(row["date"]).date())
            rows.append(clean_row)
            dates.append(date)
        if not rows:
            return
        order = np.argsort(np.asarray(dates, dtype="datetime64[ns]"))
        self.derived_beta_schedule_dates = np.asarray(dates, dtype="datetime64[ns]")[order]
        self.derived_beta_schedule_rows = [rows[int(idx)] for idx in order]

    def _current_derived_features(self) -> dict[str, Any]:
        derived_cfg = dict(self.root_split_config.get("derived_features", {}))
        if self.derived_beta_schedule_dates is None or not self.derived_beta_schedule_rows:
            return derived_cfg
        day_idx = min(max(int(self.day), 0), len(self.panel.dates) - 1)
        current_date = np.datetime64(pd.Timestamp(self.panel.dates[day_idx]).to_datetime64(), "ns")
        schedule_idx = int(np.searchsorted(self.derived_beta_schedule_dates, current_date, side="right") - 1)
        if schedule_idx < 0:
            return derived_cfg
        schedule_row = self.derived_beta_schedule_rows[schedule_idx]
        for key, value in schedule_row.items():
            if key == "date":
                continue
            derived_cfg[key] = value
        try:
            derived_cfg["beta_schedule_yyyymm"] = float(pd.Timestamp(schedule_row["date"]).strftime("%Y%m"))
        except Exception:
            pass
        return derived_cfg

    def _action_dim(self) -> int:
        if self.action_transform in {"direct_weights", "flat_softmax", "root_split_weights"}:
            return self.asset_dim
        if self.action_transform == "root_split_kp_weights":
            return self.asset_dim + 2
        if self.action_transform == "riskcash_sector_dirtree_factors":
            dim = 1 + (len(self.group_names) - 1)
            for group in self.group_names:
                if group == "cash":
                    continue
                group_size = len(self.group_to_indices[group])
                if group_size > 1:
                    dim += group_size
            return dim
        if self.action_transform == "style_mixture_weights":
            style_cfg = self.root_split_config.get("style_bank", {})
            styles = style_cfg.get("styles") or [
                "equal_weight",
                "risk_off_cash",
                "low_volatility",
                "momentum_20d",
                "momentum_60d",
                "short_reversal_5d",
                "sector_balanced",
                "cash_preservation",
            ]
            return len(styles)
        if self.action_transform == "hierarchical_softmax":
            stock_logits = sum(len(self.group_to_indices[g]) for g in self.group_names if g != "cash")
            return len(self.group_names) + stock_logits
        raise ValueError(f"Unknown action_transform: {self.action_transform}")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.day = 0
        self.portfolio_value = self.initial_amount
        self.peak_value = self.initial_amount
        self.previous_drawdown = 0.0
        self.previous_weights = np.zeros(self.asset_dim, dtype=np.float64)
        self.previous_weights[self.cash_index] = 1.0
        self.previous_error = np.zeros(self.asset_dim, dtype=np.float64)
        self.integral_error = np.zeros(self.asset_dim, dtype=np.float64)
        self.last_turnover = 0.0
        self.last_projection_residual = 0.0
        self.last_action_info = {}
        self.last_k_root_eff = 0.0
        self.last_k_inner_eff = 0.0
        self.previous_style_weights = None
        self.cash_duration_days = 0
        self.cash_duration_threshold = 0.20
        self.early_update_cooldown_remaining = 0
        self.dual_root_anchor_q = None
        self.dual_root_anchor_start_day = 0
        self.dual_root_force_refresh = True
        self.internal_trading_days_processed = 0.0
        return self._get_obs(), self._info_base()

    def step(self, action: np.ndarray):
        k_window_cfg = self.root_split_config.get("k_window_execution", {})
        if k_window_cfg and bool(k_window_cfg.get("enabled", False)):
            return self._step_k_window(action, k_window_cfg)

        target_weights = self._action_to_target_weights(action)
        reward, terminated, info = self._step_daily_target(target_weights)
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, False, info

    def _step_daily_target(
        self,
        target_weights: np.ndarray,
        *,
        action_info: dict[str, Any] | None = None,
    ) -> tuple[float, bool, dict[str, Any]]:
        target_weights = normalize_simplex(target_weights)
        pre_trade_weights = self.previous_weights.copy()
        executed_weights, controller_terms = self._apply_controller(target_weights)

        trade_delta_weights = executed_weights - pre_trade_weights
        trade_abs_weights = np.abs(trade_delta_weights)
        trade_direction_eps = float(self.root_split_config.get("trade_logging", {}).get("direction_eps", 1e-6))
        trade_direction = np.where(
            trade_delta_weights > trade_direction_eps,
            1.0,
            np.where(trade_delta_weights < -trade_direction_eps, -1.0, 0.0),
        )

        stock_turnover_l1 = float(np.sum(np.abs(executed_weights[: self.stock_dim] - pre_trade_weights[: self.stock_dim])))
        turnover_l1 = float(np.sum(np.abs(executed_weights - pre_trade_weights)))
        transaction_cost = self.transaction_cost_pct * stock_turnover_l1

        asset_returns = self.panel.returns_next[self.day]
        gross_return = float(np.dot(executed_weights[: self.stock_dim], asset_returns))
        net_return = (1.0 - transaction_cost) * (1.0 + gross_return) - 1.0

        old_value = self.portfolio_value
        self.portfolio_value *= 1.0 + net_return
        self.peak_value = max(self.peak_value, self.portfolio_value)
        current_drawdown = self.portfolio_value / max(self.peak_value, EPS) - 1.0
        drawdown_increment = max(0.0, self.previous_drawdown - current_drawdown)

        concentration = float(np.sum(executed_weights**2))
        action_change = turnover_l1
        risk_score, cash_allowed, excess_cash, cash_penalty = self._cash_prior_terms(executed_weights[self.cash_index])
        execution_penalty = float(controller_terms.get("execution_penalty", 0.0))
        reward = self._compute_reward(
            net_return=net_return,
            turnover=turnover_l1,
            drawdown_increment=drawdown_increment,
            concentration=concentration,
            action_change=action_change,
            extra_penalty=cash_penalty + execution_penalty,
        )

        next_weights = self._weights_after_market_move(executed_weights, asset_returns)
        self.previous_weights = next_weights
        if float(next_weights[self.cash_index]) > self.cash_duration_threshold:
            self.cash_duration_days += 1
        else:
            self.cash_duration_days = 0
        self.previous_drawdown = current_drawdown
        self.last_turnover = turnover_l1
        self.day += 1

        terminated = self.day >= len(self.panel.dates) - 1
        action_terms = self.last_action_info if action_info is None else action_info
        info = {
            **self._info_base(),
            "date": str(pd.Timestamp(self.panel.dates[self.day - 1]).date()),
            "next_date": str(pd.Timestamp(self.panel.dates[self.day]).date()) if self.day < len(self.panel.dates) else "",
            "portfolio_value_before": old_value,
            "portfolio_value": self.portfolio_value,
            "gross_return": gross_return,
            "net_return": net_return,
            "reward": float(reward),
            "reward_unscaled": reward / self.reward_scale,
            "risk_score": risk_score,
            "cash_allowed": cash_allowed,
            "excess_cash": excess_cash,
            "cash_prior_penalty": cash_penalty,
            "turnover_l1": turnover_l1,
            "stock_turnover_l1": stock_turnover_l1,
            "transaction_cost": transaction_cost,
            "drawdown": current_drawdown,
            "drawdown_increment": drawdown_increment,
            "concentration": concentration,
            "target_weights": target_weights.astype(np.float32),
            "executed_weights": executed_weights.astype(np.float32),
            "post_market_weights": next_weights.astype(np.float32),
            "pre_trade_weights": pre_trade_weights.astype(np.float32),
            "trade_delta_weights": trade_delta_weights.astype(np.float32),
            "trade_abs_weights": trade_abs_weights.astype(np.float32),
            "trade_direction": trade_direction.astype(np.float32),
            "trade_buy_count": float(np.sum(trade_direction[: self.stock_dim] > 0.0)),
            "trade_sell_count": float(np.sum(trade_direction[: self.stock_dim] < 0.0)),
            "trade_hold_count": float(np.sum(trade_direction[: self.stock_dim] == 0.0)),
            "trade_buy_weight_l1": float(np.sum(np.maximum(trade_delta_weights[: self.stock_dim], 0.0))),
            "trade_sell_weight_l1": float(np.sum(np.maximum(-trade_delta_weights[: self.stock_dim], 0.0))),
            "cash_trade_delta": float(trade_delta_weights[self.cash_index]),
            "cash_trade_direction": float(trade_direction[self.cash_index]),
            "target_to_executed_l1": float(np.sum(np.abs(target_weights - executed_weights))),
            **controller_terms,
            **action_terms,
        }
        return float(reward), terminated, info

    def _conditional_risky_allocation(self, weights: np.ndarray) -> np.ndarray:
        q = float(np.sum(weights[: self.stock_dim]))
        if q <= EPS:
            return np.full(self.stock_dim, 1.0 / max(self.stock_dim, 1), dtype=np.float64)
        return normalize_stock_simplex(weights[: self.stock_dim] / q)

    def _anchor_risky_from_action(self, action: np.ndarray, anchor_weights: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if self.action_transform in {"root_split_weights", "root_split_kp_weights"} and a.size >= 1 + self.stock_dim:
            return normalize_stock_simplex(a[1 : 1 + self.stock_dim])
        return self._conditional_risky_allocation(anchor_weights)

    def _weights_from_root_risky(self, q: float, risky: np.ndarray) -> np.ndarray:
        q_safe = float(np.clip(q, 0.0, 1.0))
        u_safe = normalize_stock_simplex(risky)
        out = np.zeros(self.asset_dim, dtype=np.float64)
        out[: self.stock_dim] = q_safe * u_safe
        out[self.cash_index] = 1.0 - q_safe
        return normalize_simplex(out)

    def _apply_sparse_topk_risky_target(
        self,
        anchor_weights: np.ndarray,
        u_anchor: np.ndarray,
        u_window_start: np.ndarray,
        cfg: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if not cfg or not bool(cfg.get("enabled", False)):
            return anchor_weights, u_anchor, {"topk_sparse_enabled": 0.0}

        top_k = max(1, min(int(cfg.get("top_k", 5)), self.stock_dim))
        eps = float(cfg.get("eps", 1e-12))
        u_raw = normalize_stock_simplex(u_anchor)
        u_start = normalize_stock_simplex(u_window_start)
        q_anchor = float(np.sum(anchor_weights[: self.stock_dim]))

        selection_mode = str(cfg.get("selection_mode", "dirichlet_potential")).lower()
        if selection_mode in {"anchor", "u_anchor", "dirichlet_weight"}:
            priority = np.array(u_raw, copy=True)
        elif selection_mode in {"absolute_delta", "abs_delta"}:
            priority = np.abs(u_raw - u_start)
        else:
            # Dirichlet potential: names where the sampled policy target wants
            # more conditional risky weight than the K-window start allocation.
            priority = np.maximum(u_raw - u_start, 0.0)

        selected: list[int] = []
        positive_order = np.argsort(priority)[::-1]
        for idx in positive_order:
            if len(selected) >= top_k:
                break
            if priority[idx] > eps:
                selected.append(int(idx))

        if bool(cfg.get("fill_with_anchor", True)) and len(selected) < top_k:
            for idx in np.argsort(u_raw)[::-1]:
                if len(selected) >= top_k:
                    break
                if int(idx) not in selected:
                    selected.append(int(idx))

        if not selected:
            selected = [int(idx) for idx in np.argsort(u_raw)[-top_k:]]

        selected_arr = np.asarray(selected[:top_k], dtype=int)
        selected_mask = np.zeros(self.stock_dim, dtype=bool)
        selected_mask[selected_arr] = True

        allocation_source = str(cfg.get("allocation_source", "anchor")).lower()
        if allocation_source in {"priority", "potential"}:
            allocation_values = np.maximum(priority[selected_arr], 0.0)
            if float(np.sum(allocation_values)) <= eps:
                allocation_values = np.maximum(u_raw[selected_arr], 0.0)
        else:
            # This is the "renormalize u_target inside top-K" behavior:
            # the original Dirichlet sampled risky weights define relative
            # allocation inside the selected sparse subset.
            allocation_values = np.maximum(u_raw[selected_arr], 0.0)
        allocation_values = normalize_stock_simplex(allocation_values)

        u_sparse = np.zeros(self.stock_dim, dtype=np.float64)
        u_sparse[selected_arr] = allocation_values
        sparse_anchor_weights = self._weights_from_root_risky(q_anchor, u_sparse)

        rebuild_l1 = float(np.sum(np.abs(sparse_anchor_weights - anchor_weights)))
        terms: dict[str, Any] = {
            "topk_sparse_enabled": 1.0,
            "topk_sparse_k": float(top_k),
            "topk_sparse_selected_count": float(np.sum(selected_mask)),
            "topk_sparse_tickers": "|".join(self.tickers[idx] for idx in selected_arr),
            "topk_sparse_anchor_rebuild_l1": rebuild_l1,
            "topk_sparse_anchor_rebuild_turnover": 0.5 * rebuild_l1,
            "topk_sparse_target_hhi": float(np.sum(u_sparse**2)),
            "topk_sparse_target_max_weight": float(np.max(u_sparse)) if u_sparse.size else 0.0,
            "topk_sparse_target_entropy": float(-np.sum(u_sparse * np.log(np.maximum(u_sparse, EPS)))),
            "topk_sparse_positive_candidate_count": float(np.sum(priority > eps)),
            "topk_sparse_q_anchor": q_anchor,
        }
        for idx, ticker in enumerate(self.tickers):
            terms[f"topk_sparse_priority_{ticker}"] = float(priority[idx])
            terms[f"topk_sparse_selected_{ticker}"] = float(selected_mask[idx])
            terms[f"topk_sparse_u_anchor_raw_{ticker}"] = float(u_raw[idx])
            terms[f"topk_sparse_u_anchor_final_{ticker}"] = float(u_sparse[idx])
        return sparse_anchor_weights, u_sparse, terms

    def _priority_order(
        self,
        priority: np.ndarray,
        fallback: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        primary = np.nan_to_num(np.asarray(priority, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        secondary = np.nan_to_num(np.asarray(fallback, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        positive = np.where(primary > eps)[0]
        nonpositive = np.where(primary <= eps)[0]
        positive = positive[np.argsort(primary[positive])[::-1]]
        nonpositive = nonpositive[np.argsort(secondary[nonpositive])[::-1]]
        return np.concatenate([positive, nonpositive]).astype(int)

    def _allocate_capped_flow(
        self,
        total: float,
        selected: np.ndarray,
        preference: np.ndarray,
        capacity: np.ndarray,
        eps: float,
    ) -> tuple[np.ndarray, float]:
        allocation = np.zeros(self.stock_dim, dtype=np.float64)
        remaining = max(float(total), 0.0)
        if remaining <= eps or selected.size == 0:
            return allocation, remaining

        selected = np.asarray(selected, dtype=int)
        capacity_by_stock = np.zeros(self.stock_dim, dtype=np.float64)
        capacity_by_stock[selected] = np.maximum(np.asarray(capacity, dtype=np.float64), 0.0)
        preference_by_stock = np.zeros(self.stock_dim, dtype=np.float64)
        preference_by_stock[selected] = np.maximum(np.asarray(preference, dtype=np.float64), 0.0)

        for _ in range(selected.size + 1):
            cap_left = np.maximum(capacity_by_stock - allocation, 0.0)
            active = selected[cap_left[selected] > eps]
            if remaining <= eps or active.size == 0:
                break
            pref = preference_by_stock[active]
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

    def _apply_incremental_topk_flow_target(
        self,
        target_weights: np.ndarray,
        u_anchor: np.ndarray,
        u_window_start: np.ndarray,
        cfg: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not cfg or not bool(cfg.get("enabled", False)):
            return target_weights, {"incremental_topk_enabled": 0.0}

        target = normalize_simplex(target_weights)
        prev = normalize_simplex(self.previous_weights)
        q_prev = float(np.sum(prev[: self.stock_dim]))
        q_target = float(np.sum(target[: self.stock_dim]))
        delta_q = q_target - q_prev
        eps = float(cfg.get("eps", 1e-10))
        top_k_buy = max(0, min(int(cfg.get("top_k_buy", cfg.get("top_k", 5))), self.stock_dim))
        top_k_sell = max(0, min(int(cfg.get("top_k_sell", cfg.get("top_k", 5))), self.stock_dim))

        u_current = self._conditional_risky_allocation(prev)
        u_start = normalize_stock_simplex(np.asarray(u_window_start, dtype=np.float64))
        u_anchor_norm = normalize_stock_simplex(np.asarray(u_anchor, dtype=np.float64))
        reference_mode = str(cfg.get("priority_reference", "window_start")).lower()
        u_ref = u_current if reference_mode in {"current", "previous", "prev"} else u_start
        context = context or {}

        buy_priority = np.maximum(u_anchor_norm - u_ref, 0.0)
        sell_priority = np.maximum(u_ref - u_anchor_norm, 0.0)
        group_terms: dict[str, Any] = {}
        group_cfg = cfg.get("group_aware", {})
        if group_cfg and bool(group_cfg.get("enabled", False)):
            buy_priority, sell_priority, group_terms = self._apply_group_aware_topk_priority(
                buy_priority=buy_priority,
                sell_priority=sell_priority,
                prev=prev,
                target=target,
                cfg=group_cfg,
                eps=eps,
            )
        risk_aware_terms: dict[str, Any] = {}
        risk_aware_cfg = cfg.get("risk_aware", {})
        risk_aware_enabled = bool(risk_aware_cfg.get("enabled", False))
        risk_stress_ctx = float(context.get("risk_stress", 0.0))
        recovery_score_ctx = float(context.get("recovery_score", 1.0))
        confidence_rerisk_ctx = float(context.get("confidence_rerisk", 1.0))
        confidence_derisk_ctx = float(context.get("confidence_derisk", 1.0))
        residual_breadth_excess_5d = self._feature_value("residual_breadth_excess_5d", default=0.0)
        residual_breadth_excess_20d = self._feature_value("residual_breadth_excess_20d", default=0.0)
        risk_break_signal = float(
            max(
                float(context.get("risk_break_trigger", 0.0)),
                float(context.get("derisk_early_update", 0.0)),
                float(context.get("risk_break_trigger_candidate", 0.0)),
            )
        )
        buy_allowed = 1.0
        buy_gate_binary_allowed = 1.0
        buy_gate_soft_score = 1.0
        buy_gate_hard_block = 0.0
        buy_gate_reason = "not_checked"
        rotation_stress_gate = 1.0
        sell_multiplier = np.ones(self.stock_dim, dtype=np.float64)
        residual_deterioration = np.zeros(self.stock_dim, dtype=np.float64)
        if risk_aware_enabled:
            risk_aware_terms.update(
                {
                    "incremental_topk_risk_aware_enabled": 1.0,
                    "incremental_topk_risk_stress": risk_stress_ctx,
                    "incremental_topk_recovery_score": recovery_score_ctx,
                    "incremental_topk_confidence_rerisk": confidence_rerisk_ctx,
                    "incremental_topk_confidence_derisk": confidence_derisk_ctx,
                    "incremental_topk_residual_breadth_excess_5d": residual_breadth_excess_5d,
                    "incremental_topk_residual_breadth_excess_20d": residual_breadth_excess_20d,
                    "incremental_topk_risk_break_signal": risk_break_signal,
                }
            )

            buy_gate_cfg = risk_aware_cfg.get("buy_gate", {})
            if bool(buy_gate_cfg.get("enabled", False)):
                min_conf_rerisk = float(buy_gate_cfg.get("min_confidence_rerisk", 0.0))
                min_recovery = float(buy_gate_cfg.get("min_recovery_score", 0.0))
                max_risk_stress = float(buy_gate_cfg.get("max_risk_stress", np.inf))
                min_breadth_5d = float(buy_gate_cfg.get("min_residual_breadth_excess_5d", -np.inf))
                min_breadth_20d = float(buy_gate_cfg.get("min_residual_breadth_excess_20d", -np.inf))
                checks = {
                    "conf_rerisk": confidence_rerisk_ctx >= min_conf_rerisk,
                    "recovery": recovery_score_ctx >= min_recovery,
                    "risk_stress": risk_stress_ctx <= max_risk_stress,
                    "breadth_5d": residual_breadth_excess_5d >= min_breadth_5d,
                    "breadth_20d": residual_breadth_excess_20d >= min_breadth_20d,
                }
                failed = [name for name, passed in checks.items() if not passed]
                buy_gate_binary_allowed = 0.0 if failed else 1.0
                buy_allowed = buy_gate_binary_allowed
                buy_gate_reason = "allowed" if not failed else "|".join(failed)
                for name, passed in checks.items():
                    risk_aware_terms[f"incremental_topk_buy_gate_pass_{name}"] = float(passed)

                soft_cfg = buy_gate_cfg.get("soft_scale", {})
                if bool(soft_cfg.get("enabled", False)):
                    min_scale = float(np.clip(float(soft_cfg.get("min_scale", 0.15)), 0.0, 1.0))
                    confidence_margin = max(float(soft_cfg.get("confidence_margin", 0.15)), eps)
                    recovery_margin = max(float(soft_cfg.get("recovery_margin", 0.15)), eps)
                    breadth_margin = max(float(soft_cfg.get("breadth_margin", 0.05)), eps)
                    stress_margin = max(float(soft_cfg.get("stress_margin", 0.10)), eps)

                    def lower_bound_score(value: float, threshold: float, margin: float) -> float:
                        if not np.isfinite(threshold):
                            return 1.0
                        return float(np.clip((value - (threshold - margin)) / margin, 0.0, 1.0))

                    def upper_bound_score(value: float, threshold: float, margin: float) -> float:
                        if not np.isfinite(threshold):
                            return 1.0
                        return float(np.clip(((threshold + margin) - value) / margin, 0.0, 1.0))

                    score_components = {
                        "conf_rerisk": lower_bound_score(confidence_rerisk_ctx, min_conf_rerisk, confidence_margin),
                        "recovery": lower_bound_score(recovery_score_ctx, min_recovery, recovery_margin),
                        "risk_stress": upper_bound_score(risk_stress_ctx, max_risk_stress, stress_margin),
                        "breadth_5d": lower_bound_score(residual_breadth_excess_5d, min_breadth_5d, breadth_margin),
                        "breadth_20d": lower_bound_score(residual_breadth_excess_20d, min_breadth_20d, breadth_margin),
                    }
                    combine = str(soft_cfg.get("combine", "min")).lower()
                    if combine == "mean":
                        buy_gate_soft_score = float(np.mean(list(score_components.values())))
                    else:
                        buy_gate_soft_score = float(min(score_components.values()))
                    buy_allowed = min_scale + (1.0 - min_scale) * buy_gate_soft_score

                    hard_block_risk_break = bool(soft_cfg.get("hard_block_risk_break", True))
                    hard_block_risk_stress = float(soft_cfg.get("hard_block_risk_stress", np.inf))
                    if (hard_block_risk_break and risk_break_signal > 0.0) or (
                        np.isfinite(hard_block_risk_stress) and risk_stress_ctx >= hard_block_risk_stress
                    ):
                        buy_allowed = 0.0
                        buy_gate_hard_block = 1.0

                    if buy_gate_hard_block > 0.0:
                        buy_gate_reason = "hard_block"
                    elif failed:
                        buy_gate_reason = "soft_scaled:" + "|".join(failed)

                    for name, value in score_components.items():
                        risk_aware_terms[f"incremental_topk_buy_gate_soft_component_{name}"] = value

            rotation_gate_cfg = risk_aware_cfg.get("rotation_stress_gate", {})
            if bool(rotation_gate_cfg.get("enabled", False)):
                stress_start = float(rotation_gate_cfg.get("stress_start", 0.55))
                stress_full = float(rotation_gate_cfg.get("stress_full", 0.90))
                min_scale = float(rotation_gate_cfg.get("min_scale", 0.0))
                max_scale = float(rotation_gate_cfg.get("max_scale", 1.0))
                denom = max(stress_full - stress_start, eps)
                raw_gate = 1.0 - (risk_stress_ctx - stress_start) / denom
                rotation_stress_gate = float(np.clip(raw_gate, min_scale, max_scale))

            sell_cfg = risk_aware_cfg.get("sell_side", {})
            risk_break_weight = float(sell_cfg.get("risk_break_weight", 0.0))
            residual_deterioration_weight = float(sell_cfg.get("residual_deterioration_weight", 0.0))
            confidence_derisk_weight = float(sell_cfg.get("confidence_derisk_weight", 0.0))
            if abs(residual_deterioration_weight) > eps:
                derived_cfg = self._current_derived_features()
                residual_deterioration = np.maximum(
                    self._stock_confidence_feature_vector("negative_residual_momentum_5d", derived_cfg),
                    0.0,
                )
                residual_deterioration = np.nan_to_num(
                    residual_deterioration,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            sell_multiplier = (
                1.0
                + risk_break_weight * risk_break_signal
                + residual_deterioration_weight * residual_deterioration
                + confidence_derisk_weight * confidence_derisk_ctx
            )
            sell_multiplier = np.clip(
                sell_multiplier,
                float(sell_cfg.get("min_multiplier", 0.05)),
                float(sell_cfg.get("max_multiplier", 5.0)),
            )
            sell_priority *= sell_multiplier
        else:
            risk_aware_terms["incremental_topk_risk_aware_enabled"] = 0.0

        risk_aware_terms.update(
            {
                "incremental_topk_buy_allowed": buy_allowed,
                "incremental_topk_buy_fill_scale": buy_allowed,
                "incremental_topk_buy_gate_binary_allowed": buy_gate_binary_allowed,
                "incremental_topk_buy_gate_soft_score": buy_gate_soft_score,
                "incremental_topk_buy_gate_hard_block": buy_gate_hard_block,
                "incremental_topk_buy_gate_reason": buy_gate_reason,
                "incremental_topk_rotation_stress_gate": rotation_stress_gate,
                "incremental_topk_sell_multiplier_mean": float(np.mean(sell_multiplier)),
                "incremental_topk_sell_multiplier_p90": float(np.percentile(sell_multiplier, 90)),
                "incremental_topk_residual_deterioration_mean": float(np.mean(residual_deterioration)),
                "incremental_topk_residual_deterioration_p90": float(np.percentile(residual_deterioration, 90)),
            }
        )
        buy_order = self._priority_order(buy_priority, u_anchor_norm, eps)
        sell_order = self._priority_order(sell_priority, prev[: self.stock_dim], eps)

        out = prev.copy()
        selected_buy = np.zeros(self.stock_dim, dtype=bool)
        selected_sell = np.zeros(self.stock_dim, dtype=bool)
        flow_delta = np.zeros(self.stock_dim, dtype=np.float64)
        buy_requested = max(delta_q, 0.0)
        sell_requested = max(-delta_q, 0.0)
        buy_filled = 0.0
        sell_filled = 0.0
        buy_unfilled = 0.0
        sell_unfilled = 0.0
        sell_expansion_count = 0
        sell_final_k = 0
        direction = "hold"
        rotation_enabled = bool(cfg.get("rotation_enabled", False))
        rotation_budget = max(float(cfg.get("rotation_budget_l1_per_day", cfg.get("rotation_budget", 0.0))), 0.0)
        rotation_requested = 0.0
        rotation_sell_filled = 0.0
        rotation_buy_filled = 0.0
        rotation_unfilled = 0.0
        rotation_selected_buy = np.zeros(self.stock_dim, dtype=bool)
        rotation_selected_sell = np.zeros(self.stock_dim, dtype=bool)

        if buy_requested > eps and top_k_buy > 0 and buy_allowed > 0.0:
            direction = "rerisk"
            buy_amount = min(buy_requested * buy_allowed, float(prev[self.cash_index]))
            selected = buy_order[: min(top_k_buy, buy_order.size)]
            selected_buy[selected] = True
            preference = buy_priority[selected]
            if float(np.sum(preference)) <= eps:
                preference = u_anchor_norm[selected]
            if float(np.sum(preference)) <= eps:
                preference = np.ones(selected.size, dtype=np.float64)
            allocation = buy_amount * normalize_stock_simplex(preference)
            stocks = out[: self.stock_dim].copy()
            stocks[selected] += allocation
            out[: self.stock_dim] = stocks
            flow_delta[selected] += allocation
            buy_filled = float(np.sum(allocation))
            buy_unfilled = max(buy_requested - buy_filled, 0.0)

        elif buy_requested > eps and top_k_buy > 0 and buy_allowed <= 0.0:
            direction = "rerisk_blocked"
            buy_unfilled = buy_requested

        elif sell_requested > eps and top_k_sell > 0:
            direction = "derisk"
            sell_amount = min(sell_requested, q_prev)
            initial_k = min(top_k_sell, sell_order.size)
            selected_count = initial_k
            selected = sell_order[:selected_count]
            selected_capacity = float(np.sum(prev[: self.stock_dim][selected]))
            if bool(cfg.get("sell_expansion_enabled", True)):
                while selected_capacity + eps < sell_amount and selected_count < sell_order.size:
                    selected_count += 1
                    selected = sell_order[:selected_count]
                    selected_capacity = float(np.sum(prev[: self.stock_dim][selected]))
                sell_expansion_count = max(0, selected_count - initial_k)
            sell_final_k = int(selected_count)
            selected_sell[selected] = True
            preference = sell_priority[selected]
            if float(np.sum(preference)) <= eps:
                preference = prev[: self.stock_dim][selected]
            allocation, sell_unfilled = self._allocate_capped_flow(
                sell_amount,
                selected,
                preference,
                prev[: self.stock_dim][selected],
                eps,
            )
            out[: self.stock_dim] = np.maximum(out[: self.stock_dim] - allocation, 0.0)
            flow_delta -= allocation
            sell_filled = float(np.sum(allocation))

        rotation_budget_eff = rotation_budget * rotation_stress_gate
        if (
            rotation_enabled
            and direction != "derisk"
            and rotation_budget_eff > eps
            and top_k_buy > 0
            and top_k_sell > 0
        ):
            rotation_buy_order = self._priority_order(buy_priority, u_anchor_norm, eps)
            rotation_sell_order = self._priority_order(sell_priority, out[: self.stock_dim], eps)
            buy_candidates = rotation_buy_order[buy_priority[rotation_buy_order] > eps]
            sell_candidates = rotation_sell_order[sell_priority[rotation_sell_order] > eps]
            if buy_candidates.size > 0 and sell_candidates.size > 0:
                selected_buy_rotation = buy_candidates[: min(top_k_buy, buy_candidates.size)]
                selected_sell_rotation = sell_candidates[: min(top_k_sell, sell_candidates.size)]
                rotation_selected_buy[selected_buy_rotation] = True
                rotation_selected_sell[selected_sell_rotation] = True
                sell_capacity = float(np.sum(out[: self.stock_dim][selected_sell_rotation]))
                rotation_requested = min(rotation_budget_eff, sell_capacity)
                if rotation_requested > eps:
                    sell_preference = sell_priority[selected_sell_rotation]
                    if float(np.sum(sell_preference)) <= eps:
                        sell_preference = out[: self.stock_dim][selected_sell_rotation]
                    sell_allocation, rotation_unfilled = self._allocate_capped_flow(
                        rotation_requested,
                        selected_sell_rotation,
                        sell_preference,
                        out[: self.stock_dim][selected_sell_rotation],
                        eps,
                    )
                    rotation_sell_filled = float(np.sum(sell_allocation))
                    out[: self.stock_dim] = np.maximum(out[: self.stock_dim] - sell_allocation, 0.0)
                    flow_delta -= sell_allocation
                    if rotation_sell_filled > eps:
                        buy_preference = buy_priority[selected_buy_rotation]
                        if float(np.sum(buy_preference)) <= eps:
                            buy_preference = u_anchor_norm[selected_buy_rotation]
                        if float(np.sum(buy_preference)) <= eps:
                            buy_preference = np.ones(selected_buy_rotation.size, dtype=np.float64)
                        buy_allocation = rotation_sell_filled * normalize_stock_simplex(buy_preference)
                        stocks_after_rotation = out[: self.stock_dim].copy()
                        stocks_after_rotation[selected_buy_rotation] += buy_allocation
                        out[: self.stock_dim] = stocks_after_rotation
                        flow_delta[selected_buy_rotation] += buy_allocation
                        rotation_buy_filled = float(np.sum(buy_allocation))
                    if direction == "hold":
                        direction = "rotation"
                    elif rotation_sell_filled > eps:
                        direction = f"{direction}+rotation"

        stock_sum = float(np.sum(out[: self.stock_dim]))
        if stock_sum > 1.0:
            out[: self.stock_dim] = normalize_stock_simplex(out[: self.stock_dim])
            stock_sum = 1.0
        out[self.cash_index] = max(0.0, 1.0 - stock_sum)
        out = normalize_simplex(out)

        terms: dict[str, Any] = {
            "incremental_topk_enabled": 1.0,
            "incremental_topk_direction": direction,
            "incremental_topk_top_k_buy": float(top_k_buy),
            "incremental_topk_top_k_sell": float(top_k_sell),
            "incremental_topk_q_prev": q_prev,
            "incremental_topk_q_target_in": q_target,
            "incremental_topk_delta_q": delta_q,
            "incremental_topk_buy_requested": buy_requested,
            "incremental_topk_buy_filled": buy_filled,
            "incremental_topk_buy_unfilled": buy_unfilled,
            "incremental_topk_sell_requested": sell_requested,
            "incremental_topk_sell_filled": sell_filled,
            "incremental_topk_sell_unfilled": sell_unfilled,
            "incremental_topk_sell_expansion_count": float(sell_expansion_count),
            "incremental_topk_sell_final_k": float(sell_final_k),
            "incremental_topk_selected_buy_count": float(np.sum(selected_buy)),
            "incremental_topk_selected_sell_count": float(np.sum(selected_sell)),
            "incremental_topk_rotation_enabled": 1.0 if rotation_enabled else 0.0,
            "incremental_topk_rotation_budget_l1_per_day": rotation_budget,
            "incremental_topk_rotation_budget_effective": rotation_budget_eff,
            "incremental_topk_rotation_requested": rotation_requested,
            "incremental_topk_rotation_sell_filled": rotation_sell_filled,
            "incremental_topk_rotation_buy_filled": rotation_buy_filled,
            "incremental_topk_rotation_unfilled": rotation_unfilled,
            "incremental_topk_rotation_selected_buy_count": float(np.sum(rotation_selected_buy)),
            "incremental_topk_rotation_selected_sell_count": float(np.sum(rotation_selected_sell)),
            "incremental_topk_rotation_buy_tickers": "|".join(
                self.tickers[idx] for idx in np.where(rotation_selected_buy)[0]
            ),
            "incremental_topk_rotation_sell_tickers": "|".join(
                self.tickers[idx] for idx in np.where(rotation_selected_sell)[0]
            ),
            "incremental_topk_buy_tickers": "|".join(self.tickers[idx] for idx in np.where(selected_buy)[0]),
            "incremental_topk_sell_tickers": "|".join(self.tickers[idx] for idx in np.where(selected_sell)[0]),
            "incremental_topk_input_to_output_l1": float(np.sum(np.abs(target - out))),
            "incremental_topk_flow_l1": float(np.sum(np.abs(flow_delta))),
            "incremental_topk_flow_turnover": 0.5 * float(np.sum(np.abs(flow_delta))),
            "incremental_topk_cash_after": float(out[self.cash_index]),
            **group_terms,
            **risk_aware_terms,
        }
        for idx, ticker in enumerate(self.tickers):
            terms[f"incremental_topk_buy_priority_{ticker}"] = float(buy_priority[idx])
            terms[f"incremental_topk_sell_priority_{ticker}"] = float(sell_priority[idx])
            terms[f"incremental_topk_sell_multiplier_{ticker}"] = float(sell_multiplier[idx])
            terms[f"incremental_topk_residual_deterioration_{ticker}"] = float(residual_deterioration[idx])
            terms[f"incremental_topk_selected_buy_{ticker}"] = float(selected_buy[idx])
            terms[f"incremental_topk_selected_sell_{ticker}"] = float(selected_sell[idx])
            terms[f"incremental_topk_rotation_selected_buy_{ticker}"] = float(rotation_selected_buy[idx])
            terms[f"incremental_topk_rotation_selected_sell_{ticker}"] = float(rotation_selected_sell[idx])
            terms[f"incremental_topk_flow_delta_{ticker}"] = float(flow_delta[idx])
        return out, terms

    def _apply_group_aware_topk_priority(
        self,
        *,
        buy_priority: np.ndarray,
        sell_priority: np.ndarray,
        prev: np.ndarray,
        target: np.ndarray,
        cfg: dict[str, Any],
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Soft group-aware priority modifier for incremental Top-K routing.

        This deliberately does not allocate a hard per-group flow budget. A single
        group can still receive all Top-K flow when its stocks dominate the global
        priorities. Groups only adjust stock priorities through pressure/capacity
        multipliers and provide diagnostics.
        """

        adjusted_buy = np.asarray(buy_priority, dtype=np.float64).copy()
        adjusted_sell = np.asarray(sell_priority, dtype=np.float64).copy()
        stock_group_names = [group for group in self.group_names if group != "cash"]
        if not stock_group_names:
            return adjusted_buy, adjusted_sell, {"incremental_topk_group_aware_enabled": 0.0}

        default_group_cap = float(cfg.get("default_group_cap", cfg.get("group_cap", 1.0)))
        group_caps_cfg = cfg.get("group_caps", {})
        pressure_weight = float(cfg.get("pressure_weight", 1.0))
        capacity_weight = float(cfg.get("capacity_weight", 1.0))
        sell_overweight_weight = float(cfg.get("sell_overweight_weight", 1.0))
        buy_floor = float(cfg.get("buy_priority_floor", cfg.get("priority_floor", 0.05)))
        sell_floor = float(cfg.get("sell_priority_floor", cfg.get("priority_floor", 0.05)))
        min_multiplier = float(cfg.get("min_multiplier", 0.05))
        max_multiplier = float(cfg.get("max_multiplier", 5.0))
        residual_quality_cfg = cfg.get("residual_quality", cfg.get("group_residual_quality", {}))
        residual_quality_enabled = bool(residual_quality_cfg.get("enabled", False))
        rank_mix = np.zeros(self.stock_dim, dtype=np.float64)
        residual_mix = np.zeros(self.stock_dim, dtype=np.float64)
        residual_threshold = 0.0
        residual_quality_min = 0.25
        residual_quality_max = 2.50
        buy_rank_weight = 0.0
        sell_rank_weight = 0.0
        buy_breadth_weight = 0.0
        sell_breadth_weight = 0.0
        if residual_quality_enabled:
            derived_cfg = self._current_derived_features()
            rank_5d_weight = float(residual_quality_cfg.get("rank_5d_weight", 0.70))
            rank_20d_weight = float(residual_quality_cfg.get("rank_20d_weight", 0.30))
            residual_5d_weight = float(residual_quality_cfg.get("residual_5d_weight", rank_5d_weight))
            residual_20d_weight = float(residual_quality_cfg.get("residual_20d_weight", rank_20d_weight))
            rank_weight_sum = max(abs(rank_5d_weight) + abs(rank_20d_weight), eps)
            residual_weight_sum = max(abs(residual_5d_weight) + abs(residual_20d_weight), eps)
            rank_mix = (
                rank_5d_weight * self._stock_confidence_feature_vector("residual_momentum_rank_centered_5d", derived_cfg)
                + rank_20d_weight * self._stock_confidence_feature_vector("residual_momentum_rank_centered_20d", derived_cfg)
            ) / rank_weight_sum
            residual_mix = (
                residual_5d_weight * self._stock_confidence_feature_vector("residual_momentum_5d", derived_cfg)
                + residual_20d_weight * self._stock_confidence_feature_vector("residual_momentum_20d", derived_cfg)
            ) / residual_weight_sum
            rank_mix = np.nan_to_num(rank_mix, nan=0.0, posinf=0.0, neginf=0.0)
            residual_mix = np.nan_to_num(residual_mix, nan=0.0, posinf=0.0, neginf=0.0)
            residual_threshold = float(residual_quality_cfg.get("residual_positive_threshold", 0.0))
            residual_quality_min = float(residual_quality_cfg.get("min_multiplier", 0.25))
            residual_quality_max = float(residual_quality_cfg.get("max_multiplier", 2.50))
            buy_rank_weight = float(residual_quality_cfg.get("buy_rank_weight", 0.60))
            sell_rank_weight = float(residual_quality_cfg.get("sell_rank_weight", 0.85))
            buy_breadth_weight = float(
                residual_quality_cfg.get("buy_breadth_weight", residual_quality_cfg.get("breadth_weight", 0.50))
            )
            sell_breadth_weight = float(
                residual_quality_cfg.get("sell_breadth_weight", residual_quality_cfg.get("breadth_weight", 0.50))
            )

        stock_to_group = np.full(self.stock_dim, "", dtype=object)
        terms: dict[str, Any] = {
            "incremental_topk_group_aware_enabled": 1.0,
            "incremental_topk_group_count": float(len(stock_group_names)),
            "incremental_topk_group_default_cap": default_group_cap,
            "incremental_topk_group_residual_quality_enabled": float(residual_quality_enabled),
        }
        residual_buy_multipliers: list[float] = []
        residual_sell_multipliers: list[float] = []
        residual_rank_values: list[float] = []
        residual_breadth_values: list[float] = []

        for group in stock_group_names:
            indices = np.asarray(self.group_to_indices[group], dtype=int)
            if indices.size == 0:
                continue
            stock_to_group[indices] = group
            cap = max(float(group_caps_cfg.get(group, default_group_cap)), eps)
            prev_group = float(np.sum(prev[indices]))
            target_group = float(np.sum(target[indices]))
            buy_pressure = max(target_group - prev_group, 0.0)
            sell_pressure = max(prev_group - target_group, 0.0)
            buy_capacity = float(np.clip(1.0 - prev_group / cap, 0.0, 1.0))
            sell_overweight = max(prev_group / cap - 1.0, 0.0)

            buy_multiplier = buy_floor + pressure_weight * buy_pressure + capacity_weight * buy_capacity
            sell_multiplier = sell_floor + pressure_weight * sell_pressure + sell_overweight_weight * sell_overweight
            buy_multiplier = float(np.clip(buy_multiplier, min_multiplier, max_multiplier))
            sell_multiplier = float(np.clip(sell_multiplier, min_multiplier, max_multiplier))
            residual_buy_multiplier = 1.0
            residual_sell_multiplier = 1.0
            group_rank_quality = 0.0
            group_breadth_excess = 0.0
            if residual_quality_enabled:
                group_rank_quality = float(np.mean(rank_mix[indices]))
                group_breadth_excess = float(np.mean(residual_mix[indices] > residual_threshold) - 0.5)
                residual_buy_multiplier = float(
                    np.clip(
                        1.0 + buy_rank_weight * group_rank_quality + buy_breadth_weight * group_breadth_excess,
                        residual_quality_min,
                        residual_quality_max,
                    )
                )
                residual_sell_multiplier = float(
                    np.clip(
                        1.0 - sell_rank_weight * group_rank_quality - sell_breadth_weight * group_breadth_excess,
                        residual_quality_min,
                        residual_quality_max,
                    )
                )
                buy_multiplier = float(np.clip(buy_multiplier * residual_buy_multiplier, min_multiplier, max_multiplier))
                sell_multiplier = float(np.clip(sell_multiplier * residual_sell_multiplier, min_multiplier, max_multiplier))
                residual_buy_multipliers.append(residual_buy_multiplier)
                residual_sell_multipliers.append(residual_sell_multiplier)
                residual_rank_values.append(group_rank_quality)
                residual_breadth_values.append(group_breadth_excess)

            adjusted_buy[indices] *= buy_multiplier
            adjusted_sell[indices] *= sell_multiplier

            safe_group = str(group).replace(" ", "_").replace("/", "_")
            terms[f"incremental_topk_group_prev_{safe_group}"] = prev_group
            terms[f"incremental_topk_group_target_{safe_group}"] = target_group
            terms[f"incremental_topk_group_cap_{safe_group}"] = cap
            terms[f"incremental_topk_group_buy_pressure_{safe_group}"] = buy_pressure
            terms[f"incremental_topk_group_sell_pressure_{safe_group}"] = sell_pressure
            terms[f"incremental_topk_group_buy_capacity_{safe_group}"] = buy_capacity
            terms[f"incremental_topk_group_sell_overweight_{safe_group}"] = sell_overweight
            terms[f"incremental_topk_group_buy_multiplier_{safe_group}"] = buy_multiplier
            terms[f"incremental_topk_group_sell_multiplier_{safe_group}"] = sell_multiplier
            if residual_quality_enabled:
                terms[f"incremental_topk_group_residual_rank_quality_{safe_group}"] = group_rank_quality
                terms[f"incremental_topk_group_residual_breadth_excess_{safe_group}"] = group_breadth_excess
                terms[f"incremental_topk_group_residual_buy_multiplier_{safe_group}"] = residual_buy_multiplier
                terms[f"incremental_topk_group_residual_sell_multiplier_{safe_group}"] = residual_sell_multiplier

        if residual_quality_enabled and residual_buy_multipliers:
            terms["incremental_topk_group_residual_buy_multiplier_mean"] = float(np.mean(residual_buy_multipliers))
            terms["incremental_topk_group_residual_sell_multiplier_mean"] = float(np.mean(residual_sell_multipliers))
            terms["incremental_topk_group_residual_rank_quality_mean"] = float(np.mean(residual_rank_values))
            terms["incremental_topk_group_residual_breadth_excess_mean"] = float(np.mean(residual_breadth_values))

        return adjusted_buy, adjusted_sell, terms

    def _feature_value(self, feature: str, *, default: float = 0.0) -> float:
        if feature == "drawdown_severity":
            return float(np.clip(-self.previous_drawdown / 0.10, 0.0, 2.0))
        if feature == "last_turnover":
            return float(self.last_turnover)
        if feature == "cash_duration_score":
            return float(np.clip(self.cash_duration_days / max(self.cash_duration_threshold_days, 1.0), 0.0, 1.0))
        derived_cfg = self._current_derived_features()
        market_beta = float(derived_cfg.get("market_beta", 0.50))
        market_beta_5d = float(derived_cfg.get("market_beta_5d", market_beta))
        market_beta_20d = float(derived_cfg.get("market_beta_20d", market_beta))
        market_factor_5d = str(derived_cfg.get("market_factor_feature_5d", derived_cfg.get("market_factor_feature", "SP500_Trend")))
        market_factor_20d = str(derived_cfg.get("market_factor_feature_20d", derived_cfg.get("market_factor_feature", "SP500_Trend")))
        stock_market_beta_5d = float(derived_cfg.get("stock_market_beta_5d", market_beta_5d))
        stock_market_beta_20d = float(derived_cfg.get("stock_market_beta_20d", market_beta_20d))
        stock_market_factor_5d = str(derived_cfg.get("stock_market_factor_feature_5d", "universe_return_5d"))
        stock_market_factor_20d = str(derived_cfg.get("stock_market_factor_feature_20d", market_factor_20d))
        vix_market_beta = float(derived_cfg.get("vix_market_beta", 0.25))
        vix_market_beta_5d = float(derived_cfg.get("vix_market_beta_5d", vix_market_beta))
        vix_trend_beta_5d = float(derived_cfg.get("vix_trend_beta_5d", vix_market_beta))
        vix_factor_5d = str(derived_cfg.get("vix_factor_feature_5d", "universe_return_5d"))
        vix_trend_factor_5d = str(derived_cfg.get("vix_trend_factor_feature_5d", "SP500_Trend"))
        vix_surprise_mode = str(derived_cfg.get("vix_surprise_mode", "legacy_additive")).lower()
        residual_threshold = float(derived_cfg.get("residual_breadth_threshold", 0.0))
        vix_sign = -1.0 if vix_surprise_mode in {"regression", "subtract", "ols"} else 1.0
        if feature in {
            "market_beta_5d",
            "market_beta_20d",
            "stock_market_beta_5d",
            "stock_market_beta_20d",
            "vix_market_beta_5d",
            "vix_trend_beta_5d",
            "beta_schedule_yyyymm",
        }:
            return float(derived_cfg.get(feature, default))
        if feature == "residual_universe_return_20d":
            return float(
                self._feature_value("universe_return_20d", default=default)
                - market_beta_20d * self._feature_value(market_factor_20d, default=0.0)
            )
        if feature == "residual_universe_return_5d":
            return float(
                self._feature_value("universe_return_5d", default=default)
                - market_beta_5d * self._feature_value(market_factor_5d, default=0.0)
            )
        if feature == "market_down_5d":
            return float(max(-self._feature_value("universe_return_5d", default=0.0), 0.0))
        if feature == "market_up_5d":
            return float(max(self._feature_value("universe_return_5d", default=0.0), 0.0))
        if feature == "market_down_20d":
            return float(max(-self._feature_value("universe_return_20d", default=0.0), 0.0))
        if feature == "market_up_20d":
            return float(max(self._feature_value("universe_return_20d", default=0.0), 0.0))
        if feature == "sp500_trend_delta_down_1d":
            return float(max(-self._feature_value("SP500_Trend_delta_1d", default=0.0), 0.0))
        if feature == "sp500_trend_delta_up_1d":
            return float(max(self._feature_value("SP500_Trend_delta_1d", default=0.0), 0.0))
        if feature == "residual_universe_down_20d":
            return float(max(-self._feature_value("residual_universe_return_20d", default=0.0), 0.0))
        if feature == "residual_universe_up_20d":
            return float(max(self._feature_value("residual_universe_return_20d", default=0.0), 0.0))
        if feature == "residual_universe_down_5d":
            return float(max(-self._feature_value("residual_universe_return_5d", default=0.0), 0.0))
        if feature == "residual_universe_up_5d":
            return float(max(self._feature_value("residual_universe_return_5d", default=0.0), 0.0))
        if feature == "vix_surprise_5d":
            return float(
                self._feature_value("VIX_change_5d", default=default)
                + vix_sign * vix_trend_beta_5d * self._feature_value(vix_trend_factor_5d, default=0.0)
            )
        if feature == "vix_surprise_return_5d":
            return float(
                self._feature_value("VIX_change_5d", default=default)
                + vix_sign * vix_market_beta_5d * self._feature_value(vix_factor_5d, default=0.0)
            )
        if feature == "residual_breadth_20d":
            residual = (
                self._stock_feature_vector("logret_20d", default=0.0)
                - stock_market_beta_20d * self._feature_value(stock_market_factor_20d, default=0.0)
            )
            residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
            return float(np.mean(residual > residual_threshold))
        if feature == "residual_breadth_excess_20d":
            return float(max(self._feature_value("residual_breadth_20d", default=0.5) - 0.5, 0.0))
        if feature == "residual_breadth_shortfall_20d":
            return float(max(0.5 - self._feature_value("residual_breadth_20d", default=0.5), 0.0))
        if feature == "residual_breadth_5d":
            residual = (
                self._stock_feature_vector("logret_5d", default=0.0)
                - stock_market_beta_5d * self._feature_value(stock_market_factor_5d, default=0.0)
            )
            residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
            return float(np.mean(residual > residual_threshold))
        if feature == "residual_breadth_excess_5d":
            return float(max(self._feature_value("residual_breadth_5d", default=0.5) - 0.5, 0.0))
        if feature == "residual_breadth_shortfall_5d":
            return float(max(0.5 - self._feature_value("residual_breadth_5d", default=0.5), 0.0))
        if feature == "residual_dispersion_20d":
            residual = (
                self._stock_feature_vector("logret_20d", default=0.0)
                - stock_market_beta_20d * self._feature_value(stock_market_factor_20d, default=0.0)
            )
            residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
            return float(np.std(residual))
        if feature == "residual_dispersion_5d":
            residual = (
                self._stock_feature_vector("logret_5d", default=0.0)
                - stock_market_beta_5d * self._feature_value(stock_market_factor_5d, default=0.0)
            )
            residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
            return float(np.std(residual))
        if feature == "residual_dispersion_20d_breadth_weak":
            threshold = max(float(derived_cfg.get("dispersion_breadth_excess_threshold_20d", 0.08)), EPS)
            breadth_excess = self._feature_value("residual_breadth_excess_20d", default=0.0)
            weak_gate = float(np.clip(1.0 - breadth_excess / threshold, 0.0, 1.0))
            return self._feature_value("residual_dispersion_20d", default=0.0) * weak_gate
        if feature == "residual_dispersion_5d_breadth_weak":
            threshold = max(float(derived_cfg.get("dispersion_breadth_excess_threshold_5d", 0.08)), EPS)
            breadth_excess = self._feature_value("residual_breadth_excess_5d", default=0.0)
            weak_gate = float(np.clip(1.0 - breadth_excess / threshold, 0.0, 1.0))
            return self._feature_value("residual_dispersion_5d", default=0.0) * weak_gate
        idx = self.feature_index.get(str(feature))
        if idx is None:
            return float(default)
        values = self.panel.features[self.day, :, idx].astype(np.float64)
        values = np.nan_to_num(values, nan=default, posinf=default, neginf=default)
        return float(np.mean(values))

    def _safe_feature_name(self, feature: str) -> str:
        return (
            str(feature)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
            .replace("%", "pct")
        )

    def _weighted_signal_terms(self, name: str, signal_cfg: dict[str, Any]) -> tuple[float, dict[str, float]]:
        raw = float(signal_cfg.get("intercept", 0.0))
        terms: dict[str, Any] = {
            f"{name}_raw": raw,
            f"confidence_component_{name}_intercept": raw,
        }
        for feature, weight in signal_cfg.get("feature_weights", {}).items():
            feature_name = str(feature)
            safe = self._safe_feature_name(feature_name)
            value = self._feature_value(feature_name)
            w = float(weight)
            contrib = w * value
            raw += contrib
            terms[f"confidence_component_{name}_{safe}_value"] = float(value)
            terms[f"confidence_component_{name}_{safe}_weight"] = float(w)
            terms[f"confidence_component_{name}_{safe}_contrib"] = float(contrib)
        score = sigmoid_scalar(raw)
        terms[f"{name}_raw"] = float(raw)
        terms[f"{name}"] = float(score)
        return score, terms

    def _weighted_signal_score(self, signal_cfg: dict[str, Any]) -> float:
        score, _ = self._weighted_signal_terms("signal", signal_cfg)
        return score

    def _stock_residual_vector(self, horizon: int, derived_cfg: dict[str, Any]) -> np.ndarray:
        horizon = int(horizon)
        if horizon == 20:
            stock_feature = "logret_20d"
            beta = float(derived_cfg.get("stock_market_beta_20d", derived_cfg.get("market_beta_20d", 0.50)))
            factor = str(derived_cfg.get("stock_market_factor_feature_20d", "universe_return_20d"))
        else:
            stock_feature = "logret_5d"
            beta = float(derived_cfg.get("stock_market_beta_5d", derived_cfg.get("market_beta_5d", 0.50)))
            factor = str(derived_cfg.get("stock_market_factor_feature_5d", "universe_return_5d"))
        residual = self._stock_feature_vector(stock_feature, default=0.0) - beta * self._feature_value(factor, default=0.0)
        return np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

    def _stock_confidence_feature_vector(self, feature: str, derived_cfg: dict[str, Any]) -> np.ndarray:
        feature = str(feature)
        if feature == "residual_momentum_5d":
            return self._stock_residual_vector(5, derived_cfg)
        if feature == "residual_momentum_20d":
            return self._stock_residual_vector(20, derived_cfg)
        if feature == "residual_momentum_rank_5d":
            return rank01(self._stock_residual_vector(5, derived_cfg))
        if feature == "residual_momentum_rank_20d":
            return rank01(self._stock_residual_vector(20, derived_cfg))
        if feature in {"residual_momentum_rank_centered_5d", "centered_residual_momentum_rank_5d"}:
            return rank01(self._stock_residual_vector(5, derived_cfg)) - 0.5
        if feature in {"residual_momentum_rank_centered_20d", "centered_residual_momentum_rank_20d"}:
            return rank01(self._stock_residual_vector(20, derived_cfg)) - 0.5
        if feature == "negative_residual_momentum_5d":
            return -self._stock_residual_vector(5, derived_cfg)
        if feature == "negative_residual_momentum_20d":
            return -self._stock_residual_vector(20, derived_cfg)
        if feature == "residual_volatility_20d":
            return self._stock_feature_vector("realized_vol_20d", default=0.0)
        if feature == "negative_residual_volatility_20d":
            return -self._stock_feature_vector("realized_vol_20d", default=0.0)
        if feature == "volume_confirmation":
            return 0.5 * self._stock_feature_vector("volume_ratio_delta_1d", default=0.0) + 0.5 * self._stock_feature_vector(
                "volume_zscore_20d_raw",
                default=0.0,
            )
        if feature == "trend_delta":
            return self._stock_feature_vector("macd_delta_1d", default=0.0)
        if feature == "trend_level":
            return 0.5 * self._stock_feature_vector("macd", default=0.0) + 0.5 * self._stock_feature_vector(
                "price_sma20_ratio",
                default=0.0,
            )
        if feature == "trend_break":
            return -(
                0.5 * self._stock_feature_vector("macd_delta_1d", default=0.0)
                + 0.5 * self._stock_feature_vector("price_sma20_ratio", default=0.0)
            )
        return self._stock_feature_vector(feature, default=0.0)

    def _stock_signal_vector_terms(
        self,
        name: str,
        signal_cfg: dict[str, Any],
        derived_cfg: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, float]]:
        raw = np.full(self.stock_dim, float(signal_cfg.get("intercept", 0.0)), dtype=np.float64)
        terms: dict[str, float] = {f"stock_conf_{name}_raw_mean": float(np.mean(raw))}
        for feature, weight in signal_cfg.get("feature_weights", {}).items():
            feature_name = str(feature)
            safe = self._safe_feature_name(feature_name)
            values = self._stock_confidence_feature_vector(feature_name, derived_cfg)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            w = float(weight)
            contrib = w * values
            raw += contrib
            terms[f"stock_conf_component_{name}_{safe}_value_mean"] = float(np.mean(values))
            terms[f"stock_conf_component_{name}_{safe}_contrib_mean"] = float(np.mean(contrib))
        score = 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))
        terms[f"stock_conf_{name}_raw_mean"] = float(np.mean(raw))
        terms[f"stock_conf_{name}_mean"] = float(np.mean(score))
        terms[f"stock_conf_{name}_p10"] = float(np.percentile(score, 10))
        terms[f"stock_conf_{name}_p90"] = float(np.percentile(score, 90))
        return score.astype(np.float64), terms

    def _apply_stock_confidence_slice(
        self,
        target_weights: np.ndarray,
        stock_cfg: dict[str, Any],
        topk_context: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not stock_cfg or not bool(stock_cfg.get("enabled", False)):
            return target_weights, {"stock_slice_enabled": 0.0}

        derived_cfg = self._current_derived_features()
        buy_conf, buy_terms = self._stock_signal_vector_terms("buy", stock_cfg.get("buy_score", {}), derived_cfg)
        sell_conf, sell_terms = self._stock_signal_vector_terms("sell", stock_cfg.get("sell_score", {}), derived_cfg)

        prev = self.previous_weights.copy()
        target = normalize_simplex(target_weights)
        delta = target[: self.stock_dim] - prev[: self.stock_dim]
        eps = float(stock_cfg.get("eps_weight", 1e-6))
        min_scale = float(stock_cfg.get("min_scale", 0.0))
        max_scale = float(stock_cfg.get("max_scale", 1.0))
        root_direction = "hold"
        if topk_context:
            root_direction = str(topk_context.get("root_direction", "hold"))
        buy_min_scale = float(stock_cfg.get("buy_min_scale", min_scale))
        sell_min_scale = float(stock_cfg.get("sell_min_scale", min_scale))
        if root_direction == "rerisk":
            buy_min_scale = float(stock_cfg.get("rerisk_buy_min_scale", buy_min_scale))
            sell_min_scale = float(stock_cfg.get("rerisk_sell_min_scale", sell_min_scale))
        elif root_direction == "derisk":
            buy_min_scale = float(stock_cfg.get("derisk_buy_min_scale", buy_min_scale))
            sell_min_scale = float(stock_cfg.get("derisk_sell_min_scale", sell_min_scale))
        elif root_direction == "hold":
            buy_min_scale = float(stock_cfg.get("hold_buy_min_scale", buy_min_scale))
            sell_min_scale = float(stock_cfg.get("hold_sell_min_scale", sell_min_scale))
        scale = np.ones(self.stock_dim, dtype=np.float64)
        buy_mask = delta > eps
        sell_mask = delta < -eps
        scale[buy_mask] = buy_min_scale + (max_scale - buy_min_scale) * buy_conf[buy_mask]
        scale[sell_mask] = sell_min_scale + (max_scale - sell_min_scale) * sell_conf[sell_mask]

        top_k_buy = int(stock_cfg.get("top_k_buy", 0))
        top_k_sell = int(stock_cfg.get("top_k_sell", 0))
        topk_enabled = top_k_buy > 0 or top_k_sell > 0
        selected_buy = np.zeros(self.stock_dim, dtype=bool)
        selected_sell = np.zeros(self.stock_dim, dtype=bool)
        buy_priority = np.zeros(self.stock_dim, dtype=np.float64)
        sell_priority = np.zeros(self.stock_dim, dtype=np.float64)
        sell_expansion_count = 0
        root_gap_before_expand = 0.0
        root_gap_after_expand = 0.0
        buy_unfilled_cash = 0.0
        if topk_enabled:
            u_start = None
            u_anchor = None
            if topk_context:
                u_start = topk_context.get("u_window_start")
                u_anchor = topk_context.get("u_anchor")
            if u_start is None:
                u_start = self._conditional_risky_allocation(prev)
            if u_anchor is None:
                u_anchor = self._conditional_risky_allocation(target)
            u_start = normalize_stock_simplex(np.asarray(u_start, dtype=np.float64))
            u_anchor = normalize_stock_simplex(np.asarray(u_anchor, dtype=np.float64))

            conditional_buy = np.maximum(u_anchor - u_start, 0.0)
            conditional_sell = np.maximum(u_start - u_anchor, 0.0)
            abs_buy = np.maximum(delta, 0.0)
            abs_sell = np.maximum(-delta, 0.0)
            buy_priority = (
                float(stock_cfg.get("topk_conditional_buy_weight", 0.70)) * conditional_buy
                + float(stock_cfg.get("topk_abs_buy_weight", 0.30)) * abs_buy
            )
            sell_priority = (
                float(stock_cfg.get("topk_conditional_sell_weight", 0.50)) * conditional_sell
                + float(stock_cfg.get("topk_abs_sell_weight", 0.50)) * abs_sell
            )
            if bool(stock_cfg.get("topk_multiply_by_confidence", True)):
                buy_priority *= buy_conf
                sell_priority *= sell_conf

            if top_k_buy > 0 and np.any(buy_mask):
                buy_indices = np.where(buy_mask)[0]
                keep = min(top_k_buy, buy_indices.size)
                selected_buy[buy_indices[np.argsort(buy_priority[buy_indices])[-keep:]]] = True
                scale[buy_mask & ~selected_buy] = float(
                    stock_cfg.get("nonselected_buy_scale", stock_cfg.get("nonselected_scale", 0.0))
                )
                raw_buy_demand = float(np.sum(abs_buy[buy_mask]))
                selected_buy_capacity = float(np.sum(abs_buy[selected_buy]))
                buy_unfilled_cash = max(raw_buy_demand - selected_buy_capacity, 0.0)

            if top_k_sell > 0 and np.any(sell_mask):
                sell_indices = np.where(sell_mask)[0]
                keep = min(top_k_sell, sell_indices.size)
                selected_sell[sell_indices[np.argsort(sell_priority[sell_indices])[-keep:]]] = True
                scale[sell_mask & ~selected_sell] = float(
                    stock_cfg.get("nonselected_sell_scale", stock_cfg.get("nonselected_scale", 0.0))
                )

                if bool(stock_cfg.get("sell_expansion_enabled", True)):
                    desired_stock_sum = float(np.sum(target[: self.stock_dim]))
                    selected_scaled_delta = scale * delta
                    projected_stock_sum = float(np.sum(prev[: self.stock_dim] + selected_scaled_delta))
                    root_gap_before_expand = max(projected_stock_sum - desired_stock_sum, 0.0)
                    if root_gap_before_expand > float(stock_cfg.get("sell_expansion_eps", eps)):
                        nonselected_sells = np.where(sell_mask & ~selected_sell)[0]
                        ordered = nonselected_sells[np.argsort(sell_priority[nonselected_sells])[::-1]]
                        for idx in ordered:
                            selected_sell[idx] = True
                            scale[idx] = sell_min_scale + (max_scale - sell_min_scale) * sell_conf[idx]
                            sell_expansion_count += 1
                            selected_scaled_delta = scale * delta
                            projected_stock_sum = float(np.sum(prev[: self.stock_dim] + selected_scaled_delta))
                            root_gap_after_expand = max(projected_stock_sum - desired_stock_sum, 0.0)
                            if root_gap_after_expand <= float(stock_cfg.get("sell_expansion_eps", eps)):
                                break
                    else:
                        root_gap_after_expand = root_gap_before_expand

        out = prev.copy()
        out[: self.stock_dim] = np.maximum(prev[: self.stock_dim] + scale * delta, 0.0)
        stock_sum = float(np.sum(out[: self.stock_dim]))
        if stock_sum > 1.0:
            out[: self.stock_dim] = normalize_stock_simplex(out[: self.stock_dim])
            stock_sum = 1.0
        out[self.cash_index] = max(0.0, 1.0 - stock_sum)
        out = normalize_simplex(out)

        suppressed_l1 = float(np.sum(np.abs(target - out)))
        terms: dict[str, float] = {
            "stock_slice_enabled": 1.0,
            "stock_slice_buy_count": float(np.sum(buy_mask)),
            "stock_slice_sell_count": float(np.sum(sell_mask)),
            "stock_slice_buy_min_scale": float(buy_min_scale),
            "stock_slice_sell_min_scale": float(sell_min_scale),
            "stock_slice_root_direction_rerisk": 1.0 if root_direction == "rerisk" else 0.0,
            "stock_slice_root_direction_derisk": 1.0 if root_direction == "derisk" else 0.0,
            "stock_slice_root_direction_hold": 1.0 if root_direction == "hold" else 0.0,
            "stock_slice_scale_mean": float(np.mean(scale)),
            "stock_slice_scale_buy_mean": float(np.mean(scale[buy_mask])) if np.any(buy_mask) else 0.0,
            "stock_slice_scale_sell_mean": float(np.mean(scale[sell_mask])) if np.any(sell_mask) else 0.0,
            "stock_slice_suppressed_l1": suppressed_l1,
            "stock_slice_suppressed_turnover": 0.5 * suppressed_l1,
            "topk_enabled": 1.0 if topk_enabled else 0.0,
            "topk_buy_selected_count": float(np.sum(selected_buy)),
            "topk_sell_selected_count": float(np.sum(selected_sell)),
            "topk_nonselected_suppressed_l1": float(
                np.sum(np.abs(target[: self.stock_dim] - out[: self.stock_dim]) * ~(selected_buy | selected_sell))
            )
            if topk_enabled
            else 0.0,
            "topk_root_gap_before_expand": float(root_gap_before_expand),
            "topk_root_gap_after_expand": float(root_gap_after_expand),
            "topk_sell_expansion_count": float(sell_expansion_count),
            "topk_buy_unfilled_cash": float(buy_unfilled_cash),
            "topk_buy_tickers": "|".join(self.tickers[idx] for idx in np.where(selected_buy)[0]),
            "topk_sell_tickers": "|".join(self.tickers[idx] for idx in np.where(selected_sell)[0]),
            **buy_terms,
            **sell_terms,
        }
        for idx, ticker in enumerate(self.tickers):
            terms[f"stock_conf_buy_{ticker}"] = float(buy_conf[idx])
            terms[f"stock_conf_sell_{ticker}"] = float(sell_conf[idx])
            terms[f"stock_slice_scale_{ticker}"] = float(scale[idx])
            if topk_enabled:
                terms[f"topk_buy_priority_{ticker}"] = float(buy_priority[idx])
                terms[f"topk_sell_priority_{ticker}"] = float(sell_priority[idx])
                terms[f"topk_selected_buy_{ticker}"] = float(selected_buy[idx])
                terms[f"topk_selected_sell_{ticker}"] = float(selected_sell[idx])
        return out, terms

    def _market_context_terms(self, cfg: dict[str, Any]) -> dict[str, float]:
        feature_names = cfg.get(
            "market_context_features",
            [
                "Regime_0_Prob",
                "Regime_1_Prob",
                "regime_entropy",
                "SP500_Trend",
                "SP500_Trend_delta_1d",
                "VIX",
                "VIX_change_1d",
                "VIX_change_5d",
                "VIX_percentile_252",
                "turbulence",
                "turbulence_delta_1d",
                "turbulence_percentile_252",
                "universe_return_1d",
                "universe_return_5d",
                "universe_return_20d",
                "universe_vol_20d",
            ],
        )
        terms: dict[str, float] = {}
        for feature in feature_names:
            safe_name = str(feature).replace(" ", "_")
            terms[f"market_feature_{safe_name}"] = self._feature_value(str(feature), default=np.nan)
        for feature in [
            "market_beta_5d",
            "market_beta_20d",
            "stock_market_beta_5d",
            "stock_market_beta_20d",
            "vix_market_beta_5d",
            "vix_trend_beta_5d",
            "beta_schedule_yyyymm",
        ]:
            terms[f"market_feature_{feature}"] = self._feature_value(feature, default=np.nan)
        return terms

    @property
    def cash_duration_threshold_days(self) -> float:
        return float(getattr(self, "_cash_duration_threshold_days", 20.0))

    @cash_duration_threshold_days.setter
    def cash_duration_threshold_days(self, value: float) -> None:
        self._cash_duration_threshold_days = max(float(value), 1.0)

    def _k_window_confidence_terms(
        self,
        *,
        q_anchor: float,
        q_prev: float,
        q_scheduled: float,
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        target_eps = float(cfg.get("target_strength_eps", 0.02))
        target_tau = float(cfg.get("target_strength_tau", 0.10))
        target_strength = smoothstep((abs(q_anchor - q_prev) - target_eps) / max(target_tau, EPS))

        self.cash_duration_threshold_days = float(cfg.get("cash_duration_days_scale", 20.0))
        cash_duration_score = float(np.clip(self.cash_duration_days / self.cash_duration_threshold_days, 0.0, 1.0))

        risk_cfg = cfg.get("risk_stress", {})
        recovery_cfg = cfg.get("recovery_score", {})
        risk_stress, risk_component_terms = self._weighted_signal_terms("risk_stress", risk_cfg)
        recovery_score, recovery_component_terms = self._weighted_signal_terms("recovery_score", recovery_cfg)

        derisk_mix = cfg.get("derisk_confidence_mix", {})
        rerisk_mix = cfg.get("rerisk_confidence_mix", {})
        derisk_risk_stress_contrib = float(derisk_mix.get("risk_stress", 0.70)) * risk_stress
        derisk_target_strength_contrib = float(derisk_mix.get("target_strength", 0.30)) * target_strength
        derisk_cash_duration_contrib = float(derisk_mix.get("cash_duration_score", 0.0)) * cash_duration_score
        rerisk_recovery_contrib = float(rerisk_mix.get("recovery_score", 0.60)) * recovery_score
        rerisk_target_strength_contrib = float(rerisk_mix.get("target_strength", 0.25)) * target_strength
        rerisk_cash_duration_contrib = float(rerisk_mix.get("cash_duration_score", 0.15)) * cash_duration_score
        confidence_derisk = (
            derisk_risk_stress_contrib
            + derisk_target_strength_contrib
            + derisk_cash_duration_contrib
        )
        confidence_rerisk = (
            rerisk_recovery_contrib
            + rerisk_target_strength_contrib
            + rerisk_cash_duration_contrib
        )
        confidence_rerisk_before_risk_gate = float(confidence_rerisk)
        rerisk_risk_gate = 1.0
        rerisk_gate_cfg = cfg.get("rerisk_risk_stress_gate", {})
        if rerisk_gate_cfg and bool(rerisk_gate_cfg.get("enabled", False)):
            min_gate = float(rerisk_gate_cfg.get("min_gate", 0.25))
            max_gate = float(rerisk_gate_cfg.get("max_gate", 1.0))
            strength = float(rerisk_gate_cfg.get("strength", 1.0))
            power = max(float(rerisk_gate_cfg.get("power", 1.0)), EPS)
            rerisk_risk_gate = float(np.clip(1.0 - strength * (risk_stress**power), min_gate, max_gate))
            confidence_rerisk *= rerisk_risk_gate
        confidence_derisk = float(np.clip(confidence_derisk, 0.0, 1.0))
        confidence_rerisk = float(np.clip(confidence_rerisk, 0.0, 1.0))

        delta_q_anchor = float(q_anchor - q_prev)
        eps_q_anchor = float(cfg.get("eps_q_anchor", cfg.get("eps_q", 0.005)))
        if delta_q_anchor < -eps_q_anchor:
            anchor_direction = "derisk"
        elif delta_q_anchor > eps_q_anchor:
            anchor_direction = "rerisk"
        else:
            anchor_direction = "hold"

        delta_q_scheduled = float(q_scheduled - q_prev)
        if delta_q_scheduled < -float(cfg.get("eps_q", 0.005)):
            direction = "derisk"
        elif delta_q_scheduled > float(cfg.get("eps_q", 0.005)):
            direction = "rerisk"
        else:
            direction = "hold"

        return {
            "risk_stress": risk_stress,
            "recovery_score": recovery_score,
            "target_strength": target_strength,
            "cash_duration": float(self.cash_duration_days),
            "cash_duration_score": cash_duration_score,
            "confidence_derisk": confidence_derisk,
            "confidence_rerisk": confidence_rerisk,
            "delta_q_anchor": delta_q_anchor,
            "k_window_anchor_direction": anchor_direction,
            "delta_q_scheduled": delta_q_scheduled,
            "k_window_direction": direction,
            "root_anchor_risk_day": 1.0 if anchor_direction == "rerisk" else 0.0,
            "root_anchor_cash_day": 1.0 if anchor_direction == "derisk" else 0.0,
            "root_anchor_hold_day": 1.0 if anchor_direction == "hold" else 0.0,
            "root_scheduled_risk_day": 1.0 if direction == "rerisk" else 0.0,
            "root_scheduled_cash_day": 1.0 if direction == "derisk" else 0.0,
            "root_scheduled_hold_day": 1.0 if direction == "hold" else 0.0,
            "confidence_mix_derisk_risk_stress_contrib": derisk_risk_stress_contrib,
            "confidence_mix_derisk_target_strength_contrib": derisk_target_strength_contrib,
            "confidence_mix_derisk_cash_duration_contrib": derisk_cash_duration_contrib,
            "confidence_mix_rerisk_recovery_score_contrib": rerisk_recovery_contrib,
            "confidence_mix_rerisk_target_strength_contrib": rerisk_target_strength_contrib,
            "confidence_mix_rerisk_cash_duration_contrib": rerisk_cash_duration_contrib,
            "confidence_rerisk_before_risk_gate": confidence_rerisk_before_risk_gate,
            "confidence_rerisk_risk_gate": rerisk_risk_gate,
            **risk_component_terms,
            **recovery_component_terms,
            **self._market_context_terms(cfg),
        }

    def _step_k_window(self, action: np.ndarray, cfg: dict[str, Any]):
        action_anchor_weights = self._action_to_target_weights(action)
        raw_anchor_weights = action_anchor_weights.copy()
        anchor_info = dict(self.last_action_info)
        q_action = float(np.sum(action_anchor_weights[: self.stock_dim]))
        u_action = self._anchor_risky_from_action(action, action_anchor_weights)
        u_window_start = self._conditional_risky_allocation(self.previous_weights)
        start_day = int(self.day)
        final_trade_day = len(self.panel.dates) - 2

        dual_cfg = cfg.get("dual_window", {})
        dual_enabled = bool(dual_cfg.get("enabled", False))
        root_window_days = max(1, int(dual_cfg.get("root_window_days", cfg.get("window_days", cfg.get("K", 1)))))
        stock_window_days = max(1, int(dual_cfg.get("stock_window_days", dual_cfg.get("stock_K", cfg.get("window_days", 1)))))
        root_elapsed_start = 0
        root_refreshed = 0.0
        q_action_ignored = 0.0

        if dual_enabled:
            root_elapsed_start = max(0, start_day - int(self.dual_root_anchor_start_day))
            root_due = (
                self.dual_root_anchor_q is None
                or bool(self.dual_root_force_refresh)
                or root_elapsed_start >= root_window_days
            )
            if root_due:
                self.dual_root_anchor_q = q_action
                self.dual_root_anchor_start_day = start_day
                self.dual_root_force_refresh = False
                root_elapsed_start = 0
                root_refreshed = 1.0
            else:
                q_action_ignored = 1.0
            q_anchor = float(self.dual_root_anchor_q)
            u_anchor = u_action
            anchor_weights = self._weights_from_root_risky(q_anchor, u_anchor)
            root_remaining_at_start = max(1, root_window_days - root_elapsed_start)
        else:
            anchor_weights = action_anchor_weights
            q_anchor = q_action
            u_anchor = u_action
            root_remaining_at_start = max(1, int(cfg.get("window_days", cfg.get("K", 1))))

        sparse_topk_terms: dict[str, Any] = {"topk_sparse_enabled": 0.0}
        sparse_topk_cfg = cfg.get("sparse_topk_target", {})
        if sparse_topk_cfg and bool(sparse_topk_cfg.get("enabled", False)):
            anchor_weights, u_anchor, sparse_topk_terms = self._apply_sparse_topk_risky_target(
                anchor_weights,
                u_anchor,
                u_window_start,
                sparse_topk_cfg,
            )
            q_anchor = float(np.sum(anchor_weights[: self.stock_dim]))

        base_window_days = max(1, int(cfg.get("window_days", cfg.get("K", 1))))
        window_days = min(stock_window_days, root_remaining_at_start) if dual_enabled else base_window_days
        mode = str(cfg.get("mode", "equal_slice_direct"))
        confidence_cfg = cfg.get("confidence_stop_recovery", {})
        confidence_enabled = bool(confidence_cfg.get("enabled", False))
        self.cash_duration_threshold = float(confidence_cfg.get("cash_trigger_min", 0.20))
        planned_end_day = min(start_day + window_days - 1, final_trade_day)

        daily_infos: list[dict[str, Any]] = []
        total_reward = 0.0
        terminated = False
        recovery_persistence_count = 0
        risk_break_persistence_count = 0

        for substep in range(window_days):
            if self.day >= len(self.panel.dates) - 1:
                terminated = True
                break

            remaining_days = max(1, min(window_days - substep, len(self.panel.dates) - 1 - self.day))
            stock_remaining_days = remaining_days
            if dual_enabled:
                root_elapsed_now = max(0, int(self.day) - int(self.dual_root_anchor_start_day))
                root_remaining_days = max(1, min(root_window_days - root_elapsed_now, len(self.panel.dates) - 1 - self.day))
            else:
                root_elapsed_now = substep
                root_remaining_days = remaining_days
            q_prev = float(np.sum(self.previous_weights[: self.stock_dim]))
            u_prev = self._conditional_risky_allocation(self.previous_weights)

            q_scheduled = q_prev + (q_anchor - q_prev) / float(root_remaining_days)
            u_scheduled = normalize_stock_simplex(u_prev + (u_anchor - u_prev) / float(stock_remaining_days))
            scheduled_weights = self._weights_from_root_risky(q_scheduled, u_scheduled)
            execution_target_weights = scheduled_weights

            confidence_terms: dict[str, Any] = {}
            stop_active = 0.0
            stop_reason = "none"
            suppressed_trade_l1 = 0.0
            suppressed_turnover = 0.0
            suppressed_trade_value = 0.0
            recovery_trigger = 0.0
            derisk_early_update = 0.0
            rerisk_early_update = 0.0
            window_closed_early = 0.0
            early_update_reason = "none"
            recovery_trigger_candidate = 0.0
            risk_break_trigger_candidate = 0.0
            recovery_cash_condition_met = 1.0
            recovery_anchor_condition_met = 1.0
            recovery_confidence_condition_met = 1.0
            recovery_residual_condition_met = 1.0
            recovery_breadth_condition_met = 1.0
            recovery_risk_stress_condition_met = 1.0
            recovery_residual_up_5d = 0.0
            recovery_breadth_excess_5d = 0.0
            risk_break_confidence_condition_met = 1.0
            event_trigger_allowed = 1.0
            risk_break_event_allowed = 1.0
            cooldown_remaining_at_start = float(self.early_update_cooldown_remaining)
            confidence_slice_terms: dict[str, float] = {
                "confidence_slice_enabled": 0.0,
                "confidence_slice_root_scale": 1.0,
                "confidence_slice_suppressed_l1": 0.0,
                "confidence_slice_suppressed_turnover": 0.0,
            }
            stock_slice_terms: dict[str, float] = {"stock_slice_enabled": 0.0}
            incremental_topk_terms: dict[str, Any] = {"incremental_topk_enabled": 0.0}
            incremental_topk_weights: np.ndarray | None = None

            if confidence_enabled:
                confidence_terms = self._k_window_confidence_terms(
                    q_anchor=q_anchor,
                    q_prev=q_prev,
                    q_scheduled=float(np.sum(scheduled_weights[: self.stock_dim])),
                    cfg=confidence_cfg,
                )
                eps_q = float(confidence_cfg.get("eps_q", 0.005))
                threshold_derisk = float(confidence_cfg.get("threshold_derisk", 0.55))
                threshold_rerisk = float(confidence_cfg.get("threshold_rerisk", 0.55))
                delta_q_scheduled = float(confidence_terms["delta_q_scheduled"])

                if delta_q_scheduled < -eps_q and float(confidence_terms["confidence_derisk"]) < threshold_derisk:
                    stop_active = 1.0
                    stop_reason = "derisk_confidence_low"
                elif delta_q_scheduled > eps_q and float(confidence_terms["confidence_rerisk"]) < threshold_rerisk:
                    stop_active = 1.0
                    stop_reason = "rerisk_confidence_low"

                partial_stop_enabled = bool(
                    confidence_cfg.get(
                        "partial_stop_enabled",
                        confidence_cfg.get("confidence_slice", {}).get("partial_stop_enabled", False),
                    )
                )
                if stop_active and not partial_stop_enabled:
                    execution_target_weights = self.previous_weights.copy()
                    suppressed_trade_l1 = float(np.sum(np.abs(scheduled_weights - self.previous_weights)))
                    suppressed_turnover = 0.5 * suppressed_trade_l1
                    suppressed_trade_value = float(self.portfolio_value * suppressed_turnover)

                current_cash = float(self.previous_weights[self.cash_index])
                use_cash_condition = bool(confidence_cfg.get("use_cash_condition", True))
                recovery_cash_condition_met = (
                    1.0
                    if (not use_cash_condition or current_cash > float(confidence_cfg.get("cash_trigger_min", 0.20)))
                    else 0.0
                )
                require_recovery_anchor_rerisk = bool(confidence_cfg.get("require_recovery_anchor_rerisk", False))
                recovery_anchor_condition_met = (
                    1.0
                    if (
                        not require_recovery_anchor_rerisk
                        or float(confidence_terms["delta_q_anchor"])
                        > float(confidence_cfg.get("recovery_min_anchor_gap", 0.02))
                    )
                    else 0.0
                )
                recovery_confidence_condition_met = (
                    1.0
                    if float(confidence_terms["confidence_rerisk"])
                    >= float(confidence_cfg.get("recovery_min_confidence_rerisk", 0.0))
                    else 0.0
                )
                recovery_residual_up_5d = self._feature_value("residual_universe_up_5d", default=0.0)
                recovery_breadth_excess_5d = self._feature_value("residual_breadth_excess_5d", default=0.0)
                recovery_residual_condition_met = (
                    1.0
                    if recovery_residual_up_5d
                    >= float(confidence_cfg.get("recovery_min_residual_universe_up_5d", -np.inf))
                    else 0.0
                )
                recovery_breadth_condition_met = (
                    1.0
                    if recovery_breadth_excess_5d
                    >= float(confidence_cfg.get("recovery_min_residual_breadth_excess_5d", -np.inf))
                    else 0.0
                )
                recovery_risk_stress_condition_met = (
                    1.0
                    if float(confidence_terms["risk_stress"])
                    <= float(confidence_cfg.get("recovery_max_risk_stress", np.inf))
                    else 0.0
                )
                risk_break_confidence_condition_met = (
                    1.0
                    if float(confidence_terms["confidence_derisk"])
                    >= float(confidence_cfg.get("risk_break_min_confidence_derisk", 0.0))
                    else 0.0
                )
                event_trigger_allowed = 1.0 if self.early_update_cooldown_remaining <= 0 else 0.0
                risk_break_event_allowed = (
                    1.0
                    if (event_trigger_allowed > 0.0 or bool(confidence_cfg.get("risk_break_ignore_cooldown", False)))
                    else 0.0
                )

                recovery_candidate = (
                    recovery_cash_condition_met > 0.0
                    and recovery_anchor_condition_met > 0.0
                    and recovery_confidence_condition_met > 0.0
                    and recovery_residual_condition_met > 0.0
                    and recovery_breadth_condition_met > 0.0
                    and recovery_risk_stress_condition_met > 0.0
                    and float(confidence_terms["recovery_score"])
                    > float(confidence_cfg.get("recovery_trigger_threshold", 0.70))
                )
                if recovery_candidate:
                    recovery_persistence_count += 1
                    recovery_trigger_candidate = 1.0
                else:
                    recovery_persistence_count = 0

                risk_break_candidate = (
                    risk_break_confidence_condition_met > 0.0
                    and float(confidence_terms["risk_stress"])
                    > float(confidence_cfg.get("derisk_early_update_threshold", 0.80))
                )
                if risk_break_candidate:
                    risk_break_persistence_count += 1
                    risk_break_trigger_candidate = 1.0
                else:
                    risk_break_persistence_count = 0

                if (
                    event_trigger_allowed
                    and recovery_persistence_count >= int(confidence_cfg.get("recovery_persistence_days", 1))
                ):
                    recovery_trigger = 1.0
                    rerisk_early_update = 1.0
                    window_closed_early = 1.0
                    early_update_reason = "recovery_trigger"

                if (
                    risk_break_event_allowed
                    and risk_break_persistence_count >= int(confidence_cfg.get("risk_break_persistence_days", 1))
                ):
                    derisk_early_update = 1.0
                    window_closed_early = 1.0
                    early_update_reason = (
                        "recovery_and_risk_break"
                        if early_update_reason == "recovery_trigger"
                        else "risk_break_trigger"
                    )

                slice_cfg = confidence_cfg.get("confidence_slice", {})
                if slice_cfg and bool(slice_cfg.get("enabled", False)):
                    direction = str(confidence_terms.get("k_window_direction", "hold"))
                    if direction == "derisk":
                        direction_confidence = float(confidence_terms["confidence_derisk"])
                    elif direction == "rerisk":
                        direction_confidence = float(confidence_terms["confidence_rerisk"])
                    else:
                        direction_confidence = 1.0
                    min_scale = float(slice_cfg.get(f"{direction}_min_scale", slice_cfg.get("min_scale", 0.0)))
                    max_scale = float(slice_cfg.get(f"{direction}_max_scale", slice_cfg.get("max_scale", 1.0)))
                    base_root_scale = float(
                        np.clip(min_scale + (max_scale - min_scale) * direction_confidence, 0.0, 1.0)
                    )
                    root_scale = base_root_scale
                    partial_stop_enabled = bool(slice_cfg.get("partial_stop_enabled", False))
                    stop_floor = 0.0
                    if stop_active and partial_stop_enabled:
                        stop_floor = float(
                            slice_cfg.get(
                                f"{direction}_stop_floor",
                                slice_cfg.get("stop_floor", 0.0),
                            )
                        )
                        root_scale = float(np.clip(stop_floor, 0.0, 1.0))
                    if direction == "hold" and not bool(slice_cfg.get("apply_to_hold", False)):
                        root_scale = 1.0
                        base_root_scale = 1.0
                        stop_floor = 0.0
                    confidence_slice_terms = {
                        "confidence_slice_enabled": 1.0,
                        "confidence_slice_direction_confidence": direction_confidence,
                        "confidence_slice_base_root_scale": base_root_scale,
                        "confidence_slice_root_scale": root_scale if (not stop_active or partial_stop_enabled) else 0.0,
                        "confidence_slice_stop_floor": stop_floor,
                        "confidence_slice_partial_stop_enabled": 1.0 if partial_stop_enabled else 0.0,
                        "confidence_slice_rerisk_partial_stop_day": (
                            1.0 if stop_active and partial_stop_enabled and direction == "rerisk" else 0.0
                        ),
                        "confidence_slice_derisk_partial_stop_day": (
                            1.0 if stop_active and partial_stop_enabled and direction == "derisk" else 0.0
                        ),
                        "confidence_slice_rerisk_hard_stop_day": (
                            1.0 if stop_active and not partial_stop_enabled and direction == "rerisk" else 0.0
                        ),
                        "confidence_slice_derisk_hard_stop_day": (
                            1.0 if stop_active and not partial_stop_enabled and direction == "derisk" else 0.0
                        ),
                    }
                    if not stop_active or partial_stop_enabled:
                        before_slice = execution_target_weights.copy()
                        execution_target_weights = normalize_simplex(
                            self.previous_weights + root_scale * (execution_target_weights - self.previous_weights)
                        )
                        suppressed_l1 = float(np.sum(np.abs(before_slice - execution_target_weights)))
                        requested_l1 = float(np.sum(np.abs(before_slice - self.previous_weights)))
                        executed_l1 = float(np.sum(np.abs(execution_target_weights - self.previous_weights)))
                        confidence_slice_terms["confidence_slice_suppressed_l1"] = suppressed_l1
                        confidence_slice_terms["confidence_slice_suppressed_turnover"] = 0.5 * suppressed_l1
                        confidence_slice_terms["confidence_slice_requested_l1"] = requested_l1
                        confidence_slice_terms["confidence_slice_executed_l1"] = executed_l1
                        confidence_slice_terms["confidence_slice_rerisk_requested_l1"] = (
                            requested_l1 if direction == "rerisk" else 0.0
                        )
                        confidence_slice_terms["confidence_slice_rerisk_executed_l1"] = (
                            executed_l1 if direction == "rerisk" else 0.0
                        )
                        confidence_slice_terms["confidence_slice_rerisk_suppressed_l1"] = (
                            suppressed_l1 if direction == "rerisk" else 0.0
                        )
                        confidence_slice_terms["confidence_slice_derisk_requested_l1"] = (
                            requested_l1 if direction == "derisk" else 0.0
                        )
                        confidence_slice_terms["confidence_slice_derisk_executed_l1"] = (
                            executed_l1 if direction == "derisk" else 0.0
                        )
                        confidence_slice_terms["confidence_slice_derisk_suppressed_l1"] = (
                            suppressed_l1 if direction == "derisk" else 0.0
                        )
                    else:
                        confidence_slice_terms["confidence_slice_suppressed_l1"] = float(
                            np.sum(np.abs(scheduled_weights - execution_target_weights))
                        )
                        confidence_slice_terms["confidence_slice_suppressed_turnover"] = (
                            0.5 * confidence_slice_terms["confidence_slice_suppressed_l1"]
                        )
                        confidence_slice_terms["confidence_slice_requested_l1"] = float(
                            np.sum(np.abs(scheduled_weights - self.previous_weights))
                        )
                        confidence_slice_terms["confidence_slice_executed_l1"] = 0.0
                        confidence_slice_terms["confidence_slice_rerisk_requested_l1"] = (
                            confidence_slice_terms["confidence_slice_requested_l1"] if direction == "rerisk" else 0.0
                        )
                        confidence_slice_terms["confidence_slice_rerisk_executed_l1"] = 0.0
                        confidence_slice_terms["confidence_slice_rerisk_suppressed_l1"] = (
                            confidence_slice_terms["confidence_slice_suppressed_l1"] if direction == "rerisk" else 0.0
                        )
                        confidence_slice_terms["confidence_slice_derisk_requested_l1"] = (
                            confidence_slice_terms["confidence_slice_requested_l1"] if direction == "derisk" else 0.0
                        )
                        confidence_slice_terms["confidence_slice_derisk_executed_l1"] = 0.0
                        confidence_slice_terms["confidence_slice_derisk_suppressed_l1"] = (
                            confidence_slice_terms["confidence_slice_suppressed_l1"] if direction == "derisk" else 0.0
                        )
                    if stop_active and partial_stop_enabled:
                        suppressed_trade_l1 = float(confidence_slice_terms["confidence_slice_suppressed_l1"])
                        suppressed_turnover = 0.5 * suppressed_trade_l1
                        suppressed_trade_value = float(self.portfolio_value * suppressed_turnover)

                incremental_topk_cfg = cfg.get("incremental_topk_flow", {})
                if incremental_topk_cfg and bool(incremental_topk_cfg.get("enabled", False)):
                    before_incremental_topk = execution_target_weights.copy()
                    execution_target_weights, incremental_topk_terms = self._apply_incremental_topk_flow_target(
                        execution_target_weights,
                        u_anchor,
                        u_window_start,
                        incremental_topk_cfg,
                        {
                            **confidence_terms,
                            "recovery_trigger": recovery_trigger,
                            "recovery_trigger_candidate": recovery_trigger_candidate,
                            "risk_break_trigger": derisk_early_update,
                            "derisk_early_update": derisk_early_update,
                            "risk_break_trigger_candidate": risk_break_trigger_candidate,
                            "rerisk_early_update": rerisk_early_update,
                        },
                    )
                    incremental_topk_terms["incremental_topk_input_to_output_l1"] = float(
                        np.sum(np.abs(before_incremental_topk - execution_target_weights))
                    )
                    incremental_topk_weights = execution_target_weights.copy()

                stock_slice_cfg = confidence_cfg.get("stock_slice", {})
                if stock_slice_cfg and bool(stock_slice_cfg.get("enabled", False)):
                    before_stock_slice = execution_target_weights.copy()
                    execution_target_weights, stock_slice_terms = self._apply_stock_confidence_slice(
                        execution_target_weights,
                        stock_slice_cfg,
                        {
                            "u_window_start": u_window_start,
                            "u_anchor": u_anchor,
                            "root_direction": str(confidence_terms.get("k_window_direction", "hold")),
                        },
                    )
                    stock_slice_terms["stock_slice_input_to_output_l1"] = float(
                        np.sum(np.abs(before_stock_slice - execution_target_weights))
                    )

            daily_action_info = {
                **anchor_info,
                "k_window_enabled": 1.0,
                "k_window_days": float(window_days),
                "k_window_mode": mode,
                "dual_window_enabled": 1.0 if dual_enabled else 0.0,
                "dual_root_window_days": float(root_window_days if dual_enabled else window_days),
                "dual_stock_window_days": float(stock_window_days if dual_enabled else window_days),
                "dual_root_anchor_refreshed": root_refreshed if substep == 0 else 0.0,
                "dual_stock_anchor_refreshed": 1.0 if dual_enabled and substep == 0 else 0.0,
                "dual_q_action": q_action,
                "dual_cash_action": 1.0 - q_action,
                "dual_q_action_ignored": q_action_ignored if substep == 0 else 0.0,
                "dual_root_anchor_start_day": float(self.dual_root_anchor_start_day if dual_enabled else start_day),
                "dual_root_elapsed_days": float(root_elapsed_now if dual_enabled else substep),
                "dual_root_remaining_days": float(root_remaining_days),
                "dual_stock_remaining_days": float(stock_remaining_days),
                "k_window_start_day": float(start_day),
                "k_window_planned_end_day": float(planned_end_day),
                "k_window_substep": float(substep + 1),
                "window_day": float(substep + 1),
                "k_window_remaining_days": float(remaining_days),
                "remaining_days": float(remaining_days),
                "k_window_effective_days": 0.0,
                "effective_K": 0.0,
                "q_target": q_anchor,
                "cash_target": 1.0 - q_anchor,
                "q_anchor": q_anchor,
                "cash_anchor": 1.0 - q_anchor,
                "q_scheduled": float(np.sum(scheduled_weights[: self.stock_dim])),
                "cash_scheduled": float(scheduled_weights[self.cash_index]),
                "anchor_to_schedule_l1": float(np.sum(np.abs(anchor_weights - scheduled_weights))),
                "target_to_schedule_gap": float(np.sum(np.abs(anchor_weights - scheduled_weights))),
                "raw_to_anchor_l1": float(np.sum(np.abs(raw_anchor_weights - anchor_weights))),
                "schedule_to_target_l1": float(np.sum(np.abs(scheduled_weights - execution_target_weights))),
                "confidence_stop_recovery_enabled": 1.0 if confidence_enabled else 0.0,
                "event_trigger_allowed": event_trigger_allowed,
                "risk_break_event_allowed": risk_break_event_allowed,
                "early_update_cooldown_remaining": cooldown_remaining_at_start,
                "stop_active": stop_active,
                "suppressed_trade_l1": suppressed_trade_l1,
                "suppressed_turnover": suppressed_turnover,
                "suppressed_trade_value": suppressed_trade_value,
                "recovery_trigger_candidate": recovery_trigger_candidate,
                "risk_break_trigger_candidate": risk_break_trigger_candidate,
                "recovery_persistence_count": float(recovery_persistence_count),
                "risk_break_persistence_count": float(risk_break_persistence_count),
                "recovery_cash_condition_met": recovery_cash_condition_met,
                "recovery_anchor_condition_met": recovery_anchor_condition_met,
                "recovery_confidence_condition_met": recovery_confidence_condition_met,
                "recovery_residual_condition_met": recovery_residual_condition_met,
                "recovery_breadth_condition_met": recovery_breadth_condition_met,
                "recovery_risk_stress_condition_met": recovery_risk_stress_condition_met,
                "recovery_residual_up_5d": recovery_residual_up_5d,
                "recovery_breadth_excess_5d": recovery_breadth_excess_5d,
                "risk_break_confidence_condition_met": risk_break_confidence_condition_met,
                "recovery_trigger": recovery_trigger,
                "recovery_trigger_day": float(substep + 1) if recovery_trigger else 0.0,
                "risk_break_trigger": derisk_early_update,
                "risk_break_trigger_day": float(substep + 1) if derisk_early_update else 0.0,
                "derisk_early_update": derisk_early_update,
                "derisk_early_update_day": float(substep + 1) if derisk_early_update else 0.0,
                "rerisk_early_update": rerisk_early_update,
                "rerisk_early_update_day": float(substep + 1) if rerisk_early_update else 0.0,
                "window_closed_early": window_closed_early,
                **sparse_topk_terms,
                **confidence_slice_terms,
                **incremental_topk_terms,
                **stock_slice_terms,
                **confidence_terms,
            }

            reward, terminated, info = self._step_daily_target(execution_target_weights, action_info=daily_action_info)
            executed_weights_for_gap = np.asarray(info["executed_weights"], dtype=np.float64)
            schedule_to_exec = float(np.sum(np.abs(scheduled_weights - executed_weights_for_gap)))
            info["schedule_to_exec_l1"] = schedule_to_exec
            info["schedule_to_exec_gap"] = schedule_to_exec
            info["anchor_weights"] = anchor_weights.astype(np.float32)
            info["scheduled_weights"] = scheduled_weights.astype(np.float32)
            info["raw_weights"] = raw_anchor_weights.astype(np.float32)
            if incremental_topk_weights is not None:
                info["incremental_topk_weights"] = incremental_topk_weights.astype(np.float32)
            info["stop_reason"] = stop_reason
            info["early_update_reason"] = early_update_reason
            daily_infos.append(info)
            total_reward += reward

            if window_closed_early:
                self.early_update_cooldown_remaining = int(confidence_cfg.get("early_update_cooldown_days", 0))
                if dual_enabled:
                    self.dual_root_force_refresh = True
            elif self.early_update_cooldown_remaining > 0:
                self.early_update_cooldown_remaining -= 1
            if dual_enabled and (int(self.day) - int(self.dual_root_anchor_start_day)) >= root_window_days:
                self.dual_root_force_refresh = True

            if terminated or window_closed_early:
                break

        if daily_infos:
            effective_days = float(len(daily_infos))
            self.internal_trading_days_processed += effective_days
            for daily_info in daily_infos:
                daily_info["k_window_effective_days"] = effective_days
                daily_info["effective_K"] = effective_days
                daily_info["internal_trading_days_this_step"] = 1.0
                daily_info["internal_trading_days_processed"] = self.internal_trading_days_processed
            macro_info = dict(daily_infos[-1])
            macro_info["daily_steps"] = daily_infos
            macro_info["macro_reward"] = total_reward
            macro_info["k_window_effective_days"] = effective_days
            macro_info["effective_K"] = effective_days
            macro_info["internal_trading_days_this_step"] = effective_days
            macro_info["internal_trading_days_processed"] = self.internal_trading_days_processed
            macro_info["k_window_start_date"] = str(pd.Timestamp(self.panel.dates[start_day]).date())
            macro_info["k_window_end_date"] = str(pd.Timestamp(self.panel.dates[self.day - 1]).date())
        else:
            macro_info = {
                **self._info_base(),
                "daily_steps": [],
                "macro_reward": 0.0,
                "k_window_enabled": 1.0,
                "k_window_days": float(window_days),
                "k_window_mode": mode,
                "k_window_effective_days": 0.0,
                "internal_trading_days_this_step": 0.0,
                "internal_trading_days_processed": self.internal_trading_days_processed,
            }
            terminated = True

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(total_reward), terminated, False, macro_info

    def _get_obs(self) -> np.ndarray:
        parts = []
        if self.include_features:
            parts.append(self.panel.features[self.day].reshape(-1).astype(np.float32))
        if self.include_previous_weights:
            parts.append(self.previous_weights.astype(np.float32))
        if self.include_portfolio_state:
            gross_exposure = float(np.sum(np.abs(self.previous_weights[: self.stock_dim])))
            hhi = float(np.sum(self.previous_weights**2))
            portfolio_return_since_start = self.portfolio_value / self.initial_amount - 1.0
            state = np.array(
                [
                    self.previous_weights[self.cash_index],
                    gross_exposure,
                    hhi,
                    self.previous_drawdown,
                    self.last_turnover,
                    portfolio_return_since_start,
                ],
                dtype=np.float32,
            )
            parts.append(state)
        return np.concatenate(parts).astype(np.float32)

    def _action_to_target_weights(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if self.action_transform == "direct_weights":
            return normalize_simplex(a)
        if self.action_transform == "flat_softmax":
            return softmax(a)
        if self.action_transform == "hierarchical_softmax":
            return self._hierarchical_softmax(a)
        if self.action_transform == "root_split_weights":
            return self._root_split_weights(a)
        if self.action_transform == "root_split_kp_weights":
            return self._root_split_kp_weights(a)
        if self.action_transform == "riskcash_sector_dirtree_factors":
            return self._riskcash_sector_dirtree_weights(a)
        if self.action_transform == "style_mixture_weights":
            return self._style_mixture_weights(a)
        raise ValueError(f"Unknown action_transform: {self.action_transform}")

    def _root_split_weights(self, action: np.ndarray) -> np.ndarray:
        if action.shape[0] != self.asset_dim:
            raise ValueError(f"root_split_weights expects {self.asset_dim} factors, got {action.shape[0]}")
        q_target = float(np.clip(action[0], 0.0, 1.0))
        risky = normalize_simplex(action[1:])
        target = np.zeros(self.asset_dim, dtype=np.float64)
        target[: self.stock_dim] = q_target * risky
        target[self.cash_index] = 1.0 - q_target
        self.last_action_info = {
            "q_target": q_target,
            "cash_target": 1.0 - q_target,
            "risky_hhi_target": float(np.sum(risky**2)),
            "risky_max_weight_target": float(np.max(risky)) if risky.size else 0.0,
            "risky_entropy_target": float(-np.sum(risky * np.log(np.maximum(risky, EPS)))) if risky.size else 0.0,
        }
        return normalize_simplex(target)

    def _root_split_kp_weights(self, action: np.ndarray) -> np.ndarray:
        if action.shape[0] != self.asset_dim + 2:
            raise ValueError(
                f"root_split_kp_weights expects {self.asset_dim + 2} factors, got {action.shape[0]}"
            )
        q_target = float(np.clip(action[0], 0.0, 1.0))
        risky = normalize_stock_simplex(action[1 : 1 + self.stock_dim])
        z_root_gate = float(np.clip(action[1 + self.stock_dim], 0.0, 1.0))
        z_inner_gate = float(np.clip(action[2 + self.stock_dim], 0.0, 1.0))
        target = np.zeros(self.asset_dim, dtype=np.float64)
        target[: self.stock_dim] = q_target * risky
        target[self.cash_index] = 1.0 - q_target
        self.last_action_info = {
            "q_target": q_target,
            "cash_target": 1.0 - q_target,
            "z_root_gate": z_root_gate,
            "z_inner_gate": z_inner_gate,
            "risky_hhi_target": float(np.sum(risky**2)),
            "risky_max_weight_target": float(np.max(risky)) if risky.size else 0.0,
            "risky_entropy_target": float(-np.sum(risky * np.log(np.maximum(risky, EPS)))) if risky.size else 0.0,
        }
        return normalize_simplex(target)

    def _riskcash_sector_dirtree_weights(self, action: np.ndarray) -> np.ndarray:
        expected = self._action_dim()
        if action.shape[0] != expected:
            raise ValueError(f"riskcash_sector_dirtree_factors expects {expected} factors, got {action.shape[0]}")
        q_target = float(np.clip(action[0], 0.0, 1.0))
        noncash_groups = [group for group in self.group_names if group != "cash"]
        offset = 1
        group_weights = normalize_stock_simplex(action[offset : offset + len(noncash_groups)])
        offset += len(noncash_groups)

        target = np.zeros(self.asset_dim, dtype=np.float64)
        action_info: dict[str, float] = {
            "q_target": q_target,
            "cash_target": 1.0 - q_target,
            "tree_group_hhi_target": float(np.sum(group_weights**2)),
            "tree_group_max_weight_target": float(np.max(group_weights)) if group_weights.size else 0.0,
            "tree_group_entropy_target": float(-np.sum(group_weights * np.log(np.maximum(group_weights, EPS))))
            if group_weights.size
            else 0.0,
        }

        for group_idx, group in enumerate(noncash_groups):
            indices = self.group_to_indices[group]
            group_budget = q_target * float(group_weights[group_idx])
            action_info[f"group_target_{group}"] = float(group_weights[group_idx])
            if len(indices) == 1:
                target[indices[0]] = group_budget
                action_info[f"within_hhi_{group}"] = 1.0
                action_info[f"within_entropy_{group}"] = 0.0
                action_info[f"within_max_{group}"] = 1.0
                continue
            inner = normalize_stock_simplex(action[offset : offset + len(indices)])
            offset += len(indices)
            target[np.asarray(indices, dtype=int)] = group_budget * inner
            action_info[f"within_hhi_{group}"] = float(np.sum(inner**2))
            action_info[f"within_entropy_{group}"] = float(-np.sum(inner * np.log(np.maximum(inner, EPS))))
            action_info[f"within_max_{group}"] = float(np.max(inner))

        target[self.cash_index] = 1.0 - q_target
        self.last_action_info = action_info
        return normalize_simplex(target)

    def _trailing_stock_returns(self, window: int) -> np.ndarray:
        if self.day <= 0:
            return np.zeros(self.stock_dim, dtype=np.float64)
        start = max(0, self.day - int(window))
        start_prices = np.maximum(self.panel.prices[start], EPS)
        return self.panel.prices[self.day] / start_prices - 1.0

    def _trailing_realized_vol(self, window: int = 20) -> np.ndarray:
        if self.day <= 1:
            return np.ones(self.stock_dim, dtype=np.float64)
        start = max(0, self.day - int(window))
        prices = self.panel.prices[start : self.day + 1]
        if len(prices) < 2:
            return np.ones(self.stock_dim, dtype=np.float64)
        returns = prices[1:] / np.maximum(prices[:-1], EPS) - 1.0
        vol = np.std(returns, axis=0, ddof=1) if returns.shape[0] > 1 else np.abs(returns[0])
        return np.maximum(np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0), 1e-4)

    def _ranked_stock_weights(self, scores: np.ndarray, top_k: int, temperature: float) -> np.ndarray:
        scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        top_k = max(1, min(int(top_k), self.stock_dim))
        selected = np.argsort(scores)[-top_k:]
        logits = scores[selected] / max(float(temperature), EPS)
        weights = np.zeros(self.stock_dim, dtype=np.float64)
        weights[selected] = softmax(logits)
        return normalize_stock_simplex(weights)

    def _stock_feature_vector(self, feature: str, default: float = 0.0) -> np.ndarray:
        idx = self.feature_index.get(feature)
        if idx is None:
            return np.full(self.stock_dim, float(default), dtype=np.float64)
        return self.panel.features[self.day, :, idx].astype(np.float64)

    def _style_target(self, style: str, style_cfg: dict[str, Any], risk_score: float) -> np.ndarray:
        style = str(style)
        cash_base = float(style_cfg.get("cash_base", 0.05))
        top_k = int(style_cfg.get("top_k", 8))
        temperature = float(style_cfg.get("temperature", 0.05))
        vol = self._trailing_realized_vol(int(style_cfg.get("vol_window", 20)))
        inv_vol = normalize_stock_simplex(1.0 / np.maximum(vol, 1e-4))

        if style == "equal_weight":
            cash = cash_base
            risky = np.full(self.stock_dim, 1.0 / self.stock_dim, dtype=np.float64)
        elif style == "risk_off_cash":
            cash = float(style_cfg.get("risk_off_cash_low", 0.05)) + (
                float(style_cfg.get("risk_off_cash_high", 0.35)) - float(style_cfg.get("risk_off_cash_low", 0.05))
            ) * risk_score
            risky = inv_vol
        elif style == "low_volatility":
            cash = cash_base
            risky = inv_vol
        elif style == "momentum_20d":
            cash = cash_base
            risky = self._ranked_stock_weights(self._trailing_stock_returns(20), top_k, temperature)
        elif style == "momentum_60d":
            cash = cash_base
            risky = self._ranked_stock_weights(self._trailing_stock_returns(60), top_k, temperature)
        elif style == "short_reversal_5d":
            cash = cash_base
            risky = self._ranked_stock_weights(-self._trailing_stock_returns(5), top_k, temperature)
        elif style == "sector_balanced":
            cash = cash_base
            risky = np.zeros(self.stock_dim, dtype=np.float64)
            noncash_groups = [group for group in self.group_names if group != "cash"]
            for group in noncash_groups:
                indices = self.group_to_indices[group]
                risky[np.asarray(indices, dtype=int)] = 1.0 / max(len(noncash_groups) * len(indices), 1)
            risky = normalize_stock_simplex(risky)
        elif style == "cash_preservation":
            cash = float(style_cfg.get("preservation_cash_low", 0.20)) + (
                float(style_cfg.get("preservation_cash_high", 0.55))
                - float(style_cfg.get("preservation_cash_low", 0.20))
            ) * risk_score
            risky = inv_vol
        elif style == "value_quality":
            pe = self._stock_feature_vector("PE_ratio")
            pb = self._stock_feature_vector("PB_ratio")
            debt = self._stock_feature_vector("debt_ratio")
            score = -(pe + pb + debt)
            cash = cash_base
            risky = self._ranked_stock_weights(score, top_k, max(temperature, 0.25))
        else:
            cash = cash_base
            risky = np.full(self.stock_dim, 1.0 / self.stock_dim, dtype=np.float64)

        cash = float(np.clip(cash, 0.0, 0.95))
        target = np.zeros(self.asset_dim, dtype=np.float64)
        target[: self.stock_dim] = (1.0 - cash) * normalize_stock_simplex(risky)
        target[self.cash_index] = cash
        return normalize_simplex(target)

    def _style_mixture_weights(self, action: np.ndarray) -> np.ndarray:
        style_cfg = self.root_split_config.get("style_bank", {})
        styles = list(
            style_cfg.get("styles")
            or [
                "equal_weight",
                "risk_off_cash",
                "low_volatility",
                "momentum_20d",
                "momentum_60d",
                "short_reversal_5d",
                "sector_balanced",
                "cash_preservation",
            ]
        )
        if action.shape[0] != len(styles):
            raise ValueError(f"style_mixture_weights expects {len(styles)} factors, got {action.shape[0]}")
        style_weights = normalize_stock_simplex(action)
        risk_score, _, _, _ = self._cash_prior_terms(0.0)
        if not np.isfinite(risk_score):
            risk_score = 0.5
        style_targets = [self._style_target(style, style_cfg, risk_score) for style in styles]
        target = np.zeros(self.asset_dim, dtype=np.float64)
        previous_style_weights = self.previous_style_weights
        if previous_style_weights is None or previous_style_weights.shape != style_weights.shape:
            previous_style_weights = style_weights
        action_info: dict[str, float] = {
            "style_entropy": float(-np.sum(style_weights * np.log(np.maximum(style_weights, EPS)))),
            "style_top_weight": float(np.max(style_weights)) if style_weights.size else 0.0,
            "style_top_index": float(np.argmax(style_weights)) if style_weights.size else -1.0,
            "style_turnover": float(np.sum(np.abs(style_weights - previous_style_weights))),
        }
        for idx, (style, weight, style_target) in enumerate(zip(styles, style_weights, style_targets)):
            safe_name = str(style).replace("-", "_").replace(" ", "_")
            target += float(weight) * style_target
            action_info[f"style_weight_{safe_name}"] = float(weight)
            action_info[f"style_cash_{safe_name}"] = float(style_target[self.cash_index])
            action_info[f"style_contribution_cash_{safe_name}"] = float(weight * style_target[self.cash_index])
            action_info[f"style_index_{idx}"] = float(weight)
        self.last_action_info = action_info
        self.previous_style_weights = style_weights.copy()
        return normalize_simplex(target)

    def _hierarchical_softmax(self, action: np.ndarray) -> np.ndarray:
        group_count = len(self.group_names)
        group_weights = softmax(action[:group_count])
        target = np.zeros(self.asset_dim, dtype=np.float64)
        offset = group_count
        for group_idx, group in enumerate(self.group_names):
            budget = group_weights[group_idx]
            indices = self.group_to_indices[group]
            if group == "cash":
                target[self.cash_index] = budget
                continue
            logits = action[offset : offset + len(indices)]
            offset += len(indices)
            inner = softmax(logits)
            target[np.asarray(indices, dtype=int)] = budget * inner
        return normalize_simplex(target)

    def _apply_controller(self, target_weights: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        target_weights, safety_terms = self._apply_target_safety_layers(target_weights)
        root_execution = self.root_split_config.get("execution", {})
        if (
            self.action_transform in {"root_split_weights", "root_split_kp_weights"}
            and root_execution
            and bool(root_execution.get("enabled", False))
        ):
            executed, terms = self._apply_root_split_execution(target_weights, root_execution)
            execution_penalty = float(terms.get("execution_penalty", 0.0)) + float(
                safety_terms.get("safety_penalty", 0.0)
            )
            terms.update(safety_terms)
            terms["execution_penalty"] = execution_penalty
            return executed, terms

        error = target_weights - self.previous_weights
        self.integral_error = np.clip(self.integral_error + error, -self.integral_clip, self.integral_clip)
        derivative = np.clip(error - self.previous_error, -self.derivative_clip, self.derivative_clip)

        kp = float(self.controller.get("kp", 1.0))
        ki = float(self.controller.get("ki", 0.0))
        kd = float(self.controller.get("kd", 0.0))
        ctrl_type = str(self.controller.get("type", "P")).upper()

        p_term = kp * error
        i_term = ki * self.integral_error if "I" in ctrl_type else np.zeros_like(error)
        d_term = kd * derivative if "D" in ctrl_type else np.zeros_like(error)
        raw = self.previous_weights + p_term + i_term + d_term

        delta = raw - self.previous_weights
        delta_l1 = float(np.sum(np.abs(delta)))
        if delta_l1 > self.turnover_cap > 0:
            raw = self.previous_weights + delta * (self.turnover_cap / delta_l1)

        projected = project_to_simplex(raw)
        projection_residual = float(np.linalg.norm(projected - raw, ord=1))
        self.previous_error = error
        self.last_projection_residual = projection_residual

        terms = {
            "controller_p_l1": float(np.sum(np.abs(p_term))),
            "controller_i_l1": float(np.sum(np.abs(i_term))),
            "controller_d_l1": float(np.sum(np.abs(d_term))),
            "controller_delta_l1_before_cap": delta_l1,
            "projection_residual_l1": projection_residual,
            **safety_terms,
        }
        if "safety_penalty" in safety_terms:
            terms["execution_penalty"] = float(safety_terms["safety_penalty"])
        return projected, terms

    def _apply_target_safety_layers(self, target_weights: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        adjusted = normalize_simplex(target_weights)
        terms: dict[str, Any] = {}

        bottomup_cfg = self.root_split_config.get("bottomup_veto", {})
        if bottomup_cfg and bool(bottomup_cfg.get("enabled", False)):
            adjusted, veto_terms = self._apply_bottomup_veto(adjusted, bottomup_cfg)
            terms.update(veto_terms)

        projection_cfg = self.root_split_config.get("projection_safety", {})
        if projection_cfg and bool(projection_cfg.get("enabled", False)):
            adjusted, projection_terms = self._apply_projection_safety_layer(adjusted, projection_cfg)
            existing_penalty = float(terms.get("safety_penalty", 0.0))
            projection_penalty = float(projection_terms.get("safety_penalty", 0.0))
            terms.update(projection_terms)
            terms["safety_penalty"] = existing_penalty + projection_penalty

        return normalize_simplex(adjusted), terms

    def _feature_group_mean(self, feature: str, indices: list[int], positive_only: bool = False) -> float:
        idx = self.feature_index.get(str(feature))
        if idx is None or not indices:
            return 0.0
        values = self.panel.features[self.day, np.asarray(indices, dtype=int), idx].astype(np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        value = float(np.mean(values))
        return max(value, 0.0) if positive_only else value

    def _apply_bottomup_veto(
        self,
        target_weights: np.ndarray,
        cfg: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        noncash_groups = [group for group in self.group_names if group != "cash"]
        q_raw = float(np.clip(np.sum(target_weights[: self.stock_dim]), 0.0, 1.0))
        if q_raw <= EPS or not noncash_groups:
            return target_weights, {
                "bottomup_veto_enabled": 1.0,
                "bottomup_raw_to_safe_l1": 0.0,
                "bottomup_feedback_mean": 0.0,
                "bottomup_feedback_max": 0.0,
                "bottomup_feedback_global": 0.0,
                "bottomup_feedback_active_rate": 0.0,
                "safety_penalty": 0.0,
            }

        f_max = float(cfg.get("f_max", 0.50))
        intercept = float(cfg.get("intercept", -3.0))
        active_threshold = float(cfg.get("active_threshold", 0.05))
        lambda_group = float(cfg.get("lambda_group", 0.50))
        lambda_global = float(cfg.get("lambda_global", 0.30))
        q_min = float(cfg.get("q_min", self.root_split_config.get("q_min", 0.00)))
        q_max = float(cfg.get("q_max", self.root_split_config.get("q_max", 0.995)))

        raw_group_weights = []
        inner_by_group: dict[str, np.ndarray] = {}
        feedback_values = []
        terms: dict[str, Any] = {"bottomup_veto_enabled": 1.0}
        for group in noncash_groups:
            indices = self.group_to_indices[group]
            stock_slice = target_weights[np.asarray(indices, dtype=int)]
            group_abs = float(np.sum(stock_slice))
            group_cond = group_abs / max(q_raw, EPS)
            raw_group_weights.append(group_cond)
            inner = normalize_stock_simplex(stock_slice / max(group_abs, EPS)) if group_abs > EPS else np.full(
                len(indices), 1.0 / max(len(indices), 1), dtype=np.float64
            )
            inner_by_group[group] = inner

            hhi = float(np.sum(inner**2))
            max_weight = float(np.max(inner)) if inner.size else 0.0
            equal_hhi = 1.0 / max(len(indices), 1)
            hhi_excess = max((hhi - equal_hhi) / max(1.0 - equal_hhi, EPS), 0.0)
            maxw_allowed = float(cfg.get("max_inner_weight_allowed", max(0.35, equal_hhi)))
            maxw_excess = max((max_weight - maxw_allowed) / max(1.0 - maxw_allowed, EPS), 0.0)
            prev_group = self.previous_weights[np.asarray(indices, dtype=int)]
            turnover_needed = float(np.sum(np.abs(stock_slice - prev_group)))
            turnover_scale = max(float(cfg.get("turnover_scale", 0.10)), EPS)
            turnover_z = turnover_needed / turnover_scale
            vol_z = self._feature_group_mean(str(cfg.get("vol_feature", "realized_vol_20d")), indices, positive_only=True)
            liquidity_raw = self._feature_group_mean(str(cfg.get("liquidity_feature", "volume_ratio")), indices)
            liquidity_stress = max(-liquidity_raw, 0.0)

            score = (
                intercept
                + max(float(cfg.get("beta_hhi", 1.0)), 0.0) * hhi_excess
                + max(float(cfg.get("beta_maxw", 1.5)), 0.0) * maxw_excess
                + max(float(cfg.get("beta_turnover", 0.8)), 0.0) * turnover_z
                + max(float(cfg.get("beta_vol", 0.4)), 0.0) * vol_z
                + max(float(cfg.get("beta_liquidity", 0.3)), 0.0) * liquidity_stress
            )
            feedback = f_max * sigmoid_scalar(score)
            feedback_values.append(feedback)
            terms[f"bottomup_feedback_{group}"] = float(feedback)
            terms[f"bottomup_turnover_needed_{group}"] = turnover_needed

        g_raw = normalize_stock_simplex(np.asarray(raw_group_weights, dtype=np.float64))
        feedback_arr = np.asarray(feedback_values, dtype=np.float64)
        g_safe = normalize_stock_simplex(g_raw * np.exp(-lambda_group * feedback_arr))
        f_global = float(np.clip(np.sum(g_raw * feedback_arr), 0.0, f_max))
        q_safe = float(np.clip(q_raw * (1.0 - lambda_global * f_global), q_min, q_max))

        safe = np.zeros(self.asset_dim, dtype=np.float64)
        for group_idx, group in enumerate(noncash_groups):
            indices = self.group_to_indices[group]
            safe[np.asarray(indices, dtype=int)] = q_safe * float(g_safe[group_idx]) * inner_by_group[group]
        safe[self.cash_index] = 1.0 - q_safe
        safe = normalize_simplex(safe)
        raw_to_safe_l1 = float(np.sum(np.abs(target_weights - safe)))
        penalty = float(cfg.get("lambda_veto_gap", 0.0)) * raw_to_safe_l1 + float(
            cfg.get("lambda_feedback", 0.0)
        ) * float(np.mean(feedback_arr))

        terms.update(
            {
                "bottomup_q_raw": q_raw,
                "bottomup_q_safe": q_safe,
                "bottomup_cash_raw": 1.0 - q_raw,
                "bottomup_cash_safe": 1.0 - q_safe,
                "bottomup_cash_delta": (1.0 - q_safe) - (1.0 - q_raw),
                "bottomup_raw_to_safe_l1": raw_to_safe_l1,
                "bottomup_feedback_mean": float(np.mean(feedback_arr)),
                "bottomup_feedback_max": float(np.max(feedback_arr)),
                "bottomup_feedback_global": f_global,
                "bottomup_feedback_active_rate": float(np.mean(feedback_arr > active_threshold)),
                "bottomup_group_shift_l1": float(np.sum(np.abs(g_raw - g_safe))),
                "safety_penalty": penalty,
            }
        )
        return safe, terms

    def _constraint_violation(self, weights: np.ndarray, cfg: dict[str, Any], risk_score: float) -> float:
        cash = float(weights[self.cash_index])
        cash_min_low = float(cfg.get("cash_min_low", 0.02))
        cash_min_high = float(cfg.get("cash_min_high", 0.05))
        cash_max_low = float(cfg.get("cash_max_low", 0.15))
        cash_max_high = float(cfg.get("cash_max_high", 0.40))
        cash_min = cash_min_low + (cash_min_high - cash_min_low) * risk_score
        cash_max = cash_max_low + (cash_max_high - cash_max_low) * risk_score
        violation = max(cash_min - cash, 0.0) + max(cash - cash_max, 0.0)
        max_stock = float(cfg.get("max_stock_weight", 0.15))
        violation += float(np.sum(np.maximum(weights[: self.stock_dim] - max_stock, 0.0)))
        max_group = float(cfg.get("max_group_weight", 0.45))
        for group in self.group_names:
            if group == "cash":
                continue
            indices = self.group_to_indices[group]
            violation += max(float(np.sum(weights[np.asarray(indices, dtype=int)])) - max_group, 0.0)
        turnover = float(np.sum(np.abs(weights - self.previous_weights)))
        turnover_limit = float(cfg.get("turnover_limit_calm", 0.05)) + (
            float(cfg.get("turnover_limit_stress", 0.10)) - float(cfg.get("turnover_limit_calm", 0.05))
        ) * risk_score
        violation += max(turnover - turnover_limit, 0.0)
        return violation

    def _apply_projection_safety_layer(
        self,
        target_weights: np.ndarray,
        cfg: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raw = normalize_simplex(target_weights)
        risk_score, _, _, _ = self._cash_prior_terms(0.0)
        if not np.isfinite(risk_score):
            risk_score = float(cfg.get("default_risk_score", 0.5))

        cash_min = float(cfg.get("cash_min_low", 0.02)) + (
            float(cfg.get("cash_min_high", 0.05)) - float(cfg.get("cash_min_low", 0.02))
        ) * risk_score
        cash_max = float(cfg.get("cash_max_low", 0.15)) + (
            float(cfg.get("cash_max_high", 0.40)) - float(cfg.get("cash_max_low", 0.15))
        ) * risk_score
        cash_min, cash_max = sorted((float(np.clip(cash_min, 0.0, 0.95)), float(np.clip(cash_max, 0.0, 0.95))))
        q_prev = float(np.sum(self.previous_weights[: self.stock_dim]))
        q_raw = float(np.sum(raw[: self.stock_dim]))
        max_rerisk_delta = float(cfg.get("max_rerisk_delta", 0.05))
        max_derisk_delta = float(cfg.get("max_derisk_delta", 0.15))
        q_low = max(1.0 - cash_max, q_prev - max_derisk_delta, 0.0)
        q_high = min(1.0 - cash_min, q_prev + max_rerisk_delta, 1.0)
        if q_low > q_high:
            q_low, q_high = q_high, q_low
        q_safe = float(np.clip(q_raw, q_low, q_high))
        stock_book = normalize_stock_simplex(raw[: self.stock_dim]) * q_safe

        max_stock = float(cfg.get("max_stock_weight", 0.15))
        stock_book = cap_and_redistribute(stock_book, max_stock, q_safe)

        max_group = float(cfg.get("max_group_weight", 0.45))
        for _ in range(4):
            group_weights = []
            noncash_groups = [group for group in self.group_names if group != "cash"]
            for group in noncash_groups:
                indices = self.group_to_indices[group]
                group_weights.append(float(np.sum(stock_book[np.asarray(indices, dtype=int)])))
            group_weights_arr = np.asarray(group_weights, dtype=np.float64)
            over = group_weights_arr > max_group
            if not np.any(over):
                break
            excess = 0.0
            for group_idx, group in enumerate(noncash_groups):
                if not over[group_idx]:
                    continue
                indices = np.asarray(self.group_to_indices[group], dtype=int)
                current = float(np.sum(stock_book[indices]))
                if current <= EPS:
                    continue
                scale = max_group / current
                excess += current - max_group
                stock_book[indices] *= scale
            under_groups = [idx for idx, value in enumerate(group_weights_arr) if value < max_group and not over[idx]]
            capacity = np.asarray([max_group - group_weights_arr[idx] for idx in under_groups], dtype=np.float64)
            capacity_sum = float(np.sum(capacity))
            if excess <= EPS or capacity_sum <= EPS:
                break
            for local_idx, group_idx in enumerate(under_groups):
                group = noncash_groups[group_idx]
                indices = np.asarray(self.group_to_indices[group], dtype=int)
                add_budget = excess * capacity[local_idx] / capacity_sum
                if float(np.sum(stock_book[indices])) > EPS:
                    stock_book[indices] += add_budget * normalize_stock_simplex(stock_book[indices])
                else:
                    stock_book[indices] += add_budget / max(len(indices), 1)
            stock_book = cap_and_redistribute(stock_book, max_stock, q_safe)

        projected = np.zeros(self.asset_dim, dtype=np.float64)
        projected[: self.stock_dim] = stock_book
        projected[self.cash_index] = 1.0 - float(np.sum(stock_book))
        projected = normalize_simplex(projected)

        turnover_limit = float(cfg.get("turnover_limit_calm", 0.05)) + (
            float(cfg.get("turnover_limit_stress", 0.10)) - float(cfg.get("turnover_limit_calm", 0.05))
        ) * risk_score
        delta = projected - self.previous_weights
        delta_l1 = float(np.sum(np.abs(delta)))
        turnover_scale = 1.0
        if delta_l1 > turnover_limit > 0.0:
            turnover_scale = turnover_limit / delta_l1
            projected = normalize_simplex(self.previous_weights + delta * turnover_scale)

        raw_violation = self._constraint_violation(raw, cfg, risk_score)
        projected_violation = self._constraint_violation(projected, cfg, risk_score)
        gap_l1 = float(np.sum(np.abs(projected - raw)))
        gap_l2 = float(np.linalg.norm(projected - raw, ord=2))
        penalty = (
            float(cfg.get("lambda_projection_gap", 0.0)) * gap_l1
            + float(cfg.get("lambda_raw_violation", 0.0)) * raw_violation
        )
        stock_bound_count = int(np.sum(raw[: self.stock_dim] > max_stock))
        group_bound_count = 0
        for group in self.group_names:
            if group == "cash":
                continue
            indices = self.group_to_indices[group]
            group_bound_count += int(float(np.sum(raw[np.asarray(indices, dtype=int)])) > max_group)

        terms = {
            "projection_safety_enabled": 1.0,
            "safety_projection_gap_l1": gap_l1,
            "safety_projection_gap_l2": gap_l2,
            "safety_projection_active": float(gap_l1 > float(cfg.get("active_threshold", 1e-4))),
            "safety_raw_violation": raw_violation,
            "safety_projected_violation": projected_violation,
            "safety_cash_min": cash_min,
            "safety_cash_max": cash_max,
            "safety_cash_raw": float(raw[self.cash_index]),
            "safety_cash_projected": float(projected[self.cash_index]),
            "safety_q_raw": q_raw,
            "safety_q_projected": float(np.sum(projected[: self.stock_dim])),
            "safety_turnover_limit": turnover_limit,
            "safety_turnover_raw": float(np.sum(np.abs(raw - self.previous_weights))),
            "safety_turnover_projected": float(np.sum(np.abs(projected - self.previous_weights))),
            "safety_turnover_scale": turnover_scale,
            "safety_stock_bound_active_count": float(stock_bound_count),
            "safety_group_bound_active_count": float(group_bound_count),
            "safety_rerisk_bound_active": float(q_raw > q_prev + max_rerisk_delta),
            "safety_derisk_bound_active": float(q_raw < q_prev - max_derisk_delta),
            "safety_penalty": penalty,
        }
        return projected, terms

    def _apply_root_split_execution(
        self,
        target_weights: np.ndarray,
        execution_cfg: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Execute root-split factors with separate root and risky-book speeds.

        The stochastic policy still samples raw action factors (`q_target`, `u_target`).
        This deterministic layer only controls how fast the environment moves from
        previous executed weights toward the target.
        """
        q_target = float(np.clip(np.sum(target_weights[: self.stock_dim]), 0.0, 1.0))
        q_prev = float(np.clip(np.sum(self.previous_weights[: self.stock_dim]), 0.0, 1.0))
        u_target = normalize_stock_simplex(target_weights[: self.stock_dim])
        if q_prev > EPS:
            u_prev = normalize_stock_simplex(self.previous_weights[: self.stock_dim] / q_prev)
        else:
            u_prev = np.array(u_target, copy=True)

        delta_q = q_target - q_prev
        is_derisk = delta_q < 0.0
        learned_gates = execution_cfg.get("learned_gates", {})
        learned_gates_enabled = bool(learned_gates.get("enabled", False))
        z_root_gate = float(self.last_action_info.get("z_root_gate", np.nan))
        z_inner_gate = float(self.last_action_info.get("z_inner_gate", np.nan))
        if learned_gates_enabled:
            z_root = float(np.clip(z_root_gate, 0.0, 1.0))
            z_inner = float(np.clip(z_inner_gate, 0.0, 1.0))
            if is_derisk:
                root_min = float(learned_gates.get("root_derisk_min", 0.05))
                root_max = float(learned_gates.get("root_derisk_max", 0.50))
            else:
                root_min = float(learned_gates.get("root_rerisk_min", 0.02))
                root_max = float(learned_gates.get("root_rerisk_max", 0.20))
            inner_min = float(learned_gates.get("inner_min", 0.05))
            inner_max = float(learned_gates.get("inner_max", 0.60))
            k_root_base = root_min + (root_max - root_min) * z_root
            k_inner_base = inner_min + (inner_max - inner_min) * z_inner
        else:
            k_derisk = float(execution_cfg.get("k_derisk", 0.25))
            k_rerisk = float(execution_cfg.get("k_rerisk", 0.08))
            k_inner_base = float(execution_cfg.get("k_inner", 0.30))
            k_root_base = k_derisk if is_derisk else k_rerisk
            root_min = np.nan
            root_max = np.nan
            inner_min = np.nan
            inner_max = np.nan

        gap_root = abs(delta_q)
        gap_inner = float(np.sum(np.abs(u_target - u_prev)))

        deadzone_cfg = execution_cfg.get("deadzone", {})
        deadzone_enabled = bool(deadzone_cfg.get("enabled", False))
        if deadzone_enabled:
            eps_root = float(deadzone_cfg.get("eps_root", 0.010))
            tau_root = float(deadzone_cfg.get("tau_root", 0.010))
            eps_inner = float(deadzone_cfg.get("eps_inner", 0.030))
            tau_inner = float(deadzone_cfg.get("tau_inner", 0.030))
            root_scale = deadzone_scale(gap_root, eps_root, tau_root)
            inner_scale = deadzone_scale(gap_inner, eps_inner, tau_inner)
        else:
            eps_root = 0.0
            tau_root = 0.0
            eps_inner = 0.0
            tau_inner = 0.0
            root_scale = 1.0
            inner_scale = 1.0

        k_root_eff = float(np.clip(k_root_base * root_scale, 0.0, 1.0))
        k_inner_eff = float(np.clip(k_inner_base * inner_scale, 0.0, 1.0))

        q_exec_pre_cap = q_prev + k_root_eff * delta_q
        q_min = float(execution_cfg.get("q_min", 0.0))
        q_max = float(execution_cfg.get("q_max", 1.0))
        q_exec_pre_cap = float(np.clip(q_exec_pre_cap, q_min, q_max))

        u_exec_raw = (1.0 - k_inner_eff) * u_prev + k_inner_eff * u_target
        u_exec = normalize_stock_simplex(u_exec_raw)

        raw = np.zeros(self.asset_dim, dtype=np.float64)
        raw[: self.stock_dim] = q_exec_pre_cap * u_exec
        raw[self.cash_index] = 1.0 - q_exec_pre_cap
        raw = normalize_simplex(raw)

        pre_cap_delta = raw - self.previous_weights
        pre_cap_delta_l1 = float(np.sum(np.abs(pre_cap_delta)))
        cap_scale = 1.0
        if pre_cap_delta_l1 > self.turnover_cap > 0:
            cap_scale = self.turnover_cap / pre_cap_delta_l1
            raw = self.previous_weights + pre_cap_delta * cap_scale

        projected = project_to_simplex(raw)
        projection_residual = float(np.linalg.norm(projected - raw, ord=1))

        q_exec = float(np.clip(np.sum(projected[: self.stock_dim]), 0.0, 1.0))
        if q_exec > EPS:
            u_projected = normalize_stock_simplex(projected[: self.stock_dim] / q_exec)
        else:
            u_projected = np.array(u_exec, copy=True)
        root_turnover = abs(q_exec - q_prev)
        inner_turnover = q_exec * float(np.sum(np.abs(u_projected - u_prev)))
        regularization = execution_cfg.get("regularization", {})
        root_prior = float(regularization.get("k_root_prior", 0.15))
        inner_prior = float(regularization.get("k_inner_prior", 0.30))
        gate_prior_penalty = float(regularization.get("lambda_gate_prior", 0.0)) * (
            (k_root_eff - root_prior) ** 2 + (k_inner_eff - inner_prior) ** 2
        )
        gate_smooth_penalty = float(regularization.get("lambda_gate_smooth", 0.0)) * (
            (k_root_eff - self.last_k_root_eff) ** 2 + (k_inner_eff - self.last_k_inner_eff) ** 2
        )
        target_turnover_l1 = float(np.sum(np.abs(target_weights - self.previous_weights)))
        raw_churn_penalty = float(regularization.get("lambda_raw_churn", 0.0)) * target_turnover_l1
        execution_penalty = gate_prior_penalty + gate_smooth_penalty + raw_churn_penalty

        self.previous_error = target_weights - self.previous_weights
        self.last_projection_residual = projection_residual
        self.last_k_root_eff = k_root_eff
        self.last_k_inner_eff = k_inner_eff

        terms = {
            "controller_p_l1": 0.0,
            "controller_i_l1": 0.0,
            "controller_d_l1": 0.0,
            "controller_delta_l1_before_cap": pre_cap_delta_l1,
            "projection_residual_l1": projection_residual,
            "root_exec_enabled": 1.0,
            "q_prev_exec": q_prev,
            "q_exec": q_exec,
            "cash_exec": 1.0 - q_exec,
            "delta_q_target": delta_q,
            "is_derisk": float(is_derisk),
            "gap_root": gap_root,
            "gap_inner": gap_inner,
            "eps_root": eps_root,
            "eps_inner": eps_inner,
            "tau_root": tau_root,
            "tau_inner": tau_inner,
            "deadzone_scale_root": root_scale,
            "deadzone_scale_inner": inner_scale,
            "k_root_base": k_root_base,
            "k_root_eff": k_root_eff,
            "k_inner_base": k_inner_base,
            "k_inner_eff": k_inner_eff,
            "learned_gates_enabled": float(learned_gates_enabled),
            "z_root_gate": z_root_gate,
            "z_inner_gate": z_inner_gate,
            "k_root_min_bound": float(root_min),
            "k_root_max_bound": float(root_max),
            "k_inner_min_bound": float(inner_min),
            "k_inner_max_bound": float(inner_max),
            "root_turnover": root_turnover,
            "inner_turnover": inner_turnover,
            "turnover_cap_scale": cap_scale,
            "target_turnover_l1": target_turnover_l1,
            "suppressed_root_speed": k_root_base - k_root_eff,
            "suppressed_inner_speed": k_inner_base - k_inner_eff,
            "gate_prior_penalty": gate_prior_penalty,
            "gate_smooth_penalty": gate_smooth_penalty,
            "raw_churn_penalty": raw_churn_penalty,
            "execution_penalty": execution_penalty,
        }
        return projected, terms

    def _compute_reward(
        self,
        *,
        net_return: float,
        turnover: float,
        drawdown_increment: float,
        concentration: float,
        action_change: float,
        extra_penalty: float = 0.0,
    ) -> float:
        cfg = self.reward_config
        unscaled = (
            float(cfg.get("return_weight", 1.0)) * net_return
            - float(cfg.get("turnover_penalty", 0.0)) * turnover
            - float(cfg.get("drawdown_penalty", 0.0)) * drawdown_increment
            - float(cfg.get("concentration_penalty", 0.0)) * concentration
            - float(cfg.get("action_change_penalty", 0.0)) * action_change
            - float(extra_penalty)
        )
        return self.reward_scale * unscaled

    def _cash_prior_terms(self, cash_weight: float) -> tuple[float, float, float, float]:
        prior = self.root_split_config.get("cash_prior", {})
        if not prior or not bool(prior.get("enabled", False)):
            return float("nan"), float("nan"), 0.0, 0.0

        score = float(prior.get("intercept", 0.0))
        feature_weights = prior.get("feature_weights", {})
        for feature, weight in feature_weights.items():
            if feature == "drawdown_severity":
                value = -float(self.previous_drawdown)
            elif feature == "last_turnover":
                value = float(self.last_turnover)
            else:
                idx = self.feature_index.get(str(feature))
                if idx is None:
                    value = self._feature_value(str(feature), default=0.0)
                else:
                    value = float(self.panel.features[self.day, 0, idx])
            score += float(weight) * value

        risk_score = float(1.0 / (1.0 + np.exp(-np.clip(score, -30.0, 30.0))))
        cash_low = float(prior.get("cash_low", 0.03))
        cash_high = float(prior.get("cash_high", 0.35))
        cash_allowed = cash_low + (cash_high - cash_low) * risk_score
        excess_cash = max(float(cash_weight) - cash_allowed, 0.0)
        cash_penalty = float(prior.get("lambda_cash", 0.0)) * (excess_cash**2)
        return risk_score, cash_allowed, excess_cash, cash_penalty

    def _weights_after_market_move(self, executed_weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
        stock_values = executed_weights[: self.stock_dim] * (1.0 + asset_returns)
        cash_value = executed_weights[self.cash_index]
        total = float(stock_values.sum() + cash_value)
        if total <= EPS:
            out = np.zeros(self.asset_dim, dtype=np.float64)
            out[self.cash_index] = 1.0
            return out
        next_weights = np.empty(self.asset_dim, dtype=np.float64)
        next_weights[: self.stock_dim] = stock_values / total
        next_weights[self.cash_index] = cash_value / total
        return normalize_simplex(next_weights)

    def _info_base(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "portfolio_value": self.portfolio_value,
            "cash_weight": float(self.previous_weights[self.cash_index]),
            "gross_exposure": float(np.sum(np.abs(self.previous_weights[: self.stock_dim]))),
            "hhi": float(np.sum(self.previous_weights**2)),
            "projection_residual_l1": self.last_projection_residual,
            "internal_trading_days_processed": self.internal_trading_days_processed,
        }


def make_env_from_config(
    panel: WeightPanel,
    config: dict[str, Any],
    variant: dict[str, Any],
) -> Stage01WeightPortfolioEnv:
    env_cfg = config["environment"]
    execution_cfg = env_cfg.get("execution", {})
    obs_cfg = env_cfg.get("observation", {})
    return Stage01WeightPortfolioEnv(
        panel,
        sector_map_name=config.get("universe", {}).get("sector_map", "dow30_static"),
        action_transform=variant["action_transform"],
        controller=variant.get("controller", {}),
        root_split_config=variant.get("root_split", {}),
        initial_amount=float(env_cfg.get("initial_amount", 1_000_000.0)),
        transaction_cost_pct=float(env_cfg.get("transaction_cost_pct", 0.001)),
        reward_config=env_cfg.get("reward", {}),
        reward_scale=float(env_cfg.get("reward_scale", 100.0)),
        turnover_cap=float(execution_cfg.get("turnover_cap", 0.35)),
        integral_clip=float(execution_cfg.get("integral_clip", 0.50)),
        derivative_clip=float(execution_cfg.get("derivative_clip", 0.50)),
        include_features=bool(obs_cfg.get("include_features", True)),
        include_previous_weights=bool(obs_cfg.get("include_previous_weights", True)),
        include_portfolio_state=bool(obs_cfg.get("include_portfolio_state", True)),
    )
