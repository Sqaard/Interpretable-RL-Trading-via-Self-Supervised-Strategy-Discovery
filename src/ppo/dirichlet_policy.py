"""Dirichlet actor-critic policy for simplex portfolio weights."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from torch.nn import functional as F


class DirichletDistribution(Distribution):
    """Torch Dirichlet distribution wrapper compatible with SB3 policies."""

    def __init__(self, action_dim: int, alpha_min: float = 1.0, alpha_max: float = 80.0):
        super().__init__()
        self.action_dim = int(action_dim)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.distribution: th.distributions.Dirichlet | None = None
        self.alpha: th.Tensor | None = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(self, raw_alpha: th.Tensor) -> "DirichletDistribution":
        alpha = F.softplus(raw_alpha) + self.alpha_min
        alpha = th.clamp(alpha, min=1e-4, max=self.alpha_max)
        self.alpha = alpha
        self.distribution = th.distributions.Dirichlet(alpha)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        actions = th.clamp(actions, min=1e-8)
        actions = actions / th.clamp(actions.sum(dim=1, keepdim=True), min=1e-8)
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        if self.alpha is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        # The Dirichlet mode is undefined at boundaries when any alpha <= 1.
        # The mean is deterministic, stable, and valid for evaluation.
        return self.alpha / th.clamp(self.alpha.sum(dim=1, keepdim=True), min=1e-8)

    def actions_from_params(self, raw_alpha: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(raw_alpha)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, raw_alpha: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(raw_alpha)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class DirichletActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy whose action distribution is Dirichlet on a simplex."""

    def __init__(
        self,
        *args: Any,
        alpha_min: float = 1.0,
        alpha_max: float = 80.0,
        **kwargs: Any,
    ):
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("DirichletActorCriticPolicy requires a 1-D Box action space.")

        action_dim = int(np.prod(self.action_space.shape))
        self.action_dist = DirichletDistribution(
            action_dim,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
        )
        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        raw_alpha = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(raw_alpha)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(alpha_min=self.alpha_min, alpha_max=self.alpha_max)
        return data


class HierarchicalDirichletDistribution(Distribution):
    """Dirichlet-tree distribution over final portfolio leaf weights."""

    def __init__(
        self,
        action_dim: int,
        group_indices: list[list[int]],
        alpha_min: float = 1.0,
        alpha_max: float = 80.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.group_indices = [list(group) for group in group_indices]
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.root_dim = len(self.group_indices)
        self.inner_group_indices = [i for i, group in enumerate(self.group_indices) if len(group) > 1]
        self.param_dim = self.root_dim + sum(len(self.group_indices[i]) for i in self.inner_group_indices)
        self.root_dist: th.distributions.Dirichlet | None = None
        self.inner_dists: list[tuple[int, th.distributions.Dirichlet]] = []
        self.root_alpha: th.Tensor | None = None
        self.inner_alphas: list[tuple[int, th.Tensor]] = []

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, self.param_dim)

    def proba_distribution(self, raw_params: th.Tensor) -> "HierarchicalDirichletDistribution":
        root_raw = raw_params[:, : self.root_dim]
        root_alpha = th.clamp(F.softplus(root_raw) + self.alpha_min, min=1e-4, max=self.alpha_max)
        self.root_alpha = root_alpha
        self.root_dist = th.distributions.Dirichlet(root_alpha)

        self.inner_dists = []
        self.inner_alphas = []
        offset = self.root_dim
        for group_idx in self.inner_group_indices:
            group_size = len(self.group_indices[group_idx])
            inner_raw = raw_params[:, offset : offset + group_size]
            offset += group_size
            inner_alpha = th.clamp(F.softplus(inner_raw) + self.alpha_min, min=1e-4, max=self.alpha_max)
            self.inner_alphas.append((group_idx, inner_alpha))
            self.inner_dists.append((group_idx, th.distributions.Dirichlet(inner_alpha)))
        return self

    def _compose_leaf_weights(self, root_weights: th.Tensor, inner_weights: list[tuple[int, th.Tensor]]) -> th.Tensor:
        batch_size = root_weights.shape[0]
        actions = th.zeros((batch_size, self.action_dim), dtype=root_weights.dtype, device=root_weights.device)
        inner_by_group = {group_idx: weights for group_idx, weights in inner_weights}
        for group_idx, leaf_indices in enumerate(self.group_indices):
            budget = root_weights[:, group_idx : group_idx + 1]
            idx = th.as_tensor(leaf_indices, dtype=th.long, device=root_weights.device)
            if len(leaf_indices) == 1:
                actions[:, idx] = budget
            else:
                actions[:, idx] = budget * inner_by_group[group_idx]
        return actions

    def sample(self) -> th.Tensor:
        if self.root_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        root_sample = self.root_dist.sample()
        inner_samples = [(group_idx, dist.sample()) for group_idx, dist in self.inner_dists]
        return self._compose_leaf_weights(root_sample, inner_samples)

    def mode(self) -> th.Tensor:
        if self.root_alpha is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        root_mean = self.root_alpha / th.clamp(self.root_alpha.sum(dim=1, keepdim=True), min=1e-8)
        inner_means = [
            (group_idx, alpha / th.clamp(alpha.sum(dim=1, keepdim=True), min=1e-8))
            for group_idx, alpha in self.inner_alphas
        ]
        return self._compose_leaf_weights(root_mean, inner_means)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.root_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        actions = th.clamp(actions, min=1e-8)
        actions = actions / th.clamp(actions.sum(dim=1, keepdim=True), min=1e-8)

        root_parts = []
        for leaf_indices in self.group_indices:
            idx = th.as_tensor(leaf_indices, dtype=th.long, device=actions.device)
            root_parts.append(actions[:, idx].sum(dim=1, keepdim=True))
        root_weights = th.cat(root_parts, dim=1)
        root_weights = th.clamp(root_weights, min=1e-8)
        root_weights = root_weights / th.clamp(root_weights.sum(dim=1, keepdim=True), min=1e-8)

        log_prob = self.root_dist.log_prob(root_weights)
        jacobian_log = th.zeros_like(log_prob)

        for group_idx, inner_dist in self.inner_dists:
            leaf_indices = self.group_indices[group_idx]
            idx = th.as_tensor(leaf_indices, dtype=th.long, device=actions.device)
            budget = th.clamp(root_weights[:, group_idx : group_idx + 1], min=1e-8)
            inner_weights = actions[:, idx] / budget
            inner_weights = th.clamp(inner_weights, min=1e-8)
            inner_weights = inner_weights / th.clamp(inner_weights.sum(dim=1, keepdim=True), min=1e-8)
            log_prob = log_prob + inner_dist.log_prob(inner_weights)
            jacobian_log = jacobian_log + (len(leaf_indices) - 1) * th.log(budget.squeeze(1))

        # Dirichlet-tree density over leaf weights: p(y) / |J|, where
        # |J| = product_g sector_weight_g ** (n_g - 1).
        return log_prob - jacobian_log

    def entropy(self) -> None:
        # Closed-form entropy for the transformed leaf distribution is not
        # implemented here. SB3 falls back to a log-prob approximation.
        return None

    def actions_from_params(self, raw_params: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(raw_params)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, raw_params: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(raw_params)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class HierarchicalDirichletActorCriticPolicy(DirichletActorCriticPolicy):
    """Actor-critic policy with a Dirichlet-tree distribution over leaf weights."""

    def __init__(
        self,
        *args: Any,
        group_indices: list[list[int]],
        alpha_min: float = 1.0,
        alpha_max: float = 80.0,
        **kwargs: Any,
    ):
        self.group_indices = [list(group) for group in group_indices]
        super().__init__(*args, alpha_min=alpha_min, alpha_max=alpha_max, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("HierarchicalDirichletActorCriticPolicy requires a 1-D Box action space.")

        action_dim = int(np.prod(self.action_space.shape))
        flat_indices = sorted(idx for group in self.group_indices for idx in group)
        if flat_indices != list(range(action_dim)):
            raise ValueError(
                "group_indices must partition all action dimensions exactly once. "
                f"got={flat_indices}, expected={list(range(action_dim))}"
            )

        self.action_dist = HierarchicalDirichletDistribution(
            action_dim,
            group_indices=self.group_indices,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
        )
        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(group_indices=self.group_indices)
        return data


class RootSplitBetaDirichletDistribution(Distribution):
    """Beta root split plus Dirichlet risky allocation.

    The sampled action is a factor vector, not final portfolio weights:

    action[:, 0]  = q, invested fraction
    action[:, 1:] = u, conditional risky allocation over stocks

    The environment maps this deterministically to final target weights:

    cash = 1 - q
    stock_i = q * u_i
    """

    def __init__(
        self,
        stock_dim: int,
        *,
        q_min: float = 0.00,
        q_max: float = 0.995,
        alpha_floor: float = 0.05,
        kappa_min: float = 2.0,
        kappa_max: float = 80.0,
        risky_alpha_max: float = 100.0,
    ):
        super().__init__()
        self.stock_dim = int(stock_dim)
        self.action_dim = self.stock_dim + 1
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        if not 0.0 <= self.q_min < self.q_max <= 1.0:
            raise ValueError(f"Invalid q bounds: q_min={q_min}, q_max={q_max}")
        self.alpha_floor = float(alpha_floor)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.risky_alpha_max = float(risky_alpha_max)
        self.root_dist: th.distributions.Beta | None = None
        self.risky_dist: th.distributions.Dirichlet | None = None
        self.root_alpha: th.Tensor | None = None
        self.root_beta: th.Tensor | None = None
        self.risky_alpha: th.Tensor | None = None
        self.q_mean_unit: th.Tensor | None = None

    @property
    def q_range(self) -> float:
        return self.q_max - self.q_min

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, 2 + self.stock_dim)

    def proba_distribution(self, raw_params: th.Tensor) -> "RootSplitBetaDirichletDistribution":
        root_mean_logit = raw_params[:, 0:1]
        root_kappa_raw = raw_params[:, 1:2]
        risky_raw = raw_params[:, 2:]

        q_mean_unit = th.sigmoid(root_mean_logit)
        kappa = self.kappa_min + F.softplus(root_kappa_raw)
        kappa = th.clamp(kappa, min=self.kappa_min, max=self.kappa_max)

        root_alpha = self.alpha_floor + q_mean_unit * kappa
        root_beta = self.alpha_floor + (1.0 - q_mean_unit) * kappa
        root_alpha = th.clamp(root_alpha, min=1e-4, max=self.kappa_max + self.alpha_floor)
        root_beta = th.clamp(root_beta, min=1e-4, max=self.kappa_max + self.alpha_floor)

        risky_alpha = F.softplus(risky_raw) + self.alpha_floor
        risky_alpha = th.clamp(risky_alpha, min=1e-4, max=self.risky_alpha_max)

        self.q_mean_unit = q_mean_unit
        self.root_alpha = root_alpha
        self.root_beta = root_beta
        self.risky_alpha = risky_alpha
        self.root_dist = th.distributions.Beta(root_alpha.squeeze(-1), root_beta.squeeze(-1))
        self.risky_dist = th.distributions.Dirichlet(risky_alpha)
        return self

    def _compose_action(self, q_unit: th.Tensor, risky_weights: th.Tensor) -> th.Tensor:
        q = self.q_min + self.q_range * q_unit
        return th.cat([q.unsqueeze(-1), risky_weights], dim=1)

    def sample(self) -> th.Tensor:
        if self.root_dist is None or self.risky_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit = self.root_dist.sample()
        risky = self.risky_dist.sample()
        return self._compose_action(q_unit, risky)

    def mode(self) -> th.Tensor:
        if self.root_alpha is None or self.root_beta is None or self.risky_alpha is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit = (self.root_alpha / th.clamp(self.root_alpha + self.root_beta, min=1e-8)).squeeze(-1)
        risky = self.risky_alpha / th.clamp(self.risky_alpha.sum(dim=1, keepdim=True), min=1e-8)
        return self._compose_action(q_unit, risky)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.root_dist is None or self.risky_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q = th.clamp(actions[:, 0], min=self.q_min + 1e-6, max=self.q_max - 1e-6)
        q_unit = (q - self.q_min) / self.q_range
        q_unit = th.clamp(q_unit, min=1e-6, max=1.0 - 1e-6)
        risky = th.clamp(actions[:, 1:], min=1e-8)
        risky = risky / th.clamp(risky.sum(dim=1, keepdim=True), min=1e-8)

        root_log_prob = self.root_dist.log_prob(q_unit) - np.log(self.q_range)
        risky_log_prob = self.risky_dist.log_prob(risky)
        return root_log_prob + risky_log_prob

    def entropy(self) -> th.Tensor:
        if self.root_dist is None or self.risky_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        return self.root_dist.entropy() + self.risky_dist.entropy()

    def actions_from_params(self, raw_params: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(raw_params)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, raw_params: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(raw_params)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class RootSplitBetaDirichletKpDistribution(RootSplitBetaDirichletDistribution):
    """Root split plus risky allocation plus stochastic bounded Kp gate factors.

    The sampled action is:

    action[:, 0]                 = q, invested fraction
    action[:, 1:1 + stock_dim]   = u, conditional risky allocation
    action[:, 1 + stock_dim]     = z_root_gate in [0, 1]
    action[:, 2 + stock_dim]     = z_inner_gate in [0, 1]
    """

    def __init__(
        self,
        stock_dim: int,
        *,
        q_min: float = 0.00,
        q_max: float = 0.995,
        alpha_floor: float = 0.05,
        kappa_min: float = 2.0,
        kappa_max: float = 80.0,
        risky_alpha_max: float = 100.0,
        gate_kappa_min: float = 8.0,
        gate_kappa_max: float = 80.0,
    ):
        super().__init__(
            stock_dim,
            q_min=q_min,
            q_max=q_max,
            alpha_floor=alpha_floor,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            risky_alpha_max=risky_alpha_max,
        )
        self.action_dim = self.stock_dim + 3
        self.gate_kappa_min = float(gate_kappa_min)
        self.gate_kappa_max = float(gate_kappa_max)
        self.root_gate_dist: th.distributions.Beta | None = None
        self.inner_gate_dist: th.distributions.Beta | None = None
        self.root_gate_alpha: th.Tensor | None = None
        self.root_gate_beta: th.Tensor | None = None
        self.inner_gate_alpha: th.Tensor | None = None
        self.inner_gate_beta: th.Tensor | None = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, 2 + self.stock_dim + 4)

    def _gate_alpha_beta(self, mean_logit: th.Tensor, kappa_raw: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        mean = th.sigmoid(mean_logit)
        kappa = self.gate_kappa_min + F.softplus(kappa_raw)
        kappa = th.clamp(kappa, min=self.gate_kappa_min, max=self.gate_kappa_max)
        alpha = th.clamp(self.alpha_floor + mean * kappa, min=1e-4, max=self.gate_kappa_max + self.alpha_floor)
        beta = th.clamp(self.alpha_floor + (1.0 - mean) * kappa, min=1e-4, max=self.gate_kappa_max + self.alpha_floor)
        return alpha, beta

    def proba_distribution(self, raw_params: th.Tensor) -> "RootSplitBetaDirichletKpDistribution":
        super().proba_distribution(raw_params[:, : 2 + self.stock_dim])
        gate_raw = raw_params[:, 2 + self.stock_dim :]
        root_gate_alpha, root_gate_beta = self._gate_alpha_beta(gate_raw[:, 0:1], gate_raw[:, 1:2])
        inner_gate_alpha, inner_gate_beta = self._gate_alpha_beta(gate_raw[:, 2:3], gate_raw[:, 3:4])

        self.root_gate_alpha = root_gate_alpha
        self.root_gate_beta = root_gate_beta
        self.inner_gate_alpha = inner_gate_alpha
        self.inner_gate_beta = inner_gate_beta
        self.root_gate_dist = th.distributions.Beta(root_gate_alpha.squeeze(-1), root_gate_beta.squeeze(-1))
        self.inner_gate_dist = th.distributions.Beta(inner_gate_alpha.squeeze(-1), inner_gate_beta.squeeze(-1))
        return self

    def sample(self) -> th.Tensor:
        base = super().sample()
        if self.root_gate_dist is None or self.inner_gate_dist is None:
            raise RuntimeError("Gate distribution parameters are not initialized.")
        root_gate = self.root_gate_dist.sample().unsqueeze(-1)
        inner_gate = self.inner_gate_dist.sample().unsqueeze(-1)
        return th.cat([base, root_gate, inner_gate], dim=1)

    def mode(self) -> th.Tensor:
        base = super().mode()
        if (
            self.root_gate_alpha is None
            or self.root_gate_beta is None
            or self.inner_gate_alpha is None
            or self.inner_gate_beta is None
        ):
            raise RuntimeError("Gate distribution parameters are not initialized.")
        root_gate = self.root_gate_alpha / th.clamp(self.root_gate_alpha + self.root_gate_beta, min=1e-8)
        inner_gate = self.inner_gate_alpha / th.clamp(self.inner_gate_alpha + self.inner_gate_beta, min=1e-8)
        return th.cat([base, root_gate, inner_gate], dim=1)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.root_gate_dist is None or self.inner_gate_dist is None:
            raise RuntimeError("Gate distribution parameters are not initialized.")
        base_log_prob = super().log_prob(actions[:, : 1 + self.stock_dim])
        root_gate = th.clamp(actions[:, 1 + self.stock_dim], min=1e-6, max=1.0 - 1e-6)
        inner_gate = th.clamp(actions[:, 2 + self.stock_dim], min=1e-6, max=1.0 - 1e-6)
        return (
            base_log_prob
            + self.root_gate_dist.log_prob(root_gate)
            + self.inner_gate_dist.log_prob(inner_gate)
        )

    def entropy(self) -> th.Tensor:
        if self.root_gate_dist is None or self.inner_gate_dist is None:
            raise RuntimeError("Gate distribution parameters are not initialized.")
        return super().entropy() + self.root_gate_dist.entropy() + self.inner_gate_dist.entropy()


class RiskCashSectorDirichletTreeDistribution(Distribution):
    """Bounded Beta cash/invested root plus sector and within-sector Dirichlet factors."""

    def __init__(
        self,
        group_indices: list[list[int]],
        *,
        q_min: float = 0.00,
        q_max: float = 0.995,
        alpha_floor: float = 0.05,
        kappa_min: float = 2.0,
        kappa_max: float = 80.0,
        group_alpha_max: float = 100.0,
        leaf_alpha_max: float = 120.0,
    ):
        super().__init__()
        self.group_indices = [list(group) for group in group_indices]
        self.stock_dim = sum(len(group) for group in self.group_indices)
        self.group_dim = len(self.group_indices)
        self.inner_group_indices = [i for i, group in enumerate(self.group_indices) if len(group) > 1]
        self.action_dim = 1 + self.group_dim + sum(len(self.group_indices[i]) for i in self.inner_group_indices)
        self.param_dim = 2 + self.group_dim + sum(len(self.group_indices[i]) for i in self.inner_group_indices)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        if not 0.0 <= self.q_min < self.q_max <= 1.0:
            raise ValueError(f"Invalid q bounds: q_min={q_min}, q_max={q_max}")
        self.alpha_floor = float(alpha_floor)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.group_alpha_max = float(group_alpha_max)
        self.leaf_alpha_max = float(leaf_alpha_max)
        self.root_dist: th.distributions.Beta | None = None
        self.group_dist: th.distributions.Dirichlet | None = None
        self.inner_dists: list[tuple[int, th.distributions.Dirichlet]] = []
        self.root_alpha: th.Tensor | None = None
        self.root_beta: th.Tensor | None = None
        self.group_alpha: th.Tensor | None = None
        self.inner_alphas: list[tuple[int, th.Tensor]] = []

    @property
    def q_range(self) -> float:
        return self.q_max - self.q_min

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, self.param_dim)

    def proba_distribution(self, raw_params: th.Tensor) -> "RiskCashSectorDirichletTreeDistribution":
        root_mean_logit = raw_params[:, 0:1]
        root_kappa_raw = raw_params[:, 1:2]
        q_mean_unit = th.sigmoid(root_mean_logit)
        kappa = self.kappa_min + F.softplus(root_kappa_raw)
        kappa = th.clamp(kappa, min=self.kappa_min, max=self.kappa_max)
        root_alpha = th.clamp(self.alpha_floor + q_mean_unit * kappa, min=1e-4, max=self.kappa_max + self.alpha_floor)
        root_beta = th.clamp(
            self.alpha_floor + (1.0 - q_mean_unit) * kappa,
            min=1e-4,
            max=self.kappa_max + self.alpha_floor,
        )

        offset = 2
        group_raw = raw_params[:, offset : offset + self.group_dim]
        offset += self.group_dim
        group_alpha = th.clamp(
            F.softplus(group_raw) + self.alpha_floor,
            min=1e-4,
            max=self.group_alpha_max,
        )

        self.inner_dists = []
        self.inner_alphas = []
        for group_idx in self.inner_group_indices:
            group_size = len(self.group_indices[group_idx])
            inner_raw = raw_params[:, offset : offset + group_size]
            offset += group_size
            inner_alpha = th.clamp(
                F.softplus(inner_raw) + self.alpha_floor,
                min=1e-4,
                max=self.leaf_alpha_max,
            )
            self.inner_alphas.append((group_idx, inner_alpha))
            self.inner_dists.append((group_idx, th.distributions.Dirichlet(inner_alpha)))

        self.root_alpha = root_alpha
        self.root_beta = root_beta
        self.group_alpha = group_alpha
        self.root_dist = th.distributions.Beta(root_alpha.squeeze(-1), root_beta.squeeze(-1))
        self.group_dist = th.distributions.Dirichlet(group_alpha)
        return self

    def _compose_action(self, q_unit: th.Tensor, group: th.Tensor, inners: list[tuple[int, th.Tensor]]) -> th.Tensor:
        q = self.q_min + self.q_range * q_unit
        parts = [q.unsqueeze(-1), group]
        inner_by_group = {group_idx: weights for group_idx, weights in inners}
        for group_idx in self.inner_group_indices:
            parts.append(inner_by_group[group_idx])
        return th.cat(parts, dim=1)

    def sample(self) -> th.Tensor:
        if self.root_dist is None or self.group_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit = self.root_dist.sample()
        group = self.group_dist.sample()
        inners = [(group_idx, dist.sample()) for group_idx, dist in self.inner_dists]
        return self._compose_action(q_unit, group, inners)

    def mode(self) -> th.Tensor:
        if self.root_alpha is None or self.root_beta is None or self.group_alpha is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit = (self.root_alpha / th.clamp(self.root_alpha + self.root_beta, min=1e-8)).squeeze(-1)
        group = self.group_alpha / th.clamp(self.group_alpha.sum(dim=1, keepdim=True), min=1e-8)
        inners = [
            (group_idx, alpha / th.clamp(alpha.sum(dim=1, keepdim=True), min=1e-8))
            for group_idx, alpha in self.inner_alphas
        ]
        return self._compose_action(q_unit, group, inners)

    def _split_actions(
        self, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, list[tuple[int, th.Tensor]]]:
        q = th.clamp(actions[:, 0], min=self.q_min + 1e-6, max=self.q_max - 1e-6)
        q_unit = th.clamp((q - self.q_min) / self.q_range, min=1e-6, max=1.0 - 1e-6)
        offset = 1
        group = th.clamp(actions[:, offset : offset + self.group_dim], min=1e-8)
        group = group / th.clamp(group.sum(dim=1, keepdim=True), min=1e-8)
        offset += self.group_dim
        inners: list[tuple[int, th.Tensor]] = []
        for group_idx in self.inner_group_indices:
            group_size = len(self.group_indices[group_idx])
            inner = th.clamp(actions[:, offset : offset + group_size], min=1e-8)
            offset += group_size
            inner = inner / th.clamp(inner.sum(dim=1, keepdim=True), min=1e-8)
            inners.append((group_idx, inner))
        return q_unit, group, inners

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.root_dist is None or self.group_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit, group, inners = self._split_actions(actions)
        log_prob = self.root_dist.log_prob(q_unit) - np.log(self.q_range)
        log_prob = log_prob + self.group_dist.log_prob(group)
        inner_by_group = {group_idx: weights for group_idx, weights in inners}
        for group_idx, dist in self.inner_dists:
            log_prob = log_prob + dist.log_prob(inner_by_group[group_idx])
        return log_prob

    def entropy(self) -> th.Tensor:
        if self.root_dist is None or self.group_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        entropy = self.root_dist.entropy() + self.group_dist.entropy()
        for _, dist in self.inner_dists:
            entropy = entropy + dist.entropy()
        return entropy

    def actions_from_params(self, raw_params: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(raw_params)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, raw_params: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(raw_params)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class RiskCashGroupLogisticNormalTreeDistribution(RiskCashSectorDirichletTreeDistribution):
    """Cash root Beta + logistic-normal group simplex + within-group Dirichlets.

    Group weights use additive-log-ratio coordinates with the last group as
    reference:

    z_j = log(g_j / g_K), j=1..K-1
    g = softmax([z, 0])
    """

    def __init__(
        self,
        group_indices: list[list[int]],
        *,
        q_min: float = 0.00,
        q_max: float = 0.995,
        alpha_floor: float = 0.05,
        kappa_min: float = 2.0,
        kappa_max: float = 80.0,
        leaf_alpha_max: float = 120.0,
        group_log_std_min: float = -2.5,
        group_log_std_max: float = 0.3,
    ):
        super().__init__(
            group_indices,
            q_min=q_min,
            q_max=q_max,
            alpha_floor=alpha_floor,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            group_alpha_max=1.0,
            leaf_alpha_max=leaf_alpha_max,
        )
        self.group_latent_dim = self.group_dim - 1
        if self.group_latent_dim < 1:
            raise ValueError("Logistic-normal group layer requires at least two groups.")
        self.param_dim = 2 + 2 * self.group_latent_dim + sum(
            len(self.group_indices[i]) for i in self.inner_group_indices
        )
        self.group_log_std_min = float(group_log_std_min)
        self.group_log_std_max = float(group_log_std_max)
        self.group_base_dist: th.distributions.Independent | None = None
        self.group_mu: th.Tensor | None = None
        self.group_log_std: th.Tensor | None = None
        self.group_std: th.Tensor | None = None

    def _alr_inverse(self, z: th.Tensor) -> th.Tensor:
        zeros = th.zeros((z.shape[0], 1), dtype=z.dtype, device=z.device)
        return F.softmax(th.cat([z, zeros], dim=1), dim=1)

    def _alr_forward(self, group: th.Tensor) -> th.Tensor:
        group = th.clamp(group, min=1e-8)
        group = group / th.clamp(group.sum(dim=1, keepdim=True), min=1e-8)
        return th.log(group[:, :-1]) - th.log(group[:, -1:])

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, self.param_dim)

    def proba_distribution(self, raw_params: th.Tensor) -> "RiskCashGroupLogisticNormalTreeDistribution":
        root_mean_logit = raw_params[:, 0:1]
        root_kappa_raw = raw_params[:, 1:2]
        q_mean_unit = th.sigmoid(root_mean_logit)
        kappa = self.kappa_min + F.softplus(root_kappa_raw)
        kappa = th.clamp(kappa, min=self.kappa_min, max=self.kappa_max)
        root_alpha = th.clamp(self.alpha_floor + q_mean_unit * kappa, min=1e-4, max=self.kappa_max + self.alpha_floor)
        root_beta = th.clamp(
            self.alpha_floor + (1.0 - q_mean_unit) * kappa,
            min=1e-4,
            max=self.kappa_max + self.alpha_floor,
        )

        offset = 2
        group_mu = raw_params[:, offset : offset + self.group_latent_dim]
        offset += self.group_latent_dim
        group_log_std = raw_params[:, offset : offset + self.group_latent_dim]
        offset += self.group_latent_dim
        group_log_std = th.clamp(group_log_std, min=self.group_log_std_min, max=self.group_log_std_max)
        group_std = th.exp(group_log_std)

        self.inner_dists = []
        self.inner_alphas = []
        for group_idx in self.inner_group_indices:
            group_size = len(self.group_indices[group_idx])
            inner_raw = raw_params[:, offset : offset + group_size]
            offset += group_size
            inner_alpha = th.clamp(
                F.softplus(inner_raw) + self.alpha_floor,
                min=1e-4,
                max=self.leaf_alpha_max,
            )
            self.inner_alphas.append((group_idx, inner_alpha))
            self.inner_dists.append((group_idx, th.distributions.Dirichlet(inner_alpha)))

        self.root_alpha = root_alpha
        self.root_beta = root_beta
        self.group_mu = group_mu
        self.group_log_std = group_log_std
        self.group_std = group_std
        self.root_dist = th.distributions.Beta(root_alpha.squeeze(-1), root_beta.squeeze(-1))
        self.group_base_dist = th.distributions.Independent(th.distributions.Normal(group_mu, group_std), 1)
        return self

    def sample(self) -> th.Tensor:
        if self.root_dist is None or self.group_base_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit = self.root_dist.sample()
        group_z = self.group_base_dist.base_dist.sample()
        group = self._alr_inverse(group_z)
        inners = [(group_idx, dist.sample()) for group_idx, dist in self.inner_dists]
        return self._compose_action(q_unit, group, inners)

    def mode(self) -> th.Tensor:
        if self.root_alpha is None or self.root_beta is None or self.group_mu is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit = (self.root_alpha / th.clamp(self.root_alpha + self.root_beta, min=1e-8)).squeeze(-1)
        group = self._alr_inverse(self.group_mu)
        inners = [
            (group_idx, alpha / th.clamp(alpha.sum(dim=1, keepdim=True), min=1e-8))
            for group_idx, alpha in self.inner_alphas
        ]
        return self._compose_action(q_unit, group, inners)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.root_dist is None or self.group_base_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        q_unit, group, inners = self._split_actions(actions)
        group_z = self._alr_forward(group)
        log_jacobian = th.sum(th.log(th.clamp(group, min=1e-8)), dim=1)
        log_prob = self.root_dist.log_prob(q_unit) - np.log(self.q_range)
        log_prob = log_prob + self.group_base_dist.log_prob(group_z) - log_jacobian
        inner_by_group = {group_idx: weights for group_idx, weights in inners}
        for group_idx, dist in self.inner_dists:
            log_prob = log_prob + dist.log_prob(inner_by_group[group_idx])
        return log_prob

    def entropy(self) -> th.Tensor:
        if self.root_dist is None or self.group_base_dist is None:
            raise RuntimeError("Distribution parameters are not initialized.")
        entropy = self.root_dist.entropy() + self.group_base_dist.entropy()
        for _, dist in self.inner_dists:
            entropy = entropy + dist.entropy()
        return entropy


class RootSplitBetaDirichletActorCriticPolicy(DirichletActorCriticPolicy):
    """Shared-encoder actor-critic for root cash/invested split."""

    def __init__(
        self,
        *args: Any,
        stock_dim: int,
        q_min: float = 0.00,
        q_max: float = 0.995,
        alpha_floor: float = 0.05,
        kappa_min: float = 2.0,
        kappa_max: float = 80.0,
        risky_alpha_max: float = 100.0,
        **kwargs: Any,
    ):
        self.stock_dim = int(stock_dim)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.alpha_floor = float(alpha_floor)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.risky_alpha_max = float(risky_alpha_max)
        super().__init__(*args, alpha_min=alpha_floor, alpha_max=risky_alpha_max, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("RootSplitBetaDirichletActorCriticPolicy requires a 1-D Box action space.")
        action_dim = int(np.prod(self.action_space.shape))
        if action_dim != self.stock_dim + 1:
            raise ValueError(f"action_dim={action_dim} must equal stock_dim+1={self.stock_dim + 1}")

        self.action_dist = RootSplitBetaDirichletDistribution(
            self.stock_dim,
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            risky_alpha_max=self.risky_alpha_max,
        )
        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            stock_dim=self.stock_dim,
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            risky_alpha_max=self.risky_alpha_max,
        )
        return data


class RootSplitBetaDirichletKpActorCriticPolicy(RootSplitBetaDirichletActorCriticPolicy):
    """Root-split policy whose action includes stochastic Kp gate factors."""

    def __init__(
        self,
        *args: Any,
        gate_kappa_min: float = 8.0,
        gate_kappa_max: float = 80.0,
        **kwargs: Any,
    ):
        self.gate_kappa_min = float(gate_kappa_min)
        self.gate_kappa_max = float(gate_kappa_max)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("RootSplitBetaDirichletKpActorCriticPolicy requires a 1-D Box action space.")
        action_dim = int(np.prod(self.action_space.shape))
        if action_dim != self.stock_dim + 3:
            raise ValueError(f"action_dim={action_dim} must equal stock_dim+3={self.stock_dim + 3}")

        self.action_dist = RootSplitBetaDirichletKpDistribution(
            self.stock_dim,
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            risky_alpha_max=self.risky_alpha_max,
            gate_kappa_min=self.gate_kappa_min,
            gate_kappa_max=self.gate_kappa_max,
        )
        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(gate_kappa_min=self.gate_kappa_min, gate_kappa_max=self.gate_kappa_max)
        return data


class RiskCashSectorDirichletTreeActorCriticPolicy(DirichletActorCriticPolicy):
    """Policy over local cash/invested, sector, and within-sector tree factors."""

    def __init__(
        self,
        *args: Any,
        group_indices: list[list[int]],
        q_min: float = 0.00,
        q_max: float = 0.995,
        alpha_floor: float = 0.05,
        kappa_min: float = 2.0,
        kappa_max: float = 80.0,
        group_alpha_max: float = 100.0,
        leaf_alpha_max: float = 120.0,
        **kwargs: Any,
    ):
        self.group_indices = [list(group) for group in group_indices]
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.alpha_floor = float(alpha_floor)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.group_alpha_max = float(group_alpha_max)
        self.leaf_alpha_max = float(leaf_alpha_max)
        super().__init__(*args, alpha_min=alpha_floor, alpha_max=max(group_alpha_max, leaf_alpha_max), **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("RiskCashSectorDirichletTreeActorCriticPolicy requires a 1-D Box action space.")

        self.action_dist = RiskCashSectorDirichletTreeDistribution(
            self.group_indices,
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            group_alpha_max=self.group_alpha_max,
            leaf_alpha_max=self.leaf_alpha_max,
        )
        action_dim = int(np.prod(self.action_space.shape))
        if action_dim != self.action_dist.action_dim:
            raise ValueError(f"action_dim={action_dim} must equal tree factor dim={self.action_dist.action_dim}")

        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        raw_params = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(raw_params)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            group_indices=self.group_indices,
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            group_alpha_max=self.group_alpha_max,
            leaf_alpha_max=self.leaf_alpha_max,
        )
        return data


class RiskCashGroupLogisticNormalTreeActorCriticPolicy(RiskCashSectorDirichletTreeActorCriticPolicy):
    """Policy with logistic-normal group allocation and Dirichlet leaves."""

    def __init__(
        self,
        *args: Any,
        group_log_std_min: float = -2.5,
        group_log_std_max: float = 0.3,
        **kwargs: Any,
    ):
        self.group_log_std_min = float(group_log_std_min)
        self.group_log_std_max = float(group_log_std_max)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("RiskCashGroupLogisticNormalTreeActorCriticPolicy requires a 1-D Box action space.")

        self.action_dist = RiskCashGroupLogisticNormalTreeDistribution(
            self.group_indices,
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            leaf_alpha_max=self.leaf_alpha_max,
            group_log_std_min=self.group_log_std_min,
            group_log_std_max=self.group_log_std_max,
        )
        action_dim = int(np.prod(self.action_space.shape))
        if action_dim != self.action_dist.action_dim:
            raise ValueError(f"action_dim={action_dim} must equal tree factor dim={self.action_dist.action_dim}")

        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(group_log_std_min=self.group_log_std_min, group_log_std_max=self.group_log_std_max)
        return data


def _make_mlp(input_dim: int, layer_dims: list[int], activation_fn: type[nn.Module]) -> tuple[nn.Sequential, int]:
    modules: list[nn.Module] = []
    last_dim = int(input_dim)
    for dim in layer_dims:
        modules.append(nn.Linear(last_dim, int(dim)))
        modules.append(activation_fn())
        last_dim = int(dim)
    return nn.Sequential(*modules), last_dim


class RootSplitRoutedMlpExtractor(nn.Module):
    """Actor routing extractor for root-risk and risky-allocation branches."""

    def __init__(
        self,
        *,
        features_dim: int,
        stock_dim: int,
        feature_columns: list[str],
        root_feature_names: list[str],
        activation_fn: type[nn.Module],
        root_latent_dim: int = 32,
        risky_latent_dim: int = 32,
        hidden_dim: int = 128,
        vf_arch: list[int] | None = None,
    ):
        super().__init__()
        self.features_dim = int(features_dim)
        self.stock_dim = int(stock_dim)
        self.feature_columns = list(feature_columns)
        self.feature_dim = len(self.feature_columns)
        self.root_feature_names = [name for name in root_feature_names if name in self.feature_columns]
        self.root_feature_indices = [self.feature_columns.index(name) for name in self.root_feature_names]
        self.asset_flat_dim = self.stock_dim * self.feature_dim
        self.prev_weights_dim = self.stock_dim + 1
        self.portfolio_state_dim = 6
        expected_min_dim = self.asset_flat_dim + self.prev_weights_dim + self.portfolio_state_dim
        if self.features_dim < expected_min_dim:
            raise ValueError(f"features_dim={features_dim} is smaller than expected minimum {expected_min_dim}.")
        if not self.root_feature_indices:
            raise ValueError("Routed root split policy requires at least one root feature.")

        root_input_dim = len(self.root_feature_indices) + self.portfolio_state_dim
        risky_input_dim = self.asset_flat_dim + self.stock_dim + root_latent_dim

        self.root_net, root_out_dim = _make_mlp(root_input_dim, [hidden_dim, root_latent_dim], activation_fn)
        self.risky_net, risky_out_dim = _make_mlp(risky_input_dim, [hidden_dim, risky_latent_dim], activation_fn)
        self.value_net, vf_out_dim = _make_mlp(
            self.features_dim,
            vf_arch or [256, 128, 64],
            activation_fn,
        )
        self.latent_dim_pi = root_out_dim + risky_out_dim
        self.latent_dim_vf = vf_out_dim

    def _slices(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        asset_flat = features[:, : self.asset_flat_dim]
        prev_start = self.asset_flat_dim
        prev_end = prev_start + self.prev_weights_dim
        previous_weights = features[:, prev_start:prev_end]
        state_start = prev_end
        state_end = state_start + self.portfolio_state_dim
        portfolio_state = features[:, state_start:state_end]
        return asset_flat, previous_weights, portfolio_state

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        asset_flat, previous_weights, portfolio_state = self._slices(features)
        batch_size = features.shape[0]
        asset_matrix = asset_flat.reshape(batch_size, self.stock_dim, self.feature_dim)
        root_market = asset_matrix[:, 0, self.root_feature_indices]
        root_input = th.cat([root_market, portfolio_state], dim=1)
        z_root = self.root_net(root_input)

        risky_input = th.cat(
            [
                asset_flat,
                previous_weights[:, : self.stock_dim],
                z_root.detach(),
            ],
            dim=1,
        )
        z_risky = self.risky_net(risky_input)
        return th.cat([z_root, z_risky], dim=1)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class RootSplitRoutedActionNet(nn.Module):
    """Map routed 64-d actor latent to root/risky distribution parameters."""

    def __init__(self, root_dim: int, risky_dim: int, stock_dim: int):
        super().__init__()
        self.root_dim = int(root_dim)
        self.risky_dim = int(risky_dim)
        self.stock_dim = int(stock_dim)
        self.root_mean = nn.Linear(self.root_dim, 1)
        self.root_kappa = nn.Linear(self.root_dim, 1)
        self.risky_alpha = nn.Linear(self.risky_dim + self.root_dim, self.stock_dim)

    def forward(self, latent_pi: th.Tensor) -> th.Tensor:
        z_root = latent_pi[:, : self.root_dim]
        z_risky = latent_pi[:, self.root_dim : self.root_dim + self.risky_dim]
        risky_input = th.cat([z_risky, z_root.detach()], dim=1)
        return th.cat([self.root_mean(z_root), self.root_kappa(z_root), self.risky_alpha(risky_input)], dim=1)


class RoutedRootSplitBetaDirichletActorCriticPolicy(RootSplitBetaDirichletActorCriticPolicy):
    """Root-split policy with routed actor encoders.

    The root branch sees only selected risk/market features plus portfolio
    state. The risky branch sees the stock feature panel, previous stock
    weights, and a detached root risk context.
    """

    def __init__(
        self,
        *args: Any,
        feature_columns: list[str],
        root_feature_names: list[str],
        root_latent_dim: int = 32,
        risky_latent_dim: int = 32,
        routed_hidden_dim: int = 128,
        **kwargs: Any,
    ):
        self.feature_columns = list(feature_columns)
        self.root_feature_names = list(root_feature_names)
        self.root_latent_dim = int(root_latent_dim)
        self.risky_latent_dim = int(risky_latent_dim)
        self.routed_hidden_dim = int(routed_hidden_dim)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        vf_arch = [256, 128, 64]
        if isinstance(self.net_arch, dict):
            vf_arch = list(self.net_arch.get("vf", vf_arch))
        self.mlp_extractor = RootSplitRoutedMlpExtractor(
            features_dim=self.features_dim,
            stock_dim=self.stock_dim,
            feature_columns=self.feature_columns,
            root_feature_names=self.root_feature_names,
            activation_fn=self.activation_fn,
            root_latent_dim=self.root_latent_dim,
            risky_latent_dim=self.risky_latent_dim,
            hidden_dim=self.routed_hidden_dim,
            vf_arch=vf_arch,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("RoutedRootSplitBetaDirichletActorCriticPolicy requires a 1-D Box action space.")
        action_dim = int(np.prod(self.action_space.shape))
        if action_dim != self.stock_dim + 1:
            raise ValueError(f"action_dim={action_dim} must equal stock_dim+1={self.stock_dim + 1}")

        self.action_dist = RootSplitBetaDirichletDistribution(
            self.stock_dim,
            q_min=self.q_min,
            q_max=self.q_max,
            alpha_floor=self.alpha_floor,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            risky_alpha_max=self.risky_alpha_max,
        )
        self.action_net = RootSplitRoutedActionNet(
            self.root_latent_dim,
            self.risky_latent_dim,
            self.stock_dim,
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            feature_columns=self.feature_columns,
            root_feature_names=self.root_feature_names,
            root_latent_dim=self.root_latent_dim,
            risky_latent_dim=self.risky_latent_dim,
            routed_hidden_dim=self.routed_hidden_dim,
        )
        return data
