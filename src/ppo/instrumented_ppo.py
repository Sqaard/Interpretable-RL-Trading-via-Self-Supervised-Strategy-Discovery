"""SB3 PPO subclass that saves training-time rollout diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F


class InstrumentedPPO(PPO):
    """PPO with per-sample diagnostics from the rollout buffer.

    SB3's standard logger only keeps batch means such as `train/approx_kl` and
    `train/clip_fraction`. For Stage 0.1 we also need the per-sample values that
    cannot be reconstructed from a saved `.zip` model.
    """

    def __init__(
        self,
        *args: Any,
        instrumentation_dir: str | Path | None = None,
        rollout_dates: list[str] | None = None,
        save_sample_diagnostics: bool = True,
        save_rollout_snapshots: bool = True,
        rollout_snapshot_every_n_updates: int = 25,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.instrumentation_dir = Path(instrumentation_dir) if instrumentation_dir is not None else None
        self.rollout_dates = np.asarray(rollout_dates or [], dtype=object)
        self.save_sample_diagnostics = bool(save_sample_diagnostics)
        self.save_rollout_snapshots = bool(save_rollout_snapshots)
        self.rollout_snapshot_every_n_updates = max(1, int(rollout_snapshot_every_n_updates))
        self._sample_file_initialized = False
        self._update_file_initialized = False

        if self.instrumentation_dir is not None:
            self.instrumentation_dir.mkdir(parents=True, exist_ok=True)

    def _rollout_metadata(self) -> dict[str, np.ndarray]:
        n_steps = self.rollout_buffer.buffer_size
        n_envs = self.rollout_buffer.n_envs
        flat_indices = np.arange(n_steps * n_envs)
        rollout_step = flat_indices // n_envs
        env_idx = flat_indices % n_envs

        global_start = max(0, self.num_timesteps - (n_steps * n_envs))
        global_step = global_start + flat_indices

        if self.rollout_dates.size > 1 and n_envs == 1:
            episode_len = len(self.rollout_dates) - 1
            env_step = (global_start + rollout_step) % episode_len
            dates = self.rollout_dates[env_step]
        else:
            env_step = rollout_step
            dates = np.array([""] * len(flat_indices), dtype=object)

        rewards = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.rewards).reshape(-1)
        episode_starts = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.episode_starts).reshape(-1)
        return {
            "flat_index": flat_indices,
            "rollout_step": rollout_step,
            "env_idx": env_idx,
            "global_step": global_step,
            "env_step": np.asarray(env_step),
            "date": dates,
            "reward": rewards,
            "episode_start": episode_starts,
        }

    def _iter_rollout_minibatches(self, batch_size: int | None) -> Generator[tuple[np.ndarray, Any], None, None]:
        assert self.rollout_buffer.full, ""
        total = self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs
        indices = np.random.permutation(total)

        if not self.rollout_buffer.generator_ready:
            tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]
            for tensor in tensor_names:
                self.rollout_buffer.__dict__[tensor] = self.rollout_buffer.swap_and_flatten(
                    self.rollout_buffer.__dict__[tensor]
                )
            self.rollout_buffer.generator_ready = True

        if batch_size is None:
            batch_size = total

        start_idx = 0
        while start_idx < total:
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield batch_indices, self.rollout_buffer._get_samples(batch_indices)
            start_idx += batch_size

    def _append_csv(self, path: Path, rows: list[dict[str, Any]], initialized_attr: str) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        initialized = bool(getattr(self, initialized_attr))
        df.to_csv(path, mode="a", header=not initialized, index=False)
        setattr(self, initialized_attr, True)

    def _save_rollout_snapshot(self, metadata: dict[str, np.ndarray]) -> None:
        if self.instrumentation_dir is None or not self.save_rollout_snapshots:
            return
        if self._n_updates % self.rollout_snapshot_every_n_updates != 0:
            return

        path = self.instrumentation_dir / f"rollout_snapshot_update_{self._n_updates:05d}.npz"
        np.savez_compressed(
            path,
            observations=np.asarray(self.rollout_buffer.observations),
            actions=np.asarray(self.rollout_buffer.actions),
            old_log_prob=np.asarray(self.rollout_buffer.log_probs).reshape(-1),
            old_values=np.asarray(self.rollout_buffer.values).reshape(-1),
            advantages=np.asarray(self.rollout_buffer.advantages).reshape(-1),
            returns=np.asarray(self.rollout_buffer.returns).reshape(-1),
            rewards=metadata["reward"],
            episode_starts=metadata["episode_start"],
            flat_index=metadata["flat_index"],
            rollout_step=metadata["rollout_step"],
            env_idx=metadata["env_idx"],
            global_step=metadata["global_step"],
            env_step=metadata["env_step"],
            date=metadata["date"],
        )

    def train(self) -> None:
        """Update policy and save per-sample PPO diagnostics."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        metadata = self._rollout_metadata()
        self._save_rollout_snapshot(metadata)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        sample_rows: list[dict[str, Any]] = []
        update_rows: list[dict[str, Any]] = []

        continue_training = True
        loss = th.tensor(0.0, device=self.device)
        approx_kl_divs: list[float] = []

        for epoch in range(self.n_epochs):
            epoch_kl_divs = []
            minibatch_id = 0
            for sample_indices, rollout_data in self._iter_rollout_minibatches(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                raw_advantages = rollout_data.advantages
                advantages = raw_advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                min_policy_loss_terms = th.min(policy_loss_1, policy_loss_2)
                policy_loss = -min_policy_loss_terms.mean()

                pg_losses.append(policy_loss.item())
                clip_indicator = (th.abs(ratio - 1) > clip_range).float()
                clip_fraction = th.mean(clip_indicator).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                    entropy_sample = th.full_like(log_prob, th.nan)
                else:
                    entropy_loss = -th.mean(entropy)
                    entropy_sample = entropy

                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_sample = (th.exp(log_ratio) - 1) - log_ratio
                    approx_kl_div = th.mean(approx_kl_sample).cpu().item()
                    epoch_kl_divs.append(approx_kl_div)
                    approx_kl_divs.append(approx_kl_div)

                if self.save_sample_diagnostics and self.instrumentation_dir is not None:
                    arrays = {
                        "new_log_prob": log_prob.detach().cpu().numpy(),
                        "old_log_prob": rollout_data.old_log_prob.detach().cpu().numpy(),
                        "ratio": ratio.detach().cpu().numpy(),
                        "clip_indicator": clip_indicator.detach().cpu().numpy(),
                        "approx_kl_sample": approx_kl_sample.detach().cpu().numpy(),
                        "advantage_raw": raw_advantages.detach().cpu().numpy(),
                        "advantage_normalized": advantages.detach().cpu().numpy(),
                        "return": rollout_data.returns.detach().cpu().numpy(),
                        "old_value": rollout_data.old_values.detach().cpu().numpy(),
                        "new_value": values.detach().cpu().numpy(),
                        "value_error": (rollout_data.returns - values).detach().cpu().numpy(),
                        "policy_loss_unclipped": policy_loss_1.detach().cpu().numpy(),
                        "policy_loss_clipped": policy_loss_2.detach().cpu().numpy(),
                        "policy_loss_min_term": min_policy_loss_terms.detach().cpu().numpy(),
                        "entropy": entropy_sample.detach().cpu().numpy(),
                    }
                    for local_i, flat_i in enumerate(sample_indices):
                        sample_rows.append(
                            {
                                "update_before": self._n_updates,
                                "epoch": epoch,
                                "minibatch": minibatch_id,
                                "flat_index": int(metadata["flat_index"][flat_i]),
                                "rollout_step": int(metadata["rollout_step"][flat_i]),
                                "env_idx": int(metadata["env_idx"][flat_i]),
                                "global_step": int(metadata["global_step"][flat_i]),
                                "env_step": int(metadata["env_step"][flat_i]),
                                "date": str(metadata["date"][flat_i]),
                                "reward": float(metadata["reward"][flat_i]),
                                "episode_start": bool(metadata["episode_start"][flat_i]),
                                **{name: float(values_arr[local_i]) for name, values_arr in arrays.items()},
                            }
                        )

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                minibatch_id += 1

            self._n_updates += 1
            update_rows.append(
                {
                    "update_after": self._n_updates,
                    "epoch": epoch,
                    "mean_epoch_approx_kl": float(np.mean(epoch_kl_divs)) if epoch_kl_divs else np.nan,
                    "mean_policy_gradient_loss": float(np.mean(pg_losses)) if pg_losses else np.nan,
                    "mean_value_loss": float(np.mean(value_losses)) if value_losses else np.nan,
                    "mean_entropy_loss": float(np.mean(entropy_losses)) if entropy_losses else np.nan,
                    "mean_clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else np.nan,
                    "early_stopped": not continue_training,
                    "clip_range": float(clip_range),
                }
            )
            if not continue_training:
                break

        if self.instrumentation_dir is not None:
            self._append_csv(
                self.instrumentation_dir / "training_sample_diagnostics.csv",
                sample_rows,
                "_sample_file_initialized",
            )
            self._append_csv(
                self.instrumentation_dir / "training_update_diagnostics.csv",
                update_rows,
                "_update_file_initialized",
            )

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
