"""Production 100 Hz collection adapter for B7 recurrent PPO.

The audited B4 simulator loop is reused verbatim.  This adapter adds the raw
observation/recurrent ledgers that B7 needs for full-sequence replay and
replaces the B4 terminal-only reward after the episode has been classified.
"""

from __future__ import annotations

import numpy as np
import torch

from bplus_v22.b4_direct import FEATURE_DIM
from bplus_v22.b4_env import B4EpisodeResult, run_b4_episode
from bplus_v22.b7_recurrent import (
    B7Episode,
    B7RecurrentPolicy,
    B7Selection,
    B7Transition,
    LIDAR_DIM,
    PRODUCT_HORIZON_SECONDS,
    task_reward_schedule,
)


class B7RolloutAdapter:
    """Expose the B4 collector interface while retaining recurrent inputs."""

    def __init__(self, policy: B7RecurrentPolicy):
        self.policy = policy
        self.actor = policy.actor
        self.critic = policy.critic
        self.action_std = policy.action_std
        self._bc_hidden: torch.Tensor | None = None
        self.lidar: list[np.ndarray] = []
        self.previous_speed: list[float] = []
        self.old_mean: list[np.ndarray] = []
        self.bc_mean: list[np.ndarray] = []
        self.remaining_time: list[float] = []

    def zero_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if int(batch_size) != 1:
            raise ValueError("B7 collector requires batch size one")
        self.lidar.clear()
        self.previous_speed.clear()
        self.old_mean.clear()
        self.bc_mean.clear()
        self.remaining_time.clear()
        self._bc_hidden = self.policy.zero_hidden(device)
        return self.policy.zero_hidden(device)

    def feature_step(
        self,
        lidar: torch.Tensor,
        previous_speed: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._bc_hidden is None:
            raise RuntimeError("B7 rollout adapter was not reset")
        features, next_hidden = self.actor.forward_features(lidar, previous_speed, hidden)
        bc_action, self._bc_hidden = self.policy.bc_reference(
            lidar, previous_speed, self._bc_hidden
        )
        feature = features[:, -1, :]
        if feature.shape != (1, FEATURE_DIM):
            raise AssertionError("B7 rollout feature shape drift")
        self.lidar.append(
            lidar[0, -1].detach().cpu().numpy().astype(np.float32).copy()
        )
        self.previous_speed.append(float(previous_speed[0, -1, 0].item()))
        self.bc_mean.append(
            bc_action[0, -1].detach().cpu().numpy().astype(np.float32).copy()
        )
        return feature, next_hidden

    def mean_from_feature(self, feature: torch.Tensor) -> torch.Tensor:
        mean = self.actor.output_layer(feature)
        self.old_mean.append(mean[0].detach().cpu().numpy().astype(np.float32).copy())
        return mean

    def log_prob(self, mean: torch.Tensor, raw_action: torch.Tensor) -> torch.Tensor:
        return self.policy.log_prob(mean, raw_action)

    def sample_raw(self, mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy.sample_raw(mean)

    def value(self, privileged_feature: torch.Tensor) -> torch.Tensor:
        step_index = len(self.old_mean) - 1
        remaining = max(
            0.0,
            (PRODUCT_HORIZON_SECONDS - 0.01 * step_index) / PRODUCT_HORIZON_SECONDS,
        )
        self.remaining_time.append(float(remaining))
        augmented = torch.cat(
            (
                privileged_feature,
                torch.full(
                    (len(privileged_feature), 1),
                    remaining,
                    dtype=privileged_feature.dtype,
                    device=privileged_feature.device,
                ),
            ),
            dim=1,
        )
        return self.policy.value(augmented)


def _convert_episode(
    result: B4EpisodeResult,
    adapter: B7RolloutAdapter,
    selection: B7Selection,
) -> B7Episode:
    count = len(result.transitions)
    ledgers = (
        adapter.lidar,
        adapter.previous_speed,
        adapter.old_mean,
        adapter.bc_mean,
        adapter.remaining_time,
    )
    if any(len(value) != count for value in ledgers):
        raise AssertionError("B7 recurrent rollout ledger length drift")
    collision = bool(result.outcome.collision_any)
    terminal_overtake = result.outcome.corrected_outcome3 == "overtake"
    rewards = task_reward_schedule(
        count,
        collision_any=collision,
        terminal_overtake=terminal_overtake,
    )
    transitions: list[B7Transition] = []
    for index, row in enumerate(result.transitions):
        privileged = np.concatenate(
            (
                np.asarray(row.privileged_feature, dtype=np.float32),
                np.asarray([adapter.remaining_time[index]], dtype=np.float32),
            )
        )
        transitions.append(
            B7Transition(
                step_index=index,
                lidar=adapter.lidar[index],
                previous_speed=adapter.previous_speed[index],
                privileged_feature=privileged,
                old_mean=adapter.old_mean[index],
                bc_mean=adapter.bc_mean[index],
                raw_action=np.asarray(row.raw_action, dtype=np.float32).copy(),
                executed_action=np.asarray(row.executed_action, dtype=np.float32).copy(),
                projection_delta=np.asarray(row.projection_delta, dtype=np.float32).copy(),
                old_log_prob=float(row.old_log_prob),
                old_value=float(row.old_value),
                reward=float(rewards[index]),
                terminated=index == count - 1,
            )
        )
    return B7Episode(
        scenario=result.scenario,
        episode_id=result.episode_id,
        sampler_role=selection.role,
        hard_priority=selection.hard_priority,
        transitions=tuple(transitions),
        collision_any=collision,
        terminal_overtake=terminal_overtake,
        corrected_outcome3=str(result.outcome.corrected_outcome3),
        terminal_reason=result.terminal_reason,
    )


def run_b7_episode(
    policy: B7RecurrentPolicy,
    device: torch.device,
    selection: B7Selection,
    *,
    episode_id: int,
    deterministic: bool = False,
) -> tuple[B7Episode, B4EpisodeResult]:
    """Collect one valid episode and return B7 replay plus trajectory evidence."""

    adapter = B7RolloutAdapter(policy)
    result = run_b4_episode(
        adapter,
        device,
        selection.scenario,
        episode_id=episode_id,
        deterministic=deterministic,
    )
    return _convert_episode(result, adapter, selection), result
