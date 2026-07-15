"""Strict, episode-complete replay storage for the B2 PPO learner.

This module is deliberately separate from :mod:`bplus_v22.buffer`.  The latter
is immutable B1 evidence code and carries a legacy dense-reward channel.  B2
optimizes exactly two direct outcomes (collision cost and corrected terminal
overtake), so silently retaining the legacy reward critic would change the
experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from bplus_v22 import (
    BC_FEATURE_DIM,
    HISTORY_OFFSETS,
    LIDAR_BEAMS,
    PRIVILEGED_FEATURE_DIM,
    SCALAR_HISTORY_DIM,
)
from bplus_v22.buffer import variable_discount_gae
from bplus_v22.macro import MacroSignals


B2_CHANNELS = ("collision", "performance")


def _finite_vector(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    output = np.array(array, dtype=np.float32, copy=True)
    output.setflags(write=False)
    return output


def _nonempty(value: str, name: str) -> str:
    text = str(value)
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _integer(value: int, name: str, minimum: int = 0) -> int:
    result = int(value)
    if result != value or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _digest(value: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError("applied composition digest must be lowercase SHA256")
    return text


@dataclass(frozen=True)
class MacroReplayRecord:
    """One B2 macro transition with exact behavior-policy provenance."""

    scenario_id: str
    l2_id: str
    episode_id: str
    macro_index: int
    arm: str
    training_seed: int
    policy_iteration: int
    checkpoint_schema: str
    bc_feature: np.ndarray
    lidar_history: np.ndarray
    scalar_history: np.ndarray
    privileged_critic_feature: np.ndarray
    latent: np.ndarray
    old_log_prob: float
    old_entropy: float
    entropy_intervention: float
    entropy_steer_given_intervention: float
    entropy_brake_gate_given_intervention: float
    entropy_brake_magnitude_given_brake: float
    intervention_offset: float
    conditional_brake_offset: float
    steer_std_scale: float
    brake_std_scale: float
    schedule_id: str
    requested_residual: np.ndarray
    applied_composition_digest: str
    signals: MacroSignals
    collision_value: float
    performance_value: float
    collision_trunc_next_value: float = 0.0
    performance_trunc_next_value: float = 0.0
    episode_start: bool = False
    bc_hidden_reset: bool = False

    def __post_init__(self) -> None:
        for field in ("scenario_id", "l2_id", "episode_id", "arm", "checkpoint_schema", "schedule_id"):
            object.__setattr__(self, field, _nonempty(getattr(self, field), field))
        for field in ("macro_index", "training_seed", "policy_iteration"):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        object.__setattr__(
            self,
            "bc_feature",
            _finite_vector(self.bc_feature, (BC_FEATURE_DIM,), "bc_feature"),
        )
        object.__setattr__(
            self,
            "lidar_history",
            _finite_vector(
                self.lidar_history,
                (len(HISTORY_OFFSETS), LIDAR_BEAMS),
                "lidar_history",
            ),
        )
        object.__setattr__(
            self,
            "scalar_history",
            _finite_vector(self.scalar_history, (SCALAR_HISTORY_DIM,), "scalar_history"),
        )
        object.__setattr__(
            self,
            "privileged_critic_feature",
            _finite_vector(
                self.privileged_critic_feature,
                (PRIVILEGED_FEATURE_DIM,),
                "privileged_critic_feature",
            ),
        )
        latent = _finite_vector(self.latent, (4,), "latent")
        if latent[0] not in (0.0, 1.0) or latent[2] not in (0.0, 1.0):
            raise ValueError("latent intervention/brake gates must be 0/1")
        if latent[2] > latent[0]:
            raise ValueError("latent brake gate cannot be active outside intervention")
        if latent[0] == 0.0 and latent[1] != 0.0:
            raise ValueError("NO_OP latent must canonicalize steering to zero")
        if latent[2] == 0.0 and latent[3] != 0.0:
            raise ValueError("inactive brake latent must canonicalize to zero")
        object.__setattr__(self, "latent", latent)
        object.__setattr__(
            self,
            "requested_residual",
            _finite_vector(self.requested_residual, (2,), "requested_residual"),
        )
        object.__setattr__(self, "applied_composition_digest", _digest(self.applied_composition_digest))
        if not isinstance(self.signals, MacroSignals):
            raise TypeError("signals must be MacroSignals")
        numeric = (
            self.old_log_prob,
            self.old_entropy,
            self.entropy_intervention,
            self.entropy_steer_given_intervention,
            self.entropy_brake_gate_given_intervention,
            self.entropy_brake_magnitude_given_brake,
            self.intervention_offset,
            self.conditional_brake_offset,
            self.steer_std_scale,
            self.brake_std_scale,
            self.collision_value,
            self.performance_value,
            self.collision_trunc_next_value,
            self.performance_trunc_next_value,
        )
        if not all(np.isfinite(float(value)) for value in numeric):
            raise ValueError("macro replay record contains nonfinite scalar")
        if self.steer_std_scale <= 0.0 or self.brake_std_scale <= 0.0:
            raise ValueError("macro replay std scales must be positive")
        if not self.signals.truncated and (
            self.collision_trunc_next_value != 0.0
            or self.performance_trunc_next_value != 0.0
        ):
            raise ValueError("non-truncated transition cannot carry truncation bootstrap")
        if bool(self.episode_start) != bool(self.bc_hidden_reset):
            raise ValueError("episode_start and BC hidden reset markers must agree")


def validate_complete_episode(records: Sequence[MacroReplayRecord]) -> tuple[MacroReplayRecord, ...]:
    """Validate and return one complete episode; partial episodes are rejected."""

    episode = tuple(records)
    if not episode or any(not isinstance(row, MacroReplayRecord) for row in episode):
        raise ValueError("complete episode requires MacroReplayRecord rows")
    first = episode[0]
    shared = (
        "scenario_id",
        "l2_id",
        "episode_id",
        "arm",
        "training_seed",
        "policy_iteration",
        "checkpoint_schema",
        "schedule_id",
        "intervention_offset",
        "conditional_brake_offset",
        "steer_std_scale",
        "brake_std_scale",
    )
    for index, row in enumerate(episode):
        if row.macro_index != index:
            raise ValueError("episode macro indices must be contiguous from zero")
        if any(getattr(row, name) != getattr(first, name) for name in shared):
            raise ValueError("episode identity/config changes within an episode")
        if row.episode_start != (index == 0) or row.bc_hidden_reset != (index == 0):
            raise ValueError("episode boundary/reset markers are inconsistent")
        boundary = row.signals.terminated or row.signals.truncated
        if boundary != (index == len(episode) - 1):
            raise ValueError("only the final macro may contain the episode boundary")
        if index < len(episode) - 1 and (
            row.signals.collision_cost != 0.0
            or row.signals.performance_reward != 0.0
        ):
            raise ValueError("direct outcome signals may appear only on the final macro")
    final = episode[-1]
    if final.signals.collision_cost > 0.0 and not final.signals.terminated:
        raise ValueError("collision cost requires a true terminal boundary")
    return episode


@dataclass(frozen=True)
class MacroReplayBatch:
    """Collated B2 replay.  There is intentionally no legacy reward channel."""

    scenario_id: tuple[str, ...]
    l2_id: tuple[str, ...]
    episode_id: tuple[str, ...]
    arm: tuple[str, ...]
    schedule_id: tuple[str, ...]
    applied_composition_digest: tuple[str, ...]
    bc_feature: np.ndarray
    lidar_history: np.ndarray
    scalar_history: np.ndarray
    privileged_critic_feature: np.ndarray
    latent: np.ndarray
    old_log_prob: np.ndarray
    old_entropy: np.ndarray
    entropy_intervention: np.ndarray
    entropy_steer_given_intervention: np.ndarray
    entropy_brake_gate_given_intervention: np.ndarray
    entropy_brake_magnitude_given_brake: np.ndarray
    intervention_offset: np.ndarray
    conditional_brake_offset: np.ndarray
    steer_std_scale: np.ndarray
    brake_std_scale: np.ndarray
    requested_residual: np.ndarray
    collision_advantage: np.ndarray
    collision_return: np.ndarray
    performance_advantage: np.ndarray
    performance_return: np.ndarray
    collision_cost: np.ndarray
    performance_reward: np.ndarray
    collision_value: np.ndarray
    performance_value: np.ndarray
    discount: np.ndarray
    macro_length: np.ndarray
    macro_index: np.ndarray
    collision_trunc_next_value: np.ndarray
    performance_trunc_next_value: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_start: np.ndarray
    bc_hidden_reset: np.ndarray
    training_seed: int
    policy_iteration: int
    checkpoint_schema: str

    def __post_init__(self) -> None:
        count = len(self.old_log_prob)
        if count == 0:
            raise ValueError("macro replay batch is empty")
        if any(len(value) != count for value in (
            self.scenario_id,
            self.l2_id,
            self.episode_id,
            self.arm,
            self.schedule_id,
            self.applied_composition_digest,
        )):
            raise ValueError("macro replay metadata length mismatch")
        if any(_digest(value) != value for value in self.applied_composition_digest):
            raise ValueError("macro replay composition digest drift")
        shapes = {
            "bc_feature": (count, BC_FEATURE_DIM),
            "lidar_history": (count, len(HISTORY_OFFSETS), LIDAR_BEAMS),
            "scalar_history": (count, SCALAR_HISTORY_DIM),
            "privileged_critic_feature": (count, PRIVILEGED_FEATURE_DIM),
            "latent": (count, 4),
            "intervention_offset": (count,),
            "conditional_brake_offset": (count,),
            "old_entropy": (count,),
            "entropy_intervention": (count,),
            "entropy_steer_given_intervention": (count,),
            "entropy_brake_gate_given_intervention": (count,),
            "entropy_brake_magnitude_given_brake": (count,),
            "steer_std_scale": (count,),
            "brake_std_scale": (count,),
            "requested_residual": (count, 2),
            "collision_advantage": (count,),
            "collision_return": (count,),
            "performance_advantage": (count,),
            "performance_return": (count,),
            "collision_cost": (count,),
            "performance_reward": (count,),
            "collision_value": (count,),
            "performance_value": (count,),
            "discount": (count,),
            "macro_length": (count,),
            "macro_index": (count,),
            "collision_trunc_next_value": (count,),
            "performance_trunc_next_value": (count,),
            "terminated": (count,),
            "truncated": (count,),
            "episode_start": (count,),
            "bc_hidden_reset": (count,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"macro replay batch {name} shape mismatch")
            if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
                raise ValueError(f"macro replay batch {name} is nonfinite")
            immutable = np.array(value, copy=True)
            immutable.setflags(write=False)
            object.__setattr__(self, name, immutable)
        if np.any(self.terminated & self.truncated):
            raise ValueError("macro replay batch contains terminate+truncate")

    def tensors(self, device: torch.device | str) -> dict[str, torch.Tensor]:
        """Return the exact numeric replay context consumed by B2 PPO."""

        floats = (
            "bc_feature",
            "lidar_history",
            "scalar_history",
            "privileged_critic_feature",
            "latent",
            "old_log_prob",
            "old_entropy",
            "entropy_intervention",
            "entropy_steer_given_intervention",
            "entropy_brake_gate_given_intervention",
            "entropy_brake_magnitude_given_brake",
            "intervention_offset",
            "conditional_brake_offset",
            "steer_std_scale",
            "brake_std_scale",
            "requested_residual",
            "collision_advantage",
            "collision_return",
            "performance_advantage",
            "performance_return",
            "collision_cost",
            "performance_reward",
            "collision_value",
            "performance_value",
            "discount",
            "collision_trunc_next_value",
            "performance_trunc_next_value",
        )
        output = {
            name: torch.tensor(getattr(self, name), dtype=torch.float32, device=device)
            for name in floats
        }
        output.update(
            {
                "macro_length": torch.tensor(self.macro_length, dtype=torch.int64, device=device),
                "macro_index": torch.tensor(self.macro_index, dtype=torch.int64, device=device),
                "terminated": torch.tensor(self.terminated, dtype=torch.bool, device=device),
                "truncated": torch.tensor(self.truncated, dtype=torch.bool, device=device),
                "episode_start": torch.tensor(self.episode_start, dtype=torch.bool, device=device),
                "bc_hidden_reset": torch.tensor(
                    self.bc_hidden_reset, dtype=torch.bool, device=device
                ),
            }
        )
        return output


class EpisodeCompleteMacroBuffer:
    """Collect whole episodes to a fixed episode count or transition minimum.

    The final episode may overshoot ``minimum_transitions``.  This is deliberate:
    cutting exactly at the budget would lose or mix the terminal product signal.
    Once ready, the buffer is sealed until ``reset``.
    """

    def __init__(
        self,
        minimum_transitions: int | None = None,
        *,
        target_episodes: int | None = None,
    ):
        if (minimum_transitions is None) == (target_episodes is None):
            raise ValueError(
                "episode-complete buffer requires exactly one stopping contract"
            )
        self.minimum_transitions = (
            None
            if minimum_transitions is None
            else _integer(minimum_transitions, "minimum_transitions", minimum=1)
        )
        self.target_episodes = (
            None
            if target_episodes is None
            else _integer(target_episodes, "target_episodes", minimum=1)
        )
        self.records: list[MacroReplayRecord] = []
        self.episode_count = 0
        self._rollout_identity: tuple | None = None

    @property
    def ready(self) -> bool:
        if self.target_episodes is not None:
            return self.episode_count >= self.target_episodes
        return len(self.records) >= int(self.minimum_transitions)

    def reset(self) -> None:
        self.records.clear()
        self.episode_count = 0
        self._rollout_identity = None

    def add_episode(self, records: Sequence[MacroReplayRecord]) -> None:
        if self.ready:
            raise RuntimeError("episode-complete buffer is sealed once ready")
        episode = validate_complete_episode(records)
        first = episode[0]
        identity = (
            first.arm,
            first.training_seed,
            first.policy_iteration,
            first.checkpoint_schema,
            first.schedule_id,
            float(first.intervention_offset),
            float(first.conditional_brake_offset),
            float(first.steer_std_scale),
            float(first.brake_std_scale),
        )
        if self._rollout_identity is None:
            self._rollout_identity = identity
        elif identity != self._rollout_identity:
            raise ValueError("behavior policy/config changes within one rollout")
        if any(
            (
                row.arm,
                row.training_seed,
                row.policy_iteration,
                row.checkpoint_schema,
                row.schedule_id,
                float(row.intervention_offset),
                float(row.conditional_brake_offset),
                float(row.steer_std_scale),
                float(row.brake_std_scale),
            )
            != identity
            for row in episode
        ):
            raise ValueError("behavior policy/config changes within one episode")
        self.records.extend(episode)
        self.episode_count += 1

    def collate(self) -> MacroReplayBatch:
        if not self.ready or not self.records:
            raise RuntimeError("episode-complete buffer has not reached its stopping contract")
        rows = tuple(self.records)
        discount = np.asarray([row.signals.discount for row in rows], dtype=np.float64)
        terminated = np.asarray([row.signals.terminated for row in rows], dtype=bool)
        truncated = np.asarray([row.signals.truncated for row in rows], dtype=bool)
        if not (terminated[-1] or truncated[-1]):
            raise AssertionError("episode-complete buffer ended without a boundary")

        output: dict[str, np.ndarray] = {}
        for channel, signal_name, value_name, trunc_name in (
            (
                "collision",
                "collision_cost",
                "collision_value",
                "collision_trunc_next_value",
            ),
            (
                "performance",
                "performance_reward",
                "performance_value",
                "performance_trunc_next_value",
            ),
        ):
            signal = np.asarray([getattr(row.signals, signal_name) for row in rows])
            value = np.asarray([getattr(row, value_name) for row in rows])
            trunc_next = np.asarray([getattr(row, trunc_name) for row in rows])
            advantage, returns = variable_discount_gae(
                signal,
                value,
                discount,
                terminated,
                truncated,
                trunc_next,
                last_value=0.0,
            )
            output[f"{channel}_advantage"] = advantage
            output[f"{channel}_return"] = returns

        first = rows[0]
        return MacroReplayBatch(
            scenario_id=tuple(row.scenario_id for row in rows),
            l2_id=tuple(row.l2_id for row in rows),
            episode_id=tuple(row.episode_id for row in rows),
            arm=tuple(row.arm for row in rows),
            schedule_id=tuple(row.schedule_id for row in rows),
            applied_composition_digest=tuple(
                row.applied_composition_digest for row in rows
            ),
            bc_feature=np.stack([row.bc_feature for row in rows]),
            lidar_history=np.stack([row.lidar_history for row in rows]),
            scalar_history=np.stack([row.scalar_history for row in rows]),
            privileged_critic_feature=np.stack(
                [row.privileged_critic_feature for row in rows]
            ),
            latent=np.stack([row.latent for row in rows]),
            old_log_prob=np.asarray([row.old_log_prob for row in rows], dtype=np.float32),
            old_entropy=np.asarray([row.old_entropy for row in rows], dtype=np.float32),
            entropy_intervention=np.asarray(
                [row.entropy_intervention for row in rows], dtype=np.float32
            ),
            entropy_steer_given_intervention=np.asarray(
                [row.entropy_steer_given_intervention for row in rows], dtype=np.float32
            ),
            entropy_brake_gate_given_intervention=np.asarray(
                [row.entropy_brake_gate_given_intervention for row in rows],
                dtype=np.float32,
            ),
            entropy_brake_magnitude_given_brake=np.asarray(
                [row.entropy_brake_magnitude_given_brake for row in rows],
                dtype=np.float32,
            ),
            intervention_offset=np.asarray(
                [row.intervention_offset for row in rows], dtype=np.float32
            ),
            conditional_brake_offset=np.asarray(
                [row.conditional_brake_offset for row in rows], dtype=np.float32
            ),
            steer_std_scale=np.asarray(
                [row.steer_std_scale for row in rows], dtype=np.float32
            ),
            brake_std_scale=np.asarray(
                [row.brake_std_scale for row in rows], dtype=np.float32
            ),
            requested_residual=np.stack([row.requested_residual for row in rows]),
            collision_advantage=output["collision_advantage"],
            collision_return=output["collision_return"],
            performance_advantage=output["performance_advantage"],
            performance_return=output["performance_return"],
            collision_cost=np.asarray(
                [row.signals.collision_cost for row in rows], dtype=np.float32
            ),
            performance_reward=np.asarray(
                [row.signals.performance_reward for row in rows], dtype=np.float32
            ),
            collision_value=np.asarray([row.collision_value for row in rows], dtype=np.float32),
            performance_value=np.asarray(
                [row.performance_value for row in rows], dtype=np.float32
            ),
            discount=discount.astype(np.float32),
            macro_length=np.asarray([row.signals.length for row in rows], dtype=np.int16),
            macro_index=np.asarray([row.macro_index for row in rows], dtype=np.int32),
            collision_trunc_next_value=np.asarray(
                [row.collision_trunc_next_value for row in rows], dtype=np.float32
            ),
            performance_trunc_next_value=np.asarray(
                [row.performance_trunc_next_value for row in rows], dtype=np.float32
            ),
            terminated=terminated,
            truncated=truncated,
            episode_start=np.asarray([row.episode_start for row in rows], dtype=bool),
            bc_hidden_reset=np.asarray([row.bc_hidden_reset for row in rows], dtype=bool),
            training_seed=first.training_seed,
            policy_iteration=first.policy_iteration,
            checkpoint_schema=first.checkpoint_schema,
        )


def require_b2_tensor_batch(batch: Mapping[str, torch.Tensor]) -> int:
    """Validate the numeric subset required by the pure PPO loss/replay hook."""

    required = {
        "bc_feature": (BC_FEATURE_DIM,),
        "lidar_history": (len(HISTORY_OFFSETS), LIDAR_BEAMS),
        "scalar_history": (SCALAR_HISTORY_DIM,),
        "privileged_critic_feature": (PRIVILEGED_FEATURE_DIM,),
        "latent": (4,),
        "old_log_prob": (),
        "old_entropy": (),
        "intervention_offset": (),
        "conditional_brake_offset": (),
        "steer_std_scale": (),
        "brake_std_scale": (),
        "collision_cost": (),
        "collision_advantage": (),
        "collision_return": (),
        "performance_advantage": (),
        "performance_return": (),
    }
    if not isinstance(batch, Mapping) or not set(required).issubset(batch):
        raise ValueError("B2 tensor batch lacks required replay fields")
    count = int(batch["old_log_prob"].shape[0])
    if count <= 0:
        raise ValueError("B2 tensor batch is empty")
    for name, tail in required.items():
        value = batch[name]
        if not isinstance(value, torch.Tensor) or value.shape != (count, *tail):
            raise ValueError(f"B2 tensor batch {name} shape mismatch")
        if not torch.all(torch.isfinite(value)):
            raise ValueError(f"B2 tensor batch {name} is nonfinite")
    return count
