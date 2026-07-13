"""B4 plain-End2Race direct-head PPO contracts.

This module contains the simulator-independent B4 implementation.  The PPO
policy variable is an unbounded raw Normal latent.  A fixed, fully recorded
actuator projection maps that latent to the physical command.  Only the
canonical End2Race ``output_layer`` is trainable; the privileged critic and
full resume state never enter deployment checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from bplus_v22.ppo_env import B2Scenario, B2ScenarioSets
from model import End2Race


B4_POLICY_SCHEMA = "end2race-b4-direct-head-policy-1"
B4_FULL_CHECKPOINT_SCHEMA = "end2race-b4-direct-head-full-checkpoint-1"
B4_ACTOR_SNAPSHOT_SCHEMA = "plain-end2race-state-dict"
B4_CURRICULUM_SCHEMA = "end2race-b4-curriculum-1"
B4_CURRICULUM_DOMAIN = b"end2race:b4:curriculum:v1\0"
B4_MINIBATCH_DOMAIN = b"end2race:b4:minibatch:v1\0"

FEATURE_DIM = 1680
PRIVILEGED_DIM = 12
ACTION_DIM = 2
TRAINING_ROWS = 1640
COLLISION_POOL_ROWS = 81
OVERTAKE_POOL_ROWS = 1001
FOLLOW_POOL_ROWS = 558
EPISODES_PER_ITERATION = 16
COLLISION_EPISODES = 6
OVERTAKE_EPISODES = 6
FOLLOW_EPISODES = 4
CURRICULUM_PATTERN = (
    "collision",
    "overtake",
    "follow",
    "collision",
    "overtake",
    "collision",
    "overtake",
    "follow",
    "collision",
    "overtake",
    "collision",
    "follow",
    "overtake",
    "collision",
    "overtake",
    "follow",
)
CANONICAL_ACTOR_KEYS = (
    "k",
    "dummy_embedding",
    "speed_mlp.0.weight",
    "speed_mlp.0.bias",
    "gru.weight_ih_l0",
    "gru.weight_hh_l0",
    "gru.bias_ih_l0",
    "gru.bias_hh_l0",
    "output_layer.0.weight",
    "output_layer.0.bias",
    "output_layer.2.weight",
    "output_layer.2.bias",
)


@dataclass(frozen=True)
class B4Config:
    actor_lr: float = 3e-5
    critic_lr: float = 3e-4
    steer_std: float = 0.03
    speed_std: float = 0.20
    clip_eps: float = 0.10
    target_weighted_kl: float = 0.015
    actor_epochs: int = 3
    critic_epochs: int = 3
    minibatch_size: int = 1024
    gamma: float = 0.999
    gae_lambda: float = 0.997
    entropy_coef: float = 0.0
    mean_bound_coef: float = 0.01
    max_grad_norm: float = 0.5
    episodes_per_iteration: int = EPISODES_PER_ITERATION
    collision_episodes: int = COLLISION_EPISODES
    overtake_episodes: int = OVERTAKE_EPISODES
    follow_episodes: int = FOLLOW_EPISODES
    iterations: int = 30
    seeds: tuple[int, ...] = (1,)
    snapshots: tuple[int, int, int, int] = (0, 10, 20, 30)

    def __post_init__(self) -> None:
        positive = {
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "steer_std": self.steer_std,
            "speed_std": self.speed_std,
            "clip_eps": self.clip_eps,
            "target_weighted_kl": self.target_weighted_kl,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "max_grad_norm": self.max_grad_norm,
        }
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive.values()):
            raise ValueError("B4 positive configuration field is invalid")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("B4 gamma/lambda is invalid")
        if self.entropy_coef != 0.0:
            raise ValueError("B4 entropy coefficient is frozen at zero")
        if not math.isfinite(self.mean_bound_coef) or self.mean_bound_coef < 0.0:
            raise ValueError("B4 mean bound coefficient is invalid")
        integer_fields = (
            self.actor_epochs,
            self.critic_epochs,
            self.minibatch_size,
            self.episodes_per_iteration,
            self.collision_episodes,
            self.overtake_episodes,
            self.follow_episodes,
            self.iterations,
        )
        if any(int(value) != value or int(value) <= 0 for value in integer_fields):
            raise ValueError("B4 integer configuration field is invalid")
        if self.episodes_per_iteration != sum(
            (self.collision_episodes, self.overtake_episodes, self.follow_episodes)
        ):
            raise ValueError("B4 curriculum counts do not sum to one iteration")
        if tuple(self.seeds) != (1,) or tuple(self.snapshots) != (0, 10, 20, 30):
            raise ValueError("B4 seed/snapshot contract drift")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["seeds"] = list(self.seeds)
        value["snapshots"] = list(self.snapshots)
        return value


FROZEN_B4_CONFIG = B4Config()


def validate_frozen_config(value: Mapping[str, Any]) -> B4Config:
    """Fail closed when a RunPlan attempts to tune the approved pilot."""

    expected = FROZEN_B4_CONFIG.as_dict()
    if dict(value) != expected:
        mismatches = sorted(
            key for key in set(expected) | set(value) if value.get(key) != expected.get(key)
        )
        raise ValueError(f"B4 frozen configuration drift: {mismatches}")
    return FROZEN_B4_CONFIG


def _is_plain_actor_state(state: Mapping[str, torch.Tensor]) -> bool:
    return tuple(state.keys()) == CANONICAL_ACTOR_KEYS and all(
        torch.is_tensor(value) for value in state.values()
    )


def strict_plain_actor_from_state(
    state: Mapping[str, torch.Tensor], device: torch.device | str = "cpu"
) -> End2Race:
    """Load a deployment actor without residual autodetection or wrappers."""

    if not isinstance(state, Mapping) or not _is_plain_actor_state(state):
        raise ValueError("B4 deployment state is not the canonical 12-key End2Race state_dict")
    actor = End2Race(mask_prob=0.0, hidden_scale=4).to(device)
    actor.load_state_dict(dict(state), strict=True)
    actor.eval()
    return actor


def load_strict_plain_actor(
    path: str | Path, device: torch.device | str = "cpu"
) -> End2Race:
    state = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("B4 actor-only checkpoint is not a state_dict mapping")
    return strict_plain_actor_from_state(state, device)


def _clone_cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


class B4DirectHeadPolicy(nn.Module):
    """Training-only wrapper around one canonical plain End2Race actor."""

    def __init__(self, bc_state: Mapping[str, torch.Tensor], config: B4Config = FROZEN_B4_CONFIG):
        super().__init__()
        self.config = config
        self.actor = strict_plain_actor_from_state(bc_state)
        self.critic = nn.Sequential(
            nn.Linear(PRIVILEGED_DIM, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.register_buffer(
            "action_std",
            torch.tensor([config.steer_std, config.speed_std], dtype=torch.float32),
            persistent=True,
        )
        self._freeze_backbone()
        self._initial_frozen = self.frozen_state()

    def _freeze_backbone(self) -> None:
        for name, parameter in self.actor.named_parameters():
            parameter.requires_grad_(name.startswith("output_layer."))
        trainable = tuple(
            name for name, parameter in self.actor.named_parameters() if parameter.requires_grad
        )
        if trainable != (
            "output_layer.0.weight",
            "output_layer.0.bias",
            "output_layer.2.weight",
            "output_layer.2.bias",
        ):
            raise AssertionError(f"B4 trainable actor inventory drift: {trainable}")

    @property
    def trainable_actor_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for parameter in self.actor.parameters() if parameter.requires_grad)

    @property
    def frozen_actor_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for parameter in self.actor.parameters() if not parameter.requires_grad)

    def frozen_state(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.actor.state_dict().items()
            if not name.startswith("output_layer.")
        }

    def assert_frozen_exact(self) -> None:
        current = self.frozen_state()
        if current.keys() != self._initial_frozen.keys() or any(
            not torch.equal(current[name], self._initial_frozen[name]) for name in current
        ):
            raise AssertionError("B4 frozen actor tensor changed")

    def zero_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            (1, int(batch_size), self.actor.gru.hidden_size),
            dtype=next(self.actor.parameters()).dtype,
            device=device,
        )

    def feature_step(
        self,
        lidar: torch.Tensor,
        previous_speed: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features, next_hidden = self.actor.forward_features(lidar, previous_speed, hidden)
        feature = features[:, -1, :].detach()
        if feature.shape[-1] != FEATURE_DIM:
            raise AssertionError("B4 frozen feature dimension drift")
        return feature, next_hidden.detach()

    def mean_from_feature(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 2 or feature.shape[1] != FEATURE_DIM:
            raise ValueError("B4 actor feature must be [N,1680]")
        return self.actor.output_layer(feature)

    def log_prob(self, mean: torch.Tensor, raw_action: torch.Tensor) -> torch.Tensor:
        if mean.shape != raw_action.shape or mean.shape[-1] != ACTION_DIM:
            raise ValueError("B4 mean/raw action shape mismatch")
        std = self.action_std.to(dtype=mean.dtype, device=mean.device)
        log_scale = torch.log(std)
        value = -0.5 * (((raw_action - mean) / std) ** 2 + math.log(2.0 * math.pi)) - log_scale
        return value.sum(dim=-1)

    def sample_raw(self, mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        std = self.action_std.to(dtype=mean.dtype, device=mean.device)
        raw = mean + torch.randn_like(mean) * std
        return raw, self.log_prob(mean, raw)

    def value(self, privileged_feature: torch.Tensor) -> torch.Tensor:
        if privileged_feature.ndim != 2 or privileged_feature.shape[1] != PRIVILEGED_DIM:
            raise ValueError("B4 privileged feature must be [N,12]")
        return self.critic(privileged_feature).squeeze(-1)

    def actor_state(self) -> dict[str, torch.Tensor]:
        state = _clone_cpu_state(self.actor)
        if not _is_plain_actor_state(state):
            raise AssertionError("B4 actor state schema drift")
        return state


def project_raw_action(raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the owner-declared fixed physical actuator projection."""

    if raw_action.shape[-1] != ACTION_DIM or not torch.all(torch.isfinite(raw_action)):
        raise ValueError("B4 raw action is invalid")
    steer = torch.clamp(raw_action[..., 0:1], -0.52, 0.52)
    speed = torch.clamp(raw_action[..., 1:2], 0.0, 20.0)
    executed = torch.cat((steer, speed), dim=-1)
    return executed, executed - raw_action


def mean_bound_penalty(mean: torch.Tensor) -> torch.Tensor:
    if mean.ndim != 2 or mean.shape[1] != ACTION_DIM:
        raise ValueError("B4 mean bound input must be [N,2]")
    steer = torch.relu(-0.52 - mean[:, 0]) ** 2 + torch.relu(mean[:, 0] - 0.52) ** 2
    speed = torch.relu(-mean[:, 1]) ** 2 + torch.relu(mean[:, 1] - 20.0) ** 2
    return steer + speed


@dataclass
class B4Transition:
    l2_id: str
    episode_id: int
    step_index: int
    feature: np.ndarray
    privileged_feature: np.ndarray
    raw_action: np.ndarray
    executed_action: np.ndarray
    projection_delta: np.ndarray
    old_log_prob: float
    old_value: float
    reward: float = 0.0
    terminated: bool = False

    def validate(self) -> None:
        arrays = (
            (self.feature, (FEATURE_DIM,), "feature"),
            (self.privileged_feature, (PRIVILEGED_DIM,), "privileged feature"),
            (self.raw_action, (ACTION_DIM,), "raw action"),
            (self.executed_action, (ACTION_DIM,), "executed action"),
            (self.projection_delta, (ACTION_DIM,), "projection delta"),
        )
        for value, shape, name in arrays:
            array = np.asarray(value)
            if array.shape != shape or array.dtype != np.float32 or not np.all(np.isfinite(array)):
                raise ValueError(f"B4 transition {name} is invalid")
        if not self.l2_id.startswith("L2:") or self.episode_id < 0 or self.step_index < 0:
            raise ValueError("B4 transition identity is invalid")
        scalars = (self.old_log_prob, self.old_value, self.reward)
        if not all(math.isfinite(float(value)) for value in scalars):
            raise ValueError("B4 transition scalar is invalid")
        raw = torch.from_numpy(self.raw_action)
        expected, delta = project_raw_action(raw)
        if not torch.equal(expected, torch.from_numpy(self.executed_action)):
            raise ValueError("B4 stored executed action is not projection(raw)")
        if not torch.equal(delta, torch.from_numpy(self.projection_delta)):
            raise ValueError("B4 stored projection delta is inconsistent")


@dataclass(frozen=True)
class B4Batch:
    feature: torch.Tensor
    privileged_feature: torch.Tensor
    raw_action: torch.Tensor
    executed_action: torch.Tensor
    projection_delta: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    episode_id: torch.Tensor
    step_index: torch.Tensor
    advantage: torch.Tensor
    normalized_advantage: torch.Tensor
    returns: torch.Tensor
    actor_weight: torch.Tensor
    l2_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        count = int(self.feature.shape[0])
        if count <= 0:
            raise ValueError("B4 batch is empty")
        expected = {
            "feature": (count, FEATURE_DIM),
            "privileged_feature": (count, PRIVILEGED_DIM),
            "raw_action": (count, ACTION_DIM),
            "executed_action": (count, ACTION_DIM),
            "projection_delta": (count, ACTION_DIM),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"B4 batch {name} shape drift")
        vectors = (
            self.old_log_prob,
            self.old_value,
            self.reward,
            self.terminated,
            self.episode_id,
            self.step_index,
            self.advantage,
            self.normalized_advantage,
            self.returns,
            self.actor_weight,
        )
        if any(tuple(value.shape) != (count,) for value in vectors):
            raise ValueError("B4 batch vector shape drift")
        if len(self.l2_ids) != count:
            raise ValueError("B4 batch L2 inventory drift")

    @property
    def size(self) -> int:
        return int(self.feature.shape[0])

    def to(self, device: torch.device) -> "B4Batch":
        fields = {
            name: getattr(self, name).to(device)
            for name in (
                "feature",
                "privileged_feature",
                "raw_action",
                "executed_action",
                "projection_delta",
                "old_log_prob",
                "old_value",
                "reward",
                "terminated",
                "episode_id",
                "step_index",
                "advantage",
                "normalized_advantage",
                "returns",
                "actor_weight",
            )
        }
        return B4Batch(**fields, l2_ids=self.l2_ids)


def _episode_groups(transitions: Sequence[B4Transition]) -> list[list[B4Transition]]:
    if not transitions:
        raise ValueError("B4 rollout has no transitions")
    groups: list[list[B4Transition]] = []
    current: list[B4Transition] = []
    current_id: int | None = None
    seen: set[int] = set()
    for row in transitions:
        row.validate()
        if current_id is None or row.episode_id != current_id:
            if current:
                if not current[-1].terminated:
                    raise ValueError("B4 episode is incomplete")
                groups.append(current)
                seen.add(int(current_id))
            if row.episode_id in seen:
                raise ValueError("B4 episode transitions are non-contiguous")
            current = []
            current_id = row.episode_id
        if row.step_index != len(current):
            raise ValueError("B4 episode step index drift")
        if current and current[-1].terminated:
            raise ValueError("B4 transition appears after terminal")
        current.append(row)
    if not current or not current[-1].terminated:
        raise ValueError("B4 final episode is incomplete")
    groups.append(current)
    if any(any(row.terminated for row in episode[:-1]) for episode in groups):
        raise ValueError("B4 episode has an early terminal marker")
    if any(
        any(float(row.reward) != 0.0 for row in episode[:-1])
        for episode in groups
    ):
        raise ValueError("B4 dense/nonterminal reward leaked into replay")
    if any(float(episode[-1].reward) not in (-2.0, 0.0, 1.0) for episode in groups):
        raise ValueError("B4 terminal replay reward is outside -2*C+O support")
    return groups


def weighted_mean_variance(
    value: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 1 or value.shape != weight.shape:
        raise ValueError("B4 weighted statistics shape mismatch")
    if torch.any(weight <= 0.0) or not torch.all(torch.isfinite(weight)):
        raise ValueError("B4 actor weights are invalid")
    denominator = weight.sum()
    mean = (weight * value).sum() / denominator
    variance = (weight * (value - mean) ** 2).sum() / denominator
    return mean, variance


def build_batch(
    transitions: Sequence[B4Transition],
    config: B4Config = FROZEN_B4_CONFIG,
) -> B4Batch:
    """Compute terminal-zero-bootstrap GAE independently inside each episode."""

    episodes = _episode_groups(transitions)
    count = sum(len(episode) for episode in episodes)
    episode_count = len(episodes)
    if episode_count != config.episodes_per_iteration:
        raise ValueError(
            f"B4 rollout must contain {config.episodes_per_iteration} complete episodes"
        )
    advantages = np.zeros(count, dtype=np.float32)
    returns = np.zeros(count, dtype=np.float32)
    weights = np.zeros(count, dtype=np.float32)
    cursor = 0
    for episode in episodes:
        gae = 0.0
        next_value = 0.0
        for local_index in range(len(episode) - 1, -1, -1):
            row = episode[local_index]
            nonterminal = 0.0 if row.terminated else 1.0
            delta = float(row.reward) + config.gamma * nonterminal * next_value - float(row.old_value)
            gae = delta + config.gamma * config.gae_lambda * nonterminal * gae
            global_index = cursor + local_index
            advantages[global_index] = np.float32(gae)
            returns[global_index] = np.float32(gae + float(row.old_value))
            next_value = float(row.old_value)
        weights[cursor : cursor + len(episode)] = np.float32(
            (count / episode_count) / len(episode)
        )
        cursor += len(episode)
    advantage_t = torch.from_numpy(advantages)
    weight_t = torch.from_numpy(weights)
    weighted_mean, weighted_variance = weighted_mean_variance(advantage_t, weight_t)
    normalized = (advantage_t - weighted_mean) / torch.sqrt(weighted_variance + 1e-8)
    flat = [row for episode in episodes for row in episode]
    batch = B4Batch(
        feature=torch.from_numpy(np.stack([row.feature for row in flat])),
        privileged_feature=torch.from_numpy(
            np.stack([row.privileged_feature for row in flat])
        ),
        raw_action=torch.from_numpy(np.stack([row.raw_action for row in flat])),
        executed_action=torch.from_numpy(
            np.stack([row.executed_action for row in flat])
        ),
        projection_delta=torch.from_numpy(
            np.stack([row.projection_delta for row in flat])
        ),
        old_log_prob=torch.tensor([row.old_log_prob for row in flat], dtype=torch.float32),
        old_value=torch.tensor([row.old_value for row in flat], dtype=torch.float32),
        reward=torch.tensor([row.reward for row in flat], dtype=torch.float32),
        terminated=torch.tensor([row.terminated for row in flat], dtype=torch.bool),
        episode_id=torch.tensor([row.episode_id for row in flat], dtype=torch.int64),
        step_index=torch.tensor([row.step_index for row in flat], dtype=torch.int64),
        advantage=advantage_t,
        normalized_advantage=normalized,
        returns=torch.from_numpy(returns),
        actor_weight=weight_t,
        l2_ids=tuple(row.l2_id for row in flat),
    )
    if not torch.isclose(batch.actor_weight.sum(), torch.tensor(float(count)), atol=1e-4):
        raise AssertionError("B4 actor weights do not have global mean one")
    return batch


def replay_metrics(policy: B4DirectHeadPolicy, batch: B4Batch) -> dict[str, float]:
    with torch.no_grad():
        mean = policy.mean_from_feature(batch.feature)
        new_log_prob = policy.log_prob(mean, batch.raw_action)
        log_ratio = new_log_prob - batch.old_log_prob
        ratio = torch.exp(log_ratio)
        weighted_kl = (batch.actor_weight * (batch.old_log_prob - new_log_prob)).mean()
        unweighted_kl = (batch.old_log_prob - new_log_prob).mean()
        clipped = (torch.abs(ratio - 1.0) > policy.config.clip_eps).float()
        weighted_clip = (batch.actor_weight * clipped).mean()
    return {
        "max_abs_ratio_minus_one": float(torch.max(torch.abs(ratio - 1.0)).item()),
        "weighted_kl": float(weighted_kl.item()),
        "unweighted_kl": float(unweighted_kl.item()),
        "weighted_clip_fraction": float(weighted_clip.item()),
        "unweighted_clip_fraction": float(clipped.mean().item()),
    }


def projection_metrics(batch: B4Batch) -> dict[str, float | int]:
    delta = torch.abs(batch.projection_delta)
    changed = delta > 0.0
    return {
        "transition_count": batch.size,
        "projection_transition_count": int(torch.any(changed, dim=1).sum().item()),
        "steer_projection_count": int(changed[:, 0].sum().item()),
        "speed_projection_count": int(changed[:, 1].sum().item()),
        "mean_abs_steer_projection_delta": float(delta[:, 0].mean().item()),
        "mean_abs_speed_projection_delta": float(delta[:, 1].mean().item()),
        "max_abs_steer_projection_delta": float(delta[:, 0].max().item()),
        "max_abs_speed_projection_delta": float(delta[:, 1].max().item()),
    }


def build_optimizers(
    policy: B4DirectHeadPolicy,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    actor_parameters = policy.trainable_actor_parameters
    critic_parameters = tuple(policy.critic.parameters())
    if not actor_parameters or not critic_parameters:
        raise AssertionError("B4 optimizer parameter inventory is empty")
    if set(map(id, actor_parameters)) & set(map(id, critic_parameters)):
        raise AssertionError("B4 actor/critic optimizer parameters overlap")
    actor = torch.optim.Adam(actor_parameters, lr=policy.config.actor_lr)
    critic = torch.optim.Adam(critic_parameters, lr=policy.config.critic_lr)
    return actor, critic


def _permutation(size: int, seed: int, iteration: int, epoch: int, role: str) -> torch.Tensor:
    digest = hashlib.sha256(B4_MINIBATCH_DOMAIN)
    digest.update(f"{seed}:{iteration}:{epoch}:{role}:{size}".encode("ascii"))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1))
    return torch.randperm(size, generator=generator)


def _minibatches(order: torch.Tensor, size: int) -> Iterable[torch.Tensor]:
    for start in range(0, len(order), int(size)):
        yield order[start : start + int(size)]


def update_policy(
    policy: B4DirectHeadPolicy,
    batch: B4Batch,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    seed: int,
    iteration: int,
) -> dict[str, Any]:
    """Run separated PPO actor and critic loops over one frozen rollout."""

    config = policy.config
    device = next(policy.parameters()).device
    batch = batch.to(device)
    actor_parameter_ids = {
        id(parameter) for group in actor_optimizer.param_groups for parameter in group["params"]
    }
    critic_parameter_ids = {
        id(parameter) for group in critic_optimizer.param_groups for parameter in group["params"]
    }
    if actor_parameter_ids != set(map(id, policy.trainable_actor_parameters)):
        raise ValueError("B4 actor optimizer inventory drift")
    if critic_parameter_ids != set(map(id, policy.critic.parameters())):
        raise ValueError("B4 critic optimizer inventory drift")
    if actor_parameter_ids & critic_parameter_ids:
        raise ValueError("B4 actor/critic optimizer overlap")

    actor_epochs_completed = 0
    actor_steps = 0
    actor_stopped_early = False
    actor_losses: list[float] = []
    actor_epoch_metrics: list[dict[str, float | int]] = []
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    policy.actor.output_layer.train()
    actor_batch_denominator = float(min(config.minibatch_size, batch.size))
    for epoch in range(config.actor_epochs):
        order = _permutation(batch.size, seed, iteration, epoch, "actor")
        for cpu_indices in _minibatches(order, config.minibatch_size):
            indices = cpu_indices.to(device)
            mean = policy.mean_from_feature(batch.feature[indices])
            new_log_prob = policy.log_prob(mean, batch.raw_action[indices])
            ratio = torch.exp(new_log_prob - batch.old_log_prob[indices])
            advantage = batch.normalized_advantage[indices]
            unclipped = ratio * advantage
            clipped = torch.clamp(
                ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps
            ) * advantage
            surrogate = torch.minimum(unclipped, clipped)
            # Do not renormalize weights inside a minibatch. In particular, a
            # retained tail uses the same denominator as a full minibatch so it
            # cannot silently receive one full optimizer step of average weight.
            weighted_surrogate = (
                batch.actor_weight[indices] * surrogate
            ).sum() / actor_batch_denominator
            bound = (
                batch.actor_weight[indices] * mean_bound_penalty(mean)
            ).sum() / actor_batch_denominator
            loss = -weighted_surrogate + config.mean_bound_coef * bound
            actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None for parameter in policy.frozen_actor_parameters):
                raise AssertionError("B4 frozen actor received a gradient")
            if any(parameter.grad is not None for parameter in policy.critic.parameters()):
                raise AssertionError("B4 critic received an actor gradient")
            nn.utils.clip_grad_norm_(policy.trainable_actor_parameters, config.max_grad_norm)
            actor_optimizer.step()
            actor_steps += 1
            actor_losses.append(float(loss.detach().item()))
        actor_epochs_completed += 1
        metrics = replay_metrics(policy, batch)
        actor_epoch_metrics.append({"epoch": epoch + 1, **metrics})
        if metrics["weighted_kl"] > config.target_weighted_kl:
            actor_stopped_early = epoch + 1 < config.actor_epochs
            break

    # Actor early stop deliberately does not gate the critic loop.
    critic_losses: list[float] = []
    critic_steps = 0
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    policy.critic.train()
    for epoch in range(config.critic_epochs):
        order = _permutation(batch.size, seed, iteration, epoch, "critic")
        for cpu_indices in _minibatches(order, config.minibatch_size):
            indices = cpu_indices.to(device)
            predicted = policy.value(batch.privileged_feature[indices])
            loss = torch.mean((predicted - batch.returns[indices]) ** 2)
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None for parameter in policy.actor.parameters()):
                raise AssertionError("B4 actor received a critic gradient")
            nn.utils.clip_grad_norm_(tuple(policy.critic.parameters()), config.max_grad_norm)
            critic_optimizer.step()
            critic_steps += 1
            critic_losses.append(float(loss.detach().item()))

    policy.assert_frozen_exact()
    final_metrics = replay_metrics(policy, batch)
    expected_critic_steps = config.critic_epochs * math.ceil(batch.size / config.minibatch_size)
    if critic_steps != expected_critic_steps:
        raise AssertionError("B4 critic did not complete every frozen epoch")
    return {
        "actor_epochs_completed": actor_epochs_completed,
        "actor_stopped_early": actor_stopped_early,
        "actor_optimizer_steps": actor_steps,
        "actor_epoch_metrics": actor_epoch_metrics,
        "critic_epochs_completed": config.critic_epochs,
        "critic_optimizer_steps": critic_steps,
        "actor_loss_mean": float(np.mean(actor_losses)),
        "critic_loss_mean": float(np.mean(critic_losses)),
        **final_metrics,
    }


def _curriculum_order(
    rows: Sequence[B2Scenario], seed: int, group: str, repeat: int
) -> tuple[B2Scenario, ...]:
    def key(row: B2Scenario) -> tuple[bytes, str]:
        digest = hashlib.sha256(B4_CURRICULUM_DOMAIN)
        digest.update(f"{int(seed)}:{group}:{int(repeat)}:".encode("ascii"))
        digest.update(row.l2_id.encode("ascii"))
        return digest.digest(), row.l2_id

    return tuple(sorted(rows, key=key))


@dataclass(frozen=True)
class B4ScenarioSets:
    collision: tuple[B2Scenario, ...]
    overtake: tuple[B2Scenario, ...]
    follow: tuple[B2Scenario, ...]
    development_rows: tuple[dict[str, str], ...]

    @classmethod
    def from_b2(cls, scenarios: B2ScenarioSets) -> "B4ScenarioSets":
        result = cls(
            collision=tuple(scenarios.collision),
            overtake=tuple(
                row for row in scenarios.remaining if row.archived_bc_outcome == "overtake"
            ),
            follow=tuple(
                row for row in scenarios.remaining if row.archived_bc_outcome == "follow"
            ),
            development_rows=tuple(scenarios.development_rows),
        )
        if (len(result.collision), len(result.overtake), len(result.follow)) != (
            COLLISION_POOL_ROWS,
            OVERTAKE_POOL_ROWS,
            FOLLOW_POOL_ROWS,
        ):
            raise ValueError("B4 training outcome pool count drift")
        training_l2 = {row.l2_id for group in (result.collision, result.overtake, result.follow) for row in group}
        development_l2 = {row["l2_id"] for row in result.development_rows}
        if len(training_l2) != TRAINING_ROWS or training_l2 & development_l2:
            raise ValueError("B4 training/development L2 separation drift")
        return result


class B4Curriculum:
    def __init__(self, scenarios: B4ScenarioSets, seed: int):
        if int(seed) not in FROZEN_B4_CONFIG.seeds:
            raise ValueError("B4 curriculum seed must be the owner-selected seed 1")
        self.scenarios = scenarios
        self.seed = int(seed)

    def _take(self, rows: Sequence[B2Scenario], group: str, count: int) -> list[B2Scenario]:
        output: list[B2Scenario] = []
        repeat = 0
        while len(output) < count:
            ordered = _curriculum_order(rows, self.seed, group, repeat)
            need = count - len(output)
            output.extend(ordered[:need])
            repeat += 1
        return output

    def plan(self, iterations: int = 30) -> tuple[tuple[B2Scenario, ...], ...]:
        if int(iterations) != iterations or int(iterations) <= 0:
            raise ValueError("B4 curriculum iterations must be positive")
        groups = {
            "collision": self._take(
                self.scenarios.collision, "collision", int(iterations) * COLLISION_EPISODES
            ),
            "overtake": self._take(
                self.scenarios.overtake, "overtake", int(iterations) * OVERTAKE_EPISODES
            ),
            "follow": self._take(
                self.scenarios.follow, "follow", int(iterations) * FOLLOW_EPISODES
            ),
        }
        offsets = {name: 0 for name in groups}
        result: list[tuple[B2Scenario, ...]] = []
        for _ in range(int(iterations)):
            rows: list[B2Scenario] = []
            for group in CURRICULUM_PATTERN:
                rows.append(groups[group][offsets[group]])
                offsets[group] += 1
            if tuple(row.archived_bc_outcome for row in rows).count("collision") != COLLISION_EPISODES:
                raise AssertionError("B4 collision curriculum count drift")
            result.append(tuple(rows))
        return tuple(result)

    def digest(self, iterations: int = 30) -> str:
        digest = hashlib.sha256(B4_CURRICULUM_DOMAIN)
        digest.update(B4_CURRICULUM_SCHEMA.encode("ascii") + b"\0")
        digest.update(f"{self.seed}:{int(iterations)}\0".encode("ascii"))
        for iteration, rows in enumerate(self.plan(iterations), start=1):
            for episode_index, row in enumerate(rows):
                digest.update(f"{iteration}:{episode_index}:".encode("ascii"))
                digest.update(row.l2_id.encode("ascii") + b"\n")
        return digest.hexdigest()


def actor_snapshot_sha256(state: Mapping[str, torch.Tensor]) -> str:
    if not _is_plain_actor_state(state):
        raise ValueError("B4 actor snapshot hash requires canonical plain state")
    digest = hashlib.sha256(b"end2race:b4:plain-actor-state:v1\0")
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(path if path.exists() else temporary)
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_actor_snapshot(policy: B4DirectHeadPolicy, path: str | Path) -> dict[str, Any]:
    policy.assert_frozen_exact()
    state = policy.actor_state()
    destination = Path(path)
    _atomic_torch_save(state, destination)
    loaded = load_strict_plain_actor(destination)
    observed = loaded.state_dict()
    if any(not torch.equal(observed[name].cpu(), state[name]) for name in state):
        raise AssertionError("B4 actor snapshot strict roundtrip mismatch")
    return {
        "schema": B4_ACTOR_SNAPSHOT_SCHEMA,
        "path": str(destination),
        "tensor_sha256": actor_snapshot_sha256(state),
        "key_count": len(state),
    }


def _rng_state() -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["torch_cuda"] = torch.cuda.get_rng_state_all()
    return value


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu"}
    if not required.issubset(value):
        raise ValueError("B4 resume RNG state is incomplete")
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if "torch_cuda" in value:
        if not torch.cuda.is_available():
            raise ValueError("B4 checkpoint contains CUDA RNG but CUDA is unavailable")
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def save_full_checkpoint(
    policy: B4DirectHeadPolicy,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    path: str | Path,
    *,
    completed_iteration: int,
    seed: int,
    run_plan_sha256: str,
    curriculum_sha256: str,
) -> dict[str, Any]:
    policy.assert_frozen_exact()
    if completed_iteration < 0 or completed_iteration > policy.config.iterations:
        raise ValueError("B4 checkpoint iteration is invalid")
    if seed not in policy.config.seeds:
        raise ValueError("B4 checkpoint seed is invalid")
    for name, digest in (
        ("run_plan_sha256", run_plan_sha256),
        ("curriculum_sha256", curriculum_sha256),
    ):
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"B4 {name} is invalid")
    payload = {
        "schema": B4_FULL_CHECKPOINT_SCHEMA,
        "policy_schema": B4_POLICY_SCHEMA,
        "completed_iteration": int(completed_iteration),
        "seed": int(seed),
        "run_plan_sha256": run_plan_sha256,
        "curriculum_sha256": curriculum_sha256,
        "config": policy.config.as_dict(),
        "actor_state_dict": policy.actor_state(),
        "critic_state_dict": _clone_cpu_state(policy.critic),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "initial_frozen_state": {
            name: value.clone() for name, value in policy._initial_frozen.items()
        },
        "rng_state": _rng_state(),
    }
    destination = Path(path)
    _atomic_torch_save(payload, destination)
    return payload


def load_full_checkpoint(
    path: str | Path,
    policy: B4DirectHeadPolicy,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    expected_seed: int,
    expected_run_plan_sha256: str,
    expected_curriculum_sha256: str,
    restore_rng: bool = True,
) -> int:
    # Full resume checkpoints are private, hash-bound RunPlan artifacts and
    # include Python/NumPy RNG tuples. PyTorch 2.6 therefore requires the
    # trusted-source opt-out from its state-dict-only default.
    payload = torch.load(
        Path(path),
        # Keep RNG byte tensors on CPU. Module ``load_state_dict`` copies to
        # the live device, and optimizer loading casts state to each parameter;
        # mapping the whole payload to CUDA would make ``torch.set_rng_state``
        # receive an invalid CUDA tensor during resume.
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping) or payload.get("schema") != B4_FULL_CHECKPOINT_SCHEMA:
        raise ValueError("B4 full checkpoint schema mismatch")
    checks = {
        "policy_schema": B4_POLICY_SCHEMA,
        "seed": int(expected_seed),
        "run_plan_sha256": expected_run_plan_sha256,
        "curriculum_sha256": expected_curriculum_sha256,
        "config": policy.config.as_dict(),
    }
    if any(payload.get(name) != value for name, value in checks.items()):
        raise ValueError("B4 full checkpoint identity/config mismatch")
    actor_state = payload.get("actor_state_dict")
    strict_plain_actor_from_state(actor_state)
    policy.actor.load_state_dict(actor_state, strict=True)
    policy.critic.load_state_dict(payload["critic_state_dict"], strict=True)
    actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
    critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
    initial_frozen = payload.get("initial_frozen_state")
    if not isinstance(initial_frozen, Mapping) or initial_frozen.keys() != policy._initial_frozen.keys():
        raise ValueError("B4 full checkpoint frozen inventory mismatch")
    if any(
        not torch.equal(initial_frozen[name].cpu(), policy._initial_frozen[name])
        for name in initial_frozen
    ):
        raise ValueError("B4 full checkpoint was initialized from different frozen tensors")
    policy.assert_frozen_exact()
    if restore_rng:
        _restore_rng_state(payload["rng_state"])
    completed = payload.get("completed_iteration")
    if type(completed) is not int or not 0 <= completed <= policy.config.iterations:
        raise ValueError("B4 full checkpoint completed iteration is invalid")
    return completed
