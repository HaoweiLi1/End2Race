"""B7 plain-End2Race recurrent PPO engineering contracts.

B7 keeps the deployment actor byte-schema identical to the canonical
``End2Race`` model while training the original GRU and output layer.  Rollout
actions remain fixed-variance iid Gaussian latents at 100 Hz.  The module is
simulator independent: collection lives in :mod:`bplus_v22.b7_env`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from bplus_v22.b4_direct import (
    ACTION_DIM,
    CANONICAL_ACTOR_KEYS,
    actor_snapshot_sha256,
    mean_bound_penalty,
    project_raw_action,
    strict_plain_actor_from_state,
)
from bplus_v22.ppo_env import B2Scenario


B7_POLICY_SCHEMA = "end2race-b7-plain-recurrent-policy-1"
B7_FULL_CHECKPOINT_SCHEMA = "end2race-b7-plain-recurrent-full-checkpoint-1"
B7_SAMPLER_SCHEMA = "end2race-b7-current-hard-sampler-1"
B7_SAMPLER_DOMAIN = b"end2race:b7:current-hard-sampler:v1\0"
B7_CRITIC_ORDER_DOMAIN = b"end2race:b7:critic-order:v1\0"

LIDAR_DIM = 360
PRIVILEGED_DIM = 13
ACTION_STD = (0.03, 0.20)
EPISODES_PER_ITERATION = 32
REPRESENTATIVE_EPISODES = 16
ARCHIVED_COLLISION_EPISODES = 8
CURRENT_HARD_EPISODES = 8
PRODUCT_HORIZON_SECONDS = 8.0


@dataclass(frozen=True)
class B7Config:
    head_lr: float = 1e-5
    gru_lr: float = 1e-6
    critic_lr: float = 3e-4
    steer_std: float = ACTION_STD[0]
    speed_std: float = ACTION_STD[1]
    clip_eps: float = 0.10
    safe_kl_cap: float = 0.01
    rollout_kl_cap: float = 0.015
    actor_epochs: int = 1
    actor_optimizer_steps: int = 1
    critic_epochs: int = 3
    critic_minibatch_size: int = 4096
    gamma: float = 0.999
    gae_lambda: float = 0.997
    entropy_coef: float = 0.0
    mean_bound_coef: float = 0.01
    max_grad_norm: float = 0.5
    collision_window_steps: int = 100
    episodes_per_iteration: int = EPISODES_PER_ITERATION
    representative_episodes: int = REPRESENTATIVE_EPISODES
    archived_collision_episodes: int = ARCHIVED_COLLISION_EPISODES
    current_hard_episodes: int = CURRENT_HARD_EPISODES
    iterations: int = 10
    primary_seed: int = 1
    candidate_iteration: int = 10
    max_consecutive_rejections: int = 3

    def __post_init__(self) -> None:
        positive = (
            self.head_lr,
            self.gru_lr,
            self.critic_lr,
            self.steer_std,
            self.speed_std,
            self.clip_eps,
            self.safe_kl_cap,
            self.rollout_kl_cap,
            self.gamma,
            self.gae_lambda,
            self.max_grad_norm,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("B7 positive configuration field is invalid")
        if self.entropy_coef != 0.0 or self.actor_epochs != 1 or self.actor_optimizer_steps != 1:
            raise ValueError("B7 requires one entropy-free recurrent actor step")
        if self.episodes_per_iteration != sum(
            (
                self.representative_episodes,
                self.archived_collision_episodes,
                self.current_hard_episodes,
            )
        ):
            raise ValueError("B7 sampler counts do not sum to one rollout")
        integers = (
            self.critic_epochs,
            self.critic_minibatch_size,
            self.collision_window_steps,
            self.episodes_per_iteration,
            self.iterations,
            self.candidate_iteration,
            self.max_consecutive_rejections,
        )
        if any(int(value) != value or int(value) <= 0 for value in integers):
            raise ValueError("B7 integer configuration field is invalid")
        if self.primary_seed != 1 or self.iterations != 10 or self.candidate_iteration != 10:
            raise ValueError("B7 primary seed/iteration contract drift")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("B7 gamma/lambda is invalid")
        if not math.isfinite(self.mean_bound_coef) or self.mean_bound_coef < 0.0:
            raise ValueError("B7 mean-bound coefficient is invalid")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


FROZEN_B7_CONFIG = B7Config()


class B7RecurrentPolicy(nn.Module):
    """Training-only wrapper; deployment saves ``actor.state_dict()`` only."""

    def __init__(self, bc_state: Mapping[str, torch.Tensor], config: B7Config = FROZEN_B7_CONFIG):
        super().__init__()
        self.config = config
        self.actor = strict_plain_actor_from_state(bc_state)
        self.bc_reference = strict_plain_actor_from_state(bc_state)
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
        for name, parameter in self.actor.named_parameters():
            parameter.requires_grad_(name.startswith("gru.") or name.startswith("output_layer."))
        for parameter in self.bc_reference.parameters():
            parameter.requires_grad_(False)
        self.bc_reference.eval()
        self._initial_frozen = self.frozen_state()
        self._initial_bc_reference = {
            name: value.detach().cpu().clone()
            for name, value in self.bc_reference.state_dict().items()
        }
        expected = {
            "gru.weight_ih_l0",
            "gru.weight_hh_l0",
            "gru.bias_ih_l0",
            "gru.bias_hh_l0",
            "output_layer.0.weight",
            "output_layer.0.bias",
            "output_layer.2.weight",
            "output_layer.2.bias",
        }
        observed = {name for name, parameter in self.actor.named_parameters() if parameter.requires_grad}
        if observed != expected:
            raise AssertionError(f"B7 trainable actor inventory drift: {sorted(observed)}")

    @property
    def gru_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.actor.gru.parameters())

    @property
    def head_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.actor.output_layer.parameters())

    @property
    def trainable_actor_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.gru_parameters + self.head_parameters

    @property
    def frozen_actor_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for parameter in self.actor.parameters() if not parameter.requires_grad)

    def frozen_state(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.actor.state_dict().items()
            if not name.startswith("gru.") and not name.startswith("output_layer.")
        }

    def assert_frozen_exact(self) -> None:
        current = self.frozen_state()
        if current.keys() != self._initial_frozen.keys() or any(
            not torch.equal(current[name], self._initial_frozen[name]) for name in current
        ):
            raise AssertionError("B7 frozen input encoder tensor changed")
        reference = self.bc_reference.state_dict()
        if tuple(reference) != CANONICAL_ACTOR_KEYS or any(
            not torch.equal(value.detach().cpu(), self._initial_bc_reference[name])
            for name, value in reference.items()
        ):
            raise AssertionError("B7 canonical BC reference tensor drift")

    def zero_hidden(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            (1, 1, self.actor.gru.hidden_size),
            dtype=next(self.actor.parameters()).dtype,
            device=device,
        )

    def rollout_step(
        self,
        lidar: torch.Tensor,
        previous_speed: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action, next_hidden = self.actor(lidar, previous_speed, hidden)
        mean = action[:, -1, :]
        if mean.shape != (1, ACTION_DIM):
            raise AssertionError("B7 rollout actor shape drift")
        return mean, next_hidden

    def bc_rollout_step(
        self,
        lidar: torch.Tensor,
        previous_speed: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action, next_hidden = self.bc_reference(lidar, previous_speed, hidden)
        return action[:, -1, :], next_hidden

    def sequence_means(
        self,
        lidar: torch.Tensor,
        previous_speed: torch.Tensor,
    ) -> torch.Tensor:
        """Replay one episode with deployment-equivalent framewise GRU calls."""

        if lidar.ndim != 2 or lidar.shape[1] != LIDAR_DIM:
            raise ValueError("B7 sequence lidar must be [T,360]")
        if previous_speed.ndim != 1 or len(previous_speed) != len(lidar):
            raise ValueError("B7 previous-speed sequence shape drift")
        hidden = self.zero_hidden(lidar.device)
        output: list[torch.Tensor] = []
        for step in range(len(lidar)):
            mean, hidden = self.rollout_step(
                lidar[step].reshape(1, 1, LIDAR_DIM),
                previous_speed[step].reshape(1, 1, 1),
                hidden,
            )
            output.append(mean[0])
        return torch.stack(output)

    def log_prob(self, mean: torch.Tensor, raw_action: torch.Tensor) -> torch.Tensor:
        if mean.shape != raw_action.shape or mean.shape[-1] != ACTION_DIM:
            raise ValueError("B7 mean/raw action shape mismatch")
        std = self.action_std.to(dtype=mean.dtype, device=mean.device)
        value = -0.5 * (((raw_action - mean) / std) ** 2 + math.log(2.0 * math.pi))
        value = value - torch.log(std)
        return value.sum(dim=-1)

    def sample_raw(self, mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = mean + torch.randn_like(mean) * self.action_std.to(mean)
        return raw, self.log_prob(mean, raw)

    def value(self, privileged_feature: torch.Tensor) -> torch.Tensor:
        if privileged_feature.ndim != 2 or privileged_feature.shape[1] != PRIVILEGED_DIM:
            raise ValueError("B7 privileged critic feature must be [N,13]")
        return self.critic(privileged_feature).squeeze(-1)

    def actor_state(self) -> dict[str, torch.Tensor]:
        state = {
            name: value.detach().cpu().clone() for name, value in self.actor.state_dict().items()
        }
        if tuple(state) != CANONICAL_ACTOR_KEYS:
            raise AssertionError("B7 deployment actor is not canonical plain End2Race")
        return state


def build_optimizers(
    policy: B7RecurrentPolicy,
) -> tuple[torch.optim.Adam, torch.optim.Adam]:
    actor = torch.optim.Adam(
        [
            {"params": policy.gru_parameters, "lr": policy.config.gru_lr, "role": "gru"},
            {"params": policy.head_parameters, "lr": policy.config.head_lr, "role": "head"},
        ]
    )
    critic = torch.optim.Adam(policy.critic.parameters(), lr=policy.config.critic_lr)
    return actor, critic


def collision_reward_schedule(length: int, config: B7Config = FROZEN_B7_CONFIG) -> np.ndarray:
    """Spread collision cost over the last second with start-window equivalence."""

    if int(length) != length or length <= 0:
        raise ValueError("B7 collision reward requires a positive episode length")
    horizon = min(int(config.collision_window_steps), int(length))
    normalizer = sum(config.gamma ** (-offset) for offset in range(horizon))
    rewards = np.zeros(int(length), dtype=np.float32)
    rewards[-horizon:] = np.float32(-2.0 / normalizer)
    return rewards


def task_reward_schedule(
    length: int,
    *,
    collision_any: bool,
    terminal_overtake: bool,
    config: B7Config = FROZEN_B7_CONFIG,
) -> np.ndarray:
    if collision_any:
        return collision_reward_schedule(length, config)
    rewards = np.zeros(int(length), dtype=np.float32)
    if terminal_overtake:
        rewards[-1] = np.float32(1.0)
    return rewards


@dataclass
class B7Transition:
    step_index: int
    lidar: np.ndarray
    previous_speed: float
    privileged_feature: np.ndarray
    old_mean: np.ndarray
    bc_mean: np.ndarray
    raw_action: np.ndarray
    executed_action: np.ndarray
    projection_delta: np.ndarray
    old_log_prob: float
    old_value: float
    reward: float = 0.0
    terminated: bool = False

    def validate(self) -> None:
        arrays = (
            (self.lidar, (LIDAR_DIM,), "lidar"),
            (self.privileged_feature, (PRIVILEGED_DIM,), "privileged feature"),
            (self.old_mean, (ACTION_DIM,), "old mean"),
            (self.bc_mean, (ACTION_DIM,), "BC mean"),
            (self.raw_action, (ACTION_DIM,), "raw action"),
            (self.executed_action, (ACTION_DIM,), "executed action"),
            (self.projection_delta, (ACTION_DIM,), "projection delta"),
        )
        for value, shape, label in arrays:
            array = np.asarray(value)
            if array.shape != shape or array.dtype != np.float32 or not np.all(np.isfinite(array)):
                raise ValueError(f"B7 transition {label} is invalid")
        if self.step_index < 0 or not all(
            math.isfinite(float(value))
            for value in (self.previous_speed, self.old_log_prob, self.old_value, self.reward)
        ):
            raise ValueError("B7 transition scalar is invalid")
        expected, delta = project_raw_action(torch.from_numpy(self.raw_action))
        if not torch.equal(expected, torch.from_numpy(self.executed_action)):
            raise ValueError("B7 stored command is not projection(raw latent)")
        if not torch.equal(delta, torch.from_numpy(self.projection_delta)):
            raise ValueError("B7 projection delta ledger drift")


@dataclass(frozen=True)
class B7Episode:
    scenario: B2Scenario
    episode_id: int
    sampler_role: str
    hard_priority: int | None
    transitions: tuple[B7Transition, ...]
    collision_any: bool
    terminal_overtake: bool
    corrected_outcome3: str
    terminal_reason: str

    def __post_init__(self) -> None:
        if not self.transitions or self.episode_id < 0:
            raise ValueError("B7 episode identity/transition inventory is invalid")
        if self.sampler_role not in {"representative", "archived_collision", "current_hard"}:
            raise ValueError("B7 sampler role is invalid")
        if self.terminal_reason not in {"any_agent_collision", "product_horizon"}:
            raise ValueError("B7 terminal reason is invalid")
        if self.corrected_outcome3 not in {"collision", "follow", "overtake"}:
            raise ValueError("B7 corrected outcome is invalid")
        if self.collision_any != (self.corrected_outcome3 == "collision"):
            raise ValueError("B7 collision/outcome disagreement")
        for index, row in enumerate(self.transitions):
            row.validate()
            if row.step_index != index or row.terminated != (index == len(self.transitions) - 1):
                raise ValueError("B7 episode boundary/index drift")
        expected = task_reward_schedule(
            len(self.transitions),
            collision_any=self.collision_any,
            terminal_overtake=self.terminal_overtake,
        )
        observed = np.asarray([row.reward for row in self.transitions], dtype=np.float32)
        if not np.array_equal(expected, observed):
            raise ValueError("B7 task reward ledger drift")


@dataclass(frozen=True)
class B7PreparedEpisode:
    episode: B7Episode
    lidar: torch.Tensor
    previous_speed: torch.Tensor
    privileged_feature: torch.Tensor
    old_mean: torch.Tensor
    bc_mean: torch.Tensor
    raw_action: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    reward: torch.Tensor
    advantage: torch.Tensor
    normalized_advantage: torch.Tensor
    returns: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.lidar.shape[0])


@dataclass(frozen=True)
class B7Batch:
    episodes: tuple[B7PreparedEpisode, ...]

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def total_steps(self) -> int:
        return sum(episode.length for episode in self.episodes)


def build_batch(
    episodes: Sequence[B7Episode], config: B7Config = FROZEN_B7_CONFIG
) -> B7Batch:
    if len(episodes) != config.episodes_per_iteration:
        raise ValueError(f"B7 rollout requires {config.episodes_per_iteration} complete episodes")
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("B7 episode IDs are duplicated")
    if len({episode.scenario.l2_id for episode in episodes}) != len(episodes):
        raise ValueError("B7 iteration repeats an L2 scenario")
    raw_prepared: list[dict[str, Any]] = []
    all_advantages: list[np.ndarray] = []
    for episode in episodes:
        rows = episode.transitions
        advantages = np.zeros(len(rows), dtype=np.float64)
        returns = np.zeros(len(rows), dtype=np.float64)
        gae = 0.0
        next_value = 0.0
        for index in range(len(rows) - 1, -1, -1):
            row = rows[index]
            nonterminal = 0.0 if row.terminated else 1.0
            delta = float(row.reward) + config.gamma * nonterminal * next_value - float(row.old_value)
            gae = delta + config.gamma * config.gae_lambda * nonterminal * gae
            advantages[index] = gae
            returns[index] = gae + float(row.old_value)
            next_value = float(row.old_value)
        all_advantages.append(advantages)
        raw_prepared.append({"episode": episode, "advantages": advantages, "returns": returns})
    mean = float(np.mean([np.mean(value) for value in all_advantages]))
    variance = float(
        np.mean([np.mean((value - mean) ** 2) for value in all_advantages])
    )
    prepared: list[B7PreparedEpisode] = []
    for item in raw_prepared:
        episode = item["episode"]
        rows = episode.transitions
        advantage = item["advantages"].astype(np.float32)
        normalized = ((item["advantages"] - mean) / math.sqrt(variance + 1e-8)).astype(np.float32)
        prepared.append(
            B7PreparedEpisode(
                episode=episode,
                lidar=torch.from_numpy(np.stack([row.lidar for row in rows])),
                previous_speed=torch.tensor([row.previous_speed for row in rows], dtype=torch.float32),
                privileged_feature=torch.from_numpy(
                    np.stack([row.privileged_feature for row in rows])
                ),
                old_mean=torch.from_numpy(np.stack([row.old_mean for row in rows])),
                bc_mean=torch.from_numpy(np.stack([row.bc_mean for row in rows])),
                raw_action=torch.from_numpy(np.stack([row.raw_action for row in rows])),
                old_log_prob=torch.tensor([row.old_log_prob for row in rows], dtype=torch.float32),
                old_value=torch.tensor([row.old_value for row in rows], dtype=torch.float32),
                reward=torch.tensor([row.reward for row in rows], dtype=torch.float32),
                advantage=torch.from_numpy(advantage),
                normalized_advantage=torch.from_numpy(normalized),
                returns=torch.from_numpy(item["returns"].astype(np.float32)),
            )
        )
    return B7Batch(tuple(prepared))


def _hash_order(
    rows: Sequence[B2Scenario], seed: int, iteration: int, role: str
) -> tuple[B2Scenario, ...]:
    def key(row: B2Scenario) -> tuple[bytes, str]:
        digest = hashlib.sha256(B7_SAMPLER_DOMAIN)
        digest.update(f"{seed}:{iteration}:{role}:".encode("ascii"))
        digest.update(row.l2_id.encode("ascii"))
        return digest.digest(), row.l2_id

    return tuple(sorted(rows, key=key))


@dataclass(frozen=True)
class B7Selection:
    scenario: B2Scenario
    role: str
    hard_priority: int | None = None


def hard_priority(archived_bc_outcome: str, current_outcome: str) -> int | None:
    if archived_bc_outcome in {"follow", "overtake"} and current_outcome == "collision":
        return 1
    if archived_bc_outcome == "overtake" and current_outcome in {"follow", "collision"}:
        return 2
    if archived_bc_outcome == "collision" and current_outcome == "collision":
        return 3
    if archived_bc_outcome == "collision" and current_outcome == "follow":
        return 4
    return None


class B7ScenarioSampler:
    """Prospective map-balanced representative plus one-step hard mining."""

    def __init__(self, scenarios: Sequence[B2Scenario], seed: int):
        self.rows = tuple(sorted(scenarios, key=lambda row: row.training_order))
        self.seed = int(seed)
        if len(self.rows) != 1640 or len({row.l2_id for row in self.rows}) != 1640:
            raise ValueError("B7 sampler requires the complete 1,640-row training population")
        self.by_l2 = {row.l2_id: row for row in self.rows}
        self.maps = tuple(sorted({row.map_name for row in self.rows}))
        if len(self.maps) != 4:
            raise ValueError("B7 sampler requires four training maps")

    def _map_balanced(
        self,
        rows: Sequence[B2Scenario],
        count: int,
        iteration: int,
        role: str,
        excluded: set[str],
    ) -> list[B2Scenario]:
        if count % len(self.maps):
            raise ValueError("B7 map-balanced count is not divisible by map count")
        per_map = count // len(self.maps)
        selected: list[B2Scenario] = []
        for map_name in self.maps:
            available = [
                row for row in rows if row.map_name == map_name and row.l2_id not in excluded
            ]
            ordered = _hash_order(available, self.seed, iteration, f"{role}:{map_name}")
            if len(ordered) < per_map:
                raise ValueError(f"B7 sampler lacks {role} rows on {map_name}")
            selected.extend(ordered[:per_map])
            excluded.update(row.l2_id for row in ordered[:per_map])
        return selected

    def select(
        self,
        iteration: int,
        previous_outcomes: Mapping[str, str] | None,
    ) -> tuple[B7Selection, ...]:
        if not 1 <= int(iteration) <= FROZEN_B7_CONFIG.iterations:
            raise ValueError("B7 sampler iteration is invalid")
        excluded: set[str] = set()
        representative = self._map_balanced(
            self.rows,
            REPRESENTATIVE_EPISODES,
            int(iteration),
            "representative",
            excluded,
        )
        collision = self._map_balanced(
            [row for row in self.rows if row.archived_bc_outcome == "collision"],
            ARCHIVED_COLLISION_EPISODES,
            int(iteration),
            "archived_collision",
            excluded,
        )
        hard_selected: list[tuple[B2Scenario, int]] = []
        if previous_outcomes:
            candidates: list[tuple[int, B2Scenario]] = []
            for l2_id, outcome in previous_outcomes.items():
                row = self.by_l2.get(l2_id)
                if row is None or outcome not in {"collision", "follow", "overtake"}:
                    raise ValueError("B7 previous hard-mining outcome inventory drift")
                priority = hard_priority(row.archived_bc_outcome, outcome)
                if priority is not None and row.l2_id not in excluded:
                    candidates.append((priority, row))
            candidates.sort(
                key=lambda item: (
                    item[0],
                    _hash_order([item[1]], self.seed, int(iteration), f"hard:{item[0]}")[0].l2_id,
                )
            )
            # Re-sort ties with the actual domain-separated digest.
            ordered_candidates: list[tuple[int, B2Scenario]] = []
            for priority in (1, 2, 3, 4):
                tied = [row for value, row in candidates if value == priority]
                ordered_candidates.extend(
                    (priority, row)
                    for row in _hash_order(tied, self.seed, int(iteration), f"hard:{priority}")
                )
            for priority, row in ordered_candidates[:CURRENT_HARD_EPISODES]:
                hard_selected.append((row, priority))
                excluded.add(row.l2_id)
        missing = CURRENT_HARD_EPISODES - len(hard_selected)
        if missing:
            preservation = [
                row
                for row in self.rows
                if row.archived_bc_outcome == "overtake" and row.l2_id not in excluded
            ]
            ordered = _hash_order(
                preservation, self.seed, int(iteration), "hard_fill_bc_overtake"
            )
            if len(ordered) < missing:
                raise ValueError("B7 sampler lacks BC-overtake preservation fillers")
            for row in ordered[:missing]:
                hard_selected.append((row, 0))
                excluded.add(row.l2_id)
        selections = (
            [B7Selection(row, "representative") for row in representative]
            + [B7Selection(row, "archived_collision") for row in collision]
            + [B7Selection(row, "current_hard", priority) for row, priority in hard_selected]
        )
        if len(selections) != EPISODES_PER_ITERATION or len(excluded) != EPISODES_PER_ITERATION:
            raise AssertionError("B7 sampler output size/uniqueness drift")
        return tuple(selections)


def _copy_module_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _restore_module_state(module: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    module.load_state_dict(dict(state), strict=True)


def _candidate_means(
    policy: B7RecurrentPolicy, batch: B7Batch, device: torch.device
) -> list[torch.Tensor]:
    policy.actor.eval()
    output: list[torch.Tensor] = []
    with torch.no_grad():
        for episode in batch.episodes:
            output.append(
                policy.sequence_means(
                    episode.lidar.to(device), episode.previous_speed.to(device)
                ).detach()
            )
    return output


def preupdate_replay_metrics(
    policy: B7RecurrentPolicy, batch: B7Batch, device: torch.device
) -> dict[str, float]:
    means = _candidate_means(policy, batch, device)
    max_ratio = 0.0
    max_mean = 0.0
    for episode, mean in zip(batch.episodes, means, strict=True):
        raw = episode.raw_action.to(device)
        old_log_prob = episode.old_log_prob.to(device)
        ratio = torch.exp(policy.log_prob(mean, raw) - old_log_prob)
        max_ratio = max(max_ratio, float(torch.max(torch.abs(ratio - 1.0)).item()))
        max_mean = max(
            max_mean,
            float(torch.max(torch.abs(mean - episode.old_mean.to(device))).item()),
        )
    return {"max_abs_ratio_minus_one": max_ratio, "max_abs_mean_delta": max_mean}


def _mean_kl_summary(values: Sequence[torch.Tensor]) -> dict[str, float | int]:
    if not values:
        raise ValueError("B7 KL summary has no episodes")
    episode_means = torch.stack([value.double().mean() for value in values])
    frames = torch.cat([value.double().reshape(-1) for value in values])
    return {
        "episode_count": len(values),
        "frame_count": int(frames.numel()),
        "mean": float(episode_means.mean().item()),
        "episode_p50": float(torch.quantile(episode_means, 0.50).item()),
        "episode_p95": float(torch.quantile(episode_means, 0.95).item()),
        "frame_p95": float(torch.quantile(frames, 0.95).item()),
        "frame_max": float(frames.max().item()),
    }


def actor_kl_metrics(
    policy: B7RecurrentPolicy,
    batch: B7Batch,
    candidate_means: Sequence[torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    std = policy.action_std.to(device)
    rollout_values: list[torch.Tensor] = []
    safe_values: list[torch.Tensor] = []
    for episode, mean in zip(batch.episodes, candidate_means, strict=True):
        old = episode.old_mean.to(device)
        rollout_values.append(0.5 * torch.sum(((mean - old) / std) ** 2, dim=-1))
        if episode.episode.scenario.archived_bc_outcome in {"follow", "overtake"}:
            bc = episode.bc_mean.to(device)
            safe_values.append(0.5 * torch.sum(((mean - bc) / std) ** 2, dim=-1))
    return _mean_kl_summary(rollout_values), _mean_kl_summary(safe_values)


def _critic_order(size: int, seed: int, iteration: int, epoch: int) -> torch.Tensor:
    digest = hashlib.sha256(B7_CRITIC_ORDER_DOMAIN)
    digest.update(f"{seed}:{iteration}:{epoch}:{size}".encode("ascii"))
    generator = torch.Generator().manual_seed(
        int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)
    )
    return torch.randperm(size, generator=generator)


def _update_critic(
    policy: B7RecurrentPolicy,
    batch: B7Batch,
    optimizer: torch.optim.Optimizer,
    *,
    seed: int,
    iteration: int,
    device: torch.device,
) -> dict[str, Any]:
    features = torch.cat([episode.privileged_feature for episode in batch.episodes]).to(device)
    returns = torch.cat([episode.returns for episode in batch.episodes]).to(device)
    weights = torch.cat(
        [
            torch.full((episode.length,), batch.total_steps / batch.episode_count / episode.length)
            for episode in batch.episodes
        ]
    ).to(device)
    if not torch.isclose(weights.mean(), torch.tensor(1.0, device=device), atol=1e-5):
        raise AssertionError("B7 critic episode-equivalent weights do not average one")
    losses: list[float] = []
    steps = 0
    denominator = float(min(policy.config.critic_minibatch_size, batch.total_steps))
    policy.critic.train()
    for epoch in range(policy.config.critic_epochs):
        order = _critic_order(batch.total_steps, seed, iteration, epoch)
        for start in range(0, batch.total_steps, policy.config.critic_minibatch_size):
            indices = order[start : start + policy.config.critic_minibatch_size].to(device)
            prediction = policy.value(features[indices])
            loss = (weights[indices] * (prediction - returns[indices]) ** 2).sum() / denominator
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None for parameter in policy.actor.parameters()):
                raise AssertionError("B7 actor received a critic gradient")
            nn.utils.clip_grad_norm_(tuple(policy.critic.parameters()), policy.config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            steps += 1
    expected = policy.config.critic_epochs * math.ceil(
        batch.total_steps / policy.config.critic_minibatch_size
    )
    if steps != expected:
        raise AssertionError("B7 critic did not complete all three epochs")
    return {
        "critic_epochs_completed": policy.config.critic_epochs,
        "critic_optimizer_steps": steps,
        "critic_loss_mean": float(np.mean(losses)),
    }


def update_policy(
    policy: B7RecurrentPolicy,
    batch: B7Batch,
    actor_optimizer: torch.optim.Adam,
    critic_optimizer: torch.optim.Adam,
    *,
    seed: int,
    iteration: int,
    consecutive_rejections: int,
) -> dict[str, Any]:
    """One full-rollout recurrent actor step, then isolated critic epochs."""

    device = next(policy.actor.parameters()).device
    actor_ids = {id(parameter) for group in actor_optimizer.param_groups for parameter in group["params"]}
    if actor_ids != {id(parameter) for parameter in policy.trainable_actor_parameters}:
        raise ValueError("B7 actor optimizer inventory drift")
    if actor_ids & {id(parameter) for parameter in policy.critic.parameters()}:
        raise ValueError("B7 actor/critic optimizer overlap")
    preupdate = preupdate_replay_metrics(policy, batch, device)
    if preupdate["max_abs_ratio_minus_one"] > 1e-4:
        raise AssertionError("B7 pre-update recurrent replay ratio is not one")

    actor_before = _copy_module_state(policy.actor)
    optimizer_before = copy.deepcopy(actor_optimizer.state_dict())
    lr_before = {str(group["role"]): float(group["lr"]) for group in actor_optimizer.param_groups}
    actor_optimizer.zero_grad(set_to_none=True)
    policy.actor.train()
    losses: list[float] = []
    for episode in batch.episodes:
        mean = policy.sequence_means(
            episode.lidar.to(device), episode.previous_speed.to(device)
        )
        raw = episode.raw_action.to(device)
        ratio = torch.exp(policy.log_prob(mean, raw) - episode.old_log_prob.to(device))
        advantage = episode.normalized_advantage.to(device)
        unclipped = ratio * advantage
        clipped = torch.clamp(
            ratio, 1.0 - policy.config.clip_eps, 1.0 + policy.config.clip_eps
        ) * advantage
        surrogate = torch.minimum(unclipped, clipped).mean()
        bound = mean_bound_penalty(mean).mean()
        loss = (-surrogate + policy.config.mean_bound_coef * bound) / batch.episode_count
        loss.backward()
        losses.append(float((loss.detach() * batch.episode_count).item()))
    if any(parameter.grad is not None for parameter in policy.frozen_actor_parameters):
        raise AssertionError("B7 frozen actor encoder received a gradient")
    if any(parameter.grad is not None for parameter in policy.critic.parameters()):
        raise AssertionError("B7 critic received an actor gradient")
    grad_norm = float(
        nn.utils.clip_grad_norm_(policy.trainable_actor_parameters, policy.config.max_grad_norm).item()
    )
    finite_grad = math.isfinite(grad_norm) and all(
        parameter.grad is None or torch.all(torch.isfinite(parameter.grad))
        for parameter in policy.trainable_actor_parameters
    )
    if finite_grad:
        actor_optimizer.step()
        candidate_means = _candidate_means(policy, batch, device)
        rollout_kl, safe_kl = actor_kl_metrics(policy, batch, candidate_means, device)
        accepted = (
            float(rollout_kl["mean"]) <= policy.config.rollout_kl_cap
            and float(safe_kl["mean"]) <= policy.config.safe_kl_cap
        )
    else:
        rollout_kl = {"mean": float("inf")}
        safe_kl = {"mean": float("inf")}
        accepted = False

    if accepted:
        consecutive_after = 0
    else:
        _restore_module_state(policy.actor, actor_before)
        actor_optimizer.load_state_dict(optimizer_before)
        for group in actor_optimizer.param_groups:
            group["lr"] = float(group["lr"]) * 0.5
        consecutive_after = int(consecutive_rejections) + 1
    actor_optimizer.zero_grad(set_to_none=True)
    critic = _update_critic(
        policy,
        batch,
        critic_optimizer,
        seed=seed,
        iteration=iteration,
        device=device,
    )
    policy.assert_frozen_exact()
    return {
        "actor_update_accepted": accepted,
        "actor_optimizer_steps_attempted": 1,
        "actor_optimizer_steps_committed": int(accepted),
        "actor_loss_episode_mean": float(np.mean(losses)),
        "actor_grad_norm_before_clip": grad_norm,
        "finite_actor_gradient": finite_grad,
        "preupdate_replay": preupdate,
        "old_policy_rollout_mean_kl": rollout_kl,
        "current_rollout_bc_safe_mean_kl": safe_kl,
        "actor_lr_before": lr_before,
        "actor_lr_after": {
            str(group["role"]): float(group["lr"]) for group in actor_optimizer.param_groups
        },
        "actor_and_adam_restored_on_rejection": not accepted,
        "consecutive_rejections_before": int(consecutive_rejections),
        "consecutive_rejections_after": consecutive_after,
        "early_stop_required": consecutive_after >= policy.config.max_consecutive_rejections,
        **critic,
    }


def actor_tensor_digest(policy: B7RecurrentPolicy) -> str:
    return actor_snapshot_sha256(policy.actor_state())
