"""Direct-outcome dual constraint and lexicographic policy selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np
import torch

from bplus_v22 import (
    DUAL_EMA_ALPHA,
    DUAL_INITIAL_VALUE,
    DUAL_LEARNING_RATE,
    DUAL_MAX_VALUE,
    DUAL_MIN_COMPLETED_EPISODES,
    OVERTAKE_NONINFERIORITY,
    validate_arm,
)


def normalize_advantage(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize one objective independently; never mix objective scales."""

    if value.ndim != 1 or value.numel() < 2 or not torch.all(torch.isfinite(value)):
        raise ValueError("objective advantage must be a finite vector with >=2 values")
    std = value.std(unbiased=False)
    return (value - value.mean()) / torch.clamp(std, min=float(eps))


def constrained_policy_advantage(
    collision_advantage: torch.Tensor,
    performance_advantage: torch.Tensor,
    dual_value: float,
) -> torch.Tensor:
    """Actor maximization signal: reduce collision cost, protect performance."""

    if collision_advantage.shape != performance_advantage.shape:
        raise ValueError("collision/performance advantage shapes differ")
    if not np.isfinite(dual_value) or not 0.0 <= dual_value <= DUAL_MAX_VALUE:
        raise ValueError("overtake dual must be finite and within its locked cap")
    collision = normalize_advantage(collision_advantage)
    performance = normalize_advantage(performance_advantage)
    return (-collision + float(dual_value) * performance) / (1.0 + float(dual_value))


def detached_actor_objective(
    log_probability: torch.Tensor,
    collision_advantage: torch.Tensor,
    performance_advantage: torch.Tensor,
    dual_value: float,
) -> torch.Tensor:
    """Return an actor-only objective with all critic-derived inputs detached.

    Privileged critic features may produce the two advantages upstream, but
    optimizing this scalar can only differentiate through the policy log
    probability. Diagnostic TTC/alarm tensors are deliberately absent from
    this API.
    """

    if log_probability.ndim != 1 or not torch.all(torch.isfinite(log_probability)):
        raise ValueError("actor log probability must be a finite vector")
    if log_probability.shape != collision_advantage.shape:
        raise ValueError("actor and advantage shapes differ")
    combined = constrained_policy_advantage(
        collision_advantage.detach(), performance_advantage.detach(), dual_value
    )
    return -(log_probability * combined.detach()).mean()


def separate_critic_losses(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Bind each named critic exclusively to the same named return target."""

    expected = {"reward", "collision", "performance"}
    if set(predictions) != expected or set(targets) != expected:
        raise ValueError("critics require exact reward/collision/performance channels")
    losses: dict[str, torch.Tensor] = {}
    for name in sorted(expected):
        prediction = predictions[name]
        target = targets[name]
        if prediction.ndim != 1 or prediction.shape != target.shape:
            raise ValueError(f"{name} critic target shape mismatch")
        if not torch.all(torch.isfinite(prediction)) or not torch.all(torch.isfinite(target)):
            raise ValueError(f"{name} critic contains nonfinite value")
        losses[name] = torch.mean((prediction - target.detach()) ** 2)
    return losses


@dataclass(frozen=True)
class GradientNormRecord:
    """Pre/post clipping evidence for one disjoint optimizer group."""

    group: str
    maximum: float
    pre_clip: float
    post_clip: float


def clip_gradient_group(
    group: str,
    parameters: Iterable[torch.nn.Parameter],
    maximum: float,
) -> GradientNormRecord:
    """Clip exactly one actor/critic group and return separate norm evidence."""

    if not group or not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("gradient clip group/maximum is invalid")
    selected = [parameter for parameter in parameters if parameter.grad is not None]
    if not selected:
        raise ValueError(f"gradient clip group has no gradients: {group}")
    if any(not torch.all(torch.isfinite(parameter.grad)) for parameter in selected):
        raise ValueError(f"gradient clip group contains nonfinite gradient: {group}")
    pre = float(torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(parameter.grad.detach()) for parameter in selected
    ])).item())
    torch.nn.utils.clip_grad_norm_(selected, float(maximum), error_if_nonfinite=True)
    post = float(torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(parameter.grad.detach()) for parameter in selected
    ])).item())
    return GradientNormRecord(str(group), float(maximum), pre, post)


def allowed_overtake_loss(
    paired_episode_count: int,
    tolerance: float = OVERTAKE_NONINFERIORITY,
) -> int:
    """Translate a rate tolerance into the largest exact integer net loss."""

    count = int(paired_episode_count)
    if count != paired_episode_count or count <= 0:
        raise ValueError("paired episode count must be a positive integer")
    if not np.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
        raise ValueError("overtake tolerance must lie in [0,1)")
    return int(np.floor(float(tolerance) * count + 1e-12))


@dataclass
class OvertakeDual:
    """Bounded, smoothed dual with a positive pre-registered initial value."""

    floor: float
    learning_rate: float = DUAL_LEARNING_RATE
    value: float = DUAL_INITIAL_VALUE
    maximum: float = DUAL_MAX_VALUE
    ema_alpha: float = DUAL_EMA_ALPHA
    min_completed_episodes: int = DUAL_MIN_COMPLETED_EPISODES
    ema_overtake_rate: float | None = None
    completed_episodes: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.floor) or not 0.0 <= self.floor <= 1.0:
            raise ValueError("overtake floor must be a probability")
        locked = (
            (self.learning_rate, DUAL_LEARNING_RATE, "learning_rate"),
            (self.value, DUAL_INITIAL_VALUE, "initial_value"),
            (self.maximum, DUAL_MAX_VALUE, "maximum"),
            (self.ema_alpha, DUAL_EMA_ALPHA, "ema_alpha"),
            (
                self.min_completed_episodes,
                DUAL_MIN_COMPLETED_EPISODES,
                "min_completed_episodes",
            ),
        )
        for actual, expected, name in locked:
            if actual != expected:
                raise ValueError(f"v2.2 dual {name} drift")
        if self.ema_overtake_rate is not None and (
            not np.isfinite(self.ema_overtake_rate)
            or not 0.0 <= self.ema_overtake_rate <= 1.0
        ):
            raise ValueError("dual EMA state is invalid")
        if int(self.completed_episodes) != self.completed_episodes or self.completed_episodes < 0:
            raise ValueError("dual completed-episode state is invalid")

    def update_with_record(
        self, observed_overtake_rate: float, completed_episodes: int
    ) -> "DualUpdateRecord":
        observed = float(observed_overtake_rate)
        if not np.isfinite(observed) or not 0.0 <= observed <= 1.0:
            raise ValueError("observed overtake rate must be a probability")
        count = int(completed_episodes)
        if count != completed_episodes or count <= 0:
            raise ValueError("dual update requires a positive completed-episode count")
        completed_before = self.completed_episodes
        value_before = self.value
        ema_before = self.ema_overtake_rate
        self.completed_episodes += count
        updated = self.completed_episodes >= self.min_completed_episodes
        if not updated:
            return DualUpdateRecord(
                completed_before,
                self.completed_episodes,
                observed,
                ema_before,
                self.ema_overtake_rate,
                value_before,
                self.value,
                False,
            )
        if self.ema_overtake_rate is None:
            self.ema_overtake_rate = observed
        else:
            self.ema_overtake_rate = (
                (1.0 - self.ema_alpha) * self.ema_overtake_rate
                + self.ema_alpha * observed
            )
        violation = self.floor - self.ema_overtake_rate
        self.value = float(
            np.clip(self.value + self.learning_rate * violation, 0.0, self.maximum)
        )
        return DualUpdateRecord(
            completed_before,
            self.completed_episodes,
            observed,
            ema_before,
            self.ema_overtake_rate,
            value_before,
            self.value,
            True,
        )

    def update(self, observed_overtake_rate: float, completed_episodes: int) -> float:
        return self.update_with_record(observed_overtake_rate, completed_episodes).value_after

    def state_dict(self) -> dict[str, float | int | None]:
        """Serializable state with a stable field order for checkpoints/logs."""

        return {
            "floor": self.floor,
            "learning_rate": self.learning_rate,
            "value": self.value,
            "maximum": self.maximum,
            "ema_alpha": self.ema_alpha,
            "min_completed_episodes": self.min_completed_episodes,
            "ema_overtake_rate": self.ema_overtake_rate,
            "completed_episodes": self.completed_episodes,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, float | int | None]) -> "OvertakeDual":
        expected = (
            "floor",
            "learning_rate",
            "value",
            "maximum",
            "ema_alpha",
            "min_completed_episodes",
            "ema_overtake_rate",
            "completed_episodes",
        )
        if tuple(state) != expected:
            raise ValueError("dual checkpoint fields/order drift")
        restored = cls(
            floor=float(state["floor"]),
            learning_rate=float(state["learning_rate"]),
            maximum=float(state["maximum"]),
            ema_alpha=float(state["ema_alpha"]),
            min_completed_episodes=int(state["min_completed_episodes"]),
        )
        value = float(state["value"])
        completed = int(state["completed_episodes"])
        ema = state["ema_overtake_rate"]
        if not np.isfinite(value) or not 0.0 <= value <= restored.maximum:
            raise ValueError("dual checkpoint value is invalid")
        if completed != state["completed_episodes"] or completed < 0:
            raise ValueError("dual checkpoint completed count is invalid")
        if ema is not None and (
            not np.isfinite(float(ema)) or not 0.0 <= float(ema) <= 1.0
        ):
            raise ValueError("dual checkpoint EMA is invalid")
        restored.value = value
        restored.completed_episodes = completed
        restored.ema_overtake_rate = None if ema is None else float(ema)
        return restored


@dataclass(frozen=True)
class DualUpdateRecord:
    """Deterministic transient log; never a standalone stop/select signal."""

    completed_episodes_before: int
    completed_episodes_after: int
    observed_overtake_rate: float
    ema_before: float | None
    ema_after: float | None
    value_before: float
    value_after: float
    updated: bool

    def ordered_log(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SeedDirectOutcome:
    """One seed's complete paired development accounting."""

    seed: int
    paired_episode_count: int
    fixed_collision: int
    new_collision: int
    gained_overtake: int
    lost_overtake: int

    def __post_init__(self) -> None:
        values = (
            self.seed,
            self.paired_episode_count,
            self.fixed_collision,
            self.new_collision,
            self.gained_overtake,
            self.lost_overtake,
        )
        if any(int(value) != value or value < 0 for value in values):
            raise ValueError("seed outcome counts must be nonnegative integers")
        if self.paired_episode_count <= 0:
            raise ValueError("seed paired population must be positive")
        if any(
            value > self.paired_episode_count
            for value in (
                self.fixed_collision,
                self.new_collision,
                self.gained_overtake,
                self.lost_overtake,
            )
        ):
            raise ValueError("seed transition count exceeds paired population")

    @property
    def collision_improves(self) -> bool:
        return self.fixed_collision > self.new_collision

    @property
    def overtake_feasible(self) -> bool:
        return (
            self.lost_overtake - self.gained_overtake
            <= allowed_overtake_loss(self.paired_episode_count)
        )


@dataclass(frozen=True)
class DirectOutcomeCandidate:
    """Paired two-seed development evidence used by the arm selector."""

    arm: str
    snapshot_iteration: int
    paired_episode_count: int
    fixed_collision: int
    new_collision: int
    gained_overtake: int
    lost_overtake: int
    collision_to_confirmed_pass: int
    seed_outcomes: tuple[SeedDirectOutcome, SeedDirectOutcome]
    map_or_skill_collapse: bool = False
    interaction_explained_only_by_attempt_loss: bool = False
    integrity_violation: bool = False

    def __post_init__(self) -> None:
        validate_arm(self.arm)
        counts = (
            self.snapshot_iteration,
            self.paired_episode_count,
            self.fixed_collision,
            self.new_collision,
            self.gained_overtake,
            self.lost_overtake,
            self.collision_to_confirmed_pass,
        )
        if any(int(value) != value or value < 0 for value in counts):
            raise ValueError("direct-outcome candidate counts must be nonnegative integers")
        if self.paired_episode_count <= 0:
            raise ValueError("direct-outcome candidate population must be positive")
        bounded_counts = (
            self.fixed_collision,
            self.new_collision,
            self.gained_overtake,
            self.lost_overtake,
            self.collision_to_confirmed_pass,
        )
        if any(value > self.paired_episode_count for value in bounded_counts):
            raise ValueError("direct-outcome transition count exceeds paired population")
        if len(self.seed_outcomes) != 2 or len({row.seed for row in self.seed_outcomes}) != 2:
            raise ValueError("candidate requires exactly two distinct seed outcomes")
        sums = {
            name: sum(getattr(row, name) for row in self.seed_outcomes)
            for name in (
                "paired_episode_count",
                "fixed_collision",
                "new_collision",
                "gained_overtake",
                "lost_overtake",
            )
        }
        for name, observed in sums.items():
            if observed != getattr(self, name):
                raise ValueError(f"pooled/seed direct-outcome mismatch: {name}")

    @property
    def net_collision_improvement(self) -> int:
        return int(self.fixed_collision - self.new_collision)

    @property
    def net_overtake_improvement(self) -> int:
        return int(self.gained_overtake - self.lost_overtake)

    @property
    def allowed_net_overtake_loss(self) -> int:
        return allowed_overtake_loss(self.paired_episode_count)

    @property
    def overtake_risk_difference(self) -> float:
        return float(self.net_overtake_improvement / self.paired_episode_count)

    @property
    def passes(self) -> bool:
        return bool(
            self.fixed_collision > self.new_collision
            and -self.net_overtake_improvement <= self.allowed_net_overtake_loss
            and self.collision_to_confirmed_pass >= 1
            and all(row.collision_improves for row in self.seed_outcomes)
            and all(row.overtake_feasible for row in self.seed_outcomes)
            and not self.map_or_skill_collapse
            and not self.interaction_explained_only_by_attempt_loss
            and not self.integrity_violation
        )


def select_candidate(candidates: list[DirectOutcomeCandidate]) -> DirectOutcomeCandidate | None:
    """Apply overtake as a floor, then maximize collision improvement."""

    survivors = [candidate for candidate in candidates if candidate.passes]
    if not survivors:
        return None
    return max(
        survivors,
        key=lambda candidate: (
            candidate.net_collision_improvement,
            candidate.net_overtake_improvement,
            -candidate.snapshot_iteration,
            -tuple(sorted({item.arm for item in survivors})).index(candidate.arm),
        ),
    )
