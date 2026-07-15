"""Pure B2 two-objective PPO losses, replay contract, and optimizer groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import torch
import torch.nn as nn

from bplus_v22 import DUAL_MAX_VALUE, PRIVILEGED_FEATURE_DIM
from bplus_v22.ppo_buffer import require_b2_tensor_batch


B2_CRITIC_HEADS = ("collision", "performance")


class B2Critics(nn.Module):
    """Exactly two privileged critics; no legacy dense-reward head exists."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        width = int(hidden_dim)
        if width <= 0:
            raise ValueError("B2 critic hidden dimension must be positive")
        self.collision = self._network(width)
        self.performance = self._network(width)

    @staticmethod
    def _network(width: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(PRIVILEGED_FEATURE_DIM, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, privileged: torch.Tensor) -> dict[str, torch.Tensor]:
        if (
            privileged.ndim != 2
            or privileged.shape[1] != PRIVILEGED_FEATURE_DIM
            or not torch.all(torch.isfinite(privileged))
        ):
            raise ValueError("B2 privileged critic input must be finite [B,12]")
        return {
            "collision": self.collision(privileged).squeeze(-1),
            "performance": self.performance(privileged).squeeze(-1),
        }


def two_head_critic_losses(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Bind each B2 critic exclusively to its same-named detached target."""

    expected = set(B2_CRITIC_HEADS)
    if set(predictions) != expected or set(targets) != expected:
        raise ValueError("B2 critics require exact collision/performance channels")
    output: dict[str, torch.Tensor] = {}
    for name in B2_CRITIC_HEADS:
        prediction = predictions[name]
        target = targets[name]
        if prediction.ndim != 1 or prediction.shape != target.shape:
            raise ValueError(f"B2 {name} critic target shape mismatch")
        if not torch.all(torch.isfinite(prediction)) or not torch.all(torch.isfinite(target)):
            raise ValueError(f"B2 {name} critic contains nonfinite value")
        output[name] = 0.5 * torch.mean((prediction - target.detach()) ** 2)
    return output


@dataclass(frozen=True)
class CollisionScaleRecord:
    informative: bool
    batch_variance: float
    variance_before: float | None
    variance_after: float | None
    scale: float


@dataclass
class RunningCollisionScale:
    """EMA scale that does not collapse during collision-empty rollouts.

    Updating on an all-constant collision advantage would drive an EMA toward
    zero solely because the rare event was absent.  Such batches therefore
    normalize to zero after centering but do not initialize or decay the state.
    """

    decay: float = 0.99
    variance: float | None = None
    informative_updates: int = 0
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not np.isfinite(self.decay) or not 0.0 <= self.decay < 1.0:
            raise ValueError("collision scale decay must be in [0,1)")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("collision scale epsilon must be positive")
        if self.variance is not None and (
            not np.isfinite(self.variance) or self.variance <= 0.0
        ):
            raise ValueError("collision scale variance must be positive when set")
        if int(self.informative_updates) != self.informative_updates or self.informative_updates < 0:
            raise ValueError("collision scale update count is invalid")

    def normalize(
        self, value: torch.Tensor, *, update: bool, event_present: bool
    ) -> tuple[torch.Tensor, CollisionScaleRecord]:
        if value.ndim != 1 or value.numel() < 2 or not torch.all(torch.isfinite(value)):
            raise ValueError("collision advantage must be a finite vector with >=2 values")
        centered = value - value.mean()
        batch_variance = float(centered.detach().pow(2).mean().item())
        before = self.variance
        informative = bool(event_present) and batch_variance > self.epsilon
        if update and informative:
            if self.variance is None:
                self.variance = batch_variance
            else:
                self.variance = (
                    self.decay * self.variance + (1.0 - self.decay) * batch_variance
                )
            self.informative_updates += 1
        if self.variance is None:
            normalized = torch.zeros_like(centered)
            return normalized, CollisionScaleRecord(
                informative=False,
                batch_variance=batch_variance,
                variance_before=before,
                variance_after=self.variance,
                scale=0.0,
            )
        scale_variance = self.variance
        scale = float(np.sqrt(max(scale_variance, self.epsilon)))
        normalized = centered / scale
        if not torch.all(torch.isfinite(normalized)):
            raise FloatingPointError("collision advantage normalization became nonfinite")
        return normalized, CollisionScaleRecord(
            informative=informative,
            batch_variance=batch_variance,
            variance_before=before,
            variance_after=self.variance,
            scale=scale,
        )

    def state_dict(self) -> dict[str, float | int | None]:
        return {
            "decay": self.decay,
            "variance": self.variance,
            "informative_updates": self.informative_updates,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, float | int | None]) -> "RunningCollisionScale":
        expected = ("decay", "variance", "informative_updates", "epsilon")
        if tuple(state) != expected:
            raise ValueError("collision scale checkpoint fields/order drift")
        return cls(
            decay=float(state["decay"]),
            variance=None if state["variance"] is None else float(state["variance"]),
            informative_updates=int(state["informative_updates"]),
            epsilon=float(state["epsilon"]),
        )


def _batch_normalize(value: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    if value.ndim != 1 or value.numel() < 2 or not torch.all(torch.isfinite(value)):
        raise ValueError("performance advantage must be a finite vector with >=2 values")
    centered = value - value.mean()
    return centered / torch.clamp(centered.pow(2).mean().sqrt(), min=float(epsilon))


def b2_constrained_advantage(
    collision_advantage: torch.Tensor,
    performance_advantage: torch.Tensor,
    dual_value: float,
    collision_scale: RunningCollisionScale,
    *,
    update_collision_scale: bool,
    collision_event_present: bool,
) -> tuple[torch.Tensor, CollisionScaleRecord]:
    """Return the detached B2 actor signal with independently scaled objectives."""

    if collision_advantage.shape != performance_advantage.shape:
        raise ValueError("B2 collision/performance advantage shapes differ")
    if not np.isfinite(dual_value) or not 0.0 <= float(dual_value) <= DUAL_MAX_VALUE:
        raise ValueError("B2 dual value is outside its locked range")
    collision, record = collision_scale.normalize(
        collision_advantage.detach(),
        update=update_collision_scale,
        event_present=collision_event_present,
    )
    performance = _batch_normalize(performance_advantage.detach())
    combined = (-collision + float(dual_value) * performance) / (1.0 + float(dual_value))
    return combined.detach(), record


@dataclass(frozen=True)
class PolicyReplayTerms:
    log_prob: torch.Tensor
    entropy: torch.Tensor


ReplayHook = Callable[..., PolicyReplayTerms | tuple[torch.Tensor, torch.Tensor] | Mapping[str, torch.Tensor]]


def replay_policy_terms(
    batch: Mapping[str, torch.Tensor], replay_hook: ReplayHook
) -> PolicyReplayTerms:
    """Replay a policy using the offsets saved with each transition.

    The hook intentionally receives no privileged critic feature.  It must
    reconstruct the behavior-context distribution from deployable inputs,
    stored latent, and both saved offsets.
    """

    count = require_b2_tensor_batch(batch)
    result = replay_hook(
        bc_feature=batch["bc_feature"],
        lidar_history=batch["lidar_history"],
        scalar_history=batch["scalar_history"],
        latent=batch["latent"],
        intervention_offset=batch["intervention_offset"],
        conditional_brake_offset=batch["conditional_brake_offset"],
        steer_std_scale=batch["steer_std_scale"],
        brake_std_scale=batch["brake_std_scale"],
    )
    if isinstance(result, PolicyReplayTerms):
        terms = result
    elif isinstance(result, Mapping):
        if set(result) != {"log_prob", "entropy"}:
            raise ValueError("B2 replay hook mapping must contain only log_prob/entropy")
        terms = PolicyReplayTerms(result["log_prob"], result["entropy"])
    else:
        try:
            log_prob, entropy = result
        except (TypeError, ValueError) as error:
            raise TypeError("B2 replay hook returned an invalid value") from error
        terms = PolicyReplayTerms(log_prob, entropy)
    for name, value in (("log_prob", terms.log_prob), ("entropy", terms.entropy)):
        if not isinstance(value, torch.Tensor) or value.shape != (count,):
            raise ValueError(f"B2 replay {name} must have shape [N]")
        if not torch.all(torch.isfinite(value)):
            raise ValueError(f"B2 replay {name} is nonfinite")
    return terms


@dataclass(frozen=True)
class PPOClipResult:
    loss: torch.Tensor
    ratio: torch.Tensor
    log_ratio: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    ratio_min: torch.Tensor
    ratio_mean: torch.Tensor
    ratio_max: torch.Tensor


def clipped_policy_objective(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    clip_epsilon: float,
) -> PPOClipResult:
    """Compute the standard clipped PPO surrogate and finite diagnostics."""

    if (
        new_log_prob.ndim != 1
        or new_log_prob.shape != old_log_prob.shape
        or new_log_prob.shape != advantage.shape
        or new_log_prob.numel() < 1
    ):
        raise ValueError("B2 PPO log-probability/advantage shapes differ")
    if any(not torch.all(torch.isfinite(value)) for value in (new_log_prob, old_log_prob, advantage)):
        raise ValueError("B2 PPO loss contains nonfinite input")
    epsilon = float(clip_epsilon)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("B2 PPO clip epsilon must lie in (0,1)")
    log_ratio = new_log_prob - old_log_prob.detach()
    ratio = torch.exp(log_ratio)
    if not torch.all(torch.isfinite(ratio)):
        raise FloatingPointError("B2 PPO ratio is nonfinite")
    detached_advantage = advantage.detach()
    unclipped = ratio * detached_advantage
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * detached_advantage
    loss = -torch.minimum(unclipped, clipped).mean()
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > epsilon).float().mean()
    return PPOClipResult(
        loss,
        ratio,
        log_ratio,
        approx_kl,
        clip_fraction,
        ratio.min(),
        ratio.mean(),
        ratio.max(),
    )


@dataclass(frozen=True)
class B2LossBundle:
    actor_loss: torch.Tensor
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    critic_losses: dict[str, torch.Tensor]
    ratio: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    ratio_min: torch.Tensor
    ratio_mean: torch.Tensor
    ratio_max: torch.Tensor
    collision_scale: CollisionScaleRecord


def compute_b2_losses(
    critics: B2Critics,
    batch: Mapping[str, torch.Tensor],
    replay_hook: ReplayHook,
    collision_scale: RunningCollisionScale,
    *,
    dual_value: float,
    clip_epsilon: float,
    entropy_coefficient: float,
    update_collision_scale: bool,
    precomputed_actor_advantage: torch.Tensor | None = None,
    precomputed_collision_scale: CollisionScaleRecord | None = None,
) -> B2LossBundle:
    """Build actor and two isolated critic losses for one PPO epoch.

    Optimizer stepping is deliberately outside this pure function.  The caller
    updates ``collision_scale`` once per rollout (normally on epoch zero) and
    reuses it without mutation in later PPO epochs.
    """

    require_b2_tensor_batch(batch)
    coefficient = float(entropy_coefficient)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("B2 entropy coefficient must be nonnegative")
    replay = replay_policy_terms(batch, replay_hook)
    if precomputed_actor_advantage is None:
        advantage, scale_record = b2_constrained_advantage(
            batch["collision_advantage"],
            batch["performance_advantage"],
            dual_value,
            collision_scale,
            update_collision_scale=update_collision_scale,
            collision_event_present=bool(
                torch.any(batch["collision_cost"] > 0).item()
            ),
        )
    else:
        if update_collision_scale or precomputed_collision_scale is None:
            raise ValueError(
                "precomputed B2 actor advantage requires a frozen scale record"
            )
        advantage = precomputed_actor_advantage.detach()
        if advantage.shape != batch["old_log_prob"].shape or not torch.all(
            torch.isfinite(advantage)
        ):
            raise ValueError("precomputed B2 actor advantage is invalid")
        scale_record = precomputed_collision_scale
    policy = clipped_policy_objective(
        replay.log_prob,
        batch["old_log_prob"],
        advantage,
        clip_epsilon,
    )
    entropy = replay.entropy.mean()
    actor_loss = policy.loss - coefficient * entropy
    predictions = critics(batch["privileged_critic_feature"].detach())
    critic_losses = two_head_critic_losses(
        predictions,
        {
            "collision": batch["collision_return"],
            "performance": batch["performance_return"],
        },
    )
    tensors = (actor_loss, policy.loss, entropy, policy.approx_kl, policy.clip_fraction)
    if any(not torch.all(torch.isfinite(value)) for value in tensors):
        raise FloatingPointError("B2 loss diagnostics became nonfinite")
    return B2LossBundle(
        actor_loss=actor_loss,
        policy_loss=policy.loss,
        entropy=entropy,
        critic_losses=critic_losses,
        ratio=policy.ratio,
        approx_kl=policy.approx_kl,
        clip_fraction=policy.clip_fraction,
        ratio_min=policy.ratio_min,
        ratio_mean=policy.ratio_mean,
        ratio_max=policy.ratio_max,
        collision_scale=scale_record,
    )


@dataclass
class B2Optimizers:
    actor: torch.optim.Optimizer
    collision_critic: torch.optim.Optimizer
    performance_critic: torch.optim.Optimizer
    actor_group_names: tuple[str, ...]

    def state_dict(self) -> dict[str, dict]:
        return {
            "actor": self.actor.state_dict(),
            "collision_critic": self.collision_critic.state_dict(),
            "performance_critic": self.performance_critic.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, dict]) -> None:
        expected = ("actor", "collision_critic", "performance_critic")
        if tuple(state) != expected:
            raise ValueError("B2 optimizer checkpoint fields/order drift")
        self.actor.load_state_dict(state["actor"])
        self.collision_critic.load_state_dict(state["collision_critic"])
        self.performance_critic.load_state_dict(state["performance_critic"])


def build_b2_optimizers(
    policy: nn.Module,
    critics: B2Critics,
    *,
    critic_learning_rate: float,
) -> B2Optimizers:
    """Construct exact, disjoint actor/collision/performance optimizers."""

    learning_rate = float(critic_learning_rate)
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("B2 critic learning rate must be positive")
    provider = getattr(policy, "optimizer_parameter_groups", None)
    if not callable(provider):
        raise TypeError("B2 policy must expose optimizer_parameter_groups()")
    raw_groups = list(provider())
    if not raw_groups:
        raise ValueError("B2 actor optimizer has no parameter groups")
    actor_groups = []
    names = []
    actor_parameters: list[nn.Parameter] = []
    for raw in raw_groups:
        if not isinstance(raw, Mapping) or set(raw) != {"name", "params", "lr"}:
            raise ValueError("B2 actor optimizer group schema drift")
        name = str(raw["name"])
        parameters = list(raw["params"])
        lr = float(raw["lr"])
        if not name or name in names or not parameters or not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("B2 actor optimizer group is invalid")
        if any(not isinstance(parameter, nn.Parameter) or not parameter.requires_grad for parameter in parameters):
            raise ValueError("B2 actor optimizer group contains frozen/non-parameter value")
        names.append(name)
        actor_parameters.extend(parameters)
        actor_groups.append({"name": name, "params": parameters, "lr": lr})
    actor_ids = [id(parameter) for parameter in actor_parameters]
    expected_actor = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    if len(actor_ids) != len(set(actor_ids)) or set(actor_ids) != expected_actor:
        raise ValueError("B2 actor optimizer groups overlap or omit trainable parameters")

    collision = list(critics.collision.parameters())
    performance = list(critics.performance.parameters())
    collision_ids = {id(parameter) for parameter in collision}
    performance_ids = {id(parameter) for parameter in performance}
    if (
        not collision
        or not performance
        or collision_ids & performance_ids
        or set(actor_ids) & (collision_ids | performance_ids)
        or collision_ids | performance_ids
        != {id(parameter) for parameter in critics.parameters() if parameter.requires_grad}
    ):
        raise ValueError("B2 actor/critic optimizer inventories are not exact and disjoint")
    return B2Optimizers(
        actor=torch.optim.Adam(actor_groups, betas=(0.9, 0.999), eps=1e-8),
        collision_critic=torch.optim.Adam(
            collision, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8
        ),
        performance_critic=torch.optim.Adam(
            performance, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8
        ),
        actor_group_names=tuple(names),
    )
