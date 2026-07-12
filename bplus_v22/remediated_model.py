"""Prospective post-Task-10 hierarchical intervention policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from bplus_v22 import BRAKE_BUDGET, INITIAL_BRAKE_LOGIT, STEER_BUDGET
from bplus_v22.model import V22Policy


INITIAL_INTERVENTION_LOGIT = -6.0
STEER_LIMIT = 0.52
ACTION_SCHEMA = "bplus-v2.2-hierarchical-residual-action-1"
CHECKPOINT_SCHEMA = "bplus-v2.2-hierarchical-warmstart-checkpoint-1"


@dataclass(frozen=True)
class HierarchicalResidualAction:
    intervention_gate: torch.Tensor
    steer_latent: torch.Tensor
    brake_gate: torch.Tensor
    brake_latent: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.intervention_gate,
            self.steer_latent,
            self.brake_gate,
            self.brake_latent,
        )
        if any(value.shape != self.steer_latent.shape for value in tensors):
            raise ValueError("hierarchical action component shapes differ")
        if self.steer_latent.ndim < 1 or self.steer_latent.shape[-1] != 1:
            raise ValueError("hierarchical actions must end in one coordinate")
        if any(not torch.all(torch.isfinite(value)) for value in tensors):
            raise ValueError("hierarchical action contains nonfinite value")
        for name, value in (
            ("intervention", self.intervention_gate),
            ("brake", self.brake_gate),
        ):
            if not torch.all((value == 0) | (value == 1)):
                raise ValueError(f"{name} gate must contain only 0/1")
        if torch.any(self.brake_gate > self.intervention_gate):
            raise ValueError("brake gate cannot be active outside intervention")
        inactive = self.intervention_gate == 0
        no_brake = self.brake_gate == 0
        if torch.any(self.steer_latent[inactive] != 0):
            raise ValueError("inactive intervention must canonicalize steer latent to zero")
        if torch.any(self.brake_latent[no_brake] != 0):
            raise ValueError("inactive brake must canonicalize brake latent to zero")

    def as_tensor(self) -> torch.Tensor:
        return torch.cat(
            [
                self.intervention_gate,
                self.steer_latent,
                self.brake_gate,
                self.brake_latent,
            ],
            dim=-1,
        )

    @classmethod
    def from_tensor(cls, value: torch.Tensor) -> "HierarchicalResidualAction":
        if value.ndim < 1 or value.shape[-1] != 4 or not torch.all(torch.isfinite(value)):
            raise ValueError("stored hierarchical action must be finite [...,4]")
        return cls(value[..., 0:1], value[..., 1:2], value[..., 2:3], value[..., 3:4])


@dataclass(frozen=True)
class CompositionLedger:
    """Exact latent-to-actuator accounting for one 100 Hz BC command."""

    raw_base: torch.Tensor
    deployed_base: torch.Tensor
    requested_residual: torch.Tensor
    negative_steer_headroom: torch.Tensor
    positive_steer_headroom: torch.Tensor
    brake_headroom: torch.Tensor
    applied_residual: torch.Tensor
    command: torch.Tensor
    external_clip_would_change: torch.Tensor


class HierarchicalResidualDistribution:
    """NO_OP/INTERVENE with conditional steer and conditional brake atom."""

    def __init__(
        self,
        intervention_logits: torch.Tensor,
        steer_mean: torch.Tensor,
        steer_std: torch.Tensor,
        brake_logits: torch.Tensor,
        brake_mean: torch.Tensor,
        brake_std: torch.Tensor,
    ):
        values = (
            intervention_logits,
            steer_mean,
            steer_std,
            brake_logits,
            brake_mean,
            brake_std,
        )
        if any(value.shape != steer_mean.shape for value in values):
            raise ValueError("hierarchical distribution parameter shapes differ")
        if steer_mean.ndim < 1 or steer_mean.shape[-1] != 1:
            raise ValueError("hierarchical parameters must end in one coordinate")
        if any(not torch.all(torch.isfinite(value)) for value in values):
            raise ValueError("hierarchical distribution contains nonfinite value")
        if torch.any(steer_std <= 0) or torch.any(brake_std <= 0):
            raise ValueError("hierarchical standard deviation must be positive")
        self.intervention = torch.distributions.Bernoulli(logits=intervention_logits)
        self.steer = torch.distributions.Normal(steer_mean, steer_std)
        self.brake_gate_distribution = torch.distributions.Bernoulli(logits=brake_logits)
        self.brake = torch.distributions.Normal(brake_mean, brake_std)

    @property
    def intervention_probability(self) -> torch.Tensor:
        return self.intervention.probs

    @property
    def brake_probability(self) -> torch.Tensor:
        return self.brake_gate_distribution.probs

    def sample(self) -> HierarchicalResidualAction:
        intervention = self.intervention.sample()
        brake_gate = intervention * self.brake_gate_distribution.sample()
        return HierarchicalResidualAction(
            intervention,
            intervention * self.steer.sample(),
            brake_gate,
            brake_gate * self.brake.sample(),
        )

    def deterministic(self) -> HierarchicalResidualAction:
        intervention = (self.intervention.logits > 0).to(self.intervention.logits.dtype)
        brake = intervention * (self.brake_gate_distribution.logits > 0).to(
            self.brake_gate_distribution.logits.dtype
        )
        return HierarchicalResidualAction(
            intervention,
            intervention * self.steer.mean,
            brake,
            brake * self.brake.mean,
        )

    def log_prob(self, action: HierarchicalResidualAction) -> torch.Tensor:
        action.__post_init__()
        zero = torch.zeros_like(action.intervention_gate)
        brake_density = torch.where(
            action.brake_gate.bool(),
            self.brake.log_prob(action.brake_latent),
            zero,
        )
        conditional = (
            self.steer.log_prob(action.steer_latent)
            + self.brake_gate_distribution.log_prob(action.brake_gate)
            + brake_density
        )
        value = self.intervention.log_prob(action.intervention_gate) + torch.where(
            action.intervention_gate.bool(), conditional, zero
        )
        return value.squeeze(-1)

    def entropy(self) -> torch.Tensor:
        conditional = self.steer.entropy() + self.brake_gate_distribution.entropy()
        conditional = conditional + self.brake_probability * self.brake.entropy()
        return (self.intervention.entropy() + self.intervention_probability * conditional).squeeze(-1)

    @staticmethod
    def requested_residual(action: HierarchicalResidualAction) -> torch.Tensor:
        """Return the base-independent requested residual, not an actuator command."""

        action.__post_init__()
        steer = torch.tanh(action.steer_latent) * STEER_BUDGET
        brake = torch.sigmoid(action.brake_latent) * BRAKE_BUDGET
        active_brake = action.intervention_gate.bool() & action.brake_gate.bool()
        return torch.cat(
            [
                action.intervention_gate * steer,
                torch.where(active_brake, -brake, torch.zeros_like(brake)),
            ],
            dim=-1,
        )

    @staticmethod
    def compose(
        base_action: torch.Tensor,
        action: HierarchicalResidualAction,
    ) -> CompositionLedger:
        """Project one held latent against the current 100 Hz deployed BC base."""

        action.__post_init__()
        if base_action.ndim != 2 or base_action.shape[-1] != 2:
            raise ValueError("hierarchical composition base must be [B,2]")
        if base_action.shape[:-1] != action.steer_latent.shape[:-1]:
            raise ValueError("hierarchical action/base batch shapes differ")
        raw_base = base_action
        base_steer = torch.clamp(raw_base[..., 0:1], -STEER_LIMIT, STEER_LIMIT)
        base_speed = torch.clamp_min(raw_base[..., 1:2], 0.0)
        deployed_base = torch.cat([base_steer, base_speed], dim=-1)
        requested = HierarchicalResidualDistribution.requested_residual(action)
        positive_room = torch.clamp(STEER_LIMIT - base_steer, min=0.0, max=STEER_BUDGET)
        negative_room = torch.clamp(base_steer + STEER_LIMIT, min=0.0, max=STEER_BUDGET)
        steer_delta = torch.maximum(
            -negative_room, torch.minimum(requested[..., 0:1], positive_room)
        )
        brake_room = torch.clamp(base_speed, min=0.0, max=BRAKE_BUDGET)
        speed_delta = torch.maximum(
            -brake_room, torch.minimum(requested[..., 1:2], torch.zeros_like(brake_room))
        )
        delta = torch.cat([steer_delta, speed_delta], dim=-1)
        bounded = deployed_base + delta
        intervention = action.intervention_gate.expand_as(base_action)
        composed = torch.where(intervention.bool(), bounded, deployed_base)
        active = action.intervention_gate.squeeze(-1) == 1
        if (
            torch.any(bounded[active, 0].abs() > STEER_LIMIT)
            or torch.any(bounded[active, 1] < 0)
        ):
            raise AssertionError("bound-preserving hierarchical composition failed")
        external = torch.cat(
            [
                torch.clamp(composed[..., 0:1], -STEER_LIMIT, STEER_LIMIT),
                torch.clamp_min(composed[..., 1:2], 0.0),
            ],
            dim=-1,
        )
        external_change = torch.any(external != composed, dim=-1, keepdim=True)
        if torch.any(external_change):
            raise AssertionError("hierarchical command still requires evaluator clipping")
        return CompositionLedger(
            raw_base=raw_base,
            deployed_base=deployed_base,
            requested_residual=requested,
            negative_steer_headroom=negative_room,
            positive_steer_headroom=positive_room,
            brake_headroom=brake_room,
            applied_residual=delta,
            command=composed,
            external_clip_would_change=external_change,
        )


class RemediatedV22Policy(V22Policy):
    """Historical V22 backbone with a new hierarchical intervention head."""

    def __init__(self, arm: str, **kwargs):
        super().__init__(arm, **kwargs)
        feature_dim = self.brake_gate.in_features
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(kwargs.get("initialization_seed", 20260711)) + 2)
            self.intervention_gate = nn.Linear(feature_dim, 1)
            nn.init.zeros_(self.intervention_gate.weight)
            nn.init.constant_(self.intervention_gate.bias, INITIAL_INTERVENTION_LOGIT)
        for parameter in self.intervention_gate.parameters():
            parameter.requires_grad = True
        self.register_buffer("intervention_logit_offset", torch.zeros(1))

    def distribution(self, bc_feature, lidar_history, scalar_history):
        feature = self.action_core(
            self.policy_feature(bc_feature, lidar_history, scalar_history)
        )
        steer_std = self.log_steer_std.exp().expand(len(feature), 1)
        brake_std = self.log_brake_std.exp().expand(len(feature), 1)
        return HierarchicalResidualDistribution(
            self.intervention_gate(feature) + self.intervention_logit_offset,
            self.steer_mean(feature),
            steer_std,
            self.brake_gate(feature),
            self.brake_mean(feature),
            brake_std,
        )

    @staticmethod
    def compose(base_action, action):
        return HierarchicalResidualDistribution.compose(base_action, action).command

    def load_hierarchical_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Fail closed on old three-dimensional/single-gate policy states."""

        required = {
            "intervention_gate.weight",
            "intervention_gate.bias",
            "intervention_logit_offset",
        }
        missing = required - set(state)
        if missing:
            raise ValueError(
                "hierarchical checkpoint/action-schema mismatch; missing "
                + ", ".join(sorted(missing))
            )
        self.load_state_dict(dict(state), strict=True)


def apply_intervention_logit_offset(policy: RemediatedV22Policy, offset: float) -> None:
    value = float(offset)
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError("intervention calibration offset is nonfinite")
    with torch.no_grad():
        policy.intervention_logit_offset.fill_(value)


def initialize_hierarchical_priors(
    policy: RemediatedV22Policy,
    intervention_logit: float,
    conditional_brake_logit: float,
) -> None:
    values = torch.tensor(
        [float(intervention_logit), float(conditional_brake_logit)],
        dtype=torch.float64,
    )
    if not torch.all(torch.isfinite(values)):
        raise ValueError("hierarchical warm-start prior is nonfinite")
    with torch.no_grad():
        policy.intervention_gate.bias.fill_(float(intervention_logit))
        policy.brake_gate.bias.fill_(float(conditional_brake_logit))
        policy.intervention_gate.weight.zero_()
        policy.brake_gate.weight.zero_()
        policy.intervention_logit_offset.zero_()
