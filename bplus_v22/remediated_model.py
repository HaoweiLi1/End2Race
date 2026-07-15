"""Prospective post-Task-10 hierarchical intervention policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from bplus_v22 import (
    ACTION_CORE_LR,
    ARM_SIDECAR_FINETUNE,
    BRAKE_BUDGET,
    INITIAL_BRAKE_LOGIT,
    SIDECAR_FINETUNE_LR,
    STEER_BUDGET,
)
from bplus_v22.exploration import (
    ActionNoiseKey,
    BehaviorExplorationBatch,
    DETERMINISTIC_CENTERED,
    DETERMINISTIC_MODES,
    DETERMINISTIC_STANDARD,
    KeyedComponentDraws,
    keyed_component_draws,
)
from bplus_v22.model import V22Policy


INITIAL_INTERVENTION_LOGIT = -6.0
STEER_LIMIT = 0.52
ACTION_SCHEMA = "bplus-v2.2-hierarchical-residual-action-1"
CHECKPOINT_SCHEMA = "bplus-v2.2-hierarchical-warmstart-checkpoint-1"
B2_HEAD_LEARNING_RATE = 3e-4
B3_INTERVENTION_PRIOR_PROBABILITY = 0.10
B3_CONDITIONAL_BRAKE_PRIOR_PROBABILITY = 0.50
B3_INTERVENTION_PRIOR_LOGIT = math.log(
    B3_INTERVENTION_PRIOR_PROBABILITY
    / (1.0 - B3_INTERVENTION_PRIOR_PROBABILITY)
)
B3_CONDITIONAL_BRAKE_PRIOR_LOGIT = math.log(
    B3_CONDITIONAL_BRAKE_PRIOR_PROBABILITY
    / (1.0 - B3_CONDITIONAL_BRAKE_PRIOR_PROBABILITY)
)
B3_POLICY_SCHEMA = "bplus-v2.2-b3-unified-policy-1"


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
        self.intervention_logits = intervention_logits
        self.conditional_brake_logits = brake_logits
        self.intervention = torch.distributions.Bernoulli(logits=intervention_logits)
        self.steer = torch.distributions.Normal(steer_mean, steer_std)
        self.brake_gate_distribution = torch.distributions.Bernoulli(logits=brake_logits)
        self.brake = torch.distributions.Normal(brake_mean, brake_std)

    @property
    def intervention_probability(self) -> torch.Tensor:
        return self.intervention.probs

    @property
    def conditional_brake_probability(self) -> torch.Tensor:
        return self.brake_gate_distribution.probs

    @property
    def unconditional_brake_probability(self) -> torch.Tensor:
        return self.intervention_probability * self.conditional_brake_probability

    @property
    def brake_probability(self) -> torch.Tensor:
        """Historical alias for conditional ``P(BRAKE | INTERVENE)``."""

        return self.conditional_brake_probability

    def sample(self) -> HierarchicalResidualAction:
        intervention = self.intervention.sample()
        brake_gate = intervention * self.brake_gate_distribution.sample()
        return HierarchicalResidualAction(
            intervention,
            intervention * self.steer.sample(),
            brake_gate,
            brake_gate * self.brake.sample(),
        )

    def sample_from_draws(
        self, draws: KeyedComponentDraws
    ) -> HierarchicalResidualAction:
        """Sample from all four pre-generated component draws without global RNG."""

        reference = self.steer.mean
        values = (
            draws.intervention_uniform,
            draws.steer_standard_normal,
            draws.brake_uniform,
            draws.brake_standard_normal,
        )
        if any(
            value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device != reference.device
            for value in values
        ):
            raise ValueError("keyed component draws do not match distribution")
        with torch.no_grad():
            intervention = (
                draws.intervention_uniform < self.intervention_probability
            ).to(reference.dtype)
            conditional_brake = (
                draws.brake_uniform < self.conditional_brake_probability
            ).to(reference.dtype)
            brake_gate = intervention * conditional_brake
            steer = self.steer.mean + self.steer.stddev * draws.steer_standard_normal
            brake = self.brake.mean + self.brake.stddev * draws.brake_standard_normal
            return HierarchicalResidualAction(
                intervention,
                intervention * steer,
                brake_gate,
                brake_gate * brake,
            )

    def sample_keyed(
        self, keys: Sequence[ActionNoiseKey]
    ) -> HierarchicalResidualAction:
        """Sample all components from frozen, domain-separated per-macro keys."""

        return self.sample_from_draws(keyed_component_draws(keys, self.steer.mean))

    def deterministic(self) -> HierarchicalResidualAction:
        return self.deterministic_at_thresholds(0.0, 0.0)

    def deterministic_at_thresholds(
        self,
        intervention_logit_threshold: float,
        conditional_brake_logit_threshold: float,
    ) -> HierarchicalResidualAction:
        """Use strict raw-logit thresholds; equality always selects NO_OP."""

        thresholds = torch.tensor(
            [
                float(intervention_logit_threshold),
                float(conditional_brake_logit_threshold),
            ],
            dtype=torch.float64,
        )
        if not torch.all(torch.isfinite(thresholds)):
            raise ValueError("deterministic hierarchical threshold is nonfinite")
        intervention = (
            self.intervention_logits > float(intervention_logit_threshold)
        ).to(self.intervention_logits.dtype)
        brake = intervention * (
            self.conditional_brake_logits
            > float(conditional_brake_logit_threshold)
        ).to(self.conditional_brake_logits.dtype)
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
        return self.entropy_components()["total"]

    def entropy_components(self) -> dict[str, torch.Tensor]:
        """Return total entropy and the four explicit hierarchy components.

        Conditional entries are the raw conditional entropies.  ``total`` is
        the correctly probability-weighted joint entropy used by PPO.
        """

        intervention = self.intervention.entropy().squeeze(-1)
        steer = self.steer.entropy().squeeze(-1)
        brake_gate = self.brake_gate_distribution.entropy().squeeze(-1)
        brake_magnitude = self.brake.entropy().squeeze(-1)
        conditional = steer + brake_gate + self.conditional_brake_probability.squeeze(
            -1
        ) * brake_magnitude
        total = intervention + self.intervention_probability.squeeze(-1) * conditional
        values = {
            "intervention": intervention,
            "steer_given_intervention": steer,
            "brake_gate_given_intervention": brake_gate,
            "brake_magnitude_given_brake": brake_magnitude,
            "total": total,
        }
        if any(not torch.all(torch.isfinite(value)) for value in values.values()):
            raise FloatingPointError("hierarchical entropy component is nonfinite")
        return values

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

    def _distribution_parameters(self, bc_feature, lidar_history, scalar_history):
        feature = self.action_core(
            self.policy_feature(bc_feature, lidar_history, scalar_history)
        )
        steer_std = self.log_steer_std.exp().expand(len(feature), 1)
        brake_std = self.log_brake_std.exp().expand(len(feature), 1)
        return (
            self.intervention_gate(feature),
            self.steer_mean(feature),
            steer_std,
            self.brake_gate(feature),
            self.brake_mean(feature),
            brake_std,
        )

    def distribution(self, bc_feature, lidar_history, scalar_history):
        """Historical distribution, including its persistent calibration buffer."""

        top, steer, steer_std, brake_gate, brake, brake_std = (
            self._distribution_parameters(bc_feature, lidar_history, scalar_history)
        )
        return HierarchicalResidualDistribution(
            top + self.intervention_logit_offset,
            steer,
            steer_std,
            brake_gate,
            brake,
            brake_std,
        )

    def behavior_distribution(
        self,
        bc_feature: torch.Tensor,
        lidar_history: torch.Tensor,
        scalar_history: torch.Tensor,
        exploration: BehaviorExplorationBatch,
    ) -> HierarchicalResidualDistribution:
        """Construct B2 behavior probabilities without mutating policy state."""

        if not isinstance(exploration, BehaviorExplorationBatch):
            raise TypeError("B2 behavior distribution requires exploration batch")
        if not torch.equal(
            self.intervention_logit_offset,
            torch.zeros_like(self.intervention_logit_offset),
        ):
            raise ValueError("B2 behavior requires zero historical calibration offset")
        top, steer, steer_std, brake_gate, brake, brake_std = (
            self._distribution_parameters(bc_feature, lidar_history, scalar_history)
        )
        exploration.validate_like(top)
        return HierarchicalResidualDistribution(
            top + exploration.intervention_logit_offset,
            steer,
            steer_std * exploration.steer_std_scale,
            brake_gate + exploration.conditional_brake_logit_offset,
            brake,
            brake_std * exploration.brake_std_scale,
        )

    def deterministic_action(
        self,
        bc_feature: torch.Tensor,
        lidar_history: torch.Tensor,
        scalar_history: torch.Tensor,
        mode: str = DETERMINISTIC_CENTERED,
    ) -> HierarchicalResidualAction:
        """Zero-exploration primary centered action or standard-mode diagnostic."""

        if mode not in DETERMINISTIC_MODES:
            raise ValueError(f"unknown deterministic mode: {mode}")
        if not torch.equal(
            self.intervention_logit_offset,
            torch.zeros_like(self.intervention_logit_offset),
        ):
            raise ValueError("B2 deterministic action requires zero exploration offset")
        top, steer, steer_std, brake_gate, brake, brake_std = (
            self._distribution_parameters(bc_feature, lidar_history, scalar_history)
        )
        distribution = HierarchicalResidualDistribution(
            top, steer, steer_std, brake_gate, brake, brake_std
        )
        if mode == DETERMINISTIC_CENTERED:
            return distribution.deterministic_at_thresholds(
                INITIAL_INTERVENTION_LOGIT, INITIAL_BRAKE_LOGIT
            )
        if mode == DETERMINISTIC_STANDARD:
            return distribution.deterministic()
        raise AssertionError("unreachable deterministic mode")

    @property
    def primary_deterministic_mode(self) -> str:
        return DETERMINISTIC_CENTERED

    @staticmethod
    def compose(base_action, action):
        return HierarchicalResidualDistribution.compose(base_action, action).command

    def optimizer_parameter_groups(self) -> list[dict]:
        """B2 actor groups with a reachable, separately audited head LR."""

        named = dict(self.named_parameters())
        core_prefixes = ("bc_adapter.", "action_core.")
        head_prefixes = (
            "intervention_gate.",
            "steer_mean.",
            "brake_gate.",
            "brake_mean.",
        )
        core = [
            value
            for name, value in named.items()
            if value.requires_grad and name.startswith(core_prefixes)
        ]
        heads = [
            value
            for name, value in named.items()
            if value.requires_grad
            and (name.startswith(head_prefixes) or name in {"log_steer_std", "log_brake_std"})
        ]
        groups = [
            {"name": "representation_core", "params": core, "lr": ACTION_CORE_LR},
            {"name": "action_heads", "params": heads, "lr": B2_HEAD_LEARNING_RATE},
        ]
        if self.arm == ARM_SIDECAR_FINETUNE:
            sidecar = [
                value
                for name, value in named.items()
                if value.requires_grad and name.startswith("policy_sidecar.")
            ]
            groups.append(
                {"name": "sidecar", "params": sidecar, "lr": SIDECAR_FINETUNE_LR}
            )
        if any(not group["params"] for group in groups):
            raise AssertionError("B2 actor optimizer group is empty")
        ids = [id(value) for group in groups for value in group["params"]]
        expected = {id(value) for value in self.parameters() if value.requires_grad}
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise AssertionError("B2 actor optimizer groups overlap or omit trainable state")
        return groups

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


class UnifiedV22Policy(RemediatedV22Policy):
    """B3 policy whose rollout distribution and deployed mode share logits.

    Raw gate-head initialization remains byte-for-byte ``-6``.  The frozen
    prior transform moves those raw coordinates into one effective Bernoulli
    parameterization used by sampling, replay and deterministic evaluation.
    """

    policy_schema = B3_POLICY_SCHEMA

    def __init__(self, arm: str, **kwargs):
        super().__init__(arm, **kwargs)
        self.register_buffer(
            "effective_intervention_prior_logit",
            torch.tensor([B3_INTERVENTION_PRIOR_LOGIT], dtype=torch.float32),
        )
        self.register_buffer(
            "effective_conditional_brake_prior_logit",
            torch.tensor([B3_CONDITIONAL_BRAKE_PRIOR_LOGIT], dtype=torch.float32),
        )

    def _effective_distribution_parameters(
        self,
        bc_feature: torch.Tensor,
        lidar_history: torch.Tensor,
        scalar_history: torch.Tensor,
    ):
        top, steer, steer_std, brake_gate, brake, brake_std = (
            self._distribution_parameters(bc_feature, lidar_history, scalar_history)
        )
        effective_top = (
            top
            - float(INITIAL_INTERVENTION_LOGIT)
            + self.effective_intervention_prior_logit.to(top)
        )
        effective_brake = (
            brake_gate
            - float(INITIAL_BRAKE_LOGIT)
            + self.effective_conditional_brake_prior_logit.to(brake_gate)
        )
        return effective_top, steer, steer_std, effective_brake, brake, brake_std

    def _require_zero_gate_offsets(
        self, exploration: BehaviorExplorationBatch | None = None
    ) -> None:
        if not torch.equal(
            self.intervention_logit_offset,
            torch.zeros_like(self.intervention_logit_offset),
        ):
            raise ValueError("B3 unified policy forbids persistent gate offsets")
        if exploration is not None:
            if not torch.equal(
                exploration.intervention_logit_offset,
                torch.zeros_like(exploration.intervention_logit_offset),
            ) or not torch.equal(
                exploration.conditional_brake_logit_offset,
                torch.zeros_like(exploration.conditional_brake_logit_offset),
            ):
                raise ValueError("B3 unified policy forbids behavior gate offsets")

    def distribution(
        self,
        bc_feature: torch.Tensor,
        lidar_history: torch.Tensor,
        scalar_history: torch.Tensor,
    ) -> HierarchicalResidualDistribution:
        self._require_zero_gate_offsets()
        top, steer, steer_std, brake_gate, brake, brake_std = (
            self._effective_distribution_parameters(
                bc_feature, lidar_history, scalar_history
            )
        )
        return HierarchicalResidualDistribution(
            top, steer, steer_std, brake_gate, brake, brake_std
        )

    def behavior_distribution(
        self,
        bc_feature: torch.Tensor,
        lidar_history: torch.Tensor,
        scalar_history: torch.Tensor,
        exploration: BehaviorExplorationBatch,
    ) -> HierarchicalResidualDistribution:
        if not isinstance(exploration, BehaviorExplorationBatch):
            raise TypeError("B3 behavior distribution requires behavior metadata")
        top, steer, steer_std, brake_gate, brake, brake_std = (
            self._effective_distribution_parameters(
                bc_feature, lidar_history, scalar_history
            )
        )
        exploration.validate_like(top)
        self._require_zero_gate_offsets(exploration)
        return HierarchicalResidualDistribution(
            top,
            steer,
            steer_std * exploration.steer_std_scale,
            brake_gate,
            brake,
            brake_std * exploration.brake_std_scale,
        )

    def deterministic_action(
        self,
        bc_feature: torch.Tensor,
        lidar_history: torch.Tensor,
        scalar_history: torch.Tensor,
        mode: str = DETERMINISTIC_STANDARD,
    ) -> HierarchicalResidualAction:
        if mode != DETERMINISTIC_STANDARD:
            raise ValueError("B3 deterministic deployment requires standard mode")
        return self.distribution(
            bc_feature, lidar_history, scalar_history
        ).deterministic()

    @property
    def primary_deterministic_mode(self) -> str:
        return DETERMINISTIC_STANDARD

    def load_unified_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        required = {
            "intervention_gate.weight",
            "intervention_gate.bias",
            "intervention_logit_offset",
            "effective_intervention_prior_logit",
            "effective_conditional_brake_prior_logit",
        }
        missing = required - set(state)
        if missing:
            raise ValueError(
                "B3 unified checkpoint/policy mismatch; missing "
                + ", ".join(sorted(missing))
            )
        self.load_state_dict(dict(state), strict=True)
        if not torch.equal(
            self.effective_intervention_prior_logit,
            torch.tensor(
                [B3_INTERVENTION_PRIOR_LOGIT],
                dtype=self.effective_intervention_prior_logit.dtype,
                device=self.effective_intervention_prior_logit.device,
            ),
        ) or not torch.equal(
            self.effective_conditional_brake_prior_logit,
            torch.tensor(
                [B3_CONDITIONAL_BRAKE_PRIOR_LOGIT],
                dtype=self.effective_conditional_brake_prior_logit.dtype,
                device=self.effective_conditional_brake_prior_logit.device,
            ),
        ):
            raise ValueError("B3 unified checkpoint effective prior drift")
        self._require_zero_gate_offsets()


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
