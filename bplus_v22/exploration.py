"""Non-persistent B2 behavior exploration and keyed action noise.

The behavior context in this module is rollout metadata, not model state.  It
must be stored with every macro transition and supplied again when PPO
recomputes that transition's joint log probability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import torch


BEHAVIOR_EXPLORATION_SCHEMA = "bplus-v2.2-behavior-exploration-1"
KEYED_ACTION_NOISE_SCHEMA = "bplus-v2.2-keyed-action-noise-1"
KEYED_ACTION_COMPONENTS = (
    "intervention_gate",
    "steer_latent",
    "brake_gate",
    "brake_latent",
)
DETERMINISTIC_CENTERED = "centered_fresh_prior"
DETERMINISTIC_STANDARD = "standard_bernoulli"
DETERMINISTIC_MODES = (DETERMINISTIC_CENTERED, DETERMINISTIC_STANDARD)

_ACTION_NOISE_DOMAIN = b"end2race:bplus-v2.2:keyed-action-noise:v1\0"


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not torch.isfinite(torch.tensor(result, dtype=torch.float64)):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class BehaviorExplorationConfig:
    """One immutable, serializable behavior schedule entry."""

    intervention_logit_offset: float
    conditional_brake_logit_offset: float
    steer_std_scale: float
    brake_std_scale: float
    schedule_id: str
    schema: str = BEHAVIOR_EXPLORATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BEHAVIOR_EXPLORATION_SCHEMA:
            raise ValueError("behavior exploration schema mismatch")
        top = _finite_float(
            self.intervention_logit_offset, "intervention logit offset"
        )
        brake = _finite_float(
            self.conditional_brake_logit_offset,
            "conditional brake logit offset",
        )
        steer_scale = _finite_float(self.steer_std_scale, "steer std scale")
        brake_scale = _finite_float(self.brake_std_scale, "brake std scale")
        if steer_scale <= 0.0 or brake_scale <= 0.0:
            raise ValueError("behavior exploration std scales must be positive")
        if not isinstance(self.schedule_id, str) or not self.schedule_id.strip():
            raise ValueError("behavior exploration schedule_id must be non-empty")
        object.__setattr__(self, "intervention_logit_offset", top)
        object.__setattr__(self, "conditional_brake_logit_offset", brake)
        object.__setattr__(self, "steer_std_scale", steer_scale)
        object.__setattr__(self, "brake_std_scale", brake_scale)

    @classmethod
    def zero(cls, schedule_id: str = "zero") -> "BehaviorExplorationConfig":
        return cls(0.0, 0.0, 1.0, 1.0, schedule_id)

    def as_dict(self) -> dict[str, float | str]:
        return {
            "schema": self.schema,
            "intervention_logit_offset": self.intervention_logit_offset,
            "conditional_brake_logit_offset": self.conditional_brake_logit_offset,
            "steer_std_scale": self.steer_std_scale,
            "brake_std_scale": self.brake_std_scale,
            "schedule_id": self.schedule_id,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, float | str]
    ) -> "BehaviorExplorationConfig":
        expected = (
            "schema",
            "intervention_logit_offset",
            "conditional_brake_logit_offset",
            "steer_std_scale",
            "brake_std_scale",
            "schedule_id",
        )
        if tuple(value) != expected:
            raise ValueError("behavior exploration config fields/order drift")
        return cls(
            intervention_logit_offset=float(value["intervention_logit_offset"]),
            conditional_brake_logit_offset=float(
                value["conditional_brake_logit_offset"]
            ),
            steer_std_scale=float(value["steer_std_scale"]),
            brake_std_scale=float(value["brake_std_scale"]),
            schedule_id=str(value["schedule_id"]),
            schema=str(value["schema"]),
        )

    def as_batch(self, reference: torch.Tensor) -> "BehaviorExplorationBatch":
        """Materialize exact detached ``[B,1]`` tensors like ``reference``."""

        if (
            reference.ndim != 2
            or reference.shape[1] != 1
            or not torch.is_floating_point(reference)
        ):
            raise ValueError("behavior reference must be floating [B,1]")

        def full(value: float) -> torch.Tensor:
            return torch.full_like(reference, value, requires_grad=False).detach()

        return BehaviorExplorationBatch(
            intervention_logit_offset=full(self.intervention_logit_offset),
            conditional_brake_logit_offset=full(
                self.conditional_brake_logit_offset
            ),
            steer_std_scale=full(self.steer_std_scale),
            brake_std_scale=full(self.brake_std_scale),
            schedule_ids=(self.schedule_id,) * len(reference),
        )


@dataclass(frozen=True)
class BehaviorExplorationBatch:
    """Per-transition behavior context used by rollout and PPO replay."""

    intervention_logit_offset: torch.Tensor
    conditional_brake_logit_offset: torch.Tensor
    steer_std_scale: torch.Tensor
    brake_std_scale: torch.Tensor
    schedule_ids: tuple[str, ...]
    schema: str = BEHAVIOR_EXPLORATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BEHAVIOR_EXPLORATION_SCHEMA:
            raise ValueError("behavior exploration batch schema mismatch")
        values = (
            self.intervention_logit_offset,
            self.conditional_brake_logit_offset,
            self.steer_std_scale,
            self.brake_std_scale,
        )
        reference = self.intervention_logit_offset
        if reference.ndim != 2 or reference.shape[1] != 1:
            raise ValueError("behavior exploration tensors must be [B,1]")
        if any(value.shape != reference.shape for value in values):
            raise ValueError("behavior exploration tensor shapes differ")
        if any(not torch.is_floating_point(value) for value in values):
            raise ValueError("behavior exploration tensors must be floating")
        if any(value.dtype != reference.dtype or value.device != reference.device for value in values):
            raise ValueError("behavior exploration tensor dtype/device drift")
        if any(value.requires_grad for value in values):
            raise ValueError("behavior exploration tensors must be detached")
        if any(not torch.all(torch.isfinite(value)) for value in values):
            raise ValueError("behavior exploration tensor is nonfinite")
        if torch.any(self.steer_std_scale <= 0.0) or torch.any(
            self.brake_std_scale <= 0.0
        ):
            raise ValueError("behavior exploration std scales must be positive")
        if len(self.schedule_ids) != len(reference) or any(
            not isinstance(value, str) or not value.strip()
            for value in self.schedule_ids
        ):
            raise ValueError("behavior exploration schedule_ids mismatch")

    def validate_like(self, reference: torch.Tensor) -> None:
        if (
            reference.shape != self.intervention_logit_offset.shape
            or reference.dtype != self.intervention_logit_offset.dtype
            or reference.device != self.intervention_logit_offset.device
        ):
            raise ValueError("behavior exploration batch does not match policy logits")


@dataclass(frozen=True)
class ActionNoiseKey:
    """Domain-separated identity for one macro action's four random draws."""

    pilot_seed: int
    l2_id: str
    repeat: int
    macro_index: int
    schema: str = KEYED_ACTION_NOISE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != KEYED_ACTION_NOISE_SCHEMA:
            raise ValueError("keyed action-noise schema mismatch")
        for name, value in (
            ("pilot_seed", self.pilot_seed),
            ("repeat", self.repeat),
            ("macro_index", self.macro_index),
        ):
            if int(value) != value or int(value) < 0:
                raise ValueError(f"action-noise {name} must be a nonnegative integer")
        if not isinstance(self.l2_id, str) or not self.l2_id or "\0" in self.l2_id:
            raise ValueError("action-noise l2_id is invalid")


@dataclass(frozen=True)
class KeyedComponentDraws:
    """All four component draws, generated even when a branch is inactive."""

    intervention_uniform: torch.Tensor
    steer_standard_normal: torch.Tensor
    brake_uniform: torch.Tensor
    brake_standard_normal: torch.Tensor
    schema: str = KEYED_ACTION_NOISE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != KEYED_ACTION_NOISE_SCHEMA:
            raise ValueError("keyed component-draw schema mismatch")
        values = (
            self.intervention_uniform,
            self.steer_standard_normal,
            self.brake_uniform,
            self.brake_standard_normal,
        )
        reference = self.intervention_uniform
        if reference.ndim != 2 or reference.shape[1] != 1:
            raise ValueError("keyed component draws must be [B,1]")
        if any(value.shape != reference.shape for value in values):
            raise ValueError("keyed component draw shapes differ")
        if any(value.dtype != reference.dtype or value.device != reference.device for value in values):
            raise ValueError("keyed component draw dtype/device drift")
        if any(not torch.all(torch.isfinite(value)) for value in values):
            raise ValueError("keyed component draw is nonfinite")
        if torch.any((self.intervention_uniform < 0.0) | (self.intervention_uniform >= 1.0)):
            raise ValueError("intervention uniform is outside [0,1)")
        if torch.any((self.brake_uniform < 0.0) | (self.brake_uniform >= 1.0)):
            raise ValueError("brake uniform is outside [0,1)")


def _component_seed(key: ActionNoiseKey, component: str) -> int:
    if component not in KEYED_ACTION_COMPONENTS:
        raise ValueError(f"unknown action-noise component: {component}")
    digest = hashlib.sha256(_ACTION_NOISE_DOMAIN)
    for value in (
        key.schema,
        str(key.pilot_seed),
        key.l2_id,
        str(key.repeat),
        str(key.macro_index),
        component,
    ):
        digest.update(value.encode("utf-8") + b"\0")
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


def keyed_component_draws(
    keys: Sequence[ActionNoiseKey], reference: torch.Tensor
) -> KeyedComponentDraws:
    """Generate four independently keyed draws per row without global RNG use."""

    if (
        reference.ndim != 2
        or reference.shape[1] != 1
        or not torch.is_floating_point(reference)
        or len(keys) != len(reference)
    ):
        raise ValueError("keyed action-noise reference/keys must match floating [B,1]")
    if any(not isinstance(key, ActionNoiseKey) for key in keys):
        raise TypeError("keyed action-noise keys must be ActionNoiseKey values")

    def draw(component: str, *, normal: bool) -> torch.Tensor:
        rows = []
        for key in keys:
            generator = torch.Generator(device=reference.device)
            generator.manual_seed(_component_seed(key, component))
            function = torch.randn if normal else torch.rand
            rows.append(
                function(
                    (1,),
                    generator=generator,
                    dtype=reference.dtype,
                    device=reference.device,
                )
            )
        return torch.stack(rows, dim=0).detach()

    return KeyedComponentDraws(
        intervention_uniform=draw("intervention_gate", normal=False),
        steer_standard_normal=draw("steer_latent", normal=True),
        brake_uniform=draw("brake_gate", normal=False),
        brake_standard_normal=draw("brake_latent", normal=True),
    )
