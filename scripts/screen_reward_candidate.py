"""Screen a candidate reward change offline, before spending a PPO training run.

Purpose
-------
Answer three questions without training:

1. Is the candidate **allowed** at all under the project's two hard requirements?
2. How large is the **learning signal** it would actually inject into PPO?
3. Is its gate **selective** enough that it does not punish successful behaviour?

Compliance requirements enforced in code (not prose)
----------------------------------------------------
1. **Single-stage PPO.** A candidate may only add a reward term to the existing
   production objective. Auxiliary objectives (imitation / distillation / teacher
   losses / second-stage finetuning) are rejected by ``check_compliance``.
2. **Model capability only.** A candidate may not modify actions at evaluation or
   deployment time (no shield, no post-processing, no runtime gate) and may not
   feed the actor privileged or future information.

``screen_candidate`` refuses a non-compliant candidate *before* measuring anything,
so a forbidden mechanism cannot accumulate supporting numbers.

Why raw reward magnitude is the wrong screening quantity
--------------------------------------------------------
PPO normalizes advantages per minibatch, so the learning signal produced by a
reward term is not its absolute size but its size **relative to the baseline
advantage standard deviation**, applied only on the transitions it touches. A term
whose total magnitude is a fraction of a percent of the collision penalty can still
dominate the gradient on its trigger steps if it is sparse enough. This is the
mechanism behind the historical Post-pass result, where a reward term worth about
0.42% of the collision magnitude still moved the actor by roughly 47% of the
displacement of the preceding ten updates. ``normalized_perturbation`` reports the
quantity that predicts that, which raw magnitude does not.

Independent oracle
------------------
Parts 3 and 4 are a self-contained geometry and reward implementation, migrated
from the retired standalone Post-pass shadow probe. It **deliberately imports
neither ``ppo`` nor the simulator**, including its own SAT/point-segment
``rectangle_clearance`` rather than the production reward module's. That independence is the
entire point: it exists to cross-check a production-side implementation, and
cross-checking against the code under test proves nothing.

Deliberately NOT migrated
-------------------------
The retired probe also contained ``masked_follow_teacher_huber_loss``, a masked
imitation loss. It is excluded here because a second objective violates
requirement 1 above. Its status and the reason the idea is unattractive on its own
terms are recorded in ``.agents/EXPERIMENTS.md``; do not reintroduce it without
explicit re-authorization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable, Protocol
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Production defaults; see train_ppo.py.
DEFAULT_GAMMA = 0.999
DEFAULT_GAE_LAMBDA = 0.995
DEFAULT_CLIP_RANGE = 0.20

DEPLOYED_ACTOR_INPUTS = ("lidar_360", "previous_measured_ego_speed")

FORBIDDEN_AUXILIARY_OBJECTIVES = (
    "imitation",
    "distillation",
    "teacher",
    "behaviour_cloning",
    "behavior_cloning",
    "second_stage_finetune",
)
FORBIDDEN_RUNTIME_MECHANISMS = (
    "action_override",
    "safety_shield",
    "action_post_processing",
    "runtime_gate",
    "scheduled_intervention",
)


class ComplianceError(ValueError):
    """Raised when a candidate cannot be screened because it breaks a requirement."""


# ---------------------------------------------------------------------------
# Part 1: compliance gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSpec:
    """One candidate reward change.

    ``auxiliary_objectives`` and ``runtime_mechanisms`` must be declared; an empty
    tuple is the compliant case. ``actor_inputs`` must stay the deployed contract.
    """

    name: str
    reward_terms_added: tuple[str, ...] = ()
    auxiliary_objectives: tuple[str, ...] = ()
    runtime_mechanisms: tuple[str, ...] = ()
    actor_inputs: tuple[str, ...] = DEPLOYED_ACTOR_INPUTS
    uses_future_information: bool = False
    uses_privileged_state_in_actor: bool = False
    notes: str = ""


def check_compliance(spec: CandidateSpec) -> tuple[str, ...]:
    """Return every requirement violation for ``spec``; empty means compliant."""

    violations: list[str] = []
    for objective in spec.auxiliary_objectives:
        if objective.lower() in FORBIDDEN_AUXILIARY_OBJECTIVES:
            violations.append(
                f"single-stage PPO: auxiliary objective '{objective}' is not allowed"
            )
        else:
            violations.append(
                f"single-stage PPO: undeclared auxiliary objective '{objective}'"
            )
    for mechanism in spec.runtime_mechanisms:
        violations.append(
            f"model capability only: runtime mechanism '{mechanism}' modifies "
            "deployed behaviour outside the actor"
        )
    if tuple(spec.actor_inputs) != DEPLOYED_ACTOR_INPUTS:
        violations.append(
            "model capability only: actor input contract changed from "
            f"{DEPLOYED_ACTOR_INPUTS} to {tuple(spec.actor_inputs)}"
        )
    if spec.uses_future_information:
        violations.append("model capability only: candidate consumes future information")
    if spec.uses_privileged_state_in_actor:
        violations.append(
            "model capability only: candidate feeds privileged state to the actor"
        )
    if not spec.reward_terms_added:
        violations.append(
            "nothing to screen: a candidate must add at least one reward term"
        )
    return tuple(violations)


def require_compliant(spec: CandidateSpec) -> None:
    violations = check_compliance(spec)
    if violations:
        raise ComplianceError(
            f"candidate '{spec.name}' violates project requirements:\n  - "
            + "\n  - ".join(violations)
        )


# ---------------------------------------------------------------------------
# Part 2: how a reward delta propagates into the PPO objective
# ---------------------------------------------------------------------------


def gae_advantage_delta(
    reward_delta: np.ndarray,
    episode_starts: np.ndarray,
    *,
    gamma: float = DEFAULT_GAMMA,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
) -> np.ndarray:
    """Exact advantage perturbation from ``reward_delta`` when values are fixed.

    Adding ``d_k`` at step ``k`` raises ``delta_k`` by ``d_k`` and therefore raises
    ``A_t`` by ``(gamma*lambda)**(k - t) * d_k`` for every ``t <= k`` inside the same
    episode. Credit never crosses an episode boundary.

    ``episode_starts[i]`` is True when step ``i`` is the first step of an episode,
    matching the production rollout buffer's convention.
    """

    delta = np.asarray(reward_delta, dtype=np.float64).reshape(-1)
    starts = np.asarray(episode_starts, dtype=bool).reshape(-1)
    if delta.shape != starts.shape or delta.size == 0:
        raise ValueError("reward_delta and episode_starts must be aligned and non-empty")
    if not np.isfinite(delta).all():
        raise ValueError("reward_delta must be finite")
    for value, name in ((gamma, "gamma"), (gae_lambda, "gae_lambda")):
        if not np.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must lie in (0, 1]")
    decay = float(gamma) * float(gae_lambda)
    advantage_delta = np.zeros_like(delta)
    carry = 0.0
    for index in range(delta.shape[0] - 1, -1, -1):
        if index == delta.shape[0] - 1 or starts[index + 1]:
            carry = 0.0
        advantage_delta[index] = delta[index] + decay * carry
        carry = advantage_delta[index]
    return advantage_delta


def normalized_perturbation(
    advantage_delta: np.ndarray,
    baseline_advantage_std: float,
) -> dict[str, float]:
    """Express an advantage perturbation in post-normalization units.

    PPO standardizes advantages per minibatch, so ``advantage_delta`` matters only
    relative to the baseline advantage spread. ``baseline_advantage_std`` must come
    from the control run's recorded metrics, not be assumed.
    """

    values = np.asarray(advantage_delta, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("advantage_delta must be finite")
    if not np.isfinite(baseline_advantage_std) or baseline_advantage_std <= 0.0:
        raise ValueError("baseline_advantage_std must be finite and positive")
    touched = values != 0.0
    scaled = values / float(baseline_advantage_std)
    nonzero = np.abs(scaled[touched])
    return {
        "touched_fraction": float(touched.mean()),
        "absolute_total": float(np.abs(values).sum()),
        "normalized_maximum": float(nonzero.max()) if nonzero.size else 0.0,
        "normalized_mean_on_touched": float(nonzero.mean()) if nonzero.size else 0.0,
        "normalized_root_mean_square": float(np.sqrt(np.mean(scaled * scaled))),
    }


def normalized_advantages(
    advantages: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Reproduce the production advantage normalization on plain arrays.

    PyTorch's ``std`` applies Bessel's correction, so this uses ``ddof=1``
    deliberately rather than NumPy's default population standard deviation. Getting
    this wrong silently rescales every screened magnitude.
    """

    values = np.asarray(advantages, dtype=np.float64).reshape(-1)
    valid = (
        np.ones(values.shape, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool).reshape(-1)
    )
    if values.shape != valid.shape or not valid.any() or not np.isfinite(values).all():
        raise ValueError("advantages/mask must be finite, aligned, and non-empty")
    if np.count_nonzero(valid) < 2:
        raise ValueError("advantage normalization requires at least two valid samples")
    standard_deviation = float(np.std(values[valid], ddof=1))
    output = values.copy()
    output[valid] = (values[valid] - float(np.mean(values[valid]))) / (
        standard_deviation + 1e-8
    )
    return output


def clipped_ppo_policy_loss(
    advantages: np.ndarray,
    probability_ratios: np.ndarray,
    *,
    clip_range: float = DEFAULT_CLIP_RANGE,
    mask: np.ndarray | None = None,
    normalize_advantage: bool = True,
) -> tuple[float, np.ndarray]:
    """Clipped actor surrogate on plain arrays; returns (mean loss, per-sample)."""

    advantage = np.asarray(advantages, dtype=np.float64).reshape(-1)
    ratio = np.asarray(probability_ratios, dtype=np.float64).reshape(-1)
    valid = (
        np.ones(advantage.shape, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool).reshape(-1)
    )
    if (
        advantage.shape != ratio.shape
        or advantage.shape != valid.shape
        or not valid.any()
        or not np.isfinite(advantage).all()
        or not np.isfinite(ratio).all()
        or np.any(ratio <= 0.0)
        or not np.isfinite(clip_range)
        or clip_range <= 0.0
    ):
        raise ValueError("invalid PPO surrogate inputs")
    used = normalized_advantages(advantage, valid) if normalize_advantage else advantage
    unclipped = used * ratio
    clipped = used * np.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
    samples = -np.minimum(unclipped, clipped)
    return float(np.mean(samples[valid])), samples


def clipped_policy_gradient_ceiling(
    normalized_advantage: float,
    clip_range: float = DEFAULT_CLIP_RANGE,
) -> float:
    """Largest per-transition surrogate weight this perturbation can apply.

    The clipped objective saturates the ratio at ``1 +/- clip_range``, bounding a
    single transition's contribution by ``|A| * (1 + clip_range)``. A ceiling, not
    a prediction.
    """

    if not np.isfinite(normalized_advantage):
        raise ValueError("normalized_advantage must be finite")
    if not np.isfinite(clip_range) or clip_range <= 0.0:
        raise ValueError("clip_range must be finite and positive")
    return float(abs(normalized_advantage) * (1.0 + clip_range))


def mean_squared_value_loss(
    predictions: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Un-clipped critic MSE over valid samples."""

    values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(returns, dtype=np.float64).reshape(-1)
    valid = (
        np.ones(values.shape, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool).reshape(-1)
    )
    if (
        values.shape != targets.shape
        or values.shape != valid.shape
        or not valid.any()
        or not np.isfinite(values).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("invalid value-loss inputs")
    samples = np.square(values - targets)
    return float(np.mean(samples[valid])), samples


def fixed_prediction_value_loss_delta(
    predictions: np.ndarray,
    baseline_returns: np.ndarray,
    return_delta: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Exact critic-loss delta when critic predictions are held fixed.

    Saved eval traces contain neither the rollout's critic predictions nor its
    baseline returns, so this identity is only ever validated synthetically. Do not
    use it to claim an empirical training-loss change.
    """

    values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline_returns, dtype=np.float64).reshape(-1)
    delta = np.asarray(return_delta, dtype=np.float64).reshape(-1)
    valid = (
        np.ones(values.shape, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool).reshape(-1)
    )
    if (
        values.shape != baseline.shape
        or values.shape != delta.shape
        or values.shape != valid.shape
        or not valid.any()
        or not np.isfinite(values).all()
        or not np.isfinite(baseline).all()
        or not np.isfinite(delta).all()
    ):
        raise ValueError("invalid fixed-prediction value-loss inputs")
    baseline_samples = np.square(values - baseline)
    treatment_samples = np.square(values - (baseline + delta))
    sample_delta = treatment_samples - baseline_samples
    return float(np.mean(sample_delta[valid])), sample_delta


# ---------------------------------------------------------------------------
# Part 3: independent geometry and reward oracle (imports no project module)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardConfig:
    """Fixed geometry and cap contract for one candidate treatment.

    Defaults reproduce the historical Post-pass fixed configuration. ``maximum_ttc_s``
    bounds an *ego-induced* rear closing time, not a two-body physical TTC.
    """

    vehicle_length_m: float = 0.58
    vehicle_width_m: float = 0.31
    pass_margin_m: float = 0.05
    safe_rear_gap_m: float = 0.60
    closing_deadband_mps: float = 0.10
    activation_clearance_m: float | None = 0.20
    maximum_ttc_s: float | None = 0.75
    reward_weight_per_m: float = 0.25
    maximum_step_penalty: float = 0.005
    maximum_episode_penalty: float = 0.05

    def __post_init__(self) -> None:
        mandatory = np.asarray(
            (
                self.vehicle_length_m,
                self.vehicle_width_m,
                self.pass_margin_m,
                self.safe_rear_gap_m,
                self.closing_deadband_mps,
                self.reward_weight_per_m,
                self.maximum_step_penalty,
                self.maximum_episode_penalty,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(mandatory).all():
            raise ValueError("RewardConfig values must be finite")
        if (
            self.vehicle_length_m <= 0.0
            or self.vehicle_width_m <= 0.0
            or self.pass_margin_m < 0.0
            or self.safe_rear_gap_m <= 0.0
            or self.closing_deadband_mps < 0.0
            or self.reward_weight_per_m < 0.0
            or self.maximum_step_penalty < 0.0
            or self.maximum_episode_penalty < 0.0
        ):
            raise ValueError("RewardConfig contains an invalid sign")
        for name, value in (
            ("activation_clearance_m", self.activation_clearance_m),
            ("maximum_ttc_s", self.maximum_ttc_s),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be None or finite and positive")


@dataclass
class PostpassState:
    """Episode-local state; construct a new instance on every reset."""

    entered: bool = False
    cleared: bool = False
    penalty_used: float = 0.0


@dataclass(frozen=True)
class ClosingGeometry:
    counterfactual_previous_clearance_m: float
    current_clearance_m: float
    closing_m: float


@dataclass(frozen=True)
class RewardStep:
    reward: float
    phase_active: bool
    triggered: bool
    entered: bool
    cleared: bool
    signed_rear_gap_m: float
    rear_half_clearance_m: float
    ego_induced_closing_m: float
    closing_speed_mps: float
    ttc_s: float
    unsafe_fraction: float
    proximity_fraction: float
    penalty_basis_m: float
    episode_penalty_used: float


def _pose(name: str, value: np.ndarray | Iterable[float]) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be one finite [x, y, heading] pose")
    return pose


def _dimensions(length_m: float, width_m: float) -> tuple[float, float]:
    values = np.asarray((length_m, width_m), dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("Vehicle dimensions must be finite and positive")
    return float(length_m), float(width_m)


def oriented_rectangle_vertices(
    pose: np.ndarray | Iterable[float],
    length_m: float,
    width_m: float,
) -> np.ndarray:
    """Return rear-left, rear-right, front-right, front-left vertices."""

    state = _pose("pose", pose)
    length, width = _dimensions(length_m, width_m)
    longitudinal = np.asarray((np.cos(state[2]), np.sin(state[2])), dtype=np.float64)
    lateral = np.asarray((-longitudinal[1], longitudinal[0]), dtype=np.float64)
    rear = state[:2] - 0.5 * length * longitudinal
    front = state[:2] + 0.5 * length * longitudinal
    offset = 0.5 * width * lateral
    return np.asarray(
        (rear + offset, rear - offset, front - offset, front + offset),
        dtype=np.float64,
    )


def rear_half_vertices(
    pose: np.ndarray | Iterable[float],
    length_m: float,
    width_m: float,
) -> np.ndarray:
    """Return the physical rear half of the ego OBB.

    Using the rear half rather than a point makes clearance sensitive to
    translation, yaw, and a rear-corner sweep.
    """

    state = _pose("pose", pose)
    length, width = _dimensions(length_m, width_m)
    longitudinal = np.asarray((np.cos(state[2]), np.sin(state[2])), dtype=np.float64)
    rear_half_pose = state.copy()
    rear_half_pose[:2] -= 0.25 * length * longitudinal
    return oriented_rectangle_vertices(rear_half_pose, 0.5 * length, width)


def _point_segment_vector(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    segment = end - start
    norm_sq = float(np.dot(segment, segment))
    if norm_sq <= 0.0:
        raise ValueError("Rectangle edges must have positive length")
    fraction = float(np.clip(np.dot(point - start, segment) / norm_sq, 0.0, 1.0))
    return start + fraction * segment - point


def _separating_axis_exists(first: np.ndarray, second: np.ndarray) -> bool:
    for vertices in (first, second):
        for index in range(4):
            edge = vertices[(index + 1) % 4] - vertices[index]
            axis = np.asarray((-edge[1], edge[0]), dtype=np.float64)
            projection_first = first @ axis
            projection_second = second @ axis
            if (
                projection_first.max() < projection_second.min()
                or projection_second.max() < projection_first.min()
            ):
                return True
    return False


def rectangle_clearance(first: np.ndarray, second: np.ndarray) -> float:
    """Exact OBB surface distance, zero at contact or overlap.

    Independent of the production reward geometry on purpose: this is the oracle side of a
    cross-check, so it must not share code with the implementation under test.
    """

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != (4, 2) or b.shape != (4, 2):
        raise ValueError("Rectangle clearance requires two (4, 2) arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Rectangle vertices must be finite")
    if not _separating_axis_exists(a, b):
        return 0.0
    best_sq = np.inf
    for source, target in ((a, b), (b, a)):
        for point in source:
            for index in range(4):
                vector = _point_segment_vector(
                    point, target[index], target[(index + 1) % 4]
                )
                best_sq = min(best_sq, float(np.dot(vector, vector)))
    if not np.isfinite(best_sq):
        raise RuntimeError("Failed to compute rectangle clearance")
    return float(np.sqrt(best_sq))


def signed_rear_longitudinal_gap(
    ego_pose: np.ndarray | Iterable[float],
    opponent_pose: np.ndarray | Iterable[float],
    length_m: float,
    width_m: float,
) -> float:
    """Ego-rear minus opponent-front projected on the current ego body axis."""

    ego = _pose("ego_pose", ego_pose)
    opponent = _pose("opponent_pose", opponent_pose)
    length, width = _dimensions(length_m, width_m)
    axis = np.asarray((np.cos(ego[2]), np.sin(ego[2])), dtype=np.float64)
    ego_vertices = oriented_rectangle_vertices(ego, length, width)
    opponent_vertices = oriented_rectangle_vertices(opponent, length, width)
    return float(np.min(ego_vertices @ axis) - np.max(opponent_vertices @ axis))


def rear_half_clearance(
    ego_pose: np.ndarray | Iterable[float],
    opponent_pose: np.ndarray | Iterable[float],
    length_m: float,
    width_m: float,
) -> float:
    return rectangle_clearance(
        rear_half_vertices(ego_pose, length_m, width_m),
        oriented_rectangle_vertices(opponent_pose, length_m, width_m),
    )


def ego_induced_rear_closing(
    previous_ego_pose: np.ndarray | Iterable[float],
    current_ego_pose: np.ndarray | Iterable[float],
    current_opponent_pose: np.ndarray | Iterable[float],
    length_m: float,
    width_m: float,
) -> ClosingGeometry:
    """Hold the opponent at its current pose and isolate ego-caused closing.

    Both clearances use the *current* opponent pose, so opponent motion cannot
    manufacture an ego penalty. Only a decrease counts.
    """

    previous_clearance = rear_half_clearance(
        previous_ego_pose, current_opponent_pose, length_m, width_m
    )
    current_clearance = rear_half_clearance(
        current_ego_pose, current_opponent_pose, length_m, width_m
    )
    return ClosingGeometry(
        previous_clearance,
        current_clearance,
        max(0.0, previous_clearance - current_clearance),
    )


def _unit_shortfall(value: float, safe_value: float) -> float:
    values = np.asarray((value, safe_value), dtype=np.float64)
    if not np.isfinite(values).all() or safe_value <= 0.0:
        raise ValueError("Shortfall values must be finite and safe_value positive")
    return float(np.clip((safe_value - value) / safe_value, 0.0, 1.0))


def bounded_negative_reward(
    penalty_basis_m: float,
    config: RewardConfig,
    state: PostpassState,
) -> float:
    """Non-positive, step- and episode-capped reward; mutates ``state.penalty_used``."""

    values = np.asarray((penalty_basis_m, state.penalty_used), dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Penalty basis and accumulated penalty must be non-negative")
    if state.penalty_used > config.maximum_episode_penalty + 1e-12:
        raise ValueError("Episode penalty exceeds its configured cap")
    remaining = max(0.0, config.maximum_episode_penalty - state.penalty_used)
    magnitude = min(
        config.reward_weight_per_m * penalty_basis_m,
        config.maximum_step_penalty,
        remaining,
    )
    state.penalty_used += magnitude
    return -float(magnitude)


def postpass_reward_step(
    *,
    previous_relative_progress_m: float,
    current_relative_progress_m: float,
    previous_ego_pose: np.ndarray | Iterable[float],
    current_ego_pose: np.ndarray | Iterable[float],
    current_opponent_pose: np.ndarray | Iterable[float],
    opponent_collision_latched: bool,
    transition_dt_s: float,
    config: RewardConfig,
    state: PostpassState,
) -> RewardStep:
    """Advance the candidate state machine and return one reward component.

    Phase semantics: the episode enters the post-pass phase when relative progress
    crosses ``pass_margin_m`` upward, clears permanently once the signed rear gap
    reaches ``safe_rear_gap_m``, and is suppressed entirely once the opponent has
    latched a collision (so the opponent's own loss of control is not charged to
    the ego).
    """

    scalars = np.asarray(
        (
            previous_relative_progress_m,
            current_relative_progress_m,
            transition_dt_s,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(scalars).all() or transition_dt_s <= 0.0:
        raise ValueError("Progress and transition_dt_s must be finite; dt positive")
    if (
        not state.entered
        and not state.cleared
        and previous_relative_progress_m <= config.pass_margin_m
        and current_relative_progress_m > config.pass_margin_m
    ):
        state.entered = True

    signed_gap = signed_rear_longitudinal_gap(
        current_ego_pose,
        current_opponent_pose,
        config.vehicle_length_m,
        config.vehicle_width_m,
    )
    closing = ego_induced_rear_closing(
        previous_ego_pose,
        current_ego_pose,
        current_opponent_pose,
        config.vehicle_length_m,
        config.vehicle_width_m,
    )
    if state.entered and not state.cleared and signed_gap >= config.safe_rear_gap_m:
        state.cleared = True
    phase_active = (
        state.entered and not state.cleared and not opponent_collision_latched
    )
    closing_speed = closing.closing_m / transition_dt_s
    ttc = (
        closing.current_clearance_m / closing_speed
        if closing_speed > 0.0
        else float("inf")
    )
    unsafe_fraction = _unit_shortfall(signed_gap, config.safe_rear_gap_m)
    proximity_fraction = (
        1.0
        if config.activation_clearance_m is None
        else _unit_shortfall(
            closing.current_clearance_m, config.activation_clearance_m
        )
    )
    within_ttc = config.maximum_ttc_s is None or ttc <= config.maximum_ttc_s
    within_clearance = (
        config.activation_clearance_m is None
        or closing.current_clearance_m < config.activation_clearance_m
    )
    triggered = bool(
        phase_active
        and within_clearance
        and within_ttc
        and closing_speed > config.closing_deadband_mps
    )
    basis = (
        closing.closing_m
        * unsafe_fraction
        * unsafe_fraction
        * proximity_fraction
        * proximity_fraction
        if triggered
        else 0.0
    )
    reward = bounded_negative_reward(basis, config, state) if triggered else 0.0
    return RewardStep(
        reward=reward,
        phase_active=bool(phase_active),
        triggered=triggered,
        entered=bool(state.entered),
        cleared=bool(state.cleared),
        signed_rear_gap_m=float(signed_gap),
        rear_half_clearance_m=float(closing.current_clearance_m),
        ego_induced_closing_m=float(closing.closing_m),
        closing_speed_mps=float(closing_speed),
        ttc_s=float(ttc),
        unsafe_fraction=float(unsafe_fraction),
        proximity_fraction=float(proximity_fraction),
        penalty_basis_m=float(basis),
        episode_penalty_used=float(state.penalty_used),
    )


# ---------------------------------------------------------------------------
# Part 4: cross-check protocol, selectivity selection, interval reporting
# ---------------------------------------------------------------------------


class GeometryCandidate(Protocol):
    """Minimum surface a production-side implementation must expose to be checked."""

    def signed_rear_longitudinal_gap(self, *args, **kwargs) -> float: ...
    def rear_half_clearance(self, *args, **kwargs) -> float: ...
    def ego_induced_rear_closing(self, *args, **kwargs): ...
    def bounded_postpass_reward(self, *args, **kwargs) -> float: ...


def crosscheck_geometry(
    candidate,
    *,
    case_count: int = 256,
    seed: int = 20260723,
    vehicle_length_m: float = 0.58,
    vehicle_width_m: float = 0.31,
) -> dict[str, float | int]:
    """Compare a candidate module against this oracle on random geometry/caps.

    The candidate must expose ``signed_rear_longitudinal_gap``,
    ``rear_half_clearance``, ``ego_induced_rear_closing`` and the historical
    ``bounded_postpass_reward(basis, weight, step_cap, used, episode_cap)`` API.
    This preserves the retired shadow probe's exact seed and sampling contract.
    Its recorded 256-case production-side cross-check reported a maximum absolute
    error of exactly ``0.0`` for all five quantities.
    """

    if case_count < 1:
        raise ValueError("case_count must be positive")
    rng = np.random.default_rng(seed)
    worst = {
        "signed_gap": 0.0,
        "current_clearance": 0.0,
        "counterfactual_clearance": 0.0,
        "closing": 0.0,
        "bounded_reward": 0.0,
    }
    for _ in range(case_count):
        previous_ego = rng.normal(size=3)
        current_ego = previous_ego + rng.normal(
            scale=(0.10, 0.10, 0.03),
            size=3,
        )
        opponent = rng.normal(size=3)
        mine_gap = signed_rear_longitudinal_gap(
            current_ego, opponent, vehicle_length_m, vehicle_width_m
        )
        theirs_gap = candidate.signed_rear_longitudinal_gap(
            current_ego, opponent, vehicle_length_m, vehicle_width_m
        )
        mine_closing = ego_induced_rear_closing(
            previous_ego, current_ego, opponent, vehicle_length_m, vehicle_width_m
        )
        theirs_closing = candidate.ego_induced_rear_closing(
            previous_ego, current_ego, opponent, vehicle_length_m, vehicle_width_m
        )
        worst["signed_gap"] = max(worst["signed_gap"], abs(mine_gap - theirs_gap))
        worst["current_clearance"] = max(
            worst["current_clearance"],
            abs(mine_closing.current_clearance_m - theirs_closing.current_clearance_m),
        )
        worst["counterfactual_clearance"] = max(
            worst["counterfactual_clearance"],
            abs(
                mine_closing.counterfactual_previous_clearance_m
                - theirs_closing.counterfactual_previous_clearance_m
            ),
        )
        worst["closing"] = max(
            worst["closing"], abs(mine_closing.closing_m - theirs_closing.closing_m)
        )
        basis = float(rng.uniform(0.0, 0.10))
        used = float(rng.uniform(0.0, 0.04))
        weight = 0.25
        step_cap = 0.005
        episode_cap = 0.05
        mine_state = PostpassState(penalty_used=used)
        mine_reward = bounded_negative_reward(
            basis,
            RewardConfig(
                reward_weight_per_m=weight,
                maximum_step_penalty=step_cap,
                maximum_episode_penalty=episode_cap,
            ),
            mine_state,
        )
        if hasattr(candidate, "bounded_postpass_reward"):
            theirs_reward = candidate.bounded_postpass_reward(
                basis,
                weight,
                step_cap,
                used,
                episode_cap,
            )
        elif hasattr(candidate, "bounded_negative_reward"):
            theirs_reward = candidate.bounded_negative_reward(
                basis,
                candidate.RewardConfig(
                    reward_weight_per_m=weight,
                    maximum_step_penalty=step_cap,
                    maximum_episode_penalty=episode_cap,
                ),
                candidate.PostpassState(penalty_used=used),
            )
        else:
            raise AttributeError(
                "candidate must expose bounded_postpass_reward or "
                "bounded_negative_reward with RewardConfig/PostpassState"
            )
        worst["bounded_reward"] = max(
            worst["bounded_reward"],
            abs(mine_reward - float(theirs_reward)),
        )
    return {"case_count": int(case_count), "seed": int(seed), **worst}


# Preregistered acceptance thresholds from the retired Post-pass shadow probe.
# Kept as named constants so the historical bar is not silently redefined: a new
# screen that loosens them is a different experiment and must say so.
ACCEPT_MIN_TAIL_CAPTURE = 0.90
ACCEPT_MAX_OVERTAKE_TRIGGER = 0.20
ACCEPT_MAX_FOLLOW_TRIGGER = 0.01


def postpass_gate_sweep_grid() -> tuple[RewardConfig, ...]:
    """The 41-setting gate grid the shipped Post-pass configuration was chosen from.

    One ungated baseline plus the full product of activation clearance, maximum
    ego-induced closing time, and closing deadband. Recording it prevents a future
    screen from re-proposing a point that was already measured and rejected: the
    shipped choice was clearance 0.20 m, closing time 0.75 s, deadband 0.10 m/s.
    """

    configs = [
        RewardConfig(
            activation_clearance_m=None,
            maximum_ttc_s=None,
            closing_deadband_mps=0.10,
        )
    ]
    for clearance in (0.05, 0.10, 0.15, 0.20, 0.30):
        for closing_time in (0.25, 0.50, 0.75, 1.00):
            for deadband in (0.05, 0.10):
                configs.append(
                    RewardConfig(
                        activation_clearance_m=clearance,
                        maximum_ttc_s=closing_time,
                        closing_deadband_mps=deadband,
                    )
                )
    return tuple(configs)


@dataclass(frozen=True)
class Setting:
    """One gate configuration and its offline selectivity measurement."""

    name: str
    target_capture_rate: float
    guardrail_trigger_rate: float
    acceptance_pass: bool
    extra: dict = field(default_factory=dict)


def per_panel_acceptance(
    per_panel_overtake_trigger_rates: Iterable[float],
    per_panel_follow_trigger_rates: Iterable[float],
    *,
    maximum_overtake_trigger: float = ACCEPT_MAX_OVERTAKE_TRIGGER,
    maximum_follow_trigger: float = ACCEPT_MAX_FOLLOW_TRIGGER,
) -> dict[str, float | bool]:
    """Acceptance must hold on **every** panel, not on the pooled aggregate.

    A setting can look selective in aggregate while blowing the budget on one
    panel, so the guard takes the maximum across panels. This is the rule the
    retired probe applied before marking a setting ``acceptance_pass``.
    """

    overtake = [float(x) for x in per_panel_overtake_trigger_rates]
    follow = [float(x) for x in per_panel_follow_trigger_rates]
    if not overtake or not follow:
        raise ValueError("per-panel rates must be non-empty")
    values = np.asarray(overtake + follow, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("per-panel trigger rates must be finite and lie in [0, 1]")
    worst_overtake = max(overtake)
    worst_follow = max(follow)
    return {
        "maximum_panel_overtake_trigger_rate": worst_overtake,
        "maximum_panel_follow_trigger_rate": worst_follow,
        "panel_selectivity_guard_pass": bool(
            worst_overtake <= maximum_overtake_trigger
            and worst_follow <= maximum_follow_trigger
        ),
    }


def select_setting(settings: tuple[Setting, ...]) -> dict:
    """Selectivity-first choice among settings, with an explicitly flagged fallback.

    Among settings that pass acceptance, prefer the lowest guardrail trigger rate
    (fewest penalised good episodes), breaking ties by higher target capture. If
    none passes, return the highest-capture setting but mark it as coming from
    outside the accepted set, so a caller cannot mistake a fallback for a pass.
    """

    if not settings:
        raise ValueError("at least one setting is required")
    for setting in settings:
        for value, name in (
            (setting.target_capture_rate, "target_capture_rate"),
            (setting.guardrail_trigger_rate, "guardrail_trigger_rate"),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{setting.name}: {name} must lie in [0, 1]")
    accepted = [s for s in settings if s.acceptance_pass]
    pool = accepted or list(settings)
    chosen = min(
        pool,
        key=lambda s: (s.guardrail_trigger_rate, -s.target_capture_rate)
        if accepted
        else (-s.target_capture_rate, s.guardrail_trigger_rate),
    )
    return {
        "setting": chosen.name,
        "selected_from_accepted_set": bool(accepted),
        "target_capture_rate": chosen.target_capture_rate,
        "guardrail_trigger_rate": chosen.guardrail_trigger_rate,
    }


def wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    """Two-sided 95% Wilson score interval for a rate; ``(None, None)`` if empty.

    Preferred over a normal approximation because capture and trigger rates are
    routinely measured on small cohorts where the normal interval leaves the unit
    range.
    """

    if total <= 0:
        return None, None
    if successes < 0 or successes > total:
        raise ValueError("successes must lie in [0, total]")
    probability = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


# ---------------------------------------------------------------------------
# Part 5: the screen itself
# ---------------------------------------------------------------------------


def screen_candidate(
    spec: CandidateSpec,
    reward_delta: np.ndarray,
    episode_starts: np.ndarray,
    *,
    baseline_advantage_std: float,
    settings: tuple[Setting, ...],
    maximum_guardrail_trigger_rate: float = 0.10,
    minimum_target_capture_rate: float = 0.60,
    gamma: float = DEFAULT_GAMMA,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
    clip_range: float = DEFAULT_CLIP_RANGE,
) -> dict:
    """Full screen. Raises ``ComplianceError`` before measuring anything."""

    require_compliant(spec)
    advantage_delta = gae_advantage_delta(
        reward_delta, episode_starts, gamma=gamma, gae_lambda=gae_lambda
    )
    propagation = normalized_perturbation(advantage_delta, baseline_advantage_std)
    selection = select_setting(settings)
    checks = {
        "guardrail_trigger_rate_within_budget": bool(
            selection["guardrail_trigger_rate"] <= maximum_guardrail_trigger_rate
        ),
        "target_capture_rate_sufficient": bool(
            selection["target_capture_rate"] >= minimum_target_capture_rate
        ),
        "selected_from_accepted_set": bool(selection["selected_from_accepted_set"]),
        "learning_signal_is_measurable": bool(propagation["normalized_maximum"] > 0.0),
    }
    return {
        "candidate": spec.name,
        "compliance": {
            "violations": [],
            "single_stage_ppo": True,
            "model_capability_only": True,
        },
        "propagation": propagation,
        "policy_gradient_ceiling": clipped_policy_gradient_ceiling(
            propagation["normalized_maximum"], clip_range
        ),
        "selection": selection,
        "acceptance_checks": checks,
        "ready_for_training_ab": bool(all(checks.values())),
        "scope": (
            "Offline screen only. Passing bounds the learning signal and shows the "
            "gate is selective; it does not show that training will improve safety."
        ),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline compliance and learning-signal screen for a reward candidate."
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--reward-delta-npz",
        type=Path,
        required=True,
        help="NPZ with 'reward_delta' and 'episode_starts' arrays",
    )
    parser.add_argument(
        "--baseline-advantage-std",
        type=float,
        required=True,
        help="Advantage spread of the matched control run; must be measured, not assumed",
    )
    parser.add_argument(
        "--settings-json",
        type=Path,
        required=True,
        help="JSON list of {name, target_capture_rate, guardrail_trigger_rate, acceptance_pass}",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    with np.load(args.reward_delta_npz, allow_pickle=False) as payload:
        reward_delta = payload["reward_delta"]
        episode_starts = payload["episode_starts"]
    settings = tuple(
        Setting(
            name=row["name"],
            target_capture_rate=float(row["target_capture_rate"]),
            guardrail_trigger_rate=float(row["guardrail_trigger_rate"]),
            acceptance_pass=bool(row["acceptance_pass"]),
        )
        for row in json.loads(args.settings_json.read_text(encoding="utf-8"))
    )
    spec = CandidateSpec(name=args.candidate, reward_terms_added=(args.candidate,))
    report = screen_candidate(
        spec,
        reward_delta,
        episode_starts,
        baseline_advantage_std=args.baseline_advantage_std,
        settings=settings,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
