"""Self-contained post-pass reward and PPO-loss shadow contract.

This module deliberately imports neither ``ppo`` nor the simulator.  It is an
independent oracle used to test candidate reward semantics without changing the
training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    """Fixed geometry and cap contract for one candidate treatment."""

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
    """Rear-half clearance change caused by the ego transition alone."""

    counterfactual_previous_clearance_m: float
    current_clearance_m: float
    closing_m: float


@dataclass(frozen=True)
class RewardStep:
    """Fully observable result for one transition."""

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
    longitudinal = np.asarray(
        (np.cos(state[2]), np.sin(state[2])),
        dtype=np.float64,
    )
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
    """Return the physical rear half of the ego OBB."""

    state = _pose("pose", pose)
    length, width = _dimensions(length_m, width_m)
    longitudinal = np.asarray(
        (np.cos(state[2]), np.sin(state[2])),
        dtype=np.float64,
    )
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
    """Return exact OBB surface distance, with zero at contact/overlap."""

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
                    point,
                    target[index],
                    target[(index + 1) % 4],
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
    """Return ego-rear minus opponent-front in the current ego body frame."""

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
    """Hold the opponent at its current pose and isolate ego-caused closing."""

    previous_clearance = rear_half_clearance(
        previous_ego_pose,
        current_opponent_pose,
        length_m,
        width_m,
    )
    current_clearance = rear_half_clearance(
        current_ego_pose,
        current_opponent_pose,
        length_m,
        width_m,
    )
    closing = max(0.0, previous_clearance - current_clearance)
    return ClosingGeometry(previous_clearance, current_clearance, closing)


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
    values = np.asarray(
        (penalty_basis_m, state.penalty_used),
        dtype=np.float64,
    )
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
    """Advance the candidate state machine and return one reward component."""

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
    if (
        state.entered
        and not state.cleared
        and signed_gap >= config.safe_rear_gap_m
    ):
        state.cleared = True
    phase_active = state.entered and not state.cleared and not opponent_collision_latched
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
            closing.current_clearance_m,
            config.activation_clearance_m,
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


def gae_delta_from_reward_delta(
    reward_delta: np.ndarray,
    episode_end: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    """Exact GAE/return delta when values are held fixed.

    Each ``episode_end[t]`` denotes that transition ``t`` is the final
    transition of its episode. Reward changes never propagate across it.
    """

    rewards = np.asarray(reward_delta, dtype=np.float64).reshape(-1)
    endings = np.asarray(episode_end, dtype=bool).reshape(-1)
    if rewards.shape != endings.shape or rewards.size == 0:
        raise ValueError("reward_delta and episode_end must be aligned and non-empty")
    if not np.isfinite(rewards).all():
        raise ValueError("reward_delta must be finite")
    if (
        not np.isfinite(gamma)
        or not np.isfinite(gae_lambda)
        or not 0.0 <= gamma <= 1.0
        or not 0.0 <= gae_lambda <= 1.0
    ):
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    output = np.empty_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if endings[index]:
            running = 0.0
        running = rewards[index] + gamma * gae_lambda * running
        output[index] = running
    return output


def normalized_advantages(
    advantages: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Match ``torch.std`` normalization used by ``ppo/algorithm.py``.

    PyTorch's default correction is Bessel's correction (``N - 1``), so this
    deliberately uses ``ddof=1`` rather than NumPy's default population
    standard deviation.
    """

    values = np.asarray(advantages, dtype=np.float64).reshape(-1)
    valid = (
        np.ones(values.shape, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool).reshape(-1)
    )
    if values.shape != valid.shape or not valid.any() or not np.isfinite(values).all():
        raise ValueError("Advantages/mask must be finite, aligned, and non-empty")
    if np.count_nonzero(valid) < 2:
        raise ValueError(
            "Production advantage normalization requires at least two valid samples"
        )
    standard_deviation = float(np.std(values[valid], ddof=1))
    output = values.copy()
    output[valid] = (
        values[valid] - float(np.mean(values[valid]))
    ) / (standard_deviation + 1e-8)
    return output


def clipped_ppo_policy_loss(
    advantages: np.ndarray,
    probability_ratios: np.ndarray,
    *,
    clip_range: float,
    mask: np.ndarray | None = None,
    normalize_advantage: bool = True,
) -> tuple[float, np.ndarray]:
    """Match the repository's clipped actor surrogate on plain arrays."""

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
        raise ValueError("Invalid PPO surrogate inputs")
    used_advantage = (
        normalized_advantages(advantage, valid)
        if normalize_advantage
        else advantage
    )
    unclipped = used_advantage * ratio
    clipped = used_advantage * np.clip(
        ratio,
        1.0 - clip_range,
        1.0 + clip_range,
    )
    samples = -np.minimum(unclipped, clipped)
    return float(np.mean(samples[valid])), samples


def mean_squared_value_loss(
    predictions: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Match the repository's un-clipped critic MSE on valid samples."""

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
        raise ValueError("Invalid value-loss inputs")
    samples = np.square(values - targets)
    return float(np.mean(samples[valid])), samples


def fixed_prediction_value_loss_delta(
    predictions: np.ndarray,
    baseline_returns: np.ndarray,
    return_delta: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Exact critic-loss delta when critic predictions are held fixed.

    Eval traces do not contain the production rollout's critic predictions or
    baseline returns, so this identity is validated synthetically and is not
    used to fabricate an empirical training-loss claim.
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
        raise ValueError("Invalid fixed-prediction value-loss inputs")
    baseline_samples = np.square(values - baseline)
    treatment_samples = np.square(values - (baseline + delta))
    sample_delta = treatment_samples - baseline_samples
    return float(np.mean(sample_delta[valid])), sample_delta


def _huber(value: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.abs(value)
    quadratic = absolute <= delta
    loss = np.where(
        quadratic,
        0.5 * value * value,
        delta * (absolute - 0.5 * delta),
    )
    gradient = np.where(quadratic, value, delta * np.sign(value))
    return loss, gradient


def masked_follow_teacher_huber_loss(
    student_actions: np.ndarray,
    teacher_actions: np.ndarray,
    active_mask: np.ndarray,
    *,
    action_scales: tuple[float, float] = (0.52, 10.0),
    component_weights: tuple[float, float] = (1.0, 1.0),
    huber_delta: float = 0.10,
) -> tuple[float, np.ndarray]:
    """Reference-only masked imitation loss and analytic student gradient.

    This function exists to test the proposed follow-teacher mechanism. It is
    not evidence that the BC teacher is safe in the masked states.
    """

    student = np.asarray(student_actions, dtype=np.float64)
    teacher = np.asarray(teacher_actions, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    scales = np.asarray(action_scales, dtype=np.float64)
    weights = np.asarray(component_weights, dtype=np.float64)
    if (
        student.ndim != 2
        or student.shape[1] != 2
        or student.shape != teacher.shape
        or student.shape[0] != active.shape[0]
        or not np.isfinite(student).all()
        or not np.isfinite(teacher).all()
        or not np.isfinite(scales).all()
        or not np.isfinite(weights).all()
        or np.any(scales <= 0.0)
        or np.any(weights < 0.0)
        or not np.any(weights > 0.0)
        or not np.isfinite(huber_delta)
        or huber_delta <= 0.0
    ):
        raise ValueError("Invalid follow-teacher loss inputs")
    gradient = np.zeros_like(student)
    if not active.any():
        return 0.0, gradient
    normalized_error = (student - teacher) / scales
    element_loss, element_gradient = _huber(normalized_error, huber_delta)
    denominator = float(active.sum()) * float(weights.sum())
    weighted_loss = element_loss * weights
    gradient[active] = (
        element_gradient[active] * weights / scales / denominator
    )
    return float(np.sum(weighted_loss[active]) / denominator), gradient
