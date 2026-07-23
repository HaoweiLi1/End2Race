"""Offline-only post-overtake rear-clearance reward calculations.

This module is not imported by the PPO training path. It exists only for
replaying saved evaluation traces and validating a possible future treatment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ppo.geometry import rectangle_clearance


@dataclass(frozen=True)
class EgoInducedRearClosing:
    """Rear-half clearance change caused by the ego pose transition alone."""

    counterfactual_previous_clearance_m: float
    current_clearance_m: float
    closing_m: float


def _validated_pose(name: str, pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64).reshape(-1)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be one finite [x, y, heading] pose")
    return value


def _validated_dimensions(length_m: float, width_m: float) -> tuple[float, float]:
    dimensions = np.asarray((length_m, width_m), dtype=np.float64)
    if not np.isfinite(dimensions).all() or np.any(dimensions <= 0.0):
        raise ValueError("Vehicle length and width must be finite and positive")
    return float(length_m), float(width_m)


def oriented_rectangle_vertices(
    pose: np.ndarray,
    length_m: float,
    width_m: float,
) -> np.ndarray:
    """Return rear-left, rear-right, front-right, front-left OBB vertices."""

    pose_value = _validated_pose("pose", pose)
    length, width = _validated_dimensions(length_m, width_m)
    cosine = float(np.cos(pose_value[2]))
    sine = float(np.sin(pose_value[2]))
    longitudinal = np.asarray((cosine, sine), dtype=np.float64)
    lateral = np.asarray((-sine, cosine), dtype=np.float64)
    center = pose_value[:2]
    rear = center - 0.5 * length * longitudinal
    front = center + 0.5 * length * longitudinal
    left_offset = 0.5 * width * lateral
    return np.asarray(
        (
            rear + left_offset,
            rear - left_offset,
            front - left_offset,
            front + left_offset,
        ),
        dtype=np.float64,
    )


def rear_half_vertices(
    pose: np.ndarray,
    vehicle_length_m: float,
    vehicle_width_m: float,
) -> np.ndarray:
    """Return the ego rear half as a physical OBB.

    Using the rear half, rather than the center or rear-center point, makes the
    clearance sensitive to translation, yaw, and a rear-corner sweep.
    """

    pose_value = _validated_pose("pose", pose)
    length, width = _validated_dimensions(vehicle_length_m, vehicle_width_m)
    longitudinal = np.asarray(
        (np.cos(pose_value[2]), np.sin(pose_value[2])),
        dtype=np.float64,
    )
    rear_half_pose = pose_value.copy()
    rear_half_pose[:2] -= 0.25 * length * longitudinal
    return oriented_rectangle_vertices(rear_half_pose, 0.5 * length, width)


def signed_rear_longitudinal_gap(
    ego_pose: np.ndarray,
    opponent_pose: np.ndarray,
    vehicle_length_m: float,
    vehicle_width_m: float,
) -> float:
    """Return ego-rear minus opponent-front projection in the ego body frame.

    Positive means that the complete ego rear edge is longitudinally ahead of
    the opponent front projection. Zero is edge alignment; negative means the
    vehicles still overlap longitudinally in this frame.
    """

    ego = _validated_pose("ego_pose", ego_pose)
    opponent = _validated_pose("opponent_pose", opponent_pose)
    length, width = _validated_dimensions(vehicle_length_m, vehicle_width_m)
    ego_axis = np.asarray((np.cos(ego[2]), np.sin(ego[2])), dtype=np.float64)
    ego_vertices = oriented_rectangle_vertices(ego, length, width)
    opponent_vertices = oriented_rectangle_vertices(opponent, length, width)
    ego_rear = float(np.min(ego_vertices @ ego_axis))
    opponent_front = float(np.max(opponent_vertices @ ego_axis))
    return ego_rear - opponent_front


def rear_half_clearance(
    ego_pose: np.ndarray,
    opponent_pose: np.ndarray,
    vehicle_length_m: float,
    vehicle_width_m: float,
) -> float:
    """Return exact surface clearance between ego rear-half and opponent OBB."""

    opponent = oriented_rectangle_vertices(
        opponent_pose,
        vehicle_length_m,
        vehicle_width_m,
    )
    ego_rear_half = rear_half_vertices(
        ego_pose,
        vehicle_length_m,
        vehicle_width_m,
    )
    return rectangle_clearance(ego_rear_half, opponent)


def ego_induced_rear_closing(
    previous_ego_pose: np.ndarray,
    current_ego_pose: np.ndarray,
    current_opponent_pose: np.ndarray,
    vehicle_length_m: float,
    vehicle_width_m: float,
) -> EgoInducedRearClosing:
    """Isolate rear closing caused by ego motion while holding opponent fixed.

    Both clearance calculations use the current opponent pose. Opponent motion
    therefore cannot create a false ego penalty. Only a decrease is returned;
    holding position or opening clearance has zero penalty basis.
    """

    previous_clearance = rear_half_clearance(
        previous_ego_pose,
        current_opponent_pose,
        vehicle_length_m,
        vehicle_width_m,
    )
    current_clearance = rear_half_clearance(
        current_ego_pose,
        current_opponent_pose,
        vehicle_length_m,
        vehicle_width_m,
    )
    closing = max(0.0, previous_clearance - current_clearance)
    return EgoInducedRearClosing(
        counterfactual_previous_clearance_m=float(previous_clearance),
        current_clearance_m=float(current_clearance),
        closing_m=float(closing),
    )


def rear_gap_unsafe_fraction(
    signed_rear_gap_m: float,
    safe_rear_gap_m: float,
) -> float:
    """Map signed rear-gap shortfall to [0, 1]."""

    values = np.asarray((signed_rear_gap_m, safe_rear_gap_m), dtype=np.float64)
    if not np.isfinite(values).all() or safe_rear_gap_m <= 0.0:
        raise ValueError("Rear gaps must be finite and safe_rear_gap_m positive")
    return float(
        np.clip(
            (safe_rear_gap_m - signed_rear_gap_m) / safe_rear_gap_m,
            0.0,
            1.0,
        )
    )


def postpass_penalty_basis(
    ego_rear_closing_m: float,
    signed_rear_gap_m: float,
    safe_rear_gap_m: float,
    *,
    active: bool,
) -> float:
    """Return the non-negative metre-scaled basis multiplied by reward weight.

    The eventual reward component is ``-weight * basis``. Keeping this function
    weight-free lets trace replay measure sparsity before altering training.
    """

    if not np.isfinite(ego_rear_closing_m) or ego_rear_closing_m < 0.0:
        raise ValueError("ego_rear_closing_m must be finite and non-negative")
    if not active:
        return 0.0
    unsafe_fraction = rear_gap_unsafe_fraction(
        signed_rear_gap_m,
        safe_rear_gap_m,
    )
    return float(unsafe_fraction * unsafe_fraction * ego_rear_closing_m)


def bounded_postpass_reward(
    penalty_basis_m: float,
    reward_weight_per_m: float,
    maximum_step_penalty: float,
    episode_penalty_used: float,
    maximum_episode_penalty: float,
) -> float:
    """Return one non-positive, step- and episode-capped reward component.

    ``episode_penalty_used`` is the positive magnitude accumulated before the
    current transition. The caller owns that episode state and increments it by
    ``-returned_reward``. This function is deliberately pure so the exact cap
    semantics can be replayed on saved traces before being wired into training.
    """

    values = np.asarray(
        (
            penalty_basis_m,
            reward_weight_per_m,
            maximum_step_penalty,
            episode_penalty_used,
            maximum_episode_penalty,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Post-pass reward inputs must be finite and non-negative")
    if episode_penalty_used > maximum_episode_penalty + 1e-12:
        raise ValueError("episode_penalty_used exceeds maximum_episode_penalty")
    episode_penalty_used = min(
        episode_penalty_used,
        maximum_episode_penalty,
    )
    remaining = max(0.0, maximum_episode_penalty - episode_penalty_used)
    magnitude = min(
        reward_weight_per_m * penalty_basis_m,
        maximum_step_penalty,
        remaining,
    )
    return -float(magnitude)
