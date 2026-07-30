"""Fixed gated post-pass penalty used by the explicit PPO treatment arm."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ppo.geometry import rectangle_clearance


@dataclass(frozen=True)
class FixedPostpassConfig:
    """Audited fixed contract; these values are intentionally not CLI options."""

    pass_margin_m: float = 0.05
    safe_rear_gap_m: float = 0.60
    closing_deadband_mps: float = 0.10
    activation_clearance_m: float = 0.20
    maximum_ego_induced_closing_time_s: float = 0.75
    reward_weight_per_m: float = 0.25
    maximum_step_penalty: float = 0.005
    maximum_episode_penalty: float = 0.05


FIXED_POSTPASS_CONFIG = FixedPostpassConfig()


def fixed_postpass_config_metadata(
    proximity_power: int = 2,
) -> dict[str, Any]:
    if proximity_power not in (1, 2):
        raise ValueError("Post-pass proximity power must be 1 or 2")
    return {
        **asdict(FIXED_POSTPASS_CONFIG),
        "proximity_power": int(proximity_power),
        "closing_time_definition": (
            "current rear-half OBB clearance divided by ego-induced rear-closing speed"
        ),
        "terminal_refund": False,
    }


@dataclass
class PostpassState:
    """Episode-local state, replaced on every reset."""

    entered: bool = False
    cleared: bool = False
    penalty_used: float = 0.0


@dataclass(frozen=True)
class PostpassStep:
    """One direct reward component and its physical diagnostics."""

    reward: float
    phase_active: bool
    triggered: bool
    entered: bool
    cleared: bool
    signed_rear_gap_m: float
    rear_half_clearance_m: float
    ego_induced_closing_m: float
    ego_induced_closing_speed_mps: float
    ego_induced_closing_time_s: float | None
    unsafe_fraction: float
    proximity_fraction: float
    penalty_basis_m: float
    episode_penalty_used: float


def _pose(name: str, value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be one finite [x, y, heading] pose")
    return pose


def _oriented_rectangle_vertices(
    pose: np.ndarray,
    length_m: float,
    width_m: float,
) -> np.ndarray:
    state = _pose("pose", pose)
    longitudinal = np.asarray(
        (np.cos(state[2]), np.sin(state[2])),
        dtype=np.float64,
    )
    lateral = np.asarray((-longitudinal[1], longitudinal[0]), dtype=np.float64)
    rear = state[:2] - 0.5 * length_m * longitudinal
    front = state[:2] + 0.5 * length_m * longitudinal
    offset = 0.5 * width_m * lateral
    return np.asarray(
        (rear + offset, rear - offset, front - offset, front + offset),
        dtype=np.float64,
    )


def _rear_half_vertices(
    pose: np.ndarray,
    length_m: float,
    width_m: float,
) -> np.ndarray:
    state = _pose("ego_pose", pose)
    longitudinal = np.asarray(
        (np.cos(state[2]), np.sin(state[2])),
        dtype=np.float64,
    )
    rear_half_pose = state.copy()
    rear_half_pose[:2] -= 0.25 * length_m * longitudinal
    return _oriented_rectangle_vertices(
        rear_half_pose,
        0.5 * length_m,
        width_m,
    )


def _signed_rear_longitudinal_gap(
    ego_pose: np.ndarray,
    opponent_pose: np.ndarray,
    length_m: float,
    width_m: float,
) -> float:
    ego = _pose("ego_pose", ego_pose)
    opponent = _pose("opponent_pose", opponent_pose)
    axis = np.asarray((np.cos(ego[2]), np.sin(ego[2])), dtype=np.float64)
    ego_vertices = _oriented_rectangle_vertices(ego, length_m, width_m)
    opponent_vertices = _oriented_rectangle_vertices(
        opponent,
        length_m,
        width_m,
    )
    return float(
        np.min(ego_vertices @ axis) - np.max(opponent_vertices @ axis)
    )


def _rear_half_clearance(
    ego_pose: np.ndarray,
    opponent_pose: np.ndarray,
    length_m: float,
    width_m: float,
) -> float:
    return rectangle_clearance(
        _rear_half_vertices(ego_pose, length_m, width_m),
        _oriented_rectangle_vertices(opponent_pose, length_m, width_m),
    )


class FixedPostpassPenalty:
    """Stateful direct penalty for ego-induced rear closing after a pass."""

    def __init__(
        self,
        *,
        vehicle_length_m: float,
        vehicle_width_m: float,
        transition_dt_s: float,
        proximity_power: int = 2,
    ) -> None:
        parameters = np.asarray(
            (vehicle_length_m, vehicle_width_m, transition_dt_s),
            dtype=np.float64,
        )
        if not np.isfinite(parameters).all() or np.any(parameters <= 0.0):
            raise ValueError(
                "Post-pass vehicle dimensions and transition_dt_s must be "
                "finite and positive"
            )
        if proximity_power not in (1, 2):
            raise ValueError("Post-pass proximity power must be 1 or 2")
        self.vehicle_length_m = float(vehicle_length_m)
        self.vehicle_width_m = float(vehicle_width_m)
        self.transition_dt_s = float(transition_dt_s)
        self.proximity_power = int(proximity_power)
        self.state = PostpassState()

    def reset(self, initial_relative_progress_m: float) -> None:
        if not np.isfinite(initial_relative_progress_m):
            raise ValueError("Initial relative progress must be finite")
        self.state = PostpassState(
            entered=bool(
                initial_relative_progress_m
                > FIXED_POSTPASS_CONFIG.pass_margin_m
            )
        )

    @property
    def penalty_used(self) -> float:
        return float(self.state.penalty_used)

    def step(
        self,
        *,
        previous_relative_progress_m: float,
        current_relative_progress_m: float,
        previous_ego_pose: np.ndarray,
        current_ego_pose: np.ndarray,
        current_opponent_pose: np.ndarray,
        opponent_collision_latched: bool,
    ) -> PostpassStep:
        relative = np.asarray(
            (
                previous_relative_progress_m,
                current_relative_progress_m,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(relative).all():
            raise ValueError("Post-pass relative progress must be finite")
        config = FIXED_POSTPASS_CONFIG
        if (
            not self.state.entered
            and not self.state.cleared
            and previous_relative_progress_m <= config.pass_margin_m
            and current_relative_progress_m > config.pass_margin_m
        ):
            self.state.entered = True

        previous_ego = _pose("previous_ego_pose", previous_ego_pose)
        current_ego = _pose("current_ego_pose", current_ego_pose)
        current_opponent = _pose(
            "current_opponent_pose",
            current_opponent_pose,
        )
        signed_rear_gap_m = _signed_rear_longitudinal_gap(
            current_ego,
            current_opponent,
            self.vehicle_length_m,
            self.vehicle_width_m,
        )
        counterfactual_previous_clearance_m = _rear_half_clearance(
            previous_ego,
            current_opponent,
            self.vehicle_length_m,
            self.vehicle_width_m,
        )
        current_clearance_m = _rear_half_clearance(
            current_ego,
            current_opponent,
            self.vehicle_length_m,
            self.vehicle_width_m,
        )
        ego_induced_closing_m = max(
            0.0,
            counterfactual_previous_clearance_m - current_clearance_m,
        )
        if (
            self.state.entered
            and not self.state.cleared
            and signed_rear_gap_m >= config.safe_rear_gap_m
        ):
            self.state.cleared = True

        phase_active = bool(
            self.state.entered
            and not self.state.cleared
            and not opponent_collision_latched
        )
        closing_speed_mps = ego_induced_closing_m / self.transition_dt_s
        closing_time_s = (
            current_clearance_m / closing_speed_mps
            if closing_speed_mps > 0.0
            else None
        )
        unsafe_fraction = float(
            np.clip(
                (config.safe_rear_gap_m - signed_rear_gap_m)
                / config.safe_rear_gap_m,
                0.0,
                1.0,
            )
        )
        proximity_fraction = float(
            np.clip(
                (config.activation_clearance_m - current_clearance_m)
                / config.activation_clearance_m,
                0.0,
                1.0,
            )
        )
        triggered = bool(
            phase_active
            and current_clearance_m < config.activation_clearance_m
            and closing_time_s is not None
            and closing_time_s
            <= config.maximum_ego_induced_closing_time_s
            and closing_speed_mps > config.closing_deadband_mps
        )
        penalty_basis_m = (
            ego_induced_closing_m
            * unsafe_fraction
            * unsafe_fraction
            * proximity_fraction**self.proximity_power
            if triggered
            else 0.0
        )
        remaining_penalty = max(
            0.0,
            config.maximum_episode_penalty - self.state.penalty_used,
        )
        penalty = min(
            config.reward_weight_per_m * penalty_basis_m,
            config.maximum_step_penalty,
            remaining_penalty,
        )
        self.state.penalty_used += penalty
        return PostpassStep(
            reward=-float(penalty),
            phase_active=phase_active,
            triggered=triggered,
            entered=bool(self.state.entered),
            cleared=bool(self.state.cleared),
            signed_rear_gap_m=float(signed_rear_gap_m),
            rear_half_clearance_m=float(current_clearance_m),
            ego_induced_closing_m=float(ego_induced_closing_m),
            ego_induced_closing_speed_mps=float(closing_speed_mps),
            ego_induced_closing_time_s=(
                None if closing_time_s is None else float(closing_time_s)
            ),
            unsafe_fraction=unsafe_fraction,
            proximity_fraction=proximity_fraction,
            penalty_basis_m=float(penalty_basis_m),
            episode_penalty_used=float(self.state.penalty_used),
        )
