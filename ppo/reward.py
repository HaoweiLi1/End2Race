"""Closed-track progress projection and the fixed PPO reward."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ppo.geometry import rectangle_clearance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_PROGRESS_DELTA_M = 1.0
PROGRESS_WEIGHT = 0.01
RELATIVE_WEIGHT = 0.02
COLLISION_PENALTY = -2.0


def proximity_risk_potential(clearance_m: float, safe_clearance_m: float, alpha: float) -> float:
    """Negative quadratic potential inside the configured vehicle-clearance boundary."""

    values = np.asarray((clearance_m, safe_clearance_m, alpha), dtype=np.float64)
    if not np.isfinite(values).all() or clearance_m < 0.0 or safe_clearance_m <= 0.0 or alpha < 0.0:
        raise ValueError(
            "Risk potential requires finite clearance >= 0, safe clearance > 0, and alpha >= 0"
        )
    shortfall = max(0.0, float(safe_clearance_m) - float(clearance_m))
    return float(-float(alpha) * shortfall * shortfall)


def wrapped_progress_delta(current_s: float, previous_s: float, track_length: float) -> float:
    """Return the shortest signed closed-track displacement."""

    return float((current_s - previous_s + 0.5 * track_length) % track_length - 0.5 * track_length)


def checked_progress_delta(
    current_s: float,
    previous_s: float,
    track_length: float,
    *,
    scenario_id: str,
    vehicle: str,
    max_abs_delta_m: float = MAX_PROGRESS_DELTA_M,
) -> float:
    """Compute a progress delta and fail loudly on invalid transitions."""

    values = np.asarray((current_s, previous_s, track_length), dtype=np.float64)
    if not np.isfinite(values).all() or track_length <= 0.0:
        raise ValueError(
            f"Invalid {vehicle} progress for scenario {scenario_id}: "
            f"previous_s={previous_s!r}, current_s={current_s!r}, track_length={track_length!r}"
        )
    delta = wrapped_progress_delta(current_s, previous_s, track_length)
    if not np.isfinite(delta) or abs(delta) > max_abs_delta_m:
        raise ValueError(
            f"Invalid {vehicle} progress delta for scenario {scenario_id}: "
            f"previous_s={previous_s!r}, current_s={current_s!r}, delta={delta!r}"
        )
    return delta


class ProgressProjector:
    """Project XY points onto cyclic raceline segments using the CSV ``s`` axis."""

    def __init__(self, progress_s: np.ndarray, xy: np.ndarray, track_length: float) -> None:
        progress_s = np.asarray(progress_s, dtype=np.float64).reshape(-1)
        xy = np.asarray(xy, dtype=np.float64)
        track_length = float(track_length)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] != progress_s.size:
            raise ValueError("progress_s and xy must describe matching 2D raceline points")
        if progress_s.size < 3 or not np.isfinite(progress_s).all() or not np.isfinite(xy).all():
            raise ValueError("Raceline reference must contain at least three finite points")
        if not np.isfinite(track_length) or track_length <= 0.0:
            raise ValueError("track_length must be finite and positive")
        if np.linalg.norm(xy[-1] - xy[0]) <= 1e-9:
            progress_s = progress_s[:-1]
            xy = xy[:-1]
        if progress_s.size < 3 or np.any(np.diff(progress_s) <= 0.0):
            raise ValueError("Raceline progress values must be strictly increasing after closing-point removal")
        if progress_s[0] < -1e-9 or progress_s[-1] >= track_length:
            raise ValueError("Raceline progress values must lie in [0, track_length)")

        self.progress_s = progress_s
        self.xy = xy
        self.track_length = track_length
        self._segment_start = xy
        self._segment_vector = np.roll(xy, -1, axis=0) - xy
        self._segment_norm_sq = np.einsum("ij,ij->i", self._segment_vector, self._segment_vector)
        if np.any(self._segment_norm_sq <= 0.0):
            raise ValueError("Cyclic raceline contains a zero-length segment")
        self._segment_progress = np.concatenate(
            (np.diff(progress_s), np.asarray([track_length - progress_s[-1]], dtype=np.float64))
        )
        if np.any(self._segment_progress <= 0.0):
            raise ValueError("Cyclic raceline contains a non-positive progress segment")

    @classmethod
    def from_csv(cls, path: str | Path) -> "ProgressProjector":
        data = np.loadtxt(Path(path), delimiter=";", comments="#", dtype=np.float64)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"Expected s/x/y columns in raceline CSV: {path}")
        return cls(data[:, 0], data[:, 1:3], float(data[-1, 0]))

    def project(self, point_xy: np.ndarray) -> float:
        point = np.asarray(point_xy, dtype=np.float64).reshape(-1)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError(f"Progress point must be a finite XY pair, got {point_xy!r}")
        offset = point - self._segment_start
        fraction = np.einsum("ij,ij->i", offset, self._segment_vector) / self._segment_norm_sq
        fraction = np.clip(fraction, 0.0, 1.0)
        closest = self._segment_start + fraction[:, None] * self._segment_vector
        distance_sq = np.einsum("ij,ij->i", point - closest, point - closest)
        index = int(np.argmin(distance_sq))
        progress = self.progress_s[index] + fraction[index] * self._segment_progress[index]
        return float(progress % self.track_length)


@dataclass(frozen=True)
class RewardResult:
    reward_progress: float
    reward_relative: float
    reward_collision: float
    reward_risk: float
    reward_total: float
    ego_progress_delta_m: float
    opponent_progress_delta_m: float
    relative_position_m: float
    ego_collision: bool
    opponent_collision: bool
    opponent_collision_latched: bool
    obb_clearance_m: float
    scenario_id: str

    def to_info(self) -> dict[str, Any]:
        return asdict(self)


class PPOTransitionReward:
    """Stateful progress/relative/collision reward with potential-based risk shaping."""

    def __init__(
        self,
        map_name: str,
        ego_raceline: str,
        projector: ProgressProjector | None = None,
        *,
        gamma: float,
        vehicle_length: float,
        vehicle_width: float,
        risk_safe_clearance_m: float,
        risk_potential_alpha: float,
    ) -> None:
        self.progress_reference_path = PROJECT_ROOT / "f1tenth_racetracks" / map_name / f"{ego_raceline}.csv"
        self.projector = projector or ProgressProjector.from_csv(self.progress_reference_path)
        self._previous_ego_progress: float | None = None
        self._previous_opponent_progress: float | None = None
        self._relative_position_m = 0.0
        self._opponent_collision_latched = False
        self._ego_collision_penalty_applied = False
        self._scenario_id: str | None = None
        self.gamma = float(gamma)
        self.vehicle_length = float(vehicle_length)
        self.vehicle_width = float(vehicle_width)
        self.risk_safe_clearance_m = float(risk_safe_clearance_m)
        self.risk_potential_alpha = float(risk_potential_alpha)
        parameters = np.asarray(
            (
                self.gamma,
                self.vehicle_length,
                self.vehicle_width,
                self.risk_safe_clearance_m,
                self.risk_potential_alpha,
            ),
            dtype=np.float64,
        )
        if (
            not np.isfinite(parameters).all()
            or not 0.0 < self.gamma <= 1.0
            or self.vehicle_length <= 0.0
            or self.vehicle_width <= 0.0
            or self.risk_safe_clearance_m <= 0.0
            or self.risk_potential_alpha < 0.0
        ):
            raise ValueError("Invalid PPO risk-potential parameters")
        self._previous_risk_potential: float | None = None
        self.current_obb_clearance_m: float | None = None

    @staticmethod
    def _position(raw_observation: dict[str, Any], index: int) -> np.ndarray:
        return np.asarray(
            (raw_observation["poses_x"][index], raw_observation["poses_y"][index]),
            dtype=np.float64,
        )

    @staticmethod
    def _heading(raw_observation: dict[str, Any], index: int) -> float:
        return float(np.asarray(raw_observation["poses_theta"])[index])

    def _obb_clearance(self, raw_observation: dict[str, Any], ego_index: int, opponent_index: int) -> float:
        from latticeplanner.utils import get_vertices

        ego_position = self._position(raw_observation, ego_index)
        opponent_position = self._position(raw_observation, opponent_index)
        ego_pose = np.asarray(
            (ego_position[0], ego_position[1], self._heading(raw_observation, ego_index)),
            dtype=np.float64,
        )
        opponent_pose = np.asarray(
            (opponent_position[0], opponent_position[1], self._heading(raw_observation, opponent_index)),
            dtype=np.float64,
        )
        return rectangle_clearance(
            get_vertices(ego_pose, self.vehicle_length, self.vehicle_width),
            get_vertices(opponent_pose, self.vehicle_length, self.vehicle_width),
        )

    def _risk_potential(self, clearance_m: float) -> float:
        return proximity_risk_potential(
            clearance_m,
            self.risk_safe_clearance_m,
            self.risk_potential_alpha,
        )

    def reset(self, raw_observation: dict[str, Any], *, scenario_id: str, ego_index: int = 0) -> None:
        num_agents = len(np.asarray(raw_observation["poses_x"]).reshape(-1))
        opponent_indices = [index for index in range(num_agents) if index != ego_index]
        if len(opponent_indices) != 1:
            raise ValueError(f"PPO reward requires exactly one opponent, got {num_agents - 1}")
        opponent_index = opponent_indices[0]
        ego_progress = self.projector.project(self._position(raw_observation, ego_index))
        opponent_progress = self.projector.project(self._position(raw_observation, opponent_index))
        self._previous_ego_progress = ego_progress
        self._previous_opponent_progress = opponent_progress
        self._relative_position_m = wrapped_progress_delta(
            ego_progress,
            opponent_progress,
            self.projector.track_length,
        )
        self._opponent_collision_latched = False
        self._ego_collision_penalty_applied = False
        self._scenario_id = str(scenario_id)
        clearance = self._obb_clearance(raw_observation, ego_index, opponent_index)
        self.current_obb_clearance_m = clearance
        self._previous_risk_potential = self._risk_potential(clearance)

    def step(
        self,
        previous_raw_observation: dict[str, Any],
        raw_observation: dict[str, Any],
        *,
        ego_collision: bool,
        opponent_collision: bool,
        terminated: bool,
        scenario_id: str,
        ego_index: int = 0,
    ) -> RewardResult:
        del previous_raw_observation  # Previous progress is initialized/reset and advanced internally.
        if (
            self._previous_ego_progress is None
            or self._previous_opponent_progress is None
            or self._previous_risk_potential is None
        ):
            raise RuntimeError("PPO reward must be reset before step")
        if self._scenario_id != str(scenario_id):
            raise ValueError(f"Reward scenario changed without reset: {self._scenario_id!r} -> {scenario_id!r}")
        num_agents = len(np.asarray(raw_observation["poses_x"]).reshape(-1))
        opponent_indices = [index for index in range(num_agents) if index != ego_index]
        if len(opponent_indices) != 1:
            raise ValueError(f"PPO reward requires exactly one opponent, got {num_agents - 1}")
        opponent_index = opponent_indices[0]

        ego_progress = self.projector.project(self._position(raw_observation, ego_index))
        opponent_progress = self.projector.project(self._position(raw_observation, opponent_index))
        ego_delta = checked_progress_delta(
            ego_progress,
            self._previous_ego_progress,
            self.projector.track_length,
            scenario_id=str(scenario_id),
            vehicle="ego",
        )
        opponent_delta = checked_progress_delta(
            opponent_progress,
            self._previous_opponent_progress,
            self.projector.track_length,
            scenario_id=str(scenario_id),
            vehicle="opponent",
        )
        self._previous_ego_progress = ego_progress
        self._previous_opponent_progress = opponent_progress
        self._relative_position_m += ego_delta - opponent_delta

        if opponent_collision:
            self._opponent_collision_latched = True
        reward_progress = PROGRESS_WEIGHT * ego_delta
        reward_relative = 0.0 if self._opponent_collision_latched else RELATIVE_WEIGHT * (ego_delta - opponent_delta)
        if ego_collision and not self._ego_collision_penalty_applied:
            reward_collision = COLLISION_PENALTY
            self._ego_collision_penalty_applied = True
        else:
            reward_collision = 0.0
        obb_clearance = self._obb_clearance(raw_observation, ego_index, opponent_index)
        self.current_obb_clearance_m = obb_clearance
        physical_risk_potential = self._risk_potential(obb_clearance)
        next_risk_potential = 0.0 if terminated else physical_risk_potential
        reward_risk = self.gamma * next_risk_potential - self._previous_risk_potential
        self._previous_risk_potential = next_risk_potential
        reward_total = reward_progress + reward_relative + reward_collision + reward_risk
        return RewardResult(
            reward_progress=float(reward_progress),
            reward_relative=float(reward_relative),
            reward_collision=float(reward_collision),
            reward_risk=float(reward_risk),
            reward_total=float(reward_total),
            ego_progress_delta_m=float(ego_delta),
            opponent_progress_delta_m=float(opponent_delta),
            relative_position_m=float(self._relative_position_m),
            ego_collision=bool(ego_collision),
            opponent_collision=bool(opponent_collision),
            opponent_collision_latched=bool(self._opponent_collision_latched),
            obb_clearance_m=float(obb_clearance),
            scenario_id=str(scenario_id),
        )
