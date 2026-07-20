"""Privileged pre-action physical state features for the privileged critic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ppo.geometry import rectangle_clearance
from ppo.reward import ProgressProjector, wrapped_progress_delta


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVILEGED_FEATURE_SIZE = 12
PRIVILEGED_FEATURE_NAMES = (
    "delta_s",
    "relative_lateral",
    "relative_long_velocity",
    "relative_lat_velocity",
    "sin_relative_heading",
    "cos_relative_heading",
    "ego_speed",
    "ego_yaw_rate",
    "relative_yaw_rate",
    "obb_clearance",
    "body_track_margin",
    "raceline_curvature",
)
PRIVILEGED_FEATURE_LOWS = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0)
PRIVILEGED_FEATURE_HIGHS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
DELTA_S_SCALE_M = 10.0
RELATIVE_LATERAL_SCALE_M = 2.0
LONGITUDINAL_VELOCITY_SCALE_MPS = 10.0
LATERAL_VELOCITY_SCALE_MPS = 5.0
EGO_SPEED_SCALE_MPS = 10.0
YAW_RATE_SCALE_RADPS = 5.0
CLEARANCE_SCALE_M = 2.0
CURVATURE_SCALE_PERCENTILE = 95.0
DYNAMIC_MODEL_SPEED_THRESHOLD_MPS = 0.5


class BoundaryDistanceReference:
    """Signed lateral offset and boundary widths against a cyclic lane polyline."""

    def __init__(self, lane_path: str | Path) -> None:
        data = np.loadtxt(Path(lane_path), delimiter=",", comments="#", dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 4 or data.shape[0] < 3 or not np.isfinite(data).all():
            raise ValueError(f"Expected finite x/y/w_tr_right/w_tr_left rows in lane CSV: {lane_path}")
        if np.linalg.norm(data[-1, :2] - data[0, :2]) <= 1e-9:
            data = data[:-1]
        self.points = data[:, :2]
        self.width_right = data[:, 2]
        self.width_left = data[:, 3]
        if np.any(self.width_right <= 0.0) or np.any(self.width_left <= 0.0):
            raise ValueError(f"Lane CSV boundary widths must be positive: {lane_path}")
        self._segment_vector = np.roll(self.points, -1, axis=0) - self.points
        self._segment_norm_sq = np.einsum("ij,ij->i", self._segment_vector, self._segment_vector)
        if np.any(self._segment_norm_sq <= 0.0):
            raise ValueError(f"Lane CSV contains a zero-length cyclic segment: {lane_path}")

    def normalized_body_margin(
        self,
        point_xy: np.ndarray,
        heading: float,
        vehicle_length: float,
        vehicle_width: float,
    ) -> float:
        """Return remaining lateral body clearance as a fraction of local capacity."""

        point = np.asarray(point_xy, dtype=np.float64).reshape(2)
        offset = point - self.points
        fraction = np.clip(np.einsum("ij,ij->i", offset, self._segment_vector) / self._segment_norm_sq, 0.0, 1.0)
        closest = self.points + fraction[:, None] * self._segment_vector
        residual = point - closest
        index = int(np.argmin(np.einsum("ij,ij->i", residual, residual)))
        direction = self._segment_vector[index] / np.sqrt(self._segment_norm_sq[index])
        lateral = residual[index]
        signed_offset = direction[0] * lateral[1] - direction[1] * lateral[0]
        next_index = (index + 1) % len(self.points)
        weight = fraction[index]
        width_left = (1.0 - weight) * self.width_left[index] + weight * self.width_left[next_index]
        width_right = (1.0 - weight) * self.width_right[index] + weight * self.width_right[next_index]
        lane_heading = float(np.arctan2(direction[1], direction[0]))
        relative_heading = float(heading) - lane_heading
        lateral_extent = 0.5 * (
            float(vehicle_length) * abs(float(np.sin(relative_heading)))
            + float(vehicle_width) * abs(float(np.cos(relative_heading)))
        )
        body_margin = min(
            width_left - signed_offset - lateral_extent,
            width_right + signed_offset - lateral_extent,
        )
        local_capacity = max(min(width_left, width_right) - lateral_extent, np.finfo(np.float64).eps)
        return float(np.clip(body_margin / local_capacity, 0.0, 1.0))


class PrivilegedStateExtractor:
    """Compute the 12D normalized pre-action physical state for the privileged critic."""

    def __init__(
        self,
        map_name: str,
        ego_raceline: str,
        projector: ProgressProjector,
        vehicle_length: float,
        vehicle_width: float,
    ) -> None:
        if not ego_raceline.startswith("raceline"):
            raise ValueError(f"Cannot derive a lane boundary file from ego raceline {ego_raceline!r}")
        track_dir = PROJECT_ROOT / "f1tenth_racetracks" / map_name
        raceline = np.loadtxt(track_dir / f"{ego_raceline}.csv", delimiter=";", comments="#", dtype=np.float64)
        if raceline.ndim != 2 or raceline.shape[1] < 5 or not np.isfinite(raceline[:, (0, 4)]).all():
            raise ValueError(f"Ego raceline CSV must provide finite s and curvature columns: {ego_raceline}")
        if abs(float(raceline[-1, 0]) - projector.track_length) > 1e-6:
            raise ValueError("Ego raceline closing s must match the progress projector track length")
        self.projector = projector
        self._curvature_s = raceline[:, 0]
        self._curvature = raceline[:, 4]
        self._curvature_scale = float(
            np.percentile(np.abs(self._curvature), CURVATURE_SCALE_PERCENTILE)
        )
        if not np.isfinite(self._curvature_scale) or self._curvature_scale <= np.finfo(np.float64).eps:
            raise ValueError(f"Ego raceline must provide a positive curvature scale: {ego_raceline}")
        self.boundary = BoundaryDistanceReference(track_dir / f"{ego_raceline.replace('raceline', 'lane', 1)}.csv")
        self.vehicle_length = float(vehicle_length)
        self.vehicle_width = float(vehicle_width)
        if not (self.vehicle_length > 0.0 and self.vehicle_width > 0.0):
            raise ValueError("Vehicle length and width must be positive")

    @staticmethod
    def _agent_state(raw_observation: dict[str, Any], index: int) -> tuple[np.ndarray, float, float, float]:
        position = np.asarray(
            (raw_observation["poses_x"][index], raw_observation["poses_y"][index]), dtype=np.float64
        )
        heading = float(np.asarray(raw_observation["poses_theta"])[index])
        speed = float(np.asarray(raw_observation["linear_vels_x"])[index])
        yaw_rate = float(np.asarray(raw_observation["ang_vels_z"])[index])
        return position, heading, speed, yaw_rate

    @staticmethod
    def _world_velocity(heading: float, speed: float, slip_angle: float) -> np.ndarray:
        """Match the simulator's kinematic/dynamic velocity direction switch."""

        velocity_heading = float(heading)
        if abs(float(speed)) >= DYNAMIC_MODEL_SPEED_THRESHOLD_MPS:
            velocity_heading += float(slip_angle)
        return float(speed) * np.asarray(
            (np.cos(velocity_heading), np.sin(velocity_heading)),
            dtype=np.float64,
        )

    def _obb_clearance(self, ego_pose: np.ndarray, opponent_pose: np.ndarray) -> float:
        from latticeplanner.utils import get_vertices

        ego_vertices = get_vertices(ego_pose, self.vehicle_length, self.vehicle_width)
        opponent_vertices = get_vertices(opponent_pose, self.vehicle_length, self.vehicle_width)
        return rectangle_clearance(ego_vertices, opponent_vertices)

    def features(
        self,
        raw_observation: dict[str, Any],
        *,
        ego_index: int,
        opponent_index: int,
        ego_slip_angle: float,
        opponent_slip_angle: float,
        obb_clearance_m: float | None = None,
    ) -> np.ndarray:
        ego_position, ego_heading, ego_speed, ego_yaw_rate = self._agent_state(raw_observation, ego_index)
        opponent_position, opponent_heading, opponent_speed, opponent_yaw_rate = self._agent_state(raw_observation, opponent_index)

        if not np.isfinite((ego_slip_angle, opponent_slip_angle)).all():
            raise ValueError("Privileged slip angles must be finite")

        ego_progress = self.projector.project(ego_position)
        opponent_progress = self.projector.project(opponent_position)
        delta_s = wrapped_progress_delta(ego_progress, opponent_progress, self.projector.track_length)

        cos_ego, sin_ego = np.cos(ego_heading), np.sin(ego_heading)
        relative_position = opponent_position - ego_position
        relative_lateral = -sin_ego * relative_position[0] + cos_ego * relative_position[1]

        relative_heading = opponent_heading - ego_heading
        ego_velocity_world = self._world_velocity(ego_heading, ego_speed, ego_slip_angle)
        opponent_velocity_world = self._world_velocity(
            opponent_heading,
            opponent_speed,
            opponent_slip_angle,
        )
        velocity_delta_world = opponent_velocity_world - ego_velocity_world
        relative_long_velocity = cos_ego * velocity_delta_world[0] + sin_ego * velocity_delta_world[1]
        relative_lat_velocity = -sin_ego * velocity_delta_world[0] + cos_ego * velocity_delta_world[1]

        if obb_clearance_m is None:
            clearance = self._obb_clearance(
                np.asarray((ego_position[0], ego_position[1], ego_heading), dtype=np.float64),
                np.asarray((opponent_position[0], opponent_position[1], opponent_heading), dtype=np.float64),
            )
        else:
            clearance = float(obb_clearance_m)
            if not np.isfinite(clearance) or clearance < 0.0:
                raise ValueError(f"OBB clearance must be finite and non-negative, got {clearance!r}")
        margin = self.boundary.normalized_body_margin(
            ego_position,
            ego_heading,
            self.vehicle_length,
            self.vehicle_width,
        )
        curvature = float(np.interp(ego_progress, self._curvature_s, self._curvature))

        features = np.asarray(
            (
                np.clip(delta_s / DELTA_S_SCALE_M, -1.0, 1.0),
                np.clip(relative_lateral / RELATIVE_LATERAL_SCALE_M, -1.0, 1.0),
                np.clip(relative_long_velocity / LONGITUDINAL_VELOCITY_SCALE_MPS, -1.0, 1.0),
                np.clip(relative_lat_velocity / LATERAL_VELOCITY_SCALE_MPS, -1.0, 1.0),
                np.sin(relative_heading),
                np.cos(relative_heading),
                np.clip(ego_speed / EGO_SPEED_SCALE_MPS, -1.0, 1.0),
                np.clip(ego_yaw_rate / YAW_RATE_SCALE_RADPS, -1.0, 1.0),
                np.clip((opponent_yaw_rate - ego_yaw_rate) / YAW_RATE_SCALE_RADPS, -1.0, 1.0),
                np.clip(clearance / CLEARANCE_SCALE_M, 0.0, 1.0),
                margin,
                np.clip(curvature / self._curvature_scale, -1.0, 1.0),
            ),
            dtype=np.float32,
        )
        if not np.isfinite(features).all():
            raise ValueError(f"Privileged features must be finite, got {features!r}")
        return features
