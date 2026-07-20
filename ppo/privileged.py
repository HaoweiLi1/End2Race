"""P20 pre-action physical state features for the privileged critic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ppo.geometry import CurrentStateClearances
from ppo.reward import ProgressProjector, wrapped_progress_delta


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    "obb_longitudinal_clearance",
    "obb_lateral_clearance",
    "wall_clearance",
    "ego_steering_angle",
    "ego_slip_angle",
    "left_body_margin",
    "right_body_margin",
    "sin_track_heading_error",
    "cos_track_heading_error",
    "current_curvature",
    "lookahead_mean_curvature",
)
PRIVILEGED_FEATURE_SIZE = len(PRIVILEGED_FEATURE_NAMES)
PRIVILEGED_FEATURE_LOWS = (
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    0.0, 0.0, 0.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
)
PRIVILEGED_FEATURE_HIGHS = (1.0,) * PRIVILEGED_FEATURE_SIZE

DELTA_S_SCALE_M = 10.0
RELATIVE_LATERAL_SCALE_M = 2.0
LONGITUDINAL_VELOCITY_SCALE_MPS = 10.0
LATERAL_VELOCITY_SCALE_MPS = 5.0
EGO_SPEED_SCALE_MPS = 10.0
YAW_RATE_SCALE_RADPS = 5.0
SLIP_ANGLE_SCALE_RAD = 0.5
CURVATURE_SCALE_PERCENTILE = 95.0
CURVATURE_LOOKAHEAD_M = 1.0
CURVATURE_LOOKAHEAD_SAMPLES = 16
DYNAMIC_MODEL_SPEED_THRESHOLD_MPS = 0.5


def wrap_to_pi(angle: float) -> float:
    """Wrap one finite angle to [-pi, pi)."""

    if not np.isfinite(angle):
        raise ValueError("Angle must be finite")
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class BodyTrackState:
    """Body-aware lateral margins and heading against one local lane segment."""

    signed_offset_m: float
    width_left_m: float
    width_right_m: float
    track_heading_rad: float
    heading_error_rad: float
    lateral_extent_m: float
    left_margin_m: float
    right_margin_m: float
    normalized_left_margin: float
    normalized_right_margin: float


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

    def body_track_state(
        self,
        point_xy: np.ndarray,
        heading: float,
        vehicle_length: float,
        vehicle_width: float,
    ) -> BodyTrackState:
        """Return directional body margins and the shared local track heading."""

        point = np.asarray(point_xy, dtype=np.float64).reshape(-1)
        values = np.asarray((heading, vehicle_length, vehicle_width), dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all() or not np.isfinite(values).all():
            raise ValueError("Body-track geometry requires finite position, heading, and dimensions")
        if vehicle_length <= 0.0 or vehicle_width <= 0.0:
            raise ValueError("Vehicle length and width must be positive")

        offset = point - self.points
        fraction = np.clip(
            np.einsum("ij,ij->i", offset, self._segment_vector) / self._segment_norm_sq,
            0.0,
            1.0,
        )
        closest = self.points + fraction[:, None] * self._segment_vector
        residual = point - closest
        index = int(np.argmin(np.einsum("ij,ij->i", residual, residual)))
        direction = self._segment_vector[index] / np.sqrt(self._segment_norm_sq[index])
        signed_offset = float(direction[0] * residual[index, 1] - direction[1] * residual[index, 0])
        next_index = (index + 1) % len(self.points)
        weight = float(fraction[index])
        width_left = float((1.0 - weight) * self.width_left[index] + weight * self.width_left[next_index])
        width_right = float((1.0 - weight) * self.width_right[index] + weight * self.width_right[next_index])
        track_heading = float(np.arctan2(direction[1], direction[0]))
        heading_error = wrap_to_pi(float(heading) - track_heading)
        lateral_extent = 0.5 * (
            float(vehicle_length) * abs(float(np.sin(heading_error)))
            + float(vehicle_width) * abs(float(np.cos(heading_error)))
        )
        left_margin = width_left - signed_offset - lateral_extent
        right_margin = width_right + signed_offset - lateral_extent
        epsilon = np.finfo(np.float64).eps
        left_capacity = max(width_left - lateral_extent, epsilon)
        right_capacity = max(width_right - lateral_extent, epsilon)
        return BodyTrackState(
            signed_offset_m=signed_offset,
            width_left_m=width_left,
            width_right_m=width_right,
            track_heading_rad=track_heading,
            heading_error_rad=heading_error,
            lateral_extent_m=float(lateral_extent),
            left_margin_m=float(left_margin),
            right_margin_m=float(right_margin),
            normalized_left_margin=float(np.clip(left_margin / left_capacity, -1.0, 1.0)),
            normalized_right_margin=float(np.clip(right_margin / right_capacity, -1.0, 1.0)),
        )


class PrivilegedStateExtractor:
    """Compute the fixed P20 normalized pre-action physical state."""

    def __init__(
        self,
        map_name: str,
        ego_raceline: str,
        projector: ProgressProjector,
        vehicle_length: float,
        vehicle_width: float,
        *,
        steering_min_rad: float,
        steering_max_rad: float,
        risk_longitudinal_clearance_m: float,
        risk_lateral_clearance_m: float,
        risk_wall_clearance_m: float,
    ) -> None:
        if not ego_raceline.startswith("raceline"):
            raise ValueError(f"Cannot derive a lane boundary file from ego raceline {ego_raceline!r}")
        track_dir = PROJECT_ROOT / "f1tenth_racetracks" / map_name
        raceline = np.loadtxt(track_dir / f"{ego_raceline}.csv", delimiter=";", comments="#", dtype=np.float64)
        if raceline.ndim != 2 or raceline.shape[1] < 5 or not np.isfinite(raceline[:, (0, 4)]).all():
            raise ValueError(f"Ego raceline CSV must provide finite s and curvature columns: {ego_raceline}")
        if abs(float(raceline[-1, 0]) - projector.track_length) > 1e-6:
            raise ValueError("Ego raceline closing s must match the progress projector track length")
        if raceline.shape[0] < 4 or np.any(np.diff(raceline[:, 0]) <= 0.0):
            raise ValueError("Ego raceline progress must be strictly increasing")

        self.projector = projector
        # The CSV's last row closes the loop. Force its curvature to the first row's
        # value so interpolation is cyclic even if a future map has a noisy closing row.
        self._curvature_s = np.concatenate((raceline[:-1, 0], (projector.track_length,)))
        self._curvature = np.concatenate((raceline[:-1, 4], (raceline[0, 4],)))
        self._curvature_scale = float(
            np.percentile(np.abs(raceline[:, 4]), CURVATURE_SCALE_PERCENTILE)
        )
        if not np.isfinite(self._curvature_scale) or self._curvature_scale <= np.finfo(np.float64).eps:
            raise ValueError(f"Ego raceline must provide a positive curvature scale: {ego_raceline}")

        self.boundary = BoundaryDistanceReference(track_dir / f"{ego_raceline.replace('raceline', 'lane', 1)}.csv")
        self.vehicle_length = float(vehicle_length)
        self.vehicle_width = float(vehicle_width)
        self.steering_scale_rad = max(abs(float(steering_min_rad)), abs(float(steering_max_rad)))
        self.risk_longitudinal_clearance_m = float(risk_longitudinal_clearance_m)
        self.risk_lateral_clearance_m = float(risk_lateral_clearance_m)
        self.risk_wall_clearance_m = float(risk_wall_clearance_m)
        parameters = np.asarray(
            (
                self.vehicle_length,
                self.vehicle_width,
                self.steering_scale_rad,
                self.risk_longitudinal_clearance_m,
                self.risk_lateral_clearance_m,
                self.risk_wall_clearance_m,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(parameters).all() or np.any(parameters <= 0.0):
            raise ValueError("Privileged extractor dimensions and normalization scales must be positive")

    @property
    def curvature_scale(self) -> float:
        return self._curvature_scale

    @staticmethod
    def _agent_state(raw_observation: dict[str, Any], index: int) -> tuple[np.ndarray, float, float, float]:
        position = np.asarray(
            (raw_observation["poses_x"][index], raw_observation["poses_y"][index]), dtype=np.float64
        )
        heading = float(np.asarray(raw_observation["poses_theta"])[index])
        speed = float(np.asarray(raw_observation["linear_vels_x"])[index])
        yaw_rate = float(np.asarray(raw_observation["ang_vels_z"])[index])
        if not np.isfinite(np.concatenate((position, (heading, speed, yaw_rate)))).all():
            raise ValueError("Privileged raw simulator state must be finite")
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

    def curvature_at(self, progress_m: float) -> float:
        """Return signed raceline curvature using closed-track interpolation."""

        if not np.isfinite(progress_m):
            raise ValueError("Curvature progress must be finite")
        wrapped_progress = float(progress_m) % self.projector.track_length
        return float(np.interp(wrapped_progress, self._curvature_s, self._curvature))

    def lookahead_mean_curvature_at(self, progress_m: float) -> float:
        samples = (
            float(progress_m)
            + np.arange(1, CURVATURE_LOOKAHEAD_SAMPLES + 1, dtype=np.float64)
            * (CURVATURE_LOOKAHEAD_M / CURVATURE_LOOKAHEAD_SAMPLES)
        ) % self.projector.track_length
        return float(np.mean([self.curvature_at(sample) for sample in samples]))

    def normalization_metadata(self) -> dict[str, Any]:
        """Return the exact runtime P20 normalization contract for run records."""

        return {
            "delta_s_m": DELTA_S_SCALE_M,
            "relative_lateral_m": RELATIVE_LATERAL_SCALE_M,
            "relative_long_velocity_mps": LONGITUDINAL_VELOCITY_SCALE_MPS,
            "relative_lat_velocity_mps": LATERAL_VELOCITY_SCALE_MPS,
            "ego_speed_mps": EGO_SPEED_SCALE_MPS,
            "yaw_rate_radps": YAW_RATE_SCALE_RADPS,
            "obb_longitudinal_clearance_m": self.risk_longitudinal_clearance_m,
            "obb_lateral_clearance_m": self.risk_lateral_clearance_m,
            "wall_clearance_m": self.risk_wall_clearance_m,
            "ego_steering_angle_rad": self.steering_scale_rad,
            "ego_slip_angle_rad": SLIP_ANGLE_SCALE_RAD,
            "curvature_abs_percentile": CURVATURE_SCALE_PERCENTILE,
            "curvature_radpm": self._curvature_scale,
            "curvature_lookahead_m": CURVATURE_LOOKAHEAD_M,
            "curvature_lookahead_samples": CURVATURE_LOOKAHEAD_SAMPLES,
        }

    def features(
        self,
        raw_observation: dict[str, Any],
        *,
        ego_index: int,
        opponent_index: int,
        ego_steering_angle: float,
        ego_slip_angle: float,
        opponent_slip_angle: float,
        clearances: CurrentStateClearances,
    ) -> np.ndarray:
        ego_position, ego_heading, ego_speed, ego_yaw_rate = self._agent_state(raw_observation, ego_index)
        opponent_position, opponent_heading, opponent_speed, opponent_yaw_rate = self._agent_state(
            raw_observation, opponent_index
        )
        physical_state = np.asarray(
            (ego_steering_angle, ego_slip_angle, opponent_slip_angle),
            dtype=np.float64,
        )
        if not np.isfinite(physical_state).all():
            raise ValueError("Privileged steering and slip angles must be finite")
        if not isinstance(clearances, CurrentStateClearances):
            raise TypeError("Privileged features require reward's current-state clearance result")

        ego_progress = self.projector.project(ego_position)
        opponent_progress = self.projector.project(opponent_position)
        delta_s = wrapped_progress_delta(ego_progress, opponent_progress, self.projector.track_length)

        cos_ego, sin_ego = np.cos(ego_heading), np.sin(ego_heading)
        relative_position = opponent_position - ego_position
        relative_lateral = -sin_ego * relative_position[0] + cos_ego * relative_position[1]
        relative_heading = wrap_to_pi(opponent_heading - ego_heading)

        ego_velocity_world = self._world_velocity(ego_heading, ego_speed, ego_slip_angle)
        opponent_velocity_world = self._world_velocity(opponent_heading, opponent_speed, opponent_slip_angle)
        velocity_delta_world = opponent_velocity_world - ego_velocity_world
        relative_long_velocity = cos_ego * velocity_delta_world[0] + sin_ego * velocity_delta_world[1]
        relative_lat_velocity = -sin_ego * velocity_delta_world[0] + cos_ego * velocity_delta_world[1]

        body_track = self.boundary.body_track_state(
            ego_position,
            ego_heading,
            self.vehicle_length,
            self.vehicle_width,
        )
        current_curvature = self.curvature_at(ego_progress)
        lookahead_curvature = self.lookahead_mean_curvature_at(ego_progress)

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
                np.clip(
                    clearances.obb_longitudinal_clearance_m / self.risk_longitudinal_clearance_m,
                    0.0,
                    1.0,
                ),
                np.clip(
                    clearances.obb_lateral_clearance_m / self.risk_lateral_clearance_m,
                    0.0,
                    1.0,
                ),
                np.clip(clearances.wall_clearance_m / self.risk_wall_clearance_m, 0.0, 1.0),
                np.clip(float(ego_steering_angle) / self.steering_scale_rad, -1.0, 1.0),
                np.clip(float(ego_slip_angle) / SLIP_ANGLE_SCALE_RAD, -1.0, 1.0),
                body_track.normalized_left_margin,
                body_track.normalized_right_margin,
                np.sin(body_track.heading_error_rad),
                np.cos(body_track.heading_error_rad),
                np.clip(current_curvature / self._curvature_scale, -1.0, 1.0),
                np.clip(lookahead_curvature / self._curvature_scale, -1.0, 1.0),
            ),
            dtype=np.float32,
        )
        lows = np.asarray(PRIVILEGED_FEATURE_LOWS, dtype=np.float32)
        highs = np.asarray(PRIVILEGED_FEATURE_HIGHS, dtype=np.float32)
        if features.shape != (PRIVILEGED_FEATURE_SIZE,) or features.dtype != np.float32:
            raise RuntimeError(f"Privileged feature contract violated: shape={features.shape}, dtype={features.dtype}")
        if not np.isfinite(features).all() or np.any(features < lows) or np.any(features > highs):
            raise ValueError(f"Privileged features must be finite and within declared bounds, got {features!r}")
        return features
