"""Current-state-only physical critic features for PPO V1.2 C3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rl.ppo_reward import ProgressProjector, wrapped_progress_delta


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUSTIN_DIRECTORY = PROJECT_ROOT / "f1tenth_racetracks" / "Austin"
VEHICLE_LENGTH_M = 0.58
VEHICLE_WIDTH_M = 0.31


def _rectangle_vertices(x: float, y: float, heading: float) -> np.ndarray:
    local = np.asarray(
        [
            [VEHICLE_LENGTH_M / 2, VEHICLE_WIDTH_M / 2],
            [VEHICLE_LENGTH_M / 2, -VEHICLE_WIDTH_M / 2],
            [-VEHICLE_LENGTH_M / 2, -VEHICLE_WIDTH_M / 2],
            [-VEHICLE_LENGTH_M / 2, VEHICLE_WIDTH_M / 2],
        ],
        dtype=np.float64,
    )
    rotation = np.asarray([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    return local @ rotation.T + np.asarray([x, y])


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    vector = end - start
    denominator = float(np.dot(vector, vector))
    fraction = 0.0 if denominator == 0.0 else float(np.clip(np.dot(point - start, vector) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * vector)))


def oriented_rectangle_clearance(first_pose: np.ndarray, second_pose: np.ndarray) -> float:
    """Return zero for overlap, otherwise the exact minimum edge distance."""

    first = _rectangle_vertices(*map(float, first_pose))
    second = _rectangle_vertices(*map(float, second_pose))
    for polygon in (first, second):
        for edge_index in range(4):
            edge = polygon[(edge_index + 1) % 4] - polygon[edge_index]
            axis = np.asarray([-edge[1], edge[0]], dtype=np.float64)
            axis /= np.linalg.norm(axis)
            projection_first = first @ axis
            projection_second = second @ axis
            if projection_first.max() < projection_second.min() or projection_second.max() < projection_first.min():
                break
        else:
            continue
        break
    else:
        return 0.0
    distances = []
    for polygon_a, polygon_b in ((first, second), (second, first)):
        for point in polygon_a:
            for index in range(4):
                distances.append(_point_segment_distance(point, polygon_b[index], polygon_b[(index + 1) % 4]))
    return float(min(distances))


@dataclass(frozen=True)
class PrivilegedFeatureManifest:
    curvature_scale: float
    curvature_statistic: str = "p95_abs_austin_raceline1_unique"
    vehicle_length_m: float = VEHICLE_LENGTH_M
    vehicle_width_m: float = VEHICLE_WIDTH_M

    def to_dict(self) -> dict[str, float | str]:
        return {
            "curvature_scale": self.curvature_scale,
            "curvature_statistic": self.curvature_statistic,
            "vehicle_length_m": self.vehicle_length_m,
            "vehicle_width_m": self.vehicle_width_m,
        }


class AustinPrivilegedFeatureExtractor:
    """Build the fixed 12D feature from one current pre-action simulator state."""

    def __init__(self) -> None:
        center = np.loadtxt(AUSTIN_DIRECTORY / "raceline1.csv", delimiter=";", comments="#", dtype=np.float64)
        inner = np.loadtxt(AUSTIN_DIRECTORY / "raceline0.csv", delimiter=";", comments="#", dtype=np.float64)
        outer = np.loadtxt(AUSTIN_DIRECTORY / "raceline2.csv", delimiter=";", comments="#", dtype=np.float64)
        self.center = center[:-1]
        self.inner_xy = inner[:-1, 1:3]
        self.outer_xy = outer[:-1, 1:3]
        self.projector = ProgressProjector(center[:, 0], center[:, 1:3], float(center[-1, 0]))
        scale = float(np.percentile(np.abs(self.center[:, 4]), 95.0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Austin curvature scale must be finite and positive")
        self.manifest = PrivilegedFeatureManifest(curvature_scale=scale)

    @staticmethod
    def _array(raw: dict[str, Any], name: str) -> np.ndarray:
        value = np.asarray(raw[name], dtype=np.float64).reshape(-1)
        if len(value) < 2 or not np.isfinite(value[:2]).all():
            raise ValueError(f"Privileged field {name} must contain two finite vehicles")
        return value

    def __call__(self, raw: dict[str, Any], ego_index: int = 0) -> np.ndarray:
        if ego_index != 0:
            raise ValueError("PPO V1.2 physical feature contract requires ego index 0")
        opponent_index = 1
        x = self._array(raw, "poses_x")
        y = self._array(raw, "poses_y")
        heading = self._array(raw, "poses_theta")
        speed = self._array(raw, "linear_vels_x")
        yaw_rate = self._array(raw, "ang_vels_z")

        ego_xy = np.asarray([x[ego_index], y[ego_index]])
        opponent_xy = np.asarray([x[opponent_index], y[opponent_index]])
        ego_progress = self.projector.project(ego_xy)
        opponent_progress = self.projector.project(opponent_xy)
        relative_progress = wrapped_progress_delta(opponent_progress, ego_progress, self.projector.track_length)

        delta_xy = opponent_xy - ego_xy
        cosine, sine = np.cos(heading[ego_index]), np.sin(heading[ego_index])
        relative_longitudinal = cosine * delta_xy[0] + sine * delta_xy[1]
        relative_lateral = -sine * delta_xy[0] + cosine * delta_xy[1]
        ego_velocity = speed[ego_index] * np.asarray([cosine, sine])
        opponent_velocity = speed[opponent_index] * np.asarray([np.cos(heading[opponent_index]), np.sin(heading[opponent_index])])
        relative_velocity = opponent_velocity - ego_velocity
        relative_longitudinal_velocity = cosine * relative_velocity[0] + sine * relative_velocity[1]
        relative_lateral_velocity = -sine * relative_velocity[0] + cosine * relative_velocity[1]
        relative_heading = heading[opponent_index] - heading[ego_index]

        center_index = int(np.argmin(np.linalg.norm(self.center[:, 1:3] - ego_xy, axis=1)))
        center_xy = self.center[center_index, 1:3]
        track_heading = float(self.center[center_index, 3])
        normal = np.asarray([-np.sin(track_heading), np.cos(track_heading)])
        signed_offset = float(np.dot(ego_xy - center_xy, normal))
        inner_distance = float(np.min(np.linalg.norm(self.inner_xy - center_xy, axis=1)))
        outer_distance = float(np.min(np.linalg.norm(self.outer_xy - center_xy, axis=1)))
        local_half_width = max(0.5 * (inner_distance + outer_distance), 1e-6)
        normalized_lateral_offset = signed_offset / local_half_width
        curvature = float(self.center[center_index, 4])
        clearance = oriented_rectangle_clearance(
            np.asarray([x[ego_index], y[ego_index], heading[ego_index]]),
            np.asarray([x[opponent_index], y[opponent_index], heading[opponent_index]]),
        )

        features = np.asarray(
            [
                np.clip(relative_progress / 10.0, -1.0, 1.0),
                np.clip(relative_lateral / 2.0, -1.0, 1.0),
                np.clip(relative_longitudinal_velocity / 10.0, -1.0, 1.0),
                np.clip(relative_lateral_velocity / 5.0, -1.0, 1.0),
                np.sin(relative_heading),
                np.cos(relative_heading),
                np.clip(speed[ego_index] / 10.0, 0.0, 1.0),
                np.clip(yaw_rate[ego_index] / 5.0, -1.0, 1.0),
                np.clip(yaw_rate[opponent_index] / 5.0, -1.0, 1.0),
                np.clip(clearance / 2.0, 0.0, 1.0),
                np.clip(normalized_lateral_offset, -1.0, 1.0),
                np.tanh(curvature / self.manifest.curvature_scale),
            ],
            dtype=np.float32,
        )
        if features.shape != (12,) or not np.isfinite(features).all():
            raise ValueError("Privileged physical critic feature must be finite 12D")
        return features
