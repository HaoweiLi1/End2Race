"""Small exact geometry helpers shared by PPO reward and privileged state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class CurrentStateClearances:
    """One current-state geometry result shared by reward and privileged critic."""

    obb_clearance_m: float
    obb_longitudinal_clearance_m: float
    obb_lateral_clearance_m: float
    wall_clearance_m: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.obb_clearance_m,
                self.obb_longitudinal_clearance_m,
                self.obb_lateral_clearance_m,
                self.wall_clearance_m,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("Current-state clearances must be finite and non-negative")


def _point_segment_vector(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    norm_sq = float(np.dot(segment, segment))
    if norm_sq <= 0.0:
        raise ValueError("Rectangle edges must have positive length")
    fraction = np.clip(np.dot(point - start, segment) / norm_sq, 0.0, 1.0)
    return start + fraction * segment - point


def _separating_axis_exists(vertices_a: np.ndarray, vertices_b: np.ndarray) -> bool:
    for vertices in (vertices_a, vertices_b):
        for index in range(len(vertices)):
            edge = vertices[(index + 1) % len(vertices)] - vertices[index]
            axis = np.asarray((-edge[1], edge[0]), dtype=np.float64)
            projection_a = vertices_a @ axis
            projection_b = vertices_b @ axis
            if projection_a.max() < projection_b.min() or projection_b.max() < projection_a.min():
                return True
    return False


def _validated_rectangles(vertices_a: np.ndarray, vertices_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(vertices_a, dtype=np.float64)
    second = np.asarray(vertices_b, dtype=np.float64)
    if first.shape != (4, 2) or second.shape != (4, 2) or not np.isfinite((first, second)).all():
        raise ValueError("Rectangle clearance requires two finite (4, 2) vertex arrays")
    return first, second


def _rectangle_separation_vector(vertices_a: np.ndarray, vertices_b: np.ndarray) -> np.ndarray:
    first, second = _validated_rectangles(vertices_a, vertices_b)
    if not _separating_axis_exists(first, second):
        return np.zeros(2, dtype=np.float64)
    best_vector = None
    best_distance_sq = np.inf
    for vertices_from, vertices_to in ((first, second), (second, first)):
        for point in vertices_from:
            for index in range(len(vertices_to)):
                vector = _point_segment_vector(
                    point,
                    vertices_to[index],
                    vertices_to[(index + 1) % len(vertices_to)],
                )
                distance_sq = float(np.dot(vector, vector))
                if distance_sq < best_distance_sq:
                    best_distance_sq = distance_sq
                    best_vector = vector
    if best_vector is None:
        raise RuntimeError("Failed to compute rectangle separation")
    return best_vector


def rectangle_clearance(vertices_a: np.ndarray, vertices_b: np.ndarray) -> float:
    """Return exact surface distance between convex quadrilaterals, zero at contact/overlap."""

    return float(np.linalg.norm(_rectangle_separation_vector(vertices_a, vertices_b)))


def rectangle_clearance_components(
    vertices_a: np.ndarray,
    vertices_b: np.ndarray,
    reference_heading: float,
) -> tuple[float, float, float]:
    """Return surface clearance and its longitudinal/lateral components in one body frame."""

    if not np.isfinite(reference_heading):
        raise ValueError("Reference heading must be finite")
    vector = _rectangle_separation_vector(vertices_a, vertices_b)
    longitudinal_axis = np.asarray((np.cos(reference_heading), np.sin(reference_heading)))
    lateral_axis = np.asarray((-np.sin(reference_heading), np.cos(reference_heading)))
    longitudinal = abs(float(np.dot(vector, longitudinal_axis)))
    lateral = abs(float(np.dot(vector, lateral_axis)))
    return float(np.linalg.norm(vector)), longitudinal, lateral


class OccupancyMapClearance:
    """Approximate OBB-to-wall surface clearance using the simulator distance field."""

    def __init__(self, distance_field: np.ndarray, resolution: float, origin: np.ndarray) -> None:
        field = np.asarray(distance_field, dtype=np.float64)
        origin_array = np.asarray(origin, dtype=np.float64).reshape(-1)
        if field.ndim != 2 or field.size == 0 or not np.isfinite(field).all() or np.any(field < 0.0):
            raise ValueError("Map distance field must be a finite non-negative 2D array")
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("Map resolution must be finite and positive")
        if origin_array.shape != (3,) or not np.isfinite(origin_array).all():
            raise ValueError("Map origin must contain finite x, y, and heading")
        self.distance_field = field
        self.resolution = float(resolution)
        self.origin = origin_array

    def rectangle_clearance(self, vertices: np.ndarray) -> float:
        """Return conservative sampled distance from an OBB perimeter to occupied map cells."""

        rectangle = np.asarray(vertices, dtype=np.float64)
        if rectangle.shape != (4, 2) or not np.isfinite(rectangle).all():
            raise ValueError("Map clearance requires one finite (4, 2) rectangle")
        perimeter = []
        for index in range(4):
            start = rectangle[index]
            end = rectangle[(index + 1) % 4]
            sample_count = max(
                2,
                int(np.ceil(np.linalg.norm(end - start) / (0.5 * self.resolution))) + 1,
            )
            perimeter.append(np.linspace(start, end, sample_count))
        points = np.concatenate(perimeter)

        translated = points - self.origin[:2]
        cosine = float(np.cos(self.origin[2]))
        sine = float(np.sin(self.origin[2]))
        columns = (translated[:, 0] * cosine + translated[:, 1] * sine) / self.resolution
        rows = (-translated[:, 0] * sine + translated[:, 1] * cosine) / self.resolution
        distances = map_coordinates(
            self.distance_field,
            np.vstack((rows, columns)),
            order=1,
            mode="constant",
            cval=0.0,
        )
        # EDT measures to occupied pixel centers; subtract half a cell to approximate the wall surface.
        return max(0.0, float(distances.min()) - 0.5 * self.resolution)
