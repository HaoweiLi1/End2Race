"""Small exact geometry helpers shared by PPO reward and privileged state."""

from __future__ import annotations

import numpy as np


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    fraction = np.clip(np.dot(point - start, segment) / np.dot(segment, segment), 0.0, 1.0)
    return float(np.linalg.norm(point - (start + fraction * segment)))


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


def rectangle_clearance(vertices_a: np.ndarray, vertices_b: np.ndarray) -> float:
    """Return exact surface distance between convex quadrilaterals, zero at contact/overlap."""

    first = np.asarray(vertices_a, dtype=np.float64)
    second = np.asarray(vertices_b, dtype=np.float64)
    if first.shape != (4, 2) or second.shape != (4, 2) or not np.isfinite((first, second)).all():
        raise ValueError("Rectangle clearance requires two finite (4, 2) vertex arrays")
    if not _separating_axis_exists(first, second):
        return 0.0
    best = np.inf
    for vertices_from, vertices_to in ((first, second), (second, first)):
        for point in vertices_from:
            for index in range(len(vertices_to)):
                best = min(
                    best,
                    _point_segment_distance(
                        point,
                        vertices_to[index],
                        vertices_to[(index + 1) % len(vertices_to)],
                    ),
                )
    return float(best)
