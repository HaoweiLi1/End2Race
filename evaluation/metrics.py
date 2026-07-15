"""Closed-track progress and run-level metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class ClosedTrack:
    """Piecewise-linear closed track with deterministic progress projection."""

    points: np.ndarray
    starts: np.ndarray
    vectors: np.ndarray
    squared_lengths: np.ndarray
    cumulative_lengths: np.ndarray
    length_m: float

    @classmethod
    def from_points(cls, points: np.ndarray) -> "ClosedTrack":
        coordinates = np.asarray(points, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] < 3:
            raise ValueError("Closed track requires at least three 2D points")
        if not np.isfinite(coordinates).all():
            raise ValueError("Closed track points must be finite")
        if np.linalg.norm(coordinates[-1] - coordinates[0]) > 1e-9:
            coordinates = np.vstack((coordinates, coordinates[0]))
        starts = coordinates[:-1]
        vectors = coordinates[1:] - coordinates[:-1]
        squared = np.einsum("ij,ij->i", vectors, vectors)
        keep = squared > 1e-16
        starts = starts[keep]
        vectors = vectors[keep]
        squared = squared[keep]
        if squared.size < 3:
            raise ValueError("Closed track has fewer than three nonzero segments")
        lengths = np.sqrt(squared)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        return cls(
            points=coordinates,
            starts=starts,
            vectors=vectors,
            squared_lengths=squared,
            cumulative_lengths=cumulative,
            length_m=float(cumulative[-1]),
        )

    def project(self, point: np.ndarray) -> float:
        query = np.asarray(point, dtype=np.float64).reshape(2)
        offsets = query - self.starts
        fractions = np.clip(np.einsum("ij,ij->i", offsets, self.vectors) / self.squared_lengths, 0.0, 1.0)
        projections = self.starts + fractions[:, None] * self.vectors
        distances = np.einsum("ij,ij->i", query - projections, query - projections)
        segment = int(np.argmin(distances))
        segment_length = self.cumulative_lengths[segment + 1] - self.cumulative_lengths[segment]
        return float((self.cumulative_lengths[segment] + fractions[segment] * segment_length) % self.length_m)

    def unwrap(self, previous_wrapped: float, previous_unwrapped: float, point: np.ndarray) -> tuple[float, float]:
        wrapped = self.project(point)
        delta = (wrapped - previous_wrapped + 0.5 * self.length_m) % self.length_m - 0.5 * self.length_m
        return wrapped, float(previous_unwrapped + delta)


def episode_distance(initial_xy: np.ndarray, post_step_xy: np.ndarray) -> float:
    post = np.asarray(post_step_xy, dtype=np.float64)
    if post.size == 0:
        return 0.0
    path = np.vstack((np.asarray(initial_xy, dtype=np.float64).reshape(1, 2), post.reshape(-1, 2)))
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def aggregate_episodes(episodes: Iterable[Mapping[str, Any]], total_scenarios: int) -> dict[str, Any]:
    rows = list(episodes)
    completed = len(rows)
    outcomes = [str(row["outcome"]) for row in rows]
    collision_count = sum(bool(row["ego_collision"]) for row in rows)
    opponent_only_count = sum(bool(row["opponent_only_collision"]) for row in rows)
    overtake_count = outcomes.count("overtake")
    follow_count = outcomes.count("follow")

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in rows])) if rows else 0.0

    denominator = completed if completed else 1
    return {
        "total_scenarios": int(total_scenarios),
        "completed_scenarios": completed,
        "error_scenarios": int(total_scenarios - completed),
        "ego_collision_count": collision_count,
        "opponent_only_collision_count": opponent_only_count,
        "overtake_count": overtake_count,
        "follow_count": follow_count,
        "ego_collision_rate": float(collision_count / denominator) if completed else 0.0,
        "overtake_rate": float(overtake_count / denominator) if completed else 0.0,
        "follow_rate": float(follow_count / denominator) if completed else 0.0,
        "mean_ego_speed_mps": mean("ego_mean_measured_speed_mps"),
        "mean_ego_distance_m": mean("ego_distance_m"),
        "mean_final_relative_progress_m": mean("final_relative_progress_m"),
    }
