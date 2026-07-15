"""Privileged, censored labels for the D2 representation probe."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from d0.outcomes import align_rel


@dataclass(frozen=True)
class LabelConfig:
    horizons: tuple[float, ...] = (0.5, 1.0, 2.0)
    car_length_m: float = 0.58
    car_width_m: float = 0.31
    lateral_margin_m: float = 0.20
    ttc_cap_s: float = 5.0
    time_tolerance_s: float = 1e-6

    def __post_init__(self):
        if not self.horizons or any(not math.isfinite(value) or value <= 0 for value in self.horizons):
            raise ValueError("horizons must be finite positive values")
        if tuple(sorted(set(self.horizons))) != tuple(self.horizons):
            raise ValueError("horizons must be sorted and unique")
        for value in (self.car_length_m, self.car_width_m, self.lateral_margin_m, self.ttc_cap_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("geometry constants must be finite and positive")

    @property
    def lateral_corridor_m(self) -> float:
        return self.car_width_m + self.lateral_margin_m


class ReferenceProjector:
    """Efficient local projection onto an open-chain raceline reference."""

    def __init__(self, s: np.ndarray, xy: np.ndarray):
        s = np.asarray(s, dtype=np.float64)
        xy = np.asarray(xy, dtype=np.float64)
        if s.ndim != 1 or xy.shape != (len(s), 2) or len(s) < 2:
            raise ValueError("reference arrays have invalid shapes")
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(xy)) or not np.all(np.diff(s) > 0):
            raise ValueError("reference arrays must be finite with increasing s")
        segment = np.diff(xy, axis=0)
        segment_len_sq = np.sum(segment * segment, axis=1)
        if np.any(segment_len_sq <= 0):
            raise ValueError("reference contains a zero-length segment")
        self.s = s
        self.xy = xy
        self.segment = segment
        self.segment_len_sq = segment_len_sq
        self.tree = cKDTree(xy)

    @classmethod
    def from_arrays(cls, s: Sequence[float], xy: Sequence[Sequence[float]]):
        return cls(np.asarray(s, dtype=np.float64), np.asarray(xy, dtype=np.float64))

    @classmethod
    def from_asset(cls, asset_root: str | Path, map_name: str, raceline: str = "raceline1"):
        path = Path(asset_root) / map_name / f"{raceline}.csv"
        rows = np.loadtxt(path, delimiter=";", skiprows=1, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] < 3:
            raise ValueError(f"invalid reference asset: {path}")
        return cls(rows[:, 0], rows[:, 1:3])

    def project_many(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("projection points must be finite [N,2]")
        if len(points) == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty.copy(), empty.copy()

        k = min(4, len(self.xy))
        _, nearest = self.tree.query(points, k=k)
        if k == 1:
            nearest = nearest[:, None]
        out_s = np.empty(len(points), dtype=np.float64)
        out_d = np.empty(len(points), dtype=np.float64)
        out_theta = np.empty(len(points), dtype=np.float64)
        last_segment = len(self.segment) - 1
        for row_index, (point, waypoint_ids) in enumerate(zip(points, nearest)):
            candidates = set()
            for waypoint in np.atleast_1d(waypoint_ids):
                waypoint = int(waypoint)
                if waypoint > 0:
                    candidates.add(waypoint - 1)
                if waypoint <= last_segment:
                    candidates.add(waypoint)
            best = None
            for segment_index in sorted(candidates):
                vector = self.segment[segment_index]
                fraction = float(
                    np.clip(
                        np.dot(point - self.xy[segment_index], vector)
                        / self.segment_len_sq[segment_index],
                        0.0,
                        1.0,
                    )
                )
                projection = self.xy[segment_index] + fraction * vector
                distance_sq = float(np.dot(point - projection, point - projection))
                candidate = (distance_sq, segment_index, fraction, projection)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
            if best is None:
                raise AssertionError("reference projection found no candidate segment")
            _, segment_index, fraction, projection = best
            vector = self.segment[segment_index]
            tangent = vector / math.sqrt(float(self.segment_len_sq[segment_index]))
            normal_left = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            out_s[row_index] = self.s[segment_index] + fraction * (
                self.s[segment_index + 1] - self.s[segment_index]
            )
            out_d[row_index] = float(np.dot(point - projection, normal_left))
            out_theta[row_index] = math.atan2(float(tangent[1]), float(tangent[0]))
        return out_s, out_d, out_theta

    def project_many_exhaustive(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reference implementation matching `utils.project_to_reference`."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("projection points must be finite [N,2]")
        out_s = np.empty(len(points), dtype=np.float64)
        out_d = np.empty(len(points), dtype=np.float64)
        out_theta = np.empty(len(points), dtype=np.float64)
        for row_index, point in enumerate(points):
            fraction = np.clip(
                np.sum((point - self.xy[:-1]) * self.segment, axis=1) / self.segment_len_sq,
                0.0,
                1.0,
            )
            projection = self.xy[:-1] + fraction[:, None] * self.segment
            segment_index = int(np.argmin(np.sum((point - projection) ** 2, axis=1)))
            tangent = self.segment[segment_index] / math.sqrt(float(self.segment_len_sq[segment_index]))
            normal_left = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            out_s[row_index] = self.s[segment_index] + fraction[segment_index] * (
                self.s[segment_index + 1] - self.s[segment_index]
            )
            out_d[row_index] = float(np.dot(point - projection[segment_index], normal_left))
            out_theta[row_index] = math.atan2(float(tangent[1]), float(tangent[0]))
        return out_s, out_d, out_theta


def _array(data: Mapping, name: str, shape_tail: tuple[int, ...] = ()) -> np.ndarray:
    if name not in data:
        raise ValueError(f"episode missing {name}")
    value = np.asarray(data[name], dtype=np.float64)
    expected_ndim = 1 + len(shape_tail)
    if value.ndim != expected_ndim or (shape_tail and value.shape[1:] != shape_tail):
        raise ValueError(f"episode {name} has invalid shape {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"episode {name} contains non-finite values")
    return value


def _scalar(data: Mapping, name: str) -> float:
    if name not in data:
        raise ValueError(f"episode missing {name}")
    value = float(np.asarray(data[name]).reshape(()))
    if not math.isfinite(value):
        raise ValueError(f"episode {name} is non-finite")
    return value


def _bool(data: Mapping, name: str) -> bool:
    if name not in data:
        raise ValueError(f"episode missing {name}")
    value = np.asarray(data[name])
    if value.shape != () or value.dtype.kind != "b":
        raise ValueError(f"episode {name} must be a scalar bool")
    return bool(value.reshape(()))


def _horizon_token(horizon: float) -> str:
    centiseconds = int(round(float(horizon) * 100.0))
    if abs(centiseconds / 100.0 - float(horizon)) > 1e-9:
        raise ValueError("D2 horizon must be an exact centisecond")
    return f"{centiseconds:03d}"


def _event_target(
    time: np.ndarray,
    final_time: float,
    horizon: float,
    event_matches: bool,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    delta = final_time - time
    before_terminal = delta > -tolerance
    if event_matches:
        valid = before_terminal
        target = before_terminal & (delta > 0.0) & (delta <= horizon + tolerance)
    else:
        valid = before_terminal & (delta + tolerance >= horizon)
        target = np.zeros(len(time), dtype=bool)
    return target.astype(bool), valid.astype(bool)


def build_episode_labels(
    data: Mapping,
    projector: ReferenceProjector,
    track_length: float,
    config: LabelConfig | None = None,
) -> dict:
    config = LabelConfig() if config is None else config
    time = _array(data, "time")
    if len(time) == 0 or np.any(np.diff(time) <= 0):
        raise ValueError("episode time must be nonempty and strictly increasing")
    n = len(time)
    arrays = {
        name: _array(data, name, tail)
        for name, tail in (
            ("ego_actual_speed", ()),
            ("opp_actual_speed", ()),
            ("ego_pose", (3,)),
            ("opp_pose", (3,)),
            ("ego_progress", ()),
            ("opp_progress", ()),
        )
    }
    if any(len(value) != n for value in arrays.values()):
        raise ValueError("episode label arrays have different lengths")
    final_time = _scalar(data, "final_time")
    if final_time + config.time_tolerance_s < float(time[-1]):
        raise ValueError("final_time precedes the last recorded frame")

    rel = align_rel(
        arrays["ego_progress"],
        arrays["opp_progress"],
        _scalar(data, "final_ego_progress"),
        _scalar(data, "final_opp_progress"),
        float(track_length),
    )
    if rel.status != "ok":
        raise ValueError(f"D2 relative-progress alignment failed: {rel.status}")

    _, ego_d, ego_theta = projector.project_many(arrays["ego_pose"][:, :2])
    _, opp_d, opp_theta = projector.project_many(arrays["opp_pose"][:, :2])
    ego_v_s = arrays["ego_actual_speed"] * np.cos(arrays["ego_pose"][:, 2] - ego_theta)
    opp_v_s = arrays["opp_actual_speed"] * np.cos(arrays["opp_pose"][:, 2] - opp_theta)
    closing = ego_v_s - opp_v_s
    rel_current = rel.values[:-1]
    lateral_gap = np.abs(ego_d - opp_d)
    corridor = lateral_gap <= config.lateral_corridor_m
    applicable = (rel_current < 0.0) & (closing > 0.0) & corridor
    ttc = np.full(n, config.ttc_cap_s, dtype=np.float64)
    bumper_gap = np.maximum(0.0, -rel_current - config.car_length_m)
    bumper_gap[bumper_gap <= 1e-9] = 0.0
    ttc[applicable] = np.minimum(
        config.ttc_cap_s,
        bumper_gap[applicable] / closing[applicable],
    )

    ego_collision = _bool(data, "ego_collision")
    opp_collision = _bool(data, "opp_collision")
    output = {
        "alignment_status": rel.status,
        "alignment_k": int(rel.k),
        "closing_rate": closing.astype(np.float32),
        "corridor_ttc": ttc.astype(np.float32),
        "rel_s": rel_current.astype(np.float32),
        "lateral_gap": lateral_gap.astype(np.float32),
        "ego_v_s": ego_v_s.astype(np.float32),
        "opp_v_s": opp_v_s.astype(np.float32),
    }
    for prefix, matches in (
        ("ego", ego_collision),
        ("any", ego_collision or opp_collision),
    ):
        for horizon in config.horizons:
            token = _horizon_token(horizon)
            target, valid = _event_target(
                time,
                final_time,
                float(horizon),
                bool(matches),
                config.time_tolerance_s,
            )
            output[f"{prefix}_target_{token}"] = target
            output[f"{prefix}_valid_{token}"] = valid
    return output
