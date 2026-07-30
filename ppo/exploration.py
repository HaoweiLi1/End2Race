"""Training-only speed exploration modes and the causal following-risk gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_EXPLORATION_MODE = "baseline"
CONDITIONAL_WHITE_EXPLORATION_MODE = "conditional_white"
TEMPORAL_GLOBAL_EXPLORATION_MODE = "temporal_global"
CONDITIONAL_TEMPORAL_EXPLORATION_MODE = "conditional_temporal"
CORRIDOR_TEMPORAL_EXPLORATION_MODE = "corridor_temporal"
SPEED_EXPLORATION_MODES = (
    BASELINE_EXPLORATION_MODE,
    CONDITIONAL_WHITE_EXPLORATION_MODE,
    TEMPORAL_GLOBAL_EXPLORATION_MODE,
    CONDITIONAL_TEMPORAL_EXPLORATION_MODE,
    CORRIDOR_TEMPORAL_EXPLORATION_MODE,
)

BASELINE_SPEED_STD = 0.15
CONDITIONAL_WHITE_SPEED_STD = 0.50
CONDITIONAL_TEMPORAL_SPEED_STD = 0.25
TEMPORAL_RESAMPLE_STEPS = 50
EXPLORATION_GATE_INFO_KEY = "exploration_danger_gate"


def exploration_uses_gate(mode: str) -> bool:
    if mode not in SPEED_EXPLORATION_MODES:
        raise ValueError(f"Unknown speed exploration mode: {mode!r}")
    return mode in (
        CONDITIONAL_WHITE_EXPLORATION_MODE,
        CONDITIONAL_TEMPORAL_EXPLORATION_MODE,
        CORRIDOR_TEMPORAL_EXPLORATION_MODE,
    )


def exploration_metadata(
    mode: str, corridor_gate_config: "FrontCorridorGateConfig | None" = None
) -> dict[str, Any]:
    if mode not in SPEED_EXPLORATION_MODES:
        raise ValueError(f"Unknown speed exploration mode: {mode!r}")
    if mode == CORRIDOR_TEMPORAL_EXPLORATION_MODE:
        config = corridor_gate_config or FrontCorridorGateConfig()
        gate_type = f"front_corridor_overlap_gap{config.maximum_front_gap_m:g}"
        gate = asdict(config)
    else:
        gate_type = "escalating_required_deceleration"
        gate = asdict(FollowingDangerGateConfig())
    return {
        "mode": mode,
        "baseline_speed_std": BASELINE_SPEED_STD,
        "conditional_white_speed_std": CONDITIONAL_WHITE_SPEED_STD,
        "conditional_temporal_speed_std": CONDITIONAL_TEMPORAL_SPEED_STD,
        "corridor_temporal_speed_std": BASELINE_SPEED_STD,
        "temporal_resample_steps": TEMPORAL_RESAMPLE_STEPS,
        "gate_type": gate_type,
        "gate": gate,
        "training_only": True,
        "deterministic_evaluation_unchanged": True,
    }


@dataclass(frozen=True)
class FrontCorridorGateConfig:
    """Frozen CT-v2 arming geometry validated on collision and ordinary pools."""

    maximum_front_gap_m: float = 2.0
    maximum_abs_opponent_lateral_d_m: float = 0.25
    require_positive_lateral_overlap: bool = True

    def validate(self) -> None:
        values = np.asarray(
            (
                self.maximum_front_gap_m,
                self.maximum_abs_opponent_lateral_d_m,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("Front-corridor gate thresholds must be finite and positive")
        if not self.require_positive_lateral_overlap:
            raise ValueError("CT-v2 requires positive lateral OBB overlap")


@dataclass(frozen=True)
class FollowingDangerGateConfig:
    """Frozen online counterpart of the validated escalating-required-decel gate."""

    closing_window_s: float = 0.10
    entry_gap_m: float = 2.00
    exit_gap_m: float = 2.20
    entry_closing_time_s: float = 1.50
    exit_closing_time_s: float = 2.00
    corridor_entry_abs_d_m: float = 0.20
    corridor_exit_abs_d_m: float = 0.25
    lateral_overlap_entry_m: float = 0.02
    warning_grace_s: float = 0.20
    recovery_hold_s: float = 0.10
    safe_gap_m: float = 0.50
    required_relative_deceleration_mps2: float = 1.25
    required_deceleration_persistence_s: float = 0.20
    required_deceleration_rate_window_s: float = 0.20
    minimum_required_deceleration_growth_mps3: float = 2.00
    lateral_escape_horizon_s: float = 0.40
    lateral_opening_window_s: float = 0.10
    required_deceleration_report_cap_mps2: float = 100.0

    def validate(self) -> None:
        values = np.asarray(tuple(asdict(self).values()), dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("Following-danger gate parameters must be finite and positive")
        if self.corridor_exit_abs_d_m <= self.corridor_entry_abs_d_m:
            raise ValueError("Corridor exit threshold must exceed entry threshold")
        if self.exit_gap_m <= self.entry_gap_m:
            raise ValueError("Gap exit threshold must exceed entry threshold")
        if self.exit_closing_time_s <= self.entry_closing_time_s:
            raise ValueError("Closing-time exit threshold must exceed entry threshold")


@dataclass(frozen=True)
class _Projection:
    progress_m: float
    lateral_d_m: float
    tangent_heading_rad: float


class _FrenetProjector:
    """Pointwise form of the offline validation projector."""

    def __init__(self, path: Path) -> None:
        reference = np.loadtxt(path, delimiter=";", comments="#", dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1] < 3:
            raise ValueError(f"Invalid raceline CSV: {path}")
        self.track_length_m = float(reference[-1, 0])
        progress = reference[:, 0]
        points = reference[:, 1:3]
        if np.linalg.norm(points[-1] - points[0]) <= 1e-9:
            progress = progress[:-1]
            points = points[:-1]
        if (
            len(points) < 3
            or not np.isfinite(points).all()
            or not np.isfinite(progress).all()
            or np.any(np.diff(progress) <= 0.0)
            or self.track_length_m <= progress[-1]
        ):
            raise ValueError(f"Invalid cyclic raceline geometry: {path}")
        self.progress_m = progress
        self.points_xy = points
        self.segment_xy = np.roll(points, -1, axis=0) - points
        self.segment_norm_sq = np.einsum("ij,ij->i", self.segment_xy, self.segment_xy)
        if np.any(self.segment_norm_sq <= 0.0):
            raise ValueError("Raceline contains a zero-length segment")
        self.segment_length_m = np.sqrt(self.segment_norm_sq)
        self.segment_progress_m = np.concatenate(
            (np.diff(progress), np.asarray([self.track_length_m - progress[-1]]))
        )
        self.tree = cKDTree(points)

    def project(self, point_xy: np.ndarray) -> _Projection:
        point = np.asarray(point_xy, dtype=np.float64).reshape(-1)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Projection point must be one finite XY pair")
        _distance, nearest = self.tree.query(point)
        candidates = np.asarray((nearest, (nearest - 1) % len(self.points_xy)), dtype=np.int64)
        starts = self.points_xy[candidates]
        vectors = self.segment_xy[candidates]
        offsets = point - starts
        fractions = np.clip(
            np.einsum("ci,ci->c", offsets, vectors)
            / self.segment_norm_sq[candidates],
            0.0,
            1.0,
        )
        closest = starts + fractions[:, None] * vectors
        distance_sq = np.einsum("ci,ci->c", point - closest, point - closest)
        choice = int(np.argmin(distance_sq))
        segment = int(candidates[choice])
        fraction = float(fractions[choice])
        tangent = self.segment_xy[segment] / self.segment_length_m[segment]
        normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
        progress = (
            self.progress_m[segment]
            + fraction * self.segment_progress_m[segment]
        ) % self.track_length_m
        lateral = float(np.dot(point - closest[choice], normal))
        heading = float(np.arctan2(tangent[1], tangent[0]))
        return _Projection(float(progress), lateral, heading)


def _wrap_angle(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def _causal_rate(times: list[float], values: list[float], window_s: float) -> float:
    if len(times) < 2:
        return float("nan")
    target = times[-1] - window_s
    previous = int(np.searchsorted(np.asarray(times), target, side="right") - 1)
    if previous < 0:
        return float("nan")
    elapsed = times[-1] - times[previous]
    if elapsed + 1e-12 < 0.9 * window_s:
        return float("nan")
    return float((values[-1] - values[previous]) / elapsed)


class FrontCorridorGate:
    """Stateless raw CT-v2 gate; the policy supplies the fixed 50-step hold."""

    def __init__(
        self,
        map_name: str,
        ego_raceline: str,
        *,
        vehicle_length_m: float,
        vehicle_width_m: float,
        config: FrontCorridorGateConfig | None = None,
    ) -> None:
        self.config = config or FrontCorridorGateConfig()
        self.config.validate()
        dimensions = np.asarray(
            (vehicle_length_m, vehicle_width_m), dtype=np.float64
        )
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0.0):
            raise ValueError("Vehicle dimensions must be finite and positive")
        self.vehicle_length_m = float(vehicle_length_m)
        self.vehicle_width_m = float(vehicle_width_m)
        self.projector = _FrenetProjector(
            PROJECT_ROOT / "f1tenth_racetracks" / map_name / f"{ego_raceline}.csv"
        )
        self.current_gate = False

    @staticmethod
    def _position(raw_observation: dict[str, Any], index: int) -> np.ndarray:
        return np.asarray(
            (
                np.asarray(raw_observation["poses_x"])[index],
                np.asarray(raw_observation["poses_y"])[index],
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _heading(raw_observation: dict[str, Any], index: int) -> float:
        return float(np.asarray(raw_observation["poses_theta"])[index])

    def _evaluate(
        self,
        raw_observation: dict[str, Any],
        *,
        ego_index: int,
        opponent_index: int,
    ) -> bool:
        ego = self.projector.project(self._position(raw_observation, ego_index))
        opponent = self.projector.project(
            self._position(raw_observation, opponent_index)
        )
        raw_relative_m = float(
            (
                ego.progress_m
                - opponent.progress_m
                + 0.5 * self.projector.track_length_m
            )
            % self.projector.track_length_m
            - 0.5 * self.projector.track_length_m
        )
        opponent_ahead_center_m = -raw_relative_m

        def extents(heading_error: float) -> tuple[float, float]:
            cosine = abs(float(np.cos(heading_error)))
            sine = abs(float(np.sin(heading_error)))
            return (
                0.5
                * (
                    self.vehicle_length_m * cosine
                    + self.vehicle_width_m * sine
                ),
                0.5
                * (
                    self.vehicle_length_m * sine
                    + self.vehicle_width_m * cosine
                ),
            )

        ego_longitudinal, ego_lateral = extents(
            _wrap_angle(
                self._heading(raw_observation, ego_index)
                - ego.tangent_heading_rad
            )
        )
        opponent_longitudinal, opponent_lateral = extents(
            _wrap_angle(
                self._heading(raw_observation, opponent_index)
                - opponent.tangent_heading_rad
            )
        )
        front_gap_m = (
            opponent_ahead_center_m - ego_longitudinal - opponent_longitudinal
        )
        ego_low = ego.lateral_d_m - ego_lateral
        ego_high = ego.lateral_d_m + ego_lateral
        opponent_low = opponent.lateral_d_m - opponent_lateral
        opponent_high = opponent.lateral_d_m + opponent_lateral
        lateral_overlap_m = (
            min(ego_high, opponent_high) - max(ego_low, opponent_low)
        )
        self.current_gate = bool(
            opponent_ahead_center_m > 0.0
            and opponent_ahead_center_m < 0.5 * self.projector.track_length_m
            and front_gap_m > 0.0
            and front_gap_m < self.config.maximum_front_gap_m
            and abs(opponent.lateral_d_m)
            < self.config.maximum_abs_opponent_lateral_d_m
            and lateral_overlap_m > 0.0
        )
        return self.current_gate

    def reset(
        self,
        raw_observation: dict[str, Any],
        *,
        elapsed_time_s: float = 0.0,
        ego_index: int = 0,
        opponent_index: int = 1,
    ) -> bool:
        del elapsed_time_s
        self.current_gate = False
        return self._evaluate(
            raw_observation,
            ego_index=ego_index,
            opponent_index=opponent_index,
        )

    def step(
        self,
        raw_observation: dict[str, Any],
        *,
        elapsed_time_s: float,
        ego_index: int = 0,
        opponent_index: int = 1,
    ) -> bool:
        del elapsed_time_s
        return self._evaluate(
            raw_observation,
            ego_index=ego_index,
            opponent_index=opponent_index,
        )


class EscalatingRequiredDecelerationGate:
    """Causal, stateful arming signal used only to shape training exploration."""

    def __init__(
        self,
        map_name: str,
        ego_raceline: str,
        *,
        vehicle_length_m: float,
        vehicle_width_m: float,
        config: FollowingDangerGateConfig | None = None,
    ) -> None:
        self.config = config or FollowingDangerGateConfig()
        self.config.validate()
        dimensions = np.asarray((vehicle_length_m, vehicle_width_m), dtype=np.float64)
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0.0):
            raise ValueError("Vehicle dimensions must be finite and positive")
        self.vehicle_length_m = float(vehicle_length_m)
        self.vehicle_width_m = float(vehicle_width_m)
        self.projector = _FrenetProjector(
            PROJECT_ROOT / "f1tenth_racetracks" / map_name / f"{ego_raceline}.csv"
        )
        self._clear()

    def _clear(self) -> None:
        self.times: list[float] = []
        self.relative_progress_m: list[float] = []
        self.lateral_separation_m: list[float] = []
        self.required_deceleration_mps2: list[float] = []
        self._previous_raw_relative_m: float | None = None
        self._relative_unwrapped_m: float | None = None
        self._in_encounter = False
        self._warning_since_s: float | None = None
        self._risk_since_s: float | None = None
        self._recovery_since_s: float | None = None
        self.current_gate = False

    @staticmethod
    def _position(raw_observation: dict[str, Any], index: int) -> np.ndarray:
        return np.asarray(
            (
                np.asarray(raw_observation["poses_x"])[index],
                np.asarray(raw_observation["poses_y"])[index],
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _heading(raw_observation: dict[str, Any], index: int) -> float:
        return float(np.asarray(raw_observation["poses_theta"])[index])

    def reset(
        self,
        raw_observation: dict[str, Any],
        *,
        elapsed_time_s: float = 0.0,
        ego_index: int = 0,
        opponent_index: int = 1,
    ) -> bool:
        self._clear()
        self._append_state(
            raw_observation,
            elapsed_time_s=elapsed_time_s,
            ego_index=ego_index,
            opponent_index=opponent_index,
        )
        return self.current_gate

    def _append_state(
        self,
        raw_observation: dict[str, Any],
        *,
        elapsed_time_s: float,
        ego_index: int,
        opponent_index: int,
    ) -> dict[str, float | bool]:
        now = float(elapsed_time_s)
        if not np.isfinite(now) or (self.times and now <= self.times[-1]):
            raise ValueError("Following-danger gate time must increase strictly")
        ego = self.projector.project(self._position(raw_observation, ego_index))
        opponent = self.projector.project(self._position(raw_observation, opponent_index))
        raw_relative = float(
            (
                ego.progress_m
                - opponent.progress_m
                + 0.5 * self.projector.track_length_m
            )
            % self.projector.track_length_m
            - 0.5 * self.projector.track_length_m
        )
        if self._previous_raw_relative_m is None:
            self._relative_unwrapped_m = raw_relative
        else:
            increment = float(
                (
                    raw_relative
                    - self._previous_raw_relative_m
                    + 0.5 * self.projector.track_length_m
                )
                % self.projector.track_length_m
                - 0.5 * self.projector.track_length_m
            )
            self._relative_unwrapped_m = float(self._relative_unwrapped_m + increment)
        self._previous_raw_relative_m = raw_relative
        opponent_ahead_center_m = -raw_relative

        ego_heading_error = _wrap_angle(
            self._heading(raw_observation, ego_index) - ego.tangent_heading_rad
        )
        opponent_heading_error = _wrap_angle(
            self._heading(raw_observation, opponent_index)
            - opponent.tangent_heading_rad
        )

        def extents(heading_error: float) -> tuple[float, float]:
            cosine = abs(float(np.cos(heading_error)))
            sine = abs(float(np.sin(heading_error)))
            return (
                0.5 * (self.vehicle_length_m * cosine + self.vehicle_width_m * sine),
                0.5 * (self.vehicle_length_m * sine + self.vehicle_width_m * cosine),
            )

        ego_longitudinal, ego_lateral = extents(ego_heading_error)
        opponent_longitudinal, opponent_lateral = extents(opponent_heading_error)
        front_gap_m = (
            opponent_ahead_center_m - ego_longitudinal - opponent_longitudinal
        )
        ego_low = ego.lateral_d_m - ego_lateral
        ego_high = ego.lateral_d_m + ego_lateral
        opponent_low = opponent.lateral_d_m - opponent_lateral
        opponent_high = opponent.lateral_d_m + opponent_lateral
        lateral_overlap_m = min(ego_high, opponent_high) - max(ego_low, opponent_low)
        lateral_separation_m = abs(opponent.lateral_d_m - ego.lateral_d_m)

        self.times.append(now)
        self.relative_progress_m.append(float(self._relative_unwrapped_m))
        self.lateral_separation_m.append(float(lateral_separation_m))
        closing_speed_mps = _causal_rate(
            self.times,
            self.relative_progress_m,
            self.config.closing_window_s,
        )
        lateral_opening_speed_mps = _causal_rate(
            self.times,
            self.lateral_separation_m,
            self.config.lateral_opening_window_s,
        )
        finite_closing = bool(np.isfinite(closing_speed_mps))
        positive_closing = bool(finite_closing and closing_speed_mps > 0.0)
        closing_time_s = (
            float(front_gap_m / closing_speed_mps)
            if positive_closing
            else float("inf")
        )
        opponent_ahead = bool(
            opponent_ahead_center_m > 0.0
            and opponent_ahead_center_m < 0.5 * self.projector.track_length_m
        )
        corridor_entry = abs(opponent.lateral_d_m) < self.config.corridor_entry_abs_d_m
        corridor_exit = abs(opponent.lateral_d_m) < self.config.corridor_exit_abs_d_m
        lateral_entry = lateral_overlap_m >= self.config.lateral_overlap_entry_m
        lateral_keep = lateral_overlap_m > 0.0
        raw_warning = bool(
            opponent_ahead
            and front_gap_m > 0.0
            and front_gap_m < self.config.entry_gap_m
            and lateral_entry
            and corridor_entry
            and positive_closing
            and closing_time_s < self.config.entry_closing_time_s
        )
        structural_keep = bool(
            opponent_ahead
            and front_gap_m > 0.0
            and lateral_keep
            and corridor_exit
        )
        dynamic_keep = bool(
            finite_closing
            and closing_speed_mps > 0.0
            and front_gap_m < self.config.exit_gap_m
            and closing_time_s < self.config.exit_closing_time_s
        )
        opening_speed = (
            max(float(lateral_opening_speed_mps), 0.0)
            if np.isfinite(lateral_opening_speed_mps)
            else 0.0
        )
        lateral_escape_predicted = bool(
            lateral_overlap_m
            - opening_speed * self.config.lateral_escape_horizon_s
            <= 0.0
        )
        available_gap_m = front_gap_m - self.config.safe_gap_m
        if positive_closing and available_gap_m > 1e-6:
            required_deceleration = (
                closing_speed_mps * closing_speed_mps / (2.0 * available_gap_m)
            )
        elif positive_closing:
            required_deceleration = self.config.required_deceleration_report_cap_mps2
        else:
            required_deceleration = 0.0
        required_deceleration = float(
            np.clip(
                required_deceleration,
                0.0,
                self.config.required_deceleration_report_cap_mps2,
            )
        )
        self.required_deceleration_mps2.append(required_deceleration)
        required_deceleration_rate_mps3 = _causal_rate(
            self.times,
            self.required_deceleration_mps2,
            self.config.required_deceleration_rate_window_s,
        )
        risk = bool(
            structural_keep
            and dynamic_keep
            and required_deceleration
            > self.config.required_relative_deceleration_mps2
            and not lateral_escape_predicted
            and np.isfinite(required_deceleration_rate_mps3)
            and required_deceleration_rate_mps3
            > self.config.minimum_required_deceleration_growth_mps3
        )
        return {
            "raw_warning": raw_warning,
            "structural_keep": structural_keep,
            "dynamic_keep": dynamic_keep,
            "risk": risk,
            "front_gap_m": float(front_gap_m),
            "closing_speed_mps": float(closing_speed_mps),
            "required_deceleration_mps2": required_deceleration,
        }

    def step(
        self,
        raw_observation: dict[str, Any],
        *,
        elapsed_time_s: float,
        ego_index: int = 0,
        opponent_index: int = 1,
    ) -> bool:
        state = self._append_state(
            raw_observation,
            elapsed_time_s=elapsed_time_s,
            ego_index=ego_index,
            opponent_index=opponent_index,
        )
        now = float(elapsed_time_s)
        entry = bool(state["raw_warning"])
        structural_keep = bool(state["structural_keep"])
        dynamic_keep = bool(state["dynamic_keep"])
        risk = bool(state["risk"])

        if not self._in_encounter:
            if entry:
                self._in_encounter = True
                self._warning_since_s = now
                self._risk_since_s = now if risk else None
                self._recovery_since_s = None
        else:
            if not structural_keep:
                self._in_encounter = False
            elif dynamic_keep:
                self._recovery_since_s = None
            else:
                if self._recovery_since_s is None:
                    self._recovery_since_s = now
                if (
                    now - self._recovery_since_s + 1e-12
                    >= self.config.recovery_hold_s
                ):
                    self._in_encounter = False
            if not self._in_encounter:
                self._warning_since_s = None
                self._risk_since_s = None
                self._recovery_since_s = None
                if entry:
                    self._in_encounter = True
                    self._warning_since_s = now
                    self._risk_since_s = now if risk else None

        if not self._in_encounter:
            self.current_gate = False
            return self.current_gate
        current_risk = bool(risk and dynamic_keep)
        if current_risk:
            if self._risk_since_s is None:
                self._risk_since_s = now
        else:
            self._risk_since_s = None
        mature = bool(
            self._warning_since_s is not None
            and now - self._warning_since_s + 1e-12 >= self.config.warning_grace_s
        )
        persistent = bool(
            self._risk_since_s is not None
            and now - self._risk_since_s + 1e-12
            >= self.config.required_deceleration_persistence_s
        )
        self.current_gate = bool(mature and persistent)
        return self.current_gate
