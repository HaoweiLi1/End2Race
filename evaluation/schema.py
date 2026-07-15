"""Versioned, JSON-safe scenario definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping

import numpy as np


EVALUATION_SCHEMA_VERSION = "end2race-evaluation-v1"
_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(value: str) -> str:
    slug = _SLUG_PATTERN.sub("-", value.strip()).strip("-.")
    if not slug:
        raise ValueError(f"Value does not contain a usable identifier: {value!r}")
    return slug


def _float_slug(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("Scenario floating-point values must be finite")
    text = f"{value:.12f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text.replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class Scenario:
    """All inputs that identify one paired multi-agent evaluation episode."""

    map_name: str
    ego_raceline: str
    opponent_raceline: str
    ego_start_index: int
    opponent_start_index: int
    interval_index: int
    opponent_speed_scale: float
    simulation_duration_s: float

    def __post_init__(self) -> None:
        for name in ("map_name", "ego_raceline", "opponent_raceline"):
            _slug(str(getattr(self, name)))
        if self.ego_start_index < 0 or self.opponent_start_index < 0:
            raise ValueError("Scenario start indices must be non-negative")
        if not math.isfinite(self.opponent_speed_scale) or self.opponent_speed_scale <= 0:
            raise ValueError("Opponent speed scale must be positive and finite")
        if not math.isfinite(self.simulation_duration_s) or self.simulation_duration_s <= 0:
            raise ValueError("Simulation duration must be positive and finite")

    @property
    def scenario_id(self) -> str:
        return "__".join(
            (
                f"map-{_slug(self.map_name)}",
                f"er-{_slug(self.ego_raceline)}",
                f"or-{_slug(self.opponent_raceline)}",
                f"e-{self.ego_start_index:04d}",
                f"o-{self.opponent_start_index:04d}",
                f"d-{self.interval_index}",
                f"s-{_float_slug(self.opponent_speed_scale)}",
                f"T-{_float_slug(self.simulation_duration_s)}",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["scenario_id"] = self.scenario_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Scenario":
        scenario = cls(
            map_name=str(value["map_name"]),
            ego_raceline=str(value["ego_raceline"]),
            opponent_raceline=str(value["opponent_raceline"]),
            ego_start_index=int(value["ego_start_index"]),
            opponent_start_index=int(value["opponent_start_index"]),
            interval_index=int(value["interval_index"]),
            opponent_speed_scale=float(value["opponent_speed_scale"]),
            simulation_duration_s=float(value["simulation_duration_s"]),
        )
        supplied_id = value.get("scenario_id")
        if supplied_id is not None and str(supplied_id) != scenario.scenario_id:
            raise ValueError(
                f"Scenario ID does not match its fields: {supplied_id!r} != {scenario.scenario_id!r}"
            )
        return scenario


def json_safe(value: Any) -> Any:
    """Convert numpy/path-like structures to strict JSON-compatible values."""

    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN and Inf are not valid evaluation JSON values")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__fspath__"):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} as evaluation JSON")
