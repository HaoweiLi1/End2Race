"""Fixed boundary-aware hard-neighbor collision caches for controlled PPO runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np

from ppo.collision_classification import (
    classify_collision_scenarios,
    collision_classification_config,
    load_collision_cache_artifacts,
    resolve_collision_scenarios,
)
from ppo.scenarios import (
    COLLISION_INTERVAL_INDICES,
    COLLISION_SPEED_SCALES,
    EGO_RACELINE,
    HARD_NEIGHBOR_MAX_CANDIDATES_PER_FAMILY,
    OPPONENT_RACELINES,
    ScenarioSpec,
    evaluation_startpoints,
)
from utils import find_corresponding_waypoint, load_raceline_waypoints


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARD_NEIGHBOR_CACHE_SCHEMA = 2
BOUNDARY_GENERATOR_SCHEMA = 1
CLASSIFIER_CONTRACT = "deterministic_bc_ego_collision_v1"
SPEED_FIXED_POINT_SCALE = 1000
SEMANTIC_CACHE_FILES = (
    "classification_config.json",
    "base_candidate_outcomes.jsonl",
    "boundary_pairs.jsonl",
    "boundary_candidate_outcomes.jsonl",
    "collision_scenarios.json",
    "classification_summary.json",
)
HARD_CACHE_FILES = SEMANTIC_CACHE_FILES + ("build_metadata.json", "manifest.sha256")
BASE_CACHE_FILES = (
    "classification_config.json",
    "candidate_outcomes.jsonl",
    "collision_scenarios.json",
    "classification_summary.json",
)


@dataclass(frozen=True)
class BoundaryCandidatePlan:
    scenario_id: str
    map_name: str
    ego_raceline: str
    startpoint_ordinal: int
    ego_idx: int
    opp_raceline: str
    interval_idx: int
    speed_milli: int
    sim_duration: float
    timestep: float
    integrator: str
    source_pair_ids: tuple[str, ...]


@dataclass(frozen=True)
class BoundaryDiscovery:
    pair_records: tuple[dict[str, Any], ...]
    candidates: tuple[BoundaryCandidatePlan, ...]
    generated_candidate_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _speed_milli(value: float | Decimal) -> int:
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    scaled = decimal_value * SPEED_FIXED_POINT_SCALE
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise ValueError(
            f"Speed scale {value!r} cannot be represented at 1/{SPEED_FIXED_POINT_SCALE} precision"
        )
    return int(integral)


def _speed_float(speed_milli: int) -> float:
    return float(Decimal(speed_milli) / SPEED_FIXED_POINT_SCALE)


def _family_key(scenario: ScenarioSpec) -> tuple[Any, ...]:
    return (
        scenario.map_name,
        scenario.ego_raceline,
        scenario.startpoint_ordinal,
        scenario.ego_idx,
        scenario.opp_raceline,
        scenario.sim_duration,
        scenario.timestep,
        scenario.integrator,
    )


def _hard_scenario_id(
    startpoint_ordinal: int,
    ego_idx: int,
    opp_raceline: str,
    interval_idx: int,
    speed_milli: int,
) -> str:
    return (
        f"hard-sp{startpoint_ordinal:03d}-ego{ego_idx:04d}-{opp_raceline}"
        f"-i{interval_idx:02d}-v{speed_milli:04d}"
    )


def discover_boundary_candidates(
    candidates: Sequence[ScenarioSpec],
    outcomes: Sequence[dict[str, Any]],
    *,
    interval_indices: Sequence[int] = COLLISION_INTERVAL_INDICES,
    speed_scales: Sequence[float] = COLLISION_SPEED_SCALES,
    max_candidates_per_family: int = HARD_NEIGHBOR_MAX_CANDIDATES_PER_FAMILY,
) -> BoundaryDiscovery:
    """Find one-axis collision/other edges and deterministically refine them."""

    candidates = tuple(candidates)
    outcomes = tuple(outcomes)
    if not candidates or len(candidates) != len(outcomes):
        raise ValueError("Boundary discovery requires aligned non-empty candidates and outcomes")
    if max_candidates_per_family <= 0:
        raise ValueError("max_candidates_per_family must be positive")
    intervals = tuple(sorted(int(value) for value in interval_indices))
    speed_millis = tuple(sorted(_speed_milli(value) for value in speed_scales))
    if (
        len(set(intervals)) != len(intervals)
        or len(set(speed_millis)) != len(speed_millis)
        or any(value <= 0 for value in intervals)
        or any(value <= 0 for value in speed_millis)
    ):
        raise ValueError("Boundary interval and speed grids must be positive and unique")

    outcome_by_id: dict[str, str] = {}
    for index, (scenario, outcome) in enumerate(zip(candidates, outcomes)):
        if (
            set(outcome) != {"candidate_index", "scenario_id", "outcome"}
            or outcome["candidate_index"] != index
            or outcome["scenario_id"] != scenario.scenario_id
            or outcome["outcome"] not in {"ego_collision", "other", "invalid"}
        ):
            raise ValueError(f"Invalid base outcome at candidate {index}")
        if scenario.scenario_id in outcome_by_id:
            raise ValueError(f"Duplicate base scenario ID: {scenario.scenario_id}")
        outcome_by_id[scenario.scenario_id] = str(outcome["outcome"])

    family_lattices: dict[tuple[Any, ...], dict[tuple[int, int], ScenarioSpec]] = defaultdict(dict)
    for scenario in candidates:
        key = (int(scenario.interval_idx), _speed_milli(scenario.opp_speedscale))
        lattice = family_lattices[_family_key(scenario)]
        if key in lattice:
            raise ValueError(f"Duplicate base physical key in family {_family_key(scenario)}: {key}")
        lattice[key] = scenario

    expected_lattice_size = len(intervals) * len(speed_millis)
    raw_pairs: list[dict[str, Any]] = []
    generated: dict[tuple[Any, ...], dict[str, Any]] = {}

    def add_pair(
        axis: str,
        low: ScenarioSpec,
        high: ScenarioSpec,
        refined_parameters: Sequence[tuple[int, int]],
    ) -> None:
        low_outcome = outcome_by_id[low.scenario_id]
        high_outcome = outcome_by_id[high.scenario_id]
        if {low_outcome, high_outcome} != {"ego_collision", "other"}:
            return
        pair_id = f"boundary-{len(raw_pairs):05d}-{axis}-{low.scenario_id}--{high.scenario_id}"
        generated_ids = []
        family = _family_key(low)
        for interval_idx, speed_milli in refined_parameters:
            scenario_id = _hard_scenario_id(
                low.startpoint_ordinal,
                low.ego_idx,
                low.opp_raceline,
                interval_idx,
                speed_milli,
            )
            physical_key = family + (interval_idx, speed_milli)
            generated_ids.append(scenario_id)
            if physical_key not in generated:
                generated[physical_key] = {
                    "scenario_id": scenario_id,
                    "template": low,
                    "interval_idx": interval_idx,
                    "speed_milli": speed_milli,
                    "source_pair_ids": [],
                }
            elif generated[physical_key]["scenario_id"] != scenario_id:
                raise RuntimeError(f"Boundary candidate ID is not deterministic for {physical_key}")
            generated[physical_key]["source_pair_ids"].append(pair_id)
        raw_pairs.append(
            {
                "pair_index": len(raw_pairs),
                "pair_id": pair_id,
                "axis": axis,
                "map_name": low.map_name,
                "ego_raceline": low.ego_raceline,
                "startpoint_ordinal": low.startpoint_ordinal,
                "ego_idx": low.ego_idx,
                "opp_raceline": low.opp_raceline,
                "low_scenario_id": low.scenario_id,
                "low_outcome": low_outcome,
                "high_scenario_id": high.scenario_id,
                "high_outcome": high_outcome,
                "generated_scenario_ids": generated_ids,
            }
        )

    for family in sorted(family_lattices):
        lattice = family_lattices[family]
        if len(lattice) != expected_lattice_size or set(lattice) != {
            (interval_idx, speed_milli)
            for interval_idx in intervals
            for speed_milli in speed_millis
        }:
            raise ValueError(f"Base candidate family does not form the configured lattice: {family}")
        for interval_idx in intervals:
            for low_speed, high_speed in zip(speed_millis, speed_millis[1:]):
                midpoint_sum = low_speed + high_speed
                if midpoint_sum % 2:
                    raise ValueError(
                        f"Speed boundary {low_speed}/{high_speed} has no exact milliscale midpoint"
                    )
                add_pair(
                    "speed",
                    lattice[(interval_idx, low_speed)],
                    lattice[(interval_idx, high_speed)],
                    ((interval_idx, midpoint_sum // 2),),
                )
        for speed_milli in speed_millis:
            for low_interval, high_interval in zip(intervals, intervals[1:]):
                add_pair(
                    "interval",
                    lattice[(low_interval, speed_milli)],
                    lattice[(high_interval, speed_milli)],
                    tuple(
                        (interval_idx, speed_milli)
                        for interval_idx in range(low_interval + 1, high_interval)
                    ),
                )

    generated_by_family: dict[tuple[Any, ...], list[tuple[tuple[Any, ...], dict[str, Any]]]] = defaultdict(list)
    for physical_key, record in generated.items():
        generated_by_family[physical_key[:-2]].append((physical_key, record))
    selected_keys: set[tuple[Any, ...]] = set()
    for family, records in generated_by_family.items():
        records.sort(
            key=lambda item: (
                hashlib.sha256(item[1]["scenario_id"].encode("utf-8")).hexdigest(),
                item[1]["scenario_id"],
            )
        )
        selected_keys.update(key for key, _record in records[:max_candidates_per_family])

    plans = []
    for physical_key in sorted(selected_keys):
        record = generated[physical_key]
        template: ScenarioSpec = record["template"]
        plans.append(
            BoundaryCandidatePlan(
                scenario_id=record["scenario_id"],
                map_name=template.map_name,
                ego_raceline=template.ego_raceline,
                startpoint_ordinal=template.startpoint_ordinal,
                ego_idx=template.ego_idx,
                opp_raceline=template.opp_raceline,
                interval_idx=int(record["interval_idx"]),
                speed_milli=int(record["speed_milli"]),
                sim_duration=template.sim_duration,
                timestep=template.timestep,
                integrator=template.integrator,
                source_pair_ids=tuple(record["source_pair_ids"]),
            )
        )
    selected_ids = {plan.scenario_id for plan in plans}
    pair_records = tuple(
        {
            **record,
            "selected_scenario_ids": [
                scenario_id
                for scenario_id in record["generated_scenario_ids"]
                if scenario_id in selected_ids
            ],
        }
        for record in raw_pairs
    )
    return BoundaryDiscovery(pair_records, tuple(plans), len(generated))


def materialize_boundary_candidates(
    plans: Sequence[BoundaryCandidatePlan],
    base_candidates: Sequence[ScenarioSpec],
) -> tuple[ScenarioSpec, ...]:
    """Resolve opponent waypoint indices for a deterministic boundary plan."""

    templates: dict[tuple[Any, ...], ScenarioSpec] = {}
    for scenario in base_candidates:
        templates.setdefault(_family_key(scenario), scenario)
    waypoint_cache: dict[tuple[str, str], np.ndarray] = {}
    scenarios = []
    physical_keys = set()
    for plan in plans:
        family = (
            plan.map_name,
            plan.ego_raceline,
            plan.startpoint_ordinal,
            plan.ego_idx,
            plan.opp_raceline,
            plan.sim_duration,
            plan.timestep,
            plan.integrator,
        )
        if family not in templates:
            raise ValueError(f"Boundary plan has no base family: {plan.scenario_id}")
        ego_key = (plan.map_name, plan.ego_raceline)
        opp_key = (plan.map_name, plan.opp_raceline)
        if ego_key not in waypoint_cache:
            waypoint_cache[ego_key] = load_raceline_waypoints(plan.map_name, f"{plan.ego_raceline}.csv")
        if opp_key not in waypoint_cache:
            waypoint_cache[opp_key] = load_raceline_waypoints(plan.map_name, f"{plan.opp_raceline}.csv")
        ego_waypoints = waypoint_cache[ego_key]
        opponent_waypoints = waypoint_cache[opp_key]
        mapped_index = (
            plan.ego_idx
            if plan.opp_raceline == plan.ego_raceline
            else int(find_corresponding_waypoint(ego_waypoints[plan.ego_idx], opponent_waypoints))
        )
        unique_opponent_waypoints = len(opponent_waypoints) - 1
        if unique_opponent_waypoints <= 0:
            raise ValueError(f"Opponent raceline is empty for {plan.scenario_id}")
        opp_idx = int((mapped_index + plan.interval_idx) % unique_opponent_waypoints)
        scenario = ScenarioSpec(
            scenario_id=plan.scenario_id,
            pool="hard_neighbor",
            startpoint_ordinal=plan.startpoint_ordinal,
            ego_idx=plan.ego_idx,
            opp_idx=opp_idx,
            opp_raceline=plan.opp_raceline,
            opp_speedscale=_speed_float(plan.speed_milli),
            interval_idx=plan.interval_idx,
            map_name=plan.map_name,
            ego_raceline=plan.ego_raceline,
            sim_duration=plan.sim_duration,
            timestep=plan.timestep,
            integrator=plan.integrator,
        )
        physical_key = (
            scenario.map_name,
            scenario.ego_raceline,
            scenario.ego_idx,
            scenario.opp_raceline,
            scenario.opp_idx,
            _speed_milli(scenario.opp_speedscale),
            scenario.sim_duration,
            scenario.timestep,
            scenario.integrator,
        )
        if physical_key in physical_keys:
            raise RuntimeError(f"Duplicate hard-neighbor physical scenario: {physical_key}")
        physical_keys.add(physical_key)
        scenarios.append(scenario)
    if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise RuntimeError("Hard-neighbor scenario IDs are not unique")
    return tuple(scenarios)


def _minimum_evaluation_distance(scenarios: Sequence[ScenarioSpec]) -> float:
    if not scenarios:
        raise ValueError("Cannot validate an empty scenario pool")
    waypoint_cache: dict[tuple[str, str], np.ndarray] = {}
    minimum = float("inf")
    for scenario in scenarios:
        key = (scenario.map_name, scenario.ego_raceline)
        if key not in waypoint_cache:
            waypoint_cache[key] = load_raceline_waypoints(
                scenario.map_name,
                f"{scenario.ego_raceline}.csv",
            )
        waypoints = waypoint_cache[key]
        evaluation_xy = waypoints[np.asarray(evaluation_startpoints(scenario.map_name)), :2]
        distance = float(
            np.linalg.norm(evaluation_xy - waypoints[scenario.ego_idx, :2], axis=1).min()
        )
        minimum = min(minimum, distance)
    return minimum


def _validate_final_pool(scenarios: Sequence[ScenarioSpec]) -> float:
    scenarios = tuple(scenarios)
    if not scenarios or len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise RuntimeError("Final collision pool must contain unique scenarios")
    physical_keys = set()
    for scenario in scenarios:
        reset_spec = scenario.to_reset_spec("collision")
        if not (
            np.isfinite(reset_spec.poses).all()
            and np.isfinite(reset_spec.initial_speed_feature)
        ):
            raise RuntimeError(f"Non-finite final collision scenario: {scenario.scenario_id}")
        key = (
            scenario.map_name,
            scenario.ego_raceline,
            scenario.ego_idx,
            scenario.opp_raceline,
            scenario.opp_idx,
            _speed_milli(scenario.opp_speedscale),
            scenario.sim_duration,
            scenario.timestep,
            scenario.integrator,
        )
        if key in physical_keys:
            raise RuntimeError(f"Duplicate final collision physical key: {key}")
        physical_keys.add(key)
    minimum_distance = _minimum_evaluation_distance(scenarios)
    if minimum_distance < 1.0 - 1e-12:
        raise RuntimeError(
            f"Final collision pool violates evaluation separation: {minimum_distance:.9f}m"
        )
    return minimum_distance


def _asset_hashes(map_name: str) -> tuple[dict[str, str], dict[str, str]]:
    map_dir = PROJECT_ROOT / "f1tenth_racetracks" / map_name
    map_paths = (
        map_dir / f"{map_name}_map.png",
        map_dir / f"{map_name}_map.yaml",
    )
    raceline_paths = tuple(
        map_dir / f"{raceline}.csv"
        for raceline in (EGO_RACELINE, *OPPONENT_RACELINES)
    )
    for path in (*map_paths, *raceline_paths):
        if not path.is_file():
            raise FileNotFoundError(f"Hard-neighbor cache identity asset is missing: {path}")
    map_hashes = {path.name: _sha256_file(path) for path in map_paths}
    raceline_hashes = {path.name: _sha256_file(path) for path in raceline_paths}
    return map_hashes, raceline_hashes


def _hard_cache_config(
    args: Any,
    base_cache_dir: Path,
    base_candidates: Sequence[ScenarioSpec],
    discovery: BoundaryDiscovery,
) -> dict[str, Any]:
    actor_path = Path(args.pretrained_model_path).expanduser().resolve()
    map_hashes, raceline_hashes = _asset_hashes(str(args.map_name))
    lattice_config = PROJECT_ROOT / "latticeplanner" / "lattice_config.yaml"
    if not lattice_config.is_file():
        raise FileNotFoundError(f"Opponent planner config is missing: {lattice_config}")
    base_hashes = {
        name: _sha256_file(base_cache_dir / name)
        for name in BASE_CACHE_FILES
    }
    return {
        "classification_schema": HARD_NEIGHBOR_CACHE_SCHEMA,
        "classifier_contract": CLASSIFIER_CONTRACT,
        "base_actor_path": str(actor_path),
        "base_actor_sha256": _sha256_file(actor_path),
        "hidden_scale": int(args.hidden_scale),
        "map_name": str(args.map_name),
        "map_asset_sha256": map_hashes,
        "raceline_asset_sha256": raceline_hashes,
        "opponent_planner_config_sha256": _sha256_file(lattice_config),
        "base_cache_dir": str(base_cache_dir),
        "base_cache_sha256": base_hashes,
        "base_candidate_count": len(base_candidates),
        "base_candidate_generator": {
            "ego_raceline": EGO_RACELINE,
            "opponent_racelines": list(OPPONENT_RACELINES),
            "interval_indices": list(COLLISION_INTERVAL_INDICES),
            "speed_scales": list(COLLISION_SPEED_SCALES),
        },
        "boundary_generator": {
            "schema": BOUNDARY_GENERATOR_SCHEMA,
            "adjacency": "one_configured_axis_at_a_time",
            "speed_refinement": "exact_interior_midpoint",
            "interval_refinement": "all_interior_integer_offsets",
            "speed_fixed_point_scale": SPEED_FIXED_POINT_SCALE,
            "max_candidates_per_family": HARD_NEIGHBOR_MAX_CANDIDATES_PER_FAMILY,
        },
        "boundary_pair_count": len(discovery.pair_records),
        "boundary_generated_candidate_count": discovery.generated_candidate_count,
        "boundary_selected_candidate_count": len(discovery.candidates),
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _summary(
    base_outcomes: Sequence[dict[str, Any]],
    discovery: BoundaryDiscovery,
    boundary_outcomes: Sequence[dict[str, Any]],
    final_scenarios: Sequence[ScenarioSpec],
    minimum_evaluation_distance_m: float,
) -> dict[str, Any]:
    base_counts = Counter(record["outcome"] for record in base_outcomes)
    boundary_counts = Counter(record["outcome"] for record in boundary_outcomes)
    per_raceline = Counter(scenario.opp_raceline for scenario in final_scenarios)
    per_interval = Counter(str(scenario.interval_idx) for scenario in final_scenarios)
    per_speed = Counter(
        f"{_speed_milli(scenario.opp_speedscale) / SPEED_FIXED_POINT_SCALE:.3f}"
        for scenario in final_scenarios
    )
    family_counts = Counter(
        (scenario.ego_idx, scenario.opp_raceline)
        for scenario in final_scenarios
    )
    return {
        "base_candidate_count": len(base_outcomes),
        "base_collision_count": base_counts["ego_collision"],
        "base_other_count": base_counts["other"],
        "base_invalid_count": base_counts["invalid"],
        "boundary_pair_count": len(discovery.pair_records),
        "boundary_generated_candidate_count": discovery.generated_candidate_count,
        "boundary_selected_candidate_count": len(discovery.candidates),
        "boundary_collision_count": boundary_counts["ego_collision"],
        "boundary_other_count": boundary_counts["other"],
        "boundary_invalid_count": boundary_counts["invalid"],
        "final_collision_count": len(final_scenarios),
        "final_unique_ego_start_count": len({scenario.ego_idx for scenario in final_scenarios}),
        "final_per_raceline": dict(sorted(per_raceline.items())),
        "final_per_interval": dict(sorted(per_interval.items(), key=lambda item: int(item[0]))),
        "final_per_speed": dict(sorted(per_speed.items(), key=lambda item: float(item[0]))),
        "final_max_scenarios_per_family": max(family_counts.values()),
        "final_max_family_share": max(family_counts.values()) / len(final_scenarios),
        "minimum_evaluation_start_distance_m": minimum_evaluation_distance_m,
    }


def _validate_outcomes(
    outcomes: Sequence[dict[str, Any]],
    candidates: Sequence[ScenarioSpec],
    label: str,
) -> None:
    if len(outcomes) != len(candidates):
        raise RuntimeError(f"{label} has {len(outcomes)} outcomes for {len(candidates)} candidates")
    for index, (outcome, candidate) in enumerate(zip(outcomes, candidates)):
        if (
            set(outcome) != {"candidate_index", "scenario_id", "outcome"}
            or type(outcome["candidate_index"]) is not int
            or outcome["candidate_index"] != index
            or outcome["scenario_id"] != candidate.scenario_id
            or outcome["outcome"] not in {"ego_collision", "other", "invalid"}
        ):
            raise RuntimeError(f"{label} outcome mismatch at candidate {index}")


def _expected_final_scenarios(
    base_collisions: Sequence[ScenarioSpec],
    boundary_candidates: Sequence[ScenarioSpec],
    boundary_outcomes: Sequence[dict[str, Any]],
) -> tuple[ScenarioSpec, ...]:
    boundary_collisions = tuple(
        scenario
        for scenario, outcome in zip(boundary_candidates, boundary_outcomes)
        if outcome["outcome"] == "ego_collision"
    )
    if not boundary_collisions:
        raise RuntimeError("Boundary-aware cache produced no confirmed hard-neighbor collisions")
    return tuple(base_collisions) + boundary_collisions


def _verify_manifest(cache_dir: Path) -> None:
    lines = (cache_dir / "manifest.sha256").read_text(encoding="utf-8").splitlines()
    entries = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or Path(parts[1]).name != parts[1]:
            raise RuntimeError("Hard-neighbor manifest has an invalid entry")
        digest, name = parts
        if name in entries or len(digest) != 64:
            raise RuntimeError("Hard-neighbor manifest has duplicate or invalid hashes")
        entries[name] = digest
    if set(entries) != set(SEMANTIC_CACHE_FILES):
        raise RuntimeError("Hard-neighbor manifest does not cover the semantic cache files")
    for name, expected_digest in entries.items():
        if _sha256_file(cache_dir / name) != expected_digest:
            raise RuntimeError(f"Hard-neighbor cache hash mismatch: {name}")


def _validate_build_metadata(
    path: Path,
    boundary_outcomes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    metadata = _read_json(path)
    expected_counts = {
        "candidate_count": len(boundary_outcomes),
        "collision_count": sum(
            outcome["outcome"] == "ego_collision"
            for outcome in boundary_outcomes
        ),
        "other_count": sum(
            outcome["outcome"] == "other"
            for outcome in boundary_outcomes
        ),
        "invalid_count": sum(
            outcome["outcome"] == "invalid"
            for outcome in boundary_outcomes
        ),
    }
    if set(metadata) != set(expected_counts) | {
        "env_workers",
        "wall_seconds",
        "scenarios_per_second",
    }:
        raise RuntimeError("Hard-neighbor build metadata fields are invalid")
    if any(
        type(metadata[name]) is not int or metadata[name] != value
        for name, value in expected_counts.items()
    ):
        raise RuntimeError("Hard-neighbor build metadata counts do not match its outcomes")
    if type(metadata["env_workers"]) is not int or metadata["env_workers"] <= 0:
        raise RuntimeError("Hard-neighbor build metadata has invalid env_workers")
    for name in ("wall_seconds", "scenarios_per_second"):
        value = metadata[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise RuntimeError(f"Hard-neighbor build metadata has invalid {name}")
    return metadata


def _hard_cache_exists(cache_dir: Path) -> bool:
    if not cache_dir.exists():
        return False
    if not cache_dir.is_dir():
        raise RuntimeError(f"Hard-neighbor cache path is not a directory: {cache_dir}")
    names = {path.name for path in cache_dir.iterdir()}
    if names != set(HARD_CACHE_FILES):
        raise RuntimeError(
            "Hard-neighbor cache is incomplete or contains unexpected files; "
            "use a new empty --hard_neighbor_cache_dir"
        )
    return True


def _publish_hard_cache(
    cache_dir: Path,
    config: dict[str, Any],
    base_outcomes: Sequence[dict[str, Any]],
    discovery: BoundaryDiscovery,
    boundary_outcomes: Sequence[dict[str, Any]],
    final_scenarios: Sequence[ScenarioSpec],
    summary: dict[str, Any],
    build_metadata: dict[str, Any],
) -> None:
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite hard-neighbor cache; choose a new directory: {cache_dir}"
        )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{cache_dir.name}.building-", dir=cache_dir.parent)
    )
    try:
        _write_json(temporary / "classification_config.json", config)
        _write_jsonl(temporary / "base_candidate_outcomes.jsonl", base_outcomes)
        _write_jsonl(temporary / "boundary_pairs.jsonl", discovery.pair_records)
        _write_jsonl(temporary / "boundary_candidate_outcomes.jsonl", boundary_outcomes)
        _write_json(
            temporary / "collision_scenarios.json",
            [asdict(scenario) for scenario in final_scenarios],
        )
        _write_json(temporary / "classification_summary.json", summary)
        _write_json(temporary / "build_metadata.json", build_metadata)
        manifest = "".join(
            f"{_sha256_file(temporary / name)}  {name}\n"
            for name in SEMANTIC_CACHE_FILES
        )
        (temporary / "manifest.sha256").write_text(manifest, encoding="utf-8")
        os.rename(temporary, cache_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def load_hard_neighbor_cache(
    cache_dir: Path,
    current_config: dict[str, Any],
    base_collisions: Sequence[ScenarioSpec],
    base_outcomes: Sequence[dict[str, Any]],
    discovery: BoundaryDiscovery,
    boundary_candidates: Sequence[ScenarioSpec],
) -> tuple[tuple[ScenarioSpec, ...], dict[str, Any]]:
    _verify_manifest(cache_dir)
    cached_config = _read_json(cache_dir / "classification_config.json")
    if cached_config != current_config:
        raise RuntimeError(
            "Hard-neighbor cache identity does not match the current actor, assets, base cache, "
            "or generator; use a new --hard_neighbor_cache_dir"
        )
    cached_base_outcomes = _read_jsonl(cache_dir / "base_candidate_outcomes.jsonl")
    if cached_base_outcomes != list(base_outcomes):
        raise RuntimeError("Hard-neighbor cache base outcomes do not match the validated base cache")
    cached_pairs = _read_jsonl(cache_dir / "boundary_pairs.jsonl")
    if cached_pairs != list(discovery.pair_records):
        raise RuntimeError("Hard-neighbor boundary pairs are not deterministically reconstructable")
    boundary_outcomes = _read_jsonl(cache_dir / "boundary_candidate_outcomes.jsonl")
    _validate_outcomes(boundary_outcomes, boundary_candidates, "Hard-neighbor cache")
    expected_final = _expected_final_scenarios(
        base_collisions,
        boundary_candidates,
        boundary_outcomes,
    )
    records = _read_json(cache_dir / "collision_scenarios.json")
    expected_records = [asdict(scenario) for scenario in expected_final]
    if records != expected_records:
        raise RuntimeError("Hard-neighbor final collision scenarios do not match confirmed outcomes")
    minimum_distance = _validate_final_pool(expected_final)
    expected_summary = _summary(
        base_outcomes,
        discovery,
        boundary_outcomes,
        expected_final,
        minimum_distance,
    )
    if _read_json(cache_dir / "classification_summary.json") != expected_summary:
        raise RuntimeError("Hard-neighbor cache summary does not match its evidence")
    _validate_build_metadata(
        cache_dir / "build_metadata.json",
        boundary_outcomes,
    )
    return expected_final, expected_summary


def resolve_training_collision_scenarios(
    args: Any,
    base_candidates: tuple[ScenarioSpec, ...],
    start_method: str,
) -> tuple[tuple[ScenarioSpec, ...], dict[str, Any]]:
    """Resolve either the unchanged baseline pool or the fixed hard-neighbor pool."""

    base_collisions, base_cache_hit, base_reclassified = resolve_collision_scenarios(
        args,
        base_candidates,
        start_method,
    )
    base_cache_dir = Path(args.collision_cache_dir).expanduser().resolve()
    base_config = collision_classification_config(args, len(base_candidates))
    loaded_base_collisions, base_outcomes, _base_summary = load_collision_cache_artifacts(
        base_cache_dir,
        base_config,
        base_candidates,
    )
    if loaded_base_collisions != base_collisions:
        raise RuntimeError("Resolved base collision pool changed while loading its evidence")
    base_info = {
        "mode": "baseline",
        "hard_neighbors": False,
        "cache_dir": str(base_cache_dir),
        "base_cache_dir": str(base_cache_dir),
        "base_cache_hit": base_cache_hit,
        "base_reclassified": base_reclassified,
        "base_candidate_count": len(base_candidates),
        "base_collision_count": len(base_collisions),
        "collision_count": len(base_collisions),
    }
    if not bool(args.hard_neighbors):
        return base_collisions, base_info

    discovery = discover_boundary_candidates(base_candidates, base_outcomes)
    boundary_candidates = materialize_boundary_candidates(
        discovery.candidates,
        base_candidates,
    )
    if not boundary_candidates:
        raise RuntimeError("The base cache contains no refinable collision/other boundaries")
    hard_cache_dir = Path(args.hard_neighbor_cache_dir).expanduser().resolve()
    current_config = _hard_cache_config(
        args,
        base_cache_dir,
        base_candidates,
        discovery,
    )
    if _hard_cache_exists(hard_cache_dir):
        final_scenarios, hard_summary = load_hard_neighbor_cache(
            hard_cache_dir,
            current_config,
            base_collisions,
            base_outcomes,
            discovery,
            boundary_candidates,
        )
        hard_cache_hit = True
        print(
            f"Hard-neighbor cache hit: loaded {len(final_scenarios)} collision scenarios "
            f"({hard_summary['boundary_collision_count']} boundary additions)",
            flush=True,
        )
    else:
        print(
            f"Hard-neighbor cache miss: classifying {len(boundary_candidates)} "
            f"candidates from {len(discovery.pair_records)} boundary pairs",
            flush=True,
        )
        boundary_collisions, boundary_outcomes, build_metadata = classify_collision_scenarios(
            args.pretrained_model_path,
            args.hidden_scale,
            args.map_name,
            args.env_workers,
            boundary_candidates,
            start_method,
        )
        _validate_outcomes(boundary_outcomes, boundary_candidates, "New hard-neighbor cache")
        final_scenarios = tuple(base_collisions) + tuple(boundary_collisions)
        minimum_distance = _validate_final_pool(final_scenarios)
        hard_summary = _summary(
            base_outcomes,
            discovery,
            boundary_outcomes,
            final_scenarios,
            minimum_distance,
        )
        _publish_hard_cache(
            hard_cache_dir,
            current_config,
            base_outcomes,
            discovery,
            boundary_outcomes,
            final_scenarios,
            hard_summary,
            build_metadata,
        )
        final_scenarios, hard_summary = load_hard_neighbor_cache(
            hard_cache_dir,
            current_config,
            base_collisions,
            base_outcomes,
            discovery,
            boundary_candidates,
        )
        hard_cache_hit = False
        print(
            f"Hard-neighbor cache built: {len(base_collisions)} base + "
            f"{hard_summary['boundary_collision_count']} confirmed boundary collisions = "
            f"{len(final_scenarios)}",
            flush=True,
        )
    info = {
        **base_info,
        "mode": "boundary_aware",
        "hard_neighbors": True,
        "cache_dir": str(hard_cache_dir),
        "hard_neighbor_cache_dir": str(hard_cache_dir),
        "hard_neighbor_cache_hit": hard_cache_hit,
        "boundary_pair_count": hard_summary["boundary_pair_count"],
        "boundary_candidate_count": hard_summary["boundary_selected_candidate_count"],
        "boundary_collision_count": hard_summary["boundary_collision_count"],
        "collision_count": len(final_scenarios),
    }
    return final_scenarios, info
