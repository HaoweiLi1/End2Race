"""Strict loading for a pre-built fixed collision-role training pool."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from ppo.scenarios import ScenarioSpec


FIXED_COLLISION_POOL_SCHEMA = 1
ALLOWED_SOURCE_LABELS = frozenset({"ego_collision", "near_miss"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixed_collision_pool(
    path: str | Path,
    *,
    map_name: str,
) -> tuple[tuple[ScenarioSpec, ...], dict[str, Any]]:
    """Load and validate one immutable collision-role scenario pool."""

    pool_path = Path(path).expanduser().resolve()
    if not pool_path.is_file():
        raise FileNotFoundError(f"Fixed collision pool does not exist: {pool_path}")
    with pool_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    expected_top_level = {
        "schema_version",
        "purpose",
        "source",
        "selection",
        "sampling",
        "entries",
    }
    if not isinstance(payload, dict) or set(payload) != expected_top_level:
        raise RuntimeError("Fixed collision pool has invalid top-level fields")
    if payload["schema_version"] != FIXED_COLLISION_POOL_SCHEMA:
        raise RuntimeError("Fixed collision pool schema does not match")
    if not isinstance(payload["purpose"], str) or not payload["purpose"].strip():
        raise RuntimeError("Fixed collision pool purpose must be non-empty")

    selection = payload["selection"]
    expected_selection_fields = {
        "split",
        "interval_idx",
        "near_miss_clearance_m",
        "include_outcomes",
    }
    if (
        not isinstance(selection, dict)
        or set(selection) != expected_selection_fields
        or selection["split"] != "train"
        or type(selection["interval_idx"]) is not int
        or selection["interval_idx"] <= 0
        or not isinstance(selection["near_miss_clearance_m"], (int, float))
        or selection["near_miss_clearance_m"] <= 0.0
        or selection["include_outcomes"]
        != ["ego_collision", "overtake_or_follow_near_miss"]
    ):
        raise RuntimeError("Fixed collision pool selection contract is invalid")

    source = payload["source"]
    expected_source_fields = {
        "root",
        "design_manifest_sha256",
        "candidate_scenarios_sha256",
        "candidate_labels_sha256",
        "selection_actor_sha256",
    }
    if not isinstance(source, dict) or set(source) != expected_source_fields:
        raise RuntimeError("Fixed collision pool source evidence is invalid")
    if any(
        not isinstance(source[name], str) or not source[name]
        for name in expected_source_fields
    ):
        raise RuntimeError("Fixed collision pool source evidence must be non-empty")

    entries = payload["entries"]
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("Fixed collision pool must contain entries")
    scenarios: list[ScenarioSpec] = []
    source_labels: list[str] = []
    source_ids: list[str] = []
    expected_entry_fields = {
        "source_label",
        "source_outcome",
        "min_obb_clearance_m",
        "scenario",
    }
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != expected_entry_fields:
            raise RuntimeError(
                f"Fixed collision pool entry {index} has invalid fields"
            )
        source_label = entry["source_label"]
        if source_label not in ALLOWED_SOURCE_LABELS:
            raise RuntimeError(
                f"Fixed collision pool entry {index} has invalid source label"
            )
        scenario_record = entry["scenario"]
        if not isinstance(scenario_record, dict):
            raise RuntimeError(
                f"Fixed collision pool entry {index} scenario is invalid"
            )
        scenario = ScenarioSpec(**scenario_record)
        if (
            scenario.map_name != map_name
            or scenario.pool != "collision"
            or scenario.interval_idx != selection["interval_idx"]
        ):
            raise RuntimeError(
                f"Fixed collision pool entry {index} violates map/pool/interval"
            )
        source_outcome = entry["source_outcome"]
        clearance = entry["min_obb_clearance_m"]
        if source_label == "ego_collision":
            if source_outcome != "ego_collision":
                raise RuntimeError(
                    f"Fixed collision pool entry {index} collision label disagrees"
                )
        else:
            if (
                source_outcome not in {"overtake", "follow"}
                or isinstance(clearance, bool)
                or not isinstance(clearance, (int, float))
                or not 0.0
                <= float(clearance)
                <= float(selection["near_miss_clearance_m"])
            ):
                raise RuntimeError(
                    f"Fixed collision pool entry {index} near-miss label disagrees"
                )
        scenarios.append(scenario)
        source_labels.append(source_label)
        source_ids.append(scenario.scenario_id)

    if len(set(source_ids)) != len(source_ids):
        raise RuntimeError("Fixed collision pool scenario IDs must be unique")
    physical_keys = {
        (
            scenario.map_name,
            scenario.ego_raceline,
            scenario.ego_idx,
            scenario.opp_raceline,
            scenario.opp_idx,
            scenario.opp_speedscale,
            scenario.interval_idx,
            scenario.sim_duration,
            scenario.timestep,
            scenario.integrator,
        )
        for scenario in scenarios
    }
    if len(physical_keys) != len(scenarios):
        raise RuntimeError("Fixed collision pool physical scenarios must be unique")

    counts = dict(sorted(Counter(source_labels).items()))
    sampling = payload["sampling"]
    expected_sampling = {
        "mode": "uniform_cycle_over_combined_pool",
        "scenario_count": len(scenarios),
        "source_label_counts": counts,
    }
    if sampling != expected_sampling:
        raise RuntimeError("Fixed collision pool sampling metadata disagrees")

    info = {
        "mode": "fixed_collision_pool_file",
        "fixed_collision_pool_file": str(pool_path),
        "fixed_collision_pool_sha256": sha256_file(pool_path),
        "fixed_collision_pool_purpose": payload["purpose"],
        "fixed_collision_pool_source": source,
        "fixed_collision_pool_selection": selection,
        "fixed_collision_pool_sampling": sampling,
        "collision_count": len(scenarios),
    }
    return tuple(scenarios), info
