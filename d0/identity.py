"""Canonical D0.1 identities, geometry sets, and append-only registry."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


DOMAINS = {
    "L1": b"end2race:d0.1:l1-occurrence:v1\0",
    "L2": b"end2race:d0.1:l2-scenario:v1\0",
    "L3": b"end2race:d0.1:l3-exact-start:v1\0",
    "L4": b"end2race:d0.1:l4-block:v1\0",
}
SCHEMAS = {
    "L1": "d0.1-l1-occurrence-1",
    "L2": "d0.1-l2-scenario-1",
    "L3": "d0.1-l3-exact-start-1",
    "L4": "d0.1-l4-block-1",
}
PAYLOAD_FIELDS = {
    "L1": (
        "schema",
        "source_run_id",
        "model_id",
        "model_relpath",
        "checkpoint_sha256",
        "map_name",
        "grid_id",
        "offset",
        "tag",
        "result_json_relpath",
        "result_json_sha256",
        "episode_key",
        "npz_relpath",
        "npz_sha256",
        "l2_id",
    ),
    "L2": (
        "schema",
        "asset_namespace_sha256",
        "map_name",
        "ego_raceline",
        "opponent_raceline",
        "ego_start_pose_hex",
        "opponent_start_pose_hex",
        "ego_waypoint_speed_hex",
        "ego_prev_speed_input_hex",
        "ego_initial_actual_speed_hex",
        "opponent_initial_actual_speed_hex",
        "opponent_speedscale_hex",
        "interval_idx",
        "sim_dt_hex",
        "duration_ticks",
        "noise_fraction_hex",
        "noise_seed",
    ),
    "L3": (
        "schema",
        "asset_namespace_sha256",
        "map_name",
        "ego_raceline",
        "ego_start_pose_hex",
    ),
    "L4": (
        "schema",
        "asset_namespace_sha256",
        "map_name",
        "ego_raceline",
        "member_l3_ids",
    ),
}

REGISTRY_FIELDS = (
    "registry_schema",
    "row_id",
    "opened_at_utc",
    "stage",
    "use_class",
    "split_id",
    "l2_id",
    "l3_id",
    "l4_id",
    "map_name",
    "source_manifest_sha256",
    "source_run_id",
    "decision_effect",
    "final_pool",
    "evidence_relpath",
)
REGISTRY_SCHEMA = "bplus-opened-registry-1"
REGISTRY_DOMAIN = b"end2race:bplus:opened-registry-row:v1\0"
USE_CLASSES = {
    "historical_analysis",
    "probe_fit",
    "probe_select",
    "oracle_search",
    "actor_pretrain",
    "ppo_train",
    "snapshot_select",
    "medium_confirm",
    "final_evaluation",
}
DECISION_EFFECTS = {
    "historical_only",
    "representation_choice",
    "action_choice",
    "reward_choice",
    "curriculum_choice",
    "model_choice",
    "final_only",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^L[1-4]:[0-9a-f]{64}$")


@dataclass(frozen=True)
class AssetNamespace:
    sha256: str
    entries: tuple[tuple[str, str], ...]
    canonical_bytes: bytes


@dataclass(frozen=True)
class BlockManifest:
    l3_to_l4: dict[str, str]
    components: tuple[dict, ...]


@dataclass(frozen=True)
class RegistryAppendResult:
    appended: int
    skipped: int
    total: int


@dataclass(frozen=True)
class ResolvedScenario:
    map_name: str
    offset: int
    start_ordinal: int
    raw_ego_idx: int
    resolved_ego_idx: int
    resolved_opp_idx: int
    ego_raceline: str
    opponent_raceline: str
    ego_pose: tuple[float, float, float]
    opponent_pose: tuple[float, float, float]
    ego_waypoint_speed: float
    opponent_speedscale: float
    interval_idx: int


@dataclass(frozen=True)
class S0Outputs:
    asset_namespace: AssetNamespace
    occurrences: tuple[dict, ...]
    scenarios: tuple[dict, ...]
    block_manifest: BlockManifest
    dev_nodes: tuple[dict, ...]
    dev_l3_ids: frozenset[str]
    sets: dict[str, frozenset[str]]
    sensitivity_a_pairs: tuple[dict, ...]
    sensitivity_b: dict
    reconciliation: dict


def canonical_json(payload) -> bytes:
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload is not canonical-JSON encodable: {exc}") from exc
    return text.encode("utf-8")


def _finite_hex(value) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("identity floats must be finite")
    return value.hex()


def _pose_hex(values: Sequence[float]) -> list[str]:
    if len(values) != 3:
        raise ValueError("pose must contain exactly x,y,heading")
    return [_finite_hex(v) for v in values]


def _validate_sha(value: str, name: str) -> None:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")


def _validate_payload(layer: str, payload: Mapping) -> None:
    if layer not in PAYLOAD_FIELDS:
        raise ValueError(f"unknown identity layer: {layer}")
    actual = set(payload)
    expected = set(PAYLOAD_FIELDS[layer])
    if actual != expected:
        raise ValueError(
            f"{layer} payload key set mismatch: missing={sorted(expected-actual)} "
            f"extra={sorted(actual-expected)}"
        )
    if payload["schema"] != SCHEMAS[layer]:
        raise ValueError(f"{layer} schema mismatch")
    if layer in {"L2", "L3", "L4"}:
        _validate_sha(payload["asset_namespace_sha256"], "asset_namespace_sha256")
    if layer == "L1":
        _validate_sha(payload["checkpoint_sha256"], "checkpoint_sha256")
        _validate_sha(payload["result_json_sha256"], "result_json_sha256")
        _validate_sha(payload["npz_sha256"], "npz_sha256")
        if not ID_RE.fullmatch(str(payload["l2_id"])) or not str(payload["l2_id"]).startswith("L2:"):
            raise ValueError("L1 l2_id is invalid")
    if layer in {"L2", "L3"}:
        pose_keys = ["ego_start_pose_hex"]
        if layer == "L2":
            pose_keys.append("opponent_start_pose_hex")
        for key in pose_keys:
            values = payload[key]
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError(f"{key} must contain three float.hex strings")
            for value in values:
                if not isinstance(value, str) or not math.isfinite(float.fromhex(value)):
                    raise ValueError(f"invalid {key} value")
    if layer == "L4":
        members = payload["member_l3_ids"]
        if not isinstance(members, list) or not members or members != sorted(set(members)):
            raise ValueError("L4 member_l3_ids must be a nonempty sorted unique list")
        if any(not ID_RE.fullmatch(x) or not x.startswith("L3:") for x in members):
            raise ValueError("invalid L3 member in L4 payload")


def domain_id(layer: str, payload: Mapping) -> str:
    _validate_payload(layer, payload)
    digest = hashlib.sha256(DOMAINS[layer] + canonical_json(payload)).hexdigest()
    return f"{layer}:{digest}"


def file_sha256(path: os.PathLike | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def asset_namespace_from_entries(entries: Iterable[Mapping[str, str]]) -> AssetNamespace:
    normalized = []
    for entry in entries:
        if set(entry) != {"relpath", "sha256"}:
            raise ValueError("asset entry must contain exactly relpath and sha256")
        relpath = str(entry["relpath"])
        if Path(relpath).is_absolute() or ".." in Path(relpath).parts:
            raise ValueError("asset relpath must be repository-relative")
        _validate_sha(entry["sha256"], "asset sha256")
        normalized.append({"relpath": relpath, "sha256": entry["sha256"]})
    normalized.sort(key=lambda x: x["relpath"])
    if len({x["relpath"] for x in normalized}) != len(normalized):
        raise ValueError("duplicate asset relpath")
    encoded = canonical_json(normalized)
    digest = hashlib.sha256(b"end2race:d0.1:asset-namespace:v1\0" + encoded).hexdigest()
    return AssetNamespace(
        sha256=digest,
        entries=tuple((x["relpath"], x["sha256"]) for x in normalized),
        canonical_bytes=encoded,
    )


def make_l1_payload(**values) -> dict:
    payload = {"schema": SCHEMAS["L1"], **values}
    _validate_payload("L1", payload)
    return payload


def make_l2_payload(
    asset_namespace_sha256: str,
    map_name: str,
    ego_raceline: str,
    opponent_raceline: str,
    ego_start_pose: Sequence[float],
    opponent_start_pose: Sequence[float],
    ego_waypoint_speed: float,
    ego_prev_speed_input: float,
    ego_initial_actual_speed: float,
    opponent_initial_actual_speed: float,
    opponent_speedscale: float,
    interval_idx: int,
    sim_dt: float,
    duration_ticks: int,
    noise_fraction: float,
    noise_seed: int,
) -> dict:
    payload = {
        "schema": SCHEMAS["L2"],
        "asset_namespace_sha256": asset_namespace_sha256,
        "map_name": str(map_name),
        "ego_raceline": str(ego_raceline),
        "opponent_raceline": str(opponent_raceline),
        "ego_start_pose_hex": _pose_hex(ego_start_pose),
        "opponent_start_pose_hex": _pose_hex(opponent_start_pose),
        "ego_waypoint_speed_hex": _finite_hex(ego_waypoint_speed),
        "ego_prev_speed_input_hex": _finite_hex(ego_prev_speed_input),
        "ego_initial_actual_speed_hex": _finite_hex(ego_initial_actual_speed),
        "opponent_initial_actual_speed_hex": _finite_hex(opponent_initial_actual_speed),
        "opponent_speedscale_hex": _finite_hex(opponent_speedscale),
        "interval_idx": int(interval_idx),
        "sim_dt_hex": _finite_hex(sim_dt),
        "duration_ticks": int(duration_ticks),
        "noise_fraction_hex": _finite_hex(noise_fraction),
        "noise_seed": int(noise_seed),
    }
    _validate_payload("L2", payload)
    return payload


def make_l3_payload(
    asset_namespace_sha256: str,
    map_name: str,
    ego_raceline: str,
    ego_start_pose: Sequence[float],
) -> dict:
    payload = {
        "schema": SCHEMAS["L3"],
        "asset_namespace_sha256": asset_namespace_sha256,
        "map_name": str(map_name),
        "ego_raceline": str(ego_raceline),
        "ego_start_pose_hex": _pose_hex(ego_start_pose),
    }
    _validate_payload("L3", payload)
    return payload


def build_l4_blocks(nodes: Iterable[Mapping]) -> BlockManifest:
    by_id: dict[str, dict] = {}
    for raw in nodes:
        node = dict(raw)
        required = {
            "l3_id",
            "asset_namespace_sha256",
            "map_name",
            "ego_raceline",
            "x",
            "y",
            "is_dev",
        }
        if set(node) != required:
            raise ValueError("L4 node key set mismatch")
        if not ID_RE.fullmatch(str(node["l3_id"])) or not str(node["l3_id"]).startswith("L3:"):
            raise ValueError("invalid L3 node ID")
        _validate_sha(node["asset_namespace_sha256"], "asset_namespace_sha256")
        node["x"] = float(node["x"])
        node["y"] = float(node["y"])
        if not math.isfinite(node["x"]) or not math.isfinite(node["y"]):
            raise ValueError("L4 coordinates must be finite")
        node["is_dev"] = bool(node["is_dev"])
        previous = by_id.get(node["l3_id"])
        if previous is not None:
            comparable = dict(previous, is_dev=False)
            candidate = dict(node, is_dev=False)
            if comparable != candidate:
                raise ValueError("one L3 ID resolved to conflicting nodes")
            previous["is_dev"] = previous["is_dev"] or node["is_dev"]
        else:
            by_id[node["l3_id"]] = node

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for node in by_id.values():
        key = (node["asset_namespace_sha256"], node["map_name"], node["ego_raceline"])
        groups.setdefault(key, []).append(node)

    l3_to_l4: dict[str, str] = {}
    components: list[dict] = []
    for group_key in sorted(groups):
        group = sorted(groups[group_key], key=lambda n: n["l3_id"])
        parent = list(range(len(group)))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left, right):
            a, b = find(left), find(right)
            if a != b:
                if a > b:
                    a, b = b, a
                parent[b] = a

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if math.hypot(group[i]["x"] - group[j]["x"], group[i]["y"] - group[j]["y"]) <= 1.0:
                    union(i, j)

        gathered: dict[int, list[dict]] = {}
        for i, node in enumerate(group):
            gathered.setdefault(find(i), []).append(node)
        for members in sorted(gathered.values(), key=lambda m: sorted(x["l3_id"] for x in m)):
            ids = sorted(x["l3_id"] for x in members)
            payload = {
                "schema": SCHEMAS["L4"],
                "asset_namespace_sha256": group_key[0],
                "map_name": group_key[1],
                "ego_raceline": group_key[2],
                "member_l3_ids": ids,
            }
            l4_id = domain_id("L4", payload)
            component = {
                "l4_id": l4_id,
                "asset_namespace_sha256": group_key[0],
                "map_name": group_key[1],
                "ego_raceline": group_key[2],
                "member_l3_ids": tuple(ids),
                "contains_dev": any(x["is_dev"] for x in members),
                "nodes": tuple(
                    {
                        "l3_id": x["l3_id"],
                        "x": x["x"],
                        "y": x["y"],
                        "is_dev": x["is_dev"],
                    }
                    for x in sorted(members, key=lambda n: n["l3_id"])
                ),
            }
            components.append(component)
            for l3_id in ids:
                l3_to_l4[l3_id] = l4_id
    components.sort(key=lambda x: x["l4_id"])
    return BlockManifest(l3_to_l4=l3_to_l4, components=tuple(components))


def sensitivity_a_pairs(records: Iterable[Mapping], require_full_pattern: bool = False) -> tuple[dict, ...]:
    groups: dict[bytes, list[dict]] = {}
    for raw in records:
        record = dict(raw)
        required = {"l2_id", "l2_payload", "resolved_ego_indices", "endpoint_ego_pose_speed_equal"}
        if set(record) != required:
            raise ValueError("Sensitivity-A record key set mismatch")
        payload = record["l2_payload"]
        _validate_payload("L2", payload)
        if record["l2_id"] != domain_id("L2", payload):
            raise ValueError("Sensitivity-A L2 ID mismatch")
        if not record["resolved_ego_indices"]:
            raise ValueError("Sensitivity-A record lacks resolved indices")
        reduced = {key: value for key, value in payload.items() if key != "opponent_start_pose_hex"}
        groups.setdefault(canonical_json(reduced), []).append(record)

    rows = []
    for members in groups.values():
        unique = {m["l2_id"]: m for m in members}
        members = list(unique.values())
        if len(members) == 1:
            continue
        if len(members) != 2:
            raise ValueError("Sensitivity-A qualifying group must contain exactly two L2 IDs")
        if not all(bool(m["endpoint_ego_pose_speed_equal"]) for m in members):
            raise ValueError("Sensitivity-A endpoint ego pose/speed equality failed")
        members.sort(key=lambda m: (min(int(x) for x in m["resolved_ego_indices"]), m["l2_id"]))
        retained, excluded = members
        retained_idx = min(int(x) for x in retained["resolved_ego_indices"])
        excluded_idx = min(int(x) for x in excluded["resolved_ego_indices"])
        if retained_idx >= excluded_idx:
            raise ValueError("Sensitivity-A retained index must be strictly smaller")
        payload = retained["l2_payload"]
        pair_payload = {
            "retained_l2_id": retained["l2_id"],
            "excluded_l2_id": excluded["l2_id"],
        }
        pair_id = hashlib.sha256(
            b"end2race:d0.1:sensa-pair:v1\0" + canonical_json(pair_payload)
        ).hexdigest()
        rows.append(
            {
                "pair_id": pair_id,
                "map_name": payload["map_name"],
                "ego_raceline": payload["ego_raceline"],
                "opponent_raceline": payload["opponent_raceline"],
                "speedscale_hex": payload["opponent_speedscale_hex"],
                "interval_idx": payload["interval_idx"],
                "retained_l2_id": retained["l2_id"],
                "retained_min_resolved_ego_idx": retained_idx,
                "excluded_l2_id": excluded["l2_id"],
                "excluded_min_resolved_ego_idx": excluded_idx,
                "rule_version": "d0.1-sensa-1",
            }
        )
    rows.sort(key=lambda x: (x["map_name"], x["speedscale_hex"], x["retained_l2_id"]))

    if require_full_pattern:
        expected = {
            (map_name, "raceline1", float(speed).hex(), 15)
            for map_name in ("Nuerburgring", "MoscowRaceway", "Hockenheim")
            for speed in (0.5, 0.6, 0.7, 0.8)
        }
        observed = {
            (row["map_name"], row["opponent_raceline"], row["speedscale_hex"], row["interval_idx"])
            for row in rows
        }
        if len(rows) != 12 or observed != expected:
            raise ValueError(
                f"Sensitivity-A full pattern mismatch: n={len(rows)} "
                f"missing={sorted(expected-observed)} extra={sorted(observed-expected)}"
            )
        if len({x["retained_l2_id"] for x in rows}) != 12 or len({x["excluded_l2_id"] for x in rows}) != 12:
            raise ValueError("Sensitivity-A retained/excluded IDs are not unique")
    return tuple(rows)


def sensitivity_b_membership(
    exact_l2_to_l3: Mapping[str, str],
    primary_excluded_ids: set[str] | frozenset[str],
    l3_to_l4: Mapping[str, str],
    dev_l4_ids: set[str] | frozenset[str],
) -> dict:
    exact_ids = set(exact_l2_to_l3)
    primary_excluded = set(primary_excluded_ids)
    if not primary_excluded <= exact_ids:
        raise ValueError("primary exclusions must be a subset of exact")
    missing = {l3 for l3 in exact_l2_to_l3.values() if l3 not in l3_to_l4}
    if missing:
        raise ValueError(f"L3 IDs missing block assignments: {sorted(missing)}")
    excluded = {
        l2_id
        for l2_id, l3_id in exact_l2_to_l3.items()
        if l3_to_l4[l3_id] in dev_l4_ids
    }
    already = excluded & primary_excluded
    additional = excluded - primary_excluded
    primary = exact_ids - primary_excluded
    sens_b = exact_ids - excluded
    if len(excluded) - len(already) != len(additional):
        raise AssertionError("Sensitivity-B exclusion accounting failed")
    if len(primary) - len(additional) != len(sens_b):
        raise AssertionError("Sensitivity-B primary accounting failed")
    return {
        "exact": exact_ids,
        "primary": primary,
        "sensB": sens_b,
        "excluded_from_exact_ids": excluded,
        "already_excluded_by_primary_ids": already,
        "additional_vs_primary_ids": additional,
        "excluded_from_exact": len(excluded),
        "already_excluded_by_primary": len(already),
        "additional_vs_primary": len(additional),
    }


def registry_row_id(row: Mapping[str, str]) -> str:
    expected = set(REGISTRY_FIELDS) - {"row_id"}
    actual = set(row) - {"row_id"}
    if actual != expected:
        raise ValueError(
            f"registry row key mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}"
        )
    payload = {field: str(row[field]) for field in REGISTRY_FIELDS if field not in {"row_id", "opened_at_utc"}}
    return hashlib.sha256(REGISTRY_DOMAIN + canonical_json(payload)).hexdigest()


def _validate_registry_row(row: Mapping[str, str]) -> dict[str, str]:
    if set(row) != set(REGISTRY_FIELDS):
        raise ValueError("registry row has wrong field set")
    normalized = {field: str(row[field]) for field in REGISTRY_FIELDS}
    if any(value == "" for value in normalized.values()):
        raise ValueError("registry fields must be nonempty")
    if normalized["registry_schema"] != REGISTRY_SCHEMA:
        raise ValueError("registry schema mismatch")
    if normalized["use_class"] not in USE_CLASSES:
        raise ValueError("invalid registry use_class")
    if normalized["decision_effect"] not in DECISION_EFFECTS:
        raise ValueError("invalid registry decision_effect")
    if normalized["final_pool"] not in {"true", "false"}:
        raise ValueError("registry final_pool must be lowercase boolean")
    _validate_sha(normalized["source_manifest_sha256"], "source_manifest_sha256")
    expected_id = registry_row_id(normalized)
    if normalized["row_id"] != expected_id:
        raise ValueError("registry row_id/content mismatch")
    return normalized


def validate_registry_row(row: Mapping[str, str]) -> dict[str, str]:
    """Return a normalized registry row or raise on any schema/domain defect."""

    return _validate_registry_row(row)


def append_opened_registry(path: os.PathLike | str, rows: Iterable[Mapping[str, str]]) -> RegistryAppendResult:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = sorted((_validate_registry_row(row) for row in rows), key=lambda x: x["row_id"])
    appended = skipped = 0
    with path.open("a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            reader = csv.DictReader(handle, delimiter="\t")
            existing: dict[str, dict[str, str]] = {}
            if reader.fieldnames is not None:
                if tuple(reader.fieldnames) != REGISTRY_FIELDS:
                    raise ValueError("opened registry header mismatch")
                for row in reader:
                    valid = _validate_registry_row(row)
                    prior = existing.get(valid["row_id"])
                    if prior is not None and prior != valid:
                        raise ValueError("conflicting duplicate registry row")
                    existing[valid["row_id"]] = valid
            handle.seek(0, os.SEEK_END)
            writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS, delimiter="\t", lineterminator="\n")
            if handle.tell() == 0:
                writer.writeheader()
            for row in incoming:
                prior = existing.get(row["row_id"])
                if prior is not None:
                    if prior != row:
                        raise ValueError("duplicate registry row_id has different content")
                    skipped += 1
                    continue
                writer.writerow(row)
                existing[row["row_id"]] = row
                appended += 1
            handle.flush()
            os.fsync(handle.fileno())
            return RegistryAppendResult(appended=appended, skipped=skipped, total=len(existing))
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_raceline(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            parts = line.strip().split(";")
            if len(parts) >= 6:
                rows.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[5])])
    if not rows:
        raise ValueError(f"raceline has no usable rows: {path}")
    return np.asarray(rows, dtype=np.float64)


def asset_namespace(runconfig: Mapping) -> AssetNamespace:
    root = Path(runconfig["repository_root"])
    assets_root = root / runconfig["assets_root"]
    entries = []
    for map_name in sorted({str(item[0]) for item in runconfig["grids"]}):
        for raceline in sorted({runconfig["ego_raceline"], *runconfig["opponent_racelines"]}):
            relpath = Path(runconfig["assets_root"]) / map_name / f"{raceline}.csv"
            path = root / relpath
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
            entries.append({"relpath": relpath.as_posix(), "sha256": file_sha256(path)})
    return asset_namespace_from_entries(entries)


def _raw_start_index(max_wc: int, offset: int, ordinal: int, start_count: int = 50) -> int:
    if start_count < 2 or not 0 <= ordinal < start_count:
        raise ValueError(f"start ordinal must be in 0..{start_count - 1}")
    if offset == 0:
        return ordinal * max_wc // (start_count - 1)
    return (ordinal * max_wc // start_count + offset) % max_wc


def resolve_scenario(
    runconfig: Mapping,
    map_name: str,
    offset: int,
    start_ordinal: int,
    opponent_raceline: str,
    speedscale: float,
    cache: dict | None = None,
) -> ResolvedScenario:
    cache = {} if cache is None else cache
    root = Path(runconfig["repository_root"]) / runconfig["assets_root"] / map_name

    def rows(raceline):
        key = (map_name, raceline)
        if key not in cache:
            cache[key] = _load_raceline(root / f"{raceline}.csv")
        return cache[key]

    ego_raceline = runconfig["ego_raceline"]
    ego_rows = rows(ego_raceline)
    max_wc = len(ego_rows) - 1
    start_count = int(runconfig.get("_test_start_count", 50))
    raw_ego_idx = _raw_start_index(max_wc, int(offset), int(start_ordinal), start_count)
    ego_idx = raw_ego_idx % len(ego_rows)
    ego_row = ego_rows[ego_idx]
    opp_rows = rows(opponent_raceline)
    if opponent_raceline == ego_raceline:
        opp_idx = (raw_ego_idx + int(runconfig["interval_idx"])) % len(opp_rows)
    else:
        nearest = int(np.argmin(np.linalg.norm(opp_rows[:, :2] - ego_row[:2], axis=1)))
        opp_idx = (nearest + int(runconfig["interval_idx"])) % len(opp_rows)
    return ResolvedScenario(
        map_name=map_name,
        offset=int(offset),
        start_ordinal=int(start_ordinal),
        raw_ego_idx=int(raw_ego_idx),
        resolved_ego_idx=int(ego_idx),
        resolved_opp_idx=int(opp_idx),
        ego_raceline=ego_raceline,
        opponent_raceline=opponent_raceline,
        ego_pose=tuple(float(x) for x in ego_row[:3]),
        opponent_pose=tuple(float(x) for x in opp_rows[opp_idx, :3]),
        ego_waypoint_speed=float(ego_row[3]),
        opponent_speedscale=float(speedscale),
        interval_idx=int(runconfig["interval_idx"]),
    )


def geometry_manifest(runconfig: Mapping) -> S0Outputs:
    namespace = asset_namespace(runconfig)
    cache: dict = {}
    occurrences = []
    scenario_records: dict[str, dict] = {}
    root = Path(runconfig["repository_root"]) / runconfig["assets_root"]

    endpoint_equal: dict[str, bool] = {}
    for map_name, _ in runconfig["grids"]:
        rows = _load_raceline(root / map_name / f"{runconfig['ego_raceline']}.csv")
        endpoint_equal[map_name] = bool(np.array_equal(rows[0, :], rows[-1, :]))

    start_count = int(runconfig.get("_test_start_count", 50))
    for map_name, offset in runconfig["grids"]:
        for ordinal in range(start_count):
            for opponent_raceline in runconfig["opponent_racelines"]:
                for speedscale in runconfig["opponent_speedscales"]:
                    resolved = resolve_scenario(
                        runconfig, map_name, int(offset), ordinal, opponent_raceline, speedscale, cache
                    )
                    l2_payload = make_l2_payload(
                        namespace.sha256,
                        resolved.map_name,
                        resolved.ego_raceline,
                        resolved.opponent_raceline,
                        resolved.ego_pose,
                        resolved.opponent_pose,
                        resolved.ego_waypoint_speed,
                        resolved.ego_waypoint_speed * 0.9,
                        0.0,
                        0.0,
                        resolved.opponent_speedscale,
                        resolved.interval_idx,
                        runconfig["sim_dt"],
                        runconfig["duration_ticks"],
                        runconfig["noise"],
                        runconfig["noise_seed"],
                    )
                    l3_payload = make_l3_payload(
                        namespace.sha256,
                        resolved.map_name,
                        resolved.ego_raceline,
                        resolved.ego_pose,
                    )
                    l2_id = domain_id("L2", l2_payload)
                    l3_id = domain_id("L3", l3_payload)
                    occurrence = {
                        "map_name": map_name,
                        "grid_id": f"{map_name}_off{offset}",
                        "offset": int(offset),
                        "start_ordinal": ordinal,
                        "raw_ego_idx": resolved.raw_ego_idx,
                        "resolved_ego_idx": resolved.resolved_ego_idx,
                        "resolved_opp_idx": resolved.resolved_opp_idx,
                        "opponent_raceline": opponent_raceline,
                        "speedscale_hex": float(speedscale).hex(),
                        "episode_key": (
                            f"ol{opponent_raceline.replace('raceline', '')}_"
                            f"e{resolved.raw_ego_idx}_o{resolved.resolved_opp_idx}_s{float(speedscale)}"
                        ),
                        "l2_id": l2_id,
                        "l3_id": l3_id,
                    }
                    occurrences.append(occurrence)
                    record = scenario_records.setdefault(
                        l2_id,
                        {
                            "l2_id": l2_id,
                            "l2_payload": l2_payload,
                            "l3_id": l3_id,
                            "l3_payload": l3_payload,
                            "resolved_ego_indices": [],
                            "endpoint_ego_pose_speed_equal": endpoint_equal[map_name],
                        },
                    )
                    if record["l2_payload"] != l2_payload or record["l3_id"] != l3_id:
                        raise AssertionError("one L2 ID resolved inconsistently")
                    record["resolved_ego_indices"].append(resolved.resolved_ego_idx)

    dev_l3_ids = set()
    dev_nodes = []
    for ordinal in range(start_count):
        resolved = resolve_scenario(
            runconfig, "Austin", 0, ordinal, "raceline1", 0.5, cache
        )
        payload = make_l3_payload(namespace.sha256, "Austin", "raceline1", resolved.ego_pose)
        l3_id = domain_id("L3", payload)
        dev_l3_ids.add(l3_id)
        dev_nodes.append(
            {
                "l3_id": l3_id,
                "asset_namespace_sha256": namespace.sha256,
                "map_name": "Austin",
                "ego_raceline": "raceline1",
                "x": resolved.ego_pose[0],
                "y": resolved.ego_pose[1],
                "is_dev": True,
            }
        )

    block_nodes = list(dev_nodes)
    for record in scenario_records.values():
        pose = record["l3_payload"]["ego_start_pose_hex"]
        block_nodes.append(
            {
                "l3_id": record["l3_id"],
                "asset_namespace_sha256": namespace.sha256,
                "map_name": record["l2_payload"]["map_name"],
                "ego_raceline": record["l2_payload"]["ego_raceline"],
                "x": float.fromhex(pose[0]),
                "y": float.fromhex(pose[1]),
                "is_dev": record["l3_id"] in dev_l3_ids,
            }
        )
    blocks = build_l4_blocks(block_nodes)
    exact_l2_to_l3 = {l2_id: record["l3_id"] for l2_id, record in scenario_records.items()}
    primary_excluded = {l2_id for l2_id, l3_id in exact_l2_to_l3.items() if l3_id in dev_l3_ids}
    pairs = sensitivity_a_pairs(
        (
            {
                "l2_id": record["l2_id"],
                "l2_payload": record["l2_payload"],
                "resolved_ego_indices": record["resolved_ego_indices"],
                "endpoint_ego_pose_speed_equal": record["endpoint_ego_pose_speed_equal"],
            }
            for record in scenario_records.values()
        ),
        require_full_pattern=bool(runconfig.get("_strict_canonical_contract", True)),
    )
    sensa_excluded = {row["excluded_l2_id"] for row in pairs}
    dev_l4_ids = {blocks.l3_to_l4[l3_id] for l3_id in dev_l3_ids}
    sensb = sensitivity_b_membership(exact_l2_to_l3, primary_excluded, blocks.l3_to_l4, dev_l4_ids)
    exact = frozenset(exact_l2_to_l3)
    primary = frozenset(exact - primary_excluded)
    sensa = frozenset(primary - sensa_excluded)
    sets = {"exact": exact, "primary": primary, "sensA": sensa, "sensB": frozenset(sensb["sensB"])}
    reconciliation = {
        "exact_N": len(exact),
        "primary_N": len(primary),
        "sensA_N": len(sensa),
        "sensB_N": len(sensb["sensB"]),
        "sensitivityA_pairs": len(pairs),
        "excluded_from_exact": sensb["excluded_from_exact"],
        "already_excluded_by_primary": sensb["already_excluded_by_primary"],
        "additional_vs_primary": sensb["additional_vs_primary"],
        "predictions": {
            "exact_N": 3072,
            "primary_N": 3036,
            "sensA_N": 3024,
            "sensB_N": 2772,
            "excluded_from_exact": 300,
            "already_excluded_by_primary": 36,
            "additional_vs_primary": 264,
        },
    }
    normalized_scenarios = []
    for record in scenario_records.values():
        normalized = dict(record)
        normalized["resolved_ego_indices"] = tuple(sorted(set(record["resolved_ego_indices"])))
        normalized["l4_id"] = blocks.l3_to_l4[record["l3_id"]]
        normalized_scenarios.append(normalized)
    normalized_scenarios.sort(key=lambda x: x["l2_id"])
    occurrences.sort(key=lambda x: (x["grid_id"], x["start_ordinal"], x["opponent_raceline"], x["speedscale_hex"]))
    return S0Outputs(
        asset_namespace=namespace,
        occurrences=tuple(occurrences),
        scenarios=tuple(normalized_scenarios),
        block_manifest=blocks,
        dev_nodes=tuple(sorted(dev_nodes, key=lambda x: (x["l3_id"], x["x"], x["y"]))),
        dev_l3_ids=frozenset(dev_l3_ids),
        sets=sets,
        sensitivity_a_pairs=pairs,
        sensitivity_b=sensb,
        reconciliation=reconciliation,
    )
