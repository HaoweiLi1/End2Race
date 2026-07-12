"""D0.1 geometry/scan orchestration with deterministic atomic outputs."""

from __future__ import annotations

import csv
import json
import os
import shutil
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from d0 import ANALYSIS_VERSION, CLASSIFIER_VERSION
from d0.gates import (
    GateResult,
    release_verdict,
    run_g1_duplicate_determinism,
    run_g2_inventory,
    run_g3_integrity,
    run_g4_near_duplicate,
    run_g5_collision_floors,
    run_g6_record_physics,
    run_g7_unknown_censoring,
    run_g8_reconciliation,
)
from d0.identity import (
    REGISTRY_FIELDS,
    REGISTRY_SCHEMA,
    S0Outputs,
    append_opened_registry,
    canonical_json,
    domain_id,
    file_sha256,
    geometry_manifest,
    make_l1_payload,
    registry_row_id,
    validate_registry_row,
)
from d0.outcomes import EQUALITY_FIELDS, centerline_length, classify_outcome
from d0.stats import (
    CANDIDATE_ORDER,
    ESTIMAND_ORDER,
    POOL_ORDER,
    paired_block_bootstrap,
    run_all_stats,
)


TSV_LINETERMINATOR = "\n"
RECORDED_ARRAY_KEYS = (
    "time",
    "ego_lidar",
    "opp_lidar",
    "ego_desired_steer",
    "ego_desired_speed",
    "ego_actual_speed",
    "ego_pose",
    "ego_progress",
    "opp_desired_steer",
    "opp_desired_speed",
    "opp_actual_speed",
    "opp_pose",
    "opp_progress",
)
TERMINAL_KEYS = (
    "final_time",
    "final_ego_pose",
    "final_opp_pose",
    "final_ego_progress",
    "final_opp_progress",
    "ego_collision",
    "opp_collision",
    "collision",
    "state_label",
)
STATES = ("collision", "confirmed_pass", "terminal_overtake_only", "safe_follow")


class GateFailure(RuntimeError):
    pass


class InputFailure(RuntimeError):
    pass


class InjectedFailure(RuntimeError):
    pass


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value) -> None:
    text = json.dumps(
        _jsonable(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    path.write_text(text, encoding="utf-8")


def _write_tsv(path: Path, rows: Iterable[Mapping], fields: Iterable[str]) -> None:
    fields = tuple(fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator=TSV_LINETERMINATOR,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _manifest_hashes(directory: Path, exclude=()) -> dict[str, str]:
    excluded = set(exclude) | {"output_manifest.sha256", "COMPLETE", "FAILED"}
    return {
        item.name: file_sha256(item)
        for item in sorted(directory.iterdir(), key=lambda x: x.name)
        if item.is_file() and item.name not in excluded
    }


def _write_output_manifest(directory: Path) -> dict[str, str]:
    hashes = _manifest_hashes(directory)
    lines = [f"{digest}  {name}" for name, digest in sorted(hashes.items())]
    (directory / "output_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashes


def _registry_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(REGISTRY_FIELDS)


def _prepare_destination(output_dir: Path) -> tuple[int, Path | None]:
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            return 3, None
        output_dir.rmdir()
    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists():
        return 3, None
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    return 0, partial


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _promote_complete(partial: Path, output_dir: Path) -> None:
    for item in sorted(partial.iterdir(), key=lambda value: value.name):
        if item.is_file():
            with item.open("rb") as handle:
                os.fsync(handle.fileno())
    _fsync_directory(partial)
    os.replace(partial, output_dir)
    _fsync_directory(output_dir.parent)
    complete = output_dir / "COMPLETE"
    with complete.open("w", encoding="utf-8") as handle:
        handle.write("COMPLETE\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(output_dir)


def _asset_s0_rows(s0: S0Outputs):
    scenario_by_id = {row["l2_id"]: row for row in s0.scenarios}
    exact_overlap = []
    for l2_id in sorted(s0.sets["exact"] - s0.sets["primary"]):
        row = scenario_by_id[l2_id]
        exact_overlap.append(
            {
                "l2_id": l2_id,
                "l3_id": row["l3_id"],
                "l4_id": row["l4_id"],
                "map_name": row["l2_payload"]["map_name"],
                "ego_raceline": row["l2_payload"]["ego_raceline"],
                "rule": "exact_l3_match",
            }
        )
    dev_cells = {(round(float(node["x"]), 2), round(float(node["y"]), 2)) for node in s0.dev_nodes}
    cell_rows = []
    for row in s0.scenarios:
        pose = row["l3_payload"]["ego_start_pose_hex"]
        cell = (round(float.fromhex(pose[0]), 2), round(float.fromhex(pose[1]), 2))
        if row["l2_payload"]["map_name"] == "Austin" and cell in dev_cells:
            cell_rows.append(
                {
                    "l2_id": row["l2_id"],
                    "l3_id": row["l3_id"],
                    "cell_x": f"{cell[0]:.2f}",
                    "cell_y": f"{cell[1]:.2f}",
                    "exact_dev_overlap": row["l3_id"] in s0.dev_l3_ids,
                    "diagnostic_only": True,
                }
            )
    block_rows = []
    for component in s0.block_manifest.components:
        for node in component["nodes"]:
            block_rows.append(
                {
                    "l4_id": component["l4_id"],
                    "map_name": component["map_name"],
                    "ego_raceline": component["ego_raceline"],
                    "l3_id": node["l3_id"],
                    "x_hex": float(node["x"]).hex(),
                    "y_hex": float(node["y"]).hex(),
                    "is_dev": node["is_dev"],
                    "component_contains_dev": component["contains_dev"],
                    "member_count": len(component["member_l3_ids"]),
                }
            )
    sensb_rows = []
    for l2_id in sorted(s0.sensitivity_b["excluded_from_exact_ids"]):
        row = scenario_by_id[l2_id]
        sensb_rows.append(
            {
                "l2_id": l2_id,
                "l3_id": row["l3_id"],
                "l4_id": row["l4_id"],
                "already_excluded_by_primary": l2_id in s0.sensitivity_b["already_excluded_by_primary_ids"],
                "additional_vs_primary": l2_id in s0.sensitivity_b["additional_vs_primary_ids"],
            }
        )
    return exact_overlap, cell_rows, block_rows, sensb_rows


def _write_s0(directory: Path, s0: S0Outputs, runconfig: Mapping) -> str:
    _write_json(directory / "runconfig.json", runconfig)
    exact_overlap, cell_rows, block_rows, sensb_rows = _asset_s0_rows(s0)
    manifest = {
        "schema": "d0.1-s0-manifest-1",
        "analysis_version": ANALYSIS_VERSION,
        "source_run_id": runconfig["source_run_id"],
        "runconfig_sha256": file_sha256(directory / "runconfig.json"),
        "asset_namespace_sha256": s0.asset_namespace.sha256,
        "asset_entries": [
            {"relpath": relpath, "sha256": digest}
            for relpath, digest in s0.asset_namespace.entries
        ],
        "counts": {name: len(ids) for name, ids in s0.sets.items()},
        "accounting": {
            "sensitivityA_pairs": len(s0.sensitivity_a_pairs),
            "excluded_from_exact": s0.sensitivity_b["excluded_from_exact"],
            "already_excluded_by_primary": s0.sensitivity_b["already_excluded_by_primary"],
            "additional_vs_primary": s0.sensitivity_b["additional_vs_primary"],
        },
        "estimand_ids": {name: sorted(ids) for name, ids in s0.sets.items()},
        "reconciliation": s0.reconciliation,
    }
    _write_json(directory / "s0_manifest.json", manifest)
    _write_tsv(
        directory / "dev_overlap_exact.tsv",
        exact_overlap,
        ("l2_id", "l3_id", "l4_id", "map_name", "ego_raceline", "rule"),
    )
    _write_tsv(
        directory / "dev_overlap_cell_diag.tsv",
        sorted(cell_rows, key=lambda x: x["l2_id"]),
        ("l2_id", "l3_id", "cell_x", "cell_y", "exact_dev_overlap", "diagnostic_only"),
    )
    _write_tsv(
        directory / "blocks.tsv",
        sorted(block_rows, key=lambda x: (x["l4_id"], x["l3_id"])),
        (
            "l4_id",
            "map_name",
            "ego_raceline",
            "l3_id",
            "x_hex",
            "y_hex",
            "is_dev",
            "component_contains_dev",
            "member_count",
        ),
    )
    sensa_fields = (
        "pair_id",
        "map_name",
        "ego_raceline",
        "opponent_raceline",
        "speedscale_hex",
        "interval_idx",
        "retained_l2_id",
        "retained_min_resolved_ego_idx",
        "excluded_l2_id",
        "excluded_min_resolved_ego_idx",
        "rule_version",
    )
    _write_tsv(directory / "sensitivityA_pairs.tsv", s0.sensitivity_a_pairs, sensa_fields)
    _write_tsv(
        directory / "sensitivityB_excluded.tsv",
        sensb_rows,
        ("l2_id", "l3_id", "l4_id", "already_excluded_by_primary", "additional_vs_primary"),
    )
    return file_sha256(directory / "s0_manifest.json")


def _selected_models_and_grids(runconfig: Mapping, mode: str):
    models = list(runconfig["models"])
    grids = [tuple(item) for item in runconfig["grids"]]
    if mode == "smoke":
        requested_models = runconfig.get("_smoke_models", ["bc", "cand160"])
        models = [model for model in requested_models if model in runconfig["models"]]
        grids = [tuple(item) for item in runconfig.get("_smoke_grids", grids[:1])]
    return models, grids


def expected_inventory(runconfig: Mapping, mode: str, s0: S0Outputs) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    models, grids = _selected_models_and_grids(runconfig, mode)
    selected_grid_ids = {f"{map_name}_off{offset}" for map_name, offset in grids}
    occurrences_by_grid = defaultdict(list)
    for occurrence in s0.occurrences:
        if occurrence["grid_id"] in selected_grid_ids:
            occurrences_by_grid[occurrence["grid_id"]].append(occurrence)
    expected = {}
    for model in models:
        for map_name, offset in grids:
            grid_id = f"{map_name}_off{offset}"
            for occurrence in occurrences_by_grid[grid_id]:
                inventory_key = f"{model}|{grid_id}|{occurrence['episode_key']}"
                if inventory_key in expected:
                    raise InputFailure(f"duplicate expected inventory key {inventory_key}")
                expected[inventory_key] = {"model_id": model, **occurrence}
    configured = int(runconfig["expected_occurrences"][mode])
    if len(expected) != configured:
        raise InputFailure(
            f"expected inventory count mismatch: generated={len(expected)} configured={configured}"
        )
    return expected, occurrences_by_grid


def _safe_repo_path(repo_root: Path, relative: str, required_root: Path | None = None) -> Path:
    path = Path(relative)
    if not path.is_absolute():
        path = repo_root / path
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise InputFailure(f"missing source {path}: {exc}") from exc
    if required_root is not None:
        try:
            resolved.relative_to(required_root.resolve(strict=True))
        except ValueError as exc:
            raise InputFailure(f"source escapes configured root: {resolved}") from exc
    return resolved


def _load_sources(
    runconfig: Mapping,
    mode: str,
    s0: S0Outputs,
    expected: Mapping[str, Mapping],
):
    repo_root = Path(runconfig["repository_root"]).resolve(strict=True)
    eval_root = (repo_root / runconfig["eval_root"]).resolve(strict=True)
    models, grids = _selected_models_and_grids(runconfig, mode)
    observed_inventory = []
    result_sources = {}
    provenance = []
    checkpoint_hashes = {}
    for model in models:
        model_config = runconfig["models"][model]
        checkpoint = _safe_repo_path(repo_root, model_config["path"], repo_root)
        actual = file_sha256(checkpoint)
        if actual != model_config["sha256"]:
            raise InputFailure(f"checkpoint SHA256 mismatch for {model}: {actual}")
        checkpoint_hashes[model] = actual
        for map_name, offset in grids:
            grid_id = f"{map_name}_off{offset}"
            tag = runconfig["tag_template"].format(
                run=runconfig["source_run_id"], model=model, map=map_name, offset=offset
            )
            result_rel = Path(
                runconfig["result_dir_template"].format(tag=tag, map=map_name)
            ) / "results.json"
            result_path = _safe_repo_path(repo_root, result_rel.as_posix(), eval_root)
            result_hash = file_sha256(result_path)
            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
                episodes = data["episodes"]
                final = data["final"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise InputFailure(f"invalid results JSON {result_path}: {exc}") from exc
            if not isinstance(episodes, dict):
                raise InputFailure(f"results episodes is not a mapping: {result_path}")
            validated = bool(final.get("validated"))
            for episode_key in episodes:
                observed_inventory.append(
                    {
                        "inventory_key": f"{model}|{grid_id}|{episode_key}",
                        "validated": validated,
                        "stale": False,
                    }
                )
            result_sources[(model, grid_id)] = {
                "tag": tag,
                "result_path": result_path,
                "result_relpath": result_path.relative_to(repo_root).as_posix(),
                "result_sha256": result_hash,
                "episodes": episodes,
                "final": final,
            }
            provenance.append(
                {
                    "source_run_id": runconfig["source_run_id"],
                    "model_id": model,
                    "model_relpath": model_config["path"],
                    "checkpoint_sha256": actual,
                    "map_name": map_name,
                    "grid_id": grid_id,
                    "offset": offset,
                    "tag": tag,
                    "result_json_relpath": result_path.relative_to(repo_root).as_posix(),
                    "result_json_sha256": result_hash,
                    "episode_count": len(episodes),
                    "validated": validated,
                }
            )
    g2 = run_g2_inventory(set(expected), observed_inventory)
    return repo_root, eval_root, result_sources, checkpoint_hashes, provenance, g2


def _recorded_integrity(npz, outcome_record, json_episode, expected_row) -> dict:
    has_terminal = all(key in npz for key in TERMINAL_KEYS)
    lengths = []
    finite = True
    for key in RECORDED_ARRAY_KEYS:
        if key not in npz:
            finite = False
            continue
        value = np.asarray(npz[key])
        if value.ndim == 0:
            finite = False
            continue
        lengths.append(value.shape[0])
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            finite = False
    equal_lengths = bool(lengths and len(set(lengths)) == 1)
    raw_label_ok = (
        json_episode.get("outcome") in {"following", "overtaking", "collision"}
        and json_episode.get("outcome") == json_episode.get("state_label")
    )
    npz_label = str(np.asarray(npz["state_label"]).reshape(())) if "state_label" in npz else ""
    json_npz_match = npz_label == json_episode.get("outcome")
    collision_identity = bool(np.asarray(npz["collision"]).reshape(())) == (
        bool(np.asarray(npz["ego_collision"]).reshape(()))
        or bool(np.asarray(npz["opp_collision"]).reshape(()))
    )
    exact_resolution = (
        json_episode.get("map_name") == expected_row["map_name"]
        and json_episode.get("ego_raceline") == "raceline1"
        and json_episode.get("opp_raceline") == expected_row["opponent_raceline"]
        and int(json_episode.get("ego_idx", -1)) == int(expected_row["raw_ego_idx"])
        and int(json_episode.get("opp_idx", -1)) == int(expected_row["resolved_opp_idx"])
        and int(json_episode.get("interval_idx", -1)) == int(expected_row.get("interval_idx", json_episode.get("interval_idx", -1)))
        and float(json_episode.get("opp_speedscale", float("nan"))).hex() == expected_row["speedscale_hex"]
    )
    gap = float.fromhex(outcome_record.terminal_gap_hex) if outcome_record.terminal_gap_hex != "unknown" else float("nan")
    terminal_gap_ok = bool(np.isfinite(gap) and gap > 0 and abs(gap - 0.01) <= 0.005 + 1e-12)
    return {
        "has_terminal_fields": has_terminal,
        "raw_label_ok": raw_label_ok,
        "json_npz_label_match": json_npz_match,
        "collision_identity_ok": collision_identity,
        "equal_lengths": equal_lengths,
        "finite_values": finite,
        "exact_resolution": exact_resolution,
        "frame_spacing_status": outcome_record.frame_spacing_status,
        "terminal_gap_ok": terminal_gap_ok,
    }


def _skill(payload: Mapping) -> str:
    opponent = payload["opponent_raceline"]
    speed = float.fromhex(payload["opponent_speedscale_hex"])
    if opponent == "raceline1" and speed in {0.5, 0.6}:
        return "skill_F"
    if opponent in {"raceline0", "raceline2"} and speed in {0.7, 0.8}:
        return "skill_S"
    return "other"


def _scan_occurrences(
    runconfig: Mapping,
    s0: S0Outputs,
    expected: Mapping[str, Mapping],
    repo_root: Path,
    eval_root: Path,
    result_sources: Mapping,
    checkpoint_hashes: Mapping[str, str],
):
    scenario_by_id = {row["l2_id"]: row for row in s0.scenarios}
    lengths = {
        map_name: centerline_length(repo_root / runconfig["assets_root"], map_name)
        for map_name, _ in runconfig["grids"]
    }
    records = []
    g3_sources = []
    g6_records = []
    for inventory_key in sorted(expected):
        expected_row = expected[inventory_key]
        model = expected_row["model_id"]
        grid_id = expected_row["grid_id"]
        source = result_sources[(model, grid_id)]
        key = expected_row["episode_key"]
        json_episode = source["episodes"][key]
        npz_path = _safe_repo_path(repo_root, str(json_episode["npz_path"]), eval_root)
        npz_hash = file_sha256(npz_path)
        scenario = scenario_by_id[expected_row["l2_id"]]
        l1_payload = make_l1_payload(
            source_run_id=runconfig["source_run_id"],
            model_id=model,
            model_relpath=runconfig["models"][model]["path"],
            checkpoint_sha256=checkpoint_hashes[model],
            map_name=expected_row["map_name"],
            grid_id=grid_id,
            offset=int(expected_row["offset"]),
            tag=source["tag"],
            result_json_relpath=source["result_relpath"],
            result_json_sha256=source["result_sha256"],
            episode_key=key,
            npz_relpath=npz_path.relative_to(repo_root).as_posix(),
            npz_sha256=npz_hash,
            l2_id=expected_row["l2_id"],
        )
        l1_id = domain_id("L1", l1_payload)
        with np.load(npz_path, allow_pickle=False) as npz:
            outcome = classify_outcome(
                npz,
                json_episode,
                lengths[expected_row["map_name"]],
                attempt_threshold=float(runconfig["classifier"]["attempt_m"]),
                lead_threshold=float(runconfig["classifier"]["confirmed_lead_m"]),
                hold_seconds=float(runconfig["classifier"]["confirmed_hold_s"]),
            )
            integrity = _recorded_integrity(npz, outcome, json_episode, expected_row)
        record = {
            "inventory_key": inventory_key,
            "l1_id": l1_id,
            "l2_id": expected_row["l2_id"],
            "l3_id": scenario["l3_id"],
            "l4_id": scenario["l4_id"],
            "model_id": model,
            "map_name": expected_row["map_name"],
            "grid_id": grid_id,
            "offset": expected_row["offset"],
            "episode_key": key,
            "raw_ego_idx": expected_row["raw_ego_idx"],
            "resolved_ego_idx": expected_row["resolved_ego_idx"],
            "resolved_opp_idx": expected_row["resolved_opp_idx"],
            "ego_raceline": scenario["l2_payload"]["ego_raceline"],
            "opponent_raceline": scenario["l2_payload"]["opponent_raceline"],
            "speedscale_hex": scenario["l2_payload"]["opponent_speedscale_hex"],
            "interval_idx": scenario["l2_payload"]["interval_idx"],
            "skill": _skill(scenario["l2_payload"]),
            "npz_relpath": npz_path.relative_to(repo_root).as_posix(),
            "npz_sha256": npz_hash,
            "result_json_relpath": source["result_relpath"],
            "result_json_sha256": source["result_sha256"],
            **{field: getattr(outcome, field) for field in EQUALITY_FIELDS},
        }
        records.append(record)
        g3_sources.append(
            {
                "l1_id": l1_id,
                "path": str(npz_path),
                "root": str(eval_root),
                "expected_sha256": npz_hash,
            }
        )
        g6_records.append({"inventory_key": inventory_key, **integrity})
    return records, g3_sources, g6_records


def _collapse(records: list[dict], g1: GateResult):
    if not g1.passed:
        return []
    groups = defaultdict(list)
    for record in records:
        groups[(record["model_id"], record["l2_id"])].append(record)
    canonical = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda x: x["l1_id"])
        representative = dict(group[0])
        representative["representative_l1_id"] = representative["l1_id"]
        representative["occurrence_count"] = len(group)
        representative["all_l1_ids"] = ";".join(row["l1_id"] for row in group)
        canonical.append(representative)
    return canonical


def _near_duplicate_pairs(canonical: list[dict], s0: S0Outputs):
    scenario_by_id = {row["l2_id"]: row for row in s0.scenarios}
    by_model = defaultdict(list)
    for record in canonical:
        by_model[record["model_id"]].append(record)
    pairs = []
    for model, rows in sorted(by_model.items()):
        for i, left in enumerate(rows):
            lp = scenario_by_id[left["l2_id"]]["l2_payload"]
            lpose = scenario_by_id[left["l2_id"]]["l3_payload"]["ego_start_pose_hex"]
            for right in rows[i + 1 :]:
                if left["l4_id"] != right["l4_id"] or left["l3_id"] == right["l3_id"]:
                    continue
                rp = scenario_by_id[right["l2_id"]]["l2_payload"]
                rpose = scenario_by_id[right["l2_id"]]["l3_payload"]["ego_start_pose_hex"]
                distance = float(
                    np.hypot(
                        float.fromhex(lpose[0]) - float.fromhex(rpose[0]),
                        float.fromhex(lpose[1]) - float.fromhex(rpose[1]),
                    )
                )
                if distance > 1.0:
                    continue
                def side(record, payload):
                    return {
                        "map_name": record["map_name"],
                        "ego_raceline": payload["ego_raceline"],
                        "opponent_raceline": payload["opponent_raceline"],
                        "speedscale": payload["opponent_speedscale_hex"],
                        "interval_idx": payload["interval_idx"],
                        "l3_id": record["l3_id"],
                        "l4_id": record["l4_id"],
                        "outcome": record["corrected_outcome3"],
                    }
                pairs.append({"left": side(left, lp), "right": side(right, rp)})
    return pairs


def _slices(ids: set[str], scenario_by_id: Mapping[str, Mapping]):
    slices = {"all": set(ids)}
    for l2_id in ids:
        payload = scenario_by_id[l2_id]["l2_payload"]
        map_name = payload["map_name"]
        skill = _skill(payload)
        opponent = payload["opponent_raceline"]
        speed = str(float.fromhex(payload["opponent_speedscale_hex"]))
        slices.setdefault("pool:austin" if map_name == "Austin" else "pool:cross", set()).add(l2_id)
        slices.setdefault(f"map:{map_name}", set()).add(l2_id)
        slices.setdefault(f"skill:{skill}", set()).add(l2_id)
        slices.setdefault(f"opponent:{opponent}", set()).add(l2_id)
        slices.setdefault(f"speed:{speed}", set()).add(l2_id)
    slices.setdefault("pool:austin", set())
    slices.setdefault("pool:cross", set())
    return slices


def _build_matrices(canonical: list[dict], s0: S0Outputs, estimands: Mapping[str, set[str]]):
    scenario_by_id = {row["l2_id"]: row for row in s0.scenarios}
    records = defaultdict(dict)
    for row in canonical:
        records[row["model_id"]][row["l2_id"]] = row
    candidates = sorted(model for model in records if model != "bc")
    matrices = {}
    rows_by_estimand = defaultdict(list)
    for estimand in ESTIMAND_ORDER:
        if estimand not in estimands:
            continue
        base_ids = set(estimands[estimand]) & set(records.get("bc", {}))
        for candidate in candidates:
            paired_ids = base_ids & set(records[candidate])
            for slice_id, slice_ids in sorted(_slices(paired_ids, scenario_by_id).items()):
                counts = Counter(
                    (records["bc"][l2_id]["four_state"], records[candidate][l2_id]["four_state"])
                    for l2_id in sorted(slice_ids)
                )
                matrix_id = f"{estimand}|{candidate}|{slice_id}"
                matrix_rows = []
                for bc_state in STATES:
                    for candidate_state in STATES:
                        row = {
                            "matrix_id": matrix_id,
                            "estimand": estimand,
                            "candidate": candidate,
                            "slice_id": slice_id,
                            "bc_state": bc_state,
                            "candidate_state": candidate_state,
                            "count": int(counts[(bc_state, candidate_state)]),
                            "expected_n": len(slice_ids),
                        }
                        matrix_rows.append(row)
                        rows_by_estimand[estimand].append(row)
                matrices[matrix_id] = {"rows": matrix_rows, "expected_n": len(slice_ids)}
    return matrices, rows_by_estimand


def _stats(canonical: list[dict], s0: S0Outputs, estimands: Mapping[str, set[str]], runconfig: Mapping):
    records = defaultdict(dict)
    for row in canonical:
        records[row["model_id"]][row["l2_id"]] = {
            "map_name": row["map_name"],
            "l4_id": row["l4_id"],
            "collision_any": bool(row["collision_any"]),
            "corrected_outcome3": row["corrected_outcome3"],
        }
    blocks = defaultdict(lambda: defaultdict(list))
    scenario_by_id = {row["l2_id"]: row for row in s0.scenarios}
    scanned_ids = set(records.get("bc", {}))
    for l2_id in sorted(scanned_ids):
        scenario = scenario_by_id[l2_id]
        blocks[scenario["l2_payload"]["map_name"]][scenario["l4_id"]].append(l2_id)
    blocks = {map_name: dict(values) for map_name, values in blocks.items()}
    B = int(runconfig["bootstrap"]["B"])
    seed = int(runconfig["bootstrap"]["seed"])
    if set(CANDIDATE_ORDER) <= set(records) and set(ESTIMAND_ORDER) <= set(estimands):
        return run_all_stats(estimands, records, blocks, B=B, seed=seed)

    candidates = sorted(model for model in records if model != "bc")
    order = [
        (estimand, candidate, pool)
        for estimand in ESTIMAND_ORDER
        if estimand in estimands
        for candidate in candidates
        for pool in POOL_ORDER
    ]
    children = np.random.default_rng(seed).spawn(len(order))
    results = {}
    child_order = []
    for index, (estimand, candidate, pool) in enumerate(order):
        ids = sorted(set(estimands[estimand]) & set(records["bc"]) & set(records[candidate]))
        if pool == "austin":
            ids = [item for item in ids if records["bc"][item]["map_name"] == "Austin"]
        elif pool == "cross":
            ids = [item for item in ids if records["bc"][item]["map_name"] != "Austin"]
        result = paired_block_bootstrap(
            ids, records["bc"], records[candidate], blocks, B=B, rng=children[index]
        )
        result.update({"child_index": index, "estimand": estimand, "candidate": candidate, "pool": pool})
        results[f"{estimand}|{candidate}|{pool}"] = result
        child_order.append({"child_index": index, "estimand": estimand, "candidate": candidate, "pool": pool})
    return {
        "schema": "d0.1-stats-all-1",
        "B": B,
        "seed": seed,
        "spawn_method": f"numpy.random.default_rng(seed).spawn({len(order)})",
        "child_order": child_order,
        "results": results,
    }


def _model_metrics(rows: Iterable[Mapping]) -> dict:
    rows = list(rows)
    return {
        "N": len(rows),
        "collision": sum(bool(row["collision_any"]) for row in rows),
        "ego_collision": sum(bool(row["ego_collision"]) for row in rows),
        "opp_collision": sum(bool(row["opp_collision"]) for row in rows),
        "opponent_only": sum(
            bool(row["opp_collision"]) and not bool(row["ego_collision"]) for row in rows
        ),
        "ego_only": sum(
            bool(row["ego_collision"]) and not bool(row["opp_collision"]) for row in rows
        ),
        "both": sum(
            bool(row["ego_collision"]) and bool(row["opp_collision"]) for row in rows
        ),
        "overtake": sum(row["corrected_outcome3"] == "overtake" for row in rows),
        "confirmed_pass": sum(row["four_state"] == "confirmed_pass" for row in rows),
        "interaction_attempt": sum(row["interaction_attempt"] is True for row in rows),
    }


def _phase_metrics(rows: Iterable[Mapping]) -> dict:
    counts = {"pre": 0, "alongside": 0, "post": 0}
    for row in rows:
        if not bool(row["collision_any"]):
            continue
        phase = row["collision_phase"]
        if phase not in counts:
            raise ValueError(f"released collision has invalid phase {phase!r}")
        counts[phase] += 1
    counts["total"] = sum(counts.values())
    return counts


def _path_value(value: Mapping, dotted_path: str):
    current = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def build_reconciliation(summary: Mapping, runconfig: Mapping, mode: str) -> dict:
    references = {}
    applicable_statuses = []
    for name, reference in sorted(runconfig.get("reconciliation_targets", {}).items()):
        modes = tuple(reference.get("modes", ("full",)))
        item = {
            "source": str(reference.get("source", "unspecified")),
            "checks": [],
        }
        if reference.get("source_sha256"):
            item["source_sha256"] = str(reference["source_sha256"])
        if mode not in modes:
            item["status"] = "not_applicable"
            item["explanation"] = f"reference applies to modes {list(modes)}, current mode is {mode}"
            references[name] = item
            continue
        statuses = []
        for path, target in sorted(reference.get("values", {}).items()):
            try:
                observed = _path_value(summary, path)
                status = "match" if observed == target else "mismatch"
                explanation = "exact equality" if status == "match" else "regenerated value differs from reference target"
            except KeyError:
                observed = None
                status = "missing"
                explanation = "required observed field is absent"
            statuses.append(status)
            item["checks"].append(
                {
                    "field": path,
                    "target": target,
                    "observed": observed,
                    "status": status,
                    "explanation": explanation,
                }
            )
        item["status"] = "match" if statuses and all(x == "match" for x in statuses) else "mismatch"
        if not statuses:
            item["status"] = "not_configured"
        applicable_statuses.append(item["status"])
        references[name] = item
    return {
        "schema": "d0.1-reconciliation-1",
        "mode": mode,
        "references": references,
        "all_applicable_targets_match": (
            all(status == "match" for status in applicable_statuses)
            if applicable_statuses
            else None
        ),
    }


def build_summary(
    canonical: Iterable[Mapping],
    estimands: Mapping[str, set[str]],
    geometry_reconciliation: Mapping,
    runconfig: Mapping,
    mode: str,
) -> dict:
    canonical = list(canonical)
    models = sorted({str(row["model_id"]) for row in canonical})
    by_model = defaultdict(dict)
    for row in canonical:
        by_model[str(row["model_id"])][str(row["l2_id"])] = row

    tables = {}
    bc_breakdown = {}
    strata = {}
    collision_phases = {}
    opponent_only_floor = {}
    for estimand, raw_ids in estimands.items():
        ids = set(raw_ids)
        tables[estimand] = {}
        for model in models:
            usable = [by_model[model][item] for item in sorted(ids & set(by_model[model]))]
            tables[estimand][model] = _model_metrics(usable)

        if "bc" in tables[estimand]:
            bc = tables[estimand]["bc"]
            bc_breakdown[estimand] = {
                "any_agent": bc["collision"],
                "ego_involved": bc["ego_collision"],
                "opponent_flag": bc["opp_collision"],
                "opponent_only": bc["opponent_only"],
                "ego_only": bc["ego_only"],
                "both": bc["both"],
            }

        strata[estimand] = {}
        for skill in ("skill_F", "skill_S", "other"):
            skill_ids = {
                str(row["l2_id"])
                for row in canonical
                if str(row["l2_id"]) in ids and row["skill"] == skill
            }
            entry = {"N": len(skill_ids)}
            for model in models:
                entry[model] = _model_metrics(
                    by_model[model][item]
                    for item in sorted(skill_ids & set(by_model[model]))
                )
            strata[estimand][skill] = entry

        collision_phases[estimand] = {}
        opponent_sets = {}
        for model in models:
            usable = [by_model[model][item] for item in sorted(ids & set(by_model[model]))]
            collision_phases[estimand][model] = {
                "all": _phase_metrics(usable),
                "opponent_raceline1": _phase_metrics(
                    row for row in usable if row["opponent_raceline"] == "raceline1"
                ),
            }
            opponent_sets[model] = {
                str(row["l2_id"])
                for row in usable
                if bool(row["opp_collision"]) and not bool(row["ego_collision"])
            }
        bc_ids = opponent_sets.get("bc", set())
        opponent_only_floor[estimand] = {
            "bc_count": len(bc_ids),
            "counts_by_model": {model: len(opponent_sets[model]) for model in models},
            "identical_across_all_models": bool(models)
            and all(opponent_sets[model] == bc_ids for model in models),
            "bc_l2_ids": sorted(bc_ids),
        }

    corrections = [
        row for row in canonical if row["archived_outcome3"] != row["corrected_outcome3"]
    ]
    collisions = [row for row in canonical if bool(row["collision_any"])]
    unique_ids = {str(row["l2_id"]) for row in canonical}
    summary = {
        "schema": "d0.1-summary-2",
        "analysis_version": ANALYSIS_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "mode": mode,
        "counts": {
            "canonical": len(unique_ids),
            "canonical_model_rows": len(canonical),
            "corrections": len(corrections),
            "collisions": len(collisions),
        },
        "estimands": tables,
        "bc_breakdown": bc_breakdown,
        "strata": strata,
        "collision_phases": collision_phases,
        "opponent_only_floor": opponent_only_floor,
        "correction_counts": {
            "all": len(corrections),
            "by_model": {
                model: sum(row["model_id"] == model for row in corrections) for model in models
            },
        },
        "geometry_reconciliation": dict(geometry_reconciliation),
    }
    summary["reconciliation"] = build_reconciliation(summary, runconfig, mode)
    return summary


def _summaries(
    canonical: list[dict],
    estimands: Mapping[str, set[str]],
    s0: S0Outputs,
    runconfig: Mapping,
    mode: str,
):
    summary = build_summary(canonical, estimands, s0.reconciliation, runconfig, mode)
    corrections = [
        row for row in canonical if row["archived_outcome3"] != row["corrected_outcome3"]
    ]
    collisions = [row for row in canonical if bool(row["collision_any"])]
    return summary, _jsonable(summary), corrections, collisions


def _write_scan_outputs(
    directory: Path,
    runconfig: Mapping,
    s0: S0Outputs,
    s0_hash: str,
    provenance: list[dict],
    occurrences: list[dict],
    canonical: list[dict],
    matrices: Mapping[str, Mapping],
    matrix_rows: Mapping[str, list[dict]],
    stats: Mapping,
    summary: Mapping,
    corrections: list[dict],
    collisions: list[dict],
):
    provenance_fields = (
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
        "episode_count",
        "validated",
    )
    _write_tsv(directory / "input_provenance.tsv", sorted(provenance, key=lambda x: (x["model_id"], x["grid_id"])), provenance_fields)
    occurrence_fields = (
        "inventory_key",
        "l1_id",
        "l2_id",
        "l3_id",
        "l4_id",
        "model_id",
        "map_name",
        "grid_id",
        "offset",
        "episode_key",
        "raw_ego_idx",
        "resolved_ego_idx",
        "resolved_opp_idx",
        "ego_raceline",
        "opponent_raceline",
        "speedscale_hex",
        "interval_idx",
        "skill",
        "npz_relpath",
        "npz_sha256",
        "result_json_relpath",
        "result_json_sha256",
        *EQUALITY_FIELDS,
    )
    _write_tsv(directory / "episode_occurrences.tsv", occurrences, occurrence_fields)
    canonical_fields = (
        "model_id",
        "l2_id",
        "l3_id",
        "l4_id",
        "map_name",
        "ego_raceline",
        "opponent_raceline",
        "speedscale_hex",
        "interval_idx",
        "skill",
        "representative_l1_id",
        "all_l1_ids",
        "occurrence_count",
        *EQUALITY_FIELDS,
    )
    _write_tsv(directory / "canonical_episodes.tsv", canonical, canonical_fields)
    trajectory_fields = (
        "model_id",
        "l2_id",
        "representative_l1_id",
        "alignment_status",
        "alignment_k",
        "rel_start_hex",
        "rel_terminal_hex",
        "ego_wrap_count",
        "opp_wrap_count",
        "physics_status",
        "terminal_gap_hex",
        "frame_spacing_status",
        "censored",
        "interaction_attempt",
        "confirmed_safe_pass",
        "attempted_follow_no_collision",
        "four_state",
    )
    _write_tsv(directory / "trajectory_metrics.tsv", canonical, trajectory_fields)
    correction_rows = [
        {
            "model_id": row["model_id"],
            "l2_id": row["l2_id"],
            "representative_l1_id": row["representative_l1_id"],
            "all_l1_ids": row["all_l1_ids"],
            "archived_outcome_raw": row["archived_outcome_raw"],
            "archived_outcome3": row["archived_outcome3"],
            "corrected_outcome3": row["corrected_outcome3"],
            "ego_collision": row["ego_collision"],
            "opp_collision": row["opp_collision"],
            "rel_start_hex": row["rel_start_hex"],
            "alignment_k": row["alignment_k"],
            "rel_terminal_hex": row["rel_terminal_hex"],
            "ego_wrap_count": row["ego_wrap_count"],
            "opp_wrap_count": row["opp_wrap_count"],
        }
        for row in corrections
    ]
    _write_tsv(
        directory / "outcome_corrections.tsv",
        correction_rows,
        (
            "model_id",
            "l2_id",
            "representative_l1_id",
            "all_l1_ids",
            "archived_outcome_raw",
            "archived_outcome3",
            "corrected_outcome3",
            "ego_collision",
            "opp_collision",
            "rel_start_hex",
            "alignment_k",
            "rel_terminal_hex",
            "ego_wrap_count",
            "opp_wrap_count",
        ),
    )
    collision_rows = [
        {
            "model_id": row["model_id"],
            "l2_id": row["l2_id"],
            "representative_l1_id": row["representative_l1_id"],
            "ego_collision": row["ego_collision"],
            "opp_collision": row["opp_collision"],
            "involvement": row["collision_involvement"],
            "cause": row["collision_cause"],
            "phase": row["collision_phase"],
            "final_dist_hex": row["collision_final_dist_hex"],
            "terminal_rel_hex": row["rel_terminal_hex"],
            "classifier_version": CLASSIFIER_VERSION,
        }
        for row in collisions
    ]
    _write_tsv(
        directory / "collision_events.tsv",
        collision_rows,
        (
            "model_id",
            "l2_id",
            "representative_l1_id",
            "ego_collision",
            "opp_collision",
            "involvement",
            "cause",
            "phase",
            "final_dist_hex",
            "terminal_rel_hex",
            "classifier_version",
        ),
    )
    alignment_rows = [
        {key: row[key] for key in ("model_id", "l2_id", "representative_l1_id", "alignment_status", "alignment_k", "rel_start_hex", "rel_terminal_hex")}
        for row in canonical
        if row["alignment_status"] != "ok"
    ]
    _write_tsv(
        directory / "alignment_failures.tsv",
        alignment_rows,
        ("model_id", "l2_id", "representative_l1_id", "alignment_status", "alignment_k", "rel_start_hex", "rel_terminal_hex"),
    )
    matrix_fields = ("matrix_id", "estimand", "candidate", "slice_id", "bc_state", "candidate_state", "count", "expected_n")
    for estimand in ESTIMAND_ORDER:
        rows = matrix_rows.get(estimand, [])
        _write_tsv(directory / f"transition_matrix_{estimand}.tsv", rows, matrix_fields)
        md = [f"# Transition matrices — {estimand}", ""]
        for matrix_id in sorted({row["matrix_id"] for row in rows}):
            subset = [row for row in rows if row["matrix_id"] == matrix_id]
            md.append(f"- `{matrix_id}`: N={subset[0]['expected_n'] if subset else 0}, cells={len(subset)}")
        (directory / f"transition_matrix_{estimand}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        stats_subset = {
            "schema": stats["schema"],
            "B": stats["B"],
            "seed": stats["seed"],
            "spawn_method": stats["spawn_method"],
            "child_order": [row for row in stats["child_order"] if row["estimand"] == estimand],
            "results": {key: value for key, value in stats["results"].items() if value["estimand"] == estimand},
        }
        _write_json(directory / f"stats_{estimand}.json", stats_subset)
    _write_json(directory / "d0_summary.json", summary)
    md = [
        "# D0.1 Canonical Audit Summary",
        "",
        f"- S0 manifest SHA256: `{s0_hash}`",
        f"- canonical scenarios scanned: {summary['counts']['canonical']}",
        f"- model/scenario rows: {summary['counts']['canonical_model_rows']}",
        f"- outcome corrections: {summary['counts']['corrections']}",
        f"- collision rows: {summary['counts']['collisions']}",
        "",
        "## Geometry reconciliation",
        "",
        "```json",
        json.dumps(_jsonable(summary["geometry_reconciliation"]), sort_keys=True, indent=2),
        "```",
    ]
    for title, key in (
        ("Estimand model metrics", "estimands"),
        ("BC collision breakdown", "bc_breakdown"),
        ("Skill strata", "strata"),
        ("Collision phases", "collision_phases"),
        ("Opponent-only empirical floor", "opponent_only_floor"),
        ("Outcome corrections", "correction_counts"),
        ("D0 v1 and reviewer-target reconciliation", "reconciliation"),
    ):
        md.extend(
            [
                "",
                f"## {title}",
                "",
                "```json",
                json.dumps(_jsonable(summary[key]), sort_keys=True, indent=2),
                "```",
            ]
        )
    (directory / "d0_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return correction_rows, collision_rows


def _registry_row(
    runconfig: Mapping,
    s0_hash: str,
    *,
    l2_id: str,
    l3_id: str,
    l4_id: str,
    map_name: str,
):
    row = {
        "registry_schema": REGISTRY_SCHEMA,
        "opened_at_utc": runconfig["opened_at_utc"],
        "stage": "D0.1",
        "use_class": "historical_analysis",
        "split_id": "historical_p1",
        "l2_id": l2_id,
        "l3_id": l3_id,
        "l4_id": l4_id,
        "map_name": map_name,
        "source_manifest_sha256": s0_hash,
        "source_run_id": runconfig["source_run_id"],
        "decision_effect": "historical_only",
        "final_pool": "false",
        "evidence_relpath": f"{runconfig['eval_root']}/{runconfig['source_run_id']}",
    }
    row["row_id"] = registry_row_id(row)
    return row


def _registry_rows(runconfig: Mapping, s0: S0Outputs, s0_hash: str):
    rows = [
        _registry_row(
            runconfig,
            s0_hash,
            l2_id=scenario["l2_id"],
            l3_id=scenario["l3_id"],
            l4_id=scenario["l4_id"],
            map_name=scenario["l2_payload"]["map_name"],
        )
        for scenario in s0.scenarios
    ]
    return sorted(rows, key=lambda x: x["row_id"])


def _registry_rows_from_canonical(
    runconfig: Mapping, canonical: Iterable[Mapping], s0_hash: str
):
    scenarios = {}
    for row in canonical:
        l2_id = str(row["l2_id"])
        identity = (str(row["l3_id"]), str(row["l4_id"]), str(row["map_name"]))
        prior = scenarios.get(l2_id)
        if prior is not None and prior != identity:
            raise ValueError(f"canonical registry identity drift for {l2_id}")
        scenarios[l2_id] = identity
    rows = [
        _registry_row(
            runconfig,
            s0_hash,
            l2_id=l2_id,
            l3_id=identity[0],
            l4_id=identity[1],
            map_name=identity[2],
        )
        for l2_id, identity in sorted(scenarios.items())
    ]
    return sorted(rows, key=lambda x: x["row_id"])


def _prepare_registry_snapshot(
    partial: Path,
    runconfig: Mapping,
    s0: S0Outputs,
    s0_hash: str,
    mode: str,
):
    snapshot = partial / "opened_registry.snapshot.tsv"
    if mode != "full":
        _registry_header(snapshot)
        return snapshot, [], [], None

    registry_path = Path(runconfig["repository_root"]) / runconfig["opened_registry"]
    if registry_path.exists():
        shutil.copyfile(registry_path, snapshot)
    required_rows = _registry_rows(runconfig, s0, s0_hash)
    append_opened_registry(snapshot, required_rows)
    observed_rows = [validate_registry_row(row) for row in _read_tsv(snapshot)]
    return snapshot, observed_rows, required_rows, registry_path


def _validation_json(gates: list[GateResult], release: GateResult):
    return {
        "schema": "d0.1-validation-1",
        "gates": [asdict(gate) for gate in gates],
        "release": asdict(release),
    }


def run_scan(
    mode: str,
    output_dir: str | Path,
    runconfig: Mapping,
    workers: int = 8,
    *,
    _fault_after: str | None = None,
) -> int:
    if mode not in {"geometry", "smoke", "full"}:
        return 4
    output_dir = Path(output_dir)
    code, partial = _prepare_destination(output_dir)
    if code:
        return code
    assert partial is not None
    try:
        s0 = geometry_manifest(runconfig)
        s0_hash = _write_s0(partial, s0, runconfig)
        if mode == "geometry":
            _write_output_manifest(partial)
            _promote_complete(partial, output_dir)
            return 0

        expected, _ = expected_inventory(runconfig, mode, s0)
        repo_root, eval_root, sources, checkpoint_hashes, provenance, g2 = _load_sources(
            runconfig, mode, s0, expected
        )
        if _fault_after == "inventory":
            raise InjectedFailure("synthetic fault after inventory")
        if not g2.passed:
            release = release_verdict([g2])
            _write_json(partial / "d0_validation.json", _validation_json([g2], release))
            raise GateFailure("G2 inventory failed")

        occurrences, g3_sources, g6_records = _scan_occurrences(
            runconfig, s0, expected, repo_root, eval_root, sources, checkpoint_hashes
        )
        g1 = run_g1_duplicate_determinism(occurrences)
        g3 = run_g3_integrity(g3_sources)
        g6 = run_g6_record_physics(g6_records)
        canonical = _collapse(occurrences, g1)
        g4 = run_g4_near_duplicate(_near_duplicate_pairs(canonical, s0))
        scanned_ids = {row["l2_id"] for row in canonical}
        estimands = {
            name: set(ids) & scanned_ids
            for name, ids in s0.sets.items()
            if name in ESTIMAND_ORDER
        }
        if bool(runconfig.get("_strict_canonical_contract", True)) and mode == "full":
            primary_records = [row for row in canonical if row["l2_id"] in estimands["primary"]]
            g5 = run_g5_collision_floors(primary_records)
        else:
            g5 = GateResult(
                name="G5",
                blocking=True,
                passed=True,
                counts={"waived_for_synthetic_or_nonfull": True},
                violations=(),
            )
        g7 = run_g7_unknown_censoring(occurrences)
        matrices, matrix_rows = _build_matrices(canonical, s0, estimands)
        stats = _stats(canonical, s0, estimands, runconfig)
        summary, expected_summary, corrections, collisions = _summaries(
            canonical, estimands, s0, runconfig, mode
        )
        correction_rows, collision_rows = _write_scan_outputs(
            partial,
            runconfig,
            s0,
            s0_hash,
            provenance,
            occurrences,
            canonical,
            matrices,
            matrix_rows,
            stats,
            summary,
            corrections,
            collisions,
        )
        registry_snapshot, registry_rows, required_registry_rows, registry_path = (
            _prepare_registry_snapshot(partial, runconfig, s0, s0_hash, mode)
        )
        analytical_hashes = _manifest_hashes(partial)
        required_for_g8 = set(analytical_hashes)
        g8 = run_g8_reconciliation(
            canonical_records=canonical,
            correction_rows=correction_rows,
            collision_rows=collision_rows,
            matrices=matrices,
            summary=summary,
            expected_summary=expected_summary,
            expected_estimands=estimands,
            emitted_estimands={name: sorted(ids) for name, ids in estimands.items()},
            required_manifest_files=required_for_g8,
            manifest_hashes=analytical_hashes,
            registry_rows=registry_rows,
            required_registry_rows=required_registry_rows,
        )
        gates = [g1, g2, g3, g4, g5, g6, g7, g8]
        release = release_verdict(gates)
        if not release.passed:
            _write_json(partial / "d0_validation.json", _validation_json(gates, release))
            raise GateFailure("one or more D0.1 release gates failed")

        if mode == "full":
            assert registry_path is not None
            append_opened_registry(registry_path, required_registry_rows)
            live_rows = [validate_registry_row(row) for row in _read_tsv(registry_path)]
            live_by_id = {row["row_id"]: row for row in live_rows}
            snapshot_by_id = {row["row_id"]: row for row in registry_rows}
            if live_by_id != snapshot_by_id:
                raise GateFailure("canonical registry changed after prospective G8 validation")
            shutil.copyfile(registry_path, registry_snapshot)
        _write_json(partial / "d0_validation.json", _validation_json(gates, release))
        _write_output_manifest(partial)
        _promote_complete(partial, output_dir)
        return 0
    except GateFailure as exc:
        if partial.exists():
            (partial / "FAILED").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return 2
    except InjectedFailure as exc:
        if partial.exists():
            (partial / "FAILED").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return 2
    except (InputFailure, FileNotFoundError, PermissionError) as exc:
        if partial.exists():
            (partial / "FAILED").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return 4
    except Exception as exc:  # retained as auditable Stage-0/analysis failure
        if partial.exists():
            (partial / "FAILED").write_text(
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", encoding="utf-8"
            )
        return 2


def _parse_bool(value: str) -> bool:
    if value in {"True", "true", "1"}:
        return True
    if value in {"False", "false", "0"}:
        return False
    raise ValueError(f"invalid Boolean string {value!r}")


def validate_emitted_output(output_dir: str | Path) -> GateResult:
    directory = Path(output_dir)
    violations = []
    manifest_path = directory / "output_manifest.sha256"
    manifest_hashes = {}
    try:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            manifest_hashes[name] = digest
        actual_files = {
            item.name
            for item in directory.iterdir()
            if item.is_file() and item.name not in {"output_manifest.sha256", "COMPLETE", "FAILED"}
        }
        if set(manifest_hashes) != actual_files:
            violations.append(
                f"manifest coverage mismatch missing={sorted(actual_files-set(manifest_hashes))} "
                f"extra={sorted(set(manifest_hashes)-actual_files)}"
            )
        for name, digest in sorted(manifest_hashes.items()):
            path = directory / name
            if not path.is_file() or file_sha256(path) != digest:
                violations.append(f"manifest hash mismatch for {name}")

        runconfig = json.loads((directory / "runconfig.json").read_text(encoding="utf-8"))
        canonical_raw = _read_tsv(directory / "canonical_episodes.tsv")
        canonical = []
        for row in canonical_raw:
            canonical.append(
                {
                    "model_id": row["model_id"],
                    "l2_id": row["l2_id"],
                    "l3_id": row["l3_id"],
                    "l4_id": row["l4_id"],
                    "map_name": row["map_name"],
                    "opponent_raceline": row["opponent_raceline"],
                    "skill": row["skill"],
                    "ego_collision": _parse_bool(row["ego_collision"]),
                    "opp_collision": _parse_bool(row["opp_collision"]),
                    "archived_outcome3": row["archived_outcome3"],
                    "corrected_outcome3": row["corrected_outcome3"],
                    "collision_any": _parse_bool(row["collision_any"]),
                    "four_state": row["four_state"],
                    "interaction_attempt": _parse_bool(row["interaction_attempt"]),
                    "confirmed_safe_pass": _parse_bool(row["confirmed_safe_pass"]),
                    "collision_phase": row["collision_phase"],
                }
            )
        corrections = _read_tsv(directory / "outcome_corrections.tsv")
        collisions = _read_tsv(directory / "collision_events.tsv")
        matrices = {}
        for estimand in ESTIMAND_ORDER:
            path = directory / f"transition_matrix_{estimand}.tsv"
            for row in _read_tsv(path):
                item = matrices.setdefault(
                    row["matrix_id"], {"rows": [], "expected_n": int(row["expected_n"])}
                )
                item["rows"].append(
                    {
                        "bc_state": row["bc_state"],
                        "candidate_state": row["candidate_state"],
                        "count": int(row["count"]),
                    }
                )
        summary = json.loads((directory / "d0_summary.json").read_text(encoding="utf-8"))
        s0_manifest = json.loads((directory / "s0_manifest.json").read_text(encoding="utf-8"))
        scanned_ids = {row["l2_id"] for row in canonical}
        expected_estimands = {
            name: set(s0_manifest["estimand_ids"][name]) & scanned_ids
            for name in ESTIMAND_ORDER
        }
        mode = summary.get("mode")
        if mode not in {"smoke", "full"}:
            raise ValueError(f"invalid or missing summary mode {mode!r}")
        expected_summary = build_summary(
            canonical,
            expected_estimands,
            s0_manifest["reconciliation"],
            runconfig,
            mode,
        )
        registry_snapshot = directory / "opened_registry.snapshot.tsv"
        with registry_snapshot.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != REGISTRY_FIELDS:
                raise ValueError("opened registry snapshot header mismatch")
            registry_rows = [validate_registry_row(row) for row in reader]
        s0_hash = file_sha256(directory / "s0_manifest.json")
        required_registry_rows = (
            _registry_rows_from_canonical(runconfig, canonical, s0_hash)
            if mode == "full"
            else []
        )
        if mode == "full":
            canonical_registry = (
                Path(runconfig["repository_root"]) / runconfig["opened_registry"]
            )
            if not canonical_registry.is_file():
                violations.append("canonical opened registry is missing")
            elif file_sha256(canonical_registry) != file_sha256(registry_snapshot):
                violations.append("canonical opened registry differs from released snapshot")
        g8 = run_g8_reconciliation(
            canonical_records=canonical,
            correction_rows=corrections,
            collision_rows=collisions,
            matrices=matrices,
            summary=summary,
            expected_summary=expected_summary,
            expected_estimands=expected_estimands,
            emitted_estimands={name: sorted(ids) for name, ids in expected_estimands.items()},
            required_manifest_files=actual_files,
            manifest_hashes=manifest_hashes,
            registry_rows=registry_rows,
            required_registry_rows=required_registry_rows,
        )
        violations.extend(g8.violations)
        validation = json.loads((directory / "d0_validation.json").read_text(encoding="utf-8"))
        if not validation.get("release", {}).get("passed"):
            violations.append("recorded release verdict is not passed")
        if not (directory / "COMPLETE").is_file():
            violations.append("COMPLETE marker missing")
    except Exception as exc:
        violations.append(f"emitted validation exception: {type(exc).__name__}: {exc}")
    stable = tuple(sorted(set(violations)))
    return GateResult(
        name="EMITTED",
        blocking=True,
        passed=not stable,
        counts={"manifest_files": len(manifest_hashes)},
        violations=stable,
    )
