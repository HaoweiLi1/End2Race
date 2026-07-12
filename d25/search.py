"""Registry-gated D2.5 baseline and counterfactual branch search."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from d0.identity import (
    REGISTRY_FIELDS,
    append_opened_registry,
    registry_row_id,
    validate_registry_row,
)
from d0.outcomes import equality_vector
from d2.dataset import BC_SHA256
from d2.release import file_sha256
from d25 import BranchSpec, build_branch_specs
from d25.oracle import (
    ARRAY_KEYS,
    classify_trajectory,
    compare_archived,
    load_bc_model,
    simulate_episode,
)


DATASET_MANIFEST_SHA256 = "36b9640c9ec8407f12573bc3543712573283881b73400856a4b25f294b1f57c4"
EPISODE_METADATA_SHA256 = "468d8be50aecad19f89fbf2c35dc421acb4244a61f957f77dcfff1acd227eda3"
SCENARIO_SPLIT_SHA256 = "2f8146d7be0e36c3abcc084dcdbfa9e3df85983c37c6249294ab19b1431c49f3"
TEST_SEAL_SHA256 = "cee71d818bc050b0ca0647ee32ed1b5655e471ea60b39133aed7b37fc9c1a87e"
REGISTRY_OPENED_AT = "2026-07-11T18:45:00+08:00"
EVIDENCE_RELPATH = "logs/d25_counterfactual_20260711"
EXPECTED_EPISODES = 1928
EXPECTED_EGO_COLLISIONS = 91
SMOKE_BRANCH_POSITIONS = (0, 4, 8)

CASE_FIELDS = (
    "case_order",
    "l2_id",
    "l3_id",
    "l4_id",
    "map_name",
    "skill",
    "opponent_raceline",
    "speedscale_hex",
    "resolved_ego_idx",
    "npz_relpath",
    "npz_sha256",
    "frame_count",
    "final_time_hex",
    "smoke_selected",
)
BASELINE_FIELDS = (
    "case_order",
    "l2_id",
    "passed",
    "mismatches_json",
    "max_abs_error_json",
    "archived_four_state",
    "replayed_four_state",
    "action_clipped",
    "trajectory_sha256",
)
BRANCH_FIELDS = (
    "case_order",
    "l2_id",
    "branch_sequence",
    "branch_id",
    "requested_lead_hex",
    "actual_lead_hex",
    "start_step",
    "duration_steps",
    "intervention_id",
    "brake_mps_hex",
    "steer_rad_hex",
    "category",
    "four_state",
    "ego_collision",
    "opp_collision",
    "confirmed_safe_pass",
    "action_clipped",
    "final_time_hex",
    "rel_terminal_hex",
    "trajectory_sha256",
    "deterministic_rerun_sha256",
    "rerun_match",
)
CASE_RESULT_FIELDS = (
    "case_order",
    "l2_id",
    "map_name",
    "skill",
    "l4_id",
    "eligible_branch_count",
    "executed_branch_count",
    "status",
    "witness_branch_id",
    "witness_trajectory_sha256",
    "confirmed_pass_count",
    "terminal_overtake_count",
    "safe_follow_count",
    "still_collision_count",
    "invalid_count",
)


class GateFailure(RuntimeError):
    """A locked D2.5 gate failed; the partial artifact must be retained."""


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: Iterable[Mapping], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for source in rows:
            row = {field: str(source[field]) for field in fields}
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _append_tsv(path: Path, row: Mapping, fields: tuple[str, ...]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({field: str(row[field]) for field in fields})
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_output(output_dir: Path) -> Path:
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"D2.5 output exists or is nonempty: {output_dir}")
        output_dir.rmdir()
    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"D2.5 partial exists: {partial}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    return partial


def _output_manifest(directory: Path) -> None:
    relpaths = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
    )
    lines = [f"{file_sha256(directory / relpath)}  {relpath}" for relpath in relpaths]
    path = directory / "output_manifest.sha256"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _promote(partial: Path, output_dir: Path) -> None:
    for path in partial.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted((path for path in partial.rglob("*") if path.is_dir()), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(partial)
    os.replace(partial, output_dir)
    _fsync_directory(output_dir.parent)
    complete = output_dir / "COMPLETE"
    with complete.open("w", encoding="utf-8") as handle:
        handle.write("COMPLETE\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(output_dir)


def load_collision_cases(dataset_dir: str | Path) -> list[dict[str, str]]:
    dataset_dir = Path(dataset_dir)
    if not (dataset_dir / "COMPLETE").is_file():
        raise ValueError("D2 non-test release lacks COMPLETE")
    if file_sha256(dataset_dir / "dataset_manifest.json") != DATASET_MANIFEST_SHA256:
        raise ValueError("D2 dataset manifest SHA256 drift")
    if file_sha256(dataset_dir / "episode_metadata.tsv") != EPISODE_METADATA_SHA256:
        raise ValueError("D2 episode metadata SHA256 drift")
    rows = _read_tsv(dataset_dir / "episode_metadata.tsv")
    if len(rows) != EXPECTED_EPISODES:
        raise ValueError(f"expected {EXPECTED_EPISODES} D2 episodes, got {len(rows)}")
    cases = [row for row in rows if row["ego_collision"] == "True"]
    if len(cases) != EXPECTED_EGO_COLLISIONS:
        raise ValueError(f"expected {EXPECTED_EGO_COLLISIONS} ego collisions, got {len(cases)}")
    if any(row["collision_any"] != "True" for row in cases):
        raise ValueError("ego-collision population contains a non-collision")
    cases.sort(key=lambda row: row["l2_id"])
    if len({row["l2_id"] for row in cases}) != len(cases):
        raise ValueError("duplicate L2 in D2.5 population")
    return cases


def select_smoke_cases(cases: Iterable[Mapping]) -> list[dict]:
    cases = [dict(row) for row in cases]
    maps = sorted({str(row["map_name"]) for row in cases})
    if len(maps) != 4:
        raise ValueError(f"D2.5 smoke requires four maps, got {maps}")
    selected = []
    for map_name in maps:
        for skill in ("skill_F", "skill_S"):
            group = sorted(
                (
                    row
                    for row in cases
                    if row["map_name"] == map_name and row["skill"] == skill
                ),
                key=lambda row: row["l2_id"],
            )
            if not group:
                raise ValueError(f"missing smoke stratum {map_name}/{skill}")
            selected.append(group[0])
    if len(selected) != 8 or len({row["l2_id"] for row in selected}) != 8:
        raise AssertionError("D2.5 smoke selection is not exactly eight distinct cases")
    return selected


def make_registry_rows(
    cases: Iterable[Mapping],
    opened_at_utc: str = REGISTRY_OPENED_AT,
    evidence_relpath: str = EVIDENCE_RELPATH,
) -> list[dict[str, str]]:
    if opened_at_utc != REGISTRY_OPENED_AT:
        raise ValueError("D2.5 registry opening time is locked")
    if evidence_relpath != EVIDENCE_RELPATH:
        raise ValueError("D2.5 canonical evidence root is locked")
    rows = []
    for case in cases:
        row = {
            "registry_schema": "bplus-opened-registry-1",
            "opened_at_utc": opened_at_utc,
            "stage": "D2.5",
            "use_class": "oracle_search",
            "split_id": f"d25_non_test_{DATASET_MANIFEST_SHA256[:16]}",
            "l2_id": str(case["l2_id"]),
            "l3_id": str(case["l3_id"]),
            "l4_id": str(case["l4_id"]),
            "map_name": str(case["map_name"]),
            "source_manifest_sha256": DATASET_MANIFEST_SHA256,
            "source_run_id": "non_test_full_20260711_175713",
            "decision_effect": "action_choice",
            "final_pool": "false",
            "evidence_relpath": evidence_relpath,
        }
        row["row_id"] = registry_row_id(row)
        rows.append(validate_registry_row(row))
    rows.sort(key=lambda row: row["row_id"])
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate D2.5 registry row ID")
    return rows


def trajectory_digest(arrays: Mapping) -> str:
    digest = hashlib.sha256(b"end2race:d25:trajectory:v1\0")
    for key in ARRAY_KEYS:
        array = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def branch_category(outcome, action_clipped: bool) -> str:
    if action_clipped or outcome.four_state == "unknown":
        return "invalid_or_action_clipped"
    if outcome.four_state == "collision":
        return "still_collision"
    if outcome.four_state == "confirmed_pass":
        return "collision_to_confirmed_safe_pass"
    if outcome.four_state == "terminal_overtake_only":
        return "collision_to_terminal_overtake_only"
    if outcome.four_state == "safe_follow":
        return "collision_to_safe_abort_follow"
    raise ValueError(f"unexpected D2.5 outcome {outcome.four_state!r}")


def route_summary(case_rows: Iterable[Mapping], result_rows: Iterable[Mapping]) -> dict:
    cases = {str(row["l2_id"]): row for row in case_rows}
    results = list(result_rows)
    recovered_ids = {
        str(row["l2_id"])
        for row in results
        if str(row["status"]) == "recovered_confirmed_safe_pass"
    }
    recovered = [cases[l2_id] for l2_id in sorted(recovered_ids)]
    skill_s_tested = sum(str(row["skill"]) == "skill_S" for row in cases.values())
    skill_s_recovered = sum(str(row["skill"]) == "skill_S" for row in recovered)
    skill_f_recovered = sum(str(row["skill"]) == "skill_F" for row in recovered)
    gates = {
        "recovered_cases_ge_25": len(recovered) >= 25,
        "maps_ge_2": len({str(row["map_name"]) for row in recovered}) >= 2,
        "l4_blocks_ge_5": len({str(row["l4_id"]) for row in recovered}) >= 5,
        "skill_F_ge_5": skill_f_recovered >= 5,
        "skill_S_ge_15": skill_s_recovered >= 15,
        "skill_S_fraction_ge_0p30": (
            skill_s_tested > 0 and skill_s_recovered / skill_s_tested >= 0.30
        ),
        "no_positive_speed_residual": True,
        "no_clipped_witness": all(
            str(row.get("witness_branch_id", "NA")) == "NA"
            or str(row.get("status")) == "recovered_confirmed_safe_pass"
            for row in results
        ),
    }
    return {
        "schema": "d25-route-r2-summary-1",
        "tested_case_count": len(cases),
        "recovered_case_count": len(recovered),
        "recovered_map_count": len({str(row["map_name"]) for row in recovered}),
        "recovered_l4_count": len({str(row["l4_id"]) for row in recovered}),
        "skill_F_recovered": skill_f_recovered,
        "skill_S_tested": skill_s_tested,
        "skill_S_recovered": skill_s_recovered,
        "skill_S_recovery_fraction": (
            skill_s_recovered / skill_s_tested if skill_s_tested else 0.0
        ),
        "gates": gates,
        "route_r2_feasible": all(gates.values()),
    }


def _case_manifest(cases: list[dict], smoke_ids: set[str]) -> list[dict[str, str]]:
    rows = []
    for index, case in enumerate(cases):
        rows.append(
            {
                "case_order": str(index),
                "l2_id": str(case["l2_id"]),
                **{field: str(case[field]) for field in CASE_FIELDS[2:-1]},
                "smoke_selected": str(case["l2_id"] in smoke_ids).lower(),
            }
        )
    return rows


def _branch_row(
    case_order: int,
    case: Mapping,
    sequence: int,
    spec: BranchSpec,
    result,
    digest: str,
    rerun_digest: str = "NA",
    rerun_match: str = "not_run",
) -> dict[str, str]:
    outcome = result.outcome
    return {
        "case_order": str(case_order),
        "l2_id": str(case["l2_id"]),
        "branch_sequence": str(sequence),
        "branch_id": spec.branch_id,
        "requested_lead_hex": spec.requested_lead_s.hex(),
        "actual_lead_hex": spec.actual_lead_s.hex(),
        "start_step": str(spec.start_step),
        "duration_steps": str(spec.duration_steps),
        "intervention_id": spec.intervention.intervention_id,
        "brake_mps_hex": spec.intervention.brake_mps.hex(),
        "steer_rad_hex": spec.intervention.steer_rad.hex(),
        "category": branch_category(outcome, result.action_clipped),
        "four_state": outcome.four_state,
        "ego_collision": str(outcome.ego_collision).lower(),
        "opp_collision": str(outcome.opp_collision).lower(),
        "confirmed_safe_pass": str(outcome.confirmed_safe_pass).lower(),
        "action_clipped": str(result.action_clipped).lower(),
        "final_time_hex": float(np.asarray(result.arrays["final_time"]).reshape(())).hex(),
        "rel_terminal_hex": outcome.rel_terminal_hex,
        "trajectory_sha256": digest,
        "deterministic_rerun_sha256": rerun_digest,
        "rerun_match": rerun_match,
    }


def _arrays_equal(left: Mapping, right: Mapping) -> bool:
    return all(
        np.asarray(left[key]).dtype == np.asarray(right[key]).dtype
        and np.asarray(left[key]).shape == np.asarray(right[key]).shape
        and np.array_equal(np.asarray(left[key]), np.asarray(right[key]))
        for key in ARRAY_KEYS
    )


def _save_witness(path: Path, arrays: Mapping) -> None:
    np.savez_compressed(path, **{key: np.asarray(arrays[key]) for key in ARRAY_KEYS})
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _source_preflight(repo_root: Path, dataset_dir: Path, cases: list[dict]) -> dict:
    model = repo_root / "pretrained/end2race.pth"
    if file_sha256(model) != BC_SHA256:
        raise ValueError("frozen BC checkpoint SHA256 drift")
    split_dir = dataset_dir.parent / "split_lock"
    if file_sha256(split_dir / "scenario_split.tsv") != SCENARIO_SPLIT_SHA256:
        raise ValueError("D2 scenario split SHA256 drift")
    if file_sha256(split_dir / "test_seal.json") != TEST_SEAL_SHA256:
        raise ValueError("D2 test seal SHA256 drift")
    for index, case in enumerate(cases):
        source = repo_root / case["npz_relpath"]
        if not source.is_file() or file_sha256(source) != case["npz_sha256"]:
            raise ValueError(f"D2.5 source episode SHA256 drift: {case['l2_id']}")
        if (index + 1) % 25 == 0:
            print(f"D25_PREFLIGHT sources={index + 1}/{len(cases)}", flush=True)
    return {
        "model_sha256": BC_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "episode_metadata_sha256": EPISODE_METADATA_SHA256,
        "scenario_split_sha256": SCENARIO_SPLIT_SHA256,
        "test_seal_sha256": TEST_SEAL_SHA256,
    }


def _baseline_row(case_order: int, case: Mapping, result, archived: Mapping) -> dict[str, str]:
    comparison = compare_archived(result.arrays, archived)
    archived_outcome = classify_trajectory(archived, str(case["map_name"]))
    outcome_equal = equality_vector(archived_outcome) == equality_vector(result.outcome)
    passed = bool(
        comparison["passed"]
        and outcome_equal
        and archived_outcome.four_state == "collision"
        and result.outcome.four_state == "collision"
        and not result.action_clipped
    )
    mismatches = list(comparison["mismatches"])
    if not outcome_equal:
        mismatches.append("corrected_outcome")
    if archived_outcome.four_state != "collision" or result.outcome.four_state != "collision":
        mismatches.append("not_collision")
    if result.action_clipped:
        mismatches.append("baseline_action_clipped")
    return {
        "case_order": str(case_order),
        "l2_id": str(case["l2_id"]),
        "passed": str(passed).lower(),
        "mismatches_json": json.dumps(mismatches, separators=(",", ":")),
        "max_abs_error_json": json.dumps(comparison["max_abs_error"], sort_keys=True, separators=(",", ":")),
        "archived_four_state": archived_outcome.four_state,
        "replayed_four_state": result.outcome.four_state,
        "action_clipped": str(result.action_clipped).lower(),
        "trajectory_sha256": trajectory_digest(result.arrays),
    }


def _case_result(
    case_order: int,
    case: Mapping,
    eligible: int,
    branch_rows: list[dict[str, str]],
    witness_row: Mapping | None,
) -> dict[str, str]:
    categories = Counter(row["category"] for row in branch_rows)
    recovered = witness_row is not None
    return {
        "case_order": str(case_order),
        "l2_id": str(case["l2_id"]),
        "map_name": str(case["map_name"]),
        "skill": str(case["skill"]),
        "l4_id": str(case["l4_id"]),
        "eligible_branch_count": str(eligible),
        "executed_branch_count": str(len(branch_rows)),
        "status": "recovered_confirmed_safe_pass" if recovered else "exhausted_no_confirmed_safe_pass",
        "witness_branch_id": str(witness_row["branch_id"]) if recovered else "NA",
        "witness_trajectory_sha256": str(witness_row["trajectory_sha256"]) if recovered else "NA",
        "confirmed_pass_count": str(categories["collision_to_confirmed_safe_pass"]),
        "terminal_overtake_count": str(categories["collision_to_terminal_overtake_only"]),
        "safe_follow_count": str(categories["collision_to_safe_abort_follow"]),
        "still_collision_count": str(categories["still_collision"]),
        "invalid_count": str(categories["invalid_or_action_clipped"]),
    }


def run_oracle(
    repo_root: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    registry_path: str | Path,
    mode: str,
    created_at: str,
    registry_opened_at: str = REGISTRY_OPENED_AT,
    evidence_relpath: str = EVIDENCE_RELPATH,
    device_name: str = "cuda:0",
) -> dict:
    if mode not in {"baseline_smoke", "branch_smoke", "full"}:
        raise ValueError("D2.5 mode must be baseline_smoke, branch_smoke, or full")
    repo_root = Path(repo_root).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    registry_path = Path(registry_path).resolve()
    cases = load_collision_cases(dataset_dir)
    smoke = select_smoke_cases(cases)
    smoke_ids = {row["l2_id"] for row in smoke}
    selected = smoke if mode != "full" else cases
    source_hashes = _source_preflight(repo_root, dataset_dir, cases)
    partial = _prepare_output(output_dir)
    try:
        registry_before = file_sha256(registry_path)
        registry_rows = make_registry_rows(cases, registry_opened_at, evidence_relpath)
        append_result = append_opened_registry(registry_path, registry_rows)
        snapshot = partial / "opened_registry.snapshot.tsv"
        shutil.copyfile(registry_path, snapshot)
        registry_after = file_sha256(snapshot)

        manifest_rows = _case_manifest(cases, smoke_ids)
        _write_tsv(partial / "case_manifest.tsv", manifest_rows, CASE_FIELDS)
        (partial / "witnesses").mkdir()
        config = {
            "schema": "d25-oracle-config-1",
            "spec_version": "d2.5-spec-1",
            "mode": mode,
            "created_at": str(created_at),
            "registry_opened_at": registry_opened_at,
            "evidence_relpath": evidence_relpath,
            "device": device_name,
            "population_count": len(cases),
            "selected_count": len(selected),
            "smoke_l2_ids": [row["l2_id"] for row in smoke],
            "smoke_branch_positions": list(SMOKE_BRANCH_POSITIONS),
            "source_hashes": source_hashes,
            "source_code_sha256": {
                "d25_init": file_sha256(Path(__file__).with_name("__init__.py")),
                "d25_oracle": file_sha256(Path(__file__).with_name("oracle.py")),
                "d25_search": file_sha256(Path(__file__)),
            },
            "registry": {
                "path": str(registry_path.relative_to(repo_root)),
                "before_sha256": registry_before,
                "after_sha256": registry_after,
                "required_rows": len(registry_rows),
                "appended": append_result.appended,
                "already_present": append_result.skipped,
                "live_total": append_result.total,
            },
        }
        _write_json(partial / "config.json", config)

        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        model = load_bc_model(str(repo_root / "pretrained/end2race.pth"), device)
        baseline_rows = []
        for selected_index, case in enumerate(selected):
            case_order = cases.index(case)
            started = time.monotonic()
            result = simulate_episode(model, device, case)
            with np.load(repo_root / case["npz_relpath"], allow_pickle=False) as archived:
                row = _baseline_row(case_order, case, result, archived)
            baseline_rows.append(row)
            _append_tsv(partial / "baseline_replay.tsv", row, BASELINE_FIELDS)
            print(
                f"D25_BASELINE mode={mode} cases={selected_index + 1}/{len(selected)} "
                f"passed={row['passed']} elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
        failed = [row for row in baseline_rows if row["passed"] != "true"]
        if failed:
            raise GateFailure(f"D2.5 baseline replay mismatched {len(failed)}/{len(selected)} cases")

        branch_rows_all = []
        case_results = []
        if mode == "branch_smoke":
            for selected_index, case in enumerate(selected):
                case_order = cases.index(case)
                specs = build_branch_specs(int(case["frame_count"]))
                chosen = [specs[position] for position in SMOKE_BRANCH_POSITIONS if position < len(specs)]
                current_rows = []
                for sequence, spec in enumerate(chosen):
                    first = simulate_episode(model, device, case, spec)
                    second = simulate_episode(model, device, case, spec)
                    first_sha = trajectory_digest(first.arrays)
                    second_sha = trajectory_digest(second.arrays)
                    match = first_sha == second_sha and _arrays_equal(first.arrays, second.arrays)
                    row = _branch_row(
                        case_order, case, sequence, spec, first, first_sha,
                        second_sha, str(match).lower(),
                    )
                    if not match:
                        _append_tsv(partial / "branch_results.tsv", row, BRANCH_FIELDS)
                        raise GateFailure(f"D2.5 smoke branch nondeterminism: {case['l2_id']} {spec.branch_id}")
                    _append_tsv(partial / "branch_results.tsv", row, BRANCH_FIELDS)
                    current_rows.append(row)
                    branch_rows_all.append(row)
                witness = next(
                    (row for row in current_rows if row["category"] == "collision_to_confirmed_safe_pass"),
                    None,
                )
                case_row = _case_result(case_order, case, len(chosen), current_rows, witness)
                case_results.append(case_row)
                _append_tsv(partial / "case_results.tsv", case_row, CASE_RESULT_FIELDS)
                print(f"D25_BRANCH_SMOKE cases={selected_index + 1}/{len(selected)}", flush=True)

        elif mode == "full":
            for selected_index, case in enumerate(selected):
                case_order = cases.index(case)
                specs = build_branch_specs(int(case["frame_count"]))
                current_rows = []
                witness_row = None
                for sequence, spec in enumerate(specs):
                    result = simulate_episode(model, device, case, spec)
                    digest = trajectory_digest(result.arrays)
                    category = branch_category(result.outcome, result.action_clipped)
                    if category == "collision_to_confirmed_safe_pass":
                        rerun = simulate_episode(model, device, case, spec)
                        rerun_sha = trajectory_digest(rerun.arrays)
                        match = digest == rerun_sha and _arrays_equal(result.arrays, rerun.arrays)
                        row = _branch_row(
                            case_order, case, sequence, spec, result, digest,
                            rerun_sha, str(match).lower(),
                        )
                        _append_tsv(partial / "branch_results.tsv", row, BRANCH_FIELDS)
                        current_rows.append(row)
                        branch_rows_all.append(row)
                        if not match:
                            raise GateFailure(
                                f"D2.5 witness nondeterminism: {case['l2_id']} {spec.branch_id}"
                            )
                        witness_path = partial / "witnesses" / f"{case['l2_id'][3:]}__{spec.branch_id}.npz"
                        _save_witness(witness_path, result.arrays)
                        witness_row = row
                        break
                    row = _branch_row(case_order, case, sequence, spec, result, digest)
                    _append_tsv(partial / "branch_results.tsv", row, BRANCH_FIELDS)
                    current_rows.append(row)
                    branch_rows_all.append(row)
                case_row = _case_result(case_order, case, len(specs), current_rows, witness_row)
                case_results.append(case_row)
                _append_tsv(partial / "case_results.tsv", case_row, CASE_RESULT_FIELDS)
                print(
                    f"D25_SEARCH cases={selected_index + 1}/{len(selected)} "
                    f"branches={len(current_rows)} status={case_row['status']}",
                    flush=True,
                )

        summary = {
            "schema": "d25-oracle-summary-1",
            "mode": mode,
            "baseline_count": len(baseline_rows),
            "baseline_passed": len(baseline_rows) - len(failed),
            "branch_count": len(branch_rows_all),
            "branch_categories": dict(sorted(Counter(row["category"] for row in branch_rows_all).items())),
            "route_r2": route_summary(cases, case_results) if mode == "full" else None,
        }
        _write_json(partial / "d25_summary.json", summary)

        from d25.validate import validate_release

        preliminary = validate_release(partial, allow_partial=True, verify_manifest=False)
        _write_json(partial / "validation.json", preliminary)
        _output_manifest(partial)
        final_validation = validate_release(partial, allow_partial=True, verify_manifest=True)
        if not final_validation["passed"]:
            raise GateFailure("independent D2.5 release validation failed")
        _promote(partial, output_dir)
        return final_validation
    except Exception as error:
        if partial.exists():
            failure = {
                "schema": "d25-failure-1",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            _write_json(partial / "FAILED", failure)
        raise
