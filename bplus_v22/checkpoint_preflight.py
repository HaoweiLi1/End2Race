"""Task-9 checkpoint-backed forced-zero no-learning simulator preflight."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from bplus_v22 import ARMS, MACRO_STEPS, OWNER_DECISION
from bplus_v22.identity import ZeroResidualActor
from bplus_v22.manifests import REGISTRY_RELPATH, REGISTRY_SHA256, SMOKE_FIELDS, validate_manifest_release
from bplus_v22.model import V22Policy
from bplus_v22.release import file_sha256, validate_source_preflight
from bplus_v22.sidecar import _tensor_digest
from d25.oracle import ARRAY_KEYS, compare_archived, load_bc_model, simulate_episode
from d25.search import trajectory_digest


WARMSTART_RELEASE_RELPATH = (
    "logs/bplus_v22_d3r2_20260711/artifacts/warmstart_remediation_20260712_100124"
)
WARMSTART_OUTPUT_MANIFEST_SHA256 = (
    "57c6f900d57da1c59b46354c1502304576ad2ab352b03a29c8756f4bfce83252"
)
CHECKPOINT_SHA256 = {
    "BC_FROZEN": "149c1c2ae0c38fc9db16413027b3a829ee30b322634d5fa162e43cd07ec985e9",
    "SIDECAR_FROZEN": "d0ea9d2fd2cc1e192cc2bf6f2532054fa4c805b2b8fe5e257795bd2b271d7b76",
    "SIDECAR_FINETUNE": "bac35066ef0a691b502795641806f8964183187374833a049dbe18c37f047711",
}
BC_MODEL_RELPATH = "pretrained/end2race.pth"
RESULT_FIELDS = (
    "smoke_order",
    "l2_id",
    "map_name",
    "variant",
    "checkpoint_sha256",
    "run1_trajectory_sha256",
    "run2_trajectory_sha256",
    "baseline_trajectory_sha256",
    "run1_matches_baseline",
    "run2_matches_run1",
    "four_state",
    "action_clipped",
    "micro_steps",
    "macro_decisions",
    "macro_lengths_json",
    "short_terminal_macro",
    "forced_max_abs_residual_hex",
    "natural_brake_decisions",
    "natural_max_abs_residual_hex",
    "policy_decision_sha256",
    "diagnostic_sha256",
    "trajectory_relpath",
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _arrays_equal(left: dict, right: dict) -> bool:
    return all(
        key in left
        and key in right
        and np.asarray(left[key]).dtype == np.asarray(right[key]).dtype
        and np.asarray(left[key]).shape == np.asarray(right[key]).shape
        and np.array_equal(np.asarray(left[key]), np.asarray(right[key]))
        for key in ARRAY_KEYS
    )


def _validate_output_inventory(directory: Path) -> None:
    entries = {}
    for line in (directory / "output_manifest.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in entries:
            raise ValueError("duplicate output-manifest path")
        entries[relative] = digest
    observed = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*")
        if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
    }
    if set(entries) != observed:
        raise ValueError("output-manifest inventory mismatch")
    for relative, digest in entries.items():
        if file_sha256(directory / relative) != digest:
            raise ValueError(f"output hash mismatch: {relative}")


def _validate_historical_warmstart(root: Path) -> tuple[Path, dict[str, dict]]:
    """Verify the immutable Task-6 envelope without revalidating it as live source."""

    release = root / WARMSTART_RELEASE_RELPATH
    if not (release / "COMPLETE").is_file():
        raise ValueError("accepted warm-start release lacks COMPLETE")
    if file_sha256(release / "output_manifest.sha256") != WARMSTART_OUTPUT_MANIFEST_SHA256:
        raise ValueError("accepted warm-start output-manifest hash drift")
    _validate_output_inventory(release)
    config = json.loads((release / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("task6_acceptance_passed") is not True
        or config.get("ppo_checkpoint_eligible") is not True
        or config.get("ppo_training_started") is not False
        or config.get("closed_loop_evaluation_started") is not False
        or config.get("arm_selection_performed") is not False
    ):
        raise ValueError("accepted warm-start scope/acceptance mismatch")
    checkpoints: dict[str, dict] = {}
    for arm in ARMS:
        path = release / "checkpoints" / f"{arm}.pt"
        if file_sha256(path) != CHECKPOINT_SHA256[arm]:
            raise ValueError(f"accepted warm-start checkpoint hash drift: {arm}")
        if config["reports"][arm]["checkpoint_sha256"] != CHECKPOINT_SHA256[arm]:
            raise ValueError(f"accepted warm-start config checkpoint mismatch: {arm}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema") != "bplus-v2.2-warmstart-remediation-checkpoint-2"
            or payload.get("arm") != arm
            or not isinstance(payload.get("state_dict"), dict)
            or _tensor_digest(payload["state_dict"].items()) != payload.get("state_dict_sha256")
            or payload.get("state_dict_sha256") != config["reports"][arm]["final_state_sha256"]
        ):
            raise ValueError(f"accepted warm-start checkpoint envelope mismatch: {arm}")
        checkpoints[arm] = payload
    return release, checkpoints


def _write_output_manifest(directory: Path) -> None:
    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
    )
    (directory / "output_manifest.sha256").write_text(
        "\n".join(f"{file_sha256(path)}  {path.relative_to(directory).as_posix()}" for path in paths) + "\n",
        encoding="utf-8",
    )


def run_checkpoint_preflight(
    repo_root: str | Path,
    source_preflight_dir: str | Path,
    manifest_release_dir: str | Path,
    output_dir: str | Path,
    created_at: str,
    device_name: str = "cuda:0",
) -> dict:
    root = Path(repo_root).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError("Task-9 runner must execute from repository root")
    cache = os.environ.get("NUMBA_CACHE_DIR")
    if not cache or not Path(cache).is_absolute():
        raise ValueError("Task-9 requires an isolated absolute NUMBA_CACHE_DIR")
    if not validate_source_preflight(source_preflight_dir, root)["passed"]:
        raise ValueError("Task-9 source preflight invalid")
    manifest_validation = validate_manifest_release(manifest_release_dir, root)
    if not manifest_validation["passed"]:
        raise ValueError(f"Task-9 manifest release invalid: {manifest_validation}")
    if file_sha256(root / REGISTRY_RELPATH) != REGISTRY_SHA256:
        raise ValueError("Task-9 registry hash drift")
    warm_release, checkpoints = _validate_historical_warmstart(root)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Task-9 CUDA requested but unavailable")
    device = torch.device(device_name)
    cases = _read_tsv(Path(manifest_release_dir) / "no_learning_smoke.tsv")
    if len(cases) != 8 or tuple(cases[0]) != SMOKE_FIELDS:
        raise ValueError("Task-9 smoke manifest shape drift")

    output = Path(output_dir)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("Task-9 output/partial exists")
    partial.mkdir(parents=True)
    try:
        (partial / "trajectories").mkdir()
        bc = load_bc_model(str(root / BC_MODEL_RELPATH), device)
        baselines = {}
        rows: list[dict[str, str]] = []
        for case in cases:
            source_path = root / case["npz_relpath"]
            if file_sha256(source_path) != case["npz_sha256"]:
                raise ValueError(f"Task-9 source archive hash drift: {case['l2_id']}")
            with np.load(source_path, allow_pickle=False) as archive:
                archived = {name: np.asarray(archive[name]) for name in archive.files}
            first = simulate_episode(bc, device, case)
            second = simulate_episode(bc, device, case)
            if (
                not _arrays_equal(first.arrays, second.arrays)
                or not compare_archived(first.arrays, archived)["passed"]
                or first.action_clipped
                or second.action_clipped
            ):
                raise AssertionError(f"Task-9 BC replay mismatch: {case['l2_id']}")
            baselines[case["l2_id"]] = first
            relative = f"trajectories/{case['l2_id'][3:]}__BC.npz"
            np.savez_compressed(partial / relative, **{key: first.arrays[key] for key in ARRAY_KEYS})
            digest = trajectory_digest(first.arrays)
            rows.append({
                "smoke_order": case["smoke_order"],
                "l2_id": case["l2_id"],
                "map_name": case["map_name"],
                "variant": "BC",
                "checkpoint_sha256": "NA",
                "run1_trajectory_sha256": digest,
                "run2_trajectory_sha256": trajectory_digest(second.arrays),
                "baseline_trajectory_sha256": digest,
                "run1_matches_baseline": "true",
                "run2_matches_run1": "true",
                "four_state": first.outcome.four_state,
                "action_clipped": "false",
                "micro_steps": str(len(first.arrays["time"])),
                "macro_decisions": "0",
                "macro_lengths_json": "[]",
                "short_terminal_macro": "false",
                "forced_max_abs_residual_hex": float(0.0).hex(),
                "natural_brake_decisions": "0",
                "natural_max_abs_residual_hex": float(0.0).hex(),
                "policy_decision_sha256": "NA",
                "diagnostic_sha256": "NA",
                "trajectory_relpath": relative,
            })
        del bc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for arm in ARMS:
            policy = V22Policy(arm).to(device)
            policy.load_state_dict(checkpoints[arm]["state_dict"], strict=True)
            policy.eval()
            adapter = ZeroResidualActor(policy, require_natural_zero=False).to(device).eval()
            for case in cases:
                baseline = baselines[case["l2_id"]]
                adapter.reset_runtime()
                first = simulate_episode(adapter, device, case)
                first_accounting = adapter.accounting()
                adapter.reset_runtime()
                second = simulate_episode(adapter, device, case)
                second_accounting = adapter.accounting()
                if (
                    not _arrays_equal(first.arrays, baseline.arrays)
                    or not _arrays_equal(first.arrays, second.arrays)
                    or first_accounting != second_accounting
                    or first_accounting["max_abs_residual"] != 0.0
                    or first.action_clipped
                    or second.action_clipped
                ):
                    raise AssertionError(f"Task-9 forced-zero replay mismatch: {arm}/{case['l2_id']}")
                lengths = first_accounting["macro_lengths"]
                relative = f"trajectories/{case['l2_id'][3:]}__{arm}.npz"
                np.savez_compressed(partial / relative, **{key: first.arrays[key] for key in ARRAY_KEYS})
                rows.append({
                    "smoke_order": case["smoke_order"],
                    "l2_id": case["l2_id"],
                    "map_name": case["map_name"],
                    "variant": arm,
                    "checkpoint_sha256": CHECKPOINT_SHA256[arm],
                    "run1_trajectory_sha256": trajectory_digest(first.arrays),
                    "run2_trajectory_sha256": trajectory_digest(second.arrays),
                    "baseline_trajectory_sha256": trajectory_digest(baseline.arrays),
                    "run1_matches_baseline": "true",
                    "run2_matches_run1": "true",
                    "four_state": first.outcome.four_state,
                    "action_clipped": "false",
                    "micro_steps": str(first_accounting["micro_steps"]),
                    "macro_decisions": str(first_accounting["macro_decisions"]),
                    "macro_lengths_json": json.dumps(lengths),
                    "short_terminal_macro": str(bool(lengths and lengths[-1] < MACRO_STEPS)).lower(),
                    "forced_max_abs_residual_hex": float(first_accounting["max_abs_residual"]).hex(),
                    "natural_brake_decisions": str(first_accounting["natural_brake_decisions"]),
                    "natural_max_abs_residual_hex": float(first_accounting["max_abs_natural_residual"]).hex(),
                    "policy_decision_sha256": first_accounting["policy_decision_sha256"],
                    "diagnostic_sha256": first_accounting["diagnostic_sha256"],
                    "trajectory_relpath": relative,
                })
            del adapter, policy
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        _write_tsv(partial / "task9_results.tsv", rows, RESULT_FIELDS)
        p0 = {
            "schema": "bplus-v2.2-task9-p0-completeness-1",
            "expected_cells": 8 * 4,
            "observed_cells": len(rows),
            "expected_variants": ["BC", *ARMS],
            "reruns_per_cell": 2,
            "result_tsv_sha256": file_sha256(partial / "task9_results.tsv"),
            "trajectory_inventory_sha256": hashlib.sha256(
                "\n".join(
                    f"{row['trajectory_relpath']}:{row['run1_trajectory_sha256']}" for row in rows
                ).encode("utf-8") + b"\n"
            ).hexdigest(),
            "passed": len(rows) == 32,
        }
        (partial / "p0_completeness.json").write_text(json.dumps(p0, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        config = {
            "schema": "bplus-v2.2-task9-checkpoint-forced-zero-1",
            "created_at": str(created_at),
            "owner_decision": OWNER_DECISION,
            "mode": "accepted_warmstart_checkpoint_natural_diagnostic_forced_physical_zero",
            "natural_noop_required": False,
            "forced_physical_residual_required": True,
            "policy_training_started": False,
            "closed_loop_warmstart_evaluation_started": False,
            "ppo_training_started": False,
            "arm_selection_performed": False,
            "test_opened": False,
            "final_pool": False,
            "device": str(device),
            "numba_cache_dir": cache,
            "source_preflight_relpath": str(Path(source_preflight_dir)),
            "source_preflight_output_manifest_sha256": file_sha256(Path(source_preflight_dir) / "output_manifest.sha256"),
            "manifest_release_relpath": str(Path(manifest_release_dir)),
            "manifest_release_output_manifest_sha256": file_sha256(Path(manifest_release_dir) / "output_manifest.sha256"),
            "warmstart_release_relpath": str(warm_release.relative_to(root)),
            "warmstart_release_output_manifest_sha256": WARMSTART_OUTPUT_MANIFEST_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "registry_sha256": REGISTRY_SHA256,
            "cases": len(cases),
            "variants": ["BC", *ARMS],
            "reruns_per_variant": 2,
        }
        (partial / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (partial / "validation.json").write_text(json.dumps({
            "schema": "bplus-v2.2-task9-checkpoint-forced-zero-validation-1",
            "passed": True,
            "violations": [],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_output_manifest(partial)
        (partial / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    validation = validate_checkpoint_preflight(output)
    if not validation["passed"]:
        raise AssertionError(f"created invalid Task-9 preflight: {validation}")
    return validation | {"output_manifest_sha256": file_sha256(output / "output_manifest.sha256")}


def validate_checkpoint_preflight(release_dir: str | Path) -> dict:
    release = Path(release_dir)
    violations: list[str] = []
    rows: list[dict[str, str]] = []
    try:
        if not (release / "COMPLETE").is_file():
            raise ValueError("Task-9 release lacks COMPLETE")
        _validate_output_inventory(release)
        config = json.loads((release / "config.json").read_text(encoding="utf-8"))
        if (
            config["schema"] != "bplus-v2.2-task9-checkpoint-forced-zero-1"
            or config["owner_decision"] != OWNER_DECISION
            or config["natural_noop_required"] is not False
            or config["forced_physical_residual_required"] is not True
            or config["checkpoint_sha256"] != CHECKPOINT_SHA256
            or config["warmstart_release_output_manifest_sha256"] != WARMSTART_OUTPUT_MANIFEST_SHA256
            or config["registry_sha256"] != REGISTRY_SHA256
            or any(config[name] is not False for name in (
                "policy_training_started", "closed_loop_warmstart_evaluation_started", "ppo_training_started", "arm_selection_performed", "test_opened", "final_pool"
            ))
        ):
            raise ValueError("Task-9 authority/scope mismatch")
        with (release / "task9_results.tsv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
                raise ValueError("Task-9 result header drift")
            rows = list(reader)
        if len(rows) != 32:
            raise ValueError("Task-9 result Cartesian product incomplete")
        expected = {(str(index), variant) for index in range(8) for variant in ("BC", *ARMS)}
        if {(row["smoke_order"], row["variant"]) for row in rows} != expected:
            raise ValueError("Task-9 case/variant Cartesian product mismatch")
        baseline = {row["l2_id"]: row for row in rows if row["variant"] == "BC"}
        short_terminal_seen = False
        for row in rows:
            if (
                row["run1_matches_baseline"] != "true"
                or row["run2_matches_run1"] != "true"
                or row["run1_trajectory_sha256"] != row["run2_trajectory_sha256"]
                or row["run1_trajectory_sha256"] != row["baseline_trajectory_sha256"]
                or row["action_clipped"] != "false"
            ):
                raise ValueError("Task-9 identity/repeat gate failed")
            path = release / row["trajectory_relpath"]
            with np.load(path, allow_pickle=False) as arrays:
                if set(arrays.files) != set(ARRAY_KEYS) or trajectory_digest(arrays) != row["run1_trajectory_sha256"]:
                    raise ValueError("Task-9 saved trajectory hash mismatch")
            if row["variant"] == "BC":
                if row["checkpoint_sha256"] != "NA" or row["macro_lengths_json"] != "[]":
                    raise ValueError("Task-9 BC accounting drift")
            else:
                if row["checkpoint_sha256"] != CHECKPOINT_SHA256[row["variant"]]:
                    raise ValueError("Task-9 checkpoint continuity mismatch")
                lengths = json.loads(row["macro_lengths_json"])
                if (
                    sum(lengths) != int(row["micro_steps"])
                    or len(lengths) != int(row["macro_decisions"])
                    or any(not 1 <= length <= MACRO_STEPS for length in lengths)
                    or float.fromhex(row["forced_max_abs_residual_hex"]) != 0.0
                    or len(row["policy_decision_sha256"]) != 64
                    or len(row["diagnostic_sha256"]) != 64
                ):
                    raise ValueError("Task-9 macro/forced-zero accounting failed")
                short_terminal_seen = short_terminal_seen or row["short_terminal_macro"] == "true"
                if row["four_state"] != baseline[row["l2_id"]]["four_state"]:
                    raise ValueError("Task-9 outcome field mismatch")
        if not short_terminal_seen:
            raise ValueError("Task-9 never exercised a short terminal macro")
        p0 = json.loads((release / "p0_completeness.json").read_text(encoding="utf-8"))
        inventory_sha = hashlib.sha256(
            "\n".join(f"{row['trajectory_relpath']}:{row['run1_trajectory_sha256']}" for row in rows).encode("utf-8") + b"\n"
        ).hexdigest()
        if (
            p0["schema"] != "bplus-v2.2-task9-p0-completeness-1"
            or p0["expected_cells"] != 32
            or p0["observed_cells"] != 32
            or p0["reruns_per_cell"] != 2
            or p0["passed"] is not True
            or p0["result_tsv_sha256"] != file_sha256(release / "task9_results.tsv")
            or p0["trajectory_inventory_sha256"] != inventory_sha
        ):
            raise ValueError("Task-9 independent P0 completeness/hash gate failed")
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    return {
        "schema": "bplus-v2.2-task9-checkpoint-forced-zero-validation-1",
        "passed": not violations,
        "rows": len(rows),
        "violations": violations,
    }
