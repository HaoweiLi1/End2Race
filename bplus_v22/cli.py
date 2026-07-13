#!/usr/bin/env python3
"""Audited B+ v2.2 preflight, initialization, and validation CLI."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import torch

from bplus_v22.release import create_source_preflight, validate_source_preflight
from bplus_v22.identity import run_zero_identity, validate_zero_identity
from bplus_v22.checkpoint_preflight import (
    run_checkpoint_preflight,
    validate_checkpoint_preflight,
)
from bplus_v22.manifests import create_manifest_release, validate_manifest_release
from bplus_v22.closed_loop import run_closed_loop_warmstart, validate_closed_loop_release
from bplus_v22.sidecar import (
    create_registry_plan,
    run_sidecar_initialization,
    validate_registry_plan,
    validate_sidecar_release,
)
from bplus_v22.warmstart import (
    create_warmstart_manifest,
    run_warmstart_smoke,
    validate_warmstart_manifest,
    validate_warmstart_release,
)
from bplus_v22.hierarchical_identity import (
    run_hierarchical_identity,
    validate_hierarchical_identity,
)
from bplus_v22.hierarchical_warmstart import (
    create_hierarchical_warmstart_manifest,
    run_hierarchical_warmstart,
    validate_hierarchical_warmstart_manifest,
    validate_hierarchical_warmstart_release,
)
from bplus_v22.hierarchical_checkpoint_preflight import (
    run_hierarchical_checkpoint_preflight,
    validate_hierarchical_checkpoint_preflight,
)
from bplus_v22.hierarchical_closed_loop import (
    run_hierarchical_closed_loop,
    validate_hierarchical_closed_loop,
)
from bplus_v22.ppo_runner import (
    B3_CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA as B2_CHECKPOINT_SCHEMA,
    load_policy_only_checkpoint,
    run_plumbing_smoke,
    run_pilot_job,
    validate_pilot_plan,
)
from bplus_v22.ppo_eval import (
    BASELINE_SHARD_COUNT,
    CandidateCheckpoint,
    EvaluationShard,
    LoadedCandidatePolicy,
    baseline_json_bytes,
    evaluate_bc_baseline_shard,
    evaluate_shard,
    merge_evaluation_shards,
    read_task8_development,
)
from d25.oracle import load_bc_model


B2_CAPABILITIES_SCHEMA = "bplus-v22-cli-capabilities-1"
B2_COMMANDS = (
    "ppo-baseline-preflight",
    "ppo-pilot",
    "ppo-evaluate",
    "ppo-merge-eval",
    "ppo-plumbing-smoke",
)
B2_RUN_PLAN_SCHEMA = "end2race-b2-run-plan-1"


def _canonical_json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_eval_plan(plan_path: str | Path, job_id: str | None = None):
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != B2_RUN_PLAN_SCHEMA or plan.get("kind") not in {
        "b2_eval",
        "b3_eval",
    }:
        raise ValueError("PPO evaluator requires a b2_eval or b3_eval RunPlan")
    config = plan.get("config")
    expected_contract = (
        "unified_standard_mode_v1"
        if plan["kind"] == "b3_eval"
        else "centered_fresh_prior"
    )
    expected_iteration = 40 if plan["kind"] == "b3_eval" else 20
    if (
        not isinstance(config, dict)
        or config.get("policy_contract", "centered_fresh_prior")
        != expected_contract
        or int(config.get("checkpoint_iteration", expected_iteration))
        != expected_iteration
        or config.get("evaluation_offsets") != [0.0, 0.0]
    ):
        raise ValueError("PPO eval policy/iteration contract mismatch")
    observed = str(plan.get("plan_sha256", ""))
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    expected = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if observed != expected:
        raise ValueError("B2 eval RunPlan digest mismatch")
    parent_plan_sha = str(plan.get("parent_plan_sha256", ""))
    if len(parent_plan_sha) != 64 or any(
        character not in "0123456789abcdef" for character in parent_plan_sha
    ):
        raise ValueError("B2 eval RunPlan parent digest is invalid")
    root = path.parent.parent
    if Path.cwd().resolve() != (root / "repo").resolve():
        raise ValueError("B2 evaluator must execute from staged repository")
    contract = plan.get("evaluation_contract")
    if not isinstance(contract, dict):
        raise ValueError("B2 eval plan lacks evaluation contract")
    if (
        int(contract.get("expected_scenario_count", -1)) != 288
        or int(contract.get("expected_episode_rows", -1)) != 2016
        or int(contract.get("shard_count", -1)) != 4
        or len(contract.get("checkpoint_set", ())) != 6
    ):
        raise ValueError("B2 eval Cartesian contract drift")
    jobs = {str(row["job_id"]): row for row in plan.get("jobs", [])}
    selected = None
    if job_id is not None:
        if job_id not in jobs:
            raise ValueError(f"unknown B2 eval job: {job_id}")
        selected = jobs[job_id]
        if selected.get("kind") != "evaluation_shard":
            raise ValueError("B2 eval job kind drift")
        output = root / str(selected["output_relpath"])
        if output.exists() or output.with_name(output.name + ".partial").exists():
            raise FileExistsError(output)
    manifest = root / str(contract["manifest_relpath"])
    if not manifest.is_file() or _sha256_file(manifest) != contract["manifest_sha256"]:
        raise ValueError("B2 eval Task-8 manifest drift")
    return path, plan, root, contract, selected


def _checkpoint_specs(
    root: Path,
    contract: dict,
    parent_plan_sha256: str,
    *,
    eval_kind: str,
    expected_iteration: int,
):
    if eval_kind not in {"b2_eval", "b3_eval"}:
        raise ValueError("unsupported PPO evaluation kind")
    training_sha = str(contract.get("training_manifest_sha256", ""))
    if len(training_sha) != 64:
        raise ValueError("B2 eval plan lacks training-manifest digest")
    specs = []
    paths = {}
    for row in contract["checkpoint_set"]:
        path = root / str(row["relpath"])
        if not path.is_file() or _sha256_file(path) != row["sha256"]:
            raise ValueError("B2 eval checkpoint file drift")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected_schema = (
            B3_CHECKPOINT_SCHEMA if eval_kind == "b3_eval" else B2_CHECKPOINT_SCHEMA
        )
        if (
            payload.get("schema") != expected_schema
            or payload.get("arm") != row["arm"]
            or payload.get("seed") != int(row["seed"])
            or payload.get("iteration") != int(expected_iteration)
            or payload.get("training_manifest_sha256") != training_sha
            or payload.get("run_plan_sha256") != parent_plan_sha256
        ):
            raise ValueError("B2 eval checkpoint envelope mismatch")
        spec = CandidateCheckpoint(
            arm=str(row["arm"]),
            seed=int(row["seed"]),
            checkpoint_id=str(payload["checkpoint_id"]),
            checkpoint_sha256=str(row["sha256"]),
            training_manifest_sha256=training_sha,
        )
        specs.append(spec)
        paths[spec.variant] = path
    return tuple(specs), paths, training_sha


def validate_eval_plan(plan_path: str | Path) -> dict:
    _, plan, root, contract, _ = _load_eval_plan(plan_path)
    specs, _, training_sha = _checkpoint_specs(
        root,
        contract,
        str(plan["parent_plan_sha256"]),
        eval_kind=str(plan["kind"]),
        expected_iteration=int(plan["config"]["checkpoint_iteration"]),
    )
    return {
        "passed": len(specs) == 6,
        "checkpoint_count": len(specs),
        "training_manifest_sha256": training_sha,
        "scenario_count": int(contract["expected_scenario_count"]),
    }


def run_bc_baseline_shard(
    plan_path: str | Path,
    output_path: str | Path,
    *,
    host_id: str,
    gpu_uuid: str,
    shard_index: int,
    shard_count: int = BASELINE_SHARD_COUNT,
    device_name: str = "cuda:0",
) -> dict:
    validated = validate_pilot_plan(plan_path)
    plan = validated["plan"]
    paths = validated["paths"]
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("B2 BC baseline preflight requested unavailable CUDA")
    manifest = paths["task8"] / "development_scenarios.tsv"
    manifest_sha = _sha256_file(manifest)
    rows = read_task8_development(manifest, manifest_sha)
    bc_sha = _sha256_file(paths["bc"])
    bc = load_bc_model(str(paths["bc"]), device)
    shard = evaluate_bc_baseline_shard(
        task8_rows=rows,
        scenario_manifest_sha256=manifest_sha,
        bc_model=bc,
        bc_checkpoint_sha256=bc_sha,
        device=device,
        shard_index=shard_index,
        shard_count=shard_count,
        run_plan_sha256=plan["plan_sha256"],
        source_commit=plan["source_commit"],
        source_archive_sha256=plan["source_archive_sha256"],
        inputs_archive_sha256=plan["inputs_archive_sha256"],
        producer_host_id=host_id,
        producer_gpu_uuid=gpu_uuid,
    )
    result = shard.to_dict()
    output = Path(output_path)
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("xb") as handle:
        handle.write(baseline_json_bytes(result))
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(partial, 0o444)
    os.replace(partial, output)
    return {
        "passed": True,
        "shard_index": result["shard_index"],
        "scenario_count": result["scenario_count"],
        "collision": result["collision"],
        "terminal_overtake": result["terminal_overtake"],
    }


def run_plumbing_smoke_release(
    plan_path: str | Path,
    output_path: str | Path,
    device_name: str = "cuda:0",
) -> dict:
    result = run_plumbing_smoke(plan_path, device_name=device_name)
    output = Path(output_path)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, output)
    return {
        "passed": True,
        "arm_count": len(result["arms"]),
        "scenario_count_per_arm": 4,
    }


def _write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_eval_job(plan_path: str | Path, job_id: str, device_name: str = "cuda:0") -> dict:
    _, plan, root, contract, job = _load_eval_plan(plan_path, job_id)
    parent_plan_sha = str(plan["parent_plan_sha256"])
    specs, checkpoint_paths, training_sha = _checkpoint_specs(
        root,
        contract,
        parent_plan_sha,
        eval_kind=str(plan["kind"]),
        expected_iteration=int(plan["config"]["checkpoint_iteration"]),
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("B2 evaluation CUDA requested but unavailable")
    manifest = root / str(contract["manifest_relpath"])
    rows = read_task8_development(manifest, contract["manifest_sha256"])
    bc_path = root / "repo/pretrained/end2race.pth"
    bc_sha = _sha256_file(bc_path)
    bc = load_bc_model(str(bc_path), device)

    def loader(expected, target_device):
        policy, payload = load_policy_only_checkpoint(
            checkpoint_paths[expected.variant],
            expected_arm=expected.arm,
            expected_seed=expected.seed,
            expected_iteration=int(plan["config"]["checkpoint_iteration"]),
            expected_training_manifest_sha256=training_sha,
            expected_plan_sha256=parent_plan_sha,
            expected_checkpoint_sha256=expected.checkpoint_sha256,
            device=target_device,
        )
        return LoadedCandidatePolicy(
            policy=policy,
            checkpoint_id=str(payload["checkpoint_id"]),
            checkpoint_sha256=expected.checkpoint_sha256,
            training_manifest_sha256=training_sha,
        )

    shard = evaluate_shard(
        task8_rows=rows,
        scenario_manifest_sha256=contract["manifest_sha256"],
        checkpoint_manifest_sha256=training_sha,
        bc_model=bc,
        bc_checkpoint_sha256=bc_sha,
        checkpoints=specs,
        policy_loader=loader,
        device=device,
        shard_index=int(job["shard_index"]),
        shard_count=int(job["shard_count"]),
    )
    output = root / str(job["output_relpath"])
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    try:
        (partial / "shard.json").write_text(
            json.dumps(asdict(shard), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        control_rows = []
        for row in shard.rows:
            control_rows.append(
                {
                    **row,
                    "row_index": row["task8_row_index"],
                    "variant_id": row["variant"],
                    "shard_index": shard.shard_index,
                    "manifest_sha256": contract["manifest_sha256"],
                    "checkpoint_set_sha256": contract["checkpoint_set_sha256"],
                }
            )
        fields = sorted({name for row in control_rows for name in row})
        _write_tsv(partial / "episodes.tsv", control_rows, fields)
        (partial / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output)
        return {
            "passed": True,
            "shard_index": shard.shard_index,
            "scenario_count": len(shard.rows) // 7,
            "episode_rows": len(shard.rows),
        }
    except Exception:
        if partial.exists():
            (partial / "FAILED").write_text("FAILED\n", encoding="utf-8")
        raise


def merge_eval_job(plan_path: str | Path, input_root: str | Path, output_dir: str | Path) -> dict:
    _, plan, root, contract, _ = _load_eval_plan(plan_path)
    specs, _, training_sha = _checkpoint_specs(
        root,
        contract,
        str(plan["parent_plan_sha256"]),
        eval_kind=str(plan["kind"]),
        expected_iteration=int(plan["config"]["checkpoint_iteration"]),
    )
    manifest = root / str(contract["manifest_relpath"])
    task8_rows = read_task8_development(manifest, contract["manifest_sha256"])
    source = Path(input_root)
    shards = []
    for index in range(4):
        host = "local" if index == 0 else "remote"
        path = source / f"hosts/{host}/outputs/eval/shard{index}/shard.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        shards.append(
            EvaluationShard(
                shard_index=int(payload["shard_index"]),
                shard_count=int(payload["shard_count"]),
                scenario_manifest_sha256=payload["scenario_manifest_sha256"],
                checkpoint_manifest_sha256=payload["checkpoint_manifest_sha256"],
                bc_checkpoint_sha256=payload["bc_checkpoint_sha256"],
                checkpoint_sha256_by_variant=payload["checkpoint_sha256_by_variant"],
                rows=tuple(payload["rows"]),
                schema=payload["schema"],
            )
        )
    bc_sha = _sha256_file(root / "repo/pretrained/end2race.pth")
    rows, summary = merge_evaluation_shards(
        shards=shards,
        task8_rows=task8_rows,
        scenario_manifest_sha256=contract["manifest_sha256"],
        checkpoint_manifest_sha256=training_sha,
        bc_checkpoint_sha256=bc_sha,
        checkpoints=specs,
    )
    output = Path(output_dir)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    partial.mkdir(parents=True)
    try:
        fields = sorted({name for row in rows for name in row})
        _write_tsv(partial / "episodes.tsv", rows, fields)
        (partial / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (partial / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        os.replace(partial, output)
        return {"passed": True, **summary}
    except Exception:
        if partial.exists():
            (partial / "FAILED").write_text("FAILED\n", encoding="utf-8")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="B+ v2.2 structural preflight")
    sub = parser.add_subparsers(dest="command", required=True)
    capabilities = sub.add_parser("capabilities")
    capabilities.add_argument("--json", action="store_true")
    ppo_baseline = sub.add_parser("ppo-baseline-preflight")
    ppo_baseline.add_argument("--run-plan", required=True)
    ppo_baseline.add_argument("--output", required=True)
    ppo_baseline.add_argument("--host-id", choices=("local", "remote"), required=True)
    ppo_baseline.add_argument("--gpu-uuid", required=True)
    ppo_baseline.add_argument("--shard-index", type=int, required=True)
    ppo_baseline.add_argument(
        "--shard-count", type=int, default=BASELINE_SHARD_COUNT
    )
    ppo_baseline.add_argument("--device", default="cuda:0")
    ppo_plumbing = sub.add_parser("ppo-plumbing-smoke")
    ppo_plumbing.add_argument("--run-plan", required=True)
    ppo_plumbing.add_argument("--output", required=True)
    ppo_plumbing.add_argument("--device", default="cuda:0")
    ppo_pilot = sub.add_parser("ppo-pilot")
    ppo_pilot.add_argument("--run-plan", required=True)
    ppo_pilot.add_argument("--job-id")
    ppo_pilot.add_argument("--validate-plan-only", action="store_true")
    ppo_pilot.add_argument("--resume", action="store_true")
    ppo_pilot.add_argument("--device", default="cuda:0")
    ppo_evaluate = sub.add_parser("ppo-evaluate")
    ppo_evaluate.add_argument("--run-plan", required=True)
    ppo_evaluate.add_argument("--job-id")
    ppo_evaluate.add_argument("--validate-plan-only", action="store_true")
    ppo_evaluate.add_argument("--device", default="cuda:0")
    ppo_merge = sub.add_parser("ppo-merge-eval")
    ppo_merge.add_argument("--run-plan", required=True)
    ppo_merge.add_argument("--input-root", required=True)
    ppo_merge.add_argument("--output-dir", required=True)
    create = sub.add_parser("source-preflight")
    create.add_argument("--output-dir", required=True)
    create.add_argument("--created-at", required=True)
    create.add_argument("--repo-root", default=".")
    validate = sub.add_parser("validate-source-preflight")
    validate.add_argument("release_dir")
    validate.add_argument("--repo-root", default=".")
    identity = sub.add_parser("zero-identity")
    identity.add_argument("--repo-root", default=".")
    identity.add_argument("--source-preflight-dir", required=True)
    identity.add_argument("--output-dir", required=True)
    identity.add_argument("--created-at", required=True)
    identity.add_argument("--device", default="cuda:0")
    identity.add_argument("--sidecar-release-dir")
    validate_identity = sub.add_parser("validate-zero-identity")
    validate_identity.add_argument("release_dir")
    registry_plan = sub.add_parser("sidecar-registry-plan")
    registry_plan.add_argument("--repo-root", default=".")
    registry_plan.add_argument("--source-preflight-dir", required=True)
    registry_plan.add_argument("--output-dir", required=True)
    registry_plan.add_argument("--created-at", required=True)
    validate_registry = sub.add_parser("validate-sidecar-registry-plan")
    validate_registry.add_argument("release_dir")
    validate_registry.add_argument("--repo-root", default=".")
    validate_registry.add_argument("--check-live", action="store_true")
    sidecar = sub.add_parser("sidecar-fit")
    sidecar.add_argument("--repo-root", default=".")
    sidecar.add_argument("--source-preflight-dir", required=True)
    sidecar.add_argument("--registry-plan-dir", required=True)
    sidecar.add_argument("--output-dir", required=True)
    sidecar.add_argument("--created-at", required=True)
    sidecar.add_argument("--device", default="cuda:0")
    validate_sidecar = sub.add_parser("validate-sidecar")
    validate_sidecar.add_argument("release_dir")
    validate_sidecar.add_argument("--repo-root", default=".")
    validate_sidecar.add_argument("--dataset-dir")
    validate_sidecar.add_argument("--split-dir")
    validate_sidecar.add_argument("--signals-dir")
    validate_sidecar.add_argument("--device")
    validate_sidecar.add_argument("--require-live-registry", action="store_true")
    warmstart_manifest = sub.add_parser("warmstart-manifest")
    warmstart_manifest.add_argument("--repo-root", default=".")
    warmstart_manifest.add_argument("--source-preflight-dir", required=True)
    warmstart_manifest.add_argument("--output-dir", required=True)
    warmstart_manifest.add_argument("--created-at", required=True)
    validate_manifest = sub.add_parser("validate-warmstart-manifest")
    validate_manifest.add_argument("release_dir")
    validate_manifest.add_argument("--repo-root", default=".")
    validate_manifest.add_argument("--check-live-registry", action="store_true")
    warmstart = sub.add_parser("warmstart-smoke")
    warmstart.add_argument("--repo-root", default=".")
    warmstart.add_argument("--source-preflight-dir", required=True)
    warmstart.add_argument("--manifest-dir", required=True)
    warmstart.add_argument("--output-dir", required=True)
    warmstart.add_argument("--created-at", required=True)
    warmstart.add_argument("--device", default="cuda:0")
    validate_warmstart = sub.add_parser("validate-warmstart")
    validate_warmstart.add_argument("release_dir")
    validate_warmstart.add_argument("--repo-root", default=".")
    validate_warmstart.add_argument("--device")
    validate_warmstart.add_argument("--require-live-registry", action="store_true")
    manifests = sub.add_parser("task8-manifests")
    manifests.add_argument("--repo-root", default=".")
    manifests.add_argument("--source-preflight-dir", required=True)
    manifests.add_argument("--output-dir", required=True)
    manifests.add_argument("--created-at", required=True)
    validate_manifests = sub.add_parser("validate-task8-manifests")
    validate_manifests.add_argument("release_dir")
    validate_manifests.add_argument("--repo-root", default=".")
    task9 = sub.add_parser("task9-checkpoint-preflight")
    task9.add_argument("--repo-root", default=".")
    task9.add_argument("--source-preflight-dir", required=True)
    task9.add_argument("--manifest-release-dir", required=True)
    task9.add_argument("--output-dir", required=True)
    task9.add_argument("--created-at", required=True)
    task9.add_argument("--device", default="cuda:0")
    validate_task9 = sub.add_parser("validate-task9-checkpoint-preflight")
    validate_task9.add_argument("release_dir")
    task10 = sub.add_parser("task10-warmstart")
    task10.add_argument("--repo-root", default=".")
    task10.add_argument("--source-preflight-dir", required=True)
    task10.add_argument("--manifest-release-dir", required=True)
    task10.add_argument("--output-dir", required=True)
    task10.add_argument("--created-at", required=True)
    task10.add_argument("--device", default="cuda:0")
    validate_task10 = sub.add_parser("validate-task10-warmstart")
    validate_task10.add_argument("release_dir")

    hierarchical_identity = sub.add_parser("hierarchical-zero-identity")
    hierarchical_identity.add_argument("--repo-root", default=".")
    hierarchical_identity.add_argument("--source-preflight-dir", required=True)
    hierarchical_identity.add_argument("--sidecar-release-dir", required=True)
    hierarchical_identity.add_argument("--output-dir", required=True)
    hierarchical_identity.add_argument("--created-at", required=True)
    hierarchical_identity.add_argument("--device", default="cuda:0")
    validate_hierarchical_identity_parser = sub.add_parser(
        "validate-hierarchical-zero-identity"
    )
    validate_hierarchical_identity_parser.add_argument("release_dir")
    validate_hierarchical_identity_parser.add_argument("--repo-root", default=".")

    hierarchical_manifest = sub.add_parser("hierarchical-warmstart-manifest")
    hierarchical_manifest.add_argument("--repo-root", default=".")
    hierarchical_manifest.add_argument("--source-preflight-dir", required=True)
    hierarchical_manifest.add_argument(
        "--hierarchical-identity-release-dir", required=True
    )
    hierarchical_manifest.add_argument("--output-dir", required=True)
    hierarchical_manifest.add_argument("--created-at", required=True)
    validate_hierarchical_manifest = sub.add_parser(
        "validate-hierarchical-warmstart-manifest"
    )
    validate_hierarchical_manifest.add_argument("release_dir")
    validate_hierarchical_manifest.add_argument("--repo-root", default=".")

    hierarchical_warmstart = sub.add_parser("hierarchical-warmstart")
    hierarchical_warmstart.add_argument("--repo-root", default=".")
    hierarchical_warmstart.add_argument("--source-preflight-dir", required=True)
    hierarchical_warmstart.add_argument(
        "--hierarchical-identity-release-dir", required=True
    )
    hierarchical_warmstart.add_argument("--manifest-dir", required=True)
    hierarchical_warmstart.add_argument("--output-dir", required=True)
    hierarchical_warmstart.add_argument("--created-at", required=True)
    hierarchical_warmstart.add_argument("--device", default="cuda:0")
    validate_hierarchical_warmstart = sub.add_parser(
        "validate-hierarchical-warmstart"
    )
    validate_hierarchical_warmstart.add_argument("release_dir")
    validate_hierarchical_warmstart.add_argument("--repo-root", default=".")
    validate_hierarchical_warmstart.add_argument("--device")

    hierarchical_task9 = sub.add_parser("hierarchical-task9")
    hierarchical_task9.add_argument("--repo-root", default=".")
    hierarchical_task9.add_argument("--source-preflight-dir", required=True)
    hierarchical_task9.add_argument("--manifest-release-dir", required=True)
    hierarchical_task9.add_argument("--warmstart-release-dir", required=True)
    hierarchical_task9.add_argument(
        "--warmstart-output-manifest-sha256", required=True
    )
    hierarchical_task9.add_argument("--output-dir", required=True)
    hierarchical_task9.add_argument("--created-at", required=True)
    hierarchical_task9.add_argument("--device", default="cuda:0")
    validate_hierarchical_task9 = sub.add_parser("validate-hierarchical-task9")
    validate_hierarchical_task9.add_argument("release_dir")

    hierarchical_task10 = sub.add_parser("hierarchical-task10")
    hierarchical_task10.add_argument("--repo-root", default=".")
    hierarchical_task10.add_argument("--source-preflight-dir", required=True)
    hierarchical_task10.add_argument("--manifest-release-dir", required=True)
    hierarchical_task10.add_argument("--warmstart-release-dir", required=True)
    hierarchical_task10.add_argument(
        "--warmstart-output-manifest-sha256", required=True
    )
    hierarchical_task10.add_argument("--output-dir", required=True)
    hierarchical_task10.add_argument("--created-at", required=True)
    hierarchical_task10.add_argument("--device", default="cuda:0")
    validate_hierarchical_task10 = sub.add_parser("validate-hierarchical-task10")
    validate_hierarchical_task10.add_argument("release_dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "capabilities":
        result = {
            "schema": B2_CAPABILITIES_SCHEMA,
            "commands": sorted(B2_COMMANDS),
        }
    elif args.command == "ppo-baseline-preflight":
        result = run_bc_baseline_shard(
            args.run_plan,
            args.output,
            host_id=args.host_id,
            gpu_uuid=args.gpu_uuid,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            device_name=args.device,
        )
    elif args.command == "ppo-plumbing-smoke":
        result = run_plumbing_smoke_release(
            args.run_plan, args.output, device_name=args.device
        )
    elif args.command == "ppo-pilot":
        if args.validate_plan_only:
            validated = validate_pilot_plan(args.run_plan)
            result = {
                "passed": True,
                "job_count": len(validated["plan"]["jobs"]),
                "plan_sha256": validated["plan"]["plan_sha256"],
            }
        else:
            if not args.job_id:
                raise ValueError("ppo-pilot execution requires --job-id")
            result = run_pilot_job(
                args.run_plan,
                args.job_id,
                device_name=args.device,
                resume=args.resume,
            )
    elif args.command == "ppo-evaluate":
        if args.validate_plan_only:
            result = validate_eval_plan(args.run_plan)
        else:
            if not args.job_id:
                raise ValueError("ppo-evaluate execution requires --job-id")
            result = run_eval_job(args.run_plan, args.job_id, device_name=args.device)
    elif args.command == "ppo-merge-eval":
        result = merge_eval_job(args.run_plan, args.input_root, args.output_dir)
    elif args.command == "source-preflight":
        result = create_source_preflight(args.output_dir, args.created_at, args.repo_root)
    elif args.command == "validate-source-preflight":
        result = validate_source_preflight(args.release_dir, args.repo_root)
    elif args.command == "zero-identity":
        result = run_zero_identity(
            args.repo_root,
            args.source_preflight_dir,
            args.output_dir,
            args.created_at,
            args.device,
            args.sidecar_release_dir,
        )
    elif args.command == "sidecar-registry-plan":
        result = create_registry_plan(
            args.repo_root,
            args.source_preflight_dir,
            args.output_dir,
            args.created_at,
        )
    elif args.command == "validate-sidecar-registry-plan":
        result = validate_registry_plan(
            args.release_dir, args.repo_root, check_live=args.check_live
        )
    elif args.command == "sidecar-fit":
        result = run_sidecar_initialization(
            args.repo_root,
            args.source_preflight_dir,
            args.registry_plan_dir,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-sidecar":
        result = validate_sidecar_release(
            args.release_dir,
            args.repo_root,
            dataset_dir=args.dataset_dir,
            split_dir=args.split_dir,
            signals_dir=args.signals_dir,
            device_name=args.device,
            require_live_registry=args.require_live_registry,
        )
    elif args.command == "warmstart-manifest":
        result = create_warmstart_manifest(
            args.repo_root,
            args.source_preflight_dir,
            args.output_dir,
            args.created_at,
        )
    elif args.command == "validate-warmstart-manifest":
        result = validate_warmstart_manifest(
            args.release_dir,
            args.repo_root,
            check_live_registry=args.check_live_registry,
        )
    elif args.command == "warmstart-smoke":
        result = run_warmstart_smoke(
            args.repo_root,
            args.source_preflight_dir,
            args.manifest_dir,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-warmstart":
        result = validate_warmstart_release(
            args.release_dir,
            args.repo_root,
            device_name=args.device,
            require_live_registry=args.require_live_registry,
        )
    elif args.command == "task8-manifests":
        result = create_manifest_release(
            args.repo_root,
            args.source_preflight_dir,
            args.output_dir,
            args.created_at,
        )
    elif args.command == "validate-task8-manifests":
        result = validate_manifest_release(args.release_dir, args.repo_root)
    elif args.command == "task9-checkpoint-preflight":
        result = run_checkpoint_preflight(
            args.repo_root,
            args.source_preflight_dir,
            args.manifest_release_dir,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-task9-checkpoint-preflight":
        result = validate_checkpoint_preflight(args.release_dir)
    elif args.command == "task10-warmstart":
        result = run_closed_loop_warmstart(
            args.repo_root,
            args.source_preflight_dir,
            args.manifest_release_dir,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-task10-warmstart":
        result = validate_closed_loop_release(args.release_dir)
    elif args.command == "hierarchical-zero-identity":
        result = run_hierarchical_identity(
            args.repo_root,
            args.source_preflight_dir,
            args.sidecar_release_dir,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-hierarchical-zero-identity":
        result = validate_hierarchical_identity(
            args.release_dir, repo_root=args.repo_root
        )
    elif args.command == "hierarchical-warmstart-manifest":
        result = create_hierarchical_warmstart_manifest(
            args.repo_root,
            args.source_preflight_dir,
            args.hierarchical_identity_release_dir,
            args.output_dir,
            args.created_at,
        )
    elif args.command == "validate-hierarchical-warmstart-manifest":
        result = validate_hierarchical_warmstart_manifest(
            args.release_dir, args.repo_root
        )
    elif args.command == "hierarchical-warmstart":
        result = run_hierarchical_warmstart(
            args.repo_root,
            args.source_preflight_dir,
            args.hierarchical_identity_release_dir,
            args.manifest_dir,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-hierarchical-warmstart":
        result = validate_hierarchical_warmstart_release(
            args.release_dir, args.repo_root, device_name=args.device
        )
    elif args.command == "hierarchical-task9":
        result = run_hierarchical_checkpoint_preflight(
            args.repo_root,
            args.source_preflight_dir,
            args.manifest_release_dir,
            args.warmstart_release_dir,
            args.warmstart_output_manifest_sha256,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-hierarchical-task9":
        result = validate_hierarchical_checkpoint_preflight(args.release_dir)
    elif args.command == "hierarchical-task10":
        result = run_hierarchical_closed_loop(
            args.repo_root,
            args.source_preflight_dir,
            args.manifest_release_dir,
            args.warmstart_release_dir,
            args.warmstart_output_manifest_sha256,
            args.output_dir,
            args.created_at,
            args.device,
        )
    elif args.command == "validate-hierarchical-task10":
        result = validate_hierarchical_closed_loop(args.release_dir)
    elif args.command == "validate-zero-identity":
        result = validate_zero_identity(args.release_dir)
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("passed") is not True and args.command not in {"capabilities"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
