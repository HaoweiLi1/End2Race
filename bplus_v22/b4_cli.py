"""Thin file/control-plane adapters for the B4 learner and paired evaluator."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from bplus_v22.b4_direct import B4_POLICY_SCHEMA, load_strict_plain_actor
from bplus_v22.b4_eval import (
    B4CheckpointSpec,
    B4EvaluationShard,
    B4_EVAL_SHARD_SCHEMA,
    B4_EXPECTED_RESULTS,
    B4_VARIANT_COUNT,
    evaluate_shard,
    file_sha256,
    merge_shards,
    validate_checkpoint_specs,
)
from bplus_v22.b4_runner import (
    B4_RUN_PLAN_SCHEMA,
    run_b4_plumbing_smoke,
    validate_b4_pilot_plan,
)
from bplus_v22.ppo_eval import (
    BASELINE_SHARD_COUNT,
    baseline_json_bytes,
    evaluate_bc_baseline_shard,
    read_task8_development,
)


B4_EVAL_KIND = "b4_eval"
B4_EVAL_CONFIG = {
    "policy_contract": B4_POLICY_SCHEMA,
    "checkpoint_iterations": [10, 20, 30],
    "seeds": [0, 1],
    "expected_scenario_count": 288,
    "expected_variant_count": B4_VARIANT_COUNT,
    "expected_episode_rows": B4_EXPECTED_RESULTS,
    "per_seed_overtake_gate": 132,
    "per_seed_collision_feasibility": 24,
    "per_seed_collision_product_target": 16,
    "pooled_collision_product_target": 33,
    "deterministic_speed_projection_required": 0,
    "same_iteration_seed_pair_required": True,
}


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256_file(path: str | Path) -> str:
    return file_sha256(path)


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({name for row in rows for name in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_b4_baseline_shard(
    plan_path: str | Path,
    output_path: str | Path,
    *,
    host_id: str,
    gpu_uuid: str,
    shard_index: int,
    shard_count: int = BASELINE_SHARD_COUNT,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    validated = validate_b4_pilot_plan(plan_path)
    plan = validated["plan"]
    paths = validated["paths"]
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("B4 BC baseline preflight requested unavailable CUDA")
    manifest = paths["task8"] / "development_scenarios.tsv"
    manifest_sha = _sha256_file(manifest)
    rows = read_task8_development(manifest, manifest_sha)
    bc_sha = _sha256_file(paths["bc"])
    bc = load_strict_plain_actor(paths["bc"], device)
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
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
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


def run_b4_plumbing_release(
    plan_path: str | Path,
    output_path: str | Path,
    *,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    result = run_b4_plumbing_smoke(plan_path, device_name=device_name)
    output = Path(output_path)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, output)
    return {"passed": True, "map_count": len(result["map_reports"])}


def _load_b4_eval_plan(
    plan_path: str | Path, job_id: str | None = None
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], dict[str, Any] | None]:
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != B4_RUN_PLAN_SCHEMA or plan.get("kind") != B4_EVAL_KIND:
        raise ValueError("B4 evaluator requires one b4_eval RunPlan")
    observed = plan.get("plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    if observed != hashlib.sha256(_canonical_json(unsigned)).hexdigest():
        raise ValueError("B4 EvalPlan digest mismatch")
    if plan.get("config") != B4_EVAL_CONFIG:
        raise ValueError("B4 EvalPlan frozen config drift")
    parent = plan.get("parent_plan_sha256")
    if not isinstance(parent, str) or len(parent) != 64:
        raise ValueError("B4 EvalPlan parent digest is invalid")
    root = path.parent.parent
    if Path.cwd().resolve() != (root / "repo").resolve():
        raise ValueError("B4 evaluator must execute from staged repository")
    contract = plan.get("evaluation_contract")
    if not isinstance(contract, dict):
        raise ValueError("B4 EvalPlan lacks evaluation contract")
    checkpoint_set = contract.get("checkpoint_set")
    if (
        contract.get("expected_scenario_count") != 288
        or contract.get("expected_variant_count") != B4_VARIANT_COUNT
        or contract.get("expected_episode_rows") != B4_EXPECTED_RESULTS
        or contract.get("shard_count") != 4
        or not isinstance(checkpoint_set, list)
        or len(checkpoint_set) != 6
    ):
        raise ValueError("B4 EvalPlan Cartesian contract drift")
    expected_set_sha = hashlib.sha256(_canonical_json(checkpoint_set)).hexdigest()
    if contract.get("checkpoint_set_sha256") != expected_set_sha:
        raise ValueError("B4 EvalPlan checkpoint-set digest mismatch")
    manifest = root / str(contract["manifest_relpath"])
    if not manifest.is_file() or _sha256_file(manifest) != contract["manifest_sha256"]:
        raise ValueError("B4 EvalPlan development manifest drift")
    jobs = {str(row["job_id"]): row for row in plan.get("jobs", [])}
    if len(jobs) != 4:
        raise ValueError("B4 EvalPlan shard job inventory drift")
    selected = None
    if job_id is not None:
        if job_id not in jobs:
            raise ValueError(f"unknown B4 evaluation job: {job_id}")
        selected = dict(jobs[job_id])
        if selected.get("kind") != "b4_evaluation_shard":
            raise ValueError("B4 evaluation job kind drift")
        output = root / str(selected["output_relpath"])
        if output.exists() or output.with_name(output.name + ".partial").exists():
            raise FileExistsError(output)
    return path, plan, root, contract, selected


def _b4_checkpoint_specs(
    root: Path,
    contract: Mapping[str, Any],
    parent_plan_sha256: str,
) -> tuple[tuple[B4CheckpointSpec, ...], str]:
    specs = []
    training_manifest_sha = str(contract.get("training_manifest_sha256", ""))
    for row in contract["checkpoint_set"]:
        path = root / str(row["relpath"])
        if not path.is_file() or _sha256_file(path) != row.get("sha256"):
            raise ValueError("B4 EvalPlan actor snapshot file drift")
        # This is deliberately a plain strict loader, not an envelope or
        # residual-aware loader.
        load_strict_plain_actor(path, "cpu")
        spec = B4CheckpointSpec(
            seed=int(row["seed"]),
            iteration=int(row["iteration"]),
            checkpoint_path=str(path),
            checkpoint_sha256=str(row["sha256"]),
            training_manifest_sha256=training_manifest_sha,
            training_run_plan_sha256=parent_plan_sha256,
        )
        specs.append(spec)
    checkpoint_set_sha = str(contract["checkpoint_set_sha256"])
    return validate_checkpoint_specs(specs, checkpoint_set_sha), checkpoint_set_sha


def validate_b4_eval_plan(plan_path: str | Path) -> dict[str, Any]:
    _, plan, root, contract, _ = _load_b4_eval_plan(plan_path)
    specs, _ = _b4_checkpoint_specs(root, contract, str(plan["parent_plan_sha256"]))
    return {
        "passed": True,
        "checkpoint_count": len(specs),
        "scenario_count": contract["expected_scenario_count"],
        "episode_rows": contract["expected_episode_rows"],
    }


def run_b4_eval_job(
    plan_path: str | Path,
    job_id: str,
    *,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    _, plan, root, contract, job = _load_b4_eval_plan(plan_path, job_id)
    specs, checkpoint_set_sha = _b4_checkpoint_specs(
        root, contract, str(plan["parent_plan_sha256"])
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("B4 evaluation requested unavailable CUDA")
    manifest = root / str(contract["manifest_relpath"])
    rows = read_task8_development(manifest, contract["manifest_sha256"])
    bc_path = root / "repo/pretrained/end2race.pth"
    bc_sha = _sha256_file(bc_path)
    shard = evaluate_shard(
        task8_rows=rows,
        scenario_manifest_sha256=contract["manifest_sha256"],
        checkpoint_manifest_sha256=checkpoint_set_sha,
        bc_checkpoint_path=bc_path,
        bc_checkpoint_sha256=bc_sha,
        checkpoints=specs,
        device=device,
        shard_index=int(job["shard_index"]),
        shard_count=int(job["shard_count"]),
    )
    output = root / str(job["output_relpath"])
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    try:
        (partial / "shard.json").write_text(
            json.dumps(shard.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        control_rows = [
            {
                **row,
                "row_index": row["task8_row_index"],
                "variant_id": row["variant"],
                "shard_index": shard.shard_index,
                "manifest_sha256": contract["manifest_sha256"],
                "checkpoint_set_sha256": checkpoint_set_sha,
            }
            for row in shard.rows
        ]
        _write_tsv(partial / "episodes.tsv", control_rows)
        (partial / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output)
        return {
            "passed": True,
            "shard_index": shard.shard_index,
            "scenario_count": len(shard.rows) // B4_VARIANT_COUNT,
            "episode_rows": len(shard.rows),
        }
    except Exception:
        if partial.exists():
            (partial / "FAILED").write_text("FAILED\n", encoding="utf-8")
        raise


def merge_b4_eval_job(
    plan_path: str | Path,
    input_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    _, plan, root, contract, _ = _load_b4_eval_plan(plan_path)
    specs, checkpoint_set_sha = _b4_checkpoint_specs(
        root, contract, str(plan["parent_plan_sha256"])
    )
    manifest = root / str(contract["manifest_relpath"])
    task8_rows = read_task8_development(manifest, contract["manifest_sha256"])
    source = Path(input_root)
    shards = []
    for index in range(4):
        host = "local" if index == 0 else "remote"
        path = source / f"hosts/{host}/outputs/eval/shard{index}/shard.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.pop("schema", None) != B4_EVAL_SHARD_SCHEMA:
            raise ValueError("B4 collected evaluation shard schema mismatch")
        shards.append(
            B4EvaluationShard(
                shard_index=int(payload["shard_index"]),
                shard_count=int(payload["shard_count"]),
                scenario_manifest_sha256=payload["scenario_manifest_sha256"],
                checkpoint_manifest_sha256=payload["checkpoint_manifest_sha256"],
                bc_checkpoint_sha256=payload["bc_checkpoint_sha256"],
                checkpoint_sha256_by_variant=payload["checkpoint_sha256_by_variant"],
                rows=tuple(payload["rows"]),
            )
        )
    bc_sha = _sha256_file(root / "repo/pretrained/end2race.pth")
    rows, summary = merge_shards(
        shards=shards,
        task8_rows=task8_rows,
        scenario_manifest_sha256=contract["manifest_sha256"],
        checkpoint_manifest_sha256=checkpoint_set_sha,
        bc_checkpoint_sha256=bc_sha,
        checkpoints=specs,
    )
    output = Path(output_dir)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    partial.mkdir(parents=True)
    try:
        _write_tsv(partial / "episodes.tsv", rows)
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
