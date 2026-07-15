"""Thin file/control-plane adapters for the B5-A learner."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from bplus_v22.b4_direct import load_strict_plain_actor
from bplus_v22.b5_runner import run_b5_plumbing_smoke, validate_b5_pilot_plan
from bplus_v22.ppo_eval import (
    BASELINE_SHARD_COUNT,
    baseline_json_bytes,
    evaluate_bc_baseline_shard,
    read_task8_development,
)


def _sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_b5_baseline_shard(
    plan_path: str | Path,
    output_path: str | Path,
    *,
    host_id: str,
    gpu_uuid: str,
    shard_index: int,
    shard_count: int = BASELINE_SHARD_COUNT,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    validated = validate_b5_pilot_plan(plan_path)
    plan = validated["plan"]
    paths = validated["paths"]
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("B5 BC baseline preflight requested unavailable CUDA")
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


def run_b5_plumbing_release(
    plan_path: str | Path,
    output_path: str | Path,
    *,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    result = run_b5_plumbing_smoke(plan_path, device_name=device_name)
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
    return {"passed": True, "reference_sha256": result["reference_sha256"]}
