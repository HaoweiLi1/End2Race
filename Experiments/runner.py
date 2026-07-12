#!/usr/bin/env python3
"""Immutable control plane for End2Race experiments.

New B2 work never executes from either host's mutable checkout.  A clean,
committed source tree is archived once, explicit runtime inputs are bundled,
and both hosts execute from an isolated run root:

    /home/haowei/end2race_runs/<run_id>

The public workflow is deliberately split into reviewable phases::

    ./run.sh plan ...
    ./run.sh show PLAN
    ./run.sh stage PLAN --all-hosts [--dry-run]
    ./run.sh baseline-preflight PLAN [--dry-run]
    ./run.sh preflight PLAN --all-hosts [--dry-run]
    ./run.sh execute PLAN --all-hosts [--dry-run]
    ./run.sh resume PLAN --host <local|remote> [--dry-run]
    ./run.sh status PLAN --all-hosts [--dry-run]
    ./run.sh collect PLAN [--dry-run]

After training, ``plan-eval`` freezes the six iteration-20 checkpoints.  Eval
shard 0 runs locally; shards 1--3 run sequentially under one remote GPU lock.
``merge-eval`` refuses incomplete or non-Cartesian shard output.

There is intentionally no generic ``run <job>`` or ``split <job>`` path.  The
old B2 placeholders referenced missing code and could run in a stale remote
worktree.  B1 is retained only as a non-executable legacy display entry.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_HOST = "haowei@192.168.2.127"
ISOLATED_BASE = PurePosixPath("/home/haowei/end2race_runs")
PINNED_PYTHON = "/home/haowei/miniconda3/envs/end2race/bin/python"
LOCAL_DISPLAY = ":0"
REMOTE_DISPLAY = ":1"
LOCAL_GPU_NAME = "NVIDIA GeForce RTX 3080 Laptop GPU"
REMOTE_GPU_NAME = "NVIDIA GeForce RTX 4080 SUPER"
PLAN_SCHEMA = "end2race-b2-run-plan-1"
CAPABILITIES_SCHEMA = "bplus-v22-cli-capabilities-1"
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,95}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ARMS = ("BC_FROZEN", "SIDECAR_FROZEN", "SIDECAR_FINETUNE")
SEEDS = (0, 1)
SHARD_COUNT = 4
CANONICAL_BC_SHA256 = (
    "b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4"
)
CANONICAL_SIDECAR_SHA256 = (
    "d172d0527bc03be1e0d814204a6a5a53f6de18ca20a5957f09a32c96b5cf4dab"
)
SIDECAR_RELEASE = Path(
    "Experiments/B1_route_r2_scaffold/artifacts/sidecar_init_20260712_080012"
)
TASK8_RELEASE = Path(
    "Experiments/B1_route_r2_scaffold/artifacts/task8_manifests_20260712_113241"
)
D2_METADATA = Path(
    "Experiments/A3_d2_representation/artifacts/"
    "non_test_full_20260711_175713/episode_metadata.tsv"
)
D2_METADATA_SHA256 = "468d8be50aecad19f89fbf2c35dc421acb4244a61f957f77dcfff1acd227eda3"
REQUIRED_TRAIN_CLI = (
    "ppo-baseline-preflight",
    "ppo-pilot",
    "ppo-evaluate",
    "ppo-merge-eval",
)
MODULE_PATH_CONTRACT = (
    "bplus_v22",
    "model",
    "f110_gym",
    "latticeplanner.lattice_planner",
)


class RunnerError(RuntimeError):
    """Fail-closed control-plane error."""


@dataclass(frozen=True)
class InputEntry:
    role: str
    relpath: str
    sha256: str
    size: int


@dataclass(frozen=True)
class HostSpec:
    host_id: str
    kind: str
    stage_root: str
    python: str
    display: str
    gpu_uuid: str
    gpu_name: str
    ssh_host: str | None
    expected_environment: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    kind: str
    host_id: str
    queue_id: str
    argv: tuple[str, ...]
    output_relpath: str
    numba_cache_relpath: str
    arm: str | None = None
    seed: int | None = None
    shard_index: int | None = None
    shard_count: int | None = None
    gpu_exclusive: bool = True
    shardable: bool = False


@dataclass(frozen=True)
class RunPlan:
    schema: str
    run_id: str
    kind: str
    created_at: str
    source_commit: str
    source_tree: str
    source_archive_path: str
    source_archive_sha256: str
    source_archive_size: int
    inputs_archive_path: str
    inputs_archive_sha256: str
    inputs_archive_size: int
    source_inputs: tuple[InputEntry, ...]
    inputs: tuple[InputEntry, ...]
    hosts: tuple[HostSpec, ...]
    jobs: tuple[JobSpec, ...]
    queues: dict[str, tuple[str, ...]]
    required_cli: tuple[str, ...]
    module_path_contract: tuple[str, ...]
    config: dict[str, Any]
    collection_root: str
    parent_plan_sha256: str | None = None
    evaluation_contract: dict[str, Any] | None = None
    plan_sha256: str = ""


LEGACY_SHOW_ONLY: dict[str, str] = {
    "legacy-b1-source-preflight": (
        "NON-EXECUTABLE legacy template: python -m bplus_v22.cli "
        "source-preflight --repo-root . --output-dir <explicit-new-dir> "
        "--created-at <explicit-iso-time>"
    )
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan_payload(plan: RunPlan) -> dict[str, Any]:
    payload = asdict(plan)
    payload.pop("plan_sha256", None)
    return payload


def _seal_plan(plan: RunPlan) -> RunPlan:
    return replace(plan, plan_sha256=_sha256_bytes(_canonical_json(_plan_payload(plan))))


def _verify_plan(plan: RunPlan) -> None:
    if plan.schema != PLAN_SCHEMA:
        raise RunnerError(f"unsupported plan schema: {plan.schema}")
    _validate_run_id(plan.run_id)
    expected = _sha256_bytes(_canonical_json(_plan_payload(plan)))
    if not SHA256_RE.fullmatch(plan.plan_sha256) or plan.plan_sha256 != expected:
        raise RunnerError("run plan digest mismatch")
    if not SHA256_RE.fullmatch(plan.source_archive_sha256):
        raise RunnerError("invalid source archive digest")
    if not SHA256_RE.fullmatch(plan.inputs_archive_sha256):
        raise RunnerError("invalid input archive digest")
    if not re.fullmatch(r"[0-9a-f]{40}", plan.source_commit):
        raise RunnerError("source commit must be one resolved SHA-1 commit id")
    if not re.fullmatch(r"[0-9a-f]{40}", plan.source_tree):
        raise RunnerError("source tree must be one resolved SHA-1 tree id")
    if plan.source_archive_size <= 0 or plan.inputs_archive_size <= 0:
        raise RunnerError("control archive sizes must be positive")
    if not Path(plan.source_archive_path).is_absolute() or not Path(plan.inputs_archive_path).is_absolute():
        raise RunnerError("controller archive paths must be absolute")
    if not Path(plan.collection_root).is_absolute():
        raise RunnerError("collection root must be absolute")
    for entry in (*plan.source_inputs, *plan.inputs):
        _safe_relative(entry.relpath)
        if not SHA256_RE.fullmatch(entry.sha256) or entry.size < 0:
            raise RunnerError(f"invalid input entry: {entry.relpath}")
    host_ids = {host.host_id for host in plan.hosts}
    if host_ids != {"local", "remote"}:
        raise RunnerError(f"plan must contain local and remote hosts: {host_ids}")
    jobs = {job.job_id: job for job in plan.jobs}
    if len(jobs) != len(plan.jobs):
        raise RunnerError("duplicate job id")
    for queue, ids in plan.queues.items():
        if not ids:
            raise RunnerError(f"empty queue: {queue}")
        for job_id in ids:
            if job_id not in jobs:
                raise RunnerError(f"queue references unknown job: {job_id}")
            if jobs[job_id].queue_id != queue:
                raise RunnerError(f"queue drift for job: {job_id}")
    for job in plan.jobs:
        if job.host_id not in host_ids:
            raise RunnerError(f"unknown host for job {job.job_id}")
        if job.kind == "learner" and (job.shardable or not job.gpu_exclusive):
            raise RunnerError("PPO learners must be non-shardable and GPU-exclusive")
        _safe_relative(job.output_relpath)
        _safe_relative(job.numba_cache_relpath)
    for host in plan.hosts:
        _validate_host_root(host, plan.run_id)
        if not host.expected_environment:
            raise RunnerError(f"critical environment contract is empty: {host.host_id}")
    if plan.kind == "b2_train":
        identities = {(job.arm, job.seed) for job in plan.jobs if job.kind == "learner"}
        expected_identities = {(arm, seed) for arm in ARMS for seed in SEEDS}
        if identities != expected_identities or len(plan.jobs) != 6:
            raise RunnerError("B2 train plan must contain exactly six arm-by-seed learners")
        if tuple(plan.required_cli) != REQUIRED_TRAIN_CLI:
            raise RunnerError("B2 train CLI contract drift")
    elif plan.kind == "b2_eval":
        if not plan.parent_plan_sha256 or not SHA256_RE.fullmatch(plan.parent_plan_sha256):
            raise RunnerError("B2 eval plan lacks parent plan identity")
        if not plan.evaluation_contract:
            raise RunnerError("B2 eval plan lacks evaluation contract")
        contract = plan.evaluation_contract
        scenarios = contract.get("scenarios", [])
        variants = contract.get("variants", [])
        shard_count = int(contract.get("shard_count", -1))
        if shard_count != SHARD_COUNT:
            raise RunnerError("B2 eval shard count drift")
        if int(contract.get("expected_scenario_count", -1)) != len(scenarios):
            raise RunnerError("B2 eval scenario count drift")
        if int(contract.get("expected_episode_rows", -1)) != len(scenarios) * len(variants):
            raise RunnerError("B2 eval Cartesian count drift")
        row_indices = [int(item["row_index"]) for item in scenarios]
        l2_ids = [str(item["l2_id"]) for item in scenarios]
        if row_indices != list(range(len(scenarios))) or len(set(l2_ids)) != len(l2_ids):
            raise RunnerError("B2 eval scenarios must be ordered unique physical rows/L2")
        if any(int(item["shard_index"]) != int(item["row_index"]) % SHARD_COUNT for item in scenarios):
            raise RunnerError("B2 eval scenario assignment drift")
        shard_jobs = {job.shard_index for job in plan.jobs if job.kind == "evaluation_shard"}
        if shard_jobs != set(range(SHARD_COUNT)) or len(plan.jobs) != SHARD_COUNT:
            raise RunnerError("B2 eval plan must contain exactly four shards")
    else:
        raise RunnerError(f"unsupported run-plan kind: {plan.kind}")


def _plan_to_dict(plan: RunPlan) -> dict[str, Any]:
    return asdict(plan)


def _plan_from_dict(payload: Mapping[str, Any]) -> RunPlan:
    hosts = tuple(HostSpec(**item) for item in payload["hosts"])
    jobs = tuple(
        JobSpec(
            **{
                **item,
                "argv": tuple(item["argv"]),
            }
        )
        for item in payload["jobs"]
    )
    source_inputs = tuple(InputEntry(**item) for item in payload["source_inputs"])
    inputs = tuple(InputEntry(**item) for item in payload["inputs"])
    queues = {key: tuple(value) for key, value in payload["queues"].items()}
    plan = RunPlan(
        **{
            **payload,
            "hosts": hosts,
            "jobs": jobs,
            "queues": queues,
            "source_inputs": source_inputs,
            "inputs": inputs,
            "required_cli": tuple(payload["required_cli"]),
            "module_path_contract": tuple(payload["module_path_contract"]),
        }
    )
    _verify_plan(plan)
    return plan


def load_plan(path: str | Path) -> RunPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RunnerError("run plan must be a JSON object")
    return _plan_from_dict(payload)


def write_plan(path: Path, plan: RunPlan) -> None:
    _verify_plan(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(json.dumps(_plan_to_dict(plan), indent=2, sort_keys=True).encode("utf-8"))
        handle.write(b"\n")


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(run_id):
        raise RunnerError(f"unsafe run id: {run_id!r}")


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise RunnerError(f"unsafe relative path: {value!r}")
    return path


def _validate_host_root(host: HostSpec, run_id: str) -> None:
    root = PurePosixPath(host.stage_root)
    expected = ISOLATED_BASE / run_id
    if root != expected:
        raise RunnerError(f"stage root must be exactly {expected}: {root}")
    if host.kind not in {"local", "remote"} or host.host_id != host.kind:
        raise RunnerError(f"invalid host identity: {host}")
    if host.kind == "remote" and host.ssh_host != REMOTE_HOST:
        raise RunnerError("only the fixed remote SSH endpoint is allowed")
    if host.kind == "local" and host.ssh_host is not None:
        raise RunnerError("local host must not have ssh_host")
    if host.python != PINNED_PYTHON:
        raise RunnerError(f"unpinned Python for {host.host_id}: {host.python}")
    if not host.display.startswith(":"):
        raise RunnerError(f"DISPLAY must be explicit for {host.host_id}")
    if not host.gpu_uuid or not host.gpu_name:
        raise RunnerError(f"GPU identity must be explicit for {host.host_id}")


def _git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=text
    )
    return result.stdout


def _require_clean_commit(repo: Path, commit: str) -> tuple[str, str]:
    resolved = str(_git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}" )).strip()
    head = str(_git(repo, "rev-parse", "HEAD")).strip()
    if resolved != head:
        raise RunnerError("B2 plan source commit must be the checked-out HEAD")
    dirty = str(_git(repo, "status", "--porcelain=v1", "-uall"))
    if dirty:
        raise RunnerError("B2 plan requires a completely clean committed worktree")
    tree = str(_git(repo, "rev-parse", f"{resolved}^{{tree}}" )).strip()
    return resolved, tree


def _source_has_cli_contract(repo: Path, commit: str, commands: Sequence[str]) -> None:
    try:
        source = str(_git(repo, "show", f"{commit}:bplus_v22/cli.py"))
    except subprocess.CalledProcessError as error:
        raise RunnerError("committed source lacks bplus_v22/cli.py") from error
    missing = [command for command in commands if command not in source]
    if missing:
        raise RunnerError(f"committed CLI lacks required commands: {missing}")
    if "capabilities" not in source:
        raise RunnerError("committed CLI lacks the machine-readable capabilities command")


def _create_source_archive(repo: Path, commit: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "archive", "--format=tar", "--output", str(output), commit],
        cwd=repo,
        check=True,
    )


def _tar_entry(archive: tarfile.TarFile, source: Path, arcname: str) -> InputEntry:
    if not source.is_file() or source.is_symlink():
        raise RunnerError(f"input must be one regular file: {source}")
    data = source.read_bytes()
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mode = 0o444
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    archive.addfile(info, io.BytesIO(data))
    return InputEntry("runtime", arcname, _sha256_bytes(data), len(data))


def _release_files(root: Path) -> list[Path]:
    if not root.is_dir() or not (root / "COMPLETE").is_file():
        raise RunnerError(f"incomplete canonical input release: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise RunnerError(f"empty canonical input release: {root}")
    if any(path.is_symlink() for path in files):
        raise RunnerError(f"symlinks are forbidden in runtime input release: {root}")
    return files


def _verify_existing_output_manifest(root: Path) -> None:
    """Validate an immutable upstream release before it becomes a run input."""

    manifest = root / "output_manifest.sha256"
    if not manifest.is_file():
        raise RunnerError(f"canonical input lacks output_manifest.sha256: {root}")
    seen: set[str] = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        try:
            digest, relpath = line.split("  ", 1)
        except ValueError as error:
            raise RunnerError(f"malformed output manifest line {line_number}: {root}") from error
        _safe_relative(relpath)
        if not SHA256_RE.fullmatch(digest) or relpath in seen:
            raise RunnerError(f"invalid output manifest entry: {relpath}")
        seen.add(relpath)
        path = root / relpath
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != digest:
            raise RunnerError(f"canonical input release hash mismatch: {path}")
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"COMPLETE", "output_manifest.sha256"}
    }
    if seen != expected:
        raise RunnerError(
            f"canonical input manifest/file mismatch: missing={sorted(expected-seen)}, "
            f"extra={sorted(seen-expected)}"
        )


def _create_training_inputs_archive(repo: Path, output: Path) -> tuple[InputEntry, ...]:
    sidecar = repo / SIDECAR_RELEASE
    task8 = repo / TASK8_RELEASE
    bundle = sidecar / "sidecar_bundle.pt"
    if _sha256_file(bundle) != CANONICAL_SIDECAR_SHA256:
        raise RunnerError("canonical sidecar bundle hash mismatch")
    for required in ("training_scenarios.tsv", "development_scenarios.tsv", "config.json"):
        if not (task8 / required).is_file():
            raise RunnerError(f"canonical Task-8 release lacks {required}")
    _verify_existing_output_manifest(sidecar)
    _verify_existing_output_manifest(task8)
    metadata = repo / D2_METADATA
    if not metadata.is_file() or _sha256_file(metadata) != D2_METADATA_SHA256:
        raise RunnerError("canonical D2 episode metadata hash mismatch")
    entries: list[InputEntry] = []
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
        for role, root, prefix in (
            ("sidecar_release", sidecar, "sidecar"),
            ("task8_release", task8, "task8"),
        ):
            for path in _release_files(root):
                entry = _tar_entry(archive, path, f"{prefix}/{path.relative_to(root).as_posix()}")
                entries.append(replace(entry, role=role))
        entry = _tar_entry(archive, metadata, "d2/episode_metadata.tsv")
        entries.append(replace(entry, role="d2_opened_episode_metadata"))
    return tuple(entries)


def _read_source_member(archive_path: Path, relpath: str) -> InputEntry:
    with tarfile.open(archive_path, "r") as archive:
        try:
            member = archive.getmember(relpath)
        except KeyError as error:
            raise RunnerError(f"committed source archive lacks {relpath}") from error
        if not member.isfile():
            raise RunnerError(f"source input is not a regular file: {relpath}")
        handle = archive.extractfile(member)
        if handle is None:
            raise RunnerError(f"cannot read source archive member: {relpath}")
        data = handle.read()
    return InputEntry("source", relpath, _sha256_bytes(data), len(data))


def _critical_environment(python: str) -> dict[str, str]:
    code = r'''
import importlib.metadata as m, json, platform
names = ["torch", "numpy", "numba", "gym", "scipy"]
result = {"python": platform.python_version()}
for name in names:
    try: result[name] = m.version(name)
    except m.PackageNotFoundError: result[name] = "MISSING"
print(json.dumps(result, sort_keys=True))
'''
    result = subprocess.run([python, "-c", code], check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    if value.get("torch") == "MISSING" or value.get("numpy") == "MISSING":
        raise RunnerError("pinned Python lacks required training packages")
    return {str(key): str(item) for key, item in value.items()}


def _default_hosts(
    run_id: str,
    local_gpu_uuid: str,
    remote_gpu_uuid: str,
    environment: dict[str, str],
) -> tuple[HostSpec, HostSpec]:
    root = str(ISOLATED_BASE / run_id)
    return (
        HostSpec(
            "local", "local", root, PINNED_PYTHON, LOCAL_DISPLAY,
            local_gpu_uuid, LOCAL_GPU_NAME, None, environment,
        ),
        HostSpec(
            "remote", "remote", root, PINNED_PYTHON, REMOTE_DISPLAY,
            remote_gpu_uuid, REMOTE_GPU_NAME, REMOTE_HOST, environment,
        ),
    )


def _training_jobs() -> tuple[tuple[JobSpec, ...], dict[str, tuple[str, ...]]]:
    jobs: list[JobSpec] = []
    queues: dict[str, tuple[str, ...]] = {}
    for seed, host in ((0, "remote"), (1, "local")):
        queue_id = f"learner-seed{seed}-{host}"
        ids: list[str] = []
        for arm in ARMS:
            arm_key = arm.lower()
            job_id = f"learner-{arm_key}-seed{seed}"
            ids.append(job_id)
            jobs.append(
                JobSpec(
                    job_id=job_id,
                    kind="learner",
                    host_id=host,
                    queue_id=queue_id,
                    argv=("-m", "bplus_v22.cli", "ppo-pilot"),
                    output_relpath=f"outputs/train/seed{seed}/{arm}",
                    numba_cache_relpath=f"cache/numba/{job_id}",
                    arm=arm,
                    seed=seed,
                    gpu_exclusive=True,
                    shardable=False,
                )
            )
        queues[queue_id] = tuple(ids)
    return tuple(jobs), queues


def _shared_training_config() -> dict[str, Any]:
    return {
        "iterations": 20,
        "episodes_per_iteration": 16,
        "collision_episodes_per_iteration": 8,
        "remaining_episodes_per_iteration": 8,
        "ppo_epochs": 3,
        "minibatch_size": 128,
        "clip_eps": 0.05,
        "action_core_lr": 3e-5,
        "head_lr": 3e-4,
        "sidecar_lr": 3e-6,
        "critic_lr": 5e-5,
        "entropy_coef": 0.001,
        "max_grad_norm": 0.5,
        "target_kl": 0.03,
        "replay_float32_atol": 1e-4,
        "collision_scale_decay": 0.99,
        "deterministic_contract": "centered_fresh_prior",
        "dual_freeze_through_iteration": 9,
        "bc_baseline_expected_collision": 24,
        "bc_baseline_expected_overtake": 138,
        "exploration": {
            "intervention_full_offset": 3.8027754227,
            "conditional_brake_full_offset": 6.0,
            "multipliers": [1.0] * 5 + [0.8, 0.6, 0.4, 0.2] + [0.0] * 11,
            "steer_std_scale": 0.1,
            "brake_std_scale": 1.0,
        },
        "inputs": {
            "bc_checkpoint": "repo/pretrained/end2race.pth",
            "sidecar_release": "inputs/sidecar",
            "task8_release": "inputs/task8",
            "training_manifest": "inputs/task8/training_scenarios.tsv",
            "development_manifest": "inputs/task8/development_scenarios.tsv",
            "d2_episode_metadata": "inputs/d2/episode_metadata.tsv",
        },
        "forbidden_inputs": ["D2_test", "fresh_pool", "final_pool", "eval_results"],
    }


def build_training_plan(
    *,
    repo: Path,
    run_id: str,
    commit: str,
    output: Path,
    local_gpu_uuid: str,
    remote_gpu_uuid: str,
    environment: dict[str, str] | None = None,
) -> RunPlan:
    _validate_run_id(run_id)
    commit, tree = _require_clean_commit(repo, commit)
    _source_has_cli_contract(repo, commit, REQUIRED_TRAIN_CLI)
    if output.exists():
        raise FileExistsError(output)
    source_path = output.with_suffix(".source.tar")
    inputs_path = output.with_suffix(".inputs.tar")
    if source_path.exists() or inputs_path.exists():
        raise FileExistsError("control archive already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    token = f".{run_id}.{random.randrange(1 << 32):08x}.partial"
    temporary = output.parent / token
    temporary.mkdir()
    try:
        source_partial = temporary / "source.tar"
        inputs_partial = temporary / "inputs.tar"
        _create_source_archive(repo, commit, source_partial)
        source_bc = _read_source_member(source_partial, "pretrained/end2race.pth")
        if source_bc.sha256 != CANONICAL_BC_SHA256:
            raise RunnerError("committed BC checkpoint hash mismatch")
        input_entries = _create_training_inputs_archive(repo, inputs_partial)
        source_sha = _sha256_file(source_partial)
        inputs_sha = _sha256_file(inputs_partial)
        environment = environment or _critical_environment(PINNED_PYTHON)
        jobs, queues = _training_jobs()
        plan = _seal_plan(
            RunPlan(
                schema=PLAN_SCHEMA,
                run_id=run_id,
                kind="b2_train",
                created_at=_now(),
                source_commit=commit,
                source_tree=tree,
                source_archive_path=str(source_path.resolve()),
                source_archive_sha256=source_sha,
                source_archive_size=source_partial.stat().st_size,
                inputs_archive_path=str(inputs_path.resolve()),
                inputs_archive_sha256=inputs_sha,
                inputs_archive_size=inputs_partial.stat().st_size,
                source_inputs=(source_bc,),
                inputs=input_entries,
                hosts=_default_hosts(run_id, local_gpu_uuid, remote_gpu_uuid, environment),
                jobs=jobs,
                queues=queues,
                required_cli=REQUIRED_TRAIN_CLI,
                module_path_contract=MODULE_PATH_CONTRACT,
                config=_shared_training_config(),
                collection_root=str(
                    (repo / "Experiments/B2_ppo_pilot/runs" / run_id).resolve()
                ),
            )
        )
        _verify_plan(plan)
        os.replace(source_partial, source_path)
        os.replace(inputs_partial, inputs_path)
        try:
            write_plan(output, plan)
        except Exception:
            source_path.unlink(missing_ok=True)
            inputs_path.unlink(missing_ok=True)
            raise
        return plan
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _deterministic_input_archive(
    output: Path, files: Sequence[tuple[str, str, Path]]
) -> tuple[InputEntry, ...]:
    entries: list[InputEntry] = []
    seen: set[str] = set()
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
        for role, arcname, source in sorted(files, key=lambda item: item[1]):
            if arcname in seen:
                raise RunnerError(f"duplicate input archive path: {arcname}")
            seen.add(arcname)
            entries.append(replace(_tar_entry(archive, source, arcname), role=role))
    return tuple(entries)


def _parse_checkpoint(value: str) -> tuple[str, int, Path]:
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise RunnerError("checkpoint must be ARM,SEED,PATH")
    arm, seed_text, path_text = parts
    seed = int(seed_text)
    if arm not in ARMS or seed not in SEEDS:
        raise RunnerError(f"unexpected checkpoint identity: {arm}, seed {seed}")
    path = Path(path_text).resolve()
    if not path.is_file() or path.is_symlink():
        raise RunnerError(f"checkpoint is not one regular file: {path}")
    return arm, seed, path


def _validate_training_checkpoint_source(
    parent: RunPlan, arm: str, seed: int, path: Path
) -> None:
    if path.name != "iter_0020.pt" or path.parent.name != "checkpoints":
        raise RunnerError("evaluation checkpoint must be the final learner checkpoint")
    release = path.parent.parent
    if (
        not (release / "COMPLETE").is_file()
        or release.with_name(release.name + ".partial").exists()
        or not (release / "summary.json").is_file()
    ):
        raise RunnerError("evaluation checkpoint does not come from a COMPLETE learner")
    summary = json.loads((release / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("schema") != "bplus-v2.2-b2-ppo-pilot-1"
        or summary.get("integrity_passed") is not True
        or summary.get("passed") is not True
        or summary.get("arm") != arm
        or summary.get("seed") != seed
        or summary.get("iterations") != 20
        or summary.get("run_plan_sha256") != parent.plan_sha256
        or summary.get("iteration20_checkpoint_sha256") != _sha256_file(path)
    ):
        raise RunnerError("evaluation checkpoint learner envelope mismatch")


def _eval_jobs() -> tuple[tuple[JobSpec, ...], dict[str, tuple[str, ...]]]:
    jobs: list[JobSpec] = []
    queues: dict[str, tuple[str, ...]] = {
        "eval-local": ("eval-shard0",),
        "eval-remote-sequential": ("eval-shard1", "eval-shard2", "eval-shard3"),
    }
    for shard in range(SHARD_COUNT):
        host = "local" if shard == 0 else "remote"
        queue = "eval-local" if shard == 0 else "eval-remote-sequential"
        job_id = f"eval-shard{shard}"
        jobs.append(
            JobSpec(
                job_id=job_id,
                kind="evaluation_shard",
                host_id=host,
                queue_id=queue,
                argv=("-m", "bplus_v22.cli", "ppo-evaluate"),
                output_relpath=f"outputs/eval/shard{shard}",
                numba_cache_relpath=f"cache/numba/{job_id}",
                shard_index=shard,
                shard_count=SHARD_COUNT,
                gpu_exclusive=True,
                shardable=True,
            )
        )
    return tuple(jobs), queues


def build_evaluation_plan(
    *,
    repo: Path,
    run_id: str,
    training_plan_path: Path,
    checkpoints: Sequence[str],
    output: Path,
) -> RunPlan:
    _validate_run_id(run_id)
    parent = load_plan(training_plan_path)
    if parent.kind != "b2_train":
        raise RunnerError("evaluation parent must be a B2 training plan")
    if output.exists():
        raise FileExistsError(output)
    parsed = [_parse_checkpoint(value) for value in checkpoints]
    identities = {(arm, seed) for arm, seed, _ in parsed}
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if identities != expected or len(parsed) != len(expected):
        raise RunnerError(f"evaluation requires exactly six checkpoints: {expected}")
    for arm, seed, path in parsed:
        _validate_training_checkpoint_source(parent, arm, seed, path)
    parent_inputs = Path(parent.inputs_archive_path)
    if _sha256_file(parent_inputs) != parent.inputs_archive_sha256:
        raise RunnerError("parent input archive drift")
    source_archive = Path(parent.source_archive_path)
    if _sha256_file(source_archive) != parent.source_archive_sha256:
        raise RunnerError("parent source archive drift")
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs_path = output.with_suffix(".inputs.tar")
    if inputs_path.exists():
        raise FileExistsError(inputs_path)
    with tempfile.TemporaryDirectory() as temporary_name:
        temporary = Path(temporary_name)
        _safe_extract(parent_inputs, temporary)
        files: list[tuple[str, str, Path]] = []
        task8 = temporary / "task8"
        for path in _release_files(task8):
            files.append(("task8_release", f"task8/{path.relative_to(task8).as_posix()}", path))
        checkpoint_meta: list[dict[str, Any]] = []
        for arm, seed, path in sorted(parsed):
            arcname = f"checkpoints/{arm}_seed{seed}_iter20.pt"
            sha = _sha256_file(path)
            files.append(("checkpoint", arcname, path))
            checkpoint_meta.append(
                {"arm": arm, "seed": seed, "relpath": f"inputs/{arcname}",
                 "sha256": sha, "size": path.stat().st_size}
            )
        partial = inputs_path.with_name(f".{inputs_path.name}.partial")
        if partial.exists():
            raise FileExistsError(partial)
        entries = _deterministic_input_archive(partial, files)
        input_sha = _sha256_file(partial)
        input_size = partial.stat().st_size
        task8_manifest = task8 / "development_scenarios.tsv"
        manifest_sha = _sha256_file(task8_manifest)
        with task8_manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if len(rows) != 288 or any(not row.get("l2_id") for row in rows):
            raise RunnerError("canonical development manifest must contain 288 L2 rows")
        scenarios = [
            {"row_index": index, "l2_id": row["l2_id"], "shard_index": index % SHARD_COUNT}
            for index, row in enumerate(rows)
        ]
        variants = ["BC"] + [f"{arm}::seed{seed}" for arm in ARMS for seed in SEEDS]
        training_manifest_sha = _sha256_file(task8 / "training_scenarios.tsv")
        checkpoint_set_sha = _sha256_bytes(_canonical_json(checkpoint_meta))
        jobs, queues = _eval_jobs()
        plan = _seal_plan(
            RunPlan(
                schema=PLAN_SCHEMA,
                run_id=run_id,
                kind="b2_eval",
                created_at=_now(),
                source_commit=parent.source_commit,
                source_tree=parent.source_tree,
                source_archive_path=parent.source_archive_path,
                source_archive_sha256=parent.source_archive_sha256,
                source_archive_size=parent.source_archive_size,
                inputs_archive_path=str(inputs_path.resolve()),
                inputs_archive_sha256=input_sha,
                inputs_archive_size=input_size,
                source_inputs=parent.source_inputs,
                inputs=entries,
                hosts=tuple(replace(host, stage_root=str(ISOLATED_BASE / run_id)) for host in parent.hosts),
                jobs=jobs,
                queues=queues,
                required_cli=("ppo-evaluate", "ppo-merge-eval"),
                module_path_contract=parent.module_path_contract,
                config={"evaluation_offsets": [0.0, 0.0], "checkpoint_iteration": 20},
                collection_root=str((repo / "Experiments/B2_ppo_pilot/evaluations" / run_id).resolve()),
                parent_plan_sha256=parent.plan_sha256,
                evaluation_contract={
                    "manifest_relpath": "inputs/task8/development_scenarios.tsv",
                    "manifest_sha256": manifest_sha,
                    "checkpoint_set": checkpoint_meta,
                    "checkpoint_set_sha256": checkpoint_set_sha,
                    "training_manifest_sha256": training_manifest_sha,
                    "shard_count": SHARD_COUNT,
                    "assignment": "physical_row_index_mod_shard_count",
                    "scenarios": scenarios,
                    "variants": variants,
                    "expected_scenario_count": 288,
                    "expected_episode_rows": 288 * len(variants),
                },
            )
        )
        _verify_plan(plan)
        os.replace(partial, inputs_path)
        try:
            write_plan(output, plan)
        except Exception:
            inputs_path.unlink(missing_ok=True)
            raise
        return plan


def _host(plan: RunPlan, host_id: str) -> HostSpec:
    for host in plan.hosts:
        if host.host_id == host_id:
            return host
    raise RunnerError(f"unknown host: {host_id}")


def _host_ids(args: argparse.Namespace) -> tuple[str, ...]:
    if getattr(args, "all_hosts", False):
        return ("local", "remote")
    return (args.host,)


def _display_command(argv: Sequence[str]) -> str:
    return shlex.join([str(value) for value in argv])


def _ssh_argv(host: HostSpec, remote_argv: Sequence[str]) -> list[str]:
    if host.kind != "remote" or not host.ssh_host:
        raise RunnerError("ssh requested for a non-remote host")
    return ["ssh", host.ssh_host, _display_command(remote_argv)]


def _run_command(argv: Sequence[str], *, dry_run: bool) -> int:
    print(_display_command(argv))
    if dry_run:
        return 0
    return subprocess.run(list(argv), check=False).returncode


def _archive_targets(host: HostSpec) -> tuple[str, str, str]:
    root = PurePosixPath(host.stage_root)
    return (
        str(root / "control/source.tar"),
        str(root / "control/inputs.tar"),
        str(root / "control/run_plan.json"),
    )


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as error:
                raise RunnerError(f"unsafe tar member: {member.name}") from error
            if member.issym() or member.islnk() or member.isdev():
                raise RunnerError(f"unsupported tar member: {member.name}")
        archive.extractall(destination)


def _stage_local(plan_path: Path, plan: RunPlan, host: HostSpec) -> None:
    root = Path(host.stage_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    for part in ("control", "repo", "inputs", "outputs", "cache"):
        (root / part).mkdir(parents=True)
    source_target = root / "control/source.tar"
    inputs_target = root / "control/inputs.tar"
    plan_target = root / "control/run_plan.json"
    shutil.copyfile(plan.source_archive_path, source_target)
    shutil.copyfile(plan.inputs_archive_path, inputs_target)
    shutil.copyfile(plan_path, plan_target)
    _safe_extract(source_target, root / "repo")
    _safe_extract(inputs_target, root / "inputs")
    _verify_staged_files(plan, host)
    _make_inputs_read_only(root)
    Path(_lock_path(host)).parent.mkdir(parents=True, exist_ok=True)
    (root / "control/STAGED").write_text(plan.plan_sha256 + "\n", encoding="utf-8")


def _remote_stage_commands(plan_path: Path, plan: RunPlan, host: HostSpec) -> list[list[str]]:
    root = host.stage_root
    source_target, inputs_target, plan_target = _archive_targets(host)
    prepare = (
        f"umask 077; test ! -e {shlex.quote(root)}; "
        f"mkdir -p {shlex.quote(root + '/control')} {shlex.quote(root + '/repo')} "
        f"{shlex.quote(root + '/inputs')} {shlex.quote(root + '/outputs')} "
        f"{shlex.quote(root + '/cache')} /home/haowei/.cache/end2race/locks"
    )
    extract = (
        f"set -eu; "
        f"printf '%s  %s\\n' {shlex.quote(plan.source_archive_sha256)} {shlex.quote(source_target)} | sha256sum -c -; "
        f"printf '%s  %s\\n' {shlex.quote(plan.inputs_archive_sha256)} {shlex.quote(inputs_target)} | sha256sum -c -; "
        f"tar -xf {shlex.quote(source_target)} -C {shlex.quote(root + '/repo')}; "
        f"tar -xf {shlex.quote(inputs_target)} -C {shlex.quote(root + '/inputs')}; "
        f"{shlex.quote(host.python)} {shlex.quote(root + '/repo/Experiments/runner.py')} "
        f"_verify-stage {shlex.quote(plan_target)} --host remote"
    )
    return [
        ["ssh", host.ssh_host or "", prepare],
        ["rsync", "-a", "--protect-args", plan.source_archive_path, f"{host.ssh_host}:{source_target}"],
        ["rsync", "-a", "--protect-args", plan.inputs_archive_path, f"{host.ssh_host}:{inputs_target}"],
        ["rsync", "-a", "--protect-args", str(plan_path), f"{host.ssh_host}:{plan_target}"],
        ["ssh", host.ssh_host or "", extract],
    ]


def stage(plan_path: Path, plan: RunPlan, host_ids: Sequence[str], dry_run: bool) -> int:
    for host_id in host_ids:
        host = _host(plan, host_id)
        if host.kind == "local":
            print(f"stage local -> {host.stage_root}")
            if not dry_run:
                _stage_local(plan_path, plan, host)
        else:
            for command in _remote_stage_commands(plan_path, plan, host):
                code = _run_command(command, dry_run=dry_run)
                if code:
                    return code
    return 0


def _verify_staged_files(plan: RunPlan, host: HostSpec) -> None:
    root = Path(host.stage_root)
    if root.is_symlink() or root.resolve() != root:
        raise RunnerError("stage root must be an absolute non-symlink path")
    plan_copy = load_plan(root / "control/run_plan.json")
    if plan_copy.plan_sha256 != plan.plan_sha256:
        raise RunnerError("staged plan digest mismatch")
    for filename, expected_sha, expected_size in (
        ("source.tar", plan.source_archive_sha256, plan.source_archive_size),
        ("inputs.tar", plan.inputs_archive_sha256, plan.inputs_archive_size),
    ):
        path = root / "control" / filename
        if path.stat().st_size != expected_size or _sha256_file(path) != expected_sha:
            raise RunnerError(f"staged control archive mismatch: {filename}")
    for entry in (*plan.source_inputs, *plan.inputs):
        prefix = "repo" if entry.role == "source" else "inputs"
        path = root / prefix / entry.relpath
        if not path.is_file() or path.is_symlink():
            raise RunnerError(f"missing staged input: {path}")
        if path.stat().st_size != entry.size or _sha256_file(path) != entry.sha256:
            raise RunnerError(f"staged input digest mismatch: {path}")


def _make_inputs_read_only(root: Path) -> None:
    """Prevent runtime mutation; caches are redirected outside these trees."""

    for tree in (root / "repo", root / "inputs"):
        for path in sorted(tree.rglob("*"), reverse=True):
            if path.is_symlink():
                raise RunnerError(f"symlink appeared in staged immutable tree: {path}")
            if path.is_file():
                path.chmod(path.stat().st_mode & ~0o222)
            elif path.is_dir():
                path.chmod(path.stat().st_mode & ~0o222)
        tree.chmod(tree.stat().st_mode & ~0o222)


def _job_environment(plan: RunPlan, host: HostSpec, job: JobSpec) -> dict[str, str]:
    root = Path(host.stage_root)
    return {
        "CUDA_VISIBLE_DEVICES": "0",
        "DISPLAY": host.display,
        "PYTHONHASHSEED": str(job.seed if job.seed is not None else 20260712),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONPATH": f"{root / 'repo'}:{root / 'repo/f1tenth_gym/gym'}",
        "PYTHONPYCACHEPREFIX": str(root / "cache/pycache" / job.job_id),
        "NUMBA_CACHE_DIR": str(root / job.numba_cache_relpath),
        "XDG_CACHE_HOME": str(root / "cache/xdg" / job.job_id),
        "MPLCONFIGDIR": str(root / "cache/mpl" / job.job_id),
    }


def _preflight_command(plan: RunPlan, host: HostSpec) -> list[str]:
    root = PurePosixPath(host.stage_root)
    argv = [
        host.python,
        str(root / "repo/Experiments/runner.py"),
        "_preflight-host",
        str(root / "control/run_plan.json"),
        "--host",
        host.host_id,
    ]
    return argv if host.kind == "local" else _ssh_argv(host, argv)


def _baseline_command(plan: RunPlan) -> list[str]:
    host = _host(plan, "local")
    root = PurePosixPath(host.stage_root)
    inner = [
        host.python,
        str(root / "repo/Experiments/runner.py"),
        "_baseline-host",
        str(root / "control/run_plan.json"),
        "--host",
        "local",
    ]
    return ["flock", "-n", _lock_path(host), *inner]


def baseline_preflight(plan: RunPlan, dry_run: bool) -> int:
    if plan.kind != "b2_train":
        raise RunnerError("BC baseline preflight requires a B2 training plan")
    local = _host(plan, "local")
    remote = _host(plan, "remote")
    local_marker = Path(local.stage_root) / "control/bc_baseline_preflight.json"
    remote_marker = f"{remote.stage_root}/control/bc_baseline_preflight.json"
    commands = [
        _baseline_command(plan),
        [
            "rsync",
            "-a",
            "--protect-args",
            str(local_marker),
            f"{remote.ssh_host}:{remote_marker}",
        ],
    ]
    for command in commands:
        code = _run_command(command, dry_run=dry_run)
        if code:
            return code
    return 0


def _validate_baseline_marker(plan: RunPlan, root: Path) -> dict[str, Any]:
    path = root / "control/bc_baseline_preflight.json"
    if not path.is_file() or path.is_symlink():
        raise RunnerError("BC baseline preflight marker is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "bplus-v2.2-b2-bc-baseline-preflight-1"
        or value.get("passed") is not True
        or value.get("run_plan_sha256") != plan.plan_sha256
        or value.get("source_commit") != plan.source_commit
        or value.get("scenario_count") != 288
        or value.get("collision") != 24
        or value.get("terminal_overtake") != 138
        or value.get("candidate_evaluated") is not False
        or len(value.get("rows", ())) != 288
    ):
        raise RunnerError("BC baseline preflight marker/envelope mismatch")
    return value


def baseline_host(plan_path: Path, host_id: str) -> int:
    if host_id != "local":
        raise RunnerError("BC baseline preflight runs once on the local host")
    plan = load_plan(plan_path)
    host = _host(plan, host_id)
    _verify_staged_files(plan, host)
    subprocess.run(["xdpyinfo", "-display", host.display], check=True, capture_output=True)
    _probe_gpu(host)
    if _critical_environment(host.python) != host.expected_environment:
        raise RunnerError("BC baseline preflight environment mismatch")
    root = Path(host.stage_root)
    probe_job = JobSpec(
        "bc-baseline-preflight",
        "preflight",
        host.host_id,
        "preflight",
        tuple(),
        "outputs/preflight",
        "cache/numba/bc-baseline-preflight",
        gpu_exclusive=True,
    )
    cache = root / probe_job.numba_cache_relpath
    # A failed simulator attempt publishes no marker; reusing its isolated
    # Numba cache makes the explicit baseline command safely retryable.
    cache.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **_job_environment(plan, host, probe_job)}
    output = root / "control/bc_baseline_preflight.json"
    subprocess.run(
        [
            host.python,
            "-m",
            "bplus_v22.cli",
            "ppo-baseline-preflight",
            "--run-plan",
            str(plan_path),
            "--output",
            str(output),
        ],
        check=True,
        cwd=root / "repo",
        env=env,
    )
    _validate_baseline_marker(plan, root)
    return 0


def preflight(plan: RunPlan, host_ids: Sequence[str], dry_run: bool) -> int:
    for host_id in host_ids:
        code = _run_command(_preflight_command(plan, _host(plan, host_id)), dry_run=dry_run)
        if code:
            return code
    return 0


def _probe_gpu(host: HostSpec) -> dict[str, str]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    matches = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) == 4 and parts[0] == host.gpu_uuid:
            matches.append(parts)
    if len(matches) != 1:
        raise RunnerError(f"expected GPU UUID not found exactly once: {host.gpu_uuid}")
    uuid, name, driver, memory = matches[0]
    if name != host.gpu_name:
        raise RunnerError(f"GPU name mismatch: {name!r} != {host.gpu_name!r}")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    busy = [line for line in processes.splitlines() if host.gpu_uuid in line]
    if busy:
        raise RunnerError(f"GPU already has compute processes: {busy}")
    return {"uuid": uuid, "name": name, "driver": driver, "memory_total_mib": memory}


def _probe_module_paths(plan: RunPlan, host: HostSpec) -> dict[str, str]:
    root = Path(host.stage_root)
    code = r'''
import importlib, json
result = {}
for name in %r:
    module = importlib.import_module(name)
    result[name] = str(getattr(module, "__file__", ""))
print(json.dumps(result, sort_keys=True))
''' % (tuple(plan.module_path_contract),)
    probe_job = JobSpec(
        "preflight", "preflight", host.host_id, "preflight", tuple(),
        "outputs/preflight", "cache/numba/preflight", gpu_exclusive=False,
    )
    env = {**os.environ, **_job_environment(plan, host, probe_job)}
    result = subprocess.run(
        [host.python, "-c", code], check=True, capture_output=True, text=True, env=env,
        cwd=root / "repo",
    )
    paths = json.loads(result.stdout)
    repo = (root / "repo").resolve()
    for module, value in paths.items():
        try:
            Path(value).resolve().relative_to(repo)
        except ValueError as error:
            raise RunnerError(f"module {module} escaped staged source: {value}") from error
    return {str(key): str(value) for key, value in paths.items()}


def _probe_capabilities(plan: RunPlan, host: HostSpec) -> dict[str, Any]:
    root = Path(host.stage_root)
    probe_job = JobSpec(
        "preflight", "preflight", host.host_id, "preflight", tuple(),
        "outputs/preflight", "cache/numba/preflight", gpu_exclusive=False,
    )
    env = {**os.environ, **_job_environment(plan, host, probe_job)}
    result = subprocess.run(
        [host.python, "-m", "bplus_v22.cli", "capabilities", "--json"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=root / "repo",
    )
    capabilities = json.loads(result.stdout)
    if capabilities.get("schema") != CAPABILITIES_SCHEMA:
        raise RunnerError("CLI capabilities schema mismatch")
    commands = set(capabilities.get("commands", []))
    missing = sorted(set(plan.required_cli) - commands)
    if missing:
        raise RunnerError(f"staged CLI lacks commands: {missing}")
    return capabilities


def _validate_cli_plan(plan: RunPlan, host: HostSpec) -> None:
    """Let the staged B2 implementation validate its full typed plan contract."""

    root = Path(host.stage_root)
    command = "ppo-pilot" if plan.kind == "b2_train" else "ppo-evaluate"
    probe_job = JobSpec(
        "preflight", "preflight", host.host_id, "preflight", tuple(),
        "outputs/preflight", "cache/numba/preflight", gpu_exclusive=False,
    )
    env = {**os.environ, **_job_environment(plan, host, probe_job)}
    subprocess.run(
        [
            host.python,
            "-m",
            "bplus_v22.cli",
            command,
            "--run-plan",
            str(root / "control/run_plan.json"),
            "--validate-plan-only",
        ],
        check=True,
        cwd=root / "repo",
        env=env,
    )


def preflight_host(plan_path: Path, host_id: str) -> int:
    plan = load_plan(plan_path)
    host = _host(plan, host_id)
    _verify_staged_files(plan, host)
    if plan.kind == "b2_train":
        _validate_baseline_marker(plan, Path(host.stage_root))
    for command in ("rsync", "tar", "flock", "nvidia-smi", "xdpyinfo"):
        if shutil.which(command) is None:
            raise RunnerError(f"required executable missing: {command}")
    subprocess.run(["xdpyinfo", "-display", host.display], check=True, capture_output=True)
    gpu = _probe_gpu(host)
    actual_environment = _critical_environment(host.python)
    if actual_environment != host.expected_environment:
        raise RunnerError(
            f"critical environment mismatch: {actual_environment} != {host.expected_environment}"
        )
    paths = _probe_module_paths(plan, host)
    capabilities = _probe_capabilities(plan, host)
    _validate_cli_plan(plan, host)
    root = Path(host.stage_root)
    usage = shutil.disk_usage(root)
    if usage.free < 20 * 1024**3:
        raise RunnerError("less than 20 GiB free in isolated run root")
    marker = {
        "schema": "end2race-host-preflight-1",
        "plan_sha256": plan.plan_sha256,
        "host": host_id,
        "completed_at": _now(),
        "gpu": gpu,
        "environment": actual_environment,
        "module_paths": paths,
        "capabilities": capabilities,
        "display": host.display,
    }
    path = root / "control/preflight.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


def _lock_path(host: HostSpec) -> str:
    safe_uuid = re.sub(r"[^A-Za-z0-9_.-]", "_", host.gpu_uuid)
    return f"/home/haowei/.cache/end2race/locks/end2race_{safe_uuid}.lock"


def _execute_command(
    plan: RunPlan, host: HostSpec, *, resume: bool = False
) -> list[str]:
    root = PurePosixPath(host.stage_root)
    inner = [
        host.python,
        str(root / "repo/Experiments/runner.py"),
        "_resume-host" if resume else "_execute-host",
        str(root / "control/run_plan.json"),
        "--host",
        host.host_id,
    ]
    argv = ["flock", "-n", _lock_path(host), *inner]
    return argv if host.kind == "local" else _ssh_argv(host, argv)


def execute(plan: RunPlan, host_ids: Sequence[str], dry_run: bool) -> int:
    commands = [_execute_command(plan, _host(plan, host_id)) for host_id in host_ids]
    for command in commands:
        print(_display_command(command))
    if dry_run:
        return 0
    running = [subprocess.Popen(command) for command in commands]
    failures = [process.wait() for process in running]
    return 0 if all(code == 0 for code in failures) else 1


def resume(plan: RunPlan, host_ids: Sequence[str], dry_run: bool) -> int:
    if plan.kind != "b2_train":
        raise RunnerError("explicit resume is available only for B2 training")
    commands = [
        _execute_command(plan, _host(plan, host_id), resume=True)
        for host_id in host_ids
    ]
    for command in commands:
        print(_display_command(command))
    if dry_run:
        return 0
    running = [subprocess.Popen(command) for command in commands]
    failures = [process.wait() for process in running]
    return 0 if all(code == 0 for code in failures) else 1


def _host_jobs(plan: RunPlan, host_id: str) -> list[JobSpec]:
    by_id = {job.job_id: job for job in plan.jobs}
    jobs: list[JobSpec] = []
    for queue_id, ids in plan.queues.items():
        if not ids or by_id[ids[0]].host_id != host_id:
            continue
        if any(by_id[job_id].host_id != host_id for job_id in ids):
            raise RunnerError(f"queue crosses hosts: {queue_id}")
        jobs.extend(by_id[job_id] for job_id in ids)
    return jobs


def _validate_job_output(plan: RunPlan, root: Path, job: JobSpec) -> None:
    output = root / job.output_relpath
    partial = output.with_name(output.name + ".partial")
    if not output.is_dir() or partial.exists() or not (output / "COMPLETE").is_file():
        raise RunnerError(f"job did not publish one atomic COMPLETE release: {job.job_id}")
    if job.kind == "learner":
        summary_path = output / "summary.json"
        if not summary_path.is_file():
            raise RunnerError(f"learner summary is missing: {job.job_id}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("schema") != "bplus-v2.2-b2-ppo-pilot-1"
            or summary.get("integrity_passed") is not True
            or summary.get("passed") is not True
            or summary.get("arm") != job.arm
            or summary.get("seed") != job.seed
            or summary.get("iterations") != 20
            or summary.get("run_plan_sha256") != plan.plan_sha256
        ):
            raise RunnerError(f"learner COMPLETE envelope mismatch: {job.job_id}")
    elif job.kind == "evaluation_shard":
        shard_path = output / "shard.json"
        if not shard_path.is_file():
            raise RunnerError(f"evaluation shard summary is missing: {job.job_id}")
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        if (
            shard.get("schema") != "bplus-v2.2-ppo-eval-shard-1"
            or shard.get("shard_index") != job.shard_index
            or shard.get("shard_count") != job.shard_count
        ):
            raise RunnerError(f"evaluation shard COMPLETE envelope mismatch: {job.job_id}")
    else:
        raise RunnerError(f"unsupported executable job kind: {job.kind}")


def execute_host(plan_path: Path, host_id: str, *, resume: bool = False) -> int:
    plan = load_plan(plan_path)
    host = _host(plan, host_id)
    root = Path(host.stage_root)
    marker_path = root / "control/preflight.json"
    if not marker_path.is_file():
        raise RunnerError("host preflight has not passed")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("plan_sha256") != plan.plan_sha256 or marker.get("host") != host_id:
        raise RunnerError("preflight marker does not authorize this plan/host")
    # This runs while the outer `flock` is held, closing the race between the
    # earlier preflight and learner launch.
    _probe_gpu(host)
    status_path = root / "control/status.json"
    events_path = root / "control/status.jsonl"
    jobs = _host_jobs(plan, host_id)
    if not jobs:
        raise RunnerError(f"no jobs assigned to {host_id}")
    if resume:
        if plan.kind != "b2_train":
            raise RunnerError("only B2 learner queues support explicit resume")
        if not status_path.is_file() or not events_path.is_file():
            raise RunnerError("resume requires an existing host status/event ledger")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            status.get("schema") != "end2race-host-status-1"
            or status.get("plan_sha256") != plan.plan_sha256
            or status.get("host") != host_id
            or status.get("state") not in {"FAILED", "RUNNING"}
        ):
            raise RunnerError("host status does not authorize explicit resume")
        status["state"] = "RUNNING"
        status["resume_attempts"] = int(status.get("resume_attempts", 0)) + 1
    else:
        if status_path.exists() or events_path.exists():
            raise RunnerError("execution status already exists; use explicit resume")
        status = {
            "schema": "end2race-host-status-1",
            "plan_sha256": plan.plan_sha256,
            "host": host_id,
            "state": "RUNNING",
            "jobs": {},
            "started_at": _now(),
            "resume_attempts": 0,
        }

    def persist(event: dict[str, Any]) -> None:
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, status_path)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    persist(
        {
            "event": "host_resumed" if resume else "host_started",
            "at": _now() if resume else status["started_at"],
        }
    )
    pending_seen = False
    for job in jobs:
        previous = status["jobs"].get(job.job_id)
        if resume and previous and previous.get("state") == "COMPLETE":
            if pending_seen:
                raise RunnerError("completed job appears after resume boundary")
            _validate_job_output(plan, root, job)
            continue
        pending_seen = True
        cache = root / job.numba_cache_relpath
        output = root / job.output_relpath
        partial = output.with_name(output.name + ".partial")
        if resume and previous and previous.get("state") in {"FAILED", "RUNNING"} and output.is_dir():
            _validate_job_output(plan, root, job)
            recovered_at = _now()
            status["jobs"][job.job_id] = {
                "state": "COMPLETE",
                "exit_code": 0,
                "started_at": previous.get("started_at"),
                "finished_at": recovered_at,
                "status_recovered_from_complete_release": True,
            }
            persist(
                {
                    "event": "job_status_recovered",
                    "job_id": job.job_id,
                    "at": recovered_at,
                }
            )
            continue
        resume_job = bool(
            resume
            and previous
            and previous.get("state") in {"FAILED", "RUNNING"}
        )
        if resume_job:
            if output.exists() or not partial.is_dir() or not cache.is_dir():
                raise RunnerError(f"job resume boundary is invalid: {job.job_id}")
        else:
            cache.mkdir(parents=True, exist_ok=False)
            if output.exists() or partial.exists():
                raise RunnerError(f"job output is not fresh: {output}")
        env = {**os.environ, **_job_environment(plan, host, job)}
        argv = [
            host.python,
            *job.argv,
            "--run-plan",
            str(plan_path),
            "--job-id",
            job.job_id,
        ]
        if resume_job:
            argv.append("--resume")
        started = _now()
        status["jobs"][job.job_id] = {"state": "RUNNING", "started_at": started}
        persist({"event": "job_started", "job_id": job.job_id, "at": started})
        code = subprocess.run(argv, cwd=root / "repo", env=env, check=False).returncode
        if code == 0:
            try:
                _validate_job_output(plan, root, job)
            except RunnerError:
                code = 86
        finished = _now()
        status["jobs"][job.job_id] = {
            "state": "COMPLETE" if code == 0 else "FAILED",
            "exit_code": code,
            "started_at": started,
            "finished_at": finished,
        }
        persist({"event": "job_finished", "job_id": job.job_id, "exit_code": code, "at": finished})
        if code:
            status["state"] = "FAILED"
            status["finished_at"] = finished
            persist({"event": "host_failed", "job_id": job.job_id, "at": finished})
            return code
    status["state"] = "COMPLETE"
    status["finished_at"] = _now()
    persist({"event": "host_complete", "at": status["finished_at"]})
    return 0


def _status_command(plan: RunPlan, host: HostSpec) -> list[str]:
    status = str(PurePosixPath(host.stage_root) / "control/status.json")
    if host.kind == "local":
        return ["cat", status]
    return _ssh_argv(host, ["cat", status])


def status(plan: RunPlan, host_ids: Sequence[str], dry_run: bool) -> int:
    result = 0
    for host_id in host_ids:
        code = _run_command(_status_command(plan, _host(plan, host_id)), dry_run=dry_run)
        result = result or code
    return result


def _collect_commands(plan: RunPlan, collection: Path | None = None) -> list[list[str]]:
    collection = collection or Path(plan.collection_root)
    local = _host(plan, "local")
    remote = _host(plan, "remote")
    commands: list[list[str]] = [
        [
            "cp",
            "-a",
            str(Path(local.stage_root) / "outputs"),
            str(collection / "hosts/local/outputs"),
        ],
        [
            "cp",
            "-a",
            str(Path(local.stage_root) / "control/status.json"),
            str(collection / "hosts/local/status.json"),
        ],
        [
            "rsync",
            "-a",
            "--protect-args",
            f"{remote.ssh_host}:{remote.stage_root}/outputs/",
            str(collection / "hosts/remote/outputs/"),
        ],
        [
            "rsync",
            "-a",
            "--protect-args",
            f"{remote.ssh_host}:{remote.stage_root}/control/status.json",
            str(collection / "hosts/remote/status.json"),
        ],
    ]
    if plan.kind == "b2_train":
        commands.append(
            [
                "cp",
                "-a",
                str(Path(local.stage_root) / "control/bc_baseline_preflight.json"),
                str(collection / "control/bc_baseline_preflight.json"),
            ]
        )
    return commands


def collect(plan: RunPlan, dry_run: bool) -> int:
    collection = Path(plan.collection_root)
    partial = collection.with_name(collection.name + ".partial")
    if collection.exists() or partial.exists():
        raise FileExistsError(collection)
    commands = _collect_commands(plan, partial)
    for command in commands:
        print(_display_command(command))
    if dry_run:
        return 0
    (partial / "hosts/local").mkdir(parents=True)
    (partial / "hosts/remote").mkdir(parents=True)
    (partial / "control").mkdir(parents=True)
    for command in commands:
        code = subprocess.run(command, check=False).returncode
        if code:
            return code
    for host_id in ("local", "remote"):
        status_path = partial / f"hosts/{host_id}/status.json"
        status_value = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            status_value.get("plan_sha256") != plan.plan_sha256
            or status_value.get("host") != host_id
            or status_value.get("state") != "COMPLETE"
        ):
            raise RunnerError(f"cannot collect incomplete/mismatched host: {host_id}")
        host_root = partial / f"hosts/{host_id}"
        expected_jobs = _host_jobs(plan, host_id)
        if set(status_value.get("jobs", {})) != {
            job.job_id for job in expected_jobs
        }:
            raise RunnerError(f"collected host job inventory mismatch: {host_id}")
        for job in expected_jobs:
            if status_value["jobs"][job.job_id].get("state") != "COMPLETE":
                raise RunnerError(f"collected job is not complete: {job.job_id}")
            _validate_job_output(plan, host_root, job)
    if plan.kind == "b2_train":
        _validate_baseline_marker(plan, partial)
    (partial / "run_plan.json").write_text(
        json.dumps(_plan_to_dict(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, collection)
    return 0


def validate_eval_collection(plan: RunPlan, collection: Path) -> dict[str, Any]:
    if plan.kind != "b2_eval" or not plan.evaluation_contract:
        raise RunnerError("merge requires a B2 evaluation plan")
    contract = plan.evaluation_contract
    scenarios = contract["scenarios"]
    variants = contract["variants"]
    expected: set[tuple[int, str, str]] = {
        (int(item["row_index"]), str(item["l2_id"]), str(variant))
        for item in scenarios
        for variant in variants
    }
    observed: set[tuple[int, str, str]] = set()
    for shard in range(int(contract["shard_count"])):
        host = "local" if shard == 0 else "remote"
        directory = collection / f"hosts/{host}/outputs/eval/shard{shard}"
        if not (directory / "COMPLETE").is_file():
            raise RunnerError(f"eval shard is incomplete: {shard}")
        rows_path = directory / "episodes.tsv"
        with rows_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        for row in rows:
            required = {
                "row_index", "l2_id", "variant_id", "shard_index",
                "manifest_sha256", "checkpoint_set_sha256",
            }
            if not required.issubset(row):
                raise RunnerError(f"eval shard schema mismatch: {shard}")
            key = (int(row["row_index"]), row["l2_id"], row["variant_id"])
            if int(row["shard_index"]) != shard or key[0] % SHARD_COUNT != shard:
                raise RunnerError(f"eval shard assignment mismatch: {key}")
            if row["manifest_sha256"] != contract["manifest_sha256"]:
                raise RunnerError("eval manifest identity mismatch")
            if row["checkpoint_set_sha256"] != contract["checkpoint_set_sha256"]:
                raise RunnerError("eval checkpoint-set identity mismatch")
            if key in observed:
                raise RunnerError(f"duplicate eval Cartesian row: {key}")
            observed.add(key)
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        extra = sorted(observed - expected)[:5]
        raise RunnerError(f"eval Cartesian mismatch; missing={missing}, extra={extra}")
    if len(observed) != int(contract["expected_episode_rows"]):
        raise RunnerError("eval row count disagrees with plan")
    return {
        "passed": True,
        "scenario_count": int(contract["expected_scenario_count"]),
        "variant_count": len(variants),
        "episode_rows": len(observed),
    }


def merge_eval(plan_path: Path, plan: RunPlan, dry_run: bool) -> int:
    collection = Path(plan.collection_root)
    if dry_run:
        print(
            _display_command(
                [
                    PINNED_PYTHON,
                    "-m",
                    "bplus_v22.cli",
                    "ppo-merge-eval",
                    "--run-plan",
                    str(plan_path),
                    "--input-root",
                    str(collection),
                    "--output-dir",
                    str(collection / "merged"),
                ]
            )
        )
        return 0
    summary = validate_eval_collection(plan, collection)
    output = collection / "merged"
    if output.exists():
        raise FileExistsError(output)
    local = _host(plan, "local")
    root = Path(local.stage_root)
    argv = [
        local.python,
        "-m",
        "bplus_v22.cli",
        "ppo-merge-eval",
        "--run-plan",
        str(root / "control/run_plan.json"),
        "--input-root",
        str(collection),
        "--output-dir",
        str(output),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": f"{root / 'repo'}:{root / 'repo/f1tenth_gym/gym'}",
        "PYTHONPYCACHEPREFIX": str(root / "cache/pycache/merge"),
    }
    code = subprocess.run(argv, cwd=root / "repo", env=env, check=False).returncode
    if code:
        return code
    if not (output / "COMPLETE").is_file():
        raise RunnerError("merge CLI exited zero without an atomic COMPLETE marker")
    (collection / "control_plane_validation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def _add_plan_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("plan", type=Path)


def _add_host_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--host", choices=("local", "remote"))
    group.add_argument("--all-hosts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _show(plan: RunPlan) -> None:
    print(json.dumps(_plan_to_dict(plan), indent=2, sort_keys=True))
    print("\n# host commands")
    if plan.kind == "b2_train":
        print("# one BC-only 288-row baseline preflight (local, then marker copied remote)")
        print(_display_command(_baseline_command(plan)))
    for host in plan.hosts:
        print(f"# preflight {host.host_id}")
        print(_display_command(_preflight_command(plan, host)))
        print(f"# execute {host.host_id}")
        print(_display_command(_execute_command(plan, host)))
        if plan.kind == "b2_train":
            print(f"# explicit resume {host.host_id}")
            print(_display_command(_execute_command(plan, host, resume=True)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run.sh", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="show the immutable B2 workflow and legacy display entries")
    legacy = sub.add_parser("legacy-show", help="show one non-executable B1 legacy template")
    legacy.add_argument("name", choices=sorted(LEGACY_SHOW_ONLY))

    plan = sub.add_parser("plan", help="create one immutable six-learner B2 plan")
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--source-commit", default="HEAD")
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument("--local-gpu-uuid", required=True)
    plan.add_argument("--remote-gpu-uuid", required=True)

    plan_eval = sub.add_parser("plan-eval", help="freeze six checkpoints into an eval plan")
    plan_eval.add_argument("--run-id", required=True)
    plan_eval.add_argument("--training-plan", required=True, type=Path)
    plan_eval.add_argument("--checkpoint", action="append", required=True)
    plan_eval.add_argument("--output", required=True, type=Path)

    show = sub.add_parser("show", help="verify and print one immutable plan")
    _add_plan_argument(show)
    baseline = sub.add_parser("baseline-preflight")
    _add_plan_argument(baseline)
    baseline.add_argument("--dry-run", action="store_true")
    for action in ("stage", "preflight", "execute", "resume", "status"):
        command = sub.add_parser(action)
        _add_plan_argument(command)
        _add_host_arguments(command)
    collect_parser = sub.add_parser("collect")
    _add_plan_argument(collect_parser)
    collect_parser.add_argument("--dry-run", action="store_true")
    merge = sub.add_parser("merge-eval")
    _add_plan_argument(merge)
    merge.add_argument("--dry-run", action="store_true")

    internal_verify = sub.add_parser("_verify-stage", help=argparse.SUPPRESS)
    _add_plan_argument(internal_verify)
    internal_verify.add_argument("--host", choices=("local", "remote"), required=True)
    internal_preflight = sub.add_parser("_preflight-host", help=argparse.SUPPRESS)
    _add_plan_argument(internal_preflight)
    internal_preflight.add_argument("--host", choices=("local", "remote"), required=True)
    internal_baseline = sub.add_parser("_baseline-host", help=argparse.SUPPRESS)
    _add_plan_argument(internal_baseline)
    internal_baseline.add_argument("--host", choices=("local",), required=True)
    internal_execute = sub.add_parser("_execute-host", help=argparse.SUPPRESS)
    _add_plan_argument(internal_execute)
    internal_execute.add_argument("--host", choices=("local", "remote"), required=True)
    internal_resume = sub.add_parser("_resume-host", help=argparse.SUPPRESS)
    _add_plan_argument(internal_resume)
    internal_resume.add_argument("--host", choices=("local", "remote"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "list":
            print(
                "B2: plan -> show -> stage -> baseline-preflight -> preflight -> execute "
                "[-> explicit resume] -> status -> collect"
            )
            print("B2 eval: plan-eval -> stage -> preflight -> execute -> collect -> merge-eval")
            for name in sorted(LEGACY_SHOW_ONLY):
                print(f"{name} [legacy show-only]")
            return 0
        if args.action == "legacy-show":
            print(LEGACY_SHOW_ONLY[args.name])
            return 0
        if args.action == "plan":
            built = build_training_plan(
                repo=REPO_ROOT,
                run_id=args.run_id,
                commit=args.source_commit,
                output=args.output.resolve(),
                local_gpu_uuid=args.local_gpu_uuid,
                remote_gpu_uuid=args.remote_gpu_uuid,
            )
            print(args.output.resolve())
            print(built.plan_sha256)
            return 0
        if args.action == "plan-eval":
            built = build_evaluation_plan(
                repo=REPO_ROOT,
                run_id=args.run_id,
                training_plan_path=args.training_plan.resolve(),
                checkpoints=args.checkpoint,
                output=args.output.resolve(),
            )
            print(args.output.resolve())
            print(built.plan_sha256)
            return 0
        plan_path = args.plan.resolve()
        loaded = load_plan(plan_path)
        if args.action == "show":
            _show(loaded)
            return 0
        if args.action == "baseline-preflight":
            return baseline_preflight(loaded, args.dry_run)
        if args.action == "stage":
            return stage(plan_path, loaded, _host_ids(args), args.dry_run)
        if args.action == "preflight":
            return preflight(loaded, _host_ids(args), args.dry_run)
        if args.action == "execute":
            return execute(loaded, _host_ids(args), args.dry_run)
        if args.action == "resume":
            return resume(loaded, _host_ids(args), args.dry_run)
        if args.action == "status":
            return status(loaded, _host_ids(args), args.dry_run)
        if args.action == "collect":
            return collect(loaded, args.dry_run)
        if args.action == "merge-eval":
            return merge_eval(plan_path, loaded, args.dry_run)
        if args.action == "_verify-stage":
            host = _host(loaded, args.host)
            _verify_staged_files(loaded, host)
            root = Path(host.stage_root)
            _make_inputs_read_only(root)
            (root / "control/STAGED").write_text(loaded.plan_sha256 + "\n", encoding="utf-8")
            return 0
        if args.action == "_preflight-host":
            return preflight_host(plan_path, args.host)
        if args.action == "_baseline-host":
            return baseline_host(plan_path, args.host)
        if args.action == "_execute-host":
            return execute_host(plan_path, args.host)
        if args.action == "_resume-host":
            return execute_host(plan_path, args.host, resume=True)
        raise RunnerError(f"unhandled action: {args.action}")
    except (RunnerError, FileExistsError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
