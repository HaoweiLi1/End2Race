#!/usr/bin/env python3
"""Immutable control plane for End2Race experiments.

New B2/B3/B4 work never executes from either host's mutable checkout.  A clean,
committed source tree is archived once, explicit runtime inputs are bundled,
and both hosts execute from an isolated run root:

    /home/haowei/end2race_runs/<run_id>

The public workflow is deliberately split into reviewable phases::

    ./run.sh plan ...
    ./run.sh show PLAN
    ./run.sh stage PLAN --all-hosts [--dry-run]
    ./run.sh baseline-preflight PLAN [--dry-run]
    ./run.sh preflight PLAN --all-hosts [--dry-run]
    ./run.sh plumbing-smoke PLAN [--dry-run]
    ./run.sh execute PLAN --all-hosts [--dry-run]
    ./run.sh resume PLAN --host <local|remote> [--dry-run]
    ./run.sh status PLAN --all-hosts [--dry-run]
    ./run.sh collect PLAN [--dry-run]

After training, ``plan-eval`` freezes the six final checkpoints (iteration 20
for historical B2 and iteration 40 for B3).  Eval
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
# Internal control commands execute this file by absolute path from an
# isolated staged tree.  Bind every lazy evidence-module import to that same
# staged source, never to the caller's mutable cwd or an old remote checkout.
try:
    sys.path.remove(str(REPO_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(REPO_ROOT))
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
TRAIN_KINDS = frozenset({"b2_train", "b3_train", "b4_train"})
EVAL_KINDS = frozenset({"b2_eval", "b3_eval", "b4_eval"})
EVAL_CONTROL_ONLY_PATHS = frozenset(
    {
        "Experiments/runner.py",
        "tests/test_experiment_runner.py",
    }
)
BASELINE_SHARD_EXPECTATIONS = {
    0: {"host_id": "local", "collision": 12, "terminal_overtake": 32},
    1: {"host_id": "remote", "collision": 2, "terminal_overtake": 37},
    2: {"host_id": "remote", "collision": 5, "terminal_overtake": 33},
    3: {"host_id": "remote", "collision": 5, "terminal_overtake": 36},
}
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
    "ppo-plumbing-smoke",
)
B4_REQUIRED_TRAIN_CLI = (
    "b4-baseline-preflight",
    "b4-pilot",
    "b4-evaluate",
    "b4-merge-eval",
    "b4-plumbing-smoke",
)
B4_REQUIRED_EVAL_CLI = ("b4-evaluate", "b4-merge-eval")
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
        if job.kind in {"learner", "b4_training"} and (
            job.shardable or not job.gpu_exclusive
        ):
            raise RunnerError("PPO learners must be non-shardable and GPU-exclusive")
        _safe_relative(job.output_relpath)
        _safe_relative(job.numba_cache_relpath)
    for host in plan.hosts:
        _validate_host_root(host, plan.run_id)
        if not host.expected_environment:
            raise RunnerError(f"critical environment contract is empty: {host.host_id}")
    if plan.kind in TRAIN_KINDS:
        expected_topology = {
            str(index): dict(expectation)
            for index, expectation in BASELINE_SHARD_EXPECTATIONS.items()
        }
        if plan.config.get("bc_baseline_topology") != expected_topology:
            raise RunnerError("PPO baseline topology contract drift")
        if plan.kind == "b4_train":
            identities = {
                (job.job_id, job.seed)
                for job in plan.jobs
                if job.kind == "b4_training"
            }
            if identities != {("b4-seed1", 1)} or len(plan.jobs) != 1:
                raise RunnerError("B4 train plan must contain exactly one seed-1 learner")
            if tuple(plan.required_cli) != B4_REQUIRED_TRAIN_CLI:
                raise RunnerError("B4 train CLI contract drift")
            from bplus_v22.b4_direct import B4_POLICY_SCHEMA, validate_frozen_config

            try:
                validate_frozen_config(plan.config.get("ppo", {}))
            except ValueError as error:
                raise RunnerError(str(error)) from error
            curriculum = plan.config.get("curriculum_sha256_by_seed")
            expected_config_keys = {
                "policy_contract",
                "ppo",
                "curriculum_schema",
                "curriculum_sha256_by_seed",
                "training_manifest_sha256",
                "bc_baseline_expected_collision",
                "bc_baseline_expected_overtake",
                "bc_baseline_topology",
                "overtake_gate_per_seed",
                "collision_feasibility_per_seed",
                "collision_product_target_per_seed",
                "collision_product_target_pooled",
                "deterministic_speed_projection_required",
                "inputs",
                "forbidden_inputs",
            }
            if (
                set(plan.config) != expected_config_keys
                or plan.config.get("policy_contract") != B4_POLICY_SCHEMA
                or not isinstance(curriculum, dict)
                or set(curriculum) != {"1"}
                or any(not SHA256_RE.fullmatch(str(value)) for value in curriculum.values())
                or not SHA256_RE.fullmatch(str(plan.config.get("training_manifest_sha256", "")))
                or plan.config.get("overtake_gate_per_seed") != 132
                or plan.config.get("collision_feasibility_per_seed") != 24
                or plan.config.get("collision_product_target_per_seed") != 16
                or plan.config.get("collision_product_target_pooled") != 33
                or plan.config.get("deterministic_speed_projection_required") != 0
            ):
                raise RunnerError("B4 frozen numerical/curriculum config drift")
        else:
            identities = {
                (job.arm, job.seed) for job in plan.jobs if job.kind == "learner"
            }
            expected_identities = {(arm, seed) for arm in ARMS for seed in SEEDS}
            if identities != expected_identities or len(plan.jobs) != 6:
                raise RunnerError(
                    "PPO train plan must contain exactly six arm-by-seed learners"
                )
            if tuple(plan.required_cli) != REQUIRED_TRAIN_CLI:
                raise RunnerError("PPO train CLI contract drift")
            expected_contract = (
                "unified_standard_mode_v1"
                if plan.kind == "b3_train"
                else "centered_fresh_prior"
            )
            if plan.config.get("policy_contract", "centered_fresh_prior") != expected_contract:
                raise RunnerError("PPO plan kind/policy contract mismatch")
    elif plan.kind in EVAL_KINDS:
        if not plan.parent_plan_sha256 or not SHA256_RE.fullmatch(plan.parent_plan_sha256):
            raise RunnerError("PPO eval plan lacks parent plan identity")
        if not plan.evaluation_contract:
            raise RunnerError("PPO eval plan lacks evaluation contract")
        if plan.kind == "b4_eval":
            from bplus_v22.b4_cli import B4_EVAL_CONFIG

            if plan.config != B4_EVAL_CONFIG:
                raise RunnerError("B4 EvalPlan frozen config drift")
            if tuple(plan.required_cli) != B4_REQUIRED_EVAL_CLI:
                raise RunnerError("B4 eval CLI contract drift")
        else:
            expected_contract = (
                "unified_standard_mode_v1"
                if plan.kind == "b3_eval"
                else "centered_fresh_prior"
            )
            expected_iteration = 40 if plan.kind == "b3_eval" else 20
            if (
                plan.config.get("policy_contract", "centered_fresh_prior")
                != expected_contract
                or int(plan.config.get("checkpoint_iteration", expected_iteration))
                != expected_iteration
            ):
                raise RunnerError("PPO eval kind/policy checkpoint contract mismatch")
        contract = plan.evaluation_contract
        scenarios = contract.get("scenarios", [])
        variants = contract.get("variants", [])
        shard_count = int(contract.get("shard_count", -1))
        if shard_count != SHARD_COUNT:
            raise RunnerError("PPO eval shard count drift")
        if int(contract.get("expected_scenario_count", -1)) != len(scenarios):
            raise RunnerError("PPO eval scenario count drift")
        if int(contract.get("expected_episode_rows", -1)) != len(scenarios) * len(variants):
            raise RunnerError("PPO eval Cartesian count drift")
        row_indices = [int(item["row_index"]) for item in scenarios]
        l2_ids = [str(item["l2_id"]) for item in scenarios]
        if row_indices != list(range(len(scenarios))) or len(set(l2_ids)) != len(l2_ids):
            raise RunnerError("PPO eval scenarios must be ordered unique physical rows/L2")
        if any(int(item["shard_index"]) != int(item["row_index"]) % SHARD_COUNT for item in scenarios):
            raise RunnerError("PPO eval scenario assignment drift")
        expected_job_kind = (
            "b4_evaluation_shard" if plan.kind == "b4_eval" else "evaluation_shard"
        )
        shard_jobs = {
            job.shard_index for job in plan.jobs if job.kind == expected_job_kind
        }
        if shard_jobs != set(range(SHARD_COUNT)) or len(plan.jobs) != SHARD_COUNT:
            raise RunnerError("PPO eval plan must contain exactly four shards")
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


def _validate_eval_control_only_source_delta(
    repo: Path, parent_commit: str, eval_commit: str
) -> tuple[str, ...]:
    """Allow a new EvalPlan source only when numerical code is unchanged."""

    records = [
        line.split("\t", 1)
        for line in str(
            _git(
                repo,
                "diff",
                "--name-status",
                "--no-renames",
                f"{parent_commit}..{eval_commit}",
            )
        ).splitlines()
        if line
    ]
    malformed = [record for record in records if len(record) != 2]
    changed = tuple(record[1] for record in records if len(record) == 2)
    disallowed = sorted(
        path
        for status, path in records
        if status != "M" or path not in EVAL_CONTROL_ONLY_PATHS
    )
    if malformed or disallowed:
        raise RunnerError(
            "evaluation source changes numerical/non-control files: "
            f"{disallowed or malformed}"
        )
    return changed


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


def _create_b4_inputs_archive(repo: Path, output: Path) -> tuple[InputEntry, ...]:
    """Bundle only B4's opened Task-8 and D2 metadata inputs; no sidecar."""

    task8 = repo / TASK8_RELEASE
    for required in ("training_scenarios.tsv", "development_scenarios.tsv", "config.json"):
        if not (task8 / required).is_file():
            raise RunnerError(f"canonical Task-8 release lacks {required}")
    _verify_existing_output_manifest(task8)
    metadata = repo / D2_METADATA
    if not metadata.is_file() or _sha256_file(metadata) != D2_METADATA_SHA256:
        raise RunnerError("canonical D2 episode metadata hash mismatch")
    entries: list[InputEntry] = []
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in _release_files(task8):
            entry = _tar_entry(
                archive, path, f"task8/{path.relative_to(task8).as_posix()}"
            )
            entries.append(replace(entry, role="task8_release"))
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
# Python patch releases are intentionally host-local.  B2 requires the same
# language ABI (major.minor) and exact critical package versions on both GPUs;
# a 3.10.x security/bugfix patch difference is not an experimental variable.
result = {"python": ".".join(platform.python_version_tuple()[:2])}
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


def _assert_live_environment(host: HostSpec) -> dict[str, str]:
    actual = _critical_environment(host.python)
    if actual != host.expected_environment:
        raise RunnerError(
            f"critical environment drift for {host.host_id}: "
            f"{actual} != {host.expected_environment}"
        )
    return actual


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


def _b4_training_jobs() -> tuple[tuple[JobSpec, ...], dict[str, tuple[str, ...]]]:
    jobs: list[JobSpec] = []
    queues: dict[str, tuple[str, ...]] = {}
    for seed, host in ((1, "remote"),):
        job_id = f"b4-seed{seed}"
        queue_id = f"b4-seed{seed}-{host}"
        jobs.append(
            JobSpec(
                job_id=job_id,
                kind="b4_training",
                host_id=host,
                queue_id=queue_id,
                argv=("-m", "bplus_v22.cli", "b4-pilot"),
                output_relpath=f"outputs/train/seed{seed}",
                numba_cache_relpath=f"cache/numba/{job_id}",
                seed=seed,
                gpu_exclusive=True,
                shardable=False,
            )
        )
        queues[queue_id] = (job_id,)
    return tuple(jobs), queues


def _shared_training_config(kind: str = "b2_train") -> dict[str, Any]:
    if kind not in TRAIN_KINDS:
        raise RunnerError(f"unsupported training config kind: {kind}")
    common = {
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
        "bc_baseline_topology": {
            str(index): dict(expectation)
            for index, expectation in BASELINE_SHARD_EXPECTATIONS.items()
        },
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
    if kind == "b2_train":
        return common
    common.update(
        {
            "policy_contract": "unified_standard_mode_v1",
            "iterations": 40,
            "deterministic_contract": "standard_mode_of_training_distribution",
            "dual_freeze_through_iteration": 0,
            "exploration": {
                "intervention_prior_probability": 0.10,
                "conditional_brake_prior_probability": 0.50,
                "external_gate_offsets_forbidden": True,
                "steer_std_scale": 0.1,
                "brake_std_scale": 1.0,
            },
        }
    )
    return common


def build_training_plan(
    *,
    repo: Path,
    run_id: str,
    commit: str,
    output: Path,
    local_gpu_uuid: str,
    remote_gpu_uuid: str,
    environment: dict[str, str] | None = None,
    kind: str = "b2_train",
) -> RunPlan:
    if kind not in TRAIN_KINDS:
        raise RunnerError(f"unsupported training plan kind: {kind}")
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
                kind=kind,
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
                config=_shared_training_config(kind),
                collection_root=str(
                    (
                        repo
                        / (
                            "Experiments/B3_ppo_unified/runs"
                            if kind == "b3_train"
                            else "Experiments/B2_ppo_pilot/runs"
                        )
                        / run_id
                    ).resolve()
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


def build_b4_training_plan(
    *,
    repo: Path,
    run_id: str,
    commit: str,
    output: Path,
    local_gpu_uuid: str,
    remote_gpu_uuid: str,
    environment: dict[str, str] | None = None,
) -> RunPlan:
    """Create the owner-approved immutable seed-1 B4 RunPlan."""

    _validate_run_id(run_id)
    commit, tree = _require_clean_commit(repo, commit)
    _source_has_cli_contract(repo, commit, B4_REQUIRED_TRAIN_CLI)
    if output.exists():
        raise FileExistsError(output)
    source_path = output.with_suffix(".source.tar")
    inputs_path = output.with_suffix(".inputs.tar")
    if source_path.exists() or inputs_path.exists():
        raise FileExistsError("B4 control archive already exists")
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
        input_entries = _create_b4_inputs_archive(repo, inputs_partial)
        from bplus_v22.b4_runner import expected_b4_plan_config

        config = expected_b4_plan_config(repo / TASK8_RELEASE, repo / D2_METADATA)
        environment = environment or _critical_environment(PINNED_PYTHON)
        jobs, queues = _b4_training_jobs()
        plan = _seal_plan(
            RunPlan(
                schema=PLAN_SCHEMA,
                run_id=run_id,
                kind="b4_train",
                created_at=_now(),
                source_commit=commit,
                source_tree=tree,
                source_archive_path=str(source_path.resolve()),
                source_archive_sha256=_sha256_file(source_partial),
                source_archive_size=source_partial.stat().st_size,
                inputs_archive_path=str(inputs_path.resolve()),
                inputs_archive_sha256=_sha256_file(inputs_partial),
                inputs_archive_size=inputs_partial.stat().st_size,
                source_inputs=(source_bc,),
                inputs=input_entries,
                hosts=_default_hosts(
                    run_id, local_gpu_uuid, remote_gpu_uuid, environment
                ),
                jobs=jobs,
                queues=queues,
                required_cli=B4_REQUIRED_TRAIN_CLI,
                module_path_contract=MODULE_PATH_CONTRACT,
                config=config,
                collection_root=str(
                    (repo / "Experiments/B4_direct_head_ppo/runs" / run_id).resolve()
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
    final_iteration = int(parent.config.get("iterations", -1))
    expected_name = f"iter_{final_iteration:04d}.pt"
    expected_schema = (
        "bplus-v2.2-b3-ppo-pilot-1"
        if parent.kind == "b3_train"
        else "bplus-v2.2-b2-ppo-pilot-1"
    )
    if path.name != expected_name or path.parent.name != "checkpoints":
        raise RunnerError("evaluation checkpoint must be the final learner checkpoint")
    release = path.parent.parent
    if (
        not (release / "COMPLETE").is_file()
        or release.with_name(release.name + ".partial").exists()
        or not (release / "summary.json").is_file()
    ):
        raise RunnerError("evaluation checkpoint does not come from a COMPLETE learner")
    summary = json.loads((release / "summary.json").read_text(encoding="utf-8"))
    recorded_final_sha = (
        summary.get("final_checkpoint_sha256")
        if parent.kind == "b3_train"
        else summary.get("iteration20_checkpoint_sha256")
    )
    if (
        summary.get("schema") != expected_schema
        or summary.get("integrity_passed") is not True
        or summary.get("passed") is not True
        or summary.get("arm") != arm
        or summary.get("seed") != seed
        or summary.get("iterations") != final_iteration
        or summary.get("run_plan_sha256") != parent.plan_sha256
        or recorded_final_sha != _sha256_file(path)
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


def _b4_eval_jobs() -> tuple[tuple[JobSpec, ...], dict[str, tuple[str, ...]]]:
    jobs: list[JobSpec] = []
    queues: dict[str, tuple[str, ...]] = {
        "b4-eval-local": ("eval-shard0",),
        "b4-eval-remote-sequential": (
            "eval-shard1",
            "eval-shard2",
            "eval-shard3",
        ),
    }
    for shard in range(SHARD_COUNT):
        host = "local" if shard == 0 else "remote"
        queue = (
            "b4-eval-local" if shard == 0 else "b4-eval-remote-sequential"
        )
        job_id = f"eval-shard{shard}"
        jobs.append(
            JobSpec(
                job_id=job_id,
                kind="b4_evaluation_shard",
                host_id=host,
                queue_id=queue,
                argv=("-m", "bplus_v22.cli", "b4-evaluate"),
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
    source_commit: str = "HEAD",
) -> RunPlan:
    _validate_run_id(run_id)
    parent = load_plan(training_plan_path)
    if parent.kind not in {"b2_train", "b3_train"}:
        raise RunnerError("plan-eval supports only B2/B3; use plan-b4-eval for B4")
    source_commit, source_tree = _require_clean_commit(repo, source_commit)
    _source_has_cli_contract(repo, source_commit, ("ppo-evaluate", "ppo-merge-eval"))
    source_delta = _validate_eval_control_only_source_delta(
        repo, parent.source_commit, source_commit
    )
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
    source_path = output.with_suffix(".source.tar")
    inputs_path = output.with_suffix(".inputs.tar")
    source_partial = source_path.with_name(f".{source_path.name}.partial")
    inputs_partial = inputs_path.with_name(f".{inputs_path.name}.partial")
    if any(path.exists() for path in (source_path, inputs_path, source_partial, inputs_partial)):
        raise FileExistsError("evaluation control archive already exists")
    try:
        _create_source_archive(repo, source_commit, source_partial)
        source_bc = _read_source_member(source_partial, "pretrained/end2race.pth")
        if source_bc.sha256 != CANONICAL_BC_SHA256:
            raise RunnerError("committed BC checkpoint hash mismatch")
        temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.eval.", dir=output.parent))
        try:
            _safe_extract(parent_inputs, temporary)
            files: list[tuple[str, str, Path]] = []
            task8 = temporary / "task8"
            for path in _release_files(task8):
                files.append(("task8_release", f"task8/{path.relative_to(task8).as_posix()}", path))
            checkpoint_meta: list[dict[str, Any]] = []
            for arm, seed, path in sorted(parsed):
                final_iteration = int(parent.config["iterations"])
                arcname = f"checkpoints/{arm}_seed{seed}_iter{final_iteration}.pt"
                sha = _sha256_file(path)
                files.append(("checkpoint", arcname, path))
                checkpoint_meta.append(
                    {"arm": arm, "seed": seed, "relpath": f"inputs/{arcname}",
                     "sha256": sha, "size": path.stat().st_size}
                )
            entries = _deterministic_input_archive(inputs_partial, files)
            input_sha = _sha256_file(inputs_partial)
            input_size = inputs_partial.stat().st_size
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
                    kind="b3_eval" if parent.kind == "b3_train" else "b2_eval",
                    created_at=_now(),
                    source_commit=source_commit,
                    source_tree=source_tree,
                    source_archive_path=str(source_path.resolve()),
                    source_archive_sha256=_sha256_file(source_partial),
                    source_archive_size=source_partial.stat().st_size,
                    inputs_archive_path=str(inputs_path.resolve()),
                    inputs_archive_sha256=input_sha,
                    inputs_archive_size=input_size,
                    source_inputs=(source_bc,),
                    inputs=entries,
                    hosts=tuple(replace(host, stage_root=str(ISOLATED_BASE / run_id)) for host in parent.hosts),
                    jobs=jobs,
                    queues=queues,
                    required_cli=("ppo-evaluate", "ppo-merge-eval"),
                    module_path_contract=parent.module_path_contract,
                    config={
                        "evaluation_offsets": [0.0, 0.0],
                        "checkpoint_iteration": int(parent.config["iterations"]),
                        "policy_contract": parent.config.get(
                            "policy_contract", "centered_fresh_prior"
                        ),
                        "parent_training_source_commit": parent.source_commit,
                        "control_only_source_delta": list(source_delta),
                    },
                    collection_root=str(
                        (
                            repo
                            / (
                                "Experiments/B3_ppo_unified/evaluations"
                                if parent.kind == "b3_train"
                                else "Experiments/B2_ppo_pilot/evaluations"
                            )
                            / run_id
                        ).resolve()
                    ),
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
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        os.replace(source_partial, source_path)
        os.replace(inputs_partial, inputs_path)
        try:
            write_plan(output, plan)
        except Exception:
            source_path.unlink(missing_ok=True)
            inputs_path.unlink(missing_ok=True)
            raise
        return plan
    except Exception:
        source_partial.unlink(missing_ok=True)
        inputs_partial.unlink(missing_ok=True)
        raise


def _parse_b4_checkpoint(value: str) -> tuple[int, int, Path]:
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise RunnerError("B4 checkpoint must be SEED,ITERATION,PATH")
    seed = int(parts[0])
    iteration = int(parts[1])
    path = Path(parts[2]).resolve()
    if seed != 1 or iteration not in {10, 20, 30}:
        raise RunnerError("B4 checkpoint seed/iteration is outside the frozen set")
    if not path.is_file() or path.is_symlink():
        raise RunnerError(f"B4 checkpoint is not one regular file: {path}")
    return seed, iteration, path


def _validate_b4_actor_snapshot_source(
    parent: RunPlan, seed: int, iteration: int, path: Path
) -> None:
    expected_name = f"iter_{iteration:04d}.pth"
    if path.name != expected_name or path.parent.name != "actors":
        raise RunnerError("B4 evaluation input is not a frozen actor snapshot")
    release = path.parent.parent
    summary_path = release / "summary.json"
    if (
        not (release / "COMPLETE").is_file()
        or release.with_name(release.name + ".partial").exists()
        or not summary_path.is_file()
    ):
        raise RunnerError("B4 actor snapshot does not come from a COMPLETE learner")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    snapshot_sha = summary.get("actor_snapshot_file_sha256_by_iteration", {}).get(
        str(iteration)
    )
    snapshot_tensor_sha = summary.get(
        "actor_snapshot_tensor_sha256_by_iteration", {}
    ).get(str(iteration))
    actor0_path = release / "actors/iter_0000.pth"
    actor0_file_sha = summary.get(
        "actor_snapshot_file_sha256_by_iteration", {}
    ).get("0")
    actor0_tensor_sha = summary.get(
        "actor_snapshot_tensor_sha256_by_iteration", {}
    ).get("0")
    if (
        summary.get("schema") != "end2race-b4-direct-head-pilot-1"
        or summary.get("integrity_passed") is not True
        or summary.get("passed") is not True
        or summary.get("seed") != seed
        or summary.get("iterations") != 30
        or summary.get("run_plan_sha256") != parent.plan_sha256
        or summary.get("source_commit") != parent.source_commit
        or summary.get("bc_checkpoint_sha256") != CANONICAL_BC_SHA256
        or summary.get("training_manifest_sha256")
        != parent.config["training_manifest_sha256"]
        or summary.get("curriculum_sha256")
        != parent.config["curriculum_sha256_by_seed"][str(seed)]
        or not actor0_path.is_file()
        or actor0_path.is_symlink()
        or actor0_file_sha != _sha256_file(actor0_path)
        or summary.get("bc_actor_tensor_sha256") != actor0_tensor_sha
        or snapshot_sha != _sha256_file(path)
    ):
        raise RunnerError("B4 actor snapshot learner envelope mismatch")
    from bplus_v22.b4_direct import actor_snapshot_sha256, load_strict_plain_actor

    actor0_state = load_strict_plain_actor(actor0_path, "cpu").state_dict()
    actor_state = load_strict_plain_actor(path, "cpu").state_dict()
    if (
        actor_snapshot_sha256(actor0_state) != actor0_tensor_sha
        or actor_snapshot_sha256(actor_state) != snapshot_tensor_sha
        or any(
            not actor_state[name].equal(actor0_state[name])
            for name in actor_state
            if not name.startswith("output_layer.")
        )
    ):
        raise RunnerError("B4 actor snapshot tensor envelope mismatch")


def build_b4_evaluation_plan(
    *,
    repo: Path,
    run_id: str,
    training_plan_path: Path,
    checkpoints: Sequence[str],
    output: Path,
    source_commit: str = "HEAD",
) -> RunPlan:
    """Freeze the seed-1 288x4 B4 opened-development evaluation."""

    _validate_run_id(run_id)
    parent = load_plan(training_plan_path)
    if parent.kind != "b4_train":
        raise RunnerError("B4 evaluation parent must be one b4_train RunPlan")
    source_commit, source_tree = _require_clean_commit(repo, source_commit)
    _source_has_cli_contract(repo, source_commit, B4_REQUIRED_EVAL_CLI)
    source_delta = _validate_eval_control_only_source_delta(
        repo, parent.source_commit, source_commit
    )
    parsed = [_parse_b4_checkpoint(value) for value in checkpoints]
    expected = {(1, iteration) for iteration in (10, 20, 30)}
    identities = {(seed, iteration) for seed, iteration, _ in parsed}
    if len(parsed) != 3 or identities != expected:
        raise RunnerError("B4 evaluation requires seed1 iter10/20/30 snapshots")
    for seed, iteration, path in parsed:
        _validate_b4_actor_snapshot_source(parent, seed, iteration, path)
    parent_inputs = Path(parent.inputs_archive_path)
    if _sha256_file(parent_inputs) != parent.inputs_archive_sha256:
        raise RunnerError("B4 parent input archive drift")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path = output.with_suffix(".source.tar")
    inputs_path = output.with_suffix(".inputs.tar")
    source_partial = source_path.with_name(f".{source_path.name}.partial")
    inputs_partial = inputs_path.with_name(f".{inputs_path.name}.partial")
    if any(path.exists() for path in (source_path, inputs_path, source_partial, inputs_partial)):
        raise FileExistsError("B4 evaluation control archive already exists")
    try:
        _create_source_archive(repo, source_commit, source_partial)
        source_bc = _read_source_member(source_partial, "pretrained/end2race.pth")
        if source_bc.sha256 != CANONICAL_BC_SHA256:
            raise RunnerError("committed BC checkpoint hash mismatch")
        temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.b4eval.", dir=output.parent))
        try:
            _safe_extract(parent_inputs, temporary)
            files: list[tuple[str, str, Path]] = []
            task8 = temporary / "task8"
            for path in _release_files(task8):
                files.append(
                    (
                        "task8_release",
                        f"task8/{path.relative_to(task8).as_posix()}",
                        path,
                    )
                )
            checkpoint_meta: list[dict[str, Any]] = []
            for seed, iteration, path in sorted(parsed):
                arcname = f"checkpoints/seed{seed}_iter{iteration}.pth"
                sha = _sha256_file(path)
                files.append(("plain_end2race_checkpoint", arcname, path))
                checkpoint_meta.append(
                    {
                        "seed": seed,
                        "iteration": iteration,
                        "relpath": f"inputs/{arcname}",
                        "sha256": sha,
                        "size": path.stat().st_size,
                    }
                )
            entries = _deterministic_input_archive(inputs_partial, files)
            task8_manifest = task8 / "development_scenarios.tsv"
            manifest_sha = _sha256_file(task8_manifest)
            with task8_manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            if len(rows) != 288 or any(not row.get("l2_id") for row in rows):
                raise RunnerError("canonical B4 development manifest must contain 288 L2 rows")
            scenarios = [
                {
                    "row_index": index,
                    "l2_id": row["l2_id"],
                    "shard_index": index % SHARD_COUNT,
                }
                for index, row in enumerate(rows)
            ]
            variants = ["BC"] + [
                f"seed{seed}_iter{iteration}"
                for seed in (1,)
                for iteration in (10, 20, 30)
            ]
            checkpoint_set_sha = _sha256_bytes(_canonical_json(checkpoint_meta))
            jobs, queues = _b4_eval_jobs()
            from bplus_v22.b4_cli import B4_EVAL_CONFIG

            plan = _seal_plan(
                RunPlan(
                    schema=PLAN_SCHEMA,
                    run_id=run_id,
                    kind="b4_eval",
                    created_at=_now(),
                    source_commit=source_commit,
                    source_tree=source_tree,
                    source_archive_path=str(source_path.resolve()),
                    source_archive_sha256=_sha256_file(source_partial),
                    source_archive_size=source_partial.stat().st_size,
                    inputs_archive_path=str(inputs_path.resolve()),
                    inputs_archive_sha256=_sha256_file(inputs_partial),
                    inputs_archive_size=inputs_partial.stat().st_size,
                    source_inputs=(source_bc,),
                    inputs=entries,
                    hosts=tuple(
                        replace(host, stage_root=str(ISOLATED_BASE / run_id))
                        for host in parent.hosts
                    ),
                    jobs=jobs,
                    queues=queues,
                    required_cli=B4_REQUIRED_EVAL_CLI,
                    module_path_contract=parent.module_path_contract,
                    config=dict(B4_EVAL_CONFIG),
                    collection_root=str(
                        (
                            repo
                            / "Experiments/B4_direct_head_ppo/evaluations"
                            / run_id
                        ).resolve()
                    ),
                    parent_plan_sha256=parent.plan_sha256,
                    evaluation_contract={
                        "manifest_relpath": "inputs/task8/development_scenarios.tsv",
                        "manifest_sha256": manifest_sha,
                        "checkpoint_set": checkpoint_meta,
                        "checkpoint_set_sha256": checkpoint_set_sha,
                        "training_manifest_sha256": _sha256_file(
                            task8 / "training_scenarios.tsv"
                        ),
                        "parent_training_source_commit": parent.source_commit,
                        "control_only_source_delta": list(source_delta),
                        "shard_count": SHARD_COUNT,
                        "assignment": "physical_row_index_mod_shard_count",
                        "scenarios": scenarios,
                        "variants": variants,
                        "expected_scenario_count": 288,
                        "expected_variant_count": 4,
                        "expected_episode_rows": 1152,
                    },
                )
            )
            _verify_plan(plan)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        os.replace(source_partial, source_path)
        os.replace(inputs_partial, inputs_path)
        try:
            write_plan(output, plan)
        except Exception:
            source_path.unlink(missing_ok=True)
            inputs_path.unlink(missing_ok=True)
            raise
        return plan
    except Exception:
        source_partial.unlink(missing_ok=True)
        inputs_partial.unlink(missing_ok=True)
        raise


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
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=6",
        host.ssh_host,
        _display_command(remote_argv),
    ]


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
        f"set -eu; umask 077; test ! -e {shlex.quote(root)}; "
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
        # The verification process imports staged modules before it makes the
        # source tree read-only.  Suppress bytecode at that one bootstrap seam
        # so the verifier cannot invalidate its own tracked-file inventory.
        f"PYTHONDONTWRITEBYTECODE=1 {shlex.quote(host.python)} "
        f"{shlex.quote(root + '/repo/Experiments/runner.py')} "
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
    if any(path.is_symlink() for path in (root / "inputs").rglob("*")):
        raise RunnerError("symlink appeared in staged runtime inputs")
    expected_inputs = {entry.relpath for entry in plan.inputs}
    actual_inputs = {
        path.relative_to(root / "inputs").as_posix()
        for path in (root / "inputs").rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_inputs != expected_inputs:
        raise RunnerError("staged runtime input inventory drift")


def _verify_extracted_source_tree(root: Path) -> None:
    """Bind every extracted tracked source byte to the already verified source tar."""

    archive_path = root / "control/source.tar"
    expected: dict[str, tuple[int, str]] = {}
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            relpath = _safe_relative(member.name).as_posix()
            if not member.isfile() or relpath in expected:
                raise RunnerError(f"unsupported/duplicate staged source member: {relpath}")
            handle = archive.extractfile(member)
            if handle is None:
                raise RunnerError(f"cannot hash staged source member: {relpath}")
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            expected[relpath] = (size, digest.hexdigest())
    repo = root / "repo"
    if any(path.is_symlink() for path in repo.rglob("*")):
        raise RunnerError("symlink appeared in staged extracted source")
    actual_paths = {
        path.relative_to(repo).as_posix(): path
        for path in repo.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if set(actual_paths) != set(expected):
        raise RunnerError("staged extracted source inventory drift")
    for relpath, path in actual_paths.items():
        size, digest = expected[relpath]
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise RunnerError(f"staged extracted source digest drift: {relpath}")


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


def _stage_check_command(plan: RunPlan, host: HostSpec) -> list[str]:
    root = PurePosixPath(host.stage_root)
    argv = [
        host.python,
        str(root / "repo/Experiments/runner.py"),
        "_check-stage-host",
        str(root / "control/run_plan.json"),
        "--host",
        host.host_id,
    ]
    return argv if host.kind == "local" else _ssh_argv(host, argv)


def _preflight_check_command(plan: RunPlan, host: HostSpec) -> list[str]:
    root = PurePosixPath(host.stage_root)
    argv = [
        host.python,
        str(root / "repo/Experiments/runner.py"),
        "_check-preflight-host",
        str(root / "control/run_plan.json"),
        "--host",
        host.host_id,
    ]
    return argv if host.kind == "local" else _ssh_argv(host, argv)


def _baseline_command(plan: RunPlan, host: HostSpec) -> list[str]:
    root = PurePosixPath(host.stage_root)
    inner = [
        host.python,
        str(root / "repo/Experiments/runner.py"),
        "_baseline-host",
        str(root / "control/run_plan.json"),
        "--host",
        host.host_id,
    ]
    argv = ["flock", "-n", _lock_path(host), *inner]
    return argv if host.kind == "local" else _ssh_argv(host, argv)


def _baseline_shard_path(root: str | Path, shard_index: int) -> Path:
    if shard_index not in range(SHARD_COUNT):
        raise RunnerError(f"invalid BC baseline shard index: {shard_index}")
    return Path(root) / "control/baseline_shards" / f"shard_{shard_index}.json"


def _baseline_expected_producers(plan: RunPlan) -> dict[int, tuple[str, str]]:
    from bplus_v22.ppo_eval import (
        BASELINE_SHARD_COUNT,
        EXPECTED_BC_COLLISIONS_BY_SHARD,
        EXPECTED_BC_OVERTAKES_BY_SHARD,
    )

    ordered = [BASELINE_SHARD_EXPECTATIONS[index] for index in range(SHARD_COUNT)]
    if (
        BASELINE_SHARD_COUNT != SHARD_COUNT
        or tuple(item["collision"] for item in ordered)
        != EXPECTED_BC_COLLISIONS_BY_SHARD
        or tuple(item["terminal_overtake"] for item in ordered)
        != EXPECTED_BC_OVERTAKES_BY_SHARD
    ):
        raise RunnerError("runner/evaluator BC baseline acceptance contract drift")
    result: dict[int, tuple[str, str]] = {}
    for index, expectation in BASELINE_SHARD_EXPECTATIONS.items():
        host = _host(plan, expectation["host_id"])
        result[index] = (host.host_id, host.gpu_uuid)
    return result


def _validate_baseline_shard(plan: RunPlan, path: Path, shard_index: int) -> dict[str, Any]:
    from bplus_v22.ppo_eval import validate_bc_baseline_shard

    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise RunnerError(f"BC baseline shard is missing or unsafe: {shard_index}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        shard = validate_bc_baseline_shard(value)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise RunnerError(f"BC baseline shard is invalid: {shard_index}: {error}") from error
    expected_host, expected_gpu = _baseline_expected_producers(plan)[shard_index]
    expected_manifest = _frozen_entry_sha256(plan, "task8/development_scenarios.tsv")
    expected_bc = _frozen_entry_sha256(plan, "pretrained/end2race.pth", source=True)
    if (
        shard.shard_index != shard_index
        or shard.shard_count != SHARD_COUNT
        or shard.run_plan_sha256 != plan.plan_sha256
        or shard.source_commit != plan.source_commit
        or shard.source_archive_sha256 != plan.source_archive_sha256
        or shard.inputs_archive_sha256 != plan.inputs_archive_sha256
        or shard.scenario_manifest_sha256 != expected_manifest
        or shard.bc_checkpoint_sha256 != expected_bc
        or (shard.producer_host_id, shard.producer_gpu_uuid)
        != (expected_host, expected_gpu)
    ):
        raise RunnerError(f"BC baseline shard binding mismatch: {shard_index}")
    return value


def _baseline_pull_commands(plan: RunPlan) -> list[list[str]]:
    local = _host(plan, "local")
    remote = _host(plan, "remote")
    root = PurePosixPath(local.stage_root)
    commands: list[list[str]] = []
    for shard_index in (1, 2, 3):
        incoming = root / "control/baseline_shards" / f".shard_{shard_index}.incoming.json"
        commands.append(
            [
                "rsync",
                "-a",
                "--protect-args",
                f"{remote.ssh_host}:{remote.stage_root}/control/baseline_shards/"
                f"shard_{shard_index}.json",
                str(incoming),
            ]
        )
        commands.append(
            [
                local.python,
                str(root / "repo/Experiments/runner.py"),
                "_install-baseline-shard",
                str(root / "control/run_plan.json"),
                "--host",
                "local",
                "--shard-index",
                str(shard_index),
                "--source",
                str(incoming),
            ]
        )
    return commands


def _baseline_merge_command(plan: RunPlan) -> list[str]:
    local = _host(plan, "local")
    root = PurePosixPath(local.stage_root)
    return [
        local.python,
        str(root / "repo/Experiments/runner.py"),
        "_merge-baseline-host",
        str(root / "control/run_plan.json"),
        "--host",
        "local",
    ]


def _frozen_entry_sha256(
    plan: RunPlan, relpath: str, *, source: bool = False
) -> str:
    entries = plan.source_inputs if source else plan.inputs
    matches = [entry.sha256 for entry in entries if entry.relpath == relpath]
    if len(matches) != 1 or not SHA256_RE.fullmatch(matches[0]):
        raise RunnerError(f"run plan lacks one frozen input identity: {relpath}")
    return matches[0]


def _marker_manifest_path(root: Path, filename: str) -> Path:
    candidates = (
        root / "inputs/task8" / filename,
        root / "control/input_contract" / filename,
    )
    matches = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(matches) != 1:
        raise RunnerError(f"marker validation lacks one trusted Task-8 manifest: {filename}")
    return matches[0]


def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _lexists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _quarantine_uncommitted_marker(path: Path) -> Path | None:
    partial = path.with_suffix(path.suffix + ".partial")
    if not _lexists(partial):
        return None
    if partial.is_symlink() or not partial.is_file():
        raise RunnerError(f"unsafe marker partial: {partial}")
    base = path.parent / "attempt_failures" / path.stem
    base.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (base / f"attempt_{attempt:03d}").exists():
        attempt += 1
    target = base / f"attempt_{attempt:03d}"
    target.mkdir()
    os.replace(partial, target / partial.name)
    (target / "reason.json").write_text(
        json.dumps(
            {
                "schema": "end2race-uncommitted-marker-attempt-1",
                "marker": path.name,
                "quarantined_at": _now(),
                "reason": "uncommitted_partial_found_before_explicit_retry",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def _marker_transfer_commands(
    plan: RunPlan, local_marker: Path, remote: HostSpec, kind: str
) -> list[list[str]]:
    if kind not in {"baseline", "plumbing", "ready"}:
        raise RunnerError(f"unsupported marker transfer kind: {kind}")
    root = PurePosixPath(remote.stage_root)
    incoming = root / "control" / f".{kind}.{plan.plan_sha256}.incoming.json"
    install = [
        remote.python,
        str(root / "repo/Experiments/runner.py"),
        "_install-marker",
        str(root / "control/run_plan.json"),
        "--host",
        "remote",
        "--kind",
        kind,
        "--source",
        str(incoming),
    ]
    return [
        [
            "rsync",
            "-a",
            "--protect-args",
            str(local_marker),
            f"{remote.ssh_host}:{incoming}",
        ],
        _ssh_argv(remote, install),
    ]


def baseline_preflight(plan: RunPlan, dry_run: bool) -> int:
    if plan.kind not in TRAIN_KINDS:
        raise RunnerError("BC baseline preflight requires a B2 training plan")
    local = _host(plan, "local")
    remote = _host(plan, "remote")
    local_marker = Path(local.stage_root) / "control/bc_baseline_preflight.json"
    failed_marker = Path(local.stage_root) / "control/bc_baseline_preflight.failed.json"
    stage_commands: list[list[str]] = [
        _stage_check_command(plan, local),
        _stage_check_command(plan, remote),
    ]
    for command in stage_commands:
        code = _run_command(command, dry_run=dry_run)
        if code:
            return code
    if not dry_run and _lexists(failed_marker):
        _validate_baseline_marker(plan, Path(local.stage_root), failed_marker, require_pass=False)
        print(f"terminal BC baseline acceptance failure: {failed_marker}", file=sys.stderr)
        return 2
    if not dry_run and _lexists(local_marker):
        _validate_baseline_marker(plan, Path(local.stage_root))
        transfer = _marker_transfer_commands(plan, local_marker, remote, "baseline")
        for command in transfer:
            code = _run_command(command, dry_run=False)
            if code:
                return code
        return 0

    host_commands = [_baseline_command(plan, local), _baseline_command(plan, remote)]
    for command in host_commands:
        print(_display_command(command))
    pull_commands = _baseline_pull_commands(plan)
    merge_command = _baseline_merge_command(plan)
    transfer_commands = _marker_transfer_commands(plan, local_marker, remote, "baseline")
    if dry_run:
        for command in (*pull_commands, merge_command, *transfer_commands):
            print(_display_command(command))
        return 0

    running = [subprocess.Popen(command) for command in host_commands]
    failures = [process.wait() for process in running]
    if any(code != 0 for code in failures):
        return 1
    for command in pull_commands:
        code = _run_command(command, dry_run=dry_run)
        if code:
            return code
    code = _run_command(merge_command, dry_run=False)
    if code:
        return code
    _validate_baseline_marker(plan, Path(local.stage_root))
    for command in transfer_commands:
        code = _run_command(command, dry_run=False)
        if code:
            return code
    return 0


def _validate_baseline_marker(
    plan: RunPlan,
    root: Path,
    marker_path: Path | None = None,
    *,
    require_pass: bool = True,
) -> dict[str, Any]:
    from bplus_v22.ppo_eval import (
        baseline_json_bytes,
        merge_bc_baseline_shards,
    )

    path = marker_path or root / "control/bc_baseline_preflight.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise RunnerError("BC baseline preflight marker is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerError("BC baseline preflight marker is unreadable") from error
    manifest = _marker_manifest_path(root, "development_scenarios.tsv")
    expected_manifest_sha = _frozen_entry_sha256(
        plan, "task8/development_scenarios.tsv"
    )
    expected_bc_sha = _frozen_entry_sha256(
        plan, "pretrained/end2race.pth", source=True
    )
    if (
        not isinstance(value, dict)
        or value.get("run_plan_sha256") != plan.plan_sha256
        or value.get("source_commit") != plan.source_commit
        or value.get("source_archive_sha256") != plan.source_archive_sha256
        or value.get("inputs_archive_sha256") != plan.inputs_archive_sha256
        or value.get("scenario_manifest_sha256") != expected_manifest_sha
        or _sha256_file(manifest) != expected_manifest_sha
        or value.get("bc_checkpoint_sha256") != expected_bc_sha
        or expected_bc_sha != CANONICAL_BC_SHA256
    ):
        raise RunnerError("BC baseline preflight marker/envelope mismatch")
    manifest_rows = _read_tsv_rows(manifest)
    if len(manifest_rows) != 288:
        raise RunnerError("BC baseline preflight row inventory mismatch")
    rows = value.get("rows")
    shard_summaries = value.get("shards")
    if not isinstance(rows, list) or not isinstance(shard_summaries, list):
        raise RunnerError("BC baseline preflight rows/shards are missing")
    summaries = {
        item.get("shard_index"): item
        for item in shard_summaries
        if isinstance(item, dict) and type(item.get("shard_index")) is int
    }
    if set(summaries) != set(range(SHARD_COUNT)) or len(shard_summaries) != SHARD_COUNT:
        raise RunnerError("BC baseline preflight shard summary inventory mismatch")
    serialized_shards: list[dict[str, Any]] = []
    for shard_index in range(SHARD_COUNT):
        summary = summaries[shard_index]
        raw_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict) or row.get("baseline_shard_index") != shard_index:
                continue
            raw = dict(row)
            raw.pop("baseline_shard_index", None)
            raw.pop("producer_host_id", None)
            raw.pop("producer_gpu_uuid", None)
            raw_rows.append(raw)
        serialized_shards.append(
            {
                "schema": "bplus-v2.2-b2-bc-baseline-shard-1",
                "shard_index": shard_index,
                "shard_count": SHARD_COUNT,
                "run_plan_sha256": plan.plan_sha256,
                "source_commit": plan.source_commit,
                "source_archive_sha256": plan.source_archive_sha256,
                "inputs_archive_sha256": plan.inputs_archive_sha256,
                "scenario_manifest_sha256": expected_manifest_sha,
                "bc_checkpoint_sha256": expected_bc_sha,
                "producer_host_id": summary.get("producer_host_id"),
                "producer_gpu_uuid": summary.get("producer_gpu_uuid"),
                "opened_development_only": True,
                "candidate_evaluated": False,
                "scenario_count": len(raw_rows),
                "collision": summary.get("collision"),
                "terminal_overtake": summary.get("terminal_overtake"),
                "rows": raw_rows,
            }
        )
    try:
        recomputed = merge_bc_baseline_shards(
            shards=serialized_shards,
            task8_rows=manifest_rows,
            run_plan_sha256=plan.plan_sha256,
            source_commit=plan.source_commit,
            source_archive_sha256=plan.source_archive_sha256,
            inputs_archive_sha256=plan.inputs_archive_sha256,
            scenario_manifest_sha256=expected_manifest_sha,
            bc_checkpoint_sha256=expected_bc_sha,
            expected_producers=_baseline_expected_producers(plan),
        )
    except (ValueError, TypeError) as error:
        raise RunnerError(f"BC baseline preflight semantic validation failed: {error}") from error
    if baseline_json_bytes(recomputed) != baseline_json_bytes(value):
        raise RunnerError("BC baseline preflight canonical serialization drift")
    if require_pass and (
        value.get("integrity_passed") is not True
        or value.get("passed") is not True
        or value.get("acceptance_passed") is not True
    ):
        raise RunnerError("BC baseline preflight acceptance failed")
    if not require_pass and (
        value.get("integrity_passed") is not True
        or value.get("passed") is not False
        or value.get("acceptance_passed") is not False
    ):
        raise RunnerError("BC baseline failure marker semantics mismatch")
    return value


def baseline_host(plan_path: Path, host_id: str) -> int:
    plan = load_plan(plan_path)
    host = _host(plan, host_id)
    check_stage_host(plan_path, host_id)
    subprocess.run(["xdpyinfo", "-display", host.display], check=True, capture_output=True)
    _probe_gpu(host)
    _assert_live_environment(host)
    root = Path(host.stage_root)
    shard_indices = [
        index
        for index, expectation in BASELINE_SHARD_EXPECTATIONS.items()
        if expectation["host_id"] == host_id
    ]
    if not shard_indices:
        raise RunnerError(f"host has no BC baseline shards: {host_id}")
    shard_dir = root / "control/baseline_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for shard_index in shard_indices:
        output = _baseline_shard_path(root, shard_index)
        if _lexists(output):
            _validate_baseline_shard(plan, output, shard_index)
            continue
        _quarantine_uncommitted_marker(output)
        probe_job = JobSpec(
            f"bc-baseline-shard{shard_index}",
            "preflight",
            host.host_id,
            "preflight",
            tuple(),
            "outputs/preflight",
            f"cache/numba/bc-baseline-shard{shard_index}",
            gpu_exclusive=True,
        )
        (root / probe_job.numba_cache_relpath).mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **_job_environment(plan, host, probe_job)}
        subprocess.run(
            [
                host.python,
                "-m",
                "bplus_v22.cli",
                (
                    "b4-baseline-preflight"
                    if plan.kind == "b4_train"
                    else "ppo-baseline-preflight"
                ),
                "--run-plan",
                str(plan_path),
                "--output",
                str(output),
                "--host-id",
                host.host_id,
                "--gpu-uuid",
                host.gpu_uuid,
                "--shard-index",
                str(shard_index),
                "--shard-count",
                str(SHARD_COUNT),
            ],
            check=True,
            cwd=root / "repo",
            env=env,
        )
        _validate_baseline_shard(plan, output, shard_index)
    return 0


def install_baseline_shard(
    plan_path: Path, host_id: str, shard_index: int, source: Path
) -> int:
    if host_id != "local" or shard_index not in (1, 2, 3):
        raise RunnerError("only remote baseline shards may be installed locally")
    plan = load_plan(plan_path)
    root = Path(_host(plan, host_id).stage_root)
    check_stage_host(plan_path, host_id)
    control = (root / "control").resolve()
    if source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1:
        raise RunnerError("incoming BC baseline shard is not one private regular file")
    source = source.resolve()
    try:
        source.relative_to(control)
    except ValueError as error:
        raise RunnerError("incoming BC baseline shard escaped control directory") from error
    _validate_baseline_shard(plan, source, shard_index)
    destination = _baseline_shard_path(root, shard_index)
    if source == destination.resolve():
        raise RunnerError("incoming BC baseline shard aliases destination")
    if _lexists(destination):
        _validate_baseline_shard(plan, destination, shard_index)
        if destination.stat().st_nlink != 1 or os.path.samefile(source, destination):
            raise RunnerError("published BC baseline shard has an unsafe hardlink")
        if _sha256_file(destination) != _sha256_file(source):
            raise RunnerError("published BC baseline shard differs from incoming evidence")
        source.unlink()
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    private = destination.parent / f".shard_{shard_index}.{os.getpid()}.validated.partial"
    if _lexists(private):
        raise RunnerError("private BC baseline shard install path already exists")
    try:
        with source.open("rb") as read_handle, private.open("xb") as write_handle:
            shutil.copyfileobj(read_handle, write_handle)
            write_handle.flush()
            os.fsync(write_handle.fileno())
        _validate_baseline_shard(plan, private, shard_index)
        os.chmod(private, 0o444)
        os.replace(private, destination)
        source.unlink()
    finally:
        if _lexists(private):
            private.unlink()
    _validate_baseline_shard(plan, destination, shard_index)
    return 0


def merge_baseline_host(plan_path: Path, host_id: str) -> int:
    from bplus_v22.ppo_eval import baseline_json_bytes, merge_bc_baseline_shards

    if host_id != "local":
        raise RunnerError("BC baseline merge runs only on the controller host")
    plan = load_plan(plan_path)
    root = Path(_host(plan, host_id).stage_root)
    check_stage_host(plan_path, host_id)
    canonical = root / "control/bc_baseline_preflight.json"
    failed = root / "control/bc_baseline_preflight.failed.json"
    if _lexists(failed):
        _validate_baseline_marker(plan, root, failed, require_pass=False)
        return 3
    if _lexists(canonical):
        _validate_baseline_marker(plan, root, canonical)
        return 0
    shard_values = [
        _validate_baseline_shard(plan, _baseline_shard_path(root, index), index)
        for index in range(SHARD_COUNT)
    ]
    manifest = _marker_manifest_path(root, "development_scenarios.tsv")
    merged = merge_bc_baseline_shards(
        shards=shard_values,
        task8_rows=_read_tsv_rows(manifest),
        run_plan_sha256=plan.plan_sha256,
        source_commit=plan.source_commit,
        source_archive_sha256=plan.source_archive_sha256,
        inputs_archive_sha256=plan.inputs_archive_sha256,
        scenario_manifest_sha256=_sha256_file(manifest),
        bc_checkpoint_sha256=_frozen_entry_sha256(
            plan, "pretrained/end2race.pth", source=True
        ),
        expected_producers=_baseline_expected_producers(plan),
    )
    destination = canonical if merged["passed"] is True else failed
    partial = destination.with_suffix(destination.suffix + ".partial")
    if _lexists(destination) or _lexists(partial):
        raise RunnerError("BC baseline merge destination already exists")
    with partial.open("xb") as handle:
        handle.write(baseline_json_bytes(merged))
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(partial, 0o444)
    os.replace(partial, destination)
    _validate_baseline_marker(
        plan, root, destination, require_pass=merged["passed"] is True
    )
    return 0 if merged["passed"] is True else 3


def _plumbing_command(plan: RunPlan) -> list[str]:
    host = _host(plan, "local")
    root = PurePosixPath(host.stage_root)
    inner = [
        host.python,
        str(root / "repo/Experiments/runner.py"),
        "_plumbing-host",
        str(root / "control/run_plan.json"),
        "--host",
        "local",
    ]
    return ["flock", "-n", _lock_path(host), *inner]


def _ready_command(plan: RunPlan) -> list[str]:
    host = _host(plan, "local")
    root = PurePosixPath(host.stage_root)
    return [
        host.python,
        str(root / "repo/Experiments/runner.py"),
        "_ready-host",
        str(root / "control/run_plan.json"),
        "--host",
        "local",
    ]


def plumbing_smoke(plan: RunPlan, dry_run: bool) -> int:
    if plan.kind not in TRAIN_KINDS:
        raise RunnerError("plumbing smoke requires a B2 training plan")
    local = _host(plan, "local")
    remote = _host(plan, "remote")
    local_marker = Path(local.stage_root) / "control/plumbing_smoke.json"
    commands: list[list[str]] = [
        _preflight_check_command(plan, local),
        _preflight_check_command(plan, remote),
    ]
    if dry_run or not _lexists(local_marker):
        if not dry_run:
            _quarantine_uncommitted_marker(local_marker)
        commands.append(_plumbing_command(plan))
    else:
        _validate_plumbing_marker(plan, Path(local.stage_root))
    commands.extend(_marker_transfer_commands(plan, local_marker, remote, "plumbing"))
    ready_marker = Path(local.stage_root) / "control/READY.json"
    commands.append(_ready_command(plan))
    commands.extend(_marker_transfer_commands(plan, ready_marker, remote, "ready"))
    for command in commands:
        code = _run_command(command, dry_run=dry_run)
        if code:
            return code
    return 0


def _validate_plumbing_marker(
    plan: RunPlan, root: Path, marker_path: Path | None = None
) -> dict[str, Any]:
    path = marker_path or root / "control/plumbing_smoke.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise RunnerError("B2 plumbing smoke marker is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if plan.kind == "b4_train":
        expected_keys = {
            "schema",
            "passed",
            "run_plan_sha256",
            "source_commit",
            "training_manifest_sha256",
            "bc_checkpoint_sha256",
            "d2_episode_metadata_sha256",
            "scenario_selection",
            "map_reports",
            "plain_actor_key_count",
            "trainable_actor_parameter_count",
            "stochastic_plumbing",
            "product_outcomes_reported_or_compared",
            "candidate_selection_performed",
            "ppo_pilot_iteration_completed",
        }
        reports = value.get("map_reports") if isinstance(value, dict) else None
        report_keys = {
            "map_name",
            "l2_id",
            "step_count",
            "terminal_reason",
            "trajectory_identity",
            "outcome_identity",
            "speed_projection_count",
            "steer_projection_count",
            "max_abs_replayed_log_prob_delta",
        }
        stochastic = value.get("stochastic_plumbing") if isinstance(value, dict) else None
        stochastic_keys = {
            "fixed_rng_seed",
            "episode_reports",
            "sampled_transition_count",
            "raw_stored_latent_exact",
            "raw_old_log_prob_exact",
            "projection_ledger_valid",
            "terminal_reward_ledger_valid",
            "dense_reward_sentinel",
            "dense_reward_excluded_from_reward_advantage_return",
            "preupdate_max_abs_ratio_minus_one",
            "actor_early_stop_exercised",
            "actor_epochs_completed",
            "critic_epochs_completed",
            "output_layer_changed",
            "critic_changed",
            "frozen_actor_exact",
            "fixed_action_std_exact",
            "actor_snapshot_key_count",
            "plain_actor_strict_load",
            "full_checkpoint_recovery",
            "product_metrics_compared",
        }
        stochastic_episode_keys = {
            "l2_id",
            "archived_bc_outcome",
            "map_name",
            "step_count",
            "terminal_reason",
            "collision_any",
            "corrected_outcome3",
            "terminal_reward",
            "classifier_parity",
            "zero_bootstrap_terminal",
            "projection_transition_count",
        }
        stochastic_episodes = (
            stochastic.get("episode_reports") if isinstance(stochastic, dict) else None
        )
        stochastic_valid = (
            isinstance(stochastic, dict)
            and set(stochastic) == stochastic_keys
            and stochastic.get("fixed_rng_seed") == 20260714
            and isinstance(stochastic_episodes, list)
            and len(stochastic_episodes) == 3
            and all(
                isinstance(report, dict)
                and set(report) == stochastic_episode_keys
                and isinstance(report.get("l2_id"), str)
                and report["l2_id"].startswith("L2:")
                and isinstance(report.get("map_name"), str)
                and type(report.get("step_count")) is int
                and report["step_count"] > 0
                and type(report.get("collision_any")) is bool
                and isinstance(report.get("corrected_outcome3"), str)
                and type(report.get("terminal_reward")) in (int, float)
                and report.get("classifier_parity") is True
                and report.get("zero_bootstrap_terminal") is True
                and type(report.get("projection_transition_count")) is int
                and report["projection_transition_count"] >= 0
                for report in stochastic_episodes
            )
            and [report["archived_bc_outcome"] for report in stochastic_episodes]
            == ["collision", "follow", "overtake"]
            and [report["terminal_reason"] for report in stochastic_episodes]
            == ["any_agent_collision", "product_horizon", "product_horizon"]
            and [report["terminal_reward"] for report in stochastic_episodes]
            == [-2.0, 0.0, 1.0]
            and stochastic.get("sampled_transition_count")
            == sum(report["step_count"] for report in stochastic_episodes)
            and stochastic.get("raw_stored_latent_exact") is True
            and stochastic.get("raw_old_log_prob_exact") is True
            and stochastic.get("projection_ledger_valid") is True
            and stochastic.get("terminal_reward_ledger_valid") is True
            and stochastic.get("dense_reward_sentinel") == 1_000_000.0
            and stochastic.get("dense_reward_excluded_from_reward_advantage_return")
            is True
            and type(stochastic.get("preupdate_max_abs_ratio_minus_one"))
            in (int, float)
            and 0.0
            <= float(stochastic["preupdate_max_abs_ratio_minus_one"])
            <= 1e-4
            and stochastic.get("actor_early_stop_exercised") is True
            and type(stochastic.get("actor_epochs_completed")) is int
            and 1 <= stochastic["actor_epochs_completed"] < 3
            and stochastic.get("critic_epochs_completed") == 3
            and stochastic.get("output_layer_changed") is True
            and stochastic.get("critic_changed") is True
            and stochastic.get("frozen_actor_exact") is True
            and stochastic.get("fixed_action_std_exact") is True
            and stochastic.get("actor_snapshot_key_count") == 12
            and stochastic.get("plain_actor_strict_load") is True
            and stochastic.get("full_checkpoint_recovery") is True
            and stochastic.get("product_metrics_compared") is False
        )
        expected_manifest_sha = _frozen_entry_sha256(
            plan, "task8/training_scenarios.tsv"
        )
        expected_bc_sha = _frozen_entry_sha256(
            plan, "pretrained/end2race.pth", source=True
        )
        expected_metadata_sha = _frozen_entry_sha256(
            plan, "d2/episode_metadata.tsv"
        )
        expected_maps = (
            "Austin",
            "Hockenheim",
            "MoscowRaceway",
            "Nuerburgring",
        )
        training_rows = _read_tsv_rows(
            _marker_manifest_path(root, "training_scenarios.tsv")
        )
        first_l2_by_map: dict[str, str] = {}
        for physical_index, row in enumerate(training_rows):
            if int(row.get("training_order", -1)) != physical_index:
                raise RunnerError("B4 plumbing training manifest order drift")
            first_l2_by_map.setdefault(str(row.get("map_name", "")), str(row.get("l2_id", "")))
        reports_valid = (
            isinstance(reports, list)
            and len(reports) == 4
            and tuple(report.get("map_name") for report in reports) == expected_maps
            and all(
                report.get("l2_id") == first_l2_by_map.get(str(report.get("map_name")))
                for report in reports
            )
            and all(
                isinstance(report, dict)
                and set(report) == report_keys
                and isinstance(report.get("l2_id"), str)
                and report["l2_id"].startswith("L2:")
                and type(report.get("step_count")) is int
                and report["step_count"] > 0
                and report.get("terminal_reason")
                in {"any_agent_collision", "product_horizon"}
                and report.get("trajectory_identity") is True
                and report.get("outcome_identity") is True
                and report.get("speed_projection_count") == 0
                and type(report.get("steer_projection_count")) is int
                and report["steer_projection_count"] >= 0
                and type(report.get("max_abs_replayed_log_prob_delta"))
                in (int, float)
                and 0.0
                <= float(report["max_abs_replayed_log_prob_delta"])
                <= 1e-4
                for report in reports
            )
        )
        if (
            not isinstance(value, dict)
            or set(value) != expected_keys
            or value.get("schema") != "end2race-b4-plumbing-smoke-2"
            or value.get("passed") is not True
            or value.get("run_plan_sha256") != plan.plan_sha256
            or value.get("source_commit") != plan.source_commit
            or value.get("training_manifest_sha256") != expected_manifest_sha
            or value.get("bc_checkpoint_sha256") != expected_bc_sha
            or expected_bc_sha != CANONICAL_BC_SHA256
            or value.get("d2_episode_metadata_sha256") != expected_metadata_sha
            or expected_metadata_sha != D2_METADATA_SHA256
            or value.get("scenario_selection")
            != "first_physical_training_row_per_map_outcome_blind"
            or value.get("plain_actor_key_count") != 12
            or value.get("trainable_actor_parameter_count") != 706862
            or value.get("product_outcomes_reported_or_compared") is not False
            or value.get("candidate_selection_performed") is not False
            or value.get("ppo_pilot_iteration_completed") is not False
            or not reports_valid
            or not stochastic_valid
        ):
            raise RunnerError("B4 plumbing smoke marker/envelope mismatch")
        return value
    expected_keys = {
        "schema",
        "passed",
        "run_plan_sha256",
        "source_commit",
        "training_manifest_sha256",
        "bc_checkpoint_sha256",
        "sidecar_bundle_sha256",
        "d2_episode_metadata_sha256",
        "scenario_selection",
        "selected_scenarios",
        "arms",
        "product_outcomes_reported_or_compared",
        "arm_selection_performed",
        "ppo_pilot_iteration_completed",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RunnerError("B2 plumbing smoke marker schema mismatch")
    manifest = _marker_manifest_path(root, "training_scenarios.tsv")
    expected_manifest_sha = _frozen_entry_sha256(plan, "task8/training_scenarios.tsv")
    expected_bc_sha = _frozen_entry_sha256(
        plan, "pretrained/end2race.pth", source=True
    )
    expected_sidecar_sha = _frozen_entry_sha256(
        plan, "sidecar/sidecar_bundle.pt"
    )
    training_rows = _read_tsv_rows(manifest)
    expected_maps = ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring")
    first_by_map: dict[str, dict[str, Any]] = {}
    for physical_index, row in enumerate(training_rows):
        if int(row.get("training_order", -1)) != physical_index:
            raise RunnerError("plumbing smoke training manifest order drift")
        first_by_map.setdefault(
            row.get("map_name", ""),
            {
                "training_order": physical_index,
                "map_name": row.get("map_name", ""),
                "l2_id": row.get("l2_id", ""),
                "l4_id": row.get("l4_id", ""),
                "skill": row.get("skill", ""),
                "opponent_raceline": row.get("opponent_raceline", ""),
                "speedscale_hex": row.get("speedscale_hex", ""),
                "resolved_ego_idx": int(row.get("resolved_ego_idx", -1)),
            },
        )
    expected_selection = [first_by_map[name] for name in expected_maps if name in first_by_map]
    expected_metadata_sha = _frozen_entry_sha256(
        plan, "d2/episode_metadata.tsv"
    )
    arms = value.get("arms")
    selected = value.get("selected_scenarios")
    selected_keys = {
        "training_order",
        "map_name",
        "l2_id",
        "l4_id",
        "skill",
        "opponent_raceline",
        "speedscale_hex",
        "resolved_ego_idx",
    }
    selected_types_valid = (
        isinstance(selected, list)
        and len(selected) == 4
        and all(
            isinstance(row, dict)
            and set(row) == selected_keys
            and type(row.get("training_order")) is int
            and type(row.get("resolved_ego_idx")) is int
            and all(
                type(row.get(field)) is str
                for field in (
                    "map_name",
                    "l2_id",
                    "l4_id",
                    "skill",
                    "opponent_raceline",
                    "speedscale_hex",
                )
            )
            for row in selected
        )
    )
    if (
        value.get("schema") != "bplus-v2.2-b2-plumbing-smoke-1"
        or value.get("passed") is not True
        or value.get("run_plan_sha256") != plan.plan_sha256
        or value.get("source_commit") != plan.source_commit
        or value.get("product_outcomes_reported_or_compared") is not False
        or value.get("arm_selection_performed") is not False
        or value.get("ppo_pilot_iteration_completed") is not False
        or value.get("scenario_selection")
        != "first_physical_training_row_per_map_outcome_blind"
        or value.get("training_manifest_sha256") != expected_manifest_sha
        or _sha256_file(manifest) != expected_manifest_sha
        or value.get("bc_checkpoint_sha256") != expected_bc_sha
        or expected_bc_sha != CANONICAL_BC_SHA256
        or value.get("sidecar_bundle_sha256") != expected_sidecar_sha
        or expected_sidecar_sha != CANONICAL_SIDECAR_SHA256
        or value.get("d2_episode_metadata_sha256") != expected_metadata_sha
        or expected_metadata_sha != D2_METADATA_SHA256
        or not selected_types_valid
        or selected != expected_selection
        or len(expected_selection) != 4
        or not isinstance(arms, dict)
        or set(arms) != set(ARMS)
    ):
        raise RunnerError("B2 plumbing smoke marker/envelope mismatch")
    for report in arms.values():
        if not isinstance(report, dict):
            raise RunnerError("B2 plumbing smoke arm report is not an object")
        expected_arm_keys = {
            "episode_count",
            "intervention_branch_present",
            "joint_brake_branch_present",
            "steer_only_branch_present",
            "optimizer_update_executed",
            "preupdate_replay_tolerance",
            "preupdate_replay_max_abs_log_prob_delta",
            "preupdate_replay_max_abs_entropy_delta",
            "preupdate_replay_max_abs_ratio_minus_one",
            "finite_update_metrics",
        }
        if set(report) != expected_arm_keys:
            raise RunnerError("B2 plumbing smoke arm schema mismatch")
        if (
            type(report.get("episode_count")) is not int
            or report.get("episode_count") != 4
            or report.get("intervention_branch_present") is not True
            or report.get("joint_brake_branch_present") is not True
            or report.get("steer_only_branch_present") is not True
            or report.get("optimizer_update_executed") is not True
            or report.get("finite_update_metrics") is not True
        ):
            raise RunnerError("B2 plumbing smoke arm integrity mismatch")
        tolerance = report.get("preupdate_replay_tolerance")
        values = [
            report.get("preupdate_replay_max_abs_log_prob_delta"),
            report.get("preupdate_replay_max_abs_entropy_delta"),
            report.get("preupdate_replay_max_abs_ratio_minus_one"),
        ]
        if (
            tolerance != plan.config.get("replay_float32_atol")
            or any(type(item) not in (int, float) for item in values)
            or any(not float("-inf") < float(item) < float("inf") for item in values)
            or any(float(item) < 0.0 for item in values)
            or float(values[0]) > float(tolerance)
            or float(values[1]) > float(tolerance)
            or float(values[2]) > 2.0 * float(tolerance)
        ):
            raise RunnerError("B2 plumbing smoke replay contract mismatch")
    return value


def _validate_ready_marker(
    plan: RunPlan, root: Path, marker_path: Path | None = None
) -> dict[str, Any]:
    path = marker_path or root / "control/READY.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise RunnerError("B2 READY marker is missing or unsafe")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema",
        "passed",
        "run_plan_sha256",
        "source_commit",
        "source_archive_sha256",
        "inputs_archive_sha256",
        "baseline_marker_sha256",
        "plumbing_marker_sha256",
    }
    baseline = root / "control/bc_baseline_preflight.json"
    plumbing = root / "control/plumbing_smoke.json"
    _validate_baseline_marker(plan, root, baseline)
    _validate_plumbing_marker(plan, root, plumbing)
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("schema")
        != ("end2race-b4-ready-1" if plan.kind == "b4_train" else "end2race-b2-ready-1")
        or value.get("passed") is not True
        or value.get("run_plan_sha256") != plan.plan_sha256
        or value.get("source_commit") != plan.source_commit
        or value.get("source_archive_sha256") != plan.source_archive_sha256
        or value.get("inputs_archive_sha256") != plan.inputs_archive_sha256
        or value.get("baseline_marker_sha256") != _sha256_file(baseline)
        or value.get("plumbing_marker_sha256") != _sha256_file(plumbing)
    ):
        raise RunnerError("B2 READY marker contract mismatch")
    return value


def ready_host(plan_path: Path, host_id: str) -> int:
    if host_id != "local":
        raise RunnerError("B2 READY marker is authored once on the local host")
    plan = load_plan(plan_path)
    if plan.kind not in TRAIN_KINDS:
        raise RunnerError("B2 READY marker requires a training plan")
    host = _host(plan, host_id)
    root = Path(host.stage_root)
    check_preflight_host(plan_path, host_id)
    baseline = root / "control/bc_baseline_preflight.json"
    plumbing = root / "control/plumbing_smoke.json"
    _validate_baseline_marker(plan, root, baseline)
    _validate_plumbing_marker(plan, root, plumbing)
    path = root / "control/READY.json"
    if _lexists(path):
        _validate_ready_marker(plan, root, path)
        return 0
    partial = path.with_suffix(path.suffix + ".partial")
    _quarantine_uncommitted_marker(path)
    value = {
        "schema": (
            "end2race-b4-ready-1" if plan.kind == "b4_train" else "end2race-b2-ready-1"
        ),
        "passed": True,
        "run_plan_sha256": plan.plan_sha256,
        "source_commit": plan.source_commit,
        "source_archive_sha256": plan.source_archive_sha256,
        "inputs_archive_sha256": plan.inputs_archive_sha256,
        "baseline_marker_sha256": _sha256_file(baseline),
        "plumbing_marker_sha256": _sha256_file(plumbing),
    }
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(partial, 0o444)
    os.replace(partial, path)
    _validate_ready_marker(plan, root, path)
    return 0


def plumbing_host(plan_path: Path, host_id: str) -> int:
    if host_id != "local":
        raise RunnerError("B2 plumbing smoke runs once on the local host")
    plan = load_plan(plan_path)
    host = _host(plan, host_id)
    root = Path(host.stage_root)
    check_preflight_host(plan_path, host_id)
    _validate_baseline_marker(plan, root)
    _probe_gpu(host)
    smoke_job = JobSpec(
        "plumbing-smoke",
        "preflight",
        host.host_id,
        "preflight",
        tuple(),
        "outputs/preflight",
        "cache/numba/plumbing-smoke",
        gpu_exclusive=True,
    )
    cache = root / smoke_job.numba_cache_relpath
    cache.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **_job_environment(plan, host, smoke_job)}
    output = root / "control/plumbing_smoke.json"
    subprocess.run(
        [
            host.python,
            "-m",
            "bplus_v22.cli",
            "b4-plumbing-smoke" if plan.kind == "b4_train" else "ppo-plumbing-smoke",
            "--run-plan",
            str(plan_path),
            "--output",
            str(output),
        ],
        check=True,
        cwd=root / "repo",
        env=env,
    )
    _validate_plumbing_marker(plan, root)
    os.chmod(output, 0o444)
    return 0


def install_marker_host(
    plan_path: Path, host_id: str, kind: str, source: Path
) -> int:
    """Validate and atomically install one immutable cross-host gate marker."""

    plan = load_plan(plan_path)
    host = _host(plan, host_id)
    root = Path(host.stage_root)
    check_stage_host(plan_path, host_id)
    control = (root / "control").resolve()
    if source.is_symlink() or not source.is_file():
        raise RunnerError("incoming marker is not one regular file")
    source = source.resolve()
    if source.stat().st_nlink != 1:
        raise RunnerError("incoming marker has an external hardlink")
    try:
        source.relative_to(control)
    except ValueError as error:
        raise RunnerError("incoming marker escaped staged control directory") from error
    contracts = {
        "baseline": (
            root / "control/bc_baseline_preflight.json",
            _validate_baseline_marker,
        ),
        "plumbing": (
            root / "control/plumbing_smoke.json",
            _validate_plumbing_marker,
        ),
        "ready": (
            root / "control/READY.json",
            _validate_ready_marker,
        ),
    }
    if kind not in contracts:
        raise RunnerError(f"unsupported marker install kind: {kind}")
    destination, validator = contracts[kind]
    if source == destination.resolve():
        raise RunnerError("incoming marker aliases its immutable destination")
    validator(plan, root, source)
    if _lexists(destination):
        validator(plan, root, destination)
        if destination.stat().st_nlink != 1 or os.path.samefile(source, destination):
            raise RunnerError(f"published {kind} marker has an unsafe hardlink")
        if _sha256_file(destination) != _sha256_file(source):
            raise RunnerError(f"published {kind} marker differs from incoming evidence")
        source.unlink()
        return 0
    private = control / f".{kind}.{os.getpid()}.validated.partial"
    if _lexists(private):
        raise RunnerError("private marker install path already exists")
    try:
        with source.open("rb") as read_handle, private.open("xb") as write_handle:
            shutil.copyfileobj(read_handle, write_handle)
            write_handle.flush()
            os.fsync(write_handle.fileno())
        validator(plan, root, private)
        os.chmod(private, 0o444)
        os.replace(private, destination)
        source.unlink()
    finally:
        if _lexists(private):
            private.unlink()
    if destination.stat().st_nlink != 1:
        raise RunnerError(f"installed {kind} marker has an external hardlink")
    validator(plan, root, destination)
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
    if plan.kind == "b4_train":
        command = "b4-pilot"
    elif plan.kind == "b4_eval":
        command = "b4-evaluate"
    else:
        command = "ppo-pilot" if plan.kind in TRAIN_KINDS else "ppo-evaluate"
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


def check_stage_host(plan_path: Path, host_id: str) -> int:
    plan = load_plan(plan_path)
    host = _host(plan, host_id)
    _verify_staged_files(plan, host)
    _verify_extracted_source_tree(Path(host.stage_root))
    marker = Path(host.stage_root) / "control/STAGED"
    if (
        not marker.is_file()
        or marker.is_symlink()
        or marker.read_text(encoding="utf-8") != plan.plan_sha256 + "\n"
    ):
        raise RunnerError(f"host is not atomically staged for this plan: {host_id}")
    return 0


def _validate_preflight_marker(
    plan: RunPlan, host: HostSpec, root: Path
) -> dict[str, Any]:
    path = root / "control/preflight.json"
    if not path.is_file() or path.is_symlink():
        raise RunnerError(f"host preflight marker is missing: {host.host_id}")
    marker = json.loads(path.read_text(encoding="utf-8"))
    gpu = marker.get("gpu")
    module_paths = marker.get("module_paths")
    capabilities = marker.get("capabilities")
    if (
        marker.get("schema") != "end2race-host-preflight-1"
        or marker.get("plan_sha256") != plan.plan_sha256
        or marker.get("host") != host.host_id
        or marker.get("display") != host.display
        or marker.get("environment") != host.expected_environment
        or not isinstance(gpu, dict)
        or gpu.get("uuid") != host.gpu_uuid
        or gpu.get("name") != host.gpu_name
        or not isinstance(module_paths, dict)
        or set(module_paths) != set(plan.module_path_contract)
        or not isinstance(capabilities, dict)
        or capabilities.get("schema") != CAPABILITIES_SCHEMA
        or not set(plan.required_cli).issubset(capabilities.get("commands", ()))
    ):
        raise RunnerError(f"host preflight marker contract mismatch: {host.host_id}")
    repo = (root / "repo").resolve()
    for value in module_paths.values():
        try:
            Path(str(value)).resolve().relative_to(repo)
        except ValueError as error:
            raise RunnerError("preflight module path escaped staged source") from error
    return marker


def check_preflight_host(plan_path: Path, host_id: str) -> int:
    plan = load_plan(plan_path)
    host = _host(plan, host_id)
    check_stage_host(plan_path, host_id)
    if plan.kind in TRAIN_KINDS:
        _validate_baseline_marker(plan, Path(host.stage_root))
    _validate_preflight_marker(plan, host, Path(host.stage_root))
    return 0


def preflight_host(plan_path: Path, host_id: str) -> int:
    plan = load_plan(plan_path)
    host = _host(plan, host_id)
    check_stage_host(plan_path, host_id)
    if plan.kind in TRAIN_KINDS:
        _validate_baseline_marker(plan, Path(host.stage_root))
    root = Path(host.stage_root)
    existing_marker = root / "control/preflight.json"
    if _lexists(existing_marker):
        _validate_preflight_marker(plan, host, root)
        return 0
    _quarantine_uncommitted_marker(existing_marker)
    for command in ("rsync", "tar", "flock", "nvidia-smi", "xdpyinfo"):
        if shutil.which(command) is None:
            raise RunnerError(f"required executable missing: {command}")
    subprocess.run(["xdpyinfo", "-display", host.display], check=True, capture_output=True)
    gpu = _probe_gpu(host)
    actual_environment = _assert_live_environment(host)
    paths = _probe_module_paths(plan, host)
    capabilities = _probe_capabilities(plan, host)
    _validate_cli_plan(plan, host)
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
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    _validate_preflight_marker(plan, host, root)
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
    if plan.kind not in TRAIN_KINDS | EVAL_KINDS:
        raise RunnerError("explicit resume is unavailable for this plan kind")
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
    if job.kind in {"learner", "b4_training"}:
        summary_path = output / "summary.json"
        if not summary_path.is_file():
            raise RunnerError(f"learner summary is missing: {job.job_id}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if job.kind == "b4_training":
            config_path = output / "config.json"
            if not config_path.is_file() or config_path.is_symlink():
                raise RunnerError(f"B4 learner config is missing: {job.job_id}")
            config_record = json.loads(config_path.read_text(encoding="utf-8"))
            snapshot_files = summary.get(
                "actor_snapshot_file_sha256_by_iteration"
            )
            snapshot_tensors = summary.get(
                "actor_snapshot_tensor_sha256_by_iteration"
            )
            if (
                summary.get("schema") != "end2race-b4-direct-head-pilot-1"
                or summary.get("integrity_passed") is not True
                or summary.get("passed") is not True
                or summary.get("seed") != job.seed
                or summary.get("iterations") != 30
                or summary.get("run_plan_sha256") != plan.plan_sha256
                or summary.get("source_commit") != plan.source_commit
                or summary.get("bc_checkpoint_sha256") != CANONICAL_BC_SHA256
                or summary.get("training_manifest_sha256")
                != plan.config["training_manifest_sha256"]
                or summary.get("curriculum_sha256")
                != plan.config["curriculum_sha256_by_seed"][str(job.seed)]
                or summary.get("product_kpi_evaluated") is not False
                or summary.get("fresh_pool_opened") is not False
                or not isinstance(snapshot_files, dict)
                or set(snapshot_files) != {"0", "10", "20", "30"}
                or not isinstance(snapshot_tensors, dict)
                or set(snapshot_tensors) != {"0", "10", "20", "30"}
                or any(
                    not SHA256_RE.fullmatch(str(value))
                    for value in (*snapshot_files.values(), *snapshot_tensors.values())
                )
                or summary.get("bc_actor_tensor_sha256")
                != snapshot_tensors.get("0")
                or config_record.get("schema")
                != "end2race-b4-direct-head-pilot-1"
                or config_record.get("seed") != job.seed
                or config_record.get("run_plan_sha256") != plan.plan_sha256
                or config_record.get("source_commit") != plan.source_commit
                or config_record.get("bc_checkpoint_sha256")
                != CANONICAL_BC_SHA256
                or config_record.get("bc_actor_tensor_sha256")
                != snapshot_tensors.get("0")
                or config_record.get("training_manifest_sha256")
                != plan.config["training_manifest_sha256"]
                or config_record.get("curriculum_sha256")
                != plan.config["curriculum_sha256_by_seed"][str(job.seed)]
                or config_record.get("config") != plan.config["ppo"]
            ):
                raise RunnerError(f"B4 learner COMPLETE envelope mismatch: {job.job_id}")
            from bplus_v22.b4_direct import (
                actor_snapshot_sha256,
                load_strict_plain_actor,
            )

            actor_states = {}
            for iteration in (0, 10, 20, 30):
                path = output / f"actors/iter_{iteration:04d}.pth"
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or _sha256_file(path) != snapshot_files[str(iteration)]
                ):
                    raise RunnerError(
                        f"B4 actor snapshot file mismatch: {job.job_id}/iter{iteration}"
                    )
                actor_state = load_strict_plain_actor(path, "cpu").state_dict()
                actor_states[iteration] = actor_state
                if actor_snapshot_sha256(actor_state) != snapshot_tensors[str(iteration)]:
                    raise RunnerError(
                        f"B4 actor snapshot tensor mismatch: {job.job_id}/iter{iteration}"
                    )
            if any(
                not actor_states[iteration][name].equal(actor_states[0][name])
                for iteration in (10, 20, 30)
                for name in actor_states[0]
                if not name.startswith("output_layer.")
            ):
                raise RunnerError(f"B4 frozen actor tensor drift: {job.job_id}")
            final_full = output / "checkpoints/iter_0030.pt"
            ledger_path = output / "iterations.jsonl"
            if (
                not final_full.is_file()
                or _sha256_file(final_full)
                != summary.get("final_full_checkpoint_sha256")
                or not ledger_path.is_file()
                or len(ledger_path.read_text(encoding="utf-8").splitlines()) != 30
            ):
                raise RunnerError(f"B4 final checkpoint/ledger mismatch: {job.job_id}")
        else:
            expected_schema = (
                "bplus-v2.2-b3-ppo-pilot-1"
                if plan.kind == "b3_train"
                else "bplus-v2.2-b2-ppo-pilot-1"
            )
            expected_iterations = int(plan.config["iterations"])
            if (
                summary.get("schema") != expected_schema
                or summary.get("integrity_passed") is not True
                or summary.get("passed") is not True
                or summary.get("arm") != job.arm
                or summary.get("seed") != job.seed
                or summary.get("iterations") != expected_iterations
                or summary.get("run_plan_sha256") != plan.plan_sha256
            ):
                raise RunnerError(f"learner COMPLETE envelope mismatch: {job.job_id}")
    elif job.kind in {"evaluation_shard", "b4_evaluation_shard"}:
        _validate_eval_shard_output(plan, output, job)
    else:
        raise RunnerError(f"unsupported executable job kind: {job.kind}")


def _tsv_scalar(value: Any) -> str:
    return "" if value is None else str(value)


def _validate_eval_shard_output(plan: RunPlan, output: Path, job: JobSpec) -> None:
    """Strictly bind an atomic eval shard to its immutable EvalPlan."""

    if plan.kind not in EVAL_KINDS or not plan.evaluation_contract:
        raise RunnerError("evaluation shard requires one B2 EvalPlan")
    children = list(output.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise RunnerError(f"evaluation shard has non-regular content: {job.job_id}")
    if {path.name for path in children} != {"COMPLETE", "shard.json", "episodes.tsv"}:
        raise RunnerError(f"evaluation shard file inventory mismatch: {job.job_id}")

    contract = plan.evaluation_contract
    shard = json.loads((output / "shard.json").read_text(encoding="utf-8"))
    expected_scenarios = [
        item
        for item in contract["scenarios"]
        if int(item["shard_index"]) == int(job.shard_index)
    ]
    variants = [str(value) for value in contract["variants"]]
    if plan.kind == "b4_eval":
        checkpoint_by_variant = {"BC": CANONICAL_BC_SHA256}
        checkpoint_by_variant.update(
            {
                f"seed{int(item['seed'])}_iter{int(item['iteration'])}": str(
                    item["sha256"]
                )
                for item in contract["checkpoint_set"]
            }
        )
        if (
            shard.get("schema") != "end2race-b4-eval-shard-1"
            or shard.get("shard_index") != job.shard_index
            or shard.get("shard_count") != job.shard_count
            or shard.get("scenario_manifest_sha256") != contract["manifest_sha256"]
            or shard.get("checkpoint_manifest_sha256")
            != contract["checkpoint_set_sha256"]
            or shard.get("bc_checkpoint_sha256") != CANONICAL_BC_SHA256
            or shard.get("checkpoint_sha256_by_variant") != checkpoint_by_variant
            or not isinstance(shard.get("rows"), list)
        ):
            raise RunnerError(f"B4 eval shard COMPLETE envelope mismatch: {job.job_id}")
        expected_keys = {
            (int(item["row_index"]), str(item["l2_id"]), variant)
            for item in expected_scenarios
            for variant in variants
        }
        observed_keys: set[tuple[int, str, str]] = set()
        rows = shard["rows"]
        if len(rows) != len(expected_keys):
            raise RunnerError(f"B4 eval shard row count mismatch: {job.job_id}")
        for row in rows:
            if not isinstance(row, dict):
                raise RunnerError(f"B4 eval shard row is not an object: {job.job_id}")
            key = (
                int(row.get("task8_row_index", -1)),
                str(row.get("l2_id", "")),
                str(row.get("variant", "")),
            )
            if (
                key not in expected_keys
                or key in observed_keys
                or row.get("schema") != "end2race-b4-eval-row-1"
                or row.get("scenario_manifest_sha256") != contract["manifest_sha256"]
                or row.get("checkpoint_manifest_sha256")
                != contract["checkpoint_set_sha256"]
                or row.get("checkpoint_sha256") != checkpoint_by_variant.get(key[2])
                or type(row.get("deterministic_speed_projection_count")) is not int
                or int(row["deterministic_speed_projection_count"]) < 0
                or not SHA256_RE.fullmatch(str(row.get("trajectory_sha256", "")))
            ):
                raise RunnerError(f"B4 eval shard row mismatch: {job.job_id}, {key}")
            observed_keys.add(key)
        if observed_keys != expected_keys:
            raise RunnerError(f"B4 eval shard Cartesian mismatch: {job.job_id}")
        control_rows = [
            {
                **row,
                "row_index": row["task8_row_index"],
                "variant_id": row["variant"],
                "shard_index": job.shard_index,
                "manifest_sha256": contract["manifest_sha256"],
                "checkpoint_set_sha256": contract["checkpoint_set_sha256"],
            }
            for row in rows
        ]
        expected_fields = sorted({name for row in control_rows for name in row})
        with (output / "episodes.tsv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            observed_tsv = list(reader)
            observed_fields = reader.fieldnames
        if observed_fields != expected_fields or len(observed_tsv) != len(control_rows):
            raise RunnerError(f"B4 eval shard TSV schema/count mismatch: {job.job_id}")
        for expected_row, observed_row in zip(control_rows, observed_tsv):
            expected_tsv = {
                name: _tsv_scalar(expected_row.get(name)) for name in expected_fields
            }
            if observed_row != expected_tsv:
                raise RunnerError(f"B4 eval shard TSV/JSON mismatch: {job.job_id}")
        return
    checkpoint_by_variant = {"BC": CANONICAL_BC_SHA256}
    checkpoint_by_variant.update(
        {
            f"{item['arm']}::seed{int(item['seed'])}": str(item["sha256"])
            for item in contract["checkpoint_set"]
        }
    )
    if (
        shard.get("schema") != "bplus-v2.2-ppo-eval-shard-1"
        or shard.get("shard_index") != job.shard_index
        or shard.get("shard_count") != job.shard_count
        or shard.get("scenario_manifest_sha256") != contract["manifest_sha256"]
        or shard.get("checkpoint_manifest_sha256")
        != contract["training_manifest_sha256"]
        or shard.get("bc_checkpoint_sha256") != CANONICAL_BC_SHA256
        or shard.get("checkpoint_sha256_by_variant") != checkpoint_by_variant
        or not isinstance(shard.get("rows"), list)
    ):
        raise RunnerError(f"evaluation shard COMPLETE envelope mismatch: {job.job_id}")

    expected_keys = {
        (int(item["row_index"]), str(item["l2_id"]), variant)
        for item in expected_scenarios
        for variant in variants
    }
    observed_keys: set[tuple[int, str, str]] = set()
    rows = shard["rows"]
    if len(rows) != len(expected_keys):
        raise RunnerError(f"evaluation shard row count mismatch: {job.job_id}")
    for row in rows:
        if not isinstance(row, dict):
            raise RunnerError(f"evaluation shard row is not an object: {job.job_id}")
        key = (
            int(row.get("task8_row_index", -1)),
            str(row.get("l2_id", "")),
            str(row.get("variant", "")),
        )
        if (
            key not in expected_keys
            or key in observed_keys
            or row.get("scenario_manifest_sha256") != contract["manifest_sha256"]
            or row.get("checkpoint_manifest_sha256")
            != contract["training_manifest_sha256"]
            or row.get("checkpoint_sha256") != checkpoint_by_variant.get(key[2])
            or type(row.get("external_clip_micro_steps")) is not int
            or row.get("external_clip_micro_steps") != 0
            or not SHA256_RE.fullmatch(str(row.get("trajectory_sha256", "")))
        ):
            raise RunnerError(f"evaluation shard row mismatch: {job.job_id}, {key}")
        observed_keys.add(key)
    if observed_keys != expected_keys:
        raise RunnerError(f"evaluation shard Cartesian mismatch: {job.job_id}")

    control_rows = [
        {
            **row,
            "row_index": row["task8_row_index"],
            "variant_id": row["variant"],
            "shard_index": job.shard_index,
            "manifest_sha256": contract["manifest_sha256"],
            "checkpoint_set_sha256": contract["checkpoint_set_sha256"],
        }
        for row in rows
    ]
    expected_fields = sorted({name for row in control_rows for name in row})
    with (output / "episodes.tsv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        observed_tsv = list(reader)
        observed_fields = reader.fieldnames
    if observed_fields != expected_fields or len(observed_tsv) != len(control_rows):
        raise RunnerError(f"evaluation shard TSV schema/count mismatch: {job.job_id}")
    for expected_row, observed_row in zip(control_rows, observed_tsv):
        expected_tsv = {name: _tsv_scalar(expected_row.get(name)) for name in expected_fields}
        if observed_row != expected_tsv:
            raise RunnerError(f"evaluation shard TSV/JSON mismatch: {job.job_id}")


def execute_host(plan_path: Path, host_id: str, *, resume: bool = False) -> int:
    plan = load_plan(plan_path)
    host = _host(plan, host_id)
    root = Path(host.stage_root)
    check_preflight_host(plan_path, host_id)
    if plan.kind in TRAIN_KINDS:
        _validate_baseline_marker(plan, root)
        _validate_plumbing_marker(plan, root)
        _validate_ready_marker(plan, root)
    _assert_live_environment(host)
    # This runs while the outer `flock` is held, closing the race between the
    # earlier preflight and learner launch.
    _probe_gpu(host)
    status_path = root / "control/status.json"
    events_path = root / "control/status.jsonl"
    jobs = _host_jobs(plan, host_id)
    if not jobs:
        raise RunnerError(f"no jobs assigned to {host_id}")
    if resume:
        if plan.kind not in TRAIN_KINDS | EVAL_KINDS:
            raise RunnerError("this plan kind does not support explicit resume")
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
            recovered = {
                "state": "COMPLETE",
                "exit_code": 0,
                "started_at": previous.get("started_at"),
                "finished_at": recovered_at,
                "status_recovered_from_complete_release": True,
                "prior_state": previous.get("state"),
                "prior_exit_code": previous.get("exit_code"),
            }
            if job.kind in {"evaluation_shard", "b4_evaluation_shard"}:
                recovered.update(
                    {
                        "shard_json_sha256": _sha256_file(output / "shard.json"),
                        "episodes_tsv_sha256": _sha256_file(output / "episodes.tsv"),
                    }
                )
            status["jobs"][job.job_id] = recovered
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
        if resume_job and plan.kind in EVAL_KINDS:
            raise RunnerError(
                "evaluation resume only recovers a validated atomic COMPLETE shard; "
                f"incomplete shard must use a new EvalPlan: {job.job_id}"
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


def _collect_status_commands(
    plan: RunPlan, collection: Path | None = None
) -> list[list[str]]:
    collection = collection or Path(plan.collection_root)
    local = _host(plan, "local")
    remote = _host(plan, "remote")
    return [
        [
            "cp",
            "-a",
            str(Path(local.stage_root) / "control/status.json"),
            str(collection / "hosts/local/status.json"),
        ],
        [
            "cp",
            "-a",
            str(Path(local.stage_root) / "control/status.jsonl"),
            str(collection / "hosts/local/status.jsonl"),
        ],
        [
            "cp",
            "-a",
            str(Path(local.stage_root) / "control/preflight.json"),
            str(collection / "hosts/local/preflight.json"),
        ],
        [
            "cp",
            "-a",
            str(Path(local.stage_root) / "control/STAGED"),
            str(collection / "hosts/local/STAGED"),
        ],
        [
            "rsync",
            "-a",
            "--protect-args",
            f"{remote.ssh_host}:{remote.stage_root}/control/status.json",
            str(collection / "hosts/remote/status.json"),
        ],
        [
            "rsync",
            "-a",
            "--protect-args",
            f"{remote.ssh_host}:{remote.stage_root}/control/status.jsonl",
            str(collection / "hosts/remote/status.jsonl"),
        ],
        [
            "rsync",
            "-a",
            "--protect-args",
            f"{remote.ssh_host}:{remote.stage_root}/control/preflight.json",
            str(collection / "hosts/remote/preflight.json"),
        ],
        [
            "rsync",
            "-a",
            "--protect-args",
            f"{remote.ssh_host}:{remote.stage_root}/control/STAGED",
            str(collection / "hosts/remote/STAGED"),
        ],
    ]


def _collect_payload_commands(
    plan: RunPlan, collection: Path | None = None
) -> list[list[str]]:
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
            "rsync",
            "-a",
            "--protect-args",
            f"{remote.ssh_host}:{remote.stage_root}/outputs/",
            str(collection / "hosts/remote/outputs/"),
        ],
    ]
    if plan.kind in TRAIN_KINDS:
        commands.extend(
            (
                [
                "cp",
                "-a",
                str(Path(local.stage_root) / "control/bc_baseline_preflight.json"),
                str(collection / "control/bc_baseline_preflight.json"),
                ],
                [
                    "cp",
                    "-a",
                    str(Path(local.stage_root) / "control/baseline_shards"),
                    str(collection / "control/baseline_shards"),
                ],
                [
                "cp",
                "-a",
                str(Path(local.stage_root) / "control/plumbing_smoke.json"),
                str(collection / "control/plumbing_smoke.json"),
                ],
                [
                    "cp",
                    "-a",
                    str(Path(local.stage_root) / "control/READY.json"),
                    str(collection / "control/READY.json"),
                ],
                [
                    "rsync",
                    "-a",
                    "--protect-args",
                    f"{remote.ssh_host}:{remote.stage_root}/control/"
                    "bc_baseline_preflight.json",
                    str(collection / "control/remote_bc_baseline_preflight.json"),
                ],
                [
                    "rsync",
                    "-a",
                    "--protect-args",
                    f"{remote.ssh_host}:{remote.stage_root}/control/plumbing_smoke.json",
                    str(collection / "control/remote_plumbing_smoke.json"),
                ],
                [
                    "rsync",
                    "-a",
                    "--protect-args",
                    f"{remote.ssh_host}:{remote.stage_root}/control/READY.json",
                    str(collection / "control/remote_READY.json"),
                ],
                [
                    "cp",
                    "-a",
                    str(Path(local.stage_root) / "inputs/task8/development_scenarios.tsv"),
                    str(collection / "control/input_contract/development_scenarios.tsv"),
                ],
                [
                    "cp",
                    "-a",
                    str(Path(local.stage_root) / "inputs/task8/training_scenarios.tsv"),
                    str(collection / "control/input_contract/training_scenarios.tsv"),
                ],
            )
        )
    return commands


def _validate_collected_statuses(plan: RunPlan, partial: Path) -> None:
    for host_id in ("local", "remote"):
        status_path = partial / f"hosts/{host_id}/status.json"
        events_path = partial / f"hosts/{host_id}/status.jsonl"
        if not status_path.is_file() or not events_path.is_file():
            raise RunnerError(f"collected host status evidence is missing: {host_id}")
        status_value = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            status_value.get("plan_sha256") != plan.plan_sha256
            or status_value.get("host") != host_id
            or status_value.get("state") != "COMPLETE"
        ):
            raise RunnerError(f"cannot collect incomplete/mismatched host: {host_id}")
        expected_jobs = _host_jobs(plan, host_id)
        if set(status_value.get("jobs", {})) != {job.job_id for job in expected_jobs}:
            raise RunnerError(f"collected host job inventory mismatch: {host_id}")
        if any(
            status_value["jobs"][job.job_id].get("state") != "COMPLETE"
            for job in expected_jobs
        ):
            raise RunnerError(f"collected host has incomplete jobs: {host_id}")
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if not events or events[-1].get("event") != "host_complete":
            raise RunnerError(f"collected host event ledger is incomplete: {host_id}")
        staged = partial / f"hosts/{host_id}/STAGED"
        if not staged.is_file() or staged.read_text(encoding="utf-8") != plan.plan_sha256 + "\n":
            raise RunnerError(f"collected host staging identity mismatch: {host_id}")
        copied_preflight = partial / f"hosts/{host_id}/preflight.json"
        if not copied_preflight.is_file() or copied_preflight.is_symlink():
            raise RunnerError(f"collected host preflight is missing: {host_id}")
        preflight_value = json.loads(copied_preflight.read_text(encoding="utf-8"))
        if (
            preflight_value.get("schema") != "end2race-host-preflight-1"
            or preflight_value.get("plan_sha256") != plan.plan_sha256
            or preflight_value.get("host") != host_id
        ):
            raise RunnerError(f"collected host preflight identity mismatch: {host_id}")


def _quarantine_collection_partial(collection: Path, partial: Path) -> Path:
    if partial.is_symlink() or not partial.is_dir():
        raise RunnerError(f"unsafe collection partial: {partial}")
    base = collection.with_name(collection.name + ".attempt_failures")
    base.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (base / f"attempt_{attempt:03d}").exists():
        attempt += 1
    target = base / f"attempt_{attempt:03d}"
    target.mkdir()
    os.replace(partial, target / partial.name)
    (target / "retry.json").write_text(
        json.dumps(
            {
                "schema": "end2race-collection-retry-1",
                "quarantined_at": _now(),
                "reason": "previous_uncommitted_collection_attempt",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def _record_collection_failure(
    partial: Path,
    plan: RunPlan,
    *,
    phase: str,
    error: object,
    command_index: int | None = None,
    exit_code: int | None = None,
) -> None:
    if not partial.is_dir():
        return
    value = {
        "schema": "end2race-collection-failure-1",
        "plan_sha256": plan.plan_sha256,
        "failed_at": _now(),
        "phase": phase,
        "error_type": type(error).__name__,
        "error": str(error),
        "command_index": command_index,
        "exit_code": exit_code,
    }
    (partial / "failure.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _validate_collected_baseline_shards(
    plan: RunPlan, partial: Path, baseline_value: Mapping[str, Any]
) -> None:
    shard_sha = {
        item["shard_index"]: item["file_sha256"]
        for item in baseline_value["shards"]
    }
    if set(shard_sha) != set(range(SHARD_COUNT)):
        raise RunnerError("collected BC baseline shard inventory mismatch")
    for shard_index in range(SHARD_COUNT):
        shard_path = _baseline_shard_path(partial, shard_index)
        _validate_baseline_shard(plan, shard_path, shard_index)
        if _sha256_file(shard_path) != shard_sha[shard_index]:
            raise RunnerError(
                f"collected BC baseline shard hash mismatch: {shard_index}"
            )


def collect_baseline_failure(plan: RunPlan, dry_run: bool) -> int:
    """Atomically collect a terminal pre-PPO baseline acceptance failure."""

    if plan.kind not in TRAIN_KINDS:
        raise RunnerError("baseline failure collection requires a training plan")
    collection = Path(plan.collection_root)
    partial = collection.with_name(collection.name + ".partial")
    if _lexists(collection):
        raise FileExistsError(collection)
    if _lexists(partial) and not dry_run:
        _quarantine_collection_partial(collection, partial)
    local = _host(plan, "local")
    remote = _host(plan, "remote")
    commands = [
        [
            "cp",
            "-a",
            str(Path(local.stage_root) / "control/bc_baseline_preflight.failed.json"),
            str(partial / "control/bc_baseline_preflight.failed.json"),
        ],
        [
            "cp",
            "-a",
            str(Path(local.stage_root) / "control/baseline_shards"),
            str(partial / "control/baseline_shards"),
        ],
        [
            "cp",
            "-a",
            str(Path(local.stage_root) / "inputs/task8/development_scenarios.tsv"),
            str(partial / "control/input_contract/development_scenarios.tsv"),
        ],
        [
            "cp",
            "-a",
            str(Path(local.stage_root) / "control/STAGED"),
            str(partial / "hosts/local/STAGED"),
        ],
        [
            "rsync",
            "-a",
            "--protect-args",
            f"{remote.ssh_host}:{remote.stage_root}/control/STAGED",
            str(partial / "hosts/remote/STAGED"),
        ],
    ]
    for command in commands:
        print(_display_command(command))
    if dry_run:
        return 0
    (partial / "control/input_contract").mkdir(parents=True)
    (partial / "hosts/local").mkdir(parents=True)
    (partial / "hosts/remote").mkdir(parents=True)
    for index, command in enumerate(commands):
        code = subprocess.run(command, check=False).returncode
        if code:
            _record_collection_failure(
                partial,
                plan,
                phase="baseline_failure_copy",
                error=f"command exited {code}",
                command_index=index,
                exit_code=code,
            )
            return code
    try:
        failed_path = partial / "control/bc_baseline_preflight.failed.json"
        failed_value = _validate_baseline_marker(
            plan, partial, failed_path, require_pass=False
        )
        _validate_collected_baseline_shards(plan, partial, failed_value)
        for host_id in ("local", "remote"):
            staged = partial / f"hosts/{host_id}/STAGED"
            if (
                not staged.is_file()
                or staged.is_symlink()
                or staged.read_text(encoding="utf-8") != plan.plan_sha256 + "\n"
            ):
                raise RunnerError(
                    f"baseline failure collection STAGED mismatch: {host_id}"
                )
    except Exception as error:
        _record_collection_failure(
            partial, plan, phase="baseline_failure_validation", error=error
        )
        raise
    (partial / "run_plan.json").write_text(
        json.dumps(_plan_to_dict(plan), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (partial / "FAILED.json").write_text(
        json.dumps(
            {
                "schema": "end2race-b2-baseline-failure-collection-1",
                "state": "FAILED",
                "terminal_phase": "baseline_preflight",
                "plan_sha256": plan.plan_sha256,
                "source_commit": plan.source_commit,
                "baseline_failure_sha256": _sha256_file(
                    partial / "control/bc_baseline_preflight.failed.json"
                ),
                "collected_at": _now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(partial, collection)
    return 0


def collect(plan: RunPlan, dry_run: bool) -> int:
    if plan.kind in TRAIN_KINDS and _lexists(
        Path(_host(plan, "local").stage_root)
        / "control/bc_baseline_preflight.failed.json"
    ):
        return collect_baseline_failure(plan, dry_run)
    collection = Path(plan.collection_root)
    partial = collection.with_name(collection.name + ".partial")
    if _lexists(collection):
        raise FileExistsError(collection)
    if _lexists(partial) and not dry_run:
        _quarantine_collection_partial(collection, partial)
    status_commands = _collect_status_commands(plan, partial)
    payload_commands = _collect_payload_commands(plan, partial)
    commands = [*status_commands, *payload_commands]
    for command in commands:
        print(_display_command(command))
    if dry_run:
        return 0
    (partial / "hosts/local").mkdir(parents=True)
    (partial / "hosts/remote").mkdir(parents=True)
    (partial / "control").mkdir(parents=True)
    (partial / "control/input_contract").mkdir(parents=True)
    for index, command in enumerate(status_commands):
        code = subprocess.run(command, check=False).returncode
        if code:
            _record_collection_failure(
                partial,
                plan,
                phase="status_copy",
                error=f"command exited {code}",
                command_index=index,
                exit_code=code,
            )
            return code
    try:
        _validate_collected_statuses(plan, partial)
    except Exception as error:
        _record_collection_failure(
            partial, plan, phase="status_validation", error=error
        )
        raise
    for index, command in enumerate(payload_commands):
        code = subprocess.run(command, check=False).returncode
        if code:
            _record_collection_failure(
                partial,
                plan,
                phase="payload_copy",
                error=f"command exited {code}",
                command_index=index,
                exit_code=code,
            )
            return code
    try:
        for host_id in ("local", "remote"):
            host_root = partial / f"hosts/{host_id}"
            for job in _host_jobs(plan, host_id):
                _validate_job_output(plan, host_root, job)
        if plan.kind in TRAIN_KINDS:
            baseline_value = _validate_baseline_marker(plan, partial)
            _validate_collected_baseline_shards(plan, partial, baseline_value)
            _validate_plumbing_marker(plan, partial)
            _validate_ready_marker(plan, partial)
            remote_baseline = partial / "control/remote_bc_baseline_preflight.json"
            remote_plumbing = partial / "control/remote_plumbing_smoke.json"
            remote_ready = partial / "control/remote_READY.json"
            _validate_baseline_marker(plan, partial, remote_baseline)
            _validate_plumbing_marker(plan, partial, remote_plumbing)
            _validate_ready_marker(plan, partial, remote_ready)
            if (
                _sha256_file(remote_baseline)
                != _sha256_file(partial / "control/bc_baseline_preflight.json")
                or _sha256_file(remote_plumbing)
                != _sha256_file(partial / "control/plumbing_smoke.json")
                or _sha256_file(remote_ready)
                != _sha256_file(partial / "control/READY.json")
            ):
                raise RunnerError("local/remote pre-PPO gate markers differ")
    except Exception as error:
        _record_collection_failure(
            partial, plan, phase="payload_validation", error=error
        )
        raise
    (partial / "run_plan.json").write_text(
        json.dumps(_plan_to_dict(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, collection)
    return 0


def validate_eval_collection(plan: RunPlan, collection: Path) -> dict[str, Any]:
    if plan.kind not in EVAL_KINDS or not plan.evaluation_contract:
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
    merge_command = "b4-merge-eval" if plan.kind == "b4_eval" else "ppo-merge-eval"
    if dry_run:
        print(
            _display_command(
                [
                    PINNED_PYTHON,
                    "-m",
                    "bplus_v22.cli",
                    merge_command,
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
        merge_command,
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


def _show(plan_path: Path, plan: RunPlan) -> None:
    print(json.dumps(_plan_to_dict(plan), indent=2, sort_keys=True))
    wrapper = str(REPO_ROOT / "run.sh")
    path = str(plan_path)
    print("\n# canonical phase order")
    print(_display_command([wrapper, "stage", path, "--all-hosts"]))
    if plan.kind in TRAIN_KINDS:
        print(_display_command([wrapper, "baseline-preflight", path]))
    print(_display_command([wrapper, "preflight", path, "--all-hosts"]))
    if plan.kind in TRAIN_KINDS:
        print(_display_command([wrapper, "plumbing-smoke", path]))
    print(_display_command([wrapper, "execute", path, "--all-hosts"]))
    if plan.kind in TRAIN_KINDS | EVAL_KINDS:
        print("# explicit resume only after a recorded interruption/failure")
        for host in plan.hosts:
            print(_display_command([wrapper, "resume", path, "--host", host.host_id]))
    print(_display_command([wrapper, "status", path, "--all-hosts"]))
    print(_display_command([wrapper, "collect", path]))
    if plan.kind in EVAL_KINDS:
        print(_display_command([wrapper, "merge-eval", path]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run.sh", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="show immutable B2/B3/B4 workflows and legacy entries")
    legacy = sub.add_parser("legacy-show", help="show one non-executable B1 legacy template")
    legacy.add_argument("name", choices=sorted(LEGACY_SHOW_ONLY))

    plan = sub.add_parser("plan", help="create one immutable six-learner B2 plan")
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--source-commit", default="HEAD")
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument("--local-gpu-uuid", required=True)
    plan.add_argument("--remote-gpu-uuid", required=True)

    plan_b3 = sub.add_parser(
        "plan-b3", help="create one immutable six-learner B3 unified-policy plan"
    )
    plan_b3.add_argument("--run-id", required=True)
    plan_b3.add_argument("--source-commit", default="HEAD")
    plan_b3.add_argument("--output", required=True, type=Path)
    plan_b3.add_argument("--local-gpu-uuid", required=True)
    plan_b3.add_argument("--remote-gpu-uuid", required=True)

    plan_b4 = sub.add_parser(
        "plan-b4", help="create the approved immutable seed-1 B4 plan"
    )
    plan_b4.add_argument("--run-id", required=True)
    plan_b4.add_argument("--source-commit", default="HEAD")
    plan_b4.add_argument("--output", required=True, type=Path)
    plan_b4.add_argument("--local-gpu-uuid", required=True)
    plan_b4.add_argument("--remote-gpu-uuid", required=True)

    plan_eval = sub.add_parser("plan-eval", help="freeze six checkpoints into an eval plan")
    plan_eval.add_argument("--run-id", required=True)
    plan_eval.add_argument("--training-plan", required=True, type=Path)
    plan_eval.add_argument("--source-commit", default="HEAD")
    plan_eval.add_argument("--checkpoint", action="append", required=True)
    plan_eval.add_argument("--output", required=True, type=Path)

    plan_b4_eval = sub.add_parser(
        "plan-b4-eval", help="freeze three seed-1 snapshots into a 288x4 EvalPlan"
    )
    plan_b4_eval.add_argument("--run-id", required=True)
    plan_b4_eval.add_argument("--training-plan", required=True, type=Path)
    plan_b4_eval.add_argument("--source-commit", default="HEAD")
    plan_b4_eval.add_argument("--checkpoint", action="append", required=True)
    plan_b4_eval.add_argument("--output", required=True, type=Path)

    show = sub.add_parser("show", help="verify and print one immutable plan")
    _add_plan_argument(show)
    baseline = sub.add_parser("baseline-preflight")
    _add_plan_argument(baseline)
    baseline.add_argument("--dry-run", action="store_true")
    plumbing = sub.add_parser("plumbing-smoke")
    _add_plan_argument(plumbing)
    plumbing.add_argument("--dry-run", action="store_true")
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
    internal_check_stage = sub.add_parser("_check-stage-host", help=argparse.SUPPRESS)
    _add_plan_argument(internal_check_stage)
    internal_check_stage.add_argument(
        "--host", choices=("local", "remote"), required=True
    )
    internal_check_preflight = sub.add_parser(
        "_check-preflight-host", help=argparse.SUPPRESS
    )
    _add_plan_argument(internal_check_preflight)
    internal_check_preflight.add_argument(
        "--host", choices=("local", "remote"), required=True
    )
    internal_preflight = sub.add_parser("_preflight-host", help=argparse.SUPPRESS)
    _add_plan_argument(internal_preflight)
    internal_preflight.add_argument("--host", choices=("local", "remote"), required=True)
    internal_baseline = sub.add_parser("_baseline-host", help=argparse.SUPPRESS)
    _add_plan_argument(internal_baseline)
    internal_baseline.add_argument(
        "--host", choices=("local", "remote"), required=True
    )
    internal_install_baseline = sub.add_parser(
        "_install-baseline-shard", help=argparse.SUPPRESS
    )
    _add_plan_argument(internal_install_baseline)
    internal_install_baseline.add_argument("--host", choices=("local",), required=True)
    internal_install_baseline.add_argument("--shard-index", type=int, required=True)
    internal_install_baseline.add_argument("--source", type=Path, required=True)
    internal_merge_baseline = sub.add_parser(
        "_merge-baseline-host", help=argparse.SUPPRESS
    )
    _add_plan_argument(internal_merge_baseline)
    internal_merge_baseline.add_argument("--host", choices=("local",), required=True)
    internal_plumbing = sub.add_parser("_plumbing-host", help=argparse.SUPPRESS)
    _add_plan_argument(internal_plumbing)
    internal_plumbing.add_argument("--host", choices=("local",), required=True)
    internal_ready = sub.add_parser("_ready-host", help=argparse.SUPPRESS)
    _add_plan_argument(internal_ready)
    internal_ready.add_argument("--host", choices=("local",), required=True)
    internal_install = sub.add_parser("_install-marker", help=argparse.SUPPRESS)
    _add_plan_argument(internal_install)
    internal_install.add_argument("--host", choices=("remote",), required=True)
    internal_install.add_argument(
        "--kind", choices=("baseline", "plumbing", "ready"), required=True
    )
    internal_install.add_argument("--source", type=Path, required=True)
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
                "B2: plan -> show -> stage -> baseline-preflight -> preflight "
                "-> plumbing-smoke -> execute "
                "[-> explicit resume] -> status -> collect"
            )
            print("B2 eval: plan-eval -> stage -> preflight -> execute -> collect -> merge-eval")
            print(
                "B3: plan-b3 -> show -> stage -> baseline-preflight -> preflight "
                "-> plumbing-smoke -> execute [-> explicit resume] -> status -> collect"
            )
            print("B3 eval: plan-eval -> stage -> preflight -> execute -> collect -> merge-eval")
            print(
                "B4 (OWNER-AUTHORIZED SEED1): plan-b4 -> show -> stage "
                "-> baseline-preflight -> preflight -> plumbing-smoke -> execute "
                "[-> explicit resume] -> status -> collect"
            )
            print(
                "B4 eval: plan-b4-eval -> stage -> preflight -> execute -> collect -> merge-eval"
            )
            for name in sorted(LEGACY_SHOW_ONLY):
                print(f"{name} [legacy show-only]")
            return 0
        if args.action == "legacy-show":
            print(LEGACY_SHOW_ONLY[args.name])
            return 0
        if args.action in {"plan", "plan-b3"}:
            built = build_training_plan(
                repo=REPO_ROOT,
                run_id=args.run_id,
                commit=args.source_commit,
                output=args.output.resolve(),
                local_gpu_uuid=args.local_gpu_uuid,
                remote_gpu_uuid=args.remote_gpu_uuid,
                kind="b3_train" if args.action == "plan-b3" else "b2_train",
            )
            print(args.output.resolve())
            print(built.plan_sha256)
            return 0
        if args.action == "plan-b4":
            built = build_b4_training_plan(
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
                source_commit=args.source_commit,
            )
            print(args.output.resolve())
            print(built.plan_sha256)
            return 0
        if args.action == "plan-b4-eval":
            built = build_b4_evaluation_plan(
                repo=REPO_ROOT,
                run_id=args.run_id,
                training_plan_path=args.training_plan.resolve(),
                checkpoints=args.checkpoint,
                output=args.output.resolve(),
                source_commit=args.source_commit,
            )
            print(args.output.resolve())
            print(built.plan_sha256)
            return 0
        plan_path = args.plan.resolve()
        loaded = load_plan(plan_path)
        if args.action == "show":
            _show(plan_path, loaded)
            return 0
        if args.action == "baseline-preflight":
            return baseline_preflight(loaded, args.dry_run)
        if args.action == "plumbing-smoke":
            return plumbing_smoke(loaded, args.dry_run)
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
        if args.action == "_check-stage-host":
            return check_stage_host(plan_path, args.host)
        if args.action == "_check-preflight-host":
            return check_preflight_host(plan_path, args.host)
        if args.action == "_preflight-host":
            return preflight_host(plan_path, args.host)
        if args.action == "_baseline-host":
            return baseline_host(plan_path, args.host)
        if args.action == "_install-baseline-shard":
            return install_baseline_shard(
                plan_path, args.host, args.shard_index, args.source
            )
        if args.action == "_merge-baseline-host":
            return merge_baseline_host(plan_path, args.host)
        if args.action == "_plumbing-host":
            return plumbing_host(plan_path, args.host)
        if args.action == "_ready-host":
            return ready_host(plan_path, args.host)
        if args.action == "_install-marker":
            return install_marker_host(plan_path, args.host, args.kind, args.source)
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
