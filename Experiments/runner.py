#!/usr/bin/env python3
"""Batch / unattended experiment runner.

Every long-running or unattended command in this project is declared here as a
Python job instead of being pasted into a shell. `run.sh` at the repository root
is the only entry point; it delegates to this module.

Why: unattended runs previously lived as ad-hoc ssh one-liners, which made them
impossible to review, re-run, or audit. A job here is a data structure — it can
be printed, dry-run, and hashed before anything executes on the GPU.

Usage (via ./run.sh):
    ./run.sh list                   # show every registered job
    ./run.sh show <job>             # print the exact command without running it
    ./run.sh run <job> [--local]    # run remotely (default) or locally
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_HOST = "haowei@192.168.2.127"
REMOTE_ROOT = "~/Documents/End2Race"
PYTHON = "~/miniconda3/envs/end2race/bin/python"


# Work split across the two GPUs, sized by their relative throughput.
# Local is an RTX 3080 Laptop; remote is an RTX 4080 SUPER. One quarter of the
# shards run here, three quarters on the remote host, IN PARALLEL.
#
# This is a throughput split, not a cross-check. Do NOT run the same shard on
# both hosts to compare them — that burns GPU for nothing. Cross-validate only
# when a result is obviously wrong (physically impossible value, huge run-to-run
# variance, or flat contradiction with a known baseline).
SHARD_COUNT = 4
LOCAL_SHARDS = (0,)
REMOTE_SHARDS = (1, 2, 3)


@dataclass(frozen=True)
class Shard:
    index: int
    count: int


@dataclass(frozen=True)
class Job:
    """One reviewable unit of unattended work."""

    name: str
    description: str
    argv: list[str]
    # Every simulator job needs an isolated absolute Numba cache; the runners
    # raise if it is missing, and a shared cache corrupted a Task-10 run once.
    numba_cache: str
    experiment: str
    env: dict[str, str] = field(default_factory=dict)
    # Whether the job's work can be partitioned by --shard-index/--shard-count.
    # PPO training and evaluation can; a preflight or a single fit cannot.
    shardable: bool = False

    def command(self, *, local: bool, shard: "Shard | None" = None) -> str:
        argv = list(self.argv)
        cache = self.numba_cache
        if shard is not None:
            if not self.shardable:
                raise ValueError(f"job {self.name} cannot be split across hosts")
            argv += ["--shard-index", str(shard.index), "--shard-count", str(shard.count)]
            cache = f"{cache}_shard{shard.index}"
        env = {"NUMBA_CACHE_DIR": cache, "PYTHONPATH": ".", **self.env}
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        python = sys.executable if local else PYTHON
        body = f"{prefix} {python} {' '.join(shlex.quote(a) for a in argv)}"
        if local:
            return body
        return f"ssh {REMOTE_HOST} {shlex.quote(f'cd {REMOTE_ROOT} && {body}')}"


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


B1 = "Experiments/B1_route_r2_scaffold"

JOBS: dict[str, Job] = {}


def register(job: Job) -> None:
    if job.name in JOBS:
        raise ValueError(f"duplicate job name: {job.name}")
    JOBS[job.name] = job


register(
    Job(
        name="b1-source-preflight",
        description=(
            "Legacy B1 source preflight. Kept so existing B1 stages still run; "
            "new work does NOT build hash manifests (see .agents/README.md)."
        ),
        experiment="B1_route_r2_scaffold",
        numba_cache="/tmp/end2race_numba_preflight",
        argv=[
            "-m", "bplus_v22.cli", "source-preflight",
            "--repo-root", ".",
            "--output-dir", f"{B1}/artifacts/source_preflight_{_stamp()}",
            "--created-at", _iso(),
        ],
    )
)

register(
    Job(
        name="b2-exploration-sweep",
        description=(
            "Zero-learning sweep of the intervention prior. Measures the "
            "closed-loop KPI cost of each exploration offset and selects the "
            "largest one inside the pre-registered damage bound. "
            "Blocked until bplus_v22/exploration.py exists (plan Task 1)."
        ),
        experiment="B2_ppo_pilot",
        numba_cache="/tmp/end2race_numba_explore",
        argv=[
            "-m", "bplus_v22.cli", "exploration-sweep",
            "--repo-root", ".",
            "--output-dir", f"Experiments/B2_ppo_pilot/artifacts/exploration_sweep_{_stamp()}",
            "--created-at", _iso(),
            "--device", "cuda:0",
        ],
    )
)

register(
    Job(
        name="b2-ppo-pilot-seed0",
        description=(
            "Three-arm PPO pilot from a BC-direct initialization, 20 iterations, "
            "snapshots evaluated on the frozen 288-scenario development panel. "
            "Blocked until the PPO training loop exists (plan Tasks 2-4)."
        ),
        experiment="B2_ppo_pilot",
        numba_cache="/tmp/end2race_numba_ppo",
        shardable=True,
        argv=[
            "-m", "bplus_v22.cli", "ppo-pilot",
            "--repo-root", ".",
            "--output-dir", f"Experiments/B2_ppo_pilot/artifacts/ppo_pilot_seed0_{_stamp()}",
            "--created-at", _iso(),
            "--seed", "0",
            "--device", "cuda:0",
        ],
    )
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run.sh", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="show every registered job")
    show = sub.add_parser("show", help="print a job's exact command, run nothing")
    show.add_argument("job")
    show.add_argument("--local", action="store_true")
    run = sub.add_parser("run", help="execute a job")
    run.add_argument("job")
    run.add_argument("--local", action="store_true", help="run here instead of the remote GPU host")
    split = sub.add_parser(
        "split",
        help="run a shardable job on BOTH hosts in parallel (1/4 local, 3/4 remote)",
    )
    split.add_argument("job")
    split.add_argument("--dry-run", action="store_true", help="print the shard commands, run nothing")
    args = parser.parse_args(argv)

    if args.action == "list":
        width = max(len(name) for name in JOBS)
        for name, job in JOBS.items():
            tag = " [shardable]" if job.shardable else ""
            print(
                f"{name:<{width}}  [{job.experiment}]{tag}  "
                f"{job.description.splitlines()[0]}"
            )
        return 0

    if args.job not in JOBS:
        print(f"unknown job: {args.job}\nknown: {', '.join(JOBS)}", file=sys.stderr)
        return 2
    job = JOBS[args.job]

    if args.action == "split":
        return _split(job, dry_run=args.dry_run)

    command = job.command(local=args.local)
    if args.action == "show":
        print(command)
        return 0

    print(f"[{job.experiment}] {job.name}", file=sys.stderr)
    print(command, file=sys.stderr)
    return subprocess.call(command, shell=True, cwd=REPO_ROOT)


def _split(job: Job, *, dry_run: bool) -> int:
    """Run one job's shards on both GPUs at once: 1/4 here, 3/4 remote."""

    if not job.shardable:
        print(f"job {job.name} is not shardable", file=sys.stderr)
        return 2

    plan = [(index, index in LOCAL_SHARDS) for index in range(SHARD_COUNT)]
    commands = [
        (index, local, job.command(local=local, shard=Shard(index, SHARD_COUNT)))
        for index, local in plan
    ]

    for index, local, command in commands:
        print(f"--- shard {index}/{SHARD_COUNT} on {'local' if local else 'remote'}", file=sys.stderr)
        print(command, file=sys.stderr)
    if dry_run:
        return 0

    # Launch every shard at once; the two GPUs work in parallel.
    running = [
        (index, subprocess.Popen(command, shell=True, cwd=REPO_ROOT))
        for index, _, command in commands
    ]
    failures = []
    for index, process in running:
        code = process.wait()
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"shard {index}: {status}", file=sys.stderr)
        if code != 0:
            failures.append(index)
    if failures:
        print(f"shards failed: {failures}", file=sys.stderr)
        return 1
    print(f"all {SHARD_COUNT} shards completed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
