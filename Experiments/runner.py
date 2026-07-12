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

    def command(self, *, local: bool) -> str:
        env = {"NUMBA_CACHE_DIR": self.numba_cache, "PYTHONPATH": ".", **self.env}
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        python = sys.executable if local else PYTHON
        body = f"{prefix} {python} {' '.join(shlex.quote(a) for a in self.argv)}"
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
        description="Hash and pin the current source tree before any B1 stage.",
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
    args = parser.parse_args(argv)

    if args.action == "list":
        width = max(len(name) for name in JOBS)
        for name, job in JOBS.items():
            print(f"{name:<{width}}  [{job.experiment}]  {job.description.splitlines()[0]}")
        return 0

    if args.job not in JOBS:
        print(f"unknown job: {args.job}\nknown: {', '.join(JOBS)}", file=sys.stderr)
        return 2
    job = JOBS[args.job]
    command = job.command(local=args.local)

    if args.action == "show":
        print(command)
        return 0

    print(f"[{job.experiment}] {job.name}", file=sys.stderr)
    print(command, file=sys.stderr)
    return subprocess.call(command, shell=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
