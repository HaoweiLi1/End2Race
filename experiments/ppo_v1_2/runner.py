"""Run-boundary-resumable, single-CUDA PPO V1.2 state machine."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time
from typing import Any

import torch

from model import End2Race
from utils import atomic_write_json

from .aggregate import TOP_COUNTS, global_aggregate, stage_aggregate
from .config_schema import HARD_POOL_IDS, STAGES, resolve_config
from .experiment_spec import BC_SHA256, EVALUATION_BASELINE, PROJECT_ROOT, austin_asset_hashes, canonical_hash, file_sha256
from .registry import validate_manifest
from .result_schema import validate_run_result


PYTHON = Path("/home/haowei/miniconda3/envs/end2race/bin/python")
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "SKIPPED_DEPENDENCY", "SKIPPED_EMPTY_POOL", "BLOCKED_IMPLEMENTATION"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def remaining_attempts(attempt_count: int) -> tuple[int, ...]:
    """Return the only legal fresh-attempt schedule (maximum two)."""

    if not 0 <= int(attempt_count) <= 2:
        raise ValueError("attempt_count must be in [0, 2]")
    return tuple(range(int(attempt_count) + 1, 3))


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


class SweepLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                document = json.loads(self.path.read_text(encoding="utf-8"))
                pid = int(document["pid"])
            except Exception:
                pid = -1
            if pid > 0 and self._alive(pid):
                raise RuntimeError(f"PPO V1.2 sweep lock belongs to live PID {pid}")
            self.path.unlink()
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "created_at": _utc_now()}, handle, sort_keys=True)
            handle.write("\n")
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> "SweepLock":
        self.acquire()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.release()


class Heartbeat:
    def __init__(self, path: Path, state: dict[str, Any], interval: float = 60.0) -> None:
        self.path = path
        self.state = state
        self.interval = float(interval)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    @staticmethod
    def _gpu_memory() -> str | None:
        try:
            return subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                text=True,
                timeout=5,
            ).strip()
        except Exception:
            return None

    def write(self) -> None:
        atomic_write_json(
            self.path,
            {
                **self.state,
                "pid": os.getpid(),
                "last_update": _utc_now(),
                "gpu_memory_mib": self._gpu_memory(),
            },
        )

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval):
            self.write()

    def start(self) -> None:
        self.write()
        self.thread = threading.Thread(target=self._loop, name="ppo-v1-2-heartbeat", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.write()


def selected_config(selections: dict[str, Any], stage: str, rank: int) -> dict[str, Any]:
    rows = selections[stage]["selected"]
    if not 1 <= rank <= len(rows):
        raise RuntimeError(f"Stage {stage} has no selected rank {rank}")
    return deepcopy(rows[rank - 1]["resolved_config"])


def materialize_arm(arm: dict[str, Any], selections: dict[str, Any], pool_hashes: dict[str, str | None]) -> dict[str, Any]:
    """Replace preregistered parent ranks with their concrete selected values."""

    config = deepcopy(arm["resolved_config"])
    original_seed = int(config["seed"])
    for name, reference in arm.get("parent_selections", {}).items():
        parent = selected_config(selections, str(reference["stage"]), int(reference["rank"]))
        if name == "critic_profile":
            config["critic_profile"] = parent["critic_profile"]
        elif name == "hard_configuration":
            for key in ("hard_pool_id", "hard_sampling_probability", "hard_sampling_mode"):
                config[key] = parent[key]
        elif name == "batch_size":
            config["batch_size"] = parent["batch_size"]
        elif name == "rollout":
            config["n_steps"] = parent["n_steps"]
        elif name == "kl_lr":
            for key in ("gru_lr", "head_lr", "critic_lr", "target_kl"):
                config[key] = parent[key]
        elif name == "exploration":
            for key in ("steering_latent_std", "speed_physical_std"):
                config[key] = parent[key]
        elif name == "gae":
            config["gae_lambda"] = parent["gae_lambda"]
        elif name == "reward":
            for key in ("reward_progress_weight", "reward_relative_weight", "reward_collision"):
                config[key] = parent[key]
        elif name == "full_configuration":
            runtime = {key: config[key] for key in ("experiment_profile", "evaluation_workers", "smoke", "bc_checkpoint")}
            config = deepcopy(parent)
            config.update(runtime)
            config["seed"] = original_seed
            config["master_seed"] = original_seed
        else:
            raise ValueError(f"Unknown parent selection field: {name}")

    if arm["stage"] in {"K", "E", "G", "W"}:
        transitions_per_update = 16 * int(config["n_steps"])
        config["updates"] = 204_800 // transitions_per_update
        config["evaluation_transition_budgets"] = [value for value in (51_200, 102_400, 204_800) if value % transitions_per_update == 0]
    elif arm["stage"] in {"X", "S"}:
        transitions_per_update = 16 * int(config["n_steps"])
        config["updates"] = 409_600 // transitions_per_update
        config["evaluation_transition_budgets"] = [value for value in (51_200, 102_400, 204_800, 409_600) if value % transitions_per_update == 0]
    config["hard_pool_hash"] = pool_hashes[config["hard_pool_id"]]
    runtime_fields = {key: config[key] for key in ("experiment_profile", "evaluation_workers", "smoke", "bc_checkpoint")}
    config = resolve_config(config)
    config.update(runtime_fields, master_seed=int(config["seed"]))
    materialized = deepcopy(arm)
    materialized["resolved_config"] = config
    materialized["config_hash"] = canonical_hash(config)
    materialized["seed"] = config["seed"]
    materialized["expected_transitions"] = config["expected_transitions"]
    materialized["evaluation_transition_budgets"] = config["evaluation_transition_budgets"]
    return materialized


class SweepRunner:
    def __init__(self, manifest_path: Path, run_root: Path, hard_pool_root: Path, bc_outcomes: Path, *, dry_run: bool = False) -> None:
        self.manifest_path = manifest_path.resolve()
        self.run_root = run_root.resolve()
        self.hard_pool_root = hard_pool_root.resolve()
        self.bc_outcomes = bc_outcomes.resolve()
        self.dry_run = bool(dry_run)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        validate_manifest(self.manifest)
        self.frozen_head = str(self.manifest["experiment_head"])
        self.selections: dict[str, Any] = {}
        self.state = {"current_stage": None, "current_arm": None, "start_time": _utc_now(), "completed_arms": 0, "total_arms": 125}
        self.pip_freeze = subprocess.check_output([str(PYTHON), "-m", "pip", "freeze"], text=True)

    def _assert_frozen(self) -> None:
        current_head = _git("rev-parse", "HEAD")
        worktree = _git("status", "--porcelain", "--untracked-files=normal")
        if current_head != self.frozen_head or worktree:
            raise RuntimeError(f"HEAD_OR_WORKTREE_DRIFT head={current_head} expected={self.frozen_head} worktree={worktree!r}")
        if file_sha256(PROJECT_ROOT / "pretrained" / "end2race.pth") != BC_SHA256:
            raise RuntimeError("CANONICAL_BC_HASH_DRIFT")
        if austin_asset_hashes() != self.manifest.get("austin_asset_hashes"):
            raise RuntimeError("AUSTIN_ASSET_HASH_DRIFT")

    def _verify_bc_baseline(self) -> None:
        baseline_dir = self.run_root / "baseline"
        summary_path = baseline_dir / "paired_bc_summary.json"
        rows_path = baseline_dir / "paired_bc_rows.json"
        if summary_path.is_file() and rows_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            from rl.ppo_scenarios import evaluation_scenarios
            from train_ppo_sb3 import evaluate_actor_pool

            rows, summary = evaluate_actor_pool(
                PROJECT_ROOT / "pretrained" / "end2race.pth",
                evaluation_scenarios(),
                workers=8,
            )
            atomic_write_json(rows_path, rows)
            atomic_write_json(summary_path, summary)
        observed = {key: int(summary[key]) for key in EVALUATION_BASELINE}
        if observed != EVALUATION_BASELINE:
            atomic_write_json(
                self.run_root / "EXPERIMENT_COMPLETION.json",
                {"status": "EVALUATION_DRIFT", "expected": EVALUATION_BASELINE, "actual": observed},
            )
            raise RuntimeError(f"EVALUATION_DRIFT expected={EVALUATION_BASELINE} actual={observed}")

    def _save_manifest(self) -> None:
        self.manifest["manifest_hash"] = canonical_hash({key: value for key, value in self.manifest.items() if key not in {"generated_at", "manifest_hash"}})
        atomic_write_json(self.manifest_path, self.manifest)

    def _pool_path(self, pool_id: str) -> Path:
        return self.hard_pool_root / "pools" / f"{pool_id}.json"

    def _command(self, attempt_dir: Path, config_path: Path, arm: dict[str, Any]) -> list[str]:
        return [
            str(PYTHON),
            str(PROJECT_ROOT / "train_ppo_sb3.py"),
            "--experiment-profile", "ppo_v1_2",
            "--config-json", str(config_path),
            "--run-root", str(attempt_dir.parent),
            "--run-id", attempt_dir.name,
            "--bc-outcomes", str(self.bc_outcomes),
            "--hard-pool-manifest", str(self._pool_path(arm["resolved_config"]["hard_pool_id"])),
            "--evaluation-workers", str(arm["resolved_config"].get("evaluation_workers", 8)),
        ]

    @staticmethod
    def _build_completed_result(arm: dict[str, Any], attempt_dir: Path, attempt: int) -> dict[str, Any]:
        selection = json.loads((attempt_dir / "selection.json").read_text(encoding="utf-8"))
        training = [json.loads(line) for line in (attempt_dir / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
        checkpoints = []
        for candidate in selection["candidates"]:
            update = int(candidate["update"])
            checkpoint = attempt_dir / "checkpoints" / f"update_{update:04d}" / "actor_only.pth"
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if len(state) != 12:
                raise ValueError(f"Actor checkpoint has {len(state)} keys: {checkpoint}")
            fresh = End2Race(mask_prob=0.0, hidden_scale=4)
            fresh.load_state_dict(state, strict=True)
            checkpoints.append(
                {
                    "update": update,
                    "checkpoint": str(checkpoint.relative_to(attempt_dir)),
                    "actor_sha256": file_sha256(checkpoint),
                    "valid": True,
                    "metrics": candidate["metrics"],
                }
            )
        selected = selection.get("best")
        if selected is not None:
            selected = next(row for row in checkpoints if row["update"] == int(selected["update"]))
        return {
            "schema_version": 1,
            "arm_id": arm["arm_id"],
            "stage": arm["stage"],
            "status": "COMPLETED",
            "attempt": attempt,
            "resolved_config": arm["resolved_config"],
            "config_hash": arm["config_hash"],
            "metadata": arm.get("metadata", {}),
            "checkpoints": checkpoints,
            "selected_checkpoint": selected,
            "actual_optimizer_steps": sum(int(row["actual_optimizer_steps"]) for row in training),
            "planned_optimizer_steps": sum(int(row["planned_optimizer_steps"]) for row in training),
            "hard_pool_id": arm["resolved_config"]["hard_pool_id"],
            "hard_pool_hash": arm["resolved_config"]["hard_pool_hash"],
            "hard_sampling_mode": arm["resolved_config"]["hard_sampling_mode"],
            "completed_at": _utc_now(),
        }

    @staticmethod
    def _failed_result(arm: dict[str, Any], attempt: int, returncode: int | None, error: str | None) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "arm_id": arm["arm_id"],
            "stage": arm["stage"],
            "status": "FAILED",
            "attempt": attempt,
            "resolved_config": arm["resolved_config"],
            "config_hash": arm["config_hash"],
            "metadata": arm.get("metadata", {}),
            "checkpoints": [],
            "selected_checkpoint": None,
            "returncode": returncode,
            "error": error,
            "completed_at": _utc_now(),
        }

    def _run_attempt(self, arm: dict[str, Any], attempt: int) -> dict[str, Any]:
        attempt_dir = self.run_root / arm["stage"] / arm["arm_id"] / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        config_path = attempt_dir / "resolved_config.request.json"
        atomic_write_json(config_path, arm["resolved_config"])
        command = self._command(attempt_dir, config_path, arm)
        (attempt_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        (attempt_dir / "git_status.txt").write_text(_git("status", "--short", "--branch") + "\n", encoding="utf-8")
        (attempt_dir / "pip_freeze.txt").write_text(self.pip_freeze, encoding="utf-8")
        runtime = {"pid": None, "started_at": _utc_now(), "experiment_head": self.frozen_head, "attempt": attempt}
        atomic_write_json(attempt_dir / "runtime.json", runtime)
        if self.dry_run:
            return self._failed_result(arm, attempt, None, "DRY_RUN_NO_TRAINING")
        try:
            with (attempt_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (attempt_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr, text=True)
                runtime["pid"] = process.pid
                atomic_write_json(attempt_dir / "runtime.json", runtime)
                returncode = process.wait()
            runtime.update(ended_at=_utc_now(), returncode=returncode)
            atomic_write_json(attempt_dir / "runtime.json", runtime)
            if returncode != 0:
                result = self._failed_result(arm, attempt, returncode, None)
            else:
                result = self._build_completed_result(arm, attempt_dir, attempt)
                validate_run_result(result, arm)
            atomic_write_json(attempt_dir / "run_result.json", result)
            atomic_write_json(attempt_dir / "validation.json", {"passed": result["status"] == "COMPLETED", "arm_id": arm["arm_id"], "config_hash": arm["config_hash"]})
            return result
        except Exception as error:
            runtime.update(ended_at=_utc_now(), error_type=type(error).__name__, error=str(error))
            atomic_write_json(attempt_dir / "runtime.json", runtime)
            result = self._failed_result(arm, attempt, None, f"{type(error).__name__}: {error}")
            atomic_write_json(attempt_dir / "run_result.json", result)
            atomic_write_json(attempt_dir / "validation.json", {"passed": False, "error": result["error"]})
            return result

    def _latest_result(self, arm: dict[str, Any]) -> dict[str, Any] | None:
        paths = sorted((self.run_root / arm["stage"] / arm["arm_id"]).glob("attempt_*/run_result.json"))
        return json.loads(paths[-1].read_text(encoding="utf-8")) if paths else None

    def _terminal_placeholder(self, arm: dict[str, Any]) -> dict[str, Any]:
        return {
            "arm_id": arm["arm_id"], "stage": arm["stage"], "status": arm["status"], "attempt": arm["attempt_count"],
            "resolved_config": arm["resolved_config"], "config_hash": arm["config_hash"], "metadata": arm.get("metadata", {}),
            "checkpoints": [], "selected_checkpoint": None,
        }

    def run(self) -> dict[str, Any]:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._assert_frozen()
        if not self.dry_run:
            self._verify_bc_baseline()
        heartbeat = Heartbeat(self.run_root / "heartbeat.json", self.state)
        with SweepLock(self.run_root / "SWEEP.lock"):
            heartbeat.start()
            try:
                for stage in STAGES:
                    self.state["current_stage"] = stage
                    stage_arms = [arm for arm in self.manifest["arms"] if arm["stage"] == stage]
                    for index, original_arm in enumerate(stage_arms):
                        if original_arm["status"] in TERMINAL_STATUSES:
                            continue
                        recovered = self._latest_result(original_arm)
                        if recovered is not None and recovered.get("status") == "COMPLETED":
                            original_arm["status"] = "COMPLETED"
                            original_arm["attempt_count"] = int(recovered["attempt"])
                            self._save_manifest()
                            continue
                        self._assert_frozen()
                        arm = materialize_arm(original_arm, self.selections, self.manifest["hard_pool_hashes"])
                        original_arm.update({key: arm[key] for key in ("resolved_config", "config_hash", "seed", "expected_transitions", "evaluation_transition_budgets")})
                        self.state["current_arm"] = arm["arm_id"]
                        heartbeat.write()
                        final_result = None
                        for attempt in remaining_attempts(int(original_arm["attempt_count"])):
                            original_arm["status"] = "RUNNING"
                            original_arm["attempt_count"] = attempt
                            self._save_manifest()
                            final_result = self._run_attempt(arm, attempt)
                            if final_result["status"] == "COMPLETED":
                                break
                        original_arm["status"] = final_result["status"] if final_result is not None else "FAILED"
                        self._save_manifest()
                        self.state["completed_arms"] = sum(item["status"] in TERMINAL_STATUSES for item in self.manifest["arms"])
                        heartbeat.write()
                        self._assert_frozen()

                    results = []
                    for arm in stage_arms:
                        result = self._latest_result(arm)
                        results.append(result if result is not None else self._terminal_placeholder(arm))
                    if any(arm["status"] not in TERMINAL_STATUSES for arm in stage_arms):
                        raise RuntimeError(f"Stage {stage} did not reach a barrier")
                    selection = stage_aggregate(stage, results, self.run_root / stage)
                    self.selections[stage] = selection
                    if TOP_COUNTS[stage] and not selection["selected"]:
                        atomic_write_json(self.run_root / "EXPERIMENT_COMPLETION.json", {"status": f"BLOCKED_STAGE_{stage}"})
                        raise RuntimeError(f"BLOCKED_STAGE_{stage}")
                    self._save_manifest()

                all_results = []
                for arm in self.manifest["arms"]:
                    result = self._latest_result(arm)
                    all_results.append(result if result is not None else self._terminal_placeholder(arm))
                completion = global_aggregate(self.run_root, all_results, self.selections)
                self._write_report(all_results, completion)
                return completion
            finally:
                self.state["current_arm"] = None
                heartbeat.close()

    def _write_report(self, results: list[dict[str, Any]], completion: dict[str, Any]) -> None:
        lines = ["# PPO V1.2 REPORT", "", "| status | count |", "|---|---:|"]
        for status, count in sorted(completion["status_counts"].items()):
            lines.append(f"| {status} | {count} |")
        for stage in STAGES:
            lines.extend(["", f"## Stage {stage}", "", "| rank | arm_id | ego_collision | overtake |", "|---:|---|---:|---:|"])
            rank_path = self.run_root / stage / "stage_rank.json"
            ranks = json.loads(rank_path.read_text(encoding="utf-8")) if rank_path.is_file() else []
            for row in ranks:
                lines.append(f"| {row['rank']} | {row['arm_id']} | {row.get('selected_ego_collision')} | {row.get('selected_overtake')} |")
        lines.extend(["", "## Fixed numeric references", "", "| version | ego_collision | follow | overtake |", "|---|---:|---:|---:|", "| BC | 21 | 233 | 346 |", "| V1 best | 17 | 236 | 347 |", "| V1.1 best | 15 | 232 | 353 |"])
        (self.run_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
