#!/usr/bin/env python3
"""Execute the preregistered quick-pool / mixed-horizon experiment."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "ppo_experiments" / "quick_pool_3s"
RUN_ROOT = ROOT / "runs" / "ppo" / "quick_pool_3s"
EVALUATOR = OUTPUT / "evaluate_pool.py"
BC_CHECKPOINT = ROOT / "pretrained" / "end2race.pth"
H1_MANIFEST = ROOT / "ppo" / "hard_pools" / "h1_expanded_det.json"
SEED = 20260718
WORKERS = 8
HISTORICAL_BC_COMMIT = "228ff08f3210bc54cbd107216ba9b9a2646cd022"
HISTORICAL_BC_PATH = "runs/ppo_v1_2/baseline/paired_bc_rows.json"
HISTORICAL_BC_SHA256 = "17579067f3309ad75fd71174efd8863da9e2f605be27b3e19e8b6b75bbf50bfa"
ARMS = (
    "QP3_A0_H0_8S",
    "QP3_A1_H1FULL_8S",
    "QP3_A2_H1EARLY_8S",
    "QP3_A3_H1EARLY_3S",
)
HARD_HORIZONS = {
    "QP3_A0_H0_8S": 8.0,
    "QP3_A1_H1FULL_8S": 8.0,
    "QP3_A2_H1EARLY_8S": 8.0,
    "QP3_A3_H1EARLY_3S": 3.0,
}


COMMANDS: list[dict[str, Any]] = []
PROCESSES: dict[str, tuple[subprocess.Popen[str], Any]] = {}


class IntegrityFailure(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(document: dict[str, Any]) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, document: Any) -> None:
    from utils import atomic_write_json

    atomic_write_json(path, document)


def record_command(command: list[str], purpose: str) -> None:
    COMMANDS.append({"purpose": purpose, "argv": command, "recorded_at": utc_now()})
    write_json(OUTPUT / "COMMANDS.json", COMMANDS)
    print("COMMAND " + " ".join(command), flush=True)


def run_command(command: list[str], purpose: str) -> None:
    record_command(command, purpose)
    subprocess.run(command, cwd=ROOT, check=True)


def run_evaluation(
    model_path: Path,
    output_path: Path,
    *,
    manifest: Path | None = None,
    sim_duration: float | None = None,
    purpose: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-u",
        str(EVALUATOR),
        "--model-path",
        str(model_path),
        "--output",
        str(output_path),
        "--workers",
        str(WORKERS),
    ]
    if manifest is not None:
        command.extend(("--scenario-manifest", str(manifest)))
    if sim_duration is not None:
        command.extend(("--sim-duration", str(sim_duration)))
    run_command(command, purpose)
    document = read_json(output_path)
    if not document.get("complete") or document["summary"]["error"] != 0:
        raise IntegrityFailure(f"Incomplete evaluation: {output_path}")
    return document


def hard_pool_document(scenarios: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    scenarios = sorted(scenarios, key=lambda row: str(row["scenario_id"]))
    ids = [str(row["scenario_id"]) for row in scenarios]
    if not scenarios or len(ids) != len(set(ids)):
        raise IntegrityFailure("H1_EARLY_3S must contain unique non-empty scenario IDs")
    content = {
        "pool_id": "H1_EARLY_3S",
        "scenario_ids": ids,
        "scenarios": scenarios,
        "count": len(ids),
        "distributions": {
            "interval_idx": dict(sorted(Counter(str(row["interval_idx"]) for row in scenarios).items())),
            "opp_raceline": dict(sorted(Counter(str(row["opp_raceline"]) for row in scenarios).items())),
            "opp_speedscale": dict(
                sorted(Counter(f"{float(row['opp_speedscale']):.2f}" for row in scenarios).items())
            ),
            "startpoint_ordinal": dict(
                sorted(
                    Counter(str(row["startpoint_ordinal"]) for row in scenarios).items(),
                    key=lambda item: int(item[0]),
                )
            ),
        },
        "source": source,
    }
    return {**content, "manifest_hash": canonical_hash(content)}


def build_h1_early() -> tuple[dict[str, Any], dict[str, Any]]:
    classification_path = OUTPUT / "H1_3S_CLASSIFICATION.json"
    classification = run_evaluation(
        BC_CHECKPOINT,
        classification_path,
        manifest=H1_MANIFEST,
        sim_duration=3.0,
        purpose="Classify all H1 deterministic cases under canonical BC for 3 seconds",
    )
    h1 = read_json(H1_MANIFEST)
    by_id = {str(row["scenario_id"]): row for row in h1["scenarios"]}
    if len(by_id) != 482 or classification["summary"]["total"] != 482:
        raise IntegrityFailure("H1 classification does not cover the canonical 482-case manifest")

    collision_times = {
        str(row["scenario_id"]): float(row["ego_collision_time_s"])
        for row in classification["rows"]
        if row.get("outcome") == "ego_collision" and row.get("ego_collision_time_s") is not None
    }
    selected_ids = sorted(
        scenario_id for scenario_id, collision_time in collision_times.items() if collision_time <= 2.8 + 1e-12
    )
    threshold = 2.8
    if len(selected_ids) < 32:
        threshold = 3.0
        selected_ids = sorted(
            scenario_id for scenario_id, collision_time in collision_times.items() if collision_time <= 3.0 + 1e-12
        )
    summary = {
        "schema_version": 1,
        "status": "BUILT",
        "source_h1_manifest": str(H1_MANIFEST.relative_to(ROOT)),
        "source_h1_manifest_hash": h1["manifest_hash"],
        "classification_path": str(classification_path.relative_to(ROOT)),
        "classification_sha256": sha256_file(classification_path),
        "classification_total": classification["summary"]["total"],
        "classification_collision_within_3s": len(collision_times),
        "filter_threshold_s": threshold,
        "unique_scenario_count": len(selected_ids),
    }
    if len(selected_ids) < 24:
        summary.update({"status": "STOP_3S_POOL_TOO_SMALL", "stop_reason": "fewer than 24 unique cases"})
        write_json(OUTPUT / "H1_EARLY_3S_SUMMARY.json", summary)
        raise IntegrityFailure("STOP_3S_POOL_TOO_SMALL")
    source = {
        "h1_manifest_hash": h1["manifest_hash"],
        "classification_sha256": summary["classification_sha256"],
        "filter_threshold_s": threshold,
    }
    early = hard_pool_document([by_id[scenario_id] for scenario_id in selected_ids], source)
    write_json(OUTPUT / "H1_EARLY_3S.json", early)

    precheck = run_evaluation(
        BC_CHECKPOINT,
        OUTPUT / "H1_EARLY_3S_PRECHECK_RESULTS.json",
        manifest=OUTPUT / "H1_EARLY_3S.json",
        sim_duration=3.0,
        purpose="Reproduce every H1_EARLY_3S case under canonical BC",
    )
    rows = precheck["rows"]
    reproduction = precheck["summary"]["collision"] / len(rows)
    initial_collision_count = sum(
        bool(row["initial_ego_collision"]) or bool(row["initial_opponent_collision"]) for row in rows
    )
    finite_count = sum(bool(row["observation_finite"]) and bool(row["action_finite"]) for row in rows)
    summary.update(
        {
            "status": "PASS" if reproduction >= 0.90 else "STOP_3S_REPRODUCTION_FAILED",
            "manifest_hash": early["manifest_hash"],
            "initial_collision_count": initial_collision_count,
            "finite_episode_count": finite_count,
            "reproduced_collision_count": precheck["summary"]["collision"],
            "reproduction_rate": reproduction,
            "evaluation_error": precheck["summary"]["error"],
        }
    )
    write_json(OUTPUT / "H1_EARLY_3S_SUMMARY.json", summary)
    if initial_collision_count != 0 or finite_count != len(rows) or reproduction < 0.90:
        raise IntegrityFailure("STOP_3S_REPRODUCTION_FAILED")
    return early, summary


def evenly_spaced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row["scenario_id"]))
    indices = np.linspace(0, len(ordered) - 1, count, dtype=np.int64)
    if len(set(map(int, indices))) != count:
        raise IntegrityFailure("Evenly spaced selection produced duplicate indices")
    return [ordered[int(index)] for index in indices]


def historical_bc_rows() -> list[dict[str, Any]]:
    payload = subprocess.check_output(
        ["git", "show", f"{HISTORICAL_BC_COMMIT}:{HISTORICAL_BC_PATH}"],
        cwd=ROOT,
    )
    if sha256_bytes(payload) != HISTORICAL_BC_SHA256:
        raise IntegrityFailure("Historical canonical BC rows hash mismatch")
    rows = json.loads(payload)
    counts = Counter(str(row["outcome"]) for row in rows)
    if len(rows) != 600 or counts != Counter({"overtake": 346, "follow": 233, "ego_collision": 21}):
        raise IntegrityFailure(f"Historical canonical BC composition mismatch: {counts}")
    return rows


def build_quick_panel() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = historical_bc_rows()
    selected_rows = (
        [row for row in rows if row["outcome"] == "ego_collision"]
        + evenly_spaced([row for row in rows if row["outcome"] == "follow"], 49)
        + evenly_spaced([row for row in rows if row["outcome"] == "overtake"], 50)
    )
    selected_rows.sort(key=lambda row: str(row["scenario_id"]))
    scenario_fields = {
        "scenario_id",
        "pool",
        "startpoint_ordinal",
        "ego_idx",
        "opp_idx",
        "opp_raceline",
        "opp_speedscale",
        "map_name",
        "ego_raceline",
        "interval_idx",
        "sim_duration",
        "timestep",
        "integrator",
    }
    content = {
        "panel_id": "QUICK_PANEL_120",
        "source": {
            "commit": HISTORICAL_BC_COMMIT,
            "path": HISTORICAL_BC_PATH,
            "rows_sha256": HISTORICAL_BC_SHA256,
        },
        "selection": {
            "collision": 21,
            "follow": 49,
            "overtake": 50,
            "follow_overtake_rule": "scenario_id sort then integer np.linspace indices",
        },
        "scenario_ids": [str(row["scenario_id"]) for row in selected_rows],
        "baseline_outcomes": {
            str(row["scenario_id"]): str(row["outcome"]) for row in selected_rows
        },
        "scenarios": [
            {key: value for key, value in row.items() if key in scenario_fields} for row in selected_rows
        ],
        "count": len(selected_rows),
    }
    panel = {**content, "manifest_hash": canonical_hash(content)}
    if panel["count"] != 120 or len(set(panel["scenario_ids"])) != 120:
        raise IntegrityFailure("Quick panel is not exactly 120 unique scenarios")
    write_json(OUTPUT / "QUICK_PANEL_120.json", panel)
    baseline = run_evaluation(
        BC_CHECKPOINT,
        OUTPUT / "QUICK_PANEL_BC_RESULTS.json",
        manifest=OUTPUT / "QUICK_PANEL_120.json",
        sim_duration=8.0,
        purpose="Verify canonical BC on the frozen 120-case quick panel",
    )
    expected = {"collision": 21, "follow": 49, "overtake": 50, "error": 0, "total": 120}
    if baseline["summary"] != expected:
        raise IntegrityFailure(f"Quick-panel BC composition mismatch: {baseline['summary']}")
    return panel, baseline


def run_dir(arm: str) -> Path:
    return RUN_ROOT / f"{arm}_seed{SEED}"


def checkpoint_path(arm: str, update: int) -> Path:
    return run_dir(arm) / "checkpoints" / f"end2race_ppo_{arm}_u{update:04d}_s{SEED}.pth"


def read_status(arm: str) -> dict[str, Any] | None:
    path = run_dir(arm) / "run_status.json"
    try:
        return read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def wait_for_status(arm: str, expected: str, timeout_s: float = 7200.0) -> None:
    process, _handle = PROCESSES[arm]
    started = time.monotonic()
    last_report = 0.0
    while True:
        status = read_status(arm)
        if status is not None and status.get("status") == expected:
            print(f"TRAIN_STATUS arm={arm} status={expected}", flush=True)
            return
        return_code = process.poll()
        if return_code is not None:
            raise IntegrityFailure(
                f"Trainer {arm} exited {return_code} before {expected}; status={status}"
            )
        elapsed = time.monotonic() - started
        if elapsed > timeout_s:
            raise IntegrityFailure(f"Trainer {arm} timed out waiting for {expected}")
        if elapsed - last_report >= 20.0:
            print(f"TRAIN_WAIT arm={arm} target={expected} elapsed={elapsed:.0f}s status={status}", flush=True)
            last_report = elapsed
        time.sleep(2.0)


def launch_to_screen_pause(arm: str) -> None:
    destination = run_dir(arm)
    if destination.exists():
        raise IntegrityFailure(f"Fresh run destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT / "training_logs" / f"{arm}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-u",
        str(ROOT / "train_ppo.py"),
        "--config",
        arm,
        "--seed",
        str(SEED),
        "--output_dir",
        str(destination),
        "--screen-pause",
    ]
    record_command(command, f"Train {arm} through the screen checkpoint and pause in-process")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    PROCESSES[arm] = (process, handle)
    wait_for_status(arm, "PAUSED_SCREEN")
    if not checkpoint_path(arm, 2).is_file():
        raise IntegrityFailure(f"Screen checkpoint missing: {checkpoint_path(arm, 2)}")


def send_decision(arm: str, decision: str, expected: str) -> None:
    process, handle = PROCESSES[arm]
    if process.stdin is None:
        raise IntegrityFailure(f"Trainer stdin is unavailable: {arm}")
    process.stdin.write(decision + "\n")
    process.stdin.flush()
    wait_for_status(arm, expected)
    return_code = process.wait(timeout=60)
    handle.close()
    if return_code != 0:
        raise IntegrityFailure(f"Trainer {arm} exited {return_code} after {decision}")


def paired_metrics(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, int]:
    baseline_by_id = {str(row["scenario_id"]): str(row["outcome"]) for row in baseline["rows"]}
    candidate_by_id = {str(row["scenario_id"]): str(row["outcome"]) for row in candidate["rows"]}
    if set(baseline_by_id) != set(candidate_by_id):
        raise IntegrityFailure("Paired evaluation scenario IDs differ")
    fixed = sum(
        baseline_by_id[key] == "ego_collision" and candidate_by_id[key] != "ego_collision"
        for key in baseline_by_id
    )
    new = sum(
        baseline_by_id[key] != "ego_collision" and candidate_by_id[key] == "ego_collision"
        for key in baseline_by_id
    )
    gained = sum(
        baseline_by_id[key] != "overtake" and candidate_by_id[key] == "overtake"
        for key in baseline_by_id
    )
    lost = sum(
        baseline_by_id[key] == "overtake" and candidate_by_id[key] != "overtake"
        for key in baseline_by_id
    )
    return {
        "fixed_collision": fixed,
        "new_collision": new,
        "gained_overtake": gained,
        "lost_overtake": lost,
        "net_collision_repair": fixed - new,
        "net_overtake_gain": gained - lost,
    }


def training_summary(arm: str, through_update: int) -> dict[str, Any]:
    metrics_path = run_dir(arm) / "training_metrics.jsonl"
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if int(row["update"]) <= through_update]
    if [int(row["update"]) for row in rows] != list(range(1, through_update + 1)):
        raise IntegrityFailure(f"Training metrics are incomplete for {arm} through update {through_update}")
    outcomes = ("ego_collision", "follow", "overtake")
    completed = {
        role: sum(
            int(row["rollout"]["completed_episodes_by_env_role"][role][outcome])
            for row in rows
            for outcome in outcomes
        )
        for role in ("hard", "ordinary")
    }
    length_totals = {
        role: sum(
            float(row["rollout"]["mean_episode_length_steps_by_env_role"][role])
            * sum(
                int(row["rollout"]["completed_episodes_by_env_role"][role][outcome])
                for outcome in outcomes
            )
            for row in rows
        )
        for role in ("hard", "ordinary")
    }
    sampler = read_json(run_dir(arm) / "sampler_summary.json")
    reward_keys = ("reward_progress", "reward_relative", "reward_margin", "reward_collision", "reward_total")
    return {
        "hard_transitions": sum(int(row["rollout"]["env_role_transitions"]["hard"]) for row in rows),
        "ordinary_transitions": sum(int(row["rollout"]["env_role_transitions"]["ordinary"]) for row in rows),
        "hard_completed_episodes": completed["hard"],
        "ordinary_completed_episodes": completed["ordinary"],
        "hard_unique_scenarios": int(sampler["visited_hard_scenarios"]),
        "hard_scenario_visit_distribution": sampler["visit_counts"],
        "hard_collision_episodes": sum(
            int(row["rollout"]["completed_episodes_by_env_role"]["hard"]["ego_collision"])
            for row in rows
        ),
        "hard_truncations": sum(int(row["rollout"]["hard_truncations"]) for row in rows),
        "hard_mean_episode_length_steps": length_totals["hard"] / max(completed["hard"], 1),
        "ordinary_mean_episode_length_steps": length_totals["ordinary"] / max(completed["ordinary"], 1),
        "reward_component_sums": {
            key: sum(float(row["rollout"]["reward_component_sums"][key]) for row in rows)
            for key in reward_keys
        },
        "approx_kl": float(rows[-1]["approx_kl"]),
        "approx_kl_by_update": [float(row["approx_kl"]) for row in rows],
        "clip_fraction_by_update": [float(row.get("clip_fraction", 0.0)) for row in rows],
        "actor_delta_from_bc": rows[-1]["actor_delta_from_bc"],
    }


def arm_result(
    arm: str,
    update: int,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    evaluation = {**candidate["summary"], **paired_metrics(baseline, candidate)}
    result = {
        "arm_id": arm,
        "seed": SEED,
        "transitions": update * 16 * 800,
        "hard_pool": {
            "QP3_A0_H0_8S": "H0_CURRENT_DET",
            "QP3_A1_H1FULL_8S": "H1_EXPANDED_DET",
            "QP3_A2_H1EARLY_8S": "H1_EARLY_3S",
            "QP3_A3_H1EARLY_3S": "H1_EARLY_3S",
        }[arm],
        "hard_horizon_s": HARD_HORIZONS[arm],
        "ordinary_horizon_s": 8.0,
        "checkpoint": str(checkpoint_path(arm, update).relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path(arm, update)),
        "evaluation": evaluation,
        "training": training_summary(arm, update),
    }
    return result


def evaluate_stage(
    arms: tuple[str, ...],
    update: int,
    baseline: dict[str, Any],
    stage: str,
) -> list[dict[str, Any]]:
    results = []
    panel_path = OUTPUT / "QUICK_PANEL_120.json"
    for arm in arms:
        raw = run_evaluation(
            checkpoint_path(arm, update),
            OUTPUT / f"{stage.lower()}_eval_{arm}.json",
            manifest=panel_path,
            sim_duration=8.0,
            purpose=f"Evaluate {arm} update {update} on QUICK_PANEL_120",
        )
        result = arm_result(arm, update, baseline, raw)
        evaluation = result["evaluation"]
        basic_pass = (
            evaluation["error"] == 0
            and evaluation["collision"] <= 19
            and evaluation["net_collision_repair"] >= 2
            and evaluation["overtake"] >= 48
        )
        if stage == "SCREEN":
            result["screen_pass"] = basic_pass
        results.append(result)
    return results


def write_markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| arm | collision | follow | overtake | fixed | new | net repair | pass |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        evaluation = row["evaluation"]
        passed = row.get("screen_pass", row.get("retention_pass", False))
        lines.append(
            f"| {row['arm_id']} | {evaluation['collision']} | {evaluation['follow']} | "
            f"{evaluation['overtake']} | {evaluation['fixed_collision']} | "
            f"{evaluation['new_collision']} | {evaluation['net_collision_repair']} | "
            f"{'PASS' if passed else 'FAIL'} |"
        )
    return lines


def write_final(
    verdict: str,
    h1_summary: dict[str, Any] | None,
    panel: dict[str, Any] | None,
    screen: list[dict[str, Any]] | None,
    retention: list[dict[str, Any]] | None,
    full: list[dict[str, Any]] | None,
    integrity_failure: str | None = None,
) -> None:
    if verdict not in {
        "POOL_AND_3S_SUPPORTED",
        "POOL_SUPPORTED_3S_NOT_SUPPORTED",
        "NO_QUICK_SIGNAL",
        "INVALID_EXPERIMENT",
    }:
        raise ValueError(verdict)
    lines = [f"# {verdict}", "", f"Completed: {utc_now()}", ""]
    if integrity_failure:
        lines.extend(("## Integrity failure", "", integrity_failure, ""))
    if h1_summary:
        lines.extend(
            (
                "## Phase 0",
                "",
                f"- H1_EARLY_3S count: {h1_summary['unique_scenario_count']}",
                f"- Filter threshold: {h1_summary['filter_threshold_s']} s",
                f"- BC reproduction: {h1_summary.get('reproduced_collision_count', 0)}/"
                f"{h1_summary['unique_scenario_count']} ({h1_summary.get('reproduction_rate', 0.0):.3%})",
                "",
            )
        )
    if panel:
        lines.extend(
            (
                "## Frozen quick panel",
                "",
                f"- Manifest hash: `{panel['manifest_hash']}`",
                "- BC composition: collision 21, follow 49, overtake 50, error 0",
                "",
            )
        )
    if screen:
        lines.extend(("## Screen", "", *write_markdown_table(screen), ""))
    if retention:
        lines.extend(("## Retention", "", *write_markdown_table(retention), ""))
    if full:
        lines.extend(("## Full 600", "", *write_markdown_table(full), ""))
    changed = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
    lines.extend(("## Changed files", "", "```text", *changed, "```", ""))
    lines.extend(("## Exact commands", ""))
    for row in COMMANDS:
        lines.append("```text")
        lines.append(" ".join(row["argv"]))
        lines.append("```")
        lines.append("")
    (OUTPUT / "FINAL_VERDICT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT / "EXECUTION_LOG.md").write_text("\n".join(lines), encoding="utf-8")


def stop_live_processes() -> None:
    for arm, (process, handle) in PROCESSES.items():
        if process.poll() is not None:
            if not handle.closed:
                handle.close()
            continue
        status = read_status(arm)
        try:
            if status and status.get("status") == "PAUSED_SCREEN" and process.stdin is not None:
                process.stdin.write("stop\n")
                process.stdin.flush()
                process.wait(timeout=30)
            else:
                process.terminate()
                process.wait(timeout=30)
        except Exception:
            process.kill()
            process.wait(timeout=30)
        if not handle.closed:
            handle.close()


def main() -> int:
    os.environ.setdefault("NUMBA_CACHE_DIR", str(OUTPUT / ".numba_cache"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    h1_summary = None
    panel = None
    screen = None
    retention = None
    full = None
    try:
        _early, h1_summary = build_h1_early()
        panel, baseline = build_quick_panel()

        for arm in ARMS:
            launch_to_screen_pause(arm)

        screen = evaluate_stage(ARMS, 2, baseline, "SCREEN")
        screen_document = {
            "schema_version": 1,
            "stage": "SCREEN",
            "baseline": baseline["summary"],
            "arms": screen,
            "any_screen_pass": any(row["screen_pass"] for row in screen),
        }
        write_json(OUTPUT / "SCREEN_RESULTS.json", screen_document)
        if not screen_document["any_screen_pass"]:
            for arm in ARMS:
                send_decision(arm, "stop", "STOPPED_SCREEN")
            write_json(OUTPUT / "RETENTION_RESULTS.json", {"status": "NOT_RUN_ALL_SCREEN_FAILED"})
            write_json(OUTPUT / "FULL600_RESULTS.json", {"status": "NOT_RUN"})
            write_final("NO_QUICK_SIGNAL", h1_summary, panel, screen, None, None)
            return 0

        send_decision("QP3_A1_H1FULL_8S", "stop", "STOPPED_SCREEN")
        for arm in ("QP3_A0_H0_8S", "QP3_A2_H1EARLY_8S", "QP3_A3_H1EARLY_3S"):
            send_decision(arm, "continue", "COMPLETED")

        retention = evaluate_stage(
            ("QP3_A0_H0_8S", "QP3_A2_H1EARLY_8S", "QP3_A3_H1EARLY_3S"),
            4,
            baseline,
            "RETENTION",
        )
        by_arm = {row["arm_id"]: row for row in retention}
        screen_by_arm = {row["arm_id"]: row for row in screen}
        for row in retention:
            evaluation = row["evaluation"]
            basic = (
                evaluation["error"] == 0
                and evaluation["collision"] <= 19
                and evaluation["fixed_collision"] > evaluation["new_collision"]
                and evaluation["overtake"] >= 48
            )
            row["basic_retention_pass"] = basic
            row["retention_pass"] = basic
        a2 = by_arm["QP3_A2_H1EARLY_8S"]
        a3 = by_arm["QP3_A3_H1EARLY_3S"]
        a3_eval = a3["evaluation"]
        a2_eval = a2["evaluation"]
        a3["retention_pass"] = bool(
            a3["basic_retention_pass"]
            and (
                a3_eval["collision"] <= a2_eval["collision"]
                or a3_eval["net_collision_repair"] >= a2_eval["net_collision_repair"] + 2
            )
            and a3_eval["collision"]
            <= screen_by_arm["QP3_A3_H1EARLY_3S"]["evaluation"]["collision"] + 2
        )
        retention_document = {
            "schema_version": 1,
            "stage": "RETENTION",
            "baseline": baseline["summary"],
            "arms": retention,
        }
        write_json(OUTPUT / "RETENTION_RESULTS.json", retention_document)
        if not a2["retention_pass"] and not a3["retention_pass"]:
            write_json(OUTPUT / "FULL600_RESULTS.json", {"status": "NOT_RUN_RETENTION_FAILED"})
            write_final("NO_QUICK_SIGNAL", h1_summary, panel, screen, retention, None)
            return 0

        historical_rows = historical_bc_rows()
        historical_baseline = {
            "rows": historical_rows,
            "summary": {"collision": 21, "follow": 233, "overtake": 346, "error": 0, "total": 600},
        }
        full = []
        for arm in ("QP3_A0_H0_8S", "QP3_A2_H1EARLY_8S", "QP3_A3_H1EARLY_3S"):
            raw = run_evaluation(
                checkpoint_path(arm, 4),
                OUTPUT / f"full600_eval_{arm}.json",
                purpose=f"Evaluate {arm} update 4 on the original 600-case panel",
            )
            row = arm_result(arm, 4, historical_baseline, raw)
            row["full600_forward"] = False
            full.append(row)
        full_by_arm = {row["arm_id"]: row for row in full}
        a0_full = full_by_arm["QP3_A0_H0_8S"]["evaluation"]
        for arm in ("QP3_A2_H1EARLY_8S", "QP3_A3_H1EARLY_3S"):
            evaluation = full_by_arm[arm]["evaluation"]
            full_by_arm[arm]["full600_forward"] = bool(
                evaluation["collision"] < a0_full["collision"]
                and evaluation["fixed_collision"] > evaluation["new_collision"]
                and evaluation["overtake"] >= 329
            )
        full_a2 = full_by_arm["QP3_A2_H1EARLY_8S"]
        full_a3 = full_by_arm["QP3_A3_H1EARLY_3S"]
        f2 = full_a2["evaluation"]
        f3 = full_a3["evaluation"]
        three_second_consistent = bool(
            a3["retention_pass"]
            and full_a3["full600_forward"]
            and (f3["collision"] < f2["collision"] or f3["net_collision_repair"] >= f2["net_collision_repair"] + 2)
            and f3["overtake"] >= f2["overtake"] - 2
        )
        write_json(
            OUTPUT / "FULL600_RESULTS.json",
            {
                "schema_version": 1,
                "stage": "FULL600",
                "baseline": historical_baseline["summary"],
                "arms": full,
                "three_second_consistent": three_second_consistent,
            },
        )
        if three_second_consistent:
            verdict = "POOL_AND_3S_SUPPORTED"
        elif full_a2["full600_forward"] or full_a3["full600_forward"]:
            verdict = "POOL_SUPPORTED_3S_NOT_SUPPORTED"
        else:
            verdict = "NO_QUICK_SIGNAL"
        write_final(verdict, h1_summary, panel, screen, retention, full)
        return 0
    except Exception as error:
        stop_live_processes()
        failure = f"{type(error).__name__}: {error}"
        panel_path = OUTPUT / "QUICK_PANEL_120.json"
        baseline_path = OUTPUT / "QUICK_PANEL_BC_RESULTS.json"
        gate_record: dict[str, Any] = {
            "status": "NOT_RUN_INVALID_PRECHECK",
            "gate": "QUICK_PANEL_BC_COMPOSITION",
            "training_started": bool(PROCESSES),
            "arms": [],
        }
        if panel_path.is_file() and baseline_path.is_file():
            panel = panel or read_json(panel_path)
            observed = read_json(baseline_path)
            expected_by_id = panel["baseline_outcomes"]
            actual_by_id = {
                str(row["scenario_id"]): str(row["outcome"])
                for row in observed["rows"]
            }
            gate_record.update(
                {
                    "expected": {"collision": 21, "follow": 49, "overtake": 50, "error": 0, "total": 120},
                    "observed": observed["summary"],
                    "paired_outcome_differences": [
                        {"scenario_id": scenario_id, "expected": expected_by_id[scenario_id], "observed": actual_by_id[scenario_id]}
                        for scenario_id in sorted(expected_by_id)
                        if expected_by_id[scenario_id] != actual_by_id[scenario_id]
                    ],
                }
            )
        write_json(OUTPUT / "SCREEN_RESULTS.json", {"schema_version": 1, "stage": "SCREEN", **gate_record})
        write_json(OUTPUT / "RETENTION_RESULTS.json", {"status": "NOT_RUN_UPSTREAM_INVALID"})
        write_json(OUTPUT / "FULL600_RESULTS.json", {"status": "NOT_RUN_UPSTREAM_INVALID"})
        write_json(
            OUTPUT / "INTEGRITY_FAILURE.json",
            {
                "status": "INVALID_EXPERIMENT",
                "error": failure,
                "gate_record": gate_record,
                "recorded_at": utc_now(),
            },
        )
        write_final(
            "INVALID_EXPERIMENT",
            h1_summary,
            panel,
            screen,
            retention,
            full,
            integrity_failure=failure,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
