#!/usr/bin/env python3
"""Execute the preregistered H1 screen, retention, and fresh-seed confirmation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "runs" / "ppo" / "h1_h2_conditional_v1"
BASELINE_PATH = ROOT / "ppo_experiments" / "quick_pool_3s_v2" / "FULL600_BC_RESULTS.json"
EVALUATOR = ROOT / "ppo_experiments" / "quick_pool_3s" / "evaluate_pool.py"
SUPPORT_VALIDATION = OUTPUT / "SUPPORT_VALIDATION.json"
SCREEN_SEED = 20260719
FRESH_SEEDS = (20260720, 20260721)
ARMS = ("N1-H1F-p50", "N1-H1F-p25", "N1-H1E-p50", "N1-H1E-p25")
FULL_ARMS = ("N1-H1F-p50", "N1-H1F-p25")
EARLY_ARMS = ("N1-H1E-p50", "N1-H1E-p25")
FROZEN_PATHS = (
    "ppo/policy.py",
    "ppo/environment.py",
    "ppo/reward.py",
    "ppo/scenarios.py",
    "eval_multiagent.py",
    "evaluate.sh",
    "utils.py",
    "pretrained/end2race.pth",
)

sys.path.insert(0, str(OUTPUT))
sys.path.insert(0, str(ROOT))
from stage0_posthoc import paired_metrics, read_json, sha256_file, validate_evaluation, write_json  # noqa: E402


PROCESSES: dict[tuple[str, int], tuple[subprocess.Popen[str], Any]] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def commands_document() -> dict[str, Any]:
    path = OUTPUT / "COMMANDS.json"
    if not path.exists():
        return {"schema_version": 1, "commands": []}
    document = read_json(path)
    document.setdefault("schema_version", 1)
    document.setdefault("commands", [])
    return document


def record_command(command: list[str], purpose: str) -> None:
    document = commands_document()
    document["commands"].append(
        {"purpose": purpose, "argv": command, "started_at_utc": utc_now()}
    )
    write_json(OUTPUT / "COMMANDS.json", document)
    print("COMMAND " + " ".join(command), flush=True)


def run_dir(arm: str, seed: int) -> Path:
    return RUN_ROOT / f"{arm}_seed{seed}"


def status(arm: str, seed: int) -> dict[str, Any] | None:
    path = run_dir(arm, seed) / "run_status.json"
    try:
        return read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def launch(arm: str, seed: int, *, pause: bool) -> None:
    destination = run_dir(arm, seed)
    if destination.exists():
        raise RuntimeError(f"Run destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT / "training_logs" / f"{arm}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-u",
        str(ROOT / "train_ppo.py"),
        "--config",
        arm,
        "--seed",
        str(seed),
        "--output_dir",
        str(destination),
    ]
    if pause:
        command.append("--screen-pause")
    record_command(
        command,
        f"Train {arm} seed {seed} through U1 and pause" if pause else f"Train {arm} seed {seed} through U2",
    )
    environment = os.environ.copy()
    environment.setdefault("NUMBA_CACHE_DIR", "/tmp/end2race_h1_h2_conditional_numba")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    PROCESSES[(arm, seed)] = (process, handle)


def wait_status(arm: str, seed: int, expected: str, timeout_s: float = 7200.0) -> None:
    process, _handle = PROCESSES[(arm, seed)]
    started = time.monotonic()
    last_report = -60.0
    while True:
        observed = status(arm, seed)
        if observed is not None and observed.get("status") == expected:
            print(f"TRAIN_STATUS arm={arm} seed={seed} status={expected}", flush=True)
            return
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Trainer {arm} seed {seed} exited {return_code} before {expected}; status={observed}"
            )
        elapsed = time.monotonic() - started
        if elapsed > timeout_s:
            raise RuntimeError(f"Trainer {arm} seed {seed} timed out waiting for {expected}")
        if elapsed - last_report >= 30.0:
            print(
                f"TRAIN_WAIT arm={arm} seed={seed} target={expected} elapsed={elapsed:.0f}s status={observed}",
                flush=True,
            )
            last_report = elapsed
        time.sleep(2.0)


def wait_exit(arm: str, seed: int) -> None:
    process, handle = PROCESSES[(arm, seed)]
    return_code = process.wait(timeout=120)
    handle.close()
    if return_code != 0:
        raise RuntimeError(f"Trainer {arm} seed {seed} exited {return_code}")


def send_decisions(decisions: dict[str, str]) -> None:
    for arm, decision in decisions.items():
        process, _handle = PROCESSES[(arm, SCREEN_SEED)]
        if process.stdin is None:
            raise RuntimeError(f"Trainer stdin is unavailable: {arm}")
        process.stdin.write(decision + "\n")
        process.stdin.flush()
    for arm, decision in decisions.items():
        expected = "STOPPED_SCREEN" if decision == "stop" else "COMPLETED"
        wait_status(arm, SCREEN_SEED, expected)
    for arm in decisions:
        wait_exit(arm, SCREEN_SEED)


def checkpoint(arm: str, seed: int, update: int) -> tuple[Path, dict[str, Any]]:
    manifest_path = run_dir(arm, seed) / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    matches = [record for record in manifest["checkpoints"] if int(record["update"]) == update]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {arm} seed {seed} U{update} checkpoint: {matches}")
    record = matches[0]
    path = manifest_path.parent / record["path"]
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch: {path}")
    return path, record


def numeric_integrity(arm: str, seed: int, through_update: int) -> dict[str, Any]:
    directory = run_dir(arm, seed)
    rows = [
        json.loads(line)
        for line in (directory / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if int(row["update"]) <= through_update]
    if [int(row["update"]) for row in rows] != list(range(1, through_update + 1)):
        raise RuntimeError(f"Incomplete training metrics for {arm} seed {seed} through U{through_update}")
    violations = []
    for row in rows:
        update = int(row["update"])
        if float(row["approx_kl"]) > 0.05:
            violations.append(f"U{update} approx_kl>0.05")
        if float(row["clip_fraction"]) > 0.50:
            violations.append(f"U{update} clip_fraction>0.50")
        actor = row["actor_delta_from_bc"]
        if float(actor["frozen_actor"]["max_abs_delta_from_bc"]) != 0.0:
            violations.append(f"U{update} frozen_actor_delta!=0")
        if float(actor["log_std_max_abs_delta_from_initial"]) != 0.0:
            violations.append(f"U{update} log_std_delta!=0")
        if int(row["actual_optimizer_steps"]) != int(row["planned_optimizer_steps"]):
            violations.append(f"U{update} optimizer_step_mismatch")
    if violations:
        raise RuntimeError(f"Training integrity failure for {arm} seed {seed}: {violations}")
    sampler = read_json(directory / "sampler_summary.json")
    resolved = read_json(directory / "resolved_config.json")
    return {
        "status": "PASS",
        "through_update": through_update,
        "metrics_path": str((directory / "training_metrics.jsonl").relative_to(ROOT)),
        "metrics_sha256": sha256_file(directory / "training_metrics.jsonl"),
        "resolved_config_path": str((directory / "resolved_config.json").relative_to(ROOT)),
        "resolved_config_sha256": sha256_file(directory / "resolved_config.json"),
        "sampler_summary": sampler,
        "updates": rows,
        "resolved_config": resolved,
    }


def evaluation_result(arm: str, seed: int, update: int, baseline: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path, checkpoint_record = checkpoint(arm, seed, update)
    raw_path = OUTPUT / f"h1_u{update}_eval_{arm}_seed{seed}.json"
    if raw_path.exists():
        raise RuntimeError(f"Evaluation output already exists: {raw_path}")
    command = [
        sys.executable,
        "-u",
        str(EVALUATOR),
        "--model-path",
        str(checkpoint_path),
        "--output",
        str(raw_path),
        "--workers",
        "8",
        "--sim-duration",
        "8.0",
    ]
    record_command(command, f"Evaluate {arm} seed {seed} U{update} on current CPU full-600")
    subprocess.run(command, cwd=ROOT, check=True)
    raw = read_json(raw_path)
    validate_evaluation(raw)
    expected_contract = {
        "device": "cpu",
        "persistent_spawn_workers": 8,
        "torch_num_threads_per_worker": 1,
        "collision_scope": "ego",
    }
    if raw["evaluation_contract"] != expected_contract:
        raise RuntimeError(f"Evaluation contract drift: {raw['evaluation_contract']}")
    paired = paired_metrics(baseline, raw)
    evaluation = {**raw["summary"], **paired}
    legal = bool(
        evaluation["error"] == 0
        and evaluation["overtake"] >= 328
        and evaluation["fixed_collision"] >= 4
    )
    return {
        "arm_id": arm,
        "seed": seed,
        "update": update,
        "transitions": update * 25600,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_record["sha256"],
        "raw_evaluation": str(raw_path.relative_to(ROOT)),
        "raw_evaluation_sha256": sha256_file(raw_path),
        "evaluation_contract": raw["evaluation_contract"],
        "coverage": {"rows": len(raw["rows"]), "unique_scenario_ids": len({row["scenario_id"] for row in raw["rows"]})},
        "evaluation": evaluation,
        "screen_legal": legal,
        "training": numeric_integrity(arm, seed, update),
    }


def selection_key(row: dict[str, Any], *, include_arm: bool = True) -> tuple[Any, ...]:
    evaluation = row["evaluation"]
    values: tuple[Any, ...] = (
        evaluation["delta_collision"],
        evaluation["new_collision"],
        -evaluation["fixed_collision"],
        evaluation["collision"],
        -evaluation["overtake"],
    )
    return values + ((row["arm_id"],) if include_arm else ())


def ratio_result(rows: dict[str, dict[str, Any]], pool: str, arm25: str, arm50: str) -> dict[str, Any]:
    row25, row50 = rows[arm25], rows[arm50]
    if not row25["screen_legal"] or not row50["screen_legal"]:
        status_value = "RATIO_INCONCLUSIVE"
    else:
        e25, e50 = row25["evaluation"], row50["evaluation"]
        first = (
            e25["new_collision"] <= e50["new_collision"] - 2
            and e25["fixed_collision"] >= e50["fixed_collision"] - 2
            and e25["overtake"] >= e50["overtake"] - 2
        )
        second = (
            e25["delta_collision"] < e50["delta_collision"]
            and e25["new_collision"] < e50["new_collision"]
            and e25["fixed_collision"] >= 4
        )
        status_value = "RATIO_25_SUPPORTED" if first or second else "RATIO_25_NOT_SUPPORTED"
    return {"pool": pool, "arm_25": arm25, "arm_50": arm50, "status": status_value}


def screen_selection(screen: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {row["arm_id"]: row for row in screen}
    legal_full = [by_arm[arm] for arm in FULL_ARMS if by_arm[arm]["screen_legal"]]
    legal_early = [by_arm[arm] for arm in EARLY_ARMS if by_arm[arm]["screen_legal"]]
    if not legal_full or not legal_early:
        return {
            "status": "STOP_H1_SCREEN_MISSING_LEGAL_POOL_COMPARATOR",
            "full_legal_arms": [row["arm_id"] for row in legal_full],
            "early_legal_arms": [row["arm_id"] for row in legal_early],
        }
    full_winner = min(legal_full, key=selection_key)["arm_id"]
    early_winner = min(legal_early, key=selection_key)["arm_id"]
    ratios = [
        ratio_result(by_arm, "H1_FULL", "N1-H1F-p25", "N1-H1F-p50"),
        ratio_result(by_arm, "H1_EARLY", "N1-H1E-p25", "N1-H1E-p50"),
    ]
    return {
        "status": "PASS",
        "selection_tuple": ["delta_collision", "new_collision", "-fixed_collision", "collision", "-overtake", "arm_id"],
        "H1_FULL_SCREEN_WINNER": full_winner,
        "H1_EARLY_SCREEN_WINNER": early_winner,
        "ratio_results": ratios,
    }


def retention_result(u1: dict[str, Any], u2: dict[str, Any]) -> dict[str, Any]:
    first, second = u1["evaluation"], u2["evaluation"]
    supported = bool(
        second["delta_collision"] <= first["delta_collision"] + 1
        and second["new_collision"] <= first["new_collision"] + 2
        and second["overtake"] >= 328
        and second["error"] == 0
    )
    return {
        **u2,
        "u1_evaluation": first,
        "short_retention_supported": supported,
    }


def preflight() -> dict[str, Any]:
    if subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Formal H1 training requires a clean worktree")
    if RUN_ROOT.exists():
        raise RuntimeError(f"Formal run root already exists: {RUN_ROOT}")
    support = read_json(SUPPORT_VALIDATION)
    if support.get("status") != "PASS":
        raise RuntimeError("Support validation did not pass")
    recorded = support["post_implementation_source_hashes"]
    for relative in FROZEN_PATHS:
        observed = sha256_file(ROOT / relative)
        if observed != recorded[relative]:
            raise RuntimeError(f"Frozen preflight hash drift for {relative}: {observed}")
    baseline = read_json(BASELINE_PATH)
    validate_evaluation(baseline)
    if baseline["summary"] != {"collision": 22, "error": 0, "follow": 233, "overtake": 345, "total": 600}:
        raise RuntimeError(f"Binding BC summary drift: {baseline['summary']}")
    write_json(
        OUTPUT / "FORMAL_PREFLIGHT.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "recorded_at_utc": utc_now(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "worktree_clean_before_record": True,
            "support_validation": str(SUPPORT_VALIDATION.relative_to(ROOT)),
            "support_validation_sha256": sha256_file(SUPPORT_VALIDATION),
            "frozen_source_hashes": {relative: recorded[relative] for relative in FROZEN_PATHS},
            "baseline_path": str(BASELINE_PATH.relative_to(ROOT)),
            "baseline_sha256": sha256_file(BASELINE_PATH),
            "baseline_summary": baseline["summary"],
        },
    )
    return baseline


def terminate_children() -> None:
    for process, handle in PROCESSES.values():
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        if not handle.closed:
            handle.close()


def main() -> int:
    baseline = preflight()
    try:
        for cohort in (FULL_ARMS, EARLY_ARMS):
            for arm in cohort:
                launch(arm, SCREEN_SEED, pause=True)
            for arm in cohort:
                wait_status(arm, SCREEN_SEED, "PAUSED_SCREEN")

        screen = [evaluation_result(arm, SCREEN_SEED, 1, baseline) for arm in ARMS]
        write_json(
            OUTPUT / "H1_SCREEN_RESULTS.json",
            {"schema_version": 1, "stage": "H1_SCREEN", "baseline": baseline["summary"], "arms": screen},
        )
        selection = screen_selection(screen)
        write_json(OUTPUT / "H1_SCREEN_SELECTION.json", {"schema_version": 1, **selection})
        if selection["status"] != "PASS":
            send_decisions({arm: "stop" for arm in ARMS})
            return 0

        full_winner = selection["H1_FULL_SCREEN_WINNER"]
        early_winner = selection["H1_EARLY_SCREEN_WINNER"]
        winners = (full_winner, early_winner)
        send_decisions({arm: ("continue" if arm in winners else "stop") for arm in ARMS})

        by_arm_u1 = {row["arm_id"]: row for row in screen}
        retention_rows = [
            retention_result(
                by_arm_u1[arm],
                evaluation_result(arm, SCREEN_SEED, 2, baseline),
            )
            for arm in winners
        ]
        full_row = next(row for row in retention_rows if row["arm_id"] == full_winner)
        early_row = next(row for row in retention_rows if row["arm_id"] == early_winner)
        full_eval, early_eval = full_row["evaluation"], early_row["evaluation"]
        if (
            full_eval["delta_collision"] < early_eval["delta_collision"]
            or (
                full_eval["delta_collision"] == early_eval["delta_collision"]
                and full_eval["new_collision"] < early_eval["new_collision"]
            )
        ):
            pool_preference = "H1_FULL_PREFERRED"
        elif (
            early_eval["delta_collision"] < full_eval["delta_collision"]
            or (
                early_eval["delta_collision"] == full_eval["delta_collision"]
                and early_eval["new_collision"] < full_eval["new_collision"]
            )
        ):
            pool_preference = "H1_EARLY_PREFERRED"
        else:
            pool_preference = "H1_POOL_INCONCLUSIVE"
        write_json(
            OUTPUT / "H1_RETENTION_RESULTS.json",
            {
                "schema_version": 1,
                "stage": "H1_RETENTION",
                "pool_preference": pool_preference,
                "arms": retention_rows,
            },
        )

        ranked = sorted(retention_rows, key=lambda row: selection_key(row, include_arm=False))
        if len(ranked) > 1 and selection_key(ranked[0], include_arm=False) == selection_key(ranked[1], include_arm=False):
            write_json(
                OUTPUT / "H1_SELECTED_CONFIG.json",
                {"schema_version": 1, "status": "STOP_H1_OVERALL_TIE", "tied_arms": [row["arm_id"] for row in ranked]},
            )
            return 0
        selected = ranked[0]["arm_id"]
        selected_document = {
            "schema_version": 1,
            "status": "SELECTED",
            "H1_SELECTED_CONFIG": selected,
            "selection_tuple": selection_key(ranked[0], include_arm=False),
            "pool_preference": pool_preference,
            "screen_seed": SCREEN_SEED,
            "fresh_seeds": list(FRESH_SEEDS),
        }
        write_json(OUTPUT / "H1_SELECTED_CONFIG.json", selected_document)

        for seed in FRESH_SEEDS:
            launch(selected, seed, pause=False)
        for seed in FRESH_SEEDS:
            wait_status(selected, seed, "COMPLETED")
        for seed in FRESH_SEEDS:
            wait_exit(selected, seed)
        fresh = [evaluation_result(selected, seed, 2, baseline) for seed in FRESH_SEEDS]
        screen_seed_row = next(row for row in retention_rows if row["arm_id"] == selected)
        all_seeds = [screen_seed_row, *fresh]
        product_passes = [
            bool(
                row["evaluation"]["collision"] <= 21
                and row["evaluation"]["fixed_collision"] > row["evaluation"]["new_collision"]
                and row["evaluation"]["overtake"] >= 328
                and row["evaluation"]["error"] == 0
            )
            for row in all_seeds
        ]
        median_collision = statistics.median(row["evaluation"]["collision"] for row in all_seeds)
        median_delta = statistics.median(row["evaluation"]["delta_collision"] for row in all_seeds)
        repeats = bool(
            sum(product_passes) >= 2
            and median_collision <= 21
            and median_delta < 0
            and all(row["evaluation"]["error"] == 0 for row in all_seeds)
        )
        selected_u1 = by_arm_u1[selected]["evaluation"]
        selected_u2 = screen_seed_row["evaluation"]
        continues = bool(
            selected_u2["collision"] <= selected_u1["collision"] + 1
            and selected_u2["delta_collision"] <= selected_u1["delta_collision"] + 1
            and selected_u2["new_collision"] <= selected_u1["new_collision"] + 2
        )
        write_json(
            OUTPUT / "H1_REPEATABILITY_RESULTS.json",
            {
                "schema_version": 1,
                "stage": "H1_REPEATABILITY",
                "selected_config": selected,
                "seeds": all_seeds,
                "product_pass_by_seed": {str(row["seed"]): passed for row, passed in zip(all_seeds, product_passes)},
                "median_collision": median_collision,
                "median_delta_collision": median_delta,
                "H1_PRODUCT_DIRECTION_REPEATS": repeats,
                "H1_SHORT_TRAINING_CONTINUES": continues,
            },
        )
        return 0
    except Exception as error:
        write_json(
            OUTPUT / "H1_FAILURE.json",
            {"schema_version": 1, "status": "INVALID", "error_type": type(error).__name__, "error": str(error), "recorded_at_utc": utc_now()},
        )
        raise
    finally:
        terminate_children()


if __name__ == "__main__":
    raise SystemExit(main())
