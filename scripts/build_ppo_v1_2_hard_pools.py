#!/usr/bin/env python3
"""Generate, classify, validate and hash all fixed PPO V1.2 hard pools."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from experiments.ppo_v1_2.experiment_spec import BC_SHA256, PROJECT_ROOT, austin_asset_hashes, canonical_hash, file_sha256
from experiments.ppo_v1_2.hard_pool_builder import (
    EXPANDED_PANEL_ID,
    PASS1_SEEDS,
    PASS2_SEEDS,
    classify_stochastic_rows,
    collision_step_summary,
    deterministic_expanded_startpoints,
    expanded_scenarios,
    pool_manifest,
    union_pool,
    validate_candidates,
)
from experiments.ppo_v1_2.registry import build_manifest
from rl.end2race_gymnasium_env import End2RaceGymnasiumEnv, LatticePlannerOpponentController
from rl.ppo_privileged import oriented_rectangle_clearance
from rl.ppo_reward import PPOV1TransitionReward, ProgressProjector
from rl.ppo_scenarios import ScenarioSpec, training_scenarios
from rl.sb3_end2race_policy import DEFAULT_BC_CHECKPOINT, END2RACE_OBSERVATION_SIZE, End2RaceGRUPolicy, NOOP_SPEED_BOUND
from train_ppo_sb3 import evaluate_actor_pool
from utils import atomic_write_json


WORKER_ENV: End2RaceGymnasiumEnv | None = None
WORKER_POLICY: End2RaceGRUPolicy | None = None
SCENARIO_FIELDS = {field.name for field in fields(ScenarioSpec)}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scenario(row: dict[str, Any]) -> ScenarioSpec:
    return ScenarioSpec(**{name: row[name] for name in SCENARIO_FIELDS})


def _make_core(seed: int):
    import gym
    from f110_gym.envs.base_classes import Integrator

    return gym.make(
        "f110-v0",
        map=str(PROJECT_ROOT / "f1tenth_racetracks" / "Austin" / "Austin_map"),
        map_ext=".png",
        num_agents=2,
        timestep=0.01,
        integrator=Integrator.RK4,
        seed=seed,
    )


def _preflight_all(candidates: Sequence[ScenarioSpec], output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    core = _make_core(20260715)
    controller = LatticePlannerOpponentController()

    def check(scenario: ScenarioSpec) -> dict[str, Any]:
        spec = scenario.to_reset_spec()
        result = core.reset(poses=spec.poses.copy())
        raw = result[0] if isinstance(result, tuple) else result
        collisions = np.asarray(raw["collisions"], dtype=bool)
        pose_values = np.asarray([raw["poses_x"], raw["poses_y"], raw["poses_theta"]], dtype=np.float64)
        observation_values = [np.asarray(value) for value in raw.values() if isinstance(value, (list, tuple, np.ndarray))]
        first_pose = np.asarray([raw["poses_x"][0], raw["poses_y"][0], raw["poses_theta"][0]])
        second_pose = np.asarray([raw["poses_x"][1], raw["poses_y"][1], raw["poses_theta"][1]])
        controller.reset(spec, 2, 0)
        return {
            "reset": True,
            "poses_finite": bool(np.isfinite(pose_values).all()),
            "observation_finite": bool(all(np.isfinite(value).all() for value in observation_values)),
            "initial_collision_free": not bool(collisions.any()),
            "rectangles_disjoint": oriented_rectangle_clearance(first_pose, second_pose) > 0.0,
            "planner_constructed": len(controller.state_snapshot()["planners"]) == 1,
        }

    try:
        return validate_candidates(candidates, check)
    finally:
        core.close()


def _worker_init() -> None:
    global WORKER_ENV, WORKER_POLICY
    torch.set_num_threads(1)
    dummy = training_scenarios()[0]
    WORKER_ENV = End2RaceGymnasiumEnv(
        _make_core(20260715),
        sim_duration=8.0,
        reset_provider=lambda _rng: dummy.to_reset_spec(),
        ego_index=0,
        opponent_controller=LatticePlannerOpponentController(),
        transition_reward=PPOV1TransitionReward(ProgressProjector.from_csv()),
    )
    observation_space = spaces.Box(-np.inf, np.inf, shape=(END2RACE_OBSERVATION_SIZE,), dtype=np.float32)
    action_space = spaces.Box(
        np.asarray([-0.52, -NOOP_SPEED_BOUND], dtype=np.float32),
        np.asarray([0.52, NOOP_SPEED_BOUND], dtype=np.float32),
        dtype=np.float32,
    )
    WORKER_POLICY = End2RaceGRUPolicy(
        observation_space,
        action_space,
        lambda _: 1.0,
        optimizer_profile="ppo_v1",
        critic_profile="C0_RAW_SINGLE_FRAME",
        steering_latent_std=0.05,
        speed_physical_std=0.15,
    ).cpu()
    WORKER_POLICY.set_training_mode(False)


def _stochastic_worker(task: tuple[dict[str, Any], int]) -> dict[str, Any]:
    row, seed = task
    scenario = _scenario(row)
    try:
        if WORKER_ENV is None or WORKER_POLICY is None:
            _worker_init()
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        observation, _info = WORKER_ENV.reset(seed=int(seed), options={"reset_spec": scenario.to_reset_spec()})
        hidden = torch.zeros(1, 1, WORKER_POLICY.actor_hidden_size)
        states = RNNStates((hidden, hidden.clone()), (hidden.clone(), hidden.clone()))
        episode_starts = torch.ones(1)
        collision_step = None
        final_info: dict[str, Any] = {}
        for step in range(1, 802):
            with torch.no_grad():
                action, _value, _log_prob, states = WORKER_POLICY.forward(
                    torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0),
                    states,
                    episode_starts,
                    deterministic=False,
                )
            observation, _reward, terminated, truncated, final_info = WORKER_ENV.step(action[0].cpu().numpy())
            episode_starts = torch.zeros(1)
            if bool(final_info.get("ego_collision")):
                collision_step = step
            if terminated or truncated:
                break
        if bool(final_info.get("ego_collision")):
            outcome = "ego_collision"
        else:
            outcome = "overtake" if float(final_info.get("relative_position_m", 0.0)) > 0.0 else "follow"
        return {
            "scenario_id": scenario.scenario_id,
            "seed": int(seed),
            "outcome": outcome,
            "collision_step": collision_step,
            "opponent_only_collision": bool(final_info.get("opponent_collision_latched", False) and not final_info.get("ego_collision", False)),
            "error": None,
        }
    except Exception as error:
        return {
            "scenario_id": scenario.scenario_id,
            "seed": int(seed),
            "outcome": "error",
            "collision_step": None,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _read_jsonl(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {(str(row["scenario_id"]), int(row["seed"])): row for row in rows}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()


def _run_seed_tasks(rows: Sequence[dict[str, Any]], seeds: Sequence[int], path: Path, workers: int, status_path: Path, phase: str) -> dict[tuple[str, int], dict[str, Any]]:
    completed = _read_jsonl(path)
    tasks = [(row, seed) for row in rows for seed in seeds if (str(row["scenario_id"]), int(seed)) not in completed]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_worker_init) as executor:
        for index, result in enumerate(executor.map(_stochastic_worker, tasks, chunksize=1), start=1):
            _append_jsonl(path, result)
            completed[(str(result["scenario_id"]), int(result["seed"]))] = result
            if index % 100 == 0 or index == len(tasks):
                atomic_write_json(status_path, {"phase": phase, "completed": len(completed), "expected": len(rows) * len(seeds), "last_update": _utc_now()})
    if len(completed) != len(rows) * len(seeds):
        raise RuntimeError(f"{phase} classification incomplete: {len(completed)} != {len(rows) * len(seeds)}")
    return completed


def _write_pool(path: Path, document: dict[str, Any]) -> None:
    atomic_write_json(path, document)


def _report(root: Path, summary: dict[str, Any], pools: Sequence[dict[str, Any]]) -> None:
    lines = ["# PPO V1.2 HARD POOL REPORT", "", "## Expanded preflight", "", "| candidates | valid | invalid | complete |", "|---:|---:|---:|:---:|", f"| {summary['candidate_count']} | {summary['valid_count']} | {summary['invalid_count']} | {summary['complete']} |", "", "## Pool counts and hashes", "", "| pool_id | count | manifest_hash |", "|---|---:|---|"]
    for pool in pools:
        lines.append(f"| {pool['pool_id']} | {pool['count']} | `{pool['manifest_hash']}` |")
    for pool in pools:
        lines.extend(["", f"## {pool['pool_id']} distributions"])
        for field, counts in pool["distributions"].items():
            lines.extend(["", f"### {field}", "", "| value | count |", "|---|---:|"])
            for value, count in counts.items():
                lines.append(f"| {value} | {count} |")
    (root / "HARD_POOL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _head() -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()


def run(root: Path, workers: int, phase: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    candidates = expanded_scenarios(deterministic_expanded_startpoints())
    candidate_rows = [asdict(row) for row in candidates]
    atomic_write_json(root / "expanded_candidates.json", candidate_rows)
    if phase in {"preflight", "all"} or not (root / "expanded_validation_summary.json").is_file():
        atomic_write_json(status_path, {"phase": "preflight", "completed": 0, "expected": 10_800, "last_update": _utc_now()})
        valid, invalid, validation_summary = _preflight_all(candidates, root)
        atomic_write_json(root / "expanded_valid.json", valid)
        atomic_write_json(root / "expanded_invalid.json", invalid)
        atomic_write_json(root / "expanded_validation_summary.json", validation_summary)
    else:
        valid = json.loads((root / "expanded_valid.json").read_text(encoding="utf-8"))
        invalid = json.loads((root / "expanded_invalid.json").read_text(encoding="utf-8"))
        validation_summary = json.loads((root / "expanded_validation_summary.json").read_text(encoding="utf-8"))
    if phase == "preflight":
        atomic_write_json(status_path, {"phase": "preflight", "status": "PASS", **validation_summary, "last_update": _utc_now()})
        return

    valid_scenarios = [_scenario(row) for row in valid]
    atomic_write_json(status_path, {"phase": "H1", "completed": 0, "expected": len(valid_scenarios), "last_update": _utc_now()})
    h1_rows, h1_summary = evaluate_actor_pool(DEFAULT_BC_CHECKPOINT, valid_scenarios, workers=workers)
    atomic_write_json(root / "H1_rows.json", h1_rows)
    atomic_write_json(root / "H1_summary.json", h1_summary)
    h1_ids = sorted(str(row["scenario_id"]) for row in h1_rows if row.get("outcome") == "ego_collision")
    atomic_write_json(root / "H1_expanded_deterministic.json", h1_ids)

    pass1_path = root / "H2_pass1_rows.jsonl"
    pass1 = _run_seed_tasks(valid, PASS1_SEEDS, pass1_path, workers, status_path, "H2_PASS1")
    candidate_ids = sorted({scenario_id for (scenario_id, _seed), row in pass1.items() if row["outcome"] == "ego_collision"})
    atomic_write_json(root / "H2_pass2_candidates.json", candidate_ids)
    valid_by_id = {str(row["scenario_id"]): row for row in valid}
    pass2_rows = [valid_by_id[scenario_id] for scenario_id in candidate_ids]
    pass2_path = root / "H2_pass2_rows.jsonl"
    pass2 = _run_seed_tasks(pass2_rows, PASS2_SEEDS, pass2_path, workers, status_path, "H2_PASS2")
    stochastic_rows = []
    for scenario_id in candidate_ids:
        seed_outcomes = {
            str(seed): (pass1 if seed in PASS1_SEEDS else pass2)[(scenario_id, seed)]
            for seed in (*PASS1_SEEDS, *PASS2_SEEDS)
        }
        stochastic_rows.append({**valid_by_id[scenario_id], "seed_outcomes": seed_outcomes, **collision_step_summary(seed_outcomes)})
    atomic_write_json(root / "H2_rows.json", stochastic_rows)
    h2 = classify_stochastic_rows(stochastic_rows)

    current_rows = json.loads((PROJECT_ROOT / "runs/ppo_v1/v1_1_pilot_20_updates/train_bc_outcomes.json").read_text(encoding="utf-8"))
    h0_ids = sorted(str(row["scenario_id"]) for row in current_rows if row.get("outcome") == "ego_collision")
    training_by_id = {row.scenario_id: row for row in training_scenarios()}
    expanded_by_id = {row.scenario_id: row for row in valid_scenarios}
    all_by_id = {**training_by_id, **expanded_by_id}
    pool_ids = {
        "H0_CURRENT_DET": h0_ids,
        "H1_EXPANDED_DET": h1_ids,
        "H2_STOCH_CORE": h2["H2_STOCH_CORE"],
        "H2_STOCH_BOUNDARY": h2["H2_STOCH_BOUNDARY"],
        "H2_STOCH_ALL": h2["H2_STOCH_ALL"],
        "H3_UNION_CORE": union_pool(h1_ids, h2["H2_STOCH_CORE"]),
        "H3_UNION_ALL": union_pool(h1_ids, h2["H2_STOCH_ALL"]),
    }
    pool_root = root / "pools"
    pool_root.mkdir(parents=True, exist_ok=True)
    pools = []
    for pool_id, ids in pool_ids.items():
        document = pool_manifest(pool_id, ids, all_by_id)
        _write_pool(pool_root / f"{pool_id}.json", document)
        pools.append(document)
    _report(root, validation_summary, pools)

    formal_pool_ids = ("H0_CURRENT_DET", "H1_EXPANDED_DET", "H2_STOCH_CORE", "H2_STOCH_ALL", "H3_UNION_CORE", "H3_UNION_ALL")
    formal_hashes = {pool_id: next(pool["manifest_hash"] for pool in pools if pool["pool_id"] == pool_id) for pool_id in formal_pool_ids}
    manifest = build_manifest(experiment_head=_head(), hard_pool_hashes=formal_hashes)
    manifest["austin_asset_hashes"] = austin_asset_hashes()
    for arm in manifest["arms"]:
        pool = next(document for document in pools if document["pool_id"] == arm["resolved_config"]["hard_pool_id"])
        if pool["count"] == 0:
            arm["status"] = "SKIPPED_EMPTY_POOL"
    manifest["manifest_hash"] = canonical_hash({key: value for key, value in manifest.items() if key not in {"generated_at", "manifest_hash"}})
    atomic_write_json(root.parent / "sweep_manifest.runtime.json", manifest)
    result = {
        "status": "PASS",
        "candidate_count": 10_800,
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "h1_error_count": int(h1_summary["error"]),
        "stochastic_error_count": sum(row["outcome"] == "error" for row in (*pass1.values(), *pass2.values())),
        "pool_counts": {pool["pool_id"]: pool["count"] for pool in pools},
        "pool_hashes": {pool["pool_id"]: pool["manifest_hash"] for pool in pools},
        "completed_at": _utc_now(),
    }
    atomic_write_json(root / "HARD_POOL_COMPLETION.json", result)
    atomic_write_json(status_path, {"phase": "complete", **result})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/ppo_v1_2/hard_pools"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--phase", choices=("preflight", "classification", "all"), default="all")
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    run(args.output.resolve(), args.workers, args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
