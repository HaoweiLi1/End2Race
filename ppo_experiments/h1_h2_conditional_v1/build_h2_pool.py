#!/usr/bin/env python3
"""Build the preregistered matched I7/I8 conditional-exploration pool."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
SOURCE = ROOT / "ppo" / "hard_pools" / "h2_stoch_core.json"
CHECKPOINT = ROOT / "pretrained" / "end2race.pth"
SUPPORT = OUTPUT / "SUPPORT_VALIDATION.json"
PASS1_SEEDS = (20260722, 20260723, 20260724, 20260725)
PASS2_SEEDS = (20260726, 20260727, 20260728, 20260729)
SEED_PAIRS = ((20260722, 20260723), (20260724, 20260725), (20260726, 20260727), (20260728, 20260729))
WORKERS = 8

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUTPUT))

from model import End2Race  # noqa: E402
from ppo.environment import (  # noqa: E402
    End2RaceGymnasiumEnv,
    LatticePlannerOpponentController,
    oriented_rectangle_clearance,
)
from ppo.policy import EVALUATOR_STEER_BOUND  # noqa: E402
from ppo.reward import PPOTransitionReward, ProgressProjector  # noqa: E402
from ppo.scenarios import ScenarioSpec, scenario_from_dict  # noqa: E402
from stage0_posthoc import read_json, sha256_file, write_json  # noqa: E402
from utils import load_raceline_waypoints  # noqa: E402


MODEL: End2Race | None = None
DEVICE = torch.device("cpu")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(document: dict[str, Any]) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def record_command() -> None:
    path = OUTPUT / "COMMANDS.json"
    document = read_json(path)
    document.setdefault("commands", []).append(
        {
            "purpose": "Build and classify the matched H2 interval-7/interval-8 contrast pool",
            "argv": [sys.executable, "-u", str(Path(__file__).resolve())],
            "started_at_utc": utc_now(),
        }
    )
    write_json(path, document)


def worker_init(checkpoint: str) -> None:
    global MODEL
    torch.set_num_threads(1)
    model = End2Race(mask_prob=0.0, hidden_scale=4).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True), strict=True)
    model.eval()
    MODEL = model


def _scenario_provider(scenario: ScenarioSpec):
    def provider(_rng: np.random.Generator):
        spec = scenario.to_reset_spec(sampler_branch="h2_classification")
        spec.scenario["hard_pool_id"] = "H2_CONDITIONAL_CLASSIFICATION"
        spec.scenario["hard_sampling_mode"] = "fixed_trial"
        spec.scenario["env_role"] = "hard"
        return spec

    return provider


def _poses(raw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(raw["poses_x"], dtype=np.float64)
    y = np.asarray(raw["poses_y"], dtype=np.float64)
    heading = np.asarray(raw["poses_theta"], dtype=np.float64)
    return (
        np.asarray([x[0], y[0], heading[0]], dtype=np.float64),
        np.asarray([x[1], y[1], heading[1]], dtype=np.float64),
    )


def evaluate_trial(task: dict[str, Any]) -> dict[str, Any]:
    scenario_id = str(task["scenario"]["scenario_id"])
    try:
        if MODEL is None:
            raise RuntimeError("H2 classification worker model is not initialized")
        import gym
        from f110_gym.envs.base_classes import Integrator

        scenario = scenario_from_dict(task["scenario"])
        core = gym.make(
            "f110-v0",
            map=str(ROOT / "f1tenth_racetracks" / scenario.map_name / f"{scenario.map_name}_map"),
            map_ext=".png",
            num_agents=2,
            timestep=scenario.timestep,
            integrator=Integrator.RK4,
            seed=int(task["seed"]),
        )
        environment = End2RaceGymnasiumEnv(
            core,
            sim_duration=float(task["horizon_s"]),
            reset_provider=_scenario_provider(scenario),
            ego_index=0,
            opponent_controller=LatticePlannerOpponentController(),
            transition_reward=PPOTransitionReward(ProgressProjector.from_csv()),
            privileged_critic=False,
        )
        try:
            environment.set_policy_update_index(1)
            observation, _reset_info = environment.reset(seed=int(task["seed"]))
            raw = environment._raw_observation
            collisions = np.asarray(raw["collisions"], dtype=bool).reshape(-1)
            ego_pose, opponent_pose = _poses(raw)
            center_distance = float(np.linalg.norm(ego_pose[:2] - opponent_pose[:2]))
            initial_clearance = oriented_rectangle_clearance(ego_pose, opponent_pose)
            observation_finite = bool(np.isfinite(observation).all())
            preflight_valid = bool(
                not collisions[0]
                and not collisions[1]
                and initial_clearance > 0.0
                and observation_finite
            )
            base = {
                "base_id": task["base_id"],
                "variant": task["variant"],
                "scenario_id": scenario_id,
                "seed": int(task["seed"]),
                "stochastic": bool(task["stochastic"]),
                "horizon_s": float(task["horizon_s"]),
                "initial_ego_collision": bool(collisions[0]),
                "initial_opponent_collision": bool(collisions[1]),
                "initial_center_distance_m": center_distance,
                "initial_oriented_surface_clearance_m": initial_clearance,
                "observation_finite": observation_finite,
                "preflight_valid": preflight_valid,
            }
            if not preflight_valid:
                return {
                    **base,
                    "status": "PREFLIGHT_INVALID",
                    "action_finite": None,
                    "outcome": None,
                    "ego_collision_time_s": None,
                }

            hidden = torch.zeros((1, 1, MODEL.gru.hidden_size), device=DEVICE)
            rng = np.random.default_rng(int(task["seed"]))
            desired_speeds: list[float] = []
            absolute_steering: list[float] = []
            minimum_clearance = initial_clearance
            action_finite = True
            terminated = truncated = False
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                lidar = torch.as_tensor(observation[:360], dtype=torch.float32, device=DEVICE).view(1, 1, -1)
                speed = torch.as_tensor(observation[360:], dtype=torch.float32, device=DEVICE).view(1, 1, 1)
                with torch.no_grad():
                    action_sequence, hidden = MODEL(lidar, speed, hidden)
                mean = action_sequence[0, -1].detach().cpu().numpy().astype(np.float64)
                if task["stochastic"]:
                    normalized = float(np.clip(mean[0] / EVALUATOR_STEER_BOUND, -1.0 + 1e-6, 1.0 - 1e-6))
                    latent_mean = float(np.arctanh(normalized))
                    steering = EVALUATOR_STEER_BOUND * math.tanh(latent_mean + float(rng.normal(0.0, 0.03)))
                    desired_speed = float(mean[1] + rng.normal(0.0, 0.15))
                else:
                    steering = float(np.clip(mean[0], -EVALUATOR_STEER_BOUND, EVALUATOR_STEER_BOUND))
                    desired_speed = float(mean[1])
                action = np.asarray([steering, desired_speed], dtype=np.float32)
                action_finite = action_finite and bool(np.isfinite(action).all())
                desired_speeds.append(desired_speed)
                absolute_steering.append(abs(steering))
                observation, _reward, terminated, truncated, info = environment.step(action)
                observation_finite = observation_finite and bool(np.isfinite(observation).all())
                ego_pose, opponent_pose = _poses(environment._raw_observation)
                minimum_clearance = min(
                    minimum_clearance,
                    oriented_rectangle_clearance(ego_pose, opponent_pose),
                )
            return {
                **base,
                "status": "COMPLETE",
                "preflight_valid": bool(preflight_valid and observation_finite and action_finite),
                "observation_finite": observation_finite,
                "action_finite": action_finite,
                "outcome": str(info["episode_outcome"]),
                "ego_collision_time_s": float(info["elapsed_time"]) if bool(info["ego_collision"]) else None,
                "simulation_time_s": float(info["elapsed_time"]),
                "mean_desired_speed": float(np.mean(desired_speeds)),
                "mean_abs_steering": float(np.mean(absolute_steering)),
                "final_relative_progress_m": float(info["relative_position_m"]),
                "minimum_oriented_clearance_m": float(minimum_clearance),
            }
        finally:
            environment.close()
    except Exception as error:
        return {
            "base_id": task.get("base_id"),
            "variant": task.get("variant"),
            "scenario_id": scenario_id,
            "seed": int(task["seed"]),
            "stochastic": bool(task["stochastic"]),
            "horizon_s": float(task["horizon_s"]),
            "status": "ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
        }


def run_tasks(tasks: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    print(f"H2_{label}_START tasks={len(tasks)} workers={WORKERS}", flush=True)
    started = time.monotonic()
    context = mp.get_context("spawn")
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        mp_context=context,
        initializer=worker_init,
        initargs=(str(CHECKPOINT),),
    ) as executor:
        futures = [executor.submit(evaluate_trial, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"H2_{label}_PROGRESS {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: (str(row["variant"]), str(row["base_id"]), int(row["seed"])))
    errors = [row for row in rows if row["status"] == "ERROR"]
    print(f"H2_{label}_DONE seconds={time.monotonic() - started:.1f} errors={len(errors)}", flush=True)
    if errors:
        raise RuntimeError(f"H2 {label} has {len(errors)} errors; first={errors[0]}")
    return rows


def build_variants() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = read_json(SOURCE)
    rows = sorted(
        (row for row in source["scenarios"] if int(row["interval_idx"]) == 8),
        key=lambda row: str(row["scenario_id"]),
    )
    if len(rows) != 199:
        raise RuntimeError(f"Expected 199 old interval-8 H2 core bases, got {len(rows)}")
    bases: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    raceline_lengths = {
        name: len(load_raceline_waypoints("Austin", f"{name}.csv"))
        for name in {str(row["opp_raceline"]) for row in rows}
    }
    for row in rows:
        base_id = str(row["scenario_id"])
        i8 = {**row, "scenario_id": f"h2c-i8-{base_id}"}
        i7 = {
            **row,
            "scenario_id": f"h2c-i7-{base_id}",
            "interval_idx": 7,
            "opp_idx": (int(row["opp_idx"]) - 1) % raceline_lengths[str(row["opp_raceline"])],
        }
        bases.append({"base_id": base_id, "source": row, "I7": i7, "I8": i8})
        variants.extend(
            (
                {"base_id": base_id, "variant": "I7", "scenario": i7},
                {"base_id": base_id, "variant": "I8", "scenario": i8},
            )
        )
    ids = [item["scenario"]["scenario_id"] for item in variants]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Generated H2 variant scenario IDs are not unique")
    document = {
        "schema_version": 1,
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "source_manifest_hash": source["manifest_hash"],
        "source_interval_8_base_count": len(bases),
        "variant_count": len(variants),
        "generation_rule": "I8 preserves the old interval-8 case; I7 changes interval 8 to 7 and its derived opponent index only",
        "bases": bases,
    }
    document["manifest_hash"] = canonical_hash(document)
    write_json(OUTPUT / "H2_MATCHED_BASE.json", document)
    return bases, variants


def deterministic_tasks(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **variant,
            "seed": 20260722,
            "stochastic": False,
            "horizon_s": 8.0,
        }
        for variant in variants
    ]


def stochastic_tasks(
    variants: Iterable[dict[str, Any]],
    seeds: Iterable[int],
) -> list[dict[str, Any]]:
    return [
        {**variant, "seed": seed, "stochastic": True, "horizon_s": 4.0}
        for variant in variants
        for seed in seeds
    ]


def classify_variant(
    variant: dict[str, Any],
    deterministic: dict[tuple[str, str], dict[str, Any]],
    trials: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    key = (variant["base_id"], variant["variant"])
    deterministic_row = deterministic[key]
    trial_rows = sorted(trials.get(key, []), key=lambda row: int(row["seed"]))
    collision_times = [
        float(row["ego_collision_time_s"])
        for row in trial_rows
        if row["outcome"] == "ego_collision"
    ]
    collision_count = len(collision_times)
    safe_count = len(trial_rows) - collision_count
    deterministic_safe = bool(
        deterministic_row["status"] == "COMPLETE"
        and deterministic_row["outcome"] != "ego_collision"
        and deterministic_row["observation_finite"]
        and deterministic_row["action_finite"]
    )
    complete_eight = len(trial_rows) == 8
    primary = bool(
        deterministic_safe
        and complete_eight
        and 2 <= collision_count <= 6
        and safe_count >= 2
        and all(value <= 3.8 + 1e-12 for value in collision_times)
    )
    fallback = bool(
        deterministic_safe
        and complete_eight
        and 1 <= collision_count <= 7
        and safe_count >= 1
        and statistics.median(collision_times) <= 3.8 + 1e-12
    )
    return {
        "base_id": variant["base_id"],
        "variant": variant["variant"],
        "scenario": variant["scenario"],
        "deterministic_8s": deterministic_row,
        "deterministic_safe": deterministic_safe,
        "trials": trial_rows,
        "trial_count": len(trial_rows),
        "collision_count": collision_count,
        "safe_count": safe_count,
        "collision_times_s": collision_times,
        "primary_contrast": primary,
        "fallback_contrast": fallback,
    }


def manifest(pool_id: str, rows: list[dict[str, Any]], tier: str) -> dict[str, Any]:
    scenarios = sorted((row["scenario"] for row in rows), key=lambda row: str(row["scenario_id"]))
    content = {
        "pool_id": pool_id,
        "scenario_ids": [str(row["scenario_id"]) for row in scenarios],
        "scenarios": scenarios,
        "count": len(scenarios),
        "source": {
            "matched_tier": tier,
            "source_path": str(SOURCE.relative_to(ROOT)),
            "source_sha256": sha256_file(SOURCE),
            "classification_seeds": list(PASS1_SEEDS + PASS2_SEEDS),
        },
    }
    return {**content, "manifest_hash": canonical_hash(content)}


def distribution(values: Iterable[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(value): int(counts[value]) for value in range(9)}


def numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def gate_for_interval(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    counts = Counter()
    differences = {name: [] for name in ("mean_desired_speed", "mean_abs_steering", "final_relative_progress_m", "minimum_oriented_clearance_m")}
    pair_count = 0
    for row in rows:
        if row["variant"] != variant:
            continue
        by_seed = {int(trial["seed"]): trial for trial in row["trials"]}
        for left_seed, right_seed in SEED_PAIRS:
            left, right = by_seed[left_seed], by_seed[right_seed]
            left_collision = left["outcome"] == "ego_collision"
            right_collision = right["outcome"] == "ego_collision"
            pair_count += 1
            if left_collision and right_collision:
                counts["collision/collision"] += 1
            elif not left_collision and not right_collision:
                counts["safe/safe"] += 1
            elif left_collision:
                counts["collision/safe"] += 1
            else:
                counts["safe/collision"] += 1
            if left_collision != right_collision:
                collision = left if left_collision else right
                safe = right if left_collision else left
                for name in differences:
                    differences[name].append(float(safe[name]) - float(collision[name]))
    discordant = counts["collision/safe"] + counts["safe/collision"]
    return {
        "variant": variant,
        "pair_count": pair_count,
        "discordant_pair_count": discordant,
        "discordant_pair_rate": discordant / pair_count if pair_count else 0.0,
        "both_collision_rate": counts["collision/collision"] / pair_count if pair_count else 0.0,
        "both_safe_rate": counts["safe/safe"] / pair_count if pair_count else 0.0,
        "outcome_counts": dict(counts),
        "safe_minus_collision_diagnostics": {
            name: numeric_summary(values) for name, values in differences.items()
        },
    }


def preflight() -> None:
    if subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT, text=True).strip():
        raise RuntimeError("H2 pool build requires a clean worktree")
    support = read_json(SUPPORT)
    for relative, expected in support["post_implementation_source_hashes"].items():
        if relative in {"ppo/policy.py", "ppo/environment.py", "ppo/reward.py", "ppo/scenarios.py", "eval_multiagent.py", "evaluate.sh", "utils.py", "pretrained/end2race.pth"}:
            observed = sha256_file(ROOT / relative)
            if observed != expected:
                raise RuntimeError(f"Frozen core drift before H2 pool build: {relative}")
    for name in (
        "H2_MATCHED_BASE.json",
        "H2_I7_CLASSIFICATION.json",
        "H2_I8_CLASSIFICATION.json",
        "H2_MATCHED_CONTRAST.json",
        "H2_MATCHED_CONTRAST_SUMMARY.json",
        "H2_CONDITIONAL_EXPLORATION_GATE.json",
    ):
        if (OUTPUT / name).exists():
            raise RuntimeError(f"H2 output already exists: {name}")


def main() -> int:
    preflight()
    record_command()
    try:
        bases, variants = build_variants()
        deterministic_rows = run_tasks(deterministic_tasks(variants), "DETERMINISTIC_8S")
        deterministic = {
            (str(row["base_id"]), str(row["variant"])): row for row in deterministic_rows
        }
        eligible = [
            variant
            for variant in variants
            if deterministic[(variant["base_id"], variant["variant"])]["status"] == "COMPLETE"
            and deterministic[(variant["base_id"], variant["variant"])]["outcome"] != "ego_collision"
            and deterministic[(variant["base_id"], variant["variant"])]["preflight_valid"]
        ]
        pass1_rows = run_tasks(stochastic_tasks(eligible, PASS1_SEEDS), "STOCHASTIC_PASS1")
        trial_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in pass1_rows:
            trial_map.setdefault((str(row["base_id"]), str(row["variant"])), []).append(row)
        mixed = [
            variant
            for variant in eligible
            if 0
            < sum(
                row["outcome"] == "ego_collision"
                for row in trial_map[(variant["base_id"], variant["variant"])]
            )
            < 4
        ]
        pass2_rows = run_tasks(stochastic_tasks(mixed, PASS2_SEEDS), "STOCHASTIC_PASS2")
        for row in pass2_rows:
            trial_map.setdefault((str(row["base_id"]), str(row["variant"])), []).append(row)

        classified = [classify_variant(variant, deterministic, trial_map) for variant in variants]
        by_variant = {
            variant: sorted(
                (row for row in classified if row["variant"] == variant),
                key=lambda row: str(row["base_id"]),
            )
            for variant in ("I7", "I8")
        }
        for variant in ("I7", "I8"):
            write_json(
                OUTPUT / f"H2_{variant}_CLASSIFICATION.json",
                {
                    "schema_version": 1,
                    "variant": variant,
                    "deterministic_horizon_s": 8.0,
                    "stochastic_horizon_s": 4.0,
                    "steering_latent_std": 0.03,
                    "speed_physical_std": 0.15,
                    "pass1_seeds": list(PASS1_SEEDS),
                    "pass2_seeds": list(PASS2_SEEDS),
                    "rows": by_variant[variant],
                },
            )

        primary_ids = sorted(
            base["base_id"]
            for base in bases
            if all(
                next(row for row in by_variant[variant] if row["base_id"] == base["base_id"])["primary_contrast"]
                for variant in ("I7", "I8")
            )
        )
        fallback_ids = sorted(
            base["base_id"]
            for base in bases
            if all(
                next(row for row in by_variant[variant] if row["base_id"] == base["base_id"])["fallback_contrast"]
                for variant in ("I7", "I8")
            )
        )
        if len(primary_ids) >= 24:
            tier, selected_ids = "PRIMARY", primary_ids
        elif len(fallback_ids) >= 24:
            tier, selected_ids = "FALLBACK", fallback_ids
        else:
            tier, selected_ids = "STOP_H2_MATCHED_POOL_TOO_SMALL", []
        selected_rows = [row for row in classified if row["base_id"] in selected_ids]
        contrast: dict[str, Any] = {
            "schema_version": 1,
            "status": "PASS" if selected_ids else tier,
            "selected_tier": tier,
            "selected_base_ids": selected_ids,
            "selected_base_count": len(selected_ids),
            "primary_matched_base_ids": primary_ids,
            "fallback_matched_base_ids": fallback_ids,
            "variant_manifests": {},
        }
        if selected_ids:
            contrast["variant_manifests"] = {
                "H2_MATCHED_CONTRAST_I7": manifest(
                    "H2_MATCHED_CONTRAST_I7",
                    [row for row in selected_rows if row["variant"] == "I7"],
                    tier,
                ),
                "H2_MATCHED_CONTRAST_I8": manifest(
                    "H2_MATCHED_CONTRAST_I8",
                    [row for row in selected_rows if row["variant"] == "I8"],
                    tier,
                ),
            }
        contrast["manifest_hash"] = canonical_hash(contrast)
        write_json(OUTPUT / "H2_MATCHED_CONTRAST.json", contrast)

        summary = {
            "schema_version": 1,
            "status": contrast["status"],
            "source_base_count": len(bases),
            "preflight": {
                variant: {
                    "valid": sum(row["deterministic_8s"]["preflight_valid"] for row in by_variant[variant]),
                    "invalid": sum(not row["deterministic_8s"]["preflight_valid"] for row in by_variant[variant]),
                }
                for variant in ("I7", "I8")
            },
            "deterministic_safe_counts": {
                variant: sum(row["deterministic_safe"] for row in by_variant[variant])
                for variant in ("I7", "I8")
            },
            "collision_count_distributions": {
                variant: distribution(row["collision_count"] for row in by_variant[variant])
                for variant in ("I7", "I8")
            },
            "collision_time_distributions_s": {
                variant: numeric_summary(
                    [value for row in by_variant[variant] for value in row["collision_times_s"]]
                )
                for variant in ("I7", "I8")
            },
            "primary_matched_count": len(primary_ids),
            "fallback_matched_count": len(fallback_ids),
            "selected_tier": tier,
            "selected_matched_count": len(selected_ids),
            "matched_manifest_hash": contrast["manifest_hash"],
        }
        write_json(OUTPUT / "H2_MATCHED_CONTRAST_SUMMARY.json", summary)

        if not selected_ids:
            gate = {
                "schema_version": 1,
                "status": "NOT_RUN_H2_MATCHED_POOL_TOO_SMALL",
                "H2_CONDITIONAL_POOL_VALID": False,
                "selected_matched_base_count": 0,
            }
        else:
            interval_gates = [gate_for_interval(selected_rows, variant) for variant in ("I7", "I8")]
            gate_pass = bool(
                len(selected_ids) >= 24
                and all(
                    row["discordant_pair_count"] >= 16
                    and 0.20 <= row["discordant_pair_rate"] <= 0.80
                    for row in interval_gates
                )
            )
            gate = {
                "schema_version": 1,
                "status": "PASS" if gate_pass else "FAIL",
                "H2_CONDITIONAL_POOL_VALID": gate_pass,
                "selected_tier": tier,
                "selected_matched_base_count": len(selected_ids),
                "seed_pairs": [list(pair) for pair in SEED_PAIRS],
                "simulation_reused_classification_rows": True,
                "error": 0,
                "intervals": interval_gates,
            }
        write_json(OUTPUT / "H2_CONDITIONAL_EXPLORATION_GATE.json", gate)
        print(json.dumps({"summary": summary, "gate": gate}, indent=2), flush=True)
        return 0
    except Exception as error:
        write_json(
            OUTPUT / "H2_POOL_FAILURE.json",
            {
                "schema_version": 1,
                "status": "INVALID",
                "error_type": type(error).__name__,
                "error": str(error),
                "recorded_at_utc": utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
