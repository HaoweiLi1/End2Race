#!/usr/bin/env python3
"""Evaluate End2Race actors with the historical persistent CPU worker contract."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import End2Race  # noqa: E402
from ppo.scenarios import ScenarioSpec, evaluation_scenarios  # noqa: E402
from utils import atomic_write_json  # noqa: E402


MODEL: End2Race | None = None
DEVICE = torch.device("cpu")


def worker_init(model_path: str) -> None:
    global MODEL
    torch.set_num_threads(1)
    model = End2Race(mask_prob=0.0, hidden_scale=4).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    MODEL = model


def evaluate_one(scenario: ScenarioSpec) -> dict[str, Any]:
    try:
        if MODEL is None:
            raise RuntimeError("CPU evaluation worker model is not initialized")
        from eval_multiagent import evaluate_segment

        result = evaluate_segment(
            MODEL,
            DEVICE,
            0.0,
            scenario.map_name,
            scenario.ego_idx,
            scenario.interval_idx,
            scenario.ego_raceline,
            scenario.opp_raceline,
            scenario.opp_speedscale,
            scenario.sim_duration,
            False,
            False,
            "actor.pth",
            None,
            "ego",
            scenario.scenario_id,
        )
        return result["episode_metrics"]
    except Exception as error:
        return {
            "scenario_id": scenario.scenario_id,
            "episode_key": f"error-{scenario.scenario_id}",
            "error_type": type(error).__name__,
            "error": str(error),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1, 8]")
    scenarios = evaluation_scenarios()
    if not 1 <= args.limit <= len(scenarios):
        raise ValueError("limit must be in [1, 600]")
    scenarios = scenarios[: args.limit]

    print("EVAL_DEVICE=cpu", flush=True)
    print("TORCH_NUM_THREADS=1", flush=True)
    print("EVALUATOR=persistent_spawn_pool", flush=True)
    print(f"Starting batch evaluation of {len(scenarios)} segments", flush=True)
    print(f"Model: {args.model_path}", flush=True)
    print("Map: Austin", flush=True)
    print(f"Workers: {args.workers}", flush=True)
    print("Noise level: 0.0", flush=True)

    started = time.monotonic()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=worker_init,
        initargs=(str(args.model_path.resolve()),),
    ) as executor:
        rows = list(executor.map(evaluate_one, scenarios, chunksize=1))

    errors = [row for row in rows if "error" in row]
    successful = [row for row in rows if "error" not in row]
    outcomes = Counter(str(row["outcome"]) for row in successful)
    counts = {
        "following": int(outcomes["follow"]),
        "overtaking": int(outcomes["overtake"]),
        "collision": int(outcomes["ego_collision"]),
        "error": len(errors),
    }

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in successful])) if successful else 0.0

    denominator = len(scenarios)
    final = {
        "total_episodes": denominator,
        "following_count": counts["following"],
        "overtaking_count": counts["overtaking"],
        "success_count": counts["following"] + counts["overtaking"],
        "collision_count": counts["collision"],
        "error_count": counts["error"],
        "following_rate": 100.0 * counts["following"] / denominator,
        "overtaking_rate": 100.0 * counts["overtaking"] / denominator,
        "success_rate": 100.0 * (counts["following"] + counts["overtaking"]) / denominator,
        "collision_rate": 100.0 * counts["collision"] / denominator,
        "avg_speed_mean": mean("avg_speed"),
        "speed_variance_mean": mean("speed_variance"),
        "total_distance_mean": mean("total_distance"),
    }
    episode_map = {str(row["episode_key"]): row for row in rows}
    if len(episode_map) != len(rows):
        raise RuntimeError("Evaluation episode keys are not unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, {"final": final, "episodes": dict(sorted(episode_map.items()))})

    elapsed = int(time.monotonic() - started)
    print(f"Evaluation complete in {elapsed} seconds", flush=True)
    print("Results by category:", flush=True)
    print(f"  following: {counts['following']}", flush=True)
    print(f"  overtaking: {counts['overtaking']}", flush=True)
    print(f"  collision: {counts['collision']}", flush=True)
    print(f"  error: {counts['error']}", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} evaluation workers returned errors; first={errors[0]}")


if __name__ == "__main__":
    main()
