#!/usr/bin/env python3
"""Standalone hard-neighbor sampler and BC rollout probe.

This script never enters the PPO scheduler. It expands cached *training*
collision scenarios by only one interval index and 0.05 speed scale, reruns a
small deterministic panel with the frozen BC actor, and reports whether the
new scenarios retain a collision rate above the global candidate pool.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model import End2Race
from ppo.environment import EXTERNAL_RESET_OPTION, make_environment
from ppo.policy import END2RACE_LIDAR_SIZE, STEERING_BOUND
from ppo.scenarios import ScenarioSpec, evaluation_startpoints
from utils import find_corresponding_waypoint, load_raceline_waypoints


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "post-trained/collision-cache/default"
DEFAULT_MODEL = PROJECT_ROOT / "pretrained/end2race.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis_results/hard_neighbor_probe"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe training-cache hard neighbors")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--map-name", default="Austin")
    parser.add_argument("--hidden-scale", type=int, default=4)
    parser.add_argument("--neighbors-per-seed", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.hidden_scale <= 0 or args.neighbors_per_seed <= 0:
        raise ValueError("hidden_scale and neighbors_per_seed must be positive")
    return args


def load_cache(cache_dir: Path) -> tuple[list[ScenarioSpec], dict[str, Any]]:
    records = json.loads((cache_dir / "collision_scenarios.json").read_text())
    summary = json.loads((cache_dir / "classification_summary.json").read_text())
    scenarios = [ScenarioSpec(**record) for record in records]
    if not scenarios or len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise RuntimeError("Collision cache must contain unique scenarios")
    if summary["collision_count"] != len(scenarios):
        raise RuntimeError("Collision cache summary does not match its scenarios")
    return scenarios, summary


def local_collision_support(seed: ScenarioSpec, collisions: list[ScenarioSpec]) -> int:
    return sum(
        other.scenario_id != seed.scenario_id
        and other.ego_idx == seed.ego_idx
        and other.opp_raceline == seed.opp_raceline
        and abs(other.interval_idx - seed.interval_idx) <= 2
        and abs(other.opp_speedscale - seed.opp_speedscale) <= 0.0500001
        for other in collisions
    )


def select_seeds(collisions: list[ScenarioSpec]) -> list[ScenarioSpec]:
    selected = []
    for raceline in sorted({scenario.opp_raceline for scenario in collisions}):
        group = [scenario for scenario in collisions if scenario.opp_raceline == raceline]
        group.sort(key=lambda scenario: (-local_collision_support(scenario, collisions), scenario.scenario_id))
        selected.append(group[0])
    return selected


def neighbor_pool(seed: ScenarioSpec) -> list[ScenarioSpec]:
    ego_waypoints = load_raceline_waypoints(seed.map_name, f"{seed.ego_raceline}.csv")
    opponent_waypoints = load_raceline_waypoints(seed.map_name, f"{seed.opp_raceline}.csv")
    ego_waypoint = ego_waypoints[seed.ego_idx]
    mapped_index = (
        seed.ego_idx
        if seed.opp_raceline == seed.ego_raceline
        else int(find_corresponding_waypoint(ego_waypoint, opponent_waypoints))
    )
    neighbors = []
    for interval_delta in (-1, 1):
        interval_idx = seed.interval_idx + interval_delta
        for speed_delta in (-0.05, 0.0, 0.05):
            speed = float(round(seed.opp_speedscale + speed_delta, 2))
            if interval_idx <= 0 or not 0.45 <= speed <= 0.85:
                continue
            opp_idx = int((mapped_index + interval_idx) % (len(opponent_waypoints) - 1))
            scenario_id = (
                f"hard-{seed.scenario_id}-di{interval_delta:+d}"
                f"-dv{int(round(speed_delta * 100)):+03d}"
            )
            neighbors.append(
                ScenarioSpec(
                    scenario_id=scenario_id,
                    pool="hard_neighbor",
                    startpoint_ordinal=seed.startpoint_ordinal,
                    ego_idx=seed.ego_idx,
                    opp_idx=opp_idx,
                    opp_raceline=seed.opp_raceline,
                    opp_speedscale=speed,
                    interval_idx=interval_idx,
                    map_name=seed.map_name,
                    ego_raceline=seed.ego_raceline,
                    sim_duration=seed.sim_duration,
                    timestep=seed.timestep,
                    integrator=seed.integrator,
                )
            )
    return neighbors


def sample_neighbors(
    seeds: list[ScenarioSpec], neighbors_per_seed: int, random_seed: int
) -> list[tuple[ScenarioSpec, ScenarioSpec]]:
    rng = np.random.default_rng(random_seed)
    sampled: list[tuple[ScenarioSpec, ScenarioSpec]] = []
    physical_keys: set[tuple[int, str, int, float]] = set()
    for seed in seeds:
        pool = neighbor_pool(seed)
        if neighbors_per_seed > len(pool):
            raise ValueError(
                f"Requested {neighbors_per_seed} neighbors from {len(pool)} for {seed.scenario_id}"
            )
        for index in rng.choice(len(pool), size=neighbors_per_seed, replace=False):
            neighbor = pool[int(index)]
            key = (
                neighbor.ego_idx,
                neighbor.opp_raceline,
                neighbor.interval_idx,
                neighbor.opp_speedscale,
            )
            if key in physical_keys:
                raise RuntimeError(f"Duplicate physical neighbor: {key}")
            physical_keys.add(key)
            sampled.append((seed, neighbor))
    return sampled


def minimum_evaluation_distance(scenario: ScenarioSpec) -> float:
    waypoints = load_raceline_waypoints(scenario.map_name, f"{scenario.ego_raceline}.csv")
    evaluation_xy = waypoints[np.asarray(evaluation_startpoints(scenario.map_name)), :2]
    return float(np.linalg.norm(evaluation_xy - waypoints[scenario.ego_idx, :2], axis=1).min())


def rollout(
    env: Any,
    actor: End2Race,
    scenario: ScenarioSpec,
    *,
    kind: str,
    source_scenario_id: str,
) -> dict[str, Any]:
    observation, _ = env.reset(
        options={EXTERNAL_RESET_OPTION: scenario.to_reset_spec("collision")}
    )
    if not np.isfinite(observation).all() or np.asarray(env._raw_observation["collisions"]).any():
        raise RuntimeError(f"Invalid initial state for {scenario.scenario_id}")
    hidden = None
    while True:
        actor_observation = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            actions, hidden = actor(
                actor_observation[:END2RACE_LIDAR_SIZE].reshape(1, 1, -1),
                actor_observation[END2RACE_LIDAR_SIZE:].reshape(1, 1, 1),
                hidden,
            )
        action = actions[0, -1].numpy().copy()
        action[0] = np.clip(action[0], -STEERING_BOUND, STEERING_BOUND)
        if not np.isfinite(action).all():
            raise RuntimeError(f"Non-finite BC action for {scenario.scenario_id}")
        observation, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return {
                "kind": kind,
                "source_scenario_id": source_scenario_id,
                "scenario_id": scenario.scenario_id,
                "opp_raceline": scenario.opp_raceline,
                "ego_idx": scenario.ego_idx,
                "interval_idx": scenario.interval_idx,
                "opp_speedscale": scenario.opp_speedscale,
                "ego_collision": bool(info["ego_collision"]),
                "outcome": str(info["episode_outcome"]),
                "elapsed_time_s": float(info["elapsed_time"]),
                "steps": int(info["episode_steps"]),
                "episode_return": float(info["episode_return"]),
                "minimum_obb_clearance_m": float(info["episode_min_obb_clearance_m"]),
                "minimum_wall_clearance_m": float(info["episode_min_wall_clearance_m"]),
                "minimum_evaluation_start_distance_m": minimum_evaluation_distance(scenario),
            }


def write_outputs(
    output_dir: Path,
    selected: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_scenarios.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)


def main() -> None:
    args = parse_arguments()
    cache_dir = args.cache_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    collisions, cache_summary = load_cache(cache_dir)
    seeds = select_seeds(collisions)
    sampled = sample_neighbors(seeds, args.neighbors_per_seed, args.seed)
    selected_specs = seeds + [neighbor for _source, neighbor in sampled]
    if len({scenario.scenario_id for scenario in selected_specs}) != len(selected_specs):
        raise RuntimeError("Selected scenario IDs are not unique")
    reset_specs = [scenario.to_reset_spec("collision") for scenario in selected_specs]
    if not all(
        np.isfinite(spec.poses).all() and np.isfinite(spec.initial_speed_feature)
        for spec in reset_specs
    ):
        raise RuntimeError("Selected scenarios do not have finite reset parameters")
    if not all(minimum_evaluation_distance(scenario) >= 1.0 - 1e-12 for scenario in selected_specs):
        raise RuntimeError("Hard-neighbor probe violated the training/evaluation separation")

    torch.set_num_threads(1)
    actor = End2Race(mask_prob=0.0, hidden_scale=args.hidden_scale)
    actor.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True), strict=True
    )
    actor.eval()
    env = make_environment(0, args.map_name)()
    episodes = []
    try:
        for index, seed in enumerate(seeds, start=1):
            result = rollout(
                env,
                actor,
                seed,
                kind="cached_seed",
                source_scenario_id=seed.scenario_id,
            )
            episodes.append(result)
            print(
                f"seed {index}/{len(seeds)} {seed.scenario_id}: "
                f"{result['outcome']} t={result['elapsed_time_s']:.2f}s",
                flush=True,
            )
        for index, (source, neighbor) in enumerate(sampled, start=1):
            result = rollout(
                env,
                actor,
                neighbor,
                kind="hard_neighbor",
                source_scenario_id=source.scenario_id,
            )
            episodes.append(result)
            print(
                f"neighbor {index}/{len(sampled)} {neighbor.scenario_id}: "
                f"{result['outcome']} t={result['elapsed_time_s']:.2f}s",
                flush=True,
            )
    finally:
        env.close()

    seed_results = [episode for episode in episodes if episode["kind"] == "cached_seed"]
    neighbor_results = [episode for episode in episodes if episode["kind"] == "hard_neighbor"]
    neighbor_collisions = sum(episode["ego_collision"] for episode in neighbor_results)
    valid_candidates = cache_summary["candidate_count"] - cache_summary["invalid_count"]
    baseline_rate = cache_summary["collision_count"] / valid_candidates
    neighbor_rate = neighbor_collisions / len(neighbor_results)
    source_family_results = {}
    for seed in seeds:
        family = [
            episode
            for episode in neighbor_results
            if episode["source_scenario_id"] == seed.scenario_id
        ]
        family_collisions = sum(episode["ego_collision"] for episode in family)
        source_family_results[seed.scenario_id] = {
            "opponent_raceline": seed.opp_raceline,
            "neighbor_count": len(family),
            "collision_count": family_collisions,
            "collision_rate": family_collisions / len(family),
        }
    enriched_family_count = sum(
        family["collision_rate"] > baseline_rate
        for family in source_family_results.values()
    )
    checks = {
        "cached_seeds_reproduce_collision": all(episode["ego_collision"] for episode in seed_results),
        "selected_scenarios_are_finite": True,
        "training_evaluation_separation_at_least_1m": all(
            episode["minimum_evaluation_start_distance_m"] >= 1.0 - 1e-12
            for episode in episodes
        ),
        "hard_neighbors_enrich_collision_rate": neighbor_rate > baseline_rate,
        "at_least_two_source_families_are_enriched": enriched_family_count >= 2,
    }
    summary = {
        "config": {
            "cache_dir": str(cache_dir),
            "model": str(model_path),
            "map_name": args.map_name,
            "random_seed": args.seed,
            "seed_selection": "highest local collision support per opponent raceline",
            "neighbor_definition": "same training ego start, interval +/-1, speed +/-0.05",
            "pipeline_integration": False,
        },
        "global_candidate_collision_rate": baseline_rate,
        "seed_count": len(seed_results),
        "seed_collision_count": sum(episode["ego_collision"] for episode in seed_results),
        "neighbor_count": len(neighbor_results),
        "neighbor_collision_count": neighbor_collisions,
        "neighbor_collision_rate": neighbor_rate,
        "collision_rate_enrichment": neighbor_rate / baseline_rate,
        "source_family_results": source_family_results,
        "enriched_source_family_count": enriched_family_count,
        "statistical_note": (
            "Descriptive probe only: neighbors within a selected collision family are correlated, "
            "so no independent-binomial p-value is reported."
        ),
        "checks": checks,
        "verdict": (
            "hard_neighbors_are_enriched_probe_candidates"
            if all(checks.values())
            else "hard_neighbor_design_needs_revision"
        ),
    }
    selected = [
        {
            "kind": "cached_seed",
            "local_collision_support": local_collision_support(seed, collisions),
            **asdict(seed),
        }
        for seed in seeds
    ] + [
        {
            "kind": "hard_neighbor",
            "source_scenario_id": source.scenario_id,
            **asdict(neighbor),
        }
        for source, neighbor in sampled
    ]
    write_outputs(output_dir, selected, episodes, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
