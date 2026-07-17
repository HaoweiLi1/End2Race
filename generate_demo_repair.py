"""Generate Route B planner demonstrations from a locked H0/H1 selection."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Sequence

import gym
import f110_gym  # noqa: F401 - registers the legacy F110 Gym environment
import numpy as np
from f110_gym.envs.base_classes import Integrator

from demonstration import setup_ego_planner, setup_opp_planner
from latticeplanner.utils import downsample_lidar, obsDict2oppoArray
from ppo.reward import ProgressProjector
from ppo.scenarios import ScenarioSpec
from utils import wrapped_progress_difference


LOCKED_SELECTION_SHA256 = "fe01f648af7f942a793bd1ba390931c93b6e54126ee0251b2881b881fe6ecbbe"
EXPECTED_H0_COUNT = 24
EXPECTED_H1_COUNT = 162
EXPECTED_TOTAL_COUNT = EXPECTED_H0_COUNT + EXPECTED_H1_COUNT
SAMPLE_INTERVAL_S = 0.1
EXPECTED_SAMPLES = 80
STEERING_EDGES = np.linspace(-0.52, 0.52, 21, dtype=np.float64)
SPEED_EDGES = np.linspace(0.0, 10.0, 21, dtype=np.float64)
SCENARIO_FIELDS = tuple(ScenarioSpec.__dataclass_fields__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(temporary_path, **payload)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def downsample_bc_lidar(scan: np.ndarray) -> np.ndarray:
    """Apply the original demonstration ``scan[::4][:360]`` contract."""
    downsampled = np.asarray(
        downsample_lidar(scan, original_points=1440, target_points=360),
        dtype=np.float32,
    )
    if downsampled.shape != (360,):
        raise ValueError(f"BC LiDAR downsample must have shape (360,), got {downsampled.shape}")
    if not np.isfinite(downsampled).all():
        raise ValueError("BC LiDAR downsample contains non-finite values")
    return downsampled


def build_npz_payload(
    time_s: Sequence[float],
    steering: Sequence[float],
    desired_speed: Sequence[float],
    lidar: Sequence[np.ndarray],
) -> dict[str, np.ndarray]:
    payload = {
        "time_s": np.asarray(time_s, dtype=np.float64),
        "steer": np.asarray(steering, dtype=np.float32),
        "desired_speed": np.asarray(desired_speed, dtype=np.float32),
        "lidar": np.asarray(lidar, dtype=np.float32),
    }
    expected_shapes = {
        "time_s": (EXPECTED_SAMPLES,),
        "steer": (EXPECTED_SAMPLES,),
        "desired_speed": (EXPECTED_SAMPLES,),
        "lidar": (EXPECTED_SAMPLES, 360),
    }
    for name, expected_shape in expected_shapes.items():
        if payload[name].shape != expected_shape:
            raise ValueError(
                f"Demonstration {name} must have shape {expected_shape}, got {payload[name].shape}"
            )
        if not np.isfinite(payload[name]).all():
            raise ValueError(f"Demonstration {name} contains non-finite values")
    expected_time = np.arange(1, EXPECTED_SAMPLES + 1, dtype=np.float64) * SAMPLE_INTERVAL_S
    if not np.allclose(payload["time_s"], expected_time, rtol=0.0, atol=1e-12):
        raise ValueError("Demonstration sample times are not exactly 0.1 through 8.0 seconds")
    return payload


def derive_bc_training_arrays(
    payload: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expose the exact alignment performed by ``train.py::SequenceDataset``."""
    lidar = np.asarray(payload["lidar"], dtype=np.float32)
    steering = np.asarray(payload["steer"], dtype=np.float32)
    desired_speed = np.asarray(payload["desired_speed"], dtype=np.float32)
    action = np.column_stack((steering, desired_speed)).astype(np.float32, copy=False)
    return lidar[1:], desired_speed[:-1, None], action[1:]


def _scenario_spec(row: dict[str, Any]) -> ScenarioSpec:
    return ScenarioSpec(**{name: row[name] for name in SCENARIO_FIELDS})


def _run_episode(
    scenario_source: str,
    scenario_row: dict[str, Any],
    output_directory: str,
) -> dict[str, Any]:
    output_root = Path(output_directory)
    metrics_path = output_root / "metrics" / f"{scenario_row['scenario_id']}.json"
    try:
        scenario = _scenario_spec(scenario_row)
        reset_spec = scenario.to_reset_spec()
        ego_planner, _ = setup_ego_planner(scenario.map_name, scenario.ego_raceline)
        opponent_planner = setup_opp_planner(scenario.map_name, scenario.opp_raceline)
        projector = ProgressProjector.from_csv(
            f"f1tenth_racetracks/{scenario.map_name}/raceline1.csv"
        )
        environment = gym.make(
            "f110-v0",
            map=ego_planner.map_path,
            map_ext=".png",
            timestep=scenario.timestep,
            num_agents=2,
            integrator=Integrator[scenario.integrator],
        )

        time_samples: list[float] = []
        steering_samples: list[float] = []
        speed_samples: list[float] = []
        lidar_samples: list[np.ndarray] = []
        elapsed = 0.0
        steps = 0
        ego_collision = False
        opponent_collision = False
        environment_done_before_timeout = False
        try:
            observation, _, done, _ = environment.reset(poses=reset_spec.poses)
            ego_progress = projector.project(
                np.asarray([observation["poses_x"][0], observation["poses_y"][0]])
            )
            opponent_progress = projector.project(
                np.asarray([observation["poses_x"][1], observation["poses_y"][1]])
            )
            relative_position = wrapped_progress_difference(
                ego_progress, opponent_progress, projector.track_length
            )
            previous_relative = relative_position
            next_sample_time = SAMPLE_INTERVAL_S
            tracker_steps = int(ego_planner.conf.tracker_steps)

            while not done and elapsed < scenario.sim_duration:
                ego_trajectory = ego_planner.plan(
                    observation["poses_x"][0],
                    observation["poses_y"][0],
                    observation["poses_theta"][0],
                    obsDict2oppoArray(observation, 0),
                    observation["linear_vels_x"][0],
                )
                opponent_trajectory = opponent_planner.plan(
                    observation["poses_x"][1],
                    observation["poses_y"][1],
                    observation["poses_theta"][1],
                    obsDict2oppoArray(observation, 1),
                    observation["linear_vels_x"][1],
                )
                tracker_count = 0
                while (
                    not done
                    and tracker_count < tracker_steps
                    and elapsed < scenario.sim_duration
                ):
                    ego_steer, ego_speed = ego_planner.tracker.plan(
                        observation["poses_x"][0],
                        observation["poses_y"][0],
                        observation["poses_theta"][0],
                        observation["linear_vels_x"][0],
                        ego_trajectory,
                    )
                    opponent_steer, opponent_speed = opponent_planner.tracker.plan(
                        observation["poses_x"][1],
                        observation["poses_y"][1],
                        observation["poses_theta"][1],
                        observation["linear_vels_x"][1],
                        opponent_trajectory,
                    )
                    ego_steer = float(np.clip(ego_steer, -0.52, 0.52))
                    ego_speed = float(ego_speed)
                    opponent_steer = float(np.clip(opponent_steer, -0.52, 0.52))
                    opponent_speed = float(opponent_speed * scenario.opp_speedscale)
                    observation, timestep, done, _ = environment.step(
                        np.asarray(
                            [[ego_steer, ego_speed], [opponent_steer, opponent_speed]],
                            dtype=np.float64,
                        )
                    )
                    elapsed += float(timestep)
                    steps += 1
                    tracker_count += 1

                    while (
                        elapsed + 1e-12 >= next_sample_time
                        and next_sample_time <= scenario.sim_duration + 1e-12
                    ):
                        time_samples.append(round(next_sample_time, 10))
                        steering_samples.append(ego_steer)
                        speed_samples.append(ego_speed)
                        lidar_samples.append(downsample_bc_lidar(observation["scans"][0]))
                        next_sample_time += SAMPLE_INTERVAL_S

                    ego_progress = projector.project(
                        np.asarray([observation["poses_x"][0], observation["poses_y"][0]])
                    )
                    opponent_progress = projector.project(
                        np.asarray([observation["poses_x"][1], observation["poses_y"][1]])
                    )
                    current_relative = wrapped_progress_difference(
                        ego_progress, opponent_progress, projector.track_length
                    )
                    relative_position += wrapped_progress_difference(
                        current_relative, previous_relative, projector.track_length
                    )
                    previous_relative = current_relative

                    step_ego_collision = bool(observation["collisions"][0])
                    step_opponent_collision = bool(observation["collisions"][1])
                    ego_collision = ego_collision or step_ego_collision
                    opponent_collision = opponent_collision or step_opponent_collision
                    if step_ego_collision or step_opponent_collision:
                        done = True
            environment_done_before_timeout = bool(
                done
                and elapsed < scenario.sim_duration - 1e-9
                and not ego_collision
                and not opponent_collision
            )
        finally:
            environment.close()

        complete = elapsed >= scenario.sim_duration - 1e-9
        retained = bool(
            complete
            and not ego_collision
            and not opponent_collision
            and not environment_done_before_timeout
            and len(time_samples) == EXPECTED_SAMPLES
        )
        if ego_collision:
            outcome = "ego_collision"
        elif opponent_collision:
            outcome = "opponent_collision_only"
        elif not retained:
            outcome = "incomplete"
        elif relative_position > 0.0:
            outcome = "overtake"
        else:
            outcome = "follow"

        dataset_path = None
        dataset_sha256 = None
        if retained:
            payload = build_npz_payload(
                time_samples, steering_samples, speed_samples, lidar_samples
            )
            dataset_path = output_root / "dataset" / f"{scenario.scenario_id}.npz"
            _write_npz(dataset_path, payload)
            dataset_sha256 = _sha256(dataset_path)

        result = {
            "dataset_path": str(dataset_path) if dataset_path is not None else None,
            "dataset_sha256": dataset_sha256,
            "ego_collision_occurred": ego_collision,
            "environment_done_before_timeout": environment_done_before_timeout,
            "final_relative_position_m": float(relative_position),
            "opponent_collision_occurred": opponent_collision,
            "outcome": outcome,
            "retained": retained,
            "sample_count": len(time_samples),
            "scenario": scenario_row,
            "scenario_id": scenario.scenario_id,
            "scenario_source": scenario_source,
            "simulation_time_s": min(float(elapsed), float(scenario.sim_duration)),
            "steps": steps,
        }
        _write_json(metrics_path, result)
        return result
    except Exception as error:
        result = {
            "dataset_path": None,
            "dataset_sha256": None,
            "error": f"{type(error).__name__}: {error}",
            "outcome": "error",
            "retained": False,
            "scenario": scenario_row,
            "scenario_id": scenario_row.get("scenario_id"),
            "scenario_source": scenario_source,
            "traceback": traceback.format_exc(),
        }
        _write_json(metrics_path, result)
        return result


def _distribution(values: np.ndarray, edges: np.ndarray) -> dict[str, Any]:
    histogram, _ = np.histogram(values, bins=edges)
    return {
        "count": int(values.size),
        "finite": bool(np.isfinite(values).all()),
        "histogram_counts": histogram.astype(int).tolist(),
        "histogram_edges": edges.tolist(),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "outside_histogram_count": int(np.sum((values < edges[0]) | (values > edges[-1]))),
        "std": float(np.std(values)),
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        default="runs/ppo/DR_B1_20260717/selected_scenarios.json",
    )
    parser.add_argument("--output-dir", default="runs/ppo/DR_B2_20260717")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = _parse_arguments()
    selection_path = Path(args.selection)
    output_root = Path(args.output_dir)
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1,8]")
    if _sha256(selection_path) != LOCKED_SELECTION_SHA256:
        raise ValueError("Scenario selection hash does not match the locked B2 contract")
    selection = json.loads(selection_path.read_text())
    if len(selection["h0_scenarios"]) != EXPECTED_H0_COUNT:
        raise ValueError("Locked selection does not contain 24 H0 scenarios")
    if len(selection["h1_scenarios"]) != EXPECTED_H1_COUNT:
        raise ValueError("Locked selection does not contain 162 H1 scenarios")
    scenarios = [
        *(('H0', row) for row in selection["h0_scenarios"]),
        *(('H1', row) for row in selection["h1_scenarios"]),
    ]
    if len(scenarios) != EXPECTED_TOTAL_COUNT:
        raise ValueError("Locked selection does not contain 186 total scenarios")
    scenario_ids = [row["scenario_id"] for _, row in scenarios]
    if len(set(scenario_ids)) != EXPECTED_TOTAL_COUNT:
        raise ValueError("Locked selection scenario IDs are not unique")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_root}")
    (output_root / "dataset").mkdir(parents=True, exist_ok=True)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    print(f"DR_B2_START scenarios={len(scenarios)} workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_episode, source, row, str(output_root)): row["scenario_id"]
            for source, row in scenarios
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            print(
                f"DR_B2_CASE {completed:03d}/{EXPECTED_TOTAL_COUNT} "
                f"{result['scenario_id']} outcome={result['outcome']} "
                f"retained={result['retained']}",
                flush=True,
            )
    results.sort(key=lambda item: item["scenario_id"])

    outcome_counts = {
        outcome: sum(result["outcome"] == outcome for result in results)
        for outcome in (
            "ego_collision",
            "opponent_collision_only",
            "overtake",
            "follow",
            "incomplete",
            "error",
        )
    }
    retained_results = [result for result in results if result["retained"]]
    steering_parts = []
    speed_parts = []
    dataset_manifest = []
    for result in retained_results:
        dataset_path = Path(result["dataset_path"])
        with np.load(dataset_path) as payload:
            steering_parts.append(np.asarray(payload["steer"], dtype=np.float64))
            speed_parts.append(np.asarray(payload["desired_speed"], dtype=np.float64))
        dataset_manifest.append(
            {
                "path": str(dataset_path),
                "scenario_id": result["scenario_id"],
                "sha256": result["dataset_sha256"],
            }
        )
    if not retained_results:
        raise RuntimeError("B2 generation retained zero demonstrations")
    steering = np.concatenate(steering_parts)
    desired_speed = np.concatenate(speed_parts)
    manifest_bytes = (
        json.dumps(dataset_manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    dataset_manifest_path = output_root / "dataset_manifest.json"
    dataset_manifest_path.write_bytes(manifest_bytes)

    summary = {
        "action_distribution": {
            "desired_speed": _distribution(desired_speed, SPEED_EDGES),
            "steering": _distribution(steering, STEERING_EDGES),
        },
        "attempted_count": len(results),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": _sha256(Path(__file__)),
        "outcome_counts": outcome_counts,
        "retained_count": len(retained_results),
        "retained_fraction": len(retained_results) / len(results),
        "sample_count": int(desired_speed.size),
        "scenario_selection_path": str(selection_path),
        "scenario_selection_sha256": LOCKED_SELECTION_SHA256,
        "workers": args.workers,
    }
    _write_json(output_root / "raw_summary.json", summary)
    print("DR_B2_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 1 if outcome_counts["error"] or outcome_counts["incomplete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
