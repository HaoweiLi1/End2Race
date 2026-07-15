#!/usr/bin/env python3
"""Batch CLI for run-scoped End2Race multi-agent evaluation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import multiprocessing
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import torch

from evaluation.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    checkpoint_sha256,
    initialize_run,
    load_json,
    run_directory,
    valid_episode_file,
)
from evaluation.metrics import aggregate_episodes
from evaluation.multiagent import evaluate_worker_job, initialize_worker, opponent_start_index
from evaluation.schema import EVALUATION_SCHEMA_VERSION, Scenario


EPISODE_CSV_FIELDS = (
    "scenario_id",
    "map_name",
    "ego_raceline",
    "opponent_raceline",
    "ego_start_index",
    "opponent_start_index",
    "interval_index",
    "opponent_speed_scale",
    "simulation_duration_s",
    "outcome",
    "ego_collision",
    "opponent_collision",
    "opponent_only_collision",
    "collision_step",
    "steps",
    "elapsed_time_s",
    "final_ego_progress_m",
    "final_opp_progress_m",
    "final_relative_progress_m",
    "ego_distance_m",
    "ego_mean_measured_speed_mps",
    "ego_speed_variance",
    "ego_min_measured_speed_mps",
    "ego_mean_desired_speed_mps",
    "ego_max_abs_steer_rad",
    "ego_max_steer_delta_rad",
    "ego_min_lidar_m",
    "trace_path",
    "video_path",
)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="12-key End2Race actor checkpoint")
    parser.add_argument("--output-root", default="evaluation_results")
    parser.add_argument("--suite-name", default="default")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--map", dest="map_name", default="Austin")
    parser.add_argument("--ego-racelines", nargs="+", default=["raceline1"])
    parser.add_argument("--opponent-racelines", nargs="+", default=["raceline1"])
    parser.add_argument("--speed-scales", nargs="+", type=float, default=[0.5])
    parser.add_argument("--startpoints", nargs="+", type=int, default=[0])
    parser.add_argument("--intervals", nargs="+", type=int, default=[15])
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--scenario-manifest", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hidden-scale", type=int, default=4)
    parser.add_argument("--trace-mode", choices=("none", "collision", "all"), default="none")
    parser.add_argument(
        "--video-scenarios",
        nargs="*",
        default=[],
        metavar="SCENARIO_ID",
        help="Explicit scenario IDs to render; empty disables rendering",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _safe_component(name: str, value: str) -> str:
    if not value or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{name} must contain only letters, digits, underscore, dot, and dash")
    return value


def _load_scenarios_from_manifest(path: str | Path) -> list[Scenario]:
    document = load_json(path)
    if not isinstance(document, dict) or document.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            f"Scenario manifest must use schema version {EVALUATION_SCHEMA_VERSION!r}"
        )
    raw_scenarios = document.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise ValueError("Scenario manifest must be a list or contain a 'scenarios' list")
    return [Scenario.from_dict(value) for value in raw_scenarios]


def _generate_scenarios(args: argparse.Namespace) -> list[Scenario]:
    if args.scenario_manifest:
        return _load_scenarios_from_manifest(args.scenario_manifest)
    scenarios: list[Scenario] = []
    for ego_raceline in args.ego_racelines:
        for opponent_raceline in args.opponent_racelines:
            for startpoint in args.startpoints:
                for interval in args.intervals:
                    opponent_index = opponent_start_index(
                        args.map_name, ego_raceline, opponent_raceline, startpoint, interval
                    )
                    for speed_scale in args.speed_scales:
                        scenarios.append(
                            Scenario(
                                map_name=args.map_name,
                                ego_raceline=ego_raceline,
                                opponent_raceline=opponent_raceline,
                                ego_start_index=startpoint,
                                opponent_start_index=opponent_index,
                                interval_index=interval,
                                opponent_speed_scale=speed_scale,
                                simulation_duration_s=args.duration,
                            )
                        )
    return scenarios


def _unique_scenarios(scenarios: Iterable[Scenario]) -> list[Scenario]:
    result = list(scenarios)
    identifiers = [scenario.scenario_id for scenario in result]
    if not result:
        raise ValueError("At least one scenario is required")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Scenario manifest contains duplicate stable IDs")
    return result


def _device_name(requested: str, workers: int) -> str:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() and workers == 1 else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "cuda" and workers != 1:
        raise ValueError("CUDA evaluation currently requires --workers 1")
    return requested


def _run_jobs(
    jobs: list[dict[str, Any]],
    *,
    workers: int,
    checkpoint: str,
    device: str,
    hidden_scale: int,
    checkpoint_sha: str,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if workers == 1:
        initialize_worker(checkpoint, device, hidden_scale, checkpoint_sha)
        return [evaluate_worker_job(job) for job in jobs]
    context = multiprocessing.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=initialize_worker,
        initargs=(checkpoint, device, hidden_scale, checkpoint_sha),
    ) as executor:
        futures = {executor.submit(evaluate_worker_job, job): job["scenario"]["scenario_id"] for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _write_aggregate_outputs(
    run_dir: Path,
    scenarios: list[Scenario],
    *,
    trace_mode: str,
    video_scenarios: set[str],
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for scenario in scenarios:
        path = run_dir / "episodes" / f"{scenario.scenario_id}.json"
        if valid_episode_file(
            path,
            scenario.scenario_id,
            trace_mode=trace_mode,
            require_video=scenario.scenario_id in video_scenarios,
        ):
            episodes.append(load_json(path))
    episodes.sort(key=lambda row: row["scenario_id"])
    summary = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        **aggregate_episodes(episodes, total_scenarios=len(scenarios)),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_csv(run_dir / "episodes.csv", episodes, EPISODE_CSV_FIELDS)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    if args.hidden_scale < 1:
        raise ValueError("--hidden-scale must be positive")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    suite_name = _safe_component("suite name", args.suite_name)
    run_id = _safe_component(
        "run ID", args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    scenarios = _unique_scenarios(_generate_scenarios(args))
    scenario_ids = {scenario.scenario_id for scenario in scenarios}
    selected_videos = set(args.video_scenarios)
    unknown_videos = sorted(selected_videos - scenario_ids)
    if unknown_videos:
        raise ValueError(f"Video selections are not in the scenario manifest: {unknown_videos}")
    device = _device_name(args.device, args.workers)
    checkpoint_sha = checkpoint_sha256(checkpoint)
    run_dir = run_directory(
        Path(args.output_root).resolve(), suite_name, checkpoint, checkpoint_sha, run_id
    )
    scenario_manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "scenarios": [scenario.to_dict() for scenario in scenarios],
    }
    config = {
        "suite_name": suite_name,
        "run_id": run_id,
        "hidden_scale": args.hidden_scale,
        "workers": args.workers,
        "device": device,
        "trace_mode": args.trace_mode,
        "video_scenarios": sorted(selected_videos),
        "actor_hz": 100,
        "simulator_hz": 100,
        "render_default": False,
        "steering_bound_rad": 0.52,
        "previous_measured_speed_timing": True,
    }
    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "model_stem": checkpoint.stem,
        },
        "config": config,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    initialize_run(run_dir, manifest, scenario_manifest, resume=args.resume)

    jobs: list[dict[str, Any]] = []
    skipped = 0
    for scenario in scenarios:
        episode_path = run_dir / "episodes" / f"{scenario.scenario_id}.json"
        if args.resume and valid_episode_file(
            episode_path,
            scenario.scenario_id,
            trace_mode=args.trace_mode,
            require_video=scenario.scenario_id in selected_videos,
        ):
            skipped += 1
            (run_dir / "errors" / f"{scenario.scenario_id}.json").unlink(missing_ok=True)
            continue
        episode_path.unlink(missing_ok=True)
        jobs.append(
            {
                "scenario": scenario.to_dict(),
                "run_dir": str(run_dir),
                "trace_mode": args.trace_mode,
                "record_video": scenario.scenario_id in selected_videos,
            }
        )
    results = _run_jobs(
        jobs,
        workers=args.workers,
        checkpoint=str(checkpoint),
        device=device,
        hidden_scale=args.hidden_scale,
        checkpoint_sha=checkpoint_sha,
    )
    summary = _write_aggregate_outputs(
        run_dir,
        scenarios,
        trace_mode=args.trace_mode,
        video_scenarios=selected_videos,
    )
    failures = [result for result in results if not result["ok"]]
    print(f"run_dir={run_dir}")
    print(
        f"completed={summary['completed_scenarios']} errors={summary['error_scenarios']} "
        f"resumed={skipped}"
    )
    if failures:
        for failure in failures:
            print(
                f"ERROR {failure['scenario_id']}: {failure['error']['error_type']}: "
                f"{failure['error']['message']}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
