"""Deterministically evaluate one actor on an explicit ScenarioSpec panel.

``evaluate.sh`` builds a fixed Cartesian product. This runner instead consumes an
existing JSON list of ScenarioSpec records, so a frozen panel held under
``post-trained/panels/`` can be replayed exactly.

Rebuilt 2026-07-30 per ``.agents/EXPERIMENTS.md`` §3.2 after the original tool was
cleaned up. It is needed because a panel that a decision gate depends on must stay
runnable: without it a later arm cannot compute matched removed/created against a
recorded baseline.

Contract
--------
* Reads a panel JSON; accepts a bare list of ScenarioSpec dicts, ``{"entries":
  [{"scenario": {...}}]}``, or ``{"scenarios": [...]}``.
* Verifies every record's ``opp_idx`` against ``get_opponent_startpoint`` before
  running anything, so a panel cannot be silently reinterpreted.
* One deterministic episode per scenario in a ``forkserver`` pool with worker
  threads pinned to 1.
* Writes ``results_multi.json`` (aggregate + per-episode) and
  ``eval_manifest.json`` into ``--output-dir``; when requested, writes numeric
  NPZ traces directly into ``--output-dir/traces``.
* Resumable: an existing ``episodes.partial.jsonl`` is reused and only missing
  scenarios are run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WORKER_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--map-name", default="Austin")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--hidden-scale", type=int, default=4)
    parser.add_argument("--sim-duration", type=float, default=8.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    # Frozen panels use ego scope: an opponent-only wall collision must not
    # terminate or relabel the episode, it is recorded as an event instead.
    parser.add_argument("--collision-scope", choices=("legacy", "ego"), default="ego")
    parser.add_argument("--save-traces", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_panel(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif "entries" in payload:
        rows = payload["entries"]
    elif "scenarios" in payload:
        rows = payload["scenarios"]
    else:
        raise ValueError(f"{path}: unrecognised panel schema")
    scenarios = [row["scenario"] if "scenario" in row else row for row in rows]
    required = (
        "ego_idx",
        "opp_idx",
        "opp_raceline",
        "opp_speedscale",
        "interval_idx",
        "map_name",
    )
    for scenario in scenarios:
        missing = [key for key in required if key not in scenario]
        if missing:
            raise ValueError(f"scenario missing {missing}: {scenario}")
    return scenarios


def verify_panel(scenarios: list[dict], map_name: str) -> None:
    """Fail closed if the panel's opponent index disagrees with the generator."""

    from utils import episode_key, get_opponent_startpoint

    keys: set[str] = set()
    for scenario in scenarios:
        if scenario["map_name"] != map_name:
            raise ValueError(
                f"panel map {scenario['map_name']} != --map-name {map_name}"
            )
        expected = get_opponent_startpoint(
            map_name,
            "raceline1",
            scenario["opp_raceline"],
            scenario["ego_idx"],
            scenario["interval_idx"],
        )
        if expected != scenario["opp_idx"]:
            raise ValueError(
                "panel opp_idx disagrees with generator for "
                f"ego {scenario['ego_idx']}: panel {scenario['opp_idx']} "
                f"vs computed {expected}"
            )
        key = episode_key(
            scenario["opp_raceline"],
            scenario["ego_idx"],
            scenario["opp_idx"],
            scenario["opp_speedscale"],
        )
        if key in keys:
            raise ValueError(f"duplicate episode key in panel: {key}")
        keys.add(key)


def worker_initializer() -> None:
    for name, value in WORKER_ENV.items():
        os.environ[name] = value
    import torch

    torch.set_num_threads(1)


def evaluate_one(task: dict) -> dict:
    """Run one scenario and return its episode metrics plus scenario identity."""

    import torch
    from model import End2Race
    from eval_multiagent import evaluate_segment

    scenario = task["scenario"]
    requested = task["device"]
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
        device = torch.device(requested)
    model = End2Race(hidden_scale=task["hidden_scale"]).to(device)
    model.load_state_dict(
        torch.load(task["model_path"], map_location=device, weights_only=True),
        strict=True,
    )
    model.eval()

    handle, metrics_path = tempfile.mkstemp(prefix="panel_metrics_", suffix=".json")
    os.close(handle)
    try:
        evaluate_segment(
            model,
            device,
            0.0,
            scenario["map_name"],
            scenario["ego_idx"],
            scenario["interval_idx"],
            "raceline1",
            scenario["opp_raceline"],
            scenario["opp_speedscale"],
            task["sim_duration"],
            False,
            task["save_traces"],
            task["model_path"],
            metrics_path,
            task["collision_scope"],
            task["trace_output_path"],
        )
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    finally:
        Path(metrics_path).unlink(missing_ok=True)

    metrics.update(
        {
            "scenario_id": scenario.get("scenario_id"),
            "ego_idx": scenario["ego_idx"],
            "opp_idx": scenario["opp_idx"],
            "opp_raceline": scenario["opp_raceline"],
            "opp_speedscale": scenario["opp_speedscale"],
            "interval_idx": scenario["interval_idx"],
            "map_name": scenario["map_name"],
        }
    )
    return metrics


def opponent_wall_event_episodes(trace_directory: Path, keys: list[str]) -> int | None:
    """Count episodes whose trace ever flags an opponent wall collision.

    Under ego scope such an event neither terminates nor labels the episode, so it
    is invisible in ``outcome`` and has to be read from the trace's typed marker.
    Returns ``None`` when traces are unavailable.
    """

    if not trace_directory.is_dir():
        return None
    import numpy as np

    total = 0
    for key in keys:
        path = trace_directory / f"{key}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as payload:
            if bool(np.asarray(payload["opp_wall_collision"], dtype=bool).any()):
                total += 1
    return total


def summarize(episodes: dict[str, dict], opponent_wall_events: int | None) -> dict:
    total = len(episodes)
    counts = {"overtake": 0, "follow": 0, "ego-opp": 0, "ego-wall": 0, "opp-wall": 0}
    speeds: list[float] = []
    variances: list[float] = []
    distances: list[float] = []
    for record in episodes.values():
        counts[record["outcome"]] = counts.get(record["outcome"], 0) + 1
        speeds.append(float(record["avg_speed"]))
        variances.append(float(record["speed_variance"]))
        distances.append(float(record["total_distance"]))
    ego = counts["ego-opp"] + counts["ego-wall"]
    mean = lambda values: (sum(values) / len(values)) if values else 0.0
    summary = {
        "total_episodes": total,
        "following_count": counts["follow"],
        "overtaking_count": counts["overtake"],
        "success_count": counts["follow"] + counts["overtake"],
        "collision_count": ego,
        "ego_collision_count": ego,
        "ego_opp_collision_count": counts["ego-opp"],
        "ego_wall_collision_count": counts["ego-wall"],
        "opp_wall_event_episode_count": opponent_wall_events,
        "error_count": 0,
        "following_rate": 100.0 * counts["follow"] / total if total else 0.0,
        "overtaking_rate": 100.0 * counts["overtake"] / total if total else 0.0,
        "success_rate": 100.0 * (counts["follow"] + counts["overtake"]) / total
        if total
        else 0.0,
        "collision_rate": 100.0 * ego / total if total else 0.0,
        "avg_speed_mean": mean(speeds),
        "speed_variance_mean": mean(variances),
        "total_distance_mean": mean(distances),
    }
    if counts["opp-wall"]:
        # Only reachable under legacy scope, where opp-wall terminates the episode.
        summary["opp_wall_terminated_count"] = counts["opp-wall"]
        summary["collision_count"] = ego + counts["opp-wall"]
    return summary


def main() -> None:
    args = parse_arguments()
    scenarios = load_panel(args.panel)
    verify_panel(scenarios, args.map_name)

    from utils import episode_key

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_traces:
        (args.output_dir / "traces").mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "episodes.partial.jsonl"
    done: dict[str, dict] = {}
    if partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["episode_key"]] = record
        print(f"resuming: {len(done)} episodes already complete")

    pending = []
    for scenario in scenarios:
        key = episode_key(
            scenario["opp_raceline"],
            scenario["ego_idx"],
            scenario["opp_idx"],
            scenario["opp_speedscale"],
        )
        if key in done:
            continue
        pending.append(
            {
                "scenario": scenario,
                "model_path": str(args.model_path),
                "hidden_scale": args.hidden_scale,
                "sim_duration": args.sim_duration,
                "device": args.device,
                "save_traces": args.save_traces,
                "collision_scope": args.collision_scope,
                "trace_output_path": str(args.output_dir / "traces" / f"{key}.npz") if args.save_traces else None,
            }
        )

    if pending:
        context = mp.get_context("forkserver")
        with partial_path.open("a", encoding="utf-8") as stream:
            with context.Pool(
                processes=max(1, args.workers), initializer=worker_initializer
            ) as pool:
                for index, metrics in enumerate(
                    pool.imap_unordered(evaluate_one, pending), start=1
                ):
                    done[metrics["episode_key"]] = metrics
                    stream.write(json.dumps(metrics, sort_keys=True) + "\n")
                    stream.flush()
                    if index % 50 == 0 or index == len(pending):
                        print(f"  {index}/{len(pending)} episodes")

    if len(done) != len(scenarios):
        raise RuntimeError(
            f"expected {len(scenarios)} episodes, collected {len(done)}"
        )

    trace_count = sum((args.output_dir / "traces" / f"{key}.npz").is_file() for key in done) if args.save_traces else 0

    opponent_wall_events = opponent_wall_event_episodes(
        args.output_dir / "traces", sorted(done)
    )
    results = {
        "final": summarize(done, opponent_wall_events),
        "episodes": dict(sorted(done.items())),
    }
    (args.output_dir / "results_multi.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "status": "fresh_evaluation",
        "complete": True,
        "comparison_ready": True,
        "direct_evaluator_aggregate_retained": True,
        "actor_path": str(args.model_path),
        "actor_sha256": sha256_file(args.model_path),
        "map_name": args.map_name,
        "panel_id": args.panel_id,
        "panel_file": str(args.panel),
        "panel_sha256": sha256_file(args.panel),
        "scenario_count": len(scenarios),
        "result_episode_count": len(done),
        "unique_episode_keys": len(set(done)) == len(done),
        "error_count": 0,
        "deterministic_actor": True,
        "noise": 0.0,
        "sim_duration_s": args.sim_duration,
        "collision_scope": args.collision_scope,
        "device": args.device,
        "hidden_scale": args.hidden_scale,
        "save_traces": bool(args.save_traces),
        "trace_count": trace_count,
        "trace_result_key_sets_equal": (not args.save_traces)
        or trace_count == len(done),
        "panel_opp_idx_verified_against_generator": True,
        "retention_note": (
            "Fresh deterministic evaluation, not a trace reconstruction. The panel "
            "scenario set was verified against get_opponent_startpoint before running."
        ),
    }
    (args.output_dir / "eval_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    partial_path.unlink(missing_ok=True)
    print(json.dumps(results["final"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
