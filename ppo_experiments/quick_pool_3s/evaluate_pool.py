#!/usr/bin/env python3
"""Evaluate one persisted scenario pool with the frozen persistent-CPU contract."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import End2Race  # noqa: E402
from ppo.scenarios import ScenarioSpec, evaluation_scenarios, scenario_from_dict  # noqa: E402
from utils import atomic_write_json  # noqa: E402


MODEL: End2Race | None = None
MODEL_PATH = ""
DEVICE = torch.device("cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def worker_init(model_path: str) -> None:
    global MODEL, MODEL_PATH
    torch.set_num_threads(1)
    model = End2Race(mask_prob=0.0, hidden_scale=4).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True), strict=True)
    model.eval()
    MODEL = model
    MODEL_PATH = model_path


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
            MODEL_PATH,
            None,
            "ego",
            scenario.scenario_id,
        )
        return result["episode_metrics"]
    except Exception as error:
        return {
            "scenario_id": scenario.scenario_id,
            "outcome": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }


def load_scenarios(path: Path | None, sim_duration: float | None) -> tuple[ScenarioSpec, ...]:
    if path is None:
        scenarios = evaluation_scenarios()
    else:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        rows = document["scenarios"] if isinstance(document, dict) else document
        scenarios = tuple(scenario_from_dict(row) for row in rows)
    if sim_duration is not None:
        scenarios = tuple(replace(scenario, sim_duration=sim_duration) for scenario in scenarios)
    ids = [scenario.scenario_id for scenario in scenarios]
    if not scenarios or len(ids) != len(set(ids)):
        raise ValueError("Scenario pool must be non-empty with unique scenario IDs")
    return scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario-manifest", type=Path, default=None)
    parser.add_argument("--sim-duration", type=float, default=None)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1, 8]")
    if args.sim_duration is not None and args.sim_duration <= 0.0:
        raise ValueError("sim-duration must be positive")
    scenarios = load_scenarios(args.scenario_manifest, args.sim_duration)

    print(f"EVAL_POOL_START count={len(scenarios)} workers={args.workers}", flush=True)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=worker_init,
        initargs=(str(args.model_path.resolve()),),
    ) as executor:
        futures = [executor.submit(evaluate_one, scenario) for scenario in scenarios]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"EVAL_POOL_PROGRESS {completed}/{len(futures)}", flush=True)

    rows.sort(key=lambda row: str(row["scenario_id"]))
    ids = [str(row["scenario_id"]) for row in rows]
    errors = [row for row in rows if row.get("outcome") == "error" or "error" in row]
    outcomes = Counter(str(row.get("outcome")) for row in rows if row not in errors)
    summary = {
        "collision": int(outcomes["ego_collision"]),
        "follow": int(outcomes["follow"]),
        "overtake": int(outcomes["overtake"]),
        "error": len(errors),
        "total": len(rows),
    }
    complete = len(rows) == len(scenarios) and len(ids) == len(set(ids))
    document = {
        "schema_version": 1,
        "evaluation_contract": {
            "device": "cpu",
            "persistent_spawn_workers": args.workers,
            "torch_num_threads_per_worker": 1,
            "collision_scope": "ego",
        },
        "model_path": str(args.model_path),
        "model_sha256": sha256_file(args.model_path),
        "scenario_manifest": str(args.scenario_manifest) if args.scenario_manifest else None,
        "sim_duration_override_s": args.sim_duration,
        "complete": complete,
        "summary": summary,
        "rows": rows,
    }
    atomic_write_json(args.output, document)
    elapsed = time.monotonic() - started
    print(f"EVAL_POOL_DONE seconds={elapsed:.1f} summary={summary}", flush=True)
    if not complete or errors or sum(summary[key] for key in ("collision", "follow", "overtake", "error")) != len(rows):
        raise RuntimeError(f"Evaluation integrity failure: complete={complete}, errors={len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
