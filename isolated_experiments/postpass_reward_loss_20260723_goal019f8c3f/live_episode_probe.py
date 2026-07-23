#!/usr/bin/env python3
"""Bounded CPU-only live check for the isolated post-pass reward.

This script is intentionally not a trainer.  It re-runs two saved BC tail
episodes and two nearby saved safe-overtake controls, redirects all newly
generated artifacts beneath this isolated directory, and reconciles:

1. the historical saved trace,
2. the new live trace,
3. vectorized reward replay, and
4. the transition-by-transition reward state machine.

Run only after ``validate_shadow.py --mode saved-episodes`` has completed and
all host training/evaluation/SUMO processes are idle.
"""

from __future__ import annotations

import os

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterator

import numpy as np

from shadow_contract import PostpassState, RewardConfig, postpass_reward_step
from validate_shadow import (
    HERE,
    PROJECT_ROOT,
    TrackProjector,
    _atomic_json,
    _json_ready,
    _sha256,
    _write_csv,
    evaluate_setting,
    load_tail_labels,
    load_trace,
    prepare_episode_geometry,
    require_host_process_idle,
)


DEFAULT_OUTPUT = HERE / "outputs" / "live_episode_probe"
DEFAULT_SAVED_SUMMARY = HERE / "outputs" / "saved_episode_summary.json"
DEFAULT_SELECTED_EPISODES = HERE / "outputs" / "selected_episode_results.csv"
SCENARIO_PATTERN = re.compile(
    r"evaluation-sp(?P<ordinal>\d+)-ego(?P<ego>\d+)-"
    r"raceline(?P<raceline>\d+)-v(?P<speed>[0-9.]+)"
)
EPISODE_KEY_PATTERN = re.compile(
    r"ol(?P<raceline>\d+)_e(?P<ego>\d+)_o(?P<opponent>\d+)_"
    r"s(?P<speed>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--saved-summary",
        type=Path,
        default=DEFAULT_SAVED_SUMMARY,
    )
    parser.add_argument(
        "--selected-episodes",
        type=Path,
        default=DEFAULT_SELECTED_EPISODES,
    )
    parser.add_argument(
        "--bc-results",
        type=Path,
        default=Path(
            "eval_results/end2race_Austin/multiagents/results_multi.json"
        ),
    )
    parser.add_argument(
        "--tail-labels",
        type=Path,
        default=Path(
            "analysis_results/ppo_all_experiments_20260723/"
            "collision_episode_kinematics.csv"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("pretrained/end2race.pth"),
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


@contextmanager
def exclusive_new_output(output_dir: Path) -> Iterator[None]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite a prior live probe: {output_dir}"
        )
    output_dir.mkdir(mode=0o700)
    lock = output_dir / ".live_probe.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def strict_csv_bool(value: str, *, field: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"{field} is not a strict CSV boolean: {value!r}")
    return value == "True"


def read_selected_episode_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            if raw["panel"] != "BC":
                continue
            lead = raw["first_trigger_lead_s"]
            rows.append(
                {
                    **raw,
                    "reference_primary_tail": strict_csv_bool(
                        raw["reference_primary_tail"],
                        field="reference_primary_tail",
                    ),
                    "triggered": strict_csv_bool(
                        raw["triggered"],
                        field="triggered",
                    ),
                    "preterminal_triggered": strict_csv_bool(
                        raw["preterminal_triggered"],
                        field="preterminal_triggered",
                    ),
                    "first_trigger_lead_s": (
                        None if lead == "" else float(lead)
                    ),
                }
            )
    if len(rows) != 600:
        raise ValueError(f"{path}: expected 600 BC episode rows, got {len(rows)}")
    return rows


def _scenario_parts(scenario_id: str) -> dict[str, Any]:
    match = SCENARIO_PATTERN.fullmatch(scenario_id)
    if match is None:
        raise ValueError(f"Unrecognized evaluation scenario: {scenario_id}")
    return {
        "ordinal": int(match.group("ordinal")),
        "ego": int(match.group("ego")),
        "raceline": int(match.group("raceline")),
        "speed": float(match.group("speed")),
    }


def choose_live_panel(
    selected_rows: list[dict[str, Any]],
    result_episodes: dict[str, dict[str, Any]],
    ego_waypoint_count: int,
) -> list[dict[str, Any]]:
    """Choose two captured tail cases and nearby untriggered overtakes."""

    captured = [
        row
        for row in selected_rows
        if row["reference_primary_tail"]
        and row["preterminal_triggered"]
        and row["first_trigger_lead_s"] is not None
    ]
    if not captured:
        raise RuntimeError(
            "Selected setting captured no BC primary-tail episode before terminal"
        )
    captured.sort(key=lambda row: row["first_trigger_lead_s"])
    target_indices = sorted({0, len(captured) // 2})
    targets = [captured[index] for index in target_indices]
    if len(targets) < 2 and len(captured) >= 2:
        targets.append(captured[-1])

    by_scenario = {
        str(episode["scenario_id"]): episode
        for episode in result_episodes.values()
    }
    selected_by_scenario = {
        row["scenario_id"]: row
        for row in selected_rows
    }
    controls = []
    used = {row["scenario_id"] for row in targets}
    for target in targets:
        target_parts = _scenario_parts(target["scenario_id"])
        candidates = []
        for scenario_id, episode in by_scenario.items():
            selected = selected_by_scenario[scenario_id]
            if (
                episode["outcome"] != "overtake"
                or selected["triggered"]
                or scenario_id in used
            ):
                continue
            parts = _scenario_parts(scenario_id)
            if (
                parts["raceline"] != target_parts["raceline"]
                or parts["speed"] != target_parts["speed"]
            ):
                continue
            raw_distance = abs(parts["ego"] - target_parts["ego"])
            cyclic_distance = min(
                raw_distance,
                ego_waypoint_count - raw_distance,
            )
            candidates.append(
                (
                    cyclic_distance,
                    abs(parts["ordinal"] - target_parts["ordinal"]),
                    scenario_id,
                )
            )
        if not candidates:
            raise RuntimeError(
                f"No untriggered matched overtake for {target['scenario_id']}"
            )
        control_id = min(candidates)[2]
        used.add(control_id)
        controls.append(selected_by_scenario[control_id])

    panel = []
    for role, rows in (("tail_target", targets), ("safe_overtake_control", controls)):
        for row in rows:
            scenario_id = row["scenario_id"]
            if scenario_id not in by_scenario:
                raise ValueError(f"Scenario missing from BC results: {scenario_id}")
            panel.append(
                {
                    "role": role,
                    "scenario_id": scenario_id,
                    "selected_replay": row,
                    "saved_episode": by_scenario[scenario_id],
                }
            )
    if len(panel) != 4 or len({row["scenario_id"] for row in panel}) != 4:
        raise AssertionError("Live panel must contain four unique scenarios")
    return panel


def reward_config(summary: dict[str, Any]) -> RewardConfig:
    selected = summary["selected_setting"]
    return RewardConfig(
        activation_clearance_m=selected["activation_clearance_m"],
        maximum_ttc_s=selected["maximum_ttc_s"],
        closing_deadband_mps=selected["closing_deadband_mps"],
    )


def scenario_call_parameters(
    root: Path,
    episode_key: str,
) -> dict[str, Any]:
    match = EPISODE_KEY_PATTERN.fullmatch(episode_key)
    if match is None:
        raise ValueError(f"Unrecognized episode key: {episode_key}")
    ego_index = int(match.group("ego"))
    opponent_index = int(match.group("opponent"))
    opponent_raceline = f"raceline{int(match.group('raceline'))}"
    speed = float(match.group("speed"))

    # Import after CPU/CUDA isolation variables are fixed.
    from utils import find_corresponding_waypoint, load_raceline_waypoints

    previous_cwd = Path.cwd()
    try:
        os.chdir(root)
        ego_waypoints = load_raceline_waypoints("Austin", "raceline1.csv")
        opponent_waypoints = load_raceline_waypoints(
            "Austin",
            f"{opponent_raceline}.csv",
        )
    finally:
        os.chdir(previous_cwd)
    if opponent_raceline == "raceline1":
        mapped_index = ego_index
    else:
        mapped_index = int(
            find_corresponding_waypoint(
                ego_waypoints[ego_index],
                opponent_waypoints,
            )
        )
    interval = (opponent_index - mapped_index) % len(opponent_waypoints)
    if (mapped_index + interval) % len(opponent_waypoints) != opponent_index:
        raise AssertionError("Failed to reconstruct opponent interval")
    return {
        "ego_idx": ego_index,
        "interval_idx": int(interval),
        "ego_raceline": "raceline1",
        "opp_raceline": opponent_raceline,
        "opp_speed_scale": speed,
        "expected_opp_idx": opponent_index,
    }


def sequential_reward_replay(
    trace: dict[str, np.ndarray],
    relative: np.ndarray,
    config: RewardConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    poses = np.asarray(trace["ego_pose"], dtype=np.float64)
    opponent = np.asarray(trace["opp_pose"], dtype=np.float64)
    collisions = np.asarray(trace["collisions"], dtype=bool)
    time_s = np.asarray(trace["time_s"], dtype=np.float64)
    reward = np.zeros(len(time_s), dtype=np.float64)
    trigger = np.zeros(len(time_s), dtype=bool)
    state = PostpassState()
    opponent_collision_latched = False
    for index in range(1, len(time_s)):
        opponent_collision_latched |= bool(collisions[index, 1])
        step = postpass_reward_step(
            previous_relative_progress_m=float(relative[index - 1]),
            current_relative_progress_m=float(relative[index]),
            previous_ego_pose=poses[index - 1],
            current_ego_pose=poses[index],
            current_opponent_pose=opponent[index],
            opponent_collision_latched=opponent_collision_latched,
            transition_dt_s=float(time_s[index] - time_s[index - 1]),
            config=config,
            state=state,
        )
        reward[index] = step.reward
        trigger[index] = step.triggered
    return reward, trigger, float(state.penalty_used)


def source_paths(root: Path, model_path: Path) -> list[Path]:
    paths = [
        root / "eval_multiagent.py",
        root / "model.py",
        root / "utils.py",
        root / "demonstration.py",
        root / "ppo" / "reward.py",
        root / "latticeplanner" / "lattice_config.yaml",
        root / "f1tenth_racetracks" / "Austin" / "Austin_map.yaml",
        root / "f1tenth_racetracks" / "Austin" / "Austin_map.png",
        root / "f1tenth_racetracks" / "Austin" / "raceline0.csv",
        root / "f1tenth_racetracks" / "Austin" / "raceline1.csv",
        model_path,
        HERE / "shadow_contract.py",
        HERE / "validate_shadow.py",
        Path(__file__).resolve(),
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing live-probe inputs: {missing}")
    return paths


def fingerprints(paths: list[Path]) -> dict[str, str]:
    return {str(path): _sha256(path) for path in paths}


def normalize_outcome(value: str) -> str:
    return "collision" if value == "ego_collision" else value


def run_probe(
    root: Path,
    output_dir: Path,
    summary_path: Path,
    selected_path: Path,
    results_path: Path,
    labels_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    process_guard = require_host_process_idle()
    process_guard_checks = [process_guard]
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "passed":
        raise ValueError(f"Saved-episode audit has not passed: {summary_path}")
    selected_rows = read_selected_episode_rows(selected_path)
    results_document = json.loads(results_path.read_text(encoding="utf-8"))
    result_episodes = results_document["episodes"]
    labels = load_tail_labels(labels_path, {"BC"})
    ego_waypoints = np.loadtxt(
        root / "f1tenth_racetracks" / "Austin" / "raceline1.csv",
        delimiter=";",
        comments="#",
    )
    ego_waypoint_count = len(ego_waypoints) - int(
        np.linalg.norm(ego_waypoints[-1, 1:3] - ego_waypoints[0, 1:3]) <= 1e-9
    )
    panel = choose_live_panel(
        selected_rows,
        result_episodes,
        ego_waypoint_count,
    )
    config = reward_config(summary)
    projector = TrackProjector(
        root / "f1tenth_racetracks" / "Austin" / "raceline1.csv"
    )

    tracked_sources = source_paths(root, model_path)
    hashes_before = fingerprints(tracked_sources)
    protected_inputs = [
        summary_path,
        selected_path,
        results_path,
        labels_path,
        *[
            root
            / "eval_results"
            / "end2race_Austin"
            / "multiagents"
            / "traces"
            / f"{item['saved_episode']['episode_key']}.npz"
            for item in panel
        ],
    ]
    protected_before = fingerprints(protected_inputs)

    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from model import End2Race
    import eval_multiagent as evaluator

    device = torch.device("cpu")
    model = End2Race(hidden_scale=4).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True),
        strict=True,
    )
    model.eval()

    rows = []
    saved_trace_receipts = {}
    for item in panel:
        process_guard_checks.append(require_host_process_idle())
        scenario_id = item["scenario_id"]
        episode = item["saved_episode"]
        episode_key = str(episode["episode_key"])
        call = scenario_call_parameters(root, episode_key)
        scenario_dir = output_dir / scenario_id
        scenario_dir.mkdir()
        trace_path = scenario_dir / "live_trace.npz"
        metrics_path = scenario_dir / "live_metrics.json"
        observed_keys: list[str] = []

        def isolated_paths(
            _model_path: str,
            _map_name: str,
            _noise_level: float,
            key: str | None = None,
            state_prefix: str | None = None,
        ) -> dict[str, Path]:
            del _model_path, _map_name, _noise_level
            paths = {
                "root": scenario_dir,
                "results": scenario_dir / "unused_results.json",
            }
            if key is not None:
                observed_keys.append(key)
                paths["trace"] = trace_path
                if state_prefix is not None:
                    paths["video"] = scenario_dir / "unused_video.mp4"
            return paths

        evaluator.multiagent_paths = isolated_paths
        previous_cwd = Path.cwd()
        try:
            os.chdir(root)
            result = evaluator.evaluate_segment(
                model,
                device,
                0.0,
                "Austin",
                call["ego_idx"],
                call["interval_idx"],
                call["ego_raceline"],
                call["opp_raceline"],
                call["opp_speed_scale"],
                8.0,
                False,
                True,
                str(model_path),
                str(metrics_path),
                "ego",
                scenario_id,
            )
        finally:
            os.chdir(previous_cwd)
        if observed_keys != [episode_key]:
            raise AssertionError(
                f"{scenario_id}: evaluator key mismatch {observed_keys}"
            )
        if int(result["opp_idx"]) != call["expected_opp_idx"]:
            raise AssertionError(f"{scenario_id}: opponent index mismatch")

        live_outcome = normalize_outcome(
            str(result["episode_metrics"]["outcome"])
        )
        saved_outcome = normalize_outcome(str(episode["outcome"]))
        live_trace, live_archive = load_trace(
            trace_path,
            live_outcome == "collision",
        )
        saved_trace_path = (
            root
            / "eval_results"
            / "end2race_Austin"
            / "multiagents"
            / "traces"
            / f"{episode_key}.npz"
        )
        saved_trace, saved_archive = load_trace(
            saved_trace_path,
            saved_outcome == "collision",
        )
        saved_trace_receipts[scenario_id] = saved_archive

        live_geometry = prepare_episode_geometry(live_trace, projector, config)
        saved_geometry = prepare_episode_geometry(saved_trace, projector, config)
        live_reward = evaluate_setting(
            live_geometry,
            config,
            retain_arrays=True,
        )
        saved_reward = evaluate_setting(
            saved_geometry,
            config,
            retain_arrays=True,
        )
        sequential_reward, sequential_trigger, sequential_used = (
            sequential_reward_replay(
                live_trace,
                np.asarray(live_geometry["relative"], dtype=np.float64),
                config,
            )
        )
        vector_reward = np.asarray(live_reward["reward"], dtype=np.float64)
        vector_trigger = np.asarray(live_reward["trigger"], dtype=bool)
        vector_state_machine_match = bool(
            np.array_equal(vector_trigger, sequential_trigger)
            and np.allclose(
                vector_reward,
                sequential_reward,
                atol=1e-14,
                rtol=1e-14,
            )
            and abs(
                float(live_reward["episode_penalty_used"])
                - sequential_used
            )
            <= 1e-14
        )

        same_length = len(live_trace["time_s"]) == len(saved_trace["time_s"])
        pose_max_abs_error = None
        action_max_abs_error = None
        if same_length:
            pose_max_abs_error = float(
                max(
                    np.max(
                        np.abs(
                            np.asarray(live_trace["ego_pose"], dtype=np.float64)
                            - np.asarray(saved_trace["ego_pose"], dtype=np.float64)
                        ),
                        initial=0.0,
                    ),
                    np.max(
                        np.abs(
                            np.asarray(live_trace["opp_pose"], dtype=np.float64)
                            - np.asarray(saved_trace["opp_pose"], dtype=np.float64)
                        ),
                        initial=0.0,
                    ),
                )
            )
            action_max_abs_error = float(
                np.max(
                    np.abs(
                        np.asarray(
                            live_trace["ego_raw_action"],
                            dtype=np.float64,
                        )
                        - np.asarray(
                            saved_trace["ego_raw_action"],
                            dtype=np.float64,
                        )
                    ),
                    initial=0.0,
                )
            )
        saved_live_reward_match = bool(
            live_reward["triggered"] == saved_reward["triggered"]
            and live_reward["preterminal_triggered"]
            == saved_reward["preterminal_triggered"]
            and abs(
                live_reward["penalty_sum"] - saved_reward["penalty_sum"]
            )
            <= 1e-6
        )
        label = labels.get(("BC", scenario_id))
        if item["role"] == "tail_target" and not (
            label is not None and label["merge_tail_primary"]
        ):
            raise AssertionError(f"{scenario_id}: target lost its tail label")

        checks = {
            "outcome_matches_saved": live_outcome == saved_outcome,
            "step_count_matches_saved": int(result["episode_metrics"]["steps"])
            == int(episode["steps"]),
            "selected_reward_matches_saved_replay": saved_live_reward_match,
            "vectorized_matches_step_state_machine": vector_state_machine_match,
            "tail_target_has_preterminal_signal": (
                True
                if item["role"] != "tail_target"
                else bool(live_reward["preterminal_triggered"])
            ),
            "safe_control_remains_untriggered": (
                True
                if item["role"] != "safe_overtake_control"
                else not bool(live_reward["triggered"])
            ),
        }
        rows.append(
            {
                "role": item["role"],
                "scenario_id": scenario_id,
                "episode_key": episode_key,
                "saved_outcome": saved_outcome,
                "live_outcome": live_outcome,
                "saved_steps": int(episode["steps"]),
                "live_steps": int(result["episode_metrics"]["steps"]),
                "same_trace_length": same_length,
                "pose_max_abs_error": pose_max_abs_error,
                "action_max_abs_error": action_max_abs_error,
                "saved_triggered": saved_reward["triggered"],
                "live_triggered": live_reward["triggered"],
                "saved_preterminal_triggered": saved_reward[
                    "preterminal_triggered"
                ],
                "live_preterminal_triggered": live_reward[
                    "preterminal_triggered"
                ],
                "saved_penalty_sum": saved_reward["penalty_sum"],
                "live_penalty_sum": live_reward["penalty_sum"],
                "live_trigger_steps": live_reward["trigger_steps"],
                "live_first_trigger_lead_s": live_reward[
                    "first_trigger_lead_s"
                ],
                "live_trace_size_bytes": live_archive["size_bytes"],
                **checks,
                "all_checks_pass": all(checks.values()),
            }
        )

    process_guard_checks.append(require_host_process_idle())
    hashes_after = fingerprints(tracked_sources)
    protected_after = fingerprints(protected_inputs)
    sources_unchanged = hashes_before == hashes_after
    protected_inputs_unchanged = protected_before == protected_after
    if not sources_unchanged:
        raise RuntimeError("A source file changed during the live probe")
    if not protected_inputs_unchanged:
        raise RuntimeError("A protected saved input changed during the live probe")
    if not all(row["all_checks_pass"] for row in rows):
        raise AssertionError("One or more live episode checks failed")

    _write_csv(output_dir / "live_episode_results.csv", rows)
    output = {
        "status": "passed",
        "device": "cpu",
        "thread_limit": 1,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "host_process_guard": process_guard,
        "host_process_guard_check_count": len(process_guard_checks),
        "host_process_guard_minimum_visible_processes": min(
            receipt["numeric_proc_entry_count"]
            for receipt in process_guard_checks
        ),
        "episode_count": len(rows),
        "offline_candidate_decision": summary["candidate_decision"],
        "production_integration_authorized": False,
        "selected_setting": summary["selected_setting"],
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "sources_unchanged": sources_unchanged,
        "protected_input_hashes_before": protected_before,
        "protected_input_hashes_after": protected_after,
        "protected_inputs_unchanged": protected_inputs_unchanged,
        "saved_trace_receipts": saved_trace_receipts,
        "results": rows,
        "results_csv": str(output_dir / "live_episode_results.csv"),
    }
    _atomic_json(output_dir / "live_episode_summary.json", output)
    return output


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if HERE not in output_dir.parents:
        raise ValueError(f"Output must remain beneath {HERE}: {output_dir}")
    paths = {
        "summary": resolve(root, args.saved_summary),
        "selected": resolve(root, args.selected_episodes),
        "results": resolve(root, args.bc_results),
        "labels": resolve(root, args.tail_labels),
        "model": resolve(root, args.model_path),
    }
    # Preflight before creating the one-shot output directory. ``run_probe``
    # repeats the guard after the lock is acquired to close the race window.
    require_host_process_idle()
    with exclusive_new_output(output_dir):
        receipt = run_probe(
            root,
            output_dir,
            paths["summary"],
            paths["selected"],
            paths["results"],
            paths["labels"],
            paths["model"],
        )
    print(json.dumps(_json_ready(receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
