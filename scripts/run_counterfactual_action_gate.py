import argparse
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shlex
import subprocess
import sys
import warnings

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from f110_gym.envs.base_classes import Integrator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demonstration import setup_opp_planner
from eval_multiagent import classify_collision
from latticeplanner.utils import obsDict2oppoArray
from model import End2Race
from ppo.reward import ProgressProjector
from scripts.run_bc_anchor_gate_b import obb_clearance_series, sha256_file
from utils import atomic_write_json, calculate_metrics, episode_key, evaluate_proximity_quality, load_positions_and_speeds_from_params, save_numeric_npz, wrapped_progress_difference

WORKER_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
PREFIXES = {
    "early": {"start_before_event_steps": 150, "duration_steps": 50},
    "late": {"start_before_event_steps": 100, "duration_steps": 50},
}
CANDIDATES = (
    {"name": "steer_m0p04", "steering_delta_rad": -0.04, "speed_delta_mps": 0.0, "family": "steering"},
    {"name": "steer_m0p02", "steering_delta_rad": -0.02, "speed_delta_mps": 0.0, "family": "steering"},
    {"name": "steer_p0p02", "steering_delta_rad": 0.02, "speed_delta_mps": 0.0, "family": "steering"},
    {"name": "steer_p0p04", "steering_delta_rad": 0.04, "speed_delta_mps": 0.0, "family": "steering"},
    {"name": "speed_m1p0", "steering_delta_rad": 0.0, "speed_delta_mps": -1.0, "family": "speed"},
    {"name": "speed_m0p5", "steering_delta_rad": 0.0, "speed_delta_mps": -0.5, "family": "speed"},
    {"name": "speed_p0p5", "steering_delta_rad": 0.0, "speed_delta_mps": 0.5, "family": "speed"},
    {"name": "speed_p1p0", "steering_delta_rad": 0.0, "speed_delta_mps": 1.0, "family": "speed"},
    {"name": "steer_m0p02_speed_m0p5", "steering_delta_rad": -0.02, "speed_delta_mps": -0.5, "family": "coordinated"},
    {"name": "steer_p0p02_speed_m0p5", "steering_delta_rad": 0.02, "speed_delta_mps": -0.5, "family": "coordinated"},
    {"name": "steer_m0p02_speed_p0p5", "steering_delta_rad": -0.02, "speed_delta_mps": 0.5, "family": "coordinated"},
    {"name": "steer_p0p02_speed_p0p5", "steering_delta_rad": 0.02, "speed_delta_mps": 0.5, "family": "coordinated"},
)
EXACT_FIELDS = ("ego_raw_action", "ego_executed_action")
TOLERANCE_FIELDS = ("opp_executed_action", "ego_pose", "opp_pose", "ego_measured_speed_mps", "opp_measured_speed_mps", "ego_lidar_360", "opp_lidar_360")
BOOLEAN_FIELDS = ("collisions", "ego_opp_collision", "ego_wall_collision", "opp_wall_collision", "action_applied", "terminal_post_step")
COMPACT_TRACE_DTYPES = {
    "time_s": np.float64,
    "ego_raw_action": np.float32,
    "ego_executed_action": np.float32,
    "u44_raw_action": np.float32,
    "collisions": np.bool_,
    "ego_opp_collision": np.bool_,
    "ego_wall_collision": np.bool_,
    "opp_wall_collision": np.bool_,
    "action_applied": np.bool_,
    "terminal_post_step": np.bool_,
    "intervention_active": np.bool_,
}
WORKER_DEVICE = None
WORKER_U44 = None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-panel", type=Path, required=True)
    parser.add_argument("--bc-results", type=Path, required=True)
    parser.add_argument("--u44-results", type=Path, required=True)
    parser.add_argument("--u44-trace-root", type=Path, required=True)
    parser.add_argument("--u44-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--hidden-scale", type=int, default=4)
    parser.add_argument("--sim-duration", type=float, default=8.0)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def scenario_key(scenario):
    return episode_key(scenario["opp_raceline"], scenario["ego_idx"], scenario["opp_idx"], scenario["opp_speedscale"])


def classify_stratum(bc_outcome, u44_outcome):
    collision_outcomes = ("ego-opp", "ego-wall")
    if u44_outcome in collision_outcomes:
        return "inherited_collision" if bc_outcome in collision_outcomes else "created_collision"
    if u44_outcome == "follow" and bc_outcome == "overtake":
        return "lost_overtake"
    if u44_outcome == "follow" and bc_outcome == "follow":
        return "inherited_follow"
    if u44_outcome == "overtake" and bc_outcome == "overtake":
        return "safe_control"
    return None


def source_event(trace_path, stratum):
    clearances, action_applied, ego_opp, ego_wall = obb_clearance_series(trace_path)
    if stratum.endswith("collision"):
        collision_indices = np.flatnonzero(ego_opp | ego_wall)
        if len(collision_indices) == 0:
            raise RuntimeError(f"collision stratum has no collision marker: {trace_path}")
        event_index = int(collision_indices[0])
        event_name = "first_ego_collision"
    else:
        event_index = int(np.argmin(clearances))
        event_name = "first_global_minimum_obb_clearance"
    return {
        "event_name": event_name,
        "event_index": event_index,
        "event_time_s": event_index * 0.01,
        "minimum_obb_clearance_m": float(np.min(clearances)),
        "source_action_steps": int(action_applied.sum()),
        "eligible": event_index >= 150,
    }


def fold_for_startpoint(ego_idx):
    digest = hashlib.sha256(f"counterfactual-action-fold-v1|{int(ego_idx)}".encode("utf-8")).hexdigest()
    return int(digest, 16) % 5


def build_plan(args):
    panel = json.loads(args.development_panel.read_text(encoding="utf-8"))
    bc_results = json.loads(args.bc_results.read_text(encoding="utf-8"))["episodes"]
    u44_results = json.loads(args.u44_results.read_text(encoding="utf-8"))["episodes"]
    scenarios = {scenario_key(scenario): scenario for scenario in panel}
    if len(scenarios) != len(panel) or set(bc_results) != set(scenarios) or set(u44_results) != set(scenarios):
        raise RuntimeError("development panel and result key sets do not match")
    tasks = []
    excluded = []
    for key in sorted(scenarios):
        stratum = classify_stratum(bc_results[key]["outcome"], u44_results[key]["outcome"])
        if stratum is None:
            continue
        event = source_event(args.u44_trace_root / f"{key}.npz", stratum)
        task = {
            "episode_key": key,
            "stratum": stratum,
            "source_outcome": u44_results[key]["outcome"],
            "bc_outcome": bc_results[key]["outcome"],
            "fold": fold_for_startpoint(scenarios[key]["ego_idx"]),
            "scenario": scenarios[key],
            "event": event,
            "prefixes": {
                name: {
                    "start_index": event["event_index"] - contract["start_before_event_steps"],
                    "end_index_exclusive": event["event_index"] - contract["start_before_event_steps"] + contract["duration_steps"],
                    **contract,
                }
                for name, contract in PREFIXES.items()
            },
        }
        if event["eligible"]:
            tasks.append(task)
        else:
            excluded.append(task)
    counts = {}
    for task in tasks:
        counts[task["stratum"]] = counts.get(task["stratum"], 0) + 1
    expected = {
        "inherited_collision": 109,
        "created_collision": 46,
        "lost_overtake": 13,
        "inherited_follow": 63,
        "safe_control": 225,
    }
    if counts != expected:
        raise RuntimeError(f"frozen cohort counts changed: expected {expected}, got {counts}")
    fold_counts = {}
    for task in tasks:
        fold = str(task["fold"])
        fold_counts.setdefault(fold, {})
        fold_counts[fold][task["stratum"]] = fold_counts[fold].get(task["stratum"], 0) + 1
    return {
        "schema_version": 1,
        "experiment_id": "counterfactual_first_action_preference",
        "gate": "action_existence_and_rankability",
        "status": "frozen_before_candidate_branches",
        "inputs": {
            "development_panel_sha256": sha256_file(args.development_panel),
            "bc_results_sha256": sha256_file(args.bc_results),
            "u44_results_sha256": sha256_file(args.u44_results),
            "u44_model_path": str(args.u44_model_path),
            "u44_model_sha256": sha256_file(args.u44_model_path),
        },
        "cohort_contract": {
            "map": "Austin",
            "source": "frozen Gate A development startpoints only",
            "minimum_pre_event_steps": 150,
            "strata": list(expected),
            "counts": counts,
            "excluded_short_prefix_count": len(excluded),
            "fold_assignment": "SHA256(counterfactual-action-fold-v1|ego_idx) mod 5",
            "fold_counts": fold_counts,
        },
        "prefix_contract": PREFIXES,
        "candidate_contract": {
            "reference": "residual relative to the current U44 deterministic mean action",
            "steering_clip_rad": [-0.52, 0.52],
            "candidate_count": len(CANDIDATES),
            "candidates": list(CANDIDATES),
        },
        "admission_contract": {
            "existence": {
                "inherited_collision_overtake_rescue_min": 22,
                "created_collision_overtake_rescue_min": 12,
                "lost_overtake_restore_min": 7,
                "collision_rescue_unique_startpoints_min": 10,
                "collision_rescue_required_racelines": ["raceline0", "raceline1", "raceline2"],
                "rescue_action_families_min": 2,
            },
            "rankability": {
                "inherited_collision_overtake_rescue_min": 11,
                "created_collision_overtake_rescue_min": 7,
                "lost_overtake_restore_min": 4,
                "safe_control_new_collision_max": 4,
                "safe_control_overtake_loss_max": 11,
                "state_conditioned_success_margin_over_grouped_fixed_baseline_min": 9,
            },
        },
        "tasks": tasks,
        "excluded_short_prefix_tasks": excluded,
    }


def verify_plan(args, plan):
    expected = {
        "development_panel_sha256": sha256_file(args.development_panel),
        "bc_results_sha256": sha256_file(args.bc_results),
        "u44_results_sha256": sha256_file(args.u44_results),
        "u44_model_sha256": sha256_file(args.u44_model_path),
    }
    for key, value in expected.items():
        if plan["inputs"].get(key) != value:
            raise RuntimeError(f"frozen action gate input changed: {key}")
    if len(plan["tasks"]) != 456 or len({task["episode_key"] for task in plan["tasks"]}) != 456:
        raise RuntimeError("frozen action gate task identity contract failed")
    if plan["candidate_contract"]["candidates"] != list(CANDIDATES) or plan["prefix_contract"] != PREFIXES:
        raise RuntimeError("frozen action library or prefix contract changed")


def worker_initializer(u44_model_path, hidden_scale):
    global WORKER_DEVICE, WORKER_U44
    for name, value in WORKER_ENV.items():
        os.environ[name] = value
    warnings.filterwarnings("ignore", message="Chosen integrator is RK4.*")
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the counterfactual action gate")
    WORKER_DEVICE = torch.device("cuda")
    WORKER_U44 = End2Race(hidden_scale=hidden_scale).to(WORKER_DEVICE)
    WORKER_U44.load_state_dict(torch.load(u44_model_path, map_location=WORKER_DEVICE, weights_only=True), strict=True)
    WORKER_U44.eval()


def append_full_row(trace, lap_time, lidar, opp_lidar, raw_action, executed_action, opp_action, obs, collisions, markers, action_applied, terminal):
    trace["time_s"].append(float(lap_time))
    trace["ego_lidar_360"].append(lidar)
    trace["opp_lidar_360"].append(opp_lidar)
    trace["ego_raw_action"].append(raw_action)
    trace["ego_executed_action"].append(executed_action)
    trace["opp_executed_action"].append(opp_action)
    trace["ego_measured_speed_mps"].append(float(obs["linear_vels_x"][0]))
    trace["opp_measured_speed_mps"].append(float(obs["linear_vels_x"][1]))
    trace["ego_pose"].append([obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]])
    trace["opp_pose"].append([obs["poses_x"][1], obs["poses_y"][1], obs["poses_theta"][1]])
    trace["collisions"].append(collisions)
    trace["ego_opp_collision"].append(markers[0])
    trace["ego_wall_collision"].append(markers[1])
    trace["opp_wall_collision"].append(markers[2])
    trace["action_applied"].append(action_applied)
    trace["terminal_post_step"].append(terminal)


def append_compact_row(trace, lap_time, raw_action, executed_action, u44_raw, collisions, markers, action_applied, terminal, intervention_active):
    trace["time_s"].append(float(lap_time))
    trace["ego_raw_action"].append(raw_action)
    trace["ego_executed_action"].append(executed_action)
    trace["u44_raw_action"].append(u44_raw)
    trace["collisions"].append(collisions)
    trace["ego_opp_collision"].append(markers[0])
    trace["ego_wall_collision"].append(markers[1])
    trace["opp_wall_collision"].append(markers[2])
    trace["action_applied"].append(action_applied)
    trace["terminal_post_step"].append(terminal)
    trace["intervention_active"].append(intervention_active)


def compare_replay(source_path, generated):
    with np.load(source_path, allow_pickle=False) as payload:
        source = {field: np.asarray(payload[field]) for field in payload.files}
    errors = {}
    passed = True
    for field in EXACT_FIELDS + TOLERANCE_FIELDS:
        if source[field].shape != generated[field].shape:
            error = float("inf")
        else:
            error = float(np.max(np.abs(source[field].astype(np.float64) - generated[field].astype(np.float64))))
        errors[field] = error
        passed = passed and (error == 0.0 if field in EXACT_FIELDS else error <= 1e-6)
    for field in BOOLEAN_FIELDS:
        equal = source[field].shape == generated[field].shape and np.array_equal(source[field], generated[field])
        errors[field] = 0.0 if equal else float("inf")
        passed = passed and equal
    return passed, errors


def evaluate_branch(task):
    scenario = task["scenario"]
    candidate = task.get("candidate")
    prefix_name = task.get("prefix_name")
    prefix = task["prefixes"].get(prefix_name) if prefix_name is not None else None
    np.random.seed(42)
    params = {"ego_raceline": "raceline1", "opp_raceline": scenario["opp_raceline"], "ego_idx": scenario["ego_idx"], "opp_idx": scenario["opp_idx"]}
    positions, initial_speeds = load_positions_and_speeds_from_params(params, "Austin")
    env = gym.make("f110-v0", map="f1tenth_racetracks/Austin/Austin_map", map_ext=".png", num_agents=2, timestep=0.01, integrator=Integrator.RK4)
    opponent = setup_opp_planner("Austin", scenario["opp_raceline"])
    progress_projector = ProgressProjector.from_csv("f1tenth_racetracks/Austin/raceline1.csv")
    centerline_total_length = progress_projector.track_length
    obs, _, done, _ = env.reset(poses=positions)
    initial_ego_progress = progress_projector.project(np.array([obs["poses_x"][0], obs["poses_y"][0]]))
    initial_opp_progress = progress_projector.project(np.array([obs["poses_x"][1], obs["poses_y"][1]]))
    initial_relative = wrapped_progress_difference(initial_ego_progress, initial_opp_progress, centerline_total_length)
    previous_relative = initial_relative
    relative_unwrapped = initial_relative
    hidden = torch.zeros((1, 1, WORKER_U44.gru.hidden_size), device=WORKER_DEVICE)
    previous_speed = initial_speeds[0] * 0.9
    lap_time = 0.0
    step_count = 0
    tracker_count = 0
    opponent_trajectory = None
    collision_type = None
    ego_collision_time_s = None
    ego_collision_step = None
    observation_finite = True
    action_finite = True
    ego_trajectory = []
    speeds = []
    raw_lidar_history = []
    intervention_steps = 0
    hidden_snapshots = {}
    full_trace = {name: [] for name in ("time_s", "ego_lidar_360", "opp_lidar_360", "ego_raw_action", "ego_executed_action", "opp_executed_action", "ego_measured_speed_mps", "opp_measured_speed_mps", "ego_pose", "opp_pose", "collisions", "ego_opp_collision", "ego_wall_collision", "opp_wall_collision", "action_applied", "terminal_post_step")}
    compact_trace = {name: [] for name in ("time_s", "ego_raw_action", "ego_executed_action", "u44_raw_action", "collisions", "ego_opp_collision", "ego_wall_collision", "opp_wall_collision", "action_applied", "terminal_post_step", "intervention_active")}

    while not done and lap_time < task["sim_duration"]:
        raw_lidar = np.asarray(obs["scans"][0]).reshape(-1)
        if len(raw_lidar) > 360:
            indices = np.linspace(0, len(raw_lidar) - 1, 360, dtype=int)
            raw_lidar = raw_lidar[indices]
        lidar = raw_lidar.copy()
        lidar_tensor = torch.tensor(lidar, dtype=torch.float32, device=WORKER_DEVICE).unsqueeze(0).unsqueeze(0)
        speed_tensor = torch.tensor([[[previous_speed]]], dtype=torch.float32, device=WORKER_DEVICE)
        with torch.no_grad():
            sequence, hidden = WORKER_U44(lidar_tensor, speed_tensor, hidden)
        u44_raw = np.asarray((sequence[0, -1, 0].item(), sequence[0, -1, 1].item()), dtype=np.float32)
        if candidate is None:
            for name, contract in task["prefixes"].items():
                if step_count == contract["start_index"]:
                    hidden_snapshots[name] = hidden[0, 0].detach().cpu().numpy().astype(np.float32)
        intervention_active = candidate is not None and prefix["start_index"] <= step_count < prefix["end_index_exclusive"]
        if intervention_active:
            selected_raw = np.asarray((u44_raw[0] + candidate["steering_delta_rad"], u44_raw[1] + candidate["speed_delta_mps"]), dtype=np.float32)
        else:
            selected_raw = u44_raw.copy()
        selected_executed = np.asarray((np.clip(float(selected_raw[0]), -0.52, 0.52), float(selected_raw[1])), dtype=np.float32)
        previous_speed = obs["linear_vels_x"][0]
        if tracker_count == 0:
            opponent_poses = obsDict2oppoArray(obs, 1)
            opponent_trajectory = opponent.plan(obs["poses_x"][1], obs["poses_y"][1], obs["poses_theta"][1], opponent_poses, obs["linear_vels_x"][1])
        opponent_steer, opponent_speed = opponent.tracker.plan(obs["poses_x"][1], obs["poses_y"][1], obs["poses_theta"][1], obs["linear_vels_x"][1], opponent_trajectory)
        opponent_steer = np.clip(opponent_steer, -0.52, 0.52)
        opponent_speed *= scenario["opp_speedscale"]
        opponent_action = np.asarray((opponent_steer, opponent_speed), dtype=np.float32)
        opp_lidar = np.asarray(obs["scans"][1]).reshape(-1)
        if len(opp_lidar) > 360:
            indices = np.linspace(0, len(opp_lidar) - 1, 360, dtype=int)
            opp_lidar = opp_lidar[indices]
        current_collisions = np.asarray(obs["collisions"], dtype=bool)
        current_markers = classify_collision(env, current_collisions)
        if candidate is None:
            append_full_row(full_trace, lap_time, lidar, opp_lidar, selected_raw, selected_executed, opponent_action, obs, current_collisions, current_markers, True, False)
        append_compact_row(compact_trace, lap_time, selected_raw, selected_executed, u44_raw, current_collisions, current_markers, True, False, intervention_active)
        raw_lidar_history.append(raw_lidar.copy())
        action_finite = action_finite and bool(np.isfinite((selected_executed, opponent_action, u44_raw)).all())
        intervention_steps += int(intervention_active)
        action = np.array([[selected_executed[0], selected_executed[1]], [opponent_steer, opponent_speed]], dtype=np.float64)
        obs, timestep, done, _ = env.step(action)
        lap_time += timestep
        step_count += 1
        observation_finite = observation_finite and bool(all(np.isfinite(np.asarray(value)).all() for value in obs.values() if isinstance(value, (list, tuple, np.ndarray))))
        ego_trajectory.append([obs["poses_x"][0], obs["poses_y"][0]])
        speeds.append(obs["linear_vels_x"][0])
        ego_progress = progress_projector.project(np.array([obs["poses_x"][0], obs["poses_y"][0]]))
        opp_progress = progress_projector.project(np.array([obs["poses_x"][1], obs["poses_y"][1]]))
        current_relative = wrapped_progress_difference(ego_progress, opp_progress, centerline_total_length)
        relative_unwrapped += wrapped_progress_difference(current_relative, previous_relative, centerline_total_length)
        previous_relative = current_relative
        step_collisions = np.asarray(obs["collisions"], dtype=bool)
        step_markers = classify_collision(env, step_collisions)
        if bool(step_collisions[0]):
            if ego_collision_time_s is None:
                ego_collision_time_s = float(lap_time)
                ego_collision_step = int(step_count)
            collision_type = "ego-opp" if step_markers[0] else "ego-wall"
            done = True
        if done or lap_time >= task["sim_duration"]:
            terminal_lidar = np.asarray(obs["scans"][0]).reshape(-1)
            terminal_opp_lidar = np.asarray(obs["scans"][1]).reshape(-1)
            if len(terminal_lidar) > 360:
                indices = np.linspace(0, len(terminal_lidar) - 1, 360, dtype=int)
                terminal_lidar = terminal_lidar[indices]
            if len(terminal_opp_lidar) > 360:
                indices = np.linspace(0, len(terminal_opp_lidar) - 1, 360, dtype=int)
                terminal_opp_lidar = terminal_opp_lidar[indices]
            zeros = np.zeros(2, dtype=np.float32)
            if candidate is None:
                append_full_row(full_trace, lap_time, terminal_lidar, terminal_opp_lidar, zeros, zeros, zeros, obs, step_collisions, step_markers, False, True)
            append_compact_row(compact_trace, lap_time, zeros, zeros, zeros, step_collisions, step_markers, False, True, False)
        tracker_count = (tracker_count + 1) % 10

    env.close()
    gc.collect()
    compact_arrays = {name: np.asarray(values, dtype=COMPACT_TRACE_DTYPES[name]) for name, values in compact_trace.items()}
    save_numeric_npz(Path(task["trace_path"]), compact_arrays)
    replay_pass = None
    replay_errors = None
    if candidate is None:
        full_arrays = {name: np.asarray(values, dtype=np.float64 if name in ("time_s", "ego_pose", "opp_pose") else np.bool_ if name in BOOLEAN_FIELDS else np.float32) for name, values in full_trace.items()}
        replay_pass, replay_errors = compare_replay(Path(task["source_trace_path"]), full_arrays)
        hidden_path = Path(task["hidden_path"])
        if set(hidden_snapshots) != set(PREFIXES):
            raise RuntimeError(f"missing hidden snapshots for {task['episode_key']}")
        save_numeric_npz(hidden_path, hidden_snapshots)
    avg_speed, speed_variance, total_distance = calculate_metrics(ego_trajectory, speeds)
    outcome = collision_type if collision_type is not None else "overtake" if relative_unwrapped > 0.0 else "follow"
    proximity = evaluate_proximity_quality(np.asarray(raw_lidar_history, dtype=np.float64))
    return {
        "result_key": task["result_key"],
        "episode_key": task["episode_key"],
        "stratum": task["stratum"],
        "prefix_name": prefix_name,
        "candidate_name": candidate["name"] if candidate is not None else "noop",
        "candidate_family": candidate["family"] if candidate is not None else "noop",
        "outcome": outcome,
        "ego_collision_time_s": ego_collision_time_s,
        "ego_collision_step": ego_collision_step,
        "simulation_time_s": float(lap_time),
        "steps": int(step_count),
        "intervention_steps": int(intervention_steps),
        "observation_finite": observation_finite,
        "action_finite": action_finite,
        "avg_speed": float(avg_speed),
        "speed_variance": float(speed_variance),
        "total_distance": float(total_distance),
        "final_relative_position_m": float(relative_unwrapped),
        "replay_pass": replay_pass,
        "replay_errors": replay_errors,
        **proximity,
    }


def run_tasks(pool, tasks, partial_path, result_path):
    expected = {task["result_key"] for task in tasks}
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if set(result["episodes"]) != expected:
            raise RuntimeError(f"completed result keys do not match plan: {result_path}")
        return result
    completed = {}
    if partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                completed[record["result_key"]] = record
        print(f"resuming {len(completed)}/{len(tasks)} from {partial_path.name}")
    pending = [task for task in tasks if task["result_key"] not in completed]
    if pending:
        with partial_path.open("a", encoding="utf-8") as stream:
            for index, record in enumerate(pool.imap_unordered(evaluate_branch, pending), start=1):
                completed[record["result_key"]] = record
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                if index % 50 == 0 or index == len(pending):
                    print(f"{result_path.stem}: {index}/{len(pending)} pending branches", flush=True)
    if set(completed) != expected:
        raise RuntimeError(f"result key set does not match plan: {result_path}")
    result = {"summary": {"episode_count": len(completed), "error_count": 0}, "episodes": dict(sorted(completed.items()))}
    atomic_write_json(result_path, result)
    partial_path.unlink(missing_ok=True)
    return result


def make_branch0_tasks(plan, args):
    trace_root = args.output_dir / "branch0" / "traces"
    hidden_root = args.output_dir / "branch0" / "hidden"
    trace_root.mkdir(parents=True, exist_ok=True)
    hidden_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for plan_task in plan["tasks"]:
        task = dict(plan_task)
        task["result_key"] = task["episode_key"]
        task["candidate"] = None
        task["prefix_name"] = None
        task["sim_duration"] = args.sim_duration
        task["trace_path"] = str(trace_root / f"{task['episode_key']}.npz")
        task["hidden_path"] = str(hidden_root / f"{task['episode_key']}.npz")
        task["source_trace_path"] = str(args.u44_trace_root / f"{task['episode_key']}.npz")
        tasks.append(task)
    return tasks


def make_candidate_tasks(plan, args):
    tasks = []
    trace_root = args.output_dir / "candidate_traces"
    for plan_task in plan["tasks"]:
        for prefix_name in PREFIXES:
            for candidate in CANDIDATES:
                task = dict(plan_task)
                result_key = f"{task['episode_key']}::{prefix_name}::{candidate['name']}"
                path = trace_root / prefix_name / candidate["name"] / f"{task['episode_key']}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                task["result_key"] = result_key
                task["candidate"] = candidate
                task["prefix_name"] = prefix_name
                task["sim_duration"] = args.sim_duration
                task["trace_path"] = str(path)
                tasks.append(task)
    return tasks


def validate_results(plan, branch0, candidates, args):
    task_keys = {task["episode_key"] for task in plan["tasks"]}
    if set(branch0["episodes"]) != task_keys:
        raise RuntimeError("branch0 result keys do not match plan")
    maxima = {field: 0.0 for field in EXACT_FIELDS + TOLERANCE_FIELDS + BOOLEAN_FIELDS}
    for key, record in branch0["episodes"].items():
        if not record["replay_pass"] or record["outcome"] != next(task for task in plan["tasks"] if task["episode_key"] == key)["source_outcome"]:
            raise RuntimeError(f"branch0 replay failed: {key}")
        for field, error in record["replay_errors"].items():
            maxima[field] = max(maxima[field], error)
    expected_candidate_keys = {f"{key}::{prefix}::{candidate['name']}" for key in task_keys for prefix in PREFIXES for candidate in CANDIDATES}
    if set(candidates["episodes"]) != expected_candidate_keys:
        raise RuntimeError("candidate result keys do not match plan")
    trace_count = 0
    row_count = 0
    for record in list(branch0["episodes"].values()) + list(candidates["episodes"].values()):
        if not record["observation_finite"] or not record["action_finite"]:
            raise RuntimeError(f"non-finite branch: {record['result_key']}")
        if record["candidate_name"] == "noop":
            trace_path = args.output_dir / "branch0" / "traces" / f"{record['episode_key']}.npz"
        else:
            trace_path = args.output_dir / "candidate_traces" / record["prefix_name"] / record["candidate_name"] / f"{record['episode_key']}.npz"
        with np.load(trace_path, allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
        required = set(COMPACT_TRACE_DTYPES)
        if set(arrays) != required or len({len(value) for value in arrays.values()}) != 1:
            raise RuntimeError(f"compact trace schema failed: {record['result_key']}")
        length = len(arrays["time_s"])
        if length != record["steps"] + 1 or not all(bool(np.isfinite(value).all()) for value in arrays.values()):
            raise RuntimeError(f"compact trace length/finite failed: {record['result_key']}")
        terminal = arrays["terminal_post_step"].astype(bool)
        applied = arrays["action_applied"].astype(bool)
        active = arrays["intervention_active"].astype(bool)
        if int(terminal.sum()) != 1 or not bool(terminal[-1]) or int((~applied).sum()) != 1 or bool(applied[-1]) or bool(active[-1]):
            raise RuntimeError(f"terminal contract failed: {record['result_key']}")
        expected_active = np.zeros(length, dtype=bool)
        if record["candidate_name"] != "noop":
            task = next(task for task in plan["tasks"] if task["episode_key"] == record["episode_key"])
            prefix = task["prefixes"][record["prefix_name"]]
            expected_active[prefix["start_index"]:min(prefix["end_index_exclusive"], record["steps"])] = True
            candidate = next(item for item in CANDIDATES if item["name"] == record["candidate_name"])
            expected_raw = arrays["u44_raw_action"].copy()
            expected_raw[expected_active, 0] += candidate["steering_delta_rad"]
            expected_raw[expected_active, 1] += candidate["speed_delta_mps"]
            if not np.allclose(arrays["ego_raw_action"][:-1], expected_raw[:-1], rtol=0.0, atol=1e-7):
                raise RuntimeError(f"candidate residual contract failed: {record['result_key']}")
        if not np.array_equal(active, expected_active) or int(active.sum()) != record["intervention_steps"]:
            raise RuntimeError(f"intervention interval failed: {record['result_key']}")
        expected_executed = arrays["ego_raw_action"].copy()
        expected_executed[:, 0] = np.clip(expected_executed[:, 0], -0.52, 0.52)
        expected_executed[-1] = 0.0
        if not np.array_equal(arrays["ego_executed_action"], expected_executed):
            raise RuntimeError(f"executed action contract failed: {record['result_key']}")
        trace_count += 1
        row_count += length
    return {
        "branch0_episode_count": len(branch0["episodes"]),
        "candidate_episode_count": len(candidates["episodes"]),
        "trace_count": trace_count,
        "trace_row_count": row_count,
        "branch0_field_max_abs_error": maxima,
        "branch0_exact_replay": True,
        "result_plan_trace_key_sets_equal": True,
        "all_arrays_aligned_and_finite": True,
        "terminal_contract_complete": True,
        "intervention_and_action_contract_complete": True,
    }


def action_norm(candidate):
    return (candidate["steering_delta_rad"] / 0.02) ** 2 + (candidate["speed_delta_mps"] / 0.5) ** 2


def preferred_candidate(task, prefix_name, candidate_results):
    if task["stratum"] == "safe_control":
        return "noop"
    successful = []
    for candidate in CANDIDATES:
        key = f"{task['episode_key']}::{prefix_name}::{candidate['name']}"
        record = candidate_results[key]
        if record["outcome"] == "overtake":
            successful.append((action_norm(candidate), -record["final_relative_position_m"], candidate["name"]))
    return min(successful)[2] if successful else "noop"


def existence_metrics(plan, candidate_results, prefix_name):
    rescued = {name: [] for name in ("inherited_collision", "created_collision", "lost_overtake", "inherited_follow")}
    families = set()
    for task in plan["tasks"]:
        if task["stratum"] == "safe_control":
            continue
        preferred = preferred_candidate(task, prefix_name, candidate_results)
        if preferred != "noop":
            rescued[task["stratum"]].append(task["episode_key"])
            families.add(next(candidate["family"] for candidate in CANDIDATES if candidate["name"] == preferred))
    collision_keys = rescued["inherited_collision"] + rescued["created_collision"]
    task_by_key = {task["episode_key"]: task for task in plan["tasks"]}
    criteria = plan["admission_contract"]["existence"]
    checks = {
        "inherited_collision_overtake_rescue": len(rescued["inherited_collision"]) >= criteria["inherited_collision_overtake_rescue_min"],
        "created_collision_overtake_rescue": len(rescued["created_collision"]) >= criteria["created_collision_overtake_rescue_min"],
        "lost_overtake_restore": len(rescued["lost_overtake"]) >= criteria["lost_overtake_restore_min"],
        "collision_rescue_startpoint_coverage": len({task_by_key[key]["scenario"]["ego_idx"] for key in collision_keys}) >= criteria["collision_rescue_unique_startpoints_min"],
        "collision_rescue_raceline_coverage": {task_by_key[key]["scenario"]["opp_raceline"] for key in collision_keys} == set(criteria["collision_rescue_required_racelines"]),
        "action_family_coverage": len(families) >= criteria["rescue_action_families_min"],
    }
    return {
        "verdict": "pass" if all(checks.values()) else "fail",
        "rescued_counts": {name: len(keys) for name, keys in rescued.items()},
        "rescued_scenario_keys": rescued,
        "collision_rescue_unique_startpoints": len({task_by_key[key]["scenario"]["ego_idx"] for key in collision_keys}),
        "collision_rescue_racelines": sorted({task_by_key[key]["scenario"]["opp_raceline"] for key in collision_keys}),
        "preferred_action_families": sorted(families),
        "criteria": checks,
    }


class ActionScorer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU())
        self.score_layer = nn.Sequential(nn.Linear(130, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, hidden, actions):
        encoded = self.hidden_layer(hidden)
        expanded = encoded.unsqueeze(1).expand(-1, actions.shape[0], -1)
        action_batch = actions.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        return self.score_layer(torch.cat((expanded, action_batch), dim=2)).squeeze(-1)


def candidate_action_tensor(device):
    actions = [[0.0, 0.0]]
    for candidate in CANDIDATES:
        actions.append([candidate["steering_delta_rad"] / 0.02, candidate["speed_delta_mps"] / 0.5])
    return torch.tensor(actions, dtype=torch.float32, device=device)


def selected_outcome(task, prefix_name, action_name, branch0_results, candidate_results):
    if action_name == "noop":
        return branch0_results[task["episode_key"]]
    return candidate_results[f"{task['episode_key']}::{prefix_name}::{action_name}"]


def paired_exact_p(left_only, right_only):
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    smaller = min(left_only, right_only)
    tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def train_ranker(plan, prefix_name, branch0_results, candidate_results, hidden_root):
    names = ["noop"] + [candidate["name"] for candidate in CANDIDATES]
    tasks = plan["tasks"]
    hidden = []
    labels = []
    for task in tasks:
        with np.load(hidden_root / f"{task['episode_key']}.npz", allow_pickle=False) as payload:
            hidden.append(np.asarray(payload[prefix_name], dtype=np.float32))
        labels.append(names.index(preferred_candidate(task, prefix_name, candidate_results)))
    hidden = np.stack(hidden)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.full(len(tasks), -1, dtype=np.int64)
    fixed_predictions = np.full(len(tasks), -1, dtype=np.int64)
    fold_records = []
    device = torch.device("cuda")
    action_tensor = candidate_action_tensor(device)
    for fold in range(5):
        train_indices = np.asarray([index for index, task in enumerate(tasks) if task["fold"] != fold], dtype=np.int64)
        test_indices = np.asarray([index for index, task in enumerate(tasks) if task["fold"] == fold], dtype=np.int64)
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise RuntimeError(f"empty grouped fold {fold}")
        mean = hidden[train_indices].mean(axis=0)
        std = hidden[train_indices].std(axis=0)
        std[std < 1e-6] = 1.0
        train_hidden = torch.tensor((hidden[train_indices] - mean) / std, dtype=torch.float32, device=device)
        train_labels = torch.tensor(labels[train_indices], dtype=torch.long, device=device)
        test_hidden = torch.tensor((hidden[test_indices] - mean) / std, dtype=torch.float32, device=device)
        torch.manual_seed(4200 + fold)
        model = ActionScorer(hidden.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(4300 + fold)
        for _ in range(100):
            order = torch.randperm(len(train_indices), generator=generator)
            for start in range(0, len(order), 64):
                batch = order[start:start + 64].to(device)
                scores = model(train_hidden[batch], action_tensor)
                loss = nn.functional.cross_entropy(scores, train_labels[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            predictions[test_indices] = model(test_hidden, action_tensor).argmax(dim=1).cpu().numpy()
        fixed_scores = []
        for action_index, action_name in enumerate(names):
            success = 0
            harm = 0
            for index in train_indices:
                task = tasks[index]
                record = selected_outcome(task, prefix_name, action_name, branch0_results, candidate_results)
                if task["stratum"] == "safe_control":
                    harm += int(record["outcome"] != "overtake")
                else:
                    success += int(record["outcome"] == "overtake")
            fixed_scores.append((success - 5 * harm, -action_index, action_index))
        fixed_index = max(fixed_scores)[2]
        fixed_predictions[test_indices] = fixed_index
        fold_records.append({"fold": fold, "train_count": len(train_indices), "test_count": len(test_indices), "fixed_baseline_action": names[fixed_index]})
    if bool(np.any(predictions < 0)) or bool(np.any(fixed_predictions < 0)):
        raise RuntimeError("out-of-fold predictions are incomplete")

    def summarize(selected):
        successes = {name: 0 for name in ("inherited_collision", "created_collision", "lost_overtake", "inherited_follow")}
        control_new_collision = 0
        control_overtake_loss = 0
        selected_names = []
        selected_action_by_episode = {}
        selected_outcome_by_episode = {}
        collision_removed = 0
        collision_created = 0
        overtake_lost = 0
        overtake_gained = 0
        for index, task in enumerate(tasks):
            action_name = names[int(selected[index])]
            selected_names.append(action_name)
            record = selected_outcome(task, prefix_name, action_name, branch0_results, candidate_results)
            selected_action_by_episode[task["episode_key"]] = action_name
            selected_outcome_by_episode[task["episode_key"]] = record["outcome"]
            source_collision = task["source_outcome"] in ("ego-opp", "ego-wall")
            selected_collision = record["outcome"] in ("ego-opp", "ego-wall")
            collision_removed += int(source_collision and not selected_collision)
            collision_created += int(not source_collision and selected_collision)
            source_overtake = task["source_outcome"] == "overtake"
            selected_overtake = record["outcome"] == "overtake"
            overtake_lost += int(source_overtake and not selected_overtake)
            overtake_gained += int(not source_overtake and selected_overtake)
            if task["stratum"] == "safe_control":
                control_new_collision += int(record["outcome"] in ("ego-opp", "ego-wall"))
                control_overtake_loss += int(record["outcome"] != "overtake")
            else:
                successes[task["stratum"]] += int(record["outcome"] == "overtake")
        return {
            "success_counts": successes,
            "target_success_total": successes["inherited_collision"] + successes["created_collision"] + successes["lost_overtake"],
            "safe_control_new_collision_count": control_new_collision,
            "safe_control_overtake_loss_count": control_overtake_loss,
            "selected_action_counts": dict(sorted((name, selected_names.count(name)) for name in set(selected_names))),
            "selected_action_by_episode": selected_action_by_episode,
            "selected_outcome_by_episode": selected_outcome_by_episode,
            "paired_vs_noop": {
                "collision_removed": collision_removed,
                "collision_created": collision_created,
                "collision_exact_p": paired_exact_p(collision_removed, collision_created),
                "overtake_lost": overtake_lost,
                "overtake_gained": overtake_gained,
                "overtake_exact_p": paired_exact_p(overtake_lost, overtake_gained),
            },
        }

    ranker = summarize(predictions)
    fixed = summarize(fixed_predictions)
    criteria = plan["admission_contract"]["rankability"]
    checks = {
        "inherited_collision_overtake_rescue": ranker["success_counts"]["inherited_collision"] >= criteria["inherited_collision_overtake_rescue_min"],
        "created_collision_overtake_rescue": ranker["success_counts"]["created_collision"] >= criteria["created_collision_overtake_rescue_min"],
        "lost_overtake_restore": ranker["success_counts"]["lost_overtake"] >= criteria["lost_overtake_restore_min"],
        "safe_control_new_collision": ranker["safe_control_new_collision_count"] <= criteria["safe_control_new_collision_max"],
        "safe_control_overtake_loss": ranker["safe_control_overtake_loss_count"] <= criteria["safe_control_overtake_loss_max"],
        "state_conditioned_margin_over_fixed": ranker["target_success_total"] - fixed["target_success_total"] >= criteria["state_conditioned_success_margin_over_grouped_fixed_baseline_min"],
    }
    return {
        "verdict": "pass" if all(checks.values()) else "fail",
        "model_contract": {"hidden_layer": 128, "action_score_layer": 64, "epochs": 100, "batch_size": 64, "optimizer": "Adam", "learning_rate": 1e-3, "weight_decay": 1e-4, "folds": 5},
        "oracle_label_counts": dict(sorted((names[index], int(np.sum(labels == index))) for index in np.unique(labels))),
        "ranker": ranker,
        "grouped_fixed_baseline": fixed,
        "folds": fold_records,
        "criteria": checks,
    }


def analyze(plan, branch0, candidates, quality, args):
    branch0_results = branch0["episodes"]
    candidate_results = candidates["episodes"]
    existence = {prefix: existence_metrics(plan, candidate_results, prefix) for prefix in PREFIXES}
    rankability = {}
    for prefix in PREFIXES:
        if existence[prefix]["verdict"] == "pass":
            rankability[prefix] = train_ranker(plan, prefix, branch0_results, candidate_results, args.output_dir / "branch0" / "hidden")
        else:
            rankability[prefix] = {"verdict": "not_run", "reason": "action existence gate failed"}
    late_ready = existence["late"]["verdict"] == "pass" and rankability["late"]["verdict"] == "pass"
    early_ready = existence["early"]["verdict"] == "pass" and rankability["early"]["verdict"] == "pass"
    if late_ready:
        verdict = "first_action_preference_ready_for_independent_preregistration"
    elif early_ready:
        verdict = "early_prefix_only_ready_for_prefix_reset_contract_audit"
    else:
        verdict = "fixed_local_action_library_not_admitted"
    return {
        "schema_version": 1,
        "experiment_id": "counterfactual_first_action_preference",
        "gate": "action_existence_and_rankability",
        "verdict": verdict,
        "quality_validation": quality,
        "existence": existence,
        "rankability": rankability,
        "route_decision": {
            "counterfactual_first_action_preference": "admit independent training preregistration" if late_ready else "close tested fixed-library formulation",
            "action_conditioned_controllability": "mechanism precheck passed" if late_ready or early_ready else "close tested frozen-hidden fixed-library formulation",
            "prefix_reset": "proceed to snapshot and GAE contract audit" if early_ready and not late_ready else "not selected by this gate",
            "constrained_ppo": "not logically tested by this gate",
            "interaction_phase_residual_moe": "closed by unchanged 12-key actor compatibility boundary",
        },
        "execution": {
            "device": "cuda",
            "workers": args.workers,
            "hidden_scale": args.hidden_scale,
            "sim_duration_s": args.sim_duration,
            "command": " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "worktree_status": subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.splitlines(),
        },
    }


if __name__ == "__main__":
    args = parse_arguments()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.hidden_scale != 4 or args.sim_duration != 8.0:
        raise ValueError("the action gate requires hidden scale 4 and 8 second episodes")
    if not torch.cuda.is_available() and not args.prepare_only:
        raise RuntimeError("CUDA is required for the counterfactual action gate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "action_gate_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        verify_plan(args, plan)
    else:
        plan = build_plan(args)
        atomic_write_json(plan_path, plan)
    print(json.dumps({"cohort": plan["cohort_contract"], "prefixes": plan["prefix_contract"], "candidate_count": plan["candidate_contract"]["candidate_count"]}, indent=2, sort_keys=True))
    if args.prepare_only:
        sys.exit(0)
    report_path = args.output_dir / "action_gate_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite completed action gate report: {report_path}")
    context = mp.get_context("forkserver")
    with context.Pool(processes=args.workers, initializer=worker_initializer, initargs=(str(args.u44_model_path), args.hidden_scale)) as pool:
        branch0_tasks = make_branch0_tasks(plan, args)
        branch0 = run_tasks(pool, branch0_tasks, args.output_dir / "branch0.partial.jsonl", args.output_dir / "branch0_results.json")
        if not all(record["replay_pass"] for record in branch0["episodes"].values()):
            raise RuntimeError("branch0 exact replay failed; candidate branches were not run")
        print("branch0 exact replay passed", flush=True)
        candidate_tasks = make_candidate_tasks(plan, args)
        candidates = run_tasks(pool, candidate_tasks, args.output_dir / "candidates.partial.jsonl", args.output_dir / "candidate_results.json")
    quality = validate_results(plan, branch0, candidates, args)
    report = analyze(plan, branch0, candidates, quality, args)
    atomic_write_json(report_path, report)
    print(json.dumps({"verdict": report["verdict"], "existence": report["existence"], "rankability": report["rankability"], "route_decision": report["route_decision"]}, indent=2, sort_keys=True))
