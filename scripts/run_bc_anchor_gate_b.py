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

import gym
import numpy as np
import torch
from f110_gym.envs.base_classes import Integrator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demonstration import setup_opp_planner
from eval_multiagent import classify_collision
from latticeplanner.utils import get_vertices, obsDict2oppoArray
from model import End2Race
from ppo.reward import ProgressProjector, rectangle_clearance
from utils import atomic_write_json, calculate_metrics, episode_key, evaluate_proximity_quality, load_positions_and_speeds_from_params, load_raceline_waypoints, save_numeric_npz, wrapped_progress_difference

WORKER_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
BRANCHES = ("branch0", "full_bc", "bc_steering", "bc_speed")
TRACE_DTYPES = {
    "time_s": np.float64,
    "ego_pose": np.float64,
    "opp_pose": np.float64,
    "collisions": np.bool_,
    "ego_opp_collision": np.bool_,
    "ego_wall_collision": np.bool_,
    "opp_wall_collision": np.bool_,
    "action_applied": np.bool_,
    "terminal_post_step": np.bool_,
    "intervention_active": np.bool_,
    "action_source_code": np.int8,
}
VEHICLE_LENGTH_M = 0.58
VEHICLE_WIDTH_M = 0.31
WORKER_DEVICE = None
WORKER_BC = None
WORKER_U44 = None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-panel", type=Path, required=True)
    parser.add_argument("--development-panel", type=Path, required=True)
    parser.add_argument("--gate-a-report", type=Path, required=True)
    parser.add_argument("--gate-a-evaluation-root", type=Path, required=True)
    parser.add_argument("--bc-model-path", type=Path, required=True)
    parser.add_argument("--u44-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--hidden-scale", type=int, default=4)
    parser.add_argument("--sim-duration", type=float, default=8.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--collision-only-validation", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def scenario_key(scenario):
    return episode_key(scenario["opp_raceline"], scenario["ego_idx"], scenario["opp_idx"], scenario["opp_speedscale"])


def circular_distance(left, right, waypoint_count):
    difference = abs(int(left) - int(right)) % waypoint_count
    return min(difference, waypoint_count - difference)


def obb_clearance_series(trace_path):
    with np.load(trace_path, allow_pickle=False) as payload:
        ego_pose = np.asarray(payload["ego_pose"], dtype=np.float64)
        opp_pose = np.asarray(payload["opp_pose"], dtype=np.float64)
        action_applied = np.asarray(payload["action_applied"], dtype=bool)
        ego_opp = np.asarray(payload["ego_opp_collision"], dtype=bool)
        ego_wall = np.asarray(payload["ego_wall_collision"], dtype=bool)
    if ego_pose.shape != opp_pose.shape or ego_pose.ndim != 2 or ego_pose.shape[1] != 3:
        raise RuntimeError(f"invalid pose arrays in {trace_path}")
    if len(action_applied) != len(ego_pose) or not bool(action_applied[:-1].all()) or bool(action_applied[-1]):
        raise RuntimeError(f"invalid action_applied contract in {trace_path}")
    clearances = np.empty(len(ego_pose), dtype=np.float64)
    for index in range(len(ego_pose)):
        ego_vertices = get_vertices(ego_pose[index], VEHICLE_LENGTH_M, VEHICLE_WIDTH_M)
        opp_vertices = get_vertices(opp_pose[index], VEHICLE_LENGTH_M, VEHICLE_WIDTH_M)
        clearances[index] = rectangle_clearance(ego_vertices, opp_vertices)
    if not np.isfinite(clearances).all() or np.any(clearances < 0.0):
        raise RuntimeError(f"invalid OBB clearances in {trace_path}")
    return clearances, action_applied, ego_opp, ego_wall


def intervention_window(trace_path, stratum):
    clearances, action_applied, ego_opp, ego_wall = obb_clearance_series(trace_path)
    applied_count = int(action_applied.sum())
    if stratum == "collision":
        collision_indices = np.flatnonzero(ego_opp | ego_wall)
        if len(collision_indices) == 0:
            raise RuntimeError(f"collision stratum has no ego collision marker: {trace_path}")
        event_index = int(collision_indices[0])
        start_index = max(0, event_index - 150)
        end_index = min(applied_count, event_index)
        event_name = "first_ego_collision"
    else:
        event_index = int(np.argmin(clearances))
        start_index = max(0, event_index - 100)
        end_index = min(applied_count, event_index + 50)
        event_name = "first_global_minimum_obb_clearance"
    return {
        "event_name": event_name,
        "event_index": event_index,
        "event_time_s": event_index * 0.01,
        "minimum_obb_clearance_m": float(clearances[event_index] if stratum != "collision" else np.min(clearances)),
        "start_index": int(start_index),
        "end_index_exclusive": int(end_index),
        "planned_action_steps": int(max(0, end_index - start_index)),
        "source_action_steps": applied_count,
        "eligible": int(max(0, end_index - start_index)) >= 50,
    }


def select_controls(source_tasks, safe_keys, scenarios, u44_trace_root):
    waypoint_count = len(load_raceline_waypoints("Austin", "raceline1.csv")) - 1
    available = set(safe_keys)
    selected = []
    for source in source_tasks:
        scenario = source["scenario"]
        candidates = [
            key for key in available
            if scenarios[key]["opp_raceline"] == scenario["opp_raceline"]
            and float(scenarios[key]["opp_speedscale"]) == float(scenario["opp_speedscale"])
        ]
        candidates.sort(key=lambda key: (
            circular_distance(scenario["ego_idx"], scenarios[key]["ego_idx"], waypoint_count),
            hashlib.sha256(key.encode("utf-8")).hexdigest(),
        ))
        if not candidates:
            raise RuntimeError(f"no unused safe control for {source['episode_key']}")
        control_key = candidates[0]
        available.remove(control_key)
        control_scenario = scenarios[control_key]
        window = intervention_window(u44_trace_root / f"{control_key}.npz", "control")
        selected.append({
            "episode_key": control_key,
            "role": "control",
            "stratum": "control",
            "matched_source_key": source["episode_key"],
            "circular_ego_index_distance": circular_distance(scenario["ego_idx"], control_scenario["ego_idx"], waypoint_count),
            "scenario": control_scenario,
            "window": window,
        })
    return selected


def build_plan(args):
    gate_a = json.loads(args.gate_a_report.read_text(encoding="utf-8"))
    if gate_a["verdict"] != "pass":
        raise RuntimeError("Gate A did not pass")
    collision_only_experiments = {"bc_collision_only_anchor_validation", "bc_collision_only_anchor_overlap_v2"}
    if args.collision_only_validation and (gate_a.get("experiment_id") not in collision_only_experiments or gate_a.get("gate") != "V0"):
        raise RuntimeError("collision-only validation requires the Round Z3 V0 report")
    cohort_panel = json.loads(args.cohort_panel.read_text(encoding="utf-8"))
    development_panel = json.loads(args.development_panel.read_text(encoding="utf-8"))
    scenarios = {scenario_key(scenario): scenario for scenario in development_panel}
    if len(scenarios) != len(development_panel):
        raise RuntimeError("development panel keys are not unique")
    consensus_keys = gate_a["cohort_definition"]["consensus_scenario_keys"]
    if {scenario_key(scenario) for scenario in cohort_panel} != set(consensus_keys):
        raise RuntimeError("cohort panel does not match Gate A report")

    u44_result_path = args.gate_a_evaluation_root / "u44" / "results_multi.json"
    bc_result_path = args.gate_a_evaluation_root / "bc" / "results_multi.json"
    u44_results = json.loads(u44_result_path.read_text(encoding="utf-8"))["episodes"]
    bc_results = json.loads(bc_result_path.read_text(encoding="utf-8"))["episodes"]
    u44_trace_root = args.gate_a_evaluation_root / "u44" / "traces"
    collision_keys = set(gate_a["cohort_definition"]["collision_scenario_keys"])
    lost_keys = set(gate_a["cohort_definition"]["lost_overtake_scenario_keys"])
    source_tasks = []
    excluded = []
    for key in consensus_keys:
        stratum = "collision" if key in collision_keys else "lost_overtake" if key in lost_keys else None
        if stratum is None:
            raise RuntimeError(f"Gate A consensus key has no stratum: {key}")
        window = intervention_window(u44_trace_root / f"{key}.npz", stratum)
        task = {
            "episode_key": key,
            "role": "cohort",
            "stratum": stratum,
            "matched_source_key": None,
            "circular_ego_index_distance": None,
            "scenario": scenarios[key],
            "window": window,
        }
        if window["eligible"]:
            source_tasks.append(task)
        else:
            excluded.append(task)

    safe_keys = sorted(
        key for key in scenarios
        if bc_results[key]["outcome"] == "overtake"
        and u44_results[key]["outcome"] == "overtake"
        and scenarios[key]["opp_raceline"] in ("raceline0", "raceline2")
    )
    control_tasks = select_controls(source_tasks, safe_keys, scenarios, u44_trace_root)
    expected_source_to_control = gate_a.get("control_support", {}).get("expected_source_to_control")
    if expected_source_to_control is not None:
        observed_source_to_control = {task["matched_source_key"]: task["episode_key"] for task in control_tasks}
        if observed_source_to_control != expected_source_to_control:
            raise RuntimeError("matched controls differ from the frozen overlap-support report")
    if any(not task["window"]["eligible"] for task in control_tasks):
        ineligible = [task["episode_key"] for task in control_tasks if not task["window"]["eligible"]]
        raise RuntimeError(f"matched safe controls have windows shorter than 50 steps: {ineligible}")
    collision_count = sum(task["stratum"] == "collision" for task in source_tasks)
    lost_count = sum(task["stratum"] == "lost_overtake" for task in source_tasks)
    collision_startpoints = {task["scenario"]["ego_idx"] for task in source_tasks if task["stratum"] == "collision"}
    collision_racelines = {task["scenario"]["opp_raceline"] for task in source_tasks if task["stratum"] == "collision"}
    if args.collision_only_validation:
        if lost_count != 0 or collision_count < 4 or len(collision_startpoints) < 3 or collision_racelines != {"raceline0", "raceline2"}:
            raise RuntimeError(f"inconclusive collision-only validation cohort after window filtering: C={collision_count}, starts={len(collision_startpoints)}, racelines={sorted(collision_racelines)}, L={lost_count}")
    elif collision_count < 8 or lost_count < 8:
        raise RuntimeError(f"eligible Gate B strata are too small: C={collision_count}, L={lost_count}")
    if len(control_tasks) != len(source_tasks):
        raise RuntimeError("safe control count does not equal eligible cohort count")

    return {
        "schema_version": 1,
        "experiment_id": gate_a["experiment_id"] if args.collision_only_validation else "front_corridor_temporal_bc_safe_anchor",
        "gate": "V1" if args.collision_only_validation else "B",
        "mode": "collision_only_validation" if args.collision_only_validation else "dual_stratum_development",
        "status": "frozen_before_branch_replay",
        "inputs": {
            "cohort_panel": str(args.cohort_panel),
            "cohort_panel_sha256": sha256_file(args.cohort_panel),
            "development_panel": str(args.development_panel),
            "development_panel_sha256": sha256_file(args.development_panel),
            "gate_a_report": str(args.gate_a_report),
            "gate_a_report_sha256": sha256_file(args.gate_a_report),
            "u44_result_sha256": sha256_file(u44_result_path),
            "bc_result_sha256": sha256_file(bc_result_path),
            "bc_model_path": str(args.bc_model_path),
            "bc_model_sha256": sha256_file(args.bc_model_path),
            "u44_model_path": str(args.u44_model_path),
            "u44_model_sha256": sha256_file(args.u44_model_path),
        },
        "window_contract": {
            "collision": "[first ego collision step - 150, first ego collision step)",
            "lost_overtake_and_control": "[first global minimum OBB clearance step - 100, step + 50)",
            "vehicle_length_m": VEHICLE_LENGTH_M,
            "vehicle_width_m": VEHICLE_WIDTH_M,
            "minimum_action_steps": 50,
            "terminal_post_step_excluded": True,
        },
        "control_contract": {
            "same_opponent_raceline": True,
            "same_opponent_speed_scale": True,
            "primary_sort": "circular ego waypoint index distance",
            "tie_break": "SHA256(scenario_key)",
            "without_replacement": True,
        },
        "eligible_cohort_count": len(source_tasks),
        "eligible_collision_count": collision_count,
        "eligible_lost_overtake_count": lost_count,
        "excluded_short_window_count": len(excluded),
        "excluded_short_window_tasks": excluded,
        "safe_control_pool_count": len(safe_keys),
        "tasks": source_tasks + control_tasks,
    }


def verify_plan(args, plan):
    expected_mode = "collision_only_validation" if args.collision_only_validation else "dual_stratum_development"
    observed_mode = plan.get("mode", "dual_stratum_development")
    if observed_mode != expected_mode:
        raise RuntimeError(f"frozen Gate B plan mode changed: {observed_mode} != {expected_mode}")
    expected = {
        "cohort_panel_sha256": sha256_file(args.cohort_panel),
        "development_panel_sha256": sha256_file(args.development_panel),
        "gate_a_report_sha256": sha256_file(args.gate_a_report),
        "bc_model_sha256": sha256_file(args.bc_model_path),
        "u44_model_sha256": sha256_file(args.u44_model_path),
    }
    for key, value in expected.items():
        if plan["inputs"].get(key) != value:
            raise RuntimeError(f"frozen Gate B plan input changed: {key}")
    keys = [task["episode_key"] for task in plan["tasks"]]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Gate B plan task keys are not unique")
    if len(plan["tasks"]) != 2 * plan["eligible_cohort_count"]:
        raise RuntimeError("Gate B plan cohort/control count mismatch")


def worker_initializer(bc_model_path, u44_model_path, hidden_scale):
    global WORKER_DEVICE, WORKER_BC, WORKER_U44
    for name, value in WORKER_ENV.items():
        os.environ[name] = value
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Gate B")
    WORKER_DEVICE = torch.device("cuda")
    WORKER_BC = End2Race(hidden_scale=hidden_scale).to(WORKER_DEVICE)
    WORKER_U44 = End2Race(hidden_scale=hidden_scale).to(WORKER_DEVICE)
    WORKER_BC.load_state_dict(torch.load(bc_model_path, map_location=WORKER_DEVICE, weights_only=True), strict=True)
    WORKER_U44.load_state_dict(torch.load(u44_model_path, map_location=WORKER_DEVICE, weights_only=True), strict=True)
    WORKER_BC.eval()
    WORKER_U44.eval()


def append_trace_row(trace, lap_time, lidar, opp_lidar, selected_raw, selected_executed, opp_action, u44_raw, bc_raw, obs, collisions, collision_markers, action_applied, terminal, intervention_active, source_code):
    trace["time_s"].append(float(lap_time))
    trace["ego_lidar_360"].append(lidar)
    trace["opp_lidar_360"].append(opp_lidar)
    trace["ego_raw_action"].append(selected_raw)
    trace["ego_executed_action"].append(selected_executed)
    trace["opp_executed_action"].append(opp_action)
    trace["u44_raw_action"].append(u44_raw)
    trace["bc_raw_action"].append(bc_raw)
    trace["ego_measured_speed_mps"].append(float(obs["linear_vels_x"][0]))
    trace["opp_measured_speed_mps"].append(float(obs["linear_vels_x"][1]))
    trace["ego_pose"].append([obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]])
    trace["opp_pose"].append([obs["poses_x"][1], obs["poses_y"][1], obs["poses_theta"][1]])
    trace["collisions"].append(collisions)
    trace["ego_opp_collision"].append(collision_markers[0])
    trace["ego_wall_collision"].append(collision_markers[1])
    trace["opp_wall_collision"].append(collision_markers[2])
    trace["action_applied"].append(action_applied)
    trace["terminal_post_step"].append(terminal)
    trace["intervention_active"].append(intervention_active)
    trace["action_source_code"].append(source_code)


def evaluate_branch(task):
    scenario = task["scenario"]
    branch = task["branch"]
    window = task["window"]
    trace_path = Path(task["trace_path"])
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
    u44_hidden = torch.zeros((1, 1, WORKER_U44.gru.hidden_size), device=WORKER_DEVICE)
    bc_hidden = torch.zeros((1, 1, WORKER_BC.gru.hidden_size), device=WORKER_DEVICE)
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
    trace = {name: [] for name in (
        "time_s", "ego_lidar_360", "opp_lidar_360", "ego_raw_action", "ego_executed_action",
        "opp_executed_action", "u44_raw_action", "bc_raw_action", "ego_measured_speed_mps",
        "opp_measured_speed_mps", "ego_pose", "opp_pose", "collisions", "ego_opp_collision",
        "ego_wall_collision", "opp_wall_collision", "action_applied", "terminal_post_step",
        "intervention_active", "action_source_code",
    )}

    while not done and lap_time < task["sim_duration"]:
        raw_lidar = np.asarray(obs["scans"][0]).reshape(-1)
        if len(raw_lidar) > 360:
            indices = np.linspace(0, len(raw_lidar) - 1, 360, dtype=int)
            raw_lidar = raw_lidar[indices]
        lidar = raw_lidar.copy()
        lidar_tensor = torch.tensor(lidar, dtype=torch.float32, device=WORKER_DEVICE).unsqueeze(0).unsqueeze(0)
        speed_tensor = torch.tensor([[[previous_speed]]], dtype=torch.float32, device=WORKER_DEVICE)
        with torch.no_grad():
            u44_sequence, u44_hidden = WORKER_U44(lidar_tensor, speed_tensor, u44_hidden)
            bc_sequence, bc_hidden = WORKER_BC(lidar_tensor, speed_tensor, bc_hidden)
        u44_raw = np.asarray((u44_sequence[0, -1, 0].item(), u44_sequence[0, -1, 1].item()), dtype=np.float32)
        bc_raw = np.asarray((bc_sequence[0, -1, 0].item(), bc_sequence[0, -1, 1].item()), dtype=np.float32)
        intervention_active = branch != "branch0" and window["start_index"] <= step_count < window["end_index_exclusive"]
        if intervention_active and branch == "full_bc":
            selected_raw = bc_raw.copy()
            source_code = 3
        elif intervention_active and branch == "bc_steering":
            selected_raw = np.asarray((bc_raw[0], u44_raw[1]), dtype=np.float32)
            source_code = 1
        elif intervention_active and branch == "bc_speed":
            selected_raw = np.asarray((u44_raw[0], bc_raw[1]), dtype=np.float32)
            source_code = 2
        else:
            selected_raw = u44_raw.copy()
            source_code = 0
        selected_steer = np.clip(float(selected_raw[0]), -0.52, 0.52)
        selected_speed = float(selected_raw[1])
        selected_executed = np.asarray((selected_steer, selected_speed), dtype=np.float32)
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
        append_trace_row(trace, lap_time, lidar, opp_lidar, selected_raw, selected_executed, opponent_action, u44_raw, bc_raw, obs, current_collisions, current_markers, True, False, intervention_active, source_code)
        raw_lidar_history.append(raw_lidar.copy())
        action_finite = action_finite and bool(np.isfinite((selected_executed, opponent_action, u44_raw, bc_raw)).all())
        intervention_steps += int(intervention_active)

        action = np.array([[selected_steer, selected_speed], [opponent_steer, opponent_speed]], dtype=np.float64)
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
            append_trace_row(trace, lap_time, terminal_lidar, terminal_opp_lidar, zeros, zeros, zeros, zeros, zeros, obs, step_collisions, step_markers, False, True, False, -1)
        tracker_count = (tracker_count + 1) % 10

    env.close()
    gc.collect()
    trace_arrays = {name: np.asarray(values, dtype=TRACE_DTYPES.get(name, np.float32)) for name, values in trace.items()}
    save_numeric_npz(trace_path, trace_arrays)
    avg_speed, speed_variance, total_distance = calculate_metrics(ego_trajectory, speeds)
    outcome = collision_type if collision_type is not None else "overtake" if relative_unwrapped > 0.0 else "follow"
    proximity = evaluate_proximity_quality(np.asarray(raw_lidar_history, dtype=np.float64))
    return {
        "episode_key": task["episode_key"],
        "branch": branch,
        "role": task["role"],
        "stratum": task["stratum"],
        "matched_source_key": task["matched_source_key"],
        "scenario_id": scenario["scenario_id"],
        "ego_idx": scenario["ego_idx"],
        "opp_idx": scenario["opp_idx"],
        "opp_raceline": scenario["opp_raceline"],
        "opp_speedscale": scenario["opp_speedscale"],
        "interval_idx": scenario["interval_idx"],
        "map_name": scenario["map_name"],
        "outcome": outcome,
        "ego_collision_time_s": ego_collision_time_s,
        "ego_collision_step": ego_collision_step,
        "simulation_time_s": float(lap_time),
        "steps": int(step_count),
        "intervention_steps": int(intervention_steps),
        "planned_intervention_steps": int(window["planned_action_steps"]),
        "observation_finite": observation_finite,
        "action_finite": action_finite,
        "avg_speed": float(avg_speed),
        "speed_variance": float(speed_variance),
        "total_distance": float(total_distance),
        "final_relative_position_m": float(relative_unwrapped),
        **proximity,
    }


def run_stage(pool, output_dir, branch, plan_tasks, sim_duration):
    stage_dir = output_dir / branch
    trace_dir = stage_dir / "traces"
    result_path = stage_dir / "results.json"
    partial_path = stage_dir / "episodes.partial.jsonl"
    stage_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    expected_keys = {task["episode_key"] for task in plan_tasks}
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if set(result["episodes"]) != expected_keys:
            raise RuntimeError(f"{branch}: completed result keys do not match plan")
        return result
    completed = {}
    if partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                completed[record["episode_key"]] = record
        print(f"{branch}: resuming {len(completed)}/{len(plan_tasks)}")
    pending = []
    for plan_task in plan_tasks:
        if plan_task["episode_key"] in completed:
            continue
        task = dict(plan_task)
        task["branch"] = branch
        task["sim_duration"] = sim_duration
        task["trace_path"] = str(trace_dir / f"{plan_task['episode_key']}.npz")
        pending.append(task)
    if pending:
        with partial_path.open("a", encoding="utf-8") as stream:
            for index, record in enumerate(pool.imap_unordered(evaluate_branch, pending), start=1):
                completed[record["episode_key"]] = record
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                if index % 10 == 0 or index == len(pending):
                    print(f"{branch}: {index}/{len(pending)} pending episodes")
    if set(completed) != expected_keys:
        raise RuntimeError(f"{branch}: result key set does not match plan")
    trace_keys = {path.stem for path in trace_dir.glob("*.npz")}
    if trace_keys != expected_keys:
        raise RuntimeError(f"{branch}: trace key set does not match plan")
    outcome_counts = {}
    for record in completed.values():
        outcome_counts[record["outcome"]] = outcome_counts.get(record["outcome"], 0) + 1
    result = {
        "branch": branch,
        "summary": {
            "episode_count": len(completed),
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "error_count": 0,
        },
        "episodes": dict(sorted(completed.items())),
    }
    atomic_write_json(result_path, result)
    partial_path.unlink(missing_ok=True)
    return result


def collision_identity(trace):
    ego_opp = np.asarray(trace["ego_opp_collision"], dtype=bool)
    ego_wall = np.asarray(trace["ego_wall_collision"], dtype=bool)
    indices = np.flatnonzero(ego_opp | ego_wall)
    if len(indices) == 0:
        return None, None
    index = int(indices[0])
    return ("ego-opp" if ego_opp[index] else "ego-wall"), index


def validate_branch0(plan_tasks, result, source_root, generated_root):
    exact_fields = ("ego_raw_action", "ego_executed_action")
    tolerance_fields = (
        "opp_executed_action", "ego_pose", "opp_pose", "ego_measured_speed_mps",
        "opp_measured_speed_mps", "ego_lidar_360", "opp_lidar_360",
    )
    boolean_fields = (
        "collisions", "ego_opp_collision", "ego_wall_collision", "opp_wall_collision",
        "action_applied", "terminal_post_step",
    )
    maxima = {field: 0.0 for field in exact_fields + tolerance_fields}
    episodes = {}
    all_pass = True
    source_results = json.loads((source_root.parent / "results_multi.json").read_text(encoding="utf-8"))["episodes"]
    for task in plan_tasks:
        key = task["episode_key"]
        with np.load(source_root / f"{key}.npz", allow_pickle=False) as payload:
            source = {field: np.asarray(payload[field]) for field in payload.files}
        with np.load(generated_root / f"{key}.npz", allow_pickle=False) as payload:
            generated = {field: np.asarray(payload[field]) for field in payload.files}
        field_errors = {}
        passed = True
        for field in exact_fields + tolerance_fields:
            if source[field].shape != generated[field].shape:
                error = float("inf")
            else:
                error = float(np.max(np.abs(source[field].astype(np.float64) - generated[field].astype(np.float64))))
            field_errors[field] = error
            maxima[field] = max(maxima[field], error)
            if field in exact_fields:
                passed = passed and error == 0.0
            else:
                passed = passed and error <= 1e-6
        for field in boolean_fields:
            equal = source[field].shape == generated[field].shape and np.array_equal(source[field], generated[field])
            field_errors[field] = 0.0 if equal else float("inf")
            passed = passed and equal
        source_collision = collision_identity(source)
        generated_collision = collision_identity(generated)
        source_record = source_results[key]
        generated_record = result["episodes"][key]
        outcome_equal = source_record["outcome"] == generated_record["outcome"]
        steps_equal = source_record["steps"] == generated_record["steps"]
        collision_equal = source_collision == generated_collision
        terminal = generated["terminal_post_step"].astype(bool)
        applied = generated["action_applied"].astype(bool)
        terminal_contract = int(terminal.sum()) == 1 and bool(terminal[-1]) and int((~applied).sum()) == 1 and not bool(applied[-1]) and bool(applied[:-1].all())
        passed = passed and outcome_equal and steps_equal and collision_equal and terminal_contract
        all_pass = all_pass and passed
        episodes[key] = {
            "pass": passed,
            "field_max_abs_error": field_errors,
            "outcome_equal": outcome_equal,
            "steps_equal": steps_equal,
            "first_ego_collision_equal": collision_equal,
            "terminal_contract": terminal_contract,
        }
    return {
        "pass": all_pass,
        "episode_count": len(plan_tasks),
        "exact_zero_fields": list(exact_fields),
        "atol_1e_6_fields": list(tolerance_fields),
        "boolean_exact_fields": list(boolean_fields),
        "field_max_abs_error": maxima,
        "episodes": episodes,
    }


def validate_branch_traces(plan, branch_results, output_dir):
    tasks = {task["episode_key"]: task for task in plan["tasks"]}
    required = {
        "time_s", "ego_lidar_360", "opp_lidar_360", "ego_raw_action", "ego_executed_action",
        "opp_executed_action", "u44_raw_action", "bc_raw_action", "ego_measured_speed_mps",
        "opp_measured_speed_mps", "ego_pose", "opp_pose", "collisions", "ego_opp_collision",
        "ego_wall_collision", "opp_wall_collision", "action_applied", "terminal_post_step",
        "intervention_active", "action_source_code",
    }
    source_codes = {"branch0": 0, "full_bc": 3, "bc_steering": 1, "bc_speed": 2}
    summaries = {}
    for branch in branch_results:
        episodes = branch_results[branch]["episodes"]
        if set(episodes) != set(tasks):
            raise RuntimeError(f"{branch}: result keys do not match Gate B plan")
        trace_root = output_dir / branch / "traces"
        if {path.stem for path in trace_root.glob("*.npz")} != set(tasks):
            raise RuntimeError(f"{branch}: trace keys do not match Gate B plan")
        row_count = 0
        early_terminated_windows = 0
        for key, record in episodes.items():
            task = tasks[key]
            scenario = task["scenario"]
            for field in ("scenario_id", "ego_idx", "opp_idx", "opp_raceline", "opp_speedscale", "interval_idx", "map_name"):
                if record[field] != scenario[field]:
                    raise RuntimeError(f"{branch}/{key}: result identity does not match Gate B plan")
            with np.load(trace_root / f"{key}.npz", allow_pickle=False) as payload:
                if not required <= set(payload.files):
                    raise RuntimeError(f"{branch}/{key}: missing trace fields")
                arrays = {name: np.asarray(payload[name]) for name in payload.files}
            lengths = {len(value) for value in arrays.values()}
            if len(lengths) != 1:
                raise RuntimeError(f"{branch}/{key}: trace arrays are not aligned")
            length = lengths.pop()
            row_count += length
            if length != record["steps"] + 1 or not all(bool(np.isfinite(value).all()) for value in arrays.values()):
                raise RuntimeError(f"{branch}/{key}: trace length or finite contract failed")
            shape_contract = {
                "time_s": (length,),
                "ego_lidar_360": (length, 360),
                "opp_lidar_360": (length, 360),
                "ego_raw_action": (length, 2),
                "ego_executed_action": (length, 2),
                "opp_executed_action": (length, 2),
                "u44_raw_action": (length, 2),
                "bc_raw_action": (length, 2),
                "ego_pose": (length, 3),
                "opp_pose": (length, 3),
                "collisions": (length, 2),
            }
            if any(arrays[name].shape != shape for name, shape in shape_contract.items()) or not bool(np.all(np.diff(arrays["time_s"]) > 0.0)):
                raise RuntimeError(f"{branch}/{key}: trace shape or time contract failed")
            terminal = arrays["terminal_post_step"].astype(bool)
            applied = arrays["action_applied"].astype(bool)
            if int(terminal.sum()) != 1 or not bool(terminal[-1]) or int((~applied).sum()) != 1 or bool(applied[-1]) or not bool(applied[:-1].all()):
                raise RuntimeError(f"{branch}/{key}: terminal contract failed")
            ego_opp = arrays["ego_opp_collision"].astype(bool)
            ego_wall = arrays["ego_wall_collision"].astype(bool)
            opp_wall = arrays["opp_wall_collision"].astype(bool)
            collisions = arrays["collisions"].astype(bool)
            if not np.array_equal(collisions[:, 0], ego_opp | ego_wall) or not np.array_equal(collisions[:, 1], ego_opp | opp_wall) or bool(np.any(ego_opp & ego_wall)):
                raise RuntimeError(f"{branch}/{key}: collision marker contract failed")
            observed_outcome = "ego-opp" if bool(ego_opp.any()) else "ego-wall" if bool(ego_wall.any()) else None
            expected_outcome = record["outcome"] if record["outcome"] in ("ego-opp", "ego-wall") else None
            if observed_outcome != expected_outcome:
                raise RuntimeError(f"{branch}/{key}: collision outcome does not match trace")
            active = arrays["intervention_active"].astype(bool)
            sources = arrays["action_source_code"]
            if bool(active[-1]) or sources[-1] != -1 or int(active.sum()) != record["intervention_steps"]:
                raise RuntimeError(f"{branch}/{key}: intervention marker contract failed")
            expected_active = np.zeros(length, dtype=bool)
            if branch != "branch0":
                start = task["window"]["start_index"]
                end = min(task["window"]["end_index_exclusive"], record["steps"])
                expected_active[start:end] = True
                early_terminated_windows += int(record["intervention_steps"] < record["planned_intervention_steps"])
            if not np.array_equal(active, expected_active) or not bool(np.all(sources[active] == source_codes[branch])) or not bool(np.all(sources[:-1][~active[:-1]] == 0)):
                raise RuntimeError(f"{branch}/{key}: intervention interval or source code failed")
            chosen = arrays["u44_raw_action"].copy()
            if branch == "full_bc":
                chosen[active] = arrays["bc_raw_action"][active]
            elif branch == "bc_steering":
                chosen[active, 0] = arrays["bc_raw_action"][active, 0]
            elif branch == "bc_speed":
                chosen[active, 1] = arrays["bc_raw_action"][active, 1]
            if not np.array_equal(arrays["ego_raw_action"][:-1], chosen[:-1]):
                raise RuntimeError(f"{branch}/{key}: selected raw action contract failed")
            expected_executed = arrays["ego_raw_action"].copy()
            expected_executed[:, 0] = np.clip(expected_executed[:, 0], -0.52, 0.52)
            expected_executed[-1] = 0.0
            if not np.array_equal(arrays["ego_executed_action"], expected_executed):
                raise RuntimeError(f"{branch}/{key}: executed action contract failed")
        summaries[branch] = {
            "episode_count": len(episodes),
            "trace_count": len(episodes),
            "trace_row_count": row_count,
            "early_terminated_window_count": early_terminated_windows,
        }
    return {
        "actor_episode_count": sum(summary["episode_count"] for summary in summaries.values()),
        "branches": summaries,
        "result_trace_plan_key_sets_equal": True,
        "episode_identity_matches_plan": True,
        "all_trace_arrays_aligned_and_finite": True,
        "trace_shapes_and_time_contract_complete": True,
        "collision_markers_match_outcomes": True,
        "terminal_contract_complete": True,
        "intervention_windows_and_source_codes_match_plan": True,
        "selected_and_executed_actions_match_branch_contract": True,
    }


def outcome_counts(keys, episodes):
    counts = {"overtake": 0, "follow": 0, "ego-opp": 0, "ego-wall": 0}
    for key in keys:
        outcome = episodes[key]["outcome"]
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def analyze_gate_b(plan, branch0_contract, branch_results, quality_validation, args):
    tasks = {task["episode_key"]: task for task in plan["tasks"]}
    full = branch_results["full_bc"]["episodes"]
    collision_keys = sorted(key for key, task in tasks.items() if task["stratum"] == "collision")
    lost_keys = sorted(key for key, task in tasks.items() if task["stratum"] == "lost_overtake")
    control_keys = sorted(key for key, task in tasks.items() if task["stratum"] == "control")
    rescued_collision = [key for key in collision_keys if full[key]["outcome"] not in ("ego-opp", "ego-wall")]
    rescued_collision_overtake = [key for key in rescued_collision if full[key]["outcome"] == "overtake"]
    restored_lost = [key for key in lost_keys if full[key]["outcome"] == "overtake"]
    lost_new_collision = [key for key in lost_keys if full[key]["outcome"] in ("ego-opp", "ego-wall")]
    control_new_collision = [key for key in control_keys if full[key]["outcome"] in ("ego-opp", "ego-wall")]
    control_overtake_loss = [key for key in control_keys if full[key]["outcome"] != "overtake"]
    collision_raceline_counts = {}
    rescued_raceline_counts = {}
    for key in collision_keys:
        raceline = tasks[key]["scenario"]["opp_raceline"]
        collision_raceline_counts[raceline] = collision_raceline_counts.get(raceline, 0) + 1
    for key in rescued_collision:
        raceline = tasks[key]["scenario"]["opp_raceline"]
        rescued_raceline_counts[raceline] = rescued_raceline_counts.get(raceline, 0) + 1
    required_rescue_by_raceline = 2 if not args.collision_only_validation and all(collision_raceline_counts.get(raceline, 0) >= 4 for raceline in ("raceline0", "raceline2")) else 1
    collision_rescue_threshold = math.ceil(0.50 * len(collision_keys))
    rescued_overtake_threshold = math.ceil(0.80 * len(rescued_collision))
    lost_collision_limit = math.floor(0.05 * len(lost_keys))
    lost_restore_threshold = math.ceil(0.80 * len(lost_keys))
    control_collision_limit = math.floor(0.05 * len(control_keys))
    control_overtake_loss_limit = math.floor(0.05 * len(control_keys))
    criteria = {
        "branch0_exact_replay": branch0_contract["pass"],
        "collision_rescue_at_least_50_percent": len(rescued_collision) >= collision_rescue_threshold,
        "rescued_collision_overtake_retention_at_least_80_percent": len(rescued_collision_overtake) >= rescued_overtake_threshold,
        "rescued_collision_covers_two_startpoints": len({tasks[key]["scenario"]["ego_idx"] for key in rescued_collision}) >= 2,
        "rescued_collision_raceline_coverage": all(rescued_raceline_counts.get(raceline, 0) >= required_rescue_by_raceline for raceline in ("raceline0", "raceline2") if collision_raceline_counts.get(raceline, 0) >= 2),
        "safe_control_new_collision_within_5_percent": len(control_new_collision) <= control_collision_limit,
        "safe_control_overtake_loss_within_5_percent": len(control_overtake_loss) <= control_overtake_loss_limit,
    }
    if not args.collision_only_validation:
        criteria.update({
            "lost_overtake_new_collision_within_5_percent": len(lost_new_collision) <= lost_collision_limit,
            "lost_overtake_restore_at_least_80_percent": len(restored_lost) >= lost_restore_threshold,
            "lost_overtake_restore_covers_two_startpoints": len({tasks[key]["scenario"]["ego_idx"] for key in restored_lost}) >= 2,
            "lost_overtake_restore_covers_both_racelines": len({tasks[key]["scenario"]["opp_raceline"] for key in restored_lost}) >= 2,
        })
    diagnostics = {}
    for branch in ("full_bc",) if args.collision_only_validation else ("full_bc", "bc_steering", "bc_speed"):
        episodes = branch_results[branch]["episodes"]
        diagnostics[branch] = {
            "collision_stratum_outcomes": outcome_counts(collision_keys, episodes),
            "lost_overtake_stratum_outcomes": outcome_counts(lost_keys, episodes),
            "safe_control_outcomes": outcome_counts(control_keys, episodes),
        }
    return {
        "schema_version": 1,
        "experiment_id": plan["experiment_id"] if args.collision_only_validation else "front_corridor_temporal_bc_safe_anchor",
        "gate": "V1" if args.collision_only_validation else "B",
        "verdict": "pass" if all(criteria.values()) else "fail",
        "mechanism_replication_pass": bool(all(criteria.values())) if args.collision_only_validation else None,
        "plan": {
            "eligible_cohort_count": plan["eligible_cohort_count"],
            "collision_count": len(collision_keys),
            "lost_overtake_count": len(lost_keys),
            "safe_control_count": len(control_keys),
            "excluded_short_window_count": plan["excluded_short_window_count"],
        },
        "quality_validation": quality_validation,
        "branch0_contract": branch0_contract,
        "full_bc_admission": {
            "collision": {
                "source_count": len(collision_keys),
                "rescued_count": len(rescued_collision),
                "required_rescued_count": collision_rescue_threshold,
                "rescued_overtake_count": len(rescued_collision_overtake),
                "required_rescued_overtake_count": rescued_overtake_threshold,
                "rescued_unique_startpoint_count": len({tasks[key]["scenario"]["ego_idx"] for key in rescued_collision}),
                "source_by_raceline": dict(sorted(collision_raceline_counts.items())),
                "rescued_by_raceline": dict(sorted(rescued_raceline_counts.items())),
                "required_rescued_per_raceline": required_rescue_by_raceline,
                "rescued_scenario_keys": rescued_collision,
                "rescued_overtake_scenario_keys": rescued_collision_overtake,
            },
            "lost_overtake": {
                "source_count": len(lost_keys),
                "restored_overtake_count": len(restored_lost),
                "required_restored_overtake_count": lost_restore_threshold,
                "new_collision_count": len(lost_new_collision),
                "new_collision_limit": lost_collision_limit,
                "restored_unique_startpoint_count": len({tasks[key]["scenario"]["ego_idx"] for key in restored_lost}),
                "restored_racelines": sorted({tasks[key]["scenario"]["opp_raceline"] for key in restored_lost}),
                "restored_scenario_keys": restored_lost,
                "new_collision_scenario_keys": lost_new_collision,
            },
            "safe_controls": {
                "source_count": len(control_keys),
                "new_collision_count": len(control_new_collision),
                "new_collision_limit": control_collision_limit,
                "overtake_loss_count": len(control_overtake_loss),
                "overtake_loss_limit": control_overtake_loss_limit,
                "new_collision_scenario_keys": control_new_collision,
                "overtake_loss_scenario_keys": control_overtake_loss,
            },
        },
        "branch_diagnostics": diagnostics,
        "admission_criteria": criteria,
        "execution": {
            "device": "cuda",
            "workers": args.workers,
            "hidden_scale": args.hidden_scale,
            "sim_duration_s": args.sim_duration,
            "command": " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "worktree_status": subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.splitlines(),
        },
        "next_action": "Write a separate collision-only formal-training preregistration; do not start training" if all(criteria.values()) and args.collision_only_validation else "Freeze successful full-BC branch sequences for the anchor dataset" if all(criteria.values()) else "Close the collision-only validation instance" if args.collision_only_validation else "Stop BC anchoring before anchor dataset and Gate C",
    }


if __name__ == "__main__":
    args = parse_arguments()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.hidden_scale != 4 or args.sim_duration != 8.0:
        raise ValueError("Gate B requires hidden scale 4 and 8 second episodes")
    if not torch.cuda.is_available() and not args.prepare_only:
        raise RuntimeError("CUDA is required for Gate B")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "gate_b_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        verify_plan(args, plan)
    else:
        plan = build_plan(args)
        atomic_write_json(plan_path, plan)
    print(json.dumps({
        "eligible_cohort_count": plan["eligible_cohort_count"],
        "eligible_collision_count": plan["eligible_collision_count"],
        "eligible_lost_overtake_count": plan["eligible_lost_overtake_count"],
        "safe_control_count": len(plan["tasks"]) - plan["eligible_cohort_count"],
        "excluded_short_window_count": plan["excluded_short_window_count"],
    }, indent=2, sort_keys=True))
    if args.prepare_only:
        sys.exit(0)

    report_path = args.output_dir / "gate_b_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite completed Gate B report: {report_path}")
    context = mp.get_context("forkserver")
    intervention_branches = ("full_bc",) if args.collision_only_validation else ("full_bc", "bc_steering", "bc_speed")
    with context.Pool(processes=args.workers, initializer=worker_initializer, initargs=(str(args.bc_model_path), str(args.u44_model_path), args.hidden_scale)) as pool:
        branch_results = {}
        branch_results["branch0"] = run_stage(pool, args.output_dir, "branch0", plan["tasks"], args.sim_duration)
        branch0_contract = validate_branch0(
            plan["tasks"],
            branch_results["branch0"],
            args.gate_a_evaluation_root / "u44" / "traces",
            args.output_dir / "branch0" / "traces",
        )
        atomic_write_json(args.output_dir / "branch0_contract.json", branch0_contract)
        if not branch0_contract["pass"]:
            raise RuntimeError("Gate B branch 0 exact replay contract failed; intervention branches were not run")
        print("branch0: exact replay contract passed")
        for branch in intervention_branches:
            branch_results[branch] = run_stage(pool, args.output_dir, branch, plan["tasks"], args.sim_duration)
    quality_validation = validate_branch_traces(plan, branch_results, args.output_dir)
    report = analyze_gate_b(plan, branch0_contract, branch_results, quality_validation, args)
    atomic_write_json(report_path, report)
    print(json.dumps({
        "verdict": report["verdict"],
        "plan": report["plan"],
        "full_bc_admission": report["full_bc_admission"],
        "admission_criteria": report["admission_criteria"],
    }, indent=2, sort_keys=True))
