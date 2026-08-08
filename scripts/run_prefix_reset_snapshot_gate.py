import argparse
import copy
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import subprocess
import sys
import warnings

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import End2Race
from ppo.env import CORRIDOR_TEMPORAL_EXPLORATION_MODE, EXTERNAL_RESET_OPTION, make_environment
from ppo.policy import PrivilegeGRUCritic
from ppo.scenarios import EpisodeResetSpec
from utils import atomic_write_json, load_positions_and_speeds_from_params, save_numeric_npz

WORKER_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
REWARD_FIELDS = (
    "_previous_ego_progress",
    "_previous_opponent_progress",
    "_relative_position_m",
    "_opponent_collision_latched",
    "_ego_collision_penalty_applied",
    "_scenario_id",
    "_previous_risk_potential",
    "current_clearances",
)
WRAPPER_FIELDS = (
    "_elapsed_time",
    "_previous_ego_speed",
    "_raw_observation",
    "_current_spec",
    "_episode_return",
    "_episode_steps",
    "_episode_reward_progress",
    "_episode_reward_relative",
    "_episode_reward_collision",
    "_episode_reward_risk",
    "_episode_abs_reward_risk",
    "_episode_min_obb_clearance_m",
    "_episode_min_wall_clearance_m",
    "_episode_risk_active_steps",
)
CORE_FIELDS = (
    "poses_x",
    "poses_y",
    "poses_theta",
    "collisions",
    "near_start",
    "num_toggles",
    "lap_times",
    "lap_counts",
    "current_time",
    "near_starts",
    "toggle_list",
    "start_xs",
    "start_ys",
    "start_thetas",
    "start_rot",
    "render_obs",
)
PLANNER_FIELDS = (
    "best_traj",
    "best_traj_ref_v",
    "best_traj_idx",
    "prev_traj_local",
    "prev_opp_pose",
    "goal_grid",
    "state_i",
    "state_t",
    "step_all_cost",
    "all_costs",
    "last_s",
    "step",
)
TRACE_DTYPES = {
    "time_s": np.float64,
    "observation_before": np.float32,
    "observation_after": np.float32,
    "ego_lidar_360": np.float32,
    "opp_lidar_360": np.float32,
    "ego_state": np.float64,
    "opp_state": np.float64,
    "ego_steer_buffer": np.float64,
    "opp_steer_buffer": np.float64,
    "actor_hidden_before": np.float32,
    "critic_hidden_before": np.float32,
    "actor_hidden_after": np.float32,
    "critic_hidden_after": np.float32,
    "actor_raw_action": np.float32,
    "actor_executed_action": np.float32,
    "opponent_executed_action": np.float32,
    "critic_value": np.float32,
    "reward_total": np.float64,
    "reward_components": np.float64,
    "collisions": np.bool_,
    "terminated_after": np.bool_,
    "truncated_after": np.bool_,
    "action_applied": np.bool_,
    "terminal_post_step": np.bool_,
}
WORKER_DEVICE = None
WORKER_ACTOR = None
WORKER_CRITIC = None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-b-plan", type=Path, default=Path("eval_results/front_corridor_temporal_bc_safe_anchor/gate_b/gate_b_plan.json"))
    parser.add_argument("--actor-path", type=Path, default=Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update44/actor.pth"))
    parser.add_argument("--critic-path", type=Path, default=Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update44/critic.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results/prefix_reset_snapshot_gate"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-scale", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_snapshot():
    head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(("git", "status", "--short"), cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    return {"git_head": head, "git_status_short": status}


def build_plan(args):
    gate_b = json.loads(args.gate_b_plan.read_text(encoding="utf-8"))
    tasks = [copy.deepcopy(task) for task in gate_b["tasks"] if task["role"] == "cohort"]
    tasks.sort(key=lambda task: task["episode_key"])
    strata = {name: sum(task["stratum"] == name for task in tasks) for name in ("collision", "lost_overtake")}
    if len(tasks) != 28 or strata != {"collision": 19, "lost_overtake": 9}:
        raise RuntimeError(f"Expected frozen 19/9 consensus cohort, got {len(tasks)} and {strata}")
    if len({task["episode_key"] for task in tasks}) != len(tasks) or len({task["scenario"]["ego_idx"] for task in tasks}) != 21:
        raise RuntimeError("Frozen snapshot task identity or startpoint count changed")
    if {task["scenario"]["opp_raceline"] for task in tasks} != {"raceline0", "raceline2"}:
        raise RuntimeError("Frozen snapshot tasks must cover opponent raceline0 and raceline2")
    return {
        "schema_version": 1,
        "experiment_id": "prefix_reset_snapshot_gate",
        "gate": "Z6-A",
        "status": "frozen_before_snapshot_execution",
        "inputs": {
            "gate_b_plan": str(args.gate_b_plan),
            "gate_b_plan_sha256": sha256_file(args.gate_b_plan),
            "actor_path": str(args.actor_path),
            "actor_sha256": sha256_file(args.actor_path),
            "critic_path": str(args.critic_path),
            "critic_sha256": sha256_file(args.critic_path),
        },
        "task_contract": {
            "map": "Austin",
            "source": "Gate A U42-U45 3-of-4 development consensus",
            "task_count": len(tasks),
            "collision_count": strata["collision"],
            "lost_overtake_count": strata["lost_overtake"],
            "unique_ego_startpoints": 21,
            "snapshot_position": "before consuming observation at frozen window.start_index",
            "deterministic_actor": True,
            "privilege_gru_critic": True,
            "pickle_round_trip_required": True,
        },
        "snapshot_schema": {
            "reward_fields": list(REWARD_FIELDS),
            "wrapper_fields": list(WRAPPER_FIELDS),
            "f110_core_fields": list(CORE_FIELDS),
            "planner_fields": list(PLANNER_FIELDS),
            "racecar_fields": ["state", "opp_poses", "accel", "steer_angle_vel", "steer_buffer", "in_collision", "scan_rng_state"],
            "simulator_fields": ["agent_poses", "collisions", "collision_idx"],
            "recurrent_fields": ["actor_hidden_before", "critic_hidden_before"],
        },
        "admission_contract": {
            "required_pass_count": 28,
            "required_failure_count": 0,
            "all_float_field_max_abs_error": 0.0,
            "all_boolean_fields_exact": True,
            "outcome_collision_step_terminal_exact": True,
            "failure_action": "stop prefix-reset; replay-to-prefix is forbidden",
            "pass_action": "run separate current-network burn-in and GAE semantics gate before any PPO",
        },
        "tasks": tasks,
        "source_snapshot": source_snapshot(),
    }


def write_frozen_plan(args, plan):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "snapshot_gate_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise RuntimeError("Existing snapshot plan differs from current frozen inputs")
        return plan_path
    if any(args.output_dir.iterdir()):
        raise RuntimeError(f"Output directory must be empty before plan creation: {args.output_dir}")
    atomic_write_json(plan_path, plan)
    return plan_path


def copied_fields(obj, fields):
    return {name: copy.deepcopy(getattr(obj, name)) for name in fields}


def capture_planner(controller):
    planner = controller.planner
    tracker = planner.tracker
    if planner.selection_func not in (None, np.argmin):
        raise RuntimeError("Unsupported LatticePlanner selection function in snapshot")
    if tracker.drawn_waypoints:
        raise RuntimeError("Snapshot gate does not support rendered PurePursuit state")
    return {
        "trajectory": copy.deepcopy(controller.trajectory),
        "tracker_count": int(controller.tracker_count),
        "speed_scale": float(controller.speed_scale),
        "planner": copied_fields(planner, PLANNER_FIELDS),
        "selection_func": "none" if planner.selection_func is None else "numpy_argmin",
        "tracker_prev_error": float(tracker.prev_error),
        "tracker_has_nearest_dist": hasattr(tracker, "nearest_dist"),
        "tracker_nearest_dist": None if not hasattr(tracker, "nearest_dist") else float(tracker.nearest_dist),
        "tracker_drawn_waypoints": [],
    }


def restore_planner(controller, state):
    controller.trajectory = copy.deepcopy(state["trajectory"])
    controller.tracker_count = int(state["tracker_count"])
    controller.speed_scale = float(state["speed_scale"])
    planner = controller.planner
    for name, value in state["planner"].items():
        setattr(planner, name, copy.deepcopy(value))
    planner.selection_func = None if state["selection_func"] == "none" else np.argmin
    tracker = planner.tracker
    tracker.prev_error = float(state["tracker_prev_error"])
    tracker.drawn_waypoints = []
    if state["tracker_has_nearest_dist"]:
        tracker.nearest_dist = float(state["tracker_nearest_dist"])
    elif hasattr(tracker, "nearest_dist"):
        delattr(tracker, "nearest_dist")


def capture_environment(env):
    core = env.f110_env.unwrapped
    agents = []
    for agent in core.sim.agents:
        agents.append({
            "state": np.asarray(agent.state).copy(),
            "opp_poses": copy.deepcopy(agent.opp_poses),
            "accel": float(agent.accel),
            "steer_angle_vel": float(agent.steer_angle_vel),
            "steer_buffer": np.asarray(agent.steer_buffer).copy(),
            "in_collision": bool(agent.in_collision),
            "scan_rng_state": copy.deepcopy(agent.scan_rng.bit_generator.state),
        })
    return {
        "schema_version": 1,
        "order_enforcing_has_reset": bool(getattr(env.f110_env, "_has_reset", True)),
        "racecars": agents,
        "simulator": {
            "agent_poses": np.asarray(core.sim.agent_poses).copy(),
            "collisions": np.asarray(core.sim.collisions).copy(),
            "collision_idx": np.asarray(core.sim.collision_idx).copy(),
        },
        "f110_core": copied_fields(core, CORE_FIELDS),
        "f110_current_obs": copy.deepcopy(type(core).current_obs),
        "opponent_controller": capture_planner(env.opponent_controller),
        "reward": copied_fields(env.transition_reward, REWARD_FIELDS),
        "wrapper": copied_fields(env, WRAPPER_FIELDS),
        "reset_rng_state": copy.deepcopy(env._reset_rng.bit_generator.state),
        "corridor_gate_current": None if env.following_danger_gate is None else bool(env.following_danger_gate.current_gate),
    }


def restore_environment(env, state):
    if state.get("schema_version") != 1 or set(state) != {"schema_version", "order_enforcing_has_reset", "racecars", "simulator", "f110_core", "f110_current_obs", "opponent_controller", "reward", "wrapper", "reset_rng_state", "corridor_gate_current"}:
        raise RuntimeError("Snapshot schema is incomplete or unsupported")
    core = env.f110_env.unwrapped
    if len(state["racecars"]) != len(core.sim.agents):
        raise RuntimeError("Snapshot RaceCar count changed")
    for agent, saved in zip(core.sim.agents, state["racecars"]):
        agent.state = np.asarray(saved["state"]).copy()
        agent.opp_poses = copy.deepcopy(saved["opp_poses"])
        agent.accel = float(saved["accel"])
        agent.steer_angle_vel = float(saved["steer_angle_vel"])
        agent.steer_buffer = np.asarray(saved["steer_buffer"]).copy()
        agent.in_collision = bool(saved["in_collision"])
        agent.scan_rng.bit_generator.state = copy.deepcopy(saved["scan_rng_state"])
    core.sim.agent_poses = np.asarray(state["simulator"]["agent_poses"]).copy()
    core.sim.collisions = np.asarray(state["simulator"]["collisions"]).copy()
    core.sim.collision_idx = np.asarray(state["simulator"]["collision_idx"]).copy()
    for name, value in state["f110_core"].items():
        setattr(core, name, copy.deepcopy(value))
    type(core).current_obs = copy.deepcopy(state["f110_current_obs"])
    if hasattr(env.f110_env, "_has_reset"):
        env.f110_env._has_reset = bool(state["order_enforcing_has_reset"])
    restore_planner(env.opponent_controller, state["opponent_controller"])
    for name, value in state["reward"].items():
        setattr(env.transition_reward, name, copy.deepcopy(value))
    for name, value in state["wrapper"].items():
        setattr(env, name, copy.deepcopy(value))
    env._reset_rng.bit_generator.state = copy.deepcopy(state["reset_rng_state"])
    if env.following_danger_gate is not None:
        if state["corridor_gate_current"] is None:
            raise RuntimeError("Snapshot corridor gate state is missing")
        env.following_danger_gate.current_gate = bool(state["corridor_gate_current"])


def downsample_lidar(raw_observation, index):
    lidar = np.asarray(raw_observation["scans"][index]).reshape(-1)
    if lidar.size > 360:
        lidar = lidar[np.linspace(0, lidar.size - 1, 360, dtype=int)]
    return np.asarray(lidar, dtype=np.float32)


def physical_state(env):
    core = env.f110_env.unwrapped
    return tuple(np.asarray(agent.state, dtype=np.float64).copy() for agent in core.sim.agents)


def model_step(observation, actor_hidden, critic_hidden):
    full = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=WORKER_DEVICE).reshape(1, -1)
    lidar = full[:, :360].unsqueeze(1)
    previous_speed = full[:, 360:361].unsqueeze(1)
    privileged = full[:, 361:]
    with torch.no_grad():
        action_sequence, next_actor_hidden = WORKER_ACTOR(lidar, previous_speed, actor_hidden)
        value, next_critic_hidden = WORKER_CRITIC.step(lidar, previous_speed, critic_hidden, privileged)
    action = action_sequence[0, -1].detach().cpu().numpy().astype(np.float32)
    return action, float(value[0, 0].item()), next_actor_hidden, next_critic_hidden


def trace_row(env, observation_before, observation_after, actor_hidden_before, critic_hidden_before, actor_hidden_after, critic_hidden_after, raw_action, executed_action, opponent_action, critic_value, reward, reward_components, terminated, truncated, action_applied, terminal_post_step):
    ego_state, opp_state = physical_state(env)
    core = env.f110_env.unwrapped
    return {
        "time_s": float(env._elapsed_time),
        "observation_before": np.asarray(observation_before, dtype=np.float32),
        "observation_after": np.asarray(observation_after, dtype=np.float32),
        "ego_lidar_360": downsample_lidar(env._raw_observation, 0),
        "opp_lidar_360": downsample_lidar(env._raw_observation, 1),
        "ego_state": ego_state,
        "opp_state": opp_state,
        "ego_steer_buffer": np.asarray(core.sim.agents[0].steer_buffer, dtype=np.float64).copy(),
        "opp_steer_buffer": np.asarray(core.sim.agents[1].steer_buffer, dtype=np.float64).copy(),
        "actor_hidden_before": actor_hidden_before[0, 0].detach().cpu().numpy().astype(np.float32),
        "critic_hidden_before": critic_hidden_before[0, 0].detach().cpu().numpy().astype(np.float32),
        "actor_hidden_after": actor_hidden_after[0, 0].detach().cpu().numpy().astype(np.float32),
        "critic_hidden_after": critic_hidden_after[0, 0].detach().cpu().numpy().astype(np.float32),
        "actor_raw_action": np.asarray(raw_action, dtype=np.float32),
        "actor_executed_action": np.asarray(executed_action, dtype=np.float32),
        "opponent_executed_action": np.asarray(opponent_action, dtype=np.float32),
        "critic_value": float(critic_value),
        "reward_total": float(reward),
        "reward_components": np.asarray(reward_components, dtype=np.float64),
        "collisions": np.asarray(env._raw_observation["collisions"], dtype=bool),
        "terminated_after": bool(terminated),
        "truncated_after": bool(truncated),
        "action_applied": bool(action_applied),
        "terminal_post_step": bool(terminal_post_step),
    }


def run_suffix(env, observation, actor_hidden, critic_hidden):
    rows = {name: [] for name in TRACE_DTYPES}
    first_collision_step = None
    final_outcome = None
    terminated = False
    truncated = False
    last_info = None
    last_opponent_action = np.zeros(2, dtype=np.float32)
    original_opponent_action = env.opponent_controller.action

    def recorded_opponent_action(raw_observation):
        nonlocal last_opponent_action
        last_opponent_action = np.asarray(original_opponent_action(raw_observation), dtype=np.float32)
        return last_opponent_action

    env.opponent_controller.action = recorded_opponent_action
    try:
        while not terminated and not truncated:
            if env._episode_steps >= 800:
                raise RuntimeError("Snapshot suffix exceeded the 800-step contract")
            observation_before = np.asarray(observation, dtype=np.float32).copy()
            actor_before = actor_hidden.clone()
            critic_before = critic_hidden.clone()
            raw_action, critic_value, next_actor_hidden, next_critic_hidden = model_step(observation_before, actor_hidden, critic_hidden)
            executed_action = raw_action.copy()
            executed_action[0] = np.clip(executed_action[0], -0.52, 0.52)
            observation, reward, terminated, truncated, info = env.step(executed_action)
            observation = np.asarray(observation, dtype=np.float32)
            actor_hidden = next_actor_hidden
            critic_hidden = next_critic_hidden
            reward_components = (info["reward_progress"], info["reward_relative"], info["reward_collision"], info["reward_risk"])
            row = trace_row(env, observation_before, observation, actor_before, critic_before, actor_hidden, critic_hidden, raw_action, executed_action, last_opponent_action, critic_value, reward, reward_components, terminated, truncated, True, False)
            for name, value in row.items():
                rows[name].append(value)
            if info["ego_collision"] and first_collision_step is None:
                first_collision_step = int(env._episode_steps)
            if terminated or truncated:
                final_outcome = info["episode_outcome"]
                last_info = info
        zeros_action = np.zeros(2, dtype=np.float32)
        terminal_row = trace_row(env, observation, observation, actor_hidden, critic_hidden, actor_hidden, critic_hidden, zeros_action, zeros_action, zeros_action, 0.0, 0.0, np.zeros(4, dtype=np.float64), terminated, truncated, False, True)
        for name, value in terminal_row.items():
            rows[name].append(value)
    finally:
        del env.opponent_controller.__dict__["action"]
    arrays = {name: np.asarray(values, dtype=TRACE_DTYPES[name]) for name, values in rows.items()}
    if not last_info or final_outcome not in ("ego_collision", "overtake", "follow"):
        raise RuntimeError("Snapshot suffix did not reach a valid terminal outcome")
    return arrays, {
        "action_steps": int(len(arrays["time_s"]) - 1),
        "absolute_terminal_step": int(env._episode_steps),
        "first_ego_collision_step": first_collision_step,
        "outcome": final_outcome,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "episode_return": float(last_info["episode_return"]),
    }


def validate_trace(arrays):
    if set(arrays) != set(TRACE_DTYPES):
        raise RuntimeError("Snapshot trace schema changed")
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise RuntimeError("Snapshot trace arrays are not aligned")
    if not all(np.isfinite(value).all() for name, value in arrays.items() if value.dtype != np.bool_):
        raise RuntimeError("Snapshot trace contains non-finite values")
    applied = arrays["action_applied"]
    terminal = arrays["terminal_post_step"]
    if not np.array_equal(applied, np.arange(len(applied)) < len(applied) - 1) or int(terminal.sum()) != 1 or not bool(terminal[-1]):
        raise RuntimeError("Snapshot trace terminal row contract failed")
    if not bool(arrays["terminated_after"][-1] or arrays["truncated_after"][-1]):
        raise RuntimeError("Snapshot trace did not terminate or truncate")


def compare_traces(original, restored):
    maxima = {}
    exact = True
    for name in TRACE_DTYPES:
        left = original[name]
        right = restored[name]
        if left.shape != right.shape:
            maxima[name] = None
            exact = False
        elif left.dtype == np.bool_:
            equal = np.array_equal(left, right)
            maxima[name] = 0.0 if equal else None
            exact = exact and equal
        else:
            error = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
            maxima[name] = error
            exact = exact and error == 0.0
    return exact, maxima


def atomic_write_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"Snapshot output already exists: {path}")
    try:
        temporary.write_bytes(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def worker_initializer(actor_path, critic_path, hidden_scale):
    global WORKER_DEVICE, WORKER_ACTOR, WORKER_CRITIC
    for name, value in WORKER_ENV.items():
        os.environ[name] = value
    warnings.filterwarnings("ignore", message="Chosen integrator is RK4.*")
    torch.set_num_threads(1)
    WORKER_DEVICE = torch.device("cuda")
    WORKER_ACTOR = End2Race(hidden_scale=hidden_scale).to(WORKER_DEVICE)
    WORKER_ACTOR.load_state_dict(torch.load(actor_path, map_location=WORKER_DEVICE, weights_only=True), strict=True)
    WORKER_ACTOR.eval()
    WORKER_CRITIC = PrivilegeGRUCritic(WORKER_ACTOR).to(WORKER_DEVICE)
    WORKER_CRITIC.load_state_dict(torch.load(critic_path, map_location=WORKER_DEVICE, weights_only=True), strict=True)
    WORKER_CRITIC.eval()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_task(task):
    scenario = copy.deepcopy(task["scenario"])
    scenario["env_role"] = "collision"
    scenario["sampler_branch"] = "collision"
    scenario["opponent_speed_scale"] = float(scenario["opp_speedscale"])
    positions, initial_speeds = load_positions_and_speeds_from_params(scenario, "Austin")
    spec = EpisodeResetSpec(np.asarray(positions, dtype=np.float64), float(initial_speeds[0] * 0.9), scenario)
    env = make_environment(42, "Austin", privileged=True, speed_exploration_mode=CORRIDOR_TEMPORAL_EXPLORATION_MODE)()
    observation, _info = env.reset(options={EXTERNAL_RESET_OPTION: spec})
    hidden_size = WORKER_ACTOR.gru.hidden_size
    actor_hidden = torch.zeros((1, 1, hidden_size), dtype=torch.float32, device=WORKER_DEVICE)
    critic_hidden = torch.zeros((1, 1, hidden_size), dtype=torch.float32, device=WORKER_DEVICE)
    prefix_step = int(task["window"]["start_index"])
    try:
        while env._episode_steps < prefix_step:
            raw_action, _value, actor_hidden, critic_hidden = model_step(observation, actor_hidden, critic_hidden)
            executed_action = raw_action.copy()
            executed_action[0] = np.clip(executed_action[0], -0.52, 0.52)
            observation, _reward, terminated, truncated, _info = env.step(executed_action)
            if terminated or truncated:
                raise RuntimeError(f"Task terminated before frozen prefix: {task['episode_key']}")
        observation = np.asarray(observation, dtype=np.float32)
        snapshot = {
            "schema_version": 1,
            "environment": capture_environment(env),
            "observation": observation.copy(),
            "actor_hidden": actor_hidden.detach().cpu().numpy().astype(np.float32),
            "critic_hidden": critic_hidden.detach().cpu().numpy().astype(np.float32),
        }
        payload = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
        restored_snapshot = pickle.loads(payload)
        if set(restored_snapshot) != {"schema_version", "environment", "observation", "actor_hidden", "critic_hidden"}:
            raise RuntimeError("Top-level snapshot schema changed after pickle round-trip")
        original_arrays, original_result = run_suffix(env, observation, actor_hidden, critic_hidden)
        restore_environment(env, restored_snapshot["environment"])
        restored_observation = np.asarray(env._observation(env._raw_observation), dtype=np.float32)
        restored_actor_hidden = torch.as_tensor(restored_snapshot["actor_hidden"], device=WORKER_DEVICE)
        restored_critic_hidden = torch.as_tensor(restored_snapshot["critic_hidden"], device=WORKER_DEVICE)
        pre_observation_error = float(np.max(np.abs(restored_observation.astype(np.float64) - restored_snapshot["observation"].astype(np.float64))))
        actor_hidden_error = float(np.max(np.abs(restored_actor_hidden.detach().cpu().numpy().astype(np.float64) - restored_snapshot["actor_hidden"].astype(np.float64))))
        critic_hidden_error = float(np.max(np.abs(restored_critic_hidden.detach().cpu().numpy().astype(np.float64) - restored_snapshot["critic_hidden"].astype(np.float64))))
        restored_arrays, restored_result = run_suffix(env, restored_observation, restored_actor_hidden, restored_critic_hidden)
        validate_trace(original_arrays)
        validate_trace(restored_arrays)
        exact_trace, maxima = compare_traces(original_arrays, restored_arrays)
        exact_result = original_result == restored_result
        passed = pre_observation_error == 0.0 and actor_hidden_error == 0.0 and critic_hidden_error == 0.0 and exact_trace and exact_result
        snapshot_path = Path(task["snapshot_path"])
        original_path = Path(task["original_trace_path"])
        restored_path = Path(task["restored_trace_path"])
        atomic_write_bytes(snapshot_path, payload)
        save_numeric_npz(original_path, original_arrays)
        save_numeric_npz(restored_path, restored_arrays)
        return {
            "episode_key": task["episode_key"],
            "stratum": task["stratum"],
            "ego_idx": int(scenario["ego_idx"]),
            "opp_raceline": scenario["opp_raceline"],
            "prefix_step": prefix_step,
            "snapshot_bytes": len(payload),
            "pre_observation_max_abs_error": pre_observation_error,
            "actor_hidden_restore_max_abs_error": actor_hidden_error,
            "critic_hidden_restore_max_abs_error": critic_hidden_error,
            "trace_field_max_abs_error": maxima,
            "original_result": original_result,
            "restored_result": restored_result,
            "result_exact": exact_result,
            "trace_exact": exact_trace,
            "passed": passed,
        }
    finally:
        env.close()
        gc.collect()


if __name__ == "__main__":
    args = parse_arguments()
    args.output_dir = args.output_dir.resolve()
    args.gate_b_plan = args.gate_b_plan.resolve()
    args.actor_path = args.actor_path.resolve()
    args.critic_path = args.critic_path.resolve()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Round Z6-A requires CUDA")
    plan = build_plan(args)
    plan_path = write_frozen_plan(args, plan)
    print(f"PLAN={plan_path}")
    print(f"TASKS={len(plan['tasks'])}")
    if args.prepare_only:
        sys.exit(0)
    report_path = args.output_dir / "snapshot_noop_report.json"
    if report_path.exists():
        raise RuntimeError(f"Refusing to overwrite completed report: {report_path}")
    for name in ("snapshots", "original_traces", "restored_traces"):
        path = args.output_dir / name
        if path.exists():
            raise RuntimeError(f"Refusing to resume or overwrite partial snapshot outputs: {path}")
        path.mkdir(parents=True)
    tasks = []
    for source in plan["tasks"]:
        task = copy.deepcopy(source)
        key = task["episode_key"]
        task["snapshot_path"] = str(args.output_dir / "snapshots" / f"{key}.pkl")
        task["original_trace_path"] = str(args.output_dir / "original_traces" / f"{key}.npz")
        task["restored_trace_path"] = str(args.output_dir / "restored_traces" / f"{key}.npz")
        tasks.append(task)
    context = mp.get_context("spawn")
    completed = {}
    with context.Pool(args.workers, initializer=worker_initializer, initargs=(str(args.actor_path), str(args.critic_path), args.hidden_scale)) as pool:
        for index, result in enumerate(pool.imap_unordered(run_task, tasks), start=1):
            completed[result["episode_key"]] = result
            print(f"SNAPSHOT={index}/{len(tasks)} KEY={result['episode_key']} PASS={result['passed']}", flush=True)
    if set(completed) != {task["episode_key"] for task in tasks}:
        raise RuntimeError("Snapshot result keys do not match frozen plan")
    maxima = {name: 0.0 for name in TRACE_DTYPES}
    for result in completed.values():
        for name, value in result["trace_field_max_abs_error"].items():
            if value is None:
                maxima[name] = None
            elif maxima[name] is not None:
                maxima[name] = max(maxima[name], float(value))
    pass_count = sum(bool(result["passed"]) for result in completed.values())
    criteria = {
        "all_28_tasks_completed": len(completed) == 28,
        "all_28_tasks_passed": pass_count == 28,
        "all_trace_fields_bit_exact": all(value == 0.0 for value in maxima.values()),
        "all_results_exact": all(result["result_exact"] for result in completed.values()),
        "all_snapshot_round_trips_completed": len(list((args.output_dir / "snapshots").glob("*.pkl"))) == 28,
        "all_original_traces_present": len(list((args.output_dir / "original_traces").glob("*.npz"))) == 28,
        "all_restored_traces_present": len(list((args.output_dir / "restored_traces").glob("*.npz"))) == 28,
    }
    verdict = "pass_snapshot_mechanical_gate" if all(criteria.values()) else "fail_stop_prefix_reset_snapshot"
    report = {
        "schema_version": 1,
        "experiment_id": "prefix_reset_snapshot_gate",
        "gate": "Z6-A",
        "verdict": verdict,
        "summary": {
            "task_count": len(completed),
            "pass_count": pass_count,
            "failure_count": len(completed) - pass_count,
            "collision_count": sum(result["stratum"] == "collision" for result in completed.values()),
            "lost_overtake_count": sum(result["stratum"] == "lost_overtake" for result in completed.values()),
            "unique_ego_startpoints": len({result["ego_idx"] for result in completed.values()}),
            "maximum_trace_field_errors": maxima,
            "maximum_pre_observation_error": max(result["pre_observation_max_abs_error"] for result in completed.values()),
            "maximum_actor_hidden_restore_error": max(result["actor_hidden_restore_max_abs_error"] for result in completed.values()),
            "maximum_critic_hidden_restore_error": max(result["critic_hidden_restore_max_abs_error"] for result in completed.values()),
            "snapshot_bytes_total": sum(result["snapshot_bytes"] for result in completed.values()),
        },
        "criteria": criteria,
        "episodes": dict(sorted(completed.items())),
        "evidence_boundary": {
            "established": "serialized full-state restore exactly reproduces frozen U44 deterministic suffixes on all 28 consensus development tasks",
            "not_established": "current-network burn-in after actor updates, GAE/bootstrap correctness, sampling-density gain, or PPO effectiveness",
            "next_action": "run separately preregistered current-network burn-in and GAE semantics gate before any prefix-reset training",
        },
    }
    atomic_write_json(report_path, report)
    print(f"REPORT={report_path}")
    print(f"VERDICT={verdict}")
