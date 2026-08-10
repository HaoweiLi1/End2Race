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
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import End2Race
from ppo.env import CORRIDOR_TEMPORAL_EXPLORATION_MODE, EXTERNAL_RESET_OPTION, make_environment
from ppo.policy import BASELINE_EXPLORATION_MODE, END2RACE_OBSERVATION_SIZE, NOOP_SPEED_BOUND, STEERING_BOUND, End2RaceGRUPolicy, PrivilegeGRUCritic
from ppo.rollout import End2RaceRolloutBuffer
from ppo.scenarios import EpisodeResetSpec
from utils import atomic_write_json, load_positions_and_speeds_from_params, save_numeric_npz

WORKER_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
GAMMA = 0.999
GAE_LAMBDA = 0.995
BURN_IN_TOLERANCE = 5.0e-5
LOG_RATIO_TOLERANCE = 5.0e-5
WORKER_DEVICE = None
SOURCE_ACTOR = None
SOURCE_CRITIC = None
CURRENT_ACTOR = None
CURRENT_CRITIC = None


class ExplorationCapture:

    def __init__(self):
        self.last = None
        self.current_valid_by_timestep = None
        self.current_speed_log_stds = None

    def stage_exploration(self, speed_log_std, danger_gate, temporal_active, block_id, standard_residual, joint_active, joint_block_uid, joint_block_position, joint_prefix_step, joint_collision_source, joint_standard_residual):
        self.last = {
            "speed_log_std": np.asarray(speed_log_std, dtype=np.float32).copy(),
            "danger_gate": np.asarray(danger_gate, dtype=bool).copy(),
            "temporal_active": np.asarray(temporal_active, dtype=bool).copy(),
            "block_id": np.asarray(block_id, dtype=np.int64).copy(),
            "standard_residual": np.asarray(standard_residual, dtype=np.float32).copy(),
            "joint_active": np.asarray(joint_active, dtype=bool).copy(),
            "joint_block_uid": np.asarray(joint_block_uid, dtype=np.int64).copy(),
            "joint_block_position": np.asarray(joint_block_position, dtype=np.int64).copy(),
            "joint_prefix_step": np.asarray(joint_prefix_step, dtype=np.int64).copy(),
            "joint_collision_source": np.asarray(joint_collision_source, dtype=bool).copy(),
            "joint_standard_residual": np.asarray(joint_standard_residual, dtype=np.float32).copy(),
        }


class ZeroSplitRng:

    def integers(self, _high):
        return 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, default=Path("eval_results/prefix_reset_snapshot_gate"))
    parser.add_argument("--source-actor-path", type=Path, default=Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update44/actor.pth"))
    parser.add_argument("--source-critic-path", type=Path, default=Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update44/critic.pt"))
    parser.add_argument("--current-actor-path", type=Path, default=Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update45/actor.pth"))
    parser.add_argument("--current-critic-path", type=Path, default=Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update45/critic.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results/prefix_reset_semantics_gate"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-scale", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--residual-adjudication-only", action="store_true")
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
    snapshot_plan_path = args.snapshot_dir / "snapshot_gate_plan.json"
    snapshot_report_path = args.snapshot_dir / "snapshot_noop_report.json"
    snapshot_plan = json.loads(snapshot_plan_path.read_text(encoding="utf-8"))
    snapshot_report = json.loads(snapshot_report_path.read_text(encoding="utf-8"))
    if snapshot_report.get("verdict") != "pass_snapshot_mechanical_gate":
        raise RuntimeError("Z6-B requires a passed Z6-A snapshot report")
    tasks = [copy.deepcopy(task) for task in snapshot_plan["tasks"]]
    tasks.sort(key=lambda task: task["episode_key"])
    if len(tasks) != 28 or sum(int(task["window"]["start_index"]) for task in tasks) != 9589:
        raise RuntimeError("Frozen Z6-A task or prefix-step contract changed")
    if len({task["episode_key"] for task in tasks}) != 28 or len({task["scenario"]["ego_idx"] for task in tasks}) != 21:
        raise RuntimeError("Frozen Z6-A task identity changed")
    for task in tasks:
        key = task["episode_key"]
        snapshot_path = args.snapshot_dir / "snapshots" / f"{key}.pkl"
        suffix_path = args.snapshot_dir / "original_traces" / f"{key}.npz"
        if not snapshot_path.is_file() or not suffix_path.is_file():
            raise FileNotFoundError(f"Missing Z6-A input for {key}")
        task["snapshot_path"] = str(snapshot_path)
        task["suffix_path"] = str(suffix_path)
    return {
        "schema_version": 1,
        "experiment_id": "prefix_reset_semantics_gate",
        "gate": "Z6-B",
        "status": "frozen_before_execution",
        "inputs": {
            "snapshot_plan_sha256": sha256_file(snapshot_plan_path),
            "snapshot_report_sha256": sha256_file(snapshot_report_path),
            "source_actor_path": str(args.source_actor_path),
            "source_actor_sha256": sha256_file(args.source_actor_path),
            "source_critic_path": str(args.source_critic_path),
            "source_critic_sha256": sha256_file(args.source_critic_path),
            "current_actor_path": str(args.current_actor_path),
            "current_actor_sha256": sha256_file(args.current_actor_path),
            "current_critic_path": str(args.current_critic_path),
            "current_critic_sha256": sha256_file(args.current_critic_path),
        },
        "task_contract": {
            "map": "Austin",
            "task_count": 28,
            "prefix_rows": 9589,
            "source_network": "U44",
            "current_network": "U45",
            "prefix_observation_size": 381,
            "recurrent_input_size": 361,
        },
        "admission_contract": {
            "source_snapshot_max_abs_error": 0.0,
            "fast_burn_in_tolerance": BURN_IN_TOLERANCE,
            "gae_return_tolerance": 1.0e-6,
            "log_ratio_tolerance": LOG_RATIO_TOLERANCE,
            "structured_hold_steps": 50,
            "failure_action": "stop current prefix-reset implementation",
            "pass_action": "preregister a separate Austin-only training-density gate; do not start formal PPO",
        },
        "tasks": tasks,
        "source_snapshot": source_snapshot(),
    }


def write_frozen_plan(args, plan):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "semantics_gate_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise RuntimeError("Existing Z6-B plan differs from current frozen inputs")
        return plan_path
    if any(args.output_dir.iterdir()):
        raise RuntimeError(f"Output directory must be empty before plan creation: {args.output_dir}")
    atomic_write_json(plan_path, plan)
    return plan_path


def maximum_error(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return None
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def model_step(actor, critic, observation, actor_hidden, critic_hidden):
    full = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=WORKER_DEVICE).reshape(1, -1)
    lidar = full[:, :360].unsqueeze(1)
    previous_speed = full[:, 360:361].unsqueeze(1)
    privileged = full[:, 361:]
    with torch.no_grad():
        action_sequence, next_actor_hidden = actor(lidar, previous_speed, actor_hidden)
        value, next_critic_hidden = critic.step(lidar, previous_speed, critic_hidden, privileged)
    return action_sequence[0, -1], value[0, 0], next_actor_hidden, next_critic_hidden


def reference_burn_in(actor, critic, prefix):
    hidden_size = actor.gru.hidden_size
    actor_hidden = torch.zeros((1, 1, hidden_size), dtype=torch.float32, device=WORKER_DEVICE)
    critic_hidden = torch.zeros((1, 1, hidden_size), dtype=torch.float32, device=WORKER_DEVICE)
    for observation in prefix:
        _action, _value, actor_hidden, critic_hidden = model_step(actor, critic, observation, actor_hidden, critic_hidden)
    return actor_hidden, critic_hidden


def sequence_burn_in(actor, critic, prefix):
    hidden_size = actor.gru.hidden_size
    actor_hidden = torch.zeros((1, 1, hidden_size), dtype=torch.float32, device=WORKER_DEVICE)
    critic_hidden = torch.zeros((1, 1, hidden_size), dtype=torch.float32, device=WORKER_DEVICE)
    if len(prefix) == 0:
        return actor_hidden, critic_hidden
    full = torch.as_tensor(np.asarray(prefix, dtype=np.float32), device=WORKER_DEVICE).unsqueeze(0)
    lidar = full[:, :, :360]
    previous_speed = full[:, :, 360:361]
    privileged = full[:, -1, 361:]
    with torch.no_grad():
        _actions, actor_hidden = actor(lidar, previous_speed, actor_hidden)
        _value, critic_hidden = critic.step(lidar, previous_speed, critic_hidden, privileged)
    return actor_hidden, critic_hidden


def load_actor_critic(actor_path, critic_path, hidden_scale):
    actor = End2Race(hidden_scale=hidden_scale).to(WORKER_DEVICE)
    actor.load_state_dict(torch.load(actor_path, map_location=WORKER_DEVICE, weights_only=True), strict=True)
    actor.eval()
    critic = PrivilegeGRUCritic(actor).to(WORKER_DEVICE)
    critic.load_state_dict(torch.load(critic_path, map_location=WORKER_DEVICE, weights_only=True), strict=True)
    critic.eval()
    return actor, critic


def worker_initializer(source_actor_path, source_critic_path, current_actor_path, current_critic_path, hidden_scale):
    global WORKER_DEVICE, SOURCE_ACTOR, SOURCE_CRITIC, CURRENT_ACTOR, CURRENT_CRITIC
    for name, value in WORKER_ENV.items():
        os.environ[name] = value
    warnings.filterwarnings("ignore", message="Chosen integrator is RK4.*")
    torch.set_num_threads(1)
    WORKER_DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    SOURCE_ACTOR, SOURCE_CRITIC = load_actor_critic(source_actor_path, source_critic_path, hidden_scale)
    CURRENT_ACTOR, CURRENT_CRITIC = load_actor_critic(current_actor_path, current_critic_path, hidden_scale)


def run_task(task):
    scenario = copy.deepcopy(task["scenario"])
    scenario["env_role"] = "collision"
    scenario["sampler_branch"] = "collision"
    scenario["opponent_speed_scale"] = float(scenario["opp_speedscale"])
    positions, initial_speeds = load_positions_and_speeds_from_params(scenario, "Austin")
    spec = EpisodeResetSpec(np.asarray(positions, dtype=np.float64), float(initial_speeds[0] * 0.9), scenario)
    env = make_environment(42, "Austin", privileged=True, speed_exploration_mode=CORRIDOR_TEMPORAL_EXPLORATION_MODE)()
    observation, _info = env.reset(options={EXTERNAL_RESET_OPTION: spec})
    prefix_step = int(task["window"]["start_index"])
    prefix = []
    try:
        torch.backends.cudnn.allow_tf32 = True
        source_actor_hidden, source_critic_hidden = reference_burn_in(SOURCE_ACTOR, SOURCE_CRITIC, np.empty((0, 381), dtype=np.float32))
        while env._episode_steps < prefix_step:
            prefix.append(np.asarray(observation, dtype=np.float32).copy())
            raw_action, _value, source_actor_hidden, source_critic_hidden = model_step(SOURCE_ACTOR, SOURCE_CRITIC, observation, source_actor_hidden, source_critic_hidden)
            executed_action = raw_action.detach().cpu().numpy().astype(np.float32)
            executed_action[0] = np.clip(executed_action[0], -STEERING_BOUND, STEERING_BOUND)
            observation, _reward, terminated, truncated, _info = env.step(executed_action)
            if terminated or truncated:
                raise RuntimeError(f"Task terminated before frozen prefix: {task['episode_key']}")
        prefix = np.asarray(prefix, dtype=np.float32).reshape(prefix_step, 381)
        window_observation = np.asarray(observation, dtype=np.float32)
        with Path(task["snapshot_path"]).open("rb") as stream:
            snapshot = pickle.load(stream)
        source_observation_error = maximum_error(window_observation, snapshot["observation"])
        source_actor_error = maximum_error(source_actor_hidden.detach().cpu().numpy(), snapshot["actor_hidden"])
        source_critic_error = maximum_error(source_critic_hidden.detach().cpu().numpy(), snapshot["critic_hidden"])
        source_fast_actor_hidden, source_fast_critic_hidden = sequence_burn_in(SOURCE_ACTOR, SOURCE_CRITIC, prefix)
        source_reference_action, source_reference_value, _next_actor, _next_critic = model_step(SOURCE_ACTOR, SOURCE_CRITIC, window_observation, source_actor_hidden, source_critic_hidden)
        source_fast_action, source_fast_value, _next_actor, _next_critic = model_step(SOURCE_ACTOR, SOURCE_CRITIC, window_observation, source_fast_actor_hidden, source_fast_critic_hidden)

        torch.backends.cudnn.allow_tf32 = False
        current_actor_hidden, current_critic_hidden = reference_burn_in(CURRENT_ACTOR, CURRENT_CRITIC, prefix)
        current_fast_actor_hidden, current_fast_critic_hidden = sequence_burn_in(CURRENT_ACTOR, CURRENT_CRITIC, prefix)
        current_reference_action, current_reference_value, _next_actor, _next_critic = model_step(CURRENT_ACTOR, CURRENT_CRITIC, window_observation, current_actor_hidden, current_critic_hidden)
        current_fast_action, current_fast_value, _next_actor, _next_critic = model_step(CURRENT_ACTOR, CURRENT_CRITIC, window_observation, current_fast_actor_hidden, current_fast_critic_hidden)

        errors = {
            "source_observation": source_observation_error,
            "source_actor_hidden": source_actor_error,
            "source_critic_hidden": source_critic_error,
            "source_fast_actor_hidden": maximum_error(source_fast_actor_hidden.detach().cpu().numpy(), source_actor_hidden.detach().cpu().numpy()),
            "source_fast_critic_hidden": maximum_error(source_fast_critic_hidden.detach().cpu().numpy(), source_critic_hidden.detach().cpu().numpy()),
            "source_fast_action": maximum_error(source_fast_action.detach().cpu().numpy(), source_reference_action.detach().cpu().numpy()),
            "source_fast_value": abs(float(source_fast_value.item()) - float(source_reference_value.item())),
            "current_fast_actor_hidden": maximum_error(current_fast_actor_hidden.detach().cpu().numpy(), current_actor_hidden.detach().cpu().numpy()),
            "current_fast_critic_hidden": maximum_error(current_fast_critic_hidden.detach().cpu().numpy(), current_critic_hidden.detach().cpu().numpy()),
            "current_fast_action": maximum_error(current_fast_action.detach().cpu().numpy(), current_reference_action.detach().cpu().numpy()),
            "current_fast_value": abs(float(current_fast_value.item()) - float(current_reference_value.item())),
        }
        stale_differences = {
            "actor_hidden": maximum_error(current_actor_hidden.detach().cpu().numpy(), np.asarray(snapshot["actor_hidden"])),
            "critic_hidden": maximum_error(current_critic_hidden.detach().cpu().numpy(), np.asarray(snapshot["critic_hidden"])),
            "action": maximum_error(current_reference_action.detach().cpu().numpy(), source_reference_action.detach().cpu().numpy()),
            "value": abs(float(current_reference_value.item()) - float(source_reference_value.item())),
        }
        prefix_path = Path(task["prefix_output_path"])
        save_numeric_npz(prefix_path, {
            "prefix_observations": prefix,
            "window_observation": window_observation,
            "source_actor_hidden": source_actor_hidden.detach().cpu().numpy().astype(np.float32),
            "source_critic_hidden": source_critic_hidden.detach().cpu().numpy().astype(np.float32),
            "current_actor_hidden": current_actor_hidden.detach().cpu().numpy().astype(np.float32),
            "current_critic_hidden": current_critic_hidden.detach().cpu().numpy().astype(np.float32),
            "current_fast_actor_hidden": current_fast_actor_hidden.detach().cpu().numpy().astype(np.float32),
            "current_fast_critic_hidden": current_fast_critic_hidden.detach().cpu().numpy().astype(np.float32),
            "current_reference_action": current_reference_action.detach().cpu().numpy().astype(np.float32),
            "current_fast_action": current_fast_action.detach().cpu().numpy().astype(np.float32),
            "current_reference_value": np.asarray([float(current_reference_value.item())], dtype=np.float32),
            "current_fast_value": np.asarray([float(current_fast_value.item())], dtype=np.float32),
        })
        source_exact = errors["source_observation"] == 0.0 and errors["source_actor_hidden"] == 0.0 and errors["source_critic_hidden"] == 0.0
        fast_pass = all(errors[name] <= BURN_IN_TOLERANCE for name in errors if name.startswith("source_fast_") or name.startswith("current_fast_"))
        return {
            "episode_key": task["episode_key"],
            "prefix_step": prefix_step,
            "source_exact": source_exact,
            "fast_pass": fast_pass,
            "errors": errors,
            "stale_differences": stale_differences,
        }
    finally:
        env.close()
        gc.collect()


def make_policy(actor_path, critic_path, hidden_scale, mode, device):
    observation_space = spaces.Box(low=np.full((381,), -np.inf, dtype=np.float32), high=np.full((381,), np.inf, dtype=np.float32), dtype=np.float32)
    action_space = spaces.Box(low=np.asarray((-STEERING_BOUND, -NOOP_SPEED_BOUND), dtype=np.float32), high=np.asarray((STEERING_BOUND, NOOP_SPEED_BOUND), dtype=np.float32), dtype=np.float32)
    policy = End2RaceGRUPolicy(observation_space, action_space, lambda _remaining: 1.0, checkpoint_path=actor_path, hidden_scale=hidden_scale, critic_variant="privilege_gru", speed_exploration_mode=mode).to(device)
    policy.value_net.load_state_dict(torch.load(critic_path, map_location=device, weights_only=True), strict=True)
    policy.set_training_mode(False)
    capture = ExplorationCapture()
    policy._end2race_rollout_buffer = capture
    return policy, capture


def load_current_inputs(plan, output_dir, steps):
    observations = []
    actor_hidden = []
    critic_hidden = []
    for task in plan["tasks"]:
        key = task["episode_key"]
        with np.load(output_dir / "prefixes" / f"{key}.npz") as prefix:
            actor_hidden.append(prefix["current_actor_hidden"][0, 0])
            critic_hidden.append(prefix["current_critic_hidden"][0, 0])
        with np.load(task["suffix_path"]) as suffix:
            if len(suffix["observation_before"]) < steps:
                raise RuntimeError(f"Suffix is shorter than the exploration contract: {key}")
            observations.append(suffix["observation_before"][:steps])
    return np.asarray(observations, dtype=np.float32), np.asarray(actor_hidden, dtype=np.float32), np.asarray(critic_hidden, dtype=np.float32)


def run_log_probability_gate(plan, args, mode, steps, seed):
    device = torch.device("cuda")
    policy, capture = make_policy(args.current_actor_path, args.current_critic_path, args.hidden_scale, mode, device)
    observations, actor_hidden, critic_hidden = load_current_inputs(plan, args.output_dir, steps)
    actor_initial = torch.as_tensor(actor_hidden, device=device).unsqueeze(0)
    critic_initial = torch.as_tensor(critic_hidden, device=device).unsqueeze(0)
    zero_actor = torch.zeros_like(actor_initial)
    zero_critic = torch.zeros_like(critic_initial)
    states = RNNStates((actor_initial.clone(), zero_actor), (critic_initial.clone(), zero_critic))
    initial_states = RNNStates((actor_initial.clone(), zero_actor.clone()), (critic_initial.clone(), zero_critic.clone()))
    recurrent_resets = torch.zeros(len(plan["tasks"]), dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    actions = []
    old_log_probs = []
    speed_log_stds = []
    temporal_active = []
    block_ids = []
    residuals = []
    internal_temporal_noise = []
    first_hidden_after = None
    for step in range(steps):
        step_observation = torch.as_tensor(observations[:, step], device=device)
        if mode == CORRIDOR_TEMPORAL_EXPLORATION_MODE:
            policy.prepare_rollout_exploration(np.ones(len(plan["tasks"]), dtype=bool) if step == 0 else np.zeros(len(plan["tasks"]), dtype=bool), np.ones(len(plan["tasks"]), dtype=bool) if step == 0 else np.zeros(len(plan["tasks"]), dtype=bool))
        with torch.no_grad():
            action, _value, log_prob, states = policy.forward(step_observation, states, recurrent_resets)
        if first_hidden_after is None:
            first_hidden_after = states.pi[0].clone()
        actions.append(action.detach().cpu().numpy())
        old_log_probs.append(log_prob.detach().cpu().numpy())
        speed_log_stds.append(capture.last["speed_log_std"])
        temporal_active.append(capture.last["temporal_active"])
        block_ids.append(capture.last["block_id"])
        residuals.append(capture.last["standard_residual"])
        if mode == CORRIDOR_TEMPORAL_EXPLORATION_MODE:
            internal_temporal_noise.append(policy._temporal_speed_noise.detach().cpu().numpy().copy())
    actions = np.asarray(actions, dtype=np.float32)
    old_log_probs = np.asarray(old_log_probs, dtype=np.float32)
    speed_log_stds = np.asarray(speed_log_stds, dtype=np.float32)
    temporal_active = np.asarray(temporal_active, dtype=bool)
    block_ids = np.asarray(block_ids, dtype=np.int64)
    residuals = np.asarray(residuals, dtype=np.float32)
    sequence_observations = torch.as_tensor(observations.reshape(len(plan["tasks"]) * steps, 381), device=device)
    sequence_actions = torch.as_tensor(actions.transpose(1, 0, 2).reshape(len(plan["tasks"]) * steps, 2), device=device)
    sequence_old_log_probs = torch.as_tensor(old_log_probs.transpose(1, 0).reshape(-1), device=device)
    sequence_resets = torch.zeros(len(plan["tasks"]) * steps, dtype=torch.float32, device=device)
    capture.current_valid_by_timestep = tuple(tuple(True for _task in plan["tasks"]) for _step in range(steps))
    capture.current_speed_log_stds = torch.as_tensor(speed_log_stds.transpose(1, 0).reshape(-1), device=device)
    with torch.no_grad():
        replay_log_prob, _entropy = policy.evaluate_actor_actions(sequence_observations, sequence_actions, initial_states, sequence_resets, collection_equivalent=True)
        batched_log_prob, _entropy = policy.evaluate_actor_actions(sequence_observations, sequence_actions, initial_states, sequence_resets, collection_equivalent=False)
        direct_mean, direct_states = policy._actor_forward(torch.as_tensor(observations[:, 0], device=device), initial_states.pi, recurrent_resets)
    log_ratio = replay_log_prob - sequence_old_log_probs
    batched_log_ratio = batched_log_prob - sequence_old_log_probs
    result = {
        "mode": mode,
        "steps": steps,
        "transition_count": int(len(plan["tasks"]) * steps),
        "maximum_collection_equivalent_abs_log_ratio": float(torch.abs(log_ratio).max().cpu().item()),
        "maximum_collection_equivalent_abs_ratio_minus_one": float(torch.abs(torch.exp(log_ratio) - 1.0).max().cpu().item()),
        "maximum_batched_abs_log_ratio": float(torch.abs(batched_log_ratio).max().cpu().item()),
        "maximum_batched_abs_ratio_minus_one": float(torch.abs(torch.exp(batched_log_ratio) - 1.0).max().cpu().item()),
        "exploration_reset_actor_hidden_error": float(torch.abs(first_hidden_after - direct_states[0]).max().cpu().item()),
        "nonzero_initial_hidden_count": int(np.any(actor_hidden != 0.0, axis=1).sum()),
    }
    if mode == CORRIDOR_TEMPORAL_EXPLORATION_MODE:
        internal_temporal_noise = np.asarray(internal_temporal_noise, dtype=np.float32)
        policy.prepare_rollout_exploration(np.ones(len(plan["tasks"]), dtype=bool), np.ones(len(plan["tasks"]), dtype=bool))
        with torch.no_grad():
            _action, _value, _log_prob, _states = policy.forward(torch.as_tensor(observations[:, 0], device=device), initial_states, recurrent_resets)
        revisit_residual = capture.last["standard_residual"]
        result.update({
            "first_50_all_active": bool(temporal_active[:50].all()),
            "step_51_all_inactive": bool((~temporal_active[50]).all()),
            "first_50_positive_same_block": bool((block_ids[:50] > 0).all() and np.all(block_ids[:50] == block_ids[0:1])),
            "step_51_zero_block": bool((block_ids[50] == 0).all()),
            "first_50_max_residual_error": float(np.max(np.abs(residuals[:50].astype(np.float64) - residuals[0:1].astype(np.float64)))),
            "first_50_max_internal_noise_error": float(np.max(np.abs(internal_temporal_noise[:50].astype(np.float64) - internal_temporal_noise[0:1].astype(np.float64)))),
            "revisit_residual_min_abs_difference": float(np.min(np.abs(revisit_residual.astype(np.float64) - residuals[0].astype(np.float64)))),
        })
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return result


def manual_gae(rewards, values, episode_starts, last_value, done):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.float32(0.0)
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_non_terminal = np.float32(1.0 - float(done))
            next_value = np.float32(last_value)
        else:
            next_non_terminal = np.float32(1.0 - episode_starts[step + 1])
            next_value = np.float32(values[step + 1])
        delta = np.float32(rewards[step] + GAMMA * next_value * next_non_terminal - values[step])
        last_gae = np.float32(delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae)
        advantages[step] = last_gae
    return advantages, np.asarray(advantages + values, dtype=np.float32)


def fill_semantic_buffer(buffer, rewards, values, episode_starts, recurrent_resets, hidden_values):
    hidden_size = buffer.hidden_state_shape[-1]
    for step in range(len(rewards)):
        hidden = torch.full((1, 1, hidden_size), float(hidden_values[step]), dtype=torch.float32)
        states = RNNStates((hidden, torch.zeros_like(hidden)), (hidden + 10.0, torch.zeros_like(hidden)))
        buffer.stage_exploration(speed_log_std=np.asarray([np.log(0.15)], dtype=np.float32), danger_gate=np.asarray([False]), temporal_active=np.asarray([False]), block_id=np.asarray([0]), standard_residual=np.asarray([0.0], dtype=np.float32), joint_active=np.asarray([False]), joint_block_uid=np.asarray([0], dtype=np.int64), joint_block_position=np.asarray([-1], dtype=np.int64), joint_prefix_step=np.asarray([0], dtype=np.int64), joint_collision_source=np.asarray([False]), joint_standard_residual=np.zeros((1, 2), dtype=np.float32))
        if recurrent_resets is not None:
            buffer.stage_recurrent_resets(np.asarray([recurrent_resets[step]], dtype=bool))
        buffer.add(np.zeros((1, 381), dtype=np.float32), np.zeros((1, 2), dtype=np.float32), np.asarray([rewards[step]], dtype=np.float32), np.asarray([episode_starts[step]], dtype=bool), torch.as_tensor([values[step]], dtype=torch.float32), torch.zeros(1, dtype=torch.float32), lstm_states=states)


def run_buffer_gae_gate(hidden_size):
    observation_space = spaces.Box(low=np.full((381,), -np.inf, dtype=np.float32), high=np.full((381,), np.inf, dtype=np.float32), dtype=np.float32)
    action_space = spaces.Box(low=np.asarray((-STEERING_BOUND, -NOOP_SPEED_BOUND), dtype=np.float32), high=np.asarray((STEERING_BOUND, NOOP_SPEED_BOUND), dtype=np.float32), dtype=np.float32)
    rewards = np.asarray([0.2, 0.1, -2.0, 0.3, -0.2 + GAMMA * 0.7, 0.4], dtype=np.float32)
    values = np.asarray([0.5, 0.4, 0.3, 0.6, 0.2, 0.1], dtype=np.float32)
    episode_starts = np.asarray([True, False, False, True, False, True], dtype=bool)
    recurrent_resets = np.zeros(6, dtype=bool)
    hidden_values = np.asarray([1.0, 1.1, 1.2, 2.0, 2.1, 3.0], dtype=np.float32)
    shape = (6, 1, 1, hidden_size)
    buffer = End2RaceRolloutBuffer(6, observation_space, action_space, shape, "cpu", gamma=GAMMA, gae_lambda=GAE_LAMBDA, n_envs=1, store_independent_gru_hidden=True)
    fill_semantic_buffer(buffer, rewards, values, episode_starts, recurrent_resets, hidden_values)
    expected_advantages, expected_returns = manual_gae(rewards, values, episode_starts.astype(np.float32), 0.8, False)
    buffer.compute_returns_and_advantage(torch.as_tensor([0.8], dtype=torch.float32), np.asarray([False]))
    advantage_error = maximum_error(buffer.advantages[:, 0], expected_advantages)
    return_error = maximum_error(buffer.returns[:, 0], expected_returns)
    boundary_copy = buffer.episode_starts.copy()
    recurrent_copy = buffer.recurrent_resets.copy()
    samples = next(buffer.get(6, rng=ZeroSplitRng()))
    sequence_starts = list(np.asarray(buffer.seq_start_indices, dtype=np.int64))
    sampled_actor_hidden = samples.lstm_states.pi[0][0, :, 0].detach().cpu().numpy()
    sampled_critic_hidden = samples.lstm_states.vf[0][0, :, 0].detach().cpu().numpy()

    default_buffer = End2RaceRolloutBuffer(6, observation_space, action_space, shape, "cpu", gamma=GAMMA, gae_lambda=GAE_LAMBDA, n_envs=1, store_independent_gru_hidden=True)
    fill_semantic_buffer(default_buffer, rewards, values, episode_starts, None, hidden_values)
    default_equal = bool(np.array_equal(default_buffer.recurrent_resets, default_buffer.episode_starts))
    return {
        "advantage_max_abs_error": advantage_error,
        "return_max_abs_error": return_error,
        "boundary_mask": boundary_copy[:, 0].astype(bool).tolist(),
        "recurrent_reset_mask": recurrent_copy[:, 0].astype(bool).tolist(),
        "sequence_start_indices": [int(value) for value in sequence_starts],
        "sampled_replay_reset_dtype": str(samples.episode_starts.dtype),
        "sampled_replay_reset_all_false": bool(torch.count_nonzero(samples.episode_starts).item() == 0),
        "sampled_actor_initial_hidden": [float(value) for value in sampled_actor_hidden],
        "sampled_critic_initial_hidden": [float(value) for value in sampled_critic_hidden],
        "default_recurrent_reset_equals_episode_start": default_equal,
        "prefix_transition_count_in_buffer": 0,
        "suffix_transition_count_in_buffer": 6,
    }


def run_residual_adjudication(args):
    plan_path = args.output_dir / "semantics_gate_plan.json"
    report_path = args.output_dir / "semantics_gate_report.json"
    output_path = args.output_dir / "residual_measurement_adjudication.json"
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite residual adjudication: {output_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    original = json.loads(report_path.read_text(encoding="utf-8"))
    if plan.get("gate") != "Z6-B" or len(plan.get("tasks", [])) != 28 or plan["inputs"]["current_actor_sha256"] != sha256_file(args.current_actor_path) or plan["inputs"]["current_critic_sha256"] != sha256_file(args.current_critic_path):
        raise RuntimeError("Residual adjudication inputs differ from the frozen Z6-B plan")
    prior_other_criteria = all(value for name, value in original["criteria"].items() if name != "corridor_restart_and_hold_passed")
    prior_corridor = original["corridor_log_probability"]
    prior_only_residual_failed = not original["criteria"]["corridor_restart_and_hold_passed"] and prior_corridor["first_50_all_active"] and prior_corridor["step_51_all_inactive"] and prior_corridor["first_50_positive_same_block"] and prior_corridor["step_51_zero_block"] and prior_corridor["revisit_residual_min_abs_difference"] > 0.0 and 0.0 < prior_corridor["first_50_max_residual_error"] <= LOG_RATIO_TOLERANCE
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    rerun = run_log_probability_gate(plan, args, CORRIDOR_TEMPORAL_EXPLORATION_MODE, 51, 8602)
    criteria = {
        "prior_non_residual_criteria_passed": prior_other_criteria,
        "prior_only_reconstructed_residual_exactness_failed": prior_only_residual_failed,
        "frozen_task_and_transition_contract_unchanged": len(plan["tasks"]) == 28 and rerun["transition_count"] == 1428,
        "internal_temporal_noise_bit_exact_for_50_steps": rerun["first_50_max_internal_noise_error"] == 0.0,
        "reconstructed_residual_within_frozen_likelihood_tolerance": rerun["first_50_max_residual_error"] <= LOG_RATIO_TOLERANCE,
        "collection_equivalent_likelihood_within_frozen_tolerance": rerun["maximum_collection_equivalent_abs_log_ratio"] <= LOG_RATIO_TOLERANCE and rerun["maximum_collection_equivalent_abs_ratio_minus_one"] <= LOG_RATIO_TOLERANCE,
        "restart_hold_and_release_contract_passed": rerun["first_50_all_active"] and rerun["step_51_all_inactive"] and rerun["first_50_positive_same_block"] and rerun["step_51_zero_block"] and rerun["revisit_residual_min_abs_difference"] > 0.0,
    }
    verdict = "pass_prefix_reset_semantics_after_measurement_adjudication" if all(criteria.values()) else "fail_stop_prefix_reset_semantics"
    adjudication = {
        "schema_version": 1,
        "experiment_id": "prefix_reset_semantics_gate",
        "gate": "Z6-BR",
        "verdict": verdict,
        "criteria": criteria,
        "original_machine_verdict": original["verdict"],
        "original_reconstructed_residual_error": prior_corridor["first_50_max_residual_error"],
        "rerun_corridor_log_probability": rerun,
        "evidence_boundary": {
            "established": "the only strict Z6-B failure was non-causal inverse-arithmetic telemetry; internal temporal noise, block lifetime, and replay likelihood satisfy the frozen mechanism contract",
            "not_established": "prefix-reset PPO training effectiveness, sampling ratio, role distribution, throughput, or actor performance",
            "next_action": "preregister a separate Austin-only training-density gate; do not start formal PPO",
        },
    }
    atomic_write_json(output_path, adjudication)
    print(f"ADJUDICATION={output_path}")
    print(f"VERDICT={verdict}")


if __name__ == "__main__":
    args = parse_arguments()
    args.snapshot_dir = args.snapshot_dir.resolve()
    args.source_actor_path = args.source_actor_path.resolve()
    args.source_critic_path = args.source_critic_path.resolve()
    args.current_actor_path = args.current_actor_path.resolve()
    args.current_critic_path = args.current_critic_path.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Round Z6-B requires CUDA")
    if args.residual_adjudication_only:
        run_residual_adjudication(args)
        sys.exit(0)
    plan = build_plan(args)
    plan_path = write_frozen_plan(args, plan)
    print(f"PLAN={plan_path}")
    print(f"TASKS={len(plan['tasks'])}")
    if args.prepare_only:
        sys.exit(0)
    report_path = args.output_dir / "semantics_gate_report.json"
    if report_path.exists():
        raise RuntimeError(f"Refusing to overwrite completed report: {report_path}")
    prefix_dir = args.output_dir / "prefixes"
    if prefix_dir.exists():
        raise RuntimeError(f"Refusing to resume or overwrite partial prefix outputs: {prefix_dir}")
    prefix_dir.mkdir(parents=True)
    tasks = []
    for source in plan["tasks"]:
        task = copy.deepcopy(source)
        task["prefix_output_path"] = str(prefix_dir / f"{task['episode_key']}.npz")
        tasks.append(task)
    context = mp.get_context("spawn")
    completed = {}
    with context.Pool(args.workers, initializer=worker_initializer, initargs=(str(args.source_actor_path), str(args.source_critic_path), str(args.current_actor_path), str(args.current_critic_path), args.hidden_scale)) as pool:
        for index, result in enumerate(pool.imap_unordered(run_task, tasks), start=1):
            completed[result["episode_key"]] = result
            print(f"BURNIN={index}/{len(tasks)} KEY={result['episode_key']} SOURCE_EXACT={result['source_exact']} FAST={result['fast_pass']}", flush=True)
    if set(completed) != {task["episode_key"] for task in tasks}:
        raise RuntimeError("Z6-B result keys do not match frozen plan")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    baseline_log_probability = run_log_probability_gate(plan, args, BASELINE_EXPLORATION_MODE, 1, 8601)
    corridor_log_probability = run_log_probability_gate(plan, args, CORRIDOR_TEMPORAL_EXPLORATION_MODE, 51, 8602)
    buffer_gae = run_buffer_gae_gate(CURRENT_ACTOR.gru.hidden_size if CURRENT_ACTOR is not None else 1680)
    maximum_errors = {name: max(float(result["errors"][name]) for result in completed.values()) for name in next(iter(completed.values()))["errors"]}
    nonzero_prefix = [result for result in completed.values() if result["prefix_step"] > 0]
    stale_nonzero = any(any(float(value) > 0.0 for value in result["stale_differences"].values()) for result in nonzero_prefix)
    source_exact = all(result["source_exact"] for result in completed.values())
    fast_pass = all(result["fast_pass"] for result in completed.values())
    log_probability_pass = all(result["maximum_collection_equivalent_abs_log_ratio"] <= LOG_RATIO_TOLERANCE and result["maximum_collection_equivalent_abs_ratio_minus_one"] <= LOG_RATIO_TOLERANCE and result["exploration_reset_actor_hidden_error"] == 0.0 for result in (baseline_log_probability, corridor_log_probability))
    corridor_pass = corridor_log_probability["first_50_all_active"] and corridor_log_probability["step_51_all_inactive"] and corridor_log_probability["first_50_positive_same_block"] and corridor_log_probability["step_51_zero_block"] and corridor_log_probability["first_50_max_residual_error"] == 0.0 and corridor_log_probability["revisit_residual_min_abs_difference"] > 0.0
    buffer_pass = buffer_gae["advantage_max_abs_error"] <= 1.0e-6 and buffer_gae["return_max_abs_error"] <= 1.0e-6 and buffer_gae["sequence_start_indices"] == [0, 3, 5] and buffer_gae["sampled_replay_reset_dtype"] == "torch.float32" and buffer_gae["sampled_replay_reset_all_false"] and buffer_gae["sampled_actor_initial_hidden"] == [1.0, 2.0, 3.0] and buffer_gae["sampled_critic_initial_hidden"] == [11.0, 12.0, 13.0] and buffer_gae["default_recurrent_reset_equals_episode_start"] and buffer_gae["prefix_transition_count_in_buffer"] == 0
    criteria = {
        "all_28_tasks_completed": len(completed) == 28,
        "all_source_snapshots_exact": source_exact,
        "all_prefix_rows_present": sum(result["prefix_step"] for result in completed.values()) == 9589 and len(list(prefix_dir.glob("*.npz"))) == 28,
        "current_network_reconstruction_completed": all(all(value is not None for value in result["errors"].values()) for result in completed.values()),
        "stale_hidden_is_non_vacuous": stale_nonzero,
        "buffer_boundary_and_gae_passed": buffer_pass,
        "baseline_and_corridor_log_probability_passed": log_probability_pass,
        "corridor_restart_and_hold_passed": corridor_pass,
    }
    if all(criteria.values()):
        verdict = "pass_prefix_reset_semantics_gate" if fast_pass else "pass_semantics_reject_fast_burnin"
    else:
        verdict = "fail_stop_prefix_reset_semantics"
    report = {
        "schema_version": 1,
        "experiment_id": "prefix_reset_semantics_gate",
        "gate": "Z6-B",
        "verdict": verdict,
        "summary": {
            "task_count": len(completed),
            "prefix_rows": sum(result["prefix_step"] for result in completed.values()),
            "source_exact_count": sum(result["source_exact"] for result in completed.values()),
            "fast_pass_count": sum(result["fast_pass"] for result in completed.values()),
            "maximum_errors": maximum_errors,
            "stale_nonzero_task_count": sum(any(float(value) > 0.0 for value in result["stale_differences"].values()) for result in nonzero_prefix),
            "nonzero_prefix_task_count": len(nonzero_prefix),
        },
        "criteria": criteria,
        "buffer_gae": buffer_gae,
        "baseline_log_probability": baseline_log_probability,
        "corridor_log_probability": corridor_log_probability,
        "episodes": dict(sorted(completed.items())),
        "evidence_boundary": {
            "established": "current-network burn-in, separated snapshot boundary/recurrent reset masks, GAE fixtures, and baseline/corridor likelihood replay are mechanically expressible without prefix transitions in loss",
            "not_established": "snapshot sampling ratio, role distribution, training throughput, PPO learning benefit, or four-map actor performance",
            "next_action": "only preregister a separate Austin-only prefix-reset training-density gate if this verdict passes",
        },
    }
    atomic_write_json(report_path, report)
    print(f"REPORT={report_path}")
    print(f"VERDICT={verdict}")
