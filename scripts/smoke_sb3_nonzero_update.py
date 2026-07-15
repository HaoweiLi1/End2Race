#!/usr/bin/env python3
"""One-shot, non-zero-learning-rate smoke for the real Austin F1Tenth path.

This is deliberately not a learner entry point.  It collects exactly one
100-step rollout and permits exactly one optimizer step.  A persistent run
state in the ignored artifact directory makes accidental re-execution fail
closed after the first attempt.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import platform
import random
import subprocess
import sys
import traceback
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gymnasium
import imageio.v2 as imageio
import numpy as np
import sb3_contrib
import stable_baselines3
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from model import End2Race
from rl.end2race_gymnasium_env import (
    End2RaceGymnasiumEnv,
    EpisodeResetSpec,
    LatticePlannerOpponentController,
)
from rl.sb3_end2race_policy import (
    DEFAULT_BC_CHECKPOINT,
    END2RACE_ACTION_SIZE,
    END2RACE_LIDAR_SIZE,
    EVALUATOR_STEER_BOUND,
    End2RaceGRUPolicy,
    end2race_observation,
)
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates


ARTIFACT_DIR = ROOT / "artifacts" / "sb3_nonzero_smoke"
VIDEO_PATH = ARTIFACT_DIR / "training_rollout.mp4"
ACTOR_PATH = ARTIFACT_DIR / "actor_after_update.pth"
RESULTS_PATH = ARTIFACT_DIR / "results.json"
RUN_LOG_PATH = ARTIFACT_DIR / "run.log"
RUN_STATE_PATH = ARTIFACT_DIR / "run_state.json"
PRE_UPDATE_SNAPSHOT_PATH = ARTIFACT_DIR / "pre_update_snapshot.pth"

SEED = 20260715
N_ENVS = 1
N_STEPS = 100
BATCH_SIZE = 100
N_EPOCHS = 1
LEARNING_RATE = 1e-6
FPS = 100
SIM_DURATION = 1.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_run_state(stage: str, **details: Any) -> None:
    _atomic_json(RUN_STATE_PATH, {"stage": stage, **details})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def package_python_digest(module: Any) -> dict[str, Any]:
    root = Path(module.__file__).resolve().parent
    files = sorted(root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return {"root": str(root), "python_file_count": len(files), "sha256": digest.hexdigest()}


def git_output(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def run_repair_regression() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_sb3_gru_integration",
        "-v",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(completed.stdout, end="")
    passed = completed.returncode == 0 and "Ran 11 tests" in completed.stdout and completed.stdout.rstrip().endswith("OK")
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "expected_test_count": 11,
        "passed": passed,
        "output_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
    }


class JointActionRecordingLegacyEnv:
    """Transparent recorder immediately in front of the real legacy F110 env."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.received_joint_actions: list[np.ndarray] = []
        self.reset_poses: list[np.ndarray] = []
        self.step_count = 0

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def reset(self, *, poses: np.ndarray) -> Any:
        poses = np.asarray(poses, dtype=np.float64)
        self.reset_poses.append(poses.copy())
        return self.env.reset(poses=poses)

    def step(self, joint_action: np.ndarray) -> Any:
        action = np.asarray(joint_action, dtype=np.float32)
        self.received_joint_actions.append(action.copy())
        self.step_count += 1
        return self.env.step(action)

    def render(self, mode: str = "human") -> Any:
        return self.env.render(mode=mode)

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class DiagnosticSmokeRewardWrapper(gymnasium.Wrapper):
    """Action-sensitive diagnostic reward plus in-rollout terminal-safe video."""

    def __init__(self, env: End2RaceGymnasiumEnv, video_path: Path) -> None:
        super().__init__(env)
        self.video_path = video_path
        self.reward_records: list[dict[str, Any]] = []
        self.frame_shapes: list[tuple[int, int, int]] = []
        self.frame_count = 0
        self.recording_step_count = 0
        self.first_episode_done = False
        self._writer: Any | None = None

    @property
    def integration_env(self) -> End2RaceGymnasiumEnv:
        return self.env

    def render_preflight(self) -> dict[str, Any]:
        frame = np.asarray(self.integration_env.f110_env.render(mode="rgb_array"))
        self._validate_frame(frame)
        return {"dtype": str(frame.dtype), "shape": list(frame.shape)}

    def start_recording(self) -> None:
        if self._writer is not None:
            raise RuntimeError("Video writer already started")
        self._writer = imageio.get_writer(
            self.video_path,
            fps=FPS,
            macro_block_size=1,
        )

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if frame.dtype != np.uint8:
            raise TypeError(f"Expected uint8 render frame, got {frame.dtype}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected H x W x 3 render frame, got {frame.shape}")

    def step(self, action: np.ndarray):
        observation, raw_reward, terminated, truncated, info = self.env.step(action)
        if self.integration_env._raw_observation is None:
            raise RuntimeError("Missing real post-step F1Tenth observation")
        next_measured_speed = self.integration_env._ego_speed(self.integration_env._raw_observation)
        executed_action = self.integration_env.action_trace[-1]["ego_action"]
        smoke_reward = (
            float(raw_reward)
            + 0.01 * math.tanh(next_measured_speed / 10.0)
            - 0.001 * (float(executed_action[0]) / EVALUATOR_STEER_BOUND) ** 2
            - float(bool(info["ego_collision"]))
        )
        if not math.isfinite(smoke_reward):
            raise FloatingPointError("Diagnostic smoke reward is not finite")
        info = dict(info)
        info["raw_base_reward"] = float(raw_reward)
        info["diagnostic_smoke_reward"] = smoke_reward
        self.reward_records.append(
            {
                "transition_index": len(self.reward_records),
                "raw_base_reward": float(raw_reward),
                "smoke_reward": smoke_reward,
                "next_ego_measured_speed": next_measured_speed,
                "executed_ego_action": np.asarray(executed_action).copy(),
                "ego_collision": bool(info["ego_collision"]),
                "opponent_collision": bool(info["opponent_collision"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        # This is intentionally inside Env.step(): DummyVecEnv auto-resets a
        # done environment before callbacks run, so callback recording would
        # lose the timeout/collision terminal frame.
        if not self.first_episode_done:
            if self._writer is None:
                raise RuntimeError("Video recording was not started before rollout")
            frame = np.asarray(self.integration_env.f110_env.render(mode="rgb_array"))
            self._validate_frame(frame)
            self._writer.append_data(frame)
            self.frame_shapes.append(tuple(int(value) for value in frame.shape))
            self.frame_count += 1
            self.recording_step_count += 1
            if terminated or truncated:
                self.first_episode_done = True

        return observation, smoke_reward, terminated, truncated, info

    def close_video(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def close(self) -> None:
        self.close_video()
        super().close()


class OneRolloutTraceCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.rollout_start_count = 0
        self.rollout_end_count = 0
        self.callback_step_count = 0
        self.raw_actions: list[np.ndarray] = []
        self.clipped_actions: list[np.ndarray] = []
        self.raw_actor_means: list[np.ndarray] = []

    def _on_rollout_start(self) -> None:
        self.rollout_start_count += 1

    def _on_rollout_end(self) -> None:
        self.rollout_end_count += 1

    def _on_step(self) -> bool:
        self.callback_step_count += 1
        self.raw_actions.append(np.asarray(self.locals["actions"]).copy())
        self.clipped_actions.append(np.asarray(self.locals["clipped_actions"]).copy())
        raw_mean = self.model.policy.last_raw_actor_mean
        if raw_mean is None:
            raise RuntimeError("Missing diagnostic raw actor mean")
        self.raw_actor_means.append(raw_mean.detach().cpu().numpy().copy())
        return True


def fixed_scenario() -> tuple[dict[str, Any], EpisodeResetSpec]:
    from utils import load_positions_and_speeds_from_params

    scenario = {
        "map_name": "Austin",
        "ego_raceline": "raceline1",
        "opp_raceline": "raceline1",
        "ego_idx": 0,
        "interval_idx": 15,
        "opp_idx": 15,
        "opp_speedscale": 0.5,
        "seed": SEED,
    }
    poses, initial_speeds = load_positions_and_speeds_from_params(scenario, "Austin")
    initial_speed_feature = float(initial_speeds[0] * 0.9)
    spec = EpisodeResetSpec(
        poses=np.asarray(poses, dtype=np.float64),
        initial_speed_feature=initial_speed_feature,
        scenario=deepcopy(scenario),
    )
    recorded = {
        **scenario,
        "num_agents": 2,
        "timestep": 0.01,
        "integrator": "Integrator.RK4",
        "sim_duration": SIM_DURATION,
        "reset_poses": spec.poses.copy(),
        "initial_speed_feature": initial_speed_feature,
        "initial_raceline_speeds": np.asarray(initial_speeds).copy(),
    }
    return recorded, spec


def make_real_training_env() -> tuple[
    DummyVecEnv,
    DiagnosticSmokeRewardWrapper,
    End2RaceGymnasiumEnv,
    JointActionRecordingLegacyEnv,
    LatticePlannerOpponentController,
    dict[str, Any],
]:
    import gym
    from f110_gym.envs.base_classes import Integrator

    scenario_record, spec = fixed_scenario()

    def provider(rng: np.random.Generator) -> EpisodeResetSpec:
        del rng
        return deepcopy(spec)

    legacy_core = gym.make(
        "f110-v0",
        map=str(ROOT / "f1tenth_racetracks" / "Austin" / "Austin_map"),
        map_ext=".png",
        num_agents=2,
        timestep=0.01,
        integrator=Integrator.RK4,
        seed=SEED,
    )
    recording_core = JointActionRecordingLegacyEnv(legacy_core)
    opponent = LatticePlannerOpponentController()
    integration_env = End2RaceGymnasiumEnv(
        recording_core,
        sim_duration=SIM_DURATION,
        reset_provider=provider,
        ego_index=0,
        opponent_controller=opponent,
    )
    diagnostic_env = DiagnosticSmokeRewardWrapper(integration_env, VIDEO_PATH)
    vector_env = DummyVecEnv([lambda: diagnostic_env])
    return vector_env, diagnostic_env, integration_env, recording_core, opponent, scenario_record


def make_model(vector_env: DummyVecEnv, device: str) -> RecurrentPPO:
    return RecurrentPPO(
        End2RaceGRUPolicy,
        vector_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.10,
        vf_coef=0.5,
        ent_coef=1e-4,
        max_grad_norm=0.5,
        target_kl=None,
        seed=SEED,
        device=device,
        policy_kwargs={
            "checkpoint_path": DEFAULT_BC_CHECKPOINT,
            "hidden_scale": 4,
            "critic_hidden_size": 32,
            "steer_log_std_init": -2.0,
            "speed_log_std_init": 0.0,
        },
        verbose=0,
        _init_setup_model=False,
    )


def clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def parameter_global_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared = sum(float(parameter.detach().double().square().sum().cpu()) for parameter in parameters)
    return math.sqrt(squared)


def fixed_sequence(device: torch.device) -> torch.Tensor:
    rng = np.random.default_rng(9137)
    rows = []
    for _ in range(100):
        lidar = rng.uniform(0.05, 12.0, END2RACE_LIDAR_SIZE).astype(np.float32)
        speed = float(rng.uniform(0.0, 8.0))
        rows.append(end2race_observation(lidar, speed))
    return torch.as_tensor(np.stack(rows), dtype=torch.float32, device=device)


def evaluate_fixed_sequence(policy: End2RaceGRUPolicy, observations: torch.Tensor) -> dict[str, torch.Tensor]:
    hidden = torch.zeros((1, 1, policy.actor_hidden_size), dtype=torch.float32, device=observations.device)
    cell = torch.zeros_like(hidden)
    vf_hidden = torch.zeros_like(hidden)
    actions: list[torch.Tensor] = []
    policy.set_training_mode(False)
    with torch.no_grad():
        for index, observation in enumerate(observations):
            action, _value, _log_prob, states = policy.forward(
                observation.reshape(1, -1),
                RNNStates((hidden, cell), (vf_hidden, torch.zeros_like(vf_hidden))),
                torch.tensor([index == 0], dtype=torch.float32, device=observations.device),
                deterministic=True,
            )
            actions.append(action.detach().cpu())
            hidden, cell = states.pi
            vf_hidden = states.vf[0]
    return {
        "actions": torch.cat(actions, dim=0),
        "final_hidden": hidden.detach().cpu(),
        "final_dummy_cell": cell.detach().cpu(),
    }


def save_pre_update_snapshot(
    model: RecurrentPPO,
    rng_state: dict[str, Any],
) -> dict[str, Any]:
    snapshot = {
        "policy_state_dict": clone_state_dict(model.policy),
        "actor_state_dict": clone_state_dict(model.policy.end2race_actor),
        "critic_state_dict": clone_state_dict(model.policy.value_net),
        "log_std": model.policy.log_std.detach().cpu().clone(),
        "optimizer_state_dict": deepcopy(model.policy.optimizer.state_dict()),
        "rng_state": rng_state,
        "actor_checkpoint_keys": list(model.policy.end2race_actor.state_dict()),
    }
    torch.save(snapshot, PRE_UPDATE_SNAPSHOT_PATH)
    return {
        "path": str(PRE_UPDATE_SNAPSHOT_PATH.relative_to(ROOT)),
        "sha256": sha256_file(PRE_UPDATE_SNAPSHOT_PATH),
        "policy_key_count": len(snapshot["policy_state_dict"]),
        "actor_key_count": len(snapshot["actor_state_dict"]),
        "critic_key_count": len(snapshot["critic_state_dict"]),
        "optimizer_state_entry_count": len(snapshot["optimizer_state_dict"]["state"]),
        "rng_state_hashes": {
            name: object_sha256(value) for name, value in rng_state.items()
        },
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def all_finite(*arrays: Any) -> bool:
    for value in arrays:
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all()):
                return False
        elif not np.isfinite(np.asarray(value)).all():
            return False
    return True


def replay_batch(model: RecurrentPPO) -> tuple[Any, dict[str, Any]]:
    numpy_state = np.random.get_state()
    batches = list(model.rollout_buffer.get(BATCH_SIZE))
    np.random.set_state(numpy_state)
    if len(batches) != 1:
        raise AssertionError(f"Expected exactly one minibatch, got {len(batches)}")
    batch = batches[0]
    mask = batch.mask > 1e-8
    model.policy.set_training_mode(False)
    with torch.no_grad():
        values, new_log_prob, _entropy = model.policy.evaluate_actions(
            batch.observations,
            batch.actions,
            batch.lstm_states,
            batch.episode_starts,
        )
    error = (new_log_prob - batch.old_log_prob).abs()[mask]
    ratio = torch.exp(new_log_prob - batch.old_log_prob)[mask]
    return batch, {
        "minibatch_count": len(batches),
        "valid_transition_count": int(mask.sum().item()),
        "padding_timestep_count": int((~mask).sum().item()),
        "max_logp_absolute_error": float(error.max().cpu()),
        "mean_logp_absolute_error": float(error.mean().cpu()),
        "max_ratio_deviation": float((ratio - 1.0).abs().max().cpu()),
        "all_finite": all_finite(values[mask], new_log_prob[mask], ratio),
    }


def collect_rollout_statistics(
    model: RecurrentPPO,
    trace: OneRolloutTraceCallback,
    diagnostic_env: DiagnosticSmokeRewardWrapper,
    integration_env: End2RaceGymnasiumEnv,
    core: JointActionRecordingLegacyEnv,
) -> dict[str, Any]:
    buffer = model.rollout_buffer
    buffer_actions = np.asarray(buffer.actions).reshape(N_STEPS, N_ENVS, END2RACE_ACTION_SIZE).copy()
    raw_actions = np.asarray(trace.raw_actions).reshape(N_STEPS, N_ENVS, END2RACE_ACTION_SIZE)
    clipped_actions = np.asarray(trace.clipped_actions).reshape(N_STEPS, N_ENVS, END2RACE_ACTION_SIZE)
    core_actions = np.asarray(core.received_joint_actions)
    ego_core_actions = core_actions[:, 0, :]
    opponent_core_actions = core_actions[:, 1, :]
    wrapper_actions = np.asarray([entry["ego_action"] for entry in integration_env.action_trace])
    clip_difference = np.abs(raw_actions - clipped_actions)
    buffer_core_error = np.abs(buffer_actions[:, 0, :] - ego_core_actions)

    arrays = {
        "observations": buffer.observations,
        "actions": buffer.actions,
        "log_probs": buffer.log_probs,
        "values": buffer.values,
        "rewards": buffer.rewards,
        "advantages": buffer.advantages,
        "returns": buffer.returns,
    }
    finite = {name: bool(np.isfinite(np.asarray(value)).all()) for name, value in arrays.items()}
    rewards = diagnostic_env.reward_records
    terminal_events = integration_env.terminal_events
    first_terminal = terminal_events[0] if terminal_events else None

    return {
        "rollout_count": trace.rollout_end_count,
        "rollout_start_count": trace.rollout_start_count,
        "callback_step_count": trace.callback_step_count,
        "environment_step_count": core.step_count,
        "valid_transition_count": N_STEPS * N_ENVS,
        "buffer_action_shape": list(buffer_actions.shape),
        "buffer_contains_only_ego_action": buffer_actions.shape[-1] == 2,
        "opponent_action_shape_at_core": list(opponent_core_actions.shape),
        "finite_fields": finite,
        "all_rollout_fields_finite": all(finite.values()),
        "steering": {
            "min": float(buffer_actions[..., 0].min()),
            "max": float(buffer_actions[..., 0].max()),
            "mean": float(buffer_actions[..., 0].mean()),
            "inside_evaluator_bounds": bool(
                np.all(np.abs(buffer_actions[..., 0]) <= EVALUATOR_STEER_BOUND + 1e-7)
            ),
        },
        "speed": {
            "min": float(buffer_actions[..., 1].min()),
            "max": float(buffer_actions[..., 1].max()),
            "mean": float(buffer_actions[..., 1].mean()),
        },
        "log_prob_range": [float(np.min(buffer.log_probs)), float(np.max(buffer.log_probs))],
        "value_range": [float(np.min(buffer.values)), float(np.max(buffer.values))],
        "advantage_mean": float(np.mean(buffer.advantages)),
        "advantage_std": float(np.std(buffer.advantages)),
        "return_mean": float(np.mean(buffer.returns)),
        "return_std": float(np.std(buffer.returns)),
        "sb3_pre_env_clipping_count": int(np.count_nonzero(clip_difference > 0)),
        "sb3_pre_env_clipping_max_error": float(clip_difference.max()),
        "buffer_to_core_ego_action_max_error": float(buffer_core_error.max()),
        "wrapper_to_core_ego_action_max_error": float(np.abs(wrapper_actions - ego_core_actions).max()),
        "first_action_trace": {
            "raw_actor_mean": trace.raw_actor_means[0][0],
            "sb3_sampled_action": raw_actions[0, 0],
            "sb3_clipped_action": clipped_actions[0, 0],
            "rollout_buffer_action": buffer_actions[0, 0],
            "wrapper_ego_action": wrapper_actions[0],
            "core_ego_action": ego_core_actions[0],
            "core_opponent_action": opponent_core_actions[0],
        },
        "episode": {
            "terminal_event_count_during_rollout": len(terminal_events),
            "first_outcome": None if first_terminal is None else first_terminal["reason"],
            "first_terminal_transition_index": None if first_terminal is None else first_terminal["transition_index"],
            "ego_collision": any(record["ego_collision"] for record in rewards),
            "opponent_only_collision_count": sum(
                record["opponent_collision"] and not record["ego_collision"] for record in rewards
            ),
            "timeout": any(event["reason"] == "timeout" for event in terminal_events),
        },
        "reward": {
            "definition": "raw_base_reward + 0.01*tanh(next_ego_measured_speed/10) - 0.001*(executed_ego_steering/0.52)^2 - ego_collision",
            "diagnostic_only": True,
            "record_count": len(rewards),
            "all_finite": all(math.isfinite(record["smoke_reward"]) for record in rewards),
            "total_smoke_reward_before_sb3_timeout_bootstrap": float(sum(record["smoke_reward"] for record in rewards)),
            "total_raw_base_reward": float(sum(record["raw_base_reward"] for record in rewards)),
            "buffer_reward_total_after_sb3_timeout_bootstrap": float(np.sum(buffer.rewards)),
        },
    }


def named_optimizer_parameters(policy: End2RaceGRUPolicy) -> list[tuple[str, torch.nn.Parameter]]:
    parameter_names = {id(parameter): name for name, parameter in policy.named_parameters()}
    result = []
    for group_index, group in enumerate(policy.optimizer.param_groups):
        for parameter_index, parameter in enumerate(group["params"]):
            name = parameter_names.get(id(parameter), f"optimizer_group_{group_index}.parameter_{parameter_index}")
            result.append((name, parameter))
    return result


def opponent_parameters(opponent: LatticePlannerOpponentController) -> list[tuple[str, torch.nn.Parameter]]:
    found: list[tuple[str, torch.nn.Parameter]] = []
    for planner_index, planner in opponent._planners.items():
        if isinstance(planner, torch.nn.Module):
            found.extend((f"planner_{planner_index}.{name}", parameter) for name, parameter in planner.named_parameters())
            continue
        for name, value in vars(planner).items():
            if isinstance(value, torch.nn.Parameter):
                found.append((f"planner_{planner_index}.{name}", value))
    return found


def gradient_norm(gradients: Iterable[torch.Tensor]) -> float:
    squared = sum(float(gradient.detach().double().square().sum().cpu()) for gradient in gradients)
    return math.sqrt(squared)


def one_instrumented_train(
    model: RecurrentPPO,
) -> dict[str, Any]:
    optimizer = model.policy.optimizer
    named_parameters = named_optimizer_parameters(model.policy)
    optimizer.zero_grad(set_to_none=True)
    if any(parameter.grad is not None for _, parameter in named_parameters):
        raise AssertionError("Gradients were not clear immediately before model.train()")

    raw_gradients: dict[str, torch.Tensor] = {}
    hooks = []
    for name, parameter in named_parameters:
        def capture(current: torch.nn.Parameter, parameter_name: str = name) -> None:
            if current.grad is not None:
                raw_gradients[parameter_name] = current.grad.detach().clone()

        hooks.append(parameter.register_post_accumulate_grad_hook(capture))

    original_step = optimizer.step
    instrumentation: dict[str, Any] = {
        "optimizer_step_count": 0,
        "gradient_norm_before_clipping": None,
        "gradient_norm_after_clipping": None,
        "gradients_finite_before_clipping": None,
        "gradients_finite_after_clipping": None,
    }

    def counted_step(*args: Any, **kwargs: Any) -> Any:
        instrumentation["optimizer_step_count"] += 1
        if instrumentation["optimizer_step_count"] > 1:
            raise AssertionError("Refusing a second optimizer.step()")
        before = list(raw_gradients.values())
        after = [parameter.grad for _, parameter in named_parameters if parameter.grad is not None]
        instrumentation["gradient_norm_before_clipping"] = gradient_norm(before)
        instrumentation["gradient_norm_after_clipping"] = gradient_norm(after)
        instrumentation["gradients_finite_before_clipping"] = all_finite(*before)
        instrumentation["gradients_finite_after_clipping"] = all_finite(*after)
        _write_run_state(
            "optimizer_step_entered",
            optimizer_step_count=instrumentation["optimizer_step_count"],
        )
        result = original_step(*args, **kwargs)
        _write_run_state(
            "optimizer_step_completed",
            optimizer_step_count=instrumentation["optimizer_step_count"],
        )
        return result

    optimizer.step = counted_step  # type: ignore[method-assign]
    try:
        model.train()
    finally:
        optimizer.step = original_step  # type: ignore[method-assign]
        for hook in hooks:
            hook.remove()

    adam_steps = []
    for state in optimizer.state.values():
        if "step" in state:
            step = state["step"]
            adam_steps.append(float(step.detach().cpu()) if isinstance(step, torch.Tensor) else float(step))
    logger_values = model.logger.name_to_value
    instrumentation.update(
        {
            "adam_state_entry_count": len(optimizer.state),
            "adam_step_min": min(adam_steps, default=0.0),
            "adam_step_max": max(adam_steps, default=0.0),
            "losses": {
                "policy_loss": float(logger_values["train/policy_gradient_loss"]),
                "value_loss": float(logger_values["train/value_loss"]),
                "entropy_loss": float(logger_values["train/entropy_loss"]),
                "approximate_kl": float(logger_values["train/approx_kl"]),
                "clip_fraction": float(logger_values["train/clip_fraction"]),
                "total_loss": float(logger_values["train/loss"]),
            },
        }
    )
    return instrumentation


def state_delta(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], names: Iterable[str]) -> float:
    deltas = []
    for name in names:
        left, right = before[name], after[name]
        if left.is_floating_point():
            deltas.append(float((right - left).abs().max()))
    return max(deltas, default=0.0)


def update_deltas(
    actor_before: dict[str, torch.Tensor],
    critic_before: dict[str, torch.Tensor],
    log_std_before: torch.Tensor,
    policy: End2RaceGRUPolicy,
) -> dict[str, Any]:
    actor_after = clone_state_dict(policy.end2race_actor)
    critic_after = clone_state_dict(policy.value_net)
    actor_groups = {
        "k": ["k"],
        "speed_mlp": [name for name in actor_before if name.startswith("speed_mlp.")],
        "gru": [name for name in actor_before if name.startswith("gru.")],
        "output_layer": [name for name in actor_before if name.startswith("output_layer.")],
        "dummy_embedding": ["dummy_embedding"],
    }
    actor_delta = {
        group: state_delta(actor_before, actor_after, names)
        for group, names in actor_groups.items()
    }
    critic_delta = state_delta(critic_before, critic_after, critic_before)
    log_std_delta = float((policy.log_std.detach().cpu() - log_std_before).abs().max())
    all_deltas = [*actor_delta.values(), critic_delta, log_std_delta]
    return {
        "actor": actor_delta,
        "actor_max": max(actor_delta.values()),
        "critic_max": critic_delta,
        "log_std_max": log_std_delta,
        "global_max": max(all_deltas),
        "all_finite": bool(np.isfinite(np.asarray(all_deltas)).all()),
    }


def post_update_buffer_checks(model: RecurrentPPO, batch: Any) -> dict[str, Any]:
    mask = batch.mask > 1e-8
    model.policy.set_training_mode(False)
    with torch.no_grad():
        values, new_log_prob, _entropy = model.policy.evaluate_actions(
            batch.observations,
            batch.actions,
            batch.lstm_states,
            batch.episode_starts,
        )
    log_ratio = new_log_prob - batch.old_log_prob
    ratio = torch.exp(log_ratio)
    approximate_kl = torch.mean(((ratio - 1.0) - log_ratio)[mask])

    observation = batch.observations[mask][0:1]
    repeated = observation.repeat(2, 1)
    hidden = torch.zeros((1, 2, model.policy.actor_hidden_size), device=model.device)
    with torch.no_grad():
        means, _states = model.policy.actor_mean(
            repeated,
            (hidden, torch.zeros_like(hidden)),
            torch.ones(2, device=model.device),
        )
        distribution = model.policy._distribution(means)
        bound = torch.as_tensor(EVALUATOR_STEER_BOUND, dtype=repeated.dtype, device=model.device)
        inner_bound = torch.nextafter(bound, torch.zeros_like(bound))
        boundary_actions = torch.stack(
            (
                torch.stack((-inner_bound, means[0, 1])),
                torch.stack((inner_bound, means[1, 1])),
            )
        )
        boundary_log_prob = distribution.log_prob(boundary_actions)

    return {
        "all_finite": all_finite(values[mask], new_log_prob[mask], ratio[mask], approximate_kl, boundary_log_prob),
        "ratio_min": float(ratio[mask].min().cpu()),
        "ratio_max": float(ratio[mask].max().cpu()),
        "ratio_max_deviation": float((ratio[mask] - 1.0).abs().max().cpu()),
        "approximate_kl": float(approximate_kl.cpu()),
        "value_range": [float(values[mask].min().cpu()), float(values[mask].max().cpu())],
        "log_prob_range": [float(new_log_prob[mask].min().cpu()), float(new_log_prob[mask].max().cpu())],
        "near_boundary_log_prob": boundary_log_prob.detach().cpu(),
        "near_boundary_log_prob_finite": bool(torch.isfinite(boundary_log_prob).all()),
    }


def export_and_verify_actor(policy: End2RaceGRUPolicy, pretrained_hash_before: str) -> dict[str, Any]:
    torch.save(policy.end2race_actor.state_dict(), ACTOR_PATH)
    state = torch.load(ACTOR_PATH, map_location="cpu", weights_only=True)
    expected = policy.end2race_actor.state_dict()
    reloaded = End2Race(mask_prob=0.0, hidden_scale=4)
    incompatible = reloaded.load_state_dict(state, strict=True)
    exact_error = max(float((state[name] - expected[name].detach().cpu()).abs().max()) for name in state)
    return {
        "path": str(ACTOR_PATH.relative_to(ROOT)),
        "sha256": sha256_file(ACTOR_PATH),
        "key_count": len(state),
        "keys": list(state),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "all_parameters_finite": all_finite(*state.values()),
        "policy_actor_exact_max_error": exact_error,
        "strict_load_passed": not incompatible.missing_keys and not incompatible.unexpected_keys,
        "pretrained_checkpoint_was_not_output_path": ACTOR_PATH.resolve() != DEFAULT_BC_CHECKPOINT.resolve(),
        "pretrained_sha256_before": pretrained_hash_before,
        "pretrained_sha256_after": sha256_file(DEFAULT_BC_CHECKPOINT),
    }


def verify_video(diagnostic_env: DiagnosticSmokeRewardWrapper) -> dict[str, Any]:
    if not VIDEO_PATH.exists() or VIDEO_PATH.stat().st_size <= 0:
        raise AssertionError("Training rollout video is missing or empty")
    reader = imageio.get_reader(VIDEO_PATH)
    try:
        frame_count = int(reader.count_frames())
        first = np.asarray(reader.get_data(0))
        last = np.asarray(reader.get_data(frame_count - 1))
        metadata = reader.get_meta_data()
    finally:
        reader.close()
    fps = float(metadata.get("fps", FPS))
    return {
        "path": str(VIDEO_PATH.relative_to(ROOT)),
        "exists": True,
        "size_bytes": VIDEO_PATH.stat().st_size,
        "sha256": sha256_file(VIDEO_PATH),
        "captured_frame_count": diagnostic_env.frame_count,
        "decoded_frame_count": frame_count,
        "fps": fps,
        "resolution": [int(first.shape[1]), int(first.shape[0])],
        "first_frame_readable": first.ndim == 3 and first.shape[2] == 3,
        "last_frame_readable": last.ndim == 3 and last.shape[2] == 3,
        "decoded_dtype": str(first.dtype),
        "duration_seconds": frame_count / fps,
        "expected_duration_from_recorded_steps": diagnostic_env.recording_step_count / FPS,
        "recorded_only_first_training_episode": diagnostic_env.first_episode_done,
        "source": "frames captured inside the unique training Env.step before DummyVecEnv auto-reset",
    }


def execute() -> dict[str, Any]:
    _write_run_state("preflight_started", optimizer_step_count=0)
    repair_regression = run_repair_regression()
    if not repair_regression["passed"]:
        _write_run_state("repair_regression_failed", optimizer_step_count=0)
        raise RuntimeError("Existing 11-test repair regression failed; non-zero update refused")

    pretrained_hash_before = sha256_file(DEFAULT_BC_CHECKPOINT)
    third_party_before = {
        "stable_baselines3": package_python_digest(stable_baselines3),
        "sb3_contrib": package_python_digest(sb3_contrib),
    }
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_working_tree_status_before_run": git_output("status", "--short"),
        "pretrained_checkpoint": str(DEFAULT_BC_CHECKPOINT.relative_to(ROOT)),
        "pretrained_sha256_before": pretrained_hash_before,
    }
    required_versions = {
        "python_3_10": platform.python_version_tuple()[:2] == ("3", "10"),
        "torch_2_7_0_cu128": torch.__version__ == "2.7.0+cu128",
        "gymnasium_1_2_3": gymnasium.__version__ == "1.2.3",
        "stable_baselines3_2_7_1": stable_baselines3.__version__ == "2.7.1",
        "sb3_contrib_2_7_1": sb3_contrib.__version__ == "2.7.1",
    }
    if not all(required_versions.values()):
        raise RuntimeError(f"Required environment versions do not match: {required_versions}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vector_env, diagnostic_env, integration_env, core, opponent, scenario = make_real_training_env()
    model = make_model(vector_env, device)
    setup_model_call_count = 0
    setup_learn_call_count = 0
    collect_rollouts_call_count = 0
    train_call_count = 0
    callback: Any | None = None
    try:
        model._setup_model()
        setup_model_call_count += 1
        trace = OneRolloutTraceCallback()
        _, callback = model._setup_learn(
            total_timesteps=N_STEPS * N_ENVS,
            callback=trace,
            reset_num_timesteps=True,
            tb_log_name="sb3_gru_nonzero_smoke",
            progress_bar=False,
        )
        setup_learn_call_count += 1
        callback.on_training_start(locals(), globals())

        if model.device.type != device:
            raise AssertionError(f"Requested {device}, model uses {model.device}")
        render_preflight = diagnostic_env.render_preflight()

        fixed_observations = fixed_sequence(model.device)
        fixed_before = evaluate_fixed_sequence(model.policy, fixed_observations)
        actor_before = clone_state_dict(model.policy.end2race_actor)
        critic_before = clone_state_dict(model.policy.value_net)
        policy_before = clone_state_dict(model.policy)
        log_std_before = model.policy.log_std.detach().cpu().clone()
        optimizer_before_update = deepcopy(model.policy.optimizer.state_dict())
        optimizer_named = named_optimizer_parameters(model.policy)
        optimizer_ids = [id(parameter) for _, parameter in optimizer_named]
        opponent_named = opponent_parameters(opponent)
        opponent_before = {name: parameter.detach().cpu().clone() for name, parameter in opponent_named}
        opponent_ids = {id(parameter) for _, parameter in opponent_named}
        rng_state = capture_rng_state()
        pre_snapshot = save_pre_update_snapshot(model, rng_state)

        pre_rollout = {
            "actor_parameter_norm": parameter_global_norm(model.policy.end2race_actor.parameters()),
            "critic_parameter_norm": parameter_global_norm(model.policy.value_net.parameters()),
            "log_std": log_std_before,
            "deterministic_fixed_input_action": fixed_before["actions"][0],
            "deterministic_fixed_sequence_final_hidden_norm": float(fixed_before["final_hidden"].double().norm()),
            "deterministic_fixed_sequence_dummy_cell_max_abs": float(fixed_before["final_dummy_cell"].abs().max()),
            "actor_checkpoint_keys": list(actor_before),
            "actor_checkpoint_key_count": len(actor_before),
            "optimizer_state_entry_count": len(optimizer_before_update["state"]),
            "optimizer_parameter_count": len(optimizer_ids),
            "optimizer_parameter_identities_unique": len(optimizer_ids) == len(set(optimizer_ids)),
            "opponent_trainable_parameter_count": len(opponent_named),
            "opponent_parameter_in_optimizer_count": len(opponent_ids & set(optimizer_ids)),
            "snapshot": pre_snapshot,
        }

        diagnostic_env.start_recording()
        _write_run_state("rollout_started", optimizer_step_count=0, collect_rollouts_call_count=1)
        collect_rollouts_call_count += 1
        if model.env is None or not model.collect_rollouts(
            model.env,
            callback,
            model.rollout_buffer,
            n_rollout_steps=N_STEPS,
        ):
            raise RuntimeError("The unique training rollout was interrupted")
        diagnostic_env.close_video()
        _write_run_state(
            "rollout_completed",
            optimizer_step_count=0,
            collect_rollouts_call_count=collect_rollouts_call_count,
            environment_step_count=core.step_count,
        )

        rollout = collect_rollout_statistics(model, trace, diagnostic_env, integration_env, core)
        replay_data, replay_before_update = replay_batch(model)
        video = verify_video(diagnostic_env)

        # Hard pre-update gates: a broken real action/replay/video contract
        # must never be followed by the sole non-zero optimizer update.
        pre_update_gates = {
            "one_rollout": collect_rollouts_call_count == 1 and trace.rollout_end_count == 1,
            "one_hundred_environment_steps": core.step_count == N_STEPS,
            "one_hundred_valid_transitions": replay_before_update["valid_transition_count"] == N_STEPS,
            "one_minibatch": replay_before_update["minibatch_count"] == 1,
            "rollout_finite": rollout["all_rollout_fields_finite"],
            "reward_finite": rollout["reward"]["all_finite"],
            "steering_bounded": rollout["steering"]["inside_evaluator_bounds"],
            "no_sb3_clipping": rollout["sb3_pre_env_clipping_count"] == 0,
            "buffer_core_action_identity": rollout["buffer_to_core_ego_action_max_error"] <= 1e-7,
            "replay_max": replay_before_update["max_logp_absolute_error"] <= 1e-6,
            "replay_mean": replay_before_update["mean_logp_absolute_error"] <= 1e-7,
            "replay_ratio": replay_before_update["max_ratio_deviation"] <= 1e-6,
            "video_nonempty": video["decoded_frame_count"] > 0,
            "learning_rate_exact": all(group["lr"] == LEARNING_RATE for group in model.policy.optimizer.param_groups),
            "one_epoch": model.n_epochs == 1,
            "batch_size_equals_rollout": model.batch_size == N_STEPS,
        }
        if not all(pre_update_gates.values()):
            raise AssertionError(f"Pre-update gate failure; optimizer update refused: {pre_update_gates}")

        if any(parameter.grad is not None for parameter in model.policy.optimizer.param_groups[0]["params"]):
            model.policy.optimizer.zero_grad(set_to_none=True)
        optimizer_state_immediately_before_train = deepcopy(model.policy.optimizer.state_dict())
        _write_run_state("train_called", optimizer_step_count=0, train_call_count=1)
        train_call_count += 1
        train_instrumentation = one_instrumented_train(model)

        deltas = update_deltas(actor_before, critic_before, log_std_before, model.policy)
        fixed_after = evaluate_fixed_sequence(model.policy, fixed_observations)
        fixed_action_delta = float((fixed_after["actions"] - fixed_before["actions"]).abs().max())
        fixed_hidden_delta = float((fixed_after["final_hidden"] - fixed_before["final_hidden"]).abs().max())
        post_update = post_update_buffer_checks(model, replay_data)
        checkpoint = export_and_verify_actor(model.policy, pretrained_hash_before)

        opponent_after_delta = 0.0
        for name, parameter in opponent_named:
            opponent_after_delta = max(
                opponent_after_delta,
                float((parameter.detach().cpu() - opponent_before[name]).abs().max()),
            )
        optimizer_ids_after = [id(parameter) for _, parameter in named_optimizer_parameters(model.policy)]
        full_policy_delta = state_delta(policy_before, clone_state_dict(model.policy), policy_before)
        deltas["full_policy_max"] = full_policy_delta
        third_party_after = {
            "stable_baselines3": package_python_digest(stable_baselines3),
            "sb3_contrib": package_python_digest(sb3_contrib),
        }

        update_checks = {
            "train_called_once": train_call_count == 1,
            "optimizer_step_once": train_instrumentation["optimizer_step_count"] == 1,
            "adam_step_not_above_one": train_instrumentation["adam_step_max"] == 1.0,
            "learning_rate_nonzero_exact": all(group["lr"] == LEARNING_RATE for group in model.policy.optimizer.param_groups),
            "losses_finite": all_finite(*train_instrumentation["losses"].values()),
            "gradient_norms_finite": all_finite(
                train_instrumentation["gradient_norm_before_clipping"],
                train_instrumentation["gradient_norm_after_clipping"],
            ),
            "actor_changed": deltas["actor_max"] > 0.0,
            "gru_or_output_changed": max(deltas["actor"]["gru"], deltas["actor"]["output_layer"]) > 0.0,
            "critic_changed": deltas["critic_max"] > 0.0,
            "log_std_changed": deltas["log_std_max"] > 0.0,
            "dummy_embedding_unchanged": deltas["actor"]["dummy_embedding"] == 0.0,
            "all_deltas_finite": deltas["all_finite"],
            "global_delta_positive": deltas["global_max"] > 0.0,
            "global_delta_below_1e_4": deltas["global_max"] < 1e-4,
            "post_update_finite": post_update["all_finite"],
            "post_update_kl_below_1e_3": post_update["approximate_kl"] < 1e-3,
            "near_boundary_log_prob_finite": post_update["near_boundary_log_prob_finite"],
            "fixed_sequence_action_finite": all_finite(fixed_after["actions"]),
            "fixed_sequence_hidden_finite": all_finite(fixed_after["final_hidden"]),
            "fixed_sequence_steering_bounded": bool(
                torch.all(fixed_after["actions"][:, 0].abs() <= EVALUATOR_STEER_BOUND + 1e-7)
            ),
            "fixed_sequence_action_changed": fixed_action_delta > 0.0,
            "fixed_sequence_action_change_small": fixed_action_delta < 1e-3,
            "checkpoint_strict_load": checkpoint["strict_load_passed"],
            "checkpoint_12_keys": checkpoint["key_count"] == 12,
            "checkpoint_exact_policy_actor": checkpoint["policy_actor_exact_max_error"] == 0.0,
            "pretrained_hash_unchanged": checkpoint["pretrained_sha256_before"] == checkpoint["pretrained_sha256_after"],
            "optimizer_ids_still_unique": len(optimizer_ids_after) == len(set(optimizer_ids_after)),
            "optimizer_parameter_identities_unchanged": optimizer_ids_after == optimizer_ids,
            "opponent_not_in_optimizer": len(opponent_ids & set(optimizer_ids_after)) == 0,
            "opponent_parameter_unchanged": opponent_after_delta == 0.0,
            "third_party_python_sources_unchanged": third_party_before == third_party_after,
        }

        all_checks = {
            "repair_regression": repair_regression["passed"],
            **{f"pre_update.{key}": value for key, value in pre_update_gates.items()},
            **{f"update.{key}": value for key, value in update_checks.items()},
            "video.exists_nonempty": video["exists"] and video["size_bytes"] > 0,
            "video.first_last_readable": video["first_frame_readable"] and video["last_frame_readable"],
            "video.frame_count_matches_capture": (
                video["decoded_frame_count"] == video["captured_frame_count"] > 0
            ),
            "video.fps_is_100": abs(video["fps"] - FPS) <= 1e-9,
            "video.duration_matches_recorded_steps": abs(
                video["duration_seconds"] - video["expected_duration_from_recorded_steps"]
            ) <= 1.0 / FPS,
            "video.uint8_rgb": video["decoded_dtype"] == "uint8",
            "video.stops_at_first_episode_boundary": (
                rollout["episode"]["first_terminal_transition_index"] is not None
                and video["captured_frame_count"]
                == rollout["episode"]["first_terminal_transition_index"] + 1
            ),
        }
        verdict = "PASS_FOR_TRAINING_PIPELINE_IMPLEMENTATION" if all(all_checks.values()) else "FAIL"

        results = {
            "verdict": verdict,
            "scope_warning": (
                "This diagnostic reward and single optimizer step validate only the training pipeline; "
                "they do not validate reward design, PPO performance, collision avoidance, or overtaking."
            ),
            "environment": environment,
            "required_versions": required_versions,
            "repair_regression": repair_regression,
            "scenario": scenario,
            "ppo_configuration": {
                "algorithm": "stock sb3_contrib.RecurrentPPO",
                "rollout_buffer": type(model.rollout_buffer).__module__ + "." + type(model.rollout_buffer).__name__,
                "n_envs": N_ENVS,
                "n_steps": N_STEPS,
                "batch_size": BATCH_SIZE,
                "n_epochs": N_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "gamma": model.gamma,
                "gae_lambda": model.gae_lambda,
                "clip_range": float(model.clip_range(1.0)),
                "vf_coef": model.vf_coef,
                "ent_coef": model.ent_coef,
                "max_grad_norm": model.max_grad_norm,
                "target_kl": model.target_kl,
                "seed": SEED,
                "device": str(model.device),
            },
            "explicit_call_counts": {
                "setup_model": setup_model_call_count,
                "setup_learn": setup_learn_call_count,
                "collect_rollouts": collect_rollouts_call_count,
                "train": train_call_count,
                "optimizer_step": train_instrumentation["optimizer_step_count"],
            },
            "render_preflight": render_preflight,
            "pre_rollout_snapshot": pre_rollout,
            "rollout": rollout,
            "replay_before_update": replay_before_update,
            "pre_update_gates": pre_update_gates,
            "optimizer_state": {
                "before_rollout_entry_count": len(optimizer_before_update["state"]),
                "immediately_before_train_entry_count": len(optimizer_state_immediately_before_train["state"]),
                "after_train_entry_count": len(model.policy.optimizer.state),
                "parameter_identities_unique_before": len(optimizer_ids) == len(set(optimizer_ids)),
                "parameter_identities_unchanged_after": optimizer_ids_after == optimizer_ids,
            },
            "train": train_instrumentation,
            "parameter_deltas": deltas,
            "opponent": {
                "trainable_parameter_count": len(opponent_named),
                "optimizer_overlap_count": len(opponent_ids & set(optimizer_ids_after)),
                "parameter_max_delta": opponent_after_delta,
            },
            "fixed_sequence_after_update": {
                "action_max_absolute_change": fixed_action_delta,
                "final_hidden_max_absolute_change": fixed_hidden_delta,
                "action_all_finite": all_finite(fixed_after["actions"]),
                "hidden_all_finite": all_finite(fixed_after["final_hidden"]),
                "dummy_cell_max_absolute_value": float(fixed_after["final_dummy_cell"].abs().max()),
                "steering_min": float(fixed_after["actions"][:, 0].min()),
                "steering_max": float(fixed_after["actions"][:, 0].max()),
            },
            "post_update_buffer": post_update,
            "checkpoint": checkpoint,
            "video": video,
            "third_party_integrity": {
                "before": third_party_before,
                "after": third_party_after,
                "unchanged": third_party_before == third_party_after,
            },
            "checks": all_checks,
            "formal_training_performed": False,
            "nonzero_optimizer_updates_performed": 1,
        }
        _atomic_json(RESULTS_PATH, results)
        _write_run_state(
            "completed",
            verdict=verdict,
            optimizer_step_count=train_instrumentation["optimizer_step_count"],
            collect_rollouts_call_count=collect_rollouts_call_count,
            train_call_count=train_call_count,
            results_sha256=sha256_file(RESULTS_PATH),
        )
        print(json.dumps(_jsonable({
            "verdict": verdict,
            "rollout_count": collect_rollouts_call_count,
            "optimizer_step_count": train_instrumentation["optimizer_step_count"],
            "replay_max_error": replay_before_update["max_logp_absolute_error"],
            "post_update_kl": post_update["approximate_kl"],
            "parameter_deltas": deltas,
            "video": video,
        }), indent=2))
        return results
    finally:
        if callback is not None:
            callback.on_training_end()
        diagnostic_env.close_video()
        vector_env.close()


def main() -> int:
    if ARTIFACT_DIR.exists() and any(ARTIFACT_DIR.iterdir()):
        print(
            f"Refusing to run: one-shot artifact directory is not empty: {ARTIFACT_DIR}",
            file=sys.stderr,
        )
        return 2
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG_PATH.open("w", encoding="utf-8", buffering=1) as run_log:
        class Tee(io.TextIOBase):
            def __init__(self, *streams: Any) -> None:
                self.streams = streams

            def write(self, text: str) -> int:
                for stream in self.streams:
                    stream.write(text)
                    stream.flush()
                return len(text)

            def flush(self) -> None:
                for stream in self.streams:
                    stream.flush()

        tee_out = Tee(sys.stdout, run_log)
        tee_err = Tee(sys.stderr, run_log)
        try:
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                execute()
            return 0
        except Exception as exc:
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                traceback.print_exc()
                previous: dict[str, Any] = {}
                if RUN_STATE_PATH.exists():
                    previous = json.loads(RUN_STATE_PATH.read_text(encoding="utf-8"))
                _write_run_state(
                    "failed",
                    failure_type=type(exc).__name__,
                    failure=str(exc),
                    previous_state=previous,
                )
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
