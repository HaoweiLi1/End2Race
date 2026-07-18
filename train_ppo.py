#!/usr/bin/env python3
"""Train one fixed End2Race PPO experiment profile."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from model import End2Race
from ppo import config as ppo_config
from ppo.environment import End2RaceGymnasiumEnv, LatticePlannerOpponentController
from ppo.policy import End2RaceGRUPolicy, End2RaceRecurrentPPO
from ppo.reward import PPOTransitionReward, ProgressProjector
from ppo.scenarios import FixedMixtureScenarioSampler, load_hard_pool, training_scenarios
from ppo.vec_env import CentralScheduleSubprocVecEnv


def parse_arguments() -> argparse.Namespace:
    """Parse the three supported run arguments."""
    parser = argparse.ArgumentParser(description="Train End2Race PPO")
    parser.add_argument("--config", required=True, choices=ppo_config.CONFIGS)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--screen-pause",
        action="store_true",
        help="Pause after the config's first checkpoint and read continue/stop from stdin.",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """Seed every RNG used by the formal run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_json_atomic(path: Path, value: Any) -> None:
    """Atomically replace a small JSON status file in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def append_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _assert_finite(value: Any, path: str = "metrics") -> None:
    """Reject any non-finite numeric value in a nested metrics record."""
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise RuntimeError(f"Non-finite value at {path}: {value!r}")


def _optimizer_step(model: End2RaceRecurrentPPO, *, require_initialized: bool) -> dict[str, int]:
    """Read the common Adam step of every active optimizer parameter."""
    steps: list[int] = []
    missing = 0
    active = 0
    for group in model.policy.optimizer.param_groups:
        for parameter in group["params"]:
            if not parameter.requires_grad:
                continue
            active += 1
            state = model.policy.optimizer.state.get(parameter, {})
            if "step" not in state:
                missing += 1
                continue
            raw_step = state["step"]
            step = float(raw_step.detach().cpu().item() if torch.is_tensor(raw_step) else raw_step)
            if not np.isfinite(step) or step < 0.0 or not step.is_integer():
                raise RuntimeError(f"Invalid optimizer step value: {step!r}")
            steps.append(int(step))
    if active == 0:
        raise RuntimeError("Optimizer has no active parameters")
    if steps and missing:
        raise RuntimeError(f"Optimizer state is initialized for only {len(steps)}/{active} active parameters")
    if require_initialized and missing:
        raise RuntimeError(f"Optimizer state is missing for {missing}/{active} active parameters")
    if not steps:
        return {"min": 0, "max": 0, "active_parameters": active}
    minimum, maximum = min(steps), max(steps)
    if minimum != maximum:
        raise RuntimeError(f"Active optimizer parameters have inconsistent steps: {minimum}..{maximum}")
    return {"min": minimum, "max": maximum, "active_parameters": active}


def _parameter_delta_statistics(
    candidate: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, float | int]:
    """Summarize one named actor parameter group's displacement from BC."""
    names = sorted(name for name in reference if name.startswith(prefixes))
    if not names:
        raise RuntimeError(f"No actor parameters match prefixes {prefixes}")
    squared_delta = 0.0
    squared_reference = 0.0
    maximum = 0.0
    count = 0
    for name in names:
        current = candidate[name].detach().cpu().double()
        baseline = reference[name].detach().cpu().double()
        delta = current - baseline
        squared_delta += float(delta.square().sum().item())
        squared_reference += float(baseline.square().sum().item())
        maximum = max(maximum, float(delta.abs().max().item()))
        count += delta.numel()
    rms_delta = float(np.sqrt(squared_delta / count))
    rms_reference = float(np.sqrt(squared_reference / count))
    return {
        "parameter_tensors": len(names),
        "parameter_elements": count,
        "max_abs_delta_from_bc": maximum,
        "rms_delta_from_bc": rms_delta,
        "relative_rms_delta_from_bc": rms_delta / max(rms_reference, 1e-12),
    }


def _parameter_previous_delta_statistics(
    candidate: dict[str, torch.Tensor],
    previous: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, float]:
    """Summarize one actor parameter group's displacement from the previous update."""
    names = sorted(name for name in previous if name.startswith(prefixes))
    if not names:
        raise RuntimeError(f"No actor parameters match prefixes {prefixes}")
    squared_delta = 0.0
    squared_previous = 0.0
    maximum = 0.0
    count = 0
    for name in names:
        current = candidate[name].detach().cpu().double()
        reference = previous[name].detach().cpu().double()
        delta = current - reference
        squared_delta += float(delta.square().sum().item())
        squared_previous += float(reference.square().sum().item())
        maximum = max(maximum, float(delta.abs().max().item()))
        count += delta.numel()
    rms_delta = float(np.sqrt(squared_delta / count))
    rms_previous = float(np.sqrt(squared_previous / count))
    return {
        "max_abs_delta_from_previous": maximum,
        "rms_delta_from_previous": rms_delta,
        "relative_rms_delta_from_previous": rms_delta / max(rms_previous, 1e-12),
    }


def _actor_delta_record(
    model: End2RaceRecurrentPPO,
    bc_state: dict[str, torch.Tensor],
    previous_actor_state: dict[str, torch.Tensor],
    initial_log_std: torch.Tensor,
) -> dict[str, Any]:
    actor_state = model.policy.actor_checkpoint_state_dict()
    def group(prefixes: tuple[str, ...]) -> dict[str, float | int]:
        return {
            **_parameter_delta_statistics(actor_state, bc_state, prefixes),
            **_parameter_previous_delta_statistics(actor_state, previous_actor_state, prefixes),
        }
    return {
        "gru": group(("gru.",)),
        "output_layer": group(("output_layer.",)),
        "frozen_actor": group(("k", "dummy_embedding", "speed_mlp.")),
        "log_std_max_abs_delta_from_initial": float(
            (model.policy.log_std.detach().cpu() - initial_log_std).abs().max().item()
        ),
    }


class PPOTrainingCallback(BaseCallback):
    """Collect one concise metrics record per rollout update."""

    def __init__(self) -> None:
        super().__init__()
        self.update = 0
        self.latest: dict[str, Any] = {}

    def _on_rollout_start(self) -> None:
        self.current_update = self.update + 1
        self.transitions = 0
        self.completed = Counter()
        self.completed_by_branch = Counter()
        self.completed_by_role = Counter()
        self.branches = Counter()
        self.roles = Counter()
        self.reward_sums = Counter()
        self.reward_sums_by_role = Counter()
        self.scenario_ids: set[str] = set()
        self.hard_scenario_ids: set[str] = set()
        self.episode_length_steps_by_role = Counter()
        self.hard_truncations = 0
        self.action_count = 0
        self.action_sum = np.zeros(2, dtype=np.float64)
        self.action_sum_squares = np.zeros(2, dtype=np.float64)
        self.action_min = np.full(2, np.inf, dtype=np.float64)
        self.action_max = np.full(2, -np.inf, dtype=np.float64)
        self.paired_completed: list[dict[str, Any]] = []
        self.paired_cross_update_keys: set[tuple[int, int, int, str]] = set()

    def _on_step(self) -> bool:
        infos = list(self.locals["infos"])
        dones = np.asarray(self.locals["dones"], dtype=bool)
        actions = np.asarray(self.locals["actions"], dtype=np.float64).reshape(len(infos), -1)
        if actions.shape[1] != 2:
            raise RuntimeError(f"Expected two PPO action dimensions, got {actions.shape[1]}")
        self.action_count += actions.shape[0]
        self.action_sum += actions.sum(axis=0)
        self.action_sum_squares += np.square(actions).sum(axis=0)
        self.action_min = np.minimum(self.action_min, actions.min(axis=0))
        self.action_max = np.maximum(self.action_max, actions.max(axis=0))
        for index, info in enumerate(infos):
            self.transitions += 1
            branch = str(info["sampler_branch"])
            role = str(info["scenario"]["env_role"])
            scenario_id = str(info["scenario_id"])
            self.branches[branch] += 1
            self.roles[role] += 1
            self.scenario_ids.add(scenario_id)
            if branch in {"bc_ego_collision", "hard_pool"}:
                self.hard_scenario_ids.add(scenario_id)
            for key in ("reward_progress", "reward_relative", "reward_margin", "reward_collision", "reward_total"):
                self.reward_sums[key] += float(info[key])
                self.reward_sums_by_role[(role, key)] += float(info[key])
            if dones[index]:
                if bool(info["ego_collision"]):
                    outcome = "ego_collision"
                elif float(info["relative_position_m"]) > 0.0:
                    outcome = "overtake"
                else:
                    outcome = "follow"
                self.completed[outcome] += 1
                self.completed_by_branch[(branch, outcome)] += 1
                self.completed_by_role[(role, outcome)] += 1
                self.episode_length_steps_by_role[role] += int(round(float(info["elapsed_time"]) / 0.01))
                if role == "hard" and bool(info["timeout"]):
                    self.hard_truncations += 1
                if info.get("pair_group") is not None:
                    pair_record = {
                        "pair_group": int(info["pair_group"]),
                        "pair_member": int(info["pair_member"]),
                        "pair_episode_ordinal": int(info["pair_episode_ordinal"]),
                        "scenario_id": scenario_id,
                        "policy_update_index": int(info["policy_update_index"]),
                        "episode_outcome": str(info["episode_outcome"]),
                        "episode_return": float(info["episode_return"]),
                        "elapsed_time": float(info["elapsed_time"]),
                    }
                    pair_key = (
                        pair_record["pair_group"],
                        pair_record["pair_episode_ordinal"],
                        pair_record["policy_update_index"],
                        pair_record["scenario_id"],
                    )
                    if pair_record["policy_update_index"] == self.current_update:
                        self.paired_completed.append(pair_record)
                    else:
                        self.paired_cross_update_keys.add(pair_key)
        return True

    @staticmethod
    def _difference_statistics(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "mean_absolute": None, "min": None, "max": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "mean": float(array.mean()),
            "mean_absolute": float(np.abs(array).mean()),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    def _paired_telemetry(self) -> dict[str, Any]:
        grouped: dict[tuple[int, int, int, str], list[dict[str, Any]]] = {}
        for record in self.paired_completed:
            key = (
                record["pair_group"],
                record["pair_episode_ordinal"],
                record["policy_update_index"],
                record["scenario_id"],
            )
            grouped.setdefault(key, []).append(record)
        complete: list[tuple[dict[str, Any], dict[str, Any]]] = []
        incomplete = 0
        for records in grouped.values():
            by_member = {record["pair_member"]: record for record in records}
            if len(records) == 2 and set(by_member) == {0, 1}:
                complete.append((by_member[0], by_member[1]))
            else:
                incomplete += 1
        outcomes = [
            (first["episode_outcome"], second["episode_outcome"])
            for first, second in complete
        ]
        discordant = sum((left == "ego_collision") != (right == "ego_collision") for left, right in outcomes)
        both_collision = sum(left == right == "ego_collision" for left, right in outcomes)
        both_safe = sum(left != "ego_collision" and right != "ego_collision" for left, right in outcomes)
        return_differences = [
            second["episode_return"] - first["episode_return"] for first, second in complete
        ]
        collision_time_differences = [
            second["elapsed_time"] - first["elapsed_time"]
            for first, second in complete
            if first["episode_outcome"] == second["episode_outcome"] == "ego_collision"
        ]
        complete_count = len(complete)
        return {
            "complete_same_update_pairs": complete_count,
            "incomplete_pairs": incomplete + len(self.paired_cross_update_keys),
            "cross_update_incomplete_pairs": len(self.paired_cross_update_keys),
            "discordant_pairs": discordant,
            "discordant_pair_rate": discordant / complete_count if complete_count else 0.0,
            "both_collision_pairs": both_collision,
            "both_safe_pairs": both_safe,
            "paired_return_difference_member1_minus_member0": self._difference_statistics(return_differences),
            "paired_collision_time_difference_member1_minus_member0": self._difference_statistics(
                collision_time_differences
            ),
            "paired_scenario_coverage": len(
                {first["scenario_id"] for first, _second in complete}
            ),
        }

    def _action_statistics(self, index: int) -> dict[str, float]:
        if self.action_count == 0:
            raise RuntimeError("PPO rollout produced no actions")
        mean = self.action_sum[index] / self.action_count
        variance = max(self.action_sum_squares[index] / self.action_count - mean * mean, 0.0)
        values = {
            "mean": float(mean),
            "std": float(np.sqrt(variance)),
            "min": float(self.action_min[index]),
            "max": float(self.action_max[index]),
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise RuntimeError(f"Non-finite PPO action statistics: {values}")
        return values

    def _on_rollout_end(self) -> None:
        self.update += 1
        outcomes = ("ego_collision", "follow", "overtake")
        self.latest = {
            "update": self.update,
            "transitions": self.transitions,
            "completed_episodes": dict(sorted(self.completed.items())),
            "completed_episodes_by_sampler_branch": {
                branch: {
                    outcome: int(self.completed_by_branch[(branch, outcome)])
                    for outcome in outcomes
                }
                for branch in sorted(self.branches)
            },
            "completed_episodes_by_env_role": {
                role: {
                    outcome: int(self.completed_by_role[(role, outcome)])
                    for outcome in outcomes
                }
                for role in ("hard", "ordinary")
            },
            "sampler_branch_transitions": dict(sorted(self.branches.items())),
            "env_role_transitions": {
                role: int(self.roles[role]) for role in ("hard", "ordinary")
            },
            "unique_scenario_count": len(self.scenario_ids),
            "unique_hard_scenario_count": len(self.hard_scenario_ids),
            "action_statistics": {
                "steering": self._action_statistics(0),
                "speed": self._action_statistics(1),
            },
            "reward_component_means": {
                key: float(self.reward_sums[key] / self.transitions)
                for key in ("reward_progress", "reward_relative", "reward_margin", "reward_collision", "reward_total")
            },
            "reward_component_sums": {
                key: float(self.reward_sums[key])
                for key in ("reward_progress", "reward_relative", "reward_margin", "reward_collision", "reward_total")
            },
            "reward_component_sums_by_env_role": {
                role: {
                    key: float(self.reward_sums_by_role[(role, key)])
                    for key in ("reward_progress", "reward_relative", "reward_margin", "reward_collision", "reward_total")
                }
                for role in ("hard", "ordinary")
            },
            "hard_truncations": int(self.hard_truncations),
            "mean_episode_length_steps_by_env_role": {
                role: (
                    float(self.episode_length_steps_by_role[role])
                    / max(sum(self.completed_by_role[(role, outcome)] for outcome in outcomes), 1)
                )
                for role in ("hard", "ordinary")
            },
            "paired_telemetry": self._paired_telemetry(),
        }


def build_sampler(config: ppo_config.PPOConfig) -> FixedMixtureScenarioSampler:
    full_pool = training_scenarios()
    hard_scenarios, hard_ids, manifest = load_hard_pool(
        config.hard_pool,
        config.hard_pool_manifest,
    )
    return FixedMixtureScenarioSampler(
        full_pool,
        hard_ids,
        collision_probability=config.hard_sampling_probability,
        hard_scenarios=hard_scenarios,
        hard_pool_id=str(manifest["pool_id"]),
        hard_sampling_mode=config.hard_sampling_mode,
    )


def make_training_env(
    rank: int,
    sampler: FixedMixtureScenarioSampler,
    config: ppo_config.PPOConfig,
    seed: int,
):
    """Build one legacy F110 simulator behind the frozen Gymnasium contract."""
    def factory() -> End2RaceGymnasiumEnv:
        import gym
        from f110_gym.envs.base_classes import Integrator

        core = gym.make(
            "f110-v0",
            map=str(ppo_config.PROJECT_ROOT / "f1tenth_racetracks" / "Austin" / "Austin_map"),
            map_ext=".png",
            num_agents=2,
            timestep=0.01,
            integrator=Integrator.RK4,
            seed=seed + rank,
        )
        fixed_role = None
        if config.fixed_hard_env_count is not None:
            fixed_role = "hard" if rank < config.fixed_hard_env_count else "ordinary"
        horizon = (
            config.hard_horizon_s if fixed_role == "hard" else config.ordinary_horizon_s
        )
        reset_provider = (
            _fixed_reset_provider(rank, sampler, config, seed, fixed_role)
            if fixed_role is not None
            else sampler
        )
        return End2RaceGymnasiumEnv(
            core,
            sim_duration=horizon,
            reset_provider=reset_provider,
            ego_index=0,
            opponent_controller=LatticePlannerOpponentController(),
            transition_reward=PPOTransitionReward(
                ProgressProjector.from_csv(),
                margin_weight=config.margin_weight,
                margin_threshold=config.margin_threshold,
            ),
            privileged_critic=config.critic_profile == "C3_PRIVILEGED_PHYSICAL",
        )

    return factory


def _external_reset_required(_rng: np.random.Generator):
    """Reject unscheduled worker resets without capturing a sampler object."""

    raise RuntimeError("Subprocess training resets must be supplied by the parent scheduler")


def make_subprocess_training_env(
    rank: int,
    config: ppo_config.PPOConfig,
    seed: int,
):
    """Build one worker env that contains no sampler or visit-count state."""

    def factory() -> End2RaceGymnasiumEnv:
        import gym
        from f110_gym.envs.base_classes import Integrator

        core = gym.make(
            "f110-v0",
            map=str(ppo_config.PROJECT_ROOT / "f1tenth_racetracks" / "Austin" / "Austin_map"),
            map_ext=".png",
            num_agents=2,
            timestep=0.01,
            integrator=Integrator.RK4,
            seed=seed + rank,
        )
        fixed_role = None
        if config.fixed_hard_env_count is not None:
            fixed_role = "hard" if rank < config.fixed_hard_env_count else "ordinary"
        horizon = config.hard_horizon_s if fixed_role == "hard" else config.ordinary_horizon_s
        return End2RaceGymnasiumEnv(
            core,
            sim_duration=horizon,
            reset_provider=_external_reset_required,
            ego_index=0,
            opponent_controller=LatticePlannerOpponentController(),
            transition_reward=PPOTransitionReward(
                ProgressProjector.from_csv(),
                margin_weight=config.margin_weight,
                margin_threshold=config.margin_threshold,
            ),
            privileged_critic=config.critic_profile == "C3_PRIVILEGED_PHYSICAL",
        )

    return factory


def _fixed_reset_provider(
    rank: int,
    sampler: FixedMixtureScenarioSampler,
    config: ppo_config.PPOConfig,
    seed: int,
    fixed_role: str,
):
    pair_episode_ordinal = 0

    def provider(rng: np.random.Generator):
        nonlocal pair_episode_ordinal
        if config.paired_hard_sampling and fixed_role == "hard":
            spec = sampler.reset_spec(
                rng,
                env_role="hard",
                pair_seed=seed,
                pair_group=rank // config.hard_pair_size,
                pair_member=rank % config.hard_pair_size,
                pair_episode_ordinal=pair_episode_ordinal,
            )
            pair_episode_ordinal += 1
            return spec
        return sampler.reset_spec(rng, env_role=fixed_role)

    return provider


def build_model(
    vector_env: VecEnv,
    config: ppo_config.PPOConfig,
    seed: int,
) -> End2RaceRecurrentPPO:
    """Build the fixed recurrent PPO algorithm and optimizer groups."""
    return End2RaceRecurrentPPO(
        End2RaceGRUPolicy,
        vector_env,
        learning_rate=1.0,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=ppo_config.GAMMA,
        gae_lambda=ppo_config.GAE_LAMBDA,
        clip_range=ppo_config.CLIP_RANGE,
        clip_range_vf=ppo_config.CLIP_RANGE_VF,
        normalize_advantage=ppo_config.NORMALIZE_ADVANTAGE,
        vf_coef=ppo_config.VF_COEF,
        ent_coef=ppo_config.ENT_COEF,
        max_grad_norm=ppo_config.MAX_GRAD_NORM,
        target_kl=config.target_kl,
        seed=seed,
        device=ppo_config.DEVICE,
        policy_kwargs={
            "checkpoint_path": ppo_config.BC_CHECKPOINT,
            "hidden_scale": 4,
            "critic_hidden_size": 64,
            "critic_profile": config.critic_profile,
            "gru_lr": config.gru_lr,
            "head_lr": config.head_lr,
            "steering_distribution": config.steering_distribution,
            "steering_latent_std": config.steering_latent_std,
            "speed_physical_std": config.speed_physical_std,
        },
        verbose=1,
    )


def build_training_vector_env(
    sampler: FixedMixtureScenarioSampler,
    config: ppo_config.PPOConfig,
    seed: int,
    *,
    subprocess: bool = True,
    worker_count: int = ppo_config.ENV_WORKERS,
) -> VecEnv:
    """Build the training VecEnv without changing the scenario scheduler."""
    if subprocess:
        factories = [
            make_subprocess_training_env(rank, config, seed)
            for rank in range(ppo_config.N_ENVS)
        ]
        return CentralScheduleSubprocVecEnv(
            factories,
            sampler=sampler,
            config=config,
            seed=seed,
            worker_count=worker_count,
            start_method=ppo_config.ENV_START_METHOD,
        )
    factories = [make_training_env(rank, sampler, config, seed) for rank in range(ppo_config.N_ENVS)]
    return DummyVecEnv(factories)


def save_actor(model: End2RaceRecurrentPPO, destination: Path) -> str:
    """Save, validate, and hash one BC-compatible actor checkpoint."""
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    state = {name: tensor.detach().cpu() for name, tensor in model.policy.actor_checkpoint_state_dict().items()}
    bc_state = torch.load(ppo_config.BC_CHECKPOINT, map_location="cpu", weights_only=True)
    if set(state) != set(bc_state):
        raise RuntimeError("Saved actor keys do not match the BC checkpoint schema")
    for name, tensor in state.items():
        if tensor.shape != bc_state[name].shape or tensor.dtype != bc_state[name].dtype:
            raise RuntimeError(f"Saved actor tensor does not match the BC checkpoint schema: {name}")
    torch.save(state, destination)
    fresh = End2Race(mask_prob=0.0, hidden_scale=4)
    fresh.load_state_dict(torch.load(destination, map_location="cpu", weights_only=True), strict=True)
    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_record(
    config: ppo_config.PPOConfig,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    record = asdict(config)
    record.update(
        {
            "seed": seed,
            "output_dir": str(output_dir),
            "device": ppo_config.DEVICE,
            "n_envs": ppo_config.N_ENVS,
            "env_workers": ppo_config.ENV_WORKERS,
            "env_start_method": ppo_config.ENV_START_METHOD,
            "n_epochs": config.n_epochs,
            "gamma": ppo_config.GAMMA,
            "gae_lambda": ppo_config.GAE_LAMBDA,
            "clip_range": ppo_config.CLIP_RANGE,
            "clip_range_vf": ppo_config.CLIP_RANGE_VF,
            "normalize_advantage": ppo_config.NORMALIZE_ADVANTAGE,
            "vf_coef": ppo_config.VF_COEF,
            "ent_coef": ppo_config.ENT_COEF,
            "max_grad_norm": ppo_config.MAX_GRAD_NORM,
            "target_kl": config.target_kl,
            "gru_lr": config.gru_lr,
            "head_lr": config.head_lr,
            "critic_lr": ppo_config.CRITIC_LR,
            "steering_latent_std": config.steering_latent_std,
            "speed_physical_std": config.speed_physical_std,
            "sim_duration": ppo_config.SIM_DURATION,
            "bc_checkpoint": str(ppo_config.BC_CHECKPOINT),
        }
    )
    record["transitions_per_update"] = ppo_config.N_ENVS * config.n_steps
    record["minibatches_per_epoch"] = ppo_config.N_ENVS * config.n_steps // config.batch_size
    record["minibatches_per_update"] = record["minibatches_per_epoch"]
    record["planned_optimizer_steps_per_update"] = record["minibatches_per_epoch"] * config.n_epochs
    record["total_optimizer_steps"] = record["planned_optimizer_steps_per_update"] * config.updates
    return record


def _sampler_summary(
    config: ppo_config.PPOConfig,
    sampler: FixedMixtureScenarioSampler,
) -> dict[str, Any]:
    visit_counts = dict(sorted(sampler.visit_counts.items()))
    visits = np.asarray(list(visit_counts.values()), dtype=np.float64)
    visited = int(np.count_nonzero(visits))
    return {
        "hard_pool": config.hard_pool,
        "hard_pool_id": sampler.hard_pool_id,
        "hard_pool_size": len(visit_counts),
        "hard_sampling_probability": config.hard_sampling_probability,
        "hard_sampling_mode": config.hard_sampling_mode,
        "visited_hard_scenarios": visited,
        "unvisited_hard_scenarios": len(visit_counts) - visited,
        "visit_min": int(visits.min()),
        "visit_max": int(visits.max()),
        "visit_mean": float(visits.mean()),
        "visit_std": float(visits.std()),
        "visit_counts": visit_counts,
    }


def train(
    config: ppo_config.PPOConfig,
    seed: int,
    output_dir: Path,
    *,
    screen_pause: bool = False,
) -> Path:
    """Run one fresh profile in a new output directory."""

    if not ppo_config.BC_CHECKPOINT.is_file():
        raise FileNotFoundError(f"BC checkpoint does not exist: {ppo_config.BC_CHECKPOINT}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    sampler = build_sampler(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    write_json(output_dir / "resolved_config.json", _resolved_record(config, seed, output_dir))
    checkpoint_manifest: dict[str, Any] = {
        "config": config.name,
        "seed": seed,
        "checkpoints": [],
    }
    write_json(output_dir / "checkpoint_manifest.json", checkpoint_manifest)
    status_path = output_dir / "run_status.json"
    run_status: dict[str, Any] = {
        "schema_version": 1,
        "experiment": config.name,
        "status": "RUNNING",
        "config": config.name,
        "seed": seed,
        "last_completed_update": 0,
        "stop_reason": None,
    }
    write_json_atomic(status_path, run_status)
    set_random_seed(seed)

    vector_env: VecEnv | None = None
    try:
        vector_env = build_training_vector_env(sampler, config, seed)
        vector_env.seed(seed)
        model = build_model(vector_env, config, seed)
        callback = PPOTrainingCallback()
        bc_state = torch.load(ppo_config.BC_CHECKPOINT, map_location="cpu", weights_only=True)
        previous_actor_state = {name: tensor.detach().cpu().clone() for name, tensor in bc_state.items()}
        initial_log_std = model.policy.log_std.detach().cpu().clone()
        minibatches_per_epoch = ppo_config.N_ENVS * config.n_steps // config.batch_size
        planned_optimizer_steps = minibatches_per_epoch * config.n_epochs
        for update in range(1, config.updates + 1):
            vector_env.env_method("set_policy_update_index", update)
            optimizer_before = _optimizer_step(model, require_initialized=update > 1)
            with torch.autograd.set_multithreading_enabled(config.autograd_multithreading):
                model.learn(
                    total_timesteps=ppo_config.N_ENVS * config.n_steps,
                    callback=callback,
                    log_interval=None,
                    reset_num_timesteps=update == 1,
                    progress_bar=False,
                )
            optimizer_after = _optimizer_step(model, require_initialized=True)
            actual_optimizer_steps = optimizer_after["max"] - optimizer_before["max"]
            if actual_optimizer_steps <= 0 or actual_optimizer_steps > planned_optimizer_steps:
                raise RuntimeError(
                    f"Invalid optimizer-step count at update {update}: "
                    f"{actual_optimizer_steps} not in [1, {planned_optimizer_steps}]"
                )
            actor_delta = _actor_delta_record(model, bc_state, previous_actor_state, initial_log_std)
            if (
                actor_delta["frozen_actor"]["max_abs_delta_from_bc"] != 0.0
                or actor_delta["log_std_max_abs_delta_from_initial"] != 0.0
            ):
                raise RuntimeError(f"Frozen actor state drifted at update {update}")
            metrics = {
                "update": update,
                "num_timesteps": int(model.num_timesteps),
                "rollout": callback.latest,
                "planned_optimizer_steps": planned_optimizer_steps,
                "actual_optimizer_steps": actual_optimizer_steps,
                "optimizer_step_min": optimizer_after["min"],
                "optimizer_step_max": optimizer_after["max"],
                "optimizer_active_parameters": optimizer_after["active_parameters"],
                "target_kl_early_stop": actual_optimizer_steps < planned_optimizer_steps,
                "effective_epoch_fraction": actual_optimizer_steps / minibatches_per_epoch,
                "actor_delta_from_bc": actor_delta,
            }
            for key in ("loss", "policy_gradient_loss", "value_loss", "approx_kl", "clip_fraction", "explained_variance"):
                logger_key = f"train/{key}"
                if logger_key in model.logger.name_to_value:
                    metrics[key] = float(model.logger.name_to_value[logger_key])
            if "approx_kl" not in metrics:
                raise RuntimeError(f"PPO logger did not report approx_kl at update {update}")
            _assert_finite(metrics)
            append_json(output_dir / "training_metrics.jsonl", metrics)
            previous_actor_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.policy.actor_checkpoint_state_dict().items()
            }
            run_status["last_completed_update"] = update
            if (
                config.update_kl_guardrail is not None
                and metrics["approx_kl"] > config.update_kl_guardrail
            ):
                run_status.update(
                    {
                        "status": "STOPPED_KL_GUARDRAIL",
                        "stop_reason": (
                            f"approx_kl {metrics['approx_kl']:.9g} exceeds "
                            f"{config.update_kl_guardrail:.9g}"
                        ),
                    }
                )
                write_json_atomic(status_path, run_status)
                break
            if update in config.checkpoint_updates:
                filename = f"end2race_ppo_{config.name}_u{update:04d}_s{seed}.pth"
                actor_path = checkpoint_dir / filename
                checkpoint_manifest["checkpoints"].append(
                    {
                        "update": update,
                        "path": actor_path.relative_to(output_dir).as_posix(),
                        "sha256": save_actor(model, actor_path),
                    }
                )
                write_json(output_dir / "checkpoint_manifest.json", checkpoint_manifest)
            if screen_pause and update == config.checkpoint_updates[0]:
                write_json(output_dir / "sampler_summary.json", _sampler_summary(config, sampler))
                run_status["status"] = "PAUSED_SCREEN"
                write_json_atomic(status_path, run_status)
                print("SCREEN_PAUSED: enter continue or stop", flush=True)
                decision = input().strip().lower()
                if decision == "stop":
                    run_status.update(
                        {
                            "status": "STOPPED_SCREEN",
                            "stop_reason": "Stopped by fixed screen-stage decision",
                        }
                    )
                    write_json_atomic(status_path, run_status)
                    break
                if decision != "continue":
                    raise ValueError(f"Unknown screen decision: {decision!r}")
                run_status["status"] = "RUNNING"
                write_json_atomic(status_path, run_status)
            run_status["status"] = "COMPLETED" if update == config.updates else "RUNNING"
            write_json_atomic(status_path, run_status)
    except Exception as error:
        if run_status["status"] != "STOPPED_KL_GUARDRAIL":
            message = f"{type(error).__name__}: {error}"
            if "Non-finite" in message:
                failure_status = "FAILED_NONFINITE"
            elif "checkpoint" in message.lower() or "actor tensor" in message.lower():
                failure_status = "FAILED_CHECKPOINT"
            else:
                failure_status = "FAILED_RUNTIME"
            run_status.update(
                {
                    "status": failure_status,
                    "stop_reason": message,
                }
            )
            write_json_atomic(status_path, run_status)
        raise
    finally:
        if vector_env is not None:
            vector_env.close()
        write_json(output_dir / "sampler_summary.json", _sampler_summary(config, sampler))

    return output_dir


if __name__ == "__main__":
    args = parse_arguments()
    config = ppo_config.get_config(args.config)
    output_dir = args.output_dir or (
        ppo_config.PROJECT_ROOT / "runs" / "ppo" / f"{config.name}_seed{args.seed}"
    )
    run_dir = train(config, args.seed, output_dir, screen_pause=args.screen_pause)
    print(f"PPO_RUN_DIR={run_dir}")
    with (run_dir / "run_status.json").open("r", encoding="utf-8") as handle:
        final_status = json.load(handle)
    if final_status["status"] not in {"COMPLETED", "STOPPED_SCREEN"}:
        raise SystemExit(2)
