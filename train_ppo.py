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

from model import End2Race
from ppo import config as ppo_config
from ppo.environment import End2RaceGymnasiumEnv, LatticePlannerOpponentController
from ppo.policy import End2RaceGRUPolicy, End2RaceRecurrentPPO
from ppo.reward import PPOTransitionReward, ProgressProjector
from ppo.scenarios import FixedMixtureScenarioSampler, load_hard_pool, training_scenarios


def parse_arguments() -> argparse.Namespace:
    """Parse the three supported run arguments."""
    parser = argparse.ArgumentParser(description="Train End2Race PPO")
    parser.add_argument("--config", required=True, choices=ppo_config.CONFIGS)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output_dir", type=Path, default=None)
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


def append_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


class PPOTrainingCallback(BaseCallback):
    """Collect one concise metrics record per rollout update."""

    def __init__(self) -> None:
        super().__init__()
        self.update = 0
        self.latest: dict[str, Any] = {}

    def _on_rollout_start(self) -> None:
        self.transitions = 0
        self.completed = Counter()
        self.completed_by_branch = Counter()
        self.branches = Counter()
        self.reward_sums = Counter()
        self.scenario_ids: set[str] = set()
        self.hard_scenario_ids: set[str] = set()
        self.action_count = 0
        self.action_sum = np.zeros(2, dtype=np.float64)
        self.action_sum_squares = np.zeros(2, dtype=np.float64)
        self.action_min = np.full(2, np.inf, dtype=np.float64)
        self.action_max = np.full(2, -np.inf, dtype=np.float64)

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
            scenario_id = str(info["scenario_id"])
            self.branches[branch] += 1
            self.scenario_ids.add(scenario_id)
            if branch in {"bc_ego_collision", "hard_pool"}:
                self.hard_scenario_ids.add(scenario_id)
            for key in ("reward_progress", "reward_relative", "reward_margin", "reward_collision", "reward_total"):
                self.reward_sums[key] += float(info[key])
            if dones[index]:
                if bool(info["ego_collision"]):
                    outcome = "ego_collision"
                elif float(info["relative_position_m"]) > 0.0:
                    outcome = "overtake"
                else:
                    outcome = "follow"
                self.completed[outcome] += 1
                self.completed_by_branch[(branch, outcome)] += 1
        return True

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
            "sampler_branch_transitions": dict(sorted(self.branches.items())),
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
        }


def build_sampler(config: ppo_config.PPOConfig) -> FixedMixtureScenarioSampler:
    full_pool = training_scenarios()
    hard_scenarios, hard_ids, manifest = load_hard_pool(config.hard_pool)
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
        return End2RaceGymnasiumEnv(
            core,
            sim_duration=ppo_config.SIM_DURATION,
            reset_provider=sampler,
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


def build_model(
    vector_env: DummyVecEnv,
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
            "steering_latent_std": config.steering_latent_std,
            "speed_physical_std": config.speed_physical_std,
        },
        verbose=1,
    )


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
    record["minibatches_per_update"] = ppo_config.N_ENVS * config.n_steps // config.batch_size
    record["total_optimizer_steps"] = record["minibatches_per_update"] * config.n_epochs * config.updates
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


def train(config: ppo_config.PPOConfig, seed: int, output_dir: Path) -> Path:
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
    set_random_seed(seed)

    vector_env = DummyVecEnv(
        [make_training_env(rank, sampler, config, seed) for rank in range(ppo_config.N_ENVS)]
    )
    vector_env.seed(seed)
    model = build_model(vector_env, config, seed)
    callback = PPOTrainingCallback()
    try:
        for update in range(1, config.updates + 1):
            model.learn(
                total_timesteps=ppo_config.N_ENVS * config.n_steps,
                callback=callback,
                log_interval=None,
                reset_num_timesteps=update == 1,
                progress_bar=False,
            )
            metrics = {"update": update, "num_timesteps": int(model.num_timesteps), "rollout": callback.latest}
            for key in ("loss", "policy_gradient_loss", "value_loss", "approx_kl", "clip_fraction", "explained_variance"):
                logger_key = f"train/{key}"
                if logger_key in model.logger.name_to_value:
                    metrics[key] = float(model.logger.name_to_value[logger_key])
            scalar_values = [value for value in metrics.values() if isinstance(value, float)]
            if not np.isfinite(scalar_values).all():
                raise RuntimeError(f"Non-finite PPO metrics at update {update}")
            append_json(output_dir / "training_metrics.jsonl", metrics)
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
    finally:
        vector_env.close()

    write_json(output_dir / "sampler_summary.json", _sampler_summary(config, sampler))
    return output_dir


if __name__ == "__main__":
    args = parse_arguments()
    config = ppo_config.get_config(args.config)
    output_dir = args.output_dir or (
        ppo_config.PROJECT_ROOT / "runs" / "ppo" / f"{config.name}_seed{args.seed}"
    )
    run_dir = train(config, args.seed, output_dir)
    print(f"PPO_RUN_DIR={run_dir}")
