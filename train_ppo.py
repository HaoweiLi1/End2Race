#!/usr/bin/env python3
"""Train and evaluate one fixed End2Race PPO profile."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from eval_multiagent import evaluate_segment
from model import End2Race
from ppo import config as ppo_config
from ppo.environment import End2RaceGymnasiumEnv, LatticePlannerOpponentController
from ppo.policy import End2RaceGRUPolicy, End2RaceRecurrentPPO
from ppo.reward import PPOTransitionReward, ProgressProjector
from ppo.scenarios import FixedMixtureScenarioSampler, ScenarioSpec, evaluation_scenarios, load_hard_pool, training_scenarios

EVALUATION_MODEL: End2Race | None = None
EVALUATION_MODEL_PATH: str | None = None
EVALUATION_DEVICE = torch.device("cpu")

def parse_arguments() -> argparse.Namespace:
    """Parse the five supported run arguments."""
    parser = argparse.ArgumentParser(description="Train End2Race PPO")
    parser.add_argument("--version", required=True, choices=ppo_config.VERSIONS)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--evaluation_workers", type=int, default=8)
    return parser.parse_args()


def set_random_seed(seed: int, device: str) -> None:
    """Seed every RNG used by the formal run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
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
        self.branches = Counter()
        self.reward_sums = Counter()

    def _on_step(self) -> bool:
        infos = list(self.locals["infos"])
        dones = np.asarray(self.locals["dones"], dtype=bool)
        for index, info in enumerate(infos):
            self.transitions += 1
            self.branches[str(info["sampler_branch"])] += 1
            for key in ("reward_progress", "reward_relative", "reward_collision", "reward_total"):
                self.reward_sums[key] += float(info[key])
            if dones[index]:
                if bool(info["ego_collision"]):
                    outcome = "ego_collision"
                elif float(info["relative_position_m"]) > 0.0:
                    outcome = "overtake"
                else:
                    outcome = "follow"
                self.completed[outcome] += 1
        return True

    def _on_rollout_end(self) -> None:
        self.update += 1
        self.latest = {
            "update": self.update,
            "transitions": self.transitions,
            "completed_episodes": dict(sorted(self.completed.items())),
            "sampler_branch_transitions": dict(sorted(self.branches.items())),
            "reward_component_means": {
                key: float(self.reward_sums[key] / self.transitions)
                for key in ("reward_progress", "reward_relative", "reward_collision", "reward_total")
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


def make_training_env(rank: int, sampler: FixedMixtureScenarioSampler, config: ppo_config.PPOConfig):
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
            seed=config.seed + rank,
        )
        return End2RaceGymnasiumEnv(
            core,
            sim_duration=ppo_config.SIM_DURATION,
            reset_provider=sampler,
            ego_index=0,
            opponent_controller=LatticePlannerOpponentController(),
            transition_reward=PPOTransitionReward(ProgressProjector.from_csv()),
            privileged_critic=config.critic_profile == "C3_PRIVILEGED_PHYSICAL",
        )

    return factory


def build_model(vector_env: DummyVecEnv, config: ppo_config.PPOConfig) -> End2RaceRecurrentPPO:
    """Build the fixed recurrent PPO algorithm and optimizer groups."""
    return End2RaceRecurrentPPO(
        End2RaceGRUPolicy,
        vector_env,
        learning_rate=1.0,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=ppo_config.N_EPOCHS,
        gamma=ppo_config.GAMMA,
        gae_lambda=ppo_config.GAE_LAMBDA,
        clip_range=ppo_config.CLIP_RANGE,
        clip_range_vf=ppo_config.CLIP_RANGE_VF,
        normalize_advantage=ppo_config.NORMALIZE_ADVANTAGE,
        vf_coef=ppo_config.VF_COEF,
        ent_coef=ppo_config.ENT_COEF,
        max_grad_norm=ppo_config.MAX_GRAD_NORM,
        target_kl=ppo_config.TARGET_KL,
        seed=config.seed,
        device=config.device,
        policy_kwargs={
            "checkpoint_path": ppo_config.BC_CHECKPOINT,
            "hidden_scale": 4,
            "critic_hidden_size": 64,
            "critic_profile": config.critic_profile,
        },
        verbose=1,
    )


def save_actor(model: End2RaceRecurrentPPO, destination: Path) -> None:
    """Save the actor checkpoint and confirm it strict-loads into a fresh End2Race."""
    destination.parent.mkdir(parents=True, exist_ok=False)
    state = {name: tensor.detach().cpu() for name, tensor in model.policy.actor_checkpoint_state_dict().items()}
    torch.save(state, destination)
    fresh = End2Race(mask_prob=0.0, hidden_scale=4)
    fresh.load_state_dict(torch.load(destination, map_location="cpu", weights_only=True), strict=True)


def _evaluation_worker_init(model_path: str) -> None:
    global EVALUATION_MODEL, EVALUATION_MODEL_PATH
    torch.set_num_threads(1)
    model = End2Race(mask_prob=0.0, hidden_scale=4).to(EVALUATION_DEVICE)
    state = torch.load(model_path, map_location=EVALUATION_DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    EVALUATION_MODEL = model
    EVALUATION_MODEL_PATH = model_path


def _evaluate_scenario(scenario: ScenarioSpec) -> dict[str, Any]:
    result = evaluate_segment(
        EVALUATION_MODEL, EVALUATION_DEVICE, 0.0, scenario.map_name, scenario.ego_idx,
        scenario.interval_idx, scenario.ego_raceline, scenario.opp_raceline,
        scenario.opp_speedscale, scenario.sim_duration, False, False,
        EVALUATION_MODEL_PATH, None, "ego", scenario.scenario_id,
    )
    metrics = result["episode_metrics"]
    return {
        "scenario_id": scenario.scenario_id,
        "outcome": metrics["outcome"],
        "opponent_only_collision": bool(metrics["opponent_only_collision"]),
    }


def evaluate_checkpoint(
    model_path: Path,
    scenarios: Sequence[ScenarioSpec],
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Evaluate all 600 deterministic ego-scope cases, failing on any error."""

    resolved = str(model_path.resolve())
    if workers == 1:
        _evaluation_worker_init(resolved)
        rows = [_evaluate_scenario(scenario) for scenario in scenarios]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_evaluation_worker_init,
            initargs=(resolved,),
        ) as executor:
            rows = list(executor.map(_evaluate_scenario, scenarios, chunksize=1))
    outcomes = Counter(row["outcome"] for row in rows)
    summary = {
        "ego_collision": int(outcomes["ego_collision"]),
        "follow": int(outcomes["follow"]),
        "overtake": int(outcomes["overtake"]),
        "opponent_only_collision": sum(bool(row["opponent_only_collision"]) for row in rows),
        "total": len(rows),
    }
    classified = summary["ego_collision"] + summary["follow"] + summary["overtake"]
    if summary["total"] != 600 or classified != 600:
        raise RuntimeError(f"Invalid 600-case evaluation summary: {summary}")
    return rows, summary


def select_checkpoint(baseline: dict[str, int], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    floor = math.ceil(0.95 * 346)
    eligible = [row for row in candidates if row["metrics"]["overtake"] >= floor]
    ranking = lambda row: (row["metrics"]["ego_collision"], -row["metrics"]["overtake"], row["update"])
    best = min(eligible, key=ranking) if eligible else None
    return {
        "status": "selected" if best is not None else "no_eligible_checkpoint",
        "overtake_floor": floor,
        "eligible_updates": [row["update"] for row in eligible],
        "best": best,
        "candidates": candidates,
    }


def _resolved_record(config: ppo_config.PPOConfig) -> dict[str, Any]:
    record = asdict(config)
    record["output_dir"] = str(config.output_dir)
    record["n_envs"] = ppo_config.N_ENVS
    record["n_epochs"] = ppo_config.N_EPOCHS
    record["transitions_per_update"] = ppo_config.N_ENVS * config.n_steps
    record["minibatches_per_update"] = ppo_config.N_ENVS * config.n_steps // config.batch_size
    record["total_optimizer_steps"] = record["minibatches_per_update"] * ppo_config.N_EPOCHS * config.updates
    return record


def train(config: ppo_config.PPOConfig) -> Path:
    """Run one fresh profile without resume or directory fallback."""

    if not ppo_config.BC_CHECKPOINT.is_file():
        raise FileNotFoundError(f"BC checkpoint does not exist: {ppo_config.BC_CHECKPOINT}")
    if config.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")
    sampler = build_sampler(config)
    panel = evaluation_scenarios()
    config.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(config.output_dir / "resolved_config.json", _resolved_record(config))
    set_random_seed(config.seed, config.device)

    _baseline_rows, baseline = evaluate_checkpoint(ppo_config.BC_CHECKPOINT, panel, config.evaluation_workers)
    vector_env = DummyVecEnv([make_training_env(rank, sampler, config) for rank in range(ppo_config.N_ENVS)])
    vector_env.seed(config.seed)
    model = build_model(vector_env, config)
    callback = PPOTrainingCallback()
    candidates: list[dict[str, Any]] = []
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
            append_json(config.output_dir / "training_metrics.jsonl", metrics)
            if update in config.evaluation_updates:
                actor_path = config.output_dir / "checkpoints" / f"update_{update:04d}" / "actor_only.pth"
                save_actor(model, actor_path)
                _rows, summary = evaluate_checkpoint(actor_path, panel, config.evaluation_workers)
                candidates.append({"update": update, "metrics": summary})
    finally:
        vector_env.close()

    write_json(config.output_dir / "evaluations.json", {"baseline": baseline, "candidates": candidates})
    write_json(config.output_dir / "selection.json", select_checkpoint(baseline, candidates))
    return config.output_dir


if __name__ == "__main__":
    args = parse_arguments()
    output_dir = args.output_dir or ppo_config.PROJECT_ROOT / "runs" / "ppo" / args.version
    config = ppo_config.get_config(args.version, args.seed, args.device, output_dir, args.evaluation_workers)
    run_dir = train(config)
    print(f"PPO_RUN_DIR={run_dir}")
