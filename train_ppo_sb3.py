#!/usr/bin/env python3
"""Train and evaluate the End2Race PPO V1 pilot."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
from contextlib import contextmanager
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from model import End2Race
from rl.end2race_gymnasium_env import End2RaceGymnasiumEnv, LatticePlannerOpponentController
from rl.end2race_recurrent_ppo import End2RaceRecurrentPPO
from rl.ppo_callbacks import PPOV1MetricsCallback
from rl.ppo_reward import PPOV1TransitionReward, ProgressProjector
from rl.ppo_scenarios import (
    FixedMixtureScenarioSampler,
    ScenarioSpec,
    classify_bc_ego_collisions,
    evaluation_scenarios,
    training_scenarios,
    scenario_from_dict,
)
from rl.sb3_end2race_policy import DEFAULT_BC_CHECKPOINT, End2RaceGRUPolicy
from experiments.ppo_v1_2.config_schema import CRITIC_PROFILES, HARD_POOL_IDS, SAMPLING_MODES, resolve_config as resolve_v1_2_config
from experiments.ppo_v1_2.experiment_spec import canonical_hash
from experiments.ppo_v1_2.selectors import checkpoint_flags
from utils import atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATION_MODEL: End2Race | None = None
EVALUATION_DEVICE = torch.device("cpu")

DEFAULT_CONFIG: dict[str, Any] = {
    "n_envs": 16,
    "n_steps": 800,
    "batch_size": 800,
    "n_epochs": 1,
    "gamma": 0.999,
    "gae_lambda": 0.995,
    "clip_range": 0.10,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "vf_coef": 0.5,
    "ent_coef": 0.0,
    "max_grad_norm": 0.5,
    "target_kl": None,
    "device": "cuda",
    "vector_env": "DummyVecEnv",
    "master_seed": 20260715,
    "updates": 20,
    "lr_scale": 1.0,
    "collision_sampling_probability": 0.25,
}
V1_1_EVALUATION_UPDATES = (2, 3, 5, 10, 15, 20)


def parse_evaluation_updates(value: str) -> tuple[int, ...]:
    try:
        updates = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("evaluation updates must be comma-separated integers") from error
    if not updates:
        raise argparse.ArgumentTypeError("evaluation updates must not be empty")
    return updates


def parse_int_list(value: str) -> tuple[int, ...]:
    return parse_evaluation_updates(value)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End2Race PPO V1/V1.2 trainer")
    parser.add_argument("--experiment-profile", choices=("ppo_v1", "ppo_v1_2"), default="ppo_v1")
    parser.add_argument("--config-json", type=Path, default=None, help="Exact resolved V1.2 config supplied by the sweep runner")
    parser.add_argument("--run-root", type=Path, default=Path("runs/ppo_v1"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--updates", type=int, default=DEFAULT_CONFIG["updates"])
    parser.add_argument("--n-envs", type=int, default=DEFAULT_CONFIG["n_envs"])
    parser.add_argument("--n-steps", type=int, default=DEFAULT_CONFIG["n_steps"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--device", choices=("cuda", "cpu"), default=DEFAULT_CONFIG["device"])
    parser.add_argument("--master-seed", "--seed", dest="master_seed", type=int, default=DEFAULT_CONFIG["master_seed"])
    parser.add_argument("--lr-scale", type=float, default=DEFAULT_CONFIG["lr_scale"])
    parser.add_argument("--evaluation-workers", type=int, default=8)
    parser.add_argument(
        "--collision-sampling-probability",
        type=float,
        default=DEFAULT_CONFIG["collision_sampling_probability"],
    )
    parser.add_argument("--critic-profile", choices=CRITIC_PROFILES, default="C0_RAW_SINGLE_FRAME")
    parser.add_argument("--hard-pool-id", choices=HARD_POOL_IDS, default="H0_CURRENT_DET")
    parser.add_argument("--hard-pool-manifest", type=Path, default=None)
    parser.add_argument("--hard-sampling-probability", type=float, default=0.50)
    parser.add_argument("--hard-sampling-mode", choices=SAMPLING_MODES, default="with_replacement")
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--gru-lr", type=float, default=1.0e-6)
    parser.add_argument("--head-lr", type=float, default=1.0e-5)
    parser.add_argument("--critic-lr", type=float, default=3.0e-4)
    parser.add_argument("--steering-latent-std", type=float, default=0.05)
    parser.add_argument("--speed-physical-std", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.995)
    parser.add_argument("--reward-progress-weight", type=float, default=0.010)
    parser.add_argument("--reward-relative-weight", type=float, default=0.020)
    parser.add_argument("--reward-collision", type=float, default=-2.0)
    parser.add_argument("--evaluation-transition-budgets", type=parse_int_list, default=None)
    parser.add_argument(
        "--evaluation-updates",
        type=parse_evaluation_updates,
        default=None,
        help="Comma-separated positive update indices; update 0 BC evaluation remains implicit",
    )
    parser.add_argument("--bc-outcomes", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        choices=("none", "zero_lr", "nonzero", "v1_1_zero_lr", "v1_1_nonzero", "v1_2_zero_lr", "v1_2_nonzero"),
        default="none",
    )
    parser.add_argument(
        "--dry-run-resolved-config",
        action="store_true",
        help="Print the resolved config and exit before creating environments or run artifacts",
    )
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def append_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False, default=_json_default) + "\n")


def resolved_configuration(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "config_json", None) is not None:
        with Path(args.config_json).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        config = resolve_v1_2_config(raw)
        config.update(
            experiment_profile="ppo_v1_2",
            evaluation_workers=int(getattr(args, "evaluation_workers", 8)),
            smoke=str(getattr(args, "smoke", "none")),
            bc_checkpoint=str(DEFAULT_BC_CHECKPOINT.relative_to(PROJECT_ROOT)),
            master_seed=int(config["seed"]),
        )
        return config
    if getattr(args, "experiment_profile", "ppo_v1") == "ppo_v1_2":
        transitions_per_update = int(args.n_envs) * int(args.n_steps)
        budgets = (
            list(args.evaluation_transition_budgets)
            if args.evaluation_transition_budgets is not None
            else [2 * transitions_per_update, 4 * transitions_per_update, int(args.updates) * transitions_per_update]
        )
        overrides = {
            "updates": int(args.updates),
            "n_envs": int(args.n_envs),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "device": str(args.device),
            "seed": int(args.master_seed),
            "critic_profile": str(args.critic_profile),
            "hard_pool_id": str(args.hard_pool_id),
            "hard_sampling_probability": float(args.hard_sampling_probability),
            "hard_sampling_mode": str(args.hard_sampling_mode),
            "target_kl": args.target_kl,
            "gru_lr": float(args.gru_lr),
            "head_lr": float(args.head_lr),
            "critic_lr": float(args.critic_lr),
            "steering_latent_std": float(args.steering_latent_std),
            "speed_physical_std": float(args.speed_physical_std),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "reward_progress_weight": float(args.reward_progress_weight),
            "reward_relative_weight": float(args.reward_relative_weight),
            "reward_collision": float(args.reward_collision),
            "evaluation_transition_budgets": budgets,
        }
        if args.smoke in {"v1_2_zero_lr", "v1_2_nonzero"}:
            updates = 1 if args.smoke == "v1_2_zero_lr" else 2
            overrides.update(
                updates=updates,
                n_envs=1,
                n_steps=100,
                batch_size=100,
                evaluation_transition_budgets=[100 * update for update in range(1, updates + 1)],
            )
            if args.smoke == "v1_2_zero_lr":
                overrides.update(gru_lr=0.0, head_lr=0.0, critic_lr=0.0)
        config = resolve_v1_2_config(overrides)
        config.update(
            experiment_profile="ppo_v1_2",
            evaluation_workers=int(args.evaluation_workers),
            smoke=str(args.smoke),
            bc_checkpoint=str(DEFAULT_BC_CHECKPOINT.relative_to(PROJECT_ROOT)),
            master_seed=int(config["seed"]),
        )
        return config
    config = deepcopy(DEFAULT_CONFIG)
    config.update(
        updates=args.updates,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=args.device,
        master_seed=args.master_seed,
        lr_scale=args.lr_scale,
        evaluation_workers=args.evaluation_workers,
        collision_sampling_probability=args.collision_sampling_probability,
        smoke=args.smoke,
    )
    if args.smoke == "zero_lr":
        config.update(updates=1, n_envs=1, n_steps=100, batch_size=100, lr_scale=0.0)
    elif args.smoke == "nonzero":
        config.update(updates=2, n_envs=1, n_steps=100, batch_size=100, lr_scale=1.0)
    elif args.smoke == "v1_1_zero_lr":
        config.update(
            updates=1,
            n_envs=16,
            n_steps=1600,
            batch_size=1600,
            lr_scale=0.0,
            collision_sampling_probability=0.50,
        )
    elif args.smoke == "v1_1_nonzero":
        config.update(
            updates=2,
            n_envs=16,
            n_steps=1600,
            batch_size=1600,
            lr_scale=1.0,
            collision_sampling_probability=0.50,
        )
    if config["updates"] <= 0 or config["n_envs"] <= 0 or config["n_steps"] <= 0:
        raise ValueError("updates, n_envs, and n_steps must be positive")
    transitions_per_update = config["n_envs"] * config["n_steps"]
    if config["batch_size"] <= 0 or transitions_per_update % config["batch_size"] != 0:
        raise ValueError("batch_size must evenly divide n_envs * n_steps")
    if args.smoke == "none" and config["n_envs"] not in (8, 16):
        raise ValueError("Formal PPO V1 runs use n_envs=16, or an explicit n_envs=8 fallback")
    if not np.isfinite(config["lr_scale"]) or config["lr_scale"] < 0.0:
        raise ValueError("lr_scale must be finite and non-negative")
    if not np.isfinite(config["collision_sampling_probability"]) or not (
        0.0 <= config["collision_sampling_probability"] <= 1.0
    ):
        raise ValueError("collision_sampling_probability must be finite and in [0, 1]")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PPO V1 resolved device is cuda, but CUDA is unavailable")
    if args.evaluation_updates is None:
        evaluation_updates = list(range(5, config["updates"] + 1, 5))
    else:
        evaluation_updates = list(args.evaluation_updates)
    if (
        evaluation_updates != sorted(set(evaluation_updates))
        or any(update <= 0 or update > config["updates"] for update in evaluation_updates)
    ):
        raise ValueError("evaluation_updates must be unique, increasing, positive, and no greater than updates")
    config["transitions_per_update"] = transitions_per_update
    config["minibatches_per_update"] = transitions_per_update // config["batch_size"]
    config["optimizer_steps_per_update"] = config["minibatches_per_update"] * config["n_epochs"]
    config["total_transitions"] = transitions_per_update * config["updates"]
    config["total_optimizer_steps"] = config["optimizer_steps_per_update"] * config["updates"]
    config["evaluation_updates"] = evaluation_updates
    config["gamma_times_gae_lambda"] = config["gamma"] * config["gae_lambda"]
    config["bc_checkpoint"] = str(DEFAULT_BC_CHECKPOINT.relative_to(PROJECT_ROOT))

    is_v1_1 = (
        config["smoke"] == "none"
        and config["n_envs"] == 16
        and config["n_steps"] == 1600
        and config["batch_size"] == 1600
        and config["updates"] == 20
        and config["lr_scale"] == 1.0
        and config["collision_sampling_probability"] == 0.50
    )
    is_v1 = (
        config["smoke"] == "none"
        and config["n_envs"] == DEFAULT_CONFIG["n_envs"]
        and config["n_steps"] == DEFAULT_CONFIG["n_steps"]
        and config["batch_size"] == DEFAULT_CONFIG["batch_size"]
        and config["updates"] == DEFAULT_CONFIG["updates"]
        and config["lr_scale"] == DEFAULT_CONFIG["lr_scale"]
        and config["collision_sampling_probability"] == DEFAULT_CONFIG["collision_sampling_probability"]
        and config["evaluation_updates"] == [5, 10, 15, 20]
    )
    is_v1_1_smoke = config["smoke"] in {"v1_1_zero_lr", "v1_1_nonzero"}
    if is_v1_1:
        if tuple(config["evaluation_updates"]) != V1_1_EVALUATION_UPDATES:
            raise ValueError(
                "PPO V1.1 requires --evaluation-updates 2,3,5,10,15,20"
            )
        if config["minibatches_per_update"] != 16:
            raise AssertionError("PPO V1.1 requires exactly 16 minibatches per update")
        if config["total_optimizer_steps"] != 320:
            raise AssertionError("PPO V1.1 requires exactly 320 total optimizer steps")
        if config["transitions_per_update"] != 25_600 or config["total_transitions"] != 512_000:
            raise AssertionError("PPO V1.1 transition geometry is inconsistent")
        config["configuration_profile"] = "ppo_v1_1"
    elif is_v1_1_smoke:
        if config["transitions_per_update"] != 25_600 or config["minibatches_per_update"] != 16:
            raise AssertionError("PPO V1.1 smoke rollout geometry is inconsistent")
        expected_total_steps = 16 if config["smoke"] == "v1_1_zero_lr" else 32
        if config["total_optimizer_steps"] != expected_total_steps:
            raise AssertionError("PPO V1.1 smoke optimizer-step geometry is inconsistent")
        config["configuration_profile"] = config["smoke"]
    elif is_v1:
        config["configuration_profile"] = "ppo_v1"
    else:
        config["configuration_profile"] = "custom"
    return config


def _evaluation_worker_init(model_path: str) -> None:
    global EVALUATION_MODEL
    torch.set_num_threads(1)
    model = End2Race(mask_prob=0.0, hidden_scale=4).to(EVALUATION_DEVICE)
    state = torch.load(model_path, map_location=EVALUATION_DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    EVALUATION_MODEL = model


def _evaluate_scenario(task: tuple[ScenarioSpec, str]) -> dict[str, Any]:
    scenario, model_path = task
    try:
        if EVALUATION_MODEL is None:
            _evaluation_worker_init(model_path)
        from eval_multiagent import evaluate_segment

        result = evaluate_segment(
            EVALUATION_MODEL,
            EVALUATION_DEVICE,
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
            model_path,
            None,
            "ego",
            scenario.scenario_id,
        )
        metrics = result["episode_metrics"]
        return {
            **scenario.to_dict(),
            "outcome": metrics["outcome"],
            "ego_collision": bool(metrics["ego_collision_occurred"]),
            "opponent_collision": bool(metrics["opp_collision_occurred"]),
            "opponent_only_collision": bool(metrics["opponent_only_collision"]),
            "final_relative_position_m": float(metrics["final_relative_position_m"]),
        }
    except Exception as error:
        return {
            **scenario.to_dict(),
            "outcome": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }


def evaluate_actor_pool(
    model_path: Path,
    scenarios: Sequence[ScenarioSpec],
    *,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if workers <= 0:
        raise ValueError("evaluation workers must be positive")
    tasks = [(scenario, str(model_path.resolve())) for scenario in scenarios]
    if workers == 1:
        _evaluation_worker_init(str(model_path.resolve()))
        rows = [_evaluate_scenario(task) for task in tasks]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_evaluation_worker_init,
            initargs=(str(model_path.resolve()),),
        ) as executor:
            rows = list(executor.map(_evaluate_scenario, tasks, chunksize=1))
    outcomes = Counter(row["outcome"] for row in rows)
    summary = {
        "ego_collision": int(outcomes["ego_collision"]),
        "follow": int(outcomes["follow"]),
        "overtake": int(outcomes["overtake"]),
        "opponent_only_collision": int(sum(bool(row.get("opponent_only_collision")) for row in rows)),
        "error": int(outcomes["error"]),
        "total": len(rows),
    }
    return rows, summary


def validate_evaluation(summary: dict[str, int], expected_rows: int = 600) -> None:
    classified = summary["ego_collision"] + summary["follow"] + summary["overtake"]
    if summary["total"] != expected_rows or classified != expected_rows or summary["error"] != 0:
        raise RuntimeError(f"Invalid PPO V1 evaluation summary: {summary}")


def make_training_env(rank: int, sampler: FixedMixtureScenarioSampler, config: dict[str, Any]):
    def factory() -> End2RaceGymnasiumEnv:
        import gym
        from f110_gym.envs.base_classes import Integrator

        core = gym.make(
            "f110-v0",
            map=str(PROJECT_ROOT / "f1tenth_racetracks" / "Austin" / "Austin_map"),
            map_ext=".png",
            num_agents=2,
            timestep=0.01,
            integrator=Integrator.RK4,
            seed=int(config["master_seed"] + rank),
        )
        return End2RaceGymnasiumEnv(
            core,
            sim_duration=8.0,
            reset_provider=sampler,
            ego_index=0,
            opponent_controller=LatticePlannerOpponentController(),
            transition_reward=PPOV1TransitionReward(
                ProgressProjector.from_csv(),
                progress_weight=float(config.get("reward_progress_weight", 0.01)),
                relative_weight=float(config.get("reward_relative_weight", 0.02)),
                collision_penalty=float(config.get("reward_collision", -2.0)),
            ),
            privileged_critic=config.get("critic_profile") == "C3_PRIVILEGED_PHYSICAL",
        )

    return factory


def build_model(vector_env: DummyVecEnv, config: dict[str, Any]) -> End2RaceRecurrentPPO:
    model = End2RaceRecurrentPPO(
        End2RaceGRUPolicy,
        vector_env,
        learning_rate=1.0,
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        clip_range_vf=config["clip_range_vf"],
        normalize_advantage=config["normalize_advantage"],
        vf_coef=config["vf_coef"],
        ent_coef=config["ent_coef"],
        max_grad_norm=config["max_grad_norm"],
        target_kl=config["target_kl"],
        seed=config["master_seed"],
        device=config["device"],
        policy_kwargs={
            "checkpoint_path": DEFAULT_BC_CHECKPOINT,
            "hidden_scale": 4,
            "critic_hidden_size": 64,
            "optimizer_profile": "ppo_v1",
            "critic_profile": config.get("critic_profile", "C0_RAW_SINGLE_FRAME"),
            "gru_lr": float(config.get("gru_lr", 1.0e-6)),
            "head_lr": float(config.get("head_lr", 1.0e-5)),
            "critic_lr": float(config.get("critic_lr", 3.0e-4)),
            "steering_latent_std": config.get("steering_latent_std"),
            "speed_physical_std": config.get("speed_physical_std"),
        },
        verbose=1,
    )
    model.lr_scale = float(config.get("lr_scale", 1.0))
    return model


def clone_parameters(model: End2RaceRecurrentPPO) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.policy.named_parameters()
    }


def parameter_deltas(
    before: dict[str, torch.Tensor],
    model: End2RaceRecurrentPPO,
) -> dict[str, dict[str, float]]:
    groups = {
        "gru": "lstm_actor.gru.",
        "head": "end2race_actor.output_layer.",
        "critic": "value_net.",
        "perception": "end2race_actor.k",
        "speed_mlp": "end2race_actor.speed_mlp.",
        "dummy_embedding": "end2race_actor.dummy_embedding",
        "log_std": "log_std",
    }
    current = dict(model.policy.named_parameters())
    results: dict[str, dict[str, float]] = {}
    for group_name, prefix in groups.items():
        names = [name for name in before if name == prefix or name.startswith(prefix)]
        deltas = [float((current[name].detach().cpu() - before[name]).abs().max()) for name in names]
        results[group_name] = {
            "parameter_tensors": len(names),
            "max_abs_delta": max(deltas, default=0.0),
        }
    return results


def all_rollout_fields_finite(model: End2RaceRecurrentPPO) -> bool:
    fields = ("observations", "actions", "rewards", "advantages", "returns", "log_probs", "values")
    for name in fields:
        value = getattr(model.rollout_buffer, name)
        if isinstance(value, dict):
            if not all(np.isfinite(np.asarray(part)).all() for part in value.values()):
                return False
        elif not np.isfinite(np.asarray(value)).all():
            return False
    return True


def gradient_norm(model: End2RaceRecurrentPPO) -> float:
    squared = 0.0
    for parameter in model.policy.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().double().square().sum().cpu())
    return math.sqrt(squared)


def optimizer_group_lrs(model: End2RaceRecurrentPPO) -> dict[str, float]:
    return {str(group["name"]): float(group["lr"]) for group in model.policy.optimizer.param_groups}


def optimizer_step_evidence(model: End2RaceRecurrentPPO) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    all_steps: list[int] = []
    for group in model.policy.optimizer.param_groups:
        steps: list[int] = []
        for parameter in group["params"]:
            state = model.policy.optimizer.state.get(parameter, {})
            if "step" in state:
                step = state["step"]
                steps.append(int(step.detach().cpu().item()) if isinstance(step, torch.Tensor) else int(step))
        all_steps.extend(steps)
        groups[str(group["name"])] = {
            "parameter_count": len(group["params"]),
            "parameter_state_count": len(steps),
            "unique_steps": sorted(set(steps)),
        }
    every_parameter_has_state = all(
        value["parameter_state_count"] == value["parameter_count"]
        for value in groups.values()
    )
    unique_steps = sorted(set(all_steps))
    return {
        "groups": groups,
        "every_optimizer_parameter_has_state": every_parameter_has_state,
        "all_parameter_steps_equal": every_parameter_has_state and len(unique_steps) == 1,
        "observed_cumulative_optimizer_steps": unique_steps[0] if len(unique_steps) == 1 else None,
    }


@contextmanager
def audited_gradient_clipping(model: End2RaceRecurrentPPO):
    """Observe pre-clip norms while executing the unchanged stock clip call."""

    original = torch.nn.utils.clip_grad_norm_
    group_by_parameter = {
        id(parameter): str(group["name"])
        for group in model.policy.optimizer.param_groups
        for parameter in group["params"]
    }
    samples: list[dict[str, Any]] = []

    def wrapper(parameters, max_norm, *args, **kwargs):
        parameter_list = list(parameters)
        squared: dict[str, float] = {name: 0.0 for name in ("gru", "head", "critic")}
        for parameter in parameter_list:
            if parameter.grad is not None:
                name = group_by_parameter.get(id(parameter))
                if name in squared:
                    squared[name] += float(parameter.grad.detach().double().square().sum().cpu())
        group_norms = {name: math.sqrt(value) for name, value in squared.items()}
        combined = math.sqrt(sum(value * value for value in group_norms.values()))
        samples.append(
            {
                "gru_preclip_norm": group_norms["gru"],
                "head_preclip_norm": group_norms["head"],
                "critic_preclip_norm": group_norms["critic"],
                "combined_preclip_norm": combined,
                "theoretical_clip_multiplier": min(1.0, float(max_norm) / (combined + 1e-6)),
            }
        )
        return original(parameter_list, max_norm, *args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = wrapper
    try:
        yield samples
    finally:
        torch.nn.utils.clip_grad_norm_ = original


def post_update_minibatch_kl(model: End2RaceRecurrentPPO) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    rows: list[float] = []
    model.policy.set_training_mode(False)
    try:
        for rollout_data in model.rollout_buffer.get(model.batch_size):
            mask = rollout_data.mask > 1e-8
            with torch.no_grad():
                _values, log_prob, _entropy = model.policy.evaluate_actions(
                    rollout_data.observations,
                    rollout_data.actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                log_ratio = log_prob - rollout_data.old_log_prob
                value = torch.mean(((torch.exp(log_ratio) - 1.0) - log_ratio)[mask])
            rows.append(float(value.detach().cpu()))
    finally:
        np.random.set_state(numpy_state)
    return {
        "per_minibatch": rows,
        "mean": float(np.mean(rows)),
        "max": float(np.max(rows)),
        "all_finite": bool(np.isfinite(rows).all()),
    }


def critic_rollout_telemetry(model: End2RaceRecurrentPPO) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    model.policy.set_training_mode(False)
    try:
        rollout_data = next(iter(model.rollout_buffer.get(model.batch_size)))
        with torch.no_grad():
            _means, _states, actor_features = model.policy._actor_forward(
                rollout_data.observations,
                rollout_data.lstm_states.pi,
                rollout_data.episode_starts,
            )
            predictions = model.policy._critic_values(rollout_data.observations, actor_features).flatten()
            telemetry = model.policy.critic_telemetry(rollout_data.observations, actor_features)
        mask = rollout_data.mask > 1e-8
        prediction_values = predictions[mask].detach().cpu().numpy()
        returns = rollout_data.returns[mask].detach().cpu().numpy()
        advantages = rollout_data.advantages[mask].detach().cpu().numpy()
    finally:
        np.random.set_state(numpy_state)
    return {
        **telemetry,
        "prediction_mean": float(np.mean(prediction_values)),
        "prediction_std": float(np.std(prediction_values)),
        "prediction_min": float(np.min(prediction_values)),
        "prediction_max": float(np.max(prediction_values)),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "advantage_mean": float(np.mean(advantages)),
        "advantage_std": float(np.std(advantages)),
        "advantage_p95": float(np.percentile(advantages, 95)),
        "advantage_p99": float(np.percentile(advantages, 99)),
    }


def train_metrics(model: End2RaceRecurrentPPO, preclip_samples: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    values = model.logger.name_to_value
    keys = (
        "train/loss",
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/explained_variance",
    )
    metrics = {key.removeprefix("train/"): float(values[key]) for key in keys if key in values}
    metrics["gradient_norm"] = gradient_norm(model)
    metrics["optimizer_group_lrs"] = optimizer_group_lrs(model)
    metrics["optimizer_step_evidence"] = optimizer_step_evidence(model)
    metrics["rollout_fields_finite"] = all_rollout_fields_finite(model)
    metrics["post_update_kl"] = post_update_minibatch_kl(model)
    metrics["critic_telemetry"] = critic_rollout_telemetry(model)
    metrics["preclip_gradient_samples"] = list(preclip_samples)
    scalar_values = [value for value in metrics.values() if isinstance(value, float)]
    metrics["all_scalar_metrics_finite"] = bool(np.isfinite(scalar_values).all())
    return metrics


def recurrent_replay_metrics(model: End2RaceRecurrentPPO) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    valid_count = 0
    padded_count = 0
    minibatch_count = 0
    log_prob_errors: list[np.ndarray] = []
    ratios: list[np.ndarray] = []
    model.policy.set_training_mode(False)
    try:
        for rollout_data in model.rollout_buffer.get(model.batch_size):
            mask = rollout_data.mask > 1e-8
            with torch.no_grad():
                _values, log_prob, _entropy = model.policy.evaluate_actions(
                    rollout_data.observations,
                    rollout_data.actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
            error = (log_prob - rollout_data.old_log_prob)[mask]
            ratio = torch.exp(error)
            log_prob_errors.append(error.detach().cpu().numpy())
            ratios.append(ratio.detach().cpu().numpy())
            valid_count += int(mask.sum().item())
            padded_count += int((~mask).sum().item())
            minibatch_count += 1
    finally:
        np.random.set_state(numpy_state)
    errors = np.concatenate(log_prob_errors)
    ratio_values = np.concatenate(ratios)
    return {
        "minibatch_count": minibatch_count,
        "valid_transition_count": valid_count,
        "padded_transition_count": padded_count,
        "max_abs_log_prob_replay_error": float(np.max(np.abs(errors))),
        "max_abs_ratio_deviation": float(np.max(np.abs(ratio_values - 1.0))),
        "ratio_mean": float(np.mean(ratio_values)),
        "ratio_std": float(np.std(ratio_values)),
        "ratio_min": float(np.min(ratio_values)),
        "ratio_max": float(np.max(ratio_values)),
        "all_finite": bool(np.isfinite(errors).all() and np.isfinite(ratio_values).all()),
    }


def export_actor(policy: End2RaceGRUPolicy, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in policy.actor_checkpoint_state_dict().items()}
    if len(state) != 12:
        raise RuntimeError(f"Expected 12 End2Race actor keys, got {len(state)}")
    torch.save(state, destination)
    fresh = End2Race(mask_prob=0.0, hidden_scale=4)
    fresh.load_state_dict(torch.load(destination, map_location="cpu", weights_only=True), strict=True)


def save_checkpoint(
    model: End2RaceRecurrentPPO,
    run_dir: Path,
    update: int,
) -> Path:
    checkpoint_dir = run_dir / "checkpoints" / f"update_{update:04d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    model.save(checkpoint_dir / "model.zip")
    export_actor(model.policy, checkpoint_dir / "actor_only.pth")
    return checkpoint_dir


def verify_smoke(
    model: End2RaceRecurrentPPO,
    before: dict[str, torch.Tensor],
    config: dict[str, Any],
    checkpoint_dir: Path,
    update_metrics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    deltas = parameter_deltas(before, model)
    reloaded = End2RaceRecurrentPPO.load(checkpoint_dir / "model.zip", device=config["device"])
    reload_lrs = optimizer_group_lrs(reloaded)
    expected_lrs = {
        "gru": float(config.get("gru_lr", 1e-6)) * float(config.get("lr_scale", 1.0)),
        "head": float(config.get("head_lr", 1e-5)) * float(config.get("lr_scale", 1.0)),
        "critic": float(config.get("critic_lr", 3e-4)) * float(config.get("lr_scale", 1.0)),
    }
    finite = all(
        bool(metrics["rollout_fields_finite"])
        and bool(metrics["all_scalar_metrics_finite"])
        and np.isfinite(list(metrics["optimizer_group_lrs"].values())).all()
        for metrics in update_metrics
    )
    zero_lr_smoke = config["smoke"] in {"zero_lr", "v1_1_zero_lr", "v1_2_zero_lr"}
    if zero_lr_smoke:
        parameter_gate = all(group["max_abs_delta"] == 0.0 for group in deltas.values())
    else:
        parameter_gate = (
            deltas["gru"]["max_abs_delta"] > 0.0
            and deltas["head"]["max_abs_delta"] > 0.0
            and deltas["critic"]["max_abs_delta"] > 0.0
            and all(
                deltas[name]["max_abs_delta"] == 0.0
                for name in ("perception", "speed_mlp", "dummy_embedding", "log_std")
            )
        )
    lr_gate = all(
        all(abs(metrics["optimizer_group_lrs"][name] - value) <= 1e-15 for name, value in expected_lrs.items())
        for metrics in update_metrics
    ) and all(abs(reload_lrs[name] - value) <= 1e-15 for name, value in expected_lrs.items())
    optimizer_step_gate = all(
        metrics["optimizer_step_evidence"]["all_parameter_steps_equal"]
        and metrics["optimizer_step_evidence"]["observed_cumulative_optimizer_steps"]
        == metrics["update"] * config["optimizer_steps_per_update"]
        for metrics in update_metrics
    )
    replay = recurrent_replay_metrics(model)
    replay_finite_gate = bool(
        replay["all_finite"]
        and replay["valid_transition_count"] == config["transitions_per_update"]
        and replay["minibatch_count"] == config["minibatches_per_update"]
    )
    replay_identity_gate = bool(
        not zero_lr_smoke
        or (
            replay["max_abs_log_prob_replay_error"] <= 1e-6
            and replay["max_abs_ratio_deviation"] <= 1e-6
        )
    )
    result = {
        "smoke": config["smoke"],
        "passed": bool(
            finite
            and parameter_gate
            and lr_gate
            and optimizer_step_gate
            and replay_finite_gate
            and replay_identity_gate
        ),
        "all_required_fields_finite": finite,
        "parameter_delta_gate": parameter_gate,
        "learning_rate_gate": lr_gate,
        "optimizer_step_gate": optimizer_step_gate,
        "recurrent_replay_finite_gate": replay_finite_gate,
        "recurrent_replay_identity_gate": replay_identity_gate,
        "recurrent_replay": replay,
        "parameter_deltas": deltas,
        "expected_group_lrs": expected_lrs,
        "reloaded_group_lrs": reload_lrs,
        "checkpoint_reload_succeeded": True,
        "actor_key_count": len(model.policy.actor_checkpoint_state_dict()),
        "actor_strict_load_succeeded": True,
    }
    if not result["passed"]:
        raise RuntimeError(f"PPO V1 {config['smoke']} smoke failed: {result}")
    return result


def load_or_classify_training_bc(
    args: argparse.Namespace,
    run_dir: Path,
    scenarios: Sequence[ScenarioSpec],
    workers: int,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    if args.bc_outcomes is not None:
        with args.bc_outcomes.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError("--bc-outcomes must contain a JSON list of scenario rows")
    else:
        rows, summary = evaluate_actor_pool(DEFAULT_BC_CHECKPOINT, scenarios, workers=workers)
        atomic_write_json(run_dir / "bc_training_summary.json", summary)
    collision_ids = classify_bc_ego_collisions(rows)
    atomic_write_json(run_dir / "train_bc_outcomes.json", rows)
    return rows, collision_ids


def load_hard_pool_manifest(path: Path, expected_pool_id: str) -> tuple[tuple[ScenarioSpec, ...], tuple[str, ...], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("pool_id") != expected_pool_id:
        raise ValueError(f"Hard pool ID mismatch: expected {expected_pool_id}, got {manifest.get('pool_id')}")
    expected_hash = canonical_hash({key: value for key, value in manifest.items() if key != "manifest_hash"})
    if manifest.get("manifest_hash") != expected_hash:
        raise ValueError(f"Hard pool manifest hash mismatch: {path}")
    scenarios = tuple(scenario_from_dict(row) for row in manifest.get("scenarios", []))
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    ids = tuple(map(str, manifest.get("scenario_ids", [])))
    if not ids or tuple(sorted(ids)) != ids or len(ids) != len(set(ids)):
        raise ValueError(f"Hard pool must be non-empty, unique and canonically sorted: {path}")
    if set(ids) != set(by_id):
        raise ValueError(f"Hard pool scenario rows and IDs differ: {path}")
    return scenarios, ids, manifest


def select_checkpoint(bc: dict[str, int], candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    overtake_floor = math.ceil(0.95 * bc["overtake"])
    eligible = [candidate for candidate in candidates if candidate["metrics"]["overtake"] >= overtake_floor]
    if eligible:
        best = min(
            eligible,
            key=lambda candidate: (
                candidate["metrics"]["ego_collision"],
                -candidate["metrics"]["overtake"],
                candidate["update"],
            ),
        )
        differences = {
            key: int(best["metrics"][key] - bc[key])
            for key in ("ego_collision", "follow", "overtake", "opponent_only_collision")
        }
        status = "improved" if best["metrics"]["ego_collision"] < bc["ego_collision"] else "not_improved"
    else:
        best = None
        differences = None
        status = "no_eligible_checkpoint"
    return {
        "status": status,
        "overtake_floor": overtake_floor,
        "paired_bc": bc,
        "eligible_updates": [candidate["update"] for candidate in eligible],
        "best": best,
        "best_minus_bc": differences,
        "candidates": list(candidates),
        "result_scope": "V1 development/pilot; evaluation pool was used for checkpoint selection",
    }


def paired_change_metrics(
    bc_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> dict[str, int]:
    bc_by_id = {str(row["scenario_id"]): row for row in bc_rows}
    candidate_by_id = {str(row["scenario_id"]): row for row in candidate_rows}
    if len(bc_by_id) != 600 or set(candidate_by_id) != set(bc_by_id):
        raise ValueError("Paired evaluation rows must contain the same 600 unique scenario IDs")
    pairs = [(bc_by_id[scenario_id], candidate_by_id[scenario_id]) for scenario_id in sorted(bc_by_id)]
    return {
        "fixed_collision": sum(
            bc["outcome"] == "ego_collision" and candidate["outcome"] != "ego_collision"
            for bc, candidate in pairs
        ),
        "new_collision": sum(
            bc["outcome"] != "ego_collision" and candidate["outcome"] == "ego_collision"
            for bc, candidate in pairs
        ),
        "gained_overtake": sum(
            bc["outcome"] != "overtake" and candidate["outcome"] == "overtake"
            for bc, candidate in pairs
        ),
        "lost_overtake": sum(
            bc["outcome"] == "overtake" and candidate["outcome"] != "overtake"
            for bc, candidate in pairs
        ),
    }


def run(args: argparse.Namespace) -> Path:
    config = resolved_configuration(args)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (PROJECT_ROOT / args.run_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "training_metrics.jsonl").exists() or (run_dir / "run_summary.json").exists():
        raise FileExistsError(f"Run directory already contains training state: {run_dir}")
    atomic_write_json(run_dir / "resolved_config.json", config)

    random.seed(config["master_seed"])
    np.random.seed(config["master_seed"])
    torch.manual_seed(config["master_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["master_seed"])

    train_pool = training_scenarios()
    eval_pool = evaluation_scenarios()
    atomic_write_json(run_dir / "training_scenarios.json", [scenario.to_dict() for scenario in train_pool])
    atomic_write_json(run_dir / "evaluation_scenarios.json", [scenario.to_dict() for scenario in eval_pool])

    _bc_training_rows, collision_ids = load_or_classify_training_bc(
        args,
        run_dir,
        train_pool,
        workers=config["evaluation_workers"],
    )
    if config.get("experiment_profile") == "ppo_v1_2" and args.hard_pool_manifest is not None:
        hard_scenarios, hard_ids, hard_manifest = load_hard_pool_manifest(
            args.hard_pool_manifest,
            config["hard_pool_id"],
        )
        if config.get("hard_pool_hash") not in (None, hard_manifest["manifest_hash"]):
            raise ValueError("Resolved config hard pool hash differs from the supplied manifest")
    else:
        hard_scenarios = train_pool
        hard_ids = collision_ids
        hard_manifest = None
    sampler = FixedMixtureScenarioSampler(
        train_pool,
        hard_ids,
        collision_probability=float(
            config["hard_sampling_probability"]
            if "hard_sampling_probability" in config
            else config["collision_sampling_probability"]
        ),
        hard_scenarios=hard_scenarios,
        hard_pool_id=str(config.get("hard_pool_id", "H0_CURRENT_DET")),
        hard_sampling_mode=str(config.get("hard_sampling_mode", "with_replacement")),
    )
    vector_env = DummyVecEnv(
        [make_training_env(rank, sampler, config) for rank in range(config["n_envs"])]
    )
    vector_env.seed(config["master_seed"])
    model = build_model(vector_env, config)
    callback = PPOV1MetricsCallback(run_dir, config["n_envs"])
    before = clone_parameters(model)
    update_metrics: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    evaluate_checkpoints = config["smoke"] == "none"
    if evaluate_checkpoints:
        bc_rows, bc_metrics = evaluate_actor_pool(
            DEFAULT_BC_CHECKPOINT,
            eval_pool,
            workers=config["evaluation_workers"],
        )
        bc_metrics.update(
            fixed_collision=0,
            new_collision=0,
            gained_overtake=0,
            lost_overtake=0,
        )
        atomic_write_json(run_dir / "evaluations" / "update_0000_rows.json", bc_rows)
        atomic_write_json(run_dir / "evaluations" / "update_0000_metrics.json", bc_metrics)
        validate_evaluation(bc_metrics)
    else:
        bc_metrics = {}

    final_checkpoint_dir: Path | None = None
    try:
        previous_optimizer_steps = 0
        for update in range(1, config["updates"] + 1):
            with audited_gradient_clipping(model) as preclip_samples:
                model.learn(
                    total_timesteps=config["transitions_per_update"],
                    callback=callback,
                    log_interval=None,
                    reset_num_timesteps=(update == 1),
                    progress_bar=False,
                )
            metrics = {
                "update": update,
                "num_timesteps": int(model.num_timesteps),
                **train_metrics(model, preclip_samples),
                "rollout": callback.latest_update_summary,
                "parameter_deltas_from_fresh_start": parameter_deltas(before, model),
            }
            step_evidence = metrics["optimizer_step_evidence"]
            observed = step_evidence["observed_cumulative_optimizer_steps"]
            observed = previous_optimizer_steps if observed is None else int(observed)
            actual_steps = observed - previous_optimizer_steps
            planned_steps = int(config["optimizer_steps_per_update"])
            if not step_evidence["all_parameter_steps_equal"] and observed != 0:
                raise RuntimeError(f"Optimizer parameters disagree on Adam step at update {update}: {step_evidence}")
            if config.get("target_kl") is None and actual_steps != planned_steps:
                raise RuntimeError(f"Unexpected optimizer step count at update {update}: actual={actual_steps}, planned={planned_steps}")
            if actual_steps < 0 or actual_steps > planned_steps:
                raise RuntimeError(f"Illegal optimizer step count at update {update}: actual={actual_steps}, planned={planned_steps}")
            metrics["planned_optimizer_steps"] = planned_steps
            metrics["actual_optimizer_steps"] = actual_steps
            metrics["early_stopped_by_target_kl"] = bool(actual_steps < planned_steps)
            previous_optimizer_steps = observed
            update_metrics.append(metrics)
            append_json(run_dir / "training_metrics.jsonl", metrics)
            if not metrics["rollout_fields_finite"] or not metrics["all_scalar_metrics_finite"]:
                raise RuntimeError(f"Non-finite PPO V1 training values at update {update}")

            checkpoint_due = evaluate_checkpoints and update in config["evaluation_updates"]
            smoke_final = not evaluate_checkpoints and update == config["updates"]
            if checkpoint_due or smoke_final:
                final_checkpoint_dir = save_checkpoint(model, run_dir, update)
                if checkpoint_due:
                    rows, evaluation = evaluate_actor_pool(
                        final_checkpoint_dir / "actor_only.pth",
                        eval_pool,
                        workers=config["evaluation_workers"],
                    )
                    evaluation.update(paired_change_metrics(bc_rows, rows))
                    if config.get("experiment_profile") == "ppo_v1_2":
                        evaluation.update(checkpoint_flags(evaluation))
                    atomic_write_json(run_dir / "evaluations" / f"update_{update:04d}_rows.json", rows)
                    atomic_write_json(final_checkpoint_dir / "metrics.json", evaluation)
                    validate_evaluation(evaluation)
                    candidates.append({"update": update, "metrics": evaluation})
    finally:
        vector_env.close()

    if final_checkpoint_dir is None:
        raise RuntimeError("Training completed without producing a checkpoint")
    if config["smoke"] != "none":
        smoke_result = verify_smoke(model, before, config, final_checkpoint_dir, update_metrics)
        atomic_write_json(run_dir / "smoke_verification.json", smoke_result)
    else:
        selection = select_checkpoint(bc_metrics, candidates)
        atomic_write_json(run_dir / "selection.json", selection)
    atomic_write_json(
        run_dir / "run_summary.json",
        {
            "completed_updates": config["updates"],
            "completed_transitions": int(model.num_timesteps),
            "final_checkpoint": str(final_checkpoint_dir.relative_to(run_dir)),
            "smoke": config["smoke"],
            "hard_pool_id": config.get("hard_pool_id"),
            "hard_pool_hash": hard_manifest.get("manifest_hash") if hard_manifest is not None else None,
            "hard_sampling_mode": config.get("hard_sampling_mode"),
            "hard_pool_visit_counts": dict(sorted(sampler.visit_counts.items())),
        },
    )
    return run_dir


def main() -> int:
    args = parse_arguments()
    if args.dry_run_resolved_config:
        print(json.dumps(resolved_configuration(args), indent=2, sort_keys=True, default=_json_default))
        return 0
    run_dir = run(args)
    print(f"PPO_V1_RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
