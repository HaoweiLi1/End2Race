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
)
from rl.sb3_end2race_policy import DEFAULT_BC_CHECKPOINT, End2RaceGRUPolicy
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End2Race PPO V1 trainer")
    parser.add_argument("--run-root", type=Path, default=Path("runs/ppo_v1"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--updates", type=int, default=DEFAULT_CONFIG["updates"])
    parser.add_argument("--n-envs", type=int, default=DEFAULT_CONFIG["n_envs"])
    parser.add_argument("--n-steps", type=int, default=DEFAULT_CONFIG["n_steps"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--device", choices=("cuda", "cpu"), default=DEFAULT_CONFIG["device"])
    parser.add_argument("--master-seed", type=int, default=DEFAULT_CONFIG["master_seed"])
    parser.add_argument("--lr-scale", type=float, default=DEFAULT_CONFIG["lr_scale"])
    parser.add_argument("--evaluation-workers", type=int, default=8)
    parser.add_argument("--bc-outcomes", type=Path, default=None)
    parser.add_argument("--smoke", choices=("none", "zero_lr", "nonzero"), default="none")
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
        smoke=args.smoke,
    )
    if args.smoke == "zero_lr":
        config.update(updates=1, n_envs=1, n_steps=100, batch_size=100, lr_scale=0.0)
    elif args.smoke == "nonzero":
        config.update(updates=2, n_envs=1, n_steps=100, batch_size=100, lr_scale=1.0)
    if config["updates"] <= 0 or config["n_envs"] <= 0 or config["n_steps"] <= 0:
        raise ValueError("updates, n_envs, and n_steps must be positive")
    transitions_per_update = config["n_envs"] * config["n_steps"]
    if config["batch_size"] <= 0 or transitions_per_update % config["batch_size"] != 0:
        raise ValueError("batch_size must evenly divide n_envs * n_steps")
    if args.smoke == "none" and config["n_envs"] not in (8, 16):
        raise ValueError("Formal PPO V1 runs use n_envs=16, or an explicit n_envs=8 fallback")
    if not np.isfinite(config["lr_scale"]) or config["lr_scale"] < 0.0:
        raise ValueError("lr_scale must be finite and non-negative")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PPO V1 resolved device is cuda, but CUDA is unavailable")
    config["transitions_per_update"] = transitions_per_update
    config["minibatches_per_update"] = transitions_per_update // config["batch_size"]
    config["optimizer_steps_per_update"] = config["minibatches_per_update"] * config["n_epochs"]
    config["total_transitions"] = transitions_per_update * config["updates"]
    config["gamma_times_gae_lambda"] = config["gamma"] * config["gae_lambda"]
    config["bc_checkpoint"] = str(DEFAULT_BC_CHECKPOINT.relative_to(PROJECT_ROOT))
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
            transition_reward=PPOV1TransitionReward(ProgressProjector.from_csv()),
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
        },
        verbose=1,
    )
    model.lr_scale = float(config["lr_scale"])
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
    return all(np.isfinite(np.asarray(getattr(model.rollout_buffer, name))).all() for name in fields)


def gradient_norm(model: End2RaceRecurrentPPO) -> float:
    squared = 0.0
    for parameter in model.policy.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().double().square().sum().cpu())
    return math.sqrt(squared)


def optimizer_group_lrs(model: End2RaceRecurrentPPO) -> dict[str, float]:
    return {str(group["name"]): float(group["lr"]) for group in model.policy.optimizer.param_groups}


def train_metrics(model: End2RaceRecurrentPPO) -> dict[str, Any]:
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
    metrics["rollout_fields_finite"] = all_rollout_fields_finite(model)
    scalar_values = [value for value in metrics.values() if isinstance(value, float)]
    metrics["all_scalar_metrics_finite"] = bool(np.isfinite(scalar_values).all())
    return metrics


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
        "gru": 1e-6 * config["lr_scale"],
        "head": 1e-5 * config["lr_scale"],
        "critic": 3e-4 * config["lr_scale"],
    }
    finite = all(
        bool(metrics["rollout_fields_finite"])
        and bool(metrics["all_scalar_metrics_finite"])
        and np.isfinite(list(metrics["optimizer_group_lrs"].values())).all()
        for metrics in update_metrics
    )
    if config["smoke"] == "zero_lr":
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
    result = {
        "smoke": config["smoke"],
        "passed": bool(finite and parameter_gate and lr_gate),
        "all_required_fields_finite": finite,
        "parameter_delta_gate": parameter_gate,
        "learning_rate_gate": lr_gate,
        "parameter_deltas": deltas,
        "expected_group_lrs": expected_lrs,
        "reloaded_group_lrs": reload_lrs,
        "checkpoint_reload_succeeded": True,
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


def run(args: argparse.Namespace) -> Path:
    config = resolved_configuration(args)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (PROJECT_ROOT / args.run_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
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
    sampler = FixedMixtureScenarioSampler(
        train_pool,
        collision_ids,
        collision_probability=config["collision_sampling_probability"],
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
        atomic_write_json(run_dir / "evaluations" / "update_0000_rows.json", bc_rows)
        atomic_write_json(run_dir / "evaluations" / "update_0000_metrics.json", bc_metrics)
        validate_evaluation(bc_metrics)
    else:
        bc_metrics = {}

    final_checkpoint_dir: Path | None = None
    try:
        for update in range(1, config["updates"] + 1):
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
                **train_metrics(model),
                "rollout": callback.latest_update_summary,
            }
            update_metrics.append(metrics)
            append_json(run_dir / "training_metrics.jsonl", metrics)
            if not metrics["rollout_fields_finite"] or not metrics["all_scalar_metrics_finite"]:
                raise RuntimeError(f"Non-finite PPO V1 training values at update {update}")

            checkpoint_due = evaluate_checkpoints and update % 5 == 0
            smoke_final = not evaluate_checkpoints and update == config["updates"]
            if checkpoint_due or smoke_final:
                final_checkpoint_dir = save_checkpoint(model, run_dir, update)
                if checkpoint_due:
                    rows, evaluation = evaluate_actor_pool(
                        final_checkpoint_dir / "actor_only.pth",
                        eval_pool,
                        workers=config["evaluation_workers"],
                    )
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
        },
    )
    return run_dir


def main() -> int:
    args = parse_arguments()
    run_dir = run(args)
    print(f"PPO_V1_RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
