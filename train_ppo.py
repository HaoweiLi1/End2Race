#!/usr/bin/env python3
"""Train the fixed End2Race PPO pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from ppo.algorithm import (
    MAX_GRAD_NORM,
    VALUE_LOSS_COEFFICIENT,
    WARMUP_MAX_EPOCHS,
    WARMUP_PATIENCE,
    WARMUP_TRAIN_FRACTION,
    End2RaceRecurrentPPO,
)
from ppo.collision_classification import resolve_collision_scenarios
from ppo.policy import (
    CRITIC_VARIANTS,
    P20_CRITIC_VARIANTS,
    SPEED_PHYSICAL_STD,
    STEERING_LATENT_STD,
    End2RaceGRUPolicy,
)
from ppo.privileged import (
    PRIVILEGED_FEATURE_HIGHS,
    PRIVILEGED_FEATURE_LOWS,
    PRIVILEGED_FEATURE_NAMES,
    PRIVILEGED_FEATURE_SIZE,
)
from ppo.scenarios import expanded_scenarios, ordinary_scenarios
from ppo.training_records import TrainingRecorder
from ppo.vec_env import CentralScheduleSubprocVecEnv


PROJECT_ROOT = Path(__file__).resolve().parent
POST_TRAINED_ROOT = PROJECT_ROOT / "post-trained"
PPO_CONFIG_PATH = PROJECT_ROOT / "ppo" / "ppo_config.yaml"
with PPO_CONFIG_PATH.open("r", encoding="utf-8") as file:
    PPO_CONFIG = yaml.safe_load(file)
START_METHOD = str(PPO_CONFIG["start_method"])


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train End2Race PPO")

    # Model paths
    parser.add_argument("--pretrained_model_path", type=str, default="pretrained/end2race.pth")
    parser.add_argument("--output_dir", type=str, default="post-trained/ppo_run")

    # Model configuration
    parser.add_argument("--hidden_scale", type=int, default=4)
    parser.add_argument("--critic", choices=CRITIC_VARIANTS, default="mlp")

    # Environment configuration
    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--env_workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--collision_cache_dir", type=str, default="post-trained/collision-cache/default")
    parser.add_argument("--reclassify_collisions", action="store_true")

    # Rollout configuration
    parser.add_argument("--n_steps", type=int, default=6400)
    parser.add_argument("--batch_size", type=int, default=12800)
    parser.add_argument("--num_updates", type=int, default=25)

    # Training configuration
    parser.add_argument("--actor_epochs", type=int, default=3)
    parser.add_argument("--critic_epochs", type=int, default=10)
    parser.add_argument("--gru_learning_rate", type=float, default=1.0e-6)
    parser.add_argument("--head_learning_rate", type=float, default=1.0e-5)
    parser.add_argument("--critic_learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--steering_latent_std", type=float, default=STEERING_LATENT_STD)
    parser.add_argument("--speed_physical_std", type=float, default=SPEED_PHYSICAL_STD)

    # PPO configuration
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.995)
    parser.add_argument("--clip_range", type=float, default=0.10)
    return parser.parse_args()


def configure_training_numerics() -> None:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False


def validate_arguments(args) -> None:
    pretrained_path = Path(args.pretrained_model_path).expanduser().resolve()
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"Pretrained model does not exist: {pretrained_path}")
    if not args.output_dir.strip():
        raise ValueError("output_dir must not be empty")
    if not args.map_name.strip():
        raise ValueError("map_name must not be empty")
    if not args.collision_cache_dir.strip():
        raise ValueError("collision_cache_dir must not be empty")
    output_dir = Path(args.output_dir).expanduser().resolve()
    collision_cache_dir = Path(args.collision_cache_dir).expanduser().resolve()
    if output_dir == POST_TRAINED_ROOT or POST_TRAINED_ROOT not in output_dir.parents:
        raise ValueError(f"output_dir must be inside the project post-trained directory: {POST_TRAINED_ROOT}")
    if output_dir == collision_cache_dir:
        raise ValueError("output_dir and collision_cache_dir must be different directories")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"PPO output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(f"PPO output directory must be empty: {output_dir}")
    if args.env_workers <= 0 or args.n_envs < args.env_workers or args.n_envs % 2 != 0:
        raise ValueError("n_envs must be even and at least env_workers, and env_workers must be positive")
    for name in ("hidden_scale", "n_steps", "batch_size", "num_updates", "actor_epochs", "critic_epochs"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.n_envs * args.n_steps % args.batch_size != 0:
        raise ValueError("n_envs * n_steps must be divisible by batch_size")
    if args.batch_size % (2 * args.n_steps) != 0:
        raise ValueError("batch_size must be divisible by 2 * n_steps so each env-major recurrent minibatch has equal collision and ordinary transitions")
    for name in (
        "gru_learning_rate",
        "head_learning_rate",
        "critic_learning_rate",
        "steering_latent_std",
        "speed_physical_std",
        "gamma",
        "gae_lambda",
        "clip_range",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.gamma > 1.0 or args.gae_lambda > 1.0:
        raise ValueError("gamma and gae_lambda must be at most 1")


def build_model(vector_env, args, device, recorder: TrainingRecorder) -> End2RaceRecurrentPPO:
    return End2RaceRecurrentPPO(
        End2RaceGRUPolicy,
        vector_env,
        actor_epochs=args.actor_epochs,
        critic_epochs=args.critic_epochs,
        recorder=recorder,
        learning_rate=1.0,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=VALUE_LOSS_COEFFICIENT,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=None,
        seed=args.seed,
        device=device,
        policy_kwargs={
            "checkpoint_path": args.pretrained_model_path,
            "hidden_scale": args.hidden_scale,
            "critic_variant": args.critic,
            "gru_learning_rate": args.gru_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "critic_learning_rate": args.critic_learning_rate,
            "steering_latent_std": args.steering_latent_std,
            "speed_physical_std": args.speed_physical_std,
        },
        verbose=1,
    )


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    configure_training_numerics()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(
        f"PPO training configuration: output_dir={Path(args.output_dir).expanduser().resolve()}, pretrained_model_path={Path(args.pretrained_model_path).expanduser().resolve()}, "
        f"map={args.map_name}, critic={args.critic}, n_envs={args.n_envs}, env_workers={args.env_workers}, n_steps={args.n_steps}, "
        f"batch_size={args.batch_size}, num_updates={args.num_updates}, seed={args.seed}",
        flush=True,
    )
    print("[1/5] Building collision candidates", flush=True)
    candidates = expanded_scenarios(args.map_name)
    candidate_count = len(candidates)
    print("[2/5] Loading or classifying collision pool", flush=True)
    collision_scenarios, cache_hit, reclassified = resolve_collision_scenarios(args, candidates, START_METHOD)
    print("[3/5] Building ordinary scenarios", flush=True)
    ordinary_scenario_set = ordinary_scenarios(args.map_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    recorder = TrainingRecorder(args.output_dir, args.hidden_scale)
    recorder.write_scenario_pools(
        collision_scenarios,
        ordinary_scenario_set,
        {
            "cache_dir": str(Path(args.collision_cache_dir).expanduser().resolve()),
            "cache_hit": cache_hit,
            "reclassified": reclassified,
            "candidate_count": candidate_count,
            "collision_count": len(collision_scenarios),
        },
    )
    print("[4/5] Creating vector environments", flush=True)
    vector_env = CentralScheduleSubprocVecEnv(
        args.n_envs,
        args.env_workers,
        START_METHOD,
        args.seed,
        args.map_name,
        collision_scenarios,
        ordinary_scenario_set,
        privileged=args.critic in P20_CRITIC_VARIANTS,
        reward_gamma=args.gamma,
    )
    try:
        privileged_normalization = (
            vector_env.env_method("privileged_normalization_metadata", indices=[0])[0]
            if args.critic in P20_CRITIC_VARIANTS
            else {}
        )
        recorder.write_run_config(
            args,
            PPO_CONFIG,
            {
                "WARMUP_MAX_EPOCHS": WARMUP_MAX_EPOCHS,
                "WARMUP_PATIENCE": WARMUP_PATIENCE,
                "WARMUP_TRAIN_FRACTION": WARMUP_TRAIN_FRACTION,
                "VALUE_LOSS_COEFFICIENT": VALUE_LOSS_COEFFICIENT,
                "MAX_GRAD_NORM": MAX_GRAD_NORM,
                "STEERING_LATENT_STD": args.steering_latent_std,
                "SPEED_PHYSICAL_STD": args.speed_physical_std,
                "PRIVILEGED_FEATURE_SIZE": PRIVILEGED_FEATURE_SIZE,
                "PRIVILEGED_FEATURE_NAMES": list(PRIVILEGED_FEATURE_NAMES),
                "PRIVILEGED_FEATURE_LOWS": list(PRIVILEGED_FEATURE_LOWS),
                "PRIVILEGED_FEATURE_HIGHS": list(PRIVILEGED_FEATURE_HIGHS),
                "PRIVILEGED_NORMALIZATION": privileged_normalization,
            },
        )
        print("[5/5] Building PPO model", flush=True)
        model = build_model(vector_env, args, device, recorder)
        total_rollouts = args.num_updates + 1
        model.learn(total_timesteps=args.n_envs * args.n_steps * total_rollouts, log_interval=1, progress_bar=False)
        final_actor_path = recorder.save_final_actor(model.policy.actor_checkpoint_state_dict())
        print(f"PPO final actor saved: {final_actor_path}", flush=True)
    finally:
        vector_env.close()
    print("Training completed successfully", flush=True)


if __name__ == "__main__":
    main()
