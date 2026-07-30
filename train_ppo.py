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
from ppo.hard_neighbors import resolve_training_collision_scenarios
from ppo.exploration import (
    BASELINE_EXPLORATION_MODE,
    CORRIDOR_TEMPORAL_EXPLORATION_MODE,
    SPEED_EXPLORATION_MODES,
    FrontCorridorGateConfig,
    exploration_metadata,
)
from ppo.fixed_collision_pool import load_fixed_collision_pool
from ppo.policy import (
    CRITIC_VARIANTS,
    P20_CRITIC_VARIANTS,
    SPEED_PHYSICAL_STD,
    STEERING_LATENT_STD,
    End2RaceGRUPolicy,
)
from ppo.postpass import fixed_postpass_config_metadata
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ordinary_startpoint_count",
        type=int,
        choices=(50, 150),
        default=int(PPO_CONFIG["ordinary_startpoint_count"]),
        help=(
            "Ordinary-panel size: 50 preserves the baseline; 150 appends "
            "100 evaluation-separated startpoints without replacing it"
        ),
    )
    parser.add_argument(
        "--collision_cache_dir",
        type=str,
        default=(
            "post-trained/collision-cache/"
            "pretrained_end2race_austin_collision_pool_479"
        ),
    )
    parser.add_argument("--reclassify_collisions", action="store_true")
    parser.add_argument(
        "--fixed_collision_pool_file",
        type=str,
        default=None,
        help=(
            "Use one validated fixed collision-role scenario pool instead of "
            "the classified collision cache"
        ),
    )
    parser.add_argument(
        "--risk_longitudinal_clearance_m",
        type=float,
        default=float(PPO_CONFIG["risk_longitudinal_clearance_m"]),
        help=(
            "Longitudinal vehicle-clearance scale for the existing bounded "
            "risk potential"
        ),
    )
    parser.add_argument(
        "--allow_collision_cache_actor_mismatch",
        action="store_true",
        help=(
            "Reuse an existing base collision cache when only its actor path "
            "differs; all other cache identity fields remain strict"
        ),
    )
    parser.add_argument(
        "--postpass_penalty",
        action="store_true",
        help=(
            "Enable the fixed gated post-pass penalty treatment; omitted for "
            "the unchanged baseline reward"
        ),
    )
    parser.add_argument(
        "--postpass_proximity_power",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "Post-pass proximity magnitude exponent: 2 preserves the "
            "original treatment and 1 is the linear-proximity arm"
        ),
    )
    parser.add_argument(
        "--hard_neighbors",
        action="store_true",
        help="Use a fixed boundary-aware collision cache instead of the baseline collision pool",
    )
    parser.add_argument(
        "--hard_neighbor_cache_dir",
        type=str,
        default=(
            "post-trained/collision-cache/"
            "pretrained_end2race_austin_boundary_aware_collision_pool_805"
        ),
        help="Independent cache built and consumed only when --hard_neighbors is enabled",
    )
    parser.add_argument(
        "--hard_neighbor_fraction",
        type=float,
        default=None,
        help=(
            "Optional with --hard_neighbors; fraction of collision episode resets "
            "drawn from the boundary-aware pool. If omitted, preserve the legacy "
            "uniform merged-pool schedule"
        ),
    )

    # Rollout configuration
    parser.add_argument("--n_steps", type=int, default=6400)
    parser.add_argument("--batch_size", type=int, default=12800)
    parser.add_argument("--num_updates", type=int, default=20)

    # Training configuration
    parser.add_argument("--actor_epochs", type=int, default=2)
    parser.add_argument("--critic_epochs", type=int, default=5)
    parser.add_argument("--gru_learning_rate", type=float, default=3.0e-6)
    parser.add_argument("--head_learning_rate", type=float, default=3.0e-5)
    parser.add_argument("--critic_learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--steering_latent_std", type=float, default=0.03)
    parser.add_argument("--speed_physical_std", type=float, default=0.15)
    parser.add_argument(
        "--speed_physical_std_final",
        type=float,
        default=None,
        help=(
            "Optional final speed exploration std for a linear rollout schedule"
        ),
    )
    parser.add_argument(
        "--speed_physical_std_anneal_updates",
        type=int,
        default=0,
        help=(
            "Formal update whose rollout first uses speed_physical_std_final; "
            "zero preserves fixed-std training"
        ),
    )
    parser.add_argument(
        "--speed_exploration_mode",
        choices=SPEED_EXPLORATION_MODES,
        default=BASELINE_EXPLORATION_MODE,
        help=(
            "Training-only speed exploration arm. Baseline preserves independent "
            "0.15 Gaussian sampling; other choices are the preregistered C/T/CT arms"
        ),
    )
    parser.add_argument(
        "--ordinary_offline_fast_fraction",
        type=float,
        default=None,
        help=(
            "Fraction of ordinary-role episodes drawn from the off-line fast "
            "subset (opponent on raceline0/2 with speed scale >= 0.7). Omit to "
            "keep the single uniform ordinary queue. The subset is derived from "
            "the existing ordinary scenarios, so only sampling weight changes"
        ),
    )
    parser.add_argument(
        "--corridor_gate_front_gap_m",
        type=float,
        default=None,
        help=(
            "Front-corridor arming gap for --speed_exploration_mode corridor_temporal. "
            "Omit to keep the shipped 2.0 m CT-v2 geometry bit-identically; a smaller "
            "value narrows where temporal exploration is armed"
        ),
    )

    # PPO configuration
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.995)
    parser.add_argument("--clip_range", type=float, default=0.15)
    parser.add_argument(
        "--target_kl",
        type=float,
        default=None,
        help="Optional PPO actor early-stop target; disabled when omitted",
    )
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
    if not args.hard_neighbor_cache_dir.strip():
        raise ValueError("hard_neighbor_cache_dir must not be empty")
    output_dir = Path(args.output_dir).expanduser().resolve()
    collision_cache_dir = Path(args.collision_cache_dir).expanduser().resolve()
    hard_neighbor_cache_dir = Path(args.hard_neighbor_cache_dir).expanduser().resolve()
    if output_dir == POST_TRAINED_ROOT or POST_TRAINED_ROOT not in output_dir.parents:
        raise ValueError(f"output_dir must be inside the project post-trained directory: {POST_TRAINED_ROOT}")
    if output_dir == collision_cache_dir:
        raise ValueError("output_dir and collision_cache_dir must be different directories")
    if collision_cache_dir == hard_neighbor_cache_dir:
        raise ValueError("collision_cache_dir and hard_neighbor_cache_dir must be different directories")
    if output_dir == hard_neighbor_cache_dir:
        raise ValueError("output_dir and hard_neighbor_cache_dir must be different directories")
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
        "risk_longitudinal_clearance_m",
        "gamma",
        "gae_lambda",
        "clip_range",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not np.isfinite(args.risk_longitudinal_clearance_m):
        raise ValueError(
            "risk_longitudinal_clearance_m must be finite"
        )
    if args.gamma > 1.0 or args.gae_lambda > 1.0:
        raise ValueError("gamma and gae_lambda must be at most 1")
    if args.target_kl is not None and (not np.isfinite(args.target_kl) or args.target_kl <= 0.0):
        raise ValueError("target_kl must be positive when enabled")
    if args.speed_physical_std_final is None:
        if args.speed_physical_std_anneal_updates != 0:
            raise ValueError(
                "speed_physical_std_anneal_updates requires "
                "speed_physical_std_final"
            )
    else:
        if (
            not np.isfinite(args.speed_physical_std_final)
            or args.speed_physical_std_final <= 0.0
        ):
            raise ValueError(
                "speed_physical_std_final must be positive and finite"
            )
        if args.speed_physical_std_final >= args.speed_physical_std:
            raise ValueError(
                "speed_physical_std_final must be below speed_physical_std"
            )
        if not 2 <= args.speed_physical_std_anneal_updates <= args.num_updates:
            raise ValueError(
                "speed_physical_std_anneal_updates must be in "
                "[2, num_updates]"
            )
    if args.speed_exploration_mode != BASELINE_EXPLORATION_MODE:
        if abs(float(args.speed_physical_std) - 0.15) > 1e-12:
            raise ValueError(
                "Structured speed exploration requires speed_physical_std=0.15"
            )
        if (
            args.speed_physical_std_final is not None
            or args.speed_physical_std_anneal_updates != 0
        ):
            raise ValueError(
                "Structured speed exploration cannot be combined with std annealing"
            )
    if args.ordinary_offline_fast_fraction is not None:
        value = args.ordinary_offline_fast_fraction
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(
                "ordinary_offline_fast_fraction must be a finite value in (0, 1)"
            )
    if args.corridor_gate_front_gap_m is not None:
        if args.speed_exploration_mode != CORRIDOR_TEMPORAL_EXPLORATION_MODE:
            raise ValueError(
                "corridor_gate_front_gap_m only applies to "
                "--speed_exploration_mode corridor_temporal"
            )
        if (
            not np.isfinite(args.corridor_gate_front_gap_m)
            or args.corridor_gate_front_gap_m <= 0.0
        ):
            raise ValueError("corridor_gate_front_gap_m must be finite and positive")
    if args.hard_neighbors:
        if args.hard_neighbor_fraction is not None and (
            not np.isfinite(args.hard_neighbor_fraction)
            or not 0.0 < args.hard_neighbor_fraction < 1.0
        ):
            raise ValueError(
                "hard_neighbor_fraction must be a finite value in (0, 1) "
                "when --hard_neighbors is enabled"
            )
    elif args.hard_neighbor_fraction is not None:
        raise ValueError(
            "hard_neighbor_fraction may only be set when --hard_neighbors is enabled"
        )
    if args.allow_collision_cache_actor_mismatch:
        if args.reclassify_collisions:
            raise ValueError(
                "allow_collision_cache_actor_mismatch cannot be combined "
                "with reclassify_collisions"
            )
        if args.hard_neighbors:
            raise ValueError(
                "allow_collision_cache_actor_mismatch is limited to the "
                "base collision cache"
            )
        required_cache_files = (
            "classification_config.json",
            "candidate_outcomes.jsonl",
            "collision_scenarios.json",
            "classification_summary.json",
        )
        missing_cache_files = [
            name
            for name in required_cache_files
            if not (collision_cache_dir / name).is_file()
        ]
        if missing_cache_files:
            raise FileNotFoundError(
                "Actor-mismatch reuse requires a complete existing collision "
                f"cache; missing: {missing_cache_files}"
            )
    if args.fixed_collision_pool_file is not None:
        fixed_pool_path = Path(
            args.fixed_collision_pool_file
        ).expanduser().resolve()
        if not fixed_pool_path.is_file():
            raise FileNotFoundError(
                f"Fixed collision pool does not exist: {fixed_pool_path}"
            )
        if (
            args.reclassify_collisions
            or args.allow_collision_cache_actor_mismatch
            or args.hard_neighbors
            or args.hard_neighbor_fraction is not None
        ):
            raise ValueError(
                "fixed_collision_pool_file cannot be combined with collision "
                "reclassification, actor-mismatch reuse, or hard neighbors"
            )
    if not args.postpass_penalty and args.postpass_proximity_power != 2:
        raise ValueError(
            "postpass_proximity_power may differ from 2 only when "
            "--postpass_penalty is enabled"
        )


def build_model(vector_env, args, device, recorder: TrainingRecorder) -> End2RaceRecurrentPPO:
    return End2RaceRecurrentPPO(
        End2RaceGRUPolicy,
        vector_env,
        actor_epochs=args.actor_epochs,
        critic_epochs=args.critic_epochs,
        recorder=recorder,
        speed_physical_std_final=args.speed_physical_std_final,
        speed_physical_std_anneal_updates=(
            args.speed_physical_std_anneal_updates
        ),
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
        target_kl=args.target_kl,
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
            "speed_exploration_mode": args.speed_exploration_mode,
        },
        verbose=1,
    )


def effective_ppo_config(args) -> dict:
    """Return the file configuration with explicit experiment overrides applied."""

    config = dict(PPO_CONFIG)
    config["risk_longitudinal_clearance_m"] = float(
        args.risk_longitudinal_clearance_m
    )
    config["ordinary_startpoint_count"] = int(
        getattr(
            args,
            "ordinary_startpoint_count",
            PPO_CONFIG["ordinary_startpoint_count"],
        )
    )
    return config


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    configure_training_numerics()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(
        f"PPO training configuration: output_dir={Path(args.output_dir).expanduser().resolve()}, pretrained_model_path={Path(args.pretrained_model_path).expanduser().resolve()}, "
        f"map={args.map_name}, critic={args.critic}, n_envs={args.n_envs}, env_workers={args.env_workers}, n_steps={args.n_steps}, "
        f"batch_size={args.batch_size}, num_updates={args.num_updates}, target_kl={args.target_kl}, "
        f"hard_neighbors={args.hard_neighbors}, hard_neighbor_fraction={args.hard_neighbor_fraction}, "
        f"fixed_collision_pool_file={args.fixed_collision_pool_file}, "
        f"ordinary_startpoint_count={args.ordinary_startpoint_count}, "
        f"speed_physical_std={args.speed_physical_std}, "
        f"speed_physical_std_final={args.speed_physical_std_final}, "
        f"speed_physical_std_anneal_updates={args.speed_physical_std_anneal_updates}, "
        f"speed_exploration_mode={args.speed_exploration_mode}, "
        f"corridor_gate_front_gap_m={args.corridor_gate_front_gap_m}, "
        f"ordinary_offline_fast_fraction={args.ordinary_offline_fast_fraction}, "
        f"risk_longitudinal_clearance_m={args.risk_longitudinal_clearance_m}, "
        f"postpass_penalty={args.postpass_penalty}, "
        f"postpass_proximity_power={args.postpass_proximity_power}, "
        f"allow_collision_cache_actor_mismatch={args.allow_collision_cache_actor_mismatch}, "
        f"seed={args.seed}",
        flush=True,
    )
    print("[1/5] Building collision candidates", flush=True)
    candidates = expanded_scenarios(args.map_name)
    print("[2/5] Loading or classifying collision pool", flush=True)
    if args.fixed_collision_pool_file is None:
        collision_scenarios, collision_cache_info = (
            resolve_training_collision_scenarios(
                args,
                candidates,
                START_METHOD,
            )
        )
    else:
        collision_scenarios, collision_cache_info = (
            load_fixed_collision_pool(
                args.fixed_collision_pool_file,
                map_name=args.map_name,
            )
        )
    collision_cache_info = {
        **collision_cache_info,
        "hard_neighbor_sampling_fraction": args.hard_neighbor_fraction,
        "hard_neighbor_sampling_mode": (
            "stratified_collision_episode_reset"
            if args.hard_neighbor_fraction is not None
            else "uniform_merged_collision_pool"
            if args.hard_neighbors
            else None
        ),
    }
    print("[3/5] Building ordinary scenarios", flush=True)
    ordinary_scenario_set = ordinary_scenarios(
        args.map_name,
        args.ordinary_startpoint_count,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    recorder = TrainingRecorder(args.output_dir, args.hidden_scale)
    recorder.write_scenario_pools(
        collision_scenarios,
        ordinary_scenario_set,
        collision_cache_info,
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
        risk_longitudinal_clearance_m=(
            args.risk_longitudinal_clearance_m
        ),
        hard_neighbor_fraction=args.hard_neighbor_fraction,
        postpass_penalty=args.postpass_penalty,
        postpass_proximity_power=args.postpass_proximity_power,
        speed_exploration_mode=args.speed_exploration_mode,
        corridor_gate_front_gap_m=args.corridor_gate_front_gap_m,
        ordinary_offline_fast_fraction=args.ordinary_offline_fast_fraction,
    )
    try:
        privileged_normalization = (
            vector_env.env_method("privileged_normalization_metadata", indices=[0])[0]
            if args.critic in P20_CRITIC_VARIANTS
            else {}
        )
        recorder.write_run_config(
            args,
            effective_ppo_config(args),
            {
                "WARMUP_MAX_EPOCHS": WARMUP_MAX_EPOCHS,
                "WARMUP_PATIENCE": WARMUP_PATIENCE,
                "WARMUP_TRAIN_FRACTION": WARMUP_TRAIN_FRACTION,
                "VALUE_LOSS_COEFFICIENT": VALUE_LOSS_COEFFICIENT,
                "MAX_GRAD_NORM": MAX_GRAD_NORM,
                "STEERING_LATENT_STD": args.steering_latent_std,
                "SPEED_PHYSICAL_STD": args.speed_physical_std,
                "SPEED_PHYSICAL_STD_FINAL": args.speed_physical_std_final,
                "SPEED_PHYSICAL_STD_ANNEAL_UPDATES": (
                    args.speed_physical_std_anneal_updates
                ),
                "SPEED_EXPLORATION": exploration_metadata(
                    args.speed_exploration_mode,
                    corridor_gate_config=(
                        None
                        if args.corridor_gate_front_gap_m is None
                        else FrontCorridorGateConfig(
                            maximum_front_gap_m=float(args.corridor_gate_front_gap_m)
                        )
                    ),
                ),
                "PRIVILEGED_FEATURE_SIZE": PRIVILEGED_FEATURE_SIZE,
                "PRIVILEGED_FEATURE_NAMES": list(PRIVILEGED_FEATURE_NAMES),
                "PRIVILEGED_FEATURE_LOWS": list(PRIVILEGED_FEATURE_LOWS),
                "PRIVILEGED_FEATURE_HIGHS": list(PRIVILEGED_FEATURE_HIGHS),
                "PRIVILEGED_NORMALIZATION": privileged_normalization,
                "POSTPASS_PENALTY_CONFIG": (
                    fixed_postpass_config_metadata(
                        args.postpass_proximity_power
                    )
                ),
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
