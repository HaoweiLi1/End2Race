#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from latticeplanner.utils import load_config
from ppo.env import CentralScheduleSubprocVecEnv, FrontCorridorGateConfig
from ppo.policy import *
from ppo.rollout import *
from ppo.scenarios import expanded_scenarios, ordinary_scenarios, resolve_collision_scenarios
from utils import TrainingRecorder

CONFIG = load_config("ppo/ppo_config.yaml")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train End2Race PPO")

    # Model paths
    parser.add_argument("--pretrained_model_path", type=str, default="pretrained/end2race.pth")
    parser.add_argument("--output_dir", type=str, default="post-trained/ppo")

    # Model configuration
    parser.add_argument("--hidden_scale", type=int, default=4)

    # Environment configuration
    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collision_cache_dir", type=str, default="post-trained/collision-cache/pretrained_end2race_austin_collision_pool_479")
    parser.add_argument("--first_action_preference_dataset", type=str, default="")
    parser.add_argument("--first_action_preference_step_fraction", type=float, default=0.0)
    parser.add_argument("--online_same_state_branch_ppo", action="store_true")
    parser.add_argument("--collision_prefix_branch_ppo", action="store_true")

    # Rollout configuration
    parser.add_argument("--n_steps", type=int, default=6400)
    parser.add_argument("--batch_size", type=int, default=12800)
    parser.add_argument("--num_updates", type=int, default=30)

    # Training configuration
    parser.add_argument("--actor_epochs", type=int, default=2)
    parser.add_argument("--critic_epochs", type=int, default=5)
    parser.add_argument("--gru_learning_rate", type=float, default=3.0e-6)
    parser.add_argument("--head_learning_rate", type=float, default=3.0e-5)
    parser.add_argument("--critic_learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--steering_latent_std", type=float, default=0.03)
    parser.add_argument("--speed_physical_std", type=float, default=0.15)
    parser.add_argument("--speed_noise_hold_steps", type=int, default=1)
    parser.add_argument("--front_corridor_speed_noise_hold_steps", type=int, default=0)
    parser.add_argument("--ppo_loss_sample_stride", type=int, default=1)

    # PPO configuration
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.995)
    parser.add_argument("--clip_range", type=float, default=0.20)
    return parser.parse_args()


def configure_training_numerics():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False


def build_model(vector_env, args, device, recorder):
    return End2RaceRecurrentPPO(
        End2RaceGRUPolicy,
        vector_env,
        actor_epochs=args.actor_epochs,
        critic_epochs=args.critic_epochs,
        recorder=recorder,
        first_action_preference_dataset=args.first_action_preference_dataset,
        first_action_preference_step_fraction=args.first_action_preference_step_fraction,
        online_same_state_branch_ppo=args.online_same_state_branch_ppo,
        collision_prefix_branch_ppo=args.collision_prefix_branch_ppo,
        ppo_loss_sample_stride=args.ppo_loss_sample_stride,
        learning_rate=1.0,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=CONFIG.value_loss_coefficient,
        max_grad_norm=CONFIG.max_grad_norm,
        seed=args.seed,
        device=device,
        policy_kwargs={
            "checkpoint_path": args.pretrained_model_path,
            "hidden_scale": args.hidden_scale,
            "critic_variant": "privilege_gru",
            "gru_learning_rate": args.gru_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "critic_learning_rate": args.critic_learning_rate,
            "steering_latent_std": args.steering_latent_std,
            "speed_physical_std": args.speed_physical_std,
            "speed_noise_hold_steps": args.speed_noise_hold_steps,
            "front_corridor_speed_noise_hold_steps": args.front_corridor_speed_noise_hold_steps,
        },
        verbose=1,
    )


def main():
    args = parse_arguments()
    configure_training_numerics()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(
        f"PPO training configuration: output_dir={Path(args.output_dir).expanduser().resolve()}, pretrained_model_path={Path(args.pretrained_model_path).expanduser().resolve()}, "
        f"map={args.map_name}, critic=privilege_gru, n_envs={args.n_envs}, n_steps={args.n_steps}, "
        f"batch_size={args.batch_size}, num_updates={args.num_updates}, "
        f"speed_physical_std={args.speed_physical_std}, "
        f"speed_noise_hold_steps={args.speed_noise_hold_steps}, front_corridor_speed_noise_hold_steps={args.front_corridor_speed_noise_hold_steps}, "
        f"ppo_loss_sample_stride={args.ppo_loss_sample_stride}, "
        f"front_corridor_gate_maximum_gap_m={CONFIG.front_corridor_gate_maximum_gap_m}, "
        f"ordinary_offline_fast_fraction={CONFIG.ordinary_offline_fast_fraction}, "
        f"first_action_preference_dataset={args.first_action_preference_dataset or 'disabled'}, first_action_preference_step_fraction={args.first_action_preference_step_fraction}, "
        f"online_same_state_branch_ppo={args.online_same_state_branch_ppo}, "
        f"collision_prefix_branch_ppo={args.collision_prefix_branch_ppo}, "
        f"seed={args.seed}",
        flush=True,
    )
    print("[1/5] Building collision candidates", flush=True)
    candidates = expanded_scenarios(args.map_name)
    print("[2/5] Loading collision pool", flush=True)
    collision_scenarios = resolve_collision_scenarios(
        args,
        candidates,
    )
    collision_cache_info = {
        "mode": "fixed",
        "cache_dir": str(Path(args.collision_cache_dir).expanduser().resolve()),
        "candidate_count": len(candidates),
        "collision_count": len(collision_scenarios),
    }
    print("[3/5] Building ordinary scenarios", flush=True)
    ordinary_scenario_set = ordinary_scenarios(args.map_name)
    device = torch.device("cuda")
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
        CONFIG.start_method,
        args.seed,
        args.map_name,
        collision_scenarios,
        ordinary_scenario_set,
        privileged=True,
        reward_gamma=args.gamma,
        front_corridor_speed_noise_hold_steps=args.front_corridor_speed_noise_hold_steps,
        collision_prefix_branch_ppo=args.collision_prefix_branch_ppo,
    )
    try:
        privileged_normalization = vector_env.env_method("privileged_normalization_metadata", indices=[0])[0]
        recorder.write_run_config(
            args,
            vars(CONFIG),
            {
                "SPEED_EXPLORATION": exploration_metadata(
                    args.speed_noise_hold_steps,
                    args.front_corridor_speed_noise_hold_steps,
                    corridor_gate_config=FrontCorridorGateConfig(
                        maximum_front_gap_m=float(
                            CONFIG.front_corridor_gate_maximum_gap_m
                        )
                    ),
                ),
                "PRIVILEGED_FEATURE_SIZE": PRIVILEGED_FEATURE_SIZE,
                "PRIVILEGED_FEATURE_NAMES": list(PRIVILEGED_FEATURE_NAMES),
                "PRIVILEGED_FEATURE_LOWS": list(PRIVILEGED_FEATURE_LOWS),
                "PRIVILEGED_FEATURE_HIGHS": list(PRIVILEGED_FEATURE_HIGHS),
                "PRIVILEGED_NORMALIZATION": privileged_normalization,
                "FIRST_ACTION_PREFERENCE": {
                    "enabled": bool(args.first_action_preference_step_fraction > 0.0),
                    "dataset": args.first_action_preference_dataset,
                    "target_step_fraction": args.first_action_preference_step_fraction,
                    "loss": "balanced target/control episode mean of simulator-return-filtered first-action softplus log-prob pairs",
                    "beta": "calibrated once before the first actor optimizer step",
                    "training_hindsight": True,
                    "deployment_future_information": False,
                },
                "ONLINE_SAME_STATE_BRANCH_PPO": {
                    "enabled": args.online_same_state_branch_ppo,
                    "states_per_rollout": CONFIG.online_branch_states_per_rollout,
                    "actions_per_state": CONFIG.online_branch_actions_per_state,
                    "horizon_steps": CONFIG.online_branch_horizon_steps,
                    "loss_coefficient": CONFIG.online_branch_loss_coefficient,
                    "state_source": "current student on-policy Austin rollout only",
                    "action_source": "current frozen pi_old only",
                    "continuation_source": "same current frozen pi_old",
                    "previous_actor_or_trace_input": False,
                    "actor_objective": "standard clipped PPO surrogate over leave-one-out branch-return advantages",
                },
                "COLLISION_PREFIX_BRANCH_PPO": {
                    "enabled": args.collision_prefix_branch_ppo,
                    "lookback_steps": CONFIG.collision_prefix_branch_lookback_steps,
                    "actions_per_state": CONFIG.collision_prefix_branch_actions_per_state,
                    "horizon_steps": CONFIG.collision_prefix_branch_horizon_steps,
                    "loss_coefficient": CONFIG.collision_prefix_branch_loss_coefficient,
                    "state_source": "current student collisions reconstructed one second before contact",
                    "previous_actor_or_trace_input": False,
                    "actor_objective": "standard clipped PPO surrogate over leave-one-out branch-return advantages",
                },
                "PPO_LOSS_SAMPLING": {
                    "collection_hz": 100,
                    "recurrent_replay_hz": 100,
                    "gae_hz": 100,
                    "loss_sample_stride": args.ppo_loss_sample_stride,
                    "direct_loss_hz": 100.0 / args.ppo_loss_sample_stride,
                },
            },
        )
        print("[5/5] Building PPO model", flush=True)
        model = build_model(vector_env, args, device, recorder)
        total_rollouts = args.num_updates + 1
        model.learn(total_timesteps=args.n_envs * args.n_steps * total_rollouts, log_interval=1, progress_bar=False)
    finally:
        vector_env.close()
    print("Training completed successfully", flush=True)


if __name__ == "__main__":
    main()
