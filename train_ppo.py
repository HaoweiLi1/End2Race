import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from latticeplanner.utils import load_config
from ppo.env import CentralScheduleSubprocVecEnv
from ppo.policy import *
from ppo.rollout import *
from ppo.scenarios import ScenarioSpec, ordinary_scenarios
from utils import calculate_ppo_metrics, log_ppo


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


class End2RaceRecurrentPPO(RecurrentPPO):
    """Connect End2Race recurrent rollout and split actor/critic updates to SB3."""

    def __init__(self, *args, actor_epochs, critic_epochs, output_dir, config, **kwargs):
        self.actor_epochs = actor_epochs
        self.critic_epochs = critic_epochs
        self.output_dir = output_dir
        self.config = config
        self.warmup_completed = False
        self._rollout_episode_records = []
        self._last_exploration_gates = None
        kwargs["n_epochs"] = actor_epochs
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        """Create the policy, recurrent states, rollout buffer, and minibatch RNGs."""
        setup_recurrent_ppo(self)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """Run the End2Race recurrent collector through the SB3 lifecycle hook."""
        return collect_recurrent_rollout(self, env, callback, rollout_buffer, n_rollout_steps)

    def dump_logs(self, iteration=0):
        """Write the SB3 timestep summary without wall-clock fields."""
        del iteration
        log_ppo(
            self.output_dir,
            "rollout",
            {"num_timesteps": self.num_timesteps},
            logger=self.logger,
        )

    def train(self):
        """Run critic warm-up once, then one actor phase and one critic phase."""
        self.policy.set_training_mode(True)
        if not self.warmup_completed:
            metrics = warmup_critic(self)
            log_ppo(
                self.output_dir,
                "warmup",
                metrics,
                critic_state_dict=self.policy.value_net.state_dict(),
            )
            return

        update = self._n_updates + 1
        actor_statistics = train_actor(
            self,
            self.clip_range(self._current_progress_remaining),
        )
        critic_statistics = train_critic(self)
        self._n_updates = update
        metrics = calculate_ppo_metrics(
            update,
            self.num_timesteps,
            self._rollout_episode_records,
            *actor_statistics,
            *critic_statistics,
        )
        log_ppo(
            self.output_dir,
            "formal",
            metrics,
            logger=self.logger,
            actor_state_dict=self.policy.actor_checkpoint_state_dict(),
            critic_state_dict=self.policy.value_net.state_dict(),
        )


def main():
    """Load the fixed PPO inputs, build the vector environment, and train."""
    args = parse_arguments()
    config = load_config("ppo/ppo_config.yaml")
    configure_training_numerics()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(
        f"PPO training configuration: output_dir={Path(args.output_dir).expanduser().resolve()}, pretrained_model_path={Path(args.pretrained_model_path).expanduser().resolve()}, "
        f"map={args.map_name}, critic={Critic.name}, n_envs={args.n_envs}, n_steps={args.n_steps}, "
        f"batch_size={args.batch_size}, num_updates={args.num_updates}, "
        f"steering_latent_std={args.steering_latent_std}, speed_physical_std={args.speed_physical_std}, "
        f"speed_noise_hold_steps={args.speed_noise_hold_steps}, front_corridor_speed_noise_hold_steps={args.front_corridor_speed_noise_hold_steps}, "
        f"front_corridor_gate_maximum_gap_m={config.front_corridor_gate_maximum_gap_m}, "
        f"seed={args.seed}",
        flush=True,
    )
    print("[1/4] Loading collision pool", flush=True)
    with (Path(args.collision_cache_dir).expanduser().resolve() / "collision_scenarios.json").open("r", encoding="utf-8") as file:
        collision_scenarios = tuple(ScenarioSpec(**record) for record in json.load(file))
    print(f"Loaded {len(collision_scenarios)} collision scenarios", flush=True)
    print("[2/4] Building ordinary scenarios", flush=True)
    ordinary_scenario_set = ordinary_scenarios(args.map_name, config)
    reward_weights = np.asarray([
        config.progress_weight,
        config.relative_weight,
        config.collision_penalty,
        config.risk_potential_maximum,
    ], dtype=np.float64)
    device = torch.device("cuda")
    print(f"Using device: {device}", flush=True)
    print("[3/4] Creating vector environments", flush=True)
    privileged = Critic.privileged
    vector_env = CentralScheduleSubprocVecEnv(
        args.n_envs,
        config.start_method,
        args.seed,
        args.map_name,
        config,
        collision_scenarios,
        ordinary_scenario_set,
        privileged=privileged,
        reward_gamma=args.gamma,
        reward_weights=reward_weights,
        front_corridor_speed_noise_hold_steps=args.front_corridor_speed_noise_hold_steps,
    )
    try:
        training_constants = {
            "COLLISION_POOL": {
                "cache_dir": str(Path(args.collision_cache_dir).expanduser().resolve()),
                "collision_count": len(collision_scenarios),
            },
            "CRITIC": {
                "name": Critic.name,
                "recurrent": Critic.recurrent,
                "privileged": Critic.privileged,
            },
            "ACTION_EXPLORATION": exploration_metadata(
                args.speed_noise_hold_steps,
                args.front_corridor_speed_noise_hold_steps,
                config,
            ),
            "REWARD_WEIGHTS": {
                "progress_reward": float(reward_weights[0]),
                "relative_reward": float(reward_weights[1]),
                "collision_reward": float(reward_weights[2]),
                "risk_reward": float(reward_weights[3]),
            },
        }
        if privileged:
            training_constants.update({
                "PRIVILEGED_FEATURE_SIZE": PRIVILEGED_FEATURE_SIZE,
                "PRIVILEGED_FEATURE_NAMES": list(PRIVILEGED_FEATURE_NAMES),
                "PRIVILEGED_FEATURE_LOWS": list(PRIVILEGED_FEATURE_LOWS),
                "PRIVILEGED_FEATURE_HIGHS": list(PRIVILEGED_FEATURE_HIGHS),
                "PRIVILEGED_NORMALIZATION": vector_env.env_method("privileged_normalization_metadata", indices=[0])[0],
            })
        log_ppo(args.output_dir, "setup", {
            "run_config": {"args": vars(args), "ppo_config": vars(config), **training_constants},
            "collision_scenarios": [asdict(scenario) for scenario in collision_scenarios],
            "ordinary_scenarios": [asdict(scenario) for scenario in ordinary_scenario_set],
        })
        print("[4/4] Building PPO model", flush=True)
        model = End2RaceRecurrentPPO(
            End2RaceGRUPolicy,
            vector_env,
            actor_epochs=args.actor_epochs,
            critic_epochs=args.critic_epochs,
            output_dir=args.output_dir,
            config=config,
            learning_rate=1.0,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=config.value_loss_coefficient,
            max_grad_norm=config.max_grad_norm,
            seed=args.seed,
            device=device,
            policy_kwargs={
                "checkpoint_path": args.pretrained_model_path,
                "hidden_scale": args.hidden_scale,
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
        total_rollouts = args.num_updates + 1
        model.learn(total_timesteps=args.n_envs * args.n_steps * total_rollouts, log_interval=1, progress_bar=False)
    finally:
        vector_env.close()
    print("Training completed successfully", flush=True)


if __name__ == "__main__":
    main()
