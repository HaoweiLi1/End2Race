import argparse
import math
import os
from dataclasses import fields
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim

from model import End2Race
from model_ppo import End2RaceActorCritic, load_actor_critic, load_frozen_bc
from env_ppo import End2RacePPOEnv, RewardWeights


BOOL_INFO_KEYS = (
    "terminated",
    "truncated",
    "collision",
    "timeout",
    "success",
    "severe",
    "post_overtake_collision",
    "action_was_clipped",
)

MEAN_INFO_KEYS = (
    "reward_progress",
    "reward_rel_progress",
    "reward_overtake_progress",
    "reward_opponent_risk",
    "reward_smooth",
    "reward_steer_mag",
    "reward_collision",
    "reward_overtake_success",
    "reward_speed",
    "reward_timeout",
    "opponent_risk",
    "safe_overtake_hold_time",
)


class RolloutBuffer:
    """Serial recurrent PPO rollout buffer with separate termination flags."""

    def __init__(self, rollout_steps: int, gamma: float = 0.997, gae_lambda: float = 0.95):
        self.rollout_steps = int(rollout_steps)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.lidar = np.zeros((self.rollout_steps, 360), dtype=np.float32)
        self.prev_speed = np.zeros((self.rollout_steps, 1), dtype=np.float32)
        self.raw_actions = np.zeros((self.rollout_steps, 2), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.values = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.log_probs = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.terminateds = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.truncateds = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.trunc_next_values = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.episode_starts = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.advantages = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.returns = np.zeros((self.rollout_steps,), dtype=np.float32)
        self.ptr = 0

    def reset(self) -> None:
        self.ptr = 0

    def add(
        self,
        lidar: np.ndarray,
        prev_speed: np.ndarray,
        raw_action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        terminated: bool,
        truncated: bool,
        trunc_next_value: float,
        episode_start: bool,
    ) -> None:
        if self.ptr >= self.rollout_steps:
            raise RuntimeError("RolloutBuffer overflow.")
        self.lidar[self.ptr] = np.asarray(lidar, dtype=np.float32)
        self.prev_speed[self.ptr] = np.asarray(prev_speed, dtype=np.float32).reshape(1)
        self.raw_actions[self.ptr] = np.asarray(raw_action, dtype=np.float32).reshape(2)
        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = float(log_prob)
        self.terminateds[self.ptr] = float(terminated)
        self.truncateds[self.ptr] = float(truncated)
        self.trunc_next_values[self.ptr] = float(trunc_next_value)
        self.episode_starts[self.ptr] = float(episode_start)
        self.ptr += 1

    def compute_returns_and_advantage(
        self,
        candidate_last_value: float,
        last_terminated: bool,
        last_truncated: bool,
    ) -> None:
        if self.ptr != self.rollout_steps:
            raise RuntimeError(f"Buffer has {self.ptr} steps, expected {self.rollout_steps}.")

        gae = 0.0
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                if last_terminated:
                    next_value = 0.0
                    boundary = True
                elif last_truncated:
                    next_value = float(self.trunc_next_values[t])
                    boundary = True
                else:
                    next_value = float(candidate_last_value)
                    boundary = False
            elif self.terminateds[t] > 0.5:
                next_value = 0.0
                boundary = True
            elif self.truncateds[t] > 0.5:
                next_value = float(self.trunc_next_values[t])
                boundary = True
            else:
                next_value = float(self.values[t + 1])
                boundary = False

            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            gae = delta if boundary else delta + self.gamma * self.gae_lambda * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def full_batch_tensors(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        lidar_b = torch.tensor(self.lidar, dtype=torch.float32, device=device).unsqueeze(0)
        spd_b = torch.tensor(self.prev_speed, dtype=torch.float32, device=device).unsqueeze(0)
        act_b = torch.tensor(self.raw_actions, dtype=torch.float32, device=device).unsqueeze(0)
        old_logp_b = torch.tensor(self.log_probs, dtype=torch.float32, device=device).unsqueeze(0)
        adv_b = torch.tensor(self.advantages, dtype=torch.float32, device=device).unsqueeze(0)
        ret_b = torch.tensor(self.returns, dtype=torch.float32, device=device).unsqueeze(0)
        starts_b = torch.tensor(self.episode_starts, dtype=torch.float32, device=device).unsqueeze(0)
        return lidar_b, spd_b, act_b, old_logp_b, adv_b, ret_b, starts_b


def forward_policy_sequence(
    ac: End2RaceActorCritic,
    lidar_b: torch.Tensor,
    spd_b: torch.Tensor,
    starts_b: torch.Tensor,
    hidden_size: int,
    device: torch.device,
) -> Tuple[torch.distributions.Normal, torch.Tensor]:
    hidden = torch.zeros(1, 1, hidden_size, device=device)
    means: List[torch.Tensor] = []
    stds: List[torch.Tensor] = []
    vals: List[torch.Tensor] = []
    for t in range(lidar_b.shape[1]):
        if starts_b[0, t].item() > 0.5:
            hidden = torch.zeros_like(hidden)
        dist_t, val_t, hidden = ac.forward(lidar_b[:, t : t + 1], spd_b[:, t : t + 1], hidden)
        means.append(dist_t.mean)
        stds.append(dist_t.stddev)
        vals.append(val_t)
    dist = torch.distributions.Normal(torch.cat(means, dim=1), torch.cat(stds, dim=1))
    values = torch.cat(vals, dim=1)
    return dist, values


def forward_frozen_bc_sequence(
    frozen_bc: End2Race,
    lidar_b: torch.Tensor,
    spd_b: torch.Tensor,
    starts_b: torch.Tensor,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    hidden = torch.zeros(1, 1, hidden_size, device=device)
    means: List[torch.Tensor] = []
    for t in range(lidar_b.shape[1]):
        if starts_b[0, t].item() > 0.5:
            hidden = torch.zeros_like(hidden)
        mean_t, hidden = frozen_bc(lidar_b[:, t : t + 1], spd_b[:, t : t + 1], hidden)
        means.append(mean_t)
    return torch.cat(means, dim=1)


def _obs_to_tensors(obs: Dict[str, np.ndarray], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    lidar_t = torch.tensor(obs["lidar"], dtype=torch.float32, device=device).view(1, 1, -1)
    spd_t = torch.tensor(obs["prev_speed"], dtype=torch.float32, device=device).view(1, 1, -1)
    return lidar_t, spd_t


@torch.no_grad()
def validate_replay_identity(
    ac: End2RaceActorCritic,
    buffer: RolloutBuffer,
    device: torch.device,
    atol: float = 1e-5,
) -> Dict[str, float]:
    lidar_b, spd_b, act_b, old_logp_b, _, _, starts_b = buffer.full_batch_tensors(device)
    hidden_size = ac.actor.gru.hidden_size
    dist, _ = forward_policy_sequence(ac, lidar_b, spd_b, starts_b, hidden_size, device)
    new_logp = dist.log_prob(act_b).sum(-1)
    logp_diff = new_logp - old_logp_b
    max_err = float(logp_diff.abs().max().item())
    ratio_mean = float(torch.exp(logp_diff).mean().item())
    if max_err >= atol or abs(ratio_mean - 1.0) >= atol:
        raise RuntimeError(
            f"Replay log-prob mismatch: max_err={max_err:.8g}, ratio_mean={ratio_mean:.8g}, atol={atol:.8g}"
        )
    return {
        "max_replay_logp_error": max_err,
        "replay_ratio_mean": ratio_mean,
    }


def collect_rollout(
    env: End2RacePPOEnv,
    ac: End2RaceActorCritic,
    buffer: RolloutBuffer,
    device: torch.device,
    hidden_size: int,
    scenario: Dict[str, Any] = None,
) -> Dict[str, Any]:
    buffer.reset()
    obs = env.reset(scenario=scenario)
    hidden = torch.zeros(1, 1, hidden_size, device=device)
    episode_start = True

    episode_return = 0.0
    completed_returns: List[float] = []
    last_terminated = False
    last_truncated = False
    info_acc: Dict[str, List[float]] = {key: [] for key in BOOL_INFO_KEYS + MEAN_INFO_KEYS}

    for _ in range(buffer.rollout_steps):
        lidar_t, spd_t = _obs_to_tensors(obs, device)
        with torch.no_grad():
            action, logp, value, new_hidden = ac.act(lidar_t, spd_t, hidden, deterministic=False)

        raw_np = action.view(-1).detach().cpu().numpy().astype(np.float32)
        next_obs, reward, terminated, truncated, info = env.step(raw_np)
        if terminated and truncated:
            raise RuntimeError("Environment returned both terminated=True and truncated=True.")
        episode_return += float(reward)

        if "action_was_clipped" not in info and "raw_ego_action" in info and "executed_ego_action" in info:
            raw = np.asarray(info["raw_ego_action"], dtype=np.float32)
            executed = np.asarray(info["executed_ego_action"], dtype=np.float32)
            info["action_was_clipped"] = bool(np.any(np.abs(raw - executed) > 1e-6))
        for key, vals in info_acc.items():
            if key in info:
                vals.append(float(info[key]))

        trunc_next_value = 0.0
        if truncated:
            with torch.no_grad():
                lidar_next, spd_next = _obs_to_tensors(next_obs, device)
                _, val_next, _ = ac.forward(lidar_next, spd_next, new_hidden)
                trunc_next_value = float(val_next[:, -1].item())

        buffer.add(
            obs["lidar"],
            obs["prev_speed"],
            raw_np,
            reward,
            float(value.squeeze().item()),
            float(logp.squeeze().item()),
            terminated,
            truncated,
            trunc_next_value,
            episode_start,
        )

        last_terminated = bool(terminated)
        last_truncated = bool(truncated)

        if terminated or truncated:
            completed_returns.append(episode_return)
            episode_return = 0.0
            obs = env.reset(scenario=scenario)
            hidden = torch.zeros(1, 1, hidden_size, device=device)
            episode_start = True
        else:
            obs = next_obs
            hidden = new_hidden.detach()
            episode_start = False

    with torch.no_grad():
        lidar_t, spd_t = _obs_to_tensors(obs, device)
        _, val_boot, _ = ac.forward(lidar_t, spd_t, hidden)
        candidate_last_value = float(val_boot[:, -1].item())

    buffer.compute_returns_and_advantage(candidate_last_value, last_terminated, last_truncated)

    completed_or_partial_returns = list(completed_returns)
    partial_episode_return = float("nan")
    if not (last_terminated or last_truncated):
        partial_episode_return = float(episode_return)
        completed_or_partial_returns.append(partial_episode_return)

    metrics = {
        "mean_episode_return": (
            float(np.mean(completed_or_partial_returns)) if completed_or_partial_returns else float("nan")
        ),
        "mean_completed_episode_return": float(np.mean(completed_returns)) if completed_returns else float("nan"),
        "num_completed_episodes": len(completed_returns),
        "partial_episode_return": partial_episode_return,
        "rollout_return": float(np.sum(buffer.rewards)),
        "bootstrap_value": float(candidate_last_value),
    }
    for key in BOOL_INFO_KEYS:
        values = info_acc[key]
        metric_key = "action_was_clipped_fraction" if key == "action_was_clipped" else f"{key}_rate"
        metrics[metric_key] = float(np.mean(values)) if values else float("nan")
    for key in MEAN_INFO_KEYS:
        values = info_acc[key]
        metrics[f"mean_{key}"] = float(np.mean(values)) if values else float("nan")
    return metrics


def ppo_update(
    ac: End2RaceActorCritic,
    frozen_bc: End2Race,
    buffer: RolloutBuffer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    lidar_b, spd_b, act_b, old_logp_b, adv_b, ret_b, starts_b = buffer.full_batch_tensors(device)
    adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)
    hidden_size = ac.actor.gru.hidden_size
    bc_hidden_size = frozen_bc.gru.hidden_size

    metrics: Dict[str, List[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "bc_anchor": [],
        "steer_anchor": [],
        "speed_anchor": [],
        "bound_loss": [],
        "approx_kl": [],
        "post_step_approx_kl": [],
        "clip_fraction": [],
        "ratio_mean": [],
        "ratio_min": [],
        "ratio_max": [],
        "grad_norm": [],
    }
    num_updates = 0
    early_stopped = False

    for _epoch in range(args.ppo_epochs):
        dist, values = forward_policy_sequence(ac, lidar_b, spd_b, starts_b, hidden_size, device)
        new_logp = dist.log_prob(act_b).sum(-1)
        ratio = torch.exp(new_logp - old_logp_b)
        with torch.no_grad():
            log_ratio = new_logp - old_logp_b
            approx_kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > args.clip_eps).float().mean()
            ratio_mean = ratio.mean()
            ratio_min = ratio.min()
            ratio_max = ratio.max()

        surr1 = ratio * adv_b
        surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_b
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = 0.5 * (values - ret_b).pow(2).mean()

        with torch.no_grad():
            bc_mean = forward_frozen_bc_sequence(frozen_bc, lidar_b, spd_b, starts_b, bc_hidden_size, device)
        steer_anchor = ((dist.mean[..., 0] - bc_mean[..., 0]) / 0.52).pow(2).mean()
        speed_anchor = ((dist.mean[..., 1] - bc_mean[..., 1]) / args.max_speed).pow(2).mean()
        bc_anchor = steer_anchor + speed_anchor

        steer_bound = torch.relu(dist.mean[..., 0].abs() - 0.52).pow(2).mean()
        speed_bound = (
            torch.relu(-dist.mean[..., 1]).pow(2).mean()
            + torch.relu(dist.mean[..., 1] - args.max_speed).pow(2).mean()
        )
        bound_loss = steer_bound + speed_bound
        entropy = dist.entropy().sum(-1).mean()

        loss = (
            policy_loss
            + args.vf_coef * value_loss
            - args.ent_coef * entropy
            + args.beta_bc * bc_anchor
            + args.bound_coef * bound_loss
        )

        metrics["policy_loss"].append(float(policy_loss.item()))
        metrics["value_loss"].append(float(value_loss.item()))
        metrics["entropy"].append(float(entropy.item()))
        metrics["bc_anchor"].append(float(bc_anchor.item()))
        metrics["steer_anchor"].append(float(steer_anchor.item()))
        metrics["speed_anchor"].append(float(speed_anchor.item()))
        metrics["bound_loss"].append(float(bound_loss.item()))
        metrics["approx_kl"].append(float(approx_kl.item()))
        metrics["clip_fraction"].append(float(clip_fraction.item()))
        metrics["ratio_mean"].append(float(ratio_mean.item()))
        metrics["ratio_min"].append(float(ratio_min.item()))
        metrics["ratio_max"].append(float(ratio_max.item()))

        if approx_kl.item() > args.target_kl * 1.5:
            early_stopped = True
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(ac.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        num_updates += 1

        with torch.no_grad():
            post_dist, _ = forward_policy_sequence(ac, lidar_b, spd_b, starts_b, hidden_size, device)
            post_logp = post_dist.log_prob(act_b).sum(-1)
            post_log_ratio = post_logp - old_logp_b
            post_step_approx_kl = ((post_log_ratio.exp() - 1.0) - post_log_ratio).mean()
        metrics["grad_norm"].append(float(grad_norm.item()))
        metrics["post_step_approx_kl"].append(float(post_step_approx_kl.item()))

    out = {key: float(np.mean(values)) if values else float("nan") for key, values in metrics.items()}
    out["num_updates"] = float(num_updates)
    out["early_stopped"] = float(early_stopped)
    out["std_steer"] = float(ac.log_std.detach().exp()[0].item())
    out["std_speed"] = float(ac.log_std.detach().exp()[1].item())
    return out


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO fine-tune End2Race v1 actor.")

    parser.add_argument("--model_path", type=str, default="pretrained/end2race.pth")
    parser.add_argument("--bc_model_path", type=str, default="pretrained/end2race.pth")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_actor_path", type=str, default="pretrained/end2race_ppo.pth")
    parser.add_argument("--save_full_path", type=str, default="pretrained/end2race_ppo_full.pt")

    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--max_speed", type=float, default=20.0)
    parser.add_argument("--sim_duration", type=float, default=8.0)
    parser.add_argument("--fixed_scenario", action="store_true")
    parser.add_argument("--ego_idx", type=int, default=0)
    parser.add_argument("--interval_idx", type=int, default=15)
    parser.add_argument("--ego_raceline", type=str, default="raceline1")
    parser.add_argument("--opp_raceline", type=str, default="raceline1")
    parser.add_argument("--opp_speedscale", type=float, default=0.5)
    success_group = parser.add_mutually_exclusive_group()
    success_group.add_argument("--terminate_on_success", dest="terminate_on_success", action="store_true", default=True)
    success_group.add_argument(
        "--no_terminate_on_success",
        "--no-terminate_on_success",
        dest="terminate_on_success",
        action="store_false",
    )
    severe_group = parser.add_mutually_exclusive_group()
    severe_group.add_argument(
        "--terminate_on_severe_unsafe",
        dest="terminate_on_severe_unsafe",
        action="store_true",
        default=False,
    )
    severe_group.add_argument(
        "--no_terminate_on_severe_unsafe",
        "--no-terminate_on_severe_unsafe",
        dest="terminate_on_severe_unsafe",
        action="store_false",
    )
    parser.add_argument("--stage", type=int, default=1)

    parser.add_argument("--rollout_steps", type=int, default=1024)
    parser.add_argument("--ppo_epochs", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.05)
    parser.add_argument("--actor_lr", type=float, default=1e-5)
    parser.add_argument("--critic_lr", type=float, default=5e-5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.001)
    parser.add_argument("--beta_bc", type=float, default=2.0)
    parser.add_argument("--target_kl", type=float, default=0.03)
    parser.add_argument("--bound_coef", type=float, default=0.01)
    parser.add_argument("--hidden_scale", type=int, default=4)
    parser.add_argument("--steer_std", type=float, default=0.03)
    parser.add_argument("--speed_std", type=float, default=0.25)
    replay_group = parser.add_mutually_exclusive_group()
    replay_group.add_argument(
        "--validate_replay_identity",
        dest="validate_replay_identity",
        action="store_true",
        default=True,
    )
    replay_group.add_argument(
        "--no_validate_replay_identity",
        "--no-validate_replay_identity",
        dest="validate_replay_identity",
        action="store_false",
    )
    parser.add_argument("--replay_identity_atol", type=float, default=1e-5)

    parser.add_argument("--total_iterations", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    for field in fields(RewardWeights):
        parser.add_argument(f"--{field.name}", type=float, default=None)

    return parser.parse_args()


def _apply_reward_overrides(rw: RewardWeights, args: argparse.Namespace) -> None:
    for field in fields(RewardWeights):
        value = getattr(args, field.name)
        if value is not None:
            setattr(rw, field.name, float(value))


def main() -> None:
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)

    env = End2RacePPOEnv(
        map_name=args.map_name,
        max_speed=args.max_speed,
        sim_duration=args.sim_duration,
        terminate_on_success=args.terminate_on_success,
        terminate_on_severe_unsafe=args.terminate_on_severe_unsafe,
        seed=args.train_seed,
    )
    env.stage = args.stage
    _apply_reward_overrides(env.reward_weights, args)

    min_rollout_steps = int(math.ceil(args.sim_duration / env.timestep))
    if args.rollout_steps < min_rollout_steps:
        raise ValueError(
            f"--rollout_steps must be at least ceil(sim_duration / env.timestep) = {min_rollout_steps}; "
            f"got {args.rollout_steps}."
        )

    ac = End2RaceActorCritic(
        hidden_scale=args.hidden_scale,
        steer_std=args.steer_std,
        speed_std=args.speed_std,
    ).to(device)
    source = args.resume if args.resume and os.path.isfile(args.resume) else args.model_path
    load_actor_critic(ac, source, device)
    ac.train()
    hidden_size = ac.actor.gru.hidden_size

    frozen_bc = load_frozen_bc(args.bc_model_path, device, args.hidden_scale)

    critic_params = list(ac.value_head.parameters())
    critic_param_ids = {id(p) for p in critic_params}
    actor_params = [p for p in ac.parameters() if id(p) not in critic_param_ids]
    if id(ac.log_std) not in {id(p) for p in actor_params}:
        raise RuntimeError("ac.log_std is not in the actor optimizer parameter group.")
    optimizer = optim.Adam(
        [
            {"params": critic_params, "lr": args.critic_lr},
            {"params": actor_params, "lr": args.actor_lr},
        ]
    )

    start_iter = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and ckpt.get("optimizer"):
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        if isinstance(ckpt, dict):
            start_iter = int(ckpt.get("iteration", 0))

    buffer = RolloutBuffer(args.rollout_steps, gamma=args.gamma, gae_lambda=args.gae_lambda)
    fixed_scenario = None
    if args.fixed_scenario:
        fixed_scenario = {
            "map_name": args.map_name,
            "ego_raceline": args.ego_raceline,
            "opp_raceline": args.opp_raceline,
            "ego_idx": args.ego_idx,
            "interval_idx": args.interval_idx,
            "opp_speedscale": args.opp_speedscale,
        }

    try:
        for it in range(start_iter, args.total_iterations):
            roll_metrics = collect_rollout(env, ac, buffer, device, hidden_size, scenario=fixed_scenario)
            if args.validate_replay_identity:
                roll_metrics.update(
                    validate_replay_identity(ac, buffer, device, atol=args.replay_identity_atol)
                )
            update_metrics = ppo_update(ac, frozen_bc, buffer, optimizer, device, args)

            if it % 10 == 0:
                print(
                    f"iter {it:05d}  "
                    f"ret {roll_metrics['mean_episode_return']:.2f}  "
                    f"eps {roll_metrics['num_completed_episodes']}  "
                    f"coll {roll_metrics['collision_rate']:.3f}  "
                    f"trunc {roll_metrics['truncated_rate']:.3f}  "
                    f"clip_a {roll_metrics['action_was_clipped_fraction']:.3f}  "
                    f"replay {roll_metrics.get('max_replay_logp_error', float('nan')):.2e}  "
                    f"pi {update_metrics['policy_loss']:.4f}  "
                    f"v {update_metrics['value_loss']:.4f}  "
                    f"kl {update_metrics['approx_kl']:.4f}  "
                    f"post_kl {update_metrics['post_step_approx_kl']:.4f}  "
                    f"clip {update_metrics['clip_fraction']:.3f}  "
                    f"bc {update_metrics['bc_anchor']:.4f}  "
                    f"s_anc {update_metrics['steer_anchor']:.4f}  "
                    f"v_anc {update_metrics['speed_anchor']:.4f}  "
                    f"std_v {update_metrics['std_speed']:.3f}  "
                    f"early {int(update_metrics['early_stopped'])}"
                )

            if (it + 1) % args.save_every == 0:
                ac.save_full_checkpoint(args.save_full_path, optimizer, it + 1, vars(args))
                print(f"Saved checkpoint at iteration {it + 1}")

        ac.save_full_checkpoint(args.save_full_path, optimizer, args.total_iterations, vars(args))
        ac.save_actor_backbone(args.save_actor_path)

        test_model = End2Race(hidden_scale=args.hidden_scale)
        test_model.load_state_dict(torch.load(args.save_actor_path, map_location="cpu"))
        print(f"Saved actor backbone to {args.save_actor_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
