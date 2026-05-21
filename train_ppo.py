#!/usr/bin/env python3
"""Recurrent PPO training for End2Race.

This three-file version imports all shared PPO machinery from ``utils_ppo.py``.
It supports both v1 compatibility and v2 safety-augmented modes.  The default
trainer is intentionally serial for correctness and smoke-testing; subprocess
vectorization can be added later without changing the actor/env/reward APIs.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Any, Dict

import numpy as np
import torch
import torch.optim as optim

from model import End2Race
from utils_ppo import (
    End2RaceActorCritic,
    End2RaceHazardActorCritic,
    End2RacePPOEnv,
    RolloutBuffer,
    forward_frozen_bc_sequence,
    forward_policy_sequence,
    load_actor_critic_checkpoint,
    load_end2race_actor,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recurrent PPO fine-tuning for End2Race")
    p.add_argument("--mode", type=str, default="compatibility", choices=("compatibility", "safety_augmented"))
    p.add_argument("--model_path", type=str, default="pretrained/end2race.pth")
    p.add_argument("--bc_model_path", type=str, default="pretrained/end2race.pth")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save_actor_path", type=str, default="pretrained/end2race_ppo.pth")
    p.add_argument("--save_full_path", type=str, default="")
    p.add_argument("--map_name", type=str, default="Austin")
    p.add_argument("--train_seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden_scale", type=int, default=4)

    p.add_argument("--rollout_steps", type=int, default=512)
    p.add_argument("--chunk_len", type=int, default=0, help="0 means use the full rollout sequence as one batch")
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.1)
    p.add_argument("--clip_eps_vf", type=float, default=0.2, help="Value-loss clip range; set <= 0 to disable")
    p.add_argument("--actor_lr", type=float, default=3e-5)
    p.add_argument("--critic_lr", type=float, default=1e-4)
    p.add_argument("--critic_warmup_iters", type=int, default=0, help="First N iterations train only critic (actor lr=0)")
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--beta_bc", type=float, default=1.0)
    p.add_argument("--target_kl", type=float, default=0.03)

    p.add_argument("--max_speed", type=float, default=20.0)
    p.add_argument("--sim_duration", type=float, default=8.0)
    p.add_argument("--terminate_on_success", dest="terminate_on_success", action="store_true", default=True)
    p.add_argument("--no_terminate_on_success", dest="terminate_on_success", action="store_false")
    p.add_argument("--terminate_on_severe_unsafe", action="store_true")

    p.add_argument("--total_iterations", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--vec_backend", type=str, default="serial", choices=("serial",))
    p.add_argument(
        "--save_actor_without_gate",
        action="store_true",
        help="Debug only: save v1 actor-only checkpoint without running eval_ppo.py gate.",
    )
    return p.parse_args()


def resolve_save_paths(args: argparse.Namespace) -> None:
    if not args.save_full_path:
        args.save_full_path = (
            "pretrained/end2race_ppo_full.pt" if args.mode == "compatibility" else "pretrained/end2race_ppo_aug.pt"
        )
    # Prevent accidental v2 overwrite of v1 full checkpoint when an old default is passed.
    if args.mode == "safety_augmented" and args.save_full_path == "pretrained/end2race_ppo_full.pt":
        args.save_full_path = "pretrained/end2race_ppo_aug.pt"


def make_env(args: argparse.Namespace) -> End2RacePPOEnv:
    env = End2RacePPOEnv(
        map_name=args.map_name,
        mode=args.mode,
        max_speed=args.max_speed,
        sim_duration=args.sim_duration,
        render=False,
        terminate_on_success=args.terminate_on_success,
        terminate_on_severe_unsafe=args.terminate_on_severe_unsafe,
        seed=args.train_seed,
    )
    env.stage = args.stage
    return env


def build_actor_critic(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    use_hazard = args.mode == "safety_augmented"
    source_path = args.resume if args.resume and os.path.isfile(args.resume) else args.model_path
    if use_hazard:
        inner = End2RaceActorCritic(hidden_scale=args.hidden_scale).to(device)
        ckpt = torch.load(source_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and ckpt.get("mode") == "safety_augmented" and "actor_critic" in ckpt:
            ac = End2RaceHazardActorCritic(inner).to(device)
            ac.load_state_dict(ckpt["actor_critic"])
        else:
            load_actor_critic_checkpoint(inner, source_path, device)
            ac = End2RaceHazardActorCritic(inner).to(device)
    else:
        ac = End2RaceActorCritic(hidden_scale=args.hidden_scale).to(device)
        load_actor_critic_checkpoint(ac, source_path, device)
    return ac


def _forward_value(
    ac: torch.nn.Module,
    obs: Dict[str, Any],
    hidden: torch.Tensor,
    device: torch.device,
    use_hazard: bool,
) -> float:
    lidar = torch.tensor(obs["lidar"], dtype=torch.float32, device=device).view(1, 1, -1)
    spd = torch.tensor(obs["prev_speed"], dtype=torch.float32, device=device).view(1, 1, -1)
    if use_hazard:
        haz = torch.tensor(obs["hazard"], dtype=torch.float32, device=device).view(1, 1, -1)
        _, value, _ = ac.forward(lidar, spd, haz, hidden)
    else:
        _, value, _ = ac.forward(lidar, spd, hidden)
    return float(value[:, -1].item())


def collect_rollout(
    env: End2RacePPOEnv,
    ac: torch.nn.Module,
    buffer: RolloutBuffer,
    device: torch.device,
    use_hazard: bool,
    hidden_size: int,
) -> Dict[str, float]:
    buffer.reset_ptr()
    obs = env.reset()
    hidden = torch.zeros(1, 1, hidden_size, device=device)
    episode_start = True
    episode_return = 0.0
    clipped_fracs = []
    bootstrap_obs = obs
    bootstrap_hidden = hidden.clone()
    last_terminated = False
    last_truncated = False

    for _ in range(buffer.rollout_steps):
        lidar = torch.tensor(obs["lidar"], dtype=torch.float32, device=device).view(1, 1, -1)
        spd = torch.tensor(obs["prev_speed"], dtype=torch.float32, device=device).view(1, 1, -1)
        with torch.no_grad():
            if use_hazard:
                haz = torch.tensor(obs["hazard"], dtype=torch.float32, device=device).view(1, 1, -1)
                dist, value, new_hidden = ac.forward(lidar, spd, haz, hidden)
            else:
                dist, value, new_hidden = ac.forward(lidar, spd, hidden)
            raw_action = dist.sample()
            logp = dist.log_prob(raw_action).sum(-1)
            val = value[:, -1].squeeze()

        raw_np = raw_action.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        next_obs, reward, done, info = env.step(raw_np)
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        episode_return += float(reward)
        clipped_fracs.append(float(info.get("action_was_clipped", False)))

        next_value_at_trunc = 0.0
        if truncated:
            with torch.no_grad():
                next_value_at_trunc = _forward_value(ac, next_obs, new_hidden, device, use_hazard)

        buffer.add(
            obs["lidar"],
            float(obs["prev_speed"][0]),
            obs.get("hazard") if use_hazard else None,
            raw_np,
            float(reward),
            float(val.item()),
            float(logp.squeeze().item()),
            bool(terminated),
            bool(truncated),
            float(next_value_at_trunc),
            bool(episode_start),
        )
        bootstrap_obs = next_obs
        bootstrap_hidden = new_hidden.detach()
        last_terminated = terminated
        last_truncated = truncated
        if done:
            obs = env.reset()
            hidden = torch.zeros(1, 1, hidden_size, device=device)
            episode_start = True
        else:
            obs = next_obs
            hidden = new_hidden.detach()
            episode_start = False

    with torch.no_grad():
        if last_terminated:
            last_value = 0.0
        else:
            last_value = _forward_value(ac, bootstrap_obs, bootstrap_hidden, device, use_hazard)
    buffer.compute_returns_and_advantage(last_value, last_terminated, last_truncated)
    return {
        "episode_return_partial": float(episode_return),
        "clipped_frac": float(np.mean(clipped_fracs)) if clipped_fracs else 0.0,
    }


def ppo_update(
    ac: torch.nn.Module,
    frozen_bc: End2Race,
    buffer: RolloutBuffer,
    optimizer: optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    use_hazard: bool,
    hidden_size: int,
) -> Dict[str, float]:
    losses_pi: list[float] = []
    losses_v: list[float] = []
    losses_ent: list[float] = []
    losses_bc: list[float] = []
    kls: list[float] = []
    early_stop = False
    kl_break_threshold = 1.5 * args.target_kl

    for _epoch in range(args.ppo_epochs):
        for batch in buffer.recurrent_minibatches(device, chunk_len=args.chunk_len, shuffle=True):
            lidar_b, spd_b, haz_b, act_b, old_logp_b, adv_b, ret_b, old_values_b, starts_b = batch
            adv_flat = adv_b.reshape(-1)
            adv_b = (adv_b - adv_flat.mean()) / (adv_flat.std(unbiased=False) + 1e-8)
            dist, values = forward_policy_sequence(ac, lidar_b, spd_b, haz_b, starts_b, hidden_size, use_hazard, device)
            new_logp = dist.log_prob(act_b).sum(-1)

            # Schulman k3 KL estimator; check BEFORE applying gradients
            with torch.no_grad():
                log_ratio = new_logp - old_logp_b
                approx_kl_batch = float(((log_ratio.exp() - 1) - log_ratio).mean().item())
            kls.append(approx_kl_batch)
            if approx_kl_batch > kl_break_threshold:
                early_stop = True
                break

            with torch.no_grad():
                bc_mean = forward_frozen_bc_sequence(frozen_bc, lidar_b, spd_b, starts_b, hidden_size, device)
            entropy = dist.entropy().sum(-1).mean()

            ratio = log_ratio.exp()
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            if args.clip_eps_vf > 0.0:
                v_clipped = old_values_b + torch.clamp(values - old_values_b, -args.clip_eps_vf, args.clip_eps_vf)
                value_loss = 0.5 * torch.max((values - ret_b).pow(2), (v_clipped - ret_b).pow(2)).mean()
            else:
                value_loss = 0.5 * (values - ret_b).pow(2).mean()

            bc_anchor = (dist.mean - bc_mean).pow(2).mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy + args.beta_bc * bc_anchor

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(), args.max_grad_norm)
            optimizer.step()

            losses_pi.append(float(policy_loss.item()))
            losses_v.append(float(value_loss.item()))
            losses_ent.append(float(entropy.item()))
            losses_bc.append(float(bc_anchor.item()))
        if early_stop:
            break
    return {
        "policy_loss": float(np.mean(losses_pi)) if losses_pi else 0.0,
        "value_loss": float(np.mean(losses_v)) if losses_v else 0.0,
        "entropy": float(np.mean(losses_ent)) if losses_ent else 0.0,
        "bc_anchor": float(np.mean(losses_bc)) if losses_bc else 0.0,
        "approx_kl": float(np.mean(kls)) if kls else 0.0,
    }


def main() -> None:
    args = parse_args()
    resolve_save_paths(args)
    if args.vec_backend != "serial":
        raise NotImplementedError("Only --vec_backend serial is implemented in this version.")
    device = torch.device(args.device)
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    use_hazard = args.mode == "safety_augmented"

    env = make_env(args)
    ac = build_actor_critic(args, device)
    ac.train()
    hidden_size = int(ac.actor.gru.hidden_size)

    frozen_bc = End2Race(mask_prob=0.0, hidden_scale=args.hidden_scale).to(device)
    load_end2race_actor(frozen_bc, args.bc_model_path, device)
    frozen_bc.eval()
    for param in frozen_bc.parameters():
        param.requires_grad = False

    def is_critic_param(name: str) -> bool:
        return any(key in name for key in ("value_head", "base_value_head", "delta_value"))

    critic_ps = [p for n, p in ac.named_parameters() if is_critic_param(n)]
    actor_ps = [p for n, p in ac.named_parameters() if not is_critic_param(n)]
    optimizer = optim.Adam([
        {"params": critic_ps, "lr": args.critic_lr},
        {"params": actor_ps, "lr": args.actor_lr},
    ])
    scheduler = None

    start_iter = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and ckpt.get("optimizer"):
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError as exc:
                print(f"Skipped optimizer state from {args.resume}: {exc}")
        if isinstance(ckpt, dict):
            start_iter = int(ckpt.get("iteration", 0))

    buffer = RolloutBuffer(args.rollout_steps, gamma=args.gamma, gae_lambda=args.gae_lambda, hazard_dim=7 if use_hazard else 0)
    os.makedirs(os.path.dirname(args.save_full_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.save_actor_path) or ".", exist_ok=True)

    actor_group_idx = 1  # matches order in optim.Adam([critic_ps, actor_ps]) above
    last_metrics: Dict[str, Any] = {}
    for it in range(start_iter, args.total_iterations):
        # Critic warmup: hold actor frozen while the freshly-initialized value
        # head learns reasonable estimates, otherwise GAE noise destabilizes policy.
        optimizer.param_groups[actor_group_idx]["lr"] = (
            0.0 if it < args.critic_warmup_iters else args.actor_lr
        )
        metrics_roll = collect_rollout(env, ac, buffer, device, use_hazard, hidden_size)
        metrics_opt = ppo_update(ac, frozen_bc, buffer, optimizer, device, args, use_hazard, hidden_size)
        last_metrics = {**metrics_roll, **metrics_opt}
        if it % 10 == 0:
            print(
                f"iter {it:05d}  ret {metrics_roll['episode_return_partial']:.2f}  "
                f"clip {metrics_roll['clipped_frac']:.3f}  pi {metrics_opt['policy_loss']:.4f}  "
                f"v {metrics_opt['value_loss']:.4f}  kl {metrics_opt['approx_kl']:.4f}  "
                f"bc {metrics_opt['bc_anchor']:.4f}"
            )
        if (it + 1) % args.save_every == 0:
            ac.save_full_checkpoint(
                args.save_full_path,
                optimizer,
                scheduler,
                vars(args).copy(),
                asdict(env.reward_weights),
                args.stage,
                it + 1,
                last_metrics,
            )
            print(f"Saved full checkpoint to {args.save_full_path}")

    ac.eval()
    ac.save_full_checkpoint(
        args.save_full_path,
        optimizer,
        scheduler,
        vars(args).copy(),
        asdict(env.reward_weights),
        args.stage,
        args.total_iterations,
        last_metrics,
    )
    print(f"Saved final full checkpoint to {args.save_full_path}")
    if args.save_actor_without_gate and not use_hazard:
        ac.save_actor_backbone(args.save_actor_path)
        print(f"Saved actor backbone to {args.save_actor_path} without gate because --save_actor_without_gate was set")
    elif use_hazard:
        print("Skipped actor-only save in safety_augmented mode; use the full v2 checkpoint.")
    else:
        print("Skipped actor-only save because the deployment gate is external. Run eval_ppo.py, or use --save_actor_without_gate for smoke tests.")
    env.close()


if __name__ == "__main__":
    main()
