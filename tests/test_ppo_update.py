import argparse

import numpy as np
import torch
import torch.optim as optim

from model import End2Race
from model_ppo import End2RaceActorCritic
from train_ppo import RolloutBuffer, ppo_update, validate_replay_identity


def make_args():
    return argparse.Namespace(
        ppo_epochs=1,
        clip_eps=0.2,
        max_speed=20.0,
        vf_coef=0.5,
        ent_coef=0.0,
        beta_bc=0.0,
        bound_coef=0.0,
        max_grad_norm=0.5,
        target_kl=10.0,
    )


def fill_buffer(ac, buffer):
    device = torch.device("cpu")
    hidden = torch.zeros(1, 1, ac.actor.gru.hidden_size, device=device)
    rng = np.random.default_rng(7)
    buffer.reset()
    for t in range(buffer.rollout_steps):
        lidar = rng.random(360, dtype=np.float32)
        prev_speed = np.array([float(t + 1)], dtype=np.float32)
        lidar_t = torch.tensor(lidar, dtype=torch.float32, device=device).view(1, 1, -1)
        speed_t = torch.tensor(prev_speed, dtype=torch.float32, device=device).view(1, 1, -1)
        with torch.no_grad():
            action, logp, value, hidden = ac.act(lidar_t, speed_t, hidden, deterministic=False)
        buffer.add(
            lidar=lidar,
            prev_speed=prev_speed,
            raw_action=action.view(-1).cpu().numpy(),
            reward=float(t + 1),
            value=float(value.item()),
            log_prob=float(logp.item()),
            terminated=False,
            truncated=False,
            trunc_next_value=0.0,
            episode_start=(t == 0),
        )
        hidden = hidden.detach()
    buffer.compute_returns_and_advantage(0.0, False, False)


def test_ppo_update_changes_params_and_reports_finite_metrics():
    torch.manual_seed(8)
    device = torch.device("cpu")
    ac = End2RaceActorCritic(hidden_scale=1).to(device)
    frozen_bc = End2Race(mask_prob=0.0, hidden_scale=1).to(device)
    frozen_bc.load_state_dict(ac.actor.state_dict())
    frozen_bc.eval()
    for p in frozen_bc.parameters():
        p.requires_grad = False

    buffer = RolloutBuffer(5)
    fill_buffer(ac, buffer)
    validate_replay_identity(ac, buffer, device)

    critic_params = list(ac.value_head.parameters())
    critic_ids = {id(p) for p in critic_params}
    actor_params = [p for p in ac.parameters() if id(p) not in critic_ids]
    optimizer = optim.Adam(
        [
            {"params": critic_params, "lr": 1e-3},
            {"params": actor_params, "lr": 1e-3},
        ]
    )

    opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert opt_ids == {id(p) for p in ac.parameters()}

    before = {name: p.detach().clone() for name, p in ac.named_parameters()}
    metrics = ppo_update(ac, frozen_bc, buffer, optimizer, device, make_args())

    assert metrics["num_updates"] == 1.0
    assert metrics["early_stopped"] == 0.0
    for key in (
        "policy_loss",
        "value_loss",
        "entropy",
        "bc_anchor",
        "approx_kl",
        "post_step_approx_kl",
        "clip_fraction",
        "ratio_mean",
        "ratio_min",
        "ratio_max",
        "grad_norm",
    ):
        assert np.isfinite(metrics[key])

    assert ac.value_head[0].weight.grad is not None
    assert ac.log_std.grad is not None
    assert all(p.grad is None for p in frozen_bc.parameters())
    assert any(not torch.allclose(before[name], p) for name, p in ac.named_parameters())
