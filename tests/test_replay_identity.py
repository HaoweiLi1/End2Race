import numpy as np
import torch

from model_ppo import End2RaceActorCritic
from train_ppo import RolloutBuffer, validate_replay_identity


def test_validate_replay_identity_matches_collected_log_probs():
    torch.manual_seed(3)
    device = torch.device("cpu")
    ac = End2RaceActorCritic(hidden_scale=1).to(device)
    buffer = RolloutBuffer(5)
    buffer.reset()

    hidden = torch.zeros(1, 1, ac.actor.gru.hidden_size, device=device)
    starts = [True, False, True, False, False]
    for t, episode_start in enumerate(starts):
        if episode_start:
            hidden = torch.zeros_like(hidden)
        lidar = np.random.default_rng(t).random(360, dtype=np.float32)
        prev_speed = np.array([0.5 + t], dtype=np.float32)
        lidar_t = torch.tensor(lidar, device=device).view(1, 1, -1)
        speed_t = torch.tensor(prev_speed, device=device).view(1, 1, -1)
        with torch.no_grad():
            action, logp, value, hidden = ac.act(lidar_t, speed_t, hidden, deterministic=False)
        buffer.add(
            lidar=lidar,
            prev_speed=prev_speed,
            raw_action=action.view(-1).cpu().numpy(),
            reward=0.1 * t,
            value=float(value.item()),
            log_prob=float(logp.item()),
            terminated=False,
            truncated=False,
            trunc_next_value=0.0,
            episode_start=episode_start,
        )
        hidden = hidden.detach()

    buffer.compute_returns_and_advantage(0.0, False, False)
    metrics = validate_replay_identity(ac, buffer, device, atol=1e-5)
    assert metrics["max_replay_logp_error"] < 1e-5
    assert abs(metrics["replay_ratio_mean"] - 1.0) < 1e-5
