import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from model import End2Race


class End2RaceActorCritic(nn.Module):
    """PPO actor-critic wrapping the plain End2Race BC model.

    The actor output is reinterpreted as the mean of a Gaussian policy.
    A separate value head produces V(s) from the same GRU features.
    """

    def __init__(self, hidden_scale: int = 4,
                 steer_std: float = 0.03, speed_std: float = 0.25):
        super().__init__()

        # Actor is the original End2Race model (deterministic mean)
        self.actor = End2Race(mask_prob=0.0, hidden_scale=hidden_scale)

        h = self.actor.gru.hidden_size  # 420 * hidden_scale = 1680 at scale 4

        # Value head: GRU features -> scalar value
        self.value_head = nn.Sequential(
            nn.Linear(h, h // 4),
            nn.ReLU(),
            nn.Linear(h // 4, 1),
        )

        # Learnable log-std for Gaussian policy [steer, speed]
        self.log_std = nn.Parameter(
            torch.tensor([math.log(steer_std), math.log(speed_std)], dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    # Feature extraction (mirrors model.py lines 81-97, minus output_layer)
    # ------------------------------------------------------------------
    def forward_features(
        self,
        lidar: torch.Tensor,
        speed_input: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replicate the exact feature path of End2Race.forward,
        stopping before the output layer."""
        # Learnable sigmoid pressure transform
        processed_lidar = (-1.0 / (1.0 + torch.exp(-self.actor.k * lidar)) + 1.0) * 2.0

        # Speed embedding
        speed_embedding = self.actor.speed_mlp(speed_input)

        # Concatenate
        features = torch.cat([processed_lidar, speed_embedding], dim=2)

        # GRU
        gru_out, last_hidden = self.actor.gru(features, hidden)
        return gru_out, last_hidden

    # ------------------------------------------------------------------
    # Full forward: dist + value + hidden
    # ------------------------------------------------------------------
    def forward(
        self,
        lidar: torch.Tensor,
        speed_input: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        """Returns (dist, value, last_hidden)."""
        gru_out, last_hidden = self.forward_features(lidar, speed_input, hidden)

        # Actor: mean from original output layer
        mean = self.actor.output_layer(gru_out)

        # Gaussian std from learnable log_std
        std = self.log_std.exp().view(1, 1, -1).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        # Critic: scalar value
        value = self.value_head(gru_out).squeeze(-1)  # [B, T]

        return dist, value, last_hidden

    # ------------------------------------------------------------------
    # act() — used during rollout collection (call under torch.no_grad())
    # ------------------------------------------------------------------
    def act(
        self,
        lidar: torch.Tensor,
        speed_input: torch.Tensor,
        hidden: Optional[torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, logp, value, new_hidden)."""
        dist, value, new_hidden = self.forward(lidar, speed_input, hidden)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)  # sum over action dims
        return action, logp, value, new_hidden

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def save_actor_backbone(self, path: str) -> None:
        """Save plain End2Race state_dict — loadable by original evaluators."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def save_full_checkpoint(
        self,
        path: str,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        config: dict,
    ) -> None:
        """Save full PPO training checkpoint."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "actor_critic": self.state_dict(),
                "actor": self.actor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                "config": config,
                "hidden_scale": self.actor.hidden_scale,
                "log_std": self.log_std.data.clone(),
            },
            path,
        )


# ----------------------------------------------------------------------
# Standalone loaders
# ----------------------------------------------------------------------

def load_actor_critic(
    ac: End2RaceActorCritic,
    path: str,
    device: torch.device,
) -> dict:
    """Load weights into *ac* from one of three checkpoint formats.

    1. Full PPO checkpoint  (has ``"actor_critic"`` key)  → load into *ac*
    2. Actor-only / BC dict (has ``"actor"`` key)         → load into *ac.actor*
    3. Plain End2Race state_dict                          → load into *ac.actor*

    Returns the raw checkpoint dict (or an empty dict for format 3).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "actor_critic" in ckpt:
        ac.load_state_dict(ckpt["actor_critic"])
        return ckpt

    if isinstance(ckpt, dict) and "actor" in ckpt:
        ac.actor.load_state_dict(ckpt["actor"])
        return ckpt

    # Plain state_dict (e.g. end2race.pth from BC training)
    ac.actor.load_state_dict(ckpt)
    return {}


def load_frozen_bc(
    path: str,
    device: torch.device,
    hidden_scale: int = 4,
) -> End2Race:
    """Load a frozen BC model for anchor computation."""
    bc = End2Race(mask_prob=0.0, hidden_scale=hidden_scale).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "actor" in ckpt:
        bc.load_state_dict(ckpt["actor"])
    else:
        bc.load_state_dict(ckpt)
    bc.eval()
    for p in bc.parameters():
        p.requires_grad = False
    return bc
