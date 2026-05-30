import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class End2Race(nn.Module):

    def __init__(self, mask_prob=0.0, hidden_scale=4):
        super(End2Race, self).__init__()
        
        # Store configuration
        num_features = 360
        num_actions=2
        self.mask_prob = mask_prob
        self.hidden_scale = hidden_scale
        
        # Common: Learnable sensor preprocessing parameter
        k_init = (-1 / 10.0) * torch.log(torch.tensor(0.01) / (2 - torch.tensor(0.01)))
        self.k = nn.Parameter(torch.full((num_features,), k_init.item()))
        
        # Speed-specific modules (only created if needed)
        self.speed_mlp = nn.Sequential(
            nn.Linear(1, num_features // 6),
            nn.ReLU()
        )
        self.dummy_embedding = nn.Parameter(torch.randn(1, num_features // 6))
        
        # Calculate processed feature size
        processed_features = num_features + num_features // 6
        
        # Common GRU architecture with mode-dependent dimensions
        self.gru = nn.GRU(
            input_size=processed_features,
            hidden_size=processed_features * hidden_scale,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Common output layer with mode-dependent dimensions
        self.output_layer = nn.Sequential(
            nn.Linear(processed_features * hidden_scale, processed_features),
            nn.ReLU(),
            nn.Linear(processed_features, num_actions)
        )
        
        # Initialize all parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize all parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        
        # Initialize dummy embedding
        nn.init.xavier_normal_(self.dummy_embedding)
    
    def forward_features(self, x: torch.Tensor, speed_input: Optional[torch.Tensor] = None,
                         hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return GRU features shared by BC inference and PPO actor-critic."""
        # Process LiDAR with learnable sigmoid transformation
        processed_lidar = (-1 / (1 + torch.exp(-self.k * x)) + 1) * 2
        
        # Process speed input
        batch_size, seq_len, _ = x.shape
        speed_embedding = self.speed_mlp(speed_input)
        
        # Apply dummy embedding during training
        if self.training and self.mask_prob > 0:
            mask = torch.rand(batch_size, seq_len, 1, device=speed_input.device) < self.mask_prob
            mask_batch = self.dummy_embedding.expand(batch_size, seq_len, -1)
            speed_embedding = torch.where(mask, mask_batch, speed_embedding)
        
        # Concatenate features
        features = torch.cat([processed_lidar, speed_embedding], dim=2)
        
        # Forward pass through GRU
        gru_out, last_hidden = self.gru(features, hidden)
        return gru_out, last_hidden

    def forward(self, x: torch.Tensor, speed_input: Optional[torch.Tensor] = None,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with speed conditioning.

        Args:
            x: LiDAR input tensor [batch, seq_len, num_features]
            speed_input: Previous speed tensor [batch, seq_len, 1]
            hidden: Hidden state from previous timestep

        Returns:
            actions: Predicted actions [steering, speed]
            last_hidden: Updated hidden state
        """
        gru_out, last_hidden = self.forward_features(x, speed_input, hidden)
        actions = self.output_layer(gru_out)

        return actions, last_hidden


class End2RaceActorCritic(nn.Module):
    """PPO actor-critic wrapping the plain End2Race BC model."""

    def __init__(self, hidden_scale: int = 4,
                 steer_std: float = 0.03, speed_std: float = 0.25):
        super().__init__()

        self.actor = End2Race(mask_prob=0.0, hidden_scale=hidden_scale)
        hidden_size = self.actor.gru.hidden_size

        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )
        self.log_std = nn.Parameter(
            torch.tensor([math.log(steer_std), math.log(speed_std)], dtype=torch.float32)
        )

    def forward_features(
        self,
        lidar: torch.Tensor,
        speed_input: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor.forward_features(lidar, speed_input, hidden)

    def forward(
        self,
        lidar: torch.Tensor,
        speed_input: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.distributions.Normal, torch.Tensor, torch.Tensor]:
        """Return policy distribution, value estimate, and updated hidden state."""
        gru_out, last_hidden = self.forward_features(lidar, speed_input, hidden)

        mean = self.actor.output_layer(gru_out)
        std = self.log_std.exp().view(1, 1, -1).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        value = self.value_head(gru_out).squeeze(-1)

        return dist, value, last_hidden

    def act(
        self,
        lidar: torch.Tensor,
        speed_input: torch.Tensor,
        hidden: Optional[torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action, log probability, value, and updated hidden state."""
        dist, value, new_hidden = self.forward(lidar, speed_input, hidden)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value, new_hidden
