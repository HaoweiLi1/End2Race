import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

# Privileged critic input size (see End2RacePPOEnv._priv_state in train_ppo.py).
PRIV_DIM = 12

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
        """
        Forward pass with speed conditioning.
        
        Args:
            x: LiDAR input tensor [batch, seq_len, num_features]
            speed_input: Previous speed tensor [batch, seq_len, 1]
            hidden: Hidden state from previous timestep
            
        Returns:
            gru_out: GRU output features
            last_hidden: Updated hidden state
        """
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
        Original BC-compatible forward.

        returns:
            actions:      [batch, seq_len, 2]
            last_hidden:  [1, batch, hidden_size]
        """
        gru_out, last_hidden = self.forward_features(x, speed_input, hidden)
        actions = self.output_layer(gru_out)

        return actions, last_hidden

class End2RaceResidual(End2Race):
    """D2 residual policy: frozen BC backbone + bounded asymmetric residual head.

    steer = BC_steer + tanh(r_steer) * steer_budget
    speed = BC_speed + tanh(r_speed) * (speed_down_budget if r_speed < 0 else speed_up_budget)

    The residual head reads the (frozen) GRU features. Its final layer is
    zero-initialized so the t=0 deterministic policy is exactly the BC policy.
    Budgets are registered buffers so a deployment checkpoint is a single
    self-describing state_dict; forward() keeps the End2Race interface so the
    original evaluators work unchanged.
    """

    def __init__(self, hidden_scale=4, steer_budget=0.2,
                 speed_up_budget=0.2, speed_down_budget=1.0):
        super(End2RaceResidual, self).__init__(mask_prob=0.0, hidden_scale=hidden_scale)
        gru_hidden = self.gru.hidden_size
        self.res_head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        # [steer_budget, speed_up_budget, speed_down_budget], action units.
        self.register_buffer(
            "residual_budgets",
            torch.tensor([steer_budget, speed_up_budget, speed_down_budget], dtype=torch.float32),
        )
        nn.init.xavier_uniform_(self.res_head[0].weight)
        nn.init.zeros_(self.res_head[0].bias)
        # Zero-init the last layer: zero residual output at initialization.
        nn.init.zeros_(self.res_head[2].weight)
        nn.init.zeros_(self.res_head[2].bias)

    def residual_delta(self, r):
        """Map pre-tanh residuals to bounded action-space deltas [d_steer, d_speed]."""
        t = torch.tanh(r)
        d_steer = t[..., 0:1] * self.residual_budgets[0]
        t_speed = t[..., 1:2]
        d_speed = torch.where(
            t_speed < 0.0,
            t_speed * self.residual_budgets[2],
            t_speed * self.residual_budgets[1],
        )
        return torch.cat([d_steer, d_speed], dim=-1)

    def compose(self, base, r):
        """Compose BC base actions with pre-tanh residuals; speed never below 0."""
        out = base + self.residual_delta(r)
        steer = out[..., 0:1]
        speed = torch.clamp(out[..., 1:2], min=0.0)
        return torch.cat([steer, speed], dim=-1)

    def forward(self, x: torch.Tensor, speed_input: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deployment forward: BC base + deterministic residual mean, End2Race interface."""
        gru_out, last_hidden = self.forward_features(x, speed_input, hidden)
        base = self.output_layer(gru_out)
        r = self.res_head(gru_out)
        return self.compose(base, r), last_hidden

class End2Race_PPO(nn.Module):

    def __init__(self, hidden_scale=4, steer_std=0.03, speed_std=0.25,
                 residual_mode=False, residual_steer_budget=0.2,
                 residual_speed_up_budget=0.2, residual_speed_down_budget=1.0):
        super(End2Race_PPO, self).__init__()

        # Store configuration
        self.hidden_scale = hidden_scale
        self.steer_std = steer_std
        self.speed_std = speed_std
        self.residual_mode = bool(residual_mode)

        # Common actor backbone. In residual mode the actor carries the frozen
        # BC backbone plus the trainable residual head, and log_std lives in
        # residual pre-tanh space (steer_std/speed_std are then r-space stds).
        if self.residual_mode:
            self.actor = End2RaceResidual(
                hidden_scale=hidden_scale,
                steer_budget=residual_steer_budget,
                speed_up_budget=residual_speed_up_budget,
                speed_down_budget=residual_speed_down_budget,
            )
        else:
            self.actor = End2Race(mask_prob=0.0, hidden_scale=hidden_scale)

        # Privileged critic: trained on simulator-state features, discarded at
        # deployment. Fully separate from the actor so value gradients never
        # touch the BC-pretrained backbone.
        self.critic = nn.Sequential(
            nn.Linear(PRIV_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Trainable policy standard deviation
        self.log_std = nn.Parameter(torch.tensor([math.log(steer_std), math.log(speed_std)], dtype=torch.float32))

        # Initialize PPO-specific parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize PPO-specific parameters."""
        for module in self.critic.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, speed_input: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        """
        PPO actor forward. Values come from the privileged critic instead:
        call self.critic(priv) with the simulator-state features.

        In residual mode the distribution lives in residual pre-tanh space:
        the sampled r maps deterministically to actions via actor.compose(),
        which is treated as part of the environment interface. The tanh
        Jacobian depends only on the stored sample, so it cancels exactly in
        the PPO ratio; no change-of-variables terms are needed.

        returns:
            dist:         policy distribution over [steering, speed]
                          (residual mode: over pre-tanh residuals [r_steer, r_speed])
            last_hidden:  [1, batch, hidden_size]
        """
        gru_out, last_hidden = self.actor.forward_features(x, speed_input, hidden)

        if self.residual_mode:
            mean = self.actor.res_head(gru_out)
        else:
            mean = self.actor.output_layer(gru_out)
        std = self.log_std.exp().view(1, 1, -1).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        return dist, last_hidden

    def forward_residual_rollout(self, x: torch.Tensor, speed_input: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None):
        """Residual-mode rollout forward: residual distribution plus BC base actions.

        returns:
            dist:         Normal over pre-tanh residuals [r_steer, r_speed]
            base:         frozen BC base actions [batch, seq_len, 2]
            last_hidden:  [1, batch, hidden_size]
        """
        if not self.residual_mode:
            raise RuntimeError("forward_residual_rollout requires residual_mode=True.")
        gru_out, last_hidden = self.actor.forward_features(x, speed_input, hidden)
        base = self.actor.output_layer(gru_out)
        r_mean = self.actor.res_head(gru_out)
        std = self.log_std.exp().view(1, 1, -1).expand_as(r_mean)
        dist = torch.distributions.Normal(r_mean, std)
        return dist, base, last_hidden

    def act(self, x: torch.Tensor, speed_input: Optional[torch.Tensor] = None, hidden: Optional[torch.Tensor] = None,
            deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PPO action helper.

        returns:
            action:      sampled or deterministic action [batch, seq_len, 2]
            logp:        action log probability [batch, seq_len]
            last_hidden: [1, batch, hidden_size]
        """
        dist, last_hidden = self.forward(x, speed_input, hidden)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)

        return action, logp, last_hidden
