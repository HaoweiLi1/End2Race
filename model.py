import torch
import torch.nn as nn
from typing import Optional, Tuple

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
        
        # Forward pass through GRU and output layer
        gru_out, last_hidden = self.gru(features, hidden)
        actions = self.output_layer(gru_out)

        return actions, last_hidden


class End2RaceDualHead(nn.Module):
    """Ablation variant: End2Race with the output_layer split into two heads.

    Single change vs End2Race:
      - baseline `output_layer = Linear(H, P) → ReLU → Linear(P, 2)`
        becomes two structurally-identical heads, each `Linear(H, P) → ReLU
        → Linear(P, 1)` — one for steer, one for speed.
      Same hidden_scale=4, same num_layers=1, same preprocessing, same ReLU,
      same bottleneck width P. Only the output wiring changes.
    """

    def __init__(self, mask_prob=0.0, hidden_scale=4):
        super().__init__()
        num_features = 360
        self.mask_prob = mask_prob
        self.hidden_scale = hidden_scale

        k_init = (-1 / 10.0) * torch.log(torch.tensor(0.01) / (2 - torch.tensor(0.01)))
        self.k = nn.Parameter(torch.full((num_features,), k_init.item()))

        self.speed_mlp = nn.Sequential(
            nn.Linear(1, num_features // 6),
            nn.ReLU(),
        )
        self.dummy_embedding = nn.Parameter(torch.randn(1, num_features // 6))

        processed_features = num_features + num_features // 6
        hidden = processed_features * hidden_scale

        self.gru = nn.GRU(
            input_size=processed_features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # Split heads: same structure as baseline output_layer, but two of them
        # each producing a single scalar instead of one MLP producing [steer, speed].
        self.head_steer = nn.Sequential(
            nn.Linear(hidden, processed_features),
            nn.ReLU(),
            nn.Linear(processed_features, 1),
        )
        self.head_speed = nn.Sequential(
            nn.Linear(hidden, processed_features),
            nn.ReLU(),
            nn.Linear(processed_features, 1),
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
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
        nn.init.xavier_normal_(self.dummy_embedding)

    def forward(self, x: torch.Tensor, speed_input: Optional[torch.Tensor] = None,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        processed_lidar = (-1 / (1 + torch.exp(-self.k * x)) + 1) * 2

        batch_size, seq_len, _ = x.shape
        speed_embedding = self.speed_mlp(speed_input)
        if self.training and self.mask_prob > 0:
            mask = torch.rand(batch_size, seq_len, 1, device=speed_input.device) < self.mask_prob
            mask_batch = self.dummy_embedding.expand(batch_size, seq_len, -1)
            speed_embedding = torch.where(mask, mask_batch, speed_embedding)

        features = torch.cat([processed_lidar, speed_embedding], dim=2)
        gru_out, last_hidden = self.gru(features, hidden)

        steer = self.head_steer(gru_out)
        speed = self.head_speed(gru_out)
        actions = torch.cat([steer, speed], dim=-1)

        return actions, last_hidden


class End2RaceDeep(nn.Module):
    """Ablation variant: 1-layer GRU → 2 stacked separate GRUs with a narrower second layer.

    Single change vs End2Race:
      - `self.gru` (1 layer, hidden_scale=4) is replaced by two separate GRUs:
        * `gru1`: hidden = processed_features * hidden_scale  (from CLI, default 4 → 1680)
        * `gru2`: hidden = processed_features * 2             (fixed → 840)
      Same preprocessing, speed MLP, concat, mask_prob, and shared output_layer.

    Narrowing layer 2 (hidden_scale=2, not 4) keeps the second layer cheaper
    and encourages compression into a coarser temporal representation: layer 1
    carries per-frame detail, layer 2 distills multi-second strategy.

    `forward` returns `hidden` as a tuple `(h1, h2)` because the two GRUs have
    different hidden sizes. `train.py` discards hidden, so training is unaffected;
    online inference (eval scripts) must pass/receive the tuple.
    """

    SECOND_LAYER_SCALE = 2  # fixed scale for gru2, independent of --hidden_scale

    def __init__(self, mask_prob=0.0, hidden_scale=4):
        super().__init__()
        num_features = 360
        self.mask_prob = mask_prob
        self.hidden_scale = hidden_scale

        k_init = (-1 / 10.0) * torch.log(torch.tensor(0.01) / (2 - torch.tensor(0.01)))
        self.k = nn.Parameter(torch.full((num_features,), k_init.item()))

        self.speed_mlp = nn.Sequential(
            nn.Linear(1, num_features // 6),
            nn.ReLU(),
        )
        self.dummy_embedding = nn.Parameter(torch.randn(1, num_features // 6))

        processed_features = num_features + num_features // 6
        h1 = processed_features * hidden_scale
        h2 = processed_features * self.SECOND_LAYER_SCALE

        self.gru1 = nn.GRU(processed_features, h1, num_layers=1, batch_first=True, bidirectional=False)
        self.gru2 = nn.GRU(h1,                  h2, num_layers=1, batch_first=True, bidirectional=False)

        # Identical to baseline End2Race output_layer — single shared head.
        # Input dim is h2 (the narrower second layer), output layer hidden is processed_features.
        self.output_layer = nn.Sequential(
            nn.Linear(h2, processed_features),
            nn.ReLU(),
            nn.Linear(processed_features, 2),
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
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
        nn.init.xavier_normal_(self.dummy_embedding)

    def forward(self, x: torch.Tensor, speed_input: Optional[torch.Tensor] = None,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        processed_lidar = (-1 / (1 + torch.exp(-self.k * x)) + 1) * 2

        batch_size, seq_len, _ = x.shape
        speed_embedding = self.speed_mlp(speed_input)
        if self.training and self.mask_prob > 0:
            mask = torch.rand(batch_size, seq_len, 1, device=speed_input.device) < self.mask_prob
            mask_batch = self.dummy_embedding.expand(batch_size, seq_len, -1)
            speed_embedding = torch.where(mask, mask_batch, speed_embedding)

        features = torch.cat([processed_lidar, speed_embedding], dim=2)

        h1_prev, h2_prev = (None, None) if hidden is None else hidden
        out1, h1 = self.gru1(features, h1_prev)
        out2, h2 = self.gru2(out1,     h2_prev)
        actions = self.output_layer(out2)

        return actions, (h1, h2)


class End2RaceDeepDualHead(nn.Module):
    """Ablation variant combining both changes: Deep (2 stacked GRUs) + DualHead (split heads).

    Two changes vs End2Race:
      - GRU structure same as End2RaceDeep (gru1 at hidden_scale, gru2 fixed at scale=2).
      - output_layer split into head_steer / head_speed, same structure as End2RaceDualHead.

    Not a single-variable ablation — use this to see if the two improvements
    compound (vs expected gain = Deep_gain + DualHead_gain if independent, or
    more/less if they interact).
    """

    SECOND_LAYER_SCALE = 2

    def __init__(self, mask_prob=0.0, hidden_scale=4):
        super().__init__()
        num_features = 360
        self.mask_prob = mask_prob
        self.hidden_scale = hidden_scale

        k_init = (-1 / 10.0) * torch.log(torch.tensor(0.01) / (2 - torch.tensor(0.01)))
        self.k = nn.Parameter(torch.full((num_features,), k_init.item()))

        self.speed_mlp = nn.Sequential(
            nn.Linear(1, num_features // 6),
            nn.ReLU(),
        )
        self.dummy_embedding = nn.Parameter(torch.randn(1, num_features // 6))

        processed_features = num_features + num_features // 6
        h1 = processed_features * hidden_scale
        h2 = processed_features * self.SECOND_LAYER_SCALE

        self.gru1 = nn.GRU(processed_features, h1, num_layers=1, batch_first=True, bidirectional=False)
        self.gru2 = nn.GRU(h1,                  h2, num_layers=1, batch_first=True, bidirectional=False)

        self.head_steer = nn.Sequential(
            nn.Linear(h2, processed_features),
            nn.ReLU(),
            nn.Linear(processed_features, 1),
        )
        self.head_speed = nn.Sequential(
            nn.Linear(h2, processed_features),
            nn.ReLU(),
            nn.Linear(processed_features, 1),
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
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
        nn.init.xavier_normal_(self.dummy_embedding)

    def forward(self, x: torch.Tensor, speed_input: Optional[torch.Tensor] = None,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        processed_lidar = (-1 / (1 + torch.exp(-self.k * x)) + 1) * 2

        batch_size, seq_len, _ = x.shape
        speed_embedding = self.speed_mlp(speed_input)
        if self.training and self.mask_prob > 0:
            mask = torch.rand(batch_size, seq_len, 1, device=speed_input.device) < self.mask_prob
            mask_batch = self.dummy_embedding.expand(batch_size, seq_len, -1)
            speed_embedding = torch.where(mask, mask_batch, speed_embedding)

        features = torch.cat([processed_lidar, speed_embedding], dim=2)

        h1_prev, h2_prev = (None, None) if hidden is None else hidden
        out1, h1 = self.gru1(features, h1_prev)
        out2, h2 = self.gru2(out1,     h2_prev)

        steer = self.head_steer(out2)
        speed = self.head_speed(out2)
        actions = torch.cat([steer, speed], dim=-1)

        return actions, (h1, h2)