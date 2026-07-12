"""Single locked beam-local spatiotemporal geometry encoder for D2R-G."""

from __future__ import annotations

import torch
import torch.nn as nn

from d2r import HISTORY_OFFSETS, LIDAR_BEAMS, TTC_BIN_COUNT, TTC_BIN_WIDTH_S


class D2RGeometryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.beam_encoder = nn.Sequential(
            nn.Conv1d(len(HISTORY_OFFSETS), 32, kernel_size=9, padding=4, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=7, padding=3, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2, padding_mode="circular"),
            nn.SiLU(),
        )
        self.beam_pool = nn.AdaptiveAvgPool1d(18)
        self.bc_projection = nn.Sequential(
            nn.Linear(1680, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.scalar_projection = nn.Sequential(nn.Linear(24, 32), nn.SiLU())
        self.fusion = nn.Sequential(
            nn.Linear(576 + 128 + 32, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.collision_head = nn.Linear(128, 6)
        self.geometry_head = nn.Linear(128, 3)
        self.ttc_head = nn.Linear(128, TTC_BIN_COUNT)

    def encode_beams(self, lidar: torch.Tensor, pool: bool = True) -> torch.Tensor:
        if lidar.ndim != 3 or lidar.shape[1:] != (len(HISTORY_OFFSETS), LIDAR_BEAMS):
            raise ValueError("D2R LiDAR tensor must be [B,8,360]")
        encoded = self.beam_encoder(lidar)
        return self.beam_pool(encoded) if pool else encoded

    def forward(self, lidar: torch.Tensor, bc_feature: torch.Tensor, scalar_history: torch.Tensor):
        if bc_feature.ndim != 2 or bc_feature.shape[1] != 1680:
            raise ValueError("D2R BC feature tensor must be [B,1680]")
        if scalar_history.ndim != 2 or scalar_history.shape[1] != 24:
            raise ValueError("D2R scalar history tensor must be [B,24]")
        if len(lidar) != len(bc_feature) or len(lidar) != len(scalar_history):
            raise ValueError("D2R input batch sizes differ")
        beam = self.encode_beams(lidar, pool=True).flatten(1)
        fused = self.fusion(
            torch.cat(
                [beam, self.bc_projection(bc_feature), self.scalar_projection(scalar_history)],
                dim=1,
            )
        )
        geometry_raw = self.geometry_head(fused)
        return {
            "collision_logits": self.collision_head(fused),
            "rel_s": torch.tanh(geometry_raw[:, 0]) * 10.0,
            "lateral_gap": torch.sigmoid(geometry_raw[:, 1]) * 2.0,
            "closing_rate": torch.tanh(geometry_raw[:, 2]) * 5.0,
            "ttc_logits": self.ttc_head(fused),
        }


def decode_ttc_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != TTC_BIN_COUNT:
        raise ValueError("D2R TTC logits must be [B,50]")
    centers = (
        torch.arange(TTC_BIN_COUNT, dtype=logits.dtype, device=logits.device) + 0.5
    ) * TTC_BIN_WIDTH_S
    return torch.sum(torch.softmax(logits, dim=1) * centers[None, :], dim=1)


@torch.no_grad()
def initialize_classification_bias(model: D2RGeometryNet, prevalence) -> None:
    value = torch.as_tensor(
        prevalence,
        dtype=model.collision_head.bias.dtype,
        device=model.collision_head.bias.device,
    )
    if value.shape != (6,) or torch.any(value <= 0.0) or torch.any(value >= 1.0):
        raise ValueError("D2R classification prevalence must be six values in (0,1)")
    model.collision_head.bias.copy_(torch.log(value) - torch.log1p(-value))
