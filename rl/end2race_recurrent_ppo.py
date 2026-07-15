"""Minimal PPO V1 subclass preserving named optimizer-group learning rates."""

from __future__ import annotations

from typing import Union

from sb3_contrib import RecurrentPPO
from torch.optim import Optimizer


class End2RaceRecurrentPPO(RecurrentPPO):
    lr_scale = 1.0

    def _update_learning_rate(self, optimizers: Union[list[Optimizer], Optimizer]) -> None:
        for optimizer in optimizers if isinstance(optimizers, list) else [optimizers]:
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * self.lr_scale
