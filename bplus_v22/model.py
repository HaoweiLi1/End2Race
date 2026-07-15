"""Three-arm policy representation and exact hurdle residual distribution."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bplus_v22 import (
    ACTION_CORE_LR,
    ARM_BC_FROZEN,
    ARM_SIDECAR_FINETUNE,
    ARM_SIDECAR_FROZEN,
    BC_FEATURE_DIM,
    BRAKE_BUDGET,
    INITIAL_BRAKE_LOGIT,
    INITIAL_BRAKE_STD,
    INITIAL_STEER_STD,
    POLICY_FEATURE_DIM,
    POSITIVE_SPEED_BUDGET,
    PRIVILEGED_FEATURE_DIM,
    SCALAR_HISTORY_DIM,
    SEED,
    SIDECAR_FINETUNE_LR,
    STEER_BUDGET,
    validate_arm,
)
from d2r.model import D2RGeometryNet, decode_ttc_logits
from model import End2Race


@dataclass(frozen=True)
class MacroResidualAction:
    """Stored latent action: steering latent, brake atom, conditional latent."""

    steer_latent: torch.Tensor
    brake_gate: torch.Tensor
    brake_latent: torch.Tensor

    def __post_init__(self) -> None:
        if self.steer_latent.shape != self.brake_gate.shape:
            raise ValueError("hurdle action component shapes differ")
        if self.steer_latent.shape != self.brake_latent.shape:
            raise ValueError("hurdle action component shapes differ")
        if self.steer_latent.ndim < 1 or self.steer_latent.shape[-1] != 1:
            raise ValueError("hurdle action tensors must end in one coordinate")
        if not torch.all(torch.isfinite(self.steer_latent)):
            raise ValueError("steering latent is nonfinite")
        if not torch.all(torch.isfinite(self.brake_latent)):
            raise ValueError("brake latent is nonfinite")
        if not torch.all(torch.isfinite(self.brake_gate)):
            raise ValueError("brake gate is nonfinite")
        if not torch.all((self.brake_gate == 0) | (self.brake_gate == 1)):
            raise ValueError("brake gate must contain only 0/1")

    def as_tensor(self) -> torch.Tensor:
        return torch.cat(
            [self.steer_latent, self.brake_gate, self.brake_latent], dim=-1
        )

    @classmethod
    def from_tensor(cls, value: torch.Tensor) -> "MacroResidualAction":
        if value.ndim < 1 or value.shape[-1] != 3 or not torch.all(torch.isfinite(value)):
            raise ValueError("stored hurdle action must be finite [...,3]")
        return cls(value[..., 0:1], value[..., 1:2], value[..., 2:3])


class HurdleResidualDistribution:
    """Independent steering Normal and NO_OP/BRAKE conditional distribution."""

    def __init__(
        self,
        steer_mean: torch.Tensor,
        steer_std: torch.Tensor,
        brake_logits: torch.Tensor,
        brake_mean: torch.Tensor,
        brake_std: torch.Tensor,
    ):
        tensors = (steer_mean, steer_std, brake_logits, brake_mean, brake_std)
        if any(item.shape != steer_mean.shape for item in tensors):
            raise ValueError("hurdle distribution parameter shapes differ")
        if steer_mean.ndim < 1 or steer_mean.shape[-1] != 1:
            raise ValueError("hurdle distribution parameters must end in one coordinate")
        if not all(torch.all(torch.isfinite(item)) for item in tensors):
            raise ValueError("hurdle distribution contains nonfinite parameter")
        if torch.any(steer_std <= 0) or torch.any(brake_std <= 0):
            raise ValueError("hurdle distribution standard deviation must be positive")
        self.steer = torch.distributions.Normal(steer_mean, steer_std)
        self.gate = torch.distributions.Bernoulli(logits=brake_logits)
        self.brake = torch.distributions.Normal(brake_mean, brake_std)

    @property
    def brake_probability(self) -> torch.Tensor:
        return self.gate.probs

    def sample(self) -> MacroResidualAction:
        return MacroResidualAction(
            self.steer.sample(), self.gate.sample(), self.brake.sample()
        )

    def deterministic(self) -> MacroResidualAction:
        gate = (self.gate.logits > 0.0).to(self.gate.logits.dtype)
        return MacroResidualAction(self.steer.mean, gate, self.brake.mean)

    def log_prob(self, action: MacroResidualAction) -> torch.Tensor:
        action.__post_init__()
        steer = self.steer.log_prob(action.steer_latent)
        gate = self.gate.log_prob(action.brake_gate)
        conditional = action.brake_gate * self.brake.log_prob(action.brake_latent)
        return (steer + gate + conditional).squeeze(-1)

    def entropy(self) -> torch.Tensor:
        value = self.steer.entropy() + self.gate.entropy()
        value = value + self.brake_probability * self.brake.entropy()
        return value.squeeze(-1)

    @staticmethod
    def physical_delta(action: MacroResidualAction) -> torch.Tensor:
        """Map stored latents to `[delta_steer, delta_speed]` action units."""

        action.__post_init__()
        delta_steer = torch.tanh(action.steer_latent) * STEER_BUDGET
        brake = torch.sigmoid(action.brake_latent) * BRAKE_BUDGET
        delta_speed = -action.brake_gate * brake
        output = torch.cat([delta_steer, delta_speed], dim=-1)
        if torch.any(output[..., 1] > 0.0) or POSITIVE_SPEED_BUDGET != 0.0:
            raise AssertionError("B+ v2.2 produced forbidden positive speed residual")
        return output


def _initialize_linear(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def _sidecar_feature(
    model: D2RGeometryNet,
    lidar_history: torch.Tensor,
    normalized_bc_feature: torch.Tensor,
    scalar_history: torch.Tensor,
) -> torch.Tensor:
    # The locked 360 -> 18 adaptive average pool has exactly 20 beams in
    # every bin.  Fixed-window avg_pool1d is forward-identical on CUDA and,
    # unlike AdaptiveAvgPool1d backward in the pinned PyTorch build, exposes
    # a deterministic CUDA gradient for arm C.
    encoded = model.encode_beams(lidar_history, pool=False)
    if encoded.shape[-1] != 360:
        raise AssertionError("v2.2 sidecar beam length drift")
    beam = F.avg_pool1d(encoded, kernel_size=20, stride=20).flatten(1)
    return model.fusion(
        torch.cat(
            [
                beam,
                model.bc_projection(normalized_bc_feature),
                model.scalar_projection(scalar_history),
            ],
            dim=1,
        )
    )


def _tensor_sha256(items: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(items):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class V22Policy(nn.Module):
    """Frozen BC plus A/B/C policy feature and common macro residual head."""

    def __init__(
        self,
        arm: str,
        *,
        bc_state_dict: dict[str, torch.Tensor] | None = None,
        sidecar_state_dict: dict[str, torch.Tensor] | None = None,
        sidecar_bc_mean: torch.Tensor | None = None,
        sidecar_bc_std: torch.Tensor | None = None,
        hidden_scale: int = 4,
        initialization_seed: int = SEED,
    ):
        super().__init__()
        self.arm = validate_arm(arm)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.bc = End2Race(mask_prob=0.0, hidden_scale=hidden_scale)
            self.bc_adapter = nn.Sequential(
                nn.Linear(BC_FEATURE_DIM, POLICY_FEATURE_DIM),
                nn.LayerNorm(POLICY_FEATURE_DIM),
                nn.SiLU(),
            )
            self.policy_sidecar = D2RGeometryNet()
            self.action_core = nn.Sequential(
                nn.Linear(POLICY_FEATURE_DIM, POLICY_FEATURE_DIM),
                nn.LayerNorm(POLICY_FEATURE_DIM),
                nn.SiLU(),
            )
            self.steer_mean = nn.Linear(POLICY_FEATURE_DIM, 1)
            self.brake_gate = nn.Linear(POLICY_FEATURE_DIM, 1)
            self.brake_mean = nn.Linear(POLICY_FEATURE_DIM, 1)
        if bc_state_dict is not None:
            self.bc.load_state_dict(bc_state_dict)
        if sidecar_state_dict is not None:
            self.policy_sidecar.load_state_dict(sidecar_state_dict)
        self.shadow_sidecar = copy.deepcopy(self.policy_sidecar)

        mean = torch.zeros(BC_FEATURE_DIM) if sidecar_bc_mean is None else sidecar_bc_mean
        std = torch.ones(BC_FEATURE_DIM) if sidecar_bc_std is None else sidecar_bc_std
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        if mean.shape != (BC_FEATURE_DIM,) or std.shape != (BC_FEATURE_DIM,):
            raise ValueError("sidecar BC normalization must have 1,680 values")
        if not torch.all(torch.isfinite(mean)) or not torch.all(torch.isfinite(std)):
            raise ValueError("sidecar BC normalization contains nonfinite value")
        if torch.any(std <= 0.0):
            raise ValueError("sidecar BC normalization std must be positive")
        self.register_buffer("sidecar_bc_mean", mean.clone())
        self.register_buffer("sidecar_bc_std", std.clone())

        self.log_steer_std = nn.Parameter(torch.tensor(float(INITIAL_STEER_STD)).log())
        self.log_brake_std = nn.Parameter(torch.tensor(float(INITIAL_BRAKE_STD)).log())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed) + 1)
            _initialize_linear(self.bc_adapter)
            _initialize_linear(self.action_core)
            nn.init.zeros_(self.steer_mean.weight)
            nn.init.zeros_(self.steer_mean.bias)
            nn.init.zeros_(self.brake_mean.weight)
            nn.init.zeros_(self.brake_mean.bias)
            nn.init.zeros_(self.brake_gate.weight)
            nn.init.constant_(self.brake_gate.bias, INITIAL_BRAKE_LOGIT)
        self._configure_trainable_parameters()

    def _configure_trainable_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        action_modules: tuple[nn.Module, ...] = (
            self.action_core,
            self.steer_mean,
            self.brake_gate,
            self.brake_mean,
        )
        for module in action_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.log_steer_std.requires_grad = True
        self.log_brake_std.requires_grad = True
        if self.arm == ARM_BC_FROZEN:
            for parameter in self.bc_adapter.parameters():
                parameter.requires_grad = True
        elif self.arm == ARM_SIDECAR_FINETUNE:
            for module in (
                self.policy_sidecar.beam_encoder,
                self.policy_sidecar.bc_projection,
                self.policy_sidecar.scalar_projection,
                self.policy_sidecar.fusion,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        elif self.arm != ARM_SIDECAR_FROZEN:
            raise AssertionError("unreachable v2.2 arm")
        self.bc.eval()
        self.shadow_sidecar.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.bc.eval()
        self.shadow_sidecar.eval()
        if self.arm != ARM_SIDECAR_FINETUNE:
            self.policy_sidecar.eval()
        return self

    @property
    def hidden_size(self) -> int:
        return int(self.bc.gru.hidden_size)

    def zero_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(1, int(batch_size), self.hidden_size, device=device)

    def bc_step(
        self,
        lidar: torch.Tensor,
        previous_speed: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if lidar.ndim != 3 or lidar.shape[1:] != (1, 360):
            raise ValueError("v2.2 BC step LiDAR must be [B,1,360]")
        if previous_speed.shape != (len(lidar), 1, 1):
            raise ValueError("v2.2 BC step speed must be [B,1,1]")
        feature, next_hidden = self.bc.forward_features(lidar, previous_speed, hidden)
        base = self.bc.output_layer(feature)
        return base[:, -1], feature[:, -1], next_hidden

    def _normalized_bc(self, bc_feature: torch.Tensor) -> torch.Tensor:
        if bc_feature.ndim != 2 or bc_feature.shape[1] != BC_FEATURE_DIM:
            raise ValueError("v2.2 BC feature must be [B,1680]")
        return (bc_feature - self.sidecar_bc_mean) / self.sidecar_bc_std

    def policy_feature(
        self,
        bc_feature: torch.Tensor,
        lidar_history: torch.Tensor,
        scalar_history: torch.Tensor,
    ) -> torch.Tensor:
        if scalar_history.shape != (len(bc_feature), SCALAR_HISTORY_DIM):
            raise ValueError("v2.2 scalar history must be [B,24]")
        if self.arm == ARM_BC_FROZEN:
            return self.bc_adapter(bc_feature)
        return _sidecar_feature(
            self.policy_sidecar,
            lidar_history,
            self._normalized_bc(bc_feature),
            scalar_history,
        )

    def distribution(
        self,
        bc_feature: torch.Tensor,
        lidar_history: torch.Tensor,
        scalar_history: torch.Tensor,
    ) -> HurdleResidualDistribution:
        feature = self.action_core(
            self.policy_feature(bc_feature, lidar_history, scalar_history)
        )
        steer_std = self.log_steer_std.exp().expand(len(feature), 1)
        brake_std = self.log_brake_std.exp().expand(len(feature), 1)
        return HurdleResidualDistribution(
            self.steer_mean(feature),
            steer_std,
            self.brake_gate(feature),
            self.brake_mean(feature),
            brake_std,
        )

    @torch.no_grad()
    def diagnostic(
        self,
        bc_feature: torch.Tensor,
        lidar_history: torch.Tensor,
        scalar_history: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = self.shadow_sidecar(
            lidar_history, self._normalized_bc(bc_feature), scalar_history
        )
        return {
            "collision_probability": torch.sigmoid(output["collision_logits"]),
            "ttc": decode_ttc_logits(output["ttc_logits"]),
            "rel_s": output["rel_s"],
            "lateral_gap": output["lateral_gap"],
            "closing_rate": output["closing_rate"],
        }

    @staticmethod
    def compose(base_action: torch.Tensor, action: MacroResidualAction) -> torch.Tensor:
        if base_action.ndim != 2 or base_action.shape[1] != 2:
            raise ValueError("v2.2 BC base action must be [B,2]")
        delta = HurdleResidualDistribution.physical_delta(action)
        if delta.shape != base_action.shape:
            raise ValueError("v2.2 residual/base action shapes differ")
        return base_action + delta

    def optimizer_parameter_groups(self) -> list[dict]:
        action = [
            parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("policy_sidecar.")
        ]
        groups = [{"name": "action_core", "params": action, "lr": ACTION_CORE_LR}]
        if self.arm == ARM_SIDECAR_FINETUNE:
            sidecar = [
                parameter
                for name, parameter in self.named_parameters()
                if parameter.requires_grad and name.startswith("policy_sidecar.")
            ]
            groups.append(
                {"name": "sidecar", "params": sidecar, "lr": SIDECAR_FINETUNE_LR}
            )
        ids = [id(parameter) for group in groups for parameter in group["params"]]
        if not ids or len(ids) != len(set(ids)):
            raise AssertionError("v2.2 optimizer parameter groups overlap or are empty")
        expected = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if set(ids) != expected:
            raise AssertionError("v2.2 optimizer parameter groups omit trainable parameter")
        return groups

    def frozen_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if not parameter.requires_grad
        }

    def assert_frozen_unchanged(self, snapshot: dict[str, torch.Tensor]) -> None:
        current = dict(self.named_parameters())
        if set(snapshot) != {name for name, value in current.items() if not value.requires_grad}:
            raise RuntimeError("v2.2 frozen parameter inventory changed")
        for name, expected in snapshot.items():
            value = current[name]
            if value.grad is not None and torch.any(value.grad != 0):
                raise RuntimeError(f"v2.2 frozen parameter received gradient: {name}")
            if not torch.equal(value.detach().cpu(), expected):
                raise RuntimeError(f"v2.2 frozen parameter mutated: {name}")

    def shadow_sha256(self) -> str:
        return _tensor_sha256(list(self.shadow_sidecar.state_dict().items()))

    def policy_sidecar_encoder_sha256(self) -> str:
        prefixes = ("beam_encoder.", "bc_projection.", "scalar_projection.", "fusion.")
        items = [
            (name, value)
            for name, value in self.policy_sidecar.state_dict().items()
            if name.startswith(prefixes)
        ]
        return _tensor_sha256(items)


class V22Critics(nn.Module):
    """Separate privileged critics so one value loss cannot define all targets."""

    def __init__(self):
        super().__init__()
        self.reward = self._network()
        self.collision = self._network()
        self.performance = self._network()

    @staticmethod
    def _network() -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(PRIVILEGED_FEATURE_DIM, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, privileged: torch.Tensor) -> dict[str, torch.Tensor]:
        if privileged.ndim != 2 or privileged.shape[1] != PRIVILEGED_FEATURE_DIM:
            raise ValueError("v2.2 privileged critic input must be [B,12]")
        return {
            "reward": self.reward(privileged).squeeze(-1),
            "collision": self.collision(privileged).squeeze(-1),
            "performance": self.performance(privileged).squeeze(-1),
        }
