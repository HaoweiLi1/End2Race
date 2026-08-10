"""End2Race actor adapter, fixed exploration distribution, and selectable critics."""

from __future__ import annotations

import copy as copy_module
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from gym_notices import notices as gym_notices

gym_notices.notices.clear()

from gymnasium import spaces
from torch import nn

from latticeplanner.utils import load_config
from model import End2Race
from ppo.reward import CurrentStateClearances, ProgressProjector, wrapped_progress_delta
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.distributions import Distribution

CONFIG = load_config("ppo/ppo_config.yaml")


def speed_exploration_modes():
    return ("baseline", "temporal_global", "corridor_temporal")


def exploration_uses_gate(mode: str) -> bool:
    if mode not in speed_exploration_modes():
        raise ValueError(f"Unknown speed exploration mode: {mode!r}")
    return mode == "corridor_temporal"


def exploration_metadata(mode: str, corridor_gate_config=None) -> dict[str, Any]:
    if mode not in speed_exploration_modes():
        raise ValueError(f"Unknown speed exploration mode: {mode!r}")
    if mode == "corridor_temporal":
        if corridor_gate_config is None:
            raise ValueError("Front-corridor exploration metadata requires its gate configuration")
        gate_type = f"front_corridor_overlap_gap{corridor_gate_config.maximum_front_gap_m:g}"
        gate = asdict(corridor_gate_config)
    else:
        gate_type = "none"
        gate = None
    return {
        "mode": mode,
        "baseline_speed_std": 0.15,
        "corridor_temporal_speed_std": 0.15,
        "temporal_resample_steps": CONFIG.temporal_resample_steps,
        "gate_type": gate_type,
        "gate": gate,
        "training_only": True,
        "deterministic_evaluation_unchanged": True,
    }


PRIVILEGED_FEATURE_NAMES = (
    "delta_s",
    "relative_lateral",
    "relative_long_velocity",
    "relative_lat_velocity",
    "sin_relative_heading",
    "cos_relative_heading",
    "ego_speed",
    "ego_yaw_rate",
    "relative_yaw_rate",
    "obb_longitudinal_clearance",
    "obb_lateral_clearance",
    "wall_clearance",
    "ego_steering_angle",
    "ego_slip_angle",
    "left_body_margin",
    "right_body_margin",
    "sin_track_heading_error",
    "cos_track_heading_error",
    "current_curvature",
    "lookahead_mean_curvature",
)
PRIVILEGED_FEATURE_SIZE = len(PRIVILEGED_FEATURE_NAMES)
PRIVILEGED_FEATURE_LOWS = (
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    0.0, 0.0, 0.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
)
PRIVILEGED_FEATURE_HIGHS = (1.0,) * PRIVILEGED_FEATURE_SIZE

def wrap_to_pi(angle: float) -> float:
    if not np.isfinite(angle):
        raise ValueError("Angle must be finite")
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def soft_normalize_clearance(distance_m: float, half_response_m: float) -> float:
    values = np.asarray((distance_m, half_response_m), dtype=np.float64)
    if not np.isfinite(values).all() or distance_m < 0.0 or half_response_m <= 0.0:
        raise ValueError("Clearance and its half-response scale must be finite, with clearance >= 0 and scale > 0")
    if distance_m <= half_response_m:
        ratio = float(distance_m / half_response_m)
        return float(ratio / (1.0 + ratio))
    normalized = float(1.0 / (1.0 + half_response_m / distance_m))
    return min(normalized, float(np.nextafter(1.0, 0.0)))


@dataclass(frozen=True)
class BodyTrackState:
    signed_offset_m: float
    width_left_m: float
    width_right_m: float
    track_heading_rad: float
    heading_error_rad: float
    lateral_extent_m: float
    left_margin_m: float
    right_margin_m: float
    normalized_left_margin: float
    normalized_right_margin: float


class BoundaryDistanceReference:

    def __init__(self, lane_path: str | Path) -> None:
        data = np.loadtxt(Path(lane_path), delimiter=",", comments="#", dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 4 or data.shape[0] < 3 or not np.isfinite(data).all():
            raise ValueError(f"Expected finite x/y/w_tr_right/w_tr_left rows in lane CSV: {lane_path}")
        if np.linalg.norm(data[-1, :2] - data[0, :2]) <= 1e-9:
            data = data[:-1]
        self.points = data[:, :2]
        self.width_right = data[:, 2]
        self.width_left = data[:, 3]
        if np.any(self.width_right <= 0.0) or np.any(self.width_left <= 0.0):
            raise ValueError(f"Lane CSV boundary widths must be positive: {lane_path}")
        self._segment_vector = np.roll(self.points, -1, axis=0) - self.points
        self._segment_norm_sq = np.einsum("ij,ij->i", self._segment_vector, self._segment_vector)
        if np.any(self._segment_norm_sq <= 0.0):
            raise ValueError(f"Lane CSV contains a zero-length cyclic segment: {lane_path}")

    def body_track_state(self, point_xy: np.ndarray, heading: float, vehicle_length: float, vehicle_width: float) -> BodyTrackState:
        point = np.asarray(point_xy, dtype=np.float64).reshape(-1)
        values = np.asarray((heading, vehicle_length, vehicle_width), dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all() or not np.isfinite(values).all():
            raise ValueError("Body-track geometry requires finite position, heading, and dimensions")
        if vehicle_length <= 0.0 or vehicle_width <= 0.0:
            raise ValueError("Vehicle length and width must be positive")
        offset = point - self.points
        fraction = np.clip(np.einsum("ij,ij->i", offset, self._segment_vector) / self._segment_norm_sq, 0.0, 1.0)
        closest = self.points + fraction[:, None] * self._segment_vector
        residual = point - closest
        index = int(np.argmin(np.einsum("ij,ij->i", residual, residual)))
        direction = self._segment_vector[index] / np.sqrt(self._segment_norm_sq[index])
        signed_offset = float(direction[0] * residual[index, 1] - direction[1] * residual[index, 0])
        next_index = (index + 1) % len(self.points)
        weight = float(fraction[index])
        width_left = float((1.0 - weight) * self.width_left[index] + weight * self.width_left[next_index])
        width_right = float((1.0 - weight) * self.width_right[index] + weight * self.width_right[next_index])
        track_heading = float(np.arctan2(direction[1], direction[0]))
        heading_error = wrap_to_pi(float(heading) - track_heading)
        lateral_extent = 0.5 * (
            float(vehicle_length) * abs(float(np.sin(heading_error)))
            + float(vehicle_width) * abs(float(np.cos(heading_error)))
        )
        left_margin = width_left - signed_offset - lateral_extent
        right_margin = width_right + signed_offset - lateral_extent
        epsilon = np.finfo(np.float64).eps
        left_capacity = max(width_left - lateral_extent, epsilon)
        right_capacity = max(width_right - lateral_extent, epsilon)
        return BodyTrackState(
            signed_offset_m=signed_offset,
            width_left_m=width_left,
            width_right_m=width_right,
            track_heading_rad=track_heading,
            heading_error_rad=heading_error,
            lateral_extent_m=float(lateral_extent),
            left_margin_m=float(left_margin),
            right_margin_m=float(right_margin),
            normalized_left_margin=float(np.clip(left_margin / left_capacity, -1.0, 1.0)),
            normalized_right_margin=float(np.clip(right_margin / right_capacity, -1.0, 1.0)),
        )


class PrivilegedStateExtractor:

    def __init__(
        self,
        map_name: str,
        ego_raceline: str,
        projector: ProgressProjector,
        vehicle_length: float,
        vehicle_width: float,
        *,
        steering_min_rad: float,
        steering_max_rad: float,
    ) -> None:
        if not ego_raceline.startswith("raceline"):
            raise ValueError(f"Cannot derive a lane boundary file from ego raceline {ego_raceline!r}")
        track_dir = Path(__file__).resolve().parents[1] / "f1tenth_racetracks" / map_name
        raceline = np.loadtxt(track_dir / f"{ego_raceline}.csv", delimiter=";", comments="#", dtype=np.float64)
        if raceline.ndim != 2 or raceline.shape[1] < 5 or not np.isfinite(raceline[:, (0, 4)]).all():
            raise ValueError(f"Ego raceline CSV must provide finite s and curvature columns: {ego_raceline}")
        if abs(float(raceline[-1, 0]) - projector.track_length) > 1e-6:
            raise ValueError("Ego raceline closing s must match the progress projector track length")
        if raceline.shape[0] < 4 or np.any(np.diff(raceline[:, 0]) <= 0.0):
            raise ValueError("Ego raceline progress must be strictly increasing")
        self.projector = projector
        self._curvature_s = np.concatenate((raceline[:-1, 0], (projector.track_length,)))
        self._curvature = np.concatenate((raceline[:-1, 4], (raceline[0, 4],)))
        self._curvature_scale = float(np.percentile(np.abs(raceline[:, 4]), CONFIG.curvature_scale_percentile))
        if not np.isfinite(self._curvature_scale) or self._curvature_scale <= np.finfo(np.float64).eps:
            raise ValueError(f"Ego raceline must provide a positive curvature scale: {ego_raceline}")
        self.boundary = BoundaryDistanceReference(track_dir / f"{ego_raceline.replace('raceline', 'lane', 1)}.csv")
        self.vehicle_length = float(vehicle_length)
        self.vehicle_width = float(vehicle_width)
        self.steering_scale_rad = max(abs(float(steering_min_rad)), abs(float(steering_max_rad)))
        self.obb_longitudinal_clearance_half_response_m = CONFIG.obb_longitudinal_clearance_half_response_m
        self.obb_lateral_clearance_half_response_m = CONFIG.obb_lateral_clearance_half_response_m
        self.wall_clearance_half_response_m = CONFIG.wall_clearance_half_response_m
        parameters = np.asarray(
            (
                self.vehicle_length,
                self.vehicle_width,
                self.steering_scale_rad,
                self.obb_longitudinal_clearance_half_response_m,
                self.obb_lateral_clearance_half_response_m,
                self.wall_clearance_half_response_m,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(parameters).all() or np.any(parameters <= 0.0):
            raise ValueError("Privileged extractor dimensions and normalization scales must be positive")

    @property
    def curvature_scale(self) -> float:
        return self._curvature_scale

    @staticmethod
    def _agent_state(raw_observation: dict[str, Any], index: int) -> tuple[np.ndarray, float, float, float]:
        position = np.asarray((raw_observation["poses_x"][index], raw_observation["poses_y"][index]), dtype=np.float64)
        heading = float(np.asarray(raw_observation["poses_theta"])[index])
        speed = float(np.asarray(raw_observation["linear_vels_x"])[index])
        yaw_rate = float(np.asarray(raw_observation["ang_vels_z"])[index])
        if not np.isfinite(np.concatenate((position, (heading, speed, yaw_rate)))).all():
            raise ValueError("Privileged raw simulator state must be finite")
        return position, heading, speed, yaw_rate

    @staticmethod
    def _world_velocity(heading: float, speed: float, slip_angle: float) -> np.ndarray:
        velocity_heading = float(heading)
        if abs(float(speed)) >= CONFIG.dynamic_model_speed_threshold_mps:
            velocity_heading += float(slip_angle)
        return float(speed) * np.asarray((np.cos(velocity_heading), np.sin(velocity_heading)), dtype=np.float64)

    def curvature_at(self, progress_m: float) -> float:
        if not np.isfinite(progress_m):
            raise ValueError("Curvature progress must be finite")
        wrapped_progress = float(progress_m) % self.projector.track_length
        return float(np.interp(wrapped_progress, self._curvature_s, self._curvature))

    def lookahead_mean_curvature_at(self, progress_m: float) -> float:
        samples = (
            float(progress_m)
            + np.arange(1, CONFIG.curvature_lookahead_samples + 1, dtype=np.float64)
            * (CONFIG.curvature_lookahead_m / CONFIG.curvature_lookahead_samples)
        ) % self.projector.track_length
        return float(np.mean([self.curvature_at(sample) for sample in samples]))

    def normalization_metadata(self) -> dict[str, Any]:
        return {
            "version": CONFIG.clearance_normalization_version,
            "delta_s_m": CONFIG.delta_s_scale_m,
            "relative_lateral_m": CONFIG.relative_lateral_scale_m,
            "relative_long_velocity_mps": CONFIG.longitudinal_velocity_scale_mps,
            "relative_lat_velocity_mps": CONFIG.lateral_velocity_scale_mps,
            "ego_speed_mps": CONFIG.ego_speed_scale_mps,
            "yaw_rate_radps": CONFIG.yaw_rate_scale_radps,
            "clearance_formula": CONFIG.clearance_normalization_formula,
            "clearance_scale_source": "fixed_privileged_contract",
            "obb_longitudinal_clearance_half_response_m": self.obb_longitudinal_clearance_half_response_m,
            "obb_lateral_clearance_half_response_m": self.obb_lateral_clearance_half_response_m,
            "wall_clearance_half_response_m": self.wall_clearance_half_response_m,
            "ego_steering_angle_rad": self.steering_scale_rad,
            "ego_slip_angle_rad": CONFIG.slip_angle_scale_rad,
            "curvature_abs_percentile": CONFIG.curvature_scale_percentile,
            "curvature_radpm": self._curvature_scale,
            "curvature_lookahead_m": CONFIG.curvature_lookahead_m,
            "curvature_lookahead_samples": CONFIG.curvature_lookahead_samples,
        }

    def features(
        self,
        raw_observation: dict[str, Any],
        *,
        ego_index: int,
        opponent_index: int,
        ego_steering_angle: float,
        ego_slip_angle: float,
        opponent_slip_angle: float,
        clearances: CurrentStateClearances,
    ) -> np.ndarray:
        ego_position, ego_heading, ego_speed, ego_yaw_rate = self._agent_state(raw_observation, ego_index)
        opponent_position, opponent_heading, opponent_speed, opponent_yaw_rate = self._agent_state(raw_observation, opponent_index)
        physical_state = np.asarray((ego_steering_angle, ego_slip_angle, opponent_slip_angle), dtype=np.float64)
        if not np.isfinite(physical_state).all():
            raise ValueError("Privileged steering and slip angles must be finite")
        if not isinstance(clearances, CurrentStateClearances):
            raise TypeError("Privileged features require reward's current-state clearance result")
        ego_progress = self.projector.project(ego_position)
        opponent_progress = self.projector.project(opponent_position)
        delta_s = wrapped_progress_delta(ego_progress, opponent_progress, self.projector.track_length)
        cos_ego, sin_ego = np.cos(ego_heading), np.sin(ego_heading)
        relative_position = opponent_position - ego_position
        relative_lateral = -sin_ego * relative_position[0] + cos_ego * relative_position[1]
        relative_heading = wrap_to_pi(opponent_heading - ego_heading)
        ego_velocity_world = self._world_velocity(ego_heading, ego_speed, ego_slip_angle)
        opponent_velocity_world = self._world_velocity(opponent_heading, opponent_speed, opponent_slip_angle)
        velocity_delta_world = opponent_velocity_world - ego_velocity_world
        relative_long_velocity = cos_ego * velocity_delta_world[0] + sin_ego * velocity_delta_world[1]
        relative_lat_velocity = -sin_ego * velocity_delta_world[0] + cos_ego * velocity_delta_world[1]
        body_track = self.boundary.body_track_state(ego_position, ego_heading, self.vehicle_length, self.vehicle_width)
        current_curvature = self.curvature_at(ego_progress)
        lookahead_curvature = self.lookahead_mean_curvature_at(ego_progress)
        features = np.asarray(
            (
                np.clip(delta_s / CONFIG.delta_s_scale_m, -1.0, 1.0),
                np.clip(relative_lateral / CONFIG.relative_lateral_scale_m, -1.0, 1.0),
                np.clip(relative_long_velocity / CONFIG.longitudinal_velocity_scale_mps, -1.0, 1.0),
                np.clip(relative_lat_velocity / CONFIG.lateral_velocity_scale_mps, -1.0, 1.0),
                np.sin(relative_heading),
                np.cos(relative_heading),
                np.clip(ego_speed / CONFIG.ego_speed_scale_mps, -1.0, 1.0),
                np.clip(ego_yaw_rate / CONFIG.yaw_rate_scale_radps, -1.0, 1.0),
                np.clip((opponent_yaw_rate - ego_yaw_rate) / CONFIG.yaw_rate_scale_radps, -1.0, 1.0),
                soft_normalize_clearance(clearances.obb_longitudinal_clearance_m, self.obb_longitudinal_clearance_half_response_m),
                soft_normalize_clearance(clearances.obb_lateral_clearance_m, self.obb_lateral_clearance_half_response_m),
                soft_normalize_clearance(clearances.wall_clearance_m, self.wall_clearance_half_response_m),
                np.clip(float(ego_steering_angle) / self.steering_scale_rad, -1.0, 1.0),
                np.clip(float(ego_slip_angle) / CONFIG.slip_angle_scale_rad, -1.0, 1.0),
                body_track.normalized_left_margin,
                body_track.normalized_right_margin,
                np.sin(body_track.heading_error_rad),
                np.cos(body_track.heading_error_rad),
                np.clip(current_curvature / self._curvature_scale, -1.0, 1.0),
                np.clip(lookahead_curvature / self._curvature_scale, -1.0, 1.0),
            ),
            dtype=np.float32,
        )
        lows = np.asarray(PRIVILEGED_FEATURE_LOWS, dtype=np.float32)
        highs = np.asarray(PRIVILEGED_FEATURE_HIGHS, dtype=np.float32)
        if features.shape != (PRIVILEGED_FEATURE_SIZE,) or features.dtype != np.float32:
            raise RuntimeError(f"Privileged feature contract violated: shape={features.shape}, dtype={features.dtype}")
        if not np.isfinite(features).all() or np.any(features < lows) or np.any(features > highs):
            raise ValueError(f"Privileged features must be finite and within declared bounds, got {features!r}")
        return features


CRITIC_VARIANTS = ("mlp", "independent_gru", "priviledge_mlp", "privilege_gru")
P20_CRITIC_VARIANTS = ("priviledge_mlp", "privilege_gru")
END2RACE_OBSERVATION_SIZE = 361
END2RACE_LIDAR_SIZE = 360
END2RACE_ACTION_SIZE = 2
NOOP_SPEED_BOUND = float(np.finfo(np.float32).max)


class EvaluatorCompatibleJointDistribution(Distribution):
    """Squashed latent steering Gaussian plus physical speed Gaussian."""

    def __init__(self, steer_bound: float = CONFIG.steering_bound, inverse_tanh_epsilon: float = 1e-6):
        super().__init__()
        self.steer_bound = float(steer_bound)
        self.inverse_tanh_epsilon = float(inverse_tanh_epsilon)
        self.raw_mean_actions: torch.Tensor | None = None
        self.latent_steer_mean: torch.Tensor | None = None
        self.steer_distribution: torch.distributions.Normal | None = None
        self.speed_distribution: torch.distributions.Normal | None = None

    def proba_distribution_net(self, *args: Any, **kwargs: Any) -> nn.Module:
        del args, kwargs
        return nn.Identity()

    @staticmethod
    def _atanh(value: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.log1p(value) - torch.log1p(-value))

    def proba_distribution(
        self,
        raw_mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        speed_log_std: torch.Tensor | None = None,
    ) -> "EvaluatorCompatibleJointDistribution":
        normalized_mode = (raw_mean_actions[:, 0] / self.steer_bound).clamp(-1.0 + self.inverse_tanh_epsilon, 1.0 - self.inverse_tanh_epsilon)
        self.raw_mean_actions = raw_mean_actions
        self.latent_steer_mean = self._atanh(normalized_mode)
        std = log_std.exp()
        self.steer_distribution = torch.distributions.Normal(self.latent_steer_mean, std[0])
        speed_scale = std[1] if speed_log_std is None else speed_log_std.exp()
        self.speed_distribution = torch.distributions.Normal(
            raw_mean_actions[:, 1],
            speed_scale,
        )
        self.distribution = (self.steer_distribution, self.speed_distribution)
        return self

    def _parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.distributions.Normal, torch.distributions.Normal]:
        if self.raw_mean_actions is None or self.latent_steer_mean is None or self.steer_distribution is None or self.speed_distribution is None:
            raise RuntimeError("Action distribution parameters have not been set")
        return self.raw_mean_actions, self.latent_steer_mean, self.steer_distribution, self.speed_distribution

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        _raw_means, _latent_mean, steer_distribution, speed_distribution = self._parameters()
        epsilon = torch.finfo(actions.dtype).eps
        normalized_steer = (actions[:, 0] / self.steer_bound).clamp(-1.0 + epsilon, 1.0 - epsilon)
        latent_steer = self._atanh(normalized_steer)
        scale = torch.as_tensor(self.steer_bound, dtype=actions.dtype, device=actions.device)
        log_abs_det_jacobian = torch.log(scale) + torch.log1p(-normalized_steer.square())
        return steer_distribution.log_prob(latent_steer) - log_abs_det_jacobian + speed_distribution.log_prob(actions[:, 1])

    def entropy(self) -> None:
        return None

    def sample(self) -> torch.Tensor:
        _raw_means, _latent_mean, steer_distribution, speed_distribution = self._parameters()
        steering = self.steer_bound * torch.tanh(steer_distribution.rsample())
        return torch.stack((steering, speed_distribution.rsample()), dim=1)

    def sample_with_speed_standard_noise(
        self,
        speed_standard_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Sample steering normally while applying an audited speed residual."""

        _raw_means, _latent_mean, steer_distribution, speed_distribution = self._parameters()
        noise = speed_standard_noise.to(
            dtype=speed_distribution.loc.dtype,
            device=speed_distribution.loc.device,
        ).reshape(-1)
        if noise.shape != speed_distribution.loc.shape:
            raise ValueError(
                "Speed standard noise must match the distribution batch"
            )
        steering = self.steer_bound * torch.tanh(steer_distribution.rsample())
        speed = speed_distribution.loc + speed_distribution.scale * noise
        return torch.stack((steering, speed), dim=1)

    def mode(self) -> torch.Tensor:
        raw_means, latent_mean, _steer_distribution, _speed_distribution = self._parameters()
        return torch.stack((self.steer_bound * torch.tanh(latent_mean), raw_means[:, 1]), dim=1)

    def actions_from_params(self, raw_mean_actions: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.proba_distribution(raw_mean_actions, log_std).get_actions(deterministic=deterministic)

    def log_prob_from_params(self, raw_mean_actions: torch.Tensor, log_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(raw_mean_actions, log_std)
        return actions, self.log_prob(actions)


class GRUWithLSTMStateInterface(nn.Module):
    """Expose the actor GRU through SB3's recurrent ``(h, c)`` interface."""

    def __init__(self, gru: nn.GRU):
        super().__init__()
        self.gru = gru
        self.input_size = gru.input_size
        self.hidden_size = gru.hidden_size
        self.num_layers = gru.num_layers

    def forward(self, x: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden, _cell = states
        output, next_hidden = self.gru(x.transpose(0, 1), hidden)
        return output.transpose(0, 1), (next_hidden, torch.zeros_like(next_hidden))


class MLPCritic(nn.Module):
    """Single-frame BC-style critic with the recurrent layer replaced by an MLP."""

    def __init__(self, actor: End2Race):
        super().__init__()
        self.k = nn.Parameter(actor.k.detach().clone())
        self.speed_mlp = copy_module.deepcopy(actor.speed_mlp)
        self.value_head = nn.Sequential(
            nn.Linear(actor.gru.input_size, 60),
            nn.ReLU(),
            nn.Linear(60, 1),
        )
        self._initialize_value_head()

    def _initialize_value_head(self) -> None:
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        lidar = observation[..., :END2RACE_LIDAR_SIZE]
        previous_speed = observation[..., END2RACE_LIDAR_SIZE:]
        pressure = (-1.0 / (1.0 + torch.exp(-self.k * lidar)) + 1.0) * 2.0
        speed_embedding = self.speed_mlp(previous_speed)
        return self.value_head(torch.cat((pressure, speed_embedding), dim=-1))


class PriviledgeMLPCritic(nn.Module):
    """MLP critic over the P20 privileged pre-action physical state."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(PRIVILEGED_FEATURE_SIZE, 120),
            nn.ReLU(),
            nn.Linear(120, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
        )
        self._initialize_linear_layers()

    def _initialize_linear_layers(self) -> None:
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class IndependentGRUCritic(nn.Module):
    """Independent BC-initialized pressure/speed/GRU front-end with a value head."""

    def __init__(self, actor: End2Race):
        super().__init__()
        self.k = nn.Parameter(actor.k.detach().clone())
        self.speed_mlp = copy_module.deepcopy(actor.speed_mlp)
        self.gru = copy_module.deepcopy(actor.gru)
        self.value_head = nn.Sequential(
            nn.Linear(self.gru.hidden_size, self.gru.input_size),
            nn.ReLU(),
            nn.Linear(self.gru.input_size, 1),
        )
        self._initialize_value_head()
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def _initialize_value_head(self) -> None:
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def step(self, lidar: torch.Tensor, previous_speed: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run BC-style preprocessing plus one GRU segment and return values with the next hidden."""

        pressure = (-1.0 / (1.0 + torch.exp(-self.k * lidar)) + 1.0) * 2.0
        speed_embedding = self.speed_mlp(previous_speed)
        gru_output, next_hidden = self.gru(torch.cat((pressure, speed_embedding), dim=2), hidden)
        return self.value_head(gru_output[:, -1, :]), next_hidden


class PrivilegeGRUCritic(nn.Module):
    """Independent BC-initialized GRU critic with zero-initialized P20 late fusion."""

    def __init__(self, actor: End2Race):
        super().__init__()
        self.k = nn.Parameter(actor.k.detach().clone())
        self.speed_mlp = copy_module.deepcopy(actor.speed_mlp)
        self.gru = copy_module.deepcopy(actor.gru)

        # Keep creation and initialization order identical to IndependentGRUCritic,
        # then add the zero-initialized P20 residual without disturbing that baseline.
        self.hidden_projection = nn.Linear(self.gru.hidden_size, self.gru.input_size)
        self.activation = nn.ReLU()
        self.value_output = nn.Linear(self.gru.input_size, 1)
        self._initialize_value_head()
        self.privileged_projection = nn.Linear(
            PRIVILEGED_FEATURE_SIZE,
            self.gru.input_size,
            bias=False,
        )
        nn.init.zeros_(self.privileged_projection.weight)
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def _initialize_value_head(self) -> None:
        for module in (self.hidden_projection, self.value_output):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def step(
        self,
        lidar: torch.Tensor,
        previous_speed: torch.Tensor,
        hidden: torch.Tensor,
        privileged_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the independent GRU and fuse only the current P20 before the value output."""

        if privileged_features.shape[-1] != PRIVILEGED_FEATURE_SIZE:
            raise RuntimeError(
                f"privilege_gru expects {PRIVILEGED_FEATURE_SIZE} privileged features, "
                f"got {privileged_features.shape[-1]}"
            )
        pressure = (-1.0 / (1.0 + torch.exp(-self.k * lidar)) + 1.0) * 2.0
        speed_embedding = self.speed_mlp(previous_speed)
        gru_output, next_hidden = self.gru(torch.cat((pressure, speed_embedding), dim=2), hidden)
        fused = self.hidden_projection(gru_output[:, -1, :]) + self.privileged_projection(privileged_features)
        return self.value_output(self.activation(fused)), next_hidden


class End2RaceGRUPolicy(RecurrentActorCriticPolicy):
    """Use the original End2Race actor unchanged inside recurrent PPO."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        checkpoint_path: str | Path,
        hidden_scale: int = 4,
        critic_variant: str = "mlp",
        gru_learning_rate: float = 5.0e-6,
        head_learning_rate: float = 5.0e-5,
        critic_learning_rate: float = 5.0e-4,
        steering_latent_std: float = 0.03,
        speed_physical_std: float = 0.15,
        speed_exploration_mode: str = "baseline",
        **kwargs: Any,
    ):
        kwargs.pop("use_sde", None)
        if critic_variant not in CRITIC_VARIANTS:
            raise ValueError(f"critic_variant must be one of {CRITIC_VARIANTS}, got {critic_variant!r}")
        if steering_latent_std <= 0 or speed_physical_std <= 0:
            raise ValueError("Exploration standard deviations must be positive")
        if speed_exploration_mode not in speed_exploration_modes():
            raise ValueError(
                f"speed_exploration_mode must be one of {speed_exploration_modes()}"
            )
        if (
            speed_exploration_mode != "baseline"
            and abs(float(speed_physical_std) - 0.15) > 1e-12
        ):
            raise ValueError(
                "Structured speed exploration requires the frozen 0.15 baseline std"
            )
        expected_observation_size = END2RACE_OBSERVATION_SIZE + (PRIVILEGED_FEATURE_SIZE if critic_variant in P20_CRITIC_VARIANTS else 0)
        if tuple(observation_space.shape) != (expected_observation_size,):
            raise ValueError(
                f"Critic variant {critic_variant!r} requires a {expected_observation_size}D observation space, got {observation_space.shape}"
            )
        super().__init__(observation_space, action_space, lr_schedule, net_arch=[], ortho_init=False, use_sde=False, log_std_init=0.0, lstm_hidden_size=1, n_lstm_layers=1, shared_lstm=False, enable_critic_lstm=False, **kwargs)

        self.critic_variant = critic_variant
        self.speed_exploration_mode = speed_exploration_mode
        self._rollout_danger_gates: torch.Tensor | None = None
        self._rollout_episode_starts: torch.Tensor | None = None
        self._temporal_speed_noise: torch.Tensor | None = None
        self._temporal_steps_remaining: torch.Tensor | None = None
        self.end2race_actor = End2Race(mask_prob=0.0, hidden_scale=hidden_scale)
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        self.end2race_actor.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.pretrained_model_path = str(checkpoint)

        self.lstm_actor = GRUWithLSTMStateInterface(self.end2race_actor.gru)
        self.lstm_output_dim = self.end2race_actor.gru.hidden_size
        self.lstm_hidden_state_shape = (self.end2race_actor.gru.num_layers, 1, self.end2race_actor.gru.hidden_size)
        self.lstm_critic = None
        self.critic = None
        if critic_variant == "mlp":
            self.value_net = MLPCritic(self.end2race_actor)
        elif critic_variant == "independent_gru":
            self.value_net = IndependentGRUCritic(self.end2race_actor)
        elif critic_variant == "privilege_gru":
            self.value_net = PrivilegeGRUCritic(self.end2race_actor)
        else:
            self.value_net = PriviledgeMLPCritic()
        self.action_net = nn.Identity()
        self.action_dist = EvaluatorCompatibleJointDistribution()

        self.log_std.data.copy_(torch.tensor([np.log(steering_latent_std), np.log(speed_physical_std)], dtype=self.log_std.dtype, device=self.log_std.device))
        self.log_std.requires_grad_(False)
        for parameter in self.end2race_actor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.end2race_actor.gru.parameters():
            parameter.requires_grad_(True)
        for parameter in self.end2race_actor.output_layer.parameters():
            parameter.requires_grad_(True)

        self.actor_parameters = tuple(self.end2race_actor.gru.parameters()) + tuple(self.end2race_actor.output_layer.parameters())
        self.critic_parameters = tuple(self.value_net.parameters())
        actor_groups = [
            {"params": self.end2race_actor.gru.parameters(), "lr": gru_learning_rate},
            {"params": self.end2race_actor.output_layer.parameters(), "lr": head_learning_rate},
        ]
        self.actor_optimizer = self.optimizer_class(actor_groups, lr=gru_learning_rate, **self.optimizer_kwargs)
        self.critic_optimizer = self.optimizer_class(self.critic_parameters, lr=critic_learning_rate, **self.optimizer_kwargs)
        self.optimizer = self.actor_optimizer

    @property
    def critic_is_independent_gru(self) -> bool:
        return self.critic_variant in ("independent_gru", "privilege_gru")

    @staticmethod
    def _actor_observation(obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        full = obs["actor"] if isinstance(obs, dict) else obs
        return full[..., :END2RACE_OBSERVATION_SIZE]

    def _actor_forward(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Collection path preserving batch-size-one actor execution."""

        hidden, _dummy_cell = states
        actor_obs = self._actor_observation(obs).float()
        if actor_obs.ndim == 1:
            actor_obs = actor_obs.unsqueeze(0)
        n_seq = hidden.shape[1]
        obs_sequence = actor_obs.reshape(n_seq, -1, END2RACE_OBSERVATION_SIZE).swapaxes(0, 1)
        start_sequence = episode_starts.float().reshape(n_seq, -1).swapaxes(0, 1)
        means: list[torch.Tensor] = []
        for step_obs, episode_start in zip(obs_sequence, start_sequence):
            hidden = hidden * (1.0 - episode_start).view(1, n_seq, 1)
            lidar = step_obs[:, :END2RACE_LIDAR_SIZE].unsqueeze(1)
            previous_speed = step_obs[:, END2RACE_LIDAR_SIZE:].unsqueeze(1)
            slot_means: list[torch.Tensor] = []
            slot_hidden: list[torch.Tensor] = []
            for sequence_index in range(n_seq):
                action_sequence, next_hidden = self.end2race_actor(lidar[sequence_index : sequence_index + 1], previous_speed[sequence_index : sequence_index + 1], hidden[:, sequence_index : sequence_index + 1])
                slot_means.append(action_sequence[:, -1, :])
                slot_hidden.append(next_hidden)
            hidden = torch.cat(slot_hidden, dim=1)
            means.append(torch.cat(slot_means, dim=0))
        mean_actions = torch.stack(means).transpose(0, 1).reshape(-1, END2RACE_ACTION_SIZE)
        return mean_actions, (hidden, torch.zeros_like(hidden))

    def _actor_replay_batched(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        valid_by_timestep: tuple[tuple[bool, ...], ...] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Replay one FP32 actor call per timestep over only valid sequence slots."""

        hidden, dummy_cell = states
        actor_obs = self._actor_observation(obs)
        if actor_obs.ndim == 1:
            actor_obs = actor_obs.unsqueeze(0)
        if actor_obs.dtype != torch.float32 or hidden.dtype != torch.float32 or dummy_cell.dtype != torch.float32 or episode_starts.dtype != torch.float32:
            raise RuntimeError("PPO actor replay tensors must remain float32")
        if hidden.shape != dummy_cell.shape or hidden.ndim != 3:
            raise RuntimeError("Actor replay hidden and dummy cell shapes must match and be rank 3")
        if actor_obs.device != hidden.device or episode_starts.device != actor_obs.device:
            raise RuntimeError("Actor replay tensors must share one device")
        parameters = tuple(self.end2race_actor.parameters())
        if any(parameter.dtype != torch.float32 or parameter.device != actor_obs.device for parameter in parameters):
            raise RuntimeError("Actor parameters and replay tensors must share FP32 dtype and device")
        if actor_obs.is_cuda and (torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest" or torch.backends.cudnn.benchmark):
            raise RuntimeError("CUDA actor replay requires TF32 off, highest FP32 precision, and cuDNN benchmark off")

        n_seq = hidden.shape[1]
        if n_seq <= 0 or actor_obs.ndim != 2 or actor_obs.shape[1] != END2RACE_OBSERVATION_SIZE or actor_obs.shape[0] % n_seq != 0:
            raise RuntimeError(f"Invalid actor replay layout: observations={tuple(actor_obs.shape)}, sequences={n_seq}")
        max_length = actor_obs.shape[0] // n_seq
        if episode_starts.numel() != actor_obs.shape[0]:
            raise RuntimeError("Actor replay episode starts must match observation rows")
        if valid_by_timestep is not None and (len(valid_by_timestep) != max_length or any(len(row) != n_seq for row in valid_by_timestep)):
            raise RuntimeError("Actor replay padding mask does not match the padded batch")

        obs_sequence = actor_obs.reshape(n_seq, max_length, END2RACE_OBSERVATION_SIZE).swapaxes(0, 1)
        start_sequence = episode_starts.reshape(n_seq, max_length).swapaxes(0, 1)
        means: list[torch.Tensor] = []
        for timestep, (step_obs, episode_start) in enumerate(zip(obs_sequence, start_sequence)):
            hidden = hidden * (1.0 - episode_start).view(1, n_seq, 1)
            active = list(range(n_seq)) if valid_by_timestep is None else [index for index, valid in enumerate(valid_by_timestep[timestep]) if valid]
            next_by_slot = [hidden[:, index : index + 1] for index in range(n_seq)]
            means_by_slot = [torch.zeros((1, END2RACE_ACTION_SIZE), dtype=actor_obs.dtype, device=actor_obs.device) for _ in range(n_seq)]
            if active:
                indices = torch.as_tensor(active, dtype=torch.long, device=actor_obs.device)
                action_sequence, next_hidden = self.end2race_actor(step_obs[indices, :END2RACE_LIDAR_SIZE].unsqueeze(1), step_obs[indices, END2RACE_LIDAR_SIZE:].unsqueeze(1), hidden[:, indices])
                active_means = action_sequence[:, -1, :]
                for offset, slot in enumerate(active):
                    next_by_slot[slot] = next_hidden[:, offset : offset + 1]
                    means_by_slot[slot] = active_means[offset : offset + 1]
            hidden = torch.cat(next_by_slot, dim=1)
            means.append(torch.cat(means_by_slot, dim=0))
        mean_actions = torch.stack(means).transpose(0, 1).reshape(-1, END2RACE_ACTION_SIZE)
        return mean_actions, (hidden, torch.zeros_like(hidden))

    def _actor_replay_collection_equivalent(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        valid_by_timestep: tuple[tuple[bool, ...], ...] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Replay with the same one-logical-slot actor calls used in collection."""

        hidden, dummy_cell = states
        actor_obs = self._actor_observation(obs)
        if actor_obs.ndim == 1:
            actor_obs = actor_obs.unsqueeze(0)
        if (
            actor_obs.dtype != torch.float32
            or hidden.dtype != torch.float32
            or dummy_cell.dtype != torch.float32
            or episode_starts.dtype != torch.float32
        ):
            raise RuntimeError("Exact actor replay tensors must remain float32")
        n_seq = hidden.shape[1]
        if (
            n_seq <= 0
            or actor_obs.ndim != 2
            or actor_obs.shape[1] != END2RACE_OBSERVATION_SIZE
            or actor_obs.shape[0] % n_seq != 0
        ):
            raise RuntimeError("Invalid exact actor replay layout")
        max_length = actor_obs.shape[0] // n_seq
        obs_sequence = actor_obs.reshape(
            n_seq, max_length, END2RACE_OBSERVATION_SIZE
        ).swapaxes(0, 1)
        start_sequence = episode_starts.reshape(
            n_seq, max_length
        ).swapaxes(0, 1)
        means: list[torch.Tensor] = []
        for timestep, (step_obs, episode_start) in enumerate(
            zip(obs_sequence, start_sequence)
        ):
            hidden = hidden * (1.0 - episode_start).view(1, n_seq, 1)
            active = (
                list(range(n_seq))
                if valid_by_timestep is None
                else [
                    index
                    for index, valid in enumerate(
                        valid_by_timestep[timestep]
                    )
                    if valid
                ]
            )
            next_by_slot = [
                hidden[:, index : index + 1] for index in range(n_seq)
            ]
            means_by_slot = [
                torch.zeros(
                    (1, END2RACE_ACTION_SIZE),
                    dtype=actor_obs.dtype,
                    device=actor_obs.device,
                )
                for _ in range(n_seq)
            ]
            for slot in active:
                action_sequence, next_hidden = self.end2race_actor(
                    step_obs[
                        slot : slot + 1, :END2RACE_LIDAR_SIZE
                    ].unsqueeze(1),
                    step_obs[
                        slot : slot + 1, END2RACE_LIDAR_SIZE:
                    ].unsqueeze(1),
                    hidden[:, slot : slot + 1],
                )
                means_by_slot[slot] = action_sequence[:, -1, :]
                next_by_slot[slot] = next_hidden
            hidden = torch.cat(next_by_slot, dim=1)
            means.append(torch.cat(means_by_slot, dim=0))
        mean_actions = torch.stack(means).transpose(0, 1).reshape(
            -1, END2RACE_ACTION_SIZE
        )
        return mean_actions, (hidden, torch.zeros_like(hidden))

    def _distribution(
        self,
        mean_actions: torch.Tensor,
        speed_log_std: torch.Tensor | None = None,
    ) -> EvaluatorCompatibleJointDistribution:
        if mean_actions.dtype != torch.float32 or self.log_std.dtype != torch.float32:
            raise RuntimeError("PPO actor distribution tensors must remain float32")
        return self.action_dist.proba_distribution(
            mean_actions,
            self.log_std,
            speed_log_std=speed_log_std,
        )

    def prepare_rollout_exploration(
        self,
        danger_gates: np.ndarray,
        episode_starts: np.ndarray,
    ) -> None:
        """Stage causal gate/reset state for exactly one vector action."""

        gates = np.asarray(danger_gates, dtype=bool).reshape(-1)
        starts = np.asarray(episode_starts, dtype=bool).reshape(-1)
        if gates.shape != starts.shape or gates.size == 0:
            raise ValueError("Exploration gates and episode starts must align")
        self._rollout_danger_gates = torch.as_tensor(
            gates,
            dtype=torch.bool,
            device=self.device,
        )
        self._rollout_episode_starts = torch.as_tensor(
            starts,
            dtype=torch.bool,
            device=self.device,
        )

    def _ensure_temporal_state(self, batch_size: int) -> None:
        if (
            self._temporal_speed_noise is None
            or self._temporal_speed_noise.numel() != batch_size
        ):
            self._temporal_speed_noise = torch.zeros(
                batch_size, dtype=torch.float32, device=self.device
            )
            self._temporal_steps_remaining = torch.zeros(
                batch_size, dtype=torch.int64, device=self.device
            )

    def _structured_rollout_parameters(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._rollout_danger_gates is None
            or self._rollout_episode_starts is None
        ):
            raise RuntimeError(
                "Structured exploration must be prepared before policy.forward"
            )
        if self._rollout_danger_gates.numel() != batch_size:
            raise RuntimeError("Prepared exploration batch does not match actor batch")
        gates = self._rollout_danger_gates
        starts = self._rollout_episode_starts
        baseline_log_std = torch.full(
            (batch_size,),
            float(np.log(0.15)),
            dtype=torch.float32,
            device=self.device,
        )
        self._ensure_temporal_state(batch_size)
        assert self._temporal_speed_noise is not None
        assert self._temporal_steps_remaining is not None
        self._temporal_steps_remaining[starts] = 0
        self._temporal_speed_noise[starts] = 0.0

        if self.speed_exploration_mode == "temporal_global":
            begin = self._temporal_steps_remaining <= 0
            count = int(begin.sum().item())
            if count:
                self._temporal_speed_noise[begin] = torch.randn(
                    count, dtype=torch.float32, device=self.device
                )
                self._temporal_steps_remaining[begin] = CONFIG.temporal_resample_steps
            noise = self._temporal_speed_noise.clone()
            self._temporal_steps_remaining -= 1
            return baseline_log_std, noise

        if self.speed_exploration_mode != "corridor_temporal":
            raise RuntimeError(
                f"Unexpected structured exploration mode {self.speed_exploration_mode!r}"
            )
        active = self._temporal_steps_remaining > 0
        begin = ~active & gates
        count = int(begin.sum().item())
        if count:
            self._temporal_speed_noise[begin] = torch.randn(
                count, dtype=torch.float32, device=self.device
            )
            self._temporal_steps_remaining[begin] = CONFIG.temporal_resample_steps
        active = self._temporal_steps_remaining > 0
        fresh = ~active
        fresh_count = int(fresh.sum().item())
        if fresh_count:
            self._temporal_speed_noise[fresh] = torch.randn(
                fresh_count, dtype=torch.float32, device=self.device
            )
        noise = self._temporal_speed_noise.clone()
        self._temporal_steps_remaining[active] -= 1
        return baseline_log_std, noise

    def _stage_exploration_transition(
        self,
        *,
        speed_log_std: torch.Tensor,
    ) -> None:
        rollout_buffer = self._end2race_rollout_buffer
        rollout_buffer.stage_exploration(
            speed_log_std=speed_log_std.detach().cpu().numpy(),
        )

    def _critic_observation(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """Slice flat observation rows into the feed-forward critic input."""

        full = (obs["actor"] if isinstance(obs, dict) else obs).float()
        if full.ndim == 1:
            full = full.unsqueeze(0)
        if self.critic_variant == "priviledge_mlp":
            return full[..., END2RACE_OBSERVATION_SIZE:]
        return full[..., :END2RACE_OBSERVATION_SIZE]

    def _independent_gru_forward_collection(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-timestep recurrent critic step preserving batch-size-one execution."""

        hidden, _dummy_cell = states
        full_obs = (obs["actor"] if isinstance(obs, dict) else obs).float()
        if full_obs.ndim == 1:
            full_obs = full_obs.unsqueeze(0)
        critic_obs = full_obs[..., :END2RACE_OBSERVATION_SIZE]
        privileged_features = full_obs[..., END2RACE_OBSERVATION_SIZE:]
        n_seq = hidden.shape[1]
        if critic_obs.shape[0] != n_seq:
            raise RuntimeError("Recurrent critic collection requires one timestep per sequence slot")
        hidden = hidden * (1.0 - episode_starts.float()).view(1, n_seq, 1)
        lidar = critic_obs[:, :END2RACE_LIDAR_SIZE].unsqueeze(1)
        previous_speed = critic_obs[:, END2RACE_LIDAR_SIZE:].unsqueeze(1)
        slot_values: list[torch.Tensor] = []
        slot_hidden: list[torch.Tensor] = []
        for sequence_index in range(n_seq):
            recurrent_inputs = (
                lidar[sequence_index : sequence_index + 1],
                previous_speed[sequence_index : sequence_index + 1],
                hidden[:, sequence_index : sequence_index + 1],
            )
            if self.critic_variant == "privilege_gru":
                values, next_hidden = self.value_net.step(
                    *recurrent_inputs,
                    privileged_features[sequence_index : sequence_index + 1],
                )
            else:
                values, next_hidden = self.value_net.step(*recurrent_inputs)
            slot_values.append(values)
            slot_hidden.append(next_hidden)
        next_hidden = torch.cat(slot_hidden, dim=1)
        return torch.cat(slot_values, dim=0), (next_hidden, torch.zeros_like(next_hidden))

    @staticmethod
    def _zero_states(states: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros_like(states[0])
        return zero, zero.clone()

    def supports_end2race_rollout_buffer(self) -> bool:
        return isinstance(self.lstm_actor, GRUWithLSTMStateInterface) and self.lstm_critic is None and self.critic is None

    def forward(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
        mean_actions, actor_states = self._actor_forward(obs, lstm_states.pi, episode_starts)
        batch_size = mean_actions.shape[0]
        if (
            deterministic
            or self.speed_exploration_mode == "baseline"
        ):
            distribution = self._distribution(mean_actions)
            actions = distribution.get_actions(deterministic=deterministic)
            speed_log_std = self.log_std[1].expand(batch_size)
            log_prob = distribution.log_prob(actions)
        else:
            speed_log_std, speed_standard_noise = self._structured_rollout_parameters(batch_size)
            distribution = self._distribution(
                mean_actions,
                speed_log_std=speed_log_std,
            )
            actions = (
                distribution.sample()
                if speed_standard_noise is None
                else distribution.sample_with_speed_standard_noise(
                    speed_standard_noise
                )
            )
            log_prob = distribution.log_prob(actions)
        self._stage_exploration_transition(
            speed_log_std=speed_log_std,
        )
        if self.critic_is_independent_gru:
            values, vf_states = self._independent_gru_forward_collection(obs, lstm_states.vf, episode_starts)
        else:
            values = self.value_net(self._critic_observation(obs))
            vf_states = self._zero_states(lstm_states.vf)
        return actions, values, log_prob, RNNStates(actor_states, vf_states)

    def get_distribution(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[Distribution, tuple[torch.Tensor, torch.Tensor]]:
        mean_actions, actor_states = self._actor_forward(obs, lstm_states, episode_starts)
        return self._distribution(mean_actions), actor_states

    def forward_independent_collection(self, obs: torch.Tensor, lstm_states: RNNStates, episode_starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, RNNStates]:
        if self.speed_exploration_mode != "baseline" or obs.shape[0] != 1 or episode_starts.shape != (1,):
            raise RuntimeError("Independent online branch collection requires one baseline-exploration sequence")
        mean_actions, actor_states = self._actor_forward(obs, lstm_states.pi, episode_starts)
        actions = self._distribution(mean_actions).sample()
        if self.critic_is_independent_gru:
            values, critic_states = self._independent_gru_forward_collection(obs, lstm_states.vf, episode_starts)
        else:
            values = self.value_net(self._critic_observation(obs))
            critic_states = self._zero_states(lstm_states.vf)
        return actions, values, RNNStates(actor_states, critic_states)

    def predict_values(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        """Value for bootstrap targets, continuing the correct hidden state when recurrent."""

        if self.critic_is_independent_gru:
            values, _critic_states = self._independent_gru_forward_collection(obs, lstm_states, episode_starts)
            return values
        del lstm_states, episode_starts
        return self.value_net(self._critic_observation(obs))

    def evaluate_actor_actions(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        actions: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        rollout_buffer = self._end2race_rollout_buffer
        valid_by_timestep = rollout_buffer.current_valid_by_timestep
        mean_actions, _actor_states = self._actor_replay_batched(
            obs,
            lstm_states.pi,
            episode_starts,
            valid_by_timestep,
        )
        speed_log_std = rollout_buffer.current_speed_log_stds
        distribution = self._distribution(
            mean_actions,
            speed_log_std=speed_log_std,
        )
        return distribution.log_prob(actions), distribution.entropy()

    def evaluate_independent_actor_actions(self, observations: torch.Tensor, actions: torch.Tensor, actor_hidden: torch.Tensor, episode_starts: torch.Tensor) -> torch.Tensor:
        if self.speed_exploration_mode != "baseline":
            raise RuntimeError("Independent online branch replay requires baseline exploration")
        if observations.ndim != 2 or actions.shape != (observations.shape[0], END2RACE_ACTION_SIZE) or actor_hidden.ndim != 3 or actor_hidden.shape[1] != observations.shape[0] or episode_starts.shape != (observations.shape[0],):
            raise RuntimeError("Independent online branch actor replay shapes are invalid")
        means, _states = self._actor_replay_collection_equivalent(observations, (actor_hidden, torch.zeros_like(actor_hidden)), episode_starts, None)
        return self._distribution(means).log_prob(actions)

    def evaluate_values(self, critic_inputs: torch.Tensor) -> torch.Tensor:
        """Feed-forward critic values over observation rows or stored feature rows."""

        if self.critic_is_independent_gru:
            raise RuntimeError("Independent-GRU critic values require evaluate_values_independent_gru")
        return self.value_net(self._critic_observation(critic_inputs))

    def evaluate_values_independent_gru(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        valid_by_timestep: tuple[tuple[bool, ...], ...] | None,
    ) -> torch.Tensor:
        """Replay the recurrent critic over padded sequences, one FP32 call per timestep."""

        if not self.critic_is_independent_gru:
            raise RuntimeError("evaluate_values_independent_gru requires an independent-GRU critic variant")
        hidden, dummy_cell = states
        full_obs = obs["actor"] if isinstance(obs, dict) else obs
        if full_obs.ndim == 1:
            full_obs = full_obs.unsqueeze(0)
        if full_obs.dtype != torch.float32 or hidden.dtype != torch.float32 or dummy_cell.dtype != torch.float32 or episode_starts.dtype != torch.float32:
            raise RuntimeError("Recurrent critic replay tensors must remain float32")
        if hidden.shape != dummy_cell.shape or hidden.ndim != 3:
            raise RuntimeError("Critic replay hidden and dummy cell shapes must match and be rank 3")
        if full_obs.device != hidden.device or episode_starts.device != full_obs.device:
            raise RuntimeError("Critic replay tensors must share one device")
        if full_obs.is_cuda and (torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest" or torch.backends.cudnn.benchmark):
            raise RuntimeError("CUDA critic replay requires TF32 off, highest FP32 precision, and cuDNN benchmark off")

        n_seq = hidden.shape[1]
        expected_observation_size = END2RACE_OBSERVATION_SIZE + (
            PRIVILEGED_FEATURE_SIZE if self.critic_variant == "privilege_gru" else 0
        )
        if n_seq <= 0 or full_obs.ndim != 2 or full_obs.shape[1] != expected_observation_size or full_obs.shape[0] % n_seq != 0:
            raise RuntimeError(f"Invalid critic replay layout: observations={tuple(full_obs.shape)}, sequences={n_seq}")
        max_length = full_obs.shape[0] // n_seq
        if episode_starts.numel() != full_obs.shape[0]:
            raise RuntimeError("Critic replay episode starts must match observation rows")
        if valid_by_timestep is not None and (len(valid_by_timestep) != max_length or any(len(row) != n_seq for row in valid_by_timestep)):
            raise RuntimeError("Critic replay padding mask does not match the padded batch")

        obs_sequence = full_obs.reshape(n_seq, max_length, expected_observation_size).swapaxes(0, 1)
        start_sequence = episode_starts.reshape(n_seq, max_length).swapaxes(0, 1)
        values: list[torch.Tensor] = []
        for timestep, (step_obs, episode_start) in enumerate(zip(obs_sequence, start_sequence)):
            hidden = hidden * (1.0 - episode_start).view(1, n_seq, 1)
            active = list(range(n_seq)) if valid_by_timestep is None else [index for index, valid in enumerate(valid_by_timestep[timestep]) if valid]
            next_by_slot = [hidden[:, index : index + 1] for index in range(n_seq)]
            values_by_slot = [torch.zeros((1, 1), dtype=full_obs.dtype, device=full_obs.device) for _ in range(n_seq)]
            if active:
                indices = torch.as_tensor(active, dtype=torch.long, device=full_obs.device)
                recurrent_inputs = (
                    step_obs[indices, :END2RACE_LIDAR_SIZE].unsqueeze(1),
                    step_obs[indices, END2RACE_LIDAR_SIZE:END2RACE_OBSERVATION_SIZE].unsqueeze(1),
                    hidden[:, indices],
                )
                if self.critic_variant == "privilege_gru":
                    step_values, next_hidden = self.value_net.step(
                        *recurrent_inputs,
                        step_obs[indices, END2RACE_OBSERVATION_SIZE:],
                    )
                else:
                    step_values, next_hidden = self.value_net.step(*recurrent_inputs)
                for offset, slot in enumerate(active):
                    next_by_slot[slot] = next_hidden[:, offset : offset + 1]
                    values_by_slot[slot] = step_values[offset : offset + 1]
            hidden = torch.cat(next_by_slot, dim=1)
            values.append(torch.cat(values_by_slot, dim=0))
        return torch.stack(values).transpose(0, 1).reshape(-1, 1)

    def actor_checkpoint_state_dict(self) -> dict[str, torch.Tensor]:
        return self.end2race_actor.state_dict()


def end2race_observation(lidar: np.ndarray, previous_ego_speed: float) -> np.ndarray:
    """Build ``[360 LiDAR values, previous ego speed]`` for the actor."""

    lidar = np.asarray(lidar, dtype=np.float32).reshape(-1)
    if lidar.size != END2RACE_LIDAR_SIZE:
        raise ValueError(f"Expected {END2RACE_LIDAR_SIZE} LiDAR values, got {lidar.size}")
    return np.concatenate((lidar, np.asarray([previous_ego_speed], dtype=np.float32)))
