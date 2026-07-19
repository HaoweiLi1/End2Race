"""End2Race actor adapter, fixed exploration distribution, and C0 critic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from model import End2Race
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.distributions import Distribution


END2RACE_OBSERVATION_SIZE = 361
END2RACE_LIDAR_SIZE = 360
END2RACE_ACTION_SIZE = 2
STEERING_BOUND = 0.52
NOOP_SPEED_BOUND = float(np.finfo(np.float32).max)
STEERING_LATENT_STD = 0.03
SPEED_PHYSICAL_STD = 0.15


class EvaluatorCompatibleJointDistribution(Distribution):
    """Squashed latent steering Gaussian plus physical speed Gaussian."""

    def __init__(self, steer_bound: float = STEERING_BOUND, inverse_tanh_epsilon: float = 1e-6):
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

    def proba_distribution(self, raw_mean_actions: torch.Tensor, log_std: torch.Tensor) -> "EvaluatorCompatibleJointDistribution":
        normalized_mode = (raw_mean_actions[:, 0] / self.steer_bound).clamp(-1.0 + self.inverse_tanh_epsilon, 1.0 - self.inverse_tanh_epsilon)
        self.raw_mean_actions = raw_mean_actions
        self.latent_steer_mean = self._atanh(normalized_mode)
        std = log_std.exp()
        self.steer_distribution = torch.distributions.Normal(self.latent_steer_mean, std[0])
        self.speed_distribution = torch.distributions.Normal(raw_mean_actions[:, 1], std[1])
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


class Critic(nn.Module):
    """Single-frame C0 critic fixed by the PPO training plan."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(END2RACE_OBSERVATION_SIZE, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class End2RaceGRUPolicy(RecurrentActorCriticPolicy):
    """Use the original End2Race actor unchanged inside recurrent PPO."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        checkpoint_path: str | Path,
        hidden_scale: int = 4,
        gru_learning_rate: float = 1.0e-6,
        head_learning_rate: float = 1.0e-5,
        critic_learning_rate: float = 3.0e-4,
        **kwargs: Any,
    ):
        kwargs.pop("use_sde", None)
        super().__init__(observation_space, action_space, lr_schedule, net_arch=[], ortho_init=False, use_sde=False, log_std_init=0.0, lstm_hidden_size=1, n_lstm_layers=1, shared_lstm=False, enable_critic_lstm=False, **kwargs)

        self.end2race_actor = End2Race(mask_prob=0.0, hidden_scale=hidden_scale)
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        self.end2race_actor.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.pretrained_model_path = str(checkpoint)

        self.lstm_actor = GRUWithLSTMStateInterface(self.end2race_actor.gru)
        self.lstm_output_dim = self.end2race_actor.gru.hidden_size
        self.lstm_hidden_state_shape = (self.end2race_actor.gru.num_layers, 1, self.end2race_actor.gru.hidden_size)
        self.lstm_critic = None
        self.critic = None
        self.value_net = Critic()
        self.action_net = nn.Identity()
        self.action_dist = EvaluatorCompatibleJointDistribution()

        self.log_std.data.copy_(torch.tensor([np.log(STEERING_LATENT_STD), np.log(SPEED_PHYSICAL_STD)], dtype=self.log_std.dtype, device=self.log_std.device))
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

    @staticmethod
    def _actor_observation(obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return obs["actor"] if isinstance(obs, dict) else obs

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

    def _distribution(self, mean_actions: torch.Tensor) -> EvaluatorCompatibleJointDistribution:
        if mean_actions.dtype != torch.float32 or self.log_std.dtype != torch.float32:
            raise RuntimeError("PPO actor distribution tensors must remain float32")
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def _critic_values(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        actor_obs = self._actor_observation(obs).float()
        if actor_obs.ndim == 1:
            actor_obs = actor_obs.unsqueeze(0)
        return self.value_net(actor_obs)

    @staticmethod
    def _zero_states(states: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros_like(states[0])
        return zero, zero.clone()

    def supports_actor_hidden_only_buffer(self) -> bool:
        return isinstance(self.lstm_actor, GRUWithLSTMStateInterface) and self.lstm_critic is None and self.critic is None

    def forward(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
        mean_actions, actor_states = self._actor_forward(obs, lstm_states.pi, episode_starts)
        distribution = self._distribution(mean_actions)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self._critic_values(obs), distribution.log_prob(actions), RNNStates(actor_states, self._zero_states(lstm_states.vf))

    def get_distribution(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[Distribution, tuple[torch.Tensor, torch.Tensor]]:
        mean_actions, actor_states = self._actor_forward(obs, lstm_states, episode_starts)
        return self._distribution(mean_actions), actor_states

    def predict_values(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        del lstm_states, episode_starts
        return self._critic_values(obs)

    def evaluate_actor_actions(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        actions: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        rollout_buffer = getattr(self, "_actor_hidden_rollout_buffer", None)
        valid_by_timestep = None if rollout_buffer is None else rollout_buffer.current_valid_by_timestep
        mean_actions, _actor_states = self._actor_replay_batched(obs, lstm_states.pi, episode_starts, valid_by_timestep)
        distribution = self._distribution(mean_actions)
        return distribution.log_prob(actions), distribution.entropy()

    def evaluate_values(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self._critic_values(obs)

    def evaluate_actions(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        actions: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        log_prob, entropy = self.evaluate_actor_actions(obs, actions, lstm_states, episode_starts)
        return self.evaluate_values(obs), log_prob, entropy

    def actor_checkpoint_state_dict(self) -> dict[str, torch.Tensor]:
        return self.end2race_actor.state_dict()


def end2race_observation(lidar: np.ndarray, previous_ego_speed: float) -> np.ndarray:
    """Build ``[360 LiDAR values, previous ego speed]`` for the actor."""

    lidar = np.asarray(lidar, dtype=np.float32).reshape(-1)
    if lidar.size != END2RACE_LIDAR_SIZE:
        raise ValueError(f"Expected {END2RACE_LIDAR_SIZE} LiDAR values, got {lidar.size}")
    return np.concatenate((lidar, np.asarray([previous_ego_speed], dtype=np.float32)))
