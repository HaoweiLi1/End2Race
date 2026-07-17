"""Recurrent PPO policy for the unchanged End2Race GRU actor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from model import End2Race
from ppo.config import (
    BC_CHECKPOINT,
    CRITIC_LR,
    GRU_LR,
    HEAD_LR,
    SPEED_PHYSICAL_STD,
    STEERING_LATENT_STD,
)
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.torch_layers import CombinedExtractor
from torch.optim import Optimizer


END2RACE_OBSERVATION_SIZE = 361
END2RACE_LIDAR_SIZE = 360
END2RACE_ACTION_SIZE = 2
EVALUATOR_STEER_BOUND = 0.52
NOOP_SPEED_BOUND = float(np.finfo(np.float32).max)
CRITIC_PROFILES = (
    "C0_RAW_SINGLE_FRAME",
    "C1_FROZEN_BC_FEATURE",
    "C2_DETACHED_ACTOR_HIDDEN",
    "C3_PRIVILEGED_PHYSICAL",
)


class EvaluatorCompatibleJointDistribution(Distribution):
    """Physical ``[steering, speed]`` distribution matching deployment.

    Steering is a scaled tanh transform of a Gaussian latent.  Speed remains a
    Gaussian in physical m/s.  The supplied means are the unchanged raw
    End2Race outputs; the deterministic steering mode matches evaluator
    clipping up to ``steer_bound * inverse_tanh_epsilon``.
    """

    def __init__(self, steer_bound: float = EVALUATOR_STEER_BOUND, inverse_tanh_epsilon: float = 1e-6):
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
    ) -> "EvaluatorCompatibleJointDistribution":
        normalized_mode = (raw_mean_actions[:, 0] / self.steer_bound).clamp(
            -1.0 + self.inverse_tanh_epsilon,
            1.0 - self.inverse_tanh_epsilon,
        )
        latent_steer_mean = self._atanh(normalized_mode)
        std = log_std.exp()
        self.raw_mean_actions = raw_mean_actions
        self.latent_steer_mean = latent_steer_mean
        self.steer_distribution = torch.distributions.Normal(latent_steer_mean, std[0])
        self.speed_distribution = torch.distributions.Normal(raw_mean_actions[:, 1], std[1])
        self.distribution = (self.steer_distribution, self.speed_distribution)
        return self

    def _require_parameters(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.distributions.Normal, torch.distributions.Normal]:
        return (
            self.raw_mean_actions,
            self.latent_steer_mean,
            self.steer_distribution,
            self.speed_distribution,
        )

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        _raw_means, _latent_mean, steer_distribution, speed_distribution = self._require_parameters()
        # The larger configured epsilon is only for constructing a finite
        # deterministic latent mean at evaluator clipping boundaries.  Replay
        # inversion must retain every representable interior physical action.
        action_epsilon = torch.finfo(actions.dtype).eps
        normalized_steer = (actions[:, 0] / self.steer_bound).clamp(
            -1.0 + action_epsilon,
            1.0 - action_epsilon,
        )
        latent_steer = self._atanh(normalized_steer)
        # y = steer_bound * tanh(z), so log|dy/dz| is the sum below.
        log_abs_det_jacobian = torch.log(
            torch.as_tensor(self.steer_bound, dtype=actions.dtype, device=actions.device)
        ) + torch.log1p(-normalized_steer.square())
        steer_log_prob = steer_distribution.log_prob(latent_steer) - log_abs_det_jacobian
        speed_log_prob = speed_distribution.log_prob(actions[:, 1])
        return steer_log_prob + speed_log_prob

    def entropy(self) -> None:
        return None

    def sample(self) -> torch.Tensor:
        _raw_means, _latent_mean, steer_distribution, speed_distribution = self._require_parameters()
        steering = self.steer_bound * torch.tanh(steer_distribution.rsample())
        speed = speed_distribution.rsample()
        return torch.stack((steering, speed), dim=1)

    def mode(self) -> torch.Tensor:
        raw_means, latent_mean, _steer_distribution, _speed_distribution = self._require_parameters()
        steering = self.steer_bound * torch.tanh(latent_mean)
        return torch.stack((steering, raw_means[:, 1]), dim=1)

    def actions_from_params(
        self,
        raw_mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        return self.proba_distribution(raw_mean_actions, log_std).get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        raw_mean_actions: torch.Tensor,
        log_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(raw_mean_actions, log_std)
        return actions, self.log_prob(actions)


class EvaluatorClippedPhysicalGaussianDistribution(Distribution):
    """Use a regular physical steering Gaussian and let SB3 clip for the env.

    The End2Race actor already emits steering in physical radians and the
    evaluator clips that raw output to ``[-steer_bound, steer_bound]``.  SB3
    stores the unclipped Gaussian sample in its rollout buffer and clips only
    the action sent to the environment, so replay likelihood remains exact.

    The physical steering standard deviation is ``steer_bound * latent_std``.
    This preserves the squashed distribution's local exploration scale around
    zero while removing the singular ``atanh(raw_mean / steer_bound)`` map at
    the evaluator boundary.
    """

    def __init__(self, steer_bound: float = EVALUATOR_STEER_BOUND):
        super().__init__()
        self.steer_bound = float(steer_bound)
        self.raw_mean_actions: torch.Tensor | None = None
        self.steer_distribution: torch.distributions.Normal | None = None
        self.speed_distribution: torch.distributions.Normal | None = None

    def proba_distribution_net(self, *args: Any, **kwargs: Any) -> nn.Module:
        del args, kwargs
        return nn.Identity()

    def proba_distribution(
        self,
        raw_mean_actions: torch.Tensor,
        log_std: torch.Tensor,
    ) -> "EvaluatorClippedPhysicalGaussianDistribution":
        std = log_std.exp()
        self.raw_mean_actions = raw_mean_actions
        self.steer_distribution = torch.distributions.Normal(
            raw_mean_actions[:, 0],
            std[0] * self.steer_bound,
        )
        self.speed_distribution = torch.distributions.Normal(raw_mean_actions[:, 1], std[1])
        self.distribution = (self.steer_distribution, self.speed_distribution)
        return self

    def _require_parameters(
        self,
    ) -> tuple[torch.Tensor, torch.distributions.Normal, torch.distributions.Normal]:
        return self.raw_mean_actions, self.steer_distribution, self.speed_distribution

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        _raw_means, steer_distribution, speed_distribution = self._require_parameters()
        return steer_distribution.log_prob(actions[:, 0]) + speed_distribution.log_prob(actions[:, 1])

    def entropy(self) -> None:
        return None

    def sample(self) -> torch.Tensor:
        _raw_means, steer_distribution, speed_distribution = self._require_parameters()
        return torch.stack((steer_distribution.rsample(), speed_distribution.rsample()), dim=1)

    def mode(self) -> torch.Tensor:
        raw_means, _steer_distribution, _speed_distribution = self._require_parameters()
        return raw_means

    def actions_from_params(
        self,
        raw_mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        return self.proba_distribution(raw_mean_actions, log_std).get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        raw_mean_actions: torch.Tensor,
        log_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(raw_mean_actions, log_std)
        return actions, self.log_prob(actions)


class GRUWithLSTMStateInterface(nn.Module):
    """Expose a batch-first GRU through SB3's time-major ``(h, c)`` API.

    ``h`` is the only real recurrent state. ``c`` is shape-compatible transport
    data and is ignored on input and returned as zeros on output.
    """

    def __init__(self, gru: nn.GRU):
        super().__init__()
        self.gru = gru
        self.input_size = gru.input_size
        self.hidden_size = gru.hidden_size
        self.num_layers = gru.num_layers

    def forward(
        self,
        x: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run time-major ``x`` like ``nn.LSTM`` while ignoring dummy ``c``."""

        hidden, _cell = states
        batch_first_output, next_hidden = self.gru(x.transpose(0, 1), hidden)
        time_major_output = batch_first_output.transpose(0, 1)
        return time_major_output, (next_hidden, torch.zeros_like(next_hidden))


class End2RaceGRUPolicy(RecurrentActorCriticPolicy):
    """Use the original End2Race GRU actor with stock SB3 recurrent PPO.

    Observation layout is ``[lidar_0, ..., lidar_359, previous_ego_speed]``.
    The Gaussian mean is exactly the two-dimensional output of ``End2Race``.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        checkpoint_path: str | Path = BC_CHECKPOINT,
        hidden_scale: int = 4,
        critic_hidden_size: int = 64,
        steer_bound: float = EVALUATOR_STEER_BOUND,
        inverse_tanh_epsilon: float = 1e-6,
        critic_profile: str = "C0_RAW_SINGLE_FRAME",
        gru_lr: float = GRU_LR,
        head_lr: float = HEAD_LR,
        steering_distribution: str = "squashed_latent",
        steering_latent_std: float = STEERING_LATENT_STD,
        speed_physical_std: float = SPEED_PHYSICAL_STD,
        **kwargs: Any,
    ) -> None:
        self.critic_profile = str(critic_profile)
        if self.critic_profile not in CRITIC_PROFILES:
            raise ValueError(f"Unknown critic profile: {self.critic_profile}")
        if self.critic_profile == "C3_PRIVILEGED_PHYSICAL":
            kwargs["features_extractor_class"] = CombinedExtractor
        # SB3 always injects use_sde; End2Race uses its own transformed joint
        # distribution, so drop the injected copy before the parent constructor.
        kwargs.pop("use_sde", None)

        # The parent's tiny placeholder recurrent module is replaced below.  A
        # size of one avoids allocating an unused 1680-wide LSTM during setup.
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            ortho_init=False,
            use_sde=False,
            log_std_init=0.0,
            lstm_hidden_size=1,
            n_lstm_layers=1,
            shared_lstm=False,
            enable_critic_lstm=False,
            **kwargs,
        )

        self.end2race_actor = End2Race(mask_prob=0.0, hidden_scale=hidden_scale)
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.end2race_actor.load_state_dict(state_dict, strict=True)
        self.bc_checkpoint_path = str(checkpoint)

        # RecurrentPPO reads num_layers/hidden_size from policy.lstm_actor to
        # size its stock recurrent rollout buffer.
        self.lstm_actor = GRUWithLSTMStateInterface(self.end2race_actor.gru)
        self.lstm_output_dim = self.end2race_actor.gru.hidden_size
        self.lstm_hidden_state_shape = (
            self.end2race_actor.gru.num_layers,
            1,
            self.end2race_actor.gru.hidden_size,
        )

        # The critic is deliberately independent and feed-forward.  Its RNNStates
        # entries remain zero transport tensors and never affect actor output.
        self.lstm_critic = None
        self.critic = None
        if self.critic_profile == "C0_RAW_SINGLE_FRAME":
            self.value_net = nn.Sequential(
                nn.Linear(END2RACE_OBSERVATION_SIZE, critic_hidden_size),
                nn.Tanh(),
                nn.Linear(critic_hidden_size, 1),
            )
        elif self.critic_profile == "C1_FROZEN_BC_FEATURE":
            self.value_net = nn.Sequential(
                nn.Linear(420, 128),
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.SiLU(),
                nn.Linear(128, 1),
            )
        elif self.critic_profile == "C2_DETACHED_ACTOR_HIDDEN":
            self.value_net = nn.Sequential(
                nn.LayerNorm(self.lstm_output_dim),
                nn.Linear(self.lstm_output_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 1),
            )
        else:
            self.value_net = nn.Sequential(
                nn.Linear(12, 128),
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.SiLU(),
                nn.Linear(128, 1),
            )

        self.log_std.data.copy_(
            torch.tensor(
                [np.log(steering_latent_std), np.log(speed_physical_std)],
                dtype=self.log_std.dtype,
                device=self.log_std.device,
            )
        )
        self.log_std.requires_grad_(False)
        self.steering_distribution = str(steering_distribution)
        if self.steering_distribution == "squashed_latent":
            self.action_dist = EvaluatorCompatibleJointDistribution(
                steer_bound=steer_bound,
                inverse_tanh_epsilon=inverse_tanh_epsilon,
            )
        elif self.steering_distribution == "physical_gaussian":
            self.action_dist = EvaluatorClippedPhysicalGaussianDistribution(steer_bound=steer_bound)
        else:
            raise ValueError(f"Unknown steering distribution: {self.steering_distribution}")
        # The parent head is not used by any custom actor path.  Replacing it
        # with a parameter-free module prevents dead parameters in the optimizer.
        self.action_net = nn.Identity()
        for parameter in self.end2race_actor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.end2race_actor.gru.parameters():
            parameter.requires_grad_(True)
        for parameter in self.end2race_actor.output_layer.parameters():
            parameter.requires_grad_(True)
        for parameter in self.value_net.parameters():
            parameter.requires_grad_(True)

        gru_parameters = list(self.end2race_actor.gru.parameters())
        head_parameters = list(self.end2race_actor.output_layer.parameters())
        critic_parameters = list(self.value_net.parameters())
        groups = [
            {"params": gru_parameters, "lr": gru_lr, "name": "gru", "base_lr": gru_lr},
            {"params": head_parameters, "lr": head_lr, "name": "head", "base_lr": head_lr},
            {"params": critic_parameters, "lr": CRITIC_LR, "name": "critic", "base_lr": CRITIC_LR},
        ]
        self.optimizer = self.optimizer_class(groups, lr=lr_schedule(1), **self.optimizer_kwargs)

    @property
    def actor_hidden_size(self) -> int:
        return self.end2race_actor.gru.hidden_size

    @staticmethod
    def _actor_observation(obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return obs["actor"] if isinstance(obs, dict) else obs

    def _actor_forward(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return actor means, final state, and each timestep's actor hidden."""

        hidden, _dummy_cell = states
        obs = self._actor_observation(obs)
        obs = obs.float()
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        n_seq = hidden.shape[1]
        obs_sequence = obs.reshape(n_seq, -1, END2RACE_OBSERVATION_SIZE).swapaxes(0, 1)
        start_sequence = episode_starts.float().reshape(n_seq, -1).swapaxes(0, 1)

        means: list[torch.Tensor] = []
        timestep_hidden: list[torch.Tensor] = []
        for step_obs, episode_start in zip(obs_sequence, start_sequence):
            # Reset only the env/sequence slots marked as new episodes.
            hidden = hidden * (1.0 - episode_start).view(1, n_seq, 1)
            lidar = step_obs[:, :END2RACE_LIDAR_SIZE].unsqueeze(1)
            previous_speed = step_obs[:, END2RACE_LIDAR_SIZE:].unsqueeze(1)
            # Use the same batch-size-one GRU kernel during collection and
            # replay.  This removes n_seq-dependent BLAS rounding while still
            # preserving a differentiable recurrent graph within each sequence.
            slot_means: list[torch.Tensor] = []
            slot_hidden: list[torch.Tensor] = []
            for sequence_index in range(n_seq):
                action_sequence, next_hidden = self.end2race_actor(
                    lidar[sequence_index : sequence_index + 1],
                    previous_speed[sequence_index : sequence_index + 1],
                    hidden[:, sequence_index : sequence_index + 1],
                )
                slot_means.append(action_sequence[:, -1, :])
                slot_hidden.append(next_hidden)
            hidden = torch.cat(slot_hidden, dim=1)
            means.append(torch.cat(slot_means, dim=0))
            timestep_hidden.append(hidden.squeeze(0))

        mean_actions = torch.stack(means).transpose(0, 1).reshape(-1, END2RACE_ACTION_SIZE)
        actor_features = torch.stack(timestep_hidden).transpose(0, 1).reshape(-1, self.actor_hidden_size)
        return mean_actions, (hidden, torch.zeros_like(hidden)), actor_features

    def actor_mean(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return exact End2Race means and updated ``(real_h, zero_dummy_c)``."""

        means, next_states, _actor_features = self._actor_forward(obs, states, episode_starts)
        return means, next_states

    def _distribution(self, mean_actions: torch.Tensor) -> EvaluatorCompatibleJointDistribution:
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def _critic_input(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        actor_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actor_obs = self._actor_observation(obs).float()
        if actor_obs.ndim == 1:
            actor_obs = actor_obs.unsqueeze(0)
        if self.critic_profile == "C0_RAW_SINGLE_FRAME":
            return actor_obs
        if self.critic_profile == "C1_FROZEN_BC_FEATURE":
            lidar = actor_obs[:, :END2RACE_LIDAR_SIZE]
            speed = actor_obs[:, END2RACE_LIDAR_SIZE:]
            processed_lidar = (-1.0 / (1.0 + torch.exp(-self.end2race_actor.k * lidar)) + 1.0) * 2.0
            speed_embedding = self.end2race_actor.speed_mlp(speed)
            return torch.cat((processed_lidar, speed_embedding), dim=1).detach()
        if self.critic_profile == "C2_DETACHED_ACTOR_HIDDEN":
            return actor_features.detach()
        critic = obs["critic"].float()
        return critic.unsqueeze(0) if critic.ndim == 1 else critic

    def _critic_values(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        actor_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.value_net(self._critic_input(obs, actor_features))

    @staticmethod
    def _zero_vf_states(states: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _cell = states
        zero = torch.zeros_like(hidden)
        return zero, zero.clone()

    def forward(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
        mean_actions, actor_states, actor_features = self._actor_forward(obs, lstm_states.pi, episode_starts)
        distribution = self._distribution(mean_actions)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self._critic_values(obs, actor_features)
        vf_states = (
            tuple(state.detach() for state in actor_states)
            if self.critic_profile == "C2_DETACHED_ACTOR_HIDDEN"
            else self._zero_vf_states(lstm_states.vf)
        )
        return actions, values, log_prob, RNNStates(actor_states, vf_states)

    def get_distribution(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[Distribution, tuple[torch.Tensor, torch.Tensor]]:
        mean_actions, actor_states = self.actor_mean(obs, lstm_states, episode_starts)
        return self._distribution(mean_actions), actor_states

    def predict_values(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        if self.critic_profile == "C2_DETACHED_ACTOR_HIDDEN":
            _means, _states, actor_features = self._actor_forward(obs, lstm_states, episode_starts)
            return self._critic_values(obs, actor_features)
        return self._critic_values(obs)

    def evaluate_actions(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        actions: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        mean_actions, _actor_states, actor_features = self._actor_forward(obs, lstm_states.pi, episode_starts)
        distribution = self._distribution(mean_actions)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self._critic_values(obs, actor_features)
        return values, log_prob, entropy

    def actor_checkpoint_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only the unchanged BC-compatible End2Race actor schema."""

        return self.end2race_actor.state_dict()


def end2race_observation(lidar: np.ndarray, previous_ego_speed: float) -> np.ndarray:
    """Construct the only actor observation admitted by this POC."""

    lidar = np.asarray(lidar, dtype=np.float32).reshape(-1)
    if lidar.size != END2RACE_LIDAR_SIZE:
        raise ValueError(f"Expected {END2RACE_LIDAR_SIZE} LiDAR values, got {lidar.size}")
    return np.concatenate((lidar, np.asarray([previous_ego_speed], dtype=np.float32)))


class End2RaceRecurrentPPO(RecurrentPPO):
    """Keep the three named optimizer-group learning rates fixed."""

    def _update_learning_rate(self, optimizers: list[Optimizer] | Optimizer) -> None:
        for optimizer in optimizers if isinstance(optimizers, list) else [optimizers]:
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"]
