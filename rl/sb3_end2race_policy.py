"""SB3-Contrib recurrent policy adapter for the unchanged End2Race GRU actor.

This is a proof-of-concept integration layer.  It keeps the original End2Race
module and checkpoint schema intact while presenting the recurrent state shape
expected by sb3-contrib's stock RecurrentPPO and RecurrentRolloutBuffer.
"""

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
from stable_baselines3.common.torch_layers import CombinedExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BC_CHECKPOINT = PROJECT_ROOT / "pretrained" / "end2race.pth"
END2RACE_OBSERVATION_SIZE = 361
END2RACE_LIDAR_SIZE = 360
END2RACE_ACTION_SIZE = 2
EVALUATOR_STEER_BOUND = 0.52
NOOP_SPEED_BOUND = float(np.finfo(np.float32).max)
PPO_V1_STEER_LOG_STD = -2.995732273553991
PPO_V1_SPEED_LOG_STD = -1.8971199848858813
PPO_V1_GRU_LR = 1e-6
PPO_V1_HEAD_LR = 1e-5
PPO_V1_CRITIC_LR = 3e-4
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
        if steer_bound <= 0:
            raise ValueError("steer_bound must be positive")
        if not 0 < inverse_tanh_epsilon < 1e-3:
            raise ValueError("inverse_tanh_epsilon must be in (0, 1e-3)")
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
        if raw_mean_actions.shape[-1] != END2RACE_ACTION_SIZE:
            raise ValueError(f"Expected two raw action means, got {tuple(raw_mean_actions.shape)}")
        if tuple(log_std.shape) != (END2RACE_ACTION_SIZE,):
            raise ValueError(f"Expected two log standard deviations, got {tuple(log_std.shape)}")
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
        if (
            self.raw_mean_actions is None
            or self.latent_steer_mean is None
            or self.steer_distribution is None
            or self.speed_distribution is None
        ):
            raise RuntimeError("proba_distribution() must be called first")
        return (
            self.raw_mean_actions,
            self.latent_steer_mean,
            self.steer_distribution,
            self.speed_distribution,
        )

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        _raw_means, _latent_mean, steer_distribution, speed_distribution = self._require_parameters()
        if actions.shape[-1] != END2RACE_ACTION_SIZE:
            raise ValueError(f"Expected physical [steering, speed] actions, got {tuple(actions.shape)}")
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


class GRUWithLSTMStateInterface(nn.Module):
    """Expose a batch-first GRU through SB3's time-major ``(h, c)`` API.

    ``h`` is the only real recurrent state. ``c`` is shape-compatible transport
    data and is ignored on input and returned as zeros on output.
    """

    def __init__(self, gru: nn.GRU):
        super().__init__()
        if not gru.batch_first:
            raise ValueError("End2Race GRU must be batch_first=True")
        if gru.bidirectional:
            raise ValueError("Bidirectional GRU is not supported by RecurrentPPO")
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

        hidden, cell = states
        expected = (self.num_layers, x.shape[1], self.hidden_size)
        if tuple(hidden.shape) != expected or tuple(cell.shape) != expected:
            raise ValueError(f"Expected recurrent states {expected}, got h={tuple(hidden.shape)}, c={tuple(cell.shape)}")
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
        checkpoint_path: str | Path = DEFAULT_BC_CHECKPOINT,
        hidden_scale: int = 4,
        critic_hidden_size: int = 64,
        steer_log_std_init: float = -2.0,
        speed_log_std_init: float = 0.0,
        steer_bound: float = EVALUATOR_STEER_BOUND,
        inverse_tanh_epsilon: float = 1e-6,
        optimizer_profile: str = "default",
        critic_profile: str = "C0_RAW_SINGLE_FRAME",
        gru_lr: float = PPO_V1_GRU_LR,
        head_lr: float = PPO_V1_HEAD_LR,
        critic_lr: float = PPO_V1_CRITIC_LR,
        steering_latent_std: float | None = None,
        speed_physical_std: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.critic_profile = str(critic_profile)
        if self.critic_profile not in CRITIC_PROFILES:
            raise ValueError(f"Unknown PPO V1.2 critic profile: {self.critic_profile}")
        if self.critic_profile == "C3_PRIVILEGED_PHYSICAL":
            valid_observation = (
                isinstance(observation_space, spaces.Dict)
                and set(observation_space.spaces) == {"actor", "critic"}
                and observation_space["actor"].shape == (END2RACE_OBSERVATION_SIZE,)
                and observation_space["critic"].shape == (12,)
            )
            if not valid_observation:
                raise ValueError("C3 requires Dict(actor=361D, critic=12D) observation space")
            kwargs.setdefault("features_extractor_class", CombinedExtractor)
        elif not isinstance(observation_space, spaces.Box) or observation_space.shape != (END2RACE_OBSERVATION_SIZE,):
            raise ValueError(f"Expected Box observation shape ({END2RACE_OBSERVATION_SIZE},), got {observation_space}")
        if not isinstance(action_space, spaces.Box) or action_space.shape != (END2RACE_ACTION_SIZE,):
            raise ValueError(f"Expected Box action shape ({END2RACE_ACTION_SIZE},), got {action_space}")
        if not np.allclose(action_space.low[0], -steer_bound) or not np.allclose(action_space.high[0], steer_bound):
            raise ValueError(f"Steering action bounds must be [-{steer_bound}, {steer_bound}]")
        if action_space.low[1] > -0.99 * NOOP_SPEED_BOUND or action_space.high[1] < 0.99 * NOOP_SPEED_BOUND:
            raise ValueError("Speed action bounds must use the float32 no-op range required by stock SB3")
        if kwargs.pop("use_sde", False):
            raise ValueError("This POC uses a repo-local transformed joint distribution, not gSDE")

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
        if not isinstance(state_dict, dict):
            raise TypeError(f"Expected a raw End2Race state_dict in {checkpoint}")
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

        if tuple(self.log_std.shape) != (END2RACE_ACTION_SIZE,):
            raise ValueError(f"Unexpected log_std shape: {tuple(self.log_std.shape)}")
        self.log_std.data.copy_(
            torch.tensor(
                [steer_log_std_init, speed_log_std_init],
                dtype=self.log_std.dtype,
                device=self.log_std.device,
            )
        )
        self.log_std.requires_grad_(True)
        if steering_latent_std is not None:
            if steering_latent_std <= 0.0:
                raise ValueError("steering_latent_std must be positive")
            self.log_std.data[0] = float(np.log(steering_latent_std))
        if speed_physical_std is not None:
            if speed_physical_std <= 0.0:
                raise ValueError("speed_physical_std must be positive")
            self.log_std.data[1] = float(np.log(speed_physical_std))
        self.action_dist = EvaluatorCompatibleJointDistribution(
            steer_bound=steer_bound,
            inverse_tanh_epsilon=inverse_tanh_epsilon,
        )
        # The parent head is not used by any custom actor path.  Replacing it
        # with a parameter-free module prevents dead parameters in the optimizer.
        self.action_net = nn.Identity()
        self.last_raw_actor_mean: torch.Tensor | None = None
        self.optimizer_profile = str(optimizer_profile)
        self.gru_lr = float(gru_lr)
        self.head_lr = float(head_lr)
        self.critic_lr = float(critic_lr)

        # Rebuild with an explicit, auditable partition.  The shared GRU module
        # exposed by lstm_actor is already present in end2race_actor parameters.
        if self.optimizer_profile == "default":
            optimizer_parameters = [
                *self.end2race_actor.parameters(),
                *self.value_net.parameters(),
                self.log_std,
            ]
            if len({id(parameter) for parameter in optimizer_parameters}) != len(optimizer_parameters):
                raise RuntimeError("Optimizer parameter identities must be unique")
            self.optimizer = self.optimizer_class(
                optimizer_parameters,
                lr=lr_schedule(1),
                **self.optimizer_kwargs,
            )
        elif self.optimizer_profile == "ppo_v1":
            for parameter in self.end2race_actor.parameters():
                parameter.requires_grad_(False)
            for parameter in self.end2race_actor.gru.parameters():
                parameter.requires_grad_(True)
            for parameter in self.end2race_actor.output_layer.parameters():
                parameter.requires_grad_(True)
            for parameter in self.value_net.parameters():
                parameter.requires_grad_(True)
            if steering_latent_std is None and speed_physical_std is None:
                self.log_std.data.copy_(
                    torch.tensor(
                        [PPO_V1_STEER_LOG_STD, PPO_V1_SPEED_LOG_STD],
                        dtype=self.log_std.dtype,
                        device=self.log_std.device,
                    )
                )
            self.log_std.requires_grad_(False)

            gru_parameters = list(self.end2race_actor.gru.parameters())
            head_parameters = list(self.end2race_actor.output_layer.parameters())
            critic_parameters = list(self.value_net.parameters())
            groups = [
                {
                    "params": gru_parameters,
                    "lr": self.gru_lr,
                    "name": "gru",
                    "base_lr": self.gru_lr,
                },
                {
                    "params": head_parameters,
                    "lr": self.head_lr,
                    "name": "head",
                    "base_lr": self.head_lr,
                },
                {
                    "params": critic_parameters,
                    "lr": self.critic_lr,
                    "name": "critic",
                    "base_lr": self.critic_lr,
                },
            ]
            group_ids = [id(parameter) for group in groups for parameter in group["params"]]
            expected_ids = {
                id(parameter)
                for parameter in (*gru_parameters, *head_parameters, *critic_parameters)
            }
            if len(group_ids) != len(set(group_ids)):
                raise RuntimeError("PPO V1 optimizer groups overlap")
            if set(group_ids) != expected_ids:
                raise RuntimeError("PPO V1 optimizer groups do not exactly cover GRU, head, and critic")
            self.optimizer = self.optimizer_class(
                groups,
                lr=lr_schedule(1),
                **self.optimizer_kwargs,
            )
        else:
            raise ValueError(f"Unknown End2Race optimizer profile: {self.optimizer_profile}")

    @property
    def actor_hidden_size(self) -> int:
        return self.end2race_actor.gru.hidden_size

    def _validate_actor_states(
        self,
        states: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = states
        if hidden.ndim != 3 or cell.shape != hidden.shape:
            raise ValueError(f"Expected matching [layers, n_seq, hidden] states, got {hidden.shape}, {cell.shape}")
        if hidden.shape[0] != self.end2race_actor.gru.num_layers or hidden.shape[2] != self.actor_hidden_size:
            raise ValueError(f"State shape is incompatible with End2Race GRU: {tuple(hidden.shape)}")
        return hidden, cell

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

        hidden, _dummy_cell = self._validate_actor_states(states)
        obs = self._actor_observation(obs)
        obs = obs.float()
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] != END2RACE_OBSERVATION_SIZE:
            raise ValueError(f"Expected observation width {END2RACE_OBSERVATION_SIZE}, got {obs.shape[-1]}")

        n_seq = hidden.shape[1]
        if obs.shape[0] % n_seq != 0:
            raise ValueError(f"Flat observation batch {obs.shape[0]} is not divisible by n_seq={n_seq}")
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
            if actor_features is None:
                raise ValueError("C2 value prediction requires current actor hidden features")
            return actor_features.detach()
        if not isinstance(obs, dict) or set(obs) != {"actor", "critic"}:
            raise ValueError("C3 value prediction requires isolated actor/critic Dict fields")
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
        self.last_raw_actor_mean = mean_actions.detach().clone()
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

    def critic_telemetry(self, obs: torch.Tensor | dict[str, torch.Tensor], actor_features: torch.Tensor | None = None) -> dict[str, float]:
        """Return finite scale/saturation statistics without changing gradients."""

        with torch.no_grad():
            critic_input = self._critic_input(obs, actor_features)
            first = self.value_net[0]
            if isinstance(first, nn.LayerNorm):
                normalized = first(critic_input)
                preactivation = self.value_net[1](normalized)
                activation = self.value_net[2](preactivation)
            else:
                preactivation = first(critic_input)
                activation = self.value_net[1](preactivation)
            result = {
                "critic_input_mean": float(critic_input.mean()),
                "critic_input_std": float(critic_input.std(unbiased=False)),
                "first_layer_preactivation_mean": float(preactivation.mean()),
                "first_layer_preactivation_std": float(preactivation.std(unbiased=False)),
                "first_layer_preactivation_max_abs": float(preactivation.abs().max()),
                "preactivation_abs_gt_3_fraction": float((preactivation.abs() > 3.0).float().mean()),
                "activation_saturation_fraction": float((activation.abs() > 0.99).float().mean()) if self.critic_profile == "C0_RAW_SINGLE_FRAME" else 0.0,
            }
            if self.critic_profile == "C1_FROZEN_BC_FEATURE":
                result["frozen_speed_embedding_max_abs"] = float(critic_input[:, 360:].abs().max())
            return result

    def actor_checkpoint_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only the unchanged BC-compatible End2Race actor schema."""

        return self.end2race_actor.state_dict()


def end2race_observation(lidar: np.ndarray, previous_ego_speed: float) -> np.ndarray:
    """Construct the only actor observation admitted by this POC."""

    lidar = np.asarray(lidar, dtype=np.float32).reshape(-1)
    if lidar.size != END2RACE_LIDAR_SIZE:
        raise ValueError(f"Expected {END2RACE_LIDAR_SIZE} LiDAR values, got {lidar.size}")
    return np.concatenate((lidar, np.asarray([previous_ego_speed], dtype=np.float32)))
