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
from stable_baselines3.common.distributions import DiagGaussianDistribution, Distribution


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BC_CHECKPOINT = PROJECT_ROOT / "pretrained" / "end2race.pth"
END2RACE_OBSERVATION_SIZE = 361
END2RACE_LIDAR_SIZE = 360
END2RACE_ACTION_SIZE = 2


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
        log_std_init: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if not isinstance(observation_space, spaces.Box) or observation_space.shape != (END2RACE_OBSERVATION_SIZE,):
            raise ValueError(f"Expected Box observation shape ({END2RACE_OBSERVATION_SIZE},), got {observation_space}")
        if not isinstance(action_space, spaces.Box) or action_space.shape != (END2RACE_ACTION_SIZE,):
            raise ValueError(f"Expected Box action shape ({END2RACE_ACTION_SIZE},), got {action_space}")
        if kwargs.pop("use_sde", False):
            raise ValueError("This POC uses a diagonal Gaussian distribution, not gSDE")

        # The parent's tiny placeholder recurrent module is replaced below.  A
        # size of one avoids allocating an unused 1680-wide LSTM during setup.
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            ortho_init=False,
            use_sde=False,
            log_std_init=log_std_init,
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
        self.value_net = nn.Sequential(
            nn.Linear(END2RACE_OBSERVATION_SIZE, critic_hidden_size),
            nn.Tanh(),
            nn.Linear(critic_hidden_size, 1),
        )

        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise TypeError(f"Expected DiagGaussianDistribution, got {type(self.action_dist).__name__}")
        if tuple(self.log_std.shape) != (END2RACE_ACTION_SIZE,):
            raise ValueError(f"Unexpected log_std shape: {tuple(self.log_std.shape)}")
        self.log_std.requires_grad_(True)

        # Rebuild because the parent created its optimizer before the original
        # End2Race actor and replacement critic were attached.
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

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

    def actor_mean(
        self,
        obs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return exact End2Race means and updated ``(real_h, zero_dummy_c)``."""

        hidden, _dummy_cell = self._validate_actor_states(states)
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

        mean_actions = torch.stack(means).transpose(0, 1).reshape(-1, END2RACE_ACTION_SIZE)
        return mean_actions, (hidden, torch.zeros_like(hidden))

    def _distribution(self, mean_actions: torch.Tensor) -> DiagGaussianDistribution:
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def _critic_values(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float()
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return self.value_net(obs)

    @staticmethod
    def _zero_vf_states(states: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _cell = states
        zero = torch.zeros_like(hidden)
        return zero, zero.clone()

    def forward(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
        mean_actions, actor_states = self.actor_mean(obs, lstm_states.pi, episode_starts)
        distribution = self._distribution(mean_actions)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self._critic_values(obs)
        vf_states = self._zero_vf_states(lstm_states.vf)
        return actions, values, log_prob, RNNStates(actor_states, vf_states)

    def get_distribution(
        self,
        obs: torch.Tensor,
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[Distribution, tuple[torch.Tensor, torch.Tensor]]:
        mean_actions, actor_states = self.actor_mean(obs, lstm_states, episode_starts)
        return self._distribution(mean_actions), actor_states

    def predict_values(
        self,
        obs: torch.Tensor,
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        # The value function is feed-forward by design.  Keep the recurrent
        # arguments for exact compatibility with RecurrentPPO's timeout path.
        del lstm_states, episode_starts
        return self._critic_values(obs)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        mean_actions, _actor_states = self.actor_mean(obs, lstm_states.pi, episode_starts)
        distribution = self._distribution(mean_actions)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self._critic_values(obs)
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
