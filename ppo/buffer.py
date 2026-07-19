"""Recurrent rollout storage specialized for the C0 End2Race GRU contract."""

from __future__ import annotations

from collections.abc import Generator
from typing import Optional

import numpy as np
import torch
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer, create_sequencers
from sb3_contrib.common.recurrent.type_aliases import RecurrentRolloutBufferSamples, RNNStates


class ActorHiddenRolloutBuffer(RecurrentRolloutBuffer):
    """Store only real actor GRU h; materialize the three zero states per batch."""

    def reset(self) -> None:
        RolloutBuffer.reset(self)
        self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.current_valid_by_timestep: tuple[tuple[bool, ...], ...] | None = None

    def add(self, *args, lstm_states: RNNStates, **kwargs) -> None:
        self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
        RolloutBuffer.add(self, *args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"
        if not self.generator_ready:
            self.hidden_states_pi = self.hidden_states_pi.swapaxes(1, 2)
            for tensor in (
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "episode_starts",
            ):
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))
        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentRolloutBufferSamples:
        del env
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )
        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        sequence_lengths = np.diff(
            np.concatenate((self.seq_start_indices, np.asarray([len(batch_inds)])))
        )
        self.current_valid_by_timestep = tuple(
            tuple(step < int(length) for length in sequence_lengths)
            for step in range(max_length)
        )
        actor_hidden = self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1)
        actor_hidden = self.to_torch(actor_hidden).contiguous()
        actor_cell = torch.zeros_like(actor_hidden)
        critic_hidden = torch.zeros_like(actor_hidden)
        critic_cell = torch.zeros_like(actor_hidden)

        return RecurrentRolloutBufferSamples(
            observations=self.pad(self.observations[batch_inds]).reshape(
                (padded_batch_size, *self.obs_shape)
            ),
            actions=self.pad(self.actions[batch_inds]).reshape(
                (padded_batch_size, *self.actions.shape[1:])
            ),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(
                (actor_hidden, actor_cell),
                (critic_hidden, critic_cell),
            ),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
        )
