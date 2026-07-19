"""Experiment-only A/B/C actor backends for the Phase 5-v2 audit."""

from __future__ import annotations

from types import MethodType
from typing import Any, Callable

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ppo.policy import (
    END2RACE_ACTION_SIZE,
    END2RACE_LIDAR_SIZE,
    END2RACE_OBSERVATION_SIZE,
    End2RaceGRUPolicy,
)


REFERENCE_ACTOR_FORWARD = End2RaceGRUPolicy._actor_forward


def _sequence_layout(policy: Any, obs: Any, states: Any, episode_starts: torch.Tensor):
    hidden, _dummy_cell = states
    actor_obs = policy._actor_observation(obs).float()
    if actor_obs.ndim == 1:
        actor_obs = actor_obs.unsqueeze(0)
    n_seq = hidden.shape[1]
    obs_sequence = actor_obs.reshape(n_seq, -1, END2RACE_OBSERVATION_SIZE).swapaxes(0, 1)
    start_sequence = episode_starts.float().reshape(n_seq, -1).swapaxes(0, 1)
    return hidden, actor_obs, n_seq, obs_sequence, start_sequence


def timestep_batched_actor_forward(
    policy: Any,
    obs: Any,
    states: Any,
    episode_starts: torch.Tensor,
    valid_by_timestep: tuple[tuple[bool, ...], ...] | None = None,
    *,
    microbatch_cap: int | None = None,
):
    """A/B: batch active sequence slots without changing slot or timestep order."""

    hidden, actor_obs, n_seq, obs_sequence, start_sequence = _sequence_layout(
        policy, obs, states, episode_starts
    )
    means: list[torch.Tensor] = []
    timestep_hidden: list[torch.Tensor] = []
    for timestep, (step_obs, episode_start) in enumerate(zip(obs_sequence, start_sequence)):
        hidden = hidden * (1.0 - episode_start).view(1, n_seq, 1)
        valid_indices = (
            list(range(n_seq))
            if valid_by_timestep is None
            else [index for index, valid in enumerate(valid_by_timestep[timestep]) if valid]
        )
        next_by_slot = [hidden[:, index : index + 1] for index in range(n_seq)]
        means_by_slot = [
            torch.zeros((1, END2RACE_ACTION_SIZE), dtype=actor_obs.dtype, device=actor_obs.device)
            for _ in range(n_seq)
        ]
        cap = len(valid_indices) if microbatch_cap is None else int(microbatch_cap)
        for start in range(0, len(valid_indices), max(1, cap)):
            chunk = valid_indices[start : start + max(1, cap)]
            indices = torch.as_tensor(chunk, dtype=torch.long, device=actor_obs.device)
            action_sequence, next_hidden = policy.end2race_actor(
                step_obs[indices, :END2RACE_LIDAR_SIZE].unsqueeze(1),
                step_obs[indices, END2RACE_LIDAR_SIZE:].unsqueeze(1),
                hidden[:, indices],
            )
            chunk_means = action_sequence[:, -1, :]
            for offset, slot in enumerate(chunk):
                next_by_slot[slot] = next_hidden[:, offset : offset + 1]
                means_by_slot[slot] = chunk_means[offset : offset + 1]
        hidden = torch.cat(next_by_slot, dim=1)
        means.append(torch.cat(means_by_slot, dim=0))
        timestep_hidden.append(hidden.squeeze(0))
    mean_actions = torch.stack(means).transpose(0, 1).reshape(-1, END2RACE_ACTION_SIZE)
    actor_features = torch.stack(timestep_hidden).transpose(0, 1).reshape(-1, policy.actor_hidden_size)
    return mean_actions, (hidden, torch.zeros_like(hidden)), actor_features


def packed_actor_forward(
    policy: Any,
    obs: Any,
    states: Any,
    episode_starts: torch.Tensor,
    valid_by_timestep: tuple[tuple[bool, ...], ...] | None = None,
):
    """C: run unchanged recurrent sequences through one packed GRU call."""

    hidden, actor_obs, n_seq, obs_sequence, start_sequence = _sequence_layout(
        policy, obs, states, episode_starts
    )
    max_length = obs_sequence.shape[0]
    lengths = (
        [max_length] * n_seq
        if valid_by_timestep is None
        else [
            sum(bool(valid_by_timestep[timestep][index]) for timestep in range(max_length))
            for index in range(n_seq)
        ]
    )
    for index, length in enumerate(lengths):
        if torch.count_nonzero(start_sequence[1:length, index]).item() != 0:
            raise RuntimeError("Packed backend received an internal episode start")
    hidden = hidden * (1.0 - start_sequence[0]).view(1, n_seq, 1)
    batch_obs = obs_sequence.swapaxes(0, 1)
    lidar = batch_obs[:, :, :END2RACE_LIDAR_SIZE]
    speed = batch_obs[:, :, END2RACE_LIDAR_SIZE:]
    actor = policy.end2race_actor
    processed_lidar = (-1.0 / (1.0 + torch.exp(-actor.k * lidar)) + 1.0) * 2.0
    speed_embedding = actor.speed_mlp(speed)
    features = torch.cat((processed_lidar, speed_embedding), dim=2)
    packed = pack_padded_sequence(
        features,
        torch.as_tensor(lengths, dtype=torch.long).cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    packed_output, next_hidden = actor.gru(packed, hidden)
    actor_features, _ = pad_packed_sequence(
        packed_output,
        batch_first=True,
        total_length=max_length,
    )
    mean_actions = actor.output_layer(actor_features)
    valid_mask = torch.arange(max_length, device=actor_obs.device).unsqueeze(0) < torch.as_tensor(
        lengths, device=actor_obs.device
    ).unsqueeze(1)
    mean_actions = torch.where(valid_mask.unsqueeze(-1), mean_actions, torch.zeros_like(mean_actions))
    return (
        mean_actions.reshape(-1, END2RACE_ACTION_SIZE),
        (next_hidden, torch.zeros_like(next_hidden)),
        actor_features.reshape(-1, policy.actor_hidden_size),
    )


def backend_callable(name: str) -> Callable[..., Any]:
    if name == "R1":
        return REFERENCE_ACTOR_FORWARD
    if name in ("A", "B"):
        return timestep_batched_actor_forward
    if name == "C":
        return packed_actor_forward
    raise ValueError(name)


def install_dispatch(policy: Any, backend: str) -> Callable[..., Any]:
    """Install a collection/training dispatcher and return the original method."""

    original = policy._actor_forward
    collection_a = backend in ("A", "AB", "AC")
    training_backend = "B" if backend in ("B", "AB") else "C" if backend in ("C", "AC") else "R1"

    def dispatch(self, obs, states, episode_starts, valid_by_timestep=None):
        if valid_by_timestep is None:
            function = timestep_batched_actor_forward if collection_a else REFERENCE_ACTOR_FORWARD
        else:
            function = backend_callable(training_backend)
        return function(self, obs, states, episode_starts, valid_by_timestep)

    policy._actor_forward = MethodType(dispatch, policy)
    return original

