from collections.abc import Generator
import copy as copy_module

import numpy as np
import torch

from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer, create_sequencers
from sb3_contrib.common.recurrent.type_aliases import RecurrentRolloutBufferSamples, RNNStates
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor, FloatSchedule

from utils import log_ppo


class End2RaceRolloutBuffer(RecurrentRolloutBuffer):
    """Store actor data and the recurrent critic hidden stream when needed."""

    def __init__(self, *args, store_critic_hidden: bool = False, **kwargs):
        self.store_critic_hidden = bool(store_critic_hidden)
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        RolloutBuffer.reset(self)
        self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.exploration_speed_log_stds = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        if self.store_critic_hidden:
            self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self._staged_speed_log_std: np.ndarray | None = None
        self.current_valid_by_timestep: tuple[tuple[bool, ...], ...] | None = None
        self.current_speed_log_stds: torch.Tensor | None = None

    def stage_exploration(self, *, speed_log_std: np.ndarray) -> None:
        self._staged_speed_log_std = np.asarray(speed_log_std, dtype=np.float32).reshape(-1)

    def add(self, obs, action, reward, episode_start, value, log_prob, *, lstm_states: RNNStates) -> None:
        self.exploration_speed_log_stds[self.pos] = self._staged_speed_log_std
        self._staged_speed_log_std = None
        self.hidden_states_pi[self.pos] = np.asarray(lstm_states.pi[0].cpu().numpy())
        if self.store_critic_hidden:
            self.hidden_states_vf[self.pos] = np.asarray(lstm_states.vf[0].cpu().numpy())
        RolloutBuffer.add(self, obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: int, *, rng: np.random.Generator) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        if not self.generator_ready:
            self.hidden_states_pi = self.hidden_states_pi.swapaxes(1, 2)
            names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "episode_starts",
                "exploration_speed_log_stds",
            ]
            if self.store_critic_hidden:
                self.hidden_states_vf = self.hidden_states_vf.swapaxes(1, 2)
                names.append("hidden_states_vf")
            for name in names:
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.generator_ready = True
        total = self.buffer_size * self.n_envs
        split_index = int(rng.integers(total))
        indices = np.concatenate((np.arange(total)[split_index:], np.arange(total)[:split_index]))
        env_change = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)
        for start in range(0, total, batch_size):
            yield self._get_samples(indices[start : start + batch_size], env_change)

    def _get_samples(self, batch_inds: np.ndarray, env_change: np.ndarray) -> RecurrentRolloutBufferSamples:
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(self.episode_starts[batch_inds], env_change[batch_inds], self.device)
        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        sequence_lengths = np.diff(np.concatenate((self.seq_start_indices, np.asarray([len(batch_inds)]))))
        self.current_valid_by_timestep = tuple(tuple(step < int(length) for length in sequence_lengths) for step in range(max_length))
        self.current_speed_log_stds = self.pad_and_flatten(
            self.exploration_speed_log_stds[batch_inds]
        )
        actor_hidden = self.to_torch(self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1)).contiguous()
        actor_cell = torch.zeros_like(actor_hidden)
        if self.store_critic_hidden:
            critic_hidden = self.to_torch(self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1)).contiguous()
        else:
            critic_hidden = torch.zeros_like(actor_hidden)
        critic_cell = torch.zeros_like(actor_hidden)
        return RecurrentRolloutBufferSamples(
            observations=self.pad(self.observations[batch_inds]).reshape((padded_batch_size, *self.obs_shape)),
            actions=self.pad(self.actions[batch_inds]).reshape((padded_batch_size, *self.actions.shape[1:])),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates((actor_hidden, actor_cell), (critic_hidden, critic_cell)),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
        )


def setup_recurrent_ppo(model):
    """Build the policy, recurrent states, buffer, and independent minibatch RNGs."""
    model._setup_lr_schedule()
    model.set_random_seed(model.seed)
    model.policy = model.policy_class(model.observation_space, model.action_space, model.lr_schedule, use_sde=model.use_sde, **model.policy_kwargs).to(model.device)
    lstm = model.policy.lstm_actor
    single_hidden_shape = (lstm.num_layers, model.n_envs, lstm.hidden_size)
    model._last_lstm_states = RNNStates(
        (torch.zeros(single_hidden_shape, device=model.device), torch.zeros(single_hidden_shape, device=model.device)),
        (torch.zeros(single_hidden_shape, device=model.device), torch.zeros(single_hidden_shape, device=model.device)),
    )
    hidden_buffer_shape = (model.n_steps, lstm.num_layers, model.n_envs, lstm.hidden_size)
    minibatch_root = np.random.SeedSequence([model.seed, 2])
    warmup_split_seed, warmup_shuffle_seed, actor_minibatch_seed, critic_minibatch_seed = minibatch_root.spawn(4)
    model.warmup_split_rng = np.random.default_rng(warmup_split_seed)
    model.warmup_shuffle_rng = np.random.default_rng(warmup_shuffle_seed)
    model.actor_minibatch_rng = np.random.default_rng(actor_minibatch_seed)
    model.critic_minibatch_rng = np.random.default_rng(critic_minibatch_seed)
    model.rollout_buffer = End2RaceRolloutBuffer(
        model.n_steps,
        model.observation_space,
        model.action_space,
        hidden_buffer_shape,
        model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        store_critic_hidden=model.policy.critic_is_recurrent,
    )
    model.policy._end2race_rollout_buffer = model.rollout_buffer
    model.clip_range = FloatSchedule(model.clip_range)
    action_seed = int(np.random.SeedSequence([model.seed, 3]).generate_state(1)[0])
    torch.manual_seed(action_seed)
    if model.device.type == "cuda":
        torch.cuda.manual_seed_all(action_seed)


def exploration_gates(model, infos):
    """Read the causal front-corridor gate for each environment."""
    return np.asarray([info[model.config.exploration_gate_info_key] for info in infos], dtype=bool)


def collect_recurrent_rollout(model, env, callback, rollout_buffer, n_rollout_steps):
    """Collect one recurrent rollout for default, K10, and corridor K10/K50 exploration."""
    self = model  # Preserve the callback-local name used by SB3 collectors.
    model._rollout_episode_records = []
    model.policy.set_training_mode(False)
    n_steps = 0
    rollout_buffer.reset()
    callback.on_rollout_start()
    lstm_states = copy_module.deepcopy(model._last_lstm_states)
    current_gates = (
        model._last_exploration_gates
        if model._last_exploration_gates is not None
        else exploration_gates(model, env.reset_infos)
    )

    while n_steps < n_rollout_steps:
        # A zero corridor hold disables the gate while preserving one collector.
        model.policy.prepare_rollout_exploration(current_gates, model._last_episode_starts)
        with torch.no_grad():
            obs_tensor = obs_as_tensor(model._last_obs, model.device)
            episode_starts = torch.as_tensor(
                model._last_episode_starts,
                dtype=torch.float32,
                device=model.device,
            )
            actions, values, log_probs, lstm_states = model.policy.forward(
                obs_tensor,
                lstm_states,
                episode_starts,
            )
        actions = actions.cpu().numpy()
        clipped_actions = np.clip(actions, model.action_space.low, model.action_space.high)
        new_obs, rewards, dones, infos = env.step(clipped_actions)
        model.num_timesteps += env.num_envs
        callback.update_locals(locals())
        if not callback.on_step():
            return False
        record_episodes(model, infos, dones)
        n_steps += 1

        for index, done in enumerate(dones):
            if (
                done
                and infos[index].get("terminal_observation") is not None
                and infos[index].get("TimeLimit.truncated", False)
            ):
                terminal_obs = model.policy.obs_to_tensor(infos[index]["terminal_observation"])[0]
                with torch.no_grad():
                    terminal_lstm_state = (
                        lstm_states.vf[0][:, index : index + 1, :].contiguous(),
                        lstm_states.vf[1][:, index : index + 1, :].contiguous(),
                    )
                    terminal_starts = torch.as_tensor([False], dtype=torch.float32, device=model.device)
                    terminal_value = model.policy.predict_values(
                        terminal_obs,
                        terminal_lstm_state,
                        terminal_starts,
                    )[0]
                rewards[index] += model.gamma * terminal_value

        rollout_buffer.add(
            model._last_obs,
            actions,
            rewards,
            model._last_episode_starts,
            values,
            log_probs,
            lstm_states=model._last_lstm_states,
        )
        model._last_obs = new_obs
        model._last_episode_starts = dones
        model._last_lstm_states = lstm_states
        current_gates = exploration_gates(
            model,
            [env.reset_infos[index] if done else infos[index] for index, done in enumerate(dones)],
        )
        model._last_exploration_gates = current_gates
    with torch.no_grad():
        final_starts = torch.as_tensor(dones, dtype=torch.float32, device=model.device)
        values = model.policy.predict_values(
            obs_as_tensor(new_obs, model.device),
            lstm_states.vf,
            final_starts,
        )
    rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
    callback.on_rollout_end()
    return True


def record_episodes(model, infos, dones):
    """Persist completed episodes and retain the current rollout records for metrics."""
    for info, done in zip(infos, dones):
        if not done:
            continue
        record = {
            "phase": "formal" if model.warmup_completed else "warmup",
            "update": model._n_updates + 1 if model.warmup_completed else 0,
            "scenario_id": str(info["scenario_id"]),
            "env_role": str(info["env_role"]),
            "episode_outcome": str(info["episode_outcome"]),
            "episode_return": float(info["episode_return"]),
            "episode_steps": int(info["episode_steps"]),
            "elapsed_time": float(info["elapsed_time"]),
            "termination_reason": str(info["termination_reason"]),
            "timeout": bool(info["timeout"]),
            "opponent_collision": bool(info["opponent_collision"]),
        }
        log_ppo(model.output_dir, "episode", record)
        model._rollout_episode_records.append(record)


def _warmup_split(model):
    """Split completed collision and ordinary sequences into train and validation sets."""
    starts = model.rollout_buffer.episode_starts
    sequences = {"collision": [], "ordinary": []}
    for env_index in range(model.n_envs):
        role = "collision" if env_index % 2 == 0 else "ordinary"
        boundaries = np.flatnonzero(starts[:, env_index] > 0.5).tolist()
        boundaries.append(model.n_steps)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end == model.n_steps and not bool(model._last_episode_starts[env_index]):
                continue
            sequences[role].append((env_index, start, end))

    train_sequences = []
    validation_sequences = []
    for role in ("collision", "ordinary"):
        role_sequences = sequences[role]
        order = model.warmup_split_rng.permutation(len(role_sequences))
        train_count = min(max(int(len(order) * model.config.warmup_train_fraction), 1), len(order) - 1)
        for destination, selected in ((train_sequences, order[:train_count]), (validation_sequences, order[train_count:])):
            for sequence_index in selected:
                destination.append(role_sequences[int(sequence_index)])
    return train_sequences, validation_sequences


def _flat_sequence_indices(model, sequences):
    """Convert time-major environment sequences to flattened buffer indices."""
    return np.concatenate([
        np.arange(start, end, dtype=np.int64) * model.n_envs + env_index
        for env_index, start, end in sequences
    ])


def _critic_batch_loss(model, flat_inputs, flat_returns, indices):
    """Calculate one non-recurrent critic minibatch loss."""
    inputs = torch.as_tensor(flat_inputs[indices], dtype=torch.float32, device=model.device)
    returns = torch.as_tensor(flat_returns[indices], dtype=torch.float32, device=model.device)
    return torch.nn.functional.mse_loss(model.policy.evaluate_values(inputs).flatten(), returns)


def _validation_loss(model, flat_inputs, flat_returns, indices):
    """Calculate the sample-weighted non-recurrent validation loss."""
    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), model.batch_size):
            batch = indices[start : start + model.batch_size]
            losses.append((float(_critic_batch_loss(model, flat_inputs, flat_returns, batch).item()), len(batch)))
    return sum(loss * count for loss, count in losses) / sum(count for _loss, count in losses)


def _pack_sequences(model, sequences):
    """Group recurrent sequences into approximately batch-sized transition sets."""
    groups = []
    current = []
    transitions = 0
    for sequence in sequences:
        length = sequence[2] - sequence[1]
        if current and transitions + length > model.batch_size:
            groups.append(current)
            current, transitions = [], 0
        current.append(sequence)
        transitions += length
    if current:
        groups.append(current)
    return groups


def _recurrent_critic_batch(model, sequences):
    """Build one padded recurrent critic batch from time-major rollout arrays."""
    buffer = model.rollout_buffer
    n_seq = len(sequences)
    max_length = max(end - start for _env_index, start, end in sequences)
    observation_size = buffer.obs_shape[0]
    num_layers, hidden_size = buffer.hidden_state_shape[1], buffer.hidden_state_shape[3]
    observations = np.zeros((n_seq, max_length, observation_size), dtype=np.float32)
    episode_starts = np.zeros((n_seq, max_length), dtype=np.float32)
    returns = np.zeros((n_seq, max_length), dtype=np.float32)
    valid = np.zeros((n_seq, max_length), dtype=bool)
    hidden = np.zeros((num_layers, n_seq, hidden_size), dtype=np.float32)
    for slot, (env_index, start, end) in enumerate(sequences):
        length = end - start
        observations[slot, :length] = buffer.observations[start:end, env_index]
        episode_starts[slot, :length] = buffer.episode_starts[start:end, env_index]
        returns[slot, :length] = buffer.returns[start:end, env_index]
        valid[slot, :length] = True
        hidden[:, slot] = buffer.hidden_states_vf[start, :, env_index]
    valid_by_timestep = tuple(tuple(bool(flag) for flag in valid[:, timestep]) for timestep in range(max_length))
    hidden_tensor = torch.as_tensor(hidden, device=model.device)
    return (
        torch.as_tensor(observations.reshape(n_seq * max_length, observation_size), device=model.device),
        (hidden_tensor, torch.zeros_like(hidden_tensor)),
        torch.as_tensor(episode_starts.reshape(-1), device=model.device),
        valid_by_timestep,
        torch.as_tensor(returns.reshape(-1), device=model.device),
        torch.as_tensor(valid.reshape(-1), device=model.device),
    )


def _recurrent_critic_loss(model, sequences):
    """Calculate value loss over valid steps in a padded recurrent batch."""
    observations, states, starts, valid_by_timestep, returns, mask = _recurrent_critic_batch(model, sequences)
    values = model.policy.evaluate_recurrent_values(observations, states, starts, valid_by_timestep).flatten()
    return torch.nn.functional.mse_loss(values[mask], returns[mask]), int(mask.sum().item())


def _recurrent_validation_loss(model, sequences):
    """Calculate the sample-weighted recurrent validation loss."""
    losses = []
    with torch.no_grad():
        for group in _pack_sequences(model, sequences):
            loss, count = _recurrent_critic_loss(model, group)
            losses.append((float(loss.item()), count))
    return sum(loss * count for loss, count in losses) / sum(count for _loss, count in losses)


def _apply_critic_gradient(model, loss):
    """Apply one clipped critic optimizer step."""
    model.policy.critic_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.policy.critic_parameters, model.config.max_grad_norm)
    model.policy.critic_optimizer.step()


def warmup_critic(model):
    """Fit the critic on the first rollout and restore its best validation epoch."""
    train_sequences, validation_sequences = _warmup_split(model)
    recurrent_critic = model.policy.critic_is_recurrent
    if not recurrent_critic:
        flat_inputs = model.rollout_buffer.observations.reshape(-1, *model.rollout_buffer.obs_shape)
        flat_returns = model.rollout_buffer.returns.reshape(-1)
        train_indices = _flat_sequence_indices(model, train_sequences)
        validation_indices = _flat_sequence_indices(model, validation_sequences)
    best_loss = float("inf")
    best_critic = None
    best_optimizer = None
    best_epoch = 0
    stale_epochs = 0
    warmup_train_losses = []
    warmup_validation_losses = []
    for epoch in range(model.config.warmup_max_epochs):
        epoch_train_losses = []
        if recurrent_critic:
            order = model.warmup_shuffle_rng.permutation(len(train_sequences))
            shuffled_sequences = [train_sequences[int(index)] for index in order]
            for group in _pack_sequences(model, shuffled_sequences):
                group_loss, count = _recurrent_critic_loss(model, group)
                _apply_critic_gradient(model, model.config.value_loss_coefficient * group_loss)
                epoch_train_losses.append((float(group_loss.item()), count))
            validation_loss = _recurrent_validation_loss(model, validation_sequences)
        else:
            shuffled = model.warmup_shuffle_rng.permutation(train_indices)
            for start in range(0, len(shuffled), model.batch_size):
                batch = shuffled[start : start + model.batch_size]
                batch_loss = _critic_batch_loss(model, flat_inputs, flat_returns, batch)
                _apply_critic_gradient(model, model.config.value_loss_coefficient * batch_loss)
                epoch_train_losses.append((float(batch_loss.item()), len(batch)))
            validation_loss = _validation_loss(model, flat_inputs, flat_returns, validation_indices)
        train_loss = sum(loss * count for loss, count in epoch_train_losses) / sum(count for _loss, count in epoch_train_losses)
        warmup_train_losses.append(train_loss)
        warmup_validation_losses.append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_critic = copy_module.deepcopy(model.policy.value_net.state_dict())
            best_optimizer = copy_module.deepcopy(model.policy.critic_optimizer.state_dict())
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= model.config.warmup_patience:
                break
    model.policy.value_net.load_state_dict(best_critic)
    model.policy.critic_optimizer.load_state_dict(best_optimizer)
    model.warmup_completed = True
    return {
        "phase": "warmup",
        "epochs": epoch + 1,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "warmup_train_losses": warmup_train_losses,
        "warmup_validation_losses": warmup_validation_losses,
    }


def _batch_values(model, rollout_data):
    """Evaluate one rollout batch with the selected recurrent or MLP critic."""
    if model.policy.critic_is_recurrent:
        critic_states = (rollout_data.lstm_states.vf[0], rollout_data.lstm_states.vf[1])
        return model.policy.evaluate_recurrent_values(
            rollout_data.observations,
            critic_states,
            rollout_data.episode_starts,
            model.rollout_buffer.current_valid_by_timestep,
        ).flatten()
    return model.policy.evaluate_values(rollout_data.observations).flatten()


def train_actor(model, clip_range):
    """Update only the actor and return PPO loss statistics."""
    policy_losses = []
    clip_fractions = []
    approximate_kls = []
    for parameter in model.policy.critic_parameters:
        parameter.requires_grad_(False)
    for _epoch in range(model.actor_epochs):
        for rollout_data in model.rollout_buffer.get(model.batch_size, rng=model.actor_minibatch_rng):
            mask = rollout_data.mask > 1e-8
            advantages = rollout_data.advantages
            if model.normalize_advantage:
                valid_advantages = advantages[mask]
                advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
            log_prob, _entropy = model.policy.evaluate_actor_actions(
                rollout_data.observations,
                rollout_data.actions,
                rollout_data.lstm_states,
                rollout_data.episode_starts,
            )
            ratio = torch.exp(log_prob - rollout_data.old_log_prob)
            with torch.no_grad():
                log_ratio = log_prob - rollout_data.old_log_prob
                approximate_kls.append(float(((torch.exp(log_ratio) - 1) - log_ratio)[mask].mean().cpu().item()))
                clip_fractions.append(float((torch.abs(ratio - 1) > clip_range)[mask].float().mean().cpu().item()))
            policy_loss = -torch.min(
                advantages * ratio,
                advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range),
            )[mask].mean()
            model.policy.actor_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.actor_parameters, model.config.max_grad_norm)
            model.policy.actor_optimizer.step()
            policy_losses.append(float(policy_loss.item()))
    for parameter in model.policy.critic_parameters:
        parameter.requires_grad_(True)
    return policy_losses, clip_fractions, approximate_kls


def train_critic(model):
    """Update only the critic and return sample-weighted losses."""
    for parameter in model.policy.actor_parameters:
        parameter.requires_grad_(False)
    value_loss_samples = []
    critic_epoch_value_losses = []
    for _epoch in range(model.critic_epochs):
        epoch_value_losses = []
        for rollout_data in model.rollout_buffer.get(model.batch_size, rng=model.critic_minibatch_rng):
            mask = rollout_data.mask > 1e-8
            values = _batch_values(model, rollout_data)
            value_loss = torch.nn.functional.mse_loss(values[mask], rollout_data.returns[mask])
            _apply_critic_gradient(model, model.config.value_loss_coefficient * value_loss)
            sample = (float(value_loss.item()), int(mask.sum().item()))
            epoch_value_losses.append(sample)
            value_loss_samples.append(sample)
        critic_epoch_value_losses.append(
            sum(loss * count for loss, count in epoch_value_losses) / sum(count for _loss, count in epoch_value_losses)
        )
    for parameter in model.policy.actor_parameters:
        parameter.requires_grad_(True)
    return value_loss_samples, critic_epoch_value_losses
