"""End2Race recurrent PPO rollout storage and training algorithm."""

from __future__ import annotations

from collections import deque
from collections.abc import Generator
import copy as copy_module
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from gym_notices import notices as gym_notices

gym_notices.notices.clear()

from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer, create_sequencers
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RecurrentRolloutBufferSamples, RNNStates
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.utils import FloatSchedule, safe_mean
from stable_baselines3.common.vec_env import VecNormalize

from latticeplanner.utils import load_config
from utils import TrainingRecorder, atomic_write_json, require_finite_number, require_finite_tensor

CONFIG = load_config("ppo/ppo_config.yaml")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FirstActionPreferenceDataset:

    def __init__(self, path, policy, device, seed):
        self.root = Path(path).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        gate_path = self.root / "gate_report.json"
        if not manifest_path.is_file() or not gate_path.is_file():
            raise FileNotFoundError("First-action preference manifest or gate report is missing")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.gate = json.loads(gate_path.read_text(encoding="utf-8"))
        self.manifest_sha256 = _sha256_file(manifest_path)
        self.gate_sha256 = _sha256_file(gate_path)
        if self.manifest.get("schema_version") != 1:
            raise RuntimeError("First-action preference dataset schema changed")
        if self.gate.get("verdict") != "pass" or not self.gate.get("target_labeled_episode_count") or not self.gate.get("control_labeled_episode_count"):
            raise RuntimeError("First-action preference P0/P1 gate did not pass")
        if self.manifest.get("gate_report_sha256") != _sha256_file(gate_path):
            raise RuntimeError("First-action preference gate report identity changed")
        self.policy = policy
        self.device = device
        self.episodes = []
        keys = set()
        for row in self.manifest.get("episodes", []):
            key = str(row.get("episode_key", ""))
            role = str(row.get("role", ""))
            if not key or key in keys or role not in ("target", "control"):
                raise RuntimeError("First-action preference episode identity or role changed")
            keys.add(key)
            sequence_path = self.root / row["sequence_file"]
            if not sequence_path.is_file() or _sha256_file(sequence_path) != row["sequence_sha256"]:
                raise RuntimeError(f"First-action preference sequence identity changed: {key}")
            with np.load(sequence_path, allow_pickle=False) as payload:
                if set(payload.files) != {"observations", "episode_starts"}:
                    raise RuntimeError(f"First-action preference sequence schema changed: {key}")
                observations = np.asarray(payload["observations"], dtype=np.float32)
                episode_starts = np.asarray(payload["episode_starts"], dtype=bool)
            if observations.ndim != 2 or observations.shape[1] != 361 or episode_starts.shape != (len(observations),):
                raise RuntimeError(f"First-action preference sequence shape changed: {key}")
            if len(observations) == 0 or not bool(episode_starts[0]) or bool(episode_starts[1:].any()) or not np.isfinite(observations).all():
                raise RuntimeError(f"First-action preference sequence values changed: {key}")
            states = []
            for state in row.get("states", []):
                index = int(state["decision_index"])
                if not 0 <= index < len(observations):
                    raise RuntimeError(f"First-action preference decision index changed: {key}")
                pairs = []
                for pair in state.get("pairs", []):
                    good = np.asarray(pair["good_action"], dtype=np.float32)
                    bad = np.asarray(pair["bad_action"], dtype=np.float32)
                    if good.shape != (2,) or bad.shape != (2,) or not np.isfinite(np.concatenate((good, bad))).all() or np.array_equal(good, bad):
                        raise RuntimeError(f"First-action preference action pair changed: {key}")
                    pairs.append({
                        "good": good,
                        "bad": bad,
                        "family": str(pair["family"]),
                        "direction": str(pair["direction"]),
                    })
                if not pairs:
                    raise RuntimeError(f"First-action preference state has no pairs: {key}")
                states.append({"decision_index": index, "lead_steps": int(state["lead_steps"]), "pairs": pairs})
            if not states:
                raise RuntimeError(f"First-action preference episode has no labeled states: {key}")
            self.episodes.append({
                "episode_key": key,
                "role": role,
                "stratum": str(row["stratum"]),
                "observations": observations,
                "episode_starts": episode_starts,
                "states": states,
            })
        self.target_indices = np.asarray([index for index, row in enumerate(self.episodes) if row["role"] == "target"], dtype=np.int64)
        self.control_indices = np.asarray([index for index, row in enumerate(self.episodes) if row["role"] == "control"], dtype=np.int64)
        if len(self.target_indices) != int(self.gate["target_labeled_episode_count"]) or len(self.control_indices) != int(self.gate["control_labeled_episode_count"]):
            raise RuntimeError("First-action preference labeled episode counts changed")
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xF1A57]))
        self.target_order = self.rng.permutation(self.target_indices)
        self.control_order = self.rng.permutation(self.control_indices)
        self.target_cursor = 0
        self.control_cursor = 0

    def sampler_state(self):
        return {
            "rng_state": copy_module.deepcopy(self.rng.bit_generator.state),
            "target_order": self.target_order.copy(),
            "control_order": self.control_order.copy(),
            "target_cursor": self.target_cursor,
            "control_cursor": self.control_cursor,
        }

    def load_sampler_state(self, state):
        self.rng.bit_generator.state = copy_module.deepcopy(state["rng_state"])
        self.target_order = np.asarray(state["target_order"], dtype=np.int64).copy()
        self.control_order = np.asarray(state["control_order"], dtype=np.int64).copy()
        self.target_cursor = int(state["target_cursor"])
        self.control_cursor = int(state["control_cursor"])

    def _draw(self, role, count):
        if role == "target":
            order_name = "target_order"
            cursor_name = "target_cursor"
            pool = self.target_indices
        else:
            order_name = "control_order"
            cursor_name = "control_cursor"
            pool = self.control_indices
        selected = []
        while len(selected) < count:
            order = getattr(self, order_name)
            cursor = getattr(self, cursor_name)
            remaining = len(order) - cursor
            take = min(count - len(selected), remaining)
            selected.extend(int(value) for value in order[cursor : cursor + take])
            cursor += take
            if cursor == len(order):
                order = self.rng.permutation(pool)
                cursor = 0
            setattr(self, order_name, order)
            setattr(self, cursor_name, cursor)
        return selected

    def next_batch_indices(self):
        return {
            "target": self._draw("target", CONFIG.first_action_preference_target_episodes_per_batch),
            "control": self._draw("control", CONFIG.first_action_preference_control_episodes_per_batch),
        }

    def _episode_loss(self, episode):
        observations = episode["observations"]
        states_by_index = {state["decision_index"]: state for state in episode["states"]}
        maximum_index = max(states_by_index)
        hidden = torch.zeros((1, 1, self.policy.end2race_actor.gru.hidden_size), dtype=torch.float32, device=self.device)
        sequence = torch.as_tensor(observations[: maximum_index + 1], dtype=torch.float32, device=self.device).unsqueeze(0)
        action_sequence, _hidden = self.policy.end2race_actor(sequence[:, :, :360], sequence[:, :, 360:], hidden)
        state_losses = []
        margins = []
        for index, state in sorted(states_by_index.items()):
            mean = action_sequence[0, index]
            good = torch.as_tensor(np.stack([pair["good"] for pair in state["pairs"]]), dtype=torch.float32, device=self.device)
            bad = torch.as_tensor(np.stack([pair["bad"] for pair in state["pairs"]]), dtype=torch.float32, device=self.device)
            repeated_mean = mean.unsqueeze(0).expand(len(state["pairs"]), -1)
            distribution = self.policy._distribution(repeated_mean)
            good_log_prob = distribution.log_prob(good)
            bad_log_prob = distribution.log_prob(bad)
            pair_margins = good_log_prob - bad_log_prob
            state_losses.append(F.softplus(-pair_margins).mean())
            margins.append(pair_margins)
        if not state_losses:
            raise RuntimeError(f"First-action preference episode produced no loss: {episode['episode_key']}")
        return torch.stack(state_losses).mean(), torch.cat(margins)

    def loss(self, batch_indices=None):
        selected = self.next_batch_indices() if batch_indices is None else batch_indices
        role_losses = {}
        role_margins = {}
        for role in ("target", "control"):
            losses = []
            margins = []
            for index in selected[role]:
                episode = self.episodes[int(index)]
                if episode["role"] != role:
                    raise RuntimeError("First-action preference sampler crossed roles")
                episode_loss, episode_margins = self._episode_loss(episode)
                losses.append(episode_loss)
                margins.append(episode_margins)
            role_losses[role] = torch.stack(losses).mean()
            role_margins[role] = torch.cat(margins)
        total = 0.5 * role_losses["target"] + 0.5 * role_losses["control"]
        all_margins = torch.cat((role_margins["target"], role_margins["control"]))
        return total, role_losses["target"], role_losses["control"], all_margins


class OnlineBranchRollout:

    def __init__(self, device, actions_per_state=CONFIG.online_branch_actions_per_state):
        self.device = device
        self.actions_per_state = int(actions_per_state)
        self.reset()

    def reset(self):
        self.observations = []
        self.actions = []
        self.old_log_probs = []
        self.advantages = []
        self.actor_hidden = []
        self.episode_starts = []
        self.state_ids = []
        self.ranks = []
        self.roles = []
        self.rollout_steps = []
        self.returns = []
        self.branch_lengths = []
        self.outcomes = []
        self.simulator_steps = 0
        self._finalized = False

    def add_group(self, observation, actor_hidden, episode_start, actions, old_log_probs, returns, rank, role, rollout_step, branch_lengths, outcomes):
        if self._finalized:
            raise RuntimeError("Online branch rollout was already finalized")
        actions = np.asarray(actions, dtype=np.float32)
        old_log_probs = np.asarray(old_log_probs, dtype=np.float32).reshape(-1)
        returns = np.asarray(returns, dtype=np.float32).reshape(-1)
        branch_lengths = np.asarray(branch_lengths, dtype=np.int64).reshape(-1)
        count = actions.shape[0]
        if actions.shape != (self.actions_per_state, 2) or old_log_probs.shape != (count,) or returns.shape != (count,) or branch_lengths.shape != (count,) or len(outcomes) != count:
            raise RuntimeError("Online branch group does not satisfy the fixed action-count contract")
        if not np.isfinite(actions).all() or not np.isfinite(old_log_probs).all() or not np.isfinite(returns).all():
            raise ValueError("Online branch action, log-prob, and return values must be finite")
        other_mean = (returns.sum() - returns) / float(count - 1)
        advantages = returns - other_mean
        state_id = len(set(self.state_ids))
        hidden = np.asarray(actor_hidden, dtype=np.float32)
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        for index in range(count):
            self.observations.append(observation.copy())
            self.actions.append(actions[index].copy())
            self.old_log_probs.append(float(old_log_probs[index]))
            self.advantages.append(float(advantages[index]))
            self.actor_hidden.append(hidden.copy())
            self.episode_starts.append(bool(episode_start))
            self.state_ids.append(state_id)
            self.ranks.append(int(rank))
            self.roles.append(str(role))
            self.rollout_steps.append(int(rollout_step))
            self.returns.append(float(returns[index]))
            self.branch_lengths.append(int(branch_lengths[index]))
            self.outcomes.append(str(outcomes[index]))
        self.simulator_steps += int(branch_lengths.sum())

    def finalize(self, expected_states=None, require_role_balance=False, require_nonconstant=True):
        state_count = len(set(self.state_ids))
        if expected_states is not None:
            expected = expected_states * self.actions_per_state
            if len(self.actions) != expected or state_count != expected_states:
                raise RuntimeError(f"Online branch rollout requires {expected} actions from {expected_states} states")
        if require_role_balance:
            roles, role_counts = np.unique(np.asarray(self.roles), return_counts=True)
            counts = dict(zip(roles.tolist(), role_counts.tolist()))
            expected_per_role = len(self.actions) // 2
            if counts != {"collision": expected_per_role, "ordinary": expected_per_role}:
                raise RuntimeError(f"Online branch rollout role balance changed: {counts}")
        advantages = np.asarray(self.advantages, dtype=np.float64)
        if not np.isfinite(advantages).all():
            raise RuntimeError("Online branch advantages must be finite")
        if require_nonconstant and (len(advantages) == 0 or float(advantages.std()) <= 0.0):
            raise RuntimeError("Online branch advantages must be nonconstant")
        for state_id in range(state_count):
            state_advantages = advantages[np.asarray(self.state_ids) == state_id]
            if len(state_advantages) != self.actions_per_state or abs(float(state_advantages.sum())) > 1e-5:
                raise RuntimeError("Leave-one-out online branch advantages must sum to zero within each state")
        if len(advantages) == 0 or float(advantages.std()) <= 0.0:
            self.normalized_advantages = np.zeros_like(advantages, dtype=np.float32)
        else:
            self.normalized_advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).astype(np.float32)
        self._finalized = True

    def epoch_batches(self, batch_count, rng, require_even=True):
        if not self._finalized:
            raise RuntimeError("Online branch rollout must be finalized before sampling")
        if batch_count <= 0 or (require_even and len(self.actions) % batch_count != 0):
            raise ValueError("Online branch samples must split evenly across actor minibatches")
        permutation = rng.permutation(len(self.actions))
        return np.split(permutation, batch_count) if require_even else np.array_split(permutation, batch_count)

    def loss(self, policy, indices, clip_range):
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            raise RuntimeError("Online branch loss requires at least one action")
        observations = torch.as_tensor(np.asarray(self.observations)[indices], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(self.actions)[indices], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(self.old_log_probs)[indices], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(self.normalized_advantages[indices], dtype=torch.float32, device=self.device)
        hidden = torch.as_tensor(np.asarray(self.actor_hidden)[indices], dtype=torch.float32, device=self.device).permute(1, 0, 2).contiguous()
        starts = torch.as_tensor(np.asarray(self.episode_starts)[indices], dtype=torch.float32, device=self.device)
        log_probs = policy.evaluate_independent_actor_actions(observations, actions, hidden, starts)
        ratios = torch.exp(log_probs - old_log_probs)
        loss = -torch.min(advantages * ratios, advantages * torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range)).mean()
        require_finite_tensor("Online branch PPO loss", loss)
        with torch.no_grad():
            log_ratio = log_probs - old_log_probs
            approximate_kl = float(((torch.exp(log_ratio) - 1.0) - log_ratio).mean().cpu().item())
            clip_fraction = float((torch.abs(ratios - 1.0) > clip_range).float().mean().cpu().item())
        if not math.isfinite(approximate_kl) or not math.isfinite(clip_fraction):
            raise RuntimeError("Online branch PPO ratio metrics must be finite")
        return loss, approximate_kl, clip_fraction

    def ratio_identity(self, policy):
        if not self._finalized:
            raise RuntimeError("Online branch rollout must be finalized before ratio identity")
        if not self.actions:
            return 0.0
        observations = torch.as_tensor(np.asarray(self.observations), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(self.actions), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(self.old_log_probs), dtype=torch.float32, device=self.device)
        hidden = torch.as_tensor(np.asarray(self.actor_hidden), dtype=torch.float32, device=self.device).permute(1, 0, 2).contiguous()
        starts = torch.as_tensor(np.asarray(self.episode_starts), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            log_probs = policy.evaluate_independent_actor_actions(observations, actions, hidden, starts)
            log_ratio = log_probs - old_log_probs
        maximum = float(torch.max(torch.abs(log_ratio)).cpu().item())
        if not math.isfinite(maximum):
            raise RuntimeError("Online branch pre-update log-ratio identity must be finite")
        return maximum

    def statistics(self, prefix="online_branch"):
        if not self._finalized:
            return {
                f"{prefix}_state_count": 0,
                f"{prefix}_action_count": 0,
                f"{prefix}_simulator_steps": 0,
            }
        returns = np.asarray(self.returns, dtype=np.float64)
        advantages = np.asarray(self.advantages, dtype=np.float64)
        lengths = np.asarray(self.branch_lengths, dtype=np.float64)
        outcome_counts = {name: self.outcomes.count(name) for name in ("ego_collision", "overtake", "follow", "horizon")}
        return {
            f"{prefix}_state_count": len(set(self.state_ids)),
            f"{prefix}_action_count": len(self.actions),
            f"{prefix}_simulator_steps": self.simulator_steps,
            f"{prefix}_return_mean": float(returns.mean()) if len(returns) else None,
            f"{prefix}_return_std": float(returns.std()) if len(returns) else None,
            f"{prefix}_advantage_std": float(advantages.std()) if len(advantages) else None,
            f"{prefix}_length_mean": float(lengths.mean()) if len(lengths) else None,
            f"{prefix}_outcome_counts": outcome_counts,
        }


class End2RaceRolloutBuffer(RecurrentRolloutBuffer):
    """Store the real actor and independent-critic GRU streams."""

    def __init__(self, *args, store_independent_gru_hidden: bool = False, loss_sample_stride: int = 1, **kwargs):
        self.store_independent_gru_hidden = bool(store_independent_gru_hidden)
        self.loss_sample_stride = int(loss_sample_stride)
        if self.loss_sample_stride < 1:
            raise ValueError("PPO loss sample stride must be positive")
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        RolloutBuffer.reset(self)
        self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.exploration_speed_log_stds = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.loss_sample_mask = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)
        if self.store_independent_gru_hidden:
            self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self._staged_speed_log_std: np.ndarray | None = None
        self.current_valid_by_timestep: tuple[tuple[bool, ...], ...] | None = None
        self.current_speed_log_stds: torch.Tensor | None = None

    def prepare_loss_sampling(self, rng):
        if self.generator_ready:
            raise RuntimeError("PPO loss sampling must be fixed before flattening the rollout buffer")
        self.loss_sample_mask.fill(0.0)
        if self.loss_sample_stride == 1:
            self.loss_sample_mask.fill(1.0)
            return
        for env_index in range(self.n_envs):
            starts = np.flatnonzero(self.episode_starts[:, env_index] > 0.5).tolist()
            boundaries = sorted(set([0, *starts, self.buffer_size]))
            for begin, end in zip(boundaries[:-1], boundaries[1:]):
                length = end - begin
                if length <= 0:
                    continue
                phase = int(rng.integers(min(self.loss_sample_stride, length)))
                self.loss_sample_mask[begin + phase : end : self.loss_sample_stride, env_index] = 1.0

    def stage_exploration(self, *, speed_log_std: np.ndarray) -> None:
        values = np.asarray(speed_log_std, dtype=np.float32).reshape(-1)
        if values.shape != (self.n_envs,) or not np.isfinite(values).all():
            raise RuntimeError(f"Speed log standard deviations must have shape {(self.n_envs,)} and be finite")
        self._staged_speed_log_std = values

    def add(self, obs, action, reward, episode_start, value, log_prob, *, lstm_states: RNNStates) -> None:
        if self._staged_speed_log_std is None:
            raise RuntimeError(
                "Exploration distribution fields were not staged before add"
            )
        self.exploration_speed_log_stds[self.pos] = self._staged_speed_log_std
        self._staged_speed_log_std = None
        self.hidden_states_pi[self.pos] = np.asarray(lstm_states.pi[0].cpu().numpy())
        if self.store_independent_gru_hidden:
            self.hidden_states_vf[self.pos] = np.asarray(lstm_states.vf[0].cpu().numpy())
        RolloutBuffer.add(self, obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None, *, rng: np.random.Generator) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        if not self.full:
            raise RuntimeError("Rollout buffer must be full before training")
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
                "loss_sample_mask",
            ]
            if self.store_independent_gru_hidden:
                self.hidden_states_vf = self.hidden_states_vf.swapaxes(1, 2)
                names.append("hidden_states_vf")
            for name in names:
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.generator_ready = True
        total = self.buffer_size * self.n_envs
        batch_size = total if batch_size is None else batch_size
        split_index = int(rng.integers(total))
        indices = np.concatenate((np.arange(total)[split_index:], np.arange(total)[:split_index]))
        env_change = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)
        for start in range(0, total, batch_size):
            yield self._get_samples(indices[start : start + batch_size], env_change)

    def _get_samples(self, batch_inds: np.ndarray, env_change: np.ndarray, env: Optional[VecNormalize] = None) -> RecurrentRolloutBufferSamples:
        del env
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
        if self.store_independent_gru_hidden:
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
            mask=self.pad_and_flatten(self.loss_sample_mask[batch_inds]),
        )


class End2RaceRecurrentPPO(RecurrentPPO):
    """Run critic warm-up, then separate actor and critic PPO phases."""

    def __init__(
        self,
        *args,
        actor_epochs: int,
        critic_epochs: int,
        recorder: TrainingRecorder,
        first_action_preference_dataset: str = "",
        first_action_preference_step_fraction: float = 0.0,
        online_same_state_branch_ppo: bool = False,
        collision_prefix_branch_ppo: bool = False,
        ppo_loss_sample_stride: int = 1,
        **kwargs,
    ):
        self.actor_epochs = actor_epochs
        self.critic_epochs = critic_epochs
        self.recorder = recorder
        self.warmup_completed = False
        self.rollout_index = 0
        self.current_phase = "warmup"
        self.rollout_policy_update = 0
        self._rollout_episode_records: list[dict] = []
        self._last_exploration_gates: np.ndarray | None = None
        self.first_action_preference_dataset = str(first_action_preference_dataset)
        self.first_action_preference_step_fraction = float(first_action_preference_step_fraction)
        self.first_action_preference = None
        self.first_action_preference_beta = None
        self.first_action_preference_calibration = None
        self.online_same_state_branch_ppo = bool(online_same_state_branch_ppo)
        self.collision_prefix_branch_ppo = bool(collision_prefix_branch_ppo)
        self.ppo_loss_sample_stride = int(ppo_loss_sample_stride)
        self.online_branch_rollout = None
        self.collision_prefix_histories = None
        kwargs["n_epochs"] = actor_epochs
        super().__init__(*args, **kwargs)
        if not math.isfinite(self.first_action_preference_step_fraction) or self.first_action_preference_step_fraction < 0.0:
            raise ValueError("First-action preference step fraction must be finite and nonnegative")
        if self.first_action_preference_step_fraction > 0.0:
            if not self.first_action_preference_dataset:
                raise ValueError("Positive first-action preference step fraction requires a dataset")
            self.first_action_preference = FirstActionPreferenceDataset(self.first_action_preference_dataset, self.policy, self.device, self.seed)
        if self.ppo_loss_sample_stride < 1:
            raise ValueError("PPO loss sample stride must be positive")
        if self.online_same_state_branch_ppo and self.collision_prefix_branch_ppo:
            raise ValueError("Online same-state and collision-prefix branch PPO are separate experiment arms")
        if self.online_same_state_branch_ppo or self.collision_prefix_branch_ppo:
            if self.first_action_preference is not None:
                raise ValueError("Branched PPO cannot be combined with an auxiliary actor loss")
            if self.policy.speed_exploration_mode != "stepwise_independent":
                raise ValueError("Branched PPO requires stepwise-independent exploration")
        if self.online_same_state_branch_ppo:
            if self.n_envs != CONFIG.online_branch_states_per_rollout or self.n_steps < CONFIG.online_branch_states_per_rollout:
                raise ValueError("Online same-state branch PPO requires 16 environments and at least 16 rollout steps")
            self.online_branch_rollout = OnlineBranchRollout(self.device)
        elif self.collision_prefix_branch_ppo:
            self.online_branch_rollout = OnlineBranchRollout(self.device, CONFIG.collision_prefix_branch_actions_per_state)
            self.collision_prefix_histories = [deque(maxlen=CONFIG.collision_prefix_branch_lookback_steps) for _rank in range(self.n_envs)]

    def _actor_step_space_norm(self, gradients) -> float:
        gru_count = len(tuple(self.policy.end2race_actor.gru.parameters()))
        gru_lr = float(self.policy.actor_optimizer.param_groups[0]["lr"])
        head_lr = float(self.policy.actor_optimizer.param_groups[1]["lr"])
        squared = 0.0
        for index, gradient in enumerate(gradients):
            learning_rate = gru_lr if index < gru_count else head_lr
            squared += learning_rate * learning_rate * float(torch.sum(gradient.detach().double().square()).cpu().item())
        return math.sqrt(squared)

    def _actor_policy_loss(self, rollout_data, clip_range: float) -> torch.Tensor:
        mask = rollout_data.mask > 1e-8
        advantages = rollout_data.advantages
        if self.normalize_advantage:
            valid_advantages = advantages[mask]
            advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
        log_prob, _entropy = self.policy.evaluate_actor_actions(
            rollout_data.observations,
            rollout_data.actions,
            rollout_data.lstm_states,
            rollout_data.episode_starts,
        )
        ratio = torch.exp(log_prob - rollout_data.old_log_prob)
        policy_loss = -torch.min(advantages * ratio, advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range))[mask].mean()
        require_finite_tensor("Policy loss", policy_loss)
        return policy_loss

    def _calibrate_first_action_preference(self, clip_range: float) -> None:
        if self.first_action_preference is None or self.first_action_preference_beta is not None:
            return
        actor_rng_state = copy_module.deepcopy(self.actor_minibatch_rng.bit_generator.state)
        preference_state = self.first_action_preference.sampler_state()
        ppo_norms = []
        preference_norms = []
        for _epoch in range(self.actor_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size, rng=self.actor_minibatch_rng):
                policy_loss = self._actor_policy_loss(rollout_data, clip_range)
                preference_loss, _target_loss, _control_loss, _margins = self.first_action_preference.loss()
                require_finite_tensor("First-action preference calibration loss", preference_loss)
                ppo_gradients = torch.autograd.grad(policy_loss, self.policy.actor_parameters, retain_graph=True)
                preference_gradients = torch.autograd.grad(preference_loss, self.policy.actor_parameters)
                ppo_norms.append(self._actor_step_space_norm(ppo_gradients))
                preference_norms.append(self._actor_step_space_norm(preference_gradients))
        self.actor_minibatch_rng.bit_generator.state = actor_rng_state
        self.first_action_preference.load_sampler_state(preference_state)
        if len(ppo_norms) != self.actor_epochs * self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs // self.batch_size:
            raise RuntimeError("First-action preference calibration did not cover the planned actor minibatches")
        ppo_median = float(np.median(ppo_norms))
        preference_median = float(np.median(preference_norms))
        if not math.isfinite(ppo_median) or not math.isfinite(preference_median) or ppo_median <= 0.0 or preference_median <= 0.0:
            raise RuntimeError("First-action preference calibration gradients must be finite and positive")
        beta = self.first_action_preference_step_fraction * ppo_median / preference_median
        if not math.isfinite(beta) or beta <= 0.0:
            raise RuntimeError("First-action preference beta must be finite and positive")
        self.first_action_preference_beta = beta
        self.first_action_preference_calibration = {
            "schema_version": 1,
            "dataset_manifest_sha256": self.first_action_preference.manifest_sha256,
            "dataset_gate_sha256": self.first_action_preference.gate_sha256,
            "target_step_fraction": self.first_action_preference_step_fraction,
            "ppo_step_space_norm_median": ppo_median,
            "preference_step_space_norm_median": preference_median,
            "beta": beta,
        }
        atomic_write_json(self.recorder.output_dir / "first_action_preference_calibration.json", self.first_action_preference_calibration)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.policy = self.policy_class(self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs).to(self.device)
        if not isinstance(self.policy, RecurrentActorCriticPolicy) or not self.policy.supports_end2race_rollout_buffer():
            raise TypeError("End2Race PPO requires the End2Race GRU policy")
        lstm = self.policy.lstm_actor
        single_hidden_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        self._last_lstm_states = RNNStates(
            (torch.zeros(single_hidden_shape, device=self.device), torch.zeros(single_hidden_shape, device=self.device)),
            (torch.zeros(single_hidden_shape, device=self.device), torch.zeros(single_hidden_shape, device=self.device)),
        )
        hidden_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
        minibatch_root = np.random.SeedSequence([self.seed, 2])
        warmup_split_seed, warmup_shuffle_seed, actor_minibatch_seed, critic_minibatch_seed = minibatch_root.spawn(4)
        self.warmup_split_rng = np.random.default_rng(warmup_split_seed)  # Warm-up train/validation sequence split only.
        self.warmup_shuffle_rng = np.random.default_rng(warmup_shuffle_seed)  # Warm-up critic epoch shuffle only.
        self.actor_minibatch_rng = np.random.default_rng(actor_minibatch_seed)  # Formal actor minibatch splits only.
        self.critic_minibatch_rng = np.random.default_rng(critic_minibatch_seed)  # Formal critic minibatch splits only.
        self.online_branch_minibatch_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 6]))
        self.loss_sample_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 7]))
        self.rollout_buffer = End2RaceRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            store_independent_gru_hidden=self.policy.critic_is_independent_gru,
            loss_sample_stride=self.ppo_loss_sample_stride,
        )
        self.policy._end2race_rollout_buffer = self.rollout_buffer
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

        action_seed = int(np.random.SeedSequence([self.seed, 3]).generate_state(1)[0])
        torch.manual_seed(action_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(action_seed)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        self.rollout_index += 1
        self.current_phase = "warmup" if not self.warmup_completed else "formal"
        self.rollout_policy_update = self._n_updates
        self._rollout_episode_records = []
        if self.online_same_state_branch_ppo and self.warmup_completed:
            completed = self._collect_online_branch_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        elif self.collision_prefix_branch_ppo and self.warmup_completed:
            completed = self._collect_collision_prefix_branch_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        elif self.policy.speed_exploration_mode == "stepwise_independent":
            completed = super().collect_rollouts(
                env, callback, rollout_buffer, n_rollout_steps
            )
        else:
            completed = self._collect_structured_exploration_rollouts(
                env, callback, rollout_buffer, n_rollout_steps
            )
        if completed and self.warmup_completed:
            rollout_buffer.prepare_loss_sampling(self.loss_sample_rng)
        return completed

    def _torch_rng_state(self):
        return {
            "cpu": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
        }

    def _restore_torch_rng_state(self, state) -> None:
        torch.random.set_rng_state(state["cpu"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _branch_critic_state_after_observation(self, observation_tensor, critic_state, episode_start_tensor):
        if self.policy.critic_is_independent_gru:
            _value, next_state = self.policy._independent_gru_forward_collection(observation_tensor, critic_state, episode_start_tensor)
            return next_state
        return self.policy._zero_states(critic_state)

    def _online_branch_bootstrap(self, observation, critic_state) -> float:
        observation_tensor = obs_as_tensor(np.asarray(observation, dtype=np.float32).reshape(1, -1), self.device)
        with torch.no_grad():
            value = self.policy.predict_values(observation_tensor, critic_state, torch.zeros(1, dtype=torch.float32, device=self.device))
        result = float(value.reshape(-1)[0].detach().cpu().item())
        require_finite_number("Online branch bootstrap value", result)
        return result

    def _collect_branch_group(self, env, rank, branch_snapshot, restore_snapshot, observation, actor_state, critic_state, episode_start, rollout_step, role, actions_per_state, horizon_steps):
        if self.online_branch_rollout is None:
            raise RuntimeError("Online branch rollout storage was not initialized")
        if not np.array_equal(np.asarray(branch_snapshot["observation"], dtype=np.float32), np.asarray(observation, dtype=np.float32)):
            raise RuntimeError("Online branch snapshot observation differs from the current on-policy observation")
        rng_state = self._torch_rng_state()
        observation_tensor = obs_as_tensor(np.asarray(observation, dtype=np.float32).reshape(1, -1), self.device)
        episode_start_tensor = torch.as_tensor([episode_start], dtype=torch.float32, device=self.device)
        actions = []
        old_log_probs = []
        returns = []
        lengths = []
        outcomes = []
        try:
            with torch.no_grad():
                distribution, actor_state_after = self.policy.get_distribution(observation_tensor, actor_state, episode_start_tensor)
                critic_state_after = self._branch_critic_state_after_observation(observation_tensor, critic_state, episode_start_tensor)
                candidate_actions = []
                candidate_log_probs = []
                for _candidate in range(actions_per_state):
                    action_tensor = distribution.sample()
                    log_prob_tensor = distribution.log_prob(action_tensor)
                    action = action_tensor[0].detach().cpu().numpy().astype(np.float32)
                    clipped = np.clip(action, self.action_space.low, self.action_space.high)
                    if not np.array_equal(action, clipped):
                        raise RuntimeError("Online branch first action was clipped and no longer has the stored old-policy log-probability")
                    candidate_actions.append(action.copy())
                    candidate_log_probs.append(float(log_prob_tensor[0].detach().cpu().item()))
                continuation_rng_state = self._torch_rng_state()
                for action, old_log_prob in zip(candidate_actions, candidate_log_probs):
                    actions.append(action.copy())
                    old_log_probs.append(old_log_prob)
                    self._restore_torch_rng_state(continuation_rng_state)
                    env.env_method("restore_runtime_snapshot", branch_snapshot, indices=[rank])
                    branch_actor_state = (actor_state_after[0].clone(), actor_state_after[1].clone())
                    branch_critic_state = (critic_state_after[0].clone(), critic_state_after[1].clone())
                    branch_action = action
                    discounted_return = 0.0
                    discount = 1.0
                    branch_outcome = "horizon"
                    branch_observation = np.asarray(observation, dtype=np.float32)
                    branch_length = 0
                    horizon_bootstrap = None
                    for branch_step in range(horizon_steps):
                        branch_observation, reward, terminated, truncated, info = env.env_method("step", branch_action, indices=[rank])[0]
                        branch_observation = np.asarray(branch_observation, dtype=np.float32)
                        discounted_return += discount * float(reward)
                        discount *= self.gamma
                        branch_length = branch_step + 1
                        if terminated or truncated:
                            branch_outcome = str(info["episode_outcome"])
                            if truncated and not terminated:
                                discounted_return += discount * self._online_branch_bootstrap(branch_observation, branch_critic_state)
                            break
                        next_observation_tensor = obs_as_tensor(branch_observation.reshape(1, -1), self.device)
                        branch_states = RNNStates(branch_actor_state, branch_critic_state)
                        branch_action_tensor, _value, next_states = self.policy.forward_independent_collection(next_observation_tensor, branch_states, torch.zeros(1, dtype=torch.float32, device=self.device))
                        horizon_bootstrap = float(_value.reshape(-1)[0].detach().cpu().item())
                        branch_action = branch_action_tensor[0].detach().cpu().numpy().astype(np.float32)
                        branch_action = np.clip(branch_action, self.action_space.low, self.action_space.high)
                        branch_actor_state = next_states.pi
                        branch_critic_state = next_states.vf
                    else:
                        if horizon_bootstrap is None:
                            raise RuntimeError("Online branch horizon bootstrap value is missing")
                        discounted_return += discount * horizon_bootstrap
                    require_finite_number("Online branch discounted return", discounted_return)
                    returns.append(discounted_return)
                    lengths.append(branch_length)
                    outcomes.append(branch_outcome)
        finally:
            env.env_method("restore_runtime_snapshot", restore_snapshot, indices=[rank])
            self._restore_torch_rng_state(rng_state)
        restored_observation = env.env_method("restore_runtime_snapshot", restore_snapshot, indices=[rank])[0]
        if not np.array_equal(np.asarray(restored_observation, dtype=np.float32), np.asarray(restore_snapshot["observation"], dtype=np.float32)):
            raise RuntimeError("Online branch collection did not restore the main rollout observation")
        self.online_branch_rollout.add_group(
            observation,
            actor_state[0][:, 0].detach().cpu().numpy(),
            episode_start,
            actions,
            old_log_probs,
            returns,
            rank,
            role,
            rollout_step,
            lengths,
            outcomes,
        )

    def _collect_online_branch_group(self, env, rank: int, observation: np.ndarray, lstm_states: RNNStates, episode_start: bool, rollout_step: int) -> None:
        snapshot = env.env_method("capture_runtime_snapshot", indices=[rank])[0]
        actor_state = (
            lstm_states.pi[0][:, rank : rank + 1].contiguous(),
            lstm_states.pi[1][:, rank : rank + 1].contiguous(),
        )
        critic_state = (
            lstm_states.vf[0][:, rank : rank + 1].contiguous(),
            lstm_states.vf[1][:, rank : rank + 1].contiguous(),
        )
        role = "collision" if rank % 2 == 0 else "ordinary"
        self._collect_branch_group(env, rank, snapshot, snapshot, observation, actor_state, critic_state, episode_start, rollout_step, role, CONFIG.online_branch_actions_per_state, CONFIG.online_branch_horizon_steps)

    def _collect_online_branch_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        if not isinstance(rollout_buffer, End2RaceRolloutBuffer) or self.online_branch_rollout is None:
            raise TypeError("Online same-state branch PPO requires End2RaceRolloutBuffer and branch storage")
        if self._last_obs is None:
            raise RuntimeError("No previous observation was provided")
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        self.online_branch_rollout.reset()
        callback.on_rollout_start()
        lstm_states = copy_module.deepcopy(self._last_lstm_states)
        trigger_steps = np.floor((np.arange(CONFIG.online_branch_states_per_rollout, dtype=np.float64) + 0.5) * n_rollout_steps / CONFIG.online_branch_states_per_rollout).astype(np.int64)
        if len(np.unique(trigger_steps)) != CONFIG.online_branch_states_per_rollout:
            raise RuntimeError("Online branch trigger steps are not unique")
        trigger_by_step = {int(step): index for index, step in enumerate(trigger_steps)}
        n_steps = 0
        while n_steps < n_rollout_steps:
            trigger_index = trigger_by_step.get(n_steps)
            if trigger_index is not None:
                rank = int(trigger_index)
                self._collect_online_branch_group(env, rank, self._last_obs[rank], lstm_states, bool(self._last_episode_starts[rank]), n_steps)
            with torch.no_grad():
                observation_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = torch.as_tensor(self._last_episode_starts, dtype=torch.float32, device=self.device)
                actions, values, log_probs, next_lstm_states = self.policy.forward(observation_tensor, lstm_states, episode_starts)
            actions = actions.cpu().numpy()
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high) if isinstance(self.action_space, spaces.Box) else actions
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1
            for index, done in enumerate(dones):
                if done and infos[index].get("terminal_observation") is not None and infos[index].get("TimeLimit.truncated", False):
                    terminal_obs = self.policy.obs_to_tensor(infos[index]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_state = (next_lstm_states.vf[0][:, index : index + 1].contiguous(), next_lstm_states.vf[1][:, index : index + 1].contiguous())
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_state, torch.zeros(1, dtype=torch.float32, device=self.device))[0]
                    rewards[index] += self.gamma * terminal_value
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs, lstm_states=lstm_states)
            self._last_obs = new_obs
            self._last_episode_starts = dones
            lstm_states = next_lstm_states
        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, torch.as_tensor(dones, dtype=torch.float32, device=self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        self._last_lstm_states = lstm_states
        self.online_branch_rollout.finalize(expected_states=CONFIG.online_branch_states_per_rollout, require_role_balance=True, require_nonconstant=True)
        identity = self.online_branch_rollout.ratio_identity(self.policy)
        if identity > 5e-5:
            raise RuntimeError(f"Online branch pre-update actor log-ratio identity failed: {identity}")
        callback.on_rollout_end()
        return True

    def _append_collision_prefix_history(self, lstm_states, rollout_step):
        if self.collision_prefix_histories is None or self._last_obs is None:
            raise RuntimeError("Collision-prefix actor histories were not initialized")
        for rank, history in enumerate(self.collision_prefix_histories):
            history.append({
                "observation": np.asarray(self._last_obs[rank], dtype=np.float32).copy(),
                "actor_hidden": lstm_states.pi[0][:, rank : rank + 1].detach().cpu().numpy().copy(),
                "actor_cell": lstm_states.pi[1][:, rank : rank + 1].detach().cpu().numpy().copy(),
                "critic_hidden": lstm_states.vf[0][:, rank : rank + 1].detach().cpu().numpy().copy(),
                "critic_cell": lstm_states.vf[1][:, rank : rank + 1].detach().cpu().numpy().copy(),
                "episode_start": bool(self._last_episode_starts[rank]),
                "rollout_step": int(rollout_step),
            })

    def _collect_collision_prefix_group(self, env, rank, info, restore_observation):
        if self.collision_prefix_histories is None:
            raise RuntimeError("Collision-prefix actor histories were not initialized")
        prefix_snapshot = env.env_method("take_collision_prefix_snapshot", indices=[rank])[0]
        history = self.collision_prefix_histories[rank]
        if prefix_snapshot is None or len(history) < CONFIG.collision_prefix_branch_lookback_steps:
            return False
        record = history[0]
        if not np.array_equal(np.asarray(prefix_snapshot["observation"], dtype=np.float32), record["observation"]):
            raise RuntimeError("Collision-prefix simulator replay and actor history are not aligned")
        restore_snapshot = env.env_method("capture_runtime_snapshot", indices=[rank])[0]
        if not np.array_equal(np.asarray(restore_snapshot["observation"], dtype=np.float32), np.asarray(restore_observation, dtype=np.float32)):
            raise RuntimeError("Collision-prefix main reset snapshot differs from the vector observation")
        actor_state = (
            torch.as_tensor(record["actor_hidden"], dtype=torch.float32, device=self.device),
            torch.as_tensor(record["actor_cell"], dtype=torch.float32, device=self.device),
        )
        critic_state = (
            torch.as_tensor(record["critic_hidden"], dtype=torch.float32, device=self.device),
            torch.as_tensor(record["critic_cell"], dtype=torch.float32, device=self.device),
        )
        env.env_method("set_collision_prefix_branching", True, indices=[rank])
        try:
            self._collect_branch_group(env, rank, prefix_snapshot, restore_snapshot, record["observation"], actor_state, critic_state, record["episode_start"], record["rollout_step"], str(info["env_role"]), CONFIG.collision_prefix_branch_actions_per_state, CONFIG.collision_prefix_branch_horizon_steps)
        finally:
            env.env_method("set_collision_prefix_branching", False, indices=[rank])
        return True

    def _collect_collision_prefix_branch_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if not isinstance(rollout_buffer, End2RaceRolloutBuffer) or self.online_branch_rollout is None or self.collision_prefix_histories is None:
            raise TypeError("Collision-prefix branch PPO requires End2RaceRolloutBuffer and branch storage")
        if self._last_obs is None:
            raise RuntimeError("No previous observation was provided")
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        self.online_branch_rollout.reset()
        callback.on_rollout_start()
        lstm_states = copy_module.deepcopy(self._last_lstm_states)
        n_steps = 0
        while n_steps < n_rollout_steps:
            self._append_collision_prefix_history(lstm_states, n_steps)
            with torch.no_grad():
                observation_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = torch.as_tensor(self._last_episode_starts, dtype=torch.float32, device=self.device)
                actions, values, log_probs, next_lstm_states = self.policy.forward(observation_tensor, lstm_states, episode_starts)
            actions = actions.cpu().numpy()
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high) if isinstance(self.action_space, spaces.Box) else actions
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1
            for index, done in enumerate(dones):
                if done and infos[index].get("terminal_observation") is not None and infos[index].get("TimeLimit.truncated", False):
                    terminal_obs = self.policy.obs_to_tensor(infos[index]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_state = (next_lstm_states.vf[0][:, index : index + 1].contiguous(), next_lstm_states.vf[1][:, index : index + 1].contiguous())
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_state, torch.zeros(1, dtype=torch.float32, device=self.device))[0]
                    rewards[index] += self.gamma * terminal_value
                if done and infos[index].get("ego_collision") and infos[index].get("collision_prefix_available"):
                    self._collect_collision_prefix_group(env, index, infos[index], new_obs[index])
                if done:
                    self.collision_prefix_histories[index].clear()
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs, lstm_states=lstm_states)
            self._last_obs = new_obs
            self._last_episode_starts = dones
            lstm_states = next_lstm_states
        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, torch.as_tensor(dones, dtype=torch.float32, device=self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        self._last_lstm_states = lstm_states
        self.online_branch_rollout.finalize(require_nonconstant=False)
        identity = self.online_branch_rollout.ratio_identity(self.policy)
        if identity > 5e-5:
            raise RuntimeError(f"Collision-prefix branch pre-update actor log-ratio identity failed: {identity}")
        callback.on_rollout_end()
        return True

    def _collect_structured_exploration_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        """SB3 recurrent collection with one extra causal gate side channel."""

        if not isinstance(rollout_buffer, End2RaceRolloutBuffer):
            raise TypeError("Structured exploration requires End2RaceRolloutBuffer")
        if self._last_obs is None:
            raise RuntimeError("No previous observation was provided")
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()
        lstm_states = copy_module.deepcopy(self._last_lstm_states)
        current_gates = (
            np.asarray(self._last_exploration_gates, dtype=bool).copy()
            if self._last_exploration_gates is not None
            else np.asarray(
                [
                    bool(info[CONFIG.exploration_gate_info_key])
                    for info in env.reset_infos
                ],
                dtype=bool,
            )
        )

        while n_steps < n_rollout_steps:
            self.policy.prepare_rollout_exploration(
                current_gates,
                self._last_episode_starts,
            )
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = torch.as_tensor(
                    self._last_episode_starts,
                    dtype=torch.float32,
                    device=self.device,
                )
                actions, values, log_probs, lstm_states = self.policy.forward(
                    obs_tensor,
                    lstm_states,
                    episode_starts,
                )
            actions = actions.cpu().numpy()
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            for index, done in enumerate(dones):
                if (
                    done
                    and infos[index].get("terminal_observation") is not None
                    and infos[index].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[index]["terminal_observation"]
                    )[0]
                    with torch.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, index : index + 1, :].contiguous(),
                            lstm_states.vf[1][:, index : index + 1, :].contiguous(),
                        )
                        terminal_starts = torch.as_tensor(
                            [False], dtype=torch.float32, device=self.device
                        )
                        terminal_value = self.policy.predict_values(
                            terminal_obs,
                            terminal_lstm_state,
                            terminal_starts,
                        )[0]
                    rewards[index] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states
            current_gates = np.asarray(
                [
                    bool(
                        (
                            env.reset_infos[index]
                            if done
                            else infos[index]
                        )[CONFIG.exploration_gate_info_key]
                    )
                    for index, done in enumerate(dones)
                ],
                dtype=bool,
            )
            self._last_exploration_gates = current_gates.copy()

        with torch.no_grad():
            final_starts = torch.as_tensor(
                dones, dtype=torch.float32, device=self.device
            )
            values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device),
                lstm_states.vf,
                final_starts,
            )
        rollout_buffer.compute_returns_and_advantage(
            last_values=values,
            dones=dones,
        )
        callback.on_rollout_end()
        return True

    def dump_logs(self, iteration: int = 0) -> None:
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if self.ep_info_buffer and self.ep_info_buffer[0]:
            self.logger.record("rollout/ep_rew_mean", safe_mean([info["r"] for info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([info["l"] for info in self.ep_info_buffer]))
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.ep_success_buffer:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def _update_info_buffer(self, infos: list[dict], dones: Optional[np.ndarray] = None) -> None:
        super()._update_info_buffer(infos, dones)
        if dones is None:
            dones = np.zeros(len(infos), dtype=bool)
        for info, done in zip(infos, dones):
            if not done:
                continue
            record = {
                "phase": self.current_phase,
                "rollout_index": self.rollout_index,
                "formal_update": 0 if not self.warmup_completed else self._n_updates + 1,
                "rollout_policy_update": self.rollout_policy_update,
                "scenario_id": str(info["scenario_id"]),
                "scenario_pool": str(info["scenario"]["pool"]),
                "env_role": str(info["env_role"]),
                "sampler_branch": str(info["sampler_branch"]),
                "episode_outcome": str(info["episode_outcome"]),
                "episode_return": float(info["episode_return"]),
                "episode_steps": int(info["episode_steps"]),
                "elapsed_time": float(info["elapsed_time"]),
                "ego_collision": bool(info["ego_collision"]),
                "termination_reason": str(info["termination_reason"]),
                "timeout": bool(info["timeout"]),
                "opponent_collision": bool(info["opponent_collision"]),
            }
            self.recorder.record_episode(record)
            self._rollout_episode_records.append(record)

    def _warmup_split(self) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
        starts = self.rollout_buffer.episode_starts
        sequences = {"collision": [], "ordinary": []}
        for env_index in range(self.n_envs):
            if starts[0, env_index] <= 0.5:
                raise RuntimeError("Warm-up rollout must begin with freshly reset environments")
            role = "collision" if env_index % 2 == 0 else "ordinary"
            boundaries = np.flatnonzero(starts[:, env_index] > 0.5).tolist()
            boundaries.append(self.n_steps)
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if end == self.n_steps and not bool(self._last_episode_starts[env_index]):
                    continue
                if end > start:
                    sequences[role].append((env_index, start, end))

        train_sequences: list[tuple[int, int, int]] = []
        validation_sequences: list[tuple[int, int, int]] = []
        for role in ("collision", "ordinary"):
            role_sequences = sequences[role]
            if len(role_sequences) < 2:
                raise RuntimeError(f"Warm-up requires at least two {role} recurrent sequences")
            order = self.warmup_split_rng.permutation(len(role_sequences))
            train_count = min(max(int(len(order) * CONFIG.warmup_train_fraction), 1), len(order) - 1)
            for destination, selected in ((train_sequences, order[:train_count]), (validation_sequences, order[train_count:])):
                for sequence_index in selected:
                    destination.append(role_sequences[int(sequence_index)])
        return train_sequences, validation_sequences

    def _flat_sequence_indices(self, sequences: list[tuple[int, int, int]]) -> np.ndarray:
        parts = [np.arange(start, end, dtype=np.int64) * self.n_envs + env_index for env_index, start, end in sequences]
        return np.concatenate(parts) if parts else np.asarray([], dtype=np.int64)

    def _flat_critic_inputs(self) -> np.ndarray:
        return self.rollout_buffer.observations.reshape(-1, *self.rollout_buffer.obs_shape)

    def _critic_batch_loss(self, flat_inputs: np.ndarray, flat_returns: np.ndarray, indices: np.ndarray) -> torch.Tensor:
        inputs = torch.as_tensor(flat_inputs[indices], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(flat_returns[indices], dtype=torch.float32, device=self.device)
        return torch.nn.functional.mse_loss(self.policy.evaluate_values(inputs).flatten(), returns)

    def _validation_loss(self, flat_inputs: np.ndarray, flat_returns: np.ndarray, indices: np.ndarray) -> float:
        losses = []
        with torch.no_grad():
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                losses.append((float(self._critic_batch_loss(flat_inputs, flat_returns, batch).item()), len(batch)))
        return sum(loss * count for loss, count in losses) / sum(count for _loss, count in losses)

    def _pack_sequences(self, sequences: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
        """Group sequences so each recurrent minibatch holds about batch_size transitions."""

        groups: list[list[tuple[int, int, int]]] = []
        current: list[tuple[int, int, int]] = []
        transitions = 0
        for sequence in sequences:
            length = sequence[2] - sequence[1]
            if current and transitions + length > self.batch_size:
                groups.append(current)
                current, transitions = [], 0
            current.append(sequence)
            transitions += length
        if current:
            groups.append(current)
        return groups

    def _independent_gru_sequence_batch(self, sequences: list[tuple[int, int, int]]):
        """Build a padded recurrent critic batch from time-major rollout arrays."""

        buffer = self.rollout_buffer
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
        observations_tensor = torch.as_tensor(observations.reshape(n_seq * max_length, observation_size), device=self.device)
        starts_tensor = torch.as_tensor(episode_starts.reshape(-1), device=self.device)
        returns_tensor = torch.as_tensor(returns.reshape(-1), device=self.device)
        mask_tensor = torch.as_tensor(valid.reshape(-1), device=self.device)
        hidden_tensor = torch.as_tensor(hidden, device=self.device)
        states = (hidden_tensor, torch.zeros_like(hidden_tensor))
        return observations_tensor, states, starts_tensor, valid_by_timestep, returns_tensor, mask_tensor

    def _independent_gru_sequence_loss(self, sequences: list[tuple[int, int, int]]) -> tuple[torch.Tensor, int]:
        observations, states, starts, valid_by_timestep, returns, mask = self._independent_gru_sequence_batch(sequences)
        values = self.policy.evaluate_values_independent_gru(observations, states, starts, valid_by_timestep).flatten()
        return torch.nn.functional.mse_loss(values[mask], returns[mask]), int(mask.sum().item())

    def _independent_gru_validation_loss(self, validation_sequences: list[tuple[int, int, int]]) -> float:
        losses = []
        with torch.no_grad():
            for group in self._pack_sequences(validation_sequences):
                loss, count = self._independent_gru_sequence_loss(group)
                losses.append((float(loss.item()), count))
        return sum(loss * count for loss, count in losses) / sum(count for _loss, count in losses)

    def _apply_critic_gradient(self, loss: torch.Tensor, loss_name: str) -> None:
        require_finite_tensor(loss_name, loss)
        self.policy.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters, CONFIG.max_grad_norm)
        require_finite_tensor(f"{loss_name} gradient norm", grad_norm)
        self.policy.critic_optimizer.step()

    def _warmup_critic(self) -> None:
        train_sequences, validation_sequences = self._warmup_split()
        independent_gru = self.policy.critic_is_independent_gru
        if not independent_gru:
            flat_inputs = self._flat_critic_inputs()
            flat_returns = self.rollout_buffer.returns.reshape(-1)
            train_indices = self._flat_sequence_indices(train_sequences)
            validation_indices = self._flat_sequence_indices(validation_sequences)
        best_loss = float("inf")
        best_critic = None
        best_optimizer = None
        best_epoch = 0
        stale_epochs = 0
        warmup_train_losses: list[float] = []
        warmup_validation_losses: list[float] = []
        for epoch in range(CONFIG.warmup_max_epochs):
            epoch_train_losses: list[tuple[float, int]] = []
            if independent_gru:
                order = self.warmup_shuffle_rng.permutation(len(train_sequences))
                shuffled_sequences = [train_sequences[int(index)] for index in order]
                for group in self._pack_sequences(shuffled_sequences):
                    group_loss, count = self._independent_gru_sequence_loss(group)
                    self._apply_critic_gradient(CONFIG.value_loss_coefficient * group_loss, "Warm-up loss")
                    epoch_train_losses.append((float(group_loss.item()), count))
                validation_loss = self._independent_gru_validation_loss(validation_sequences)
            else:
                shuffled = self.warmup_shuffle_rng.permutation(train_indices)
                for start in range(0, len(shuffled), self.batch_size):
                    batch = shuffled[start : start + self.batch_size]
                    batch_loss = self._critic_batch_loss(flat_inputs, flat_returns, batch)
                    self._apply_critic_gradient(CONFIG.value_loss_coefficient * batch_loss, "Warm-up loss")
                    epoch_train_losses.append((float(batch_loss.item()), len(batch)))
                validation_loss = self._validation_loss(flat_inputs, flat_returns, validation_indices)
            train_loss = sum(loss * count for loss, count in epoch_train_losses) / sum(
                count for _loss, count in epoch_train_losses
            )
            require_finite_number("Warm-up train loss", train_loss)
            require_finite_number("Warm-up validation loss", validation_loss)
            warmup_train_losses.append(train_loss)
            warmup_validation_losses.append(validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_critic = copy_module.deepcopy(self.policy.value_net.state_dict())
                best_optimizer = copy_module.deepcopy(self.policy.critic_optimizer.state_dict())
                best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= CONFIG.warmup_patience:
                    break
        if best_critic is None or best_optimizer is None:
            raise RuntimeError("Critic warm-up did not produce a valid checkpoint")
        self.policy.value_net.load_state_dict(best_critic)
        self.policy.critic_optimizer.load_state_dict(best_optimizer)
        self.warmup_completed = True
        metrics = {
            "phase": "warmup",
            "epochs": epoch + 1,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "warmup_train_losses": warmup_train_losses,
            "warmup_validation_losses": warmup_validation_losses,
        }
        checkpoint_path = self.recorder.save_warmup_critic(self.policy.value_net.state_dict())
        metrics["critic_checkpoint"] = str(checkpoint_path)
        self.recorder.record_metrics(metrics)
        print(f"Warm-up complete: best_validation_loss={best_loss:.6f}, checkpoint={checkpoint_path}", flush=True)

    def _batch_values(self, rollout_data) -> torch.Tensor:
        """Critic values for one recurrent minibatch, dispatched by critic variant."""

        if self.policy.critic_is_independent_gru:
            critic_states = (rollout_data.lstm_states.vf[0], rollout_data.lstm_states.vf[1])
            return self.policy.evaluate_values_independent_gru(
                rollout_data.observations, critic_states, rollout_data.episode_starts, self.rollout_buffer.current_valid_by_timestep
            ).flatten()
        return self.policy.evaluate_values(rollout_data.observations).flatten()

    @staticmethod
    def _mean_episode_metric(records: list[dict], name: str) -> float | None:
        if not records:
            return None
        return float(np.mean([record[name] for record in records]))

    @classmethod
    def _episode_metrics(cls, episodes: list[dict]) -> dict[str, float | int | None]:
        metrics: dict[str, float | int | None] = {
            "episode_count": len(episodes),
            "mean_episode_steps": cls._mean_episode_metric(episodes, "episode_steps"),
            "mean_episode_return": cls._mean_episode_metric(episodes, "episode_return"),
        }
        for role in ("collision", "ordinary"):
            role_episodes = [record for record in episodes if record["env_role"] == role]
            metrics[f"{role}_role_episode_count"] = len(role_episodes)
            metrics[f"mean_{role}_episode_return"] = cls._mean_episode_metric(role_episodes, "episode_return")
        return metrics

    def train(self) -> None:
        self.policy.set_training_mode(True)
        if not self.warmup_completed:
            self._warmup_critic()
            return

        clip_range = self.clip_range(self._current_progress_remaining)
        policy_losses = []
        clip_fractions = []
        approximate_kls = []
        preference_losses = []
        preference_target_losses = []
        preference_control_losses = []
        preference_margins = []
        online_branch_losses = []
        online_branch_approximate_kls = []
        online_branch_clip_fractions = []
        branched_ppo_enabled = self.online_same_state_branch_ppo or self.collision_prefix_branch_ppo
        branch_loss_coefficient = CONFIG.online_branch_loss_coefficient if self.online_same_state_branch_ppo else CONFIG.collision_prefix_branch_loss_coefficient
        update = self._n_updates + 1
        actor_optimizer_steps_per_epoch = (
            self.actor_epochs
            * self.rollout_buffer.buffer_size
            * self.rollout_buffer.n_envs
            // self.batch_size
        ) // self.actor_epochs
        online_branch_ratio_identity = None
        if branched_ppo_enabled:
            if self.online_branch_rollout is None:
                raise RuntimeError("Online branch rollout storage is missing before actor training")
            online_branch_ratio_identity = self.online_branch_rollout.ratio_identity(self.policy)
            if online_branch_ratio_identity > 5e-5:
                raise RuntimeError("Online branch pre-update actor log-ratio identity failed")
        self._calibrate_first_action_preference(clip_range)

        for parameter in self.policy.critic_parameters:
            parameter.requires_grad_(False)
        for epoch in range(self.actor_epochs):
            online_branch_batches = (
                self.online_branch_rollout.epoch_batches(actor_optimizer_steps_per_epoch, self.online_branch_minibatch_rng, require_even=self.online_same_state_branch_ppo)
                if branched_ppo_enabled and self.online_branch_rollout is not None
                else None
            )
            for minibatch, rollout_data in enumerate(
                self.rollout_buffer.get(self.batch_size, rng=self.actor_minibatch_rng),
                start=1,
            ):
                mask = rollout_data.mask > 1e-8
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    valid_advantages = advantages[mask]
                    advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
                log_prob, _entropy = self.policy.evaluate_actor_actions(
                    rollout_data.observations,
                    rollout_data.actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approximate_kl = float(
                        ((torch.exp(log_ratio) - 1) - log_ratio)[mask].mean().cpu().item()
                    )
                    require_finite_number("Approximate KL", approximate_kl)
                    approximate_kls.append(approximate_kl)
                    clip_fractions.append(float((torch.abs(ratio - 1) > clip_range)[mask].float().mean().cpu().item()))
                policy_loss = -torch.min(advantages * ratio, advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range))[mask].mean()
                require_finite_tensor("Policy loss", policy_loss)
                self.policy.actor_optimizer.zero_grad()
                if branched_ppo_enabled:
                    if online_branch_batches is None or self.online_branch_rollout is None:
                        raise RuntimeError("Online branch actor minibatches are missing")
                    branch_indices = online_branch_batches[minibatch - 1]
                    if len(branch_indices):
                        branch_loss, branch_kl, branch_clip_fraction = self.online_branch_rollout.loss(self.policy, branch_indices, clip_range)
                        combined_loss = policy_loss + branch_loss_coefficient * branch_loss
                        require_finite_tensor("Combined main and branched PPO loss", combined_loss)
                        combined_loss.backward()
                        online_branch_losses.append(float(branch_loss.detach().cpu().item()))
                        online_branch_approximate_kls.append(branch_kl)
                        online_branch_clip_fractions.append(branch_clip_fraction)
                    else:
                        policy_loss.backward()
                elif self.first_action_preference is None:
                    policy_loss.backward()
                else:
                    preference_loss, preference_target_loss, preference_control_loss, preference_batch_margins = self.first_action_preference.loss()
                    require_finite_tensor("First-action preference loss", preference_loss)
                    if self.first_action_preference_beta is None:
                        raise RuntimeError("First-action preference beta was not calibrated")
                    combined_loss = policy_loss + self.first_action_preference_beta * preference_loss
                    require_finite_tensor("Combined PPO and first-action preference loss", combined_loss)
                    combined_loss.backward()
                    preference_losses.append(float(preference_loss.detach().cpu().item()))
                    preference_target_losses.append(float(preference_target_loss.detach().cpu().item()))
                    preference_control_losses.append(float(preference_control_loss.detach().cpu().item()))
                    preference_margins.extend(float(value) for value in preference_batch_margins.detach().cpu().tolist())
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters, CONFIG.max_grad_norm)
                require_finite_tensor("Actor gradient norm", grad_norm)
                self.policy.actor_optimizer.step()
                policy_losses.append(float(policy_loss.item()))
        for parameter in self.policy.critic_parameters:
            parameter.requires_grad_(True)

        for parameter in self.policy.actor_parameters:
            parameter.requires_grad_(False)
        value_loss_samples: list[tuple[float, int]] = []
        critic_epoch_value_losses: list[float] = []
        for _epoch in range(self.critic_epochs):
            epoch_value_losses: list[tuple[float, int]] = []
            for rollout_data in self.rollout_buffer.get(self.batch_size, rng=self.critic_minibatch_rng):
                mask = rollout_data.mask > 1e-8
                values = self._batch_values(rollout_data)
                value_loss = torch.nn.functional.mse_loss(values[mask], rollout_data.returns[mask])
                self._apply_critic_gradient(CONFIG.value_loss_coefficient * value_loss, "Value loss")
                sample = (float(value_loss.item()), int(mask.sum().item()))
                epoch_value_losses.append(sample)
                value_loss_samples.append(sample)
            critic_epoch_value_losses.append(
                sum(loss * count for loss, count in epoch_value_losses)
                / sum(count for _loss, count in epoch_value_losses)
            )
        for parameter in self.policy.actor_parameters:
            parameter.requires_grad_(True)

        self._n_updates += 1
        policy_gradient_loss = float(np.mean(policy_losses)) if policy_losses else 0.0
        value_loss_mean = sum(loss * count for loss, count in value_loss_samples) / sum(
            count for _loss, count in value_loss_samples
        )
        approximate_kl_mean = float(np.mean(approximate_kls))
        clip_fraction_mean = float(np.mean(clip_fractions))
        episodes = self._rollout_episode_records
        collision_times = [record["elapsed_time"] for record in episodes if record["episode_outcome"] == "ego_collision"]
        episode_metrics = self._episode_metrics(episodes)
        metrics = {
            "phase": "formal",
            "update": update,
            "rollout_policy_update": update - 1,
            "num_timesteps": self.num_timesteps,
            "policy_gradient_loss": policy_gradient_loss,
            "value_loss": value_loss_mean,
            "critic_epoch_value_losses": critic_epoch_value_losses,
            "approx_kl_mean": approximate_kl_mean,
            "clip_fraction_mean": clip_fraction_mean,
            "ppo_loss_sample_stride": self.ppo_loss_sample_stride,
            "ppo_loss_selected_transition_count": int(self.rollout_buffer.loss_sample_mask.sum()),
            "ppo_loss_total_transition_count": int(self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs),
            "ego_collision_count": sum(record["episode_outcome"] == "ego_collision" for record in episodes),
            "overtake_count": sum(record["episode_outcome"] == "overtake" for record in episodes),
            "follow_count": sum(record["episode_outcome"] == "follow" for record in episodes),
            "mean_ego_collision_time": float(np.mean(collision_times)) if collision_times else None,
            **episode_metrics,
        }
        if self.first_action_preference is not None:
            margin_array = np.asarray(preference_margins, dtype=np.float64)
            metrics.update({
                "first_action_preference_beta": self.first_action_preference_beta,
                "first_action_preference_loss_mean": float(np.mean(preference_losses)),
                "first_action_preference_target_loss_mean": float(np.mean(preference_target_losses)),
                "first_action_preference_control_loss_mean": float(np.mean(preference_control_losses)),
                "first_action_preference_margin_mean": float(np.mean(margin_array)),
                "first_action_preference_satisfied_fraction": float(np.mean(margin_array > 0.0)),
            })
        if branched_ppo_enabled:
            if self.online_branch_rollout is None or online_branch_ratio_identity is None:
                raise RuntimeError("Online branch metrics are unavailable")
            prefix = "online_branch" if self.online_same_state_branch_ppo else "collision_prefix_branch"
            metrics.update({
                f"{prefix}_preupdate_max_abs_log_ratio": online_branch_ratio_identity,
                f"{prefix}_policy_loss_mean": float(np.mean(online_branch_losses)) if online_branch_losses else None,
                f"{prefix}_approx_kl_mean": float(np.mean(online_branch_approximate_kls)) if online_branch_approximate_kls else None,
                f"{prefix}_clip_fraction_mean": float(np.mean(online_branch_clip_fractions)) if online_branch_clip_fractions else None,
                **self.online_branch_rollout.statistics(prefix),
            })
        actor_path, critic_path = self.recorder.save_formal_checkpoints(update, self.policy.actor_checkpoint_state_dict(), self.policy.value_net.state_dict())
        metrics["actor_checkpoint"] = str(actor_path)
        metrics["critic_checkpoint"] = str(critic_path)
        self.recorder.record_metrics(metrics)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/policy_gradient_loss", policy_gradient_loss)
        self.logger.record("train/value_loss", value_loss_mean)
        self.logger.record("train/approx_kl", approximate_kl_mean)
        self.logger.record("train/clip_fraction", clip_fraction_mean)
        print(f"Formal update {update}: policy_loss={policy_gradient_loss:.6f}, value_loss={value_loss_mean:.6f}, actor={actor_path}, critic={critic_path}", flush=True)
