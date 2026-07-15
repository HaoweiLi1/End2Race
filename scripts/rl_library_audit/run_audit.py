#!/usr/bin/env python3
"""Reproducible recurrent-PPO correctness audit for the pinned End2Race env.

This file intentionally does not import or train the End2Race learner.  The only
training updates below are tiny toy PPO updates with learning_rate=0.
"""

from __future__ import annotations

import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.distributions import Categorical

import sb3_contrib
import stable_baselines3
import tianshou
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer, create_sequencers, pad
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent import RecurrentPPO as RecurrentPPOClass
from tianshou.data import Batch, Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PGPolicy, PPOPolicy
from tianshou.utils.net.common import Recurrent as TianshouRecurrent


ROOT = Path(__file__).resolve().parent
RESULT_PATH = ROOT / "results.json"


def verdict(error: float) -> str:
    if error <= 1e-6:
        return "PASS"
    if error <= 1e-4:
        return "WARN"
    return "FAIL"


class AlternatingEndEnv(gym.Env):
    """Constant-observation env: first episode terminates, second times out."""

    metadata = {"render_modes": []}

    def __init__(self, horizon: int = 20):
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.horizon = horizon
        self.episode_index = -1
        self.step_index = 0
        self.completed_end_types: list[str] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.episode_index += 1
        self.step_index = 0
        return np.ones((1,), dtype=np.float32), {"episode_index": self.episode_index}

    def step(self, action: int):
        del action
        self.step_index += 1
        terminated = self.step_index >= self.horizon and self.episode_index % 2 == 0
        truncated = self.step_index >= self.horizon and self.episode_index % 2 == 1
        if terminated:
            self.completed_end_types.append("terminated")
        if truncated:
            self.completed_end_types.append("truncated")
        info = {"episode_index": self.episode_index, "step_index": self.step_index}
        return np.ones((1,), dtype=np.float32), 1.0, terminated, truncated, info


class AccumulatingGRUActor(nn.Module):
    """A real GRU whose categorical action flips as constant input accumulates."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, 1, batch_first=True)
        with torch.no_grad():
            for parameter in self.gru.parameters():
                parameter.zero_()
            # PyTorch GRU gate order is reset, update, new.  A large update-gate
            # bias makes h grow gradually, exposing history under constant obs.
            self.gru.bias_ih_l0[1] = 2.0
            self.gru.weight_ih_l0[2, 0] = 1.0

    def forward(self, obs: Any, state: Batch | dict | None = None, info: Any = None):
        del info
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 2:
            obs_tensor = obs_tensor.unsqueeze(1)
        if state is None:
            hidden = torch.zeros((1, obs_tensor.shape[0], 1), dtype=obs_tensor.dtype)
        else:
            hidden_value = state.h if isinstance(state, Batch) else state["h"]
            hidden = torch.as_tensor(hidden_value, dtype=obs_tensor.dtype).transpose(0, 1).contiguous()
        output, next_hidden = self.gru(obs_tensor, hidden)
        signal = torch.sin(25.0 * output[:, -1, 0])
        categorical_logits = torch.stack((-6.0 * signal, 6.0 * signal), dim=-1)
        probabilities = torch.softmax(categorical_logits, dim=-1)
        # Tianshou stores batch-major recurrent state.
        next_state = Batch(h=next_hidden.transpose(0, 1).detach())
        return probabilities, next_state


class ZeroCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, obs: Any):
        batch_size = np.asarray(obs).shape[0] if not torch.is_tensor(obs) else obs.shape[0]
        # Multiplication (rather than a Parameter view via expand) honors the
        # torch.no_grad() block used by Tianshou 0.5.1 return computation.
        return self.anchor * torch.ones((batch_size, 1), device=self.anchor.device)


class LoggingOnlyPPOPolicy(PPOPolicy):
    """Adds rollout log-probability telemetry; it does not alter state handling."""

    def forward(self, batch: Batch, state: Any = None, **kwargs: Any) -> Batch:
        result = super().forward(batch, state=state, **kwargs)
        result.policy = Batch(
            rollout_logp=result.dist.log_prob(result.act).detach().cpu().numpy(),
        )
        return result


def max_parameter_delta(before: dict[str, torch.Tensor], module: nn.Module) -> float:
    delta = 0.0
    for name, value in module.state_dict().items():
        if torch.is_floating_point(value):
            delta = max(delta, float((value.detach().cpu() - before[name]).abs().max()))
    return delta


def run_tianshou_test() -> dict[str, Any]:
    torch.manual_seed(20260715)
    np.random.seed(20260715)
    raw_env = AlternatingEndEnv(horizon=20)
    vector_env = DummyVectorEnv([lambda: raw_env])
    actor = AccumulatingGRUActor()
    critic = ZeroCritic()
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0.0)
    policy = LoggingOnlyPPOPolicy(
        actor,
        critic,
        optimizer,
        Categorical,
        action_space=raw_env.action_space,
        action_scaling=False,
        advantage_normalization=False,
    )
    buffer = ReplayBuffer(size=64, stack_num=1)
    collector = Collector(policy, vector_env, buffer)
    stats = collector.collect(n_episode=2)

    batch, indices = buffer.sample(0)
    rollout_logp = np.asarray(batch.policy.rollout_logp).reshape(-1).copy()
    stored_post_hidden = np.asarray(batch.policy.hidden_state.h).reshape(len(batch), -1)[:, 0].copy()
    processed = policy.process_fn(batch, buffer, indices)
    replay_logp = processed.logp_old.detach().cpu().numpy().reshape(-1).copy()
    errors = np.abs(replay_logp - rollout_logp)

    # Prove that history, not observation, changes the deterministic action.
    recurrent_state = None
    greedy_actions: list[int] = []
    hidden_trace: list[float] = []
    for _ in range(20):
        probabilities, recurrent_state = actor(np.ones((1, 1), dtype=np.float32), recurrent_state)
        greedy_actions.append(int(probabilities.argmax(dim=-1).item()))
        hidden_trace.append(float(recurrent_state.h.reshape(-1)[0]))

    before = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
    policy.learn(processed, batch_size=10, repeat=1)
    parameter_delta = max_parameter_delta(before, policy)
    vector_env.close()

    max_error = float(errors.max())
    return {
        "transitions": int(len(processed)),
        "episodes": int(stats["n/ep"]),
        "episode_lengths": np.asarray(stats["lens"]).astype(int).tolist(),
        "episode_end_types": raw_env.completed_end_types[:2],
        "all_observations_identical": bool(np.all(np.asarray(processed.obs) == 1.0)),
        "history_greedy_actions": greedy_actions,
        "history_action_unique_count": len(set(greedy_actions)),
        "hidden_trace_first_last": [hidden_trace[0], hidden_trace[-1]],
        "buffer_hidden_semantics_probe": {
            "stored_first_hidden": float(stored_post_hidden[0]),
            "stored_second_hidden": float(stored_post_hidden[1]),
            "note": "These are actor-returned post-observation states for each transition.",
        },
        "rollout_logp_first_10": rollout_logp[:10].tolist(),
        "replay_logp_first_10": replay_logp[:10].tolist(),
        "max_abs_replay_logp_minus_rollout_logp": max_error,
        "mean_abs_error": float(errors.mean()),
        "verdict": verdict(max_error),
        "optimizer_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "max_parameter_delta_after_learn": parameter_delta,
        "logging_subclass_scope": "telemetry only: adds result.policy.rollout_logp; no recurrent state or PPO logic changed",
    }


def configure_stock_lstm_for_visible_history(model: RecurrentPPO) -> None:
    """Keep stock policy code, but choose deterministic weights with visible memory."""

    lstm = model.policy.lstm_actor
    assert lstm.hidden_size == 1 and lstm.input_size == 1
    with torch.no_grad():
        for parameter in lstm.parameters():
            parameter.zero_()
        # PyTorch LSTM gate order: input, forget, cell, output.
        lstm.bias_ih_l0[0] = 3.0
        lstm.bias_ih_l0[1] = 2.0
        lstm.weight_ih_l0[2, 0] = 1.0
        lstm.bias_ih_l0[3] = 5.0
        model.policy.action_net.weight.copy_(torch.tensor([[-30.0], [30.0]]))
        model.policy.action_net.bias.copy_(torch.tensor([24.0, -24.0]))


def deterministic_sb3_history_actions(model: RecurrentPPO, length: int = 20) -> list[int]:
    shape = (1, 1, 1)
    zero = torch.zeros(shape, device=model.device)
    states = RNNStates((zero.clone(), zero.clone()), (zero.clone(), zero.clone()))
    actions: list[int] = []
    for step in range(length):
        episode_starts = torch.tensor([step == 0], dtype=torch.float32, device=model.device)
        with torch.no_grad():
            action, _, _, states = model.policy.forward(
                torch.ones((1, 1), device=model.device), states, episode_starts, deterministic=True
            )
        actions.append(int(action.item()))
    return actions


def setup_and_collect(model: RecurrentPPO, total_timesteps: int = 40):
    _, callback = model._setup_learn(
        total_timesteps,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="recurrent_audit",
        progress_bar=False,
    )
    callback.on_training_start(locals(), globals())
    assert model.env is not None
    ok = model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)
    if not ok:
        raise RuntimeError("Toy rollout was interrupted")
    return callback


def evaluate_sb3_buffer(model: RecurrentPPO, random_seed: int = 1) -> dict[str, Any]:
    np.random.seed(random_seed)
    valid_errors: list[np.ndarray] = []
    padding_errors: list[np.ndarray] = []
    valid_count = 0
    padded_count = 0
    minibatch_shapes: list[dict[str, int]] = []
    for rollout_data in model.rollout_buffer.get(model.batch_size):
        actions = rollout_data.actions.long().flatten()
        mask = rollout_data.mask > 1e-8
        with torch.no_grad():
            _, replay_logp, _ = model.policy.evaluate_actions(
                rollout_data.observations,
                actions,
                rollout_data.lstm_states,
                rollout_data.episode_starts,
            )
        error = (replay_logp - rollout_data.old_log_prob).abs().detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        valid_errors.append(error[mask_np])
        if np.any(~mask_np):
            padding_errors.append(error[~mask_np])
        valid = int(mask_np.sum())
        padded = int((~mask_np).sum())
        valid_count += valid
        padded_count += padded
        minibatch_shapes.append({"valid": valid, "padded": padded, "flat_size": int(mask.numel())})
    valid_error = np.concatenate(valid_errors)
    padding_error = np.concatenate(padding_errors) if padding_errors else np.array([], dtype=np.float32)
    return {
        "valid_timestep_count": valid_count,
        "padding_timestep_count": padded_count,
        "minibatches": minibatch_shapes,
        "max_abs_replay_logp_minus_rollout_logp": float(valid_error.max()),
        "mean_abs_error": float(valid_error.mean()),
        "max_abs_error_on_padding_not_used_for_verdict": float(padding_error.max()) if padding_error.size else None,
        "verdict": verdict(float(valid_error.max())),
    }


def make_sb3_model(policy: Any, raw_env: gym.Env) -> RecurrentPPO:
    return RecurrentPPO(
        policy,
        raw_env,
        learning_rate=0.0,
        n_steps=40,
        batch_size=10,
        n_epochs=1,
        seed=20260715,
        device="cpu",
        policy_kwargs={"lstm_hidden_size": 1, "n_lstm_layers": 1, "net_arch": [], "ortho_init": False},
        verbose=0,
    )


def run_sb3_lstm_test() -> dict[str, Any]:
    torch.manual_seed(20260715)
    np.random.seed(20260715)
    raw_env = AlternatingEndEnv(horizon=20)
    model = make_sb3_model("MlpLstmPolicy", raw_env)
    configure_stock_lstm_for_visible_history(model)
    greedy_actions = deterministic_sb3_history_actions(model)
    callback = setup_and_collect(model)

    raw_rewards = model.rollout_buffer.rewards.copy().reshape(-1)
    raw_episode_starts = model.rollout_buffer.episode_starts.copy().reshape(-1)
    pre_action_hidden = model.rollout_buffer.hidden_states_pi.copy().reshape(40, -1)[:, 0]
    comparison = evaluate_sb3_buffer(model)

    before = {name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()}
    model.train()
    parameter_delta = max_parameter_delta(before, model.policy)
    callback.on_training_end()
    model.env.close()

    comparison.update(
        {
            "transitions": 40,
            "episode_end_types": raw_env.completed_end_types[:2],
            "episode_start_indices": np.where(raw_episode_starts > 0.5)[0].astype(int).tolist(),
            "all_observations_identical": True,
            "history_greedy_actions": greedy_actions,
            "history_action_unique_count": len(set(greedy_actions)),
            "pre_action_hidden_first_three": pre_action_hidden[:3].tolist(),
            "normal_terminal_buffer_reward": float(raw_rewards[19]),
            "timeout_buffer_reward_after_bootstrap": float(raw_rewards[39]),
            "timeout_reward_was_bootstrapped": bool(not np.isclose(raw_rewards[39], 1.0)),
            "optimizer_learning_rate": float(model.policy.optimizer.param_groups[0]["lr"]),
            "max_parameter_delta_after_train": parameter_delta,
            "policy_class": f"{model.policy.__class__.__module__}.{model.policy.__class__.__qualname__}",
        }
    )
    return comparison


class GRUDummyCellPolicy(RecurrentActorCriticPolicy):
    """Audit-only proof that the existing LSTM-shaped API can transport GRU state."""

    def __init__(self, *args: Any, **kwargs: Any):
        lr_schedule = args[2] if len(args) >= 3 else kwargs["lr_schedule"]
        super().__init__(*args, **kwargs)
        self.lstm_actor = nn.GRU(
            self.features_dim,
            self.lstm_output_dim,
            num_layers=self.lstm_actor.num_layers,
        )
        if self.lstm_critic is not None:
            self.lstm_critic = nn.GRU(
                self.features_dim,
                self.lstm_output_dim,
                num_layers=self.lstm_actor.num_layers,
            )
        # The parent built its optimizer before the audit-only module swap.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    @staticmethod
    def _process_sequence(
        features: torch.Tensor,
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        recurrent_module: nn.Module,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not isinstance(recurrent_module, nn.GRU):
            return RecurrentActorCriticPolicy._process_sequence(
                features, lstm_states, episode_starts, recurrent_module  # type: ignore[arg-type]
            )
        n_seq = lstm_states[0].shape[1]
        feature_sequence = features.reshape((n_seq, -1, recurrent_module.input_size)).swapaxes(0, 1)
        start_sequence = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        hidden = lstm_states[0]  # real GRU state
        outputs = []
        for feature, episode_start in zip(feature_sequence, start_sequence):
            hidden = (1.0 - episode_start).view(1, n_seq, 1) * hidden
            output, hidden = recurrent_module(feature.unsqueeze(0), hidden)
            outputs.append(output)
        flat_output = torch.flatten(torch.cat(outputs).transpose(0, 1), start_dim=0, end_dim=1)
        dummy_cell = torch.zeros_like(hidden)
        return flat_output, (hidden, dummy_cell)


def run_gru_dummy_cell_smoke_test() -> dict[str, Any]:
    torch.manual_seed(20260715)
    np.random.seed(20260715)
    raw_env = AlternatingEndEnv(horizon=20)
    model = make_sb3_model(GRUDummyCellPolicy, raw_env)
    callback = setup_and_collect(model)
    actor_cells = model.rollout_buffer.cell_states_pi.copy()
    critic_cells = model.rollout_buffer.cell_states_vf.copy()
    comparison = evaluate_sb3_buffer(model, random_seed=1)
    before = {name: value.detach().cpu().clone() for name, value in model.policy.state_dict().items()}
    model.train()
    parameter_delta = max_parameter_delta(before, model.policy)
    callback.on_training_end()
    model.env.close()
    return {
        "worked_without_modifying_recurrent_ppo": True,
        "worked_without_modifying_recurrent_rollout_buffer": True,
        "actor_buffer_cell_all_zero": bool(np.all(actor_cells == 0.0)),
        "critic_buffer_cell_all_zero": bool(np.all(critic_cells == 0.0)),
        "valid_logp_max_error": comparison["max_abs_replay_logp_minus_rollout_logp"],
        "valid_logp_verdict": comparison["verdict"],
        "optimizer_learning_rate": float(model.policy.optimizer.param_groups[0]["lr"]),
        "max_parameter_delta_after_train": parameter_delta,
        "required_customization": "custom policy _process_sequence plus optimizer rebuild; h is real state and c is ignored/zero",
    }


def source_location(obj: Any) -> dict[str, Any]:
    # getsource() is deliberate: it verifies that inspect can load the exact installed
    # implementation, not only a re-export or signature.
    source = inspect.getsource(obj)
    lines, start = inspect.getsourcelines(obj)
    return {
        "path": str(Path(inspect.getsourcefile(obj) or "").resolve()),
        "function": obj.__qualname__,
        "start_line": start,
        "end_line": start + len(lines) - 1,
        "source_characters_read_by_inspect_getsource": len(source),
    }


def collect_source_locations() -> dict[str, Any]:
    return {
        "tianshou": {
            "Collector.collect": source_location(Collector.collect),
            "Collector._reset_state": source_location(Collector._reset_state),
            "PGPolicy.forward": source_location(PGPolicy.forward),
            "PPOPolicy.process_fn": source_location(PPOPolicy.process_fn),
            "PPOPolicy.learn": source_location(PPOPolicy.learn),
            "ReplayBuffer": source_location(ReplayBuffer),
            "ReplayBuffer.add": source_location(ReplayBuffer.add),
            "ReplayBuffer.get": source_location(ReplayBuffer.get),
            "Batch.split": source_location(Batch.split),
            "Recurrent.forward": source_location(TianshouRecurrent.forward),
        },
        "sb3_contrib": {
            "RecurrentPPO._setup_model": source_location(RecurrentPPOClass._setup_model),
            "RecurrentPPO.collect_rollouts": source_location(RecurrentPPOClass.collect_rollouts),
            "RecurrentPPO.train": source_location(RecurrentPPOClass.train),
            "RecurrentRolloutBuffer": source_location(RecurrentRolloutBuffer),
            "RecurrentRolloutBuffer.get": source_location(RecurrentRolloutBuffer.get),
            "RecurrentRolloutBuffer._get_samples": source_location(RecurrentRolloutBuffer._get_samples),
            "create_sequencers": source_location(create_sequencers),
            "pad": source_location(pad),
            "RecurrentActorCriticPolicy": source_location(RecurrentActorCriticPolicy),
            "RecurrentActorCriticPolicy._process_sequence": source_location(
                RecurrentActorCriticPolicy._process_sequence
            ),
            "RecurrentActorCriticPolicy.evaluate_actions": source_location(
                RecurrentActorCriticPolicy.evaluate_actions
            ),
        },
    }


def pip_show(package: str) -> dict[str, str]:
    output = subprocess.check_output([sys.executable, "-m", "pip", "show", package], text=True)
    result: dict[str, str] = {}
    for line in output.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            result[key] = value
    return result


def main() -> None:
    results = {
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "gymnasium": gym.__version__,
            "tianshou": tianshou.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "sb3_contrib": sb3_contrib.__version__,
            "pip_show": {
                name: pip_show(name) for name in ("stable-baselines3", "sb3-contrib", "tianshou")
            },
        },
        "source_locations": collect_source_locations(),
        "dynamic_tests": {
            "tianshou_stock_ppo": run_tianshou_test(),
            "sb3_contrib_stock_lstm_recurrent_ppo": run_sb3_lstm_test(),
            "sb3_contrib_gru_dummy_cell_adapter": run_gru_dummy_cell_smoke_test(),
        },
        "acceptance_thresholds": {"PASS": "<= 1e-6", "WARN": "(1e-6, 1e-4]", "FAIL": "> 1e-4"},
        "scope": "toy environments only; no End2Race learner training; no site-packages modifications",
    }
    RESULT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results["dynamic_tests"], indent=2, sort_keys=True))
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
