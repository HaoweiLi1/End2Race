import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from sb3_contrib.common.recurrent.buffers import create_sequencers
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo.env import CentralScheduleSubprocVecEnv
from ppo.policy import BASELINE_EXPLORATION_MODE, END2RACE_OBSERVATION_SIZE, End2RaceGRUPolicy, PriviledgeMLPCritic
from ppo.rollout import End2RaceRecurrentPPO, End2RaceRolloutBuffer, MAX_GRAD_NORM, VALUE_LOSS_COEFFICIENT, WARMUP_MAX_EPOCHS, WARMUP_PATIENCE
from ppo.scenarios import ScenarioSpec, ordinary_scenarios
from stable_baselines3.common.utils import explained_variance
from train_ppo import START_METHOD, configure_training_numerics
from utils import TrainingRecorder, atomic_write_json, require_finite_number, require_finite_tensor


COST_GAMMA = 0.999
COST_GAE_LAMBDA = 0.995
COST_BUDGET = 0.10
INITIAL_DUAL = 1.0
DUAL_LEARNING_RATE = 0.5
MAXIMUM_DUAL = 20.0
COST_CRITIC_LEARNING_RATE = 3.0e-4
COST_CRITIC_EPOCHS = 5
COLLISION_REWARD = -2.0
OOF_FOLDS = 5
OOF_EPOCHS = 10
OOF_MINIMUM_MSE_SKILL = 0.05
OOF_MINIMUM_EPISODE_START_AUROC = 0.65
OOF_MINIMUM_EARLY_AUROC = 0.65
OOF_EARLY_REMAINING_STEPS = 100


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "preflight", "formal"), default="prepare")
    parser.add_argument("--gate-dir", type=str, default="eval_results/constrained_ppo_collision_cost_gate_v1")
    parser.add_argument("--output-dir", type=str, default="post-trained/constrained_ppo_collision_cost_v1")
    parser.add_argument("--actor-path", type=str, default="pretrained/end2race.pth")
    parser.add_argument("--collision-cache-dir", type=str, default="post-trained/collision-cache/pretrained_end2race_austin_collision_pool_479")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=12800)
    parser.add_argument("--num-updates", type=int, default=30)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def source_snapshot():
    head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(("git", "status", "--short"), cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    return {"git_head": head, "git_status_short": status}


def gate_plan(args):
    actor_path = Path(args.actor_path).expanduser().resolve()
    collision_path = Path(args.collision_cache_dir).expanduser().resolve() / "collision_scenarios.json"
    if not actor_path.is_file() or not collision_path.is_file():
        raise FileNotFoundError("Canonical actor or frozen collision pool is missing")
    if args.seed != 42 or args.n_envs != 16 or args.n_steps != 6400 or args.batch_size != 12800 or args.num_updates != 30:
        raise RuntimeError("Z9 fixed training contract changed")
    return {
        "schema_version": 1,
        "experiment_id": "constrained_ppo_collision_cost_v1",
        "gate": "Z9-A",
        "status": "frozen_before_execution",
        "inputs": {
            "actor_path": str(actor_path),
            "actor_sha256": sha256_file(actor_path),
            "collision_scenarios_path": str(collision_path),
            "collision_scenarios_sha256": sha256_file(collision_path),
        },
        "training_contract": {
            "map": "Austin",
            "actor_initialization": "canonical BC",
            "actor_output": "unchanged 361D input and 12-key checkpoint",
            "reward_objective": "existing reward with first-ego-collision reward component removed exactly",
            "cost": "one on the first ego-collision transition and zero otherwise",
            "cost_critic": "training-only P20 MLP 20-120-30-1",
            "cost_gamma": COST_GAMMA,
            "cost_gae_lambda": COST_GAE_LAMBDA,
            "cost_budget_completed_episode_rate": COST_BUDGET,
            "initial_dual": INITIAL_DUAL,
            "dual_learning_rate": DUAL_LEARNING_RATE,
            "dual_bounds": [0.0, MAXIMUM_DUAL],
            "dual_update": "once after every collected rollout from completed-episode ego-collision rate",
            "actor_advantage": "normalize A_reward-lambda*A_cost once, then standard clipped PPO surrogate",
            "cost_critic_learning_rate": COST_CRITIC_LEARNING_RATE,
            "cost_critic_epochs": COST_CRITIC_EPOCHS,
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "formal_updates": args.num_updates,
            "actor_epochs": 2,
            "reward_critic_epochs": 5,
            "role_distribution": "8 collision and 8 ordinary logical envs",
            "exploration": BASELINE_EXPLORATION_MODE,
            "seed": args.seed,
        },
        "preflight_contract": {
            "transition_count": args.n_envs * args.n_steps,
            "reward_collision_unique_pricing_max_abs_error": 0.0,
            "cost_event_equals_completed_collision_episode_count": True,
            "cost_advantage_minimum_std": 1.0e-3,
            "combined_actor_gradient_minimum_relative_l2_change": 1.0e-4,
            "startpoint_grouped_oof_folds": OOF_FOLDS,
            "oof_epochs": OOF_EPOCHS,
            "oof_minimum_mse_skill_over_train_mean": OOF_MINIMUM_MSE_SKILL,
            "oof_minimum_episode_start_auroc": OOF_MINIMUM_EPISODE_START_AUROC,
            "oof_minimum_early_auroc": OOF_MINIMUM_EARLY_AUROC,
            "oof_early_remaining_steps": OOF_EARLY_REMAINING_STEPS,
            "failure_action": "stop this implementation without claiming the Constrained PPO class is refuted",
            "pass_action": "run exactly one 30-update formal Austin trajectory",
        },
        "formal_acceptance": {
            "fixed_checkpoint_band": [27, 28, 29, 30],
            "final_joint_target": "four-map collision < 40 and overtake > 1500",
            "minimum_product_gate": "at U30 every map collision <= canonical BC and overtake >= canonical BC",
            "late_stability": "at least two consecutive checkpoints in U27-U30 pass every per-map BC floor",
            "evaluation": "CUDA deterministic four maps times 600, saved numeric traces, paired identity tests",
            "no_checkpoint_selection": True,
            "no_extension_or_dual_sweep": True,
        },
        "evidence_boundary": "A failed preflight stops only this exact implementation. A failed formal run closes only this fixed objective, budget, critic, and dual configuration.",
        "source_snapshot": source_snapshot(),
    }


def write_gate_plan(args):
    directory = Path(args.gate_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "gate_plan.json"
    plan = gate_plan(args)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != plan:
            raise RuntimeError("Existing Z9 gate plan differs from current frozen contract")
    else:
        if any(directory.iterdir()):
            raise RuntimeError("Z9 gate directory is nonempty before plan creation")
        atomic_write_json(path, plan)
    return plan


class ConstrainedRolloutBuffer(End2RaceRolloutBuffer):

    def reset(self):
        super().reset()
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.current_cost_advantages = None
        self.current_cost_returns = None

    def get(self, batch_size=None, *, rng):
        if not self.full:
            raise RuntimeError("Rollout buffer must be full before training")
        if not self.generator_ready:
            self.hidden_states_pi = self.hidden_states_pi.swapaxes(1, 2)
            names = [
                "observations", "actions", "values", "log_probs", "advantages", "returns", "hidden_states_pi", "episode_starts", "recurrent_resets",
                "exploration_speed_log_stds", "exploration_danger_gates", "exploration_temporal_active", "exploration_block_ids", "exploration_standard_residuals",
                "costs", "cost_values", "cost_advantages", "cost_returns",
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

    def _get_samples(self, batch_inds, env_change, env=None):
        sample = super()._get_samples(batch_inds, env_change, env)
        self.current_cost_advantages = self.to_torch(self.pad_and_flatten(self.cost_advantages[batch_inds]))
        self.current_cost_returns = self.to_torch(self.pad_and_flatten(self.cost_returns[batch_inds]))
        return sample


class ConstrainedEnd2RacePPO(End2RaceRecurrentPPO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed = int(np.random.SeedSequence([self.seed, 0x434F5354]).generate_state(1)[0])
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            self.cost_value_net = PriviledgeMLPCritic().to(self.device)
        self.cost_parameters = tuple(self.cost_value_net.parameters())
        self.cost_optimizer = torch.optim.Adam(self.cost_parameters, lr=COST_CRITIC_LEARNING_RATE)
        self.cost_minibatch_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0x434D494E]))
        self.cost_telemetry_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0x4354454C]))
        self.lambda_cost = INITIAL_DUAL
        self._capture_cost_info = False
        self._cost_rows = []
        self._collision_reward_rows = []
        self._done_rows = []
        self._ego_index_rows = []
        self._scenario_id_rows = []
        self.last_constraint_rollout = None

    def _setup_model(self):
        super()._setup_model()
        source = self.rollout_buffer
        self.rollout_buffer = ConstrainedRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            source.hidden_state_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            store_independent_gru_hidden=self.policy.critic_is_independent_gru,
        )
        self.policy._end2race_rollout_buffer = self.rollout_buffer

    def _update_info_buffer(self, infos, dones=None):
        if self._capture_cost_info:
            if dones is None:
                raise RuntimeError("Constraint capture requires done markers")
            self._cost_rows.append([float(bool(info["ego_collision"])) for info in infos])
            self._collision_reward_rows.append([float(info["reward_collision"]) for info in infos])
            self._done_rows.append([bool(value) for value in dones])
            self._ego_index_rows.append([int(info["scenario"]["ego_idx"]) for info in infos])
            self._scenario_id_rows.append([str(info["scenario_id"]) for info in infos])
        super()._update_info_buffer(infos, dones)

    @staticmethod
    def _gae(rewards, values, episode_starts, last_values, dones, gamma, gae_lambda):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
        for step in reversed(range(rewards.shape[0])):
            if step == rewards.shape[0] - 1:
                next_non_terminal = 1.0 - np.asarray(dones, dtype=np.float32)
                next_values = np.asarray(last_values, dtype=np.float32)
            else:
                next_non_terminal = 1.0 - np.asarray(episode_starts[step + 1], dtype=np.float32)
                next_values = values[step + 1]
            delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[step] = last_gae
        return advantages, advantages + values

    def _cost_predictions(self, observations):
        features = torch.as_tensor(observations[..., END2RACE_OBSERVATION_SIZE:], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            predictions = self.cost_value_net(features.reshape(-1, features.shape[-1])).reshape(features.shape[:-1])
        return predictions.detach().cpu().numpy().astype(np.float32)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        self._cost_rows = []
        self._collision_reward_rows = []
        self._done_rows = []
        self._ego_index_rows = []
        self._scenario_id_rows = []
        self._capture_cost_info = True
        try:
            completed = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        finally:
            self._capture_cost_info = False
        if not completed:
            return False
        costs = np.asarray(self._cost_rows, dtype=np.float32)
        collision_rewards = np.asarray(self._collision_reward_rows, dtype=np.float32)
        dones_by_step = np.asarray(self._done_rows, dtype=bool)
        ego_indices = np.asarray(self._ego_index_rows, dtype=np.int64)
        scenario_ids = np.asarray(self._scenario_id_rows, dtype=str)
        expected_shape = (self.n_steps, self.n_envs)
        if any(array.shape != expected_shape for array in (costs, collision_rewards, dones_by_step, ego_indices, scenario_ids)):
            raise RuntimeError("Constraint transition capture lost the rollout step/env layout")
        if not np.array_equal(collision_rewards, costs * COLLISION_REWARD):
            raise RuntimeError("Collision reward and unique cost event are not identical")
        original_rewards = self.rollout_buffer.rewards.copy()
        self.rollout_buffer.rewards = original_rewards - collision_rewards
        reward_correction_error = float(np.max(np.abs(self.rollout_buffer.rewards - (original_rewards + 2.0 * costs))))
        cost_values = self._cost_predictions(self.rollout_buffer.observations)
        last_cost_values = self._cost_predictions(np.asarray(self._last_obs, dtype=np.float32))
        cost_advantages, cost_returns = self._gae(costs, cost_values, self.rollout_buffer.episode_starts, last_cost_values, self._last_episode_starts, COST_GAMMA, COST_GAE_LAMBDA)
        self.rollout_buffer.costs = costs
        self.rollout_buffer.cost_values = cost_values
        self.rollout_buffer.cost_advantages = cost_advantages
        self.rollout_buffer.cost_returns = cost_returns
        with torch.no_grad():
            starts = torch.as_tensor(self._last_episode_starts, dtype=torch.float32, device=self.device)
            last_reward_values = self.policy.predict_values(self.policy.obs_to_tensor(self._last_obs)[0], self._last_lstm_states.vf, starts).detach().flatten()
        self.rollout_buffer.compute_returns_and_advantage(last_values=last_reward_values, dones=self._last_episode_starts)
        completed_collisions = sum(record["episode_outcome"] == "ego_collision" for record in self._rollout_episode_records)
        completed_episodes = len(self._rollout_episode_records)
        self.last_constraint_rollout = {
            "costs": costs,
            "collision_rewards": collision_rewards,
            "dones_by_step": dones_by_step,
            "ego_indices": ego_indices,
            "scenario_ids": scenario_ids,
            "original_rewards": original_rewards,
            "adjusted_rewards": self.rollout_buffer.rewards.copy(),
            "reward_correction_max_abs_error": reward_correction_error,
            "cost_event_count": int(costs.sum()),
            "completed_collision_episode_count": int(completed_collisions),
            "completed_episode_count": int(completed_episodes),
            "completed_episode_collision_rate": float(completed_collisions / completed_episodes),
            "cost_advantage_std": float(cost_advantages.std()),
            "cost_return_std": float(cost_returns.std()),
        }
        return True

    def _cost_batch_loss(self, observations, returns, mask):
        features = observations[..., END2RACE_OBSERVATION_SIZE:]
        values = self.cost_value_net(features).flatten()
        return torch.nn.functional.mse_loss(values[mask], returns[mask])

    def _warmup_cost_critic(self):
        train_sequences, validation_sequences = self._warmup_split()
        inputs = self.rollout_buffer.observations.reshape(-1, self.rollout_buffer.obs_shape[0])
        returns = self.rollout_buffer.cost_returns.reshape(-1)
        train_indices = self._flat_sequence_indices(train_sequences)
        validation_indices = self._flat_sequence_indices(validation_sequences)
        best_loss = float("inf")
        best_state = None
        best_optimizer = None
        best_epoch = 0
        stale_epochs = 0
        grad_norms = []
        train_losses = []
        validation_losses = []
        for epoch in range(WARMUP_MAX_EPOCHS):
            shuffled = self.cost_minibatch_rng.permutation(train_indices)
            samples = []
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                observations = torch.as_tensor(inputs[batch], dtype=torch.float32, device=self.device)
                targets = torch.as_tensor(returns[batch], dtype=torch.float32, device=self.device)
                mask = torch.ones(len(batch), dtype=torch.bool, device=self.device)
                loss = self._cost_batch_loss(observations, targets, mask)
                self.cost_optimizer.zero_grad()
                (VALUE_LOSS_COEFFICIENT * loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.cost_parameters, MAX_GRAD_NORM)
                require_finite_tensor("Cost warm-up gradient norm", grad_norm)
                self.cost_optimizer.step()
                grad_norms.append(float(grad_norm.detach().cpu().item()))
                samples.append((float(loss.item()), len(batch)))
            with torch.no_grad():
                validation_samples = []
                for start in range(0, len(validation_indices), self.batch_size):
                    batch = validation_indices[start : start + self.batch_size]
                    observations = torch.as_tensor(inputs[batch], dtype=torch.float32, device=self.device)
                    targets = torch.as_tensor(returns[batch], dtype=torch.float32, device=self.device)
                    mask = torch.ones(len(batch), dtype=torch.bool, device=self.device)
                    loss = self._cost_batch_loss(observations, targets, mask)
                    validation_samples.append((float(loss.item()), len(batch)))
            train_loss = sum(loss * count for loss, count in samples) / sum(count for _loss, count in samples)
            validation_loss = sum(loss * count for loss, count in validation_samples) / sum(count for _loss, count in validation_samples)
            require_finite_number("Cost warm-up train loss", train_loss)
            require_finite_number("Cost warm-up validation loss", validation_loss)
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            print(f"Cost warm-up epoch {epoch + 1}: train_loss={train_loss:.6f}, validation_loss={validation_loss:.6f}", flush=True)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.cost_value_net.state_dict())
                best_optimizer = copy.deepcopy(self.cost_optimizer.state_dict())
                best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= WARMUP_PATIENCE:
                    break
        if best_state is None or best_optimizer is None:
            raise RuntimeError("Cost critic warm-up did not produce a checkpoint")
        self.cost_value_net.load_state_dict(best_state)
        self.cost_optimizer.load_state_dict(best_optimizer)
        path = self.recorder.checkpoints_dir / "cost_critic_warmup.pt"
        torch.save(TrainingRecorder._cpu_state_dict(best_state), path)
        rate = self.last_constraint_rollout["completed_episode_collision_rate"]
        lambda_before = self.lambda_cost
        self.lambda_cost = float(np.clip(self.lambda_cost + DUAL_LEARNING_RATE * (rate - COST_BUDGET), 0.0, MAXIMUM_DUAL))
        metrics = {
            "phase": "constrained_warmup",
            "epochs": epoch + 1,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "train_losses": train_losses,
            "validation_losses": validation_losses,
            "cost_critic_grad_norm_mean": float(np.mean(grad_norms)),
            "cost_critic_grad_norm_max": float(np.max(grad_norms)),
            "cost_event_count": self.last_constraint_rollout["cost_event_count"],
            "completed_episode_count": self.last_constraint_rollout["completed_episode_count"],
            "completed_episode_collision_rate": rate,
            "cost_budget": COST_BUDGET,
            "lambda_before": lambda_before,
            "lambda_after": self.lambda_cost,
            "cost_critic_checkpoint": str(path),
        }
        self.recorder.record_metrics(metrics)
        return metrics

    def _cost_statistics(self):
        predictions = []
        returns = []
        with torch.no_grad():
            for data in self.rollout_buffer.get(self.batch_size, rng=self.cost_telemetry_rng):
                mask = data.mask > 1e-8
                values = self.cost_value_net(data.observations[..., END2RACE_OBSERVATION_SIZE:]).flatten()
                predictions.append(values[mask].detach().cpu().numpy())
                returns.append(self.rollout_buffer.current_cost_returns[mask].detach().cpu().numpy())
        prediction = np.concatenate(predictions).astype(np.float64)
        target = np.concatenate(returns).astype(np.float64)
        return {
            "cost_value_loss": float(np.mean(np.square(prediction - target))),
            "cost_explained_variance": float(explained_variance(prediction, target)),
            "cost_prediction_mean": float(prediction.mean()),
            "cost_prediction_std": float(prediction.std()),
            "cost_return_mean": float(target.mean()),
            "cost_return_std": float(target.std()),
        }

    def _formal_cost_critic(self):
        statistics_pre = self._cost_statistics()
        grad_norms = []
        epoch_losses = []
        for _epoch in range(COST_CRITIC_EPOCHS):
            samples = []
            for data in self.rollout_buffer.get(self.batch_size, rng=self.cost_minibatch_rng):
                mask = data.mask > 1e-8
                loss = self._cost_batch_loss(data.observations, self.rollout_buffer.current_cost_returns, mask)
                self.cost_optimizer.zero_grad()
                (VALUE_LOSS_COEFFICIENT * loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.cost_parameters, MAX_GRAD_NORM)
                require_finite_tensor("Cost value gradient norm", grad_norm)
                self.cost_optimizer.step()
                grad_norms.append(float(grad_norm.detach().cpu().item()))
                samples.append((float(loss.item()), int(mask.sum().item())))
            epoch_losses.append(sum(loss * count for loss, count in samples) / sum(count for _loss, count in samples))
        statistics_post = self._cost_statistics()
        return statistics_pre, statistics_post, grad_norms, epoch_losses

    def train(self):
        if not self.warmup_completed:
            super().train()
            self._warmup_cost_critic()
            return
        reward_advantages = self.rollout_buffer.advantages.copy()
        cost_advantages = self.rollout_buffer.cost_advantages.copy()
        lambda_used = self.lambda_cost
        combined = reward_advantages - lambda_used * cost_advantages
        if not np.isfinite(combined).all() or float(combined.std()) <= 0.0:
            raise RuntimeError("Combined reward-cost advantage is invalid")
        self.rollout_buffer.advantages = combined.astype(np.float32)
        super().train()
        statistics_pre, statistics_post, grad_norms, epoch_losses = self._formal_cost_critic()
        update = self._n_updates
        rate = self.last_constraint_rollout["completed_episode_collision_rate"]
        self.lambda_cost = float(np.clip(self.lambda_cost + DUAL_LEARNING_RATE * (rate - COST_BUDGET), 0.0, MAXIMUM_DUAL))
        cost_path = self.recorder.checkpoints_dir / f"cost_critic_u{update:04d}.pt"
        torch.save(TrainingRecorder._cpu_state_dict(self.cost_value_net.state_dict()), cost_path)
        metrics = {
            "phase": "constrained_formal",
            "update": update,
            "reward_collision_removed_from_training_return": True,
            "reward_correction_max_abs_error": self.last_constraint_rollout["reward_correction_max_abs_error"],
            "cost_event_count": self.last_constraint_rollout["cost_event_count"],
            "completed_collision_episode_count": self.last_constraint_rollout["completed_collision_episode_count"],
            "completed_episode_count": self.last_constraint_rollout["completed_episode_count"],
            "completed_episode_collision_rate": rate,
            "cost_budget": COST_BUDGET,
            "lambda_used": lambda_used,
            "lambda_after": self.lambda_cost,
            "reward_advantage_mean": float(reward_advantages.mean()),
            "reward_advantage_std": float(reward_advantages.std()),
            "cost_advantage_mean": float(cost_advantages.mean()),
            "cost_advantage_std": float(cost_advantages.std()),
            "combined_advantage_mean": float(combined.mean()),
            "combined_advantage_std": float(combined.std()),
            "cost_critic_epoch_losses": epoch_losses,
            "cost_critic_grad_norm_mean": float(np.mean(grad_norms)),
            "cost_critic_grad_norm_max": float(np.max(grad_norms)),
            "cost_critic_checkpoint": str(cost_path),
            **{f"{name}_pre": value for name, value in statistics_pre.items()},
            **{f"{name}_post": value for name, value in statistics_post.items()},
        }
        self.recorder.record_metrics(metrics)
        print(f"Constraint update {update}: collision_rate={rate:.6f}, lambda={lambda_used:.6f}->{self.lambda_cost:.6f}, cost_ev={statistics_post['cost_explained_variance']:.6f}", flush=True)


def build_model(vector_env, recorder, args):
    return ConstrainedEnd2RacePPO(
        End2RaceGRUPolicy,
        vector_env,
        actor_epochs=2,
        critic_epochs=5,
        recorder=recorder,
        learning_rate=1.0,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.999,
        gae_lambda=0.995,
        clip_range=0.20,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=VALUE_LOSS_COEFFICIENT,
        max_grad_norm=MAX_GRAD_NORM,
        seed=args.seed,
        device=torch.device("cuda"),
        policy_kwargs={
            "checkpoint_path": args.actor_path,
            "hidden_scale": 4,
            "critic_variant": "privilege_gru",
            "gru_learning_rate": 3.0e-6,
            "head_learning_rate": 3.0e-5,
            "critic_learning_rate": 3.0e-4,
            "steering_latent_std": 0.03,
            "speed_physical_std": 0.15,
            "speed_exploration_mode": BASELINE_EXPLORATION_MODE,
        },
        verbose=1,
    )


def load_scenarios(args):
    rows = json.loads((Path(args.collision_cache_dir) / "collision_scenarios.json").read_text(encoding="utf-8"))
    collisions = tuple(ScenarioSpec(**row) for row in rows)
    ordinary = ordinary_scenarios("Austin")
    if len(collisions) != 479 or len(ordinary) != 600:
        raise RuntimeError("Frozen Austin collision or ordinary pool changed")
    return collisions, ordinary


def build_environment_and_model(args, recorder):
    collisions, ordinary = load_scenarios(args)
    vector_env = CentralScheduleSubprocVecEnv(args.n_envs, START_METHOD, args.seed, "Austin", collisions, ordinary, privileged=True, reward_gamma=0.999, speed_exploration_mode=BASELINE_EXPLORATION_MODE)
    model = build_model(vector_env, recorder, args)
    recorder.write_scenario_pools(collisions, ordinary, {"mode": "frozen canonical BC Austin collision pool", "collision_count": len(collisions)})
    recorder.write_run_config(args, {}, {
        "algorithm": "Lagrangian Constrained PPO",
        "cost_gamma": COST_GAMMA,
        "cost_gae_lambda": COST_GAE_LAMBDA,
        "cost_budget": COST_BUDGET,
        "initial_dual": INITIAL_DUAL,
        "dual_learning_rate": DUAL_LEARNING_RATE,
        "maximum_dual": MAXIMUM_DUAL,
        "cost_critic": "P20 MLP 20-120-30-1",
        "cost_critic_learning_rate": COST_CRITIC_LEARNING_RATE,
        "cost_critic_epochs": COST_CRITIC_EPOCHS,
        "collision_reward_removed_from_training_return": True,
    })
    return vector_env, model


def rank_auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    position = 0
    while position < len(scores):
        end = position + 1
        while end < len(scores) and scores[order[end]] == scores[order[position]]:
            end += 1
        ranks[order[position:end]] = 0.5 * (position + 1 + end)
        position = end
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def completed_episode_targets(model):
    rollout = model.last_constraint_rollout
    starts = np.asarray(model.rollout_buffer.episode_starts, dtype=bool)
    dones = rollout["dones_by_step"]
    costs = rollout["costs"]
    ego_indices = rollout["ego_indices"]
    observations = np.asarray(model.rollout_buffer.observations, dtype=np.float32)
    target = np.zeros_like(costs, dtype=np.float32)
    valid = np.zeros_like(dones, dtype=bool)
    episode_start_mask = np.zeros_like(dones, dtype=bool)
    early_mask = np.zeros_like(dones, dtype=bool)
    collision_label = np.zeros_like(dones, dtype=bool)
    episodes = []
    for env_index in range(model.n_envs):
        start = 0
        while start < model.n_steps:
            terminal_positions = np.flatnonzero(dones[start:, env_index])
            if len(terminal_positions) == 0:
                break
            end = start + int(terminal_positions[0])
            if not starts[start, env_index]:
                start = end + 1
                continue
            collision = bool(costs[end, env_index] > 0.5)
            length = end - start + 1
            rows = np.arange(start, end + 1)
            valid[rows, env_index] = True
            episode_start_mask[start, env_index] = True
            collision_label[rows, env_index] = collision
            if collision:
                target[rows, env_index] = np.power(COST_GAMMA, end - rows).astype(np.float32)
            early_rows = rows[(end - rows) >= OOF_EARLY_REMAINING_STEPS]
            early_mask[early_rows, env_index] = True
            episodes.append({"env_index": env_index, "start": start, "end": end, "ego_idx": int(ego_indices[start, env_index]), "collision": collision, "length": length})
            start = end + 1
    if not np.array_equal(starts[valid & episode_start_mask], np.ones(int(episode_start_mask.sum()), dtype=bool)):
        raise RuntimeError("Completed episode extraction lost boundaries")
    return observations, target, valid, episode_start_mask, early_mask, collision_label, episodes


def startpoint_folds(episodes):
    ego_indices = sorted({episode["ego_idx"] for episode in episodes}, key=lambda value: hashlib.sha256(f"z9-startpoint|{value}".encode("utf-8")).hexdigest())
    assignment = {ego_idx: index % OOF_FOLDS for index, ego_idx in enumerate(ego_indices)}
    return assignment


def oof_cost_critic(model):
    observations, targets, valid, episode_start_mask, early_mask, collision_labels, episodes = completed_episode_targets(model)
    assignment = startpoint_folds(episodes)
    features = observations[..., END2RACE_OBSERVATION_SIZE:]
    ego_indices = model.last_constraint_rollout["ego_indices"]
    fold_ids = np.full(ego_indices.shape, -1, dtype=np.int64)
    for ego_idx, fold in assignment.items():
        fold_ids[ego_indices == ego_idx] = fold
    if np.any(valid & (fold_ids < 0)):
        raise RuntimeError("A complete episode startpoint is missing from the OOF assignment")
    predictions = np.full(targets.shape, np.nan, dtype=np.float32)
    baselines = np.full(targets.shape, np.nan, dtype=np.float32)
    fold_rows = []
    devices = [model.device.index if model.device.index is not None else torch.cuda.current_device()]
    for fold in range(OOF_FOLDS):
        test = valid & (fold_ids == fold)
        train = valid & ~test
        train_indices = np.flatnonzero(train.reshape(-1))
        test_indices = np.flatnonzero(test.reshape(-1))
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise RuntimeError("Startpoint OOF fold is empty")
        seed = int(np.random.SeedSequence([model.seed, 0x4F4F46, fold]).generate_state(1)[0])
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            critic = PriviledgeMLPCritic().to(model.device)
        optimizer = torch.optim.Adam(critic.parameters(), lr=COST_CRITIC_LEARNING_RATE)
        rng = np.random.default_rng(np.random.SeedSequence([model.seed, 0x4F4F53, fold]))
        flat_features = features.reshape(-1, features.shape[-1])
        flat_targets = targets.reshape(-1)
        for _epoch in range(OOF_EPOCHS):
            order = rng.permutation(train_indices)
            for start in range(0, len(order), model.batch_size):
                batch = order[start : start + model.batch_size]
                batch_features = torch.as_tensor(flat_features[batch], dtype=torch.float32, device=model.device)
                batch_targets = torch.as_tensor(flat_targets[batch], dtype=torch.float32, device=model.device)
                loss = torch.nn.functional.mse_loss(critic(batch_features).flatten(), batch_targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                optimizer.step()
        with torch.no_grad():
            for start in range(0, len(test_indices), model.batch_size):
                batch = test_indices[start : start + model.batch_size]
                values = critic(torch.as_tensor(flat_features[batch], dtype=torch.float32, device=model.device)).flatten().detach().cpu().numpy()
                predictions.reshape(-1)[batch] = values.astype(np.float32)
        train_mean = float(flat_targets[train_indices].mean())
        baselines.reshape(-1)[test_indices] = train_mean
        fold_episodes = [episode for episode in episodes if assignment[episode["ego_idx"]] == fold]
        fold_rows.append({
            "fold": fold,
            "train_transition_count": int(len(train_indices)),
            "test_transition_count": int(len(test_indices)),
            "test_startpoint_count": len({episode["ego_idx"] for episode in fold_episodes}),
            "test_episode_count": len(fold_episodes),
            "test_collision_episode_count": sum(episode["collision"] for episode in fold_episodes),
            "train_target_mean": train_mean,
        })
    if not np.isfinite(predictions[valid]).all() or not np.isfinite(baselines[valid]).all():
        raise RuntimeError("OOF predictions are incomplete")
    mse = float(np.mean(np.square(predictions[valid].astype(np.float64) - targets[valid].astype(np.float64))))
    baseline_mse = float(np.mean(np.square(baselines[valid].astype(np.float64) - targets[valid].astype(np.float64))))
    mse_skill = float(1.0 - mse / baseline_mse)
    start_labels = collision_labels[episode_start_mask]
    start_scores = predictions[episode_start_mask]
    early_labels = collision_labels[early_mask]
    early_scores = predictions[early_mask]
    return {
        "complete_episode_count": len(episodes),
        "complete_collision_episode_count": sum(episode["collision"] for episode in episodes),
        "complete_transition_count": int(valid.sum()),
        "unique_startpoint_count": len(assignment),
        "folds": fold_rows,
        "oof_mse": mse,
        "train_mean_baseline_mse": baseline_mse,
        "oof_mse_skill": mse_skill,
        "episode_start_positive_count": int(start_labels.sum()),
        "episode_start_negative_count": int((~start_labels).sum()),
        "episode_start_auroc": rank_auroc(start_labels, start_scores),
        "early_transition_count": int(early_mask.sum()),
        "early_positive_count": int(early_labels.sum()),
        "early_negative_count": int((~early_labels).sum()),
        "early_auroc": rank_auroc(early_labels, early_scores),
    }


def normalized_surrogate(model, advantages, data, mask, clip_range):
    valid = advantages[mask]
    normalized = (advantages - valid.mean()) / (valid.std() + 1e-8)
    log_prob, _entropy = model.policy.evaluate_actor_actions(data.observations, data.actions, data.lstm_states, data.episode_starts)
    ratio = torch.exp(log_prob - data.old_log_prob)
    return -torch.min(normalized * ratio, normalized * torch.clamp(ratio, 1 - clip_range, 1 + clip_range))[mask].mean()


def accumulated_actor_gradient(model, combined, seed):
    gradients = [torch.zeros_like(parameter) for parameter in model.policy.actor_parameters]
    losses = []
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x47524144]))
    for data in model.rollout_buffer.get(model.batch_size, rng=rng):
        mask = data.mask > 1e-8
        reward_advantages = data.advantages
        advantages = reward_advantages - model.lambda_cost * model.rollout_buffer.current_cost_advantages if combined else reward_advantages
        loss = normalized_surrogate(model, advantages, data, mask, 0.20)
        batch_gradients = torch.autograd.grad(loss, model.policy.actor_parameters)
        for total, value in zip(gradients, batch_gradients):
            total.add_(value.detach())
        losses.append(float(loss.detach().cpu().item()))
    return gradients, losses


def gradient_comparison(left, right):
    left_flat = torch.cat([tensor.flatten().double().cpu() for tensor in left])
    right_flat = torch.cat([tensor.flatten().double().cpu() for tensor in right])
    difference = right_flat - left_flat
    left_norm = float(torch.linalg.vector_norm(left_flat).item())
    right_norm = float(torch.linalg.vector_norm(right_flat).item())
    difference_norm = float(torch.linalg.vector_norm(difference).item())
    cosine = float(torch.dot(left_flat, right_flat).item() / max(left_norm * right_norm, 1e-30))
    return {"reward_gradient_l2": left_norm, "combined_gradient_l2": right_norm, "difference_l2": difference_norm, "difference_relative_to_reward": difference_norm / max(left_norm, 1e-30), "cosine": cosine}


def run_preflight(args, plan):
    gate_directory = Path(args.gate_dir).expanduser().resolve()
    report_path = gate_directory / "preflight_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(json.dumps({"verdict": report["verdict"], "criteria": report["criteria"]}, indent=2))
        return report
    rollout_directory = gate_directory / "rollout"
    if rollout_directory.exists() and any(rollout_directory.iterdir()):
        raise RuntimeError("Refusing to overwrite partial Z9 preflight rollout")
    recorder = TrainingRecorder(rollout_directory, 4)
    configure_training_numerics()
    vector_env, model = build_environment_and_model(args, recorder)
    try:
        actor_before = state_digest(model.policy.actor_checkpoint_state_dict())
        cost_before = state_digest(model.cost_value_net.state_dict())
        _total, callback = model._setup_learn(args.n_envs * args.n_steps, progress_bar=False)
        callback.on_training_start(locals(), globals())
        completed = model.collect_rollouts(vector_env, callback, model.rollout_buffer, args.n_steps)
        if not completed or not model.rollout_buffer.full:
            raise RuntimeError("Z9 preflight rollout did not complete")
        oof = oof_cost_critic(model)
        model.train()
        callback.on_training_end()
        cost_after = state_digest(model.cost_value_net.state_dict())
        reward_gradients, reward_losses = accumulated_actor_gradient(model, False, args.seed)
        combined_gradients, combined_losses = accumulated_actor_gradient(model, True, args.seed)
        gradients = gradient_comparison(reward_gradients, combined_gradients)
        actor_after = state_digest(model.policy.actor_checkpoint_state_dict())
        rollout = model.last_constraint_rollout
        criteria = {
            "full_102400_transition_rollout": int(rollout["costs"].size) == plan["preflight_contract"]["transition_count"],
            "reward_collision_is_uniquely_removed": rollout["reward_correction_max_abs_error"] == 0.0 and np.array_equal(rollout["collision_rewards"], rollout["costs"] * COLLISION_REWARD),
            "cost_event_matches_completed_collision_episode": rollout["cost_event_count"] == rollout["completed_collision_episode_count"] and rollout["cost_event_count"] > 0,
            "cost_advantage_is_finite_and_nondegenerate": np.isfinite(model.rollout_buffer.cost_advantages).all() and rollout["cost_advantage_std"] >= 1.0e-3,
            "cost_critic_warmup_changed_parameters": cost_before != cost_after,
            "actor_unchanged_by_preflight": actor_before == actor_after,
            "combined_gradient_is_finite_and_distinct": all(math.isfinite(value) for value in gradients.values()) and gradients["difference_relative_to_reward"] >= 1.0e-4,
            "startpoint_grouped_oof_has_coverage": oof["unique_startpoint_count"] >= 20 and oof["complete_collision_episode_count"] >= 10 and all(row["test_collision_episode_count"] > 0 for row in oof["folds"]),
            "startpoint_grouped_oof_mse_skill_passed": oof["oof_mse_skill"] >= OOF_MINIMUM_MSE_SKILL,
            "startpoint_grouped_episode_start_auroc_passed": oof["episode_start_auroc"] is not None and oof["episode_start_auroc"] >= OOF_MINIMUM_EPISODE_START_AUROC,
            "startpoint_grouped_early_auroc_passed": oof["early_auroc"] is not None and oof["early_auroc"] >= OOF_MINIMUM_EARLY_AUROC,
            "dual_moves_up_when_rate_exceeds_budget": rollout["completed_episode_collision_rate"] > COST_BUDGET and model.lambda_cost > INITIAL_DUAL,
        }
        verdict = "pass_to_one_formal_run" if all(criteria.values()) else "fail_stop_exact_constrained_implementation"
        report = {
            "schema_version": 1,
            "experiment_id": "constrained_ppo_collision_cost_v1",
            "gate": "Z9-A",
            "verdict": verdict,
            "criteria": criteria,
            "rollout": {name: value for name, value in rollout.items() if not isinstance(value, np.ndarray)},
            "oof_cost_critic": oof,
            "actor_gradient": gradients,
            "reward_only_minibatch_losses": reward_losses,
            "combined_minibatch_losses": combined_losses,
            "cost_critic_state_changed": cost_before != cost_after,
            "actor_state_unchanged": actor_before == actor_after,
            "lambda_after_warmup": model.lambda_cost,
            "evidence_boundary": plan["evidence_boundary"],
        }
        atomic_write_json(report_path, report)
        print(json.dumps({"verdict": verdict, "criteria": criteria, "oof": oof, "gradient": gradients}, indent=2))
        return report
    finally:
        vector_env.close()


def run_formal(args, plan):
    gate_directory = Path(args.gate_dir).expanduser().resolve()
    report = json.loads((gate_directory / "preflight_report.json").read_text(encoding="utf-8"))
    if report["verdict"] != "pass_to_one_formal_run" or not all(report["criteria"].values()):
        raise RuntimeError("Z9 formal run is blocked by the preflight verdict")
    output_directory = Path(args.output_dir).expanduser().resolve()
    recorder = TrainingRecorder(output_directory, 4)
    configure_training_numerics()
    vector_env, model = build_environment_and_model(args, recorder)
    try:
        total_rollouts = args.num_updates + 1
        model.learn(total_timesteps=args.n_envs * args.n_steps * total_rollouts, log_interval=1, progress_bar=False)
    finally:
        vector_env.close()
    final_actor = output_directory / "checkpoints" / f"actor_u{args.num_updates:04d}.pth"
    final_cost = output_directory / "checkpoints" / f"cost_critic_u{args.num_updates:04d}.pt"
    if not final_actor.is_file() or not final_cost.is_file():
        raise RuntimeError("Z9 formal training finished without final checkpoints")
    summary = {
        "schema_version": 1,
        "experiment_id": "constrained_ppo_collision_cost_v1",
        "status": "training_complete_evaluation_pending",
        "formal_updates": args.num_updates,
        "final_actor": str(final_actor),
        "final_actor_sha256": sha256_file(final_actor),
        "final_cost_critic": str(final_cost),
        "final_cost_critic_sha256": sha256_file(final_cost),
        "evaluation_band": plan["formal_acceptance"]["fixed_checkpoint_band"],
    }
    atomic_write_json(output_directory / "training_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    args = parse_arguments()
    plan = write_gate_plan(args)
    if args.mode == "prepare":
        print(json.dumps(plan, indent=2))
    elif args.mode == "preflight":
        run_preflight(args, plan)
    else:
        run_formal(args, plan)
