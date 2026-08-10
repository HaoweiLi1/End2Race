import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
import gymnasium
import numpy as np
import stable_baselines3
import torch
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import End2Race
from ppo.policy import BASELINE_EXPLORATION_MODE
from ppo.policy import END2RACE_ACTION_SIZE
from ppo.policy import END2RACE_OBSERVATION_SIZE
from ppo.policy import JOINT_TEMPORAL_BLOCK_STEPS
from ppo.policy import JOINT_TEMPORAL_PREFIX_STEPS
from ppo.policy import JOINT_TEMPORAL_RHO
from ppo.policy import NOOP_SPEED_BOUND
from ppo.policy import PREFIX_JOINT_TEMPORAL_EXPLORATION_MODE
from ppo.policy import SPEED_PHYSICAL_STD
from ppo.policy import STEERING_BOUND
from ppo.policy import STEERING_LATENT_STD
from ppo.policy import End2RaceGRUPolicy
from ppo.policy import joint_temporal_conditional_log_prob
from ppo.policy import joint_temporal_sequence_log_prob
from ppo.policy import joint_temporal_standardized_residuals
from ppo.rollout import End2RaceRolloutBuffer


DEFAULT_OUTPUT = PROJECT_ROOT / "post-trained" / "ppo_prefix_reset_joint_temporal_rho0p90_gates" / "e0" / "e0_report.json"
FIXED_SEED = 20260809
SYNTHETIC_BLOCKS = 100000


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the prefix joint-temporal exploration E0 gate")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def make_physical_actions(latent_means, speed_means, residuals):
    steering = STEERING_BOUND * torch.tanh(latent_means + STEERING_LATENT_STD * residuals[:, 0])
    speed = speed_means + SPEED_PHYSICAL_STD * residuals[:, 1]
    return torch.stack((steering, speed), dim=1)


def direct_mvn_log_prob(mean_actions, actions, rho):
    length = mean_actions.shape[0]
    residuals, steering_jacobian = joint_temporal_standardized_residuals(mean_actions, actions)
    covariance = torch.full((length, length), rho, dtype=mean_actions.dtype, device=mean_actions.device)
    covariance.diagonal().fill_(1.0)
    distribution = torch.distributions.MultivariateNormal(torch.zeros(length, dtype=mean_actions.dtype, device=mean_actions.device), covariance_matrix=covariance)
    standardized = distribution.log_prob(residuals[:, 0]) + distribution.log_prob(residuals[:, 1])
    scales = -length * (math.log(STEERING_LATENT_STD) + math.log(SPEED_PHYSICAL_STD))
    return standardized + scales - steering_jacobian.sum()


def reference_case(length, rng):
    common = torch.tensor(rng.normal(size=(1, 2)), dtype=torch.float64)
    innovation = torch.tensor(rng.normal(size=(length, 2)), dtype=torch.float64)
    residuals = math.sqrt(JOINT_TEMPORAL_RHO) * common + math.sqrt(1.0 - JOINT_TEMPORAL_RHO) * innovation
    behavior_latent = torch.tensor(rng.normal(0.0, 0.12, size=length), dtype=torch.float64)
    behavior_speed = torch.tensor(rng.uniform(2.5, 6.0, size=length), dtype=torch.float64)
    actions = make_physical_actions(behavior_latent, behavior_speed, residuals).detach()
    candidate_latent = behavior_latent + torch.tensor(rng.normal(0.0, 0.015, size=length), dtype=torch.float64)
    candidate_speed = behavior_speed + torch.tensor(rng.normal(0.0, 0.05, size=length), dtype=torch.float64)
    candidate_means = torch.stack((STEERING_BOUND * torch.tanh(candidate_latent), candidate_speed), dim=1)
    return candidate_means, actions


def covariance_gate(rng):
    common = rng.normal(size=(SYNTHETIC_BLOCKS, 1, END2RACE_ACTION_SIZE)).astype(np.float32)
    innovation = rng.normal(size=(SYNTHETIC_BLOCKS, JOINT_TEMPORAL_BLOCK_STEPS, END2RACE_ACTION_SIZE)).astype(np.float32)
    residuals = math.sqrt(JOINT_TEMPORAL_RHO) * common + math.sqrt(1.0 - JOINT_TEMPORAL_RHO) * innovation
    means = residuals.mean(axis=0, dtype=np.float64)
    variances = residuals.var(axis=0, ddof=1, dtype=np.float64)
    dimension_means = residuals.mean(axis=(0, 1), dtype=np.float64)
    dimension_variances = residuals.var(axis=(0, 1), ddof=1, dtype=np.float64)
    centered_steering = residuals[:, :, 0].astype(np.float64) - means[:, 0]
    centered_speed = residuals[:, :, 1].astype(np.float64) - means[:, 1]
    covariance_steering = centered_steering.T @ centered_steering / (SYNTHETIC_BLOCKS - 1)
    covariance_speed = centered_speed.T @ centered_speed / (SYNTHETIC_BLOCKS - 1)
    standard_steering = np.sqrt(np.diag(covariance_steering))
    standard_speed = np.sqrt(np.diag(covariance_speed))
    correlation_steering = covariance_steering / np.outer(standard_steering, standard_steering)
    correlation_speed = covariance_speed / np.outer(standard_speed, standard_speed)
    cross_correlation = (centered_steering.T @ centered_speed / (SYNTHETIC_BLOCKS - 1)) / np.outer(standard_steering, standard_speed)
    off_diagonal = ~np.eye(JOINT_TEMPORAL_BLOCK_STEPS, dtype=bool)
    metrics = {
        "blocks": SYNTHETIC_BLOCKS,
        "dimension_means": dimension_means.tolist(),
        "dimension_variances": dimension_variances.tolist(),
        "maximum_absolute_dimension_mean": float(np.abs(dimension_means).max()),
        "minimum_dimension_variance": float(dimension_variances.min()),
        "maximum_dimension_variance": float(dimension_variances.max()),
        "diagnostic_maximum_absolute_temporal_coordinate_mean": float(np.abs(means).max()),
        "diagnostic_minimum_temporal_coordinate_variance": float(variances.min()),
        "diagnostic_maximum_temporal_coordinate_variance": float(variances.max()),
        "criterion_scope": "Each action dimension aggregated across synthetic blocks and temporal coordinates; temporal dependence is tested separately by the 50x50 correlations.",
        "steering_correlation_diagonal_maximum_error": float(np.abs(np.diag(correlation_steering) - 1.0).max()),
        "speed_correlation_diagonal_maximum_error": float(np.abs(np.diag(correlation_speed) - 1.0).max()),
        "steering_correlation_off_diagonal_maximum_error": float(np.abs(correlation_steering[off_diagonal] - JOINT_TEMPORAL_RHO).max()),
        "speed_correlation_off_diagonal_maximum_error": float(np.abs(correlation_speed[off_diagonal] - JOINT_TEMPORAL_RHO).max()),
        "steering_speed_cross_correlation_rms": float(np.sqrt(np.mean(np.square(cross_correlation)))),
        "steering_speed_cross_correlation_maximum_absolute": float(np.abs(cross_correlation).max()),
    }
    passed = (
        metrics["maximum_absolute_dimension_mean"] <= 0.01
        and metrics["minimum_dimension_variance"] >= 0.98
        and metrics["maximum_dimension_variance"] <= 1.02
        and metrics["steering_correlation_diagonal_maximum_error"] <= 0.01
        and metrics["speed_correlation_diagonal_maximum_error"] <= 0.01
        and metrics["steering_correlation_off_diagonal_maximum_error"] <= 0.01
        and metrics["speed_correlation_off_diagonal_maximum_error"] <= 0.01
        and metrics["steering_speed_cross_correlation_rms"] <= 0.005
        and metrics["steering_speed_cross_correlation_maximum_absolute"] <= 0.016
    )
    del common, innovation, residuals, centered_steering, centered_speed
    return {"passed": passed, "metrics": metrics}


def likelihood_reference_gate(rng):
    maximum_float64_log_prob_error = 0.0
    maximum_float32_per_step_error = 0.0
    maximum_gradient_error = 0.0
    lengths = list(range(1, JOINT_TEMPORAL_BLOCK_STEPS + 1))
    rng.shuffle(lengths)
    for length in lengths:
        candidate_values, actions = reference_case(length, rng)
        conditional_values = candidate_values.clone().requires_grad_(True)
        conditional_log_probs, _residuals = joint_temporal_sequence_log_prob(conditional_values, actions)
        direct_values = candidate_values.clone().requires_grad_(True)
        direct_log_prob = direct_mvn_log_prob(direct_values, actions, JOINT_TEMPORAL_RHO)
        maximum_float64_log_prob_error = max(maximum_float64_log_prob_error, abs(float(conditional_log_probs.sum().detach() - direct_log_prob.detach())))
        conditional_gradient = torch.autograd.grad(conditional_log_probs.sum(), conditional_values)[0]
        direct_gradient = torch.autograd.grad(direct_log_prob, direct_values)[0]
        maximum_gradient_error = max(maximum_gradient_error, float((conditional_gradient - direct_gradient).abs().max()))
        float32_log_probs, _float32_residuals = joint_temporal_sequence_log_prob(candidate_values.float(), actions.float())
        maximum_float32_per_step_error = max(maximum_float32_per_step_error, float((float32_log_probs.double() - conditional_log_probs.detach()).abs().max()))
    metrics = {
        "tested_lengths": sorted(lengths),
        "float64_joint_log_prob_maximum_error": maximum_float64_log_prob_error,
        "float32_per_step_log_prob_maximum_error": maximum_float32_per_step_error,
        "candidate_mean_gradient_maximum_error": maximum_gradient_error,
    }
    passed = maximum_float64_log_prob_error <= 1.0e-6 and maximum_float32_per_step_error <= 5.0e-5 and maximum_gradient_error <= 1.0e-5
    return {"passed": passed, "metrics": metrics}


def cut_tail_log_probs(mean_actions, actions, cut):
    residual_sum = torch.zeros(END2RACE_ACTION_SIZE, dtype=mean_actions.dtype, device=mean_actions.device)
    for position in range(cut):
        residual, _steering_jacobian = joint_temporal_standardized_residuals(mean_actions[position], actions[position])
        residual_sum = residual_sum + residual
    log_probs = []
    for position in range(cut, JOINT_TEMPORAL_BLOCK_STEPS):
        log_prob, residual = joint_temporal_conditional_log_prob(mean_actions[position], actions[position], residual_sum, position)
        log_probs.append(log_prob)
        residual_sum = residual_sum + residual
    return torch.stack(log_probs)


def cut_gate(rng):
    candidate_values, actions = reference_case(JOINT_TEMPORAL_BLOCK_STEPS, rng)
    maximum_total_error = 0.0
    maximum_per_step_error = 0.0
    maximum_gradient_error = 0.0
    cut_rows = []
    for cut in range(1, JOINT_TEMPORAL_BLOCK_STEPS):
        full_values = candidate_values.clone().requires_grad_(True)
        full_log_probs, _full_residuals = joint_temporal_sequence_log_prob(full_values, actions)
        reference_tail = full_log_probs[cut:]
        reference_gradient = torch.autograd.grad(reference_tail.sum(), full_values)[0]
        cut_values = candidate_values.clone().requires_grad_(True)
        reconstructed_tail = cut_tail_log_probs(cut_values, actions, cut)
        reconstructed_gradient = torch.autograd.grad(reconstructed_tail.sum(), cut_values)[0]
        total_error = abs(float(reference_tail.sum().detach() - reconstructed_tail.sum().detach()))
        per_step_error = float((reference_tail.detach() - reconstructed_tail.detach()).abs().max())
        gradient_error = float((reference_gradient - reconstructed_gradient).abs().max())
        maximum_total_error = max(maximum_total_error, total_error)
        maximum_per_step_error = max(maximum_per_step_error, per_step_error)
        maximum_gradient_error = max(maximum_gradient_error, gradient_error)
        cut_rows.append({"cut": cut, "tail_steps": JOINT_TEMPORAL_BLOCK_STEPS - cut, "total_log_prob_error": total_error, "per_step_log_prob_maximum_error": per_step_error, "gradient_maximum_error": gradient_error})
    metrics = {
        "cuts_tested": len(cut_rows),
        "maximum_total_log_prob_error": maximum_total_error,
        "maximum_per_step_log_prob_error": maximum_per_step_error,
        "maximum_gradient_error": maximum_gradient_error,
        "cuts": cut_rows,
    }
    passed = len(cut_rows) == 49 and maximum_total_error <= 1.0e-6 and maximum_per_step_error <= 1.0e-6 and maximum_gradient_error <= 1.0e-5
    return {"passed": passed, "metrics": metrics, "candidate_values": candidate_values, "actions": actions}


def flatten_buffer(buffer):
    buffer.hidden_states_pi = buffer.hidden_states_pi.swapaxes(1, 2)
    names = [
        "observations",
        "actions",
        "values",
        "log_probs",
        "advantages",
        "returns",
        "hidden_states_pi",
        "episode_starts",
        "recurrent_resets",
        "exploration_speed_log_stds",
        "exploration_danger_gates",
        "exploration_temporal_active",
        "exploration_block_ids",
        "exploration_standard_residuals",
        "joint_temporal_active",
        "joint_temporal_block_uids",
        "joint_temporal_block_positions",
        "joint_temporal_prefix_steps",
        "joint_temporal_collision_sources",
        "joint_temporal_standard_residuals",
    ]
    for name in names:
        buffer.__dict__[name] = buffer.swap_and_flatten(buffer.__dict__[name])
    buffer.generator_ready = True


def make_buffer(size):
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(END2RACE_OBSERVATION_SIZE,), dtype=np.float32)
    action_space = spaces.Box(low=np.asarray((-STEERING_BOUND, -NOOP_SPEED_BOUND), dtype=np.float32), high=np.asarray((STEERING_BOUND, NOOP_SPEED_BOUND), dtype=np.float32), dtype=np.float32)
    return End2RaceRolloutBuffer(size, observation_space, action_space, (size, 1, 1, 2), "cpu", gamma=0.999, gae_lambda=0.995, n_envs=1)


def buffer_context_gate(candidate_values, actions):
    residuals, _jacobian = joint_temporal_standardized_residuals(candidate_values.float(), actions.float())
    buffer = make_buffer(JOINT_TEMPORAL_BLOCK_STEPS)
    buffer.observations[:, 0] = np.arange(JOINT_TEMPORAL_BLOCK_STEPS, dtype=np.float32)[:, None]
    buffer.actions[:, 0] = actions.float().numpy()
    buffer.joint_temporal_active[:, 0] = True
    buffer.joint_temporal_block_uids[:, 0] = 9000000000001
    buffer.joint_temporal_block_positions[:, 0] = np.arange(JOINT_TEMPORAL_BLOCK_STEPS, dtype=np.int64)
    buffer.joint_temporal_prefix_steps[:, 0] = np.arange(JOINT_TEMPORAL_BLOCK_STEPS, dtype=np.int64)
    buffer.joint_temporal_collision_sources[:, 0] = True
    buffer.joint_temporal_standard_residuals[:, 0] = residuals.numpy()
    flatten_buffer(buffer)
    maximum_position_error = 0
    maximum_action_error = 0.0
    maximum_context_length_error = 0
    for cut in range(1, JOINT_TEMPORAL_BLOCK_STEPS):
        context = buffer._joint_context_for_sequence(cut)
        maximum_context_length_error = max(maximum_context_length_error, abs(int(context["positions"].numel()) - cut))
        maximum_position_error = max(maximum_position_error, int((context["positions"] - torch.arange(cut)).abs().max()))
        maximum_action_error = max(maximum_action_error, float((context["actions"] - actions[:cut].float()).abs().max()))

    first = make_buffer(20)
    first.observations[:, 0] = np.arange(20, dtype=np.float32)[:, None]
    first.actions[:, 0] = actions[:20].float().numpy()
    first.joint_temporal_active[:, 0] = True
    first.joint_temporal_block_uids[:, 0] = 9000000000001
    first.joint_temporal_block_positions[:, 0] = np.arange(20, dtype=np.int64)
    first.joint_temporal_prefix_steps[:, 0] = np.arange(20, dtype=np.int64)
    first.joint_temporal_collision_sources[:, 0] = True
    first.joint_temporal_standard_residuals[:, 0] = residuals[:20].numpy()
    first.finalize_joint_context_carry()
    second = make_buffer(50)
    second.joint_context_next_carry = first.joint_context_next_carry
    second.reset()
    second.observations[:, 0] = np.arange(20, 70, dtype=np.float32)[:, None]
    second.actions[:30, 0] = actions[20:].float().numpy()
    second.actions[30:, 0] = actions[:20].float().numpy()
    second.joint_temporal_active[:, 0] = True
    second.joint_temporal_block_uids[:30, 0] = 9000000000001
    second.joint_temporal_block_uids[30:, 0] = 9000000000002
    second.joint_temporal_block_positions[:30, 0] = np.arange(20, 50, dtype=np.int64)
    second.joint_temporal_block_positions[30:, 0] = np.arange(20, dtype=np.int64)
    second.joint_temporal_prefix_steps[:30, 0] = np.arange(20, 50, dtype=np.int64)
    second.joint_temporal_prefix_steps[30:, 0] = np.arange(50, 70, dtype=np.int64)
    second.joint_temporal_collision_sources[:, 0] = True
    second.joint_temporal_standard_residuals[:30, 0] = residuals[20:].numpy()
    second.joint_temporal_standard_residuals[30:, 0] = residuals[:20].numpy()
    incoming_uid_before_finalize = int(second.joint_context_carry[0]["block_uid"])
    second.finalize_joint_context_carry()
    incoming_uid_after_finalize = int(second.joint_context_carry[0]["block_uid"])
    outgoing_uid_after_finalize = int(second.joint_context_next_carry[0]["block_uid"])
    flatten_buffer(second)
    cross_context = second._joint_context_for_sequence(0)
    cross_fixed_residual_error = float((cross_context["fixed_residual_sum"] - residuals[:20].sum(dim=0)).abs().max())
    cross_current_context = second._joint_context_for_sequence(5)
    cross_position_error = int((cross_current_context["positions"] - torch.arange(20, 25)).abs().max())
    cross_action_error = float((cross_current_context["actions"] - actions[20:25].float()).abs().max())

    metrics = {
        "within_rollout_cuts_tested": 49,
        "within_rollout_maximum_context_length_error": maximum_context_length_error,
        "within_rollout_maximum_position_error": maximum_position_error,
        "within_rollout_maximum_action_error": maximum_action_error,
        "cross_rollout_cut": 20,
        "cross_rollout_fixed_context_rows": int(cross_context["fixed_count"]),
        "cross_rollout_current_context_rows_at_rollout_start": int(cross_context["positions"].numel()),
        "cross_rollout_current_context_rows_at_step5": int(cross_current_context["positions"].numel()),
        "cross_rollout_fixed_residual_sum_error": cross_fixed_residual_error,
        "cross_rollout_position_error": cross_position_error,
        "cross_rollout_action_error": cross_action_error,
        "incoming_uid_before_finalize": incoming_uid_before_finalize,
        "incoming_uid_after_finalize": incoming_uid_after_finalize,
        "outgoing_uid_after_finalize": outgoing_uid_after_finalize,
        "context_rows_enter_loss_advantage_or_metrics": 0,
    }
    passed = (
        maximum_context_length_error == 0
        and maximum_position_error == 0
        and maximum_action_error == 0.0
        and cross_context["block_uid"] == 9000000000001
        and int(cross_context["fixed_count"]) == 20
        and int(cross_context["positions"].numel()) == 0
        and int(cross_current_context["fixed_count"]) == 20
        and int(cross_current_context["positions"].numel()) == 5
        and cross_fixed_residual_error <= 5.0e-6
        and cross_position_error == 0
        and cross_action_error == 0.0
        and incoming_uid_before_finalize == incoming_uid_after_finalize == 9000000000001
        and outgoing_uid_after_finalize == 9000000000002
    )
    return {"passed": passed, "metrics": metrics}


class ExplorationCapture:

    def __init__(self):
        self.rows = []

    def stage_exploration(self, **kwargs):
        self.rows.append({name: np.asarray(value).copy() for name, value in kwargs.items()})


def build_policy(mode):
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(END2RACE_OBSERVATION_SIZE,), dtype=np.float32)
    action_space = spaces.Box(low=np.asarray((-STEERING_BOUND, -NOOP_SPEED_BOUND), dtype=np.float32), high=np.asarray((STEERING_BOUND, NOOP_SPEED_BOUND), dtype=np.float32), dtype=np.float32)
    return End2RaceGRUPolicy(observation_space, action_space, lambda _progress: 0.0, checkpoint_path=PROJECT_ROOT / "pretrained" / "end2race.pth", critic_variant="mlp", speed_exploration_mode=mode)


def disabled_and_sampler_gate():
    torch.manual_seed(FIXED_SEED)
    baseline = build_policy(BASELINE_EXPLORATION_MODE)
    torch.manual_seed(FIXED_SEED)
    treatment = build_policy(PREFIX_JOINT_TEMPORAL_EXPLORATION_MODE)
    baseline_capture = ExplorationCapture()
    treatment_capture = ExplorationCapture()
    baseline._end2race_rollout_buffer = baseline_capture
    treatment._end2race_rollout_buffer = treatment_capture
    treatment.configure_joint_temporal_generators(FIXED_SEED, 3)
    observation = torch.linspace(0.01, 1.0, END2RACE_OBSERVATION_SIZE, dtype=torch.float32).repeat(3, 1)
    hidden_size = treatment.end2race_actor.gru.hidden_size
    zero_states = RNNStates((torch.zeros(1, 3, hidden_size), torch.zeros(1, 3, hidden_size)), (torch.zeros(1, 3, hidden_size), torch.zeros(1, 3, hidden_size)))
    episode_starts = torch.zeros(3, dtype=torch.float32)
    treatment.prepare_rollout_exploration(np.zeros(3, dtype=bool), np.zeros(3, dtype=bool), np.zeros(3, dtype=bool), np.zeros(3, dtype=np.int64), np.zeros(3, dtype=bool))
    torch.manual_seed(FIXED_SEED + 1)
    baseline_rng_before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        baseline_actions, _baseline_values, baseline_log_prob, _baseline_states = baseline(observation, zero_states, episode_starts, deterministic=False)
    baseline_rng_after = torch.random.get_rng_state().clone()
    torch.manual_seed(FIXED_SEED + 1)
    treatment_rng_before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        disabled_actions, _disabled_values, disabled_log_prob, _disabled_states = treatment(observation, zero_states, episode_starts, deterministic=False)
    treatment_rng_after = torch.random.get_rng_state().clone()
    telemetry_equal = len(baseline_capture.rows) == len(treatment_capture.rows) == 1 and all(np.array_equal(baseline_capture.rows[0][name], treatment_capture.rows[0][name]) for name in baseline_capture.rows[0])

    sampler_capture = ExplorationCapture()
    treatment._end2race_rollout_buffer = sampler_capture
    treatment.configure_joint_temporal_generators(FIXED_SEED, 3)
    active_counts = np.zeros(3, dtype=np.int64)
    leak_counts = np.zeros(3, dtype=np.int64)
    active_positions = []
    active_uids = []
    all_finite = True
    maximum_bound_excess = 0.0
    for step in range(JOINT_TEMPORAL_PREFIX_STEPS + 1):
        prefix_active = np.asarray((True, True, False), dtype=bool)
        prefix_collision = np.asarray((True, False, True), dtype=bool)
        prefix_steps = np.asarray((step, step, step), dtype=np.int64)
        treatment.prepare_rollout_exploration(np.zeros(3, dtype=bool), np.zeros(3, dtype=bool), prefix_active, prefix_steps, prefix_collision)
        with torch.no_grad():
            _actions, _values, _log_prob, _states = treatment(observation, zero_states, episode_starts, deterministic=False)
        row = sampler_capture.rows[-1]
        active = row["joint_active"]
        active_counts += active.astype(np.int64)
        leak_counts[1:] += active[1:].astype(np.int64)
        if active[0]:
            active_positions.append(int(row["joint_block_position"][0]))
            active_uids.append(int(row["joint_block_uid"][0]))
        all_finite = all_finite and all(np.isfinite(row[name]).all() for name in ("speed_log_std", "standard_residual", "joint_standard_residual"))
        maximum_bound_excess = max(maximum_bound_excess, max(0.0, float(np.abs(_actions[:, 0].numpy()).max()) - STEERING_BOUND))
    expected_positions = list(range(JOINT_TEMPORAL_BLOCK_STEPS)) * 3
    uid_runs = [active_uids[index] for index in (0, 50, 100)]
    metrics = {
        "disabled_action_bitwise_equal": bool(torch.equal(baseline_actions, disabled_actions)),
        "disabled_log_prob_bitwise_equal": bool(torch.equal(baseline_log_prob, disabled_log_prob)),
        "disabled_global_rng_before_bitwise_equal": bool(torch.equal(baseline_rng_before, treatment_rng_before)),
        "disabled_global_rng_after_bitwise_equal": bool(torch.equal(baseline_rng_after, treatment_rng_after)),
        "disabled_telemetry_bitwise_equal": telemetry_equal,
        "active_counts_by_slot": active_counts.tolist(),
        "leak_counts_by_slot": leak_counts.tolist(),
        "active_position_sequence_exact": active_positions == expected_positions,
        "active_unique_block_uids": len(set(active_uids)),
        "active_block_start_uids": uid_runs,
        "finite": all_finite,
        "maximum_steering_bound_excess": maximum_bound_excess,
    }
    passed = all(bool(metrics[name]) for name in ("disabled_action_bitwise_equal", "disabled_log_prob_bitwise_equal", "disabled_global_rng_before_bitwise_equal", "disabled_global_rng_after_bitwise_equal", "disabled_telemetry_bitwise_equal", "active_position_sequence_exact", "finite")) and active_counts.tolist() == [150, 0, 0] and leak_counts.sum() == 0 and len(set(active_uids)) == 3 and maximum_bound_excess == 0.0
    return {"passed": passed, "metrics": metrics}


def rho_zero_gate(rng):
    candidate_values, actions = reference_case(JOINT_TEMPORAL_BLOCK_STEPS, rng)
    conditional_values = candidate_values.clone().requires_grad_(True)
    conditional_log_probs, _residuals = joint_temporal_sequence_log_prob(conditional_values, actions, rho=0.0)
    direct_values = candidate_values.clone().requires_grad_(True)
    direct_log_prob = direct_mvn_log_prob(direct_values, actions, 0.0)
    conditional_gradient = torch.autograd.grad(conditional_log_probs.sum(), conditional_values)[0]
    direct_gradient = torch.autograd.grad(direct_log_prob, direct_values)[0]
    metrics = {
        "joint_log_prob_error": abs(float(conditional_log_probs.sum().detach() - direct_log_prob.detach())),
        "gradient_maximum_error": float((conditional_gradient - direct_gradient).abs().max()),
        "all_finite": bool(torch.isfinite(conditional_log_probs).all() and torch.isfinite(conditional_gradient).all()),
    }
    passed = metrics["joint_log_prob_error"] <= 1.0e-10 and metrics["gradient_maximum_error"] <= 1.0e-10 and metrics["all_finite"]
    return {"passed": passed, "metrics": metrics}


def actor_contract_gate():
    checkpoint_path = PROJECT_ROOT / "pretrained" / "end2race.pth"
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    actor = End2Race(mask_prob=0.0, hidden_scale=4)
    actor.load_state_dict(state_dict, strict=True)
    actor.eval()
    lidar = torch.linspace(0.01, 1.0, 360, dtype=torch.float32).reshape(1, 1, 360)
    speed = torch.tensor([[[3.0]]], dtype=torch.float32)
    hidden = torch.zeros(1, 1, actor.gru.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        first_action, first_hidden = actor(lidar, speed, hidden)
        second_action, second_hidden = actor(lidar, speed, hidden)
    metrics = {
        "checkpoint_key_count": len(state_dict),
        "strict_load": True,
        "deterministic_action_bitwise_equal": bool(torch.equal(first_action, second_action)),
        "deterministic_hidden_bitwise_equal": bool(torch.equal(first_hidden, second_hidden)),
        "finite": bool(torch.isfinite(first_action).all() and torch.isfinite(first_hidden).all()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    passed = len(state_dict) == 12 and metrics["deterministic_action_bitwise_equal"] and metrics["deterministic_hidden_bitwise_equal"] and metrics["finite"]
    return {"passed": passed, "metrics": metrics}


if __name__ == "__main__":
    args = parse_arguments()
    output_path = Path(args.output).expanduser().resolve()
    rng = np.random.default_rng(FIXED_SEED)
    tests = {}
    tests["synthetic_covariance"] = covariance_gate(rng)
    tests["likelihood_reference"] = likelihood_reference_gate(rng)
    cut_result = cut_gate(rng)
    candidate_values = cut_result.pop("candidate_values")
    actions = cut_result.pop("actions")
    tests["all_minibatch_cuts"] = cut_result
    tests["buffer_and_cross_rollout_context"] = buffer_context_gate(candidate_values, actions)
    tests["rho_zero_baseline_equivalence"] = rho_zero_gate(rng)
    tests["disabled_fast_path_and_sampler_scope"] = disabled_and_sampler_gate()
    tests["deployment_actor_contract"] = actor_contract_gate()
    verdict = "pass" if all(test["passed"] for test in tests.values()) else "fail"
    report = {
        "gate": "E0",
        "verdict": verdict,
        "fixed_seed": FIXED_SEED,
        "method": {
            "rho": JOINT_TEMPORAL_RHO,
            "block_steps": JOINT_TEMPORAL_BLOCK_STEPS,
            "eligible_prefix_steps": JOINT_TEMPORAL_PREFIX_STEPS,
            "steering_latent_std": STEERING_LATENT_STD,
            "speed_physical_std": SPEED_PHYSICAL_STD,
        },
        "tests": tests,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "source_sha256": {
            "preregistration": sha256_file(PROJECT_ROOT / ".agents" / "FINAL_PREFIX_LOCAL_JOINT_TEMPORAL_EXPLORATION_PREREGISTRATION.md"),
            "policy": sha256_file(PROJECT_ROOT / "ppo" / "policy.py"),
            "rollout": sha256_file(PROJECT_ROOT / "ppo" / "rollout.py"),
            "environment": sha256_file(PROJECT_ROOT / "ppo" / "env.py"),
            "train": sha256_file(PROJECT_ROOT / "train_ppo.py"),
            "gate_script": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(output_path, report)
    print(json.dumps({"output": str(output_path), "verdict": verdict, "tests": {name: value["passed"] for name, value in tests.items()}}, indent=2))
    if verdict != "pass":
        raise SystemExit(1)
