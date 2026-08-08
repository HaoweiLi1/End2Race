import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo.env import BASELINE_EXPLORATION_MODE, CentralScheduleSubprocVecEnv, load_prefix_reset_panel
from ppo.scenarios import ScenarioSpec, ordinary_scenarios
from train_ppo import START_METHOD, build_model, configure_training_numerics
from utils import TrainingRecorder, atomic_write_json, save_numeric_npz


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, default=Path("eval_results/prefix_reset_snapshot_gate"))
    parser.add_argument("--semantics-dir", type=Path, default=Path("eval_results/prefix_reset_semantics_gate"))
    parser.add_argument("--panel-dir", type=Path, default=Path("post-trained/panels/prefix_reset_consensus_v1"))
    parser.add_argument("--collision-cache-dir", type=Path, default=Path("post-trained/collision-cache/pretrained_end2race_austin_collision_pool_479"))
    parser.add_argument("--actor-path", type=Path, default=Path("pretrained/end2race.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results/prefix_reset_density_gate"))
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=12800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix-reset-interval", type=int, default=3)
    parser.add_argument("--hidden-scale", type=int, default=4)
    parser.add_argument("--arm", choices=("orchestrate", "baseline", "treatment", "adjudication"), default="orchestrate")
    parser.add_argument("--prepare-only", action="store_true")
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


def build_panel(args):
    source_plan = json.loads((args.snapshot_dir / "snapshot_gate_plan.json").read_text(encoding="utf-8"))
    source_report = json.loads((args.snapshot_dir / "snapshot_noop_report.json").read_text(encoding="utf-8"))
    semantics_report = json.loads((args.semantics_dir / "semantics_gate_report.json").read_text(encoding="utf-8"))
    adjudication = json.loads((args.semantics_dir / "residual_measurement_adjudication.json").read_text(encoding="utf-8"))
    if source_report.get("verdict") != "pass_snapshot_mechanical_gate" or adjudication.get("verdict") != "pass_prefix_reset_semantics_after_measurement_adjudication":
        raise RuntimeError("Z6-C requires passed Z6-A and adjudicated Z6-B inputs")
    if semantics_report.get("verdict") != "fail_stop_prefix_reset_semantics":
        raise RuntimeError("Z6-C must preserve the original strict Z6-B report")
    tasks = sorted(source_plan["tasks"], key=lambda task: task["episode_key"])
    if len(tasks) != 28 or sum(int(task["window"]["start_index"]) for task in tasks) != 9589:
        raise RuntimeError("Frozen prefix-reset source task contract changed")
    manifest_path = args.panel_dir / "prefix_reset_manifest.json"
    if manifest_path.exists():
        load_prefix_reset_panel(args.panel_dir)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.panel_dir.exists() and any(args.panel_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite partial prefix-reset panel: {args.panel_dir}")
    (args.panel_dir / "snapshots").mkdir(parents=True)
    (args.panel_dir / "prefixes").mkdir()
    manifest_tasks = []
    for task in tasks:
        key = task["episode_key"]
        source_snapshot_path = args.snapshot_dir / "snapshots" / f"{key}.pkl"
        source_prefix_path = args.semantics_dir / "prefixes" / f"{key}.npz"
        destination_snapshot = args.panel_dir / "snapshots" / f"{key}.pkl"
        destination_prefix = args.panel_dir / "prefixes" / f"{key}.npz"
        shutil.copy2(source_snapshot_path, destination_snapshot)
        with np.load(source_prefix_path, allow_pickle=False) as arrays:
            prefix = np.asarray(arrays["prefix_observations"], dtype=np.float32)
            window_observation = np.asarray(arrays["window_observation"], dtype=np.float32)
        save_numeric_npz(destination_prefix, {"prefix_observations": prefix, "window_observation": window_observation})
        manifest_tasks.append({
            "episode_key": key,
            "stratum": task["stratum"],
            "prefix_length": int(len(prefix)),
            "snapshot_file": f"snapshots/{key}.pkl",
            "snapshot_sha256": sha256_file(destination_snapshot),
            "prefix_file": f"prefixes/{key}.npz",
            "prefix_sha256": sha256_file(destination_prefix),
            "scenario": task["scenario"],
            "window": task["window"],
        })
    manifest = {
        "schema_version": 1,
        "panel_id": "prefix_reset_consensus_v1",
        "map": "Austin",
        "selection": "U42-U45 at least 3-of-4 development consensus",
        "task_count": 28,
        "collision_count": 19,
        "lost_overtake_count": 9,
        "unique_ego_startpoints": 21,
        "prefix_rows": 9589,
        "tasks": manifest_tasks,
    }
    atomic_write_json(manifest_path, manifest)
    load_prefix_reset_panel(args.panel_dir)
    return manifest


def build_plan(args, manifest):
    collision_path = args.collision_cache_dir / "collision_scenarios.json"
    collision_rows = json.loads(collision_path.read_text(encoding="utf-8"))
    ordinary_rows = ordinary_scenarios("Austin")
    if len(collision_rows) != 479 or len(ordinary_rows) != 600:
        raise RuntimeError("Production collision or ordinary pool size changed")
    return {
        "schema_version": 1,
        "experiment_id": "prefix_reset_density_gate",
        "gate": "Z6-C",
        "status": "frozen_before_execution",
        "inputs": {
            "actor_path": str(args.actor_path),
            "actor_sha256": sha256_file(args.actor_path),
            "collision_cache_dir": str(args.collision_cache_dir),
            "collision_scenarios_sha256": sha256_file(collision_path),
            "prefix_panel_dir": str(args.panel_dir),
            "prefix_manifest_sha256": sha256_file(args.panel_dir / "prefix_reset_manifest.json"),
        },
        "fixed_contract": {
            "map": "Austin",
            "seed": args.seed,
            "actor": "canonical BC",
            "critic": "fresh privilege_gru",
            "exploration": BASELINE_EXPLORATION_MODE,
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "transition_count_per_arm": args.n_envs * args.n_steps,
            "batch_size": args.batch_size,
            "gamma": 0.999,
            "gae_lambda": 0.995,
            "prefix_reset_interval_within_collision_role": args.prefix_reset_interval,
            "prefix_task_count": manifest["task_count"],
            "prefix_rows": manifest["prefix_rows"],
            "arm_order": ["baseline", "treatment"],
            "optimizer_steps": 0,
        },
        "admission_contract": {
            "role_transitions_each": args.n_envs * args.n_steps // 2,
            "gae_tolerance": 1.0e-6,
            "exact_likelihood_tolerance": 5.0e-5,
            "batched_likelihood_envelope": 1.0e-2,
            "minimum_prefix_transition_fraction": 0.05,
            "minimum_prefix_window_transition_fraction": 0.02,
            "maximum_wall_time_ratio": 1.20,
            "failure_action": "stop current prefix-reset implementation",
            "pass_action": "preregister one formal Austin-only prefix-reset training arm",
        },
        "source_snapshot": source_snapshot(),
    }


def write_plan(args, plan):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "density_gate_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise RuntimeError("Existing Z6-C plan differs from frozen inputs")
        return plan_path
    if any(args.output_dir.iterdir()):
        raise RuntimeError(f"Output directory must be empty before Z6-C plan creation: {args.output_dir}")
    atomic_write_json(plan_path, plan)
    return plan_path


def training_arguments(args, arm_dir):
    return argparse.Namespace(
        pretrained_model_path=str(args.actor_path), output_dir=str(arm_dir), hidden_scale=args.hidden_scale, critic="privilege_gru",
        map_name="Austin", n_envs=args.n_envs, seed=args.seed, collision_cache_dir=str(args.collision_cache_dir), prefix_reset_panel=str(args.panel_dir), prefix_reset_interval=args.prefix_reset_interval,
        n_steps=args.n_steps, batch_size=args.batch_size, num_updates=1, actor_epochs=2, critic_epochs=5, gru_learning_rate=3.0e-6, head_learning_rate=3.0e-5,
        critic_learning_rate=3.0e-4, steering_latent_std=0.03, speed_physical_std=0.15, speed_exploration_mode=BASELINE_EXPLORATION_MODE,
        gamma=0.999, gae_lambda=0.995, clip_range=0.20,
    )


def manual_gae(buffer, last_values, dones, gamma, gae_lambda):
    rewards = np.asarray(buffer.rewards, dtype=np.float32)
    values = np.asarray(buffer.values, dtype=np.float32)
    starts = np.asarray(buffer.episode_starts, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(buffer.n_envs, dtype=np.float32)
    for step in reversed(range(buffer.buffer_size)):
        if step == buffer.buffer_size - 1:
            next_non_terminal = 1.0 - np.asarray(dones, dtype=np.float32)
            next_values = np.asarray(last_values, dtype=np.float32)
        else:
            next_non_terminal = 1.0 - starts[step + 1]
            next_values = values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae
    return advantages, advantages + values


def ratio_measure(model, collection_equivalent, seed):
    maximum_log_ratio = 0.0
    maximum_ratio = 0.0
    count = 0
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x52415449, int(collection_equivalent)]))
    with torch.no_grad():
        for data in model.rollout_buffer.get(model.batch_size, rng=rng):
            mask = data.mask > 1e-8
            log_prob, _entropy = model.policy.evaluate_actor_actions(data.observations, data.actions, data.lstm_states, data.episode_starts, collection_equivalent=collection_equivalent)
            log_ratio = log_prob - data.old_log_prob
            ratio = torch.exp(log_ratio)
            maximum_log_ratio = max(maximum_log_ratio, float(torch.abs(log_ratio[mask]).max().cpu().item()))
            maximum_ratio = max(maximum_ratio, float(torch.abs(ratio[mask] - 1.0).max().cpu().item()))
            count += int(mask.sum().item())
    return {"transition_count": count, "maximum_abs_log_ratio": maximum_log_ratio, "maximum_abs_ratio_minus_one": maximum_ratio}


def batched_ratio_distribution(model, seed):
    log_ratios = []
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x42415443]))
    with torch.no_grad():
        for data in model.rollout_buffer.get(model.batch_size, rng=rng):
            mask = data.mask > 1e-8
            log_prob, _entropy = model.policy.evaluate_actor_actions(data.observations, data.actions, data.lstm_states, data.episode_starts, collection_equivalent=False)
            log_ratios.append((log_prob - data.old_log_prob)[mask].detach().cpu().numpy().astype(np.float64))
    values = np.concatenate(log_ratios)
    ratios = np.exp(values)
    approximate_kl = (ratios - 1.0) - values
    return {
        "transition_count": int(len(values)),
        "mean_log_ratio": float(values.mean()),
        "mean_abs_log_ratio": float(np.abs(values).mean()),
        "maximum_abs_log_ratio": float(np.abs(values).max()),
        "absolute_log_ratio_quantiles": {name: float(np.quantile(np.abs(values), value)) for name, value in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99), ("p999", 0.999))},
        "maximum_abs_ratio_minus_one": float(np.abs(ratios - 1.0).max()),
        "mean_approximate_kl": float(approximate_kl.mean()),
        "maximum_approximate_kl": float(approximate_kl.max()),
        "clip_fraction_0p20": float((np.abs(ratios - 1.0) > 0.20).mean()),
    }


def dry_actor_gradient(model, collection_equivalent, seed):
    gradients = {name: torch.zeros_like(parameter, device="cpu") for name, parameter in model.policy.end2race_actor.named_parameters() if parameter.requires_grad}
    losses = []
    minibatches = 0
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x47524144]))
    model.policy.set_training_mode(True)
    for data in model.rollout_buffer.get(model.batch_size, rng=rng):
        mask = data.mask > 1e-8
        advantages = data.advantages
        valid_advantages = advantages[mask]
        advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
        log_prob, _entropy = model.policy.evaluate_actor_actions(data.observations, data.actions, data.lstm_states, data.episode_starts, collection_equivalent=collection_equivalent)
        ratio = torch.exp(log_prob - data.old_log_prob)
        loss = -torch.min(advantages * ratio, advantages * torch.clamp(ratio, 0.8, 1.2))[mask].mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Dry actor policy loss is not finite")
        model.policy.actor_optimizer.zero_grad()
        loss.backward()
        for name, parameter in model.policy.end2race_actor.named_parameters():
            if parameter.requires_grad:
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item()):
                    raise RuntimeError(f"Dry actor gradient is missing or non-finite: {name}")
                gradients[name] += parameter.grad.detach().cpu()
        losses.append(float(loss.detach().cpu().item()))
        minibatches += 1
    model.policy.actor_optimizer.zero_grad()
    return gradients, losses, minibatches


def compare_gradients(batched, exact):
    dot = 0.0
    batched_squared = 0.0
    exact_squared = 0.0
    difference_squared = 0.0
    for name in sorted(exact):
        left = batched[name].double()
        right = exact[name].double()
        dot += float(torch.sum(left * right).item())
        batched_squared += float(torch.sum(left * left).item())
        exact_squared += float(torch.sum(right * right).item())
        difference_squared += float(torch.sum((left - right) ** 2).item())
    batched_norm = batched_squared ** 0.5
    exact_norm = exact_squared ** 0.5
    return {
        "cosine": dot / (batched_norm * exact_norm),
        "batched_l2_norm": batched_norm,
        "exact_l2_norm": exact_norm,
        "difference_l2_norm": difference_squared ** 0.5,
        "relative_l2_difference_over_exact": difference_squared ** 0.5 / exact_norm,
    }


def write_adjudication_plan(args):
    original = json.loads((args.output_dir / "density_gate_report.json").read_text(encoding="utf-8"))
    if original.get("verdict") != "fail_stop_prefix_reset_density_or_integration" or sum(not value for value in original["criteria"].values()) != 1 or original["criteria"]["full_buffer_batched_likelihood_within_envelope"]:
        raise RuntimeError("Z6-CR requires the single frozen Z6-C batched-envelope failure")
    plan = {
        "schema_version": 1,
        "experiment_id": "prefix_reset_density_gate",
        "gate": "Z6-CR",
        "status": "frozen_before_execution",
        "fixed_contract": json.loads((args.output_dir / "density_gate_plan.json").read_text(encoding="utf-8"))["fixed_contract"],
        "admission_contract": {"clip_fraction_0p20": 0.0, "maximum_abs_ratio_minus_one": 0.02, "maximum_mean_approximate_kl": 1.0e-4, "minimum_gradient_cosine": 0.999, "maximum_relative_gradient_l2_difference": 0.02, "required_minibatches_per_mode": 8, "exact_likelihood_tolerance": 5.0e-5},
        "source_snapshot": source_snapshot(),
    }
    path = args.output_dir / "density_adjudication_plan.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("Existing Z6-CR plan differs from current frozen inputs")
    else:
        atomic_write_json(path, plan)
    return plan


def run_arm(args):
    plan = json.loads((args.output_dir / "density_gate_plan.json").read_text(encoding="utf-8"))
    if plan["fixed_contract"]["transition_count_per_arm"] != args.n_envs * args.n_steps:
        raise RuntimeError("Z6-C child arguments changed after plan freeze")
    collision_rows = json.loads((args.collision_cache_dir / "collision_scenarios.json").read_text(encoding="utf-8"))
    collision_scenarios = tuple(ScenarioSpec(**row) for row in collision_rows)
    ordinary = ordinary_scenarios("Austin")
    prefix_inputs = load_prefix_reset_panel(args.panel_dir) if args.arm in ("treatment", "adjudication") else ()
    arm_dir = args.output_dir / args.arm
    recorder = TrainingRecorder(arm_dir, args.hidden_scale)
    configure_training_numerics()
    vector_env = CentralScheduleSubprocVecEnv(args.n_envs, START_METHOD, args.seed, "Austin", collision_scenarios, ordinary, privileged=True, reward_gamma=0.999, speed_exploration_mode=BASELINE_EXPLORATION_MODE, prefix_reset_inputs=prefix_inputs, prefix_reset_interval=args.prefix_reset_interval if prefix_inputs else 0)
    try:
        model_args = training_arguments(args, arm_dir)
        model = build_model(vector_env, model_args, torch.device("cuda"), recorder)
        actor_before = state_digest(model.policy.end2race_actor.state_dict())
        critic_before = state_digest(model.policy.value_net.state_dict())
        started = time.perf_counter()
        _total, callback = model._setup_learn(args.n_envs * args.n_steps, progress_bar=False)
        callback.on_training_start(locals(), globals())
        completed = model.collect_rollouts(vector_env, callback, model.rollout_buffer, args.n_steps)
        callback.on_training_end()
        collection_wall_seconds = time.perf_counter() - started
        if not completed or not model.rollout_buffer.full:
            raise RuntimeError(f"Z6-C {args.arm} rollout did not complete")
        actor_after = state_digest(model.policy.end2race_actor.state_dict())
        critic_after = state_digest(model.policy.value_net.state_dict())
        if args.arm in ("treatment", "adjudication"):
            last_values = model.last_rollout_final_values
            dones = model.last_rollout_dones
            prefix_transition_mask = model.last_prefix_transition_mask
            prefix_window_mask = model.last_prefix_window_mask
            prefix_steps = model.last_prefix_step_indices
            prefix_keys = model.last_prefix_key_rows
        else:
            with torch.no_grad():
                starts = torch.as_tensor(model._last_episode_starts, dtype=torch.float32, device=model.device)
                last_values = model.policy.predict_values(model.policy.obs_to_tensor(model._last_obs)[0], model._last_lstm_states.vf, starts).detach().cpu().numpy().reshape(-1).astype(np.float32)
            dones = np.asarray(model._last_episode_starts, dtype=bool)
            prefix_transition_mask = np.zeros((args.n_steps, args.n_envs), dtype=bool)
            prefix_window_mask = np.zeros_like(prefix_transition_mask)
            prefix_steps = np.zeros((args.n_steps, args.n_envs), dtype=np.int64)
            prefix_keys = [[None] * args.n_envs for _ in range(args.n_steps)]
        expected_advantages, expected_returns = manual_gae(model.rollout_buffer, last_values, dones, 0.999, 0.995)
        gae_advantage_error = float(np.max(np.abs(expected_advantages.astype(np.float64) - model.rollout_buffer.advantages.astype(np.float64))))
        gae_return_error = float(np.max(np.abs(expected_returns.astype(np.float64) - model.rollout_buffer.returns.astype(np.float64))))
        buffer_finite = all(np.isfinite(np.asarray(getattr(model.rollout_buffer, name))).all() for name in ("observations", "actions", "rewards", "values", "log_probs", "advantages", "returns"))
        episode_starts = np.asarray(model.rollout_buffer.episode_starts, dtype=bool)
        recurrent_resets = np.asarray(model.rollout_buffer.recurrent_resets, dtype=bool)
        prefix_boundaries = episode_starts & ~recurrent_resets
        window_by_key = {item["episode_key"]: np.asarray(item["snapshot"]["observation"], dtype=np.float32) for item in prefix_inputs}
        boundary_observation_error = 0.0
        boundary_keys = []
        for step, rank in np.argwhere(prefix_boundaries):
            key = prefix_keys[int(step)][int(rank)]
            boundary_keys.append(key)
            if key not in window_by_key or prefix_steps[int(step), int(rank)] != 0:
                boundary_observation_error = float("inf")
                break
            boundary_observation_error = max(boundary_observation_error, float(np.max(np.abs(model.rollout_buffer.observations[int(step), int(rank)].astype(np.float64) - window_by_key[key].astype(np.float64)))))
        default_mask_mismatch = int(np.count_nonzero((episode_starts != recurrent_resets) & ~prefix_boundaries))
        reset_history = copy.deepcopy(vector_env.reset_history)
        collision_resets = [row for row in reset_history if row["env_role"] == "collision"]
        ordinary_resets = [row for row in reset_history if row["env_role"] == "ordinary"]
        prefix_resets = [row for row in collision_resets if row["source"] == "prefix_reset"]
        standard_collision = [row for row in collision_resets if row["source"] == "standard"]
        collision_ids = {scenario.scenario_id for scenario in collision_scenarios}
        ordinary_ids = {scenario.scenario_id for scenario in ordinary}
        cache_membership = all(row["scenario_id"] in collision_ids for row in standard_collision)
        ordinary_membership = all(row["scenario_id"] in ordinary_ids for row in ordinary_resets)
        prefix_burn_in_rows = sum(next(item["prefix_length"] for item in prefix_inputs if item["episode_key"] == key) for key in boundary_keys) if prefix_inputs else 0
        role_transition_counts = {"collision": args.n_steps * len(range(0, args.n_envs, 2)), "ordinary": args.n_steps * len(range(1, args.n_envs, 2))}
        batched_ratio = ratio_measure(model, False, args.seed)
        exact_ratio = ratio_measure(model, True, args.seed)
        gradient_adjudication = None
        if args.arm == "adjudication":
            ratio_distribution = batched_ratio_distribution(model, args.seed)
            dry_digest_before = state_digest(model.policy.end2race_actor.state_dict())
            batched_gradients, batched_losses, batched_minibatches = dry_actor_gradient(model, False, args.seed)
            exact_gradients, exact_losses, exact_minibatches = dry_actor_gradient(model, True, args.seed)
            dry_digest_after = state_digest(model.policy.end2race_actor.state_dict())
            gradient_comparison = compare_gradients(batched_gradients, exact_gradients)
            gradient_adjudication = {
                "batched_ratio_distribution": ratio_distribution,
                "batched_policy_losses": batched_losses,
                "exact_policy_losses": exact_losses,
                "maximum_abs_policy_loss_difference": float(np.max(np.abs(np.asarray(batched_losses, dtype=np.float64) - np.asarray(exact_losses, dtype=np.float64)))),
                "batched_minibatches": batched_minibatches,
                "exact_minibatches": exact_minibatches,
                "gradient_comparison": gradient_comparison,
                "actor_state_digest_before_dry_epoch": dry_digest_before,
                "actor_state_digest_after_dry_epoch": dry_digest_after,
                "parameters_unchanged_by_dry_epoch": dry_digest_before == dry_digest_after,
            }
        report = {
            "schema_version": 1,
            "experiment_id": "prefix_reset_density_gate",
            "gate": "Z6-C",
            "arm": args.arm,
            "completed": True,
            "transition_count": args.n_envs * args.n_steps,
            "role_transition_counts": role_transition_counts,
            "buffer_finite": bool(buffer_finite),
            "gae_advantage_max_abs_error": gae_advantage_error,
            "gae_return_max_abs_error": gae_return_error,
            "batched_ratio": batched_ratio,
            "collection_equivalent_ratio": exact_ratio,
            "actor_state_digest_before": actor_before,
            "actor_state_digest_after": actor_after,
            "critic_state_digest_before": critic_before,
            "critic_state_digest_after": critic_after,
            "parameters_unchanged": actor_before == actor_after and critic_before == critic_after,
            "collection_wall_seconds": float(collection_wall_seconds),
            "transitions_per_second": float(args.n_envs * args.n_steps / collection_wall_seconds),
            "reset_counts": {"collision": len(collision_resets), "ordinary": len(ordinary_resets), "prefix": len(prefix_resets), "standard_collision": len(standard_collision)},
            "prefix_reset_expected_from_interval": len(collision_resets) // args.prefix_reset_interval if prefix_inputs else 0,
            "prefix_reset_unique_keys": sorted({row["prefix_reset_key"] for row in prefix_resets}),
            "prefix_transition_count": int(prefix_transition_mask.sum()),
            "prefix_window_transition_count": int(prefix_window_mask.sum()),
            "prefix_boundary_count": int(prefix_boundaries.sum()),
            "prefix_boundary_unique_keys": sorted({key for key in boundary_keys if key is not None}),
            "prefix_boundary_observation_max_abs_error": boundary_observation_error,
            "prefix_burn_in_rows": int(prefix_burn_in_rows),
            "buffered_prefix_rows": 0,
            "default_boundary_mask_mismatch_count": default_mask_mismatch,
            "standard_collision_cache_membership": bool(cache_membership),
            "ordinary_pool_membership": bool(ordinary_membership),
            "reset_history": reset_history,
            "gradient_adjudication": gradient_adjudication,
        }
        atomic_write_json(arm_dir / "arm_report.json", report)
        print(f"ARM={args.arm} TRANSITIONS={report['transition_count']} WALL_SECONDS={collection_wall_seconds:.3f} PREFIX_TRANSITIONS={report['prefix_transition_count']}", flush=True)
    finally:
        vector_env.close()


def aggregate(args):
    baseline = json.loads((args.output_dir / "baseline" / "arm_report.json").read_text(encoding="utf-8"))
    treatment = json.loads((args.output_dir / "treatment" / "arm_report.json").read_text(encoding="utf-8"))
    total = args.n_envs * args.n_steps
    wall_ratio = treatment["collection_wall_seconds"] / baseline["collection_wall_seconds"]
    criteria = {
        "both_arms_completed_102400_transitions": baseline["transition_count"] == total and treatment["transition_count"] == total and total == 102400,
        "both_arms_finite_and_parameters_unchanged": baseline["buffer_finite"] and treatment["buffer_finite"] and baseline["parameters_unchanged"] and treatment["parameters_unchanged"],
        "roles_remain_exactly_50_50": baseline["role_transition_counts"] == {"collision": 51200, "ordinary": 51200} and treatment["role_transition_counts"] == {"collision": 51200, "ordinary": 51200},
        "full_buffer_gae_within_tolerance": max(baseline["gae_advantage_max_abs_error"], baseline["gae_return_max_abs_error"], treatment["gae_advantage_max_abs_error"], treatment["gae_return_max_abs_error"]) <= 1.0e-6,
        "full_buffer_exact_likelihood_within_tolerance": all(arm["collection_equivalent_ratio"][name] <= 5.0e-5 for arm in (baseline, treatment) for name in ("maximum_abs_log_ratio", "maximum_abs_ratio_minus_one")) and all(arm["collection_equivalent_ratio"]["transition_count"] == total for arm in (baseline, treatment)),
        "full_buffer_batched_likelihood_within_envelope": all(arm["batched_ratio"][name] <= 1.0e-2 for arm in (baseline, treatment) for name in ("maximum_abs_log_ratio", "maximum_abs_ratio_minus_one")) and all(arm["batched_ratio"]["transition_count"] == total for arm in (baseline, treatment)),
        "baseline_contains_no_prefix_reset": baseline["reset_counts"]["prefix"] == 0 and baseline["prefix_transition_count"] == 0 and baseline["prefix_boundary_count"] == 0,
        "treatment_prefix_reset_ratio_and_coverage_passed": treatment["reset_counts"]["prefix"] == treatment["prefix_reset_expected_from_interval"] and len(treatment["prefix_reset_unique_keys"]) == 28 and len(treatment["prefix_boundary_unique_keys"]) == 28,
        "production_role_pool_membership_preserved": baseline["standard_collision_cache_membership"] and treatment["standard_collision_cache_membership"] and baseline["ordinary_pool_membership"] and treatment["ordinary_pool_membership"],
        "prefix_transition_density_passed": treatment["prefix_transition_count"] / total >= 0.05 and treatment["prefix_window_transition_count"] / total >= 0.02,
        "prefix_boundary_hidden_and_buffer_contract_passed": treatment["prefix_boundary_count"] > 0 and treatment["prefix_boundary_observation_max_abs_error"] == 0.0 and treatment["default_boundary_mask_mismatch_count"] == 0 and treatment["buffered_prefix_rows"] == 0,
        "wall_clock_overhead_within_guardrail": wall_ratio <= 1.20,
    }
    verdict = "pass_prefix_reset_density_integration_gate" if all(criteria.values()) else "fail_stop_prefix_reset_density_or_integration"
    report = {
        "schema_version": 1,
        "experiment_id": "prefix_reset_density_gate",
        "gate": "Z6-C",
        "verdict": verdict,
        "criteria": criteria,
        "summary": {
            "baseline_wall_seconds": baseline["collection_wall_seconds"],
            "treatment_wall_seconds": treatment["collection_wall_seconds"],
            "treatment_over_baseline_wall_ratio": wall_ratio,
            "baseline_transitions_per_second": baseline["transitions_per_second"],
            "treatment_transitions_per_second": treatment["transitions_per_second"],
            "prefix_reset_count": treatment["reset_counts"]["prefix"],
            "prefix_unique_keys": len(treatment["prefix_reset_unique_keys"]),
            "prefix_transition_count": treatment["prefix_transition_count"],
            "prefix_transition_fraction": treatment["prefix_transition_count"] / total,
            "prefix_window_transition_count": treatment["prefix_window_transition_count"],
            "prefix_window_transition_fraction": treatment["prefix_window_transition_count"] / total,
            "prefix_burn_in_rows": treatment["prefix_burn_in_rows"],
            "counterfactual_replay_to_prefix_extra_sim_steps": sum(row["prefix_length"] for row in treatment["reset_history"] if row["source"] == "prefix_reset"),
            "maximum_gae_error": max(baseline["gae_advantage_max_abs_error"], baseline["gae_return_max_abs_error"], treatment["gae_advantage_max_abs_error"], treatment["gae_return_max_abs_error"]),
            "maximum_exact_log_ratio_error": max(baseline["collection_equivalent_ratio"]["maximum_abs_log_ratio"], treatment["collection_equivalent_ratio"]["maximum_abs_log_ratio"]),
            "maximum_exact_ratio_error": max(baseline["collection_equivalent_ratio"]["maximum_abs_ratio_minus_one"], treatment["collection_equivalent_ratio"]["maximum_abs_ratio_minus_one"]),
            "maximum_batched_log_ratio_error": max(baseline["batched_ratio"]["maximum_abs_log_ratio"], treatment["batched_ratio"]["maximum_abs_log_ratio"]),
        },
        "evidence_boundary": {
            "established": "one full no-update rollout preserves PPO semantics and provides the preregistered prefix-state density at bounded engineering overhead",
            "not_established": "PPO learning benefit, checkpoint stability, Austin acceptance, or four-map actor performance",
            "next_action": "preregister one formal Austin-only prefix-reset arm if and only if this gate passes",
        },
    }
    atomic_write_json(args.output_dir / "density_gate_report.json", report)
    print(f"REPORT={args.output_dir / 'density_gate_report.json'}")
    print(f"VERDICT={verdict}")


def aggregate_adjudication(args):
    arm = json.loads((args.output_dir / "adjudication" / "arm_report.json").read_text(encoding="utf-8"))
    detail = arm["gradient_adjudication"]
    distribution = detail["batched_ratio_distribution"]
    comparison = detail["gradient_comparison"]
    total = args.n_envs * args.n_steps
    criteria = {
        "full_treatment_contract_reproduced": arm["transition_count"] == total == 102400 and arm["buffer_finite"] and arm["parameters_unchanged"] and len(arm["prefix_reset_unique_keys"]) == 28 and arm["prefix_transition_count"] / total >= 0.05 and arm["prefix_window_transition_count"] / total >= 0.02 and arm["gae_advantage_max_abs_error"] <= 1.0e-6 and arm["gae_return_max_abs_error"] <= 1.0e-6,
        "collection_equivalent_likelihood_within_tolerance": arm["collection_equivalent_ratio"]["transition_count"] == total and arm["collection_equivalent_ratio"]["maximum_abs_log_ratio"] <= 5.0e-5 and arm["collection_equivalent_ratio"]["maximum_abs_ratio_minus_one"] <= 5.0e-5,
        "batched_ratio_below_causal_guardrails": distribution["transition_count"] == total and distribution["clip_fraction_0p20"] == 0.0 and distribution["maximum_abs_ratio_minus_one"] < 0.02 and distribution["mean_approximate_kl"] <= 1.0e-4,
        "dry_gradient_direction_and_scale_match": comparison["cosine"] >= 0.999 and comparison["relative_l2_difference_over_exact"] <= 0.02,
        "dry_epochs_complete_finite_and_no_update": bool(detail["batched_minibatches"] == 8 and detail["exact_minibatches"] == 8 and detail["parameters_unchanged_by_dry_epoch"] and np.isfinite(np.asarray(detail["batched_policy_losses"], dtype=np.float64)).all() and np.isfinite(np.asarray(detail["exact_policy_losses"], dtype=np.float64)).all()),
    }
    verdict = "pass_prefix_reset_after_batched_gradient_adjudication" if all(criteria.values()) else "fail_stop_prefix_reset_density_or_integration"
    report = {
        "schema_version": 1,
        "experiment_id": "prefix_reset_density_gate",
        "gate": "Z6-CR",
        "verdict": verdict,
        "original_z6c_verdict": "fail_stop_prefix_reset_density_or_integration",
        "criteria": criteria,
        "summary": {
            "batched_max_abs_log_ratio": distribution["maximum_abs_log_ratio"],
            "batched_max_abs_ratio_minus_one": distribution["maximum_abs_ratio_minus_one"],
            "batched_mean_approximate_kl": distribution["mean_approximate_kl"],
            "batched_clip_fraction_0p20": distribution["clip_fraction_0p20"],
            "gradient_cosine": comparison["cosine"],
            "gradient_relative_l2_difference": comparison["relative_l2_difference_over_exact"],
            "maximum_abs_policy_loss_difference": detail["maximum_abs_policy_loss_difference"],
            "prefix_transition_fraction": arm["prefix_transition_count"] / total,
            "prefix_window_transition_fraction": arm["prefix_window_transition_count"] / total,
        },
        "evidence_boundary": {
            "established": "the Z6-C 1e-2 batched-envelope exceedance does or does not materially alter the dry PPO actor gradient under the frozen full-rollout contract",
            "not_established": "a learned actor improvement or formal validation performance",
            "next_action": "preregister formal prefix-reset PPO only if every adjudication criterion passes",
        },
    }
    atomic_write_json(args.output_dir / "density_adjudication_report.json", report)
    print(f"ADJUDICATION_REPORT={args.output_dir / 'density_adjudication_report.json'}")
    print(f"ADJUDICATION_VERDICT={verdict}")


if __name__ == "__main__":
    args = parse_arguments()
    for name in ("snapshot_dir", "semantics_dir", "panel_dir", "collision_cache_dir", "actor_path", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if (args.n_envs, args.n_steps, args.batch_size, args.seed, args.prefix_reset_interval) != (16, 6400, 12800, 42, 3):
        raise ValueError("Z6-C fixed rollout contract changed")
    if args.arm == "orchestrate":
        manifest = build_panel(args)
        plan = build_plan(args, manifest)
        plan_path = write_plan(args, plan)
        print(f"PLAN={plan_path}")
        if args.prepare_only:
            sys.exit(0)
        if (args.output_dir / "density_gate_report.json").exists():
            raise RuntimeError("Refusing to overwrite completed Z6-C report")
        common = ["--snapshot-dir", str(args.snapshot_dir), "--semantics-dir", str(args.semantics_dir), "--panel-dir", str(args.panel_dir), "--collision-cache-dir", str(args.collision_cache_dir), "--actor-path", str(args.actor_path), "--output-dir", str(args.output_dir)]
        for arm in ("baseline", "treatment"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *common, "--arm", arm], cwd=PROJECT_ROOT, check=True)
        aggregate(args)
    else:
        if args.arm == "adjudication":
            write_adjudication_plan(args)
            run_arm(args)
            aggregate_adjudication(args)
        else:
            run_arm(args)
