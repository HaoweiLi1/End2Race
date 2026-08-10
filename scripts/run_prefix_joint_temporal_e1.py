import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ppo.env import CentralScheduleSubprocVecEnv
from ppo.env import load_prefix_reset_panel
from ppo.policy import BASELINE_EXPLORATION_MODE
from ppo.policy import PREFIX_JOINT_TEMPORAL_EXPLORATION_MODE
from ppo.scenarios import ScenarioSpec
from ppo.scenarios import ordinary_scenarios
from scripts.run_prefix_reset_density_gate import manual_gae
from scripts.run_prefix_reset_density_gate import ratio_measure
from scripts.run_prefix_reset_density_gate import sha256_file
from scripts.run_prefix_reset_density_gate import source_snapshot
from scripts.run_prefix_reset_density_gate import state_digest
from train_ppo import START_METHOD
from train_ppo import build_model
from train_ppo import configure_training_numerics
from utils import TrainingRecorder
from utils import atomic_write_json


DEFAULT_OUTPUT = PROJECT_ROOT / "post-trained" / "ppo_prefix_reset_joint_temporal_rho0p90_gates" / "e1"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the prefix joint-temporal exploration E1 gate")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-dir", type=Path, default=Path("post-trained/panels/prefix_reset_consensus_v1"))
    parser.add_argument("--collision-cache-dir", type=Path, default=Path("post-trained/collision-cache/pretrained_end2race_austin_collision_pool_479"))
    parser.add_argument("--actor-path", type=Path, default=Path("pretrained/end2race.pth"))
    parser.add_argument("--arm", choices=("orchestrate", "baseline", "treatment"), default="orchestrate")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=12800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix-reset-interval", type=int, default=3)
    parser.add_argument("--hidden-scale", type=int, default=4)
    return parser.parse_args()


def training_arguments(args, arm_dir, mode):
    return argparse.Namespace(
        pretrained_model_path=str(args.actor_path), output_dir=str(arm_dir), hidden_scale=args.hidden_scale, critic="privilege_gru",
        map_name="Austin", n_envs=args.n_envs, seed=args.seed, collision_cache_dir=str(args.collision_cache_dir), prefix_reset_panel=str(args.panel_dir), prefix_reset_interval=args.prefix_reset_interval,
        n_steps=args.n_steps, batch_size=args.batch_size, num_updates=1, actor_epochs=2, critic_epochs=5, gru_learning_rate=3.0e-6, head_learning_rate=3.0e-5,
        critic_learning_rate=3.0e-4, steering_latent_std=0.03, speed_physical_std=0.15, speed_exploration_mode=mode,
        gamma=0.999, gae_lambda=0.995, clip_range=0.20,
    )


def dry_actor_two_epochs(model, seed):
    model.policy.set_training_mode(True)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x45314452]))
    minibatches = 0
    valid_total = 0
    loss_minimum = float("inf")
    loss_maximum = -float("inf")
    gradient_norm_minimum = float("inf")
    gradient_norm_maximum = 0.0
    for _epoch in range(2):
        for data in model.rollout_buffer.get(model.batch_size, rng=rng):
            mask = data.mask > 1e-8
            advantages = data.advantages
            valid_advantages = advantages[mask]
            advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
            log_prob, _entropy = model.policy.evaluate_actor_actions(data.observations, data.actions, data.lstm_states, data.episode_starts, collection_equivalent=False)
            ratio = torch.exp(log_prob - data.old_log_prob)
            loss = -torch.min(advantages * ratio, advantages * torch.clamp(ratio, 0.8, 1.2))[mask].mean()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("E1 dry actor loss is not finite")
            model.policy.actor_optimizer.zero_grad()
            loss.backward()
            gradient_squared = 0.0
            for parameter in model.policy.actor_parameters:
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item()):
                    raise RuntimeError("E1 dry actor gradient is missing or non-finite")
                gradient_squared += float(torch.sum(parameter.grad.detach().double() ** 2).cpu().item())
            gradient_norm = gradient_squared ** 0.5
            loss_value = float(loss.detach().cpu().item())
            loss_minimum = min(loss_minimum, loss_value)
            loss_maximum = max(loss_maximum, loss_value)
            gradient_norm_minimum = min(gradient_norm_minimum, gradient_norm)
            gradient_norm_maximum = max(gradient_norm_maximum, gradient_norm)
            minibatches += 1
            valid_total += int(mask.sum().item())
    model.policy.actor_optimizer.zero_grad()
    return {
        "planned_minibatches": 16,
        "completed_minibatches": minibatches,
        "valid_transition_visits": valid_total,
        "loss_minimum": loss_minimum,
        "loss_maximum": loss_maximum,
        "gradient_norm_minimum": gradient_norm_minimum,
        "gradient_norm_maximum": gradient_norm_maximum,
        "finite": bool(np.isfinite((loss_minimum, loss_maximum, gradient_norm_minimum, gradient_norm_maximum)).all()),
    }


def validate_block_sequences(buffer):
    active = np.asarray(buffer.joint_temporal_active, dtype=bool)
    uids = np.asarray(buffer.joint_temporal_block_uids, dtype=np.int64)
    positions = np.asarray(buffer.joint_temporal_block_positions, dtype=np.int64)
    errors = []
    incomplete_at_rollout_end = 0
    for uid in np.unique(uids[active]):
        rows = np.argwhere(active & (uids == uid))
        envs = np.unique(rows[:, 1])
        values = positions[active & (uids == uid)]
        if len(envs) != 1 or not np.array_equal(values, np.arange(len(values), dtype=np.int64)):
            errors.append({"uid": int(uid), "envs": envs.tolist(), "positions": values.tolist()})
        if len(values) < 50 and rows[-1, 0] == buffer.buffer_size - 1:
            incomplete_at_rollout_end += 1
    return {"error_count": len(errors), "errors": errors[:10], "incomplete_blocks_at_rollout_end": incomplete_at_rollout_end}


def build_plan(args):
    e0_path = args.output_dir.parent / "e0" / "e0_report.json"
    e0 = json.loads(e0_path.read_text(encoding="utf-8"))
    if e0.get("verdict") != "pass":
        raise RuntimeError("E1 requires a passed E0 report")
    prefix_inputs = load_prefix_reset_panel(args.panel_dir)
    strata = [item["stratum"] for item in prefix_inputs]
    if len(prefix_inputs) != 28 or strata.count("collision") != 19 or strata.count("lost_overtake") != 9:
        raise RuntimeError("E1 prefix panel contract changed")
    plan = {
        "schema_version": 1,
        "experiment_id": "ppo_prefix_reset_joint_temporal_rho0p90",
        "gate": "E1",
        "status": "frozen_before_execution",
        "fixed_contract": {
            "map": "Austin",
            "seed": args.seed,
            "actor": "canonical BC",
            "critic": "fresh privilege_gru",
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "transition_count_per_arm": args.n_envs * args.n_steps,
            "batch_size": args.batch_size,
            "actor_epochs_for_dry_gradient": 2,
            "gamma": 0.999,
            "gae_lambda": 0.995,
            "prefix_reset_interval": args.prefix_reset_interval,
            "prefix_task_count": 28,
            "collision_source_count": 19,
            "lost_overtake_source_count": 9,
            "arm_order": ["baseline", "treatment"],
            "baseline_mode": BASELINE_EXPLORATION_MODE,
            "treatment_mode": PREFIX_JOINT_TEMPORAL_EXPLORATION_MODE,
            "optimizer_steps": 0,
        },
        "admission_contract": {
            "role_transitions_each": 51200,
            "all_prefix_sources_required": 28,
            "all_collision_sources_treated": 19,
            "minimum_active_fraction": 0.02,
            "treatment_leak_count": 0,
            "gae_maximum_error": 1.0e-6,
            "exact_likelihood_maximum_error": 5.0e-5,
            "batched_likelihood_maximum_error": 0.02,
            "dry_actor_minibatches": 16,
            "action_identity_rows": 102400,
            "maximum_wall_ratio": 1.35,
        },
        "inputs": {
            "actor": str(args.actor_path),
            "actor_sha256": sha256_file(args.actor_path),
            "prefix_manifest": str(args.panel_dir / "prefix_reset_manifest.json"),
            "prefix_manifest_sha256": sha256_file(args.panel_dir / "prefix_reset_manifest.json"),
            "collision_scenarios": str(args.collision_cache_dir / "collision_scenarios.json"),
            "collision_scenarios_sha256": sha256_file(args.collision_cache_dir / "collision_scenarios.json"),
            "e0_report": str(e0_path),
            "e0_report_sha256": sha256_file(e0_path),
        },
        "source_sha256": {
            "preregistration": sha256_file(PROJECT_ROOT / ".agents" / "FINAL_PREFIX_LOCAL_JOINT_TEMPORAL_EXPLORATION_PREREGISTRATION.md"),
            "policy": sha256_file(PROJECT_ROOT / "ppo" / "policy.py"),
            "rollout": sha256_file(PROJECT_ROOT / "ppo" / "rollout.py"),
            "environment": sha256_file(PROJECT_ROOT / "ppo" / "env.py"),
            "train": sha256_file(PROJECT_ROOT / "train_ppo.py"),
            "e0_script": sha256_file(PROJECT_ROOT / "scripts" / "run_prefix_joint_temporal_e0.py"),
            "e1_script": sha256_file(Path(__file__).resolve()),
        },
        "source_snapshot": source_snapshot(),
    }
    return plan


def run_arm(args):
    plan = json.loads((args.output_dir / "e1_plan.json").read_text(encoding="utf-8"))
    mode = BASELINE_EXPLORATION_MODE if args.arm == "baseline" else PREFIX_JOINT_TEMPORAL_EXPLORATION_MODE
    if plan["fixed_contract"]["transition_count_per_arm"] != args.n_envs * args.n_steps or plan["fixed_contract"][f"{args.arm}_mode"] != mode:
        raise RuntimeError("E1 child contract changed after plan freeze")
    arm_dir = args.output_dir / args.arm
    report_path = arm_dir / "arm_report.json"
    if report_path.exists():
        raise RuntimeError(f"Refusing to overwrite E1 arm report: {report_path}")
    collision_rows = json.loads((args.collision_cache_dir / "collision_scenarios.json").read_text(encoding="utf-8"))
    collision_scenarios = tuple(ScenarioSpec(**row) for row in collision_rows)
    ordinary = ordinary_scenarios("Austin")
    prefix_inputs = load_prefix_reset_panel(args.panel_dir)
    prefix_strata = {item["episode_key"]: item["stratum"] for item in prefix_inputs}
    recorder = TrainingRecorder(arm_dir, args.hidden_scale)
    configure_training_numerics()
    vector_env = CentralScheduleSubprocVecEnv(args.n_envs, START_METHOD, args.seed, "Austin", collision_scenarios, ordinary, privileged=True, reward_gamma=0.999, speed_exploration_mode=mode, prefix_reset_inputs=prefix_inputs, prefix_reset_interval=args.prefix_reset_interval)
    try:
        model = build_model(vector_env, training_arguments(args, arm_dir, mode), torch.device("cuda"), recorder)
        actor_before = state_digest(model.policy.end2race_actor.state_dict())
        critic_before = state_digest(model.policy.value_net.state_dict())
        started = time.perf_counter()
        _total, callback = model._setup_learn(args.n_envs * args.n_steps, progress_bar=False)
        callback.on_training_start(locals(), globals())
        completed = model.collect_rollouts(vector_env, callback, model.rollout_buffer, args.n_steps)
        callback.on_training_end()
        collection_wall_seconds = time.perf_counter() - started
        if not completed or not model.rollout_buffer.full:
            raise RuntimeError(f"E1 {args.arm} rollout did not complete")
        actor_after_collection = state_digest(model.policy.end2race_actor.state_dict())
        critic_after_collection = state_digest(model.policy.value_net.state_dict())
        last_values = model.last_rollout_final_values
        dones = model.last_rollout_dones
        expected_advantages, expected_returns = manual_gae(model.rollout_buffer, last_values, dones, 0.999, 0.995)
        gae_advantage_error = float(np.max(np.abs(expected_advantages.astype(np.float64) - model.rollout_buffer.advantages.astype(np.float64))))
        gae_return_error = float(np.max(np.abs(expected_returns.astype(np.float64) - model.rollout_buffer.returns.astype(np.float64))))
        buffer_finite = all(np.isfinite(np.asarray(getattr(model.rollout_buffer, name))).all() for name in ("observations", "actions", "rewards", "values", "log_probs", "advantages", "returns", "joint_temporal_standard_residuals"))
        exploration = model._exploration_statistics()
        block_sequences = validate_block_sequences(model.rollout_buffer)
        reset_history = copy.deepcopy(vector_env.reset_history)
        prefix_resets = [row for row in reset_history if row["source"] == "prefix_reset"]
        prefix_reset_keys = sorted({str(row["prefix_reset_key"]) for row in prefix_resets})
        active = np.asarray(model.rollout_buffer.joint_temporal_active, dtype=bool)
        active_keys = sorted({str(model.last_prefix_key_rows[step][rank]) for step, rank in np.argwhere(active)})
        active_collision_keys = sorted(key for key in active_keys if prefix_strata.get(key) == "collision")
        active_non_collision_keys = sorted(key for key in active_keys if prefix_strata.get(key) != "collision")
        active_ordinary_count = int(active[:, 1::2].sum())
        active_outside_prefix_count = int(np.count_nonzero(active & ~np.asarray(model.last_prefix_transition_mask, dtype=bool)))
        active_outside_window_count = int(np.count_nonzero(active & ~np.asarray(model.last_prefix_window_mask, dtype=bool)))
        batched_ratio = ratio_measure(model, False, args.seed)
        exact_ratio = ratio_measure(model, True, args.seed)
        batched_adjudication = None
        model.policy.set_training_mode(True)
        if args.arm == "treatment" and max(batched_ratio["maximum_abs_log_ratio"], batched_ratio["maximum_abs_ratio_minus_one"]) >= 0.02 and max(exact_ratio["maximum_abs_log_ratio"], exact_ratio["maximum_abs_ratio_minus_one"]) <= 5.0e-5:
            batched_adjudication = model._adjudicate_batched_replay()
        dry_actor = dry_actor_two_epochs(model, args.seed)
        actor_after_dry = state_digest(model.policy.end2race_actor.state_dict())
        critic_after_dry = state_digest(model.policy.value_net.state_dict())
        report = {
            "schema_version": 1,
            "experiment_id": "ppo_prefix_reset_joint_temporal_rho0p90",
            "gate": "E1",
            "arm": args.arm,
            "mode": mode,
            "completed": True,
            "transition_count": args.n_envs * args.n_steps,
            "role_transition_counts": {"collision": args.n_envs * args.n_steps // 2, "ordinary": args.n_envs * args.n_steps // 2},
            "collection_wall_seconds": collection_wall_seconds,
            "transitions_per_second": args.n_envs * args.n_steps / collection_wall_seconds,
            "buffer_finite": bool(buffer_finite),
            "gae_advantage_max_abs_error": gae_advantage_error,
            "gae_return_max_abs_error": gae_return_error,
            "batched_ratio": batched_ratio,
            "collection_equivalent_ratio": exact_ratio,
            "batched_replay_adjudication": batched_adjudication,
            "dry_actor": dry_actor,
            "exploration": exploration,
            "block_sequences": block_sequences,
            "prefix_reset_count": len(prefix_resets),
            "prefix_reset_unique_keys": prefix_reset_keys,
            "prefix_reset_unique_key_count": len(prefix_reset_keys),
            "active_unique_keys": active_keys,
            "active_collision_source_keys": active_collision_keys,
            "active_collision_source_key_count": len(active_collision_keys),
            "active_non_collision_source_keys": active_non_collision_keys,
            "active_ordinary_transition_count": active_ordinary_count,
            "active_outside_prefix_transition_count": active_outside_prefix_count,
            "active_outside_150_step_window_count": active_outside_window_count,
            "prefix_transition_count": int(np.asarray(model.last_prefix_transition_mask, dtype=bool).sum()),
            "prefix_window_transition_count": int(np.asarray(model.last_prefix_window_mask, dtype=bool).sum()),
            "joint_sampler_generator_count": 0 if model.policy._joint_temporal_generators is None else len(model.policy._joint_temporal_generators),
            "actor_state_digest_before": actor_before,
            "actor_state_digest_after_collection": actor_after_collection,
            "actor_state_digest_after_dry": actor_after_dry,
            "critic_state_digest_before": critic_before,
            "critic_state_digest_after_collection": critic_after_collection,
            "critic_state_digest_after_dry": critic_after_dry,
            "parameters_unchanged": actor_before == actor_after_collection == actor_after_dry and critic_before == critic_after_collection == critic_after_dry,
            "optimizer_steps": 0,
            "reset_history": reset_history,
        }
        atomic_write_json(report_path, report)
        print(json.dumps({"arm": args.arm, "report": str(report_path), "wall_seconds": collection_wall_seconds, "active_fraction": exploration["joint_temporal_active_fraction"], "batched_ratio": batched_ratio, "exact_ratio": exact_ratio}, indent=2), flush=True)
    finally:
        vector_env.close()


def aggregate(args):
    baseline = json.loads((args.output_dir / "baseline" / "arm_report.json").read_text(encoding="utf-8"))
    treatment = json.loads((args.output_dir / "treatment" / "arm_report.json").read_text(encoding="utf-8"))
    total = args.n_envs * args.n_steps
    wall_ratio = treatment["collection_wall_seconds"] / baseline["collection_wall_seconds"]
    treatment_batched_maximum = max(treatment["batched_ratio"]["maximum_abs_log_ratio"], treatment["batched_ratio"]["maximum_abs_ratio_minus_one"])
    treatment_exact_maximum = max(treatment["collection_equivalent_ratio"]["maximum_abs_log_ratio"], treatment["collection_equivalent_ratio"]["maximum_abs_ratio_minus_one"])
    adjudication_required = treatment_batched_maximum >= 0.02
    adjudication_passed = not adjudication_required or treatment["batched_replay_adjudication"] is not None and treatment["batched_replay_adjudication"]["verdict"] == "pass"
    criteria = {
        "both_arms_complete_102400": baseline["transition_count"] == treatment["transition_count"] == total == 102400,
        "role_balance_exact": baseline["role_transition_counts"] == treatment["role_transition_counts"] == {"collision": 51200, "ordinary": 51200},
        "buffers_finite": baseline["buffer_finite"] and treatment["buffer_finite"],
        "baseline_disabled_no_joint_activity": baseline["exploration"]["joint_temporal_active_count"] == 0 and baseline["joint_sampler_generator_count"] == 0,
        "prefix_queue_all_28_both_arms": baseline["prefix_reset_unique_key_count"] == treatment["prefix_reset_unique_key_count"] == 28,
        "all_19_collision_sources_treated": treatment["active_collision_source_key_count"] == 19,
        "active_fraction_at_least_2_percent": treatment["exploration"]["joint_temporal_active_fraction"] >= 0.02,
        "no_treatment_leak": treatment["exploration"]["joint_temporal_treatment_leak_count"] == 0 and not treatment["active_non_collision_source_keys"] and treatment["active_ordinary_transition_count"] == 0 and treatment["active_outside_prefix_transition_count"] == 0 and treatment["active_outside_150_step_window_count"] == 0,
        "block_sequences_contiguous": treatment["block_sequences"]["error_count"] == 0,
        "gae_reference_within_1e6": max(baseline["gae_advantage_max_abs_error"], baseline["gae_return_max_abs_error"], treatment["gae_advantage_max_abs_error"], treatment["gae_return_max_abs_error"]) <= 1.0e-6,
        "exact_collection_replay_within_5e5": treatment_exact_maximum <= 5.0e-5,
        "batched_replay_pass_or_adjudicated": adjudication_passed,
        "action_identity_all_treatment_rows": treatment["exploration"]["joint_temporal_action_identity_checked_count"] == total,
        "parameters_unchanged_and_zero_steps": baseline["parameters_unchanged"] and treatment["parameters_unchanged"] and baseline["optimizer_steps"] == treatment["optimizer_steps"] == 0,
        "dry_actor_16_of_16_finite": baseline["dry_actor"]["completed_minibatches"] == treatment["dry_actor"]["completed_minibatches"] == 16 and baseline["dry_actor"]["finite"] and treatment["dry_actor"]["finite"],
        "wall_ratio_at_most_1p35": wall_ratio <= 1.35,
    }
    verdict = "pass_after_batched_replay_adjudication" if all(criteria.values()) and adjudication_required else "pass" if all(criteria.values()) else "fail"
    report = {
        "schema_version": 1,
        "experiment_id": "ppo_prefix_reset_joint_temporal_rho0p90",
        "gate": "E1",
        "verdict": verdict,
        "criteria": criteria,
        "summary": {
            "transition_count_per_arm": total,
            "baseline_wall_seconds": baseline["collection_wall_seconds"],
            "treatment_wall_seconds": treatment["collection_wall_seconds"],
            "treatment_over_baseline_wall_ratio": wall_ratio,
            "prefix_unique_keys_baseline": baseline["prefix_reset_unique_key_count"],
            "prefix_unique_keys_treatment": treatment["prefix_reset_unique_key_count"],
            "treated_collision_source_count": treatment["active_collision_source_key_count"],
            "joint_temporal_active_count": treatment["exploration"]["joint_temporal_active_count"],
            "joint_temporal_active_fraction": treatment["exploration"]["joint_temporal_active_fraction"],
            "joint_temporal_block_count": treatment["exploration"]["joint_temporal_block_count"],
            "treatment_leak_count": treatment["exploration"]["joint_temporal_treatment_leak_count"],
            "action_identity_checked_count": treatment["exploration"]["joint_temporal_action_identity_checked_count"],
            "batched_maximum_error": treatment_batched_maximum,
            "exact_maximum_error": treatment_exact_maximum,
            "batched_adjudication_required": adjudication_required,
            "batched_adjudication": treatment["batched_replay_adjudication"],
            "gae_maximum_error": max(baseline["gae_advantage_max_abs_error"], baseline["gae_return_max_abs_error"], treatment["gae_advantage_max_abs_error"], treatment["gae_return_max_abs_error"]),
            "dry_actor_completed_minibatches_per_arm": {"baseline": baseline["dry_actor"]["completed_minibatches"], "treatment": treatment["dry_actor"]["completed_minibatches"]},
        },
        "evidence_boundary": {
            "established": "The frozen 102400-transition no-update integration contract, exact correlated likelihood, treatment exposure, leakage, GAE, action identity, dry gradients, and engineering overhead pass or fail.",
            "not_established": "Any actor learning benefit, Austin improvement, cross-map generalization, or final 600-episode performance.",
            "next_action": "Run E2 only if every E1 criterion passes.",
        },
    }
    atomic_write_json(args.output_dir / "e1_report.json", report)
    print(json.dumps({"report": str(args.output_dir / "e1_report.json"), "verdict": verdict, "criteria": criteria, "summary": report["summary"]}, indent=2), flush=True)
    if verdict == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    args = parse_arguments()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.panel_dir = args.panel_dir.expanduser().resolve()
    args.collision_cache_dir = args.collision_cache_dir.expanduser().resolve()
    args.actor_path = args.actor_path.expanduser().resolve()
    if (args.n_envs, args.n_steps, args.batch_size, args.seed, args.prefix_reset_interval, args.hidden_scale) != (16, 6400, 12800, 42, 3, 4):
        raise ValueError("E1 fixed contract changed")
    if args.arm == "orchestrate":
        report_path = args.output_dir / "e1_report.json"
        if report_path.exists():
            raise RuntimeError(f"Refusing to overwrite completed E1 report: {report_path}")
        plan = build_plan(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plan_path = args.output_dir / "e1_plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
                raise RuntimeError("Existing E1 plan differs from current frozen inputs")
        else:
            atomic_write_json(plan_path, plan)
        common = ["--output-dir", str(args.output_dir), "--panel-dir", str(args.panel_dir), "--collision-cache-dir", str(args.collision_cache_dir), "--actor-path", str(args.actor_path)]
        for arm in ("baseline", "treatment"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *common, "--arm", arm], cwd=PROJECT_ROOT, check=True)
        aggregate(args)
    else:
        run_arm(args)
