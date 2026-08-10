import argparse
import copy
import io
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import End2Race
from ppo.env import CentralScheduleSubprocVecEnv
from ppo.env import load_prefix_reset_panel
from ppo.policy import BASELINE_EXPLORATION_MODE
from ppo.policy import PREFIX_JOINT_TEMPORAL_EXPLORATION_MODE
from ppo.scenarios import ScenarioSpec
from ppo.scenarios import ordinary_scenarios
from scripts.run_prefix_joint_temporal_e1 import training_arguments
from scripts.run_prefix_reset_density_gate import sha256_file
from scripts.run_prefix_reset_density_gate import source_snapshot
from scripts.run_prefix_reset_density_gate import state_digest
from train_ppo import START_METHOD
from train_ppo import build_model
from train_ppo import configure_training_numerics
from utils import TrainingRecorder
from utils import atomic_write_json


DEFAULT_OUTPUT = PROJECT_ROOT / "post-trained" / "ppo_prefix_reset_joint_temporal_rho0p90_gates" / "e2"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the prefix joint-temporal exploration E2 gate")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-dir", type=Path, default=Path("post-trained/panels/prefix_reset_consensus_v1"))
    parser.add_argument("--collision-cache-dir", type=Path, default=Path("post-trained/collision-cache/pretrained_end2race_austin_collision_pool_479"))
    parser.add_argument("--actor-path", type=Path, default=Path("pretrained/end2race.pth"))
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=12800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix-reset-interval", type=int, default=3)
    parser.add_argument("--hidden-scale", type=int, default=4)
    return parser.parse_args()


def nested_equal(left, right):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def optimizer_digest(optimizer):
    stream = io.BytesIO()
    torch.save(optimizer.state_dict(), stream)
    import hashlib
    return hashlib.sha256(stream.getvalue()).hexdigest()


def run_optimizer_path(model, mode, actor_rng_state, critic_rng_state):
    model.policy.speed_exploration_mode = mode
    model.policy.set_training_mode(True)
    actor_losses = []
    actor_gradient_norms = []
    approximate_kls = []
    clip_fractions = []
    actor_valid_counts = []
    critic_losses = []
    critic_gradient_norms = []
    critic_valid_counts = []
    actor_rng = np.random.default_rng()
    actor_rng.bit_generator.state = copy.deepcopy(actor_rng_state)
    critic_rng = np.random.default_rng()
    critic_rng.bit_generator.state = copy.deepcopy(critic_rng_state)

    for parameter in model.policy.critic_parameters:
        parameter.requires_grad_(False)
    for data in model.rollout_buffer.get(model.batch_size, rng=actor_rng):
        mask = data.mask > 1e-8
        advantages = data.advantages
        valid_advantages = advantages[mask]
        advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
        log_prob, _entropy = model.policy.evaluate_actor_actions(data.observations, data.actions, data.lstm_states, data.episode_starts, collection_equivalent=False)
        log_ratio = log_prob - data.old_log_prob
        ratio = torch.exp(log_ratio)
        loss = -torch.min(advantages * ratio, advantages * torch.clamp(ratio, 0.8, 1.2))[mask].mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("E2 actor loss is not finite")
        model.policy.actor_optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.policy.actor_parameters, 0.5)
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError("E2 actor gradient is not finite")
        model.policy.actor_optimizer.step()
        valid_log_ratio = log_ratio[mask].detach()
        actor_losses.append(float(loss.detach().cpu().item()))
        actor_gradient_norms.append(float(gradient_norm.detach().cpu().item()))
        approximate_kls.append(float(((torch.exp(valid_log_ratio) - 1.0) - valid_log_ratio).mean().cpu().item()))
        clip_fractions.append(float((torch.abs(torch.exp(valid_log_ratio) - 1.0) > 0.2).float().mean().cpu().item()))
        actor_valid_counts.append(int(mask.sum().item()))
    for parameter in model.policy.critic_parameters:
        parameter.requires_grad_(True)

    for parameter in model.policy.actor_parameters:
        parameter.requires_grad_(False)
    for data in model.rollout_buffer.get(model.batch_size, rng=critic_rng):
        mask = data.mask > 1e-8
        values = model._batch_values(data)
        value_loss = torch.nn.functional.mse_loss(values[mask], data.returns[mask])
        loss = 0.5 * value_loss
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("E2 critic loss is not finite")
        model.policy.critic_optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.policy.critic_parameters, 0.5)
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError("E2 critic gradient is not finite")
        model.policy.critic_optimizer.step()
        critic_losses.append(float(value_loss.detach().cpu().item()))
        critic_gradient_norms.append(float(gradient_norm.detach().cpu().item()))
        critic_valid_counts.append(int(mask.sum().item()))
    for parameter in model.policy.actor_parameters:
        parameter.requires_grad_(True)

    actor_checkpoint = model.policy.actor_checkpoint_state_dict()
    return {
        "mode": mode,
        "actor_optimizer_steps": len(actor_losses),
        "critic_optimizer_steps": len(critic_losses),
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "actor_gradient_norms": actor_gradient_norms,
        "critic_gradient_norms": critic_gradient_norms,
        "approximate_kls": approximate_kls,
        "clip_fractions": clip_fractions,
        "actor_valid_counts": actor_valid_counts,
        "critic_valid_counts": critic_valid_counts,
        "actor_state": copy.deepcopy(model.policy.end2race_actor.state_dict()),
        "critic_state": copy.deepcopy(model.policy.value_net.state_dict()),
        "actor_optimizer_state": copy.deepcopy(model.policy.actor_optimizer.state_dict()),
        "critic_optimizer_state": copy.deepcopy(model.policy.critic_optimizer.state_dict()),
        "actor_checkpoint": actor_checkpoint,
        "actor_state_digest": state_digest(model.policy.end2race_actor.state_dict()),
        "critic_state_digest": state_digest(model.policy.value_net.state_dict()),
        "actor_optimizer_digest": optimizer_digest(model.policy.actor_optimizer),
        "critic_optimizer_digest": optimizer_digest(model.policy.critic_optimizer),
    }


def load_actor_contract(path, hidden_scale):
    state = torch.load(path, map_location="cpu", weights_only=True)
    actor = End2Race(mask_prob=0.0, hidden_scale=hidden_scale)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    lidar = torch.linspace(0.01, 1.0, 360, dtype=torch.float32).reshape(1, 1, 360)
    speed = torch.tensor([[[3.0]]], dtype=torch.float32)
    hidden = torch.zeros(1, 1, actor.gru.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        action, next_hidden = actor(lidar, speed, hidden)
    return state, action, next_hidden


if __name__ == "__main__":
    args = parse_arguments()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.panel_dir = args.panel_dir.expanduser().resolve()
    args.collision_cache_dir = args.collision_cache_dir.expanduser().resolve()
    args.actor_path = args.actor_path.expanduser().resolve()
    if (args.n_envs, args.n_steps, args.batch_size, args.seed, args.prefix_reset_interval, args.hidden_scale) != (16, 6400, 12800, 42, 3, 4):
        raise ValueError("E2 fixed contract changed")
    e1_plan_path = args.output_dir.parent / "e1" / "e1_plan.json"
    e1_report_path = args.output_dir.parent / "e1" / "e1_report.json"
    e1_plan = json.loads(e1_plan_path.read_text(encoding="utf-8"))
    e1_report = json.loads(e1_report_path.read_text(encoding="utf-8"))
    if e1_report.get("verdict") not in ("pass", "pass_after_batched_replay_adjudication"):
        raise RuntimeError("E2 requires a passed E1 report")
    current_source = {
        "policy": sha256_file(PROJECT_ROOT / "ppo" / "policy.py"),
        "rollout": sha256_file(PROJECT_ROOT / "ppo" / "rollout.py"),
        "environment": sha256_file(PROJECT_ROOT / "ppo" / "env.py"),
        "train": sha256_file(PROJECT_ROOT / "train_ppo.py"),
    }
    if any(e1_plan["source_sha256"][name] != digest for name, digest in current_source.items()):
        raise RuntimeError("Training source changed after E1")
    report_path = args.output_dir / "e2_report.json"
    if report_path.exists():
        raise RuntimeError(f"Refusing to overwrite completed E2 report: {report_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "experiment_id": "ppo_prefix_reset_joint_temporal_rho0p90",
        "gate": "E2",
        "status": "frozen_before_execution",
        "fixed_contract": {
            "source_buffer": "one regenerated disabled baseline buffer under the exact E1 plan, frozen for both optimizer paths",
            "actor_epochs_per_path": 1,
            "critic_epochs_per_path": 1,
            "actor_minibatches_per_path": 8,
            "critic_minibatches_per_path": 8,
            "legacy_mode": BASELINE_EXPLORATION_MODE,
            "disabled_new_mode": PREFIX_JOINT_TEMPORAL_EXPLORATION_MODE,
            "joint_active_rows": 0,
        },
        "inputs": {
            "e1_plan": str(e1_plan_path),
            "e1_plan_sha256": sha256_file(e1_plan_path),
            "e1_report": str(e1_report_path),
            "e1_report_sha256": sha256_file(e1_report_path),
        },
        "source_sha256": {**current_source, "e2_script": sha256_file(Path(__file__).resolve())},
        "source_snapshot": source_snapshot(),
    }
    atomic_write_json(args.output_dir / "e2_plan.json", plan)

    collision_rows = json.loads((args.collision_cache_dir / "collision_scenarios.json").read_text(encoding="utf-8"))
    collision_scenarios = tuple(ScenarioSpec(**row) for row in collision_rows)
    ordinary = ordinary_scenarios("Austin")
    prefix_inputs = load_prefix_reset_panel(args.panel_dir)
    recorder = TrainingRecorder(args.output_dir / "source_buffer", args.hidden_scale)
    configure_training_numerics()
    vector_env = CentralScheduleSubprocVecEnv(args.n_envs, START_METHOD, args.seed, "Austin", collision_scenarios, ordinary, privileged=True, reward_gamma=0.999, speed_exploration_mode=BASELINE_EXPLORATION_MODE, prefix_reset_inputs=prefix_inputs, prefix_reset_interval=args.prefix_reset_interval)
    try:
        model = build_model(vector_env, training_arguments(args, args.output_dir / "source_buffer", BASELINE_EXPLORATION_MODE), torch.device("cuda"), recorder)
        collection_started = time.perf_counter()
        _total, callback = model._setup_learn(args.n_envs * args.n_steps, progress_bar=False)
        callback.on_training_start(locals(), globals())
        completed = model.collect_rollouts(vector_env, callback, model.rollout_buffer, args.n_steps)
        callback.on_training_end()
        collection_wall_seconds = time.perf_counter() - collection_started
        if not completed or not model.rollout_buffer.full:
            raise RuntimeError("E2 source buffer collection did not complete")
        if int(np.asarray(model.rollout_buffer.joint_temporal_active, dtype=bool).sum()) != 0:
            raise RuntimeError("E2 disabled source buffer contains joint-active rows")
        e1_baseline = json.loads((args.output_dir.parent / "e1" / "baseline" / "arm_report.json").read_text(encoding="utf-8"))
        reset_history_matches_e1 = vector_env.reset_history == e1_baseline["reset_history"]

        source_actor = copy.deepcopy(model.policy.end2race_actor.state_dict())
        source_critic = copy.deepcopy(model.policy.value_net.state_dict())
        source_actor_optimizer = copy.deepcopy(model.policy.actor_optimizer.state_dict())
        source_critic_optimizer = copy.deepcopy(model.policy.critic_optimizer.state_dict())
        actor_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0x45324143]))
        critic_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0x45324352]))
        actor_rng_state = copy.deepcopy(actor_rng.bit_generator.state)
        critic_rng_state = copy.deepcopy(critic_rng.bit_generator.state)

        legacy = run_optimizer_path(model, BASELINE_EXPLORATION_MODE, actor_rng_state, critic_rng_state)
        legacy_checkpoint_path = args.output_dir / "legacy_actor.pth"
        torch.save(legacy["actor_checkpoint"], legacy_checkpoint_path)
        model.policy.end2race_actor.load_state_dict(source_actor, strict=True)
        model.policy.value_net.load_state_dict(source_critic, strict=True)
        model.policy.actor_optimizer.load_state_dict(source_actor_optimizer)
        model.policy.critic_optimizer.load_state_dict(source_critic_optimizer)
        disabled = run_optimizer_path(model, PREFIX_JOINT_TEMPORAL_EXPLORATION_MODE, actor_rng_state, critic_rng_state)
        disabled_checkpoint_path = args.output_dir / "disabled_actor.pth"
        torch.save(disabled["actor_checkpoint"], disabled_checkpoint_path)

        legacy_keys, legacy_action, legacy_hidden = load_actor_contract(legacy_checkpoint_path, args.hidden_scale)
        disabled_keys, disabled_action, disabled_hidden = load_actor_contract(disabled_checkpoint_path, args.hidden_scale)
        criteria = {
            "source_buffer_complete_and_disabled": model.rollout_buffer.buffer_size * model.rollout_buffer.n_envs == 102400 and int(np.asarray(model.rollout_buffer.joint_temporal_active, dtype=bool).sum()) == 0,
            "regenerated_reset_history_matches_e1": reset_history_matches_e1,
            "optimizer_step_counts_match": legacy["actor_optimizer_steps"] == disabled["actor_optimizer_steps"] == 8 and legacy["critic_optimizer_steps"] == disabled["critic_optimizer_steps"] == 8,
            "minibatch_valid_counts_match": legacy["actor_valid_counts"] == disabled["actor_valid_counts"] and legacy["critic_valid_counts"] == disabled["critic_valid_counts"],
            "actor_losses_bitwise_match": legacy["actor_losses"] == disabled["actor_losses"],
            "critic_losses_bitwise_match": legacy["critic_losses"] == disabled["critic_losses"],
            "clip_and_kl_bitwise_match": legacy["clip_fractions"] == disabled["clip_fractions"] and legacy["approximate_kls"] == disabled["approximate_kls"],
            "gradient_norms_bitwise_match": legacy["actor_gradient_norms"] == disabled["actor_gradient_norms"] and legacy["critic_gradient_norms"] == disabled["critic_gradient_norms"],
            "actor_parameters_bitwise_match": nested_equal(legacy["actor_state"], disabled["actor_state"]),
            "critic_parameters_bitwise_match": nested_equal(legacy["critic_state"], disabled["critic_state"]),
            "actor_optimizer_state_bitwise_match": nested_equal(legacy["actor_optimizer_state"], disabled["actor_optimizer_state"]),
            "critic_optimizer_state_bitwise_match": nested_equal(legacy["critic_optimizer_state"], disabled["critic_optimizer_state"]),
            "strict_12_key_checkpoints": len(legacy_keys) == len(disabled_keys) == 12,
            "deterministic_actor_bitwise_match": torch.equal(legacy_action, disabled_action) and torch.equal(legacy_hidden, disabled_hidden),
            "all_metrics_finite": bool(np.isfinite(np.asarray(legacy["actor_losses"] + legacy["critic_losses"] + legacy["actor_gradient_norms"] + legacy["critic_gradient_norms"] + legacy["approximate_kls"] + disabled["actor_losses"] + disabled["critic_losses"] + disabled["actor_gradient_norms"] + disabled["critic_gradient_norms"] + disabled["approximate_kls"], dtype=np.float64)).all()),
        }
        report = {
            "schema_version": 1,
            "experiment_id": "ppo_prefix_reset_joint_temporal_rho0p90",
            "gate": "E2",
            "verdict": "pass" if all(criteria.values()) else "fail",
            "criteria": criteria,
            "source_buffer": {
                "transition_count": 102400,
                "collection_wall_seconds": collection_wall_seconds,
                "joint_active_count": 0,
                "reset_history_matches_e1": reset_history_matches_e1,
            },
            "legacy": {key: value for key, value in legacy.items() if key not in ("actor_state", "critic_state", "actor_optimizer_state", "critic_optimizer_state", "actor_checkpoint")},
            "disabled_new_path": {key: value for key, value in disabled.items() if key not in ("actor_state", "critic_state", "actor_optimizer_state", "critic_optimizer_state", "actor_checkpoint")},
            "checkpoint_artifacts": {
                "legacy": str(legacy_checkpoint_path),
                "legacy_sha256": sha256_file(legacy_checkpoint_path),
                "disabled": str(disabled_checkpoint_path),
                "disabled_sha256": sha256_file(disabled_checkpoint_path),
            },
            "evidence_boundary": {
                "established": "With one frozen disabled baseline buffer and identical initial model/optimizer states, the legacy and disabled-new paths do or do not produce bitwise-identical one-epoch actor and critic updates.",
                "not_established": "Treatment learning benefit or final actor performance.",
                "next_action": "Start the sole formal training run only if E0, E1, and E2 all pass.",
            },
        }
        atomic_write_json(report_path, report)
        print(json.dumps({"report": str(report_path), "verdict": report["verdict"], "criteria": criteria, "legacy_actor_digest": legacy["actor_state_digest"], "disabled_actor_digest": disabled["actor_state_digest"]}, indent=2), flush=True)
        if report["verdict"] != "pass":
            raise SystemExit(1)
    finally:
        vector_env.close()
