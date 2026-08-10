import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo.collision_anchor import CollisionBCAnchor
from ppo.env import CentralScheduleSubprocVecEnv
from ppo.policy import CORRIDOR_TEMPORAL_EXPLORATION_MODE, End2RaceGRUPolicy
from ppo.rollout import End2RaceRecurrentPPO, MAX_GRAD_NORM, VALUE_LOSS_COEFFICIENT
from ppo.scenarios import ScenarioSpec, ordinary_scenarios
from train_ppo import START_METHOD, configure_training_numerics
from utils import TrainingRecorder, atomic_write_json


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dataset", type=Path, default=Path("post-trained/panels/collision_bc_anchor_v1"))
    parser.add_argument("--gate-dir", type=Path, default=Path("eval_results/collision_bc_anchor_training_gate_v1"))
    parser.add_argument("--u44-actor", type=Path, default=Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update44/actor.pth"))
    parser.add_argument("--u44-critic", type=Path, default=Path("post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update44/critic.pt"))
    parser.add_argument("--collision-cache-dir", type=Path, default=Path("post-trained/collision-cache/pretrained_end2race_austin_collision_pool_479"))
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def source_snapshot():
    head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(("git", "status", "--short"), cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    return {"git_head": head, "git_status_short": status}


def actor_loss(model, data):
    mask = data.mask > 1e-8
    advantages = data.advantages
    valid = advantages[mask]
    advantages = (advantages - valid.mean()) / (valid.std() + 1e-8)
    log_prob, _entropy = model.policy.evaluate_actor_actions(data.observations, data.actions, data.lstm_states, data.episode_starts)
    ratio = torch.exp(log_prob - data.old_log_prob)
    return -torch.min(advantages * ratio, advantages * torch.clamp(ratio, 0.80, 1.20))[mask].mean()


if __name__ == "__main__":
    args = parse_arguments()
    configure_training_numerics()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for collision BC anchor beta calibration")
    anchor_dataset = args.anchor_dataset.expanduser().resolve()
    gate_dir = args.gate_dir.expanduser().resolve()
    u44_actor = args.u44_actor.expanduser().resolve()
    u44_critic = args.u44_critic.expanduser().resolve()
    collision_cache_dir = args.collision_cache_dir.expanduser().resolve()
    required = (anchor_dataset / "manifest.json", u44_actor, u44_critic, collision_cache_dir / "collision_scenarios.json")
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Collision BC anchor beta-calibration inputs are incomplete")
    if gate_dir.exists() and any(gate_dir.iterdir()):
        raise RuntimeError(f"Collision BC anchor calibration directory must be empty: {gate_dir}")
    gate_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "experiment_id": "collision_bc_anchor_training_gate_v1",
        "status": "frozen_before_rollout",
        "anchor_dataset": str(anchor_dataset),
        "anchor_manifest_sha256": sha256_file(anchor_dataset / "manifest.json"),
        "u44_actor": str(u44_actor),
        "u44_actor_sha256": sha256_file(u44_actor),
        "u44_critic": str(u44_critic),
        "u44_critic_sha256": sha256_file(u44_critic),
        "collision_pool_sha256": sha256_file(collision_cache_dir / "collision_scenarios.json"),
        "rollout": {"map": "Austin", "seed": 42, "n_envs": 16, "n_steps": 6400, "batch_size": 12800, "exploration": "corridor_temporal"},
        "beta_formula": "0.25 * median(PPO step-space norm / anchor step-space norm)",
        "source_snapshot": source_snapshot(),
    }
    atomic_write_json(gate_dir / "gate_plan.json", plan)

    collision_rows = json.loads((collision_cache_dir / "collision_scenarios.json").read_text(encoding="utf-8"))
    collisions = tuple(ScenarioSpec(**row) for row in collision_rows)
    ordinary = ordinary_scenarios("Austin")
    if len(collisions) != 479 or len(ordinary) != 600:
        raise RuntimeError("Collision BC anchor calibration pool counts changed")
    recorder = TrainingRecorder(gate_dir / "rollout", 4)
    recorder.write_scenario_pools(collisions, ordinary, {"mode": "frozen canonical BC Austin collision pool", "collision_count": len(collisions)})
    vector_env = CentralScheduleSubprocVecEnv(16, START_METHOD, 42, "Austin", collisions, ordinary, privileged=True, reward_gamma=0.999, speed_exploration_mode=CORRIDOR_TEMPORAL_EXPLORATION_MODE)
    try:
        model = End2RaceRecurrentPPO(
            End2RaceGRUPolicy,
            vector_env,
            actor_epochs=2,
            critic_epochs=5,
            recorder=recorder,
            collision_bc_anchor_dataset="",
            collision_bc_anchor_beta=0.0,
            learning_rate=1.0,
            n_steps=6400,
            batch_size=12800,
            gamma=0.999,
            gae_lambda=0.995,
            clip_range=0.20,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=VALUE_LOSS_COEFFICIENT,
            max_grad_norm=MAX_GRAD_NORM,
            seed=42,
            device=torch.device("cuda"),
            policy_kwargs={
                "checkpoint_path": str(u44_actor),
                "hidden_scale": 4,
                "critic_variant": "privilege_gru",
                "gru_learning_rate": 3.0e-6,
                "head_learning_rate": 3.0e-5,
                "critic_learning_rate": 3.0e-4,
                "steering_latent_std": 0.03,
                "speed_physical_std": 0.15,
                "speed_exploration_mode": CORRIDOR_TEMPORAL_EXPLORATION_MODE,
            },
            verbose=1,
        )
        critic_state = torch.load(u44_critic, map_location="cuda", weights_only=True)
        model.policy.value_net.load_state_dict(critic_state, strict=True)
        actor_before = tensor_state(model.policy.actor_checkpoint_state_dict())
        _total, callback = model._setup_learn(16 * 6400, progress_bar=False)
        callback.on_training_start(locals(), globals())
        completed = model.collect_rollouts(vector_env, callback, model.rollout_buffer, 6400)
        callback.on_training_end()
        if not completed or not model.rollout_buffer.full:
            raise RuntimeError("Collision BC anchor U44 proxy rollout did not complete")
        if not all(torch.equal(actor_before[name], model.policy.actor_checkpoint_state_dict()[name].detach().cpu()) for name in actor_before):
            raise RuntimeError("U44 actor changed during the no-update proxy rollout")

        anchor = CollisionBCAnchor(anchor_dataset, model.policy, model.device)
        canonical_error = None
        canonical_loss = None
        current_actor = tensor_state(model.policy.actor_checkpoint_state_dict())
        model.policy.end2race_actor.load_state_dict(torch.load(PROJECT_ROOT / "pretrained" / "end2race.pth", map_location="cuda", weights_only=True), strict=True)
        canonical_error = anchor.maximum_action_error()
        with torch.no_grad():
            canonical_loss = float(anchor.loss()[0].detach().cpu().item())
        model.policy.end2race_actor.load_state_dict(current_actor, strict=True)
        model.policy.set_training_mode(True)
        model.collision_bc_anchor = anchor

        anchor_total, anchor_steering, anchor_speed = anchor.loss()
        anchor_gradients = torch.autograd.grad(anchor_total, model.policy.actor_parameters)
        anchor_step_norm = model._actor_step_space_norm(anchor_gradients)

        rng = np.random.default_rng(np.random.SeedSequence([42, 0x414E4348]))
        ppo_step_norms = []
        anchor_step_norms = []
        ppo_losses = []
        for data in model.rollout_buffer.get(12800, rng=rng):
            loss = actor_loss(model, data)
            gradients = torch.autograd.grad(loss, model.policy.actor_parameters)
            ppo_losses.append(float(loss.detach().cpu().item()))
            ppo_step_norms.append(model._actor_step_space_norm(gradients))
            anchor_step_norms.append(anchor_step_norm)
        if len(ppo_step_norms) != 8 or anchor_step_norm <= 0.0:
            raise RuntimeError("Collision BC anchor beta calibration minibatches are incomplete")
        beta = 0.25 * float(np.median(np.asarray(ppo_step_norms, dtype=np.float64) / np.asarray(anchor_step_norms, dtype=np.float64)))
        strict_actor = model.policy.actor_checkpoint_state_dict()

        criteria = {
            "dataset_18_unique_150_steps": len(anchor.episodes) == 18 and all(int(episode["mask"].sum().item()) == 150 for episode in anchor.episodes),
            "canonical_action_error_at_most_1e_5": canonical_error <= 1.0e-5,
            "canonical_anchor_loss_at_most_1e_10": canonical_loss <= 1.0e-10,
            "u44_anchor_gradient_finite_nonzero": math.isfinite(anchor_step_norm) and anchor_step_norm > 0.0,
            "beta_unique_finite_positive": math.isfinite(beta) and beta > 0.0,
            "strict_12_key_actor": len(strict_actor) == 12 and all(torch.isfinite(value).all() for value in strict_actor.values()),
            "u44_rollout_full": model.rollout_buffer.buffer_size * model.rollout_buffer.n_envs == 102400,
        }
        verdict = "calibration_complete" if all(criteria.values()) else "calibration_invalid"
        report = {
            "schema_version": 1,
            "experiment_id": "collision_bc_anchor_training_gate_v1",
            "verdict": verdict,
            "criteria": criteria,
            "dataset": {"episode_count": len(anchor.episodes), "anchor_step_count": sum(int(episode["mask"].sum().item()) for episode in anchor.episodes)},
            "canonical": {"maximum_action_error": canonical_error, "anchor_loss": canonical_loss},
            "u44_proxy": {"anchor_loss": float(anchor_total.detach().cpu().item()), "anchor_steering_loss": float(anchor_steering.detach().cpu().item()), "anchor_speed_loss": float(anchor_speed.detach().cpu().item()), "anchor_step_space_norm": anchor_step_norm},
            "beta": beta,
            "minibatches": {"ppo_losses": ppo_losses, "ppo_step_space_norms": ppo_step_norms, "anchor_step_space_norms": anchor_step_norms},
            "actor_unchanged_during_rollout": all(torch.equal(actor_before[name], current_actor[name]) for name in actor_before),
            "next_action": "Use the single reported beta in formal training" if verdict == "calibration_complete" else "Fix only the invalid calibration contract",
        }
        atomic_write_json(gate_dir / "gate_report.json", report)
        print(json.dumps({"verdict": verdict, "criteria": criteria, "beta": beta, "canonical": report["canonical"], "u44_proxy": report["u44_proxy"]}, indent=2))
    finally:
        vector_env.close()
