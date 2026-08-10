import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from latticeplanner.utils import load_config
from ppo.env import EXTERNAL_RESET_OPTION, CentralScheduleSubprocVecEnv, make_environment
from ppo.policy import End2RaceGRUPolicy
from ppo.rollout import FirstActionPreferenceDataset, End2RaceRecurrentPPO
from ppo.scenarios import ScenarioSpec, ordinary_scenarios
from train_ppo import configure_training_numerics
from utils import TrainingRecorder

CONFIG = load_config("ppo/ppo_config.yaml")


def compare_transition(left, right):
    left_observation, left_reward, left_terminated, left_truncated, left_info = left
    right_observation, right_reward, right_terminated, right_truncated, right_info = right
    if not np.array_equal(left_observation, right_observation) or left_reward != right_reward or left_terminated != right_terminated or left_truncated != right_truncated:
        raise RuntimeError("Runtime snapshot did not reproduce the next transition exactly")
    for name in ("ego_collision", "opponent_collision", "elapsed_time", "episode_return", "episode_steps", "episode_outcome", "termination_reason"):
        if left_info[name] != right_info[name]:
            raise RuntimeError(f"Runtime snapshot transition info changed: {name}")


def snapshot_test():
    environment = make_environment(42, "Austin", privileged=True, reward_gamma=0.999, speed_exploration_mode="baseline")()
    try:
        scenario = ordinary_scenarios("Austin")[0]
        observation, _info = environment.reset(seed=42, options={EXTERNAL_RESET_OPTION: scenario.to_reset_spec("ordinary")})
        action = np.asarray((0.01, 3.0), dtype=np.float32)
        for _step in range(5):
            observation, _reward, terminated, truncated, _info = environment.step(action)
            if terminated or truncated:
                raise RuntimeError("Runtime snapshot smoke scenario terminated before capture")
        snapshot = environment.capture_runtime_snapshot()
        first = environment.step(action)
        restored = environment.restore_runtime_snapshot(snapshot)
        if not np.array_equal(restored, observation):
            raise RuntimeError("Runtime snapshot did not restore the capture observation")
        second = environment.step(action)
        compare_transition(first, second)
        return {"observation_size": int(observation.size), "exact_next_transition": True}
    finally:
        environment.close()


def branch_test():
    collision_rows = json.loads((PROJECT_ROOT / "post-trained/collision-cache/pretrained_end2race_austin_collision_pool_479/collision_scenarios.json").read_text(encoding="utf-8"))
    collisions = tuple(ScenarioSpec(**row) for row in collision_rows)
    ordinary = ordinary_scenarios("Austin")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vector_env = CentralScheduleSubprocVecEnv(16, CONFIG.start_method, 42, "Austin", collisions, ordinary, privileged=True, reward_gamma=0.999, speed_exploration_mode="baseline")
    try:
        with tempfile.TemporaryDirectory(prefix="online_branch_ppo_test_") as output_dir:
            recorder = TrainingRecorder(output_dir, 4)
            model = End2RaceRecurrentPPO(
                End2RaceGRUPolicy,
                vector_env,
                actor_epochs=2,
                critic_epochs=1,
                recorder=recorder,
                online_same_state_branch_ppo=True,
                learning_rate=1.0,
                n_steps=16,
                batch_size=256,
                gamma=0.999,
                gae_lambda=0.995,
                clip_range=0.20,
                clip_range_vf=None,
                normalize_advantage=True,
                ent_coef=0.0,
                vf_coef=CONFIG.value_loss_coefficient,
                max_grad_norm=CONFIG.max_grad_norm,
                seed=42,
                device=device,
                policy_kwargs={
                    "checkpoint_path": str(PROJECT_ROOT / "pretrained/end2race.pth"),
                    "hidden_scale": 4,
                    "critic_variant": "privilege_gru",
                    "gru_learning_rate": 3.0e-6,
                    "head_learning_rate": 3.0e-5,
                    "critic_learning_rate": 3.0e-4,
                    "steering_latent_std": 0.03,
                    "speed_physical_std": 0.15,
                    "speed_exploration_mode": "baseline",
                },
                verbose=0,
            )
            preference = FirstActionPreferenceDataset(PROJECT_ROOT / "post-trained/panels/first_action_preference_v1", model.policy, device, 42)
            preference_loss, target_loss, control_loss, preference_margins = preference.loss()
            preference_values = [float(value.detach().cpu().item()) for value in (preference_loss, target_loss, control_loss)]
            if not np.isfinite(preference_values).all() or len(preference.target_indices) != 46 or len(preference.control_indices) != 19:
                raise RuntimeError("Merged first-action preference dataset or loss contract failed")
            _total, callback = model._setup_learn(16 * 16, progress_bar=False)
            callback.on_training_start(locals(), globals())
            observations_before = np.asarray(model._last_obs, dtype=np.float32).copy()
            actor_before = {name: value.detach().cpu().clone() for name, value in model.policy.actor_checkpoint_state_dict().items()}
            model.online_branch_rollout.reset()
            for rank in range(16):
                model._collect_online_branch_group(vector_env, rank, model._last_obs[rank], model._last_lstm_states, bool(model._last_episode_starts[rank]), rank)
            model.online_branch_rollout.finalize()
            for rank in range(16):
                restored = vector_env.env_method("restore_runtime_snapshot", vector_env.env_method("capture_runtime_snapshot", indices=[rank])[0], indices=[rank])[0]
                if not np.array_equal(restored, observations_before[rank]):
                    raise RuntimeError(f"Online branch collection changed main observation for rank {rank}")
            ratio_identity = model.online_branch_rollout.ratio_identity(model.policy)
            if ratio_identity > 5e-5:
                raise RuntimeError(f"Online branch old/new log-prob identity failed: {ratio_identity}")
            model.policy.actor_optimizer.zero_grad()
            indices = np.arange(64, dtype=np.int64)
            loss, approximate_kl, clip_fraction = model.online_branch_rollout.loss(model.policy, indices, 0.20)
            (CONFIG.online_branch_loss_coefficient * loss).backward()
            gradient_norm = float(torch.sqrt(sum(torch.sum(parameter.grad.detach().double().square()) for parameter in model.policy.actor_parameters if parameter.grad is not None)).cpu().item())
            if not np.isfinite(gradient_norm) or gradient_norm <= 0.0:
                raise RuntimeError("Online branch PPO actor gradient must be finite and nonzero")
            actor_after = model.policy.actor_checkpoint_state_dict()
            if any(not torch.equal(actor_before[name], actor_after[name].detach().cpu()) for name in actor_before):
                raise RuntimeError("Online branch mechanical test changed actor parameters without an optimizer step")
            model.policy.actor_optimizer.zero_grad()
            model.warmup_completed = True
            completed = model.collect_rollouts(vector_env, callback, model.rollout_buffer, 16)
            if not completed:
                raise RuntimeError("Online branch lifecycle rollout stopped early")
            model.train()
            metrics_rows = [json.loads(line) for line in (Path(output_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            if len(metrics_rows) != 1 or metrics_rows[0].get("online_branch_action_count") != 64:
                raise RuntimeError("Online branch lifecycle metrics contract failed")
            lifecycle_actor_changed = any(not torch.equal(actor_before[name], model.policy.actor_checkpoint_state_dict()[name].detach().cpu()) for name in actor_before)
            if not lifecycle_actor_changed or model._n_updates != 1:
                raise RuntimeError("Online branch lifecycle did not complete one actor update")
            callback.on_training_end()
            return {
                "device": str(device),
                "preupdate_max_abs_log_ratio": ratio_identity,
                "branch_loss": float(loss.detach().cpu().item()),
                "branch_approximate_kl": approximate_kl,
                "branch_clip_fraction": clip_fraction,
                "branch_gradient_norm": gradient_norm,
                "lifecycle_update_completed": True,
                "lifecycle_actor_changed": lifecycle_actor_changed,
                "lifecycle_metrics_branch_action_count": metrics_rows[0]["online_branch_action_count"],
                "first_action_preference_loss": preference_values[0],
                "first_action_preference_target_loss": preference_values[1],
                "first_action_preference_control_loss": preference_values[2],
                "first_action_preference_margin_count": int(preference_margins.numel()),
                **model.online_branch_rollout.statistics(),
            }
    finally:
        vector_env.close()


if __name__ == "__main__":
    configure_training_numerics()
    torch.manual_seed(42)
    np.random.seed(42)
    print(json.dumps({"snapshot": snapshot_test(), "online_branch": branch_test()}, indent=2, sort_keys=True))
