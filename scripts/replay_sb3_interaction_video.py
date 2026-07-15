#!/usr/bin/env python3
"""Generate a camera-corrected visual replay without any PPO update.

The historical training MP4 is intentionally preserved.  This script loads
the policy and RNG state captured immediately before that rollout, runs the
same fixed Austin scenario through the normal policy/environment action path,
and writes a separately named visual replay.  It never calls collect_rollouts,
train, learn, backward, or optimizer.step.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3.common.utils import obs_as_tensor

from scripts.smoke_sb3_nonzero_update import (
    ARTIFACT_DIR,
    BATCH_SIZE,
    EVALUATOR_STEER_BOUND,
    N_STEPS,
    PRE_UPDATE_SNAPSHOT_PATH,
    RESULTS_PATH,
    SEED,
    clone_state_dict,
    make_model,
    make_real_training_env,
    sha256_file,
    state_delta,
    verify_video,
)


OUTPUT_VIDEO = ARTIFACT_DIR / "interaction_replay.mp4"
OUTPUT_RESULTS = ARTIFACT_DIR / "interaction_replay_results.json"


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def main() -> int:
    if OUTPUT_VIDEO.exists() or OUTPUT_RESULTS.exists():
        raise RuntimeError("Refusing to overwrite an existing interaction replay artifact")
    if not PRE_UPDATE_SNAPSHOT_PATH.exists() or not RESULTS_PATH.exists():
        raise FileNotFoundError("The original pre-update snapshot and results.json are required")

    original_results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    original_video_path = ROOT / original_results["video"]["path"]
    original_video_hash_before = sha256_file(original_video_path)
    snapshot = torch.load(PRE_UPDATE_SNAPSHOT_PATH, map_location="cpu", weights_only=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vector_env, diagnostic_env, integration_env, core, _opponent, scenario = make_real_training_env(
        video_path=OUTPUT_VIDEO
    )
    model = make_model(vector_env, device)
    try:
        model._setup_model()
        incompatible = model.policy.load_state_dict(snapshot["policy_state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise AssertionError(f"Policy snapshot strict load failed: {incompatible}")
        policy_before = clone_state_dict(model.policy)
        optimizer_state_before = len(model.policy.optimizer.state)

        observation = vector_env.reset()
        render_preflight = diagnostic_env.render_preflight()
        restore_rng_state(snapshot["rng_state"])
        recurrent_states = model._last_lstm_states
        episode_starts = np.ones((1,), dtype=bool)
        model.policy.set_training_mode(False)
        diagnostic_env.start_recording()

        actions: list[np.ndarray] = []
        clipped_actions: list[np.ndarray] = []
        log_probabilities: list[float] = []
        values: list[float] = []
        done_step = None
        for step in range(N_STEPS):
            with torch.no_grad():
                action_tensor, value_tensor, log_prob_tensor, recurrent_states = model.policy.forward(
                    obs_as_tensor(observation, model.device),
                    recurrent_states,
                    torch.as_tensor(episode_starts, dtype=torch.float32, device=model.device),
                    deterministic=False,
                )
            action = action_tensor.detach().cpu().numpy()
            clipped = np.clip(action, model.action_space.low, model.action_space.high)
            next_observation, _reward, dones, _infos = vector_env.step(clipped)
            actions.append(action[0].copy())
            clipped_actions.append(clipped[0].copy())
            log_probabilities.append(float(log_prob_tensor.detach().cpu()[0]))
            values.append(float(value_tensor.detach().cpu()[0]))
            observation = next_observation
            episode_starts = dones
            if bool(dones[0]):
                done_step = step + 1
                break

        diagnostic_env.close_video()
        video = verify_video(diagnostic_env)
        video["source"] = (
            "camera-corrected visual replay from the saved pre-update policy/RNG snapshot; "
            "not the historical training rollout"
        )

        policy_after = clone_state_dict(model.policy)
        policy_delta = state_delta(policy_before, policy_after, policy_before)
        optimizer_state_after = len(model.policy.optimizer.state)
        action_array = np.asarray(actions)
        clipped_array = np.asarray(clipped_actions)
        core_ego_actions = np.asarray(core.received_joint_actions)[:, 0, :]
        original_first_action = np.asarray(
            original_results["rollout"]["first_action_trace"]["core_ego_action"],
            dtype=np.float32,
        )
        first_action_error = float(np.abs(action_array[0] - original_first_action).max())
        bufferless_core_error = float(np.abs(clipped_array - core_ego_actions).max())
        original_video_hash_after = sha256_file(original_video_path)

        checks = {
            "snapshot_policy_strict_load": not incompatible.missing_keys and not incompatible.unexpected_keys,
            "one_hundred_real_steps": core.step_count == N_STEPS,
            "timeout_at_step_100": done_step == N_STEPS,
            "ego_collision_absent": not any(record["ego_collision"] for record in diagnostic_env.reward_records),
            "action_path_to_core_exact": bufferless_core_error <= 1e-7,
            "historical_first_action_reproduced": first_action_error <= 1e-7,
            "steering_bounded": bool(np.all(np.abs(action_array[:, 0]) <= EVALUATOR_STEER_BOUND + 1e-7)),
            "actions_logp_values_finite": bool(
                np.isfinite(action_array).all()
                and np.isfinite(log_probabilities).all()
                and np.isfinite(values).all()
            ),
            "render_preflight_ego_visible": (
                render_preflight["vehicle_visibility"]["ego_center_yellow_pixels"] >= 8
            ),
            "render_preflight_opponent_visible": (
                render_preflight["vehicle_visibility"]["opponent_red_pixels"] >= 8
            ),
            "raw_ego_visible_every_frame": (
                video["raw_vehicle_visibility"]["ego_center_yellow_pixels_min"] >= 8
            ),
            "raw_opponent_visible_every_frame": (
                video["raw_vehicle_visibility"]["opponent_red_pixels_min"] >= 8
            ),
            "decoded_ego_visible_every_frame": (
                video["decoded_vehicle_visibility"]["ego_center_yellow_pixels_min"] >= 8
            ),
            "decoded_opponent_visible_every_frame": (
                video["decoded_vehicle_visibility"]["opponent_red_pixels_min"] >= 8
            ),
            "video_has_100_frames": video["decoded_frame_count"] == N_STEPS,
            "video_is_100_fps": video["fps"] == 100.0,
            "no_policy_parameter_change": policy_delta == 0.0,
            "optimizer_state_remains_empty": optimizer_state_before == optimizer_state_after == 0,
            "historical_training_video_preserved": (
                original_video_hash_before == original_video_hash_after == original_results["video"]["sha256"]
            ),
        }
        result = {
            "verdict": "PASS_VISUAL_REPLAY" if all(checks.values()) else "FAIL",
            "provenance": {
                "is_historical_training_rollout": False,
                "is_snapshot_visual_replay": True,
                "optimizer_updates": 0,
                "collect_rollouts_calls": 0,
                "train_calls": 0,
                "learn_calls": 0,
                "snapshot": str(PRE_UPDATE_SNAPSHOT_PATH.relative_to(ROOT)),
                "seed": SEED,
            },
            "scenario": scenario,
            "configuration": {
                "device": str(model.device),
                "steps": N_STEPS,
                "batch_size_not_used": BATCH_SIZE,
            },
            "render_preflight": render_preflight,
            "interaction": {
                "real_environment_steps": core.step_count,
                "done_step": done_step,
                "ego_collision": any(record["ego_collision"] for record in diagnostic_env.reward_records),
                "first_action": action_array[0],
                "historical_training_first_action": original_first_action,
                "historical_first_action_max_error": first_action_error,
                "action_to_core_max_error": bufferless_core_error,
                "steering_range": [float(action_array[:, 0].min()), float(action_array[:, 0].max())],
                "speed_range": [float(action_array[:, 1].min()), float(action_array[:, 1].max())],
                "log_prob_range": [float(np.min(log_probabilities)), float(np.max(log_probabilities))],
                "value_range": [float(np.min(values)), float(np.max(values))],
            },
            "video": video,
            "no_update_evidence": {
                "policy_max_parameter_delta": policy_delta,
                "optimizer_state_entries_before": optimizer_state_before,
                "optimizer_state_entries_after": optimizer_state_after,
            },
            "historical_video": {
                "path": str(original_video_path.relative_to(ROOT)),
                "sha256_before": original_video_hash_before,
                "sha256_after": original_video_hash_after,
                "overwritten": original_video_hash_before != original_video_hash_after,
            },
            "checks": checks,
        }
        temporary = OUTPUT_RESULTS.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
                default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(OUTPUT_RESULTS)
        print(json.dumps({
            "verdict": result["verdict"],
            "video": video,
            "historical_first_action_max_error": first_action_error,
            "policy_max_parameter_delta": policy_delta,
        }, indent=2))
        return 0 if all(checks.values()) else 1
    finally:
        diagnostic_env.close_video()
        vector_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
