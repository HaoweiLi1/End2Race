import json
import hashlib
from pathlib import Path
import sys
import tempfile

from gymnasium import spaces
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo.env import EXTERNAL_RESET_OPTION, make_environment
from ppo.policy import End2RaceGRUPolicy
from ppo.rollout import FirstActionPreferenceDataset
from ppo.scenarios import ordinary_scenarios
from latticeplanner.utils import load_config

def exploration_test():
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(381,), dtype=np.float32)
    action_space = spaces.Box(low=np.asarray((-0.52, -8.0), dtype=np.float32), high=np.asarray((0.52, 8.0), dtype=np.float32), dtype=np.float32)

    def build_policy(speed_hold, corridor_hold):
        return End2RaceGRUPolicy(observation_space, action_space, lambda _progress: 1.0, checkpoint_path=PROJECT_ROOT / "pretrained/end2race.pth", critic_variant="privilege_gru", speed_noise_hold_steps=speed_hold, front_corridor_speed_noise_hold_steps=corridor_hold)

    if build_policy(1, 0).exploration_mode != "stepwise_independent":
        raise RuntimeError("Default stepwise exploration mode changed")
    global_policy = build_policy(10, 0)
    global_noises = []
    for step in range(25):
        global_policy.prepare_rollout_exploration(np.asarray([False]), np.asarray([step == 0]))
        _log_std, noise = global_policy._structured_rollout_parameters(1)
        global_noises.append(float(noise[0]))
    if len(set(global_noises[:10])) != 1 or len(set(global_noises[10:20])) != 1 or len(set(global_noises[20:])) != 1 or global_noises[0] == global_noises[10] or global_noises[10] == global_noises[20]:
        raise RuntimeError("Global K10 speed residual did not follow the ten-step hold contract")

    corridor_policy = build_policy(10, 50)
    corridor_noises = []
    for step in range(70):
        gate = 12 <= step < 62
        corridor_policy.prepare_rollout_exploration(np.asarray([gate]), np.asarray([step == 0]))
        _log_std, noise = corridor_policy._structured_rollout_parameters(1)
        corridor_noises.append(float(noise[0]))
    if len(set(corridor_noises[12:62])) != 1 or corridor_noises[11] == corridor_noises[12] or corridor_noises[61] == corridor_noises[62]:
        raise RuntimeError("K10/K50 speed residual did not resample at corridor phase boundaries")
    return {"global_speed_k10_blocks": 3, "corridor_speed_k50_block_steps": 50, "phase_boundary_resample": True}


def corridor_gate_config_test():
    default_environment = make_environment(42, "Austin", privileged=True, reward_gamma=0.999, front_corridor_speed_noise_hold_steps=50)()
    try:
        default_config = default_environment.corridor_gate_config
        if default_config.maximum_abs_opponent_lateral_d_m != 0.25:
            raise RuntimeError("Default front-corridor lateral-offset threshold changed")
        if not default_config.require_positive_lateral_overlap:
            raise RuntimeError("Front-corridor OBB overlap requirement changed")
        default_gate = default_environment.following_danger_gate
        point = default_gate.projector.segment_start[20]
        tangent = default_gate.projector.segment_vector[20] / default_gate.projector.segment_length[20]
        normal = np.asarray((-tangent[1], tangent[0]))
        opponent = point + tangent + 0.28 * normal
        heading = float(np.arctan2(tangent[1], tangent[0]))
        observation = {
            "poses_x": np.asarray((point[0], opponent[0])),
            "poses_y": np.asarray((point[1], opponent[1])),
            "poses_theta": np.asarray((heading, heading)),
        }
        if default_gate._evaluate(observation, ego_index=0, opponent_index=1):
            raise RuntimeError("Default corridor unexpectedly admitted the off-center opponent")
        return {"maximum_abs_opponent_lateral_d_m": 0.25, "positive_lateral_obb_overlap_required": True, "off_center_overlap_admitted": False}
    finally:
        default_environment.close()


def snapshot_test():
    config = load_config("ppo/ppo_config.yaml")
    environment = make_environment(42, "Austin", privileged=True, reward_gamma=0.999)()
    try:
        scenario = ordinary_scenarios("Austin", config)[0]
        observation, _info = environment.reset(seed=42, options={EXTERNAL_RESET_OPTION: scenario.to_reset_spec("ordinary")})
        action = np.asarray((0.01, 3.0), dtype=np.float32)
        for _step in range(5):
            observation, _reward, terminated, truncated, _info = environment.step(action)
            if terminated or truncated:
                raise RuntimeError("Runtime snapshot smoke scenario terminated before capture")
        snapshot = environment.capture_runtime_snapshot()
        first = environment.step(action)
        restored = environment.restore_runtime_snapshot(snapshot)
        second = environment.step(action)
        if not np.array_equal(restored, observation) or not np.array_equal(first[0], second[0]) or first[1:4] != second[1:4]:
            raise RuntimeError("Runtime snapshot did not reproduce the next transition exactly")
        return {"observation_size": int(observation.size), "exact_next_transition": True}
    finally:
        environment.close()


def fixed_dataset_test():
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(381,), dtype=np.float32)
    action_space = spaces.Box(low=np.asarray((-0.52, -8.0), dtype=np.float32), high=np.asarray((0.52, 8.0), dtype=np.float32), dtype=np.float32)
    policy = End2RaceGRUPolicy(observation_space, action_space, lambda _progress: 1.0, checkpoint_path=PROJECT_ROOT / "pretrained/end2race.pth", critic_variant="privilege_gru")
    with tempfile.TemporaryDirectory(prefix="fixed_preference_fixture_") as directory:
        root = Path(directory)
        episodes = root / "episodes"
        episodes.mkdir()
        rows = []
        for role, good, bad in (
            ("target", (0.02, 0.5), (0.0, 0.0)),
            ("control", (0.0, 0.0), (-0.02, -0.5)),
        ):
            sequence = episodes / f"{role}.npz"
            np.savez_compressed(
                sequence,
                observations=np.zeros((2, 361), dtype=np.float32),
                episode_starts=np.asarray((True, False), dtype=bool),
            )
            sequence_sha256 = hashlib.sha256(sequence.read_bytes()).hexdigest()
            rows.append({
                "episode_key": f"fixture-{role}",
                "role": role,
                "stratum": f"fixture-{role}",
                "sequence_file": f"episodes/{role}.npz",
                "sequence_sha256": sequence_sha256,
                "states": [{
                    "decision_index": 1,
                    "lead_steps": 50,
                    "pairs": [{"good_action": list(good), "bad_action": list(bad), "family": "coordinated", "direction": "candidate_preferred" if role == "target" else "noop_preferred"}],
                }],
            })
        gate = {"schema_version": 1, "verdict": "pass", "target_labeled_episode_count": 1, "control_labeled_episode_count": 1}
        gate_path = root / "gate_report.json"
        gate_path.write_text(json.dumps(gate, sort_keys=True), encoding="utf-8")
        manifest = {"schema_version": 1, "gate_report_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(), "episodes": rows}
        (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        dataset = FirstActionPreferenceDataset(root, policy, torch.device("cpu"), 42)
        loss, target_loss, control_loss, margins = dataset.loss()
        values = [float(value.detach().cpu().item()) for value in (loss, target_loss, control_loss)]
        if not np.isfinite(values).all() or not margins.numel():
            raise RuntimeError("Fixed first-action preference dataset loss is invalid")
        return {"target_episodes": len(dataset.target_indices), "control_episodes": len(dataset.control_indices), "margin_count": int(margins.numel()), "uses_previous_ppo_dataset": False}


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    print(json.dumps({"exploration": exploration_test(), "corridor_gate": corridor_gate_config_test(), "snapshot": snapshot_test(), "fixed_dataset": fixed_dataset_test()}, indent=2, sort_keys=True))
