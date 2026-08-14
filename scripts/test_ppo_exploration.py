import json
from pathlib import Path
import sys

from gymnasium import spaces
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo.env import make_environment
from ppo.policy import End2RaceGRUPolicy
from latticeplanner.utils import load_config


def exploration_test():
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(381,), dtype=np.float32)
    action_space = spaces.Box(
        low=np.asarray((-0.52, -8.0), dtype=np.float32),
        high=np.asarray((0.52, 8.0), dtype=np.float32),
        dtype=np.float32,
    )

    def policy(speed_hold, corridor_hold):
        return End2RaceGRUPolicy(
            observation_space,
            action_space,
            lambda _progress: 1.0,
            checkpoint_path=PROJECT_ROOT / "pretrained/end2race.pth",
            speed_noise_hold_steps=speed_hold,
            front_corridor_speed_noise_hold_steps=corridor_hold,
        )

    global_policy = policy(10, 0)
    global_noise = []
    for step in range(25):
        global_policy.prepare_rollout_exploration(np.asarray([False]), np.asarray([step == 0]))
        _log_std, noise = global_policy._structured_rollout_parameters(1)
        global_noise.append(float(noise[0]))

    corridor_policy = policy(10, 50)
    corridor_noise = []
    for step in range(70):
        corridor_policy.prepare_rollout_exploration(np.asarray([12 <= step < 62]), np.asarray([step == 0]))
        _log_std, noise = corridor_policy._structured_rollout_parameters(1)
        corridor_noise.append(float(noise[0]))

    assert len(set(global_noise[:10])) == len(set(global_noise[10:20])) == len(set(global_noise[20:])) == 1
    assert global_noise[0] != global_noise[10] != global_noise[20]
    assert len(set(corridor_noise[12:62])) == 1
    assert corridor_noise[11] != corridor_noise[12]
    assert corridor_noise[61] != corridor_noise[62]
    return {"global_speed_hold_steps": 10, "corridor_speed_hold_steps": 50}


def corridor_gate_test():
    config = load_config("ppo/ppo_config.yaml")
    environment = make_environment(
        42,
        "Austin",
        config,
        privileged=True,
        reward_gamma=0.999,
        front_corridor_speed_noise_hold_steps=50,
    )()
    try:
        point = environment.projector.segment_start[20]
        tangent = environment.projector.segment_vector[20] / environment.projector.segment_length[20]
        normal = np.asarray((-tangent[1], tangent[0]))
        opponent = point + tangent + 0.28 * normal
        heading = float(np.arctan2(tangent[1], tangent[0]))
        observation = {
            "poses_x": np.asarray((point[0], opponent[0])),
            "poses_y": np.asarray((point[1], opponent[1])),
            "poses_theta": np.asarray((heading, heading)),
        }
        admitted = environment._front_corridor_gate(observation)
        assert config.front_corridor_gate_maximum_abs_opponent_lateral_d_m == 0.25
        assert config.front_corridor_gate_require_positive_lateral_overlap
        assert not admitted
        return {"maximum_lateral_offset_m": 0.25, "off_center_opponent_admitted": admitted}
    finally:
        environment.close()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    print(json.dumps({"exploration": exploration_test(), "corridor_gate": corridor_gate_test()}, indent=2))
