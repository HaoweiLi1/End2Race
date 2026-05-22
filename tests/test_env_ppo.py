import numpy as np

from env_ppo import End2RacePPOEnv, RewardState, RewardWeights


CENTERLINE = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float64)


def make_obs(ego_x, opp_x, ego_v=5.0, opp_v=4.0, collision=False):
    return {
        "scans": [np.ones(1440, dtype=np.float32) * 10.0, np.ones(1440, dtype=np.float32) * 10.0],
        "poses_x": np.array([ego_x, opp_x], dtype=np.float64),
        "poses_y": np.array([0.0, 0.0], dtype=np.float64),
        "poses_theta": np.array([0.0, 0.0], dtype=np.float64),
        "linear_vels_x": np.array([ego_v, opp_v], dtype=np.float64),
        "collisions": np.array([1.0 if collision else 0.0, 0.0], dtype=np.float64),
    }


class DummySim:
    def __init__(self, next_obs, done=False):
        self.next_obs = next_obs
        self.done = done
        self.last_action = None

    def step(self, action):
        self.last_action = action.copy()
        return self.next_obs, 0.0, self.done, {}

    def close(self):
        pass


def make_env(raw_obs, next_obs, sim_duration=1.0, terminate_on_success=True, severe=False, rw=None):
    env = object.__new__(End2RacePPOEnv)
    env.max_speed = 20.0
    env.sim_duration = sim_duration
    env.terminate_on_success = terminate_on_success
    env.terminate_on_severe_unsafe = severe
    env.reward_weights = rw or RewardWeights()
    env.env = DummySim(next_obs)
    env.timestep = 0.01
    env.centerline = CENTERLINE
    env._raw_obs = raw_obs
    env._reward_state = RewardState.from_obs(raw_obs, CENTERLINE)
    env._prev_exec_action = np.array([0.0, 5.0], dtype=np.float32)
    env._prev_speed = 5.0
    env._t = 0.0
    env._opp_planner_step = lambda _obs: np.array([0.0, 4.0], dtype=np.float32)
    return env


def test_timeout_truncates_without_termination():
    raw_obs = make_obs(0.0, 10.0)
    next_obs = make_obs(0.1, 10.0)
    env = make_env(raw_obs, next_obs, sim_duration=0.01)
    obs, _, terminated, truncated, info = env.step(np.array([0.0, 5.0], dtype=np.float32))

    assert not terminated
    assert truncated
    assert info["timeout"]
    assert env._raw_obs is next_obs
    assert env._prev_speed == next_obs["linear_vels_x"][0]
    assert obs["lidar"].shape == (360,)
    assert obs["lidar"].dtype == np.float32
    assert obs["prev_speed"].shape == (1,)
    assert obs["prev_speed"].dtype == np.float32


def test_collision_terminates_and_dominates_timeout():
    raw_obs = make_obs(0.0, 10.0)
    next_obs = make_obs(0.1, 10.0, collision=True)
    env = make_env(raw_obs, next_obs, sim_duration=0.01)
    _, _, terminated, truncated, info = env.step(np.array([0.0, 5.0], dtype=np.float32))

    assert terminated
    assert not truncated
    assert info["collision"]
    assert info["timeout"]


def test_safe_overtake_success_truncates():
    raw_obs = make_obs(0.0, 0.0)
    next_obs = make_obs(2.1, 0.0)
    rw = RewardWeights(safe_overtake_hold_duration=0.01)
    env = make_env(raw_obs, next_obs, sim_duration=10.0, rw=rw)
    _, _, terminated, truncated, info = env.step(np.array([0.0, 5.0], dtype=np.float32))

    assert not terminated
    assert truncated
    assert info["success"]


def test_severe_unsafe_can_terminate():
    raw_obs = make_obs(0.0, 1.0, ego_v=5.0, opp_v=0.0)
    next_obs = make_obs(0.0, 0.1, ego_v=5.0, opp_v=0.0)
    env = make_env(raw_obs, next_obs, sim_duration=10.0, severe=True)
    _, _, terminated, truncated, info = env.step(np.array([0.0, 5.0], dtype=np.float32))

    assert terminated
    assert not truncated
    assert info["severe"]


def test_action_clipping_info_and_executed_action():
    raw_obs = make_obs(0.0, 10.0)
    next_obs = make_obs(0.1, 10.0)
    env = make_env(raw_obs, next_obs, sim_duration=10.0)
    _, _, _, _, info = env.step(np.array([2.0, -5.0], dtype=np.float32))

    assert np.allclose(info["raw_ego_action"], [2.0, -5.0])
    assert np.allclose(info["executed_ego_action"], [0.52, 0.0])
    assert info["action_was_clipped"]
    assert np.allclose(env.env.last_action[0], [0.52, 0.0])
