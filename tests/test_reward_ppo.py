import numpy as np

from env_ppo import RewardState, RewardWeights, compute_shaped_reward


CENTERLINE = np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float64)


def make_obs(
    ego_x,
    opp_x,
    ego_v=5.0,
    opp_v=4.0,
    collision=False,
    ego_y=0.0,
    opp_y=0.0,
    ego_theta=0.0,
):
    return {
        "poses_x": np.array([ego_x, opp_x], dtype=np.float64),
        "poses_y": np.array([ego_y, opp_y], dtype=np.float64),
        "poses_theta": np.array([ego_theta, 0.0], dtype=np.float64),
        "linear_vels_x": np.array([ego_v, opp_v], dtype=np.float64),
        "collisions": np.array([1.0 if collision else 0.0, 0.0], dtype=np.float64),
    }


def test_forward_progress_reward_positive():
    rw = RewardWeights()
    state = RewardState.from_obs(make_obs(0.0, 10.0), CENTERLINE)
    _, terms = compute_shaped_reward(
        make_obs(0.1, 10.0),
        state,
        CENTERLINE,
        rw,
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
        dt=0.01,
    )
    assert terms["reward_progress"] > 0.0


def test_backward_progress_clips():
    rw = RewardWeights()
    state = RewardState.from_obs(make_obs(10.0, 20.0), CENTERLINE)
    _, terms = compute_shaped_reward(
        make_obs(9.0, 20.0),
        state,
        CENTERLINE,
        rw,
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
        dt=0.01,
    )
    assert np.isclose(terms["reward_progress"], -rw.w_progress * rw.progress_clip_back)


def test_overtake_progress_and_success_bonus_once():
    rw = RewardWeights(safe_overtake_hold_duration=0.1)
    state = RewardState.from_obs(make_obs(0.0, 0.0), CENTERLINE)
    _, terms1 = compute_shaped_reward(
        make_obs(2.1, 0.0, ego_v=5.0, opp_v=4.0),
        state,
        CENTERLINE,
        rw,
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
        dt=0.1,
    )
    _, terms2 = compute_shaped_reward(
        make_obs(2.2, 0.0, ego_v=5.0, opp_v=4.0),
        state,
        CENTERLINE,
        rw,
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
        dt=0.1,
    )
    assert terms1["reward_rel_progress"] > 0.0
    assert terms1["reward_overtake_progress"] > 0.0
    assert state.safe_overtake_held
    assert terms1["reward_overtake_success"] == rw.w_overtake_success
    assert terms2["reward_overtake_success"] == 0.0


def test_safe_factor_reduces_progress_reward_under_high_risk():
    rw = RewardWeights(
        w_opponent_risk=0.0,
        side_s_thresh=1.0,
        side_dist_thresh=2.0,
        rear_s_thresh=0.0,
        front_s_thresh=0.0,
    )
    low_state = RewardState.from_obs(make_obs(0.0, 0.0), CENTERLINE)
    high_state = RewardState.from_obs(make_obs(0.0, 0.0), CENTERLINE)
    _, low_terms = compute_shaped_reward(
        make_obs(0.8, 0.0, ego_y=2.0, opp_y=0.0),
        low_state,
        CENTERLINE,
        rw,
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
        dt=0.01,
    )
    _, high_terms = compute_shaped_reward(
        make_obs(0.8, 0.0, ego_y=0.0, opp_y=0.0),
        high_state,
        CENTERLINE,
        rw,
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
        dt=0.01,
    )
    assert high_terms["opponent_risk"] > low_terms["opponent_risk"]
    assert high_terms["safe_factor"] < low_terms["safe_factor"]
    assert high_terms["reward_rel_progress"] < low_terms["reward_rel_progress"]


def test_collision_penalty_and_post_overtake_collision():
    rw = RewardWeights()
    state = RewardState.from_obs(make_obs(0.0, 0.0), CENTERLINE)
    state.overtake_started = True
    _, terms = compute_shaped_reward(
        make_obs(0.5, 0.0, collision=True),
        state,
        CENTERLINE,
        rw,
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
        dt=0.01,
    )
    assert terms["reward_collision"] == -rw.w_collision
    assert terms["reward_post_overtake_collision"] == -rw.w_post_overtake_collision
    assert state.post_overtake_collision


def test_unsafe_merge_back_penalizes_poor_clearance_with_aggressive_steer():
    rw = RewardWeights()
    state = RewardState.from_obs(make_obs(0.0, 0.0), CENTERLINE)
    state.overtake_started = True
    _, terms = compute_shaped_reward(
        make_obs(1.0, 0.0, ego_y=0.0, opp_y=0.1),
        state,
        CENTERLINE,
        rw,
        np.array([0.0, 5.0]),
        np.array([0.52, 5.0]),
        dt=0.01,
    )
    assert terms["unsafe_merge_back"] > 0.0
    assert terms["reward_unsafe_merge_back"] < 0.0


def test_smaller_lateral_separation_increases_unsafe_merge_back():
    rw = RewardWeights()
    close_lat_state = RewardState.from_obs(make_obs(0.0, 0.0), CENTERLINE)
    wide_lat_state = RewardState.from_obs(make_obs(0.0, 0.0), CENTERLINE)
    close_lat_state.overtake_started = True
    wide_lat_state.overtake_started = True
    obs = make_obs(1.0, 0.0, ego_y=0.0, opp_y=1.0, ego_theta=-np.pi / 4)
    _, close_terms = compute_shaped_reward(
        obs,
        close_lat_state,
        CENTERLINE,
        rw,
        np.array([0.0, 5.0]),
        np.array([0.2, 5.0]),
        dt=0.01,
    )
    obs_rotated = make_obs(1.0, 0.0, ego_y=0.0, opp_y=1.0, ego_theta=np.pi / 4)
    _, wide_terms = compute_shaped_reward(
        obs_rotated,
        wide_lat_state,
        CENTERLINE,
        rw,
        np.array([0.0, 5.0]),
        np.array([0.2, 5.0]),
        dt=0.01,
    )
    assert abs(close_terms["rel_y_ego"]) < abs(wide_terms["rel_y_ego"])
    assert close_terms["unsafe_merge_back"] > wide_terms["unsafe_merge_back"]


def test_safe_overtake_requires_new_margin_and_hold_duration():
    rw = RewardWeights()
    state = RewardState.from_obs(make_obs(0.0, 0.0), CENTERLINE)
    compute_shaped_reward(
        make_obs(0.5, 0.0),
        state,
        CENTERLINE,
        rw,
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
        dt=rw.safe_overtake_hold_duration,
    )
    assert not state.safe_overtake_held

    state = RewardState.from_obs(make_obs(0.0, 0.0), CENTERLINE)
    compute_shaped_reward(
        make_obs(2.1, 0.0),
        state,
        CENTERLINE,
        rw,
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
        dt=rw.safe_overtake_hold_duration - 0.01,
    )
    assert not state.safe_overtake_held

    compute_shaped_reward(
        make_obs(2.2, 0.0),
        state,
        CENTERLINE,
        rw,
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
        dt=0.02,
    )
    assert state.safe_overtake_held


def test_severe_unsafe_ttc_flag():
    rw = RewardWeights(severe_ttc=0.15)
    state = RewardState.from_obs(make_obs(0.0, 1.0, ego_v=5.0, opp_v=0.0), CENTERLINE)
    _, terms = compute_shaped_reward(
        make_obs(0.0, 0.1, ego_v=5.0, opp_v=0.0),
        state,
        CENTERLINE,
        rw,
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
        dt=0.01,
    )
    assert state.severe_unsafe
    assert terms["severe_unsafe"] == 1.0


def test_smoothness_penalty_uses_executed_action_delta():
    rw = RewardWeights()
    prev_action = np.array([0.0, 5.0])
    executed = np.array([0.1, 7.0])
    _, terms = compute_shaped_reward(
        make_obs(0.1, 10.0),
        RewardState.from_obs(make_obs(0.0, 10.0), CENTERLINE),
        CENTERLINE,
        rw,
        prev_action,
        executed,
        dt=0.01,
    )
    expected_raw = 0.1**2 + (rw.smooth_speed_scale * 2.0) ** 2
    assert np.isclose(terms["reward_smooth"], -rw.w_smooth * expected_raw)
