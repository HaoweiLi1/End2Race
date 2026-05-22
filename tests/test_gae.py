import numpy as np
import torch

from train_ppo import RolloutBuffer


def add_step(
    buf,
    i,
    reward,
    value=0.0,
    terminated=False,
    truncated=False,
    trunc_next_value=0.0,
):
    buf.add(
        lidar=np.zeros(360, dtype=np.float32),
        prev_speed=np.zeros(1, dtype=np.float32),
        raw_action=np.zeros(2, dtype=np.float32),
        reward=reward,
        value=value,
        log_prob=0.0,
        terminated=terminated,
        truncated=truncated,
        trunc_next_value=trunc_next_value,
        episode_start=(i == 0),
    )


def make_buffer(steps):
    buf = RolloutBuffer(len(steps), gamma=1.0, gae_lambda=1.0)
    buf.reset()
    for i, step in enumerate(steps):
        add_step(buf, i, **step)
    return buf


def test_no_boundary_uses_candidate_last_value():
    buf = make_buffer([{"reward": 1.0}, {"reward": 2.0}, {"reward": 3.0}])
    buf.compute_returns_and_advantage(5.0, last_terminated=False, last_truncated=False)
    assert np.allclose(buf.returns, [11.0, 10.0, 8.0])


def test_final_true_termination_zero_bootstrap():
    buf = make_buffer([{"reward": 1.0}, {"reward": 2.0}, {"reward": 3.0, "terminated": True}])
    buf.compute_returns_and_advantage(999.0, last_terminated=True, last_truncated=False)
    assert np.allclose(buf.returns, [6.0, 5.0, 3.0])


def test_final_truncation_uses_trunc_next_value():
    buf = make_buffer(
        [
            {"reward": 1.0, "value": 10.0},
            {"reward": 2.0, "value": 20.0},
            {"reward": 3.0, "value": 30.0, "truncated": True, "trunc_next_value": 7.0},
        ]
    )
    buf.compute_returns_and_advantage(999.0, last_terminated=False, last_truncated=True)
    assert np.allclose(buf.returns, [13.0, 12.0, 10.0])


def test_mid_rollout_termination_cuts_chain_and_zero_bootstraps():
    buf = make_buffer([{"reward": 1.0}, {"reward": 2.0, "terminated": True}, {"reward": 3.0}])
    buf.compute_returns_and_advantage(5.0, last_terminated=False, last_truncated=False)
    assert np.allclose(buf.returns, [3.0, 2.0, 8.0])


def test_mid_rollout_truncation_cuts_chain_and_bootstraps():
    buf = make_buffer(
        [
            {"reward": 1.0},
            {"reward": 2.0, "truncated": True, "trunc_next_value": 7.0},
            {"reward": 3.0},
        ]
    )
    buf.compute_returns_and_advantage(5.0, last_terminated=False, last_truncated=False)
    assert np.allclose(buf.returns, [10.0, 9.0, 8.0])


def test_full_batch_shapes_and_dtypes():
    buf = make_buffer([{"reward": 0.0}, {"reward": 0.0}, {"reward": 0.0}, {"reward": 0.0}])
    buf.compute_returns_and_advantage(0.0, False, False)
    lidar_b, spd_b, act_b, old_logp_b, adv_b, ret_b, starts_b = buf.full_batch_tensors(device="cpu")

    assert lidar_b.shape == (1, 4, 360)
    assert spd_b.shape == (1, 4, 1)
    assert act_b.shape == (1, 4, 2)
    assert old_logp_b.shape == (1, 4)
    assert adv_b.shape == (1, 4)
    assert ret_b.shape == (1, 4)
    assert starts_b.shape == (1, 4)
    assert lidar_b.dtype == torch.float32
