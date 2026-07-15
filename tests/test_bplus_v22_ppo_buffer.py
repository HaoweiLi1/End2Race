#!/usr/bin/env python3
"""B2 episode-complete replay and two-channel GAE regressions."""

import numpy as np

from bplus_v22 import BC_FEATURE_DIM, HISTORY_OFFSETS, LIDAR_BEAMS
from bplus_v22.macro import aggregate_micro_signals
from bplus_v22.ppo_buffer import (
    EpisodeCompleteMacroBuffer,
    MacroReplayRecord,
    validate_complete_episode,
)


def signals(*, length=10, collision=False, performance=False, terminated=False, truncated=False):
    collision_values = np.zeros(length, dtype=np.float64)
    performance_values = np.zeros(length, dtype=np.float64)
    term = np.zeros(length, dtype=bool)
    trunc = np.zeros(length, dtype=bool)
    if collision:
        collision_values[-1] = 1.0
    if performance:
        performance_values[-1] = 1.0
    term[-1] = terminated
    trunc[-1] = truncated
    return aggregate_micro_signals(
        np.zeros(length), collision_values, performance_values, term, trunc
    )


def record(
    episode,
    macro,
    signal,
    *,
    offset=(1.25, 2.5),
    collision_value=0.0,
    performance_value=0.0,
    collision_next=0.0,
    performance_next=0.0,
):
    return MacroReplayRecord(
        scenario_id=f"scenario-{episode}",
        l2_id=f"L2:{episode}",
        episode_id=f"episode-{episode}",
        macro_index=macro,
        arm="BC_FROZEN",
        training_seed=0,
        policy_iteration=3,
        checkpoint_schema="b2-test-checkpoint-1",
        bc_feature=np.zeros(BC_FEATURE_DIM, np.float32),
        lidar_history=np.zeros((len(HISTORY_OFFSETS), LIDAR_BEAMS), np.float32),
        scalar_history=np.zeros(24, np.float32),
        privileged_critic_feature=np.zeros(12, np.float32),
        latent=np.zeros(4, np.float32),
        old_log_prob=-0.25 - macro,
        old_entropy=0.75 + macro,
        entropy_intervention=0.10,
        entropy_steer_given_intervention=0.20,
        entropy_brake_gate_given_intervention=0.30,
        entropy_brake_magnitude_given_brake=0.40,
        intervention_offset=offset[0],
        conditional_brake_offset=offset[1],
        steer_std_scale=0.1,
        brake_std_scale=1.0,
        schedule_id="schedule-0",
        requested_residual=np.zeros(2, np.float32),
        applied_composition_digest="0" * 64,
        signals=signal,
        collision_value=collision_value,
        performance_value=performance_value,
        collision_trunc_next_value=collision_next,
        performance_trunc_next_value=performance_next,
        episode_start=macro == 0,
        bc_hidden_reset=macro == 0,
    )


def assert_raises(kind, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
        raise AssertionError(f"expected {kind.__name__}")
    except kind:
        pass


def main() -> None:
    first = record(1, 0, signals())
    collision_terminal = record(
        1,
        1,
        signals(length=3, collision=True, terminated=True),
        collision_value=0.2,
    )
    episode_one = validate_complete_episode([first, collision_terminal])
    assert len(episode_one) == 2 and episode_one[-1].signals.length == 3

    # No partial episode, early boundary, or changing behavior context is accepted.
    assert_raises(ValueError, validate_complete_episode, [first])
    assert_raises(
        ValueError,
        validate_complete_episode,
        [record(1, 0, signals(terminated=True)), collision_terminal],
    )
    changed = record(
        1, 1, signals(length=3, terminated=True), offset=(9.0, 2.5)
    )
    assert_raises(ValueError, validate_complete_episode, [first, changed])

    buffer = EpisodeCompleteMacroBuffer(minimum_transitions=3)
    buffer.add_episode(episode_one)
    assert not buffer.ready and buffer.episode_count == 1
    truncated_overtake = record(
        2,
        0,
        signals(length=4, performance=True, truncated=True),
        performance_value=0.1,
        collision_next=0.3,
        performance_next=0.4,
    )
    buffer.add_episode([truncated_overtake])
    assert buffer.ready and buffer.episode_count == 2
    batch = buffer.collate()
    assert len(batch.old_log_prob) == 3
    assert batch.macro_length.tolist() == [10, 3, 4]
    assert batch.episode_start.tolist() == [True, False, True]
    assert batch.terminated.tolist() == [False, True, False]
    assert batch.truncated.tolist() == [False, False, True]
    assert batch.collision_cost.tolist()[1] > 0.0
    assert batch.performance_reward.tolist()[2] > 0.0
    assert np.all(np.isfinite(batch.collision_return))
    assert np.all(np.isfinite(batch.performance_return))
    assert not hasattr(batch, "reward_return")
    tensors = batch.tensors("cpu")
    assert tensors["bc_feature"].shape == (3, 1680)
    assert tensors["latent"].shape == (3, 4)
    assert tensors["intervention_offset"].tolist() == [1.25] * 3
    assert tensors["conditional_brake_offset"].tolist() == [2.5] * 3
    assert np.allclose(tensors["steer_std_scale"].numpy(), [0.1] * 3)
    assert tensors["brake_std_scale"].tolist() == [1.0] * 3
    assert_raises(RuntimeError, buffer.add_episode, [truncated_overtake])

    # A whole long episode may overshoot the minimum; it must never be cut.
    overshoot = EpisodeCompleteMacroBuffer(minimum_transitions=2)
    long_episode = [
        record(3, 0, signals()),
        record(3, 1, signals()),
        record(3, 2, signals(terminated=True)),
    ]
    overshoot.add_episode(long_episode)
    assert overshoot.ready and len(overshoot.collate().old_log_prob) == 3

    fixed_episodes = EpisodeCompleteMacroBuffer(target_episodes=2)
    fixed_episodes.add_episode(episode_one)
    assert not fixed_episodes.ready
    fixed_episodes.add_episode([truncated_overtake])
    assert fixed_episodes.ready and fixed_episodes.episode_count == 2

    # Collision-empty rollouts remain finite and exactly zero with zero values.
    empty_collision = EpisodeCompleteMacroBuffer(minimum_transitions=2)
    empty_collision.add_episode(
        [
            record(4, 0, signals()),
            record(4, 1, signals(performance=True, terminated=True)),
        ]
    )
    empty_batch = empty_collision.collate()
    assert np.array_equal(
        empty_batch.collision_advantage, np.zeros(2, dtype=np.float32)
    )
    assert np.array_equal(empty_batch.collision_return, np.zeros(2, dtype=np.float32))

    # Strict latent and bootstrap validation catches silent schema drift.
    bad = dict(
        episode=5,
        macro=0,
        signal=signals(terminated=True),
    )
    normal = record(**bad)
    values = dict(normal.__dict__)
    values["latent"] = np.array([0.0, 1.0, 0.0, 0.0], np.float32)
    assert_raises(ValueError, MacroReplayRecord, **values)
    values = dict(normal.__dict__)
    values["collision_trunc_next_value"] = 1.0
    assert_raises(ValueError, MacroReplayRecord, **values)
    print("ALL TESTS PASSED")
if __name__ == "__main__":
    main()
