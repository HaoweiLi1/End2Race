"""Acceptance tests for the deliberately small PPO V1 implementation."""

from __future__ import annotations

from copy import deepcopy
import unittest

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from eval_multiagent import collision_scope_stops_episode
from model import End2Race
from rl.end2race_gymnasium_env import End2RaceGymnasiumEnv, EpisodeResetSpec
from rl.end2race_recurrent_ppo import End2RaceRecurrentPPO
from rl.ppo_reward import (
    PPOV1TransitionReward,
    ProgressProjector,
    checked_progress_delta,
    wrapped_progress_delta,
)
from rl.ppo_scenarios import (
    EVALUATION_STARTPOINTS,
    TRAINING_STARTPOINTS,
    FixedMixtureScenarioSampler,
    evaluation_scenarios,
    training_scenarios,
)
from rl.sb3_end2race_policy import (
    END2RACE_ACTION_SIZE,
    END2RACE_OBSERVATION_SIZE,
    NOOP_SPEED_BOUND,
    PPO_V1_CRITIC_LR,
    PPO_V1_GRU_LR,
    PPO_V1_HEAD_LR,
    PPO_V1_SPEED_LOG_STD,
    PPO_V1_STEER_LOG_STD,
    End2RaceGRUPolicy,
)


def square_projector() -> ProgressProjector:
    return ProgressProjector(
        np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]),
        4.0,
    )


def square_point(progress: float) -> tuple[float, float]:
    progress %= 4.0
    if progress < 1.0:
        return progress, 0.0
    if progress < 2.0:
        return 1.0, progress - 1.0
    if progress < 3.0:
        return 3.0 - progress, 1.0
    return 0.0, 4.0 - progress


def raw_observation(ego_progress: float, opponent_progress: float, collisions=(False, False)):
    ego = square_point(ego_progress)
    opponent = square_point(opponent_progress)
    return {
        "poses_x": np.asarray([ego[0], opponent[0]], dtype=np.float64),
        "poses_y": np.asarray([ego[1], opponent[1]], dtype=np.float64),
        "poses_theta": np.zeros(2, dtype=np.float64),
        "linear_vels_x": np.asarray([1.0, 1.0], dtype=np.float64),
        "scans": np.full((2, 360), 5.0, dtype=np.float32),
        "collisions": np.asarray(collisions, dtype=bool),
    }


class NoOpOpponent:
    def reset(self, spec, num_agents, ego_index):
        self.num_agents = num_agents

    def actions(self, raw_observation):
        return np.zeros((self.num_agents, 2), dtype=np.float32)

    def state_snapshot(self):
        return {}


class LegacyCore:
    num_agents = 2
    timestep = 0.1

    @property
    def unwrapped(self):
        return self

    def reset(self, *, poses):
        self.observation = raw_observation(0.1, 0.5)
        return deepcopy(self.observation), self.timestep, False, {}

    def step(self, action):
        self.observation = raw_observation(0.2, 0.55)
        return deepcopy(self.observation), self.timestep, False, {}

    def close(self):
        return None


def fixed_provider(rng):
    del rng
    return EpisodeResetSpec(
        poses=np.zeros((2, 3), dtype=np.float64),
        initial_speed_feature=0.0,
        scenario={"scenario_id": "synthetic-episode", "sampler_branch": "all_training"},
    )


class TinyEnd2RaceEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(END2RACE_OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(
            np.asarray([-0.52, -NOOP_SPEED_BOUND], dtype=np.float32),
            np.asarray([0.52, NOOP_SPEED_BOUND], dtype=np.float32),
            dtype=np.float32,
        )
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.full(END2RACE_OBSERVATION_SIZE, 1.0, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        observation = np.full(END2RACE_OBSERVATION_SIZE, 1.0 + 0.01 * self.steps, dtype=np.float32)
        reward = float(0.01 * action[1] - 0.001 * abs(action[0]))
        return observation, reward, False, self.steps >= 4, {}


class TestPPOV1Reward(unittest.TestCase):
    def test_progress_projection_seam_and_invalid_delta(self):
        projector = square_projector()
        self.assertAlmostEqual(projector.project(np.asarray([0.25, 0.0])), 0.25)
        self.assertAlmostEqual(projector.project(np.asarray([0.0, 0.1])), 3.9)
        self.assertAlmostEqual(wrapped_progress_delta(0.1, 3.9, 4.0), 0.2)
        with self.assertRaisesRegex(ValueError, "bad-scenario.*previous_s.*current_s"):
            checked_progress_delta(2.0, 0.0, 4.0, scenario_id="bad-scenario", vehicle="ego")

    def test_reward_components_collision_once_and_latch_reset(self):
        reward = PPOV1TransitionReward(square_projector())
        previous = raw_observation(0.1, 0.5)
        reward.reset(previous, scenario_id="reward-case")

        current = raw_observation(0.2, 0.55)
        first = reward.step(
            previous,
            current,
            ego_collision=False,
            opponent_collision=False,
            scenario_id="reward-case",
        )
        self.assertAlmostEqual(first.reward_progress, 0.001)
        self.assertAlmostEqual(first.reward_relative, 0.001)
        self.assertAlmostEqual(first.reward_total, first.reward_progress + first.reward_relative + first.reward_collision)

        latched_observation = raw_observation(0.3, 0.6, (False, True))
        latched = reward.step(
            current,
            latched_observation,
            ego_collision=False,
            opponent_collision=True,
            scenario_id="reward-case",
        )
        self.assertTrue(latched.opponent_collision_latched)
        self.assertEqual(latched.reward_relative, 0.0)

        collision_observation = raw_observation(0.4, 0.65, (True, False))
        collision = reward.step(
            latched_observation,
            collision_observation,
            ego_collision=True,
            opponent_collision=False,
            scenario_id="reward-case",
        )
        repeated = reward.step(
            collision_observation,
            collision_observation,
            ego_collision=True,
            opponent_collision=False,
            scenario_id="reward-case",
        )
        self.assertEqual(collision.reward_collision, -2.0)
        self.assertEqual(repeated.reward_collision, 0.0)

        reward.reset(previous, scenario_id="reward-case")
        after_reset = reward.step(
            previous,
            current,
            ego_collision=True,
            opponent_collision=False,
            scenario_id="reward-case",
        )
        self.assertFalse(after_reset.opponent_collision_latched)
        self.assertEqual(after_reset.reward_collision, -2.0)


class TestPPOV1Scenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = training_scenarios()
        cls.evaluation = evaluation_scenarios()

    def test_fixed_pool_contract(self):
        self.assertEqual(len(TRAINING_STARTPOINTS), 50)
        self.assertEqual(len(EVALUATION_STARTPOINTS), 50)
        self.assertEqual(len(self.training), 600)
        self.assertEqual(len(self.evaluation), 600)
        self.assertEqual(len({scenario.scenario_id for scenario in self.training}), 600)
        self.assertEqual(len({scenario.scenario_id for scenario in self.evaluation}), 600)
        self.assertEqual(EVALUATION_STARTPOINTS[0], 0)
        self.assertEqual(EVALUATION_STARTPOINTS[-1], 2096)

    def test_sampler_seed_reproducibility(self):
        collision_ids = [scenario.scenario_id for scenario in self.training[:7]]
        sampler = FixedMixtureScenarioSampler(self.training, collision_ids)
        rng_a = np.random.default_rng(20260715)
        rng_b = np.random.default_rng(20260715)
        sequence_a = [tuple(item.scenario_id for item in [sampler.sample(rng_a)[0]]) for _ in range(100)]
        sequence_b = [tuple(item.scenario_id for item in [sampler.sample(rng_b)[0]]) for _ in range(100)]
        self.assertEqual(sequence_a, sequence_b)


class TestPPOV1Policy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        observation_space = spaces.Box(-np.inf, np.inf, shape=(END2RACE_OBSERVATION_SIZE,), dtype=np.float32)
        action_space = spaces.Box(
            np.asarray([-0.52, -NOOP_SPEED_BOUND], dtype=np.float32),
            np.asarray([0.52, NOOP_SPEED_BOUND], dtype=np.float32),
            dtype=np.float32,
        )
        cls.policy = End2RaceGRUPolicy(
            observation_space,
            action_space,
            lambda _: 1.0,
            optimizer_profile="ppo_v1",
        )

    def test_frozen_trainable_partition_and_fixed_std(self):
        named = dict(self.policy.named_parameters())
        self.assertFalse(named["end2race_actor.k"].requires_grad)
        self.assertFalse(named["end2race_actor.dummy_embedding"].requires_grad)
        self.assertTrue(all(not parameter.requires_grad for name, parameter in named.items() if name.startswith("end2race_actor.speed_mlp.")))
        self.assertTrue(all(parameter.requires_grad for name, parameter in named.items() if name.startswith("lstm_actor.gru.")))
        self.assertTrue(all(parameter.requires_grad for name, parameter in named.items() if name.startswith("end2race_actor.output_layer.")))
        self.assertTrue(all(parameter.requires_grad for name, parameter in named.items() if name.startswith("value_net.")))
        self.assertFalse(self.policy.log_std.requires_grad)
        np.testing.assert_allclose(
            self.policy.log_std.detach().numpy(),
            [PPO_V1_STEER_LOG_STD, PPO_V1_SPEED_LOG_STD],
            rtol=0.0,
            atol=1e-7,
        )

    def test_three_exact_nonoverlapping_optimizer_groups(self):
        groups = self.policy.optimizer.param_groups
        self.assertEqual([group["name"] for group in groups], ["gru", "head", "critic"])
        self.assertEqual([group["base_lr"] for group in groups], [PPO_V1_GRU_LR, PPO_V1_HEAD_LR, PPO_V1_CRITIC_LR])
        ids = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        expected = {
            id(parameter)
            for module in (self.policy.end2race_actor.gru, self.policy.end2race_actor.output_layer, self.policy.value_net)
            for parameter in module.parameters()
        }
        self.assertEqual(set(ids), expected)

    def test_actor_export_is_strict_twelve_key_schema(self):
        state = self.policy.actor_checkpoint_state_dict()
        self.assertEqual(len(state), 12)
        fresh = End2Race(mask_prob=0.0, hidden_scale=4)
        incompatible = fresh.load_state_dict(state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_stock_train_preserves_named_learning_rates(self):
        env = TinyEnd2RaceEnv()
        model = End2RaceRecurrentPPO(
            End2RaceGRUPolicy,
            env,
            learning_rate=1.0,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            gamma=0.999,
            gae_lambda=0.995,
            device="cpu",
            policy_kwargs={"optimizer_profile": "ppo_v1"},
            verbose=0,
        )
        before = {group["name"]: group["lr"] for group in model.policy.optimizer.param_groups}
        model.learn(total_timesteps=4)
        after = {group["name"]: group["lr"] for group in model.policy.optimizer.param_groups}
        expected = {"gru": PPO_V1_GRU_LR, "head": PPO_V1_HEAD_LR, "critic": PPO_V1_CRITIC_LR}
        self.assertEqual(before, expected)
        self.assertEqual(after, expected)
        env.close()


class TestPPOV1IntegrationCompatibility(unittest.TestCase):
    def test_env_reward_hook_and_legacy_default(self):
        legacy = End2RaceGymnasiumEnv(
            LegacyCore(),
            sim_duration=1.0,
            reset_provider=fixed_provider,
            opponent_controller=NoOpOpponent(),
        )
        legacy.reset(seed=1)
        _observation, legacy_reward, terminated, truncated, legacy_info = legacy.step(np.asarray([0.0, 1.0]))
        self.assertEqual(legacy_reward, 0.1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertNotIn("reward_progress", legacy_info)
        legacy.close()

        ppo = End2RaceGymnasiumEnv(
            LegacyCore(),
            sim_duration=1.0,
            reset_provider=fixed_provider,
            opponent_controller=NoOpOpponent(),
            transition_reward=PPOV1TransitionReward(square_projector()),
        )
        ppo.reset(seed=1)
        _observation, ppo_reward, _terminated, _truncated, info = ppo.step(np.asarray([0.0, 1.0]))
        self.assertAlmostEqual(ppo_reward, info["reward_total"])
        self.assertAlmostEqual(
            info["reward_total"],
            info["reward_progress"] + info["reward_relative"] + info["reward_collision"],
        )
        self.assertEqual(info["scenario_id"], "synthetic-episode")
        ppo.close()

    def test_evaluator_collision_scope_default_compatibility(self):
        self.assertTrue(collision_scope_stops_episode("legacy", False, True))
        self.assertFalse(collision_scope_stops_episode("ego", False, True))
        self.assertTrue(collision_scope_stops_episode("ego", True, False))
        self.assertTrue(collision_scope_stops_episode("ego", True, True))


if __name__ == "__main__":
    unittest.main()
