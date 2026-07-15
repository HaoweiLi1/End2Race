"""Acceptance tests for the deliberately small PPO V1 implementation."""

from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import tempfile
import unittest

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from eval_multiagent import collision_scope_stops_episode
from model import End2Race
from rl.end2race_gymnasium_env import End2RaceGymnasiumEnv, EpisodeResetSpec
from rl.end2race_recurrent_ppo import End2RaceRecurrentPPO
from rl.ppo_callbacks import PPOV1MetricsCallback
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
from train_ppo_sb3 import (
    DEFAULT_CONFIG,
    V1_1_EVALUATION_UPDATES,
    paired_change_metrics,
    resolved_configuration,
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


def configuration_args(**overrides):
    values = {
        "updates": DEFAULT_CONFIG["updates"],
        "n_envs": DEFAULT_CONFIG["n_envs"],
        "n_steps": DEFAULT_CONFIG["n_steps"],
        "batch_size": DEFAULT_CONFIG["batch_size"],
        "device": "cpu",
        "master_seed": DEFAULT_CONFIG["master_seed"],
        "lr_scale": DEFAULT_CONFIG["lr_scale"],
        "evaluation_workers": 1,
        "collision_sampling_probability": DEFAULT_CONFIG["collision_sampling_probability"],
        "evaluation_updates": None,
        "smoke": "none",
    }
    values.update(overrides)
    return Namespace(**values)


class CallbackTestLogger:
    def __init__(self):
        self.records = {}

    def record(self, name, value):
        self.records[name] = value


class CallbackTestVecEnv:
    def __init__(self):
        self.reset_infos = [
            {"scenario_id": "scenario-a", "sampler_branch": "all_training"},
            {"scenario_id": "scenario-b", "sampler_branch": "bc_ego_collision"},
        ]


class CallbackTestModel:
    def __init__(self, env):
        self.env = env
        self.logger = CallbackTestLogger()
        self.num_timesteps = 0

    def get_env(self):
        return self.env


def callback_info(scenario_id, branch, *, collision=False, relative_position=-1.0):
    return {
        "scenario_id": scenario_id,
        "sampler_branch": branch,
        "ego_collision": collision,
        "relative_position_m": relative_position,
        "opponent_collision_latched": False,
        "termination_reason": "ego_collision" if collision else None,
        "elapsed_time": 1.0,
        "reward_total": -2.0 if collision else 0.01,
        "reward_progress": 0.01,
        "reward_relative": 0.0,
        "reward_collision": -2.0 if collision else 0.0,
    }


class TestPPOV11Configuration(unittest.TestCase):
    def test_v1_default_derived_values_remain_unchanged(self):
        config = resolved_configuration(configuration_args())
        self.assertEqual(config["configuration_profile"], "ppo_v1")
        self.assertEqual(config["transitions_per_update"], 12_800)
        self.assertEqual(config["minibatches_per_update"], 16)
        self.assertEqual(config["optimizer_steps_per_update"], 16)
        self.assertEqual(config["total_transitions"], 256_000)
        self.assertEqual(config["total_optimizer_steps"], 320)
        self.assertEqual(config["collision_sampling_probability"], 0.25)
        self.assertEqual(config["evaluation_updates"], [5, 10, 15, 20])

    def test_v1_1_derived_values_and_fixed_evaluation_schedule(self):
        config = resolved_configuration(
            configuration_args(
                n_steps=1600,
                batch_size=1600,
                collision_sampling_probability=0.50,
                evaluation_updates=V1_1_EVALUATION_UPDATES,
            )
        )
        self.assertEqual(config["configuration_profile"], "ppo_v1_1")
        self.assertEqual(config["transitions_per_update"], 25_600)
        self.assertEqual(config["minibatches_per_update"], 16)
        self.assertEqual(config["optimizer_steps_per_update"], 16)
        self.assertEqual(config["total_transitions"], 512_000)
        self.assertEqual(config["total_optimizer_steps"], 320)
        self.assertEqual(config["evaluation_updates"], [2, 3, 5, 10, 15, 20])
        for name in (
            "gamma",
            "gae_lambda",
            "clip_range",
            "n_epochs",
            "vf_coef",
            "ent_coef",
            "max_grad_norm",
            "target_kl",
        ):
            self.assertEqual(config[name], DEFAULT_CONFIG[name])

    def test_v1_1_schedule_probability_and_divisibility_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "PPO V1.1 requires"):
            resolved_configuration(
                configuration_args(
                    n_steps=1600,
                    batch_size=1600,
                    collision_sampling_probability=0.50,
                )
            )
        with self.assertRaisesRegex(ValueError, "collision_sampling_probability"):
            resolved_configuration(configuration_args(collision_sampling_probability=1.01))
        with self.assertRaisesRegex(ValueError, "evenly divide"):
            resolved_configuration(configuration_args(batch_size=801))

    def test_v1_1_smoke_profiles_keep_full_rollout_geometry(self):
        zero_lr = resolved_configuration(configuration_args(smoke="v1_1_zero_lr"))
        nonzero = resolved_configuration(configuration_args(smoke="v1_1_nonzero"))
        self.assertEqual(zero_lr["configuration_profile"], "v1_1_zero_lr")
        self.assertEqual(zero_lr["transitions_per_update"], 25_600)
        self.assertEqual(zero_lr["minibatches_per_update"], 16)
        self.assertEqual(zero_lr["total_optimizer_steps"], 16)
        self.assertEqual(zero_lr["evaluation_updates"], [])
        self.assertEqual(zero_lr["lr_scale"], 0.0)
        self.assertEqual(nonzero["configuration_profile"], "v1_1_nonzero")
        self.assertEqual(nonzero["total_transitions"], 51_200)
        self.assertEqual(nonzero["total_optimizer_steps"], 32)
        self.assertEqual(nonzero["evaluation_updates"], [])
        self.assertEqual(nonzero["lr_scale"], 1.0)

    def test_paired_change_metrics(self):
        bc_rows = []
        candidate_rows = []
        for index in range(600):
            scenario_id = f"scenario-{index:03d}"
            if index == 0:
                bc_outcome, candidate_outcome = "ego_collision", "follow"
            elif index == 1:
                bc_outcome, candidate_outcome = "follow", "ego_collision"
            elif index == 2:
                bc_outcome, candidate_outcome = "follow", "overtake"
            elif index == 3:
                bc_outcome, candidate_outcome = "overtake", "follow"
            else:
                bc_outcome = candidate_outcome = "follow"
            bc_rows.append({"scenario_id": scenario_id, "outcome": bc_outcome})
            candidate_rows.append({"scenario_id": scenario_id, "outcome": candidate_outcome})
        self.assertEqual(
            paired_change_metrics(bc_rows, candidate_rows),
            {
                "fixed_collision": 1,
                "new_collision": 1,
                "gained_overtake": 1,
                "lost_overtake": 1,
            },
        )


class TestPPOV1MetricsCallback(unittest.TestCase):
    def test_episode_reset_scenario_and_rollout_boundary_statistics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            env = CallbackTestVecEnv()
            model = CallbackTestModel(env)
            callback = PPOV1MetricsCallback(temporary_directory, n_envs=2)
            callback.init_callback(model)
            callback.on_training_start({}, {})
            callback.on_rollout_start()

            steps = [
                (
                    [
                        callback_info("scenario-a", "all_training"),
                        callback_info("scenario-b", "bc_ego_collision"),
                    ],
                    [False, False],
                ),
                (
                    [
                        callback_info("scenario-a", "all_training", collision=True),
                        callback_info("scenario-b", "bc_ego_collision"),
                    ],
                    [True, False],
                ),
                (
                    [
                        callback_info("scenario-c", "bc_ego_collision"),
                        callback_info("scenario-b", "bc_ego_collision", relative_position=-0.5),
                    ],
                    [False, True],
                ),
            ]
            for step_index, (infos, dones) in enumerate(steps):
                if step_index == 1:
                    env.reset_infos[0] = {
                        "scenario_id": "scenario-c",
                        "sampler_branch": "bc_ego_collision",
                    }
                if step_index == 2:
                    env.reset_infos[1] = {
                        "scenario_id": "scenario-d",
                        "sampler_branch": "all_training",
                    }
                model.num_timesteps += 2
                callback.update_locals(
                    {
                        "infos": infos,
                        "dones": np.asarray(dones, dtype=bool),
                        "actions": np.zeros((2, 2), dtype=np.float32),
                    }
                )
                self.assertTrue(callback.on_step())
            callback.on_rollout_end()

            summary = callback.latest_update_summary
            self.assertEqual(summary["transitions"], 6)
            self.assertEqual(summary["sampler_branch_transitions"], {"all_training": 2, "bc_ego_collision": 4})
            self.assertEqual(summary["completed_episodes"], 2)
            self.assertEqual(
                summary["completed_episodes_by_sampler_branch"],
                {"all_training": 1, "bc_ego_collision": 1},
            )
            self.assertEqual(summary["ego_collision_episodes_by_sampler_branch"], {"all_training": 1})
            self.assertEqual(summary["follow_episodes_by_sampler_branch"], {"bc_ego_collision": 1})
            self.assertEqual(summary["overtake_episodes_by_sampler_branch"], {})
            self.assertEqual(
                summary["reset_count_by_sampler_branch"],
                {"all_training": 2, "bc_ego_collision": 2},
            )
            self.assertEqual(summary["unique_scenario_id_count"], 3)
            self.assertEqual(
                summary["unique_scenario_ids"],
                ["scenario-a", "scenario-b", "scenario-c"],
            )
            self.assertEqual(summary["partial_episodes_carried_in"], 0)
            self.assertEqual(summary["partial_episodes_carried_out"], 1)
            self.assertEqual(summary["partial_episodes_carried_across_rollout_boundary"], 1)

            callback.on_training_start({}, {})
            callback.on_rollout_start()
            model.num_timesteps += 2
            callback.update_locals(
                {
                    "infos": [
                        callback_info("scenario-c", "bc_ego_collision"),
                        callback_info("scenario-d", "all_training"),
                    ],
                    "dones": np.asarray([False, False], dtype=bool),
                    "actions": np.zeros((2, 2), dtype=np.float32),
                }
            )
            self.assertTrue(callback.on_step())
            callback.on_rollout_end()
            self.assertEqual(callback.latest_update_summary["partial_episodes_carried_in"], 1)
            self.assertEqual(callback.latest_update_summary["partial_episodes_carried_out"], 2)
            self.assertEqual(callback.latest_update_summary["reset_count_by_sampler_branch"], {})


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
        sampler = FixedMixtureScenarioSampler(
            self.training,
            collision_ids,
            collision_probability=0.50,
        )
        rng_a = np.random.default_rng(20260715)
        rng_b = np.random.default_rng(20260715)
        sequence_a = [(scenario.scenario_id, branch) for scenario, branch in (sampler.sample(rng_a) for _ in range(100))]
        sequence_b = [(scenario.scenario_id, branch) for scenario, branch in (sampler.sample(rng_b) for _ in range(100))]
        self.assertEqual(sequence_a, sequence_b)
        self.assertEqual({branch for _scenario_id, branch in sequence_a}, {"all_training", "bc_ego_collision"})


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
