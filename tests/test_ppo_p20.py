from __future__ import annotations

import gc
from pathlib import Path
import tempfile
import unittest

import gym
from gymnasium import spaces
import numpy as np
import torch

from f110_gym.envs.base_classes import Integrator
from model import End2Race
from ppo.environment import End2RaceGymnasiumEnv
from ppo.geometry import CurrentStateClearances
from ppo.policy import (
    CRITIC_VARIANTS,
    END2RACE_OBSERVATION_SIZE,
    DetachedGRUCritic,
    End2RaceGRUPolicy,
    IndependentGRUCritic,
    MLPCritic,
    PriviledgeMLPCritic,
)
from ppo.privileged import (
    CURVATURE_LOOKAHEAD_M,
    CURVATURE_LOOKAHEAD_SAMPLES,
    PRIVILEGED_FEATURE_HIGHS,
    PRIVILEGED_FEATURE_LOWS,
    PRIVILEGED_FEATURE_NAMES,
    PRIVILEGED_FEATURE_SIZE,
    BoundaryDistanceReference,
)
from ppo.reward import (
    PPOTransitionReward,
    anisotropic_risk_potential,
    potential_shaping_reward,
)
from ppo.scenarios import ScenarioSpec
from sb3_contrib.common.recurrent.type_aliases import RNNStates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_MODEL = PROJECT_ROOT / "pretrained" / "end2race.pth"
EXPECTED_NAMES = (
    "delta_s",
    "relative_lateral",
    "relative_long_velocity",
    "relative_lat_velocity",
    "sin_relative_heading",
    "cos_relative_heading",
    "ego_speed",
    "ego_yaw_rate",
    "relative_yaw_rate",
    "obb_longitudinal_clearance",
    "obb_lateral_clearance",
    "wall_clearance",
    "ego_steering_angle",
    "ego_slip_angle",
    "left_body_margin",
    "right_body_margin",
    "sin_track_heading_error",
    "cos_track_heading_error",
    "current_curvature",
    "lookahead_mean_curvature",
)


def make_real_environment(*, privileged: bool) -> End2RaceGymnasiumEnv:
    scenario = ScenarioSpec(
        "p20-real-smoke",
        "ordinary",
        0,
        100,
        115,
        "raceline1",
        0.7,
        15,
        "Austin",
    ).to_reset_spec("ordinary")
    core = gym.make(
        "f110-v0",
        map=str(PROJECT_ROOT / "f1tenth_racetracks" / "Austin" / "Austin_map"),
        map_ext=".png",
        num_agents=2,
        timestep=0.01,
        integrator=Integrator.RK4,
        seed=0,
    )
    return End2RaceGymnasiumEnv(
        core,
        lambda _rng: scenario,
        "Austin",
        "raceline1",
        privileged=privileged,
        reward_gamma=0.999,
    )


class P20ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.standard_env = make_real_environment(privileged=False)
        cls.privileged_env = make_real_environment(privileged=True)
        cls.standard_observation, _ = cls.standard_env.reset(seed=0)
        cls.privileged_observation, _ = cls.privileged_env.reset(seed=0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.standard_env.close()
        cls.privileged_env.close()

    def test_01_names_size_and_bounds(self) -> None:
        self.assertEqual(PRIVILEGED_FEATURE_NAMES, EXPECTED_NAMES)
        self.assertEqual(PRIVILEGED_FEATURE_SIZE, 20)
        self.assertEqual(len(PRIVILEGED_FEATURE_LOWS), 20)
        self.assertEqual(len(PRIVILEGED_FEATURE_HIGHS), 20)
        self.assertEqual(PRIVILEGED_FEATURE_LOWS[9:12], (0.0, 0.0, 0.0))
        self.assertTrue(all(value == -1.0 for index, value in enumerate(PRIVILEGED_FEATURE_LOWS) if index not in (9, 10, 11)))
        self.assertEqual(PRIVILEGED_FEATURE_HIGHS, (1.0,) * 20)
        self.assertEqual(CURVATURE_LOOKAHEAD_M, 1.0)
        self.assertEqual(CURVATURE_LOOKAHEAD_SAMPLES, 16)
        metadata = self.privileged_env.privileged_normalization_metadata()
        self.assertEqual(metadata["curvature_lookahead_m"], 1.0)
        self.assertEqual(metadata["curvature_lookahead_samples"], 16)
        self.assertEqual(
            metadata["obb_longitudinal_clearance_m"],
            self.privileged_env.transition_reward.risk_longitudinal_clearance_m,
        )
        self.assertEqual(
            metadata["obb_lateral_clearance_m"],
            self.privileged_env.transition_reward.risk_lateral_clearance_m,
        )
        self.assertEqual(
            metadata["wall_clearance_m"],
            self.privileged_env.transition_reward.risk_wall_clearance_m,
        )

    def test_02_observation_dimensions_and_real_austin_smoke(self) -> None:
        self.assertEqual(self.standard_env.observation_space.shape, (361,))
        self.assertEqual(self.privileged_env.observation_space.shape, (381,))
        self.assertEqual(self.standard_observation.shape, (361,))
        self.assertEqual(self.privileged_observation.shape, (381,))
        self.assertEqual(self.privileged_observation.dtype, np.float32)
        features = self.privileged_observation[-20:]
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_less(np.asarray(PRIVILEGED_FEATURE_LOWS) - 1e-7, features)
        np.testing.assert_array_less(features, np.asarray(PRIVILEGED_FEATURE_HIGHS) + 1e-7)

        next_observation, _reward, _terminated, _truncated, _info = self.privileged_env.step(
            np.asarray((0.0, 3.0), dtype=np.float32)
        )
        self.assertEqual(next_observation.shape, (381,))
        self.assertTrue(np.isfinite(next_observation[-20:]).all())

    def test_03_actor_slices_first_361_and_is_checkpoint_compatible(self) -> None:
        torch.manual_seed(0)
        actor = End2Race(mask_prob=0.0, hidden_scale=4)
        actor.load_state_dict(torch.load(PRETRAINED_MODEL, map_location="cpu", weights_only=True), strict=True)
        actor.eval()
        actor_observation = torch.rand((2, END2RACE_OBSERVATION_SIZE), dtype=torch.float32)
        privileged_tail = torch.rand((2, PRIVILEGED_FEATURE_SIZE), dtype=torch.float32)
        full_observation = torch.cat((actor_observation, privileged_tail), dim=1)
        sliced = End2RaceGRUPolicy._actor_observation(full_observation)
        self.assertTrue(torch.equal(sliced, actor_observation))
        hidden = torch.zeros((1, 2, actor.gru.hidden_size), dtype=torch.float32)
        with torch.no_grad():
            original_output, original_hidden = actor(
                actor_observation[:, :360].unsqueeze(1),
                actor_observation[:, 360:].unsqueeze(1),
                hidden,
            )
            sliced_output, sliced_hidden = actor(
                sliced[:, :360].unsqueeze(1),
                sliced[:, 360:].unsqueeze(1),
                hidden,
            )
        torch.testing.assert_close(sliced_output, original_output, rtol=0.0, atol=0.0)
        torch.testing.assert_close(sliced_hidden, original_hidden, rtol=0.0, atol=0.0)

    def test_04_clearance_features_are_reward_geometry(self) -> None:
        extractor = self.privileged_env.privileged_extractor
        clearances = self.privileged_env.transition_reward.current_clearances
        self.assertIsNotNone(clearances)
        features = self.privileged_env._observation(self.privileged_env._raw_observation)[-20:]
        expected = np.asarray(
            (
                np.clip(clearances.obb_longitudinal_clearance_m / extractor.risk_longitudinal_clearance_m, 0.0, 1.0),
                np.clip(clearances.obb_lateral_clearance_m / extractor.risk_lateral_clearance_m, 0.0, 1.0),
                np.clip(clearances.wall_clearance_m / extractor.risk_wall_clearance_m, 0.0, 1.0),
            ),
            dtype=np.float32,
        )
        np.testing.assert_array_equal(features[9:12], expected)

    def test_05_steering_comes_from_current_simulator_state_not_pending_action(self) -> None:
        core = getattr(self.privileged_env.f110_env, "unwrapped", self.privileged_env.f110_env)
        ego_state = core.sim.agents[0].state
        original_steering = float(ego_state[2])
        try:
            ego_state[2] = 0.1
            first = self.privileged_env._observation(self.privileged_env._raw_observation)[-20:]
            pending_action = np.asarray((-0.4, 8.0), dtype=np.float32)
            self.assertEqual(float(pending_action[0]), np.float32(-0.4))
            second = self.privileged_env._observation(self.privileged_env._raw_observation)[-20:]
            self.assertAlmostEqual(float(first[12]), 0.1 / self.privileged_env.privileged_extractor.steering_scale_rad, places=6)
            np.testing.assert_array_equal(second, first)
        finally:
            ego_state[2] = original_steering

    def test_06_directional_body_margins_and_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lane.csv"
            np.savetxt(
                path,
                np.asarray(
                    (
                        (0.0, 0.0, 1.0, 1.0),
                        (10.0, 0.0, 1.0, 1.0),
                        (10.0, -10.0, 1.0, 1.0),
                        (0.0, -10.0, 1.0, 1.0),
                    )
                ),
                delimiter=",",
            )
            boundary = BoundaryDistanceReference(path)
            center = boundary.body_track_state(np.asarray((5.0, 0.0)), 0.0, 0.58, 0.31)
            right = boundary.body_track_state(np.asarray((5.0, -0.4)), 0.0, 0.58, 0.31)
            left = boundary.body_track_state(np.asarray((5.0, 0.4)), 0.0, 0.58, 0.31)
            outside_left = boundary.body_track_state(np.asarray((5.0, 1.1)), 0.0, 0.58, 0.31)
            rotated = boundary.body_track_state(np.asarray((5.0, 0.0)), np.pi / 4.0, 0.58, 0.31)
        self.assertLess(left.normalized_left_margin, right.normalized_left_margin)
        self.assertGreater(left.normalized_right_margin, right.normalized_right_margin)
        self.assertLess(outside_left.normalized_left_margin, 0.0)
        self.assertGreater(rotated.lateral_extent_m, center.lateral_extent_m)
        self.assertLess(rotated.left_margin_m, center.left_margin_m)
        self.assertLess(rotated.right_margin_m, center.right_margin_m)

    def test_07_heading_pair_and_cyclic_curvature(self) -> None:
        features = self.privileged_env._observation(self.privileged_env._raw_observation)[-20:]
        self.assertAlmostEqual(float(features[16] ** 2 + features[17] ** 2), 1.0, places=6)
        extractor = self.privileged_env.privileged_extractor
        length = extractor.projector.track_length
        self.assertAlmostEqual(extractor.curvature_at(-0.1), extractor.curvature_at(length - 0.1), places=12)
        self.assertAlmostEqual(extractor.curvature_at(length + 0.1), extractor.curvature_at(0.1), places=12)
        self.assertAlmostEqual(
            extractor.lookahead_mean_curvature_at(-0.1),
            extractor.lookahead_mean_curvature_at(length - 0.1),
            places=12,
        )
        epsilon = 1e-7
        self.assertAlmostEqual(extractor.curvature_at(length - epsilon), extractor.curvature_at(epsilon), places=5)

    def test_08_risk_on_off_produce_identical_p20(self) -> None:
        base_reward = self.privileged_env.transition_reward
        raw_observation = self.privileged_env._raw_observation
        common = dict(
            gamma=base_reward.gamma,
            vehicle_length=base_reward.vehicle_length,
            vehicle_width=base_reward.vehicle_width,
            map_clearance=base_reward.map_clearance,
            risk_longitudinal_clearance_m=base_reward.risk_longitudinal_clearance_m,
            risk_lateral_clearance_m=base_reward.risk_lateral_clearance_m,
            risk_wall_clearance_m=base_reward.risk_wall_clearance_m,
        )
        reward_off = PPOTransitionReward("Austin", "raceline1", risk_potential_maximum=0.0, **common)
        reward_on = PPOTransitionReward("Austin", "raceline1", risk_potential_maximum=0.05, **common)
        reward_off.reset(raw_observation, scenario_id="same")
        reward_on.reset(raw_observation, scenario_id="same")
        self.assertEqual(reward_off.current_clearances, reward_on.current_clearances)
        steering, ego_slip, opponent_slip = self.privileged_env._privileged_physical_state()
        extractor = self.privileged_env.privileged_extractor
        off_features = extractor.features(
            raw_observation,
            ego_index=0,
            opponent_index=1,
            ego_steering_angle=steering,
            ego_slip_angle=ego_slip,
            opponent_slip_angle=opponent_slip,
            clearances=reward_off.current_clearances,
        )
        on_features = extractor.features(
            raw_observation,
            ego_index=0,
            opponent_index=1,
            ego_steering_angle=steering,
            ego_slip_angle=ego_slip,
            opponent_slip_angle=opponent_slip,
            clearances=reward_on.current_clearances,
        )
        np.testing.assert_array_equal(off_features, on_features)


class RewardRiskRegressionTests(unittest.TestCase):
    def test_anisotropic_potential_and_terminal_shaping(self) -> None:
        safe = anisotropic_risk_potential(
            0.6,
            0.2,
            0.2,
            longitudinal_safe_m=0.6,
            lateral_safe_m=0.2,
            wall_safe_m=0.2,
            maximum_magnitude=0.05,
        )
        danger = anisotropic_risk_potential(
            0.0,
            0.0,
            0.0,
            longitudinal_safe_m=0.6,
            lateral_safe_m=0.2,
            wall_safe_m=0.2,
            maximum_magnitude=0.05,
        )
        self.assertEqual(safe, 0.0)
        self.assertAlmostEqual(danger, -0.05)
        reward, carried = potential_shaping_reward(-0.02, -0.03, 0.999, terminated=False)
        self.assertAlmostEqual(reward, 0.999 * -0.03 - -0.02)
        self.assertEqual(carried, -0.03)
        terminal_reward, terminal_carried = potential_shaping_reward(-0.02, -0.03, 0.999, terminated=True)
        self.assertEqual(terminal_reward, 0.02)
        self.assertEqual(terminal_carried, 0.0)


class CriticForwardTests(unittest.TestCase):
    def test_four_critics_forward_and_privileged_backward_checkpoint(self) -> None:
        self.assertEqual(CRITIC_VARIANTS, ("mlp", "detached_gru", "independent_gru", "priviledge_mlp"))
        actor = End2Race(mask_prob=0.0, hidden_scale=4)
        actor.load_state_dict(torch.load(PRETRAINED_MODEL, map_location="cpu", weights_only=True), strict=True)
        batch_size = 2
        actor_observation = torch.zeros((batch_size, END2RACE_OBSERVATION_SIZE), dtype=torch.float32)

        mlp = MLPCritic(actor)
        self.assertEqual(tuple(mlp(actor_observation).shape), (batch_size, 1))

        detached = DetachedGRUCritic(actor.gru.hidden_size, actor.gru.input_size)
        detached_features = torch.zeros((batch_size, actor.gru.hidden_size), dtype=torch.float32)
        self.assertEqual(tuple(detached(detached_features).shape), (batch_size, 1))

        independent = IndependentGRUCritic(actor)
        hidden = torch.zeros((1, batch_size, actor.gru.hidden_size), dtype=torch.float32)
        independent_values, next_hidden = independent.step(
            actor_observation[:, :360].unsqueeze(1),
            actor_observation[:, 360:].unsqueeze(1),
            hidden,
        )
        self.assertEqual(tuple(independent_values.shape), (batch_size, 1))
        self.assertEqual(tuple(next_hidden.shape), tuple(hidden.shape))

        privileged = PriviledgeMLPCritic()
        self.assertEqual(privileged.network[0].in_features, 20)
        self.assertEqual(privileged.network[0].out_features, 120)
        self.assertEqual(privileged.network[2].in_features, 120)
        self.assertEqual(privileged.network[2].out_features, 30)
        inputs = torch.linspace(-1.0, 1.0, batch_size * PRIVILEGED_FEATURE_SIZE).reshape(batch_size, -1)
        values = privileged(inputs)
        self.assertEqual(tuple(values.shape), (batch_size, 1))
        values.square().mean().backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in privileged.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "privileged_critic.pt"
            torch.save(privileged.state_dict(), path)
            restored = PriviledgeMLPCritic()
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            torch.testing.assert_close(restored(inputs), values.detach())

        del mlp, detached, independent, privileged, restored, actor
        gc.collect()

    def test_each_policy_variant_completes_one_rollout_forward(self) -> None:
        action_space = spaces.Box(
            low=np.asarray((-0.52, -10.0), dtype=np.float32),
            high=np.asarray((0.52, 10.0), dtype=np.float32),
            dtype=np.float32,
        )
        for variant in CRITIC_VARIANTS:
            observation_size = 381 if variant == "priviledge_mlp" else 361
            policy = End2RaceGRUPolicy(
                spaces.Box(-np.inf, np.inf, shape=(observation_size,), dtype=np.float32),
                action_space,
                lambda _progress: 1e-4,
                checkpoint_path=PRETRAINED_MODEL,
                hidden_scale=4,
                critic_variant=variant,
            )
            hidden_size = policy.end2race_actor.gru.hidden_size
            zero = torch.zeros((1, 1, hidden_size), dtype=torch.float32)
            states = RNNStates(pi=(zero.clone(), zero.clone()), vf=(zero.clone(), zero.clone()))
            observation = torch.zeros((1, observation_size), dtype=torch.float32)
            with torch.no_grad():
                actions, values, log_prob, next_states = policy.forward(
                    observation,
                    states,
                    torch.ones(1, dtype=torch.float32),
                    deterministic=True,
                )
            self.assertEqual(tuple(actions.shape), (1, 2), variant)
            self.assertEqual(tuple(values.shape), (1, 1), variant)
            self.assertEqual(tuple(log_prob.shape), (1,), variant)
            self.assertTrue(torch.isfinite(actions).all(), variant)
            self.assertTrue(torch.isfinite(values).all(), variant)
            self.assertEqual(tuple(next_states.pi[0].shape), (1, 1, hidden_size), variant)
            del policy, states, next_states
            gc.collect()


if __name__ == "__main__":
    unittest.main()
