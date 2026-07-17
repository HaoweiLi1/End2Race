from dataclasses import replace
import unittest

from ppo import config as ppo_config
from ppo.reward import PPOTransitionReward


class AxisProjector:
    track_length = 100.0

    def project(self, point):
        return float(point[0] % self.track_length)


def observation(ego_x, opponent_x):
    return {
        "poses_x": [ego_x, opponent_x],
        "poses_y": [0.0, 0.0],
        "poses_theta": [0.0, 0.0],
    }


def step(reward, previous, current, *, opponent_collision=False):
    return reward.step(
        previous,
        current,
        ego_collision=False,
        opponent_collision=opponent_collision,
        scenario_id="test",
    )


class SignalRepairTest(unittest.TestCase):
    def test_zero_margin_preserves_existing_reward_values(self):
        previous = observation(0.0, 1.0)
        current = observation(0.1, 1.1)
        baseline = PPOTransitionReward(AxisProjector())
        explicit_zero = PPOTransitionReward(AxisProjector(), margin_weight=0.0, margin_threshold=0.5)
        baseline.reset(previous, scenario_id="test")
        explicit_zero.reset(previous, scenario_id="test")

        baseline_result = step(baseline, previous, current)
        zero_result = step(explicit_zero, previous, current)

        self.assertEqual(baseline_result.reward_progress, zero_result.reward_progress)
        self.assertEqual(baseline_result.reward_relative, zero_result.reward_relative)
        self.assertEqual(baseline_result.reward_collision, zero_result.reward_collision)
        self.assertEqual(baseline_result.reward_total, zero_result.reward_total)
        self.assertEqual(zero_result.reward_margin, 0.0)

    def test_positive_margin_matches_oriented_clearance_formula(self):
        previous = observation(0.0, 1.0)
        current = observation(0.1, 1.1)
        reward = PPOTransitionReward(AxisProjector(), margin_weight=0.02, margin_threshold=0.5)
        reward.reset(previous, scenario_id="test")

        result = step(reward, previous, current)

        self.assertAlmostEqual(result.reward_margin, -0.02 * (0.5 - 0.42) ** 2)
        self.assertAlmostEqual(
            result.reward_total,
            result.reward_progress + result.reward_relative + result.reward_margin + result.reward_collision,
        )

    def test_opponent_collision_latch_clears_margin_immediately_and_afterward(self):
        initial = observation(0.0, 1.0)
        first = observation(0.1, 1.1)
        second = observation(0.2, 1.2)
        reward = PPOTransitionReward(AxisProjector(), margin_weight=0.02, margin_threshold=0.5)
        reward.reset(initial, scenario_id="test")

        latched = step(reward, initial, first, opponent_collision=True)
        after_latch = step(reward, first, second)

        self.assertTrue(latched.opponent_collision_latched)
        self.assertEqual(latched.reward_margin, 0.0)
        self.assertTrue(after_latch.opponent_collision_latched)
        self.assertEqual(after_latch.reward_margin, 0.0)

    def test_signal_repair_configs_match_preregistered_contract(self):
        self.assertEqual(ppo_config.SG_LR10.updates, 24)
        self.assertEqual(ppo_config.SG_LR10.checkpoint_updates, (8, 16, 24))
        self.assertEqual(ppo_config.SG_LR10.n_epochs, 2)
        self.assertEqual(ppo_config.SG_LR10.margin_weight, 0.0)
        self.assertEqual(ppo_config.SG_FULL.margin_weight, 0.02)
        self.assertEqual(ppo_config.SG_FULL.margin_threshold, 0.5)
        self.assertEqual(ppo_config.SG_FULL.gru_lr, ppo_config.SG_LR10.gru_lr)
        self.assertEqual(ppo_config.SG_FULL.gru_lr, 1.0e-5)
        self.assertEqual(ppo_config.SG_FULL.head_lr, ppo_config.SG_LR10.head_lr)
        self.assertEqual(ppo_config.SG_FULL.head_lr, 1.0e-4)
        self.assertEqual(ppo_config.SG_FULL.target_kl, ppo_config.SG_LR10.target_kl)
        self.assertEqual(ppo_config.SG_FULL.target_kl, 0.010)

    def test_margin_and_epoch_validation(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            PPOTransitionReward(AxisProjector(), margin_weight=-0.01)
        invalid = replace(ppo_config.SG_LR10, name="invalid", n_epochs=0)
        original = ppo_config.CONFIGS
        try:
            ppo_config.CONFIGS = {"invalid": invalid}
            with self.assertRaisesRegex(ValueError, "n_epochs"):
                ppo_config._validate(invalid)
        finally:
            ppo_config.CONFIGS = original


if __name__ == "__main__":
    unittest.main()
