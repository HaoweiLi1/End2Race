import unittest

import numpy as np
import torch

from ppo import config as ppo_config
from ppo.policy import (
    EVALUATOR_STEER_BOUND,
    EvaluatorClippedPhysicalGaussianDistribution,
    EvaluatorCompatibleJointDistribution,
)


class PPOActionDistributionTest(unittest.TestCase):
    def test_v1_3_c_changes_only_the_registered_distribution_axis(self):
        baseline = ppo_config.V1_3_A
        candidate = ppo_config.V1_3_C
        differing = {
            field
            for field in baseline.__dataclass_fields__
            if getattr(baseline, field) != getattr(candidate, field)
        }
        self.assertEqual(differing, {"name", "steering_distribution"})
        self.assertEqual(candidate.steering_distribution, "physical_gaussian")

    def test_deterministic_action_matches_evaluator_after_clipping(self):
        raw_means = torch.tensor(
            [[-0.70, 4.0], [-0.20, 5.0], [0.20, 6.0], [0.70, 7.0]],
            dtype=torch.float32,
        )
        log_std = torch.log(torch.tensor([0.05, 0.15], dtype=torch.float32))
        distribution = EvaluatorClippedPhysicalGaussianDistribution().proba_distribution(raw_means, log_std)
        actions = distribution.get_actions(deterministic=True).numpy()
        actions[:, 0] = np.clip(actions[:, 0], -EVALUATOR_STEER_BOUND, EVALUATOR_STEER_BOUND)
        expected = raw_means.numpy().copy()
        expected[:, 0] = np.clip(expected[:, 0], -EVALUATOR_STEER_BOUND, EVALUATOR_STEER_BOUND)
        np.testing.assert_array_equal(actions, expected)

    def test_physical_gaussian_replay_ratio_is_identity(self):
        torch.manual_seed(20260717)
        raw_means = torch.tensor([[-0.55, 4.0], [0.10, 6.0]], dtype=torch.float32)
        log_std = torch.log(torch.tensor([0.05, 0.15], dtype=torch.float32))
        distribution = EvaluatorClippedPhysicalGaussianDistribution()
        actions = distribution.proba_distribution(raw_means, log_std).sample()
        old_log_prob = distribution.log_prob(actions)
        replay_log_prob = distribution.proba_distribution(raw_means, log_std).log_prob(actions)
        torch.testing.assert_close(torch.exp(replay_log_prob - old_log_prob), torch.ones(2))

    def test_physical_gaussian_removes_atanh_boundary_curvature(self):
        old_means = torch.tensor([[0.51900, 5.0]], dtype=torch.float64)
        new_means = torch.tensor([[0.51902, 5.0]], dtype=torch.float64)
        log_std = torch.log(torch.tensor([0.05, 0.15], dtype=torch.float64))

        squashed_old = EvaluatorCompatibleJointDistribution().proba_distribution(old_means, log_std)
        old_latent = squashed_old.latent_steer_mean.clone()
        squashed_new = EvaluatorCompatibleJointDistribution().proba_distribution(new_means, log_std)
        new_latent = squashed_new.latent_steer_mean.clone()
        squashed_kl = 0.5 * ((new_latent - old_latent) / log_std[0].exp()).square()

        physical_std = EVALUATOR_STEER_BOUND * log_std[0].exp()
        physical_kl = 0.5 * ((new_means[:, 0] - old_means[:, 0]) / physical_std).square()

        self.assertGreater(float(squashed_kl.item()), 0.019)
        self.assertLess(float(physical_kl.item()), 1.0e-6)
        self.assertGreater(float((squashed_kl / physical_kl).item()), 10_000.0)


if __name__ == "__main__":
    unittest.main()
