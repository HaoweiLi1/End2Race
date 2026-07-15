"""Acceptance tests for the removable SB3-Contrib End2Race GRU POC."""

from __future__ import annotations

import unittest

from scripts.smoke_sb3_gru import run_poc


class TestSB3GRUIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.poc_results = run_poc()

    def test_a_bc_sequence_identity(self):
        result = self.poc_results["bc_sequence_identity"]
        self.assertGreaterEqual(result["timesteps"], 100)
        self.assertLessEqual(result["max_action_absolute_error"], 1e-6)
        self.assertLessEqual(result["max_hidden_absolute_error"], 1e-6)
        self.assertLessEqual(result["adapter_max_absolute_error"], 1e-6)
        self.assertEqual(result["adapter_dummy_cell_max_absolute_value"], 0.0)
        self.assertEqual(
            result["adapter_interface"],
            {
                "input_size": 420,
                "hidden_size": 1680,
                "num_layers": 1,
                "accepts_h_c_tuple": True,
            },
        )
        self.assertTrue(result["gaussian_log_std_trainable"])
        self.assertTrue(result["gaussian_log_std_in_optimizer"])
        self.assertEqual(result["deterministic_mean_log_std_invariance_error"], 0.0)

    def test_b_episode_reset_identity(self):
        result = self.poc_results["episode_reset_identity"]
        self.assertGreaterEqual(result["parallel_envs"], 2)
        self.assertNotEqual(result["env_reset_steps"][0], result["env_reset_steps"][1])
        self.assertLessEqual(result["max_action_absolute_error"], 1e-6)
        self.assertLessEqual(result["max_hidden_absolute_error"], 1e-6)
        self.assertLessEqual(result["reset_slot_matches_fresh_zero_state_max_error"], 1e-6)
        self.assertEqual(result["dummy_cell_max_absolute_value"], 0.0)
        self.assertGreater(result["unaffected_env_differs_from_erroneous_zero_reset_min_margin"], 1e-6)

    def test_c_stock_ppo_replay_identity(self):
        result = self.poc_results["ppo_replay_identity"]
        self.assertLessEqual(result["max_logp_absolute_error"], 1e-6)
        self.assertLessEqual(result["mean_logp_absolute_error"], 1e-7)
        self.assertLessEqual(result["max_ratio_deviation"], 1e-6)
        self.assertGreaterEqual(result["coverage"]["parallel_envs"], 2)
        self.assertGreaterEqual(result["coverage"]["episode_boundary_count"], 2)
        self.assertTrue(result["coverage"]["ordinary_continuous_sequence"])
        self.assertTrue(result["coverage"]["timeout_truncation"])
        self.assertTrue(result["coverage"]["padding_mask"])
        self.assertGreater(result["padding_timesteps"], 0)

    def test_d_timeout_and_collision_bootstrap(self):
        result = self.poc_results["timeout_bootstrap"]
        wrapper = self.poc_results["gymnasium_wrapper"]
        self.assertEqual(wrapper["observation_shape"], [361])
        self.assertEqual(wrapper["lidar_values"], 360)
        self.assertEqual(wrapper["previous_speed_values"], 1)
        self.assertTrue(wrapper["actor_observation_is_plain_array"])
        self.assertFalse(wrapper["privileged_simulator_fields_in_actor_observation"])
        self.assertEqual(wrapper["wrapper_reward_transform"], "none")
        self.assertGreaterEqual(result["collision_events"], 1)
        self.assertGreaterEqual(result["timeout_events"], 1)
        self.assertTrue(result["collision_terminated_not_truncated"])
        self.assertTrue(result["timeout_truncated_not_terminated"])
        self.assertLessEqual(result["collision_zero_bootstrap_max_error"], 1e-7)
        self.assertLessEqual(result["timeout_terminal_value_bootstrap_max_error"], 1e-6)
        self.assertLessEqual(result["terminal_advantage_no_cross_episode_max_error"], 1e-6)

    def test_e_actor_checkpoint_compatibility(self):
        result = self.poc_results["checkpoint_compatibility"]
        self.assertTrue(result["strict_load_succeeded"])
        self.assertTrue(result["actor_keys_match_bc_schema"])
        self.assertEqual(result["actor_key_count"], 12)
        self.assertEqual(result["roundtrip_max_absolute_error"], 0.0)
        self.assertTrue(result["temporary_output_only"])

    def test_zero_lr_smoke_train_does_not_update_parameters(self):
        result = self.poc_results["smoke_train"]
        self.assertEqual(result["parallel_envs"], 2)
        self.assertLessEqual(result["max_episode_duration_seconds"], 1.0)
        self.assertEqual(result["rollouts"], 1)
        self.assertEqual(result["ppo_train_calls"], 1)
        self.assertEqual(result["learning_rate"], 0.0)
        self.assertEqual(result["max_parameter_delta"], 0.0)
        self.assertFalse(self.poc_results["third_party_sources_modified"])
        self.assertFalse(self.poc_results["end2race_learner_training_performed"])


if __name__ == "__main__":
    unittest.main()
