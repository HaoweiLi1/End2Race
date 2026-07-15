"""Hard gates for the removable SB3-Contrib End2Race GRU integration."""

from __future__ import annotations

import unittest

from scripts.smoke_sb3_gru import run_poc


class TestSB3GRUIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_poc(include_real_f110=True)

    def test_actor_and_hidden_identity(self):
        result = self.results["bc_sequence_identity"]
        self.assertEqual(result["timesteps"], 100)
        self.assertLessEqual(result["max_deterministic_action_error"], 1e-6)
        self.assertLessEqual(result["max_hidden_absolute_error"], 1e-6)
        self.assertLessEqual(result["adapter_max_absolute_error"], 1e-6)
        self.assertEqual(result["adapter_dummy_cell_max_absolute_value"], 0.0)
        self.assertEqual(result["deterministic_action_log_std_invariance_error"], 0.0)
        self.assertTrue(result["distribution_log_std_trainable"])
        self.assertTrue(result["distribution_log_std_in_optimizer"])
        self.assertLessEqual(result["joint_log_prob_independent_oracle_error"], 1e-10)

    def test_async_reset_provider_and_fixed_opponent(self):
        result = self.results["reset_and_opponent_contract"]
        self.assertEqual(result["parallel_envs"], 2)
        self.assertNotEqual(result["episode_lengths"][0], result["episode_lengths"][1])
        self.assertGreaterEqual(result["total_auto_resets"], 3)
        self.assertTrue(all(shape == [2, 3] for shape in result["all_reset_pose_shapes"]))
        self.assertTrue(result["all_reset_poses_reached_strict_core"])
        self.assertTrue(result["provider_instances_independent"])
        self.assertTrue(result["parallel_env_rng_samples_differ"])
        self.assertTrue(result["controller_instances_independent"])
        self.assertTrue(result["planner_instances_disjoint_between_envs"])
        self.assertTrue(result["opponent_state_clean_after_every_reset"])
        self.assertEqual(result["seeded_reset_poses_max_error"], 0.0)
        self.assertTrue(result["options_override_still_called_provider"])
        self.assertEqual(result["options_override_pose_max_error"], 0.0)
        self.assertEqual(result["options_override_initial_speed_error"], 0.0)
        self.assertTrue(result["scenario_only_override_applied"])
        self.assertTrue(result["strict_core_rejects_reset_without_poses"])
        self.assertEqual(result["opponent_determinism_max_error"], 0.0)
        self.assertTrue(result["opponent_action_has_no_direct_ego_action_argument"])
        self.assertTrue(result["opponent_replan_and_tracker_frequency_correct"])

    def test_speed_timing_and_lidar_contract(self):
        result = self.results["speed_and_lidar_contract"]
        self.assertEqual(result["speed_feature_trace"], [11.0, 21.0, 22.0])
        self.assertEqual(result["independent_evaluator_oracle"], [11.0, 21.0, 22.0])
        self.assertEqual(result["speed_feature_max_error"], 0.0)
        self.assertTrue(result["previous_desired_commands_not_used"])
        self.assertTrue(result["current_measured_speeds_not_used"])
        self.assertEqual(result["lidar_beam_index_max_error"], 0.0)
        self.assertEqual(result["lidar_expected_first_last_indices"], [0, 719])
        self.assertEqual(result["fail_fast"], {"short_scan": True, "nan_scan": True, "inf_scan": True})
        self.assertEqual(result["privileged_field_metamorphic_observation_error"], 0.0)

    def test_ego_only_collision_and_terminal_precedence(self):
        result = self.results["termination_semantics"]
        self.assertEqual(len(result["rows"]), 10)
        self.assertTrue(result["all_cases_match"])
        self.assertEqual(result["base_reward_max_error"], 0.0)
        self.assertTrue(result["opponent_only_collision_continues"])
        self.assertTrue(result["opponent_collision_timeout_is_truncation"])
        self.assertTrue(result["true_terminal_beats_timeout"])

    def test_physical_ego_action_probability_execution_identity(self):
        result = self.results["ego_action_contract"]
        self.assertLessEqual(result["max_buffer_ego_action_vs_core_error"], 1e-7)
        self.assertLessEqual(result["max_sb3_ego_action_vs_core_error"], 1e-7)
        self.assertLessEqual(result["max_wrapper_ego_action_vs_core_error"], 1e-7)
        self.assertEqual(result["steering_out_of_bound_count"], 0)
        self.assertEqual(result["sb3_pre_env_action_clipping_count"], 0)
        self.assertEqual(result["sb3_pre_env_action_max_clip_delta"], 0.0)
        self.assertTrue(result["action_sensitive_reward_verified"])
        self.assertTrue(result["action_sensitive_observation_verified"])
        self.assertLessEqual(result["action_sensitive_observation_max_error"], 1e-7)
        self.assertEqual(result["ppo_buffer_action_width"], 2)
        self.assertFalse(result["opponent_action_present_in_ppo_buffer"])
        self.assertEqual(result["fully_traced_ego_transition_count"], 20)

    def test_stock_recurrent_replay_identity_and_coverage(self):
        result = self.results["ppo_replay_identity"]
        coverage = result["coverage"]
        self.assertEqual(result["valid_timesteps"], result["expected_valid_timesteps"])
        self.assertEqual(result["valid_timesteps"], 20)
        self.assertTrue(result["every_transition_counted_once"])
        self.assertGreater(result["padding_timesteps"], 0)
        self.assertGreater(result["continuation_sequence_count"], 0)
        self.assertGreater(result["nonzero_pre_action_hidden_continuation_count"], 0)
        self.assertGreater(result["raw_nonzero_pre_action_state_count"], 0)
        self.assertLessEqual(result["continuation_hidden_oracle_max_error"], 1e-6)
        self.assertLessEqual(result["max_logp_absolute_error"], 1e-6)
        self.assertLessEqual(result["mean_logp_absolute_error"], 1e-7)
        self.assertLessEqual(result["max_ratio_deviation"], 1e-6)
        self.assertTrue(coverage["ordinary_continuous_sequence"])
        self.assertTrue(coverage["padding_mask"])
        self.assertGreater(coverage["timeout_event_count"], 0)
        self.assertGreater(coverage["ego_true_terminal_event_count"], 0)
        self.assertGreater(coverage["opponent_only_collision_continuation_count"], 0)

    def test_timeout_collision_bootstrap_and_advantage_boundary(self):
        result = self.results["timeout_bootstrap"]
        self.assertGreater(result["ego_collision_events"], 0)
        self.assertGreater(result["timeout_events"], 0)
        self.assertLessEqual(result["ego_collision_zero_bootstrap_max_error"], 1e-7)
        self.assertLessEqual(result["timeout_terminal_value_bootstrap_max_error"], 1e-6)
        self.assertLessEqual(result["terminal_advantage_no_cross_episode_max_error"], 1e-6)

    def test_gradient_ownership_and_optimizer_partition(self):
        result = self.results["gradient_and_optimizer_contract"]
        policy_loss = result["policy_loss"]
        value_loss = result["value_loss"]
        self.assertTrue(policy_loss["actor"]["finite"])
        self.assertEqual(policy_loss["actor"]["nonzero_count"], result["active_actor_parameter_count"])
        self.assertEqual(policy_loss["critic"]["nonzero_count"], 0)
        self.assertTrue(policy_loss["distribution"]["finite"])
        self.assertGreater(policy_loss["distribution"]["nonzero_count"], 0)
        self.assertEqual(value_loss["actor"]["nonzero_count"], 0)
        self.assertTrue(value_loss["critic"]["finite"])
        self.assertEqual(value_loss["critic"]["nonzero_count"], 4)
        self.assertEqual(value_loss["distribution"]["nonzero_count"], 0)
        self.assertTrue(result["dummy_embedding_policy_gradient_absent"])
        self.assertTrue(result["optimizer_parameter_identities_unique"])
        self.assertTrue(result["optimizer_parameters_fully_classified"])
        self.assertEqual(result["unused_inherited_action_net_parameter_count"], 0)
        self.assertEqual(result["opponent_planner_parameter_count"], 0)

    def test_actor_checkpoint_compatibility(self):
        result = self.results["checkpoint_compatibility"]
        self.assertTrue(result["strict_load_succeeded"])
        self.assertTrue(result["actor_keys_match_bc_schema"])
        self.assertEqual(result["actor_key_count"], 12)
        self.assertEqual(result["roundtrip_max_absolute_error"], 0.0)
        self.assertTrue(result["temporary_output_only"])

    def test_real_austin_f110_contract_smoke(self):
        result = self.results["real_f110_contract_smoke"]
        self.assertTrue(result["resources_available"])
        self.assertTrue(result["passed"], result.get("error"))
        self.assertGreaterEqual(result["auto_reset_count"], 1)
        self.assertTrue(result["terminal_observation_present"])
        self.assertTrue(result["time_limit_truncated"])
        self.assertTrue(all(shape == [2, 3] for shape in result["reset_pose_shapes"]))

    def test_zero_lr_stock_train_api_smoke(self):
        result = self.results["zero_lr_api_smoke"]
        self.assertEqual(result["parallel_envs"], 2)
        self.assertEqual(result["rollouts"], 1)
        self.assertEqual(result["ppo_train_calls"], 1)
        self.assertEqual(result["learning_rate"], 0.0)
        self.assertEqual(result["max_parameter_delta"], 0.0)
        self.assertFalse(self.results["end2race_learner_training_performed"])


if __name__ == "__main__":
    unittest.main()
