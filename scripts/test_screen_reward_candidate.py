from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo.reward import anisotropic_risk_potential  # noqa: E402
from scripts.screen_reward_candidate import (  # noqa: E402
    DEFAULT_GAE_LAMBDA,
    DEFAULT_GAMMA,
    CandidateSpec,
    ComplianceError,
    Setting,
    check_compliance,
    clipped_policy_gradient_ceiling,
    gae_advantage_delta,
    normalized_perturbation,
    screen_candidate,
    select_setting,
)


def compliant_spec() -> CandidateSpec:
    return CandidateSpec(name="postpass_like", reward_terms_added=("postpass",))


class ComplianceGateTests(unittest.TestCase):
    """The two project requirements must be enforced in code, not in prose."""

    def test_compliant_reward_only_candidate_passes(self):
        self.assertEqual(check_compliance(compliant_spec()), ())

    def test_auxiliary_objective_is_rejected_as_multi_stage(self):
        for objective in ("imitation", "teacher", "distillation",
                          "second_stage_finetune"):
            with self.subTest(objective=objective):
                violations = check_compliance(
                    CandidateSpec(
                        name="follow_teacher",
                        reward_terms_added=("postpass",),
                        auxiliary_objectives=(objective,),
                    )
                )
                self.assertTrue(
                    any("single-stage PPO" in v for v in violations),
                    violations,
                )

    def test_unknown_auxiliary_objective_is_still_rejected(self):
        violations = check_compliance(
            CandidateSpec(
                name="mystery",
                reward_terms_added=("postpass",),
                auxiliary_objectives=("some_new_loss",),
            )
        )
        self.assertTrue(any("undeclared auxiliary objective" in v for v in violations))

    def test_runtime_mechanisms_are_rejected_as_shields(self):
        for mechanism in ("safety_shield", "action_override",
                          "action_post_processing", "runtime_gate",
                          "scheduled_intervention"):
            with self.subTest(mechanism=mechanism):
                violations = check_compliance(
                    CandidateSpec(
                        name="shielded",
                        reward_terms_added=("postpass",),
                        runtime_mechanisms=(mechanism,),
                    )
                )
                self.assertTrue(
                    any("model capability only" in v for v in violations),
                    violations,
                )

    def test_actor_input_contract_change_is_rejected(self):
        violations = check_compliance(
            CandidateSpec(
                name="wider_actor",
                reward_terms_added=("postpass",),
                actor_inputs=("lidar_360", "previous_measured_ego_speed",
                              "opponent_pose"),
            )
        )
        self.assertTrue(any("actor input contract changed" in v for v in violations))

    def test_future_and_privileged_information_are_rejected(self):
        future = check_compliance(
            CandidateSpec(
                name="oracle_like",
                reward_terms_added=("postpass",),
                uses_future_information=True,
            )
        )
        privileged = check_compliance(
            CandidateSpec(
                name="privileged_actor",
                reward_terms_added=("postpass",),
                uses_privileged_state_in_actor=True,
            )
        )
        self.assertTrue(any("future information" in v for v in future))
        self.assertTrue(any("privileged state to the actor" in v for v in privileged))

    def test_candidate_with_no_reward_term_has_nothing_to_screen(self):
        violations = check_compliance(CandidateSpec(name="empty"))
        self.assertTrue(any("nothing to screen" in v for v in violations))

    def test_screen_refuses_before_measuring(self):
        with self.assertRaises(ComplianceError):
            screen_candidate(
                CandidateSpec(
                    name="follow_teacher",
                    reward_terms_added=("postpass",),
                    auxiliary_objectives=("teacher",),
                ),
                np.zeros(4),
                np.asarray([True, False, False, False]),
                baseline_advantage_std=1.0,
                settings=(Setting("only", 1.0, 0.0, True),),
            )


class AdvantagePropagationTests(unittest.TestCase):

    def test_impulse_matches_closed_form(self):
        length = 8
        impulse_index = 5
        delta = np.zeros(length)
        delta[impulse_index] = 1.0
        starts = np.zeros(length, dtype=bool)
        starts[0] = True

        actual = gae_advantage_delta(delta, starts)

        decay = DEFAULT_GAMMA * DEFAULT_GAE_LAMBDA
        expected = np.asarray(
            [
                decay ** (impulse_index - t) if t <= impulse_index else 0.0
                for t in range(length)
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_credit_does_not_leak_across_episode_boundary(self):
        delta = np.zeros(6)
        delta[4] = 1.0                      # inside the second episode
        starts = np.asarray([True, False, False, True, False, False])

        actual = gae_advantage_delta(delta, starts)

        self.assertTrue(np.all(actual[:3] == 0.0))
        decay = DEFAULT_GAMMA * DEFAULT_GAE_LAMBDA
        np.testing.assert_allclose(actual[3], decay, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(actual[4], 1.0, rtol=0.0, atol=1e-15)
        self.assertEqual(actual[5], 0.0)

    def test_superposition_holds(self):
        starts = np.asarray([True] + [False] * 7)
        a = np.zeros(8); a[2] = 0.4
        b = np.zeros(8); b[6] = -0.9

        np.testing.assert_allclose(
            gae_advantage_delta(a + b, starts),
            gae_advantage_delta(a, starts) + gae_advantage_delta(b, starts),
            rtol=0.0,
            atol=1e-15,
        )

    def test_rejects_mismatched_lengths_and_bad_decay(self):
        with self.assertRaises(ValueError):
            gae_advantage_delta(np.zeros(3), np.zeros(4, dtype=bool))
        with self.assertRaises(ValueError):
            gae_advantage_delta(np.zeros(3), np.zeros(3, dtype=bool), gamma=0.0)
        with self.assertRaises(ValueError):
            gae_advantage_delta(np.asarray([np.nan]), np.asarray([True]))


class NormalizationTests(unittest.TestCase):
    """The screen's central claim: sparsity, not total magnitude, sets the signal."""

    def test_sparse_small_term_can_outweigh_a_dense_larger_term(self):
        length = 1000
        starts = np.zeros(length, dtype=bool); starts[0] = True

        sparse = np.zeros(length)
        sparse[500] = -0.005                       # one step, tiny absolute size
        dense = np.full(length, -0.005 / length)   # same total, spread out

        sparse_report = normalized_perturbation(
            gae_advantage_delta(sparse, starts), baseline_advantage_std=0.05
        )
        dense_report = normalized_perturbation(
            gae_advantage_delta(dense, starts), baseline_advantage_std=0.05
        )

        # Same reward mass spread over one step versus every step: the totals are
        # comparable, so any difference in learning signal comes from sparsity.
        self.assertLess(
            abs(sparse_report["absolute_total"] - dense_report["absolute_total"])
            / dense_report["absolute_total"],
            0.25,
        )
        self.assertGreater(
            sparse_report["normalized_maximum"],
            dense_report["normalized_maximum"],
        )
        self.assertLess(sparse_report["touched_fraction"], 1.0)

    def test_normalized_maximum_scales_inversely_with_baseline_spread(self):
        delta = np.asarray([-0.01])
        starts = np.asarray([True])
        tight = normalized_perturbation(delta, 0.01)["normalized_maximum"]
        loose = normalized_perturbation(delta, 1.00)["normalized_maximum"]
        self.assertAlmostEqual(tight / loose, 100.0, places=9)

    def test_zero_delta_reports_no_signal(self):
        report = normalized_perturbation(np.zeros(5), 0.1)
        self.assertEqual(report["normalized_maximum"], 0.0)
        self.assertEqual(report["touched_fraction"], 0.0)

    def test_rejects_nonpositive_baseline_std(self):
        for bad in (0.0, -1.0, float("nan")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalized_perturbation(np.asarray([1.0]), bad)

    def test_gradient_ceiling_uses_clip_range(self):
        self.assertAlmostEqual(
            clipped_policy_gradient_ceiling(2.0, clip_range=0.20), 2.4, places=12
        )
        with self.assertRaises(ValueError):
            clipped_policy_gradient_ceiling(1.0, clip_range=0.0)


class SelectionTests(unittest.TestCase):

    def test_accepted_settings_prefer_selectivity_over_capture(self):
        result = select_setting(
            (
                Setting("broad", target_capture_rate=1.00,
                        guardrail_trigger_rate=0.19, acceptance_pass=True),
                Setting("selective", target_capture_rate=0.92,
                        guardrail_trigger_rate=0.03, acceptance_pass=True),
            )
        )
        self.assertEqual(result["setting"], "selective")
        self.assertTrue(result["selected_from_accepted_set"])

    def test_rejected_fallback_takes_capture_and_is_flagged(self):
        result = select_setting(
            (
                Setting("high_capture", target_capture_rate=0.80,
                        guardrail_trigger_rate=0.50, acceptance_pass=False),
                Setting("low_capture", target_capture_rate=0.70,
                        guardrail_trigger_rate=0.01, acceptance_pass=False),
            )
        )
        self.assertEqual(result["setting"], "high_capture")
        self.assertFalse(result["selected_from_accepted_set"])

    def test_rejects_out_of_range_rates(self):
        with self.assertRaises(ValueError):
            select_setting((Setting("bad", 1.5, 0.0, True),))
        with self.assertRaises(ValueError):
            select_setting(())


class EndToEndScreenTests(unittest.TestCase):

    def _screen(self, settings, delta_value=-0.005):
        length = 200
        starts = np.zeros(length, dtype=bool); starts[0] = True
        delta = np.zeros(length); delta[150] = delta_value
        return screen_candidate(
            compliant_spec(),
            delta,
            starts,
            baseline_advantage_std=0.05,
            settings=settings,
        )

    def test_selective_accepted_candidate_is_ready(self):
        report = self._screen(
            (Setting("selective", 0.92, 0.03, True),)
        )
        self.assertTrue(report["ready_for_training_ab"])
        self.assertTrue(report["acceptance_checks"]["learning_signal_is_measurable"])
        self.assertGreater(report["policy_gradient_ceiling"], 0.0)

    def test_guardrail_overrun_blocks_readiness(self):
        report = self._screen((Setting("broad", 1.00, 0.19, True),))
        self.assertFalse(report["ready_for_training_ab"])
        self.assertFalse(
            report["acceptance_checks"]["guardrail_trigger_rate_within_budget"]
        )

    def test_fallback_outside_accepted_set_blocks_readiness(self):
        report = self._screen((Setting("high_capture", 0.80, 0.02, False),))
        self.assertFalse(report["ready_for_training_ab"])
        self.assertFalse(report["acceptance_checks"]["selected_from_accepted_set"])

    def test_zero_reward_delta_is_not_ready(self):
        report = self._screen((Setting("selective", 0.92, 0.03, True),),
                              delta_value=0.0)
        self.assertFalse(report["ready_for_training_ab"])
        self.assertFalse(report["acceptance_checks"]["learning_signal_is_measurable"])

    def test_report_records_the_offline_only_scope(self):
        report = self._screen((Setting("selective", 0.92, 0.03, True),))
        self.assertIn("does not show that training will improve safety",
                      report["scope"])


class ProductionRewardUnchangedTests(unittest.TestCase):
    """The screen must not alter or shadow the production reward."""

    def test_production_risk_potential_is_untouched_and_still_matches_contract(self):
        rng = np.random.default_rng(20260730)
        for _ in range(64):
            longitudinal = float(rng.uniform(0.0, 2.0))
            lateral = float(rng.uniform(0.0, 0.8))
            wall = float(rng.uniform(0.0, 0.8))
            value = anisotropic_risk_potential(
                longitudinal,
                lateral,
                wall,
                longitudinal_safe_m=0.6,
                lateral_safe_m=0.2,
                wall_safe_m=0.2,
                maximum_magnitude=0.05,
            )
            self.assertLessEqual(value, 0.0)
            self.assertGreaterEqual(value, -0.05)


# ---------------------------------------------------------------------------
# Migrated oracle: geometry, reward state machine, cross-check protocol
# ---------------------------------------------------------------------------

from scripts.screen_reward_candidate import (  # noqa: E402
    PostpassState,
    RewardConfig,
    bounded_negative_reward,
    clipped_ppo_policy_loss,
    crosscheck_geometry,
    ego_induced_rear_closing,
    fixed_prediction_value_loss_delta,
    mean_squared_value_loss,
    normalized_advantages,
    oriented_rectangle_vertices,
    postpass_reward_step,
    rear_half_clearance,
    rectangle_clearance,
    signed_rear_longitudinal_gap,
    wilson_interval,
)

LENGTH_M = 0.58
WIDTH_M = 0.31


class OracleGeometryTests(unittest.TestCase):

    def test_axis_aligned_signed_rear_gap_has_surface_semantics(self):
        gap = signed_rear_longitudinal_gap(
            (1.20, 0.0, 0.0), (0.0, 0.0, 0.0), LENGTH_M, WIDTH_M
        )
        self.assertAlmostEqual(gap, 1.20 - LENGTH_M, places=12)
        self.assertLess(
            signed_rear_longitudinal_gap(
                (0.30, 0.50, 0.0), (0.0, 0.0, 0.0), LENGTH_M, WIDTH_M
            ),
            0.0,
        )

    def test_vertices_are_rear_to_front_with_expected_extent(self):
        vertices = oriented_rectangle_vertices((1.0, 2.0, 0.0), LENGTH_M, WIDTH_M)
        np.testing.assert_allclose(vertices[:2, 0], 1.0 - 0.5 * LENGTH_M)
        np.testing.assert_allclose(vertices[2:, 0], 1.0 + 0.5 * LENGTH_M)
        self.assertAlmostEqual(float(np.ptp(vertices[:, 1])), WIDTH_M)

    def test_clearance_is_zero_at_overlap_and_positive_when_separated(self):
        self.assertEqual(
            rear_half_clearance((0.30, 0.0, 0.0), (0.0, 0.0, 0.0), LENGTH_M, WIDTH_M),
            0.0,
        )
        self.assertGreater(
            rear_half_clearance((3.0, 0.0, 0.0), (0.0, 0.0, 0.0), LENGTH_M, WIDTH_M),
            0.0,
        )

    def test_clearance_is_invariant_under_rigid_transform(self):
        a = oriented_rectangle_vertices((0.0, 0.0, 0.3), LENGTH_M, WIDTH_M)
        b = oriented_rectangle_vertices((1.4, 0.2, -0.2), LENGTH_M, WIDTH_M)
        baseline = rectangle_clearance(a, b)
        angle = 0.7
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        shift = np.asarray((3.1, -2.4))
        moved = rectangle_clearance(a @ rotation.T + shift, b @ rotation.T + shift)
        self.assertAlmostEqual(baseline, moved, places=12)

    def test_forward_motion_opens_clearance_with_zero_closing(self):
        result = ego_induced_rear_closing(
            (0.90, 0.50, 0.0), (0.95, 0.50, 0.0), (0.0, 0.0, 0.0), LENGTH_M, WIDTH_M
        )
        self.assertGreater(
            result.current_clearance_m, result.counterfactual_previous_clearance_m
        )
        self.assertEqual(result.closing_m, 0.0)

    def test_yaw_sweep_closes_without_center_translation(self):
        toward = ego_induced_rear_closing(
            (0.70, 0.40, 0.0), (0.70, 0.40, 0.45), (0.20, 0.0, 0.0), LENGTH_M, WIDTH_M
        )
        away = ego_induced_rear_closing(
            (0.70, 0.40, 0.0), (0.70, 0.40, -0.45), (0.20, 0.0, 0.0), LENGTH_M, WIDTH_M
        )
        self.assertGreater(toward.closing_m, 0.0)
        self.assertEqual(away.closing_m, 0.0)

    def test_opponent_motion_is_never_charged_to_the_ego(self):
        ego = (0.90, 0.55, 0.10)
        result = ego_induced_rear_closing(
            ego, ego, (0.20, 0.10, 0.0), LENGTH_M, WIDTH_M
        )
        self.assertEqual(result.closing_m, 0.0)


class OracleRewardStateMachineTests(unittest.TestCase):

    def _step(self, state, *, latched=False, previous=-0.30, current=0.10):
        return postpass_reward_step(
            previous_relative_progress_m=previous,
            current_relative_progress_m=current,
            previous_ego_pose=(0.0, 0.0, 0.0),
            current_ego_pose=(0.4, 0.0, 0.0),
            current_opponent_pose=(0.3, 0.0, 0.0),
            opponent_collision_latched=latched,
            transition_dt_s=0.01,
            config=RewardConfig(),
            state=state,
        )

    def test_phase_entry_triggers_a_negative_reward(self):
        result = self._step(PostpassState())
        self.assertTrue(result.entered)
        self.assertTrue(result.triggered)
        self.assertLess(result.reward, 0.0)

    def test_prepass_transition_yields_nothing(self):
        result = self._step(PostpassState(), previous=-0.30, current=-0.20)
        self.assertFalse(result.entered)
        self.assertFalse(result.triggered)
        self.assertEqual(result.reward, 0.0)

    def test_opponent_collision_suppresses_the_phase(self):
        result = self._step(PostpassState(), latched=True)
        self.assertTrue(result.entered)
        self.assertFalse(result.phase_active)
        self.assertFalse(result.triggered)
        self.assertEqual(result.reward, 0.0)

    def test_safe_rear_gap_clears_the_phase_permanently(self):
        state = PostpassState(entered=True)
        cleared = postpass_reward_step(
            previous_relative_progress_m=0.10,
            current_relative_progress_m=0.20,
            previous_ego_pose=(3.0, 0.0, 0.0),
            current_ego_pose=(3.0, 0.0, 0.0),
            current_opponent_pose=(0.0, 0.0, 0.0),
            opponent_collision_latched=False,
            transition_dt_s=0.01,
            config=RewardConfig(),
            state=state,
        )
        self.assertTrue(cleared.cleared)
        self.assertFalse(cleared.phase_active)

    def test_episode_cap_binds_and_reset_clears_it(self):
        config = RewardConfig()
        state = PostpassState()
        for index in range(40):
            self._step(state, previous=-0.30 if index == 0 else 0.10)
        self.assertAlmostEqual(
            state.penalty_used, config.maximum_episode_penalty, places=12
        )
        fresh = PostpassState()
        self.assertFalse(fresh.entered)
        self.assertEqual(fresh.penalty_used, 0.0)

    def test_bounded_reward_respects_step_and_remaining_caps(self):
        config = RewardConfig(reward_weight_per_m=1.0, maximum_step_penalty=0.02,
                              maximum_episode_penalty=0.20)
        self.assertAlmostEqual(
            bounded_negative_reward(0.005, config, PostpassState()), -0.005, places=12
        )
        self.assertAlmostEqual(
            bounded_negative_reward(0.050, config, PostpassState()), -0.02, places=12
        )
        self.assertAlmostEqual(
            bounded_negative_reward(0.050, config, PostpassState(penalty_used=0.195)),
            -0.005,
            places=12,
        )
        self.assertEqual(
            bounded_negative_reward(0.050, config, PostpassState(penalty_used=0.20)),
            0.0,
        )

    def test_config_rejects_invalid_signs(self):
        for kwargs in ({"vehicle_length_m": 0.0}, {"safe_rear_gap_m": -1.0},
                       {"activation_clearance_m": 0.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RewardConfig(**kwargs)

    def test_sequential_state_machine_matches_vectorized_formula(self):
        config = RewardConfig(
            activation_clearance_m=0.20,
            maximum_ttc_s=0.75,
        )
        relative = np.asarray((-0.10, 0.00, 0.06, 0.10, 0.15))
        ego_poses = np.asarray(
            (
                (0.0, 0.70, 0.0),
                (0.0, 0.55, 0.0),
                (0.0, 0.35, 0.0),
                (0.0, 0.25, 0.0),
                (0.0, 0.20, 0.0),
            ),
            dtype=np.float64,
        )
        opponent_poses = np.zeros_like(ego_poses)
        dt = 0.10

        vector_state = PostpassState()
        vector_reward = np.zeros(len(relative), dtype=np.float64)
        vector_trigger = np.zeros(len(relative), dtype=bool)
        for index in range(2, len(relative)):
            gap = signed_rear_longitudinal_gap(
                ego_poses[index],
                opponent_poses[index],
                config.vehicle_length_m,
                config.vehicle_width_m,
            )
            closing = ego_induced_rear_closing(
                ego_poses[index - 1],
                ego_poses[index],
                opponent_poses[index],
                config.vehicle_length_m,
                config.vehicle_width_m,
            )
            closing_speed = closing.closing_m / dt
            closing_time = (
                closing.current_clearance_m / closing_speed
                if closing_speed > 0.0
                else float("inf")
            )
            unsafe = np.clip(
                (config.safe_rear_gap_m - gap) / config.safe_rear_gap_m,
                0.0,
                1.0,
            )
            proximity = np.clip(
                (
                    config.activation_clearance_m
                    - closing.current_clearance_m
                )
                / config.activation_clearance_m,
                0.0,
                1.0,
            )
            triggered = bool(
                closing.current_clearance_m < config.activation_clearance_m
                and closing_time <= config.maximum_ttc_s
                and closing_speed > config.closing_deadband_mps
            )
            vector_trigger[index] = triggered
            if triggered:
                basis = closing.closing_m * unsafe**2 * proximity**2
                vector_reward[index] = bounded_negative_reward(
                    basis,
                    config,
                    vector_state,
                )

        sequential_state = PostpassState()
        sequential_reward = np.zeros(len(relative), dtype=np.float64)
        sequential_trigger = np.zeros(len(relative), dtype=bool)
        for index in range(1, len(relative)):
            step = postpass_reward_step(
                previous_relative_progress_m=float(relative[index - 1]),
                current_relative_progress_m=float(relative[index]),
                previous_ego_pose=ego_poses[index - 1],
                current_ego_pose=ego_poses[index],
                current_opponent_pose=opponent_poses[index],
                opponent_collision_latched=False,
                transition_dt_s=dt,
                config=config,
                state=sequential_state,
            )
            sequential_reward[index] = step.reward
            sequential_trigger[index] = step.triggered

        np.testing.assert_allclose(
            sequential_reward,
            vector_reward,
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_array_equal(sequential_trigger, vector_trigger)
        self.assertAlmostEqual(
            sequential_state.penalty_used,
            vector_state.penalty_used,
            places=14,
        )


class CrosscheckProtocolTests(unittest.TestCase):
    """The protocol must agree with itself and must detect a real divergence."""

    def test_identical_implementation_gives_exactly_zero_error(self):
        import scripts.screen_reward_candidate as oracle

        report = crosscheck_geometry(oracle, case_count=64)
        for key in ("signed_gap", "current_clearance",
                    "counterfactual_clearance", "closing", "bounded_reward"):
            self.assertEqual(report[key], 0.0, key)
        self.assertEqual(report["case_count"], 64)

    def test_protocol_detects_a_perturbed_candidate(self):
        import scripts.screen_reward_candidate as oracle

        class Perturbed:
            @staticmethod
            def signed_rear_longitudinal_gap(*args, **kwargs):
                return oracle.signed_rear_longitudinal_gap(*args, **kwargs) + 1e-3

            rear_half_clearance = staticmethod(oracle.rear_half_clearance)
            ego_induced_rear_closing = staticmethod(oracle.ego_induced_rear_closing)
            bounded_negative_reward = staticmethod(oracle.bounded_negative_reward)
            RewardConfig = oracle.RewardConfig
            PostpassState = oracle.PostpassState

        report = crosscheck_geometry(Perturbed, case_count=16)
        self.assertAlmostEqual(report["signed_gap"], 1e-3, places=12)
        self.assertEqual(report["closing"], 0.0)

    def test_rejects_nonpositive_case_count(self):
        import scripts.screen_reward_candidate as oracle

        with self.assertRaises(ValueError):
            crosscheck_geometry(oracle, case_count=0)


class PpoObjectiveTests(unittest.TestCase):

    def test_normalization_uses_bessel_correction(self):
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        actual = normalized_advantages(values)
        expected = (values - values.mean()) / (values.std(ddof=1) + 1e-8)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
        # Population std would give a visibly different scale.
        population = (values - values.mean()) / (values.std(ddof=0) + 1e-8)
        self.assertGreater(float(np.abs(actual - population).max()), 1e-3)

    def test_masked_normalization_leaves_invalid_samples_untouched(self):
        values = np.asarray([1.0, 2.0, 3.0, 99.0])
        mask = np.asarray([True, True, True, False])
        actual = normalized_advantages(values, mask)
        self.assertEqual(actual[3], 99.0)
        self.assertAlmostEqual(float(actual[:3].mean()), 0.0, places=12)

    def test_normalization_requires_two_valid_samples(self):
        with self.assertRaises(ValueError):
            normalized_advantages(np.asarray([1.0, 2.0]),
                                  np.asarray([True, False]))

    def test_clipped_surrogate_matches_manual_case(self):
        advantage = np.asarray([1.0, -1.0])
        ratio = np.asarray([1.5, 1.5])
        loss, samples = clipped_ppo_policy_loss(
            advantage, ratio, clip_range=0.2, normalize_advantage=False
        )
        # Positive advantage is clipped at 1.2; negative advantage is not.
        np.testing.assert_allclose(samples, np.asarray([-1.2, 1.5]), atol=1e-12)
        self.assertAlmostEqual(loss, float(np.mean([-1.2, 1.5])), places=12)

    def test_clipped_surrogate_rejects_nonpositive_ratio(self):
        with self.assertRaises(ValueError):
            clipped_ppo_policy_loss(np.asarray([1.0]), np.asarray([0.0]))

    def test_value_loss_and_fixed_prediction_delta_are_consistent(self):
        predictions = np.asarray([0.5, -0.2, 1.0])
        baseline = np.asarray([0.4, 0.0, 0.8])
        delta = np.asarray([0.1, -0.3, 0.0])
        base_loss, _ = mean_squared_value_loss(predictions, baseline)
        treatment_loss, _ = mean_squared_value_loss(predictions, baseline + delta)
        mean_delta, _ = fixed_prediction_value_loss_delta(
            predictions, baseline, delta
        )
        self.assertAlmostEqual(mean_delta, treatment_loss - base_loss, places=12)

    def test_zero_return_delta_changes_no_value_loss(self):
        predictions = np.asarray([0.1, 0.2])
        baseline = np.asarray([0.0, 0.3])
        mean_delta, samples = fixed_prediction_value_loss_delta(
            predictions, baseline, np.zeros(2)
        )
        self.assertEqual(mean_delta, 0.0)
        np.testing.assert_allclose(samples, np.zeros(2), atol=0.0)


class WilsonIntervalTests(unittest.TestCase):

    def test_empty_total_returns_none(self):
        self.assertEqual(wilson_interval(0, 0), (None, None))

    def test_boundary_rates_stay_inside_the_unit_range(self):
        low, high = wilson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertLess(high, 1.0)
        low, high = wilson_interval(10, 10)
        self.assertGreater(low, 0.0)
        # At p = 1 the analytic upper bound is exactly 1.0; allow float rounding.
        self.assertAlmostEqual(high, 1.0, places=12)
        self.assertLessEqual(high, 1.0)

    def test_interval_brackets_the_point_estimate_and_narrows_with_n(self):
        for total in (20, 2000):
            low, high = wilson_interval(total // 2, total)
            self.assertLess(low, 0.5)
            self.assertGreater(high, 0.5)
        narrow = wilson_interval(1000, 2000)
        wide = wilson_interval(10, 20)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_rejects_successes_outside_total(self):
        with self.assertRaises(ValueError):
            wilson_interval(11, 10)


if __name__ == "__main__":
    unittest.main()


class PreregisteredThresholdTests(unittest.TestCase):
    """The historical acceptance bar and gate grid must not be silently redefined."""

    def test_thresholds_match_the_retired_probe(self):
        from scripts.screen_reward_candidate import (
            ACCEPT_MAX_FOLLOW_TRIGGER,
            ACCEPT_MAX_OVERTAKE_TRIGGER,
            ACCEPT_MIN_TAIL_CAPTURE,
        )

        self.assertEqual(ACCEPT_MIN_TAIL_CAPTURE, 0.90)
        self.assertEqual(ACCEPT_MAX_OVERTAKE_TRIGGER, 0.20)
        self.assertEqual(ACCEPT_MAX_FOLLOW_TRIGGER, 0.01)

    def test_gate_grid_has_41_unique_settings_including_the_shipped_point(self):
        from scripts.screen_reward_candidate import postpass_gate_sweep_grid

        grid = postpass_gate_sweep_grid()
        self.assertEqual(len(grid), 41)
        keys = {
            (c.activation_clearance_m, c.maximum_ttc_s, c.closing_deadband_mps)
            for c in grid
        }
        self.assertEqual(len(keys), 41)
        self.assertIn((None, None, 0.10), keys)          # ungated baseline
        self.assertIn((0.20, 0.75, 0.10), keys)          # shipped configuration
        self.assertIn((0.30, 1.00, 0.05), keys)          # widest point tried

    def test_per_panel_guard_uses_the_worst_panel_not_the_average(self):
        from scripts.screen_reward_candidate import per_panel_acceptance

        # Mean overtake trigger is 0.105, inside the 0.20 budget, but one panel is
        # at 0.19 and one follow panel breaches the 0.01 budget.
        result = per_panel_acceptance([0.02, 0.19], [0.001, 0.05])
        self.assertAlmostEqual(result["maximum_panel_overtake_trigger_rate"], 0.19)
        self.assertAlmostEqual(result["maximum_panel_follow_trigger_rate"], 0.05)
        self.assertFalse(result["panel_selectivity_guard_pass"])

    def test_per_panel_guard_passes_when_every_panel_is_inside_budget(self):
        from scripts.screen_reward_candidate import per_panel_acceptance

        result = per_panel_acceptance([0.02, 0.19], [0.001, 0.009])
        self.assertTrue(result["panel_selectivity_guard_pass"])

    def test_per_panel_guard_rejects_empty_and_out_of_range(self):
        from scripts.screen_reward_candidate import per_panel_acceptance

        with self.assertRaises(ValueError):
            per_panel_acceptance([], [0.0])
        with self.assertRaises(ValueError):
            per_panel_acceptance([1.5], [0.0])
