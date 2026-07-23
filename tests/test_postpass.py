from __future__ import annotations

import unittest

import numpy as np

from scripts.postpass_reward_calculation import (
    bounded_postpass_reward,
    ego_induced_rear_closing,
    oriented_rectangle_vertices,
    postpass_penalty_basis,
    rear_gap_unsafe_fraction,
    rear_half_clearance,
    signed_rear_longitudinal_gap,
)


VEHICLE_LENGTH_M = 0.58
VEHICLE_WIDTH_M = 0.31


class PostPassGeometryTests(unittest.TestCase):

    def test_axis_aligned_signed_rear_gap_has_physical_surface_semantics(self):
        opponent = np.asarray((0.0, 0.0, 0.0))
        ego = np.asarray((1.20, 0.0, 0.0))

        gap = signed_rear_longitudinal_gap(
            ego,
            opponent,
            VEHICLE_LENGTH_M,
            VEHICLE_WIDTH_M,
        )

        self.assertAlmostEqual(gap, 1.20 - VEHICLE_LENGTH_M, places=12)
        self.assertLess(
            signed_rear_longitudinal_gap(
                np.asarray((0.30, 0.50, 0.0)),
                opponent,
                VEHICLE_LENGTH_M,
                VEHICLE_WIDTH_M,
            ),
            0.0,
        )

    def test_vertices_are_rear_to_front_and_have_expected_dimensions(self):
        vertices = oriented_rectangle_vertices(
            np.asarray((1.0, 2.0, 0.0)),
            VEHICLE_LENGTH_M,
            VEHICLE_WIDTH_M,
        )

        np.testing.assert_allclose(vertices[:2, 0], 1.0 - 0.5 * VEHICLE_LENGTH_M)
        np.testing.assert_allclose(vertices[2:, 0], 1.0 + 0.5 * VEHICLE_LENGTH_M)
        self.assertAlmostEqual(float(np.ptp(vertices[:, 1])), VEHICLE_WIDTH_M)

    def test_forward_motion_opens_rear_clearance_and_has_zero_closing(self):
        opponent = np.asarray((0.0, 0.0, 0.0))
        previous_ego = np.asarray((0.90, 0.50, 0.0))
        current_ego = np.asarray((0.95, 0.50, 0.0))

        result = ego_induced_rear_closing(
            previous_ego,
            current_ego,
            opponent,
            VEHICLE_LENGTH_M,
            VEHICLE_WIDTH_M,
        )

        self.assertGreater(
            result.current_clearance_m,
            result.counterfactual_previous_clearance_m,
        )
        self.assertEqual(result.closing_m, 0.0)

    def test_yaw_can_sweep_rear_half_toward_opponent_without_center_motion(self):
        opponent = np.asarray((0.20, 0.0, 0.0))
        previous_ego = np.asarray((0.70, 0.40, 0.0))
        current_ego = np.asarray((0.70, 0.40, 0.45))

        result = ego_induced_rear_closing(
            previous_ego,
            current_ego,
            opponent,
            VEHICLE_LENGTH_M,
            VEHICLE_WIDTH_M,
        )

        self.assertGreater(result.closing_m, 0.0)
        away = ego_induced_rear_closing(
            previous_ego,
            np.asarray((0.70, 0.40, -0.45)),
            opponent,
            VEHICLE_LENGTH_M,
            VEHICLE_WIDTH_M,
        )
        self.assertEqual(away.closing_m, 0.0)

    def test_opponent_motion_is_not_attributed_when_ego_pose_is_unchanged(self):
        current_opponent = np.asarray((0.20, 0.10, 0.0))
        ego = np.asarray((0.90, 0.55, 0.10))

        result = ego_induced_rear_closing(
            ego,
            ego,
            current_opponent,
            VEHICLE_LENGTH_M,
            VEHICLE_WIDTH_M,
        )

        self.assertEqual(result.closing_m, 0.0)

    def test_rear_half_clearance_is_zero_at_overlap(self):
        self.assertEqual(
            rear_half_clearance(
                np.asarray((0.30, 0.0, 0.0)),
                np.asarray((0.0, 0.0, 0.0)),
                VEHICLE_LENGTH_M,
                VEHICLE_WIDTH_M,
            ),
            0.0,
        )

    def test_penalty_basis_is_sparse_and_quadratically_gated(self):
        self.assertEqual(rear_gap_unsafe_fraction(0.60, 0.60), 0.0)
        self.assertEqual(rear_gap_unsafe_fraction(-0.10, 0.60), 1.0)
        self.assertAlmostEqual(rear_gap_unsafe_fraction(0.30, 0.60), 0.5)
        self.assertEqual(
            postpass_penalty_basis(0.01, 0.0, 0.60, active=False),
            0.0,
        )
        self.assertAlmostEqual(
            postpass_penalty_basis(0.01, 0.30, 0.60, active=True),
            0.0025,
        )

    def test_bounded_reward_respects_step_and_episode_caps(self):
        self.assertAlmostEqual(
            bounded_postpass_reward(0.005, 1.0, 0.02, 0.10, 0.20),
            -0.005,
        )
        self.assertAlmostEqual(
            bounded_postpass_reward(0.050, 1.0, 0.02, 0.10, 0.20),
            -0.02,
        )
        self.assertAlmostEqual(
            bounded_postpass_reward(0.050, 1.0, 0.02, 0.195, 0.20),
            -0.005,
        )
        self.assertEqual(
            bounded_postpass_reward(0.050, 1.0, 0.02, 0.20, 0.20),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
