import unittest

import numpy as np

from generate_demo_repair import (
    EXPECTED_SAMPLES,
    build_npz_payload,
    derive_bc_training_arrays,
    downsample_bc_lidar,
)


class DemoRepairGenerationTests(unittest.TestCase):
    def test_downsample_matches_original_stride_four_contract(self):
        raw = np.arange(1440, dtype=np.float32)
        actual = downsample_bc_lidar(raw)
        np.testing.assert_array_equal(actual, raw[::4][:360])

    def test_npz_schema_and_train_alignment_match_train_py(self):
        times = np.arange(1, EXPECTED_SAMPLES + 1, dtype=np.float64) * 0.1
        steering = np.arange(EXPECTED_SAMPLES, dtype=np.float32)
        desired_speed = np.arange(EXPECTED_SAMPLES, dtype=np.float32) + 100.0
        lidar = np.repeat(
            np.arange(EXPECTED_SAMPLES, dtype=np.float32)[:, None], 360, axis=1
        )
        payload = build_npz_payload(times, steering, desired_speed, lidar)
        train_lidar, previous_speed, target_action = derive_bc_training_arrays(payload)
        self.assertEqual(train_lidar.shape, (79, 360))
        self.assertEqual(previous_speed.shape, (79, 1))
        self.assertEqual(target_action.shape, (79, 2))
        np.testing.assert_array_equal(train_lidar[:, 0], np.arange(1, 80))
        np.testing.assert_array_equal(previous_speed[:, 0], np.arange(100, 179))
        np.testing.assert_array_equal(target_action[:, 0], np.arange(1, 80))
        np.testing.assert_array_equal(target_action[:, 1], np.arange(101, 180))

    def test_schema_rejects_wrong_sample_count(self):
        with self.assertRaisesRegex(ValueError, "time_s must have shape"):
            build_npz_payload([], [], [], [])


if __name__ == "__main__":
    unittest.main()
