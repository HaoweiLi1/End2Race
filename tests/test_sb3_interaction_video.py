"""Read-only semantic checks for the camera-corrected visual replay."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "sb3_nonzero_smoke" / "interaction_replay_results.json"


class TestSB3InteractionVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RESULTS.exists():
            raise AssertionError("Run the no-update interaction replay once before this read-only test")
        cls.result = json.loads(RESULTS.read_text(encoding="utf-8"))

    def test_visual_replay_provenance(self):
        provenance = self.result["provenance"]
        self.assertFalse(provenance["is_historical_training_rollout"])
        self.assertTrue(provenance["is_snapshot_visual_replay"])
        self.assertEqual(provenance["optimizer_updates"], 0)
        self.assertEqual(provenance["collect_rollouts_calls"], 0)
        self.assertEqual(provenance["train_calls"], 0)
        self.assertEqual(provenance["learn_calls"], 0)

    def test_both_cars_visible_in_every_raw_and_decoded_frame(self):
        video = self.result["video"]
        self.assertGreaterEqual(video["raw_vehicle_visibility"]["ego_center_yellow_pixels_min"], 8)
        self.assertGreaterEqual(video["raw_vehicle_visibility"]["opponent_red_pixels_min"], 8)
        self.assertGreaterEqual(video["decoded_vehicle_visibility"]["ego_center_yellow_pixels_min"], 8)
        self.assertGreaterEqual(video["decoded_vehicle_visibility"]["opponent_red_pixels_min"], 8)

    def test_video_container_and_frames(self):
        video = self.result["video"]
        path = ROOT / video["path"]
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)
        self.assertEqual(video["decoded_frame_count"], 100)
        self.assertEqual(video["fps"], 100.0)
        reader = imageio.get_reader(path)
        try:
            first = np.asarray(reader.get_data(0))
            last = np.asarray(reader.get_data(99))
        finally:
            reader.close()
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(first.shape, last.shape)
        self.assertEqual(first.shape[2], 3)

    def test_real_interaction_action_path(self):
        interaction = self.result["interaction"]
        self.assertEqual(interaction["real_environment_steps"], 100)
        self.assertEqual(interaction["done_step"], 100)
        self.assertFalse(interaction["ego_collision"])
        self.assertLessEqual(interaction["action_to_core_max_error"], 1e-7)
        self.assertLessEqual(interaction["historical_first_action_max_error"], 1e-7)

    def test_no_optimizer_or_parameter_update(self):
        evidence = self.result["no_update_evidence"]
        self.assertEqual(evidence["policy_max_parameter_delta"], 0.0)
        self.assertEqual(evidence["optimizer_state_entries_before"], 0)
        self.assertEqual(evidence["optimizer_state_entries_after"], 0)

    def test_historical_training_video_not_overwritten(self):
        historical = self.result["historical_video"]
        self.assertFalse(historical["overwritten"])
        self.assertEqual(historical["sha256_before"], historical["sha256_after"])

    def test_all_semantic_gates_pass(self):
        self.assertTrue(all(self.result["checks"].values()))
        self.assertEqual(self.result["verdict"], "PASS_VISUAL_REPLAY")


if __name__ == "__main__":
    unittest.main()
