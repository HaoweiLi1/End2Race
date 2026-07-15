"""Read-only acceptance gates for the one-shot non-zero PPO smoke.

This module never imports or executes the smoke runner.  Re-running the real
optimizer update from unittest would violate the experiment contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import imageio.v2 as imageio
import numpy as np
import torch

from model import End2Race


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "sb3_nonzero_smoke"
RESULTS_PATH = ARTIFACT_DIR / "results.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestSB3NonzeroUpdateArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RESULTS_PATH.exists():
            raise AssertionError(
                "Run scripts/smoke_sb3_nonzero_update.py exactly once before this read-only test"
            )
        cls.results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    def test_01_existing_repair_regression_passed(self):
        self.assertTrue(self.results["repair_regression"]["passed"])
        self.assertEqual(self.results["repair_regression"]["expected_test_count"], 11)

    def test_02_unique_rollout_and_environment_steps(self):
        calls = self.results["explicit_call_counts"]
        self.assertEqual(calls["collect_rollouts"], 1)
        self.assertEqual(self.results["rollout"]["rollout_count"], 1)
        self.assertEqual(self.results["rollout"]["environment_step_count"], 100)
        self.assertEqual(self.results["replay_before_update"]["valid_transition_count"], 100)

    def test_03_unique_optimizer_step(self):
        calls = self.results["explicit_call_counts"]
        self.assertEqual(calls["train"], 1)
        self.assertEqual(calls["optimizer_step"], 1)
        self.assertEqual(self.results["train"]["adam_step_max"], 1.0)

    def test_04_learning_rate_is_exact_nonzero_value(self):
        learning_rate = self.results["ppo_configuration"]["learning_rate"]
        self.assertGreater(learning_rate, 0.0)
        self.assertEqual(learning_rate, 1e-6)
        self.assertEqual(self.results["ppo_configuration"]["n_epochs"], 1)
        self.assertEqual(self.results["replay_before_update"]["minibatch_count"], 1)

    def test_05_actor_critic_and_log_std_changed(self):
        delta = self.results["parameter_deltas"]
        self.assertGreater(delta["actor_max"], 0.0)
        self.assertGreater(max(delta["actor"]["gru"], delta["actor"]["output_layer"]), 0.0)
        self.assertGreater(delta["critic_max"], 0.0)
        self.assertGreater(delta["log_std_max"], 0.0)
        self.assertGreater(delta["global_max"], 0.0)
        self.assertLess(delta["global_max"], 1e-4)

    def test_06_dummy_embedding_unchanged(self):
        self.assertEqual(self.results["parameter_deltas"]["actor"]["dummy_embedding"], 0.0)

    def test_07_opponent_excluded_and_unchanged(self):
        opponent = self.results["opponent"]
        self.assertEqual(opponent["optimizer_overlap_count"], 0)
        self.assertEqual(opponent["parameter_max_delta"], 0.0)

    def test_08_preupdate_replay_identity(self):
        replay = self.results["replay_before_update"]
        self.assertLessEqual(replay["max_logp_absolute_error"], 1e-6)
        self.assertLessEqual(replay["mean_logp_absolute_error"], 1e-7)
        self.assertLessEqual(replay["max_ratio_deviation"], 1e-6)

    def test_09_postupdate_ratio_and_kl_finite(self):
        post = self.results["post_update_buffer"]
        self.assertTrue(post["all_finite"])
        self.assertTrue(np.isfinite(post["ratio_min"]))
        self.assertTrue(np.isfinite(post["ratio_max"]))
        self.assertTrue(np.isfinite(post["approximate_kl"]))
        self.assertLess(post["approximate_kl"], 1e-3)

    def test_10_action_trace_buffer_to_real_core(self):
        rollout = self.results["rollout"]
        self.assertEqual(rollout["buffer_action_shape"], [100, 1, 2])
        self.assertTrue(rollout["buffer_contains_only_ego_action"])
        self.assertEqual(rollout["sb3_pre_env_clipping_count"], 0)
        self.assertLessEqual(rollout["buffer_to_core_ego_action_max_error"], 1e-7)

    def test_11_training_rollout_video_is_readable(self):
        video = self.results["video"]
        path = ROOT / video["path"]
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)
        self.assertEqual(sha256_file(path), video["sha256"])
        reader = imageio.get_reader(path)
        try:
            first = reader.get_data(0)
            last = reader.get_data(video["decoded_frame_count"] - 1)
        finally:
            reader.close()
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(first.shape[-1], 3)
        self.assertEqual(last.shape, first.shape)
        self.assertEqual(video["decoded_frame_count"], video["captured_frame_count"])
        self.assertAlmostEqual(video["fps"], 100.0)
        self.assertLessEqual(
            abs(video["duration_seconds"] - video["expected_duration_from_recorded_steps"]),
            0.01,
        )

    def test_12_actor_only_checkpoint_strict_load(self):
        checkpoint = self.results["checkpoint"]
        path = ROOT / checkpoint["path"]
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = End2Race(mask_prob=0.0, hidden_scale=4)
        incompatible = model.load_state_dict(state, strict=True)
        self.assertEqual(len(state), 12)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in state.values()))

    def test_13_pretrained_checkpoint_hash_unchanged(self):
        checkpoint = self.results["checkpoint"]
        pretrained = ROOT / self.results["environment"]["pretrained_checkpoint"]
        self.assertEqual(checkpoint["pretrained_sha256_before"], checkpoint["pretrained_sha256_after"])
        self.assertEqual(sha256_file(pretrained), checkpoint["pretrained_sha256_before"])

    def test_14_no_nan_or_inf_and_final_verdict(self):
        self.assertTrue(self.results["rollout"]["all_rollout_fields_finite"])
        self.assertTrue(self.results["train"]["gradients_finite_before_clipping"])
        self.assertTrue(self.results["train"]["gradients_finite_after_clipping"])
        self.assertTrue(self.results["parameter_deltas"]["all_finite"])
        self.assertTrue(self.results["post_update_buffer"]["all_finite"])
        self.assertTrue(all(self.results["checks"].values()))
        self.assertEqual(self.results["verdict"], "PASS_FOR_TRAINING_PIPELINE_IMPLEMENTATION")


if __name__ == "__main__":
    unittest.main()
