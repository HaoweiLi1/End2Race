"""Contract tests for the preregistered Simple PPO V1.2 infrastructure."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gymnasium import spaces
import gymnasium as gym
import numpy as np
from sb3_contrib.common.recurrent.type_aliases import RNNStates
import torch

from experiments.ppo_v1_2.config_schema import CRITIC_PROFILES, STAGE_COUNTS
from experiments.ppo_v1_2.experiment_spec import canonical_hash
from experiments.ppo_v1_2.hard_pool_builder import (
    PASS1_SEEDS,
    PASS2_SEEDS,
    classify_stochastic_rows,
    deterministic_expanded_startpoints,
    expanded_scenarios,
    union_pool,
    validate_candidates,
)
from experiments.ppo_v1_2.registry import build_arms, build_manifest, validate_manifest
from experiments.ppo_v1_2.runner import Heartbeat, SweepLock, SweepRunner, remaining_attempts
from experiments.ppo_v1_2.selectors import checkpoint_flags, rank_arms, select_checkpoint
from model import End2Race
from rl.end2race_recurrent_ppo import End2RaceRecurrentPPO
from rl.ppo_scenarios import FixedMixtureScenarioSampler, training_scenarios
from rl.sb3_end2race_policy import END2RACE_OBSERVATION_SIZE, End2RaceGRUPolicy, NOOP_SPEED_BOUND


def action_space() -> spaces.Box:
    return spaces.Box(
        np.asarray([-0.52, -NOOP_SPEED_BOUND], dtype=np.float32),
        np.asarray([0.52, NOOP_SPEED_BOUND], dtype=np.float32),
        dtype=np.float32,
    )


def policy(profile: str) -> End2RaceGRUPolicy:
    actor = spaces.Box(-np.inf, np.inf, shape=(END2RACE_OBSERVATION_SIZE,), dtype=np.float32)
    observation = spaces.Dict({"actor": actor, "critic": spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)}) if profile == "C3_PRIVILEGED_PHYSICAL" else actor
    return End2RaceGRUPolicy(observation, action_space(), lambda _: 1.0, optimizer_profile="ppo_v1", critic_profile=profile)


def states(batch: int = 2) -> RNNStates:
    pair = (torch.zeros(1, batch, 1680), torch.zeros(1, batch, 1680))
    return RNNStates(pair, pair)


class TestCriticProfiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policies = {name: policy(name) for name in CRITIC_PROFILES}
        cls.actor_obs = torch.linspace(0.1, 4.0, steps=2 * 361).reshape(2, 361)
        cls.episode_starts = torch.zeros(2)

    def observation(self, profile: str, critic: torch.Tensor | None = None):
        if profile == "C3_PRIVILEGED_PHYSICAL":
            return {"actor": self.actor_obs.clone(), "critic": torch.zeros(2, 12) if critic is None else critic}
        return self.actor_obs.clone()

    def test_actor_action_hidden_identity_for_all_profiles(self):
        outputs = {}
        for name, instance in self.policies.items():
            actions, _values, _log_prob, next_states = instance.forward(
                self.observation(name), states(), self.episode_starts, deterministic=True
            )
            outputs[name] = (actions.detach(), next_states.pi[0].detach())
        reference = outputs["C0_RAW_SINGLE_FRAME"]
        for name, output in outputs.items():
            torch.testing.assert_close(output[0], reference[0], rtol=0.0, atol=0.0, msg=name)
            torch.testing.assert_close(output[1], reference[1], rtol=0.0, atol=0.0, msg=name)

    def test_value_loss_has_no_actor_or_frozen_feature_gradient(self):
        for name in ("C1_FROZEN_BC_FEATURE", "C2_DETACHED_ACTOR_HIDDEN"):
            instance = self.policies[name]
            instance.optimizer.zero_grad()
            obs = self.observation(name)
            if name.startswith("C2"):
                _means, _next, actor_features = instance._actor_forward(obs, states().pi, self.episode_starts)
            else:
                actor_features = None
            instance._critic_values(obs, actor_features).square().mean().backward()
            self.assertTrue(all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0 for parameter in instance.end2race_actor.parameters()), name)
            self.assertTrue(any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0 for parameter in instance.value_net.parameters()), name)

    def test_c2_timeout_bootstrap_uses_transport_hidden(self):
        instance = self.policies["C2_DETACHED_ACTOR_HIDDEN"]
        transport = states(batch=1).pi
        transport = (torch.full_like(transport[0], 0.125), torch.zeros_like(transport[1]))
        observation = self.actor_obs[:1]
        predicted = instance.predict_values(observation, transport, torch.zeros(1))
        _means, _next, actor_features = instance._actor_forward(observation, transport, torch.zeros(1))
        expected = instance.value_net(actor_features.detach())
        torch.testing.assert_close(predicted, expected, rtol=0.0, atol=0.0)

    def test_c3_actor_critic_field_isolation(self):
        instance = self.policies["C3_PRIVILEGED_PHYSICAL"]
        critic_a = torch.zeros(2, 12)
        critic_b = torch.ones(2, 12)
        action_a, value_a, _logp_a, hidden_a = instance.forward(self.observation("C3_PRIVILEGED_PHYSICAL", critic_a), states(), self.episode_starts, deterministic=True)
        action_b, value_b, _logp_b, hidden_b = instance.forward(self.observation("C3_PRIVILEGED_PHYSICAL", critic_b), states(), self.episode_starts, deterministic=True)
        torch.testing.assert_close(action_a, action_b, rtol=0.0, atol=0.0)
        torch.testing.assert_close(hidden_a.pi[0], hidden_b.pi[0], rtol=0.0, atol=0.0)
        self.assertGreater(float((value_a - value_b).abs().max()), 0.0)
        changed_actor = {"actor": self.actor_obs + 0.25, "critic": critic_a}
        _changed_action, changed_value, _changed_logp, _changed_hidden = instance.forward(changed_actor, states(), self.episode_starts, deterministic=True)
        torch.testing.assert_close(value_a, changed_value, rtol=0.0, atol=0.0)

    def test_all_profiles_export_exact_strict_twelve_keys(self):
        for name, instance in self.policies.items():
            state = instance.actor_checkpoint_state_dict()
            self.assertEqual(len(state), 12, name)
            loaded = End2Race(mask_prob=0.0, hidden_scale=4).load_state_dict(state, strict=True)
            self.assertEqual((loaded.missing_keys, loaded.unexpected_keys), ([], []), name)

    def test_c3_uses_stock_recurrent_dict_rollout_buffer(self):
        class TinyDictEnv(gym.Env):
            def __init__(self):
                actor = spaces.Box(-np.inf, np.inf, shape=(361,), dtype=np.float32)
                self.observation_space = spaces.Dict({"actor": actor, "critic": spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)})
                self.action_space = action_space()
                self.steps = 0

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.steps = 0
                return {"actor": np.ones(361, dtype=np.float32), "critic": np.zeros(12, dtype=np.float32)}, {}

            def step(self, action):
                self.steps += 1
                observation = {"actor": np.full(361, 1.0 + self.steps / 100.0, dtype=np.float32), "critic": np.full(12, self.steps / 100.0, dtype=np.float32)}
                return observation, float(action[1] * 0.01), False, self.steps >= 4, {}

        env = TinyDictEnv()
        model = End2RaceRecurrentPPO(
            End2RaceGRUPolicy,
            env,
            learning_rate=1.0,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            policy_kwargs={"optimizer_profile": "ppo_v1", "critic_profile": "C3_PRIVILEGED_PHYSICAL", "gru_lr": 0.0, "head_lr": 0.0, "critic_lr": 0.0},
            verbose=0,
        )
        model.learn(total_timesteps=4)
        self.assertEqual(type(model.rollout_buffer).__name__, "RecurrentDictRolloutBuffer")
        self.assertTrue(np.isfinite(model.rollout_buffer.returns).all())
        env.close()


class TestHardPools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.startpoints = deterministic_expanded_startpoints()
        cls.expanded = expanded_scenarios(cls.startpoints)

    def test_expanded_grid_is_deterministic_complete_and_unique(self):
        self.assertEqual(self.startpoints, deterministic_expanded_startpoints())
        self.assertEqual(len(self.startpoints), 100)
        self.assertEqual(len(self.expanded), 10_800)
        self.assertEqual(len({row.scenario_id for row in self.expanded}), 10_800)
        self.assertEqual(Counter(row.interval_idx for row in self.expanded), {8: 2700, 10: 2700, 12: 2700, 15: 2700})
        self.assertEqual(sorted({row.opp_speedscale for row in self.expanded}), [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85])

    def test_invalid_preflight_is_explicit_and_does_not_abort(self):
        sample = self.expanded[:3]
        def preflight(row):
            if row == sample[1]:
                raise RuntimeError("synthetic failure")
            return {name: True for name in ("reset", "poses_finite", "observation_finite", "initial_collision_free", "rectangles_disjoint", "planner_constructed")}
        valid, invalid, summary = validate_candidates(sample, preflight)
        self.assertEqual((len(valid), len(invalid)), (2, 1))
        self.assertEqual(invalid[0]["error_type"], "RuntimeError")
        self.assertTrue(summary["complete"])

    def test_h2_seed_sets_and_h3_union(self):
        rows = []
        for index, count in enumerate((0, 1, 2, 8)):
            outcomes = {}
            for offset, seed in enumerate((*PASS1_SEEDS, *PASS2_SEEDS)):
                outcomes[str(seed)] = {"outcome": "ego_collision" if offset < count else "follow", "collision_step": offset if offset < count else None}
            rows.append({"scenario_id": f"s{index}", "seed_outcomes": outcomes, "collision_count": count})
        pools = classify_stochastic_rows(rows)
        self.assertEqual(pools["H2_STOCH_BOUNDARY"], ["s1"])
        self.assertEqual(pools["H2_STOCH_CORE"], ["s2", "s3"])
        self.assertEqual(pools["H2_STOCH_ALL"], ["s1", "s2", "s3"])
        self.assertEqual(union_pool(["s0", "s2"], pools["H2_STOCH_CORE"]), ["s0", "s2", "s3"])

    def test_sampler_reproducibility_and_balanced_cycles(self):
        scenarios = training_scenarios()
        hard = scenarios[:7]
        ids = [row.scenario_id for row in hard]
        first = FixedMixtureScenarioSampler(scenarios, ids, collision_probability=1.0, hard_sampling_mode="per_env_balanced_cycle")
        second = FixedMixtureScenarioSampler(scenarios, ids, collision_probability=1.0, hard_sampling_mode="per_env_balanced_cycle")
        rng_a = np.random.default_rng(20260715)
        rng_b = np.random.default_rng(20260715)
        sequence_a = [first.sample(rng_a)[0].scenario_id for _ in range(14)]
        sequence_b = [second.sample(rng_b)[0].scenario_id for _ in range(14)]
        self.assertEqual(sequence_a, sequence_b)
        self.assertEqual(len(set(sequence_a[:7])), 7)
        self.assertEqual(len(set(sequence_a[7:])), 7)


class TestRegistrySelectorsAndRunnerSafety(unittest.TestCase):
    def test_manifest_has_exact_stable_125_arm_matrix(self):
        arms = build_arms()
        self.assertEqual(len(arms), 125)
        self.assertEqual(Counter(row["stage"] for row in arms), STAGE_COUNTS)
        self.assertEqual(len({row["arm_id"] for row in arms}), 125)
        manifest = build_manifest(experiment_head="0" * 40)
        validate_manifest(manifest)
        self.assertTrue(all(row["config_hash"] == canonical_hash(row["resolved_config"]) for row in arms))

    def test_checkpoint_and_arm_sorting_tuples(self):
        checkpoints = [
            {"update": 2, "metrics": {"ego_collision": 20, "follow": 230, "overtake": 350}},
            {"update": 4, "metrics": {"ego_collision": 19, "follow": 232, "overtake": 349}},
        ]
        self.assertEqual(select_checkpoint(checkpoints)["update"], 4)
        self.assertTrue(checkpoint_flags({"ego_collision": 14, "overtake": 354})["beats_v1_1_best"])
        results = [
            {"arm_id": "b", "status": "COMPLETED", "selected_checkpoint": checkpoints[0], "checkpoints": checkpoints},
            {"arm_id": "a", "status": "COMPLETED", "selected_checkpoint": checkpoints[1], "checkpoints": checkpoints},
        ]
        self.assertEqual([row["arm_id"] for row in rank_arms(results)], ["a", "b"])
        results.append({"arm_id": "failed", "status": "FAILED", "selected_checkpoint": None, "checkpoints": []})
        self.assertEqual([row["arm_id"] for row in rank_arms(results)], ["a", "b"])

    def test_retry_schedule_and_head_worktree_drift_fail_fast(self):
        self.assertEqual(remaining_attempts(0), (1, 2))
        self.assertEqual(remaining_attempts(1), (2,))
        self.assertEqual(remaining_attempts(2), ())
        runner = SweepRunner.__new__(SweepRunner)
        runner.frozen_head = "a" * 40
        with patch("experiments.ppo_v1_2.runner._git", side_effect=["b" * 40, ""]):
            with self.assertRaisesRegex(RuntimeError, "HEAD_OR_WORKTREE_DRIFT"):
                runner._assert_frozen()
        with patch("experiments.ppo_v1_2.runner._git", side_effect=["a" * 40, " M train_ppo_sb3.py"]):
            with self.assertRaisesRegex(RuntimeError, "HEAD_OR_WORKTREE_DRIFT"):
                runner._assert_frozen()

    def test_live_lock_rejects_second_runner_and_heartbeat_records_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = SweepLock(root / "SWEEP.lock")
            first.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "live PID"):
                    SweepLock(root / "SWEEP.lock").acquire()
                state = {"current_stage": "C", "current_arm": "C-test", "start_time": "now", "completed_arms": 0, "total_arms": 125}
                heartbeat = Heartbeat(root / "heartbeat.json", state, interval=3600)
                heartbeat.write()
                document = json.loads((root / "heartbeat.json").read_text(encoding="utf-8"))
                self.assertEqual(document["current_arm"], "C-test")
                self.assertEqual(document["pid"], __import__("os").getpid())
            finally:
                first.release()


if __name__ == "__main__":
    unittest.main()
