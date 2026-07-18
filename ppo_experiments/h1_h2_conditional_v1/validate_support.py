#!/usr/bin/env python3
"""No-training validation for conditional-exploration support."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import End2Race  # noqa: E402
from ppo import config as ppo_config  # noqa: E402
from train_ppo import (  # noqa: E402
    PPOTrainingCallback,
    _fixed_reset_provider,
    build_sampler,
    make_training_env,
)
from utils import atomic_write_json  # noqa: E402


OUTPUT = Path(__file__).resolve().parent
SOURCE_PATHS = (
    "ppo/config.py",
    "ppo/policy.py",
    "ppo/environment.py",
    "ppo/reward.py",
    "ppo/scenarios.py",
    "train_ppo.py",
    "eval_multiagent.py",
    "evaluate.sh",
    "utils.py",
    "pretrained/end2race.pth",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reset_signature(spec) -> dict:
    scenario = dict(spec.scenario)
    return {
        "scenario": scenario,
        "poses": np.asarray(spec.poses).tolist(),
        "initial_speed_feature": float(spec.initial_speed_feature),
    }


def validate_nonpaired_identity() -> dict:
    config = ppo_config.get_config("QP3_A1_H1FULL_8S")
    reference_sampler = build_sampler(config)
    candidate_sampler = build_sampler(config)
    reference_rng = np.random.default_rng(20260718)
    candidate_rng = np.random.default_rng(20260718)
    provider = _fixed_reset_provider(0, candidate_sampler, config, 20260718, "hard")
    checked = 64
    for index in range(checked):
        reference = reference_sampler.reset_spec(reference_rng, env_role="hard")
        candidate = provider(candidate_rng)
        if reset_signature(reference) != reset_signature(candidate):
            raise RuntimeError(f"Non-paired hard reset changed at ordinal {index}")
    return {"config": config.name, "role": "hard", "reset_count": checked, "exact_match": True}


def validate_paired_assignment() -> dict:
    config = ppo_config.get_config("N2-I8-paired-8s")
    # The H2 manifest does not exist until classification, so use the unchanged
    # full-H1 hard pool solely to test pair assignment mechanics.
    smoke_config = ppo_config.get_config("N1-H1F-p25")
    smoke_config = type(smoke_config)(
        **{
            **asdict(smoke_config),
            "paired_hard_sampling": True,
            "hard_pair_size": 2,
        }
    )
    sampler = build_sampler(smoke_config)
    left = _fixed_reset_provider(0, sampler, smoke_config, 20260730, "hard")
    right = _fixed_reset_provider(1, sampler, smoke_config, 20260730, "hard")
    left_rng = np.random.default_rng(100)
    right_rng = np.random.default_rng(101)
    rows = []
    for ordinal in range(32):
        left_spec = left(left_rng)
        right_spec = right(right_rng)
        left_scenario = left_spec.scenario
        right_scenario = right_spec.scenario
        if left_scenario["scenario_id"] != right_scenario["scenario_id"]:
            raise RuntimeError(f"Paired scenario mismatch at ordinal {ordinal}")
        expected = {
            "pair_group": 0,
            "pair_episode_ordinal": ordinal,
        }
        for key, value in expected.items():
            if left_scenario[key] != value or right_scenario[key] != value:
                raise RuntimeError(f"Paired metadata mismatch for {key} at ordinal {ordinal}")
        if left_scenario["pair_member"] != 0 or right_scenario["pair_member"] != 1:
            raise RuntimeError(f"Pair members differ from 0/1 at ordinal {ordinal}")
        rows.append(left_scenario["scenario_id"])
    return {
        "target_config": config.name,
        "smoke_pool": smoke_config.hard_pool,
        "pair_group": 0,
        "episode_count_per_member": len(rows),
        "same_scenario_sequence": True,
        "independent_reset_rng_seeds": [100, 101],
        "scenario_ids": rows,
    }


def paired_smoke_config():
    base = ppo_config.get_config("N1-H1F-p25")
    return type(base)(
        **{
            **asdict(base),
            "paired_hard_sampling": True,
            "hard_pair_size": 2,
        }
    )


def validate_f110_info_smoke() -> dict:
    config = paired_smoke_config()
    sampler = build_sampler(config)
    vector_env = DummyVecEnv(
        [make_training_env(rank, sampler, config, 20260730) for rank in (0, 1)]
    )
    try:
        vector_env.seed(20260730)
        vector_env.env_method("set_policy_update_index", 1)
        observations = vector_env.reset()
        reset_infos = list(vector_env.reset_infos)
        if reset_infos[0]["scenario_id"] != reset_infos[1]["scenario_id"]:
            raise RuntimeError("F110 paired reset produced different scenario IDs")
        required = {
            "env_role",
            "pair_group",
            "pair_member",
            "pair_episode_ordinal",
            "scenario_id",
            "policy_update_index",
            "episode_outcome",
            "episode_return",
            "elapsed_time",
        }
        for info in reset_infos:
            missing = required - set(info)
            if missing:
                raise RuntimeError(f"F110 paired reset info is missing {sorted(missing)}")
        _next_observations, _rewards, _dones, step_infos = vector_env.step(
            np.zeros((2, 2), dtype=np.float32)
        )
        if not np.isfinite(observations).all() or not all(required <= set(info) for info in step_infos):
            raise RuntimeError("F110 paired one-step smoke is non-finite or missing info")
        return {
            "status": "PASS",
            "env_count": 2,
            "same_initial_scenario": reset_infos[0]["scenario_id"],
            "required_info_fields": sorted(required),
            "reset_observation_finite": True,
            "one_step_completed": True,
        }
    finally:
        vector_env.close()


def validate_paired_telemetry_smoke() -> dict:
    callback = PPOTrainingCallback()
    callback.update = 0
    callback._on_rollout_start()
    base = {
        "sampler_branch": "hard_pool",
        "scenario": {"env_role": "hard"},
        "scenario_id": "paired-smoke",
        "reward_progress": 0.0,
        "reward_relative": 0.0,
        "reward_margin": 0.0,
        "reward_collision": 0.0,
        "reward_total": 0.0,
        "elapsed_time": 1.0,
        "timeout": False,
        "pair_group": 0,
        "pair_episode_ordinal": 0,
        "policy_update_index": 1,
    }
    callback.locals = {
        "infos": [
            {
                **base,
                "pair_member": 0,
                "ego_collision": True,
                "relative_position_m": -1.0,
                "episode_outcome": "ego_collision",
                "episode_return": -2.0,
            },
            {
                **base,
                "pair_member": 1,
                "ego_collision": False,
                "relative_position_m": 1.0,
                "episode_outcome": "overtake",
                "episode_return": 0.5,
            },
        ],
        "dones": np.asarray([True, True]),
        "actions": np.zeros((2, 2), dtype=np.float32),
    }
    callback._on_step()
    telemetry = callback._paired_telemetry()
    if not (
        telemetry["complete_same_update_pairs"] == 1
        and telemetry["discordant_pairs"] == 1
        and telemetry["incomplete_pairs"] == 0
    ):
        raise RuntimeError(f"Paired telemetry smoke mismatch: {telemetry}")
    return {"status": "PASS", "telemetry": telemetry}


def validate_checkpoint_strict_load() -> dict:
    path = ROOT / "pretrained" / "end2race.pth"
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = End2Race(mask_prob=0.0, hidden_scale=4)
    model.load_state_dict(state, strict=True)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "strict_load": True,
        "key_count": len(state),
    }


def main() -> int:
    document = {
        "schema_version": 1,
        "status": "PASS",
        "nonpaired_behavior_identity": validate_nonpaired_identity(),
        "paired_sampler_no_training_smoke": validate_paired_assignment(),
        "f110_paired_reset_step_smoke": validate_f110_info_smoke(),
        "paired_telemetry_smoke": validate_paired_telemetry_smoke(),
        "canonical_checkpoint": validate_checkpoint_strict_load(),
        "formal_config_contracts": {
            name: asdict(ppo_config.get_config(name))
            for name in (
                "N1-H1F-p50",
                "N1-H1F-p25",
                "N1-H1E-p50",
                "N1-H1E-p25",
                "N2-I8-single-8s",
                "N2-I8-paired-8s",
                "N2-I8-paired-4s",
                "N2-I7-paired-4s",
            )
        },
        "post_implementation_source_hashes": {
            relative: sha256_file(ROOT / relative) for relative in SOURCE_PATHS
        },
    }
    atomic_write_json(OUTPUT / "SUPPORT_VALIDATION.json", document)
    print(json.dumps({"status": document["status"], "checks": list(document)[2:]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
