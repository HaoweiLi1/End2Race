"""Closed configuration vocabulary for the preregistered PPO V1.2 sweep."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


CRITIC_PROFILES = (
    "C0_RAW_SINGLE_FRAME",
    "C1_FROZEN_BC_FEATURE",
    "C2_DETACHED_ACTOR_HIDDEN",
    "C3_PRIVILEGED_PHYSICAL",
)
HARD_POOL_IDS = (
    "H0_CURRENT_DET",
    "H1_EXPANDED_DET",
    "H2_STOCH_CORE",
    "H2_STOCH_ALL",
    "H3_UNION_CORE",
    "H3_UNION_ALL",
)
ALL_HARD_POOL_IDS = (*HARD_POOL_IDS, "H2_STOCH_BOUNDARY")
SAMPLING_MODES = ("with_replacement", "per_env_balanced_cycle")
LEGAL_STATUSES = (
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "SKIPPED_DEPENDENCY",
    "SKIPPED_EMPTY_POOL",
    "BLOCKED_IMPLEMENTATION",
)
STAGES = ("C", "H", "B", "R", "K", "E", "G", "W", "X", "S")
STAGE_COUNTS = {"C": 4, "H": 48, "B": 6, "R": 4, "K": 16, "E": 6, "G": 4, "W": 12, "X": 16, "S": 9}

DEFAULT_V1_2_CONFIG: dict[str, Any] = {
    "n_envs": 16,
    "n_steps": 1600,
    "batch_size": 1600,
    "n_epochs": 1,
    "gamma": 0.999,
    "gae_lambda": 0.995,
    "clip_range": 0.10,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "vf_coef": 0.5,
    "ent_coef": 0.0,
    "max_grad_norm": 0.5,
    "target_kl": None,
    "gru_lr": 1.0e-6,
    "head_lr": 1.0e-5,
    "critic_lr": 3.0e-4,
    "steering_latent_std": 0.05,
    "speed_physical_std": 0.15,
    "reward_progress_weight": 0.010,
    "reward_relative_weight": 0.020,
    "reward_collision": -2.0,
    "hard_pool_id": "H0_CURRENT_DET",
    "hard_sampling_probability": 0.50,
    "hard_sampling_mode": "with_replacement",
    "critic_profile": "C0_RAW_SINGLE_FRAME",
    "updates": 8,
    "evaluation_transition_budgets": [51_200, 102_400, 204_800],
    "seed": 20260715,
    "device": "cuda",
}


def resolve_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a fully derived config and reject values outside the V1.2 contract."""

    config = deepcopy(DEFAULT_V1_2_CONFIG)
    config.update(overrides or {})
    if config["critic_profile"] not in CRITIC_PROFILES:
        raise ValueError(f"Unknown critic profile: {config['critic_profile']}")
    if config["hard_pool_id"] not in HARD_POOL_IDS:
        raise ValueError(f"Unknown formal hard pool: {config['hard_pool_id']}")
    if config["hard_sampling_mode"] not in SAMPLING_MODES:
        raise ValueError(f"Unknown hard sampling mode: {config['hard_sampling_mode']}")
    positive_ints = ("n_envs", "n_steps", "batch_size", "n_epochs", "updates")
    if any(not isinstance(config[key], int) or config[key] <= 0 for key in positive_ints):
        raise ValueError("n_envs, n_steps, batch_size, n_epochs and updates must be positive integers")
    transitions_per_update = config["n_envs"] * config["n_steps"]
    if transitions_per_update % config["batch_size"] and config["batch_size"] % transitions_per_update:
        raise ValueError("batch_size and n_envs * n_steps must have an exact whole-batch relation")
    probability = float(config["hard_sampling_probability"])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("hard_sampling_probability must be finite and in [0, 1]")
    finite_keys = (
        "gamma", "gae_lambda", "clip_range", "vf_coef", "ent_coef", "max_grad_norm",
        "gru_lr", "head_lr", "critic_lr", "steering_latent_std", "speed_physical_std",
        "reward_progress_weight", "reward_relative_weight", "reward_collision",
    )
    if any(not math.isfinite(float(config[key])) for key in finite_keys):
        raise ValueError("All numeric PPO V1.2 settings must be finite")
    if not (0.0 < config["gamma"] <= 1.0 and 0.0 < config["gae_lambda"] <= 1.0):
        raise ValueError("gamma and gae_lambda must be in (0, 1]")
    if any(config[key] < 0.0 for key in ("gru_lr", "head_lr", "critic_lr")):
        raise ValueError("learning rates must be non-negative")
    if config["steering_latent_std"] <= 0.0 or config["speed_physical_std"] <= 0.0:
        raise ValueError("fixed exploration standard deviations must be positive")
    if config["target_kl"] is not None and (not math.isfinite(float(config["target_kl"])) or config["target_kl"] <= 0):
        raise ValueError("target_kl must be null or finite and positive")
    budgets = [int(value) for value in config["evaluation_transition_budgets"]]
    total_transitions = transitions_per_update * config["updates"]
    if budgets != sorted(set(budgets)) or any(value <= 0 or value > total_transitions for value in budgets):
        raise ValueError("evaluation transition budgets must be unique, increasing and within the arm budget")
    if any(value % transitions_per_update for value in budgets):
        raise ValueError("evaluation transition budgets must align to update boundaries")
    config["evaluation_transition_budgets"] = budgets
    config["evaluation_updates"] = [value // transitions_per_update for value in budgets]
    config["transitions_per_update"] = transitions_per_update
    config["minibatches_per_update"] = math.ceil(transitions_per_update / config["batch_size"])
    config["optimizer_steps_per_update"] = config["minibatches_per_update"] * config["n_epochs"]
    config["expected_transitions"] = total_transitions
    config["planned_optimizer_steps"] = config["optimizer_steps_per_update"] * config["updates"]
    config["total_optimizer_steps"] = config["planned_optimizer_steps"]
    return config
