"""Generate the complete preregistered 125-arm PPO V1.2 manifest."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from typing import Any

from .config_schema import (
    CRITIC_PROFILES,
    HARD_POOL_IDS,
    LEGAL_STATUSES,
    SAMPLING_MODES,
    STAGE_COUNTS,
    STAGES,
    resolve_config,
)
from .experiment_spec import BASELINE_COMMIT, BC_SHA256, canonical_hash


def _arm(
    arm_id: str,
    stage: str,
    overrides: dict[str, Any],
    *,
    parents: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = resolve_config(overrides)
    config.update(
        experiment_profile="ppo_v1_2",
        master_seed=int(config["seed"]),
        evaluation_workers=8,
        smoke="none",
        bc_checkpoint="pretrained/end2race.pth",
    )
    return {
        "arm_id": arm_id,
        "stage": stage,
        "parent_selections": parents or {},
        "resolved_config": config,
        "config_hash": canonical_hash(config),
        "seed": config["seed"],
        "expected_transitions": config["expected_transitions"],
        "evaluation_transition_budgets": config["evaluation_transition_budgets"],
        "status": "PENDING",
        "attempt_count": 0,
        "metadata": metadata or {},
    }


def build_arms() -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    for critic in CRITIC_PROFILES:
        arms.append(_arm(f"C-{critic}", "C", {"critic_profile": critic}))

    c1_parent = {"critic_profile": {"stage": "C", "rank": 1}}
    for pool_id, probability, mode in product(HARD_POOL_IDS, (0.25, 0.35, 0.40, 0.50), SAMPLING_MODES):
        probability_code = f"p{int(round(probability * 100)):02d}"
        mode_code = "wr" if mode == "with_replacement" else "bc"
        arms.append(
            _arm(
                f"H-{pool_id}-{probability_code}-{mode_code}",
                "H",
                {"hard_pool_id": pool_id, "hard_sampling_probability": probability, "hard_sampling_mode": mode},
                parents=c1_parent,
            )
        )

    ch_parent = {
        "critic_profile": {"stage": "C", "rank": 1},
        "hard_configuration": {"stage": "H", "rank": 1},
    }
    for batch_size in (800, 1600, 3200, 6400, 12800, 25600):
        arms.append(_arm(f"B-{batch_size:05d}", "B", {"batch_size": batch_size}, parents=ch_parent))

    rollout_rows = (
        (800, 16, (51_200, 102_400, 204_800)),
        (1600, 8, (51_200, 102_400, 204_800)),
        (3200, 4, (51_200, 102_400, 204_800)),
        (6400, 2, (102_400, 204_800)),
    )
    for n_steps, updates, budgets in rollout_rows:
        arms.append(
            _arm(
                f"R-{n_steps:04d}",
                "R",
                {"n_steps": n_steps, "batch_size": 6400, "updates": updates, "evaluation_transition_budgets": list(budgets)},
                parents=ch_parent,
            )
        )

    chbr_parent = {
        **ch_parent,
        "batch_size": {"stage": "B", "rank": 1},
        "rollout": {"stage": "R", "rank": 1},
    }
    lr_profiles = {
        "L0": (1.0e-6, 1.0e-5),
        "L1": (1.0e-6, 7.5e-6),
        "L2": (1.0e-6, 5.0e-6),
        "L3": (5.0e-7, 5.0e-6),
    }
    for lr_id, target_kl in product(lr_profiles, (None, 0.020, 0.010, 0.005)):
        kl_code = "none" if target_kl is None else f"{int(round(target_kl * 1000)):03d}"
        gru_lr, head_lr = lr_profiles[lr_id]
        arms.append(
            _arm(
                f"K-{lr_id}-kl{kl_code}",
                "K",
                {"gru_lr": gru_lr, "head_lr": head_lr, "target_kl": target_kl},
                parents=chbr_parent,
                metadata={"lr_profile": lr_id},
            )
        )

    ranked_parent = {**chbr_parent, "kl_lr": {"stage": "K", "rank": 1}}
    exploration = ((0.05, 0.15), (0.04, 0.12), (0.03, 0.10), (0.02, 0.08), (0.03, 0.15), (0.05, 0.10))
    for index, (steer_std, speed_std) in enumerate(exploration):
        arms.append(
            _arm(
                f"E-E{index}", "E",
                {"steering_latent_std": steer_std, "speed_physical_std": speed_std},
                parents=ranked_parent,
                metadata={"exploration_profile": f"E{index}"},
            )
        )

    e_parent = {**ranked_parent, "exploration": {"stage": "E", "rank": 1}}
    for value in (0.990, 0.995, 0.997, 1.000):
        arms.append(_arm(f"G-{value:.3f}", "G", {"gae_lambda": value}, parents=e_parent))

    g_parent = {**e_parent, "gae": {"stage": "G", "rank": 1}}
    for progress, relative, collision in product((0.010, 0.015), (0.020, 0.030), (-2.0, -3.0, -4.0)):
        arms.append(
            _arm(
                f"W-p{int(progress*1000):03d}-r{int(relative*1000):03d}-c{abs(int(collision)):02d}",
                "W",
                {"reward_progress_weight": progress, "reward_relative_weight": relative, "reward_collision": collision},
                parents=g_parent,
            )
        )

    x_fixed = {
        "rollout": {"stage": "R", "rank": 1},
        "exploration": {"stage": "E", "rank": 1},
        "gae": {"stage": "G", "rank": 1},
        "reward": {"stage": "W", "rank": 1},
    }
    for critic_rank, hard_rank, batch_rank, kl_rank in product((1, 2), repeat=4):
        parents = {
            **x_fixed,
            "critic_profile": {"stage": "C", "rank": critic_rank},
            "hard_configuration": {"stage": "H", "rank": hard_rank},
            "batch_size": {"stage": "B", "rank": batch_rank},
            "kl_lr": {"stage": "K", "rank": kl_rank},
        }
        code = f"c{critic_rank}h{hard_rank}b{batch_rank}k{kl_rank}"
        arms.append(
            _arm(
                f"X-{code}", "X",
                {"updates": 16, "evaluation_transition_budgets": [51_200, 102_400, 204_800, 409_600]},
                parents=parents,
                metadata={"interaction_ranks": [critic_rank, hard_rank, batch_rank, kl_rank]},
            )
        )

    for config_rank, seed in product((1, 2, 3), (20260715, 20260716, 20260717)):
        arms.append(
            _arm(
                f"S-x{config_rank}-seed{seed}", "S",
                {"updates": 16, "evaluation_transition_budgets": [51_200, 102_400, 204_800, 409_600], "seed": seed},
                parents={"full_configuration": {"stage": "X", "rank": config_rank}},
                metadata={"x_rank": config_rank},
            )
        )

    ids = [arm["arm_id"] for arm in arms]
    actual = {stage: sum(arm["stage"] == stage for arm in arms) for stage in STAGES}
    if len(arms) != 125 or len(ids) != len(set(ids)) or actual != STAGE_COUNTS:
        raise AssertionError(f"Invalid PPO V1.2 matrix: total={len(arms)}, counts={actual}")
    return arms


def build_manifest(*, experiment_head: str, hard_pool_hashes: dict[str, str | None] | None = None) -> dict[str, Any]:
    if len(experiment_head) != 40:
        raise ValueError("experiment_head must be a full Git SHA")
    arms = build_arms()
    pool_hashes = {pool_id: None for pool_id in HARD_POOL_IDS}
    pool_hashes.update(hard_pool_hashes or {})
    for arm in arms:
        pool_id = arm["resolved_config"]["hard_pool_id"]
        arm["resolved_config"]["hard_pool_hash"] = pool_hashes[pool_id]
        arm["config_hash"] = canonical_hash(arm["resolved_config"])
    manifest = {
        "schema_version": 1,
        "experiment_id": "simple_ppo_v1_2",
        "baseline_commit": BASELINE_COMMIT,
        "experiment_head": experiment_head,
        "canonical_bc_sha256": BC_SHA256,
        "hard_pool_hashes": pool_hashes,
        "stage_order": list(STAGES),
        "stage_counts": STAGE_COUNTS,
        "training_arm_count": len(arms),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arms": arms,
    }
    manifest["manifest_hash"] = canonical_hash({key: value for key, value in manifest.items() if key not in {"generated_at", "manifest_hash"}})
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    arms = manifest.get("arms", [])
    if manifest.get("training_arm_count") != 125 or len(arms) != 125:
        raise ValueError("Sweep manifest must contain exactly 125 training arms")
    if len({arm.get("arm_id") for arm in arms}) != 125:
        raise ValueError("Sweep arm IDs must be unique")
    if any(arm.get("status") not in LEGAL_STATUSES for arm in arms):
        raise ValueError("Sweep manifest contains an illegal arm status")
    counts = {stage: sum(arm.get("stage") == stage for arm in arms) for stage in STAGES}
    if counts != STAGE_COUNTS:
        raise ValueError(f"Sweep stage counts differ from the preregistration: {counts}")
    for arm in arms:
        if canonical_hash(arm["resolved_config"]) != arm["config_hash"]:
            raise ValueError(f"Config hash mismatch for {arm['arm_id']}")


def clone_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(manifest)
