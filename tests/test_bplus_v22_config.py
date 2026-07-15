#!/usr/bin/env python3
"""Locked v2.2 constants and authority hash regression."""

from dataclasses import replace
import hashlib
from pathlib import Path

from bplus_v22 import (
    ACTION_CORE_LR,
    ARMS,
    ARM_BC_FROZEN,
    ARM_SIDECAR_FINETUNE,
    ARM_SIDECAR_FROZEN,
    LOCKED_CONFIG,
    MACRO_GAMMA,
    OWNER_DECISION,
    POSITIVE_SPEED_BUDGET,
    DUAL_INITIAL_VALUE,
    DUAL_MAX_VALUE,
    INITIAL_BRAKE_LOGIT,
    OVERTAKE_NONINFERIORITY,
    SIDECAR_FINETUNE_LR,
    V22_PLAN_SHA256,
    V22_SPEC_SHA256,
    validate_arm,
)


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    assert ARMS == (ARM_BC_FROZEN, ARM_SIDECAR_FROZEN, ARM_SIDECAR_FINETUNE)
    assert LOCKED_CONFIG.owner_decision == OWNER_DECISION
    assert LOCKED_CONFIG.macro_steps == 10
    assert MACRO_GAMMA == 0.997**10
    assert POSITIVE_SPEED_BUDGET == 0.0
    assert OVERTAKE_NONINFERIORITY == 0.01
    assert DUAL_INITIAL_VALUE == 1.0 and DUAL_MAX_VALUE == 3.0
    assert INITIAL_BRAKE_LOGIT == -6.0
    assert SIDECAR_FINETUNE_LR * 10 == ACTION_CORE_LR
    assert sha256(
        "docs/superpowers/specs/2026-07-11-ppo-safety-first-bplus-v2.2.md"
    ) == V22_SPEC_SHA256
    assert sha256(
        "docs/superpowers/plans/2026-07-11-bplus-v2.2-d3r2-implementation-plan.md"
    ) == V22_PLAN_SHA256
    for arm in ARMS:
        assert validate_arm(arm) == arm
    try:
        validate_arm("FULL_BC_UNFREEZE")
        raise AssertionError("unlocked arm accepted")
    except ValueError:
        pass
    try:
        replace(LOCKED_CONFIG, positive_speed_budget=0.01)
        raise AssertionError("locked positive speed accepted")
    except ValueError:
        pass
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
