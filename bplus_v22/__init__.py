"""Locked constants for the owner-approved B+ v2.2 policy redirect."""

from __future__ import annotations

from dataclasses import dataclass


SCHEMA = "bplus-v2.2-locked-config-1"
OWNER_DECISION = "OWNER_REDIRECT_BPLUS_V22_DIRECT_POLICY_KPI_TTC_DIAGNOSTIC_ONLY"

ARM_BC_FROZEN = "BC_FROZEN"
ARM_SIDECAR_FROZEN = "SIDECAR_FROZEN"
ARM_SIDECAR_FINETUNE = "SIDECAR_FINETUNE"
ARMS = (ARM_BC_FROZEN, ARM_SIDECAR_FROZEN, ARM_SIDECAR_FINETUNE)

SEED = 20260711
PILOT_SEEDS = (0, 1)
MACRO_STEPS = 10
MICRO_GAMMA = 0.997
MACRO_GAMMA = MICRO_GAMMA**MACRO_STEPS
MACRO_LAMBDA = 0.99

BC_FEATURE_DIM = 1680
POLICY_FEATURE_DIM = 128
PRIVILEGED_FEATURE_DIM = 12
HISTORY_OFFSETS = (0, 5, 10, 20, 35, 50, 75, 100)
LIDAR_BEAMS = 360
SCALAR_HISTORY_DIM = 24

STEER_BUDGET = 0.2
BRAKE_BUDGET = 1.0
POSITIVE_SPEED_BUDGET = 0.0
ACTION_CORE_LR = 3e-5
SIDECAR_FINETUNE_LR = 3e-6
INITIAL_STEER_STD = 0.15
INITIAL_BRAKE_STD = 0.25
INITIAL_BRAKE_LOGIT = -6.0
OVERTAKE_NONINFERIORITY = 0.01
DUAL_INITIAL_VALUE = 1.0
DUAL_MAX_VALUE = 3.0
DUAL_LEARNING_RATE = 0.5
DUAL_EMA_ALPHA = 0.2
DUAL_MIN_COMPLETED_EPISODES = 32

BC_CHECKPOINT_SHA256 = "b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4"
D2_DATASET_MANIFEST_SHA256 = "36b9640c9ec8407f12573bc3543712573283881b73400856a4b25f294b1f57c4"
D2_SPLIT_MANIFEST_SHA256 = "2f8146d7be0e36c3abcc084dcdbfa9e3df85983c37c6249294ab19b1431c49f3"
D2_TEST_SEAL_SHA256 = "cee71d818bc050b0ca0647ee32ed1b5655e471ea60b39133aed7b37fc9c1a87e"
D2R_SIGNALS_MANIFEST_SHA256 = "d653d77dcf270ce9b9e714d23a9b5600b15dfb90996e627610153be1763b513a"
D25_OUTPUT_MANIFEST_SHA256 = "42a31686a1c654bfe702085d0a7ae4f587e02e4807ae9eba33fae7ad600dcca3"
V22_SPEC_SHA256 = "7faa0133428cd7d4cbaaf90a4dd9fd7247fd1cff770fd0a3a0630ef458dbe976"
V22_PLAN_SHA256 = "fc24a4da3292cb5dfe7d517a20a128355bfecb6c143a7f178cdb6c2aacabef57"


@dataclass(frozen=True)
class LockedConfig:
    schema: str = SCHEMA
    owner_decision: str = OWNER_DECISION
    seed: int = SEED
    pilot_seeds: tuple[int, int] = PILOT_SEEDS
    macro_steps: int = MACRO_STEPS
    micro_gamma: float = MICRO_GAMMA
    macro_lambda: float = MACRO_LAMBDA
    steer_budget: float = STEER_BUDGET
    brake_budget: float = BRAKE_BUDGET
    positive_speed_budget: float = POSITIVE_SPEED_BUDGET
    action_core_lr: float = ACTION_CORE_LR
    sidecar_finetune_lr: float = SIDECAR_FINETUNE_LR
    overtake_noninferiority: float = OVERTAKE_NONINFERIORITY
    dual_initial_value: float = DUAL_INITIAL_VALUE
    dual_max_value: float = DUAL_MAX_VALUE
    dual_learning_rate: float = DUAL_LEARNING_RATE
    dual_ema_alpha: float = DUAL_EMA_ALPHA
    dual_min_completed_episodes: int = DUAL_MIN_COMPLETED_EPISODES

    def __post_init__(self) -> None:
        expected = {
            "schema": SCHEMA,
            "owner_decision": OWNER_DECISION,
            "seed": SEED,
            "pilot_seeds": PILOT_SEEDS,
            "macro_steps": MACRO_STEPS,
            "micro_gamma": MICRO_GAMMA,
            "macro_lambda": MACRO_LAMBDA,
            "steer_budget": STEER_BUDGET,
            "brake_budget": BRAKE_BUDGET,
            "positive_speed_budget": POSITIVE_SPEED_BUDGET,
            "action_core_lr": ACTION_CORE_LR,
            "sidecar_finetune_lr": SIDECAR_FINETUNE_LR,
            "overtake_noninferiority": OVERTAKE_NONINFERIORITY,
            "dual_initial_value": DUAL_INITIAL_VALUE,
            "dual_max_value": DUAL_MAX_VALUE,
            "dual_learning_rate": DUAL_LEARNING_RATE,
            "dual_ema_alpha": DUAL_EMA_ALPHA,
            "dual_min_completed_episodes": DUAL_MIN_COMPLETED_EPISODES,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"B+ v2.2 locked config drift: {name}")
        if self.sidecar_finetune_lr * 10 != self.action_core_lr:
            raise ValueError("B+ v2.2 sidecar LR must be exactly action-core LR / 10")


LOCKED_CONFIG = LockedConfig()


def validate_arm(arm: str) -> str:
    """Return one canonical arm ID or fail closed."""

    value = str(arm)
    if value not in ARMS:
        raise ValueError(f"unknown B+ v2.2 arm: {value}")
    return value
