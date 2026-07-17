"""Fixed, reproducible PPO training profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

N_ENVS = 16
DEVICE = "cuda"

GAMMA = 0.999
GAE_LAMBDA = 0.995
CLIP_RANGE = 0.10
CLIP_RANGE_VF = None
NORMALIZE_ADVANTAGE = True
VF_COEF = 0.5
ENT_COEF = 0.0
MAX_GRAD_NORM = 0.5
TARGET_KL = None

GRU_LR = 1.0e-6
HEAD_LR = 1.0e-5
CRITIC_LR = 3.0e-4

STEERING_LATENT_STD = 0.05
SPEED_PHYSICAL_STD = 0.15

SIM_DURATION = 8.0
BC_CHECKPOINT = PROJECT_ROOT / "pretrained" / "end2race.pth"


@dataclass(frozen=True)
class PPOConfig:
    name: str
    n_steps: int
    batch_size: int
    updates: int
    checkpoint_updates: tuple[int, ...]
    hard_pool: str
    hard_sampling_probability: float
    hard_sampling_mode: str
    critic_profile: str
    gru_lr: float = GRU_LR
    head_lr: float = HEAD_LR
    target_kl: float | None = TARGET_KL
    steering_distribution: str = "squashed_latent"
    steering_latent_std: float = STEERING_LATENT_STD
    speed_physical_std: float = SPEED_PHYSICAL_STD
    margin_weight: float = 0.0
    margin_threshold: float = 0.0
    n_epochs: int = 1
    update_kl_guardrail: float | None = None


V1 = PPOConfig(
    name="v1",
    n_steps=800,
    batch_size=800,
    updates=20,
    checkpoint_updates=(5, 10, 15, 20),
    hard_pool="h0_current_det",
    hard_sampling_probability=0.25,
    hard_sampling_mode="with_replacement",
    critic_profile="C0_RAW_SINGLE_FRAME",
)

V1_1 = replace(
    V1,
    name="v1_1",
    n_steps=1600,
    batch_size=1600,
    checkpoint_updates=(2, 3, 5, 10, 15, 20),
    hard_sampling_probability=0.50,
)

V1_2_H0_CONTROL = replace(
    V1_1,
    name="v1_2_h0_control",
    updates=8,
    checkpoint_updates=(2, 4, 8),
)

AH_H0_P50_WR = replace(
    V1_2_H0_CONTROL,
    name="AH-H0-p50-wr",
    updates=4,
    checkpoint_updates=(1, 2, 4),
)

AH_H0_P50_BC = replace(
    AH_H0_P50_WR,
    name="AH-H0-p50-bc",
    hard_sampling_mode="per_env_balanced_cycle",
)

AH_H1_P50_BC = replace(
    AH_H0_P50_BC,
    name="AH-H1-p50-bc",
    hard_pool="h1_expanded_det",
)

AH_H2CORE_P50_BC = replace(
    AH_H0_P50_BC,
    name="AH-H2core-p50-bc",
    hard_pool="h2_stoch_core",
)

AH_H3CORE_P50_BC = replace(
    AH_H0_P50_BC,
    name="AH-H3core-p50-bc",
    hard_pool="h3_union_core",
)

AP_SELECTED_P35 = replace(
    AH_H0_P50_WR,
    name="AP-selected-p35",
    hard_sampling_probability=0.35,
)

AB_B06400 = replace(
    AH_H0_P50_WR,
    name="AB-b06400",
    batch_size=6400,
)

AB_B12800 = replace(
    AH_H0_P50_WR,
    name="AB-b12800",
    batch_size=12800,
)

AR_N00800 = replace(
    AH_H0_P50_WR,
    name="AR-n00800",
    n_steps=800,
    updates=8,
    checkpoint_updates=(4, 8),
)

AR_N03200 = replace(
    AH_H0_P50_WR,
    name="AR-n03200",
    n_steps=3200,
    updates=2,
    checkpoint_updates=(1, 2),
)

AK_L0_KL010 = replace(
    AH_H0_P50_WR,
    name="AK-L0-kl010",
    checkpoint_updates=(2, 4),
    target_kl=0.010,
)

AK_L1_KLNONE = replace(
    AH_H0_P50_WR,
    name="AK-L1-klnone",
    checkpoint_updates=(2, 4),
    gru_lr=5.0e-7,
    head_lr=5.0e-6,
)

AK_L1_KL010 = replace(
    AH_H0_P50_WR,
    name="AK-L1-kl010",
    checkpoint_updates=(2, 4),
    gru_lr=5.0e-7,
    head_lr=5.0e-6,
    target_kl=0.010,
)

BE_E0_CURRENT = replace(
    AH_H0_P50_WR,
    name="BE-E0-current",
    checkpoint_updates=(2, 4),
)

BE_E1_COUPLED_LOW = replace(
    AH_H0_P50_WR,
    name="BE-E1-coupled-low",
    checkpoint_updates=(2, 4),
    steering_latent_std=0.03,
    speed_physical_std=0.10,
)

BE_E2_STEER_LOW = replace(
    AH_H0_P50_WR,
    name="BE-E2-steer-low",
    checkpoint_updates=(2, 4),
    steering_latent_std=0.03,
)

BE_E3_SPEED_LOW = replace(
    AH_H0_P50_WR,
    name="BE-E3-speed-low",
    checkpoint_updates=(2, 4),
    speed_physical_std=0.10,
)

SG_LR10 = replace(
    AH_H0_P50_WR,
    name="SG-lr10",
    updates=24,
    checkpoint_updates=(8, 16, 24),
    n_epochs=2,
    gru_lr=1.0e-5,
    head_lr=1.0e-4,
    target_kl=0.010,
)

SG_FULL = replace(
    SG_LR10,
    name="SG-full",
    margin_weight=0.02,
    margin_threshold=0.5,
)

V1_3_B = replace(
    AH_H0_P50_WR,
    name="v1_3_b",
    updates=8,
    checkpoint_updates=(2, 4, 8),
    n_epochs=4,
    target_kl=0.010,
    update_kl_guardrail=0.020,
)

V1_3_A = replace(
    V1_2_H0_CONTROL,
    name="v1_3_a",
    updates=8,
    checkpoint_updates=(8,),
    n_epochs=1,
    gru_lr=3.0e-6,
    head_lr=3.0e-5,
    target_kl=0.010,
    update_kl_guardrail=0.020,
)

V1_3_C = replace(
    V1_3_A,
    name="v1_3_c",
    steering_distribution="physical_gaussian",
)

CONFIGS = {
    config.name: config
    for config in (
        V1,
        V1_1,
        V1_2_H0_CONTROL,
        AH_H0_P50_WR,
        AH_H0_P50_BC,
        AH_H1_P50_BC,
        AH_H2CORE_P50_BC,
        AH_H3CORE_P50_BC,
        AP_SELECTED_P35,
        AB_B06400,
        AB_B12800,
        AR_N00800,
        AR_N03200,
        AK_L0_KL010,
        AK_L1_KLNONE,
        AK_L1_KL010,
        BE_E0_CURRENT,
        BE_E1_COUPLED_LOW,
        BE_E2_STEER_LOW,
        BE_E3_SPEED_LOW,
        SG_LR10,
        SG_FULL,
        V1_3_B,
        V1_3_A,
        V1_3_C,
    )
}

_HARD_SAMPLING_MODES = {"with_replacement", "per_env_balanced_cycle"}
_STEERING_DISTRIBUTIONS = {"squashed_latent", "physical_gaussian"}
_CRITIC_PROFILES = {
    "C0_RAW_SINGLE_FRAME",
    "C1_FROZEN_BC_FEATURE",
    "C2_DETACHED_ACTOR_HIDDEN",
    "C3_PRIVILEGED_PHYSICAL",
}


def _validate(config: PPOConfig) -> None:
    if config.name not in CONFIGS or CONFIGS[config.name] != config:
        raise ValueError(f"Unknown PPO config: {config.name}")
    if not config.name or any(
        character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
        for character in config.name
    ):
        raise ValueError(f"Illegal PPO config name: {config.name}")
    if config.n_steps <= 0 or config.batch_size <= 0 or config.updates <= 0:
        raise ValueError("n_steps, batch_size, and updates must be positive")
    if config.n_epochs <= 0:
        raise ValueError("n_epochs must be positive")
    if (N_ENVS * config.n_steps) % config.batch_size != 0:
        raise ValueError("batch_size must evenly divide N_ENVS * n_steps")
    if tuple(sorted(set(config.checkpoint_updates))) != config.checkpoint_updates:
        raise ValueError("checkpoint_updates must be unique and strictly increasing")
    if (
        not config.checkpoint_updates
        or config.checkpoint_updates[0] <= 0
        or config.checkpoint_updates[-1] > config.updates
    ):
        raise ValueError("checkpoint_updates must be positive, non-empty, and no greater than updates")
    if not 0.0 <= config.hard_sampling_probability <= 1.0:
        raise ValueError("hard_sampling_probability must be in [0, 1]")
    if not config.hard_pool or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in config.hard_pool
    ):
        raise ValueError(f"Illegal hard-pool name: {config.hard_pool}")
    if config.hard_sampling_mode not in _HARD_SAMPLING_MODES:
        raise ValueError(f"Unknown hard sampling mode: {config.hard_sampling_mode}")
    if config.critic_profile not in _CRITIC_PROFILES:
        raise ValueError(f"Unknown critic profile: {config.critic_profile}")
    if config.gru_lr <= 0.0 or config.head_lr <= 0.0:
        raise ValueError("gru_lr and head_lr must be positive")
    if config.target_kl is not None and config.target_kl <= 0.0:
        raise ValueError("target_kl must be positive or None")
    if config.steering_distribution not in _STEERING_DISTRIBUTIONS:
        raise ValueError(f"Unknown steering distribution: {config.steering_distribution}")
    if config.update_kl_guardrail is not None and config.update_kl_guardrail <= 0.0:
        raise ValueError("update_kl_guardrail must be positive or None")
    if config.steering_latent_std <= 0.0 or config.speed_physical_std <= 0.0:
        raise ValueError("steering_latent_std and speed_physical_std must be positive")
    if config.margin_weight < 0.0 or config.margin_threshold < 0.0:
        raise ValueError("Margin weight and threshold must be non-negative")


for _config in CONFIGS.values():
    _validate(_config)


def get_config(name: str) -> PPOConfig:
    """Return one validated, explicit experiment profile."""

    if name not in CONFIGS:
        raise ValueError(f"Unknown PPO config: {name}")
    return CONFIGS[name]
