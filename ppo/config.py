"""Fixed, reproducible PPO training profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

N_ENVS = 16
N_EPOCHS = 1

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
    version: str
    n_steps: int
    batch_size: int
    updates: int
    evaluation_updates: tuple[int, ...]
    hard_pool: str
    hard_sampling_probability: float
    hard_sampling_mode: str
    critic_profile: str
    seed: int
    device: str
    output_dir: Path
    evaluation_workers: int


V1 = PPOConfig(
    version="v1",
    n_steps=800,
    batch_size=800,
    updates=20,
    evaluation_updates=(5, 10, 15, 20),
    hard_pool="h0_current_det",
    hard_sampling_probability=0.25,
    hard_sampling_mode="with_replacement",
    critic_profile="C0_RAW_SINGLE_FRAME",
    seed=20260715,
    device="cuda",
    output_dir=PROJECT_ROOT / "runs" / "ppo" / "v1",
    evaluation_workers=8,
)

V1_1 = PPOConfig(
    version="v1_1",
    n_steps=1600,
    batch_size=1600,
    updates=20,
    evaluation_updates=(2, 3, 5, 10, 15, 20),
    hard_pool="h0_current_det",
    hard_sampling_probability=0.50,
    hard_sampling_mode="with_replacement",
    critic_profile="C0_RAW_SINGLE_FRAME",
    seed=20260715,
    device="cuda",
    output_dir=PROJECT_ROOT / "runs" / "ppo" / "v1_1",
    evaluation_workers=8,
)

V1_2 = PPOConfig(
    version="v1_2",
    n_steps=1600,
    batch_size=1600,
    updates=8,
    evaluation_updates=(2, 4, 8),
    hard_pool="h0_current_det",
    hard_sampling_probability=0.50,
    hard_sampling_mode="with_replacement",
    critic_profile="C0_RAW_SINGLE_FRAME",
    seed=20260715,
    device="cuda",
    output_dir=PROJECT_ROOT / "runs" / "ppo" / "v1_2",
    evaluation_workers=8,
)

VERSIONS = (V1.version, V1_1.version, V1_2.version)


def _validate(config: PPOConfig) -> None:
    if config.version not in VERSIONS:
        raise ValueError(f"Unknown PPO version: {config.version}")
    if config.n_steps <= 0 or config.batch_size <= 0 or config.updates <= 0:
        raise ValueError("n_steps, batch_size, and updates must be positive")
    if (N_ENVS * config.n_steps) % config.batch_size != 0:
        raise ValueError("batch_size must evenly divide N_ENVS * n_steps")
    if tuple(sorted(set(config.evaluation_updates))) != config.evaluation_updates:
        raise ValueError("evaluation_updates must be unique and strictly increasing")
    if not config.evaluation_updates or config.evaluation_updates[-1] > config.updates:
        raise ValueError("evaluation_updates must be non-empty and no greater than updates")
    if not 0.0 <= config.hard_sampling_probability <= 1.0:
        raise ValueError("hard_sampling_probability must be in [0, 1]")
    if config.evaluation_workers <= 0:
        raise ValueError("evaluation_workers must be positive")


def get_config(
    version: str,
    seed: int,
    device: str,
    output_dir: Path,
    evaluation_workers: int,
) -> PPOConfig:
    """Return one explicit profile with run-specific resources resolved."""

    if version == V1.version:
        profile = V1
    elif version == V1_1.version:
        profile = V1_1
    elif version == V1_2.version:
        profile = V1_2
    else:
        raise ValueError(f"Unknown PPO version: {version}")
    config = replace(
        profile,
        seed=int(seed),
        device=str(device),
        output_dir=Path(output_dir),
        evaluation_workers=int(evaluation_workers),
    )
    _validate(config)
    return config
