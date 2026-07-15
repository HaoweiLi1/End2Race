"""B2 scenario, curriculum, observation-history, and critic-state contracts.

This module contains the environment-independent pieces of the B2 runner.  The
live simulator loop is added below these pure contracts so the same scenario
and observation semantics can be tested without a GPU or F110 process.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from bplus_v22 import HISTORY_OFFSETS, LIDAR_BEAMS
from ppo_utils import RewardState, relative_geometry
from utils import wrap_rel_s


CURRICULUM_SCHEMA = "bplus-v2.2-b2-curriculum-1"
CURRICULUM_DOMAIN = b"end2race:bplus-v2.2:b2-curriculum:v1\0"
TRAINING_ROWS = 1640
TRAINING_COLLISION_ROWS = 81
TRAINING_REMAINING_ROWS = 1559
DEVELOPMENT_ROWS = 288
EPISODES_PER_ITERATION = 16
COLLISION_EPISODES_PER_ITERATION = 8
REMAINING_EPISODES_PER_ITERATION = 8
PRIVILEGED_DIM = 12


def _bool_text(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected canonical bool text, got {value!r}")


def _archived_outcome(npz_relpath: str) -> str:
    parts = PurePosixPath(str(npz_relpath)).parts
    if len(parts) < 2 or parts[-2] not in {"collision", "follow", "overtake"}:
        raise ValueError(f"cannot recover archived outcome from {npz_relpath!r}")
    return parts[-2]


@dataclass(frozen=True)
class B2Scenario:
    training_order: int
    l2_id: str
    l4_id: str
    map_name: str
    skill: str
    opponent_raceline: str
    speedscale: float
    resolved_ego_idx: int
    bc_collision_any: bool
    archived_bc_outcome: str

    def __post_init__(self) -> None:
        if self.training_order < 0:
            raise ValueError("B2 scenario order must be nonnegative")
        if not self.l2_id.startswith("L2:") or not self.l4_id.startswith("L4:"):
            raise ValueError("B2 scenario identity schema mismatch")
        if not self.map_name or not self.skill or not self.opponent_raceline:
            raise ValueError("B2 scenario categorical field is empty")
        if not np.isfinite(self.speedscale) or self.speedscale <= 0.0:
            raise ValueError("B2 scenario speedscale is invalid")
        if self.resolved_ego_idx < 0:
            raise ValueError("B2 scenario ego index is invalid")
        if self.archived_bc_outcome not in {"collision", "follow", "overtake"}:
            raise ValueError("B2 archived outcome is invalid")
        if self.bc_collision_any != (self.archived_bc_outcome == "collision"):
            raise ValueError("B2 collision metadata/outcome disagree")

    def simulator_case(self) -> dict[str, str]:
        return {
            "map_name": self.map_name,
            "resolved_ego_idx": str(self.resolved_ego_idx),
            "opponent_raceline": self.opponent_raceline,
            "speedscale_hex": float(self.speedscale).hex(),
        }


@dataclass(frozen=True)
class B2ScenarioSets:
    collision: tuple[B2Scenario, ...]
    remaining: tuple[B2Scenario, ...]
    development_rows: tuple[dict[str, str], ...]

    @property
    def training(self) -> tuple[B2Scenario, ...]:
        return tuple(sorted(self.collision + self.remaining, key=lambda row: row.training_order))


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_b2_scenario_sets(
    task8_release: str | Path,
    d2_episode_metadata: str | Path,
) -> B2ScenarioSets:
    """Join the opened Task-8 universe to opened D2 collision metadata by L2."""

    release = Path(task8_release)
    if not (release / "COMPLETE").is_file():
        raise ValueError("Task-8 release is incomplete")
    training = _read_tsv(release / "training_scenarios.tsv")
    development = _read_tsv(release / "development_scenarios.tsv")
    metadata_rows = _read_tsv(Path(d2_episode_metadata))
    metadata: dict[str, dict[str, str]] = {}
    for row in metadata_rows:
        l2_id = row["l2_id"]
        if l2_id in metadata:
            raise ValueError(f"duplicate D2 metadata L2: {l2_id}")
        metadata[l2_id] = row
    if len(training) != TRAINING_ROWS or len(development) != DEVELOPMENT_ROWS:
        raise ValueError("Task-8 B2 row count drift")
    if len({row["l2_id"] for row in training}) != len(training):
        raise ValueError("Task-8 training L2 is not unique")

    built: list[B2Scenario] = []
    for expected_order, row in enumerate(training):
        if int(row["training_order"]) != expected_order:
            raise ValueError("Task-8 training physical order drift")
        l2_id = row["l2_id"]
        if l2_id not in metadata:
            raise ValueError(f"Task-8 L2 missing D2 metadata: {l2_id}")
        meta = metadata[l2_id]
        for name in ("l4_id", "map_name", "skill", "opponent_raceline"):
            if row[name] != meta[name]:
                raise ValueError(f"Task-8/D2 metadata mismatch for {l2_id}: {name}")
        outcome = _archived_outcome(row["npz_relpath"])
        built.append(
            B2Scenario(
                training_order=expected_order,
                l2_id=l2_id,
                l4_id=row["l4_id"],
                map_name=row["map_name"],
                skill=row["skill"],
                opponent_raceline=row["opponent_raceline"],
                speedscale=float.fromhex(row["speedscale_hex"]),
                resolved_ego_idx=int(row["resolved_ego_idx"]),
                bc_collision_any=_bool_text(meta["collision_any"]),
                archived_bc_outcome=outcome,
            )
        )
    collision = tuple(row for row in built if row.bc_collision_any)
    remaining = tuple(row for row in built if not row.bc_collision_any)
    if len(collision) != TRAINING_COLLISION_ROWS or len(remaining) != TRAINING_REMAINING_ROWS:
        raise ValueError("B2 training collision partition drift")
    if len({row.l4_id for row in collision}) != 61:
        raise ValueError("B2 collision L4 count drift")
    return B2ScenarioSets(collision, remaining, tuple(development))


def _domain_order(rows: Sequence[B2Scenario], seed: int, group: str, repeat: int):
    def key(row: B2Scenario) -> bytes:
        digest = hashlib.sha256()
        digest.update(CURRICULUM_DOMAIN)
        digest.update(str(int(seed)).encode("ascii") + b"\0")
        digest.update(group.encode("ascii") + b"\0")
        digest.update(str(int(repeat)).encode("ascii") + b"\0")
        digest.update(row.l2_id.encode("ascii"))
        return digest.digest()

    return tuple(sorted(rows, key=lambda row: (key(row), row.l2_id)))


class B2Curriculum:
    """Frozen 8+8 complete-episode lists, identical across arms of one seed."""

    def __init__(self, scenarios: B2ScenarioSets, seed: int):
        self.scenarios = scenarios
        self.seed = int(seed)

    def _take(self, rows: Sequence[B2Scenario], group: str, count: int):
        output: list[B2Scenario] = []
        repeat = 0
        cursor = 0
        ordered = _domain_order(rows, self.seed, group, repeat)
        while len(output) < count:
            if cursor == len(ordered):
                repeat += 1
                cursor = 0
                ordered = _domain_order(rows, self.seed, group, repeat)
            output.append(ordered[cursor])
            cursor += 1
        return output

    def plan(self, iterations: int = 20) -> tuple[tuple[B2Scenario, ...], ...]:
        if int(iterations) != iterations or iterations <= 0:
            raise ValueError("B2 curriculum iterations must be positive")
        collision = self._take(
            self.scenarios.collision,
            "collision",
            int(iterations) * COLLISION_EPISODES_PER_ITERATION,
        )
        remaining = self._take(
            self.scenarios.remaining,
            "remaining",
            int(iterations) * REMAINING_EPISODES_PER_ITERATION,
        )
        planned: list[tuple[B2Scenario, ...]] = []
        for iteration in range(int(iterations)):
            start = iteration * COLLISION_EPISODES_PER_ITERATION
            c_rows = collision[start : start + COLLISION_EPISODES_PER_ITERATION]
            r_start = iteration * REMAINING_EPISODES_PER_ITERATION
            r_rows = remaining[r_start : r_start + REMAINING_EPISODES_PER_ITERATION]
            interleaved = tuple(item for pair in zip(c_rows, r_rows) for item in pair)
            if len(interleaved) != EPISODES_PER_ITERATION:
                raise AssertionError("B2 curriculum iteration size drift")
            planned.append(interleaved)
        return tuple(planned)

    def digest(self, iterations: int = 20) -> str:
        digest = hashlib.sha256(CURRICULUM_DOMAIN)
        digest.update(CURRICULUM_SCHEMA.encode("ascii") + b"\0")
        digest.update(str(self.seed).encode("ascii") + b"\0")
        for iteration, rows in enumerate(self.plan(iterations), start=1):
            for episode_index, row in enumerate(rows):
                digest.update(f"{iteration}:{episode_index}:".encode("ascii"))
                digest.update(row.l2_id.encode("ascii") + b"\n")
        return digest.hexdigest()


class ActorHistory:
    """Deployable 100 Hz histories with episode-start clamping."""

    def __init__(self) -> None:
        self._lidar: list[torch.Tensor] = []
        self._speed: list[torch.Tensor] = []
        self._steer: list[torch.Tensor] = []
        self._command_speed: list[torch.Tensor] = []

    def append(
        self,
        lidar: torch.Tensor,
        actual_speed: torch.Tensor,
        previous_applied_command: torch.Tensor,
    ) -> None:
        if lidar.ndim != 2 or lidar.shape[1] != LIDAR_BEAMS:
            raise ValueError("B2 history lidar must be [B,360]")
        if actual_speed.shape != (len(lidar), 1):
            raise ValueError("B2 history speed must be [B,1]")
        if previous_applied_command.shape != (len(lidar), 2):
            raise ValueError("B2 previous command must be [B,2]")
        values = (lidar, actual_speed, previous_applied_command)
        if any(not torch.all(torch.isfinite(value)) for value in values):
            raise ValueError("B2 history input contains nonfinite value")
        self._lidar.append((torch.clamp(lidar, 0.0, 30.0) / 30.0).detach())
        self._speed.append((actual_speed / 10.0).detach())
        self._steer.append((previous_applied_command[:, 0:1] / 0.52).detach())
        self._command_speed.append((previous_applied_command[:, 1:2] / 10.0).detach())

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._lidar:
            raise RuntimeError("B2 history is empty")
        current = len(self._lidar) - 1
        indices = [max(0, current - int(offset)) for offset in HISTORY_OFFSETS]
        lidar = torch.stack([self._lidar[index] for index in indices], dim=1)
        scalar = torch.cat(
            [
                torch.stack([self._speed[index] for index in indices], dim=1).squeeze(-1),
                torch.stack([self._steer[index] for index in indices], dim=1).squeeze(-1),
                torch.stack(
                    [self._command_speed[index] for index in indices], dim=1
                ).squeeze(-1),
            ],
            dim=1,
        )
        if lidar.shape[1:] != (len(HISTORY_OFFSETS), LIDAR_BEAMS) or scalar.shape[1:] != (24,):
            raise AssertionError("B2 history shape drift")
        return lidar, scalar


def build_privileged_features(
    obs: Mapping[str, np.ndarray],
    reference,
    reward_state: RewardState,
    opponent_speedscale: float,
) -> np.ndarray:
    """Build the frozen 12D current/past-only critic feature."""

    geometry = relative_geometry(obs, reference)
    rel_s = wrap_rel_s(
        geometry["ego_s_raw"] - geometry["opp_s_raw"], reference.track_length
    )
    phase = 2.0 * math.pi * geometry["ego_s_raw"] / reference.track_length
    value = np.asarray(
        [
            rel_s / 6.0,
            geometry["lat_gap"],
            geometry["ego_v_s"] / 10.0,
            geometry["opp_v_s"] / 10.0,
            geometry["ego_d"],
            geometry["opp_d"],
            reward_state.safe_overtake_hold_time / 0.7,
            float(reward_state.overtake_started),
            float(reward_state.safe_overtake_held),
            float(opponent_speedscale),
            math.sin(phase),
            math.cos(phase),
        ],
        dtype=np.float32,
    )
    if value.shape != (PRIVILEGED_DIM,) or not np.all(np.isfinite(value)):
        raise ValueError("B2 privileged feature is invalid")
    return value


def ordered_l2(rows: Iterable[B2Scenario]) -> tuple[str, ...]:
    return tuple(row.l2_id for row in rows)


# ---------------------------------------------------------------------------
# Live canonical episode collection
# ---------------------------------------------------------------------------


@dataclass
class B2CollectedMacro:
    """One complete macro transition before conversion to the replay buffer."""

    l2_id: str
    l4_id: str
    episode_repeat: int
    macro_index: int
    episode_start: bool
    bc_feature: np.ndarray
    lidar_history: np.ndarray
    scalar_history: np.ndarray
    privileged_feature: np.ndarray
    action: np.ndarray
    old_log_prob: float
    old_entropy: float
    entropy_intervention: float
    entropy_steer_given_intervention: float
    entropy_brake_gate_given_intervention: float
    entropy_brake_magnitude_given_brake: float
    behavior: dict[str, float | str]
    collision_value: float
    performance_value: float
    length: int = 0
    discount: float = 0.0
    collision_cost: float = 0.0
    performance_reward: float = 0.0
    terminated: bool = False
    composition_sha256: str = ""
    intervention_micro_steps: int = 0
    brake_micro_steps: int = 0
    applied_abs_steer_sum: float = 0.0
    applied_brake_sum: float = 0.0

    def validate(self) -> None:
        arrays = (
            (self.bc_feature, (1680,), "bc_feature"),
            (self.lidar_history, (len(HISTORY_OFFSETS), LIDAR_BEAMS), "lidar_history"),
            (self.scalar_history, (24,), "scalar_history"),
            (self.privileged_feature, (PRIVILEGED_DIM,), "privileged_feature"),
            (self.action, (4,), "action"),
        )
        for value, shape, name in arrays:
            if np.asarray(value).shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"B2 collected {name} is invalid")
        scalars = (
            self.old_log_prob,
            self.old_entropy,
            self.entropy_intervention,
            self.entropy_steer_given_intervention,
            self.entropy_brake_gate_given_intervention,
            self.entropy_brake_magnitude_given_brake,
            self.collision_value,
            self.performance_value,
            self.discount,
            self.collision_cost,
            self.performance_reward,
            self.applied_abs_steer_sum,
            self.applied_brake_sum,
        )
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("B2 collected macro contains nonfinite scalar")
        if not 1 <= int(self.length) <= 10:
            raise ValueError("B2 collected macro length is outside [1,10]")
        if self.discount != float(0.997 ** int(self.length)):
            raise ValueError("B2 collected macro discount drift")
        if len(self.composition_sha256) != 64:
            raise ValueError("B2 composition digest is invalid")


@dataclass(frozen=True)
class B2EpisodeResult:
    scenario: B2Scenario
    transitions: tuple[B2CollectedMacro, ...]
    arrays: Mapping[str, np.ndarray]
    outcome: object
    micro_steps: int
    external_clip_micro_steps: int

    def __post_init__(self) -> None:
        if not self.transitions or self.micro_steps <= 0:
            raise ValueError("B2 episode result is empty")
        if sum(row.length for row in self.transitions) != self.micro_steps:
            raise ValueError("B2 episode macro/micro accounting mismatch")
        if self.external_clip_micro_steps != 0:
            raise ValueError("B2 episode required external clipping")


def _critic_values(value_function, privileged: np.ndarray, device: torch.device):
    if value_function is None:
        return 0.0, 0.0
    tensor = torch.as_tensor(privileged, dtype=torch.float32, device=device).reshape(1, -1)
    with torch.no_grad():
        value = value_function(tensor)
    if isinstance(value, Mapping):
        collision = value["collision"]
        performance = value["performance"]
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        collision, performance = value
    else:
        raise TypeError("B2 critic must return collision/performance mapping or pair")
    collision_value = float(torch.as_tensor(collision).reshape(-1)[0].item())
    performance_value = float(torch.as_tensor(performance).reshape(-1)[0].item())
    if not np.isfinite(collision_value) or not np.isfinite(performance_value):
        raise ValueError("B2 critic returned nonfinite value")
    return collision_value, performance_value


def _digest_composition(digest, ledger) -> None:
    for name in (
        "raw_base",
        "deployed_base",
        "requested_residual",
        "negative_steer_headroom",
        "positive_steer_headroom",
        "brake_headroom",
        "applied_residual",
        "command",
        "external_clip_would_change",
    ):
        value = getattr(ledger, name).detach().cpu().contiguous()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())


def run_b2_episode(
    policy,
    value_function,
    device: torch.device,
    scenario: B2Scenario,
    behavior_config,
    pilot_seed: int,
    episode_repeat: int,
    *,
    sim_duration: float = 8.0,
) -> B2EpisodeResult:
    """Collect one complete on-policy episode with canonical evaluator mechanics."""

    # Imports stay local so pure scenario/history tests do not require Gym.
    import gym
    from f110_gym.envs.base_classes import Integrator
    from bplus_v22.exploration import ActionNoiseKey
    from bplus_v22.remediated_model import HierarchicalResidualDistribution
    from d25.oracle import _resolve_case, _trajectory_arrays, classify_trajectory
    from demonstration import setup_opp_planner
    from latticeplanner.utils import obsDict2oppoArray, project_point_to_centerline
    from ppo_utils import RewardWeights, compute_shaped_reward
    from utils import load_positions_and_speeds_from_params, load_reference_line

    np.random.seed(42)
    resolved = _resolve_case(scenario.simulator_case())
    map_name = resolved["map_name"]
    environment = gym.make(
        "f110-v0",
        map=f"f1tenth_racetracks/{map_name}/{map_name}_map",
        map_ext=".png",
        num_agents=2,
        timestep=0.01,
        integrator=Integrator.RK4,
    )
    try:
        positions, initial_speeds = load_positions_and_speeds_from_params(resolved, map_name)
        opponent = setup_opp_planner(map_name, resolved["opp_raceline"])
        opponent.tracker.prev_error = 0.0
        opponent.prev_opp_pose = np.asarray([0.0, 0.0])
        opponent.prev_traj_local = np.zeros_like(opponent.prev_traj_local)
        opponent.best_traj = None
        opponent.goal_grid = None
        tracker_count = 0
        tracker_steps = 10
        opp_traj = None

        hidden = policy.zero_hidden(1, device)
        previous_speed = float(initial_speeds[0]) * 0.9
        previous_command = torch.zeros((1, 2), dtype=torch.float32, device=device)
        history = ActorHistory()
        reference = load_reference_line(map_name, "raceline1")
        centerline_wp = np.loadtxt(
            f"f1tenth_racetracks/{map_name}/raceline1.csv", delimiter=";", skiprows=1
        )
        centerline = centerline_wp[:, 1:3]
        track_length = float(
            sum(
                np.linalg.norm(centerline[index + 1] - centerline[index])
                for index in range(len(centerline) - 1)
            )
        )
        obs, _, done, _ = environment.reset(poses=positions)
        reward_weights = RewardWeights()
        reward_state = RewardState.from_obs(obs, reference, reward_weights)
        initial_ego_progress, _ = project_point_to_centerline(
            np.asarray([obs["poses_x"][0], obs["poses_y"][0]]), centerline
        )
        initial_opp_progress, _ = project_point_to_centerline(
            np.asarray([obs["poses_x"][1], obs["poses_y"][1]]), centerline
        )
        final_state = "overtaking" if initial_ego_progress > initial_opp_progress else "following"
        final_ego_progress = float(initial_ego_progress)
        final_opp_progress = float(initial_opp_progress)
        collision = ego_collision = opp_collision = False
        records = {
            name: []
            for name in (
                "time",
                "ego_lidar",
                "opp_lidar",
                "ego_desired_steer",
                "ego_desired_speed",
                "ego_actual_speed",
                "ego_pose",
                "ego_progress",
                "opp_desired_steer",
                "opp_desired_speed",
                "opp_actual_speed",
                "opp_pose",
                "opp_progress",
            )
        }
        transitions: list[B2CollectedMacro] = []
        held_action = None
        current: B2CollectedMacro | None = None
        composition_digest = None
        lap_time = 0.0
        micro_step = 0
        macro_index = 0
        policy.eval()

        while not done and lap_time < float(sim_duration):
            ego_lidar = np.asarray(obs["scans"][0]).reshape(-1)
            if len(ego_lidar) != LIDAR_BEAMS:
                ego_lidar = ego_lidar[
                    np.linspace(0, len(ego_lidar) - 1, LIDAR_BEAMS, dtype=int)
                ]
            opp_lidar = np.asarray(obs["scans"][1]).reshape(-1)
            if len(opp_lidar) != LIDAR_BEAMS:
                opp_lidar = opp_lidar[
                    np.linspace(0, len(opp_lidar) - 1, LIDAR_BEAMS, dtype=int)
                ]
            lidar_tensor = torch.as_tensor(
                ego_lidar, dtype=torch.float32, device=device
            ).reshape(1, 1, LIDAR_BEAMS)
            speed_tensor = torch.tensor(
                [[[previous_speed]]], dtype=torch.float32, device=device
            )
            actual_speed = torch.tensor(
                [[float(obs["linear_vels_x"][0])]], dtype=torch.float32, device=device
            )
            with torch.no_grad():
                base, bc_feature, next_hidden = policy.bc_step(
                    lidar_tensor, speed_tensor, hidden
                )
            history.append(lidar_tensor[:, -1], actual_speed, previous_command)

            if micro_step % 10 == 0:
                if current is not None:
                    current.composition_sha256 = composition_digest.hexdigest()
                    current.discount = float(0.997 ** current.length)
                    current.validate()
                    transitions.append(current)
                lidar_history, scalar_history = history.tensors()
                exploration = behavior_config.as_batch(
                    torch.zeros((1, 1), dtype=bc_feature.dtype, device=device)
                )
                with torch.no_grad():
                    distribution = policy.behavior_distribution(
                        bc_feature, lidar_history, scalar_history, exploration
                    )
                    held_action = distribution.sample_keyed(
                        [
                            ActionNoiseKey(
                                int(pilot_seed),
                                scenario.l2_id,
                                int(episode_repeat),
                                int(macro_index),
                            )
                        ]
                    )
                    old_log_prob = float(distribution.log_prob(held_action).item())
                    entropy_components = distribution.entropy_components()
                    old_entropy = float(entropy_components["total"].item())
                privileged = build_privileged_features(
                    obs, reference, reward_state, resolved["opp_speedscale"]
                )
                collision_value, performance_value = _critic_values(
                    value_function, privileged, device
                )
                current = B2CollectedMacro(
                    l2_id=scenario.l2_id,
                    l4_id=scenario.l4_id,
                    episode_repeat=int(episode_repeat),
                    macro_index=macro_index,
                    episode_start=macro_index == 0,
                    bc_feature=bc_feature[0].detach().cpu().numpy().astype(np.float32),
                    lidar_history=lidar_history[0].detach().cpu().numpy().astype(np.float32),
                    scalar_history=scalar_history[0].detach().cpu().numpy().astype(np.float32),
                    privileged_feature=privileged,
                    action=held_action.as_tensor()[0].detach().cpu().numpy().astype(np.float32),
                    old_log_prob=old_log_prob,
                    old_entropy=old_entropy,
                    entropy_intervention=float(
                        entropy_components["intervention"].item()
                    ),
                    entropy_steer_given_intervention=float(
                        entropy_components["steer_given_intervention"].item()
                    ),
                    entropy_brake_gate_given_intervention=float(
                        entropy_components["brake_gate_given_intervention"].item()
                    ),
                    entropy_brake_magnitude_given_brake=float(
                        entropy_components["brake_magnitude_given_brake"].item()
                    ),
                    behavior=behavior_config.as_dict(),
                    collision_value=collision_value,
                    performance_value=performance_value,
                )
                composition_digest = hashlib.sha256(
                    b"end2race:bplus-v2.2:b2-composition:v1\0"
                )
                macro_index += 1
            if held_action is None or current is None or composition_digest is None:
                raise AssertionError("B2 macro action was not initialized")
            with torch.no_grad():
                ledger = HierarchicalResidualDistribution.compose(base, held_action)
            if torch.any(ledger.external_clip_would_change):
                raise AssertionError("B2 composed action requires external clipping")
            _digest_composition(composition_digest, ledger)
            command = ledger.command[0]
            ego_steer = float(command[0].item())
            ego_speed = float(command[1].item())
            previous_command = command.reshape(1, 2).detach()
            current.length += 1
            current.intervention_micro_steps += int(held_action.intervention_gate.item())
            current.brake_micro_steps += int(held_action.brake_gate.item())
            current.applied_abs_steer_sum += abs(float(ledger.applied_residual[0, 0].item()))
            current.applied_brake_sum += -float(ledger.applied_residual[0, 1].item())

            if tracker_count == 0:
                opp_traj = opponent.plan(
                    obs["poses_x"][1],
                    obs["poses_y"][1],
                    obs["poses_theta"][1],
                    obsDict2oppoArray(obs, 1),
                    obs["linear_vels_x"][1],
                )
            opp_steer, opp_speed = opponent.tracker.plan(
                obs["poses_x"][1],
                obs["poses_y"][1],
                obs["poses_theta"][1],
                obs["linear_vels_x"][1],
                opp_traj,
            )
            opp_steer = float(np.clip(opp_steer, -0.52, 0.52))
            opp_speed = float(opp_speed) * resolved["opp_speedscale"]
            ego_progress, _ = project_point_to_centerline(
                np.asarray([obs["poses_x"][0], obs["poses_y"][0]]), centerline
            )
            opp_progress, _ = project_point_to_centerline(
                np.asarray([obs["poses_x"][1], obs["poses_y"][1]]), centerline
            )
            if ego_progress < initial_ego_progress - track_length / 2:
                ego_progress += track_length
            if opp_progress < initial_opp_progress - track_length / 2:
                opp_progress += track_length
            values = {
                "time": float(lap_time),
                "ego_lidar": ego_lidar.copy(),
                "opp_lidar": opp_lidar.copy(),
                "ego_desired_steer": ego_steer,
                "ego_desired_speed": ego_speed,
                "ego_actual_speed": float(obs["linear_vels_x"][0]),
                "ego_pose": [obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]],
                "ego_progress": float(ego_progress),
                "opp_desired_steer": opp_steer,
                "opp_desired_speed": opp_speed,
                "opp_actual_speed": float(obs["linear_vels_x"][1]),
                "opp_pose": [obs["poses_x"][1], obs["poses_y"][1], obs["poses_theta"][1]],
                "opp_progress": float(opp_progress),
            }
            for name, value in values.items():
                records[name].append(value)
            previous_speed = float(obs["linear_vels_x"][0])
            obs, timestep, done, _ = environment.step(
                np.asarray([[ego_steer, ego_speed], [opp_steer, opp_speed]])
            )
            lap_time += float(timestep)
            # This call updates only the historical safe-pass state used by the
            # critic.  Its shaped reward return is deliberately discarded.
            compute_shaped_reward(obs, reward_state, reference, reward_weights, float(timestep))
            ego_progress, _ = project_point_to_centerline(
                np.asarray([obs["poses_x"][0], obs["poses_y"][0]]), centerline
            )
            opp_progress, _ = project_point_to_centerline(
                np.asarray([obs["poses_x"][1], obs["poses_y"][1]]), centerline
            )
            if ego_progress < initial_ego_progress - track_length / 2:
                ego_progress += track_length
            if opp_progress < initial_opp_progress - track_length / 2:
                opp_progress += track_length
            final_state = "overtaking" if ego_progress > opp_progress else "following"
            final_ego_progress = float(ego_progress)
            final_opp_progress = float(opp_progress)
            if np.any(obs["collisions"]):
                collision = True
                ego_collision = bool(obs["collisions"][0])
                opp_collision = bool(np.any(obs["collisions"][1:]))
                done = True
            tracker_count = (tracker_count + 1) % tracker_steps
            hidden = next_hidden.detach()
            micro_step += 1

        if current is None or composition_digest is None:
            raise AssertionError("B2 episode produced no macro")
        current.composition_sha256 = composition_digest.hexdigest()
        current.discount = float(0.997 ** current.length)
        transitions.append(current)
        state_label = "collision" if collision else final_state
        terminal = {
            "collision": collision,
            "ego_collision": ego_collision,
            "opp_collision": opp_collision,
            "final_time": float(lap_time),
            "final_ego_pose": [obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]],
            "final_opp_pose": [obs["poses_x"][1], obs["poses_y"][1], obs["poses_theta"][1]],
            "final_ego_progress": final_ego_progress,
            "final_opp_progress": final_opp_progress,
        }
        arrays = _trajectory_arrays(records, terminal, state_label)
        outcome = classify_trajectory(arrays, map_name)
        transitions[-1].terminated = True
        transitions[-1].collision_cost = float(outcome.collision_any)
        transitions[-1].performance_reward = float(outcome.corrected_outcome3 == "overtake")
        for row in transitions:
            row.validate()
        return B2EpisodeResult(
            scenario,
            tuple(transitions),
            arrays,
            outcome,
            micro_step,
            0,
        )
    finally:
        environment.close()
        gc.collect()
