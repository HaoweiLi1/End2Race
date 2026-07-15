"""Fixed PPO V1 scenario pools and reproducible 75/25 reset sampler."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from rl.end2race_gymnasium_env import EpisodeResetSpec
from utils import find_corresponding_waypoint, load_positions_and_speeds_from_params, load_raceline_waypoints


MAP_NAME = "Austin"
EGO_RACELINE = "raceline1"
OPPONENT_RACELINES = ("raceline0", "raceline1", "raceline2")
OPPONENT_SPEED_SCALES = (0.5, 0.6, 0.7, 0.8)
INTERVAL_IDX = 15
SIM_DURATION = 8.0
TIMESTEP = 0.01

TRAINING_STARTPOINTS = (
    21, 63, 110, 151, 189, 231, 272, 319, 356, 398,
    440, 487, 519, 571, 613, 650, 692, 739, 780, 823,
    861, 904, 949, 989, 1032, 1064, 1106, 1149, 1189, 1234,
    1272, 1315, 1356, 1404, 1441, 1488, 1525, 1567, 1608, 1656,
    1703, 1745, 1787, 1824, 1865, 1912, 1954, 1997, 2033, 2075,
)

EVALUATION_STARTPOINTS = (
    0, 42, 85, 128, 171, 213, 256, 299, 342, 384,
    427, 470, 513, 556, 598, 641, 684, 727, 769, 812,
    855, 898, 941, 983, 1026, 1069, 1112, 1154, 1197, 1240,
    1283, 1326, 1368, 1411, 1454, 1497, 1539, 1582, 1625, 1668,
    1711, 1753, 1796, 1839, 1882, 1924, 1967, 2010, 2053, 2096,
)


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    pool: str
    startpoint_ordinal: int
    ego_idx: int
    opp_idx: int
    opp_raceline: str
    opp_speedscale: float
    map_name: str = MAP_NAME
    ego_raceline: str = EGO_RACELINE
    interval_idx: int = INTERVAL_IDX
    sim_duration: float = SIM_DURATION
    timestep: float = TIMESTEP
    integrator: str = "RK4"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_reset_spec(self, *, sampler_branch: str | None = None) -> EpisodeResetSpec:
        scenario = self.to_dict()
        scenario["opponent_speed_scale"] = self.opp_speedscale
        if sampler_branch is not None:
            scenario["sampler_branch"] = sampler_branch
        poses, initial_speeds = load_positions_and_speeds_from_params(scenario, self.map_name)
        return EpisodeResetSpec(
            poses=np.asarray(poses, dtype=np.float64),
            initial_speed_feature=float(initial_speeds[0] * 0.9),
            scenario=scenario,
        )


def _scenario_id(pool: str, ordinal: int, ego_idx: int, opp_raceline: str, speed_scale: float) -> str:
    speed_code = int(round(100.0 * speed_scale))
    return f"{pool}-sp{ordinal:02d}-ego{ego_idx:04d}-{opp_raceline}-v{speed_code:03d}"


def build_scenario_pool(pool: str, startpoints: Sequence[int]) -> tuple[ScenarioSpec, ...]:
    if pool not in {"training", "evaluation"}:
        raise ValueError(f"Unknown PPO V1 scenario pool: {pool}")
    if len(startpoints) != 50:
        raise ValueError(f"PPO V1 {pool} pool requires exactly 50 startpoints")
    ego_waypoints = load_raceline_waypoints(MAP_NAME, f"{EGO_RACELINE}.csv")
    opponent_waypoints = {
        raceline: load_raceline_waypoints(MAP_NAME, f"{raceline}.csv")
        for raceline in OPPONENT_RACELINES
    }
    scenarios: list[ScenarioSpec] = []
    for ordinal, ego_idx in enumerate(startpoints):
        ego_waypoint = ego_waypoints[ego_idx % len(ego_waypoints)]
        for opp_raceline in OPPONENT_RACELINES:
            if opp_raceline == EGO_RACELINE:
                mapped_index = ego_idx % len(ego_waypoints)
            else:
                mapped_index = int(find_corresponding_waypoint(ego_waypoint, opponent_waypoints[opp_raceline]))
            opp_idx = (mapped_index + INTERVAL_IDX) % len(opponent_waypoints[opp_raceline])
            for speed_scale in OPPONENT_SPEED_SCALES:
                scenarios.append(
                    ScenarioSpec(
                        scenario_id=_scenario_id(pool, ordinal, ego_idx, opp_raceline, speed_scale),
                        pool=pool,
                        startpoint_ordinal=ordinal,
                        ego_idx=int(ego_idx),
                        opp_idx=int(opp_idx),
                        opp_raceline=opp_raceline,
                        opp_speedscale=float(speed_scale),
                    )
                )
    if len(scenarios) != 600 or len({scenario.scenario_id for scenario in scenarios}) != 600:
        raise RuntimeError(f"PPO V1 {pool} pool expansion did not produce 600 unique scenarios")
    return tuple(scenarios)


def training_scenarios() -> tuple[ScenarioSpec, ...]:
    return build_scenario_pool("training", TRAINING_STARTPOINTS)


def evaluation_scenarios() -> tuple[ScenarioSpec, ...]:
    return build_scenario_pool("evaluation", EVALUATION_STARTPOINTS)


def classify_bc_ego_collisions(rows: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    rows = list(rows)
    if len(rows) != 600:
        raise ValueError(f"BC training classification must contain 600 rows, got {len(rows)}")
    errors = [row for row in rows if row.get("outcome") == "error"]
    if errors:
        raise ValueError(f"BC training classification contains {len(errors)} errors")
    collision_ids = tuple(
        str(row["scenario_id"])
        for row in rows
        if row.get("outcome") == "ego_collision"
    )
    if not collision_ids:
        raise ValueError("BC training classification produced an empty bc_ego_collision_ids set")
    return collision_ids


class FixedMixtureScenarioSampler:
    """Sample 75% full training pool and 25% fixed BC ego-collision pool."""

    def __init__(
        self,
        scenarios: Sequence[ScenarioSpec],
        bc_ego_collision_ids: Sequence[str],
        *,
        collision_probability: float = 0.25,
    ) -> None:
        if len(scenarios) != 600:
            raise ValueError(f"Training sampler requires 600 scenarios, got {len(scenarios)}")
        if not 0.0 <= collision_probability <= 1.0:
            raise ValueError("collision_probability must be in [0, 1]")
        by_id = {scenario.scenario_id: scenario for scenario in scenarios}
        if len(by_id) != len(scenarios):
            raise ValueError("Training scenario IDs must be unique")
        missing = sorted(set(bc_ego_collision_ids) - set(by_id))
        if missing:
            raise ValueError(f"BC collision IDs are absent from training pool: {missing[:3]}")
        collision_scenarios = tuple(by_id[scenario_id] for scenario_id in bc_ego_collision_ids)
        if not collision_scenarios:
            raise ValueError("bc_ego_collision_ids must be non-empty")
        self.scenarios = tuple(scenarios)
        self.bc_collision_scenarios = collision_scenarios
        self.collision_probability = float(collision_probability)

    def sample(self, rng: np.random.Generator) -> tuple[ScenarioSpec, str]:
        if float(rng.random()) < self.collision_probability:
            pool = self.bc_collision_scenarios
            branch = "bc_ego_collision"
        else:
            pool = self.scenarios
            branch = "all_training"
        return pool[int(rng.integers(0, len(pool)))], branch

    def __call__(self, rng: np.random.Generator) -> EpisodeResetSpec:
        scenario, branch = self.sample(rng)
        return deepcopy(scenario.to_reset_spec(sampler_branch=branch))
