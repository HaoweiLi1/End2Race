"""PPO scenario generation and deterministic role queues."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import yaml

from utils import find_corresponding_waypoint, load_positions_and_speeds_from_params, load_raceline_waypoints

with Path(__file__).with_name("ppo_config.yaml").open("r", encoding="utf-8") as file:
    PPO_CONFIG = yaml.safe_load(file)

EGO_RACELINE = str(PPO_CONFIG["ego_raceline"])
OPPONENT_RACELINES = tuple(PPO_CONFIG["opponent_racelines"])
ORDINARY_SPEED_SCALES = tuple(float(value) for value in PPO_CONFIG["ordinary_speed_scales"])
ORDINARY_INTERVAL_INDEX = int(PPO_CONFIG["ordinary_interval_index"])
ORDINARY_STARTPOINT_COUNT = int(PPO_CONFIG["ordinary_startpoint_count"])
ORDINARY_STARTPOINT_MIN_DISTANCE = float(PPO_CONFIG["ordinary_startpoint_min_distance"])
COLLISION_INTERVAL_INDICES = tuple(int(value) for value in PPO_CONFIG["collision_interval_indices"])
COLLISION_SPEED_SCALES = tuple(float(value) for value in PPO_CONFIG["collision_speed_scales"])
COLLISION_STARTPOINT_COUNT = int(PPO_CONFIG["collision_startpoint_count"])
COLLISION_STARTPOINT_MIN_DISTANCE = float(PPO_CONFIG["collision_startpoint_min_distance"])
SIM_DURATION = float(PPO_CONFIG["episode_horizon"])
TIMESTEP = float(PPO_CONFIG["simulator_timestep"])


@dataclass
class EpisodeResetSpec:
    poses: np.ndarray
    initial_speed_feature: float
    scenario: dict[str, Any]


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    pool: str
    startpoint_ordinal: int
    ego_idx: int
    opp_idx: int
    opp_raceline: str
    opp_speedscale: float
    interval_idx: int
    map_name: str
    ego_raceline: str = EGO_RACELINE
    sim_duration: float = SIM_DURATION
    timestep: float = TIMESTEP
    integrator: str = "RK4"

    def to_reset_spec(self, env_role: str) -> EpisodeResetSpec:
        scenario = asdict(self)
        scenario["opponent_speed_scale"] = self.opp_speedscale
        scenario["sampler_branch"] = env_role
        scenario["env_role"] = env_role
        poses, initial_speeds = load_positions_and_speeds_from_params(scenario, self.map_name)
        return EpisodeResetSpec(np.asarray(poses, dtype=np.float64), float(initial_speeds[0] * 0.9), scenario)


def _raceline_data(map_name: str) -> np.ndarray:
    path = Path("f1tenth_racetracks") / map_name / f"{EGO_RACELINE}.csv"
    data = np.loadtxt(path, delimiter=";", comments="#", dtype=np.float64)
    if np.linalg.norm(data[-1, 1:3] - data[0, 1:3]) > 1e-9:
        raise ValueError(f"{EGO_RACELINE}.csv must contain the duplicated closing endpoint")
    return data


def generate_training_startpoints(map_name: str, startpoint_count: int, minimum_evaluation_distance: float) -> tuple[int, ...]:
    """Generate mid-gap training points away from the evaluation panel."""
    data = _raceline_data(map_name)
    unique = data[:-1]
    evaluation_indices = np.arange(ORDINARY_STARTPOINT_COUNT) * (len(data) - 1) // (ORDINARY_STARTPOINT_COUNT - 1)
    evaluation_xy = unique[evaluation_indices % len(unique), 1:3]
    distances = np.linalg.norm(unique[:, None, 1:3] - evaluation_xy[None, :, :], axis=2)
    allowed = np.flatnonzero(np.min(distances, axis=1) >= minimum_evaluation_distance - 1e-12)
    track_length = float(data[-1, 0])
    selected = []
    for target in (np.arange(startpoint_count) + 0.5) * track_length / startpoint_count:
        progress_delta = np.abs(unique[allowed, 0] - target)
        progress_delta = np.minimum(progress_delta, track_length - progress_delta)
        order = np.lexsort((allowed, progress_delta))
        pick = next((int(allowed[position]) for position in order if int(allowed[position]) not in selected), None)
        if pick is None:
            raise RuntimeError(f"Map {map_name} has too few startpoints outside the evaluation-separation zone")
        selected.append(pick)
    return tuple(sorted(selected, key=lambda index: float(unique[index, 0])))


def ordinary_scenarios(map_name: str) -> tuple[ScenarioSpec, ...]:
    ego_waypoints = load_raceline_waypoints(map_name, f"{EGO_RACELINE}.csv")
    opponent_waypoints = {raceline: load_raceline_waypoints(map_name, f"{raceline}.csv") for raceline in OPPONENT_RACELINES}
    scenarios = []
    startpoints = generate_training_startpoints(map_name, ORDINARY_STARTPOINT_COUNT, ORDINARY_STARTPOINT_MIN_DISTANCE)
    for ordinal, ego_idx in enumerate(startpoints):
        ego_waypoint = ego_waypoints[ego_idx]
        for opp_raceline in OPPONENT_RACELINES:
            mapped_index = ego_idx if opp_raceline == EGO_RACELINE else int(find_corresponding_waypoint(ego_waypoint, opponent_waypoints[opp_raceline]))
            opp_idx = (mapped_index + ORDINARY_INTERVAL_INDEX) % len(opponent_waypoints[opp_raceline])
            for speed_scale in ORDINARY_SPEED_SCALES:
                scenario_id = f"ordinary-sp{ordinal:02d}-ego{ego_idx:04d}-{opp_raceline}-v{int(100 * speed_scale):03d}"
                scenarios.append(ScenarioSpec(scenario_id, "ordinary", ordinal, ego_idx, opp_idx, opp_raceline, speed_scale, ORDINARY_INTERVAL_INDEX, map_name))
    expected_count = ORDINARY_STARTPOINT_COUNT * len(OPPONENT_RACELINES) * len(ORDINARY_SPEED_SCALES)
    if len(scenarios) != expected_count or len({scenario.scenario_id for scenario in scenarios}) != expected_count:
        raise RuntimeError(f"Ordinary scenario panel must contain {expected_count:,} unique scenarios")
    return tuple(scenarios)


def expanded_scenarios(map_name: str) -> tuple[ScenarioSpec, ...]:
    ego_waypoints = load_raceline_waypoints(map_name, f"{EGO_RACELINE}.csv")
    opponent_waypoints = {raceline: load_raceline_waypoints(map_name, f"{raceline}.csv") for raceline in OPPONENT_RACELINES}
    scenarios = []
    startpoints = generate_training_startpoints(map_name, COLLISION_STARTPOINT_COUNT, COLLISION_STARTPOINT_MIN_DISTANCE)
    for ordinal, ego_idx in enumerate(startpoints):
        ego_waypoint = ego_waypoints[ego_idx]
        for opp_raceline in OPPONENT_RACELINES:
            mapped_index = ego_idx if opp_raceline == EGO_RACELINE else int(find_corresponding_waypoint(ego_waypoint, opponent_waypoints[opp_raceline]))
            for interval_idx in COLLISION_INTERVAL_INDICES:
                opp_idx = (mapped_index + interval_idx) % (len(opponent_waypoints[opp_raceline]) - 1)
                for speed_scale in COLLISION_SPEED_SCALES:
                    scenario_id = f"collision-sp{ordinal:03d}-ego{ego_idx:04d}-{opp_raceline}-i{interval_idx:02d}-v{round(100 * speed_scale):03d}"
                    scenarios.append(ScenarioSpec(scenario_id, "collision", ordinal, ego_idx, opp_idx, opp_raceline, speed_scale, interval_idx, map_name))
    expected_count = COLLISION_STARTPOINT_COUNT * len(OPPONENT_RACELINES) * len(COLLISION_INTERVAL_INDICES) * len(COLLISION_SPEED_SCALES)
    if len(scenarios) != expected_count:
        raise RuntimeError(f"Collision scenario panel must contain {expected_count:,} scenarios")
    if len({scenario.scenario_id for scenario in scenarios}) != expected_count:
        raise RuntimeError(f"Collision scenario panel must contain {expected_count:,} unique scenario IDs")
    return tuple(scenarios)


class RoleScenarioQueue:

    def __init__(self, scenarios: Sequence[ScenarioSpec], seed_sequence: np.random.SeedSequence):
        self.scenarios = tuple(scenarios)
        self.rng = np.random.default_rng(seed_sequence)
        self.order = np.empty(0, dtype=np.int64)
        self.cursor = 0
        self.cycle = 0
        self._start_cycle()

    def _start_cycle(self) -> None:
        self.order = np.asarray(self.rng.permutation(len(self.scenarios)), dtype=np.int64)
        self.cursor = 0
        self.cycle += 1

    def next(self) -> ScenarioSpec:
        if self.cursor == len(self.order):
            self._start_cycle()
        scenario = self.scenarios[int(self.order[self.cursor])]
        self.cursor += 1
        return scenario

    def state_dict(self) -> dict[str, Any]:
        return {"order": self.order.copy(), "cursor": self.cursor, "cycle": self.cycle, "rng_state": deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        order = np.asarray(state["order"], dtype=np.int64)
        if sorted(order.tolist()) != list(range(len(self.scenarios))):
            raise ValueError("Scenario queue order is invalid")
        cursor = int(state["cursor"])
        cycle = int(state["cycle"])
        if not 0 <= cursor <= len(order) or cycle <= 0:
            raise ValueError("Scenario queue cursor or cycle is invalid")
        self.order = order.copy()
        self.cursor = cursor
        self.cycle = cycle
        self.rng.bit_generator.state = deepcopy(state["rng_state"])


class ScenarioScheduler:

    def __init__(self, seed: int, collision_scenarios: Sequence[ScenarioSpec], ordinary_scenarios: Sequence[ScenarioSpec]):
        collision_seed, ordinary_seed = np.random.SeedSequence(seed).spawn(2)
        self.collision = RoleScenarioQueue(collision_scenarios, collision_seed)
        self.ordinary = RoleScenarioQueue(ordinary_scenarios, ordinary_seed)

    def next(self, rank: int) -> EpisodeResetSpec:
        env_role = "collision" if rank % 2 == 0 else "ordinary"
        queue = self.collision if env_role == "collision" else self.ordinary
        return queue.next().to_reset_spec(env_role)

    def state_dict(self) -> dict[str, Any]:
        return {"collision": self.collision.state_dict(), "ordinary": self.ordinary.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.collision.load_state_dict(state["collision"])
        self.ordinary.load_state_dict(state["ordinary"])
