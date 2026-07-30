"""PPO scenario generation and deterministic role queues."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import yaml

from utils import (
    find_corresponding_waypoint,
    get_circular_startpoints,
    load_positions_and_speeds_from_params,
    load_raceline_waypoints,
)

with Path(__file__).with_name("ppo_config.yaml").open("r", encoding="utf-8") as file:
    PPO_CONFIG = yaml.safe_load(file)

EGO_RACELINE = str(PPO_CONFIG["ego_raceline"])
OPPONENT_RACELINES = tuple(PPO_CONFIG["opponent_racelines"])
EVALUATION_STARTPOINT_COUNT = 50
ORDINARY_SPEED_SCALES = tuple(float(value) for value in PPO_CONFIG["ordinary_speed_scales"])
ORDINARY_INTERVAL_INDEX = int(PPO_CONFIG["ordinary_interval_index"])
ORDINARY_STARTPOINT_COUNT = int(PPO_CONFIG["ordinary_startpoint_count"])
ORDINARY_STARTPOINT_MIN_DISTANCE = float(PPO_CONFIG["ordinary_startpoint_min_distance"])
COLLISION_INTERVAL_INDICES = tuple(int(value) for value in PPO_CONFIG["collision_interval_indices"])
COLLISION_SPEED_SCALES = tuple(float(value) for value in PPO_CONFIG["collision_speed_scales"])
COLLISION_STARTPOINT_COUNT = int(PPO_CONFIG["collision_startpoint_count"])
COLLISION_STARTPOINT_MIN_DISTANCE = float(PPO_CONFIG["collision_startpoint_min_distance"])
HARD_NEIGHBOR_MAX_CANDIDATES_PER_FAMILY = int(PPO_CONFIG["hard_neighbor_max_candidates_per_family"])
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


def evaluation_startpoints(map_name: str) -> tuple[int, ...]:
    data = _raceline_data(map_name)
    indices = np.arange(EVALUATION_STARTPOINT_COUNT) * (len(data) - 1) // (EVALUATION_STARTPOINT_COUNT - 1)
    return tuple(int(index) for index in indices)


def generate_separated_startpoints(map_name: str, startpoint_count: int, minimum_evaluation_distance: float) -> tuple[int, ...]:
    data = _raceline_data(map_name)
    unique = data[:-1]
    evaluation_indices = np.asarray(evaluation_startpoints(map_name), dtype=np.int64)
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


def ordinary_startpoints(
    map_name: str,
    startpoint_count: int = ORDINARY_STARTPOINT_COUNT,
) -> tuple[int, ...]:
    """Return the baseline 50 starts plus deterministic maximin additions.

    The first 50 entries deliberately retain the original ordinary panel and
    therefore preserve its scenario IDs.  Additional entries are selected
    only from waypoints at least the configured physical distance from the
    current circular Austin600 startpoint panel.
    """

    baseline = generate_separated_startpoints(
        map_name,
        ORDINARY_STARTPOINT_COUNT,
        ORDINARY_STARTPOINT_MIN_DISTANCE,
    )
    if startpoint_count == ORDINARY_STARTPOINT_COUNT:
        return baseline
    if startpoint_count < ORDINARY_STARTPOINT_COUNT:
        raise ValueError(
            "ordinary startpoint extension cannot remove baseline startpoints"
        )

    data = _raceline_data(map_name)
    unique = data[:-1]
    evaluation_indices = np.asarray(
        get_circular_startpoints(
            map_name,
            f"{EGO_RACELINE}.csv",
            EVALUATION_STARTPOINT_COUNT,
            0,
        ),
        dtype=np.int64,
    )
    evaluation_xy = unique[evaluation_indices, 1:3]
    physical_distances = np.linalg.norm(
        unique[:, None, 1:3] - evaluation_xy[None, :, :],
        axis=2,
    )
    allowed = np.flatnonzero(
        np.min(physical_distances, axis=1)
        >= ORDINARY_STARTPOINT_MIN_DISTANCE - 1e-12
    )
    allowed_set = set(int(index) for index in allowed)
    if not set(baseline).issubset(allowed_set):
        raise RuntimeError(
            "Baseline ordinary startpoints violate the current evaluation "
            "separation contract"
        )
    if len(allowed) < startpoint_count:
        raise RuntimeError(
            f"Map {map_name} has too few evaluation-separated startpoints"
        )

    selected = list(baseline)
    progress = unique[:, 0]
    track_length = float(data[-1, 0])
    while len(selected) < startpoint_count:
        candidate_progress = progress[allowed, None]
        selected_progress = progress[np.asarray(selected, dtype=np.int64)][
            None, :
        ]
        progress_distances = np.abs(candidate_progress - selected_progress)
        progress_distances = np.minimum(
            progress_distances,
            track_length - progress_distances,
        )
        minimum_progress_distance = np.min(progress_distances, axis=1)
        minimum_progress_distance[
            np.isin(allowed, np.asarray(selected, dtype=np.int64))
        ] = -np.inf
        best_distance = float(np.max(minimum_progress_distance))
        best_positions = np.flatnonzero(
            minimum_progress_distance >= best_distance - 1e-12
        )
        if len(best_positions) == 0 or not np.isfinite(best_distance):
            raise RuntimeError(
                f"Map {map_name} cannot extend the ordinary startpoint panel"
            )
        selected.append(int(allowed[int(best_positions[0])]))
    return tuple(selected)


def collision_candidate_startpoints(map_name: str) -> tuple[int, ...]:
    return generate_separated_startpoints(map_name, COLLISION_STARTPOINT_COUNT, COLLISION_STARTPOINT_MIN_DISTANCE)


def ordinary_scenarios(
    map_name: str,
    startpoint_count: int = ORDINARY_STARTPOINT_COUNT,
) -> tuple[ScenarioSpec, ...]:
    ego_waypoints = load_raceline_waypoints(map_name, f"{EGO_RACELINE}.csv")
    opponent_waypoints = {raceline: load_raceline_waypoints(map_name, f"{raceline}.csv") for raceline in OPPONENT_RACELINES}
    scenarios = []
    for ordinal, ego_idx in enumerate(
        ordinary_startpoints(map_name, startpoint_count)
    ):
        ego_waypoint = ego_waypoints[ego_idx]
        for opp_raceline in OPPONENT_RACELINES:
            mapped_index = ego_idx if opp_raceline == EGO_RACELINE else int(find_corresponding_waypoint(ego_waypoint, opponent_waypoints[opp_raceline]))
            opp_idx = (mapped_index + ORDINARY_INTERVAL_INDEX) % len(opponent_waypoints[opp_raceline])
            for speed_scale in ORDINARY_SPEED_SCALES:
                scenario_id = f"ordinary-sp{ordinal:02d}-ego{ego_idx:04d}-{opp_raceline}-v{int(100 * speed_scale):03d}"
                scenarios.append(ScenarioSpec(scenario_id, "ordinary", ordinal, ego_idx, opp_idx, opp_raceline, speed_scale, ORDINARY_INTERVAL_INDEX, map_name))
    expected_count = startpoint_count * len(OPPONENT_RACELINES) * len(ORDINARY_SPEED_SCALES)
    if len(scenarios) != expected_count or len({scenario.scenario_id for scenario in scenarios}) != expected_count:
        raise RuntimeError(f"Ordinary scenario panel must contain {expected_count:,} unique scenarios")
    return tuple(scenarios)


def expanded_scenarios(map_name: str) -> tuple[ScenarioSpec, ...]:
    ego_waypoints = load_raceline_waypoints(map_name, f"{EGO_RACELINE}.csv")
    opponent_waypoints = {raceline: load_raceline_waypoints(map_name, f"{raceline}.csv") for raceline in OPPONENT_RACELINES}
    scenarios = []
    for ordinal, ego_idx in enumerate(collision_candidate_startpoints(map_name)):
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

    def __init__(
        self,
        seed: int,
        collision_scenarios: Sequence[ScenarioSpec],
        ordinary_scenarios: Sequence[ScenarioSpec],
        hard_neighbor_fraction: float | None = None,
        ordinary_offline_fast_fraction: float | None = None,
    ):
        collision_seed, ordinary_seed = np.random.SeedSequence(seed).spawn(2)
        self._init_ordinary(ordinary_scenarios, ordinary_seed, seed,
                            ordinary_offline_fast_fraction)
        self.hard_neighbor_fraction = None
        self.hard_neighbor_fraction_numerator = 0
        self.hard_neighbor_fraction_denominator = 1
        self.hard_neighbor_positions: frozenset[int] = frozenset()
        self.collision_source_cursor = 0
        self.base_collision: RoleScenarioQueue | None = None
        self.hard_neighbor: RoleScenarioQueue | None = None

        if hard_neighbor_fraction is None:
            self.collision = RoleScenarioQueue(collision_scenarios, collision_seed)
            return

        fraction = Fraction(str(float(hard_neighbor_fraction))).limit_denominator(1_000)
        if (
            not np.isfinite(hard_neighbor_fraction)
            or not 0 < fraction < 1
            or abs(float(fraction) - float(hard_neighbor_fraction)) > 1e-12
        ):
            raise ValueError("hard_neighbor_fraction must be a finite rational value in (0, 1)")
        base_scenarios = tuple(scenario for scenario in collision_scenarios if scenario.pool == "collision")
        hard_scenarios = tuple(scenario for scenario in collision_scenarios if scenario.pool == "hard_neighbor")
        unexpected_pools = sorted(
            {scenario.pool for scenario in collision_scenarios}
            - {"collision", "hard_neighbor"}
        )
        if unexpected_pools:
            raise ValueError(f"Unexpected collision scenario pools: {unexpected_pools}")
        if not base_scenarios or not hard_scenarios:
            raise ValueError(
                "Stratified hard-neighbor sampling requires non-empty base and hard-neighbor pools"
            )

        hard_seed = np.random.SeedSequence([int(seed), 0x48415244])
        self.collision = None
        self.base_collision = RoleScenarioQueue(base_scenarios, collision_seed)
        self.hard_neighbor = RoleScenarioQueue(hard_scenarios, hard_seed)
        self.hard_neighbor_fraction = float(fraction)
        self.hard_neighbor_fraction_numerator = fraction.numerator
        self.hard_neighbor_fraction_denominator = fraction.denominator
        self.hard_neighbor_positions = frozenset(
            ((2 * index + 1) * fraction.denominator) // (2 * fraction.numerator)
            for index in range(fraction.numerator)
        )
        if len(self.hard_neighbor_positions) != fraction.numerator:
            raise RuntimeError("Hard-neighbor sampling positions are not unique")

    @staticmethod
    def is_same_line(scenario: ScenarioSpec) -> bool:
        return scenario.opp_raceline == EGO_RACELINE

    @staticmethod
    def is_offline_fast(scenario: ScenarioSpec) -> bool:
        """Opponent on a different raceline and moving fast.

        This is the regime that carries CT-v2's whole regression: on the three
        held-out maps it costs 20-28 collisions against B's 14, and the off-line
        commanded-speed reduction it pays (-0.16 m/s) buys no surface margin
        there (paired delta +0.005 m on commonly-safe episodes).  Membership is a
        pure function of the scenario grid, so the split is reproducible and
        depends on no model's outcomes.
        """
        return scenario.opp_raceline != EGO_RACELINE and scenario.opp_speedscale >= 0.7

    def _init_ordinary(
        self,
        ordinary_scenarios: Sequence[ScenarioSpec],
        ordinary_seed: np.random.SeedSequence,
        seed: int,
        fraction: float | None,
    ) -> None:
        """Three-way ordinary split that holds same-line weight at its natural share.

        An earlier two-way version (off-line-fast versus everything else) was
        rejected after 10 updates: giving the off-line-fast queue 2/3 halved the
        same-line share from 33.3% to 16.7%, and since the corridor gate only
        fires on same-line following that removed most of the arm's gate
        exposure.  Its collision-role return (-0.713) fell back to B's (-0.759)
        instead of tracking CT-v2's (-0.401), even though the collision role's
        pool and sampling were untouched -- the same-line mechanism simply never
        formed.  gap=1.0 had already shown the mechanism is destroyed by
        reducing same-line exposure, so any reweighting must preserve it.

        Here ``fraction`` is the off-line-fast share; same-line keeps exactly the
        share it has in the uniform panel, and off-line-slow absorbs the
        remainder.  Every scenario stays reachable, so only weight changes.
        """
        self.ordinary_offline_fast_fraction = None
        self.ordinary_offline_fast_numerator = 0
        self.ordinary_offline_fast_denominator = 1
        self.ordinary_position_queue: tuple[str, ...] = ()
        self.ordinary_source_cursor = 0
        self.ordinary_queues: dict[str, RoleScenarioQueue] = {}

        if fraction is None:
            self.ordinary = RoleScenarioQueue(ordinary_scenarios, ordinary_seed)
            return

        groups = {"same_line": [], "offline_fast": [], "offline_slow": []}
        for scenario in ordinary_scenarios:
            if self.is_same_line(scenario):
                groups["same_line"].append(scenario)
            elif self.is_offline_fast(scenario):
                groups["offline_fast"].append(scenario)
            else:
                groups["offline_slow"].append(scenario)
        if not all(groups.values()):
            raise ValueError(
                "Stratified ordinary sampling requires non-empty same-line, "
                "off-line-fast and off-line-slow groups"
            )

        total = len(ordinary_scenarios)
        natural_same_line = Fraction(len(groups["same_line"]), total)
        fast_share = Fraction(str(float(fraction))).limit_denominator(1_000)
        if (
            not np.isfinite(fraction)
            or not 0 < fast_share < 1
            or abs(float(fast_share) - float(fraction)) > 1e-12
        ):
            raise ValueError(
                "ordinary_offline_fast_fraction must be a finite rational value in (0, 1)"
            )
        slow_share = 1 - natural_same_line - fast_share
        if slow_share <= 0:
            raise ValueError(
                "ordinary_offline_fast_fraction must leave positive weight for the "
                f"off-line-slow group (maximum is {float(1 - natural_same_line):.4f})"
            )

        shares = {
            "same_line": natural_same_line,
            "offline_fast": fast_share,
            "offline_slow": slow_share,
        }
        denominator = 1
        for share in shares.values():
            denominator = denominator * share.denominator // gcd(denominator, share.denominator)
        counts = {name: int(share * denominator) for name, share in shares.items()}
        if sum(counts.values()) != denominator:
            raise RuntimeError("Ordinary stratification shares do not tile the cycle")

        # Spread each group's slots as evenly as possible over the cycle, using
        # the same midpoint rule the hard-neighbor stratifier uses.
        assignment: dict[int, str] = {}
        for name in ("offline_fast", "same_line", "offline_slow"):
            count = counts[name]
            for index in range(count):
                position = ((2 * index + 1) * denominator) // (2 * count)
                while position in assignment:
                    position = (position + 1) % denominator
                assignment[position] = name
        if len(assignment) != denominator:
            raise RuntimeError("Ordinary stratification cycle is not fully assigned")

        seeds = {
            "same_line": ordinary_seed,
            "offline_fast": np.random.SeedSequence([int(seed), 0x4F464653]),
            "offline_slow": np.random.SeedSequence([int(seed), 0x4F464C57]),
        }
        self.ordinary = None
        self.ordinary_queues = {
            name: RoleScenarioQueue(tuple(scenarios), seeds[name])
            for name, scenarios in groups.items()
        }
        self.ordinary_position_queue = tuple(
            assignment[position] for position in range(denominator)
        )
        self.ordinary_offline_fast_fraction = float(fast_share)
        self.ordinary_offline_fast_numerator = fast_share.numerator
        self.ordinary_offline_fast_denominator = fast_share.denominator

    def _next_ordinary_scenario(self) -> ScenarioSpec:
        if self.ordinary_offline_fast_fraction is None:
            return self.ordinary.next()
        name = self.ordinary_position_queue[
            self.ordinary_source_cursor % len(self.ordinary_position_queue)
        ]
        self.ordinary_source_cursor += 1
        return self.ordinary_queues[name].next()

    def _next_collision_scenario(self) -> ScenarioSpec:
        if self.hard_neighbor_fraction is None:
            return self.collision.next()
        cycle_position = (
            self.collision_source_cursor % self.hard_neighbor_fraction_denominator
        )
        use_hard_neighbor = cycle_position in self.hard_neighbor_positions
        self.collision_source_cursor += 1
        queue = self.hard_neighbor if use_hard_neighbor else self.base_collision
        return queue.next()

    def next(self, rank: int) -> EpisodeResetSpec:
        if rank % 2 == 0:
            return self._next_collision_scenario().to_reset_spec("collision")
        return self._next_ordinary_scenario().to_reset_spec("ordinary")

    def _ordinary_state_dict(self) -> dict[str, Any]:
        if self.ordinary_offline_fast_fraction is None:
            return {"ordinary": self.ordinary.state_dict()}
        return {
            "ordinary_sampling_mode": "stratified_offline_fast_v2",
            "ordinary_queues": {
                name: queue.state_dict()
                for name, queue in self.ordinary_queues.items()
            },
            "ordinary_position_queue": list(self.ordinary_position_queue),
            "ordinary_source_cursor": self.ordinary_source_cursor,
            "ordinary_offline_fast_numerator": self.ordinary_offline_fast_numerator,
            "ordinary_offline_fast_denominator": self.ordinary_offline_fast_denominator,
        }

    def _load_ordinary_state_dict(self, state: dict[str, Any]) -> None:
        if self.ordinary_offline_fast_fraction is None:
            self.ordinary.load_state_dict(state["ordinary"])
            return
        if state.get("ordinary_sampling_mode") != "stratified_offline_fast_v2":
            raise ValueError("Scenario scheduler ordinary sampling mode does not match")
        if (
            int(state["ordinary_offline_fast_numerator"])
            != self.ordinary_offline_fast_numerator
            or int(state["ordinary_offline_fast_denominator"])
            != self.ordinary_offline_fast_denominator
        ):
            raise ValueError("Scenario scheduler ordinary off-line-fast fraction does not match")
        if tuple(state["ordinary_position_queue"]) != self.ordinary_position_queue:
            raise ValueError("Scenario scheduler ordinary cycle assignment does not match")
        cursor = int(state["ordinary_source_cursor"])
        if cursor < 0:
            raise ValueError("Scenario scheduler ordinary source cursor is invalid")
        for name, queue in self.ordinary_queues.items():
            queue.load_state_dict(state["ordinary_queues"][name])
        self.ordinary_source_cursor = cursor

    def state_dict(self) -> dict[str, Any]:
        if self.hard_neighbor_fraction is None:
            return {
                "collision": self.collision.state_dict(),
                **self._ordinary_state_dict(),
            }
        return {
            "sampling_mode": "stratified_hard_neighbor",
            "base_collision": self.base_collision.state_dict(),
            "hard_neighbor": self.hard_neighbor.state_dict(),
            "collision_source_cursor": self.collision_source_cursor,
            "hard_neighbor_fraction_numerator": self.hard_neighbor_fraction_numerator,
            "hard_neighbor_fraction_denominator": self.hard_neighbor_fraction_denominator,
            **self._ordinary_state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.hard_neighbor_fraction is None:
            self.collision.load_state_dict(state["collision"])
            self._load_ordinary_state_dict(state)
            return
        if state.get("sampling_mode") != "stratified_hard_neighbor":
            raise ValueError("Scenario scheduler sampling mode does not match")
        if (
            int(state["hard_neighbor_fraction_numerator"])
            != self.hard_neighbor_fraction_numerator
            or int(state["hard_neighbor_fraction_denominator"])
            != self.hard_neighbor_fraction_denominator
        ):
            raise ValueError("Scenario scheduler hard-neighbor fraction does not match")
        collision_source_cursor = int(state["collision_source_cursor"])
        if collision_source_cursor < 0:
            raise ValueError("Scenario scheduler collision source cursor is invalid")
        self.base_collision.load_state_dict(state["base_collision"])
        self.hard_neighbor.load_state_dict(state["hard_neighbor"])
        self._load_ordinary_state_dict(state)
        self.collision_source_cursor = collision_source_cursor
