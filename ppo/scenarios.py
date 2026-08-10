"""PPO scenario generation and deterministic role queues."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from math import gcd
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from latticeplanner.utils import load_config
from utils import (
    find_corresponding_waypoint,
    load_positions_and_speeds_from_params,
    load_raceline_waypoints,
)

CONFIG = load_config("ppo/ppo_config.yaml")

COLLISION_CLASSIFICATION_SCHEMA = 1


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
    ego_raceline: str = CONFIG.ego_raceline
    sim_duration: float = CONFIG.episode_horizon
    timestep: float = CONFIG.simulator_timestep
    integrator: str = "RK4"

    def to_reset_spec(self, env_role: str) -> EpisodeResetSpec:
        scenario = asdict(self)
        scenario["opponent_speed_scale"] = self.opp_speedscale
        scenario["sampler_branch"] = env_role
        scenario["env_role"] = env_role
        poses, initial_speeds = load_positions_and_speeds_from_params(scenario, self.map_name)
        return EpisodeResetSpec(np.asarray(poses, dtype=np.float64), float(initial_speeds[0] * 0.9), scenario)


def _raceline_data(map_name: str) -> np.ndarray:
    path = Path("f1tenth_racetracks") / map_name / f"{CONFIG.ego_raceline}.csv"
    data = np.loadtxt(path, delimiter=";", comments="#", dtype=np.float64)
    if np.linalg.norm(data[-1, 1:3] - data[0, 1:3]) > 1e-9:
        raise ValueError(f"{CONFIG.ego_raceline}.csv must contain the duplicated closing endpoint")
    return data


def evaluation_startpoints(map_name: str) -> tuple[int, ...]:
    data = _raceline_data(map_name)
    indices = np.arange(CONFIG.evaluation_startpoint_count) * (len(data) - 1) // (CONFIG.evaluation_startpoint_count - 1)
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


def ordinary_startpoints(map_name: str) -> tuple[int, ...]:
    """Return the fixed production ordinary startpoints."""

    return generate_separated_startpoints(
        map_name,
        CONFIG.ordinary_startpoint_count,
        CONFIG.ordinary_startpoint_min_distance,
    )


def collision_candidate_startpoints(map_name: str) -> tuple[int, ...]:
    return generate_separated_startpoints(map_name, CONFIG.collision_startpoint_count, CONFIG.collision_startpoint_min_distance)


def ordinary_scenarios(map_name: str) -> tuple[ScenarioSpec, ...]:
    ego_waypoints = load_raceline_waypoints(map_name, f"{CONFIG.ego_raceline}.csv")
    opponent_waypoints = {raceline: load_raceline_waypoints(map_name, f"{raceline}.csv") for raceline in CONFIG.opponent_racelines}
    scenarios = []
    for ordinal, ego_idx in enumerate(ordinary_startpoints(map_name)):
        ego_waypoint = ego_waypoints[ego_idx]
        for opp_raceline in CONFIG.opponent_racelines:
            mapped_index = ego_idx if opp_raceline == CONFIG.ego_raceline else int(find_corresponding_waypoint(ego_waypoint, opponent_waypoints[opp_raceline]))
            opp_idx = (mapped_index + CONFIG.ordinary_interval_index) % len(opponent_waypoints[opp_raceline])
            for speed_scale in CONFIG.ordinary_speed_scales:
                scenario_id = f"ordinary-sp{ordinal:02d}-ego{ego_idx:04d}-{opp_raceline}-v{int(100 * speed_scale):03d}"
                scenarios.append(ScenarioSpec(scenario_id, "ordinary", ordinal, ego_idx, opp_idx, opp_raceline, speed_scale, CONFIG.ordinary_interval_index, map_name))
    expected_count = CONFIG.ordinary_startpoint_count * len(CONFIG.opponent_racelines) * len(CONFIG.ordinary_speed_scales)
    if len(scenarios) != expected_count or len({scenario.scenario_id for scenario in scenarios}) != expected_count:
        raise RuntimeError(f"Ordinary scenario panel must contain {expected_count:,} unique scenarios")
    return tuple(scenarios)


def expanded_scenarios(map_name: str) -> tuple[ScenarioSpec, ...]:
    ego_waypoints = load_raceline_waypoints(map_name, f"{CONFIG.ego_raceline}.csv")
    opponent_waypoints = {raceline: load_raceline_waypoints(map_name, f"{raceline}.csv") for raceline in CONFIG.opponent_racelines}
    scenarios = []
    for ordinal, ego_idx in enumerate(collision_candidate_startpoints(map_name)):
        ego_waypoint = ego_waypoints[ego_idx]
        for opp_raceline in CONFIG.opponent_racelines:
            mapped_index = ego_idx if opp_raceline == CONFIG.ego_raceline else int(find_corresponding_waypoint(ego_waypoint, opponent_waypoints[opp_raceline]))
            for interval_idx in CONFIG.collision_interval_indices:
                opp_idx = (mapped_index + interval_idx) % (len(opponent_waypoints[opp_raceline]) - 1)
                for speed_scale in CONFIG.collision_speed_scales:
                    scenario_id = f"collision-sp{ordinal:03d}-ego{ego_idx:04d}-{opp_raceline}-i{interval_idx:02d}-v{round(100 * speed_scale):03d}"
                    scenarios.append(ScenarioSpec(scenario_id, "collision", ordinal, ego_idx, opp_idx, opp_raceline, speed_scale, interval_idx, map_name))
    expected_count = CONFIG.collision_startpoint_count * len(CONFIG.opponent_racelines) * len(CONFIG.collision_interval_indices) * len(CONFIG.collision_speed_scales)
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

class ScenarioScheduler:

    def __init__(
        self,
        seed: int,
        collision_scenarios: Sequence[ScenarioSpec],
        ordinary_scenarios: Sequence[ScenarioSpec],
    ):
        collision_seed, ordinary_seed = np.random.SeedSequence(seed).spawn(2)
        self._init_ordinary(
            ordinary_scenarios,
            ordinary_seed,
            seed,
            CONFIG.ordinary_offline_fast_fraction,
        )
        self.collision = RoleScenarioQueue(collision_scenarios, collision_seed)

    @staticmethod
    def is_same_line(scenario: ScenarioSpec) -> bool:
        return scenario.opp_raceline == CONFIG.ego_raceline

    @staticmethod
    def is_offline_fast(scenario: ScenarioSpec) -> bool:
        return scenario.opp_raceline != CONFIG.ego_raceline and scenario.opp_speedscale >= CONFIG.ordinary_offline_fast_min_speed_scale

    def _init_ordinary(
        self,
        ordinary_scenarios: Sequence[ScenarioSpec],
        ordinary_seed: np.random.SeedSequence,
        seed: int,
        fraction: float | None,
    ) -> None:
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

        # Spread each group's slots as evenly as possible over the cycle.
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
        return self.collision.next()

    def next(self, rank: int) -> EpisodeResetSpec:
        if rank % 2 == 0:
            return self._next_collision_scenario().to_reset_spec("collision")
        return self._next_ordinary_scenario().to_reset_spec("ordinary")

def collision_classification_config(args, candidate_count: int) -> dict:
    return {
        "classification_schema": COLLISION_CLASSIFICATION_SCHEMA,
        "pretrained_model_path": str(Path(args.pretrained_model_path).expanduser().resolve()),
        "hidden_scale": int(args.hidden_scale),
        "map_name": str(args.map_name),
        "ego_raceline": CONFIG.ego_raceline,
        "opponent_racelines": list(CONFIG.opponent_racelines),
        "collision_startpoint_count": CONFIG.collision_startpoint_count,
        "collision_interval_indices": list(CONFIG.collision_interval_indices),
        "collision_speed_scales": list(CONFIG.collision_speed_scales),
        "collision_startpoint_min_distance": CONFIG.collision_startpoint_min_distance,
        "simulator_timestep": CONFIG.simulator_timestep,
        "episode_horizon": CONFIG.episode_horizon,
        "candidate_count": candidate_count,
    }


def validate_collision_cache_identity(cached_config: dict, current_config: dict, *, candidate_count: int) -> None:
    if json.dumps(cached_config, sort_keys=True) != json.dumps(current_config, sort_keys=True):
        raise RuntimeError(
            f"Collision cache configuration does not match the current {candidate_count} candidates; "
            "build a matching cache before training"
        )


def _load_candidate_outcomes(path: Path, candidates: tuple[ScenarioSpec, ...]) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        outcomes = [json.loads(line) for line in file]
    candidate_count = len(candidates)
    if len(outcomes) != candidate_count:
        raise RuntimeError(f"Collision cache has {len(outcomes)} outcomes for {candidate_count} candidates")
    expected_keys = {"candidate_index", "scenario_id", "outcome"}
    for candidate_index, (outcome, candidate) in enumerate(zip(outcomes, candidates)):
        if set(outcome) != expected_keys or type(outcome["candidate_index"]) is not int or outcome["candidate_index"] != candidate_index:
            raise RuntimeError(f"Collision cache candidate_index must be 0 through {candidate_count - 1}")
        if outcome["scenario_id"] != candidate.scenario_id:
            raise RuntimeError(f"Collision cache scenario_id mismatch at candidate {candidate_index}/{candidate_count}")
        if outcome["outcome"] not in {"ego_collision", "other", "invalid"}:
            raise RuntimeError(f"Collision cache has an invalid outcome at candidate {candidate_index}/{candidate_count}")
    return outcomes


def _load_collision_scenarios(
    path: Path,
    candidates: tuple[ScenarioSpec, ...],
    outcomes: list[dict],
) -> tuple[ScenarioSpec, ...]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise RuntimeError("Collision cache must contain at least one collision ScenarioSpec")
    candidate_by_id = {candidate.scenario_id: candidate for candidate in candidates}
    expected_ids = [outcome["scenario_id"] for outcome in outcomes if outcome["outcome"] == "ego_collision"]
    actual_ids = [record.get("scenario_id") for record in records if isinstance(record, dict)]
    if len(actual_ids) != len(records) or len(set(actual_ids)) != len(actual_ids):
        raise RuntimeError("Collision cache collision scenario IDs must be unique")
    if actual_ids != expected_ids:
        raise RuntimeError("Collision cache collision scenarios do not match ego_collision outcomes")
    collision_scenarios = []
    expected_fields = set(asdict(candidates[0]))
    for record in records:
        if set(record) != expected_fields:
            raise RuntimeError(f"Collision cache ScenarioSpec fields are invalid for {record['scenario_id']}")
        scenario = ScenarioSpec(**record)
        current_candidate = candidate_by_id.get(scenario.scenario_id)
        if current_candidate is None:
            raise RuntimeError(f"Collision cache ScenarioSpec does not match current candidate {scenario.scenario_id}")
        current_record = asdict(current_candidate)
        if any(
            type(record[name]) is not type(current_record[name]) or record[name] != current_record[name]
            for name in expected_fields
        ):
            raise RuntimeError(f"Collision cache ScenarioSpec does not match current candidate {scenario.scenario_id}")
        collision_scenarios.append(scenario)
    return tuple(collision_scenarios)


def load_collision_cache(
    cache_dir: Path,
    current_config: dict,
    candidates: tuple[ScenarioSpec, ...],
) -> tuple[ScenarioSpec, ...]:
    with (cache_dir / "classification_config.json").open("r", encoding="utf-8") as file:
        cached_config = json.load(file)
    candidate_count = len(candidates)
    validate_collision_cache_identity(cached_config, current_config, candidate_count=candidate_count)
    outcomes = _load_candidate_outcomes(cache_dir / "candidate_outcomes.jsonl", candidates)
    return _load_collision_scenarios(cache_dir / "collision_scenarios.json", candidates, outcomes)


def resolve_collision_scenarios(
    args,
    candidates: tuple[ScenarioSpec, ...],
) -> tuple[ScenarioSpec, ...]:
    candidate_count = len(candidates)
    cache_dir = Path(args.collision_cache_dir).expanduser().resolve()
    current_config = collision_classification_config(args, candidate_count)
    collision_scenarios = load_collision_cache(cache_dir, current_config, candidates)
    print(
        f"Loaded {len(collision_scenarios)} collision scenarios from {candidate_count} cached candidates",
        flush=True,
    )
    return collision_scenarios
