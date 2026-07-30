"""PPO scenario generation and deterministic role queues."""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from math import gcd
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Sequence
import numpy as np
import yaml

from utils import (
    find_corresponding_waypoint,
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
_ORDINARY_OFFLINE_FAST_FRACTION = PPO_CONFIG["ordinary_offline_fast_fraction"]
ORDINARY_OFFLINE_FAST_FRACTION = (
    None
    if _ORDINARY_OFFLINE_FAST_FRACTION is None
    else float(_ORDINARY_OFFLINE_FAST_FRACTION)
)
COLLISION_INTERVAL_INDICES = tuple(int(value) for value in PPO_CONFIG["collision_interval_indices"])
COLLISION_SPEED_SCALES = tuple(float(value) for value in PPO_CONFIG["collision_speed_scales"])
COLLISION_STARTPOINT_COUNT = int(PPO_CONFIG["collision_startpoint_count"])
COLLISION_STARTPOINT_MIN_DISTANCE = float(PPO_CONFIG["collision_startpoint_min_distance"])
SIM_DURATION = float(PPO_CONFIG["episode_horizon"])
TIMESTEP = float(PPO_CONFIG["simulator_timestep"])
COLLISION_CLASSIFICATION_SCHEMA = 1
_COLLISION_ENV = None
_COLLISION_ACTOR = None


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


def ordinary_startpoints(map_name: str) -> tuple[int, ...]:
    """Return the fixed production ordinary startpoints."""

    return generate_separated_startpoints(
        map_name,
        ORDINARY_STARTPOINT_COUNT,
        ORDINARY_STARTPOINT_MIN_DISTANCE,
    )


def collision_candidate_startpoints(map_name: str) -> tuple[int, ...]:
    return generate_separated_startpoints(map_name, COLLISION_STARTPOINT_COUNT, COLLISION_STARTPOINT_MIN_DISTANCE)


def ordinary_scenarios(map_name: str) -> tuple[ScenarioSpec, ...]:
    ego_waypoints = load_raceline_waypoints(map_name, f"{EGO_RACELINE}.csv")
    opponent_waypoints = {raceline: load_raceline_waypoints(map_name, f"{raceline}.csv") for raceline in OPPONENT_RACELINES}
    scenarios = []
    for ordinal, ego_idx in enumerate(ordinary_startpoints(map_name)):
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
    ):
        collision_seed, ordinary_seed = np.random.SeedSequence(seed).spawn(2)
        self._init_ordinary(
            ordinary_scenarios,
            ordinary_seed,
            seed,
            ORDINARY_OFFLINE_FAST_FRACTION,
        )
        self.collision = RoleScenarioQueue(collision_scenarios, collision_seed)

    @staticmethod
    def is_same_line(scenario: ScenarioSpec) -> bool:
        return scenario.opp_raceline == EGO_RACELINE

    @staticmethod
    def is_offline_fast(scenario: ScenarioSpec) -> bool:
        """Opponent on a different raceline and moving fast.

        This is the regime that carries the front-corridor temporal arm's whole
        regression: on the three held-out maps it costs 20-28 collisions against
        the production baseline's 14, and the off-line
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
        instead of tracking the front-corridor temporal arm's (-0.401), even though the collision role's
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
        return {
            "collision": self.collision.state_dict(),
            **self._ordinary_state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.collision.load_state_dict(state["collision"])
        self._load_ordinary_state_dict(state)


def collision_classification_config(args, candidate_count: int) -> dict:
    return {
        "classification_schema": COLLISION_CLASSIFICATION_SCHEMA,
        "pretrained_model_path": str(Path(args.pretrained_model_path).expanduser().resolve()),
        "hidden_scale": int(args.hidden_scale),
        "map_name": str(args.map_name),
        "ego_raceline": EGO_RACELINE,
        "opponent_racelines": list(OPPONENT_RACELINES),
        "collision_startpoint_count": COLLISION_STARTPOINT_COUNT,
        "collision_interval_indices": list(COLLISION_INTERVAL_INDICES),
        "collision_speed_scales": list(COLLISION_SPEED_SCALES),
        "collision_startpoint_min_distance": COLLISION_STARTPOINT_MIN_DISTANCE,
        "simulator_timestep": TIMESTEP,
        "episode_horizon": SIM_DURATION,
        "candidate_count": candidate_count,
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def validate_collision_cache_identity(cached_config: dict, current_config: dict, *, candidate_count: int) -> None:
    if json.dumps(cached_config, sort_keys=True) != json.dumps(current_config, sort_keys=True):
        raise RuntimeError(
            f"Collision cache configuration does not match the current {candidate_count} candidates; "
            "specify a new empty --collision_cache_dir"
        )


def collision_cache_exists(cache_dir: Path) -> bool:
    required_paths = (
        cache_dir / "classification_config.json",
        cache_dir / "candidate_outcomes.jsonl",
        cache_dir / "collision_scenarios.json",
        cache_dir / "classification_summary.json",
    )
    existing_count = sum(path.exists() for path in required_paths)
    if existing_count not in (0, len(required_paths)):
        raise RuntimeError("Collision classification cache is incomplete; specify a new empty --collision_cache_dir")
    return existing_count == len(required_paths)


def write_collision_cache(
    cache_dir: Path,
    config: dict,
    outcomes: list[dict],
    collision_scenarios: tuple[ScenarioSpec, ...],
    summary: dict,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_json(cache_dir / "classification_config.json", config)
    with (cache_dir / "candidate_outcomes.jsonl").open("w", encoding="utf-8") as file:
        for outcome in outcomes:
            file.write(json.dumps(outcome) + "\n")
    _write_json(cache_dir / "collision_scenarios.json", [asdict(scenario) for scenario in collision_scenarios])
    _write_json(cache_dir / "classification_summary.json", summary)


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


def _validate_classification_summary(path: Path, outcomes: list[dict], candidate_count: int) -> dict:
    with path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    collision_count = sum(outcome["outcome"] == "ego_collision" for outcome in outcomes)
    invalid_count = sum(outcome["outcome"] == "invalid" for outcome in outcomes)
    expected_counts = {
        "candidate_count": candidate_count,
        "collision_count": collision_count,
        "other_count": candidate_count - collision_count - invalid_count,
        "invalid_count": invalid_count,
    }
    expected_keys = set(expected_counts) | {"env_workers", "wall_seconds", "scenarios_per_second"}
    if (
        not isinstance(summary, dict)
        or set(summary) != expected_keys
        or any(type(summary[name]) is not int or summary[name] != value for name, value in expected_counts.items())
    ):
        raise RuntimeError(f"Collision cache summary does not match {candidate_count} candidate outcomes")
    if type(summary["env_workers"]) is not int or summary["env_workers"] <= 0:
        raise RuntimeError("Collision cache summary has invalid env_workers")
    for name in ("wall_seconds", "scenarios_per_second"):
        value = summary[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise RuntimeError(f"Collision cache summary has invalid {name}")
    return summary


def load_collision_cache_artifacts(
    cache_dir: Path,
    current_config: dict,
    candidates: tuple[ScenarioSpec, ...],
) -> tuple[tuple[ScenarioSpec, ...], list[dict], dict]:
    with (cache_dir / "classification_config.json").open("r", encoding="utf-8") as file:
        cached_config = json.load(file)
    candidate_count = len(candidates)
    validate_collision_cache_identity(cached_config, current_config, candidate_count=candidate_count)
    outcomes = _load_candidate_outcomes(cache_dir / "candidate_outcomes.jsonl", candidates)
    collision_scenarios = _load_collision_scenarios(cache_dir / "collision_scenarios.json", candidates, outcomes)
    summary = _validate_classification_summary(cache_dir / "classification_summary.json", outcomes, candidate_count)
    return collision_scenarios, outcomes, summary


def load_collision_cache(
    cache_dir: Path,
    current_config: dict,
    candidates: tuple[ScenarioSpec, ...],
) -> tuple[ScenarioSpec, ...]:
    collision_scenarios, _outcomes, _summary = load_collision_cache_artifacts(cache_dir, current_config, candidates)
    return collision_scenarios


def _collision_worker_init(pretrained_model_path: str, hidden_scale: int, map_name: str) -> None:
    import torch
    from model import End2Race
    from ppo.env import limit_worker_threads, make_environment

    global _COLLISION_ENV, _COLLISION_ACTOR
    limit_worker_threads()
    _COLLISION_ENV = make_environment(0, map_name)()
    _COLLISION_ACTOR = End2Race(mask_prob=0.0, hidden_scale=hidden_scale)
    _COLLISION_ACTOR.load_state_dict(torch.load(pretrained_model_path, map_location="cpu", weights_only=True), strict=True)
    _COLLISION_ACTOR.eval()


def _classify_collision_candidate(task: tuple[int, ScenarioSpec]) -> tuple[int, str]:
    import torch
    from ppo.env import EXTERNAL_RESET_OPTION
    from ppo.policy import END2RACE_LIDAR_SIZE, STEERING_BOUND

    index, scenario = task
    if _COLLISION_ENV is None or _COLLISION_ACTOR is None:
        raise RuntimeError("Collision classification worker is not initialized")
    try:
        observation, _info = _COLLISION_ENV.reset(options={EXTERNAL_RESET_OPTION: scenario.to_reset_spec("collision")})
        raw = _COLLISION_ENV._raw_observation
        finite = np.isfinite(observation).all() and all(
            np.isfinite(np.asarray(value)).all()
            for value in raw.values()
            if isinstance(value, (list, tuple, np.ndarray))
        )
        if not finite or np.asarray(raw["collisions"], dtype=bool).any():
            return index, "invalid"
        hidden = None
        while True:
            actor_observation = torch.as_tensor(observation, dtype=torch.float32)
            with torch.no_grad():
                actions, hidden = _COLLISION_ACTOR(
                    actor_observation[:END2RACE_LIDAR_SIZE].reshape(1, 1, -1),
                    actor_observation[END2RACE_LIDAR_SIZE:].reshape(1, 1, 1),
                    hidden,
                )
            action = actions[0, -1].numpy().copy()
            action[0] = np.clip(action[0], -STEERING_BOUND, STEERING_BOUND)
            if not np.isfinite(action).all():
                raise RuntimeError("actor produced a non-finite action")
            observation, _reward, terminated, truncated, info = _COLLISION_ENV.step(action)
            if terminated or truncated:
                return index, "ego_collision" if info["ego_collision"] else "other"
    except Exception as error:
        raise RuntimeError(f"Collision classification failed for {scenario.scenario_id}") from error


def classify_collision_scenarios(
    pretrained_model_path: str | Path,
    hidden_scale: int,
    map_name: str,
    n_envs: int,
    candidates: tuple[ScenarioSpec, ...],
    start_method: str,
) -> tuple[tuple[ScenarioSpec, ...], list[dict], dict]:
    candidate_count = len(candidates)
    context = mp.get_context(start_method)
    collisions = []
    outcomes = []
    collision_count = 0
    invalid_count = 0
    started_at = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=n_envs,
        mp_context=context,
        initializer=_collision_worker_init,
        initargs=(str(Path(pretrained_model_path).expanduser().resolve()), hidden_scale, map_name),
    ) as executor:
        for completed, (index, outcome) in enumerate(
            executor.map(_classify_collision_candidate, enumerate(candidates), chunksize=4),
            start=1,
        ):
            if index != completed - 1 or outcome not in {"ego_collision", "other", "invalid"}:
                raise RuntimeError(f"Invalid classification result at candidate {completed - 1}/{candidate_count}")
            outcomes.append({"candidate_index": index, "scenario_id": candidates[index].scenario_id, "outcome": outcome})
            if outcome == "ego_collision":
                collisions.append(candidates[index])
                collision_count += 1
            elif outcome == "invalid":
                invalid_count += 1
            if completed % 100 == 0 or completed == candidate_count:
                print(
                    f"Collision classification: {completed}/{candidate_count}, collision={collision_count}, invalid={invalid_count}",
                    flush=True,
                )
    if not collisions:
        raise RuntimeError(f"The pretrained model produced no ego-collision scenarios from {candidate_count} candidates")
    wall_seconds = time.perf_counter() - started_at
    summary = {
        "candidate_count": candidate_count,
        "collision_count": collision_count,
        "other_count": candidate_count - collision_count - invalid_count,
        "invalid_count": invalid_count,
        "env_workers": n_envs,
        "wall_seconds": wall_seconds,
        "scenarios_per_second": candidate_count / wall_seconds,
    }
    return tuple(collisions), outcomes, summary


def resolve_collision_scenarios(
    args,
    candidates: tuple[ScenarioSpec, ...],
    start_method: str,
) -> tuple[tuple[ScenarioSpec, ...], bool, bool]:
    candidate_count = len(candidates)
    if candidate_count == 0:
        raise RuntimeError("Collision candidate set is empty")
    cache_dir = Path(args.collision_cache_dir).expanduser().resolve()
    current_config = collision_classification_config(args, candidate_count)
    if collision_cache_exists(cache_dir):
        collision_scenarios = load_collision_cache(cache_dir, current_config, candidates)
        print(
            f"Collision cache hit: loaded {len(collision_scenarios)} collision scenarios from {candidate_count} candidates",
            flush=True,
        )
        return collision_scenarios, True, False
    print(f"Collision cache miss: classifying {candidate_count} candidates", flush=True)
    collision_scenarios, outcomes, summary = classify_collision_scenarios(
        args.pretrained_model_path,
        args.hidden_scale,
        args.map_name,
        args.n_envs,
        candidates,
        start_method,
    )
    write_collision_cache(cache_dir, current_config, outcomes, collision_scenarios, summary)
    return collision_scenarios, False, False
