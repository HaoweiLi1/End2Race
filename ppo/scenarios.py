"""PPO scenario generation and deterministic role queues."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import yaml

from utils import find_corresponding_waypoint, load_positions_and_speeds_from_params, load_raceline_waypoints

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


def ordinary_startpoints(map_name: str) -> tuple[int, ...]:
    return generate_separated_startpoints(map_name, ORDINARY_STARTPOINT_COUNT, ORDINARY_STARTPOINT_MIN_DISTANCE)


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


def collision_classification_config(args, candidate_count: int) -> dict:
    return {
        "classification_schema": 1,
        "pretrained_model_path": str(Path(args.pretrained_model_path).expanduser().resolve()),
        "hidden_scale": int(args.hidden_scale),
        "map_name": str(args.map_name),
        "ego_raceline": str(PPO_CONFIG["ego_raceline"]),
        "opponent_racelines": [str(value) for value in PPO_CONFIG["opponent_racelines"]],
        "collision_startpoint_count": int(PPO_CONFIG["collision_startpoint_count"]),
        "collision_interval_indices": [int(value) for value in PPO_CONFIG["collision_interval_indices"]],
        "collision_speed_scales": [float(value) for value in PPO_CONFIG["collision_speed_scales"]],
        "collision_startpoint_min_distance": float(PPO_CONFIG["collision_startpoint_min_distance"]),
        "simulator_timestep": float(PPO_CONFIG["simulator_timestep"]),
        "episode_horizon": float(PPO_CONFIG["episode_horizon"]),
        "candidate_count": candidate_count,
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def collision_cache_exists(cache_dir: Path) -> bool:
    required_paths = (
        cache_dir / "classification_config.json",
        cache_dir / "candidate_outcomes.jsonl",
        cache_dir / "collision_scenarios.json",
        cache_dir / "classification_summary.json",
    )
    existing_count = sum(path.exists() for path in required_paths)
    if existing_count not in (0, len(required_paths)):
        raise RuntimeError("Collision classification cache is incomplete; use --reclassify_collisions")
    return existing_count == len(required_paths)


def write_collision_cache(cache_dir: Path, config: dict, outcomes: list[dict], collision_scenarios: tuple[ScenarioSpec, ...], summary: dict) -> None:
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


def _load_collision_scenarios(path: Path, candidates: tuple[ScenarioSpec, ...], outcomes: list[dict]) -> tuple[ScenarioSpec, ...]:
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
        if any(type(record[name]) is not type(current_record[name]) or record[name] != current_record[name] for name in expected_fields):
            raise RuntimeError(f"Collision cache ScenarioSpec does not match current candidate {scenario.scenario_id}")
        collision_scenarios.append(scenario)
    return tuple(collision_scenarios)


def _validate_classification_summary(path: Path, outcomes: list[dict], candidate_count: int) -> None:
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
    if not isinstance(summary, dict) or set(summary) != expected_keys or any(type(summary[name]) is not int or summary[name] != value for name, value in expected_counts.items()):
        raise RuntimeError(f"Collision cache summary does not match {candidate_count} candidate outcomes")
    if type(summary["env_workers"]) is not int or summary["env_workers"] <= 0:
        raise RuntimeError("Collision cache summary has invalid env_workers")
    for name in ("wall_seconds", "scenarios_per_second"):
        value = summary[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise RuntimeError(f"Collision cache summary has invalid {name}")


def load_collision_cache(cache_dir: Path, current_config: dict, candidates: tuple[ScenarioSpec, ...]) -> tuple[ScenarioSpec, ...]:
    with (cache_dir / "classification_config.json").open("r", encoding="utf-8") as file:
        cached_config = json.load(file)
    candidate_count = len(candidates)
    if json.dumps(cached_config, sort_keys=True) != json.dumps(current_config, sort_keys=True):
        raise RuntimeError(f"Collision cache configuration does not match the current {candidate_count} candidates; use --reclassify_collisions")
    outcomes = _load_candidate_outcomes(cache_dir / "candidate_outcomes.jsonl", candidates)
    collision_scenarios = _load_collision_scenarios(cache_dir / "collision_scenarios.json", candidates, outcomes)
    _validate_classification_summary(cache_dir / "classification_summary.json", outcomes, candidate_count)
    return collision_scenarios


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
