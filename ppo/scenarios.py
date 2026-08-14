from dataclasses import asdict, dataclass
import numpy as np
from utils import *

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
    ego_raceline: str
    sim_duration: float
    timestep: float
    integrator: str = "RK4"

    def to_reset_spec(self, env_role: str) -> dict:
        scenario = asdict(self)
        scenario["opponent_speed_scale"] = self.opp_speedscale
        scenario["sampler_branch"] = env_role
        scenario["env_role"] = env_role
        poses, initial_speeds = load_positions_and_speeds_from_params(scenario, self.map_name)
        return {
            "poses": np.asarray(poses, dtype=np.float64),
            "initial_speed_feature": float(initial_speeds[0] * 0.9),
            "scenario": scenario,
        }

class ScenarioScheduler:

    def __init__(self, seed, collision_scenarios, ordinary_scenarios):
        collision_seed, ordinary_seed = np.random.SeedSequence(seed).spawn(2)
        self.collision = self._scenario_queue(collision_scenarios, collision_seed)
        self.ordinary = self._scenario_queue(ordinary_scenarios, ordinary_seed)

    @staticmethod
    def _scenario_queue(scenarios, seed_sequence):
        rng = np.random.default_rng(seed_sequence)
        while True:
            for index in rng.permutation(len(scenarios)):
                yield scenarios[int(index)]

    def next(self, rank):
        if rank % 2 == 0:
            return next(self.collision).to_reset_spec("collision")
        return next(self.ordinary).to_reset_spec("ordinary")

def ordinary_scenarios(map_name, config):
    ego_waypoints = load_raceline_waypoints(map_name, f"{config.ego_raceline}.csv")
    opponent_waypoints = {raceline: ego_waypoints if raceline == config.ego_raceline else load_raceline_waypoints(map_name, f"{raceline}.csv") for raceline in config.opponent_racelines}
    scenarios = []
    startpoints = get_circular_startpoints(map_name, f"{config.ego_raceline}.csv", config.ordinary_startpoint_count * 2, 0)[1::2]
    for ordinal, ego_idx in enumerate(startpoints):
        for opp_raceline in config.opponent_racelines:
            opp_idx = get_opponent_startpoint_from_waypoints(ego_waypoints, opponent_waypoints[opp_raceline], ego_idx, config.ordinary_interval_index, opp_raceline == config.ego_raceline)
            for speed_scale in config.ordinary_speed_scales:
                scenarios.append(ScenarioSpec(f"ordinary-sp{ordinal:02d}-ego{ego_idx:04d}-{opp_raceline}-v{int(100 * speed_scale):03d}", "ordinary", ordinal, ego_idx, opp_idx, opp_raceline, speed_scale, config.ordinary_interval_index, map_name, config.ego_raceline, config.episode_horizon, config.simulator_timestep))
    return tuple(scenarios)


def expanded_scenarios(map_name, config):
    ego_waypoints = load_raceline_waypoints(map_name, f"{config.ego_raceline}.csv")
    opponent_waypoints = {raceline: ego_waypoints if raceline == config.ego_raceline else load_raceline_waypoints(map_name, f"{raceline}.csv") for raceline in config.opponent_racelines}
    scenarios = []
    startpoints = get_circular_startpoints(map_name, f"{config.ego_raceline}.csv", config.collision_startpoint_count * 2, 0)[1::2]
    for ordinal, ego_idx in enumerate(startpoints):
        for opp_raceline in config.opponent_racelines:
            for interval_idx in config.collision_interval_indices:
                opp_idx = get_opponent_startpoint_from_waypoints(ego_waypoints, opponent_waypoints[opp_raceline], ego_idx, interval_idx, opp_raceline == config.ego_raceline)
                for speed_scale in config.collision_speed_scales:
                    scenarios.append(ScenarioSpec(f"collision-sp{ordinal:03d}-ego{ego_idx:04d}-{opp_raceline}-i{interval_idx:02d}-v{round(100 * speed_scale):03d}", "collision", ordinal, ego_idx, opp_idx, opp_raceline, speed_scale, interval_idx, map_name, config.ego_raceline, config.episode_horizon, config.simulator_timestep))
    return tuple(scenarios)