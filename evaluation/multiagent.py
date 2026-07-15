"""End2Race actor versus a fixed Lattice Planner opponent."""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
from pathlib import Path
import traceback
from typing import Any, Callable, Mapping

import numpy as np
import torch

from evaluation.artifacts import atomic_write_json, atomic_write_npz, checkpoint_sha256
from evaluation.metrics import ClosedTrack, episode_distance
from evaluation.schema import EVALUATION_SCHEMA_VERSION, Scenario
from model import End2Race


LIDAR_SIZE = 360
STEERING_BOUND_RAD = 0.52
SIMULATION_TIMESTEP_S = 0.01
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRACE_ARRAY_NAMES = (
    "time_s",
    "ego_lidar_360",
    "opp_lidar_360",
    "ego_raw_action",
    "ego_executed_action",
    "opp_executed_action",
    "ego_measured_speed_mps",
    "opp_measured_speed_mps",
    "decision_poses",
    "post_step_poses",
    "post_step_collisions",
    "post_step_ego_progress_m",
    "post_step_opp_progress_m",
    "post_step_relative_progress_m",
)


def downsample_lidar(scan: Any) -> np.ndarray:
    values = np.asarray(scan).reshape(-1)
    if values.size < LIDAR_SIZE:
        raise ValueError(f"LiDAR scan has {values.size} beams; at least {LIDAR_SIZE} are required")
    if not np.isfinite(values).all():
        raise ValueError("LiDAR scan contains NaN or Inf")
    if values.size > LIDAR_SIZE:
        indices = np.linspace(0, values.size - 1, LIDAR_SIZE, dtype=int)
        values = values[indices]
    return np.asarray(values, dtype=np.float32)


def collision_flags(collisions: Any, ego_index: int = 0) -> tuple[bool, bool]:
    values = np.asarray(collisions, dtype=bool).reshape(-1)
    ego_collision = bool(values.size > ego_index and values[ego_index])
    opponent_collision = any(bool(values[index]) for index in range(values.size) if index != ego_index)
    return ego_collision, bool(opponent_collision)


def load_raceline(map_name: str, raceline: str) -> np.ndarray:
    path = REPOSITORY_ROOT / "f1tenth_racetracks" / map_name / f"{raceline}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Raceline does not exist: {path}")
    values = np.loadtxt(path, delimiter=";", skiprows=1)
    if values.ndim != 2 or values.shape[1] < 6:
        raise ValueError(f"Raceline must contain at least six columns: {path}")
    waypoints = np.column_stack((values[:, 1], values[:, 2], values[:, 3], values[:, 5]))
    if not np.isfinite(waypoints).all():
        raise ValueError(f"Raceline contains NaN or Inf: {path}")
    return waypoints


def opponent_start_index(
    map_name: str,
    ego_raceline: str,
    opponent_raceline: str,
    ego_start_index: int,
    interval_index: int,
) -> int:
    ego_waypoints = load_raceline(map_name, ego_raceline)
    opponent_waypoints = (
        ego_waypoints if opponent_raceline == ego_raceline else load_raceline(map_name, opponent_raceline)
    )
    ego_index = int(ego_start_index) % len(ego_waypoints)
    if opponent_raceline == ego_raceline:
        mapped_index = ego_index
    else:
        distances = np.linalg.norm(opponent_waypoints[:, :2] - ego_waypoints[ego_index, :2], axis=1)
        mapped_index = int(np.argmin(distances))
    return int((mapped_index + interval_index) % len(opponent_waypoints))


def default_track_setup(scenario: Scenario) -> tuple[np.ndarray, np.ndarray, ClosedTrack]:
    ego_waypoints = load_raceline(scenario.map_name, scenario.ego_raceline)
    opponent_waypoints = (
        ego_waypoints
        if scenario.opponent_raceline == scenario.ego_raceline
        else load_raceline(scenario.map_name, scenario.opponent_raceline)
    )
    ego_index = scenario.ego_start_index % len(ego_waypoints)
    opponent_index = scenario.opponent_start_index % len(opponent_waypoints)
    poses = np.asarray(
        (ego_waypoints[ego_index, :3], opponent_waypoints[opponent_index, :3]), dtype=np.float64
    )
    initial_speeds = np.asarray(
        (ego_waypoints[ego_index, 3], opponent_waypoints[opponent_index, 3]), dtype=np.float64
    )
    reference_waypoints = load_raceline(scenario.map_name, "raceline1")
    return poses, initial_speeds, ClosedTrack.from_points(reference_waypoints[:, :2])


def default_environment_factory(scenario: Scenario) -> Any:
    import gym
    import f110_gym.envs.f110_env  # noqa: F401 - registers f110-v0
    from f110_gym.envs.base_classes import Integrator

    map_path = REPOSITORY_ROOT / "f1tenth_racetracks" / scenario.map_name / f"{scenario.map_name}_map"
    return gym.make(
        "f110-v0",
        map=str(map_path),
        map_ext=".png",
        num_agents=2,
        ego_idx=0,
        timestep=SIMULATION_TIMESTEP_S,
        integrator=Integrator.RK4,
    )


def default_planner_factory(scenario: Scenario) -> Any:
    from argparse import Namespace
    import yaml

    from latticeplanner.lattice_planner import LatticePlanner

    config_path = REPOSITORY_ROOT / "latticeplanner" / "lattice_config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = Namespace(**yaml.load(stream, Loader=yaml.FullLoader))
    map_directory = REPOSITORY_ROOT / "f1tenth_racetracks" / scenario.map_name
    planner = LatticePlanner(
        config,
        str(map_directory / f"{scenario.map_name}_map"),
        str(map_directory / f"{scenario.opponent_raceline}.csv"),
    )
    planner.set_parameters(
        {
            "cost_weights": np.asarray((1.0, 2.0, 0.5, 0.0), dtype=np.float64),
            "traj_v_scale": 1.0,
        }
    )
    return planner


def _reset_observation(result: Any) -> Mapping[str, Any]:
    if not isinstance(result, tuple):
        return result
    if len(result) in (2, 4):
        return result[0]
    raise ValueError(f"Unsupported environment reset result with {len(result)} entries")


def _step_result(result: Any) -> tuple[Mapping[str, Any], float, bool]:
    if not isinstance(result, tuple):
        raise TypeError("Environment step must return a tuple")
    if len(result) == 4:
        observation, reward, done, _ = result
        return observation, float(reward), bool(done)
    if len(result) == 5:
        observation, reward, terminated, truncated, _ = result
        return observation, float(reward), bool(terminated or truncated)
    raise ValueError(f"Unsupported environment step result with {len(result)} entries")


def _poses(observation: Mapping[str, Any]) -> np.ndarray:
    return np.column_stack(
        (
            np.asarray(observation["poses_x"], dtype=np.float64),
            np.asarray(observation["poses_y"], dtype=np.float64),
            np.asarray(observation["poses_theta"], dtype=np.float64),
        )
    )


@dataclass
class TraceBuffer:
    time_s: list[float] = field(default_factory=list)
    ego_lidar_360: list[np.ndarray] = field(default_factory=list)
    opp_lidar_360: list[np.ndarray] = field(default_factory=list)
    ego_raw_action: list[np.ndarray] = field(default_factory=list)
    ego_executed_action: list[np.ndarray] = field(default_factory=list)
    opp_executed_action: list[np.ndarray] = field(default_factory=list)
    ego_measured_speed_mps: list[float] = field(default_factory=list)
    opp_measured_speed_mps: list[float] = field(default_factory=list)
    decision_poses: list[np.ndarray] = field(default_factory=list)
    post_step_poses: list[np.ndarray] = field(default_factory=list)
    post_step_collisions: list[np.ndarray] = field(default_factory=list)
    post_step_ego_progress_m: list[float] = field(default_factory=list)
    post_step_opp_progress_m: list[float] = field(default_factory=list)
    post_step_relative_progress_m: list[float] = field(default_factory=list)

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "time_s": np.asarray(self.time_s, dtype=np.float64),
            "ego_lidar_360": np.asarray(self.ego_lidar_360, dtype=np.float32).reshape(-1, LIDAR_SIZE),
            "opp_lidar_360": np.asarray(self.opp_lidar_360, dtype=np.float32).reshape(-1, LIDAR_SIZE),
            "ego_raw_action": np.asarray(self.ego_raw_action, dtype=np.float32).reshape(-1, 2),
            "ego_executed_action": np.asarray(self.ego_executed_action, dtype=np.float32).reshape(-1, 2),
            "opp_executed_action": np.asarray(self.opp_executed_action, dtype=np.float32).reshape(-1, 2),
            "ego_measured_speed_mps": np.asarray(self.ego_measured_speed_mps, dtype=np.float32),
            "opp_measured_speed_mps": np.asarray(self.opp_measured_speed_mps, dtype=np.float32),
            "decision_poses": np.asarray(self.decision_poses, dtype=np.float64).reshape(-1, 2, 3),
            "post_step_poses": np.asarray(self.post_step_poses, dtype=np.float64).reshape(-1, 2, 3),
            "post_step_collisions": np.asarray(self.post_step_collisions, dtype=np.bool_).reshape(-1, 2),
            "post_step_ego_progress_m": np.asarray(self.post_step_ego_progress_m, dtype=np.float64),
            "post_step_opp_progress_m": np.asarray(self.post_step_opp_progress_m, dtype=np.float64),
            "post_step_relative_progress_m": np.asarray(
                self.post_step_relative_progress_m, dtype=np.float64
            ),
        }
        assert tuple(arrays) == TRACE_ARRAY_NAMES
        return arrays


class OpponentDriver:
    def __init__(self, planner: Any, speed_scale: float) -> None:
        self.planner = planner
        self.speed_scale = float(speed_scale)
        self.trajectory: Any = None
        self.tracker_count = 0

    def action(self, observation: Mapping[str, Any]) -> np.ndarray:
        from latticeplanner.utils import obsDict2oppoArray

        pose_x = float(np.asarray(observation["poses_x"])[1])
        pose_y = float(np.asarray(observation["poses_y"])[1])
        pose_theta = float(np.asarray(observation["poses_theta"])[1])
        speed = float(np.asarray(observation["linear_vels_x"])[1])
        if self.tracker_count == 0 or self.trajectory is None:
            opponent_poses = obsDict2oppoArray(observation, 1)
            self.trajectory = self.planner.plan(pose_x, pose_y, pose_theta, opponent_poses, speed)
        steering, desired_speed = self.planner.tracker.plan(
            pose_x, pose_y, pose_theta, speed, self.trajectory
        )
        tracker_steps = int(getattr(self.planner.conf, "tracker_steps", 10))
        self.tracker_count = (self.tracker_count + 1) % tracker_steps
        return np.asarray(
            (
                np.clip(float(steering), -STEERING_BOUND_RAD, STEERING_BOUND_RAD),
                float(desired_speed) * self.speed_scale,
            ),
            dtype=np.float32,
        )


class MultiAgentEvaluator:
    """A model instance is retained across all scenarios handled by this worker."""

    def __init__(
        self,
        model: End2Race,
        device: torch.device | str,
        *,
        environment_factory: Callable[[Scenario], Any] = default_environment_factory,
        planner_factory: Callable[[Scenario], Any] = default_planner_factory,
        track_setup: Callable[[Scenario], tuple[np.ndarray, np.ndarray, ClosedTrack]] = default_track_setup,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.environment_factory = environment_factory
        self.planner_factory = planner_factory
        self.track_setup = track_setup
        self.model.eval()

    def evaluate_scenario(
        self,
        scenario: Scenario,
        run_dir: str | Path,
        *,
        trace_mode: str,
        record_video: bool,
    ) -> dict[str, Any]:
        if trace_mode not in {"none", "collision", "all"}:
            raise ValueError(f"Unsupported trace mode: {trace_mode}")
        run_path = Path(run_dir)
        poses, initial_speeds, track = self.track_setup(scenario)
        if np.asarray(poses).shape != (2, 3) or np.asarray(initial_speeds).shape != (2,):
            raise ValueError("Track setup must return poses (2, 3) and initial speeds (2,)")
        environment = self.environment_factory(scenario)
        planner = self.planner_factory(scenario)
        opponent = OpponentDriver(planner, scenario.opponent_speed_scale)
        frames: list[np.ndarray] = []
        trace = TraceBuffer()
        post_ego_speeds: list[float] = []
        try:
            observation = _reset_observation(environment.reset(poses=np.asarray(poses).copy()))
            if record_video:
                environment.render()
            initial_poses = _poses(observation)
            ego_wrapped = track.project(initial_poses[0, :2])
            opponent_wrapped = track.project(initial_poses[1, :2])
            ego_unwrapped = ego_wrapped
            opponent_unwrapped = opponent_wrapped
            previous_speed_feature = float(initial_speeds[0]) * 0.9
            hidden = torch.zeros(
                (1, 1, self.model.gru.hidden_size), dtype=torch.float32, device=self.device
            )
            elapsed = 0.0
            collision_step: int | None = None
            any_opponent_collision = False
            any_opponent_only_collision = False

            while elapsed + 1e-12 < scenario.simulation_duration_s:
                decision_poses = _poses(observation)
                ego_lidar = downsample_lidar(observation["scans"][0])
                opponent_lidar = downsample_lidar(observation["scans"][1])
                measured_speeds = np.asarray(observation["linear_vels_x"], dtype=np.float64)
                if measured_speeds.shape[0] < 2 or not np.isfinite(measured_speeds[:2]).all():
                    raise ValueError("Both measured speeds must be finite")

                with torch.no_grad():
                    lidar_tensor = torch.as_tensor(ego_lidar, device=self.device).reshape(1, 1, LIDAR_SIZE)
                    speed_tensor = torch.tensor(
                        [[[previous_speed_feature]]], dtype=torch.float32, device=self.device
                    )
                    action_sequence, hidden = self.model(lidar_tensor, speed_tensor, hidden)
                raw_action = np.asarray(action_sequence[0, -1].detach().cpu(), dtype=np.float32)
                if raw_action.shape != (2,) or not np.isfinite(raw_action).all():
                    raise ValueError("Actor must produce one finite [steering, desired_speed] action")
                executed_action = raw_action.copy()
                executed_action[0] = np.clip(
                    executed_action[0], -STEERING_BOUND_RAD, STEERING_BOUND_RAD
                )
                opponent_action = opponent.action(observation)
                joint_action = np.stack((executed_action, opponent_action))

                decision_time = elapsed
                post_observation, step_duration, _base_done = _step_result(environment.step(joint_action))
                if not np.isfinite(step_duration) or step_duration <= 0:
                    step_duration = float(
                        getattr(getattr(environment, "unwrapped", environment), "timestep", SIMULATION_TIMESTEP_S)
                    )
                elapsed += step_duration
                post_poses = _poses(post_observation)
                post_collisions = np.asarray(post_observation["collisions"], dtype=bool).reshape(-1)
                if post_collisions.size != 2:
                    raise ValueError(f"Expected two collision flags, got {post_collisions.size}")
                ego_collision, opponent_collision = collision_flags(post_collisions, ego_index=0)
                any_opponent_collision = any_opponent_collision or opponent_collision
                any_opponent_only_collision = any_opponent_only_collision or (
                    opponent_collision and not ego_collision
                )

                ego_wrapped, ego_unwrapped = track.unwrap(
                    ego_wrapped, ego_unwrapped, post_poses[0, :2]
                )
                opponent_wrapped, opponent_unwrapped = track.unwrap(
                    opponent_wrapped, opponent_unwrapped, post_poses[1, :2]
                )
                relative_progress = ego_unwrapped - opponent_unwrapped
                post_speeds = np.asarray(post_observation["linear_vels_x"], dtype=np.float64)
                if post_speeds.shape[0] < 2 or not np.isfinite(post_speeds[:2]).all():
                    raise ValueError("Both post-step measured speeds must be finite")
                post_ego_speeds.append(float(post_speeds[0]))

                trace.time_s.append(decision_time)
                trace.ego_lidar_360.append(ego_lidar.copy())
                trace.opp_lidar_360.append(opponent_lidar.copy())
                trace.ego_raw_action.append(raw_action.copy())
                trace.ego_executed_action.append(executed_action.copy())
                trace.opp_executed_action.append(opponent_action.copy())
                trace.ego_measured_speed_mps.append(float(measured_speeds[0]))
                trace.opp_measured_speed_mps.append(float(measured_speeds[1]))
                trace.decision_poses.append(decision_poses.copy())
                trace.post_step_poses.append(post_poses.copy())
                trace.post_step_collisions.append(post_collisions.copy())
                trace.post_step_ego_progress_m.append(ego_unwrapped)
                trace.post_step_opp_progress_m.append(opponent_unwrapped)
                trace.post_step_relative_progress_m.append(relative_progress)

                if record_video:
                    frame = environment.render(mode="rgb_array")
                    if frame is not None:
                        frames.append(np.asarray(frame))
                observation = post_observation
                # Preserve the original evaluator timing: the decision-time measured
                # speed is paired with the next decision LiDAR observation.
                previous_speed_feature = float(measured_speeds[0])
                if ego_collision:
                    collision_step = len(trace.time_s) - 1
                    break

            arrays = trace.arrays()
            steps = len(trace.time_s)
            if not steps:
                raise RuntimeError("Evaluation episode produced no simulator steps")
            ego_collision = collision_step is not None
            outcome = (
                "collision"
                if ego_collision
                else ("overtake" if trace.post_step_relative_progress_m[-1] > 0.0 else "follow")
            )
            trace_relative_path: str | None = None
            if trace_mode == "all" or (trace_mode == "collision" and ego_collision):
                trace_relative_path = f"traces/{scenario.scenario_id}.npz"
                atomic_write_npz(run_path / trace_relative_path, arrays)

            video_relative_path: str | None = None
            if record_video and not frames:
                raise RuntimeError("Video was selected but the renderer produced no frames")
            if record_video:
                import imageio.v2 as imageio

                video_relative_path = f"videos/{scenario.scenario_id}.mp4"
                temporary_video = run_path / "videos" / f".{scenario.scenario_id}.mp4.tmp.mp4"
                try:
                    imageio.mimwrite(temporary_video, frames, fps=100, macro_block_size=1)
                    temporary_video.replace(run_path / video_relative_path)
                finally:
                    temporary_video.unlink(missing_ok=True)

            executed_actions = arrays["ego_executed_action"]
            steering_delta = np.diff(executed_actions[:, 0])
            episode = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                **scenario.to_dict(),
                "outcome": outcome,
                "ego_collision": ego_collision,
                "opponent_collision": any_opponent_collision,
                "opponent_only_collision": any_opponent_only_collision,
                "collision_step": collision_step,
                "steps": steps,
                "elapsed_time_s": float(elapsed),
                "final_ego_progress_m": float(trace.post_step_ego_progress_m[-1]),
                "final_opp_progress_m": float(trace.post_step_opp_progress_m[-1]),
                "final_relative_progress_m": float(trace.post_step_relative_progress_m[-1]),
                "ego_distance_m": episode_distance(initial_poses[0, :2], arrays["post_step_poses"][:, 0, :2]),
                "ego_mean_measured_speed_mps": float(np.mean(post_ego_speeds)),
                "ego_speed_variance": float(np.var(post_ego_speeds)),
                "ego_min_measured_speed_mps": float(np.min(post_ego_speeds)),
                "ego_mean_desired_speed_mps": float(np.mean(executed_actions[:, 1])),
                "ego_max_abs_steer_rad": float(np.max(np.abs(executed_actions[:, 0]))),
                "ego_max_steer_delta_rad": float(np.max(np.abs(steering_delta))) if steering_delta.size else 0.0,
                "ego_min_lidar_m": float(np.min(arrays["ego_lidar_360"])),
                "trace_path": trace_relative_path,
                "video_path": video_relative_path,
            }
            atomic_write_json(run_path / "episodes" / f"{scenario.scenario_id}.json", episode)
            return episode
        finally:
            close = getattr(environment, "close", None)
            if close is not None:
                close()
            gc.collect()


def load_actor_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str,
    hidden_scale: int,
) -> End2Race:
    target = torch.device(device)
    model = End2Race(hidden_scale=hidden_scale).to(target)
    try:
        state_dict = torch.load(checkpoint_path, map_location=target, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=target)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


_WORKER_EVALUATOR: MultiAgentEvaluator | None = None


def initialize_worker(
    checkpoint_path: str,
    device: str,
    hidden_scale: int,
    expected_checkpoint_sha: str,
) -> None:
    global _WORKER_EVALUATOR
    actual_sha = checkpoint_sha256(checkpoint_path)
    if actual_sha != expected_checkpoint_sha:
        raise ValueError(
            f"Checkpoint changed after run creation: expected {expected_checkpoint_sha}, got {actual_sha}"
        )
    model = load_actor_checkpoint(checkpoint_path, device, hidden_scale)
    _WORKER_EVALUATOR = MultiAgentEvaluator(model, device)


def evaluate_worker_job(job: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("Evaluation worker was not initialized")
    scenario = Scenario.from_dict(job["scenario"])
    run_dir = Path(str(job["run_dir"]))
    try:
        episode = _WORKER_EVALUATOR.evaluate_scenario(
            scenario,
            run_dir,
            trace_mode=str(job["trace_mode"]),
            record_video=bool(job["record_video"]),
        )
        (run_dir / "errors" / f"{scenario.scenario_id}.json").unlink(missing_ok=True)
        return {"scenario_id": scenario.scenario_id, "ok": True, "episode": episode}
    except Exception as error:  # Batch isolation is intentional here.
        failure = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "scenario_id": scenario.scenario_id,
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(run_dir / "errors" / f"{scenario.scenario_id}.json", failure)
        return {"scenario_id": scenario.scenario_id, "ok": False, "error": failure}
