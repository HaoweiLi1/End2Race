import json
from dataclasses import asdict
import math
import numbers
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
import numpy as np


__all__ = [
    "load_raceline_waypoints",
    "load_raceline_with_speed",
    "calculate_metrics",
    "follow_vehicle_camera",
    "set_score_label",
    "update_point_batches",
    "create_multiagent_render_callback",
    "create_planner_render_callback",
    "create_single_agent_render_callback",
    "find_corresponding_waypoint",
    "load_positions_and_speeds_from_params",
    "get_circular_startpoints",
    "get_opponent_startpoint",
    "evaluation_root",
    "singleagent_paths",
    "episode_key",
    "multiagent_paths",
    "load_json",
    "atomic_write_json",
    "update_singleagent_results",
    "save_numeric_npz",
    "evaluate_proximity_quality",
    "wrapped_progress_difference",
    "aggregate_multiagent_batch",
    "require_finite_number",
    "require_finite_tensor",
    "TrainingRecorder",
]

def load_raceline_waypoints(map_name, raceline_file):
    """Load raceline waypoints as an (N, 4) array of [x, y, theta, speed]"""
    raceline_path = f"f1tenth_racetracks/{map_name}/{raceline_file}"
    with open(raceline_path, 'r') as f:
        lines = f.readlines()[1:]

    waypoints = []
    for line in lines:
        parts = line.strip().split(';')
        if len(parts) >= 6:
            waypoints.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[5])])
    return np.array(waypoints)

def load_raceline_with_speed(map_name, raceline_file, start_idx):
    """Load raceline waypoints with position and speed information"""
    waypoints = load_raceline_waypoints(map_name, raceline_file)

    # Get starting position and speed
    idx = start_idx % len(waypoints)
    start_pose = np.array([waypoints[idx, :3]])
    initial_speed = waypoints[idx, 3]

    return start_pose, initial_speed, waypoints

def calculate_metrics(trajectory, speeds):
    """Calculate performance metrics"""
    avg_speed = np.mean(speeds) if speeds else 0
    speed_variance = np.var(speeds) if speeds else 0
    total_distance = sum(np.linalg.norm(np.array(trajectory[i+1]) - np.array(trajectory[i]))
                        for i in range(len(trajectory)-1)) if len(trajectory) > 1 else 0
    return avg_speed, speed_variance, total_distance

def follow_vehicle_camera(event, car_index=0, margin=800.0):
    """Center the camera on the specified vehicle and apply symmetric margins."""
    x_vertices = event.cars[car_index].vertices[::2]
    y_vertices = event.cars[car_index].vertices[1::2]
    center_x = float(np.mean(x_vertices))
    center_y = float(np.mean(y_vertices))
    event.left, event.right = center_x - margin, center_x + margin
    event.top, event.bottom = center_y + margin, center_y - margin
    return center_x, center_y

def set_score_label(event, x_offset, y_offset, vertical_anchor='bottom'):
    """Position the score label relative to the viewport bounds."""
    event.score_label.x = event.left + x_offset
    if vertical_anchor == 'top':
        event.score_label.y = event.top + y_offset
    else:
        event.score_label.y = event.bottom + y_offset

def update_point_batches(event, batches, points, color, batch_objects=None, scale=10.0):
    """Populate or update pyglet point batches with the provided 2D points."""
    from pyglet.gl import GL_POINTS

    color_stream = list(color)
    for idx, point in enumerate(points):
        x_coord, y_coord = float(point[0]) * scale, float(point[1]) * scale
        if idx < len(batches):
            batches[idx].vertices = [x_coord, y_coord, 0.0]
        else:
            batch_item = event.batch.add(
                1,
                GL_POINTS,
                None,
                ('v3f/stream', [x_coord, y_coord, 0.0]),
                ('c3B/stream', color_stream)
            )
            batches.append(batch_item)
            if batch_objects is not None:
                batch_objects.append(batch_item)

def create_multiagent_render_callback(render_info, visited_points, drawn_points, batch_objects, colors=None, margin=800.0):
    """Create a render callback that visualizes two vehicles and their trajectories."""
    if colors is None:
        colors = [(255, 255, 0), (255, 0, 0)]

    def render_callback(event):
        follow_vehicle_camera(event, margin=margin)
        set_score_label(event, 800, 100, vertical_anchor='bottom')

        event.score_label.text = (
            f"State: {render_info['state']} | "
            f"Ego: {render_info['ego_speed']:.1f}m/s, {render_info['ego_steer']:+.2f}rad | "
            f"Opp: {render_info['opp_speed']:.1f}m/s, {render_info['opp_steer']:+.2f}rad"
        )

        for vehicle_idx, color in enumerate(colors):
            if vehicle_idx < len(drawn_points) and vehicle_idx < len(visited_points):
                update_point_batches(
                    event,
                    drawn_points[vehicle_idx],
                    visited_points[vehicle_idx],
                    color,
                    batch_objects=batch_objects,
                    scale=50.0
                )

    return render_callback

def create_planner_render_callback(render_info, planner_getter, draw_grid_pts, draw_traj_pts, margin=800.0):

    def render_callback(event):
        planner = planner_getter()

        follow_vehicle_camera(event, margin=margin)
        set_score_label(event, 800, 100, vertical_anchor='bottom')

        event.score_label.text = (
            f"Ego: {render_info['ego_speed']:.1f}m/s, {render_info['ego_steer']:+.2f}rad | "
            f"Opp: {render_info['opp_speed']:.1f}m/s, {render_info['opp_steer']:+.2f}rad"
        )

        if planner and planner.goal_grid is not None:
            goal_grid_pts = np.column_stack((planner.goal_grid[:, 0], planner.goal_grid[:, 1]))
            update_point_batches(event, draw_grid_pts, goal_grid_pts, color=(183, 193, 222), scale=50.0)

            if planner.best_traj is not None:
                best_traj_pts = np.column_stack((planner.best_traj[:, 0], planner.best_traj[:, 1]))
                update_point_batches(event, draw_traj_pts, best_traj_pts, color=(183, 193, 222), scale=50.0)

        if planner:
            planner.tracker.render_waypoints(event)

    render_callback.render_info = render_info
    return render_callback

def create_single_agent_render_callback(render_info, visited_points, drawn_points, batch_objects, lap_num):
    """Create render callback with proper trajectory visualization"""
    from pyglet.gl import GL_POINTS

    def render_callback(event):
        # Camera following ego vehicle
        x_vertices = event.cars[0].vertices[::2]
        y_vertices = event.cars[0].vertices[1::2]
        event.left = float(np.min(x_vertices)) - 800
        event.right = float(np.max(x_vertices)) + 800
        event.top = float(np.max(y_vertices)) + 800
        event.bottom = float(np.min(y_vertices)) - 800
        event.score_label.x = event.left + 800
        event.score_label.y = event.top - 1500

        event.score_label.text = (
            f"Laps: {render_info['laps']}/{lap_num} | "
            f"Time: {render_info['lap_time']:.1f}s | "
            f"Speed: {render_info['speed']:.1f}m/s | "
            f"Steer: {render_info['steer']:+.2f}rad"
        )

        # Draw trajectory points (this is the key part that was missing)
        for i, pt in enumerate(visited_points):
            x, y = 50.0 * pt[0], 50.0 * pt[1]
            if i < len(drawn_points):
                drawn_points[i].vertices = [x, y, 0.0]
            else:
                b = event.batch.add(1, GL_POINTS, None,
                              ('v3f/stream', [x, y, 0.0]),
                              ('c3B/stream', [255, 255, 0]))  # Yellow trajectory
                drawn_points.append(b)
                batch_objects.append(b)

    return render_callback

def find_corresponding_waypoint(ego_waypoint, opp_waypoints):
    """Find the waypoint on opponent raceline closest to ego waypoint spatially"""
    ego_position = ego_waypoint[:2]
    distances = np.linalg.norm(opp_waypoints[:, :2] - ego_position, axis=1)
    return np.argmin(distances)

def load_positions_and_speeds_from_params(params, map_name):
    """Load initial positions and speeds based on segment parameters (from run_lattice_planner.py)"""
    ego_waypoints = load_raceline_waypoints(map_name, params['ego_raceline'] + '.csv')
    if params['opp_raceline'] != params['ego_raceline']:
        opp_waypoints = load_raceline_waypoints(map_name, params['opp_raceline'] + '.csv')
    else:
        opp_waypoints = ego_waypoints

    # Get positions and speeds using direct indices
    ego_idx = params['ego_idx'] % len(ego_waypoints)
    opp_idx = params['opp_idx'] % len(opp_waypoints)
    positions = np.array([ego_waypoints[ego_idx, :3], opp_waypoints[opp_idx, :3]])
    initial_speeds = np.array([ego_waypoints[ego_idx, 3], opp_waypoints[opp_idx, 3]])

    return positions, initial_speeds

def get_circular_startpoints(map_name, raceline_file, num_startpoints, offset):
    raceline_path = os.path.join('f1tenth_racetracks', map_name, raceline_file)
    waypoints = np.loadtxt(raceline_path, delimiter=';', comments='#')
    unique_waypoints = waypoints[:-1]
    track_length = waypoints[-1, 0]
    offset_progress = unique_waypoints[offset % len(unique_waypoints), 0]
    targets = (offset_progress + np.arange(num_startpoints) * track_length / num_startpoints) % track_length
    return [int(np.argmin(np.abs(unique_waypoints[:, 0] - target))) for target in targets]

def get_opponent_startpoint(map_name, ego_raceline, opp_raceline, ego_idx, interval_idx):
    ego_waypoints = load_raceline_waypoints(map_name, f'{ego_raceline}.csv')[:-1]
    opp_waypoints = load_raceline_waypoints(map_name, f'{opp_raceline}.csv')[:-1]
    ego_idx = ego_idx % len(ego_waypoints)
    mapped_idx = ego_idx if opp_raceline == ego_raceline else find_corresponding_waypoint(ego_waypoints[ego_idx], opp_waypoints)
    return (mapped_idx + interval_idx) % len(opp_waypoints)

# --- Evaluation artifact helpers (paths, JSON results, traces, metrics) ---

def evaluation_root(model_path, map_name, noise_level):
    """Root directory for all evaluation artifacts of one model/map/noise combination"""
    model = Path(model_path)
    update_dir = model.parent
    experiment_dir = update_dir.parent

    if model.name == "actor.pth" and update_dir.name.startswith("update"):
        root = (
            Path("eval_results")
            / experiment_dir.name
            / update_dir.name
            / map_name
        )
    elif model.name == "end2race.pth" and model.parent.name == "pretrained":
        root = Path("eval_results") / "pretrained_end2race" / map_name
    else:
        model_name = model.stem
        noise_suffix = (
            f"_noise{int(noise_level * 100)}"
            if noise_level > 0
            else ""
        )
        return Path("eval_results") / f"{model_name}_{map_name}{noise_suffix}"

    if noise_level > 0:
        root = root / f"noise{int(noise_level * 100)}"
    return root

def singleagent_paths(model_path, map_name, noise_level, lap_num):
    """Artifact paths for a single-agent evaluation run"""
    root = evaluation_root(model_path, map_name, noise_level) / "singleagent"
    return {
        "root": root,
        "results": root / "results_single.json",
        "video": root / "videos" / f"lap{lap_num}.mp4",
        "trace": root / "traces" / f"lap{lap_num}.npz",
    }

def episode_key(opponent_raceline, ego_index, opponent_index, opponent_speed_scale):
    """Unique key identifying one multi-agent episode"""
    raceline_number = opponent_raceline.replace("raceline", "").replace(".csv", "")
    return f"ol{raceline_number}_e{ego_index}_o{opponent_index}_s{opponent_speed_scale}"

def multiagent_paths(model_path, map_name, noise_level, key=None, state_prefix=None):
    """Artifact paths for a multi-agent evaluation run"""
    root = evaluation_root(model_path, map_name, noise_level) / "multiagents"
    paths = {"root": root, "results": root / "results_multi.json"}
    if key is not None:
        paths["trace"] = root / "traces" / f"{key}.npz"
        if state_prefix is not None:
            paths["video"] = root / "videos" / f"{state_prefix}_{key}.mp4"
    return paths

def load_json(path, default):
    """Load a JSON file, returning default if it does not exist"""
    source = Path(path)
    if not source.is_file():
        return default
    with source.open("r", encoding="utf-8") as stream:
        return json.load(stream)

def atomic_write_json(path, value):
    """Write JSON to a temporary file, then atomically replace the destination"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)

def update_singleagent_results(path, lap_key, metrics):
    """Insert one lap entry into the single-agent results JSON"""
    destination = Path(path)
    document = load_json(destination, {})
    if not isinstance(document, dict):
        raise ValueError(f"Single-agent results must contain a JSON object: {destination}")
    document[str(lap_key)] = dict(metrics)
    atomic_write_json(destination, dict(sorted(document.items())))

def save_numeric_npz(path, arrays):
    """Atomically save arrays as a compressed .npz trace"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)

def evaluate_proximity_quality(lidar_history):
    lidar = np.asarray(lidar_history, dtype=np.float64)
    beam_angles = -np.pi + np.arange(360) * 2.0 * np.pi / 360
    body_edge_distance = np.minimum(
        0.58 / 2 / np.maximum(np.abs(np.cos(beam_angles)), 1e-12),
        0.31 / 2 / np.maximum(np.abs(np.sin(beam_angles)), 1e-12),
    )
    surface_distance = np.clip(lidar - body_edge_distance, 0.0, None)
    danger_sectors = {}
    for sector_name, sector_indices in {
        "rear": list(range(0, 30)) + list(range(330, 360)),
        "rear_right": range(30, 60),
        "right": range(60, 120),
        "front_right": range(120, 150),
        "front": range(150, 210),
        "front_left": range(210, 240),
        "left": range(240, 300),
        "rear_left": range(300, 330),
    }.items():
        sector_minimum = float(np.min(surface_distance[:, sector_indices]))
        if sector_minimum < 0.15:
            danger_sectors[sector_name] = round(sector_minimum, 4)
    below_threshold = np.flatnonzero(np.min(surface_distance, axis=1) < 0.15).astype(int).tolist()
    return {
        "global_min_surface_dist": round(float(np.min(surface_distance)), 4),
        "danger_sectors": danger_sectors,
        "proximity_below_threshold_timesteps": below_threshold,
    }

def wrapped_progress_difference(ego_progress, opp_progress, track_length):
    """Signed progress difference wrapped to [-track_length/2, track_length/2)"""
    return (ego_progress - opp_progress + 0.5 * track_length) % track_length - 0.5 * track_length

def aggregate_multiagent_batch(results_path, temporary_directory, total_segments):
    """Merge worker-local metrics and exit codes into results_multi.json once."""
    counts = {"following": 0, "overtaking": 0, "collision": 0, "error": 0}
    collision_counts = {"ego-opp": 0, "ego-wall": 0, "opp-wall": 0}
    categories = {1: "following", 2: "overtaking", 3: "collision"}
    batch_episodes = {}
    batch_metrics = []
    for exit_path in sorted(Path(temporary_directory).glob("*.exit")):
        exit_code = int(exit_path.read_text(encoding="utf-8").strip())
        counts[categories.get(exit_code, "error")] += 1
        if exit_code not in categories:
            continue
        metrics_path = exit_path.with_suffix(".metrics.json")
        metrics = load_json(metrics_path, None)
        key = str(metrics.pop("episode_key"))
        if exit_code == 3:
            collision_counts[metrics["outcome"]] += 1
        batch_episodes[key] = metrics
        batch_metrics.append(metrics)
    success_count = counts["following"] + counts["overtaking"]
    denominator = total_segments if total_segments else 1
    following_rate = counts["following"] * 100.0 / denominator
    overtaking_rate = counts["overtaking"] * 100.0 / denominator
    success_rate = success_count * 100.0 / denominator
    collision_rate = counts["collision"] * 100.0 / denominator

    def batch_mean(field):
        return float(np.mean([float(metrics[field]) for metrics in batch_metrics])) if batch_metrics else 0.0

    final = {
        "total_episodes": int(total_segments),
        "following_count": counts["following"],
        "overtaking_count": counts["overtaking"],
        "success_count": success_count,
        "collision_count": counts["collision"],
        "ego_opp_collision_count": collision_counts["ego-opp"],
        "ego_wall_collision_count": collision_counts["ego-wall"],
        "opp_wall_collision_count": collision_counts["opp-wall"],
        "error_count": counts["error"],
        "following_rate": following_rate,
        "overtaking_rate": overtaking_rate,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "avg_speed_mean": batch_mean("avg_speed"),
        "speed_variance_mean": batch_mean("speed_variance"),
        "total_distance_mean": batch_mean("total_distance"),
    }
    destination = Path(results_path)
    document = load_json(destination, {"final": {}, "episodes": {}})
    episodes = document.get("episodes", {}) if isinstance(document, dict) else {}
    if not isinstance(episodes, dict):
        raise ValueError(f"Multi-agent episodes must contain a JSON object: {destination}")
    episodes.update(batch_episodes)
    atomic_write_json(destination, {"final": final, "episodes": dict(sorted(episodes.items()))})
    return final


def require_finite_number(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise RuntimeError(f"{name} must be finite, got {value!r}")
    return float(value)


def require_finite_tensor(name, value):
    import torch

    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{name} must be a finite tensor")


class TrainingRecorder:

    def __init__(self, output_dir, hidden_scale):
        self.output_dir = Path(output_dir).expanduser().resolve()
        if self.output_dir.exists():
            if not self.output_dir.is_dir():
                raise RuntimeError(f"PPO output path is not a directory: {self.output_dir}")
            if any(self.output_dir.iterdir()):
                raise RuntimeError(f"PPO output directory must be empty: {self.output_dir}")
        else:
            self.output_dir.mkdir(parents=True)
        self.hidden_scale = int(hidden_scale)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir()
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.episodes_path = self.output_dir / "episodes.jsonl"
        self.metrics_path.touch()
        self.episodes_path.touch()

    @staticmethod
    def _write_json(path, payload):
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")

    @staticmethod
    def _append_jsonl(path, payload):
        with Path(path).open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, allow_nan=False) + "\n")

    @staticmethod
    def _cpu_state_dict(state_dict):
        return {name: tensor.detach().cpu() for name, tensor in state_dict.items()}

    def write_run_config(self, args, ppo_config, training_constants):
        self._write_json(
            self.output_dir / "run_config.json",
            {
                "args": dict(vars(args)),
                "ppo_config": ppo_config,
                **training_constants,
            },
        )

    def write_scenario_pools(self, collision_scenarios, ordinary_scenarios, cache_info):
        self._write_json(self.output_dir / "collision_scenarios.json", [asdict(scenario) for scenario in collision_scenarios])
        self._write_json(self.output_dir / "ordinary_scenarios.json", [asdict(scenario) for scenario in ordinary_scenarios])
        self._write_json(self.output_dir / "collision_cache_info.json", cache_info)

    def record_episode(self, record):
        self._append_jsonl(self.episodes_path, record)

    @classmethod
    def _require_finite_metrics(cls, name, value):
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            require_finite_number(name, value)
        elif isinstance(value, Mapping):
            for child_name, child_value in value.items():
                cls._require_finite_metrics(f"{name}.{child_name}", child_value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, child_value in enumerate(value):
                cls._require_finite_metrics(f"{name}[{index}]", child_value)

    def record_metrics(self, record):
        for name, value in record.items():
            self._require_finite_metrics(name, value)
        self._append_jsonl(self.metrics_path, record)

    def _save_actor(self, path, state_dict):
        import torch
        from model import End2Race

        checkpoint = self._cpu_state_dict(state_dict)
        if len(checkpoint) != 12:
            raise RuntimeError(f"Expected a 12-key actor checkpoint, got {len(checkpoint)} keys")
        torch.save(checkpoint, path)
        with torch.random.fork_rng(devices=[]):
            actor = End2Race(mask_prob=0.0, hidden_scale=self.hidden_scale)
            actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        return path

    def save_warmup_critic(self, state_dict):
        import torch

        path = self.checkpoints_dir / "critic_warmup.pt"
        torch.save(self._cpu_state_dict(state_dict), path)
        return path

    def save_formal_checkpoints(self, update, actor_state_dict, critic_state_dict):
        import torch

        actor_path = self._save_actor(self.checkpoints_dir / f"actor_u{update:04d}.pth", actor_state_dict)
        critic_path = self.checkpoints_dir / f"critic_u{update:04d}.pt"
        torch.save(self._cpu_state_dict(critic_state_dict), critic_path)
        return actor_path, critic_path
