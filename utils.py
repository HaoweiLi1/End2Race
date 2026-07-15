import json
import os
import tempfile
from pathlib import Path
import numpy as np

PROXIMITY_THRESHOLD = 0.15

LIDAR_SECTORS = [
    ("rear", list(range(0, 30)) + list(range(330, 360))),
    ("rear_right", list(range(30, 60))),
    ("right", list(range(60, 120))),
    ("front_right", list(range(120, 150))),
    ("front", list(range(150, 210))),
    ("front_left", list(range(210, 240))),
    ("left", list(range(240, 300))),
    ("rear_left", list(range(300, 330))),
]

STEER_WINDOW_SECONDS = 1.0
STEER_MAX_REVERSALS = 6
STEER_MIN_AMP = 0.3
STEER_MAX_JUMP = 0.6

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

def get_ego_idx_range(map_name, ego_raceline, num_startpoints):
    """Generate evenly distributed evaluation points"""
    raceline_path = os.path.join('f1tenth_racetracks', map_name, ego_raceline)
    waypoints = np.loadtxt(raceline_path, delimiter=';', skiprows=1)
    max_waypoints = len(waypoints)
    ego_idx_range = np.linspace(0, max_waypoints - 1, num_startpoints, dtype=int).tolist()
    return ego_idx_range

# --- Evaluation artifact helpers (paths, JSON results, traces, metrics) ---

def evaluation_root(model_path, map_name, noise_level):
    """Root directory for all evaluation artifacts of one model/map/noise combination"""
    model_name = Path(model_path).stem
    noise_suffix = f"_noise{int(noise_level * 100)}" if noise_level > 0 else ""
    return Path("eval_results") / f"{model_name}_{map_name}{noise_suffix}"

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
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
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
    """Atomically save aligned numeric arrays as a compressed .npz trace"""
    converted = {name: np.asarray(value) for name, value in arrays.items()}
    lengths = set()
    for name, array in converted.items():
        if array.ndim < 1:
            raise ValueError(f"Trace array {name!r} has no leading dimension")
        if array.dtype == object or not (
            np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
        ):
            raise TypeError(f"Trace array {name!r} must be numeric or bool, got {array.dtype}")
        lengths.add(int(array.shape[0]))
    if len(lengths) != 1:
        raise ValueError(f"Trace arrays have inconsistent leading dimensions: {sorted(lengths)}")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(temporary_path, **converted)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)

def scalar_mean(values):
    return float(np.mean(values)) if values else 0.0

def scalar_variance(values):
    return float(np.var(values)) if values else 0.0

def scalar_min(values):
    return float(np.min(values)) if values else 0.0

def scalar_max_abs(values):
    return float(np.max(np.abs(values))) if values else 0.0

def scalar_max_delta(values):
    return float(np.max(np.abs(np.diff(values)))) if len(values) > 1 else 0.0

def _lidar_edge_distance(n_beams):
    """Distance from the vehicle center to its rectangular body edge per beam."""
    vehicle_length = 0.58
    vehicle_width = 0.31
    beam_angles = -np.pi + np.arange(n_beams) * 2.0 * np.pi / n_beams
    abs_cos = np.abs(np.cos(beam_angles))
    abs_sin = np.abs(np.sin(beam_angles))
    length_distance = np.divide(
        0.5 * vehicle_length,
        abs_cos,
        out=np.full(n_beams, np.inf, dtype=np.float64),
        where=abs_cos > 1e-12,
    )
    width_distance = np.divide(
        0.5 * vehicle_width,
        abs_sin,
        out=np.full(n_beams, np.inf, dtype=np.float64),
        where=abs_sin > 1e-12,
    )
    return np.minimum(length_distance, width_distance)

def _surface_distance_from_lidar(lidar):
    """Convert center-origin LiDAR ranges to distance from the vehicle body."""
    values = np.asarray(lidar, dtype=np.float64)
    return np.clip(values - _lidar_edge_distance(values.shape[-1]), 0.0, None)

def evaluate_proximity_quality(lidar_history):
    """Evaluate historical proximity metrics from raw ego 360-beam LiDAR."""
    lidar = np.asarray(lidar_history, dtype=np.float64)
    if lidar.ndim != 2 or lidar.shape[1] != 360:
        raise ValueError(f"lidar_history must have shape [T, 360], got {lidar.shape}")
    if lidar.shape[0] == 0:
        return {
            "global_min_surface_dist": 0.0,
            "danger_sectors": {},
            "proximity_below_threshold_timesteps": [],
        }
    surface_distance = _surface_distance_from_lidar(lidar)
    global_minimum = float(np.min(surface_distance))
    danger_sectors = {}
    for sector_name, sector_indices in LIDAR_SECTORS:
        sector_minimum = float(np.min(surface_distance[:, sector_indices]))
        if sector_minimum < PROXIMITY_THRESHOLD:
            danger_sectors[sector_name] = round(sector_minimum, 4)
    below_threshold = np.flatnonzero(
        np.min(surface_distance, axis=1) < PROXIMITY_THRESHOLD
    ).astype(int).tolist()
    return {
        "global_min_surface_dist": round(global_minimum, 4),
        "danger_sectors": danger_sectors,
        "proximity_below_threshold_timesteps": below_threshold,
    }

def evaluate_steering_quality(steer, sample_interval):
    """Evaluate the historical jump, reversal, oscillation, and autocorrelation metrics."""
    steering = np.asarray(steer, dtype=np.float64).reshape(-1)
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    delta = np.diff(steering)
    large_delta_indices = np.flatnonzero(np.abs(delta) >= STEER_MIN_AMP)
    large_delta_signs = np.sign(delta[large_delta_indices])
    reversal_timesteps = [
        int(large_delta_indices[index] + 1)
        for index in range(1, len(large_delta_indices))
        if large_delta_signs[index] != large_delta_signs[index - 1]
    ]

    window_size = max(1, int(round(STEER_WINDOW_SECONDS / sample_interval)))
    max_reversals = 0
    oscillation_timesteps = set()
    window_starts = range(max(1, len(steering) - window_size + 1))
    for window_start in window_starts:
        window_end = min(len(steering), window_start + window_size)
        reversal_count = sum(
            window_start <= timestep < window_end
            for timestep in reversal_timesteps
        )
        max_reversals = max(max_reversals, reversal_count)
        if reversal_count > STEER_MAX_REVERSALS:
            oscillation_timesteps.update(range(window_start, window_end))

    jump_timesteps = (np.flatnonzero(np.abs(delta) > STEER_MAX_JUMP) + 1).astype(int)
    anomaly_timesteps = sorted(
        set(int(value) for value in jump_timesteps) | oscillation_timesteps
    )
    max_steer_delta = float(np.max(np.abs(delta))) if delta.size else 0.0

    if steering.size < 2:
        autocorrelation = 1.0
    else:
        centered = steering - np.mean(steering)
        denominator = float(np.sum(centered ** 2))
        if denominator == 0.0:
            autocorrelation = 1.0
        else:
            numerator = float(np.sum(centered[:-1] * centered[1:]))
            autocorrelation = numerator / denominator
    return {
        "steering_anomaly_timesteps": anomaly_timesteps,
        "max_steer_delta": round(max_steer_delta, 4),
        "max_steer_reversals": int(max_reversals),
        "steer_autocorr_lag1": round(float(autocorrelation), 4),
    }

def wrapped_progress_difference(ego_progress, opp_progress, track_length):
    """Signed progress difference wrapped to [-track_length/2, track_length/2)"""
    return (ego_progress - opp_progress + 0.5 * track_length) % track_length - 0.5 * track_length

def aggregate_multiagent_batch(results_path, temporary_directory, total_segments):
    """Merge worker-local metrics and exit codes into results_multi.json once."""
    counts = {"following": 0, "overtaking": 0, "collision": 0, "error": 0}
    categories = {1: "following", 2: "overtaking", 3: "collision"}
    batch_episodes = {}
    batch_metrics = []
    for exit_path in sorted(Path(temporary_directory).glob("*.exit")):
        try:
            exit_code = int(exit_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            exit_code = -1
        counts[categories.get(exit_code, "error")] += 1
        if exit_code not in categories:
            continue
        metrics_path = exit_path.with_suffix(".metrics.json")
        try:
            metrics = load_json(metrics_path, None)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            metrics = None
        if not isinstance(metrics, dict) or not metrics.get("episode_key"):
            continue
        key = str(metrics["episode_key"])
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
    atomic_write_json(
        destination,
        {"final": final, "episodes": dict(sorted(episodes.items()))},
    )
    return final

def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate multi-agent batch evaluation results")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--map-name", required=True)
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument("--temp-dir", required=True)
    parser.add_argument("--total-segments", type=int, required=True)
    args = parser.parse_args()
    paths = multiagent_paths(args.model_path, args.map_name, args.noise)
    aggregate_multiagent_batch(paths["results"], args.temp_dir, args.total_segments)

if __name__ == "__main__":
    _main()
