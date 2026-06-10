import os
import json
import numpy as np

def get_eval_results_dir(model_path, map_name, noise_level):
    """Return the shared eval_results directory for a model/map/noise setting."""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    parts = [model_name, map_name]
    if noise_level > 0:
        parts.append(f"noise{int(noise_level * 100)}")
    return os.path.join("eval_results", "_".join(parts))

def load_json_file(path):
    """Load a JSON object from path, returning an empty dict if it does not exist."""
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def write_json_file(path, data):
    """Atomically write data as indented JSON, creating parent directories as needed."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)

def write_json_entry(path, key, value):
    """Insert or replace a single keyed entry in the JSON object stored at path."""
    data = load_json_file(path)
    data[key] = value
    write_json_file(path, data)

def multi_episode_key(opp_raceline, ego_idx, opp_idx, opp_speed_scale):
    """Build the canonical episode key for a multiagent evaluation segment."""
    opp_raceline_num = opp_raceline.replace('raceline', '')
    return f"ol{opp_raceline_num}_e{ego_idx}_o{opp_idx}_s{opp_speed_scale}"

def _multi_episode_sort_key(key):
    parts = key.split('_')
    if len(parts) != 4:
        return (float('inf'), float('inf'), float('inf'), float('inf'), key)
    try:
        return (
            int(parts[0].replace('ol', '')),
            int(parts[1].replace('e', '')),
            int(parts[2].replace('o', '')),
            float(parts[3].replace('s', '')),
            key
        )
    except ValueError:
        return (float('inf'), float('inf'), float('inf'), float('inf'), key)

def _parse_key_value_log(path):
    values = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            values[key] = value
    return values

def _parse_multi_episode_log(path):
    values = _parse_key_value_log(path)
    episode_key = values.get('EPISODE_KEY')
    if not episode_key:
        return None
    try:
        return episode_key, {
            'state': int(values['STATE']),
            'state_label': values.get('STATE_LABEL', 'unknown'),
            'avg_speed': float(values['AVG_SPEED']),
            'speed_variance': float(values['SPEED_VARIANCE']),
            'total_distance': float(values['TOTAL_DISTANCE']),
            'collision_occurred': values.get('COLLISION_OCCURRED', '').lower() == 'true',
            'global_min_surface_dist': float(values['GLOBAL_MIN_SURFACE_DIST']),
            'danger_sectors': json.loads(values.get('DANGER_SECTORS', '{}')),
            'proximity_below_threshold_timesteps': json.loads(
                values.get('PROXIMITY_BELOW_THRESHOLD_TIMESTEPS', '[]')
            ),
            'steering_anomaly_timesteps': json.loads(values.get('STEERING_ANOMALY_TIMESTEPS', '[]')),
            'max_steer_delta': float(values['MAX_STEER_DELTA']),
            'max_steer_reversals': int(values['MAX_STEER_REVERSALS']),
            'steer_autocorr_lag1': float(values['STEER_AUTOCORR_LAG1']),
        }
    except (KeyError, ValueError, json.JSONDecodeError):
        return None

def _percentage(count, total):
    if total == 0:
        return 0
    value = round(count * 100.0 / total, 1)
    return int(value) if value.is_integer() else value

def _mean_metric(metrics, name):
    values = [
        metric[name]
        for metric in metrics
        if isinstance(metric.get(name), (int, float))
    ]
    return round(sum(values) / len(values), 6) if values else 0

def write_multiagent_results_from_logs(
    model_path,
    map_name,
    noise_level,
    log_dir,
    total_episodes,
    following_count,
    overtaking_count,
    collision_count,
    error_count,
):
    """Merge multiagent episode logs and write results_multi.json."""
    result_path = os.path.join(
        get_eval_results_dir(model_path, map_name, noise_level),
        'results_multi.json'
    )
    data = load_json_file(result_path)
    episodes = data.get('episodes', {})
    if not isinstance(episodes, dict):
        episodes = {}

    batch_metrics = []
    for filename in sorted(os.listdir(log_dir)):
        if not filename.endswith('.log'):
            continue
        parsed = _parse_multi_episode_log(os.path.join(log_dir, filename))
        if parsed is None:
            continue
        episode_key, metric = parsed
        episodes[episode_key] = metric
        batch_metrics.append(metric)

    success_count = following_count + overtaking_count
    ordered_episodes = {
        key: episodes[key]
        for key in sorted(episodes, key=_multi_episode_sort_key)
    }
    final = {
        'total_episodes': total_episodes,
        'following_count': following_count,
        'overtaking_count': overtaking_count,
        'success_count': success_count,
        'collision_count': collision_count,
        'error_count': error_count,
        'following_rate': _percentage(following_count, total_episodes),
        'overtaking_rate': _percentage(overtaking_count, total_episodes),
        'success_rate': _percentage(success_count, total_episodes),
        'collision_rate': _percentage(collision_count, total_episodes),
        'avg_speed_mean': _mean_metric(batch_metrics, 'avg_speed'),
        'speed_variance_mean': _mean_metric(batch_metrics, 'speed_variance'),
        'total_distance_mean': _mean_metric(batch_metrics, 'total_distance'),
    }
    write_json_file(result_path, {
        'final': final,
        'episodes': ordered_episodes,
    })
    return result_path

def write_multiagent_results_from_logs_cli():
    """CLI bridge used by evaluate.sh to keep bash aggregation short."""
    import sys

    result_path = write_multiagent_results_from_logs(
        sys.argv[1],
        sys.argv[2],
        float(sys.argv[3]),
        sys.argv[4],
        int(sys.argv[5]),
        int(sys.argv[6]),
        int(sys.argv[7]),
        int(sys.argv[8]),
        int(sys.argv[9]),
    )
    print(result_path)

def load_raceline_with_speed(map_name, raceline_file, start_idx):
    """Load raceline waypoints with position and speed information"""
    raceline_path = f"f1tenth_racetracks/{map_name}/{raceline_file}"
    with open(raceline_path, 'r') as f:
        lines = f.readlines()[1:]
    
    waypoints = []
    for line in lines:
        parts = line.strip().split(';')
        if len(parts) >= 6:
            waypoints.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[5])])
    waypoints = np.array(waypoints)
    
    # Get starting position and speed
    idx = start_idx % len(waypoints)
    start_pose = np.array([[waypoints[idx, 0], waypoints[idx, 1], waypoints[idx, 2]]])
    initial_speed = waypoints[idx, 3]
    
    return start_pose, initial_speed, waypoints

def calculate_metrics(trajectory, speeds):
    """Calculate performance metrics"""
    avg_speed = np.mean(speeds) if speeds else 0
    speed_variance = np.var(speeds) if speeds else 0
    total_distance = sum(np.linalg.norm(np.array(trajectory[i+1]) - np.array(trajectory[i]))
                        for i in range(len(trajectory)-1)) if len(trajectory) > 1 else 0
    return avg_speed, speed_variance, total_distance

PROXIMITY_THRESHOLD = 0.15
STEER_WINDOW_SECONDS = 1.0
STEER_MAX_REVERSALS = 6
STEER_MIN_AMP = 0.3
STEER_MAX_JUMP = 0.6

LIDAR_SECTORS = [
    ('rear', list(range(0, 30)) + list(range(330, 360))),
    ('rear_right', list(range(30, 60))),
    ('right', list(range(60, 120))),
    ('front_right', list(range(120, 150))),
    ('front', list(range(150, 210))),
    ('front_left', list(range(210, 240))),
    ('left', list(range(240, 300))),
    ('rear_left', list(range(300, 330))),
]

def _lidar_edge_distance(n_beams):
    half_length = 0.58 / 2
    half_width = 0.31 / 2
    angles = -np.pi + np.arange(n_beams) * (2 * np.pi / n_beams)
    cos_abs = np.abs(np.cos(angles)) + 1e-12
    sin_abs = np.abs(np.sin(angles)) + 1e-12
    return np.minimum(half_length / cos_abs, half_width / sin_abs)

def _surface_distance_from_lidar(lidar):
    lidar = np.asarray(lidar, dtype=float)
    if lidar.ndim == 1:
        lidar = lidar.reshape(1, -1)
    return np.clip(lidar - _lidar_edge_distance(lidar.shape[1]), 0.0, None)

def evaluate_proximity_quality(lidar):
    """Return distance quality metrics from a [T, N] lidar sequence."""
    surface_dist = _surface_distance_from_lidar(lidar)
    min_dist_per_step = np.min(surface_dist, axis=1)
    danger_sectors = {}
    for name, indices in LIDAR_SECTORS:
        sector_min = float(np.min(surface_dist[:, indices]))
        if sector_min < PROXIMITY_THRESHOLD:
            danger_sectors[name] = round(sector_min, 4)

    return {
        'global_min_surface_dist': round(float(np.min(surface_dist)), 4),
        'danger_sectors': danger_sectors,
        'proximity_below_threshold_timesteps': np.flatnonzero(
            min_dist_per_step < PROXIMITY_THRESHOLD
        ).astype(int).tolist(),
    }

def _steer_window_frames(sample_interval):
    return max(1, int(round(STEER_WINDOW_SECONDS / sample_interval)))

def _large_steer_delta_signs(steer):
    delta = np.diff(steer)
    return np.where(np.abs(delta) >= STEER_MIN_AMP, np.sign(delta), 0.0)

def _steer_reversal_timesteps(steer):
    signs = _large_steer_delta_signs(steer)
    timesteps = []
    prev_sign = 0.0
    for delta_idx, sign in enumerate(signs):
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            timesteps.append(delta_idx + 1)
        prev_sign = sign
    return timesteps

def _max_steer_reversals(steer, window_size):
    reversal_timesteps = _steer_reversal_timesteps(steer)
    max_reversals = 0
    for start in range(max(1, len(steer) - window_size + 1)):
        end = start + window_size
        count = sum(start <= step < end for step in reversal_timesteps)
        max_reversals = max(max_reversals, count)
    return max_reversals

def _steer_jump_timesteps(steer):
    if len(steer) < 2:
        return []
    return (np.flatnonzero(np.abs(np.diff(steer)) > STEER_MAX_JUMP) + 1).astype(int).tolist()

def _steer_oscillation_timesteps(steer, window_size):
    reversal_timesteps = _steer_reversal_timesteps(steer)
    if not reversal_timesteps:
        return []

    timesteps = set()
    for start in range(max(1, len(steer) - window_size + 1)):
        end = min(len(steer), start + window_size)
        count = sum(start <= step < end for step in reversal_timesteps)
        if count > STEER_MAX_REVERSALS:
            timesteps.update(range(start, end))
    return sorted(timesteps)

def _steer_autocorr_lag1(steer):
    if len(steer) < 2:
        return 1.0
    centered = steer - np.mean(steer)
    norm = float(np.sum(centered ** 2))
    if norm == 0.0:
        return 1.0
    return float(np.sum(centered[:-1] * centered[1:]) / norm)

def evaluate_steering_quality(steer, sample_interval):
    """Return steering quality metrics from one uniformly sampled steering sequence."""
    steer = np.asarray(steer, dtype=float)
    window_size = _steer_window_frames(sample_interval)
    jump_timesteps = _steer_jump_timesteps(steer)
    oscillation_timesteps = _steer_oscillation_timesteps(steer, window_size)
    max_delta = float(np.max(np.abs(np.diff(steer)))) if len(steer) >= 2 else 0.0

    return {
        'steering_anomaly_timesteps': sorted(set(jump_timesteps + oscillation_timesteps)),
        'max_steer_delta': round(max_delta, 4),
        'max_steer_reversals': _max_steer_reversals(steer, window_size),
        'steer_autocorr_lag1': round(_steer_autocorr_lag1(steer), 4),
    }

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
    base_path = f"f1tenth_racetracks/{map_name}"
    
    # Load ego raceline with speed
    ego_path = os.path.join(base_path, params['ego_raceline'] + '.csv')
    with open(ego_path, 'r') as f:
        lines = f.readlines()[1:]
    ego_waypoints = []
    for line in lines:
        parts = line.strip().split(';')
        if len(parts) >= 6:
            ego_waypoints.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[5])])
    ego_waypoints = np.array(ego_waypoints)
    
    # Load opponent raceline with speed
    opp_path = os.path.join(base_path, params['opp_raceline'] + '.csv')
    if params['opp_raceline'] != params['ego_raceline']:
        with open(opp_path, 'r') as f:
            lines = f.readlines()[1:]
        opp_waypoints = []
        for line in lines:
            parts = line.strip().split(';')
            if len(parts) >= 6:
                opp_waypoints.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[5])])
        opp_waypoints = np.array(opp_waypoints)
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
