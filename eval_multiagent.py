import argparse
import json
import sys
import warnings
import gym
import numpy as np
import torch
import os
import gc
import imageio
from f110_gym.envs.base_classes import Integrator
import f110_gym.envs.f110_env as f110_env

from model import End2Race
from latticeplanner.utils import project_point_to_centerline, obsDict2oppoArray
from demonstration import setup_opp_planner
from utils import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate model on segment with opponent')
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default='pretrained/end2race.pth')
    parser.add_argument("--hidden_scale", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.0)
    
    # Segment parameters
    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--ego_idx", type=int, default=0)
    parser.add_argument("--interval_idx", type=int, default=15)
    parser.add_argument("--ego_raceline", type=str, default="raceline1")
    parser.add_argument("--opp_raceline", type=str, default="raceline0")
    parser.add_argument("--opp_speedscale", type=float, default=0.5)
    parser.add_argument("--sim_duration", type=float, default=8.0)
    parser.add_argument("--render", action='store_true')

    return parser.parse_args()

def evaluate_segment(model, device, noise_level, map_name, ego_idx, interval_idx, 
                    ego_raceline, opp_raceline, opp_speed_scale, sim_duration,
                    render=False, model_path='pretrained/end2race.pth'):
    """Evaluate a single segment with model against lattice planner opponent"""
    
    np.random.seed(42)
    num_features = 360
    sim_timestep = 0.01
    result_dir = get_eval_results_dir(model_path, map_name, noise_level)
    
    # Calculate opponent index using same logic as run_lattice_planner.py
    params = {
        'ego_raceline': ego_raceline,
        'opp_raceline': opp_raceline,
        'ego_idx': ego_idx,
        'opp_idx': 0  # Will be calculated below
    }
    
    # Load waypoints to calculate opp_idx
    base_path = f"f1tenth_racetracks/{map_name}"
    ego_path = os.path.join(base_path, f"{ego_raceline}.csv")
    with open(ego_path, 'r') as f:
        lines = f.readlines()[1:]
    ego_waypoints = []
    for line in lines:
        parts = line.strip().split(';')
        if len(parts) >= 6:
            ego_waypoints.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[5])])
    ego_waypoints = np.array(ego_waypoints)
    
    if opp_raceline != ego_raceline:
        opp_path = os.path.join(base_path, f"{opp_raceline}.csv")
        with open(opp_path, 'r') as f:
            lines = f.readlines()[1:]
        opp_waypoints = []
        for line in lines:
            parts = line.strip().split(';')
            if len(parts) >= 6:
                opp_waypoints.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[5])])
        opp_waypoints = np.array(opp_waypoints)
        
        ego_waypoint = ego_waypoints[ego_idx % len(ego_waypoints)]
        ego_map_idx = find_corresponding_waypoint(ego_waypoint, opp_waypoints)
        opp_idx = (ego_map_idx + interval_idx) % len(opp_waypoints)
    else:
        opp_idx = (ego_idx + interval_idx) % len(ego_waypoints)
    
    params['opp_idx'] = opp_idx
    
    # Setup environment
    env = gym.make("f110-v0", map=f"f1tenth_racetracks/{map_name}/{map_name}_map", map_ext=".png", num_agents=2, timestep=sim_timestep, integrator=Integrator.RK4)
    
    # Add render callback for proper camera positioning and trajectory visualization
    if render:
        # Initialize render info and trajectory tracking
        render_info = {"ego_speed": 0.0, "ego_steer": 0.0, "opp_speed": 0.0, "opp_steer": 0.0, "lap_time": 0.0, "state": "unknown"}
        visited_points = [[], []]  # [ego_points, opp_points]
        drawn_points = [[], []]    # [ego_drawn, opp_drawn]
        batch_objects = []

        render_callback = create_multiagent_render_callback(
            render_info,
            visited_points,
            drawn_points,
            batch_objects
        )

        env.add_render_callback(render_callback)
    else:
        render_info = None
        visited_points = None
        batch_objects = []
    
    # Initialize video frames list if rendering
    video_frames = []
    # Load positions using function from utils
    positions, initial_speeds = load_positions_and_speeds_from_params(params, map_name)
    # Initialize opponent planner using function from run_lattice_planner.py
    opponent = setup_opp_planner(map_name, opp_raceline)
    tracker_steps = 10  # Default tracker steps
    
    # Initialize model state
    hidden_size = model.gru.hidden_size
    hidden_state = torch.zeros((1, 1, hidden_size), device=device)
    prev_speed = initial_speeds[0] * 0.9
    
    # Load centerline
    centerline_path = f"f1tenth_racetracks/{map_name}/raceline1.csv"
    centerline_wp = np.loadtxt(centerline_path, delimiter=';', skiprows=1)
    centerline = np.vstack((centerline_wp[:, 1], centerline_wp[:, 2])).T
    centerline_total_length = sum(np.linalg.norm(centerline[i+1] - centerline[i]) for i in range(len(centerline)-1))
    
    # Reset environment
    obs, _, done, _ = env.reset(poses=positions)
    
    # Only initialize rendering if render flag is set
    if render:
        env.render()
    
    # Track initial state
    initial_ego_progress, _ = project_point_to_centerline(np.array([obs['poses_x'][0], obs['poses_y'][0]]), centerline)
    initial_opp_progress, _ = project_point_to_centerline(np.array([obs['poses_x'][1], obs['poses_y'][1]]), centerline)
    initial_state = "overtaking" if initial_ego_progress > initial_opp_progress else "following"
    
    # Simulation metrics
    lap_time = 0.0
    collision_occurred = False
    final_state = initial_state
    ego_trajectory = []
    speeds = []
    tracker_count = 0
    opp_traj = None
    validation_steers = []
    validation_lidars = []
    
    # Main simulation loop
    while not done and lap_time < sim_duration:
        # Model inference for ego
        lidar_360 = np.array(obs["scans"][0]).flatten()
        if len(lidar_360) != num_features:
            indices = np.linspace(0, len(lidar_360)-1, num_features, dtype=int)
            lidar_360 = lidar_360[indices]
        lidar = lidar_360.copy()
        
        # Apply noise
        if noise_level > 0:
            num_points_to_mask = int(len(lidar) * noise_level)
            if num_points_to_mask > 0:
                mask_indices = np.random.choice(len(lidar), min(num_points_to_mask, len(lidar)), replace=False)
                lidar[mask_indices] = 0.0
        
        with torch.no_grad():
            lidar_tensor = torch.tensor(lidar, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            speed_tensor = torch.tensor([[[prev_speed]]], dtype=torch.float32, device=device)
            action_sequence, hidden_state = model(lidar_tensor, speed_tensor, hidden_state)
            
            action_tensor = action_sequence[:, -1, :]
            ego_steer = action_tensor[0, 0].item()
            ego_speed = action_tensor[0, 1].item()
        
        ego_steer = np.clip(ego_steer, -0.52, 0.52)
        validation_steers.append(float(ego_steer))
        validation_lidars.append(lidar_360.copy())
        prev_speed = obs['linear_vels_x'][0]
        
        # Opponent lattice planner
        if tracker_count == 0:
            opp_poses = obsDict2oppoArray(obs, 1)
            opp_traj = opponent.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], opp_poses, obs['linear_vels_x'][1])
        
        opp_steer, opp_speed = opponent.tracker.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], obs['linear_vels_x'][1], opp_traj)
        
        opp_steer = np.clip(opp_steer, -0.52, 0.52)
        opp_speed *= opp_speed_scale
        
        # Update render info and trajectory tracking
        if render:
            render_info.update({
                'ego_speed': ego_speed,
                'ego_steer': ego_steer,
                'opp_speed': opp_speed,
                'opp_steer': opp_steer,
                'state': final_state
            })
            
            # Add current positions to trajectory
            visited_points[0].append([obs['poses_x'][0], obs['poses_y'][0]])  # Ego trajectory
            visited_points[1].append([obs['poses_x'][1], obs['poses_y'][1]])  # Opponent trajectory
        
        # Step environment
        action = np.array([[ego_steer, ego_speed], [opp_steer, opp_speed]])
        obs, timestep, done, _ = env.step(action)
        lap_time += timestep
        
        # Capture video frame if rendering
        if render:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                video_frames.append(frame)
        
        # Track metrics
        ego_trajectory.append([obs['poses_x'][0], obs['poses_y'][0]])
        speeds.append(obs['linear_vels_x'][0])
        
        # Update state tracking
        ego_progress, _ = project_point_to_centerline(np.array([obs['poses_x'][0], obs['poses_y'][0]]), centerline)
        opp_progress, _ = project_point_to_centerline(np.array([obs['poses_x'][1], obs['poses_y'][1]]), centerline)
        
        # Handle lap wrapping
        if ego_progress < initial_ego_progress - centerline_total_length/2:
            ego_progress += centerline_total_length
        if opp_progress < initial_opp_progress - centerline_total_length/2:
            opp_progress += centerline_total_length
        
        final_state = "overtaking" if ego_progress > opp_progress else "following"
        
        # Check collision
        if np.any(obs['collisions']):
            collision_occurred = True
            done = True
        
        tracker_count = (tracker_count + 1) % tracker_steps
    
    # Save video if rendering was enabled
    if render and video_frames:
        # Generate video filename
        if collision_occurred:
            state_prefix = "c"  # 'c' for collision
        else:
            state_prefix = "o" if final_state == "overtaking" else "f"
        episode_key = multi_episode_key(opp_raceline, ego_idx, opp_idx, opp_speed_scale)
        video_filename = f"{state_prefix}_{episode_key}.mp4"
        
        os.makedirs(result_dir, exist_ok=True)
        video_path = os.path.join(result_dir, video_filename)
        
        imageio.mimwrite(video_path, video_frames, fps=100, macro_block_size=1)
        print(f"Video saved to {video_path}")
    
    # Clean up visualization objects
    if render:
        for batch_obj in batch_objects:
            batch_obj.delete()
        env.render_callbacks = []
    
    env.close()
    gc.collect()
    
    # Calculate final metrics
    avg_speed, speed_variance, total_distance = calculate_metrics(ego_trajectory, speeds)
    proximity_quality = evaluate_proximity_quality(np.array(validation_lidars))
    steering_quality = evaluate_steering_quality(np.array(validation_steers), sample_interval=sim_timestep)

    # Determine final state number
    if collision_occurred:
        final_state_num = 3
    elif final_state == "overtaking":
        final_state_num = 2
    else:
        final_state_num = 1
    
    state_label = "collision" if collision_occurred else final_state

    return {
        'episode_key': multi_episode_key(opp_raceline, ego_idx, opp_idx, opp_speed_scale),
        'state': final_state_num,
        'state_label': state_label,
        'avg_speed': float(avg_speed) if not collision_occurred else 0,
        'speed_variance': float(speed_variance) if not collision_occurred else 0,
        'total_distance': float(total_distance),
        'collision_occurred': bool(collision_occurred),
        'global_min_surface_dist': proximity_quality['global_min_surface_dist'],
        'danger_sectors': proximity_quality['danger_sectors'],
        'proximity_below_threshold_timesteps': proximity_quality['proximity_below_threshold_timesteps'],
        'steering_anomaly_timesteps': steering_quality['steering_anomaly_timesteps'],
        'max_steer_delta': steering_quality['max_steer_delta'],
        'max_steer_reversals': steering_quality['max_steer_reversals'],
        'steer_autocorr_lag1': steering_quality['steer_autocorr_lag1'],
    }

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set device - prefer CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = End2Race(hidden_scale=args.hidden_scale).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Run evaluation
    result = evaluate_segment(
        model, device, args.noise,
        args.map_name, args.ego_idx, args.interval_idx,
        args.ego_raceline, args.opp_raceline, args.opp_speedscale,
        args.sim_duration, args.render, args.model_path
    )

    # Print metrics to stdout (terminal for standalone runs; parsed from logs by evaluate.sh)
    print(f"EPISODE_KEY={result['episode_key']}")
    print(f"STATE={result['state']}")
    print(f"STATE_LABEL={result['state_label']}")
    print(f"COLLISION_OCCURRED={str(result['collision_occurred']).lower()}")
    print(f"AVG_SPEED={result['avg_speed']}")
    print(f"SPEED_VARIANCE={result['speed_variance']}")
    print(f"TOTAL_DISTANCE={result['total_distance']}")
    print(f"GLOBAL_MIN_SURFACE_DIST={result['global_min_surface_dist']}")
    print(f"DANGER_SECTORS={json.dumps(result['danger_sectors'], sort_keys=True)}")
    print(f"PROXIMITY_BELOW_THRESHOLD_TIMESTEPS={json.dumps(result['proximity_below_threshold_timesteps'])}")
    print(f"STEERING_ANOMALY_TIMESTEPS={json.dumps(result['steering_anomaly_timesteps'])}")
    print(f"MAX_STEER_DELTA={result['max_steer_delta']}")
    print(f"MAX_STEER_REVERSALS={result['max_steer_reversals']}")
    print(f"STEER_AUTOCORR_LAG1={result['steer_autocorr_lag1']}")

    # Exit with state code
    sys.exit(result['state'])
