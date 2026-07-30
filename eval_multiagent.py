import argparse
import sys
import gym
import numpy as np
import torch
import os
import gc
import imageio
from f110_gym.envs.base_classes import Integrator
from model import End2Race
from latticeplanner.utils import project_point_to_centerline, obsDict2oppoArray
from demonstration import setup_opp_planner
from ppo.reward import ProgressProjector
from utils import *

def collision_scope_stops_episode(collision_scope, ego_collision, opponent_collision):
    if collision_scope == "legacy":
        return bool(ego_collision or opponent_collision)
    if collision_scope == "ego":
        return bool(ego_collision)
    raise ValueError(f"Unknown collision scope: {collision_scope}")

def classify_collision(env, collisions):
    ego_opp = bool(env.unwrapped.sim.collision_idx[0] == 1)
    ego_wall = bool(collisions[0] and not ego_opp)
    return (
        ego_opp,
        ego_wall,
        bool(collisions[1] and not ego_opp and not ego_wall),
    )

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
    parser.add_argument("--opp_raceline", type=str, default="raceline1")
    parser.add_argument("--opp_speedscale", type=float, default=0.5)
    parser.add_argument("--sim_duration", type=float, default=8.0)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--save_trace", action="store_true")
    parser.add_argument("--metrics_out", type=str, default=None)
    parser.add_argument("--collision_scope", choices=("legacy", "ego"), default="legacy")
    
    return parser.parse_args()

def evaluate_segment(model, device, noise_level, map_name, ego_idx, interval_idx,
                    ego_raceline, opp_raceline, opp_speed_scale, sim_duration,
                    render=False, save_trace=False,
                    model_path="pretrained/end2race.pth", metrics_out=None,
                    collision_scope="legacy"):
    """Evaluate a single segment with model against lattice planner opponent"""
    
    np.random.seed(42)
    num_features = 360

    opp_idx = get_opponent_startpoint(
        map_name,
        ego_raceline,
        opp_raceline,
        ego_idx,
        interval_idx,
    )

    params = {'ego_raceline': ego_raceline, 'opp_raceline': opp_raceline, 'ego_idx': ego_idx, 'opp_idx': opp_idx}
    key = episode_key(opp_raceline, ego_idx, opp_idx, opp_speed_scale)
    output_paths = multiagent_paths(model_path, map_name, noise_level, key)
    
    # Setup environment
    env = gym.make("f110-v0", map=f"f1tenth_racetracks/{map_name}/{map_name}_map", map_ext=".png", num_agents=2, timestep=0.01, integrator=Integrator.RK4)
    
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
    if collision_scope == "ego":
        progress_projector = ProgressProjector.from_csv(centerline_path)
        centerline_total_length = progress_projector.track_length

        def project_progress(point):
            return progress_projector.project(point)
    elif collision_scope == "legacy":
        centerline_total_length = sum(np.linalg.norm(centerline[i+1] - centerline[i]) for i in range(len(centerline)-1))

        def project_progress(point):
            progress, _ = project_point_to_centerline(point, centerline)
            return progress
    else:
        raise ValueError(f"Unknown collision scope: {collision_scope}")
    
    # Reset environment
    obs, _, done, _ = env.reset(poses=positions)
    observation_finite = bool(
        all(
            np.isfinite(np.asarray(value)).all()
            for value in obs.values()
            if isinstance(value, (list, tuple, np.ndarray))
        )
    )
    action_finite = True
    
    # Only initialize rendering if render flag is set
    if render:
        env.render()
    
    # Track initial state
    initial_ego_progress = project_progress(np.array([obs['poses_x'][0], obs['poses_y'][0]]))
    initial_opp_progress = project_progress(np.array([obs['poses_x'][1], obs['poses_y'][1]]))
    initial_relative = wrapped_progress_difference(
        initial_ego_progress,
        initial_opp_progress,
        centerline_total_length,
    )
    previous_relative = initial_relative
    relative_unwrapped = initial_relative
    initial_state = "overtaking" if relative_unwrapped > 0.0 else "following"
    
    # Simulation metrics
    lap_time = 0.0
    collision_occurred = False
    collision_type = None
    ego_collision_time_s = None
    step_count = 0
    final_state = initial_state
    ego_trajectory = []
    speeds = []
    tracker_count = 0
    opp_traj = None
    ego_desired_speeds = []
    ego_lidar_minima = []
    opp_speeds = []
    opp_desired_speeds = []
    opp_lidar_minima = []
    ego_raw_lidar_history = []
    trace = None
    if save_trace:
        trace = {
            "time_s": [],
            "ego_lidar_360": [],
            "opp_lidar_360": [],
            "ego_raw_action": [],
            "ego_executed_action": [],
            "opp_executed_action": [],
            "ego_measured_speed_mps": [],
            "opp_measured_speed_mps": [],
            "ego_pose": [],
            "opp_pose": [],
            "collisions": [],
            "ego_opp_collision": [],
            "ego_wall_collision": [],
            "opp_wall_collision": [],
            "action_applied": [],
            "terminal_post_step": [],
        }
    
    # Main simulation loop
    while not done and lap_time < sim_duration:
        # Model inference for ego
        raw_lidar_360 = np.array(obs["scans"][0]).flatten()
        if len(raw_lidar_360) > num_features:
            indices = np.linspace(0, len(raw_lidar_360)-1, num_features, dtype=int)
            raw_lidar_360 = raw_lidar_360[indices]
        lidar = raw_lidar_360.copy()
        
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

        ego_raw_action = np.array([ego_steer, ego_speed], dtype=np.float32)
        ego_steer = np.clip(ego_steer, -0.52, 0.52)
        ego_executed_action = np.array([ego_steer, ego_speed], dtype=np.float32)
        action_finite = action_finite and bool(np.isfinite(ego_executed_action).all())
        prev_speed = obs['linear_vels_x'][0]
        
        # Opponent lattice planner
        if tracker_count == 0:
            opp_poses = obsDict2oppoArray(obs, 1)
            opp_traj = opponent.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], opp_poses, obs['linear_vels_x'][1])
        
        opp_steer, opp_speed = opponent.tracker.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], obs['linear_vels_x'][1], opp_traj)
        
        opp_steer = np.clip(opp_steer, -0.52, 0.52)
        opp_speed *= opp_speed_scale
        opp_executed_action = np.array([opp_steer, opp_speed], dtype=np.float32)
        action_finite = action_finite and bool(np.isfinite(opp_executed_action).all())

        opp_lidar = np.array(obs["scans"][1]).flatten()
        if len(opp_lidar) > num_features:
            indices = np.linspace(0, len(opp_lidar)-1, num_features, dtype=int)
            opp_lidar = opp_lidar[indices]

        ego_desired_speeds.append(float(ego_speed))
        ego_lidar_minima.append(float(np.min(raw_lidar_360)))
        ego_raw_lidar_history.append(raw_lidar_360.copy())
        opp_desired_speeds.append(float(opp_speed))
        opp_lidar_minima.append(float(np.min(opp_lidar)))
        if trace is not None:
            current_collisions = np.asarray(obs["collisions"], dtype=bool)
            current_ego_opp, current_ego_wall, current_opp_wall = classify_collision(
                env, current_collisions
            )
            trace["time_s"].append(float(lap_time))
            trace["ego_lidar_360"].append(lidar)
            trace["opp_lidar_360"].append(opp_lidar)
            trace["ego_raw_action"].append(ego_raw_action)
            trace["ego_executed_action"].append(ego_executed_action)
            trace["opp_executed_action"].append(opp_executed_action)
            trace["ego_measured_speed_mps"].append(float(obs['linear_vels_x'][0]))
            trace["opp_measured_speed_mps"].append(float(obs['linear_vels_x'][1]))
            trace["ego_pose"].append([obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]])
            trace["opp_pose"].append([obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1]])
            trace["collisions"].append(current_collisions)
            trace["ego_opp_collision"].append(current_ego_opp)
            trace["ego_wall_collision"].append(current_ego_wall)
            trace["opp_wall_collision"].append(current_opp_wall)
            trace["action_applied"].append(True)
            trace["terminal_post_step"].append(False)
        
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
        step_count += 1
        observation_finite = observation_finite and bool(
            all(
                np.isfinite(np.asarray(value)).all()
                for value in obs.values()
                if isinstance(value, (list, tuple, np.ndarray))
            )
        )
        
        # Capture video frame if rendering
        if render:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                video_frames.append(frame)
        
        # Track metrics
        ego_trajectory.append([obs['poses_x'][0], obs['poses_y'][0]])
        speeds.append(obs['linear_vels_x'][0])
        opp_speeds.append(obs['linear_vels_x'][1])
        
        # Update state tracking
        ego_progress = project_progress(np.array([obs['poses_x'][0], obs['poses_y'][0]]))
        opp_progress = project_progress(np.array([obs['poses_x'][1], obs['poses_y'][1]]))
        
        current_relative = wrapped_progress_difference(
            ego_progress,
            opp_progress,
            centerline_total_length,
        )
        relative_delta = wrapped_progress_difference(
            current_relative,
            previous_relative,
            centerline_total_length,
        )
        relative_unwrapped += relative_delta
        previous_relative = current_relative
        final_state = "overtaking" if relative_unwrapped > 0.0 else "following"
        
        # Check collision
        step_collisions = np.asarray(obs["collisions"], dtype=bool)
        step_ego_collision = bool(step_collisions[0])
        step_opp_collision = bool(step_collisions[1])
        step_ego_opp, step_ego_wall, step_opp_wall = classify_collision(
            env, step_collisions
        )
        if step_ego_collision and ego_collision_time_s is None:
            ego_collision_time_s = float(lap_time)
        if collision_scope_stops_episode(collision_scope, step_ego_collision, step_opp_collision):
            collision_occurred = True
            if step_ego_opp:
                collision_type = "ego-opp"
            if step_ego_wall:
                collision_type = "ego-wall"
            if step_opp_wall:
                collision_type = "opp-wall"
            done = True

        if trace is not None and (done or lap_time >= sim_duration):
            terminal_raw_lidar = np.asarray(obs["scans"][0]).reshape(-1)
            if len(terminal_raw_lidar) > num_features:
                indices = np.linspace(0, len(terminal_raw_lidar) - 1, num_features, dtype=int)
                terminal_raw_lidar = terminal_raw_lidar[indices]
            terminal_lidar = terminal_raw_lidar.copy()
            if noise_level > 0:
                num_points_to_mask = int(len(terminal_lidar) * noise_level)
                if num_points_to_mask > 0:
                    mask_indices = np.random.choice(
                        len(terminal_lidar),
                        min(num_points_to_mask, len(terminal_lidar)),
                        replace=False,
                    )
                    terminal_lidar[mask_indices] = 0.0

            terminal_opp_lidar = np.asarray(obs["scans"][1]).reshape(-1)
            if len(terminal_opp_lidar) > num_features:
                indices = np.linspace(0, len(terminal_opp_lidar) - 1, num_features, dtype=int)
                terminal_opp_lidar = terminal_opp_lidar[indices]

            trace["time_s"].append(float(lap_time))
            trace["ego_lidar_360"].append(terminal_lidar)
            trace["opp_lidar_360"].append(terminal_opp_lidar)
            trace["ego_raw_action"].append(np.zeros(2, dtype=np.float32))
            trace["ego_executed_action"].append(np.zeros(2, dtype=np.float32))
            trace["opp_executed_action"].append(np.zeros(2, dtype=np.float32))
            trace["ego_measured_speed_mps"].append(float(obs['linear_vels_x'][0]))
            trace["opp_measured_speed_mps"].append(float(obs['linear_vels_x'][1]))
            trace["ego_pose"].append([obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]])
            trace["opp_pose"].append([obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1]])
            trace["collisions"].append(step_collisions)
            trace["ego_opp_collision"].append(step_ego_opp)
            trace["ego_wall_collision"].append(step_ego_wall)
            trace["opp_wall_collision"].append(step_opp_wall)
            trace["action_applied"].append(False)
            trace["terminal_post_step"].append(True)
        
        tracker_count = (tracker_count + 1) % tracker_steps
    
    # Save video if rendering was enabled
    if render and video_frames:
        state_prefix = (
            {"ego-opp": "eoc", "ego-wall": "ewc", "opp-wall": "owc"}[collision_type]
            if collision_occurred
            else ("o" if final_state == "overtaking" else "f")
        )
        video_path = multiagent_paths(model_path, map_name, noise_level, key, state_prefix)["video"]
        os.makedirs(video_path.parent, exist_ok=True)
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

    if trace is not None:
        dtypes = {
            "time_s": np.float64,
            "ego_pose": np.float64,
            "opp_pose": np.float64,
            "collisions": np.bool_,
            "ego_opp_collision": np.bool_,
            "ego_wall_collision": np.bool_,
            "opp_wall_collision": np.bool_,
            "action_applied": np.bool_,
            "terminal_post_step": np.bool_,
        }
        save_numeric_npz(
            output_paths["trace"],
            {name: np.asarray(values, dtype=dtypes.get(name, np.float32)) for name, values in trace.items()},
        )

    outcome = (
        collision_type
        if collision_occurred
        else ("overtake" if relative_unwrapped > 0.0 else "follow")
    )
    final_state_num = {
        "follow": 1,
        "overtake": 2,
        "ego-opp": 3,
        "ego-wall": 3,
        "opp-wall": 3,
    }[outcome]
    proximity_quality = evaluate_proximity_quality(
        np.asarray(ego_raw_lidar_history, dtype=np.float64)
    )
    episode_metrics = {
        "episode_key": key,
        "outcome": outcome,
        "avg_speed": float(avg_speed),
        "speed_variance": float(speed_variance),
        "total_distance": float(total_distance),
        "observation_finite": observation_finite,
        "action_finite": action_finite,
        "ego_collision_time_s": ego_collision_time_s,
        "simulation_time_s": float(lap_time),
        "steps": int(step_count),
        "final_relative_position_m": float(relative_unwrapped),
        **proximity_quality,
        "ego_avg_desired_speed": float(np.mean(ego_desired_speeds)),
        "ego_min_lidar": float(np.min(ego_lidar_minima)),
        "opp_avg_speed": float(np.mean(opp_speeds)),
        "opp_speed_variance": float(np.var(opp_speeds)),
        "opp_avg_desired_speed": float(np.mean(opp_desired_speeds)),
        "opp_min_lidar": float(np.min(opp_lidar_minima)),
    }
    if metrics_out:
        try:
            atomic_write_json(metrics_out, episode_metrics)
        except Exception as error:
            raise RuntimeError(
                f"Failed to write worker metrics to {metrics_out}: {error}"
            ) from error
    
    return {
        'state': final_state_num,
        'avg_speed': float(avg_speed),
        'speed_variance': float(speed_variance),
        'total_distance': float(total_distance),
        'ego_idx': ego_idx,
        'opp_idx': opp_idx,
        'opp_raceline': opp_raceline,
        'opp_speed_scale': opp_speed_scale,
        'episode_metrics': episode_metrics,
    }

if __name__ == "__main__":
    args = parse_arguments()

    try:
        # Set device - prefer CUDA if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = End2Race(hidden_scale=args.hidden_scale).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True), strict=True)
        model.eval()

        # Run evaluation
        result = evaluate_segment(
            model, device, args.noise,
            args.map_name, args.ego_idx, args.interval_idx,
            args.ego_raceline, args.opp_raceline, args.opp_speedscale,
            args.sim_duration, args.render, args.save_trace, args.model_path,
            args.metrics_out,
            args.collision_scope,
        )
    except Exception as error:
        print(f"EVALUATION_ERROR={error}", file=sys.stderr)
        sys.exit(4)
    
    # Print results
    print(f"STATE={result['state']}")
    if result["state"] == 3:
        print(f"COLLISION_TYPE={result['episode_metrics']['outcome']}")
    print(f"AVG_SPEED={result['avg_speed']:.3f}")
    print(f"SPEED_VARIANCE={result['speed_variance']:.3f}")
    print(f"TOTAL_DISTANCE={result['total_distance']:.3f}")
    
    # Exit with state code
    sys.exit(result['state'])
