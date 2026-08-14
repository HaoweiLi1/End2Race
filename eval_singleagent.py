import argparse
import gym
import numpy as np
import torch
import os
import gc
import imageio
from f110_gym.envs.base_classes import Integrator
from model import End2Race
from latticeplanner.utils import TrackProjector
from utils import *

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate model lap completion ability')
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default='pretrained/end2race.pth')
    parser.add_argument("--hidden_scale", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.0)
    
    # Evaluation parameters
    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--lap_num", type=int, default=1)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--save_trace", action="store_true")
    
    return parser.parse_args()

def evaluate_laps(
    model,
    device,
    noise_level,
    map_name,
    render,
    lap_num,
    save_trace=False,
    model_path="pretrained/end2race.pth",
):
    """Evaluate model's ability to complete laps independently"""
    
    np.random.seed(42)
    num_features = 360
    start_idx = 0
    raceline = f'{map_name}_raceline.csv'
    env = gym.make("f110-v0", map=f"f1tenth_racetracks/{map_name}/{map_name}_map", map_ext=".png", num_agents=1, timestep=0.01, integrator=Integrator.RK4)

    output_paths = singleagent_paths(model_path, map_name, noise_level, lap_num)

    # Generate video path if rendering
    video_path = None
    if render:
        video_path = str(output_paths["video"])
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    # Initialize visualization
    render_info = {"speed": 0.0, "steer": 0.0, "lap_time": 0.0, "laps": 0}
    visited_points = []
    drawn_points = []
    batch_objects = []
    
    if render:
        render_callback = create_single_agent_render_callback(render_info, visited_points, drawn_points, batch_objects, lap_num)
        env.add_render_callback(render_callback)
    
    # Load starting position and initial speed
    start_pose, initial_speed, waypoints = load_raceline_with_speed(map_name, raceline, start_idx)
    start_position = np.array([waypoints[0, 0], waypoints[0, 1]])
    
    # Load centerline for progress tracking
    raceline_path = os.path.join('f1tenth_racetracks', map_name, raceline)
    progress_projector = TrackProjector.from_csv(raceline_path)
    centerline_total_length = progress_projector.track_length
    
    # Reset environment
    obs, _, done, _ = env.reset(poses=start_pose)
    
    # Initialize model state
    hidden_size = model.gru.hidden_size
    hidden_state = torch.zeros((1, 1, hidden_size), device=device)
    prev_speed = initial_speed * 0.9  # Always use speed conditioning
    
    # Track initial state
    initial_progress = progress_projector.progress_at(np.array([obs['poses_x'][0], obs['poses_y'][0]]))
    
    # Simulation metrics
    lap_time = 0.0
    collision_occurred = False
    trajectory = []
    speeds = []
    lap_count = 0
    lap_times = []
    video_frames = []
    near_start_flag = True
    min_lap_time = 10.0
    lap_start_time = 0.0
    desired_speeds = []
    lidar_minima = []
    trace = None
    if save_trace:
        trace = {
            "time_s": [],
            "lidar_360": [],
            "raw_action": [],
            "executed_action": [],
            "measured_speed_mps": [],
            "pose": [],
            "collision": [],
        }
    
    if render:
        env.render('human')
        env.render()
    
    # Main simulation loop
    while not done and lap_count < lap_num:
        # Model inference
        raw_lidar_360 = np.array(obs["scans"][0]).flatten()
        if len(raw_lidar_360) > num_features:
            indices = np.linspace(0, len(raw_lidar_360)-1, num_features, dtype=int)
            raw_lidar_360 = raw_lidar_360[indices]
        lidar = raw_lidar_360.copy()
        
        # Apply noise if specified
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

        raw_action = np.array([ego_steer, ego_speed], dtype=np.float32)
        ego_steer = np.clip(ego_steer, -0.52, 0.52)
        executed_action = np.array([ego_steer, ego_speed], dtype=np.float32)
        desired_speeds.append(float(ego_speed))
        lidar_minima.append(float(np.min(raw_lidar_360)))
        if trace is not None:
            trace["time_s"].append(float(lap_time))
            trace["lidar_360"].append(lidar)
            trace["raw_action"].append(raw_action)
            trace["executed_action"].append(executed_action)
            trace["measured_speed_mps"].append(float(obs['linear_vels_x'][0]))
            trace["pose"].append([obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]])
            trace["collision"].append(bool(obs['collisions'][0]))
        prev_speed = obs['linear_vels_x'][0]
        
        # Update render info
        render_info["speed"] = ego_speed
        render_info["steer"] = ego_steer
        render_info["lap_time"] = lap_time
        render_info["laps"] = lap_count
        
        # Step environment
        action = np.array([[ego_steer, ego_speed]])
        obs, timestep, done, _ = env.step(action)
        lap_time += timestep
        
        # Track metrics
        trajectory.append([obs['poses_x'][0], obs['poses_y'][0]])
        speeds.append(obs['linear_vels_x'][0])
        
        if render:
            visited_points.append([obs['poses_x'][0], obs['poses_y'][0]])

        # Lap completion detection
        current_position = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        distance_to_start = np.linalg.norm(current_position - start_position)
        
        if distance_to_start < 0.5:  # Within 0.5 meters of start
            if not near_start_flag and lap_time - lap_start_time > min_lap_time:
                lap_count += 1
                lap_duration = lap_time - lap_start_time
                lap_times.append(lap_duration)
                lap_start_time = lap_time
                print(f"Lap {lap_count}/{lap_num} completed in {lap_duration:.2f}s")
                if lap_count >= lap_num:
                    print(f"Successfully completed all {lap_num} laps!")
            near_start_flag = True
        else:
            near_start_flag = False

        # Check collision
        if obs['collisions'][0]:
            collision_occurred = True
            done = True
            print(f"Wall collision at {lap_time:.2f}s")
        
        # Capture video frame
        if render:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                video_frames.append(frame)
    
    # Calculate results
    if len(trajectory) > 0:
        last_position = trajectory[-1]
        last_progress = progress_projector.progress_at(np.array(last_position))
        
        if last_progress < initial_progress - centerline_total_length/2:
            last_progress += centerline_total_length
        
        lap_fraction = (last_progress - initial_progress) / centerline_total_length
        if lap_fraction < 0:
            lap_fraction += 1
        
        # Only add lap_fraction if we haven't completed all target laps
        if lap_count >= lap_num:
            total_lap_progress = lap_count  # Don't add fraction for completed laps
        else:
            total_lap_progress = lap_count + lap_fraction
    else:
        total_lap_progress = lap_count
    
    mean_lap_time = np.mean(lap_times) if lap_times else 0
    lap_time_variance = np.var(lap_times) if len(lap_times) > 1 else 0
    
    # Save video if requested
    if render:
        for batch_obj in batch_objects:
            batch_obj.delete()
        env.render_callbacks = []
        
        if video_frames:
            imageio.mimwrite(video_path, video_frames, fps=100, macro_block_size=1)
            print(f"Video saved to {video_path}")
    
    env.close()
    gc.collect()
    
    # Calculate final metrics
    avg_speed, speed_variance, total_distance = calculate_metrics(trajectory, speeds)

    if trace is not None:
        dtypes = {"time_s": np.float64, "pose": np.float64, "collision": np.bool_}
        save_numeric_npz(
            output_paths["trace"],
            {name: np.asarray(values, dtype=dtypes.get(name, np.float32)) for name, values in trace.items()},
        )

    result = {
        "lap_progress": float(total_lap_progress),
        "collision_occurred": bool(collision_occurred),
        "time_elapsed": float(lap_time),
        "lap_times": [float(value) for value in lap_times],
        "mean_lap_time": float(mean_lap_time),
        "lap_time_variance": float(lap_time_variance),
        "avg_speed": float(avg_speed),
        "speed_variance": float(speed_variance),
        "total_distance": float(total_distance),
        "avg_desired_speed": float(np.mean(desired_speeds)),
        "min_lidar": float(np.min(lidar_minima)),
    }
    update_singleagent_results(
        output_paths["results"],
        f"lap{lap_num}",
        result,
    )
    
    # Print results directly
    print("\n" + "="*50)
    print(f"LAP EVALUATION RESULTS - speed model")
    print("="*50)
    print(f"Map: {map_name}")
    print(f"Target Laps: {lap_num}")
    print(f"Laps Completed: {lap_count}")
    print(f"Lap Progress: {total_lap_progress:.2f} laps")
    print(f"Time Elapsed: {lap_time:.1f}s")
    print(f"Average Speed: {avg_speed:.3f} m/s")
    print(f"Speed Variance: {speed_variance:.3f} m²/s²")
    print(f"Total Distance: {total_distance:.1f} m")
    
    if lap_times:
        print(f"\nLap Times: {[f'{t:.2f}s' for t in lap_times]}")
        print(f"Mean Lap Time: {mean_lap_time:.2f}s")
        print(f"Lap Time Variance: {lap_time_variance:.3f}s²")
    
    # Print status
    if collision_occurred:
        print(f"\nStatus: Collision occurred")
    elif lap_count >= lap_num:
        print(f"\nStatus: Successfully completed all laps")
    else:
        print(f"\nStatus: Incomplete - stopped before completing all laps")

    return result


if __name__ == "__main__":
    args = parse_arguments()
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal evaluation")
    device = torch.device("cuda")
    
    # Load model
    model = End2Race(hidden_scale=args.hidden_scale).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Run evaluation
    evaluate_laps(model, device, args.noise, args.map_name, args.render, args.lap_num, args.save_trace, args.model_path)
