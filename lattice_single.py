import argparse
import csv
import json
from pathlib import Path

import gym
import imageio
import numpy as np

from demonstration import setup_ego_planner
from latticeplanner.utils import downsample_lidar
from utils import create_planner_render_callback, load_raceline_with_speed


SAMPLE_INTERVAL = 0.1
SEQUENCE_LENGTH = 80
LIDAR_FEATURES = 180
STEERING_BOUND = 0.52
SIMULATION_TIMESTEP = 0.01
VIDEO_FPS = 100


def parse_arguments():
    """Parse single-agent collection arguments."""
    parser = argparse.ArgumentParser(description="Single-agent lattice data collector")
    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--raceline", type=str, default="raceline1")
    parser.add_argument("--ego_idx", type=int, default=0)
    parser.add_argument("--lap_num", type=int, default=1)
    parser.add_argument("--max_duration", type=float, default=180.0)
    parser.add_argument("--output_directory", type=Path, default=Path("dataset_lattice_single"))
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def save_success(output_directory, base_name, collected_data):
    """Save the complete lap and fixed-length training slices."""
    header = ["time", "current_speed", "steer", "desired_speed"] + [
        f"lidar_{index}" for index in range(LIDAR_FEATURES)
    ]
    entire_directory = output_directory / "success_entire"
    slice_directory = output_directory / "success"
    entire_directory.mkdir(parents=True, exist_ok=True)
    slice_directory.mkdir(parents=True, exist_ok=True)

    entire_path = entire_directory / f"{base_name}.csv"
    with entire_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(collected_data)

    last_start = len(collected_data) - SEQUENCE_LENGTH
    if last_start < 0:
        raise ValueError(
            f"Successful lap has fewer than {SEQUENCE_LENGTH} samples"
        )
    slice_starts = list(range(0, last_start + 1, SEQUENCE_LENGTH))
    if slice_starts[-1] != last_start:
        slice_starts.append(last_start)

    for path in slice_directory.glob(f"{base_name}_*.csv"):
        path.unlink()
    for slice_index, start in enumerate(slice_starts):
        slice_path = slice_directory / f"{base_name}_{slice_index:03d}.csv"
        with slice_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(collected_data[start : start + SEQUENCE_LENGTH])

    print(f"Complete single-agent data saved to {entire_path}")
    print(f"Saved {len(slice_starts)} training slices to {slice_directory}")
    return entire_directory


def save_failure(
    output_directory,
    base_name,
    map_name,
    raceline,
    ego_idx,
    elapsed_time,
    laps_completed,
    collision_occurred,
):
    """Save metadata for a collision or an incomplete run."""
    failure_name = "collision" if collision_occurred else "incomplete"
    failure_directory = output_directory / failure_name
    failure_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = failure_directory / f"{base_name}.json"
    metadata = {
        "mode": "single_agent_lattice",
        "map_name": map_name,
        "ego_raceline": raceline,
        "ego_idx": int(ego_idx),
        "simulation_time": float(elapsed_time),
        "laps_completed": int(laps_completed),
        "collision_occurred": bool(collision_occurred),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Failure metadata saved to {metadata_path}")
    return failure_directory


def run_lattice_planner(args):
    """Collect one or more single-agent laps with the legacy lattice planner."""
    planner, _ = setup_ego_planner(args.map_name, args.raceline)
    env = gym.make(
        "f110-v0",
        map=planner.map_path,
        map_ext=".png",
        timestep=SIMULATION_TIMESTEP,
        num_agents=1,
    )

    render_info = {
        "ego_steer": 0.0,
        "ego_speed": 0.0,
        "opp_steer": 0.0,
        "opp_speed": 0.0,
    }
    draw_grid_points = []
    draw_trajectory_points = []
    render_callback = None
    video_writer = None
    temporary_video_path = None
    base_name = f"{args.map_name}_{args.raceline}_e{args.ego_idx}"
    if args.render:
        args.output_directory.mkdir(parents=True, exist_ok=True)
        temporary_video_path = args.output_directory / f".{base_name}.tmp.mp4"
        video_writer = imageio.get_writer(
            temporary_video_path,
            fps=VIDEO_FPS,
            macro_block_size=1,
        )
        render_callback = create_planner_render_callback(
            render_info,
            lambda: planner,
            draw_grid_points,
            draw_trajectory_points,
            margin=800.0,
        )
        env.add_render_callback(render_callback)

    start_pose, _, waypoints = load_raceline_with_speed(
        args.map_name,
        f"{args.raceline}.csv",
        args.ego_idx,
    )
    obs, _, done, _ = env.reset(poses=start_pose)
    if args.render:
        env.render()

    elapsed_time = 0.0
    next_record_time = SAMPLE_INTERVAL
    collected_data = []
    empty_opponents = np.empty((0, 3), dtype=np.float64)
    tracker_steps = int(planner.conf.tracker_steps)
    while (
        not done
        and obs["lap_counts"][0] < args.lap_num
        and elapsed_time < args.max_duration
    ):
        trajectory = planner.plan(
            obs["poses_x"][0],
            obs["poses_y"][0],
            obs["poses_theta"][0],
            empty_opponents,
            obs["linear_vels_x"][0],
        )

        for _ in range(tracker_steps):
            if (
                done
                or obs["lap_counts"][0] >= args.lap_num
                or elapsed_time >= args.max_duration
            ):
                break

            ego_steer, ego_speed = planner.tracker.plan(
                obs["poses_x"][0],
                obs["poses_y"][0],
                obs["poses_theta"][0],
                obs["linear_vels_x"][0],
                trajectory,
            )
            ego_steer = np.clip(ego_steer, -STEERING_BOUND, STEERING_BOUND)
            obs, timestep, done, _ = env.step(
                np.array([[ego_steer, ego_speed]])
            )
            elapsed_time += timestep

            while elapsed_time >= next_record_time:
                lidar = downsample_lidar(
                    np.asarray(obs["scans"][0]).ravel(),
                    original_points=len(obs["scans"][0]),
                    target_points=LIDAR_FEATURES,
                )
                collected_data.append(
                    [
                        round(next_record_time, 4),
                        obs["linear_vels_x"][0],
                        ego_steer,
                        ego_speed,
                    ]
                    + lidar.tolist()
                )
                next_record_time += SAMPLE_INTERVAL

            if video_writer is not None:
                render_callback.render_info.update(
                    {"ego_steer": ego_steer, "ego_speed": ego_speed}
                )
                frame = env.render(mode="rgb_array")
                if frame is not None:
                    video_writer.append_data(frame)

    collision_occurred = bool(obs["collisions"][0])
    laps_completed = int(obs["lap_counts"][0])
    if video_writer is not None:
        video_writer.close()

    if not collision_occurred and laps_completed >= args.lap_num:
        artifact_directory = save_success(
            args.output_directory,
            base_name,
            collected_data,
        )
        status = "Lap completed"
    else:
        artifact_directory = save_failure(
            args.output_directory,
            base_name,
            args.map_name,
            args.raceline,
            args.ego_idx,
            elapsed_time,
            laps_completed,
            collision_occurred,
        )
        status = "Collision occurred" if collision_occurred else "Incomplete"

    if temporary_video_path is not None:
        video_path = artifact_directory / f"{base_name}.mp4"
        temporary_video_path.replace(video_path)
        print(f"Video saved to {video_path}")

    print(f"Map: {args.map_name}")
    print(f"Start waypoint: {args.ego_idx % len(waypoints)}")
    print(f"Laps completed: {laps_completed}/{args.lap_num}")
    print(f"Samples collected: {len(collected_data)}")
    print(f"Time elapsed: {elapsed_time:.2f}s")
    print(f"Status: {status}")

    for item in draw_grid_points + draw_trajectory_points:
        item.delete()
    env.close()


if __name__ == "__main__":
    arguments = parse_arguments()
    run_lattice_planner(arguments)
