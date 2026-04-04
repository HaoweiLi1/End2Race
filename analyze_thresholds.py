"""Analyze metric distributions across all episodes to calibrate thresholds."""
import os
import numpy as np
import sys

DT = 0.1
SECTOR_FRONT = list(range(0, 30)) + list(range(330, 360))
SECTOR_RIGHT = list(range(60, 120))
SECTOR_REAR  = list(range(150, 210))
SECTOR_LEFT  = list(range(240, 300))
SECTOR_SIDE  = SECTOR_RIGHT + SECTOR_LEFT

scan_dir = sys.argv[1] if len(sys.argv) > 1 else "Dataset_Austin_0404/success/"

csv_files = sorted([
    os.path.join(scan_dir, f) for f in os.listdir(scan_dir) if f.endswith('.csv')
])
print(f"Analyzing {len(csv_files)} episodes...\n")

# Collect per-episode stats
all_global_min_lidar = []       # min lidar across entire episode
all_side_min_lidar = []         # min side lidar per episode
all_front_min_lidar = []
all_reversal_rates = []         # reversals per second, whole episode
all_max_window_reversals = []   # max reversals in any 1s window
all_max_jerk = []               # max |steering accel|
all_max_steer_jump = []         # max |single-step steer change|
all_steer_variance = []

for f in csv_files:
    data = np.loadtxt(f, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    steer = data[:, 1]
    lidar = data[:, 3:]

    # Lidar stats
    global_min = np.min(lidar)
    side_lidar = lidar[:, SECTOR_SIDE]
    side_min = np.min(side_lidar)
    front_lidar = lidar[:, SECTOR_FRONT]
    front_min = np.min(front_lidar)

    all_global_min_lidar.append(global_min)
    all_side_min_lidar.append(side_min)
    all_front_min_lidar.append(front_min)

    # Steering stats
    all_steer_variance.append(np.var(steer))

    if len(steer) >= 3:
        steer_rate = np.diff(steer) / DT
        steer_accel = np.diff(steer_rate) / DT
        steer_diff = np.diff(steer)
        sign_changes = np.abs(np.diff(np.sign(steer_rate)))

        # Overall reversal rate
        n_reversals = np.sum(sign_changes > 0)
        duration = len(steer) * DT
        all_reversal_rates.append(n_reversals / duration)

        # Max reversals in any 1s window
        window = 10
        max_win_rev = 0
        for i in range(len(sign_changes) - window + 1):
            win_rev = np.sum(sign_changes[i:i+window] > 0)
            max_win_rev = max(max_win_rev, win_rev)
        all_max_window_reversals.append(max_win_rev)

        all_max_jerk.append(np.max(np.abs(steer_accel)))
        all_max_steer_jump.append(np.max(np.abs(steer_diff)))

def print_percentiles(name, arr):
    arr = np.array(arr)
    print(f"{name} (n={len(arr)}):")
    for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
        print(f"  P{p:3d}: {np.percentile(arr, p):.4f}")
    print()

print("=" * 50)
print("LIDAR PROXIMITY")
print("=" * 50)
print_percentiles("Global min lidar (m)", all_global_min_lidar)
print_percentiles("Side min lidar (m)", all_side_min_lidar)
print_percentiles("Front min lidar (m)", all_front_min_lidar)

print("=" * 50)
print("STEERING")
print("=" * 50)
print_percentiles("Steer variance (rad^2)", all_steer_variance)
print_percentiles("Reversal rate (per sec)", all_reversal_rates)
print_percentiles("Max window reversals (per 1s)", all_max_window_reversals)
print_percentiles("Max |steering jerk| (rad/s^2)", all_max_jerk)
print_percentiles("Max |steer jump| (rad)", all_max_steer_jump)
