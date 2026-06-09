#!/bin/bash

# Parameters (converted from argparse defaults)
MODEL_PATH="pretrained/end2race.pth"
HIDDEN_SCALE=4
NOISE=0.0
NUM_WORKERS=10
MAP_NAME="Austin"
RENDER=true
SIM_DURATION=8.0
EGO_RACELINE="raceline1"
OPP_RACELINES=("raceline0" "raceline1" "raceline2")
OPP_SPEED_SCALES=(0.5 0.6 0.7 0.8)
INTERVAL_IDX=15
NUM_STARTPOINTS=2

# Generate ego_idx_range
raceline_path="f1tenth_racetracks/${MAP_NAME}/${EGO_RACELINE}.csv"
max_waypoints=$(tail -n +3 "$raceline_path" | wc -l)
ego_idx_range=()
for ((i=0; i<NUM_STARTPOINTS; i++)); do
    idx=$((i * max_waypoints / (NUM_STARTPOINTS - 1)))
    ego_idx_range+=($idx)
done

# Calculate total segments
total_segments=$((${#ego_idx_range[@]} * ${#OPP_RACELINES[@]} * ${#OPP_SPEED_SCALES[@]}))

echo "Starting batch evaluation of $total_segments segments"
echo "Model: $MODEL_PATH"
echo "Map: $MAP_NAME"
echo "Workers: $NUM_WORKERS"
echo "Noise level: $NOISE"

start_time=$(date +%s)

model_name=$(basename "$MODEL_PATH")
model_name="${model_name%.*}"
result_dir="eval_results/${model_name}_${MAP_NAME}"
if awk -v noise="$NOISE" 'BEGIN { exit !(noise > 0) }'; then
    noise_suffix=$(awk -v noise="$NOISE" 'BEGIN { printf "%d", noise * 100 }')
    result_dir="${result_dir}_noise${noise_suffix}"
fi
result_path="${result_dir}/results_multi.json"

# Temporary directory to store individual results
temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT

pct() {
    awk -v count="$1" -v total="$total_segments" 'BEGIN {
        if (total == 0) {
            printf "0.0"
        } else {
            printf "%.1f", count * 100 / total
        }
    }'
}

# Generate parameter combinations and run evaluations
job_id=0

for ego_idx in "${ego_idx_range[@]}"; do
    for opp_raceline in "${OPP_RACELINES[@]}"; do
        for speed_scale in "${OPP_SPEED_SCALES[@]}"; do
            exit_result_path="$temp_dir/$job_id.exit"
            log_result_path="$temp_dir/$job_id.log"
            cmd=(
                python eval_multiagent.py
                --model_path "$MODEL_PATH"
                --map_name "$MAP_NAME"
                --ego_idx "$ego_idx"
                --interval_idx "$INTERVAL_IDX"
                --ego_raceline "$EGO_RACELINE"
                --opp_raceline "$opp_raceline"
                --opp_speedscale "$speed_scale"
                --sim_duration "$SIM_DURATION"
                --hidden_scale "$HIDDEN_SCALE"
                --noise "$NOISE"
            )
            
            if [ "$RENDER" = true ]; then
                cmd+=(--render)
            fi
            
            while [ $(jobs -r | wc -l) -ge $NUM_WORKERS ]; do
                sleep 0.1
            done
            
            ("${cmd[@]}" > "$log_result_path" 2>&1; echo $? > "$exit_result_path") &
            ((job_id++))
        done
    done
done

wait

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo ""
echo "Evaluation complete in ${elapsed} seconds"

# Count results by exit code
following_count=0
overtaking_count=0
collision_count=0
error_count=0

for result_file in "$temp_dir"/*.exit; do
    [ -e "$result_file" ] || continue
    exit_code=$(cat "$result_file")
    case $exit_code in
        1) ((following_count++)) ;;
        2) ((overtaking_count++)) ;;
        3) ((collision_count++)) ;;
        *) ((error_count++)) ;;
    esac
done

success_count=$((following_count + overtaking_count))

python - "$result_path" "$temp_dir" "$total_segments" "$following_count" "$overtaking_count" "$collision_count" "$error_count" "$elapsed" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    result_path,
    temp_dir,
    total_segments,
    following_count,
    overtaking_count,
    collision_count,
    error_count,
    elapsed,
) = sys.argv[1:]

total_segments = int(total_segments)
following_count = int(following_count)
overtaking_count = int(overtaking_count)
collision_count = int(collision_count)
error_count = int(error_count)
success_count = following_count + overtaking_count

def multi_episode_sort_key(key):
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

if os.path.exists(result_path):
    with open(result_path, 'r') as f:
        data = json.load(f)
else:
    data = {}

episodes = data.get("episodes", {})
if not isinstance(episodes, dict):
    episodes = {}

def parse_log(path):
    values = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            values[key] = value
    return values

def optional_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def optional_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def optional_json(value, fallback):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback

batch_keys = []
for path in sorted(Path(temp_dir).glob("*.log")):
    values = parse_log(path)
    episode_key = values.get("EPISODE_KEY")
    try:
        state = int(values["STATE"])
        avg_speed = float(values["AVG_SPEED"])
        speed_variance = float(values["SPEED_VARIANCE"])
        total_distance = float(values["TOTAL_DISTANCE"])
    except (KeyError, ValueError):
        continue
    if not episode_key:
        continue
    state_label = values.get("STATE_LABEL", "unknown")
    collision_occurred = values.get("COLLISION_OCCURRED", "").lower() == "true"
    danger_sectors = optional_json(values.get("DANGER_SECTORS", "{}"), {})
    proximity_timesteps = optional_json(values.get("PROXIMITY_BELOW_THRESHOLD_TIMESTEPS", "[]"), [])
    steering_timesteps = optional_json(values.get("STEERING_ANOMALY_TIMESTEPS", "[]"), [])
    episodes[episode_key] = {
        "state": state,
        "state_label": state_label,
        "avg_speed": avg_speed,
        "speed_variance": speed_variance,
        "total_distance": total_distance,
        "collision_occurred": collision_occurred,
        "global_min_surface_dist": optional_float(values.get("GLOBAL_MIN_SURFACE_DIST")),
        "danger_sectors": danger_sectors,
        "proximity_below_threshold_timesteps": proximity_timesteps,
        "steering_anomaly_timesteps": steering_timesteps,
        "max_steer_delta": optional_float(values.get("MAX_STEER_DELTA")),
        "max_steer_reversals": optional_int(values.get("MAX_STEER_REVERSALS")),
        "steer_autocorr_lag1": optional_float(values.get("STEER_AUTOCORR_LAG1")),
    }
    batch_keys.append(episode_key)

batch_metrics = [episodes[key] for key in batch_keys if key in episodes]

def pct(count):
    return round(count * 100.0 / total_segments, 1) if total_segments else 0.0

def mean_metric(name):
    values = [
        metric.get(name)
        for metric in batch_metrics
        if isinstance(metric.get(name), (int, float))
    ]
    return round(sum(values) / len(values), 6) if values else 0.0

proximity_danger_episode_count = sum(
    1
    for metric in batch_metrics
    if metric.get("proximity_below_threshold_timesteps")
)
steering_anomaly_episode_count = sum(
    1
    for metric in batch_metrics
    if metric.get("steering_anomaly_timesteps")
)
proximity_danger_timestep_count = sum(
    len(metric.get("proximity_below_threshold_timesteps") or [])
    for metric in batch_metrics
)
steering_anomaly_timestep_count = sum(
    len(metric.get("steering_anomaly_timesteps") or [])
    for metric in batch_metrics
)

ordered_episodes = {
    key: episodes[key]
    for key in sorted(episodes, key=multi_episode_sort_key)
}
final = {
    "total_episodes": total_segments,
    "recorded_episodes": len(batch_keys),
    "following_count": following_count,
    "overtaking_count": overtaking_count,
    "success_count": success_count,
    "collision_count": collision_count,
    "error_count": error_count,
    "proximity_danger_episode_count": proximity_danger_episode_count,
    "steering_anomaly_episode_count": steering_anomaly_episode_count,
    "proximity_danger_timestep_count": proximity_danger_timestep_count,
    "steering_anomaly_timestep_count": steering_anomaly_timestep_count,
    "success_rate": pct(success_count),
    "collision_rate": pct(collision_count),
    "error_rate": pct(error_count),
    "proximity_danger_episode_rate": pct(proximity_danger_episode_count),
    "steering_anomaly_episode_rate": pct(steering_anomaly_episode_count),
    "avg_speed_mean": mean_metric("avg_speed"),
    "speed_variance_mean": mean_metric("speed_variance"),
    "total_distance_mean": mean_metric("total_distance"),
    "global_min_surface_dist_mean": mean_metric("global_min_surface_dist"),
    "max_steer_delta_mean": mean_metric("max_steer_delta"),
    "max_steer_reversals_mean": mean_metric("max_steer_reversals"),
    "steer_autocorr_lag1_mean": mean_metric("steer_autocorr_lag1"),
    "elapsed_seconds": int(elapsed),
}
data = {
    "final": final,
    "episodes": ordered_episodes,
}
directory = os.path.dirname(result_path)
if directory:
    os.makedirs(directory, exist_ok=True)
tmp_path = f"{result_path}.tmp"
with open(tmp_path, 'w') as f:
    json.dump(data, f, indent=2)
    f.write("\n")
os.replace(tmp_path, result_path)
print(f"Metrics saved to {result_path}")
PY

echo ""
echo "Results by category:"
echo "  following: $following_count ($(pct "$following_count")%)"
echo "  overtaking: $overtaking_count ($(pct "$overtaking_count")%)"
echo "  success: $success_count ($(pct "$success_count")%)"
echo "  collision: $collision_count ($(pct "$collision_count")%)"
echo "  error: $error_count ($(pct "$error_count")%)"
