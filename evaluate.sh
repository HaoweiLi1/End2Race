#!/bin/bash

# Parameters (converted from argparse defaults)
MODEL_PATH="${MODEL_PATH:-pretrained/end2race_new.pth}"
PYTHON="${PYTHON:-python}"
HIDDEN_SCALE=4
NOISE="${NOISE:-0.0}"
NUM_WORKERS=12
MAP_NAME="${MAP_NAME:-Austin}"
COLLISION_SCOPE="${COLLISION_SCOPE:-ego}"
RENDER=false
SAVE_TRACE="${SAVE_TRACE:-false}"
SIM_DURATION=8.0
EGO_RACELINE="raceline1"
OPP_RACELINES=("raceline0" "raceline1" "raceline2")
OPP_SPEED_SCALES=(0.5 0.6 0.7 0.8)
INTERVAL_IDX=15
NUM_STARTPOINTS=50

# Load the shared circular startpoint panel.
mapfile -t ego_idx_range < <(
    "$PYTHON" -c '
import sys
from utils import get_circular_startpoints
print(*get_circular_startpoints(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])), sep="\n")
' "$MAP_NAME" "${EGO_RACELINE}.csv" "$NUM_STARTPOINTS" 0
)

# Calculate total segments
total_segments=$((${#ego_idx_range[@]} * ${#OPP_RACELINES[@]} * ${#OPP_SPEED_SCALES[@]}))

echo "Starting batch evaluation of $total_segments segments"
echo "Model: $MODEL_PATH"
echo "Map: $MAP_NAME"
echo "Workers: $NUM_WORKERS"
echo "Noise level: $NOISE"

# Temporary directory to store individual results
temp_dir=$(mktemp -d)

show_progress() {
    local completed percent filled empty bar space
    completed=$(find "$temp_dir" -maxdepth 1 -type f -name '*.exit' | wc -l)
    percent=$((completed * 100 / total_segments))
    filled=$((completed * 40 / total_segments))
    empty=$((40 - filled))
    printf -v bar '%*s' "$filled" ''
    printf -v space '%*s' "$empty" ''
    printf '\rProgress: [%s%s] %3d%% (%d/%d)' \
        "${bar// /#}" "${space// /-}" "$percent" "$completed" "$total_segments"
}

# Generate parameter combinations and run evaluations
job_id=0
show_progress

for ego_idx in "${ego_idx_range[@]}"; do
    for opp_raceline in "${OPP_RACELINES[@]}"; do
        for speed_scale in "${OPP_SPEED_SCALES[@]}"; do
            cmd="\"$PYTHON\" eval_multiagent.py --model_path $MODEL_PATH --map_name $MAP_NAME --ego_idx $ego_idx --interval_idx $INTERVAL_IDX --ego_raceline $EGO_RACELINE --opp_raceline $opp_raceline --opp_speedscale $speed_scale --sim_duration $SIM_DURATION --hidden_scale $HIDDEN_SCALE --noise $NOISE --collision_scope $COLLISION_SCOPE"
            metrics_result_path="$temp_dir/$job_id.metrics.json"
            cmd="$cmd --metrics_out \"$metrics_result_path\""
            
            if [ "$RENDER" = true ]; then
                cmd="$cmd --render"
            fi

            if [ "$SAVE_TRACE" = true ]; then
                cmd="$cmd --save_trace"
            fi
            
            while [ "$(jobs -rp | wc -l)" -ge "$NUM_WORKERS" ]; do
                show_progress
                sleep 0.1
            done
            
            (eval "$cmd" >"$temp_dir/$job_id.log" 2>&1; echo $? > "$temp_dir/$job_id.exit") &
            ((job_id++))
        done
    done
done

while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
    show_progress
    sleep 0.1
done
wait
show_progress
echo

if ! result_counts=$("$PYTHON" -c '
import sys
from utils import aggregate_multiagent_batch, multiagent_paths
paths = multiagent_paths(sys.argv[1], sys.argv[2], float(sys.argv[3]))
final = aggregate_multiagent_batch(paths["results"], sys.argv[4], int(sys.argv[5]))
print(
    final["following_count"],
    final["overtaking_count"],
    final["success_count"],
    final["collision_count"],
    final["ego_opp_collision_count"],
    final["ego_wall_collision_count"],
    final["opp_wall_collision_count"],
    final["error_count"],
)
' "$MODEL_PATH" "$MAP_NAME" "$NOISE" "$temp_dir" "$total_segments"); then
    echo "ERROR: failed to aggregate evaluation metrics" >&2
    echo "Worker artifacts preserved at: $temp_dir" >&2
    exit 1
fi
read -r following_count overtaking_count success_count collision_count \
    ego_opp_collision_count ego_wall_collision_count opp_wall_collision_count \
    error_count <<< "$result_counts"

if [ "$error_count" -ne 0 ]; then
    echo "ERROR: $error_count evaluation worker(s) failed" >&2
    echo "Worker artifacts preserved at: $temp_dir" >&2
    exit 1
fi

rm -rf "$temp_dir"

echo ""
echo "Results by category:"
echo "  following: $following_count ($(echo "scale=1; $following_count * 100 / $total_segments" | bc)%)"
echo "  overtaking: $overtaking_count ($(echo "scale=1; $overtaking_count * 100 / $total_segments" | bc)%)"
echo "  success: $success_count ($(echo "scale=1; $success_count * 100 / $total_segments" | bc)%)"
echo "  collision: $collision_count ($(echo "scale=1; $collision_count * 100 / $total_segments" | bc)%)"
echo "    ego-opp: $ego_opp_collision_count"
echo "    ego-wall: $ego_wall_collision_count"
echo "    opp-wall: $opp_wall_collision_count"
echo "  error: $error_count ($(echo "scale=1; $error_count * 100 / $total_segments" | bc)%)"
