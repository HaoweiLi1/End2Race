#!/bin/bash
# ol1-focused batch evaluation: opponent on raceline1 only (200 segments).
# Usage: bash evaluate_ol1.sh [model_path] [speed_model_path] [result_tag]
# When speed_model_path is given, steering comes from model_path and speed
# from speed_model_path (composite policy).

# Parameters (converted from argparse defaults)
MODEL_PATH="${1:-pretrained/end2race.pth}"
SPEED_MODEL_PATH="${2:-}"
RESULT_TAG="${3:-}"
HIDDEN_SCALE=4
NOISE=0.0
NUM_WORKERS=8
MAP_NAME="Austin"
RENDER=false
SIM_DURATION=8.0
EGO_RACELINE="raceline1"
OPP_RACELINES=("raceline1")
OPP_SPEED_SCALES=(0.5 0.6 0.7 0.8)
INTERVAL_IDX=15
NUM_STARTPOINTS=50

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
if [ -n "$SPEED_MODEL_PATH" ]; then
    echo "Speed model: $SPEED_MODEL_PATH"
fi
if [ -n "$RESULT_TAG" ]; then
    echo "Result tag: $RESULT_TAG"
fi
echo "Map: $MAP_NAME"
echo "Workers: $NUM_WORKERS"
echo "Noise level: $NOISE"

start_time=$(date +%s)

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
            json_result_path="$temp_dir/$job_id.json"
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
                --metrics_out "$json_result_path"
            )

            if [ -n "$SPEED_MODEL_PATH" ]; then
                cmd+=(--speed_model_path "$SPEED_MODEL_PATH")
            fi
            if [ -n "$RESULT_TAG" ]; then
                cmd+=(--result_tag "$RESULT_TAG")
            fi

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

# Count states from worker exit codes (1=following, 2=overtaking, 3=collision)
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

if ! result_path=$(python -c 'from utils import write_multiagent_results_cli; write_multiagent_results_cli()' \
    "$MODEL_PATH" "$MAP_NAME" "$NOISE" "$temp_dir" "$total_segments" \
    "$following_count" "$overtaking_count" "$collision_count" "$error_count" "$RESULT_TAG"); then
    echo "ERROR: failed to aggregate metrics" >&2
    exit 1
fi
echo "Metrics saved to $result_path"

echo ""
echo "Results by category:"
echo "  following: $following_count ($(pct "$following_count")%)"
echo "  overtaking: $overtaking_count ($(pct "$overtaking_count")%)"
echo "  success: $success_count ($(pct "$success_count")%)"
echo "  collision: $collision_count ($(pct "$collision_count")%)"
echo "  error: $error_count ($(pct "$error_count")%)"
