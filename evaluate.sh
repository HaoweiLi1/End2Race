#!/bin/bash

# Parameters (converted from argparse defaults)
MODEL_PATH="pretrained/end2race.pth"
HIDDEN_SCALE=4
NOISE=0.0
NUM_WORKERS=8
MAP_NAME="Austin"
RENDER=true
SIM_DURATION=8.0
EGO_RACELINE="raceline1"
OPP_RACELINES=("raceline0" "raceline1" "raceline2")
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

# Aggregate from metrics JSONs with strict completeness validation.
# Worker exit code 0 = success; outcomes are read from the JSON files only.
if ! result_line=$(python aggregate_eval.py \
    --tmp_dir "$temp_dir" \
    --expected_total "$total_segments" \
    --model_path "$MODEL_PATH" \
    --map_name "$MAP_NAME" \
    --noise "$NOISE" \
    --require_npz); then
    echo "ERROR: evaluation incomplete; aggregation rejected" >&2
    exit 1
fi
echo "$result_line"
