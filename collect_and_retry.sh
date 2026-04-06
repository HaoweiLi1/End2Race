#!/bin/bash

# Parameters
WORKERS=4
MAP_NAME="Austin"
EGO_RACELINE="raceline1"
OPP_RACELINES=("raceline0" "raceline1" "raceline2")
OPP_SPEED_SCALES=(0.5 0.6 0.7 0.8)
INTERVAL_IDX=15
SIM_DURATION=8.0
COLLISION_METHOD="merged"
NUM_STARTPOINTS=50
MAX_RETRIES=10
SPEED_REWARD_STEP=0.2
CURVATURE_COST_STEP=0.05
BASE_SPEED_REWARD=2.5
BASE_CURVATURE_COST=0.3

# Generate ego_idx_range
raceline_path="f1tenth_racetracks/${MAP_NAME}/${EGO_RACELINE}.csv"
max_waypoints=$(tail -n +3 "$raceline_path" | wc -l)
ego_idx_range=()
for ((i=0; i<NUM_STARTPOINTS; i++)); do
    idx=$((i * (max_waypoints - 1) / (NUM_STARTPOINTS - 1)))
    ego_idx_range+=($idx)
done

# Safety margins per opp_raceline: raceline1 → default, raceline0/2 → wider
get_safety_margins() {
    local rl="$1"
    if [ "$rl" = "raceline1" ]; then
        CASE_SAFETY_W=0.05
        CASE_SAFETY_L=0.07
    else
        CASE_SAFETY_W=0.15
        CASE_SAFETY_L=0.20
    fi
}

# Calculate total jobs
total_jobs=$((${#OPP_RACELINES[@]} * ${#OPP_SPEED_SCALES[@]} * ${#ego_idx_range[@]}))

echo "Lattice Planner Batch Data Collection + Retry"
echo "====================================="
echo "Map: $MAP_NAME"
echo "Ego raceline: $EGO_RACELINE"
echo "Opponent racelines: ${OPP_RACELINES[*]}"
echo "Speed scales: ${OPP_SPEED_SCALES[*]}"
echo "Interval: $INTERVAL_IDX"
echo "Time per run: ${SIM_DURATION}s"
echo "Starting points: $NUM_STARTPOINTS"
echo "Total jobs: $total_jobs"
echo "Workers: $WORKERS"
echo "Max retries: $MAX_RETRIES"

# ── Phase 1: Initial collection ──────────────────────────────────────────

echo ""
echo "Phase 1: Collecting..."
echo "====================================="

for opp_raceline in "${OPP_RACELINES[@]}"; do
    get_safety_margins "$opp_raceline"
    for opp_speed in "${OPP_SPEED_SCALES[@]}"; do
        for ego_idx in "${ego_idx_range[@]}"; do
            cmd="python demonstration.py --map_name $MAP_NAME --raceline $EGO_RACELINE --opp_raceline $opp_raceline --opp_speed_scale $opp_speed --ego_idx $ego_idx --interval_idx $INTERVAL_IDX --sim_duration $SIM_DURATION --collision_method $COLLISION_METHOD --safety_w $CASE_SAFETY_W --safety_l $CASE_SAFETY_L"

            while [ $(jobs -r | wc -l) -ge $WORKERS ]; do
                sleep 0.1
            done

            eval "$cmd" >/dev/null 2>&1 &
        done
    done
done

wait
echo "All initial simulations completed"

# Find output directory
DATASET_DIR=$(ls -d Dataset_${MAP_NAME}_* 2>/dev/null | tail -1)
if [ -z "$DATASET_DIR" ]; then
    echo "No dataset directory found."
    exit 1
fi

SUCCESS_DIR="$DATASET_DIR/success"
COLLISION_DIR="$DATASET_DIR/collision"
REJECTED_DIR="$DATASET_DIR/rejected"
DEMO_OUTPUT_DIR="Dataset_${MAP_NAME}_$(date +%m%d)"

# Print initial statistics
echo ""
echo "Initial Results: $DATASET_DIR"
echo "-------------------------------------"

success_count=0
collision_count=0
follow_count=0
overtake_count=0

if [ -d "$SUCCESS_DIR" ]; then
    for csv_file in "$SUCCESS_DIR"/*_ol*_e*_o*_s*.csv; do
        if [ -f "$csv_file" ]; then
            filename=$(basename "$csv_file")
            ((success_count++))
            if [[ $filename == f_* ]]; then
                ((follow_count++))
            else
                ((overtake_count++))
            fi
        fi
    done
fi

if [ -d "$COLLISION_DIR" ]; then
    collision_count=$(ls "$COLLISION_DIR"/*.json 2>/dev/null | wc -l)
fi

total_simulations=$((success_count + collision_count))
echo "  Total simulations: $total_simulations"
echo "  Successful: $success_count (Follow: $follow_count, Overtake: $overtake_count)"
echo "  Collisions: $collision_count"

# ── Phase 2: Validate & Retry ────────────────────────────────────────────

echo ""
echo "Phase 2: Validate & Retry"
echo "====================================="

# Extract case key: o_ol0_e85_o94_s0.5.csv → ol0_e85_o94_s0.5
get_case_key() {
    local base=$(basename "$1")
    base="${base%.*}"
    echo "${base#?_}"
}

# Parse case key into OPP_RL_NUM, EGO_IDX, OPP_SPEED
parse_case_key() {
    OPP_RL_NUM=$(echo "$1" | sed 's/^ol\([0-9]*\)_.*/\1/')
    EGO_IDX=$(echo "$1" | sed 's/.*_e\([0-9]*\)_.*/\1/')
    OPP_SPEED=$(echo "$1" | sed 's/.*_s\(.*\)$/\1/')
}

# Move bad files to rejected/
remove_case_files() {
    local case_key="$1"
    mkdir -p "$REJECTED_DIR"
    for f in "$SUCCESS_DIR"/*_"${case_key}".csv \
             "$COLLISION_DIR"/*_"${case_key}".json \
             "$COLLISION_DIR"/*_"${case_key}".mp4; do
        [ -f "$f" ] && mv "$f" "$REJECTED_DIR/"
    done
}

# Move re-run output back to DATASET_DIR (if output dir differs)
relocate_output() {
    local case_key="$1"
    [ "$DEMO_OUTPUT_DIR" = "$DATASET_DIR" ] && return
    mkdir -p "$SUCCESS_DIR" "$COLLISION_DIR"
    for f in "$DEMO_OUTPUT_DIR/success/"*_"${case_key}".csv; do
        [ -f "$f" ] && mv "$f" "$SUCCESS_DIR/"
    done
    for f in "$DEMO_OUTPUT_DIR/collision/"*_"${case_key}".json \
             "$DEMO_OUTPUT_DIR/collision/"*_"${case_key}".mp4; do
        [ -f "$f" ] && mv "$f" "$COLLISION_DIR/"
    done
}

# Run one retry case
run_retry_case() {
    local case_key="$1"
    local speed_reward="$2"
    local curvature_cost="$3"

    parse_case_key "$case_key"

    local rl_num="$OPP_RL_NUM"
    if [ "$rl_num" = "1" ]; then
        local sw=0.05 sl=0.07
    else
        local sw=0.15 sl=0.20
    fi

    python demonstration.py \
        --map_name "$MAP_NAME" \
        --ego_idx "$EGO_IDX" \
        --raceline "$EGO_RACELINE" \
        --opp_raceline "raceline${rl_num}" \
        --opp_speed_scale "$OPP_SPEED" \
        --interval_idx "$INTERVAL_IDX" \
        --collision_method "$COLLISION_METHOD" \
        --sim_duration "$SIM_DURATION" \
        --safety_w "$sw" \
        --safety_l "$sl" \
        --overtake_speed_reward "$speed_reward" \
        --overtake_curvature_cost "$curvature_cost"
}

# Collect failed case keys (full scan)
collect_full_retry_list() {
    local out_file="$1"
    > "$out_file"

    if [ -d "$SUCCESS_DIR" ]; then
        for csv in "$SUCCESS_DIR"/*.csv; do
            [ -f "$csv" ] || continue
            if ! python episode_validator.py "$csv" >/dev/null 2>&1; then
                get_case_key "$csv" >> "$out_file"
            fi
        done
    fi

    if [ -d "$COLLISION_DIR" ]; then
        for json_file in "$COLLISION_DIR"/*.json; do
            [ -f "$json_file" ] || continue
            get_case_key "$json_file" >> "$out_file"
        done
    fi

    sort -u -o "$out_file" "$out_file"
}

# Re-validate only cases from previous retry list
collect_incremental_retry_list() {
    local prev_file="$1"
    local out_file="$2"
    > "$out_file"

    while IFS= read -r case_key; do
        [ -z "$case_key" ] && continue

        # Collision?
        for jf in "$COLLISION_DIR"/*_"${case_key}".json; do
            if [ -f "$jf" ]; then
                echo "$case_key" >> "$out_file"
                continue 2
            fi
        done

        # Failed validation?
        for csv in "$SUCCESS_DIR"/*_"${case_key}".csv; do
            if [ -f "$csv" ]; then
                if ! python episode_validator.py "$csv" >/dev/null 2>&1; then
                    echo "$case_key" >> "$out_file"
                fi
                break
            fi
        done
    done < "$prev_file"

    sort -u -o "$out_file" "$out_file"
}

# Retry loop
speed_reward="$BASE_SPEED_REWARD"
curvature_cost="$BASE_CURVATURE_COST"

for round in $(seq 1 "$MAX_RETRIES"); do
    echo ""
    echo "Round ${round}: Validating..."
    echo "-------------------------------------"

    retry_file="${DATASET_DIR}/retry_round${round}.txt"

    if [ "$round" -eq 1 ]; then
        collect_full_retry_list "$retry_file"
    else
        collect_incremental_retry_list "${DATASET_DIR}/retry_round$((round - 1)).txt" "$retry_file"
    fi

    n_retry=$(wc -l < "$retry_file" | tr -d ' ')
    if [ "$n_retry" -eq 0 ]; then
        echo "  All cases passed. Done."
        break
    fi

    # Alternating weight adjustment: odd → speed_reward, even → curvature_cost
    if [ $((round % 2)) -eq 1 ]; then
        speed_reward=$(echo "$speed_reward - $SPEED_REWARD_STEP" | bc)
        speed_reward=$(echo "if ($speed_reward < 0.5) 0.5 else $speed_reward" | bc)
    else
        curvature_cost=$(echo "$curvature_cost + $CURVATURE_COST_STEP" | bc)
        curvature_cost=$(echo "if ($curvature_cost > 1.0) 1.0 else $curvature_cost" | bc)
    fi

    echo "  Failed cases: $n_retry"
    echo "  Weights: speed_reward=$speed_reward, curvature_cost=$curvature_cost"

    count=0
    while IFS= read -r case_key; do
        [ -z "$case_key" ] && continue
        count=$((count + 1))
        parse_case_key "$case_key"
        echo "  [$count/$n_retry] ego=$EGO_IDX opp_rl=$OPP_RL_NUM speed=$OPP_SPEED"
        remove_case_files "$case_key"
        run_retry_case "$case_key" "$speed_reward" "$curvature_cost"
        relocate_output "$case_key"
    done < "$retry_file"
done

# ── Final report ──────────────────────────────────────────────────────────

echo ""
echo "Final Report"
echo "====================================="

success_count=0
follow_count=0
overtake_count=0
fail_count=0

if [ -d "$SUCCESS_DIR" ]; then
    for csv_file in "$SUCCESS_DIR"/*.csv; do
        [ -f "$csv_file" ] || continue
        filename=$(basename "$csv_file")

        if python episode_validator.py "$csv_file" >/dev/null 2>&1; then
            ((success_count++))
            if [[ $filename == f_* ]]; then
                ((follow_count++))
            else
                ((overtake_count++))
            fi
        else
            ((fail_count++))
        fi
    done
fi

collision_count=0
if [ -d "$COLLISION_DIR" ]; then
    collision_count=$(ls "$COLLISION_DIR"/*.json 2>/dev/null | wc -l)
fi

rejected_count=0
if [ -d "$REJECTED_DIR" ]; then
    rejected_count=$(ls "$REJECTED_DIR" 2>/dev/null | wc -l)
fi

total=$((success_count + fail_count + collision_count))
echo "  Total simulations: $total"
echo "  Passed: $success_count (Follow: $follow_count, Overtake: $overtake_count)"
echo "  Still failing: $fail_count"
echo "  Collisions: $collision_count"
echo "  Rejected: $rejected_count"
