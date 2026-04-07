#!/bin/bash

# Parameters
WORKERS=5
MAP_NAME="Austin"
EGO_RACELINE="raceline1"
OPP_RACELINES=("raceline0" "raceline1" "raceline2")
OPP_SPEED_SCALES=(0.5 0.6 0.7 0.8)
INTERVAL_IDX=15
SIM_DURATION=8.0
NUM_STARTPOINTS=50

# ── Retry preset tables ────────────────────────────────────────────────────
# Format: "follow_weight speed_reward curvature_cost safety_w safety_l"
# Preset 0 = Phase 1 default, Preset 1+ = retry rounds

# raceline1 (same raceline as ego, tighter margins)
PRESETS_RL1=(
    "0.01 2.5 0.3 0.05 0.07"
    "0.01 2.5 0.4 0.05 0.07"
    "0.01 2.0 0.3 0.05 0.07"
    "0.01 2.0 0.4 0.05 0.07"
    "0.01 1.5 0.4 0.05 0.07"
    "0.05 1.5 0.4 0.05 0.07"
    "0.05 1.5 0.5 0.05 0.07"
    "0.10 1.5 0.5 0.08 0.10"
    "0.15 1.5 0.5 0.08 0.10"
)

# raceline0 / raceline2 (different raceline, wider margins)
PRESETS_OTHER=(
    "0.01 2.5 0.3 0.08 0.12"
    "0.01 2.5 0.4 0.08 0.12"
    "0.01 2.0 0.3 0.08 0.12"
    "0.01 2.0 0.4 0.08 0.12"
    "0.01 1.5 0.4 0.08 0.12"
    "0.05 1.5 0.4 0.08 0.12"
    "0.05 1.5 0.5 0.08 0.12"
    "0.10 1.5 0.5 0.12 0.16"
    "0.15 1.5 0.5 0.12 0.16"
    "0.01 0.5 0.5 0.3 0.4"
)

# MAX_RETRIES = min of both preset table lengths
if [ ${#PRESETS_RL1[@]} -le ${#PRESETS_OTHER[@]} ]; then
    MAX_RETRIES=${#PRESETS_RL1[@]}
else
    MAX_RETRIES=${#PRESETS_OTHER[@]}
fi

# Generate ego_idx_range
raceline_path="f1tenth_racetracks/${MAP_NAME}/${EGO_RACELINE}.csv"
max_waypoints=$(tail -n +3 "$raceline_path" | wc -l)
ego_idx_range=()
for ((i=0; i<NUM_STARTPOINTS; i++)); do
    idx=$((i * (max_waypoints - 1) / (NUM_STARTPOINTS - 1)))
    ego_idx_range+=($idx)
done

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
echo "Retry presets: $MAX_RETRIES"

# ── Directory setup ─────────────────────────────────────────────────────────

DATASET_DIR="Dataset_${MAP_NAME}_$(date +%m%d)"
SUCCESS_DIR="$DATASET_DIR/success"
LOW_QUALITY_DIR="$DATASET_DIR/low_quality"
COLLISION_DIR="$DATASET_DIR/collision"

mkdir -p "$SUCCESS_DIR" "$LOW_QUALITY_DIR" "$COLLISION_DIR"

# ── Helper functions ────────────────────────────────────────────────────────

get_case_key() {
    local base=$(basename "$1")
    base="${base%.*}"
    echo "${base#?_}"
}

parse_case_key() {
    OPP_RL_NUM=$(echo "$1" | sed 's/^ol\([0-9]*\)_.*/\1/')
    EGO_IDX=$(echo "$1" | sed 's/.*_e\([0-9]*\)_.*/\1/')
    OPP_SPEED=$(echo "$1" | sed 's/.*_s\(.*\)$/\1/')
}

case_exists() {
    local opp_rl_num="$1" ego_idx="$2" opp_speed="$3"
    local pattern="*_ol${opp_rl_num}_e${ego_idx}_o*_s${opp_speed}"
    for dir in "$SUCCESS_DIR" "$LOW_QUALITY_DIR" "$COLLISION_DIR"; do
        for f in "$dir"/${pattern}.csv "$dir"/${pattern}.json; do
            [ -f "$f" ] && return 0
        done
    done
    return 1
}

remove_case_files() {
    local case_key="$1"
    for dir in "$SUCCESS_DIR" "$LOW_QUALITY_DIR" "$COLLISION_DIR"; do
        for f in "$dir"/*_"${case_key}".csv \
                 "$dir"/*_"${case_key}".json \
                 "$dir"/*_"${case_key}".mp4; do
            [ -f "$f" ] && rm "$f"
        done
    done
}

# Run one case with a specific preset (selects table by raceline)
run_case_with_preset() {
    local case_key="$1"
    local preset_idx="$2"

    parse_case_key "$case_key"
    local rl_num="$OPP_RL_NUM"

    # Select preset table by raceline
    if [ "$rl_num" = "1" ]; then
        read -r fw sr cc sw sl <<< "${PRESETS_RL1[$preset_idx]}"
    else
        read -r fw sr cc sw sl <<< "${PRESETS_OTHER[$preset_idx]}"
    fi

    python demonstration.py \
        --render \
        --map_name "$MAP_NAME" \
        --ego_idx "$EGO_IDX" \
        --raceline "$EGO_RACELINE" \
        --opp_raceline "raceline${rl_num}" \
        --opp_speed_scale "$OPP_SPEED" \
        --interval_idx "$INTERVAL_IDX" \
        --sim_duration "$SIM_DURATION" \
        --safety_w "$sw" \
        --safety_l "$sl" \
        --overtake_follow_weight "$fw" \
        --overtake_speed_reward "$sr" \
        --overtake_curvature_cost "$cc"
}

organize_folders() {
    # Batch validate: pass all success CSVs to validator at once,
    # then move the ones that fail
    if [ -d "$SUCCESS_DIR" ]; then
        local csv_list=()
        for csv in "$SUCCESS_DIR"/*.csv; do
            [ -f "$csv" ] && csv_list+=("$csv")
        done
        [ ${#csv_list[@]} -eq 0 ] && return

        # Run validator in scan mode, grep FAIL lines to get filenames
        local fail_files
        fail_files=$(python episode_validator.py --scan_dir "$SUCCESS_DIR" 2>/dev/null | grep '^\[FAIL\]' | sed 's/\[FAIL\] //')

        while IFS= read -r fname; do
            [ -z "$fname" ] && continue
            local base="${fname%.csv}"
            [ -f "$SUCCESS_DIR/$fname" ] && mv "$SUCCESS_DIR/$fname" "$LOW_QUALITY_DIR/"
            [ -f "$SUCCESS_DIR/${base}.mp4" ] && mv "$SUCCESS_DIR/${base}.mp4" "$LOW_QUALITY_DIR/"
        done <<< "$fail_files"
    fi
}

print_status() {
    local label="$1"
    local s_count=0 s_follow=0 s_overtake=0
    local lq_count=0 c_count=0

    for csv in "$SUCCESS_DIR"/*.csv; do
        [ -f "$csv" ] || continue
        ((s_count++))
        [[ $(basename "$csv") == f_* ]] && ((s_follow++)) || ((s_overtake++))
    done
    for csv in "$LOW_QUALITY_DIR"/*.csv; do
        [ -f "$csv" ] || continue
        ((lq_count++))
    done
    c_count=$(ls "$COLLISION_DIR"/*.json 2>/dev/null | wc -l)
    c_count=$((c_count + 0))

    local total=$((s_count + lq_count + c_count))
    echo ""
    echo "$label"
    echo "-------------------------------------"
    echo "  Success:     $s_count (Follow: $s_follow, Overtake: $s_overtake)"
    echo "  Low quality: $lq_count"
    echo "  Collision:   $c_count"
    echo "  Total:       $total"
}

collect_retry_list() {
    local out_file="$1"
    > "$out_file"

    for csv in "$LOW_QUALITY_DIR"/*.csv; do
        [ -f "$csv" ] || continue
        get_case_key "$csv" >> "$out_file"
    done

    for json_file in "$COLLISION_DIR"/*.json; do
        [ -f "$json_file" ] || continue
        get_case_key "$json_file" >> "$out_file"
    done

    sort -u -o "$out_file" "$out_file"
}

# ── Phase 1: Initial collection (preset 0) ──────────────────────────────

echo ""
echo "Phase 1: Collecting (preset 0)..."
echo "====================================="

skipped=0
launched=0
for opp_raceline in "${OPP_RACELINES[@]}"; do
    opp_rl_num="${opp_raceline#raceline}"
    # Select preset 0 from the appropriate table
    if [ "$opp_rl_num" = "1" ]; then
        read -r P1_FW P1_SR P1_CC P1_SW P1_SL <<< "${PRESETS_RL1[0]}"
    else
        read -r P1_FW P1_SR P1_CC P1_SW P1_SL <<< "${PRESETS_OTHER[0]}"
    fi
    for opp_speed in "${OPP_SPEED_SCALES[@]}"; do
        for ego_idx in "${ego_idx_range[@]}"; do
            if case_exists "$opp_rl_num" "$ego_idx" "$opp_speed"; then
                ((skipped++))
                continue
            fi

            cmd="python demonstration.py --render --map_name $MAP_NAME --raceline $EGO_RACELINE --opp_raceline $opp_raceline --opp_speed_scale $opp_speed --ego_idx $ego_idx --interval_idx $INTERVAL_IDX --sim_duration $SIM_DURATION --safety_w $P1_SW --safety_l $P1_SL --overtake_follow_weight $P1_FW --overtake_speed_reward $P1_SR --overtake_curvature_cost $P1_CC"

            while [ $(jobs -r | wc -l) -ge $WORKERS ]; do
                sleep 0.1
            done

            eval "$cmd" >/dev/null 2>&1 &
            ((launched++))
        done
    done
done

wait

if [ "$skipped" -gt 0 ]; then
    echo "Skipped $skipped already collected, launched $launched new"
else
    echo "All $launched simulations completed"
fi

# Check for missing cases after Phase 1
echo "Checking for missing cases..."
missing=0
for opp_raceline in "${OPP_RACELINES[@]}"; do
    opp_rl_num="${opp_raceline#raceline}"
    for opp_speed in "${OPP_SPEED_SCALES[@]}"; do
        for ego_idx in "${ego_idx_range[@]}"; do
            if ! case_exists "$opp_rl_num" "$ego_idx" "$opp_speed"; then
                ((missing++))
                case_key="ol${opp_rl_num}_e${ego_idx}_o0_s${opp_speed}"
                echo "  [WARN] Missing: opp_rl=$opp_rl_num ego=$ego_idx speed=$opp_speed, retrying..."
                run_case_with_preset "$case_key" 0 2>&1
            fi
        done
    done
done
[ "$missing" -eq 0 ] && echo "  All cases present."

organize_folders
print_status "After Phase 1"

# ── Phase 2: Retry with preset table ────────────────────────────────────

echo ""
echo "Phase 2: Validate & Retry"
echo "====================================="

# Resume: skip completed rounds
start_round=1
for r in $(seq 1 "$((MAX_RETRIES - 1))"); do
    if [ -f "${DATASET_DIR}/retry_round${r}.txt" ]; then
        start_round=$((r + 1))
    else
        break
    fi
done

if [ "$start_round" -gt 1 ]; then
    echo "Resuming from round $start_round (rounds 1-$((start_round - 1)) already done)"
fi

max_rounds=$((MAX_RETRIES - 1))  # preset 0 used in Phase 1, presets 1..N-1 for retries
for round in $(seq "$start_round" "$max_rounds"); do
    preset_idx=$round

    echo ""
    echo "Round ${round}: Collecting retry list..."
    echo "-------------------------------------"

    retry_file="${DATASET_DIR}/retry_round${round}.txt"
    collect_retry_list "$retry_file"

    n_retry=$(wc -l < "$retry_file" | tr -d ' ')
    if [ "$n_retry" -eq 0 ]; then
        echo "  All cases passed. Done."
        break
    fi

    # Show presets being used
    read -r fw1 sr1 cc1 sw1 sl1 <<< "${PRESETS_RL1[$preset_idx]}"
    read -r fw2 sr2 cc2 sw2 sl2 <<< "${PRESETS_OTHER[$preset_idx]}"
    echo "  Retry cases: $n_retry"
    echo "  Preset $preset_idx (rl1):   follow=$fw1 speed=$sr1 curvature=$cc1 sw=$sw1 sl=$sl1"
    echo "  Preset $preset_idx (other): follow=$fw2 speed=$sr2 curvature=$cc2 sw=$sw2 sl=$sl2"

    # Remove old files
    while IFS= read -r case_key; do
        [ -z "$case_key" ] && continue
        remove_case_files "$case_key"
    done < "$retry_file"

    # Re-run in parallel
    count=0
    while IFS= read -r case_key; do
        [ -z "$case_key" ] && continue
        count=$((count + 1))
        parse_case_key "$case_key"
        echo "  [$count/$n_retry] ego=$EGO_IDX opp_rl=$OPP_RL_NUM speed=$OPP_SPEED"

        while [ $(jobs -r | wc -l) -ge $WORKERS ]; do
            sleep 0.1
        done

        run_case_with_preset "$case_key" "$preset_idx" >/dev/null 2>&1 &
    done < "$retry_file"
    wait

    # Check for missing cases (silent crashes) and retry them once
    while IFS= read -r case_key; do
        [ -z "$case_key" ] && continue
        parse_case_key "$case_key"
        if ! case_exists "$OPP_RL_NUM" "$EGO_IDX" "$OPP_SPEED"; then
            echo "  [WARN] Missing output for $case_key, retrying..."
            run_case_with_preset "$case_key" "$preset_idx" 2>&1
        fi
    done < "$retry_file"

    organize_folders
    print_status "After Round $round (preset $preset_idx)"
done

# ── Final report ──────────────────────────────────────────────────────────

print_status "Final Report"
