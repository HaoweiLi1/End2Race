#!/bin/bash
set -euo pipefail

# ── Usage ──────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    cat <<'EOF'
Usage: ./collect_and_retry.sh <dataset_dir>

Validates success episodes and retries failed/collision cases with
progressively more conservative overtake weights.

Environment variables (with defaults):
    RACELINE=raceline1           Ego raceline
    INTERVAL_IDX=15              Ego-opponent start gap
    COLLISION_METHOD=merged      Collision cost method
    SIM_DURATION=8.0             Simulation duration (seconds)
    MAX_RETRIES=3                Maximum retry rounds
    SPEED_REWARD_STEP=0.2        Speed reward reduction (odd rounds)
    CURVATURE_COST_STEP=0.05     Curvature cost increase (even rounds)

Safety margins are set per opp_raceline:
    raceline1 → safety_w=0.05  safety_l=0.07
    raceline0/2 → safety_w=0.15 safety_l=0.20

Example:
    ./collect_and_retry.sh Dataset_Austin_0406
    MAX_RETRIES=5 ./collect_and_retry.sh Dataset_Austin_0406
EOF
    exit 1
fi

# ── Configuration ──────────────────────────────────────────────────────────
DATASET_DIR="${1%/}"
# Extract map name: Dataset_Austin_0405 → Austin
MAP_NAME=$(echo "$(basename "$DATASET_DIR")" | sed 's/^Dataset_\(.*\)_[0-9]*$/\1/')

RACELINE="${RACELINE:-raceline1}"
INTERVAL_IDX="${INTERVAL_IDX:-15}"
COLLISION_METHOD="${COLLISION_METHOD:-merged}"
SIM_DURATION="${SIM_DURATION:-8.0}"

MAX_RETRIES="${MAX_RETRIES:-3}"
SPEED_REWARD_STEP="${SPEED_REWARD_STEP:-0.2}"
CURVATURE_COST_STEP="${CURVATURE_COST_STEP:-0.05}"

# demonstration.py defaults
BASE_SPEED_REWARD=2.5
BASE_CURVATURE_COST=0.3

SUCCESS_DIR="${DATASET_DIR}/success"
COLLISION_DIR="${DATASET_DIR}/collision"
REJECTED_DIR="${DATASET_DIR}/rejected"

# demonstration.py outputs to date-based dir; may differ from DATASET_DIR
DEMO_OUTPUT_DIR="Dataset_${MAP_NAME}_$(date +%m%d)"

echo "Dataset:    $DATASET_DIR"
echo "Map:        $MAP_NAME"
echo "Raceline:   $RACELINE"
echo "Max retry:  $MAX_RETRIES"
echo ""

# ── Helpers ────────────────────────────────────────────────────────────────

# Extract case key from filename (strip state prefix and extension)
#   o_ol0_e85_o94_s0.5.csv → ol0_e85_o94_s0.5
get_case_key() {
    local base
    base=$(basename "$1")
    base="${base%.*}"        # remove extension
    echo "${base#?_}"        # remove state prefix + underscore
}

# Parse case key into variables: OPP_RL_NUM, EGO_IDX, OPP_SPEED
parse_case_key() {
    local key="$1"
    OPP_RL_NUM=$(echo "$key" | sed 's/^ol\([0-9]*\)_.*/\1/')
    EGO_IDX=$(echo "$key"    | sed 's/.*_e\([0-9]*\)_.*/\1/')
    OPP_SPEED=$(echo "$key"  | sed 's/.*_s\(.*\)$/\1/')
}

# Safety margins per opp_raceline: raceline1 → default, raceline0/2 → wider
get_safety_margins() {
    local rl_num="$1"
    if [[ "$rl_num" == "1" ]]; then
        CASE_SAFETY_W=0.05
        CASE_SAFETY_L=0.07
    else
        CASE_SAFETY_W=0.15
        CASE_SAFETY_L=0.20
    fi
}

# Collect all case keys that need retry (failed validation + collisions)
collect_retry_list() {
    local out_file="$1"
    > "$out_file"

    # Failed validation in success/
    if [[ -d "$SUCCESS_DIR" ]]; then
        for csv in "$SUCCESS_DIR"/*.csv; do
            [[ -f "$csv" ]] || continue
            if ! python3 episode_validator.py "$csv" >/dev/null 2>&1; then
                get_case_key "$csv" >> "$out_file"
            fi
        done
    fi

    # All collision cases
    if [[ -d "$COLLISION_DIR" ]]; then
        for json_file in "$COLLISION_DIR"/*.json; do
            [[ -f "$json_file" ]] || continue
            get_case_key "$json_file" >> "$out_file"
        done
    fi

    sort -u -o "$out_file" "$out_file"
}

# Move bad files to rejected/
remove_case_files() {
    local case_key="$1"
    mkdir -p "$REJECTED_DIR"

    for f in "$SUCCESS_DIR"/*_"${case_key}".csv \
             "$COLLISION_DIR"/*_"${case_key}".json \
             "$COLLISION_DIR"/*_"${case_key}".mp4; do
        [[ -f "$f" ]] && mv "$f" "$REJECTED_DIR/"
    done
}

# Move re-run output from DEMO_OUTPUT_DIR to DATASET_DIR (if dirs differ)
relocate_output() {
    local case_key="$1"
    [[ "$DEMO_OUTPUT_DIR" == "$DATASET_DIR" ]] && return

    mkdir -p "$SUCCESS_DIR" "$COLLISION_DIR"

    for f in "$DEMO_OUTPUT_DIR/success/"*_"${case_key}".csv; do
        [[ -f "$f" ]] && mv "$f" "$SUCCESS_DIR/"
    done
    for f in "$DEMO_OUTPUT_DIR/collision/"*_"${case_key}".json \
             "$DEMO_OUTPUT_DIR/collision/"*_"${case_key}".mp4; do
        [[ -f "$f" ]] && mv "$f" "$COLLISION_DIR/"
    done
}

# Re-run one case
run_case() {
    local case_key="$1"
    local speed_reward="$2"
    local curvature_cost="$3"

    parse_case_key "$case_key"
    get_safety_margins "$OPP_RL_NUM"

    python3 demonstration.py \
        --map_name "$MAP_NAME" \
        --ego_idx "$EGO_IDX" \
        --raceline "$RACELINE" \
        --opp_raceline "raceline${OPP_RL_NUM}" \
        --opp_speed_scale "$OPP_SPEED" \
        --interval_idx "$INTERVAL_IDX" \
        --collision_method "$COLLISION_METHOD" \
        --sim_duration "$SIM_DURATION" \
        --safety_w "$CASE_SAFETY_W" \
        --safety_l "$CASE_SAFETY_L" \
        --overtake_speed_reward "$speed_reward" \
        --overtake_curvature_cost "$curvature_cost"
}

# ── Main loop ──────────────────────────────────────────────────────────────
speed_reward="$BASE_SPEED_REWARD"
curvature_cost="$BASE_CURVATURE_COST"

# Round 1: full scan
echo "════════════════════════════════════════════════════"
echo "  Round 1: Full validation..."
echo "════════════════════════════════════════════════════"

retry_file="${DATASET_DIR}/retry_round1.txt"
collect_retry_list "$retry_file"
n_retry=$(wc -l < "$retry_file" | tr -d ' ')

if [[ "$n_retry" -eq 0 ]]; then
    echo "  All episodes passed. No retries needed."
else
    # Round 1: reduce speed_reward (odd round)
    speed_reward=$(echo "$speed_reward - $SPEED_REWARD_STEP" | bc)
    speed_reward=$(echo "if ($speed_reward < 0.5) 0.5 else $speed_reward" | bc)

    echo "  ${n_retry} failed cases"
    echo "  Retry weights: speed_reward=${speed_reward}, curvature_cost=${curvature_cost}"
    echo ""

    count=0
    while IFS= read -r case_key; do
        [[ -z "$case_key" ]] && continue
        count=$((count + 1))
        parse_case_key "$case_key"
        echo "  [${count}/${n_retry}] ego=${EGO_IDX} opp_rl=${OPP_RL_NUM} speed=${OPP_SPEED}"
        remove_case_files "$case_key"
        run_case "$case_key" "$speed_reward" "$curvature_cost"
        relocate_output "$case_key"
    done < "$retry_file"
    echo ""

    # Rounds 2+: only re-validate the retried cases from previous round
    for round in $(seq 2 "$MAX_RETRIES"); do
        echo "════════════════════════════════════════════════════"
        echo "  Round ${round}: Validating retried cases only..."
        echo "════════════════════════════════════════════════════"

        prev_file="${DATASET_DIR}/retry_round$((round - 1)).txt"
        retry_file="${DATASET_DIR}/retry_round${round}.txt"
        > "$retry_file"

        # Only check cases from previous retry list
        while IFS= read -r case_key; do
            [[ -z "$case_key" ]] && continue

            # Check if it ended up in collision/
            found_collision=false
            for jf in "$COLLISION_DIR"/*_"${case_key}".json; do
                if [[ -f "$jf" ]]; then
                    found_collision=true
                    break
                fi
            done
            if $found_collision; then
                echo "$case_key" >> "$retry_file"
                continue
            fi

            # Check if success but failed validation
            for csv in "$SUCCESS_DIR"/*_"${case_key}".csv; do
                if [[ -f "$csv" ]]; then
                    if ! python3 episode_validator.py "$csv" >/dev/null 2>&1; then
                        echo "$case_key" >> "$retry_file"
                    fi
                    break
                fi
            done
        done < "$prev_file"

        sort -u -o "$retry_file" "$retry_file"
        n_retry=$(wc -l < "$retry_file" | tr -d ' ')

        if [[ "$n_retry" -eq 0 ]]; then
            echo "  All retried cases passed."
            break
        fi

        # Alternating: odd rounds → speed_reward, even rounds → curvature_cost
        if (( round % 2 == 1 )); then
            speed_reward=$(echo "$speed_reward - $SPEED_REWARD_STEP" | bc)
            speed_reward=$(echo "if ($speed_reward < 0.5) 0.5 else $speed_reward" | bc)
        else
            curvature_cost=$(echo "$curvature_cost + $CURVATURE_COST_STEP" | bc)
            curvature_cost=$(echo "if ($curvature_cost > 1.0) 1.0 else $curvature_cost" | bc)
        fi

        echo "  ${n_retry} still failing"
        echo "  Retry weights: speed_reward=${speed_reward}, curvature_cost=${curvature_cost}"
        echo ""

        count=0
        while IFS= read -r case_key; do
            [[ -z "$case_key" ]] && continue
            count=$((count + 1))
            parse_case_key "$case_key"
            echo "  [${count}/${n_retry}] ego=${EGO_IDX} opp_rl=${OPP_RL_NUM} speed=${OPP_SPEED}"
            remove_case_files "$case_key"
            run_case "$case_key" "$speed_reward" "$curvature_cost"
            relocate_output "$case_key"
        done < "$retry_file"

        echo ""
    done
fi

# ── Final report ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  Final Report"
echo "════════════════════════════════════════════════════"

if [[ -d "$SUCCESS_DIR" ]] && ls "$SUCCESS_DIR"/*.csv &>/dev/null; then
    python3 episode_validator.py --scan_dir "$SUCCESS_DIR"
else
    echo "No success episodes."
fi

n_collision=0
if [[ -d "$COLLISION_DIR" ]]; then
    n_collision=$(find "$COLLISION_DIR" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
fi

n_rejected=0
if [[ -d "$REJECTED_DIR" ]]; then
    n_rejected=$(find "$REJECTED_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
fi

echo ""
echo "Remaining collisions: ${n_collision}"
echo "Rejected files:       ${n_rejected}"
