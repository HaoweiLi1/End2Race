#!/bin/bash
# export DISPLAY=:1
# nohup bash collect.sh > /dev/null 2>&1 &
# pkill -f collect.sh && pkill -f demonstration.py

# ── Mode ──────────────────────────────────────────────────────────────────
# true  : sweep over lattice parameter combinations (cost_weights × safety_margin)
# false : single run using demonstration.py's default lattice parameters
MULTIPARAMETERS=false

# ── Shared parameters ────────────────────────────────────────────────────
WORKERS=8
RENDER=true
MAP_NAME="Austin"
EGO_RACELINE="raceline1"
OPP_RACELINES=("raceline0" "raceline1" "raceline2")
OPP_SPEED_SCALES=(0.5 0.6 0.7 0.8)
INTERVAL_IDX=15
SIM_DURATION=8.0
NUM_STARTPOINTS=50

# ── Multi-parameter sweep ranges (only used if MULTIPARAMETERS=true) ─────
# cost_weights: [follow_cost, speed_reward, curvature_cost, collision_cost]
FOLLOW_COSTS=(0.05 0.10 0.15 0.20)
SPEED_REWARDS=(1.6 1.8 2.0 2.2)
CURVATURE_COSTS=(0.3 0.4 0.5 0.6)
COLLISION_COSTS=(0.5 1.0 1.5)
# safety_margin: 1x=(0.03,0.04)  3x=(0.09,0.12)  5x=(0.15,0.20)
SAFETY_MARGINS=("0.03 0.04" "0.09 0.12" "0.15 0.20")

# ── Generate ego indices ──────────────────────────────────────────────────
raceline_path="f1tenth_racetracks/${MAP_NAME}/${EGO_RACELINE}.csv"
max_waypoints=$(tail -n +3 "$raceline_path" | wc -l)
ego_idx_range=()
for ((i=0; i<NUM_STARTPOINTS; i++)); do
    idx=$((i * (max_waypoints - 1) / (NUM_STARTPOINTS - 1)))
    ego_idx_range+=($idx)
done

# ── Logging helpers ──────────────────────────────────────────────────────
ROOT_DIR="Dataset_${MAP_NAME}"
mkdir -p "$ROOT_DIR"
LOG_FILE="${ROOT_DIR}/collect.log"
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"; }

# ── Inner loop: run one (opp_raceline × opp_speed × ego_idx) sweep ───────
# Extra flags beyond the defaults are passed via $1 (e.g. cost_weights + safety_margin).
run_sweep() {
    local extra_args="$1"
    local launched=0
    for opp_raceline in "${OPP_RACELINES[@]}"; do
        for opp_speed in "${OPP_SPEED_SCALES[@]}"; do
            for ego_idx in "${ego_idx_range[@]}"; do
                while [ "$(jobs -r | wc -l)" -ge $WORKERS ]; do
                    sleep 0.1
                done

                local cmd="python demonstration.py \
                    --map_name $MAP_NAME \
                    --raceline $EGO_RACELINE \
                    --opp_raceline $opp_raceline \
                    --opp_speed_scale $opp_speed \
                    --ego_idx $ego_idx \
                    --interval_idx $INTERVAL_IDX \
                    --sim_duration $SIM_DURATION \
                    $extra_args"
                [ "$RENDER" = true ] && cmd="$cmd --render"

                eval "$cmd" >/dev/null 2>&1 &
                ((launched++))
            done
        done
    done
    wait
    echo "$launched"
}

# ── Count success/collision in a given dataset dir ───────────────────────
count_results() {
    local success_dir="$1" collision_dir="$2"
    local s_total=0 s_follow=0 s_overtake=0 c_total=0
    if [ -d "$success_dir" ]; then
        for csv in "$success_dir"/*.csv; do
            [ -f "$csv" ] || continue
            ((s_total++))
            [[ $(basename "$csv") == f_* ]] && ((s_follow++)) || ((s_overtake++))
        done
    fi
    if [ -d "$collision_dir" ]; then
        c_total=$(ls "$collision_dir"/*.json 2>/dev/null | wc -l)
        c_total=$((c_total + 0))
    fi
    echo "$s_total $s_follow $s_overtake $c_total"
}

# ── Mode: single run (default lattice params) ────────────────────────────
if [ "$MULTIPARAMETERS" = false ]; then
    jobs_per_run=$((${#OPP_RACELINES[@]} * ${#OPP_SPEED_SCALES[@]} * ${#ego_idx_range[@]}))
    log "Batch Data Collection (Single Parameter Set)"
    log "============================================="
    log "Map: $MAP_NAME  |  Ego raceline: $EGO_RACELINE"
    log "Opponents: ${OPP_RACELINES[*]}  Speeds: ${OPP_SPEED_SCALES[*]}"
    log "Total jobs: $jobs_per_run  |  Workers: $WORKERS"

    launched=$(run_sweep "")

    # demonstration.py places output under Dataset_{MAP}_* directories.
    output_dirs=($(ls -d "Dataset_${MAP_NAME}"_*/ 2>/dev/null))
    log ""
    log "Output directories: ${#output_dirs[@]}"
    for d in "${output_dirs[@]}"; do
        read -r s_total s_follow s_overtake c_total \
            < <(count_results "$d/success" "$d/collision")
        log "  $d : Success $s_total (F:$s_follow O:$s_overtake)  Collision: $c_total"
    done
    log "All simulations completed ($launched jobs)."
    exit 0
fi

# ── Mode: multi-parameter sweep ──────────────────────────────────────────
PROGRESS_FILE="${ROOT_DIR}/progress.txt"
touch "$PROGRESS_FILE"

n_groups=$((${#FOLLOW_COSTS[@]} * ${#SPEED_REWARDS[@]} * ${#CURVATURE_COSTS[@]} * ${#COLLISION_COSTS[@]} * ${#SAFETY_MARGINS[@]}))
jobs_per_group=$((${#OPP_RACELINES[@]} * ${#OPP_SPEED_SCALES[@]} * ${#ego_idx_range[@]}))
completed=$(wc -l < "$PROGRESS_FILE" | tr -d ' ')

log "Batch Data Collection (Multi-Parameter)"
log "============================================="
log "Map: $MAP_NAME  |  Ego raceline: $EGO_RACELINE"
log "Opponents: ${OPP_RACELINES[*]}  Speeds: ${OPP_SPEED_SCALES[*]}"
log "Start points: $NUM_STARTPOINTS  |  Workers: $WORKERS"
log "Jobs per group: $jobs_per_group  |  Parameter groups: $n_groups"
log "Total jobs: $((n_groups * jobs_per_group))  |  Completed groups: $completed"

gi=0
for fc in "${FOLLOW_COSTS[@]}"; do
for sr in "${SPEED_REWARDS[@]}"; do
for cc in "${CURVATURE_COSTS[@]}"; do
for colc in "${COLLISION_COSTS[@]}"; do
for sm in "${SAFETY_MARGINS[@]}"; do
    read -r sw sl <<< "$sm"
    ((gi++))

    PARAM_DIR="cw${fc}_${sr}_${cc}_${colc}_sm${sw}_${sl}"
    DATASET_DIR="${ROOT_DIR}/${PARAM_DIR}"

    if grep -qxF "$PARAM_DIR" "$PROGRESS_FILE"; then
        continue
    fi

    SUCCESS_DIR="$DATASET_DIR/success"
    COLLISION_DIR="$DATASET_DIR/collision"
    mkdir -p "$SUCCESS_DIR" "$COLLISION_DIR"

    log ""
    log "[$gi/$n_groups] $PARAM_DIR"
    log "  cost_weights: [$fc, $sr, $cc, $colc]  safety_margin: [$sw, $sl]"

    extra="--cost_weights $fc $sr $cc $colc --safety_margin $sw $sl"
    launched=$(run_sweep "$extra")

    read -r s_total s_follow s_overtake c_total \
        < <(count_results "$SUCCESS_DIR" "$COLLISION_DIR")
    log "  Done: $launched jobs | Success: $s_total (F:$s_follow O:$s_overtake) Collision: $c_total"

    echo "$PARAM_DIR" >> "$PROGRESS_FILE"
done
done
done
done
done

log ""
log "============================================="
log "All $n_groups groups done."
