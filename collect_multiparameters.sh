#!/bin/bash
export DISPLAY=:1

# ── Fixed parameters ──────────────────────────────────────────────────────

WORKERS=6
MAP_NAME="Austin"
EGO_RACELINE="raceline1"
OPP_RACELINES=("raceline0" "raceline1" "raceline2")
OPP_SPEED_SCALES=(0.5 0.6 0.7 0.8)
INTERVAL_IDX=15
SIM_DURATION=8.0
NUM_STARTPOINTS=50

# ── Adjustable parameter ranges ──────────────────────────────────────────
# cost_weights: [follow_cost, speed_reward, curvature_cost, collision_cost]

FOLLOW_COSTS=(0.05 0.10 0.15 0.20)
SPEED_REWARDS=(1.6 1.8 2.0 2.2)
CURVATURE_COSTS=(0.3 0.4 0.5 0.6)
COLLISION_COSTS=(0.5 1.0 1.5)

# safety_margin: multiples of base (0.03, 0.04)
# 1x=(0.03,0.04)  3x=(0.09,0.12)  5x=(0.15,0.20)
SAFETY_MARGINS=("0.03 0.04" "0.09 0.12" "0.15 0.20")

# ── Generate ego indices ──────────────────────────────────────────────────

raceline_path="f1tenth_racetracks/${MAP_NAME}/${EGO_RACELINE}.csv"
max_waypoints=$(tail -n +3 "$raceline_path" | wc -l)
ego_idx_range=()
for ((i=0; i<NUM_STARTPOINTS; i++)); do
    idx=$((i * (max_waypoints - 1) / (NUM_STARTPOINTS - 1)))
    ego_idx_range+=($idx)
done

# ── Count totals ─────────────────────────────────────────────────────────

n_groups=$((${#FOLLOW_COSTS[@]} * ${#SPEED_REWARDS[@]} * ${#CURVATURE_COSTS[@]} * ${#COLLISION_COSTS[@]} * ${#SAFETY_MARGINS[@]}))
jobs_per_group=$((${#OPP_RACELINES[@]} * ${#OPP_SPEED_SCALES[@]} * ${#ego_idx_range[@]}))

echo "Batch Data Collection (Multi-Parameter)"
echo "============================================="
echo "Map: $MAP_NAME"
echo "Ego raceline: $EGO_RACELINE"
echo "Opponent racelines: ${OPP_RACELINES[*]}"
echo "Speed scales: ${OPP_SPEED_SCALES[*]}"
echo "Start points: $NUM_STARTPOINTS"
echo "Jobs per group: $jobs_per_group"
echo "Parameter groups: $n_groups"
echo "Total jobs: $((n_groups * jobs_per_group))"
echo "Workers: $WORKERS"

# ── Run all parameter combinations ───────────────────────────────────────

gi=0
for fc in "${FOLLOW_COSTS[@]}"; do
for sr in "${SPEED_REWARDS[@]}"; do
for cc in "${CURVATURE_COSTS[@]}"; do
for colc in "${COLLISION_COSTS[@]}"; do
for sm in "${SAFETY_MARGINS[@]}"; do
    read -r sw sl <<< "$sm"
    ((gi++))

    PARAM_DIR="cw${fc}_${sr}_${cc}_${colc}_sm${sw}_${sl}"
    DATASET_DIR="Dataset_${MAP_NAME}/${PARAM_DIR}"
    SUCCESS_DIR="$DATASET_DIR/success"
    COLLISION_DIR="$DATASET_DIR/collision"
    mkdir -p "$SUCCESS_DIR" "$COLLISION_DIR"

    echo ""
    echo "[$gi/$n_groups] $DATASET_DIR"
    echo "  cost_weights: [$fc, $sr, $cc, $colc]  safety_margin: [$sw, $sl]"

    launched=0
    for opp_raceline in "${OPP_RACELINES[@]}"; do
        for opp_speed in "${OPP_SPEED_SCALES[@]}"; do
            for ego_idx in "${ego_idx_range[@]}"; do

                while [ $(jobs -r | wc -l) -ge $WORKERS ]; do
                    sleep 0.1
                done

                python demonstration.py \
                    --render \
                    --map_name "$MAP_NAME" \
                    --raceline "$EGO_RACELINE" \
                    --opp_raceline "$opp_raceline" \
                    --opp_speed_scale "$opp_speed" \
                    --ego_idx "$ego_idx" \
                    --interval_idx "$INTERVAL_IDX" \
                    --sim_duration "$SIM_DURATION" \
                    --cost_weights $fc $sr $cc $colc \
                    --safety_margin $sw $sl \
                    >/dev/null 2>&1 &

                ((launched++))
            done
        done
    done

    wait

    # Validate and move low quality
    LOW_QUALITY_DIR="$DATASET_DIR/low_quality"
    mkdir -p "$LOW_QUALITY_DIR"

    if [ -d "$SUCCESS_DIR" ] && ls "$SUCCESS_DIR"/*.csv >/dev/null 2>&1; then
        fail_files=$(python episode_validator.py --input_csv "$SUCCESS_DIR" 2>/dev/null | grep '^\[FAIL\]' | sed 's/\[FAIL\] //')
        while IFS= read -r fname; do
            [ -z "$fname" ] && continue
            base="${fname%.csv}"
            [ -f "$SUCCESS_DIR/$fname" ] && mv "$SUCCESS_DIR/$fname" "$LOW_QUALITY_DIR/"
            [ -f "$SUCCESS_DIR/${base}.mp4" ] && mv "$SUCCESS_DIR/${base}.mp4" "$LOW_QUALITY_DIR/"
        done <<< "$fail_files"
    fi

    # Stats
    s_total=0; s_follow=0; s_overtake=0; lq_total=0; c_total=0
    for csv in "$SUCCESS_DIR"/*.csv; do
        [ -f "$csv" ] || continue
        ((s_total++))
        [[ $(basename "$csv") == f_* ]] && ((s_follow++)) || ((s_overtake++))
    done
    for csv in "$LOW_QUALITY_DIR"/*.csv; do [ -f "$csv" ] && ((lq_total++)); done
    c_total=$(ls "$COLLISION_DIR"/*.json 2>/dev/null | wc -l)
    c_total=$((c_total + 0))

    echo "  Done: $launched jobs | Success: $s_total (F:$s_follow O:$s_overtake) Low quality: $lq_total Collision: $c_total"

done
done
done
done
done

echo ""
echo "============================================="
echo "All $n_groups groups done."
