#!/bin/bash
# train_eval.sh — train 32 models (4 model_type × 4 loading_type × 2 seeds) then evaluate each.
# Phase 1: parallel training, 2 concurrent trainings share the same GPU.
# Phase 2: sequential per-model eval; each model runs a segment grid with EVAL_WORKERS parallelism.
# Resume-safe: skip any training whose .pth already exists; skip any eval whose summary row already exists.
#
# Usage: nohup bash train_eval.sh > /dev/null 2>&1 &
#        tail -f Dataset_Austin/train_eval.log
# Stop:  pkill -f train_eval.sh && pkill -f train.py && pkill -f eval_multiagent.py
export DISPLAY=:1

# ── Shared parameters ────────────────────────────────────────────────────
MAP_NAME="Austin"
DATASET_DIR="Dataset_${MAP_NAME}"
HIDDEN_SCALE=4
NUM_REPEATS=2                        # _1 and _2 per (model_type, loading_type) to damp training noise
TRAIN_WORKERS=2                      # concurrent trainings on the same GPU
EVAL_WORKERS=8                       # segment-level parallelism inside each per-model eval
NOISE=0.0
RENDER=true                         # batch mode: no MP4s (~19200 videos otherwise)
PRETRAINED_DIR="pretrained"

# best_200 excluded — SequenceDataset eager-loads CSVs on CPU and would OOM.
MODEL_TYPES=(base dual_head deep deep_dual)
LOADING_TYPES=(best_group follow_first overtake_first merge)

# ── Eval grid (matches evaluate.sh defaults) ─────────────────────────────
EGO_RACELINE="raceline1"
OPP_RACELINES=("raceline0" "raceline1" "raceline2")
OPP_SPEED_SCALES=(0.5 0.6 0.7 0.8)
INTERVAL_IDX=15
SIM_DURATION=8.0
NUM_STARTPOINTS=50

# ── Generate ego indices ──────────────────────────────────────────────────
raceline_path="f1tenth_racetracks/${MAP_NAME}/${EGO_RACELINE}.csv"
max_waypoints=$(tail -n +3 "$raceline_path" | wc -l)
ego_idx_range=()
for ((i=0; i<NUM_STARTPOINTS; i++)); do
    idx=$((i * max_waypoints / (NUM_STARTPOINTS - 1)))
    ego_idx_range+=($idx)
done
segments_per_model=$((${#ego_idx_range[@]} * ${#OPP_RACELINES[@]} * ${#OPP_SPEED_SCALES[@]}))

# ── Logging helpers ──────────────────────────────────────────────────────
mkdir -p "$PRETRAINED_DIR" "$DATASET_DIR"
LOG_FILE="${DATASET_DIR}/train_eval.log"
SUMMARY_CSV="${DATASET_DIR}/train_eval_summary.csv"
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"; }

# Summary header — only write if file is new (preserves across resumes).
if [ ! -s "$SUMMARY_CSV" ]; then
    echo "model_name,following,overtaking,collision,error,total,eval_secs" > "$SUMMARY_CSV"
fi

total_models=$((${#MODEL_TYPES[@]} * ${#LOADING_TYPES[@]} * NUM_REPEATS))

# ── Phase 1: Training (parallel, TRAIN_WORKERS at a time) ────────────────
log "Phase 1: Training ${total_models} models  |  concurrent=${TRAIN_WORKERS}"
log "============================================="
log "Map: $MAP_NAME  |  Dataset: $DATASET_DIR  |  hidden_scale=$HIDDEN_SCALE"
log "Model types:   ${MODEL_TYPES[*]}"
log "Loading types: ${LOADING_TYPES[*]}  (best_200 excluded)"

start_phase1=$(date +%s)
t_idx=0
for mt in "${MODEL_TYPES[@]}"; do
    for lt in "${LOADING_TYPES[@]}"; do
        for ((seq=1; seq<=NUM_REPEATS; seq++)); do
            ((t_idx++))
            model_name="${mt}_${lt}_${HIDDEN_SCALE}_${seq}"
            model_path="${PRETRAINED_DIR}/${model_name}.pth"

            if [ -f "$model_path" ]; then
                log "[$t_idx/$total_models] SKIP (exists): $model_name"
                continue
            fi

            while [ "$(jobs -r | wc -l)" -ge $TRAIN_WORKERS ]; do
                sleep 1
            done

            log "[$t_idx/$total_models] START: $model_name"
            (
                t0=$(date +%s)
                python train.py \
                    --dataset_dir "$DATASET_DIR" \
                    --loading_type "$lt" \
                    --model_type "$mt" \
                    --hidden_scale "$HIDDEN_SCALE" \
                    --model_path "$model_path" \
                    >> "$LOG_FILE" 2>&1
                rc=$?
                elapsed=$(( $(date +%s) - t0 ))
                if [ $rc -eq 0 ] && [ -f "$model_path" ]; then
                    echo "$(date '+%m-%d %H:%M:%S') DONE: $model_name (${elapsed}s)" >> "$LOG_FILE"
                else
                    echo "$(date '+%m-%d %H:%M:%S') FAIL rc=$rc: $model_name (${elapsed}s)" >> "$LOG_FILE"
                fi
            ) &
        done
    done
done
wait

log ""
log "Phase 1 done in $(( $(date +%s) - start_phase1 ))s"

# ── Phase 2: Per-model evaluation (segment grid parallel via EVAL_WORKERS) ──
log ""
log "Phase 2: Evaluating up to ${total_models} models × ${segments_per_model} segments  |  concurrent=${EVAL_WORKERS}"
log "============================================="

start_phase2=$(date +%s)
e_idx=0
for mt in "${MODEL_TYPES[@]}"; do
    for lt in "${LOADING_TYPES[@]}"; do
        for ((seq=1; seq<=NUM_REPEATS; seq++)); do
            ((e_idx++))
            model_name="${mt}_${lt}_${HIDDEN_SCALE}_${seq}"
            model_path="${PRETRAINED_DIR}/${model_name}.pth"

            if [ ! -f "$model_path" ]; then
                log "[$e_idx/$total_models] SKIP (no model): $model_name"
                continue
            fi
            if grep -q "^${model_name}," "$SUMMARY_CSV"; then
                log "[$e_idx/$total_models] SKIP (done): $model_name"
                continue
            fi

            log "[$e_idx/$total_models] EVAL: $model_name"
            t0=$(date +%s)
            temp_dir=$(mktemp -d)
            job_id=0

            for ego_idx in "${ego_idx_range[@]}"; do
                for opp_raceline in "${OPP_RACELINES[@]}"; do
                    for speed_scale in "${OPP_SPEED_SCALES[@]}"; do
                        cmd="python eval_multiagent.py --model_path $model_path --model_type $mt --map_name $MAP_NAME --ego_idx $ego_idx --interval_idx $INTERVAL_IDX --ego_raceline $EGO_RACELINE --opp_raceline $opp_raceline --opp_speedscale $speed_scale --sim_duration $SIM_DURATION --hidden_scale $HIDDEN_SCALE --noise $NOISE"
                        [ "$RENDER" = true ] && cmd="$cmd --render"

                        while [ "$(jobs -r | wc -l)" -ge $EVAL_WORKERS ]; do
                            sleep 0.1
                        done

                        (eval "$cmd" >/dev/null 2>&1; echo $? > "$temp_dir/$job_id") &
                        ((job_id++))
                    done
                done
            done
            wait

            # Aggregate exit codes (1=follow, 2=overtake, 3=collision, *=error)
            f_count=0; o_count=0; c_count=0; err_count=0
            for result_file in "$temp_dir"/*; do
                [ -f "$result_file" ] || continue
                case $(cat "$result_file") in
                    1) ((f_count++)) ;;
                    2) ((o_count++)) ;;
                    3) ((c_count++)) ;;
                    *) ((err_count++)) ;;
                esac
            done
            rm -rf "$temp_dir"
            total=$((f_count + o_count + c_count + err_count))
            elapsed=$(( $(date +%s) - t0 ))

            log "  ${elapsed}s  f=$f_count  o=$o_count  c=$c_count  err=$err_count  (total=$total)"
            echo "$model_name,$f_count,$o_count,$c_count,$err_count,$total,$elapsed" >> "$SUMMARY_CSV"
        done
    done
done

log ""
log "Phase 2 done in $(( $(date +%s) - start_phase2 ))s"

# ── Summary table ────────────────────────────────────────────────────────
log ""
log "============================================="
log "All done. Summary → $SUMMARY_CSV"
column -t -s, "$SUMMARY_CSV" | tee -a "$LOG_FILE"
