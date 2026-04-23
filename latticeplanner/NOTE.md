# NOTE — Changes Since Code Release (`c711a0f`)

This note lists the concrete modifications, new features, and tooling added on top of
the original public code release. Baseline = commit `c711a0f "code release"`;
HEAD (as of this note) = `e85bd9f`.

Scope covers the planner, data-collection pipeline, dataset validator, and the training/eval
stack. The document is placed under `latticeplanner/` because the planner is the
root cause of most downstream changes (cost functions, sampling, CLI).

---

## 1. Lattice Planner (`latticeplanner/`)

### 1.1 New collision-cost functions in `lattice_planner.py`
Three selectable collision-cost modes (switched by commenting lines in `LatticePlanner.eval()`):

| Option | Function | Semantics |
|---|---|---|
| 1 | `get_obstacle_collision_with_v` (original, modified) | Static OBB against frozen opponent pose. Now accepts `safety_w`, `safety_l` as arguments (previously hardcoded to 0.15 / 0.20). |
| 2 | `get_obstacle_collision_mixed_ttc_obb` (new) | Weighted blend `0.8·static + 0.2·TTC`. |
| 3 | `get_obstacle_collision_merged_ttc_obb` (new, **default**) | OR-fusion `max(static, TTC)` — either signal fires full cost, no dilution. Raw range 15–25 when colliding. |

The TTC branch is extracted into shared helpers:
- `_ttc_opp_setup` — derives opponent velocity and current arc length on its raceline.
- `_ttc_core` — for each `(trajectory, velocity)` pair predicts opponent position at
  arrival time `t_j` (raceline interpolation when waypoints + speed are available,
  else linear extrapolation) and computes
  `cost = 20 · (1 − min_ttc / ittc_thres)`.

TTC requires opponent waypoints. `LatticePlanner` now holds `self.opp_waypoints`
(initialized `None`); `demonstration.py` wires it via
`ego_planner.opp_waypoints = opp_planner.waypoints`.

### 1.2 Configurable safety margins
`LatticePlanner.__init__` adds `self.safety_w = 0.03`, `self.safety_l = 0.04`;
both are overwritten from the `--safety_margin` CLI arg in `demonstration.py`.
All three collision functions forward these margins into `get_vertices`.

### 1.3 Sampling / config tuning
- `sample_lookahead_square` default widths: `linspace(-1.25, 1.25, 11)` → `linspace(-1.05, 1.05, 11)` (narrower grid, tighter to the raceline).
- `lattice_config.yaml`: `lh_grid_ub 1.0 → 1.2`, `ittc_thres 2.0 → 1.0`.

### 1.4 `utils.py`
New `interp_raceline(waypoints, s_arr, s_max, s_query)` — numba-jitted arc-length
interpolation with wrap-around, used by the TTC cost.

---

## 2. Data Collection (`demonstration.py`, `collect.sh`)

### 2.1 CLI-driven planner parameters
`demonstration.py` previously hardcoded cost weights and safety margins inside
`setup_ego_planner`. New CLI surface:

| Argument | Default | Meaning |
|---|---|---|
| `--cost_weights` | `0.05 2.3 0.3 0.5` | `[follow, speed, curvature, collision]` |
| `--safety_margin` | `0.03 0.04` | `[safety_w, safety_l]` in meters |
| `--ego_idx` | `42` (was `0`) | |
| `--opp_speed_scale` | `0.7` (was `0.8`) | |

`setup_ego_planner` now takes `args` directly and forwards margins + weights to the
planner. Opponent planner `speed_reward` tuned `2.0 → 1.8`.

### 2.2 Dataset directory layout
Old: `Dataset_{MAP}_{MMDD}/…` (timestamped, not reproducible across sweeps).
New: `Dataset_{MAP}/cw{wf}_{ws}_{wc}_{wcol}_sm{sw}_{sl}/{success,collision}/…`
Directory name deterministically encodes the planner config used for the run, so
parameter sweeps produce stable, re-runnable paths.

Final-state labels renamed `"overtaking" → "overtake"`, `"following" → "follow"`;
filename prefix stays `o_` / `f_`. Render overlay now reports `ego_state`.

### 2.3 `collect.sh` — batch driver
Rewritten into a unified runner:
- `MULTIPARAMETERS=false` → runs defaults.
- `MULTIPARAMETERS=true` → sweeps the grid
  `FOLLOW_COSTS × SPEED_REWARDS × CURVATURE_COSTS × COLLISION_COSTS × SAFETY_MARGINS`
  (~357 parameter groups × ~600 segments each on Austin).
- Per-group parallelism via `WORKERS=6` backgrounding `demonstration.py`.
- Crash/resume resilience: progress persisted to `Dataset_{MAP}/progress.txt`.

---

## 3. Episode Validator (`episode_validator.py`, new)

Did not exist in `c711a0f`. Post-hoc classifies each 8-second episode CSV and
reorganizes `success/` vs. `low_quality/`.

### 3.1 Fail rules (OR-merged into two categories)
1. **Proximity** — any of 8 LiDAR sectors reaches `surface_dist < 0.15 m`.
   - Calibrated against f1tenth_gym: car body 0.58 × 0.31 m, LiDAR at geometric
     center, `half_l = 0.29`, `half_w = 0.155`. Beam mapping
     `angle_i = -π + i·(2π/N)` (beam 0 = rear, 180 = front, 90 = right, 270 = left).
2. **Steering**
   - Oscillation: > 6 reversals of amplitude ≥ 0.3 rad within any 1 s window.
   - Jump: single-step `|Δsteer| > 0.6 rad`.
   - Sawtooth: lag-1 autocorrelation < −0.5 (only if `var(steer) ≥ 0.09`).

Status ∈ { `PASS`, `FAIL (proximity)`, `FAIL (steering)`, `FAIL (proximity + steering)` }.

### 3.2 Metrics retained
`global_min_surface_dist`, `danger_sectors` (dict), `max_steer_reversals`,
`steer_autocorr_lag1`. Removed: `speed_*`, `side/front_min_surface_dist`,
`n_frames`, `duration` (speed in the CSV is planner output, not vehicle behavior).

### 3.3 Modes

| Flag | Behavior |
|---|---|
| `--input_csv <path>` | Validate one CSV, print, exit 0/1. |
| `--input_dir <cw_dir>` | Revalidate `success/ ∪ low_quality/`, reclassify files, write `fails.csv` + `success.csv`. Idempotent. |
| `--multidataset_dir <root>` | Parallel (workers=8) over every `cw*/` subdir, aggregate per-group logs into `<root>/validate.log`, then emit 5 dataset manifests at root. |

BLAS threads are capped to 1 before `import numpy` to prevent oversubscription in the pool.

### 3.4 Manifests (for `train.py` consumption)
Each is a CSV of relative paths.

| Manifest | Selection rule | Typical size (Austin) |
|---|---|---|
| `manifest_best_group.csv` | Single `cw*/success/` with the largest PASS count | ~561 |
| `manifest_best_200.csv` | Union of top-200 `cw*/` by PASS count (all their PASS CSVs) | ~107k |
| `manifest_follow_first.csv` | Per segment: best `f_*` else best `o_*` | ~592 |
| `manifest_overtake_first.csv` | Per segment: best `o_*` else best `f_*` | ~592 |
| `manifest_merge.csv` | Per segment: best `f_*` **and** best `o_*` (up to 2) | ~784 |

Per-segment score (proximity not used — validator already gated it):
```
score = steer_autocorr_lag1 − SCORE_BETA · (max_steer_reversals / STEER_MAX_REVERSALS)
SCORE_BETA = 0.5
```
Segment key = `(opp_raceline, ego_idx, opp_speed)`. On Austin: 3 × 50 × 4 = 600 unique segments.

---

## 4. Training Stack (`train.py`, `model.py`, `eval_*.py`)

### 4.1 `train.py` — manifest loading path
Two mutually exclusive paths share the same `SequenceDataset`:
- Legacy: `--data_path <success_dir>` (glob one directory, backwards-compatible).
- New: `--dataset_dir <root> --loading_type {best_group,best_200,follow_first,overtake_first,merge}`
  reads the corresponding `manifest_<type>.csv`. Takes precedence when set.

`SequenceDataset` is CPU-eager-loaded; `best_200` (~107k CSVs) OOMs on typical RAM
and needs a streaming loader before use. The other four manifests (~600–800 CSVs) fit.

### 4.2 `model.py` — 4 model variants (`--model_type`)

All share lidar preprocessing (per-beam learnable sigmoid `self.k`, 360 params),
`speed_mlp` + `dummy_embedding` with `mask_prob` regularization.

| Variant | Change vs baseline | Params | Hidden state |
|---|---|---|---|
| `base` (`End2Race`) | — | 11.3M | `tensor[1, B, 1680]` |
| `dual_head` | Shared output head → separate `head_steer` / `head_speed` | 12.0M | `tensor[1, B, 1680]` |
| `deep` | 1-layer GRU → two stacked GRUs (`gru1` at hidden_scale, `gru2` fixed at scale 2 → 840) | 17.3M | tuple `(h1, h2)` |
| `deep_dual` | Both changes combined | 17.7M | tuple `(h1, h2)` |

Deep variants are controlled by class constant `SECOND_LAYER_SCALE = 2`.

### 4.3 `eval_singleagent.py` / `eval_multiagent.py`
Accept `--model_type`; detect tuple hidden state via `hasattr(model, 'gru1')` and
init `(h1, h2)` accordingly. `evaluate.sh` gained a multi-startpoint segment grid;
its top-of-file constants (`MODEL_PATH`, `MODEL_TYPE`, `NUM_STARTPOINTS`,
`OPP_RACELINES`, …) are edited directly to switch models.

---

## 5. Known Quirks / Caveats

1. **Rear-direction opponent invisible in `f1tenth_gym`** — `get_blocked_view_indices`
   in `laser_models.py` doesn't wrap ±π. Wall returns unaffected; validator unaffected.
2. **`smoke_test.sh` is stale** — still uses retired flags `--dataset_loader` /
   `--model_variant`. Will fail at argparse; rewrite before reuse.
3. **`best_200` OOMs on CPU** with `SequenceDataset`. Use the other four manifests
   until a streaming dataset exists.
4. **Validator has no fallbacks** for degenerate inputs (short episodes, constant
   steering, missing paths) — raises directly.
5. **`mask_prob` is a regularizer, not conditional inference** — at train time 10%
   of timesteps replace the speed embedding with a learnable `dummy_embedding` token.
6. **Collision method is not a CLI flag** — switched by commenting lines in
   `LatticePlanner.eval()`. Default is `merged`.


