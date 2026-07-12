# D2R-G Deployable Spatiotemporal Geometry Representation — Locked Specification

Version: `d2r-g-spec-1`  
Date: 2026-07-11 (Asia/Singapore)  
Authority: `CURRENT_HANDOFF.md` §§13–16 and the B+ design D2 gate.

## 1. Purpose and boundary

D2R-G is a separately locked representation redesign, not an unregistered
fifth family in the closed D2 search. D2 showed that deployable temporal T2
passes every collision-classification sub-gate but fails `TTC<2` MAE at
1.068 seconds versus the frozen 0.300-second requirement. D2.5 then showed
that the bounded macro-residual action space is feasible in 67/91 non-test
ego-collision cases. The unresolved question is whether deployable causal
observations can encode opponent geometry and timing accurately enough to
choose actions.

D2R-G tests exactly one new family: a beam-local spatiotemporal LiDAR encoder
with privileged geometry labels used only as training targets. It does not
train an actor, run PPO, open the D2 test, change the D2 outcome labels, or
weaken any original D2 gate.

The historical `probe_side_rear_risk.py` is not D2R evidence: it collects a
different online distribution and uses a random frame split that leaks
episodes. It is not executed or used for a decision.

## 2. Frozen data, split, and registry

Use the exact D2 non-test release `non_test_full_20260711_175713`, deployable
signal release `deployable_signals_20260711_182229`, and existing five-outer,
three-inner map-stratified L4 folds. The 1,108-episode grouped D2 test and its
seal remain unchanged and unopened.

Before any D2R fitting, append exactly 1,928 stage `D2R-G`, use-class
`probe_fit`, decision-effect `representation_choice`, `final_pool=false`
rows to the canonical append-only registry. Reuse is recorded as a new stage;
existing D2/D2.5 rows are never rewritten. The stage opening timestamp and
canonical evidence root are `2026-07-11T21:00:00+08:00` and
`logs/d2r_geometry_20260711`, frozen before the append.

All feature construction, normalization, sampling, calibration, thresholds,
and model fitting use only the relevant fit/inner folds. L1/L2/L3/L4 groups
never cross a fold. No branch outcome or witness identity enters D2R-G.

## 3. Deployable causal inputs

For frame `t`, use offsets `(0, 5, 10, 20, 35, 50, 75, 100)` simulator
frames, corresponding to 0–1.0 seconds of causal history. At episode start,
each unavailable tap clamps to that episode's first frame.

Inputs are:

- eight 360-beam ego LiDAR scans, clipped to `[0,30]` m and divided by 30;
- ego actual speed at all eight taps, divided by 10;
- previous desired steering at all eight taps, divided by 0.52;
- previous desired speed at all eight taps, divided by 10;
- the current frozen BC recurrent feature, standardized using fit-fold mean
  and standard deviation only.

Every input is available to the deployed policy. Ego/opponent poses,
progress, map identity, raceline, opponent speed, collision outcome, future
frames, D2.5 branch result, and oracle geometry are forbidden inputs.

## 4. Single locked architecture

Treat the eight LiDAR taps as input channels over the circular beam axis:

1. circular `Conv1d(8,32,kernel=9)`, SiLU;
2. circular `Conv1d(32,32,kernel=7)`, SiLU;
3. circular `Conv1d(32,32,kernel=5)`, SiLU;
4. adaptive average pool to 18 beam bins, yielding 576 values;
5. frozen-feature projection `Linear(1680,128)`, LayerNorm, SiLU;
6. scalar-history projection `Linear(24,32)`, SiLU;
7. concatenate 576+128+32, then `Linear(736,128)`, LayerNorm, SiLU.

The fixed heads are:

- six collision-horizon logits matching D2 exactly;
- signed relative-progress estimate, clipped target `[-10,10]` m;
- lateral-gap estimate, clipped target `[0,2]` m;
- signed closing-rate estimate, clipped target `[-5,5]` m/s;
- 50-way categorical TTC logits for bins `[0.0,0.1),...,[4.9,5.0]`.

TTC prediction is the posterior expected bin center, with the last center
4.95 seconds. There is no architecture, width, tap, or output-decoding sweep.

## 5. Locked fitting

Use seed `20260711`, AdamW, learning rate `5e-4`, weight decay `1e-4`, batch
size 512, and six epochs with no early stopping. Retain every frame with a
2-second collision target or ground-truth `TTC<2`, plus every twentieth
remaining fit-fold frame. Apply inverse-probability weight 20 to sampled
background so the loss estimates the full fit-fold distribution.

Loss is the weighted sum:

- collision BCE: `1.0`;
- TTC cross entropy: `1.0`, with an additional 25x multiplier on `TTC<2`;
- relative-progress SmoothL1: `0.25`;
- lateral-gap SmoothL1: `0.25`;
- closing-rate SmoothL1: `0.50`.

Classification-head fit-fold biases use fit-fold prevalence. Inner-OOF
Platt calibration and safe-episode thresholds follow D2 exactly. No fold may
use held-out prevalence, normalization, stopping, or thresholds.

## 6. OOF gate and sealed test

Run the same five outer and three inner L4-grouped procedure as D2. D2R-G
passes non-test OOF only if all original conditions are true:

- 1-second ego event recall `>=0.60` at safe-episode FA `<=0.10`;
- 2-second ego event recall `>=0.40` at safe-episode FA `<=0.10`;
- 1-second ego Brier skill score `>=0.10`;
- `TTC<2` MAE `<=0.30` seconds;
- at least 30 independent outer-held-out ego-collision episodes.

Also report, but do not substitute for the gate: relative-progress, lateral-
gap, and closing-rate MAE; TTC 0.25/0.5-second bin slices; map/skill/raceline/
speedscale slices; earliest warning; calibration; and seed determinism.

If OOF fails, stop with the D2 test sealed and do not change this family. If
OOF passes, fit once on all non-test blocks, freeze thresholds, create
`TEST_OPENING_STARTED`, append test registry rows, and open the existing test
once. The identical complete gate applies. A test failure cannot trigger a
new architecture or threshold.

## 7. Relationship to D2.5 and PPO

D2.5 witness/action outputs are reserved for a later, separately locked
action-ranking/distillation stage only after D2R-G passes. They are not used
to fit or select D2R-G. D3/PPO remains blocked until D2R-G passes its complete
OOF and once-opened test gate and any required Route-R2 distillation gate is
separately satisfied.

## 8. Stop and release rules

Stop on any source/model/split/registry hash drift, test identifier access,
history crossing an episode boundary, non-circular beam padding, fold
leakage, nonfinite loss/prediction, seed nondeterminism, incomplete OOF row,
manifest mismatch, or failed independent validation.

Release source hashes, registry snapshot, exact sampled counts/weights, all
inner/outer reports, model bundles, OOF predictions, slice tables, gate
report, independent validation, output manifest, and `COMPLETE` last. Failed
or superseded smokes remain preserved in fresh directories.
