# D2 Episode-Held-Out Representation Probe — Locked Specification

Version: `d2-spec-1`  
Date: 2026-07-11 (Asia/Singapore)  
Parent authority: `CURRENT_HANDOFF.md` §§13–14 and
`2026-07-10-ppo-safety-first-bplus-design.md` §6.2.

## 1. Causal question and boundary

D2 asks whether the frozen BC actor feature available at deployment predicts
actionable collision/TTC risk on unseen canonical episodes and neighboring
start-region blocks. D2 performs no PPO training and cannot select a PPO
checkpoint, reward, curriculum, or final-evaluation scenario.

The primary data population is the D0.1 BC-only Primary estimand: exactly
3,036 L2 scenarios in the reviewed release
`d01_full_reconcile_20260711_170200_a`. Candidate-policy trajectories are not
probe examples. Every L2 uses its lexicographically selected D0.1
`representative_l1_id`; duplicate L1 occurrences are provenance only.

## 2. Frozen inputs and provenance

- BC checkpoint: `pretrained/end2race.pth`, SHA256
  `b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4`.
- D0.1 output manifest SHA256:
  `425d62097b1463e72fca33f4e08690385bfbd21e6be3a91db900b92e4664bd89`.
- D0.1 summary SHA256:
  `56c9dcdc4af24afdd8b0f69a10e9b71487c75d23466bdde28a5090a214f92505`.
- D0.1 S0 manifest, canonical table, occurrence table, block table, NPZ
  hashes, raceline hashes, and live append-only registry must rehash before
  extraction.
- Runtime is the pinned `end2race` Python environment. Feature extraction is
  remote-only on `haowei@192.168.2.127`; reviewed manifests/reports are
  mirrored locally by explicit allowlist.

## 3. Outcome-blind split lock

The split generator is allowed to read only these projected D0 fields:
`model_id`, `l2_id`, `l3_id`, `l4_id`, `map_name`, `ego_raceline`,
`opponent_raceline`, `speedscale_hex`, `interval_idx`, `skill`, and
`representative_l1_id`. Split assignment must be invariant when every outcome,
collision, phase, and source-directory label is poisoned.

Use domain-separated SHA256 with seed `20260711`:

- test rank domain: `end2race:d2:test-rank:v1`;
- outer-fold domain: `end2race:d2:outer-fold:v1`;
- inner-fold domain: `end2race:d2:inner-fold:v1`.

Within each map, sort L4 IDs by the test-rank digest and reserve
`ceil(0.35 * number_of_map_L4_blocks)` blocks. This freezes 34 of 97 Austin
blocks and 18 of 49 blocks on each cross map, 88 test blocks total. There is
no outcome-based retry: fewer than 30 test ego-collision episodes is a D2 test
failure.

All L2/L3 variants in an L4 block share one split. The remaining 156 L4
blocks receive five map-stratified outer folds using deterministic greedy
scenario-count balancing, with the outer digest as the only tie-break. For
each outer fold, its training blocks receive three independently derived,
map-stratified inner folds using the same balancing rule and an outer-fold-
specific inner domain.

The public split manifest contains no outcomes and no outcome-bearing NPZ
paths. Before branch selection only non-test source locators may be emitted.
The test seal records the test-ID digest, split-config digest, and counts, but
not labels or features.

## 4. Exact causal feature replay

The deployed feature is the 1,680-dimensional output of
`End2Race.forward_features` at each archived 100 Hz pre-step frame.

For every episode:

1. set the model to `eval`, freeze all parameters, and reset the GRU hidden
   state to exact zeros;
2. replay one frame at a time with batch size 1; sequence batching is
   forbidden because it is not bit-identical to archived GPU inference;
3. use the archived 360-beam ego LiDAR at frame `t`;
4. use raceline waypoint speed `* 0.9` at `t=0`, then
   `ego_actual_speed[t-1]` for every `t>0` (the evaluator's one-frame lag);
5. save the float32 GRU output and derive the BC action with the frozen output
   layer;
6. require bit equality against archived desired speed and clipped desired
   steering at every frame. Any mismatch, non-finite value, source-hash
   mismatch, unexpected length, or hidden-state carryover is a hard stop.

No opponent pose, Frenet coordinate, collision flag, map phase, target, or
critic feature enters the probe input. Privileged geometry is label-only.

## 5. Label contract

All labels are evaluated at archived pre-step time `t`. The terminal event is
at `final_time`. The D0.1 open-chain chord-sum length and whole-series integer
branch alignment are reused; alignment is label construction, not actor input.

For each pose, project velocity onto the local `raceline1` tangent:

`v_s = actual_speed * cos(vehicle_heading - reference_tangent)`.

The signed current closing-rate target is `ego_v_s - opponent_v_s` in m/s.

Corridor TTC uses constants frozen from the PPO geometry:

- vehicle length `0.58 m`;
- vehicle width `0.31 m`;
- lateral margin `0.20 m`;
- corridor condition `abs(ego_d - opponent_d) <= 0.51 m`;
- cap `5.0 s`.

When the ego is behind (`rel_s < 0`), is closing
(`ego_v_s - opponent_v_s > 0`), and is in the corridor:

`TTC = min(5.0, max(0, -rel_s - 0.58) / closing_rate)`.

Otherwise TTC is `5.0 s`. Exact contact yields zero. TTC is a privileged
diagnostic target and is never a deployment input.

Binary targets are ego collision and any-agent collision within 0.5, 1.0,
and 2.0 seconds. A matching terminal collision is positive when
`0 < final_time - t <= horizon`. A frame is negative only when the complete
horizon is observed without that event. If the horizon crosses a normal
episode end or a nonmatching terminal/competing collision, it is censored,
not negative. A 3-second view may be reported as diagnostic but cannot affect
selection.

## 6. Probe families and training

The locked sequence is:

1. prevalence baseline;
2. linear multi-task head on the frozen feature;
3. MLP multi-task head: `1680 -> 128 -> ReLU -> 8`;
4. causal short-history module only if the MLP fails outer OOF gates.

The temporal branch has two ordered, capacity-matched implementations. T1 is
tested first: frozen-feature taps at frames `[t, t-10, t-25, t-50]`, padded
only with the same episode's first frame, followed by `6720 -> 32 -> 8`
(approximately the same parameter count as `1680 -> 128 -> 8`). If T1 fails,
T2 is the final temporal alternative and no third architecture is tried. T2
uses the current frozen feature plus deployable history: 360-beam LiDAR
deltas at 0.10/0.25/0.50 seconds, actual-speed taps and previous-command taps
at 0/0.10/0.25/0.50 seconds. Its `2772 -> 77 -> 8` head is likewise
capacity-matched. T2 may run only after a fresh non-test-only signal sidecar
passes source hashes, episode/frame alignment, registry, causal-padding, and
test-exclusion checks.

The six binary heads use masked BCE-with-logits. The continuous heads predict
closing rate and capped TTC with masked smooth-L1 losses. Loss weights,
optimizer, epoch count, early-stopping rule, and deterministic frame sampler
are implementation-manifest fields and are frozen before any OOF metric is
computed. All positive/corridor-critical frames are retained; background
frames may be deterministically thinned for fitting, but every valid frame is
used for validation/test prediction and episode metrics.

For each of five outer folds, models train only on its outer-training blocks.
Three inner grouped folds provide out-of-fold calibration predictions and
select per-horizon thresholds. The outer model is refit on all outer-training
blocks and evaluated once on the outer fold using only inner-selected
thresholds. Pooled outer predictions decide the model family. Normalization,
class weighting, sampling, stopping, and thresholds never use the outer fold.

After family selection, fit that family on all non-test blocks. Its final
thresholds come only from non-test grouped OOF predictions. Create a durable
`TEST_OPENING_STARTED` marker and append test rows to `opened_registry.tsv`
before reading any test NPZ, label, or feature. Open the test once. A test
failure cannot trigger a family switch, threshold change, or resplit.

## 7. Metrics and exact alarm semantics

Report AUCPR, Brier score, Brier skill score, 10-bin equal-width ECE,
reliability tables, and frame recall for every binary head. Also report map,
skill, opponent raceline, and speedscale slices.

For one horizon/head, an episode alarms if any valid frame probability is at
or above the frozen threshold. A safe episode is one with no any-agent
collision. Safe-episode false-alarm rate is the fraction of safe episodes
that alarm anywhere in the episode. Ego-collision event recall is the fraction
of ego-collision episodes with at least one alarm in the corresponding
pre-impact horizon. Threshold selection maximizes event recall subject to
safe-episode false-alarm rate `<= 0.10`; ties choose the higher threshold.
Report the earliest-warning lead-time distribution and the fraction warned at
least 0.5/1.0/2.0 seconds before impact separately.

TTC MAE is computed over all valid frames with ground-truth `TTC < 2.0 s`.
The prevalence Brier reference is fit-set prevalence for fold evaluation and
all-non-test prevalence for the final test.

## 8. Branch gate and stop rules

A family passes outer OOF only if all are true:

- 1-second ego event recall `>= 0.60` at safe-episode FA `<= 0.10`;
- 2-second ego event recall `>= 0.40` at safe-episode FA `<= 0.10`;
- 1-second ego Brier skill score `>= 0.10`;
- TTC MAE for `TTC < 2 s` `<= 0.30 s`;
- at least 30 independent outer-held-out ego-collision episodes contribute.

If the MLP passes, select frozen-feature B+. If it fails, evaluate the causal
temporal family under the same nested procedure. If temporal passes, select
it. If both fail, stop D3 and revisit observations/actions.

The selected family must pass the identical gate on the once-opened test,
including at least 30 test ego-collision episodes. Otherwise stop D3. D2.5 may
still run as an action-space diagnostic but cannot authorize a deployable D3
actor from a failed representation branch.

## 9. Required release artifacts

- frozen config and all source/model hashes;
- outcome-blind split manifest, non-test source manifest, test seal, and fold
  accounting;
- feature/label storage manifests and extraction validation;
- replay bit-equality report;
- append-only registry snapshots before non-test fitting and after test open;
- inner calibration, outer OOF, family comparison, thresholds, slice metrics,
  calibration tables, and deterministic rerun checks;
- selected-family checkpoint and hash;
- one-open test report or a precise stop report;
- `COMPLETE` written last after fsync and independent validation.

Any missing artifact, split overlap, registry mismatch, test-open ambiguity,
non-deterministic split, replay mismatch, or post-test model/threshold change
invalidates the D2 release.
