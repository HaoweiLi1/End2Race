# End2Race PPO Handoff

Generated: 2026-07-05 13:31 +0800  
Repo: `/home/haowei/Documents/End2Race`  
Base git commit: `bae93dd` (`Fix lattice utils`)  
Remote machine: `haowei@192.168.2.127:~/Documents/End2Race`

This handoff is for the next agent continuing the End2Race post-BC PPO work.

## User Goal

Primary PPO goal:

- Start from the BC imitation model.
- Use PPO post-RL fine-tuning to further reduce collision rate.
- Secondary goal: improve overtake rate.
- Smoothness is not a primary KPI.
- The practical failure the user cares about is unsafe overtaking/merge behavior, especially collision during/after passing.

Current narrowed scope:

- Single raceline / OL1 pipeline first.
- Keep training/evaluation distribution matched on OL1 before expanding.
- Use three-grid OL1 eval for robustness:
  - canonical
  - off21
  - off42

## Current Worktree State

`git status --short` currently shows:

```text
 M analyze_collisions.py
 M ppo_utils.py
 M train_ppo.py
?? pretrained/end2race_anchor_baseline_local_seed0_20260705_113917.pth
?? pretrained/end2race_anchor_baseline_local_seed0_20260705_113917_full.pt
```

Meaning:

- Source modifications are limited to PPO training/logging/collision analysis.
- The untracked checkpoints are from the local anchor baseline experiment and can be archived or ignored.
- Do not revert these changes unless explicitly asked.

Important current code additions:

- `train_ppo.py`
  - `--freeze_speed` D1 composite policy already exists.
  - D1-b curriculum args:
    - `--lateral_offset_prob`
    - `--lateral_offset_min`
    - `--lateral_offset_max`
  - `_reset_sim_with_offset()` spawns ego laterally offset along raceline normal, with lidar clearance rejection.
  - Rollout metrics include `alongside_frac` and `alongside_lat_gap`.
  - Training log print uses `flush=True`; this fixes local startup-guard false positives caused by Python block buffering.
- `ppo_utils.py`
  - logs `loff`, `along`, `alat`
  - logs anchor split `sanc` and `vanc`
- `analyze_collisions.py`
  - collision phase now has `pre / alongside / post`
  - `alongside` is final `|rel_s| < --alongside_thresh`, default `0.6`

## Environment Notes

Local:

- Do not explicitly set `DISPLAY`; use the default local environment.
- Previous local D1-b "hang" was not a real f110/display hang.
- Root cause was log buffering: redirected Python output did not flush `iter=` lines before the startup guard fired.
- Fix already applied: `print(..., flush=True)` in `train_ppo.py`.
- Verification logs: `logs/local_displayless_verify_20260705/`

Remote:

- Use `DISPLAY=:1`.
- Remote path: `haowei@192.168.2.127:~/Documents/End2Race`
- For unattended runs, preserve `status.tsv` and `master.log`.

## Baselines

Three-grid BC OL1 reference:

| grid | collision | overtake |
|---|---:|---:|
| canonical | 5/200 = 2.5% | 3/200 = 1.5% |
| off21 | 12/200 = 6.0% | 0/200 = 0.0% |
| off42 | 7/200 = 3.5% | 5/200 = 2.5% |
| pooled | 24/600 = 4.00% | 8/600 = 1.33% |

Pooled BC Wilson 95% CI for collision: `[2.70%, 5.88%]`.

Use matched-grid comparisons. Do not compare single-grid numbers across different grid offsets as if they are equivalent.

## Experiment History And Conclusions

### 1. Early PPO Audits

Key issues found in the original PPO pipeline:

- Training samples too little of the post-overtake phase.
- GAE credit window was too short for delayed crash consequences.
- Critic could not easily infer privileged geometry from lidar/history.
- BC anchor acted uniformly, including states where BC was believed to be wrong.
- Collision attribution originally risked mixing opponent-only collisions.
- Reward/logging did not clearly separate pre/alongside/post collision modes.

Fixes/diagnostics added over time:

- Privileged critic.
- `gae_lambda=0.99`, `critic_lr=5e-4`.
- Ego collision attribution.
- Front-risk/risk-field diagnostics.
- Per-phase collision analysis.
- Three-grid OL1 validation.

Earlier notable results:

| experiment | rough OL1 result | takeaway |
|---|---:|---|
| no-gate PPO | ~40% collision | PPO collapses into unsafe close-follow/overtake attempts |
| risk gate | 58.5% collision | reward gating alone not enough |
| credit fix 1b | 25% collision, 7% overtake | credit/critic helped, but not sufficient |
| resume600 | 38.5% collision | 1b apparent recovery was a wave/trough, not stable convergence |
| preBC10 | 22.5% collision, 3% overtake | stronger BC anchor helps but suppresses overtake and still far above BC |
| alpha risk-field | ~24% collision | wider front risk moved spacing a little but did not restore braking response |

Major interpretation from those rounds:

- Credit repair and privileged critic are necessary but not sufficient.
- Global BC anchor is not a reliable recovery mechanism for unsafe close states.
- The policy repeatedly found a "close-follow / alongside contact" attractor.

### 2. D1-a: Frozen BC Speed / PPO Steering

D1-a design:

- Composite policy:
  - frozen BC network outputs speed
  - trainable PPO network outputs steering
  - PPO distribution becomes one-dimensional over steer
- Reason:
  - previous failures were strongly associated with speed drift/closing bias
  - output-level BC anchor cannot strictly preserve speed because speed/steer share representation

Important nuance:

- This is not a final ideal architecture claim.
- It is an experimental decomposition: isolate the safety effect of preserving BC speed from steering exploration.

D1-a results:

| source | collision | overtake | conclusion |
|---|---:|---:|---|
| seed1 canonical | 6/200 = 3.0% | 3/200 = 1.5% | near BC |
| seed1 off21 | 9/200 = 4.5% | 2/200 = 1.0% | better than BC collision count |
| seed1 off42 | 6/200 = 3.0% | 6/200 = 3.0% | better than BC collision count |
| seed1 pooled | 21/600 = 3.50% | 11/600 = 1.83% | practical safety support |

Strict per-grid note:

- canonical seed1 is 6/200 vs BC 5/200, so do not claim strict per-grid pass.
- But pooled result is inside/near BC safety envelope and much better than unsafe PPO variants.

Relevant report:

- `logs/d1a_seed1_confirmation_20260705_0846/report.md`
- `logs/all_results_summary_remote.md`

Conclusion:

- D1-a supports the claim that speed-path drift/shared representation was a major cause of previous degradation.
- D1-a does not solve overtake rate by itself. That was expected.

### 3. D1-b: Lateral Offset Curriculum

D1-b design:

- Keep D1 speed-freeze.
- Add lateral offset spawn curriculum:
  - `--lateral_offset_prob 0.5`
  - offset range `[0.3, 0.8]`
- Motivation:
  - provide positive exposure to lateral/overtake corridor states.

D1-b remote complete results:

| run | canonical collision / overtake | off21 collision / overtake | off42 collision / overtake | pooled collision | pooled overtake |
|---|---:|---:|---:|---:|---:|
| BC reference | 5/200 / 3/200 | 12/200 / 0/200 | 7/200 / 5/200 | 24/600 = 4.00% | 8/600 = 1.33% |
| D1-b seed1 | 5/200 / 0/200 | 8/200 / 1/200 | 4/200 / 4/200 | 17/600 = 2.83% | 5/600 = 0.83% |
| D1-b seed2r | 11/200 / 2/200 | 17/200 / 2/200 | 14/200 / 7/200 | 42/600 = 7.00% | 11/600 = 1.83% |
| D1-b complete seeds combined | 16/400 / 2/400 | 25/400 / 3/400 | 18/400 / 11/400 | 59/1200 = 4.92% | 16/1200 = 1.33% |

Collision buckets:

| run | car/pre | car/alongside | car/post | wall/pre | wall/alongside | total |
|---|---:|---:|---:|---:|---:|---:|
| seed1 | 5 | 12 | 0 | 0 | 0 | 17 |
| seed2r | 9 | 30 | 0 | 3 | 0 | 42 |
| combined | 14 | 42 | 0 | 3 | 0 | 59 |

Conclusion:

- D1-b as currently configured is not successful.
- It does not improve overtake over BC in the combined result.
- It introduces/uncovers a dominant `car/alongside` collision mode.
- The curriculum teaches corridor exposure/maintenance but does not reliably teach safe transition from on-raceline following into the lateral pass corridor.

Relevant report:

- `logs/d1b_summary_20260705_092447.md`

### 4. Anchor Baseline: Can Stronger BC Anchor Replace Speed-Freeze?

This was run to answer the user's concern:

- Is D1 speed-freeze too artificial/complex?
- Can better BC anchor solve speed drift while keeping a single network?

Config:

- single-network PPO
- no `--freeze_speed`
- no lateral offset: `--lateral_offset_prob 0.0`
- `--anchor_speed_scale 7.5`
- `--beta_bc 5.0`
- `--gae_lambda 0.99`
- `--critic_lr 5e-4`
- `--time_gap 0.8`
- `--front_base_margin 0.9`

Results:

| source | collisions | overtake | verdict |
|---|---:|---:|---|
| BC reference | 24/600 = 4.00% | 8/600 = 1.33% | baseline |
| anchor seed0 | 49/600 = 8.17% | 12/600 = 2.00% | fail |
| anchor seed1 | 27/600 = 4.50% | 10/600 = 1.67% | yellow |
| anchor pooled | 76/1200 = 6.33% | 22/1200 = 1.83% | fail vs BC |

Wilson CIs:

| source | collision CI |
|---|---:|
| BC repeated as 48/1200 | [3.03%, 5.26%] |
| anchor pooled 76/1200 | [5.09%, 7.86%] |

Collision structure:

- No car/post collisions.
- Dominant failures remain car/pre and car/alongside.
- Local seed0 especially failed all three grids:
  - canonical 16/200
  - off21 16/200
  - off42 17/200

Conclusion:

- Stronger global BC anchor can partially stabilize speed drift but does not reliably close the safety gap.
- It should not replace D1 speed-freeze as the current main safety baseline.
- Do not spend the next round just tuning `beta_bc` or `anchor_speed_scale`.

Relevant report:

- `logs/anchor_baseline_summary_20260705_113917.md`

## Current Best Interpretation

The current PPO problem is not simply "reward too weak" or "critic too weak" anymore.

Evidence-supported interpretation:

1. Privileged critic and longer credit helped but did not solve safety.
2. Single-network PPO tends to drift speed/closing behavior away from BC; global BC anchor is only a soft pressure and does not fully prevent this.
3. D1 speed-freeze gives the cleanest safety baseline so far.
4. D1-b offset curriculum does not improve overtake because the policy still lacks a safe, learned transition through the alongside phase.
5. The current dominant failure is `car/alongside`, not the originally feared post-overtake rear-risk bucket.

Important: This does not prove speed-freeze is the final architecture. It proves that in this codebase and training setup, hard speed preservation is currently a better experimental control than output-level BC anchor.

## Recommended Next Step

Next round should be a code/design modification, not another same-axis experiment.

Recommended target:

- Keep D1 speed-freeze or another equally strict speed constraint.
- Add explicit side-clearance shaping/gating for the alongside phase.
- Then rerun D1-b-style training.

Concrete proposal for next agent:

1. Add new reward parameters in `RewardWeights` / CLI:
   - `side_gap_thresh` around `0.7` to `0.8`
   - `side_risk_weight`
   - optional `side_progress_gate` boolean/weight
2. Define alongside unsafe condition using existing per-step geometry:
   - `abs(rel_s) < 0.6` or a slightly wider band such as `< 0.8`
   - `lat_gap < side_gap_thresh`
3. Penalize unsafe alongside:
   - dense penalty proportional to `(side_gap_thresh - lat_gap) / side_gap_thresh`
   - optionally gate positive relative progress while unsafe alongside, so PPO cannot earn pass progress by squeezing beside the opponent.
4. Add diagnostics:
   - `mean_side_risk`
   - `unsafe_alongside_frac`
   - `first_cross_lat_gap` or `corridor_entry_lat_gap`
5. Run a small smoke test with `--lateral_offset_prob 0.0` to verify default behavior unchanged if side risk weight is zero.
6. Run D1-b-side-risk:
   - `--freeze_speed`
   - `--lateral_offset_prob 0.5`
   - `--lateral_offset_min 0.3`
   - `--lateral_offset_max 0.8`
   - existing safety settings: `--gae_lambda 0.99 --critic_lr 5e-4 --time_gap 0.8 --front_base_margin 0.9`
   - evaluate canonical/off21/off42.

Expected success criteria:

- Hard constraint: pooled collision should remain inside the BC envelope and not exceed BC point by much.
- Specific target: `car/alongside` bucket should decrease relative to D1-b combined `42/59`.
- Secondary target: overtake rate should exceed BC `1.33%`; if safety improves but overtake does not, the next issue is corridor entry/behavior generation rather than side safety.

Stop/diagnosis rule:

- If `car/alongside` decreases but `car/pre` rises, side-risk may be pushing the policy back into close-follow contacts; revisit front-risk/gap control.
- If both side-risk and collision improve but overtake remains flat, add a transition curriculum that starts near raceline but with a small lateral velocity/heading bias or staged offset schedule. Do not immediately widen the offset curriculum.

## Do Not Do Next

Avoid these unless the user explicitly asks:

- Do not rerun the same anchor baseline as the next main step.
- Do not expand to all racelines before OL1 side-clearance is understood.
- Do not claim D1-b succeeded; combined overtake equals BC and safety is not robust.
- Do not claim D1-a strictly passes every grid; say it practically supports the safety axis, while canonical seed1 was one collision above matched BC.
- Do not treat local D1-b startup guard failures as real display/f110 hangs; they were log buffering false positives and are fixed.

## Useful Paths

Main summary reports:

- `logs/anchor_baseline_summary_20260705_113917.md`
- `logs/d1b_summary_20260705_092447.md`
- `logs/d1a_seed1_confirmation_20260705_0846/report.md`
- `logs/all_results_summary_remote.md`
- `logs/d1b_local_failure_revised_20260705.md`

Experiment directories:

- `logs/anchor_baseline_local_seed0_20260705_113917/`
- `logs/anchor_baseline_remote_seed1_20260705_113917/`
- `logs/d1b_seed1_20260705_092447/`
- `logs/d1b_seed2r_20260705_092447/`
- `logs/d1a_seed1_confirmation_20260705_0846/`

Current scripts:

- `logs/anchor_baseline_unattended/run_anchor_baseline.sh`
- `logs/d1b_unattended/`

## Verification Checklist For Next Agent

Before modifying:

```bash
cd /home/haowei/Documents/End2Race
git status --short
python -m py_compile train_ppo.py ppo_utils.py analyze_collisions.py
```

After modifying side-risk:

```bash
python -m py_compile train_ppo.py ppo_utils.py analyze_collisions.py
```

Run a short smoke:

```bash
python train_ppo.py \
  --total_iterations 2 \
  --gae_lambda 0.99 \
  --critic_lr 5e-4 \
  --time_gap 0.8 \
  --front_base_margin 0.9 \
  --freeze_speed \
  --lateral_offset_prob 0.0
```

Then D1-b-side-risk training, using an unattended script and preserving logs.

Local run:

- Do not set DISPLAY explicitly.
- Use `PYTHONUNBUFFERED=1` if wrapping logs.

Remote run:

- Use `DISPLAY=:1`.

## Final Current Decision

Answer to "next round: modify or experiment first?":

Modify first.

Reason:

- The latest anchor validation answered the main uncertainty.
- Stronger global BC anchor is not reliable enough to replace D1 speed-freeze.
- Current bottleneck is now well-localized to alongside side-clearance and corridor-entry behavior.

