# D2.5 Counterfactual Recoverability Oracle — Locked Specification

Version: `d2.5-spec-1`  
Date: 2026-07-11 (Asia/Singapore)  
Authority: `CURRENT_HANDOFF.md` §§13–15 and B+ design §6.3.

## 1. Question and boundary

D2.5 performs no learning. It asks whether the bounded B+ macro-residual
action space contains coherent interventions that convert non-test BC
ego-collisions into confirmed safe passes rather than merely safe aborts.

D2 already failed its complete representation gate. Therefore D2.5 is an
action-space diagnostic only: even a positive result cannot open the D2 test,
select a deployable actor, or authorize D3/PPO without a separately redesigned
and locked representation stage.

## 2. Population and registry

Use exactly the 91 ego-collision episodes in the D2 non-test release
`non_test_full_20260711_175713`. Candidate/test episodes and the 1,108 sealed
D2 test episodes are excluded. Append one `D2.5/oracle_search/action_choice`
registry row per case before simulator replay; never rewrite existing rows.
The stage-wide registry opening time is
`2026-07-11T18:45:00+08:00`, and every row points to the canonical evidence
root `logs/d25_counterfactual_20260711`.

Report all 91 cases, including any baseline replay failure. Route counts use
distinct L2 cases and distinct L4 blocks, not branches.

The eight-case baseline smoke is selected without consulting any branch
outcome. For each map in lexical order, choose the lexically smallest L2 ID
among eligible `skill_F` cases and the lexically smallest L2 ID among eligible
`skill_S` cases. This yields exactly two cases on each of the four maps. The
branch smoke executes and reruns branch-library positions 0, 4, and 8 for each
of those cases.

## 3. Baseline determinism gate

Replay every case from its resolved start with the frozen BC checkpoint and
the evaluator's exact 100 Hz loop/opponent planner. Before searching a case,
the no-op replay must reproduce the archived collision:

- identical episode length, final time, collision flags, and corrected
  four-state outcome;
- bit-identical float32 desired actions, actual speeds, poses, progress, and
  terminal fields;
- no action clipping or source/hash mismatch.

Any mismatch blocks that case from search and is surfaced. If more than zero
of the first eight map/skill-stratified smoke cases mismatch, stop the full
oracle and repair replay. Full route feasibility is invalid if any of the 91
baselines remains unreconciled.

## 4. Branch timing and macro semantics

The original impact step is the archived `final_time / 0.01`. Candidate leads
are 3.0, 2.0, and 1.0 seconds, in that order. A branch start is rounded down
to the nearest 10-step macro boundary, so actual lead is never shorter than
requested. Omit a lead when its start would precede episode step zero.

The residual is constant for 0.5, 0.3, or 0.1 seconds (50/30/10 simulator
steps), in that order, then returns to exact BC/no-op for the rest of the
8-second episode. One macro action corresponds to one coherent residual; the
oracle does not resample within its duration.

## 5. Frozen intervention library

At each valid lead/duration, test this ordered ten-action library:

1. brake 1.0 + steer -0.1;
2. brake 1.0 + steer +0.1;
3. brake 0.5 + steer -0.1;
4. brake 0.5 + steer +0.1;
5. steer -0.2;
6. steer +0.2;
7. steer -0.1;
8. steer +0.1;
9. brake 1.0;
10. brake 0.5.

Steering units are radians; brake units are m/s subtracted from the current BC
desired speed. Positive speed residual is forbidden. Composition uses
`steer = BC + dsteer`, `speed = BC - brake`, then the evaluator bounds
`[-0.52, 0.52]` and `[0, +inf)`. A branch that requires any clipping is
reported `action_clipped` and cannot be a recovery witness.

The maximum library is 90 branches per case. Search stops at the first valid
confirmed-safe-pass witness in the fixed order; existence is then proven. A
case without such a witness must exhaust all valid branches. No library value
is changed after results begin.

## 6. Classification

Use the D0.1 corrected whole-series branch alignment and exact classifier:

- `collision_to_confirmed_safe_pass`;
- `collision_to_terminal_overtake_only`;
- `collision_to_safe_abort_follow`;
- `still_collision`;
- `invalid_or_action_clipped`.

Confirmed pass requires no collision and corrected lead `>=2.0 m` over the
entire last 0.7 seconds. Terminal overtake without that hold does not count as
a route witness. Opponent-only collisions are not safe passes.

## 7. Route-R2 feasibility gate

R2 is feasible only if all are true:

- at least 25 distinct L2 cases have a valid confirmed-safe-pass branch;
- witnesses span at least two maps and five L4 blocks;
- at least five `skill_F` and fifteen `skill_S` cases recover;
- at least 30% of tested ego-involved `skill_S` collision cases recover;
- no witness uses positive speed residual or action clipping.

If safe outcomes are dominated by follow/abort or the gate fails, R2 is not a
valid anti-conservative route. Record the frontier and recommend action-space
expansion, hierarchical teacher/options, or a reduced claim; do not start PPO.

## 8. Required evidence

- source/model/scenario/registry hashes;
- baseline replay identity table for all cases;
- branch manifest and one row per executed branch;
- full witness trajectories and actions for every recovered case;
- per-case exhausted/stopped status and best outcome;
- map/L4/skill coverage and R2 gate report;
- deterministic rerun of all smoke branches and every reported witness;
- atomic output, output manifest, independent validator, and `COMPLETE` last.
