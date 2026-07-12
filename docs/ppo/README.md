# End2Race PPO — Current Entry Point

Updated: 2026-07-12 (Asia/Singapore)

This directory is the concise entry point for the collision-reduction PPO
project. It replaces the need to reconstruct current state from many dated
run directories.

## Product objective

The objective is lexicographic:

1. corrected overtake rate must not be lower than BC;
2. subject to that constraint, reduce any-agent collision rate, with product
   target `RR <= 0.70` versus BC;
3. overtake improvement is optional after the first two requirements hold.

No current experiment has demonstrated this product objective. In
particular, TTC probes and supervised warm-start gates are diagnostics, not
product success.

## Current state

- The deployed historical baseline remains `cand160`; it did not satisfy the
  new B+ objective and is not the endpoint of this research.
- D2 and D2R-G retain their original failed gates. The grouped D2 test was
  never opened and remains retired/sealed.
- D2.5 demonstrated 67 confirmed-safe-pass recoveries among 91 tested
  non-test BC ego-collision cases using a bounded action library. This is an
  existence result for those cases, not a global ceiling or RR estimate.
- The first supervised warm-start passed a step-level gate but failed its
  288-scenario closed-loop mechanism evaluation: every arm created more
  collisions than it fixed and lost more overtakes than it gained.
- A hierarchical action remediation fixed always-on steering and unsafe
  composition structurally, but its replacement Task 6 failed: all arms had
  zero positive calibration recall. No replacement Task 9/10 or PPO run was
  started.
- No v2.2 PPO optimization result exists.

## Interpretation

The diagnostic work established three useful facts:

- deployable observations contain useful collision-risk information;
- bounded residual actions can recover many observed BC collision cases;
- naive supervised witness imitation is not currently a reliable PPO
  admission gate.

The latest discussion therefore proposes retiring warm-start as a mandatory
gate and testing a minimal BC-direct PPO pilot. That redirect is a proposed
next design, not a completed or approved experiment. Fresh identity remains
BC-exact; exploration must be introduced only in the training behavior
distribution and must be included in the recorded PPO log-probability.

## Read order

1. [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md) — result ledger and claim
   boundaries.
2. [NEXT_PPO_DIRECTION.md](NEXT_PPO_DIRECTION.md) — simplified proposed PPO
   experiment.
3. [ARTIFACT_RETENTION.md](ARTIFACT_RETENTION.md) — what local evidence is
   retained and what generated data was removed.
4. `CURRENT_HANDOFF.md` — full historical authority ledger; newest numbered
   section wins when older sections conflict.

## Execution boundary

- Remote unattended Codex authority is revoked.
- This cleanup is local only; no remote file, process, checkpoint, registry,
  or artifact is changed.
- Do not open the D2 test or a fresh/final pool during development.
- Do not claim generalization from Task 6, Task 9, Task 10, or the 288-scenario
  development population.

