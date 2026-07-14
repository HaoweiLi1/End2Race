# End2Race PPO — Current Entry Point

Updated: 2026-07-14 (Asia/Singapore)

This directory is the concise entry point for the collision-reduction PPO
project. It replaces the need to reconstruct current state from many dated
run directories.

## Product objective

The original objective was lexicographic. For B4 only, the owner prospectively
approved a safety-first 5% corrected-overtake tolerance:

1. corrected overtake must remain at least 95% of BC;
2. subject to that guardrail, reduce any-agent collision rate, with product
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
- B2 ran six direct PPO candidates. Several reduced collision, but every one
  lost too many overtakes under B2's frozen gate; no arm was selected.
- B3 unified stochastic training and deterministic deployment and passed
  implementation review, but remains `PAUSED UNRUN` with no RunPlan.
- B4 ran one seed1 plain-End2Race output-head-only PPO for 30 iterations and
  completed 2,400 product-grid episodes. BC/iter10/iter20/iter30
  collision-overtake counts were `24/342`, `24/332`, `36/294`, `39/296`.
  No snapshot was feasible; B4 is `B4_SUBSTANTIVE_NEGATIVE` and selected none.
- A read-only replay diagnosis found monotonic BC-relative action drift, nearly
  exact 100 Hz iid exploration, and 9.375x collision oversampling. This ranks a
  deployment-compatible BC-relative trust region as the first *unrun*
  one-variable hypothesis; it does not authorize B5 or settle representation
  sufficiency.
- Fresh/final pools remain sealed. No B3/B5 or B4 continuation is authorized.

## Interpretation

The diagnostic work established three useful facts:

- deployable observations contain useful collision-risk information;
- bounded residual actions can recover many observed BC collision cases;
- naive supervised witness imitation is not currently a reliable PPO
  admission gate.

B4 showed that the tested frozen-feature direct head can change which cases
collide at iter10 without net safety gain, then regress both collision and
overtake at later snapshots. This is a configuration-level negative, not a
proof about residual policies, PPO in general, or GRU representation quality.
Any next experiment requires a new prospective owner decision.

## Read order

1. [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md) — result ledger and claim
   boundaries.
   The longer Chinese record is `docs/EXPERIMENT_RECORD.md`.
2. `.agents/B4_DIRECT_HEAD_PPO_RESULT.md` — exact B4 identities, numerical
   result, failures/fixes, evidence paths and stop decision.
3. `.agents/B4_SUBSTANTIVE_NEGATIVE_ANALYSIS.md` — reproducible cause analysis,
   claim boundaries and external-review packet.
4. [evidence/b4_substantive_negative/README.md](evidence/b4_substantive_negative/README.md)
   — compact Git-tracked tables and reproduction command.
5. [NEXT_PPO_DIRECTION.md](NEXT_PPO_DIRECTION.md) — historical proposal that
   preceded the now-closed B4 run; it is not current execution authority.
6. [ARTIFACT_RETENTION.md](ARTIFACT_RETENTION.md) — what local evidence is
   retained and how historical material was archived.
7. `.agents/HANDOFF.md` — full historical authority ledger; newest numbered
   section wins when older sections conflict.
8. `.agents/COMPUTE_CAPACITY_AND_EXECUTION_GUIDE.md` — measured local/remote
   training/evaluation capacity, saturation points, CPU affinity and the
   managed-runner changes required before concurrent jobs are authoritative.

For exhaustive lookup, use `AUTHORITY_INDEX.md`,
`ANALYSIS_AND_REPORT_INDEX.md`, `DOCUMENT_INDEX.tsv`, and
`LOG_FILE_INDEX.tsv` in this directory.

## Execution boundary

- Remote unattended Codex authority is revoked.
- The authorized B4 remote run is complete; do not resume or launch another
  remote job without new authority.
- Do not open the D2 test or a fresh/final pool during development.
- Do not claim generalization from Task 6, Task 9, Task 10, or the 288-scenario
  development population.
- Low GPU utilization from one current PPO learner is expected: collection is
  one environment, batch-1 recurrent inference and synchronous CPU/GPU ledger
  transfer. Do not duplicate a job to fill the device. Use the measured
  profiles in `.agents/COMPUTE_CAPACITY_AND_EXECUTION_GUIDE.md` only after the
  resource topology is frozen in a new RunPlan.
