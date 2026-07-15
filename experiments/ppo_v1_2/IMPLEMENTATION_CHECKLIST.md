# PPO V1.2 Implementation Checklist

Authority: `.agents/PPO_V1_2_EXPERIMENT_GUIDE.md` at baseline
`4fac86858802353e5b0892ff9d3c874bc15d781b`.

## Prompt 1: infrastructure only

- [x] Preserve stock `RecurrentPPO` train/rollout/GAE/loss/buffer code.
- [x] Implement exactly C0/C1/C2/C3; only the value path may differ.
- [x] Preserve actor observation/action/hidden/distribution and 12-key export.
- [x] Prevent C1/C2 critic gradients from reaching frozen/actor features.
- [x] Use only pre-action current-state 12D C3 features and Dict isolation.
- [x] Add deterministic 100-startpoint, 10,800-candidate expanded pool builder.
- [x] Record every invalid reset/geometry/finite/planner preflight row.
- [x] Add fixed-seed H1/H2/H3 classifiers and canonical pool hashing.
- [x] Add `with_replacement` and per-env `per_env_balanced_cycle` samplers.
- [x] Add the complete V1.2 trainer CLI and dry-run resolved config.
- [x] Add exact 125-arm registry, stage dependencies, hashes and statuses.
- [x] Add fixed checkpoint/arm selectors and result schemas.
- [x] Add lock, 60-second heartbeat, two-attempt fresh retry and HEAD drift gates.
- [x] Add run-boundary resume; never resume a model mid-run.
- [x] Add run/stage/global aggregation and validation.
- [x] Keep at most one CUDA training arm active.
- [x] Pass the original 27-test regression suite.
- [x] Pass critic identity, leakage, timeout, isolation and strict-load tests.
- [x] Pass hard-pool determinism/set/sampler tests.
- [x] Pass manifest/selector/runner safety tests.
- [x] Generate a dry-run manifest containing exactly 125 training arms.
- [x] Run one-update zero-LR and two-update nonzero smoke for every critic.
- [x] Produce objective-only `IMPLEMENTATION_REPORT.md`.
- [x] Do not run full hard-pool classification or any formal screen arm.

## Prompt 2: hard pools only

- [ ] Require Prompt 1 PASS, committed implementation and clean worktree.
- [ ] Freeze current HEAD, BC hash and Austin asset hashes in runtime manifest.
- [ ] Generate all 10,800 candidates and preflight all candidates.
- [ ] Save expanded candidates, valid rows, invalid rows and validation summary.
- [ ] Classify every valid scenario deterministically into H1.
- [ ] Classify pass 1 with seeds 20260715--20260718.
- [ ] Classify candidate pass 2 with seeds 20260719--20260722.
- [ ] Generate H0, H1, H2 core/boundary/all and H3 core/all.
- [ ] Verify non-empty, unique IDs, canonical sort, hash and distributions.
- [ ] Continue after per-scenario errors; fail the stage if globally incomplete.
- [ ] Mark empty-pool arms `SKIPPED_EMPTY_POOL`; never substitute a pool.
- [ ] Update manifest hard-pool hashes and runnable statuses.
- [ ] Produce objective-only `HARD_POOL_REPORT.md`.
- [ ] Do not start PPO training.

## Prompt 3: formal sweep

- [ ] Require Prompt 1/2 PASS, committed code/results and clean frozen HEAD.
- [ ] Re-evaluate paired BC as exactly 21/233/346/0 or stop with drift status.
- [ ] Run barriers C(4), H(48), B(6), R(4), K(16), E(6), G(4), W(12), X(16), S(9).
- [ ] Start every attempt from canonical BC; maximum two attempts per arm.
- [ ] Use only manifest configs; do not alter, extend or prune the matrix.
- [ ] Save every required per-arm command/config/runtime/log/metric/result artifact.
- [ ] Save every required stage result/rank/selection/failure artifact.
- [ ] Save all required GLOBAL and completion artifacts.
- [ ] Verify all checkpoint evaluations are 600-case, finite, exclusive and error-free.
- [ ] Preserve failed attempts and continue independent arms.
- [ ] Produce table-only objective `REPORT.md`.

## Prompt 4: read-only audit

- [ ] Disable all training and optimizer paths during audit.
- [ ] Verify experiment HEAD, BC hash and asset hashes.
- [ ] Verify exactly 125 arms, legal terminal states and attempt counts.
- [ ] Recompute config hashes, checkpoint schemas and all 600-case validations.
- [ ] Recompute checkpoint selection, stage ranks, top-k and dependency chain.
- [ ] Reconcile target-KL optimizer steps with Adam state.
- [ ] Verify frozen deltas, actor contracts and hard-pool sampling evidence.
- [ ] Regenerate global artifacts in memory/temp and compare bitwise.
- [ ] Trace every report number to a lower-level result.
- [ ] Produce `AUDIT_REPORT.md`, `AUDIT_RESULTS.json`, and `MISMATCHES.tsv` only.
