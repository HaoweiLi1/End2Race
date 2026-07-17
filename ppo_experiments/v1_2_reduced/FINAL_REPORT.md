# End2Race PPO V1.2 Reduced — Diagnostic Closure Report

## Status

`CLOSED_BY_HOLDOUT_VERDICT` with `long_repeat = COMPLETED_DIAGNOSTIC_ONLY`.

The reduced 22-run training program and the three 409,600-transition long-repeat runs are complete. All 15 common-budget long checkpoints now have valid 600-case evaluations. These development-panel results are retained only as repeatability and paired-churn diagnostics.

No checkpoint selection, arm ranking, best-model claim, deployment promotion, or reinterpretation of the holdout verdict was performed. The deployment recommendation remains canonical BC:

```text
pretrained/end2race.pth
sha256 b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4
```

## Frozen contract

```text
map                 Austin
panel               canonical development 600-case panel
EGO_IDX_OFFSET       0
collision scope      ego
render               false
trace                false
noise                0.0
default workers      8
authorized U4 retry  4 workers
evaluate.sh          be774b398d63725a4ac9329816c79c68f673e3d60f11793b53cfff5fcfceee4a
eval_multiagent.py   56cb38aaa6f6b1362ef27bdf5a162410b0d27f9e310c8f9f5c12e13e1e73550e
```

Each seed was trained fresh from canonical BC for exactly 16 updates and 409,600 transitions. Total long-repeat training was 1,228,800 transitions. Training finished before evaluation began.

## Per-checkpoint diagnostics

`fixed/new` are BC collision → PPO non-collision and BC non-collision → PPO collision. `gained/lost OT` are the corresponding overtake transitions.

| Seed | Transitions | Update | Collision | Follow | Overtake | Fixed | New | Gained OT | Lost OT | Attempt | Workers |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 20260715 | 51,200 | 2 | 15 | 232 | 353 | 9 | 3 | 10 | 3 | original valid | 8 |
| 20260715 | 102,400 | 4 | 25 | 232 | 343 | 6 | 10 | 7 | 10 | original valid | 8 |
| 20260715 | 204,800 | 8 | 20 | 232 | 348 | 9 | 8 | 11 | 9 | original valid | 8 |
| 20260715 | 307,200 | 12 | 26 | 228 | 346 | 8 | 13 | 10 | 10 | original valid | 8 |
| 20260715 | 409,600 | 16 | 26 | 228 | 346 | 5 | 10 | 10 | 10 | original valid | 8 |
| 20260716 | 51,200 | 2 | 22 | 234 | 344 | 5 | 6 | 3 | 5 | original valid | 8 |
| 20260716 | 102,400 | 4 | 20 | 235 | 345 | 8 | 7 | 5 | 6 | original valid | 8 |
| 20260716 | 204,800 | 8 | 25 | 233 | 342 | 6 | 10 | 4 | 8 | original valid | 8 |
| 20260716 | 307,200 | 12 | 22 | 232 | 346 | 9 | 10 | 10 | 10 | original valid | 8 |
| 20260716 | 409,600 | 16 | 21 | 238 | 341 | 9 | 9 | 7 | 12 | original valid | 8 |
| 20260717 | 51,200 | 2 | 22 | 232 | 346 | 4 | 5 | 5 | 5 | original valid | 8 |
| 20260717 | 102,400 | 4 | 22 | 234 | 344 | 4 | 5 | 5 | 7 | authorized recovery | 4 |
| 20260717 | 204,800 | 8 | 21 | 234 | 345 | 7 | 7 | 8 | 9 | original valid | 8 |
| 20260717 | 307,200 | 12 | 24 | 231 | 345 | 7 | 10 | 9 | 10 | original valid | 8 |
| 20260717 | 409,600 | 16 | 24 | 234 | 342 | 7 | 10 | 5 | 9 | original valid | 8 |

Every row has 600 unique canonical scenarios, zero evaluation errors, ego collision scope, no render, and no trace. Exact checkpoint, result, and log hashes are recorded in `FINAL_REPEATABILITY.json`.

## Cross-seed repeatability by budget

Standard deviations below are population standard deviations across the three seeds.

| Transitions | Collision mean ± std (median) | Overtake mean ± std (median) | Fixed mean | New mean | Gained OT mean | Lost OT mean |
|---:|---:|---:|---:|---:|---:|---:|
| 51,200 | 19.67 ± 3.30 (22) | 347.67 ± 3.86 (346) | 6.00 | 4.67 | 6.00 | 4.33 |
| 102,400 | 22.33 ± 2.05 (22) | 344.00 ± 0.82 (344) | 6.00 | 7.33 | 5.67 | 7.67 |
| 204,800 | 22.00 ± 2.16 (21) | 345.00 ± 2.45 (345) | 7.33 | 8.33 | 7.67 | 8.67 |
| 307,200 | 24.00 ± 1.63 (24) | 345.67 ± 0.47 (346) | 8.00 | 11.00 | 9.67 | 10.00 |
| 409,600 | 23.67 ± 2.05 (24) | 343.00 ± 2.16 (342) | 7.00 | 9.67 | 7.33 | 10.33 |

Canonical BC on this panel is 21 collision / 233 follow / 346 overtake. The long-repeat distributions remain within the previously documented BC-adjacent churn band. Later budgets do not establish retention or transferable improvement; this statement is diagnostic and is not a ranking or selection result.

## Recovery evidence

The first seed-20260717 U4 attempt reported 600 aggregate episodes and aggregate `error_count = 0`, but contained only 599 unique episode rows. The missing scenario was:

```text
evaluation-sp17-ego727-raceline2-v0.5
episode key ol2_e727_o745_s0.5
```

The original output was never patched or merged. It remains preserved with:

```text
result sha256  d8ca5bc9f9ce263dfacc68e9c72f51ddbae6549bdc1fd954e1dcf5f3acecf304
tree sha256    6605bcdf98b720d837ea13a2f37c608ca5d7888417dba10b52ab128e60def53b
log sha256     7f7cc29a197d2d7e951377e1f51d47c3af9716e15ad20c26cc3aa0e20c7415d1
```

Under explicit owner authorization, the missing scenario smoke passed and one full U4 evaluation-only recovery was run with four workers. It passed with 600 unique rows and 0 errors; result SHA-256 is `6462535b8f017f9f886d0b5a7cc632a5a0526e178d47cb93bb4fd367562a3af6`. No retraining, checkpoint regeneration, config/core/dependency change, partial-row merge, automatic retry, or second retry occurred. U8, U12, and U16 then ran once each with eight workers and passed.

Complete recovery metadata is retained under:

```text
eval_results/_failure_evidence/v1_2_reduced_long_repeat_20260717/
```

## Final audit

The closing read-only audit passed every registered condition:

| Check | Result |
|---|---:|
| Registered training runs | 22/22 PASS |
| Terminal training runs | 22/22 PASS |
| Total training transitions | 3,174,400 PASS |
| Actor checkpoints | 61/61 exact 12-key PASS |
| Checkpoint hashes recorded | 61/61 PASS |
| Valid checkpoint evaluations | 61/61 PASS |
| Long-repeat evaluations | 15/15 PASS |
| Cases/errors per evaluation | 600 / 0 PASS |
| Training completed before evaluation | PASS |
| Required formal files and JSON parsing | PASS |
| Frozen code/model hashes | PASS |
| Original failure evidence retained | PASS |
| No selection/ranking; holdout unchanged | PASS |

## Final verdict

- Long-repeat training: complete for all three seeds.
- Long checkpoint evaluation: 15/15 valid final results.
- Diagnostic repeatability and paired churn: complete.
- Checkpoint selection and ranking: intentionally not performed.
- Holdout verdict: unchanged, `CONSISTENT_WITH_CHURN`.
- Deployment recommendation: unchanged, canonical BC.

This closes the reduced V1.2 line as a validated negative result, not as a PPO actor promotion.
