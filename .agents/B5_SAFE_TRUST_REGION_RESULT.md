# B5-A safe-reference trust-region result

Status: **INTEGRITY PASS; OPENED-DEVELOPMENT SURVIVOR; EXTERNAL RESULT REVIEW REQUIRED**

Date: 2026-07-14

## 1. Executive decision

B5-A completed the one authorized seed1 learner, all 30 iterations, and the
prospectively frozen Austin `3 racelines x 4 speeds x 50 startpoints` opened
development evaluation. The selected snapshot is `seed1_iter10`:

```text
BC:            24 collision / 342 overtake / 234 follow
B5-A iter10:   22 collision / 347 overtake / 231 follow
paired:         9 fixed collision / 7 new collision
               12 gained overtake / 7 lost overtake
speed projection: 0
```

It satisfies every B5-A feasibility rule, so the formal verdict is
`OPENED_DEVELOPMENT_SURVIVOR`. It did not reach the opened-development target
of at most 16 collisions. This is not fresh/final confirmation, not a product
success, and not authority to run seed0 or open sealed data.

The result supports the narrow hypothesis that constraining cumulative drift
on fixed canonical-BC safe histories can preserve behavior better than B4.
It does not prove that this average fixed-history constraint is sufficient for
closed-loop safety: iter20 and iter30 remained inside the cap but rose to 25
and 27 collisions.

## 2. Immutable execution boundary

| Item | Identity |
|---|---|
| RunPlan ID | `b5_seed1_20260714_021544` |
| embedded RunPlan SHA256 | `20e0af679b13f8ab1e3ee296ffe11189a8c584cc2ef363384fddc8e04d16af63` |
| RunPlan file SHA256 | `736b82576fdcb6682f13ae81bc13127d07df85316308db3eee8d15033d84a78e` |
| staged learner source commit | `482491969b01a632f5726b81316953397c6abd49` |
| corrected B5 implementation boundary | `ba25e34c0e503638c5540b0f7c98394da2c1b995` |
| collection-only repair commit | `e39eb39731ab343b8e25485b29979c8e1d831880` |
| curriculum SHA256 | `40275f3d928b753fdc683ca20df83ad4097d9e8ac3c92f4a150fba3a50a5afa1` |
| safe-reference SHA256 | `6b0e69417ff4e127f3959b5c3506115d5e4420cbb551a1a7bf2abea758b0fe4c` |
| canonical BC SHA256 | `b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4` |

The learner ran remotely on the RTX 4080 SUPER with `DISPLAY=:1` from
10:28:32 to 10:52:03 +08:00 and exited zero. Local had no authorized learner;
its GPU was used only for correctness checks and the non-overlapping product
shard0 queue.

The active collection is:

```text
Experiments/B5_safe_trust_region/runs/b5_seed1_20260714_021544
```

The opened evaluation is:

```text
Experiments/B5_safe_trust_region/opened_evaluations/
  b5_opened_seed1_20260714_021544
```

## 3. Reference and blocking evidence

The reference contains 64 training-only episodes and 51,264 actor frames:

```text
4 maps x 2 BC-safe outcomes x 8 unique-L4 episodes
```

Every source frame was replayed through batch-one, frame-by-frame canonical BC
semantics. Stored BC steer and speed replay errors were both zero. The
pre-RunPlan audit was:

| Model | D_safe |
|---|---:|
| canonical BC | `9.973614132483584e-12` |
| B4 iter10 | `0.02612874743139282` |
| B4 iter20 | `0.12315663286348792` |
| B4 iter30 | `0.15834632780825247` |

Thus BC passed the `<=1e-10` identity check and B4 iter10 crossed the frozen
`0.01` cap before any B5 training began. The threshold was not changed after
these values were observed.

The replacement RunPlan reproduced the exact 288-panel BC preflight topology
`12/32, 2/37, 5/33, 5/36`, merging to 24 collisions and 138 terminal
overtakes. Local and remote preflight, four-map deterministic identity, live
collision/horizon stochastic plumbing, terminal-only reward ledger, ratio-one,
forced actor+Adam rollback, critic isolation and plain actor strict load all
passed before READY.

## 4. Training integrity and cap behavior

The learner completed 480 complete episodes and 345,689 transitions. All
30 iteration records and all full/actor checkpoints passed collection-time
validation.

| Training diagnostic | Result |
|---|---:|
| actor epochs considered | 87 |
| actor epochs accepted | 46 |
| actor epochs skipped after all retries | 41 |
| accepted multiplier `1` | 22 |
| accepted multiplier `1/2` | 3 |
| accepted multiplier `1/4` | 8 |
| accepted multiplier `1/8` | 2 |
| accepted multiplier `1/16` | 11 |
| rollout-KL early-stop iterations | 2 |
| critic epochs | 90 / 90 |
| max accepted `D_safe` over iterations | `0.009995685208732995` |
| final `D_safe` | `0.009962557880718594` |
| max weighted rollout KL | `0.020437857136130333` |
| iterations with weighted KL above `0.015` | 2 |
| max pre-update `abs(ratio-1)` | `7.82012939453125e-05` |
| stochastic steer projections | 327 |
| stochastic speed projections | 0 |

Snapshot safe metrics were:

| Snapshot | D_safe |
|---|---:|
| iter10 | `0.009688012839975628` |
| iter20 | `0.009867385381965327` |
| iter30 | `0.009962557880718594` |

The cap was materially binding. It was not a decorative regularizer: nearly
half of the considered actor epochs were skipped, and 24 accepted epochs used
a reduced multiplier. Every rejection restored the output head and complete
Adam state before the next attempt. The two rollout-KL overshoots retained the
accepted epoch and stopped only later actor epochs, preserving B4 semantics.

`D_safe` is an episode-equal average. Individual frames, episodes and strata
can exceed `0.01`; the final frame p95 was `0.0315530` and the largest stratum
mean was `0.0123323`. These are diagnostics, not contract violations.

The deployment snapshots contain exactly the canonical 12 End2Race keys.
Frozen tensors are exact to BC, `output_layer.*` changed, and no critic, std,
residual, sidecar, gate or optimizer key is present.

## 5. Opened-development evaluation

The evaluator, grid, classifier and canonical BC checkpoint were unchanged.
B4's immutable 600 BC rows were reused; B5 added 1,800 candidate episodes.
Local executed shard0 and remote executed shards1-4 sequentially, with one
CPU-bound evaluator queue per host. All 2,400 merged rows and NPZ hashes passed.

| Variant | Collision | Overtake | Follow | Fixed / new C | Gained / lost O | Speed projection | Feasible |
|---|---:|---:|---:|---:|---:|---:|---|
| BC | 24 | 342 | 234 | — | — | 0 | baseline |
| iter10 | 22 | 347 | 231 | 9 / 7 | 12 / 7 | 0 | **yes** |
| iter20 | 25 | 349 | 226 | 5 / 6 | 12 / 5 | 0 | no: collision |
| iter30 | 27 | 343 | 230 | 5 / 8 | 11 / 10 | 0 | no: collision |

The 95% overtake floor was 325. All snapshots passed it. Iter10 reduced
collision by 2 (`RR=22/24=0.9167`) while increasing overtake by 5. It did not
hit the development collision target `<=16`. The prospective selector therefore
chooses iter10 and no other snapshot.

Selected actor-only checkpoint:

```text
Experiments/B5_safe_trust_region/opened_evaluations/
  b5_opened_seed1_20260714_021544/models/seed1_iter10.pth
SHA256 4dde873cb0c81ac9836eeeb3b0b740d82ff1887a23b8e568008abffd27476312
```

It is a review candidate, not a promoted final checkpoint.

## 6. Comparison with B4

Because B5-A retained B4's scientific configuration and exact seed1
curriculum order, the opened panel provides the intended one-variable
development comparison:

| Snapshot | B4 collision / overtake | B5 collision / overtake |
|---|---:|---:|
| iter10 | 24 / 332 | 22 / 347 |
| iter20 | 36 / 294 | 25 / 349 |
| iter30 | 39 / 296 | 27 / 343 |

B4 iter10 had `11 fixed / 11 new` collisions and `8 gained / 18 lost`
overtakes. B5 iter10 had `9/7` and `12/7`. Later B5 snapshots also avoided
B4's large overtake collapse. This is strong opened-development evidence that
the safe-reference cap improved BC behavior preservation for this seed.

It is not proof of generalization or a complete safety solution. Collision
rose from 22 to 25 to 27 while the fixed-history average stayed under the cap.
That result is compatible with candidate-induced off-manifold trajectories not
being fully controlled by an average on canonical BC histories.

No automatic label such as `exploration-limited`, `representation-limited`,
or `cap solved safety` is justified. IID exploration, terminal credit, frozen
representation, cap strictness and sampling remain unisolated hypotheses.

## 7. Operational issues and remediation

The earlier RunPlan `b5_seed1_20260714_020022` stopped before READY because a
CUDA plumbing fixture mixed CPU features with a CUDA policy. That failure is
already preserved in HANDOFF §28.5 and produced no learner iteration.

The valid learner and product evaluation had no execution failure. After the
remote learner had atomically published COMPLETE, three collection attempts
failed only in evidence validation:

1. B5 plumbing revalidation required the 265 MiB safe reference, but the
   collection payload did not include it.
2. Copying it into `control/input_contract` fixed direct plumbing validation,
   but the nested READY validator still resolved the old staged path.
3. A second diagnostic retry reproduced the nested READY failure.

Commit `e39eb39731ab343b8e25485b29979c8e1d831880` includes the reference in the
collection input contract and threads that exact path through both plumbing
and READY validation, with regression coverage. The fourth collection attempt
completed atomically. The three failed partials remain quarantined and never
altered staged source, checkpoints, ledgers, scenarios or product rows.

## 8. Evidence packet

Git-reviewable compact evidence is under:

```text
docs/ppo/evidence/b5_safe_trust_region/
```

The exact result/evidence content boundary is commit
`d57d6e9bc4c49fdd9e522f4b4e825277239b405d`. Reviewers can inspect the full
B5 implementation-through-result diff
`072e0df..d57d6e9bc4c49fdd9e522f4b4e825277239b405d`.

It contains the full 64-episode selection record, reference audit, complete
30-iteration learner ledger, training summary, 2,400 paired product rows and
merged result. Large NPZ/replay/checkpoint artifacts remain outside Git.

Key immutable file hashes:

| File | SHA256 |
|---|---|
| reference audit | `7a1b9c3982c2238f93709284fbc92c67fd129cb76ce8b44e2de92e4fbc7233e1` |
| learner summary | `5d98008a1bbdc8c83bb8c7e0aad7912686073861714a2dd823c512719f4ad2e9` |
| learner iteration ledger | `fb4e32669470bd6367793b30474be1d414112d6363c18ea7c2cb3e5915317837` |
| product summary | `8d23bc4c40c62bb397006453d94d7093e36f1d7edd5d8ea9b51b1fe188a97b62` |
| product paired rows | `4c48fbbfce9f495aaf7e21f5a3f7afe52674d2e5f8973e8159645d467f6830a3` |

## 9. Next legal step

Stop after publishing this result packet. External result review must inspect
the exact code and evidence diff. If it returns GO, seed0 replication requires
a separate prospective owner decision. No seed0 run, B5-B/AR(1), threshold or
optimizer change, GRU unfreeze, residual, sampler/reward change, extra
iterations, candidate promotion or fresh/final opening is automatic.
