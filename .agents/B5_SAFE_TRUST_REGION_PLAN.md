# B5-A plain-End2Race safe-reference trust-region plan

Status: **OWNER-APPROVED STRICT SINGLE-VARIABLE IMPLEMENTATION FROZEN; RUNPLAN NOT YET CREATED**

Date: 2026-07-14

This is the prospective authority for B5-A. It does not alter B2/B3/B4
history, reopen B4, authorize B5-B/AR(1), or open any fresh/final pool.

## 1. Scientific question

> With every B4 scientific setting and the exact seed1 curriculum order held
> fixed, can a cumulative canonical-BC mean-KL cap on a fixed set of safe BC
> trajectories prevent nonselective behavior drift, so that collision falls
> below 24, fixed collision exceeds new collision, and corrected overtake stays
> at or above 325 on the opened Austin 600-case panel?

B4 is an integrity-valid substantive negative. Its closest snapshot, iter10,
fixed 11 BC collisions but introduced 11 new collisions. Mean BC-relative
action drift grew through iter10/20/30 and later snapshots jointly worsened
collision and overtake. B5-A tests only the highest-ranked causal hypothesis:
missing preservation relative to canonical BC.

## 2. Strict single-variable boundary

The following B4 values remain literal and are not performance-tuned:

| Contract | Frozen B5-A value |
|---|---:|
| deployment actor | canonical plain `End2Race` |
| actor initialization | strict canonical BC load |
| trainable actor tensors | `output_layer.*` only |
| frozen actor tensors | `k`, speed MLP, dummy embedding, GRU |
| actor LR | `3e-5` |
| PPO clip | `0.10` |
| actor epochs | at most `3` |
| rollout weighted KL | `0.015`, B4 post-accepted-epoch stop semantics |
| critic | independent privileged `12-128-128-1` scalar MLP |
| critic LR / epochs | `3e-4 / 3` |
| action distribution | iid factorized Normal at 100 Hz |
| fixed std | steer `0.03`, speed `0.20` |
| actuator projection | steer `[-0.52,0.52]`, speed `[0,20]` |
| reward | terminal-only `-2*C + O` |
| curriculum | `6 collision / 6 overtake / 4 follow` |
| curriculum order | exact B4 seed1 30-iteration order |
| gamma / lambda | `0.999 / 0.997` |
| mean-bound coefficient | `0.01` |
| seed | `1` only |
| iterations | `30` |
| deployment snapshots | `0 / 10 / 20 / 30` |

The exact B4 seed1 curriculum digest is
`40275f3d928b753fdc683ca20df83ad4097d9e8ac3c92f4a150fba3a50a5afa1`.

The only new scientific mechanism is:

```text
fixed 64-episode canonical-BC safe reference
cumulative D_safe <= 0.01 after every accepted actor epoch
deterministic actor+Adam rollback/retry solver
```

The LR ladder is enforcement machinery for that hard constraint, not a second
experimental arm. B5-A does not implement AR(1), residual, sidecar, gate,
anchor-loss coefficient, dual, GRU unfreeze, sampler change, reward change,
new seed, or longer training.

## 3. Fixed safe-reference population

### 3.1 Eligible data

Only the 1,640 B4 training L2 rows are eligible. The two approved safe BC
outcomes are `follow` and `overtake`. The reference must not contain any row
from the Austin 600-case panel, candidate-induced trajectories, D2 sealed test
data, or fresh/final confirmation pools.

Reference episodes remain eligible for the unchanged B4 curriculum; selecting
them does not remove or reweight any training episode.

### 3.2 Prospective selection rule

For each of `4 maps x 2 outcomes`:

1. group eligible rows by L4;
2. within each L4 retain the L2 with the minimum domain-separated SHA-256 key;
3. rank the representatives by a second domain-separated SHA-256 key;
4. take the first eight distinct L4 representatives;
5. fail preflight if any stratum has fewer than eight unique L4 values.

The selection is exactly 64 unique L2 and 64 unique-within-stratum L4 rows. It
is not manually chosen after inspecting behavior.

### 3.3 Canonical feature replay

Each source NPZ must match its frozen manifest SHA-256. Canonical BC features
and means are generated with:

```text
actor.eval(), mask_prob=0
batch size = 1
one forward_features call per 0.01 s frame
hidden = 0 at episode start, then recurrently propagated
initial previous speed = evaluator initial ego speed * 0.9
subsequent previous speed = previous actual ego speed
all actor frames retained
```

Fused full-sequence GRU execution is forbidden because it is not numerically
identical to the deployed batch-one recurrent loop.

The immutable reference NPZ contains episode metadata, lengths, all 1680D
frozen features, canonical BC means, episode indices and step indices. Its one
file digest is bound into the RunPlan and every full resume checkpoint.

## 4. Safe metric

For frame `t` of episode `e`:

```text
d[e,t] = 0.5 * ((mu_steer - mu_BC_steer) / 0.03)^2
       + 0.5 * ((mu_speed - mu_BC_speed) / 0.20)^2
D_safe = mean_episode(mean_frame(d[e,t]))
```

Actor means are produced in model precision; differences, normalization and
aggregation are evaluated in float64. Because every map/outcome stratum has
eight episodes, episode equality also gives map/outcome equality.

The hard cap is `D_safe <= 0.01`. It is an average empirical latent-mean KL on
fixed BC histories. It is not a per-state bound, candidate-trajectory bound,
or closed-loop safety guarantee. Frame/episode p50, p95, max and each stratum
mean are diagnostics only.

## 5. Required pre-RunPlan audit

Before a B5 RunPlan may exist, the frozen reference is evaluated read-only on
canonical BC and B4 seed1 iter10/20/30.

Blocking requirements:

```text
BC D_safe <= 1e-10
B4 iter10 D_safe > 0.01
```

All later B4 values are reported. The threshold is already frozen and must not
be changed after observing this audit. If iter10 does not cross the cap, this
reference does not target the measured B4 failure and B5-A must not start.

## 6. Actor hard-cap solver

For each otherwise unchanged B4 actor epoch:

1. save `output_layer` parameters and complete actor Adam state;
2. freeze the B4 deterministic minibatch permutation for this epoch;
3. try LR multipliers `[1, 1/2, 1/4, 1/8, 1/16]` in order;
4. before every attempt restore the same pre-epoch actor and Adam state;
5. run the same rollout and minibatch order;
6. evaluate full-reference `D_safe`;
7. accept the first attempt with `D_safe <= 0.01`;
8. if all fail, restore actor and Adam exactly and skip that epoch.

An accepted lower-multiplier attempt retains its resulting Adam moments, but
the optimizer LR is reset to the frozen B4 base `3e-5` before checkpointing or
the next epoch. The multiplier is temporary solver state, not a persistent LR
schedule.

Only `D_safe` causes rollback/retry. After an accepted epoch, B4 rollout KL is
computed exactly as before. If weighted KL exceeds `0.015`, the accepted epoch
is retained and only later actor epochs in that iteration are stopped.

The critic always completes all three unweighted per-state MSE epochs,
regardless of actor acceptance, skip, or rollout-KL stop.

Every iteration ledger reports attempts, multipliers, candidate/accepted
optimizer steps, accepted/skipped epochs, safe metrics, rollout KL, critic
epochs, raw-action projection and terminal outcomes.

## 7. Checkpoint and deployment contract

Deployment snapshots contain only the canonical 12-key
`End2Race.state_dict()` and strict-load through the original BC model. Critic,
optimizer, std/log-std, safe reference, retry state, residual, sidecar, gate,
action core and RNG are forbidden from deployment.

Private full checkpoints bind RunPlan, curriculum and reference digests and
store actor, critic, optimizer states, frozen identity and RNG only for exact
iteration-boundary resume. A temporary retry LR may never be stored.

## 8. Blocking implementation evidence

Before training:

1. deterministic 64-episode selection and unique-L4 balance;
2. source NPZ hash and batch-one canonical replay checks;
3. iteration-0 actor identity and zero safe KL;
4. pre-RunPlan B4 iter10 cap-crossing audit;
5. forced violation proving actor parameters and every Adam field restore exact;
6. lower-multiplier acceptance from the same minibatch order;
7. critic-three-epoch isolation on actor rejection;
8. unchanged B4 terminal/GAE/raw-latent/projection/replay contracts;
9. four-map deterministic identity plus collision/horizon stochastic plumbing;
10. frozen tensors and plain 12-key checkpoint strict load;
11. B2/B3 compatibility programs, Python compilation and CLI dry-load.

These are correctness blockers, not TTC, Brier, warm-start or representation
quality gates.

## 9. Immutable execution plan

After the reference audit and an externally addressable source commit:

- create one unique `plan-b5`;
- stage read-only source/inputs on isolated local/remote roots;
- reproduce the existing 24-collision/138-overtake 288-panel BC preflight;
- run B5 plumbing and publish identical READY markers;
- run exactly one seed1 learner on the remote RTX 4080 SUPER with
  `DISPLAY=:1`;
- do not start seed0 or a second arm merely because GPUs permit it;
- monitor iteration ledger, GPU identity, process and atomic output through 30;
- collect and revalidate the release locally.

Remote is chosen because it is faster. Local GPU remains useful for tests and
later shard0 evaluation, not unauthorized scientific replicas.

## 10. Opened-development evaluation

The Austin panel is an **opened-development regression panel**, not fresh or
final confirmation. It uses the unchanged production evaluator and grid:

```text
3 opponent racelines x 4 opponent speeds x 50 startpoints = 600 cases/model
```

The immutable 600 canonical-BC B4 rows are reused after provenance/model-hash
validation. Only B5 iter10/20/30 are newly evaluated: 1,800 episodes.

- local: shard0 (`1/5`);
- remote: shards1-4 (`4/5`), sequential because evaluation is CPU-bound.

The hosts may run non-overlapping queues concurrently. Every worker uses a
private absolute Numba cache; remote exports `DISPLAY=:1`.

## 11. Selection and verdict

Per snapshot feasibility:

```text
corrected overtake >= 325
any-agent collision < 24
fixed collision > new collision
deterministic speed projection count == 0
```

Collision `<=16` is an opened-development target only. Select the feasible
snapshot with lowest collision, then highest overtake, then earlier iteration.
If none is feasible, record `B5_A_SUBSTANTIVE_NEGATIVE` and stop.

A pass is only an `OPENED_DEVELOPMENT_SURVIVOR`. External result review must
precede any prospectively approved seed0 replication. Fresh/final stays sealed.

## 12. Interpretation discipline

- Many skipped epochs: the cap is highly binding; do not blame exploration.
- Nonzero updates, protected overtake, no collision repair: preservation is
  supported but repair is not; exploration, credit, representation, reward,
  cap strictness and sample size remain possible.
- Fixed approximately equals new: the fixed-history average cap did not give
  enough closed-loop selectivity; off-manifold paths remain possible.
- Survivor: review first; do not automatically run seed0 or sealed data.

No result may be automatically labelled `exploration-limited`. AR(1), GRU
unfreeze, sampler/reward/LR changes, residual, dual, seeds or B5-B require a
separate prospective decision.

## 13. External-review surfaces

- `bplus_v22/b5_safe.py`: reference metric, actor+Adam retry and checkpoint;
- `scripts/build_b5_safe_reference.py`: selection/framewise canonical replay;
- `scripts/audit_b5_safe_reference.py`: BC/B4 cap audit;
- `bplus_v22/b5_runner.py`, `b5_cli.py`, `bplus_v22/cli.py`: plumbing/learner;
- `Experiments/runner.py`: one remote seed1 job and strict collection;
- `scripts/b5_opened_product_eval.py`: unchanged evaluator authorization and
  merge against immutable BC rows;
- `tests/test_b5_safe.py`, `tests/test_b5_control_plane.py`: solver, optimizer,
  reference, checkpoint and single-variable regressions.

The corrected implementation review boundary is exact commit
`ba25e34c0e503638c5540b0f7c98394da2c1b995`; review the actual diff
`072e0df..ba25e34c0e503638c5540b0f7c98394da2c1b995`. Commit `7bb258c` is the
initial implementation, while `ba25e34` adds the narrow CUDA fixture repair
recorded in HANDOFF §28.5. The later provenance-only commit that records this
SHA may be the replacement RunPlan source commit; it does not change the B5
implementation. RunPlan/reference digests are recorded in `.agents/HANDOFF.md`.
