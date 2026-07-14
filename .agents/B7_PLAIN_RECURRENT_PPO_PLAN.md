# B7 plain recurrent PPO engineering plan

Status: prospective implementation and execution plan, 2026-07-14

## 1. Decision and bounded claim

B7 stops the Route-B gate/macro/wrapper loop and tests one product-oriented
plain-model configuration.  The deliverable remains a canonical 12-key
`End2Race.state_dict()` that the original evaluator strict-loads.  There is no
residual head, sidecar, intervention gate, wrapper, distillation stage,
auxiliary risk head, AR(1) process, reward sweep, or architecture arm.

The engineering hypothesis is:

> Low-LR adaptation of the original recurrent representation, collision credit
> localized to the final second, more independent complete episodes, one actor
> optimizer step per rollout, previous-policy hard-scenario mining, and a
> current-rollout BC-safe KL rollback can produce selective collision repair
> without losing more than 5% of corrected overtake.

This is deliberately a combined engineering repair, not a single-variable
causal ablation.  A negative result closes the frozen-feature/plain-recurrent
PPO line under this owner-approved stop rule; it does not prove that PPO or the
plain `End2Race` function class is mathematically incapable.

## 2. Frozen actor and frequency contract

Initialization is a strict load of `pretrained/end2race.pth`.

Frozen actor tensors:

```text
k
speed_mlp.*
dummy_embedding
```

Trainable actor tensors and optimizer groups:

```text
gru.*           lr = 1e-6
output_layer.*  lr = 1e-5
```

Simulator, GRU inference, actor mean, iid Gaussian sample and the original
deterministic evaluator all remain at 100 Hz.  The opponent planner cadence,
RK4 0.01-second integration, any-agent collision terminal, and literal 8-second
product horizon remain unchanged.

The fixed raw-latent policy is:

```text
steer ~ Normal(mean_steer, 0.03)
speed ~ Normal(mean_speed, 0.20)
```

The simulator executes the unchanged fixed projection.  PPO probability ratios
remain defined on the stored raw latent.  The existing mean-bound coefficient
`0.01` is retained solely as the prior B4/B5 actuator-bound regularizer; it is
not a safety proxy or new scientific arm.

## 3. Collision reward redistribution

Task outcomes remain:

```text
any-agent collision  -2
safe overtake         +1
safe follow            0
```

For a collision episode of length `T`, let `H=min(100,T)` and

```text
Z_H = sum(gamma**(-k), k=0..H-1)
```

The last `H` transitions receive `-2/Z_H`; all earlier transitions receive
zero.  A collision episode never receives an overtake reward.  For a safe
overtake, only the last transition receives `+1`.

This construction preserves the discounted collision return as viewed from
the beginning of the one-second window and every earlier state.  It
intentionally changes credit inside that window; it is not claimed to preserve
the return at every one of the last 100 states.  No TTC, clearance, progress,
raceline, braking or smoothness reward enters replay.

## 4. Thirty-two complete episodes and hard-scenario selection

Every iteration contains exactly 32 unique training L2 scenarios:

```text
16 representative
 8 archived canonical-BC collision
 8 current hard / preservation fill
```

The prospective selector is versioned and domain-hashed:

- representative: exactly four hash-selected rows per map from the complete
  1,640-row training population; archived outcome is not used for this role;
- archived collision: exactly two hash-selected BC-collision rows per map after
  excluding representative L2s;
- current hard: selected only from the immediately preceding iteration's 32
  current-policy outcomes, after excluding already selected L2s, with priority:
  1. archived BC-safe to current collision;
  2. archived BC-overtake to current follow/collision;
  3. archived BC-collision still collision;
  4. archived BC-collision to safe follow;
- missing current-hard slots, including all eight slots in iteration 1, are
  hash-filled from unused archived BC-overtake preservation scenarios.

Priority overlap is removed by taking the first matching priority.  Ties use a
fixed domain-separated hash.  The queue is one-iteration current-policy state,
not a cumulative cache of stale outcomes.  Only scenario identity is reused;
every selected scenario is reset and rolled out with the current policy, and
no old transition is reused.

No Austin, 288-development, D2 sealed, fresh or final row participates in
training or hard mining.

## 5. Recurrent replay and actor update

The collector stores, for every 100 Hz actor frame:

```text
360D LiDAR
previous actual speed with evaluator-compatible step-0 initialization
13D privileged critic input
old-policy actor mean
canonical-BC actor mean on the same observation history
raw latent and old log-prob
executed projected command and projection delta
old critic value
episode identity, role, boundary and redistributed task reward
```

The thirteenth critic feature is normalized remaining product time:

```text
max(0, (8.0 - 0.01*step_index) / 8.0)
```

Actor replay starts each episode from zero hidden and performs frame-by-frame
GRU calls.  A fused full-sequence cuDNN call is forbidden because prior repo
evidence found recurrent numerical drift from batch-one evaluator execution.

For each episode, PPO surrogate and mean-bound loss are averaged over its
frames and divided by 32.  Backward is executed once per episode to release its
graph while accumulating gradients.  Only after all 32 episodes does the
runner clip the combined actor gradient and execute exactly one Adam step.

Frozen values:

```text
actor epochs      1
actor Adam steps  1 per rollout
clip epsilon      0.10
entropy           0
gamma             0.999
GAE lambda        0.997
actor grad norm   0.5
```

Advantage normalization is episode-equivalent: first average within each
episode, then across 32 episodes.  Pre-update framewise replay must satisfy
`max |ratio-1| <= 1e-4`.

## 6. Actor acceptance and rollback

Before the actor step the runner snapshots the complete actor and actor Adam
state.  After the step it computes two nonnegative fixed-variance Gaussian
mean-KLs, each averaged within episode and then across episodes:

1. old-policy rollout mean-KL on all 32 stored observation sequences;
2. canonical-BC mean-KL on every episode whose archived BC outcome is follow
   or overtake.

The actor step is accepted only when:

```text
old-policy rollout mean-KL       <= 0.015
current-rollout BC-safe mean-KL  <= 0.010
```

The second metric is accurately scoped: the observation sequences were
generated by the pre-update/old policy in the current rollout.  It is not a
guarantee over candidate-induced future states.

On either violation or a nonfinite actor gradient:

- restore actor tensors and the complete pre-step Adam state;
- commit zero actor optimizer steps for that iteration;
- still train the critic;
- halve both actor group LRs for the next iteration.

An accepted step resets the consecutive-rejection count.  Three consecutive
rejected actor updates stop training before iteration 10; no candidate is then
evaluated.  There is no retry ladder or parameter interpolation.

The rollout-KL hard gate uses the exact analytic mean-KL for equal fixed
variances, not the sample-noisy `old_logp-new_logp` estimator.  PPO ratios still
use raw-action log-probability as usual.

## 7. Critic

The training-only scalar critic is:

```text
13 -> 128 -> SiLU -> 128 -> SiLU -> 1
```

It receives the existing 12D privileged feature plus normalized remaining
time.  It uses episode-equivalent per-state weights, LR `3e-4`, three epochs,
4096-transition minibatches including the tail, and grad norm `0.5`.  Actor and
critic optimizers are disjoint; actor rejection never suppresses critic epochs.
The critic is absent from deployment snapshots.

## 8. Run and evaluation rule

Primary run only:

```text
seed              1
iterations        10
candidate         iteration 10 only
snapshot lottery  forbidden
```

Iteration 0 must be tensor/action/trajectory/outcome identical to canonical BC.
The iteration-10 actor must strict-load as the canonical 12-key model.

Seed1 is evaluated once on the existing opened-development 288 panel, reusing
the immutable mixed-topology BC rows with collision `24` and corrected
overtake `138`.  The four physical candidate shards preserve that topology:

```text
local GPU   shard 0
remote GPU  shards 1,2,3 concurrently
```

The 288 panel has no Austin `startpoint_ordinal`; its cluster statistic is
therefore prospectively defined at L4.  Each L4 contributes
`fixed_collision-new_collision`, and an exact conditional one-sided sign-flip
tail is computed by dynamic programming over the nonzero cluster effects.

The minimum seed1 continue gate requires all of:

```text
overtake >= 132
fixed - new >= 6
collision <= 18
L4 cluster one-sided p <= 0.10
deterministic speed projection count == 0
```

Because BC collision is 24, `fixed-new >= 6` and `collision <= 18` are
algebraically redundant.  Both are retained in the result solely to expose the
paired effect and the absolute KPI; they are not counted as independent
evidence.  Collision `<=16` is the opened-development target.

If seed1 fails, B7 is a substantive negative and the plain recurrent PPO line
closes without tuning the window, LRs, batch, std, cap or iteration count.  If
seed1 passes, its checkpoint is frozen, external review precedes a separately
authorized seed0 run with the identical configuration.  Only if seed0 also
passes may the already-opened Austin 600 panel be run.  Sealed/final data remain
closed throughout.

## 9. Compute allocation

The remote RTX 4080 SUPER runs the single authoritative seed1 learner with
`DISPLAY=:1`.  A second copy of the same seed is forbidden; low utilization
during serial environment collection is expected.  Local resources run unit
tests, compilation, deterministic smoke and later physical evaluation shard 0.
Once iteration 10 is immutable, local shard 0 and the three remote shards start
concurrently so both GPUs perform useful, nonduplicated work.

Every process freezes one numerical-library thread and unbuffered output.  The
remote job is launched from a clean source archive at the exact reviewed commit,
not from the remote machine's dirty historical checkout.  Logs, process state,
GPU telemetry, iteration ledger, checkpoint and COMPLETE envelope are monitored
until a terminal result exists.

## 10. Blocking checks before the learner

1. canonical 12-key iteration-0 strict identity;
2. deterministic four-map trajectory/outcome and recurrent replay identity;
3. raw latent, old log-prob, projection and terminal boundary inheritance from
   the B4 production collector;
4. collision reward discount-equivalence regression;
5. 32 unique L2 sampler roles, map quotas, priority and deterministic hash;
6. framewise pre-update ratio-one;
7. exactly one accumulated actor optimizer step and GRU/head gradients only;
8. rejected step restores actor plus complete Adam state and halves both LRs;
9. critic always completes three episode-weighted epochs;
10. actor-only checkpoint strict-loads while critic/std/optimizer remain absent;
11. 288 candidate merge enforces exact row/shard/provenance inventory and the
    L4 cluster statistic;
12. historical B4/B5/B6 compatibility programs and `py_compile` remain green.
