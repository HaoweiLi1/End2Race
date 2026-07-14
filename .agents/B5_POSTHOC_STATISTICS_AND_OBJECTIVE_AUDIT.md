# B5-A post-hoc statistics and objective-alignment audit

Status: **COMPLETE; B5-B OBJECTIVE-WEIGHTING LEARNER NO-GO**

Date: 2026-07-14

## 1. Executive decision

B5-A's historical preregistered verdict remains
`OPENED_DEVELOPMENT_SURVIVOR`, but the external-review qualification is now:

> Feasibility gate passed; paired safety effect statistically inconclusive;
> checkpoint not promoted.

The strongest supported B5-A result is behavior preservation relative to B4,
not a demonstrated repeatable collision improvement. The proposed next learner
that reweights the actor objective by the already-open Austin outcome
frequencies is **not authorized** by the phase-0 mechanism audit. It did not
reliably rotate either the full-rollout gradient or a restored actor+Adam
candidate epoch away from the safe-cap/B4-drift directions, and it reduced the
collision-gradient function norm to about one fifth of the original.

No B5-B learner, product evaluation, AR(1) arm, seed0 replication, or sealed
confirmation run was started.

## 2. Corrections to the proposed analysis

The following parts of the external proposal are correct:

- occurrence-level McNemar values for B5 iter10 are approximately `0.804`
  for collision and `0.359` for overtake;
- `collision < 24` and `fixed > new` are algebraically redundant against a
  24-collision baseline;
- 16 discordant collision cases provide a high noise floor;
- 50 startpoints, rather than 600 independent rows, are the natural first
  clustering unit;
- changing only SGD minibatch size cannot fix objective or exploration
  mismatch;
- actor/simulator/evaluator should remain at 100 Hz, while temporally
  correlated exploration remains a separate future hypothesis.

Five qualifications were required:

1. The `0.1067/1.52/1.56` weights are **opened-Austin development weights**,
   not a universal or fresh product-prevalence estimate. The same panel has
   already affected B4/B5 selection and diagnosis.
2. Startpoint sign-flip enumeration is exact conditional on the observed block
   vectors, but its inferential validity still assumes startpoint-level sign
   symmetry/exchangeability. It is not an assumption-free causal randomization
   test.
3. Snapshot selection must use the same sign flip jointly across iter10/20/30.
   Treating three snapshots as independent is only an intuition, not a valid
   adjustment.
4. Prevalence weights belong only in weighted advantage normalization and PPO
   surrogate. Rollout KL, clip fraction, `D_safe`, critic MSE, and mean-bound
   regularization keep the historical episode-equal trust weight. Otherwise
   the experiment changes both task pricing and trust/regularizer geometry.
5. A full-rollout first-order gradient is not the executed Adam epoch. The
   audit therefore also restores historical actor parameters, Adam state, and
   minibatch order and executes one counterfactual base-LR actor epoch.

The scalar reward remains `-2*C+O`; no outcome reweighting turns it into a
lexicographic safety guarantee.

## 3. Statistical qualification of B5-A

### 3.1 Occurrence and clustered inference

| Metric / snapshot | Positive / negative | Net | occurrence McNemar two-sided | startpoint sign-flip one-sided | startpoint bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| collision iter10 | `9 / 7` | `+2` | `0.803619` | `0.408356` | `[-6, 10]` |
| collision iter20 | `5 / 6` | `-1` | `1.000000` | `0.725586` | `[-7, 5]` |
| collision iter30 | `5 / 8` | `-3` | `0.581055` | `0.815430` | `[-11, 6]` |
| overtake iter10 | `12 / 7` | `+5` | `0.359283` | `0.193130` | `[-4, 14]` |
| overtake iter20 | `12 / 5` | `+7` | `0.143463` | `0.059235` | `[0, 14]` |
| overtake iter30 | `11 / 10` | `+1` | `1.000000` | `0.500000` | `[-8, 10]` |

The exact joint startpoint sign-flip distribution has 1,120 collision states.
For the maximum net collision effect across the three correlated snapshots,
the selection-aware one-sided probability is `0.578968`. This max statistic is
conservative with respect to the overtake gate because it does not condition
on passing that gate.

The B5 iter10 `24 -> 22` collision change therefore cannot be described as a
confirmed safety improvement. It is a churn-level opened-development result.
The already-recorded historical selector is not rewritten, but the checkpoint
is not promoted.

### 3.2 Future development labels

For a future Austin opened-panel experiment against the same BC count, the
recommended reporting labels are prospective and conjunctive:

- net collision effect `1-2`: outcome churn;
- net effect at least `6` **and** one-sided startpoint-cluster evidence at most
  `0.10`: directional opened-development evidence;
- net effect at least `8`, overtake at least `325`, zero deterministic speed
  projection, and cluster-directional evidence: opened-development target hit;
- only a separately authorized sealed-set pass can become confirmation.

These labels are not retroactively used to change B5-A's preregistered verdict.

## 4. Function-space mechanism audit

### 4.1 Fixed probes

The remote RTX 4080 SUPER audit used source commit
`05701c1a804b0bb79b47ddcc1386898c7ad8e547` and `DISPLAY=:1`. It constructed
four fixed probes:

| Probe | Frames | Weighting |
|---|---:|---|
| BC collision histories | 768 | 24 episodes x 32 deterministic frames, episode equal |
| BC overtake histories | 10,944 | 342 x 32, episode equal |
| BC follow histories | 7,488 | 234 x 32, episode equal |
| 64-episode safe reference | 51,264 | every frame, episode equal |

Canonical BC histories were replayed batch-one, one recurrent step at a time.
Maximum stored-action errors were `2.68e-7 rad` and `1.91e-6 m/s`; the safe
reference BC-mean error was `2.86e-6`.

The audit reports standardized action-function directions rather than only
Euclidean parameter cosines. It compares:

- actual B5 checkpoint increments;
- the full-rollout negative loss gradient, decomposed into archived BC
  collision/overtake/follow contributions using one shared normalized
  advantage;
- one restored base-LR actor+Adam candidate epoch with the exact historical
  minibatch order;
- B4 iter30 minus canonical BC as the measured global-drift direction;
- the current safe-reference displacement as the first-order cap-increase
  direction.

### 4.2 Full-rollout first-order result

Across B5 iterations 1-10, opened-Austin weighting:

- reduced mean cosine with the B4 global direction in only `6/10` iterations;
- reduced safe-cap alignment in only `1/9` iterations with a defined cap
  direction;
- changed the median B4 cosine by only `-0.04324`;
- increased the median safe-cap cosine by `+0.01115`;
- reduced the median collision-component function norm to `0.19518` of the
  original.

This is already weak evidence for the proposed mechanism, but it is not the
final decision because the executed learner uses Adam and sequential
minibatches.

### 4.3 Restored actor+Adam candidate epoch

For each of iterations 1-10, the audit restored the pre-update output head and
complete actor Adam state, reused the exact B5 minibatch permutation, and
executed one base-LR candidate epoch. The original-objective candidate
reproduced the historical first-attempt `D_safe` with maximum absolute error
`4.9964e-7`.

| Diagnostic | Original objective | Opened-Austin weighting |
|---|---:|---:|
| base candidate epochs satisfying `D_safe <= 0.01` | `7 / 10` | `5 / 10` |
| iterations where weighting lowered B4-direction cosine | — | `4 / 10` |
| iterations where weighting lowered safe-cap cosine | — | `3 / 9` |
| median change in B4-direction cosine | — | `+0.06150` |
| median change in safe-cap cosine | — | `+0.01081` |

Thus the more pipeline-faithful counterfactual is adverse to the proposed
phase-0 prediction. The correction can still rotate individual iterations,
but it does not systematically reduce cap pressure or global-drift alignment.

## 5. Causal interpretation

The audit does **not** show that outcome/distribution mismatch is irrelevant.
It shows that this exact post-hoc weighting is not an evidence-backed repair:

- it is estimated from an already-open panel;
- weighted advantage normalization changes the relative centering and scale,
  so its effect is not a simple multiplication of three fixed gradients;
- Adam moments and minibatch order materially change the executed direction;
- collision-component function norm falls by roughly 80%, risking a policy
  that merely stays close to BC;
- the corrected base epoch violates the safe cap more often in this replay
  counterfactual.

The current evidence continues to support these narrower claims:

1. the B5 cap prevented the large B4 overtake collapse and global drift;
2. B5-A did not establish a repeatable collision improvement;
3. fixed-history mean KL does not constrain every candidate-induced state;
4. 16 terminal outcomes per update and iid 100 Hz exploration remain plausible
   limitations, not isolated causes;
5. frozen-GRU representation insufficiency remains unproved.

## 6. Execution decision and next legal step

The proposed B5-B objective-weighting learner required phase-0 evidence that
the correction reduced cap/global-slowing components without erasing the
collision signal. That requirement failed. Therefore:

```text
B5-B objective-weighting learner: NO-GO, UNRUN
AR(1) learner:                    NOT AUTHORIZED, UNRUN
seed0 / sealed confirmation:      NOT AUTHORIZED, UNRUN
```

No GPU learner should be launched merely to see whether this post-hoc weight
happens to work. A subsequent experiment needs a new prospective owner
decision after external review of this packet. Candidate-generated
preservation, a constraint-tangent update, temporally correlated exploration,
or a larger episode rollout remain distinct hypotheses; none is automatically
selected here.

## 7. Evidence

Compact evidence is under:

```text
docs/ppo/evidence/b5_posthoc_statistics/
docs/ppo/evidence/b5_objective_alignment/
```

The source scripts are:

```text
scripts/analyze_b5_statistical_noise.py
scripts/analyze_b5_objective_alignment.py
```

Important evidence SHA256 values:

| File | SHA256 |
|---|---|
| statistical summary | `aa8ac14b88f4cdb2c21371f53f1efe05db5d2ef4c4fcf7b9d8163918ecaed2e5` |
| paired inference | `b462b77608ae082f2ca4181b8d6fe6c6a745c0028380ba12d4ca7c1c285728cb` |
| objective summary | `e66a3ac182390aaf576cbf0802c6b8e5a40409cd4084debae2ff2a03a8527ed7` |
| candidate actor+Adam epochs | `a7349abc0cc2f492b400b124a20d7d70c44ac5a897e40de9fd6e8758e8ffbe9c` |
| function updates | `86fdcbda027cb9ce8f01a83f42d1d8178105148ae0108ca60e41d891a8d359ec` |
| gradient alignment | `e8a33d45371cbc615c2842dd470a1fe61a568c905304055c89b766ebcc997870` |
