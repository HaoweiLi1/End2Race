# B4 Plain-End2Race Direct-Head PPO — Substantive-Negative Diagnosis

Date: 2026-07-14 (Asia/Singapore)

Status: **POST-HOC READ-ONLY ANALYSIS OF A CLOSED EXPERIMENT**

Numerical verdict: **B4_SUBSTANTIVE_NEGATIVE (unchanged)**

Next experiment authority: **none**

Exact analysis/code/table review boundary:
`dd49ce00bc82095a1cdd832caa485bce01c1991f`

## 1. Executive judgment

B4's numerical failure is real. The collector/update/checkpoint/evaluator
contracts passed, but no snapshot improved collision while satisfying the
product rules. The evidence does not support the claim that the 706,862-
parameter output head was too small: iter10 changed at least 22 collision
identities, fixed 11 BC collisions, created 11 new collisions, gained 8
overtakes and lost 18. The actor had enough capacity to move behavior; it did
not change the right states selectively.

The strongest evidence-backed diagnosis is:

> The unconstrained full-action head accumulated a broad BC-relative policy
> shift—most visibly a global speed reduction—while PPO's rollout-relative
> clip/KL supplied no cumulative trust region to BC. Sparse terminal labels,
> 100 Hz iid action noise and a collision-heavy training distribution made
> that shared-head update difficult to localize. Iter10 therefore exchanged
> fixed and new collisions; later updates increased both BC-relative drift and
> product regression.

This is not proof that a residual deployment head is required. It is evidence
that this task likely needs some BC-preserving training structure. A
training-only BC-relative action/KL constraint would preserve the required
plain `End2Race.state_dict()` just as a residual parameterization would supply
structural protection in a different way.

Frozen-GRU representation sufficiency remains unresolved. The descriptive
feature-neighborhood result shows substantial fixed/new mixing, but it is too
small and trajectory-dependent to establish that risk information is absent.

## 2. Evidence and method boundary

Inputs are the immutable seed1 B4 training release and the final Austin product
evaluation recorded in `.agents/B4_DIRECT_HEAD_PPO_RESULT.md`:

- run `b4_seed1_20260714_003027`;
- training source `9e5afdc9584343a163c4704597dad87487bd750a`;
- 30 iterations, 16 complete episodes per iteration, 480 terminal labels;
- BC plus iter10/20/30 on `3 x 4 x 50 = 600` cases each;
- 2,400 validated metric/NPZ pairs in total.

The compact outputs are in
`docs/ppo/evidence/b4_substantive_negative/`. They are produced by
`scripts/analyze_b4_substantive_negative.py`; `summary.json` records the exact
input-ledger hashes.

For action drift, all four heads are evaluated on the same 471,786 BC
observation-history transitions. The frozen BC GRU is replayed exactly as the
product evaluator does: batch one, one recurrent step per call, previous actual
speed shifted by one step. Replayed BC outputs match stored evaluator actions
within `2.98e-7 rad` steering and `2.86e-6 m/s` speed. This avoids confounding
policy drift with different candidate-induced trajectories.

The BC-relative Gaussian KL uses the fixed B4 exploration standard deviations:

```text
KL(candidate || BC on the same state)
  = 0.5 * ((delta_steer / 0.03)^2 + (delta_speed / 0.20)^2)
```

Collision object/phase labels in `changed_cases.tsv` are explicitly marked
`*_inferred`: opponent proximity means final center distance at most 1.0 m;
phase uses final relative centerline progress. They are useful grouping
diagnostics, not simulator-native contact labels.

## 3. Outcome anatomy: capacity existed, selectivity did not

| Variant | Collision | Overtake | Fixed/new collision | Gained/lost overtake |
|---|---:|---:|---:|---:|
| BC | 24 | 342 | — | — |
| iter10 | 24 | 332 | 11 / 11 | 8 / 18 |
| iter20 | 36 | 294 | 14 / 26 | 10 / 58 |
| iter30 | 39 | 296 | 14 / 29 | 12 / 58 |

Iter10 is not a no-learning result. It changed 11 of 24 baseline collision
cases and introduced the same number elsewhere. At least two independent flags
can apply to one case—for example, collision-to-overtake is both a fixed
collision and a gained overtake—so the analysis counts transition flags
independently rather than with an `elif` chain.

The 11 iter10 fixed and 11 new collisions are also geometrically similar under
the available terminal diagnostic:

| Group | Opponent-proximity inferred | Alongside phase | Median terminal time |
|---|---:|---:|---:|
| fixed (BC collision trajectory) | 11/11 | 10/11 | 5.87 s |
| new (iter10 collision trajectory) | 11/11 | 9/11 | 5.18 s |

Thus iter10 did not merely exchange opponent collisions for a new wall-failure
mode. It moved failures within the same broad interaction family.

## 4. Direct evidence of cumulative BC-relative drift

The shared-state action replay gives the cleanest diagnosis:

| Snapshot | Mean `abs(delta steer)` | Mean `delta speed` | Mean `abs(delta speed)` | Mean BC-relative KL | P95 KL |
|---|---:|---:|---:|---:|---:|
| iter10 | 0.00292 rad | -0.03134 m/s | 0.03359 m/s | 0.02674 | 0.07202 |
| iter20 | 0.00351 rad | -0.09548 m/s | 0.09570 m/s | 0.13772 | 0.25488 |
| iter30 | 0.00624 rad | -0.10203 m/s | 0.10423 m/s | 0.18824 | 0.33718 |

The dominant broad change is progressive slowing. It applies not just to BC
collision histories but also to BC-follow and BC-overtake histories. Iter10's
mean signed speed changes are approximately `-0.0355`, `-0.0324` and `-0.0305
m/s` in those three groups respectively. This is a global style shift, not an
intervention restricted to the 4% product collision states.

Output-head parameter distance from BC also rises monotonically:

| Snapshot | Head delta L2 from BC | Relative L2 from BC |
|---|---:|---:|
| iter10 | 0.14026 | 0.47097% |
| iter20 | 0.19938 | 0.66949% |
| iter30 | 0.24304 | 0.81608% |

All eight non-output state-dict tensors remain bit-exact to BC. A sub-1%
parameter displacement can still produce the measured broad action change
because the same nonlinear head is shared by every state.

PPO's observed rollout KL does not contradict this result. It compares an
iteration to its rollout policy, not a snapshot to canonical BC. Five of 30
iterations ended above the `0.015` weighted-KL target; the maximum was
`0.082855` (5.52 times target). The stop can only react after an optimizer epoch
has already occurred. This is valid PPO bookkeeping, but it is not a cumulative
BC trust region.

## 5. The iter10 precursor is a nonselective response

On the actual collision-generating history for each changed case, the final
0.5 seconds show:

| Group | Mean signed steer delta | Mean absolute steer delta | Mean signed speed delta | Cases with negative mean speed delta |
|---|---:|---:|---:|---:|
| 11 fixed | +0.00459 rad | 0.00665 rad | -0.04060 m/s | 11/11 |
| 11 new | +0.00185 rad | 0.00511 rad | -0.03734 m/s | 11/11 |

Both successful repairs and new failures received broadly similar small
slowing, while steering direction varied by case. Therefore "PPO learned to
brake" is not a sufficient explanation: the same broad action tendency can
improve one interaction and damage another. The evidence points to state
selectivity/behavior preservation rather than inability to change the action.

This table is teacher-forced on each group's collision trajectory. It describes
the actions the two heads assign to the same observed history; it does not by
itself identify the counterfactual causal action at a collision.

## 6. Exploration was exactly high-frequency iid

Across 351,946 training transitions, reconstructing the old policy from the
stored frozen feature and pre-update checkpoint gives:

| Quantity | Steering | Speed |
|---|---:|---:|
| empirical noise std | 0.029991 | 0.199711 |
| lag-1 autocorrelation | 0.000677 | 0.001183 |
| std of 50-step mean / raw std | 0.14113 | 0.14310 |
| iid theoretical 50-step ratio | 0.14142 | 0.14142 |

The maximum reconstructed old-log-probability error is `6.34e-5`; the ledger
contains 306 steering projections and zero speed projections. These values
both revalidate the raw-latent probability contract and demonstrate that the
exploration component had essentially no temporal persistence. Over 0.5 s its
mean amplitude attenuated by about `1/sqrt(50)`.

That fact supports the external expert's concern: a sustained 0.1–0.5 s
avoidance maneuver was not explored as one coherent random variable. It does
not prove that temporally coherent noise would improve product KPIs; changing
noise would be a new experiment.

## 7. Curriculum shift is real but not a simple missing-condition bug

| Population | Episodes | Collision | Overtake | Follow |
|---|---:|---:|---:|---:|
| B4 curriculum | 480 | 180 (37.5%) | 180 (37.5%) | 120 (25.0%) |
| Austin product BC | 600 | 24 (4.0%) | 342 (57.0%) | 234 (39.0%) |

Collision prevalence was amplified `9.375x`. There were 381 unique L2
scenarios for 480 terminal labels, while the trainable head had 706,862
parameters. Transition count was large, but terminal task supervision still
came from only 480 episode outcomes.

At iter10, fixed/new collisions were concentrated in `raceline0` at speeds
0.7/0.8, `raceline1` at 0.5/0.6 and `raceline2` at 0.7/0.8. Every affected
coarse raceline/speed cell had Austin curriculum coverage. The 18 lost
overtakes clustered in four cells: six at `raceline0/0.8`, five at
`raceline0/0.7`, four at `raceline2/0.8`, and three at `raceline1/0.5`.

Therefore "the failing product conditions were absent from training" is not
supported at this coarse level. The stronger, narrower claim is that weighting
collision states about 9.4 times more heavily than the product population can
encourage a shared full-action head to learn a broad driving-style change.
Fine-grained startpoint/state coverage remains limited.

## 8. Frozen-feature result does not settle representation sufficiency

For the 11 fixed BC collision trajectories and 11 new iter10 collision
trajectories, the final-0.5-second frozen-feature means have:

```text
median cosine distance within fixed = 0.3944
median cosine distance within new   = 0.3353
median cosine distance cross-group  = 0.4002
nearest-neighbor opposite-label rate = 45.5%
centroid cosine distance             = 0.0904
```

This diagnostic does not reveal a clean fixed/new separation and is consistent
with the head applying similar responses to nearby interaction features. But
`n=11+11`, the histories are policy-induced, and cosine distance of a window
mean is only one representation probe. It cannot show that the GRU lacks the
information, nor that PPO could exploit it after an unfreeze. B4 therefore
leaves the representation hypothesis open.

## 9. Hypothesis adjudication

| Hypothesis | Judgment from current evidence |
|---|---|
| B4 failed because of collector/checkpoint/evaluator corruption | Not supported; smoke, replay and 2,400-row integrity evidence pass. |
| The output head was too small to learn anything | Rejected by large case turnover and monotonic action/parameter changes. |
| Updates accumulated relative to BC without a BC trust region | Strongly supported by shared-state action KL and global speed drift. |
| Direct-head parameterization lacked behavior-preserving structure | Strongly supported as an optimization diagnosis; not proof residual is uniquely required. |
| 100 Hz iid exploration was temporally incoherent | Directly established for noise; its causal contribution is supported, not proven. |
| Collision-heavy curriculum differed from product distribution | Directly established; causal contribution is plausible, not isolated. |
| Frozen GRU representation is insufficient | Unresolved. |
| Reward weight alone caused the negative | Not sufficient: iter20/30 worsened both collision and overtake. |
| Projection caused the negative | Not supported: 306 rare steer projections, zero speed projections. |
| More iterations would repair B4 | Contrary to the observed iter10 -> 20 -> 30 trajectory. |
| Seed1 proves all seeds/configurations fail | Not established; the owner authorized only seed1. |

## 10. Decision implication, without authorizing a run

No B3 fallback, seed0, GRU unfreeze, LR change, noise change, sampler change or
B5 run is authorized by this analysis.

If the owner later authorizes exactly one next hypothesis, the evidence ranks a
deployment-compatible **BC-relative trust region/action anchor** first. It
directly addresses the measured monotonic BC-relative drift while retaining the
plain 12-key `End2Race` deployment checkpoint. All other scientific variables
should remain frozen for that test. This is a prioritization, not a prediction
of success.

Temporally coherent exploration ranks second because iid behavior is measured
exactly but its causal link to the product failures is less direct. Sampler
reweighting ranks third; coarse failing cells were already represented. GRU
unfreeze should not be selected from B4 alone because representation
insufficiency was not established.

## 11. External-review packet

An external reviewer should receive and inspect, in order:

1. `.agents/B4_DIRECT_HEAD_PPO_PLAN.md` — prospective frozen contract;
2. `.agents/B4_DIRECT_HEAD_PPO_RESULT.md` — execution integrity, product result
   and stop decision;
3. this document — claim-level causal diagnosis and limitations;
4. `scripts/analyze_b4_substantive_negative.py` and
   `tests/test_b4_negative_analysis.py` — reproduction and flag/log-prob
   regression;
5. `docs/ppo/evidence/b4_substantive_negative/README.md`, `summary.json` and the
   six TSV tables — compact evidence;
6. the Git diff from `4b06b7af0d6c84d45e688bd54478705ef021927f` to the final
   analysis boundary `dd49ce00bc82095a1cdd832caa485bce01c1991f` — actual
   implementation, remediation, execution record and analysis code/tables.

The ignored multi-gigabyte replay/NPZ release is not duplicated into Git. Its
location and input-ledger hashes are recorded so an authorized reviewer with
the retained workspace can reproduce every compact table.
