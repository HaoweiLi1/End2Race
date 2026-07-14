# B6 temporally coherent exploration phase-0 plan

Status: **FROZEN REPLACEMENT PHASE-0 RUNPLAN; NO LEARNER AUTHORIZED**

Date: 2026-07-14

Initial implementation boundary:
`496c2d37e36a34897a50db8ba046208f9cc13656`

Frozen artifacts:

```text
selection digest: 7224f1f3da6a35febc50392cc35b4844076c77094f508d78dbe7b9b3fafb93fd
selection file:   7612d66ecd4708fb20905266c88033e5ffe6dd903aae4cdca310d7d1ae7c2b44
RunPlan file:     07a185042f26e87ca152e5d90ad9ea6a1a2ae0f35dc91c96a47d2b83adda60eb
```

## 1. Question

The only mechanism under test is whether, for the unchanged canonical BC
actor and unchanged per-dimension marginal action standard deviation, a
stationary AR(1) exploration process produces better direct closed-loop task
outcomes than the current 100 Hz iid process.

```text
simulator / GRU / actor mean: 100 Hz, unchanged
actor checkpoint:             canonical plain End2Race, unchanged
action projection:            unchanged
learning:                     none
iid:                          epsilon_t = sigma * xi_t
AR(1):                        rho=0.95, stationary initialization
```

This phase cannot show that PPO can learn an observed maneuver. It only tests
whether temporally coherent random exploration creates a more useful behavior
distribution without directly worsening safe behavior.

## 2. Corrections to the external proposal

The proposal correctly keeps the controller/evaluator at 100 Hz, tests AR(1)
before a learner, uses direct collision/overtake outcomes, and forbids Austin
or sealed data for mechanism selection. Five details must be fixed before
execution:

1. "More repair without more harm" needs a prospective effect-size and
   cluster-aware statistical contract; otherwise phase-0 can create another
   noise survivor.
2. The 1,440 rows are not independent. The training population has exactly 60
   L4 identities containing all three archived BC outcomes. L4 is the
   inferential and bootstrap cluster.
3. Safe cases need not be approximately hash matched. One collision, one
   overtake and one follow L2 are selected within each of the same 60 L4
   identities, giving exact matched triplets.
4. Common innovation seeds mean the same domain-keyed `xi_t` at a common step,
   not the same realized noise. AR(1) filters that innovation by definition.
   Arm execution order is separately hash balanced.
5. Collision repair alone is insufficient. Safe-to-collision harm and lost
   overtake are conjunctive gates. Projection and temporal persistence are
   reported directly.

The external document's future conditional log-probability formula is
correct. The implementation must recompute both current and previous
candidate means. Episode step zero uses the stationary marginal Normal, not
the smaller conditional innovation variance.

## 3. Frozen training-only population

Source universe:

```text
Task-8 1,640-row training partition only
81 collision / 1,001 overtake / 558 follow L2
development 288, Austin 600, fresh/final and sealed test excluded
```

Selection is fully deterministic and domain separated:

1. intersect L4 identities containing collision, overtake and follow;
2. require exactly 60 such L4 identities;
3. within each `(L4, archived outcome)`, select the SHA256-minimum L2 under the
   B6 selection domain;
4. retain all 60 matched triplets.

This yields 180 scenarios. Four fixed innovation seeds `(0,1,2,3)` and two
arms produce:

```text
60 L4 x 3 outcomes x 4 innovations x 2 arms = 1,440 episodes
```

Every episode uses the existing corrected terminal classifier and first
any-agent-collision / 8-second horizon semantics.

## 4. Common-random-number action contract

For each `(L2, innovation seed, step)` a SHA256/Box-Muller mapping produces a
global-RNG-independent two-dimensional standard Normal innovation `xi_t`.
The runtime and source identity are recorded; cross-platform libm bit identity
is not assumed.

```text
iid:
  epsilon_t = sigma * xi_t

AR(1):
  epsilon_0 = sigma * xi_0
  epsilon_t = 0.95 * epsilon_(t-1)
            + sqrt(1 - 0.95^2) * sigma * xi_t

raw action = canonical BC mean + epsilon_t
executed action = existing fixed projection(raw action)
```

The marginal standard deviation remains `(0.03 rad, 0.20 m/s)`. The common
innovation stream reduces paired simulation variance; it is not a claim that
iid and AR(1) execute identical perturbations.

Although there is no learner, each episode stores the appropriate behavior
log-probability and replays it from frozen features. This makes the proposed
future conditional probability contract testable without authorizing PPO.

## 5. Direct outcome estimands

For every matched `(L4, archived outcome, innovation seed)` pair:

```text
collision repair = archived collision and stochastic outcome is non-collision
safe harm        = archived overtake/follow and stochastic outcome is collision
lost overtake    = archived overtake and stochastic outcome is not overtake
```

The primary contrast is always `AR(1) - iid`. Occurrence McNemar is reported
descriptively. Exact conditional sign-flip inference and paired bootstrap use
the 60 L4 aggregate effects; repeated innovations and the three outcome rows
within one L4 are not treated as independent clusters.

## 6. Prospective conjunctive GO gate

### Integrity

```text
framewise max abs(ratio - 1)          <= 1e-6
iid batched max abs(ratio - 1)        <= 1e-4 (existing B4 contract)
AR(1) batched max abs(ratio - 1)      <= 3.3e-4
abs iid lag-1 correlation             <= 0.02, both action dimensions
AR(1) lag-1 correlation               in [0.93,0.97], both dimensions
marginal std relative error           <= 5%, both arms/dimensions
```

The AR(1) batched bound is not tuned to outcomes. It is the existing B4
`1e-4` batch-GEMM tolerance scaled by the conditional standard-deviation
factor `1/sqrt(1-0.95^2)=3.2026`, then rounded upward. Framewise replay remains
the hard probability-variable identity test.

### Collision repair

```text
net paired repairs      >= 12 of 240 collision pairs (equivalent to 5 pp)
L4 sign-flip one-sided p <= 0.10
```

### Safe collision non-inferiority

```text
AR(1)-iid safe-to-collision point difference <= 0
L4 bootstrap one-sided 90% upper bound       <= +0.02
```

### Overtake preservation

```text
AR(1)-iid lost-overtake point difference <= 0
L4 bootstrap one-sided 90% upper bound   <= +0.05
```

All four sections must pass. The 2% safe-collision and 5% overtake margins are
phase-0 non-inferiority limits on this matched training population, not
product claims. A GO authorizes only a separate learner proposal; it does not
authorize a learner automatically.

## 7. Evidence and stopping discipline

The run writes one atomic JSON per episode and is resumable without changing
the scenario, innovation or arm order. Final evidence contains the immutable
selection, RunPlan, all episode identities/outcomes/trajectory digests,
paired rows, cluster effects, noise persistence, projection ledger, summary
and report.

If any direct gate fails:

```text
rho=0.95 AR(1) phase-0: NO-GO
learner:               UNRUN
Austin 600:            UNTOUCHED
seed0 / sealed pool:   UNTOUCHED
```

No rho sweep, alternate seed count, learner, LR change, cap change, GRU
unfreeze or objective change is permitted as remediation inside this run.

## 8. Correctness-only replacement before valid outcomes

The first remote attempt from execution source
`8055eed8794fcbbea96f94d33874fb80da356558` was stopped incomplete after 98
of 1,440 episodes. No summary was generated and no outcome count was inspected.
An online integrity check observed maximum batch-replay log-probability error
`1.3589859008789062e-4`.

The cause was computational rather than scientific: the AR sampler carried
the private pre-addition float32 noise into the next step, while replay can
only reconstruct `stored_raw - replayed_mean`. The replacement sampler carries
that stored-latent displacement. A second distinction was also made explicit:
batch-one rollout and batched head GEMM can differ by normal float32
accumulation order, so the project-level B4 contract is
`max abs(exp(new_logp-old_logp)-1) <= 1e-4`, while the pure conditional formula
has an exact unit test.

The failed directory is preserved remotely as:

```text
/home/haowei/end2race_analysis/
  b6_temporal_phase0_8055eed.failed_ar_state_replay
```

The scientific selection, innovations, rho and outcome gates are unchanged.
A replacement RunPlan must bind the corrected implementation before any valid
episode is started; none of the 98 invalid-attempt rows may be resumed or
combined with it.

Active replacement boundary:

```text
corrected implementation: d71efe948d8b6d9523535840e1364e5608481051
selection digest:         7224f1f3da6a35febc50392cc35b4844076c77094f508d78dbe7b9b3fafb93fd
selection file:           7612d66ecd4708fb20905266c88033e5ffe6dd903aae4cdca310d7d1ae7c2b44
replacement RunPlan:      b3725809c65b5ac66aae4bfb853accc87c95af35fb4cf53d5d19039f09a679d5
```

The replacement selection is byte-identical to the original; only the
correctness implementation boundary and replay integrity contract changed.

## 9. Second correctness-only replacement

The execution source `5ad2f60f3b930c27b062c558bf53897ca127b745`
was also stopped incomplete, after 75 rows and before outcome analysis. Its
maximum batched ratio delta was `1.3494491577148438e-4`, just above the reused
B4 `1e-4` threshold. The stored-latent AR state was already correct.

This exposed a separate numerical issue: AR(1)'s conditional standard
deviation is `sqrt(1-rho^2)=0.31225` times the marginal standard deviation, so
the same batch-one versus batched-head mean rounding is amplified by about
`3.2026` in standardized log probability. The final replacement therefore
separates:

```text
framewise probability identity: hard <= 1e-6 ratio delta
iid batched numerical replay:   hard <= 1e-4
AR1 batched numerical replay:   hard <= 3.3e-4 (derived scaling)
```

The second failed directory is preserved remotely as:

```text
/home/haowei/end2race_analysis/
  b6_temporal_phase0_v2_5ad2f60.failed_batched_ratio_tolerance
```

Again, none of its rows may be reused. The final replacement keeps the same
selection, innovations, rho, outcome estimands and scientific gates.
