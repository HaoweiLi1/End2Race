# B6 temporally coherent exploration phase-0 plan

Status: **FROZEN PHASE-0 RUNPLAN; NO LEARNER AUTHORIZED**

Date: 2026-07-14

Implementation boundary:
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
hardware-independent two-dimensional standard Normal innovation `xi_t`.

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
max conditional log-prob replay error <= 1e-5
abs iid lag-1 correlation             <= 0.02, both action dimensions
AR(1) lag-1 correlation               in [0.93,0.97], both dimensions
marginal std relative error           <= 5%, both arms/dimensions
```

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
