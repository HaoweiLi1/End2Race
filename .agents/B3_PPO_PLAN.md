# B3 Unified-Policy PPO Plan

Status: **prospective design implemented; no numerical RunPlan yet**  
Date: 2026-07-13  
Parent evidence: B2 training `b2_direct_20260713_081422`, evaluation
`b2_eval_20260713_165800`

Implementation record: `.agents/B3_IMPLEMENTATION_RECORD.md`

## 1. Objective and reason for B3

The owner-fixed lexicographic objective is unchanged:

1. corrected overtake must not be below BC;
2. subject to that constraint, any-agent collision target is `RR <= 0.70`.

B2 proved that PPO can reduce collision, but all six candidates lost overtake.
The primary implementation defect is a policy-identity mismatch: rollout PPO
optimized Bernoulli gates in raw-logit coordinates while product evaluation
used a separate centered threshold at the fresh `-6` bias. B3 removes that
second decision rule. It does not relax the product gate, restart warm-start,
change reward weights, open the fresh pool, or rank B2 arms.

## 2. Immutable historical boundary

- B2 plans, checkpoints and evaluation remain immutable FAILED evidence.
- B3 starts from canonical BC, the same sidecar initialization and fresh
  residual heads; no B2 candidate is resumed.
- `INITIAL_INTERVENTION_LOGIT=-6.0` and `INITIAL_BRAKE_LOGIT=-6.0` remain the
  raw-head initialization constants.
- Existing B2 classes and loaders remain available for historical validation.
  A B3 checkpoint must fail closed in a B2 loader and vice versa.

## 3. One effective policy distribution

For raw learned head outputs `z_I` and `z_B`, define:

```text
p_I0 = 0.10
p_B0 = 0.50

ell_I = z_I - INITIAL_INTERVENTION_LOGIT + logit(p_I0)
ell_B = z_B - INITIAL_BRAKE_LOGIT        + logit(p_B0)
```

At fresh initialization, `ell_I=logit(0.10)` and `ell_B=0`. Therefore stochastic
rollout has analytic `P(I)=0.10`, `P(B|I)=0.50`, joint brake probability 0.05,
while strict deterministic mode (`ell > 0`) is exact NO_OP. The old centered
threshold is forbidden in B3.

The same `ell_I/ell_B` must be used by:

- keyed rollout sampling;
- stored old log-probability;
- PPO replay/new log-probability and entropy;
- checkpoint reload;
- deterministic evaluation mode.

There is no top/brake logit-offset schedule in B3. Every stored behavior offset
must be exactly zero. Steer/brake sampling scales are fixed at 0.1/1.0 and are
stored/replayed as before; these scales do not change the deterministic Normal
mode. Sampled latent, requested residual, bound-preserving executed command and
logged joint probability must remain one auditable chain.

## 4. Dual and objective

Reward, collision cost, GAE, PPO clipping, optimizer LRs, curriculum, action
budgets and dual constants remain unchanged from B2. No reward-weight sweep is
authorized.

Because B3 stochastic rollout is the policy being optimized rather than an
external exploration phase, every completed episode counts toward dual warm-up.
The dual update function is called once after every complete rollout:

- iteration 1: records 16 episodes, no value update;
- iteration 2: reaches 32 episodes and may update after the rollout;
- the updated value can first affect iteration 3;
- initial value 1, clamp `[0,3]`, LR 0.5, EMA 0.2 remain unchanged.

Dual behavior is training-only smoothing. Product selection continues to use
paired corrected outcomes directly.

## 5. Training and evaluation budget

- arms: A/B/C unchanged;
- seeds: 0 and 1 unchanged;
- 16 complete episodes per iteration with the frozen 50/50 curriculum;
- **40 iterations**, fresh start; no automatic 60-iteration extension;
- iteration 20 is a saved training diagnostic only;
- iteration 40 is the one opened-development candidate snapshot;
- all six learners finish before any product evaluation or arm selection.

The 288-scenario opened-development evaluation and topology-matched BC 24/138
baseline remain unchanged. The fresh/final pool stays sealed unless one arm
passes the existing per-seed and pooled direction/product gates.

## 6. Blocking implementation tests

Only defects that would invalidate PPO accounting or the product decision are
blocking:

1. fresh deterministic A/B/C behavior is exactly BC with zero residual;
2. analytic fresh probabilities are 0.10/0.50/0.05;
3. B3 rejects centered mode and nonzero gate offsets;
4. unchanged checkpoint replay has ratio one within the existing float32 bound;
5. sampled latent, executed action and logged probability are consistent;
6. composition remains bound-preserving with zero external clipping;
7. B2 and B3 checkpoint/action-policy schemas cannot be confused;
8. checkpoint/resume preserves effective priors, dual, optimizer and RNG state;
9. dual first becomes eligible only after 32 completed B3 episodes;
10. a B3 RunPlan is exactly 40 iterations and contains no external gate-offset
    schedule or fresh-pool input.

TTC, Brier, supervised witness recall and warm-start calibration are nonblocking
historical diagnostics and must not be reintroduced as B3 admission gates.

## 7. Execution boundary

Implementation, unit/integration tests, dry-run plan generation, isolated
staging and no-learning preflight are authorized by the owner's instruction to
rewrite and implement. Creating the first two-host numerical B3 RunPlan and
burning GPU requires the implementation record to show every §6 invariant
passing and must use a clean committed source. No push is implied.
