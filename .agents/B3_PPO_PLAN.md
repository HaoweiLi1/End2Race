# B3 Unified-Policy PPO Plan

Status: **implementation and independent boundary review complete; no numerical RunPlan yet**
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

The conditional-brake boundary is deliberately learnable rather than a second
deployment rule. At `ell_B=0`, the Bernoulli distribution has probability 0.5
and maximum entropy. Its entropy derivative with respect to the logit is zero
at that maximum; the PPO learning signal instead comes from the sampled
log-probability derivative `d log p(a) / d ell_B = a - 0.5`, which is `+0.5`
for brake and `-0.5` for no-brake. Thus PPO gradients can move the same decision
surface later used by standard deterministic deployment. The deterministic
comparison remains strict: with intervention active, `ell_B == 0` means
no-brake and any positive `ell_B` means brake.

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
4. with intervention active, conditional-brake `ell_B == 0` selects no-brake
   while `ell_B > 0` selects brake;
5. unchanged checkpoint replay has ratio one within the existing float32 bound;
6. sampled latent, executed action and logged probability are consistent;
7. composition remains bound-preserving with zero external clipping;
8. B2 and B3 checkpoint/action-policy schemas cannot be confused;
9. checkpoint/resume preserves effective priors, dual, optimizer and RNG state;
10. dual first becomes eligible only after 32 completed B3 episodes;
11. a B3 RunPlan is exactly 40 iterations and contains no external gate-offset
    schedule or fresh-pool input.

TTC, Brier, supervised witness recall and warm-start calibration are nonblocking
historical diagnostics and must not be reintroduced as B3 admission gates.

## 7. Execution boundary

Implementation, unit/integration tests, dry-run plan generation, isolated
staging and no-learning preflight are authorized by the owner's instruction to
rewrite and implement. Creating the first two-host numerical B3 RunPlan and
burning GPU requires the implementation record to show every §6 invariant
passing and must use a clean committed source. No push is implied.

## 8. Frozen execution plan and expected wall time

The implementation is frozen by the following local commits:

- `19e83aed96126a61d9a848135fe860adc17ec48f` — unified B3 policy,
  runner, evaluator and control-plane implementation;
- `c320e83` — implementation checkpoint documentation;
- `21085bc` — explicit conditional-brake `0 / +epsilon` boundary regression
  and the matching §3 learning explanation.

The owner-relayed independent review verified the boundary test and the
policy-gradient argument and returned **GO**. No additional TTC, warm-start or
representation admission gate is authorized.

The next execution must use one immutable RunPlan and this order:

1. `./run.sh plan-b3` from a clean committed worktree;
2. inspect the complete plan with `show` before any staging;
3. stage the isolated source/input bundles on local and remote hosts;
4. reproduce the topology-matched BC baseline (`24` collision / `138`
   corrected terminal overtake), pass both host preflights, then run the
   existing four-map plumbing smoke and publish the shared `READY.json`;
5. run all six learners: A/B/C x seeds 0/1, exactly 40 iterations each, with
   seed1 queued locally and seed0 queued remotely; do not filter using an
   early seed or iteration-20 diagnostic;
6. collect and validate all six iteration-40 checkpoints;
7. freeze one B3 EvalPlan and evaluate BC plus six candidates on the existing
   288 opened-development scenarios (2,016 paired rows), using local shard 0
   and remote shards 1–3;
8. merge once and report corrected-overtake feasibility first, then collision
   RR and paired transitions. No arm is selected unless all frozen gates pass.

Measured B2 timings provide the planning estimate. Three local 20-iteration
learners took 2 h 47 min; doubling to 40 iterations makes the local queue the
expected 5.5–6 h bottleneck. The unchanged 288x7 evaluation previously took
about 1 h 20 min. Plan review, staging, baseline/preflight/smoke, collection
and reporting add about 0.75–1.25 h. Therefore the expected end-to-end wall
time is **7.5–8.5 h**, or **9–11 h** with network/recovery contingency.

This estimate ends at the opened-development KPI report. It does not include
fresh/final-pool confirmation. If no arm passes, B3 stops. If an arm passes,
the fresh/final pool remains sealed until a separate prospective confirmation
plan is reviewed and authorized.
