# End2Race Safety-First B+ v2.2 — Objective-Aligned Route-R2 Policy Redirect

Version: `bplus-v2.2-owner-redirect-3`  
Recorded: 2026-07-11T22:08:00+08:00  
Prospective Task-6 remediation amendment: 2026-07-12T09:47:24+08:00  
Prospective post-Task-10 action/calibration amendment: 2026-07-12T15:27:13+08:00  
Owner decision: approved prospectively by the project owner  
Authority: `CURRENT_HANDOFF.md` §§13 and 17–18, then this specification.

## 1. Supersession and claim preservation

This document prospectively redirects the next B+ stage from representation
proof to policy-level optimization. It supersedes only these earlier rules:

- TTC `<2s` MAE `<=0.30s` is no longer a prerequisite for D3/PPO;
- failure of the D2/D2R TTC proxy no longer requires another representation
  family before a policy-level experiment;
- the unopened D2 probe test is retired sealed and is not needed by v2.2.

It does not retroactively alter any result. In particular:

- D2 selected no family under its original complete gate;
- D2R-G remains
  `STOP_D3_TEST_UNOPENED_D2R_G_FAILED_TTC_AND_2S_FA` under its locked gate;
- neither result may be relabeled as a pass;
- their OOF metrics, manifests, registry rows, and unopened-test state remain
  immutable evidence.

Current project-level decision:

`OWNER_REDIRECT_BPLUS_V22_DIRECT_POLICY_KPI_TTC_DIAGNOSTIC_ONLY`.

The redirect is a project-owner decision made before any v2.2 policy result.
It is not a post-hoc threshold change within D2/D2R. TTC, Brier score, alarm
recall, geometry error, and calibration remain diagnostics, but none may
replace the policy objective or veto a policy solely by itself.

This document authorizes implementation and structural preflight under the
single controlling primary assistant. It does not authorize an unattended
remote Codex goal. Policy training may start only after the implementation
plan's structural, registry, manifest, and zero-residual identity checks pass.

The project owner prospectively amended Task 6 after the first fixed-update
smoke exposed an all-NO_OP brake gate. The immutable release
`warmstart_smoke_20260712_091950` remains a valid integrity artifact but its
Task-6 stage decision is `FAILED`: every arm had `gate_recall=0`, and every
arm's gate BCE exceeded the newly frozen marginal-predictor bound. Its files,
metrics, hashes, and internal integrity validation are not rewritten.

This amendment does not invalidate either 16/16 zero-residual identity
release. `INITIAL_BRAKE_LOGIT=-6.0` remains the byte-for-byte fresh-model
constant used by structural identity. The empirical-prior bias below is a
separate, warm-start-only re-initialization applied after fresh identity and
immediately before supervised Task-6 fitting.

The first deterministic closed-loop warm-start release,
`task10_warmstart_20260712_105740`, is preserved with output-manifest hash
`605d3413df35cef8ddd9cdd4769164f52016edeaa7c9e58e1c34ba234fb9ed46`.
Its artifact integrity passed, but its mechanism decision remains `FAILED`:
all arms lost net overtakes, introduced collisions, and required steering
clipping. No arm was ranked or selected. The project owner prospectively
approved the hierarchical-action and held-out-calibration remediation below
before any replacement Task-6 fit or replacement Task-10 result. The old
three-dimensional checkpoints remain valid historical Task-6 releases, but
the new action-schema decision revokes their eligibility for future PPO.

## 2. Fixed lexicographic objective

Let `C(pi)` be any-agent collision rate and `O(pi)` the corrected terminal
safe-overtake rate on a paired, pre-registered scenario population. The
objective remains:

```text
minimize C(pi)
subject to O(pi) >= O(BC) - delta_O
```

Training/development non-inferiority tolerance remains `delta_O = 1`
percentage point. Final product selection is stricter:

For a locked paired development population of `N_gate` episodes, implement
the 1-point tolerance exactly as an integer transition constraint:

```text
allowed_net_overtake_loss = floor(0.01 * N_gate)
lost_overtake - gained_overtake <= allowed_net_overtake_loss
```

Apply it to each seed's complete pooled development manifest and to the pooled
two-seed accounting. Do not replace it with the stricter zero-loss count gate.
Mechanism-enriched map/skill panels are collapse diagnostics, not separate
zero-loss gates. If `N_gate < 100`, the mathematically exact 1-point tolerance
allows zero whole-episode losses; the manifest should be pooled before the
first result rather than changing rounding after observing outcomes.

- Austin fresh pool collision `RR <= 0.70` versus BC;
- known-cross-map fresh pool collision `RR <= 0.70` versus BC;
- one-sided 95% collision upper bound excludes `RR = 1` in both pools;
- candidate overtake point count is at least BC in both pools;
- paired one-sided 95% overtake risk-difference lower bound is greater than
  `-1` percentage point;
- no material map, raceline, speed, or skill-stratum collapse;
- at least one paired BC-collision case converts to a confirmed safe pass.

No TTC, alarm, representation, warm-start loss, PPO reward, or training curve
may substitute for these closed-loop outcomes.

## 3. Evidence that motivates the redirect

### 3.1 Deployable risk information exists

On the same 1,928-episode / 1,505,848-frame non-test population:

| representation | 1s recall | 1s safe FA | 1s Brier skill | TTC<2 MAE |
|---|---:|---:|---:|---:|
| frozen-feature temporal T1 | 0.648 | 0.075 | 0.078 | 1.073s |
| T2 with deployable LiDAR/speed/command history | 0.725 | 0.095 | 0.105 | 1.068s |
| D2R-G beam-local deployable encoder | 0.868 | 0.099 | 0.130 | 0.800s |

This establishes that deployable observations contain useful predictive risk
signal and that the tested frozen-feature-only route is materially weaker.
It does not prove that freezing alone caused the difference: raw inputs,
beam-local architecture, capacity, and multi-task supervision also changed.
It also does not prove PPO can recover the supervised signal from sparse
environment reward.

### 3.2 False alarms make direct action coupling unsafe

A read-only post-OOF alignment audit applied each episode's outer-fold
threshold and joined D2R predictions to corrected D0.1 BC outcomes. Among
1,823 safe episodes, the 1-second head alarmed in 180:

- safe corrected overtakes: 128 / 1,139 (11.24%);
- confirmed safe passes: 110 / 1,061 (10.37%);
- safe follows: 52 / 684 (7.60%).

An alarm is not an executed brake and these counts are not predicted lost
overtakes. They show why the alarm head must not be wired directly to a fixed
brake rule. v2.2 measures the full `alarm -> action -> outcome` chain.

### 3.3 Route-R2 has a confirmed reachable set, not a ceiling

D2.5 found a non-clipped confirmed-safe-pass witness for 67 of 91 non-test BC
ego-collision cases using the fixed bounded branch library. The precise claim
is:

> The fixed D2.5 library demonstrated a confirmed safe-pass recovery for
> 67/91 tested non-test ego-collision cases, spanning four maps and 54 L4
> blocks; this satisfies the pre-registered Route-R2 feasibility conditions.

This is not a 73.6% theoretical ceiling, not a population collision-reduction
estimate, and not evidence that the 24 exhausted cases are impossible under
other policies or actions. It does not cover opponent-only collisions and
does not prove a deployable actor can select witness actions.

## 4. Single causal question and three arms

The v2.2 pilot asks:

> Does a deployable, pretrained risk sidecar—and specifically allowing that
> sidecar to adapt at a small learning rate—improve the collision/overtake
> frontier when policy mechanics, action bounds, data, seeds, and selection
> are identical?

All three arms keep the 11,516,908-parameter BC actor backbone frozen. Full BC
backbone unfreezing is outside this comparison.

### 4.1 Common 128-dimensional policy interface

Every arm supplies one 128-dimensional policy feature to the same residual
action core.

- Arm A — `BC_FROZEN`:
  current frozen 1,680-dimensional BC recurrent feature through a trainable
  `Linear(1680,128) -> LayerNorm -> SiLU` adapter.
- Arm B — `SIDECAR_FROZEN`:
  the 128-dimensional fusion feature from one full-non-test D2R-G sidecar;
  all sidecar parameters are frozen during warm-start and PPO.
- Arm C — `SIDECAR_FINETUNE`:
  byte-identical initialization to B; the sidecar is trainable at
  `3e-6`, exactly one tenth of the common residual action-core learning rate
  `3e-5`.

The common action core is
`Linear(128,128) -> LayerNorm -> SiLU`, followed by the identical action and
value interfaces described below. A is permitted to train only its adapter
and common action core; B trains only the common action core; C trains the
sidecar and common action core at their locked learning rates.

The sidecar uses the locked D2R deployable inputs and causal taps
`(0,5,10,20,35,50,75,100)`: 360-beam LiDAR, actual ego speed, previous
desired steer/speed, and current frozen BC feature. No pose, progress, map,
raceline, opponent state, collision label, future observation, D2.5 witness
identity, or branch outcome is a deployment input.

### 4.2 Full-non-test sidecar initialization

OOF fold models remain evidence models and are not selected for deployment.
Before policy warm-start, fit exactly one initialization sidecar on all 1,928
opened non-test episodes using the already locked D2R-G architecture, seed,
sampling, loss, six epochs, and optimizer. This fit:

- opens no sealed D2 test data;
- performs no architecture or threshold selection;
- is released as `SIDECAR_INITIALIZATION_ONLY`, not a D2R gate pass;
- appends prospective D3-R2 registry rows before fitting;
- supplies byte-identical initialization to B and C.

A separate immutable copy of the full-fit sidecar and its classification
heads is used only for online diagnostics in all arms. In C, PPO gradients
must not mutate this shadow diagnostic copy.

## 5. Common macro residual policy

### 5.1 Temporal semantics

The BC actor and simulator remain at 100 Hz. The residual policy chooses one
latent action every `K=10` micro-steps and holds that latent for at most 0.1
seconds; terminal events may shorten the macro transition. The requested
physical residual is recomposed from the held latent and the current 100 Hz
deployed BC command at every micro-step. Holding a physical delta computed at
the macro boundary is forbidden.

```text
Gamma = 0.997^10
Lambda = 0.99 per macro transition
R_macro = sum(i=0..k-1) 0.997^i * r_(t+i), 1 <= k <= 10
```

One macro choice produces one stored policy action, one log probability, one
reward/cost transition, and one PPO ratio. Repeating one macro log probability
as ten actions is forbidden.

### 5.2 Hierarchical action support

The post-Task-10 action schema is versioned as
`bplus-v2.2-hierarchical-residual-action-1` and stores exactly
`[I, z_steer, B, z_brake]`, where:

- `I in {0,1}` is the top-level `NO_OP`/`INTERVENE` atom;
- `z_steer` requests a steering residual in `[-0.2,+0.2]` radians only when
  `I=1`;
- `B in {0,1}` is the conditional brake gate and must satisfy `B <= I`;
- `z_brake` requests a brake magnitude in `(0,1.0)` m/s only when `I=B=1`.

Canonical storage is mandatory: `I=0` stores all three conditional fields as
positive zero; `I=1,B=0` stores `z_brake` as positive zero. This represents
the observed D2.5 steer-only, brake-only, and steer-plus-brake labels without
making steering a permanently active channel. `NO_OP` is an exact atom and
the deterministic **fresh-model** initialization. Both the fresh top-level
intervention gate and the unchanged conditional brake gate have zero weights
and bias exactly `-6.0`; `INITIAL_BRAKE_LOGIT=-6.0` is not changed.

The latent probability and entropy are respectively:

```text
log p(a|s) = log p(I)
           + I * [log p(z_steer) + log p(B|I=1)
                  + B * log p(z_brake)]

H(a|s) = H(I) + P(I=1) * [H(z_steer) + H(B)
                          + P(B=1|I=1) * H(z_brake)].
```

Implementation uses conditional branches/`where`, never `0 * log_prob`, so
an inactive branch cannot create `NaN` from an underflowed conditional
density. PPO stores and replays the canonical four-dimensional latent; its
log probability is a latent-measure probability, not a density over the
state-dependent physical command.

At each 100 Hz micro-step, first form the deployed BC base
`[clip(raw_steer,-0.52,0.52), max(raw_speed,0)]`. Then request
`tanh(z_steer)*0.2` and `sigmoid(z_brake)*1.0`, and project those requested
residuals onto the current base's steering/braking headroom. This projection
preserves every already-feasible D2.5 residual and prevents an infeasible
request from reaching evaluator clipping. `I=0` returns the deployed BC base
through an exact branch. The composition ledger records raw/deployed base,
requested and applied residual, both steering headrooms, brake headroom,
final command, and whether the evaluator's historical clip would change it.
Any such external change is a blocking integrity failure.

Positive speed residual is fixed at zero in all arms. D2.5 established
Route-R2 feasibility without positive speed; historical positive speed
residual also introduced new collision risk. A future positive-speed
experiment requires a separate prospective decision based on observed
closed-loop lost-overtake mechanisms.

The implementation logs both gate probabilities/choices, both latents,
requested and applied residuals, composition headroom, composed command,
external-clipping identity, and frozen diagnostic sidecar outputs.

## 6. Route-R2 warm-start

All arms receive the same action-level warm-start before PPO:

- positive trajectories are the 67 exact D2.5 confirmed-safe-pass witnesses;
- intervention macro ticks use the witness steering and brake labels;
- non-intervention ticks on those trajectories use `NO_OP`/zero-steer labels;
- preservation examples are selected from corrected confirmed-safe-pass BC
  episodes using a single map/skill/raceline/L4-stratified hash rule frozen in
  the implementation release before fitting;
- preservation ticks use BC/no-op residual labels;
- D2.5 poses, progress, future results, and witness identity never enter the
  actor input.

Warm-start trains for a fixed number of updates with no validation-based
stopping. Its supervised loss cannot select an arm or substitute for policy
outcomes. The Task-6 mechanics gate below can block all arms from PPO, but it
cannot rank them or authorize promotion. The first meaningful policy
comparison remains paired closed-loop collision/overtake behavior.

### 6.1 Two initialization phases and warm-start-only empirical prior

Initialization has two distinct phases:

1. **Fresh/identity phase.** Construct `V22Policy` with the immutable
   `INITIAL_BRAKE_LOGIT=-6.0`, zero gate weights, zero steering mean, and the
   common seeded action-core initialization. Structural identity naturally
   chooses NO_OP at every checked macro boundary. The simulator then composes
   an explicitly checked physical zero residual. This phase owns the existing
   zero-residual identity evidence.
2. **Warm-start fit phase.** After the fresh initialization has been recorded
   and before the first Task-6 optimizer update, overwrite only the trainable
   brake-gate bias with the logit of the empirical brake prevalence in the
   frozen fit schedule. Do not change the fresh constant, gate weights, any
   BC parameter, sidecar parameter, shadow diagnostic, labels, schedule, or
   action support.

The fit prior is computed only from the exact scheduled training-label
occurrences, not from the diagnostic subset and not from an unweighted unique
example inventory:

```text
n_brake_fit = 90089
n_total_fit = 262144
p_brake_fit = 90089 / 262144 = 0.3436622619628906
warmstart_gate_bias = log(p_brake_fit / (1-p_brake_fit))
                    = -0.6470161225499584
```

The release must record the counts, decimal/hex prevalence, derived
float64/float32 bias, fresh state hash, post-prior/pre-update state hash, and
final state hash for every arm. A/B/C receive the same empirical prior.

The frozen 873-example diagnostic subset has 200 brake-positive labels. Its
best constant marginal predictor has BCE
`H(200/873)=0.538180595747381`; the prospective operational bound is the
strict inequality `gate_loss < 0.5382`.

Task 6 passes only if every arm, on that unchanged diagnostic subset, satisfies
all of:

- `gate_recall > 0`;
- `gate_loss < 0.5382`;
- `gate_specificity > 0.05`, prospectively defining and rejecting near-all-
  positive collapse;
- finite precision and specificity are reported, along with the full gate
  confusion counts.

These are shared action-mechanics acceptance bars, not an arm ranking. If any
arm fails, Task 6 fails for all arms; produce and preserve a failed release,
do not start PPO, and do not proceed to Tasks 7/8.

### 6.2 Checkpoint continuity into PPO

The checkpoint that feeds each PPO pilot is the exact, independently validated
warm-started checkpoint for that same arm from a Task-6 release with all
acceptance bars passed. Warm-start is supposed to move the policy away from
BC; near-NO_OP is a property only of fresh initialization and is not a
post-warm-start requirement.

The future PPO entry point must require the warm-start release path, arm,
checkpoint file SHA256, state-dict SHA256, and
`task6_acceptance_passed=true`. It must fail closed if asked to construct a
fresh `-6.0` policy for PPO, load another arm, or load an unaccepted/failed
Task-6 checkpoint. At the time of this amendment the v2.2 PPO entry point does
not yet exist; this requirement resolves the prior specification ambiguity
before its implementation.

### 6.3 Post-Task-10 replacement Task 6

This subsection prospectively supersedes §6.1 only for the new hierarchical
action schema. It does not rewrite either old Task-6 release. New checkpoints
must use schema `bplus-v2.2-hierarchical-warmstart-checkpoint-1`, load with
`strict=True`, and reject any old three-coordinate/single-gate state without
missing-key initialization.

Reuse the frozen D2 outer-fold assignment
(`scenario_split.tsv`, SHA256
`2f8146d7be0e36c3abcc084dcdbfa9e3df85983c37c6249294ab19b1431c49f3`).
The new action-head fit uses folds 0--3 only:

- 58 witness episodes / 47 witness L4 blocks;
- 252 intervention and 4,446 witness-NO_OP macros;
- 484 preservation episodes / 39,204 preservation-NO_OP macros;
- 542 episodes and 43,902 unique macros in total.

Outer fold 4 is excluded from every optimizer update and both initialization
priors. It supplies nine positive witness episodes / seven L4 blocks / 39
intervention macros (14 steer-only, 25 steer-plus-brake, no brake-only). The
negative calibration population is exactly the 75 fold-4 strict
confirmed-safe-pass candidates that entered neither the old 602-episode
warm-start selection nor the frozen 288-row Task-8/10 development manifest.
They span 31 L4 blocks and 6,075 macros. Pin these source files:

- old episodes: `baaa916db54364308458413d81c26e52b31585c07d1eecf1c8f5c1a8ca0bda20`;
- old macro examples: `01043ad1b02a4948b51140944b7f8736b493e02bf689be9e99b90454f7983f93`;
- canonical episodes: `793193deefc942f556ec23ee4e34fea3597eac761eb0b1f676af2667ff6b62e2`;
- Task-8 development TSV: `8ff0d96b91aac134ab006e70900785c13c345dcb544867740aa8dd57072dfc46`.

The 75 negative episodes are already opened non-test data, but their new use
directly selects a deployed intervention offset. Before any score is computed,
prospectively append one registry-reuse row per L2 under
`D3-R2-v2.2 / actor_pretrain / action_choice`, with a distinct hierarchical
calibration split ID. The manifest must freeze registry before/expected-after
snapshots and hashes; the fit runner appends idempotently, verifies the exact
75-row transition, saves the observed after snapshot, and refuses any other
live registry state. Existing D2, D2R, D2.5, sidecar, and old Task-6 rows are
never rewritten.

This is L4-held-out evidence for the newly fitted action head only. All nine
positives were used in the historical Task 6, four appear in Task 8, and the
B/C sidecar was full-non-test fitted; therefore it is neither historically
fresh nor representation/generalization evidence. The 75 negatives contain
no `skill_F`; that limitation must be recorded and Task 10 must retain its
separate `skill_F` panel.

The fixed training schedule cycles outcome-blind over the 43,902 unique fit
macros; it must not recreate the old 128/64/64 class-mixture schedule. The
top-level prior is the unique-fit `I=1` prevalence `252/43902`; the
conditional brake prior is computed only within those 252 intervention
macros (`175/252`). The release records exact counts and float64/float32
logits. Loss masks are fixed:

- intervention BCE on all fit macros, with a frozen fit-count positive
  weight to make the rare class learnable;
- steering physical-residual loss only where `I=1`;
- conditional brake BCE only where `I=1`;
- brake-magnitude physical-residual loss only where `I=B=1`.

After all fixed optimizer updates, calibrate each arm independently. For each
of the 75 negative episodes, take the maximum raw intervention logit over all
its macros, order by `(logit descending, fixed domain hash, L2)`, and place
the threshold just above the eighth-largest maximum with `nextafter`. The
resulting offset is stored in the checkpoint and is used by deterministic,
sampling, rollout, and replay paths. Negative labels alone select it; positive
scores may only pass/fail the following frozen mechanics bars. Deterministic
`INTERVENE` means `raw_logit + stored_offset > 0`; equality is `NO_OP`. All
bars below are required
for every arm:

- false-intervention episodes `<=7/75` on the negative calibration set;
- intervention-window episode recall `>=6/9`;
- intervention-macro recall `>=20/39`;
- steer-only episode recall `>=2/4` and brake-containing episode recall
  `>=3/5`;
- teacher-forced conditional brake recall `>=0.5` on the 25 positive-brake
  macros and specificity `>=0.5` on the 14 steer-only macros;
- finite precision, specificity, BCE, confusion counts, per-arm threshold,
  offset, raw-score distributions, and per-episode decisions are reported;
- exact NO_OP in both physical channels and zero external composition
  clipping remain structural gates.

The former `gate_loss<0.5382` bound remains part of the old single-gate
release only. It must not be relabeled as the new intervention bound: the old
diagnostic marginals were `H(200/873)=0.5381806`, while the corresponding
historical top-level and conditional marginals would be
`H(291/873)=0.6365142` and `H(200/291)=0.6212556`. New loss values are
diagnostics and cannot rank arms. Failure of any new bar fails all arms and
keeps PPO blocked.

## 7. Constrained PPO objective

The current scalar-return-only PPO is not sufficient for v2.2. Maintain
separate episode signals, critics, returns, and advantages for:

- primary any-agent collision cost;
- ego-involved collision diagnostic;
- terminal overtake / confirmed-safe-pass return;
- optional dense progress diagnostic.

The macro rollout buffer must additionally store the canonical four-value
latent action, old macro log probability, BC feature, LiDAR history, scalar
history, action/checkpoint schema, calibration offset, macro length, and
termination/truncation state. A replay under the unchanged policy must
recompute one ratio exactly equal to one per macro. Physical command/delta
must never be substituted for the latent PPO action.

Policy optimization minimizes collision cost while a nonnegative dual
variable enforces the overtake floor. Collision and overtake advantages are
normalized separately. Actor, collision critic, and performance critic use
separate optimizers or separate clipping operations with pre/post-clip norms.

The dual schedule is locked before training:

- initial dual `lambda_0 = 1.0`;
- maximum dual `lambda_max = 3.0`;
- dual learning rate `0.5` per rate-violation unit;
- completed-episode overtake-rate EMA coefficient `0.2`;
- no dual update until 32 completed training episodes have been observed;
- after the minimum count, update the EMA and clamp
  `lambda <- clip(lambda + 0.5 * (floor - EMA_O), 0, 3)`;
- use the bounded actor signal
  `(-A_collision + lambda * A_overtake) / (1 + lambda)`.

Because both advantages are normalized, `lambda_0=1` gives equal initial
scale rather than a pure-collision first update. The cap keeps at least 25% of
the combined standardized signal on collision reduction. Some lag and
oscillation around the overtake floor remain expected properties of the dual,
especially in a 20-iteration pilot. A dual rise/fall or a transient training
overtake dip is diagnostic and cannot by itself kill an arm; only scheduled
closed-loop paired snapshot outcomes apply the development gate.

Checkpoint selection is lexicographic regardless of the smooth training
objective:

1. reject candidates that fail overtake protection;
2. among survivors, minimize paired collision deterioration / maximize paired
   net collision fixes;
3. break ties in favor of the earlier snapshot.

TTC shaping is disabled. The immutable D2R head is logged only as a causal
diagnostic: warned/not warned, action/no action, and final paired transition.

## 8. Data, registry, and comparison integrity

- Use only previously opened D0.1/D2 non-test data for sidecar initialization,
  warm-start, training, and development selection.
- Keep the 1,108-episode D2 test sealed and unused.
- Do not touch the fresh Austin or cross-map final pools during arm choice.
- Freeze one training manifest, one development manifest family, and seeds
  before the first arm result.
- All three arms use identical scenarios, ordering, stochastic seeds,
  update counts, action budgets, PPO mechanics, and evaluation jobs.
- Registry reuse is appended under stage `D3-R2-v2.2`; existing D2, D2.5,
  and D2R rows are never rewritten.
- D2.5 witness L4 blocks used for warm-start must not be presented as
  held-out policy generalization evidence.

Development reporting must include representative, `skill_F`, `skill_S`,
map, raceline, speedscale, and D2.5-recoverable/non-witness slices. Training
or opened development results are never final evidence.

The frozen v2.2 development release is L2-disjoint from its PPO training
list, but it is not L4-disjoint from Route-R2 warm-start witnesses. Every one
of its 288 rows is explicitly `held_out_policy_generalization=false`, and no
independently proven recoverable case exists outside the witness L4 set. Thus
Task 10 and all pilot/medium results on these panels are mechanism and
within-opened-development evidence only. A collision fix may reflect replay
of a witness-like action; it must not be described as L4 generalization.

## 9. Pilot ladder and objective-aligned gates

### 9.1 Structural preflight

Before any policy training:

- `K=1` replay matches the current PPO semantics;
- zero residual at `K=10` reproduces BC actions and evaluator output bitwise;
- one macro action/log-prob/ratio accounting is exact;
- partial macro termination and truncation bootstrap are correct;
- NO_OP/BRAKE log probabilities and entropy are independently checked;
- A/B/C initial BC action sequences are identical;
- B/C sidecar checkpoints are initially byte-identical;
- frozen parameters receive no gradient or mutation;
- C alone updates its policy sidecar, while the shadow diagnostic copy stays
  bit-identical;
- manifest, registry, source, and release validation pass.

For clarity, the historical zero-residual identity adapter was not a pure
forced-zero test. At each macro boundary it first required the fresh deterministic policy
to select NO_OP and map to exact physical zero; this is load-bearing on the
unchanged `-6.0` fresh bias and zero steering head. Only after that assertion
does the 100 Hz simulator path explicitly compose the checked zero tensor.
Thus both historical K=10 identity and identical initial A/B/C action-sequence
claims remain true fresh-initialization claims for the old action schema. The
prior-only Task-6 amendment did not invalidate them. The later hierarchical
action-schema change does: those releases remain historical PASS artifacts
but cannot certify the new top-level gate or composition. Before replacement
Task 6, run a fresh fitted-sidecar natural-NO_OP identity for the new schema,
requiring the naturally chosen top-level NO_OP, exact deployed-BC command,
bitwise full trajectories, and identical A/B/C initial action sequences.

After Task 6, Task 9 performs a distinct checkpoint-continuity preflight. It
must load each exact accepted same-arm warm-start checkpoint and execute its
natural deterministic hurdle decision for logged diagnostics, then explicitly
replace the physical residual with an exact zero tensor before simulator
composition. This checkpoint-backed forced-zero replay does **not** require
the warm-started gate to choose NO_OP; deviation from BC is expected after
warm-start. It verifies checkpoint envelope/state continuity, BC/evaluator
plumbing, repeated-run determinism, A/B/C Cartesian evaluation identity, and
K=10/short-terminal accounting without turning the Task-6 imitation metrics
into an arm ranking. It is no-learning evidence and cannot itself authorize
PPO; any forced-zero mismatch blocks the subsequent closed-loop warm-start
evaluation and PPO.

The replacement Task 9 must load only the new accepted hierarchical
checkpoint and log its natural four-dimensional decision before forcing the
physical residual to zero. The old Task-9 release remains historical evidence
for the old checkpoint and cannot authorize the replacement Task 10.

### 9.1.1 Post-Task-10 remediation ladder

The required order is: prospective spec/plan and source release; structural
tests; new fresh natural-NO_OP identity; frozen replacement Task-6 manifest;
remote CUDA fit and negative-only calibration; replacement checkpoint-backed
Task 9; then replacement Task 10. Task 10 runs each arm in three diagnostic
modes on the unchanged scenarios: full hierarchical action, steering disabled
after the intervention decision, and braking disabled after the intervention
decision. These ablations diagnose channels but cannot rank arms. Every mode
must retain bound-preserving composition and complete paired accounting.

No Task 11/PPO starts until the full-action replacement Task 10 has zero
external clipping, no global/near-global intervention collapse, and passes a
new independent audit of collision/overtake transitions. The old Task 10
remains `FAILED` regardless of the replacement result.

### 9.2 Closed-loop pilot

Each arm runs the same ladder:

1. load that arm's accepted warm-start checkpoint with exact release/file/
   state hashes, then run the fixed seed for 20 PPO iterations and the
   pre-registered snapshots;
2. if the objective-aligned mechanism gate passes, extend to 40 iterations;
3. add the second fixed seed only for surviving arms;
4. select at most one arm for an 80–100 iteration medium confirmation.

An arm survives development only if all are true:

- paired `fixed_collision > new_collision` on both seeds;
- each seed and the pooled accounting satisfy
  `lost_overtake - gained_overtake <= floor(0.01 * N_gate)` on their complete
  locked development populations;
- at least one BC collision becomes a confirmed safe pass;
- no map or skill slice shows an undisclosed collapse;
- collision improvement is not explained entirely by loss of interaction
  attempts;
- no action-clipping, replay, completeness, or source-integrity violation;
- both seeds move in the same safety direction.

Overtake protection is a feasibility constraint, not an objective to maximize
after the floor is met. Reject every overtake-inferior arm; among the remaining
arms select the largest paired net collision improvement. Break an exact
collision tie by larger paired net overtake improvement, then by the earlier
snapshot. TTC/Brier/alarm metrics cannot break a policy-outcome tie.

This ranking authorizes only internal development survival and selection of at
most one medium candidate. It is not policy generalization, product promotion,
or final evidence. Task 10 itself remains diagnostic and cannot rank arms.

If every arm fails, stop the pilot. Do not return automatically to a sixth
representation probe. Diagnose the observed `alarm -> action -> outcome`
failure as policy learning, action mapping, credit assignment, or
anti-conservatism before proposing a new stage.

### 9.3 Medium and final promotion

The selected arm must pass a two-seed 80–100 iteration medium confirmation
with the same direct transition gates before one final candidate is frozen.
Only then may the fresh final manifests be opened once. Final claims use the
product gates in §2 and the original paired clustered statistical hierarchy.
The medium confirmation still uses opened development evidence; it is not a
fresh-pool test. The first fresh-pool generalization evidence appears only at
the one-open final evaluation. If a development/medium gain shrinks there,
the gap is a measured generalization gap and does not authorize retroactive
changes to the development gate.

## 10. Explicit exclusions from v2.2

The following are not part of the A/B/C comparison:

- full or partial BC backbone unfreezing;
- `beta_bc` sweep or an arbitrary `0.5–1.0` anchor choice;
- positive speed residual;
- TTC reward/potential shaping;
- warning-threshold-to-fixed-brake control;
- sidecar architecture, tap, width, loss, or threshold sweep;
- opening or repurposing the sealed D2 test;
- selecting a new arm after final results are opened.

Any later experiment in one of these dimensions must be justified by the
closed-loop failure mechanism and change one causal factor prospectively.

## 11. Required releases

Each stage releases source hashes, environment fingerprint, registry snapshot,
scenario/model/job manifests, exact command/config, training logs, checkpoints,
paired episode outcomes, transition matrices, map/skill/raceline/speed slices,
alarm/action/outcome diagnostics, independent validation, output manifest,
and `COMPLETE` last.

The v2.2 terminal principle is:

> D2/D2R diagnosed deployable risk signal and D2.5 demonstrated a bounded
> confirmed-recovery set. The next decision is made by closed-loop collision
> and overtake outcomes, while TTC remains a diagnostic rather than a proxy
> veto.
