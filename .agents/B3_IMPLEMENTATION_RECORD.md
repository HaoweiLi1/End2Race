# B3 unified-policy PPO implementation record

Status: **IMPLEMENTED AND CPU-CONTRACT-TESTED; NO NUMERICAL RUNPLAN CREATED**  
Date: 2026-07-13  
Design authority: `.agents/B3_PPO_PLAN.md`  
Parent result: B2 training `b2_direct_20260713_081422`, evaluation
`b2_eval_20260713_165800`

Implementation commit: `19e83aed96126a61d9a848135fe860adc17ec48f`

## 1. What B3 is fixing

B2's stochastic rollout and deterministic product evaluation did not use the
same gate decision surface. PPO optimized Bernoulli probabilities in raw
`-6` coordinates, while the primary evaluator applied a separate centered
threshold at the fresh bias. The B2 candidates therefore supplied valid
evidence about the deployed centered rule, but the deployed decisions were not
the mode of the distribution optimized by PPO.

B3 removes the second rule. Sampling, old/new log probability, entropy,
checkpoint reload and deterministic evaluation all use one effective gate
distribution. Historical B2 classes and releases are preserved and continue to
load under the centered contract.

## 2. Prospective mathematical decision

Claude's useful core recommendation was to move the deployable decision
boundary into the model and stop using a centered evaluator. The literal
`effective = raw - (-6)` formula was not adopted because it makes a fresh gate
probability 0.5; combining that with B2's old exploration offsets would produce
an unsafe fresh intervention rate.

B3 instead freezes explicit stochastic priors:

```text
effective_intervention = raw_intervention - (-6) + logit(0.10)
effective_brake        = raw_brake        - (-6) + logit(0.50)
```

Consequences at fresh initialization:

- raw head constants remain byte-for-byte `-6`;
- `P(intervene)=0.10`;
- `P(brake | intervene)=0.50`;
- `P(joint brake)=0.05`;
- strict standard deterministic mode is NO_OP because `logit(0.10) < 0`;
- no top/brake behavior offset schedule exists in B3.

This is one policy distribution, not a deployment threshold layered on top of
another policy. Normal steer/brake scale metadata remains replayed because it
is part of the stochastic latent distribution; all gate offsets must be zero.

## 3. Implementation map

### Policy and probability identity

`bplus_v22/remediated_model.py`

- adds `UnifiedV22Policy` and versioned B3 prior buffers;
- derives the effective logits from learned raw heads;
- rejects centered deterministic mode;
- rejects persistent or per-rollout nonzero gate offsets;
- exposes standard mode as the primary deterministic contract;
- rejects B2 state in the B3 loader and vice versa.

### PPO training, replay and checkpoint continuity

`bplus_v22/ppo_runner.py`

- versions B2/B3 policy, pilot and checkpoint schemas;
- accepts a B3 RunPlan with exactly 40 iterations;
- builds fresh B3 policies from canonical BC and the existing sidecar
  initialization, never from B2 or warm-start action-head checkpoints;
- stores zero gate offsets with every B3 macro and replays them through the
  same effective distribution;
- preserves keyed action sampling, bound-preserving executed action,
  serialized old log-probability, PPO ratio and entropy accounting;
- calls the existing bounded overtake dual after every complete B3 rollout;
  iteration 2 reaches the 32-episode eligibility point and iteration 3 is the
  first actor update that can consume the changed value;
- saves B3 iteration-40 as the final candidate checkpoint while retaining B2
  iteration-20 behavior for historical runs.

### Product evaluation

`bplus_v22/ppo_eval.py` and `bplus_v22/cli.py`

- the evaluator selects the primary mode from the loaded policy contract;
- B2 remains centered-primary with standard mode diagnostic-only;
- B3 uses standard mode as primary, so primary/standard action counts must be
  identical and are not labeled as a diagnostic comparison;
- B3 EvalPlans require B3 checkpoint schema and iteration 40;
- external clipping remains fail-closed and paired collision/overtake outcome
  accounting is unchanged.

### Immutable two-host control plane

`Experiments/runner.py`

- adds `plan-b3` and versioned `b3_train` / `b3_eval` kinds;
- freezes six learners (A/B/C x seeds 0/1), 40 iterations, zero gate offsets
  and standard deterministic evaluation;
- uses final iteration 40 when freezing candidate checkpoints;
- keeps the existing local seed1 / remote seed0 learner queues and local 1/4 /
  remote 3/4 evaluation topology;
- reuses the topology-matched BC 24-collision / 138-overtake preflight and the
  staged-source/READY fail-closed workflow.

## 4. Tests executed

All of the following standalone programs passed under
`/home/haowei/miniconda3/envs/end2race/bin/python` with `PYTHONPATH=.`:

```text
tests/test_bplus_v22_exploration.py
tests/test_bplus_v22_objective.py
tests/test_bplus_v22_ppo.py
tests/test_bplus_v22_ppo_buffer.py
tests/test_bplus_v22_ppo_env.py
tests/test_bplus_v22_remediated_model.py
tests/test_bplus_v22_ppo_runner.py
tests/test_bplus_v22_ppo_eval.py
tests/test_experiment_runner.py
```

The new assertions cover:

- exact 0.10 / 0.50 / 0.05 fresh probabilities;
- fresh A/B/C standard deterministic NO_OP;
- with top intervention active, effective conditional-brake logit equality at
  zero selects no-brake and a `+1e-4` perturbation selects brake;
- keyed sampling near the analytic prior;
- unchanged replay ratio one;
- centered/nonzero-offset rejection;
- B2/B3 checkpoint mismatch rejection;
- B3 checkpoint/resume and policy-only loading;
- dual update eligibility after 32 completed episodes;
- B3 standard-mode evaluator accounting;
- B3 EvalPlan checkpoint schema/iteration 40;
- exact 40-iteration control config and final-checkpoint envelope.

The Gym deprecation notice is pre-existing and non-failing. The experiment
runner test intentionally exercises a terminal BC baseline failure path before
printing `ALL TESTS PASSED`.

The broader compatibility sweep is **20/21 B+ standalone programs passing**.
The sole failure is the already documented
`test_bplus_v22_hierarchical_warmstart.py`: an immutable historical release
still names its pre-migration `logs/...` sidecar path. The release bytes are
preserved and must not be edited merely to make a path-resolving legacy test
green. All other historical B+ identity, warm-start, closed-loop, sidecar and
release tests pass.

## 5. What has not happened

- no B3 immutable RunPlan has been created;
- no source or input archive has been staged;
- no local or remote GPU has run B3;
- no simulator product evaluation has run;
- no candidate, arm or seed has been selected;
- the D2/fresh/final pool remains sealed;
- no new collision RR or corrected-overtake result exists.

Therefore this record is implementation evidence, not a scientific result.

## 6. Review checklist for Claude or the next agent

Review code rather than the narrative, focusing on these exact questions:

1. Does every B3 sampling/replay/evaluation path use the same effective logits?
2. Can any nonzero intervention/brake logit offset enter a B3 macro ledger?
3. Is the sampled latent exactly the latent whose log-probability is stored and
   whose bound-preserving composition reaches the simulator?
4. Does standard deterministic evaluation use the trained distribution's mode,
   with no centered fallback?
5. Can B2/B3 checkpoint schemas or final iterations be confused by CLI/runner?
6. Does the dual timing really make iteration 3 the first update affected by a
   post-32-episode multiplier?
7. Are there remaining hard-coded B2 iteration-20 or centered assumptions in
   any executable B3 path?
8. Does the 40-iteration budget introduce any unreviewed objective, reward,
   curriculum, action-budget or product-gate change?

Post-implementation audit addendum: the explicit B3 conditional-brake
`0 / +epsilon` boundary regression and the policy-log-probability gradient
explanation are now part of the fixed review surface. This closes the only
missing contract identified by the first Claude review without changing B3
behavior or any numerical setting.

Do not demand a new TTC, warm-start or proxy-quality gate. A blocking finding
must threaten policy/log-probability correctness, action execution, checkpoint
continuity or the direct lexicographic KPI decision.

## 7. Next allowed operational step

This exact source was committed as `19e83aed96126a61d9a848135fe860adc17ec48f`.
After independent review, create one
unique B3 RunPlan with `./run.sh plan-b3`, inspect it with `show`, and run only
the existing staging, topology BC baseline, host preflight and plumbing-smoke
phases. Numerical learners must not start unless those gates bind the same
committed source and all six jobs to a shared READY marker.
