# B2 pre-GPU implementation audit

Date: 2026-07-12 (Asia/Singapore)
Scope: B2 code/control-plane correctness before any six-learner numerical run
Product target: corrected overtake feasibility first, any-agent collision RR second
Status after remediation: **GO_FOR_STAGING; no learner has run**

## 1. Independent audit

Claude Code was invoked read-only with `--model opus --effort max`; the serving
model was `claude-opus-4-8`. It read the approved plan, current diff, B2 policy,
rollout, replay, PPO, checkpoint, evaluator, CLI, runner and tests. It did not
edit files, use a GPU/simulator, access the remote host or propose a proxy
experiment.

Initial verdict: `NO_GO_FOR_GPU`, with one blocking defect.

### Blocking finding: batch-shape float32 replay equality

Production rollout computes old log-prob/entropy sequentially at batch size 1,
whereas pre-update replay computes the full rollout in one batch. The original
gate used `torch.equal`. Deterministic kernels guarantee repeatability for the
same shapes, not bitwise invariance across different GEMM shapes.

Codex measured the actual mechanism after perturbing trainable policy weights:

```text
CPU:  torch.equal=False, max |delta log_prob|=1.0252e-05,
      max |delta entropy|=2.2352e-07
CUDA: torch.equal=False, max |delta log_prob|=8.1062e-06,
      max |delta entropy|=3.2783e-07
```

Fresh zero heads happened to compare bitwise equal, which is why a same-shape
unit test could hide the production failure.

Prospective remediation, before any B2 numerical rollout:

- freeze the replay integrity bound at `1e-4` for max absolute log-prob and
  entropy drift;
- record both observed maxima and `max |ratio-1|` every iteration;
- keep all wrong-latent/offset/std/context failures fail-closed (their errors
  are orders of magnitude larger);
- change the regression to collect old terms row-by-row at batch size 1, then
  replay the 129-row rollout in one batch; the same test also exercises a
  singleton final minibatch;
- amend the approved plan so it no longer promises impossible bitwise
  shape-invariance.

## 2. Earlier internal implementation blockers and disposition

The Codex collaboration audits found these additional blockers before the Opus
review. All were fixed prospectively before numerical execution:

1. **Singleton/minibatch-dependent objective** — actor advantages are now
   scaled/normalized once over the complete rollout and sliced unchanged; a
   singleton tail minibatch is valid.
2. **Resume was serialization-only** — explicit resume now restores and checks
   plan/config, curriculum prefix, checkpoint/replay hashes, scenario-repeat
   cursor, RNG, dual, collision scale, critics and all optimizers. A torn JSONL
   tail and uncommitted iteration files are preserved under attempt-failure
   evidence and execution returns to the last complete boundary.
3. **Replay evidence disappeared after update** — each iteration now saves a
   portable compressed macro ledger with every field required by the plan,
   plus top/conditional/joint action and per-head entropy diagnostics.
4. **Eval checkpoint not bound to training plan** — each candidate must come
   from a COMPLETE learner release and its checkpoint envelope must name the
   EvalPlan's exact parent training RunPlan.
5. **Exit 0 could masquerade as COMPLETE** — host execution and collection now
   independently require the atomic COMPLETE directory and typed arm/seed/plan
   output envelope.
6. **Merge did not answer the product question** — merge now locks the live BC
   baseline to `24 collision / 138 corrected overtake`, reports per-seed and
   pooled direct transitions/KPI/slices/L4 bootstrap, and emits the frozen
   lexicographic direction/point-target verdict without selecting an arm.

## 2.1 Opus follow-up

After the replay-bound and BC-only preflight changes, the same read-only
`claude-opus-4-8` / max reviewer returned **`GO_FOR_STAGING`** with no blocking
fixes. It independently verified that the production-shaped test collects old
terms one row at a time, the code records and enforces the `1e-4` bound, the
baseline marker is bound to staged input hashes/RunPlan/source, and both host
preflights require it before execution.

Its two non-blocking suggestions were also completed before staging:

- the isolated BC-preflight Numba cache is safely reusable after a failed
  unpublished attempt;
- the PPO runner regression now deliberately changes a serialized behavior
  offset and proves the replay gate raises.

## 3. Dual timing ruling

Iterations 1–9 do not count exploration-contaminated outcomes toward the dual.
Iteration 10 supplies the first 16 zero-exploration episodes; iteration 11
supplies the next 16 and can change the dual after its own on-policy update.
Iteration 12 is therefore the first actor update that can consume a changed
dual. This is intentional, recorded before outcomes, and does not mean the dual
was expected to move at iteration 10.

## 4. BC-only budget protection

The independent reviewer recommended reproducing the frozen 288-row BC result
before running six learners rather than discovering environment drift at final
merge. The original implementation ran all rows locally:

```text
288 scenarios
24 any-agent collisions
138 corrected terminal overtakes
0 candidates and 0 updates
```

The first live attempt correctly failed closed at `24/139`. Forensics proved
that only physical row 199 (`L2:e2fd…64f0`, a 6.958 mm historical follow)
changes from safe-follow on the remote RTX 4080 to terminal-overtake-only on
the local RTX 3080. The final evaluation assigns that row to remote shard 3.
Therefore the local-all-288 gate was prospectively replaced, without changing
the 24/138 target, by the exact final topology: local shard0 and remote shards
1–3, each 72 rows and each with its frozen collision/overtake counts.

The four atomic candidate-free shard files and merged marker bind the training
RunPlan, source/input archives, manifest, BC, producer host/GPU and physical
row assignment. Count failure is terminal and preserves all 288 rows; it cannot
be retried into a pass. Both host preflights still refuse training without the
same merged marker. This is a correctness/baseline check, not a model or proxy
gate.

## 5. Non-blocking observations retained honestly

- clipping counters equal zero only because any actual clipping raises before a
  result is published; they are structural integrity evidence, not model score;
- the simulator intentionally resets NumPy to the canonical seed per episode;
  action noise and minibatches use domain-separated keys, while saved global RNG
  state covers other stochastic state;
- the 288 panel is opened-development and not fresh/generalization evidence;
- standard `raw logit > 0` evaluation remains diagnostic-only; centered fresh
  threshold is the prospectively frozen primary contract;
- no D2 test/final pool, TTC gate, third warm-start or exploration sweep is
  authorized by this review.

## 6. Remaining operational gates

Before numerical learners:

1. rerun the complete standalone regression suite and record the one known
   migrated historical warm-start path failure separately;
2. commit the exact source/docs/tests so `git archive` can pin a clean tree;
3. create/show/dry-run one immutable RunPlan;
4. stage both isolated roots;
5. pass the one BC-only 288-row baseline preflight and both host preflights;
6. run the frozen four-map all-arm plumbing smoke without using outcomes to
   tune anything;
7. only then execute all six 20-iteration learners and the exact 288x7 paired
   evaluation.

## 7. P3/control-plane pre-run audit closure

Before the first immutable RunPlan, the four-map/all-arm P3 gate and its
control evidence were audited in three read-only `claude-opus-4-8` / max
passes plus two independent Codex collaboration audits.

The first Opus pass returned `NO_GO` because a new branch-invariant regression
caught its own `AssertionError` sentinel and therefore could not turn red. The
guard itself was correct; the test was rewritten to capture the exact expected
message and assert outside the `try`. The same pattern was removed from the
older wrong-replay-context test.

The parallel adversarial audits then reproduced and fixed authorization gaps
before any GPU run:

- baseline/P3 markers now bind exact frozen manifest/checkpoint identities,
  exact key sets and strict nested scalar types; numeric SHA, `False == 0` and
  `4.0 == 4` attacks are negative tests;
- baseline rows enforce the four-state truth table and recompute 24/138;
- P3 reports only scenario identity plus boolean branch/replay/update
  integrity, never product outcomes, trajectory length, minibatch count or
  arm scores;
- valid markers are retryable, but dangling symlinks, external hardlinks,
  different published finals and nonfinite replay values fail closed;
- a private copied inode is validated before atomic remote install;
- one identical cross-host `READY.json` binds RunPlan, source/input archives
  and exact baseline/P3 marker hashes;
- preflight and execute/resume revalidate `STAGED`, every extracted tracked
  source byte and the exact runtime-input inventory; the learner CLI also
  requires READY;
- collection is status-first, saves both status/event ledgers and gate copies,
  records validation failures, preserves failed partial attempts and supports
  a clean retry without path nesting.

The second Opus pass found one remaining live race: the package environment was
recorded at preflight but not re-probed at execute. The final implementation
now checks the pinned Python/torch/numpy/numba/gym/scipy environment alongside
the GPU identity inside the outer per-GPU `flock`, on both fresh and resume
paths. It also made remote fresh-root creation fail closed with `set -eu`.

The final targeted Opus verdict was **`GO_FOR_COMMIT`**, with no remaining
blocker. It explicitly traced the local and SSH lock placement, live
environment/GPU re-probes, READY enforcement on direct `ppo-pilot`, and both
non-vacuous sentinel regressions. The two Codex follow-up audits independently
returned **GO** after reproducing the earlier malformed-marker attacks as
rejected.

Local execution evidence after the final changes:

- `tests/test_experiment_runner.py`: PASS;
- `tests/test_bplus_v22_ppo_runner.py`: PASS;
- complete standalone matrix: 39/40 PASS;
- the sole failure remains the known migrated immutable-path
  `test_bplus_v22_hierarchical_warmstart.py`, not a B2 regression;
- `py_compile` and `git diff --check`: PASS.

This closure authorizes a clean commit and isolated staging only. It is not a
P3, learner, KPI, D2-test or fresh-pool result.

## 8. Topology-matched baseline audit closure

After the first immutable RunPlan exposed the local-RTX3080 `24/139` versus
remote/final-topology `24/138` marginal outcome, the baseline gate was replaced
prospectively by the exact evaluation topology: local shard 0 plus remote
shards 1--3. A fresh read-only `claude-opus-4-8` / max audit initially returned
`NO_GO` for two contract-honesty defects rather than a control-plane defect:

- `.agents/README.md` still described the removed local-all-288 operation;
- the now-unreachable exported all-288 evaluator and its green test were still
  present as dead code.

Both were removed before commit. The runner also now checks that its per-shard
expectations cannot drift from the evaluator's acceptance tuples. The same
Opus reviewer then returned **`GO` with no blocker**, after tracing every shard,
merge, transfer, terminal-failure, preflight and direct-learner path. Two Codex
adversarial audits independently returned GO.

The complete standalone matrix after those changes is 39/40 PASS. The sole
failure remains the known migrated immutable-path
`test_bplus_v22_hierarchical_warmstart.py`; all B2 topology/control/learner
tests, `py_compile` and `git diff --check` pass. This authorizes a new commit and
new RunPlan only. It does not authorize reuse of the failed old RunPlan or a
change from the frozen 24/138 topology result.
