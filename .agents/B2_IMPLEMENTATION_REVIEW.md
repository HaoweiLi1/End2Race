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
merge. The control plane now requires one local BC-only baseline preflight:

```text
288 scenarios
24 any-agent collisions
138 corrected terminal overtakes
0 candidates and 0 updates
```

Its ordered L2/outcome/trajectory marker is bound to the training RunPlan and
source commit and copied to the remote control root. Both host preflights refuse
training without it. This is a correctness/baseline check, not a new model or
proxy gate.

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
