# Phase 5B production integration — verification report

Owner decision executed: integrate Phase 5B only. Collection stays on the
exact batch-1 path; the PPO training actor replay is replaced by the
validated per-timestep active-sequence batched FP32 GRU implementation.
No A, no C, no backend selectors. Production diff:
`ppo_phase5b_training_replay.patch` (repo root). Nothing committed or pushed.

## Production changes

- `ppo/policy.py`: new `_actor_replay_batched` (validated B), called only
  from `evaluate_actions`; `_actor_forward` restored to the pure batch-1
  collection loop (the Phase-4 skip branch moved into B, which subsumes it —
  invalid padded positions never enter the actor).
- `train_ppo.py`: module-level `torch.backends.cudnn.allow_tf32 = False`,
  `torch.backends.cuda.matmul.allow_tf32 = False`,
  `torch.set_float32_matmul_precision("highest")`,
  `torch.backends.cudnn.benchmark = False`, set before any CUDA model
  execution on every entry path that builds the model. The prior R0/R1 audit
  showed this flag flip is bitwise-neutral for the existing pipeline.
- `tests/test_ppo_batched_replay.py`: 5 unit tests (independent batch-1
  reference; padded-layout equivalence, invalid-position actor-call count and
  bitwise-carried hidden, episode-start reset, n_seq=1 bitwise identity with
  the collection path, masked-gradient cosine/relative-L2).

## Review items

1. Active indices are built in slot order from `valid_by_timestep` and
   written back via explicit slot mapping — original sequence IDs preserved
   (unit-tested against an independent reference).
2. `hidden[:, indices]` gathers initial hidden in the same order as the GRU
   batch rows; `next_hidden[:, offset]` returns to the owning slot.
3. Invalid padding never enters the actor (unit test counts processed rows
   == valid positions) and its hidden is carried bitwise unchanged.
4. Outputs are restored to the original padded-flat layout (layout test).
5. Sequencer, masks, advantage normalization, losses, and optimizer-step
   boundaries untouched: U1 minibatch mask/advantages/old-logp bitwise
   identical across sides; 4 optimizer steps at U1, 8 after U2 on both.
6. Critic C0 and collection unchanged: U1 rollout buffer hash bitwise
   identical between the HEAD reference worktree and the integrated tree.
7. `save_actor` untouched; strict 12-key checkpoints load on both sides.

Scope note: the measured contract covers the C0 profile on N1-H1F-p50
(seed 20260917). Other critic profiles use the same replay math (all slots
active) but have no measured contract artifact.

## Test runs

- B unit tests: 5/5 PASS.
- Full suite (includes existing PPO tests): 18/18 PASS.

## Performance (one warm-up, one measured update, central_subproc, 6 workers)

| | rollout s | train s | total s | total t/s |
|---|---:|---:|---:|---:|
| HEAD reference (recorded) | 19.024 | 15.182 | 34.432 | 743.5 |
| B integrated | 19.395 | 3.538 | 23.135 | 1106.5 |

Train improvement **76.7%** (gate ≥65% PASS); total-update improvement
**32.8%** (gate ≥25% PASS). Padding ratio 1.50 unchanged; rollout outcomes
identical to the reference rollout (13/9/13).

## Fixed-seed two-update reference-vs-B (seed 20260917)

Reference ran in a detached worktree at HEAD `e1c0d2b`; both processes used
the same TF32-off flags. U1 rollouts bitwise identical. Gates
(worst-of-4-minibatches; artifact `TWO_UPDATE_COMPARISON.json`):

| Gate | Bound | Measured | |
|---|---|---:|---|
| policy KL | ≤1e-8 | 4.56e-11 | PASS |
| gradient cosine | ≥0.999999 | 0.9999999995 | PASS |
| parameter-delta cosine | ≥0.999999 | 0.99999999996 | PASS |
| gradient relative L2 | ≤1e-4 | 3.24e-5 | PASS |
| parameter-delta relative L2 | ≤1e-4 | 9.37e-6 | PASS |
| policy-loss absolute difference | ≤1e-6 | 2.68e-7 | PASS |

Values match the validated pre-integration audit (gradient relative L2
3.24e-5 there as well). U2 rollouts diverge as expected: the U1 parameter
delta (relative 9.4e-6) feeds back through closed-loop collection.

## Full-600 evaluation of the final U2 actors

| Actor | collision | follow | overtake | error |
|---|---:|---:|---:|---:|
| BC baseline (recorded) | 21 | 233 | 346 | 0 |
| reference U2 (`cfbbfaf5…`) | 21 | 233 | 346 | 0 |
| B U2 (`18942de4…`) | 25 | 231 | 344 | 0 |

11 per-scenario outcome flips between the two U2 actors. This difference is
the documented consequence of U2 rollout divergence (any nonzero U1 delta
changes the U2 collection trajectory), not a B replay defect: the U1
contract is exact within gates, and the flip count and collision shift sit
inside the previously measured ε-close checkpoint churn band on this panel
(V1.x: collision 20.72±3.05, ≈7 scenario flips between adjacent
checkpoints). No gate was pre-registered on this comparison; it is recorded
for external review.

## Artifacts

- `b_integration_warmup.json`, `b_integration_repeat1.json` — performance.
- `reference_meta.json` / `b_integrated_meta.json` + `*_capture.npz` — two-update
  captures; `TWO_UPDATE_COMPARISON.json` — gate evaluation.
- `reference_u2_actor.pth` / `b_integrated_u2_actor.pth` — strict 12-key U2
  checkpoints; `*_u2_full600.json` — evaluations.
- `run_two_updates.py`, `compare_two_updates.py` — harnesses (experiment
  directory only; not part of the production diff).
