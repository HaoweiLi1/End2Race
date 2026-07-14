# B4 substantive-negative compact evidence

This directory contains the small, Git-reviewable outputs of
`scripts/analyze_b4_substantive_negative.py`. The source experiment is closed;
these files are a read-only diagnosis, not a RunPlan or authority to tune B4.

## Input boundary

- run: `b4_seed1_20260714_003027`;
- learner source: `9e5afdc9584343a163c4704597dad87487bd750a`;
- product grid: Austin, 3 opponent racelines x 4 speed scales x 50
  startpoints, 600 episodes per variant;
- variants: canonical BC and seed1 iterations 10, 20 and 30;
- large inputs remain under the ignored `Experiments/B4_direct_head_ppo/`
  release. `summary.json` records the hashes of the exact compact input ledgers.

## Files

| File | Meaning |
|---|---|
| `summary.json` | Provenance, product counts, training/update facts, exploration aggregate, feature-overlap diagnostic and claim boundaries. |
| `changed_cases.tsv` | Independent fixed/new collision and gained/lost overtake flags, plus explicitly inferred terminal collision geometry. |
| `action_drift.tsv` | Candidate-vs-BC action and equal-std Gaussian KL on the same 471,786 BC observation histories. |
| `parameter_drift.tsv` | Output-head parameter distance from BC and the previous snapshot; frozen tensors are checked separately in `summary.json`. |
| `exploration_noise.tsv` | Per-iteration reconstruction of raw Normal noise, lag-1 correlation, 50-step averaging ratio, projection count and log-probability error. |
| `condition_coverage.tsv` | Austin curriculum counts and product transitions by opponent raceline/speed condition. |
| `iter10_collision_precursors.tsv` | Same-state BC-vs-iter10 action differences in the final 0.5/1.0 seconds of the 11 fixed and 11 new collision trajectories. |

## Reproduction

From the repository root, with the retained B4 artifacts present:

```bash
python scripts/analyze_b4_substantive_negative.py --device cuda:0
python tests/test_b4_negative_analysis.py
```

The recurrent replay intentionally uses batch one and advances the GRU one
step at a time, matching `eval_multiagent.py`. A fused full-sequence call is
algebraically equivalent but can accumulate materially different floating
point hidden states in this model. The exact replay reproduces stored BC
actions to the errors reported in `summary.json`.

## Claim boundary

The tables establish outcome changes, BC-relative drift, iid exploration,
curriculum proportions and frozen-parameter identity. They support—but do not
causally prove—the diagnosis that missing BC-preserving constraints, high
frequency exploration and distribution weighting contributed to nonselective
updates. They do not determine GRU representation sufficiency, seed variance,
or whether any proposed B5 change would succeed.
