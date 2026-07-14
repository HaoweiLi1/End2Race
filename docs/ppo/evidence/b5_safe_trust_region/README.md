# B5-A external-review evidence

This directory is the compact Git packet for the B5-A seed1 result. The
Austin panel is opened development, not fresh/final confirmation.

## Files

- `reference_build.json`: all 64 selected training L2/L4 rows, strata,
  source identities and framewise replay checks.
- `reference_audit.json`: canonical BC and B4 iter10/20/30 `D_safe` values.
- `training_summary.json`: atomic 30-iteration learner release envelope and
  actor/full-checkpoint identities.
- `training_iterations.jsonl`: complete per-iteration cap attempts,
  accepted/skipped epochs, rollout/critic metrics, projections and outcomes.
- `product_summary.json`: prospective feasibility and selected snapshot.
- `product_report.md`: human-readable merged table.
- `product_paired_rows.tsv`: all BC/B5 paired rows for 600 cases x 4 variants.

Large reference features, replay NPZs, simulator NPZs and checkpoints remain
in the ignored experiment roots documented in
`.agents/B5_SAFE_TRUST_REGION_RESULT.md`.

## Result

```text
verdict: OPENED_DEVELOPMENT_SURVIVOR
selected: seed1_iter10
BC:      collision 24, overtake 342
iter10:  collision 22, overtake 347, fixed/new 9/7, gained/lost 12/7
target:  collision <=16 (not reached)
fresh/final opened: false
```

The learner source commit is
`482491969b01a632f5726b81316953397c6abd49`; the immutable RunPlan SHA is
`20e0af679b13f8ab1e3ee296ffe11189a8c584cc2ef363384fddc8e04d16af63`.

