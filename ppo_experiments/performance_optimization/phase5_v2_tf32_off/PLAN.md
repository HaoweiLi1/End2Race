# Phase 5-v2 TF32-off audit execution record

## Provenance history

1. The initial legacy-artifact gate is retained as historical evidence only:
   `STOP_STALE_REFERENCE_ARTIFACTS / NO_MECHANISM_TEST_RUN`.
2. Owner authorized current-HEAD reference regeneration. R0, R1, zero-LR,
   fixed actions, the frozen rollout, state/RNG, and minibatch order were
   regenerated once on `e1c0d2b61e4ebc5c619f4c013dad330acf1fdfa0`.
3. Owner later withdrew the ratio-based logp gate near the FP32 floor and
   replaced it with max absolute logp error `<=1e-5`. The existing bundle was
   reused unchanged for every subsequent audit.

## Completed stages

1. Stage 0 rerun after experimental-script fixes: Float64 semantic PASS;
   `TF32_DOMINANT_FOR_CORE_FORWARD_NUMERICS / LOGP_ABSOLUTE_ERROR_PASS`.
2. Stage 1: R0/R1 contract bitwise exact; separate warm-up plus one measured
   full update for each.
3. Stage 2: A batch16 open-loop, 1,400-step teacher-forced, two-process
   R1/A closed-loop diagnostic, warm-up, and one measured full update.
4. Stage 3: B and C all-four-minibatch forward/loss/gradient/delta/Adam audit
   from identical model/optimizer/RNG/rollout/minibatch order.
5. Stage 4: B/C frozen actor/train timing and full-pipeline warm-up plus one
   measured update.
6. Stage 5: because A/B/C passed and B retained materially lower numerical
   error than C, both A+B and A+C received one warm-up and one measured full
   rollout/update with strict 12-key checkpoints.
7. Constructed collection, training, and full-update Pareto frontiers.

## Final disposition

`PHASE5_COMBINED_FORWARD_AB_REQUIRES_PRODUCT_TEST`

- BEST_LOW_ERROR: B
- BEST_BALANCED: A+B
- BEST_MAX_SPEED: A+C
- Unique next product-distribution candidate: A+B

No experimental backend was merged into the production pipeline.
