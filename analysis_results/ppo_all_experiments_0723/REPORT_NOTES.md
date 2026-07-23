# Report notes

## Source inventory

- Training source: `post-trained/*/run_config.json`, `metrics.jsonl`, `episodes.jsonl`, actor/critic checkpoints.
- Evaluation source: `eval_results/*/results_multi.json` and per-episode NPZ traces.
- Bounded audit tables: parameter matrix, update metrics, panel metrics, 55,200-row PPO-vs-BC comparison plus 600 BC self-reference rows,
  collision feature table, tail-mechanism table, scenario frequency, and paired comparisons.

## Structure

1. Answer-first safety and tail-swing conclusion.
2. Data-quality and exact-reproduction boundary.
3. Recorded parameters, full experiment paths, and training telemetry.
4. Exact scenario-paired evaluation and collision commonality.
5. Auditable tail-swing classifier, threshold sensitivity, conclusion, and limitations.

## Visualization choices

| View | Question | Form | Reason |
|---|---|---|---|
| Candidate path | Do 45U, clip .25, or hard-neighbor improve? | grouped bar | Exact checkpoint values and discrete arms |
| Tail comparison | Is the target failure mechanism reduced? | bar | Small exact counts across representative checkpoints |
| Persistent scenarios | Which episodes fail repeatedly? | ranked bar | Shared 92-panel denominator |

## Exclusions

- No wall-time comparison because runs may overlap and hardware utilization is uncontrolled.
- No inferential error bars because there is one training seed and the panel includes duplicated physical starts.
- No true slip-angle/contact-point claim because those simulator fields were not saved in trace NPZ.
