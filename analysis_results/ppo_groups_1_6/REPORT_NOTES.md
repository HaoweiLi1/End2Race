# Report notes

## Source inventory

- Raw training: `post-trained/*/config.json`, `post-trained/*/metrics.jsonl`, actor/critic checkpoints.
- Raw evaluation: `eval_results/*/results_multi.json`, `eval_results/*/traces/*.npz`.
- Bounded analytical outputs: `run_summary.csv`, `panel_metrics.csv`, `training_eval_join.csv`,
  `group_summary.csv`, `group_control_audit.csv`, `scenario_pairwise.csv`,
  `scenario_frequency.csv`, `trace_summary_selected.csv`, `npz_audit.json`.

## Report structure map

1. Technical summary and KPI strip.
2. Group-by-group findings with control boundaries.
3. Training telemetry diagnostics.
4. Evaluation/NPZ data-quality evidence and scenario identity.
5. Scope, methods, limitations, recommendation, further questions.

## Chart map

| Chart | Question | Fields | Form | Why |
|---|---|---|---|---|
| U5+ mean collisions | Which arm is consistently safer after warm start? | group, arm, mean collisions | grouped bar | Discrete arm comparison with a zero baseline |
| Group 5 path | How do clean clip arms move across checkpoints? | update, collision, clip | grouped bar | Seven fixed checkpoints, two directly comparable arms |
| KL vs eval collisions | Does KL telemetry rank safety? | mean KL, collisions, early stop | scatter | 68 points reveal dispersion and outliers |
| Persistent scenarios | Which scenarios repeatedly fail? | scenario, panel collision rate | bar | Exact ranking across a common 68-panel denominator |

## Omissions and caveats

- No wall-time chart: runs overlap, so elapsed time is not an algorithm-speed comparison.
- No long temporal line chart: each arm has only 5 or 7 formal checkpoints.
- No error bars: there is one training seed per arm.
- Legacy NPZ collision markers are not used as terminal collision truth.
