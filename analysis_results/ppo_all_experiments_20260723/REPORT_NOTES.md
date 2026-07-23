# Report notes

## Source inventory

- Raw model/training: `train_ppo.py`, `ppo/*.py`, `post-trained/*/run_config.json`, `metrics.jsonl`, checkpoints.
- Raw evaluation: 93 `results_multi.json` files and 55,799 NPZ traces under `eval_results/`.
- Core outputs: `eval_panels.csv`, `eval_episode_outcomes.csv`, `collision_episode_kinematics.csv`,
  `paired_vs_bc.csv`, `group_summary.csv`, `actor_parameter_deltas.csv`, `npz_audit.json`.
- Reproducibility: `analyze_all.py`, `validate_analysis.py`, and
  `ppo_all_experiments_tail_analysis.ipynb`.

## Chart map

| Chart | Question | Form | Reason |
|---|---|---|---|
| Current total path | Which current arm controls total collisions over training? | line | 10 ordered checkpoints across 45 updates |
| Current tail path | Does the target mechanism approach zero? | line | same ordered checkpoints and denominator |
| Collision mix | Which failure modes replace one another? | grouped bar | exact discrete class counts per checkpoint |
| Actor delta | Did policy tensors move and stay within the controlled frontend boundary? | line | ordered BC-relative parameter movement |
| Persistent scenarios | Which scenario identities fail across policies? | ranked bar | common denominator of 86 unique actor policies |

## Evidence boundary

- The report answers fixed-panel descriptive and paired questions, not cross-seed or cross-map generalization.
- Primary/relaxed/strict sensitivity is visible; no valid checkpoint reaches zero on relaxed or primary.
- Pose-derived slip proxy is not simulator tire slip angle.
- The isolated 599-row eval remains visible but is excluded from inference.
