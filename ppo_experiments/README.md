# PPO experiment records

This directory is the tracked authority for concise PPO evidence. Training code lives in `ppo/` and starts through `train_ppo.py`; records here do not authorize a new run.

- `environment.json` records the frozen source, dependencies, GPU, BC checkpoint, and Austin asset hashes.
- `verification.json` records the July 2026 refactor-equivalence gates.
- `v1/` and `v1_1/` retain the early pilot results. Their apparent development-panel winner was later rejected by the paired holdout recorded under `v1_2_reduced/`; it is not the deployment recommendation.
- `v1_2/` retains the completed critic/hard-pool evidence and the stopped original sweep.
- `v1_2_reduced/` is the final V1.x churn/repeatability/holdout authority.
- `signal_repair/` and `demo_repair/` retain the closed successor-route records.
- `v1_3/` is the unified V1.3 authority. Detailed immutable terminal evidence remains under `v1_3_a/`, `v1_3_b/`, and `v1_3_d/`.

V1.3-A and B are complete fail-fast negative results. V1.3-D completed all three fixed-U8 training runs and established controlled actor movement, but no candidate performance evaluation completed. V1.3-C formal, V1.3-E performance evaluation, and V1.3-M were incomplete and were removed after their completed mechanism/evaluator diagnostics were consolidated into `v1_3/`.

Raw runs and logs live under `runs/ppo/`; evaluation outputs live under `eval_results/`. They are retained only for complete terminal evidence. Canonical BC `pretrained/end2race.pth` remains the deployment authority.
