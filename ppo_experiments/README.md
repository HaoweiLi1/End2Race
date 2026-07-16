# PPO experiment record

This directory contains the concise, reproducible evidence retained from PPO V1, V1.1, and the user-stopped V1.2 sweep. It is historical data only; formal training is configured in `ppo/config.py` and starts through `train_ppo.py`.

- `environment.json` records the frozen source, dependencies, GPU, BC checkpoint, and Austin asset hashes.
- `verification.json` records the refactor equivalence gates.
- `v1/` and `v1_1/` retain the selected results and training metrics.
- `v1_2/` retains the completed critic stage, one completed Stage H arm, hard-pool summary, and explicit incomplete status.

Raw evaluation rows, copied scenario panels, command files, process logs, locks, heartbeats, and failed attempts are intentionally omitted.
