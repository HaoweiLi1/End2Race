# Isolated post-pass reward and loss probe

This directory is intentionally independent from the production PPO pipeline.
Nothing here is imported by `train_ppo.py` or by `ppo/`.

## Isolation contract

- Do not edit `train_ppo.py`, `ppo/`, `run.sh`, existing checkpoints, caches, or
  evaluation results.
- Use saved numeric NPZ traces read-only.
- Use one CPU thread and no CUDA.
- Refuse saved-episode or live validation while a training, evaluation, SUMO,
  or another post-pass audit process is visible.
- Fail closed when `/proc` is sandbox-filtered and cannot expose a credible
  host process table; heavy modes must run only after an external host check.
- Verify source hashes before/after live simulation and refuse to overwrite a
  previous live-probe directory.
- Write only beneath this directory's `outputs/` folder.
- Keep reward experiments separate from the standard PPO objective. The loss
  audit measures how a reward delta propagates through GAE and the existing
  clipped PPO/value objectives; it does not silently add a production loss.

## Files

- `shadow_contract.py`: self-contained geometry, reward, GAE/PPO-loss, and
  optional masked follow-teacher reference loss.
- `validate_shadow.py`: synthetic invariants and saved-episode replay. The
  expensive replay mode validates all 2,400 trace archives member-by-member
  (CRC, shape, dtype), reconciles JSON/NPZ/labels, and keeps only compact
  episode summaries in memory.
- `live_episode_probe.py`: final four-episode CPU-only simulator check after
  all external training/evaluation processes are idle. It selects two captured
  BC tail cases and two nearby untriggered safe-overtake controls, redirects
  traces here, and compares vector replay with the step state machine.

## Intended order

1. `validate_shadow.py --mode unit --candidate-module scripts/postpass_reward_calculation.py`
2. Wait until host training/evaluation is idle.
3. `validate_shadow.py --mode saved-episodes`
4. `live_episode_probe.py`

The formulas and fixed sweep are experimental evidence only. A passing probe
does not authorize wiring the treatment into the PPO pipeline.
