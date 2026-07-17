# INVALID_EXPERIMENT

Completed: 2026-07-17T14:16:05.974193+00:00

## Integrity failure

IntegrityFailure: Quick-panel BC composition mismatch: {'collision': 20, 'error': 0, 'follow': 49, 'overtake': 51, 'total': 120}

- Expected: collision 21, follow 49, overtake 50, error 0.
- Observed: collision 20, follow 49, overtake 51, error 0.
- Paired drift: `evaluation-sp30-ego1283-raceline0-v070` changed from historical `ego_collision` to current `overtake`.
- No PPO training process or run directory was started.

## Phase 0

- H1_EARLY_3S count: 138
- Filter threshold: 2.8 s
- BC reproduction: 138/138 (100.000%)

## Frozen quick panel

- Manifest hash: `447a996bf7d9617e86950e456cd8cb08d68ba9b8f349cab7959bc205f437bf52`
- File SHA-256: `7c633628d039215a00b13b77376f255bebeacff2b990561ea62b34f439cf7918`
- Screen, retention, and full-600 stages were not run because the BC precheck was invalid.

## Changed files

```text
 M .gitignore
 M eval_multiagent.py
 M ppo/config.py
 M ppo/scenarios.py
 M train_ppo.py
?? ppo_experiments/quick_pool_3s/
```

## Exact commands

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/pretrained/end2race.pth --output /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/H1_3S_CLASSIFICATION.json --workers 8 --scenario-manifest /home/haowei/Documents/End2Race/ppo/hard_pools/h1_expanded_det.json --sim-duration 3.0
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/pretrained/end2race.pth --output /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/H1_EARLY_3S_PRECHECK_RESULTS.json --workers 8 --scenario-manifest /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/H1_EARLY_3S.json --sim-duration 3.0
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/pretrained/end2race.pth --output /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/QUICK_PANEL_BC_RESULTS.json --workers 8 --scenario-manifest /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/QUICK_PANEL_120.json --sim-duration 8.0
```
