# NO_QUICK_SIGNAL

Completed: 2026-07-17T16:10:04.759399+00:00

## Reused Phase 0

- Full H1: 482
- H1 collision within 3 s: 152
- H1_EARLY_3S: 138 at <= 2.8 s
- Reproduction: 138/138
- Interpretation: early-failure subset, not full H1 coverage.
- Quick manifest hash: `447a996bf7d9617e86950e456cd8cb08d68ba9b8f349cab7959bc205f437bf52`
- Binding current CPU BC: collision 20, follow 49, overtake 51, error 0.

## Screen

| arm | collision | follow | overtake | fixed | new | net repair | gained | lost | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| QP3_A0_H0_8S | 16 | 52 | 52 | 5 | 1 | 4 | 2 | 1 | PASS |
| QP3_A1_H1FULL_8S | 14 | 49 | 57 | 6 | 0 | 6 | 6 | 0 | PASS |
| QP3_A2_H1EARLY_8S | 14 | 51 | 55 | 7 | 1 | 6 | 6 | 2 | PASS |
| QP3_A3_H1EARLY_3S | 16 | 51 | 53 | 6 | 2 | 4 | 4 | 2 | PASS |

## Retention

| arm | collision | follow | overtake | fixed | new | net repair | gained | lost | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| QP3_A0_H0_8S | 18 | 49 | 53 | 3 | 1 | 2 | 3 | 1 | FAIL |
| QP3_A2_H1EARLY_8S | 16 | 52 | 52 | 6 | 2 | 4 | 4 | 3 | PASS |
| QP3_A3_H1EARLY_3S | 19 | 49 | 52 | 2 | 1 | 1 | 2 | 1 | FAIL |

## Current CPU full-600 BC

- collision 22, follow 233, overtake 345, error 0

## Full 600 candidates

| arm | collision | follow | overtake | fixed | new | net repair | gained | lost | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| QP3_A0_H0_8S | 24 | 229 | 347 | 4 | 6 | -2 | 8 | 6 | FAIL |
| QP3_A2_H1EARLY_8S | 23 | 233 | 344 | 8 | 9 | -1 | 8 | 9 | FAIL |
| QP3_A3_H1EARLY_3S | 25 | 233 | 342 | 4 | 7 | -3 | 4 | 7 | FAIL |

## Scope

This is a preregistered quick-experiment result, not a final PPO improvement claim or held-out proof.

## Exact commands

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/train_ppo.py --config QP3_A0_H0_8S --seed 20260718 --output_dir /home/haowei/Documents/End2Race/runs/ppo/quick_pool_3s_v2/QP3_A0_H0_8S_seed20260718 --screen-pause
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/train_ppo.py --config QP3_A1_H1FULL_8S --seed 20260718 --output_dir /home/haowei/Documents/End2Race/runs/ppo/quick_pool_3s_v2/QP3_A1_H1FULL_8S_seed20260718 --screen-pause
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/train_ppo.py --config QP3_A2_H1EARLY_8S --seed 20260718 --output_dir /home/haowei/Documents/End2Race/runs/ppo/quick_pool_3s_v2/QP3_A2_H1EARLY_8S_seed20260718 --screen-pause
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/train_ppo.py --config QP3_A3_H1EARLY_3S --seed 20260718 --output_dir /home/haowei/Documents/End2Race/runs/ppo/quick_pool_3s_v2/QP3_A3_H1EARLY_3S_seed20260718 --screen-pause
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/quick_pool_3s_v2/QP3_A0_H0_8S_seed20260718/checkpoints/end2race_ppo_QP3_A0_H0_8S_u0002_s20260718.pth --output /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s_v2/screen_eval_QP3_A0_H0_8S.json --workers 8 --scenario-manifest /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/QUICK_PANEL_120.json --sim-duration 8.0
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/quick_pool_3s_v2/QP3_A1_H1FULL_8S_seed20260718/checkpoints/end2race_ppo_QP3_A1_H1FULL_8S_u0002_s20260718.pth --output /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s_v2/screen_eval_QP3_A1_H1FULL_8S.json --workers 8 --scenario-manifest /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/QUICK_PANEL_120.json --sim-duration 8.0
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/quick_pool_3s_v2/QP3_A2_H1EARLY_8S_seed20260718/checkpoints/end2race_ppo_QP3_A2_H1EARLY_8S_u0002_s20260718.pth --output /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s_v2/screen_eval_QP3_A2_H1EARLY_8S.json --workers 8 --scenario-manifest /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/QUICK_PANEL_120.json --sim-duration 8.0
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/quick_pool_3s_v2/QP3_A3_H1EARLY_3S_seed20260718/checkpoints/end2race_ppo_QP3_A3_H1EARLY_3S_u0002_s20260718.pth --output /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s_v2/screen_eval_QP3_A3_H1EARLY_3S.json --workers 8 --scenario-manifest /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/QUICK_PANEL_120.json --sim-duration 8.0
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/quick_pool_3s_v2/QP3_A0_H0_8S_seed20260718/checkpoints/end2race_ppo_QP3_A0_H0_8S_u0004_s20260718.pth --output /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s_v2/retention_eval_QP3_A0_H0_8S.json --workers 8 --scenario-manifest /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/QUICK_PANEL_120.json --sim-duration 8.0
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/quick_pool_3s_v2/QP3_A2_H1EARLY_8S_seed20260718/checkpoints/end2race_ppo_QP3_A2_H1EARLY_8S_u0004_s20260718.pth --output /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s_v2/retention_eval_QP3_A2_H1EARLY_8S.json --workers 8 --scenario-manifest /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/QUICK_PANEL_120.json --sim-duration 8.0
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/quick_pool_3s_v2/QP3_A3_H1EARLY_3S_seed20260718/checkpoints/end2race_ppo_QP3_A3_H1EARLY_3S_u0004_s20260718.pth --output /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s_v2/retention_eval_QP3_A3_H1EARLY_3S.json --workers 8 --scenario-manifest /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/QUICK_PANEL_120.json --sim-duration 8.0
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/pretrained/end2race.pth --output /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s_v2/FULL600_BC_RESULTS.json --workers 8
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/quick_pool_3s_v2/QP3_A0_H0_8S_seed20260718/checkpoints/end2race_ppo_QP3_A0_H0_8S_u0004_s20260718.pth --output /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s_v2/full600_eval_QP3_A0_H0_8S.json --workers 8
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/quick_pool_3s_v2/QP3_A2_H1EARLY_8S_seed20260718/checkpoints/end2race_ppo_QP3_A2_H1EARLY_8S_u0004_s20260718.pth --output /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s_v2/full600_eval_QP3_A2_H1EARLY_8S.json --workers 8
```

```text
/home/haowei/miniconda3/envs/end2race/bin/python -u /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s/evaluate_pool.py --model-path /home/haowei/Documents/End2Race/runs/ppo/quick_pool_3s_v2/QP3_A3_H1EARLY_3S_seed20260718/checkpoints/end2race_ppo_QP3_A3_H1EARLY_3S_u0004_s20260718.pth --output /home/haowei/Documents/End2Race/ppo_experiments/quick_pool_3s_v2/full600_eval_QP3_A3_H1EARLY_3S.json --workers 8
```
