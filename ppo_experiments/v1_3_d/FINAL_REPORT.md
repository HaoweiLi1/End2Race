# End2Race PPO V1.3-D Final Report

**Final verdict:** `STOP_PROTOCOL_DRIFT`

V1.3-D is the infrastructure-stable replication of V1.3-C: the same physical-Gaussian PPO configuration with PyTorch autograd multithreading disabled.

## Training process

| Seed | Status | Last U | KL sequence | Steps | In window | Process pass |
|---:|---|---:|---|---|---:|:---:|
| 20260735 | COMPLETED | 8 | 0.0076,0.0071,0.0065,0.0028,0.0028,0.0024,0.0026,0.0025 | 1,16,16,16,16,16,16,16 | 8 | Y |
| 20260736 | COMPLETED | 8 | 0.0088,0.0158,0.0049,0.0029,0.0028,0.0022,0.0013,0.0024 | 1,2,16,16,16,16,16,16 | 6 | Y |
| 20260737 | COMPLETED | 8 | 0.0084,0.0069,0.0026,0.0013,0.0034,0.0022,0.0020,0.0024 | 1,13,16,16,16,16,16,16 | 6 | Y |

## BC protocol check

Expected BC: `21 collision / 233 follow / 346 overtake`.

Observed BC: `22 collision / 234 follow / 344 overtake`.

Evidence: `eval_results/end2race_bc_v1_3_d_Austin/multiagents/results_multi.json` (`98c6aa9c5fcfa35c87e638c18ca9e67a37b6741807722db75314007bc84302de`); log: `runs/ppo/v1_3_d_logs/eval_bc.log` (`65483fb5b7131cf6ee8ba9f7aab3eef2ac87a2526b5eaecbea91b1f96684d68f`).

The exact-match gate failed, so no PPO candidate evaluation was run.

## Conclusion

Canonical BC exact-match gate failed: expected {'collision': 21, 'follow': 233, 'overtake': 346}, observed {'collision': 22, 'follow': 234, 'overtake': 344}. Candidate evaluations were not run.

`selection_performed=false`, `holdout_performed=false`, and `promotion_performed=false`. Canonical BC remains the deployment recommendation unless a later preregistered confirmation says otherwise.
