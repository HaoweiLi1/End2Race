# PPO V1.2 IMPLEMENTATION REPORT

> Historical scope: this report verifies the original 125-arm infrastructure at the recorded commit. It does not certify the later compact 28-arm runner or manifests. Compact execution authority is `.agents/COMPACT_SWEEP_PLAN.md`; do not treat the PASS entries below as compact-support readiness.

## Files

| Path | Status |
|---|---|
| `.gitignore` | PASS |
| `configs/ppo_v1_2/sweep_manifest.json` | PASS |
| `experiments/ppo_v1_2/IMPLEMENTATION_CHECKLIST.md` | PASS |
| `experiments/ppo_v1_2/config_schema.py` | PASS |
| `experiments/ppo_v1_2/experiment_spec.py` | PASS |
| `experiments/ppo_v1_2/registry.py` | PASS |
| `experiments/ppo_v1_2/selectors.py` | PASS |
| `experiments/ppo_v1_2/result_schema.py` | PASS |
| `experiments/ppo_v1_2/hard_pool_builder.py` | PASS |
| `experiments/ppo_v1_2/aggregate.py` | PASS |
| `experiments/ppo_v1_2/runner.py` | PASS |
| `rl/sb3_end2race_policy.py` | PASS |
| `rl/end2race_gymnasium_env.py` | PASS |
| `rl/ppo_privileged.py` | PASS |
| `rl/ppo_scenarios.py` | PASS |
| `rl/ppo_callbacks.py` | PASS |
| `train_ppo_sb3.py` | PASS |
| `scripts/prepare_ppo_v1_2.py` | PASS |
| `scripts/build_ppo_v1_2_hard_pools.py` | PASS |
| `scripts/run_ppo_v1_2_sweep.py` | PASS |
| `scripts/aggregate_ppo_v1_2.py` | PASS |
| `scripts/validate_ppo_v1_2.py` | PASS |
| `scripts/audit_ppo_v1_2.py` | PASS |
| `tests/test_ppo_v1_2.py` | PASS |

## Tests

| Command / contract | Count | Status |
|---|---:|---|
| `python -m unittest tests.test_sb3_gru_integration tests.test_ppo_v1 tests.test_ppo_v1_2 -v` | 41 | PASS |
| Original `tests.test_sb3_gru_integration` + `tests.test_ppo_v1` regression | 27 | PASS |
| V1.2 critic/hard-pool/manifest/selector/lock/heartbeat/retry/drift tests | 14 | PASS |
| Austin/F110/Lattice Planner expanded preflight contract sample | 10/10 valid | PASS |
| Dry-run sweep manifest training arms | 125 | PASS |
| Stage counts `C/H/B/R/K/E/G/W/X/S` | `4/48/6/4/16/6/4/12/16/9` | PASS |
| `git diff --check` | 0 errors | PASS |

## Smoke

| Critic profile | Zero-LR updates | Nonzero updates | Finite | Frozen delta | Trainable delta | Full reload | 12-key strict load | Status |
|---|---:|---:|---|---|---|---|---|---|
| `C0_RAW_SINGLE_FRAME` | 1 | 2 | PASS | PASS | PASS | PASS | PASS | PASS |
| `C1_FROZEN_BC_FEATURE` | 1 | 2 | PASS | PASS | PASS | PASS | PASS | PASS |
| `C2_DETACHED_ACTOR_HIDDEN` | 1 | 2 | PASS | PASS | PASS | PASS | PASS | PASS |
| `C3_PRIVILEGED_PHYSICAL` | 1 | 2 | PASS | PASS | PASS | PASS | PASS | PASS |

## Hashes

| Artifact | SHA256 / canonical hash | Status |
|---|---|---|
| Baseline commit | `4fac86858802353e5b0892ff9d3c874bc15d781b` | PASS |
| `pretrained/end2race.pth` | `b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4` | PASS |
| `.agents/PPO_V1_2_EXPERIMENT_GUIDE.md` | `e8649033aad672ad2e3a1769423ee630938709627f400db7345d84c5f4095369` | PASS |
| `configs/ppo_v1_2/sweep_manifest.json` file | `80d1309a9fcd4b7a381e8b8f6f83581bf60b501c94d0a04997ec0ac7b2879219` | PASS |
| `configs/ppo_v1_2/sweep_manifest.json` canonical manifest | `b37dbd61e47a1fdd33a10ce57a8344bf838d31d937ccd819f124c7c984a54ba9` | PASS |
| C0 zero-LR smoke verification | `7f8a570adb1d90966e4dc6f34a65de2eadf52b189f09455692c7788a861ce15a` | PASS |
| C0 nonzero smoke verification | `0d46cb563ed1d02a2cb15123958026a8cb32a7a00247eec22ffd3696e6e98fbf` | PASS |
| C1 zero-LR smoke verification | `d7c9ad34055e526fc025e228d20467b864c6dd1ec6e298de9dc2c122b3ef94d3` | PASS |
| C1 nonzero smoke verification | `40473d65e620ca6ea34e8b64d2bf38cb934ca5e7c5269ab0f8adc4933c07577a` | PASS |
| C2 zero-LR smoke verification | `17a6ed424ef696d26c628da130642dabb948eced5ece80eefab2d951b6b1be67` | PASS |
| C2 nonzero smoke verification | `24f08d60e24049e8a77829add50f5bc130fdf20b59cf9a7259920150c2a65f73` | PASS |
| C3 zero-LR smoke verification | `d7c9ad34055e526fc025e228d20467b864c6dd1ec6e298de9dc2c122b3ef94d3` | PASS |
| C3 nonzero smoke verification | `6837e3c90583b927387cc5ca3cafc2e8b21fd8426b08970b368dec11cae22b4a` | PASS |
