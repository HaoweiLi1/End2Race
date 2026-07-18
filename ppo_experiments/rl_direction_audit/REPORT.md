# End2Race PPO RL-only mechanism audit report

- Source HEAD: `5bfccef9f8053a0d857cb0728c146a9fe3b4dc15`
- Canonical BC SHA-256: `b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4`
- Device: `cuda`
- Final status: `MECHANISM_AUDIT_COMPLETE_NO_PRODUCT_CLAIM`
- Product claim: **none**; this audit did not authorize or perform a held-out product test.

## Answers to the five mechanism questions

### 1. Is there a stable actor-gradient direction?

`POOL_DEPENDENT_YES_ONLY_FOR_H1; H0_AND_H2_REMAIN_INCONCLUSIVE; H3_IS_CONFLICTING`.

| Pool | Episodes | Final verdict | Median pairwise cosine | Collision action-sign agreement |
|---|---:|---|---:|---:|
| H0_CURRENT_DET | 256 | INCONCLUSIVE | -0.073576 | 0.5685 |
| H1_EXPANDED_DET | 128 | DIRECTION_PRESENT | 0.862685 | 0.7535 |
| H2_STOCH_CORE | 256 | INCONCLUSIVE | 0.029196 | 0.6818 |
| H3_UNION_CORE | 128 | DIRECTION_ABSENT_OR_CONFLICTING | -0.091086 | 0.5407 |

H1 is the only pool that passes all preregistered direction gates. H0 and H2 still do not pass after the single allowed 256-episode extension; H3 is explicitly conflicting.

### 2. How many independent episodes are needed?

`128_IS_THE_FIRST_PREREGISTERED_TESTED_COUNT_THAT_PASSES_FOR_H1; NO_UNIVERSAL_COUNT_EXISTS_BECAUSE_H0_AND_H2_FAIL_TO_STABILIZE_AT_256`.

The evidence supports 128 as the first tested passing count for H1, not as a universal minimum and not as proof that fewer episodes would fail. Transition count is not treated as independent sample count.

### 3. Is the 2-3 second credit window being lost?

`NOT_SUPPORTED_BY_THE_PREREGISTERED_OFFLINE_IMPROVEMENT_GATE`.

P2 earliest-actionable window: `{'median': 2.5, 'p25': 1.375, 'p75': 3.0}`. P3 chose `C0_CURRENT` on `H1_EXPANDED_DET` with verdict `KEEP_CURRENT_CREDIT`.

### 4. Does reward rank safe alternatives correctly?

`YES_FOR_THE_PREREGISTERED_LOCAL_REPAIRS; EXPLORATION_COVERAGE_IS_INSUFFICIENT`.

P2 verdict: `EXPLORATION_COVERAGE_INSUFFICIENT`. Reproduced collisions: 22; repairable: 20 (0.9091); best-safe return above no-op: 1.0000.

P2 aggregation revision: `{'raw_branches_reused': True, 'reason': "Completion audit found that revision 1 collapsed 0.25 s and 0.50 s pulses into one family, although duration is part of the guide's pulse specification. Revision 2 uses offset plus duration as the template; raw branches and all thresholds are unchanged.", 'revision': 2, 'superseded_result_sha256': '34cf58a2a63a7500a26f9a7b6b490873b74e295768e431ac514dcf19a03311ed', 'superseded_verdict': 'LOCAL_ACTION_NOT_FOUND'}`. The corrected result reuses the identical raw branch SHA and unchanged thresholds; duration is part of a pulse template.

### 5. Does sequential minibatch update destroy a useful direction?

`CONTROLLED_STEP_INSUFFICIENT`.

| Method | Seed | Mean exact KL | p99 sequence KL | Fixed collision | New collision | SAFE new collision | Overtake lost |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1_SEQUENTIAL_MINIBATCH | 20260761 | 0.002501 | 0.003010 | 4 | 4 | 0 | 2 |
| S2_FULL_ROLLOUT_ONE_STEP | 20260761 | 0.001644 | 0.003163 | 4 | 5 | 2 | 5 |
| S3_TRANSACTIONAL_BACKTRACKED_ONE_STEP | 20260761 | 0.001644 | 0.003163 | 4 | 5 | 2 | 5 |
| S1_SEQUENTIAL_MINIBATCH | 20260762 | 0.005479 | 0.045672 | 3 | 4 | 1 | 2 |
| S2_FULL_ROLLOUT_ONE_STEP | 20260762 | 0.004386 | 0.010215 | 6 | 5 | 3 | 5 |
| S3_TRANSACTIONAL_BACKTRACKED_ONE_STEP | 20260762 | 0.004386 | 0.010215 | 6 | 5 | 3 | 5 |
| S1_SEQUENTIAL_MINIBATCH | 20260763 | 0.001498 | 0.013525 | 10 | 3 | 0 | 3 |
| S2_FULL_ROLLOUT_ONE_STEP | 20260763 | 0.001514 | 0.002224 | 5 | 5 | 3 | 5 |
| S3_TRANSACTIONAL_BACKTRACKED_ONE_STEP | 20260763 | 0.001514 | 0.002224 | 5 | 5 | 3 | 5 |

Neither S2 nor S3 produced a single fully passing seed: smaller KL did not prevent SAFE collisions or overtake losses, so sequential minibatch geometry is not confirmed as the primary failure source.

## Decision and next allowed action

- Action: `PREREGISTER_A_NARROW_SUSTAINED_ACTION_EXPLORATION_INTERVENTION_THEN_REPEAT_P1_AND_P4`
- Reason: Reward direction passed and all best repairs were below 3 sigma per step, but every 0.25 s repair had iid sequence probability below 1%; more samples of unchanged iid noise do not directly address that temporal coverage failure.
- P5: `NOT_TRIGGERED`
- P6: `NOT_AUTHORIZED`

No long PPO training, demonstration mixing, architecture change, reward sweep, or product checkpoint selection was performed.

## Final verification

- Repository unittest discovery: 13/13 passed in conda env `end2race` (`pytest` is not installed in that environment).
- All nine audit diagnostic scripts compiled successfully.
- Frozen contract: 9/9 file hashes matched; canonical BC strict schema: 12 keys.
- Frozen product surfaces `ppo/`, `model.py`, `train_ppo.py`, and `pretrained/` have no diff from source HEAD.

## Reproducibility artifacts

- preregistration: `ppo_experiments/rl_direction_audit/AUDIT_PREREGISTRATION.json` (`ed6dad39fe420238dabbf3fef22ea590e25abe173213685896e8863fa0887fe2`)
- safe_reference: `ppo_experiments/rl_direction_audit/SAFE_REFERENCE.json` (`731d039f10cab2cb9d55d6050b0836cadb74b84f27b2d0bc12689d74778af40b`)
- p1: `ppo_experiments/rl_direction_audit/P1_GRADIENT_DIRECTION.json` (`4b54e0c4184d8f934c975f55355963ee87070ccce5508a7e2594f68da2391655`)
- p1_extension: `ppo_experiments/rl_direction_audit/P1_GRADIENT_DIRECTION_EXTENSION.json` (`48071e1cb45a1ee9cbd2fce7c7a197c7c44faa5da9bb83b3e93811c68065bc2d`)
- p2: `ppo_experiments/rl_direction_audit/P2_COUNTERFACTUAL_ACTIONABILITY.json` (`0335748dc8b128d77b722952771e9fdb65c7952fbcdcba2162b2a68a990623e7`)
- p3: `ppo_experiments/rl_direction_audit/P3_CREDIT_HORIZON.json` (`b586e495f543e2cbdfb3f191d8de54a02676fbfaa5885d87f584cc114e627e56`)
- p4_panel: `ppo_experiments/rl_direction_audit/P4_PANEL.json` (`6c573190850b706201e24756cd4bf521f871ac4dbe5aeb68b43c633fe2f27101`)
- p4: `ppo_experiments/rl_direction_audit/P4_CONTROLLED_STEP.json` (`5d8b2c49a54c4befa2d9b24f095642bab6fabecee31160aeb911a04a35734390`)
- Raw audit root: `runs/ppo/RL_DIRECTION_AUDIT_20260717`
