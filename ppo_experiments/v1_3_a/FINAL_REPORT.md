# End2Race PPO V1.3-A Final Report

**Final verdict:** `FAIL_KL_UNSTABLE`

V1.3-A keeps one PPO epoch and changes only the actor learning rates to 3x nominal. The only product checkpoint is U8 for every seed.

## Frozen configuration

```text
n_envs=16, n_steps=1600, batch_size=1600, n_epochs=1, updates=8
GRU LR=3e-6, head LR=3e-5, critic LR=3e-4
target_kl=0.01, post-update guardrail=0.02
C0 critic, H0 p0.50 with replacement, reward/exploration unchanged
```

## Training process

| Seed | Status | Last U | KL sequence | Steps sequence | In [0.002,0.010] | Process pass |
|---:|---|---:|---|---|---:|:---:|
| 20260718 | STOPPED_KL_GUARDRAIL | 5 | 0.0078,0.0067,0.0026,0.0018,0.0967 | 1,16,16,16,7 | 3 | N |

## Fail-fast stop

Not started: `[20260719, 20260720, 20260721, 20260722]`. No candidate or BC evaluation was run.

## Conclusion

A formal update exceeded the locked KL guardrail; fixed 3x actor LR is not stable.

`selection_performed=false`, `holdout_performed=false`, and `promotion_performed=false`. Canonical BC remains the deployment recommendation.
