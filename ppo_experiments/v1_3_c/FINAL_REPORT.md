# End2Race PPO V1.3-C Final Report

**Final verdict:** `INVALID_INFRASTRUCTURE`

V1.3-C changes only V1.3-A's steering likelihood from the singular atanh-squashed parameterization to a physical Gaussian whose action is clipped by the existing environment.

## Training process

| Seed | Status | Last U | KL sequence | Steps | In window | Process pass |
|---:|---|---:|---|---|---:|:---:|
| 20260729 | RUNNING | 6 | 0.0067,0.0020,0.0025,0.0024,0.0026,0.0029 | 16,16,16,16,16,16 | 5 | N |

## Fail-fast stop

Not started: `[20260730, 20260731, 20260732, 20260733]`. No BC or candidate evaluation was run.

## Conclusion

The first formal seed was interrupted after U6 by a host-level SIGSEGV in PyTorch pt_autograd/libc10. Later seeds and all evaluations were not started. This is an infrastructure-invalid result, not a PPO performance failure.

`selection_performed=false`, `holdout_performed=false`, and `promotion_performed=false`. Canonical BC remains the deployment recommendation unless a later preregistered confirmation says otherwise.
