# B6 temporal-exploration phase-0 result

Decision: **NO_GO**

This is a no-learning, training-only, matched-L4 mechanism audit. It does not
evaluate a candidate checkpoint and cannot establish PPO learnability.

| Gate | Pass |
|---|---:|
| integrity | `True` |
| collision repair | `False` |
| safe-to-collision non-inferiority | `False` |
| overtake preservation | `False` |

| Direct paired effect (AR1 - iid) | Net | Rate | L4 sign-flip p | 90% upper cluster bound |
|---|---:|---:|---:|---:|
| collision repair | `8` | `0.033333` | `0.262035` | `0.091667` |
| safe-to-collision harm | `48` | `0.100000` | `0.000000` | `0.120833` |
| lost overtake | `17` | `0.070833` | `0.000473` | `0.095833` |

The learner remains unrun. A phase-0 GO would only authorize a separate
learner proposal; a NO-GO closes this AR(1) setting without changing the
canonical actor, evaluator, or sealed data.
