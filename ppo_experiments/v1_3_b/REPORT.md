# End2Race PPO V1.3-B Results

**Final verdict:** `FAIL_KL_UNSTABLE`

V1.3-B changes only PPO rollout reuse from one to four epochs at the original actor learning rates. All product decisions use the fixed U8 checkpoint for every seed.

## Frozen configuration

```text
n_envs=16, n_steps=1600, batch_size=1600, updates=8
n_epochs=4, target_kl=0.01, update_kl_guardrail=0.02
GRU LR=1e-6, head LR=1e-5, critic=C0_RAW_SINGLE_FRAME
H0 probability=0.50 with replacement, reward and exploration unchanged
```

## Training stop

Formal evaluation was not started because at least one seed failed the frozen training-stability gate.

| Seed | Update | Approx KL | Clip fraction | Actual/planned steps | Target-KL early stop |
|---:|---:|---:|---:|---:|:---:|
| 20260723 | 1 | 0.001673 | 0.076807 | 64/64 | N |
| 20260723 | 2 | 0.013600 | 0.059975 | 24/64 | Y |
| 20260723 | 3 | 0.001280 | 0.058027 | 64/64 | N |
| 20260723 | 4 | 0.001683 | 0.057744 | 64/64 | N |
| 20260723 | 5 | 0.119444 | 0.030603 | 28/64 | Y |

Seed `20260723` stopped after U5: `approx_kl 0.119443722 exceeds 0.02`.

Not started under fail-fast policy: `[20260724, 20260725, 20260726, 20260727]`.

Only guardrail-passing U2 and U4 checkpoints exist; no U5 checkpoint was saved.

## Interpretation boundary

At least one formal seed crossed the preregistered post-update KL guardrail. The four-epoch update window is therefore not a controlled setting.

The canonical development panel is used only for this preregistered mechanism test. No checkpoint is promoted to `posttrained/`, and no deployment claim is made.
