# B6 temporal-exploration phase-0 replacement RunPlan

This replacement was frozen after the first attempt stopped on an AR-state
replay correctness failure and before any valid stochastic outcome was
generated.

```text
corrected implementation: d71efe948d8b6d9523535840e1364e5608481051
selection digest:         7224f1f3da6a35febc50392cc35b4844076c77094f508d78dbe7b9b3fafb93fd
selection file SHA256:    7612d66ecd4708fb20905266c88033e5ffe6dd903aae4cdca310d7d1ae7c2b44
RunPlan SHA256:           b3725809c65b5ac66aae4bfb853accc87c95af35fb4cf53d5d19039f09a679d5
```

The 180 selected training scenarios are byte-identical to the first plan.
The replacement changes only computational correctness:

- AR state is reconstructed from the stored raw latent displacement;
- batched replay uses the existing B4 `max abs(ratio-1) <= 1e-4` contract.

No old episode is resumable into this run. Results must be written to a new
output directory and appended here only after all 1,440 episodes complete.
