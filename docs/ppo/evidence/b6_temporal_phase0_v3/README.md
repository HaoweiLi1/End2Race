# B6 temporal-exploration phase-0 final replacement RunPlan

This is the active prospective RunPlan. It was frozen before any valid B6
outcome was generated.

```text
implementation:        5a4c48f2debb8f4dd58807c966d47635408698d9
selection digest:      7224f1f3da6a35febc50392cc35b4844076c77094f508d78dbe7b9b3fafb93fd
selection file SHA256: 7612d66ecd4708fb20905266c88033e5ffe6dd903aae4cdca310d7d1ae7c2b44
RunPlan SHA256:        4a3923dbe2cf87073aa0aadb0bc59d8d8222882c107cb3d31c5e50f275dbbe7f
```

The scenario selection, innovations, rho and direct-outcome gates are
byte-for-byte/scientifically unchanged from the first plan. Two incomplete
attempts are preserved separately and cannot be resumed:

- v1: private pre-addition AR state was not exactly reconstructible;
- v2: one B4 batch-GEMM tolerance was incorrectly reused despite AR(1)'s
  smaller conditional standard deviation.

The active integrity contract separately checks exact framewise probability
identity and bounded batched numerical replay. This run starts from an empty
output directory and must complete all 1,440 episodes.
