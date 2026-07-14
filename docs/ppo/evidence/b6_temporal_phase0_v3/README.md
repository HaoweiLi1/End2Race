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
identity and bounded batched numerical replay.

## Completed result

All 1,440 episodes completed on the remote RTX 4080 SUPER. Local independent
summarization of the copied atomic rows reproduced the remote summary and
paired table byte for byte.

Decision: **NO-GO; learner unrun**.

```text
collision repair AR1-iid:       +8/240  (+3.33 pp), L4 p=0.2620
safe-to-collision harm AR1-iid: +48/480 (+10.00 pp), L4 p=4.19e-9
lost overtake AR1-iid:          +17/240 (+7.08 pp), L4 p=0.000473
```

Files:

- `episode_results.jsonl`: all 1,440 atomic episode ledgers, compacted without
  changing any field;
- `paired_results.tsv`: 720 complete iid/AR(1) pairs;
- `summary.json` and `report.md`: frozen gates and final decision;
- `remote_run.log`: execution progress and runtime warnings;
- `COMPLETE`: digest of `summary.json`.

The result packet is interpreted in
`.agents/B6_TEMPORAL_EXPLORATION_PHASE0_RESULT.md`.
