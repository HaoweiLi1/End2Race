# B7 plain recurrent PPO compact result evidence

This directory contains the compact, Git-reviewable evidence for the
authoritative B7 seed1 run. The exact source is
`3e262e2bf00acd8ef9338122a82780e68a825981`; the immutable RunPlan digest is
`3cd0f801f59609fcf6ab02a674851f49678de6b0fb04dc6a27201ff08c2672ad`.

The run completed nine iterations and stopped after three consecutive safe-KL
rejections. It produced no iteration-10 candidate, so evaluation was correctly
unrun. Read `.agents/B7_PLAIN_RECURRENT_PPO_RESULT.md` for the decision and
bounded interpretation.

The full 1.5 GiB release remains at:

```text
Experiments/B7_plain_recurrent_ppo/runs/b7_seed1_20260714_114132/remote/seed1
```

and on the remote host under the matching immutable staging root. The compact
copy preserves the atomic summary, full nine-row iteration ledger, training log
and production smoke without committing large replay/checkpoint binaries.
