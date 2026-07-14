# Handoff and Authority Index

Updated: 2026-07-14. This index consolidates every handoff found in the local
workspace and makes its authority boundary explicit.

## Current authority

`.agents/HANDOFF.md` is the only current handoff. Newer numbered sections
supersede older sections. Its newest §35 records the completed B7 early stop,
absence of a candidate and exhaustion of that run's execution authority.

Current execution facts:

- the B7-specific remote authority was consumed by the completed seed1 run;
- the valid B7 source is `3e262e2bf00acd8ef9338122a82780e68a825981` and
  RunPlan digest is
  `3cd0f801f59609fcf6ab02a674851f49678de6b0fb04dc6a27201ff08c2672ad`;
- B7 stopped at iteration 9 after three consecutive actor-update rejections;
  no candidate, evaluation or seed0 exists;
- Austin 600 was not reopened for B7, and the D2 test/final pools remain sealed;
- the next legal experiment requires a new prospective owner decision.

## Historical handoffs

Historical handoffs are preserved under `docs/archive/handoffs/`. They are
provenance, not live instructions.

| file | historical scope | current status |
|---|---|---|
| `HANDOFF.md` | early PPO continuation state | superseded by `.agents/HANDOFF.md` |
| `ppo_handoff_20260705.md` | early PPO failure/sweep status | summarized by P1/D0.1 reports |
| `d1b_local_failure_handoff_20260705.md` | one local D1 failure handoff | legacy diagnostic only |
| `ppo_audit_handoff_20260710.md` | P0/P1 audit and B+ transition | important provenance; §§13–19 are newer authority |
| `REMOTE_CONTINUATION_20260710.md` | remote goal continuation instructions | revoked by current owner authority |

The historical remote goal files inside
`logs/ppo_next_unattended_20260710_230212/` remain in place because they are
part of the canonical D0.1 run provenance. `GOAL_OBJECTIVE.md` is not live
authority.

## Reading order for a new agent

1. `docs/ppo/README.md`.
2. `docs/ppo/EXPERIMENT_HISTORY.md`.
3. `.agents/B7_PLAIN_RECURRENT_PPO_RESULT.md` for the latest execution result.
4. `.agents/HANDOFF.md` newest section, then only the historical sections
   needed for provenance.
5. Original canonical reports when exact numbers or hashes are required.

No historical handoff may be used to resume unattended execution or to
override a newer failed gate.
