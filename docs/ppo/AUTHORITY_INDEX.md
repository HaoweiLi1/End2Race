# Handoff and Authority Index

Updated: 2026-07-12. This index consolidates every handoff found in the local
workspace and makes its authority boundary explicit.

## Current authority

`.agents/HANDOFF.md` is the only current handoff. Newer numbered sections
supersede older sections. Its newest §19 records the local Tier-3 organization
and the proposed BC-direct PPO direction; it does not authorize a numerical
PPO run.

Current execution facts:

- the primary assistant controls work;
- unattended remote Codex authority is revoked;
- active remote address remains `haowei@192.168.2.127`, but the Tier-3
  organization did not access or modify it;
- the D2 test/final pools remain unopened;
- hierarchical Task 6 is the latest numerical result and is FAILED;
- replacement Task 9/10 and PPO have not run.

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
3. `.agents/HANDOFF.md` newest section, then only the historical sections
   needed for provenance.
4. `docs/ppo/NEXT_PPO_DIRECTION.md` for the proposed, not-yet-run experiment.
5. Original canonical reports when exact numbers or hashes are required.

No historical handoff may be used to resume unattended execution or to
override a newer failed gate.

