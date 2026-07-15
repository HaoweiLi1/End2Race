# Proposed Next Direction — Minimal BC-Direct PPO Pilot

Status: proposed design, not yet executed or approved as a numerical run.

## Question to answer

Can a PPO residual policy initialized to BC-identical behavior learn to reduce
collisions without lowering corrected overtake rate?

This is the product question. A supervised warm-start is no longer treated as
a scientific prerequisite.

## Minimal policy

The BC model remains the driving policy. A small residual policy can choose
NO_OP or a bounded steering/braking correction. Fresh deterministic behavior
must remain exactly BC.

`INITIAL_INTERVENTION_LOGIT = -6.0` remains unchanged. Exploration is added
only to the stochastic training behavior distribution; deterministic
evaluation disables it. The PPO log-probability must describe the exact
sampled and executed distribution, including any exploration mixture or
logit offset.

## Minimum preflight

Only three blocking mechanics checks are justified before the pilot:

1. fresh deterministic BC identity;
2. bound-preserving action composition with no hidden simulator clipping;
3. sampled action, executed action, stored log-probability, and PPO ratio are
   consistent.

TTC, warning scores, warm-start classification, and witness imitation may be
logged as diagnostics but cannot block or promote PPO.

## Pilot outline

- Start from the canonical BC checkpoint.
- Keep the BC backbone frozen for the short mechanism pilot.
- Train only the residual policy/risk sidecar and separate critics.
- Use a declared training-only intervention exploration schedule. Do not
  assume that changing a deterministic bias alone creates exploration.
- Keep residual bounds and zero positive-speed budget for the first pilot.
- Clamp the overtake dual and delay updates until enough completed episodes
  exist.
- Evaluate deterministic snapshots on paired development scenarios.
- Promotion order is lexicographic: overtake feasibility first, collision
  reduction second.

The short pilot is a mechanism test, not generalization evidence. Fresh-pool
evaluation remains a later one-open stage.

## Required logging

- intervention attempts and realized action types;
- raw/sampled/executed actions and log-probabilities;
- clipping/projection deltas, which must be zero after canonical composition;
- any-agent and ego collision;
- corrected terminal overtake and confirmed safe pass;
- paired transitions: collision→safe pass, collision→follow,
  overtake→follow, and safe→new collision;
- dual value, collision/overtake advantages, KL, entropy, and gradient norms;
- per-seed and pooled checkpoint decision with no hidden arm ranking.

## Stop conditions

- stop immediately for action/log-probability inconsistency, hidden clipping,
  missing episodes, or a registry/final-pool violation;
- reject any candidate that violates the declared overtake tolerance;
- if sufficient intervention exploration occurs but no risk-conditioned
  behavior is learned, that is direct evidence that additional initialization
  or a different credit-assignment design may be needed;
- do not return to an unlimited sequence of supervised probe gates.

