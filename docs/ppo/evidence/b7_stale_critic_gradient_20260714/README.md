# B7 first staged-attempt remediation evidence

The first authoritative B7 staging attempt used source
`1c096c235e8c1a664ee2782cd8dd2f41c6988b11` and RunPlan digest
`360063d4289d2db18f0bd74a49a6ed967527635cd30feab2aefa9736a2a109b1`.
The production smoke passed. Iteration 1 completed and was accepted, then the
iteration-2 actor-isolation check failed closed with:

```text
AssertionError: B7 critic received an actor gradient
```

The actor objective did not actually backpropagate through the critic. The
previous iteration's critic backward left non-`None` gradients populated, and
the next actor-isolation assertion observed that stale state. No iteration-10
candidate or evaluation existed. The incomplete run is not resumable as an
authoritative result.

The remediation clears the critic optimizer gradients before every actor
backward. A new consecutive-iteration regression first leaves real critic
gradients populated after iteration 1, then verifies that iteration 2 completes
one actor step and all three critic epochs without weakening the isolation
assertion. All B7 and nine legacy compatibility programs were rerun.

Only the compact immutable evidence needed to audit the failure is retained
here; the 274 MiB incomplete remote replay/checkpoint directory is excluded.
