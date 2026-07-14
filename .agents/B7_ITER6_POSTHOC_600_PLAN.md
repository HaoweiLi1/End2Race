# B7 iter6 post-hoc Austin-600 evaluation plan

Status: **ABORTED BY OWNER; NO VALID EVALUATION RESULT**, 2026-07-14

The owner revoked this evaluation while it was running. Local shard0 stopped
at 84/120 metrics and has no `COMPLETE`; remote shards1-4 had reached 480/480
before the termination command arrived. No merge, paired analysis or 600-case
summary was run. Both output roots were renamed with
`seed1_iter6_posthoc.ABORTED_BY_OWNER` and must not be resumed or counted.

## Frozen selection

The original B7 protocol ended with no iter10 candidate. The owner subsequently
authorized selection of the best available model and a 600-case evaluation.
Before any candidate outcome was evaluated, the sole model was frozen as:

```text
selection: last actor update accepted by both B7 KL gates
iteration: 6
seed: 1
training RunPlan: 3cd0f801f59609fcf6ab02a674851f49678de6b0fb04dc6a27201ff08c2672ad
actor tensor digest: c72683f96852127d07edc0a41b581ac62806a6e4abc7906f6ff42bdf9c8eee2d
actor file SHA256: 4b090011f0be5c6c3f50eec0823a2bd1f54d8bd58f005f240ac4b362d6196391
schema: canonical 12-key End2Race.state_dict()
```

Full checkpoints iter6 and iter9 contain byte-equal actor tensors because
iterations 7-9 were rolled back. No iter1/2/3 model is evaluated, and iter6 is
not selected using product outcomes.

## Evaluation

Use the unchanged B4 product evaluator contract:

```text
Austin
ego raceline1
opponent raceline0/raceline1/raceline2
opponent speedscale 0.5/0.6/0.7/0.8
50 deterministic startpoints
3 x 4 x 50 = 600 episodes
8 second horizon
```

Reuse the immutable B4 canonical-BC rows (`24 collision / 342 overtake`) and
run only 600 new iter6 episodes. Shard0 runs locally; shards1-4 run remotely.
All inference is deterministic and the original strict plain-End2Race loader is
used.

Report collision/overtake/follow, fixed/new collision, gained/lost overtake,
speed projection, occurrence McNemar and 50-startpoint cluster sign-flip.

This is opened-development, post-hoc diagnostic evidence. It cannot rewrite the
historical B7 `EARLY_STOP_NO_CANDIDATE` verdict or authorize seed0/sealed data.
After the owner cancellation above, the plan is historical only and no valid
diagnostic result exists.
