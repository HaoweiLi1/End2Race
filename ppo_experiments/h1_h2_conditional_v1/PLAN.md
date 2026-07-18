# H1/H2 Conditional Exploration v1

This directory executes the fixed-budget experiment registered in
`.agents/End2Race_PPO_H1_H2_Conditional_Exploration_Implementation_Guide_2026-07-17.md`.

Execution barriers:

1. Record the starting source hashes and run the existing A1 update-2 checkpoint once on full-600 as a post-hoc diagnostic.
2. Implement and validate only fixed hard-ratio, paired hard-sampling, and paired telemetry support.
3. Run the four-arm H1 U1 screen, continue one full and one early winner to U2, then run two fresh seeds of the selected configuration.
4. Rebuild the matched I7/I8 H2 contrast pool and apply the no-training conditional-exploration gate.
5. If the H2 gate passes, run the registered H2 screen, retention, and conditional fresh seed.
6. Generate the final report, verdict, checkpoint index, and cleanliness audit.

No unregistered arm, seed, checkpoint, retry, reward, actor, critic, or evaluation change is permitted.
