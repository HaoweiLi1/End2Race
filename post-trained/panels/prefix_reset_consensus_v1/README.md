# Prefix-reset consensus panel v1

Austin-only training input for the preregistered prefix-reset curriculum.

- 28 U42--U45 at-least-3-of-4 consensus tasks: 19 collision and 9 lost-overtake.
- 21 unique ego startpoints and 9,589 total pre-window observations.
- `snapshots/` stores the mechanically exact F110/planner/reward state.
- `prefixes/` stores only `prefix_observations` and `window_observation`; saved U44/U45 hidden states are deliberately excluded because every reset must burn in the current in-memory network.
- `prefix_reset_manifest.json` freezes task identity, source stratum, prefix length and content digests.

This panel is a persistent training input, not an evaluation result. Do not delete it with cleanable analysis products.
