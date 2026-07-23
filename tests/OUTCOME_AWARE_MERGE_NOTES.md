# Outcome-aware hard pool — review & merge notes

**Status:** isolated, verified, **not yet wired into the live pipeline.** Nothing
under `train_ppo.py`, `run.sh`, `ppo/hard_neighbors.py`,
`ppo/collision_classification.py`, or the shipped caches was modified. The
in-flight PPO runs are unaffected.

## What was added (new files only)

| File | Role |
|---|---|
| `ppo/outcome_aware_hard.py` | The module: richer BC replay classifier, configurable filter, self-verifying cache (own schema), drop-in resolver. |
| `tests/test_outcome_aware_hard.py` | 15 simulator-free tests + 1 gated real-BC smoke. |
| `tests/build_outcome_aware_cache.py` | Standalone driver: `--dry-run` inspection (subset) and full cache build. |

## What it does

Replaces the binary `collision ↔ other` boundary label with
`collision ↔ (genuinely safe overtake)`. For every boundary pair it replays the
frozen BC actor and records, for the **other** endpoint, the real terminal
outcome (`overtake` / `follow`) and min clearance; for the **collision**
endpoint, the collision-moment geometry (`post_overtake_rear` / `rear_end_opp` /
`wall` / `side_other`, plus a fishtail severity flag, bearing, Δheading, kinematic
slip, and best-effort true tire slip). A recorded, configurable filter then
selects which boundary collisions enter training.

Filter modes (`--filter-mode`): `all` (≡ boundary-aware-v1), `safe_overtake`,
`fishtail`, `fishtail_rearend`. Filter spec is part of the cache identity, so a
different mode/threshold requires a different cache dir — same discipline as the
shipped hard cache.

## Verification already done

- **Anchor (no sim):** `all` mode re-derives `boundary-aware-v1`'s
  `collision_scenarios.json` **byte-for-byte** from its own recorded outcomes.
- **Filter logic (no sim):** each mode's selection rule, clearance threshold,
  and all-vs-any source-pair policy.
- **Cache (no sim):** build → load round-trip, tamper rejection, config-mismatch
  rejection, label schema round-trip.
- **Real BC (tiny, 2 scenarios / 1 worker):** a confirmed boundary collision
  reproduces `ego_collision` under the frozen-BC contract; an `other` endpoint
  resolves to overtake/follow; geometry fields populate. Proves the classifier
  is contract-identical to the shipped one.

Run them:

```bash
python -m pytest tests/test_outcome_aware_hard.py -q          # fast, sim-free
END2RACE_RUN_SIM=1 python -m pytest tests/test_outcome_aware_hard.py -q   # + tiny real BC
python tests/build_outcome_aware_cache.py --dry-run --limit-pairs 12 --env-workers 2 --filter-mode safe_overtake
```

## Build the cache (when the pipeline is free)

Full build replays ~1183 boundary + ~914 other-endpoint + ~479 base-collision
rollouts. Run it when GPU/CPU is idle so it does not contend with training:

```bash
python tests/build_outcome_aware_cache.py \
    --filter-mode safe_overtake \
    --env-workers 12 \
    --output-cache-dir post-trained/collision-cache/outcome-aware-v1-safe
```

## Merge into the live pipeline (apply after review)

### 1. `train_ppo.py` — add CLI flags (near the other hard-neighbor args)

```python
parser.add_argument("--outcome_aware_hard", action="store_true",
    help="Use the outcome-aware filtered collision pool instead of the baseline/hard pool")
parser.add_argument("--outcome_aware_cache_dir", type=str,
    default="post-trained/collision-cache/outcome-aware-v1-safe")
parser.add_argument("--outcome_aware_filter_mode", default="all",
    choices=("all", "safe_overtake", "fishtail", "fishtail_rearend"))
parser.add_argument("--outcome_aware_safe_clearance_m", type=float, default=0.10)
parser.add_argument("--outcome_aware_require_all_pairs_safe", action="store_true", default=True)
parser.add_argument("--outcome_aware_keep_rear_end", action="store_true", default=True)
parser.add_argument("--outcome_aware_drop_unsafe_base", action="store_true", default=False)
```

### 2. `train_ppo.py` — dispatch in `main()` (replace the single resolve call)

```python
if getattr(args, "outcome_aware_hard", False):
    from ppo.outcome_aware_hard import resolve_outcome_aware_collision_scenarios
    collision_scenarios, collision_cache_info = resolve_outcome_aware_collision_scenarios(
        args, candidates, START_METHOD)
else:
    collision_scenarios, collision_cache_info = resolve_training_collision_scenarios(
        args, candidates, START_METHOD)
```

The resolver returns the same `(scenarios, info)` contract and preserves
`scenario.pool` tags (`collision` / `hard_neighbor`), so `vec_env`, the
scheduler, `--hard_neighbor_fraction`, and the reward path need **no changes**.

### 3. `validate_arguments` — guard mutual exclusion

`--outcome_aware_hard` and `--hard_neighbors` both source the collision pool;
reject enabling both at once, and require the outcome-aware cache dir to differ
from `output_dir` and `collision_cache_dir` (mirror the existing checks).

### 4. `run.sh` — example A/B group

```bash
# Group 14: outcome-aware safe-overtake pool vs the clip-0.20 U30 baseline.
# Baseline pool identity is preserved by --outcome_aware_filter_mode all.
$PYTHON train_ppo.py --critic privilege_gru --num_updates 45 --actor_epochs 2 --critic_epochs 5 \
    --batch_size 12800 --gru_learning_rate 3e-6 --head_learning_rate 3e-5 --critic_learning_rate 3e-4 \
    --steering_latent_std 0.03 --speed_physical_std 0.15 --clip_range 0.20 \
    --outcome_aware_hard --outcome_aware_filter_mode safe_overtake \
    --outcome_aware_cache_dir post-trained/collision-cache/outcome-aware-v1-safe \
    --output_dir post-trained/ppo_privilege_gru_0724_long45_clip020_oasafe
evaluate_run ppo_privilege_gru_0724_long45_clip020_oasafe 0001 0005 0010 0015 0020 0025 0030 0035 0040 0045
```

## Open design points for your call (documented, not pre-decided)

- **Default filter mode.** The proposal was "keep only safe-overtake side." The
  0722 analysis showed follow-side pairs contribute 58% of hard collisions, so
  dropping them may discard the tightest cases. `safe_overtake` implements the
  proposal; `fishtail_rearend` keeps the second failure mode's quota. Decide from
  the full-build audit, not a priori.
- **`safe` threshold** (`--outcome_aware_safe_clearance_m`, default 0.10 m):
  separates genuine passes from lucky near-misses; tune from the audit's
  clearance distribution.
- **Base-collision filtering** (`--outcome_aware_drop_unsafe_base`) defaults off;
  the 479 base BC failures are the core targets and are kept intact.
