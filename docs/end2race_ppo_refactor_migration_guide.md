# End2Race PPO Refactor and Migration Guide

**Audience:** Codex, Claude Code, or another coding agent working on the End2Race PPO training pipeline.

**Goal:** Refactor the current PPO implementation into a cleaner, behavior-preserving two-file PPO layout while reusing suitable repo-wide helpers from `utils.py`.

**Current relevant files:**

- `env_ppo.py`
- `train_ppo.py`
- `utils.py`
- `model.py`
- `demonstration.py`
- `eval_singleagent.py`
- `eval_multiagent.py`

**Target PPO files:**

```text
train_ppo.py
ppo_utils.py
```

The existing `utils.py` should remain the repo-wide shared helper file. Only generic, non-PPO-specific utilities should be added to `utils.py`. PPO-specific reward, curriculum, checkpoint, tensor replay, and training diagnostics should go into the new `ppo_utils.py`.

---

## 1. Refactor principle

This is a **behavior-preserving refactor**.

Do not change:

```text
1. PPO actor observation:
   LiDAR 360 + previous ego speed only.

2. PPO action/log-prob consistency:
   Store raw sampled Gaussian actions in the rollout buffer.
   Execute clipped actions in the simulator.

3. Reward values:
   `compute_shaped_reward()` must produce the same scalar reward and same `reward_terms` before and after refactor.

4. Boundary semantics:
   collision/env true done -> terminated=True
   timeout/sim_duration -> truncated=True
   timeout truncation must bootstrap V(s_next)

5. Actor-only checkpoint compatibility:
   saving `ac.actor.state_dict()` must still produce a checkpoint loadable by original `End2Race` evaluators.

6. Current PPO v1 architecture:
   no privileged opponent pose or hazard enters the actor.
```

This task is about moving code, clarifying ownership, and removing duplicated logic, not redesigning PPO.

---

## 2. Final file responsibilities

### 2.1 `utils.py`: repo-wide general utilities

`utils.py` should contain helpers that are broadly useful outside PPO.

Allowed categories:

```text
- raceline CSV loading
- waypoint/index matching
- two-agent segment setup
- generic LiDAR downsampling
- reference-line geometry
- closed-track progress wrapping
- generic two-agent reference geometry, if named generally
- evaluation/rendering utilities already present
```

`utils.py` must not import PPO classes, `train_ppo.py`, or `ppo_utils.py`.

Avoid adding:

```text
- RewardWeights
- RewardState
- compute_shaped_reward
- PPO checkpoint helpers
- torch recurrent replay helpers
- reward_weight_names
- apply_reward_overrides
- PPO logging keys
- sample_scenario with PPO curriculum stages
```

### 2.2 `ppo_utils.py`: PPO-specific helper file

Create a new file:

```text
ppo_utils.py
```

It should contain PPO-specific helper code that is not the main training loop.

Allowed categories:

```text
- PPO constants and metric keys
- PPO reward weights/state/reward function
- PPO curriculum scenario sampler
- checkpoint load/save helpers
- torch tensor conversion helpers
- recurrent full-sequence replay helpers
- replay identity validator
- value bootstrap helper
- reward override helper
- fixed scenario helper
- iteration summary formatter
```

`ppo_utils.py` may import:

```python
import os
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from model import End2Race, End2Race_PPO
from utils import (
    STEER_LIMIT,
    LIDAR_DIM,
    ACTION_DIM,
    LIDAR_MAX_RANGE,
    downsample_lidar_for_model,
    load_raceline_with_speed,
    load_positions_and_speeds_from_params,
    resolve_two_agent_indices,
    wrap_rel_s,
    unwrap_progress,
    project_to_reference,
    load_reference_line,
    compute_two_agent_reference_geometry,  # optional name, see below
)
```

Only import names that are actually used. Do not use `from utils import *`.

### 2.3 `train_ppo.py`: PPO main file

`train_ppo.py` should contain the core PPO pipeline:

```text
- End2RacePPOEnv
- RolloutBuffer
- collect_rollout()
- ppo_update()
- parse_arguments()
- main()
```

Optionally, if desired:

```text
- PPOTrainer class
```

The goal is that reading `train_ppo.py` gives the main training flow without burying the reader in geometry, reward, checkpoint, or recurrent replay helper details.

### 2.4 `env_ppo.py`

After migration, either delete `env_ppo.py` or leave it as a temporary legacy wrapper that is no longer used by `train_ppo.py`.

Preferred final state:

```text
No active import from env_ppo.py in train_ppo.py.
```

If deleting is too risky, keep `env_ppo.py` temporarily but make it clear it is deprecated. Do not maintain two divergent copies of `End2RacePPOEnv`.

---

## 3. Functions/classes to move into `utils.py`

Move only functions that are not clearly PPO-specific.

### 3.1 Constants

Add to `utils.py` near the top or in a constants section:

```python
STEER_LIMIT = 0.52
LIDAR_DIM = 360
ACTION_DIM = 2
LIDAR_MAX_RANGE = 30.0
```

Rationale:

```text
STEER_LIMIT and LiDAR dimension are End2Race/F1Tenth constants used across evaluation, PPO, and data collection. They are not PPO-only.
```

If `ACTION_DIM` feels too PPO/action-buffer-specific, it may remain in `ppo_utils.py`. `STEER_LIMIT`, `LIDAR_DIM`, and `LIDAR_MAX_RANGE` are more clearly shared.

### 3.2 Generic LiDAR downsampling

Move from `env_ppo.py` and rename:

Current:

```python
def downsample_for_eval_compat(lidar, target_points=LIDAR_DIM):
    ...
```

Recommended in `utils.py`:

```python
def downsample_lidar_for_model(lidar, target_points=LIDAR_DIM, max_range=LIDAR_MAX_RANGE):
    """Convert simulator LiDAR to the fixed-size End2Race model input."""
    lidar = np.asarray(lidar, dtype=np.float32).reshape(-1)
    if len(lidar) != target_points:
        lidar = lidar[np.linspace(0, len(lidar) - 1, target_points, dtype=int)]
    lidar = np.nan_to_num(lidar, nan=0.0, posinf=max_range, neginf=0.0)
    return np.clip(lidar, 0.0, max_range).astype(np.float32)
```

Update call sites:

```python
# train_ppo.py / End2RacePPOEnv._policy_obs()
return {
    "lidar": downsample_lidar_for_model(obs["scans"][0]),
    "prev_speed": np.array([self._prev_speed], dtype=np.float32),
}
```

Optional follow-up, not required in this refactor:

```text
Replace duplicated LiDAR downsampling logic in eval_singleagent.py and eval_multiagent.py with this helper.
```

Do not change evaluation behavior in this pass unless tests are updated.

### 3.3 Reference-line container

Move from `env_ppo.py` to `utils.py`.

Current:

```python
class ReferenceLine:
    def __init__(self, s, xy, track_length):
        self.s = s
        self.xy = xy
        self.track_length = track_length
```

Recommended:

```python
from dataclasses import dataclass

@dataclass
class ReferenceLine:
    """Closed-track reference line for progress and Frenet-like geometry."""
    s: np.ndarray
    xy: np.ndarray
    track_length: float
```

Do not change field names.

### 3.4 `load_reference_line()`

Move from `env_ppo.py` to `utils.py` unchanged except imports and class definition.

Keep all validation:

```text
- `s` must be strictly increasing.
- first/last xy distance must be no more than 2x the median segment length.
```

Rationale:

```text
This is generic track reference geometry. It is not PPO reward-specific.
```

### 3.5 `wrap_rel_s()`

Move to `utils.py`.

Keep signature and behavior:

```python
def wrap_rel_s(delta_s, track_length):
    """Wrap relative progress to [-L/2, L/2] on the closed track."""
    return float((float(delta_s) + 0.5 * track_length) % track_length - 0.5 * track_length)
```

Do not rename during this pass unless all call sites are updated carefully.

### 3.6 `unwrap_progress()`

Move to `utils.py`.

Keep signature and behavior:

```python
def unwrap_progress(p_raw, p_last, track_length):
    """Choose the lap-unwrapped progress nearest to the previous progress."""
    k0 = int(np.floor((float(p_last) - float(p_raw)) / track_length))
    candidates = [float(p_raw) + (k0 + k) * track_length for k in (-1, 0, 1, 2)]
    p = min(candidates, key=lambda value: abs(value - float(p_last)))
    return float(p), float(p - float(p_last))
```

### 3.7 `project_to_reference()`

Move to `utils.py`.

Keep signature and return values:

```python
def project_to_reference(point, ref):
    """Project point to reference line and return (s, signed d, tangent theta)."""
    ...
    return float(s), d, theta
```

Rationale:

```text
This is a general Frenet-like track geometry helper. It is currently used by PPO reward, but it can also support evaluation metrics and validators.
```

Do not replace this with `latticeplanner.utils.project_point_to_centerline()` because the current function returns signed lateral offset and tangent heading, which PPO reward needs.

### 3.8 `resolve_two_agent_indices()`

Move to `utils.py`, but simplify internals to reuse existing utilities.

Current `utils.py` already has:

```text
- load_raceline_with_speed()
- find_corresponding_waypoint()
- load_positions_and_speeds_from_params()
```

Recommended implementation:

```python
def resolve_two_agent_indices(map_name, ego_raceline, opp_raceline, ego_idx, interval_idx, opp_idx=None):
    """Resolve ego/opponent waypoint indices for a two-agent segment.

    If `opp_idx` is provided, it is respected modulo the opponent waypoint count.
    Otherwise, opponent is placed `interval_idx` waypoints ahead of ego. For
    different racelines, ego is first mapped to the closest opponent-raceline waypoint.
    """
    _, _, ego_wp = load_raceline_with_speed(map_name, f"{ego_raceline}.csv", 0)
    if opp_raceline == ego_raceline:
        opp_wp = ego_wp
    else:
        _, _, opp_wp = load_raceline_with_speed(map_name, f"{opp_raceline}.csv", 0)

    ego_idx = int(ego_idx) % len(ego_wp)
    if opp_idx is not None:
        return ego_idx, int(opp_idx) % len(opp_wp)

    if opp_raceline == ego_raceline:
        return ego_idx, (ego_idx + int(interval_idx)) % len(opp_wp)

    ego_map_idx = int(find_corresponding_waypoint(ego_wp[ego_idx], opp_wp))
    return ego_idx, (ego_map_idx + int(interval_idx)) % len(opp_wp)
```

After this migration, delete `load_raceline_xytheta_speed()` unless it is still used elsewhere.

### 3.9 Optional: generic two-agent reference geometry

Current `env_ppo.py` has:

```python
def relative_geometry(obs, ref):
    ...
```

This can be moved to `utils.py` only if renamed to be general:

```python
def compute_two_agent_reference_geometry(obs, ref, ego_idx=0, opp_idx=1):
    """Compute two-agent geometry in a reference-line frame."""
    ego_pos = np.array([obs["poses_x"][ego_idx], obs["poses_y"][ego_idx]], dtype=np.float64)
    opp_pos = np.array([obs["poses_x"][opp_idx], obs["poses_y"][opp_idx]], dtype=np.float64)
    ...
```

Recommended return keys should match current reward code to avoid behavior changes:

```text
ego_s_raw
opp_s_raw
ego_d
opp_d
lat_gap
rel_dist
ego_v_s
opp_v_s
```

If this migration causes too much risk, leave `relative_geometry()` in `ppo_utils.py` for now.

Recommended conservative choice:

```text
Move project/reference helpers to utils.py.
Keep relative_geometry() in ppo_utils.py during first refactor.
```

---

## 4. Functions/classes to keep in `ppo_utils.py`

These are PPO-specific and should not go into global `utils.py`.

### 4.1 PPO reward configuration and state

Move from `env_ppo.py` to `ppo_utils.py`:

```python
class RewardWeights:
    ...

class RewardState:
    ...
```

`RewardState.from_obs()` should still call `relative_geometry()` and `wrap_rel_s()`.

If `relative_geometry()` remains in `ppo_utils.py`, `RewardState.from_obs()` can use it directly.
If it is moved to `utils.py`, import it explicitly.

### 4.2 PPO reward helpers

Move from `env_ppo.py` to `ppo_utils.py`:

```python
def rel_progress_potential(...):
    ...

def relative_geometry(...):
    ...  # unless moved to utils.py with a generic name

def clearance_risk(...):
    ...

def compute_shaped_reward(...):
    ...
```

Do not change reward values.

Do not change keys in `reward_terms`.

Do not change in-place update behavior of `RewardState`:

```text
reward_state.last_ego_s = ego_s
reward_state.last_opp_s = opp_s
safe_overtake_hold_time update
safe_overtake_held update
had_safe_overtake_bonus update
```

### 4.3 PPO scenario curriculum

Move from `env_ppo.py` to `ppo_utils.py`:

```python
def sample_opp_speedscale(stage, rng):
    ...

def sample_scenario(stage, rng, map_name, ego_raceline_choices, opp_raceline_choices):
    ...
```

Rationale:

```text
The stage-based interval and speedscale ranges are PPO training curriculum, not general evaluation utilities.
```

Use `utils.resolve_two_agent_indices()` inside `sample_scenario()`.

### 4.4 PPO reward argument helpers

Move from `train_ppo.py` to `ppo_utils.py`:

```python
REQUIRED_V1_REWARD_FIELDS = (...)

def reward_weight_names() -> Tuple[str, ...]:
    ...

def apply_reward_overrides(reward_weights: RewardWeights, args: argparse.Namespace) -> None:
    ...
```

Do not import `argparse` just for type annotation if it complicates dependencies. It is acceptable to annotate `args: Any`.

Update error message in `reward_weight_names()` to not mention `env_ppo.py` after refactor.

Old message:

```text
train_ppo_v1.py requires the compact v1 env_ppo.py
```

New message:

```text
train_ppo.py requires the compact v1 PPO RewardWeights definition.
```

### 4.5 Fixed scenario helper

Move from `train_ppo.py` to `ppo_utils.py`:

```python
def make_fixed_scenario(args):
    ...
```

### 4.6 Checkpoint helpers

Move from `train_ppo.py` to `ppo_utils.py`:

```python
def load_actor_critic(ac: End2Race_PPO, path: str, device: torch.device) -> Dict[str, Any]:
    ...

def load_frozen_bc(path: str, device: torch.device, hidden_scale: int) -> End2Race:
    ...

def save_actor_backbone(ac: End2Race_PPO, path: str) -> None:
    ...

def save_full_checkpoint(ac, path, optimizer, iteration, config) -> None:
    ...
```

These are training/checkpoint specific and should not go into `utils.py`.

### 4.7 Torch tensor and recurrent replay helpers

Move from `train_ppo.py` to `ppo_utils.py`:

```python
def obs_to_tensors(obs, device):
    ...

def zero_hidden(hidden_size, device):
    ...

def forward_policy_sequence(ac, lidar_b, speed_b, starts_b, device):
    ...

def forward_frozen_bc_sequence(bc, lidar_b, speed_b, starts_b, device):
    ...

@torch.no_grad()
def validate_replay_identity(ac, buffer, device, atol):
    ...

@torch.no_grad()
def value_of_obs(ac, obs, hidden, device):
    ...
```

Important dependency issue:

```text
validate_replay_identity() currently type-references RolloutBuffer.
RolloutBuffer should remain in train_ppo.py.
Avoid importing train_ppo.py into ppo_utils.py.
```

Therefore change the annotation:

```python
def validate_replay_identity(ac: End2Race_PPO, buffer: Any, device: torch.device, atol: float) -> Dict[str, float]:
    ...
```

or use a minimal protocol if desired. Do not import `RolloutBuffer` from `train_ppo.py`.

### 4.8 PPO logging helpers

Move from `train_ppo.py` to `ppo_utils.py`:

```python
BOOL_INFO_KEYS = (...)
MEAN_INFO_KEYS = (...)

def summarize_iteration(iteration, rollout, update):
    ...
```

---

## 5. Code that should remain in `train_ppo.py`

### 5.1 `End2RacePPOEnv`

Move `End2RacePPOEnv` from `env_ppo.py` into `train_ppo.py`.

After migration, it should import helper functions instead of defining them locally:

```python
from utils import (
    STEER_LIMIT,
    downsample_lidar_for_model,
    load_positions_and_speeds_from_params,
    load_reference_line,
    resolve_two_agent_indices,
)
from ppo_utils import (
    RewardWeights,
    RewardState,
    compute_shaped_reward,
    sample_scenario,
    sample_opp_speedscale,
)
```

Do not use `from utils import *`.

Core behavior must remain unchanged:

```text
- gym.make("f110-v0", num_agents=2, timestep=0.01, Integrator.RK4)
- opponent planner from setup_opp_planner()
- opponent replans every tracker_steps
- policy observation is lidar + prev_speed only
- action clipping is internal to env execution
- terminated/truncated logic unchanged
```

### 5.2 `RolloutBuffer`

Keep in `train_ppo.py`.

Reason:

```text
The buffer and GAE boundary handling are PPO core logic. Keeping them visible in train_ppo.py makes correctness review easier.
```

Do not change fields:

```text
lidar
prev_speed
raw_actions
rewards
values
log_probs
terminateds
truncateds
trunc_next_values
episode_starts
advantages
returns
```

Do not change GAE semantics.

### 5.3 `collect_rollout()`

Keep in `train_ppo.py`.

Do not change:

```text
- stores raw sampled action
- computes trunc_next_value only when truncated=True
- resets hidden on episode boundary
- calls buffer.compute_returns_and_advantage(candidate_last_value)
```

### 5.4 `ppo_update()`

Keep in `train_ppo.py`.

Do not change:

```text
- advantage normalization
- PPO ratio/clipped surrogate
- value loss
- entropy term
- BC anchor
- bound loss
- KL early stop
- post-step KL logging
```

Only update imports to use `forward_policy_sequence()` and `forward_frozen_bc_sequence()` from `ppo_utils.py`.

### 5.5 `parse_arguments()` and `main()`

Keep in `train_ppo.py`.

Update imports and references after helper migration.

In `parse_arguments()`, dynamic reward args should still be generated from:

```python
for name in reward_weight_names():
    parser.add_argument(f"--{name}", type=float, default=None)
```

---

## 6. Expected imports after refactor

### 6.1 `utils.py`

Should not import torch/model/PPO code.

Allowed imports:

```python
import os
import json
from dataclasses import dataclass
import numpy as np
```

Do not import:

```text
torch
model
train_ppo
ppo_utils
gym
f110_gym
demonstration
```

### 6.2 `ppo_utils.py`

Expected imports:

```python
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from model import End2Race, End2Race_PPO
from utils import (
    STEER_LIMIT,
    LIDAR_DIM,
    ACTION_DIM,
    load_raceline_with_speed,
    resolve_two_agent_indices,
    wrap_rel_s,
    unwrap_progress,
    project_to_reference,
)
```

Only import names actually used.

If `ACTION_DIM` remains in `ppo_utils.py`, do not import it from `utils.py`.

### 6.3 `train_ppo.py`

Expected imports:

```python
from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Tuple

import gym
import f110_gym
import numpy as np
import torch
import torch.optim as optim
from f110_gym.envs.base_classes import Integrator

from latticeplanner.utils import obsDict2oppoArray
from demonstration import setup_opp_planner
from model import End2Race, End2Race_PPO
from utils import (
    STEER_LIMIT,
    LIDAR_DIM,
    ACTION_DIM,
    downsample_lidar_for_model,
    load_positions_and_speeds_from_params,
    load_reference_line,
    resolve_two_agent_indices,
)
from ppo_utils import (
    BOOL_INFO_KEYS,
    MEAN_INFO_KEYS,
    RewardWeights,
    RewardState,
    apply_reward_overrides,
    compute_shaped_reward,
    forward_frozen_bc_sequence,
    forward_policy_sequence,
    load_actor_critic,
    load_frozen_bc,
    make_fixed_scenario,
    obs_to_tensors,
    reward_weight_names,
    sample_opp_speedscale,
    sample_scenario,
    save_actor_backbone,
    save_full_checkpoint,
    summarize_iteration,
    validate_replay_identity,
    value_of_obs,
    zero_hidden,
)
```

Remove:

```python
from env_ppo import End2RacePPOEnv, RewardWeights
```

---

## 7. Exact migration checklist

### Phase 1: Update `utils.py`

Add or migrate:

```text
[ ] STEER_LIMIT
[ ] LIDAR_DIM
[ ] LIDAR_MAX_RANGE
[ ] optional ACTION_DIM
[ ] downsample_lidar_for_model()
[ ] ReferenceLine
[ ] load_reference_line()
[ ] wrap_rel_s()
[ ] unwrap_progress()
[ ] project_to_reference()
[ ] resolve_two_agent_indices()
```

Optional:

```text
[ ] compute_two_agent_reference_geometry()
```

Delete from PPO helper after migration:

```text
[ ] load_raceline_xytheta_speed() unless still used
```

### Phase 2: Create `ppo_utils.py`

Move in:

```text
[ ] BOOL_INFO_KEYS
[ ] MEAN_INFO_KEYS
[ ] REQUIRED_V1_REWARD_FIELDS
[ ] RewardWeights
[ ] RewardState
[ ] rel_progress_potential()
[ ] relative_geometry() unless moved to utils.py
[ ] clearance_risk()
[ ] compute_shaped_reward()
[ ] sample_opp_speedscale()
[ ] sample_scenario()
[ ] reward_weight_names()
[ ] apply_reward_overrides()
[ ] make_fixed_scenario()
[ ] load_actor_critic()
[ ] load_frozen_bc()
[ ] save_actor_backbone()
[ ] save_full_checkpoint()
[ ] obs_to_tensors()
[ ] zero_hidden()
[ ] forward_policy_sequence()
[ ] forward_frozen_bc_sequence()
[ ] validate_replay_identity()
[ ] value_of_obs()
[ ] summarize_iteration()
```

### Phase 3: Merge env into `train_ppo.py`

Move class from `env_ppo.py`:

```text
[ ] End2RacePPOEnv
```

Update it to call:

```text
[ ] downsample_lidar_for_model()
[ ] load_reference_line()
[ ] sample_scenario()
[ ] sample_opp_speedscale()
[ ] resolve_two_agent_indices()
[ ] load_positions_and_speeds_from_params()
[ ] compute_shaped_reward()
```

### Phase 4: Update `train_ppo.py`

Remove local definitions that moved to `ppo_utils.py`:

```text
[ ] reward_weight_names()
[ ] load_actor_critic()
[ ] load_frozen_bc()
[ ] save_actor_backbone()
[ ] save_full_checkpoint()
[ ] obs_to_tensors()
[ ] zero_hidden()
[ ] forward_policy_sequence()
[ ] forward_frozen_bc_sequence()
[ ] validate_replay_identity()
[ ] value_of_obs()
[ ] apply_reward_overrides()
[ ] make_fixed_scenario()
[ ] summarize_iteration()
```

Keep:

```text
[ ] End2RacePPOEnv
[ ] RolloutBuffer
[ ] collect_rollout()
[ ] ppo_update()
[ ] parse_arguments()
[ ] main()
```

### Phase 5: Remove/deprecate `env_ppo.py`

Preferred:

```text
[ ] Delete env_ppo.py if no scripts import it.
```

Conservative:

```text
[ ] Leave env_ppo.py temporarily, but do not import it from train_ppo.py.
[ ] Add a short comment at top: deprecated; End2RacePPOEnv now lives in train_ppo.py.
```

Before deletion, search:

```bash
grep -R "from env_ppo\|import env_ppo" -n .
```

---

## 8. Do-not-change list

The coding agent must not change the following unless explicitly asked:

```text
1. Reward numeric defaults in RewardWeights.
2. Reward formula in compute_shaped_reward().
3. Reward term names in info.
4. Scenario sampling ranges for PPO stages.
5. F1Tenth timestep = 0.01.
6. Integrator = Integrator.RK4.
7. Actor observation keys: "lidar", "prev_speed".
8. Action clipping limits.
9. Raw action storage in RolloutBuffer.
10. Terminated/truncated GAE logic.
11. PPO loss formula.
12. Checkpoint format keys: actor_critic, actor, optimizer, iteration, config, hidden_scale, log_std.
13. Actor-only checkpoint saving as ac.actor.state_dict().
14. No privileged simulator geometry entering actor observation.
```

---

## 9. Verification commands

Run syntax checks:

```bash
python -m py_compile utils.py ppo_utils.py train_ppo.py model.py
```

Run import check:

```bash
python - <<'PY'
import utils
import ppo_utils
import train_ppo
print('imports_ok')
PY
```

Check that `train_ppo.py` does not import `env_ppo.py`:

```bash
grep -n "env_ppo" train_ppo.py && exit 1 || echo "no env_ppo import"
```

Check no wildcard import was added:

```bash
grep -R "from utils import \*" -n train_ppo.py ppo_utils.py utils.py && exit 1 || echo "no wildcard utils import"
```

Run a minimal fixed-scenario smoke test if dependencies and pretrained checkpoint are available:

```bash
python train_ppo.py \
  --model_path pretrained/end2race.pth \
  --bc_model_path pretrained/end2race.pth \
  --map_name Austin \
  --fixed_scenario \
  --ego_idx 0 \
  --interval_idx 15 \
  --opp_raceline raceline1 \
  --opp_speedscale 0.5 \
  --sim_duration 1.0 \
  --rollout_steps 100 \
  --total_iterations 1 \
  --save_every 0 \
  --log_every 1 \
  --device cpu
```

If `--rollout_steps` must be at least one full episode, for `sim_duration=1.0` and timestep 0.01, use 100 steps. For default `sim_duration=8.0`, use at least 800 steps.

---

## 10. Behavior identity tests to add if possible

If time permits, add a small script or temporary Python snippet to compare old and new helper outputs. If `env_ppo.py` is still available during refactor, compare before deleting.

### 10.1 Geometry identity

```python
from env_ppo import wrap_rel_s as old_wrap, unwrap_progress as old_unwrap
from utils import wrap_rel_s as new_wrap, unwrap_progress as new_unwrap

for delta in [-100, -10, -1, 0, 1, 10, 100]:
    assert old_wrap(delta, 50.0) == new_wrap(delta, 50.0)

for raw, last in [(0.1, 49.9), (49.9, 0.1), (10.0, 10.2), (25.0, 75.0)]:
    assert old_unwrap(raw, last, 50.0) == new_unwrap(raw, last, 50.0)
```

### 10.2 Reward identity

Create identical fake or recorded `obs`, initialize identical `RewardState`, and compare:

```python
old_reward, old_terms = old_compute_shaped_reward(obs, old_state, ref, rw, dt)
new_reward, new_terms = new_compute_shaped_reward(obs, new_state, ref, rw, dt)

assert old_reward == new_reward
assert old_terms.keys() == new_terms.keys()
for key in old_terms:
    assert abs(old_terms[key] - new_terms[key]) < 1e-9
```

### 10.3 Actor-only checkpoint load

After the smoke test saves actor checkpoint:

```python
from model import End2Race
import torch

model = End2Race(mask_prob=0.0, hidden_scale=4)
model.load_state_dict(torch.load('pretrained/end2race_ppo_v1.pth', map_location='cpu', weights_only=False))
print('actor_checkpoint_loads')
```

---

## 11. Acceptance criteria

The refactor is accepted only if all are true:

```text
[ ] `python -m py_compile utils.py ppo_utils.py train_ppo.py model.py` passes.
[ ] `train_ppo.py` no longer imports `env_ppo.py`.
[ ] `from utils import *` is not used in PPO files.
[ ] `End2RacePPOEnv` is present in `train_ppo.py`.
[ ] PPO-specific reward classes/functions are in `ppo_utils.py`, not global `utils.py`.
[ ] Generic reference/track/LiDAR helpers are in `utils.py`.
[ ] `RolloutBuffer`, `collect_rollout()`, and `ppo_update()` remain in `train_ppo.py`.
[ ] Reward defaults and reward term keys are unchanged.
[ ] Terminated/truncated GAE logic is unchanged.
[ ] Fixed-scenario one-iteration smoke test runs or fails only because external assets/checkpoints are missing.
[ ] Actor-only checkpoint still loads into `End2Race`.
```

---

## 12. Suggested final top-of-file comments

### `ppo_utils.py`

```python
"""PPO-specific helpers for End2Race fine-tuning.

This file intentionally contains PPO reward, curriculum, checkpoint, and
recurrent replay helpers. Generic raceline, LiDAR, and track geometry helpers
belong in utils.py. The main PPO environment, rollout buffer, collection loop,
and PPO update live in train_ppo.py.
"""
```

### `train_ppo.py`

```python
"""Compact v1 PPO fine-tuning script for End2Race.

Design assumptions:
- Actor observation stays deployable: LiDAR 360 + previous ego speed + GRU hidden.
- The model class is End2Race_PPO from model.py.
- Reward uses simulator geometry internally but does not expose privileged state to actor.
- Collision is true termination. Time limit is truncation and bootstraps V(s_next).
"""
```

### `utils.py` new section heading

```python
# ---------------------------------------------------------------------------
# Track/reference geometry and LiDAR preprocessing
# ---------------------------------------------------------------------------
```

---

## 13. Summary for coding agent

Perform a clean, behavior-preserving migration:

```text
1. Move generic LiDAR/reference/index helpers into utils.py.
2. Create ppo_utils.py for PPO-specific reward, curriculum, checkpoint, tensor replay, and logging helpers.
3. Move End2RacePPOEnv into train_ppo.py.
4. Keep RolloutBuffer, collect_rollout, ppo_update, parse_arguments, and main in train_ppo.py.
5. Remove train_ppo.py dependency on env_ppo.py.
6. Do not change PPO math, reward values, observation boundary, or GAE semantics.
7. Run syntax/import/smoke checks.
```

