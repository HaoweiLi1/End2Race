# End2Race PPO Critic-Experiment Common Fixed-Parameter Contract

**Status:** Owner-approved common configuration for the next critic experiments  
**Purpose:** Freeze all non-critic variables so that later experiments change only the critic input/architecture.  
**Important:** This contract intentionally overrides several older experiment documents. The local coding agent must implement the values below exactly and must not silently fall back to the current repository defaults.

---

## 0. Scope and authority

The following are **not experiment variables**:

- scenario pools and scenario schedule;
- vector-environment composition;
- rollout size;
- minibatch size;
- actor training budget;
- PPO/GAE/credit parameters;
- reward;
- action-exploration distribution;
- random seed;
- critic warm-up protocol;
- critic optimization budget;
- evaluator and checkpoint schema.

The only intended experiment variable in the next stage is:

```text
critic input and critic architecture
```

Examples include raw single-frame, detached actor hidden, independent recurrent critic, compact privileged physical critic, or a separately approved hybrid. The exact critic arms and the total number of formal outer updates are **not fixed by this document** and must be specified in the later critic experiment plan.

---

# 1. Software and deployment contract

```yaml
python: "3.10"
pytorch: "2.7.0+cu128"
gymnasium: "1.2.3"
stable_baselines3: "2.7.1"
sb3_contrib: "2.7.1"
algorithm_base: "sb3_contrib.RecurrentPPO"
```

Fixed system behavior:

```text
Map/training domain: Austin
Learned agent: ego only
Opponent: fixed Lattice Planner + Pure Pursuit
Simulator frequency: 100 Hz
Actor/GRU inference frequency: 100 Hz
Opponent planner replan frequency: existing implementation, unchanged
```

The final deployable actor checkpoint must remain the original End2Race **12-key actor-only state dict** and must strict-load into a newly created original `End2Race` model. Critic, optimizer, RNG, sampler, and warm-up state are training-only and must not enter the deployable actor checkpoint.

---

# 2. Scenario-pool contract

## 2.1 Hard branch

```yaml
hard_pool:
  id: "H1_EXPANDED_DET"
  manifest: "authoritative repository H1-full manifest"
  expected_case_count: 482
  use_early_subset: false
  use_h2: false
  use_h3: false
  mutate_cases: false
```

Every H1 case must retain its authoritative:

- ego start;
- opponent raceline;
- opponent speed scale;
- interval;
- initial pose;
- map and episode settings.

## 2.2 Ordinary branch

```yaml
ordinary_pool:
  id: "authoritative ordinary training panel"
  expected_case_count: 600
  mutate_cases: false
```

## 2.3 Hard/ordinary mixture

```yaml
scenario_mix:
  hard_envs: 8
  ordinary_envs: 8
  hard_fraction_by_env_role: 0.50
  ordinary_fraction_by_env_role: 0.50
```

The 50:50 split is produced by fixed environment roles. Do not perform a new Bernoulli hard/ordinary draw at every reset.

## 2.4 Scenario sampling

**New fixed owner decision:**

```yaml
scenario_sampling:
  mode: "global_without_replacement_cycle"
  hard_queue_shared_across_hard_envs: true
  ordinary_queue_shared_across_ordinary_envs: true
  shuffle_at_start_of_each_cycle: true
  persist_queue_and_cursor_in_full_training_state: true
```

Required behavior:

```text
1. Deterministically shuffle the complete role-specific pool.
2. Consume each scenario once before any scenario is reused.
3. When the queue is exhausted, deterministically reshuffle and start a new cycle.
4. Use one shared hard queue for all eight hard logical environments.
5. Use one shared ordinary queue for all eight ordinary logical environments.
```

Do not use:

```yaml
online_hard_mining: false
curriculum: false
dynamic_case_weighting: false
scenario_mutation: false
```

---

# 3. Randomness and seed contract

```yaml
seed:
  run_seed: 0
  same_seed_for_all_critic_arms: true
```

Use `run_seed=0` to initialize all relevant random components:

- Python random;
- NumPy;
- PyTorch CPU;
- PyTorch CUDA;
- environment randomness;
- hard and ordinary queue shuffling;
- action sampling;
- recurrent minibatch ordering.

Internally separate the random streams for at least:

```text
scenario scheduling
environment randomness
action exploration
minibatch ordering
```

The sub-stream implementation is internal and is not an experiment parameter. All critic arms must use the same deterministic derivation from seed 0.

Changing physical vector-environment rank must not change the random stream assigned to a logical hard/ordinary environment.

---

# 4. Vector-environment and episode contract

```yaml
environment:
  n_envs: 16
  physical_role_order:
    - hard
    - ordinary
    - hard
    - ordinary
    - hard
    - ordinary
    - hard
    - ordinary
    - hard
    - ordinary
    - hard
    - ordinary
    - hard
    - ordinary
    - hard
    - ordinary

  simulator_hz: 100
  actor_hz: 100

  hard_episode_horizon_seconds: 8.0
  ordinary_episode_horizon_seconds: 8.0
  max_episode_steps: 800

  ego_collision_semantics: "terminated"
  time_limit_semantics: "truncated_with_bootstrap"
  opponent_only_collision_terminates_ego: false
  opponent_only_collision_reward_penalty: false
```

Do not shorten hard or ordinary episodes to 3 or 4 seconds.

---

# 5. Rollout and recurrent minibatch contract

```yaml
rollout:
  n_steps_per_env: 6400
  n_envs: 16
  transitions_per_outer_update: 102400
```

Calculation:

```text
16 × 6400 = 102,400 transitions per outer update
```

At an 800-step maximum episode length, this is approximately:

```text
at least 64 completed hard episode-equivalents
at least 64 completed ordinary episode-equivalents
approximately 128 total episode-equivalents per update
```

Early collision termination may increase the number of completed episodes.

## 5.1 Minibatch

```yaml
minibatch:
  batch_size_valid_transitions: 12800
  recurrent_generator: "current verified stock/custom-compatible recurrent generator"
  custom_sequence_stratified_sampler: false
```

Calculation:

```text
102,400 / 12,800 = 8 logical minibatches per epoch
```

With `n_steps=6400`, one environment block contains 6,400 transitions. The interleaved H/O rank order and 12,800-transition minibatch are intended to make a logical minibatch cover approximately:

```text
6,400 hard valid transitions
+
6,400 ordinary valid transitions
```

## 5.2 Mandatory preflight composition audit

Before formal critic experiments, collect one rollout and record every logical minibatch:

- valid hard transitions;
- valid ordinary transitions;
- hard ratio;
- hard/ordinary recurrent sequence count;
- unique hard/ordinary scenario count;
- padding rows and padding ratio.

Required result:

```text
every logical minibatch contains both hard and ordinary data
hard valid-transition ratio is within 45%–55%
8 logical minibatches per epoch
```

If this fails:

```text
STOP_FIXED_MINIBATCH_CONTRACT_INVALID
```

Do not automatically implement the earlier S3 sequence-level stratified sampler and do not silently modify batch size or environment order.

---

# 6. Actor architecture and parameter groups

Actor observation:

```text
360D LiDAR
+
previous measured ego speed using the verified deployment-timing semantics
```

Actor path:

```text
LiDAR → frozen pressure preprocessing
previous measured speed → frozen speed MLP
concatenate → GRU → output head
→ steering mean and desired-speed mean
```

## 6.1 Trainable actor parameters

```yaml
actor_trainable:
  - gru
  - output_head
```

## 6.2 Frozen actor parameters

```yaml
actor_frozen:
  - lidar_preprocess_k
  - speed_mlp
  - dummy_embedding_parameters
  - log_std
```

## 6.3 Actor learning rates

```yaml
actor_optimizer:
  gru_lr: 1.0e-6
  output_head_lr: 1.0e-5
  optimizer_type: "current verified Adam implementation"
```

All unspecified Adam arguments remain exactly equal to the current verified repository implementation/SB3 defaults.

---

# 7. Actor optimization budget

**New fixed owner decision:**

```yaml
actor_training:
  epochs_per_outer_update: 3
  minibatches_per_epoch: 8
  optimizer_steps_per_outer_update: 24
```

Calculation:

```text
3 actor epochs × 8 minibatches = 24 actor optimizer steps per outer update
```

Rules:

- old log probabilities remain the rollout behavior-policy log probabilities;
- PPO advantages are computed once for the rollout and remain fixed;
- critic parameters are frozen during actor-only optimization;
- actor gradient norm is clipped independently from critic gradients;
- do not increase actor epochs to 5 or 10;
- do not use a single shared SB3 `n_epochs` value to train actor and critic together.

A separate actor/critic training loop is required because actor uses 3 epochs and critic uses 8 epochs.

---

# 8. PPO and credit-assignment parameters

```yaml
ppo:
  gamma: 0.999
  gae_lambda: 0.995

  clip_range: 0.10
  clip_range_vf: null

  normalize_advantage: true
  advantage_normalization_scope: "current verified recurrent-minibatch valid-mask implementation"

  vf_coef: 0.5
  ent_coef: 0.0

  actor_max_grad_norm: 0.5
  critic_max_grad_norm: 0.5

  target_kl: null
  use_sde: false
```

Interpretation boundaries:

- `gamma` and `gae_lambda` remain fixed; do not redesign credit propagation.
- `clip_range=0.10` is the fixed conservative PPO actor-ratio clip.
- `clip_range_vf=None` means no value-function clipping.
- `normalize_advantage=True` retains the verified implementation.
- `vf_coef=0.5` preserves the historical value-loss scaling.
- `ent_coef=0.0` disables entropy bonus.
- `target_kl=None`; record approximate KL but do not early-stop actor epochs using target KL.
- Actor and critic gradients must be clipped separately at 0.5.

Do not modify:

- termination/truncation semantics;
- timeout bootstrap;
- GAE formula;
- return-target construction;
- advantage normalization formula.

---

# 9. Reward contract

```yaml
reward:
  ego_progress_delta_weight: 0.01
  relative_track_progress_delta_weight: 0.02
  first_ego_collision_penalty: -2.0
```

Equivalent form:

```text
reward
=
0.01 × ego progress delta
+
0.02 × relative track progress delta
-
2.0 × first ego collision
```

Do not add:

- TTC reward;
- clearance/proximity reward;
- steering smoothness reward;
- terminal overtake bonus;
- margin shaping;
- reward redistribution.

TTC may later be evaluated as a privileged critic input only under a separately approved critic design; it is not part of the common reward.

---

# 10. Action-exploration contract

```yaml
exploration:
  steering_distribution: "squashed latent Gaussian"
  steering_bound_radians: 0.52
  steering_latent_std: 0.03

  speed_distribution: "Gaussian in physical desired-speed units"
  speed_physical_std_mps: 0.15

  temporal_process: "iid_per_100Hz_step"
  log_std_trainable: false
  entropy_bonus_coefficient: 0.0
```

Required action contract:

```text
latent steering sample
→ 0.52 × tanh(latent)
→ same action used for stored PPO action, log probability, wrapper action,
  and F110 execution
```

Exploration is active only during stochastic rollout collection. Deterministic evaluation uses the actor mean/mode with no exploration noise.

Do not use:

- AR(1);
- finite-window sustained exploration;
- P2 action pulses;
- action hold;
- LiDAR noise;
- speed-observation noise;
- dynamically changing standard deviations.

---

# 11. Critic common optimization contract

The critic **input and architecture** are the experiment variables. The training protocol below is identical for every critic arm.

```yaml
critic_optimizer:
  learning_rate: 3.0e-4
  optimizer_type: "independent Adam optimizer"
```

The critic optimizer must contain critic parameters only. Critic-only steps must not update actor parameters or advance the actor optimizer state.

---

# 12. One-time critic warm-up before formal Update 1

**New fixed owner decision:** warm-up is mandatory and is not an experiment arm.

```yaml
critic_warmup:
  enabled: true
  timing: "before formal outer update 1"
  warmup_rollouts: 1
  warmup_transitions: 102400

  actor_frozen: true
  actor_optimizer_steps: 0

  maximum_critic_epochs: 20
  patience: 3

  split_unit: "scenario or complete recurrent sequence"
  train_fraction: 0.80
  validation_fraction: 0.20
  stratify_hard_ordinary: true

  restore_best_validation_checkpoint: true
  preserve_best_critic_optimizer_state: true
  discard_warmup_rollout_before_actor_training: true
  exposed_min_delta_parameter: false
```

Warm-up procedure:

```text
1. Load the canonical BC actor.
2. Initialize the selected critic arm.
3. Freeze the actor.
4. Collect one 102,400-transition warm-up rollout W0.
5. Split W0 by scenario/complete recurrent sequence, not individual transitions.
6. Train only the critic for at most 20 epochs.
7. Stop when validation loss fails to improve for 3 consecutive epochs.
8. Restore the best validation critic checkpoint and matching optimizer state.
9. Discard W0 for actor learning.
10. Collect a fresh formal rollout W1 for formal outer update 1.
```

The warm-up rollout and warm-up critic steps are excluded from the formal outer-update count and formal transition budget.

Use the same reward, gamma, termination/truncation, bootstrap, and return-target implementation as the formal PPO pipeline. Warm-up must not create a new credit definition.

---

# 13. Critic training inside every formal outer update

**New fixed owner decision:**

```yaml
critic_training:
  epochs_per_outer_update: 8
  minibatches_per_epoch: 8
  optimizer_steps_per_outer_update: 64

  actor_frozen_during_critic_training: true
  recompute_advantages_after_critic_training: false
```

Calculation:

```text
8 critic epochs × 8 minibatches = 64 critic optimizer steps per outer update
```

Formal outer-update sequence:

```text
current actor π_k and critic V_k
→ collect 102,400-transition rollout D_k
→ compute old log probabilities, returns, and GAE advantages once
→ freeze critic; train actor for 3 epochs / 24 steps
→ freeze actor; train critic for 8 epochs / 64 steps
→ obtain π_(k+1), V_(k+1)
→ collect the next rollout
```

The critic trained at the end of update `k` mainly supplies value predictions, timeout bootstrap, and GAE baselines for update `k+1`.

Do not:

- recompute the current rollout’s advantages after extra critic training;
- run the actor again on the same rollout after critic training;
- share one optimizer between actor and critic;
- train critic for 20 or 50 epochs inside every formal update.

---

# 14. Checkpoint and resume contract

Save after every formal outer update:

```text
1. actor-only 12-key deployable checkpoint;
2. full PPO training state;
3. critic parameters;
4. actor optimizer state;
5. critic optimizer state;
6. Python/NumPy/PyTorch/CUDA RNG state;
7. scenario queue order and cursor;
8. resolved configuration;
9. update telemetry;
10. SHA-256 hashes.
```

Warm-up completion must also save a resumable training-state checkpoint before formal update 1.

Resume must restore the exact critic, optimizers, RNG, and scenario queue/cursor. Do not resume long training from an actor-only checkpoint.

---

# 15. Required telemetry

## 15.1 Rollout level

Record per outer update:

- total transitions;
- hard/ordinary transitions;
- completed hard/ordinary episodes;
- partial hard/ordinary episodes;
- unique hard/ordinary scenarios;
- scenario queue cycle and cursor;
- episode-length p10/p50/p90;
- hard/ordinary collision/follow/overtake;
- rollout wall time.

## 15.2 Actor training

Record per actor minibatch/epoch:

- valid transitions and padding ratio;
- hard/ordinary ratio;
- policy loss;
- approximate KL;
- clip fraction;
- entropy diagnostic;
- GRU/head pre-clip gradient norms;
- actor clip multiplier;
- actor parameter delta.

Verify:

```text
3 epochs
8 minibatches per epoch
24 actor optimizer steps per outer update
```

## 15.3 Critic training

Record per critic epoch:

- train value loss;
- validation value loss when applicable;
- explained variance;
- critic pre-clip gradient norm;
- critic clip multiplier;
- critic parameter delta.

Verify:

```text
warm-up: actor optimizer steps = 0
formal update: 8 critic epochs
64 critic optimizer steps per outer update
```

---

# 16. Mandatory implementation checks before formal training

The coding agent must run and report:

1. **Resolved-config audit**
   - every value in this document is reflected in the runtime config;
   - report all fields that override repository defaults.

2. **H1/ordinary authority audit**
   - manifest paths;
   - hashes;
   - counts 482 and 600.

3. **Global no-replacement scheduler test**
   - no duplicate before cycle exhaustion;
   - deterministic reshuffle;
   - queue/cursor resume identity.

4. **H/O interleaved minibatch audit**
   - eight logical minibatches;
   - each contains both roles;
   - hard ratio 45%–55%.

5. **Warm-up zero-actor-update test**
   - actor parameters bitwise unchanged;
   - actor optimizer step count zero;
   - critic parameters change finitely;
   - patience and best-checkpoint restoration work.

6. **Separate optimizer test**
   - actor phase changes only actor;
   - critic phase changes only critic;
   - actor and critic optimizer states are isolated.

7. **Recurrent replay test**
   - old-log-prob replay identity;
   - correct pre-action hidden;
   - correct episode-start reset;
   - correct padding masks;
   - correct timeout bootstrap.

8. **Action identity**
   - stored action;
   - likelihood action;
   - wrapper action;
   - F110 executed action;
   - all remain identical under the verified transform.

9. **Checkpoint compatibility**
   - final actor-only checkpoint contains exactly the original 12 keys;
   - strict-load PASS.

Any failure must stop implementation/training and preserve evidence. Do not silently alter the contract to make a test pass.

---

# 17. Explicitly not fixed yet

The following must be defined in a later critic-experiment execution plan:

```text
1. Exact critic arms and names.
2. Exact physical-feature list for compact privileged critic.
3. Whether TTC enters any critic arm.
4. Whether a hybrid critic is included.
5. Number of formal outer updates per critic arm.
6. Formal checkpoint evaluation schedule.
7. Final success/selection criteria.
```

Do not start the formal critic comparison until these items are approved.

---

# 18. Compact machine-readable configuration

```yaml
common_fixed_config:
  seed: 0

  software:
    python: "3.10"
    pytorch: "2.7.0+cu128"
    gymnasium: "1.2.3"
    stable_baselines3: "2.7.1"
    sb3_contrib: "2.7.1"

  scenario:
    hard_pool: "H1_EXPANDED_DET"
    hard_count: 482
    ordinary_pool: "authoritative_ordinary_training_panel"
    ordinary_count: 600
    hard_envs: 8
    ordinary_envs: 8
    sampling: "global_without_replacement_cycle"
    online_hard_mining: false
    curriculum: false
    scenario_mutation: false

  environment:
    map: "Austin"
    n_envs: 16
    env_order: "H,O,H,O,H,O,H,O,H,O,H,O,H,O,H,O"
    simulator_hz: 100
    actor_hz: 100
    episode_horizon_seconds: 8.0
    max_episode_steps: 800
    ego_collision: "terminated"
    timeout: "truncated_with_bootstrap"
    opponent_only_collision_terminates_ego: false

  rollout:
    n_steps_per_env: 6400
    transitions_per_outer_update: 102400
    batch_size: 12800
    minibatches_per_epoch: 8

  actor:
    trainable: ["gru", "output_head"]
    frozen: ["k", "speed_mlp", "dummy_embedding", "log_std"]
    gru_lr: 1.0e-6
    head_lr: 1.0e-5
    epochs_per_outer_update: 3
    optimizer_steps_per_outer_update: 24

  critic_common:
    lr: 3.0e-4
    warmup_enabled: true
    warmup_rollouts: 1
    warmup_transitions: 102400
    warmup_max_epochs: 20
    warmup_patience: 3
    warmup_restore_best: true
    warmup_discard_rollout: true
    epochs_per_outer_update: 8
    optimizer_steps_per_outer_update: 64

  ppo:
    gamma: 0.999
    gae_lambda: 0.995
    clip_range: 0.10
    clip_range_vf: null
    normalize_advantage: true
    vf_coef: 0.5
    ent_coef: 0.0
    actor_max_grad_norm: 0.5
    critic_max_grad_norm: 0.5
    target_kl: null
    use_sde: false

  reward:
    ego_progress_delta_weight: 0.01
    relative_progress_delta_weight: 0.02
    first_ego_collision_penalty: -2.0

  exploration:
    steering_distribution: "squashed_latent_gaussian"
    steering_bound_radians: 0.52
    steering_latent_std: 0.03
    speed_distribution: "physical_gaussian"
    speed_physical_std_mps: 0.15
    temporal_process: "iid"
    log_std_trainable: false

  formal_experiment_variables:
    - "critic_input"
    - "critic_architecture"

  not_yet_fixed:
    - "critic_arms"
    - "formal_outer_updates"
    - "evaluation_schedule"
    - "critic_selection_rule"
```

---

# 19. Copy-paste instruction for the local coding agent

```text
Implement the End2Race PPO common fixed-parameter contract in:

End2Race_PPO_Critic_Experiments_Common_Fixed_Parameters.md

This is an implementation-only task. Do not start the formal critic comparison yet.

The contract overrides older repository/default experiment values for:
- H/O interleaved physical env order;
- global role-specific no-replacement scenario cycles;
- run seed 0;
- n_steps 6400;
- rollout size 102400;
- batch size 12800;
- actor epochs 3;
- separate actor and critic optimizers;
- one critic warm-up rollout before formal update 1;
- warm-up max 20 epochs with patience 3;
- critic epochs 8 per formal update.

Do not use a single SB3 n_epochs value for both actor and critic.
Do not recompute the current rollout’s advantages after critic-only epochs.
Do not modify reward, gamma, lambda, exploration, evaluator, action transform,
checkpoint schema, or episode semantics.
Do not introduce a new critic arm yet.

First:
1. inspect current HEAD/worktree/config;
2. produce a changed-file impact map;
3. implement the common config and separate training phases;
4. add the required unit/integration tests;
5. run the preflight audits;
6. report exact resolved values, test results, changed files, and remaining blockers.

Fail closed on any mismatch. Preserve evidence. Do not silently redesign the contract.
```
