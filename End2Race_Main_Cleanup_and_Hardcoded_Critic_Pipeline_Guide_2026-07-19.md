# End2Race `main` 分支清理与固定 Critic 实验管线硬编码指导文档

**状态：** 替代上一版 `End2Race_Main_Branch_Cleanup_Guide_2026-07-19.md`  
**目标仓库：** `HaoweiLi1/End2Race`  
**审查基线：** `main@1d404a412c9faa2a451bb6318848b2d5c349ad97`  
**参数权威：** `End2Race_PPO_Critic_Experiments_Common_Fixed_Parameters.md`  
**执行方式：** 所有非 critic 参数直接写死到源码；每个 critic 实验直接修改 critic 相关源码后重新运行  
**禁止事项：** 不 commit、不 push、不维护 config sweep、critic profile selector 或历史 backend selector  
**代码风格：** 参考根目录 `model.py`、`train.py`，优先短函数、直接控制流、错误直接抛出

---

# 0. Owner 最终决定

## 0.1 参数不再通过 config profile 管理

以下内容全部是唯一正式常量，直接写入源码：

```text
软件版本
地图与 simulator 合同
H1 / ordinary scenario 内容
H/O env 数量与物理顺序
scenario queue 调度
seed 与随机流派生
rollout size
minibatch size
actor optimizer 参数
actor epochs
PPO / GAE 参数
reward
exploration
critic warm-up
critic optimizer 参数
critic epochs
checkpoint 与 resume 合同
```

不得再通过：

```text
PPOConfig dataclass
CONFIGS registry
--config
critic_profile
backend selector
experiment arm registry
YAML/JSON 配置覆盖
```

改变这些固定值。

## 0.2 Critic 是唯一源码实验变量

不保留：

```text
C0/C1/C2/C3 runtime selector
critic registry
critic factory by name
if critic_profile == ...
multiple critic modules simultaneously
```

每次 critic 实验：

1. 在当前 active critic 源码位置直接实现一个 critic；
2. 同步修改 critic 所需 observation、buffer state 和 value replay；
3. 运行预检；
4. 运行正式训练；
5. 保存源码 hash 和完整 Git diff；
6. 下一 critic 实验再直接修改源码。

主分支任一时刻只存在一套 active critic 实现。

## 0.3 当前清理后初始 critic

清理完成后的起始实现使用当前简单 C0：

```text
361D actor observation
→ Linear(361,64)
→ Tanh
→ Linear(64,1)
```

它只作为第一个可运行源码基线，不代表最终 critic 选择。

测试 detached actor hidden、independent recurrent、compact privileged physical
或其他 owner 批准 critic 时，直接修改 active critic 源码，不增加 selector。

## 0.4 历史目录仍然全部清除

完成一次性归档后删除：

```text
tests/
scripts/
runs/
ppo_experiments/
posttrained/
eval_results/
```

最终仓库只保留：

```text
PPO_HISTORY.md
```

作为 PPO 历史摘要。

实现固定管线期间需要的测试与审计脚本全部放在：

```text
/tmp/end2race_fixed_pipeline_validation/
```

验证完成后不进入仓库。

---

# 1. 唯一固定训练合同

## 1.1 软件

```text
Python                 3.10
PyTorch                2.7.0+cu128
Gymnasium              1.2.3
stable-baselines3      2.7.1
sb3-contrib            2.7.1
algorithm base         RecurrentPPO
```

在正式入口启动时直接检查版本。版本不符直接抛出异常，不做 fallback。

## 1.2 部署

```text
Map                     Austin
Learned agent           ego only
Opponent                Lattice Planner + Pure Pursuit
Simulator               100 Hz
Actor / GRU             100 Hz
Deployable checkpoint   original End2Race 12-key actor state dict
```

Critic、optimizer、RNG、scenario queue 和 warm-up state 不进入部署 actor。

## 1.3 Scenario

### Hard

```text
H1_EXPANDED_DET
482 cases
manifest = ppo/hard_pools/h1_expanded_det.json
```

禁止：

```text
H0
H1 early
H2
H3
case mutation
online mining
curriculum
dynamic weighting
```

### Ordinary

```text
authoritative ordinary training panel
600 cases
```

### Env role

固定 16 logical env：

```text
rank 0  hard
rank 1  ordinary
rank 2  hard
rank 3  ordinary
...
rank 14 hard
rank 15 ordinary
```

即：

```text
H,O,H,O,H,O,H,O,H,O,H,O,H,O,H,O
```

禁止使用旧的：

```text
rank 0–7 hard
rank 8–15 ordinary
```

### Episode

```text
hard horizon      8.0 s
ordinary horizon  8.0 s
max steps         800
ego collision     terminated
time limit        truncated with bootstrap
opponent collision does not terminate ego
```

## 1.4 Scenario queue

使用两个父进程共享全局 queue：

```text
one hard queue shared by all 8 hard logical envs
one ordinary queue shared by all 8 ordinary logical envs
```

每个 queue：

```text
deterministically shuffle full pool
consume every scenario once
only after exhaustion reshuffle
persist order, cursor and cycle
```

禁止：

```text
with replacement
per-env balanced cycle
worker-local sampler
Bernoulli hard/ordinary branch draw
```

## 1.5 Seed 与随机流

唯一正式：

```text
run seed = 0
```

使用 `numpy.random.SeedSequence(0)` 或稳定 hash 派生独立 stream：

```text
hard scenario queue RNG
ordinary scenario queue RNG
each logical environment RNG
action sampling RNG
recurrent minibatch ordering RNG
```

逻辑 env stream 按：

```text
(role, role_index)
```

派生，不按物理 rank 直接 `seed + rank` 派生。

例如：

```text
hard0..hard7
ordinary0..ordinary7
```

改变 H/O 在物理 rank 中的排列时，同一 logical role stream 不得变化。

Critic 网络初始化不得改变 action exploration RNG。

## 1.6 Rollout

```text
n_envs                  16
n_steps per env         6400
transitions/update      102400
```

## 1.7 Minibatch

```text
valid batch size        12800
logical minibatches     8 per epoch
custom stratified sampler = false
```

依赖：

```text
interleaved H/O ranks
+
env-major recurrent generator
+
batch size = 2 × n_steps
```

每个 logical minibatch 应为：

```text
6400 hard valid transitions
6400 ordinary valid transitions
```

允许 recurrent padding，但 valid H ratio必须在 45%–55%。

## 1.8 Actor

```text
trainable:
  GRU
  output head

frozen:
  k
  speed_mlp
  dummy_embedding
  log_std
```

LR：

```text
GRU         1e-6
output head 1e-5
```

Actor phase：

```text
3 epochs
8 minibatches/epoch
24 optimizer steps/update
```

Actor 与 critic 使用独立 optimizer。

## 1.9 PPO / GAE

```text
gamma                   0.999
gae_lambda              0.995
clip_range              0.10
clip_range_vf           None
normalize_advantage     True
vf_coef                 0.5
ent_coef                0.0
actor max_grad_norm     0.5
critic max_grad_norm    0.5
target_kl               None
use_sde                 False
```

不得修改：

```text
GAE
return target
timeout bootstrap
advantage normalization formula
termination semantics
```

## 1.10 Reward

```text
0.01 × ego progress delta
+ 0.02 × relative track progress delta
- 2.0 × first ego collision
```

删除并禁止：

```text
margin
TTC reward
clearance reward
proximity reward
steering reward
terminal overtake reward
reward redistribution
```

## 1.11 Exploration

```text
steering:
  squashed latent Gaussian
  bound = 0.52 rad
  latent std = 0.03

speed:
  physical Gaussian
  std = 0.15 m/s

temporal process:
  iid at 100 Hz

log_std:
  frozen

entropy:
  0
```

禁止：

```text
physical steering Gaussian + env clipping
AR(1)
finite sustained exploration
P2 pulse
action hold
LiDAR noise
speed-observation noise
dynamic std
```

## 1.12 TF32

在任何 CUDA model 构建或 GRU 调用前：

```python
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.benchmark = False
```

Actor、critic、buffer、loss、gradient、optimizer全部 FP32。

---

# 2. 固定 critic 优化协议

## 2.1 独立 optimizer

```text
actor optimizer:
  GRU + output head only

critic optimizer:
  active critic parameters only
  LR = 3e-4
```

禁止单一 optimizer 同时包含 actor 和 critic。

Actor phase：

```text
critic requires_grad = false
critic optimizer state不变化
```

Critic phase：

```text
actor requires_grad = false
actor optimizer state不变化
```

## 2.2 Warm-up

正式 Update 1 前执行一次。

```text
warm-up rollouts       1
warm-up transitions    102400
actor optimizer steps  0
critic max epochs      20
patience               3
train/validation       80/20
split unit             complete recurrent sequence
stratify               hard/ordinary
restore best critic    yes
restore matching critic optimizer state yes
discard W0 for actor   yes
```

固定流程：

```text
BC actor + initial critic
→ collect W0
→ actor frozen
→ sequence-level H/O stratified split
→ critic train max20
→ validation 3 epochs no improvement则stop
→ restore validation best critic+optimizer
→ discard W0
→ collect fresh W1
→ formal Update 1
```

不要暴露 `min_delta` 配置。使用实现所需的固定浮点比较即可。

## 2.3 Formal update

每个 outer update：

```text
1. collect D_k with current actor and critic
2. compute old logp, value, returns, GAE once
3. freeze critic
4. actor train 3 epochs × 8 minibatches = 24 steps
5. freeze actor
6. critic train 8 epochs × 8 minibatches = 64 steps
7. do not recompute D_k advantages
8. next rollout uses updated actor and critic
```

Critic phase：

```text
epochs = 8
steps = 64/update
LR = 3e-4
grad clip = 0.5
```

禁止：

```text
single shared n_epochs
actor→critic→recompute advantage→actor again
critic 20/50 epochs each update
```

---

# 3. Critic 源码实验规则

## 3.1 不保留 profile

删除：

```text
CRITIC_PROFILES
critic_profile argument
C0/C1/C2/C3 branches
dict observation profile dispatch
runtime critic selection
```

## 3.2 Active critic 源码位置

建议把当前唯一 critic 实现集中在：

```text
ppo/policy.py
```

使用直接结构：

```python
class Critic(nn.Module):
    ...

def critic_input(...):
    ...

def critic_values(...):
    ...
```

不要创建：

```text
ppo/critics/
critic_factory.py
registry.py
base_critic.py
```

## 3.3 每次实验直接修改

### Raw single-frame

只修改：

```text
Critic
critic_input
```

### Detached actor hidden

修改：

```text
Critic
critic_input
actor replay返回所需 hidden feature
buffer接口（若需要）
```

### Independent recurrent critic

修改：

```text
Critic
critic recurrent state
rollout buffer
collection value state
timeout bootstrap
training replay
full-state save/resume
```

不得假装 actor-h-only buffer仍适用。

### Privileged physical critic

修改：

```text
environment current pre-action feature
observation space
critic_input
Critic
buffer observation
```

Actor仍只能读取原361D actor observation。

## 3.4 一次只存在一个 critic

完成某个 critic 后，不把上一个 critic保留为：

```text
else branch
legacy class
commented code
selector option
```

Git diff 和 Git history负责回溯。

## 3.5 每个 critic run 的 provenance

每个 run 输出目录必须保存：

```text
SOURCE_HASHES.json
SOURCE_DIFF.patch
RUNTIME.json
TRAINING_CONTRACT.json
```

至少 hash：

```text
HEAD
model.py
train_ppo.py
ppo/policy.py
ppo/buffer.py
ppo/environment.py
ppo/scenarios.py
ppo/vec_env.py
ppo/reward.py
pretrained/end2race.pth
h1_expanded_det.json
```

`SOURCE_DIFF.patch`：

```bash
git diff --no-ext-diff --binary --full-index
```

即使不 commit，也能精确重建 critic 实现。

---

# 4. 清理后的源码结构

```text
End2Race/
├── pretrained/
│   └── end2race.pth
├── ppo/
│   ├── __init__.py
│   ├── buffer.py
│   ├── config.py
│   ├── environment.py
│   ├── policy.py
│   ├── reward.py
│   ├── scenarios.py
│   ├── vec_env.py
│   └── hard_pools/
│       └── h1_expanded_det.json
├── f1tenth_gym/
├── f1tenth_racetracks/
├── latticeplanner/
├── model.py
├── train.py
├── train_ppo.py
├── demonstration.py
├── collect.sh
├── evaluate.sh
├── eval_singleagent.py
├── eval_multiagent.py
├── utils.py
├── PPO_HISTORY.md
├── README.md
├── install.sh
└── LICENSE
```

删除：

```text
tests/
scripts/
runs/
ppo_experiments/
posttrained/
eval_results/
```

---

# 5. 各文件最终职责

## 5.1 `ppo/config.py`

不再有 dataclass、replace、CONFIGS 或 config validation。

只保存固定常量：

```text
PROJECT_ROOT

N_ENVS = 16
ENV_WORKERS = 6
ENV_START_METHOD = "forkserver"

RUN_SEED = 0

N_STEPS = 6400
BATCH_SIZE = 12800
ACTOR_EPOCHS = 3
CRITIC_EPOCHS = 8

GAMMA = 0.999
GAE_LAMBDA = 0.995
CLIP_RANGE = 0.10
CLIP_RANGE_VF = None
NORMALIZE_ADVANTAGE = True
VF_COEF = 0.5
ENT_COEF = 0.0
ACTOR_MAX_GRAD_NORM = 0.5
CRITIC_MAX_GRAD_NORM = 0.5
TARGET_KL = None

GRU_LR = 1e-6
HEAD_LR = 1e-5
CRITIC_LR = 3e-4

STEERING_BOUND = 0.52
STEERING_LATENT_STD = 0.03
SPEED_PHYSICAL_STD = 0.15

SIM_DURATION = 8.0

HARD_POOL_PATH
HARD_POOL_SIZE = 482
ORDINARY_POOL_SIZE = 600

WARMUP_MAX_EPOCHS = 20
WARMUP_PATIENCE = 3
WARMUP_TRAIN_FRACTION = 0.8

FORMAL_UPDATES
CHECKPOINT_UPDATES
```

`FORMAL_UPDATES` 和 `CHECKPOINT_UPDATES` 不是 CLI 实验参数。
每个 critic 实验需要不同预算时直接修改源码并记录 diff。

## 5.2 `ppo/scenarios.py`

只保留：

```text
ordinary 600 panel
H1 manifest loading
strict ScenarioSpec
hard queue
ordinary queue
queue state_dict/load_state_dict
EpisodeResetSpec creation
```

删除：

```text
evaluation panel generation
H0/H2/H3
variant manifests
paired hard sampling
per-env cycle
with-replacement
probability mixture
fallback
```

新增一个简单 queue 类：

```text
RoleScenarioQueue
```

职责：

```text
pool
order
cursor
cycle
rng
next()
state_dict()
load_state_dict()
```

## 5.3 `ppo/vec_env.py`

父进程持有：

```text
hard RoleScenarioQueue
ordinary RoleScenarioQueue
logical env RNG states
```

`_role(rank)`：

```python
return "hard" if rank % 2 == 0 else "ordinary"
```

logical role index：

```python
role_index = rank // 2
```

worker仍不持有 sampler。

保留：

```text
6 workers
forkserver
thread limit
rank-order reconstruction
terminal observation
auto reset
normal close
exception propagation
```

删除所有旧 paired metadata 和实验 runtime telemetry。

## 5.4 `ppo/environment.py`

保留：

```text
EpisodeResetSpec
external reset option
planner cache
one opponent controller
361D actor observation
previous measured speed timing
ego collision
timeout truncation
reward
```

初始 C0 baseline删除：

```text
privileged critic features
Dict observation
paired metadata
policy update index
multiple opponents generic path
planner factory injection
```

以后测试 privileged critic时直接修改此文件，再保存 diff。

## 5.5 `ppo/reward.py`

只保留三项固定 reward。

删除：

```text
margin_weight
margin_threshold
reward_margin
OBB clearance import
```

## 5.6 `ppo/buffer.py`

C0 起始版本保留 actor-h-only optimized buffer和 Phase 5B validity metadata。

增加当前固定 pipeline需要的最小 metadata：

```text
logical env role per transition / sequence
scenario id或sequence id（仅用于warm-up split和telemetry）
```

不要加入 generic arbitrary metadata registry。

若 active critic是 independent recurrent，再直接修改 buffer存储 critic state。

## 5.7 `ppo/policy.py`

保留：

```text
squashed latent joint distribution
GRU adapter
End2Race actor
batch-1 collection forward
Phase 5B actor training replay
single active Critic
actor optimizer
critic optimizer
12-key actor export
```

删除：

```text
physical Gaussian
critic profiles
backend selector
C1/C2/C3 dead branches
unused actor features
```

提供直接方法：

```text
collect_actor(...)
evaluate_actor_actions(...)
predict_values(...)
evaluate_values(...)
```

Actor phase不运行 critic forward。
Critic phase不运行 actor backward。

## 5.8 `train_ppo.py`

CLI只保留操作参数：

```text
--output_dir
--resume
```

不保留：

```text
--config
--seed
--screen-pause
```

训练流程直接：

```text
assert runtime versions
set fixed numeric contract
seed fixed streams
build queues
build 16-env / 6-worker VecEnv
build actor + active critic
optional resume
if fresh: critic warm-up
for fixed FORMAL_UPDATES:
    collect rollout
    fixed GAE/returns
    actor phase 3 epochs
    critic phase 8 epochs
    save actor checkpoint
    save full state
    write concise telemetry
close env
```

错误直接抛出。只用 `finally` 关闭 VecEnv。

删除：

```text
screen state machine
KL guardrail stop state
paired telemetry
historical parameter-delta audit
old config resolution
DummyVecEnv builder
legacy reset provider
```

---

# 6. Full training state

每个正式 update保存：

```text
training_state_uXXXX.pt
```

内容：

```text
update
num_timesteps

actor state
critic state

actor optimizer state
critic optimizer state

Python RNG
NumPy global RNG
Torch CPU RNG
Torch CUDA RNG

action sampling generator state
minibatch generator state

hard queue order/cursor/cycle/RNG
ordinary queue order/cursor/cycle/RNG
environment RNG states

warm-up completed flag
resolved fixed contract
source hashes
```

同时保存：

```text
actor_uXXXX.pth
```

只含12-key actor。

Resume只接受完整 state，不从 actor-only checkpoint resume。

不存在完整 state或 hash不匹配时直接失败。

---

# 7. 固定 preflight

正式 critic 实验前，每个源码 critic 版本运行一次。

不做五次重复，不做性能 sweep。

## P0 Runtime

检查：

```text
software exact versions
TF32 off
FP32
BC hash
H1 hash/count482
ordinary count600
```

## P1 Queue

无 simulator：

```text
hard前482次无重复
ordinary前600次无重复
下一次进入cycle2
相同state save/load后序列exact
```

## P2 Env roles

```text
rank roles = H,O,H,O,...,H,O
8 hard
8 ordinary
logical stream不依赖physical rank
```

## P3 Minibatch composition

收集一份102400 rollout，不训练。

8个 minibatch都必须：

```text
contains hard and ordinary
valid H ratio 45%–55%
```

失败：

```text
STOP_FIXED_MINIBATCH_CONTRACT_INVALID
```

不自动实现S3，不自动改 batch。

## P4 Warm-up

检查：

```text
actor params bitwise unchanged
actor optimizer steps 0
critic params finite changed
validation split sequence-level
H/O stratified
patience works
best critic+optimizer restored
W0 discarded
```

## P5 Separate optimizers

Actor phase：

```text
actor changes
critic exact unchanged
critic optimizer exact unchanged
```

Critic phase：

```text
critic changes
actor exact unchanged
actor optimizer exact unchanged
```

## P6 Recurrent/action

检查：

```text
pre-action actor hidden
episode reset
padding mask
timeout bootstrap
old logp replay
stored/likelihood/wrapper/F110 action identity
```

## P7 Checkpoint/resume

```text
actor 12 keys
fresh End2Race strict load
full state resume next scenario/action/minibatch sequence exact
```

任一失败直接停止。

---

# 8. Telemetry

每个 update只记录固定必要量。

## Rollout

```text
transitions
hard/ordinary transitions
completed/partial episodes by role
unique scenarios by role
queue cycle/cursor
episode length p10/p50/p90
collision/follow/overtake by role
rollout time
```

## Actor phase

每 minibatch/epoch：

```text
valid/padding
H/O ratio
policy loss
KL
clip fraction
entropy diagnostic
GRU/head grad norm
clip multiplier
parameter delta
```

验证：

```text
3 epochs
8 batches/epoch
24 steps
```

## Critic phase

每 epoch：

```text
train value loss
validation value loss only during warm-up
explained variance
grad norm
clip multiplier
parameter delta
```

验证：

```text
warm-up actor step = 0
formal 8 epochs
64 steps/update
```

---

# 9. 历史目录清理

## 9.1 删除前归档

生成唯一：

```text
PPO_HISTORY.md
```

记录：

```text
V1/V1.1/V1.2
critic旧screen
H-series
P1–P4
conditional H1/H2
Phase 1–5B性能优化
关键commit
关键数值
删除文件数量/大小
```

## 9.2 删除

```text
tests/
scripts/
runs/
ppo_experiments/
posttrained/
eval_results/
```

不创建 archive 替代目录。

## 9.3 临时验证

所有实现测试放：

```text
/tmp/end2race_fixed_pipeline_validation/
```

最终不提交。

---

# 10. README 与 `.gitignore`

## README

PPO命令改为：

```bash
python train_ppo.py \
  --output_dir runs/ppo/c0_raw
```

resume：

```bash
python train_ppo.py \
  --output_dir runs/ppo/c0_raw \
  --resume runs/ppo/c0_raw/training_state_uXXXX.pt
```

说明：

```text
所有训练参数硬编码
critic实验直接修改源码
每个run保存source hash和diff
```

## `.gitignore`

```gitignore
runs/
eval_results/
posttrained/
ppo_experiments/
tests/
scripts/

__pycache__/
*.py[cod]
.pytest_cache/
*.patch
```

注意：若项目其他非PPO功能仍有正式 `scripts/`，不要全局ignore或删除。
先确认当前目录内容只是一性实验脚本。

---

# 11. 执行顺序

## Stage 1

```bash
git switch main
git pull --ff-only
git status --short
git rev-parse HEAD
```

要求基线一致且 tracked clean。

## Stage 2 Inventory

记录：

```text
tracked files
sizes
hashes
current configs
current critic branches
current scripts/tests/results
```

## Stage 3 Last historical tests

删除 tests前运行一次并归档结果。

## Stage 4 PPO_HISTORY

生成并检查无 TODO/TBD。

## Stage 5 Remove historical directories

禁止 `git clean -fdx`。

## Stage 6 Simplify production

按第5节重构并硬编码固定合同。

## Stage 7 Temporary preflight

在 `/tmp` 写验证脚本，执行 P0–P7。

## Stage 8 One integration run

只运行一次完整：

```text
critic warm-up W0
formal Update 1
```

不运行 full-600，不做多 seed，不做性能重复。

检查：

```text
102400 warm-up transitions
actor warm-up delta 0
fresh W1
24 actor steps
64 critic steps
12-key actor
full-state resume
worker close
no zombie
```

## Stage 9 Final diff

生成：

```text
/tmp/end2race_cleanup/end2race_fixed_pipeline_cleanup.diff
/tmp/end2race_cleanup/CLEANUP_REPORT.md
/tmp/end2race_cleanup/name_status.txt
```

不 commit，不 push。

---

# 12. 最终验收

最终 verdict：

```text
READY_FOR_CLEANUP_REVIEW
```

要求：

```text
all non-critic parameters hardcoded
no PPOConfig/CONFIGS
no --config/--seed
only one active critic
no critic selector
H/O interleaved
global role queues
seed0 fixed streams
n_steps6400
batch12800
actor3 epochs
critic warm-up
critic8 epochs
separate optimizers
reward fixed
exploration fixed
Phase1–5B retained
12-key actor
full-state resume
tests/scripts/results removed
PPO_HISTORY complete
temporary preflight pass
no commit/push
```

否则：

```text
CLEANUP_INCOMPLETE
```

直接报告真实阻断，不增加 fallback。
