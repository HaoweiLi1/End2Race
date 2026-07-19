# End2Race `main` 分支清理与最小化指导文档

**目标仓库：** `HaoweiLi1/End2Race`  
**审查基线：** `main@1d404a412c9faa2a451bb6318848b2d5c349ad97`  
**执行性质：** 本地清理、代码最小化、最终审查；禁止自动 commit、push 或修改远端  
**代码风格参考：** 根目录 `model.py`、`train.py`  
**最终原则：** 只保留当前实际运行的单一 PPO production path；历史实验通过 Git 历史和一份归档 Markdown 保留，不再把实验过程、审计脚本、原始结果和废弃分支留在主代码树中。

---

## 0. 最终决定

### 0.1 可以整体清除的目录

在完成本文规定的一次性归档后，以下目录可以从当前工作树和 Git tracking 中整体删除：

```text
tests/
scripts/
runs/
ppo_experiments/
posttrained/
eval_results/
```

说明：

- `tests/`：当前主要是本轮 PPO 性能与 batched replay 的专项回归测试。它们已经完成验证任务；若 owner 明确要求主仓库保持接近原版 `model.py`、`train.py` 的简洁风格，可以删除。删除前必须运行一次并把结果写入归档文档。
- `scripts/`：只删除实验、审计、一次性生成和历史验证脚本。根目录正式入口 `collect.sh`、`evaluate.sh` 不属于该目录，不得删除。
- `runs/`：训练产物，不属于源码。
- `ppo_experiments/`：历史实验记录、raw JSON、patch、profile、reference capture 和性能 artifact，不属于 production runtime。
- `posttrained/`：历史候选 actor checkpoint。当前没有需要保留为正式部署模型的 PPO checkpoint。
- `eval_results/`：评价输出，不属于源码。

### 0.2 必须保留

```text
pretrained/end2race.pth

model.py
train.py
train_ppo.py

ppo/
  __init__.py
  buffer.py
  config.py
  environment.py
  policy.py
  reward.py
  scenarios.py
  vec_env.py
  hard_pools/h1_expanded_det.json

eval_singleagent.py / evaluate_singleagent.py
eval_multiagent.py / evaluate_multiagent.py
evaluate.sh
collect.sh
demonstration.py
utils.py

f1tenth_gym/
f1tenth_racetracks/
latticeplanner/

README.md
install.sh
LICENSE
```

评价脚本的实际文件名以当前仓库为准，不要为了统一命名顺手重命名。

### 0.3 只保留一个历史归档文件

在删除 `ppo_experiments/`、`runs/`、`posttrained/` 和 `eval_results/` 前，生成：

```text
PPO_HISTORY.md
```

它是主仓库中唯一保留的 PPO 历史与性能记录。不得再保留几十个 JSON、raw trace、patch、profile、checkpoint 或实验子目录。

---

# 1. 代码风格要求

参考原版 `model.py` 和 `train.py`：

1. 一个函数只完成一个直接任务。
2. 控制流从上到下可读，不建立 registry、dispatcher、plugin、fallback chain。
3. 不为已经固定的 production path保留通用扩展接口。
4. 不添加“未来可能会用”的配置、分支、兼容层和 telemetry。
5. 输入错误直接抛出异常。
6. 不捕获异常后自动选择另一条路径。
7. 不自动修复缺失文件、错误 manifest、非法配置或输出目录冲突。
8. 不使用广泛的 `except BaseException` 掩盖 production error；仅允许 worker finally 中关闭资源。
9. 不修改 site-packages。
10. 不把测试 helper、profile hook、shadow backend 或实验 oracle 放进 production module。
11. 不保留 A、C、A+B、A+C 或旧 batch-1 training replay 的 production selector。
12. 当前正式路径只有：

```text
Phase 1  parent-scheduled 6-worker subprocess VecEnv
Phase 2  immutable planner asset cache
Phase 3  C0 actor-h-only rollout buffer
Phase 4  invalid padding skip
Phase 5B timestep active-slot batched FP32 training replay
```

Collection 始终保留 batch-size-one actor forward。

---

# 2. 清理后的目标仓库结构

目标不是强制完全等于下方结构，但生产顶层应接近：

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
├── pretrained/
├── PPO_HISTORY.md
├── README.md
├── install.sh
└── LICENSE
```

不要新建：

```text
docs/
archive/
legacy/
experimental/
backends/
helpers/
common/
compat/
```

只为了安放被删除代码而创建新目录，等价于没有清理。

---

# 3. 一次性归档要求：`PPO_HISTORY.md`

删除任何历史目录前，coding agent 必须从当前 main、Git history 和现有 artifacts 汇总一份完整但紧凑的 Markdown。

## 3.1 必须记录的 commit 边界

至少记录：

```text
ef8570bc522fb3b4dc6df2636bbe3a6e9afc4da4
  Finalize H1 H2 conditional exploration experiment

e21b8e71ff148a20079b22e78f3bd61c886dc35b
  Phase 1: central subprocess environments

261e6ac6db2db21dd64d94fc09beed5774058ab2
  Phase 2: planner static asset cache

94cef72b830354cb6c56e1da0f8493bbd65eb35f
  Phase 3: actor-h-only rollout buffer

ff5aaa735e0fdaa6e40cc784643550d570c30e70
  Phase 4: invalid-padding actor skip

e1c0d2b61e4ebc5c619f4c013dad330acf1fdfa0
  performance/equivalence evidence

1d404a412c9faa2a451bb6318848b2d5c349ad97
  Phase 1–4 + Phase 5B default production pipeline
```

## 3.2 必须记录的实验历史

不要复制 raw JSON。用表格总结：

### BC 与早期 PPO

```text
Canonical BC panel
V1 checkpoints
V1.1 checkpoints
V1.1 selected U2
```

至少记录 collision/follow/overtake 和 fixed/new collision。

### Critic

记录 C0/C1/C2/C3：

```text
结构
训练协议边界
selected checkpoint
collision
overtake
最终结论
```

### Pool / H-series

记录：

```text
H0
H1
H2 core
H3 core
pool size
U1/U2/U4 trajectory
最终解释
```

### Batch / rollout / LR / KL / exploration

记录已经实际完成的关键结果，不记录没有运行的 arm。

### P1–P4 与 H1/H2 conditional

记录：

```text
H1 direction present
P2 actionability
P3 current credit retained
P4 update geometry未解决产品问题
H1 full p50 2/3 seed forward signal
H2 matched pool too small，未运行训练
```

### 性能优化

记录完整表：

```text
Original baseline
Phase 1
Phase 2
Phase 3
Phase 4
Phase 5B final
```

至少包括：

```text
rollout seconds
train seconds
total update seconds
transitions/s
buffer MiB
PSS/RSS
```

### Phase 5

明确记录：

```text
A:
实现正确，但会立即改变 closed-loop collection trajectory，
不进入 production。

B:
training-only timestep batching，
成为 production default。

C:
速度最快但数值误差高于 B，
不进入 production。
```

B 的冻结数值指标必须写入：

```text
policy KL                    4.56e-11
gradient cosine              0.9999999995
gradient relative L2         3.24e-5
parameter-delta cosine       0.99999999996
parameter-delta relative L2  9.37e-6
policy-loss difference       2.68e-7
```

## 3.3 最终 production 合同

归档末尾明确写：

```text
16 logical environments
6 forkserver workers
parent-only scenario scheduling
planner asset cache
C0 actor-h-only buffer
collection batch-1
training replay B
FP32
cuDNN TF32 off
CUDA matmul TF32 off
float32 matmul precision highest
cuDNN benchmark off
12-key actor checkpoint
```

## 3.4 删除清单

归档中加入：

```text
Deleted tracked files
Deleted generated directories
Deleted checkpoints
Deleted evaluation outputs
Deleted experiment scripts
Deleted tests
```

记录文件数量和总字节数即可。不要把每个大型 JSON 的全文放进 Markdown。

## 3.5 关键文件哈希

删除前记录：

```text
pretrained/end2race.pth SHA-256
h1_expanded_det.json SHA-256
main HEAD
model.py SHA-256
train_ppo.py SHA-256
ppo/policy.py SHA-256
ppo/buffer.py SHA-256
ppo/environment.py SHA-256
ppo/vec_env.py SHA-256
```

---

# 4. 目录级删除指导

## 4.1 `tests/`

### 决定

删除整个目录。

### 删除前

运行一次：

```bash
python -m unittest discover -s tests -v
```

把：

```text
总测试数
通过数
失败数
运行命令
HEAD
```

写入 `PPO_HISTORY.md`。

### 注意

不要把测试 helper 移入 `ppo/`。  
不要为了“保留一点测试能力”创建 `ppo/test_utils.py`。  
Git history 已经保留完整测试。

---

## 4.2 `scripts/`

### 决定

删除整个目录，前提是它只包含：

```text
one-off audit
profile
hard-pool generator
experiment runner
replay validator
smoke
result aggregator
cleanup helper
```

### 删除前检查

```bash
git grep -n "scripts/" -- \
  ':!scripts/**' \
  ':!PPO_HISTORY.md'
```

任何生产代码 import `scripts.*` 都是需要先解除的错误依赖。

根目录正式文件不得删除：

```text
collect.sh
evaluate.sh
demonstration.py
```

---

## 4.3 `runs/`

删除整个目录及所有 checkpoint、metrics、status、manifest 和 sampler summary。

以后 `runs/` 只作为本地生成目录，由 `.gitignore` 忽略。

---

## 4.4 `ppo_experiments/`

生成 `PPO_HISTORY.md` 后删除整个目录。

必须删除：

```text
raw JSON
reference captures
performance profiles
patch copies
npy action dumps
full600 rows
replay scripts
integration reports
validation reports
old plans
stage selections
experiment matrices
```

不要保留：

```text
ppo_experiments/archive/
ppo_experiments/final/
ppo_experiments/selected/
```

删除整个目录才是真正清理。

---

## 4.5 `posttrained/`

删除全部历史 PPO actor checkpoint。

保留：

```text
pretrained/end2race.pth
```

以后新 PPO checkpoint 输出到 ignored `runs/`，通过明确人工选择后再决定是否单独发布，不提前建立 tracked `posttrained/`。

---

## 4.6 `eval_results/`

删除全部结果、trace、video 和 worker temporary files。

以后由 `.gitignore` 忽略。

---

# 5. 生产代码冗余审查与删除要求

以下项目是当前 main 中确认存在或高度可能残留的历史实验功能。coding agent 必须逐项使用 `git grep` 确认，并删除不再被唯一默认路径使用的实现。

---

## 5.1 `ppo/config.py`

当前文件包含大量 V1/V1.1/V1.2/Stage H/K/E/SG/V1.3/QP3/N1/N2 历史 profile。

### 删除

删除所有历史配置，只保留当前唯一默认：

```text
N1-H1F-p50
```

删除关联的：

```text
V1
V1_1
V1_2_H0_CONTROL
AH_*
AP_*
AB_*
AR_*
AK_*
BE_*
SG_*
V1_3_*
QP3_*
N1_H1F_P25
N1_H1E_*
N2_*
```

`CONFIGS` 中只保留一个 production config。

### 删除废弃字段

在唯一配置和 runtime 中无使用时删除：

```text
paired_hard_sampling
hard_pair_size
hard_pool_manifest
margin_weight
margin_threshold
steering_distribution
update_kl_guardrail
autograd_multithreading
```

`target_kl` 若唯一值永远为 `None`，直接使用常量，不需要 profile 字段。

`critic_profile` 若 production 只允许 C0，删除配置字段，policy 固定 C0。

### 保留

```text
N_ENVS = 16
ENV_WORKERS = 6
ENV_START_METHOD = "forkserver"

n_steps
batch_size
updates
checkpoint_updates

GRU LR
head LR
critic LR

gamma
GAE lambda
clip
vf coef
max grad norm

steering latent std = 0.03
speed physical std = 0.15

hard pool = H1 expanded deterministic
fixed hard env count = 8
8-second horizon
```

不要在清理时改变这些数值。

---

## 5.2 `ppo/policy.py`

### 删除旧 critic branches

Production 只保留 C0。

删除：

```text
CRITIC_PROFILES
C1_FROZEN_BC_FEATURE
C2_DETACHED_ACTOR_HIDDEN
C3_PRIVILEGED_PHYSICAL
CombinedExtractor
dict observation handling
detached actor feature critic
privileged critic branch
```

将 value path 简化为：

```text
361D actor observation
→ current C0 value_net
→ scalar value
```

### 删除 `physical_gaussian`

删除：

```text
EvaluatorClippedPhysicalGaussianDistribution
steering_distribution selector
physical_gaussian branch
V1_3_C / V1_3_D dependencies
```

只保留已经验证的 squashed latent steering distribution。

### 删除不再需要的 actor feature 输出

`actor_features` 仅为 C2 critic 服务。

删除 C2 后：

```text
_actor_forward()
_actor_replay_batched()
```

无需构造、stack 或返回每 timestep hidden feature。

简化为：

```text
means
next recurrent state
```

这也会减少无意义的 tensor stack/reshape。

### 保留

```text
EvaluatorCompatibleJointDistribution
GRUWithLSTMStateInterface
End2RaceGRUPolicy
collection batch-1 _actor_forward()
training _actor_replay_batched()
12-key actor export
End2RaceRecurrentPPO
actor-h-only rollout buffer setup
fixed optimizer groups
```

### 不保留 selector

禁止：

```text
training_backend
collection_backend
exact_backend
batched_backend
A/B/C selector
```

B 就是唯一 training replay。

---

## 5.3 `ppo/environment.py`

### 删除 privileged critic

删除：

```text
PrivilegedFeatureManifest
AustinPrivilegedFeatureExtractor
oriented_rectangle_clearance
rectangle geometry helpers
privileged_critic flag
privileged_feature_extractor
Dict observation space
```

如果 `oriented_rectangle_clearance` 仍被其他 production 文件引用，先删除对应历史 margin/privileged 功能，而不是保留 utility。

### 删除历史 pair metadata

删除：

```text
pair_group
pair_member
pair_episode_ordinal
policy_update_index
set_policy_update_index()
```

它们属于旧 paired H2 实验和 telemetry。

### 简化 opponent

当前正式环境固定：

```text
2 agents
ego index 0
one opponent index 1
```

删除仅为泛化测试存在的：

```text
multiple opponent loops
plural opponent_racelines
plural opponent_speed_scales
_per_opponent_value()
planner_factory injection
```

直接读取：

```text
scenario["opp_raceline"]
scenario["opp_speedscale"]
```

### 保留

```text
EpisodeResetSpec
EXTERNAL_RESET_OPTION
LatticePlannerOpponentController
planner template cache
End2RaceGymnasiumEnv
previous-speed timing
ego-only collision
timeout truncation
transition reward
```

不要改变 F110 action、reset 或 termination 语义。

---

## 5.4 `ppo/reward.py`

当前正式 reward 是三项：

```text
progress
relative progress
first ego collision
```

### 删除 margin reward

删除：

```text
reward_margin
margin_weight
margin_threshold
oriented rectangle clearance import
margin branch
```

`RewardResult` 和训练 callback 不再记录 `reward_margin`。

### 保留

```text
ProgressProjector
wrapped_progress_delta
checked_progress_delta
reward_progress
reward_relative
reward_collision
reward_total
opponent collision latch
```

---

## 5.5 `ppo/scenarios.py`

### 删除评价 pool代码

若 `evaluation_scenarios()` 没有 production import，删除：

```text
EVALUATION_STARTPOINTS
evaluation_scenarios()
pool == "evaluation" branch
```

评价系统应继续使用现有 evaluator 自己的权威 panel，不由 PPO trainer重复生成。

### 删除 paired sampling

删除：

```text
_sample_paired_hard()
pair_seed
pair_group
pair_member
pair_episode_ordinal
paired reset validation
```

### 删除不用的 sampler mode

若唯一 default 是 `with_replacement`，删除：

```text
per_env_balanced_cycle
_cycles
hard_sampling_mode selector
```

若 owner 明确还要保留 balanced-cycle 作为正式未来功能，则不要在本轮猜测；当前清理默认按唯一 production path删除。

### 简化 hard pool loader

只保留当前实际读取：

```text
ppo/hard_pools/h1_expanded_det.json
```

删除：

```text
variant manifest
旧 H0/H2/H3 fallback
自动选择 variant
旧 manifest compatibility
```

manifest schema 不符时直接报错。

### 删除旧 hard pool文件

在 `ppo/hard_pools/` 中只保留：

```text
h1_expanded_det.json
```

删除所有：

```text
h0*
h1_early*
h2*
h3*
matched*
conditional*
```

---

## 5.6 `ppo/buffer.py`

这是当前核心优化，不做风格性重写。

保留：

```text
ActorHiddenRolloutBuffer
actor h-only storage
stock-compatible sequence generation
current_valid_by_timestep
lazy zero transport states
```

只删除：

```text
unused imports
dead comments
不再存在 critic profile 的说明
```

不要因为追求短代码而改写 sequencer 或 validity 计算。

---

## 5.7 `ppo/vec_env.py`

保留 Phase 1 的 verified core，不做大规模风格重写。

### 保留

```text
CentralEpisodeScheduler
CentralScheduleSubprocVecEnv
parent-only reset scheduling
rank reconstruction
terminal_observation
worker exception propagation
normal close/join
required VecEnv methods
```

### 删除

确认没有使用后删除：

```text
paired episode ordinal
paired sampler branch
experiment timing command
runtime telemetry command
profile-only IPC fields
debug histories beyond minimal sampler accounting
```

不要重新加入：

```text
shared memory
自动 worker count
spawn fallback
DummyVecEnv fallback
```

当前 Linux production path 固定 `forkserver`，错误直接暴露。

---

## 5.8 `train_ppo.py`

当前文件仍混合了正式训练与大量历史审计/实验控制。

### 删除 CLI 与状态机

删除：

```text
--screen-pause
input() continue/stop
PAUSED_SCREEN
STOPPED_SCREEN
STOPPED_KL_GUARDRAIL
复杂 run_status 状态机
write_json_atomic()
```

保留正常 Python exception。失败时进程非零退出即可。

### 删除 paired telemetry

删除 callback 中：

```text
paired_completed
paired_cross_update_keys
_paired_telemetry()
pair return difference
pair collision-time difference
```

### 删除历史参数审计 telemetry

production 不需要每个 update 重算全模型相对 BC 的详细 RMS：

```text
_parameter_delta_statistics
_parameter_previous_delta_statistics
_actor_delta_record
```

删除关联输出。

冻结参数由 `requires_grad=False` 和一次性验证保证，不必每次 update 遍历全部参数。

### 删除 optimizer state introspection

若它只用于历史证据，删除：

```text
_optimizer_step()
planned vs actual optimizer state consistency审计
```

SB3 正常 train failure会直接抛出。保留 logger 中的 update、loss、KL 即可。

### 简化 callback

每个 update只记录：

```text
update
num_timesteps
completed ego_collision/follow/overtake
hard/ordinary transition count
unique scenario count
reward component means
action mean/std/min/max
loss
policy loss
value loss
approx KL
clip fraction
explained variance
```

不要记录只有旧 paired/H2 实验使用的字段。

### 删除旧 environment builder

删除当前不再使用的：

```text
make_training_env()
_fixed_reset_provider()
Dummy/single-process production construction
```

只保留：

```text
make_subprocess_training_env()
build_training_vector_env()
```

### 保留

```text
TF32-off initialization
seed
build sampler
build model
build 6-worker VecEnv
train loop
save 12-key actor
resolved config
checkpoint manifest
concise metrics
sampler summary
finally close VecEnv
```

### CLI 建议

保留：

```text
--config N1-H1F-p50
--seed
--output_dir
```

为了不破坏现有调用，`--config` 暂时保留，但唯一 choice 只有 `N1-H1F-p50`。

不要在清理提交中顺手重设计 CLI。

---

# 6. README 与 `.gitignore`

## 6.1 README

删除：

```text
posttrained/ 作为 tracked selected checkpoint 的描述
ppo_experiments/ 作为 tracked evidence 的描述
历史 config 的运行示例
旧测试/脚本入口
```

PPO 部分只保留：

```bash
python train_ppo.py \
  --config N1-H1F-p50 \
  --seed 20260715
```

说明：

```text
runs/ 是本地输出
评价单独运行
最终 actor 是12-key checkpoint
```

README 不写性能研究长报告；详细历史只链接 `PPO_HISTORY.md`。

## 6.2 `.gitignore`

加入或确认：

```gitignore
runs/
eval_results/
posttrained/
ppo_experiments/

__pycache__/
*.py[cod]
.pytest_cache/

*.patch
```

不要忽略：

```text
pretrained/
ppo/hard_pools/h1_expanded_det.json
PPO_HISTORY.md
```

---

# 7. 执行顺序

## Stage 1：冻结当前状态

```bash
git switch main
git pull --ff-only
git status --short
git rev-parse HEAD
```

要求：

```text
HEAD = 1d404a412c9faa2a451bb6318848b2d5c349ad97
tracked worktree clean
```

若 main 已前进，停止并重新 review，不要基于旧清单直接删除。

不要自动 stash、reset 或 clean 用户文件。

---

## Stage 2：生成删除前 inventory

```bash
mkdir -p /tmp/end2race_cleanup

git ls-files \
  tests \
  scripts \
  runs \
  ppo_experiments \
  posttrained \
  eval_results \
  > /tmp/end2race_cleanup/tracked_delete_candidates.txt

git ls-files > /tmp/end2race_cleanup/tracked_before.txt

du -sh \
  tests \
  scripts \
  runs \
  ppo_experiments \
  posttrained \
  eval_results \
  2>/dev/null \
  > /tmp/end2race_cleanup/size_before.txt
```

记录关键哈希：

```bash
sha256sum \
  pretrained/end2race.pth \
  ppo/hard_pools/h1_expanded_det.json \
  model.py \
  train_ppo.py \
  ppo/policy.py \
  ppo/buffer.py \
  ppo/environment.py \
  ppo/vec_env.py \
  > /tmp/end2race_cleanup/key_hashes_before.txt
```

---

## Stage 3：运行最后一次旧测试

```bash
python -m unittest discover -s tests -v \
  2>&1 | tee /tmp/end2race_cleanup/tests_before_delete.log
```

失败直接停止。不要修改测试预期来通过。

---

## Stage 4：生成 `PPO_HISTORY.md`

根据第 3 节要求生成文档。

生成后人工检查：

```bash
grep -n "TODO\|TBD\|unknown\|待补" PPO_HISTORY.md
```

归档中不得有未解决占位符。

---

## Stage 5：删除历史目录

使用 Git 感知删除 tracked 文件，并删除 untracked generated output。

目标结果：

```text
tests/        absent
scripts/      absent
runs/         absent
ppo_experiments/ absent
posttrained/  absent
eval_results/ absent
```

不要使用：

```bash
git clean -fdx
```

它可能删除数据集、地图、未跟踪研究资产和用户文件。

---

## Stage 6：production dead-code cleanup

按第 5 节逐文件处理。

每删除一个 feature，立即运行：

```bash
git grep -n "<deleted symbol>"
```

结果必须为空，不能留下：

```text
config字段
import
README示例
JSON schema
callback telemetry
manifest key
```

关键 symbol 至少检查：

```text
C1_FROZEN_BC_FEATURE
C2_DETACHED_ACTOR_HIDDEN
C3_PRIVILEGED_PHYSICAL
physical_gaussian
EvaluatorClippedPhysicalGaussianDistribution
paired_hard_sampling
pair_group
pair_member
pair_episode_ordinal
reward_margin
margin_weight
margin_threshold
screen_pause
PAUSED_SCREEN
STOPPED_SCREEN
update_kl_guardrail
```

---

## Stage 7：静态验证

```bash
python -m compileall \
  model.py \
  train.py \
  train_ppo.py \
  ppo

python -c "import ppo.buffer, ppo.config, ppo.environment, ppo.policy, ppo.reward, ppo.scenarios, ppo.vec_env"

python train_ppo.py --help

bash -n collect.sh
bash -n evaluate.sh

git diff --check
```

任何错误直接修正真实代码，不添加 fallback。

---

## Stage 8：一次最小生产集成运行

由于本轮会修改 production code，而 tests 已删除，必须运行一次真实训练集成检查。

只运行一次，不做性能 benchmark，不做五次重复，不做 full-600，不做多 seed。

使用唯一 default config，输出到临时目录：

```bash
python train_ppo.py \
  --config N1-H1F-p50 \
  --seed 20260917 \
  --output_dir /tmp/end2race_cleanup_smoke
```

检查：

```text
run completed
finite metrics
checkpoint written
checkpoint exactly 12 keys
fresh End2Race strict load PASS
worker normal close
no zombie
BC checkpoint hash unchanged
```

完成后：

```bash
rm -rf /tmp/end2race_cleanup_smoke
```

不把 smoke output 放回仓库。

---

## Stage 9：最终文件树与依赖审计

```bash
git ls-files | sort > /tmp/end2race_cleanup/tracked_after.txt

git grep -n \
  -e "ppo_experiments/" \
  -e "posttrained/" \
  -e "eval_results/" \
  -e "tests/" \
  -e "scripts/"
```

只允许 `.gitignore` 和 `PPO_HISTORY.md` 中出现历史目录名称。

检查 production import graph：

```bash
python - <<'PY'
import ast
from pathlib import Path

for path in [Path("train_ppo.py"), *sorted(Path("ppo").glob("*.py"))]:
    tree = ast.parse(path.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    print(path)
    for name in sorted(filter(None, imports)):
        print(" ", name)
PY
```

不得出现对：

```text
tests
scripts
ppo_experiments
runs
posttrained
eval_results
```

的 import。

---

# 8. 最终交付物

coding agent 只返回本地文件，不 commit，不 push。

## 8.1 清理报告

生成：

```text
/tmp/end2race_cleanup/CLEANUP_REPORT.md
```

必须包含：

```text
before HEAD
before tracked file count
before repo size
deleted directories
deleted file count
deleted byte count
production files modified
dead symbols removed
remaining production tree
compile/import result
one integration smoke result
checkpoint strict-load result
worker cleanup result
git status
```

## 8.2 完整 diff

```bash
git diff \
  --no-ext-diff \
  --binary \
  --full-index \
  --unified=120 \
  > /tmp/end2race_cleanup/end2race_cleanup.diff
```

## 8.3 删除文件列表

```bash
git diff --name-status \
  > /tmp/end2race_cleanup/name_status.txt
```

## 8.4 最终状态

```bash
git status --short
git diff --check
wc -l /tmp/end2race_cleanup/end2race_cleanup.diff
sha256sum /tmp/end2race_cleanup/end2race_cleanup.diff
```

返回：

```text
PPO_HISTORY.md
/tmp/end2race_cleanup/CLEANUP_REPORT.md
/tmp/end2race_cleanup/end2race_cleanup.diff
/tmp/end2race_cleanup/name_status.txt
/tmp/end2race_cleanup/key_hashes_before.txt
```

---

# 9. 禁止事项

1. 不 commit。
2. 不 push。
3. 不 force reset。
4. 不自动 stash。
5. 不运行 `git clean -fdx`。
6. 不删除 `pretrained/end2race.pth`。
7. 不删除 H1 active manifest。
8. 不修改 reward、GAE、action distribution、termination、scenario mix 或 Phase 1–5B 数学语义。
9. 不重新实现 PPO。
10. 不升级 dependency。
11. 不保留 legacy backend selector。
12. 不创建新的 archive tree。
13. 不将 tests/scripts 移入 production package。
14. 不新增 fallback、兼容层或 broad exception handling。
15. 不因为 cleanup 发现旧 artifact 缺失而自动重跑历史实验。
16. 不运行 full-600 或多 seed。
17. 不把清理后的 smoke 结果称为新实验结果。

---

# 10. 最终验收标准

最终 verdict 只能是：

```text
READY_FOR_CLEANUP_REVIEW
```

或：

```text
CLEANUP_INCOMPLETE
```

`READY_FOR_CLEANUP_REVIEW` 要求：

```text
PPO_HISTORY.md完整
历史输出目录全部移除
tests/scripts全部移除
只有一个 production PPO config
只有 C0 critic
只有 squashed latent action distribution
无 margin reward
无 paired H2 功能
无 A/C backend
collection仍为batch-1
training replay仍为B
Phase 1–3核心保持
12-key checkpoint保持
compile/import通过
一次真实集成运行通过
无worker/zombie
无commit/push
```

若任一 production 合同无法确认，返回：

```text
CLEANUP_INCOMPLETE
```

并直接列出真实阻断原因，不添加临时 fallback。
