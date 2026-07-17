# End2Race PPO V1.3-B：受控 Actor 更新区间实现与五 Seed 验证指南

**状态：** `READY_FOR_LOCAL_CODEX_IMPLEMENTATION`  
**仓库：** `HaoweiLi1/End2Race`  
**代码审阅边界：** `main@f8af84ed4a9dc8065315230da7cb4ad70ab4c6f0`  
**正式配置名：** `v1_3_b`  
**正式训练 seed：** `20260723, 20260724, 20260725, 20260726, 20260727`  
**唯一正式判定 checkpoint：** 每个 seed 的 `U8`  
**开发评价面板：** Austin canonical development 600，`EGO_IDX_OFFSET=0`  
**本文件用途：** 指导本地 Codex 以最小代码改动实现、验证和执行 V1.3-B。不要把它扩展成新的参数 sweep、critic 项目或 reward 项目。

---

# 1. 执行结论先行

V1.3-B 只回答一个问题：

> 在保持 actor LR、critic、reward、hard pool、batch、rollout 和 exploration 全部不变时，把同一批 on-policy rollout 的 PPO 数据复用从 `n_epochs=1` 增加到 `n_epochs=4`，能否形成受控、非偶然、跨五个训练 seed 一致的 actor 改善？

V1.3-B **不是**以下任务：

- 不是重新搜索 critic；
- 不是重新搜索 hard pool、batch size 或 exploration std；
- 不是提高 actor LR；
- 不是修改 reward；
- 不是训练 residual、gate、macro policy 或 safety wrapper；
- 不是从已有 PPO checkpoint 继续训练；
- 不是在 U2、U4、U8 中为每个 seed 选择最好 checkpoint；
- 不是新的 holdout 或最终部署确认。

本实验只有一个训练配置、五个 fresh seed、一个固定最终 checkpoint。若它失败，不能再通过挑选中间 checkpoint 把失败改写为成功。

---

# 2. 为什么这是当前最小且合理的下一步

## 2.1 已确认的事实

当前正式 V1.1/V1.2 训练族主要使用：

```text
GRU LR       = 1e-6
head LR      = 1e-5
n_epochs     = 1
n_steps      = 1600
batch_size   = 1600
```

每个 update：

```text
16 env × 1600 steps = 25,600 transitions
25,600 / 1,600      = 16 minibatches
n_epochs=1          = 16 optimizer steps/update
```

V1.1 已经完成 20 updates、320 optimizer steps、734 个完整 episode，其中有 207 个 ego-collision episode。因此现有负面结果不能解释为“训练中没有 collision 信号”。

V1.1 最终 actor 参数相对 BC 的绝对变化很小，但 deterministic 600-case 结果在不同 checkpoint 间明显翻转。参数空间变化小不等于递归闭环策略的函数变化小。

V1.2 的 46 个去重 development checkpoint 整体仍以 BC 为中心，fixed/new collision 平均近似对称。多 seed 长训练也没有保留早期 `15/353` 峰值，独立 holdout 上该候选反而从 BC 的 `32/332` 退化到 `37/329`。

四种 critic 已经实际比较。更好的 value fitting、actor hidden critic 和 privileged critic 都没有转化为更好的 collision/overtake，因此当前没有依据把 critic 设计列为下一步首要变量。

## 2.2 尚未被干净验证的区间

当前 main 还测试过更激进的配置：

```text
GRU LR  = 1e-5
head LR = 1e-4
n_epochs = 2
```

两个 seed 在 U1 的 approximate KL 已达到约 `0.05` 和 `0.09`，超过 guardrail。

因此已观察到的是：

```text
默认低强度：
    多数 update 移动较小，但产品结果双向 churn

10× LR + 2 epochs：
    U1 即出现过大 KL
```

中间仍未被验证的是：

```text
保持原 LR
只增加 rollout 数据复用次数
由 target_kl 和 post-update guardrail 控制
```

这就是 V1.3-B。它不预设结果一定成功，只用于区分：

1. 原配置是否主要因为 actor 优化强度不足；
2. 或者在 actor 已有足够移动后，advantage 方向仍然跨 seed 不一致。

## 2.3 必须避免的过度表述

执行和报告中不得写：

- “当前 PPO 没有学到任何东西”；
- “模型必须挣脱 BC 才能改善”；
- “n_epochs=4 一定会成功”；
- “critic 已经完美”；
- “五个 seed 通过就证明全局泛化”；
- “21 collision 与 32 collision 是 evaluator 零点冲突”。

正确表述是：

- `21/233/346` 来自 canonical development panel，`EGO_IDX_OFFSET=0`；
- `32/236/332` 来自已使用过的 `holdout33`，是另一组 startpoints；
- V1.3-B 只在 canonical development panel 上做机制验证；
- `holdout33` 已经使用过，本实验禁止再次使用；
- 即使 V1.3-B 通过，也需要以后在一个新的预注册 panel 上做独立确认。

---

# 3. 冻结的训练合同

除 `n_epochs`、`target_kl` 和新增的 post-update KL guardrail 外，以下全部冻结：

```text
actor                  original End2Race GRU actor
trainable actor         GRU + output_layer
frozen actor            k + speed_mlp + dummy_embedding
log_std                 frozen
critic                  C0_RAW_SINGLE_FRAME
critic LR               3e-4

n_envs                  16
n_steps                 1600
batch_size              1600
updates                 8
checkpoint_updates      2, 4, 8

gamma                   0.999
gae_lambda              0.995
clip_range              0.10
clip_range_vf           None
normalize_advantage     True
vf_coef                 0.5
ent_coef                0.0
max_grad_norm           0.5

GRU LR                  1e-6
head LR                 1e-5
target_kl               0.010
n_epochs                4

steering latent std     0.05
speed physical std      0.15

hard pool               h0_current_det
hard probability        0.50
hard sampling mode      with_replacement

reward progress         0.01
reward relative         0.02
reward ego collision    -2.0
sim duration             8.0 s
```

派生量必须在 `resolved_config.json` 中验证：

```text
transitions/update              = 25,600
minibatches/epoch               = 16
planned optimizer steps/update  = 64
planned total optimizer steps   = 512
```

每个正式 run 必须从以下 canonical BC fresh start：

```text
pretrained/end2race.pth
sha256 = b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4
```

禁止 resume 任何 V1、V1.1、V1.2、SG 或其他 PPO checkpoint。

---

# 4. 允许修改的文件

正式准备阶段只允许修改或新增：

```text
ppo/config.py
train_ppo.py
.gitignore
ppo_experiments/v1_3_b/aggregate_results.py
ppo_experiments/v1_3_b/PRECHECK.json
```

本指南已经位于：

```text
ppo_experiments/v1_3_b/IMPLEMENTATION_GUIDE.md
```

禁止修改：

```text
model.py
ppo/policy.py
ppo/environment.py
ppo/reward.py
ppo/scenarios.py
ppo/hard_pools/*
eval_multiagent.py
evaluate.sh
utils.py
f1tenth_gym/*
latticeplanner/*
SB3 或 sb3-contrib site-packages
```

原因：当前实验只验证 actor update 数据复用强度。修改上述任何文件都会改变问题本身。

---

# 5. `ppo/config.py` 的最小实现

## 5.1 新增一个可选 guardrail 字段

在 `PPOConfig` 中增加：

```python
update_kl_guardrail: float | None = None
```

在 `_validate()` 中增加：

```python
if config.update_kl_guardrail is not None and config.update_kl_guardrail <= 0.0:
    raise ValueError("update_kl_guardrail must be positive or None")
```

所有已有 config 默认保持 `None`，不能改变已有 config 的解析结果。

## 5.2 新增唯一正式配置

基于 `AH_H0_P50_WR` 增加：

```python
V1_3_B = replace(
    AH_H0_P50_WR,
    name="v1_3_b",
    updates=8,
    checkpoint_updates=(2, 4, 8),
    n_epochs=4,
    target_kl=0.010,
    update_kl_guardrail=0.020,
)
```

把 `V1_3_B` 加入 `CONFIGS`。

不得新增：

- 第二个 V1.3-B 配置；
- LR 对照 arm；
- epoch 2/3/5 arm；
- critic 或 reward arm；
- H1/H3、batch 6400、steer-low 等组合。

V1.3-B 不是 sweep。

---

# 6. `train_ppo.py` 的最小遥测和停止逻辑

## 6.1 为什么需要这些遥测

本实验需要知道：

1. `n_epochs=4` 实际执行了多少 optimizer steps；
2. `target_kl` 是否提前停止了后续 minibatch；
3. actor 相对 BC 的参数变化是否仍然极小；
4. update 是否进入不受控 KL 区间。

不要实现新的 PPO loss、rollback、trust region 或 BC regularizer。

## 6.2 记录实际 optimizer steps

增加一个只读 helper，从 `model.policy.optimizer.state` 中读取 Adam `step`：

```text
before_step
model.learn(...)
after_step
optimizer_steps_this_update = after_step - before_step
```

每个 update 记录：

```text
planned_optimizer_steps = 64
actual_optimizer_steps
optimizer_step_min
optimizer_step_max
target_kl_early_stop = actual_optimizer_steps < 64
effective_epoch_fraction = actual_optimizer_steps / 16.0
```

要求：

- active optimizer parameters 都有 state；
- step 值 finite；
- 同一 update 后各 active parameter 的 step 一致；
- `actual_optimizer_steps > 0`。

## 6.3 记录 actor 相对 BC 的参数变化

在 run 开始时保存 canonical BC 的 CPU snapshot。每个 update 后，对以下两组分别计算：

```text
gru
output_layer
```

记录：

```text
max_abs_delta_from_bc
rms_delta_from_bc
relative_rms_delta_from_bc
```

定义：

```text
relative_rms_delta = rms(candidate - BC) / max(rms(BC), 1e-12)
```

不要用该指标做 checkpoint selection。它只是验证“actor 是否仍然近似不动”的诊断量。

## 6.4 延续现有训练指标

每个 update 仍需记录：

```text
loss
policy_gradient_loss
value_loss
approx_kl
clip_fraction
explained_variance
rollout completed episodes
rollout outcomes
action mean/std/min/max
reward component means
unique scenario count
```

所有标量必须 finite。

## 6.5 Post-update KL guardrail

`target_kl=0.01` 使用 stock SB3 行为，只能停止尚未执行的 minibatch，不能撤销已经发生的 step。

因此在每次 `model.learn()` 返回后，再检查：

```python
if approx_kl > config.update_kl_guardrail:
    status = "STOPPED_KL_GUARDRAIL"
```

规则：

- guardrail 值固定为 `0.020`；
- 先写入该 update 的 `training_metrics.jsonl`；
- 再写入 `run_status.json`；
- 不保存该 update 的 actor checkpoint；
- 不继续下一 update；
- 不 rollback；
- 不自动降低 LR；
- 不自动重跑。

## 6.6 `run_status.json`

run 创建后立即写：

```json
{
  "status": "RUNNING",
  "config": "v1_3_b",
  "seed": 20260723,
  "last_completed_update": 0,
  "stop_reason": null
}
```

每个 update 原子更新。成功完成 U8 后：

```json
{
  "status": "COMPLETED",
  "config": "v1_3_b",
  "seed": 20260723,
  "last_completed_update": 8,
  "stop_reason": null
}
```

KL 超限时：

```json
{
  "status": "STOPPED_KL_GUARDRAIL",
  "config": "v1_3_b",
  "seed": 20260723,
  "last_completed_update": 1,
  "stop_reason": "approx_kl 0.023 exceeds 0.020"
}
```

异常退出时保留已有 run 目录和日志，不重命名，不覆盖。

## 6.7 Checkpoint 顺序

每个 update 的顺序必须是：

```text
model.learn
→ 收集并验证 metrics
→ 写 training_metrics.jsonl
→ 检查 post-update KL guardrail
→ guardrail 通过后，才允许保存 U2/U4/U8 checkpoint
→ 更新 run_status.json
```

---

# 7. 路径规范

## 7.1 正式训练输出

```text
runs/ppo/v1_3_b_seed20260723/
runs/ppo/v1_3_b_seed20260724/
runs/ppo/v1_3_b_seed20260725/
runs/ppo/v1_3_b_seed20260726/
runs/ppo/v1_3_b_seed20260727/
```

每个 run 必须包含：

```text
resolved_config.json
run_status.json
training_metrics.jsonl
checkpoint_manifest.json
sampler_summary.json
checkpoints/
  end2race_ppo_v1_3_b_u0002_s<seed>.pth
  end2race_ppo_v1_3_b_u0004_s<seed>.pth
  end2race_ppo_v1_3_b_u0008_s<seed>.pth
```

## 7.2 日志路径

不要把 shell log 写到尚未创建的 run 目录。这样会提前创建目录并触发 trainer 的 no-overwrite 检查。

统一使用：

```text
runs/ppo/v1_3_b_logs/train_seed<seed>.log
runs/ppo/v1_3_b_logs/eval_<checkpoint_stem>.log
runs/ppo/v1_3_b_logs/eval_bc.log
```

## 7.3 BC 配对基线

创建一次本地只读副本：

```text
runs/ppo/v1_3_b_baseline/end2race_bc_v1_3_b.pth
```

它必须与 canonical BC SHA-256 完全一致。

其 evaluator 输出自动位于：

```text
eval_results/end2race_bc_v1_3_b_Austin/multiagents/results_multi.json
```

## 7.4 Candidate evaluator 输出

示例：

```text
eval_results/end2race_ppo_v1_3_b_u0008_s20260723_Austin/multiagents/results_multi.json
```

## 7.5 Git 中保留的精简证据

只提交：

```text
ppo_experiments/v1_3_b/IMPLEMENTATION_GUIDE.md
ppo_experiments/v1_3_b/aggregate_results.py
ppo_experiments/v1_3_b/PRECHECK.json
ppo_experiments/v1_3_b/RESULTS.json
ppo_experiments/v1_3_b/REPORT.md
```

不要提交：

```text
runs/ppo/v1_3_b_seed*/
runs/ppo/v1_3_b_logs/
runs/ppo/v1_3_b_baseline/
eval_results/
checkpoint binaries
videos
traces
worker temp files
```

在 `.gitignore` 中增加：

```gitignore
runs/ppo/v1_3_b_seed*/
runs/ppo/v1_3_b_logs/
runs/ppo/v1_3_b_baseline/
```

提交时禁止使用 `git add -A`。必须显式列出允许提交的文件。

---

# 8. `aggregate_results.py` 的职责

新增：

```text
ppo_experiments/v1_3_b/aggregate_results.py
```

它只能读取已有 artifacts，不能训练、评价、修改 checkpoint 或补写 evaluator row。

## 8.1 输入

固定读取：

```text
5 个 run 的 resolved_config.json
5 个 run 的 run_status.json
5 个 run 的 training_metrics.jsonl
5 个 run 的 checkpoint_manifest.json
1 个 BC results_multi.json
15 个 candidate results_multi.json（5 seeds × U2/U4/U8）
```

## 8.2 Evaluation 严格验证

对 BC 和每个 candidate 均要求：

```text
600 episode rows
600 unique scenario_id
error = 0
ego collision scope
render = false
trace = false
noise = 0
scenario_id set 与 BC 完全一致
所有数值 finite
```

禁止：

- 忽略缺失 row；
- 合并两个 evaluation attempt；
- 用 aggregate total 代替 unique row 验证；
- 自动重跑；
- 修改原始 JSON。

## 8.3 Outcome 定义

每个 scenario 按以下顺序分类：

```text
if ego_collision_occurred:
    collision
elif state_label indicates overtake:
    overtake
else:
    follow
```

若出现未知 `state_label`，立即失败。

## 8.4 Paired 指标

对每个 candidate 计算：

```text
fixed_collision:
    BC collision → PPO non-collision

new_collision:
    BC non-collision → PPO collision

G:
    fixed_collision - new_collision

gained_overtake:
    BC non-overtake → PPO overtake

lost_overtake:
    BC overtake → PPO non-overtake
```

同时计算：

```text
mean_avg_speed
mean_total_distance
speed_ratio_to_bc
distance_ratio_to_bc
```

## 8.5 Training 摘要

把每个 seed 每个 update 的以下字段写入 `RESULTS.json`：

```text
approx_kl
clip_fraction
planned_optimizer_steps
actual_optimizer_steps
effective_epoch_fraction
target_kl_early_stop
GRU max/RMS/relative-RMS delta
head max/RMS/relative-RMS delta
completed episodes
collision/follow/overtake counts
action statistics
```

原始文件路径和 SHA-256 也必须记录。

---

# 9. 正式评价零点

V1.3-B 必须先在当前代码下重新运行一次 canonical BC，原因不是怀疑历史结果，而是要生成当前实验的 exact paired rows。

评价命令必须固定：

```bash
MODEL_PATH=runs/ppo/v1_3_b_baseline/end2race_bc_v1_3_b.pth \
COLLISION_SCOPE=ego \
RENDER=false \
SAVE_TRACE=false \
NUM_WORKERS=4 \
NUM_STARTPOINTS=50 \
EGO_IDX_OFFSET=0 \
NOISE=0.0 \
SIM_DURATION=8.0 \
bash evaluate.sh
```

基线必须得到：

```text
ego collision = 21
follow        = 233
overtake      = 346
error         = 0
total         = 600
```

若不一致：

```text
STOP_PROTOCOL_DRIFT
```

不得改用 `holdout33` 的 `32/236/332`；那是不同面板。

评价统一使用 `NUM_WORKERS=4`，因为历史上 8-worker evaluation 出现过一次 `libllvmlite` native crash，4-worker full recovery 已通过。V1.3-B 不需要再次承担该并发风险。

---

# 10. 正式成功标准

正式产品判断只使用五个 seed 的 U8。

U2 和 U4 只用于描述学习曲线，不能用于选择、替代或补救 U8。

## 10.1 训练稳定性条件

五个 seed 必须全部：

```text
run_status = COMPLETED
last_completed_update = 8
所有训练标量 finite
每个 update actual_optimizer_steps > 0
每个 update approx_kl <= 0.020
无 action NaN/Inf
无 checkpoint schema 错误
```

`clip_fraction` 记录但不作为单独硬门。报告中必须指出它是否长期接近 0 或长期高于 0.2。

## 10.2 U8 单 seed 条件

每个 seed 的 U8 都必须满足：

```text
G = fixed_collision - new_collision >= 5

ego collision <= 16

overtake >= 340
# 即最多比 BC 的 346 少 6

speed_ratio_to_bc >= 0.99

distance_ratio_to_bc >= 0.99
```

这些条件防止把：

- 轻微随机 churn；
- 全局减速；
- 放弃交互；
- 只在一个 seed 上出现的尾部结果

写成成功。

## 10.3 跨 seed 条件

必须是：

```text
5/5 seed 的 U8 全部通过单-seed 条件
```

不允许：

- 4/5 通过后删除失败 seed；
- 为失败 seed 改用 U2 或 U4；
- 对不同 seed 使用不同 checkpoint；
- 根据结果新增第六个 seed；
- 根据 development 结果修改阈值。

## 10.4 最终 verdict

只允许以下三个 verdict：

```text
PASS_STABLE_ACTOR_UPDATE
    五个 seed 全部完成，KL 受控，五个 U8 全部满足产品门。

FAIL_KL_UNSTABLE
    任一 seed 因 post-update KL > 0.020 停止，或 U8 前出现非有限/动作异常。

FAIL_NO_STABLE_IMPROVEMENT
    五个 seed 均稳定完成，但至少一个 U8 不满足产品门。
```

诊断解释：

```text
若 FAIL_NO_STABLE_IMPROVEMENT 且多数 update KL < 0.002：
    结果与 actor 仍处于低移动区间一致；不能自动提高 LR，需另行设计。

若 FAIL_NO_STABLE_IMPROVEMENT 且 KL 多数位于 0.002–0.01：
    actor 已有受控移动，但 advantage/product 方向跨 seed 不一致；
    继续提高 LR 或 epochs 的依据不足。

若 PASS_STABLE_ACTOR_UPDATE：
    只能说明该固定配置在当前 development panel 上产生一致改进；
    不得直接部署，下一步必须是新的预注册 holdout。
```

不要把简单的 `5/5` 方向一致写成严格独立统计证明。它是强工程一致性门，不是完整显著性分析。

---

# 11. 实现前检查与 PRECHECK

## 11.1 起始检查

本地 Codex 必须执行：

```bash
git checkout main
git pull --ff-only
git status --porcelain
```

要求 worktree 为空。

记录：

```text
HEAD
Python version
PyTorch version
stable-baselines3 version
sb3-contrib version
CUDA device
BC SHA-256
ppo/config.py SHA-256
train_ppo.py SHA-256
ppo/policy.py SHA-256
ppo/environment.py SHA-256
ppo/reward.py SHA-256
ppo/scenarios.py SHA-256
eval_multiagent.py SHA-256
evaluate.sh SHA-256
utils.py SHA-256
h0_current_det.json SHA-256
```

检查无活动训练或评价进程，且正式输出目录均不存在。

## 11.2 静态验证

实现后运行：

```bash
python -m compileall -q train_ppo.py ppo ppo_experiments/v1_3_b/aggregate_results.py
python train_ppo.py --help
git diff --check
```

用 Python 断言：

```text
get_config("v1_3_b") 存在
n_epochs = 4
updates = 8
checkpoint_updates = (2,4,8)
target_kl = 0.01
update_kl_guardrail = 0.02
transitions/update = 25,600
planned optimizer steps/update = 64
planned total optimizer steps = 512
```

同时验证所有旧 config 仍能加载，旧 config 的 `update_kl_guardrail` 都为 `None`。

## 11.3 一次非正式 smoke

只运行一次临时 one-update smoke：

- 使用 V1.3-B 的所有参数；
- `updates=1`、`checkpoint_updates=(1,)` 仅在内存中的临时 dataclass copy 中修改；
- 输出到 `/tmp/end2race_v1_3_b_smoke`；
- 不创建正式 run 目录；
- smoke 完成后删除临时目录。

smoke 必须验证：

```text
1 rollout = 25,600 transitions
planned optimizer steps = 64
actual optimizer steps in (0, 64]
finite metrics
run_status schema valid
checkpoint 12-key strict load
GRU/head delta > 0
frozen k/speed_mlp/dummy/log_std delta = 0
```

不要运行 600-case smoke。

## 11.4 PRECHECK.json

写入：

```text
ppo_experiments/v1_3_b/PRECHECK.json
```

至少包含：

```text
source HEAD
changed files
all recorded hashes
resolved V1.3-B config
static validation results
smoke command
smoke status
smoke metrics summary
smoke artifact location and cleanup status
```

PRECHECK 必须为 `PASS` 才能开始正式训练。

---

# 12. 提交边界

正式训练前，先提交实现：

```text
commit message:
Implement PPO v1.3-b controlled actor update test
```

该提交只允许包含：

```text
ppo/config.py
train_ppo.py
.gitignore
ppo_experiments/v1_3_b/aggregate_results.py
ppo_experiments/v1_3_b/PRECHECK.json
```

本指南所在 commit 不需要重写。

提交后要求：

```bash
git status --porcelain
```

为空。

记录 implementation commit SHA。正式五 seed 期间禁止修改任何代码、config、evaluator 或 manifest。

---

# 13. 正式训练执行

## 13.1 顺序

五个 seed 串行执行：

```text
20260723
20260724
20260725
20260726
20260727
```

禁止同 GPU 并行训练。

## 13.2 命令

先创建独立日志目录：

```bash
mkdir -p runs/ppo/v1_3_b_logs
```

每个 seed：

```bash
set -o pipefail
python train_ppo.py \
  --config v1_3_b \
  --seed <SEED> \
  --output_dir runs/ppo/v1_3_b_seed<SEED> \
  2>&1 | tee runs/ppo/v1_3_b_logs/train_seed<SEED>.log
```

不要提前创建 `runs/ppo/v1_3_b_seed<SEED>`。

## 13.3 每个 seed 后的检查

只读取，不修改：

```text
run_status.json
resolved_config.json
training_metrics.jsonl
checkpoint_manifest.json
sampler_summary.json
```

要求：

```text
status = COMPLETED
last_completed_update = 8
3 checkpoints exist exactly at U2/U4/U8
all three strict-load as 12-key End2Race
```

若任一 seed 不是 `COMPLETED`：

- 停止，不启动后续 seed；
- 不重跑；
- 不改参数；
- 保留证据；
- 最终 verdict 为 `FAIL_KL_UNSTABLE` 或基础设施失败。

---

# 14. 正式评价执行

所有五个训练 run 完成后才能开始评价。

## 14.1 Baseline

```bash
mkdir -p runs/ppo/v1_3_b_baseline
cp pretrained/end2race.pth runs/ppo/v1_3_b_baseline/end2race_bc_v1_3_b.pth
sha256sum pretrained/end2race.pth runs/ppo/v1_3_b_baseline/end2race_bc_v1_3_b.pth
```

然后运行第 9 节固定命令。

## 14.2 Candidate 顺序

先评价五个 U8，再评价 U2 和 U4：

```text
U8: 5 seeds
U2: 5 seeds
U4: 5 seeds
```

这不会改变正式判定规则；只是优先取得 primary endpoint。

每个 checkpoint 使用：

```bash
MODEL_PATH=<CHECKPOINT> \
COLLISION_SCOPE=ego \
RENDER=false \
SAVE_TRACE=false \
NUM_WORKERS=4 \
NUM_STARTPOINTS=50 \
EGO_IDX_OFFSET=0 \
NOISE=0.0 \
SIM_DURATION=8.0 \
bash evaluate.sh
```

每次评价前要求对应 `eval_results/<stem>_Austin` 不存在。

## 14.3 Evaluation 失败政策

若出现：

```text
进程非零退出
少于 600 unique rows
error > 0
scenario set 不一致
非有限值
```

则：

- 保留原始输出和 log；
- 停止后续评价；
- 不自动 retry；
- 不降低 workers 再跑；
- 不合并 partial rows；
- 报告基础设施失败并等待 owner 决定。

---

# 15. 结果生成

所有评价有效后执行：

```bash
python ppo_experiments/v1_3_b/aggregate_results.py
```

生成：

```text
ppo_experiments/v1_3_b/RESULTS.json
ppo_experiments/v1_3_b/REPORT.md
```

`REPORT.md` 只包含：

1. 实现 commit 和核心 hashes；
2. 固定 config；
3. baseline；
4. 每 seed U2/U4/U8 的 collision/follow/overtake；
5. fixed/new、gained/lost；
6. speed/distance ratio；
7. 每 update KL、clip fraction、实际 optimizer steps；
8. GRU/head 参数变化；
9. 五个 U8 的固定门；
10. 唯一最终 verdict。

不得写未经结果支持的机制故事。

最终结果提交：

```text
commit message:
Record PPO v1.3-b five-seed results
```

只提交：

```text
ppo_experiments/v1_3_b/RESULTS.json
ppo_experiments/v1_3_b/REPORT.md
```

如 PRECHECK 因实现 commit SHA 需要补充，只允许同步更新 `PRECHECK.json` 的 commit 字段，不改变实验合同。

---

# 16. 明确禁止的行为

Codex 不得：

1. 修改 critic；
2. 修改 reward；
3. 修改 H0 或 sampler；
4. 使用 H1/H3；
5. 改 steering/speed std；
6. 提高或降低 LR；
7. 增加第六个 seed；
8. 根据 U2/U4 结果停止一个正常 run；
9. 为不同 seed 选择不同 checkpoint；
10. 把 U2 峰值当最终结果；
11. 使用 holdout33；
12. 创建新的 holdout；
13. 修改 evaluator；
14. 修改 SB3 PPO loss、GAE 或 rollout buffer；
15. 加 rollback、BC KL anchor、residual、gate 或 macro action；
16. 自动 retry 失败的训练或评价；
17. 覆盖已有 run/eval 目录；
18. 把 checkpoint 复制到 `posttrained/`；
19. 把 development PASS 写成部署 PASS；
20. 使用 `git add -A` 提交原始 artifacts。

---

# 17. Codex 复制执行 Prompt

```text
你正在 HaoweiLi1/End2Race 的 main 分支实现和执行 End2Race PPO V1.3-B。

唯一权威文档：
ppo_experiments/v1_3_b/IMPLEMENTATION_GUIDE.md

目标：
只验证一个固定 actor-update window：保持原 actor LR、C0 critic、reward、H0 sampler、batch、rollout 和 exploration 不变，把 n_epochs 从 1 增加到 4，并使用 stock target_kl=0.01 和 post-update aggregate KL guardrail 0.02。运行五个 fresh seed，正式结果只使用每个 seed 的 U8。

严格边界：
- 不重新设计方案；
- 不新增参数 arm；
- 不修改 policy/environment/reward/scenarios/evaluator/F110/Lattice Planner/site-packages；
- 不使用 holdout33；
- 不从 PPO checkpoint resume；
- 不为不同 seed 选择最好 checkpoint；
- 不自动 retry；
- 不提交 raw runs、eval_results 或 checkpoint。

先完成：
1. clean main + pull --ff-only；
2. 按指南只修改 ppo/config.py、train_ppo.py、.gitignore；
3. 新增 aggregate_results.py；
4. 完成静态检查和一次 /tmp one-update smoke；
5. 写 PRECHECK.json；
6. 提交 “Implement PPO v1.3-b controlled actor update test”；
7. 确认 worktree clean。

然后串行运行 seed：
20260723, 20260724, 20260725, 20260726, 20260727。

任何 seed 若 run_status 不是 COMPLETED，立即停止，不运行后续 seed。

五个训练全部完成后：
- 用 4 workers、EGO_IDX_OFFSET=0 重跑一次 canonical BC；
- 必须得到 21 collision / 233 follow / 346 overtake / 0 error；
- 依次评价五个 U8，再评价 U2/U4；
- 每个结果必须有 600 unique rows 和 0 error。

最后运行 aggregate_results.py，生成 RESULTS.json 和 REPORT.md。只使用五个 U8 应用固定门：每个 seed G>=5、collision<=16、overtake>=340、speed ratio>=0.99、distance ratio>=0.99，且所有 update approx_kl<=0.02。

最终只允许：
PASS_STABLE_ACTOR_UPDATE
FAIL_KL_UNSTABLE
FAIL_NO_STABLE_IMPROVEMENT

遇到任何基础设施错误，保存证据并停止，不自行修复实验合同。
```

---

# 18. 完成条件

V1.3-B 只有在以下文件存在且通过 schema 验证时才算完成：

```text
ppo_experiments/v1_3_b/PRECHECK.json
ppo_experiments/v1_3_b/RESULTS.json
ppo_experiments/v1_3_b/REPORT.md
```

并且：

```text
5/5 formal training runs have terminal status
5/5 U8 evaluations valid
15/15 total candidate evaluations valid
1/1 BC evaluation valid
all actor checkpoints are exact 12-key strict-loadable
all artifacts point to recorded SHA-256
one and only one final verdict is reported
```

完成不等于成功。若 verdict 为失败，必须原样保留结果，不得追加参数搜索把 V1.3-B 改造成 V1.3-B sweep。
