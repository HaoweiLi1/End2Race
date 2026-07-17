# End2Race PPO V1.3-A：固定 3× Actor LR 的受控更新验证指南

**状态：** `READY_FOR_LOCAL_CODEX_IMPLEMENTATION`  
**仓库：** `HaoweiLi1/End2Race`  
**代码审阅边界：** `main@641299e0101ebdb3c05a3642d36e4fcab7a789e5`  
**正式配置名：** `v1_3_a`  
**正式训练 seed：** `20260718, 20260719, 20260720, 20260721, 20260722`  
**唯一正式判定 checkpoint：** 每个 seed 的 `U8`  
**开发评价面板：** Austin canonical development 600，`EGO_IDX_OFFSET=0`  
**正式证据目录：** `ppo_experiments/v1_3_a/`  
**用途：** 指导本地 Codex 以最小代码改动实现、验证和执行 V1.3-A。不得把本指南扩展为 LR sweep、checkpoint search、critic/reward 项目或新 PPO 算法。

---

# 1. 执行结论先行

V1.3-A 只回答一个问题：

> 在保持 PPO 数据、reward、critic、sampler、batch、rollout、epoch 数和 exploration 全部不变时，将 GRU/head actor LR 固定提高到当前值的 `3×`，能否进入一个非微小但仍受控的更新区间，并在五个 fresh seed 的固定 U8 上产生一致的 collision 改善？

V1.3-A 不是：

- 不是重新搜索 2×、3×、4× 或更大的 LR；
- 不是自动寻找“最大安全 KL”；
- 不是增加 `n_epochs`；
- 不是修改 critic、reward、GAE、hard pool、batch、rollout 或 exploration；
- 不是 residual policy、safety shield 或新网络；
- 不是从 V1/V1.1/V1.2/SG checkpoint resume；
- 不是在 U2/U4/U8 中挑最好 checkpoint；
- 不是用五个 seed 中最好的一个代表整条线；
- 不是独立泛化或部署确认。

本实验只有：

```text
一个固定配置
五个 fresh seed
每个 seed 一个 U8 checkpoint
一个 paired BC development 判决
```

若失败，不追加 LR 档位、epoch、seed 或训练预算。

---

# 2. 方案审计：与真实结果的对应关系

## 2.1 必须读取的正式证据

本地 Codex 在实现前必须读取：

```text
ppo_experiments/v1_2_reduced/CHECKPOINT_DISTRIBUTION_ANALYSIS.json
ppo_experiments/v1_2_reduced/HOLDOUT_RESULTS.json
ppo_experiments/v1_2_reduced/FINAL_REPEATABILITY.json
ppo_experiments/signal_repair/SG_P1_RESULTS.json
ppo/config.py
train_ppo.py
```

`.agents/`、HANDOFF 与自然语言判断不是结果权威。

## 2.2 已确认的事实

当前 records 直接支持：

1. canonical development BC 为 `21 collision / 233 follow / 346 overtake`；
2. 46 个去重 V1.x checkpoint 的 collision 均值为 `20.72`、标准差 `3.05`，整体仍以 BC 为中心；
3. paired churn 平均为 `6.96 fixed / 6.67 new collision`，没有形成稳定净修复；
4. development 最佳的 `15/353` 在 holdout33 上变为 `37/329`，paired BC 为 `32/332`；
5. 三 seed 长训练 U16 为 `26/346`、`21/341`、`24/342`，延长同一微 LR 配方没有建立下降趋势；
6. 当前主训练族的 actor LR 为 `GRU=1e-6`、`head=1e-5`，`n_epochs=1`；
7. `10× LR + n_epochs=2` 的 Signal Repair run 在 U1 出现约 `0.048–0.093` 的 approximate KL，并触发 guardrail。

## 2.3 合理但尚未证明的推断

现有数据只允许提出：

```text
1× nominal LR 下，整体结果与 BC-adjacent churn 一致；
10× LR 并同时使用 2 epochs 时，更新明显过大；
因此一个固定的中等 LR 仍未被干净测试。
```

这不证明“核心问题就是 LR 太小”，也不证明更大 LR 会改善。V1.3-A 是一个可证伪的单变量实验。

## 2.4 为什么固定为 3×，不做 2×/3×/4× 标定 sweep

选择：

```text
GRU LR  = 3e-6
head LR = 3e-5
```

原因仅为：

- `3×` 接近 `1×` 与 `10×` 的对数中点 `sqrt(10) ≈ 3.16`；
- 保持 `n_epochs=1`，明显低于已失控的 `10× LR + 2 epochs`；
- 一个固定配置比先跑九个 calibration run 再选择 LR 更简单，也避免用过程指标制造新的选择自由度。

`3×` 是固定的工程探针，不是最优 LR 声明。若它仍过小或过大，V1.3-A 直接失败，不在本轮继续搜索。

## 2.5 必须避免的过度表述

执行和报告中不得写：

- “V1.x 没有学到任何东西”；
- “critic、reward 或 H0 已经最优”；
- “参数变化小就是唯一根因”；
- “3× 一定处于正确区间”；
- “五 seed 通过就证明所有地图泛化”；
- “development PASS 等于可部署”。

正确表述是：

> V1.3-A 检验固定 3× actor LR 在现有 PPO 合同下，是否产生受控且跨 seed 一致的 development improvement。

---

# 3. 冻结训练合同

除 actor LR 和新增的只读遥测/停止护栏外，以下全部冻结。

## 3.1 软件与环境

```text
Python                  3.10
stable-baselines3       2.7.1
sb3-contrib             2.7.1
simulator timestep      0.01 s
actor frequency         100 Hz
map                     Austin
sim duration            8.0 s
learned agent           ego only
opponent                fixed Lattice Planner
```

正式 preregistration 必须记录实际版本。若版本不同，停止，不自动升级或降级。

## 3.2 Canonical BC

```text
path:
pretrained/end2race.pth

sha256:
b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4
```

所有 seed fresh start。禁止 resume。

## 3.3 Actor 与 checkpoint

保持原始 End2Race 12-key actor：

```text
trainable:
- gru
- output_layer

frozen:
- k
- speed_mlp
- dummy_embedding
- log_std
```

最终 actor checkpoint 必须 strict-load 到：

```python
End2Race(mask_prob=0.0, hidden_scale=4)
```

## 3.4 Observation / recurrent / action contract

不得修改：

```text
actor observation         LiDAR_t + previous measured ego speed
real recurrent state      GRU h
SB3 dummy state           zero c transport
hidden reset              episode-start mask
steering distribution     latent Normal -> 0.52 * tanh
speed distribution        physical Normal in m/s
steering latent std       0.05, frozen
speed physical std        0.15, frozen
```

## 3.5 Critic

```text
critic_profile = C0_RAW_SINGLE_FRAME
critic_lr      = 3e-4
```

不增加 privileged state、actor-hidden critic 或 critic recurrence。

## 3.6 Reward

```text
0.01 * ego_progress_delta
+ 0.02 * relative_progress_delta
- 2.0 * first ego collision
margin_weight    = 0.0
margin_threshold = 0.0
```

不启用 Signal Repair margin，不增加 TTC、clearance、smoothness 或 terminal reward。

## 3.7 Sampler

```text
hard_pool                 = h0_current_det
hard_sampling_probability = 0.50
hard_sampling_mode        = with_replacement
```

H0 在本实验中是固定控制，不代表它已被证明最优。

## 3.8 PPO geometry

```text
n_envs              = 16
n_steps             = 1600
batch_size          = 1600
n_epochs            = 1
updates             = 8
checkpoint_updates  = (8,)

gamma               = 0.999
gae_lambda          = 0.995
clip_range          = 0.10
clip_range_vf       = None
normalize_advantage = True
vf_coef             = 0.5
ent_coef            = 0.0
max_grad_norm       = 0.5
target_kl           = 0.010

GRU LR              = 3e-6
head LR             = 3e-5
critic LR           = 3e-4
```

派生量：

```text
transitions/update              = 16 * 1600 = 25,600
minibatches/update              = 25,600 / 1,600 = 16
planned optimizer steps/update  = 16
planned optimizer steps/seed    = 128
transitions/seed                 = 204,800
five-seed transitions           = 1,024,000
```

---

# 4. 正式 seeds 与唯一 endpoint

固定 seeds：

```text
20260718
20260719
20260720
20260721
20260722
```

这些 seed 在 preregistration 前锁定，不根据任何结果替换或追加。

唯一正式判定 endpoint：

```text
U8
```

U1–U7 仅有训练日志，不保存产品候选 checkpoint。不得把中间状态改成候选。

---

# 5. 允许修改的文件

正式准备阶段只允许修改或新增：

```text
ppo/config.py
train_ppo.py
.gitignore                                  # 仅在需要忽略新增本地目录时
ppo_experiments/v1_3_a/aggregate_results.py
ppo_experiments/v1_3_a/PRECHECK.json
```

本指南路径：

```text
ppo_experiments/v1_3_a/IMPLEMENTATION_GUIDE.md
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
SB3 / sb3-contrib site-packages
pretrained/end2race.pth
```

若运行必须修改冻结文件，停止并上报，不自行扩大范围。

---

# 6. `ppo/config.py` 最小实现

## 6.1 可选 guardrail 字段

若当前 main 尚未由 V1.3-B 实现加入该字段，在 `PPOConfig` 增加：

```python
update_kl_guardrail: float | None = None
```

验证：

```python
if config.update_kl_guardrail is not None and config.update_kl_guardrail <= 0.0:
    raise ValueError("update_kl_guardrail must be positive or None")
```

所有历史 config 保持 `None`。若该字段已经存在且语义完全一致，直接复用，禁止创建第二个字段。

## 6.2 唯一正式配置

基于 `V1_2_H0_CONTROL`：

```python
V1_3_A = replace(
    V1_2_H0_CONTROL,
    name="v1_3_a",
    updates=8,
    checkpoint_updates=(8,),
    n_epochs=1,
    gru_lr=3.0e-6,
    head_lr=3.0e-5,
    target_kl=0.010,
    steering_latent_std=0.05,
    speed_physical_std=0.15,
    margin_weight=0.0,
    margin_threshold=0.0,
    update_kl_guardrail=0.020,
)
```

加入 `CONFIGS`。

不得新增：

- `v1_3_a_lr2`、`v1_3_a_lr4`；
- head-only / GRU-only LR；
- 第二个 epoch 配置；
- H1/H3、lower-std、batch 或 critic 组合。

## 6.3 旧配置回归

所有已有 config 必须逐字段保持原值。新增 guardrail 默认值不得改变历史 run 解析。

---

# 7. `train_ppo.py` 的最小遥测与停止逻辑

不得改 PPO loss、optimizer 类型、rollout buffer 或 SB3 internals。

## 7.1 实际 optimizer steps

SB3-Contrib 2.7.1 在 target-KL 提前退出时，logger 的 `n_updates` 不能作为实际 Adam step 的唯一证据。必须从 optimizer state 读取实际 step。

每次 `model.learn()` 前后记录：

```text
planned_optimizer_steps = 16
optimizer_step_before
optimizer_step_after
actual_optimizer_steps = after - before
target_kl_early_stop = actual_optimizer_steps < 16
```

要求：

```text
actual_optimizer_steps > 0
actual_optimizer_steps <= 16
所有数值 finite
```

## 7.2 Actor 相对 BC 的参数变化

run 开始时保存 canonical BC CPU snapshot。每个 update 后对：

```text
gru
output_layer
frozen_actor
```

分别记录：

```text
max_abs_delta_from_bc
rms_delta_from_bc
relative_rms_delta_from_bc
max_abs_delta_from_previous
rms_delta_from_previous
relative_rms_delta_from_previous
```

定义：

```text
rms(x) = sqrt(sum(x^2) / numel)
relative_rms_delta = rms(candidate - reference) / max(rms(reference), 1e-12)
```

要求：

```text
frozen_actor.max_abs_delta_from_bc == 0.0
policy.log_std delta == 0.0
```

这些是诊断量，不用于 checkpoint 选择。

## 7.3 Post-update KL guardrail

`target_kl=0.010` 是 stock SB3 的软早停，不能撤销已经发生的 optimizer step。因此每次 outer update 完成后检查：

```text
approx_kl <= 0.020
```

顺序：

1. 完成 update；
2. 将该 update 的完整 metrics 写入 `training_metrics.jsonl`；
3. 写 `run_status.json`；
4. 若 `approx_kl > 0.020`，状态写为 `STOPPED_KL_GUARDRAIL`；
5. 不保存正式候选 checkpoint；
6. 关闭环境并退出非零；
7. 不自动 retry。

## 7.4 保留现有日志并补充字段

每个 update 至少记录：

```text
loss
policy_gradient_loss
value_loss
approx_kl
clip_fraction
explained_variance
planned_optimizer_steps
actual_optimizer_steps
target_kl_early_stop
GRU/head/frozen parameter deltas
log_std delta
rollout completed episodes by branch
action mean/std/min/max
reward component means
unique scenario counts
```

所有 scalar 必须 finite。

## 7.5 `run_status.json`

每个 run 目录必须维护：

```json
{
  "schema_version": 1,
  "experiment": "v1_3_a",
  "config": "v1_3_a",
  "seed": 20260718,
  "status": "RUNNING",
  "last_completed_update": 0,
  "stop_reason": null
}
```

允许状态：

```text
RUNNING
COMPLETED
STOPPED_KL_GUARDRAIL
FAILED_NONFINITE
FAILED_CHECKPOINT
FAILED_RUNTIME
```

---

# 8. 聚合脚本

允许新增：

```text
ppo_experiments/v1_3_a/aggregate_results.py
```

它只能：

- 读取 JSON/JSONL；
- 校验 run/checkpoint/evaluation；
- 计算 paired churn；
- 计算速度/距离比；
- 写 formal records。

禁止：

- 启动训练或评价；
- 修改 config；
- 自动重试；
- 选择 seed/checkpoint；
- 读取 U1–U7 作为产品候选；
- 比较旧 46 个 checkpoint 后选择阈值。

---

# 9. 目录规范

## 9.1 Tracked records

```text
ppo_experiments/v1_3_a/
  IMPLEMENTATION_GUIDE.md
  PRECHECK.json
  PREREGISTRATION.json
  STATUS.json
  RESULTS.json
  FINAL_REPORT.md
  aggregate_results.py
```

只提交精简 records 和脚本，不提交 checkpoint、完整日志、trace 或视频。

## 9.2 Training artifacts

```text
runs/ppo/v1_3_a_seed20260718/
runs/ppo/v1_3_a_seed20260719/
runs/ppo/v1_3_a_seed20260720/
runs/ppo/v1_3_a_seed20260721/
runs/ppo/v1_3_a_seed20260722/
```

每个 run 必须包含：

```text
resolved_config.json
training_metrics.jsonl
run_status.json
checkpoint_manifest.json
sampler_summary.json
checkpoints/end2race_ppo_v1_3_a_u0008_s<seed>.pth
```

正式 checkpoint filename 必须唯一。

## 9.3 Logs

为避免 `tee` 提前创建 trainer output directory：

```text
runs/ppo/v1_3_a_logs/
  train_seed<seed>.log
  eval_bc.log
  eval_seed<seed>.log
```

先创建 logs 目录，再让 `train_ppo.py` 自己创建 run dir。

## 9.4 Evaluation artifacts

BC reference copy：

```text
runs/ppo/v1_3_a_baseline/end2race_bc_v1_3_a.pth
```

候选自动写入：

```text
eval_results/end2race_ppo_v1_3_a_u0008_s<seed>_Austin/
  multiagents/results_multi.json
```

不得覆盖已有 evaluation output。存在即停止并审计。

---

# 10. Preflight 与 PRECHECK

## 10.1 起始检查

Codex 必须：

```bash
git checkout main
git pull --ff-only
git status --porcelain
```

要求 clean worktree。记录：

```text
HEAD
Python/PyTorch/SB3/sb3-contrib versions
CUDA device
canonical BC hash
ppo/config.py hash
train_ppo.py hash
ppo/policy.py hash
ppo/environment.py hash
ppo/reward.py hash
ppo/scenarios.py hash
eval_multiagent.py hash
evaluate.sh hash
utils.py hash
h0_current_det.json hash
```

确认：

```text
无 active formal trainer/evaluator
五个 run dirs 不存在
五个 candidate eval dirs 不存在
baseline eval dir 不存在
```

## 10.2 静态验证

实现后运行：

```bash
python -m compileall -q train_ppo.py ppo ppo_experiments/v1_3_a/aggregate_results.py
python train_ppo.py --help
git diff --check
```

Python 断言：

```text
get_config("v1_3_a") 存在
n_envs = 16
n_steps = 1600
batch_size = 1600
n_epochs = 1
updates = 8
checkpoint_updates = (8,)
GRU LR = 3e-6
head LR = 3e-5
critic LR = 3e-4
target_kl = 0.01
update_kl_guardrail = 0.02
margin = 0
std = 0.05 / 0.15
H0 p0.5 with-replacement C0
transitions/update = 25,600
planned optimizer steps/update = 16
planned total steps = 128
```

验证所有旧 config 仍可加载且未改变。

## 10.3 一次非正式 smoke

只运行一次 one-update 临时 smoke：

- 基于 `v1_3_a` 的内存 dataclass copy；
- 临时改为 `updates=1`、`checkpoint_updates=(1,)`；
- 输出 `/tmp/end2race_v1_3_a_smoke`；
- 不创建 formal run dir；
- 完成后删除临时目录。

必须验证：

```text
25,600 transitions
planned optimizer steps = 16
actual optimizer steps in [1,16]
finite metrics
approx_kl <= 0.020
GRU/head delta > 0
frozen delta = 0
12-key strict-load PASS
```

Smoke 只验证实现，不进入正式结果。

## 10.4 `PRECHECK.json`

必须记录：

```text
source HEAD
changed files
all required hashes
resolved config
static checks
smoke command/status/metrics
smoke cleanup status
```

`PRECHECK.status` 必须为 `PASS`。

---

# 11. Preregistration 与提交边界

正式训练前创建：

```text
ppo_experiments/v1_3_a/PREREGISTRATION.json
```

至少锁定：

```text
hypothesis
source/implementation commit
canonical BC hash
full resolved config
five seeds
U8-only endpoint
KL process gate
per-seed product gate
five-seed aggregate gate
BC evaluation contract
training/evaluation ordering
all expected paths
no-retry policy
forbidden changes
```

实现 commit 建议：

```text
Implement PPO v1.3-a fixed 3x actor LR test
```

实现 commit 只允许包含：

```text
ppo/config.py
train_ppo.py
.gitignore                      # 如确有必要
ppo_experiments/v1_3_a/aggregate_results.py
ppo_experiments/v1_3_a/PRECHECK.json
ppo_experiments/v1_3_a/PREREGISTRATION.json
ppo_experiments/v1_3_a/STATUS.json
```

提交后 worktree 必须 clean。正式训练期间禁止修改代码、config、evaluator 或 records 中锁定的合同。

---

# 12. 正式训练

## 12.1 顺序

五个 seed 串行：

```text
20260718
20260719
20260720
20260721
20260722
```

禁止同 GPU 并行训练。

## 12.2 命令

```bash
mkdir -p runs/ppo/v1_3_a_logs

python train_ppo.py \
  --config v1_3_a \
  --seed 20260718 \
  --output_dir runs/ppo/v1_3_a_seed20260718 \
  2>&1 | tee runs/ppo/v1_3_a_logs/train_seed20260718.log
```

其余 seed 仅替换 seed 和 output dir。

必须开启 shell `pipefail`：

```bash
set -o pipefail
```

每个 run 结束后只做结构/hash 校验，不做 development evaluation。五个训练全部结束后才进入评价。

## 12.3 禁止在线决策

训练期间不得：

- 看 U1–U7 后停止某个正常 run；
- 根据一个 seed 的 KL 改其他 seed LR；
- 因一个 seed 表现差而替换 seed；
- 增加 update；
- 保存额外产品 checkpoint；
- 在训练未全部完成时评价候选。

唯一例外是已预注册的 `approx_kl > 0.020` guardrail，它会使整条 V1.3-A 判为 KL failure。

---

# 13. 训练过程 gate

## 13.1 Run validity

每个 seed 必须：

```text
status = COMPLETED
last_completed_update = 8
training_metrics.jsonl = 8 rows
num_timesteps final = 204,800
U8 checkpoint exactly one
12-key strict-load PASS
all scalar finite
frozen deltas = 0
each update actual_optimizer_steps in [1,16]
each update approx_kl <= 0.020
```

## 13.2 “受控且非微小”的操作定义

对每个 seed 的 8 个 updates：

```text
至少 6/8 个 approx_kl 位于 [0.002, 0.010]
所有 8/8 个 approx_kl <= 0.020
```

这是本实验的操作定义，不是通用 PPO 定理。

若任一 seed 不满足：

- 有 update `>0.020`：`FAIL_KL_UNSTABLE`；
- 全程稳定但少于 6 个 update 进入 `[0.002,0.010]`：`FAIL_UPDATE_WINDOW_NOT_REACHED`。

不因“产品结果看起来不错”而忽略过程 gate。

## 13.3 仅记录的诊断

记录但不单独作为硬门：

```text
clip_fraction
explained_variance
policy/value loss
GRU/head parameter drift
completed episodes by branch
action statistics
reward component means
```

报告必须指出 clip fraction 是否长期接近 0 或长期高于 0.2，但不得据此挑 seed。

---

# 14. Canonical BC paired reference

为了获得当前实验的 exact paired rows，正式候选评价前运行一次 canonical BC。

先复制 checkpoint 到唯一 basename：

```bash
mkdir -p runs/ppo/v1_3_a_baseline
cp pretrained/end2race.pth \
  runs/ppo/v1_3_a_baseline/end2race_bc_v1_3_a.pth
sha256sum pretrained/end2race.pth \
  runs/ppo/v1_3_a_baseline/end2race_bc_v1_3_a.pth
```

两个 SHA 必须相同。

评价合同：

```bash
MODEL_PATH=runs/ppo/v1_3_a_baseline/end2race_bc_v1_3_a.pth \
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

必须复现：

```text
collision = 21
follow    = 233
overtake  = 346
error     = 0
total     = 600
```

若不一致：

```text
STOP_PROTOCOL_DRIFT
```

`NUM_WORKERS=4` 固定，以避免历史 8-worker native crash 风险。不得使用 holdout33 的 `32/236/332`。

---

# 15. 五个 U8 的 development evaluation

只评价：

```text
end2race_ppo_v1_3_a_u0008_s20260718.pth
end2race_ppo_v1_3_a_u0008_s20260719.pth
end2race_ppo_v1_3_a_u0008_s20260720.pth
end2race_ppo_v1_3_a_u0008_s20260721.pth
end2race_ppo_v1_3_a_u0008_s20260722.pth
```

统一合同与 BC 完全相同，仅替换 `MODEL_PATH`。

每次评价必须：

```text
600 episode rows
600 unique scenario IDs
error = 0
collision + follow + overtake = 600
all summary metrics finite
collision_scope = ego
noise = 0
render = false
trace = false
```

无效评价不得补行、合并或覆盖。保留 evidence，停止并等待 owner 授权；不得自动 retry。

---

# 16. Paired 指标

按相同 scenario ID 将每个候选与 BC 配对，计算：

```text
fixed_collision:
    BC collision -> candidate non-collision

new_collision:
    BC non-collision -> candidate collision

G:
    fixed_collision - new_collision

gained_overtake:
    BC non-overtake -> candidate overtake

lost_overtake:
    BC overtake -> candidate non-overtake

speed_ratio_to_bc:
    candidate avg_speed_mean / BC avg_speed_mean

distance_ratio_to_bc:
    candidate total_distance_mean / BC total_distance_mean
```

所有 episode key 必须一一对应。

说明：在同一 600-case paired panel 上，`G >= 5` 与总 collision `<=16` 代数等价；fixed/new 仍必须记录，用来揭示 churn，而不是把二者当作两份独立证据。

---

# 17. 正式成功标准

## 17.1 每 seed U8 产品门

五个 seed 的 U8 都必须：

```text
G >= 5
collision <= 16
overtake >= 340
speed_ratio_to_bc >= 0.99
distance_ratio_to_bc >= 0.99
```

这排除：

- 只减少 1–2 个的随机波动；
- 通过全局减速减少碰撞；
- 放弃超车；
- 只在一个 seed 上出现的尾部极值。

## 17.2 跨 seed 门

```text
5/5 seed 全部通过过程 gate
5/5 seed 全部通过 U8 产品门
```

不允许：

- 4/5 后删除失败 seed；
- 为失败 seed 换 checkpoint；
- 加第六个 seed；
- 用均值或中位数覆盖失败 seed；
- 事后放宽阈值。

## 17.3 最终 verdict

只允许：

```text
PASS_STABLE_3X_LR_DEVELOPMENT
    五 seed 全部完成；更新窗口受控且非微小；五个 U8 全部通过产品门。

FAIL_KL_UNSTABLE
    任一 update approx_kl > 0.020，或出现 non-finite/action/checkpoint failure。

FAIL_UPDATE_WINDOW_NOT_REACHED
    稳定完成，但任一 seed 少于 6/8 updates 进入 [0.002,0.010]。

FAIL_NO_STABLE_IMPROVEMENT
    五 seed 过程合法，但至少一个 U8 不满足产品门。

STOP_PROTOCOL_DRIFT
    paired BC 未复现 21/233/346/0。

INVALID_INFRASTRUCTURE
    正式评价不完整、重复、error 或其他协议无效。
```

PASS 只表示当前 canonical development panel 上的机制证据。不得直接 promotion 或部署。

---

# 18. 与 V1.3-B 的边界

当前 main 已包含独立的：

```text
ppo_experiments/v1_3_b/IMPLEMENTATION_GUIDE.md
```

两条路线分别检验：

```text
V1.3-A: 固定 n_epochs=1，只把 actor LR 改为 3×
V1.3-B: 固定原 actor LR，只把 n_epochs 改为 4
```

规则：

- A 不依赖 B 的结果，B 也不依赖 A；
- 不允许因 A 结果修改 B preregistration，反之亦然；
- 若共享 telemetry/guardrail 已由另一条路线实现，必须复用同一语义，不复制第二套；
- 不允许在 development 上从 A/B 中挑“最佳 actor”并直接声称成功；
- 若 A、B 任一或同时通过，后续新 holdout/face-off 必须另行预注册。

---

# 19. Formal records

## 19.1 `STATUS.json`

允许状态：

```text
GUIDE_READY
PRECHECK_PASS
PREREGISTERED
TRAINING
TRAINING_COMPLETE
EVALUATING
PASS_STABLE_3X_LR_DEVELOPMENT
FAIL_KL_UNSTABLE
FAIL_UPDATE_WINDOW_NOT_REACHED
FAIL_NO_STABLE_IMPROVEMENT
STOP_PROTOCOL_DRIFT
INVALID_INFRASTRUCTURE
```

必须记录当前 phase、source head、implementation commit、updated_at 和 failure reason。

## 19.2 `RESULTS.json`

至少包含：

```text
canonical BC reference path/hash/counts/rows hash
five training run paths/hashes/statuses
8-update KL and actual-step arrays per seed
parameter drift per update
five U8 checkpoint hashes
five evaluation result hashes
paired fixed/new/G and overtake churn
speed/distance ratios
all gate booleans
final verdict
selection_performed = false
holdout_performed = false
promotion_performed = false
```

## 19.3 `FINAL_REPORT.md`

必须包含：

1. exact frozen config；
2. five-seed process table；
3. BC + five U8 product table；
4. paired churn table；
5. gate-by-gate verdict；
6. evidence hashes；
7. interpretation boundary；
8. canonical BC deployment recommendation unchanged。

---

# 20. Failure policy

- 不自动 retry training；
- 不换 seed；
- 不增加 update；
- 不改 LR；
- 不改 target KL；
- 不增加 epoch；
- 不评中间 checkpoint；
- 不用 holdout33；
- 不补齐 599/600 结果；
- 不合并两个 attempt；
- 不在失败后加第二个 config。

遇到 failure：

1. 保留 run/eval 目录和 log；
2. 更新 `STATUS.json`；
3. 写 exact command、exit code、last valid artifact、hash；
4. 停止；
5. 等待 owner 明确授权。

---

# 21. Codex 严格执行顺序

1. 只读检查 main、records、dependencies 和 hashes；
2. 检查 V1.3-B 是否已经实现共享 guardrail/telemetry；
3. 仅实现一个 `v1_3_a` config 和缺失的最小共享遥测；
4. 静态验证与 one-update smoke；
5. 写 `PRECHECK.json`；
6. 写并提交 `PREREGISTRATION.json`；
7. 提交 implementation，确认 clean worktree；
8. 串行训练五 seed；
9. 验证所有 run、metrics 和 U8 checkpoints；
10. 运行一次 paired canonical BC；
11. 验证 BC 恰为 21/233/346/0；
12. 评价五个 U8；
13. 计算 paired metrics；
14. 应用过程 gate 和产品 gate；
15. 写 `RESULTS.json`、`FINAL_REPORT.md`、`STATUS.json`；
16. 提交 records；
17. 停止，不自动进入 holdout。

---

# 22. 提供给本地 Codex 的完整 prompt

```text
Implement and execute End2Race PPO V1.3-A exactly from:
ppo_experiments/v1_3_a/IMPLEMENTATION_GUIDE.md

Repository:
/home/haowei/Documents/End2Race
branch main

This is a one-config experiment, not a sweep.
The only formal config is v1_3_a:
- n_envs 16
- n_steps 1600
- batch_size 1600
- n_epochs 1
- updates 8
- checkpoint_updates (8,)
- GRU LR 3e-6
- head LR 3e-5
- critic LR 3e-4
- target_kl 0.01
- post-update KL guardrail 0.02
- clip 0.10
- max_grad_norm 0.5
- C0 critic
- H0 p0.5 with replacement
- steering/speed std 0.05/0.15
- existing reward, margin disabled
- canonical BC fresh start

Formal seeds:
20260718, 20260719, 20260720, 20260721, 20260722

Only U8 is a formal candidate. Do not save/evaluate U2 or U4.
Do not add LR2/LR4 calibration runs.
Do not modify PPO loss, reward, critic, sampler, policy architecture,
environment, evaluator, hard pools, SB3, or the canonical BC.

First:
- pull main and require a clean worktree;
- read the authoritative V1.2/holdout/SG records;
- record all required hashes and dependency versions;
- detect whether V1.3-B already implemented identical guardrail/telemetry;
- reuse identical shared code if present;
- add only the v1_3_a config and missing minimal telemetry;
- run static checks and one temporary one-update smoke;
- write PRECHECK.json and PREREGISTRATION.json;
- commit before formal training;
- require clean worktree.

Train all five seeds serially before any candidate evaluation.
Stop the whole line on any post-update approx_kl > 0.020 or invalid run.
Run one exact paired BC evaluation with NUM_WORKERS=4 and require 21/233/346/0.
Evaluate only the five U8 checkpoints on the same canonical 600-case panel.

For every seed compute fixed_collision, new_collision, G,
gained/lost overtake, speed ratio, and distance ratio.
PASS requires 5/5 seeds to have:
- at least 6/8 updates with approx_kl in [0.002,0.010];
- all updates approx_kl <= 0.020;
- G >= 5;
- collision <= 16;
- overtake >= 340;
- speed ratio >= 0.99;
- distance ratio >= 0.99.

No retry, no seed replacement, no checkpoint selection, no threshold changes,
no extra config, and no holdout33.
Write STATUS.json, RESULTS.json, and FINAL_REPORT.md with all hashes and gates.
If development passes, stop and report that a new separately preregistered holdout is required.
```

---

# 23. 完成条件

V1.3-A 只有在以下全部完成后才算执行闭环：

```text
PRECHECK PASS
PREREGISTRATION committed before formal runs
implementation commit recorded
5/5 formal training runs terminal
5/5 U8 checkpoints valid 12-key
paired BC valid and exactly 21/233/346/0
5/5 candidate evaluations valid 600/0
paired metrics complete
RESULTS.json complete
FINAL_REPORT.md complete
STATUS.json terminal
selection_performed = false
holdout_performed = false
promotion_performed = false
```

“实验完成”不等于“模型改善”。
