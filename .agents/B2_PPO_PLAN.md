# B2 BC-Direct PPO：目标对齐训练与首次策略学习测量计划

状态：**APPROVED FOR MANAGED IMPLEMENTATION AND B2 PILOT — 必须先完成本文已纳入的 blocking fixes 与 preflight**

日期：2026-07-12（Asia/Singapore）

仓库基线：`chore/commit-evidence-pipeline` @ `c1a22b601b32a757da321e0761350c601cc0f794`

实验目录：`Experiments/B2_ppo_pilot/`（实现时创建；数值运行使用仓库外隔离 run root）

独立审计：`.agents/B2_PPO_REVIEW.md`（Claude Code `claude-opus-4-8` / max，
结论 `APPROVE_WITH_BLOCKING_FIXES`；blocking fixes 已前瞻性并入本文）

---

## 0. 本文解决什么问题

当前仓库有四处引用一个不存在且从未进入 Git 历史的文件：

```text
docs/superpowers/plans/2026-07-12-ppo-pilot-bc-direct.md
```

`Experiments/runner.py` 还登记了两个对应 CLI 尚不存在的 B2 job。它们
现在不可执行，也不能视为已审阅计划。

本轮先把 live `.agents/README.md` 导航改指向本文。以下 live 引用必须在
Task 0 统一修正：

- `.agents/HANDOFF.md` §20/§21 的 “plan is written” 叙事；
- `Experiments/INDEX.md` 的旧 plan path；
- `Experiments/runner.py` 的两个占位 B2 jobs。

`docs/ppo/TIER3_ORGANIZATION_RECORD.md` 已如实披露旧草案曾被删除，应保留
该历史记录。批准后按 Task 0 统一修正 live references。

本文重新定义 B2。它不是对旧 warm-start 阶梯的继续修补，而是一个
**BC-identical 初始策略 → on-policy PPO → deterministic paired KPI evaluation**
的最小闭环。它只保留会保护动作概率正确性或产品 KPI 的门。

本文不改变任何历史判定：

- D2 / D2R 仍按原门为 FAIL；TTC 在策略阶段仍是 diagnostic-only；
- B1 Task 10 仍为 FAILED；
- hierarchical replacement Task 6 仍为 acceptance-FAILED；
- 所有旧 warm-start checkpoint 均不具备 B2 PPO 资格；
- D2 sealed test、fresh Austin pool、cross-map final pool 均保持未打开；
- 项目所有者已明确要求 Codex 托管执行 B2、审计以及结果后的目标对齐优化；
  因此本文授权 Tasks 0–7 和冻结的 20-iteration pilot，但不授权打开 D2 test、
  fresh/final pools，也不授权回到第三轮 warm-start 或新的 proxy probe。

---

## 1. 项目目标与本次实验问题

### 1.1 用户固定的词典序目标

1. **硬约束**：corrected overtake rate 不低于 BC；
2. 在满足 1 的候选中，降低 any-agent collision rate；
3. 产品目标：相对 BC 的 collision `RR <= 0.70`；
4. ego-involved collision、confirmed safe pass、interaction attempt 和 TTC
   只作分解或诊断，不能替代前两项。

### 1.2 B2 的可证伪问题

```text
H_direct:
  在 deterministic behavior 初始与 BC 完全一致的条件下，给层级残差
  policy 一个正确记账、足够但受控的训练期探索分布，PPO 能否学习出
  “超车非劣且碰撞减少”的 deterministic policy？

H_sidecar:
  作为完整 representation arm，预训练且冻结的 B 是否比随机初始化、可训练
  BC adapter 的 A 得到更好的 collision/overtake frontier？这不是只改变
  “是否冻结”的纯因果消融。

H_adapt:
  在其余条件相同下，C 的低学习率 sidecar adaptation 是否进一步优于 B？
```

这直接检验 supervised warm-start 是否是 PPO 的必要前提。

解释是单向的：

- B2 成功：足以证伪“必须 warm-start 才能学”的必要性主张；
- B2 失败：只说明**这一版**探索、回报、优化和训练预算没有成功，
  不能证明 warm-start 必需，也不能证明动作或表示空间不可行。

### 1.3 “首次测量”的准确措辞

历史 PPO、P1 和 B1 Task 10 已经记录过 collision/overtake。不得再写
“两周一次都没测过 KPI”。准确说法是：

> 尚无 B+ v2.2 PPO 学习后的 candidate 接受当前词典序开发门，更没有
> candidate 在 fresh final pool 上证明产品目标。

B2 的 20-iteration 结果是首次直接测量 **v2.2 PPO 学习是否朝产品 KPI
移动**，不是最终泛化或产品证明。

---

## 2. 为什么停止 warm-start 主线

### 2.1 已经得到的有效诊断

- D0.1 建立 primary `N=3036` 的 canonical evidence；
- D2.5 固定分支库在 91 个 non-test BC ego-collision cases 中找到
  67 个 confirmed-safe-pass witness；这证明所测案例存在有界残差解，
  不是理论天花板或全分布 RR；
- D2/D2R 证明 deployable observations 含风险信号，但不证明 PPO 一定能
  从稀疏回报学到；
- B1 Task 10 证明旧 warm-start 在真实闭环中会新增碰撞并损失超车；
- hierarchical remediation 修复了常开转向与有界合成，但自然采样的
  replacement Task 6 又出现 positive recall 0。

### 2.2 不再继续的原因

两轮 warm-start 的失败支持“训练暴露与部署频率严重错配”，但没有严格
证明 67 witness 不可蒸馏，也没有证明 balanced schedule 不会成功。
停止的理由是**决策价值**：无论再跑一轮 Task 6 成功或失败，最有价值的
下一动作仍是直接运行一个正确的 PPO 闭环。因此不再把 supervised
imitation 当 admission gate。

### 2.3 前瞻性治理变更（owner 已批准）

owner 对未来 B2 做如下 override：

- 退休 v2.2 spec §6.2 和 §9.1.1 中“必须加载 accepted warm-start
  checkpoint 才能 PPO”的要求；
- 不重写旧 spec、旧 release 或旧 FAIL/PASS；
- B2 从 canonical BC + fitted sidecar initialization + fresh residual heads
  构造策略，绝不加载任何 Task-6 action-head checkpoint；
- Task 6/9/10 指标改为历史诊断，不再作为 B2 admission gate；
- 最低 correctness preflight 见 §10，产品决策只看 §12 的直接结果。

---

## 3. 当前代码事实：可复用与缺失

### 3.1 可复用原语

| 能力 | 现有位置 | B2 用法 |
|---|---|---|
| 四维层级 latent、sample/log-prob/entropy | `bplus_v22/remediated_model.py` | 直接复用并补 exploration context |
| 每 100 Hz 基于当前 BC headroom 的有界合成 | 同上 | 直接复用；外部 clipping 必须为零 |
| A/B/C policy 与参数隔离 | `bplus_v22/model.py` | 直接复用 |
| macro clock 与 variable-length aggregate | `bplus_v22/macro.py` | 直接复用 |
| variable-discount GAE 骨架 | `bplus_v22/buffer.py` | 新 B2 buffer 调用该纯函数 |
| collision/performance advantage、dual、selector | `bplus_v22/objective.py` | 增加 clipped PPO surrogate 后复用 |
| critic MLP 网络骨架 | `bplus_v22/model.py::V22Critics` | B2 建立两个目标专用 heads |
| deterministic paired transition evaluation | `bplus_v22/hierarchical_closed_loop.py` | 解耦 warm-start loader 后复用 |
| v1 clipped PPO/recurrent replay 参考 | `train_ppo.py` | 只迁移通用算法，不复用旧动作/单回报语义 |

### 3.2 尚未完成的 B2 功能

以下缺一项都不能称为“只差一个循环”：

1. `MacroRecord` 尚未保存四维 latent、actor replay inputs、critic input、
   exploration offsets、schema、episode boundary 和 scenario identity；
2. `detached_actor_objective` 是 `-logp*A`，不是 clipped PPO ratio；
3. 没有 actor/two-critic 的 minibatch update 与分组 optimizer；
4. 没有对 frozen BC recurrent feature、LiDAR/scalar histories 的 exact replay；
5. 没有 12D privileged critic feature 的 live builder；
6. 没有训练 checkpoint/resume（policy、critics、optimizers、dual、RNG、
   scenario cursor、exploration schedule）；
7. 没有 stochastic zero-learning exploration evaluator；现有 Task 10 actor
   调用 `deterministic()`；
8. 没有通用 PPO snapshot paired evaluator；现有 evaluator 绑定 warm-start release；
9. `bplus_v22.cli` 没有 `exploration-sweep` 或 `ppo-pilot`；
10. 当前 B2 runner job 会启动不存在的命令，且错误地把一个 learner
    标成 shardable；
11. 当前 local job 使用 `sys.executable`，可能落到没有 torch 的 base
    Python，而不是 pinned `end2race` interpreter；
12. 当前没有 B2 artifact 目录、配置、状态或报告格式。

---

## 4. Policy 与三臂设计

### 4.1 共同底座

- canonical BC checkpoint：`pretrained/end2race.pth`；
- BC backbone 在 20-iteration pilot 中冻结；
- residual macro cadence：10 Hz，latent held for `K=10` micro-steps；
- 每个 100 Hz micro-step 重新相对当前 BC command 做 headroom projection；
- steer residual budget `0.2 rad`；
- brake residual budget `1.0 m/s`；
- positive speed residual budget `0.0`；
- fresh top intervention bias 与 conditional brake bias 均保持 `-6.0`；
- deterministic evaluation 的所有 exploration offsets 必须为零；
- primary deterministic deployment rule 在任何 outcome 前冻结为 centered
  threshold：raw top logit 严格大于 fresh `-6.0` 才 INTERVENE，且在已介入时
  raw conditional-brake logit 严格大于 fresh `-6.0` 才 BRAKE；continuous latent
  使用 learned mean；
- fresh zero-weight/equal-bias policy 因严格 `>` 而选择 NO_OP，并与 BC 行为
  一致；标准 Bernoulli mode（raw logit `>0`）只作 diagnostic，不参与选择。

这个合同解决“20 轮 PPO clip 无法把 `-6` 推过 0”的可达性问题，同时不改
fresh identity、不从 development outcome 校准 threshold。primary report 必须把
centered threshold 写入 policy schema，不能把它描述成标准 mode。

### 4.2 三臂

| Arm | Actor representation | PPO 可训练部分 |
|---|---|---|
| A `BC_FROZEN` | frozen BC feature adapter | adapter + action core/heads/std |
| B `SIDECAR_FROZEN` | fitted D2R sidecar | action core/heads/std；sidecar frozen |
| C `SIDECAR_FINETUNE` | 同一 fitted D2R sidecar | action core/heads/std + sidecar encoder，LR = action LR / 10 |

B/C 必须从同一个 `sidecar_init_20260712_080012/sidecar_bundle.pt` 初始化。
三臂 action heads 都 fresh，不使用 warm-start action weights。

### 4.3 不在本 pilot 改动的变量

- 不解冻 BC；
- 不开放 positive speed；
- 不重新训练 TTC head；
- 不加入 BC anchor sweep；
- 不改变 macro cadence；
- 不做 reward-weight sweep；
- 不打开 D2 test 或 final pool；
- 不运行第三轮 supervised warm-start；
- 不把 A/B/C 以外的新模型家族加入同一轮。

---

## 5. 训练期探索：必须是完整层级行为分布

### 5.1 现有单 offset 为什么不够

`apply_intervention_logit_offset()` 只改变 top-level intervention logits。
conditional brake gate 仍是 `-6.0`。若 top 总 logit 变为 `-2`：

```text
P(I=1)       = sigmoid(-2) * 100% = 11.9203%
P(B=1 | I=1) = sigmoid(-6) * 100% = 0.2473%
P(brake)                         = 0.02947%
```

400 macro decisions 期望只有 `0.118` 次 brake，而非 48 次；其余 intervention
主要成为随机 steer。单一 top offset 不能解决刹车探索。

### 5.2 B2 exploration contract

新增不可训练的 `BehaviorExplorationConfig`：

```text
intervention_logit_offset
conditional_brake_logit_offset
steer_std_scale       # pilot 固定为 1.0
brake_std_scale       # pilot 固定为 1.0
schedule_id
```

它必须作为 distribution construction 的显式、非持久化 context 传入；不得
调用历史 `apply_intervention_logit_offset()` 去修改 policy registered buffer，
也不得修改两个 gate bias。top logits 使用 raw head + transition top offset，
conditional brake logits 使用 raw head + transition brake offset。context 不进入
`state_dict` 或 optimizer；policy checkpoint 中历史 calibration offset 必须为 0。

要求：

1. 两个 offsets 只在 stochastic rollout distribution 中生效；
2. 同一 rollout 收集和其 PPO epochs 内 offsets 不变；
3. 每个 macro record 保存两个 offsets 与 `schedule_id`；
4. replay 用该 transition 原 offsets 重建新 log-prob；
5. deterministic evaluation 使用精确零 offsets；
6. checkpoint 保存当前/下一 iteration schedule，但 base policy 参数不吸收
   一个未声明的 eval offset；
7. resume 后第一条 rollout 的分布必须与中断前计划一致；
8. 同一 latent 在 unchanged policy + unchanged offsets 下必须满足 float32
   replay bound：`max |delta log_prob| <= 1e-4`、`max |delta entropy| <= 1e-4`；
   batch=1 rollout 与 batched GEMM 不要求 bitwise equality，实际最大漂移和
   `max |ratio-1|` 必须逐 iteration 记录；
9. 训练日志分别报告 top intervention、conditional brake 和 unconditional
   brake frequencies；不得把 intervention 次数写成 brake 次数。

不采用未记账的 epsilon override，不在 simulator 外强制刹车，不把探索
动作伪装成 policy action。

### 5.3 固定 exploration schedule：不再新增 sweep

为避免“先造一个 exploration selector，再由它阻止 PPO”，B2 不运行
outcome-based prior sweep，也不新增 `exploration-sweep` 数值 job。fresh logits
是常数，所需 offsets 可解析计算；真实 exposure 直接在 PPO rollout 中测量。

在任何 B2 development 结果前冻结共同 schedule：

```text
target P(I=1)       = 0.10
target P(B=1 | I=1) = 0.50

full intervention offset = logit(0.10) - (-6.0)
                         = 3.8027754227
full brake offset        = logit(0.50) - (-6.0)
                         = 6.0

iterations  1--5: multiplier = 1.0
iteration       6: multiplier = 0.8
iteration       7: multiplier = 0.6
iteration       8: multiplier = 0.4
iteration       9: multiplier = 0.2
iterations 10--20: multiplier = 0.0

all iterations: steer_std_scale = 0.1
all iterations: brake_std_scale = 1.0
```

每个 iteration 的两个 offsets 都等于 `full_offset * multiplier`。降低 steering
sampling scale 只抑制随机横向扰动，不删除 learned steer mean 或转向动作空间。
初始 joint
brake probability 是 5%，即 1,024 macro rollout 期望约 102 次 interventions、
51 次 brake-containing actions；这才同时解决顶层和条件刹车探索。

只允许两种运行前验证：

1. 无 simulator outcome 的 Monte Carlo sampler test，检查 top/conditional/
   joint frequency 与解析概率一致；
2. §10 P3 的 4-scenario plumbing smoke，检查 latent、log-prob、执行与
   checkpoint，不读取或比较 collision/overtake 结果。

不得看完 288 development、training outcome 或某个 arm 后改变该 schedule；
不得补第二张 grid。若 Claude 要修改上述概率或 decay，必须在 owner 批准和
任何 B2 rollout 前完成。

---

## 6. Rollout 与 replay 数据合同

每个 macro transition 必须保存：

```text
scenario_id / l2_id / episode_id / macro_index
arm / training_seed / policy_iteration / checkpoint_schema
BC feature [1680]
LiDAR history [8,360]
scalar history [24]
privileged critic feature [12]
canonical latent [I,z_steer,B,z_brake]
old macro log_prob
intervention_offset / conditional_brake_offset / schedule_id
requested residual and per-micro applied composition digest
macro length k in [1,10]
discount = 0.997^k
collision_cost / overtake_reward
terminated / truncated
two current values and two truncation bootstrap values
episode_start / BC hidden reset marker
```

物理 command 不得替代 latent action。projection 非单射不构成 PPO 问题，
因为 PPO 概率测度定义在 canonical latent 上；但 sampled latent、stored
latent、replayed log-prob 和真正执行它所产生的 command ledger 必须一一对应。

### 6.1 episode-complete collector

每个 iteration 固定运行16个完整 episode：8个来自 collision-bearing cycle、
8个来自 remaining cycle，并在训练前冻结每 seed 的20×16 ordered L2。它不是
可切断 episode 的硬 macro capacity；实际 macro batch 随 terminal 长度变化。
同 seed 的 A/B/C 必须使用相同16个scenario，不能因某臂早终止而改变后续列表。

- 8 秒 simulator horizon 是 product task terminal：写入 corrected terminal
  overtake reward，两个 critic bootstrap 都为 0；
- 第一次 any-agent collision 是 task terminal：collision cost=1、bootstrap=0；
- collector 不制造 time-limit truncation；infrastructure abort / simulator crash
  是 integrity failure，不能伪装成 truncation；
- terminal reward 在 episode finalization 时回填到该 episode 最后一条 macro；
- 不丢 partial episode，不跨 update 保留一个缺少终局标签的 episode。

### 6.2 recurrent replay

- BC frozen feature 可在 rollout 中保存并 detached；
- A 可直接 replay 保存的 BC feature；
- B 可 replay frozen sidecar feature或重算，但统一实现应保存原始 histories；
- C 必须用保存的 LiDAR/scalar/BC feature 重算 sidecar，才能让梯度进入 encoder；
- episode boundary 重置 BC hidden 和 histories；
- short terminal macro 不 padding、不复制 log-prob；
- true any-agent collision 与 8 秒 product horizon termination bootstrap 都为 0；
- B2 正常运行不产生 collector truncation；legacy truncation纯函数只保留测试；
- opponent-only collision属于 primary any-agent cost，不能沿用 v1 中将其
  当普通 truncation 的语义。

---

## 7. 直接目标、critic 与 PPO loss

### 7.1 actor 使用的两个信号

不再把 TTC、alarm、warm-start loss 或 dense progress 写入 actor objective。

```text
collision_cost:
  该 episode 第一次 any-agent collision 所在 terminal macro = 1
  其余 macro = 0

overtake_reward:
  episode 完成后 corrected terminal overtake = true，则最后一个 macro = 1
  否则 = 0
```

confirmed-safe-pass、ego collision、interaction attempt、progress 和 TTC
全部独立记录，仅用于解释 paired transitions。

### 7.2 critics

- `collision` critic 只拟合 collision return；
- `performance` critic 只拟合 corrected-overtake return；
- B2 新建精确 two-head `B2Critics`、two-channel loss 与 `B2MacroBuffer`，只
  实例化这两个 actor-driving critics；legacy `reward` critic 和
  `reward` buffer channel 保留给历史 B1/v1 兼容代码，但不进入 B2 model、
  optimizer、GAE 或 checkpoint；
- 12D privileged critic input 前瞻性锁为：
  `rel_s/6`、`lat_gap`、`ego_v_s/10`、`opp_v_s/10`、`ego_d`、`opp_d`、
  `safe_pass_hold/0.7`、`overtake_started`、`safe_pass_held`、
  `opponent_speedscale`、`sin(track_phase)`、`cos(track_phase)`；
- 所有字段只使用当前或历史 simulator state，不含未来 outcome；
- episode reset 清空 hold/started/held state；任何 privileged field 不得进入
  actor tensor，actor loss 对 critic input 必须严格断梯度。

### 7.3 advantages 与 dual

继续使用 variable-length macro GAE，但不用每 batch 标准差把空 collision
channel 的 critic 噪声放大成单位信号。performance 独立归一；collision 使用
只由至少包含一个 observed collision terminal 的 rollout 更新的 running scale。
首个 event-bearing rollout 前或 scale 非法时，collision actor advantage 置零。
running scale 与更新计数必须 checkpoint/resume。

actor 合成：

```text
gamma_micro = 0.997
discount_k  = 0.997^k
lambda_macro = 0.99
A = (-scale_collision(A_collision) + dual * normalize(A_overtake)) / (1 + dual)
```

dual：初值 1、上界 3、LR 0.5、EMA 0.2。iteration 1–9 只要 exploration
multiplier 非零就固定为 1，不调用 update。iteration 10 起每个完整 rollout
恰好调用一次 update，并仍要求累计完成至少 32 个 training episodes。

dual floor 在训练前从已打开的 canonical BC outcome 按 ordered L2 join 到
每个 seed 的冻结20×16 curriculum，一次性冻结为该 seed 的 BC corrected
overtake rate 减 1 percentage point；P0 simulator identity 负责证明 live BC 与
canonical baseline 一致。训练中不得重算或随 arm 改变。同 seed 三臂共享 floor。
dual 只用于平滑训练，checkpoint 选择仍直接看 paired outcomes。

Implementation timing is frozen as follows: iterations 1–9 do not call or
advance the dual. Iteration 10 contributes the first 16 zero-exploration
episodes and iteration 11 contributes the next 16; the first eligible dual
change occurs after the iteration-11 on-policy update and can first affect the
iteration-12 actor update. Exploration-contaminated episodes do not count
toward this 32-episode warm-up, and a rollout's outcome never changes the dual
used to optimize that same rollout.

### 7.4 clipped PPO

新增真正的 PPO surrogate：

```text
ratio = exp(new_log_prob - old_log_prob)
L_actor = -mean(min(ratio*A,
                    clip(ratio,1-eps,1+eps)*A))
```

初始提议值（Claude 审阅后、运行前锁定）：

```text
rollout_batch    = 16 complete episodes / iteration (8 collision + 8 remaining)
ppo_epochs       = 3
minibatch_size   = 128; deterministic keyed shuffle; keep final short batch
clip_eps         = 0.05
action_core_lr   = 3e-5
gate_mean_std_head_lr = 3e-4
sidecar_lr       = 3e-6 (C only)
critic_lr        = 5e-5
entropy_coef     = 0.001
max_grad_norm    = 0.5 per disjoint optimizer group
target_kl        = 0.03
optimizer        = Adam(betas=(0.9,0.999), eps=1e-8)
collision_scale_ema_decay = 0.99 (只在 event-bearing rollout 更新)
```

每个 epoch 前重建 hierarchical distribution；记录 pre/post-update KL、
clip fraction、ratio min/mean/max、各 head entropy、各 optimizer group
gradient norm。任何 NaN/Inf、schema mismatch 或 unchanged replay 超过上述
`1e-4` float32 bound
属于 integrity failure。

actor optimizer groups 必须显式互斥：adapter/action core、gate/mean/std heads、
C-only sidecar；BC 与 frozen sidecar 既不在 optimizer 也不得产生 grad。更高的
head LR 让状态依赖在 20 轮内可测，但 primary centered threshold 已消除必须
把 bias 从 `-6` 推到 0 的不合理要求。

---

## 8. Checkpoint / resume 合同

每个不可覆盖 checkpoint 至少包含：

```text
schema/version
git commit and dirty-state declaration
arm / seed / iteration / scenario cursor
policy state (base exploration offsets 必须为 0)
two critics state
actor/critic optimizer states
dual state and completed episode count
keyed action-noise schema/counter and deterministic minibatch-shuffle cursor
torch CPU/CUDA and numpy/random RNG states used outside keyed action sampling
behavior schedule and next-iteration schedule
training manifest/config path
completed rollout/update counters
```

恢复测试必须证明：从同一 checkpoint 继续的下一次 scenario、sampled latent、
old log-prob 和 update 顺序与不中断路径一致。checkpoint 不保存一个会让
deterministic evaluation 偏离 BC/learned base policy 的隐式 exploration offset。

action sampling 不依赖调用顺序变化的全局流。每个 component draw 使用冻结 key
`(pilot_seed,l2_id,repeat,macro_index,component)`；top、steer、brake gate、brake
latent 四个 draw 即使 inactive 也全部生成。global RNG state仍保存用于网络或
optimizer 的其他随机行为，但不能替代 keyed action contract。

---

## 9. 数据划分与泛化边界

### 9.1 已打开的数据

- canonical source 固定为
  `Experiments/B1_route_r2_scaffold/artifacts/task8_manifests_20260712_113241`；
  不得使用带旧 `manifest_order` 缺陷的 `...103509` release；
- PPO training universe：该 release 的 `training_scenarios.tsv` 全部 1,640 rows；
  通过已打开的 D2 non-test metadata 按 L2 join，冻结出 81 个 BC any-agent
  collision rows（61 个 L4）和其余 1,559 rows。每个 iteration 分别从两个
  独立 keyed、无放回 cycle 取8个 episode，交替组成16-row list，形成精确50%
  collision-bearing / 50% remaining curriculum；cycle 用尽后按相同 seed
  进入下一 repeat；
- development：Task-8 的 288 rows；
- D2 sealed test：不使用；
- fresh Austin/cross-map final：不使用。

TSV 中迁移前冻结的 `npz_relpath=eval_results/...` 已不再是 live path，不能
作为 simulator 运行依赖。B2 必须只用 manifest 的 map、raceline、
speedscale、resolved ego/opponent identity 等字段重建场景；NPZ 字段只作
历史 provenance。任何代码若尝试沿旧路径读取 NPZ，应 fail closed。

collision 标签只用于 scenario curriculum，不进入 actor/critic tensors。训练前
报告实际 81/1559 join、L4/map/skill counts、ordered L2 digest 与每 seed 前20轮
scenario order。288 development panel 不做同样 enrichment。

### 9.2 288 development 的边界

288 rows 是 L2-disjoint 的 opened-development panel，但不是 L4 policy
generalization evidence，且 192 rows 是 skill mechanism enrichment。B/C 的
sidecar initialization 使用过全部 1,928 non-test episodes，因此这 288 对
表示层也不是 held-out；它只能检验 action/PPO closed-loop mechanism。

因此它可以：

- 在预注册的 primary iteration 比较 snapshots/arms；
- 比较 A/B/C；
- 判断是否有可继续的机制信号。

它不可以：

- 宣称 fresh/generalized `RR <= 0.70`；
- 把两个 training seeds 在同一 288 scenario 上的结果当成 576 个独立场景；
- 在看完结果后改变 exploration schedule、reward 或 gate。

---

## 10. 最低必要 preflight

这些是 correctness checks，不是新的科学代理门。

### P0 — fresh deterministic identity

- offsets=0；
- primary centered threshold 下 A/B/C 全部因 raw logits 恰等 fresh `-6.0`
  而自然选择 top NO_OP；
- action/trajectory/outcome 与 BC 一致；
- fitted B/C sidecar 初始化相同；
- 不重新制造历史 hash chain。

### P0b — BC-only opened-development baseline reproduction

在六个 learner 启动前，本地只运行一次冻结 BC 的 288-row Task-8 panel；不加载
candidate、不训练、不排名。必须完整得到 `24` 次 any-agent collision 与 `138`
次 corrected terminal overtake，并保存 ordered L2/outcome/trajectory digest marker。
该 marker 绑定 training RunPlan/source commit，复制到远端 control root；两端
preflight 都必须验证它。任何偏差先阻断 learner，避免训练完成后才发现 evaluator
或环境基准漂移。

### P1 — sampled / executed / logged consistency

对 NO_OP、steer-only、brake-containing 和 short-terminal cases：

- sampled 4D latent == stored latent；
- stored offsets == behavior offsets；
- composed command 不需外部 clip；
- production-shaped batch=1 collect -> batched replay 满足
  `max |delta log_prob| <= 1e-4` 与 `max |delta entropy| <= 1e-4`，并记录实际
  `max |ratio-1|`；
- declaration、execution、old/new log-prob 对同一 latent；
- top-only offset test之外，必须新增 joint brake frequency/log-prob test。
- behavior context 不修改 state_dict、gate bias 或 historical calibration buffer；
- centered deterministic threshold 的上下边界与 strict-equality NO_OP 有负向测试。

### P2 — macro/GAE/critic integration

- K=1 与手算 micro semantics 一致；
- K=10 只有一个 action/log-prob/ratio；
- terminal length 1–9 正确；
- collision 与 8 秒 product-horizon terminal zero-bootstrap 正确，正常 collector
  不生成 truncation；
- collision/performance 两 channels 不串 target；
- privileged critic 对 actor 断梯度；
- frozen BC 无梯度和 mutation。

### P3 — one-iteration simulator smoke

只运行 4 个 training scenarios、每臂一个很短的 rollout/update：

- 目的仅是验证 live simulator/environment 接线、production-shaped
  collect→replay→一次有限 update，以及 intervention、steer-only、joint-brake
  三条真实动作分支均被走通；
- live P3 不重复 checkpoint/resume 数值实验。严格 interrupted-vs-uninterrupted
  continuation 由 production-shaped runner regression 锁定：checkpoint reload 后
  比较下一 keyed action/log-prob、update diagnostics 及最终 policy/critic/
  collision-scale state；该测试仍是正式运行前的 blocking test；
- full exploration multiplier 的 top/conditional/joint 解析频率由冻结 sampler
  test 验证。4 个 live scenarios 只检查 branch presence，不用小样本频率判断；
- 一次 PPO update 内部必然消费 terminal collision/overtake 信号，但 P3 不把
  product outcome 写入 marker，不比较、不排名、不选 arm，也不据其改 reward、
  exploration 或任何配置；
- P3 通过后生成一份两端字节相同的 `READY.json`，绑定 RunPlan、source/input
  archive、BC baseline marker 与 P3 marker SHA；learner 在 GPU lock 内重新验证
  完整 extracted source/runtime-input 哈希、pinned Python/package environment、
  GPU identity 和 READY 后才可启动；
- 不选 arm、不调 reward、不调 exploration；
- 通过后直接进入正式 pilot。

---

## 11. 正式实验阶梯

### Stage A — 固定 schedule 正确性确认

- 只运行 §5.3 的 sampler test 与 §10 P3 plumbing smoke；
- 不形成 policy candidate；不得序列化、查看或比较 outcome 以做选择，也不得
  选择 offset；P3 update 对冻结 terminal signals 的内部消费不构成产品测量；
- 通过后直接使用冻结 schedule 运行六个 learners。

### Stage B — 20-iteration 两 seed 三臂 pilot

运行完整六个独立 learner：

```text
A/seed0  A/seed1
B/seed0  B/seed1
C/seed0  C/seed1
```

不得先用 seed0 淘汰 arm；两个 seeds 全部跑完 20 iterations。每臂使用相同：

- training scenario universe 与冻结的 50/50 dual-cycle curriculum；
- exploration schedule；
- rollout/update counts；
- PPO/dual constants；
- snapshot iterations；
- deterministic evaluation jobs。

同一 seed 下 A/B/C 使用相同 scenario order 和 keyed RNG。随机键至少包含
`(pilot_seed, l2_id, repeat, macro_index, component)`，使某个 arm 提前终止
episode 时不会把后续场景的随机流整体错位。critics 也按同一 seed 初始化。

primary development evaluation 只有 fresh iteration 0（centered threshold 下
共享 BC identity）和 iteration 20；iteration 5/10 可以保存 checkpoint 和 training diagnostics，
但不得打开 288、选择 arm 或改变训练。

iteration-20 primary 使用 §4.1 centered deterministic contract。标准 Bernoulli
mode 只输出诊断结果；不得在看到 outcome 后交换 primary/diagnostic 身份。

### Stage C — development selection / stop

对每个 arm 的同一个 snapshot iteration 合并两个 seeds；不得为每个 seed
选不同 iteration。按 §12 判断：

- 无 arm 有两 seed 同方向的 objective-aligned signal：B2 pilot 结束为
  negative，不自动回到 warm-start；
- 有 direction-pass 但未到 RR target：最多延长到固定 40 iterations；
- 到 40 仍无 target-pass：停止本设计，不开 final pool；
- 有 target-pass：冻结一个 candidate，另写 medium/final prospective plan。

本文不授权 40 iteration、medium 或 final；它们需要 Stage B 报告后重新批准。

---

## 12. KPI、统计与判定

### 12.1 每个 snapshot 必须报告

- complete paired episode count；
- BC/candidate any-agent collision counts；
- collision `RR = candidate / BC`；
- fixed collision / new collision；
- ego-involved collision breakdown；
- corrected terminal overtake counts/rates；
- gained overtake / lost overtake；
- collision → confirmed safe pass；
- collision → safe follow；
- overtake → follow；
- safe → new collision；
- interaction-attempt change；
- intervention/brake/steer exposure；
- map/skill/raceline/speed/L4 clustered slices；
- L4/scenario-cluster bootstrap interval，明确属于 opened-development。

### 12.2 20-iteration direction-pass

每个 seed 及 pooled 都必须满足：

1. `lost_overtake - gained_overtake <= floor(0.01 * N_gate)`；
2. `fixed_collision > new_collision`；
3. 至少一个 `collision -> confirmed safe pass`；
4. 改善不能全部由 interaction attempts 消失解释；
5. 无完整性、clipping、replay 或缺失 episode 问题；
6. 两个 seeds 的 net collision improvement 同方向。

这只允许延长机制实验，不是产品 pass。

报告标签固定为 `DEVELOPMENT_SURVIVOR`，不得写成 target hit。

### 12.3 development target-pass

288 panel 的 frozen BC 基准是每个 training seed 重放相同的：

```text
BC any-agent collision = 24 / 288
BC corrected overtake  = 138 / 288
```

除 direction-pass 外，`OPENED_DEV_KPI_POINT_TARGET_HIT` 同时要求：

```text
per-seed candidate collision <= 16   # 16/24 <= 0.70; 17/24 > 0.70
pooled candidate collision   <= 33   # baseline count 48 across two replicates
per-seed candidate overtake  >= 138  # strict no decline
pooled candidate overtake    >= 276
```

两个 seeds 重复相同 288 scenarios，pooled `N=576` 不是 576 个独立场景；
pooled count 只是两个 training replicates 的汇总。必须同时报告每 seed 的
RR/RD、两 seed 范围与方向，不允许一个 seed 恶化被 pooled 值掩盖。

开发阶段保留 1pp overtake 容差用于噪声控制；最终产品门仍是 corrected
overtake point count/rate 不低于 BC，不使用 1pp 放宽。

统计报告使用 L4 为 resampling unit 的 paired cluster bootstrap，给 collision
RR one-sided 95% upper bound 与 overtake RD one-sided 95% lower bound；只有
两个 training seeds，不做跨 seed t-test。开发 CI 只作描述，不能包装成
fresh generalization。

### 12.4 选择顺序

1. 丢弃 overtake-infeasible candidates；
2. 丢弃 integrity failure；
3. direction-pass / target-pass 分层报告；
4. 同层中选择 paired net collision improvement 最大者；
5. tie 才看 net overtake、较早 snapshot、固定 arm order；
6. TTC、Brier、alarm、training loss 不参与排序。

---

## 13. 本地/远端运行拓扑

### 13.1 现有 `split` 为什么不能跑 learner

当前 `_split` 只是启动四个独立进程，没有 central learner、gradient
all-reduce、rollout gather 或 checkpoint merge。三个 remote shards 还会在
同一 GPU、同一 output path 上并发写入。一个 on-policy learner 不能这样
拆分后拼接。

### 13.2 正确并行方式

- PPO training：以完整 `arm × seed` 为 job，不做 scenario shard；
- 同一 training seed 的 A/B/C 必须在同一设备上依次运行，避免 arm 与
  device 完全混杂；
- 两 seed pilot 的首选分配是 remote 完整运行 seed0 的 A/B/C queue，local
  完整运行 seed1 的 A/B/C queue；两台 GPU 各自同时最多一个 learner；
- 这个两-seed 设计无法同时严格满足 1/4:3/4 算力比例。科学配对优先于
  强行凑比例；若未来预注册四个 seeds，才按一个完整 seed queue 本地、
  三个完整 seed queues 远端实现 1/4:3/4；
- device 与 seed 的对应关系必须在训练前冻结并报告。若两 seed 方向严重
  矛盾，属于预先定义的跨设备复核触发条件，而不是自动选择有利 seed；
- deterministic snapshot evaluation：可以按 scenario 分成 4 shards，
  每个 shard 的配对单元必须同时包含同一 scenario 的 BC 与全部待比较
  arm/snapshot；local 运行 shard0，remote 用一个进程顺序运行 shards1–3，
  每 shard 写唯一 host-specific 目录，最后显式 fetch/merge/complete-check；
- evaluation 前把该 seed 的冻结 checkpoint set 同步到两台 host，并记录
  checkpoint identity/digest；这不是递归 hash manifest，而是确保两个 shards
  实际加载同一模型。evaluation offsets 必须为零；
- 1/4:3/4 是吞吐目标，不是让同一张 remote GPU 同时起三个进程。

### 13.3 runner 必须先修复

1. local job 固定使用
   `/home/haowei/miniconda3/envs/end2race/bin/python`；
2. remote 只允许 `ssh haowei@192.168.2.127`；
3. 每个 job/shard 有唯一 absolute `NUMBA_CACHE_DIR`；
4. 每个 job/shard 有唯一 output directory；
5. CLI 缺失时 `show` 就 fail closed，而不是执行时才报 argparse error；
6. training jobs `shardable=False`；
7. eval merge 检查 scenario Cartesian completeness 和重复 L2；
8. `./run.sh show` 与 `--dry-run` 先审阅，之后才可执行；
9. 不运行或修改远端 stale/dirty repository；从 clean local commit 创建唯一
   `git archive`，本地与远端都解包到仓库外
   `/home/haowei/end2race_runs/<run_id>/repo`；
10. `show` 生成显式 `run_id` 和冻结命令，后续 `run` 消费同一计划；不得
    因重新 import runner 而换 timestamp/output path；
11. job 声明 `gpu_exclusive`、`required_cli`、prerequisites 与允许 host；
12. runner 不负责自动 push、commit、打开 seal 或 final pool。
13. 只按 allowlist stage BC、完整 sidecar init release 和 corrected Task-8
    release；RunPlan 保存 source/input digest，禁止复制 D2 test/final pool、
    `eval_results` 或整个 `_archive`；
14. preflight 断言 `bplus_v22`、项目根模块、`f110_gym` 与 `latticeplanner` 的
    `__file__` 都解析到 staged repo，阻断 Conda editable install 泄漏；
15. remote simulator 固定 `DISPLAY=:1`，两端使用 pinned end2race Python、
    `CUBLAS_WORKSPACE_CONFIG=:4096:8`、独立 pycache/Numba/XDG/MPL cache；
16. 每台 GPU 用 UUID-named `flock` 独占整个 seed queue，并在拿锁后复查
    `nvidia-smi`；
17. training RunPlan 与 EvalPlan 都不可变且必须显式 `run_id`；`show`、stage、
    preflight、execute、collect 消费同一 plan，绝不在 import 时换 timestamp；
18. eval 前先回收并冻结六个 iteration-20 checkpoint digests；local shard0、
    remote shard1–3 串行，collect 拒绝覆盖；merge 硬验 288 scenarios×7 variants
    =2,016 rows、无重复/缺失，才可计算 KPI。

---

## 14. B2 文件与实现任务

### Task 0 — 批准与文档一致性

- Claude Code Opus 4.8/max 已只读审阅，blocking fixes 与 Codex 裁定见
  `.agents/B2_PPO_REVIEW.md`；
- owner 已明确批准 managed B2 与 prospective override；
- 本文状态已改为 APPROVED；
- 更新 `.agents/HANDOFF.md`、`.agents/README.md`、`Experiments/INDEX.md`；
- 删除“plan already written”的错误叙事，或指向本文；
- 不修改冻结历史 spec，只在 live authority 记录 override。

### Task 1 — exploration API

修改/新增：

- `bplus_v22/exploration.py`
- `bplus_v22/remediated_model.py`
- `tests/test_bplus_v22_exploration.py`

实现双 offset context、概率转换、schedule serialization、joint log-prob
和 deterministic-eval zeroing；实现 keyed component sampling 与 centered
deterministic policy，历史 persistent offset API 只保留兼容。

### Task 2 — replay buffer 与 critic input

新增/修改：

- `bplus_v22/ppo_buffer.py`
- `bplus_v22/buffer.py`（只复用/必要时抽纯函数，不削弱旧 fail-closed record）
- `bplus_v22/model.py`
- `tests/test_bplus_v22_ppo_buffer.py`

新增 B2-only two-head critics、完整 MacroReplayRecord、episode-complete batch
collator、12D critic builder 与 recurrent boundary replay，不削弱 legacy 三通道
fail-closed API。

### Task 3 — clipped PPO update

修改/新增：

- `bplus_v22/objective.py`
- `bplus_v22/ppo.py`
- `tests/test_bplus_v22_ppo.py`

迁移 v1 的 ratio/clip/KL 思路，接入双 advantage、两个 critics、
独立 optimizers 和 frozen-parameter assertions。

### Task 4 — rollout environment 与 checkpoint

新增：

- `bplus_v22/ppo_env.py`
- `bplus_v22/ppo_runner.py`
- `tests/test_bplus_v22_ppo_rollout.py`
- `tests/test_bplus_v22_ppo_checkpoint.py`

职责：

- fixed-manifest scenario iterator；
- 50/50 dual-cycle curriculum 与 pre-training canonical BC floor join；
- 100 Hz BC recurrent step + 10 Hz latent sampling；
- per-micro composition ledger；
- terminal corrected outcome assignment；
- exact checkpoint/resume；
- one-iteration smoke。

### Task 5 — deterministic KPI evaluator

新增 `bplus_v22/ppo_eval.py`，从
`hierarchical_closed_loop.py` 提取通用 actor/evaluator，不能要求 warm-start
release。实现 BC cache、snapshot evaluation、shard result与merge。

### Task 6 — CLI / runner / experiment layout

修改：

- `bplus_v22/cli.py`
- `Experiments/runner.py`
- `run.sh`（仅在需要固定解释器时）

创建：

```text
Experiments/B2_ppo_pilot/
  README.md
  configs/
  runs/
  checkpoints/
  evaluations/
  reports/
```

该仓库内目录只保存计划、配置、报告与回收后的结果；实际 local/remote 运行
发生在不可变 RunPlan 指定的仓库外隔离 root。

CLI 只新增：

```text
capabilities
ppo-baseline-preflight
ppo-plumbing-smoke
ppo-pilot
ppo-evaluate
ppo-merge-eval
```

runner control plane 固定为：

```text
./run.sh plan / show / stage / baseline-preflight / preflight / plumbing-smoke
./run.sh execute / resume / status / collect
./run.sh plan-eval / merge-eval
```

`execute` 只消费已存在的 canonical JSON plan；不能根据动态 Job 重新生成命令。
RunPlan 摘要只用于操作一致性，不恢复历史递归 artifact hash chain。

删除当前虚假的 `b2-exploration-sweep` job；固定 schedule 不需要数值 sweep。
用六个 `shardable=False` 的 arm×seed training jobs 替换
`b2-ppo-pilot-seed0`。冻结 checkpoint 的 evaluation 才允许
`partition_mode=scenario_eval`，merge job 永不 shard。

新工作遵守当前政策：不再为每个 release 创建哈希链；但必须记录 git
commit/dirty state、完整 config、命令、环境、seed、exit status、completeness
和结果表。

### Task 7 — preflight、fixed-exploration smoke 与正式 pilot

严格按 §10–13 顺序。任何数值 run 必须在 Tasks 0–6 代码审阅、测试和 dry-run
通过后执行；owner 已授权 managed B2，不需要逐阶段再次询问。仍必须在 run
record 中写明 preflight verdict，任何 integrity failure 立即 fail closed。

---

## 15. 预注册停止条件

立即停止当前 job：

- sampled/stored/replayed latent 或 offsets 不一致；
- unchanged policy ratio 不为 1；
- external clipping 或 evaluator 改写 composed command；
- NaN/Inf、buffer boundary、bootstrap、scenario completeness 错误；
- frozen BC mutation、privileged actor leakage；
- checkpoint resume 无法复现下一 rollout 序列；
- D2 seal/final pool 被意外访问；
- staged source/input/plan digest 或模块实际解析路径不符合 job 声明。

结束 B2 设计且不继续调参：

- 固定六个 20-iteration jobs 完成后，没有 arm 在两个 seeds 都满足
  direction-pass；或
- 唯一预注册的 40-iteration extension 完成后仍无 target-pass；或
- 同一类实现缺陷修复后重复出现三次，且无法保证概率/执行正确性。

不得因以下原因单独停臂：TTC、Brier、alarm recall、warm-start recall、
单 seed、training loss 不好看、dual 暂时振荡。

除完整性故障外，iteration 5/10 不进行 development KPI evaluation，也不
提前停臂；六个 arm×seed jobs 都跑满 20 iterations。若实现 bug 使一组结果
无效，修复后所有受影响的 arm×seed 从 iteration 0 重跑，不能保留有利部分。

不得在 B2 失败后自动启动：第三轮 warm-start、第六个探针、reward sweep、
第二套 exploration schedule、BC 解冻或 positive-speed 实验。这些都需要新的
owner 决策和新的问题定义。

---

## 16. Claude 审阅记录

以下清单已由 Claude Code `claude-opus-4-8` / max 只读审阅。完整裁定与
blocking fixes 见 `.agents/B2_PPO_REVIEW.md`；结果为
`APPROVE_WITH_BLOCKING_FIXES`，不是新的 GPU gate。

下列问题保留为实现 review checklist：

1. 退休 warm-start admission gate 是否被准确记录，而未改写旧结果？
2. B2 success/failure 的非对称解释是否正确？
3. 双 gate exploration 是否完整，joint brake 概率是否计算正确？
4. `P(I)=0.10`、`P(B|I)=0.50` 与 1.0→0 decay 是否足够且不过度？
5. exploration 选择规则是否避免 288 development double-dip？
6. MacroReplayRecord 是否包含 PPO exact replay 所需全部字段？
7. collision/overtake terminal signal 是否与产品 KPI 一致？
8. B2 不实例化 legacy reward critic、仅使用 collision/performance 两 heads
   是否正确？
9. 12D critic input 的精确字段应是什么，是否存在 actor leakage？
10. PPO hyperparameters 是否应沿用提议值，哪些必须在运行前更改？
11. 两 seed 全跑是否优于 seed0 survivor filter？
12. direction-pass 与 target-pass 是否清晰区分？
13. 288 development 的 CI/claim boundary 是否准确？
14. 本地/远端任务拓扑是否避免独立 learner 被错误拼接？
15. 是否还有任何门无法回答“它挡住时会保护交付物什么”？

---

## 17. 审阅后的唯一下一步

在 Claude 和 owner 批准前：**不实现、不运行、不同步远端。**

批准后只进入 Task 0–6 的代码实现和 correctness tests；完成后报告 dry-run
命令与测试证据，再请求 fixed-exploration smoke/pilot 数值执行授权。不得把本文的
“方向获批”解释成自动 GPU 授权。
