# B4 Plain-End2Race Direct-Head PPO：外部审计与下一阶段前瞻性实验协议

状态：**SUPERSEDED AUDIT DRAFT — owner 已于 2026-07-13 作出 D1-B / D2-B 裁定**

> 本文件保留为外部审计草案及意见演化证据，不再是 B4 的生效执行规格。
> 生效 authority 是 `.agents/B4_DIRECT_HEAD_PPO_PLAN.md`。Owner 接受本文大部分
> correctness 修正，但否决首轮自定义 truncated Gaussian 和 episode-weighted
> critic，改用 raw Normal latent + 固定 actuator projection，并明确只让 actor
> objective/advantage/KL 使用 episode-equivalent weighting。不得从本草案抽取与正式
> 计划冲突的参数或 gate。

日期：2026-07-13（Asia/Singapore）

当前源码边界：`4b06b7a`（分支 `chore/commit-evidence-pipeline`）

父级数值证据：

- B2 training RunPlan：`b2_direct_20260713_081422`
- B2 canonical EvalPlan：`b2_eval_20260713_165800`
- B3 实现边界：`19e83ae`、`21085bc`；已审阅 GO，但尚无 RunPlan 或数值结果

本文原始用途：提供给外部审计专家，回答两件事：

1. 外部提出的“移除 residual、直接微调原始 `End2Race.output_layer`”是否在理论、
   当前仓库事实和项目进度上成立；
2. 若将其作为下一条产品候选路线，什么样的实验合同才足以产生可审计结果。

本文不是 B3 失败结论，也不是删除 residual 历史代码的授权。B2/B3 文档、源码、
checkpoint 和运行证据必须保持原样。

---

## 0. 审计结论摘要

### 0.1 总体裁定

外部建议的**核心架构判断正确**：PPO 不要求 residual 参数化；从 BC
checkpoint 初始化原始 actor，再直接更新原有参数，是合法且常见的 policy-gradient
fine-tuning 路线。plain `End2Race.state_dict()` 也确实比 residual/B3 checkpoint
具有更强的原始 evaluator 兼容性。

但外部方案按当前文字**尚不具备直接开跑资格**。它混合了一个正确的架构方向、
若干正确的仓库观察，以及数个尚未闭合的实验合同。最主要的四个阻断项是：

1. **产品门冲突**：当前 authority 要求 corrected overtake 不低于 BC；外部方案
   改为最多下降 5%。这是目标变更，不是超参数细化，必须由项目所有者前瞻性确认。
2. **动作概率合同不完整**：无界 `Normal(mean, fixed_std)` 加 evaluator/environment
   clipping 不能在本项目中默认视为“采样动作 = 执行动作 = 记账动作”。`bound loss`
   只约束 mean，不限制 sampled action。
3. **表示论证自相矛盾**：外部方案正确指出冻结 BC feature 可能缺失风险信息，
   但首轮 head-only PPO 同样只读取冻结的 BC GRU feature，因此不能声称它已经解决
   representation 问题。它只能作为更简单、更兼容的第一控制实验。
4. **标量 reward 不保证 guardrail**：`2 * collision_delta + overtake_delta` 是合法
   标量目标，但既不等价于词典序约束，也不保证 5% overtake 门。它仍可能通过
   `collision -> safe follow` 和减少交互获得更高回报，重现 B2 的主要失败形态。

因此本文的建议是：**保留 B3 为已实现但未运行的开放假设；把 plain-End2Race
direct-head PPO 定义为 B4 challenger。先完成外部审阅和两个 owner decision，之后
只能选择一个前瞻性 RunPlan 开跑，不得边看结果边改变目标、reward 或动作分布。**

### 0.2 逐项裁定

| 外部主张 | 裁定 | 审计限定 |
|---|---|---|
| Residual 不是 PPO 必要组成 | **正确** | PPO 定义 surrogate/update，不规定 actor 必须输出 residual |
| BC 初始化后直接 policy-gradient fine-tuning 合法 | **正确** | 引用的 DAPG 使用 BC 初始化后 NPG，并带 demonstration-augmented loss；它支持路线可行性，不证明本方案 `BC anchor=0` 最优 |
| fine-tuned policy 可解释为相对 BC 的隐式 residual | **正确** | 控制意义成立；函数类不必包含“BC + 额外 MLP head”的全部函数 |
| residual checkpoint 不能被纯 `End2Race` strict load | **正确** | 当前 evaluator 的兼容来自 `res_head.*` 特判，不是 state_dict 同构 |
| head-only actor 约 0.707M、约占 6.3% | **正确** | 实际为 706,862 / 11,301,482 = 6.2546% |
| head-only 能解决冻结 representation 缺陷 | **不成立** | 它与 residual head 一样消费冻结 GRU feature；只是在容量、参数共享和 checkpoint schema 上不同 |
| 小 LR、PPO clip、target KL 等价于 residual hard budget | **不成立** | 它们限制单次优化或平均分布漂移，不提供逐状态硬动作预算，也不限制累计偏移 |
| action mean bound loss 能保证动作有界 | **不成立** | 非零方差 Normal 仍有越界概率；mean 在界内也不够 |
| paired terminal reward 的结果表 | **数值正确** | 作为 fixed-scenario action-independent offset，理想 policy-gradient 与 `-2C_pi + O_pi` 同方向；有限样本 critic/GAE 行为并不必然等价 |
| 旧 `gamma=.997, lambda=.95` 的 1 秒传播约 0.004 | **正确但对象有限** | 适用于 legacy 100 Hz `train_ppo.py`；当前 B2/B3 已使用 10 Hz variable-discount GAE、`lambda=.99`，其 1 秒系数约 0.670，不存在同一缺陷 |
| any-agent collision 应为 true terminal | **正确** | legacy v1 仍错；B2/B3 已经修复，不是 B4 首创 |
| actor KL early stop 不应阻断 critic | **正确** | legacy v1 与当前 B2/B3 update loop 都会发生这一耦合，B4 必须拆开 |
| 5% 门对应单 seed `>=132`、pooled `>=263` | **算术正确** | 与当前严格 `138/276` authority 冲突；若先要求每 seed `>=132`，pooled 实际至少 264，因此 `>=263` 是冗余门 |

---

## 1. 当前仓库事实基线

### 1.1 原始 actor 与 checkpoint

`model.py:9-116` 定义原始 `End2Race`：360D LiDAR 变换、previous-speed MLP、
GRU 和原始 `output_layer`。当前 canonical checkpoint：

```text
path   = pretrained/end2race.pth
sha256 = b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4
keys   = 12
strict End2Race load = PASS
```

按当前源码精确计数：

```text
End2Race total parameters = 11,301,482
output_layer parameters   =    706,862
share                     =      6.2546%
```

`output_layer` 是 `1680 -> 420 -> 2` 的两层非线性 head，不是一个只有两个 bias
的线性控制器。

### 1.2 residual 的实际兼容边界

`model.py:118-176` 的 `End2RaceResidual` 新增：

```text
res_head.*
residual_budgets
```

其 residual head 为 215,426 个参数，最后一层零初始化，因此 fresh deterministic
residual 为零。当前 `eval_multiagent.py:14-30` 会先检查 checkpoint 是否含
`res_head.*`，再在 `End2Race` 和 `End2RaceResidual` 间选择类。因此当前 residual
可运行，但这不是“纯 BC evaluator strict compatibility”。

plain direct-head checkpoint 若只保存 `actor.state_dict()`，可以保持与 canonical BC
完全相同的 key/shape schema；这一点是 B4 最强、最可验证的产品优势。

### 1.3 B2 已经做过什么

B2 不是随机策略训练，也不是 supervised warm-start。它从 canonical BC、fresh
residual/gate heads 出发，完成 A/B/C 三臂、seeds 0/1、20 iterations，并在冻结的
288 场景上评估 BC 加六个 candidate，共 2,016 行。

canonical B2 结果位于：

```text
Experiments/B2_ppo_pilot/evaluations/
  b2_eval_20260713_165800/merged/summary.json
Experiments/B2_ppo_pilot/evaluations/
  b2_eval_20260713_165800/merged/episodes.tsv
```

结果：

| Candidate | collision / RR | corrected overtake | fixed/new | gained/lost | 判定 |
|---|---:|---:|---:|---:|---|
| A seed0 | 26 / 1.083 | 124 | 11/13 | 7/21 | FAIL |
| A seed1 | 11 / 0.458 | 113 | 20/7 | 7/32 | FAIL |
| B seed0 | 17 / 0.708 | 118 | 14/7 | 6/26 | FAIL |
| B seed1 | 17 / 0.708 | 97 | 19/12 | 7/48 | FAIL |
| C seed0 | 8 / 0.333 | 95 | 21/5 | 1/44 | FAIL |
| C seed1 | 9 / 0.375 | 88 | 20/5 | 1/51 | FAIL |
| BC | 24 / 1.000 | 138 | — | — | baseline |

B2 完整性通过、外部 clipping 为 0，但六个 candidate 都未通过真实 overtake
方向门。尤其 C 的强碰撞改善主要来自减少交互和把 collision 变为 safe follow；
两个 seed 的 `collision -> confirmed safe pass` 都是 0。

这意味着 B4 不能把“去掉 residual”当作已经由数据证明的修复。它必须明确检验：

> plain full-action head update 是否能在不依赖 gate/residual composition 的情况下，
> 学到不是单纯抑制交互的安全策略。

### 1.4 B3 当前进度

B3 已实现并通过独立边界审阅，但没有 numerical RunPlan、GPU rollout 或结果。
其唯一主修是统一 sampling、old/new log-prob、checkpoint replay 和 deterministic
deployment 的 effective-logit policy。详见 `.agents/B3_PPO_PLAN.md:10-105`。

因此：

- 不能说 B3 已经改善 KPI；
- 也不能仅凭“residual 非必要”说 B3 的 policy-identity 假设已被否定；
- B4 同时改变 architecture、action distribution、reward、critic、GAE、curriculum、
  loss weighting、guardrail 和 iteration budget，不能作为 B3 的干净消融。

---

## 2. 理论与数学审计

### 2.1 PPO、BC initialization 和 residual 是三件独立的事

该区分正确：

```text
PPO                  = 如何从 on-policy data 计算和裁剪 policy update
BC initialization    = theta_0 从哪里来
Residual policy      = 动作/策略函数如何参数化
```

PPO 原论文只要求在旧策略采集的数据上优化 surrogate objective，并不要求 residual
head。Residual RL 原论文把传统反馈控制信号与 RL residual 叠加，明确是一种可选
分解。DAPG 则提供 BC pretraining 后继续 on-policy policy-gradient fine-tuning 的
成熟先例，但其 RL 阶段仍加入 demonstration gradient，不能被用来证明“完全不需要
任何 BC anchor”在 End2Race 上一定稳定。

论文入口：

- PPO：https://arxiv.org/abs/1707.06347
- Residual RL：https://arxiv.org/abs/1812.03201
- DAPG：https://arxiv.org/abs/1709.10087

### 2.2 “隐式 residual”解释成立，但不是容量等价

对任何 fine-tuned deterministic mean，可事后定义：

```text
Delta a_theta(s) = mu_theta(s) - mu_BC(s)
```

所以控制效果可以被解释为相对 BC 的隐式修正。它不证明原始 End2Race 的函数类
包含额外 residual MLP 的全部函数，也不证明两种参数化有相同优化几何。B4 应将
“更简单、checkpoint 同构”作为假设优势，而不是宣称 function class 更强。

### 2.3 representation 论证必须降级

外部建议对 residual 的批评是：若 residual 只读冻结 BC feature，它无法恢复 feature
中不存在的信息。这个条件判断成立，但 B4 head-only 首轮也使用完全相同的冻结
`gru_out`。差别只是：

```text
Residual:    frozen feature -> original output_layer + new res_head -> composition
B4 head-only frozen feature -> updated original output_layer          -> full action
```

因此 B4 首轮能回答的是：

> 在现有 BC feature 足够的前提下，直接重映射完整 action 是否比层级 residual/gate
> 更容易满足产品门？

它不能回答“原始 feature 是否充分”。若 B4 head-only 失败，下一步可以前瞻性设计
低 LR GRU 解冻；但不得在看到某个 seed 后临时解冻，也不得把 head-only 失败直接
归因为 representation。

### 2.4 paired reward 的正确解释

外部建议定义：

```text
DeltaR_pair = 2 * (C_BC - C_pi) + (O_pi - O_BC)
```

给出的八种 transition 分值均正确。对一个固定 scenario，`C_BC`、`O_BC` 与当前
sampled action 无关，因此在精确 REINFORCE 期望下，它与：

```text
R_train = -2 * C_pi + O_pi
```

只有 action-independent constant 的差别，期望 policy gradient 相同。

但把 `DeltaR_pair` 作为 terminal critic target，并不自动成为无副作用的 control
variate：有限 batch、近似 critic、GAE 和 curriculum reweighting 都会影响实际方差
和优化。特别是 BC-overtake 场景中，候选保持 overtake 时 `DeltaR_pair=0`；直接
`R_train` 则保留明确的 terminal overtake 正信号。

**B4 修订建议：**

- actor/critic 的唯一 terminal reward 使用 `R_train = -2*C_pi + O_pi`；
- `DeltaR_pair` 和外部建议的 transition 表完整记录为 paired diagnostic；
- 不把二者混在两个 independently normalized advantage channel；
- 不声称标量权重 2 能强制满足 overtake 门。

若外部审计坚持使用 `DeltaR_pair` 训练，必须说明其有限样本动机，并决定 critic
是否可见 `C_BC/O_BC`；否则本文默认采用更直接的 `R_train`。

### 2.5 reward 权重与 guardrail 是不同层次

`-2*C + O` 表示一次 collision 的标量代价大于一次 overtake 的标量收益，但它
仍是加权和，不是约束。一个通过大幅减少交互修复多个 collision、同时丢失若干
overtake 的策略仍可能提高训练回报。

因此 B4 的正确表述是：

> 本 pilot 检验这个简单标量目标在 direct-head 参数化下是否自然满足预注册
> overtake guardrail；若不满足，实验失败并停止，而不是宣称 reward 已保证约束。

本轮不做 reward-weight sweep、不自动加 dual、不事后加入 BC anchor。

### 2.6 GAE 数值核对

外部建议的 100 Hz 微步系数：

```text
gamma * lambda = 0.999 * 0.995 = 0.994005
1 s  : 0.994005^100 = 0.5481
2 s  :                    0.3004
3 s  :                    0.1647
5 s  :                    0.0495
```

legacy `train_ppo.py` 默认 `.997 * .95`，1 秒系数约 0.00438，外部批评正确。

但当前 B2/B3 已不是该流程：`bplus_v22/__init__.py:18-21` 使用 10-step macro、
`gamma_micro=.997`、`lambda_macro=.99`，`bplus_v22/buffer.py:66-117` 使用每个
macro 的 variable discount。其传播约为：

```text
per 0.1 s macro = 0.997^10 * 0.99 = 0.96070
1 s             = 0.66968
2 s             = 0.44847
3 s             = 0.30033
5 s             = 0.13469
```

所以 B4 的 `.999/.995` 是一个可冻结的新选择，但不能宣传为修复 B2/B3 的已知
credit-collapse；相比 B2/B3，它在 1–5 秒范围反而衰减更快。

---

## 3. 开跑前必须由 owner/外部审计明确的两个决策

### Decision D1 — 产品 overtake 门是否真的改变

当前仓库 authority（`.agents/README.md:19-25`、`.agents/B3_PPO_PLAN.md:10-22`）是：

```text
per-seed corrected overtake >= 138 / 288
pooled corrected overtake   >= 276 / 576
```

外部建议改为最多 5% relative loss：

```text
per-seed corrected overtake >= ceil(.95 * 138) = 132
pooled corrected overtake   >= ceil(.95 * 276) = 263
```

算术没有问题，但这是实质放宽。并且若两个 seed 都 `>=132`，pooled 自动 `>=264`，
所以 pooled `>=263` 不再携带额外约束信息。

外部审计必须在以下两项中选一项，写入批准记录；RunPlan 不允许同时保留两套门：

- **D1-A（默认、推荐）**：保持当前 strict non-decline，避免在 B2 失败后移动门；
- **D1-B（产品需求确已改变时）**：采用每 seed `>=132`，并将 pooled 描述门改为
  与 per-seed 一致的 `>=264`；同时明确旧 B2 判定不追溯改写。

若选择 D1-B，应写清业务理由，例如 collision reduction 已被提升为主目标且最多
5% overtake loss 可接受；不能只写“新方案更容易通过”。

### Decision D2 — B3 与 B4 的执行顺序

两种合法选择：

- **D2-A：先运行 B3。** 优点是 B3 已实现、已审阅，且只隔离 B2 已知的
  train/deploy policy mismatch；因果解释更干净。缺点是最终 checkpoint 不是 plain
  `End2Race`，若 strict checkpoint compatibility 是新硬需求，它只能提供诊断价值。
- **D2-B：暂停 B3 数值运行，先实现 B4。** 仅当“最终产物必须由原始
  `End2Race(...).load_state_dict(..., strict=True)` 加载”被确认为硬产品需求时推荐。
  B3 保留为未运行、未证伪的历史分支，不删除、不写成失败。

本文倾向 D2-B 的条件性理由是部署 schema 简化，而不是 residual 在数学上错误。

---

## 4. B4 可证伪问题与假设

### 4.1 主问题

> 从 canonical BC 初始化、冻结 `k/speed_mlp/dummy_embedding/GRU`、只训练原始
> `output_layer` 时，完整动作 PPO 能否产生一个 plain-End2Race checkpoint，
> 在预注册 overtake 门内降低 288 场景 any-agent collision？

### 4.2 主假设

- H1：iteration-0 deterministic actor 与 canonical BC checkpoint、action 和同设备
  trajectory 完全一致。
- H2：训练期 full-action distribution、old/new log-prob 和执行命令是一条相同、
  可重放的概率链。
- H3：head-only PPO 至少能改变 collision/overtake frontier；它不预设一定通过门。
- H4：若某 candidate 降低 collision，其改善不应全部来自 interaction suppression；
  paired transitions 必须显示 fixed/new、collision->pass/follow、gained/lost overtake。
- H5：最终 actor-only state_dict 与 BC keys/shapes 完全相同，critic/distribution state
  不进入 deployment checkpoint。

### 4.3 本轮不能回答的问题

- 解冻 GRU 是否更好；
- sidecar 是否有增益；
- residual 的 function class 是否优于 direct head；
- 5% 门是否是正确产品偏好（它必须在运行前由 D1 决定）；
- 288 opened-development 之外的 generalization；
- B3 unified policy 是否会成功。

---

## 5. 冻结模型合同

### 5.1 deployment actor

最终部署模型只能是现有：

```python
actor = End2Race(mask_prob=0.0, hidden_scale=4)
actor.load_state_dict(candidate_state, strict=True)
```

actor key set 必须与 canonical BC 完全相同。禁止进入 actor-only checkpoint：

```text
critic.*
action_std / log_std
res_head.*
residual_budgets
sidecar.*
action_core.*
intervention_gate.*
brake_gate.*
optimizer / scheduler / RNG state
```

训练恢复所需的 full checkpoint 可独立保存，但永远不是 deployment candidate。

### 5.2 trainable/frozen 参数

冻结并逐字节审计：

```text
k
speed_mlp.*
dummy_embedding
gru.*
```

唯一 trainable actor 参数：

```text
output_layer.*  # 706,862 parameters
```

训练期另有独立 critic；critic gradient 不得流入 actor，actor gradient 不得流入
privileged feature provider。每次保存 candidate 前，必须将所有 frozen tensor 与
iteration-0 snapshot 做 exact equality 检查。

### 5.3 critic

默认使用一个独立 12D privileged critic：

```text
12 -> 128 -> SiLU -> 128 -> SiLU -> 1
```

12D 字段沿用 B2 已审计定义：relative progress、lateral gaps、ego/opp longitudinal
speed、ego/opp lateral offset、safe-pass hold、overtake-started、safe-pass-held、
opponent speedscale、sin/cos track phase。字段只能来自当前/历史 simulator state；
不得把 future candidate outcome 输入 critic。

本轮只有一个 scalar return，因此不再使用 collision/performance 两个 critic 或
independently normalized advantages。

### 5.4 frozen feature replay

rollout 时在 `torch.no_grad()` 下保存每个 100 Hz step 的 1680D GRU feature。
PPO update 对保存 feature 调用当前 `output_layer`，不重跑冻结 GRU，不对 800-step
序列做反向传播。

这在 head-only 条件下是合法的，因为 PPO batch 的 observation/feature 来自旧策略
已采集状态；更新后的 head 只需要在同一保存 feature 上计算 new log-prob。必须有
pre-update replay test 证明：未更新参数时 every-step ratio 为 1。

---

## 6. 动作分布：对外部建议的必要修订

### 6.1 为什么无界 Normal 不能直接接受

原建议：

```text
steer ~ Normal(mu_steer, 0.03)
speed ~ Normal(mu_speed, 0.20)
```

即使 mean 在合法范围内，Normal 仍有非零越界概率。当前 legacy env 会对 steering
和 speed clipping（`train_ppo.py:306-351,446-449`），而原始 evaluator 至少会对
steering clipping（`eval_multiagent.py:217-230`）。mean bound loss
（`train_ppo.py:826-832`）不能阻止 sampled action 越界。

B1 已经把 hidden external clipping 认定为执行动作/log-prob 风险；B2/B3 合同要求
composition 自身有界且 external clipping 为 0。B4 不应降低这一标准。

### 6.2 B4 默认分布

为保持 fixed physical-unit exploration，同时不增加 deployment state，默认使用
**仅训练期存在的 factorized truncated Gaussian**：

```text
steer ~ TruncatedNormal(mu_steer, 0.03, low=-0.52, high=+0.52)
speed ~ TruncatedNormal(mu_speed, 0.20, low=0.0,  high=20.0)
```

要求：

- sampling 使用规范化 truncated CDF/inverse-CDF；
- old/new log-prob 包含 state-dependent truncation normalization term；
- deterministic training mode 是 `clamp(mu, low, high)`；
- iteration-0 deterministic executed action 与原始 evaluator 的 BC action 相同；
- candidate 在 288 deterministic evaluation 上 speed projection count 必须为 0，
  因原始 evaluator 当前不会替 speed 做同样的 `[0,20]` clamp；
- evaluator 后处理不得再改变训练期 sampled command；external delta 必须精确为 0；
- fixed std 是训练配置，不是 actor parameter，不写入 actor state_dict；
- entropy coefficient 为 0，因此不需要用 entropy 推动 fixed std。

若外部审计不接受 truncated Gaussian，可批准另一个**单一**有界分布，但必须在
RunPlan 前固定，并同时满足 deterministic BC identity、exact log-prob、ratio-one
replay、无外部 clipping 和无 deployment wrapper。不能等结果后在 Normal、tanh
Normal、clipped Normal 之间切换。

### 6.3 必做动作合同测试

1. analytic 1D log-prob 与数值积分归一化一致；
2. boundary means、极小 tail mass 和 float32 下无 NaN/Inf；
3. keyed sampling 可 checkpoint/resume；
4. unchanged replay 的 max `|ratio-1|` 在前瞻性 tolerance 内；
5. sampled command、stored command、simulator command 完全相同；
6. 每个 rollout 与 evaluation 的 external postprocess delta 为 0；
7. deterministic action 与原始 BC evaluator action 在 iteration 0 一致；
8. actor-only checkpoint 用一个不含 residual 特判的 strict loader fixture 加载。

任何一项失败都属于完整性失败，不得用其 KPI 结果做科学裁定。

---

## 7. 训练 population 与 curriculum

继续使用已打开、与 288 development L2-disjoint 的 Task-8 training manifest。
当前可用训练池按 archived BC outcome 为：

```text
BC collision =   81
BC overtake  = 1001
BC follow    =  558
total        = 1640
train/dev L2 overlap = 0
```

每 iteration 固定 16 个完整 episode：

```text
6 BC-collision
6 BC-overtake
4 BC-follow
```

规则：

- 每 seed 在运行前生成完整 30-iteration order；
- 同一 seed 的所有 snapshot 当然共享同一 rollout history；不同 seed 使用各自 keyed
  order；
- 每组内部无放回；用尽后以新 keyed repeat order 进入下一轮；
- 30 iterations 中 collision 需要 180 个 slots，因此 81 个 collision pool 会重复；
  overtake 180/1001、follow 120/558 在首轮不需重复；
- manifest order、group、repeat index、L2/L4/map 全部写入 rollout ledger；
- 不加入 hard-start、lateral offset、sampler sweep 或结果驱动的 curriculum 变化。

该 curriculum 是机制训练分布，不是 288 开发 population 的无偏抽样。最终裁定
只能来自完整 paired development evaluation，不能从训练组平均 reward 推断 KPI。

---

## 8. episode、reward 与 credit assignment

### 8.1 termination

每个 episode：

```text
第一次 any-agent collision -> true terminal, bootstrap = 0
8-second product horizon    -> true terminal, bootstrap = 0
```

基础设施错误、缺帧、simulator exception、进程中断是 invalid rollout；不得编码为
truncation，也不得用于 update。B4 不使用 legacy “opponent-only collision truncation”。

### 8.2 outcome

episode 结束后调用与 paired product evaluation 相同的 corrected classifier，至少写出：

```text
collision_any
corrected_outcome3 = collision / follow / overtake
confirmed_safe_pass
interaction_attempt
ego_collision / opp_collision
```

### 8.3 唯一 actor reward

默认：

```text
R_train = -2 * collision_any + 1 * corrected_terminal_overtake
```

只在 episode 最后 transition 写入；其余 step reward 为 0。paired baseline delta：

```text
DeltaR_pair = 2 * (C_BC - C_pi) + (O_pi - O_BC)
```

只用于解释和复核。TTC、dense progress、clearance、warm-start loss、gate prior、
BC action anchor 均不进入 actor reward。

### 8.4 GAE

固定：

```text
gamma      = 0.999
gae_lambda = 0.995
```

在每个 episode 内独立反向计算；terminal zero bootstrap；绝不能跨 episode 传播。
优势只生成一个 scalar channel。

### 8.5 episode-equivalent actor weighting

外部建议指出 collision episode 通常更短，按 transition 平均会让长 episode 占更大
权重。该问题成立。B4 对每个 transition 定义：

```text
T_i = episode i 的 transition 数
N   = rollout 总 transition 数
E   = rollout episode 数（固定 16）
w_it = (N / E) / T_i
```

于是全 batch 的普通加权 mean 满足：

```text
(1/N) * sum_i sum_t w_it * L_it
  = (1/E) * sum_i (1/T_i) * sum_t L_it
```

要求：

- advantage 使用同一 `w_it` 做 weighted mean/variance normalization；
- actor surrogate、approx KL、clip fraction 都报告 weighted 与 unweighted 两版，
  actor early stop 使用 weighted KL；
- uniform transition minibatch 加 `w_it`，保证其梯度估计对应冻结的全 batch 目标；
- critic 默认也使用相同 episode weights，另报 unweighted value loss 诊断；
- 不允许在每个 minibatch 内重新把 weights 归一到不同总量而悄悄改变短 episode 权重。

这是 B4 新机制，必须用两个不同长度的 synthetic episodes 做精确 loss regression。

---

## 9. PPO 与 optimizer 合同

### 9.1 冻结配置

| 参数 | B4 默认值 |
|---|---:|
| actor architecture | 原始 `End2Race` |
| initialization | canonical BC strict load |
| trainable actor | `output_layer.*` only |
| frozen actor | `k`、`speed_mlp`、`dummy_embedding`、GRU |
| critic | 独立 12D MLP，单 scalar return |
| steer distribution | truncated Gaussian，std 0.03 rad fixed |
| speed distribution | truncated Gaussian，std 0.20 m/s fixed |
| actor LR | `3e-5` |
| critic LR | `3e-4` |
| PPO clip epsilon | `0.10` |
| target KL | `0.015`，weighted rollout KL |
| actor epochs | 最多 3 |
| critic epochs | 固定 3，不受 actor KL stop 影响 |
| minibatch | 1024 transitions，最后 short batch 保留 |
| entropy coefficient | 0 |
| max grad norm | 0.5，actor/critic 分别裁剪 |
| mean bound coefficient | 0.01 |
| BC anchor | 0 |
| dual | 无 |
| warm-start | 无 |
| iterations | 30 |
| actor snapshots | 0、10、20、30 |
| training seeds | 0、1 |
| architecture arms | 1 |

这些值是一个可证伪 pilot 配置，不是由现有结果证明的最优值。尤其 `BC anchor=0`
和无 dual 意味着 overtake guardrail 只由 reward、curriculum、局部 PPO 限制和最终
selection 检验；若失败，本轮应如实停止。

### 9.2 actor 与 critic 必须分开更新

当前 legacy `train_ppo.py:774-868` 在 KL 超限时整轮 break；当前 B2/B3
`bplus_v22/ppo_runner.py:604-668` 也在 actor/两 critic step 之前检查 KL。B4 修正为：

```text
actor loop:
    最多 3 epochs
    每个 epoch 后重算 weighted rollout KL
    KL > 0.015 时只停止后续 actor epochs

critic loop:
    始终完成 3 epochs
    不消费 actor early-stop flag
```

actor optimizer只能持有 `output_layer.*`；critic optimizer只能持有 critic 参数。
禁止一个 combined optimizer 或对 `ac.parameters()` 统一裁剪。

### 9.3 snapshot 与恢复

- iteration 0 保存一个 byte-identical BC actor state_dict 证明，不作为新模型宣传；
- actor candidate 固定在 iterations 10/20/30；
- full training checkpoint 每个完整 iteration 原子保存，以便 interruption resume；
- resume 必须恢复 optimizer、critic、iteration、curriculum cursor、RNG 和 keyed
  sampler state；
- 不从半个 episode 或半个 PPO epoch 恢复；
- 不因 seed0 或 iteration10 的 outcome 改变 seed1 或后续训练。

---

## 10. preflight 与阻断测试

### P0 — authority / population

- D1、D2 已由 owner 明确选择；
- canonical BC hash 和 12-key schema 匹配；
- training 1640、development 288、L2 overlap 0；
- training outcome pools 精确为 81/1001/558；
- fresh/final pool 未作为输入。

### P1 — iteration-0 identity

同一 host、同一 simulator/config、同一 scenario 上：

- actor state tensors 与 BC exact equal；
- raw model outputs exact equal；
- evaluator-postprocessed deterministic actions exact equal；
- hidden states、trajectory arrays、terminal outcomes exact equal；
- 两 host 不要求 trajectory bytes 一致；各自只与同 host BC 对照。

已有 B2 topology 证据显示一个近边界场景在 RTX 3080/4080 上可出现 138/139
overtake 差异，因此禁止把跨设备 bit identity 写成门。

### P2 — distribution / replay

- truncated distribution normalization、boundary、finite-gradient tests；
- sampled/stored/executed action identity；
- old-log-prob exact replay；
- unchanged ratio-one；
- fixed std 不出现在 actor state；
- external postprocess delta 0。

### P3 — buffer / GAE / weighting

- any-agent collision 与 horizon 均 terminal zero bootstrap；
- invalid infrastructure episode 不进入 buffer；
- terminal reward 只写最后一步；
- GAE 不跨 episode；
- 1/2/3/5 秒传播值 regression；
- 不同 episode length 的 actor/critic weighted loss 与手算一致；
- single weighted advantage normalization，无 objective-by-objective normalization。

### P4 — optimizer isolation

- 只有 `output_layer.*` 得 actor gradient 和 optimizer state；
- frozen actor tensor update 前后 exact equal；
- critic feature/parameters对 actor断梯度；
- actor KL stop 后 critic 仍完成所有 epochs；
- actor/critic grad norm 分开记录。

### P5 — deployment checkpoint

```python
bc = End2Race(mask_prob=0.0, hidden_scale=4)
candidate = End2Race(mask_prob=0.0, hidden_scale=4)

bc_state = torch.load("pretrained/end2race.pth")
cand_state = torch.load("candidate.pth")

assert tuple(bc_state.keys()) == tuple(cand_state.keys())
for key in bc_state:
    assert bc_state[key].shape == cand_state[key].shape
candidate.load_state_dict(cand_state, strict=True)
```

并用只 import `End2Race`、没有 residual autodetection 的 fixture 验证。失败即不得进入
paired evaluation。

### P6 — production-shaped smoke

四图各至少一个完整 episode，包含 collect -> replay -> actor update -> critic update ->
actor-only save -> strict reload。只检验 mechanics，不按 smoke outcome 调配置或选模型。

---

## 11. 托管运行与评估计划

### 11.1 control plane

若实现获批，应在 `Experiments/runner.py` 新增 versioned B4 plan/job，而不是手写 SSH
命令。沿用现有不可覆盖 RunPlan、两端 isolated staging、baseline/preflight、GPU lock、
atomic COMPLETE、collect/recovery 机制；不建立新的 ad-hoc 哈希证据体系。

推荐训练拓扑：

```text
remote RTX 4080 SUPER: seed0，30 iterations
local  RTX 3080:       seed1，30 iterations
```

每张 GPU 同时只跑一个 learner。两 seed 都完成后才创建唯一 EvalPlan。

### 11.2 candidate 集合

预注册评估七个 variants：

```text
BC
seed0_iter10
seed0_iter20
seed0_iter30
seed1_iter10
seed1_iter20
seed1_iter30
```

在同一个 288-scenario opened-development manifest 上共 2,016 paired rows。所有
训练和 checkpoint 收集完成前，不得根据训练 outcome 删除 snapshot。

selection 只能比较**同一 iteration 的两个 seeds pooled pair**：iter10、iter20、
iter30。不得为 seed0 选 iter10、seed1 选 iter30 后拼成一个 candidate。

### 11.3 baseline

沿用 topology-matched baseline：

```text
BC any-agent collision       = 24 / 288
BC corrected terminal overtake = 138 / 288
```

per-shard 期望保持 `[12/32, 2/37, 5/33, 5/36]`（collision/overtake）。baseline
漂移时 fail closed；不能把 138 改成 139 来通过本地单机重放。

---

## 12. 完整性报告与 paired KPI

每个 variant、每 seed、每 snapshot 必须报告：

- complete episode rows、missing/duplicate Cartesian keys；
- checkpoint key/shape strict compatibility；
- frozen-tensor equality；
- sampled/executed/postprocessed action delta；
- replay max log-prob delta、ratio-one delta；
- collision、RR、fixed/new；
- ego/opp collision；
- corrected overtake、gained/lost；
- collision->confirmed pass、collision->safe follow；
- overtake->follow、safe->new collision；
- confirmed safe pass、interaction attempt；
- weighted/unweighted advantage、KL、clip fraction、policy/value loss；
- actor/critic updates、early-stop epoch、grad norms；
- per map/skill/raceline/speed/L4 slices；
- L4-clustered paired bootstrap interval，仅标注 opened-development。

训练 scalar reward、critic loss 或 KL 不参与 candidate substantive ranking；它们只用于
解释失败机制。

---

## 13. substantive gate 与选择顺序

### 13.1 共同 integrity gate

任何 candidate 必须先满足：

1. 288/288 complete；
2. plain `End2Race` strict load；
3. frozen tensors exact unchanged；
4. 无 action/log-prob/replay failure；
5. external action delta 0；
6. 无非法 speed projection；
7. baseline 与 classifier 合同匹配。

### 13.2 overtake feasibility

按 D1 只使用一套：

```text
D1-A strict:
  each seed >= 138
  pooled    >= 276

D1-B 5%:
  each seed >= 132
  pooled    >= 264  # 与 per-seed 门一致；263 只作为原始算术备注
```

不通过者立即判 `FAILED_OVERTAKE_GUARDRAIL`，不得因 collision 很低而选中。

### 13.3 collision 与机制门

在 overtake-feasible candidates 中：

1. 每 seed collision 不高于 BC 24；
2. pooled `fixed_collision > new_collision`；
3. 两 seed 的 net collision improvement 同方向；
4. 完整报告 interaction-attempt 变化；
5. `collision -> confirmed pass` 为 0 不自动构成 integrity failure，但必须阻止将
   结果描述成“学习了安全超车恢复”；若只得到 safe follow，应按实际机制表述。

若产品目标继续要求 `RR <= .70`：

```text
each seed collision <= 16
pooled collision    <= 33
```

### 13.4 snapshot 选择

1. 丢弃 integrity failure；
2. 丢弃 overtake-infeasible iteration pair；
3. 丢弃任一 seed collision >24 或两 seed 方向不一致者；
4. 在剩余 iter10/20/30 pair 中选择 pooled collision 最低者；
5. collision tie 时选 pooled overtake 更高者；
6. 再 tie 时选更早 iteration。

若没有候选，B4 结束为 negative。不得继续评估同一 checkpoint、修改 reward 后 resume、
打开 fresh pool或只选“最安全 seed”。

---

## 14. 结果分支与停止规则

| 结果 | 裁定 |
|---|---|
| mechanics/integrity failure | 结果作废；保留 FAILED evidence；只修工程合同并创建新 RunPlan |
| 所有 snapshot 违反 overtake 门 | B4 substantive FAIL；说明 simple scalar/direct-head 未守住 guardrail |
| overtake 过门但 collision 未改善 | B4 negative；不得靠延长 iteration 自动续跑 |
| collision 改善但只转为 safe follow | 如实记录安全方向改善；不能声称恢复安全超车 |
| overtake 过门、两 seed collision 改善但 RR>.70 | development survivor；需要另行决定是否值得 fresh confirmation |
| 同 iteration 两 seed/pooled 过全部门 | 只允许提出独立 fresh/final confirmation plan；当前 fresh pool 仍封存 |

本轮禁止自动：

- 解冻 GRU；
- 增加 BC anchor 或 dual；
- 修改 reward 权重；
- 调 std、LR、clip、KL；
- 从 30 延长到 40/60；
- 加第二 architecture arm；
- 运行 B3 作为 outcome-driven fallback；
- 打开 fresh/final pool。

任何后续变化必须基于完整 B4 报告另写前瞻性计划。

---

## 15. 代码组织建议

### 15.1 不删除历史代码

外部建议中“从主路径移除 residual flags/classes”只能作为未来清理方向，不能在 B4
结果前执行。历史 `train_ppo.py`、`model.py` residual class、B2/B3 loaders 和测试仍
被已完成 evidence 引用。删除会破坏复核能力。

### 15.2 B4 最小新增面

建议：

- `model.py`：保留 `End2Race` 不变；训练 wrapper 可以是新 class，但不能改变
  `End2Race.state_dict()` schema；
- 新的 versioned B4 runner/buffer/distribution 可放在独立模块，避免把 B4 行为塞进
  legacy `train_ppo.py` 的大量旧 flags；
- 复用 B2 已审计的 scenario loader、12D critic feature、corrected classifier、paired
  evaluator 和 control plane，不复用 residual/gate policy；
- `Experiments/runner.py` 只在实现审阅 GO 后新增 B4 plan/job；
- B2/B3 兼容测试继续通过。

外部专家应重点审查“复用 scenario/evaluator 基础设施”是否会无意实例化 sidecar、
gate、residual composition 或 two-head objective。B4 actor path 中这些对象都不应存在。

---

## 16. 请求外部审计专家明确回答的问题

请按顺序回答，不要只给总体 GO/NO-GO：

1. 是否同意“Residual 非 PPO 必要组成，但该事实本身不否定未运行的 B3”？
2. strict plain-End2Race checkpoint compatibility 是否应成为硬产品条件？若是，是否
   同意选择 D2-B 暂停 B3、先 B4？
3. D1 应保持 strict 138/276，还是业务上确已改为每 seed 132 的 5% 门？理由是什么？
4. 是否同意 head-only 只是 frozen-feature control，不能宣称解决 representation？
5. 是否同意无界 Normal + bound loss 不满足本项目动作概率合同？
6. proposed truncated Gaussian 的 log-prob、deterministic identity 与 deployment
   compatibility 是否完整；是否存在更简单但同样严格的单一替代？
7. 是否同意训练用 `-2C+O`、paired delta 只做 diagnostic；若不同意，有限样本理由
   和 critic 输入应如何冻结？
8. scalar reward 无法保证 overtake 门的风险是否被 stop/selection 规则充分控制？
9. episode-equivalent weights 在 minibatch、advantage normalization、KL 和 critic
   loss 中的定义是否一致？
10. actor/critic 分离 update 是否修复了当前 KL early-stop coupling？
11. 七 variants、同 iteration 跨 seed selection 是否足以防止 checkpoint cherry-pick？
12. 在不引入第二 arm/sweep 的前提下，是否批准进入实现；若否，请指出会使 KPI
    结论无效的具体 blocker，而不是新增 TTC/warm-start 代理门。

---

## 17. 当前建议的批准语句模板

只有在外部审计与 owner decision 完成后，authority 才可追加类似以下内容：

```text
B4 direct-head PPO implementation is approved under D1-[A/B] and D2-[A/B].
The deployment actor must remain a strict plain End2Race state_dict.
The bounded training distribution, scalar terminal reward, 6/6/4 curriculum,
episode-equivalent weighting, separated actor/critic loops, 30 iterations,
two seeds, fixed snapshots, and paired gates in this document are frozen.
Approval covers implementation and preflight only; numerical GPU execution
requires an independently reviewed clean committed source and one immutable RunPlan.
```

在此语句出现前，本文仅是外部审计材料。

---

## 18. 主要仓库证据入口

- 当前总状态：`.agents/HANDOFF.md:2075-2259`
- PPO 全开发史：`.agents/PPO_DEVELOPMENT_REPORT.md:193-338`
- B2 目标/GAE/门：`.agents/B2_PPO_PLAN.md:385-470,718-790`
- B3 未运行计划：`.agents/B3_PPO_PLAN.md:1-178`
- 原始/residual actor：`model.py:9-176`
- legacy PPO distribution/update：`model.py:178-289`、`train_ppo.py:717-889`
- legacy termination/clipping：`train_ppo.py:306-355,446-449`
- actor-only checkpoint helper：`ppo_utils.py:511-570`
- evaluator residual 特判与 steering postprocess：`eval_multiagent.py:14-30,217-230`
- B2 scalar/objective implementation：`bplus_v22/ppo.py:169-198,263-390`
- B2/B3 coupled update loop：`bplus_v22/ppo_runner.py:541-680`
- variable-discount GAE：`bplus_v22/buffer.py:66-117`
- complete-episode any-agent terminal：`bplus_v22/ppo_env.py:680-771`
- canonical B2 results：
  `Experiments/B2_ppo_pilot/evaluations/b2_eval_20260713_165800/merged/`
