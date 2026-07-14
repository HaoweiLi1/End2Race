# End2Race 实验记录

> 本文件是**全部实验轨道的历史记录**，含已废弃轨道。
> 当前工作状态与下一步请看 `.agents/HANDOFF.md`（唯一权威入口）。
> 仓库文件/代码结构说明请看 `.agents/REPO_GUIDE.md`。
> 生成时间：2026-07-12；B5-A 执行结果更新于 2026-07-14。

---

## 0. 项目目标（未变）

用户 2026-07-10 锁定的**词典序**目标：

1. **硬约束**：超车率不低于 BC（Austin 及跨图）；
2. 在此前提下**显著降低任意方碰撞率**（目标 `RR ≤ 0.70` vs BC）；
3. 可选：之后再缓慢提升超车率。

主 KPI 是 any-agent 碰撞率（保持历史可比性），ego-only 作为分解项报告。

**2026-07-13 已完成第一次 B+ v2.2 PPO 候选的直接词典序开发评估。**
六个候选均未同时满足超车非劣与 `RR <= 0.70`；因此没有 arm selection，
也没有打开 fresh/final pool。历史 PPO 跑曾同时记录碰撞与超车，但这次是
D2/B1 之后第一次用当前 B+ 目标直接裁决学习后的候选。

2026-07-14 owner 对 B4 作了前瞻性目标变更：安全优先，corrected overtake 最多
相对 BC 降低 5%。这不追溯改写 B2/B3。B4 已按该规则执行并关闭，结果见 §6。
B5-A 随后在保持 B4 科学设置不变时加入 canonical-BC safe-reference hard cap；
唯一 seed1 在 opened development 选出 iter10 survivor，但尚未 fresh/final 确认。

---

## 1. 轨道总览

| 轨道 | 时间 | 结论 | 数据是否还在用 |
|---|---|---|---|
| 初始基线 `end2race_*` | 07-03~04 | BC 基线建立 | 否（已被 P1 取代） |
| D1 / D1a / D1b | 07-04~05 | PPO 早期变体，失败 | 否 |
| anchor_baseline | 07-05 | 全参数 PPO + `beta_bc=5.0` 软锚定；提升不显著 | 否 |
| 旧 D2（PPO 变体 b/c/cpot） | 07-05~06 | residual PPO 调参，提升不显著 | 否 |
| D3 | 07-06 | 设计阶段，未产出模型 | 否 |
| D4a / D4b / D4c | 07-06~08 | 产出候选 cand040/cand120/cand160 | 否（模型已被 P1 评估） |
| **P1 验证** | 07-10 | cand160 与 BC **统计等价** | ✅ 是（活链源头） |
| **D0.1 正典审计** | 07-10 | 修正 estimand：N=3024 → **3036** | ✅ 是 |
| **D2 表示探针** | 07-11 | 4 个家族**全部未过** TTC 门 | ✅ 是（数据集仍在用） |
| **D2.5 反事实** | 07-11 | 67/91 可恢复，Route-R2 动作空间可行 | ✅ 是 |
| **D2R 几何探针** | 07-11 | **未过** TTC + 2s 误报门 | ✅ 是 |
| **B+ v2.2 / B2→B5-A PPO** | 07-11~14 | B2 六候选失败；B3 暂停；B4 negative；B5-A iter10 为 opened-development survivor | ✅ 是（当前证据） |

---

## 2. 已废弃轨道（结论已固化，原始数据可弃）

这些轨道的**结论已经写进 `logs/*.md` 报告**，原始 NPZ/checkpoint 不再被任何活证据引用。

### 2.1 PPO 调参期（D1 → D4c，07-04 至 07-08）

反复尝试用 PPO 改进 BC，手段包括：奖励塑形、credit assignment、anchor 强度、residual 预算、场景课程。

**共同结局：提升不显著。** 最好的候选 cand160 在 P1 验证中被判定为「在 Austin 训练图的未触碰 holdout 网格上与 BC **统计等价**」。

**回顾性发现（2026-07-12 审计）**：这批实验从未真正解锁过 PPO 的潜力，因为它们都被锁住了，只是锁法不同——

- **residual 系列（22 次跑）**：actor 的 11,301,482 个参数**冻结 98.13%**，仅 215,426 个残差头参数可训练；且全部带 `--residual_speed_up_budget 0.0`，**速度通道结构上只能刹车、无法加速**，因此超车率在物理上不可能超过 BC。
- **anchor 系列（全参数可训练）**：带 `--beta_bc 5.0` 强 BC 锚定损失，权重虽可动但被强力拽回 BC。

即：**从未跑过"解冻 + 动作空间自由"的 PPO。** 两种锁的结果都是"提升不显著"，高度一致。

（注意：`beta_bc=5.0` 的"锚定过强"只有 2 个 seed、无消融，**不足以作为因果结论**，仅作观察记录。）

### 2.2 相关产物

- `eval_results/` 中 73 个 `*_local_*` 目录（33 GB）
- `pretrained/` 中 185 个 local-only checkpoint（约 9 GB）
- `logs/aborted_runs/`（134 MB）

现有审计记录称这些早期本地产物没有完整远端备份。这个事实必须在
任何大文件删除前重新只读核验；Tier 3 仅将相关日志归档，没有删除
`eval_results/` 或 checkpoint。处置边界见
`docs/ppo/ARTIFACT_RETENTION.md`。

---

## 3. 活证据链（当前）

链条：**P1 验证 → D0.1 → D2 数据集 → {D2.5, D2R} → B+ v2.2**

活证据链使用的是 P1/D0.1 冻结来源；本地33GB早期
`eval_results/` 不应被假定为这条链的替代副本。删除前仍需按 manifest
逐项核验本地/远端覆盖关系，不能只依据目录名下结论。

### 3.1 P1 验证（07-10）

- cand160 vs BC，Austin holdout + 跨图，配对检验。
- 结论：**Austin holdout 上统计等价**；跨图有名义显著的超车提升（p=0.043，未校正）和边缘碰撞改善（p=0.053）。
- 不得声称 holdout 优越性。
- 报告：`logs/p1_final_report_20260710.md`、`logs/final_model_report_20260710.md`

### 3.2 D0.1 正典审计（07-10）

- 修正 estimand 不一致：N=3024 → **3036**。
- 16,800 occurrence 完整性成立；旧的"双显著"和跨图显著性结论**已撤回**。

### 3.3 D2 表示探针（07-11）

**问题**：PPO 撞车，是因为"看不见危险"，还是"看见了但不会躲"？

**做法**：从**部署时真实可得的输入**（LiDAR、速度、指令历史）监督预测：1 秒/2 秒后是否碰撞（分类），以及 TTC/几何量（回归）。

**数据**：BC 基线 1,928 个非测试 episode，1,505,848 帧。grouped test（1,108 episodes）**封存未开**。

**预注册门**（源自已批准的 B+ 设计文档 §"Pre-registered gate"）：
1s 召回 ≥0.60 且安全误报 ≤0.10；2s 召回 ≥0.40；Brier skill ≥0.10；**TTC<2 区间 MAE ≤0.30 秒**；≥30 个独立 held-out 碰撞 episode。

**结果**：

| 家族 | 输入 | 1s召回 | 1s误报 | 1s BSS | TTC MAE | 判定 |
|---|---|---:|---:|---:|---:|---|
| linear | 冻结 BC 特征 | 0.769 | 0.106 | 0.155 | 0.976s | FAIL |
| static MLP | 冻结 BC 特征 | 0.703 | 0.083 | 0.131 | 1.099s | FAIL |
| temporal T1 | **仅**冻结 BC 特征（多帧 taps） | 0.648 | 0.075 | 0.078 | 1.073s | FAIL |
| temporal T2 | 冻结特征 + 原始 LiDAR/速度历史 | 0.725 | 0.095 | 0.105 | 1.068s | FAIL |

决策：`STOP_D3_TEST_UNOPENED_CONTINUE_D2P5_DIAGNOSTIC`

### 3.4 D2.5 反事实可恢复性（07-11）

**问题**：BC 撞车的那些案例，用有界的宏残差动作**能不能救回来**？

**做法**：对 91 个 ego 碰撞 episode，用固定分支库穷举搜索（3/2/1 秒提前量 × 0.5/0.3/0.1 秒时长 × 10 个刹车/转向残差），找"确认安全通过"的见证动作。**无学习**，纯搜索。

**结果**：执行 3,340 个分支。

- **67/91 找到确认安全通过的 witness**；24/91 搜遍分支库仍无解。
- 全部 91 个 no-op 基线**逐位复现**归档评估器输出。
- witness 覆盖四张图、54 个 L4 块；skill_F 31/37、skill_S 29/37。
- **无 witness 使用 clipping 或正向速度残差**。

决策：`D25_COMPLETE_ROUTE_R2_FEASIBLE_D2_REDESIGN_REQUIRED`

**⚠️ 措辞纪律**：67/91 是**固定分支库**在**非测试 ego-collision 子集**上的**已证实可恢复集**。它**不是**全动作空间的理论天花板，也**不能**直接换算成全分布 any-agent RR。（早期一份分析曾错误地把它称为"74% 天花板"，已撤回。）

### 3.5 D2R 时空几何探针（07-11）

**问题**：换一个自训的 beam-local 卷积编码器（不再只读冻结 BC 特征），能不能过 TTC 门？

**结果**：

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| 1s ego 召回 | 0.868 | ≥0.600 | pass |
| 1s 安全误报 | 0.099 | ≤0.100 | pass |
| 2s ego 召回 | 0.835 | ≥0.400 | pass |
| **2s 安全误报** | **0.103** | ≤0.100 | **fail** |
| 1s Brier skill | 0.130 | ≥0.100 | pass |
| **TTC<2 MAE** | **0.800s** | ≤0.300s | **fail** |

决策：`STOP_D3_TEST_UNOPENED_D2R_G_FAILED_TTC_AND_2S_FA`

**TTC 失败的机制**（有硬数据）：越危险，高估越离谱——

| 真实 TTC 区间 | 帧数 | 预测均值 |
|---|---:|---:|
| 0–0.25s | 4,795 | 0.913s |
| 0.25–0.5s | 2,706 | 1.471s |
| 1.5–2.0s | 23,759 | 2.338s |

原因是数据稀缺，不是架构问题：最危险的 0–0.5 秒帧只有 7,501 个，碰撞前 1 秒的正样本仅占全部帧的 **0.69%**。

### 3.6 关键横向发现：冻结的 BC 特征是贫瘠的

把 D2/D2R 的三个家族按「多大程度绕开冻结的 BC 表示」排序，是一条单调曲线：

| 家族 | 输入 | 1s 召回 | Brier skill |
|---|---|---:|---:|
| T1 | **只读**冻结 BC 特征 | 0.648 | 0.078 |
| T2 | 冻结特征 + 原始 LiDAR/速度历史 | 0.725 | 0.105 |
| D2R-G | 冻结特征 + **自训 beam-local 编码器** | **0.868** | **0.130** |

同一批数据、同一个标签、同一个 split。**绕得越开，越能看见危险**（召回相对提升 34%）。

**⚠️ 因果强度限定**：这三者同时改变了输入、架构、空间归纳偏置和辅助监督，**不是干净的消融**。严格能说的只有：*只读冻结特征，明显弱于允许重新编码原始 LiDAR*。方向性支持"解冻编码器"，但不构成因果证明。

---

## 4. 治理转折：TTC override（07-11）

### 事实澄清

`TTC MAE ≤ 0.30s` **是预注册且已批准的**——它写在 `docs/superpowers/specs/2026-07-10-ppo-safety-first-bplus-design.md` 的 `#### Pre-registered gate` 一节，文档状态为 `Approved design`。

（一份早期分析曾断言它是"后来随意添加、从未批准"，**该断言是错的，已撤回**。Codex 是在严格执行一份已批准的规范。）

### 项目所有者的前瞻性 override

**理由**（部署需求，非"没考过所以改考卷"）：

1. **PPO 不需要 TTC 回归**。策略要学的是"该刹车了"，不是输出一个标定准确的秒数。
2. **D2.5 的 67 个 witness 全部使用 1–3 秒提前量的宏动作**——策略实际消费的是二值报警 + 几何量。
3. **碰撞率与超车率才是用户预注册的真实目标**（07-10 锁定）。TTC 是后加的**中间代理**。放弃代理、回到真实目标是**收紧**而非放松。

**决定**：TTC 从 PPO 准入门**降级为只读诊断**；D2/D2R 的历史判定**保持 FAIL，不得改写为 pass**；sealed D2 test 直接退休，不再打开。

已批准文档自身的失败分支写的是 `stop D3 and revisit observation/action representation` ——即重新审视**观测与动作表示**，这正是 v2.2 Route-R2 的方向，而非"再跑第六个监督探针"。

---

## 5. B+ v2.2 Route-R2（当前轨道，07-11 至今）

### 5.1 设计

- **三臂消融**（回答"可训练的风险表示有没有用"）：
  - **A `BC_FROZEN`**：仅用冻结 BC 特征的宏残差
  - **B `SIDECAR_FROZEN`**：加入预训练 D2R sidecar，但**冻结**
  - **C `SIDECAR_FINETUNE`**：同 B，但**解冻** sidecar（小学习率）
- **动作**：10 Hz 宏残差，层级结构（介入门 → 刹车/纯转向 → 幅度），有界合成。
- **约束 PPO**：碰撞与超车分离的 critic/advantage；非负 dual 只在超车跌破 floor 时激活；checkpoint 按词典序选择。
- **唯一晋级门是真实闭环 KPI**：开发门按每 seed 及 pooled 的完整
  manifest 使用 `fixed_collision > new_collision`，并允许预注册的1个百分点
  overtake 容差；最终产品门仍要求 corrected overtake 不低于 BC。
  TTC/Brier/报警不得参与排名。

### 5.2 任务阶梯与结果

| Task | 内容 | 结果 |
|---|---|---|
| 1–5 | 结构、spec、preflight、registry | PASS |
| 6（第一次） | witness warm-start | **FAILED**（见 5.3） |
| 6（补救） | 修正 gate bias 后重跑 | PASS（recall 0.735–0.995） |
| 7 | 独立 outcome 信号、断梯度、dual 状态恢复 | PASS |
| 8 | 冻结 288 开发场景 / 67 witness / 18 评估任务 | PASS |
| 9 | 64 次零残差重放 identity | PASS（16/16 bit-identical） |
| **10** | **warm-start checkpoint 闭环评估** | **FAILED**（见 5.4） |
| 6（层级重写后） | 新自然采样计划 | **FAILED**（见 5.5） |
| B2 | BC-direct PPO pilot，三臂×双 seed×20 iterations | **训练与冻结开发评估完成；六候选方向门全部 FAILED** |
| B3 | unified train/deploy residual policy | **IMPLEMENTED, REVIEWED GO, PAUSED UNRUN** |
| B4 | plain End2Race output-head-only PPO，seed1×30 | **训练与 3×4×50 评估完成；SUBSTANTIVE NEGATIVE** |
| B5-A | B4 严格单变量 + canonical-BC safe-reference hard cap，seed1×30 | **训练与 3×4×50 opened 评估完成；iter10 保留历史 SURVIVOR 标签，但 safety effect 统计不确定且未达 collision target** |

### 5.3 Task 6 第一次失败：gate bias 初始化错配（已修复）

**根因**：`INITIAL_BRAKE_LOGIT = -6.0`（隐含刹车先验 **0.2473%**），但 warm-start 训练数据的刹车正样本率是 **22.91%**——**差 93 倍**。gate 权重零初始化 ⇒ 初始 logits 恒为 −6 ⇒ 判据 `logit > 0` 恒为 NO_OP ⇒ `gate_recall` **必然为 0**。

**数值确认**：常数 logit −6.0 在 p=0.2291 下的理论 BCE = **1.3770**，与三个臂实测的 `diagnostic_before.gate_loss` **1.3770** 六位小数吻合。

训练后 gate_loss 0.83–0.86，仍**差于**恒定边缘预测器的 0.5382——模型连边缘分布都没学到。「loss 下降」与「gate 全选 NO_OP」不是矛盾，是同一个缺陷的两面。

**设计冲突**：`-6.0` 是**刻意**为了让初始策略近似 NO_OP、从而保证零残差 identity 成立。一个常数同时服务 identity 门和 warm-start 目标，两者直接冲突。

**修复**：显式分离两个初始化阶段——fresh/identity 阶段保留 `-6.0` 不变（现有 16/16 identity 证据因此**无需重跑**）；warm-start 阶段在第一次优化前用**拟合调度的经验先验**重置 gate bias。失败的 release 原样保留并标记 FAILED。

### 5.4 Task 10 失败：warm-start checkpoint 闭环上有害

288 个开发场景，逐臂闭环（无学习）：

| 臂 | 碰撞 | 修复/新增 | 超车 得/失 | 刹车 episode | 刹车宏决策占比 |
|---|---:|---:|---:|---:|---:|
| BC（基线） | 24 | — | 138 次超车 | — | — |
| A BC_FROZEN | 91 | 11 / **78** | 26 / 31 | 206/288 | 6.77% |
| B SIDECAR_FROZEN | 54 | 14 / **44** | 15 / **71** | 287/288 | **35.55%** |
| C SIDECAR_FINETUNE | 67 | 13 / **56** | 28 / 54 | 287/288 | 23.40% |

**三臂全部：新增碰撞远多于修复，且净损失超车。**

**机制（独立审计发现，比"过度触发"更精确）**：

1. **刹车门明显保留了训练配比。** B 臂全场刹车宏决策 **35.55%**，而 warm-start 训练调度的刹车先验是 **34.37%**——几乎相等。96 个 `representative_preservation` 并非全是安全 episode；其中 91 个 BC-safe episode 上，B 仍为 **91/91** 至少刹一次，宏决策刹车率 **30.60%**。这支持部署校准失败，但不能单独证明模型完全没有输入依赖。
2. **转向残差无门限、常开。** A 臂有 **20 个新碰撞发生在整个 episode 从未刹车的情况下**——唯一活跃通道是转向。三臂平均 |steer delta| ≈ 0.03 rad，施加在**每一个**宏决策上。刹车有 gate 把关，**转向没有**。
3. **clipping 全部来自转向越界**：39 个被裁 episode 中残差最大仅 0.1391（预算 ±0.2 内），被裁的是「BC 转向 + 残差」合成动作越过物理边界。副作用：执行动作 ≠ 策略声明的动作，会污染 PPO 的 log-prob 记账。

**误报风险的定量兑现**：既有 D2R OOF 账本在这 288 个 opened-development 场景子集上重算得到 **9.85%**（26/264）BC-safe episode 假警，与此前约 **9.9%** 的担忧一致。它不是独立模型或 fresh-population 复现，但准确预示了这个开发子集上的闭环风险。

**Task 6 指标预测不了闭环排序**：B 的 Task-6 recall 最低，却在闭环中刹得最凶。⇒ Task 6 只能是 mechanics gate，不能排名。

### 5.5 Task 6 第二次失败：正类饥饿（当前状态）

层级动作重写后，改用**自然采样**（避免训练配比决定部署频率），先验降到 **0.574%**（252/43,902）。

| 臂 | 假介入 | 正例 episode | 正例 macro | 条件 specificity |
|---|---|---|---|---|
| A / B / C | 7/75 | 0/9 | 0/39 | 0/14 |

**为什么三臂四项指标完全相同**：阈值取第8大负例 episode max并使用
严格越线规则，因此构造性保证假介入 `<=7/75`；存在 ties 时不保证恰好7。
本次三臂实测均为7/75，所以这只能作为校准一致性断言，不能当作臂成绩。
三臂全部正例分数都落在各自阈值之下，recall 为0。

**证据指向正类暴露稀疏、批间方差和层级优化耦合；不是有效的
sidecar 排名**：

- 新自然采样计划只有 **1,502** 次 intervention 暴露，旧计划有 131,072 次——**差 87.3 倍**；
- **231/1,024** 个 batch 完全没有正例；
- 即使在 fit 数据上，A/B/C 的 intervention recall 也只有 **0.151 / 0.087 / 0.067**；
- fold-4 的39个正类 macro 上，raw-logit range 为 A:0.027、B:0.636、
  C:1.162。这只说明 B/C 在正窗口内分数变化更大、A 在该切片近似常数；
  range 不是判别指标，不能单独证明学到了正确方向的输入依赖。当前
  calibration episode-max AUC 仍远低于0.5。

**钟摆两次都荡到极端**：第一次训练配比 34% → 到处乱刹；第二次自然配比 0.57% → 永不刹车。中间没有落点。

---

## 6. 当前诊断（截至 2026-07-13）

### 已知为真

1. **危险是可从部署观测中解码的**（D2R：1s 召回 0.868 / 误报 0.099）。但**可解码 ≠ PPO 能从稀疏奖励中学到**。
2. **有界宏残差动作空间足以恢复大量碰撞**（D2.5：67/91，在固定分支库内）。
3. **冻结的 BC 特征在危险信息上明显贫瘠**（方向性，非干净因果）。
4. **误报会兑现为超车损失**：9.85% 的安全 episode 触发报警，而总共只有 24 个碰撞。报警→刹车的直接耦合是不安全的。

### B2 首次直接 PPO 产品 KPI 测量

专用 B+ runner 已实现，训练 RunPlan `b2_direct_20260713_081422` 完成
三臂×双 seed×20 iterations。冻结评估 EvalPlan
`b2_eval_20260713_165800` 在本地 shard 0 与远端 shards 1–3 上完成
288 场景×7 variants，共 2,016 行；完整性通过、外部 clipping 为 0。

BC 为 24 次碰撞、138 次超车。逐 seed 结果：

| 候选 | 碰撞 / RR | 超车 | fixed/new collision | gained/lost overtake | 判定 |
|---|---:|---:|---:|---:|---|
| A seed0 | 26 / 1.083 | 124 | 11/13 | 7/21 | FAIL |
| A seed1 | 11 / 0.458 | 113 | 20/7 | 7/32 | FAIL |
| B seed0 | 17 / 0.708 | 118 | 14/7 | 6/26 | FAIL |
| B seed1 | 17 / 0.708 | 97 | 19/12 | 7/48 | FAIL |
| C seed0 | 8 / 0.333 | 95 | 21/5 | 1/44 | FAIL |
| C seed1 | 9 / 0.375 | 88 | 20/5 | 1/51 | FAIL |

这证明 PPO 并非没有学到安全行为：C 臂显著减少碰撞。但它主要通过压低
交互和超车来取得安全改善，两个 seed 的 `collision->confirmed_pass` 都为 0。
六个候选全部净损失超车，连开发期 1pp 容差都未通过，更不满足产品硬约束。

`any_opened_dev_point_target_hit=false`、`arm_selection_performed=false`、
`fresh_pool_opened=false`。因此不需要对这些候选继续 medium/final 实验。

### B3 当前修订与未决问题

B2 的 read-only 审计发现，PPO rollout/log-prob 使用 raw-logit Bernoulli，
而产品评估使用独立 centered threshold。三臂 pooled 的 standard deterministic
intervention 全为 0，centered primary rule 却分别产生 27,851 / 29,388 /
24,379 次介入。训练期 dual 看到的是近 BC 的 stochastic rollout，不是部署时
大量介入并损失超车的 centered policy。

B3 已前瞻性统一 sampling、stored/replayed log-prob 和 deterministic mode：
fresh `P(intervene)=0.10`、`P(brake|intervene)=0.50`，严格 mode 仍为 NO_OP；
centered mode 与外部 gate offsets 被拒绝。A/B/C×双 seed 固定为40 iterations，
不续训 B2 checkpoint、不改 reward/dual/product gate。实现提交 `19e83ae`，
边界回归提交 `21085bc`，CPU 合同测试通过并经独立审阅 GO；尚未创建 RunPlan
或使用 GPU。

B3 没有创建 RunPlan。owner 随后因 plain-checkpoint 产品要求前瞻性暂停 B3，转而
执行 B4；B4 negative 后也没有自动恢复 B3 的 authority。

### B4 plain-End2Race direct-head PPO

B4 严格加载 canonical BC，冻结 GRU 和输入编码，只更新已有 `output_layer.*`；
deployment snapshot 仍是 12-key plain `End2Race.state_dict()`。唯一 seed1 在远端
RTX 4080 SUPER 完成 30 iterations。最终评估严格使用原 BC Austin grid：3 条
opponent raceline × 4 个 speed scale × 50 个 startpoint，共 600 episodes/variant。

| variant | collision | overtake | follow | fixed/new collision | 判定 |
|---|---:|---:|---:|---:|---|
| BC | 24 | 342 | 234 | — | baseline |
| iter10 | 24 | 332 | 244 | 11/11 | 无净安全改善 |
| iter20 | 36 | 294 | 270 | 14/26 | 碰撞、超车均失败 |
| iter30 | 39 | 296 | 265 | 14/29 | 碰撞、超车均失败 |

5% overtake floor 为 325。iter10 通过 floor 但 collision 未下降且 `fixed=new`；
iter20/30 两门均失败。最终 `selected_candidate=null`，B4 为
`B4_SUBSTANTIVE_NEGATIVE`。这否定的是本次 frozen-feature、head-only PPO 配置，
不证明 residual、GRU unfreeze 或其他未运行设计会成功。fresh/final pool 未打开。
完整记录见 `.agents/B4_DIRECT_HEAD_PPO_RESULT.md`。

### B5-A canonical-BC safe-reference trust region

B5-A 完整保留 B4 的 actor、可训练参数、LR、clip、epochs、100 Hz iid
exploration、reward、精确 seed1 curriculum、30 iterations 和 snapshots；唯一新增
机制是在 64 个 training-only canonical-BC safe histories 上约束平均
`D_safe<=0.01`，并以 actor+Adam 完整回退和固定 LR ladder 执行 hard cap。

远端 seed1 完成 30/30；87 个 considered actor epochs 中接受 46、跳过 41，
critic 为 90/90 epochs，最大 iteration-level `D_safe=0.0099957`。Austin opened
600-case 结果为：

| variant | collision | overtake | follow | fixed/new collision | 判定 |
|---|---:|---:|---:|---:|---|
| BC | 24 | 342 | 234 | — | baseline |
| iter10 | 22 | 347 | 231 | 9/7 | **opened-development survivor** |
| iter20 | 25 | 349 | 226 | 5/6 | collision fail |
| iter30 | 27 | 343 | 230 | 5/8 | collision fail |

iter10 通过 overtake `>=325`、collision `<24`、`fixed>new` 和 speed projection 0，
但未达到 collision `<=16` 的 opened target。该结果支持 BC preservation 假设，
但后期 collision 在 cap 内仍回升，不能推断 closed-loop 安全已经解决，也不能自动
归因为 exploration/representation。fresh/final 仍封存；seed0 需要外部结果审阅后
另行批准。完整记录见 `.agents/B5_SAFE_TRUST_REGION_RESULT.md`。

后续只读审计将 600 rows 按 50 个 startpoint 聚类：iter10 collision 的
occurrence McNemar two-sided `p=0.803619`，startpoint sign-flip one-sided
`p=0.408356`，bootstrap 95% interval `[-6,10]`；三 snapshot joint max-effect
probability 为 `0.578968`。因此历史 feasibility verdict 保留，但不能写成已确认
安全提升，checkpoint 不晋升。

同一审计还在远端恢复 B5 iterations 1–10 的 actor+Adam 和原 minibatch order，
测试 opened-Austin outcome 权重。加权 base epoch 的 safe-cap 通过数为 `5/10`
（原 objective `7/10`），且没有系统降低 B4/global 或 cap direction alignment；
collision-gradient function norm 中位数仅为原来的 `0.19518`。所以 B5-B
objective-weighting learner 为 `NO-GO, UNRUN`。详见
`.agents/B5_POSTHOC_STATISTICS_AND_OBJECTIVE_AUDIT.md`。

---

## 7. 代理弯路与真实 KPI 失败（供未来参考）

两个代理轨道具有相似失败模式；B2 必须单独分类，因为它直接使用用户的
碰撞/超车目标，而不是又一道代理考试：

| 弯路 | 中间产物 | 自设及格线 | 代价 |
|---|---|---|---|
| 1 | TTC 回归探针 | TTC MAE ≤ 0.30s | 5 个家族全败 |
| 2 | warm-start 蒸馏 | gate recall / loss 门 | 2 次失败，约 4,600 行 |
| — | 首轮 B2 PPO | **用户真实词典序方向门** | 碰撞可下降，但六候选均以超车下降换安全，属于真实目标失败 |

前两项的共同特征是：中间产物带着不直接属于产品目标的及格线，失败后容易
继续换架构而忘记交付物。B2 不属于这种情况；它的超车非劣门正是产品硬约束，
不能因为碰撞降低就 override。

**教训**：中间门的存在意义是**防止把错误的东西推进生产**，不是**证明中间产物本身科学上完美**。当一道门连续挡住多个架构、而它测的东西又不是交付物需要的，应该质疑门，而不是继续换架构。质疑的合法方式是**项目所有者前瞻性 override**（说明部署为什么不需要它），而不是"没考过所以改考卷"。
