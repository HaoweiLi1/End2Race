# 后 U44 候选方法：预注册、执行与判决（GATES）

更新时间：2026-08-09（Asia/Singapore）

本文件由两份文档合并而成，**原文全部保留，未做删改或压缩**：

| 原文件 | 在本文件中的位置 |
|---|---|
| `SELECTIVE_BC_SAFE_BEHAVIOR_ANCHORING_PREREGISTRATION_FINAL_REVIEWED.md` | 第一部分，章节号 `§0`--`§35` 不变 |
| `COUNTERFACTUAL_ACTION_GATE_COMPLETE_REPORT.md` §1--§23 | 第二部分「附录 A」，章节号改为 `§A1`--`§A23` |
| `COUNTERFACTUAL_ACTION_GATE_COMPLETE_REPORT.md` §24--§29 | 第三部分「附录 B」，章节号改为 `§B24`--`§B29` |

第四部分「附录 C」是 2026-08-09 由 Claude 独立复核后新增的内容，**不改写前三部分的任何判决**，
只补充独立核验记录、跨轮汇总和一条尚未被任何一轮检验的轴。

第五部分「附录 D」是 2026-08-09 由 Codex 对合并文档、当前 `HANDOFF.md` 与 Z2/Z5
机器报告所做的独立审计。它保留前四部分原文，但对附录 C 中三处实质性推断错误、两处统计/
术语口径遗漏作显式更正，并给出 Z2/Z5 的当前有效判读。**凡附录 D 明确列为更正的条目，读取
本文件时以附录 D 为准；实验的当前执行状态和项目停止规则仍以 `HANDOFF.md` 为准。**

本文件不替代 `.agents/HANDOFF.md` 的当前连续性权威；当前状态、production 决策与停止规则
以 `HANDOFF.md` 为准，完整科学分析以 `ANALYSIS.md` 为准。

---

# 第一部分：预注册与全部 Round 判决

（中文名：**基于历史结局筛选的反事实 BC 分支功能性正则化**；旧称“选择性 BC 安全行为锚定”）

更新时间：2026-08-08（Asia/Singapore；两份审查合并版；Round Z0、BC Gate B、Round Z2、
Round Z4-A与Z5具体实例均已关闭，Round Z3因matched-control support不足只能判inconclusive；
Round Z6-A snapshot机械Gate 28/28通过；Round Z6-B语义Gate经独立Z6-BR测量裁决通过，停在no-update训练密度Gate：在既有修订基础上，进一步要求
共识 cohort 必须同时是 U44 regression，冻结 identity-only development/validation startpoint split，
重写 Gate B 两类回归的独立准入线，补强 no-op replay 状态合同，固定 Gate C/formal anchor
minibatch 配对，并把 U42--U45 等权平均提升为最先执行的零训练候选；新增最终 1--2 轮优先级与
抗性审查，明确哪些额外方法不准入；合并时保留原审查的显式授权边界）

## 0. 当前状态

| 项目 | 状态 |
|---|---|
| 实验性质 | 单阶段 PPO + 训练期反事实 BC 分支正则化；anchor 数据由**历史 U44 结局离线筛选**并来自Gate B成功BC分支；**不是纯 PPO** |
| 当前阶段 | Round Z0与原BC Gate B失败；Round Z2 top-1 selector关闭；Round Z3 collision-only独立validation因control不足而inconclusive；Round Z4-A历史增量失败；Round Z5 budget-constrained frozen-hidden selector未建立优势并关闭tested实例；Round Z6-A snapshot机械Gate通过；Z6-B语义必要条件经Z6-BR裁决通过；未生成新actor |
| 正式训练授权 | **未授权**。Gate A--D 全部通过后，仍需用户明确授权这一非纯 PPO 目标 |
| 当前 production | 保持 Production U30，不因本方案改变 |
| 训练地图 | 仅 Austin |
| 正式评估 | Austin、Hockenheim、MoscowRaceway、Nuerburgring，各 600 episode，CUDA、ego collision scope |
| 最终 actor | 输入、网络结构、12-key checkpoint 与部署推理方式不变；训练后删除 teacher，不增加运行时模块 |

本文件是已完成到Gate B的候选实验合同与结果记录；§4.3为Gate A，§5.5为Gate B。Gate B失败
后本方向已经停止，不改阈值、不换teacher、不扫loss权重后重试，也没有formal actor。

### 0.1 授权边界与硬停止点

本文件是预注册合同，不是执行授权。合并、审阅或把本文件交给本地 agent，均不自动授权构造
actor、运行评估、修改训练源码或启动训练。后续按以下边界执行：

1. §21 的 U42--U45 等权平均是固定的 Round Z0 候选；只有用户明确授权后，才能一次性构造与
   固定评估。失败后不得改 band、权重或再做三点/加权平均。
2. 若 §21 没有形成相对 U44 的严格 Pareto 改善，只有在用户另行明确接受 §3.3 的 hindsight
   数据边界并授权 Gate A--D 后，才能按 §3.4--§8 顺序执行；每一 gate 只有在前一 gate 完整
   通过后才能开始。
3. **45-update formal training 始终需要独立明确授权。** Gate A--D 全部通过后必须停机、提交
   gate report、代码 diff、测试结果和固定 `beta_anchor`，等待用户授权这一非纯 PPO 训练目标。
4. §3.3 只允许在获得授权后，用历史策略的未来结局构造**离线冻结训练数据**；online PPO
   rollout 和部署期始终严禁读取任何未来信息。
5. Gate A/B/C/D 或 §21 任一实现合同失败时，只修复实现错误；科学 gate 失败时不得改阈值、
   teacher、动作分量、checkpoint band 或数据 split 后重试。
6. §14--§20 与 §22 除明确引用 §21/§13 的部分外均为研究审计，不是可执行队列；agent 不得
   自动切换到 Constrained PPO、counterfactual preference、prefix-reset 或其他候选。


## 1. 要解决的问题

前向走廊门控时间相关速度探索 U44 在四图上把 canonical BC 的 collision 从 `129` 降到
`62`，同时 overtake 从 `1445` 提到 `1478`；但它相对 BC 仍然新造了 `27` 次 collision。
这 27 个场景的配对 BC 结果已从现存四图正式结果重新核对：

下面两张表是**同一批 27 个 episode 的两种交叉分类**，不是同一个划分：

| created-27 的 BC 结局构成 | 数量 |
|---|---:|
| 安全 overtake | **22** |
| 安全 follow | 5 |
| 合计 | 27 |

| created-27 的 regime 与速度构成 | 数量 |
|---|---:|
| off-line（raceline0/2） | **20** |
| same-line（raceline1） | 7 |
| speed scale 0.7 或 0.8 | 18 |

`22/27 = 81.5%` 说明 BC 在这些回归场景中主要不是靠放弃超车来保持安全。与 BC 四图全体
episode 的 `1445/2400 = 60.2%` 比较，双侧 Fisher exact `p=0.0283`；但若只和其他 BC
安全 episode 比较，双侧 `p=0.0684`。因此这里采用的准确表述是：**BC 是有希望的安全行为
参考，但现有 27 个样本不能单独证明锚定必然改善泛化。**

四图 created-27 只能说明目标现象存在。Hockenheim、MoscowRaceway 和 Nuerburgring 不得
参与 anchor cohort、窗口、阈值、loss 权重或训练选择。正式方法必须先在独立的 Austin
训练侧场景上通过 Gate A--D。

### 1.1 created collision 的 checkpoint 稳定性（2026-08-08 复核，影响 Gate A 设计）

用同一走廊 run 的 U42--U45 四图正式结果对同一 BC 基线复算：

| update | collision | inherited | created |
|---:|---:|---:|---:|
| 42 | 66 | 30 | 36 |
| 43 | 63 | 31 | 32 |
| 44 | 62 | 35 | **27** |
| 45 | 61 | 33 | 28 |

created 集合在相邻 checkpoint 之间的 Jaccard 只有 `0.39--0.58`，而 inherited 是
`0.59--0.78`。四点并集 `54`、四点全都出现的核心只有 `14`；按"在几个 checkpoint 上出现"
分布为 `1 次 21 / 2 次 11 / 3 次 8 / 4 次 14`。**U44 的 27 个 created 里只有 14 个在四点
都出现，3 个是 U44 独有。** 逐图核心占比为 Austin `4/9`、Hockenheim `4/7`、
MoscowRaceway `5/9`、Nuerburgring `1/2`。

同时，这四个 checkpoint 在**权重空间上极其接近**：`||theta_U44|| = 162.75`，而
`||U42-U45|| = 0.0347`，相对距离约 `2e-4`；两两相对距离都在 `9e-5--2e-4` 量级。

**这两组数放在一起给出本文件最重要的一条背景事实：相对权重变化约 `1e-4`，就能让约一半的
created collision 身份发生翻转。** 因此 created collision 大部分不是稳定的策略缺陷，而是
处在决策边界上的临界事件。两个直接后果：

1. **Gate A 不能用单一 checkpoint 的失败集定义 anchor cohort**（§4.1 已据此改为多
   checkpoint 共识）；否则约一半锚点落在瞬态失败上，有效剂量被稀释。
2. 任何方法在**单个 checkpoint** 上报告的 created collision 减少，其中很大一部分是
   checkpoint 噪声。这是 §18.5 rule 5"必须报区间"在本方向上的具体理由，也是 §10.1 固定
   U42--U45 四点的理由。

该复核只用已封存的四图正式结果，没有训练、没有新评估。

## 2. 核心假说与唯一新增变量

### 2.1 核心假说

U44 已学到强 same-line 安全行为，但共享 actor 更新使一部分原本由 BC 安全完成的 Austin
通过行为发生回归。若在**训练侧已验证的 BC 安全超车状态序列**上约束新 student 不偏离
canonical BC，同时保留其余状态上的原 PPO 更新，则可能保住走廊探索的 same-line 收益，
并减少 off-line 平行侧后新碰撞。

这是假说，不是已经定位的机制。此前诊断已经否定了“固定 output-head 梯度冲突”、固定
collision-role 临界比例、所测几何/速率的早期稳定可分性，以及“当前几何进入输入但被 GRU
丢失”等解释。本实验不复活这些方向。

### 2.2 唯一新增训练变量

正式臂只增加一个目标：

```text
actor_loss = ppo_actor_loss + beta_anchor * bc_safe_anchor_loss
```

保持不变：

- canonical BC fresh-start；
- Austin-only 场景、现有 479 collision cache、ordinary 场景和 50/50 role 调度；
- 当前四项 reward，不新增 reward、cost critic 或 terminal 规则；
- `privilege_gru` critic；
- 前向走廊门控时间相关速度探索：speed std `0.15`、hold `50` steps、front gap `2.0m`；
- actor 输入 361D、原 End2Race GRU/output head、固定 pressure `k`；
- seed 42、45 updates、现有 PPO 超参数；
- 不使用 U30/U44 作为动作 teacher，不使用运行时 gate、shield、future collision 或测试地图标签。

U44 只允许在 Gate A 中作为 **Austin 训练侧回归状态的生成器和筛选参照**，不能提供目标
动作；唯一动作 teacher 是冻结的 canonical BC。

## 3. 固定输入与数据隔离

### 3.1 Gate A/B 的 Austin 训练侧面板

固定使用：

```text
post-trained/panels/heldout_hard_v1/train_difficult_scenarios.json
```

该文件含 868 个 Austin difficult scenarios，覆盖 91 个实际入选 train startpoint；其
startpoint 母集合为预先封存的 100 个 train split startpoint。面板由 Production U30 对
21,600 个候选作碰撞/near-miss筛选产生，并非 BC 或 U44 的自然分布样本。因此：

- 它只用于构造训练侧 anchor 和机制 gate；
- 不拥有正式验收权；
- 不把其中的成功率外推成四图自然分布成功率；
- 不使用 `heldout_eval` split 构造 anchor。

Gate A 在执行任何 actor 评估前，先按 §3.4 对 startpoint 做 identity-only 冻结切分。
canonical BC 与 **U42--U45 四个 checkpoint** 只在 development startpoints 对应场景上做
fresh deterministic CUDA 评估并保存 numeric traces（共 `5 * N_dev_scenarios` episode）；
validation startpoints 在候选 actor 与主 checkpoint 冻结前不得运行 BC/U42--U45 或任何
treatment。四个 checkpoint 是 §4.1 共识 cohort 所必需的，理由见 §1.1。现有脚本可直接消费
过滤后的固定 panel；不为此新建改变场景语义的 wrapper。

### 3.2 不可使用的数据

- 四图 created-27 的 Hockenheim/MoscowRaceway/Nuerburgring episode；
- near400、hard73、hard334 的 held-out/eval split；
- collision 发生后的 observation；
- critic privileged state、opponent planner 内部状态或 oracle action 作为 actor 输入；
- U30/U44 输出作为 teacher target。

### 3.3 Future information：明确区分在线与离线

本方法**确实使用未来结局**，但只在离线、且只对冻结的历史轨迹使用。为避免自相矛盾，规则
按下述两分法执行，不得再写成对 future information 的一概禁止：

**禁止（在线／部署）**

- formal training 的 online PPO rollout 读取当前 student 的 future collision、future outcome
  或任何前瞻量；
- 部署期出现 gate、shield、oracle、teacher 或任何运行时后处理；
- 用未来事件在 formal training 中重新计算或调整 anchor window。

**允许（离线／历史，且必须显式声明）**

- 在**预先冻结的 U44 Austin train-side traces** 上，用历史 future event 离线定义固定
  anchor window（§5.2）；
- 用 U44 的最终结局离线筛选 anchor cohort（§4.1 的 `U44 regression` 判据同样是 hindsight
  选择，不只是窗口）；
- 上述筛选结果在生成 anchor dataset 时一次冻结，formal training 只消费冻结产物。

因此本方法的准确名称是 **hindsight-selected counterfactual BC-branch regularization**，不能称为
"不使用未来信息"。启动 Gate A 之前需要用户明确接受"训练数据可由历史策略的未来结局离线
筛选"这一条；这与 GUIDE §1 禁止的"正式结果依赖部署期未来信息"是两件事。

### 3.4 Development / validation startpoint split（必须先于任何模型评估）

`train_difficult_scenarios.json` 中实际出现的 unique `ego_idx` 只按 identity 划分，不能查看
BC/U42--U45 outcome 后再切分：

```text
score(ego_idx) = SHA256("bc-anchor-v1|Austin|" + str(ego_idx))
```

按 score 从小到大排序，前 `ceil(0.20 * N_unique_startpoints)` 个作为 **validation**，其余作为
**development**。生成并冻结：

```text
anchor_startpoint_split_v1.json
  split_version
  source_panel_sha256
  development_ego_idx
  validation_ego_idx
  scenario_keys_by_split
```

硬规则：

- Gate A、Gate B、anchor dataset、Gate C、beta、所有阈值和任何实现调试只使用 development；
- validation 不进入 teacher 接管、safe-control选择、动作窗口、anchor loss 或 shadow update；
- validation 只在正式 candidate 的 U44 checkpoint 已预先冻结后运行一次，同时评估 BC、历史
  U44 与 candidate；不得用于早停、改 beta、改 checkpoint 或决定是否继续训练；
- 两个 split 的 `ego_idx` 和完整 scenario key 必须零交集；manifest 在第一次 actor eval 前写盘，
  后续不得重算。


## 4. Gate A：Austin 训练侧 anchor 是否存在

### 4.1 cohort 定义

先按 BC 结果定义候选，再查看 U44，避免先看 treatment 结局后任意挑 BC 行为：

```text
BC-safe overtake
  = BC 无 ego collision
  AND BC outcome == overtake
  AND opponent raceline 为 raceline0 或 raceline2
```

在上述候选中，定义单 checkpoint regression：

```text
regression(u)
  = U(u) 出现 ego collision
  OR U(u) outcome != overtake            u in {42,43,44,45}
```

**主 anchor cohort 采用多 checkpoint 共识，而不是 U44 单点**（理由见 §1.1：单点 created
集合约一半是 checkpoint 瞬态）：

```text
consensus regression
  = regression(44)
  AND |{u in {42,43,44,45} : regression(u)}| >= 3
```

`regression(44)` 是硬条件：状态前缀、未来事件窗口和 teacher 接管都以 U44 为源。若某场景只在
U42/U43/U45 中回归而 U44 本身安全超车，则不能进入本 cohort；否则没有一致的 U44 adverse
prefix 或 §5.2 窗口可供锚定。

因此 Gate A 需要在 §3.4 的 **development** split 上评估 canonical BC 与 U42--U45，
而不是只评 U44。状态前缀与 anchor 数据仍统一取 **U44**——它是 §10.1 预注册的主
checkpoint；共识只决定**锚在哪些场景**，不决定用谁的动作或前缀。validation split 保持未读。

主 anchor cohort 是 BC-safe overtake 候选与 consensus regression 的交集。所有 speed scale 均保留，并固定按 raceline、speed scale、
collision / lost-overtake 分层报告；不得看完结果后只保留最好的一层。

**该 cohort 内含两类方向相反的回归，分层报告不是格式要求而是安全要求。** `U44 出现 ego
collision` 要求 student 更保守，`U44 outcome != overtake`（U44 安全但改为跟车）要求 student
更进取，而 §6.3 的 anchor loss 对两者一视同仁。§4.2 条件 4 只保证碰撞类占比不为零，真正
检验这一张力的是 §5.4 的双准入线（救回 `>=50%` 碰撞 **且** 超车保有 `>=80%`）。Gate A 与
Gate B 的所有结果必须按这两类**分开报告**，任一类单独失败即视为 Gate 失败，不得用另一类
的改善抵消。

### 4.2 准入线

Gate A 通过必须同时满足：

1. **共识** anchor cohort 至少 20 个 episode；
2. 至少覆盖 10 个不同 ego startpoint；
3. raceline0 和 raceline2 均至少出现 3 个 episode；
4. 共识 cohort 中至少 8 个在 U44 上是 ego collision，不能全部只是 overtake→follow；
5. BC 轨迹、U42--U45 轨迹、episode identity 和 terminal contract 全部完整且有限；
6. **稳定性对照**：同时报告 U44 单点 cohort 大小与共识 cohort 大小。若 U44 单点 `>= 20`
   但共识 `< 20`，判定为 **Gate A 失败**——这说明 train-side 的回归同样以瞬态为主，
   本方法缺少稳定的锚定对象，应停止而不是退回单点定义。

任一条件失败：停止本方向，不改 cohort 定义或把四图测试 episode 补进训练数据。

### 4.3 执行结果（2026-08-08）

Gate A已按上述固定合同执行并**通过**。在任何actor replay前，868条Austin训练侧困难场景按
identity-only hash冻结为development `718 episode / 72 ego startpoint`与validation
`150 / 19`，起点和scenario key均零交集。Gate A只读取development；validation没有运行任何
actor评估，也没有参与cohort、阈值或分层选择。

canonical BC与U42--U45均以fresh deterministic CUDA、8秒、ego collision scope保存trace。
五臂aggregate的`ego collision / overtake / follow`依次为：BC `308/318/92`、U42
`191/377/150`、U43 `187/385/146`、U44 `214/373/131`、U45 `202/377/139`。五臂共3,590条
actor-episode全部通过result/trace/panel key集合、episode identity、有限数值、数组对齐、
collision marker和terminal row合同，0 error且无partial结果。

BC-safe overtake为308条；U44单checkpoint回归为46条；满足`regression(44)`且U42--U45至少
3/4回归的主cohort为28条，覆盖21个ego startpoint。固定分层如下：

| stratum | count |
|---|---:|
| raceline0 / raceline2 | `20 / 8` |
| U44 ego collision / lost-overtake | `19 / 9` |
| speed `0.45/0.50/0.55/0.60/0.65` | `1/1/0/3/2` |
| speed `0.70/0.75/0.80/0.85` | `2/5/7/7` |

0.55层的0是完整报告后的真实零计数，不是筛后删除。六条§4.2准入线全部通过；`C=19`与`L=9`
也都满足§5.4要求进入Gate B前至少8条的样本数前提。
这只证明存在稳定锚定对象，不证明BC从U44状态prefix接管有效。28条development cohort已经冻结；
不得退回46条U44单点集合或从validation补样本。按本次用户要求，执行停在最近节点Gate A并先
汇报，Gate B尚未实现或启动。

## 5. Gate B：BC 动作在 U44 回归状态上是否真能救援

BC 从场景起点完整运行成功，只说明存在另一条安全轨迹，不能证明 BC 动作适用于 U44 已经
访问的状态。Gate B 必须做闭环干预。

### 5.1 teacher 与 hidden 合同

- U44 和 BC 分别维护自己的 GRU hidden；
- 在接管前，两者读取同一条 U44 observation prefix，各自递归更新 hidden；
- 接管后，BC 根据反事实 simulator 产生的新 observation 继续递归，不能重复使用 BC 自己原始
  轨迹上的 hidden；
- student/U44 hidden 同样沿反事实 observation 继续推进，保证接管窗口结束后可恢复 U44；
- 所有执行动作都是确定性 mean action，不采样探索噪声。

### 5.2 固定干预窗口

- U44 ego collision：从 collision 前 1.5s 开始，执行 BC mean action 1.5s；
- U44 未碰撞但丢失 overtake：以 U44 最小 surface clearance 时刻为中心，执行从其前 1.0s
  到后 0.5s 的 BC mean action；
- 之后恢复 U44，继续到正常 episode 终点；
- 只使用事件时刻定义离线因果测试窗口，正式训练 mask 不允许读取未来事件。

### 5.3 固定分支

每个主 anchor episode 跑五个分支，**branch 0 必须先通过**：

0. **no-intervention replay**：用与其余分支完全相同的 branch runner、hidden 推进和环境
   重放，但全程执行 U44 动作；
1. 完整 BC steering + speed；
2. 仅 BC steering，speed 保持 U44；
3. 仅 BC speed，steering 保持 U44；
4. U44 原始闭环（保存的历史 trace，仅作参照）。

**branch 0 的准入线（不通过则 Gate B 无效，先修实现）：** 相对保存的 U44 trace，
所有 `action_applied=true` 行必须同时满足：ego raw/executed action 最大绝对误差为 `0`；
opponent executed action、ego/opponent pose、measured speed 与 360D LiDAR 采用
`rtol=0, atol=1e-6` 的逐行一致；outcome、首次 ego collision 类型/时间步、episode 长度和
terminal contract 完全一致。若原 trace 未保存某一状态字段，必须在 report 中列明，不能用
未检查冒充通过。保存的 trace **不能替代 branch 0**——它没有验证新执行代码本身。§30.4 与
§32.2 都出现过因首帧速度合同写错而产生 `4.85` / `5.45` 动作误差的先例，缺少该分支就无法
把 BC 接管效果与 runner 偏差分离。

第 2/3 分支只用于解释，不用于事后选择不同训练臂。若完整 BC 分支失败，不得以某个组件偶然
更好为由直接改成 steering-only 或 speed-only；那属于新方案，需要重新预注册。

另选等量的"BC 与 U44 均安全 overtake"episode 作正对照。controls 只从 development 中取，
按相同 opponent raceline、speed scale 后，以 circular ego-startpoint distance 最小为第一排序、
scenario key 哈希为 tie-break，优先不复用；不足时 Gate B 失败，不放宽到 validation。每条 control
以自身现有 evaluator 定义的 `minimum_obb_clearance` **第一次**达到全局最小时刻为中心，使用
前1.0s到后0.5s的固定窗口，并施加相同 branch 0/BC 接管协议。这样 safe control 的窗口不依赖
源失败episode的未来时刻，也不会因没有collision而留下未定义接管点。

**窗口边界规则（生成 anchor dataset 前一次固定，不得逐 episode 裁量）：**

- collision 发生时刻早于 `1.5s` 时，窗口起点截断到 episode 第一个 `action_applied=true` 行；
- minimum surface clearance 出现并列最小值时，取**第一个**出现的时间步；
- 窗口越过 episode 终点时截断，`terminal_post_step` 行**绝不**作为动作行；
- 截断后窗口内 `action_applied=true` 的步数少于 `50`（`0.5s`）的 episode 直接剔除，
  并在 Gate A/B 报告中计数，不得静默丢弃。

### 5.4 准入线

先定义 development 主 cohort 的两个互斥子集：

```text
C = U44 ego collision regression
L = U44 无 ego collision、但 outcome != overtake 的 lost-overtake regression
```

完整 BC 分支必须同时满足（全部用整数计数，不能池化抵消）：

**Collision stratum C**

- 至少 `ceil(0.50 * |C|)` 个从 ego collision 变为非碰撞；
- 在被救回的 episode 中，至少 `ceil(0.80 * N_rescued_C)` 个最终仍为 overtake，而不是只通过
  减速变成 follow；
- 被救回的 collision 至少来自 2 个不同 ego startpoint，且 raceline0 / raceline2 各至少 1 个；
  若两条 raceline 各自的 C 样本均不少于4，则各至少2个被救回。

**Lost-overtake stratum L**

- 新造 ego collision 至多 `floor(0.05 * |L|)` 个；
- 至少 `ceil(0.80 * |L|)` 个恢复为 overtake；
- 恢复结果至少覆盖2个不同 ego startpoint，且不能全部来自同一 raceline。

**Matched safe controls**

- 新造 ego collision 至多 `floor(0.05 * N_control)` 个；
- overtake 损失至多 `floor(0.05 * N_control)` 个。

Gate A 后若 `|C| < 8` 或 `|L| < 8`，直接停止：这两类无法支撑 §9.2 的独立 loss 与 Gate C 的
分层 minibatch。C、L、safe controls 任一组未达标即 Gate B 失败，不得用另一组改善抵消，
也不得改成 steering-only / speed-only 或更换窗口后重试。

### 5.5 执行结果（2026-08-08，科学失败并关闭）

Gate A的28条cohort全部满足窗口至少50个动作步，剔除0条；从development的262条BC/U44共同
安全overtake中按§5.3无放回匹配28条controls。branch 0先独立完成56条正式replay：ego raw与
executed action、opponent action、两车pose/speed及双360D LiDAR的最大绝对误差全部为0，
outcome、首次ego collision、长度、marker和terminal合同逐条一致，故干预结果有效。

完整BC分支结果：

| stratum | 实测 | 准入线 | 判决 |
|---|---:|---:|---|
| C collision救回 | `10/19`，且10条全为overtake | 至少10条；救回中至少80% overtake | 通过 |
| C覆盖 | 9个起点；raceline0/2=`8/2` | 至少2起点；两线各至少2条 | 通过 |
| L恢复overtake | **`0/9`** | 至少8条 | **失败** |
| L新造collision | `0/9` | 至多0条 | 通过 |
| safe controls | `0`新collision，**2条overtake→follow** | 至多1条新collision、1条overtake损失 | **失败** |

diagnostic component branch为：steering-only在`C/L/control`分别得到`7/19、1/9、28/28`
overtake；speed-only为`5/19、1/9、26/28`，并在control新造1次collision。它们不改变正式branch
失败，也不允许事后替换方法。四branch共224条trace全部通过key、finite、数组对齐、marker、
terminal、冻结窗口、action-source与selected/executed action合同；validation没有运行branch。

Gate B因此按§5.4科学失败：不生成anchor sequence dataset，不进入Gate C/D，不申请45-update
formal训练；不得删除L、放宽control门、更换窗口或改成单分量后重试。正面机制只限于“完整BC
在同一U44 prefix上能把10条碰撞救成超车”，不能外推为统一teacher有效。

## 6. Anchor 数据与 loss 合同

### 6.1 必须使用真实 counterfactual BC branch sequence

Gate B 的完整 BC 分支一旦执行第一步，后续 observation、measured speed、opponent response 与
GRU hidden 都会偏离保存的 U44 原轨迹。因此**不得**把“BC 在成功反事实分支上的后续动作”
错误地监督到“U44 原轨迹的后续 observation”上；那会产生 state/action mismatch。

Gate B 通过后，只把逐episode严格改善的 branch 1 保存为 anchor：

```text
C-anchor：U44 ego collision -> BC branch 无 ego collision且最终 overtake
L-anchor：U44 安全 follow   -> BC branch 仍无 ego collision且恢复 overtake
```

collision只被救成follow的分支可计入Gate B安全诊断，但**不进入anchor dataset**，因为本项目还
要求提升/保持overtake。任一stratum最终可用anchor少于8条即停止，不以未改善episode补足。

每条anchor sequence由两段组成：

1. 从episode起点到干预开始：branch runner执行U44，保存真实U44 prefix observation；
2. 干预窗口：执行完整BC branch，保存该反事实分支自己产生的observation与BC action。

窗口结束后的恢复U44段只用于判定最终outcome，不进入anchor loss。这样训练target与Gate B中
真正造成改善的状态/动作严格配对。该方法更准确地属于
**hindsight-selected counterfactual BC-branch regularization**；仍然不是纯PPO。

每条序列保留：

- scenario identity、regression stratum与episode-start mask；
- float32的361D mixed observation sequence（U44 pre-branch + BC counterfactual branch）；
- intervention-start index、valid timestep mask和anchor window mask；
- 冻结BC在同一mixed sequence上递归产生的latent steering mean与physical speed mean；
- U44/BC branch最终outcome、collision类型与progress，仅用于审计；
- branch-0与branch-1 replay contract摘要。

student hidden不得预计算或存入数据。每次actor update都必须由当前student参数从sequence起点
递归mixed observation得到；teacher BC也必须按同一sequence从起点递归。不得在干预开始处清零
hidden，不得使用BC原始自然轨迹hidden，也不得在第一步以后退回U44原trace observation。

### 6.2 Anchor window

loss只在Gate B固定的1.5s BC干预窗口内计算；U44 pre-branch段仅用于recurrent burn-in，不计
loss。窗口identity与counterfactual branch observation在生成dataset时一次冻结，formal training
不重新运行simulator、不读取当前student未来collision，也不按训练loss重新选择episode。

### 6.3 Loss

现有策略对 steering 使用 latent Gaussian，对 speed 使用 physical Gaussian，且两个 std 固定为
`0.03` 和 `0.15`。因此用同方差 Gaussian 的 mean KL：

```text
bc_safe_anchor_loss
  = mean_valid(
        0.5 * ((latent_steer_student - latent_steer_BC) / 0.03)^2
      + 0.5 * ((speed_student - speed_BC) / 0.15)^2
    )
```

其中 `latent_steer = atanh(clamp(physical_steer / steer_bound))`，必须复用现有 distribution 的
变换与 epsilon，不能另写近似公式。raw unit 下 steering 系数比 speed 大约25倍，表示“一倍
探索标准差的偏离等价”，不解释为steering重要性高25倍。

两个动作分量分别报告loss、梯度范数和Gate B结果，但formal run固定使用完整二维loss；不做
steering-only/speed-only训练sweep。

## 7. Gate C：固定 beta 的 shadow update

canonical BC 初始化时 anchor loss **精确为零**——student权重等于BC，且两者读取完全相同的
mixed U44/BC-branch sequence、hidden递归相同，因此动作逐位相同——所以初始化点不能用于判断
保护作用。Gate C 固定
使用 U44 actor 作为“已经漂移的 student 代理”，在 Austin 训练侧真实 PPO rollout 上只计算
梯度，不执行 optimizer step。

由此产生一个必须承认的性质：`beta_anchor` 是在**训练末期漂移量**上标定的，而训练前段
anchor 项接近惰性，机制只在 student 漂移积累后才真正介入。本方案接受该性质，不因此改为
自适应 beta；相关限制记入 §12。

### 7.1 beta 的一次性确定

**固定的 rollout 与 minibatch 集合（消除"哪四个"这一自由度）：** actor 与 critic 均取
`post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update44/` 的
`actor.pth` 与 `critic.pt`；exploration 为 `corridor_temporal`（std `0.15`、hold `50`、
gap `2.0m`）；scenario 队列取该 run 的 `ordinary_scenarios.json` 与
`collision_scenarios.json`；seed 42、`n_envs=16`、`n_steps=6400`、`batch_size=12800`。
`16*6400/12800 = 8`，因此使用**第一个 actor epoch 的全部 8 个 minibatch**，不再从中挑选。
参数冻结 ⇒ `ratio ≡ 1`、ratio-clip 恒不激活，语义与 §24 的 update-entry gradient 一致。

**Anchor minibatch 与 PPO minibatch 的固定配对。** C/L episode 分别按
`SHA256("anchor-sequence-v1|" + scenario_key)` 排序。Gate C 的第 `i` 个 PPO minibatch 固定配
第 `i` 个 anchor minibatch；每个 anchor minibatch 含 C 8条、L 8条，按各自排序循环且在一轮
耗尽前不重复。正式训练沿用同一 sampler：每个 update 的16个actor step按
`(update, actor_epoch, minibatch_index)` 生成确定性循环位置，不读取当前 loss 或 outcome。
若任一 stratum 少于8条，按 §5.4 停止。Gate C report 必须保存8个 anchor minibatch 的完整
scenario-key清单，不能只保存seed。

**LR 加权的 step-space 范数。** actor 有两个 parameter group（GRU `3e-6`、head `3e-5`，
相差 10 倍），且 §24 实测 head 梯度范数本就比 GRU 大一至两个数量级。裸梯度范数比不反映
实际参数步长，因此一律使用：

```text
||g||_eta = sqrt( eta_gru^2 * ||g_gru||^2 + eta_head^2 * ||g_head||^2 )
```

beta 由 8 个 minibatch 的中位数一次确定：

```text
beta_anchor
  = 0.25 * median( ||g_ppo||_eta / max(||g_anchor||_eta, 1e-12) )
```

该公式只执行一次，不扫 `0.1/0.25/0.5`，不根据四图或 near400 结果修改。`0.25` 表示 anchor
在准入点的目标 **step-space** 贡献约为 PPO actor step 的 25%，是本方案唯一预注册的辅助剂量。

### 7.2 Shadow-update 准入线

**虚拟 step 的显式定义（clip-aware）。** 正式训练在 `MAX_GRAD_NORM = 0.5` 上对
`policy.actor_parameters` 整体做全局 clip，加入 anchor 会改变 clip 乘子并**间接压缩 PPO
分量**。因此虚拟 step 必须先合成、再跑与正式代码完全相同的 clip，最后按 group LR 缩放：

```text
dTheta_0 = - LRGroup[ Clip_0.5( g_ppo ) ]
dTheta   = - LRGroup[ Clip_0.5( g_ppo + beta_anchor * g_anchor ) ]
```

只用一阶方向导数判定，不重新 forward（冻结参数且无历史 Adam state 时，重新 forward 的
loss 变化并不比一阶内积更接近真实 Adam step）。这只是 LR 加权、clip-aware 的 SGD proxy，
**不得写成真实 Adam step**。

准入线（`g_same` / `g_offfast` 为 §7.3 分解出的 per-regime PPO 梯度；`g_anchor,coll` 与
`g_anchor,lost` 为两类回归各自的 anchor 梯度）：

1. **保留 same-line 收益**：`-g_same^T dTheta >= 0.8 * (-g_same^T dTheta_0)`，且
   `-g_same^T dTheta_0 > 0`；
2. **anchor 两类分别生效**：`-g_anchor,coll^T dTheta > 0` **且**
   `-g_anchor,lost^T dTheta > 0`；不接受两者池化后为正；
3. **不反转 off-line-fast**：`-g_offfast^T dTheta >= 0`；
4. **稳定性**：上述 1--3 每一条都必须在 **8 个 minibatch 中至少 7 个**成立；同时报告全部
   8 个的逐 minibatch 数值，不得只给池化平均；
5. `||Clip_0.5(g_ppo + beta*g_anchor)||` 相对 `||Clip_0.5(g_ppo)||` 的 step-space 比值
   记录并报告（clip 后二者可能都被压到 `0.5`，该比值不再是独立判据，只作诊断）；
6. steering 与 speed 任一分量不得贡献超过 95% 的 `||g_anchor||_eta`。

若不存在同时满足这些条件的固定 beta：停止。不得自适应调 beta，也不得改为 Lagrangian、
PCGrad、双 head 或额外 optimizer。

### 7.3 实现前置依赖

准入线第 2 条需要按 regime 拆分 PPO 梯度。该能力由 §24 的 count-weighted 分解提供，但其
一次性脚本 `scripts/diagnose_ppo_regime_gradients.py` 已在 `77fc29b` 删除，Gate C 前必须
按 `ANALYSIS.md` §24 与 `EXPERIMENTS.md` 的合同重建，并保留其硬校验：

- 分组损失必须是 `loss_terms[mask].sum() / total_valid_count`，不得对各子集分别取 `.mean()`；
- 各 regime 梯度之和必须重构完整 PPO 梯度，最大分解误差 `> 2e-5` 直接 raise；
- regime 标签复用 `ppo/scenarios.py` 的 `is_same_line` / `is_offline_fast`，为 per-transition
  掩码，不得用 episode 级标签近似。

## 8. Gate D：关闭 anchor 时逐位等价

在接入 formal training 前，`beta_anchor=0` 必须对现有前向走廊门控时间相关速度探索完成一个
formal update 的逐位等价回归：

- 相同 scenario order 与 env seed；
- rollout mean action、sampled action、old log-prob、reward、value、GAE 一致；
- actor 与 critic checkpoint tensor 逐项一致；
- PPO metrics 一致；
- anchor RNG、数据 loader 和 teacher forward 在关闭时不得被执行。

这条只证明新代码关闭时没有改旧路径。它不证明 anchor 有效。失败则先修实现，不得训练。

## 9. 唯一正式训练臂

Gate A--D 全部通过并获得用户明确授权后，才允许创建 `run.sh` 和最小代码改动。
不重复训练一条新的 anchor-off control。该省略依赖一条必须写明的推理链：**Gate D 证明
一个 formal update 逐位等价，加上 §8 要求的“关闭时不消耗 anchor RNG、不执行数据 loader 与
teacher forward”，再加上固定 seed 与确定性代码路径，才能推出整条 45-update 轨迹等价。**
三个前提缺一，该省略即不成立，必须补跑 anchor-off control。正式比较使用已封存的 U44 轨迹
与四图结果。

### 9.1 实验身份

```text
EXPERIMENT_ID=front_corridor_temporal_bc_safe_anchor
```

### 9.2 固定训练配置

| 参数 | 固定值 |
|---|---:|
| 初始化 | `pretrained/end2race.pth` canonical BC |
| map | Austin |
| seed | 42 |
| critic | `privilege_gru` |
| logical envs / workers | `16 / 16` |
| steps per env | 6400 |
| batch size | 12800 |
| formal updates | 45 |
| actor / critic epochs | `2 / 5` |
| GRU / head / critic LR | `3e-6 / 3e-5 / 3e-4` |
| steering latent std / speed physical std | `0.03 / 0.15` |
| speed exploration | front-corridor temporal, hold 50 steps, gap 2.0m |
| gamma / GAE lambda / clip | `0.999 / 0.995 / 0.20` |
| collision cache | canonical BC Austin collision pool 479 |
| reward | 当前 production 四项，不改权重 |
| 唯一新增项 | fixed `beta_anchor` 的二维 BC-safe anchor loss |

每个 actor minibatch 固定配一个 anchor minibatch。anchor minibatch 含 16 条完整 observation
sequence；只在冻结的 1.5s mask 上计 loss，但从序列起点递归 hidden。序列按固定 seed 循环，
不按当前 loss 或 outcome 做 prioritized sampling。

**两类回归的 batch 构成必须在 Gate C 之前固定，不得看结果后更改。** 本方案预注册
**固定分层权重**：

```text
bc_safe_anchor_loss
  = 0.5 * L_collision_regression + 0.5 * L_lost_overtake
```

理由是 §4.1 已经指出两类的安全压力方向相反，若采用自然频率，样本较多的一类会淹没另一类，
而自然频率在 Austin train-side panel 上的比值事先未知，等于把一个隐含超参交给数据。
`0.5/0.5` 本身也是一个设计选择，但它是**预先声明**的，不随结果调整。若某一类在 Gate B 后满足§6.1严格改善、可进入anchor dataset的episode少于8个，停止而不是
退回自然频率或用未改善分支补足。

**聚合顺序固定为"先 episode 内、后 episode 间"**：先对每条序列在其有效窗口内求平均 loss，
再对 episode 求平均。不得把所有有效 timestep 直接池化——被边界截断的窗口步数不同，池化会
让它们获得不同权重。

## 10. 正式评估与验收

### 10.1 Checkpoint 合同

预先固定评估 U42、U43、U44、U45，不从中挑最低 collision。U44 是主 checkpoint；其余三个
只用于判断后期方向与波动。所有 checkpoint 必须来自同一次 canonical-BC fresh 45-update run。

### 10.2 正式 panel

四张地图各 600 episode，全部使用 CUDA、ego scope、确定性 mean action并保存 numeric traces：

- Austin600：正式验收；
- Hockenheim600：正式验收；
- MoscowRaceway600：正式验收；
- Nuerburgring600：正式验收。

每张地图报告 collision/overtake、ego-opp/ego-wall/opponent-wall、相对 BC 和 U44 的
removed/created、lost/gained及配对 exact test。四图合计只作汇总，不能覆盖逐图失败。
near400只在 U44 主 checkpoint报告一次机制副作用，不拥有正式否决权。

### 10.3 三层判决

| 层级 | 条件 | 含义 |
|---|---|---|
| 实现通过 | Gate A--D 全部通过 | 允许消耗一次正式训练预算 |
| 机制通过 | 相对U44，四图 created collision 减少、same-line collision 不反弹、overtake 不下降，且**保留出的 Austin validation startpoint split** 上同向 | 行为保持机制有作用，但不自动成为production |
| 最终目标通过 | 四图合计`collision < 40`且`overtake > 1500`，并且每图collision不高于BC、overtake不低于BC | 达到用户定义的正式目标 |

**直接锚定范围与乐观 floor（必须与上表一起阅读）。** 边缘统计不能确定交集，因此这里给出
四图 created-27 的真实三维联合交叉表：

| | BC-overtake | BC-follow | 小计 |
|---|---:|---:|---:|
| **off-line（raceline0/2）** | **20** | 0 | 20 |
| same-line（raceline1） | 2 | 5 | 7 |
| 小计 | 22 | 5 | 27 |

§4.1 的 anchor cohort 要求 BC 为安全 overtake 且 opponent 在 raceline0/2，因此直接锚定
范围是 `created ∩ off-line ∩ BC-overtake = 20`。仅凭边缘数（20、22、27）只能推出
`15 <= N <= 20`、floor `[42,47]`；真实联合表把它定在 `N = 20`。

U44 四图 `62` 次 collision = `inherited 35 + created 27`。**在"只修复直接锚定范围、且完全
不影响其他场景"这一强假设下**，乐观 collision floor 为 `62 - 20 = 42`，仍不满足 `<40`；
要到 `35` 必须连 same-line created（7）一起消除，而 §2.1 明确不锚定 same-line。

**这些是四图评估 episode，而 anchor 来自 Austin train-side panel，两者不是同一批 episode。
因此上表只刻画问题射程，不是训练效果的上界，也不承诺跨地图泛化。**

**两个目标是耦合的，不能各自取最优。** BC 到 U44 的 overtake `lost 41` 构成为：

| | U44 碰撞 | U44 安全但跟车 | 小计 |
|---|---:|---:|---:|
| off-line | 20 | 14 | **34** |
| same-line | 2 | 5 | 7 |

锚定射程内是 `34`，所以乐观 overtake ceiling 是 `1478 + 34 = 1512`（**不是 `1519`**——那
包含 7 个射程外的 same-line）。而 off-line 的那 20 个"U44 碰撞"与上表的 20 个 off-line
created **是同一批 episode**，同时计入两个指标：

- 若锚定把它们修成**超车** → 乐观 `42 collision / 1512 overtake`；
- 若修成**跟车** → `42 collision / 1492 overtake`，overtake 目标不达标。

因此"最终目标通过"这一层在本臂射程内至多**边缘可达**，且要求那 20 个 episode 全部恢复为
成功超车。预注册时即承认这一点，不得在结果出来后把 `42` 附近的成绩重述为"接近目标"。

**"anchor loss 在训练中下降"不是机制证据。** 同一份冻结 offline dataset 反复训练，训练
loss 下降几乎必然发生，也可能只是记住了这些 prefix。它一律记为**实现与优化 telemetry**。
机制证据只能来自：四图 created collision 变化、same-line / off-line-fast / off-line-slow
分层、lost/gained overtake 的配对身份，以及 §3.4 预先冻结、完全不进入 Gate A/B/C 或 anchor
训练的 Austin train-side validation startpoint split。validation 只在 candidate U44 已冻结后
运行一次 BC / 历史U44 / candidate，不用于早停、beta、checkpoint或是否继续训练；不得事后划分。

若只优于 U44 的 `62/1478`、但未达到 `<40/>1500`，记录为"扩展经验前沿但未完成最终目标"，
不得把较弱改进写成目标达成。若相对 U44 collision 与 overtake 双轴均不改善，直接否决。

## 11. 停止规则

1. Gate A、B、C 或 D 任一失败：不正式训练，不修改阈值或 beta 重试。
2. 正式 run 只允许一条；不追加 beta sweep、component sweep、hold/gap/std sweep。
3. 不从 U42--U45 事后挑单点 production；主判读保持 U44，band 只作稳定性证据。
4. 若正式训练未达到最终目标，完整记录机制结果与四图配对身份后关闭本方向；只有独立的新
   证据指出 anchor cohort、teacher 质量或实现合同有误才可重开。
5. 不把失败转成 shield、运行时 BC fallback、multi-map PPO、二阶段 U44 微调或 test-map
   定向训练。

## 12. 已知限制

- Gate A 面板是 Production U30 筛出的困难集，样本不是 Austin 自然分布；它只提供训练侧
  adverse states，不提供接受证据。
- Anchor sequence 来自“U44 pre-branch + 成功BC counterfactual branch”的mixed prefix。正式
  student仍可能访问新的状态，离线functional regularization不能保证覆盖所有闭环漂移。
- 使用 U44 筛选训练侧 regression，使方法依赖一条既有 PPO 轨迹；虽然正式 student 从 BC
  fresh-start，但应准确称为“使用历史 student 失败状态构造的单阶段 PPO”，不能称为完全
  无先验的纯 PPO。
- 四图 created-27 只用于提出问题和最终评估，不能进入训练 cohort；因此 Austin Gate A
  找到的 anchor 是否能跨地图迁移仍是正式实验的核心未知。
- seed 固定为42。checkpoint band用于观察训练轨迹稳定性，不提供多 seed 不确定性。
- **本臂的 collision 目标不可达，overtake 目标仅边缘可达。** 见 §10.3 的联合交叉表：直接
  锚定范围 `N = 20`，乐观 collision floor `42`（达不到 `<40`）；overtake ceiling `1512`
  （不是 `1519`），且要求 20 个 off-line created episode 全部恢复为成功超车，若修成跟车则
  只有 `1492`。两个指标共享同一批 episode，不能各自取最优。
- **本方法使用历史未来结局离线筛选训练数据**（窗口与 cohort 均是，见 §3.3）。它不违反
  部署期约束，但必须以 hindsight-selected counterfactual BC-branch regularization 的名义报告，不能称为
  "不使用未来信息"。
- **`beta_anchor` 的标定点不具代表性。** 它在 U44 这个已漂移代理上确定，相当于按训练末期
  漂移量定剂量；canonical BC 初始化时 anchor loss 精确为零，训练前段该项接近惰性。本方案
  接受固定 beta 带来的前后段剂量不一致，不改为自适应。
- **anchor cohort 混合了两类安全压力相反的回归**（新造碰撞 vs 丢失超车），单一二维 loss
  同时承担两个方向。§4.1/§5.4 的分层与双准入线是唯一约束，若正式训练出现"碰撞降低但超车
  同步下降"，应优先怀疑该混合，而不是先调 beta。
- **Gate C 依赖一份已删除的实现。** §24 的 per-regime count-weighted 梯度分解脚本已在
  `77fc29b` 删除，必须按 §7.3 重建后才能执行 Gate C。
- **锚定对象本身可能大部分是临界事件。** §1.1 显示相对权重变化 `1e-4` 即可翻转约一半
  created collision 身份。§4.1 已改为多 checkpoint 共识以去掉这部分噪声，但共识 cohort
  仍来自同一条走廊 run 的四个相邻 checkpoint，不能排除整条 run 共有的偏差。

## 13. 执行顺序

```text
用户明确授权 Round Z0
  -> 按 §21 构造 U42--U45 等权平均 actor，固定一次四图 CUDA + near400 诊断
  -> 若相对 U44 形成严格 Pareto 改善或达到最终目标：记录并收口，不继续训练方法搜索
  -> 否则关闭 §21，不改 band/权重；后续预注册链仍需单独授权
用户明确接受 §3.3「训练数据可由历史策略未来结局离线筛选」并授权 Gate A--D
  -> 按 §3.4 在任何 actor eval 前冻结 development/validation startpoint manifest
  -> Gate A：BC 与 U42--U45 仅在 development difficult panel 上 fresh CUDA replay
             （共识 cohort 见 §4.1；validation保持未读）
  -> Gate B：branch 0 no-op replay 校验 -> BC 接管测试 + 安全正对照
  -> 构造冻结 anchor sequence dataset（含 §9.2 的 0.5/0.5 分层与聚合顺序）
  -> 按 §7.3 重建 §24 的 per-regime 梯度分解（含分解误差硬校验）
  -> Gate C：LR 加权 + clip-aware 的固定 beta shadow update（全部 8 个 minibatch）
  -> 最小实现
  -> Gate D：beta=0 一个 formal update 逐位等价
  -> 用户明确授权「hindsight-selected counterfactual BC-branch regularization」
  -> 唯一一条 45-update Austin fresh training
  -> U42--U45 四图 CUDA 评估 + validation split，U44 主判读
  -> 更新 ANALYSIS/HANDOFF；删除 run.sh 中已完成命令
```

在对应授权前，不得构造或评估 §21 actor，也不得执行 Gate A--D。Gate A--D 获得授权后仍不得
提前启动 formal training；Gate B 的 branch 0 未通过前，其余分支结果一律不作数。

**当前执行进度（2026-08-08）：** Round Z0已按§21.4否决；Gate A按§4.3通过，Gate B按§5.5
完成。branch 0在56条上全部逐位复现，完整BC救回C层`10/19`且全部为overtake，但L恢复`0/9`、
controls损失`2/28`次overtake，故Gate B科学失败并关闭本方向。validation未运行branch；没有
anchor dataset、Gate C/D、formal训练或新actor。

## 14. 2026-08-08 方法审计追加：为什么整体成功概率有限

本节只追加方法判断，不改写 §0--§13 的预注册合同，也不构成执行授权。当前问题已经不再是
普通的 PPO 超参数选择，而是要让同一个 actor 在细粒度交互阶段中同时完成安全、超车和跨地图
泛化。现有证据支持以下判断：

1. **目标位于当前经验前沿之外。** U44 为 `62 collision / 1478 overtake`，ordinary 异线高速
   重加权 U30 为 `73 / 1516`，Production U30 headline 约为 `94 / 1508`。不同 actor 分别携带
   安全和超车能力，但没有集中到同一个 actor。粗 regime 事后拼接也只有约 `58 / 1527`，说明
   仅按 same-line/off-line 调整强度不足以达到 `<40 / >1500`。
2. **collision 信号稀疏且延迟。** 决定接触的动作通常发生在终局前 `0.5--1.5s`，标准 PPO
   只从整条 rollout 的 return/advantage 间接得到方向，不能直接观察"同一状态换另一段动作会
   怎样"。困难池、固定 collision penalty 或增加全局探索强度都主要增加失败曝光，没有提供
   反事实动作归因。**这一条同时有正面证据支持后续的动作条件方向**：`ANALYSIS.md` §20 的
   oracle 可达性与共享动作库表明残余碰撞在动作接口上可解，§28 的逐 episode hindsight 上界
   `25 / 1557` 表明目标行为已分散存在于同结构 actor 中。这与 §30--§32 连续否决的"状态分类"
   路线性质不同，是 §17 方向的既有先验。
3. **失败身份迁移，而不是固定难例收口。** 现有方法大量消除 BC collision，同时各自新造
   二十余次不同 collision。降低总数不等于学会稳定的状态条件动作映射。**§1.1 进一步表明这
   种迁移在同一条 run 的相邻 checkpoint 之间就已发生**：相对权重变化约 `1e-4` 就让约一半
   created collision 身份翻转，四点并集 `54` 而核心仅 `14`。因此相当一部分 created
   collision 是决策边界上的临界事件，不是稳定的策略缺陷；任何方法的单 checkpoint 改善都
   必须按区间判读。
4. **现有证据不支持简单的观测缺失解释。** §32 的 fold-local 维度匹配结果显示 U44 GRU
   hidden 对当前几何的线性解码能力在 9/9 目标上高于 actor 输入；§24 又没有建立跨 checkpoint
   稳定的 output-head PPO 梯度冲突。当前更像“hidden 含有部分信息，但 PPO 没有稳定学会在
   不同阶段选择正确动作”，不能用新增几何辅助目标或固定梯度投影直接解决。
5. **Austin-only 训练与四图验收之间存在不可消除的泛化风险。** 任何 teacher、anchor prefix、
   cost budget 或动作响应标签都只能来自 Austin；跨地图结果只能在方案冻结后检验，不能反向
   参与阈值、权重或 checkpoint 选择。

下表是审计者的**主观先验**，不是统计结论、准入阈值或实验结果，不得写入正式 verdict：

| 方法 | 若正式训练，降低部分 collision | 相对 U44 collision/overtake 双轴改善 | 达到 `<40 / >1500` |
|---|---:|---:|---:|
| §0--§13 hindsight-selected BC functional regularization | `50%--60%` | `25%--40%` | **≈0（见下）** |
| §16 Constrained PPO | `50%--65%` | `15%--25%` | `5%--10%` |
| §17 action-conditioned controllability auxiliary head | `40%--60%` | `25%--40%` | `10%--20%` |

当前臂那一格不是主观先验而是**算术结论**：§10.3 已经证明锚定范围限定在 off-line 时，
乐观 collision floor 为 `42`，达不到 `<40`。因此只有在 anchor 意外波及 same-line created
或 inherited-35 时该目标才可能达成，本预注册不把它列为合理预期。

这里"概率较低"不等于方向无价值。三种方法分别处理**行为遗忘、约束权衡、动作后果学习**，
但没有任何一个方法自动同时解决状态覆盖、长时 credit、policy extraction 和跨地图泛化。

## 15. 三种方法的关系与权威边界

| 方法 | 准确名称 | 新增训练信号 | 是否纯 RL / 纯 PPO | 当前状态 |
|---|---|---|---|---|
| 当前预注册臂 | `Single-stage PPO with hindsight-selected counterfactual BC-branch regularization` | Gate B成功BC反事实分支上的动作mean | 含 imitation-style functional regularization；不是纯 PPO | **只有本方法已形成 §0--§13 执行合同；正式训练未授权** |
| Pro 候选一 | `Lagrangian Constrained PPO with an independent collision-cost critic` | first-ego-collision cost、cost advantage、dual variable | 不含 imitation，属于约束式 RL；但 actor loss 不再只有 reward PPO surrogate | 仅研究提案；见 §16，尚未预注册或授权 |
| Pro 候选二 | `PPO with an action-conditioned controllability auxiliary head` | 同状态前缀下候选动作序列的 collision/progress 结果 | 含训练期环境监督 auxiliary loss；不是标准纯 PPO | 仅研究提案；见 §17，尚未预注册或授权 |

§16、§17 是为了保留替代方法的机制、风险和最低前置检查，**不是当前实验的 fallback 队列**。
Gate A--D 任一失败时，仍按 §11 停止；不得自动切换到 cost critic、action-value head 或本文件
后续衍生方案。若用户选择其中一项，必须先把它写成独立的单变量预注册，明确 control、数据、
loss、优化器、预算、panel、准入线和停止规则，再考虑实现。

## 16. Pro 候选一：带独立 collision-cost critic 的 Constrained PPO

### 16.1 核心假说

如果用户坚持不允许任何模仿 loss，最自然的新方向不是继续扫描 collision penalty，而是把
性能目标与安全约束分开：保留现有 reward critic，同时增加一个只在训练期存在的 cost critic。
最小 cost 定义为：

```text
cost_t = 1  if t 是本 episode 首次 ego collision
         0  otherwise
```

约束问题为：

```text
maximize   J_reward(theta)
subject to J_cost(theta) <= d
```

实际训练可写成 Lagrangian 形式；若采用“最小化 loss”的代码符号：

```text
actor_loss = ppo_reward_loss + lambda_cost * ppo_cost_loss
lambda_cost <- max(0, lambda_cost + dual_lr * (estimated_J_cost - cost_budget_d))
```

cost critic、cost GAE、cost surrogate、`lambda_cost` 和 dual update 全部只在训练期存在；最终
actor 仍可保持 361D 输入和 12-key checkpoint。该方法不使用 BC/U30/U44 动作 teacher，也不
需要部署期 shield。

### 16.2 为什么它可能有效

- 不再用一个固定 reward penalty 同时表达"尽量超车"和"碰撞不得超过预算"；
  **但要注意约束方向可能反了**：当前 reward 中一次性 `COLLISION_PENALTY = -2.0` 相对
  progress/relative 两项的量级，已经把碰撞定价得远高于单次超车增益；实测的失败模式是
  "拿超车换安全"，不是安全定价不足。若最终采用 Lagrangian，更贴合已观测现象的形式是
  **reward 主攻安全、约束放在 `overtake` 保有量**，而不是 reward 主攻 progress、约束放在
  collision。该重构必须在独立预注册中显式选择并说明理由；
- 当 collision 超预算时自动提高安全压力，低于预算时允许 reward objective 恢复超车；
- cost critic 理论上可以独立估计安全 return，避免 reward value 同时拟合两个尺度差异很大的
  目标；
- 仍是一条 Austin fresh RL 轨迹，actor 输入、动作接口和部署方式不变。

### 16.3 本项目中的主要风险

1. **没有新增动作信息。** 它比固定 penalty 更规范，但仍只告诉 actor“这条轨迹发生了
   collision”，不告诉 actor 在关键窗口应改变 steering、speed 还是二者组合。
2. **cost 极稀疏。** 每个 collision episode 只有一次正 cost，cost critic 与 cost advantage
   可能高方差或只记住场景身份。
3. **训练预算与验收预算不一致。** 当前 50/50 collision/ordinary role 是人为训练分布，
   `J_cost <= d` 在该分布上的含义不等于四图自然600面板上的 collision 数。`d` 不能从四图
   treatment结果反向调节。
4. **dual dynamics 是新的优化系统。** `lambda` 初值、dual LR、更新频率、投影和 cost
   normalization 都会影响稳定性；若逐项扫描就不再是一个干净实验。
5. **最容易出现新的安全—超车交易。** `lambda` 足够大时，actor 可能通过减速/跟车满足约束，
   而不是学到更精确的并排和超车后控制。

### 16.4 独立预注册前必须解决的问题

- **先解决与现有 reward 的重复计价**：`ppo/reward.py` 的 `COLLISION_PENALTY = -2.0` 是
  首次 ego collision 的一次性罚，与 §16.1 的 `cost_t` 是**同一个事件**。不先移除或重标定
  reward 中该项，cost critic 与 `lambda_cost` 调节的就是一个已被定价过的量，实验也不再是
  单变量。必须在预注册中明确二选一：reward 去掉 collision 项、或 cost 改用 reward 未覆盖
  的量；
- 固定 `gamma_cost`、terminal/timeout cost 语义、cost return 与 cost GAE 公式；
- 只用 Austin 训练侧数据定义 `cost_budget_d`，并解释它与 50/50 role 的关系；
- 固定 `lambda` 初值、dual LR、更新频率、上下界和记录字段，不做事后 dual sweep；
- 先在按 startpoint 分组的 Austin train-side 数据上证明 cost critic 不是常数预测器，并报告
  calibration、正例覆盖和 cost-advantage 有效样本数；
- 明确 control 使用哪条 fresh trajectory。不得把历史 U44 当作天然 control，除非当前代码、
  worker拓扑、scenario queue、RNG和完整训练轨迹的等价链已经闭合；
- 正式验收仍固定四图 CUDA、逐图 BC 下限、配对身份和 `<40 / >1500`，不能用训练 cost
  达预算替代 deterministic evaluation。

未完成上述预注册前，Constrained PPO 只是“更规范的安全权衡”候选，不能称为比当前方法更有
把握的解决方案。

## 17. Pro 候选二：Action-conditioned controllability auxiliary head

### 17.1 要解决的不是“未来会不会碰撞”，而是“什么动作有用”

该方向不再训练一个未来 collision 分类器，而是在同一 actor-visible hidden 上查询一段候选
动作序列的短期控制后果：

```text
Q_aux(h_t, a_t:t+H)
  -> (first_ego_collision_cost,
      progress_delta,
      final_overtake_or_follow,
      minimum_clearance_diagnostic)
```

**本节尚未定义 auxiliary loss 的形式。** §17.3 给出的是评估指标（ranking accuracy、
rescue rate、regret），不是训练目标；四个输出是各自回归、pairwise ranking、还是混合，
各分量权重如何，都必须在独立预注册中先固定。缺这一项时 §17 只是研究方向，不是可执行方案。

`minimum_clearance` 只作诊断，不得悄悄成为新的未注册 reward。训练数据只来自 Austin
train-side prefix；Hockenheim、MoscowRaceway、Nuerburgring及所有正式panel不得用于动作库、
horizon、标签、网络或阈值设计。auxiliary head 在训练结束后删除，最终 actor 可以保持原输入和
12-key checkpoint，但训练过程已包含环境监督 auxiliary loss，因此不是标准纯 PPO。

### 17.2 反事实动作数据的最小合同

每个 prefix 必须从**完全相同**的 simulator state、opponent planner state、actor observation
prefix 和 GRU hidden 开始，先通过 no-op branch 精确复现原轨迹，再运行一个预先固定的小型
动作序列库。动作族可包含：

- 原策略 mean action；
- 轻微持续减速 / 加速；
- 左 / 右小幅持续 steering；
- steering 与 speed 的固定组合；
- 干预结束后恢复原策略。

动作幅度、持续时间 `H`、恢复方式和候选数量必须在读取结果前固定；不得看到哪类动作有效后
扩库或改 horizon。每个失败 prefix 都要有同 raceline/speed 分层的安全正对照，以检测动作库
是否通过牺牲原本安全超车来制造表面安全收益。

### 17.3 更细粒度交互阶段的实现方式

不应先把状态硬编码成“接近/并排/超车后”，也不应继续用“未来是否碰撞”作为阶段标签。更
贴近当前缺口的定义是**动作响应阶段**：令每个状态的 response signature 为全部候选动作的
paired outcome 向量，再按哪类动作形成稳定改善得到软标签：

```text
brake-beneficial
lateral-left/right-beneficial
coordinated-steer-speed-beneficial
release-or-accelerate-beneficial
no-local-solution
```

训练/部署时的阶段判断只能读取 `h_t` 或 actor-visible history；真值几何、未来结局和分支结果
只用于 Austin 离线标签。主指标不是普通 MSE，而是按 startpoint 分组外推的：

- pairwise action-ranking accuracy；
- top-1 candidate 的真实闭环救援率；
- top-1 progress/collision regret；
- 安全正对照误伤率；
- 各动作族和各 startpoint/raceline 的覆盖，而不是池化平均。

### 17.4 预测 head 本身不足以改变策略

这是 Pro 方案必须在独立预注册中补清的关键边界。一个 action-conditioned head 即使预测准确，
也不保证 PPO actor 会按它选动作。至少需要明确下面二者中的哪一种，且不得混用：

1. **Representation-only auxiliary head**：只让预测 loss 反向进入 GRU，actor 仍完全由 PPO
   surrogate更新。这最接近 Pro 原提案，但可能出现“head预测正确，actor动作不变”。
2. **Q-filtered action preference**：只有真实反事实闭环证明某候选以固定 margin Pareto 优于
   原动作时，才让 actor 在该窗口接近候选动作序列。这更直接，但新增了第三个 actor objective，
   本质上是 simulator-selected imitation / policy regularization，必须另行授权，不能仍称为
   单纯的 controllability head。

不得让 actor 在连续动作空间中无限制最大化 learned `Q_aux`；这会利用模型误差。任何 actor
extraction 都应限制在固定局部动作库或其预注册邻域内，并用真实闭环分支而不是预测值决定
是否产生动作监督。

### 17.5 正式训练前的必要门

正式训练前至少必须依次证明：

1. **action existence**：足够多的训练侧失败存在同时改善 collision 与 progress/overtake 的
   候选动作，而不是只有更强制动能换取安全；
2. **safe-control specificity**：同一干预在匹配安全episode上的新造collision和overtake损失低；
3. **actor-visible rankability**：只用 hidden/history 的ranker在按startpoint分组的validation
   上稳定超过原动作与固定动作基线；
4. **exact replay**：no-op分支逐动作、outcome、碰撞时刻和terminal合同一致；
5. **coverage**：结果不由单一startpoint、raceline、speed或单一动作族贡献；
6. **冻结设计**：动作库、horizon、label、head、loss权重和policy-extraction方式在正式训练前
   一次固定，不按四图结果修改。

任一项失败只能否决对应命题：没有候选动作、hidden不能排名、或预测不能转化为策略；不得把
三种失败统一解释为“actor容量不足”。

## 18. 反事实动作信息能证明什么、不能证明什么

反事实动作信息能直接回答标准 PPO 数据回答不了的问题：**在完全相同的状态前缀上，局部换一
段动作是否真实改善后续结局。** 它因此是定位当前 credit/action-selection 缺口的重要证据，
但不是充分条件。完整链条为：

```text
候选库中存在可接受动作
  -> actor-visible hidden 能跨 startpoint 排名
  -> 训练目标能让 actor 实际采用该动作
  -> 新 student 访问的新状态仍被覆盖
  -> Austin 学到的映射能跨地图泛化
```

任一箭头都可能失败：

- **动作库覆盖失败**：真正有效动作不在候选集，或需要更长/更早的协调序列；
- **短视性**：短窗口 clearance/progress 改善，但恢复原策略后仍碰撞或丢失超车；
- **可辨识性失败**：事后确有优动作，但当前 hidden 无法提前区分；
- **policy extraction失败**：head能预测，shared actor仍不改变动作；
- **covariate shift**：student修复旧prefix后进入新状态并制造新collision；
- **不可兼得**：某些状态的候选动作全部落在安全—超车交易线上，不存在局部Pareto改善；
- **地图特化**：Austin startpoint上成立，跨地图失效。

所以“提供反事实动作信息”本身不能宣称解决当前问题。它最大的价值是把后续决策分流：

| 无训练结果 | 支持的下一步 | 不支持的推论 |
|---|---|---|
| BC接管稳定救援且正对照安全 | §0--§13 BC functional regularization仍有机制依据 | 不等于正式训练会跨地图成功 |
| BC失败，但固定动作库有跨startpoint Pareto改善且hidden可排名 | 可独立预注册action-conditioned或Q-filtered方法 | 不能现场改写当前anchor臂 |
| 候选动作存在，但hidden无法排名 | 重新审查actor-visible history/表征 | 不能只加更强collision penalty |
| hidden可排名、head也准确，但actor不采用 | shared action mapping/优化通路是候选瓶颈 | 不能把预测准确率写成控制收益 |
| 只有更早prefix存在优动作 | 长时credit/训练起点是候选瓶颈 | 不能继续缩短干预窗口 |
| 固定局部库没有可接受动作 | 动作horizon、动作族或真实Pareto边界需重审 | 不能直接判定全局动作接口不可解 |

## 19. 由上述审计派生的其他研究想法（不属于 Pro 的两项提案）

本节只保存思路，当前未授权、未预注册，不进入 §13 执行顺序。

### 19.1 绝对纯 PPO 候选：prefix-reset interaction curriculum

如果反事实审计显示关键状态存在局部可学习动作，但标准8秒episode的credit过长，可以在Austin
训练侧保存交互前的完整simulator snapshot，并通过prefix重放恢复actor GRU hidden，让一部分
PPO episode直接从这些状态开始。它保持标准PPO actor loss与现有reward，不使用teacher、cost
critic或动作监督；短窗口结束必须按 truncation 正确 bootstrap，不能伪装成真正terminal。

该方法与普通hard pool不同：hard pool仍从场景起点开始，只提高失败场景频率；prefix reset
直接提高"仍有操控权的关键窗口"的transition密度。最大风险是simulator/opponent planner
state与hidden不能精确恢复，以及短窗口训练分布偏离完整episode。实现前必须先通过no-op完整
复现和prefix-vs-full-episode return/GAE合同检查。

**它还引入两个必须一次固定的隐含变量**，否则不是单变量实验：(a) 每个 rollout 中从 prefix
起始的 episode 占比；(b) prefix 的选择集合与其筛选口径（若用历史失败筛选，同样属于 §3.3
的离线 hindsight，必须按该节声明）。此外 prefix-reset 会改变训练状态分布，从而改变现有
`50/50 collision/ordinary role` 与 479 collision cache 的实际含义，这两项的语义必须在预
注册中重新定义，不能沿用当前描述。

### 19.2 更直接但非纯 PPO：counterfactual action-preference regularization

若固定动作库存在跨startpoint稳定的Pareto改善，可以直接把**真实闭环验证过**的候选动作序列
作为局部Q-filtered target，而不是先依赖一个可被actor利用误差的连续Q模型。它比BC anchor
覆盖更广、teacher来自simulator counterfactual而不是某个历史actor，但同样使用hindsight与
imitation-style loss，必须独立授权；动作库、margin、窗口和权重均是新变量。

### 19.3 若shared mapping被直接证明不足：interaction-phase residual / mixture-of-experts

只有在“hidden能稳定排名动作，但当前shared output mapping无法吸收”被直接证明后，才考虑
由hidden产生soft gate，在制动、横向通过、联合控制、release等残差expert之间连续混合。粗
same-line/off-line gate已经不足，expert应按§17.3的动作响应阶段定义。该方向会改变actor key和
部署结构；除非用户重新授权放弃12-key兼容，否则不合法。不得仅凭现有模型能力分散就直接训练
MoE，也不得用二阶段distillation绕过当前禁止的repair流程。

## 20. 研究排序与停止边界

当前排序必须区分“近期可执行性”和“理论突破潜力”：

| 排序口径 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 近期准备程度 | 当前 BC functional regularization | Constrained PPO | §19.1 prefix-reset curriculum | action-conditioned auxiliary head |
| 突破当前安全—超车前沿的理论潜力 | action-conditioned / counterfactual preference | §19.1 prefix-reset curriculum | 当前 BC functional regularization | Constrained PPO |
| 实现与归因风险从低到高 | 当前 BC functional regularization | Constrained PPO | §19.1 prefix-reset curriculum | action-conditioned / counterfactual方法 |
| 是否仍是纯 PPO | 否 | 否 | **是** | 否 |

**§19.1 此前被排序表整体遗漏，本次补入。** 它是四个候选中唯一保持纯 PPO actor loss 的方案，
也是唯一直接针对 §14.2 所诊断的"collision 信号稀疏且延迟"的方案；它不需要 teacher、cost
critic 或动作监督，也不需要新的授权类别。它仍未预注册，排序不产生执行顺序。

上述排序不产生自动执行顺序。Round Z0已关闭，Gate A通过、Gate B科学失败，因此当前BC执行链
已经停止，Gate C/D不再准入。Pro 的两项方法和 §19 的衍生想法若要继续，必须由用户明确改选并完成各自独立
预注册。任何方法都不得读取测试地图训练、改变四图正式panel、引入部署期future信息或把诊断
成功写成actor性能。

## 21. 由 §1.1 派生的零训练候选：U42--U45 checkpoint 权重平均

本节是**独立候选**，不是 §0--§13 的一部分，也不是 Gate 失败后的 fallback。两份审查合并后
按 §0.1 将其预注册为 Round Z0；现已一次性构造、评估并按§21.4关闭，未授权任何新训练。

### 21.1 动机

§1.1 给出两条并列事实：四个相邻 checkpoint 的相对权重距离只有 `1e-4` 量级
（`||theta_U44|| = 162.75`，`||U42-U45|| = 0.0347`），而它们的 created collision 身份
Jaccard 只有 `0.39--0.58`、四点并集 `54` 而核心仅 `14`。也就是说，**极小的权重扰动就能翻转
约一半的失败身份**。

由此产生一个可检验的假说：这些临界失败大部分是 checkpoint 特异的，权重空间上的共识可能
只保留"四点共有"的失败模式。作为动机（**不是预测**）：一个只犯四点共有失败的假想模型，
四图 collision 为 `core created 14 + core inherited 22 = 36`，低于 `<40`。

权重平均不等于行为交集，上述 `36` 不能写成对平均模型的预期。

### 21.2 可行性（2026-08-08 实测）

U42--U45 的四个 `actor.pth` 的 12 个 key 与全部 shape 完全一致；等权元素级平均后可以
`strict=True` 加载进 `End2Race(hidden_scale=4)`，仍是严格 12-key。平均模型与 U44 的相对
距离为 `6.2e-5`。同一训练轨迹与极小权重距离支持“处于同一局部区域”的假说，说明没有明显的
跨 basin 数值警报；**这仍不证明参数线性插值等于行为插值**，所以必须靠一次固定评估判决。

### 21.3 若要执行，必须预注册的内容

- band 固定为 U42--U45、权重固定为等权，**在评估前写死**，不得看结果后换 band 或权重；
- 构造路径固定为 `post-trained/zero_train_ctv2_u42_u45_equal_average/actor.pth`；保存四个源
  checkpoint 的相对路径与SHA-256。所有浮点tensor先转float64等权求和/4，再cast回原dtype；
  若出现非浮点state key，四源必须逐值相等后复制，否则fail closed；
- fresh `End2Race(hidden_scale=4)` strict-load、12-key、finite检查通过后，给eval使用唯一alias
  `CTv2_U42_U45_EQUAL_AVG`，避免同名checkpoint trace覆盖；
- 只评一次四图各 600 episode（CUDA、ego scope、确定性 mean action、保存 traces）；
- 报告相对 BC 与相对 U44 的逐图 collision/overtake、removed/created、lost/gained 与配对
  exact test，以及 created 集合与 §1.1 四点核心 `14` 的重合度——**后者是本假说的直接检验**；
- 它产出一个新的可部署 actor，因此只有在用户按 §0.1 明确授权后才能一次性构造与评估；即使
  评估成功，也不自动修改 production alias；
- 它是对 checkpoint band 的事后聚合操作，与 §11 rule 3 禁止的"从 U42--U45 事后挑单点"
  性质不同，但同样必须一次固定、一次评估，失败即记录并关闭，不得改成三点平均或加权平均
  重试。

### 21.4 一次性判决线

平均 actor 只允许一次正式评估。它必须逐图满足 canonical BC 下限，并相对历史 U44 满足以下
严格 Pareto 条件之一：

```text
四图 collision < 62 且 overtake >= 1478
或
四图 collision <= 62 且 overtake > 1478
```

同时报告 U44→average 的 removed/created、lost/gained、exact paired test、created 集合与四点
core-14 的重合，以及 near400 `collision/overtake`。只改善1个总数但配对身份和near400同时
恶化，不得称为稳定突破。若不满足上述条件，§21关闭；不得改成U43--U45、三点平均、加权平均
或根据四图结果选择权重。

**执行结果（2026-08-08，已关闭）：** 四图逐图为Austin `18/342`、Hockenheim `19/341`、
MoscowRaceway `16/386`、Nuerburgring `14/396`，合计`67/1465`。Hockenheim overtake
`341 < BC 343`，且相对U44四图collision `62→67`（removed/created `13/18`, `p=0.473`）、
overtake `1478→1465`（lost/gained `24/11`, `p=0.0410`），故同时未通过BC逐图线与严格Pareto线。
near400为`32/285`。平均actor相对BC created 32次，core-14中仍保留13次（Jaccard `0.394`），
没有实现“只保留共识失败”。不做任何三点、非等权、EMA/SWA或事后权重重试。

### 21.5 边界

- 权重平均**不是训练方法**，它不解释任何机制，也不产生新的行为能力；即使成功，也只说明
  相邻 checkpoint 的临界失败可以被共识抵消一部分。
- 若平均模型性能崩塌，那本身是有价值的负结果：说明相邻 checkpoint 之间的行为差异不是
  小扰动，§1.1 的"临界事件"解释需要修正。
- 它不改变 §0--§13 的任何 gate，也不改变 production 决策；Production 仍为 U30。

## 22. 合并审查结论：最后1--2轮优先级、未准入方法与收口条件

### 22.1 最终优先级

在当前约束下，最后阶段只保留两类可供用户分别授权的预算动作；以下排序不构成执行授权：

1. **Round Z0：U42--U45等权平均（已完成并关闭）**。固定结果`67/1465`，未通过Hockenheim
   BC下限或U44严格Pareto线，不得更改band/权重重试。
2. **Round Z1：本文件Gate A--D（Gate B失败，已关闭）。** Gate A通过，但Gate B的L与control
   门失败，因此不生成anchor dataset、不进入Gate C/D，也不申请45-update BC
   functional-regularization正式训练。

若Z0失败且Gate A/B表明BC动作不是稳定有效teacher，本BC方向立即关闭。此时仅允许把已经通过
branch-0合同的prefix branching engine复用于一个**独立预注册**的
`simulator-return-filtered first-action preference`候选；不得在本文件中现场把BC target换成
动作库target。该候选是最后一个具有更高理论射程的新训练方向，因为它可覆盖inherited和
same-line失败，但它仍需固定动作库、first-action pairwise loss、beta与独立准入线。

### 22.2 抗性审查后不新增的方案

以下想法经过约束与现有证据审查后不加入执行队列：

- **U44 + RW30跨臂权重平均/插值**：依赖两条独立PPO训练轨迹，是否同一basin未验证，并违反
  “完整方法只依赖一条RL轨迹”的严格口径；若看四图后调插值系数还会形成测试集选择。
- **重新跑EMA/SWA、SAM或Lookahead**：§21已经给出同轨迹checkpoint平均的最低成本检验；
  重新训练会额外引入decay/optimizer超参，却没有新增动作信息。
- **CVaR、worst-group PPO或继续调场景权重**：仍只改变失败权重，不告诉actor关键窗口应采用
  哪段steering/speed动作；现有hard-pool与reweight结果先验较低。
- **Residual/MoE/runtime selector**：改变actor key、部署结构或引入第二控制路径，不满足当前
  361D/12-key兼容要求。
- **跨地图训练或用三张测试图选阈值**：直接违反Austin-only训练与泛化验证边界。

因此没有遗漏一个同时“低复杂度、高证据、满足全部约束”的第三条训练方案。文档中保留的
counterfactual preference与prefix-replay已是最后两个机制类别；前者需要独立预注册，后者只有在
反事实审计证明“有优动作但8秒credit/访问密度不足”时才有准入依据。

### 22.3 最终收口条件

满足任一项即结束本阶段，不再新增方法：

1. §21形成可接受的新Pareto actor；
2. BC Gate A/B/C/D任一科学gate失败，且独立counterfactual action-existence gate也失败；
3. 唯一正式训练完成但未形成相对U44的稳定双轴改善；
4. branch engine无法在当前simulator/planner/RNN合同下精确复现，导致所有hindsight方法失去
   因果基础。

收口时保留U44、RW30、Production U30三点产品前沿，并明确写成“当前约束和方法族的经验收口，
不是动作空间、End2Race结构或PPO的理论上限”。

## 23. Round Z2：独立反事实动作存在性与可排序性Gate（2026-08-08，运行前冻结）

本节不是失败BC target的替换分支。用户在Gate B关闭后明确授权继续完成剩余候选；依据§22.1，
先复用已经通过56/56精确重放的branch engine做零训练筛选，只有本节准入后才允许为获支持的
唯一训练候选另写预注册。Constrained PPO没有被Gate B逻辑否决，但不与本Gate并行训练；
Residual/MoE因仍要求361D输入和12-key actor兼容而直接不准入。

### 23.1 冻结cohort与数据边界

只使用Gate A已经冻结的Austin development startpoints及既有BC/U44结果；不读取Gate A
validation或三张测试地图。先要求U44事件前至少还有150个action step，再全取：

| stratum | 定义 | 固定数 |
|---|---|---:|
| inherited collision | BC与U44均ego collision | 109 |
| created collision | U44 collision、BC非collision | 46 |
| lost-overtake | BC overtake、U44 follow | 13 |
| inherited-follow诊断 | BC与U44均follow | 63 |
| safe control | BC与U44均overtake | 225 |

共456条；短prefix剔除112条。fold只能按
`SHA256("counterfactual-action-fold-v1|" + ego_idx) mod 5`分组，禁止同一ego startpoint跨fold。
inherited-follow只报告机制覆盖，不参与三条主准入分母。

### 23.2 固定prefix与动作库

collision事件为U44首次ego collision；其余为U44全局最小OBB clearance。每条只运行两个互不
重叠的最多50-step residual block；若环境未因collision提前terminal，结束后恢复U44：

- early：`[event-150, event-100)`；
- late：`[event-100, event-50)`。

动作是相对每一步当前U44 deterministic mean的固定residual：steering `-0.04/-0.02/+0.02/
+0.04 rad`，speed `-1.0/-0.5/+0.5/+1.0 m/s`，以及`steering +/-0.02`与`speed +/-0.5`
的四个组合，共12个候选；另以no-op U44作基线。steering仍clip到`[-0.52,0.52]`。不得看结果
后增加幅度、动作族、prefix或持续时间。

456条no-op必须先逐动作、opponent action、pose/speed、双360D LiDAR、collision marker、长度、
outcome和terminal精确复现既有U44；任一失败则不运行候选。候选共`456*2*12=10,944`次branch，
只保存合同所需compact numeric trace；20个persistent CUDA worker各只加载一次U44。

### 23.3 Action-existence准入线

early与late分别判决，不允许取两者逐scenario并集后通过。对每个target prefix，只有候选最终为
无ego collision的overtake才叫Pareto rescue；若多个候选成功，按residual归一化平方范数最小、
final relative progress最大、动作名排序确定唯一oracle label。每个prefix必须同时满足：

1. inherited collision rescue至少`22/109`；
2. created collision rescue至少`12/46`；
3. lost-overtake恢复至少`7/13`；
4. 两类collision救回合计覆盖至少10个ego startpoint和raceline0/1/2；
5. oracle labels至少覆盖两个动作family（steering、speed、coordinated中的两个）。

### 23.4 Frozen-hidden action-conditioned rankability准入线

仅对通过§23.3的prefix训练离线head，不更新actor。输入是分支起点的1680D U44 hidden；候选动作
以`steering/0.02, speed/0.5`编码。固定scorer为`hidden Linear(1680,128)+ReLU`，与动作拼接后
`Linear(130,64)+ReLU+Linear(64,1)`；5-fold按startpoint外推，fold-local标准化，Adam
`lr=1e-3, weight_decay=1e-4`、batch 64、100 epochs，seed为`4200+fold`，不调参。

每个fold的固定动作baseline只用其train folds选择，score固定为
`target overtake successes - 5*safe-control harms`。out-of-fold head必须同时满足：

1. inherited/created/lost-overtake分别至少`11/109、7/46、4/13`；
2. 225条safe controls中新造collision至多4、overtake loss至多11；
3. 三个正式target stratum的成功总数至少比grouped fixed-action baseline多9。

### 23.5 冻结分流与停止规则

- late existence与rankability均通过：只准入独立的
  `simulator-return-filtered first-action preference`训练预注册；仍不等于actor已改善；
- late失败、early两门通过：不训练preference，先做prefix snapshot精确恢复与fresh GAE合同审计，
  只有该审计通过才可预注册prefix-reset PPO；
- 两个prefix的existence均失败：关闭本固定局部动作库的action-conditioned/preference形式；
  不把有限动作库的失败写成连续动作空间不存在解；
- existence通过但rankability失败：关闭当前frozen-hidden scorer与first-action extraction；不得把
  oracle上界写成可部署收益；
- Constrained PPO始终标为“本Gate未检验”。只有本Gate不能准入动作信息路线后，才单独决定
  是否值得支付cost-critic预检预算，不得把它写成自动fallback。

### 23.6 执行结果（2026-08-08，未准入）

456/456 no-op在动作、对手动作、pose/speed、双360D LiDAR、collision marker和terminal上的
全局最大误差均为0。early/late各12个候选共10,944条，连同no-op共11,400条compact trace、
8,185,913行全部通过key、finite、对齐、terminal、窗口、residual和executed-action合同。

| 层 | early oracle | late oracle | early OOF head | late OOF head |
|---|---:|---:|---:|---:|
| inherited collision→overtake | `93/109` | `96/109` | `19/109` | `34/109` |
| created collision→overtake | `44/46` | `45/46` | `11/46` | `17/46` |
| lost-overtake→overtake | `10/13` | `7/13` | **`1/13`** | **`0/13`** |
| safe controls新collision/loss | oracle不介入 | oracle不介入 | **`15/17`** | `3/5` |
| 三主层target success | oracle `147` | oracle `148` | `31` | `51` |
| grouped fixed baseline | `0`（noop） | `79` | head多31 | **head少28** |

两个prefix的existence六门全部通过，覆盖48个collision startpoint、三条raceline和全部三类动作
family；局部动作不存在的解释被否定。两个rankability Gate均失败：early虽然比noop多31次target
success，但lost只恢复1条且controls误伤超限；late对controls合格，但lost恢复0条，且51次target
success低于fold内选择的固定`steer +0.02 rad / speed +0.5 m/s`的79次。

相对no-op，early/late head的collision removed/created分别为`41/19`、`56/7`，overtake
lost/gained为`17/36`、`5/54`；这些是在future-event定位的训练侧prefix上的诊断改善，不是可部署
actor成绩。结果支持“动作可达、选择/触发不可泛化”，不支持“再提供反事实动作就会自动解决”。

按§23.5关闭tested fixed-library action-conditioned与first-action preference；prefix-reset没有
取得early-only前提，不进入snapshot/GAE审计；Residual/MoE仍由兼容边界关闭。Constrained PPO
未被本Gate检验，不能据此严格关闭。

## 24. Round Z3：collision-only BC anchoring 独立 validation Gate

本节于任何 validation actor 评估和 branch 运行前冻结。冻结时
`eval_results/bc_collision_only_anchor_validation/`不存在，既有Gate A只有development结果；
`post-trained/panels/bc_safe_anchor_v1/validation_scenarios.json`的150条、19个ego startpoint
没有被BC、U42--U45或任何branch打开。

这是一个**新的窄变体**，不改写§5.5对原双分层方法的失败判决。其准确命题是：

> 只在稳定的U44-created collision层使用完整canonical BC teacher，不构造
> lost-overtake anchor；先在完全独立的Austin validation startpoint上检验teacher的碰撞
> 救援和progress不误伤是否同时存在。

它仍然是`PPO + selective BC functional regularization`，不是纯PPO。Round Z3只是训练前
必要条件Gate；通过也不授权生成anchor dataset、修改PPO或启动formal training。

### 24.1 固定数据与cohort

1. 固定panel为已冻结的150条validation ScenarioSpec，与development的startpoint零交集；
2. 以与Gate A完全相同的CUDA、ego collision scope、确定性mean action和8秒episode，
   一次性评估canonical BC与U42--U45，保存全部数值trace；
3. `BC-safe overtake`仍在查看student结果前定义为：BC无ego collision、最终
   `overtake`且opponent为`raceline0/2`；
4. 主cohort `C_val`要求U44最终为`ego-opp`或`ego-wall`，且U42--U45四个
   checkpoint中至少3个不是`overtake`。这保留原Gate A的稳定回归定义，不用
   U44单点碰撞补样；
5. 每条`C_val`的干预窗口仍为U44首次ego collision前150个action step；不足50
   个可执行step者在任何branch前剔除；
6. matched controls从同一validation panel内BC/U44共同安全overtake中选取，固定为
   同opponent raceline、同speed scale，再按循环ego waypoint距离和scenario-key SHA-256
   破平，无放回且与`C_val`等数。control窗口仍为U44全局最小OBB clearance
   前100到后50个step。

### 24.2 样本充足性与结果边界

在运行branch前，可执行`C_val`必须同时满足：

1. 至少4条；
2. 覆盖至少3个ego startpoint；
3. 来至`raceline0`和`raceline2`的source均非空。

任一条不满足时Round Z3标记为**inconclusive / validation样本不足**，不运行干预
branch，不允许换成U44单点cohort、从development补样或放宽稳定性条件。这不是
collision-only方法的科学失败；需要新的独立panel才能重开。

### 24.3 唯一正式branch与准入线

只运行`branch0`和`full_bc`，不重复steering-only/speed-only诊断；该组件问题已在
development Gate B回答。

1. `branch0`必须在全部`C_val + controls`上逐动作、opponent action、两车
   pose/speed、双360D LiDAR、outcome、collision marker、长度和terminal精确复现保存
   U44；失败时只否实实现合同，不运行`full_bc`，不判决teacher；
2. `full_bc`在冻结窗口内同时替换steering和speed，窗口后恢复U44；
3. collision rescue数至少为`ceil(0.50 * |C_val|)`；
4. 被救回的collision中，最终`overtake`至少为`ceil(0.80 * rescue_count)`；
5. 救回结果覆盖至少2个ego startpoint；对source数至少2条的每个raceline，至少
   救回1条；
6. controls新造ego collision和overtake损失分别不超过`floor(0.05 * N_control)`。

六条全部通过才写`mechanism_replication_pass=true`。通过只说明“collision-only BC teacher
在未见startpoint上仍具有可选择的局部救援信号”，不证明regularization训练后会改善actor。

在样本充足、branch0精确、产物合同完整的前提下，任一teacher/coverage/control条件失败，
则严格关闭这个`canonical BC × stable collision cohort × fixed 1.5s window × collision-only`
实例。不得看validation结果后修改门、换窗口或只保留成功的raceline。

### 24.4 通过后仍需的证据

Round Z3通过后只允许起草一份独立formal-training预注册。该训练必须：

- 从canonical BC fresh-start，只在Austin训练；
- 只使用development Gate B中真实救成overtake的C-anchor，validation分支不进训练；
- 没有L-anchor、第二treatment或二阶段repair；
- 先用shadow-gradient固定唯一`beta_anchor`，不扫权重；
- 最终由四图各600条CUDA面板、逐图BC下限、配对身份变化和预注册的
  collision/overtake双目标判决，诊断cohort不具有验收权。

### 24.5 执行结果（2026-08-08，inconclusive）

BC与U42--U45已在冻结validation上完成各150条CUDA评估，共750条result/trace全部
通过panel key、scenario identity、finite、数组对齐、collision marker和terminal合同。各actor
`ego collision / overtake / follow`为：BC `67/60/23`、U42 `38/67/45`、U43 `42/67/41`、
U44 `49/61/40`、U45 `43/64/43`。

稳定`C_val`有7条，覆盖6个从未用于development的ego startpoint，`raceline0/2=6/1`；
速度`0.60/0.70/0.75/0.85=2/1/1/3`。§24.2的三条cohort充足性门全部通过。

但在任何branch前冻结matched controls时，同opponent raceline、同speed scale、无放回
的对照池不足：`raceline0 / speed 0.60`有2条source但没有合格control，
`raceline0 / speed 0.85`有3条source但只有2条合格control。这是分层数量缺口，不是贪心
匹配顺序造成；任何无放回精确匹配都无解。

因此Round Z3按预注册停在control-plan阶段：**没有运行branch0、full-BC或任何actor
训练**。判决是`inconclusive / matched-control support insufficient`，不是teacher通过，
也不是collision-only方法失败。不得事后放宽speed/raceline匹配、改成有放回control、
从development补control或直接训练。只有新的、在任何actor outcome前冻结且具有充足
分层control support的独立Austin panel，才能重开这个validation Gate。

## 25. Round Z4：representation-changing action-response Gate

### 25.1 要回答的问题与证据边界

Round Z2只证明了**冻结的**U44 hidden配合已测线性head不能稳定选择局部动作，不能把它
外推为“改变表征也无效”。Round Z4首先检验一个更窄、成本更低的必要条件：在不改变
actor、不训练PPO且不访问未来信息的条件下，把动作干预前最近50步actor可见历史编码进
一个可训练GRU，能否在未见ego startpoint上比冻结hidden selector和状态无关固定动作
更好地预测动作后果并选择动作。

本Gate只检验：

`late 50-step actor-visible history encoder × frozen 13-action library × 3-class action outcome`

这一实例。失败不否决所有representation-only auxiliary learning；通过也不证明把该辅助
目标接入student GRU后PPO会改善，只允许进入一次真正的shared-GRU shadow-gradient Gate。

### 25.2 Z4-A development折外协议

数据冻结为Round Z2已经完成的456个development task及其`branch0 + late 12-action`
真实闭环结果，不重跑、不删样本、不改变动作库。每个task使用late分支点之前并包含当前
时刻的最近50步：`[late_start-49, late_start]`。

输入严格限制为actor部署时可见量：每步360D ego LiDAR与上一时刻ego speed。LiDAR使用
冻结U44的标量sigmoid变换，speed使用冻结U44的`speed_mlp`，拼成每步420D序列。输入中
禁止ego/opponent pose、opponent speed、碰撞/超车标签、距未来事件的距离、stratum、
branch结果或scenario identity。冻结U44 late hidden只作为共同的基线输入。

动作顺序由Round Z2 plan冻结：`noop`在前，随后原12个残差动作；动作编码为
`(steering_delta / 0.02, speed_delta / 0.5)`。每个scenario-action的监督标签是该动作分支
真实终局的三分类`collision / follow / overtake`，不是事后选出的唯一top-1动作。

按既有规则
`SHA256("counterfactual-action-fold-v1|ego_idx") mod 5`
分5折；同一ego startpoint的所有task只能在同一折。每折只用其余4折训练，测试折不参与
权重、类别权重、固定动作选择或任何超参数决定。

每折分别用训练折计算hidden逐维均值/标准差，以及420D历史特征在训练scenario和50个
time step上的逐维均值/标准差；标准差小于`1e-6`的维度固定为1。训练折与测试折都只用
该训练折统计量标准化，不允许使用测试折统计量。

共同冻结hidden control：`1680 -> Linear(192) -> ReLU`，拼接2D动作后经
`Linear(194,64) -> ReLU -> Linear(64,3)`。

representation treatment：冻结hidden经`Linear(1680,128) -> ReLU`；50x420历史经
`Linear(420,64) -> ReLU -> GRU(64,64)`；两者拼成192D后使用与control相同容量的
动作head。两者都固定为Adam、`lr=1e-3`、`weight_decay=1e-4`、100 epochs、scenario
batch 64；fold seed为`5200 + fold`，shuffle seed为`5300 + fold`。类别权重只由训练折
三类标签计算为`N / (3 * N_class)`。不做超参扫描、早停或看测试折选epoch。

测试折每个动作的固定选择分数为：

`P(overtake) - 5 * P(collision)`

取最大值，精确平局按冻结动作顺序选择。状态无关fixed baseline在每折训练数据上选择
`全部非control场景的overtake数 - 5 * safe-control harm数`最高的单一动作，平局同样按
动作顺序；再应用到该测试折。

### 25.3 Z4-A准入线与停止规则

先验证456个task、13个动作、hidden与50步输入一一对应，所有数组finite，fold无ego
泄漏，输入张量不含被禁止字段。合同失败只否实现，修复后原样重跑。

treatment必须同时满足：

1. inherited collision、created collision、lost overtake分别至少救回`11/109`、`7/46`、
   `4/13`；
2. 225条safe controls中新造ego collision不超过4，overtake损失不超过11；
3. 三个目标层总成功数至少比折内fixed baseline高9；
4. 三个目标层总成功数至少比matched frozen-hidden control高9。

四条全部通过才允许Z4-B。任一行为门失败，严格关闭本节定义的具体实例，不运行独立
validation分支、不接入actor、不训练PPO，也不得事后更换history长度、网络、类别目标、
动作分数或阈值。

### 25.4 Z4-B独立validation协议（仅Z4-A通过时执行）

若Z4-A通过，固定相同架构和训练参数，在全部development task上训练一次并冻结。
然后才允许在Round Z3的150条validation actor panel上按**同一套、在任何candidate
outcome前冻结的cohort定义**生成late prefix的12-action分支并评估。validation不得用于
重训、校准或选epoch。

Z4-B必须另行报告validation cohort充足性、branch0复现、动作存在性、固定模型行为门与
control harm。即使Z4-B通过，也只说明action-response监督具有跨startpoint机制证据；
下一步仍是共享student GRU上的一次shadow-gradient Gate，而不是直接启动正式PPO。

### 25.5 执行结果（2026-08-08，Z4-A fail）

456个development task、13个动作形成5,928个真实action-outcome标签；50步历史共22,800行，
hidden为`456x1680`、历史为`456x50x420`。全部输入finite，70个ego startpoint严格按冻结hash
分入5折且折间零重叠；代码只从trace读取`ego_lidar_360`和前一时刻ego measured speed，未把
pose、opponent状态、stratum或branch结果送入模型。五折均按训练折统计量、类别权重和固定
seed完成100 epochs，无空类别或不完整预测。

折外结果如下，三列依次为
`inherited collision / created collision / lost-overtake`恢复为overtake：

| selector | 三目标层 | target total | safe-control新collision / overtake loss |
|---|---:|---:|---:|
| grouped fixed action | `45 / 33 / 1` | `79` | `5 / 5` |
| frozen-hidden outcome control | `68 / 33 / 3` | `104` | `13 / 20` |
| 50-step history treatment | **`68 / 32 / 2`** | **`102`** | **`13 / 21`** |

history treatment的5,928标签三分类准确率为`65.92%`，略高于frozen-hidden control的
`64.49%`；但动作选择才是本Gate的必要条件。treatment相对control改变162条动作、52条终局，
只产生19条treatment-only overtake，同时丢掉20条control-only overtake，target净值为
`102 < 104`。它虽超过fixed baseline 23条并通过两个collision层最低线，却同时失败：

- lost-overtake只有`2/13 < 4/13`；
- safe controls新collision为`13 > 4`；
- safe controls overtake loss为`21 > 11`；
- 相对frozen-hidden control的target margin为`-2 < +9`。

因此Z4-A按预注册判为`fail_close_tested_representation_instance`。不运行Z4-B validation
candidate branches，不训练full-development模型，不接入actor，也不启动PPO。

严格证据边界：本结果否决的是本节固定的`late 50-step actor-visible history encoder ×
fixed 12-action library × class-weighted 3-class outcome target × fixed probability score`实例。
它证明“给该head增加这段历史”没有形成增量动作选择价值；不能据此否决会直接改变student
GRU、使用不同因果监督或不依赖future-event窗口的所有representation learning。不过这些
变体必须提出新的、可独立证伪的必要条件，不能通过重调本Gate的history长度、网络、loss或
动作分数来重开已失败实例。

## 26. Round Z5：harm-matched frozen-hidden operating-point Gate

### 26.1 修正的问题与证据边界

Round Z2的late top-1 preference head为target `51`、safe-control harm `3 collision / 5
overtake loss`；Round Z4-A的frozen-hidden outcome head为`104`与`13 / 20`。前者比fixed
action的79低，后者更高，但两者处于完全不同的干预率和control harm操作点，不能由`51 < 79`
或`104 > 79`单独判决frozen hidden是否具有可用的状态条件动作信息。

Round Z5只检验Z4-A中已经定义的frozen-hidden三分类outcome extraction，在与grouped fixed
baseline相同的control harm预算下能否仍超过其target。它不重跑模拟，不改变456条task、late
窗口、13动作、真实branch outcome、outer startpoint folds、模型结构、loss或训练epoch。
50步history treatment只作同协议次要对照，不承担frozen-hidden方法类的主判决。

### 26.2 Nested grouped-CV合同

外层继续固定为：

`SHA256("counterfactual-action-fold-v1|ego_idx") mod 5`

每个外层test fold只在最终应用一次。该fold的`lambda / tau`、标准化、类别权重和模型参数都
只能来自其余4个outer-train folds。

在每个outer-train集合内部，再按：

`SHA256("action-response-inner-v1|ego_idx") mod 4`

生成4个startpoint-disjoint inner folds。每个inner model的hidden/history标准化、类别权重与
训练均只使用对应inner-train；inner-validation predictions拼成outer-train的完整inner-OOF
概率。inner folds任一为空、startpoint泄漏或预测不完整都只判实现失败。

模型与Z4-A完全相同。固定训练预算仍是Adam、`lr=1e-3`、`weight_decay=1e-4`、100 epochs、
scenario batch 64。inner seed固定为`6200 + 10*outer_fold + inner_fold`，shuffle seed为
`6300 + 10*outer_fold + inner_fold`；最终outer model seed为`6400 + outer_fold`，shuffle
seed为`6500 + outer_fold`。不做early stopping或epoch选择。

### 26.3 操作点与inner选择

预先固定：

`lambda_grid = [0, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 32]`

对每个`lambda`：

```text
score(a) = P(overtake | h,a) - lambda * P(collision | h,a)
best_nonnoop = argmax over the 12 non-noop actions
margin = score(best_nonnoop) - score(noop)
intervene iff margin >= tau; otherwise choose noop
```

动作平局按冻结动作顺序。`tau`候选是该outer-train的inner-OOF margin全部有限唯一值，另加
`+infinity`作为全noop；因此完整枚举每个可实现干预集合，不靠事后加密网格。

control预算固定为grouped fixed baseline在全OOF上的两项实测率：

```text
new collision rate <= 5 / 225
overtake loss rate <= 5 / 225
```

inner候选用整数交叉乘法判断率，不作浮点四舍五入。可行候选中依次选择：三主层target
success最多、control新collision最少、control overtake loss最少、干预scenario最少、
lambda较大、tau较大；最后按lambda_grid顺序确定。全noop保证至少一个可行候选。

选出的`lambda/tau`冻结后，在全部outer-train上重新训练一次同结构模型，只对outer-test
预测并选择动作。不得读取outer-test outcome修改操作点。

grouped fixed baseline仍在每个outer-train上按既有
`全部非control overtake - 5 * safe-control harm`
选择单一动作并应用到outer-test；其聚合必须精确复现target 79、control `5/5`，否则只判实现
合同失败。

### 26.4 主判据、诊断曲线与停止规则

primary frozen-hidden nested selector只有同时满足以下三条才写
`frozen_hidden_conditional_value_reopened=true`：

1. 聚合225条safe controls中新collision不超过5；
2. 聚合225条safe controls中overtake loss不超过5；
3. 三主层target success严格大于79。

若三条通过，撤回“frozen hidden条件化价值为负”这一类级外推，重新打开frozen-feature
action-conditioned/Q-filtered selector；但Round Z2的具体13-class top-1 extraction仍保持失败，
本结果也不解决future-event窗口的部署trigger，不准入validation或PPO。

若两项harm均达预算但target不高于79，则严格关闭本节的
`frozen hidden × 3-class outcome × nested lambda/tau calibration × fixed 12-action library`
实例。若outer聚合harm超预算，则说明inner calibration不能跨startpoint保持预算，同样关闭该
实例；不得看outer结果后改lambda grid、预算或tie-break。

另输出基于outer-OOF概率的全局经验target-vs-control-harm Pareto曲线，只作诊断；它使用outer
outcome画图，不能替代nested单点判决。history treatment按完全相同nested协议报告，但只回答
显式50步历史在harm-matched条件下是否相对frozen hidden有增量，不改变primary判决。

### 26.5 Z5独立seed结果后的exact-Z4-seed复核合同（复核运行前冻结）

首次Z5严格按§26.2的新outer seeds执行，属于独立模型重复；它不是原Z4-A五个outer模型的
逐点校准。首次机器结果已经产生：frozen hidden nested为target 69、control `1/3`，fixed为
79、`5/5`，因此独立seed主Gate失败。这个结果保持有效，不覆盖、不改阈值。

为准确回答“原Z4 scorer在nested校准后怎样”，只允许增加一次seed sensitivity复核：inner
fold、inner seeds、lambda/tau、预算、tie-break和全部数据保持§26.2--§26.4不变；唯一变化是
最终outer model恢复Z4-A原seed `5200 + outer_fold`与shuffle seed `5300 + outer_fold`。
该复核写入新report，不覆盖首次Z5。

判读预先固定：

- 原seed复核也未同时满足`target > 79`与control `<=5/5`：确认原Z4的104优势来自激进操作点，
  tested harm-matched outcome selector关闭；
- 原seed复核通过而独立seed失败：只能判`seed-sensitive / inconclusive at method-class level`，
  不得选取较好seed宣布通过，也不准入validation或PPO；
- 两次都通过才可写稳定重开，但首次已经失败，因此本轮不可能得到该判决。

该敏感性复核不新增模拟，也不把outer-test outcome用于lambda/tau选择。

### 26.6 执行结果（2026-08-08，两个seed合同均失败）

两次nested运行都完成456条、70个startpoint、5个outer folds与每个outer-train内部4个
startpoint-disjoint inner folds；5,928个标签、hidden/history输入均finite，inner/outer预测完整。
grouped fixed baseline在两次均精确复现target 79、control `5/5`。

| selector合同 | inherited / created / lost | target | control collision / overtake loss | 判决 |
|---|---:|---:|---:|---|
| Z5 independent outer seeds，frozen hidden | `42 / 25 / 2` | **69** | **`1 / 3`** | harm通过，target失败 |
| exact Z4 outer seeds，frozen hidden | `42 / 23 / 1` | **66** | **`3 / 4`** | harm通过，target失败 |
| grouped fixed baseline | `45 / 33 / 1` | **79** | `5 / 5` | 对照 |
| Z5 independent outer seeds，50-step history | `40 / 23 / 0` | 63 | `6 / 8` | target与harm均失败 |
| exact Z4 outer seeds，50-step history | `40 / 15 / 0` | 55 | `7 / 7` | target与harm均失败 |

independent-seed frozen相对fixed的target配对discordance为`28 gained / 38 lost`，双侧exact
`p=0.268`；exact-Z4-seed为`29 / 42`，`p=0.154`。两者都不是显著优于fixed的边缘失败，方向上
也均少10--13条target。

诊断性全OOF曲线展示了为什么不得在同一outcome上选点：independent-seed概率若事后直接选择
全局`lambda/tau`，在control `3/5`可读到target 84；但nested无泄漏外推只有69。exact Z4 seeds
的事后matched-harm最优点也只有target 72、control `4/5`。原Z4的target 104来自未约束的激进
操作点，不在fixed的control预算上。

因此本节主判决是`close_tested_harm_matched_outcome_selector`：Round Z2的具体top-1 extraction
保持失败；Z4的frozen-hidden三分类outcome extraction在两套outer seed合同下都不能在matched
control harm上超过fixed。关闭范围是固定late future-event窗口、12动作、当前hidden、三分类
loss与nested calibration实例，不是frozen representation的理论信息上界。

50步history在相同nested协议下也没有增量，进一步确认§25的具体history treatment失败。
不运行validation candidate、actor接入或PPO。下一项若继续prefix-reset，必须先做完整simulator、
opponent planner、reward/wrapper与recurrent hidden的snapshot no-op逐位恢复工程门。

### 26.7 统计口径校正（2026-08-08，Z6执行前冻结）

§26.6的程序判决保持不变，但科学表述收窄如下：两次frozen相对fixed的target配对差异分别为
`28/38, p=0.268`与`29/42, p=0.154`，所以本轮是在`n=456`下**未检出严格优于fixed**，不是
证明frozen selector显著更差。66个discordant pair在双侧`p<0.05`下至少需要约`42/24`的分裂；
若真实paired净优势只有10，本设计的检验功效约为19%，而70个startpoint聚类还会降低有效样本量。

本轮也不是“精确使用满`5/5` harm的matched比较”。Nested选择约束的是两项期望外推率都不高于
`5/225`；外层实测只用了`1/3`与`3/4`，而fixed恰好用了`5/5`。因此准确命名是
**budget-constrained nested comparison**。在不读取outer outcome的前提下，不能强迫有限outer
样本恰好落到`5/5`；事后把outer点推到边界会泄漏。独立seed与exact-Z4-seed的事后frontier
`84 @ 3/5`和`72 @ 4/5`相差12，说明seed敏感性不可忽略，但只有两个seed，不能据此估计一个
稳定的“±6方差”。

所以关闭依据是：预注册的`target > 79`准入线两次均未满足，且没有无泄漏证据建立条件化增益；
关闭限于本节tested实例。此前用Z2的`51 < 79`声称“条件化价值为负”的理由已被Z4-A推翻，不能
继续作为关闭依据。50步历史在Z4-A及两次nested协议中三次都没有超过frozen hidden且harm更差，
足以关闭这个精确history treatment；但三次复用同一456-task数据，不得称为三套独立数据复现。

特别保留未测边界：会把辅助loss反传进student GRU、从而改变actor内部表征的
representation-only变体（2b）没有被Z2、Z4-A或Z5检验。它必须提出新的监督必要条件，不能继承
本Gate的通过或失败。

## 27. Round Z6-A：prefix-reset snapshot no-op工程门

### 27.1 被检验的必要条件

本Gate只回答一个问题：**不从场景起点重放时，训练进程能否把交互窗口起点的完整状态序列化、
恢复，并在相同网络与确定性动作下逐位复现原后缀。** 这是prefix-reset提高关键窗口采样密度的
工程必要条件，不是PPO有效性检验。

若该必要条件失败，prefix-reset退化为replay-to-prefix；每个episode仍需支付前缀仿真成本，
本项目按预注册直接停止，不把完整重放当fallback，也不启动prefix-reset训练。失败只否定当前
F110/LatticePlanner/PPO栈下的高效实现，不科学否定一般curriculum思想。

### 27.2 冻结任务与网络

- 地图只用Austin development；不读取validation或三张测试地图；
- 任务固定为Gate A由U42--U45至少3/4共识筛出的28条cohort，直接复用Gate B冻结窗口；其中
  collision 19条、lost-overtake 9条、21个ego startpoint、opponent raceline0/2=`20/8`；
- 每条在其冻结`window.start_index`的**当前observation被actor消费前**保存snapshot；
- actor固定U44 12-key checkpoint；critic固定同一U44的`privilege_gru` checkpoint；
- actor与critic都使用float32、确定性均值动作、初始hidden全零；不采样探索噪声、不更新参数；
- snapshot必须先完成一次Python pickle序列化/反序列化round-trip，再用于恢复，避免只证明内存对象
  引用可复用；
- 28条全量执行，不抽样、不按结果删场景。runner在任何结果写出前冻结plan并拒绝覆盖已有完整输出。

### 27.3 必须覆盖的状态

Snapshot至少包含并恢复：

1. F110每个RaceCar的7D state、`opp_poses`、accel、steer-angle velocity、steering delay buffer、
   collision latch和scan RNG bit-generator state；
2. Simulator的agent poses、collision flags与collision indices；F110Env的pose/collision镜像、
   current time、lap times/counts、near-start/toggle状态、起点与旋转矩阵，以及Gym
   `OrderEnforcing._has_reset`；scan map本身为同一Austin只读对象，不复制；
3. LatticePlanner controller的trajectory、tracker count与speed scale；planner的best trajectory、
   previous local trajectory/opponent pose、goal grid、nearest state、step costs、last progress、step
   counter与selection state；PurePursuit tracker的previous error和nearest distance；
4. PPO wrapper的raw observation、previous measured speed、elapsed time、current reset spec、reset RNG、
   episode return/step/reward分量、minimum clearances、risk-active count和corridor gate当前值；
5. reward内部的ego/opponent previous progress、relative position、collision latches、scenario identity、
   previous risk potential与current clearances；
6. actor与privilege-GRU critic在当前observation被消费前的recurrent hidden。

### 27.4 逐位判据

每条任务先从场景起点运行至冻结prefix，保存并round-trip snapshot，再继续到原终局；随后在同一
environment实例加载snapshot与hidden，第二次运行同一后缀。只有28/28全部满足以下条件才通过：

1. 恢复后的当前381D observation与snapshot前逐元素相同；actor/critic hidden逐元素相同；
2. 后缀长度相同；每步381D observation、actor raw/executed action、opponent executed action、
   critic value、reward总量与四个reward分量逐元素相同，最大绝对误差均为0；
3. 每步两车7D state、steering buffers、两车360D LiDAR、collision markers与wrapper elapsed time
   逐元素相同；所有数组finite且对齐；
4. terminated/truncated序列、首次ego collision step、episode outcome、terminal合同完全一致；
5. snapshot schema字段完整，pickle round-trip成功，runner没有跳过或重跑失败条目。

任何一条、任一字段非零即`fail_stop_prefix_reset_snapshot`；不放宽到数值容差、不删除边界场景、
不切换CPU来规避差异。通过只能写`pass_snapshot_mechanical_gate`。

### 27.5 通过后仍未解决的内容

本Gate通过也不准入formal PPO。下一道独立Z6-B必须检验：actor参数更新后，不能复用U44旧hidden，
而要用保存的actor-visible observation prefix对**当前网络**无梯度burn-in重算actor/critic hidden；
prefix transition不得进入rollout loss或GAE，reset窗口的episode-start、bootstrap、terminated/
truncated与advantage边界必须与标准RecurrentPPO语义一致。本Gate只使用确定性mean action，未
审计训练期exploration RNG/residual block；Z6-B必须预先固定prefix后恢复还是重采样探索状态，
并验证log-probability可按同一合同重建。之后才可讨论单位算力关键窗口密度与一次Austin-only
prefix-reset训练。

本Gate与representation-only 2b不同：prefix-reset改变采样分布与credit长度但保持标准PPO loss；
2b直接增加辅助loss并改变表征。两者都让训练中的表征可能变化，但不能互相代替证据。

### 27.6 执行结果（2026-08-08）

机器结果为`pass_snapshot_mechanical_gate`。28条冻结任务全部完成：来源分层collision 19、
lost-overtake 9，共21个ego startpoint；28份snapshot均完成pickle往返，28份原后缀与28份恢复
后缀全部存在且由独立复核重新加载。本轮当前环境原后缀终局为14次ego collision、6次overtake、
8次follow；它们不是对来源分层或U44性能的重新估计，只是恢复覆盖面。

所有预注册连续字段最大绝对误差均为0，包括恢复点381D observation、actor/critic hidden、
后缀每步observation、actor raw/executed action、opponent executed action、critic value、reward
及四分量、两车7D state与steering buffer、双360D LiDAR、elapsed time。Boolean collision、
terminated/truncated、action-applied与terminal-post-step逐元素相同；28/28 outcome、首次ego
collision step、absolute terminal step与episode return完全相同。严格JSON、finite、shape、key
集合和trace terminal合同全部通过。

冻结prefix为0--701步，26/28大于0，中位345.5步；原完整轨迹共16,385 action step，prefix合计
9,589步（58.5%）。后缀为99--800步，中位154.5。该58.5%只表示这批固定任务可避免的仿真步，
没有计入snapshot加载、并发和训练开销，不能当作实测wall-clock加速。

判决：Z6-A通过，证明当前F110/LatticePlanner/PPO wrapper栈存在精确、无需replay-to-prefix的
快照实现；不启动训练。下一步严格停在§27.5的Z6-B current-network burn-in与GAE/bootstrap语义
Gate。该Gate未通过前，旧U44 hidden不得作为参数更新后网络的起始hidden，prefix transition不得
进入PPO loss，也不得把人为窗口边界当成真实terminal。

## 28. Round Z6-B：current-network burn-in与PPO语义Gate

### 28.1 问题与固定证据边界

本Gate不训练actor，只检验Z6-A通过后剩余的三个工程必要条件：参数改变后能否从冻结prefix用
**当前网络**重建recurrent state；snapshot episode边界能否同时切断GAE/sequence又保留非零
burn-in hidden；窗口重新采样的探索分布能否在PPO replay中重建同一log-probability。

任务仍固定为Z6-A的28条Austin development共识任务及其window start，不读取validation或三张
测试地图。来源轨迹与snapshot固定U44；“当前网络”固定为同一真实训练轨迹相邻的U45 actor与
privilege-GRU critic，不做合成扰动、不更新参数。U45只用于证明参数变化后的状态重建，不是
checkpoint选择或性能比较。

每条Gate-only重放从场景起点运行一次，记录reset observation到window start前的prefix。Actor与
critic recurrent branch都只消费每步361D `LiDAR + previous speed`；P20特权量只在critic当前
value head late-fusion，因此不参与hidden递归，但window当前value必须使用snapshot的完整381D
pre-action observation。Prefix action、reward、value和advantage一律不进入候选rollout或loss。

### 28.2 Current-network burn-in判据

每条任务保存长度严格等于`window.start_index`的finite `float32 [P,381]` prefix，以及window当前
381D observation。固定两种计算路径：

1. collection-equivalent reference：从零hidden开始，按时间逐步、batch-size-one消费prefix；
2. fast sequence burn-in：同一网络一次消费完整`[1,P,*]`序列；`P=0`时返回严格零hidden。

只有以下条件全部满足才通过：

1. U44 reference的window observation、actor hidden与critic hidden相对Z6-A snapshot最大误差0；
2. U44和U45各自的fast hidden相对其reference最大绝对误差均不高于`5e-5`；
3. 用两种hidden消费window当前observation所得actor mean action与critic value误差均不高于`5e-5`；
4. U45 reference对28条全部成功，且26条非零prefix中至少一条U45 hidden/action/value与旧U44状态
   非零不同，证明“不得复用旧hidden”不是空判据；
5. 28条prefix总行数必须为9,589，identity、shape、finite和prefix长度全部匹配冻结plan。

若fast path超过`5e-5`但逐步reference仍逐位重现，记为`pass_semantics_reject_fast_burnin`：只否决
一次整段GRU加速，不否决无仿真的逐步burn-in。任一任务无法由当前网络重建，才
`fail_stop_prefix_reset_burnin`。

### 28.3 Boundary mask与GAE合同

现有RecurrentPPO的单一`episode_starts`同时承担三件事：GAE边界、sequence切分和hidden清零。
Snapshot reset需要前两项为true、hidden清零为false，所以本Gate固定增加一个opt-in
`recurrent_resets` mask：

- `episode_starts=true`仍供GAE和`create_sequencers`切断不同snapshot episode；
- `recurrent_resets=false`使replay使用该sequence已保存的burn-in hidden；
- 普通production collector不提供独立mask时，`recurrent_resets := episode_starts`逐元素复制，
  默认行为必须保持完全相同。

使用`gamma=0.999, lambda=0.995`的合成buffer同时覆盖：真实terminated不bootstrap、8秒timeout按
现有collector先加入`gamma*V_terminal`再切断、rollout末非terminal使用`last_values` bootstrap，
以及两个snapshot episode相邻但prefix transition完全缺席。GAE/return必须与独立手算递推最大
误差`<=1e-6`；sequencer必须在snapshot边界切段；replay收到的reset mask必须为false并保留已存
非零hidden。默认mask回归任一字段不等即失败。

### 28.4 Exploration与log-probability合同

正式prefix训练的探索模式尚未获准选择，因此工程Gate同时覆盖当前baseline逐步独立速度高斯和
U44来源配置的front-corridor-gated temporally correlated速度噪声，不据此选择训练臂。

Snapshot reset固定为**重新采样探索状态**，不恢复U44 residual/RNG：探索侧start为true，actor/
critic的`recurrent_resets`为false。对U45 current-network burn-in hidden：

1. baseline与corridor两种模式的collection old log-prob必须由现有collection-equivalent replay
   重建，`max |log_ratio|`与`max |ratio-1|`均`<=5e-5`；
2. corridor合成正门在首步开启、随后关闭：同一residual与正block id必须恰好保持50步，第51步
   inactive且block id为0；再次访问同一snapshot必须采到新residual；
3. exploration reset不得改变burn-in hidden，prefix rows不得进入rollout buffer、PPO ratio或GAE。

任何一条失败即`fail_stop_prefix_reset_semantics`，不通过改seed、容差、hold或探索模式规避。

### 28.5 通过与停止规则

全部准入线通过只能写`pass_prefix_reset_semantics_gate`，仍不准入formal PPO。它只允许下一道
Austin-only训练密度Gate预注册：必须另行固定snapshot episode占比、prefix集合/抽样、与现有
50/50 role及479 collision cache的关系、使用baseline还是corridor exploration，以及完整训练对照。
Z6-B失败则停止当前prefix-reset实现；不复用U44旧hidden、不把snapshot边界伪装成terminal、
不让prefix进入loss，也不回退replay-to-prefix。方法2b与Constrained PPO仍不受本Gate裁决。

### 28.6 首次执行结果（2026-08-08，strict machine fail）

28条任务与9,589行prefix全部完成；U44 source observation/actor hidden/critic hidden相对Z6-A
snapshot均为0误差。U45 current-network的fast sequence相对逐步reference最大hidden/action/value
误差分别不超过`5.84e-6 / 2.38e-6 / 1.49e-7`，通过事前`5e-5`线；26条非零prefix全部确认旧
U44状态与U45 current state非零不同。U44来源网络的一次整段fast path最大hidden约`0.0040`、
action约`0.00175`，按§28.2只拒绝该source fast path，逐步fallback仍逐位复现。

Boundary/GAE子门通过：advantage/return相对独立手算最大误差`1.49e-8 / 0`；边界在`0/3/5`
切段，replay reset全false并保留三组非零hidden；默认未stage mask时
`recurrent_resets == episode_starts`逐元素成立。Baseline 28条和corridor 1,428条的
collection-equivalent `max |log_ratio|`与`max |ratio-1|`均为0；corridor前50步active、同正block，
第51步inactive/block 0，再访问新residual也全部成立。

原机器判决仍为`fail_stop_prefix_reset_semantics`，唯一false criterion是
`first_50_max_residual_error == 0`：从已采样action按`(action-mean)/std`反算的telemetry residual
最大漂移`3.18885e-6`。这不改写为pass，也不删除原report。源码调用链显示该反算字段只进入
exploration telemetry，PPO likelihood replay只使用保存的action与speed log-std；所以“反算值
逐位相同”不是prefix-reset作用机制的必要条件，不能据此科学否决方法。下一步只能先执行§29
测量有效性裁决，不能直接进入训练密度Gate。

## 29. Round Z6-BR：temporal residual测量有效性裁决

### 29.1 为什么需要独立裁决

本轮不改Z6-B原判据或report，只回答其唯一失败量是否测到了真实必要机制。固定使用Z6-B已经保存
的U45 current-network hidden、51步suffix observation、seed 8602和28条任务；不重放simulator、
不换模型、任务、hold或seed。

直接记录policy内部实际用于`sample_with_speed_standard_noise`的`_temporal_speed_noise`，并与
buffer telemetry的反算`standard_residual`分开。准入线全部事前固定：

1. Z6-B除`corridor_restart_and_hold_passed`外所有primary criteria必须为true；该失败项内除
   `first_50_max_residual_error == 0`外的active/block/revisit条件必须已通过；
2. 内部temporal noise在首个block的50步逐元素最大误差必须为0；第51步必须inactive/block 0；
3. 反算telemetry residual只需落在Z6-B事前已经冻结的likelihood容差`5e-5`内，不新增容差；
4. collection-equivalent `max |log_ratio|`与`max |ratio-1|`仍必须`<=5e-5`；
5. 冻结plan、U45模型身份、任务数和transition数必须不变。

全部通过才写`pass_prefix_reset_semantics_after_measurement_adjudication`：含义是Z6-B原始严格
report发生非因果telemetry false fail，current-network/mask/GAE/likelihood必要条件科学通过，
可准入单独训练密度预注册。任何一条失败则维持`fail_stop_prefix_reset_semantics`并停止该方法。
该裁决不证明PPO有效，也不允许静默放宽未来ratio合同。

### 29.2 执行结果（2026-08-08，pass）

Z6-BR没有运行simulator；固定读取Z6-B的28条U45 current-network hidden与51步suffix，transition
仍为1,428，seed仍为8602。原strict report除`corridor_restart_and_hold_passed`外全部primary
criteria为true；该项内部又只有反算residual逐位为0失败，active、同正block、第51步release、
revisit新residual均已通过。

直接记录的内部`_temporal_speed_noise`在首50步最大误差0；第51步全部inactive、block id 0；
再次访问的反算residual最小绝对差`0.08756`。反算telemetry residual误差仍为`3.18885e-6`，低于
原冻结`5e-5`；collection-equivalent `max |log_ratio|`与`max |ratio-1|`仍为0。七条裁决criteria
全部true，最终判决为`pass_prefix_reset_semantics_after_measurement_adjudication`。

原Z6-B strict machine fail保留且不改写；科学解释是它测到FP32逆运算的telemetry非逐位性，
没有推翻用于采样的内部noise、block寿命或PPO likelihood。Z6-A/B合起来只准入下一道Z6-C
no-update training-density/integration Gate。Z6-C必须固定snapshot episode比例、prefix sampler、
50/50 role与479 cache语义、baseline或corridor探索，并实测完整102,400-transition layout、ratio、
GAE与wall-clock；未通过前不启动formal PPO。

## 30. Round Z6-C：no-update训练密度与完整集成Gate（运行前冻结）

### 30.1 被检验的必要条件与非目标

本Gate只回答：prefix-reset接入真实16-env训练collector后，能否在不更新任何参数的前提下，保持
完整PPO数据合同，并以有限墙钟开销把足量transition放到28个冻结交互窗口。它不检验策略是否
学会，也不读取四图正式评测或产生checkpoint。通过只准入单独的prefix-reset训练预注册。

固定比较两个各自fresh构建的完整rollout：标准baseline与prefix treatment。两臂均从canonical
BC `pretrained/end2race.pth`初始化，fresh `privilege_gru` critic、seed 42、Austin、16 env、
6,400 steps/env、batch 12,800、`gamma=0.999`、`lambda=0.995`、baseline逐步独立速度高斯探索；
不做critic warm-up、actor/critic optimizer step或checkpoint保存。顺序固定baseline后treatment，
wall-clock只作单次工程守门值，不作统计显著性或学习收益证据。

### 30.2 Prefix训练输入与抽样合同

训练输入固定为Z6-A/B已经验证的28个Austin U42--U45至少3/4共识prefix：collision 19、
lost-overtake 9、21个ego startpoint；每项包含完整simulator snapshot及从reset observation到
window前的381D prefix observation。它们必须迁入独立持久panel，不能让正式训练依赖可清理的
`eval_results`。Panel manifest冻结28个key、prefix长度合计9,589、snapshot/prefix文件名及内容
摘要；任一缺失、身份或摘要变化即fail closed。

Prefix只替换**collision角色episode reset的1/3**：全局collision reset按1-based ordinal
`3,6,9,...`选prefix，其余2/3继续消费既有479项collision cache；ordinary角色仍只消费production
600项uniform ordinary pool。Prefix queue用`SeedSequence([42, 0x50524658])`独立shuffle，完整遍历
28项后才重排下一cycle；prefix reset不消费479 queue。环境slot奇偶不变，因此每个完整rollout
必须仍严格为51,200 collision-role与51,200 ordinary-role transition。Prefix的`env_role`保持
`collision`，但`pool/sampler_branch`明确标为`prefix_reset_consensus_v1/prefix_reset`，不得伪装成
479 cache样本。

Snapshot恢复后，actor与critic hidden都必须用**当前内存网络**对该项prefix逐步、batch-size-one、
无梯度burn-in；禁止复用U44/U45保存hidden或未经逐update校准的fast sequence。训练buffer只接收
window当前observation之后的transition：边界`episode_starts=true`，但
`recurrent_resets=false`；探索侧仍把它当新episode并重新采样。普通reset继续满足两mask相同。

### 30.3 完整rollout判据

两臂各自必须同时满足：

1. buffer严格`6400 x 16 = 102,400` transition，所有observation/action/reward/value/log-prob/
   advantage/return finite；按env slot计数严格50/50 role；actor与critic参数在rollout前后逐tensor
   相同；
2. 以保存的最终value/done独立重算全buffer GAE，advantage和return最大误差均`<=1e-6`；
3. 全buffer collection-equivalent replay覆盖102,400项，`max |log_ratio|`与`max |ratio-1|`
   均`<=5e-5`；batched replay仍需落在既有`1e-2` envelope；
4. baseline中prefix transition/reset均为0。Treatment的prefix reset数必须恰好等于
   `floor(collision reset总数/3)`，28个prefix key全部至少出现一次；非prefix collision reset全在
   冻结479 cache，ordinary reset全在冻结600 pool；
5. treatment中prefix-origin transition至少占完整buffer的5%，且每个prefix reset起始后的前150步
   （若更早terminal则取实际长度）合计至少占完整buffer的2%。这两项是训练密度必要条件，不是
   actor收益；
6. treatment snapshot边界全部满足`episode_starts=true/recurrent_resets=false`，非snapshot边界两者
   相同；prefix observation本身进入buffer的transition数为0；
7. treatment实测wall-clock每transition不得比baseline慢20%以上。另报告按prefix长度计算的
   replay-to-prefix额外sim步数，但该反事实步数不叫实测加速，也不替代wall-clock判据。

任何一条失败即`fail_stop_prefix_reset_density_or_integration`，停止当前prefix-reset实现，不通过
改变比例、探索、prefix集合、容差或只跑短rollout重试。全部通过才写
`pass_prefix_reset_density_integration_gate`；它只允许另行冻结formal训练的update预算、单一
treatment、checkpoint band与四图验收线。方法2b、collision-only新panel与Constrained PPO均不受
本Gate科学裁决。

## 31. Round Z6-CR：batched replay刀锋失败的因果裁决（运行前冻结）

### 31.1 裁决问题

Z6-C原machine report必须保留。它的11/12条criteria通过，唯一失败是treatment batched replay
`max |log_ratio|=0.0106971`和`max |ratio-1|=0.0107546`略高于预注册`1e-2` envelope；同一
102,400 transition的collection-equivalent两项均严格为0。该失败不是非因果telemetry，因为
batched replay实际用于PPO actor loss；但`1e-2`是现有实现触发exact audit的工程envelope，不是
prefix-reset作用机制本身的自然必要界。因此不允许直接改写Z6-C为pass，也不允许据此科学否决
方法，先单独检验这点数值差异是否实质改变PPO梯度。

### 31.2 固定重跑与指标

使用Z6-C完全相同的canonical BC、fresh privilege-GRU critic、seed 42、Austin 16 x 6,400、
baseline探索、28-prefix panel及collision reset 1/3比例，重跑一个无更新treatment rollout；不改
prefix比例、数据、网络、batch或随机种子。必须先复现Z6-C的102,400 transition、28-key覆盖、
`>=5%/>=2%`两条密度、GAE `<=1e-6`、collection-equivalent ratio `<=5e-5`及参数不变。

在同一冻结buffer上用同一组env-major minibatch indices分别执行batched和collection-equivalent
actor replay，均保持参数不动：

1. 保存所有valid transition的batched `log_ratio`分布、`ratio`分布、近似KL
   `((exp(log_ratio)-1)-log_ratio)`与clip=0.20下的clip fraction；
2. 各自按现有per-minibatch advantage normalization和PPO clipped surrogate做一个dry actor epoch，
   每个minibatch只backward、不step，将8个minibatch的actor gradient逐参数求和；
3. 比较两条累计gradient的cosine与相对L2差，并报告每minibatch policy loss差；所有tensor必须finite。

### 31.3 冻结判据与停止线

只有以下全部成立才写`pass_prefix_reset_after_batched_gradient_adjudication`：

1. Z6-C的非batched必要条件全部复现，collection-equivalent两项仍`<=5e-5`；
2. batched ratio的clip fraction严格为0，最大绝对ratio偏差小于0.02，mean approximate KL
   `<=1e-4`；
3. batched与exact累计actor gradient cosine `>=0.999`，相对L2差`<=0.02`；
4. 两种dry epoch全部8个minibatch完成，gradient/loss finite，参数前后逐tensor不变。

通过含义仅是Z6-C的`1e-2`刀锋失败没有推翻“可用同一PPO更新语义训练”的必要条件；原Z6-C
machine fail不覆盖。任一条失败则维持`fail_stop_prefix_reset_density_or_integration`并停止当前
prefix-reset实现。该裁决仍不证明PPO学习有效，也不允许先训练再解释。

## 32. Round Z6-F：prefix-reset单次正式PPO（运行前冻结）

### 32.1 准入依据与唯一训练变量

只有Z6-CR全部通过才执行本节。Z6-C原始batched-envelope machine fail继续保留；Z6-CR若证明
full dry actor gradient满足事前方向/尺度界，只解除该刀锋失败对正式训练的阻断，不把工程Gate
写成actor有效。

正式run固定`EXPERIMENT_ID=ppo_prefix_reset_consensus1of3`。它相对production baseline PPO的唯一
实验变量是：collision角色每三个reset中一个从`prefix_reset_consensus_v1`的28个冻结snapshot
起始；其余配置全部保持production。禁止同时加入corridor temporal探索、BC loss、cost critic、
新reward、场景重权或第二阶段repair。

| 参数 | 固定值 |
|---|---:|
| 初始化 | canonical BC `pretrained/end2race.pth` |
| 训练地图 | Austin only |
| seed | 42 |
| critic | `privilege_gru` |
| logical envs / workers | `16 / 16` |
| steps per env / batch | `6400 / 12800` |
| critic warm-up / formal updates | `1 rollout / 30` |
| actor / critic epochs | `2 / 5` |
| GRU / head / critic LR | `3e-6 / 3e-5 / 3e-4` |
| steering latent std / speed physical std | `0.03 / 0.15` |
| exploration | baseline逐步独立速度高斯 |
| gamma / lambda / clip | `0.999 / 0.995 / 0.20` |
| collision / ordinary | 479 cache / production uniform 600；transition仍50/50 |
| prefix | 28项共识panel；collision reset interval 3；逐步current-network burn-in |

Warm-up与30个formal rollout都启用prefix。每轮必须记录prefix transition/window fraction和boundary
数；任一formal update的collection-equivalent ratio超过`5e-5`，或batched最大ratio/log-ratio偏差
达到`0.02`，立即中止且不得从已有目录续写。训练只保存标准12-key actor与训练期critic，部署时
snapshot、prefix、critic全部删除；actor输入、结构与BC完全兼容。

### 32.2 Checkpoint与评测合同

主checkpoint固定U30；U27、U28、U29只构成预先固定的晚期band，不按结果选单点。训练完整性要求
warm-up加30条finite metrics、U1--U30 actor/critic齐全、最终actor存在、无写进程、所有prefix与
ratio telemetry满足上节运行时合同。

U27--U30全部在Austin、Hockenheim、MoscowRaceway、Nuerburgring各固定600 episode上CUDA、
deterministic mean action、ego collision scope、numeric trace评估。四图都具有正式验收权；
训练与checkpoint选择不得读取后三图。每图相对canonical BC和历史U44报告collision removed/
created、overtake lost/gained及paired exact p；opp-wall单列。U30是唯一主判决，band只判断波动。

### 32.3 判决层级与停止规则

1. **训练实现通过**：完整性、ratio、prefix密度与12-key合同全部通过；只说明run有效。
2. **最低验收通过**：U30每张地图ego collision均不高于canonical BC且overtake均不低于BC；任一
   地图失败即不进入production候选。
3. **最终目标通过**：U30四图合计严格`collision < 40`且`overtake > 1500`，同时满足逐图BC线。
4. **稳定性支持**：U27--U30中至少连续两个checkpoint满足逐图BC线；它不允许替代U30主判决。

若只改善U44或扩展现有collision--overtake前沿但未达到最终目标，必须按真实数字写“经验前沿
扩展，目标未完成”；若U30相对U44两轴均不改善则明确否决该正式实例。无论结果如何，本配置只跑
一次：不扫prefix比例、窗口、panel、exploration、updates或学习率，不从U27--U29挑production，
不继续到U45。完成四图配对分析后关闭本prefix-reset实例；方法2b、collision-only与Constrained
PPO仍需各自必要条件，不能由本run代判。

### 30.4 Z6-C执行结果（2026-08-08，原machine fail保留）

Baseline与treatment各完成102,400 transition，均严格collision/ordinary=`51,200/51,200`、全字段
finite、GAE/return独立重算最大误差0、参数前后不变、collection-equivalent ratio两项0。Treatment
有39次prefix reset，精确等于119次collision reset的`floor(119/3)`；28 key全覆盖，非prefix
collision全部属于479 cache，ordinary全部属于600 pool。Prefix-origin为8,634 transition
（8.43%），其中首150步窗口5,438（5.31%）；39个snapshot边界的window observation误差0，
`episode_starts=true/recurrent_resets=false`，普通边界mask mismatch 0，prefix rows进入buffer计数0。

Baseline/treatment收集墙钟为`98.97/112.32s`，比值`1.1349`，通过不慢20%的工程线；若使用
replay-to-prefix会另付13,965个不进buffer的sim step，该数只作反事实成本。唯一false criterion是
treatment batched replay `max |log_ratio|/|ratio-1|=0.010697/0.010755`略超`0.01`，baseline为
`0.006786/0.006809`；因此原verdict保持`fail_stop_prefix_reset_density_or_integration`，没有直接
准入formal PPO。

### 31.4 Z6-CR执行结果（2026-08-08，pass）

独立full treatment重跑再次得到102,400 transition、28-key覆盖、prefix/window fraction
`8.43%/5.31%`、GAE误差0、collection-equivalent ratio两项0及参数不变。Batched异常高度集中：
mean absolute log-ratio `4.15e-6`，p99 `2.79e-5`，p99.9 `1.48e-4`，最大仍`0.010697`；mean
approximate KL `9.05e-10`，clip=0.20 fraction严格0。

同一8个minibatch的batched/exact dry actor epoch均完成且不step。累计gradient cosine
`0.9999838`，相对L2差`0.005706`，最大minibatch policy-loss差`4.73e-7`，全部通过事前
`>=0.999/<=0.02`线；actor digest前后相同。判决为
`pass_prefix_reset_after_batched_gradient_adjudication`：Z6-C原`0.01`刀锋失败未推翻可用同一PPO
语义训练的必要条件，只准入§32一次formal run，不证明actor会改善。

### 32.4 Z6-F执行结果（2026-08-09，最低验收通过，最终目标与稳定性未通过）

唯一预注册run完整结束：warm-up加30条formal metrics共31行、U1--U30 actor/critic齐全，全部
finite，30个actor均为12-key，所有formal update完成16/16 actor optimizer steps。Pre-update
batched ratio偏差最大为U26的`0.019095 < 0.02`，collection-equivalent exact最大0；prefix与
首150步window fraction全程范围分别为`8.43%--12.47%`和`4.26%--5.47%`。训练实现层通过。

U27--U30共16个四图CUDA deterministic包全部完成；每包600 result与600 numeric trace同key，
合计9,600 episode/trace，0 error，finite、terminal、action-applied与ego collision typed marker
全部通过。逐图collision/overtake为：

| update | Austin | Hockenheim | MoscowRaceway | Nuerburgring | 四图合计 | 逐图BC线 |
|---:|---:|---:|---:|---:|---:|---|
| U27 | `17/365` | `29/363` | `36/391` | `23/399` | `105/1518` | 失败：Hockenheim `29>27` |
| U28 | `20/367` | `27/365` | `32/395` | `22/398` | `101/1525` | 通过 |
| U29 | `23/367` | `28/362` | `27/399` | `23/400` | `101/1528` | 失败：Hockenheim `28>27` |
| U30 | `21/365` | `27/365` | `33/393` | `22/399` | `103/1522` | **通过** |

U30相对BC四图collision removed/created=`60/34`（双侧精确`p=0.00955`），overtake
lost/gained=`16/93`（`p=2.21e-14`）；因此最低验收通过。相对U44则collision
removed/created=`40/81`（`p=0.000244`，显著更差），overtake lost/gained=`33/77`
（`p=3.30e-5`，显著更好），是明确的安全--超车交易而非U44 Pareto改进。

最终`collision < 40`目标以103显著未达，虽然`overtake > 1500`以1522达到。Late band只有U28、
U30通过，中间U29失败，故“至少连续两个checkpoint”稳定性支持未通过。相邻checkpoint的四图
配对变化均不显著，aggregate仅在`101--105/1518--1528`窄带内波动；稳定性失败是预注册逐图硬
门的真实刀锋结果，不能改写成统计显著崩溃，也不能事后放宽Hockenheim门。

判决：**经验高超车端前沿扩展，最终目标未完成，关闭本tested prefix-reset配置。** 本run证明
增加关键窗口采样密度可产生一个逐图不差于BC的U30 actor，但没有把安全与progress统一到目标
区域。它不构成对prefix-reset方法类的严谨否决；不扫interval、panel、窗口、LR、exploration或
updates，不延长U45，不从U28挑production。Production保持原U30，方法2b、collision-only与
Constrained PPO仍需各自独立必要条件。

## 33. Round Z7：collision-only BC anchoring overlap-supported 独立重开

本节在任何Z7 actor outcome产生前冻结。冻结时目标评估目录不存在；旧Z3的7条validation
source与control缺口不参与样本选择。Z7不修改§24.5的inconclusive判决，而是用全新起点和事前
固定的support-overlap estimand回答同一个teacher必要条件。准确方法仍是
`Single-stage PPO with selective BC functional regularization`的collision-only子变体，不是
纯PPO；本Gate只检验teacher，不授权训练。

### 33.1 冻结panel与分阶段screen

固定Austin panel为40个循环等progress起点乘`raceline0/2`、interval `8/10/12/15`、speed
`0.45--0.85`每0.05一级的完整Cartesian product，共2,880条。起点由字符串
`collision-only-anchor-overlap-v2|Austin|<offset>`的SHA-256排序搜索offset，取第一个与历史
heldout candidate和Austin600起点精确零交集的40点集合；固定offset为1629，起点为：

`1629, 1681, 1734, 1786, 1839, 1891, 1943, 1996, 2048, 5, 57, 109, 162, 214, 267, 319, 371,
424, 476, 529, 581, 633, 686, 738, 791, 843, 895, 948, 1000, 1053, 1105, 1157, 1210, 1262,
1315, 1367, 1419, 1472, 1524, 1577`。

冻结ScenarioSpec的SHA-256为
`4dd93f37cdb03a6e2e7ecd3fda06df1a6de5702fa5eddf8b54843737bb60322d`；2,880个episode key唯一，
两条opponent raceline各1,440条、四个interval各720条、九个speed各320条。

为减少无效仿真而不改变cohort定义，actor screen固定分两步：

1. canonical BC与U44在全部2,880条上CUDA deterministic、ego collision scope、8秒运行并保存
   numeric trace；
2. 只有`BC=overtake`且`U44 in {ego-opp, ego-wall}`的潜在source才补跑U42/U43/U45；候选集合
   在后三个actor outcome可见前冻结。因为其他场景在定义上不可能成为source，省略它们的三次
   评估与完整五actor screen完全等价。

稳定source仍要求U42--U45至少3/4不是overtake；U44必须是ego collision，BC必须overtake。
source窗口固定为U44第一次ego collision前150个action step，少于50步者在branch前排除。
safe control要求BC与U44都overtake，窗口仍为U44全局最小OBB clearance前100至后50步，少于
50步者排除。不得使用Z3、development或standard panel补样。

### 33.2 support-overlap estimand与V0样本门

Z3失败在所有source都要求1:1 exact control，而未预先处理分层support。Z7明确把estimand收窄为：

> 在每个`(opponent raceline, speed scale)`分层中，同时存在eligible stable source和eligible
> safe control的overlap-supported source。

每层source按`SHA256("collision-only-anchor-overlap-v2|source|" + scenario_key)`排序，只取前
`min(N_source, N_control)`条；该规则可看actor outcome确定support数量，但不看任何branch结果。
随后按循环ego waypoint距离、scenario-key SHA破平为每条source选同raceline、同speed、无放回
control。这样exact control在运行branch前由构造保证；代价是结论只适用于overlap-supported
source，不外推到被support裁掉的稳定collision。

V0只有同时满足以下条件才运行branch：

1. overlap-supported source至少12条；
2. 覆盖至少8个ego startpoint；
3. `raceline0/2`各至少2条；
4. exact同raceline/speed、无放回control与source等数；
5. 所有被需要的actor result/trace key、identity、finite、typed collision marker、terminal和
   action-applied合同全部通过。

任一条失败只判`inconclusive / independent-panel sample insufficient`，不运行branch，不科学
否决collision-only teacher，也不扩大panel或改变support规则。

### 33.3 V1唯一branch、必要条件与停止规则

只运行branch0与full-BC；不再运行steering-only/speed-only。branch0须在全部source+control上
对U44 reference逐步复现：ego raw/executed action严格误差0，opponent action、两车pose/speed、
双LiDAR最大误差`<=1e-6`，boolean marker、outcome、steps、首次collision identity/time step及
terminal合同完全一致。branch0失败只否实现，不运行full-BC。

full-BC在冻结1.5秒窗口同时替换steering和speed，随后恢复U44。必要条件沿用§24.3且事前固定：

1. collision rescue至少`ceil(0.50 * N_source)`；
2. 被救回者中最终overtake至少`ceil(0.80 * N_rescued)`；
3. rescue覆盖至少2个ego startpoint；对source至少2条的每个raceline至少救回1条；
4. controls新造ego collision与overtake loss分别不超过`floor(0.05 * N_control)`。

全部通过才写`mechanism_replication_pass=true`，只证明teacher在新起点的overlap-supported
collision上有可选择局部信号；随后仍需另写shadow-gradient与formal PPO预注册。若V0样本充分、
branch0精确且产物完整，而任一full-BC必要条件失败，则严谨关闭
`canonical BC × overlap-supported stable collision × fixed 1.5s window × collision-only`
实例；不得看结果后改窗口、teacher、support cap、门限或raceline。无论通过或失败，Z7只运行
一次，不把validation branch动作写进训练anchor。

### 33.4 执行结果（2026-08-09，V0通过、V1必要条件失败）

BC与U44完整2,880条screen及U42/U43/U45各59条候选screen全部完成，共5,937个fresh actor
episode及同数numeric trace，0 error；result/trace key、scenario identity、finite、typed collision
marker、terminal与action-applied合同全部通过。BC完整panel为`123 collision / 2472 overtake`，
U44为`104/2457`；二者交集产生59条潜在source。U42/U43/U45在这59条上的collision/overtake为
`38/18、35/22、40/18`。

四checkpoint至少3/4非overtake的稳定collision为41条，全部窗口至少50步。overlap cap没有裁掉
source：最终41 source + 41 exact control，覆盖21个ego startpoint，raceline0/2=`31/10`；V0五条
门全部通过，旧Z3的control-support阻断被真正解除。

branch0在82条上精确复现U44：raw/executed action、opponent action、两车pose/speed、双LiDAR及
全部报告数值字段最大误差均0，boolean、outcome、steps、首次collision和terminal合同完全一致。
full-BC结果为：

| 必要条件 | 实测 | 预注册线 | 判决 |
|---|---:|---:|---|
| collision rescue | `18/41 = 43.9%` | `>=21/41` | **失败** |
| rescued最终overtake | `18/18` | `>=15/18` | 通过 |
| rescue起点/raceline覆盖 | 11起点；r0/r2=`16/2` | 至少2起点、每条raceline至少1 | 通过 |
| control新collision | `4/41 = 9.8%` | `<=2/41` | **失败** |
| control overtake loss | `4/41 = 9.8%` | `<=2/41` | **失败** |

source最终为18 overtake、23 ego-opp；control为37 overtake、4 ego-opp，四个control harm身份完全
重合。Rescue率与harm率的95% Wilson区间分别约`29.9%--59.0%`与`3.86%--22.55%`；区间只描述
不确定性，不替代事前硬门。

因此V1 verdict为`fail`，`mechanism_replication_pass=false`。这是样本充分、精确重放、对照闭合
后的必要条件反例，严谨关闭
`canonical BC × overlap-supported stable collision × fixed 1.5s window × collision-only`
实例：teacher平均救援低于要求且对本来安全的matched controls产生超门限伤害。不得生成anchor
dataset、做shadow beta或formal PPO。结论不外推到其他teacher、其他窗口或被support排除的source，
也不改写原双分层Gate B与旧Z3各自的判决。

## 34. Round Z8：GRU-changing paired action-response auxiliary Gate（2b首次直接检验）

本节在任何Z8模型训练或报告产生前冻结。它直接针对§17.4的representation-only 2b：辅助loss
反传进入student GRU，actor output head不接收该loss，训练后response head可删除。Z2、Z4-A与
Z5都冻结了原GRU；本轮不再把冻结表征probe误写成被训练表征的上限。

### 34.1 证据边界与开发数据复用

本Gate零新仿真，复用Round Z2已经通过精确重放与trace合同的456条Austin development task、
late窗口、noop+12 residual真实闭环结局。其五层分母固定为inherited collision 109、created
collision 46、lost-overtake 13、inherited-follow 63、safe control 225，按70个ego startpoint的
既有五折分组。

必须披露：这些branch aggregate已经被Z2/Z4/Z5查看过，所以本轮只能作development机制筛查；
通过后必须另开独立startpoint反事实validation，不能直接准入PPO。失败则只关闭本节固定的监督、
窗口、优化合同，不否决所有2b或GRU的信息论上限。

### 34.2 与已失败Z4-A不同的监督和真实GRU更新

Z4-A用冻结U44 hidden加外置50步history GRU预测三分类outcome；Z8不增加部署输入，也不训练外置
history encoder。每个decision state先按真实U44逐步语义从episode起点burn-in到late start前50步，
随后用actor原本的420D GRU输入运行最后50步。冻结U44 GRU必须重建已保存late hidden和该步raw
action，最大误差均`<=1e-5`，否则只否实现并停止。

Control用保存的frozen U44 hidden训练response head。Treatment从同一U44 GRU初始化，让同一
response loss反传进原1680D GRU；LiDAR参数`k`、speed MLP和actor output layer全部冻结。Head固定：
hidden `1680->192 ReLU`，拼二维归一化residual action，再`194->64 ReLU->2`。两个输出不是旧三
分类，而是：

1. candidate最终是否发生ego collision；
2. candidate最终relative progress相对同状态noop的米制差。

Loss固定为`0.5 × class-balanced BCE + 0.5 × SmoothL1`；progress只按训练split均值/标准差归一化。
GRU/head LR=`3e-6/3e-4`，Adam weight decay `1e-4`，gradient norm clip 1.0，batch64，固定10 epoch；
不做early stop、checkpoint选择、loss权重或LR扫描。GRU参数相对变化每fold须`>=1e-7`且test hidden
平均相对L2须`>=1e-5`，只用来确认representation确实改变，不作为效果替代指标。

### 34.3 无泄漏train/calibration/test与操作点

对每个test fold `f`，calibration固定为`(f+1) mod 5`，其余三个fold只用于训练；因此每个ego
startpoint恰好一次进入test，且test outcome既不训练模型也不选择操作点。两套独立seed base固定
为7100和8100，fold seed=`base+f`，shuffle seed=`base+f+100`；两套都必须通过。

Response score为预测progress delta减`lambda × P(collision)`。`lambda`固定网格
`0,.25,.5,1,2,4,8,16`；noop threshold只枚举calibration中best-nonnoop减noop的预测margin及
infinity。Calibration只允许safe-control新collision与overtake loss各
`<=floor(.05*N_control_cal)`；可行点依次按target success高、两类harm低、干预少、lambda大、
threshold大破平。冻结后只在对应test fold应用一次。该3/1/1 rotating split避免Z4-A激进操作点
与Z5 outer后验调budget的问题，不声称每个test fold恰好用满harm预算。

### 34.4 事前判据与停止规则

每套seed都必须同时满足：

1. 五fold的GRU参数和test hidden确实按§34.2改变；
2. 聚合225 safe controls中新collision与overtake loss各不超过5；
3. treatment target success至少88，即严格超过既有matched-harm fixed baseline 79至少9；
4. treatment比本轮同监督frozen-hidden control至少多9个target success；
5. inherited collision、created collision、lost-overtake恢复分别至少11、7、4。

两seed全过才写`pass_to_independent_validation`；它只证明该paired collision/progress loss能改变
GRU并产生跨startpoint可行动作结构，仍不证明PPO actor会利用它。任一seed失败则关闭
`U44-initialized GRU × late 50-step actual recurrent window × paired collision/progress auxiliary ×
10 epoch × rotating calibration`具体2b实例，不运行独立validation或PPO，不改epoch/LR/loss/
lambda/threshold重试。报告必须显式保留`representation_only_2b_class_refuted=false`，避免把具体
负结果外推为所有辅助表征学习无效。

### 34.5 执行结果（2026-08-09，具体2b实例失败）

首次预计算误用了batched burn-in并被事前exact门拦下（hidden/action误差
`0.021586/0.009939`），没有训练报告；修正为逐episode、逐step、batch-size-one后，456条的
U44 late hidden与raw action重建最大误差均严格0。正式输入为`456×50×420`，所有finite；5,928
个action-response标签完整，noop progress delta最大绝对值0。

两套seed的五fold treatment均真实改变GRU：参数相对L2范围约`0.00210--0.00260`，test hidden
平均相对L2约`0.00938--0.01231`。因此本轮确实触及2b机制，而不是再次读取冻结hidden。冻结
output head下actor mean动作也随hidden改变；平均steering/speed绝对变化约
`0.00092--0.00188 rad / 0.00964--0.0191 m/s`，个别最大值更大，只作functional-drift诊断。

最终无泄漏test汇总为：

| seed | frozen hidden target（I/C/L） | frozen control collision/loss | trainable GRU target（I/C/L） | treatment control collision/loss |
|---:|---:|---:|---:|---:|
| 7100 | `58 (37/21/0)` | `12/13` | **`50 (30/20/0)`** | **`14/15`** |
| 8100 | `61 (38/22/1)` | `19/20` | **`58 (36/21/1)`** | **`12/13`** |

Seed7100 treatment相对frozen的target独有/对照独有成功为`2/10`，paired exact `p=0.0386`，即
显著变差；seed8100为`11/14, p=0.690`，无优势。Seed8100的control harm相对frozen减少7次，
paired `1/8, p=0.0391`，但绝对`12/13`仍明显超过5/5预算，不能以相对改善替代准入。两个seed
lost-overtake仅0和1。

Calibration fold内每折都遵守其1或2条harm上限，但冻结操作点迁移到独立test fold后聚合harm
失控；这正是本轮train/calibration/test拆分要检测的泛化失败，不允许再用test outcome回调
lambda或threshold。两seed均只通过“表征确实改变”与低门槛I/C条数，失败target 88、相对frozen
+9、lost 4和5/5 controls。

判决为`fail_close_tested_representation_only_instance`：关闭
`U44-initialized GRU × late 50-step actual recurrent window × paired collision/progress auxiliary ×
10 epoch × rotating calibration`具体2b实例，不运行独立validation或PPO，不扫描epoch、LR、
loss权重、lambda或threshold。`representation_only_2b_class_refuted=false`保持明确：证据否定的
是该监督/优化实例，不是所有能反传进GRU的辅助表征目标。

## 35. Round Z9：collision-cost Constrained PPO 独立预注册

本节与机器`gate_plan`在任何Z9 rollout前冻结。用户已经授权继续完成剩余候选；本轮只检验
`Lagrangian Constrained PPO with an independent collision-cost critic`的一个固定实例，不把
cost-critic预检失败或单seed正式训练失败外推成约束式RL理论不可能。

### 35.1 唯一化目标与固定算法

现有reward在首次ego collision给`-2.0`。Z9不保留重复计价：每次rollout收集后，在GAE前从
reward return中逐transition精确减去`reward_collision`分量，即碰撞步加回2.0；同一事件只进入：

```text
cost_t = 1[first ego collision transition]
```

Reward critic仍为当前fresh `privilege_gru`，cost critic固定为训练期P20 MLP
`20 -> 120 ReLU -> 30 ReLU -> 1`。`gamma_cost=0.999`、`lambda_cost_GAE=0.995`；actor把
`A_reward - lambda * A_cost`合成后只归一化一次，再进入当前clipped PPO surrogate。最终actor
仍是361D输入、12-key checkpoint；cost critic和dual变量部署时删除。

安全预算固定为当前人为50/50训练分布上**已完成episode**的first-ego-collision率`d=0.10`。
`lambda_0=1.0`、dual LR `0.5`，每次完整rollout后更新一次并投影到`[0,20]`；不扫budget、初值、
dual LR、上界或更新频率。Cost critic LR `3e-4`、正式每update 5 epochs；actor、reward critic、
探索、seed42、16 env、6400 step、batch12800、Austin池和30 formal updates保持当前合同。

### 35.2 Z9-A训练前Gate

先收集canonical BC的一个完整102,400-transition Austin rollout，不做actor optimizer step。
除机械合同外，对完整episode构造真实discounted first-collision cost-to-go，按`ego_idx`把起点
严格分成5折；每折用其余起点训练同构P20 cost critic 10 epochs，只在held-out起点预测。必须
同时满足：

1. 102,400行完整，cost event数严格等于已完成collision episode数；
2. reward去重逐行最大误差0，collision reward非零位置与cost位置完全相同；
3. cost advantage全部finite且标准差至少`1e-3`；
4. cost warm-up真实改变cost critic而不改变actor；
5. 合成advantage相对reward-only的全buffer actor梯度相对L2至少`1e-4`且全部finite；
6. 五折至少20个独立起点、10个collision episode且每fold有正例；
7. OOF MSE相对fold-train均值常数baseline的skill至少`0.05`；
8. episode首步最终collision AUROC至少`0.65`；距terminal至少100步的early AUROC至少`0.65`；
9. warm-up collision率高于0.10时dual必须按公式上升。

任一条失败即停止当前`P20 MLP × d=.10 × fixed dual`实现，不运行formal；这只是治理性不准入，
不是Constrained PPO方法类的必要条件证伪。全部通过才允许一次formal Austin训练。

### 35.3 Formal验收与停止规则

Formal固定从canonical BC fresh-start到U30，不续训、不模型选择。训练完成后只评U27--U30，
Austin、Hockenheim、MoscowRaceway、Nuerburgring各固定600条，CUDA deterministic、ego scope、
保存numeric trace；四图在算法、budget和checkpoint band冻结前均不参与任何调节。

三个判据分开报告：

1. 最终联合目标：U30四图合计`collision < 40`且`overtake > 1500`；
2. 最低产品门：U30每张图collision不高于canonical BC且overtake不低于BC；
3. late稳定性：U27--U30至少两个相邻checkpoint都逐图通过BC线。

同时报告每图与四图合计、相对BC/U44/production U30的removed/created、lost/gained和exact paired
检验，训练collision rate、cost value fit、dual轨迹及两类advantage尺度。无论成功或失败都关闭
该固定实例；失败不得改`d`、dual、cost critic或延长U45重试。正式失败只说明该目标定义和优化
配置没有突破当前前沿，不证明所有constraint定义无效。

### 35.4 Z9-A执行结果（2026-08-09，OOF可学习性失败，formal未运行）

最终有效preflight完整收集102,400 transition；其中98,737条属于153个已完成episode，覆盖85个
ego startpoint，57个episode发生ego collision。五折每折17个held-out起点，collision episode
为`11/14/14/9/9`，因此不是正例或起点coverage不足。

机械链路全部通过：57个cost event严格等于57个已完成collision episode；`reward_collision`
非零位置与cost完全相同，逐行从reward return去重最大误差0；cost advantage/return标准差为
`0.27325/0.25761`。Reward critic和cost critic都完成warm-up，actor摘要前后不变。训练分布已
完成episode collision率`57/153=37.25%`，故dual按冻结公式从`1.0`升至`1.13627`。在该lambda下，
全buffer合成梯度相对reward-only的差分相对L2为`0.31087`、cosine `0.96687`：cost信号确实能
进入actor，失败不能解释为零梯度或重复计价。

但三条startpoint-OOF可学习性门全部失败：

| 判据 | 实测 | 预注册线 | 判决 |
|---|---:|---:|---|
| MSE skill vs fold-train mean | `0.04038` | `>=0.05` | 失败，差0.00962 |
| Episode-start collision AUROC | `0.42855` | `>=0.65` | 失败 |
| 距terminal至少100步 early AUROC | `0.60703` | `>=0.65` | 失败 |

OOF MSE为`0.10857`，常数baseline为`0.11314`；即模型有4.04%的回归改进，但不足事前最低线。
Episode-start共有57正/96负，early共有16,301正transition/67,200负transition。这里不把AUROC
低于0.5写成“显著反向”，因为本Gate未预注册显著性检验；硬判决只来自三条固定阈值。

前两个执行目录均在任何actor update和科学report前被机械错误拦截：一处把SB3要求的tensor
bootstrap误传为numpy，另一处给尾部未完成episode的新起点查询OOF fold。两次都不产生科学
判决；修正只恢复冻结语义，最终有效目录使用新路径且没有覆盖旧产物。

最终verdict为`fail_stop_exact_constrained_implementation`。按§35.2，不运行30-update formal，
不评四图，不扫描budget、dual或cost critic。这严谨关闭的是
`reward去collision + P20 MLP cost critic + d=.10 + lambda0=1 + dual_lr=.5`的当前准入实例；
**不严谨否决Constrained PPO方法类**，因为高质量cost prediction并非所有policy-gradient
constraint算法的数学必要条件，Monte-Carlo cost仍可传梯度。本轮真正建立的是：当前cost链路
可执行、梯度非零，但现有P20 MLP不能按预注册要求跨起点提前估计稀疏collision cost，因此没有
足够证据支付正式训练预算。


---

# 第二部分（附录 A）：Round Z2 反事实动作存在性与可排序性 Gate 完整技术报告

以下为原 `COUNTERFACTUAL_ACTION_GATE_COMPLETE_REPORT.md` 的 §1--§23 原文，章节号加前缀 `A`。
原文件头部元信息：更新时间 2026-08-08；实验状态已完成、无 actor 更新、无正式训练；
机器判决 `fixed_local_action_library_not_admitted`；生产状态保持 U30 不变。

更新时间：2026-08-08（Asia/Singapore）  
实验状态：已完成；无 actor 更新；无正式训练  
机器判决：`fixed_local_action_library_not_admitted`  
生产状态：保持 U30，不变

## A1. 技术结论

Round Z2 得到一个清晰但是否定训练准入的结果：

1. **局部有效动作确实大量存在。** Early 和 late 两个窗口的 action-existence 六条门全部通过。Early oracle 在 inherited/created/lost 三层分别救回 `93/109、44/46、10/13`；late 为 `96/109、45/46、7/13`。
2. **当前 U44 frozen hidden 无法稳定选择这些动作。** 按 ego startpoint 分组的五折 OOF ActionScorer，early 只有 `19/109、11/46、1/13`，并在 225 个 safe controls 中新造 15 次 collision、丢失 17 次 overtake；late 为 `34/109、17/46、0/13`，虽然 controls 为合格的 `3/5`，但 target success 只有 51，低于 grouped fixed-action baseline 的 79。
3. **因此瓶颈不是“动作空间没有局部解”，而是“何时触发、对谁选什么动作、如何同时保持 progress”。** 反事实分支可以产生强 oracle 标签，但不会自动成为可部署策略。
4. **当前固定 12 动作、两个 50-step 窗口、1680D frozen-hidden scorer 与 first-action preference extraction 均不准入后续训练。** Prefix-reset 没有获得 early-only 前提；Residual/MoE 继续受 361D 输入和 12-key actor 兼容边界限制。
5. **Constrained PPO 没有被本 Gate 检验。** 它不能由本结果严格推出失败，也没有自动训练授权。

核心结果如下。所有“成功”均指候选最终为无 ego collision 的 `overtake`。

| 层 | 分母 | Early oracle | Late oracle | Early OOF head | Late OOF head |
|---|---:|---:|---:|---:|---:|
| Inherited collision | 109 | 93 | 96 | 19 | 34 |
| Created collision | 46 | 44 | 45 | 11 | 17 |
| Lost-overtake | 13 | 10 | 7 | **1** | **0** |
| Inherited-follow 诊断 | 63 | 18 | 18 | 5 | 3 |
| Safe control 新 collision / overtake loss | 225 | 不介入 | 不介入 | **15 / 17** | 3 / 5 |
| 三主层 target success | 168 | 147 | 148 | 31 | 51 |
| Grouped fixed-action baseline | 168 | 0（noop） | 79 | head 比 fixed 多 31 | **head 比 fixed 少 28** |

这是一项 Austin development、单 U44、零 actor-update 的机制诊断，不是四图正式模型成绩。

## A2. 本 Gate 回答什么问题

此前 BC Gate B 只能回答“BC teacher 在指定失败 prefix 上是否稳定优于 U44”，不能回答：

- 同一 prefix 上是否存在比 BC 更合适的局部动作；
- 当前 U44 hidden 是否包含足以选择这些动作的信息；
- 强反事实动作标签能否形成跨 startpoint 的状态条件规则；
- 更早窗口是否比更晚窗口更容易排序，从而支持 prefix-reset 的长 credit 假设。

Round Z2 将问题拆成两个顺序 Gate：

1. **Action existence：** 固定小型动作库中，是否有足够多动作能把失败闭环变成最终 overtake，同时覆盖足够多起点、raceline 和动作 family。
2. **Action rankability：** 只看分支起点的 frozen 1680D U44 hidden，固定小网络能否在未见 ego startpoint 上选出动作，并通过 collision、progress、control harm 和 fixed-action baseline 的全部门槛。

只要 existence 通过但 rankability 失败，就说明反事实提供了可达性上界，却没有提供可部署选择器。

## A3. 证据范围与指标定义

### A3.1 数据边界

- 地图：只使用 Austin。
- 数据角色：Gate A 已冻结的 development startpoints。
- 未使用：Gate A validation、Hockenheim、MoscowRaceway、Nuerburgring。
- Source actor：U44 deterministic actor。
- 训练：无 PPO、无 BC loss、无 auxiliary actor update。
- 模拟：8 秒、`timestep=0.01`、RK4、双车闭环。
- Development 总场景：718。
- 最终 Gate cohort：456。
- 属于五个分层但事件前不足 150 个 action step：112，固定剔除。
- 不属于五个预注册分层的其余场景：150。

### A3.2 Outcome 与计数定义

- Ego collision：`ego-opp` 或 `ego-wall`。
- Overtake：无 ego collision 且 episode 结束时 unwrapped relative progress 大于 0。
- Follow：无 ego collision且 unwrapped relative progress 不大于 0。
- Rescue：候选最终 outcome 必须为 `overtake`。Collision 只变成 follow 不算成功。
- 三主层 target success：inherited collision、created collision、lost-overtake 的成功数之和；固定分母 `109+46+13=168`。
- Inherited-follow：只报告机制覆盖，不参与主准入总数。
- Safe-control 新 collision：225 条 BC/U44 共同 overtake 中，被选择器改成 ego collision 的数量。
- Safe-control overtake loss：225 条 safe controls 中，被选择器改成非 overtake 的数量。
- Oracle：看完同一 prefix 的 12 个候选闭环后，事后选择满足规则的动作，只是可达性上界。
- OOF head：只用其他 ego startpoint folds 训练，在当前 fold 预测动作。
- Grouped fixed baseline：每折只从 train folds 选一个所有测试场景共用的动作，用来检验 hidden 是否真的增加状态条件价值。

### A3.3 Claim 类型

- “动作存在”是本固定动作库、固定窗口、Austin development 上的描述性机制结论。
- “OOF head 失败”是按冻结分组与固定阈值得到的预测性负结果。
- Paired exact 结果描述 future-event 条件下的闭环反事实差异。
- 本实验不建立部署 actor 的因果收益，也不证明连续动作空间或其他 representation 不可能成功。

## A4. 冻结输入与身份

`action_gate_plan.json` 在任何候选分支前冻结输入哈希、456 条任务、112 条排除任务、fold、prefix、动作库和准入线。

| 输入 | 路径 | SHA-256 |
|---|---|---|
| Development panel | `post-trained/panels/bc_safe_anchor_v1/development_scenarios.json` | `61ccc0652a0f7cfafa142f88202bf235da1d7a6f5c7c3a06c3bc6e9c8c1e200e` |
| BC results | `eval_results/front_corridor_temporal_bc_safe_anchor/gate_a/development/bc/results_multi.json` | `d99d10ad58fccadf075c3ec6fccdfdad7f94fb82d815ea8945e0c2864c55d4d5` |
| U44 results | `eval_results/front_corridor_temporal_bc_safe_anchor/gate_a/development/u44/results_multi.json` | `ca0efbdeef127eea2a86f1796235aa61fa80c62430a01f49c17b57154ce4c642` |
| U44 actor | `post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update44/actor.pth` | `fb0c9895eb2ff004e414da09e4ee27675e825f0e6413a095377d66838e411bf7` |

运行前 base commit 为：

```text
4af10fe75df56ff431db8918b9c52d480943a310
```

Runner 当时是未提交的新文件，因此 commit 本身不足以重建实验；必须同时匹配当前 runner SHA-256：

```text
scripts/run_counterfactual_action_gate.py
d7ed3fb1ff1a4720f614af5acbdc7083ad2516d7995ae887c4263e246e2526c1
```

## A5. Cohort 与五折设计

### A5.1 固定分层

| Stratum | BC/U44 定义 | 数量 | Gate 角色 |
|---|---|---:|---|
| `inherited_collision` | BC collision；U44 collision | 109 | 正式 safety target |
| `created_collision` | BC 非 collision；U44 collision | 46 | 正式 regression target |
| `lost_overtake` | BC overtake；U44 follow | 13 | 正式 progress target |
| `inherited_follow` | BC follow；U44 follow | 63 | 诊断层 |
| `safe_control` | BC overtake；U44 overtake | 225 | 正对照与 harm 门 |
| 合计 |  | **456** |  |

分类由 `classify_stratum()` 完成。不满足这些组合的 episode 返回 `None`，不进入 Gate。

### A5.2 Fold 规则

Fold 固定为：

```text
SHA256("counterfactual-action-fold-v1|" + ego_idx) mod 5
```

同一 `ego_idx` 不允许跨 fold。

| Fold | 总数 | Inherited collision | Created collision | Lost-overtake | Inherited-follow | Safe control |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 76 | 23 | 9 | 1 | 8 | 35 |
| 1 | 100 | 25 | 11 | 3 | 12 | 49 |
| 2 | 65 | 11 | 8 | 0 | 10 | 36 |
| 3 | 102 | 24 | 7 | 5 | 9 | 57 |
| 4 | 113 | 26 | 11 | 4 | 24 | 48 |

Lost-overtake 总共只有 13 条，且 fold 2 没有该层测试样本。这会增加单折方差，但 late OOF 在全部 13 条上恢复 0 条，不能把失败解释为接近通过。

## A6. 事件、prefix 与动作库

### A6.1 事件定位

`source_event()` 使用既有 U44 full trace：

- `inherited_collision`、`created_collision`：第一次 typed ego collision marker；
- `lost_overtake`、`inherited_follow`、`safe_control`：第一次全局最小 OBB clearance；
- 若 collision 层没有 collision marker，立即报错；
- `event_index >= 150` 才有资格进入正式 cohort。

### A6.2 Prefix

| Prefix | 起止区间 | 最长 action steps | 与事件的关系 |
|---|---|---:|---|
| Early | `[event-150, event-100)` | 50 | 更早 |
| Late | `[event-100, event-50)` | 50 | 更接近事件 |

每步 0.01 秒，因此完整窗口最长 0.5 秒。若 ego collision 提前发生，环境真实 terminal，干预步数可以少于 50。窗口结束后恢复 U44。

候选不是从序列化 simulator snapshot 起跑，而是从场景初始状态完整确定性重放到窗口。这保留 opponent planner、U44 GRU 和车辆物理状态的真实闭环演化，代价是每个候选都必须完整运行一遍 episode。

### A6.3 动作语义

每个 residual 都相对候选自身当前 trajectory 上、当前时刻的 U44 deterministic mean：

```text
selected_raw_steer = u44_raw_steer + steering_delta
selected_raw_speed = u44_raw_speed + speed_delta
selected_executed_steer = clip(selected_raw_steer, -0.52, 0.52)
selected_executed_speed = selected_raw_speed
```

动作不是 source trajectory 的绝对动作复制。窗口外：

```text
selected_raw = u44_raw
```

### A6.4 固定动作库

No-op 之外有 12 个候选：

| 动作名 | Family | Steering delta (rad) | Speed delta (m/s) |
|---|---|---:|---:|
| `steer_m0p04` | steering | -0.04 | 0 |
| `steer_m0p02` | steering | -0.02 | 0 |
| `steer_p0p02` | steering | +0.02 | 0 |
| `steer_p0p04` | steering | +0.04 | 0 |
| `speed_m1p0` | speed | 0 | -1.0 |
| `speed_m0p5` | speed | 0 | -0.5 |
| `speed_p0p5` | speed | 0 | +0.5 |
| `speed_p1p0` | speed | 0 | +1.0 |
| `steer_m0p02_speed_m0p5` | coordinated | -0.02 | -0.5 |
| `steer_p0p02_speed_m0p5` | coordinated | +0.02 | -0.5 |
| `steer_m0p02_speed_p0p5` | coordinated | -0.02 | +0.5 |
| `steer_p0p02_speed_p0p5` | coordinated | +0.02 | +0.5 |

候选总数：

```text
456 episodes * 2 prefixes * 12 actions = 10,944 branches
```

## A7. 预注册准入线

### A7.1 Action-existence 六门

Early 与 late 各自独立判决，必须同时满足：

1. Inherited collision rescue `>=22/109`；
2. Created collision rescue `>=12/46`；
3. Lost-overtake restore `>=7/13`；
4. Collision rescue 覆盖至少 10 个唯一 ego startpoint；
5. Collision rescue 覆盖 `raceline0/1/2`；
6. Oracle labels 覆盖至少两个动作 family。

### A7.2 Rankability 六门

只有某 prefix 的 existence 六门全过，才训练该 prefix 的离线 scorer。OOF head 必须同时满足：

1. Inherited rescue `>=11/109`；
2. Created rescue `>=7/46`；
3. Lost-overtake restore `>=4/13`；
4. 225 controls 中新 collision `<=4`；
5. 225 controls 中 overtake loss `<=11`；
6. 三主层 target success 至少比 grouped fixed-action baseline 多 9。

任一项失败即为该 prefix rankability 失败。不能用净 collision 改善替代 progress 或 control 独立门。

## A8. 代码逻辑与调用链

Runner：`scripts/run_counterfactual_action_gate.py`，当前 906 行。

### A8.1 总调用链

```text
parse_arguments
  -> build_plan / verify_plan
  -> forkserver Pool(worker_initializer)
  -> run_tasks(make_branch0_tasks)
       -> evaluate_branch(noop)
       -> compare_replay
       -> 保存 early/late 1680D hidden
  -> branch0 全部通过
  -> run_tasks(make_candidate_tasks)
       -> evaluate_branch(456 * 2 * 12)
  -> validate_results
  -> existence_metrics(early/late)
  -> train_ranker(early/late，仅 existence 通过时)
  -> analyze
  -> atomic_write_json(action_gate_report.json)
```

### A8.2 函数级职责

| 起始行 | 函数/类 | 主要职责 | Fail-closed 行为 |
|---:|---|---|---|
| 77 | `parse_arguments` | 声明 CLI | required paths；主入口锁 hidden/sim/CUDA |
| 92 | `scenario_key` | 统一 episode identity | 使用共享 `episode_key()` |
| 96 | `classify_stratum` | 固定五层 outcome 分类 | 未定义组合返回 `None` |
| 109 | `source_event` | 定位 collision/min-clearance 事件 | Collision 层无 marker 报错 |
| 130 | `fold_for_startpoint` | 按 ego_idx 确定五折 | 确定性 SHA-256 |
| 135 | `build_plan` | 核对输入 key，冻结全部合同 | Counts 必须精确为 `109/46/13/63/225` |
| 239 | `verify_plan` | 续跑前复核 plan | Hash、456 keys、动作或 prefix 变化即停 |
| 255 | `worker_initializer` | 限线程；每 worker 加载一次 U44 | 无 CUDA/state_dict 不兼容即停 |
| 269 | `append_full_row` | 写 no-op 内存 full trace | 由 `compare_replay` 检验 |
| 288 | `append_compact_row` | 写持久 compact trace 行 | 由 `validate_results` 检验 |
| 302 | `compare_replay` | Full no-op 与 source 逐字段比较 | 动作 0 误差；其他数值 `<=1e-6`；bool 严格 |
| 321 | `evaluate_branch` | 完整闭环、干预、outcome、trace/hidden | Finite、hidden、terminal 异常报错 |
| 475 | `run_tasks` | Partial resume、并行、原子结果 | Completed key 集必须精确匹配 plan |
| 506 | `make_branch0_tasks` | 展开 456 个 no-op | 路径与 identity 固定 |
| 525 | `make_candidate_tasks` | 展开 10,944 个候选 | `episode::prefix::action` 唯一 key |
| 544 | `validate_results` | 重放、schema、finite、terminal、window、action 全检 | 任一失败不产生科学 verdict |
| 613 | `action_norm` | 归一化动作平方范数 | 固定 scale：0.02 rad、0.5 m/s |
| 617 | `preferred_candidate` | 生成唯一 oracle label | 非 overtake 不成功；control 固定 noop |
| 629 | `existence_metrics` | 计算 existence 六门与覆盖 | 六门全过才 pass |
| 661 | `ActionScorer` | Hidden/action 打分网络 | 固定架构 |
| 674 | `candidate_action_tensor` | 构造 13 个归一化动作 | Noop 加 12 候选 |
| 681 | `selected_outcome` | 读取某选择器对应闭环 outcome | Noop/candidate 明确分流 |
| 687 | `paired_exact_p` | 双侧 exact binomial p | 无 discordant 返回 1 |
| 696 | `train_ranker` | 五折 OOF head 与 grouped fixed baseline | 空 fold/预测不完整报错；六门全过才 pass |
| 827 | `analyze` | 合并质量、existence、rankability 与路线判决 | 固定分流 |
| 868 | `__main__` | 执行顺序、覆盖保护、原子写最终报告 | 已有最终 report 拒绝覆盖；no-op 未全过不跑候选 |

### A8.3 `evaluate_branch()` 闭环语义

每个任务执行以下步骤：

1. 固定 NumPy seed 42。
2. 用 Austin、双车、0.01 秒 timestep、RK4 创建环境。
3. Ego 使用 `raceline1`；opponent 使用场景冻结的 raceline 和 speedscale。
4. U44 GRU hidden 从 0 初始化。
5. 每步将 ego 360D LiDAR 与上一时刻实际速度输入 U44。
6. `hidden_scale=4` 对应 1680D hidden。
7. No-op 在 prefix 起点处理完当前 observation 后保存 hidden，因此 scorer 看到的是作出该步动作时的内部状态。
8. Opponent planner 每 10 个环境 step 重规划一次 trajectory；tracker 每步输出动作。
9. 若处于干预窗口，给当前 U44 mean 添加 residual；否则执行 U44 mean。
10. 每步同时记录 `u44_raw_action`、`ego_raw_action`、`ego_executed_action`，避免把请求动作与真实执行动作混淆。
11. 环境 step 后更新基于 raceline1 投影的 unwrapped relative progress。
12. 第一次 ego collision typed 为 `ego-opp` 或 `ego-wall` 并强制结束。
13. 无 collision 时，以最终 unwrapped relative progress 的正负判定 overtake/follow。
14. 终端额外写一行零动作，固定 `terminal_post_step=true`、`action_applied=false`、`intervention_active=false`。

### A8.4 No-op 精确重放

No-op 会在内存中保留完整数组并与既有 U44 source trace 比较。

严格 0 误差字段：

- `ego_raw_action`
- `ego_executed_action`

允许 `atol=1e-6, rtol=0` 的数值字段：

- `opp_executed_action`
- `ego_pose`
- `opp_pose`
- `ego_measured_speed_mps`
- `opp_measured_speed_mps`
- `ego_lidar_360`
- `opp_lidar_360`

严格 shape/boolean 相等字段：

- `collisions`
- `ego_opp_collision`
- `ego_wall_collision`
- `opp_wall_collision`
- `action_applied`
- `terminal_post_step`

此外 outcome 必须与 plan 中的 U44 source outcome 相等。任一 no-op 失败都不会调度候选。

### A8.5 Compact trace schema

所有 11,400 条持久 NPZ 都只包含 numeric arrays，禁止 pickle。

| 字段 | Dtype | 语义 |
|---|---|---|
| `time_s` | float64 | 当前 trace 行时间 |
| `ego_raw_action` | float32 | 选择器请求动作，shape `[N,2]` |
| `ego_executed_action` | float32 | Clip 后实际动作，shape `[N,2]` |
| `u44_raw_action` | float32 | 同一当前状态的 U44 mean |
| `collisions` | bool | 环境 collision array |
| `ego_opp_collision` | bool | Typed ego-opponent marker |
| `ego_wall_collision` | bool | Typed ego-wall marker |
| `opp_wall_collision` | bool | Typed opponent-wall marker |
| `action_applied` | bool | 该行是否实际送动作 |
| `terminal_post_step` | bool | 是否为额外终端行 |
| `intervention_active` | bool | 是否在 residual 窗口内 |

No-op 另保存两个 hidden arrays：

```text
early: float32[1680]
late:  float32[1680]
```

### A8.6 Result record schema

每个 no-op/candidate result 包含：

- `result_key`、`episode_key`、`stratum`
- `prefix_name`、`candidate_name`、`candidate_family`
- `outcome`
- `ego_collision_time_s`、`ego_collision_step`
- `simulation_time_s`、`steps`、`intervention_steps`
- `observation_finite`、`action_finite`
- `avg_speed`、`speed_variance`、`total_distance`
- `final_relative_position_m`
- proximity 质量字段：`global_min_surface_dist`、`danger_sectors`、`proximity_below_threshold_timesteps`
- No-op 专属：`replay_pass`、`replay_errors`

最终 `action_gate_report.json` 还保存：

- 每个 prefix 的 oracle rescue episode keys；
- 每个 prefix 的 oracle label 计数；
- OOF head 和 fixed baseline 的逐 episode selected action/outcome；
- fold 级 train/test 数和 fixed action；
- 六条准入门；
- paired exact 计数与 p 值；
- route decision。

因此本 Markdown 不复制数千行 identity，但 machine-readable identity 没有被聚合表替代。

### A8.7 `validate_results()` 的逐 trace 硬门

每条 trace 都必须满足：

1. Result key 集与 plan 精确一致；
2. NPZ 字段集精确等于 compact schema；
3. 所有数组长度相同；
4. `length == steps + 1`；
5. 所有数组 finite；
6. 只有一行 terminal，且必须是最后一行；
7. 只有最后一行 `action_applied=false`；
8. 终端行不能处于 intervention；
9. `intervention_active` 精确等于冻结 prefix 与实际终止共同决定的 mask；
10. 候选 active 行的 raw action 必须满足 residual，`atol=1e-7, rtol=0`；
11. Executed steering 必须是 raw steering clip 后结果；
12. 终端 executed action 必须为 0；
13. Executed action 数组必须精确相等。

## A9. Oracle label 与 ActionScorer

### A9.1 Oracle label

Safe controls 按合同固定为 `noop`。其他层遍历 12 个候选，只有最终 outcome 为 `overtake` 才进入成功集合。

多解按以下顺序唯一化：

1. 最小归一化动作平方范数：

   ```text
   (steering_delta / 0.02)^2 + (speed_delta / 0.5)^2
   ```

2. 更高 final relative progress；
3. 动作名字典序。

没有成功候选则 label 为 `noop`。

### A9.2 ActionScorer 架构

输入：

- Hidden：1680D U44 hidden；
- Actions：13 个二维动作，包括 noop；
- 动作编码：`[steering/0.02, speed/0.5]`。

网络：

```text
hidden 1680
  -> Linear(1680,128)
  -> ReLU
  -> 与每个二维动作拼接为 130D
  -> Linear(130,64)
  -> ReLU
  -> Linear(64,1)
  -> 13 个 action logits
```

训练固定：

| 参数 | 值 |
|---|---:|
| Loss | 13 类 cross-entropy |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Batch size | 64 |
| Epochs | 100 |
| Folds | 5 |
| Model seed | `4200 + fold` |
| Shuffle seed | `4300 + fold` |

每折只用 train folds 计算 hidden mean/std；`std < 1e-6` 的维度设为 1。没有 validation 选择、早停、超参扫描或结果后调参。

### A9.3 Grouped fixed-action baseline

每个 fold 在其 train folds 上遍历 noop 和 12 个动作，以以下固定分数选择一个动作：

```text
target overtake successes - 5 * safe-control harms
```

其中 safe-control harm 是 action 后不再 overtake 的 control 数。相同分数按动作索引确定。

这个 baseline 同样依赖 future-event prefix，不能部署；它的用途只是回答：看 hidden 的 head 是否优于一个不看状态、统一施加的动作。

### A9.4 Paired exact 统计

`paired_exact_p(left_only, right_only)`：

```text
discordant = left_only + right_only
p = 2 * Binomial(n=discordant, p=0.5) 较小侧累计概率
```

上限截断为 1；无 discordant pair 时返回 1。报告双侧 exact p，不把它解释为部署因果效果。

## A10. 执行与效率优化

### A10.1 执行顺序

1. `--prepare-only` 扫描 718 条 source trace，冻结 456 条正式 task。
2. 运行 1 条 no-op 与 1 条 candidate 烟测。
3. 烟测结果：no-op 最大误差 0；candidate 准确干预 50 步。
4. 正式 no-op 先运行，456/456 全部通过。
5. No-op 通过后才展开 10,944 个 candidate branches。
6. 完成后验证全部 trace，再计算 existence 与 rankability。
7. 最终 report 原子写入。

### A10.2 并行与恢复

- Multiprocessing context：`forkserver`。
- 每个 persistent worker 只加载一次 U44 到 CUDA。
- `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`NUMEXPR_NUM_THREADS` 均设为 1。
- PyTorch CPU thread 设为 1。
- 使用 `imap_unordered`，只改变完成顺序；最终 JSON 按 result key 排序。
- 每个完成记录立即追加并 flush 到 partial JSONL。
- 重启时按 `result_key` 跳过已完成任务。
- 全部完成后原子写 result JSON，并删除 partial。
- Orphan trace 会由同一确定性 task 覆盖。

首次正式运行使用 12 workers。在已完成 94 条 no-op 后安全中断，加入重复 RK4 warning 的窄范围抑制并验证恢复。Candidate 阶段完成 651 条后再次安全暂停；确认机器 20 个 CPU 核、GPU 占用约 9.4 GB、利用率接近 99% 后改为 20 个 persistent workers 续跑。

这一优化只改变执行层：已完成 key 不重算，动作、seed、fold、任务、窗口、阈值、模型和判决逻辑全部不变。

### A10.3 I/O 设计

No-op 在内存保留 full LiDAR/pose/action/speed arrays 做逐字段比较，但只落盘：

- Compact trace；
- Early/late hidden；
- Result 与 replay error maxima。

Candidate 只保存 compact trace，避免为 10,944 条分支重复写双 360D LiDAR 与两车 pose。

当前输出目录约 268 MB，包含：

| 类型 | 数量 |
|---|---:|
| No-op compact traces | 456 |
| Hidden NPZ | 456 |
| Candidate compact traces | 10,944 |
| JSON | 4 |
| 总文件数 | 11,860 |

当前没有 Round Z2 进程，没有 partial 文件。

## A11. 数据与实现质量结果

| 检查 | 结果 | 判决 |
|---|---:|---|
| No-op episodes | 456 / 456 | PASS |
| Candidate episodes | 10,944 / 10,944 | PASS |
| Compact traces | 11,400 | PASS |
| Trace rows | 8,185,913 | PASS |
| No-op 全字段最大绝对误差 | 0.0 | PASS |
| Result / plan / trace key sets | 完全相等 | PASS |
| 数组 finite 与长度对齐 | 全部通过 | PASS |
| Terminal post-step 合同 | 全部通过 | PASS |
| Intervention window | 全部通过 | PASS |
| Raw residual | 全部通过 | PASS |
| Executed clip | 全部通过 | PASS |
| Partial 文件 | 0 | PASS |

实际 no-op 最大误差全部为 0，比非动作字段允许的 `1e-6` 门更严格。因此后续科学失败不是重放不一致、trace 错位或候选动作未正确执行造成的。

## A12. Action-existence 结果

### A12.1 六门逐项结果

| 指标 | 准入线 | Early | Late | 判决 |
|---|---:|---:|---:|---|
| Inherited collision rescue | `>=22/109` | 93/109 | 96/109 | 两侧 PASS |
| Created collision rescue | `>=12/46` | 44/46 | 45/46 | 两侧 PASS |
| Lost-overtake restore | `>=7/13` | 10/13 | 7/13 | 两侧 PASS |
| Collision unique startpoints | `>=10` | 48 | 48 | 两侧 PASS |
| Collision opponent racelines | 0/1/2 全覆盖 | 0/1/2 | 0/1/2 | 两侧 PASS |
| Oracle action families | `>=2` | 3 | 3 | 两侧 PASS |
| Inherited-follow overtake（诊断） | 无门 | 18/63 | 18/63 | 仅报告 |

Collision 两层合计：

- Early：`93+44=137/155`；
- Late：`96+45=141/155`。

这足以否定“当前动作接口在关键窗口没有局部可达动作”的解释。

### A12.2 Oracle label 分布

| 动作 | Family | Early | Late |
|---|---|---:|---:|
| `noop` | baseline | 291 | 290 |
| `steer_m0p04` | steering | 0 | 1 |
| `steer_m0p02` | steering | 19 | 11 |
| `steer_p0p02` | steering | 22 | 25 |
| `steer_p0p04` | steering | 5 | 1 |
| `speed_m1p0` | speed | 7 | 6 |
| `speed_m0p5` | speed | 25 | 21 |
| `speed_p0p5` | speed | 54 | 71 |
| `speed_p1p0` | speed | 13 | 9 |
| `steer_m0p02_speed_m0p5` | coordinated | 3 | 4 |
| `steer_p0p02_speed_m0p5` | coordinated | 6 | 3 |
| `steer_m0p02_speed_p0p5` | coordinated | 3 | 10 |
| `steer_p0p02_speed_p0p5` | coordinated | 8 | 4 |
| 合计 |  | **456** | **456** |

Noop 包含 225 条按合同固定 noop 的 safe controls，也包含无候选成功的 target/diagnostic 场景。`speed_p0p5` 是最常见的非 noop label，但三类动作 family 都有真实成功样本，因此 oracle 并非只依赖一个动作。

## A13. Rankability 结果

### A13.1 Head 与 fixed baseline

| 选择器 | Inherited | Created | Lost | Follow 诊断 | Target total | Control 新 collision | Control loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Early OOF head | 19 | 11 | 1 | 5 | 31 | 15 | 17 |
| Early grouped fixed（noop） | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Late OOF head | 34 | 17 | 0 | 3 | 51 | 3 | 5 |
| Late fixed（`steer +0.02, speed +0.5`） | 45 | 33 | 1 | 4 | 79 | 5 | 5 |

Early 的五个 train folds 都选 noop 作为 fixed baseline。Late 的五个 train folds 都选：

```text
steer_p0p02_speed_p0p5
steering delta = +0.02 rad
speed delta = +0.5 m/s
```

### A13.2 六门逐项判决

| 准入项 | 阈值 | Early 值 | Early | Late 值 | Late |
|---|---:|---:|---|---:|---|
| Inherited rescue | `>=11/109` | 19 | PASS | 34 | PASS |
| Created rescue | `>=7/46` | 11 | PASS | 17 | PASS |
| Lost-overtake restore | `>=4/13` | 1 | **FAIL** | 0 | **FAIL** |
| Safe-control 新 collision | `<=4/225` | 15 | **FAIL** | 3 | PASS |
| Safe-control overtake loss | `<=11/225` | 17 | **FAIL** | 5 | PASS |
| Target success margin over fixed | `>=+9` | +31 | PASS | -28 | **FAIL** |

Early 失败三门：lost-overtake、control collision、control overtake loss。  
Late 失败两门：lost-overtake、state-conditioned margin over fixed。

### A13.3 OOF head 选择动作分布

| 动作 | Early head | Late head |
|---|---:|---:|
| `noop` | 220 | 216 |
| `steer_m0p04` | 23 | 6 |
| `steer_m0p02` | 10 | 12 |
| `steer_p0p02` | 17 | 22 |
| `steer_p0p04` | 15 | 12 |
| `speed_m1p0` | 20 | 42 |
| `speed_m0p5` | 16 | 17 |
| `speed_p0p5` | 52 | 37 |
| `speed_p1p0` | 42 | 43 |
| `steer_m0p02_speed_m0p5` | 13 | 5 |
| `steer_p0p02_speed_m0p5` | 10 | 17 |
| `steer_m0p02_speed_p0p5` | 7 | 11 |
| `steer_p0p02_speed_p0p5` | 11 | 16 |
| 合计 | **456** | **456** |

Head 并没有塌缩到单一动作，但这种多样性没有转化为更高 utility；late 反而显著落后于单一 fixed action。

### A13.4 相对 no-op 的 paired exact 结果

| 选择器 | Outcome 轴 | Removed / lost | Created / gained | 双侧 exact p |
|---|---|---:|---:|---:|
| Early OOF head | Collision | 41 removed | 19 created | `0.0062176027` |
| Early OOF head | Overtake | 17 lost | 36 gained | `0.0126603358` |
| Late OOF head | Collision | 56 removed | 7 created | `1.3636714e-10` |
| Late OOF head | Overtake | 5 lost | 54 gained | `1.9067359e-11` |
| Late grouped fixed | Collision | 79 removed | 17 created | `9.9415135e-11` |
| Late grouped fixed | Overtake | 5 lost | 83 gained | `2.6897131e-19` |

这些结果证明：在 future-event 已知的窗口内，head 和 fixed action 相对 no-op 都有净改善。它们不证明部署 actor 能在线找到同一窗口。

## A14. 为什么 Gate 失败

### A14.1 Future-event trigger 不可部署

所有候选收益都发生在由未来 collision 或未来全局最小 clearance 定位的窗口。部署 actor 在当前时刻没有该未来事件标签。

因此即使 late fixed action 很强，也仍缺少：

- 何时启动；
- 对哪些状态启动；
- 何时停止；
- 如何避免在 safe controls 上误触发。

### A14.2 Frozen hidden 不能稳定排序

Oracle 与 head 的差距很大：

| 层 | Early oracle rate | Early head rate | Late oracle rate | Late head rate |
|---|---:|---:|---:|---:|
| Inherited collision | 85.3% | 17.4% | 88.1% | 31.2% |
| Created collision | 95.7% | 23.9% | 97.8% | 37.0% |
| Lost-overtake | 76.9% | 7.7% | 53.8% | 0.0% |

可能原因包括：

1. 当前 U44 hidden 没有编码可部署的 interaction phase；
2. 同一 hidden 附近可能有多个等价动作，单标签 cross-entropy 制造标签歧义；
3. 13 类分类 loss 与最终 safety/progress utility 不一致；
4. Startpoint 外推需要当前 actor history 没有的状态变量；
5. Collision 与 lost-overtake 需要不同甚至相反的动作策略。

本 Gate 不能在这五个解释之间唯一归因，因此不允许直接加宽 head 或改 loss 后重试。

### A14.3 Safety 与 progress 没有统一选择规则

Early/late head 在 collision 两层都有机制收益，但 lost-overtake 分别只有 `1/13` 和 `0/13`。这表明只提供 collision-oriented 反事实动作仍可能沿安全—超车前沿移动，而不是突破该前沿。

### A14.4 反事实信息不是自动解决方案

Counterfactual branch 能回答：

> 如果在一个已经由未来事件定位的窗口持续施加某 residual，最终 outcome 会怎样？

它不能自动回答：

> 部署 actor 如何只根据当前可见 history，跨地图、跨起点稳定选择动作并承担后续 covariate shift？

因此“提供反事实动作信息”是必要的诊断工具，但不是充分的训练方案。

### A14.5 不应首先扩大动作库

现有库已经：

- 在 collision 层取得 `137/155` 和 `141/155` oracle rescue；
- 覆盖 48 个 startpoint；
- 覆盖三条 opponent raceline；
- 覆盖三类动作 family；
- 产生一个比 late head 更强的 fixed action。

这反驳“主要缺口是局部动作库覆盖不足”。扩幅度、加 horizon 或增加动作 family 会改变问题，但不会直接解决触发与选择。

## A15. 限制与不可外推项

1. 只检验单一 U44、Austin development、固定 12 个 piecewise-constant residual 和两个 0.5 秒窗口。
2. 不证明连续动作空间、不同 horizon、不同 recurrent representation 或正式 PPO 理论上不可能成功。
3. Cohort 来自 difficult development，不是 Austin 自然分布，不能估计自然出现频率。
4. 不包含三张测试地图，因此不能声称跨地图泛化。
5. Oracle tie-break 偏向更小动作，再偏向 final relative progress；换标签定义可能改变类别分布，但不能结果后修改当前 Gate。
6. Safe controls 的 oracle label 固定为 noop；existence 本身不测试对 controls 施加动作。该风险由 OOF/fixed control harm 门承担。
7. Lost-overtake 只有 13 条，fold 2 为 0 条，估计方差较高；但 head 的 `1/13、0/13` 离 `4/13` 门不近。
8. Full no-op LiDAR/pose arrays 只在运行时比较，没有重复落盘；保留 source full trace、compact replay trace、逐字段 maxima 和严格 verdict。
9. Runner 是未提交工作树文件，复现需要 script hash，而不能只看 base commit。
10. 本 Gate 只证明 frozen hidden 对当前 label 的固定 scorer 未准入；没有直接证明 shared actor mapping 本身一定不足。

## A16. 代码审计备注

当前实现中有两个无语义影响的重复语句：

1. `worker_initializer()` 连续两次调用 `torch.set_num_threads(1)`；
2. `preferred_candidate()` 连续两次执行 `successful = []`。

它们不会改变任何结果，也不构成重跑理由。后续若整理代码，可以做最小删除。

另一个需要显式记录的 provenance 细节：最终 `action_gate_report.json` 为补充逐 episode selection 映射而确定性重算过 head，`execution.command` 因此记录为：

```text
/home/haowei/miniconda3/envs/end2race/bin/python -
```

该字段不是原始正式启动命令。聚合结果与首次运行逐项一致；真实复现必须使用 §18 的完整 CLI、冻结输入 hash 与 runner hash。

## A17. 六类方法的当前状态

“六类”包含实际 Gate 失败、机制形式失败和架构边界关闭，不等于六个 formal training 都已运行。

| 方法类别 | 当前证据 | 状态 | 准确边界 |
|---|---|---|---|
| Hindsight-selected counterfactual BC functional regularization | Gate A 通过；Gate B 的 lost-overtake 恢复与 controls 失败 | **关闭当前形式** | 不进入 Gate C/D 或 formal training；不是纯 PPO |
| Action-conditioned controllability auxiliary head | Round Z2 动作存在，但 frozen-hidden fixed-library OOF 排序失败 | **关闭 tested 形式** | 不否决不同 representation、连续动作或新 trigger 的理论可能 |
| Counterfactual first-action preference | Oracle 强，当前 extraction 未过 progress/control/fixed 门 | **关闭 tested 形式** | 不能把 oracle label 直接写成 imitation/preference loss |
| Prefix-reset interaction curriculum | Early/late existence 都强，early/late rankability 都失败 | **不准入** | 没有 early-only 的长 credit 证据 |
| Interaction-phase residual / MoE | 会改变 shared actor 与 12-key checkpoint 合同 | **按兼容边界关闭** | 不是 Round Z2 的经验失败；放弃兼容后需重新预注册 |
| Lagrangian Constrained PPO | Round Z2 不使用 cost critic | **未测试、未授权** | 先唯一化 reward/cost 语义并做 cost-critic 预检 |

因此，当前可以说五类已关闭或不准入；不能说 Constrained PPO 已被严格证明无效。

## A18. 完整复现命令

项目解释器固定为：

```text
/home/haowei/miniconda3/envs/end2race/bin/python
```

### A18.1 冻结计划

```bash
/home/haowei/miniconda3/envs/end2race/bin/python scripts/run_counterfactual_action_gate.py \
  --development-panel post-trained/panels/bc_safe_anchor_v1/development_scenarios.json \
  --bc-results eval_results/front_corridor_temporal_bc_safe_anchor/gate_a/development/bc/results_multi.json \
  --u44-results eval_results/front_corridor_temporal_bc_safe_anchor/gate_a/development/u44/results_multi.json \
  --u44-trace-root eval_results/front_corridor_temporal_bc_safe_anchor/gate_a/development/u44/traces \
  --u44-model-path post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update44/actor.pth \
  --output-dir eval_results/counterfactual_first_action_preference/action_gate \
  --workers 20 \
  --hidden-scale 4 \
  --sim-duration 8.0 \
  --prepare-only
```

### A18.2 正式运行

使用完全相同命令，去掉 `--prepare-only`。

Runner 拒绝覆盖已有最终 report。若需从头独立复现，必须使用新的 output directory；不要写入当前已完成目录。

### A18.3 最小静态验证

```bash
/home/haowei/miniconda3/envs/end2race/bin/python -m py_compile scripts/run_counterfactual_action_gate.py
```

### A18.4 结果身份验证

```bash
sha256sum \
  scripts/run_counterfactual_action_gate.py \
  eval_results/counterfactual_first_action_preference/action_gate/action_gate_plan.json \
  eval_results/counterfactual_first_action_preference/action_gate/action_gate_report.json \
  eval_results/counterfactual_first_action_preference/action_gate/branch0_results.json \
  eval_results/counterfactual_first_action_preference/action_gate/candidate_results.json
```

## A19. 输出与哈希

主目录：

```text
eval_results/counterfactual_first_action_preference/action_gate/
```

| 文件 | 用途 | SHA-256 |
|---|---|---|
| `action_gate_plan.json` | 冻结输入、任务、排除、fold、prefix、动作和阈值 | `0ae070736fc290763f3ca8bc884be4ade1325965c728647def7474bfe4ad3896` |
| `action_gate_report.json` | 质量、existence、rankability、paired、route decision | `4d6c97fa0c705838cd936a4285dffef8455e8cb1c02607aa4add6939be47b0ae` |
| `branch0_results.json` | 456 条 no-op 结果 | `905c16d987361f30a6093fa032b6314c299f2f593ef53666123b092f534b984f` |
| `candidate_results.json` | 10,944 条候选结果 | `875fc96dfa97f5e4b6f8468f53bd67172e042a16f5ea49a54aa48e5855c4d92d` |

子目录：

```text
branch0/traces/       456 compact NPZ
branch0/hidden/       456 hidden NPZ，每个含 early/late
candidate_traces/
  early/<12 actions>/ 5,472 compact NPZ
  late/<12 actions>/  5,472 compact NPZ
```

## A20. 当前停止规则

1. 不改当前 12 动作幅度、family、两个 prefix、head 宽度、fold、label 或阈值后重跑。
2. 不只报告 oracle 上界绕过 rankability。
3. 不依据当前 label 启动 action-conditioned auxiliary loss 或 first-action preference imitation loss。
4. Prefix-reset 只有在出现“early 有可排序动作、late 没有”，或 fresh rollout 直接证明长 credit/GAE 错配时才重开。本轮不满足。
5. Residual/MoE 只有用户明确放弃 unchanged 361D input / 12-key actor compatibility，并先证明 shared mapping 不足时才重开。
6. Constrained PPO 不继承本 Gate 的失败；若继续，必须先唯一化安全 reward 与 collision cost 的关系，完成 cost-critic 可学习性/校准预检，再单独预注册。
7. 未获得新的用户选择与预注册前，不启动 formal training。
8. 完全合理的终点是关闭这条研究线并保留 production U30。

## A21. 仍可能改变下一步决策的问题

1. 能否从 actor 在线可见 history 构造不依赖未来事件的 interaction-phase trigger，并先通过 startpoint-OOF 检测 Gate？
2. Late fixed `steer +0.02 / speed +0.5` 的强结果主要来自哪一种几何阶段，是否只是 future-event 条件化造成的 selection effect？
3. Frozen hidden 排序失败究竟来自 representation 缺信息、oracle 多标签歧义，还是分类 loss 与 utility 不一致？
4. 若选择 Constrained PPO，应该删除 collision reward、只约束 collision cost，还是保留现有 reward 安全项并约束 progress/overtake？这两个定义不是同一个实验。
5. 论文是否要把本结果作为“局部可达性强、policy extraction 失败”的负机制证据，还是只作为内部筛选结果保留？

## A22. 最终判断

Round Z2 的价值不在于产生了新 actor，而在于把原本宽泛的“PPO 学不会合适动作”收窄为三个更具体的问题：

```text
局部动作可达性：强
在线阶段触发：未解决
跨 startpoint 动作排序：当前 frozen hidden 失败
安全与 progress 的统一选择：未解决
```

因此，继续扫描 collision penalty、动作幅度或 head 宽度都没有充分依据。当前固定动作信息路线应关闭；若还要支付一次新研究预算，只能在明确的新假设下选择 Constrained PPO，或先取得可部署 phase trigger / actor-visible representation 的新证据。否则应停止并保留 U30。

## A23. 与现有权威文档的对应关系

- 当前状态与停止规则：`.agents/HANDOFF.md` §9.17、§10。
- 科学分析：`.agents/ANALYSIS.md` §36。
- Runner 实现合同：`.agents/EXPERIMENTS.md` §13。
- 合并预注册与 Round Z2 冻结门：本文件第一部分 §23。

本文件是对上述权威内容和 machine-readable artifacts 的独立完整说明，不替代 `.agents/HANDOFF.md` 的当前连续性权威。


---

# 第三部分（附录 B）：原报告的追加节（§24--§29 原文）

以下为原报告在 Round Z2 主体之后陆续追加的各轮小结，章节号加前缀 `B`。它们与第一部分
`§25`--`§35` 描述同一批实验，但措辞与侧重不同，两者一并保留以免丢失任何已记录判断。


## B24. 2026-08-08追加：Z4/Z5对§17 frozen-hidden结论的校正与追认

Round Z4-A改用每个scenario-action真实终局的三分类监督和
`P(overtake)-5*P(collision)`，同一U44 late hidden得到target 104，高于fixed 79；但它只选择
7/456次noop，safe-control harm恶化为`13 collision / 20 overtake loss`。因此§17不能再只用
`51 < 79`外推“frozen hidden本身没有条件动作信息”：Z2证明的是13-class top-1 preference
extraction失败，Z4证明不同outcome extraction可在激进操作点提取更多救援信息。

Round Z5随后在ego-startpoint 5 outer x 4 inner nested CV中，只用inner-OOF选择lambda与noop
margin，并把两项control harm率限制为不高于fixed的`5/225`。结果：

| 合同 | target | control collision / overtake loss |
|---|---:|---:|
| fixed | 79 | `5 / 5` |
| frozen outcome，independent outer seeds | 69 | `1 / 3` |
| frozen outcome，exact Z4 outer seeds | 66 | `3 / 4` |

独立seed概率若看完全部outer outcome后事后选一个全局点，可读到`84 @ 3/5`；无泄漏nested
只有69。exact Z4 seeds的事后matched-harm最高也仅`72 @ 4/5`。所以Z4的104来自更激进的
操作点，不能作为同harm下优于fixed的证据。

两次frozen相对fixed的target配对为`28/38, p=0.268`与`29/42, p=0.154`，因此科学表述是
“在当前样本下未检出严格优于fixed”，不是“证明frozen显著更差”。Outer实际只使用`1/3`和
`3/4` harm，而fixed使用`5/5`；这是budget-constrained comparison，不是恰好用满budget的
matched比较。事后把outer操作点推到5/5会读取outer outcome并泄漏。两套事后frontier相差12
说明seed敏感性，但只有两个seed，不能据此估计稳定方差。

当前准确状态是：

1. Z2的具体top-1 preference extraction保持失败；
2. Z4/Z5的frozen-hidden三分类outcome selector在预注册harm预算下未建立target增益；
3. 这追认§17的“关闭tested形式”，但理由升级为无泄漏budget-constrained证据，不是把单一`51<79`
   当作frozen representation的信息上限；
4. 50步history在nested复核也只有`63 @ 6/8`与`55 @ 7/7`，没有增量；
5. future-event窗口仍不可部署：低noop率只是在已经由未来event定位的窗口内，不能说明完整
   episode何时开启/停止干预。
6. Z2、Z4-A与Z5都冻结U44 actor hidden；辅助loss反传进student GRU、使表征本身变化的2b从未
   被检验，不能被这三轮连带关闭。

§20第4条“必须early-only才重开prefix-reset”只是一条旧程序触发条件，不是科学否决。
prefix-reset仍未训练检验；下一步先做完整simulator/opponent planner/wrapper/reward/recurrent
hidden snapshot的no-op逐位恢复，以及actor更新后用当前网络burn-in恢复hidden和truncation/GAE
语义门。只有工程门通过，才讨论训练密度Gate。

## B25. 2026-08-08追加：Round Z6-A snapshot工程门

Z6-A不训练actor，只检验prefix-reset能否跳过前缀而保持精确后缀。任务固定为Gate A的28条
U42--U45至少3/4共识Austin development cohort（collision/lost=`19/9`、21个startpoint），在
Gate B冻结window start、当前381D observation被消费前保存F110、LatticePlanner/PurePursuit、
PPO wrapper/reward/corridor gate及U44 actor/privilege-GRU critic hidden。每份snapshot先做pickle
往返，再在同一environment恢复。

结果28/28通过：恢复点observation/hidden，以及后缀每步actor raw/executed action、opponent
action、critic value、reward四分量、两车7D state与steering buffer、双360D LiDAR、collision、
terminal/outcome/首次collision step全部最大误差0。独立复核重新读取28份pickle和两侧56份NPZ，
逐字段`np.array_equal`。26/28为非零prefix，prefix中位345.5步；完整轨迹16,385步中可跳过
9,589步（58.5%），但这不是wall-clock benchmark。

这里的collision/lost=`19/9`是来源任务分层；当前环境原后缀实际终局为14次ego collision、6次
overtake、8次follow，Z6-A不使用这些计数重估U44性能。本轮还只使用确定性mean action，没有
审计训练期exploration RNG/residual block。

判决只到`pass_snapshot_mechanical_gate`。它撤销“snapshot恢复可能做不到”的工程担忧，不证明
PPO有效。参数更新后不得复用旧U44 hidden；下一步必须独立检验用actor-visible observation prefix
对当前网络无梯度burn-in，冻结prefix后探索状态与log-probability语义，以及prefix不进loss/GAE、
窗口边界bootstrap和真实terminal/truncated语义。该Z6-B失败时停止prefix-reset，不回退到
replay-to-prefix。

## B26. 2026-08-08追加：Round Z6-B语义Gate与Z6-BR测量裁决

Z6-B固定Z6-A的28条任务、9,589个prefix observation；U44只重建来源snapshot，真实相邻U45
actor/critic模拟PPO参数改变后的current network。U44逐步reference对window observation与两路
hidden误差全0；U45一次sequence burn-in相对逐步reference最大actor/critic hidden误差
`5.84e-6 / 4.77e-6`、action`2.38e-6`、value`1.49e-7`，通过事前`5e-5`。26/26非零prefix
确认旧U44 state/output与U45不等，不能复用旧hidden。

Buffer用独立`episode_starts`和`recurrent_resets`表达snapshot边界：前者切GAE/sequence，后者
为false以保留burn-in hidden；普通路径未stage时两个mask相同。合成terminated、timeout和
rollout-cut的advantage/return误差`1.49e-8/0`，切段`0/3/5`且prefix transition为0。Baseline
28条和corridor 1,428条collection-equivalent log-ratio/ratio误差均0。

原strict machine report唯一失败是反算telemetry residual要求bit exact：前50步最大误差
`3.18885e-6`。Active、block id、第51步release和revisit均已通过。该反算字段不进入PPO
likelihood，只用于telemetry；因此保留原machine fail，并按独立预注册Z6-BR用相同U45、seed、
task和51步直接记录内部temporal noise。内部首50步误差0，collection-equivalent log-ratio仍0，
反算误差也低于原已冻结`5e-5`界，裁决为
`pass_prefix_reset_semantics_after_measurement_adjudication`。

这只证明current-network burn-in、boundary/recurrent reset、GAE与探索likelihood可以一致表达。
U44 source的一次整段fast path仍出现约`0.0040` hidden误差，说明U45 fast通过不能跨update外推。
下一步必须先做Z6-C no-update训练密度/集成Gate，固定snapshot比例、role/cache与探索模式，实测
完整rollout的ratio/GAE和wall-clock；不直接启动formal PPO。

## B27. 2026-08-08追加：Round Z6-C/Z6-CR训练密度与因果裁决

Z6-C把prefix reset接入真实16-env、102,400-transition collector，但不执行optimizer step。
Baseline/treatment均保持collision/ordinary=`51,200/51,200`、finite、参数不变、GAE误差0及
collection-equivalent ratio误差0。Treatment在119次collision reset中插入39次prefix reset，
28 key全覆盖；prefix/window transition为8.43%/5.31%，墙钟相对baseline为`1.1349x`。

原machine判据12条中仅普通batched envelope失败：最大`|ratio-1|=0.010755`略超事前`0.01`，
所以原fail不改写。由于普通batched路径会参与真实PPO，随后独立预注册Z6-CR并完整重跑；其
clip fraction为0、mean approximate KL `9.05e-10`，相同8个minibatch的普通/exact dry actor
累计梯度cosine `0.9999838`、相对L2差`0.005706`、最大policy-loss差`4.73e-7`，且参数未step。

科学判决是：Z6-C的`0.01`刀锋失败没有推翻“当前实现能以同一PPO更新语义提供足够prefix状态
密度”这一必要条件；只准入一次固定formal prefix-reset PPO。它仍未证明学习收益、Austin验收
或四图泛化，也不代判2b、collision-only与Constrained PPO。原Z6-C fail与Z6-CR pass必须并列
保留，不能简写为“Z6-C通过”。

## B28. 2026-08-09追加：Round Z6-F正式prefix-reset PPO

Z6-F从canonical BC fresh start，只把collision角色每第3次reset替换为28项共识prefix；其余
Austin seed42、16×6,400、privilege-GRU、50/50 role、479/600 pools、baseline探索及标准PPO
loss不变。Warmup+30 formal完整结束；31行metrics、30对checkpoint均finite，actor全部12-key，
每update完成16/16 actor steps。Batched ratio最大`0.019095 < 0.02`，exact全0，prefix/window
最低`8.43%/4.26%`。

U27--U30四图共16包、9,600 episode/trace严格通过。逐图与合计结果：

| update | Austin | Hockenheim | Moscow | Nuerburgring | 合计 | 逐图BC门 |
|---:|---:|---:|---:|---:|---:|---|
| U27 | `17/365` | `29/363` | `36/391` | `23/399` | `105/1518` | 失败 |
| U28 | `20/367` | `27/365` | `32/395` | `22/398` | `101/1525` | 通过 |
| U29 | `23/367` | `28/362` | `27/399` | `23/400` | `101/1528` | 失败 |
| U30 | `21/365` | `27/365` | `33/393` | `22/399` | `103/1522` | 通过 |

U30相对BC collision removed/created=`60/34, p=0.00955`，overtake lost/gained=
`16/93, p=2.21e-14`，因此最低验收通过。相对U44为collision `40/81, p=0.000244`、overtake
`33/77, p=3.30e-5`，是显著的安全--超车交易。最终`collision<40`失败；只有U28/U30通过而不
连续，稳定性不支持。

结论不是“prefix-reset无效”：它相对BC显著双轴改善并把高超车端推到1522，但仍有103次碰撞，
没有解决目标区域中的联合选择。准确判决为“经验前沿扩展，目标未完成；关闭tested配置，不
否决方法类”。不扫prefix参数、不延长U45、不选U28作production；production保持原U30。

## B29. 2026-08-09追加：剩余三条方法的最终执行收口

Z7用40个新起点构造2,880条独立panel，最终41 source/41 exact control、branch0全部数值误差0。
Full BC救回`18/41`且18条全overtake，但低于21条线；controls新collision/lost各`4/41>2`。
因此严格关闭当前collision-only teacher/window实例，不外推其他teacher或一般BC anchoring。

Z8首次让paired collision/progress auxiliary直接更新student GRU，而不是冻结hidden probe。两seed
都真实改变representation，但treatment target仅50/58、controls harm 14/15和12/13，lost仅0/1；
关闭late50、10-epoch具体2b，不外推所有representation learning。

Z9解决Constrained PPO重复计价：reward GAE中移除首次collision `-2.0`，同一事件只作cost。
102,400行的cost/reward、GAE、dual和actor梯度链路全部成立；但85起点五折OOF的MSE skill、
episode-start/early AUROC为`.04038/.42855/.60703`，未过`.05/.65/.65`。因此按事前Gate停止当前
P20 MLP、`d=.10`、固定dual实例，不运行formal。该Gate不是所有Constrained PPO的数学必要条件，
所以结论是“当前实例不准入”，不是“约束式PPO已证伪”。至此所有已授权具体实验均已执行到
各自停止规则允许的最深阶段；没有未完成run。


---

# 第四部分（附录 C）：2026-08-09 独立复核与跨轮汇总

本附录由 Claude 在全部 Round 完成后独立复核产物写成。它**不改写前三部分的任何判决**，
只做四件事：记录我实际重算过什么、把各轮结果放进同一张带证据级别的表、给出跨轮的干预类型
模式，以及点名一条至今没有被任何一轮检验的轴。所有"事实"条目都注明我是从哪个产物算出来的；
所有"推断"条目都显式标为推断。

## C1. 独立核验记录（我自己重算过的部分）

| 核验项 | 我的做法 | 结果 |
|---|---|---|
| Round Z2 四个 JSON 的 SHA-256 | 对 `action_gate/` 直接 `sha256sum` | 与 §A19 记录**精确匹配** |
| Round Z2 产物数量 | 数 `candidate_traces` 与 `branch0` 下的 NPZ | `10,944` + `912`（456 trace + 456 hidden），268MB，与 §A10.3 一致 |
| Round Z2 全部表格算术 | 逐表求和 | 分层 `109+46+13+63+225=456`；五折各列求和等于各层总数；oracle label 两列各 456；head 动作分布两列各 456；`147/148/31/51/79`、`+31/-28`、§A14.2 十二个百分比 —— **全部自洽** |
| Round Z2 branch0 重放 | 读 `quality_validation.branch0_field_max_abs_error` | 全部 15 个字段最大绝对误差 `0.0` |
| Gate B 判决 | 读 `gate_b_report.json` | `verdict: fail`；失败四条为三条 lost-overtake restore 加 `safe_control_overtake_loss_within_5_percent`（`2/28=7.1%>5%`）；`lost_overtake` 的 `restored_overtake_count` 为 **0/9** |
| Z3 共识 cohort | 读 `gate_v0_report.json` | `verdict: pass`；U44 单点回归 `16` → 四点共识 `7`（**56% 瞬态**）；`consensus_by_u44_stratum` 全为 collision、lost_overtake 为 0 |
| Z4-A 判据 | 读 `criteria` | `target_margin_over_fixed: true`、`target_margin_over_frozen_hidden: false`；noop 选择率 `7/456` |
| Z5 配对检验 | 读 `paired_frozen_vs_fixed` | 独立 seed `28:38, p=0.268`；Z4 seed `29:42, p=0.154` |
| Z6-F 四图结果 | 从 `ppo_prefix_reset_consensus1of3/update{27..30}` 的 `results_multi.json` 逐 episode 重算 | `105/1518`、`101/1525`、`101/1528`、`103/1522` —— 与 §32/§B28 记录一致 |
| SWA 四图结果 | 从 `zero_train_ctv2_u42_u45_equal_average` 逐 episode 重算 | `67/1465`；对 U44 配对碰撞 `13/18, p=0.473`、超车 `24/11, p=0.041` |
| SWA 的核心检验 | 计算 created 与四点核心的重合 | created `32`，其中 `13` 落四点核心、`30` 落并集、仅 `2` 全新；inherited `35` 全部含四点核心 `22` —— **§21.1 的动机值 `36` 与实测 `67` 差距很大，权重共识假说被证伪** |
| created 的 checkpoint 稳定性 | 用 U42--U45 四图正式包对同一 BC 基线复算 | 相邻 Jaccard `0.39--0.58`（inherited 对照 `0.59--0.78`）；四点并集 `54`、核心 `14`；出现次数分布 `1次21 / 2次11 / 3次8 / 4次14` |
| checkpoint 权重距离 | 直接读 4 个 `actor.pth` 求范数 | `||theta_U44||=162.75`，`||U42-U45||=0.0347`（相对 `2e-4`）；等权平均可 `strict=True` 载入，仍严格 12-key |
| Z7 / Z8 / Z9 机器判决 | 读各自 report JSON | Z7 V0 `pass`（2,880 场景、40 起点、与既有候选及 Austin600 起点重叠 `0`）；Z8 `fail_close_tested_representation_only_instance` 且 `representation_only_2b_class_refuted: false`、`trainable_gru_changed_representation_all_folds: true`；Z9 `fail_stop_exact_constrained_implementation`，`reward_correction_max_abs_error: 0.0`、`cost_event_count = completed_collision_episode_count = 57` |

我在本轮**没有发现任何执行错误**。branch0 全字段零误差、哈希匹配、nested CV 结构正确、
Z6-A/B/C 的逐位恢复与语义门、以及 §B24--§B29 对刀锋失败（Z6-C `0.010755`、Z6-B 反算
residual `3.18885e-6`）的"原 fail 与复核 pass 并列保留"处理，都符合本仓库最严格的标准。

## C2. 当前经验前沿与证据级别（推断部分已标注）

**证据级别必须与数字一起引用**，否则会把重建包读成正式验收包：

| 模型 | 四图 collision / overtake | 证据级别 |
|---|---:|---|
| Canonical BC | `129 / 1445` | **正式 CUDA**（2026-08-06 补跑） |
| Production U30 | `94 / 1508` | Austin600 正式；跨地图为 headline，MoscowRaceway 仅 `160/600` 中断残留、Nuerburgring 仅 traces |
| 前向走廊时间相关探索 U44 | `62 / 1478` | **正式 CUDA 四图配对包**（§26） |
| ordinary 异线高速重加权 U30 | `73 / 1516` | **trace 重建包**：`status=complete_trace_reconstruction`、`direct_evaluator_aggregate_retained=false`、manifest **无 `device`、无顶层 `collision_scope`**；四图各 600 条完整 trace 齐备，计数与配对可复算 |
| U42--U45 等权平均（SWA） | `67 / 1465` | **正式 CUDA**（device/scope/panel 齐全） |
| prefix-reset U30（Z6-F） | `103 / 1522` | §32/§B28 记录为 U27--U30 共 16 包、9,600 episode 严格通过 |
| 粗 regime 事后拼接 | `58 / 1527` | 不可部署上界（§28） |
| 逐 episode hindsight oracle | `25 / 1557` | 不可部署上界（§28） |
| **目标** | **`<40 / >1500`** | — |

三条基于上表的**推断**：

1. 除两个不可部署上界外，六个可部署点近似落在同一条安全--超车前沿上。**已执行的全部方法
   都在这条线上滑动，没有一个把线本身推开。**
2. 在现有可复算证据下，**重加权 U30 是唯一在四图两轴上同时优于 production U30 的点**
   （`-21` collision、`+8` overtake）。但它的证据级别低于 U44/SWA/BC，**切换 production 前
   必须补一次固定 CUDA 四图确认**；同时 production 自身的 MoscowRaceway/Nuerburgring 缺包，
   使该比较目前是"重建对 headline"，不是同级配对。
3. prefix-reset U30 在四图上被重加权 U30 **双轴支配**（`103/1522` 对 `73/1516`）。这不改变
   §32/§B28 "经验前沿扩展、目标未完成"的判决，只是说明它落在前沿的进攻端且不是该端最优点。

## C3. 跨轮的干预类型模式（推断）

把历史上所有训练臂与本轮全部 Gate 放在一起，按**干预对象**而不是按方法名分类：

| 干预对象 | 实例 | 结果 |
|---|---|---|
| **特定 regime 的探索** | 前向走廊门控时间相关速度噪声（同线前走廊） | 跨地图同线碰撞 `66 -> 22` |
| **特定 regime 的采样权重** | `ordinary_offline_fast_fraction=0.6`（异线且 speed≥0.7） | 唯一四图双轴优于 production 的点 |
| **时间窗口的 transition 密度** | prefix-reset（Z6-F） | 前沿滑动；被重加权 U30 支配 |
| **全局难度 / 多样性** | 805 hard pool、ordinary150、interval15 pool | 三面板全面变差或配对不显著 |
| **冻结产物上的选择器 / 锚定** | BC anchoring（Gate B、Z3、Z7）、Z2、Z4-A、Z5、SWA | 全部未准入 |
| **训练中改变表征的辅助目标** | Z8（2b 首次直接检验） | 表征确实改变（`trainable_gru_changed_representation_all_folds: true`），但 target `50/58`、control harm `14/15` 与 `12/13`、lost `0/1`，未准入 |
| **约束式目标重构** | Z9 preflight | 实现链路成立（reward 去重计价误差 `0.0`、cost 事件 `57=57`），但 cost critic 的 OOF skill/AUROC `.04038/.42855/.60703` 未过 `.05/.65/.65` |

**推断：唯一见效过的两次干预，都是"把训练分布对准一个已被测量定位的 regime"。**
对准时间窗口（prefix-reset）、对准全局难度、或在冻结产物上加选择器，都没有移动前沿。
这不是定律，只是本项目已测配置上的一致经验。

## C4. 一条至今没有被任何一轮检验的轴：转向探索通道（推断，未测）

**事实（已核实）：** 历史上全部 8 条训练臂的 `run_config.json` 中，
`STEERING_LATENT_STD` 一律为 `0.03`——baseline、conditional white、global temporal、
conditional temporal、corridor gated、hard-neighbor、重加权、prefix-reset 无一例外。
**全部速度探索研究都只动了速度通道，转向通道从未被改变过。**

**事实（已核实）：** 探索是纯训练期机制。`eval_multiagent.py` 直接取网络 mean action，
评测时没有任何噪声或门；`FrontCorridorGate` 的条件量是 `maximum_front_gap_m`、
`maximum_abs_opponent_lateral_d_m`、正向横向 OBB 重叠，全部是模拟器特权几何。
**因此一个训练期探索门不需要任何部署期 trigger。**

**推断：** `§30.6` rule 3 与 `HANDOFF §10.1` item 16 关闭 side-phase steering exploration 的
理由是"缺少可靠部署期 conditioning"。该理由适用于**部署期相位门控**，但**不适用于训练期
探索门**——前向走廊速度探索本身就没有部署期 conditioning，却是有效的。因此这条轴在当前
文档里属于**按不适用的前提被关闭**，不是被证据关闭。

支持它值得一试的已测证据：
- 新造失败的接触几何以近平行侧/后擦碰为主，相对 yaw 中位 `3.67°`（§9.9 / §28）；
- 损失集中在**异线绕行超车**，是横向机动；
- Round Z2 找到的最强单一固定干预是 `steer +0.02 / speed +0.5`（§A13.1），**带转向分量**；
- oracle 标签中转向族合计 early `46` / late `38`，其中 `steer_p0p02` 单独 `22/25`（§A12.2）。

**边界：这是一条未测假说，不是发现。** 它同样可能只是又一次前沿滑动。若要执行，必须按
本文件既有标准单独预注册：单变量、seed 42、区间评测（≥4 相邻 checkpoint）、三面板配对身份，
并预先写明失败即关闭、不扫幅度。

## C5. 全局判断

**已执行的全部实验没有执行错误。** 它们买到了三样东西：避免了多次注定失败的正式训练；
产出了若干改变判断的正面结论（hidden 对当前几何的解码 9/9 优于输入；关键窗口局部动作
`137/155` 存在；hindsight 上界 `25/1557`；约 50% created failure 是 checkpoint 瞬态）；
以及一套被反复修正过的验收方法学（区间报告、共识 cohort、匹配操作点、fold-local PCA、
nested CV、"原 fail 与复核 pass 并列保留"）。

**同时必须诚实记录两条代价：**

1. **前六轮全部在冻结产物上。** 直到 Z8 才第一次让辅助损失反传进 student GRU，直到 Z6-F 才
   第一次真正训练。在此之前所有"关闭"都只能是冻结实例的关闭——这一点 §B24 第 6 条已经写明，
   本附录只是把它提前到方法学层面。
2. **最便宜的一步至今未执行。** 重加权 U30 在四图两轴上优于 production，而它的 CUDA 确认
   （2,400 episode）加 production 的 MoscowRaceway/Nuerburgring 补齐（1,200 episode）合计
   **3,600 episode、零训练**——不到 Round Z2 单轮 `11,400` 条闭环 episode 的三分之一。

**关于 `<40 / >1500`：** U44 的 `62 = inherited 35 + created 27`，而没有任何已测方法有机制
去动 inherited 那 35 条；created 又有约一半是刀锋瞬态。结合六个可部署点近似共线这一事实，
**当前推断是该目标在现有约束集（361D 输入、12-key actor、Austin-only 训练、单阶段 PPO）内
不可达**。若目标不可谈判，需要讨论的是松哪一个约束，而不是再增加一条同类臂。这是推断，
不是证明；`25/1557` 说明目标行为存在于同结构 actor 的行为并集中，但没有证明它能被装进
单一参数向量。

## C6. 本附录不改变的事项

- 不改变前三部分的任何 verdict、停止规则或证据边界；
- 不改变 production：仍为 U30；
- 不授权任何训练；C4 的转向探索轴是**待预注册的候选**，不是获批项目；
- 不把任何诊断成功写成 actor 性能，也不把任何未测轴写成可行方案。


---

# 第五部分（附录 D）：2026-08-09 Codex 独立审计、更正与当前判读

本附录回答两个容易混淆的问题：Z2/Z5 到底有没有“探索动作”，以及它们对当前 PPO actor
到底产生了什么效果。审计结论先写在前面：

1. Z2/Z5 的最终有效机器报告、主表算术、预注册判决和“没有 actor update”边界均成立；没有发现
   会使这两轮结果失效的未处理执行错误。
2. Z2做了**反事实分支动作探索**，但没有做PPO的on-policy探索或参数更新；Z5没有产生任何新
   环境分支，只在Z2已有闭环结局上训练和校准离线选择器。
3. Z2建立了很强的“局部动作存在性”正面证据；Z5没有在固定control harm预算下建立frozen-hidden
   selector严格优于状态无关固定动作的证据。两者共同定位了“动作存在，但可部署选择、触发和
   actor extraction没有成立”，不能写成“动作探索无效”。
4. 合并文档并非完全无误。附录 C 的“双轴支配”“没有方法触及 inherited-35”以及由此推出的
   “当前约束下不可达”存在实质性错误或过度外推；§26.7 对Z5 target检验使用`n=456`的说法也不
   精确。本附录逐项更正，但不改写历史原文。

## D1. 审计范围、证据源与复核层级

本次审计按“机器产物与当前权威文档优先于合并报告措辞”的顺序核对：

| 层级 | 本次读取或重算的内容 | 用途 |
|---|---|---|
| 当前项目权威 | `.agents/HANDOFF.md`，重点为§9.17--§9.28与§10 | 核对当前状态、production和停止边界 |
| Z2冻结合同 | `action_gate_plan.json` | 核对456条task、五层分母、fold、窗口与12动作库 |
| Z2机器结果 | `action_gate_report.json`、`branch0_results.json`、`candidate_results.json` | 核对existence、rankability、选择映射与配对结果 |
| Z5机器结果 | `harm_matched_operating_point_report.json`与`harm_matched_operating_point_z4_seed_report.json` | 核对nested协议、两套seed、harm与paired exact结果 |
| 合并文档 | 本文件§0--§35、附录A--C | 查找历史结论与后续修正之间的冲突、遗漏和过度外推 |

Z2四个主JSON的SHA-256复核为：

```text
action_gate_plan.json       0ae070736fc290763f3ca8bc884be4ade1325965c728647def7474bfe4ad3896
action_gate_report.json     4d6c97fa0c705838cd936a4285dffef8455e8cb1c02607aa4add6939be47b0ae
branch0_results.json        905c16d987361f30a6093fa032b6314c299f2f593ef53666123b092f534b984f
candidate_results.json      875fc96dfa97f5e4b6f8468f53bd67172e042a16f5ea49a54aa48e5855c4d92d
```

它们与附录A记录一致。此次没有重跑11,400条闭环仿真；原因不是把Markdown当作证据，而是冻结
输入、机器报告、逐episode选择映射、质量检查和文件身份均仍在，足以复核本次所讨论的计数、
配对关系和方法边界。

## D2. “探索动作”必须区分的四种含义

| 含义 | Z2 | Z5 |
|---|---|---|
| 在simulator中执行不同动作并观察真实闭环后果 | **是**。每个task在early/late窗口各运行12个residual | **否**。直接复用Z2分支结局 |
| PPO rollout中的随机on-policy exploration | 否 | 否 |
| 学习一个状态条件动作选择器 | 是，离线OOF `ActionScorer` | 是，离线三分类outcome model与nested操作点 |
| 更新最终actor或产生可部署checkpoint | 否 | 否 |

因此，最准确的命名是：

- **Z2：future-event-located counterfactual branch action exploration and offline rankability Gate**；
- **Z5：offline harm-budget-constrained selector calibration Gate**。

Z2不是“让PPO自己多试动作”。它先用U44完整确定性轨迹的**未来**首次碰撞或全局最小clearance
定位窗口，再相对候选自身当前U44 mean持续施加固定residual。动作库包含4个steering、4个speed、
4个steering+speed组合；窗口为early/late各最多50步。总量严格是：

```text
456 no-op replay
+ 456 tasks × 2 windows × 12 residual actions
= 11,400 closed-loop traces
```

这些分支是真实闭环动作干预，不是静态Q值预测；但窗口位置使用未来事件，只能承担训练侧机制
诊断，不能直接部署。Z5则连新的闭环动作都没有执行：它只对既有13种选择（noop+12 residual）
重新学习outcome概率，并用inner-OOF数据选择`lambda`与noop阈值。

## D3. Z2的实际效果：动作可达性强，tested选择器不准入

### D3.1 动作存在性是明确的正结果

成功被严格定义为候选最终成为“无ego collision的overtake”，不是只把碰撞改成follow：

| 目标层 | early oracle | late oracle |
|---|---:|---:|
| inherited collision | `93/109` | `96/109` |
| created collision | `44/46` | `45/46` |
| lost-overtake | `10/13` | `7/13` |
| 三层合计 | **`147/168`** | **`148/168`** |
| collision两层合计 | `137/155` | `141/155` |

两窗口均通过全部预注册action-existence门，并覆盖48个collision startpoint、raceline0/1/2和三类
动作family。这个结果严谨排除的是“在已测窗口和动作库内没有局部闭环解”；它不证明部署actor
知道何时以及选择哪一个解。

### D3.2 原Z2选择器的负结果也成立，但范围必须锁定

| 选择器 | inherited / created / lost | target | controls新collision / overtake loss |
|---|---:|---:|---:|
| early 13-class OOF head | `19 / 11 / 1` | `31` | `15 / 17` |
| late 13-class OOF head | `34 / 17 / 0` | `51` | `3 / 5` |
| late grouped fixed action | `45 / 33 / 1` | `79` | `5 / 5` |

根据机器报告中的逐episode结果，本次额外重算late target的head-vs-fixed配对：head-only成功20条、
fixed-only成功48条，双侧exact `p=0.000914`。所以附录A中“该Z2 extraction显著落后于fixed”在
**这个13-class标签、这个head和这个操作点**上成立。该p值沿用机器报告的episode-pair exact
口径，不是按ego startpoint聚类稳健的推断；它不改变下面的类级外推边界。

但它不能外推成“U44 hidden没有条件动作信息”。Z4-A换成三分类outcome监督与更激进操作点后，
同一frozen hidden得到target 104，说明Z2的`51<79`混合了标签、打分规则与干预率。Z2严谨关闭的
是原13-class top-1 preference extraction，不是frozen representation的信息论上限。

### D3.3 Z2没有产生PPO性能收益

Z2没有修改U44、没有接入actor loss、没有训练student，也没有四图candidate checkpoint。因此：

- `147/148`是hindsight oracle可达性，不是actor成功数；
- head相对noop的净碰撞/超车改善只存在于future-event已经给定的Austin development窗口；
- Z2对“当前PPO结果是否优化”的答案是**未检验，不是优化失败**；
- 它对后续最有价值的贡献是把瓶颈从“局部动作不存在”收窄到“触发、条件选择、progress保持和
  actor extraction”。

## D4. Z5的实际效果：没有新动作探索，未建立条件化选择优势

### D4.1 Z5修复了Z2与Z4-A的操作点混淆

Z2 late head是较保守点`51 @ 3/5`，Z4-A frozen outcome selector是激进点`104 @ 13/20`；直接
比较二者与fixed `79 @ 5/5`没有同一harm口径。Z5在每个outer-train内做4-fold grouped inner-OOF，
只用inner结果选择`lambda/tau`，然后冻结并一次应用到outer-test。因此它检验的是“在control
harm率不高于`5/225`的预算下，条件选择是否严格超过fixed 79”，不是又一次动作库搜索。

| 合同 | inherited / created / lost | target | controls collision / loss | target paired vs fixed |
|---|---:|---:|---:|---:|
| independent outer seeds | `42 / 25 / 2` | **69** | `1 / 3` | `28 : 38`, `p=0.268` |
| exact Z4 outer seeds | `42 / 23 / 1` | **66** | `3 / 4` | `29 : 42`, `p=0.154` |
| grouped fixed | `45 / 33 / 1` | **79** | `5 / 5` | — |

两套seed都满足harm上限但没有满足事前`target > 79`，所以
`frozen_hidden_conditional_value_reopened=false`和关闭tested实例的程序判决正确。

### D4.2 统计表述必须比原文更精确

本轮总数据确实有456条，但用途分为：

```text
168 target = 109 inherited collision + 46 created collision + 13 lost-overtake
225 safe controls
63 inherited-follow diagnostics
```

`p=0.268/0.154`的target paired exact检验只发生在**168个target episode**上，不是456个episode。
§26.7和附录B把它写成“在`n=456`下未检出”不精确；正确说法是“模型训练/选择使用456条，target
比较分母为168，controls harm分母为225”。

此外，fold按ego startpoint分组避免了训练泄漏，但上述exact binomial p值仍把discordant episode
当作配对单位，并不是cluster-robust startpoint级推断。§26.7的“约19%功效”只能理解为固定66个
discordant且近似独立时的条件计算，不是完整的cluster-aware功效分析。预注册Gate并不依赖
`p<0.05`，所以这项限制不推翻Gate判决，但限制了“方法更差/相等”的统计外推。

两套nested selector外层只实际使用到`1/3`和`3/4` harm，fixed使用`5/5`。因此这仍是
**budget-constrained comparison**，不是恰好matched到同一个经验harm点。无泄漏地让有限outer
样本恰好用满预算并未实现；事后在outer结果上选择`84 @ 3/5`或`72 @ 4/5`会引入乐观选择，不能
替代主判决。

### D4.3 Z5能够支持和不能支持的结论

可以支持：

- 在当前late窗口、12 residual、frozen U44 hidden、三分类outcome loss与nested校准合同下，
  没有建立相对状态无关fixed action的严格target增益；
- Z4-A的104主要来自激进干预操作点，不能作为同harm预算优势；
- 显式50步history在Z4-A和两套nested seed下均没有超过frozen hidden，关闭该具体history treatment
  是充分的项目判决。

不能支持：

- frozen selector显著差于fixed；两个paired p值均未显著；
- frozen hidden完全没有动作相关信息；
- 所有action-conditioned方法、不同动作库、不同窗口或会改变student GRU的辅助学习都失败；
- 当前PPO actor已被改进或已被反事实方法训练失败。Z5没有actor update。

## D5. Z2/Z5共同检验到因果链的哪一段

| 必要环节 | 当前证据 | 判读 |
|---|---|---|
| 固定局部动作库中存在闭环优动作 | Z2 oracle `147/168`、`148/168` | **强成立，仅限future-event窗口与Austin development** |
| actor-visible信息可跨startpoint选择动作 | Z2原head失败；Z5未严格超过fixed | **tested extraction未建立** |
| 选择器能在相同control harm下提高target | Z5是预算约束但欠用harm，69/66对79 | **未建立；也未证明严格更差** |
| 部署时能发现窗口起止 | 两轮均由未来event定位 | **未检验** |
| 选择规则能写入actor并经PPO保持 | 两轮均无actor update | **未检验** |
| 新actor对自身访问的新状态仍有效 | 无student闭环训练 | **未检验** |
| Austin规则跨三张测试图泛化 | 无candidate actor | **未检验** |

这张链条解释了为什么“已经找到很多有效动作”与“没有得到更好的actor”并不矛盾。反事实数据
解决的是局部可达性测量；它不会自动提供在线trigger、可靠selector、actor优化通路或跨地图泛化。

## D6. 对合并文档中错误与遗漏的逐条裁决

| 位置/原主张 | 审计判决 | 更正后的准确表述 |
|---|---|---|
| 第一部分§0仍写“停在Z6-B、未生成新actor” | **历史快照，不是当前状态** | 后续已完成Z6-F并生成正式actor；当前状态读`HANDOFF.md`和本文件§32.4/附录B29 |
| 附录A§A17“当前可说五类关闭/不准入” | **Z2完成时的历史判决** | 后续Z6训练、Z7、Z8、Z9已分别触及相关实例；不能作为2026-08-09最终状态表 |
| §26.7“在`n=456`下未检出target优势” | **分母表述错误** | 数据集456；target paired comparison为168，controls为225 |
| 附录C§C2/C3称`103/1522`被`73/1516`双轴支配 | **数学错误** | 73碰撞更少，但1516超车少于1522；两点互不支配 |
| C2/C3称六点都在同一条线、全部方法只滑动且唯一两次见效是regime对准 | **证据不足且受上一条错误影响** | 可描述安全--超车trade-off；不能由有限、证据等级不同的点证明同一曲线或“只有两种干预见效” |
| C5称没有任何已测方法能动U44 inherited-35 | **与现有配对算术矛盾** | Z6-F相对U44移除40次collision；U44 created只有27，因此至少`40-27=13`次移除必来自inherited-35 |
| C5据此推断目标在当前约束下不可达 | **过度外推** | 只能说所有已测可部署actor均未达标，且现有点显示强trade-off；不可达性未被证明 |
| C4称“转向探索通道未检验” | **方向基本正确但需限定术语** | 所有训练均已有`STEERING_LATENT_STD=0.03`；未检验的是改变其幅度/时间相关性/训练期门控，不是从未探索steering动作 |
| C4用Z2最强fixed动作支持转向探索 | **只能作假说先验** | 该动作同时含`steer+0.02`和`speed+0.5`且依赖future-event窗口，不能隔离steering exploration的因果收益 |
| C1/C5“全部实验没有执行错误”“避免注定失败训练” | **措辞过强** | 最终承担判决的有效产物没有发现未解决执行错误；此前确有被fail-closed拦截的机械错误。Gate只表示不准入，不证明被阻止的formal训练必然失败 |
| §25/Z4标题“representation-changing” | **易误读** | Z4-A训练外置history encoder但没有改变student actor GRU；首次直接改变原GRU的是Z8 |

上述更正不推翻Z2、Z5、Z6-F、Z7、Z8或Z9的机器verdict；它们修正的是跨轮统计口径与类级推断。

## D7. 更正后的当前经验点关系

只看合计headline且把collision视为越低越好、overtake视为越高越好：

| actor | collision / overtake | 与另外两点的关系 | 证据边界 |
|---|---:|---|---|
| U44 | `62 / 1478` | 更安全、超车更少 | 正式CUDA四图配对包 |
| ordinary异线高速重加权U30 | `73 / 1516` | 位于中间 | 完整trace重建；device/scope provenance弱于正式包 |
| prefix-reset Z6-F U30 | `103 / 1522` | 超车最高、碰撞最多 | 正式CUDA四图配对包 |

三点按这两个聚合指标**互不支配**。Production `94/1508`按数值被重加权`73/1516`双轴改善，
但两者证据等级不同，所以这不足以自动切换production；附录C提出补正式CUDA确认仍是合理的
评估建议。Z6-F不是被重加权点支配，而是更偏进攻端的不同trade-off点。

有限点近似呈现trade-off，不等于已经估计出一条稳定Pareto曲线；更不能由此证明任何新机制都
只能沿线滑动。准确结论是：**当前没有一个已测可部署actor同时达到`collision < 40`与
`overtake > 1500`。**

## D8. 当前最强科学结论

### D8.1 已经得到的正面知识

1. **局部动作不是主要缺口。** 已测动作库在两个各0.5秒的事件前窗口中对collision两层有
   137/155与141/155的oracle rescue，并且成功动作跨起点、raceline和family存在。
2. **固定动作本身就有很强的条件面板收益。** late统一动作取得79个target success，说明该
   future-event定位面板有明显的共同局部响应；任何状态条件selector都必须证明自己在相同harm
   口径下比这个基线增加价值。
3. **操作点控制是必要的。** Z2的51和Z4-A的104方向相反，Z5证明未约束干预率会把selector能力
   与安全税混在一起。
4. **Z2分支系统是可信的机制工具。** 456/456 no-op及字段误差0、10,944候选合同通过，故oracle与
   selector差距不能归因于分支重放错误。

### D8.2 已经得到的负面知识

1. Z2的13-class top-1 scorer没有同时满足lost-overtake、control harm和相对fixed门。
2. Z5的三分类frozen-hidden selector在两套seed的无泄漏nested合同下均未达到`target > 79`。
3. 显式50步actor-visible history没有给frozen hidden增加选择效用；该精确history treatment已由
   Z4-A和Z5一致关闭。
4. 仅增加反事实标签和离线head不足以自动形成可部署PPO改进；两轮没有跨过trigger、extraction、
   student distribution shift或跨地图这四道门。

### D8.3 仍然未知，禁止从本轮推断

1. 不依赖未来event的在线phase trigger能否建立；
2. 不同动作horizon、连续局部策略或不同监督能否超过fixed动作；
3. 训练期改变steering探索幅度、相关性或门控是否能改善最终actor；
4. 一个全新辅助监督能否比Z8已失败的paired collision/progress目标更有效；
5. 当前架构与Austin-only约束下联合目标是否理论可达。

## D9. 对后续决策的约束性含义

1. **不应重跑Z2/Z5同一实例。** 它们的输入、动作库、窗口、标签、nested校准和停止规则已经得到
   完整回答；调阈值、动作幅度或seed属于结果后扫描。
2. **继续“多提供一些反事实动作”没有独立依据。** 当前action existence已接近饱和；新方法若
   仍走反事实路线，必须把可部署trigger、在control预算下的selector增益和actor写入机制预先
   定义清楚，否则只是重复测量oracle。
3. **Z2/Z5不能用来否决PPO训练pipeline。** 它们没有训练actor。真正训练过的本轮候选是Z6-F，
   其结果是`103/1522`：相对BC最低线通过并到达更高超车端，但未改善U44安全端、未达联合目标。
4. **转向探索是未测轴，不是已验证解。** 附录C指出所有历史臂都没有改变steering探索参数，这一
   事实有价值；但Z2的coordinated fixed residual不能预测正式训练效果。若未来授权，必须作为
   新单变量预注册，而不能由本报告直接准入。
5. **联合目标应记录为“当前未达到”，不是“已证明不可达”。** 当前最强证据是经验trade-off和
   多个具体实例失败；这足以停止无机制的同类扫描，但不足以作理论不可能性结论。

## D10. Z2/Z5最终一句话判读

| Round | 最终科学判读 | 是否优化当前PPO actor |
|---|---|---|
| Z2 | **成功证明局部反事实动作大量存在；失败的是已测跨起点选择器和部署触发链，不是动作探索本身** | 否；没有actor update |
| Z5 | **成功消除了Z2/Z4-A操作点混淆；在预注册harm预算下未建立frozen-hidden selector严格优于fixed的证据** | 否；零新仿真、没有actor update |

因此，若问题是“Z2/Z5效果怎么样”，答案不是“它们让PPO变好了”或“动作探索全部无效”，而是：
**它们提供了可信的正向可达性证据和负向选择器准入证据，但没有产生任何可部署策略收益。**
