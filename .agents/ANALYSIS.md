# End2Race PPO 实验 ANALYSIS

更新时间：2026-08-09（Asia/Singapore；Round Z9 collision-cost Constrained PPO preflight完成；机械链路通过但startpoint-OOF三门失败，formal停止）

本文是 End2Race PPO 的**完整实验分析记录**：保留实验设计、控制变量、面板和分母定义、
逐 checkpoint/逐分层结果、同场景配对变化、统计不确定性、机制判断、负结果原因和证据边界。
其内容已从原 `.agents/HANDOFF.md` 完整迁移，并吸收历史分析产物中仍有决策价值的
信息；即使后续清理原始 JSON、CSV、trace 和 report，本文仍应足以恢复**已经形成的实验
判决、核心统计与适用边界**。这不等于保留任意重新分层或逐 transition 重算能力；清理前
额外固化的不可替代紧凑证据见 §21。

三份文档的职责边界：

- `.agents/HANDOFF.md`：当前 production、运行合同、核心实验判决、停止/重开规则；
- `.agents/ANALYSIS.md`：本文，完整结果、方法、数据分层和分析判断；
- `.agents/EXPERIMENTS.md`：历史实验工具与回归测试的实现逻辑和重建合同。

本文不把分析产物路径或其哈希作为证据索引。Actor checkpoint 的唯一 SHA-256 登记表位于
`HANDOFF.md` §2；本文只记录实验臂、run和checkpoint update，不重复摘要，避免两份表漂移。
各章内容与时效如下：

| 章节 | 内容 | 时效 |
|---|---|---|
| §1 | 当前状态、最后活动、production 决策 | 2026-08-08，最优先 |
| §2-§9 | 架构、数值合同、CLI、记录格式 | 已对齐当前源码 |
| §10 | 0721 七个受控 run | 历史结论，仍有效 |
| §16 | 默认reward审计、Post-pass、risk-L12与following-response候选 | 已审计；训练臂已收口，离线候选未准入 |
| §17 | Hard-neighbor / boundary-aware pool | 已收口，已否决 |
| §18 | 速度探索（逐步独立、条件白噪声、全局时间相关、条件时间相关、走廊门控及异线高速重加权） | 已收口，无未完成 run；仅10步/25步保持未测 |
| §19 | 其余已完成训练/eval调查（采样、std、跨地图、噪声） | 已收口 |
| §20 | Actor可观测性、oracle可达性、共享动作库与制动响应曲面 | 离线诊断已收口；只作机制证据 |
| §21 | 清理前核心证据固化 | 历史证据与恢复边界 |
| §22 | PPO职责收口与语义等价性 | 当前重构验收依据 |
| §23 | Regime动作收益可分性与共享actor人工偏好梯度 | 2026-08-05无训练诊断；被§24修正适用边界 |
| §24 | Fresh first-step PPO regime梯度 | 2026-08-06完成；否决naive梯度投影训练 |
| §25 | Collision/ordinary role配比假说 | 2026-08-06复核；证据不足，不启动25% role臂 |
| §26 | 前向走廊时间相关速度探索U44四图BC验收 | 2026-08-06完成；正式CUDA安全候选 |
| §27 | ordinary异线高速重加权U30按新四图口径重判 | 2026-08-06完成；计数通过，CUDA provenance待确认 |
| §28 | Austin-only四图碰撞身份、接触几何与经验上限 | 2026-08-06无训练诊断；后续最小筛查见§29--§32 |
| §29 | Phase-spillover与Pressure-conditioning最小诊断 | 2026-08-07完成；两项均不准入训练 |
| §30 | Interaction-phase早期可分性最小诊断 | 2026-08-07完成；未通过后续动作/credit准入门 |
| §31 | 真值几何/速率线性早期可分性预检 | 2026-08-07完成；未通过辅助训练前置门，不能外推成信息上界 |
| §32 | 当前交互几何表征缺口预检 | 2026-08-07完成；缺口不成立，该量族辅助目标不准入 |
| §33 | Round Z0：U42--U45等权checkpoint平均 | 2026-08-08完成；四图`67/1465`，不满足BC逐图下限或U44 Pareto线，关闭 |
| §34 | BC-safe anchoring Gate A | 2026-08-08完成；共识cohort 28条、`C/L=19/9`，六条准入线通过；Gate B未启动 |
| §35 | BC-safe anchoring Gate B | 2026-08-08完成；C救回`10/19`，L恢复`0/9`，control损失`2/28`，科学失败并关闭方向 |
| §36 | Round Z2反事实动作存在性与可排序性 | 2026-08-08完成；动作oracle强但hidden排序未过progress/control/固定基线门，关闭tested action-conditioned/preference路线 |
| §37 | Round Z3 collision-only BC anchoring独立validation | 2026-08-08停止；7条cohort通过样本门，但精确matched controls无解，branch/训练均未运行，只能判inconclusive |
| §38 | Round Z4-A representation-changing action-response Gate | 2026-08-08完成；50步历史treatment为`68/32/2`、controls `13/21`、target `102 < 104` frozen-hidden，关闭该具体实例 |
| §39 | Round Z5 budget-constrained frozen-hidden operating point | 2026-08-08完成；nested frozen为`69 @ 1/3`与exact-Z4-seed `66 @ 3/4`，未检出严格优于fixed `79 @ 5/5`，关闭tested outcome selector；2b未测 |
| §40 | Round Z6-A prefix-reset snapshot no-op工程门 | 2026-08-08完成；28条多checkpoint共识任务全部逐位恢复，下一步只准入current-network burn-in/GAE语义门，不准入PPO |
| §41 | Round Z6-B current-network burn-in/GAE语义Gate | 2026-08-08完成；原strict report因非因果telemetry反算误差fail，独立Z6-BR确认内部noise/log-ratio/mask/GAE必要条件通过；只准入无更新训练密度Gate |
| §42 | Round Z6-C/Z6-CR no-update训练密度与批量重放裁决 | 2026-08-08完成；原`0.01` batched envelope刀锋失败保留，完整重跑与8-minibatch干梯度确认同一PPO更新语义可用；只准入一次正式训练 |
| §43 | Round Z6-F单次正式prefix-reset PPO | 2026-08-09完成；U30四图`103/1522`逐图通过BC线但未达`<40`安全目标，U28/U30非连续通过，关闭tested配置而不否决方法类 |
| §44 | Round Z7 collision-only BC anchoring独立重开 | 2026-08-09完成；41 source/41 exact control，branch0精确；rescue `18/41 < 21`且control harm `4/41 > 2`，关闭当前teacher/window实例 |
| §45 | Round Z8 GRU-changing paired action-response auxiliary Gate | 2026-08-09完成；两seed真实改变GRU但target仅50/58、control harm 14/15与12/13，关闭具体2b实例，不否决方法类 |
| §46 | Round Z9 collision-cost Constrained PPO preflight | 2026-08-09完成；102,400行机械链路通过，OOF skill/AUROC三门失败，停止当前实例且不外推方法类 |

写入新结论时必须先用当前源码、run记录和机器可读结果核对；一旦原始产物被用户清理，
历史实验的配置、数字、边界和停止规则以本文对应专题为准。

仓库：`/home/haowei/Documents/End2Race`

分支：`main`

提交和工作树状态必须在接手时实时查询；本文不维护会在下一次提交后立即过期的HEAD值。
本轮一次性审计脚本和分析产物已在结论固化后删除。禁止reset或checkout。
用户已单独授权后续清理历史分析产物、实验工具和回归测试源码，其文档固化边界见§21；
该授权不扩展到其他目录。被否决的Post-pass生产模块与L12运行时override已在后续
reward清理中移除；§16保留的是历史实验合同与否决证据，不是当前可用接口。
同日又在`EXPERIMENTS.md`完成可重建合同后退役ordinary150、外部fixed collision pool、
boundary-aware hard-neighbor 805/比例采样和actor-path mismatch cache复用。以下专题
继续按历史事实描述其设计和结果；“历史入口”不表示当前CLI仍存在。ordinary异线高速
重加权按用户决定保留为默认关闭研究工具。
`.agents/`已纳入Git版本管理；普通`git status`必须显示这里的后续修改。

本文是 `.agents/` 中完整实验分析的权威记录；当前production与允许的下一步以
`HANDOFF.md`为第一入口。Hard-neighbor的事实、结果、production决策
和仍有行动价值的历史设计边界已去重合并到§17；原独立
`.agents/HARD_NEIGHBOR_HANDOFF.md`已在2026-07-29完成合并后移除，不再存在第二个入口。
旧H1/H2、sustained exploration、cleanup和fixed-parameter GUIDE已与当前代码冲突，也不再
逐字嵌入本文。历史实现可从Git history追溯；当前代码行为以实际源码为准，历史实验判决
以本文已经固化的核心记录为准。

---

## 0. 覆盖范围与阅读方式

2026-07-30迁移时重新盘点了41个历史分析实验ID；有决策价值的内容
按实验问题而不是按目录名称归并如下：

| 实验族 | 本文位置 | 已固化内容 |
|---|---|---|
| 默认reward、甩尾基线、Post-pass公式/训练、risk sweep、L12、following-response | §16 | 公式、参数、时机、训练分量、Austin600与held-out分层、停止规则 |
| Hard-neighbor、boundary-aware cache、分层比例 | §17 | cache构造、采样合同、完整805池A/B、低比例证据边界 |
| 各速度探索模式、走廊门宽、checkpoint曲线、45 updates、异线高速重加权与checkpoint区间 | §18 | 全部闭环结果、same/off-line分解、跨地图、稳定区间和失败迁移 |
| Baseline重现/晚期稳定、speed std、退火、ordinary150、interval15 pool、跨地图BC/U30、单车masking | §19 | 受控结果、配对变化、噪声/选择偏差边界和否决口径 |
| Actor可观测性、Austin13 oracle、334共享动作库/ranking、制动幅度×提前量 | §20 | 探针、可达上限、候选覆盖、局部回报曲面和禁止越界 |

§1–§15保留解释这些实验所需的代码/数值/面板历史快照；当前运行入口和production摘要以
`HANDOFF.md`为准。本文不列41个目录的路径，也不要求目录永久保留：核心配置、分母、结果、
机制与边界已写入对应专题。

---

## 1. 当前状态

### 1.0 最后活动状态（接手第一件要看的事）

**没有任何训练或 eval 进程在运行**（本次用避免自匹配的
`pgrep -af '[r]un\.sh|[t]rain_ppo\.py|[e]val_multiagent\.py|[e]valuate\.sh'`
检查，未发现真实任务；进程状态会变，接手时必须重查）。

**2026-08-09最后完成活动是Round Z8 GRU-changing paired action-response auxiliary Gate，无PPO。**
456条late recurrent输入按batch-size-one重建U44 hidden/action最大误差0；paired collision/progress
loss在两seed的五fold都使GRU参数相对变化约0.0021--0.0026、test hidden变化约0.0094--0.0123，
确认本轮真正触及2b。Seed7100 treatment target/control为`50 @ 14/15`，低于frozen
`58 @ 12/13`；seed8100为`58 @ 12/13`，低于frozen `61 @ 19/20`且绝对harm仍超5/5；lost仅0/1。
两seed都未过target88、相对frozen +9、lost4和control门。§45关闭该具体2b，不运行validation/
PPO，不外推所有representation-only辅助目标。

**同日此前完成活动是Round Z7 collision-only BC anchoring overlap-supported独立重开，
无训练。** 新40起点×2 raceline×4 interval×9 speed共2,880条panel与历史Austin起点精确零
交集；分阶段screen完成5,937条required actor replay/trace，得到41条稳定eligible collision与
41条同raceline/speed无放回controls，覆盖21起点、r0/r2=`31/10`，V0全部通过。branch0在82条上
全部动作、pose/speed与双LiDAR最大误差0。full-BC只救回`18/41 < 21`，虽然18/18最终overtake，
但control又新造collision并丢overtake各`4/41 > 2`。§44据此严谨关闭当前canonical BC ×
overlap-supported stable collision × 1.5秒窗口实例；没有生成anchor或actor，不外推其他teacher/
窗口。旧Z3仍保持inconclusive，但其control-support阻断已由本轮真正解除。

**2026-08-08最后完成活动是Round Z6-B current-network burn-in/GAE语义Gate及Z6-BR测量裁决，
无actor更新。** 28条Z6-A任务重新得到9,589行prefix；U44 source observation与两路hidden对
snapshot的最大误差均为0。真实相邻U45 current-network的fast sequence burn-in相对逐步reference
最大hidden/action/value误差为`5.84e-6 / 2.38e-6 / 1.49e-7`，通过事前`5e-5`线；26条非零prefix
全部表明旧U44 state与U45 state不同。Snapshot boundary用独立`recurrent_resets`拆开后，GAE/
return相对手算误差`1.49e-8 / 0`，sequence切分、非零hidden保留和默认路径等价均通过；baseline
与corridor的collection-equivalent log-ratio均为0。原strict report唯一false是从action反算的
telemetry residual有`3.18885e-6`舍入；Z6-BR直接读取内部temporal noise，首50步误差0，并复核
active/block/revisit与likelihood全部通过。因此科学判决为语义必要条件通过，但这只准入独立的
no-update训练密度/吞吐Gate，不准入formal PPO。完整边界见§41。

**同日此前完成活动是Round Z6-A prefix-reset snapshot no-op工程门，无actor更新。**
固定Gate A的28条U42--U45至少3/4共识development任务，在各自冻结窗口起点、当前observation
被网络消费前保存F110/LatticePlanner/reward/wrapper状态与U44 actor/privilege-GRU critic hidden；
每份snapshot先做pickle往返，再在同一environment恢复并重跑确定性后缀。28/28全部通过，
381D observation、actor/critic hidden、actor/opponent action、critic value、reward及四分量、
两车state/steering buffer、双LiDAR、collision和terminal/outcome全部最大误差0。26/28前缀大于
0，prefix中位345.5步；共跳过9,589/16,385个原始前缀步（58.5%）。§40只证明机械快照可行；
在Z6-A完成当时current-network burn-in、GAE/bootstrap与PPO效果仍未检验；前两项后来见§41，
PPO效果仍未检验，不得直接训练。

**同日此前完成活动是Round Z5 budget-constrained frozen-hidden operating-point Gate，无actor更新。**
它复用Round Z2的全部真实branch outcome，用startpoint nested CV只在inner-OOF上选择
`P(overtake)-lambda*P(collision)`与noop margin。独立outer seeds的frozen selector为target 69、
controls `1/3`；恢复Z4原outer seeds的复核为66、`3/4`，均在低于fixed `5/5` harm时仍低于
fixed target 79；配对`p=0.268/0.154`，应读作未检出优势而不是显著劣于fixed。事后全OOF
选点在独立seed上可读到84，但nested只有69，证明该读法乐观。§39关闭当前tested
budget-constrained outcome selector，不运行validation或PPO；会反传进student GRU的2b仍未测。

**同日此前完成活动是Round Z4-A representation-changing action-response Gate，无actor更新。**
复用Round Z2的456条Austin development task与5,928个late action-outcome标签，按70个ego
startpoint分组五折；treatment只增加当前动作前50步actor可见LiDAR+previous-speed历史GRU。
其三目标层恢复为`68/109、32/46、2/13`，safe controls新collision/overtake loss=`13/21`，
target success `102`不仅未过progress/control门，也低于同监督协议frozen-hidden control的
`104`。§38按预注册关闭该具体历史编码实例；未运行独立validation分支、未接入actor或PPO。

**同日此前完成活动是Round Z3 collision-only BC anchoring独立validation Gate，无branch、无训练。**
从未打开的150条validation完成BC/U42--U45共750条评估，得到7条/6起点稳定collision cohort；
但同raceline、同speed、无放回controls在两个分层确定性不足，plan前fail closed。§37只能判
inconclusive，既不通过也不否决collision-only teacher。

**同日此前完成活动是Round Z2反事实动作Gate，无actor更新。** 456条Austin development
场景先完成456/456 U44精确重放，再运行early/late各12个动作，共10,944条candidate branch；
11,400条compact trace全部通过合同。动作oracle在early对inherited/created/lost三层救回
`93/109、44/46、10/13`，late为`96/109、45/46、7/13`，说明局部可行动作广泛存在；但按
startpoint外推的action-conditioned head在early仅`19/109、11/46、1/13`且误伤17/225 controls，
late为`34/109、17/46、0/13`，并以51次target success落后固定动作baseline的79次。§36按预注册
关闭当前fixed-library action-conditioned与first-action preference形式；prefix-reset没有获得
“只有early有效”的旧程序触发条件，不准入；这不是对reset后PPO训练密度机制的科学否决。
Constrained PPO与MoE架构本身也未被本Gate检验，后者另受当前12-key工程边界排除。

**同日此前完成活动是BC-safe anchoring Gate B，无训练。** Gate A冻结的28条cohort与
28条matched controls完成branch 0、完整BC、steering-only、speed-only共224次CUDA replay；
branch 0的全部动作、状态和LiDAR误差为0，224条trace合同全部通过。完整BC救回C层`10/19`
碰撞且全为overtake，但L层恢复`0/9`、safe controls损失`2/28`次overtake，未通过双分层与
control门。§35按预注册科学失败并关闭本方向；不生成anchor dataset，不进入Gate C/D或formal
训练。当前没有训练或评估进程，validation未运行branch。

**同日此前活动是Round Z0无训练checkpoint平均与固定评估。** U42--U45按预注册float64等权
平均得到严格12-key actor；四图CUDA为`67 collision / 1465 overtake`，相对U44 `62/1478`
双轴变差，且Hockenheim `19/341`低于BC的overtake下限343，故§33按停止规则关闭，不改band或
权重重试。

**2026-08-07此前活动是两项无训练预检，不是新run。** §31在U44 Austin开发面板上检查真值
几何/速率对未来平行侧后碰撞的线性早期可分性，未通过辅助训练前置门；§32随后检查actor输入
到GRU hidden的当前交互几何表征缺口，fold-local维度匹配后hidden在9/9目标上优于输入，
预注册缺口不成立，因此不启动这9个几何/速率目标的辅助表征训练。更早的§30没有建立跨提前
时刻、跨startpoint fold稳定的actor-visible区分，按预注册停在第一道准入门，没有运行动作
响应、PPO credit或训练。此前2026-08-06先用fresh rollout、reward、critic、GAE
和PPO loss复核§23，未稳定复现人工偏好目标的output-head反向梯度，因此否决naive梯度投影；
再否决用单checkpoint episode return推导25% collision-role训练。随后按用户新的四图逐图BC
最低线重判已有模型：U44正式通过，ordinary异线高速重加权U30计数也通过但历史manifest缺
CUDA device，正式候选前仍需固定四图确认。没有产生新actor；详见§24--§27。

**没有未完成的run。** 2026-07-30一轮已全部收口：前向走廊门控时间相关速度噪声完成
45个formal updates；ordinary异线高速重加权比例0.6完成30个formal updates；第一版
重加权因改变same-line份额而在U10主动
停止，另两次更早的45u中断副本已被完整run取代。评测同样完成：40个面板、
22,400 episodes、0 error。

**上一训练轮的三条结论**（详见 §18.6 / §18.7）：

1. **延长到45 updates —— 方向关闭。** 与走廊门控时间相关速度噪声30 updates在三个
面板上无差别；前30 updates的
   checkpoint参数和训练指标一致，已验证是纯延长。
2. **ordinary异线高速重加权（比例0.6）—— 否决。** 它是唯一在跨地图上对production双轴占优的
   臂（54–57 / 1146–1155 对 B 的 80 / 1142），但 near400 碰撞 63–77 是 B 的 2.3–2.8 倍、
   四个 checkpoint 全部高度显著。
3. **新增硬约束：同线暴露量不可移动。** 见 §18.5 第 7 条。

**production 未改变**：仍是 B（`--speed_exploration_mode baseline`，即不传该 flag）。

**当前没有已获准训练臂。** K10/K25和Group13仍属于“未测试、低优先”，不得写成已否决，
也不得仅因尚未测试就自动启动；只有新的机制证据通过独立准入门，或任务分布/控制合同发生
明确变化，才按对应停止与重开规则提出新训练。

### 1.1 一页结论

- 当前唯一 PPO 入口是根目录 `train_ppo.py`，实现集中在 `ppo/`。
- 当前支持四种 critic：`mlp`、`independent_gru`、`priviledge_mlp`、`privilege_gru`。
- Canonical BC actor 是 `pretrained/end2race.pth`，严格 12-key state dict。
- Actor 永远只读取 361D：360 个下采样 LiDAR + previous measured ego speed。
- Privileged observation 是 381D：前 361D actor observation + 后 20D privileged state；actor 仍严格只切前 361D。
- 默认production reward固定为四项：progress、relative progress、首次ego collision
  penalty，以及两车OBB/地图墙距的potential-based risk shaping。Post-pass生产模块和
  CLI已删除；risk纵向尺度固定从YAML读取`0.6m`，不再提供L12 override；
  required-deceleration/escalation只做过离线saved-trace筛选，从未接入production或训练。
- 当前 risk 阈值为纵向 `0.6m`、横向 `0.2m`、墙距 `0.2m`、最大 potential `0.05`。
- Post-pass和L12都已完成离线/训练/固定面板验证，均未通过production验收：Post-pass没有稳定改善甩尾且可造成新甩尾/超车损失；L12只在困难近距regime显著改善，在interval-15目标分布净收益为0，并显著损失near-miss超车。默认保持Post-pass off和risk纵向`0.6m`，详见§16。
- 当前 CLI 默认 formal updates 为30，actor/critic epochs 为`2/5`，GRU/head/critic LR 为
  `3e-6 / 3e-5 / 3e-4`，seed为42，clip range为0.20；target-KL已固定关闭并移除入口。
  §8记录当前有效入口；不要依赖历史flag总数判断接口是否存在。
- **当前 production actor 是 `production U30`**：`privilege_gru`、clip 0.20、30 formal updates、
  seed 42、默认 reward/exploration/pool。权威路径是
  `post-trained/ppo_privilege_gru_clip020/update30/actor.pth`。迁移时验证过四个历史来源
  中的U30 actor逐字节等价；来源容器已清理，SHA-256见**§1.3**。后续引用
  历史表中的`B`、`U30`和`baseline`均指该模型及其固定配置；新的分析统一写
  `production U30`。
- **production U30 的固定面板基线数字（后续一切对照都用这组）**：Austin600
  `14 collision / 366 overtake`；near400 `28 / 325`；hard73 `54 / 12`；三张 held-out
  地图合计1800场景`80 / 1142`。
- 0721 已完成7个受控run和35个Austin600 checkpoint eval，20-update 内 clip 0.15 最好；但延长到 30 updates 后 clip 0.20 收敛到上述 B/U30，成为实际 production 候选。这是单seed强信号，不应写成架构因果已最终证明。
- **当前起点公式下**的BC Austin600为总碰撞`33`（ego-opp 27、ego-wall 5、
  opp-wall 1）、超车`339`、follow`228`、0 error。§10.5里的`22/344/234`属于
  2026-07-22旧起点面板，只能作历史对照，不能与当前B/U30的`14/366/220`直接配对。
- 历史`--target_kl`按SB3 early-stop语义实现，但`0.02/0.04`均未优于关闭状态，
  `0.04`在U5--U20旧Austin面板平均碰撞51.5。production winner始终关闭；当前CLI、
  early-stop分支和专用telemetry已删除，常规approx-KL诊断仍保留。
- 历史boundary-aware hard-neighbor cache实现正确。完整805池fresh-start
  A/B已经完成，但Austin600安全结果差于479池基线：U45碰撞`12 -> 17`，U25+均值
  `13.0 -> 21.4`，因此不进入production。20%分层采样臂在同面板U35/U40/U45也持续
  更差（碰撞`14/11/12 -> 27/20/19`），已否决；10%臂完成45 updates，但仅有U1--U20
  完整eval，仍未定案。用户主动终止该未知方向并退役训练入口，不能写成10%已证伪。
  默认继续使用479池，详见§17。
- Eval trace已补terminal post-step frame；新trace用 `action_applied=false`、`terminal_post_step=true` 标记最后一行。0721旧trace没有该行，旧碰撞标签仍以 `results_multi.json` 为准。
- **速度探索训练已收口**：历史完成5种模式；当前代码只保留逐步独立速度高斯噪声
  (`baseline`)、全局时间相关速度噪声(`temporal_global`)和前向走廊门控时间相关速度噪声
  (`corridor_temporal`)。
  `conditional_temporal`、`conditional_white`（连同其escalating门）与speed-std退火
  已删除但重建合同保留。按当时协议，所有实验均未通过Austin600 + near400联合验收，
  production保持`baseline`。用户后来把正式验收改为四图逐图BC下限；在该新口径下，前向
  走廊门控时间相关速度噪声U44已正式通过，ordinary异线高速重加权U30的历史结果也重新成为
  候选，但后者仍缺CUDA provenance确认，详见§26--§27。该重新判读不删除旧near400副作用；
  但1.0m门宽失去收益，2.0m保持为YAML默认，1.5m未训练，不能写成已证伪。ordinary异线
  高速重加权改由YAML定义并默认关闭。详见§18。
- **Regime无训练审计没有识别出稳定PPO梯度根因。** 人工counterfactual成功动作偏好在
  output head上得到S-O/S-N cosine `-0.962/-0.959`，但fresh真实PPO rollout的U10/U20/U30
  为`+0.663/+0.622/-0.186`，符号不稳定；naive梯度投影因此被否决。动作收益线性探针也没有
  建立hidden对observation的稳定优势。当前保持361D输入、reward和critic，详见§23--§24。
- **已被证否的方向，不要重跑**：Post-pass penalty（§16）、risk L12（§16）、完整 805 hard pool（§17）、speed std 0.25/0.50/退火（§19）、ordinary 起点 50→150（§19）、interval-15 difficult pool（§19）。每条的配对 eval 结果、样本量和 p 值都已写进对应章节，直接引用即可。
- **已被证伪的悲观假设**：那些反复失败的 hard 场景**不是物理不可解的**。334 条 held-out hard failures 用共享 oracle library 全部 334/334 找到无碰撞干预，其中 71.6% 还能保留超车（§20）。所以"BC/PPO 打不过 = 场景不可行"这个说法不成立，不要再用它解释失败。
- 2026-07-30 本次修订时未发现活动的`run.sh`或`train_ppo.py`；进程状态会变化，接手时仍必须重新检查，见§1.0。

### 1.2 当前工作树边界

本轮一次性审计脚本、notebook、NPZ、JSON和counterfactual续跑记录已按用户要求删除；准确
工作树范围仍须重新运行`git status --short`，且`.agents/`修改现在必须显示在该输出中。
禁止reset、checkout或清理未被用户明确点名的状态。2026-07-30已完成一次显式授权的
模型/评测清理：重复、已否决、中断及历史扫描目录已删除，当前只保留HANDOFF §2登记的
canonical/未定案模型与§4.3登记的评测根。剩余`eval_results/`、`post-trained/`、
canonical BC及其他用户文件重新回到受保护状态。

历史清理曾删除大量analysis artifacts；它们大多仍可从历史提交读取，但**本文不依赖这些
文件**：各实验线的结论、样本量、配对增删和p值都已写进正文（§16-§23），原始
JSON/CSV/trace属于可清理产物。所以看到某个分析目录不存在时，正确反应是从本文读结论，
而不是去找文件或重跑评估。

清理前的历史回归集合混有三个非测试文件，属于已知状态而非错误：`probe_hard_neighbors.py`、
`build_outcome_aware_cache.py`、`OUTCOME_AWARE_MERGE_NOTES.md`。

保留run以自己的`run_config.json`、`collision_cache_info.json`和机器可读产物为准；
不得因为历史来源已清理而推翻本文已经记录的实验结论。

### 1.3 模型身份

模型SHA-256的唯一权威登记在`HANDOFF.md` §2，本文不重复。

分析中使用的production baseline是B/U30：

```text
post-trained/ppo_privilege_gru_clip020/update30/actor.pth
```

迁移时曾验证45u extension、structured-exploration control和current-code reproduction
三个历史run的U30 checkpoint与该文件等价；这些来源已清理。U45是另一checkpoint，
必须引用`post-trained/ppo_privilege_gru_clip020/update45/actor.pth`。

### 1.4 事实优先级

发生冲突时按以下顺序判断：

1. 当前源码、Git 状态、实际进程和机器可读 artifacts；
2. 对仍保留的run，其`run_config.json`、`metrics.jsonl`、`episodes.jsonl`和checkpoint；
3. 本ANALYSIS已经固化的历史实验记录与当前实现说明；
4. README 的公共使用说明；
5. Git history 中的旧 GUIDE、旧实验计划和聊天结论。

旧 run 的 `run_config.json` 永远优先于今天的 CLI defaults。不能用当前默认值解释历史 checkpoint。

---

## 2. 当前代码结构

```text
train_ppo.py            # 唯一 PPO 训练入口
evaluate.sh             # 固定 Austin600 调度
eval_multiagent.py      # 双车 deterministic eval + numeric trace
eval_singleagent.py     # 单车多圈 eval，支持 LiDAR beam masking
ppo/                    # 训练实现，见下表
pretrained/end2race.pth # canonical BC actor，禁止覆盖
post-trained/           # run 目录 + collision-cache/（cache 是训练输入，不是分析产物）
eval_results/           # eval 面板输出
```

训练与评估产物数量多且持续变化，**本文不逐个列举也不引用其内部路径**，用 `ls` 现查。
命名规则：`post-trained/<run>` 与 `eval_results/<alias>_<panel>_<Map>` 对应。

**唯一不可随意清理的是 `post-trained/collision-cache/`**：479池是当前训练输入；
805 boundary cache是已退役方向的归档重建输入。后者不再被production训练入口读取，
但删除后若重开历史方向需重新分类。`eval_results/`和历史分析产物都是产品；
相关结论已固化进§16-§20。

`ppo/` 各模块职责：

| 文件 | 当前职责 |
|---|---|
| `train_ppo.py` | 薄CLI、参数检查、五个PPO模块装配和训练入口 |
| `ppo/env.py` | 单logical env、361/381D observation、前向走廊门、一env一worker VecEnv、parent-side调度和auto reset |
| `ppo/policy.py` | Actor adapter、动作分布、四种critic、P20特权状态、三种保留的训练期速度探索 |
| `ppo/reward.py` | Closed-track progress、OBB/map-wall geometry、collision latch、固定四项reward与risk potential |
| `ppo/scenarios.py` | Scenario生成、role-specific no-replacement queues、collision classification/cache、可选ordinary异线高速重加权 |
| `ppo/rollout.py` | Recurrent buffer、critic warm-up、formal actor/critic phases、metrics |
| `ppo/ppo_config.yaml` | Multiprocessing、timing、risk、scenario、前向走廊门距和ordinary重加权配置 |
| `utils.py` | 通用评测工具，以及PPO run记录、JSONL和actor/critic checkpoint保存 |
| `analyze_baseline_reward_seed42.py` | 默认四项reward的固定Austin600合同、分量和风险时机审计 |
| `validate_following_response_reward.py` | Required-deceleration/escalation reward候选的离线因果重放；未接入production |
| `eval_multiagent.py` | 单场景deterministic eval及numeric trace，含terminal post-step frame |
| `eval_singleagent.py` | 单车 10 圈 eval，`noise` 参数是 LiDAR beam masking 不是加性噪声（§19） |
| `evaluate.sh` | 固定Austin600调度和结果汇总 |
| `run.sh` | **当前不存在**：2026-07-30 随历史实验命令一并删除。下一项实验需新建，见§13 |

当前不存在的旧路径只有 `ppo/config.py`、`ppo/buffer.py`、`runs/` 和 `ppo_experiments/`。
**历史实验工具和回归测试在清理前都存在且在用**（run.sh 曾直接调用其中的脚本）；旧版本 HANDOFF
曾把这两个目录写成"不存在"，那是错的。

---

## 3. 运行环境和数值合同

正式本地解释器：

```text
/home/haowei/miniconda3/envs/end2race/bin/python
```

历史确认的主要版本是 Python 3.10、PyTorch 2.7.0+cu128、Gymnasium 1.2.3、stable-baselines3 2.7.1、sb3-contrib 2.7.1。

不要依赖非交互 shell 的裸 `python`；从 repo root 使用绝对解释器。当前 multiprocessing contract 是 Linux `forkserver`。CUDA 初始化前创建 subprocess environments；worker 内将 OpenMP/MKL/OpenBLAS/Torch threads限制为1。

训练数值设置：cuDNN TF32 off、CUDA matmul TF32 off、float32 matmul precision highest、cuDNN benchmark off。

Collection 保持按 logical slot 的 batch-size-one actor/critic execution；training replay 才按 timestep 对 active sequence slots做FP32 batch。不要为了提速静默改变 collection kernel 后仍声称deployment数值合同不变。

---

## 4. Actor 和动作分布

Actor input是 `[360D LiDAR, previous_measured_ego_speed]`，shape 361。主路径是 learned pressure parameter `k` + `speed_mlp` → GRU → output layer。

真正 recurrent state 是 GRU hidden `h`。SB3 recurrent interface 的 cell `c` 是 dummy zero，不进入actor计算或checkpoint。Episode start时hidden清零；episode跨rollout边界时继续携带。

动作分布：

```text
steering: latent Normal -> tanh -> ±0.52 rad, latent std 0.03 frozen
speed: physical Normal, std 0.15 m/s frozen
```

Buffer保存并用同一physical action计算log probability。`log_std.requires_grad=False`。当前 `ent_coef=0` 且 distribution `entropy()` 返回 `None`；未来启用entropy loss前必须补实现和测试。

Actor只训练End2Race GRU和output layer；pressure `k`、speed MLP、log_std和dummy cell冻结。Actor optimizer和critic optimizer完全分离，已验证参数集合没有重叠。

---

## 5. 五种 Critic

CLI准确拼写：`mlp`、`detached_gru`、`independent_gru`、`priviledge_mlp`、`privilege_gru`。

### 5.1 `mlp`

输入是当前361D actor observation，并保留BC表示：

```text
360D LiDAR -> copied trainable pressure k
previous speed -> copied trainable speed_mlp
concat 420 -> Linear(420,60) -> ReLU -> Linear(60,1)
```

它没有recurrent state。Critic自己的 `k`、speed MLP和value head不与actor共享。

### 5.2 `detached_gru`

输入是actor处理当前observation后的post-observation GRU hidden。`hidden_scale=4`时：

```text
stored detached actor hidden 1680D
-> LayerNorm
-> Linear(1680,420)
-> ReLU
-> Linear(420,1)
```

Feature在rollout collection时计算、detach并写入buffer。Actor update后不会重算旧rollout feature；value loss不反传actor。Timeout和rollout-end value使用正确continuation actor hidden处理terminal/final observation。

当前不需要复杂moving-feature fixed probe。先使用pre/post value loss、EV和feature mean/std/abs-max；只有实际异常时再增加定向诊断。

### 5.3 `independent_gru`

独立critic深拷贝canonical BC的 `k`、speed MLP和GRU，全部可训练：

```text
BC pressure + speed representation
-> independent GRU hidden 1680D
-> Linear(1680,420)
-> ReLU
-> Linear(420,1)
```

Buffer保存每个transition的pre-step critic hidden；replay从sequence-start hidden恢复，episode start清零，padding不进loss。Warm-up按完整episode sequence做BPTT。

跨rollout/update存在“旧参数生成的前缀hidden + 新参数继续”的recurrent PPO近似，但rollout内values/returns/GAE自洽、collection/replay一致且没有泄漏。只清critic hidden会破坏正在进行episode的历史；完整burn-in收益不足以证明值得增加复杂度。因此当前不reset env、不清hidden、不增加burn-in。

### 5.4 `priviledge_mlp`

Environment observation是前361D actor observation加后20D normalized pre-action privileged state，总计381D。Actor只切前361D，critic只切后20D：

```text
20 -> Linear(20,120) -> ReLU -> Linear(120,30) -> ReLU -> Linear(30,1)
```

20D顺序：delta_s、relative_lateral、relative_long_velocity、relative_lat_velocity、sin/cos relative heading、ego_speed、ego_yaw_rate、relative_yaw_rate、三项clearance、ego steering/slip、左右body margin、sin/cos track heading error、current/lookahead curvature。

全部来自当前pre-action raw simulator state和静态地图，没有sampled action、next observation、future collision或terminal outcome。Relative velocity在 `|speed| >= 0.5m/s` dynamic branch使用simulator slip angle，低速branch与simulator一致使用kinematic heading。

所有维度已归一化/clip到 `[-1,1]` 或 `[0,1)`；三项clearance使用 `d / (d + half_response)` softsign，不再把risk判定半径当硬截断尺度。Half-response仍为纵向/横向/墙距 `0.6/0.2/0.2m`，但这只控制critic input响应曲线；reward计算和risk阈值没有随归一化修复而改变。OBB clearance是两车外边缘距离；body track margin考虑车身朝向投影。

0721 `privilege_gru` 20个formal updates的telemetry确认三项clearance精确high-saturation均为0；OBB纵向/横向 `>=0.95` 仅约1.0%-4.0%/2.0%-5.2%。左右body margin约一半以上达到高端不是同一种错误：两侧是相对赛道左右边界的互补余量，远侧可饱和而近侧仍保留约束信息。当前决定保持完整P20，不按线性权重代理删除特征。

### 5.5 `privilege_gru`

与 `independent_gru` 共用BC初始化的独立recurrent分支，并在value head前加入零初始化的20D privileged projection：

```text
Linear(1680,420) + Linear(20,420,bias=False) -> ReLU -> Linear(420,1)
```

Actor仍只使用361D；20D特权输入只影响critic。

---

## 6. Scenario、角色和 Collision Cache

当前环境合同：Austin，16 logical envs且一env一worker，100Hz，8秒/800步。Even logical
ranks是collision，odd ranks是ordinary。True ego collision为terminated；time limit为
truncated并bootstrap terminal observation。Opponent-only collision不结束ego episode。

Collision和ordinary各有parent-owned deterministic shuffle queue。每个role全部scenario使用一次后才reshuffle，同role env共享消费。Env-major recurrent minibatch和 `batch_size % (2*n_steps) == 0` 校验保证每个minibatch collision/ordinary transitions相同。

Ordinary panel：`50 startpoints * 3 racelines * 4 speeds = 600`，interval 15。
历史`ordinary_startpoint_count=150`曾扩到1800，但入口已经退役；当前始终固定50。
重建合同和失败边界见`EXPERIMENTS.md` §0.5.1与本文§19.1。

Collision candidates：`100 startpoints * 3 racelines * 4 intervals * 9 speeds = 10,800`，intervals 8/10/12/15，speeds 0.45到0.85。这是 **base** 网格；boundary-aware cache 会在其内部细化出 interval 9/11/13/14 和 speed 中点（§17.2）。

当前default cache：479 ego collision、10,285 other、36 invalid；classification使用8 workers，
耗时4494.83秒。Cache位于
`post-trained/collision-cache/pretrained_end2race_austin_collision_pool_479/`，
严格校验count/order/IDs/config/summary；partial或mismatch会fail closed。

当前缺口：schema 1 cache只记录pretrained绝对路径；同路径模型被覆盖时旧cache可能复用。
只要模型文件可能被覆盖，就必须使用新cache目录或`--reclassify_collisions`，不能仅凭路径
推断旧cache仍匹配。

---

## 7. Reward 和风险几何

总reward：

```text
r = 0.01 * ego_progress_delta
  + 0.02 * (ego_progress_delta - opponent_progress_delta)
  - 2.0 * first_ego_collision
  + gamma * Phi(next) - Phi(current)
```

当前没有第五项或reward运行时override。历史Post-pass是直接即时penalty而非potential
shaping；历史L12只是把纵向分母从0.6改为1.2。两者均已否决并从生产调用链移除，
完整历史实现思路、结果和否决理由见§16。

Progress使用closed-track wrapped delta并对异常单步位移fail closed。

Vehicle risk normalized distance为 `hypot(longitudinal_clearance/0.6, lateral_clearance/0.2)`；wall distance为ego OBB到occupancy-map墙面的clearance除以0.2。取两者最小值后：`Phi = -0.05 * max(0, 1-distance)^2`。

真正terminal使用 `Phi(next)=0`；timeout保留physical next potential并由PPO bootstrap。这是risk credit redistribution，不替代collision penalty。最大potential 0.05不应仅因为episode signed累计小而提高；局部transition梯度和正常超车误激活更重要。

当前 `0.6/0.2/0.2` 来自最小episode grid：相较纵向0.8m，它减少safe overtake/follow激活而collision warning median未下降。不要让risk阈值成为critic arm变量。

每episode保存risk signed/absolute reward、minimum OBB clearance、minimum wall clearance和risk active fraction，足以判断reward是否激活。

---

## 8. PPO、Warm-up 与 Formal Update

当前CLI defaults：

```text
critic = privilege_gru
hidden_scale = 4
n_envs = 16（一 env 一 worker；无独立worker参数）
seed = 42
n_steps = 6400
batch_size = 12800
num_updates = 30
actor_epochs / critic_epochs = 2 / 5
GRU LR / head LR / critic LR = 3e-6 / 3e-5 / 3e-4
steering latent std / speed physical std = 0.03 / 0.15
gamma / GAE lambda / clip = 0.999 / 0.995 / 0.20
clip_range_vf = None
normalize_advantage = true
ent_coef = 0
target_kl = fixed None（CLI与early-stop分支已删除）
vf_coef / max_grad_norm = 0.5 / 0.5
```

以上是核心 PPO 轴。训练分布清理后，`train_ppo.py`保留的相关入口如下；默认实验开关
等价于“关闭”，**一个都不属于production配置**：

```text
# collision pool / 采样
--collision_cache_dir            post-trained/collision-cache/pretrained_end2race_austin_collision_pool_479
ordinary startpoints             固定50
cache actor identity             严格匹配

# exploration
--speed_exploration_mode         baseline                  -> §18
```

前向走廊门控时间相关速度噪声的门距与ordinary异线高速重加权不再占用CLI，改由YAML键
`front_corridor_gate_maximum_gap_m: 2.0`和`ordinary_offline_fast_fraction: null`
定义。旧conditional-temporal和speed-std退火不再属于活动训练入口。

历史`--allow_collision_cache_actor_mismatch`曾作为cache身份检查的逃生舱，只忽略actor
路径而保持其他字段严格；它现已退役。actor身份不匹配时必须使用新的空cache目录完成
重新分类，不得在调用端临时绕过。

每rollout为 `16 * 6400 = 102,400 transitions`。每epoch有8 minibatches；默认每update计划16个actor optimizer steps和40个critic steps。

`num_updates=N` 会收集 `N+1` rollouts：rollout 1只用于critic warm-up，随后执行N个formal updates。Warm-up当前最多30 epochs、patience 3、按collision/ordinary role分别做80/20完整episode sequence split；每epoch记录train/validation loss并恢复最佳critic和optimizer state。

Warm-up后不reset environment/recurrent states。当前没有证据表明全env reset收益值得改变采样连续性；只清critic hidden则语义错误。

Formal update：

```text
fixed rollout/returns/advantages
-> actor phase, critic frozen
-> critic phase, actor frozen
-> full-rollout post-update critic statistics
-> actor/critic checkpoint
```

Critic拟合rollout collection后计算的固定GAE lambda returns；actor update后不重算当前rollout returns/advantages。这是标准PPO old-rollout语义。

历史可选target-KL在每个recurrent minibatch上用有效mask计算：

```text
approx_kl = mean(exp(log_ratio) - 1 - log_ratio)
stop when approx_kl > 1.5 * target_kl
```

触发时跳过当前minibatch的optimizer step并退出本update剩余actor phase，随后critic仍正常
训练。它不是rollback或严格trust region。由于受控实验未显示收益且0.04明显有害，当前
实现已经移除该分支；这里仅保留重建语义。

Timeout bootstrap对feed-forward、detached和independent critic分别使用正确terminal observation与continuation hidden。

---

## 9. 记录、Checkpoint 和结果解释

Run目录：

```text
post-trained/<run>/
  run_config.json
  collision_cache_info.json
  collision_scenarios.json
  ordinary_scenarios.json
  metrics.jsonl
  episodes.jsonl
  checkpoints/critic_warmup.pt
  checkpoints/actor_uXXXX.pth
  checkpoints/critic_uXXXX.pt
  actor_final.pth
```

Output dir必须位于project `post-trained/`内且为空。Actor checkpoint保存后会创建fresh CPU End2Race并strict-load，必须恰好12 keys。Critic checkpoint不能部署。

当前不保存optimizer、scenario queue、environment或recurrent state，因此不能exact resume。中断后可以分析已有checkpoint，但新进程不能称为同一条exact trajectory continuation。

Warm-up metrics包括epochs、best epoch/loss、每epochtrain/validation losses、critic grad norm、wall time和checkpoint。

Formal critic metrics：

```text
value_loss and critic_epoch_value_losses
value_loss_pre_update/post_update
explained_variance_pre_update/post_update
collision_value_loss_pre/post
ordinary_value_loss_pre/post
collision_explained_variance_pre/post
ordinary_explained_variance_pre/post
value_prediction_post_mean/std
return_mean/std
critic_grad_norm_mean/max
```

Formal actor metrics包括policy gradient loss、KL mean/max、clip fraction mean/max、actor
grad norm mean/max。target-KL专用字段随退役功能一并删除；历史run可能仍包含这些字段。

Episode metrics包括episode count/steps/return、per-role return和relative position、collision/overtake/follow counts、reward components、risk telemetry、OBB/wall minima。Detached额外记录feature mean/std/abs-max；privileged记录每维normalized min/max/mean/std和saturation fraction。

`rollout_policy_update=k-1`、`checkpoint_update=k` 表示formal update k的episodes由actor k-1收集，不能用该rollout直接评价actor k。

不建议再记录Python/CUDA/RNG dumps、每层parameter delta、calibration bins、Spearman表、大量hidden probes或重复raw feature统计。只有出现具体异常时才加定向诊断。

Critic loss和EV不等于actor产品性能。训练rollout stochastic且对应上一checkpoint。Actor必须通过预先冻结的deterministic evaluator schedule评价。当前使用固定Austin600，在U1/U5/U10/U15/U20和长训U25/U30评估；禁止看日志后临时换面板或只挑一个winner数字。

启用 `SAVE_TRACE=true` 的新eval每个NPZ最后追加一行post-step state。该行action数组为占位零值，必须结合 `action_applied=false` 解读；`terminal_post_step=true` 精确标识terminal/timeout frame。2026-07-22 BC Austin600已验证600/600个trace都采用新schema，普通8秒episode通常为802行（801个applied-action frame + 1个terminal frame）。0721目录是在修复前生成，仍是旧schema；不要混用两种行语义。

### 9.1 固定面板与数据口径

后续章节反复引用四个面板。它们的角色、选择来源和局限不同，不能只看名称或把样本量
当作可互换：

| 面板 | 固定构成与选择来源 | B/U30 `collision / overtake` | 合法用途 | 主要边界 |
|---|---|---:|---|---|
| Austin600 | Austin当前50个circular eval起点 × 3 opponent racelines × 4 speeds `0.5/0.6/0.7/0.8`，interval 15 | `14 / 366` | 快速同面板主指标、checkpoint轨迹 | tuple不在训练pool内，但训练ordinary起点到最近eval起点中位仅约2.1m，而8秒episode行程中位约45m；不是独立地图留出 |
| near400 | U30在未见起点候选中**未碰撞且最小OBB间距在`[0,0.1]m`**的场景，预先哈希抽取400条；interval `8/10/12/15`、speed `0.45–0.85` | `28 / 325` | 贴身但本来可通过的KPI守门，捕捉新造碰撞/丢超车 | 是按U30行为筛出的困难成功集，不代表自然发生率；不得用它单独选模型 |
| hard73 | 同一未见起点instrument里，U30分类为ego collision的334条中仅取interval-15的73条 | `54 / 12` | 检测困难工况专门化、查看目标regime | 失败条件筛选导致高基率；不是production验收指标，改善经常与Austin/near恶化同时出现 |
| 跨地图1800 | Hockenheim、MoscowRaceway、Nuerburgring各自50起点 × 3 racelines × 4 speeds，共每图600、interval 15 | `80 / 1142` | Austin训练后的地图迁移 | 只有3张固定地图，不代表赛道总体；它在CT-v2门控设计后已参与设计，不能再作为corridor系列的干净最终留出 |

near400/hard73来自同一21,600候选instrument：200个新起点按物理区块预先拆成100个train、
100个held-out，候选网格为3 racelines × 4 intervals × 9 speeds；train/held-out起点不交叠，
选择时只运行冻结B/U30，没有查看L06/L12或后续exploration臂。held-out侧得到334个
U30 collision、400个near-miss；train侧得到468个collision、400个near-miss。
因此它们适合做**预先冻结的困难分布诊断**，但不能估计自然碰撞率。

L12章节另有`cache372`、`heldout-near600`和完整`hard334`等一次性诊断集合。它们用于回答
特定迁移问题，不是新增production gate；定义和结果只在§16.8使用。Oracle章节沿用
`hard334`，但只做动作可达性诊断，见§20。

高层表若写`collision`，默认沿用evaluator headline总碰撞；当前Austin600的B有
13个ego collision和1个opp-wall，因此显示14。凡是评估actor安全责任、碰撞模式或
McNemar配对，正文会明确写`ego collision`并排除opp-wall；不能把对手自己撞墙算成
ego新造失败。`overtake/follow/ego-opp/ego-wall/opp-wall`之和必须等于面板总数。

每个完整面板的最低数据合同是：scenario数与唯一key一致、0 error、有限数值、每场景一条
trace、结果与trace key-set一致、collision marker与轨迹终态一致；新schema还要求最后一行
为`terminal_post_step=true`且`action_applied=false`。同场景A/B必须报告
`removed/created`，超车必须报告`lost/gained`，并用精确McNemar检验；多个checkpoint重复
同一批场景时不得把行数冒充独立样本。

---

## 10. 0721受控实验与Austin600结果

7个run均已训练20 updates并完成U1/U5/U10/U15/U20 deterministic Austin600。共同参数为seed 42、`n_steps=6400`、actor/critic epochs `2/5`、GRU/head/critic LR `3e-6/3e-5/3e-4`、std `0.03/0.15`、target-KL off、同一canonical BC和collision cache。每组只改变表中实验轴。

### 10.1 Critic对照

| run | U1 | U5 | U10 | U15 | U20 | U5-U20 collision均值 | last-5 EV / collision EV |
|---|---:|---:|---:|---:|---:|---:|---:|
| `independent_gru_0721_base` | 19/345 | 19/339 | 27/339 | 29/330 | 34/329 | 27.25 | 0.911 / 0.904 |
| `privilege_mlp_0721_base` | 24/343 | 25/345 | 25/347 | 21/349 | 25/360 | 24.00 | 0.694 / 0.647 |
| `privilege_gru_0721_base` | 16/350 | 18/343 | 18/339 | 14/349 | 14/349 | 16.00 | 0.912 / 0.905 |

单元格为 `collision/overtake`。在本轮受控参数下，`privilege_gru` 是唯一同时给出最低checkpoint碰撞数、较稳定后段结果和高critic EV的结构；`independent_gru` 随更新明显恶化，`privilege_mlp` 的P20输入没有弥补缺少时序LiDAR分支。当前合理结论是保留 `privilege_gru` 为主候选，而不是宣称所有设置/seed下必然优于其他critic。

### 10.2 Batch size对照

| batch | U1/U5/U10/U15/U20 collision | U5-U20均值 | KL mean中位数 / minibatch KL max | last-5 EV |
|---:|---|---:|---:|---:|
| 12800 | 16/18/18/14/14 | 16.00 | 0.0327 / 3.094 | 0.912 |
| 25600 | 18/22/21/22/16 | 20.25 | 0.0036 / 1.245 | 0.872 |
| 51200 | 22/17/15/17/21 | 17.50 | 0.0177 / 0.553 | 0.830 |

大batch降低了KL尾部但减少每update优化步数并降低critic EV，Austin600没有得到稳定收益。当前保持12800；不能把单seed下12800与51200的1.5次平均碰撞差解释成确定效应。

### 10.3 Clip range对照

| clip | U1/U5/U10/U15/U20 collision | U5-U20均值 | KL mean中位数 / minibatch KL max |
|---:|---|---:|---:|
| 0.10 | 28/23/17/28/20 | 22.00 | 0.0461 / 2.874 |
| 0.15 | 16/18/18/14/14 | 16.00 | 0.0327 / 3.094 |
| 0.20 | 24/23/22/18/14 | 19.25 | 0.0591 / 4.324 |

20 updates内0.15总体最好；0.10波动且均值较差；0.20后段持续改善并在U20达到14，因此被保留为30-update长训挑战者。Clip和KL不是单调对应关系，不能仅按KL max挑clip。

### 10.4 Checkpoint身份漂移与公共难例

`privilege_gru_0721_base` 的U15/U20碰撞总数都为14，但只共享8个场景：U20修复6个，同时新产生6个，碰撞集union为20、Jaccard为0.40。因此正确表述是“总数稳定、失败身份仍漂移”，不是“同一批失败稳定在14”。Target-KL可以限制持续策略漂移，但不能单独教会避碰，也不会回滚已造成尖峰的前一步。

35个checkpoint panel共出现92个不同碰撞场景。以下3个在35/35均碰撞，另1个为30/35：

```text
ol0_e213_o221_s0.7    35/35
ol1_e1497_o1512_s0.5  35/35
ol2_e727_o745_s0.5    35/35
ol1_e1368_o1383_s0.6  30/35
```

它们是公共难例，不等于已经证明物理不可行。完整hard-neighbor collision sampling
后来已完成独立A/B并因安全结果恶化而否决；该结果不能与LR/clip/target-KL实验混写，
也不能反推所有低比例或其他fixed-pool方案都已被证明无效。

### 10.5 BC基线与解释边界

2026-07-22修复后BC Austin600结果为22 collision、344 overtake、234 follow、0 error。0721最好的 `privilege_gru` U15/U20是14/349，相对BC有明确面板内改善。

Austin600不属于训练的collision/ordinary scenario pools：实物核对显示eval的50个ego indices与0721 ordinary 50个、collision cache 81个ego indices交集均为0，600个ordinary完整tuple与eval完整tuple交集也为0。它可继续作为固定快速checkpoint面板；但所有上述结论仍是单seed、固定面板内证据。不能用同一总数掩盖失败身份变化，也不能从critic EV、training rollout或单个checkpoint直接推出产品泛化。

---

## 11. 最近 Review 和最小验证

2026-07-20的collection/replay审查仍有效：actor/critic参数集合无重叠、log_std冻结，recurrent padding不进loss，361D actor slicing不受privileged suffix影响。历史12D/373D是已废弃中间合同；当前唯一合同是P20/381D。

2026-07-21 commit `b82e22b` 修复clearance归一化。当前metadata版本是 `p20_clearance_softsign_v2`；修复只发生在critic feature extraction，reward的OBB/墙距和potential计算未改。真实0721 telemetry确认三个clearance不再精确饱和到1。

2026-07-22 commit `bdeec48` 增加：

1. 历史`--target_kl` CLI、有限正数校验和model传递；
2. 与本机 stable-baselines3/sb3-contrib 2.7.1一致的masked approximate-KL公式及 `1.5 * target` early stop；
3. planned/completed actor step及触发位置metrics；
4. eval terminal post-step trace、`action_applied` 和 `terminal_post_step` 标志。

该target-KL功能现已退役；eval terminal trace修复与它是同一历史commit但彼此独立，terminal
post-step合同继续保留。

清理前共有16个`test_*.py`，覆盖 postpass（含独立 oracle 交叉校验和 episode replay）、
risk 纵向 A/B、hard neighbors、outcome-aware、fixed pool、structured/speed exploration、
speed std 退火、ordinary 起点扩展和多个分析脚本。2026-07-29用正式conda环境重跑当前
6个reward相关模块，结果为 **34 tests：29 passed / 5 skipped**；5项skip是预期路径下
缺少旧Austin600 terminal-trace fixtures，不是测试失败，完整600-trace行为评估另有
机器结果。Hard-neighbor 5/5、exploration 26 tests等仍是各专题当时的实现证据，
不能冒充当前HEAD全量结果。当前改动必须重新运行相关范围后才能声明最新green：

清理前的全套验证通过标准与重建顺序见`EXPERIMENTS.md`；源码清理后不得把历史通过状态
表述为当前代码已重新执行通过。

清理后仍保留一个可直接执行的精简回归入口：
`scripts/test_screen_reward_candidate.py`。它覆盖reward候选合规门禁、GAE/损失量级和独立
Post-pass oracle的核心合同；2026-07-30直接运行与`unittest discover`均为
**55/55通过**。这不等于原16个测试模块仍完整存在。

---

## 12. 剩余问题与优先级

### 12.1 当前执行边界

1. 本次检查未发现活动`run.sh`或`train_ppo.py`；启动新实验前仍需重查（§1.0）。
2. **`post-trained/` 下保留的canonical模型根目录全部视为已完成或已封存，一律不得续写。**
   hard-neighbor 10%虽然尚未形成统一判决，也只能读取已有checkpoint或另建新输出目录；
   `train_ppo.py` 要求output dir为空，禁止绕过。
3. `run.sh` 已整体删除，仓库当前没有训练命令入口（§13）；
   BC Austin600 已完成，不要重复覆盖。
4. 每个checkpoint按预注册的 U1/U5/U10/U15/U20 及长训 U25/U30 eval；training metrics
   不能替代actor eval。新实验的验收面板见§15 第10条。

### 12.2 建议修复

1. Schema 1 default cache仍只有模型路径；新schema 2 hard cache会校验actor、base cache、
   map/raceline与planner config的构建身份。不要用schema 2的保护反推schema 1已修复。
2. 历史测试集合已经很大，但**缺一个覆盖五 critic collection/replay 和 risk geometry 的核心
   回归文件**——现有测试都是围绕单个实验特性写的。补这个文件价值最高。
3. 历史测试集合混入了三个非测试文件（§1.2）。`unittest discover` 不会收集它们，但会让
   集合语义变模糊；若重建，应将其归入实验工具，不要混入回归测试。

### 12.3 非阻断，按需再做

1. Exact resume：只有明确需要无人值守断点续训时才保存optimizer/queue/recurrent/env state。
2. Target-KL的实验开关已具备；正式对照前只需取消Group 6对应注释，不要同时改reward、采样或P20。
3. GRU stale hidden：接受为recurrent PPO近似，不要为严格比较强制reset物理episode。
4. Privileged输入：当前保持完整P20，不再基于单次weight×std代理删维。
5. Following-response required-deceleration/escalation reward继续冻结：它只完成
   Austin600 saved-trace离线选择性筛选，未通过训练A/B准入，也从未接入production。
   其中closing time只是沿赛道相对进度与OBB前向间距构成的局部线性耗尽时间，不得称为
   完整物理TTC。完整805池和20%分层臂已经结束并被否决；若未来补齐10%的统一
   eval，只能解释为采样比例实验，不能同时改变reward、LR、clip、target-KL或其他轴。
6. Speed exploration 已收口。已有证据关闭的是：speed std、corridor gate 宽度、门控覆盖率、
   ordinary 采样权重，以及 45u 延长（§18.6 实测与 u30 无差别）。
   temporal hold 只实际运行过 K=50，K10/K25 从未训练或评估，因此不能写成“hold 已扫完”；
   但按§18.5 第 8 条它属于同一类“强度旋钮”，先验很低。若要开启只能在新目录中预注册
   单变量 K 对照并做区间评估。
7. 已知的 evaluator 陷阱：全量 trace 路径按 actor 文件名 stem 命名，**不同 run 的同名
   checkpoint 会互相覆盖 NPZ**（outcome JSON 不受影响）。多臂对照必须像
   structured exploration 那样给每臂唯一 alias（`B_u0030`/`T_u0030`/…）。

---

## 13. `run.sh` 的性质和验收合同

2026-07-30 先把旧 `run.sh` 的历史命令、`if false` 块、shell函数和自动checkpoint
选择器全部移除，随后把文件本身一并删除。**仓库当前没有训练命令入口。**
下一项实验在完成预注册后新建 `run.sh`，只写该实验一条可直接复制运行的显式命令；
不要去找一个"空队列桩"，它不存在。

历史命令不再靠注释保存在执行入口中。旧分段的状态和收口位置如下；表中的“未运行”
和“无权威结论”必须原样保留，不得写成已否决或已完成验证：

| 已清理的历史分段 | 轴 | 状态与收口位置 |
|---|---|---|
| Groups 1-3 | 0721 critic / batch / clip 对照 | §10 |
| Group 4 | actor LR 1/3/5 档 | §10、§19 |
| Group 5 | clip 0.15 / 0.20 延长到 30 updates | §1.1（产出 B/U30） |
| Group 6 | target-KL 0.02 / 0.04 | §11 |
| Groups 7-8 | 45-update 延长、clip 0.25 边界 | §19 |
| Group 9 | 完整 805 hard pool | §17，已否决 |
| Groups 10-11 | 20% / 10% 分层 hard 采样 | §17；20%已否决，10%未定案 |
| Group 12 | critic LR 5e-4 选择器 | §19 |
| Group 13 | GRU/head LR 解耦 2×2 | 注释，未运行 |
| Post-pass 段 | `q²` / `q` 三臂 | §16，已否决 |
| risklon 段 | L06 / L12 对照 | §16，已否决 |
| interval15 difficult 段 | fixed collision pool | §19，已否决 |
| structured exploration 段 | B / C / T / CT 四臂 | §18 |
| speedstd / anneal 段 | std 0.25、0.50、退火 | §19，已否决 |
| ordinary150 段 | 起点 50 → 150 | §19，已否决 |

前向走廊门控时间相关速度噪声及之后的历史实验当时由独立launcher驱动；其实现与结果
已分别固化在 `EXPERIMENTS.md` 和§18。历史run的真实参数应读取对应
`run_config.json`，不能从今天的CLI默认值推断，也不能把已清理的命令重新粘回 `run.sh`
作为“记录”。

新实验按 `.agents/GUIDE.md` §3 执行：优先复用现有入口，新建 `run.sh` 并在其中放置一组
可直接复制的显式命令；不使用shell函数、动态数组、自动模型选择器或隐藏checkpoint
排名。实验结束或明确放弃后，先把状态和结果写入HANDOFF/ANALYSIS，再恢复空队列保护。

接手时先检查，不要重新执行脚本：

```bash
git status --short
git rev-parse HEAD
pgrep -af '[r]un\.sh|[t]rain_ppo\.py|[e]val_multiagent\.py|[e]valuate\.sh' || true
ls post-trained/ | sort
for d in post-trained/ppo_*/; do printf '%-70s %s\n' "$d" "$(wc -l < $d/metrics.jsonl 2>/dev/null)"; done
```

最后一行是判断常见fresh-start run是否完成最快的方法：`metrics.jsonl` 行数应为
`1 + num_updates`（1个warm-up row + N个formal rows）。走廊门控时间相关速度噪声
45 updates完整run当前为
46行、含`actor_u0045.pth`和`actor_final.pth`；更早的同名中断副本已经被完整run取代，
不得再按“只有6行”解释当前目录。

每个 run 最低验收：`metrics.jsonl` = 1 warm-up + N formal；formal update 连续；
actor/critic checkpoints 完整；`actor_final.pth` strict 12-key；metrics finite；
对应 eval 目录和 `results_multi.json` 数量完整；所有 worker 退出。Target-KL run 还必须
核对 planned/completed actor steps 和触发字段，而不是只看 KL mean。

---

## 14. 过时 GUIDE 清理记录

2026-07-20删除：

```text
.agents/End2Race_Main_Branch_Cleanup_Guide_2026-07-19.md
.agents/End2Race_Main_Cleanup_and_Hardcoded_Critic_Pipeline_Guide_2026-07-19.md
.agents/End2Race_PPO_Critic_Experiments_Common_Fixed_Parameters.md
.agents/End2Race_PPO_H1F_p50_40Update_LongRun_Trend_Execution_Guide_2026-07-18.md
.agents/End2Race_PPO_Pool_Mainline_and_H2_Robustness_Implementation_Guide_2026-07-18_REVISED.md
```

Cleanup guides基于 `main@1d404a4`，要求只保留单一C0或每次改源码切critic，已被runtime五critic selector取代。Fixed-parameter contract仍写H1 482 pool、8 critic epochs、旧warm-up和无risk reward。H1F/Pool/H2 guides依赖已删除的 `ppo/config.py`、H1 manifests、sustained exploration和旧实验目录，不能执行。

旧HANDOFF内嵌13,000多行历史GUIDE全文，导致旧状态和当前事实混杂；用户明确要求移除过时内容，因此不再保留逐字archive。仍有效原则已吸收到本文：canonical actor不可覆盖、actor checkpoint保持12-key、训练与产品评价分离、失败不静默换参数、role/minibatch保持可比、run config与checkpoint身份清楚、不从training rollout挑winner。

需要历史细节时查Git history，不要把旧GUIDE复制回 `.agents/` 作为current authority。

---

## 15. 下一会话检查清单

1. 先读§1.0（最后活动状态）和§1.1，再跑 `git status --short`、`pgrep`、§13 的 metrics 行数循环。
2. **在提出任何新实验前先读§16/§17/§18/§19 的否决表**。已经有 6 个方向被配对 eval 否决；
   重复它们不需要再训练或再评估，直接引用结论即可。
3. 以各run的 `run_config.json` 解释参数；当前defaults不能反推0721或更早checkpoint。
4. 复用collision cache前确认canonical actor未被同路径覆盖；否则使用新的空cache目录分类。
5. 训练中检查warm-up row、formal rows、KL、role-specific EV/loss和checkpoint完整性。
6. 结果同时看总数和逐场景身份；总数相同不代表同一批失败。任何新臂都要报
   removed/created 和配对 McNemar p，不能只报净变化。
7. 新eval trace认 `terminal_post_step`；旧0721 trace的碰撞终帧依赖 `results_multi.json`。
   多臂对照必须给每臂唯一 actor alias，否则 NPZ 会互相覆盖（§12.3 第7条）。
8. 一次只改一个轴。已有的失败案例里，最难解释的都是同时动了两个轴的。
9. 修复代码时保持四critic、361/381D slicing、P20 softsign、GAE/timeout和12-key actor contracts。
10. 验收面板是固定的四件套：Austin600（主）、near400（超车 KPI 底线）、hard73（特化信号，
    **不是验收指标**）、三张 held-out 地图 1800（泛化）。不要看完日志再换面板。

当前交接终点：P20归一化、eval terminal frame和四critic继续保留；target-KL、
boundary-aware/outcome-aware活动源码、speed-std退火及旧conditional-temporal入口已在
保存重建合同后删除。当前探索代码保留逐步独立速度高斯噪声、全局时间相关速度噪声和
前向走廊门控时间相关速度噪声三种mode。实验侧，reward
（Post-pass、L12）、pool（805 hard、fixed interval-15）、采样（ordinary 150）、
噪声尺度（std 0.25/0.50/退火）四类共 6 个方向已被配对 eval 否决；speed exploration
的逐步独立、条件白噪声、全局时间相关、条件时间相关和走廊门控时间相关五组实验已完成，
但都没通过Austin+near验收，只有走廊门控时间相关速度噪声拿到显著的
跨地图收益。2026-07-30 又补完两臂：45u 延长与 u30 无差别（方向关闭），ordinary 异线高速
ordinary异线高速重加权在跨地图上首次对production双轴占优但被near400否决。
production保持逐步独立速度白噪声U30和
全默认配置，无未完成 run，见§1.0 与§18.6/§18.7。

---

## 16. Reward专项：默认四项、Post-pass、L12与following-response候选（2026-07-29）

### 16.1 当前生产决策

| 实验项 | 本质 | 当前实现状态 | 实验证据 | production决策 |
|---|---|---|---|---|
| 默认四项合同 | progress、relative、首次ego collision penalty、potential-based risk | production默认；Austin600分量/边界/时机已审计 | 600 episodes/traces、0 errors；分量求和、一次性碰撞罚、risk边界和Post-pass全零均通过 | 保持不变 |
| Post-pass | 新增的、超车后ego车尾接近对手时的直接即时penalty | 历史上曾接入完整调用链并支持`q²`与`q`；现已删除模块、CLI和遥测 | `q²`两条训练臂甩尾均恶化；`q`只有方向性改善且损失12次超车 | 已退役；不再靠加权重/LR继续调 |
| L12 | 现有risk potential的纵向安全尺度`0.6 -> 1.2m` | 历史CLI override已删除；production只读YAML的`0.6` | 困难近距池和未见起点上有真实收益，但interval-15净收益为0，near-miss超车显著下降 | 已退役；production保持L06 |
| Following response | 以required relative deceleration及其escalation识别同线追近 | 仅saved-trace离线候选；未接reward调用链、未训练 | 目标覆盖6/7，但clean OL1 follow误触21/192，且无same-corridor成功超车正对照 | 未通过A/B准入；保持离线 |

因此当前默认reward仍是四项：

```text
r_default = r_progress + r_relative + r_collision + r_risk(L=0.6)
```

Post-pass与L12的生产实现已删除；本文和`EXPERIMENTS.md`保留足以理解/重建历史实验的
公式、参数和结果。L12证明了risk shaping能改变行为，但没有证明这种改变符合目标分布。
§18的五组速度探索实验改的是exploration分布，不是reward或pool，两条线互不构成对方
的翻案证据。

### 16.2 默认四项reward：固定面板合同与时机审计

这项工作没有改变reward，而是先回答两个问题：当前默认四项是否按合同计算，以及现有
risk potential为什么没有解决残余甩尾/OL1碰撞。审计对象是B/U30的固定Austin600：
权威actor为`B/U30`的`actor_u0030.pth`；
600个唯一scenario、600条numeric trace、0 error，结果集与trace key-set完全一致。
reward合同是`progress=0.01`、`relative=0.02`、首次ego collision=`-2.0`、
`gamma=0.999`、risk `L/lateral/wall/max=0.6/0.2/0.2/0.05`。

合同检查全部通过：

| 检查 | 结果 |
|---|---|
| `reward_total`等于四个默认分量之和 | 600/600通过 |
| ego collision penalty严格一次性 | 13/13 ego collision通过 |
| 历史Post-pass关闭时所有transition严格为0 | 通过；该接口随后已删除 |
| risk potential始终非正、有界，terminal/timeout语义一致 | 通过 |
| outcome marker | 366 overtake / 220 follow / 11 ego-opp / 2 ego-wall / 1 opp-wall |

13个ego collision分解为7个post-overtake rear、4个rear-end和2个wall，其中4个满足
fishtail定义，8个发生在OL1 scenario。现有risk不是“没有激活”：11个目标碰撞都曾
激活risk；问题是时机和局部作用量：

| 机制指标 | B/U30 |
|---|---:|
| fishtail最后连续risk激活提前量P50 | `0.265s` |
| yaw-rate 50deg/s起势提前量P50 | `1.145s` |
| yaw50起势到连续risk的延迟P50 | `0.820s` |
| OL1非fishtail连续risk提前量P50 | `0.450s` |
| 目标碰撞最后1s negative-risk magnitude P50 | `0.0480` |
| 目标碰撞最后1s net risk shaping P50 | `+0.00070` |
| 同窗口positive relative reward P50 | `+0.0318` |

因此现有risk实现和potential望远镜性质没有错误；对于甩尾，clearance是偏晚的后果信号，
而不是yaw/slip起势信号。这个审计直接动机化了后面的Post-pass、L12和following-response
离线筛选，但它本身只是在单seed、已参与设计的固定Austin600上做描述性诊断，不能证明
任何参数改动训练后会改善策略，也不能作为跨地图泛化证据。

### 16.3 Post-pass：要解决什么

目标问题是：ego已经完成超车后，ego车尾重新靠近后方opponent，特别是超车后甩尾
导致的rear collision。它刻意不采用笼统proximity penalty，因为“距离近但没有继续
恶化”的赛车状态不应持续受罚。

历史调用链（现已全部从production删除）：

```text
train_ppo.py
  --postpass_penalty                    # 默认False
  --postpass_proximity_power {1,2}      # 默认2；仅开Post-pass时可改
-> CentralScheduleSubprocVecEnv
-> End2RaceGymnasiumEnv
-> PPOTransitionReward
-> FixedPostpassPenalty (ppo/postpass.py)
```

每个transition严格按以下顺序：

1. 保存`previous_relative_position`;
2. 用前后投影progress计算ego/opponent delta；
3. 更新`current_relative_position`；
4. 更新`opponent_collision_latched`；
5. 用previous ego pose、current ego pose、current opponent pose计算Post-pass；
6. 与原四项直接相加。

Reset状态：

```text
entered = initial_relative_progress > 0.05m
cleared = False
penalty_used = 0
```

正常场景ego从后方起步；当relative progress从`<=0.05m`跨到`>0.05m`时进入
Post-pass phase。ego车尾相对opponent的signed rear gap达到`0.60m`后永久cleared。
opponent已经发生碰撞时关闭该项，避免把对手自身失控归因给ego。

### 16.4 Post-pass：几何、公式与固定参数

历史模块`ppo/postpass.py::FixedPostpassConfig`使用以下固定参数，而不是常规超参；
该模块现已删除：

| 参数 | 值 |
|---|---:|
| pass margin | `0.05m` |
| safe rear gap | `0.60m` |
| activation rear-half clearance | `0.20m` |
| closing deadband | `0.10m/s` |
| maximum ego-induced closing time | `0.75s` |
| weight | `0.25/m` |
| per-step cap | `0.005` |
| per-episode cap | `0.05` |

先把current opponent pose固定，分别计算previous/current ego rear-half OBB到该
opponent OBB的clearance：

```text
c = max(0, clearance(previous_ego, current_opponent)
           - clearance(current_ego, current_opponent))
v_close = c / 0.01
t_close = current_rear_half_clearance / v_close
```

这不是完整的两车物理TTC，因为故意固定了current opponent，只提取
ego-induced rear closing；代码和metadata使用
`ego_induced_closing_time_s`，禁止在论文中泛称TTC。

触发条件必须同时满足：

```text
phase entered and not cleared
opponent_collision_latched == False
rear_half_clearance < 0.20m
v_close > 0.10m/s
t_close <= 0.75s
```

定义：

```text
u = clip((0.60 - signed_rear_gap) / 0.60, 0, 1)
q = clip((0.20 - rear_half_clearance) / 0.20, 0, 1)
basis = c * u² * q^p
penalty = min(0.25 * basis, 0.005, 0.05 - penalty_used)
r_postpass = -penalty
```

`p=2`是原始`q²`臂，`p=1`是后来测试的q-linear臂。二者门控、phase、caps完全
相同，只改变已触发后的幅度。该项是直接改变行为偏好的penalty：不写成potential
difference、不乘额外gamma、不在terminal refund、不增加PPO loss或特殊advantage；
现有GAE自动传播。

`RewardResult.to_info()`逐transition暴露rear gap、rear-half clearance、
ego-induced closing speed/time和penalty basis等物理量；当前持久化到episode records/
`metrics.jsonl`的训练遥测则是reward/episode penalty、phase active fraction、
trigger episode rate/steps和首次触发时刻/距终止提前量。不要把逐步`info`字段写成
已经持久化的训练时间序列。

### 16.5 Post-pass：验证和结果分布

聚焦回归测试在2026-07-29以正式conda环境重跑：

```text
34 tests: 29 passed, 5 skipped
```

复现命令：

```bash
/home/haowei/miniconda3/envs/end2race/bin/python -m unittest \
  test_postpass test_postpass_episode_replay \
  test_postpass_reward_integration test_risk_longitudinal_ab \
  test_compare_postpass_formulas test_compare_risk_potential_variants
```

命令范围是`test_postpass`、`test_postpass_episode_replay`、
`test_postpass_reward_integration`、`test_risk_longitudinal_ab`、
`test_compare_postpass_formulas`和`test_compare_risk_potential_variants`。通过项覆盖：
独立oracle交叉校验、256组随机几何交叉校验、episode cap/reset、
opponent-collision抑制、`q²/q`只改幅度、默认off保持原四项、terminal不refund、
CLI/run metadata和训练遥测。5项skip是旧Austin600 terminal trace fixtures当前不在
预期路径，不等于失败；后续完整600 trace评估已经实际完成。

三个训练臂都从同一B/U30 actor开始、10 formal updates、clip`0.20`、gamma`0.999`、
GAE`0.995`、speed std`0.15`、steering latent std`0.03`、Post-pass以外默认reward保持
L06。critic LR均为`3e-4`；“LR1/LR3”指actor GRU/head LR：

| 臂 | GRU/head LR | `p` | 评估checkpoint | provenance边界 |
|---|---:|---:|---|---|
| LR1/q² | `1e-6 / 1e-5` | 2（旧run config未显式序列化，按固定默认） | U10 | 复用BC分类479池，显式允许actor-path mismatch |
| LR1/q | `1e-6 / 1e-5` | 1 | U10 | 同上 |
| LR3/q² | `3e-6 / 3e-5` | 2 | U10 | 同上 |

所以三臂彼此可比较固定面板行为，但第一条training-summary中的natural continuation
不是严格因果control；collision pool也不是按各treatment actor重新分类。

最早的宽门控先在4个Austin600 panel（BC、U30、普通U45、hard U45）共2400 episodes/
1,903,873 transitions上离线扫了96个设置：

```text
pass margin: 0/0.05/0.10/0.30m
safe rear gap: 0.30/0.60m
closing deadband: 0/0.05/0.10/0.15/0.20/0.30m/s
clear mode: latched/reactive
```

按“先最大化26个既有primary-tail捕获，再最小化follow误触/总penalty/触发范围”选出的
`pass0.05_safe0.30_dead0.20mps_latched`捕获26/26 reference tails，first-signal lead
P50`0.315s`，但同时触发897/1424成功超车（62.99%）、5/914 follow和27/62全部碰撞，
929/2400 episodes至少触发一次，活跃步占4.04%。这证明宽门控有覆盖但选择性不可接受；
reference tail只是历史heuristic，不是ground-truth接触点标签。

另有一个只用于剂量比较的离线targeted replay，把相同gate下的幅度写成
`no_q`、`q-linear`、`q²`三式。它只覆盖每个checkpoint的9个follow和15个overtake
targeted episodes，证明`q²`在相同trigger上比`q`/`no_q`显著更浅；`no_q`从未接入
production，也没有训练臂，不能把这项小面板重放写成第四个策略实验。

四个checkpoint合并后，该targeted replay共有36个follow和60个overtake；三式的gate
完全相同，overtake均为10/60 episodes、138 steps触发，follow均为0/36。仅幅度不同：

| 离线公式 | 60个overtake总penalty | 每overtake平均 |
|---|---:|---:|
| `no_q` | `0.15099` | `0.002516` |
| `q-linear` | `0.04026` | `0.000671` |
| `q²` | `0.01711` | `0.000285` |

第一条`q²` continuation共10 updates、1,443个训练episodes：

| 训练量 | 结果 |
|---|---:|
| 触发episodes | `393/1443 = 27.23%` |
| 全episode平均penalty | `0.001881` |
| 触发episode平均penalty | `0.006908` |
| Post-pass绝对总量 / collision绝对总量 | `2.7148 / 650 = 0.4177%` |
| early U1-U3 collision rate | `21.16%` |
| late U8-U10 collision rate | `22.92%` |
| 配对early→late碰撞 | 消除7 / 新造15，`p=0.134` |
| stage-aligned no-Post-pass vs treatment | `32.3 vs 32.5` collisions/update |
| mean minimum OBB clearance | `0.37275 -> 0.36661m` |
| collision episode触发 | `155/325 = 47.69%`，首次触发lead P50 `0.21s` |
| overtake episode触发 | `227/745 = 30.47%` |
| follow episode触发 | `11/373 = 2.95%` |
| 触发episode cap利用率 | mean `13.82%` / P90 `33.93%`；仅1 episode hit cap |
| 每trigger step平均penalty | `0.000349`，为step cap的`6.98%` |
| **actor 相对参数位移 / 前10个update的位移** | **`0.474`（47.4%）** |
| actor 绝对位移 / BC→U30 绝对位移 | `0.2545` |

**这两行必须与上面的 `0.4177%` 一起读，它们构成本条线最反直觉的事实**：reward 绝对量级
只有 collision 量级的 0.42%，但 actor 参数位移达到前 10 个 update 的 47.4%。当时只能事后
观察到这个不一致，无法事前预测。

机制是 `normalize_advantage=True`：advantage 按 minibatch 标准化，因此决定学习信号的不是
reward 的绝对量级，而是**相对 baseline advantage 标准差、且只作用在被触发 transition 上的
扰动**。一个足够稀疏的项即使总量微小，也能在其触发步上主导梯度。

因此**"剂量太小所以无效"这个推断是错的**——剂量在参数空间里并不小；Post-pass 的真正问题
是触发时机（rear clearance 是结果传感器而非起势传感器），不是幅度。任何"再加大权重"的
提案都应先用 `scripts/screen_reward_candidate.py` 的 `normalized_perturbation` 量化归一化
扰动，而不是比较绝对量级。

这里的no-Post-pass continuation只是自然参考，不是严格因果control：critic初始化、
LR/RNG路径和collision-cache actor provenance并不完全相同；不能仅凭这一行定案。
后续同面板deterministic eval才用于判断最终行为。

因此原`q²`剂量确实很小，且训练侧没有出现机制指标改善。首次trigger在collision
episode中位只领先终止约`0.21s`；个案sp21进一步显示，它比因果转向晚约`0.5s`，
在heading/slip已经发散且actor开始反向自救后才触发。rear clearance在甩尾中是
“结果传感器”而不是“起势传感器”，所以单纯加剂量不能修复信用时机。

同一新版Austin600、逐episode trace重算结果如下。`总碰撞`包含每臂固定出现的1次
opp-wall；`ego碰撞`只计ego-opp/ego-wall；甩尾定义为
`post_overtake_rear AND (slip>=8deg OR delta-heading>=20deg)`：

| 模型 | 总碰撞 | ego碰撞 | postOT | 甩尾 | rear-end | ego-wall | 超车 | 均速 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U30，无Post-pass | 14 | 13 | 7 | 4 | 4 | 2 | 366 | 5.391 |
| LR1 / `q²` / U10 | 20 | 19 | 12 | 8 | 2 | 5 | 360 | 5.398 |
| LR1 / `q` / U10 | 14 | 13 | 6 | 2 | 3 | 4 | 354 | 5.345 |
| LR3 / `q²` / U10 | 15 | 14 | 9 | 7 | 3 | 2 | 368 | 5.426 |

与U30逐scenario配对的ego collision、overtake和甩尾身份：

| treatment | ego collision消除/新造（p） | overtake丢失/新增（p） | fishtail消除/新造（p） |
|---|---:|---:|---:|
| LR1 / `q²` | `1/7`（0.0703） | `6/0`（0.0313） | `1/5`（0.2188），`4 -> 8` |
| LR1 / `q` | `5/5`（1.0） | `15/3`（0.00754） | `3/1`（0.625），`4 -> 2` |
| LR3 / `q²` | `2/3`（1.0） | `2/4`（0.6875） | `0/3`（0.25），`4 -> 7` |

9个“BC曾甩尾”的sentinel场景只适合检查继承失败，不适合做production判决：BC为9/9
甩尾，B/U30已降为1/9；LR1/q²的U1/U5/U10为2/1/1，LR1/q为1/2/2，LR3/q²为1/1/1。
它看不见PPO新造的甩尾，曾给出与完整Austin600相反的排序，因此不得再用这个9场景
panel单独接受或否决reward。

结论边界：

- `q²`应否决：两个LR臂都增加甩尾，LR1还把ego碰撞`13 -> 19`
  （碰撞消除1/新造7，配对`p=0.0703`）并增加wall collision。
- q-linear是唯一方向性改善甩尾的臂，但`4 -> 2`只有消除3/新造1，
  `p=0.625`，同时超车`366 -> 354`、均速下降、wall collision`2 -> 4`；
  不能视为通过。
- 旧的无clearance/time gate版本曾触发`897/1424 = 63.0%`成功超车，范围过宽；
  新gate解决了乱触发，却把触发推迟到甩尾后果阶段。两者之间不是继续扫weight即可
  解决的连续参数问题。
- Post-pass与risk在“车距危险”状态上有重叠，但语义不同：risk是一般、策略不变的
  credit redistribution；Post-pass是超车后直接改变偏好。现有结果不支持为这点额外
  增加production reward复杂度。

### 16.6 L12：要解决什么和怎样实现

L12不是第五项reward，也不改变`anisotropic_risk_potential`的组合方式、幂次、
最大值、gamma或terminal处理。唯一变量是：

```text
# 历史CLI，现已删除
--risk_longitudinal_clearance_m 1.2
# production/default: ppo/ppo_config.yaml = 0.6
```

现有vehicle risk距离：

```text
d_vehicle = hypot(longitudinal_clearance / L,
                  lateral_clearance / 0.2)
d = min(d_vehicle, wall_clearance / 0.2)
Phi = -0.05 * max(0, 1 - d)^2
r_risk = gamma * Phi(next) - Phi(previous)
```

L06的vehicle potential纵向支持边界是0.6m；是否实际非零还同时取决于0.2m横向
clearance。L12把纵向边界扩大到1.2m，希望把OL1同线跟车/追尾的风险信用提前到仍有
0.7-1.0s操控权的窗口。它理论上仍是potential-based shaping：真实terminal令
`Phi(next)=0`，timeout保留physical next potential，discounted telescope保持成立。
它改变信用分配，不应改变最优策略偏好。

CLI值通过`train_ppo.py -> vec_env -> environment -> PPOTransitionReward`传递，并写入
effective config；测试同时确认不会修改全局YAML默认值。

### 16.7 L12：离线筛选

在U30 Austin600固定轨迹上，L06/L12/L20和`min`/bounded-sum共6个变体进行了
3,600条variant-episode重放；原始600 episodes按结果分为clean overtake 366、
fishtail 4、OL1碰撞8（postOT 3、rear-end 4）、safe OL1 follow 116、
tight follow 76和ego-wall 2。全部变体满足`Phi <= 0`与配置上界，256组production
公式交叉校验最大误差`6.94e-18`，discounted telescope最大残差`3.99e-17`，
wall marginal retention全部通过。

| 变体 | OL1首次/连续提前量P50 | safe-follow触发episodes；加权active-state | clean-overtake触发episodes；加权active-state | fishtail首次/连续提前量P50 |
|---|---:|---:|---:|---:|
| B0：L06 + `min` | `0.485/0.485s` | `2/116；0.032%` | `116/366；1.601%` | `0.455/0.265s` |
| L12 + `min` | `0.790/0.715s` | `54/116；8.315%` | `177/366；3.057%` | `0.455/0.265s` |
| L20 + `min` | `0.880/0.880s` | `111/116；61.102%` | `228/366；5.731%` | `1.305/0.265s` |

L20虽把OL1连续提前量推到0.88s，却让safe-follow active-state达到61.1%，范围过宽。
`SUM06/SUM12/SUM20`的筛选指标分别与对应`min`变体一致；墙项本来就完整保留，因此
`min -> bounded sum`没有收益，未进入production。更重要的是：所有clearance变体的
fishtail连续提前量都固定在0.265s；只扩大clearance支持域无法提前捕捉yaw/slip起势，
所以L12从设计上就不是甩尾解法。该离线重放只验证公式、时机和选择性，不能证明
训练后策略会改善。

### 16.8 L12：受控训练、确定性迁移和分层结果

受控A/B从同一U30 actor、seed 42训练15 updates，Control保持L06，treatment只改L12。
两臂都使用U30重新分类的collision cache
`post-trained/collision-cache/ppo_privilege_gru_clip020_update30_austin_collision_pool_372`，
不允许actor mismatch，
Post-pass关闭；其余合同相同：`n_steps=6400`、batch`12800`、actor/critic
epochs`2/5`、GRU/head/critic LR`3e-6/3e-5/3e-4`、steering/speed std
`0.03/0.15`、gamma/GAE`0.999/0.995`、clip`0.20`、target-KL关闭。
两臂均比较各自U15 checkpoint；模型身份由run名、配置和checkpoint编号区分，不在
ANALYSIS复制分析产品摘要或哈希。

updates 11-15的role指标：

| role/指标 | Control L06 | L12 |
|---|---:|---:|
| collision-role episodes/collisions/rate | `455/259/56.92%` | `402/154/38.31%` |
| collision-role return | `-0.7880` | `-0.3586` |
| collision-role minimum gap | `0.1143m` | `0.1821m` |
| collision-role risk active | `14.07%` | `23.09%` |
| ordinary-role episodes/collisions/rate | `320/9/2.81%` | `321/6/1.87%` |
| ordinary-role return | `0.5157` | `0.5313` |
| ordinary-role minimum gap | `0.6675m` | `0.6794m` |
| ordinary-role risk active | `2.01%` | `6.30%` |

这说明L12在训练分布上的作用是真实的：更早的risk credit让actor留下更大余量并减少
collision-role早停，不是reward hacking。可是同一Austin600的checkpoint结果不稳定：

| checkpoint | L06 collision/OL1/fishtail/overtake | L12 collision/OL1/fishtail/overtake |
|---|---:|---:|
| U5 | 14 / 5 / 4 / 365 | 17 / 7 / 5 / 358 |
| U10 | 15 / 7 / 5 / 353 | 13 / 7 / 3 / 353 |
| U15 | 15 / 7 / 5 / 361 | 19 / 8 / 5 / 353 |

在三个相关checkpoint上汇总，碰撞`44 -> 49`，配对身份为消除18/新造23，
event-level McNemar `p=0.533`；按scenario聚类后11个方向改善、11个恶化。超车
`1079 -> 1064`，event-level丢失32/新增17、`p=0.044`，但scenario-clustered为
18恶化/9改善/573持平，sign test `p=0.122`；32个丢失超车中18个变为ego-opp
collision、14个变为follow。由于同一600场景跨checkpoint重复且存在多重比较，
不能把1800行当独立样本，也不能把`p=0.044`写成已确认损害；准确口径是
“没有稳定安全收益，且有不利超车迁移信号”。尤其不能用单独U10的`15 -> 13`
挑选L12。

更高功效的确定性配对给出真正的分布结构：

| 面板 | L06 -> L12碰撞；消除/新造（p） | L06 -> L12超车；丢失/新增（p） | fishtail消除/新造（p） | 结论 |
|---|---:|---:|---:|---|
| U30固定失败cache，372 | `217 -> 184；81/48`（0.00465） | `107 -> 121；45/59`（0.202） | `36/28`（0.382），`61 -> 53` | 对已筛困难工况真实有效 |
| held-out near，600 | `15 -> 19；6/10`（0.454） | `394 -> 391；9/6`（0.607） | `0/4`（0.125），`3 -> 7` | 没有可确认净迁移 |
| 未见起点hard failures，334 | `192 -> 165；68/41`（0.0124） | `101 -> 115；36/50`（0.161） | `34/23`（0.185），`60 -> 49` | 能跨起点迁移，不是逐位置记忆 |
| 其中interval-15，73 | `43 -> 43；10/10`（1.0） | `15 -> 15；9/9`（1.0） | `7/6`（1.0），`10 -> 9` | 目标regime净收益为0 |
| held-out near-miss，400 | `30 -> 40；21/31`（0.212） | `330 -> 304；44/18`（0.00130） | `13/13`（1.0），`16 -> 16` | KPI守门失败 |

372 cache的收益覆盖全部interval而非单一近距档：interval 8为`67 -> 51`
（消除28/新造12，`p=0.0166`），10为`53 -> 48`（20/15，`p=0.500`），
12为`47 -> 43`（15/11，`p=0.557`），15为`50 -> 42`（18/10，
`p=0.185`）；其中与eval完全相同几何的33场景为`20 -> 12`
（11/3，`p=0.0574`）。碰撞模式从
`postOT/rear/side/wall = 109/77/14/17`变为`85/66/19/14`。

held-out near600的碰撞模式则从`7/6/1/1`变为`13/4/1/1`，说明总数不显著之外仍有
postOT迁移风险。near-miss400的关键流向是`overtake -> ego-opp` 26、
`overtake -> follow` 18、`overtake -> overtake` 286；超车损失不是只由保守跟车
解释。

**334条未见起点hard failures按interval分层，收益随interval单调衰减到零**——这是
"有效但脱靶"最直接的证据，也是本条线最该记住的一张表：

| interval | 场景数 | L06 -> L12碰撞 | 消除/新造 | 配对p | 超车变化 |
|---:|---:|---:|---:|---:|---:|
| 8 | 99 | `58 -> 52` | — | 0.327 | `33 -> 34` |
| 10 | 89 | `47 -> 37` | — | 0.0987 | `33 -> 41` |
| 12 | 73 | `44 -> 33` | — | 0.0801 | `20 -> 25` |
| **15（目标regime）** | **73** | **`43 -> 43`** | **10/10** | **1.0** | **`15 -> 15`** |

即interval越近收益越大、越接近目标interval-15收益越小，到15恰好为零。同一分层里
raceline2（`42 -> 31`，p=0.0266）和speed 0.55（`20 -> 10`，p=0.0063）看似显著，
但这是26个分层上的多重比较，不足以支撑任何子regime结论，不要据此复活L12。

上述确定性transfer共覆盖1,944 actor-episodes；hard-instrument另覆盖1,468
actor-episodes。所有面板均满足finite、scenario payload逐臂相同、trace数量与
scenario数量一致、collision marker/trace contract一致。困难集按起点严格拆分，
train/held-out不相交，且筛选标签只用U30、没有查看L06/L12 treatment结果，避免
treatment leakage。

所以L12不是“公式无效”，而是“有效但脱靶”：

1. 它在interval 8/10/12等近距困难regime学到了可跨起点迁移的留余量能力；
2. 到全部为interval-15的目标geometry，消除与新造完全抵消；
3. 在本来贴近但能成功通过的near-miss上，扩大支持域会干扰成功轨迹，显著丢超车；
4. fishtail没有稳定收益：Austin只有U10短暂下降，U5/U15不保持；held-out hard
   `60 -> 49`也伴随34消除/23新造、`p=0.185`，near-miss为`16 -> 16`。

因此production继续使用`risk_longitudinal_clearance_m=0.6`。不要再扫L20、potential
composition或把L12与其他改动一起重训；若未来训练/eval任务分布发生实质变化，必须
重新建立同reward的匹配control，再把L12作为单变量重测，不能沿用当前训练侧收益。

### 16.9 Following-response required-deceleration：离线候选，未进入训练

它要解决的是OL1同线跟车中“ego已经在刹车，但净接近消除得太晚/太浅”的失败，不是
post-pass甩尾。全部计算来自保存轨迹且严格因果：

```text
front_gap = Frenet中心前后关系 - 两车OBB沿赛道切向的前后支撑
closing_speed = trailing 0.10s窗口内的unwrapped relative-progress变化率
closing_time = front_gap / max(closing_speed, eps)
required_relative_decel = closing_speed² / (2 * (front_gap - safe_gap))
escalation = required_relative_decel在过去0.20s内的增长率
```

`closing_time`只是当前局部线性模型下的前向OBB gap耗尽时间，不是完整物理TTC。候选
还使用带hysteresis的同走廊门控（进入/退出横向偏移`0.20/0.25m`）、gap
`2.0/2.2m`、closing-time`1.5/2.0s`、safe gap`0.5m`、step/episode caps
`0.005/0.05`。完整离线固定参数是：

```text
closing-rate window                 0.10s
gap entry / exit                    2.0 / 2.2m
corridor entry / exit               0.20 / 0.25m
lateral-overlap entry               0.02m
phase grace / recovery              0.20 / 0.10s
safe gap / response horizon         0.50m / 1.0s
linear persistence                  0.10s
required-decel threshold            1.50m/s²
escalating required-decel/persistence 1.25m/s² / 0.20s
escalation rate window              0.20s
minimum escalation growth           2.0m/s³
lateral escape horizon/window       0.40 / 0.10s
smooth-L1 delta                     0.50
weight / step cap / episode cap     0.10 / 0.005 / 0.05
```

该脚本复用了`EscalatingRequiredDecelerationGate`的因果几何，但没有修改
`ppo/reward.py`，也没有产生PPO训练run。

固定B/U30 Austin600完整重放为600 episodes/traces、0 errors。目标proxy是7个
`OL1 + ego-opp collision`，安全对照是192个clean OL1 follow；它们是scenario/outcome
proxy，不是已证明同一物理机制的ground truth。600场景按opponent line恰为
`200/200/200`，有13次ego碰撞和1次opp-wall；result/trace key-set一致。赛道轴/Frenet
方向、closing sign、无未来信息、grace/persistence因果性等内部自检全部通过。

| 离线候选 | 目标覆盖 | clean OL1 follow误触 | clean overtake误触 | 首次penalty lead |
|---|---:|---:|---:|---:|
| linear closing excess | `6/7` | `100/192 = 52.08%` | `0/366` | P50 `1.215s` |
| required deceleration | `6/7` | `27/192 = 14.06%` | `0/366` | min/P50 `0.20/0.89s` |
| escalating required deceleration | `6/7` | `21/192 = 10.94%` | `0/366` | min/P50 `0.75/0.945s` |

预注册准入要求clean OL1 follow positive rate `<10%`，escalating候选没有通过
（10.94%）。更重要的是该面板中same-corridor成功超车为`0`：`0/366` clean overtake
误触全部来自cross-line超车，不能验证该门控会不会压掉真正的同走廊成功行为。因此
`ready_for_training_ab=false`。

完整准入矩阵为6项：

| 准入检查 | 实测 | 判定 |
|---|---:|---:|
| 目标OL1 collision覆盖至少6/7 | `6/7` | PASS |
| 目标首次penalty最小lead至少0.40s | `0.75s`（escalating） | PASS |
| clean OL1 follow误触严格小于10% | `21/192 = 10.94%` | **FAIL** |
| observed cross-line clean overtake误触低于5% | `0/366` | PASS |
| same-corridor successful-overtake有非零支持 | `0` | **FAIL** |
| 目标required-decel P10高于clean-follow P90 | 满足 | PASS |

escalating候选在目标组累计penalty`0.12135`、每episode P50/P90
`0.01144/0.03466`，累计激活`1.42s`、持续时间P50`0.16s`；clean-follow组累计
penalty`0.20653`、每episode P90`0.001087`、累计激活`2.66s`。目标episode单次剂量
更集中，但安全组总样本多，因此“总penalty更大”不能直接写成目标/非目标倒置。

准确结论是“候选具有目标覆盖和更早时机，但当前选择性/正对照不足，未获训练准入”，
不是“required-deceleration reward已经训练失败”。停止规则：

1. 不把它写进production、不创建训练臂、不与exploration gate的结果混为一谈；
2. 不把closing time称为完整TTC；
3. 只有在新固定面板补足same-corridor成功正对照、clean-follow误触通过预注册阈值后，
   才能重新评估是否做单变量A/B；此前不要继续扫weight/cap。

**与§18 exploration gate的关系（共享实现语义，两种用途，不要混淆结论）：**
本候选离线验证时用当时`ppo/exploration.py`中的`EscalatingRequiredDecelerationGate`对
escalating replay逐步交叉校验；与该candidate匹配的关键值是corridor entry/exit
`0.20/0.25m`、safe gap`0.50m`、required relative deceleration`1.25m/s²`、
persistence`0.20s`、front gap`2.0m`、closing time`1.5s`。§18 的C/CT复用该门作为
**exploration触发器**而不是reward触发器；CT-v2后来改用更宽的front-corridor gate。
**该门已随C臂于2026-07-30从代码移除**（仅剩CT-v2的front-corridor门）；上列参数是其
完整重建合同，`EXPERIMENTS.md`同步保留。
注意离线脚本自身还单独报告`required_decel=1.50m/s²`候选，不能与escalating候选混写。

两条线对这个门的结论并不矛盾，而是互补：

- 作为 **reward** 门：`21/192 = 10.94%` 的 clean-follow 误触**太高**（预注册要求 <10%），
  因为误罚正常跟车会直接改变策略偏好；
- 作为 **exploration** 门：同样的稀疏性反而**太低**——§18 实测 C 的后期门曝光仅约
  `0.15%`、CT 的时序激活曝光约 `0.51%`，导致干预量不足以产生学习信号。

所以“门太宽”和“门太窄”针对的是两个不同的判据。两边配置并非运行时共享的同一实例；
修改一边不会自动修改另一边。若未来有意保持离线candidate与online gate等价，必须
显式同步参数并让逐步交叉校验继续为零误差；无论如何，两条线的历史数字不能跨用。

### 16.10 三条实验线的最终边界

三项都针对安全信用，但证据边界和失败原因不同：

- Post-pass：新增的相位特定直接penalty；主要问题是rear-clearance gate感知后果太晚，
  `q²`又过弱，放大到`q`后只得到不显著的甩尾改善并损害超车。
- L12：原risk potential的更早、更宽支持域；训练机制有效，但收益集中在错误regime，
  目标interval-15没有净收益，near-miss KPI显著变差。
- Following response：required-deceleration/escalation只通过离线saved-trace筛选；
  时机和目标覆盖有希望，但clean-follow误触略超阈值且缺same-corridor成功正对照，
  所以没有进入训练，不能继承Post-pass/L12的“训练后否决”结论。

本节正文已经保留可支持决策的配置、样本量、配对增删、p值、机制指标和验证边界。
不再维护JSON/CSV路径、复算脚本清单或文件摘要；这些文件后续可能被清理，接手者应直接
以本节的核心记录理解为什么三条reward方向被保留、否决或冻结。

清理 artifacts 后会永久失去三项能力，需要时只能重新评估、不能翻旧文件：按其他口径
重算、查具体"消除/新造"的场景身份（本节只保留计数与p值）、做本节未列出的分层。
当前四项reward实现仍在`ppo/reward.py`；被删除的Post-pass/L12生产接口只能根据本节与
`EXPERIMENTS.md`中的合同重建。共享门控`ppo/exploration.py`属于探索实验，不是reward；其中escalating门已随C臂移除，重建合同见`EXPERIMENTS.md`。

当前明确停止规则：不要恢复Post-pass或L12接口，不要继续扫其权重/阈值，不要把
训练池上的下降写成production改善；following-response当前只允许保留为离线候选，
不得写成已实现或已训练。Production保持固定四项和risk L06。

---

## 17. Hard-neighbor / Boundary-aware collision pool（2026-07-29收口）

### 17.1 当前production决策

| 方案 | 当前状态 | 已有证据 | production决策 |
|---|---|---|---|
| Base collision pool | 479条冻结BC确认碰撞 | 是完整805池A/B的control | 保持默认 |
| Boundary-aware完整805池 | 479 base + 326 boundary collision；45 updates和Austin600均完成 | U45碰撞`12 -> 17`，U25+均值`13.0 -> 21.4` | 否决，接口退役 |
| 20% boundary分层采样 | 45 formal updates完成 | 同面板U35/U40/U45碰撞`27/20/19`，base为`14/11/12`；同时超车更少 | 否决，接口退役 |
| 10% boundary分层采样 | 45 formal updates完成 | U1/U5/U10/U15/U20各有完整600包，但缺U25--U45晚期评测 | 未定案但主动终止，接口退役 |
| Outcome-aware后续接口 | 实现存在、没有独立A/B | 不属于原始805池A/B，不能继承其正负结论 | 独立实验接口，不是production winner |

因此准确结论不是“hard-neighbor没有测试”，也不是“所有困难采样均无效”，而是：

> 原始boundary-aware cache构建正确、能稳定改变训练分布；但把326条boundary collision
> 全量合入479池后，策略在Austin600上更快、超车更多，同时总碰撞明显增加。完整805池
> 已被否决；20%分层臂也在可配对的晚期checkpoint持续变差并被否决。10%臂没有完成
> 晚期判决，用户选择停止而不是继续投入；源码退役不改变它的未知证据状态。

### 17.2 三套场景和cache构建合同

必须区分：

| 集合 | 构成 | 用途 |
|---|---|---|
| Austin600 | 50 eval起点 × 3 opponent racelines × 4 speeds，interval 15 | 固定快速eval；不参与cache构建 |
| Ordinary600 | 50 training起点 × 3 racelines × 4 speeds，interval 15 | ordinary role训练 |
| Collision candidates 10,800 | 100 training起点 × 3 racelines × 4 intervals × 9 speeds | 冻结BC分类与cache构建 |

Austin600与训练pool的scenario tuple不重合，但仍是同一Austin赛道上的近分布面板；它能
做受控同面板比较，不能单独证明跨地图泛化。

Base cache分类结果：

```text
10,800 candidates
  479 ego_collision
  10,285 other
  36 invalid
```

Boundary-aware builder只检查离散网格中一轴相邻的
`ego_collision <-> other`翻转边，排除invalid端点，在边界内部生成精确speed midpoint
或内部整数interval，再由同一冻结BC完整重放。当前schema-2 cache结果：

```text
1,042 boundary pairs
1,183 unique generated candidates
  326 confirmed ego_collision
  857 other
  0 invalid

final collision pool = 479 base + 326 boundary = 805
```

只有326条确认碰撞进入最终pool；857条other只作为构建证据，不进入原始hard-neighbor
训练。Cache位于：

```text
post-trained/collision-cache/pretrained_end2race_austin_collision_pool_479/
post-trained/collision-cache/pretrained_end2race_austin_boundary_aware_collision_pool_805/
```

Schema-2身份绑定冻结actor SHA、base-cache文件、map/raceline、opponent planner和
generator metadata；manifest验证失败时fail closed，构建使用临时目录并原子发布，不
覆盖已有正式目录。

2026-07-30 cache目录完成规范迁移。历史run内的`run_config.json`和
`collision_cache_info.json`仍保留当时真实使用的旧路径
（`default`、`boundary-aware-v1`、`u30-risk-l12-seed42`）；它们是历史参数记录，
不得为了路径整洁而改写。当前代码和新实验只使用上述规范目录。

### 17.3 历史训练采样语义（源码已退役）

Hard-neighbor是pool/sampling改动，不是reward项。Even logical ranks仍使用collision
role，odd ranks仍使用ordinary role；env-major recurrent minibatch继续保持两种role
的transition平衡。

历史三种调用语义：

```text
# production/default
不传 --hard_neighbors
-> 单一479条base collision queue

# 原始完整805池实验
--hard_neighbors
-> 805条merged collision queue
-> 一个完整queue cycle中boundary占326/805 = 40.5%

# 后续分层比例实验
--hard_neighbors --hard_neighbor_fraction 0.10  # 或0.20
-> base479与boundary326使用独立无放回shuffle queues
-> collision episode resets按确定性周期精确选择10%或20% boundary
```

`hard_neighbor_fraction`是collision episode reset的来源比例，不是全部episode比例，也
不能因episode长度不同直接解释成相同的transition占比。省略fraction时保持旧的805池
uniform-merged语义。

### 17.4 Probe、实现验证和证据边界

历史standalone probe围绕3个刻意选择的高支持种子生成18个neighbors，其中11个碰撞：

```text
11 / 18 = 61.1%
base valid candidate collision rate = 4.45%
descriptive enrichment = 13.73x
```

这只能证明局部邻域存在富集，不能证明生产pool的无偏yield、PPO收益或泛化。Probe从来
不是production input，也不应作为接手入口。

2026-07-22实现阶段曾通过5/5 focused hard-neighbor tests、manifest复核、361D finite
VecEnv smoke、compileall和diff检查；fraction模式后来也增加了CLI、确定性周期、queue
state等测试。它们证明实现合同，不证明行为收益；接手时仍应在当前正式环境重新运行
相关测试。

### 17.5 完整805池A/B结果和失败原因

两臂都从canonical BC开始，记录的PPO配置一致：`privilege_gru`、45 updates、
actor/critic epochs `2/5`、batch 12,800、clip 0.20、LR
`3e-6/3e-5/3e-4`、std `0.03/0.15`；唯一实验轴是479池与完整805池。

| 指标 | Base479 | Boundary-aware805 |
|---|---:|---:|
| Austin碰撞 U1/5/10/15/20/25/30/35/40/45 | 21/18/16/13/13/17/11/14/11/12 | 19/18/19/15/14/22/26/20/22/17 |
| 全checkpoint平均碰撞 | 14.6 | 19.2 |
| U25+平均碰撞 | 13.0 | 21.4 |
| U45碰撞 | 12 | 17 |
| U45超车 | 357 | 365 |
| U45平均速度 | 5.337m/s | 5.428m/s |
| U45 `merge_tail` | 7 | 4 |

`merge_tail`是group summary的特定尾部接触口径；另一份诊断artifact的
`structural_tail`把hard U45计为5。两者不能混成一个未命名的“甩尾数”。确定无歧义的
主结论来自总ego collision：完整805池在后半程持续差于base，并在U45多5次碰撞。

训练诊断不支持“critic学不动”：

| U25-U45 post-update EV | Base479 | Boundary-aware805 |
|---|---:|---:|
| overall | 0.930 | 0.921 |
| collision role | 0.920 | 0.909 |

差异不足以解释行为恶化。与更高超车和均速一起看，较合理的机制解释是：boundary cache
只告诉PPO“哪里多采样”，没有提供“此处应跟车还是超车”的安全动作标签；在现有reward
下，actor在训练分布中学到更积极的超车策略，部分训练收益没有迁移成eval安全收益。
这是证据支持的诊断，不是单一因果证明。

### 17.6 低比例臂和停止规则

完整805臂、10%臂和20%臂均完成1条warmup和45条formal metrics，`max_update=45`。
完整805臂和20%臂已经否决且原checkpoint已清理；仍保留的未定案臂只有：

```text
post-trained/ppo_hard_neighbor_boundary_aware_fraction0p10/
```

10%臂的`collision_cache_info.json`记录：

```text
hard_neighbor_sampling_mode = stratified_collision_episode_reset
hard_neighbor_sampling_fraction = 0.10
base_collision_count = 479
boundary_collision_count = 326
```

20%臂与base在U35/U40/U45的episode key完全一致，配对结果为：

| update | collision base -> 20% | removed / created（p） | overtake base -> 20% | lost / gained（p） |
|---:|---:|---:|---:|---:|
| 35 | `14 -> 27` | `7 / 20`（0.0192） | `349 -> 334` | `24 / 9`（0.0135） |
| 40 | `11 -> 20` | `7 / 16`（0.0931） | `355 -> 340` | `22 / 7`（0.00813） |
| 45 | `12 -> 19` | `8 / 15`（0.210） | `357 -> 346` | `22 / 11`（0.0801） |

三个晚期checkpoint在安全和超车两个轴上方向一致变差，且U35碰撞、U35/U40超车达到
配对显著，因此20%臂否决。10%臂只保留U1/U5/U10/U15/U20五个完整600包，不能评价
晚期U25--U45；不得从training return、早期eval或目录存在推断它有效。若未来补评，必须：

1. 固定同一评估面板和collision定义；
2. 同时报总碰撞、超车、速度及逐episode消除/新造身份；
3. 明确tail/fishtail使用哪个定义；
4. 不把结果与reward、LR、clip或其他pool变更混合；
5. 不向现有run目录继续写入。

最终停止规则：production保持479池；hard-neighbor CLI、cache builder接入和fraction
scheduler已经退役。不要继续把完整805池或20%分层臂解释为候选winner，也不要仅凭cache
enrichment恢复它。10%只能记作“未完成判决后主动停止”；若任务分布实质变化后重开，
必须按`EXPERIMENTS.md`重建接口并重新建立control。Outcome-aware等后续方案必须建立
自己的control，不能继承原始805池结论。

### 17.7 已合并的历史设计边界

本节保留原hard-neighbor专题中仍会影响未来决策、但没有必要继续维护第二份HANDOFF的信息。
它们是设计边界或历史分析，不应冒充新的A/B结论。

**Base479本身已经包含局部密集失败和广泛outcome boundary。** 按“相同`ego_idx`和
opponent raceline、interval差不超过2、speed差不超过0.05、排除自身”的历史支持定义，
479条中有298条至少邻接1条collision，165条至少邻接2条（覆盖34个ego起点，raceline0/1/2
分别50/62/53）。从10,800条base outcomes确定性构造一轴相邻关系后，有1,042对
`ego_collision <-> other`翻转边，涉及446/479条base collision和81个ego起点。
这些数字可由上述`collision_scenarios.json`和`boundary_pairs.jsonl`重算。它们证明边界来源
足够广，也说明简单重采样已有hard子集不会增加场景覆盖；但不证明具体场景物理可解或PPO可学。

**当时的四种方案取舍：**

1. 对已有高支持collision重加权最便宜，但只改变权重、可能增加位置记忆，没有增加参数覆盖；
2. 固定boundary-aware cache增加唯一场景、保持训练分布静态且可复现，因此成为第一版；
   其完整805池后来已由§17.5的A/B否决；
3. current-policy动态mining理论上能追踪新弱点，但会同时改变actor和训练分布，引入标签波动、
   chasing failure、额外仿真成本和归因困难；它从未完成受控A/B，不能继承805池结论；
4. “当前actor失败、safe oracle成功”的对抗课程更接近困难但可解场景，但需要经过验证且可部署的
   teacher。§20的oracle reachability使用未来碰撞时刻和场景特定搜索，只是可达性见证，不能
   直接充当该teacher。

**采样与cache身份合同：** even logical ranks固定为collision role，odd ranks固定为ordinary
role，所以rollout transition是50/50而episode数不必是50/50；两个`RoleScenarioQueue`
各自维护独立无放回permutation，跨update连续推进。Schema-1 default cache只记录actor路径，
不能识别同路径内容被覆盖，这是已知身份缺口；schema-2则绑定actor、base cache、map/raceline、
scenario generator、horizon/timestep/integrator和refinement规则，任何语义身份变化都应
fail closed。`env_workers`只影响构建速度，不进入语义身份。构建先写同父目录临时目录，
通过manifest后原子发布，不覆盖已有正式目录。

**为什么probe的11条不直接并入479池：** 它们来自3个刻意选择的高支持seed，`11/18=61.1%`
只代表局部富集；直接去重合并后只占collision pool的2.24%，折算全部transitions约1.12%，
而复制加权不会增加3个source families之外的多样性。正式builder因此从1,042条自动边界生成
1,183条候选，用同一冻结BC完整重放后只合并326条确认`ego_collision`；857条`other`只作为
构建证据，第一版没有同时改ordinary pool，以保持单变量归因。

原独立专题文件当时位于被忽略的`.agents/`目录、从未受Git跟踪；完成本节和§17.1-17.6的
核心结论合并后已主动删除。后续不要恢复第二份hard-neighbor handoff，更新统一写入本文。

---

## 18. 速度探索专项：条件噪声、时间相关噪声、走廊门控与异线高速重加权（2026-07-28 至 07-30）

### 18.1 当前生产决策

| 实验 | mode | 本质 | Austin600 | near400 | hard73 | 跨地图1800 | 决策 |
|---|---|---|---:|---:|---:|---:|---|
| **Production逐步独立速度白噪声** | `baseline` | 每步独立白噪声，标准差0.15 | **14 / 366** | **28 / 325** | 54 / 12 | **80 / 1142** | **production** |
| 条件白噪声放大 | `conditional_white` | 旧required-deceleration门内把独立白噪声标准差从0.15改为0.50；无时间保持 | 15 / 355 | 41 / 299 | 49 / 25 | — | 否决：门太稀疏 |
| 全局时间相关速度噪声 | `temporal_global` | 无门控；全状态标准差0.15的同一速度残差保持50步 | 16 / 359 | 53 / 286 | **19 / 31** | 64 / 1144 | 否决：near显著恶化 |
| 条件时间相关速度噪声（历史入口已删除） | `conditional_temporal` | 旧required-deceleration门触发标准差0.25、50步速度残差块 | 15 / 342 | 40 / 295 | 46 / 17 | — | 否决：门太稀疏且未通过综合验收 |
| 前向走廊门控时间相关速度噪声、2米门宽 | `corridor_temporal` | 前向走廊门、标准差0.15、保持50步 | 20 / 338 | 35 / 286 | — | **46 / 1122** | 部分：跨地图显著好，Austin/near变差 |
| 前向走廊门控时间相关速度噪声、1米门宽 | `corridor_temporal` | 同上，门收窄到1.0米 | 21 / 350 | — | — | 76 / 1127 | 否决：收窄即失去收益 |
| 前向走廊门控时间相关速度噪声、45 updates | `corridor_temporal` | 仅把训练长度改为45 updates | 17–20 / 332–345 | 33–37 / 271–288 | — | 43–46 / 1117–1134 | 否决：与30 updates无差别，方向关闭（18.6） |
| ordinary异线高速重加权、比例0.6 | `corridor_temporal` + YAML `ordinary_offline_fast_fraction: 0.6` | 保持走廊门控时间相关探索，ordinary角色内把异线高速权重33.3%→60%，同线份额锁死 | 16–21 / 363–370 | **63–77** / 291–306 | — | 54–57 / **1146–1155** | 否决：near400四点全显著恶化（18.7.3） |

单元格为 `ego collision / overtake`；给出区间的行是 u27–u30（45u 行是 u42–u45）四个
checkpoint 的极差，单值行是单个 checkpoint。**production 保持
`--speed_exploration_mode baseline`**，即不传该 flag。所有臂都是 seed 42、从 canonical
BC fresh-start、reward/pool/LR/clip/actor 完全固定，唯一变量是 speed exploration
（45 updates行额外只改`--num_updates`，ordinary异线高速重加权行额外只改ordinary采样权重）。

1米门宽 U30 的 Austin 旧摘要曾写成 `21 / 348`。清理审计时用存活的600条完整数值trace，
按当前统一口径（仅ego碰撞；非碰撞episode按terminal wrapped relative progress分类）
重算为 **`21 / 350`**；600条trace均有限、terminal marker完整。三张跨地图合计仍为
`76 / 1127`。旧直接聚合JSON未保留，因此后续统一使用trace可复核的`21 / 350`，不再混用
旧的348。

两条最终机制实验的权重和评测已按完整名称规范保留：

```text
post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/
eval_results/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/

post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps_ordinary_offline_fast_reweight_0p60/
eval_results/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps_ordinary_offline_fast_reweight_0p60/
```

走廊实验保留U1--U45全部actor/critic；U1--U30已验证与原30-update run逐字节相同。
重加权实验保留U1--U30全部actor/critic。评测目录以`update<N>/<MAP_NAME>/multiagents`
保存标准600场景，并以`update<N>/Austin/near400`单独保存固定near400，避免旧目录中
600+400条trace混放。走廊实验现存66个完整面板中的46个属于该实验：Austin主面板
U10/U15/U20/U25/U26--U30/U42--U45，near400的U10/U15/U20/U25/U30/U42--U45，
以及三张跨地图U27--U30/U42--U45；重加权实验保留U27--U30的Austin600、near400和
三张跨地图，共20个面板。

清理前这些目录只剩数值trace、没有原始`results_multi.json`。现已从每条trace按ego
collision scope、终态相对progress和终端行合同重建逐episode结果，并逐面板核对台账。
每个目录的`eval_manifest.json`标记`complete_trace_reconstruction`：
trace和result key集合相等、数值有限、collision subtype互斥、terminal/action合同通过，
同时明确`direct_evaluator_aggregate_retained=false`。因此它们可继续做配对比较，但若要求
“原评估器直接聚合JSON”的发布级原件，仍需重跑而不能把重建结果冒充原件。

上表中的条件白噪声、全局时间相关和条件时间相关数字采用清理前完成去重和口径校正后的
最终实验台账。早期分析表曾混入不同checkpoint或诊断子集，不能继续引用。三组Austin
评估现在分别整理到对应完整实验名下的`update30/Austin/diagnostic_common228`，每组只有
228条共同子集trace。全局时间相关速度噪声的三张跨地图目录各有600条trace，但都没有
聚合结果JSON；因此它们是诊断或不完整评估遗留物，不是按当前eval规范保存的正式结果包。
后续若重跑完整面板，应以新生成的`results_multi.json`和逐episode结果更新本表。

**区间而非单点是必需的**：前向走廊门控时间相关速度噪声的Austin600碰撞在u26–u30为
`36,26,22,26,20`
（极差 16），跨地图在 u27–u30 为 `42,49,51,46`（极差 9，σ 3.9），而行为算子在同一区间内
是平的。Production的Austin600 u24–u30为`18,16,14,14,13,13,14`（极差5）。因此
**任何corridor实验的单checkpoint比较都不可判读**，新实验必须报区间。

前向走廊门控时间相关速度噪声（2米门宽）的同场景配对结果：跨地图碰撞`80 -> 46`
（removed 57 / created 23，p=0.000183）、超车 `1142 -> 1122`
（lost 49 / gained 29，p=0.0308）；Austin碰撞 `14 -> 20`
（removed 11 / created 17，p=0.345）、超车 `366 -> 338`
（lost 32 / gained 4，p=1.94e-6）；near400碰撞 `28 -> 35`
（removed 23 / created 30，p=0.410）、超车 `325 -> 286`
（lost 54 / gained 15，p=2.61e-6）。

全局时间相关速度噪声的跨地图单点也要保留配对口径：相对production，碰撞`80 -> 64`
（removed 51 / created 35，p=0.105），超车`1142 -> 1144`
（lost 36 / gained 38，p=0.908），所以“净少16次”本身未达到配对显著；相对BC则为
碰撞`96 -> 64`（60/28，p=0.000846）、超车`1106 -> 1144`
（23/61，p=4.08e-5）。这说明全局时间相关速度噪声确实优于BC，但在已很强的production
模型上证据是方向性而非显著胜出；前向走廊门控时间相关速度噪声才把同线收益放大到
对production显著。

### 18.2 要解决什么

Production的失败模式里有一类是**同线跟车追尾（same-line / OL1 rear-end）**：ego跟在同一条线上的
opponent 后面，需要持续减速却做不到。诊断假说是 PPO 的探索噪声**每步独立重采样**，使连续
多步同向速度扰动远少于时间相关探索；这会降低策略采到并强化“持续减速约0.5秒”动作序列的
概率。它是时间相关探索的设计依据，不是“白噪声绝对不可能采到该序列”或已证明的唯一根因。

三种原始臂的完整定义如下：

- **条件白噪声放大**：门外与production完全相同，每步独立采样速度噪声、物理标准差为
  `0.15 m/s`；旧required-deceleration门为真时，仅把当步独立噪声标准差改为`0.50 m/s`。
  它不缓存噪声，不产生时间相关块。
- **全局时间相关速度噪声**：不使用任何危险门；每个并行episode采样一个标准速度残差，
  用`0.15 m/s`缩放后连续复用50个仿真步，即`0.50s`，然后重采样。episode reset会立即
  清掉剩余块并重新开始。它只改自相关，不改每一步的边际标准差。
- **条件时间相关速度噪声**：门外仍按production逐步独立采样，标准差`0.15 m/s`；旧门
  从假变真且当前没有活动块时，采样一个按标准差`0.25 m/s`缩放的速度残差并固定50步。块一旦开始，
  即使门随后关闭也会完整执行；episode reset清空。它同时改变门内幅度和时间相关性，
  因而不是前两种机制的纯单变量复制。

条件白噪声放大与条件时间相关速度噪声共用的旧门是有迟滞和状态记忆的在线因果判据：
`closing_window=0.10s`、entry/exit gap=`2.0/2.2m`、
entry/exit closing-time=`1.5/2.0s`、走廊entry/exit横向偏移=`0.20/0.25m`，
再要求relative required deceleration达到`1.25m/s²`并在`0.20s`窗口内持续/升级。
它还包含`0.20s`warning grace、`0.10s`recovery hold、`0.50m`safe gap和横向逃逸检查。
全部输入来自当前及历史观测，不使用未来碰撞时刻、oracle动作或特权触发器。后期实测门
曝光分别仅约`0.15%`和`0.51%`，这是两组干预不足的重要原因。

干预的共同目的，是提高策略实际采到连续减速动作序列的概率。全局时间相关速度噪声检验
纯时间相关性，条件白噪声放大检验稀疏危险状态内的纯幅度放大，条件时间相关速度噪声检验
旧门内的幅度与时间相关组合；后续前向走廊门控时间相关速度噪声才把门换成更宽的
same-corridor几何门。四者都只影响训练期随机采样，确定性eval和部署时actor
的输入、权重结构与输出路径不变。

规范化后的模型与训练记录入口：

```text
post-trained/ppo_conditional_white_speed_noise_0p50/update1...update30/{actor.pth,critic.pt}
post-trained/ppo_global_temporal_speed_noise_0p15_hold50steps/update1...update30/{actor.pth,critic.pt}
post-trained/ppo_conditional_temporal_speed_noise_0p25_hold50steps/update1...update30/{actor.pth,critic.pt}
```

每个实验根目录同时保留`run_config.json`、`trajectory_manifest.json`、`metrics.jsonl`、
`episodes.jsonl`和场景池记录。旧实验目录与这些文件通过hardlink共享实际权重数据，因此
规范化没有复制大文件；规范路径本身不依赖旧文件名。

调用链：

```text
train_ppo.py
  --speed_exploration_mode {baseline,temporal_global,corridor_temporal}
  # 历史上还有 conditional_white / conditional_temporal，均已删除
ppo/ppo_config.yaml
  front_corridor_gate_maximum_gap_m: 2.0
-> CentralScheduleSubprocVecEnv
-> End2RaceGymnasiumEnv
-> ppo/policy.py中的时间相关速度噪声状态
-> ppo/env.py中的前向走廊门
```

每个 transition 的结构化 exploration state 会存入 buffer，用于 PPO log-probability 重建；
已验证 collection-equivalent pre-update ratio error 在所有报告窗口里**精确为 0**。

### 18.3 机制确实成立（这是本线最重要的正面结论）

不要把这条线读成"时序探索没用"。它的**机制被直接证实了**：

Production有8个已知的OL1碰撞目标。把每个production碰撞前1.5秒窗口对齐后：

| 实验 | 8个目标上的结果 | 指令速度低于对手的目标数 | 最小指令速度 − 对手速度中位数 |
|---|---|---:|---:|
| Production逐步独立速度白噪声 | 7 ego-opp, 1 ego-wall | 0/8 | +0.733 m/s |
| 条件白噪声放大 | 3 follow, 2 overtake, 2 ego-opp, 1 ego-wall | 3/8 | +0.388 m/s |
| **全局时间相关速度噪声** | **7 follow, 1 overtake；8个全部化解** | **4/8** | **−0.013 m/s** |
| 条件时间相关速度噪声 | 4 follow, 1 overtake, 2 ego-opp, 1 ego-wall | 1/8 | +0.303 m/s |

即：**0.5秒的相关块足以让actor学会缺失的持续减速**，全局时间相关速度噪声把8个已知
目标全部救回，
且指令速度真的降到对手速度以下。这不是训练指标假象。

训练侧（updates 26-30）也一致：

| 实验 | 训练碰撞率 | 超车 | 平均最小 OBB 余量 |
|---|---:|---:|---:|
| Production逐步独立速度白噪声 | 24.4% | 347 | 0.367 m |
| 条件白噪声放大 | 26.6% | 327 | 0.386 m |
| 全局时间相关速度噪声 | **16.9%** | **360** | 0.428 m |
| 条件时间相关速度噪声 | 17.0% | 339 | **0.461 m** |

条件门的实际曝光远低于名义标准差所暗示的：条件白噪声放大的后期门曝光约**0.15%**，
条件时间相关速度噪声的时序激活曝光约**0.51%**，而全局时间相关速度噪声是100%。
所以条件白噪声放大基本等于没干预。

### 18.4 为什么全部没通过验收：失败迁移，不是没学会

前向走廊门控时间相关速度噪声的配对trace诊断（2,800对匹配场景）给出了最清楚的机制，
**这是本线的核心结论**：

| 面板 | same-line 碰撞 | off-line 碰撞 | 合计 |
|---|---|---|---|
| 跨地图 1800 | `66 -> 22` | `14 -> 24` | `80 -> 46`（好） |
| Austin600 | `8 -> 5` | `6 -> 15` | `14 -> 20`（差） |
| near400 | `4 -> 0` | `24 -> 35` | `28 -> 35`（差） |

**符号在所有面板上都一样：same-line 碰撞下降，off-line 碰撞上升。** 净结果好不好，
只取决于该面板里 same-line 占比多少。跨地图面板 same-line 占 66/80，所以净收益大；
Austin600 只占 8/14，所以被 off-line 的增加淹没。

这个分解在全局时间相关速度噪声上已经出现，并非走廊门控版本独有。跨地图相对production：

| opponent line | production collision | 全局时间相关速度噪声 collision | 变化 |
|---|---:|---:|---:|
| raceline1（同线） | 66 | **33** | `-33` |
| raceline0（异线） | 8 | 13 | `+5` |
| raceline2（异线） | 6 | 18 | `+12` |

联合raceline×speed后，同线四个速度档全部改善；异线伤害的15/17集中在
`off-line × speed 0.8`（`11 -> 26`），其余7个异线单元合计只`3 -> 5`。
因此不能用边际speed表写成“0.8下同线也恶化”；它是一个典型composition/Simpson陷阱。
全局时间相关速度噪声的可迁移能力是同线持续减速，副作用集中在异线高速绕行。

超车损失同样是结构化的：跨地图净损 20 次中 19 次是 off-line、15 次发生在 speed scale 0.8；
Austin 净损 28 次中 off-line 净损 29；near400 净损 39 次中 36 次 off-line、32 次集中在
speed 0.8/0.85。**损失主要集中在高速、绕线超车，而不是同线跟车。**

机制边界：配对结果直接证明了“same-line下降、off-line上升”，但没有唯一识别出off-line
恶化的内部因果。一个与数据一致的解释是：时序相关探索教会了actor持续减速，而参数更新带来的
行为变化没有被限制在门内，随后在需要果断绕行的off-line高速场合表现为犹豫和新碰撞。它应写成
当前最佳解释，不应写成已证明的唯一原因。CT 的Austin结果也呈现相同方向：ego-opp碰撞
`12 -> 9`，但ego-wall `2 -> 6`、超车 `366 -> 342`。

Gate宽度对照不支持“把2.0m简单收窄到1.0m”作为解法：

| gate front gap | 固定训练池离线门曝光 | 跨地图碰撞 | Austin600 碰撞 |
|---:|---:|---:|---:|
| 2.0 m | 17.53% | 47（U27-U30 均值） | 23.5 |
| 1.0 m | 4.43% | 73.25 | 22.75 |

收窄门把曝光降到 4.43% 后，跨地图收益直接消失（回到接近 B 的 80），Austin 却几乎没改善。
预注册结论是 `narrowing lost the same-line benefit -> revert to 2.0 m`。离线 gate 选择性
扫描也显示：在 479 collision pool 上，gap 2.0 的门在 same-line episode 上触发率 100%、
步曝光 69.25%，在 off-line episode 上触发率仅 2.56%——这证明episode级门选择性较好；
但训练后的参数更新仍可改变门外行为。这里只比较了1.0m和2.0m，不能外推成“所有可能gate宽度
都已穷尽”，但已足以否决继续把收窄宽度当作默认下一步。

这里的4.43%是固定训练池离线重放口径，不是30个online rollout metric的直接均值。1米臂
`metrics.jsonl`的30-update均值分别为`exploration_danger_gate_fraction=3.94%`和
`exploration_temporal_active_fraction=4.80%`；两者因状态分布和“门触发/时间块仍活跃”
定义不同，不应混成同一个曝光率。

因此：**不要继续扫 std 或 gate 宽度**。这两个轴已有实际对照且没有通过验收。hold 轴只有
K=50 的实现与结果；K10/K25 没有运行产物，当前结论只能是“未测试且已暂停”，不能写成
“失败模式已经一致”。

### 18.5 停止规则和未完成问题

本节是实验完成时使用的历史多面板停止规则。2026-08-06之后的新实验验收以§25.6为准：
Austin600与三张跨地图各600都必须正式评估，Austin600和crossmap1800都具有验收权；near400
和hard子集只作机制诊断。这里的旧阈值不自动成为新实验阈值，仍须逐实验预注册。

停止规则：

1. production 保持 `baseline`，不传 `--speed_exploration_mode`；
2. 不再扫 std / gate 宽度；hold 不得冒充已扫完。只有任务重新授权并预注册单变量对照时，
   才能在新目录测试 K10/K25；
3. 不要用 hard73 的改善（T 的 `54 -> 19`）当验收证据——它和 near400 的显著恶化同时出现，
   是特化信号不是接受信号；
4. 不要用跨地图`80 -> 46`单独给前向走廊门控时间相关速度噪声翻案——同一模型在
   Austin600和near400上都更差，而且跨地图面板正是设计该门控时用过的面板，
   `evidence_boundary`已经标注这不是干净的
   泛化估计；
5. 任何新臂必须同时报 Austin600 + near400 + 跨地图三个面板，并给 removed/created 和配对 p；
   **且必须是区间（≥4 个相邻 checkpoint），不接受单点**——理由见 18.1 的极差数据；
6. **预注册必须给每个必报面板都设阈值。** ordinary异线高速重加权的预注册只给跨地图
   设了阈值，导致按判据
   算"成功"、按实质必须否决（18.7.3）；
7. **同线暴露量是硬约束，不是可调参数。** 两次独立实验（gap 1.0 收窄门控、第一版两路
   ordinary 重加权）都因为压低同线暴露而摧毁了同线机制。任何新干预必须先算出它对
   `exploration_danger_gate_fraction` 和同线场景份额的影响，并在 run 内核对；
8. **不要再扫"干预强度"类旋钮。** σ（0.25/0.50/退火）、门控宽度（2.0/1.0）、门控覆盖率
   （0.15%/0.51%/17.5%/4.4%）、ordinary 采样权重（33.3%/60%）都已测过，结论一致：
   保守度可以在 regime 之间搬运，不能被消除。

**已关闭**：45u 延长（18.6，实测与 u30 无差别）。

**仍未测试**：hold 时长轴（K10/K25）。它从未运行，**不得写成已否决**。但按第 8 条，
它是又一个强度旋钮，先验很低；若要跑必须新授权 + 单变量预注册 + 区间评估。

关于 K25 还有两条 2026-08-06 固定下来的事实：

- **实现上不是纯运行参数。** `ppo/policy.py` 把 `TEMPORAL_RESAMPLE_STEPS = 50` 写成模块常量，
  没有对应 CLI 或 YAML 入口；K25 需要一处受控的代码/配置改动，不能只靠现有运行参数完成，
  预注册时必须把这处改动本身写进单变量说明。
- **它带一个验收矛盾，必须先解决再谈启动。** 如果 K25 的目的包含恢复 near400，则 near400
  不能只作无否决权诊断，必须先被重新预注册为该目标的守门指标；如果目标仍只是"四图逐图
  不差于 canonical BC"，则前向走廊时间相关速度探索 U44 已经通过，K25 没有启动理由。
  两种情况下都不存在"先跑了再说"的合法路径。

**合法的重开条件**：任务分布实质变化（新地图集合、新 opponent 速度域、新 interval），
或引入一个不属于"强度"类的新控制（例如让 actor 能显式区分 regime 的观测/结构改动——
但那超出"不修改 actor"的现行约束）。

### 18.6 前向走廊门控时间相关速度噪声延长到45 updates：方向关闭（2026-07-30）

原预注册只评checkpoint 45且**排除near400**。执行时按18.5规则5把near400加了回去，并改成
区间评估（u42–u45），因为单点对 corridor 臂不可判读。

**先验证它确实是纯延长**：45u与CT-v2 30u的U30 checkpoint参数一致，且前30个formal
update的`ego_collision_count`/`overtake_count`逐条一致。所以u31–u45可以直接读作
“同一策略再训15步”，不需要重建基线。

**训练侧**45个formal rows中，三个连续十步窗口几乎重合：

| 窗口 | 训练碰撞 | 平均最小 OBB 间距 | 碰撞角色 return |
|---|---:|---:|---:|
| u11–20 | 23.4 | 0.478 | −0.0885 |
| u21–30 | 24.5 | 0.470 | −0.1271 |
| u31–40 | 23.3 | 0.466 | −0.1120 |

**确定性区间**（u42–u45，与 B/U30 同场景配对；每格 `removed/created`，McNemar 精确检验）：

| 面板 | 45 updates区间 | 走廊门控30 updates | Production 30 updates | 对production的p |
|---|---|---|---|---|
| 跨地图 1800 | 43–46 / 1117–1134 | 46 / 1122 | 80 / 1142 | 1.1e-4 – 3.7e-4（四点全显著） |
| Austin600 | 17–20 / 332–345 | 20 / 338 | 14 / 366 | 0.33 – 0.68（均不显著） |
| near400 | 33–37 / 271–288 | 35 / 286 | 28 / 325 | 0.31 – 0.59（均不显著） |

**每个面板的45 updates区间都覆盖走廊门控30 updates的值**。结论：该探索方式在u11
附近已经收敛，
**延长到 45 updates 不改变训练侧也不改变任何评测面板**。这个方向关闭，不需要再跑。

### 18.7 Ordinary异线高速采样重加权：新接口、一次否决、一次设计缺陷（2026-07-30）

#### 18.7.1 要解决什么

18.4已确立前向走廊门控时间相关速度噪声的回退**全部集中在异线高速regime**。在那里
它付出一个稳定的全局
指令速度下降（跨地图 −0.16 m/s），而这个下降**买不到任何余量**：共同非碰撞 episode 的
配对 Δ 最小表面间距只有 +0.005 m，边缘带（<0.10 m）27→29。也就是纯损耗，代价约 20 次
跨地图超车和 28 次 Austin 超车。

门控宽度（暴露量）这个旋钮已在 gap 1.0 上否决。剩下的假说是**表征性的**：共享 GRU 没有
被训练分布逼着区分"异线高速"与"同线跟车"，于是学出单一全局速度设定点并外溢。干预方式是
提高异线高速 regime 的**采样权重**，而不是改变干预强度。

当前接口（YAML为`null`时与均匀路径等价）：

```text
ppo/ppo_config.yaml ordinary_offline_fast_fraction: <f|null>
  -> ppo/env.py CentralScheduleSubprocVecEnv
  -> ppo/scenarios.py ScenarioScheduler._init_ordinary / _next_ordinary_scenario
```

三路拆分：`same_line`（`opp_raceline == ego_raceline`）、`offline_fast`
（异线且 `opp_speedscale >= 0.7`）、`offline_slow`。`f` 是 offline_fast 的份额；
**same_line 恒定保持它在均匀池中的自然份额**，offline_slow 吸收余量。场景集合不变——
只有权重变，可达 scenario_id 集合与均匀采样完全相同（有测试守住）。
Austin ordinary600 的自然构成是 200/200/200，所以 same_line 锁在 33.3%，`f` 上界 2/3。

#### 18.7.2 第一版两路拆分：设计缺陷，u10 停止（不要重做）

第一版把 ordinary 拆成"异线快 vs 其余"两路、`f=2/3`。它把 **same_line 份额从 33.3%
压到 16.7%**，而 corridor 门几乎只在同线触发，于是无意中又动了已被 gap 1.0 否决的
暴露量旋钮。u10 时的证据（碰撞角色的池与采样在各臂间完全相同，所以这是策略信号而非构成变化）：

| 臂 | 碰撞角色 return (u1–10 均值) | 门控占比 |
|---|---:|---:|
| 两路 f=2/3 | −0.7131 | 15.06% |
| 前向走廊门控时间相关速度噪声 | −0.4012 | 17.43% |
| B | −0.7592 | — |

同线机制根本没有形成，因此该臂已在U10主动停止；它是设计缺陷记录，不应重做。

**这条与1米门宽合起来给出一条可复用的硬约束：走廊门控时间相关速度噪声的同线收益对
同线暴露量极度敏感，
任何后续干预都必须把同线暴露量当作不可移动的量。** 三路拆分锁死 same_line 后门控占比
恢复到18.6%（走廊门控同期18.6%），机制随之保住。

#### 18.7.3 Ordinary异线高速重加权比例0.6：跨地图双轴优于production，但near400否决

该臂完成30个formal updates；固定配置为
`ordinary_offline_fast_fraction=0.6`、门2.0m、hold50、std0.15。
异线高速权重 33.3%→60%（1.8×），same_line 锁死 33.3%，offline_slow 33.3%→6.7%。

区间 u27–u30，与 B/U30 同场景配对：

| 面板 | 异线高速重加权区间 | Production U30 | `removed/created` | McNemar p |
|---|---|---|---|---|
| 跨地图 1800 | **54–57 / 1146–1155** | 80 / 1142 | 50–53 / 25–29 | 3.8e-3 – 1.2e-2（四点全显著） |
| Austin600 | 16–21 / 363–370 | 14 / 366 | 8–10 / 12–17 | 0.23 – 0.83（均不显著） |
| **near400** | **63–77 / 291–306** | 28 / 325 | 18–21 / **55–67** | **8.4e-8 – 8.2e-5（四点全显著）** |

**正面结论（不要丢）**：ordinary异线高速重加权是**迄今唯一在跨地图上对production
双轴占优的实验**——碰撞54–57对80，同时超车1146–1155**高于**production的1142。
它也完全收复了走廊门控时间相关速度噪声的Austin超车损失
（363–370对走廊门控的332–338，production是366）。所以"异线速度税可以通过采样权重转移"这一点
是成立的，不要读成"重加权没用"。

**否决理由**：near400 碰撞 63–77，是 B 的 2.3–2.8 倍，四个 checkpoint 全部高度显著，
新造55–67个production没有的碰撞，且**全部集中在异线**（60–75，production是24）。
走廊门控30/45 updates在同一面板上对production**均不显著**，所以这不是这条线的共性
代价，是异线高速重加权特有的。

**失败机制**：ordinary异线高速重加权收复超车靠的是降低保守度，而保守度是一个
**全局标量**——它没有被消除，
只是被从"异线常规"移到了"最紧的贴身通过"。near400 全部是 B 贴身通过（间距 ≤0.1 m）的
场景，正是保守度下降最先兑现成碰撞的地方。**采样权重能改变保守度分布在哪个 regime，
不能改变它的总量。**

**预注册缺陷（记录以免复用）**：near400被列进评测面板，却只有跨地图设了阈值。
按那个不完整判据OFR算“成功”
（碰撞 ≤55 且超车 ≥1135）。**否决是按 18.4/18.5 的实质口径做的，不是按该判据。**
后续预注册必须给每个必报面板都设阈值。

### 18.8 验证规模与适用边界

2026-07-30 一轮共 40 个评测面板、22,400 episode、0 error（每个面板的 `summary.json`
均满足 `completed == total_scenarios` 且 `errors == 0`）。

**证据边界**：全部单seed（42）、固定面板。跨地图面板在设计前向走廊门时被用来标定
选择性，因此它对corridor系列**不是干净的泛化估计**；ordinary异线高速重加权的跨地图
优势继承同一边界。
Austin600 与训练起点沿赛道仅相距中位 2.1 m（episode 行程中位 45 m），也不是独立留出集。
本轮没有引入新地图。

前向走廊门控时间相关速度噪声的checkpoint曲线值得单独记住：Austin600碰撞在
U10/15/20/25/30为
`30 / 35 / 37 / 26 / 20`，U26-U30 带为 `36, 26, 22, 26, 20`（均值 26，极差 16）。
对比production的U24-U30带`18,16,14,14,13,13,14`（极差5）——**走廊门控版本不只是
更差，还明显更不稳定**，
所以它的 U30 = 20 不能当作代表值使用。

---

## 19. 其余已完成调查索引（2026-07-25 至 07-29）

以下每条都已完成并有配对证据，**不需要重跑训练或 eval**。除非任务分布发生实质变化，
直接引用结论即可。基线一律是 B/U30：Austin600 `14/366`、near400 `28/325`、hard73 `54/12`、
跨地图1800 `80/1142`。

### 19.1 已否决的方向

| 方向 | 改了什么 | Austin600 | near400 | hard73 | 结论 |
|---|---|---:|---:|---:|---|
| speed std 0.50（from U30, 15u） | 探索 std `0.15 -> 0.50` | 15-20 | 41-42 | 31-34 | 否决：hard 改善但 near 从 28 涨到 41+ |
| speed std 0.25（from BC, 30u） | 探索 std `0.15 -> 0.25` | 31 | 52 | 27 | 否决：Austin 翻倍 |
| std 退火 0.40→0.15（10u 内） | 前 10 update 高 std 后退火 | 44 | 68 | 32 | 否决：最差的一条，Austin 44 |
| ordinary 起点 50 → 150 | `--ordinary_startpoint_count 150` | 35 | 55 | 38 | 否决：三个面板全面变差 |
| interval-15 difficult pool | `--fixed_collision_pool_file` 指定困难池 | 见下 | — | — | 否决：配对 p 全部不显著 |

- **std 系列的统一失败模式**：在已测的std 0.25、0.50和0.40→0.15退火三条臂上，
  hard73改善同时伴随near400恶化。它们与§18呈现相似的特化/守门损失；这是已测配置的
  一致经验结果，不是对任意std的数学定律。现有证据不支持继续扫。
- **std 0.50不是“完全没学会”，而是明显专门化。** 该臂从B/U30续训15 updates，
  matched control同样从U30续训且保持std 0.15。hard73在U5/U10/U15分别
  `49->34`（removed 19/created 4，p=0.00260）、
  `45->34`（16/5，p=0.0266）、`43->31`（18/6，p=0.0227），三个checkpoint都改善；
  U15训练rollout平均最小OBB间距从`0.343m`升到`0.647m`，接近翻倍。可是near400超车
  同期为`337->290`、`312->292`、`330->292`，三点配对p分别
  `2.11e-7 / 0.0119 / 8.14e-6`；Austin在U5/U15也显著丢超车。结论是全局放大白噪声
  能把策略推向困难集安全解，但代价是自然/near-miss分布的进攻能力，不是actor没有容量。
- **std 0.25的fresh-start结果把同一签名放大。** 与B/U30同场景比较：Austin headline
  collision `14->31`（removed 8/created 25，p=0.00455），超车`366->332`
  （lost 44/gained 10，p=3.39e-6）；near400 collision `28->52`
  （23/47，p=0.00558），超车`325->268`（81/24，p=2.08e-8）。相反hard73 collision
  `54->27`（28/1，p=1.12e-7），超车`12->27`（lost 3/gained 18，p=0.00149）。
  hard两个轴都变好而Austin/near两个轴都变差，准确名称是**困难分布专门化**，不是
  “全局保守化”。
- **退火没有把安全能力带回baseline工作点。** 唯一schedule是updates 1–10把speed std
  从0.40线性降到0.15，updates 10–30固定0.15；没有做第二套退火schedule。U10/U20/U30
  的Austin为`21/361、18/362、44/351`，near400为
  `65/299、66/301、68/313`，hard73为`42/15、43/18、32/28`。到U30时near400共同
  非碰撞episode的最小间距中位相对control上移`+0.0508m`
  （paired sign p约`5.0e-13`），但Austin共同非碰撞episode反而下降`-0.0214m`；
  这不是“先高std学会、后低std完全回收噪声税”的预测轨迹。只否决这一条
  `0.40->0.15 in 10u` schedule，不是退火的一般数学不可能性。
- **ordinary 150**：该配置在三个面板上都变差，但实验同时增加了场景多样性、并在固定
  rollout预算下把每个ordinary场景的平均重复覆盖降到原来的约三分之一。因此只能否决
  `ordinary_startpoint_count=150` 这一整套配置；不能单独断言“多样性无用”，也没有直接证据
  证明是collision-role梯度被稀释。该flag和扩展逻辑已在可重建合同固化后退役，
  production固定50。
- **interval-15 difficult pool**：这不是L06/L12 reward对比。Control和treatment都使用
  production L06、post-pass off；唯一变量是collision-role场景池从广域U30碰撞池换成冻结的
  interval-15 collision+near-miss池。U5 hard73碰撞 `49 -> 47`
  （removed 9 / created 7，p=0.804）、甩尾 `12 -> 10`
  （removed 6 / created 4，p=0.754）、超车 `16 -> 14`
  （lost 7 / gained 5，p=0.774）；Austin600在U5/U10/U15分别为
  `14 -> 14`（removed 6 / created 6，p=1.000）、
  `16 -> 20`（removed 5 / created 9，p=0.424）、
  `15 -> 16`（removed 8 / created 9，p=1.000），没有任何checkpoint改善碰撞点估计。
  这只否决该interval-15困难池训练分布，**不能作为L12有效或无效的证据**；L12结论仍只由§16
  自己的reward A/B支持。通用fixed-pool loader没有独立A/B；用户仍决定在完整schema和
  测试合同写入`EXPERIMENTS.md`后退役该入口，删除理由是缩小活动训练面，不是证明loader
  抽象无用。

### 19.2 已确立的正面事实（可以直接引用，不要重测）

**1. 当前代码曾精确复现baseline。** 历史current-code reproduction实验
用当前代码从canonical BC重跑30 updates，产出的U30 actor与原始B/U30
**逐字节相同**，三个面板数字也完全一致
（14/366、28/325、54/12）。这不只是"结果接近"，是同一组参数。
所以后续所有 postpass/exploration/pool 改动都没有破坏 baseline 可复现性；
该历史来源已清理，今后直接使用canonical B，不必重训或重评。

**2. PPO 的改进能跨地图迁移。** 在三张 held-out 地图（Hockenheim / MoscowRaceway /
Nuerburgring，各 600 场景）上，U30 相对 BC：碰撞 `96 -> 80`（removed 37 / created 21，
pooled p=0.0479），超车 `1106 -> 1142`（p=3.18e-6）。逐地图碰撞点估计全部改善，但没有
单张地图达到 p<0.05（MoscowRaceway 最接近，p=0.0522）；三张地图的超车改善**全部显著**。
边界：1800 个固定场景身份是分析单元，三张地图不构成对所有赛道的代表性抽样。

| 地图 | BC -> U30 collision；removed/created（p） | BC -> U30 overtake；lost/gained（p） |
|---|---:|---:|
| Hockenheim 600 | `27 -> 26；9/8`（1.0） | `343 -> 356；3/16`（0.00443） |
| MoscowRaceway 600 | `43 -> 32；19/8`（0.0522） | `373 -> 385；7/19`（0.0290） |
| Nuerburgring 600 | `26 -> 22；9/5`（0.424） | `390 -> 401；2/13`（0.00739） |
| 合并1800 | `96 -> 80；37/21`（0.0479） | `1106 -> 1142；12/48`（3.18e-6） |

这张表同时说明不要把aggregate改善写成“三张图安全均显著”：碰撞只有pooled结果刚过
0.05，单图证据是同方向点估计；超车才是三图各自都显著。跨地图面板保存了每episode
轨迹并通过3,600条BC/U30 trace的key-set、finite和collision marker核对。

**3. 那些反复失败的困难场景不是物理不可解的。** 对 334 条 held-out hard failures，
共享 oracle library 找到 **334/334（100%）** 无碰撞干预，其中 71.6% 还能保留超车；
按 interval 分层的救回率也都是 100%。另有一组分场景搜索在 fishtail cohort 上 4/4 救回
（2 个还能变成超车）。
**边界（很重要）**：这些干预使用了场景的未来碰撞时刻和场景特定的优化参数，
**不可部署**，只是"在搜索的分段仿射动作族下的经验可达性见证"。反过来也成立——
搜索失败不等于物理不可能。所以：可以用它反驳"场景不可行"，不能用它当作 PPO 的上界或
任何 reward/shield 设计的直接依据。

**4. 单车鲁棒性没有被 PPO 换掉。** BC 与 U30 在三张地图 × 10/20/30% LiDAR beam masking
共 18 组 10 圈评估中，**全部 180 圈零碰撞完成**，且 U30 在 9 组匹配对比中平均圈速全部快于
BC（10/20/30% masking 分别快 1.0% / 1.4% / 2.0%）。共同的退化是速度：masking 从 10% 升到
30%，BC 平均圈速时间 +44.4%、U30 +43.1%。注意 `eval_singleagent.py` 的 `noise` 参数是
**随机遮蔽 36/72/108 条 beam**，不是加性噪声。

逐图的`U30 - BC`平均圈时差（秒/圈，负数为U30更快）为：

| 地图 | 10% masking | 20% masking | 30% masking |
|---|---:|---:|---:|
| Hockenheim | `-0.434` | `-0.943` | `-2.845` |
| MoscowRaceway | `-0.360` | `-1.216` | `-1.218` |
| Nuerburgring | `-1.291` | `-1.332` | `-1.578` |

这里每个单元只有一条10-lap轨迹，没有随机重复，不能给“更快”附显著性；它只证明在这
18条固定评估中PPO没有用单车稳定性换取多车收益。`noise=0.1/0.2/0.3`每个仿真步都保持
精确10%/20%/30%的beam为零，全部1,463,423 trace rows有限。

**5. B 的后期 checkpoint 是一个稳定低碰撞区，不是单点幸运。** 完整轨迹为：

| update | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Austin collision | 16 | 19 | 20 | 19 | 18 | 16 | 14 | 14 | 13 | 13 | 14 |
| Austin overtake | 350 | 353 | 349 | 348 | 350 | 350 | 357 | 358 | 360 | 361 | 366 |
| near400 collision | 54 | 43 | 64 | 55 | 55 | 55 | 47 | 44 | 27 | 34 | 28 |
| near400 overtake | 294 | 301 | 283 | 288 | 289 | 297 | 304 | 310 | 323 | 318 | 325 |

Austin从U26起稳定在13–14；U28/U29能省1次碰撞但分别少6/5次超车，U30在near400也不是
孤立低点（U28–U30均处于27–34带）。没有checkpoint在两个面板、两个轴上同时支配U30，
所以选U30有依据。near400较Austin更易随checkpoint变化，后续新方法必须报相邻checkpoint
区间而不是只挑单点。

### 19.3 未形成独立结论的历史接口

| 接口 | 状态 |
|---|---|
| Outcome-aware hard pool | 历史实现与测试已清理，**没有跑过 A/B**，不能继承§17 的任何结论；重建合同见`EXPERIMENTS.md` |
| Group 12 critic LR 5e-4 选择器 | 历史实验已训练并完成判读；原checkpoint已清理，当时的排名脚本按 GUIDE §3 已停用，不作为新实验入口 |
| Group 13 GRU/head LR 解耦 2×2 | run.sh 中注释，未运行 |
| clip 0.25 / 45-update 延长 | 历史实验已训练；clip0.25原checkpoint已清理，clip0.20完整U1--U45已迁入canonical B |

---

## 20. Actor可观测性、oracle可达性与制动响应曲面（2026-07-27）

### 20.1 结论先行

这组工作**没有修改或训练actor**，也没有提出production后处理。它依次回答三个容易被混淆的
问题：

| 问题 | 实验 | 已建立的事实 | 不能推出什么 |
|---|---|---|---|
| actor现有输入/GRU是否包含关键动态信息 | 冻结B/U30，线性解码当前361D observation和1680D GRU hidden | 安全状态下yaw/slip/上一转向强可解码；closing speed明显依赖GRU历史；失败状态精度下降且按cohort不同 | 不能仅凭R²证明policy已经会选择正确动作，也不能证明需要/不需要改actor |
| 当前残余碰撞在动作接口下是否可救 | 对Austin当前13个ego failures做未来时刻辅助的分段动作搜索 | 13/13找到无碰撞见证，10/13仍超车 | 不是0/600模型成绩，不可部署，不证明PPO必能学到 |
| 这类动作能否跨未见起点复用、局部回报是否有坡度 | 13动作库覆盖334个held-out failures；再扫持续制动幅度/提前量 | 334/334至少有一个库动作能救；持续`-0.15m/s`已产生17–46%救援和显著平均return增益 | 共享动作仍使用未来碰撞时刻；150步协调override不是PPO一步策略梯度 |

综合结论：

1. “残余失败是车辆动力学或动作上下界不可解”被否决；
2. “actor对所有关键状态完全不可观测”没有证据支持，但失败状态的表征精度不是饱和完美；
3. “局部回报在小制动处完全平坦、PPO没有任何方向”被制动扫描否决；
4. 真正难点是**在正确状态、正确时刻产生持续动作并保持跨regime泛化**。§18的时间相关探索
   正是据此设计；它确实学会同线减速，却因off-line迁移损失未通过production gate。

因此oracle只保留为**机制上界/可达性诊断**。按用户约束，不添加运行时shield，不把未来
碰撞时刻、oracle参数或特权触发器接到部署policy，也不把本组结果改写成
“PPO + imitation”已验证方案。

### 20.2 冻结actor可观测性：设计与结果

对象是B/U30当前Austin600的600条trace。逐步重放actor得到474,949个动作，
最大raw-action误差为**0**；说明抓取的361D observation、GRU hidden与实际actor调用一致。
以事件前1.5秒、每2步采样构造44,281行：

- 407个安全episode用于训练，124个安全episode用于validation，56个安全episode作safe test；
- 13个ego collision episode全部隔离为collision holdout，没有进入探针训练；
- safe episode按episode身份切分，避免同一轨迹跨train/test泄漏；
- 探针是带weight decay、validation early stop的单层线性模型；它测线性可解码性，
  不是训练一个新policy。

四个严格因果target：

| target | 定义 |
|---|---|
| yaw rate | 相邻heading的wrapped差分 / `0.01s` |
| slip angle | 过去5步位移方向与车体heading之差；速度低于1m/s置0 |
| previous executed steering | 上一trace行的实际clipped steering，reset置0 |
| relative closing speed | 过去5步ego/opponent世界速度差投影到ego纵轴 |

主要R²（都相对同一train target mean计算）：

| 特征源 | cohort | yaw | slip | previous steer | closing speed |
|---|---|---:|---:|---:|---:|
| GRU hidden | safe test，4,138行 | **0.908** | **0.904** | **0.925** | **0.734** |
| current observation | safe test | 0.779 | 0.720 | 0.772 | 0.183 |
| GRU hidden | collision holdout，952行 | 0.424 | 0.361 | 0.353 | **0.611** |
| current observation | collision holdout | **0.741** | **0.595** | **0.618** | -0.527 |

不能从最后两行得出“GRU系统性丢失本体信息”。collision aggregate混合了性质完全不同的
cohort：

| collision cohort | 行数 | hidden yaw/slip/closing R² | current observation yaw/slip/closing R² |
|---|---:|---:|---:|
| OL1非甩尾 | 525 | `0.706 / 0.541 / 0.520` | `0.780 / 0.628 / -0.592` |
| fishtail | 300 | `0.876 / 0.872 / 0.850` | `0.774 / 0.835 / -0.073` |
| 其他ego collision（主要wall/异常动态） | 127 | `-0.907 / -0.751 / 0.046` | `0.614 / 0.122 /`低或不稳 |

失败集合里极少数wall/异常动态贡献了很大平方误差，聚合比较存在composition/Simpson风险。
正确口径是：

- recurrent hidden对**需要时序的closing speed**贡献明确，单帧observation在collision
  holdout上甚至为负R²；
- fishtail窗口里hidden仍保留较强yaw/slip信息，不能说GRU完全看不到失稳；
- OL1失败上单帧几何对yaw/slip略优，而hidden对closing明显更有用；
- 绝对误差在失败状态上高于安全状态，所以“可解码”不等于“精度足以闭环决策”；
- 线性探针只回答信息是否以线性方式存在，不能证明actor输出头会利用它，也不能单凭
  aggregate R²决定是否修改actor结构。

这项诊断没有新增模型权重；被测actor就是§1.3登记的B/U30。

### 20.3 Austin13分场景oracle：经验可达性，不是模型成绩

对B/U30当前Austin600中的13个ego collision分别搜索一个1.5秒分段仿射动作schedule：
5段×0.3秒，每段可缩放/偏置steering并改变speed；CEM为6轮×24候选、elite 6，
总计`13 × 6 × 24 = 1,872`次rollout。schedule起点使用**已知baseline未来碰撞时刻**，
每个场景独立优化。baseline replay和best-result replay均精确，actor权重未变。

| cohort | baseline failures | rescued | rescued且overtake |
|---|---:|---:|---:|
| fishtail | 4 | 4 | 2 |
| OL1非fishtail | 7 | 7 | 6 |
| 其他ego collision | 2 | 2 | 2 |
| 合计 | **13** | **13** | **10** |

若只在这13个失败上使用各自best intervention、并假设其余587个episode完全不变，会得到
`0 collision / 376 overtake`（baseline为13 ego collision / 366 overtake）。这只是
**privileged counterfactual arithmetic**，不是一个可运行actor的600场景成绩。

为了防止过度解释，还把13个最优干预移植到13个matched safe controls：

- 新造3个collision；
- 丢失2次原有overtake。

这证明动作schedule需要按状态/时机使用，不能把“某段大制动能救失败”直接全局混入BC数据
或部署后处理。分场景搜索失败本来也不能证明物理不可行；本次正面结果只说明在已搜索的
动作族里存在可行见证。

### 20.4 334条held-out hard failures上的共享动作库

为了检查Austin13的干预是不是只记住13个位置，把上节13个best schedules冻结成共享库，
逐一应用到§9.1的334条未见起点U30 failures。没有针对334条再做CEM；总rollout严格为
`334 × 13 = 4,342`，全部finite，baseline replay精确。

| 分层 | 场景 | 至少一个库动作无碰撞 | 至少一个库动作无碰撞且overtake |
|---|---:|---:|---:|
| 全部 | 334 | **334（100%）** | **239（71.6%）** |
| interval 8 / 10 / 12 / 15 | 99 / 89 / 73 / 73 | 各100% | 74.7% / 71.9% / 63.0% / 75.3% |
| postOT / rear-end / side / wall | 168 / 106 / 38 / 22 | 各100% | 78.6% / 71.7% / 73.7% / 13.6% |
| fishtail / non-fishtail | 156 / 178 | 各100% | 75.0% / 68.5% |

这否决“Austin13只靠特定位置参数才可救”的强版本：同一小库能跨未见起点、interval和
collision mode复用。但它仍然使用每个新场景的**已知未来碰撞时刻**对齐schedule，
所以不是部署policy，也不能拿71.6%直接预测PPO超车率。

随后做了candidate-ranking探针，问“给定同一13候选库，当前observation或冻结hidden能否
选出安全动作”。按startpoint严格切为44 train / 15 validation / 15 test；test含72场景。
决策特征取override开始前、距未来碰撞`0.69–1.8s`的状态。测试结果：

| selector | test top-1 safe | top-1 overtake | pairwise safe accuracy |
|---|---:|---:|---:|
| action-only | `68/72 = 94.4%` | 23 | 85.9% |
| current observation + action | `68/72 = 94.4%` | 22 | 85.3% |
| frozen hidden + action | `66/72 = 91.7%` | 24 | 84.4% |
| best fixed library schedule #12 | `68/72 = 94.4%` | 23 | — |
| uniform-random期望 | `56.2/72 = 78.0%` | 24.2 | — |

observation与best fixed完全相同；hidden相对fixed只有1个独有救回、3个独有失败，
McNemar `p=0.625`。fixed schedule的4个失败全部集中在同一个startpoint。
因此没有证据支持增加复杂state-conditioned library selector；**停止规则是不用这个
null结果声称“状态信息不存在”**，因为fixed schedule已占68/72，剩余头寸只有4，而且
所有schedule timing仍用了未来碰撞时刻。这个probe检验候选排序，不检验closed-loop
policy能否生成/持续执行动作。

### 20.5 持续制动幅度 × 提前量响应曲面

早期12场景pilot曾扫更密的制动幅度`-0.15…-3.0m/s`和提前量
`0.5/0.7/1.0/1.5/1.8s`，用途只是确认全量网格；它不是独立结论。正式扫描覆盖全部334条
held-out failures，固定steering不变、持续1.5秒，只扫：

```text
speed delta = 0, -0.15, -0.30, -0.60, -1.00, -3.00 m/s
requested lead = 0.5, 1.0, 1.8 s
```

共`334 × 6 × 3 = 6,012`个唯一rollout；0-delta逐条精确复现baseline，所有结果finite。
因部分碰撞发生得早，1.0/1.8秒requested lead会被episode起点截断；主表同时保留
requested和effective lead，以下先报全334场景的requested-lead结果：

| speed delta | lead 0.5s rescue | lead 1.0s rescue | lead 1.8s rescue |
|---:|---:|---:|---:|
| `-0.15m/s`（当前speed std的1σ） | `58/334 = 17.4%` | `102/334 = 30.5%` | `155/334 = 46.4%` |
| `-0.30m/s` | 30.2% | 56.3% | 67.4% |
| `-0.60m/s` | 58.7% | 74.9% | 87.1% |
| `-1.00m/s` | 71.9% | 88.6% | 91.9% |
| `-3.00m/s` | 85.6% | 94.0% | 94.6% |

最关键的不是`-3m/s`端点，而是**1σ持续制动已经有局部收益**。`-0.15m/s`在
0.5/1.0/1.8秒提前量下，平均episode return相对baseline分别
`+0.418 / +0.711 / +1.078`，平均post-schedule discounted return增益为
`+0.390 / +0.640 / +0.909`。救援率和回报随幅度、提前量平滑上升，不是等到`-2`或`-3`
才突然跳变。因此：

- “oracle端点约20σ，所以PPO邻域完全没有梯度”是错误推论；
- “小动作仍然全撞，所以回报平台为零”也被数据否决；
- 但这里一次override协调了连续150步，**不等于**PPO每步独立采样一个1σ动作；
- 这项实验支持研究探索的时间结构，却不保证把噪声保持50步就能获得production收益。
  §18正好给出闭环答案：T/CT-v2确实学会持续减速并减少same-line碰撞，但共享参数造成
  off-line迁移和near400损失，所以最终未采纳。

### 20.6 数据边界、清理后保留内容与停止规则

本章把原来分散的可观测性replay、诊断notebook、13场景oracle、334场景共享库、
ranking probe、12场景pilot和334场景全量braking landscape合并为一条因果链。
后续清理历史分析产物后，仍可直接引用本章的设计、样本量、分层结果与边界；
会永久失去的是逐scenario动作参数、每个candidate的身份和重新做未列分层的能力。

本章所有实验：

- 都使用同一个B/U30 actor，模型身份只引用§1.3，不新增analysis产物哈希；
- 都保持B/U30 actor权重不变；可观测性实验的逐步actor重放最大动作误差为0，
  oracle/library/braking实验则分别完成baseline动作与结局复现检查；
- 除可观测性probe拟合了离线线性读出外，没有训练或替换PPO actor；
- oracle/library/ranking都使用未来碰撞时间，只能作diagnostic；
- 未测试“oracle imitation + PPO”、辅助动作loss、DAgger、runtime shield或actor输入扩展，
  不得把这些写成已失败或已成功。

停止规则：

1. 不再用“物理不可解”“动作接口上限”解释当前残余失败；
2. 不以oracle counterfactual `0/600`作为模型指标或论文production结果；
3. 不根据ranking probe的null结果增加复杂selector，也不据此宣称表征充分/不足已经定案；
4. 不重复全量制动扫描；若任务分布、动作上下界或episode horizon没有改变，其结论已足够；
5. 任何后续学习方案仍必须是单次PPO可部署actor，并在Austin600、near400和未被其设计使用的
   泛化面板上闭环验证；不接受运行时未来信息或动作后处理。

---

## 21. 清理前核心证据固化（2026-07-30）

本节专门服务于用户决定清理历史分析产物、实验工具和回归测试源码后的连续性。
2026-07-30 清理前盘点为：

| 对象 | 清理前规模 | Markdown 中保留到什么粒度 |
|---|---:|---|
| 历史分析产物 | 约389 MiB、约1,800个文件、41个顶层实验ID | 实验问题、控制变量、分母、主/守门/泛化结果、配对计数与p值、关键分层、机制和停止规则 |
| 实验工具 | 46个Python + 19个shell runner，约23,218行Python | 算法、CLI、输入输出合同和关键常量见 `EXPERIMENTS.md`；不保证逐行源码重建 |
| 回归测试源码 | 清理前18个Python文件：16个`test_*.py` + 2个历史工具，约3,457行 | 原测试的case、fixture、断言和工具逻辑见 `EXPERIMENTS.md`；完整历史套件会消失，但核心reward候选合同已合并成`scripts/test_screen_reward_candidate.py`并保留55项可执行测试 |

本节补的是前文最难从汇总数字反推的三类信息：41个实验ID的最终归宿、B/U30残余失败的
具体身份和时间结构、以及13条oracle见证动作。其余headline结果仍以§16–§20为准，不在
这里重复造第二套数字。

### 21.1 41个历史分析实验ID的归宿

下表按清理前顶层目录逐项登记。`固化位置`说明删除目录后到哪里理解该实验；“容器/被替代”
表示它不是独立新结论，不能把目录存在本身算作又一次实验。

| 实验ID | 最终状态与保留结论 | 固化位置 |
|---|---|---|
| `_rejected_offlinefast23_dilutes_sameline_20260730_003906` | 第一版两路OFR设计缺陷；压低same-line份额，U10停止，不重做 | §18.7.2 |
| `arm_band_comparison_20260730` | exploration各臂后期band统一比较；B仍为production | §18.1、§18.6–§18.7 |
| `baseline_late_checkpoint_stability_20260728` | B的U20–U30轨迹；U26–U30为13–14碰撞稳定低区 | §19.2(5) |
| `baseline_repro_currentcode_seed42_20260728` | 当前代码从BC精确复现B/U30权重和三面板结果 | §19.2(1) |
| `baseline_reward_seed42_u30_austin600_20260726` | 默认四项reward逐transition合同、13个ego失败taxonomy与risk时机 | §16.2、§21.2 |
| `bc_fishtail_targeted_validation` | 9个BC甩尾sentinel；证明小panel会漏掉PPO新造失败，不得定案 | §16.5 |
| `conditional_temporal_exploration_20260728` | C/CT旧required-decel门的离线预飞和online交叉校验；门曝光过稀 | §18.1–§18.3 |
| `crossmap_bc_u30_20260729` | BC/B/T三图评估、CT-v2预飞；BC→B跨图碰撞96→80、超车1106→1142 | §18.1、§19.2(2) |
| `ctv2_45u_band_eval_20260730` | 前向走廊门控时间相关速度噪声U42–U45四图+near区间；延长无收益 | §18.6 |
| `ctv2_checkpoint_curve_20260729` | 前向走廊门控时间相关速度噪声U10/15/20/25/30 Austin+near曲线 | §18.8 |
| `ctv2_corridor_temporal_20260729` | 前向走廊门控时间相关速度噪声2米门宽主实验；跨图改善、Austin/near损失和same/off-line机制 | §18.1、§18.4 |
| `ctv2_corridor_temporal_45u_20260729` | 完整45-update训练容器 | §18.6 |
| `ctv2_corridor_temporal_45u_20260729_interrupted_205032` | 被完整45u run替代的中断副本；无独立结论 | §1.0、§18.6 |
| `ctv2_crossmap_band_20260729` | 前向走廊门控时间相关速度噪声U27–U30跨图区间；证明单checkpoint不可判读 | §18.1、§18.5 |
| `ctv2_gap1_band_eval_20260729` | gap1 U27–U30 band；跨图收益消失 | §18.4 |
| `ctv2_gap1_corridor_temporal_20260729` | gap1训练容器；唯一变量是front gap 2.0→1.0m | §18.4 |
| `ctv2_gate_width_sweep_20260729` | 真实训练池1.0/2.0m gate曝光对照；不代表穷尽所有宽度 | §18.4 |
| `ctv2_late_stability_20260729` | 前向走廊门控时间相关速度噪声U26–U30 Austin晚期波动带 | §18.1、§18.8 |
| `following_response_reward_u30_austin600_20260725` | required-deceleration/escalation离线候选；未获训练准入 | §16.9 |
| `interval15_difficult_pool_experiment_20260727` | L06下只换interval15 collision+near-miss池；无checkpoint改善 | §19.1 |
| `l12_deterministic_transfer_20260726` | cache372有效、heldout-near600无改善；排除纯interval错配 | §16.8 |
| `l12_heldout_hard_instrument_20260726` | 21,600候选instrument；hard334改善但near400超车显著下降 | §9.1、§16.8 |
| `l12_train_eval_diagnostic_20260726` | L12训练侧碰撞/间距改善与eval背离诊断；不是额外A/B | §16.8 |
| `offline_fast_band_eval_20260730` | ordinary异线高速重加权比例0.6的U27–U30五面板结果容器 | §18.7.3 |
| `offline_fast_reweight_20260730` | OFR训练、预注册和同线份额锁定；near400否决 | §18.7 |
| `ordinary150_from_bc_seed42_20260728` | ordinary starts 50→150整套配置否决；多样性与覆盖度混杂 | §19.1 |
| `postpass_formula_comparison_u30_u1_u5_u10` | no-q/q/q²相同gate下的离线剂量比较；no-q未训练 | §16.5 |
| `postpass_training_diagnosis` | q²训练遥测、阶段比较和因果时机；剂量小且触发偏晚 | §16.5 |
| `risk_potential_offline_sweep_u30_austin600_20260726` | B0/L12/L20/SUM06/SUM12/SUM20离线扫；sum不增加墙边际信息 | §16.7 |
| `singleagent_bc_u30_noise_20260729` | 3图×3 masking×BC/U30，180圈零碰撞；noise是beam masking | §19.2(4) |
| `speedstd025_from_bc_seed42_20260728` | std0.25 hard特化、Austin/near双轴恶化 | §19.1 |
| `speedstd050_experiment_20260727` | std0.50困难面板安全改善但near/Austin代价；只到U15 | §19.1 |
| `speedstd_anneal040_to015_seed42_20260728` | 唯一0.40→0.15/10u退火；没有把能力带回baseline工作点 | §19.1 |
| `structured_speed_exploration_20260728` | 逐步独立、条件白噪声、全局时间相关和条件时间相关四组实验及OL1机制重放；全局时间相关速度噪声确实学到持续减速 | §18.1–§18.4 |
| `u30_actor_observability_20260727` | 冻结actor线性可观测性；动作重放误差0 | §20.2 |
| `u30_braking_landscape_heldout334_20260727` | 早期较宽扫描容器；正式结论由minimal全量网格取代 | §20.5 |
| `u30_braking_landscape_minimal_heldout334_20260727` | 334×6幅度×3提前量正式响应曲面 | §20.5 |
| `u30_observability_oracle_20260727` | 可观测性/oracle notebook与汇总容器；无独立模型结果 | §20.1–§20.3 |
| `u30_oracle_reachability_20260727` | Austin13分场景CEM：13/13救回、10/13仍超车 | §20.3、§21.3 |
| `u30_shared_oracle_library_heldout334_20260727` | 冻结13动作库覆盖334/334；239可保超车 | §20.4 |
| `u30_shared_oracle_ranking_probe_20260727` | fixed schedule已68/72，state selector无增益；停止复杂selector | §20.4 |

### 21.2 B/U30 的13个ego碰撞身份与risk时机

这是当前Austin600中actor责任失败的完整身份集合；headline `14 collision` 另含1个
`opp-wall`，不在本表。`risk first/continuous lead`分别是首次risk active和碰撞前最后一段
连续risk的提前量；它们不能互换。表中数值按保存trace重放，时间单位秒。

| scenario | outcome / mode | fishtail | collision | risk first / continuous lead | yaw50 lead | min wall |
|---|---|---:|---:|---:|---:|---:|
| `ol0_e1509_o1500_s0.8` | ego-opp / post-overtake-rear | yes | 5.94 | 0.29 / 0.29 | 0.81 | 0.594m |
| `ol0_e1551_o1546_s0.5` | ego-wall / wall | no | 1.04 | 0.22 / 0.22 | 0.61 | 0.000m |
| `ol0_e210_o218_s0.7` | ego-opp / post-overtake-rear | no | 3.49 | 0.25 / 0.25 | 1.36 | 0.447m |
| `ol0_e335_o342_s0.8` | ego-opp / post-overtake-rear | yes | 6.69 | 1.99 / 0.36 | 1.48 | 0.552m |
| `ol0_e922_o926_s0.7` | ego-opp / post-overtake-rear | yes | 5.41 | 0.24 / 0.24 | 0.67 | 0.661m |
| `ol1_e1425_o1440_s0.5` | ego-opp / rear-end-opponent | no | 1.72 | 0.45 / 0.45 | 0.78 | 0.386m |
| `ol1_e1509_o1524_s0.5` | ego-opp / rear-end-opponent | no | 2.41 | 0.52 / 0.52 | 1.50 | 0.551m |
| `ol1_e1509_o1524_s0.6` | ego-opp / post-overtake-rear | no | 5.72 | 2.24 / 2.24 | — | 0.298m |
| `ol1_e210_o225_s0.6` | ego-opp / rear-end-opponent | no | 2.36 | 0.53 / 0.53 | 1.44 | 0.609m |
| `ol1_e252_o267_s0.5` | ego-opp / rear-end-opponent | no | 2.99 | 0.39 / 0.39 | 1.50 | 0.467m |
| `ol1_e587_o602_s0.5` | ego-opp / post-overtake-rear | no | 3.69 | 1.37 / 0.27 | 1.21 | 0.177m |
| `ol1_e629_o644_s0.5` | ego-wall / wall | no | 7.46 | 0.38 / 0.38 | 1.49 | 0.000m |
| `ol1_e713_o728_s0.5` | ego-opp / post-overtake-rear | yes | 2.62 | 0.62 / 0.16 | 1.50 | 0.083m |

这13个身份同时是§20.3 oracle搜索的source scenarios。表内4个fishtail、8个OL1和2个wall
不是互斥分组：`ol1_e713...`同时属于OL1与fishtail，`ol1_e629...`同时属于OL1与wall。
这解释了为什么不同章节按互斥cohort或按标签计数时不能直接相加。

### 21.3 13条oracle见证动作

每条schedule在已知baseline未来碰撞前`lead`秒开始，持续150步；5段各0.30秒。每段执行：

```text
steer_override = steer_scale[i] * actor_steer + steer_bias[i]
speed_override = actor_speed + speed_delta[i]
```

以下保留到3位小数，足以恢复动作族和做数量级检查，但不是checkpoint或可部署policy。
`result`是干预后的确定性结局。

| source scenario | result | lead | steer scales 0..4 | steer biases 0..4 | speed deltas 0..4 (m/s) |
|---|---|---:|---|---|---|
| `ol0_e1509_o1500_s0.8` | overtake | 1.558 | `0.890/0.899/0.953/0.992/0.900` | `+0.025/+0.010/-0.040/-0.041/-0.018` | `-0.930/-0.759/-0.582/-0.723/-0.242` |
| `ol0_e1551_o1546_s0.5` | overtake | 0.577 | `1.182/0.553/0.688/0.571/0.805` | `+0.050/+0.157/+0.061/+0.035/+0.061` | `-0.660/+0.015/-0.174/+0.564/+0.038` |
| `ol0_e210_o218_s0.7` | overtake | 0.786 | `0.943/0.910/0.874/0.981/0.957` | `-0.030/+0.024/+0.071/+0.040/+0.060` | `-0.183/-0.391/-0.465/-0.563/-0.315` |
| `ol0_e335_o342_s0.8` | follow | 1.159 | `0.956/0.942/0.725/0.909/0.888` | `+0.052/+0.053/+0.058/+0.005/+0.057` | `-0.406/-0.560/-0.021/-0.762/-0.200` |
| `ol0_e922_o926_s0.7` | overtake | 1.800 | `0.914/0.645/0.616/0.885/0.941` | `+0.039/+0.002/-0.019/+0.014/-0.005` | `-0.145/-0.924/-0.399/-1.280/-0.980` |
| `ol1_e1425_o1440_s0.5` | overtake | 0.910 | `0.750/0.750/0.750/0.750/0.750` | `+0.080/+0.080/+0.080/+0.080/+0.080` | `-1.000/-1.000/-1.000/-1.000/-1.000` |
| `ol1_e1509_o1524_s0.5` | overtake | 1.360 | `1.189/0.928/0.979/0.164/0.963` | `-0.003/-0.030/+0.033/+0.040/-0.063` | `-1.110/-0.964/-1.224/-1.717/-0.822` |
| `ol1_e1509_o1524_s0.6` | follow | 1.475 | `0.882/1.194/1.268/1.085/0.564` | `-0.028/+0.020/-0.052/+0.017/-0.165` | `-1.205/-2.539/-2.061/-1.528/-2.314` |
| `ol1_e210_o225_s0.6` | overtake | 1.800 | `0.927/0.837/0.819/0.759/0.674` | `-0.063/-0.025/-0.043/-0.049/-0.026` | `-2.450/-1.462/-0.745/-1.207/-1.443` |
| `ol1_e252_o267_s0.5` | overtake | 1.686 | `0.789/1.134/0.776/0.773/1.124` | `+0.021/-0.003/-0.008/+0.021/-0.038` | `-2.673/-1.945/-2.935/-2.787/-1.183` |
| `ol1_e587_o602_s0.5` | overtake | 0.952 | `0.708/0.865/1.060/0.903/0.799` | `-0.021/+0.120/+0.062/+0.077/-0.071` | `-1.423/+0.038/-0.222/-1.677/-0.966` |
| `ol1_e629_o644_s0.5` | overtake | 0.300 | `0.925/1.215/0.620/0.987/1.006` | `-0.004/-0.003/+0.007/-0.013/-0.010` | `-0.406/-0.214/-0.352/-0.529/+0.750` |
| `ol1_e713_o728_s0.5` | follow | 1.800 | `1.000/1.000/1.000/1.000/1.000` | `0/0/0/0/0` | `-3.000/-3.000/-3.000/-3.000/-3.000` |

移植到334个held-out failures时，不是按source scenario固定配对，而是对每个新scenario尝试
全部13条schedule，并继续用该新scenario的已知baseline碰撞时刻对齐。覆盖率100%不能解释为
“上表每一条都通用”，也不能从表中挑一个动作做runtime后处理。

### 21.4 清理后仍可复核和不能复核的边界

**仍可复核：**

- production actor身份、默认配置和各treatment checkpoint身份；
- 各实验的控制变量、样本量、panel角色、headline、removed/created、p值和关键分层；
- reward/pool/exploration/oracle各方向为什么接受或否决；
- 固定panel生成合同、outcome/taxonomy/McNemar/trace schema；
- B/U30残余13个ego失败的身份、分类和关键时机；
- oracle动作族的结构、13条具体参数、搜索预算与不可部署边界；
- 关键脚本的CLI、函数签名、常量、输入输出和测试断言（见`EXPERIMENTS.md`）。

**不能再复核：**

- 没写入本文的新分层、新阈值或逐transition统计；
- treatment所有removed/created scenario ID全集和每条trace；
- 334×13 candidate矩阵、334×18 braking网格的逐scenario行；
- 原始浮点全精度oracle参数（上表保留到3位小数）；
- 脚本和测试的逐行源码、注释、异常文本及执行顺序细节；
- 旧report/notebook/SQLite中的临时查询和可视化。

因此清理后的正确声明是：

> 已经形成的核心判决、关键数字、物理机制和重开边界已固化；原始证据和任意再分析能力已
> 主动放弃。Markdown是实验记录，不是原始数据或可执行代码的无损压缩。

若将来需要发表级逐场景复核，必须重新运行对应固定panel；不得从本节汇总表反向伪造原始
CSV/JSON/trace。若只为继续当前PPO工程和避免重复已否决方向，四份`.agents`文档已经覆盖
所需核心信息。

### 21.5 Panel、fixed pool 与选择审计的可恢复性校正

清理前曾出现一个过强判断：hard73、hard334、near400 被认为没有存活 eval trace，因此
删除历史产物后只能重新分类21,600个候选。2026-07-30按**物理场景 key**重做集合核验后，
该判断被否定。现存trace可精确恢复：

```text
near400 = T_ctv2_near_Austin 的400个trace key
hard334 = control_u15_Austin的734个key - near400
hard73  = interval15_control_u0015_Austin的1073个key - Austin600的600个key - near400
```

三个结果分别为400/334/73条，与原始ScenarioSpec列表逐条一致；参与差集的集合互不重叠。
因此未来新actor仍可在**同一物理场景身份**上评估，并按
`(map, opponent_raceline, ego_idx, opp_idx, opponent_speed_scale)`与历史结果配对。
丢失的是原始`scenario_id/pool/startpoint_ordinal`等记录身份，而不是仿真所需的物理输入。

一个同时被纠正的错误是：跨raceline时不能由
`(opp_idx - ego_idx) mod 2096`推回`interval_idx`。hard73只有37/73满足这个数值关系；
raceline0/2的opponent index经过各自轨迹几何映射。hard73必须使用显式ScenarioSpec，或
使用上述已经逐条验证的集合差恢复。

interval-15 fixed训练池的物理输入也有第二份存活快照：对应训练run保存的
`collision_scenarios.json`含有有序229条ScenarioSpec，逐条等于原wrapper里的
`entries[*].scenario`；`collision_cache_info.json`保存了selection、sampling和来源摘要。
所以训练过的场景及顺序不会随历史产物清理而消失。不过loader要求的
`source_label/source_outcome/min_clearance`仍只在原合规wrapper和完整候选标签中，
要逐字复现pool provenance，仍应保留该wrapper。

保留价值应按目标分级：

1. **继续工程和配对eval**：hard73、hard334、near400三个显式panel和interval-15
   fixed-pool wrapper，总计约0.45 MiB，已保存在
   `post-trained/panels/heldout_hard_v1/`。
2. **保留provenance摘要**：`design_manifest.json`和`classification_summary.json`
   约0.07 MiB，已随panel保存；记录只用冻结U30选择、train/held-out起点不相交、
   排除已有训练/eval起点。
3. **发表级独立选择审计**：21,600条候选和对应标签约21.16 MiB，也已随panel保存。
   只有这一级能重新执行selection并核对原panel成员；重新跑分类会形成新panel，不能恢复
   “当时未查看treatment”的历史事实。
4. **无需重复保存**：Austin600、跨地图panel、near600已有确定性生成合同和存活trace；
   partial/final episodes重复副本、日志、stdout和重复scenario规格不属于必要输入。

所以“约24 MiB全部是继续实验所必需”不正确；更准确的结论是：约0.45 MiB是低成本、
高价值的操作性输入，约21.23 MiB是可选但有价值的selection audit证据。两级现已统一保存，
历史分析输出可以清理。

## 22. PPO职责收口与语义等价性（2026-07-31）

### 22.1 改动边界

本轮是结构重构，不是新实验。production reward、actor 361D输入与12-key结构、
P20/381D critic合同、479 collision pool、600 ordinary pool、rank奇偶角色、seed、
PPO超参和三种保留探索模式均未改变。

重构后的生产结构固定为：

```text
ppo/env.py        单环境、前向走廊门、一env一worker VecEnv
ppo/policy.py     actor、四critic、P20 extractor、探索分布与时间状态
ppo/reward.py     progress、OBB/map-wall geometry与固定四项reward
ppo/scenarios.py  scenario、queue、collision classification/cache
ppo/rollout.py    recurrent buffer、warm-up与formal PPO
ppo_config.yaml   固定配置
```

训练记录与checkpoint保存并入`utils.py`。移除的活动入口为`detached_gru`、
`--env_workers`和`--reclassify_collisions`；不再生成与最后一个正式update重复的
`actor_final.pth`。cache生命周期改为：空目录分类并完整写入，完整且identity匹配则加载，
partial或identity mismatch拒绝写入并要求新的空目录。

### 22.2 三模式配对短训练

重构前后均从canonical BC开始，固定seed42、4 env、800 steps/env、batch1600、
actor/critic epochs 2/5、clip0.20、1个formal update。分别验证：

1. 逐步独立速度高斯噪声；
2. 全局时间相关速度噪声；
3. 前向走廊门控时间相关速度噪声。

全部三组的`episodes.jsonl`逐记录相同，collision/ordinary scenario records与cache info
相同，U1 actor和critic逐tensor相同。headline如下；每个数的重构前后比较均为严格相等：

| 模式 | policy loss | value loss | approx KL | clip fraction | EV post |
|---|---:|---:|---:|---:|---:|
| 逐步独立速度噪声 | 0.0147010052 | 0.2674632408 | 0.0279283547 | 0.2820312455 | 0.6750825454 |
| 全局时间相关速度噪声 | 0.0046016551 | 0.0860081140 | 0.0311918305 | 0.1993749924 | 0.7753824423 |
| 前向走廊门控时间相关速度噪声 | 0.0099303209 | 0.4144551620 | 0.0201755972 | 0.2484374940 | 0.8311480560 |

探索遥测也严格相等：

| 模式 | gate fraction | temporal active | same-block pairs | max block residual error |
|---|---:|---:|---:|---:|
| 逐步独立速度噪声 | 0 | 0 | 0 | 0 |
| 全局时间相关速度噪声 | 0 | 1.0 | 3134 | 3.2186508e-6 |
| 前向走廊门控时间相关速度噪声 | 0.3409375 | 0.3571875 | 1119 | 1.6689301e-6 |

这不仅证明transition数量相同；episode身份、回报、探索状态、PPO更新统计和最终权重都相同，
因此一env一worker与职责搬迁在该受控首轮上保持数值语义。

### 22.3 额外合同验证

- 所有现存Python文件可编译，`git diff --check`通过；
- `train_ppo.py --help`只含保留的CLI，默认仍是`privilege_gru/clip0.20/30 updates`；
- 四种保留critic均成功构建，固定零输入上的actor确定性动作完全相同；
- 真实`eval_multiagent.py`严格加载重构后actor并完成一个Austin双车episode；
- 已有479 cache严格命中且场景ID/顺序不变；
- production `forkserver`在一个冻结碰撞候选上完成新分类；
- 空cache可写入并严格回读，partial与identity mismatch均fail closed；
- `scripts/test_screen_reward_candidate.py`以unittest入口运行60项全部通过。

短训产物只用于重构等价验证，结论固化后已移入系统回收站。它们不是长期实验资产，
也不能被引用为新的算法或性能结论。

## 23. Regime动作收益可分性与共享actor梯度冲突（2026-08-05）

### 23.1 判决表

固定对象是production U30与前向走廊门控时间相关速度噪声U30；没有训练或更新任何actor。
生产决策仍是production U30。AUROC范围是10次`map + ego startpoint`分组切分的经验
2.5%--97.5%分位，不是独立重复实验的置信区间；gradient cosine是同一诊断目标下的原始
局部梯度方向。

| 审计 | 有效单位 | 关键结果 | 判决 |
|---|---:|---|---|
| A：361D observation vs 1680D hidden | 176个稳定counterfactual episode | 全集合减速标签AUROC中位`0.811 vs 0.695`；O+N production动作标签`0.614 vs 0.603` | hidden没有建立稳定优势；不能据此增加或宣判现有输入充分 |
| B：全部固定cohort | S/O/N=`54/69/59` | S-O/S-N output head `-0.967/-0.971`；O-N `+0.999` | 强冲突集中在output head |
| B敏感性：full-window干预确实改善 | S/O/N=`36/20/17` | S-O/S-N output head `-0.962/-0.959`；O-N `+0.998` | 主结论不由无效动作标签驱动 |
| C：GAE/credit | 0 | 未采集包含reward/value/advantage与完整privileged critic state的新rollout | 条件未触发；不得写成GAE已正确或错误 |

当前动作边界：**保持361D actor输入、当前reward与privilege-GRU critic；下一项有训练工作的
合法新轴必须隔离same-line与O/N在共享输出映射上的相反更新。**

### 23.2 要解决什么、控制与调用链

§18已经证明时间相关探索真实学会同线持续减速，却在需要果断绕行的off-line高速与near400
紧凑通过中产生新碰撞/丢超车。未决问题不是“机制是否有效”，而是：

1. 当前单帧observation或production GRU hidden是否能线性区分什么动作真正有利；
2. same-line安全偏好与off-line/near行为保持偏好在共享actor哪一层冲突；
3. 是否已有必要重开reward/critic/credit。

审计流程保持分析与训练隔离：

```text
评测trace与结果核对 -> cohort和事件窗固定 -> 从episode起点重放hidden
-> 五路闭环counterfactual -> grouped linear probe -> 冻结actor分层梯度
```

本轮一次性实现和分析产物在结果固化后按用户要求删除；已删除工具的功能重建合同保留在
`EXPERIMENTS.md` §8。这里的判决不依赖那些产物继续存在。

Production U30是所有representation与梯度的冻结actor；treatment只提供paired outcome、
动作对照和闭环counterfactual。361D observation按实际actor合同重建；1680D hidden从episode
起点逐步replay，最大anchor raw/executed action误差均为`1.91e-6`，梯度重放production动作
误差为0。没有把窗口当作新的零hidden序列。

事件窗口严格沿用§18.4已有规则：treatment新造碰撞以treatment碰撞时刻结束，消除碰撞以
production碰撞时刻结束，其余取共同可比结束时刻，再向前1.5秒；只保留finite且
`action_applied=true`的行。每条窗口有150--151个动作行。

### 23.3 Cohort、counterfactual标签与表征结果

三个cohort互斥；普通面板不含near400：

| cohort | 定义 | n | 地图分布 |
|---|---|---:|---|
| S | ordinary面板same-line production ego碰撞、treatment修复 | 54 | Austin/Hockenheim/Moscow/Nuerburgring=`7/15/21/11` |
| O | ordinary面板off-line且speed scale≥0.7，production成功、treatment新碰撞或丢超车 | 69 | `29/21/12/7` |
| N | near400中production成功、treatment新碰撞或丢超车 | 59 | Austin 59 |

预审计的O=70被数据质量门校正为69。唯一差异是Austin
`ol0_e629_o638_s0.8`：旧production评估按legacy scope在opponent撞墙时提前结束；按当前
ego scope续跑后production也发生ego-opponent碰撞，所以它不是“production成功”。三张
production跨地图各有完整600 trace，但缺`results_multi.json`与manifest；本审计用trace
重建并与历史合计`80 collision / 1142 overtake`核对一致。这是中等lineage caveat，不是
cohort阻断项。

每个场景从adverse arm开局，只在配对窗口比较五个闭环分支：anchor、成功臂full动作、
只换speed、只换steering、以及持续`-0.15m/s`。只有outcome严格从ego collision提升到follow/
overtake，或从follow提升到overtake，才算动作有利。6/182个anchor重新仿真时发生outcome
翻转，全部从A标签排除；不选择性重跑。余下176个场景的多标签为：

| 标签 | 全部 | S | O | N |
|---|---:|---:|---:|---:|
| 持续减速有利 | 61 | 33 | 14 | 14 |
| 完整production动作有利 | 37 | 0 | 20 | 17 |
| steering修正有利 | 40 | 8 | 18 | 14 |
| steering修正必要（且减速不足） | 20 | 2 | 9 | 9 |
| 固定干预库未解决/不明确 | 91 | 17 | 42 | 32 |

标签可重叠，不强迫三分类。`beneficial_speed`只有在成功臂相对anchor的窗口平均speed确实更低
时才计入“减速”；避免把O/N中的加速恢复误叫减速。

A使用事件末最后一个可执行状态，训练集统计量标准化，单层linear logit，validation early
stop与threshold，按`map + ego startpoint`做10次60/20/20 grouped split；cohort-only是一条
显式基线。主要test AUROC：

| 标签/范围 | cohort-only | 361D observation | production 1680D hidden |
|---|---:|---:|---:|
| 减速有利，全部176 | `0.708 [0.562,0.809]` | **`0.811 [0.674,0.885]`** | `0.695 [0.504,0.803]` |
| 减速有利，O+N 124 | `0.515 [0.336,0.657]` | `0.698 [0.541,0.898]` | `0.680 [0.380,0.833]` |
| production动作有利，O+N 124 | `0.407 [0.279,0.734]` | `0.614 [0.398,0.875]` | `0.603 [0.490,0.774]` |
| steering必要，O+N 124 | `0.349 [0.187,0.705]` | `0.591 [0.383,0.940]` | `0.613 [0.271,0.908]` |

Hidden在窗口内的最终预测一致率通常约0.78--0.81，高于observation约0.69--0.71，说明recurrent
表示更平滑；但它没有转化为稳定更高的held-out AUROC。N内production动作标签的hidden
中位AUROC为0.691，经验范围`[0.503,1.000]`，样本仅56/正例17，不能单独宣布成功。
weight decay `0.001/0.01/0.1`三点的上述中位AUROC相同，主判断不依赖这一旋钮。

### 23.4 分层梯度结果与机制解释

为避免“在production actor上拟合production动作得到零梯度”，B使用同一个非退化偏好目标：
在production actor上提高成功臂动作相对adverse臂动作的Gaussian log-probability margin。
Steering严格使用当前`atanh` squashed distribution（bound 0.52、latent std 0.03），speed使用
physical Gaussian std 0.15。每个episode先从起点无梯度重放hidden，在事件窗口做BPTT；
episode等权后再求cohort平均。

成功臂减adverse臂的窗口平均动作方向：

| cohort | steering | speed m/s | 解释 |
|---|---:|---:|---|
| S | `-0.00746` | `-0.20480` | same-line安全主要要求持续降速 |
| O | `+0.00827` | `+0.02648` | 保持production绕行要求相反方向 |
| N | `+0.01145` | `+0.04552` | 紧凑通过与O方向一致 |

全部固定cohort的平均梯度cosine：

| pair | GRU | output head | steering最后一行 | speed最后一行 |
|---|---:|---:|---:|---:|
| S-O | `-0.5544` | **`-0.9669`** | `-0.9912` | `-0.9878` |
| S-N | `-0.6155` | **`-0.9709`** | `-0.9933` | `-0.9925` |
| O-N | `+0.8055` | **`+0.9986`** | `+0.9989` | `+0.9965` |

只保留full-window动作替换确实提高outcome、且anchor稳定的S/O/N=`36/20/17`后：

| pair | GRU | output head | steering最后一行 | speed最后一行 |
|---|---:|---:|---:|---:|
| S-O | `-0.5045` | **`-0.9618`** | `-0.9854` | `-0.9875` |
| S-N | `-0.3197` | **`-0.9593`** | `-0.9872` | `-0.9661` |
| O-N | `+0.4644` | **`+0.9978`** | `+0.9984` | `+0.9653` |

五个`map + ego startpoint` hash fold逐折删除的validated output-head范围分别是
S-O `[-0.9664,-0.9529]`、S-N `[-0.9621,-0.9520]`、O-N `[+0.9964,+0.9986]`；符号和
定位都不依赖单一fold。

全部cohort的平均梯度层范数（原始偏好loss尺度）：

| 参数层 | S | O | N |
|---|---:|---:|---:|
| `gru.weight_ih_l0` | 24.754 | 12.810 | 19.232 |
| `gru.weight_hh_l0` | 10.866 | 5.050 | 10.019 |
| `gru.bias_ih_l0` | 2.796 | 1.361 | 1.987 |
| `gru.bias_hh_l0` | 0.535 | 0.244 | 0.483 |
| `output_layer.0.weight` | 78.110 | 55.709 | 73.555 |
| `output_layer.0.bias` | 4.200 | 2.899 | 3.913 |
| `output_layer.2.weight` | 570.205 | 789.543 | 943.264 |
| `output_layer.2.bias` | 38.472 | 39.743 | 51.909 |

GRU/head组合范数为S `27.184/576.830`、O `13.839/792.508`、N `21.782/947.558`；乘以
当前LR `3e-6/3e-5`后分别为`8.16e-5/1.73e-2`、`4.15e-5/2.38e-2`、
`6.53e-5/2.84e-2`。这些不是历史Adam step：checkpoint不保存optimizer state，且诊断目标
不是历史PPO minibatch objective。它们只能说明**当前冻结actor上的局部偏好梯度**。

机制边界：S需要的人工成功动作方向与O/N相反，而O与N彼此一致；在本节固定actor的
log-prob margin目标下，最后输出映射收到近乎反向梯度。GRU也有反向分量，但严格子集上的
幅度较弱。**这不是当前PPO优化根因的定位。** §24使用fresh rollout、reward、critic、GAE和
PPO loss后没有稳定复现负冲突，因此本节只能说明局部动作偏好结构，不能支持双head、gating
或梯度投影训练。

### 23.5 为什么C没有运行、停止与重开规则

现存eval NPZ没有reward、value、GAE、log-prob或完整20D privileged critic state；历史
checkpoint也没有rollout buffer/optimizer/environment recurrent state，所以不能恢复历史
advantage。本轮原定门控是：只有A显示regime可分但B不能解释干扰，或准备因符号错配重开
critic/reward，才付出新rollout成本运行C。

B在人工偏好目标下给出经counterfactual筛选稳定的output-head反向方向，A没有建立hidden
充分性的强结论，因此当时C没有触发。§24随后否决把B外推成真实PPO稳定冲突。正确表述仍是
“当前没有依据先改critic”，不是“GAE已验证正确”，也不是“output head根因已定位”。

停止规则：

1. 不因A的null/不稳定结果立即增加actor输入；线性不可解码不等于信息不存在；
2. 不把B写成历史PPO/Adam更新的精确回放；
3. 不重复exploration强度、reward、critic或generic observability sweep，也不再把head冲突写成已定位根因；
4. 不用6条anchor不稳定场景训练probe或选择性重跑到预期outcome；
5. 只有新的fresh PPO证据先稳定识别可干预机制，才设计对应训练控制；
6. 新鲜同前缀paired rollout若直接显示“counterfactual更好但advantage为负”，才重开C和critic/credit；
7. 本轮一次性脚本和分析产物已按用户要求删除；不得仅为复核本文数值重新生成analysis树，
   只有输入模型、面板、动作合同改变，或用户明确要求重新审计时才按`EXPERIMENTS.md` §8重建。

### 23.6 可独立于分析产物保留的核心记录

- 冻结模型：production U30与前向走廊门控时间相关速度噪声U30，身份见`HANDOFF.md` §2；
- 固定cohort：S/O/N=`54/69/59`，near400不与ordinary面板混合；
- counterfactual有效176/182；减速/production/steering-required标签=`61/37/20`，未解决91；
- 表征结论：hidden更平滑但没有稳定优于361D observation，不能据此改输入；
- 梯度结论：人工偏好目标下严格改善子集S-O/S-N head约`-0.96`、O-N约`+1.00`；§24表明
  该方向不能代表fresh PPO的稳定梯度；
- C未运行；当前保持production、actor输入、reward与critic，不再以输出映射隔离为默认下一轴。

## 24. Fresh first-step PPO regime梯度诊断（2026-08-06，已完成）

### 24.1 问题与证据边界

目标不是重新证明§23的人工偏好梯度，而是直接测量fresh PPO rollout在optimizer执行前的
advantage加权梯度，判断same-line与off-line-fast是否在中后期checkpoint的共享output head
上形成稳定冲突。全程不执行optimizer step，不修改actor/critic。

历史checkpoint没有scheduler cursor、rollout buffer、optimizer或environment recurrent state，
因此这不是历史U1/U10/U20/U30更新的精确回放。每个cell从相同静态collision/ordinary场景池、
seed 42和初始scheduler位置重新采102400个transition；能控制场景池与随机种子，不能恢复历史
队列位置。U1已经是两条训练轨迹各完成一次不同更新后的模型，不承担“两臂差异应为零”的
sanity角色；harness sanity由同checkpoint、同探索、同seed的重复运行逐array完全一致来验证。

### 24.2 Cell、分层和PPO语义

固定K=`1/10/20/30`，每个K运行三格：

1. production actor + production critic + baseline独立速度噪声；
2. 走廊时间相关探索actor + 该臂critic + baseline独立速度噪声；
3. 同一个走廊actor/critic + 前向走廊门控时间相关速度噪声。

第2与第3格干净隔离**当前探索模式**；第1与第2格同时包含训练轨迹形成的actor和critic差异，
只能解释为“已学习checkpoint状态差异”，不能单独归因于actor。三格不是完整2x2 factorial，
不识别actor×exploration interaction；除非本轮出现无法解释的交互迹象，否则不为完整性增加
production actor + corridor noise第四格。

transition按场景元数据分成`same_line`、`offline_fast`（异线且speed scale≥0.7）和
`offline_slow`。每个minibatch的分组loss都用`group sum / N_total`，保证三组梯度精确重构原
PPO minibatch梯度；禁止对各组分别取mean造成隐式regime重加权。每组同时报告有效transition
数；same-line或offline-fast少于256的minibatch不进入主cosine摘要。

冻结参数使old/new log-prob ratio约等于1，clip fraction应为0；这正是每个formal update第一
个optimizer step之前的`-A·grad log pi`，不是2 epoch乘8 minibatch后续15个串行step，也不是
Adam-preconditioned update。三组虚拟对称投影只报告未裁剪、未经过Adam的候选梯度方向、范数
变化以及对same/fast梯度的一阶内积，不称为真实训练步。

### 24.3 保存字段和失败门

每个cell保存reward、value、return、GAE、old/new log-prob、20D critic privileged state、
动作、episode start、regime、scenario id、episode id/outcome，以及danger gate、temporal active、
block id、standard residual和per-transition speed log-std。这样一次rollout同时满足未来credit
审计所需字段，不需重跑仿真。

正式cell必须同时通过：collection-equivalent log-ratio误差`<=5e-5`；分组梯度重构最大误差
`<=2e-5`；102400个有效transition全覆盖；场景队列文件在两条训练轨迹间逐字节相同；同配置
重复烟测的全部transition arrays和聚合梯度完全一致。任一失败即停止，不进入机制判读。

### 24.4 判读分支

固定seed 42，不用多seed显著性措辞。主单位是K10/K20/K30的control-relative effect size，
U1只作早期描述：

- **走廊checkpoint特异冲突支持**：走廊actor在baseline探索下，至少2/3个中后期K的聚合
  head same-vs-fast cosine `<=-0.30`，相对同K production至少再负`0.20`，且合格minibatch中
  至少75%为负；再检查走廊探索是否进一步改变冲突。
- **共同冲突而非走廊特异**：两臂均稳定`<=-0.30`但control-relative差不足`0.20`。这只支持
  共享PPO优化问题，不能用来解释走廊臂特有副作用。
- **不支持**：中后期聚合冲突不稳定或接近0；不启动梯度投影训练。
- **不可判读**：有效分层样本不足、ratio/reconstruction失败，或checkpoint/队列身份不一致；
  先修测量，不能把null写成机制否决。

即使第一分支成立，也不会自动启动30-update投影臂。虚拟投影若对same或offline-fast任一梯度的
一阶内积为负，或方向主要由放大的正交残差支配，则先停止；真正训练还必须单独按Austin600、
near400和跨地图checkpoint区间预注册，不能复用历史单点`20/338`或`46/1122`作为门槛。

### 24.5 正式结果

全部12格各覆盖102400个有效transition；每格8个minibatch均满足same-line/offline-fast至少
256条的样本门。collection-equivalent old/new log-prob误差全为0，三组梯度重构最大误差范围
`7.75e-7 -- 3.81e-6`，低于`2e-5`门。Production K1的production-batched replay最大ratio
偏移为`0.0171`，其余格不超过`0.0075`，但全部clip fraction为0；因此应称为“首步近似
ratio=1”，不能称为数学恒等。所有cell都没有调用optimizer或修改checkpoint。

表中格式为`aggregate head cosine / 合格minibatch中位数 / 负号个数(共8)`：

| K | Production checkpoint + baseline | 走廊checkpoint + baseline | 走廊checkpoint + 实际走廊探索 |
|---:|---:|---:|---:|
| 1（描述） | `+0.957 / +0.311 / 3` | `-0.895 / +0.232 / 4` | `+0.786 / +0.085 / 3` |
| 10 | `-0.008 / -0.543 / 7` | `-0.722 / -0.652 / 7` | `+0.663 / +0.075 / 4` |
| 20 | `+0.708 / +0.490 / 3` | `+0.841 / +0.043 / 4` | `+0.622 / +0.491 / 2` |
| 30 | `+0.814 / -0.314 / 5` | `-0.819 / +0.223 / 2` | `-0.186 / +0.313 / 4` |

对应aggregate GRU cosine：Production为`+0.057/+0.046/+0.034/+0.068`；走廊checkpoint切回
baseline为`-0.121/+0.107/+0.040/+0.010`；实际走廊探索为
`+0.172/+0.162/-0.027/-0.088`。GRU同样没有跨K稳定负冲突。

反事实baseline探索下，走廊checkpoint在K10和K30的aggregate确实比Production更负，达到
预注册的aggregate效应量门；但只有K10达到`>=6/8`负minibatch，K30只有`2/8`且中位数为正，
K20 aggregate又为正。因此完整联合标准未通过。更直接地，实际训练使用的走廊探索格在
K10/K20都明显为正，K30也只有轻微aggregate负值且正负minibatch各半。

虚拟对称投影也没有持续满足保护门。走廊checkpoint+baseline在K1和K30加入offline-slow后，
投影合成向量对offline-fast梯度的一阶内积分别为`-2.865`与`-1.784`；K30投影/原合成范数比
达到`1.96`。所以它不能被描述为“消除冲突且不损害两侧”，更不是缺少Adam state时可恢复的
历史optimizer step。

### 24.6 判决、机制修正和停止规则

**判决：不支持稳定走廊特异PPO head冲突，不启动梯度投影训练。** 这不是说§23的数据错误；
§23测到的是固定失败cohort上“提高人为选定成功动作相对失败动作概率”的局部偏好梯度，且
接近rank-1的最后输出层会放大动作方向相反所产生的cosine。这里直接使用fresh rollout、实际
reward/value/GAE和production PPO loss，显示其符号与强度随checkpoint、minibatch合成和探索
模式显著变化，不能从`-0.96`外推到实际训练中的稳定cancellation。

机制结果与验收结果分开：诊断工具证明了三类advantage梯度可按`sum/N_total`精确重构，且
时间相关探索能够显著改变同一checkpoint看到的fresh梯度几何；但该变化没有形成跨K一致、
可由一个固定投影规则修复的模式。尤其K10同一走廊checkpoint从baseline探索`-0.722`变为
实际走廊探索`+0.663`，说明“模型参数冲突”和“采样分布/优势改变”不能混为一因。

证据边界：单seed 42；fresh初始scheduler位置；每格使用本臂critic，因此Production与走廊
checkpoint之差同时包含actor与critic训练轨迹；三格不是完整2x2，不能识别interaction；只测
每次update第一个optimizer step之前的几何，不覆盖后续15个串行minibatch/epoch step或Adam。

停止规则：

1. 不依据§23的人工偏好cosine直接实现PCGrad/对称投影训练；
2. 不用单个K的aggregate负值翻案，特别是K30的`-0.819`同时只有`2/8`minibatch为负；
3. 不再补完整2x2只为解释本次null；它不会改变实际走廊探索格未稳定冲突的主判决；
4. 只有新设计先在fresh PPO数据上跨checkpoint稳定复现冲突，并使虚拟投影对same-line和
   offline-fast的一阶内积均非负，才重开训练；
5. 当前保持production actor、reward、critic和baseline探索，不产生新actor。

### 24.7 跨地图评测基础复核与device边界

在决定不启动投影训练后，使用冻结production U30对Hockenheim、MoscowRaceway、
Nuerburgring三张固定600场景panel做fresh ego-scope评测。CUDA与历史权威口径逐图完全一致：

| map | CUDA碰撞 / 超车 | CPU碰撞 / 超车 |
|---|---:|---:|
| Hockenheim | `26 / 356` | `26 / 355` |
| MoscowRaceway | `32 / 385` | `31 / 386` |
| Nuerburgring | `22 / 401` | `21 / 401` |
| 合计 | **`80 / 1142`** | **`78 / 1142`** |

CUDA碰撞按same-line/off-line分解为`66/14`，CPU为`64/14`。五个CPU/CUDA outcome差异中，
Hockenheim有三条`follow/overtake`边界翻转；MoscowRaceway的
`ol1_e1577_o1592_s0.5`由CPU `overtake`变为CUDA `ego-opp`；Nuerburgring的
`ol1_e579_o594_s0.8`由CPU `follow`变为CUDA `ego-wall`。所以历史`80/1142、66/14`
不是重建误差，而是CUDA推理协议下可复现的结果；device必须写入manifest，并在对照与处理臂
之间固定，不能将CPU与CUDA结果拼接成一个规范包。一次性CPU评测包在确认边界后已清除；后续
正式评测只运行CUDA，不再重复CPU对照。

这次fresh评测同时发现旧评估器的trace保存语义有破坏性：先写model-derived canonical路径、
再移动到指定output会移走已有trace。现已改为把每条目标trace路径直接传入`evaluate_segment`。
73场景回归得到73个result和73个trace、key集合完全相等，且既有Austin `600/400/73`
三个trace根计数前后不变。第一次触发旧逻辑后，原跨地图canonical GPU traces无法恢复；一次性
fresh CPU/CUDA包在确认device边界并把数字写入本文后清理，不作为长期证据。后续需要规范trace包
时应直接用CUDA重新评估并保存，不得借用CPU trace或把本文数字反写成逐episode包。

该复核只修复测量基础，不产生新模型，也不改变§24.6“不启动梯度投影训练”的结论。

## 25. Collision/ordinary role配比假说审计（2026-08-06，无训练）

### 25.1 判决表

| 证据 | 观察 | 能支持什么 | 不能支持什么 |
|---|---:|---|---|
| fresh U30 episode return | collision role `+0.3220`；ordinary role `-0.2022`；线性break-even `0.386` | U30这条fresh rollout存在role trade-off | 不能识别稳定的50/50目标错配 |
| fresh U10/U20 | `+0.7041/+0.4717`；`+0.5664/+0.2967` | 两个role可同时改善 | 不支持刚性role补偿 |
| 历史正式rollout | U27--U30/U42--U45的break-even跨`0.019--0.878`，U42/U44两role同时改善 | trade-off随checkpoint变化 | 不存在已验证固定临界点 |
| 实现合同 | role贯穿scheduler、seed、minibatch、warmup和telemetry | 25%不是一个孤立配置值 | 不能称为干净单变量臂 |
| 最终决策 | **不训练25% collision-role臂** | 保持production 50/50合同 | 该轴不是“已测试失败”，只是当前假说不足以准入 |

### 25.2 假说和复算口径

假说是：前向走廊门控时间相关速度探索在固定collision cache中收益大，在ordinary自然网格中
代价大；训练按env rank奇偶固定50/50 role，可能把验收分布不接受的交易算成正收益。提出者
用episode return写出线性混合并估计break-even约`0.412`，据此建议把collision role降到25%。

本轮没有新采样。复算对象是§24保存的production与走廊checkpoint在K=`1/10/20/30`的fresh
102400-transition rollout，以及两条训练轨迹已有的formal-update metrics。role按逻辑env奇偶
划分；只纳入已有terminal outcome的完整episode，return为该episode逐步reward和。该统计单位
是episode，不是PPO transition，也不是同scenario配对差。

### 25.3 Fresh rollout结果与配对边界

| K | production→走廊：collision role | production→走廊：ordinary role | 线性break-even |
|---:|---:|---:|---:|
| 1 | `-0.0975` | `+0.3531` | 无单一trade-off |
| 10 | `+0.7041` | `+0.4717` | 两role同时改善 |
| 20 | `+0.5664` | `+0.2967` | 两role同时改善 |
| 30 | `+0.3220` | `-0.2022` | `0.3857` |

U30完整episode数为production `78 collision + 64 ordinary = 142`、走廊
`70 + 64 = 134`。两臂虽然使用相同479 collision候选和600 ordinary候选，但episode时长改变
会改变各env在6400步内推进到的队列位置；按env内完成顺序只有132个共同位置，其中仅34个
scenario identity相同。因此上述均值不是严格配对，不能报告McNemar或把差值当作固定场景
因果效应。

走廊门控本身的选择性成立。U30 same-line transition激活率为collision role `62.82%`、
ordinary role `61.80%`；off-line为`0.089%/0%`。全部gated transition中collision/ordinary
约占`39.5%/60.5%`。这证明门控落在同线regime，不证明之后学到的共享参数更新只局限于同线。

### 25.4 历史正式rollout不支持稳定break-even

同update比较两条训练轨迹已有role mean episode return，得到：

| update | collision Δ | ordinary Δ | 描述性break-even |
|---:|---:|---:|---:|
| 27 | `+0.4450` | `-0.1215` | `0.214` |
| 28 | `+0.2532` | `-0.0048` | `0.019` |
| 29 | `+0.4588` | `-0.2506` | `0.353` |
| 30 | `-0.0094` | `+0.0676` | `0.878`（收益符号与假说相反） |
| 42 | `+0.3188` | `+0.0175` | 两role同时改善 |
| 43 | `+0.2838` | `-0.0413` | `0.127` |
| 44 | `+0.0103` | `+0.0161` | 两role同时改善 |
| 45 | `+0.0500` | `-0.0671` | `0.573` |

这些不是独立重复样本，但足以否定“已知且稳定的0.412临界点”。只看fresh U30会把一个强烈
依赖checkpoint和实际采样队列的描述性切片误写成训练目标定律。用户固定seed 42；不增加多seed
来挽救这个假说，已有checkpoint轴已经暴露主要不稳定性。

### 25.5 为什么25%不是干净单变量

当前50/50合同至少同时存在于：ScenarioScheduler按rank奇偶选择collision/ordinary；logical
seed含`rank % 2`；rollout buffer按env-major rank构建`collision_by_transition`；critic warmup按
role分别切train/validation；role-specific EV/loss与记录逻辑；训练入口要求batch size可被
`2*n_steps`整除，使每个recurrent minibatch含相等的完整env序列。当前`batch_size=12800`、
`n_steps=6400`时，一个minibatch只有两个完整env序列，无法在不改变minibatch组成或role平衡的
情况下表达25%。所以这不是“改三行比例”，而是同时改变数据分布、RNG映射和优化合同。

更根本地，episode-return混合
`p*Δcollision + (1-p)*Δordinary`不是PPO优化目标。PPO按transition advantage更新，策略变化后
episode长度、完成数、状态访问、GAE和队列推进都会变化，不能用静态线性式外推新训练结果。

### 25.6 结论、当前验收口径和重开规则

**结论：role配比错配是可解释既有specialization的假说之一，但没有被识别为根因；不运行
25% collision-role训练。** 本审计没有否决所有role-sampling研究，只否决以单个fresh U30
episode-return切片作为准入依据。

用户最新把**Austin、Hockenheim、MoscowRaceway和Nuerburgring各600 episode**设为默认
正式验收集合。Austin600与三张跨地图合计crossmap1800都具有验收权，三张跨地图还必须逐图
报告。最低验收线是canonical BC：Austin、Hockenheim、MoscowRaceway和Nuerburgring的
当前ego-scope `collision/overtake`分别为`33/339、27/343、43/373、26/390`，候选在每张地图上都必须满足
ego collision不高于BC且overtake不低于BC，opp-wall单列；Production U30继续作为当前部署模型和机制/reference
对照，但不是最低验收线。near400和hard73只保留机制/特化诊断意义。三张跨地图已经参与走廊探索设计，因此是
当前开发验收集，不得称为从未参与设计的最终泛化留出。历史章节中的旧多面板阈值仍只用于
解释当时为什么作出历史决策；新实验只需为高于BC最低线的额外目标和checkpoint band重新预注册。

该口径改变后应先重判已有模型，不应直接启动reference-KL新训练。前向走廊门控时间相关速度
噪声U44逐图为`18/344、16/347、15/390、13/397`，U45逐图为
`17/345、20/344、11/391、13/396`，顺序均为Austin、Hockenheim、MoscowRaceway、
Nuerburgring的`ego collision/overtake`。U44与U45都严格通过四张地图各自的BC计数下限。
四图合计U44为`62/1478`、U45为`61/1476`，当前ego-scope BC为`129/1445`。旧Austin trace的
`ol0_e629_o638_s0.8`在4.09s因opponent-wall被legacy口径截断；2026-08-06的单场景CUDA
ego-scope补跑显示episode继续后在4.83s发生ego-opponent collision，所以Austin仍为`33/339`。
Hockenheim的4个与Nuerburgring的2个opponent-wall event均已在8.01s终点完整记录且均为
overtake；它们已包含在`343/390`中，不得再重复相加。2026-08-06已补全BC四图
规范`results_multi.json`/manifest并完成U44逐episode配对；结果见§26。U44已通过当前
四图BC验收线，因此不启动训练期reference-policy regularization。ordinary异线高速重加权
U27--U30也已按新口径补做既有结果重判；四点均逐图通过BC线，但历史manifest缺CUDA device，
正式候选边界见§27。

重开必须同时满足：seed 42下多个预先固定checkpoint使用严格scenario-identity匹配或等价固定
队列，role收益方向稳定；分析对象是transition advantage而非只看episode return；实现能改变
role采样而不改变recurrent minibatch/seed语义，或明确把这些混淆预注册为多变量实验。否则保持
50/50 production合同，不继续扫role比例。

## 26. 前向走廊时间相关速度探索U44四图BC验收（2026-08-06）

本节数值单位均为`ego collision / overtake`。

| 模型 | 训练身份 | 四图合计 | 口径 | 判决 |
|---|---|---:|---|---|
| Canonical BC | 原始End2Race actor | `129 / 1445` | 正式ego-scope对照 | 每图最低验收线 |
| **Production U30** | BC fresh-start，逐步独立速度噪声，30 updates | `94 / 1508` | 历史headline；跨图缺完整配对包 | 当前production，不是本次最低对照 |
| 前向走廊时间相关速度探索 U44 | BC fresh-start，corridor-temporal，该轨迹的update 44 | `62 / 1478` | 完整四图配对 | **接受为四图BC验收候选** |
| 同轨迹 U45 | 同一fresh 45-update run的终点 | `61 / 1476` | 完整四图计数 | 只作相邻checkpoint稳定性支持 |

U44是历史旧称`CT-v2`的轨迹checkpoint，**不是Production U30续训到update 44**。
两条轨迹都从`pretrained/end2race.pth`fresh-start；前者run config的`num_updates=45`、
`speed_exploration_mode=corridor_temporal`，后者用baseline逐步独立噪声。U44相对Production U30
同时差了exploration方式与update数，所以不能把`94/1508→62/1478`写成单变量因果效应；
本次判决只问U44是否通过用户定义的BC绝对下限。

### 26.1 为什么只补跑一个BC episode

BC四图各保留600条numeric trace，U44四图各保留600条trace和逐episode result，两侧
scenario key逐图完全相等。绝大多数BC trace可直接按当前`collision_scope=ego`重建终局。
唯一例外是Austin `ol0_e629_o638_s0.8`：旧trace在4.09s因opponent-wall被legacy口径
提前截断，截断前终局不能代表ego-scope下的8s结果。因此只对该场景使用canonical BC、
CUDA、零噪声和ego scope fresh补跑：opponent撞墙后episode继续，ego在4.83s与opponent
碰撞。Austin总结果仍是`33 collisions / 339 overtakes`，但该场景的子类从`opp-wall`
改为`ego-opp`。

Hockenheim的4个和Nuerburgring的2个opponent-wall event均在8.01s终点出现，旧trace已保留
完整终局，且六个均已按最终relative progress计为overtake。因此正式BC四图为：

| 地图 | collision | overtake | ego-opp | ego-wall | opponent-wall event |
|---|---:|---:|---:|---:|---:|
| Austin | 33 | 339 | 28 | 5 | 1 |
| Hockenheim | 27 | 343 | 27 | 0 | 4 |
| MoscowRaceway | 43 | 373 | 43 | 0 | 0 |
| Nuerburgring | 26 | 390 | 25 | 1 | 2 |
| 四图合计 | 129 | 1445 | 123 | 6 | 7 |

重建前先用Austin的599个未变episode对旧评估器直接result交叉检查：outcome全部一致，
速度/距离/动作等指标最大浮点差小于`1e-6`；快速进度投影与production
`ProgressProjector`sampling cross-check的最大差为0。只有上述单episode按fresh ego-scope结果
改变。

### 26.2 配对结果

| 地图 | BC→U44 collision | removed / created | exact p | BC→U44 overtake | lost / gained | exact p |
|---|---:|---:|---:|---:|---:|---:|
| Austin | `33→18` | `24 / 9` | `0.01353099` | `339→344` | `11 / 16` | `0.44206834` |
| Hockenheim | `27→16` | `18 / 7` | `0.04328525` | `343→347` | `14 / 18` | `0.59661490` |
| MoscowRaceway | `43→15` | `37 / 9` | `4.0560e-5` | `373→390` | `12 / 29` | `0.01150779` |
| Nuerburgring | `26→13` | `15 / 2` | `0.00234985` | `390→397` | `4 / 11` | `0.11846924` |
| 四图合计 | `129→62` | `94 / 27` | `7.1351e-10` | `1445→1478` | `41 / 74` | `0.00268570` |

碰撞子型BC→U44逐图为：ego-opp `28→13、27→15、43→15、25→13`；ego-wall
`5→5、0→1、0→0、1→0`。opponent-wall event保持`1/4/0/2`不变。四图池化结果同时
显著降低ego collision和增加overtake，所以U44对BC不是“用放弃超车换安全”。

U44新造collision identity（安全副作用守门集）：

- Austin：`ol0_e1090_o1088_s0.8`、`ol0_e1425_o1428_s0.7`、`ol0_e1509_o1500_s0.7`、
  `ol0_e461_o472_s0.5`、`ol1_e2012_o2027_s0.7`、`ol1_e377_o392_s0.5`、
  `ol1_e629_o644_s0.6`、`ol2_e2054_o2088_s0.8`、`ol2_e461_o480_s0.6`。
- Hockenheim：`ol0_e1150_o1152_s0.7`、`ol0_e1545_o1545_s0.6`、`ol1_e395_o410_s0.7`、
  `ol2_e1222_o1239_s0.8`、`ol2_e395_o408_s0.8`、`ol2_e431_o444_s0.8`、
  `ol2_e719_o743_s0.8`。
- MoscowRaceway：`ol0_e1223_o1227_s0.8`、`ol0_e1319_o1330_s0.7`、`ol0_e1577_o1574_s0.8`、
  `ol0_e193_o208_s0.7`、`ol0_e97_o110_s0.8`、`ol1_e1480_o1495_s0.5`、
  `ol1_e1577_o1592_s0.6`、`ol1_e483_o498_s0.5`、`ol2_e740_o760_s0.8`。
- Nuerburgring：`ol0_e1961_o1956_s0.8`、`ol0_e2094_o2089_s0.5`。

U44丢失overtake identity（性能副作用守门集）：

- Austin：`ol0_e1090_o1088_s0.8`、`ol0_e1425_o1428_s0.7`、`ol0_e1509_o1500_s0.7`、
  `ol0_e377_o383_s0.7`、`ol0_e461_o472_s0.5`、`ol0_e755_o759_s0.7`、
  `ol2_e1174_o1210_s0.8`、`ol2_e1509_o1547_s0.8`、`ol2_e2054_o2088_s0.8`、
  `ol2_e461_o480_s0.6`、`ol2_e838_o865_s0.8`。
- Hockenheim：`ol0_e1150_o1152_s0.7`、`ol0_e1150_o1152_s0.8`、`ol0_e1545_o1545_s0.6`、
  `ol1_e431_o446_s0.6`、`ol1_e755_o770_s0.5`、`ol2_e1006_o1028_s0.8`、
  `ol2_e1042_o1067_s0.8`、`ol2_e1222_o1239_s0.8`、`ol2_e1545_o1572_s0.8`、
  `ol2_e1689_o1719_s0.8`、`ol2_e1725_o1755_s0.8`、`ol2_e395_o408_s0.8`、
  `ol2_e431_o444_s0.8`、`ol2_e719_o743_s0.8`。
- MoscowRaceway：`ol0_e1223_o1227_s0.8`、`ol0_e1255_o1259_s0.8`、`ol0_e1319_o1330_s0.7`、
  `ol0_e1577_o1574_s0.8`、`ol0_e193_o208_s0.7`、`ol0_e97_o110_s0.8`、
  `ol1_e1158_o1173_s0.5`、`ol1_e1480_o1495_s0.5`、`ol1_e1577_o1592_s0.6`、
  `ol1_e386_o401_s0.7`、`ol2_e322_o339_s0.8`、`ol2_e740_o760_s0.8`。
- Nuerburgring：`ol0_e1025_o1031_s0.8`、`ol0_e1961_o1956_s0.8`、
  `ol0_e2094_o2089_s0.5`、`ol1_e446_o461_s0.6`。

### 26.3 完整性与checkpoint判决

BC和U44共8个四图包全部通过：每包600 unique episode、0 errors、result/trace key精确相等、
全数值有限、ego-opp/ego-wall marker与outcome一致、碰撞子型互斥、terminal row恰好一行且
`action_applied=false`只出现在该行。

四图后期band：

| update | Austin | Hockenheim | MoscowRaceway | Nuerburgring | 四图合计 |
|---:|---:|---:|---:|---:|---:|
| 42 | `20/332` | `17/336` | `15/385` | `14/396` | `66/1449` |
| 43 | `20/335` | `18/345` | `10/391` | `15/394` | `63/1465` |
| 44 | `18/344` | `16/347` | `15/390` | `13/397` | `62/1478` |
| 45 | `17/345` | `20/344` | `11/391` | `13/396` | `61/1476` |

U42/U43的Austin overtake低于BC `339`，U44/U45则是相邻两个逐图通过checkpoint。
U44是在配对前已固定的验收候选，所以正式接受U44，把U45作为后期稳定性支持，
不事后按总collision的`62 vs 61`重选单点。

**判决：U44通过用户当前“四图都不差于canonical BC”的正式验收。** 它不是相对
Production U30的Pareto改进：历史headline合计下，U30约为`94 collisions / 1508 overtakes`，
U44为`62 / 1478`，即更安全但少30次超车。当前用户明确的最低对照是BC，所以U44可接受；
若未来把目标提高为“不差于Production U30”，必须先fresh重建U30四图完整trace再配对，
不得用上述headline直接声称显著性。当前不启动reference-KL、hold/gap/std继续扫描或其他
新训练。

旧near400诊断面板上的确定性结果是Production U30 `28 collisions / 325 overtakes`、U44
`37 / 288`。这不是当前正式验收panel，不能否决已经完成的四图BC判决；但它仍是已观测到的
副作用边界：U44在U30原本贴身但成功的困难场景上多9次碰撞、少37次超车。因此“四图通过”
只能解释为达到用户当前BC下限，不能扩大为“所有自然/近失分布均改善”或“全面优于U30”。
若未来目标改为保留U30性能，near400应恢复为机制守门，与fresh四图U30配对结果一起使用。

### 26.4 为什么当前结束exploration，以及什么情况才重开

前向走廊时间相关速度探索本身已经完成了它的机制目标：训练期只在对手位于前方
同走廊、表面纵向gap在`(0, 2m)`、对手相对ego raceline横向偏移不超过0.25m且OBB有正横向
重叠时，把同一速度高斯残差保持50步（0.5s）。噪声边际std仍为0.15，门外仍逐步独立采样；
确定性eval时不存在该噪声或运行时shield，最终actor仍是原End2Race 361D输入和12-key
checkpoint。因此这条线是纯训练期exploration改动，不是reward、imitation或部署后处理。

GPT Pro后续提出的U30 reference-policy regularization用冻结Production U30在异线高速状态上约束
student。该提案的目标是将U44的same-line安全与U30的off-line超车合并到一个actor，但它：

1. 没有进入正式训练，所以不能写成已证伪；
2. 依赖预先训练的U30 teacher和第二actor objective，属于“单阶段PPO + 训练期策略保持”，
   不是纯PPO exploration；
3. 原本需要failure coverage、same-prefix teacher validity和fixed-beta virtual update三项零训练
   准入检查，但当前BC验收目标已由U44满足，这些检查与正式训练现在都是多余成本。

因此本方向的状态是**当前目标下关闭，未被理论或实验永久否决**。只在以下条件同时满足时重开：

- 任务目标明确提高为“不仅通过BC线，还要保留Production U30超车表现并保留U44安全收益”；
- 用户明确允许冻结teacher和训练期辅助actor loss；
- 先用同observation prefix证明U30在U44受损状态上是可靠teacher，mask覆盖真实损害而几乎不覆盖
  same-line目标，一个预先固定的beta虚拟更新同时保留same-line PPO方向并降off-line reference KL。

当前不需要任何新训练。§27重判后产品候选不再只有U30/U44：若安全优先选择U44；若Austin
或near400优先保留Production U30；若四图超车优先，ordinary异线高速重加权U30在计数上是
中间安全、高超车候选，但正式纳入前还需一次固定CUDA四图确认。在用户明确选择前，不改默认
模型路径、训练参数或production登记。

## 27. ordinary异线高速重加权U30按四图BC口径重判（2026-08-06）

本节数值单位均为`ego collision / overtake`。该实验保持2m前向走廊门、速度std `0.15`、
50步时间保持和其余PPO配置，只在ordinary角色内把异线高速场景份额从自然的三分之一提高到
`0.60`，同线份额保持不变。它在旧Austin+near400联合协议下已被否决；本节不是新训练或
事后重选，而是用户将正式验收改为四张地图逐图BC下限后，对现有U27--U30的完整重判。

### 27.1 判决表

| 模型 / checkpoint | Austin | Hockenheim | MoscowRaceway | Nuerburgring | 四图合计 | near400 | 当前判决 |
|---|---:|---:|---:|---:|---:|---:|---|
| Canonical BC | `33/339` | `27/343` | `43/373` | `26/390` | `129/1445` | — | 逐图最低线 |
| 重加权 U27 | `19/366` | `14/366` | `20/386` | `21/394` | `74/1512` | `63/306` | 逐图通过BC，旧near副作用 |
| 重加权 U28 | `21/363` | `13/371` | `20/386` | `21/394` | `75/1514` | `65/299` | 逐图通过BC，旧near副作用 |
| 重加权 U29 | `21/370` | `15/374` | `19/387` | `22/394` | `77/1525` | `77/291` | 逐图通过BC，旧near副作用 |
| **重加权 U30** | **`16/368`** | **`17/368`** | **`17/389`** | **`23/391`** | **`73/1516`** | **`64/302`** | **计数通过；CUDA provenance待确认** |
| 前向走廊时间相关速度探索 U44 | `18/344` | `16/347` | `15/390` | `13/397` | `62/1478` | `37/288` | 正式CUDA安全候选 |
| Production U30 | `14/366` | `26/356` | `32/385` | `22/401` | `94/1508` | `28/325` | 当前production；跨图headline |

U27--U30四个相邻checkpoint全部逐图满足BC线，说明重加权U30不是从单点尖峰里挑出的偶然
通过者。它在四图上处于U44与Production U30之间：相对U44多11次collision、同时多38次
overtake；相对Production headline少21次collision、多8次overtake。但Production跨地图缺
当前规范逐episode包，所以后一比较只能报告headline，不能写成配对Pareto结论。

### 27.2 相对BC的严格配对结果

四张地图中，重加权U30与BC的600个episode key逐图精确相等：

| 地图 | BC→重加权U30 collision | removed / created | exact p | BC→重加权U30 overtake | lost / gained | exact p |
|---|---:|---:|---:|---:|---:|---:|
| Austin | `33→16` | `23 / 6` | `0.00231570` | `339→368` | `2 / 31` | `1.3085e-7` |
| Hockenheim | `27→17` | `17 / 7` | `0.06391466` | `343→368` | `8 / 33` | `0.00011222` |
| MoscowRaceway | `43→17` | `34 / 8` | `6.8771e-5` | `373→389` | `10 / 26` | `0.01133098` |
| Nuerburgring | `26→23` | `10 / 7` | `0.62905884` | `390→391` | `6 / 7` | `1.0` |
| 四图合计 | `129→73` | `84 / 28` | `1.1107e-7` | `1445→1516` | `26 / 97` | `7.9677e-11` |

池化结果明确通过当前四图BC最低线；逐图上Nuerburgring安全和性能余量最薄，collision
`23≤26`、overtake `391≥390`，两项配对检验均不显著。因此正式确认时不得只重跑表现余量大的
三图，也不得把池化显著性写成每张地图都显著。

### 27.3 与U44的产品取舍

U44与重加权U30的场景身份也逐图精确相等。以U44为对照、重加权U30为处理：

| 地图 | U44→重加权U30 collision removed / created | exact p | U44→重加权U30 overtake lost / gained | exact p |
|---|---:|---:|---:|---:|
| Austin | `12 / 10` | `0.83181190` | `8 / 32` | `0.00018217` |
| Hockenheim | `11 / 12` | `1.0` | `11 / 32` | `0.00191396` |
| MoscowRaceway | `11 / 13` | `0.83881974` | `12 / 11` | `1.0` |
| Nuerburgring | `3 / 13` | `0.02127075` | `10 / 4` | `0.17956543` |
| 四图合计 | `37 / 48` | `0.27799929` | `41 / 79` | `0.00066672` |

因此重加权U30不是U44的严格Pareto改进：四图collision点估计`62→73`变差且配对不显著，
overtake `1478→1516`显著增加。Nuerburgring承担了最明显的新增collision身份。产品选择必须
显式决定安全与超车权重，不能只因它相对BC双轴改善就称其“最佳”。

### 27.4 机制结果与旧协议边界

不要把本节读成“异线高速重加权已经消除前向走廊探索的副作用”。旧near400四点为
`63--77 collisions / 291--306 overtakes`，U30为`64/302`，明显差于Production U30
`28/325`和U44 `37/288`。这正是它在历史协议下被否决的原因。用户已把near400降为机制/特化
诊断且取消独立否决权，所以旧判决不能继续阻止四图BC资格；但副作用事实仍成立，若产品目标
重新包含贴身成功能力，near400必须恢复为守门指标。

当前只能确认：对ordinary异线高速采样重加权后，四图自然协议上的总超车显著提高，同时保留
相对BC的安全优势。它是否通过修复特定off-line机制实现，还是改变了更广泛的训练分布，现有
结果没有识别；§24已否决稳定output-head PPO梯度冲突，§25也否决固定role配比根因，不能用
这两种已被削弱的机制解释本臂。

### 27.5 数据质量与正式证据边界

现存重加权U27--U30四图包均由完整numeric traces重建：每包600 unique episode、0 errors、
result/trace key一致、所需数组对齐且有限、ego collision marker互斥，terminal row合同通过。
因此本节的计数和同场景配对可复算，数据本体足以支持“历史轨迹中观察到该结果”。

但重建manifest没有保存`device`和顶层`collision_scope`，也明确标记原始evaluator aggregate未
保留。当前正式协议要求CUDA、ego scope并在manifest中显式记录；因此本节不能把旧包提升为
与U44相同等级的正式CUDA验收。若用户考虑把重加权U30纳入最终production选择，只需对预先
固定的U30 checkpoint做一次四图CUDA、ego-scope、保存trace的确认，不训练、不改参数、不从
U27--U30事后选择。四图结果应继续逐图报告，尤其检查Nuerburgring薄余量。

Production U30的正式配对也存在独立缺口：Austin保留600 traces但没有manifest；现存
Hockenheim package没有trace/device且`26/355`与已记录CUDA `26/356`不一致；MoscowRaceway和
Nuerburgring当前未保留结果包。只有要宣称某候选正式超过Production U30时，才需要fresh重评
Production四图；这不影响以BC为最低线接受U44。

### 27.6 收口判决与停止规则

1. 不启动新训练；不重跑K25、reference-KL、role比例、reward、gate、std、hold或梯度投影；
2. Production别名继续指向U30，直到用户明确产品目标并授权切换；
3. 四图安全优先选择U44；Austin/near400优先保留Production U30；若四图超车优先且接受
   中间安全点，先完成重加权U30的一次固定CUDA确认；
4. 不把near400旧否决删除，也不让它在当前口径下越权否决四图BC资格；
5. 不从U27--U30或U42--U45按事后最小collision挑checkpoint；U30和U44分别是当前预先固定
   候选；
6. 未来若重开transition advantage或role机制分析，历史临时张量已清理，必须fresh采样；这不是
   零成本重放，也不是恢复已识别根因。

## 28. Austin-only PPO四图碰撞身份、接触几何与经验上限诊断（2026-08-06，无训练）

本节回答新的目标问题：在训练严格限制为Austin、另外三张地图只用于泛化测试的前提下，现有
actor为什么停在四图`62--94`次collision，离`collision < 40 / overtake > 1500`还有什么缺口。
本节只复算既有模型结果与numeric traces，没有训练、没有修改actor/reward/critic、没有生成
checkpoint。数值单位默认为四图共2400个确定性episode上的`ego collision / overtake`。

### 28.1 判决表

| 模型 | same-line collision | off-line-fast | off-line-slow | 总collision | overtake | 本节判决 |
|---|---:|---:|---:|---:|---:|---|
| Canonical BC | 64 | 58 | 7 | 129 | 1445 | 行为起点 |
| Production U30 | 74 | — | — | 94 | 1508 | off-line合计20；跨图只保留headline，无法再拆fast/slow |
| 前向走廊时间相关速度探索U44 | **21** | 35 | **6** | **62** | 1478 | same-line安全端 |
| ordinary异线高速重加权U30 | 35 | **31** | 7 | 73 | **1516** | off-line超车端 |

Production U30的三张跨地图CUDA记录为same-line/off-line `66/14`，Austin为`8/6`，故四图
headline为`74/20`；但跨地图完整trace未保留，所以它不进入下文Venn、首次碰撞帧几何和
正式配对复算。BC、U44与重加权U30则各有四张地图完全相同的600个episode key和2400条
numeric trace，可作严格身份与几何比较。

**判决：这不是物理动力学或actor参数容量的已证实上限，而是当前Austin-only、单阶段PPO、
共享actor和已测粗粒度探索/采样旋钮形成的经验前沿。** 三个策略分别位于不同same/off-line
工作点，不能用一个总collision数字写成单调“更好/更差”。

### 28.2 问题、cohort与几何定义

训练仍只允许Austin。Hockenheim、MoscowRaceway和Nuerburgring只用于测试泛化；本节利用其
既有评测结果刻画失败，不授权把测试地图放回训练或按地图逐项调参。

互斥regime：

- `same-line`：opponent使用ego的`raceline1`；四图共800场景；
- `off-line-fast`：opponent位于raceline0/2且speed scale为0.7/0.8；共800场景；
- `off-line-slow`：opponent位于raceline0/2且speed scale为0.5/0.6；共800场景。

碰撞身份以`(map, episode key)`配对：

- inherited/shared：BC与treatment都发生ego collision；
- removed：BC碰撞、treatment不碰撞；
- created：BC不碰撞、treatment碰撞。

ego-opponent几何取`ego_opp_collision`首次为真的post-step terminal frame。将对手中心位置变换到
ego车体系，bearing绝对值`<=45°`记为front、`>=135°`记为rear，其余为side；relative yaw是
两车yaw差wrap到`[-180°,180°]`后的绝对值，`<=30°`只称为parallel。**bearing是中心相对方位，
不是碰撞接触法向或冲量方向**；因此本节能判断前/侧/后与平行/斜交，不能给出精确物理撞击角。

实现事实的搜索入口保持为：场景regime由`ppo/scenarios.py`的`is_same_line`和
`is_offline_fast`定义；collision marker和两车pose由multiagent evaluator的numeric trace合同
产生。本节未增加新的production模块或实验脚本。

### 28.3 主要限制不是BC硬例，而是碰撞身份迁移

相对BC的严格配对结果：

| treatment | 总collision | inherited | removed | created | paired exact p |
|---|---:|---:|---:|---:|---:|
| U44 | 62 | 35 | **94** | 27 | `7.14e-10` |
| 重加权U30 | 73 | 45 | **84** | 28 | `1.11e-7` |

U44的removed/created按regime为：removed `same 50 / off-fast 39 / off-slow 5`，created
`same 7 / off-fast 16 / off-slow 4`。重加权U30为：removed `38/43/3`，created `9/16/3`。
两者都能消除多数BC失败，也都在此前干净的场景新造约二十余次碰撞。

三模型碰撞Venn完全分解为：

| 碰撞成员关系 | 场景数 | same / off-fast / off-slow |
|---|---:|---:|
| 仅BC碰撞 | 67 | `35 / 29 / 3` |
| 仅U44碰撞 | 20 | `5 / 13 / 2` |
| 仅重加权U30碰撞 | 21 | `7 / 13 / 1` |
| BC+U44，不含重加权 | 17 | `3 / 14 / 0` |
| BC+重加权，不含U44 | 27 | `15 / 10 / 2` |
| U44+重加权、BC干净 | 7 | `2 / 3 / 2` |
| BC+U44+重加权均碰撞 | **18** | `11 / 5 / 2` |

所以BC的129个collision中只有18个在三者上都保留。两个PPO actor新造场景的并集为
`27+28-7=48`，其中只有7个共同，即41个是臂特异迁移。U44与重加权U30共同碰撞场景总数为
25（18个BC继承、7个共同新造），不是62/73中的大多数。**这直接否定“剩余主要是一批固定、
不可解BC硬例”的解释。** 这18或25个也只能叫跨当前模型稳定失败，不能写成物理不可解。

新增碰撞跨地图分布为U44 `Austin/Hockenheim/Moscow/Nuerburgring=9/7/9/2`，重加权U30
`6/7/8/7`；不是单张地图独占。按speed scale，U44为`5/4/7/11`，重加权为`5/5/3/15`
（依次0.5/0.6/0.7/0.8）。0.8有富集，尤其重加权臂，但仍不能把全部问题简化成最高速度档。

### 28.4 新造失败以平行侧擦和超车后侧后接触为主

全体ego-opponent collision中，BC有`120/123`、U44有`53/56`、重加权U30有`67/67`
满足relative yaw不超过30度；三者relative yaw中位均约5度。当前双车任务的主要collision
本来就是同向、近似平行的交互，不是迎头或大角度交叉。

只看PPO相对BC新造的collision：

| treatment | 新造总collision | 新造ego-opp / wall | parallel ego-opp | opponent front / side / rear | relative yaw P25 / P50 / P75 |
|---|---:|---:|---:|---:|---:|
| U44 | 27 | `23 / 4` | `21/23` | `2 / 12 / 9` | `2.77° / 4.54° / 6.35°` |
| 重加权U30 | 28 | `25 / 3` | `25/25` | `3 / 9 / 13` | `2.68° / 4.55° / 6.71°` |

U44的新造车辆碰撞中`21/23`、重加权中`22/25`在碰撞时对手位于ego侧方或后方。结合近似
平行yaw，这支持“并排侧擦或ego刚超过后发生侧后重新接触”的几何描述；不应在没有完整
relative-progress时序标注的情况下把每个rear sector都正式命名为post-pass collision。

按regime看，U44的off-line-fast 35次collision全部为车辆侧/后接触（side 23、rear 12）；
重加权U30的off-line-fast为side 17、rear 13、wall 1，没有front rear-end。same-line则混合
front/side/wall：U44为`8/8/5`，重加权为`13/14/3`，后者另有5次rear。由此应保留两个直接
失败家族：

1. same-line跟车/避让：纵向减速、侧向规避与墙余量协调不足；
2. off-line通过：并排到前后排序切换时，速度与横向净空协调不足，形成平行侧/后接触。

更高层的共同问题是**interaction phase conditioning**：同一个actor必须根据对手处于前方、
并排或刚落后，以及净空是否仍在收缩，在“持续减速保持跟车”和“果断完成通过并保持侧后净空”
之间切换。它不是一个统一的“再加碰撞惩罚”问题，也不是单纯“转向太激进”。

### 28.5 为什么现有旋钮卡住，以及目标是否仍可达

三个已观察工作点清楚显示粗旋钮的作用：Production U30把off-line collision压到20，却有74次
same-line；U44把same-line压到21，却有41次off-line；重加权U30把二者推到35/38。继续改变
探索持续时间、噪声幅度、走廊门宽、ordinary异线比例或updates，首先移动的是这个工作点，
现有证据没有显示其能在同一regime内部识别“应减速”和“应完成超车”的具体状态。

两个上界用于区分“没有能力”和“不会选择”：

- **粗regime拼接**：same-line取U44、off-line-fast取重加权U30、off-line-slow取U44，得到
  `21+31+6=58 collision`；相应overtake为`11+722+794=1527`。即使拥有完美scenario元数据，
  只按这三个regime选actor也过不了`collision < 40`。
- **逐episode未来结局上界**：每个场景只要U44或重加权U30任一成功就取成功，并优先取
  overtake，得到`25 collision / 1557 overtake / 818 follow`。这是不可部署的hindsight oracle，
  但证明现有相同actor结构的行为并集已经超过目标。

因此`<40/>1500`在动作/结构层面没有被排除；缺口位于**regime内部的细粒度状态选择**。
它需要至少把现有58次粗拼接collision再减少19次，同时只允许从1527丢26次以内的overtake。
这不是继续调一个全局安全剂量能够合理保证的选择性。

当前最保守的因果边界：行为迁移已经被配对结果验证，但“共享output head固定梯度冲突”已被
§24否决，“固定50/50 role混合根因”已被§25否决。不能把现象进一步写成某一层网络或某个
optimizer公式的已知故障。Austin-only训练与三张未见地图之间的几何分布差异可能放大迁移，
但新增碰撞也存在于Austin且跨四图共享侧/后几何，因此“只是地图记忆”同样解释不完整。

### 28.6 合法下一步、停止规则和证据边界

下一步如果继续，只应先做**Austin-only离线相位可分性诊断**，不是训练：

1. 以Austin内的成功通过和新造平行侧/后碰撞为主cohort，按raceline、speed、起点/曲率和
   事件相位匹配，防止用明显不同场景获得虚高可分性；
2. 固定碰撞/最近接时刻前`1.5s、1.0s、0.5s`窗口，比较signed longitudinal gap、lateral
   offset、closing rate、ego speed/command、steering、yaw/slip代理，以及按实际actor从episode
   起点重放的GRU hidden；
3. 按startpoint分组切分，不能随机拆transition造成相邻帧泄漏；主要判据是仍有操控权的早期
   窗口能否稳定区分“继续安全通过”和“即将发生平行侧后接触”；
4. 361D observation或hidden若能稳定区分，才讨论一个保持actor结构、无runtime组件、单阶段
   Austin PPO中的条件行为保持目标；若不能区分，才把现有观测合同列为候选结构瓶颈。

停止规则：

- 用户明确禁止multi-map PPO；三张跨地图只能测试，不能用于训练、选阈值或逐地图调参；
- 不再扫ordinary异线高速比例、hold K、speed std、退火、gate宽度、updates、reward或collision
  role比例来追`<40/>1500`；这些是粗剂量轴，现有结果不足以支持其所需的regime内选择性；
- 不把逐episode `25/1557` oracle冒充可部署模型，也不据此加入runtime selector/shield；
- 不依据中心bearing把collision写成精确接触法向；需要物理撞击角时必须增加仿真接触点/法向
  记录后重新定义；
- 不从本节直接启动reference-policy、imitation、蒸馏或辅助loss。它们仍需用户授权，并且必须
  先通过Austin-only早期状态可分性和行为保持预检；
- 若外部审查提出新轴，必须解释它如何在同一个same/off-line regime内部区别安全与危险状态，
  而不是只把策略推向U44或重加权U30已有前沿上的另一个点。

证据边界：单seed 42；固定四图面板；跨地图已经参与过历史门控设计，不是干净最终留出；
重加权U30完整trace支持本节计数与几何，但manifest缺CUDA device/collision scope；Production U30
缺当前完整跨地图trace，因此只进入headline regime表。本节是描述性与诊断性复算，除既有
fresh-start训练臂外没有新的因果实验。即便这些边界存在，“大量removed同时存在大量created”、
“新增车辆碰撞以近似平行侧/后接触为主”和“粗regime拼接仍达不到40”都由完整同场景身份与
numeric pose直接支持。

## 29. Phase-spillover与Pressure-conditioning最小诊断（2026-08-07，无训练）

本节只筛查两个外部审查提出的候选机制，不训练、不修改actor/reward/critic，也不增加评测
panel。输入是canonical BC与前向走廊门控时间相关速度探索U44在同一Austin600上的完整数值
trace。事件窗口固定为碰撞或最近接时刻前`1.5s`，pressure另在`1.5/1.0/0.5s`三个时刻取样。

### 29.1 判决表

| 候选机制 | 核心cohort与结果 | 机制判决 | 是否准入训练 |
|---|---|---|---|
| gate关闭后50步block跨相位继续 | U44相对BC新造的6个ego-opponent collision中，碰撞前1.5s为`0/6`出现gate或active block；24个同条件安全对照同为`0/24` | 这些Austin新造车辆碰撞窗口没有可被phase-bounded提前截断的block | **否** |
| frozen BC pressure在关键beam饱和 | 新造碰撞18个事件时刻样本的pressure中位`0.539`、`|dp/dx|`中位`0.311/m`，`p<0.05`与`|dp/dx|<0.05/m`均为`0/18` | 关键侧方/侧后回波没有明显pressure饱和或灵敏度塌缩 | **否** |

### 29.2 Phase-spillover重建

主cohort是U44发生ego-opponent collision、BC同场景无ego collision的6个Austin episode；它们
全部是off-line raceline。安全对照在同一opponent raceline与speed scale内选择BC和U44都不
碰撞、ego起点循环距离最近的4个，共24个。正向机制对照是BC碰撞而U44修复的9个same-line
episode。

每个U44 trace用当前production完全相同的Frenet投影、车辆`0.58m × 0.31m`几何、`front gap
<2.0m`、opponent横向中心`|d|<0.25m`和正OBB横向重叠重建gate；当gate首次为真时按当前实现
启动50步block，并在gate关闭后继续倒计时。事件前1.5s结果为：

| cohort | episode | active steps | gate关闭后的active steps | 进入并排/后方后的active steps | 有跨相位block的episode |
|---|---:|---:|---:|---:|---:|
| U44新造ego-opponent collision | 6 | 0 | 0 | 0 | 0 |
| matched safe | 24 | 0 | 0 | 0 | 0 |
| U44修复的same-line collision | 9 | 1009 | 98 | 7 | 1 |

same-line正向对照共重建28个block，其中1个进入并排/后方相位，且`5/9` episode确实存在gate
关闭后的active step，说明零计数不是重建器未触发。**直接的phase-spillover假说未获支持：
U44这6个Austin新造车辆碰撞在失败窗口内从未进入走廊门，也没有遗留block。** phase-bounded
仍可能通过改变训练分布间接改变共享参数，但本诊断没有给这种间接效应提供正证据，不能据此
支付一条45-update训练。

### 29.3 Pressure-conditioning重建

当前BC、Production U30和U44的360维`k`保持相同，统计为min/mean/max/std
`0.2408/0.8110/1.3349/0.2419`。在上述6个新造碰撞和24个安全对照的三个提前时刻，按真实
LiDAR FOV与降采样角度定位opponent bearing，并在相邻5个beam中取最近回波。所有碰撞
`18/18`和安全`72/72`时刻的opponent bearing均在LiDAR FOV内。

| cohort | raw LiDAR中位 | 对应k中位 | pressure中位 | `|dp/dx|`中位 | `p<0.05` | `|dp/dx|<0.05/m` |
|---|---:|---:|---:|---:|---:|---:|
| 新造collision | `1.218m` | `0.880` | `0.539` | `0.311/m` | `0/18` | `0/18` |
| matched safe | `1.448m` | `0.724` | `0.526` | `0.264/m` | `3/72` | `3/72` |

为排除“只取最近回波”造成的假象，又只对同一5-beam局部窗口做一次必要校验：collision
`90`个beam-time样本的pressure/灵敏度中位为`0.404/0.300m^-1`，低于上述两个阈值的样本仍
均为0；safe的`360`个样本为`0.463/0.245m^-1`，各有20个低值。**饱和反而只零星出现在安全
对照，失败窗口保留了充足的局部变化率。** 因此没有继续做shadow gradient、pressure LR或
trainable-k实现；非零梯度本身也不能补上缺失的conditioning证据。

### 29.4 durable verdict与停止规则

1. 不启动trainable pressure正式run；它仍是12-key兼容的未训练参数轴，但当前失败窗口没有
   显示pressure饱和或灵敏度不足。只有新的匹配Austin证据显示关键回波被明显压平，才重开。
2. 不启动phase-bounded temporal block正式run；当前U44新造Austin车辆碰撞窗口中没有block
   可截断。只有fresh训练分布中的匹配失败窗口证明跨相位block显著富集，才重开。
3. 两个负结果只否决当前因果理由，不证明`k`或相位终止在所有任务上永久无用，也不解释U44
   的全部共享参数行为迁移。
4. 当前不产生新actor、不组合两个方向、不扫描LR/hold/gate。Production与四图候选状态保持
   不变；更高`<40/>1500`目标仍缺少能在同一regime内部稳定选择动作的新证据。

## 30. Interaction-phase早期可分性最小诊断（2026-08-07，无训练）

### 30.1 判决表

| 阶段 | 核心结果 | 判决 |
|---|---|---|
| actor-visible早期可分性 | 30个平行侧/后collision；1.5/1.0/0.5s最佳actor-visible AUROC为`0.627/0.773/0.598`，GRU hidden为`0.529/0.617/0.580`；唯一高点的fold范围`0.538--1.000` | **未稳定通过** |
| steering动作响应面 | 上游准入门未通过 | **未运行** |
| fresh PPO coverage/credit | 上游准入门未通过 | **未运行** |
| phase-gated steering正式训练 | 缺少可靠early conditioning | **不准入** |

固定控制是U44、Austin near400开发面板、seed 42、CUDA逐步重放、startpoint分组5折和固定
linear ridge。没有新actor、正式eval或训练。

### 30.2 为什么只做第一道准入门

外部审查建议一次完成Austin-only interaction phase可观测性、动作响应、fresh PPO coverage/
credit和候选训练设计。问题方向与§28.6一致，但按原规格会同时引入phase定义、probe、闭环
动作响应和PPO归因四类问题，不符合“先用最便宜证据决定是否继续”的要求。因此本轮预先固定：

1. 只检验actor当前可见信息能否在仍有操控权的提前窗口区分未来安全通过与平行侧/后碰撞；
2. 若该门不稳定通过，停止，不运行转向响应面、P20上界、fresh PPO credit或正式训练；
3. 不新建脚本/结果树，不改actor、critic、reward、pool、训练入口或正式eval。

near400是Austin开发/机制面板，不具有四图正式验收权；这里使用它是因为U44包含37个完整
ego-opponent collision trace和288个安全overtake，避免Austin600只有6个新造车辆碰撞的极小
样本问题。结果只能决定是否值得继续诊断，不能改变production或正式候选排序。

### 30.3 Cohort、相位和匹配

在U44 near400的37个ego-opponent collision中，以碰撞帧ego车体系定义接触几何：相对yaw
不超过30度且对手中心侧向分量不小于纵向分量为`side_parallel`，同角度范围内对手在后且
纵向分量占优为`rear_parallel`。得到：

| 几何 | episode |
|---|---:|
| side parallel | 23 |
| rear parallel | 7 |
| side angled | 4 |
| front parallel | 3 |

主正例为前两类共30个。事件时刻对collision取首次ego-opponent marker，对安全overtake取
两车中心最近接时刻；分别回退1.5s、1.0s和0.5s。当前interaction phase只由当时两车pose和
`0.58m × 0.31m` OBB投影定义为front/side/recent/clear，未来outcome不进入phase。

每个正例只在288个安全overtake中选择同opponent raceline、同speed scale、同当前phase的
最近对照；距离变量固定为signed longitudinal center offset、OBB lateral surface clearance、
局部曲率和左右LiDAR最小值。优先不重复使用安全episode，每个正例最多4个。最终数据为：

| 距事件 | collision | 样本总数 | 唯一安全episode |
|---|---:|---:|---:|
| 1.5s | 30 | 132 | 83 |
| 1.0s | 30 | 141 | 91 |
| 0.5s | 30 | 144 | 90 |

按ego startpoint的50-waypoint sector分组，贪心分配为5折，每折恰好6个collision；同一sector
不跨fold。probe固定为标准化、按维度归一的linear ridge，正负类在训练折内等权，不扫描
正则、模型或阈值。比较四组输入：5D matching geometry、raw 360D LiDAR+previous speed、
frozen BC pressure 360D+previous speed、U44 GRU hidden 1680D。

### 30.4 重放校验

hidden必须从episode起点逐步重放。评估器首帧输入不是trace中reset后的实测0速，而是ego
raceline起点参考速度的`0.9×`；之后每步输入上一帧实测速度。严格复现该合同时，所有参与
轨迹、所有`action_applied=true`行的网络raw action与trace最大绝对误差为`0`。

一次临时预检曾错误地用trace首帧0速初始化，产生最大约`4.85`的动作误差；该结果和由其产生
的hidden统计全部作废，不进入下表。把整段序列一次性送入cuDNN也会因计算路径不同累积约
`0.045`的raw-action差，因此最终只采用与正式评估器相同的逐步CUDA重放。

### 30.5 结果

| 距事件 | geometry AUROC（fold范围） | raw+speed | pressure+speed | hidden |
|---|---:|---:|---:|---:|
| 1.5s | 0.487（0.413--0.615） | **0.627**（0.500--0.800） | 0.606（0.400--0.792） | 0.529（0.344--0.870） |
| 1.0s | 0.556（0.250--0.740） | 0.690（0.556--0.958） | **0.773**（0.538--1.000） | 0.617（0.326--0.889） |
| 0.5s | **0.667**（0.490--0.907） | 0.572（0.383--0.960） | 0.598（0.571--0.913） | 0.580（0.448--0.840） |

1.0s pressure的0.773是唯一超过0.75的actor-visible点，但每折只有6个正例，fold范围从0.538
到1.000，不能称为稳定。1.5s时raw/pressure只有0.627/0.606；0.5s时两者反而降至
0.572/0.598，而5D geometry已达0.667。GRU hidden在三个时刻均不超过0.617，且fold方向不稳。

这组结果不支持“GRU已经稳定编码了可用于条件动作的早期危险状态”，也不支持“frozen
pressure明确丢失raw LiDAR中的决定性信息”；raw和pressure没有跨三个提前时刻保持一致优势。
它只提示1.0s pressure可能含局部信号，不能据此选择phase阈值或训练机制。

### 30.6 durable verdict与停止规则

1. **不进入动作响应阶段。** 未先建立稳定早期状态判别时，扫描转向残差和持续时间会把相位
   检测误差、动作可达性和闭环迁移混在一起。
2. **不进入fresh PPO-credit审计。** 当前最便宜的上游准入门未通过，没有必要采集更重的
   rollout/GAE/P20张量；本结论不等于宣称现有GAE已经正确。
3. **不启动side-phase temporally correlated steering exploration，也不重开trainable k。**
   前者缺少可靠部署期conditioning，后者同时被§29的非饱和证据和本节raw/pressure无稳定
   差距约束。
4. 负结果只适用于本次U44 near400、单seed 42、固定线性读出和30个正例；它不证明361D观测
   永久不可用，也不把当前经验前沿写成理论上限。
5. 只有新增独立Austin正例后，预先固定的actor-visible表征在startpoint分组下同时表现出
   跨fold与跨提前时刻的稳定区分，才重开后续action-response/credit链；不得在当前小样本上
   追加MLP、阈值或特征扫描来挑出一个好点。

## 31. 真值几何/速率对当前策略失败的线性早期可分性预检（2026-08-07，无训练）

### 31.1 判决表

| 检验 | 结果 | 判决 |
|---|---|---|
| 真值几何+速率的线性早期可分性 | 1.5/1.0/0.75/0.5s 中位 AUROC `0.429/0.725/0.667/0.656`，fold 范围最宽 `0.213--0.866` | **未达本轮预注册门** |
| 同一读出在 0.25s | `0.861`（fold `0.603--0.980`） | 通过阈值，但已在操控权边界 |
| 速率项的独立贡献 | RATE-only 在 0.25s 仅 `0.481`；静态几何同点 `0.825` | **本任务下 closing rate 无决定性增益** |
| 训练期辅助表征臂 | 本轮预检未通过，但该预检不检验辅助损失的作用通路 | **本轮不直接启动完整训练；方向未否决** |

用户提出的机制假说是：几何信息存在于输入、但没有稳定进入 GRU 内部表征，因此值得用训练期
辅助表征损失改变 GRU 的状态形成方式。本轮不训练、不采样、不改任何源码，只回答一个**更窄
的前置问题**：这些几何量本身，经固定线性读出，在还有操控权的提前窗口能不能区分当前策略
未来的平行侧/后碰撞。这既不是信息论上界，也不是辅助损失的作用通路，限制见 §31.2。

### 31.2 本轮测了什么、以及它不能回答什么

本轮检验的命题是：**在 U44 当前策略产生的轨迹上，选定的 privileged 真值几何与一阶速率，
经固定线性 ridge，能否在仍有操控权的提前窗口预测未来的平行侧/后碰撞。** 它绕过网络表征，
直接用真值,因此可以判断"这一族量在这个预测任务上有没有线性可用信号"。

**它不能回答三件事，本节任何结论都不得外推到这三件事上：**

1. **它不是信息论上界。** 测得的是固定线性读出能抽取的量，属于这些真值特征信息含量的
   下界。GRU、辅助头与 actor 都是非线性模型，几何量之间的交互可能被非线性表示重新组织。
   除非使用接近 Bayes 最优的分类器，否则不能称为天花板。
2. **它测的任务不是辅助损失的作用通路。** 辅助几何目标的用途是让 PPO 更容易依据当前几何
   选择动作，而不是部署一个"未来会不会撞"的分类器。某个特征在**当前策略**的轨迹分布上
   预测力弱，不蕴含它对改进策略动作没有价值——要改变的正是那个分布。
3. **它的覆盖面很窄。** 只有成对 ego-opponent 几何与一阶速率。未覆盖 ego 转向/速度指令
   历史、yaw rate/slip 与横向动态、墙面余量与赛道局部结构、对手 planner 意图，以及 GRU
   中更长时间历史及其非线性交互。

### 31.3 Cohort 与复现校验

固定控制与 §30 相同：U44、Austin near400 开发面板、seed 42、单一固定 linear ridge
（`lambda=1.0`）、按 50-waypoint startpoint sector 分组。数据来自 U44 near400 的 400 条
numeric trace。

独立重建的 cohort 与 §30.3 完全一致：37 个 ego-opponent collision、288 个安全 overtake；
碰撞帧接触几何分类为 `side_parallel 23 / rear_parallel 7 / side_angled 4 /
front_parallel 3`，主正例为前两类共 30 个，碰撞相对 yaw 中位 `3.67°`。

事件时刻对 collision 取首次 ego-opponent marker，对安全 overtake 取两车中心最近接时刻。
匹配要求同 opponent raceline、同 speed scale、同当前 phase，按体坐标纵/横向偏移取最近，
每个正例最多 4 个对照并优先不复用。样本量为 1.5/1.0/0.75/0.5/0.25s 各
`145/130/144/142/143`，正例恒为 30。1.5s 有 3 例、0.25s 有 2 例因无同 phase 候选而放宽到
同 raceline+speed，已计数。§30 对应样本量为 `132/141/144`；差异来自 phase 定义与匹配距离
变量的实现细节，不影响下节的 G5 复现一致性。

### 31.4 特征集与结果

四组特征，全部为 privileged 真值或 actor 可见几何，不经过任何网络：

- `G5`：signed longitudinal center offset、OBB lateral clearance、局部曲率、左右 LiDAR 最小值
  （§30 的 5D matching geometry，用作流程复现对照）；
- `GEO static`：signed longitudinal / lateral center offset、OBB longitudinal / lateral /
  total clearance（用户提案中的"signed longitudinal gap、lateral clearance"）；
- `RATE only`：上述纵横偏移与两个 OBB clearance 的时间导数、两车实测速度差
  （用户提案中的"closing rate"，§30 未测过的唯一新维度）；
- `GEO+RATE`：前两者并集。

中位数取 10 次分组 5 折重复；括号为全部 50 折的范围。

| 特征集 | 1.5s | 1.0s | 0.75s | 0.5s | 0.25s |
|---|---:|---:|---:|---:|---:|
| `G5`（§30 复现） | 0.490 | 0.589 | 0.715 | 0.692 | 0.833 |
| `GEO static` | 0.418 | 0.559 | 0.674 | 0.582 | 0.825 |
| `RATE only` | **0.567** | 0.604 | 0.535 | 0.670 | **0.481** |
| `GEO+RATE`（合并） | 0.429 | **0.725** | 0.667 | 0.656 | **0.861** |

`GEO+RATE` 的 fold 范围依次为 `0.213--0.643`、`0.361--0.866`、`0.319--0.812`、
`0.323--0.850`、`0.603--0.980`。

**流程复现校验**：`G5` 得到 `0.490/0.589/0.692`，§30 记录为 `0.487/0.556/0.667`；本轮
pipeline 独立复现了 §30 的 5D 几何 probe。

**稳健性**：`lambda` 取 `0.1/1.0/10.0` 时 `GEO+RATE` 分别为
`0.448/0.748/0.648/0.689/0.812`、`0.429/0.725/0.667/0.656/0.861`、
`0.433/0.667/0.632/0.639/0.792`，形态不变。200 次标签置换的分组 null 中位约 `0.50`、
95 分位为 1.0s `0.659`、0.25s `0.630`；观测值对应 `p=0.005` 与 `p<0.005`，因此 1.0s 与
0.25s 的信号本身是真实的，不是纯噪声。

### 31.5 可以支持与不能支持的判断

**可以支持：**

- 在 U44 当前策略产生的轨迹上，选定的 `GEO+RATE` 真值特征经固定线性 ridge，无法在
  `>=0.5s` 稳定预测未来的平行侧/后碰撞；最好的 1.0s 中位为 `0.725`，而 fold 分布很宽。
- 在这个预测任务上，closing rate 没有表现出决定性独立增益：`RATE only` 在唯一较可分的
  0.25s 反而降到 `0.481`，而静态几何同点已达 `0.825`；晚期出现的可分性是位置性的。
- 因此按本轮预注册门，不应据此直接启动完整训练。

**不能支持（明确记录，防止后续引用时读强）：**

- 不能说"信息不存在"或"这是表征学习的上界"。见 §31.2 第 1 点。
- 不能说"几何量对改进策略无价值"。本轮测的是对**当前策略失败**的预测力，不是辅助损失的
  作用通路。见 §31.2 第 2 点。
- 不能说"失败只是闭环轨迹属性、不存在可早期识别的危险状态"。闭环属性与早期状态属性不
  互斥；更好的当前状态表征仍可能改变后续闭环。而且未覆盖的量族见 §31.2 第 3 点。
- 不能把 fold 下界当作统计置信下界。只有 30 个正例、每折 6 个；10 次重复分折不增加独立
  正例数，最小 fold 值容易被单个 fold 拉低。本轮预注册门里"fold 下界 >0.6"这一半设计
  不合理，后续预检不应照抄。

与 §28 的关系只能写成一致性观察，不是推论：两条 PPO 臂共新造 48 个不同场景、仅 7 个共同，
失败身份是臂特异的；本节表明这些失败在 `>=0.5s` 的成对几何/速率上没有线性可分信号。两者
都与"失败强烈依赖具体策略"相容，但都不排除存在本轮未覆盖的早期状态信号。

### 31.6 判决、预注册对照与下一步

本轮预注册门为"中位 AUROC >= 0.75 且 fold 下界 > 0.6，且该点位于仍有操控权的提前窗口"。
逐点核对：

| 提前时刻 | 中位 >=0.75 | fold 下界 >0.6 | 操控权 | 是否通过 |
|---|---|---|---|---|
| 1.5s | 否（0.429） | 否（0.213） | 有 | 否 |
| 1.0s | 否（0.725） | 否（0.361） | 有 | 否 |
| 0.75s | 否（0.667） | 否（0.319） | 有 | 否 |
| 0.5s | 否（0.656） | 否（0.323） | 有 | 否 |
| 0.25s | 是（0.861） | 是（0.603） | **边界** | 阈值通过但窗口不可用 |

**判决：本轮不直接启动 `CT-v2 + 辅助表征 loss` 的完整训练。** 用户此前是条件授权，本轮
提供的证据不足以触发它。**但这不是对训练期辅助表征学习的否决**：本轮检验的是一个不同的
命题（真值特征对当前策略失败的线性预测力），没有检验辅助损失的实际作用通路。

**正确的下一步仍是原最小预检，但用途必须限定清楚：**

1. 比较 raw/frozen pressure 与 GRU hidden 对**当前连续几何**的解码能力；
2. 指标用连续量的 `R^2`/`MAE`，**不预测未来 collision**；目标取 P20 中的
   `delta_s`、`relative_lateral`、`obb_longitudinal_clearance`、`obb_lateral_clearance`、
   `relative_long_velocity`、`relative_lat_velocity`、`wall_clearance`；
3. 判据是：当前几何在输入中可解、在 hidden 中明显不可解 → 证明存在真实的内部表征缺口；
4. 该缺口只授权**一次** `PPO + auxiliary representation loss` 训练，不保证最终安全收益；
5. 首次实验不叠加 reference KL、行为锚或任何其他变量。

该预检必须包含的三个对照，否则正分支不可读：

- **滞后对照。** 用 hidden 解码 `t`、`t-0.1s`、`t-0.2s`、`t-0.3s` 的同一几何量并报告最佳
  滞后。若 hidden 的劣势能被时间平滑解释（即它准确编码了**过去某时刻**的几何），那是
  递归状态的正常行为，不是表征缺口，辅助损失不应据此启动。
- **维度匹配对照。** raw 361D 与 hidden 1680D 维度差别很大；必须给出维度匹配的读出
  （例如两侧各降到同一 rank，或对 raw 做同维随机投影），并同时报告 train/test `R^2` 差，
  以区分"信息缺失"与"读出容量差异"。
- **样本与分组。** 当前几何解码不受 30 个正例限制：400 条 trace × 约 800 步可提供十万量级
  样本，按 ego startpoint 分组切分即可。统计功效不是这项预检的制约，因此不得沿用本节
  "fold 下界"式的门槛设计，应预注册 `R^2` 差值阈值与分组区间。

停止规则：

1. 不在当前 30 个正例上追加 MLP、非线性 probe、特征扫描或阈值搜索来挑出好点（同
   §30.6 rule 5）。
2. 若上述解码预检显示 hidden 与输入的差距可由滞后或读出容量解释，则不启动辅助表征训练；
   若显示真实缺口，按上面第 4、5 条只授权一次单变量训练。
3. 本节结论只适用于 U44 near400、单 seed 42、固定线性 ridge 与所选量族；引用时必须连同
   §31.2 的三条限制一起引用，不得单独引用表格数字得出"没有优化空间"或"方向已否决"。

## 32. 当前交互几何的表征缺口预检（2026-08-07，无训练）

### 32.1 判决表

| 检验 | 结果 | 判决 |
|---|---|---|
| 回放合同 | 400条trace、全部`action_applied`行的raw action最大绝对误差`0` | 通过 |
| 输入 vs hidden 解码当前几何（全维） | hidden 在 9 个目标中 5 个优于输入 | 缺口方向与假说相反 |
| 维度匹配读出（**fold-local** PCA K=64 / K=256） | hidden 在 **9/9** 目标上优于两种输入，优势 `0.086--0.407` | **缺口不成立** |
| 滞后对照（**fold-local** PCA-256） | lag 0 在 9 个目标中 8 个最优 | hidden 不是"只编码过去" |
| 预注册准入门 | 无任何目标满足"输入`R^2>=0.5` 且 hidden `<=` 输入`-0.25`" | **不准入** |

**结论：`GRU hidden` 对当前交互几何的线性可解码性不低于、且在维度匹配下一致高于 actor 自身
输入。§31.6 设定的"真实内部表征缺口"条件不成立，方向与假说相反。**

### 32.2 预注册与设计

本节执行 §31.6 规定的预检，**判据在拟合前写死**：判定存在真实内部表征缺口须同时满足
(a) 输入侧 `R^2 >= 0.5`；(b) `hidden R^2 <= 输入 R^2 - 0.25`；(c) 两者 5 折范围不重叠；
(d) 该差距不能被最佳滞后解释；(e) 在维度匹配读出下仍然存在。

固定控制：U44 actor、Austin near400 的 400 条 numeric trace、seed 42、CUDA 逐步回放、
按 ego startpoint 的 50-waypoint sector 分组 5 折、ridge 读出且 `lambda` 由**训练折内的
分组内层 CV** 从 `1e1--1e7` 选出。不训练、不采样、不改任何源码。

**回放合同校验。** 按 `eval_multiagent.py` 合同逐步回放：首帧 speed 输入为 ego raceline
起点参考速度的 `0.9x`（取 `load_raceline_waypoints` 的第 4 列，**不是** raceline CSV 的第
3 列，后者是 heading），其后每步输入上一帧实测速度。全部 400 条 trace、所有
`action_applied=true` 行的网络 raw action 与 trace 的最大绝对误差为 `0`。一次早期实现误用
CSV 第 3 列作为起点速度，产生最大 `5.45` 的动作误差，该结果已作废、不进入下表。

**采样与目标。** 每条 trace 每 10 步取一个样本，共 `31,345` 个样本、24 个 startpoint
sector。目标为 9 个连续量，定义严格对齐 `PrivilegedStateExtractor.features` 的物理量
（这里保留物理单位、不做 P20 的归一化与 clip）：`delta_s`、`relative_lateral`、
`obb_longitudinal_clearance`、`obb_lateral_clearance`、`relative_long_velocity`、
`relative_lat_velocity`、`wall_clearance`、`left_margin`、`right_margin`。相对速度由
pose 有限差分得到，因此包含真实 slip，不依赖 trace 缺失的 slip 角；`wall_clearance` 使用
f110 scan simulator 的 Austin 距离场，与训练期同一 `OccupancyMapClearance` 合同。

### 32.3 结果

全维读出（`raw 361D` / `pressure 361D` / `hidden 1680D`），分组 5 折 test `R^2`：

| 目标 | raw+speed | pressure+speed | GRU hidden | 最佳输入 − hidden |
|---|---:|---:|---:|---:|
| `delta_s` | 0.219 | 0.310 | 0.269 | +0.041 |
| `relative_lateral` | 0.104 | 0.105 | **0.290** | −0.185 |
| `obb_lon_clearance` | 0.137 | 0.312 | −2.133 | +2.445 |
| `obb_lat_clearance` | −0.009 | 0.026 | **0.149** | −0.123 |
| `relative_long_velocity` | 0.262 | 0.316 | 0.225 | +0.090 |
| `relative_lat_velocity` | 0.098 | 0.120 | **0.429** | −0.308 |
| `wall_clearance` | 0.073 | 0.288 | **0.504** | −0.216 |
| `left_margin` | 0.492 | 0.605 | **0.737** | −0.132 |
| `right_margin` | 0.527 | 0.633 | 0.560 | +0.073 |

`obb_lon_clearance` 的 `-2.133` 完全由单个 sector fold 造成（逐折为
`-12.97 / 0.45 / 0.63 / 0.63 / 0.60`）。同一 fold 也把 hidden 的全目标均值从约 `0.50`
拉到 `-1.44`。这是 1680 维读出在分组外推下的条件数问题，不是稳定的表征劣势。

**控制 A：维度匹配读出（fold-local PCA）。** 在每个 outer fold 内**只用训练 sector** 拟合
均值与主成分方向，再变换该 fold 的 test sector；随后保持同一 ridge 与内层 `lambda` 选择：

| 目标 | K=64 raw | K=64 pressure | K=64 hidden | K=256 raw | K=256 pressure | K=256 hidden |
|---|---:|---:|---:|---:|---:|---:|
| `delta_s` | 0.214 | 0.271 | **0.611** | 0.212 | 0.296 | **0.703** |
| `relative_lateral` | 0.120 | 0.100 | **0.233** | 0.116 | 0.115 | **0.272** |
| `obb_lon_clearance` | 0.121 | 0.299 | **0.454** | 0.146 | 0.316 | **0.553** |
| `obb_lat_clearance` | −0.004 | 0.043 | **0.158** | −0.008 | 0.059 | **0.278** |
| `relative_long_velocity` | 0.268 | 0.297 | **0.514** | 0.260 | 0.314 | **0.607** |
| `relative_lat_velocity` | 0.100 | 0.114 | **0.333** | 0.084 | 0.114 | **0.465** |
| `wall_clearance` | 0.058 | 0.168 | **0.407** | 0.066 | 0.229 | **0.516** |
| `left_margin` | 0.485 | 0.579 | **0.670** | 0.497 | 0.598 | **0.731** |
| `right_margin` | 0.489 | 0.585 | **0.671** | 0.537 | 0.603 | **0.705** |

**维度匹配后 hidden 在 9/9 目标上严格优于两种输入**，K=64 优势 `0.086--0.340`、K=256
优势 `0.102--0.407`。全维表里 hidden 的三处劣势与 `obb_lon_clearance` 的崩溃在维度匹配后
全部消失，确认它们是读出容量/条件数效应。

**数据泄漏更正（2026-08-07）。** 本控制的初版在**全部 31,345 个样本**上拟合 PCA 之后才
进入分组交叉验证，test sector 参与了均值与主成分方向的估计，因此不是严格 held-out；滞后
对照初版同样使用该全数据 PCA。上表与下表均已改为 fold-local 拟合。修正前后数值差异很小
（例如 K=256 `delta_s` `0.699 -> 0.703`、`relative_lat_velocity` `0.459 -> 0.465`、
`left_margin` `0.730 -> 0.731`），"hidden 在 9/9 上更优"的结论不变；但初版结果不得再被
引用，本节只保留 fold-local 版本。

**控制 B：滞后（fold-local PCA-256）。** 用 hidden 解码 `t`、`t-0.1s`、`t-0.2s`、`t-0.3s`
的同一几何量：

| 目标 | lag 0 | −0.1s | −0.2s | −0.3s | 最佳 |
|---|---:|---:|---:|---:|---|
| `delta_s` | **0.703** | 0.698 | 0.695 | 0.695 | 0 |
| `relative_lateral` | **0.272** | 0.256 | 0.251 | 0.247 | 0 |
| `obb_lon_clearance` | **0.553** | 0.541 | 0.531 | 0.524 | 0 |
| `obb_lat_clearance` | **0.278** | 0.255 | 0.224 | 0.205 | 0 |
| `relative_long_velocity` | **0.607** | 0.605 | 0.603 | 0.597 | 0 |
| `relative_lat_velocity` | 0.465 | 0.477 | 0.480 | **0.483** | −0.3s |
| `wall_clearance` | **0.516** | 0.495 | 0.452 | 0.399 | 0 |
| `left_margin` | **0.731** | 0.713 | 0.679 | 0.632 | 0 |
| `right_margin` | **0.705** | 0.694 | 0.668 | 0.627 | 0 |

9 个目标中 8 个在 lag 0 最优（唯一例外 `relative_lat_velocity` 在 −0.3s 高 `0.018`），
因此 hidden 的编码对准的是**当前**几何，不是被时间平滑成过去某时刻的几何。控制 B 的最初
实现用全维 hidden，出现 `-1512`、`-4124` 一类数值崩溃，该版本作废。

### 32.4 判决与可支持的结论

逐条核对 §32.2 的预注册门：**(a) 部分满足**——`left_margin` 与 `right_margin` 在
pressure 输入上达到 `0.598/0.603`（全维口径 `0.605/0.633`），`right_margin` 在 raw 上也有
`0.537`；其余目标的输入侧在 `0.03--0.32` 之间，不满足。**(b) 在 9 个目标上全部不满足，且
符号相反**——没有任何目标出现 `hidden <= 输入 - 0.25`，fold-local 维度匹配后 hidden 一致
更高。(d)(e) 两个对照也都不支持缺口。**因此 §31.6 的条件未满足，不启动
`PPO + auxiliary representation loss` 训练。**

（2026-08-07 更正：本段初稿写成"(a) 在 9 个目标上都不满足"，与 §32.3 表中
`left_margin`/`right_margin` 的 pressure 数值矛盾，属于算术错误。判决不依赖 (a)——决定性的
是 (b) 全部不满足且符号相反。）

**可以支持：**

- 在 U44 上，`GRU hidden` 对当前交互几何的线性可解码性**不低于**、且在维度匹配读出下
  **一致高于** actor 自身的 361D 输入。用户假说中"信息可能存在于输入却没有稳定进入内部
  表征"的方向被本预检否定；把这些几何量作为辅助目标压进 hidden，缺乏可指望的增量。
- 冻结 BC pressure 在全维读出中 **9/9** 目标优于 raw LiDAR；在fold-local维度匹配的
  K=64与K=256读出中均为 **8/9** 优于raw，唯一例外是`relative_lateral`。frozen `k` 是比
  原始扫描更好的几何线性基，这与 §29 不解冻 `k` 的决定一致，且是独立证据。

**不能支持：**

- 不能说这些几何量被编码得"好"。fold-local维度匹配下，hidden的`R^2`在K=64为
  `0.158--0.671`、K=256为`0.272--0.731`，没有任何目标接近1。
- 不能说"辅助表征学习无效"。本节只否定了**该量族的表征缺口前提**；辅助损失若通过其他
  通路起作用（不同量族、或正则化/优化效应而非解码效应），不在本节射程内。
- 线性可解码性仍不等于信息含量（同 §23.5 rule 1、§31.2）。但这里的方向对 hidden 有利，
  所以"hidden 丢了信息"这一主张在**本可以支持它的同一测量口径下**没有成立。
- 结论只适用于 U44、单 seed 42、线性 ridge 读出与该 9 个目标；不外推到其他 checkpoint。

### 32.5 方法学记录与停止规则

1. **不得在未做维度匹配的情况下比较 361D 与 1680D 读出。** 本节的朴素全维结果曾给出
   hidden `R^2 = -2.133`、看似严重缺口；`lambda` 正确内层选择后变为 `-2.133`→ 单折崩溃，
   PCA 匹配后变为 `+0.555` 且优于输入。同一数据在三种读出设定下给出三种相反结论，
   后续任何 representation probe 必须同时报告维度匹配结果与 train/test 差。
2. 不启动 `CT-v2 + 辅助表征 loss` 训练；不因某个单目标的小幅差异挑点重跑。
3. **任何降维必须在 fold 内拟合。** 本节初版在全样本上拟合 PCA 后才做分组 CV，test sector
   参与了均值与主成分估计；虽然改成 fold-local 后结论未变（差异 `<0.01`），但这类泄漏在
   分组外推场景下不可预先假定无害，后续所有 probe 一律 fold-local 拟合并在文档中声明。
4. 若将来提出**不属于本节 9 个量族**的辅助目标，必须先按 §32.2 的同一口径（回放校验、
   分组切分、内层 `lambda` 选择、fold-local 维度匹配、滞后对照）重做缺口预检，不得继承
   本节结论，也不得跳过预检。
5. 本节与 §31 合起来的净结论是：走廊探索失败迁移的机制**仍未定位**。已被排除的解释包括
   稳定 output-head 梯度冲突（§24）、collision/ordinary role 配比错配（§25）、
   几何/速率的早期可分性（§31）以及当前几何的内部表征缺口（本节）。剩余候选必须先给出
   自己的最便宜准入门，再谈训练。

## 33. Round Z0：U42--U45等权checkpoint平均（2026-08-08，无训练）

### 33.1 判决表

正式单元格均为`ego collision / overtake`；near400是机制诊断，无独立验收权。

| actor | Austin600 | Hockenheim600 | MoscowRaceway600 | Nuerburgring600 | 四图合计 | near400 | 判决 |
|---|---:|---:|---:|---:|---:|---:|---|
| canonical BC | `33 / 339` | `27 / 343` | `43 / 373` | `26 / 390` | `129 / 1445` | — | 逐图最低线 |
| U44 | `18 / 344` | `16 / 347` | `15 / 390` | `13 / 397` | **`62 / 1478`** | `37 / 288` | 固定主对照 |
| U42--U45等权平均 | `18 / 342` | **`19 / 341`** | `16 / 386` | `14 / 396` | `67 / 1465` | `32 / 285` | **否决并关闭Round Z0** |

平均actor在Hockenheim的overtake `341 < 343`，未逐图满足canonical BC下限；四图相对U44又是
`+5 collision / -13 overtake`，不满足预注册的任一严格Pareto条件。Production保持U30，
平均actor不进入production，也不改变U44的四图安全候选地位。

### 33.2 问题、唯一操作与执行合同

本轮检验的不是新训练方法，而是§28/预注册§21提出的一个低成本假说：同一走廊训练轨迹上
U42--U45权重距离很小、但created collision身份波动很大，等权参数共识是否能抵消
checkpoint特异的临界失败。

调用链为：

```text
average_actor_checkpoints.py
  -> 读取固定U42/U43/U44/U45 actor state dict
  -> 各浮点tensor按该顺序转float64、等权求和/4、cast回原dtype
  -> 非浮点tensor要求四源逐值相等
  -> End2Race(hidden_scale=4) strict-load
evaluate_scenario_panel.py
  -> 四图固定600 + Austin near400，CUDA、ego scope、deterministic mean、8秒、保存trace
```

没有训练、optimizer、reward、actor结构或输入变化；band固定U42--U45，四点权重均为0.25，
没有查看结果后改成三点或非等权平均。输出仍为12-key actor，所有tensor有限，评估alias与
canonical输出是同inode硬链接。平均actor相对U42/U43/U44/U45的L2距离依次为
`1.0771e-4 / 6.1200e-5 / 6.1609e-5 / 1.1707e-4`；模型SHA只登记在HANDOFF §2。

四图场景均为当前50 circular starts × 3 opponent racelines × 4 speeds、interval15；每图600。
冻结场景的key和六个ScenarioSpec身份字段在评估前与现有BC、U44两侧逐项一致。near400复用
封存的400条输入，未重新按平均actor结果筛选。四图与near合计2,800条均通过：0 error、结果/
trace key集合相等、数值数组有限且逐行对齐、collision marker与outcome一致、末行唯一
`terminal_post_step=true`且`action_applied=false`，manifest明确CUDA与ego scope。

### 33.3 相对BC的配对结果

`removed / created`用于collision，`lost / gained`用于overtake；p均为同场景双侧exact McNemar。

| 地图 | collision | removed / created | p | overtake | lost / gained | p |
|---|---:|---:|---:|---:|---:|---:|
| Austin | `33→18` | `24 / 9` | `0.0135` | `339→342` | `12 / 15` | `0.701` |
| Hockenheim | `27→19` | `18 / 10` | `0.185` | `343→341` | `18 / 16` | `0.864` |
| MoscowRaceway | `43→16` | `36 / 9` | `6.57e-5` | `373→386` | `13 / 26` | `0.0533` |
| Nuerburgring | `26→14` | `16 / 4` | `0.0118` | `390→396` | `6 / 12` | `0.238` |
| 四图合计 | `129→67` | `94 / 32` | `2.91e-8` | `1445→1465` | `49 / 69` | `0.0798` |

平均actor仍显著减少BC碰撞，但它新造32次collision，多于U44的27次；Hockenheim的净
overtake下降直接触发逐图BC守门失败。碰撞子类合计为`62 ego-opponent / 5 ego-wall`，四图
opponent-wall event为`1 / 4 / 0 / 2`，未并入ego collision。

### 33.4 相对U44的配对结果与near400

| 地图 | collision | removed / created | p | overtake | lost / gained | p |
|---|---:|---:|---:|---:|---:|---:|
| Austin | `18→18` | `4 / 4` | `1.000` | `344→342` | `5 / 3` | `0.727` |
| Hockenheim | `16→19` | `2 / 5` | `0.453` | `347→341` | `7 / 1` | `0.0703` |
| MoscowRaceway | `15→16` | `5 / 6` | `1.000` | `390→386` | `8 / 4` | `0.388` |
| Nuerburgring | `13→14` | `2 / 3` | `1.000` | `397→396` | `4 / 3` | `1.000` |
| 四图合计 | `62→67` | `13 / 18` | `0.473` | `1478→1465` | `24 / 11` | **`0.0410`** |

near400为`37→32 collision`（removed/created `13/8`, `p=0.383`）和`288→285 overtake`
（lost/gained `15/12`, `p=0.701`）。因此平均并非所有维度都单调变差：它在这个U30筛选的诊断
panel上净少5次碰撞；但该变化不显著、同时损失3次超车，而且不能覆盖四图正式验收与
Hockenheim最低线失败。

### 33.5 机制结果、失败解释与证据边界

U42--U45相对BC共同created的core仍精确为14。平均actor相对BC共created 32次，其中与core
重合13次：覆盖平均created的`13/32 = 40.6%`、覆盖core的`13/14 = 92.9%`，Jaccard为`0.394`；
重合逐图为`4/3/5/1`。也就是说，等权平均几乎保留了全部稳定core，同时又产生19个core外
created collision。**机制假说未成立：权重平均没有把行为变成四个checkpoint失败集合的交集。**

不要把这条线读成“相邻checkpoint参数不兼容”：平均actor可严格加载、数值稳定，且结果仍
处于相邻模型的性能量级。失败发生在更具体的映射上——极小权重距离不保证闭环行为线性插值，
参数平均既不能选择各checkpoint的安全动作，也不能消除决策边界两侧的非核心失败。near400
净改善同时说明它不是简单的全局崩塌，而是失败身份再次迁移。

证据边界：固定单训练轨迹、固定一个等权band、CUDA确定性四图与一个诊断panel；没有多seed，
也没有测试其他平均权重。该边界是预注册要求，不构成继续扫权重的理由。

### 33.6 停止与重开规则

1. Round Z0关闭；不尝试U43--U45、三点平均、加权平均、EMA/SWA或根据四图结果选权重。
2. 不把near400的`-5 collision`单独写成接受证据；该panel没有验收权。
3. 平均actor和完整评估保留为负结果与模型身份资产，但不修改production alias。
4. 只有一个**预先独立提出、不是根据本轮结果调权重**的新参数组合理论，且具有独立于四图的
   训练侧选择依据，才可能重开checkpoint组合；当前预注册链转入BC Gate A--D。

## 34. BC-safe anchoring Gate A：Austin训练侧稳定锚定对象存在（2026-08-08，无训练）

### 34.1 判决

Gate A **通过**，但本节只证明development split中存在数量和分层均满足预注册门的稳定回归
cohort，不证明BC动作能从U44已访问状态救回终局。按用户要求，本轮完成并记录Gate A后停止，
尚未实现或启动Gate B反事实接管。

| 预注册条件 | 实测 | 判决 |
|---|---:|---|
| 共识cohort `>=20` | **28** | 通过 |
| unique ego startpoint `>=10` | **21** | 通过 |
| raceline0 / raceline2各`>=3` | **20 / 8** | 通过 |
| 共识cohort中U44 ego collision `>=8` | **19** | 通过 |
| BC、U42--U45轨迹/identity/terminal完整有限 | **3,590 / 3,590 actor-episode** | 通过 |
| U44单点`>=20`但共识`<20`则失败 | 单点 **46**、共识 **28** | 通过 |

Gate B要求两类回归都至少8条；当前固定分层为`C=19`、`L=9`，也满足继续做双分层反事实
准入测试的样本数前提。这不是Gate B本身的通过结果。

### 34.2 先冻结split，再做fresh replay

输入是Austin训练侧困难面板的868个scenario、91个unique `ego_idx`。任何actor评估之前，严格按
`SHA256("bc-anchor-v1|Austin|" + str(ego_idx))`排序并冻结起点级split：development为
718个scenario/72个起点，validation为150个scenario/19个起点；两侧起点和scenario key均零
交集。development panel SHA-256为
`61ccc0652a0f7cfafa142f88202bf235da1d7a6f5c7c3a06c3bc6e9c8c1e200e`。本节只评估development，
没有对validation调用actor evaluator，也没有用validation选择cohort、阈值或stratum。

五个actor均为fresh deterministic CUDA、8秒、`collision_scope=ego`、保存numeric trace：

| actor | ego collision | ego-opp / ego-wall | overtake | follow |
|---|---:|---:|---:|---:|
| canonical BC | 308 | `246 / 62` | 318 | 92 |
| U42 | 191 | `139 / 52` | 377 | 150 |
| U43 | 187 | `136 / 51` | 385 | 146 |
| U44 | 214 | `161 / 53` | 373 | 131 |
| U45 | 202 | `149 / 53` | 377 | 139 |

这些aggregate只描述整个困难development面板，不能替代下述逐episode cohort定义。五臂各718个
result与718个NPZ严格同key；共3,590条trace全部检查了所有数组首维对齐和有限值、严格递增
`time_s`、`steps+1`长度、collision typed marker与outcome、末行唯一
`terminal_post_step=true`且`action_applied=false`。所有result identity与冻结panel的
`scenario_id/ego_idx/opp_idx/raceline/speed/interval/map`逐项一致，0 error、无partial结果。

### 34.3 单点回归与多checkpoint共识

先只看BC，得到308条`BC outcome=overtake`且opponent为raceline0/2的BC-safe overtake；然后才
读取U42--U45。按预注册，U44必须回归，且四个checkpoint至少3个回归：

| 回归checkpoint数（BC-safe 308条内） | 0 | 1 | 2 | 3 | 4 |
|---:|---:|---:|---:|---:|---:|
| episode数 | 231 | 21 | 20 | 17 | 19 |

U44单点回归为46条，多checkpoint共识保留28条，即单点集合中18条（39.1%）未通过稳定性过滤；
这与§1.1的“相邻checkpoint存在大量临界身份翻转”一致，但剩余共识仍超过全部Gate A数量门，
所以不能把train-side回归整体解释为单点瞬态。

固定28条共识cohort的完整分层为：

| 分层 | episode数 |
|---|---:|
| raceline0 / raceline2 | `20 / 8` |
| U44 collision / lost-overtake | `19 / 9` |
| speed 0.45 / 0.50 / 0.55 / 0.60 / 0.65 | `1 / 1 / 0 / 3 / 2` |
| speed 0.70 / 0.75 / 0.80 / 0.85 | `2 / 5 / 7 / 7` |

19条collision stratum全为U44 `ego-opp`，没有`ego-wall`；9条lost-overtake全为安全`follow`。
全部speed均按原始筛选结果报告并保留，0.55层的实测共识数为0，不是事后删除。冻结cohort panel含28条scenario、21个起点，
SHA-256为`ffacda43c4bdf397173dfaf9c23795f165f52401fa9a5eb1e75409fb46e0afc1`；后续Gate B只能消费
这份development输入，不得回到U44单点46条或从validation补样本。

### 34.4 机制结果、证据边界与下一停止点

正面机制结果是：**Austin困难训练侧确实存在跨相邻晚期checkpoint重复出现的“BC原本安全超车、
U44发生碰撞或丢超车”对象**，且两条raceline、21个起点和两种方向相反的回归都获得最低覆盖。
因此BC functional regularization不是在空cohort上成立的纯概念，Gate B有合法输入。

仍不能据此推出BC是可用teacher。BC从episode起点成功与“在U44 hidden/observation prefix上
接管1.5秒仍能救回”是不同命题；`C=19`要求更保守，`L=9`要求恢复进取性，同一个二维BC loss
可能无法同时满足。Gate B必须先让no-intervention branch逐行复现保存的U44 trace，再分别检查
collision救回、救回后超车保有、lost-overtake恢复和matched safe-control副作用。任何一层失败
都关闭方向，不能用另一层抵消。

证据只覆盖一个Austin训练侧困难分布、同一次seed-42走廊训练轨迹的U42--U45和canonical BC；
它没有验证自然训练分布、四图泛化、多seed、反事实动作有效性或最终formal训练。validation仍
封存。下一节点只能是预注册Gate B，且本轮没有启动它。

## 35. BC-safe anchoring Gate B：BC接管只救碰撞、不救丢超车（2026-08-08，无训练）

### 35.1 判决表

每格按固定stratum报告最终outcome；`C`为19条U44 ego collision，`L`为9条U44安全follow，
control为28条BC/U44均安全overtake。branch 2/3只作解释，不具有事后替换正式方法的资格。

| branch | C：overtake / collision | L：overtake / follow / collision | control：overtake / follow / collision | 判决 |
|---|---:|---:|---:|---|
| 0：全程U44 no-op replay | `0 / 19` | `0 / 9 / 0` | `28 / 0 / 0` | **56/56精确复现，通过实现门** |
| 1：完整BC steering+speed | **`10 / 9`** | **`0 / 9 / 0`** | **`26 / 2 / 0`** | **Gate B失败，关闭BC anchoring** |
| 2：仅BC steering | `7 / 12` | `1 / 8 / 0` | `28 / 0 / 0` | 诊断；不能替代branch 1 |
| 3：仅BC speed | `5 / 14` | `1 / 8 / 0` | `26 / 1 / 1` | 诊断；不能替代branch 1 |

完整BC分支在collision stratum恰好通过全部门：救回`10/19 >= ceil(0.5*19)=10`，10条全部仍为
overtake，覆盖9个起点，raceline0/2救回`8/2`，也满足每线至少2条。但lost-overtake要求恢复
`ceil(0.8*9)=8`，实测为**0/9**；虽然没有新造collision，恢复数、起点覆盖和raceline覆盖三门
同时失败。safe controls允许最多1条overtake损失，实测为**2/28**，也独立失败。因此11条
准入判据只通过7条，Gate B科学失败；不构造anchor dataset，不进入Gate C/D或formal训练。

### 35.2 固定计划、调用链与高效执行合同

Gate A冻结的28条cohort全部满足窗口内至少50个`action_applied=true`步，短窗口剔除数为0。
另从development中BC/U44都安全overtake的262条候选里，按相同raceline与speed、circular
ego-index距离、scenario-key SHA tie-break无放回匹配28条controls。窗口在任何branch replay
前一次冻结：collision用首次ego碰撞前1.5秒；L/control用U44 trace上首次全局最小OBB clearance
前1.0秒到后0.5秒，terminal行不执行动作。43/56条窗口为150步，其余因episode边界固定截断为
85--138步，全部仍高于50步。

调用链为：

```text
run_bc_anchor_gate_b.py --prepare-only
  -> 冻结cohort窗口、matched controls与全部输入身份
run_bc_anchor_gate_b.py（CUDA、12 forkserver workers）
  -> 每个persistent worker只加载一次canonical BC与U44
  -> 56条branch 0 -> 父进程逐字段硬校验 -> 通过后才运行3×56条干预
  -> 全224条result/trace/action-source合同 -> Gate B双分层判决
```

效率改动只发生在执行层：没有减少分支或trace字段，也没有更换窗口、模型或准入线。与Gate A
runner逐scenario加载actor不同，Gate B每个worker只加载一次两份actor并连续消费任务；四个
stage都支持`episodes.partial.jsonl`按episode key恢复，完成后才原子写正式result。

### 35.3 实现与数据质量门

正式branch 0的56条（28 cohort + 28 controls）相对保存U44 trace全部通过。所有
`action_applied=true`行的U44 raw/executed action最大绝对误差均为0；opponent action、ego/opp
pose、measured speed与双360D LiDAR的全局最大绝对误差同样均为0，不只是低于`1e-6`。
outcome、首次ego collision类型/step、episode长度、collision marker与terminal row逐条相同。
因此后续差异来自已声明的动作替换，不是branch runner偏差。

四个branch共224个episode/224条NPZ全部通过：result/trace/plan key集合相等，所有数组逐行对齐
且有限，collision marker与outcome一致，末行唯一terminal、动作源code与冻结窗口逐步一致，
selected raw action及steering clip后的executed action逐位符合分支合同。完整BC、steering-only、
speed-only分别有6/5/7条因干预后较早碰撞而未执行满计划窗口；这些是branch outcome的一部分，
不是静默剔除或数据缺失。

### 35.4 机制结果

不要把本轮简写成“BC动作无效”。在完全相同的U44状态prefix上，固定1.5秒完整BC接管确实把
10条ego-opponent collision变成了10条overtake，超过steering-only的7条和speed-only的5条；
其中完整分支独有的救回说明纵横向组合存在非线性协同。**BC提供了局部有效的碰撞修复动作。**

但它不是当前所需的统一功能锚：

1. 9条L中完整BC接管全部仍为follow，两个单分量也各只恢复1条。干预窗口内BC与U44动作并非
   相同：L逐episode平均绝对动作差的中位数为steering `0.00690 rad`、speed `0.0531 m/s`，
   全局最大差为`0.0313 rad / 0.2276 m/s`。0/9不能解释成teacher没有实际介入；更符合证据的
   边界判断是：在U44已访问状态和固定短窗口上，这些BC动作不足以恢复超车。
2. 两条原本BC/U44都安全超车的control被完整BC接管变成follow，已超过预注册1条上限；两条
   都在steering-only下保持overtake，其中一条在speed-only下变follow，另一条只有完整二维组合
   才变follow。这表明副作用不只来自碰撞层，而是局部替换本身会改变progress结果。
3. C与L要求方向相反：前者需要安全修复，后者需要恢复进取性。当前完整BC分支只对C有强机制
   作用。若把这两类序列以固定0.5/0.5加入同一个对称BC loss，现有证据不支持它能同时保护安全
   和超车，反而已有control overtake损失。

### 35.5 失败原因、证据边界与停止规则

失败发生在teacher的**状态条件可迁移性**，不是branch实现或样本不存在：Gate A证明稳定对象
存在，branch 0证明runner精确，branch 1证明BC在C上能因果救援；但同一固定teacher/window在L
和safe controls上没有满足progress保持。BC从episode起点自然运行成功，仍不能推出其动作适合
U44已经形成的hidden、相对姿态与速度状态。

证据只覆盖Austin development困难分布、单seed-42 U44、固定1.5秒窗口和canonical BC；它不
证明BC在所有状态无效，也不比较其他teacher或更长接管。但这些都是当前预注册方法的固定合同，
不能因为失败而现场改成steering-only、speed-only、换窗口或删掉L。validation没有运行branch，
四图测试没有进入cohort或判定。

停止规则：

1. 当前hindsight-selected counterfactual BC-branch regularization方向关闭；不生成anchor
   sequence dataset，不重建Gate C梯度分解，不做Gate D或45-update formal训练；
2. 不把C层`10/19`单独当作准入，也不删除L或safe-control门；
3. 不以steering-only control零损失事后改写正式branch；其C仅`7/19`、L仅`1/9`，本身也失败；
4. 若复用已通过的branch engine研究动作库/first-action preference，必须独立预注册动作集合、
   target选择、剂量和准入线，不能在本文件内把BC target现场换掉；
5. 只有任务目标明确改为“只修复该collision cohort且允许单独的安全专用策略”，才可把C层
   positive mechanism作为新问题的起点；当前单actor安全+超车目标下不重开。

## 36. Round Z2：反事实动作存在，但frozen-hidden不能稳定选择（2026-08-08，无actor更新）

### 36.1 Verdict table

单位均为最终无ego collision的overtake；head为按ego startpoint分组的五折out-of-fold结果。

| 层 | 分母 | early oracle | late oracle | early head | late head |
|---|---:|---:|---:|---:|---:|
| inherited collision | 109 | `93` | `96` | `19` | `34` |
| created collision | 46 | `44` | `45` | `11` | `17` |
| lost-overtake | 13 | `10` | `7` | **`1`** | **`0`** |
| inherited-follow诊断 | 63 | `18` | `18` | `5` | `3` |
| safe controls新collision / overtake loss | 225 | oracle不介入 | oracle不介入 | **`15 / 17`** | `3 / 5` |
| 三主层target success | 168 | `147` | `148` | `31` | `51` |
| grouped fixed-action baseline | 168 | `0`（noop） | `79` | head `+31` | **head `-28`** |

**判决：** early/late action-existence均通过，但两个rankability Gate均失败。关闭当前固定12动作库
的action-conditioned scorer与first-action preference extraction；prefix-reset未取得early-only
前提，不准入；Residual/MoE仍违反12-key兼容。Constrained PPO不在本实验的逻辑覆盖内。

### 36.2 问题、固定设计与调用链

Gate B只检验BC teacher，不能回答同一prefix上其他小动作是否有效。本轮扩大到Gate A Austin
development中的109 inherited collision、46 created collision、13 lost-overtake、63
inherited-follow及225 BC/U44共同安全overtake；全部要求U44事件前至少150步。事件为首次ego
collision或第一次全局最小OBB clearance，不读取validation或测试地图。

```text
run_counterfactual_action_gate.py --prepare-only
-> 冻结456条、startpoint五折、early/late与12个residual
-> 456条U44 no-op精确重放
-> 20个persistent CUDA worker运行10,944条candidate branch
-> 全trace合同
-> oracle existence
-> frozen-hidden ActionScorer五折外推
```

early/late分别为`[event-150,event-100)`与`[event-100,event-50)`；动作相对每一步当前U44 mean，
包含steering `+/-0.02,+/-0.04 rad`、speed `+/-0.5,+/-1.0 m/s`及四个联合动作。窗口结束恢复
U44，碰撞按环境真实提前terminal。动作库、持续时间、阈值和head在branch前写死。

### 36.3 质量与机制结果

456/456 no-op的ego raw/executed、opponent action、两车pose/speed、双360D LiDAR及boolean字段
全局最大误差均为0。10,944条candidate加456条no-op共11,400条trace、8,185,913行；result、
plan、trace key相等，数组finite/对齐，terminal、干预窗口、residual raw和clip后executed action
全部通过，无partial。

不要把否决读成“动作空间没有解”。两个prefix的六条existence门全部通过：early/late分别救回
collision `137/155`与`141/155`，lost-overtake `10/13`与`7/13`；collision救回覆盖48个起点、
raceline0/1/2以及steering、speed、coordinated三类。当前动作接口在future-event定位的关键窗口
具有很高的局部可达性。

更强的机制事实是，late每折train数据都选择同一个固定`steer +0.02 rad / speed +0.5 m/s`；它在
OOF场景上取得79次三主层success，而state-conditioned head只有51次。固定动作相对no-op的
collision removed/created=`79/17`（exact `p=9.94e-11`），overtake lost/gained=`5/83`
（`p=2.69e-19`）。这不构成可部署方案：动作窗口由future event定位，且它仍新造5次control
collision并只恢复`1/13` lost-overtake；但它直接反驳“缺的是更大的局部动作库覆盖”。

### 36.4 Acceptance失败与机制解释

early head在三主层为`19/109、11/46、1/13`，controls新collision/loss=`15/17`；相对no-op
collision removed/created=`41/19`（`p=0.00622`），overtake lost/gained=`17/36`
（`p=0.0127`）。它有净改善，但安全正对照和progress层独立失败。

late head为`34/109、17/46、0/13`，controls=`3/5`；相对no-op为collision `56/7`
（`p=1.36e-10`）、overtake `5/54`（`p=1.91e-11`）。它满足control界，却完全没有恢复
lost-overtake，并比固定动作baseline少28次target success，未证明hidden带来状态条件选择价值。

因此当前瓶颈被进一步收窄为：**不是局部动作不存在，而是future-event窗口无法部署、hidden到
动作类别的跨startpoint排序不稳定，以及collision动作与lost-overtake动作没有统一选择规则。**
反事实信息给出了强oracle，却没有自动变成可部署策略信号。

### 36.5 证据边界与停止规则

1. 这是Austin development、单U44、固定12动作与两个50-step prefix的零actor-update诊断；
   不能外推为连续动作空间、不同history结构或正式PPO的理论不可能；
2. prefix由未来collision/min-clearance定义，所有paired净改善都不是部署actor成绩；
3. 不改动作幅度、窗口、head宽度、fold或准入线重试，不用oracle数字绕过rankability；
4. 关闭当前fixed-library action-conditioned和first-action preference；
5. prefix-reset只有在出现“早期有可排序动作而晚期没有”或fresh rollout直接建立长credit错配时
   才能重开，本轮两侧existence均强、两侧rankability均失败，不满足；
6. Residual/MoE只有用户明确放弃12-key兼容并先证明shared mapping不足时才能重开；
7. Constrained PPO不使用本Gate动作标签，因此保持未测试，不能被本结果写成失败。

## 37. Round Z3：collision-only BC anchoring 独立validation Gate（2026-08-08，无branch/无训练）

### 37.1 判决表

| 阶段 | 实测 | 预注册条件 | 判决 |
|---|---:|---:|---|
| validation actor screen | BC/U42/U43/U44/U45各150条，750 result/trace全部合同通过 | 完整、finite、identity和trace一致 | 通过 |
| 稳定collision cohort | 7条，6起点，raceline0/2=`6/1` | 至少4条、3起点、两条raceline均非空 | 通过 |
| matched-control support | `r0/s0.60: 2 source / 0 control`；`r0/s0.85: 3/2` | 同raceline、同speed、无放回、与cohort等数 | **失败；无可行匹配** |
| branch0 / full BC | 0 / 0 | controls先冻结后才可运行 | **未运行** |
| 科学判决 | 没有teacher outcome | 样本/对照无法闭合时只能isonclusive | **inconclusive；不否决collision-only** |

五个actor在该150条上的`ego collision / overtake / follow`为BC `67/60/23`、U42
`38/67/45`、U43 `42/67/41`、U44 `49/61/40`、U45 `43/64/43`。这些数只用于
按冻结规则定cohort与control pool，不是candidate actor验收，因此不对它们作新的
production优劣结论。

### 37.2 问题、固定设计与调用链

Gate B的C层在development上有`10/19`救援且`10/10`最终超车，但原双分层方法
因L层`0/9`和controls丢`2/28`关闭。Round Z3只问一个更窄的问题：删除L后，
同一canonical BC teacher的collision救援能否在从未打开的validation startpoint上复现，
同时不误伤匹配安全超车。这是新变体，不翻转原Gate B判决。

```text
evaluate_scenario_panel.py x BC,U42,U43,U44,U45
-> analyze_bc_anchor_gate_a.py --split-name validation --collision-only-validation
-> 冻结stable U44 collision cohort
-> run_bc_anchor_gate_b.py --prepare-only --collision-only-validation
-> matched-control support不足，在branch0前fail closed
```

固定panel是150条Austin validation ScenarioSpec、19个ego startpoint，与718条development的72个
startpoint零交集。`BC-safe overtake`、U42--U45至少3/4回归、U44 ego collision、150-step
窗口及control匹配均在任何validation outcome可见前写入预注册。

### 37.3 机制结果与数据质量

750条评估的result/panel/trace key完全一致，scenario identity全字段相等，数组finite且
逐行对齐，collision marker和terminal row合同通过，没有partial或error。BC-safe overtake
有59条；U44单点回归16条；按四checkpoint至少3次回归且U44为collision后留下7条。

这7条覆盖6个独立起点，速度`0.60/0.70/0.75/0.85=2/1/1/3`。cohort存在性的
必要条件成立，但**teacher救援机制没有被测量**：没有branch0、full-BC或干预后
outcome。不能把“7条cohort存在”写成collision-only teacher可复现。

### 37.4 为什么停止

预注册要求每条source在同一validation panel内匹配同opponent raceline、同speed scale的
BC/U44共同安全overtake，且无放回。source分层为`r0/s0.60=2`、`r0/s0.70=1`、
`r0/s0.85=3`、`r2/s0.75=1`；安全对照在前两个缺口分层分别为0和2。因此即使
改用全局最优匹配也无解，不是贪心顺序问题。

放宽speed、改成有放回、从development补control或减少source，都会在看到validation分层后
改变方法。故runner在冻结plan前fail closed，没有消耗branch计算，也没有制造一个伪
teacher判决。

### 37.5 停止与重开规则

1. 本轮只标记`inconclusive / matched-control support insufficient`；不得写成collision-only
   通过或失败；
2. 不在当前validation panel上改speed/raceline匹配、不放回要求或剔除难匹配source后
   重跑；
3. 不生成collision-only anchor dataset，不进入shadow beta或formal PPO；
4. 只有一个在任何actor outcome前冻结、与development/当前validation起点独立，并按
   source raceline/speed预先保证control support的新Austin panel，才能重开；
5. 即使新Gate通过，也只能起草独立formal-training预注册，不自动授权训练。

耐清理的核心结论：Round Z3在750条完整actor replay中找到7条、6起点的独立稳定
collision cohort，但冻结matched controls在`r0/s0.60`与`r0/s0.85`分层确定性无解；
没有运行任何branch或训练，因此不得对collision-only teacher作科学成败判决。

## 38. Round Z4-A：50步actor-visible历史没有产生增量动作选择价值（2026-08-08，无actor更新）

### 38.1 Verdict table

分母依次为inherited collision 109、created collision 46、lost-overtake 13和safe controls 225；
数值全部是按ego startpoint分组五折的out-of-fold闭环终局。

| selector | inherited / created / lost恢复为overtake | 三目标层total | safe-control新collision / overtake loss | 判决 |
|---|---:|---:|---:|---|
| grouped fixed action | `45 / 33 / 1` | `79` | `5 / 5` | 状态无关基线 |
| frozen-hidden outcome control | `68 / 33 / 3` | **`104`** | `13 / 20` | 不单独准入；共同监督对照 |
| 50-step history treatment | **`68 / 32 / 2`** | **`102`** | **`13 / 21`** | **Z4-A失败；不进入validation/PPO** |

treatment只通过两个collision层最低数和相对fixed baseline `+23`；它失败lost-overtake
`2 < 4`、control collision `13 > 4`、control overtake loss `21 > 11`，并以
`102 - 104 = -2`失败预注册的相对frozen-hidden `+9`增量门。

### 38.2 问题、单变量设计与调用链

Round Z2直接检验的是冻结U44 hidden，不能给“训练改变表征”的方法作理论上限。Round Z4-A
因此把问题收窄成一个可廉价证伪的必要条件：若额外编码动作前最近50步actor可见历史仍不能
改善动作选择，就没有理由先付出共享student GRU与PPO耦合训练的成本。

本轮不重跑模拟；复用Round Z2已经通过全trace合同的456条branch0和late 12-action结果：

```text
456条task x (noop + 12 residual actions)
-> 5,928个真实终局collision/follow/overtake标签
-> frozen U44 sigmoid LiDAR transform + frozen speed_mlp
-> 50 x 420 actor-visible history
-> ego-startpoint grouped 5-fold control/treatment训练
-> 固定P(overtake)-5*P(collision)选择动作
-> 对真实branch outcome作OOF行为判决
```

共同输入是late时刻保存的1680D U44 hidden。control为`1680->192`后拼2D动作并经64D head输出
三分类；treatment把hidden压到128D，同时用`420->64->GRU64`编码50步历史，再拼2D动作走
相同64D head。两者都用Adam `1e-3`、weight decay `1e-4`、100 epochs、scenario batch 64、
训练折三类权重，无early stop或hyperparameter sweep。hidden与420D feature都只用训练折
均值/方差标准化。固定动作也只在各fold的训练startpoint上按
`non-control overtake - 5*safe-control harm`选择；五折均重新得到Round Z2相同的
`steer +0.02 rad / speed +0.5 m/s`。

### 38.3 数据质量与机制结果

456个task覆盖70个ego startpoint，五折test task数为`76/100/65/102/113`、test startpoint数
为`9/18/11/16/16`；hash fold重算一致且startpoint集合两两不交。输入张量为
hidden `456x1680`、history `456x50x420`，共22,800个历史time rows；全部finite。构造器只读取
`ego_lidar_360`和前一时刻`ego_measured_speed_mps`，禁止的两车pose、opponent状态、stratum、
距event距离和branch outcome均未进入feature。13动作标签三类总数为collision/follow/overtake
`1642/1079/3207`，每折训练集三类均非空。

三分类OOF accuracy从frozen-hidden的`64.49%`升到history的`65.92%`，所以不是模型完全没有
拟合标签；但更高平均分类准确率没有转化为目标行为。两selector在456条中选择不同动作162条、
产生不同终局52条；history相对control新增19条overtake，同时丢掉20条control已有overtake。
分层净结果是inherited持平68、created少1、lost少1、safe-control overtake少1。

history相对noop仍是一个强但不可部署的hindsight窗口干预：collision removed/created=
`117/25`（paired exact `p=1.99e-15`），overtake gained/lost=`108/21`
（`p=2.67e-15`）。这说明动作库和future-event窗口有广泛局部作用，不代表selector满足安全
正对照或progress恢复，更不能当作actor成绩。

### 38.4 为什么necessary condition失败

若50步历史提供了冻结hidden缺失、且可跨startpoint泛化的动作响应信息，treatment至少应在
相同标签、fold、优化预算和action head下取得明确增量。实测target `-2`、control overtake
再多损1条，lost-overtake也从3降到2；因此本实例没有建立“新增history representation使动作
类别选择更好”这个必要条件。

最符合证据的解释是：标签中确有可预测的总体outcome结构，所以两个模型都远高于旧top-1
preference head；但固定future-event窗口里collision救援与safe/progress保持仍不是同一个
稳定决策规则。类别均衡loss和全动作平均accuracy可以提高，却不能保证概率差
`P(overtake)-5P(collision)`在少数关键scenario上具有正确排序。这是对当前固定实例的机制解释，
不是对任意辅助loss或GRU容量的理论结论。

### 38.5 证据边界、停止与重开规则

1. 科学关闭范围严格是`late 50-step actor-visible history encoder × fixed 12-action library ×
   class-weighted 3-class outcome target × fixed probability score`；
2. 不运行Z4-B validation candidate branches，不在全部development上另训模型，不接student
   GRU，不启动PPO；
3. 不用history长度、hidden宽度、loss、score或动作库的事后扫描重开本实例；
4. 不能由此否决会直接更新student GRU、使用不同因果监督或不靠future-event定位的全部
   representation-only方法；新变体必须先给出自己的必要条件和独立停止线；
5. collision-only BC anchoring的Round Z3仍是control support不足导致的inconclusive，本轮与
   canonical BC teacher无关，不能跨机制替它作结论；
6. prefix-reset的reset训练机制、Constrained PPO和MoE架构本身都不使用本Gate的历史head
   增量判据，本轮不能替它们作科学结论；MoE仅因当前12-key兼容合同而工程不准入。

耐清理的核心结论：在5,928个真实分支标签和严格startpoint折外条件下，增加50步actor-visible
历史只把outcome accuracy提高1.43个百分点，却使目标成功从frozen-hidden的104降到102，并
在225条safe controls上产生13次collision和21次overtake loss；该具体representation实例的
必要条件被直接推翻。

## 39. Round Z5：budget-constrained后未建立frozen-hidden优势（2026-08-08，零新仿真）

### 39.1 Verdict table

三目标层分母仍为inherited collision 109、created collision 46、lost-overtake 13；control为
225条BC/U44共同安全overtake。`collision/loss`是safe controls中新collision和overtake loss。

| selector | inherited / created / lost | target | control collision / loss | 科学判决 |
|---|---:|---:|---:|---|
| grouped fixed action | `45 / 33 / 1` | **79** | `5 / 5` | matched baseline |
| frozen hidden，independent outer seeds | `42 / 25 / 2` | **69** | **`1 / 3`** | harm通过，target低10 |
| frozen hidden，exact Z4 outer seeds | `42 / 23 / 1` | **66** | **`3 / 4`** | harm通过，target低13 |
| history，independent outer seeds | `40 / 23 / 0` | 63 | `6 / 8` | 两类门均失败 |
| history，exact Z4 outer seeds | `40 / 15 / 0` | 55 | `7 / 7` | 两类门均失败 |

**判决：** 两个frozen-hidden seed合同都在control harm不高于fixed时低于target 79，关闭当前
`frozen hidden × 3-class outcome × nested lambda/tau calibration × late fixed 12 actions`
实例。Round Z2的具体top-1 extraction继续关闭；不能再用Z4未约束的104宣称条件选择优于fixed。

### 39.2 为什么必须新增本Gate

Z2 late head以约47% noop得到target 51、controls `3/5`；Z4-A frozen outcome head只noop
7/456，得到104但controls恶化到`13/20`。两点同时改变了干预率和harm，不能说明hidden条件化
价值为正或负。本轮不新增branch，只重用5,928个真实action-outcome，把操作点校准从outer-test
严格隔离。

外层沿用70个ego startpoint的冻结5 folds。每个outer-train内部用独立hash分4个inner folds，
inner-OOF穷举11个`lambda`与全部可实现noop margin阈值；只保留control collision/loss率均不
高于`5/225`的点，再按target、harm、干预数和冻结tie-break选一个。随后在完整outer-train重训
模型，把选定操作点只应用一次到outer-test。

首次用独立outer seeds；由于这不是原Z4模型的精确扫描，看到结果后先冻结复核判读，再只把
outer seed恢复为Z4的`5200+fold / 5300+fold`。inner split、inner模型、网格、预算和tie-break
完全不变；两份结果都保留，不择优。

### 39.3 质量、配对与诊断曲线

两次均完成5个outer及其各4个inner folds；456 scenario、70 startpoints、13动作、5,928标签、
22,800历史行全部finite，预测无缺口，fold没有startpoint泄漏。默认Z4模式另在临时输出中逐项
精确复现原报告的frozen `104 @ 13/20`、history `102 @ 13/21`和fixed `79 @ 5/5`，证明扩展
没有改变旧路径。

independent frozen相对fixed在三目标层为`28 left-only / 38 fixed-only`，exact `p=0.268`；
exact-Z4-seed为`29/42`，`p=0.154`。controls上前者实际更保守到`1/3`，后者为`3/4`，因此
失败不是预算略超，而是在预算内缺少target。

全OOF经验Pareto只作诊断：

- independent-seed若看完全部outer outcome再选同一个全局点，可得到target 84、control `3/5`；
- 同一概率经过无泄漏inner选择、outer一次评估只有69、`1/3`；
- exact Z4 seeds即使事后全局选点，在`<=5/5`下最高也只有72、`4/5`。

第一组`84 -> 69`直接量化了用同一OOF outcome选阈值再报性能的乐观偏差；它不能作为方法通过
证据。原Z4的104则是激进操作点的另一端，不是matched-harm结果。

### 39.4 机制判断

Z4已经证明outcome监督能从hidden提取大量碰撞救援信息；Round Z5进一步证明，当selector必须
把safe-control误伤压到fixed水平，这些高分动作的跨startpoint排序不能保留足够target。主要
瓶颈因此不是“hidden完全没有信息”，而是**对安全状态的置信度校准与目标动作排序不能同时
泛化**。这比Z2单一`51 < 79`结论更准确，也解释了为何激进操作点能达到104。

显式50步history在两次nested结果中分别只有63和55，且controls `6/8、7/7`，没有改善这一
前沿；它与§38“history分类准确率略升但选择效用不升”的负结果一致。

### 39.5 证据边界与停止规则

1. 严格关闭当前late future-event窗口、固定12动作、frozen U44 hidden、三分类loss和nested
   calibration实例；不改lambda grid、预算、tie-break或seed继续扫描；
2. Round Z2 top-1 preference extraction同样保持关闭，但不能把两者外推为hidden的信息论上限、
   连续动作或不同因果监督理论无效；
3. 不运行独立validation candidate、不接actor、不训练PPO；
4. 98.5%干预率只发生在future-event定位后的窗口内，本Gate仍没有解决部署trigger；
5. prefix-reset机制与selector不同，仍科学未测。其下一道必要条件是snapshot恢复，而非直接
   formal training；
6. snapshot必须覆盖F110物理/延迟/RNG、lap与collision状态、LatticePlanner与tracker、PPO
   wrapper/reward状态、actor/critic recurrent hidden及truncation bootstrap。机械no-op复现后，
   还必须审计actor更新后如何从保存observation prefix用当前网络无梯度burn-in重算hidden。

耐清理的核心结论：在与fixed相同或更低的control harm下，frozen selector两次nested target为
69和66，均低于79；原Z4的104不能跨操作点解释为安全条件选择价值，当前tested形式关闭。

### 39.6 统计口径追加校正

上述“低10/13”是点估计，不是可检出的劣势。Target配对`28/38, p=0.268`与
`29/42, p=0.154`都只能支持“未检出严格优于fixed”。66个discordant pair在双侧
`p<0.05`下至少约需`42/24`分裂；真实paired净优势若为10，二项近似功效约19%，且70个
startpoint聚类会进一步降低有效样本量。

Nested过程约束的是两项outer泛化harm率`<=5/225`，外层实测为`1/3`和`3/4`；它们比fixed
`5/5`更保守，因此本轮是**budget-constrained**而不是“恰好harm matched”。不读取outer
outcome就无法强迫有限样本恰好用满预算；事后frontier使用outer outcome，不能补成无偏
`5/5`估计。两套事后frontier的`84 @ 3/5`与`72 @ 4/5`显示material seed sensitivity，但两个
seed不足以估计稳定方差。

程序关闭仍成立，因为预注册主条件是`target > 79`，两次都未达到；科学结论应写成“当前样本
与预算合同下未建立条件化增益”，不是“证明frozen hidden劣于fixed”。50步history在Z4-A和
两次nested中均未超过frozen且harm更差，可关闭该精确treatment；三轮复用同一456-task数据，
不得称为三套独立样本复现。会把辅助loss反传进student GRU并改变actor表征的2b一次都未测，
不能被本节连带关闭。

## 40. Round Z6-A：prefix-reset snapshot no-op工程门（2026-08-08，无actor更新）

### 40.1 Verdict table

单位是一条冻结Austin development任务；所有比较均为同一任务、同一U44 actor/critic、同一
snapshot前后缀的逐步一一配对。

| 层 | 任务数 | snapshot前缀 | 原后缀 vs 恢复后缀 | 判决 |
|---|---:|---:|---:|---|
| stable collision | 19 | Gate B冻结window start | 全字段最大误差0 | 通过 |
| stable lost-overtake | 9 | Gate B冻结window start | 全字段最大误差0 | 通过 |
| 合计 | **28，21 startpoints** | 26条`>0`；中位345.5步 | **28/28逐位通过** | **pass snapshot mechanical Gate** |

没有训练、validation、测试地图或新checkpoint。表中`19/9`是Gate A对U44的来源分层，不是本轮
环境重放的终局计数；本轮原后缀终局是14次ego collision、6次overtake、8次follow。这些不是
方法性能指标，只说明恢复比较覆盖了三类终局，Z6-A不重新裁决U44行为质量。

### 40.2 问题与真实调用链

Prefix-reset只有在无需从场景起点重新仿真时，才会真正提高单位算力内关键交互窗口密度。当前
production向量环境的`state_dict()`只保存scenario scheduler，不能恢复单环境物理、opponent
planner、reward或recurrent state。因此先实现独立、零训练的机械必要条件Gate：

```text
run_prefix_reset_snapshot_gate.py
-> ppo.env.make_environment(privileged=True, corridor_temporal)
-> End2RaceGymnasiumEnv
-> F110Env / Simulator / RaceCar
-> LatticePlannerOpponentController / LatticePlanner / PurePursuitPlanner
-> PPOTransitionReward
-> U44 End2Race actor + U44 PrivilegeGRUCritic
```

Runner不修改production模块，通过显式捕获/恢复当前对象的可变状态来检验实现可行性。每条在
冻结`window.start_index`、当前381D observation被actor/critic消费前保存；snapshot经过pickle
序列化/反序列化后，先跑原后缀，再在同一environment加载并跑恢复后缀。确定性mean action、
float32 actor/critic hidden、CUDA网络与Austin-only任务全部固定。

### 40.3 状态覆盖与质量合同

Snapshot覆盖：两台RaceCar 7D state、accel/steering velocity、2-step steering buffer、
opponent poses、collision latch与scan RNG；Simulator pose/collision/collision-index；F110的
current time、lap/toggle/start与render/current-observation镜像；opponent controller trajectory、
tracker count、speed scale以及LatticePlanner全部动态轨迹/cost/nearest/step字段和PurePursuit
误差；wrapper raw observation、previous measured speed、elapsed/current spec/reset RNG和全部
episode累计量；reward previous progress、relative position、collision latches、risk potential与
current clearances；corridor gate；actor和privilege-GRU critic hidden。Austin scan map与projector
几何是同一进程内只读对象，不复制。

28份snapshot全部通过pickle往返；28份原后缀与28份恢复后缀都具有对齐finite数组和唯一terminal
post-step row。独立复核重新读取全部56份NPZ和28份pickle，逐字段`np.array_equal`，没有只信runner
summary。

### 40.4 机制结果与效率含义

恢复后的当前observation、actor/critic snapshot hidden误差均为0；逐步381D observation、双车
360D LiDAR、两车7D state、steering buffer、actor raw/executed action、opponent executed action、
critic value、reward总量及progress/relative/collision/risk四分量、collision flags、terminal与
outcome全部最大绝对误差0。后缀动作步数范围99--800，中位154.5。

28条原轨迹共16,385个action step；冻结prefix合计9,589步，占58.5%。如果训练能直接从这些
snapshot起跑，这批任务理论上可省去这部分前缀仿真；本Gate没有测wall-clock并发开销，不能把
58.5%直接写成真实吞吐提升。两条任务的prefix为0，另26条大于0；10条prefix至少500步，所以
结果不是只在reset初态成立。

### 40.5 为什么仍不能训练

本Gate只证明**冻结U44网络、确定性mean action下的机械恢复**。本轮没有采样训练期探索噪声，
也没有把structured exploration的RNG/residual block作为snapshot状态审计；Z6-B必须预先定义
prefix后是恢复旧探索状态还是从窗口重新采样，并验证对应的log-probability合同。PPO更新后，
U44旧hidden不再是当前actor/critic对同一observation prefix的hidden；直接复用会制造策略—状态
不一致。GAE还需要确定哪些prefix transition排除在loss外、reset窗口的episode-start如何标记、
窗口首值/末值如何bootstrap、真实terminal与人为窗口边界如何区分。本轮没有检验这些语义，也
没有检验PPO是否能利用更密梯度。

### 40.6 停止与下一步

以下是Z6-A完成时冻结的顺序；Z6-B后来已执行并由§41更新当前状态。

1. Z6-A通过，prefix-reset不再因snapshot不可实现而停止；不得把结果解释为PPO方法通过；
2. 下一步只允许独立预注册Z6-B：用保存的actor-visible observation prefix对**当前网络**无梯度
   burn-in，逐位对齐从场景起点正常执行得到的actor/critic hidden、action与value；
3. 同一Gate必须冻结探索状态语义并验证log-probability重建、prefix transition不进入rollout
   loss/GAE、窗口末真实terminal与截断bootstrap对齐标准RecurrentPPO；任一失败即停止，不以旧
   U44 hidden或replay-to-prefix替代；
4. Z6-B通过后，才能设计一次Austin-only、标准PPO loss的训练密度Gate；四图只用于最终测试；
5. 2b仍是未测的不同方法：它增加辅助loss并改变student GRU；Z6-A的正结果不能替它提供证据。

耐清理核心记录：28条U42--U45共识development任务、21个startpoint、19/9两层全部通过序列化
snapshot恢复；所有已登记连续/布尔字段最大误差0；prefix中位345.5步、合计可跳过9,589/
16,385步。Z6-A当时只建立机械必要条件；current-network burn-in与GAE后来见§41，PPO仍未测。

## 41. Round Z6-B：current-network burn-in、boundary与GAE语义（2026-08-08，无actor更新）

### 41.1 Verdict table

固定单位为Z6-A的28条Austin development任务；U44是来源snapshot，真实相邻U45 actor/critic只
模拟“参数已改变的current network”。没有优化器step、PPO rollout或性能评估。

| 子门 | 固定分母 | 主结果 | 判决 |
|---|---:|---:|---|
| source prefix复现 | 28任务、9,589行 | observation/actor hidden/critic hidden最大误差`0/0/0` | 通过 |
| U45 current fast burn-in | 28任务 | hidden/action/value最大误差`5.84e-6 / 2.38e-6 / 1.49e-7` | 通过`5e-5`线 |
| boundary + GAE | 6-transition合成合同 | advantage/return误差`1.49e-8 / 0`；切段`0/3/5` | 通过 |
| baseline likelihood | 28 transitions | collection-equivalent log-ratio/ratio误差`0/0` | 通过 |
| corridor likelihood | 1,428 transitions | collection-equivalent `0/0`；batched `3.18e-5 / 3.18e-5` | 通过 |
| strict residual telemetry | 28 × 51 | 反算residual误差`3.18885e-6 > 0` | 原机器Gate失败 |
| Z6-BR内部noise裁决 | 同28 × 51 | 内部noise首50步误差0；第51步release；revisit最小差`0.08756` | **测量false fail；语义Gate通过** |

最终科学判决是`pass_prefix_reset_semantics_after_measurement_adjudication`。原始
`fail_stop_prefix_reset_semantics`不覆盖、不删除：它准确记录一个过严观测量失败；Z6-BR另行
证明该观测量不参与方法必要机制。

### 41.2 问题、调用链与current-network合同

Z6-A只证明冻结U44 hidden能随snapshot恢复；PPO参数更新后若继续复用旧hidden，recurrent state
与当前网络不一致。本轮固定真实U45 checkpoint，避免用人为权重扰动制造容易通过的例子：

```text
run_prefix_reset_semantics_gate.py
-> End2Race / PrivilegeGRUCritic current-network burn-in
-> End2RaceRolloutBuffer.stage_recurrent_resets
-> create_sequencers + standard SB3 GAE
-> End2RaceGRUPolicy collection/replay likelihood
```

每条从reset observation到window start前记录`[P,381]`。Actor与critic recurrent branch都只消费
前361D；privilege-GRU的20D P20只在每步value late-fusion，不改变critic hidden，所以保存361D
历史足够重建两路hidden，window current value仍消费完整381D。Prefix action/reward/value不写入
候选buffer。

U44逐步reference对Z6-A snapshot 28/28精确。U45逐步reference从零hidden消费同一observation
prefix；26条非零prefix中26条均与旧U44状态/输出至少一项非零不同，直接确认旧hidden不能复用。
U45一次整段GRU相对逐步reference最大actor hidden `5.84e-6`、critic hidden `4.77e-6`、action
`2.38e-6`、value `1.49e-7`。但U44来源网络的一次整段路径最大hidden约`0.0040`、action
`0.00175`、value`8.91e-5`，说明fast path不是跨checkpoint自动成立的数值定理；当前U45通过
不能授权以后每个update无检查使用。

### 41.3 为什么必须拆开两个start mask

标准单一`episode_starts`同时被用于GAE切断、sequence切分与GRU清零。Snapshot episode开头需要
前两者为true，却必须使用burn-in得到的非零hidden，所以一个mask无法表达。最小opt-in实现新增
`recurrent_resets`：

```text
episode_starts=true     -> GAE boundary + create_sequencers boundary
recurrent_resets=false  -> replay保留该sequence保存的burn-in hidden
```

普通collector没有stage独立mask时逐元素复制`episode_starts`，默认行为不变。合成合同在第
`0/3/5`行开始三个snapshot sequence，replay初始actor hidden为`1/2/3`、critic为`11/12/13`，
收到的recurrent reset全false。真实terminated不bootstrap、timeout预先加入`gamma*V_terminal`
再切断、rollout末非terminal使用`last_values`三种情况在同一递推中通过；prefix transition计数0。

### 41.4 Exploration与原strict false fail

Snapshot访问固定重新采样探索状态：探索start为true，recurrent reset为false。Baseline首步28条
与corridor 51步×28条的collection-equivalent likelihood都严格重建；corridor前50步active且同
正block，第51步inactive/block 0，再访问同snapshot得到不同residual。

原strict判据检查的是buffer里
`(sampled_action_speed - mean_speed) / exp(speed_log_std)`反算出的telemetry residual是否逐位相同。
FP32先做`mean + std*z`再做逆运算不保证恢复同一bit pattern，实测误差`3.18885e-6`。源码中该字段
只供exploration telemetry均值/标准差/同block诊断；PPO distribution replay使用保存的action和
speed log-std，不读取它。Z6-BR保持任务、U45、seed、51步和事前`5e-5` likelihood容差不变，直接
读取真正用于采样的内部temporal noise：首50步最大误差0，log-ratio仍0。因此原fail是测量
operationalization错误，不是方法必要条件被推翻。

### 41.5 仍未建立的内容与下一步

1. 没有测snapshot episode占rollout比例、prefix抽样、50/50 role与479 cache如何重新定义；
2. 没有测逐步burn-in或fast burn-in的wall-clock，58.5% simulator步节省不等于吞吐提升；
3. U45 fast path通过但U44 fast path失败，下一Gate必须使用逐步exact路径，或在每个update先做
   fail-closed fast-vs-reference校准；不能永久假定`5e-5`；
4. 没有收集完整prefix-reset rollout、执行optimizer step或评估actor；PPO能否利用更密梯度未知；
5. 下一步只准入独立Z6-C no-update training-density/integration Gate：固定snapshot比例、role语义、
   探索模式和采样器后，验证真实102,400-transition布局、ratio identity、GAE与wall-clock；
6. Z6-C未通过前不新建formal训练臂；方法2b和Constrained PPO仍未被本节检验。

耐清理核心记录：28条、9,589 prefix rows；U44 source逐位复现；U45 current fast最大误差
`5.84e-6`且旧hidden在26/26非零prefix上非等价；GAE/return`1.49e-8/0`；baseline/corridor
collection-equivalent log-ratio均0。原strict fail来自只用于telemetry的反算residual
`3.18885e-6`；独立裁决内部50步noise误差0，故语义必要条件通过，但训练密度与PPO效果仍未测。

## 42. Round Z6-C/Z6-CR：完整no-update训练密度与batched replay因果裁决（2026-08-08，无actor更新）

### 42.1 判决表

| 阶段 | 原始判决 | 核心结果 | 科学解释 |
|---|---|---|---|
| Z6-C | `fail_stop_prefix_reset_density_or_integration` | 12条准入线通过11条；treatment batched最大`|log-ratio|/|ratio-1|=0.010697/0.010755`，略超事前`0.01` | 原machine fail必须保留；单个最大值尚未说明实际PPO梯度被实质改变 |
| Z6-CR | `pass_prefix_reset_after_batched_gradient_adjudication` | full treatment复现；clip fraction 0；8个minibatch累计梯度cosine `0.9999838`、相对L2差`0.005706` | 事前因果裁决通过，只撤销该刀锋失败对正式训练的阻断，不证明方法有效 |

Z6-C与Z6-CR都没有optimizer step，actor/critic参数摘要前后不变。原Z6-C报告没有覆盖或改写；
Z6-CR是独立预注册、同配置完整重跑后的追加裁决。

### 42.2 冻结输入、单变量与完整rollout

两臂都从canonical BC actor和fresh privilege-GRU critic开始，固定Austin、seed 42、16 logical
env、每env 6,400步、总计102,400 transition、batch 12,800、baseline逐步独立速度高斯探索，
以及production的479 collision cache与600 ordinary pool。Baseline关闭prefix；treatment只改变
collision-role每第3次reset：用`SeedSequence([42, 0x50524658])`确定性遍历28条多checkpoint
共识snapshot，不消费479 queue，其余collision reset继续消费原cache。两臂实测role均严格为
collision/ordinary=`51,200/51,200`。

Treatment发生119次collision reset，其中39次prefix reset，严格等于`floor(119/3)`；其余80次
都来自479 cache，72次ordinary reset都来自600 pool。28个prefix key全部至少出现一次。
Prefix-origin transition为8,634/102,400（8.4316%），每次reset后首150步窗口为5,438
（5.3105%），同时通过事前5%和2%密度线。39个snapshot边界均满足window observation误差0、
`episode_starts=true`、`recurrent_resets=false`；prefix burn-in observation不进入rollout buffer。

当前内存actor/critic每次都从零hidden逐步消费保存的actor-visible prefix，合计13,965行。这些
行只用于重建recurrent state，不参与reward、GAE或PPO loss。若退回replay-to-prefix则需额外
模拟同样13,965步，本项目明确不把它作为fallback。

### 42.3 GAE、likelihood与吞吐

两臂102,400行全部finite；独立float64反推GAE的advantage/return最大误差均为0；逐transition
collection-equivalent replay的最大`|log-ratio|`与`|ratio-1|`均为0。Baseline普通batched最大
`|log-ratio|/|ratio-1|=0.006786/0.006809`；treatment为`0.010697/0.010755`，因此Z6-C按事前硬线
严格失败，不能事后把`0.01`改成`0.011`。

Baseline/treatment收集墙钟为`98.97/112.32s`，比值`1.1349`，通过不慢于baseline 20%的工程
线；吞吐分别为`1034.69/911.67 transition/s`。这是相同机器当次完整收集的工程测量，不应外推
为训练总墙钟或其他硬件性能。

### 42.4 Z6-CR为何是必要裁决而不是放宽阈值

普通batched路径正是PPO训练路径，所以不能像Z6-B的纯telemetry残差那样直接判作无关；但单个
最大误差也不是prefix-reset方法的自然必要条件。Z6-CR在看结果后没有调阈值，而是在再次完整
收集前冻结两个因果门：分布层要求最大ratio偏差`<0.02`、mean approximate KL`<=1e-4`、
clip=0.20 fraction为0；更新层在同一确定性8-minibatch dry actor epoch上比较普通batched与
collection-equivalent exact replay，要求累计梯度cosine`>=0.999`、相对L2差`<=0.02`。

重跑复现102,400 transition、28-key覆盖、8.4316%/5.3105%密度、GAE与exact likelihood为0。
Batched绝对log-ratio的mean/p99/p99.9为`4.15e-6 / 2.79e-5 / 1.48e-4`；最大ratio偏差
`0.010755`，mean approximate KL `9.05e-10`，clip fraction严格0。两个dry epoch都完成8个
minibatch且finite，不step参数；累计梯度cosine `0.9999838`、相对L2差`0.005706`，最大
minibatch policy-loss差`4.73e-7`。因此原最大值超线没有实质改变该冻结PPO actor更新。

### 42.5 机制结果、证据边界与停止规则

机制层正面结论是：在保持50/50 role、production cache/ordinary pool、标准PPO loss及默认探索
不变时，prefix-reset能把至少8.4%的训练transition置于共识困难窗口后缀，并在13.5%收集墙钟
开销内维持可执行的GAE、likelihood和actor梯度语义。它解决的是“能否可靠且足够密地采到关键
交互状态”，不是“PPO能否从中学会跨地图安全超车”。

证据没有建立训练收益、checkpoint稳定性、Austin验收或四图泛化。下一步只准入预注册的一次
`ppo_prefix_reset_consensus1of3`：canonical BC fresh start、30个formal update，不扫比例、窗口、
panel、学习率、exploration或训练长度；每update的exact ratio必须`<=5e-5`且batched最大ratio
偏差必须`<0.02`，否则立即停止。完成后固定评估U27--U30四图，不挑checkpoint、不延长到U45。
无论成功或失败都关闭本prefix-reset实例；它不能代判未测的2b、collision-only或Constrained
PPO，也不能把原Z6-C machine fail改写成通过。

## 43. Round Z6-F：prefix-reset单次正式PPO（2026-08-09，训练与四图评测完成）

### 43.1 判决表

| 层级 | 预注册条件 | 结果 | 判决 |
|---|---|---|---|
| 训练实现 | 31行finite metrics、30对checkpoint、12-key、ratio/prefix硬门 | 全部通过 | **通过**，run有效 |
| U30最低验收 | 四张图各自collision不高于BC且overtake不低于BC | 四图均通过 | **通过**，具备最低产品候选资格 |
| 最终目标 | 四图`collision < 40`且`overtake > 1500` | `103 / 1522` | **失败**，安全目标差63次以上 |
| Late稳定性 | U27--U30至少连续两点逐图通过BC线 | 只有U28、U30通过 | **失败**，通过点不连续 |

最终科学表述是：**prefix-reset在这个固定配置上扩展了经验高超车端前沿，但没有完成安全--
progress联合目标；关闭tested配置，不严谨否决prefix-reset方法类。** Production不自动切换。

### 43.2 唯一训练变量与完整性

Run从canonical BC fresh start，固定Austin、seed42、privilege-GRU critic、16 logical env、每env
6,400步、batch12,800、warm-up一轮加30个formal update、actor/critic epochs `2/5`、LR
`3e-6/3e-5/3e-4`、baseline独立速度高斯、`gamma/lambda/clip=.999/.995/.20`、50/50 role及
production 479/600 pools。唯一变化是collision角色每第3次reset使用28项共识prefix。

共31行metrics、30个actor和30个critic、5,166条完成episode记录；所有metrics和权重finite，
actor全部严格12-key，每个formal update均完成16/16 actor optimizer steps。规范`update1`--
`update30`入口与原checkpoint为hardlink，没有复制或改写权重。

Pre-update普通batched最大ratio偏差在U26为`0.019095 < 0.02`，U12为`0.015328`；其余更低。
Collection-equivalent exact偏差全程0。Prefix fraction范围`8.43%--12.47%`，首150步window范围
`4.26%--5.47%`，均持续高于5%/2%。训练后mean approximate KL存在明显尖峰：最小`0.00346`、
最大U14 `0.43747`，另有U21 `0.15071`、U23 `0.20171`、U29 `0.19475`；它们不是当前
`target_kl=None`合同的停止条件，但说明优化轨迹高方差，不能只报告late aggregate平稳。

### 43.3 评测数据质量

U27--U30在Austin、Hockenheim、MoscowRaceway、Nuerburgring各600条固定场景上CUDA、
deterministic mean action、ego collision scope、8秒、完整numeric trace评测。16包共9,600
result与9,600 trace；每包actor/panel SHA匹配，600个唯一key与trace集合相等，0 error。

逐条读取全部NPZ验证：所有数组首维一致且finite，`time_s`严格递增，末行唯一
`terminal_post_step=true/action_applied=false`且之前全部action applied；`ego_opp_collision`与
`ego_wall_collision` marker分别和result outcome严格等价且不同时为true；opp-wall事件数与汇总
一致。每包从600条episode重算collision/overtake/follow与typed collision，均和aggregate相同。

### 43.4 Late band逐图结果

每格为`ego collision / overtake`；BC为逐图最低线。

| checkpoint | Austin（BC `33/339`） | Hockenheim（`27/343`） | Moscow（`43/373`） | Nuerburgring（`26/390`） | 四图 | 逐图门 |
|---|---:|---:|---:|---:|---:|---|
| U27 | `17/365` | **`29/363`** | `36/391` | `23/399` | `105/1518` | 失败：Hock +2 collision |
| U28 | `20/367` | `27/365` | `32/395` | `22/398` | `101/1525` | **通过** |
| U29 | `23/367` | **`28/362`** | `27/399` | `23/400` | `101/1528` | 失败：Hock +1 collision |
| U30 | `21/365` | `27/365` | `33/393` | `22/399` | **`103/1522`** | **通过** |

U30 typed collision为Austin `14 ego-opp + 7 ego-wall`、Hockenheim `26+1`、Moscow `32+1`、
Nuerburgring `22+0`。四图opp-wall event episode为`1/4/0/2`，按ego scope单列，不计入103。

四点aggregate很窄，但episode身份并非冻结：U27→U28、U28→U29、U29→U30的四图collision
removed/created分别为`22/18`、`17/17`、`24/26`，overtake lost/gained为`12/19`、`14/17`、
`24/18`，双侧精确p均不显著。Hockenheim的+2/+1刀锋使U27/U29按硬门失败，不能把aggregate
稳定改写成预注册稳定性通过；同时也不能把这两个刀锋失败夸大为显著checkpoint崩溃。

### 43.5 相对canonical BC的配对结果

表中collision为BC→checkpoint的removed/created，overtake为lost/gained。

| checkpoint | collision removed/created | p | overtake lost/gained | p |
|---|---:|---:|---:|---:|
| U27 | `64/40` | `0.02365` | `21/94` | `3.24e-12` |
| U28 | `61/33` | `0.00508` | `17/97` | `8.64e-15` |
| U29 | `62/34` | `0.00557` | `16/99` | `8.56e-16` |
| U30 | **`60/34`** | **`0.00955`** | **`16/93`** | **`2.21e-14`** |

U30逐图配对为：Austin collision `20/8, p=0.0357`、overtake `4/30, p=6.16e-6`；Hockenheim
`11/11, p=1`与`3/25, p=2.74e-5`；Moscow `21/11, p=0.110`与`7/27, p=8.21e-4`；
Nuerburgring `8/4, p=0.388`与`2/11, p=0.0225`。因此U30相对BC的四图合计双轴改善有配对支持，
但Hockenheim安全只是净零且没有余量。

### 43.6 相对U44：明确的安全--超车交易

| checkpoint | collision removed/created | p | overtake lost/gained | p |
|---|---:|---:|---:|---:|
| U27 | `42/85` | `0.000170` | `39/79` | `0.000293` |
| U28 | `41/80` | `0.000499` | `35/82` | `1.65e-5` |
| U29 | `42/81` | `0.000556` | `31/81` | `2.53e-6` |
| U30 | **`40/81`** | **`0.000244`** | **`33/77`** | **`3.30e-5`** |

U30相对U44少安全41次、但多超车44次，两个方向都显著。它没有支配U44，而是移动到高超车/
高碰撞端；相对现有四图点，`1522`超过已记录重加权U30的`1516`，所以可称经验高超车端前沿
扩展，但不能称安全突破。最终目标要求`<40`，实测103，差距不是Hockenheim刀锋造成的。

### 43.7 机制判断、边界与停止规则

Z6-A/B/C/CR及formal训练共同证明：精确snapshot、current-network burn-in、GAE/likelihood、
足够prefix密度与一次标准PPO更新都可实现；正式actor又相对BC显著双轴改善。因此“关键窗口
采样完全无效”不成立。失败发生在更高层：单靠这组28个hindsight共识prefix及1/3 schedule，
没有把策略推到`<40/>1500`联合区域，而且Hockenheim逐图安全只有零到负2的刀锋余量。

这支持“训练密度能改变前沿，但仍未解决跨状态/地图的安全--progress统一选择”，不支持
“prefix-reset方法类被证伪”。只有本配置被关闭：不扫interval、panel、窗口、exploration、LR、
updates，不延长U45，不把U28事后选为production。Production保持原U30；若未来重开方法类，必须
提出能解释并修复U44→Z6-F显著安全回退的新单变量机制，而不是重复增加关键状态密度。

## 44. Round Z7：collision-only BC anchoring overlap-supported独立重开（2026-08-09，无训练）

### 44.1 判决表

| 阶段 | 样本/合同 | 结果 | 判决 |
|---|---:|---:|---|
| 新独立panel | 40新起点 × 2 raceline × 4 interval × 9 speed = 2,880 | 与历史heldout/Austin600起点精确零交集 | 冻结有效 |
| 分阶段actor screen | BC/U44各2,880；U42/U43/U45各59 | 5,937 result/trace，0 error，全部质量合同通过 | 有效且等价于完整source screen |
| V0 support | 41 stable eligible source + 41 exact control；21起点；r0/r2=`31/10` | 五条样本/匹配门全过 | **通过，旧Z3阻断解除** |
| branch0 | 82条 | action、opponent、pose/speed、双LiDAR最大误差均0 | **通过** |
| full-BC source | 18/41 rescue；18/18最终overtake | 要求rescue至少21 | **失败** |
| full-BC controls | 新collision=4/41；overtake loss=4/41 | 各允许最多2 | **失败** |

科学判决：**严谨关闭当前canonical BC × overlap-supported stable collision × 固定1.5秒窗口的
collision-only functional anchoring实例。** 没有生成anchor dataset、shadow beta或新actor。

### 44.2 问题、独立panel与效率等价

旧Z3在7条source上因同raceline/speed无放回control不足而停在plan阶段，只能inconclusive。
Z7不是放宽旧panel，而是冻结40个新的循环等progress起点：offset 1629，和历史candidate/Austin600
精确零交集。每个起点完整覆盖`raceline0/2`、interval `8/10/12/15`及speed `0.45--0.85`，共
2,880个唯一ScenarioSpec；panel在任何actor outcome前冻结。

BC完整panel结果为`123 collision / 2472 overtake / 285 follow`，U44为`104/2457/319`。只有
`BC=overtake、U44=ego collision`的59条在定义上可能成为source，因此后三个checkpoint只在该
冻结子集上运行；U42/U43/U45分别为`38/18/3、35/22/2、40/18/1`（collision/overtake/follow）。
省略其余2,821条的三次评估不会改变任何source标签，只减少8,463次无信息闭环。

所有5,937条required result都有numeric trace；key、identity、finite、shape/time、唯一terminal、
action-applied及typed ego collision marker均通过。候选集合在U42/U43/U45 outcome前冻结，避免
按稳定结果回改全面板。

### 44.3 overlap-supported estimand和V0

稳定source要求BC overtake、U44 collision且U42--U45至少3/4非overtake。41条满足且首次U44
collision前窗口均至少50步。safe control要求BC/U44共同overtake；窗口是U44全局最小OBB
clearance前100至后50步。每个`(raceline,speed)`内source按事前SHA排序，只取
`min(source,control)`，再按循环ego距离与scenario-key SHA无放回匹配。

本轮所有41条稳定source都有exact support，没有被cap裁掉；匹配得到41 control，覆盖21个source
起点，r0/r2=`31/10`。因此V0的`>=12 source、>=8起点、两条raceline各>=2、exact control闭合、
全部质量合同`同时通过。结论适用于overlap-supported source；即使本轮cap未裁样，也不外推到
另一个数据分布中没有同层control support的source。

### 44.4 branch0与full-BC结果

Branch plan冻结82条、source到control的一一映射、U44 collision/control窗口和所有模型/输入身份。
branch0逐条重新闭环后，ego raw/executed action严格误差0；opponent action、两车pose/speed、双
360D LiDAR也都是0；boolean marker、outcome、steps、首次collision identity/time step与terminal
完全一致。branch0总48,983行，证明后续差异来自BC intervention而不是replay漂移。

full-BC在冻结窗口同时替换steering和speed，之后恢复U44。41 source中18条被救回，全部最终
overtake，另23条仍为ego-opp；rescue覆盖11个起点，r0/r2=`16/2`。机制不是完全无效，但
`18/41=43.9%`低于预注册`ceil(.5×41)=21`。按raceline看r0为16/31，r2仅2/10；按speed仍碰撞/
救回混合，不能用某个成功层替代全体门。

41个baseline共同安全overtake control中，full-BC造成4个ego-opp，恰好也是4个overtake loss；
r0/r2各2条，分散在speed `0.60/0.75/0.80`。预注册上限为`floor(.05×41)=2`，两条control门均
失败。Rescue率95% Wilson约`29.9%--59.0%`，harm率约`3.86%--22.55%`；这些区间说明有限样本
不确定性，但不允许事后修改事前50%/5%方法合同。

### 44.5 机制解释、证据边界与停止规则

正面机制证据仍应保留：BC局部接管能把18条稳定collision全部救成overtake，说明teacher动作在
一部分新起点确有安全-progress兼容信号。失败也同样是机制级的：同一固定teacher/window对
23/41目标无效，并把4/41已安全overtake变成碰撞。因而一个不具备未来outcome oracle的functional
regularizer无法只继承18条成功行为而天然回避这些失败/有害状态；这正是正式训练前必须成立的
teacher必要条件。

本轮满足样本门、exact controls、branch0与产物完整性，所以不是“不达指标但可能只是实现错”或
“功效不足的null”；它直接违反预注册方法实例的救援和control必要条件。关闭该实例，不扫窗口、
teacher、support cap、门限、raceline或beta，不用validation分支生成anchor。该判决不外推到
其他teacher、其他干预窗口或一般BC正则化方法类；原双分层Gate B仍因L=0/9关闭，旧Z3仍保持
inconclusive。下一科学未决项只剩会改变GRU表征的2b，以及目标定义修正后的Constrained PPO。

## 45. Round Z8：GRU-changing paired action-response auxiliary Gate（2026-08-09，无PPO）

### 45.1 判决表

| seed | frozen target I/C/L | frozen controls collision/loss | trainable GRU target I/C/L | treatment controls collision/loss | 判决 |
|---:|---:|---:|---:|---:|---|
| 7100 | `58 = 37/21/0` | `12/13` | **`50 = 30/20/0`** | **`14/15`** | 失败 |
| 8100 | `61 = 38/22/1` | `19/20` | **`58 = 36/21/1`** | **`12/13`** | 失败 |

事前要求每seed treatment target至少88、比frozen至少+9、lost至少4、control两类harm各最多5。
两套seed全部失败。判决：**关闭本节paired collision/progress representation-only 2b具体实例，
不进入独立validation或PPO；不否决所有2b方法类。**

### 45.2 首次真正触及2b的地方

Z2、Z4-A、Z5都在U44 frozen hidden上训练selector，最多新增一个外置history encoder，不能限制
“辅助loss反传改变student GRU”的2b。Z8复用同一456条/70起点Austin development task与late
noop+12动作闭环标签，但treatment从U44原GRU初始化，response loss直接更新1680D GRU；`k`、
speed MLP与actor output layer冻结。Control只在保存的U44 hidden上训练同容量response head。

监督也不再是旧三分类：每个state-action同时预测ego collision indicator和最终relative progress
相对noop的米制差，loss固定`0.5 class-balanced BCE + 0.5 SmoothL1`。GRU/head LR为
`3e-6/3e-4`，weight decay `1e-4`，batch64、10 epoch、gradient clip1.0，无early stop或模型选择。
Head只用于机制Gate，训练后删除；本轮没有optimizer接入PPO或生成actor checkpoint。

### 45.3 recurrent输入与exact修复

每条state使用真实U44 actor-visible 420D输入：从episode起点逐步burn-in到late start前50步，
保留detached hidden，再让原GRU消费最后50步。首次实现错误地把前缀合并成batched burn-in，事前
exact门测得hidden/action误差`0.021586/0.009939`并在任何训练报告前停止；这不是科学结果。

修正后每条episode严格batch-size-one逐步重放。456条late hidden和该步raw action相对Round Z2
保存值最大误差均0；正式feature/initial/frozen tensor分别为`456×50×420`、`456×1680`、
`456×1680`且全部finite。该修复恢复预注册语义，没有改数据、seed、loss或门限。

### 45.4 无泄漏操作点与真实表征变化

对test fold f，calibration固定为`(f+1)%5`，其余三fold训练；五次轮转后每个startpoint恰好一次
test。模型只在train labels拟合；`lambda=0,.25,.5,1,2,4,8,16`和noop threshold只在calibration
选择，calibration safe controls按各折分母只允许1或2条collision/loss。Test outcome从未用于
模型或操作点。两套独立seed base 7100/8100都执行相同协议。

五fold treatment的GRU参数相对U44 L2为`0.00210--0.00260`，test hidden平均相对L2为
`0.00938--0.01231`，全部超过执行门；所以负结果不是“GRU没更新”。冻结output head时平均
steering/speed动作漂移为`0.00092--0.00188 rad / 0.00964--0.0191 m/s`；seed7100某fold最大speed
漂移达0.634 m/s。后者不是验收指标，但说明小参数变化可在个别状态产生较大functional drift。

### 45.5 结果与机制失败

Seed7100的trainable GRU相对frozen target独有/对照独有成功为`2/10`，paired exact
`p=0.0386`，显著更差；controls从12/13进一步到14/15。Seed8100 target为`11/14, p=0.690`，
无可检出优势；controls相对frozen少7次harm（两类均`1/8, p=0.0391`），但绝对12/13仍远超
5/5门。不能用一个seed的相对安全改善抵消绝对预算、target与lost恢复失败。

更关键的是，所有calibration fold都满足各自1或2条harm预算，迁移到独立test后两seed treatment
却达到14/15和12/13。这说明paired collision/progress head在development内能拟合局部响应，
但其操作点与表征变化不能跨startpoint维持safe-control specificity。两seed lost-overtake只有0/1，
延续前几轮“progress恢复最难”的独立信号。

### 45.6 证据边界与停止规则

这是首次直接训练student GRU，因此可以否决本节具体2b实例的必要机制：在exact recurrent输入、
真实表征改变和无泄漏calibration下，它没有形成比同监督frozen hidden更好的harm-constrained
action structure。不得扫描epoch、GRU/head LR、loss权重、window、lambda或threshold，也不运行
同实例独立validation/PPO。

但456条branch标签此前被Z2/Z4/Z5查看过，本轮仍是development机制筛查；且2b是一个目标大类。
结论不外推到未定义的新辅助目标、不同反事实horizon或所有representation learning，报告字段
明确保持`representation_only_2b_class_refuted=false`。当前六方案中最后仍需直接论证的是修正
目标定义后的Constrained PPO；它不能由本节action selector结果代判。

## 46. Round Z9：collision-cost Constrained PPO preflight（2026-08-09，无actor更新）

### 46.1 判决表

| 层 | 预注册要求 | 实测 | 判决 |
|---|---|---|---|
| 完整rollout | 102,400 transition | 102,400 | 通过 |
| cost/reward唯一化 | event身份相等、逐行误差0 | 57=57、误差0 | 通过 |
| cost signal | advantage finite且std≥0.001 | std `0.27325` | 通过 |
| actor调用链 | 合成梯度差分相对L2≥1e-4 | `0.31087` | 通过 |
| 起点OOF coverage | ≥20起点、≥10 collision、每fold正例 | 85起点、57 collision、每fold9--14 | 通过 |
| OOF MSE skill | ≥0.05 | `0.04038` | **失败** |
| Episode-start AUROC | ≥0.65 | `0.42855` | **失败** |
| Early≥100步 AUROC | ≥0.65 | `0.60703` | **失败** |
| dual方向 | rate>d时上升 | `.37255>.10`，`1→1.13627` | 通过 |

最终machine verdict为`fail_stop_exact_constrained_implementation`。30-update formal与四图评测
均未运行；actor未改变，也没有Z9 actor checkpoint。

### 46.2 被检验的准确方法

本轮解决了原提案的重复计价歧义：环境仍记录既有首次ego collision `reward_collision=-2.0`，
但rollout buffer在reward GAE前逐行减掉这一分量；同一首次collision只作为`cost_t=1`。Reward
critic保持fresh `privilege_gru`；训练期cost critic固定读取P20，结构20-120-30-1。Cost
`gamma/lambda_GAE=.999/.995`，预算是50/50人为训练分布上已完成episode collision率`.10`，dual
固定初值1、LR.5、范围0--20。若准入，actor会使用一次标准化后的
`A_reward-lambda*A_cost`进入原clipped PPO，部署actor接口不变。

这是一项不可分割的算法变量：移出reward collision与新增constraint共同把“固定罚分”改为
“显式安全预算”。它没有teacher、动作库、prefix、部署shield或测试地图调参。

### 46.3 数据、完整episode与五折外推

Canonical BC用baseline独立高斯探索在Austin生产479 collision/600 ordinary池上收集16×6,400
transition，逻辑role严格8/8。完整episode共153，57 collision、96非collision，collision率
37.25%；尾部未完成episode不进入OOF真值。完整episode含98,737 transition、85个独立ego
waypoint。

每个collision episode的Monte-Carlo cost-to-go为`0.999^(T-t)`，其他episode为0。所有相同
`ego_idx`进入同fold；85个起点按冻结SHA顺序均分为每fold17个。五折test episode/collision为
`24/11、35/14、34/14、32/9、28/9`，test transition为
`14,043/23,082/20,890/21,979/18,743`。每折只用其余起点训练同构P20 MLP 10 epochs；test既不
调epoch也不调阈值。

### 46.4 正面机制结果：不是零cost、零梯度或dual失效

57个`ego_collision` transition与57个episode terminal一一对应，`reward_collision`数组严格
等于`-2×cost`；去重后buffer reward相对`original+2×cost`最大误差0。Cost advantage/return
标准差`0.27325/0.25761`，不是稀疏到全零。独立cost critic warm-up训练/validation MSE从
`0.1014/0.0835`下降到`0.0512/0.0627`，参数确实改变，actor摘要不变。

Dual按完成episode rate从1升到`1+.5×(.372549-.10)=1.1362745`。在该值下，reward-only与
合成actor全buffer梯度L2为`25.3429/20.0068`，差分L2`7.8783`、相对`0.31087`、cosine
`0.96687`。因此实现已经把cost优势传入真实recurrent actor surrogate；不能把负结果说成
“Constrained PPO没有接上loss”。

### 46.5 失败机制：跨起点提前估计不足

OOF cost MSE为`0.10857`，fold-train均值常数baseline为`0.11314`，skill只有4.04%，比5%线少
0.96个百分点，是刀锋失败；但两条排序门不是刀锋。Episode首步57正/96负的AUROC为0.42855，
距terminal至少100步的83,501行中16,301正/67,200负，AUROC为0.60703，均低于0.65。

结果不是“没有任何碰撞信息”：early AUROC高于0.5且MSE优于常数。准确表述是，固定P20 MLP
在held-out startpoint上只获得弱的提前排序结构，达不到为稀疏constraint提供稳定低方差baseline
的事前最低要求。Episode-start低于0.5只报告方向，不做显著反向宣称，因为没有预注册cluster
显著性检验。

### 46.6 两次机械停止与证据边界

第一次完整收集后，reward GAE重算把numpy bootstrap传给要求tensor的SB3 API；第二次完整收集
后，OOF fold helper对尾部未完成episode的新起点查表。两次都在warm-up/actor update与科学
report前退出，各自保留独立目录；修复只更正类型和valid mask，不改数据、网络、fold、seed或
判据。第三次才是有效实验。

预注册把OOF Gate定义为formal治理门，故三条失败足以停止当前实例，但**不是Constrained PPO
方法类的必要条件证伪**：constraint policy gradient原则上可依靠Monte-Carlo cost，其他cost
表示或约束目标也可能不同。严格结论只关闭
`reward去collision + P20 MLP + d=.10 + lambda0=1 + dual_lr=.5`配置，不扫参、不formal、不四图。
Production保持原U30。至此本阶段所有已授权具体方案均已按各自停止规则执行或关闭；存活的是
若干方法类的理论可能性，不是仍在队列中的实验。

## 47. 跨轮汇总、证据级别与干预类型模式（2026-08-09，Claude 独立复核，无新仿真）

本节不改写 §36--§46 的任何判决。它只补三件各轮小结没有并排给出的内容：我实际重算过什么、
全部可部署点在同一张带证据级别的表里长什么样、以及把历史训练臂和本轮全部 Gate 按**干预
对象**（而不是方法名）归类后出现的模式。完整逐项记录见 `.agents/GATES.md` 附录 C。

### 47.1 独立核验记录

我重算或直接核对过：Round Z2 四个 JSON 的 SHA-256（与记录精确匹配）、`10,944 + 912` 个 NPZ、
branch0 全部 15 个字段最大绝对误差 `0.0`、Z2 全部表格算术自洽（分层 `109+46+13+63+225=456`、
五折各列、oracle label 与 head 动作分布两列各 456、`147/148/31/51/79`、`+31/-28`）；
Gate B 的 `lost_overtake.restored_overtake_count = 0/9` 与四条失败判据；Z3 的
`16 -> 7`（56% 瞬态）；Z4-A 的 `target_margin_over_fixed: true` 与 noop 率 `7/456`；
Z5 的配对 `28:38, p=0.268` 与 `29:42, p=0.154`；Z6-F 四图 `105/1518`、`101/1525`、
`101/1528`、`103/1522`（逐 episode 复算）；SWA 四图 `67/1465` 与其对 U44 的配对
`13/18, p=0.473`、`24/11, p=0.041`；Z7/Z8/Z9 的机器 verdict 与关键字段。

**未发现执行错误。** branch0 零误差、哈希匹配、nested CV 结构正确，以及 §42/§43 对刀锋失败
（Z6-C `0.010755`、Z6-B 反算 residual `3.18885e-6`）"原 fail 与复核 pass 并列保留"的处理，
都符合本仓库最严格标准。

### 47.2 证据级别与前沿

四图 `collision / overtake` 与证据级别见 `HANDOFF.md` §9.28 的表；此处只记结论：

- 六个可部署点近似共线，**全部已执行方法都在安全--超车前沿上滑动**，没有一个把前沿推开；
- 唯一越过前沿的 `25 / 1557` 需要未来信息，不可部署；
- **ordinary 异线高速重加权 U30 是唯一四图双轴优于 production 的点**，但它是 trace 重建包
  （`direct_evaluator_aggregate_retained=false`、无 `device`/无顶层 `collision_scope`），
  而 production 自身缺 Moscow/Nuerburgring 规范包；当前比较是"重建对 headline"，
  补齐需 `3600` episode、零训练；
- prefix-reset U30 `103/1522` 被重加权 U30 `73/1516` 双轴支配。

### 47.3 干预类型模式（推断）

| 干预对象 | 实例 | 结果 |
|---|---|---|
| 特定 regime 的**探索** | 前向走廊门控时间相关速度噪声 | 跨地图同线碰撞 `66 -> 22` |
| 特定 regime 的**采样权重** | `ordinary_offline_fast_fraction=0.6` | 唯一四图双轴优于 production |
| **时间窗口**的 transition 密度 | prefix-reset（§43） | 前沿滑动，被重加权 U30 支配 |
| **全局难度/多样性** | 805 pool、ordinary150、interval15 pool | 全面变差或配对不显著 |
| **冻结产物**上的选择器/锚定 | Gate B、Z3、Z7、Z2、Z4-A、Z5、SWA | 全部未准入 |
| 训练中**改变表征**的辅助目标 | Z8（§45） | 表征确实改变，但 target 与 control harm 未过 |
| **约束式目标重构** | Z9 preflight（§46） | 机械链路成立，cost critic OOF 三门失败 |

**推断（非定律）**：本项目里唯一见效过的两次干预，都是"把训练分布对准一个已被测量定位的
regime"。对准时间窗口、对准全局难度、或在冻结产物上加选择器，都没有移动前沿。

### 47.4 尚未被任何一轮检验的轴

**已核实事实**：历史上全部 8 条训练臂的 `STEERING_LATENT_STD` 一律为 `0.03`——§18 的五组
速度探索实验、hard-neighbor、重加权、prefix-reset 无一例外，**转向探索通道从未被改变过**。
另一条已核实事实：探索是纯训练期机制，正式评测直接取 mean action、无噪声无门，而
`FrontCorridorGate` 本身条件在模拟器特权几何上、同样没有部署期 conditioning。

**推断**：`HANDOFF.md` §10.1 第 16 条以"缺少可靠部署期 conditioning"关闭 side-phase steering
exploration，该理由适用于部署期相位门控，不适用于训练期探索门。相关已测证据（新造失败以
近平行侧/后擦碰为主、相对 yaw 中位 `3.67°`，§28/§9.9；Z2 最强单一固定干预
`steer +0.02 / speed +0.5` 带转向分量，`GATES.md` §A13.1）指向横向通道。

**边界**：这是未测假说，不是发现，同样可能只是又一次前沿滑动。在用户明确裁定并按本仓库
标准重新预注册（单变量、seed 42、≥4 相邻 checkpoint 区间、三面板配对身份、失败即关闭）
之前，§10.1 第 16 条继续有效。

### 47.5 本节不改变的事项

不改变任何既有 verdict、停止规则或证据边界；不改变 production（仍为 U30）；不授权任何训练；
不把诊断成功写成 actor 性能，也不把未测轴写成可行方案。
