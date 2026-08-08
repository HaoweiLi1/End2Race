# End2Race 当前 HANDOFF

更新时间：2026-08-09（Asia/Singapore；Round Z9 collision-cost Constrained PPO preflight完成：机械链路通过、起点OOF三门失败，formal停止）

## 0. 文档职责和读取顺序

本文件是**接手当前仓库时的第一入口**，只保留会改变下一步行动的当前事实、核心合同、
实验最终判决和停止/重开规则。完整数据与机制推导已经迁入：

- `ANALYSIS.md`：完整实验设计、面板定义、配对结果、分层数据、机制判断和证据边界；
- `EXPERIMENTS.md`：历史实验工具与回归测试的实现逻辑和重建合同；
- `GUIDE.md`：用户要求的实验执行、命名和文件布局规范；
- `GATES.md`：后 U44 六类候选方法的**预注册、执行与判决全记录**。它由原
  `SELECTIVE_BC_SAFE_BEHAVIOR_ANCHORING_PREREGISTRATION_FINAL_REVIEWED.md` 与
  `COUNTERFACTUAL_ACTION_GATE_COMPLETE_REPORT.md` 于 2026-08-09 无损合并而成
  （第一部分 §0--§35 = 原预注册；附录 A §A1--§A23 = Round Z2 完整技术报告；
  附录 B §B24--§B29 = 原报告追加节；附录 C = Claude 独立复核与跨轮汇总）。
  本文件 §9.17--§9.28 的判决与它一致；冲突时以本文件为准。

如果本文件和 `ANALYSIS.md` 的历史数字冲突：

1. 当前源码、模型文件、run config 和机器可读结果优先；
2. 当前 production、运行状态和允许的下一步以本文件为准；
3. 历史实验的完整数字、推导与边界以 `ANALYSIS.md` 为准；
4. 不要用当前 CLI 默认值反推旧 checkpoint 的配置。

用户已决定后续清理历史分析产物、实验工具和回归测试源码。清理前的核心判决、
关键统计、production U30残余失败身份和oracle动作族已固化到 `ANALYSIS.md` §21；测试与工具的
重建合同在 `EXPERIMENTS.md`。这些文档保留的是**结论与关键实现合同**，不是原始数据和源码
的无损压缩。`eval_results/`、模型/run目录不属于本次三目录清理范围。除actor checkpoint
外，不维护JSON/CSV/report/trace哈希；模型身份见§2。

仓库：`/home/haowei/Documents/End2Race`

分支：`main`

提交和工作树状态必须在接手时实时查询，不能从本文件反推。`.agents/`已纳入Git版本管理；
本轮一次性梯度诊断脚本、张量和评测复核产物在核心结果与重建合同写入文档后清理，不作为
后续实验入口。

---

## 1. 当前状态和 production 决策

### 1.1 运行状态

2026-08-09 Round Z9 collision-cost Constrained PPO preflight已完成，无Z9 actor update。固定实例
从reward return逐行移除首次collision `-2.0`，同一事件只作cost；102,400 transition中153个
完整episode、57 collision、85起点。Cost event=collision episode=57、reward去重误差0、cost
advantage std `.27325`，dual按37.25% rate从1升至`1.13627`；合成actor梯度相对reward-only
差分L2为31.1%，证明cost链路非零。但startpoint五折OOF的MSE skill/episode-start AUROC/
early≥100步AUROC仅`.04038/.42855/.60703`，均低于`.05/.65/.65`。按预注册停止当前
`P20 MLP × d=.10 × dual_lr=.5`实例，不运行30-update或四图、不扫参；不否决Constrained PPO
方法类。两次更早执行只触发类型/valid-mask机械错误，均无actor update和科学report。见§9.27与
`ANALYSIS.md` §46。

2026-08-09 Round Z8 GRU-changing paired action-response auxiliary Gate已完成，无PPO、新仿真或
actor checkpoint。456条late输入按真实batch-size-one recurrent语义重建U44 hidden/action最大
误差0；两seed五fold的GRU参数与test hidden均确实改变。Seed7100 treatment为target 50、controls
collision/loss `14/15`，低于frozen 58、`12/13`；seed8100为58、`12/13`，低于frozen 61、
`19/20`且绝对harm仍超5/5；lost恢复仅0/1。两seed均失败target88、相对frozen+9、lost4和controls
门。关闭当前paired collision/progress、late50、10-epoch具体2b，不调参、不进validation/PPO；
不否决所有representation-only辅助目标。见§9.26与`ANALYSIS.md` §45。

2026-08-09 Round Z7 collision-only BC anchoring overlap-supported独立重开已完成，无训练。
新40起点×2 raceline×4 interval×9 speed的2,880条Austin panel与历史heldout/Austin600起点精确
零交集；BC/U44全面板和U42/U43/U45的59条潜在source共完成5,937 result/trace，0 error。稳定
eligible source为41条，精确同raceline/speed无放回control也是41条，覆盖21起点、r0/r2=`31/10`，
V0全部通过，旧Z3的support阻断已解除。branch0在82条上全部动作、pose/speed、双LiDAR最大误差
0；full-BC只救回`18/41 < 21`，虽18/18均overtake，但control新collision与overtake loss各
`4/41 > 2`。判决为严谨关闭当前canonical BC × overlap-supported stable collision × 固定1.5秒
窗口实例；没有anchor、shadow beta或actor，不扫teacher/window/support/beta。见§9.25与
`ANALYSIS.md` §44。

2026-08-09 Round Z6-F正式训练与U27--U30四图评测完成，当前无训练/评测进程。31行metrics、
30对actor/critic、5,166条训练episode全部finite；30个actor均12-key且每update完成16/16 actor
steps。Pre-update batched ratio最大`0.019095 < 0.02`，exact最大0；prefix/window最低
`8.43%/4.26%`。16个CUDA deterministic包共9,600 result/trace全部通过key、finite、terminal与
typed collision合同。U30逐图`21/365、27/365、33/393、22/399`，合计`103/1522`：最低BC线
通过，但`collision<40`最终目标失败；U27/U29在Hockenheim分别`29/28 > 27`，只有U28/U30通过，
连续稳定性失败。判决为高超车端经验前沿扩展、目标未完成；tested配置关闭，不否决方法类，
production不变。见§9.24与`ANALYSIS.md` §43。

2026-08-08 Round Z6-C与Z6-CR已完成，无actor更新。Baseline/treatment各完成102,400 transition，
role均严格`51,200/51,200`；treatment有39次prefix reset、28 key全覆盖，prefix/window transition
为`8.43%/5.31%`，GAE误差0、collection-equivalent ratio误差0、参数不变，墙钟比baseline
`1.1349x`。原Z6-C因普通batched最大`|ratio-1|=0.010755`略超事前`0.01`而保留machine fail；
独立预注册Z6-CR完整复现后，clip fraction为0、mean KL `9.05e-10`，8-minibatch dry actor梯度
cosine `0.9999838`、相对L2差`0.005706`，通过因果裁决。这只准入§9.23固定的一次正式
prefix-reset PPO，不证明方法有效。完整数字见`ANALYSIS.md` §42。

2026-08-08 Round Z6-B及独立Z6-BR测量裁决已完成，无actor更新或formal训练。28条source
snapshot observation/hidden精确；U45 current-network fast burn-in相对逐步reference最大
hidden/action/value误差`5.84e-6 / 2.38e-6 / 1.49e-7`，26/26非零prefix确认旧U44 state不可复用。
Boundary/GAE advantage/return误差`1.49e-8 / 0`；baseline与corridor collection-equivalent
log-ratio均0。原strict report唯一false是从action反算telemetry residual有`3.18885e-6` FP32误差；
原machine fail保留，Z6-BR直接测内部noise首50步误差0并通过全部事前裁决线，确认这是非因果
测量false fail。科学判决是语义必要条件通过，只准入Z6-C no-update训练密度/吞吐Gate，不准入PPO。

2026-08-08 Round Z6-A prefix-reset snapshot no-op工程门已完成，无actor更新。Gate A冻结的
28条U42--U45至少3/4共识development任务全部在窗口起点保存F110、LatticePlanner、reward/
wrapper与U44 actor/critic hidden，pickle往返后恢复并重跑确定性后缀。28/28通过；381D
observation、hidden、actor/opponent action、critic value、reward分量、两车state/steering buffer、
双LiDAR、collision与terminal/outcome全部最大误差0。26条prefix大于0，中位345.5步，合计可跳过
9,589/16,385步（58.5%，不是wall-clock测速）。这只准入下一道current-network burn-in与GAE/
bootstrap语义Gate，不准入PPO；不得复用旧U44 hidden或退回replay-to-prefix。见§9.21与
`ANALYSIS.md` §40。

2026-08-08 Round Z5 budget-constrained frozen-hidden operating-point Gate已完成，零新仿真、无actor
更新。456条既有branch outcome按ego startpoint做5 outer x 4 inner nested CV；`lambda/noop`
阈值只在inner-OOF选择。独立outer seeds的frozen selector为target 69、controls `1/3`；恢复
Z4原outer seeds的复核为66、`3/4`，均在不高于fixed `5/5` harm时低于fixed target 79。
配对`p=0.268/0.154`，准确科学表述是未检出严格优于fixed，不是显著劣于fixed；外层又只用了
`1/3`与`3/4`预算，故不是精确用满`5/5`的harm matched比较。事后全OOF选点在独立seed可读84，
但nested仅69；该乐观值不承担判决。当前tested selector按预注册关闭，2b仍未测；未运行
validation、actor接入或PPO。见§9.20与`ANALYSIS.md` §39。

2026-08-08 Round Z4-A representation-changing action-response Gate已完成，无actor更新。
它复用Round Z2已验证的456条development task与late 13-action真实终局，按70个ego startpoint
做五折OOF；treatment只增加动作前50步actor可见LiDAR+previous-speed GRU。treatment恢复
inherited/created/lost为`68/109、32/46、2/13`，controls新collision/overtake loss=`13/21`，
target success `102`低于同监督frozen-hidden control的`104`。按预注册关闭该具体表征实例，
没有运行Z4-B validation candidate、共享student GRU或PPO。见§9.19与`ANALYSIS.md` §38。

2026-08-08 Round Z3 collision-only BC anchoring独立validation Gate已停止，无branch、无训练。
从未打开的150条/19起点validation完成BC、U42--U45共750条CUDA replay，全部result/
trace合同通过；得到7条、6起点的稳定U44 collision cohort，raceline0/2=`6/1`。
但预注册的同raceline、同speed、无放回matched controls在`r0/s0.60`缺2条、`r0/s0.85`
缺1条，任何精确匹配都无解。runner在plan阶段fail closed，没有运行branch0/full-BC。
判决为**inconclusive / matched-control support insufficient**，不证明collision-only teacher通过或失败；
不得在当前panel上放宽匹配或直接训练。见§9.18与`ANALYSIS.md` §37。

2026-08-08 Round Z2反事实动作Gate已完成，无actor更新。456条Austin development场景完成
456次精确U44 no-op和10,944次early/late固定动作branch，11,400条trace合同全部通过。oracle动作
在early对inherited/created/lost救回`93/109、44/46、10/13`，late为`96/109、45/46、7/13`；
但startpoint分组外推head在early只有`19/109、11/46、1/13`并误伤17/225 controls，late为
`34/109、17/46、0/13`且target success `51 < 79` fixed-action baseline。当前fixed-library
action-conditioned与first-action preference关闭；prefix-reset不准入；MoE继续因12-key边界
关闭。Constrained PPO、prefix-reset的实际训练机制和MoE架构本身均未被该Gate直接检验；
后两者分别只是未过旧程序触发条件、以及违反当前12-key部署边界，不能写成科学否决。见§9.17与
`ANALYSIS.md` §36。

同日 BC-safe anchoring Gate B已完成并科学失败，无训练。Gate A冻结的28条cohort和28条
matched controls完成branch 0、完整BC、steering-only、speed-only共224次CUDA replay；branch 0
的action、opponent action、两车pose/speed、双360D LiDAR最大误差全部为0，224条trace合同均
通过。完整BC在C层救回`10/19`碰撞且10条全为overtake，但L层恢复`0/9`，safe controls又损失
`2/28`次overtake；两个独立门失败。按预注册关闭本方向，不生成anchor dataset，不进入Gate C/D
或45-update formal训练；validation未运行branch。当前没有训练或评估进程。见§9.16与
`ANALYSIS.md` §35。Gate A的cohort证据仍有效，见§9.15与`ANALYSIS.md` §34。

同日Round Z0也已完成，无训练：固定U42--U45按float64等权平均得到12-key actor，四图CUDA为
`67 collision / 1465 overtake`，相对U44 `62/1478`双轴变差；Hockenheim `19/341`低于BC的
overtake下限343。该臂按预注册关闭，不改band或权重重试。2,800条正式/诊断episode均通过结果
与trace完整性合同。完整配对、core-14机制结果与停止规则见§9.14及`ANALYSIS.md` §33。

2026-08-06 fresh first-step PPO regime梯度诊断已经完成并按预注册否决梯度投影训练；没有
`train_ppo.py`训练任务，也没有产生新actor。production U30跨地图固定面板也已完成fresh复核：
CUDA结果精确复现历史合计`80 collisions / 1142 overtakes`和same-line/off-line碰撞`66/14`；
CPU为`78/1142`和`64/14`；碰撞差来自两个临界episode，另有三条follow/overtake边界翻转，
说明后续配对实验必须固定推理device。

第一次fresh trace评测暴露了评估器的旧缺陷：它先写model-derived canonical目录，再把trace
移动到目标目录，会掏空已有canonical trace根。该实现已改为worker直接写目标目录，并通过
73-episode回归（73 result/73 trace、key集合相等，三个既有Austin trace根计数不变）。旧的
跨地图canonical GPU traces已不可用；一次性CPU/CUDA复核包在数值固化后按用户要求清除。
后续正式评测只使用CUDA，不再做CPU双跑；需要trace时直接在CUDA评测中保存。系统中仍
可能存在历史 `tail -f` 监控进程，它们不表示训练仍在继续。
进程状态会变化，接手时必须重新运行：

```bash
pgrep -af '[r]un\.sh|[t]rain_ppo\.py|[e]val_multiagent\.py|[e]valuate\.sh' || true
```

除上文明确记录的Z6-F活动评测外，没有需要续跑的实验。其他已有 run 目录均视为完成、否决、
被替代或封存，禁止在原目录续写。
中断后若需重跑，必须使用新目录；当前 checkpoint 不包含 optimizer、scenario queue、
environment 或 recurrent state，不能称为 exact resume。

同日通过`192.168.50.2:2222`只读核对远端：远端代码树没有新增改动，唯一未跟踪文件是旧的
regime审计Markdown；其全部结论已经由本地§9.6和`ANALYSIS.md` §23覆盖。没有把远端旧版
HANDOFF覆盖回本地，也不保留第二份根目录审计文档。

Round Z0之前的最新完成活动见§9.13及`ANALYSIS.md` §32；其直接前置诊断见§9.12及`ANALYSIS.md` §31。
2026-08-07只完成了Austin开发面板上的真值几何/速率早期可分性与当前交互几何表征缺口两项
无训练预检；没有新actor、正式eval或训练。更早的interaction-phase线性筛查见§9.11及
`ANALYSIS.md` §30，fresh PPO梯度诊断
见§9.7及`ANALYSIS.md` §24，它不是历史update replay：checkpoint不含
scheduler、optimizer或environment state；它只测固定场景队列上、ratio约等于1时的首步PPO
梯度。K10/K20/K30没有稳定复现人工偏好审计的`-0.96`冲突，虚拟投影也未持续保护两个主
regime，因此不启动30-update梯度投影臂。

### 1.2 当前唯一 production

Production actor 仍为 **production U30**：

```text
run:
  post-trained/ppo_privilege_gru_clip020/
checkpoint:
  update30/actor.pth
critic:
  privilege_gru
seed:
  42
clip:
  0.20
formal updates:
  30
reward:
  固定四项；无Post-pass接口；risk longitudinal固定0.6m
exploration:
  baseline；speed std 0.15；不使用时间相关或条件探索
collision pool:
  default 479；hard-neighbor off
ordinary starts:
  50
```

同一训练轨迹的 U1--U45 已合并到上述canonical实验族；U30仍是production checkpoint，
U35/U40/U45只用于收敛性记录。本文当前状态统一称其为`production U30`；历史表中的
`B`仅是旧实验标签，不用于组织新的分析。

### 1.3 一页最终结论

- Canonical BC 是 `pretrained/end2race.pth`；禁止覆盖。
- Actor 部署输入保持 361D：360 LiDAR + previous measured ego speed。
- Privileged critic 输入为 381D：actor 361D + 20D pre-action privileged state；
  特权信息不会进入 actor。
- Production reward 保持四项：progress、relative progress、首次 ego collision penalty、
  potential-based risk shaping。
- 被否决的Post-pass生产模块和L12运行时override均已移除；following-response
  required-deceleration仅是离线候选。
- Production collision pool 保持479；完整805 hard-neighbor、比例采样、外部fixed pool、
  ordinary150和actor-mismatch复用接口已经在文档固化后退役。ordinary异线高速重加权
  作为默认关闭的研究工具保留。
- Production speed exploration 保持逐步独立速度白噪声。条件白噪声放大、全局时间相关
  速度噪声、条件时间相关速度噪声、走廊门控时间相关速度噪声、延长训练和异线高速
  重加权在历史多面板协议下均未通过。用户最新明确：后续正式验收默认必须运行
  **Austin、Hockenheim、MoscowRaceway和Nuerburgring各600 episode**；Austin600与
  三张跨地图合计crossmap1800都具有正式验收权，三张跨地图也必须逐图报告。near400和
  hard73只保留机制/特化诊断意义，不再拥有独立否决权。最低验收线是canonical BC：每张
  正式地图的ego collision不得高于BC、overtake不得低于BC，opp-wall单列；Production U30仍是当前部署模型和
  reference候选，但不是新模型必须逐项超过的最低线。
- Canonical BC四图正式验收下限（当前`collision_scope=ego`口径）：

| 地图 | ego collision | overtake |
|---|---:|---:|
| Austin | 33 | 339 |
| Hockenheim | 27 | 343 |
| MoscowRaceway | 43 | 373 |
| Nuerburgring | 26 | 390 |
| 三张跨地图合计 | 96 | 1106 |

- 旧Austin trace中`ol0_e629_o638_s0.8`在4.09s因opponent-wall被legacy口径提前截断，
  不能从该截断trace推断ego-scope终局。2026-08-06仅对该场景做CUDA ego-scope补跑：
  opponent撞墙后episode继续，ego在4.83s发生ego-opponent collision，因此Austin仍为`33/339`，
  只是碰撞子类从`opp-wall`改为`ego-opp`。Hockenheim的4个和Nuerburgring的2个
  opponent-wall event均发生在8.01s终点，trace已包含完整终局且均为overtake；它们已在
  `343/390`中，不得再重复相加。所以当前ego-scope四图总基线为`129 collisions / 1445 overtakes`。
- 新四图BC下限使前向走廊门控时间相关速度噪声U44/U45重新进入复核，而不是产生新训练需求。
  现有规范结果中的逐图`collision/overtake`为：U44 `Austin 18/344、Hockenheim 16/347、
  MoscowRaceway 15/390、Nuerburgring 13/397`；U45为`17/345、20/344、11/391、13/396`。
  U44与U45都逐图严格满足BC计数下限；U45的Hockenheim overtake `344 >= 343`。
  这不推翻该实验在旧Austin+near协议下的历史否决，而是任务验收线改变后对同一结果的重新
  判读。2026-08-06已补齐BC四图规范result/manifest并完成配对验收：

| 地图 | BC→U44 collision | removed / created | collision p | BC→U44 overtake | lost / gained | overtake p |
|---|---:|---:|---:|---:|---:|---:|
| Austin | `33→18` | `24 / 9` | `0.0135` | `339→344` | `11 / 16` | `0.442` |
| Hockenheim | `27→16` | `18 / 7` | `0.0433` | `343→347` | `14 / 18` | `0.597` |
| MoscowRaceway | `43→15` | `37 / 9` | `4.06e-5` | `373→390` | `12 / 29` | `0.0115` |
| Nuerburgring | `26→13` | `15 / 2` | `0.00235` | `390→397` | `4 / 11` | `0.118` |
| 四图合计 | `129→62` | `94 / 27` | `7.14e-10` | `1445→1478` | `41 / 74` | `0.00269` |

  BC→U44的ego-opp逐图为`28→13、27→15、43→15、25→13`，ego-wall为
  `5→5、0→1、0→0、1→0`；opponent-wall event保持`1/4/0/2`不变。四图两侧均通过
  600 unique、0 errors、finite trace、result/trace key精确相等、collision marker和terminal row合同。
  后期band中U42/U43的Austin overtake为`332/335 < 339`，U44/U45则连续逐图通过BC线；
  因此预先固定复核的U44现正式接受为**四图BC验收候选**，U45只作相邻checkpoint
  稳定性支持，不事后改选单点。Production部署别名暂时仍为U30，直到用户明确切换；
  不启动reference-KL或其他新训练。
- 上述U44是历史旧称`CT-v2`的**前向走廊门控时间相关速度探索**轨迹的formal
  update 44，不是Production U30续训。它从canonical BC fresh-start，该run预计45 updates；
  与Production的核心差异是训练期在2m前向走廊内将同一速度噪声残差保持50步（0.5s），
  speed std仍为0.15；确定性eval与部署actor结构不变。
- 按现存headline合计，Production U30约为`94 collisions / 1508 overtakes`，U44为
  `62 / 1478`。因此U44通过的是“四图都不差于canonical BC”，不是相对Production U30
  的Pareto改进；它比U30少32次碰撞、也少30次超车。Production跨地图缺完整逐episode
  规范包，这里只能作总量trade-off说明，不报配对显著性。
- 旧near400诊断仍显示U44为`37 collisions / 288 overtakes`，Production U30为`28 / 325`。
  near400按用户当前口径没有正式否决权，因此不推翻四图BC验收；但它明确说明旧协议下发现的
  “贴身成功场景受损”副作用没有被证明消失。若将来把验收目标提高到Production级性能或近失
  鲁棒性，这组`+9 collisions / -37 overtakes`必须重新成为守门依据，不能只引用四图总量。
- GPT Pro提出的训练期U30 reference-policy regularization现在**停止而非实验否决**：
  当前BC验收目标已由U44满足，而该方案会引入冻结teacher和第二actor loss，不再是纯PPO
  exploration。只有用户把目标提高为“保留Production U30超车同时保留U44安全”，并明确授权
  训练期策略保持目标时，才重开same-prefix teacher有效性、mask覆盖和fixed-beta虚拟更新预检。
- ordinary异线高速重加权比例0.6的U30在旧协议中因near400显著退化被否决；用户把正式
  验收改为四图逐图BC下限后，现有U27--U30结果需要重新判读。四个checkpoint均逐图满足
  BC线，U30为`Austin 16/368、Hockenheim 17/368、MoscowRaceway 17/389、
  Nuerburgring 23/391`，四图合计`73/1516`。相对BC的四图配对为collision
  `129→73`（removed/created `84/28`，`p=1.11e-7`），overtake `1445→1516`
  （lost/gained `26/97`，`p=7.97e-11`）。它不是相对U44的Pareto改进：相对U44为
  `+11 collisions / +38 overtakes`，collision removed/created `37/48`（`p=0.278`），
  overtake lost/gained `41/79`（`p=6.67e-4`）。旧near400仍为`64/302`，明显差于
  Production U30的`28/325`和U44的`37/288`。
- 上述重加权结果由完整600-trace重建包支持，四图场景身份与BC/U44逐图一致，但旧manifest
  没有记录CUDA device和顶层collision scope，因此它现在是**计数与配对已验证、正式CUDA
  provenance待确认的产品候选**，不能直接取代已完成CUDA正式验收的U44。
- **当前未决是三档产品取舍，不是新训练：**四图安全优先选U44（`62/1478`）；四图超车
  优先且接受中间安全点时，先对重加权U30做一次固定CUDA四图确认（当前`73/1516`）；Austin
  主场或贴身成功能力优先则保留Production U30（headline `94/1508`，near400 `28/325`）。
  在用户明确选择前不改production路径或默认训练配置，不启动K25、reference-KL或新探索臂。
- 用户把更高目标明确为四图`collision < 40`且`overtake > 1500`，同时规定**训练只能使用
  Austin**；Hockenheim、MoscowRaceway和Nuerburgring只允许测试泛化，禁止multi-map PPO。
  §9.9和`ANALYSIS.md` §28表明现有结果不是物理/actor容量上限，而是当前Austin-only PPO的
  经验前沿：U44主要保留same-line安全，重加权U30主要保留off-line-fast超车，两者修复大量
  BC碰撞同时各自新造约二十余次碰撞。当前不授权继续扫重加权比例、hold、std、gate或updates；
  若要追求新目标，先审查§28提出的Austin-only相位可分性诊断，再决定是否存在合法新训练轴。

- Production U30 固定面板基线：

| 面板 | collision | overtake | 角色 | production逐episode结果 |
|---|---:|---:|---|---|
| Austin600 | 14 | 366 | 正式验收；不是独立地图留出 | `update30/Austin/multiagents/` |
| near400 | 28 | 325 | 历史贴身成功诊断；当前无否决权 | `update30/Austin/near400/` |
| hard73 | 54 | 12 | 专门化诊断，不是验收门 | `update30/Austin/hard73/` |
| 三张跨地图合计1800 | 80 | 1142 | 正式验收；必须同时逐图报告；已参与走廊探索设计，不是干净最终留出 | `update30/<MAP>/multiagents/` |

Austin600、near400和hard73都有production逐episode结果。三张跨地图的历史canonical GPU
trace在旧评估器搬移缺陷触发后已不可用；一次性CUDA结果包曾复现上表数值，但不含trace且已在
文档固化后清理。现存Hockenheim package未记录device、没有trace，且其`26/355`与已记录CUDA
`26/356`不一致；MoscowRaceway和Nuerburgring当前没有保留结果包。因此不能把现存跨地图目录
称为规范CUDA结果包。当前正式最低对照是canonical BC，不是Production U30；只有未来要宣称
“超过Production U30”或做正式U30配对时，才fresh重评Production四图并保存trace。
near400 与 hard73 的包是 2026-07-30 清理后用
`scripts/evaluate_scenario_panel.py` 重新评估补齐的：清理删掉了旧的 near400 trace 目录，
而这两个面板此前没有 production 侧的规范包。重评精确复现了上表数字
（near400 `28/325`、hard73 `54/12`），因此表中数值未变。

- 当前起点公式下 BC Austin600 为：总碰撞 33（ego-opp 27、ego-wall 5、
  opp-wall 1）、overtake 339、follow 228。历史 `22/344/234` 来自旧起点公式，
  不能与当前 production U30 的 `14/366/220` 配对。
- 目前最重要的正面机制发现：
  时间相关和走廊门控探索能显著改善same-line跟车碰撞；同时可观测到off-line状态出现新的
  速度/安全代价。门在off-line几乎不触发，所以这是门外的间接学习迁移，而不是直接误触发；
  但其精确优化机制尚未识别。
- 目前最重要的负面结论：
  继续加 reward 剂量、加困难池、加全局探索幅度、把走廊门控时间相关探索延长到
  45 updates，或重加权
  ordinary 异线高速，都没有得到同时通过**旧Austin+near+泛化协议**的单一模型；该历史
  结论不等于它们在当前四图BC最低线下仍无资格，重判见上文和`ANALYSIS.md` §26--§27。
- 2026-08-05人工成功动作偏好审计在共享output head上得到S-O/S-N cosine
  `-0.962/-0.959`；但2026-08-06 fresh真实PPO rollout复核没有稳定复现该冲突，实际走廊
  探索的U10/U20/U30 aggregate cosine为`+0.663/+0.622/-0.186`。因此人工偏好结果只能作为
  局部动作方向描述，不能据此隔离output head或训练梯度投影；当前迁移机制仍未定位。
- Oracle 诊断证明残余碰撞在动作接口上可解，但使用未来碰撞时刻和离线搜索；
  它不是模型成绩，不允许作为 runtime shield 或部署后处理。

### 1.4 当前工作树边界

当前没有需要接续的训练改动；一次性审计脚本、notebook、NPZ、JSON和counterfactual续跑
记录已按用户要求删除或在本轮文档固化后清理。`.agents/`已经纳入Git版本管理，后续修改
必须出现在普通`git status`中。2026-07-30已经在
四份文档交叉核对后完成模型/评测清理：只保留§2登记的canonical模型、一条历史未定案
hard-neighbor 10%模型、规范panel/cache和§4.3所述评测根。该10%模型只作为历史资产，
其训练接口已经按用户决定退役，不再属于“待完成实验”。除此之外仍禁止：

- `git reset --hard`
- `git checkout -- <path>`
- 清理当前白名单中的 `eval_results/`、`post-trained/` 资产或其他用户文件
- 覆盖 canonical BC
- 在已有 run 目录继续写入

清理后接手顺序仍是
`HANDOFF -> ANALYSIS -> GUIDE -> EXPERIMENTS`；不得因为原始目录已不存在而把文档中明确
标记为“离线候选”“被替代”“未测试”的项目改写为正式实验结论。

本次清理删除45个旧模型根和260个旧评测根；删除不可恢复。清理后的实测验收：

- 9个`post-trained/`顶层根、8个`eval_results/`顶层根，`eval_results_old/`已移除；
- 6条canonical训练轨迹checkpoint连续且无缺失，未定案hard-neighbor 10%保留45个actor；
- 10个关键actor均可被`End2Race(hidden_scale=4)`严格加载；
- 76个正式评测包共43,000 episode，63个600包、13个400包，result/trace key完全一致、
  聚合计数一致、0 error；
- 两个base collision cache按当前canonical actor路径严格通过身份验证；
- reward/合规回归 `scripts/test_screen_reward_candidate.py` 55/55 通过
  （2026-07-30 实测；数量会随新增测试变化，接手时重跑而不要照抄该数字）；
  `evaluate.sh` 通过 shell 语法检查。`run.sh` 已随历史实验命令一并删除，见§3.1。

---

## 2. 模型身份登记

### 2.1 规则

只对 actor checkpoint 记录 SHA-256。不对 analysis/eval 产品记录哈希。
SHA 的用途仅是确认评测模型身份、识别等价 checkpoint 和检测 canonical 文件被覆盖。

表中历史保留模型值实测于2026-07-30；production U30与前向走廊门控时间相关速度噪声U30
在2026-08-05无训练审计入口再次核对，SHA-256未变。

### 2.2 基准与等价集

| 模型 | 路径 | SHA-256 |
|---|---|---|
| Canonical BC | `pretrained/end2race.pth` | `b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4` |
| **production U30** | `post-trained/ppo_privilege_gru_clip020/update30/actor.pth` | `e7e902d92bb7cbea6ec0c08b9a4754dd9da6fe50dac98dfc2c5abc90096bcba8` |

完整U1--U45权重与训练记录已经迁入canonical根目录。迁移时验证了短训、45-update
延长、structured-exploration control和current-code reproduction中的U30 actor等价；
这些来源容器已在2026-07-30清理，不再是合法引用入口。

### 2.3 保留的 treatment checkpoint

| 方向 | checkpoint/run | SHA-256 |
|---|---|---|
| base479 U45 | `post-trained/ppo_privilege_gru_clip020/update45/actor.pth` | `5233f5096a610e3db618f944fa78cc20530998124a2478ecf4454850e54c2325` |
| hard-neighbor 10% U45，历史未定案、已主动停止 | `post-trained/ppo_hard_neighbor_boundary_aware_fraction0p10/checkpoints/actor_u0045.pth` | `2a24861a32331e39e887361def1813b5eea0324f196363df45314581c6c08f0a` |
| 条件白噪声放大 U30（训练模式已退役，见下） | `post-trained/ppo_conditional_white_speed_noise_0p50/update30/actor.pth` | `ab3f584b0ce445a7e241cd22f5da1faa559348acfd6ed320d514823ce69eb049` |
| 全局时间相关速度噪声 U30 | `post-trained/ppo_global_temporal_speed_noise_0p15_hold50steps/update30/actor.pth` | `2920c30ff88dff78a61e2bd4afbeb1faf83f2c5d9c3a66b2d910049a34db2b91` |
| 条件时间相关速度噪声 U30 | `post-trained/ppo_conditional_temporal_speed_noise_0p25_hold50steps/update30/actor.pth` | `28bb4aafe1a81d6041750c8f1eae6f087c005334401544a2a32b1f6096516390` |
| 前向走廊门控时间相关速度噪声、2米门宽 U30 | `post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update30/actor.pth` | `b8ecc0a52bc01e521f1daff6abf2611091d5d33df2e5aef73a3b93f091b89182` |
| **前向走廊门控时间相关速度噪声 U44，四图BC验收候选** | `post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update44/actor.pth` | `fb0c9895eb2ff004e414da09e4ee27675e825f0e6413a095377d66838e411bf7` |
| 前向走廊门控时间相关速度噪声 U45 | `post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update45/actor.pth` | `305dfa8160a987e2b166d8ce548009cd667fa8cfdd0722880f43249ccb07295c` |
| ordinary异线高速重加权、比例0.6 U30 | `post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps_ordinary_offline_fast_reweight_0p60/update30/actor.pth` | `4c10ff9f4e2e2f76afadb51e8f18f86173815c364da346ad07d88ebc1c29a341` |
| U42--U45等权平均，Round Z0已否决 | `post-trained/zero_train_ctv2_u42_u45_equal_average/actor.pth` | `41b57c6a589fc5b960409d3bef63a9a815609659b8171c47d03d7bd219633e50` |
| Prefix-reset Z6-F U27，late band | `post-trained/ppo_prefix_reset_consensus1of3/update27/actor.pth` | `918c1d122fa78788f864d3060b7441c5178595cdf0a265566a8d72740a54d7dd` |
| Prefix-reset Z6-F U28，late band唯一非主通过点 | `post-trained/ppo_prefix_reset_consensus1of3/update28/actor.pth` | `02144c38972665820353dc96c6ef222ed02bf91785c1930d75b92e7983e6f9b4` |
| Prefix-reset Z6-F U29，late band | `post-trained/ppo_prefix_reset_consensus1of3/update29/actor.pth` | `8367e5460719a67f70b965a64b46f689bf52603130499daf242ecea9d61052d3` |
| Prefix-reset Z6-F U30，主判决点、最低BC线通过但目标未完成 | `post-trained/ppo_prefix_reset_consensus1of3/update30/actor.pth` | `872674b42ca940772281d1b7f2a4745341e3772d8da754e504adcf2f182a6121` |

Round Z0的评估别名`CTv2_U42_U45_EQUAL_AVG.pth`与表中`actor.pth`是同inode硬链接且SHA一致；
它只是避免trace串臂的等价入口，不是第二个模型。摘要实测于2026-08-08。

**条件门控高方差逐步独立速度噪声的特殊状态**：其 U30 checkpoint 与评测**保留**，但 2026-07-30 已从
训练代码移除 `conditional_white` 模式及其 `EscalatingRequiredDecelerationGate`。
因此该模型**可以继续评估和配对，但当前代码无法复现其训练**；重跑前须按
`EXPERIMENTS.md` 重建模式与门。移除理由（门曝光仅约0.15%、无机制证据）见§3.8。

Post-pass、risk L12、完整805 hard pool、1米走廊、speed-std、退火、ordinary150和
interval-15 difficult等已否决模型的结论与数值仍保留在本文件和`ANALYSIS.md`，但原始
checkpoint和重复评测已清理；因此它们不再出现在actor身份登记表中。

前向走廊门控时间相关速度噪声与ordinary异线高速重加权已经完整迁入canonical目录：

```text
post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/
post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps_ordinary_offline_fast_reweight_0p60/
```

前者保存U1--U45全部actor/critic和完整训练记录；后者保存U1--U30。对应评测按
`eval_results/<完整实验名>/update<N>/<MAP_NAME>/`组织：标准600场景位于
`multiagents/`，Austin的固定near400位于`near400/`。Treatment各保留面板有完整数值trace、
逐episode `results_multi.json`和`eval_manifest.json`；production的Austin600/near400有
逐episode结果，三张跨地图只有完整trace、缺result与manifest。跨地图历史结论仍由§8与
`ANALYSIS.md` §18保存；需要新的机器可读配对时先重建并校验，不能假定目录已经是规范包。

历史未定案hard-neighbor 10%的模型和Austin评测分别位于同名canonical模型根目录和
`eval_results/ppo_hard_neighbor_boundary_aware_fraction0p10/`，保留
U1/U5/U10/U15/U20五个完整600包。它是“训练完成、晚期统一判决未完成、用户主动
停止”，不再是待办实验，也不得写成有效或无效。20%臂已由U35/U40/U45配对结果否决，
原模型和评测已清理。

**布局例外（有意保留，不是遗漏）**：hard-neighbor 10% 是唯一仍使用旧
`checkpoints/actor_uNNNN.pth` 布局的保留模型，另有若干顶层 `*_u000N.pth` 评估别名副本。
其余6个保留模型都已按 GUIDE §6 迁为 `update<N>/{actor.pth,critic.pt}`。因为该臂尚未定案、
且已经主动停止，所以没有再为历史资产做布局迁移。清点 checkpoint 时必须同时匹配
两种布局，只找 `update<N>/actor.pth` 会把它误报为空。

---

## 3. 当前代码与运行合同

### 3.1 入口和模块

```text
train_ppo.py            唯一 PPO 训练入口
ppo/env.py              单环境、前向走廊门与一env一worker VecEnv
ppo/policy.py           actor、四种critic、P20与三种速度探索模式
ppo/reward.py           固定reward、progress与OBB/map-wall geometry
ppo/scenarios.py        场景、队列、collision classification/cache
ppo/rollout.py          recurrent buffer、warm-up与PPO更新
ppo/ppo_config.yaml     固定运行、reward、场景和研究配置
utils.py                通用评测工具与训练记录/checkpoint保存
evaluate.sh             多车固定面板调度
eval_multiagent.py      deterministic 双车eval与numeric trace
eval_singleagent.py     单车多圈与LiDAR beam masking
post-trained/           run、checkpoint、collision-cache
```

2026-07-30 `run.sh` 已整体删除（先清空历史命令，随后连同文件一并移除）。仓库当前**没有**训练命令入口：下一项实验必须在预注册完成后新建 `run.sh`，只写该实验一条可直接复制运行的显式命令。
旧实验的状态索引保留在 `ANALYSIS.md` §13；其中 temporal hold K10/K25 与 Group13
明确为未运行，hard-neighbor 10%明确为训练完成但无晚期统一eval结论，20%已否决。当前脚本
没有待运行实验，误执行会明确失败，不会产生伪完成记录。

新增/实验性模块的功能和测试入口由 `EXPERIMENTS.md` 记录；本文件只保留生产合同。

### 3.2 环境

正式解释器：

```text
/home/haowei/miniconda3/envs/end2race/bin/python
```

当前环境合同：Austin、16 logical env、16 workers（一env一worker）、100Hz、8秒/800步。
Linux `forkserver`；worker 内 OpenMP/MKL/OpenBLAS/Torch threads=1。
CUDA 初始化前创建 subprocess environments。

Collection 保持 logical-slot batch-size-one actor/critic execution；training replay 才按
timestep 对 active recurrent sequences 做 FP32 batch。不得为了提速静默改变 collection
数值合同。

### 3.3 Actor

Actor input：

```text
360D LiDAR + previous measured ego speed = 361D
```

GRU hidden 是真实 recurrent state；SB3 cell `c` 只是 dummy zero。Episode start清零，
rollout边界不清零。

动作分布：

```text
steering: latent Normal -> tanh -> ±0.52 rad；latent std 0.03 frozen
speed: physical Normal；std 0.15 m/s frozen
```

Actor只训练 End2Race GRU 和 output layer；pressure `k`、speed MLP、log_std冻结。
Actor checkpoint必须严格12 keys。

### 3.4 Critic

当前支持：

```text
mlp
independent_gru
priviledge_mlp
privilege_gru
```

Production 使用 `privilege_gru`。它有独立 BC 初始化 recurrent branch，并把20D privileged
projection加到value head；actor仍只看到361D。

20D privileged state全部为当前 pre-action simulator/map量，不含 sampled action、
next observation、future collision 或 terminal outcome。当前合同是 P20/381D；
历史12D/373D已废弃。

### 3.5 PPO defaults

```text
critic                         privilege_gru
hidden_scale                   4
n_envs / worker processes      16 / 16
seed                           42
n_steps                        6400
batch_size                     12800
num_updates                    30
actor_epochs / critic_epochs   2 / 5
GRU/head/critic LR             3e-6 / 3e-5 / 3e-4
steering/speed std             0.03 / 0.15
gamma / GAE lambda             0.999 / 0.995
clip                           0.20
normalize_advantage            true
ent_coef                       0
target_kl                      fixed off（训练入口已移除）
vf_coef / max_grad_norm        0.5 / 0.5
```

训练入口默认值现已与 production U30 的 clip0.20、30 updates、privilege_gru 对齐。

每 rollout 为102,400 transitions。`num_updates=N` 会先收集一个critic warm-up rollout，
再执行N个formal updates。Actor phase与critic phase分离；returns/advantages在同一old
rollout上固定。

历史target-KL `0.02/0.04`均未优于关闭状态，`0.04`还造成明显恶化；production winner
始终关闭。当前训练入口和actor early-stop分支均已删除，常规approx-KL仅作为telemetry
保留。历史实现合同见`EXPERIMENTS.md`。

### 3.6 Scenario与cache

Even logical ranks固定collision role，odd ranks固定ordinary role；env-major minibatch保持
两种role transition数量相同。

Ordinary默认：

```text
50 starts × 3 racelines × 4 speeds = 600
interval = 15
```

Collision candidate网格：

```text
100 starts × 3 racelines × 4 intervals × 9 speeds = 10,800
intervals 8/10/12/15
speeds 0.45...0.85
```

Default cache：479 ego collision、10,285 other、36 invalid；构建耗时约4494.83秒。
`post-trained/collision-cache/` 是训练输入，不是普通分析产品，不得随分析结果一起清理。

当前规范目录：

```text
pretrained_end2race_austin_collision_pool_479
pretrained_end2race_austin_boundary_aware_collision_pool_805
ppo_privilege_gru_clip020_update30_austin_collision_pool_372
```

2026-07-30仅迁移了cache目录位置，历史run的`run_config.json`和
`collision_cache_info.json`继续保留运行时原始路径，不把迁移后的路径伪写成原始参数。

Schema-1 default cache只记录actor路径，不能检测同路径模型被覆盖。若 canonical BC 身份变化，
必须指定一个新的空`--collision_cache_dir`；partial或identity mismatch会fail closed，
当前入口不会覆盖已有cache。

### 3.7 Reward

Production reward：

```text
r = 0.01 * ego_progress_delta
  + 0.02 * (ego_progress_delta - opponent_progress_delta)
  - 2.0 * first_ego_collision
  + gamma * Phi(next) - Phi(current)
```

Risk potential：

```text
vehicle_distance =
  hypot(longitudinal_clearance / 0.6,
        lateral_clearance / 0.2)
wall_distance = ego_OBB_wall_clearance / 0.2
distance = min(vehicle_distance, wall_distance)
Phi = -0.05 * max(0, 1 - distance)^2
```

真正terminal使用 `Phi(next)=0`；timeout保留physical next potential并bootstrap。
当前生产代码不存在Post-pass第五项，也不允许在CLI覆盖risk纵向尺度；纵向尺度固定从
`ppo_config.yaml`读取0.6。L12历史实验只是把该尺度从0.6改成1.2，并非新增reward项。

**清理criterion（这条不是自明的，改代码前先读）**：reward合同冻结且不可从CLI配置；
训练分布/探索工具只有在仍有明确研究价值时才保留。2026-07-30先删除Post-pass生产模块
和L12运行时override，随后在完整重建合同写入`EXPERIMENTS.md`后，又退役ordinary150、
外部fixed pool、完整805/比例hard-neighbor和actor-mismatch cache复用。ordinary异线
高速重加权保留为默认关闭工具；它不代表production winner。

区别不在于"是否被否决"，而在于**改的是任务定义还是训练分布**：reward 就是任务规格，
让它可配置等于让"目标是什么"变成运行时参数；pool 和 exploration 只改怎么采样和怎么
试探，不改目标。所以：

- 不要从当前CLI缺少某个历史flag反推实验从未做过；这些接口的设计、测试和结果已经固化
  在`ANALYSIS.md`与`EXPERIMENTS.md`。hard-neighbor 10%是主动终止的未定案方向，
  不能改写成已证伪；
- 反过来，也不要因为"退火 flag 还在"就认为可以新增 reward 项或 reward CLI 覆盖。
  任何 reward 改动都要先过 `scripts/screen_reward_candidate.py` 的合规门禁与
  归一化学习信号量化，并需要用户重新授权。

### 3.8 实验flag的 production 默认

```text
collision pool                         strict default 479 cache
ordinary startpoints                   fixed 50
--speed_exploration_mode               baseline
front_corridor_gate_maximum_gap_m       2.0（YAML；仅前向走廊门控时间相关速度噪声）
ordinary_offline_fast_fraction          null（YAML）
```

当前CLI只保留 **全局时间相关速度噪声（`temporal_global`）与前向走廊门控时间相关速度噪声
（`corridor_temporal`）** 两个非默认模式。
条件门控高方差逐步独立速度噪声、旧所需减速度门控时间相关速度噪声、speed-std退火和
target-KL均已退役。
历史结果仍保留在本文与`ANALYSIS.md`，功能重建合同在`EXPERIMENTS.md`。

**为什么只留这两个**：按当时的Austin600+near400联合验收，三个探索臂都未通过，但证据强度
不同。条件门控高方差逐步独立速度噪声（`conditional_white`）的门曝光仅约`0.15%`——
判决就是"门太稀疏"，即干预量不足以产生学习信号，属于
**没有机制证据**的臂，其专用门控却占`ppo/exploration.py`近一半代码，因此连同
`EscalatingRequiredDecelerationGate`一并移除。保留的两个模式相反：它们携带本项目
**最重要的正面机制发现**（见§1.3），失败在 off-line 代价而非机制不成立，是最可能被重启
的方向，故保留可运行。删除标准因此是**"有无机制证据"，不是"是否通过验收"**。

上述`ppo/exploration.py`指重构前的历史实现；当前三种保留模式已并入`ppo/policy.py`，
前向走廊门并入`ppo/env.py`，仓库不再包含独立exploration模块。

### 3.9 2026-07-31职责重构验收

重构没有改变production reward、actor输入/结构、场景池、sampling weight、探索参数或
PPO数学。旧的algorithm/environment/geometry/privileged/vector-env/recorder/classification
职责已合并到§3.1所列五个PPO模块与`utils.py`；`detached_gru`、`--env_workers`、
`--reclassify_collisions`和重复的`actor_final.pth`保存路径已移除。

在BC、seed42、4 env、800 steps、batch1600、1 formal update下，逐步独立速度噪声、
全局时间相关速度噪声、前向走廊门控时间相关速度噪声分别做了重构前后配对。三组均满足：

- `episodes.jsonl`逐记录相同，collision/ordinary场景池与cache记录相同；
- actor与critic U1 checkpoint逐tensor完全相同；
- policy loss、value loss、KL、clip fraction、post-update explained variance逐位相同；
- gate fraction、temporal active fraction、block pair计数与block内残差误差逐位相同。

逐步独立速度噪声的五项基准保持
`0.0147010052 / 0.2674632408 / 0.0279283547 / 0.2820312455 / 0.6750825454`。
另外已验证四种critic均可构建、12-key actor可被真实`eval_multiagent.py`严格加载、
production `forkserver`可对新空cache分类、partial和identity-mismatch cache均fail closed。
六个临时短训run已移入系统回收站，不是后续证据入口。

同一本地文件系统快照已于2026-07-31完整镜像到
`haowei@192.168.2.209:/home/haowei/Documents/End2Race/`。镜像后dry-run为零差异；
远端编译、CLI、import、60项unittest和真实production U30 evaluator smoke均通过。
Codex/Claude项目memory已做并集合并，覆盖前备份位于
`/home/haowei/memory_backups/end2race_migration_20260731/`。

---

## 4. 记录、评估和面板合同

### 4.1 Run记录

每个run至少应有：

```text
run_config.json
collision_cache_info.json
collision_scenarios.json
ordinary_scenarios.json
metrics.jsonl
episodes.jsonl
checkpoints/critic_warmup.pt
checkpoints/actor_uXXXX.pth
checkpoints/critic_uXXXX.pt
```

常见fresh-start完成条件：`metrics.jsonl = 1 warm-up + N formal rows`、formal updates连续、
预期actor/critic checkpoint存在、所有数值finite、没有写入进程、eval面板完整。
Resume/extension需按自己的run config判断，不能只看行数。

`rollout_policy_update=k-1` 表示formal update k的rollout由actor k-1采集；
training rollout不能直接评价actor k。

### 4.2 Eval trace

新trace最后一行必须：

```text
terminal_post_step = true
action_applied = false
```

最后一行动作数组是占位值。0721旧trace没有terminal post-step row，旧碰撞标签以
`results_multi.json`为准。

不同run的同名checkpoint stem可能覆盖NPZ；多臂评估必须用唯一actor alias。

### 4.3 固定面板定义

| 面板 | 构成 | 用途 | 边界 |
|---|---|---|---|
| Austin600 | 当前50 circular starts × 3 racelines × 4 speeds，interval15 | 快速主指标、checkpoint轨迹 | 与训练起点物理距离近；不是独立地图留出 |
| near400 | U30未碰撞且minimum OBB clearance `[0,0.1]m`的held-out困难成功场景 | 检测新造碰撞和超车损失 | U30条件筛选，不代表自然发生率 |
| hard73 | U30 held-out collision中的interval15子集 | 检测困难工况专门化 | 高基率诊断，不能单独验收 |
| crossmap1800 | Hockenheim/Moscow/Nuerburgring各600 | 地图迁移 | corridor门设计后已参与设计 |

hard73、hard334、near400、完整21,600候选/标签及interval-15 fixed pool的规范输入已集中到
`post-trained/panels/heldout_hard_v1/`；该目录是跨实验复用输入，不是某个actor的评估输出。

**面板判定口径是 ego collision scope，不是 legacy。** 这条决定了 overtake 计数：
`ego` 下对手撞墙既不终止也不改写episode，只作为事件记录（`opp_wall_event_episode_count`）；
`legacy` 下它会终止episode并把结果标成 `opp-wall`。near400 上恰有3个这样的场景，用错口径
会把 `28/325` 读成 `28/322`。`scripts/evaluate_scenario_panel.py` 默认 `--collision-scope ego`，
manifest 逐包记录实际口径；跨包比较前必须确认两侧一致。

Austin headline collision包含 opp-wall；做actor责任归因时必须单列ego collision。
同场景二元比较必须报告：

```text
collision: removed / created + exact McNemar p
overtake:  lost / gained + exact McNemar p
```

多个checkpoint重复同一面板时，不能把所有行当独立样本。

完整600面板最低合同：600 unique scenarios、0 errors、有限数值、600 traces、
results/trace key-set一致、collision marker一致、terminal row语义正确。

2026-07-30清理后的保留评测有两种状态，不能混用：

- 前向走廊与ordinary异线高速重加权的canonical目录中，带`results_multi.json`的
  Austin600、near400和跨地图600包均满足上述完整合同。
- production 的 Austin 目录现在有三个完整包：`multiagents/`(600)、`near400/`(400)、
  `hard73/`(73)，后两个是2026-07-30用`scripts/evaluate_scenario_panel.py`重评补齐的
  fresh evaluation（manifest `status: fresh_evaluation`），与既有臂的
  `complete_trace_reconstruction`包不同源但同面板同口径，可直接配对。
- BC只有Austin目录保留正式`results_multi.json`；三张跨地图目录保留
  600条numeric trace和已固化总表，但缺逐episode聚合文件。条件白噪声/条件时间相关
  只保留228条共同诊断trace；全局时间相关另保留三张跨地图各600条trace。这些目录
  已明确标成diagnostic或trace-only，不能冒充完整正式评测。

---

## 5. 主要基线实验

### 5.1 0721受控实验

共同参数：seed42、6400 steps、batch12800、actor/critic epochs2/5、
LR `3e-6/3e-5/3e-4`、std `0.03/0.15`、target-KL off、同BC/cache。

20-update Austin600结论：

| 轴 | 被测设置 | 最终判断 |
|---|---|---|
| critic | independent GRU / privilege MLP / privilege GRU | privilege GRU最稳定且后段最低；保留 |
| batch | 12,800 / 25,600 / 51,200 | 大batch没有稳定收益；保持12,800 |
| clip | 0.10 / 0.15 / 0.20 | 20u内0.15最好；0.20后段改善，延长至30u后成为B |

Privilege GRU 的U15/U20碰撞总数都为14，但只共享8个：消除6、新造6、Jaccard0.40。
总数稳定不代表失败身份稳定。

### 5.2 B/U30不是单点幸运值

B晚期 Austin600：

```text
update      24  25  26  27  28  29  30
collision   18  16  14  14  13  13  14
overtake   350 358 361 365 363 364 366
```

U26-U30形成稳定低区，因此14不是从宽幅震荡中挑出的孤立尖峰。
near400在相同区间仍有较大策略轨迹变化，不能只凭一个near checkpoint比较方法。

---

## 6. Reward方向：核心判决

完整设计、公式、训练分量、配对结果和时机分析见 `ANALYSIS.md` §16。

### 6.1 判决表

| 方向 | 改了什么 | 关键结果 | 决策 |
|---|---|---|---|
| Post-pass q² LR1 | 新增超车后rear-closing直接penalty | Austin fishtail 4→8；总碰撞13→19 | 否决 |
| Post-pass q LR1 | q²→q | fishtail4→2但超车366→354；证据不足且有KPI代价 | 否决 |
| Post-pass q² LR3 | 只提高LR | fishtail4→7 | 否决 |
| L12 | risk纵向0.6→1.2 | 训练碰撞率下降、min gap增加；interval15净0且near超车显著下降 | 否决 |
| following-response | required relative deceleration + escalation离线候选 | 对OL1失败6/7，成功超车0/366误触；未训练 | 保持离线，未准入 |

### 6.2 为什么不采用Post-pass

门控能定位接近碰撞的超车后相位，但触发经常晚于因果转向/甩尾起势；它更多感知后果，
而不是在仍有操控权时提供信号。提高剂量、改q线性或提高LR没有稳定解决甩尾，并会新造
碰撞或损失超车。

不要把Post-pass写成potential shaping：它是直接行为penalty，没有terminal refund。

### 6.3 为什么不采用L12

L12没有新增reward项，只扩大risk longitudinal support。它在训练近距分布确实让actor留下
更多余量并减少碰撞，说明reward机制有效；但收益集中在interval8–12，interval15为
10消除/10新造、净0。held-out near-miss上超车显著下降。

准确结论是“有效但脱靶”，不是“potential shaping无效”。Production保持0.6。

### 6.4 Reward停止规则

- 不再扫Post-pass权重、cap、q幂次或LR。
- 不把L12设为默认，不继续扫1.2/2.0等clearance距离。
- 不把following-response写成已训练失败；它从未进入闭环PPO。
- **不引入 follow-teacher（masked imitation）辅助损失。** 它曾有一份完整参考实现并通过
  两项单测（含解析梯度有限差分），但从未接入：第二个目标违反单阶段PPO要求，需用户
  显式重新授权。两条独立理由说明它本身也不值得复活：BC 在 masked 的同线跟车状态里
  恰恰是最弱的（U30 的13个ego碰撞有8个在OL1），而它想买的持续减速能力已被合规的
  全局时间相关速度噪声实验拿到过。实现细节与依据见 `EXPERIMENTS.md` §5.3.6。
- 新 reward 候选先过 `scripts/screen_reward_candidate.py` 的离线筛查：它会拒绝带辅助
  目标、运行时 shield、特权/未来信息或改动 actor 输入的候选，并量化该 reward 增量经
  GAE 进入 PPO 后的**归一化**学习信号（绝对量级不是判据）。
- 只有任务分布、默认reward结构或可用状态信息发生实质变化，才允许重新预注册。

---

## 7. Collision pool与采样方向

完整结果见 `ANALYSIS.md` §17和§19。

### 7.1 Hard-neighbor

Boundary-aware流程从479 base collisions扩展到805，新增326条确认碰撞。
完整805池与479 control fresh-start训练45 updates：

```text
Austin U45 collision: 12 -> 17
U25+ mean:             13.0 -> 21.4
```

机制上它提高了困难邻域覆盖，但只告诉PPO“哪里多采样”，没有提供该状态应采取什么动作。
Production保持479；当前训练入口已经删除`--hard_neighbors`及比例采样，不再以
“关闭的实验flag”形式保留。

20%分层臂在同面板U35/U40/U45碰撞从base的`14/11/12`升至`27/20/19`，
同时超车下降，已否决。10%臂已训练45 updates但仅有U1--U20完整eval，实验结果仍是
“未定案”，不是已证明有效；用户已决定不再补评并退役整个hard-neighbor训练入口，
这是主动停止，不是把未知改写成负结果。

### 7.2 Ordinary150

把ordinary starts从50扩到150后：

```text
Austin collision 14 -> 35
near collision   28 -> 55
hard collision   54 -> 38
```

它同时增加多样性并把单场景重复覆盖降到约三分之一，所以只能否决整套
`ordinary_startpoint_count=150`，不能单独断言“多样性无用”。

### 7.3 Interval15 difficult pool

用冻结interval15 collision+near-miss池替换广域collision pool，默认L06 reward训练。
多个checkpoint配对均无显著主指标收益，只否决该池配置，不是L12证据。

### 7.4 Pool停止规则

- 默认479池保持不变。
- 不重跑完整805池或ordinary150。
- 不把hard73改善当production收益；它多次表现为分布专门化信号。
- 若未来重新做pool，必须保持其他变量、池规模/构成和role预算可解释。
- 当前源码不再提供ordinary150、外部fixed pool、hard-neighbor或actor-mismatch入口；
  需要重开时按`EXPERIMENTS.md`的退役接口合同重建到独立实验分支，不要临时拼回production。
- hard-neighbor和outcome-aware源码模块均已删除；当前源码没有调用方，也不影响
  全局时间相关速度噪声、前向走廊门控时间相关速度噪声或ordinary异线高速重加权。历史边界构建、cache schema和筛选算法已保存在
  `EXPERIMENTS.md`，保留的cache/模型产物不由production入口读取。

---

## 8. 速度探索：核心判决

完整实现、四臂、分层、checkpoint band和机制推导见 `ANALYSIS.md` §18–§19。

### 8.1 各臂定义

| 完整名称 | 机制 |
|---|---|
| Production逐步独立速度白噪声 | 每步独立采样，物理标准差0.15米/秒 |
| 条件白噪声放大 | 门外保持每步独立标准差0.15；旧required-deceleration门内改为每步独立标准差0.50，不做时间保持 |
| 全局时间相关速度噪声 | 不使用门控；全状态标准差0.15的同一个速度残差连续保持50步（0.5秒）后重采样 |
| 条件时间相关速度噪声（历史，源码入口已移除） | 门外每步独立标准差0.15；旧required-deceleration门触发时采样标准差0.25的残差并保持50步；门中途关闭也跑完时间块，episode reset清空 |
| 前向走廊门控时间相关速度噪声 | 前向same-corridor门内使用标准差0.15、保持50步的时间相关速度噪声 |
| ordinary异线高速重加权 | 前向走廊门控时间相关速度噪声保持不变，仅在ordinary角色内提高异线高速场景采样权重；不是reward或optimizer改动 |

条件白噪声放大和条件时间相关速度噪声共用旧门
`escalating_required_deceleration`：先以同走廊、前向净空和closing
time形成warning，再要求required relative deceleration持续增长并超过阈值；它是训练期
因果门，不使用未来碰撞信息。实测后期曝光很低：条件白噪声放大约0.15%，条件时间相关
速度噪声约0.51%。因此三组实验分别检验“危险状态内只放大白噪声”“只改变时间相关性”
和“在危险状态内同时改变幅度与时间相关性”。
三者均只改变训练期采样；确定性evaluation和部署actor接口不变。

ordinary异线高速重加权把ordinary场景拆成
`same_line`、`offline_fast`（异线且opponent speed scale >=0.7）和`offline_slow`三组；
YAML中`ordinary_offline_fast_fraction: 0.6`把三组自然的33.3%/33.3%/33.3%改为
33.3%/60.0%/6.7%，固定same-line份额、也不改变可达场景集合。目标是让前向走廊门控
时间相关速度噪声模型学会区分
同线跟车和异线高速超车。它收复了Austin/跨图超车并在跨图形成54–57碰撞、
1146–1155超车，但near400碰撞恶化到63–77，故已否决。

### 8.2 固定面板总表

单元格为 `collision / overtake`：

| 实验 | Austin600 | near400 | hard73 | crossmap1800 | 决策 |
|---|---:|---:|---:|---:|---|
| **Production逐步独立速度白噪声** | **14 / 366** | **28 / 325** | 54 / 12 | 80 / 1142 | production |
| 条件白噪声放大 | 15 / 355 | 41 / 299 | 49 / 25 | — | 否决 |
| 全局时间相关速度噪声 | 16 / 359 | 53 / 286 | 19 / 31 | 64 / 1144 | 否决 |
| 条件时间相关速度噪声 | 15 / 342 | 40 / 295 | 46 / 17 | — | 否决 |
| 前向走廊门控时间相关速度噪声、2米门宽 | 20 / 338 | 35 / 286 | — | 46 / 1122 | 否决 |
| 前向走廊门控时间相关速度噪声、1米门宽 | 21 / 350 | — | — | 76 / 1127 | 否决 |
| 前向走廊门控时间相关速度噪声、45 updates | 17–20 / 332–345 | 33–37 / 271–288 | — | 43–46 / 1117–1134 | 否决 |
| ordinary异线高速重加权、比例0.6 | 16–21 / 363–370 | 63–77 / 291–306 | — | 54–57 / 1146–1155 | 否决 |

Canonical模型入口：

```text
post-trained/ppo_conditional_white_speed_noise_0p50/update30/actor.pth
post-trained/ppo_global_temporal_speed_noise_0p15_hold50steps/update30/actor.pth
post-trained/ppo_conditional_temporal_speed_noise_0p25_hold50steps/update30/actor.pth
```

每条实验的U1--U30 actor/critic、训练records和场景池均已按GUIDE路径规范归档。

对应评估遗留物也已按完整实验名整理：

```text
eval_results/ppo_conditional_white_speed_noise_0p50/update30/Austin/diagnostic_common228/
eval_results/ppo_global_temporal_speed_noise_0p15_hold50steps/update30/Austin/diagnostic_common228/
eval_results/ppo_conditional_temporal_speed_noise_0p25_hold50steps/update30/Austin/diagnostic_common228/
eval_results/ppo_global_temporal_speed_noise_0p15_hold50steps/update30/{Hockenheim,MoscowRaceway,Nuerburgring}/multiagents/
```

前三个目录只有228条共同诊断trace，不能作为Austin600。三张跨地图目录各有600条trace，
但缺`results_multi.json`；对应`eval_manifest.json`已标记为不完整，不能当正式结果包。

### 8.3 学到的能力是真实的

全局时间相关速度噪声模型的跨地图same-line碰撞：

```text
Production 66 -> 全局时间相关速度噪声 33
```

跨地图总碰撞 `80 -> 64`，而overtake `1142 -> 1144`。
该模型把production模型的8个Austin OL1目标碰撞全部消除；碰撞前速度差中位从约
+0.733降到-0.013米/秒。
因此时间相关探索确实让actor学会了持续条件减速。

前向走廊门控时间相关速度噪声把跨地图碰撞进一步降至46，证明same-corridor gate能放大
该机制。

### 8.4 为什么仍然失败

全局时间相关速度噪声模型的跨地图变化：

```text
same-line: 66 -> 33
raceline0:  8 -> 13
raceline2:  6 -> 18
```

异线伤害主要集中在off-line × speed0.8。轨迹显示异线不是“错误减速”，而是整体速度指令
略升、余量变小。共享actor参数把same-line能力迁移成off-line副作用。

前向走廊门控时间相关速度噪声虽然跨图安全显著改善，却在Austin和near损失超车或新造
碰撞；门宽收窄到1米后直接失去
same-line收益，说明机制依赖足够same-line曝光。

45u与U30的结果区间重合，延长训练没有继续收敛出Pareto改进。

ordinary异线高速重加权收回Austin/跨图超车，且跨地图第一次双轴优于production；
但near碰撞升到63–77，
新造伤害集中在异线。它只是把保守度从一个regime搬到另一个regime。

### 8.5 其他探索幅度实验

| 配置 | Austin | near | hard | 结论 |
|---|---:|---:|---:|---|
| speed std0.25 | 31 | 52 | 27 | hard专门化，主/KPI全面恶化 |
| speed std0.50 | 15–20 | 41–42 | 31–34 | hard改善，near超车显著下降 |
| anneal0.40→0.15 | 44 | 68 | 32 | 未回收到baseline工作点 |

全局提高σ确实能提高余量和困难场景安全，但产生全局噪声税/专门化，不能作为production。

### 8.6 Exploration停止规则

- 不再扫speed std、退火schedule、corridor gap、门控覆盖率或ordinary异线高速权重。
- 不延长前向走廊门控时间相关速度噪声超过45 updates。
- hold只实际训练过K50；K10/K25未测，不能写成已证伪。但它们仍是同类强度旋钮，
  在没有新机制前先验低。
- 不用crossmap `80→46`单独给前向走廊门控时间相关速度噪声翻案：该面板已参与
  corridor设计，且Austin/near失败。
- 重开必须引入能隔离same-line能力与off-line副作用的新结构或新任务分布，而不是再调剂量。

---

## 9. 可观测性与Oracle：核心结论

完整探针、cohort R²、oracle参数、共享动作库和制动曲面见 `ANALYSIS.md` §20。

### 9.1 Actor现有表征包含关键信息

冻结B/U30 actor的线性探针显示：

- GRU hidden对relative closing speed显著优于单帧observation；
- fishtail cohort的yaw/slip信息在hidden中可读；
- 失败状态解码精度不是完美，且不同collision cohort差异很大；
- 聚合collision R²会发生composition/Simpson误导，不能直接判定表征充分或不足。

所以“actor完全看不见危险状态”和“actor表征已经完全充分”都没有被证明。

### 9.2 动作接口不是物理上限

Austin13特权oracle使用未来碰撞时刻和场景特定CEM：

```text
13/13 collision可救
10/13同时保持overtake
```

如果把其他587场景假设不变，会得到反事实 `0 collision / 376 overtake`。
这只是可达性见证，不是actor闭环成绩；把动作迁移到matched safe controls会新造3次碰撞、
损失2次超车，证明固定模板不能无条件部署。

### 9.3 共享动作库

13条冻结schedule在334个held-out hard failures上：

```text
334/334找到无碰撞候选
239/334同时保持overtake
```

简单固定schedule本身已救68/72测试场景；observation/hidden ranking没有显著超过它。
这不等于“state information无用”，而是候选库任务接近饱和、排名没有头寸。

### 9.4 局部梯度并非平台

334场景持续1.5秒制动扫描：

| speed delta | lead0.5s | lead1.0s | lead1.8s |
|---|---:|---:|---:|
| -0.15m/s | 17.4% | 30.5% | 46.4% |
| -0.30m/s | 30.2% | 56.3% | 67.4% |
| -0.60m/s | 58.7% | 74.9% | 87.1% |
| -1.00m/s | 71.9% | 88.6% | 91.9% |
| -3.00m/s | 85.6% | 94.0% | 94.6% |

1σ持续制动已经有明显救援率和return增益，因此“约20σ所以PPO邻域无梯度”及
“return是纯阶跃平台”均被证伪。问题是状态条件化、长时间动作协调和跨regime共享参数，
不是简单没有局部信号。

### 9.5 禁止越界

- 不把oracle写成PPO模型成绩。
- 不引入runtime shield、未来碰撞触发器或模型输出后处理。
- Oracle imitation、辅助动作loss、DAgger、actor输入扩展都没有被测试，
  不能写成成功或失败。
- 不重复全量制动扫描，除非动作边界、任务分布或episode horizon改变。

### 9.6 Regime可分性与梯度冲突（2026-08-05）

冻结production U30与前向走廊门控时间相关速度噪声U30，不训练actor。按既有配对事件规则
固定`S=54 / O=69 / N=59`；O从预审计70校正为69，因为唯一旧`opp-wall`提前终止场景按
当前ego-scope续跑后production也发生ego碰撞，不满足production成功。

五路闭环counterfactual为176/182条场景给出稳定anchor；6条outcome翻转场景不进入表征标签。
动作收益标签不是S/O/N别名：有效集合中持续减速有利61、完整production动作有利37、
steering修正必要20，另有91条在固定1.5秒干预库下未解决；标签可重叠。

- 线性探针没有建立hidden的稳定优势。全部有效场景的减速标签中，group-held-out AUROC
  中位数为observation `0.811`、hidden `0.695`；O+N的production动作标签为`0.614/0.603`，
  分组切分区间很宽。hidden的窗口预测更平稳，但不能据此判定输入充分或缺失。
- 人工成功动作偏好梯度在全部cohort上的S-O/S-N output-head cosine为`-0.967/-0.971`；
  只保留full-window counterfactual真正改善结局的`36/20/17`条S/O/N后仍为
  `-0.962/-0.959`，而O-N为`+0.998`。这描述固定actor上的局部动作偏好，不是历史PPO、
  Adam或fresh PPO rollout梯度；§9.7已经否决将它外推成稳定优化冲突。

因此本节不再提供“隔离same-line与O/N输出映射”的训练准入依据。保持361D输入、现有reward
和critic；Advantage/credit审计本轮没有触发，也没有证明GAE正确。只有新的任务目标或新的
fresh PPO证据先稳定识别可干预机制，才重新设计训练控制。

本轮按用户要求只保留Markdown结果与`.agents`权威记录；一次性脚本和分析产物已删除。
不得仅为“复核已有结论”重新生成analysis结果树或复活这些脚本；只有输入模型、面板、
动作合同改变，或用户明确要求重新审计时才重建。

### 9.7 Fresh PPO首步梯度复核（2026-08-06）

冻结production与前向走廊门控时间相关速度噪声两条轨迹的U1/U10/U20/U30 actor/critic，
每格fresh采102400 transition，不执行optimizer。每个K比较production+baseline、走廊
checkpoint+baseline，以及同一走廊checkpoint+实际走廊探索。

所有12格通过transition覆盖、样本门、collection-equivalent ratio和分组梯度重构。实际走廊
探索格K10/K20/K30的same-line vs offline-fast output-head aggregate cosine分别为
`+0.663/+0.622/-0.186`，负minibatch仅`4/8、2/8、4/8`；没有稳定负冲突。走廊checkpoint
切回baseline探索时为`-0.722/+0.841/-0.819`，但对应负minibatch为`7/8、4/8、2/8`，同样
不满足跨checkpoint稳定性。K1只作描述，不作零差sanity。

结论：§9.6的`-0.96`是人工成功动作偏好梯度的有效局部描述，但不能外推成真实PPO rollout
上的稳定更新冲突。当前不实现或训练naive symmetric gradient projection；只有能在fresh
PPO数据上先稳定复现冲突、并证明投影后same-line/offline-fast一阶方向均不受损的新设计才可重开。

同日评测基础复核使用冻结production U30和三张各600场景的固定跨地图panel。CUDA逐图为
Hockenheim `26/356`、MoscowRaceway `32/385`、Nuerburgring `22/401`，合计`80/1142`，
same-line/off-line碰撞`66/14`，精确对上历史权威数。CPU逐图为`26/355、31/386、21/401`，
合计`78/1142`；因此device属于评测协议，不允许混用。该复核没有改变§9.7判决，也没有开启训练。

### 9.8 Collision/ordinary role配比假说复核（2026-08-06）

有人据fresh U30 rollout的episode return提出：走廊时间相关探索在collision role净赚、在
ordinary role净亏，当前50/50混合把它算成正收益；把collision role降到25%可能消除自然
分布副作用。复核确认这只是**值得记录但尚未成立的描述性假说**，当前不启动该训练臂：

- fresh U30端点中，走廊臂相对production在collision/ordinary role的episode return差为
  `+0.3220/-0.2022`，线性break-even约`0.386`；门控选择性也成立，same-line transition
  激活率为collision `62.82%`、ordinary `61.80%`，off-line低于`0.1%`；
- 但fresh rollout两臂完成episode数不同（production/走廊为`142/134`），按env内完成顺序
  只有`132`个可比较位置且仅`34`个scenario identity相同，不能称为严格配对；
- 同样的fresh端点在U10/U20上两个role都改善，只有U30出现交易；历史正式rollout在
  U27--U30/U42--U45的break-even随update从约`0.019`到`0.878`摆动，并在U42/U44两个role
  同时改善，不存在稳定的`0.412`临界点；
- episode return的线性混合不是PPO的transition-weighted advantage目标，不能据此预测改变
  sampler后的非线性重训练；
- 当前50/50角色合同贯穿scheduler奇偶rank、logical seed、recurrent minibatch、critic warmup
  和role telemetry。改成25%会同时改变数据分布与minibatch合同，不是一个干净的小参数实验。

因此不把role配比写成已识别根因，也不运行25% collision-role臂。只有在seed 42、多个预先
固定checkpoint上用scenario-identity匹配或等价固定队列证明role收益符号稳定，并给出不改变
recurrent minibatch语义的单变量实现，才可重开。用户不要求多seed，不为此增加seed sweep。
本轮transition-level临时张量已经按清理决定删除；这不推翻已写入本文的结论，但意味着未来
若满足重开条件，必须fresh采样新的transition/advantage数据，不能把重开描述为零成本历史重放。

### 9.9 四图碰撞身份与接触几何诊断（2026-08-06，无训练）

使用BC、前向走廊时间相关速度探索U44和ordinary异线高速重加权U30的四图同场景结果与
numeric traces复算；没有训练、没有新checkpoint。Production U30跨地图完整trace未保留，
只在regime总量表中使用已记录headline，不参与本节碰撞角和身份Venn复算。

| 模型 | same-line | off-line | 四图总collision | 四图overtake |
|---|---:|---:|---:|---:|
| BC | 64 | 65 | 129 | 1445 |
| Production U30（headline） | 74 | 20 | 94 | 1508 |
| 前向走廊时间相关速度探索U44 | **21** | 41 | **62** | 1478 |
| ordinary异线高速重加权U30 | 35 | 38 | 73 | **1516** |

U44相对BC消除94次、继承35次并新造27次collision；重加权U30消除84次、继承45次并
新造28次。BC的129次collision中仅18个场景在三者上都碰撞；两个PPO模型合计新造48个
不同场景，仅7个被两者共同新造。结论是**失败身份迁移**，不是剩余一小批不可解硬例。

新增ego-opponent碰撞的几何高度一致：U44为23次，其中21次相对yaw不超过30度、21次对手
位于ego侧方或后方，相对yaw中位`4.54°`；重加权U30为25次，全部相对yaw不超过30度、22次
位于侧方或后方，中位`4.55°`。两臂新增ego-wall仅`4/3`次。这里的方位是碰撞帧对手中心在
ego车体系中的bearing，不是接触法向；能支持的结论是新增失败以平行侧擦/超车后侧后接触为主，
不能把它写成精确撞击角。

按三个粗regime事后拼接最佳现有actor，只能到约`58 collision / 1527 overtake`，仍过不了
`<40`目标；按每个episode未来结局事后选择U44或重加权U30可到`25/1557`。后者不可部署，
但证明目标行为已分散存在于同结构actor中。当前最精确的机制判断是：单个actor没有学会在
同一regime内部按“对手在前/并排/刚落后、净空是否收缩”选择正确的纵横向动作；精确优化器
根因仍未知，§9.7已经否决把它简化成稳定output-head梯度冲突。

停止/重开规则：不据此直接训练新臂；不再用粗regime权重或探索剂量旋钮追`<40/>1500`。
只有Austin-only、按startpoint分组的离线诊断能在仍有操控权的提前窗口，用actor可见历史
稳定区分“成功通过”与“未来平行侧后碰撞”，才重开一个保持actor结构、无runtime shield、
不使用测试地图训练的条件策略学习设计；若不能区分，应把当前361D观测合同视为候选瓶颈，
而不是继续扫PPO参数。

### 9.10 Phase-spillover与Pressure-conditioning最小诊断（2026-08-07，无训练）

仅用BC与前向走廊门控时间相关速度探索U44的匹配Austin600 traces，完成外部审查要求的两个
最小筛查，没有训练、代码改动或新checkpoint：

- U44相对BC新造的6个ego-opponent collision在碰撞前1.5s内，按当前2m走廊门和50步hold
  精确重建均为`0/6`出现gate、active block或跨相位block；24个同raceline/speed且两模型都
  安全的近起点对照同为`0/24`。正向对照中，U44修复的9个same-line BC collision重建出
  28个block、1009个active step，`5/9`有gate关闭后的active step，说明零结果不是重建失效。
- 三个提前时刻上，新造collision的18个opponent-bearing关键回波全部在LiDAR FOV内；pressure
  中位`0.539`、`|dp/dx|`中位`0.311/m`，没有一个低于`0.05`。相邻5-beam共90个样本的中位为
  `0.404/0.300m^-1`，同样零低值；没有出现失败特异的pressure饱和或灵敏度塌缩。

**判决：phase-bounded和trainable pressure都不准入正式训练。** 前者在实际Austin新造碰撞
窗口没有block可截断，后者没有conditioning瓶颈证据；不再做shadow gradient、LR、hold或
gate扫描。只有fresh训练分布证明跨相位block在匹配失败窗口富集，或新的匹配Austin数据证明
关键LiDAR回波被BC pressure明显压平，才分别重开。完整cohort、数值和证据边界见
`ANALYSIS.md`对应专题。

### 9.11 Interaction-phase早期可分性最小诊断（2026-08-07，无训练）

外部审查提出的完整`observability + action response + PPO credit`联合审计方向合理，但一次
捆绑四类问题不符合最小实验原则。本轮只执行其第一个准入门：在U44 Austin near400开发
面板上，用30个平行侧/后ego-opponent collision和288个安全overtake构造同opponent raceline、
同speed、同当前interaction phase的最近匹配对照；按50-waypoint startpoint sector做5折，
每折固定6个collision。只用固定linear ridge，不扫超参、不训练actor、不运行闭环响应面。

| 距事件 | 样本数（collision） | geometry AUROC | raw LiDAR+speed | frozen pressure+speed | GRU hidden |
|---|---:|---:|---:|---:|---:|
| 1.5s | 132（30） | 0.487 | 0.627 | 0.606 | 0.529 |
| 1.0s | 141（30） | 0.556 | 0.690 | **0.773** | 0.617 |
| 0.5s | 144（30） | 0.667 | 0.572 | 0.598 | 0.580 |

唯一较高的1.0s pressure结果分组范围仍为`0.538--1.000`；其余actor-visible结果没有跨时间
稳定，GRU hidden三个时刻均不超过`0.617`。0.5s时geometry本身已达`0.667`，不能把晚期
可分性归因于actor学出了可靠危险表征。GRU从episode起点按评估器真实首帧速度合同逐步重放，
参与样本的最大raw-action误差为0；此前用reset后0速替代raceline初始速度的临时结果作废。

**判决：不继续动作响应、PPO-credit或side-phase steering正式训练。** 当前证据最多支持
“1.0s pressure可能含有局部信号”，不支持“在仍有操控权的早期窗口稳定识别未来平行侧后
碰撞”。该负结果不证明361D观测永久不可用，也不否决未来在更多独立Austin正例上复核；只有
预先固定的actor-visible表征在startpoint分组下稳定跨fold、跨提前时刻通过，才重开联合审计。
完整设计、fold范围和证据边界见`ANALYSIS.md` §30。

### 9.12 真值几何/速率对当前策略失败的线性早期可分性预检（2026-08-07，无训练）

用户提出：粗旋钮（reward、探索剂量、采样比例、训练长度）在纯PPO内基本到头，但"改变GRU
如何形成状态表征"仍未验证，因此考虑训练期辅助表征损失——只用Austin当前时刻pre-action
物理真值，辅助头训练时存在、部署前删除，actor保持12-key契约，准确命名为
"PPO + 训练期辅助表征学习"。用户给出的是**条件授权**：先做最小预检，通过才训练。

本轮先检验了一个更窄的命题：**在U44当前策略产生的轨迹上，privileged真值几何与一阶速率
经固定线性ridge，能否在仍有操控权的提前窗口预测未来的平行侧/后碰撞。** 这不是原预检，
也**不是信息论上界**——测得的只是固定线性读出可抽取的量，属于这些真值特征信息含量的下界；
GRU、辅助头与actor都是非线性模型。

cohort独立重建后与§30.3完全一致（37 collision / 288 overtake，接触几何`23/7/4/3`，主正例
30）；`G5`复现得到`0.490/0.589/0.692`对§30记录的`0.487/0.556/0.667`。

| 特征集 | 1.5s | 1.0s | 0.75s | 0.5s | 0.25s |
|---|---:|---:|---:|---:|---:|
| `GEO static`真值 | 0.418 | 0.559 | 0.674 | 0.582 | 0.825 |
| `RATE only`真值 | 0.567 | 0.604 | 0.535 | 0.670 | **0.481** |
| `GEO+RATE`（合并） | 0.429 | **0.725** | 0.667 | 0.656 | **0.861** |

预注册准入门为"中位AUROC ≥0.75、fold下界 >0.6，且位于仍有操控权的窗口"。1.5--0.5s四个
时刻全部不通过（最好的1.0s为`0.725`、fold下界仅`0.361`）；唯一通过的0.25s约相当于接触前
`1.4m`，已在操控权边界。`lambda`取`0.1/1.0/10.0`形态不变；200次标签置换null的95分位为
1.0s `0.659`、0.25s `0.630`，所以这两点信号真实（`p<=0.005`），但不足以支撑逐episode判别。

**判决：本轮不直接启动`CT-v2 + 辅助表征loss`的完整训练；辅助表征方向未被否决。**
准确成立的只有三条：该量族在此预测任务上没有通过本轮预注册门；closing rate无决定性独立
增益；因此不应据此直接开训。

**以下推论不成立，引用本节时必须一并引用：** 本轮不是表征学习上界（线性读出≠信息含量）；
辅助目标的用途是让PPO更容易依据当前几何选动作，而不是部署未来碰撞分类器，因此"对当前
策略失败预测力弱"不蕴含"对改进策略无价值"；闭环轨迹属性与早期状态属性不互斥，且本轮未
覆盖ego指令历史、yaw rate/slip与横向动态、墙面余量与赛道结构、对手planner意图以及GRU
更长历史的非线性交互；30个正例、每折6个的"fold下界"是小样本fold最小值，不是统计置信
下界，该门槛设计不应照抄。

**下一步仍回到原最小预检，用途限定为：** 比较raw/frozen pressure与GRU hidden对**当前
连续几何**的解码能力，指标用`R^2`/`MAE`而非未来collision；必须带滞后对照（hidden解码
`t-0.1/0.2/0.3s`同一量，排除"只是时间平滑"）、维度匹配对照（raw 361D vs hidden 1680D，
并报train/test差）和按startpoint分组的十万量级样本。若显示真实内部表征缺口，只授权**一次**
`PPO + auxiliary representation loss`训练，不保证最终安全收益，且首次不叠加reference KL、
行为锚或其他变量。完整设计、fold范围、稳健性检验和证据边界见`ANALYSIS.md` §31。

### 9.13 当前交互几何的表征缺口预检（2026-08-07，无训练）

执行§31.6规定的预检，判据拟合前写死：判定存在真实内部表征缺口须同时满足输入侧`R^2>=0.5`、
`hidden R^2 <= 输入 - 0.25`、5折范围不重叠、不能被最佳滞后解释、且在维度匹配读出下仍存在。

固定控制：U44、Austin near400的400条trace、seed 42、CUDA逐步回放、按50-waypoint startpoint
sector分组5折、ridge且`lambda`由训练折内分组内层CV选出。**回放合同校验：全部400条trace、
所有`action_applied`行的raw action最大绝对误差为`0`**（首帧speed为ego raceline起点参考速度
的`0.9x`，取`load_raceline_waypoints`第4列；早期误用raceline CSV第3列即heading，产生`5.45`
误差的版本已作废）。样本`31,345`个、24个sector、9个连续目标（对齐P20物理量，保留物理单位）。

维度匹配读出（**fold-local** PCA K=256，每个outer fold内只用训练sector拟合均值与主成分
方向再变换test sector）分组5折test `R^2`：

| 目标 | raw+speed | pressure+speed | GRU hidden |
|---|---:|---:|---:|
| `delta_s` | 0.212 | 0.296 | **0.703** |
| `obb_lon_clearance` | 0.146 | 0.316 | **0.553** |
| `relative_long_velocity` | 0.260 | 0.314 | **0.607** |
| `relative_lat_velocity` | 0.084 | 0.114 | **0.465** |
| `wall_clearance` | 0.066 | 0.229 | **0.516** |
| `left_margin` | 0.497 | 0.598 | **0.731** |
| `right_margin` | 0.537 | 0.603 | **0.705** |

**hidden在9/9目标上严格优于两种输入**，K=256优势`0.102--0.407`、K=64优势`0.086--0.340`。
滞后对照（fold-local PCA-256解码`t/-0.1/-0.2/-0.3s`）中9个目标有8个在lag 0最优（唯一例外
`relative_lat_velocity`在-0.3s高`0.018`），所以hidden对准的是**当前**几何而非被平滑成过去。

初版控制在全部31,345样本上拟合PCA后才做分组CV，test sector参与了均值与主成分估计，属于
数据泄漏；已全部改为fold-local重算。修正前后差异`<0.01`（如`delta_s` `0.699->0.703`），
结论不变，但初版数值不得再引用。

**判决：预注册条件未满足且符号相反，不启动`PPO + auxiliary representation loss`训练。**
用户假说中"信息可能存在于输入却没有稳定进入内部表征"被本预检否定——GRU hidden对当前交互
几何的线性可解码性不低于、且维度匹配下一致高于actor自身361D输入，把这些几何量压进hidden
缺乏可指望的增量。

**边界：** 不能说这些量被编码得"好"（fold-local维度匹配下hidden的`R^2`为K=64
`0.158--0.671`、K=256 `0.272--0.731`）；不能说辅助表征学习整体
无效（本节只否定该量族的缺口前提，未覆盖其他量族或正则化/优化通路）；线性可解码性仍不等于
信息含量，但这里方向对hidden有利，所以"hidden丢了信息"在**本可以支持它的同一口径下**没有
成立。附带独立结论：冻结BC pressure在全维读出中9/9目标优于raw LiDAR；fold-local维度匹配
K=64与K=256均为8/9，唯一例外是`relative_lateral`。这与§29不解冻`k`的决定一致。

**方法学警告（对后续所有probe有效）：不得在未做维度匹配的情况下比较361D与1680D读出；
任何降维必须在fold内拟合。** 同一数据在朴素固定`lambda`、内层选`lambda`、PCA维度匹配三种
设定下给出了三种相反结论；PCA的fold-local与全样本拟合此次差异`<0.01`，但不可预先假定无害。
完整设计、逐折数值、控制A/B与证据边界见`ANALYSIS.md` §32。

### 9.14 Round Z0：U42--U45等权checkpoint平均（2026-08-08，无训练）

固定U42--U45四个相邻actor等权平均，浮点tensor以float64按update顺序累加后cast回原dtype，
非浮点要求逐值相同；输出通过12-key、finite和fresh strict-load。四图CUDA/ego-scope结果：

| 地图 | BC | U44 | 等权平均 |
|---|---:|---:|---:|
| Austin | `33/339` | `18/344` | `18/342` |
| Hockenheim | `27/343` | `16/347` | `19/341` |
| MoscowRaceway | `43/373` | `15/390` | `16/386` |
| Nuerburgring | `26/390` | `13/397` | `14/396` |
| 四图 | `129/1445` | `62/1478` | **`67/1465`** |

相对U44，collision removed/created为`13/18`（`p=0.473`），overtake lost/gained为`24/11`
（`p=0.0410`）；相对BC为`94/32`（`p=2.91e-8`）和`49/69`（`p=0.0798`）。near400为
`32/285`，相对U44 collision `13/8`（`p=0.383`）、overtake `15/12`（`p=0.701`）。

机制假说同样失败：U42--U45共同created core-14中有13个仍被平均actor保留，而平均actor共
created 32个，另有19个core外失败；权重平均没有把闭环行为变成失败集合交集。Round Z0关闭，
不做三点、非等权、EMA/SWA或事后权重选择；production保持U30。完整逐图配对、完整性合同、
证据边界和重开规则见`ANALYSIS.md` §33。

### 9.15 BC-safe anchoring Gate A（2026-08-08，无训练）

Gate A只检验Austin训练侧是否存在可供后续反事实接管审计的稳定锚定对象。868条困难场景的
91个起点在任何actor评估前按预注册identity hash冻结：development为718条/72起点，validation
为150条/19起点，两侧起点与scenario key零交集。只对development运行canonical BC和U42--U45
五次fresh deterministic CUDA replay，validation没有运行actor评估。

| actor | ego collision | overtake | follow |
|---|---:|---:|---:|
| BC | 308 | 318 | 92 |
| U42 | 191 | 377 | 150 |
| U43 | 187 | 385 | 146 |
| U44 | 214 | 373 | 131 |
| U45 | 202 | 377 | 139 |

先按BC定义得到308条raceline0/2 safe overtake；U44单点回归46条，要求U44回归且四个晚期
checkpoint至少3个回归后，固定主cohort为28条。它覆盖21个起点，raceline0/2=`20/8`，
U44 collision/lost-overtake=`19/9`；速度`0.45/0.50/0.55/0.60/0.65/0.70/0.75/0.80/0.85`
分别为`1/1/0/3/2/2/5/7/7`，其中0.55是真实零计数而非筛后删除。六条Gate A门全部通过，且
`C=19、L=9`均达到Gate B前置的每层8条。

五臂共3,590条trace全部通过result/panel/trace key一致、七字段scenario identity、数组finite与
逐行对齐、collision marker和terminal row合同，0 error且无partial结果。**机制结论只到
“稳定对象存在”**：单点46条中有18条未进入共识，说明临界身份波动真实存在，但28条稳定对象
仍足以继续；它不证明BC动作从U44已访问状态接管能救回碰撞和丢失超车。

判决：Gate A通过；28条development cohort冻结，不得退回U44单点集合或从validation补样本。
下一合法节点是Gate B branch-0精确重放与BC反事实接管；按用户要求当前先停在Gate A汇报，
Gate B尚未启动。完整定义、质量检查、分层、证据边界和停止规则见`ANALYSIS.md` §34。

### 9.16 BC-safe anchoring Gate B（2026-08-08，无训练）

28条Gate A cohort均满足固定窗口至少50个动作步，另从development无放回匹配28条BC/U44共同
安全overtake controls，短窗口剔除0条。12个persistent CUDA worker各只加载一次BC/U44，依次
完成56条branch 0与三组56条干预；执行层加速没有减少分支、字段或改变阈值。

branch 0在56/56上精确复现保存U44：ego raw/executed action、opponent action、两车pose/speed、
双360D LiDAR全局最大误差均为0，outcome、首次collision、长度、marker与terminal逐条相同。
四branch共224条trace的key、finite、对齐、窗口、action source与执行动作合同全部通过。

| 正式完整BC分层 | 实测 | 门 | 判决 |
|---|---:|---:|---|
| C碰撞救回 | `10/19`，10条全为overtake | 至少10；救回中至少80%超车 | 通过 |
| C覆盖 | 9起点，raceline0/2=`8/2` | 至少2起点，两线各2 | 通过 |
| L恢复超车 | **`0/9`** | 至少8 | **失败** |
| L新碰撞 | `0` | 至多0 | 通过 |
| controls | 0新碰撞，**2/28丢超车** | 各至多1 | **失败** |

组件诊断：steering-only在C/L/control恢复或保有overtake为`7/19、1/9、28/28`；speed-only为
`5/19、1/9、26/28`且control新造1次collision。它们不能事后替代完整二维branch。机制上BC
确有局部碰撞修复能力，但不具备当前方法要求的统一状态条件teacher能力：它没有恢复任何L，
还破坏两个safe control的progress结果。

判决：Gate B科学失败，BC functional regularization方向关闭。不得删L、放宽control门、换窗口
或改成单分量重试；不生成anchor dataset，不进入Gate C/D或formal训练。若复用已验证branch
engine研究动作库/first-action preference，必须独立预注册并由用户明确改选。完整证据与边界见
`ANALYSIS.md` §35。

### 9.17 Round Z2：反事实动作存在性与可排序性（2026-08-08，无actor更新）

只用Austin Gate A development且事件前至少有150步的456条：109 inherited collision、46
created collision、13 lost-overtake、63 inherited-follow诊断、225 safe controls。456/456 no-op
逐动作、opponent action、pose/speed、双LiDAR、marker与terminal全局最大误差0；early/late各
12个固定residual共10,944条branch，合计11,400条trace、8,185,913行全部通过合同。

| Gate | early `[event-150,event-100)` | late `[event-100,event-50)` | 判决 |
|---|---:|---:|---|
| oracle inherited/created/lost | `93/109、44/46、10/13` | `96/109、45/46、7/13` | 两侧存在性通过 |
| OOF head inherited/created/lost | `19/109、11/46、1/13` | `34/109、17/46、0/13` | 两侧rankability失败 |
| OOF head controls collision/loss | `15/17` | `3/5` | early失败；late通过 |
| OOF target success vs fixed | `31 vs 0(noop)` | `51 vs 79` | late状态条件选择不如固定动作 |

相对no-op，early head collision removed/created=`41/19`（exact `p=0.00622`）、overtake
lost/gained=`17/36`（`p=0.0127`）；late head为`56/7`（`p=1.36e-10`）与`5/54`
（`p=1.91e-11`）。这些净改善只发生在按未来event选择的训练侧prefix，不能当作部署actor结果；
head仍未过lost-overtake和状态条件增益门。late固定动作甚至为`79/17`与`5/83`，说明瓶颈不是
局部动作不存在，而是何时、对谁选择动作及progress保持。

判决：关闭当前fixed-library action-conditioned controllability和first-action preference形式；
不得改动作库、阈值或只报oracle上界重试。prefix-reset没有出现“early通过、late失败”的旧程序
触发条件，故当时未准入，但该条件不是对“重置后PPO能否利用更密梯度”的直接检验，不能写成
科学否决；Residual/MoE违反当前12-key兼容边界，架构本身也未测。Constrained PPO同样未被本
Gate检验。完整证据与边界见`ANALYSIS.md` §36。

### 9.18 Round Z3：collision-only BC anchoring validation（2026-08-08，inconclusive）

本轮是用户重新区分“科学否决”与“项目关闭”后的第一个独立验证：只保留Gate B
中有正面数据的collision层，删除L-anchor，但不改canonical BC teacher、稳定回归定义、
1.5秒窗口和control匹配。任何validation outcome可见前已冻结数据、阈值和停止规则。

| 阶段 | 实测 | 判决 |
|---|---:|---|
| 五actor validation screen | 5 x 150 = 750条，0 error，750 traces | 通过 |
| 稳定collision cohort | 7条，6起点，raceline0/2=`6/1` | 通过样本门 |
| control support | `r0/s0.60: 2 source/0 control`；`r0/s0.85: 3/2` | **无解** |
| branch0/full-BC/formal training | `0/0/0` | 未运行 |

该分层缺口意味着不存在同opponent raceline、同speed、无放回的1:1对照，不是匹配算法
选错。放宽speed、有放回、从development补样或删source都是看数后改方法，因此本轮不做。

判决：只能写`inconclusive`；collision-only变体仍未被科学否决，但当前也不准入训练。
只有一个在actor outcome前冻结、起点独立且预先保证同raceline/speed control support的新
Austin panel才能重开；不在当前validation上修改匹配重跑。完整边界见`ANALYSIS.md` §37。

### 9.19 Round Z4-A：representation-changing action-response（2026-08-08，无actor更新）

本轮严格补测Round Z2没有触及的“新增可训练历史表征”必要条件，但不直接改student GRU。
复用已通过质量合同的456条task、late prefix和12个residual，加noop得到5,928个真实
collision/follow/overtake标签。70个ego startpoint按冻结hash分5折；输入只含最近50步actor
可见LiDAR与previous measured speed，经冻结U44前端形成`50x420`历史，禁止pose、opponent状态、
stratum、event距离或未来outcome进入feature。

| OOF selector | inherited / created / lost恢复 | target total | controls collision / overtake loss |
|---|---:|---:|---:|
| grouped fixed | `45 / 33 / 1` | `79` | `5 / 5` |
| frozen-hidden control | `68 / 33 / 3` | **`104`** | `13 / 20` |
| 50-step history treatment | **`68 / 32 / 2`** | **`102`** | **`13 / 21`** |

history的三分类accuracy为`65.92%`，略高于control `64.49%`，但动作选择未获得增量：target
少2条，lost-overtake只有`2/13 < 4/13`，controls两条门也均失败。按预注册判
`fail_close_tested_representation_instance`；不运行independent validation、full-development
训练、actor接入或PPO。

关闭范围只限`late 50-step history × fixed 12-action × class-weighted 3-class outcome × fixed
probability score`，不能扩写成所有representation learning失败。不得扫描history长度、网络、
loss或score重开本实例；全新监督机制仍需独立必要条件。Constrained PPO与本Gate正交，仍未测。
完整证据见`ANALYSIS.md` §38。

### 9.20 Round Z5：budget-constrained frozen-hidden operating point（2026-08-08，零新仿真）

Z2的51与Z4-A的104处在不同noop率/control harm上，原比较不能承担类级判决。本轮保持
456 task、late 12-action、5,928真实outcome、模型/loss和outer folds不变，在每个outer-train
内部新增4-fold startpoint OOF，只用inner outcome选择11个lambda和全部可实现noop margin；
预算固定为fixed的两项`5/225`，outer-test只评估一次。

| 合同 | target | controls collision / loss | 判决 |
|---|---:|---:|---|
| grouped fixed | **79** | `5 / 5` | 对照 |
| frozen，independent outer seeds | **69** | **`1 / 3`** | target失败 |
| frozen，exact Z4 outer seeds | **66** | **`3 / 4`** | target失败 |
| history，independent / exact Z4 seeds | `63 / 55` | `6/8、7/7` | target/harm失败 |

frozen相对fixed的target配对为`28/38`（`p=0.268`）与`29/42`（`p=0.154`）。诊断性事后全局
扫描在independent seeds可得到`84 @ 3/5`，但nested只有69；exact Z4 seeds的事后matched-harm
最高也仅`72 @ 4/5`。不得使用84替代无泄漏判决。

判决：预注册的`target > 79`两次均未满足，关闭当前`frozen hidden × 3-class outcome × nested
operating point × late fixed 12 actions`；`p=0.268/0.154`表示未检出优势，不是证明显著劣于
fixed。Nested只约束harm率`<=5/225`，外层实际欠用预算到`1/3`和`3/4`，不能称为恰好harm
matched；无泄漏条件下不能强迫outer恰好用满5/5。Z2 top-1具体extraction继续关闭，Z4未约束
104不再作为同budget优于fixed的证据。关闭不是hidden信息论上限，会反传进student GRU的2b
仍未测。完整证据见`ANALYSIS.md` §39。

### 9.21 Round Z6-A：prefix-reset snapshot no-op（2026-08-08，无actor更新）

固定Gate A的28条U42--U45至少3/4共识development任务（来源分层collision/lost=`19/9`、21个ego
startpoint），直接复用Gate B窗口。每条在当前381D observation被消费前保存完整F110物理/
steering delay/scan RNG/lap/collision状态、LatticePlanner与PurePursuit动态状态、PPO wrapper/
reward/corridor gate，以及U44 actor与同update privilege-GRU critic hidden；snapshot必须先经过
pickle序列化/反序列化。

28/28原后缀与恢复后缀逐位相同。恢复点observation和两路hidden、每步actor raw/executed action、
opponent action、critic value、reward及四分量、双车7D state/steering buffer、双360D LiDAR、
collision flags、terminal/outcome/首次collision step的最大误差全部为0；28份snapshot、28+28份
finite对齐trace与terminal-post-step合同经独立重读通过。Prefix为0--701步，26/28大于0、
中位345.5；完整轨迹16,385步中prefix合计9,589步（58.5%）。这只是可避免仿真步比例，不是
wall-clock加速测量。当前环境原后缀终局为14次ego collision、6次overtake、8次follow；它们只
用于证明三类终局都可精确恢复，不重估Gate A的来源标签或U44性能。

判决：**机械snapshot必要条件通过**，不再因“只能replay-to-prefix”停止；但formal训练仍不
准入。以下是Z6-A完成时的下一步，后来已由§9.22完成：用保存的actor-visible observation prefix对当前更新后网络无梯度
burn-in，逐位对齐正常起点执行的actor/critic hidden、action/value。本轮只跑确定性mean action，
没有审计训练期exploration RNG/residual block；Z6-B还必须冻结窗口后探索状态语义并验证
log-probability重建、prefix transition不进入loss/GAE、窗口边界bootstrap和真实terminal/
truncated符合标准RecurrentPPO。任一失败即停止，不复用旧U44 hidden、不以replay-to-prefix
fallback。完整证据见`ANALYSIS.md` §40。

### 9.22 Round Z6-B/Z6-BR：current-network burn-in与GAE语义（2026-08-08，无actor更新）

固定Z6-A的28条Austin development任务与9,589个prefix observation；U44作source snapshot，
同一真实训练轨迹U45 actor/critic固定作参数改变后的current network。U44逐步重放对snapshot的
window observation与actor/critic hidden最大误差全0；U45整段fast burn-in相对逐步reference最大
actor hidden `5.84e-6`、critic hidden `4.77e-6`、action `2.38e-6`、value `1.49e-7`，通过事前
`5e-5`线。26条非零prefix全部至少有一项U45 state/output不同于旧U44，旧hidden不得复用。

Rollout buffer新增opt-in `recurrent_resets`：snapshot的`episode_starts=true`继续切断GAE并建立
新sequence，`recurrent_resets=false`保留该sequence的burn-in hidden；普通collector未stage时
逐元素复制episode start，默认行为不变。合成terminated/timeout/rollout-cut合同的advantage/
return误差`1.49e-8 / 0`，切段`0/3/5`、actor initial hidden`1/2/3`、critic`11/12/13`全部符合。
Prefix transition计数0。

Baseline 28 transition与corridor 1,428 transition的collection-equivalent `max |log_ratio|`和
`max |ratio-1|`均0。原strict machine verdict因反算telemetry residual在50步内误差
`3.18885e-6 > 0`失败；该字段只用于telemetry，不进入PPO likelihood。独立预注册Z6-BR保持
U45、seed、任务、51步与原`5e-5`界不变，直接读取内部temporal noise：首50步误差0，第51步
inactive/block 0，revisit新residual，全部通过。因此保留原machine fail，同时科学裁决为
`pass_prefix_reset_semantics_after_measurement_adjudication`。

判决只准入Z6-C no-update training-density/integration Gate。U44来源fast path曾有hidden约
`0.0040`误差，说明fast path不能跨update无条件外推；Z6-C必须用逐步exact burn-in，或每个update
先fail-closed校准fast path，并实测102,400-transition布局、ratio identity、GAE与wall-clock。
Snapshot比例、role/cache语义、探索模式、PPO学习收益和四图性能均未测；formal训练仍不准入。
完整证据见`ANALYSIS.md` §41。

### 9.23 Round Z6-C/Z6-CR：no-update训练密度与batched replay裁决（2026-08-08，无actor更新）

固定canonical BC、fresh privilege-GRU critic、Austin seed42、16×6,400、batch12,800、baseline
独立速度高斯、479 collision cache和600 ordinary pool。Baseline与treatment都完成102,400行，
collision/ordinary严格`51,200/51,200`、finite、参数不变、GAE/return误差0、collection-equivalent
ratio误差0。Treatment只把每第3次collision reset换为共识snapshot：119次collision reset中
39次prefix，28 key全覆盖；prefix transition 8,634（8.43%），首150步窗口5,438（5.31%）。
Baseline/treatment墙钟`98.97/112.32s`，比值`1.1349`。

原Z6-C 12项只失败普通batched envelope：treatment最大`|log-ratio|/|ratio-1|`为
`0.010697/0.010755 > 0.01`，因此原`fail_stop_prefix_reset_density_or_integration`必须保留。
Z6-CR在再次完整收集前冻结因果线，同一8个minibatch的普通/exact dry actor epoch得到gradient
cosine `0.9999838`、相对L2差`0.005706`、最大policy-loss差`4.73e-7`，clip fraction 0、mean KL
`9.05e-10`且参数未step，故独立裁决通过。它只说明该刀锋最大值未实质改变当前PPO actor更新。

下一步仅允许预注册的`ppo_prefix_reset_consensus1of3`单次fresh Austin训练：30 formal updates，
每update exact ratio`<=5e-5`、batched最大ratio偏差`<0.02`硬停止；完成后固定CUDA评估U27--U30
四图各600，不选点、不扫prefix比例/窗口/panel/LR/exploration/updates、不延长U45。无论结果如何
关闭本实例；方法2b、collision-only与Constrained PPO不由本实验代判。完整证据见
`ANALYSIS.md` §42。

### 9.24 Round Z6-F：prefix-reset单次正式PPO（2026-08-09，完成并关闭tested配置）

从canonical BC fresh start，唯一变化为collision角色每3次reset中1次使用28项共识prefix；Austin
seed42、privilege-GRU、16×6,400、warmup+30 formal、50/50 role、479/600 pools与baseline探索
均保持production。训练31行metrics、30对finite checkpoint、12-key和16/16 actor step均通过；
batched ratio最大U26 `0.019095 < 0.02`、exact全0，prefix/window最低`8.43%/4.26%`。

16个正式包共9,600 episode/trace全部通过完整性合同。逐图`collision/overtake`：

| update | Austin | Hockenheim | Moscow | Nuerburgring | 四图 | BC逐图门 |
|---:|---:|---:|---:|---:|---:|---|
| U27 | `17/365` | `29/363` | `36/391` | `23/399` | `105/1518` | 失败 |
| U28 | `20/367` | `27/365` | `32/395` | `22/398` | `101/1525` | 通过 |
| U29 | `23/367` | `28/362` | `27/399` | `23/400` | `101/1528` | 失败 |
| U30 | `21/365` | `27/365` | `33/393` | `22/399` | **`103/1522`** | **通过** |

U30相对BC collision removed/created `60/34, p=0.00955`，overtake lost/gained
`16/93, p=2.21e-14`，最低验收通过。相对U44则collision `40/81, p=0.000244`显著更差、
overtake `33/77, p=3.30e-5`显著更好，是安全--超车交易。最终`<40/>1500`只通过超车；U28/U30
通过但不连续，稳定性失败。相邻band aggregate在`101--105/1518--1528`，配对变化均不显著；
不能把Hockenheim刀锋失败夸大成显著崩溃，也不能放宽硬门。

判决：**高超车端经验前沿扩展，最终目标未完成，关闭本tested配置。** 不扫interval/panel/window/
LR/exploration/updates，不延长U45，不从U28挑production。它证明prefix密度可改变前沿，但没有
解决跨状态/地图的安全--progress统一选择；不科学否决prefix-reset方法类。方法2b、collision-only
与Constrained PPO仍需各自必要条件。完整证据见`ANALYSIS.md` §43。

### 9.25 Round Z7：collision-only BC anchoring独立重开（2026-08-09，无训练）

本轮没有修改旧Z3的7条panel或匹配门，而是在actor outcome前冻结全新2,880条Austin Cartesian
panel，并把estimand事前限定为每个raceline/speed层同时有exact safe control support的稳定
collision。BC/U44各跑2,880条；只对59条`BC=overtake、U44=collision`候选补跑U42/U43/U45，
与完整source screen等价且少8,463次无信息仿真。5,937条result/trace全部通过质量合同。

最终41条稳定source全部有support，匹配41 controls，覆盖21起点、r0/r2=`31/10`。V0五门全部
通过；branch0在82条上raw/executed/opponent action、两车pose/speed和双LiDAR最大误差均0。
full-BC必要条件如下：

| 条件 | 实测 | 线 | 判决 |
|---|---:|---:|---|
| collision rescue | `18/41` | `>=21` | **失败** |
| rescued overtake | `18/18` | `>=15` | 通过 |
| rescue覆盖 | 11起点，r0/r2=`16/2` | 2起点、每线1 | 通过 |
| control新collision | `4/41` | `<=2` | **失败** |
| control overtake loss | `4/41` | `<=2` | **失败** |

这是样本充分、exact control闭合、branch0精确后的必要条件反例，故严谨关闭
`canonical BC × overlap-supported stable collision × fixed 1.5s window × collision-only`实例。
不得生成anchor dataset、做shadow beta或formal PPO，也不得扫描teacher/window/support cap/
门限。正面机制证据保留：18条救援全部成为overtake；但它不能抵消23条未救和4条safe-control
伤害。结论不外推其他teacher/窗口/一般BC正则化；旧Z3仍是inconclusive历史事实。

### 9.26 Round Z8：GRU-changing paired action-response auxiliary（2026-08-09，无PPO）

本轮是2b首次直接检验：复用456条/70起点development与late 13-action真实结局，control在frozen
U44 hidden上训练response head；treatment让同一`collision indicator + progress delta vs noop`
loss反传进U44初始化的原1680D GRU，`k`、speed MLP、actor output layer冻结。每个test fold固定
下一foldcalibration、其余三fold训练；lambda/noop门只看calibration，两个seed都必须通过。

正式recurrent重建456/456 hidden/action最大误差0。两seed五fold的GRU参数相对L2均约
`0.0021--0.0026`、test hidden平均相对L2约`0.0094--0.0123`，故不是“表征没变”。结果：

| seed | frozen target @ control collision/loss | trainable GRU target @ control collision/loss | lost |
|---:|---:|---:|---:|
| 7100 | `58 @ 12/13` | **`50 @ 14/15`** | 0 |
| 8100 | `61 @ 19/20` | **`58 @ 12/13`** | 1 |

Seed7100相对frozen target配对`2/10, p=0.0386`，显著更差；seed8100 target `11/14, p=0.690`，
虽control harm少7次（`1/8, p=0.0391`），绝对12/13仍远超5/5。两seed都未达到target88、相对
frozen+9、lost4与controls门。关闭
`U44 GRU × late50 × paired collision/progress × 10 epoch × rotating calibration`具体实例；
不运行独立validation/PPO，不扫epoch/LR/loss/操作点。该结果不否决所有2b方法类。

### 9.27 Round Z9：collision-cost Constrained PPO preflight（2026-08-09，无actor更新）

固定目标把现有首次碰撞`-2.0`从reward GAE精确移除，只保留`cost=1[first ego collision]`；
reward critic仍为`privilege_gru`，独立训练期cost critic为P20 MLP。预算`.10`、lambda初值1、
dual LR.5、范围0--20，全部在rollout前冻结。

有效rollout为102,400 transition；153个完整episode中57 collision，覆盖85起点。57个cost event
与57个collision episode相等，reward去重最大误差0，cost advantage/return std
`.27325/.25761`。Dual `1→1.13627`。Reward-only/合成actor梯度L2 `25.3429/20.0068`，差分相对
L2 `.31087`、cosine `.96687`；cost信号与优化调用链成立。

按ego起点五折、每fold17个test起点，collision episode `11/14/14/9/9`。OOF MSE
`.10857`对常数`.11314`，skill `.04038 < .05`；episode-start AUROC
`.42855 < .65`；early≥100步 AUROC `.60703 < .65`。三条可学习性门失败，machine verdict为
`fail_stop_exact_constrained_implementation`。Formal和四图未运行，actor不变。关闭当前
`reward去collision + P20 MLP + d=.10 + lambda0=1 + dual_lr=.5`实例；不把OOF gate误写成
Constrained PPO方法类的数学证伪。

---

### 9.28 跨轮汇总与证据级别（2026-08-09，Claude 独立复核，无新仿真）

本节只做两件既有各轮小结没有并排给出的事：把全部可部署点放进同一张**带证据级别**的表，
以及记录一条按不适用前提被关闭的轴。完整推导见 `GATES.md` 附录 C。

**证据级别必须与数字一起引用。** 四图 `collision / overtake`：

| 模型 | 四图 | 证据级别 |
|---|---:|---|
| Canonical BC | `129 / 1445` | 正式 CUDA |
| Production U30 | `94 / 1508` | Austin600 正式；跨地图 headline，Moscow 仅 `160/600` 中断残留、Nuerburgring 仅 traces |
| 前向走廊时间相关探索 U44 | `62 / 1478` | 正式 CUDA 四图配对包 |
| ordinary 异线高速重加权 U30 | `73 / 1516` | **trace 重建包**；`direct_evaluator_aggregate_retained=false`、manifest 无 `device`/无顶层 `collision_scope`；600×4 完整 trace 齐备，计数与配对可复算 |
| U42--U45 等权平均 | `67 / 1465` | 正式 CUDA |
| prefix-reset U30（Z6-F） | `103 / 1522` | 正式 CUDA，U27--U30 共 16 包 |
| 粗 regime 事后拼接 | `58 / 1527` | 不可部署上界 |
| 逐 episode hindsight oracle | `25 / 1557` | 不可部署上界 |

三条由上表得到的判断：

1. 六个可部署点近似落在同一条安全--超车前沿上；**已执行的全部方法都在这条线上滑动。**
2. 重加权 U30 是唯一在四图两轴上同时优于 production 的点（`-21` collision、`+8` overtake），
   但证据级别低于 U44/SWA/BC。**切换 production 前必须补一次固定 CUDA 四图确认**，同时
   production 自身缺 Moscow/Nuerburgring 规范包，当前比较是"重建对 headline"。
   补齐成本为 `2400 + 1200 = 3600` episode、零训练。
3. prefix-reset U30 在四图上被重加权 U30 **双轴支配**（`103/1522` 对 `73/1516`）。这不改变
   §9.24 "扩展前沿、目标未完成"的判决，只说明它不是前沿进攻端的最优点。

**一条按不适用前提被关闭的轴（需要用户裁定，未授权）。** §10.1 第 16 条关闭 side-phase
steering exploration 的理由是"缺少可靠部署期 conditioning"。已核实的两个事实是：

- 历史上全部 8 条训练臂的 `STEERING_LATENT_STD` 一律为 `0.03`，**转向探索通道从未被改变过**，
  §18 的五组探索实验全部只动速度通道；
- 探索是纯训练期机制：`eval_multiagent.py` 评测时直接取 mean action、无噪声无门；
  `FrontCorridorGate` 本身条件在模拟器特权几何上，**同样没有部署期 conditioning**。

因此该理由适用于**部署期相位门控**，不适用于**训练期探索门**。相关已测证据（新造失败的
接触几何以近平行侧/后擦碰为主、相对 yaw 中位 `3.67°`；Z2 最强单一固定干预是
`steer +0.02 / speed +0.5`，带转向分量）指向横向通道。**这是未测假说，不是发现**；在用户
明确裁定并重新预注册前，§10.1 第 16 条继续有效，不得据此开跑。

## 10. 当前允许与不允许的下一步

BC Gate、Round Z2、Round Z4-A和Round Z5具体实例均已关闭；Round Z3历史validation仍因
matched-control support不足而inconclusive，但Round Z7已经用新独立panel闭合support并严格否决
当前collision-only teacher/window实例。必须分开“科学未决”和“当前工程
准入”：

- 当前canonical BC × stable collision × 1.5秒collision-only变体已由Z7严谨关闭；只有更换teacher
  或机制性窗口定义才是新方法，不能继续调panel、support cap或beta；
- prefix-reset的Z6-A--Z6-F完整链已测：U30最低BC线通过且扩展高超车端前沿，但最终安全目标与
  连续稳定性未通过；当前tested配置关闭，方法类未被严谨否决；
- Constrained PPO当前固定实例已完成reward/cost唯一化与preflight；机械链路通过但startpoint
  OOF三门失败，formal按合同停止。只关闭该P20/budget/dual实例，不否决方法类；
- representation-only大类不是信息论上被否决；Z8已首次直接训练student GRU，并关闭paired
  collision/progress、late50、10-epoch具体2b。其他全新监督仍属新方法，但不得调Z8参数重试；
- MoE架构本身未测，但在最终actor必须保持12-key兼容的当前任务合同下工程不合法；只有用户
  明确放弃该边界才值得科学检验。

当前没有仍在队列中的已授权实验：Z6、Z7、Z8与Z9具体实例都已完成各自允许的最深Gate或formal，
并按停止规则关闭。下一步若存在，必须是用户重新授权、带新机制假说的新预注册；不能现场调
prefix、teacher、2b loss、cost budget、dual或critic继续试。

### 10.1 已关闭

不要重复：

1. Post-pass权重/q/LR；
2. risk L12或更长clearance sweep；
3. 完整805 hard-neighbor pool；
4. ordinary150；
5. interval15 fixed difficult pool同配置；
6. speed std0.25/0.50；
7. 0.40→0.15退火；
8. 前向走廊门控时间相关速度噪声的1米/2米门宽；
9. 前向走廊门控时间相关速度噪声延长到45 updates；
10. 不再训练ordinary异线高速重加权比例0.6或继续增加异线高速权重；现有固定U30仅按§1.3
    与`ANALYSIS.md` §27作为产品候选重判，不能把候选资格误写成授权继续扫权重；
11. 在没有新证据时先扩actor输入、改critic/reward，或重复一般物理量可观测性probe；
12. 依据人工偏好梯度直接训练same-line/offline-fast对称梯度投影。
13. multi-map PPO；用户明确规定仅在Austin训练，其余三图只用于泛化测试。
14. 在没有新的失败窗口富集证据时训练phase-bounded temporal block；
15. 在没有新的关键LiDAR压缩证据时解冻actor pressure `k`或扫描其学习率。
16. 在interaction-phase早期可分性未稳定通过时运行side-phase steering响应面、fresh PPO
    credit审计或正式训练相位门控转向探索。
17. U42--U45三点/非等权平均、EMA/SWA或根据四图结果选择checkpoint组合；固定等权Round Z0
    已以`67/1465`失败，见§9.14。
18. 当前BC functional regularization的anchor dataset、Gate C/D与formal训练；Gate B已因L恢复
    `0/9`和controls丢超车`2/28`关闭，见§9.16。
19. 当前12动作、early/late prefix、frozen-hidden ActionScorer及其first-action extraction；oracle
    动作存在但两侧rankability均失败，见§9.17。不得改动作幅度、窗口、阈值或只报oracle重试；
20. 按Round Z2旧触发规则直接启动prefix-reset；“early通过、late失败”没有出现，所以旧方案
    没有自动准入。该项是程序停止，不是科学否决；若继续必须改用直接检验reset训练密度的独立预注册；
21. Residual/MoE/runtime selector在361D/12-key兼容要求不变时不合法。
22. 在当前collision-only validation panel上放宽speed/raceline匹配、改成有放回control、
    从development补control或删除难匹配source；Round Z3已因control support确定性不足
    而inconclusive，不是teacher失败。
23. Round Z4-A固定的50步历史GRU、12动作、三分类loss与概率score；它没有超过frozen-hidden
    control，并独立失败lost-overtake与safe-control门。不得扫描history长度、网络或score重试。
24. Round Z5的lambda/noop阈值、harm预算、nested folds或seed；两套outer seed合同均未达到
    `target > 79`准入线，不得使用事后OOF 84或增加网格重开；该项目关闭不等于统计显著劣于fixed。
25. 复用U44旧hidden、绕过每update likelihood硬门，或重跑/扫描Z6-C的prefix比例、窗口、panel、
    exploration、updates与学习率；Z6-C原machine fail与Z6-CR追加裁决都必须保留。
26. 重跑或扫描Z7 collision-only的canonical BC teacher、1.5秒窗口、support cap、门限或beta；
    新独立panel已在41 source/41 exact control上得到rescue `18<21`和control harm `4>2`，当前
    方法实例已严谨关闭。更换teacher或机制性窗口属于新方法，必须另行预注册。
27. 重跑或扫描Z8的late50窗口、paired collision/progress loss、10 epoch、GRU/head LR、loss
    权重、lambda或noop threshold；两seed已真实改变GRU但都失败target、lost与absolute control
    门。新辅助监督属于新方法，不能继承Z8准入。
28. 重跑或扫描Z9的P20 cost critic、`d=.10`、lambda初值1、dual LR.5或其上下界；reward/cost与
    actor梯度链路已通过，但startpoint OOF的skill/AUROC三门失败，formal按预注册停止。不同
    constraint目标或cost representation属于新方法，且本Gate不构成方法类数学否决。

### 10.2 未完成但不是高优先

- 训练期辅助表征学习（PPO + auxiliary representation loss）：§31.6规定的准入预检已在
  §32执行完毕，**结论是缺口不存在且符号相反**——维度匹配下GRU hidden在9/9几何目标上优于
  actor输入。因此以该9个几何/速率量族为辅助目标的训练**不准入**。这不否决辅助表征学习
  整体：换用不属于该量族的目标（对手planner意图代理、ego动作序列可达性量等）必须按§32.2
  同一口径重做缺口预检，不得继承结论或跳过预检。Round Z4-A已进一步否决固定50步历史与
  12动作三分类结果监督；Round Z8又直接训练原GRU并否决paired collision/progress的具体2b；
  新目标不得复用两者失败配置做参数扫描；
- temporal hold K10/K25：仅提出，未训练；
- Group13 GRU/head LR 2×2：未运行；
- outcome-aware hard：历史上有实现、没有独立A/B；源码已删除，只有重建合同。

不得把这些写成已否决。

Hard-neighbor 10%另行归档为“训练完成但晚期eval未完成、用户主动终止”；当前入口已退役，
不再列为可继续执行的下一步，也不得写成已证伪。

### 10.3 合法重开条件

只有以下变化才足以重开已关闭方向：

- 任务分布或最终验收面板实质改变；
- actor可观测信息或策略结构改变；
- 能把same-line行为与off-line副作用解耦的新控制，而不是强度旋钮；
- 新的严格单变量设计能修复已有实验的明确混淆。

2026-08-06 fresh PPO复核表明人工偏好梯度的output-head反向并不等于实际PPO更新稳定反向；
后续“解耦”设计必须先在fresh rollout的advantage加权梯度上建立稳定可重复的冲突，不得只
引用冻结cohort的`-0.96`。Credit/GAE只在新鲜paired rollout直接出现动作收益与advantage
符号错配时重开。§9.8的role-return混合只是credit审计的一部分，不等于已验证GAE归因正确，
也不授权修改50/50 role合同。

仍应遵守用户要求：目标是单次PPO训练得到可部署actor，不依赖运行时oracle、
未来碰撞信息、特权触发器或部署后动作后处理。

---

## 11. 新实验执行规则

1. 使用conda环境 `end2race` 的绝对Python。
2. 长任务使用tmux，避免意外中断。
3. 优先修改现有脚本参数；不要为每次运行新建bash/log。
4. 新实验先完成预注册，再新建 `run.sh` 并只写该实验的显式命令；
   一个时期只保留当前这一项实验，不把已完成或未运行提案继续注释在文件中。
5. 每次只改一个轴；若不得不多变量，预注册每个变化和不可分离的限制。
6. 从canonical BC fresh-start，除非问题明确是checkpoint continuation。
7. 不使用自动checkpoint选择器；预先固定update或报告完整checkpoint band。
8. 当前正式验收默认运行Austin、Hockenheim、MoscowRaceway和Nuerburgring各600 episode；
   Austin600与crossmap1800都具有验收权且跨地图必须逐图报告；每图ego collision不得高于
   canonical BC且overtake不得低于canonical BC，opp-wall单列。near400和hard子集只能作机制诊断。
9. 同场景必须报告removed/created、lost/gained和配对p。
10. 训练统计不能替代deterministic eval。
11. output目录必须为空；禁止覆盖已有run。
12. 不把运行脚本混入回归测试集合；实验工具只用于测试确实新增的组件。
13. 文件布局和命名按 `GUIDE.md`，不要重新制造复杂的独立分析结果树。

---

## 12. 接手检查清单

1. 阅读本文件 §1、§6–§10。
2. 需要完整数字时阅读 `ANALYSIS.md` 对应专题，不要从聊天恢复。
3. 需要脚本/test逻辑时阅读 `EXPERIMENTS.md`，不要在HANDOFF重复维护。
4. 运行：

```bash
git rev-parse HEAD
git status --short
pgrep -af '[r]un\.sh|[t]rain_ppo\.py|[e]val_multiagent\.py|[e]valuate\.sh' || true
```

5. 复用cache前核对canonical BC哈希；身份不匹配则使用新cache目录按当前分类actor重建，
   禁止复用身份不匹配的cache。
6. 解释旧run时先读它的 `run_config.json`，不要套当前defaults。
7. 检查metrics行、checkpoint、final actor和进程状态后再宣布run完成。
8. 新eval使用唯一actor alias，防止同名checkpoint trace覆盖。
9. 正式评测默认且固定使用CUDA/GPU；CUDA不可用时停止，不静默退回CPU，也不重复CPU对照。
10. 验证面板scenario数、0 errors、trace key、marker和finite数值。
11. 任何production变更必须同时更新本文件的§1、§2、§3.8和对应判决。

历史交接快照（Round Z6前，已被§1.1与§9.27取代，不可作为当前行动）：

> 架构与实验接口已经覆盖reward、pool、采样和探索多个方向。当前不启动新训练；已开始执行
> BC-anchor预注册的训练前Gate链。正式CUDA
> 四图BC验收已经确认前向走廊门控时间相关速度噪声U44为安全候选：`62/1478`，相对BC
> collision removed/created `94/27`（`p=7.14e-10`），overtake lost/gained `41/74`
> （`p=0.00269`）。旧ordinary异线高速重加权U30在新口径下计数为`73/1516`，相对BC也
> 显著双轴改善，但其历史package未记录CUDA device，故只列为待一次固定CUDA确认的高超车
> 候选；旧near400 `64/302`仍是明确副作用。碰撞身份与姿态复算进一步表明当前是失败迁移：
> U44/RW30分别新造`27/28`次，新增车辆碰撞绝大多数为相对yaw约`4.5°`的平行侧后接触。
> 粗regime最优拼接只有`58/1527`，而不可部署的逐episode事后上界为`25/1557`，所以更高目标
> 缺的是regime内部的状态条件选择，不是继续增加探索/采样剂量。2026-08-07的最小Austin诊断
> 又否决了两个直接候选：U44新造的6个车辆碰撞窗口没有走廊block可作phase-bounded截断，关键
> LiDAR回波也没有frozen pressure饱和。随后30个平行侧/后碰撞的startpoint分组线性probe
> 仅在1.0s出现不稳定的pressure AUROC 0.773，1.5s/0.5s分别为0.606/0.598，GRU hidden最高
> 0.617，因此不准入动作响应、PPO-credit或相位门控转向探索。§31随后显示所测真值几何/
> 速率在`>=0.5s`的线性早期可分性仍未通过预注册门；§32的fold-local维度匹配读出则显示
> hidden在9/9当前几何目标上优于输入，否定了该量族“输入有信息但hidden丢失”的缺口前提，
> 因而不启动这9个目标的辅助表征训练。随后Round Z0等权checkpoint平均得到`67/1465`，
> Hockenheim `19/341`未过BC超车下限，且core-14仍保留13个，已按预注册关闭并禁止改权重重试。
> BC-safe anchoring Gate A随后在冻结的718条Austin development困难场景上完成：BC-safe
> overtake 308条，U44单点回归46条，3/4共识28条，覆盖21个起点，raceline0/2=`20/8`，
> U44 collision/lost-overtake=`19/9`；五臂3,590条trace合同全部通过，validation未评估。
> Gate A六条准入线全部通过；随后Gate B branch 0在56条上以全部字段最大误差0精确复现U44。
> 完整BC接管能把C层`10/19`碰撞全部救成超车，但L恢复为`0/9`，safe controls另有`2/28`
> 丢超车，因此按预注册科学失败并关闭BC anchoring方向。224条branch trace合同全部通过，
> validation未运行branch；没有anchor dataset、Gate C/D、formal训练或新actor。
> Round Z2随后在456条更宽Austin development场景上完成456次精确no-op和10,944次固定动作
> branch。early/late oracle均证明局部动作广泛存在，但五折hidden ActionScorer分别只恢复
> lost-overtake `1/13、0/13`，early误伤17/225 controls，late target success `51 < 79`固定动作
> baseline，故fixed-library action-conditioned/preference关闭，prefix-reset不准入。六类方法中
> prefix-reset训练机制与MoE架构本身仍未被直接检验；后者受12-key工程边界排除。随后Round Z3
> 用150条独立validation找到7条稳定collision
> cohort，但精确matched controls分层不足，在任何branch前按预注册停止；collision-only BC
> 变体只能判inconclusive。Round Z4-A再用Round Z2的5,928个真实late action-outcome标签检验
> 50步actor-visible历史GRU：treatment为`68/32/2`、controls `13/21`、target `102`，低于同协议
> frozen-hidden control的104，因此关闭该具体表征实例，未进入validation或训练。
> Round Z5随后以5x4 nested startpoint CV约束fixed的control预算：frozen selector在独立outer
> seeds为`69 @ 1/3`，恢复Z4 outer seeds为`66 @ 3/4`，均未达到`target > 79`准入线；配对
> `p=0.268/0.154`，所以是未检出优势而非显著劣于fixed。独立seed的事后全OOF曲线虽可读到84，
> 但无泄漏nested只有69；故当前tested selector关闭，不进入validation或PPO，2b仍未测。
> Round Z6-A随后完成28条共识任务的完整snapshot pickle往返与后缀复跑，所有state/action/
> value/reward/LiDAR/terminal字段最大误差0；26条为非零prefix，中位345.5步，机械Gate通过。
> Round Z6-B随后用真实相邻U45完成current-network burn-in、boundary/hidden mask、GAE和两种
> exploration likelihood审计；原strict report只因非因果telemetry反算舍入失败，独立Z6-BR
> 直接测内部noise后通过。下一步不是训练，而是Z6-C no-update训练密度/吞吐Gate；旧U44 hidden和
> replay-to-prefix都不是fallback。当前科学未决项还包括collision-only新panel、prefix-reset
> 训练密度、Constrained PPO以及不同机制的全新representation目标；MoE只有放弃12-key边界才
> 进入候选。Constrained PPO的
> collision-cost定义与当前reward collision项重复，必须先
> 由用户选择“移出reward后约束collision”或“保留reward并约束overtake”，不能代替用户决定。
> 用户明确禁止multi-map PPO；训练只可使用Austin。Production部署别名暂时仍指向U30，当前
> 没有未完成run；Constrained PPO尚未预注册，当前不启动formal训练。
