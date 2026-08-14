# End2Race 当前 HANDOFF

更新时间：2026-08-14（Asia/Singapore；进入汇报整理期，不再新增tech；当前无排队训练）

## 0. 文档职责和读取顺序

本文件是**接手当前仓库时的第一入口**，只保留会改变下一步行动的当前事实、核心合同、
实验最终判决和停止/重开规则。完整数据与机制推导已经迁入：

- `ANALYSIS.md`：完整实验设计、面板定义、配对结果、分层数据、机制判断和证据边界；
- `EXPERIMENTS.md`：历史实验工具与回归测试的实现逻辑和重建合同；
- `GUIDE.md`：用户要求的实验执行、命名和文件布局规范；
- `STYLE.md`：当前Python代码风格与重构约束。

专题预注册、阶段报告和历史重构计划已经在2026-08-10完成归并并删除。后U44各轮的当前判决与
停止/重开边界以本文件§9.14--§10和§13--§14为准；完整数字与方法学更正在`ANALYSIS.md` §33--§60，
可重建的脚本、数据和训练调用合同在`EXPERIMENTS.md` §13--§26。不得再寻找或恢复第二份专题
HANDOFF、已删除的专题Gate汇编或单方法预注册文档作为并行权威。

如果本文件和 `ANALYSIS.md` 的历史数字冲突：

1. 当前源码、模型文件、run config 和机器可读结果优先；
2. 当前 production、运行状态和允许的下一步以本文件为准；
3. 历史实验的完整数字、推导与边界以 `ANALYSIS.md` 为准；
4. 不要用当前 CLI 默认值反推旧 checkpoint 的配置。

用户已决定清理历史分析产品、实验工具、失败训练模型和过期评测。清理前的核心判决、
关键统计、production U30残余失败身份和oracle动作族已固化到 `ANALYSIS.md`；测试与工具的
重建合同在 `EXPERIMENTS.md`。这些文档保留的是**结论与关键实现合同**，不是原始数据和源码
的无损压缩。2026-08-10已删除被关闭方法的run/eval、一次性Gate产品和旧near/hard/noise面板；
原始产品缺失不改变已记录判决。当前训练输入只保留`post-trained/collision-cache/`；first-action
固定反事实数据若仍在磁盘也只是历史资产，活动代码不再读取。正式四图600由`evaluate.sh`直接生成，
不重复保存ScenarioSpec副本，也不维护任何文件哈希；当前仍存在的模型路径见§2。

仓库：`/home/haowei/Documents/End2Race`

分支：`main`

提交和工作树状态必须在接手时实时查询，不能从本文件反推。`.agents/`已纳入Git版本管理；
本轮一次性梯度诊断脚本、张量和评测复核产物在核心结果与重建合同写入文档后清理，不作为
后续实验入口。

---

## 1. 当前状态和 production 决策

### 1.1 运行状态

用户最新固定模型评测合同：今后任何新actor只评Austin、Hockenheim、MoscowRaceway和
Nuerburgring四张既定面板，各600 episode、CUDA deterministic、ego collision scope。不得新跑
near400、hard73、其他hard/near/noise/startpoint/single-agent或额外地图eval。历史结果继续保留为
当时实验事实，但不再参与新模型验收或选择。训练侧离线Gate和反事实branch只作机制证据，不是
模型eval，也不能替代固定四图600。

**当前活动（2026-08-13，接手时仍须用`pgrep`和metrics复核）：没有活跃的`train_ppo.py`、
`evaluate.sh`或`eval_multiagent.py`进程，根目录也没有`run.sh`。冻结前曾定义的168个collision target +
225个safe-overtake control BC-native固定偏好臂没有构造dataset、没有训练、没有评测，当前不是排队任务。
K75在U20后中断并按用户决定清理；K100、去lateral-offset
门、canonical-BC固定来源偏好和在线碰撞触发偏好均已完成正式评测但未形成新前沿，固定实例关闭。
production U30已按当前合同补齐四图各600完整trace。**

**当前范围冻结（用户2026-08-13明确决定）：从本节点开始不再提出、实现、登记或排队任何新的
technical method。当前工作只允许整理现有代码、统一活动入口、核对既有证据和准备汇报。已经讨论但
未实现的Lattice-reference+BC、原`instrument_train`宇宙BC-native、action-conditioned短期Q/cost
critic、group-robust PPO、reverse constrained PPO、selective reference retention和每5个update刷新
偏好数据均只作为讨论历史，不是待办、准入项或隐含授权。冻结前的168/225 BC-native臂也已随
`run.sh`移除而退出队列；保留其历史合同不等于授权执行。**

本轮共同对照K10/K50 U30为`74/1566`。新臂U30依次为：固定BC偏好`67/1436`、在线碰撞偏好
`85/1466`、K100 `110/1425`、去lateral-offset门`69/1369`。相对对照的collision
removed/created与p依次为`55/48,.5546`、`33/44,.2543`、`49/85,.00237`、`55/50,.6965`；
overtake lost/gained与p为`159/29,6.43e-23`、`122/22,5.45e-18`、`177/36,1.47e-23`、
`224/27,8.36e-40`。四条均关闭；K75只中断、未获得效果结论。完整边界见`ANALYSIS.md` §59。

全局K10转向-速度时间相关探索从canonical BC fresh start、Austin-only、seed42运行30 updates；相对
speed K10唯一增加steering latent residual K10，不启用前向走廊。U30四图为`12/311、10/322、
17/378、14/387`，合计`53/1398`。相对speed K10 `85/1488`，collision `62 removed / 30 created,
p=.00111`，overtake `123 lost / 33 gained,p=2.10e-13`；安全收益主要伴随follow增加122，不是安全
超车增加。该点又被first-action-preference U44 `49/1530`严格支配，固定实例关闭，不扫描steering
hold/std或走廊叠加。完整判决见§9.36与`ANALYSIS.md` §57，实现合同见`EXPERIMENTS.md` §27、§31。

本轮四个canonical-BC fresh-start、Austin-only、seed42正式训练均已完成，并对冻结最终actor完成
四图各600 deterministic ego-scope评测。16个包共9,600条episode/trace全部通过600 unique、0 error、
finite、key-set、长度、terminal/action与typed collision合同。正式主点为：全局K10速度residual U30
`85/1488`；走廊外K10、2m前向走廊内K50双频速度residual U30 `74/1566`；100 Hz完整序列上
stride10 direct PPO loss U30 `133/1452`；online same-state branched-return PPO U45 `130/1501`。

其中双频速度探索相对同轮全局K10为collision `57 removed / 46 created,p=.3245`、overtake
`33 lost / 111 gained,p=4.58e-11`，所以它是新的高超车前沿点，不是已确认的安全改进；相对RW30
`73/1516`碰撞`38/39,p=1`而超车`32/82,p=3.14e-6`。stride10与online branch固定实例没有形成
有用前沿，当前关闭；全局K10相对BC有真实正效应，但被既有模型支配，作为机制证据归档。
collision触发的1秒前branched-return PPO仍未正式训练，也没有模型/eval目录。完整执行、目录、
数据审计与判决见`ANALYSIS.md` §56和`EXPERIMENTS.md` §30。

双频速度探索随后按预先连续checkpoint完成U27--U30四图各600收敛band：四图合计依次为
`76/1567、74/1568、84/1557、74/1566`。U27、U28与固定主点U30聚集在
`74--76 collision / 1566--1568 overtake`；U29出现一次`84/1557`的非单调回退。U27→U30配对
collision为`17 removed / 15 created,p=.8601`、overtake为`15 lost / 14 gained,p=1`；U28→U30
分别为`11/11,p=1`与`10/8,p=.8145`。因此U30不是孤立幸运点，晚段已形成稳定高超车工作区；
但U28→U29的overtake `18 lost / 7 gained,p=.0433`说明相邻checkpoint仍有身份抖动，不能写成
逐update单调收敛。U30继续作为事先固定主点，不事后改选U28。16个band包共9,600条episode/trace
全部通过600 unique、0 error、finite、key-set、长度、terminal/action与typed collision合同；U28
Hockenheim仅对一次SIGSEGV缺失场景作同模型精确补跑后重聚合，最终包完整。

2026-08-10 PPO主线已清除不参与当前训练语义的保护与旁路：`train_ppo.py`不再维护独立参数
预检或CPU设备回退，collision cache只从既有固定目录严格读取，不在训练进程内自动分类/重建；
已删除未被resume调用的scenario/vector scheduler状态恢复、policy缺失rollout buffer时的静默路径，
以及正式update前的全buffer ratio/dry-gradient诊断。保留worker异常传播与关闭、checkpoint、固定cache
身份和内容核对，以及实际collection/replay likelihood。默认production的actor、critic、reward、pool、
sampling、rollout和optimizer数学未改变；online same-state机械回归在清理后通过。

同日进一步清理`ppo/`：已关闭的prefix-reset与prefix-local joint-temporal正式训练实现、CLI、snapshot
panel loader、专用buffer字段和训练分支全部退役；历史算法与判决仍由§9.21--§9.24、§9.29和
`EXPERIMENTS.md`保留。环境不再累计或回传仅供旧诊断使用的reward分项、risk活跃比例和episode最小
净空；正式metrics不再重复写入config、update/timestep别名、preference全量样本键或逐minibatch梯度
分解。保留reward计算、P20净空、episode outcome、checkpoint和实际PPO likelihood。60项reward回归、
10,800/479 cache核对、默认逐步/K10/K10-K50探索
合同与online branch真实F110/CUDA生命周期在清理后通过；训练数学与production默认未改变。

随后把PPO内部参数集中到`ppo/ppo_config.yaml`；`train_ppo.py`加载一次并把config传给
`rollout.py`，其余需要配置的模块沿用LatticePlanner相同的`load_config()`方式，不存在独立
`ppo/config.py`。
reward系数、场景网格、P20归一化尺度、探索模式/幅度和warm-up/PPO优化参数不再在各模块顶部重复定义；
YAML项均有当前代码消费者。代码只保留361D/动作维数、critic类型、P20字段顺序和agent索引等活动结构合同。
2026-08-14又删除了first-action preference的YAML、CLI、dataset/loss、beta标定和runtime snapshot合同，
这些内容只在历史实验章节中说明。

2026-08-14场景调度继续按用户要求精简：删除默认关闭的ordinary异线高速分类/重加权分支及
对应YAML项；`RoleScenarioQueue`并入`ScenarioScheduler`的两个无放回生成器；ordinary/collision
训练起点直接复用`get_circular_startpoints(..., count * 2, 0)[1::2]`，不再保留独立距离筛选函数。
`collision_scenarios.json`的唯一读取逻辑移入`train_ppo.py`，场景生成中的一次性ID局部变量已内联，
同raceline对手直接复用ego waypoints。
四张正式地图的600条ordinary、10,800条collision候选及seed42前120次交替调度在修改前后逐项一致，
所以当前production场景集合、顺序和训练数学不变。历史重加权实验仅按`EXPERIMENTS.md`重建。

2026-08-14继续清理rollout/policy：`rollout.py`不再设置模块级config，也删除了重复的
`gym_notices`清理。薄`End2RaceRecurrentPPO`适配类已移入`train_ppo.py`，只保留SB3要求的setup、
collection、logging和train入口；buffer、warm-up、actor update与critic update实现是`rollout.py`的
普通函数，`build_model()`已删除并由入口直接实例化模型。`TrainingRecorder`已删除；setup、episode、rollout、warm-up、
formal的JSON/JSONL、checkpoint、SB3标量和终端摘要统一由`utils.log_ppo()`处理，正式update指标统一由
纯函数`utils.calculate_ppo_metrics()`计算。`policy.py`不再通过`critic_variant`运行时分支选型；只启用一个`class Critic`，
其余MLP、independent GRU和privileged MLP实现作为同名注释块保留，切换时只改注释。当前启用
`privilege_gru`，旧新29个state tensor、固定输入value和next hidden逐元素相同。删除recorder前后
2-env真实warm-up/formal run的场景JSON、2个JSONL和3个checkpoint state dict逐项一致；类移动后再次
用相同的2-env、1800步、K10/K50、U1配置验证，上述产物仍逐项一致。重复的
`collision_cache_info.json`已并入`run_config.json`的`COLLISION_POOL`字段。

同日继续清理`ppo/env.py`：删除模块级PPO config、全局planner cache、未使用的direct reset provider、
重复的gate reset/step、worker rank回传重排及privileged critic对相同位置的第二次progress投影。
reward、gate与critic共用同一个`TrackProjector`；地图路径复用`latticeplanner.utils.get_map_paths()`，
角度和环形进度复用已有函数。PPO env与`eval_multiagent.py`的opponent重规划/跟踪统一调用
`demonstration.lattice_opponent_action()`；planner动态字段由`LatticePlanner.reset()`和tracker reset管理，
env不再逐字段重建Lattice内部状态。`CentralScheduleSubprocVecEnv`仍保留父进程reset调度，因为SB3原生
worker内auto-reset不能表达当前collision/ordinary共享无放回队列。最终`env.py`由849行降为649行；
相同2-env、1800步、K10/K50、U1真实运行的场景JSON、episodes、metrics和三个checkpoint逐项一致。

随后按同一原则删除`agent_pose()`、模块级agent/reset常量、`FrontCorridorGate`和单一消费者
`LatticePlannerOpponentController`。gate几何、opponent planner episode状态与共享lattice action调用均
直接收进`End2RaceGymnasiumEnv`；`LatticePlanner.reset()`和`PurePursuitPlanner.reset()`仍由各自对象
管理内部动态状态。删除未使用的PPO render转发后，`env.py`进一步降为580行。再次执行相同2-env、
1800步、K10/K50、U1真实回归：9条episode逐字节一致，所有metric数值一致，三个checkpoint各
12个tensor逐元素一致。

PPO训练和SB3调用链不使用render；旧`metadata`虽然宣称human/rgb_array，但环境没有配置Gymnasium
`render_mode`，VecEnv的`get_images()`实际只会返回空图。该无效路径已整体删除：PPO wrapper不再转发
render，worker不再处理render命令，VecEnv不再伪装图像采集。`demonstration.py`和
`eval_multiagent.py`仍直接使用原始F110 renderer，不受影响。

`Simulator-return-filtered first-action preference`已完成seed42数据构造、canonical-BC fresh
45-update Austin训练以及U42--U45四图各600正式eval。16个包共9,600 result/trace全部为600
unique、0 error且通过finite、key、terminal与typed collision审计；预注册U44为`49 collision /
1530 overtake`，同时计数支配旧U44 `62/1478`与RW30 `73/1516`。相对旧U44为collision
removed/created `41/28,p=.1480`、overtake lost/gained `25/77,p=2.45e-7`；相对RW30为
`41/17,p=.00223`与`30/44,p=.1302`。U42--U45均形成新前沿，任务完成但更高`<40/>1500`
未达；production仍不自动切换，等待用户明确决定。见§9.33与`ANALYSIS.md` §54。

`Collision-only BC functional regularization`已完成45轮训练和U42--U45固定四图各600，共16个
正式包、9,600 result/trace；预注册主点U44为`58 collision / 1390 overtake`，没有形成新前沿，
固定实例已关闭，见§9.30与`ANALYSIS.md` §49。

`Calibrated collision-cost Constrained PPO`的第一次目录在任何formal optimizer step前因
cost buffer继承漂移停止，冻结且未评测。最小修复后的fresh `_rerun`已完整完成30个Austin formal
update和U27--U30固定四图各600。训练62行metrics、30组actor/reward/cost checkpoint、每轮16/16
actor step、reward/cost唯一化和U27--U30 strict 12-key全部通过；16个CUDA包共9,600 result/trace、
0 error并通过统一审计。U30主点为`119 collision / 1528 overtake`，Hockenheim collision高于BC；
相对U44/RW30均显著新增collision，没有形成新前沿。30/30训练pooled collision rate高于`d=.19`，
dual从1经warm-up后单调升至3.0988。固定实例已关闭，见§9.31与`ANALYSIS.md` §50；production不变。
冻结前的168/225 BC-native合同从未执行，不能赋予dataset label数量或模型性能结论；
其`run.sh`已移除，当前不排队。

2026-08-10二值front signed interaction-phase potential离线Gate的方法学更正与V2方向诊断已完成，
只读取既有BC/U44四图各600 result/trace，无新仿真、训练、actor或模型eval。V1两条绝对release
条件受source/control原risk mass `5.549x`差异混淆，已撤销科学必要条件地位；归一化释放比例为
`79.78%/84.83%`，只能写没有明确选择性。程序停止保留，但当前具体公式科学效果未决。V2用过去
`.1s`因果净空斜率得到event/提前`.5/1.0/1.5s` AUROC `.620/.722/.486/.539`；只有`.5s`有局部
信号。现有`q_vehicle`在`.5s`两组各仅3/23为正，`1.0s`两组均0/23，所以只给原risk项
增加方向门接不到早期窗口。当前仍不排训练，不否决interaction-phase类。见§9.32与`ANALYSIS.md` §53。

2026-08-09 最后一轮prefix-local joint temporal exploration的原confirmatory实例已经按exact停止
合同关闭。修复跨rollout residual语义后的第一条post-failure exploratory fresh训练完成U1--U13，
但在U14任何optimizer step前因普通batched recurrent replay超过`.02`且冻结dry-gradient裁决失败而
按合同停止。该失败只可能在exact两项仍`<=5e-5`时进入裁决，故不是相关likelihood再次错误；它证明
普通batched路径在训练中确会出现不可接受的梯度近似漂移。U14裁决明细因原实现先raise、后写metrics
而未持久化，这是记录缺陷；不得据此猜测具体batched误差或哪一条dry criterion失败。该目录冻结、
不resume、不评测，最后完整actor为U13。

post-failure exact-actor exploratory fresh训练与固定评测现已全部完成；该实验相关进程已经结束。
它从canonical BC只训练Austin、seed42、16×6,400、30 formal updates。相对上述U14失败实例，唯一
新轴是`prefix_joint_temporal`正式actor loss直接使用逐slot collection-equivalent recurrent replay；
baseline/disabled路径、样本、reward、rho、H、std、prefix集合、调度和optimizer参数均不变。训练
31行metrics、30对checkpoint均完整；30/30 actor mode均为collection-equivalent、exact最大0、
batched诊断最大`.032192`（U29，明确记为不适用于actual exact update）、每轮16/16 actor steps、
leak 0、action identity 102,400，active fraction范围`2.8418%--3.8096%`。U27--U30的16个CUDA
deterministic四图600包共9,600 result/trace全部通过key、finite、terminal、typed collision与actor/
panel身份合同；最终判决见§9.29。

第一次正式目录在warm-up和U1完整后，于U2任何optimizer step前因incoming/outgoing carry共用字段而
停止；U1 exact/batched为`0/.000995`、active`3,438=3.3574%`、72 blocks、16/16 steps，post-update
mean/max KL为`.3855/2.9517`。拆分双carry并重跑E0/E1/E2后，第二个fresh目录又在U2任何optimizer
step前按更严格的exact合同停止：`max|log ratio|=.342732`、`max|ratio-1|=.290172`，远超`5e-5`。
这证明双carry只修复“找不到历史”，没有修复历史的概率含义；第二次目录同样冻结、不resume、不评测。

准确根因是旧实现把上一rollout的physical action和observation拿到U1更新后的actor上重新计算
standardized residual，因actor mean和GRU参数已改变而把已经实现的探索历史重新解释。修复后，
上一rollout的standardized residual sum作为当前rollout起点的固定探索状态；candidate梯度只重放
当前rollout内、由本轮collection actor产生的context，且从本轮实际保存的incoming GRU hidden开始。
修复后的E0/E1/E2再次完整通过：E1两臂各102,400 transition，treatment active
`3,613/102,400=3.5283%`、75 blocks、28/28 prefix与19/19 collision source、泄漏0、action identity
102,400/102,400、GAE误差0、exact 0、batched`.006809<.02`、wall ratio`1.0275`；E2逐tensor
bitwise通过。随后增加与正式配置完全相同的warm-up+2-formal生命周期门：U2在真实U1 actor更新后
exact 0、batched`.000997`、16/16 steps、active`3,685=3.5986%`、75 blocks、泄漏0并完整结束。

该exact-actor exploratory处理保持Z6-F的28项prefix与每3次collision-role reset中1次prefix调度，只在19项
collision-source恢复后的前150步使用`rho=.90、H=50`满秩二维相关探索；边际steering/speed std仍为
`.03/.15`，actor loss仍是标准clipped PPO。原预注册写明exact越界后不得重跑，因此完成结果只能
提供post-failure exact-actor exploratory证据，不能改写成原confirmatory实例通过。它只评了
U27--U30四图各600，没有运行near400；四点均未达到产品线，production不变。

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
门。关闭当前paired collision/progress、late50、10-epoch GRU-changing auxiliary具体实例，
不调参、不进validation/PPO；不否决所有representation-changing auxiliary目标。见§9.26与
`ANALYSIS.md` §45。

2026-08-09 Round Z7 collision-only BC anchoring overlap-supported独立重开已完成，无训练。
新40起点×2 raceline×4 interval×9 speed的2,880条Austin panel与历史heldout/Austin600起点精确
零交集；BC/U44全面板和U42/U43/U45的59条潜在source共完成5,937 result/trace，0 error。稳定
eligible source为41条，精确同raceline/speed无放回control也是41条，覆盖21起点、r0/r2=`31/10`，
V0全部通过，旧Z3的support阻断已解除。branch0在82条上全部动作、pose/speed、双LiDAR最大误差
0；full-BC只救回`18/41 < 21`，虽18/18均overtake，但control新collision与overtake loss各
`4/41 > 2`。判决为严谨关闭当前canonical BC × overlap-supported stable collision × 固定1.5秒
窗口实例；没有anchor、shadow beta或actor，不扫teacher/window/support/beta。见§9.25与
`ANALYSIS.md` §44。

2026-08-09 Round Z6-F正式训练与U27--U30四图评测完成；该实验相关进程已经结束。31行metrics、
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
但nested仅69；该乐观值不承担判决。当前tested selector按预注册关闭；改变GRU表征的
auxiliary在该轮仍未测。未运行validation、actor接入或PPO。见§9.20与`ANALYSIS.md` §39。

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
50-step top-1 action-conditioned与其first-action extraction关闭；这不包含§9.33后来独立预注册并
通过的single-step simulator-return pairwise actor loss。Prefix-reset当时不准入；MoE继续因12-key边界
关闭。Constrained PPO、prefix-reset的实际训练机制和MoE架构本身均未被该Gate直接检验；
后两者分别只是未过旧程序触发条件、以及违反当前12-key部署边界，不能写成科学否决。见§9.17与
`ANALYSIS.md` §36。

同日 BC-safe anchoring Gate B已完成并科学失败，无训练。Gate A冻结的28条cohort和28条
matched controls完成branch 0、完整BC、steering-only、speed-only共224次CUDA replay；branch 0
的action、opponent action、两车pose/speed、双360D LiDAR最大误差全部为0，224条trace合同均
通过。完整BC在C层救回`10/19`碰撞且10条全为overtake，但L层恢复`0/9`，safe controls又损失
`2/28`次overtake；两个独立门失败。按预注册关闭本方向，不生成anchor dataset，不进入Gate C/D
或45-update formal训练；validation未运行branch。该Gate相关进程已经结束。见§9.16与
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
  ordinary150和actor-mismatch复用接口已经在文档固化后退役。2026-08-14又按用户决定移除
  ordinary异线高速重加权训练入口；历史模型、结果和重建语义继续保留。
- Production speed exploration 保持逐步独立速度白噪声。条件白噪声放大、全局时间相关
  速度噪声、条件时间相关速度噪声、走廊门控时间相关速度噪声、延长训练和异线高速
  重加权在历史多面板协议下均未通过。用户最新明确：后续正式验收默认必须运行
  **Austin、Hockenheim、MoscowRaceway和Nuerburgring各600 episode**；Austin600与
  三张跨地图合计crossmap1800都具有正式验收权，三张跨地图也必须逐图报告。后续不再新跑
  near400、hard73或其他附加eval；其旧数字只作历史记录。最低验收线是canonical BC：每张
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
- 旧near400诊断中U44为`37 collisions / 288 overtakes`、Production U30为`28 / 325`；这只保留为
  历史事实。按用户最新合同，near400不再运行，也不参与当前或后续模型验收、选择与翻案。
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
- **新的首选部署候选是first-action preference U44：**固定CUDA四图为`49/1530`，逐图均过BC线，
  同时计数支配旧安全端U44 `62/1478`和高超车RW30 `73/1516`。相对RW30的collision改善有配对
  确认（removed/created `41/17,p=.00223`），overtake `30/44,p=.1302`未检出恶化；相对旧U44则
  overtake改善显著（lost/gained `25/77,p=2.45e-7`），collision `41/28,p=.1480`方向改善但未检出。
  U42--U45均为`50--53 collision / 1530--1546 overtake`，不是单点瞬态。Production路径仍保持
  原U30，直到用户明确批准切换；若批准应使用预注册U44，不得事后改选U42。
- 用户把更高目标明确为四图`collision < 40`且`overtake > 1500`，同时规定**训练只能使用
  Austin**；Hockenheim、MoscowRaceway和Nuerburgring只允许测试泛化，禁止multi-map PPO。
  §9.9和`ANALYSIS.md` §28表明现有结果不是物理/actor容量上限，而是当前Austin-only PPO的
  经验前沿：U44主要保留same-line安全，重加权U30主要保留off-line-fast超车，两者修复大量
  BC碰撞同时各自新造约二十余次碰撞。当前不授权继续扫重加权比例、hold、std、gate或updates；
  若要追求新目标，先审查§28提出的Austin-only相位可分性诊断，再决定是否存在合法新训练轴。

- Production U30历史四图总量约为`94 collision / 1508 overtake`，其中Austin600为`14/366`、
  三张跨地图合计为`80/1142`。Austin保留规范逐episode 600包；三张跨地图保留目录是
  trace-only或provenance不完整，不能冒充当前规范CUDA结果包。当前正式最低对照是canonical BC，
  不是Production U30；只有未来要宣称“超过Production U30”或做正式U30配对时，才按
  根目录`evaluate.sh`的固定600合同fresh重评Production四图并保存完整result/trace。
  Production旧near400 `28/325`与hard73 `54/12`只作为已经固化的历史诊断数字，其评测产品已清理。

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

当前没有需要接续的进程。用户于2026-08-10进一步明确授权清理`scripts/`中过期测试和临时工具；
22个已完成/关闭实验的Gate、数据构造器、一次性分析器和checkpoint平均测试已删除，其算法与
重建合同保留在`EXPERIMENTS.md`。First-action preference只保留正式checkpoint、历史数据与完整文档证据；
训练期PPO模块、BC fixed-dataset builder、runtime snapshot和专用回归均已删除。显式ScenarioSpec评测器在
固定四图JSON删除后没有当前消费者，也已退役；`scripts/`当前只保留PPO探索测试与reward筛查器/测试。
更早的一次性审计脚本、notebook、NPZ、
JSON和counterfactual续跑记录也已在对应文档固化后清理。`.agents/`已经纳入Git版本管理，后续修改
必须出现在普通`git status`中。

2026-08-10再次按当前判决清理`post-trained/`、`eval_results/`和`post-trained/panels/`：永久删除
65个失败或重复run/Gate/eval/panel目标（删除前合计仍约127.01 GiB），再删除33个旧near400、hard73和
noise子面板，以及first-action preference已由canonical U42--U45替代的旧式45轮checkpoint容器。
本次清理后实测为8个`post-trained/`顶层项（6个run、collision cache、panels）、7个`eval_results/`
实验根和1个panel目录；占用约12 GiB / 108 GiB。保留对象限于§2登记的模型族、固定四图600评测、
collision cache和first-action preference历史资产。被删除的
失败run不再出现在§2模型登记中，但其科学结论、边界与停止规则继续保留。hard-neighbor 10%只作为
历史未定案结论；其训练接口和资产已经退役，不属于待完成实验，也没有因清理变成“已否决”。

当前保留目录名都是run config或manifest中的完整实验/dataset ID。`PJTE_*`、`failed_*`、旧Gate别名
已删除；不为缩短名称而改动保留run根，否则会让历史run config与磁盘身份失配。除此之外仍禁止：

- `git reset --hard`
- `git checkout -- <path>`
- 未经新授权清理当前白名单中的 `eval_results/`、`post-trained/` 资产或其他用户文件
- 覆盖 canonical BC
- 在已有 run 目录继续写入

清理后接手顺序仍是
`HANDOFF -> ANALYSIS -> GUIDE -> EXPERIMENTS`；不得因为原始目录已不存在而把文档中明确
标记为“离线候选”“被替代”“未测试”的项目改写为正式实验结论。

删除不可恢复；不得因某个旧二进制或trace已不存在而重跑已经有判决的实验。接手时只需重新列出
当前小白名单并核对§2 checkpoint路径，不要把本段数字复制成永久目录约束。

---

## 2. 模型身份登记

### 2.1 规则

模型用实验目录、update和checkpoint路径识别，不再计算或登记文件哈希。

### 2.2 基准与等价集

| 模型 | 路径 |
|---|---|
| Canonical BC | `pretrained/end2race.pth` |
| **production U30** | `post-trained/ppo_privilege_gru_clip020/update30/actor.pth` |

完整U1--U45权重与训练记录已经迁入canonical根目录。迁移时验证了短训、45-update
延长、structured-exploration control和current-code reproduction中的U30 actor等价；
这些来源容器已在2026-07-30清理，不再是合法引用入口。

### 2.3 保留的 treatment checkpoint

| 方向 | checkpoint/run |
|---|---|
| base479 U45 | `post-trained/ppo_privilege_gru_clip020/update45/actor.pth` |
| 全局速度K10 U30，机制对照 | `post-trained/ppo_global_temporal_speed_noise_hold10steps/update30/actor.pth` |
| **全局速度K10 + 前向走廊K50 U30，高超车前沿** | `post-trained/ppo_global_hold10_front_corridor_hold50_speed_noise/update30/actor.pth` |
| 前向走廊门控时间相关速度噪声、2米门宽 U30 | `post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update30/actor.pth` |
| **前向走廊门控时间相关速度噪声 U44，四图BC验收候选** | `post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update44/actor.pth` |
| 前向走廊门控时间相关速度噪声 U45 | `post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/update45/actor.pth` |
| ordinary异线高速重加权、比例0.6 U30 | `post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps_ordinary_offline_fast_reweight_0p60/update30/actor.pth` |
| First-action preference U42，稳定性点、新前沿 | `post-trained/ppo_corridor_temporal_first_action_preference_v1/update42/actor.pth` |
| First-action preference U43，稳定性点、新前沿 | `post-trained/ppo_corridor_temporal_first_action_preference_v1/update43/actor.pth` |
| **First-action preference U44，预注册主点、新部署候选** | `post-trained/ppo_corridor_temporal_first_action_preference_v1/update44/actor.pth` |
| First-action preference U45，稳定性点、新前沿 | `post-trained/ppo_corridor_temporal_first_action_preference_v1/update45/actor.pth` |

Round Z0平均权重、三条失败速度探索、prefix-reset、prefix joint-temporal、collision-only BC
functional regularization和calibrated Constrained PPO等已关闭模型的结论与数值仍保留在本文件和
`ANALYSIS.md`，但原始checkpoint和重复评测已按授权清理，因此不再出现在actor身份登记表中。

前向走廊门控时间相关速度噪声与ordinary异线高速重加权已经完整迁入canonical目录：

```text
post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps/
post-trained/ppo_front_corridor_temporal_speed_noise_0p15_hold50steps_ordinary_offline_fast_reweight_0p60/
```

前者保存U1--U45全部actor/critic和完整训练记录；后者保存U1--U30。对应评测按
`eval_results/<完整实验名>/update<N>/<MAP_NAME>/multiagents/`组织，只保留固定600场景。
First-action preference只保留正式U42--U45 canonical checkpoint；其旧式U1--U45
`checkpoints/`容器已删除，训练metrics、run config与U42--U45固定四图600评测保留。
2026-08-13汇报清理又删除了全局speed K10和K10/K50两个run内各61个中间actor/critic文件的
`checkpoints/`容器；两个U30 actor均通过硬链接保留在`update30/actor.pth`。
K75中断目录与K100/去lateral-offset门/两条失败偏好的run和eval在本次开始前已不存在，本次未重复删除。
Production U30已重新按当前固定四图600、deterministic、ego-collision、完整100 Hz trace合同评测：
Austin `14/366`、Hockenheim `26/356`、Moscow `32/385`、Nuerburgring `22/401`，合计
`94 collision / 1508 overtake`。四包均为600 unique、600同名trace、0 error，并通过finite、
terminal/action和typed collision审计。当前保留的92个正式eval包全部满足600/600、0 error和
result/trace key-set一致，没有残缺包。

历史hard-neighbor 10%仍保持“训练完成、晚期统一判决未完成、用户主动停止”的科学状态；用户决定
不再补评且训练入口已退役，其模型和Austin U1/U5/U10/U15/U20评测已于2026-08-12清理。删除不能
改写成性能失败，也不再作为当前模型登记或特殊checkpoint布局存在。

---

## 3. 当前代码与运行合同

### 3.1 入口和模块

```text
train_ppo.py            PPO训练入口、薄SB3适配类与直接模型实例化
ppo/env.py              单环境、前向走廊门与parent-scheduled一env一worker VecEnv
ppo/policy.py           actor、一个启用的Critic、三个注释备选、P20与数值hold步数速度探索
ppo/reward.py           固定reward、progress与OBB/map-wall geometry
ppo/scenarios.py        场景生成与collision/ordinary交替调度
ppo/rollout.py          recurrent buffer、统一rollout、warm-up与正式PPO更新辅助函数
ppo/ppo_config.yaml     固定运行、reward、场景和探索配置
utils.py                通用评测、赛道投影、起点与PPO训练结果记录
evaluate.sh             多车固定面板调度
eval_multiagent.py      deterministic 双车eval与numeric trace
eval_singleagent.py     单车多圈与LiDAR beam masking
post-trained/           run、checkpoint、collision-cache
```

根目录当前没有`run.sh`。168/225 BC-native固定偏好只保留历史合同，没有dataset、run或评测产物，
不是待执行入口。
旧实验的状态索引保留在 `ANALYSIS.md` §13；其中 temporal hold K10/K25 与 Group13
明确为未运行，hard-neighbor 10%明确为训练完成但无晚期统一eval结论，20%已否决。历史状态不能
授权恢复旧命令或重建`run.sh`。

新增/实验性模块的功能和测试入口由 `EXPERIMENTS.md` 记录；本文件只保留生产合同。
已完成且关闭的prefix-reset与prefix-local joint temporal不再存在于当前PPO模块或CLI；需要历史
重建时只读取`EXPERIMENTS.md`，不得把§9.21--§9.24、§9.29的记录误当成活动代码。

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
始终关闭。当前训练入口和actor early-stop分支均已删除，常规approx-KL mean仅作为训练健康指标
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
必须先在训练入口之外构建匹配cache，再通过`--collision_cache_dir`指定；missing、partial或identity
mismatch会直接失败，训练入口不会生成或覆盖cache。

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
外部fixed pool、完整805/比例hard-neighbor和actor-mismatch cache复用。2026-08-14进一步
退役ordinary异线高速重加权；它的历史结果不代表production winner。

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
--speed_noise_hold_steps               1
--front_corridor_speed_noise_hold_steps 0
front_corridor_gate_maximum_gap_m       2.0（YAML；仅前向走廊门控时间相关速度噪声）
```

已失败的`--online_same_state_branch_ppo`、未正式训练的`--collision_prefix_branch_ppo`和已完成失败评测的
`--online_collision_preference_step_fraction`及固定dataset first-action两个CLI均已从活动代码删除；
当前训练路径没有反事实动作学习分支。

已失败的`--front_corridor_ignore_opponent_lateral_offset`消融入口也已删除，走廊横向门固定回`.25m`。
当前CLI用整数hold直接表达全局时间相关动作噪声与前向走廊双频速度噪声，不再暴露
`temporal_global/corridor_temporal`文字模式。
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

上述`ppo/exploration.py`指重构前的历史实现；当前数值hold实现已并入`ppo/policy.py`，
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
collision_scenarios.json
ordinary_scenarios.json
metrics.jsonl
episodes.jsonl
checkpoints/critic_warmup.pt
checkpoints/actor_uXXXX.pth
checkpoints/critic_uXXXX.pt
```

collision cache的解析后路径和场景数量记录在`run_config.json`的`COLLISION_POOL`字段中。

常见fresh-start完成条件：`metrics.jsonl = 1 warm-up + N formal rows`、formal updates连续、
预期actor/critic checkpoint存在、所有数值finite、没有写入进程、eval面板完整。
Resume/extension需按自己的run config判断，不能只看行数。

当前`episodes.jsonl`只记录`phase`与`update`：`update=k`表示该rollout将用于训练formal actor k，
因此它仍由actor k-1采集，不能直接评价actor k。

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

当前唯一模型验收分布由根目录`evaluate.sh`直接生成：Austin、Hockenheim、MoscowRaceway和
Nuerburgring分别通过`MAP_NAME`运行，每图50个circular start × 3条opponent raceline × 4个speed
scale、interval 15，共600 episode，四图合计2,400。原`standard_multiagent_600_v1`四份JSON与
该调用链逐条完全一致、没有当前代码消费者，已于2026-08-10作为重复资产删除。Austin与训练起点
物理接近，另外三图衡量地图迁移，但这些地图已被历史实验反复查看，不能再称全新盲测集。

场景身份的权威现在是`evaluate.sh -> get_circular_startpoints() -> get_opponent_startpoint()`与每个
`results_multi.json`保存的episode key。若racetrack CSV或两个生成函数改变，必须视为评测合同变化，
并在同一新合同下fresh评control和treatment；不得静默恢复旧JSON或跨合同直接配对。

历史near400、hard73、hard334、noise和single-agent面板不再参与选择或验收，其二进制输入/评测已
在2026-08-10清理；它们的旧结论只作为本文与`ANALYSIS.md`中的历史证据。不得为新模型恢复这些
面板，也不得用它们替代固定四图600。

**面板判定口径是 ego collision scope，不是 legacy。** 这条决定了 overtake 计数：
`ego` 下对手撞墙既不终止也不改写episode，只作为事件记录（`opp_wall_event_episode_count`）；
`legacy` 下它会终止episode并把结果标成 `opp-wall`。near400 上恰有3个这样的场景，用错口径
会把 `28/325` 读成 `28/322`。`evaluate.sh`当前默认`COLLISION_SCOPE=ego`；跨包比较前仍必须确认
结果口径一致，不能把旧legacy结果混入。

Austin headline collision包含 opp-wall；做actor责任归因时必须单列ego collision。
同场景二元比较必须报告：

```text
collision: removed / created + exact McNemar p
overtake:  lost / gained + exact McNemar p
```

多个checkpoint重复同一面板时，不能把所有行当独立样本。

每张图600面板最低合同：600 unique scenarios、0 errors、有限数值、600 traces、
results/trace key-set一致、collision marker一致、terminal row语义正确。

当前保留eval根只含固定600 `multiagents/`包；没有near/hard/noise别名。前向走廊、ordinary异线
高速重加权与first-action preference的正式四图包满足完整合同。Canonical BC和production的部分
历史跨地图目录是trace-only，缺逐episode aggregate/manifest；它们只能支撑已经固化的历史数字，
若要与新actor重新配对，必须按当前固定600合同重评，不能因为目录名存在就视为规范包。

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

### 5.2 Production U30不是单点幸运值

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
  当前保留的全局时间相关速度噪声或前向走廊门控时间相关速度噪声。历史边界构建、cache
  schema和筛选算法已保存在`EXPERIMENTS.md`，保留的cache/模型产物不由production入口读取。

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
| ordinary异线高速重加权（历史，源码入口已移除） | 前向走廊门控时间相关速度噪声保持不变，仅在ordinary角色内提高异线高速场景采样权重；不是reward或optimizer改动 |

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
1146–1155超车，但near400碰撞恶化到63–77，故已否决。2026-08-14已删除活动代码中的
三组分类、比例排槽和两个`ordinary_offline_fast_*`配置；若要复现只能按`EXPERIMENTS.md`
的历史合同在独立实验分支重建。

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

本表是历史结果，不是当前磁盘资产索引。条件白噪声、全局时间相关速度噪声和条件时间相关速度
噪声三个失败run及其诊断/crossmap评测已于2026-08-10清理；判决与机制数字保留在本文和
`ANALYSIS.md`，不得因checkpoint不存在而重跑或把它们误列为当前候选。仍有行动价值的前向走廊
U44和ordinary异线高速重加权U30模型与固定四图600评测继续保留，身份见§2。

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

## 9. 后U44诊断与方法验证

完整探针、cohort R²、oracle参数、共享动作库和制动曲面见 `ANALYSIS.md` §20。

### 9.0 后U44方法族导航

以下表格是§9.14--§9.34的唯一分类入口。详细记录继续按原实验轮次保留稳定编号，避免已有
引用失效；同一类方法应从本表连续阅读，不再按临时序号查找。

| 方法族 | 详细记录 | 当前结论 |
|---|---|---|
| Reward与优化目标 | §9.27、§9.31--§9.32；基础reward见§6 | P20 collision-cost约束实例与二值front potential均未形成新前沿；只关闭已测公式/表征 |
| 训练pool与采样 | §9.8；完整pool证据见§7 | role配比证据不足；既有hard/ordinary/difficult pool实例按§7停止 |
| 探索与curriculum | §9.21--§9.24、§9.29、§9.35--§9.36；基础速度探索见§8 | K10/K50 speed形成高超车点；全局steering+speed K10退化为保守交换，固定实例关闭 |
| BC/参考策略正则 | §9.15--§9.16、§9.18、§9.25、§9.30 | collision-only BC functional regularization正式点为`58/1390`，固定实例关闭 |
| 反事实动作学习 | §9.17、§9.19--§9.20、§9.33--§9.34 | frozen selector失败；first-action preference形成新前沿；online same-state branched PPO仅完成机械验证 |
| 表征与辅助监督 | §9.13、§9.19、§9.26 | 50步显式历史与paired collision/progress GRU auxiliary具体实例关闭 |
| Checkpoint组合 | §9.14 | U42--U45等权平均失败，当前组合方式关闭 |
| 诊断与工程Gate | §9.1--§9.13、§9.21--§9.23、§9.28 | 只承担机制、恢复和证据级别判断，不是模型eval |

### 9.1 Actor现有表征包含关键信息

冻结production U30 actor的线性探针显示：

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

判决：关闭当前fixed-library 50-step action-conditioned controllability和top-1 first-action
extraction形式；不得改动作库、阈值或只报oracle上界重试。这不否决后来只替换single step并直接
更新最终actor log-prob的§9.33方法。prefix-reset没有出现“early通过、late失败”的旧程序
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
104不再作为同budget优于fixed的证据。关闭不是hidden信息论上限；会反传进student GRU的
representation-changing auxiliary在该轮仍未测。完整证据见`ANALYSIS.md` §39。

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
关闭本实例；改变GRU表征的paired action-response auxiliary、collision-only BC regularization与
Constrained PPO不由本实验代判。完整证据见
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
解决跨状态/地图的安全--progress统一选择；不科学否决prefix-reset方法类。改变GRU表征的
paired action-response auxiliary、collision-only BC regularization
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

本轮是`GRU-changing paired action-response auxiliary`的首次直接检验：复用456条/70起点
development与late 13-action真实结局，control在frozen
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
不运行独立validation/PPO，不扫epoch/LR/loss/操作点。该结果不否决所有改变GRU表征的辅助监督。

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
以及记录一条按不适用前提被关闭的轴。完整核验、更正与推导已固化在`ANALYSIS.md` §47。

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

三条由上表得到的判断（已按`ANALYSIS.md` §47的独立审计更正）：

1. 有限可部署点显示明显的安全--超车 trade-off，但证据级别不完全相同，不能据此证明它们位于
   同一条稳定前沿，也不能断言所有新机制都只会沿线滑动。
2. 重加权 U30 是唯一在四图两轴上同时优于 production 的点（`-21` collision、`+8` overtake），
   但证据级别低于 U44/SWA/BC。**切换 production 前必须补一次固定 CUDA 四图确认**，同时
   production 自身缺 Moscow/Nuerburgring 规范包，当前比较是"重建对 headline"。
   补齐成本为 `2400 + 1200 = 3600` episode、零训练。
3. prefix-reset U30 `103/1522` 与重加权 U30 `73/1516` **互不支配**：重加权少30次碰撞，
   prefix-reset多6次超车。§9.24“高超车端扩展、目标未完成”的判决保持不变；不得把明显的
   安全差距改写成数学上的双轴支配。

**一条按不适用前提被关闭的轴（需要用户裁定，未授权）。** §10.1 第 16 条关闭 side-phase
steering exploration 的理由是"缺少可靠部署期 conditioning"。已核实的两个事实是：

- 历史上全部 8 条训练臂的 `STEERING_LATENT_STD` 一律为 `0.03`；准确边界是转向探索的幅度、
  时间相关性和训练期门控从未改变，不是训练时从未采样steering动作。§18的五组探索实验全部
  只动速度通道；
- 探索是纯训练期机制：`eval_multiagent.py` 评测时直接取 mean action、无噪声无门；
  `End2RaceGymnasiumEnv._front_corridor_gate()`本身条件在模拟器特权几何上，**同样没有部署期 conditioning**。

因此该理由适用于**部署期相位门控**，不适用于**训练期探索门**。相关已测证据（新造失败的
接触几何以近平行侧/后擦碰为主、相对 yaw 中位 `3.67°`；Z2 最强单一固定干预是
`steer +0.02 / speed +0.5`，带转向分量）指向横向通道。**这是未测假说，不是发现**；在用户
明确裁定并重新预注册前，§10.1 第 16 条继续有效，不得据此开跑。

### 9.28.1 后续方法族状态总览

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
- representation-changing auxiliary大类不是信息论上被否决；Z8已首次直接训练student GRU，
  并关闭paired collision/progress、late50、10-epoch具体实例。其他全新监督仍属新方法，但不得
  调Z8参数重试；
- MoE架构本身未测，但在最终actor必须保持12-key兼容的当前任务合同下工程不合法；只有用户
  明确放弃该边界才值得科学检验。

Z6、Z7、Z8与原Z9 `P20 × d=.10`具体实例都已完成各自允许的最深Gate或formal并按停止规则关闭；
用户随后明确授权的§9.30 collision-only BC functional regularization和§9.31 calibrated
collision-cost Constrained PPO也都已完成训练、固定四图600与统一审计。
两条固定实例均未形成新前沿并已关闭；当前没有活动或排队中的训练臂。Calibrated
collision-cost Constrained PPO不能写成旧Z9 `d=.10` preflight通过，两轮的budget、治理角色和
证据边界保持独立。

用户随后授权的最后一轮exploration已经完成，完整判决见§9.29。它保持Z6-F的28项prefix与reset
合同，只在19项collision-source恢复后的前150步加入`rho=.90、H=50`的满秩二维时间相关探索；
边际steering/speed std仍为`.03/.15`。该臂是source gate + steering/speed时间相关性的单一方法级
复合treatment，不识别各组件贡献。最终四个固定600点全部失败，不再处于执行队列。

### 9.29 Prefix-local joint temporal exact-actor exploratory（2026-08-09，训练与四图600完成）

原confirmatory先后在U2暴露cross-rollout carry覆盖与旧动作被新actor重新解释两项概率实现错误；
按预注册exact停止合同已经关闭。修复后第一条post-failure batched fresh run完成U1--U13，但U14
普通batched replay达到`.02`触发线且dry-gradient裁决失败，任何U14 optimizer step前停止。裁决
细项因raise先于metrics持久化而缺失，只能确认batched近似梯度不可继续使用，不能猜具体cosine/L2。

最终单轴修订让`prefix_joint_temporal`正式actor loss直接使用逐slot collection-equivalent replay，
其余训练合同不变。修订后E0七项与全规模warm-up+U1+U2 lifecycle通过；正式fresh run完成1行
warm-up+30行formal、30对actor/critic。30/30 `actor_replay_mode=collection_equivalent`、exact最大0、
batched诊断最大U29 `.032192`并记为`not_applicable_exact_actor_replay`，每轮16/16 actor steps、
leak 0、action identity 102,400，active fraction`2.8418%--3.8096%`、block数60--82，全部finite。
U27--U30 actor均strict 12-key。这个修订消除了实际PPO梯度的batched近似混淆，但其证据级别仍是
`post_failure_exact_actor_exploratory_not_original_confirmatory`。

16个固定CUDA deterministic包共9,600 episode/result/trace、0 error；四个panel均600 unique，
actor/panel路径、result/trace/panel key、数值finite、数组对齐、terminal/action-applied与typed
collision合同全部通过。没有运行near400。逐图`collision/overtake`：

| update | Austin | Hockenheim | Moscow | Nuerburgring | 四图 | 四图BC门 | `<40/>1500` |
|---:|---:|---:|---:|---:|---:|---|---|
| U27 | `31/358` | `38/355` | `36/386` | `27/395` | `132/1494` | 失败 | 失败/失败 |
| U28 | `29/362` | `38/361` | `38/388` | `27/397` | `132/1508` | 失败 | 失败/通过 |
| U29 | `25/363` | `36/362` | `34/388` | `26/397` | `121/1510` | 失败 | 失败/通过 |
| U30 | `26/366` | `36/361` | `32/388` | `23/398` | **`117/1513`** | **失败** | **失败/通过** |

U30相对BC（129/1445）：collision removed/created `67/55, p=.3193`，总数少12但未检出配对差异；
overtake lost/gained `29/97, p=9.24e-10`，显著增加68。相对U44（62/1478）：collision
removed/created `44/99, p=4.89e-6`，显著净增55；overtake lost/gained `53/88, p=.00403`，显著
净增35，是明确的安全--超车交易。相对同机制前身Z6-F U30（103/1522）：collision
removed/created `33/47, p=.1456`，overtake lost/gained `43/34, p=.3620`；aggregate多14碰撞且少9
超车，双轴被Z6-F U30支配，但两项配对差在本样本下不显著。U27--U30也都在aggregate上被各自
Z6-F同update双轴支配；U27/U28/U29的collision配对p分别`.00280/.000639/.02652`，U30为`.1456`。

预注册U30 L2-M三条只通过超车保有量；`collision<=91且p<.05`失败，相邻U29/U30相对Z6-F的
collision方向也失败，故机制门失败。四个固定点均未达到L3，`formal_600_task_achieved=false`。

判决：**精确相关探索的训练机制成立，但当前复合treatment没有优化现有PPO，tested配置关闭，
production不变。** 它把actor推到高超车/高碰撞区，U30还被更简单的Z6-F U30双轴支配；因此不得
扫描rho、H、std、prefix比例、窗口、LR或延长update来重试。该结果严谨否决当前
`28项mixed prefix × 19 collision source × rho=.90 × H=50 × .03/.15 × exact actor replay`
实例，不否决所有collision-only prefix或所有时间相关联合探索方法类。重开必须提出新的机制级
证据，不能只改剂量。

### 9.30 Collision-only BC functional regularization正式训练（2026-08-09，完成并关闭固定实例）

Collision-only BC functional regularization从canonical BC fresh start，只训练Austin、seed42、
45 updates；保持产生历史U44的
`corridor_temporal`探索与全部PPO参数，唯一新增项是18条Z7 rescued-overtake source、每条150步的
canonical BC functional anchor，固定`beta=.006405998602049812`。U44在训练前固定为唯一主点，
U42/U43/U45只记录band。训练1行warm-up+45行formal全部finite，每轮16/16 actor step、beta固定、
18条anchor身份不变；anchor loss从初始化`9.42e-14`漂移到U45 post-update `.28627`，说明该loss
真实进入GRU与output head，而不是零作用路径。

16个CUDA deterministic四图600包共9,600 result/trace、0 error；全部panel 600 unique，
result/trace/panel key一致，数值、数组、terminal与typed collision合同通过。U42--U45分别为：

| update | Austin | Hockenheim | Moscow | Nuerburgring | 四图 |
|---:|---:|---:|---:|---:|---:|
| U42 | `23/309` | `7/324` | `14/364` | `18/380` | `62/1377` |
| U43 | `23/302` | `10/319` | `13/364` | `18/376` | `64/1361` |
| **U44主点** | **`22/314`** | **`6/328`** | **`15/364`** | **`15/384`** | **`58/1390`** |
| U45 | `19/317` | `14/322` | `13/366` | `19/375` | `65/1380` |

主点相对历史U44 `62/1478`：collision removed/created `40/36,p=.7310`，净少4但未检出；
overtake lost/gained `119/31,p=2.34e-13`，显著净少88。相对BC `129/1445`为collision
`106/35,p=1.69e-9`、overtake `98/43,p=4.15e-6`；它显著更安全但也显著更少超车。四图overtake
`314/328/364/384`全部低于各自BC下限，故逐图BC门全失败；也未Pareto支配U44或RW30。只有
Hockenheim的collision相对历史U44可检出改善（`16→6`，removed/created `13/3,p=.0213`），其他
三图没有建立collision净收益。

最终判决：**regularizer机制执行成立，产品接受失败；当前固定实例关闭，production不变。**
不得扫描beta、teacher/window、anchor重采样、lost-overtake混入、训练长度或从相邻band选点重试。
该结论只覆盖`18 anchors × 150 steps × canonical BC × fixed beta × corridor-temporal × 45 updates`
的单seed实例，不否决所有collision-only BC regularization。完整逐图配对、质量异常包边界和机制
解释见`ANALYSIS.md` §49。2026-08-10按用户授权退役`collision_anchor.py`及其两个训练CLI；历史
实现可按`EXPERIMENTS.md` §22重建，但当前源码不能再启用该已关闭方法。

### 9.31 Calibrated collision-cost Constrained PPO正式训练（2026-08-10，完成并关闭固定实例）

第一次`direct_formal`目录在warm-up和formal rollout 1后、任何formal optimizer step前因
`ConstrainedRolloutBuffer`遗漏父buffer新增joint-temporal字段展平而停止；它没有actor checkpoint
或性能结论，冻结且不评测。修复为只展平四个cost数组、其余字段委托父buffer后，fresh `_rerun`
保持canonical BC、Austin、seed42、16×6,400、30 updates、P20 cost critic、`d=.19`、
`lambda0=1`与dual LR`.5`不变。

Rerun完整得到2行warm-up、30行formal和30行constrained formal；30/30轮16/16 actor step，
30组actor/reward/cost checkpoint齐备，reward collision去重误差0且cost event与完成collision
episode逐轮相等。Cost critic post-update EV在U27--U30为`.5177/.5114/.4947/.5357`，证明cost
链路工作；但30/30 pooled collision rate都高于`.19`，范围`.2394--.4451`，dual从1经warm-up后
单调升到`3.0988`，没有出现约束平衡。

16个固定CUDA deterministic四图600包共9,600 result/trace、0 error，全部通过600 unique、
actor/panel身份、key、finite、terminal和typed collision合同。结果为：

| update | Austin | Hockenheim | Moscow | Nuerburgring | 四图 | 逐图BC门 |
|---:|---:|---:|---:|---:|---:|---|
| U27 | `22/371` | `33/363` | `32/394` | `22/398` | `109/1526` | 失败 |
| U28 | `16/374` | `27/371` | `37/394` | `26/397` | `106/1536` | 通过 |
| U29 | `23/371` | `28/371` | `33/394` | `25/397` | `109/1533` | 失败 |
| **U30主点** | **`22/372`** | **`37/365`** | **`35/394`** | **`25/397`** | **`119/1528`** | **失败** |

U30相对BC `129/1445`：collision removed/created `51/41,p=.3481`，overtake lost/gained
`17/100,p=1.73e-15`；相对U44 `62/1478`为`38/95,p=8.25e-7`与`38/88,p=9.85e-6`；相对
RW30 `73/1516`为`26/72,p=3.69e-6`与`29/41,p=.1882`。因此它相对U44是显著增加collision
换显著增加overtake，相对RW30则显著增加46次collision、净增12次overtake未检出，不是新前沿。
U28虽在本band内双轴支配U27/U29/U30，仍未支配U44或RW30，且不得事后替代预注册U30主点。

判决：**cost/dual与actor更新机制成立，约束和产品接受均失败；当前固定实例关闭，production
不变。** Budget由历史U44 `26/141`固定，但U44使用corridor-temporal探索、本臂使用baseline
independent Gaussian，不能把`.19`称为在本臂分布上已证明可达。不得扫描budget、dual、P20、
reward/cost定义、训练长度或从U28选点重试。结论只覆盖该固定组合，不否决所有Constrained PPO；
合法重开须有新的cost representation/constraint目标或匹配本臂分布的可达性证据，而非只改数值。
完整机制、配对和证据边界见`ANALYSIS.md` §50。

### 9.32 二值front signed interaction-phase potential离线Gate（2026-08-10，无训练）

候选保持当前`gamma*Phi(s')-Phi(s)`与`.05`上界，只把vehicle shortfall乘以二值front：对手中心
在ego车体纵向前方且OBB横向投影重叠时为1，否则为0；wall potential不变。Source为BC无ego
collision而U44为ego-opp collision的23条；control为同地图、opponent raceline、speed且初始ego
位置最近的U44安全超车，23条无放回，20条同时也是BC安全超车。Source/control事件分别为首次
ego-opp collision与全局最小OBB净空，比较前150个transition。

Source front覆盖在事件/提前`.5/1.0/1.5s`为`4/3/3/1`，control为`0/0/3/3`。Current→signed的
负risk shaping mass在source为`1.123715→.227223`，释放`.896492`；control为
`.202517→.030730`，释放`.171787`。23个matched control-source released为正/零/负
`4/1/18`，均值`-.031509`、中位`-.047306`；距离`<=10m`的20对仍为`2/1/17`。执行后复核确认
这两条绝对量条件无效：source/control原负risk mass先验相差`5.549x`，而归一化释放比例为
`79.78%/84.83%`。逐episode比例又只有8对双侧分母非零，正/零/负`3/4/1`、中位0。准确表述是
没有观察到明确选择性，不是方向反转；V1 machine fail保留为程序事实，不构成科学必要条件证伪。

全四图2,400条U44 episode共有4,352次front翻转，但只有35次产生非零potential跳变、29次产生
非零新增shaping、17次超过`.02`；P99新增shaping为0。因此抖动只保留为实现风险，不是停止原因。
本轮也没有用任意抖动阈值卡Gate。

V2固定过去`.1s`因果方向，未截断normalized OBB distance closing-rate在event/提前
`.5/1.0/1.5s`的source-outcome AUROC为`.620/.722/.486/.539`；`.5s`按21个地图+source-startpoint
cluster bootstrap的95%为`.583--.843`，但未跨更早窗口稳定。当前potential实际使用的截断`q_vehicle`在`.5s`的source/
control各仅3/23为正，`1.0s`两组均0/23为正；所以只在原shortfall上乘方向项不会在
`.5--1.5s`产生足够新信用。AUROC只作当前轨迹诊断，不是动作价值或方法类Gate。

判决：**既有预算优先级下保持不实现、不训练、不扫front/overlap/potential/迟滞；这是程序/资源
停止，当前二值公式的训练效果仍未测试。** 更合理的新公式必须在Austin development侧把方向信号
放入当前shortfall support之外，或改变支持范围，并与L12停止规则对账；不得利用四图正式身份调
公式后再称为干净泛化证据。不否决PBRS或interaction-phase方法类。完整定义与边界见
`ANALYSIS.md` §53。

### 9.33 Simulator-return-filtered first-action preference（2026-08-10，训练与固定四图600完成并通过）

该方法针对Z2“动作存在但frozen selector不能跨startpoint稳定选择”的缺口，不再训练额外selector。
它在同一个seed42 Austin snapshot上只替换第一步动作，后续恢复同一冻结U44；terminal
`(no ego collision,overtake)`严格Pareto更好的动作通过pairwise softplus直接提高最终student
actor的相对log probability。正式actor loss为PPO加preference，因此不是纯PPO；部署仍为361D、
strict 12-key，不保留teacher、snapshot、future event或辅助head。

计划1,179个state中1,173可用、6个当前重放提前终止；所有可用snapshot/noop round-trip精确。
历史outcome漂移93/1,173只作诊断，当前标签全部来自同seed42 noop/candidate。14,076条candidate
branch得到target `46 episode/83 state/337 pair`、control `19/34/103`，共440 pair；candidate/noop
preferred为200/240。正式run从canonical BC fresh start，只训练Austin，完成1行warm-up和45行
formal，每轮16/16 actor step。首轮step-space标定冻结`beta=.0059823916`；target margin mean从
U1 `-1.4058`升到U45 `+.5059`，证明loss确实改变最终actor。

16个固定四图600包共9,600 episode/trace、0 error，全部通过unique/key/finite/terminal/typed
collision审计。结果为：

| update | Austin | Hockenheim | Moscow | Nuerburgring | 四图 |
|---:|---:|---:|---:|---:|---:|
| U42 | `11/377` | `8/379` | `18/393` | `16/397` | `53/1546` |
| U43 | `11/372` | `8/372` | `19/388` | `12/401` | `50/1533` |
| **U44主点** | **`8/372`** | **`9/375`** | **`18/386`** | **`14/397`** | **`49/1530`** |
| U45 | `13/368` | `10/377` | `15/390` | `14/398` | `52/1533` |

U44逐图BC门全过。相对旧U44 `62/1478`：collision removed/created `41/28,p=.1480`，
overtake lost/gained `25/77,p=2.45e-7`；相对RW30 `73/1516`：`41/17,p=.00223`与
`30/44,p=.1302`。因此它同时计数Pareto支配两条旧前沿；相对RW的安全改善有配对确认，另一轴
未检出恶化。U42--U45四点都形成新前沿，排除只靠U44瞬态。更高`collision<40/overtake>1500`
未达。

判决：**固定实例通过，是当前首选部署候选；production路径仍等待用户明确授权后才切换。** 不扫
beta、动作库、lead、pair权重/温度、学习率、update数或事后选U42。合法新研究必须改变科学对象
或做多seed复现，不能把本次成功变成未授权调参。单seed、Austin-only训练、固定四图以及包内未
原生生成`eval_manifest.json`是证据边界；Moscow U44曾在完整落盘后teardown返回139，同参数重跑
正常且结果相同，最终包完整。完整数据与机制见`ANALYSIS.md` §54，实现合同见`EXPERIMENTS.md` §25。

后续固定trace诊断进一步限定了收益来源：相对旧U44的41个removed collision中33个为off-line，
36个最终overtake；28个created中19个为same-line。旧U44 collision的same/off结构`21/41`迁移成
新U44残留`32/17`。新actor在2,400条中1,990条episode平均desired speed更高，总均值
`+.1176m/s`；它主要清空异线侧后接触并完成高速超车，而不是四图一致学到更安全规则。三张非训练
地图合计collision `44→41`（`25/22,p=.771`，未确认安全泛化），overtake `1134→1158`
（`20/44,p=.00369`，确认progress泛化）。49个残留为47车辆/2墙，车辆中side/rear `40/47`、
yaw<=30度 `45/47`，same-line `32/49`；后续若追求`<40`应只针对该平行侧后接触族，不再扫全局
速度/惩罚。完整分层见`ANALYSIS.md` §55。

### 9.34 Online same-state branched PPO（2026-08-11，正式训练与四图600已完成）

该臂是对§9.33上一代数据闭环的结构性替代，不复用其440个pair、U44 snapshot/hidden/action或任何
旧PPO actor。每个当前student formal rollout按不看未来的固定时序取16个on-policy Austin状态，当前
冻结`pi_old`在同一状态采4个第一动作并各自继续最多100步；同状态leave-one-out return advantage
进入`.10`权重的第二个clipped PPO surrogate。主rollout worker与torch RNG在分支后恢复，部署端仍是
原361D strict 12-key actor。它没有模仿loss，但因额外分支stratum与固定权重，不称为完全未改动的
标准PPO。

正式45-update训练完整：45轮各有actor/critic checkpoint，final actor为strict 12-key。累计从720个
on-policy状态生成2,880条branch与266,795额外simulator step；branch终局为horizon 2,463、ego collision
197、overtake 156、follow 64，平均每轮advantage std `.03638`。85.5%分支只到100步horizon，说明
固定时序全状态抽样主要提供短期return差，而不是富集collision/overtake决策。

U45四图依次为Austin `23/359`、Hockenheim `33/361`、Moscow `42/388`、Nuerburgring `32/393`，
总计`130/1501`。相对BC collision `55/56,p=1`、overtake `24/80,p=3.22e-8`；相对RW30显著多
collision且少15次overtake，相对first-action preference U44也显著多collision并少29次overtake。
判决：**当前16 states × 4 actions × 100-step horizon × coefficient .10固定实例关闭，不扫描剂量。**
它证明在线branch能产生并更新信号，但没有把该信号转化为可用安全前沿；不外推否决所有collision
富集的online branch设计。完整证据见`ANALYSIS.md` §56。

### 9.35 四个PPO实验接口的历史状态（2026-08-11；活动代码已收口）

| 描述性方法 | CLI | 已验证机制 | 性能状态 |
|---|---|---|---|
| 全局K10时间相关速度探索 | `--speed_noise_hold_steps 10` | residual每10步换一次，actor mean仍100 Hz | U30 `85/1488`；优于BC但被既有前沿支配 |
| 走廊外K10/2m走廊内K50双频速度探索 | 上项再加`--front_corridor_speed_noise_hold_steps 50` | gate进出立即换块，走廊内连续50步 | U27--U30=`76/1567、74/1568、84/1557、74/1566`；保留U30固定高超车主点 |
| 100 Hz序列、10 Hz direct PPO loss | 历史CLI已删除 | 200位置选20个loss位；100 Hz GAE/replay保留 | U30 `133/1452`；方法退役，模型/eval与实现已清除 |
| collision触发的1秒前同状态分支PPO | 历史CLI已删除 | 真实collision恢复1.00s前；4候选产生3 collision/1 horizon，advantage std `.9270` | 只完成机械测试，未训练，接口退役 |

默认`speed_noise_hold_steps=1`、`front_corridor_speed_noise_hold_steps=0`，继续走当前production PPO数学。
历史碰撞回溯不保存
逐步大snapshot，而是保存episode起点snapshot与ego action，碰撞后确定性重放到event-1.00s；parent
最近100步observation/hidden必须与重放snapshot逐位一致。其4候选和后续continuation均来自当前冻结
`pi_old`，不复用上一代PPO数据；无有效collision prefix的rollout只执行原main PPO。

准确边界：K10/K50使用per-transition marginal log-prob，不是K步joint block likelihood；10 Hz只抽
formal actor/value loss位置，不降低环境、GAE或GRU频率；collision-prefix使用训练期collision hindsight
选状态，但部署actor没有future输入或runtime branch。完整实现和测试合同见`EXPERIMENTS.md` §27--§29。
前三项已有四图600性能结论；collision-triggered一项只停留在机械测试，不能写成性能否决，但在
当前“不新增tech”的范围冻结下也不是待办，活动实现已经删除。
双频速度探索U27--U30的完整收敛band见`ANALYSIS.md` §56.2；它支持U30端点可重复，但不授权
checkpoint选择或把collision方向改写为已确认安全改善。

### 9.36 全局K10转向-速度时间相关探索（2026-08-11，训练与四图600完成并关闭）

固定处理从canonical BC开始，只在Austin训练、seed42、30 formal updates；speed latent residual和
steering latent residual均每10个`.01s` step重新采样，两个残差独立，actor mean仍100 Hz更新，
steering/speed边际std仍`.03/.15`。`front_corridor_speed_noise_hold_steps=0`，所以没有前向走廊gate；
相对§9.35 speed K10主对照唯一新增steering K10。

训练1行warm-up、30行formal metrics、30组actor/critic和4,211个episode完整finite。U30 strict 12-key
actor以对应run的U30 checkpoint路径识别。四图各600
result/trace全部通过unique、key-set、finite、terminal/action和typed collision合同；依次为Austin
`12/311`、Hockenheim `10/322`、Moscow `17/378`、Nuerburgring `14/387`，合计`53/1398`。

相对speed K10 `85/1488`的同场景配对：collision `62 removed / 30 created,p=.0011109`，overtake
`123 lost / 33 gained,p=2.10e-13`。62次collision removal中只有24次变为overtake、38次变为follow；
四图desired speed均值下降`.2483m/s`，净follow增加122。结果是明确的全局保守化交换，不是更连贯
steering带来安全超车。Production U30当前已有完整四图raw包，但本treatment原始eval已按授权清理，
所以两者仍不能重新计算paired p；保留当时已固化的历史比较，不得用缺失一侧补造配对统计。

判决：**固定全局steering K10 + speed K10实例关闭。** 它在Austin、Hockenheim、Nuerburgring未守住
逐图BC overtake下限，并被first-action-preference U44 `49/1530`以更低collision和更高overtake严格
支配。不得继续扫描steering hold/std或直接叠加走廊K50；本结论不外推否决所有时间相关steering探索。
完整证据与边界见`ANALYSIS.md` §57。

## 10. 当前允许与不允许的下一步

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
29. 重跑或扫描calibrated collision-cost Constrained PPO的P20 cost critic、`d=.19`、lambda初值1、
    dual LR.5、reward/cost定义、
    30-update长度或事后选择U28；U30为`119/1528`且相对U44/RW30显著新增collision，当前固定实例
    已关闭。新的constraint目标或cost representation是新方法，不由本条代判。
30. 重复V1的两条绝对release Gate、或扫描二值front阈值、OBB overlap、potential magnitude、
    迟滞和平滑参数。原`4/1/18`受`5.549x`基线risk mass差异混淆，不能再作为科学否决理由；
    当前二值公式只是预算上不排队，训练效果未测试。V2又表明未截断方向仅`.5s`有中等信号，而
    原shortfall在`.5--1.5s`几乎无support。只有把相对运动方向置于有效提前窗口的新公式才按新
    方法重开，并必须与L12边界对账；若用户明确授权直接训练当前低先验公式，应另写效果实验合同，
    不能把本诊断改写成已通过。
31. 重跑或扫描`ppo_loss_sample_stride10`的stride、随机相位、epoch或学习率；固定stride10 U30为
    `133/1452`，相对BC两轴均未检出，相对现有前沿明显更差。它只否决当前100 Hz replay上的
    1/10 direct-loss实例，不否决真正10 Hz环境/actor/GRU，因为后者从未测试。用户已明确退役该方法；
    `post-trained/ppo_loss_sample_stride10`、对应`eval_results`以及专用CLI/mask/metrics/test实现均已删除。
32. 重跑或扫描online same-state branched-return PPO的16 states、4 actions、100-step horizon、`.10`
    coefficient或45-update长度；固定U45为`130/1501`并被RW30与first-action preference U44计数支配。
    collision-triggered一秒前branch是不同的状态选择机制，仍未训练，不能被本条连带关闭。
33. 重跑或扫描全局steering K10 + speed K10的steering hold、std，或直接叠加前向走廊K50；固定U30
    为`53/1398`，相对speed K10以显著超车损失换取安全，并被first-action-preference U44严格支配。
    该停止只覆盖当前全局K10组合，不否决所有时间相关steering探索。

### 10.2 未完成但不是高优先

- 训练期辅助表征学习（PPO + auxiliary representation loss）：§31.6规定的准入预检已在
  §32执行完毕，**结论是缺口不存在且符号相反**——维度匹配下GRU hidden在9/9几何目标上优于
  actor输入。因此以该9个几何/速率量族为辅助目标的训练**不准入**。这不否决辅助表征学习
  整体：换用不属于该量族的目标（对手planner意图代理、ego动作序列可达性量等）必须按§32.2
  同一口径重做缺口预检，不得继承结论或跳过预检。Round Z4-A已进一步否决固定50步历史与
  12动作三分类结果监督；Round Z8又直接训练原GRU并否决paired collision/progress的具体
  GRU-changing auxiliary实例；
  新目标不得复用两者失败配置做参数扫描；
- temporal hold K25（0.25s）仍仅提出、未训练。K75（0.75s）在U20中断并按用户决定清理，效果未知；
  K100（1.0s）U30正式结果`110/1425`，相对K50 `74/1566`两轴显著变差，固定长度关闭。不得用K100
  失败把未完成K75或未训练K25改写成已否决；也不再为补齐连续长度曲线而重跑它们。
- collision触发的1秒前同状态分支PPO只完成过机械测试、没有正式性能结论；其活动接口已删除。
  在当前范围冻结下它不是未完成待办，也不恢复实现；历史状态仍不能改写成科学失败。
- Group13 GRU/head LR 2×2：未运行；
- outcome-aware hard：历史上有实现、没有独立A/B；源码已删除，只有重建合同。

不得把这些写成已否决。

已完成项不再列在本节：全局speed K10 U30作为被既有前沿支配的正机制证据归档；K10/K50 U30
`74/1566`保留为高超车前沿；全局steering+speed K10 U30、stride10 U30与online same-state U45
固定实例关闭。不得把前两者误写成未训练，也不得因后三个固定实例失败外推否决所有时间相关
steering、所有时间降采样或所有在线branch方法类。

Hard-neighbor 10%另行归档为“训练完成但晚期eval未完成、用户主动终止”；当前入口已退役，
不再列为可继续执行的下一步，也不得写成已证伪。

### 10.3 合法重开条件

本节只保存历史上的科学重开边界，不构成当前开发授权。用户已在2026-08-13冻结新tech；除非用户
今后明确撤销该冻结，不得据此设计、实现或运行新方法，也不得把讨论清单转成`run.sh`。

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

## 11. 既定实验与评测执行规则

当前不接受新的实验设计或tech实现，也没有待运行的`run.sh`。以下只保留为用户未来明确解除冻结后的执行规则。

1. 使用conda环境 `end2race` 的绝对Python。
2. 长任务使用tmux，避免意外中断。
3. 优先修改现有脚本参数；不要为每次运行新建bash/log。
4. 无待运行实验时不创建`run.sh`；不得加入Lattice+BC、Q/cost critic、group-robust、reverse
   constrained或其他讨论方案，也不得把已完成方法以注释形式堆回脚本。
5. 每次只改一个轴；若不得不多变量，预注册每个变化和不可分离的限制。
6. 从canonical BC fresh-start，除非问题明确是checkpoint continuation。
7. 不使用自动checkpoint选择器；预先固定update或报告完整checkpoint band。
8. 新actor只评Austin、Hockenheim、MoscowRaceway和Nuerburgring各600 episode；
   Austin600与crossmap1800都具有验收权且跨地图必须逐图报告；每图ego collision不得高于
   canonical BC且overtake不得低于canonical BC，opp-wall单列。不得新跑near400、hard73、
   其他hard/near/noise/startpoint/single-agent或额外地图eval；训练侧Gate/branch不算模型eval。
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
pgrep -af '[r]un\.sh|[t]rain_ppo\.py|[r]un_constrained_ppo\.py|[e]val_multiagent\.py|[e]valuate_scenario_panel\.py|[e]valuate\.sh' || true
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
> 但无泄漏nested只有69；故当前tested selector关闭，不进入validation或PPO；改变GRU表征的
> auxiliary在该轮仍未测。
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

## 13. 2026-08-12第一动作偏好结果与活动代码收口

collision-triggered临时偏好和64-target/64-control canonical-BC固定偏好均完成30-update训练与
U27--U30四图各600评测。固定BC数据实际只有13个target、5个control labeled episode（75/20 pair）；
U30为`67/1436`，相对K10/K50对照collision `55 removed/48 created,p=.5546`、overtake
`159 lost/29 gained,p=6.43e-23`。在线臂30轮产生1,019个pair、执行37,752条真实terminal branch；
U30为`85/1466`，相对对照collision `33/44,p=.2543`、overtake `122/22,p=5.45e-18`。两条机制都
真实生成偏好并更新最终actor，但都确认损失大量超车、没有新安全前沿，固定实例关闭。完整结果与
证据边界见`ANALYSIS.md` §59。

2026-08-13活动代码先行收口：已删除online collision preference的CLI、rollout collector、环境
snapshot/action-history旁路、临时dataset/loss、metrics与专用回归；已删除移除走廊lateral-offset门的
CLI和环境分支；已删除正式失败的全局steering hold CLI及policy temporal steering state，steering恢复
每个`.01s` step独立采样。历史算法仍由`EXPERIMENTS.md`保存，但不能直接运行。

2026-08-14按用户决定移除其余全部first-action活动代码：固定dataset loader、softplus pair loss、
PPO/preference步长比例beta标定、两个训练CLI、四个YAML batch/action参数、环境runtime snapshot与
restore、executed-action旁路、BC构建脚本和专用测试均已删除。旧实现逻辑是：在collision或safe-control
事件前150/100/50步保存环境状态，只替换第一步动作，随后回到冻结actor闭环；只为terminal结果在
`(no ego collision, overtake)`上严格Pareto更优的动作生成good/bad pair；训练时target/control按episode
等权，用`softplus(-(log pi(good)-log pi(bad)))`加入PPO actor loss，并在首次formal update前按梯度步长
中位数标定一次beta。它没有部署期额外head，但依赖训练期反事实future outcome和大量快照/身份检查。

保留的历史结果是：64/64 canonical-BC固定偏好实际得到13 target、5 control episode与75/20 pair，
U30为`67 collision / 1436 overtake`；online臂U30为`85/1466`，两者相对K10/K50都大量损失overtake，
固定实例关闭。更早的simulator-return-filtered U44为`49/1530`，仍是历史成功结果，但当前源码无法
继续训练或重建该方法。删除前最小测试通过K10/K50、走廊gate、runtime snapshot exact next-transition
和1 target/1 control loss；删除后测试只保留并通过K10/K50与走廊gate。

当前K10/K50只描述speed探索的时间保持周期：模拟步长`.01s`，K10让同一个speed标准残差保持10步
即`.1s`；K10/K50在走廊外仍为K10，前向走廊gate为true时保持50步即`.5s`，进入/离开gate边界重采样。
它不是另一套前向走廊规划器；steering仍每`.01s`独立采样。`collect_rollouts()`现在统一覆盖默认、K10
和K10/K50，gate读取集中在一个小函数；`front_corridor_speed_noise_hold_steps=0`即关闭gate效果。

环境仍不输出SB3 Monitor格式，因为训练直接使用`CentralScheduleSubprocVecEnv`且`info`没有Monitor的
`episode={r,l,t}`字段。训练需要的episode数据由`utils.log_ppo()`写入`episodes.jsonl`；原SB3 Monitor
buffer及其均值日志长期为空，2026-08-14已删除。episode return、step、outcome和仿真elapsed time仍保留。

清理后的验证包括：Python编译、55项reward测试、K10/K50与走廊gate测试、四种critic构造，以及真实
Austin F110 `2 env x 1,800 step` warm-up加一次formal update。该smoke只验证执行链和重构等价性，
不是新性能实验，也不能替代固定四图评测。

## 14. 汇报期技术范围

当前汇报只区分三种状态：

1. **已有模型与结论**：production、RW30、K10/K50、成功但含上一代U44数据闭环的simulator-return-
   filtered first-action preference，以及所有已关闭实例；
2. **未执行合同而非待办**：168 collision target + 225 safe-overtake control BC-native固定偏好
   曾完成命令设计，但dataset/run/eval均不存在，`run.sh`已移除；
3. **讨论历史而非待办**：Lattice-reference+BC、原`instrument_train`宇宙BC-native、每5个update刷新
   preference、action-conditioned短期Q/cost critic、group-robust PPO、reverse constrained PPO、
   selective reference retention及其他未实现变体。

从本节点开始不得新增第四类，不得把第2/3类写进CLI、配置、`run.sh`或新文档。若后续汇报需要比较，
只能引用既有实验事实，并明确“未实现/未训练”不等于“被科学否决”。
