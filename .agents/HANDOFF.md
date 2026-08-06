# End2Race 当前 HANDOFF

更新时间：2026-08-06（Asia/Singapore；四图碰撞身份与接触几何诊断完成，等待外部审查）

## 0. 文档职责和读取顺序

本文件是**接手当前仓库时的第一入口**，只保留会改变下一步行动的当前事实、核心合同、
实验最终判决和停止/重开规则。完整数据与机制推导已经迁入：

- `ANALYSIS.md`：完整实验设计、面板定义、配对结果、分层数据、机制判断和证据边界；
- `EXPERIMENTS.md`：历史实验工具与回归测试的实现逻辑和重建合同；
- `GUIDE.md`：用户要求的实验执行、命名和文件布局规范。

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

当前没有需要续跑的实验。已有 run 目录均视为完成、否决、被替代或封存，禁止在原目录续写。
中断后若需重跑，必须使用新目录；当前 checkpoint 不包含 optimizer、scenario queue、
environment 或 recurrent state，不能称为 exact resume。

同日通过`192.168.50.2:2222`只读核对远端：远端代码树没有新增改动，唯一未跟踪文件是旧的
regime审计Markdown；其全部结论已经由本地§9.6和`ANALYSIS.md` §23覆盖。没有把远端旧版
HANDOFF覆盖回本地，也不保留第二份根目录审计文档。

最新完成活动见§9.7及`ANALYSIS.md` §24。fresh诊断不是历史update replay：checkpoint不含
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

---

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

### 10.2 未完成但不是高优先

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

5. 复用cache前核对canonical BC哈希；身份不匹配则新cache或reclassify。
6. 解释旧run时先读它的 `run_config.json`，不要套当前defaults。
7. 检查metrics行、checkpoint、final actor和进程状态后再宣布run完成。
8. 新eval使用唯一actor alias，防止同名checkpoint trace覆盖。
9. 正式评测默认且固定使用CUDA/GPU；CUDA不可用时停止，不静默退回CPU，也不重复CPU对照。
10. 验证面板scenario数、0 errors、trace key、marker和finite数值。
11. 任何production变更必须同时更新本文件的§1、§2、§3.8和对应判决。

当前交接终点：

> 架构与实验接口已经覆盖reward、pool、采样和探索多个方向。当前不启动新训练。正式CUDA
> 四图BC验收已经确认前向走廊门控时间相关速度噪声U44为安全候选：`62/1478`，相对BC
> collision removed/created `94/27`（`p=7.14e-10`），overtake lost/gained `41/74`
> （`p=0.00269`）。旧ordinary异线高速重加权U30在新口径下计数为`73/1516`，相对BC也
> 显著双轴改善，但其历史package未记录CUDA device，故只列为待一次固定CUDA确认的高超车
> 候选；旧near400 `64/302`仍是明确副作用。碰撞身份与姿态复算进一步表明当前是失败迁移：
> U44/RW30分别新造`27/28`次，新增车辆碰撞绝大多数为相对yaw约`4.5°`的平行侧后接触。
> 粗regime最优拼接只有`58/1527`，而不可部署的逐episode事后上界为`25/1557`，所以更高目标
> 缺的是regime内部的状态条件选择，不是继续增加探索/采样剂量。用户明确禁止multi-map PPO；
> 训练只可使用Austin。Production部署别名暂时仍指向U30，当前没有未完成run，等待其他agent
> 和GPT Pro审查§9.9/`ANALYSIS.md` §28后再决定是否只做Austin离线可分性诊断。
