# End2Race PPO 实验 ANALYSIS

更新时间：2026-08-06（Asia/Singapore；fresh PPO梯度、role-mix与CUDA评测协议审计完成）

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
| §1 | 当前状态、最后活动、production 决策 | 2026-07-30，最优先 |
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

**2026-08-05最后活动是无训练诊断，不是新run。** Production U30与前向走廊门控时间相关
速度噪声U30的S/O/N cohort完成动作收益counterfactual、361D observation/1680D hidden
线性探针和共享actor分层梯度审计；没有产生新actor。最新结论是output-head相反更新强于
GRU冲突，详见§23。Advantage/credit子审计按预定条件没有触发，仍属于未运行问题。

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

接手时的合法选项：只剩 §18.5 中"仍未测试"的 hold 时长轴（K10/K25），而按同节第 8 条
它的先验很低；或按第 9 条的重开条件引入新的任务分布/新控制。

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
- **速度探索已收口**：历史完成5种模式；当前代码只保留逐步独立速度高斯噪声
  (`baseline`)、全局时间相关速度噪声(`temporal_global`)和前向走廊门控时间相关速度噪声
  (`corridor_temporal`)。
  `conditional_temporal`、`conditional_white`（连同其escalating门）与speed-std退火
  已删除但重建合同保留。按当时协议，所有实验均未通过
  Austin600 + near400联合验收，production保持`baseline`。当前验收口径已由§25.6覆盖；
  前向走廊门控时间相关速度噪声
  在三张held-out地图上把
  碰撞从`80 -> 46`（`p=1.8e-4`），所以front-corridor逻辑仍有研究价值、不能整体删除；
  但1.0m门宽失去收益，2.0m保持为YAML默认，1.5m未训练，不能写成已证伪。ordinary异线
  高速重加权改由YAML定义并默认关闭。详见§18。
- **Regime无训练审计把副作用定位到output head。** 严格counterfactual改善子集上，S-O/S-N
  output-head cosine为`-0.962/-0.959`，O-N为`+0.998`；GRU冲突更弱。动作收益线性探针
  没有建立hidden对observation的稳定优势。当前保持361D输入、reward和critic，详见§23。
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

机制判断：S需要的动作方向与O/N相反，而O与N彼此一致；在hidden不正交到足以隔离它们时，
共享最后输出映射收到近乎反向更新。GRU也有冲突，但严格子集上的幅度与稳定性明显弱于
output head。因此“共享参数干扰发生在哪里”的当前答案是：**主要在output head，尤其
steering/speed最后一行；GRU是次级，不是零。** 这不等于某个尚未训练的双head/gating方案
已经有效。

### 23.5 为什么C没有运行、停止与重开规则

现存eval NPZ没有reward、value、GAE、log-prob或完整20D privileged critic state；历史
checkpoint也没有rollout buffer/optimizer/environment recurrent state，所以不能恢复历史
advantage。本轮原定门控是：只有A显示regime可分但B不能解释干扰，或准备因符号错配重开
critic/reward，才付出新rollout成本运行C。

B已经给出强且经counterfactual筛选稳定的output-head冲突，A又没有建立hidden充分性的强结论；
因此C没有触发。正确表述是“当前没有依据先改critic”，不是“GAE已验证正确”。

停止规则：

1. 不因A的null/不稳定结果立即增加actor输入；线性不可解码不等于信息不存在；
2. 不把B写成历史PPO/Adam更新的精确回放；
3. 不重复exploration强度、reward、critic或generic observability sweep来解释已定位的head冲突；
4. 不用6条anchor不稳定场景训练probe或选择性重跑到预期outcome；
5. 合法下一步必须在保持输入/reward/critic的单变量设计中隔离S与O/N的输出映射或更新；
6. 只有head冲突被控制后仍失败，或新鲜同前缀paired rollout直接显示“counterfactual更好但
   advantage为负”，才重开C和critic/credit；
7. 本轮一次性脚本和分析产物已按用户要求删除；不得仅为复核本文数值重新生成analysis树，
   只有输入模型、面板、动作合同改变，或用户明确要求重新审计时才按`EXPERIMENTS.md` §8重建。

### 23.6 可独立于分析产物保留的核心记录

- 冻结模型：production U30与前向走廊门控时间相关速度噪声U30，身份见`HANDOFF.md` §2；
- 固定cohort：S/O/N=`54/69/59`，near400不与ordinary面板混合；
- counterfactual有效176/182；减速/production/steering-required标签=`61/37/20`，未解决91；
- 表征结论：hidden更平滑但没有稳定优于361D observation，不能据此改输入；
- 梯度结论：严格改善子集S-O/S-N head约`-0.96`，O-N约`+1.00`，冲突集中output head；
- C未运行；当前保持production、actor输入、reward与critic；下一合法轴是输出映射/更新隔离。

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
四图BC验收线，因此不启动训练期reference-policy regularization。

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

当前不需要任何新训练或新评测。只剩一个产品选择：若安全优先，将U44切换为production；
若当前最高超车数优先，保持Production U30。在用户明确选择前，不改默认模型路径、训练参数或
production登记。
