# End2Race PPO 实验指南

本文件只规定长期有效的实验方法、运行边界和产物组织方式。
当前最佳模型、reward/pool 配置、正在运行的任务与核心判决记录在
`.agents/HANDOFF.md`；完整实验设计、数据、分析和边界记录在
`.agents/ANALYSIS.md`；代码风格约定记录在 `.agents/STYLE.md`。不要复制到这里。

## 1. 目标与边界

- 目标是通过 PPO 训练得到能够独立完成控制的 actor。
- 正式结果不得依赖评估或部署时的 action 后处理、安全 shield、未来碰撞信息、
  oracle 动作或 actor 不可获得的特权触发条件。
- oracle、特权量和离线重放可以用于诊断可达性与失败机制，但诊断结果不能冒充
  actor 本身的性能。
- 除非用户明确重新授权，保持 actor 网络、输入接口和部署方式不变。
- 除非用户明确重新授权，采用单阶段 PPO；不把 imitation、蒸馏或二次微调混入
  PPO 单变量实验。

## 2. 实验设计

- 正式实验从 canonical BC 初始化，完整运行预先规定的 PPO updates。
- 当前 pipeline 不实现完整 resume。正式延长训练必须从 canonical BC 重新完整运行到
  新的目标 update；不得把既有 PPO actor 作为初始化、再跑若干 update，冒充原轨迹的
  连续 U31、U32 等。
- 固定使用 seed 42；不以多 seed 扫描替代机制分析。
- 每次只改变一个预注册变量。对照组与实验组必须保持初始化模型、训练预算、
  数据协议、评估 panel 和代码版本一致。
- 先做成本低、机制明确的验证，再决定是否训练；不要进行没有明确假说的参数 sweep。
- 训练开始前写清主指标、守门指标、预期机制、通过条件和停止条件。
- 不因已经投入大量实验而继续无效方向；结果不支持假说时应明确停止。

### 2.1 训练前的离线筛选

能在已保存 trace 上离线重放判定的候选，必须先离线筛选，再决定是否消耗训练预算。
离线筛选的产物至少包括：

- 预注册的数值准入检查（每条是明确的阈值判定，不是事后叙述）；
- 目标 cohort 的覆盖率与首次触发提前量；
- 安全对照 cohort 的误触率，以及**正对照**：本该被保留的成功行为是否会被压掉；
- 一个显式布尔结论，例如 `ready_for_training_ab`。

正对照缺失时不得判定通过。只有目标覆盖高、没有验证过"不误伤成功行为"的候选，
不具备训练准入资格。

离线筛选可以证伪，不能证实：即使全部检查通过，也只说明选择性可接受，
不代表训练后策略会改善。

### 2.2 续训实验的混杂声明

从既有 checkpoint 热启动时，同期的"自然延续"运行不是 control。必须逐条列出与
treatment 不同的因素，至少包括 critic 是否重新 warm-up、optimizer/RNG 路径、
学习率、以及 collision cache 的分类 actor 是否与训练起点一致。

actor 热启动只加载 actor 权重，critic、optimizer、update计数、RNG、scenario调度和
未结束episode状态均不会恢复，因此它是一条新实验轨迹。其 checkpoint 不得并入原
`EXPERIMENT_ID/update<N>/` 序列。

任何显式放宽既有不变量的开关（例如允许 cache 与 actor 身份不匹配）都必须记入
`run_config.json`，并在 ANALYSIS 中完整记录；若影响production判断或下一步行动，
同时在 HANDOFF 中写明，不能只留在命令历史里。

### 2.3 跨实验线共享的组件

一个门控、特征或几何组件被多条实验线复用时，必须在两条线的记录中互相标注。
同一组件在不同用途下的判据可能相反——作为 reward 触发器嫌太宽的门，
作为 exploration 触发器可能嫌太窄。改动共享组件时必须说明改的是哪一用途，
并检查另一侧受到的影响；两侧的历史数字不可跨用。

## 3. 运行与 checkpoint

- 所有训练、评估和诊断命令都在 conda 环境 `end2race` 中运行。
- 长时间任务统一使用 tmux，避免终端断开导致任务中止。tmux 会话名使用
  `EXPERIMENT_ID` 或能够直接对应实验的简短名称。
- 优先复用现有训练、评估和诊断入口，通过已有参数完成实验；只有现有入口无法表达
  已获授权的实验变量时，才对原脚本做最小修改。
- 不为一次运行新建 bash 文件或日志文件。训练命令统一放在仓库根的 `run.sh`；
  不创建 `run_<experiment>.sh`、`nohup.out`、`train.log` 或 `eval.log`。
  **`run.sh` 当前不存在**（2026-07-30 随历史实验命令一并删除），所以下一项实验需要
  新建它，而不是去找一个空队列桩。
- `run.sh` 只保留当前实验一条可直接复制运行的显式命令，不使用自动模型选择器、
  动态数组、shell 函数或隐藏的 checkpoint 排名逻辑。
- 实验完成、明确中断或放弃后，先把状态和核心结果写入 HANDOFF/ANALYSIS，再删除
  `run.sh` 中该命令；若已无待运行实验，直接删除 `run.sh`。不得用长期注释块充当实验记录。
- 不新建 `tests/` 或其他平行测试树。确需保留的新组件回归测试统一放在 `scripts/`，
  按 `scripts/test_<component>.py` 与被测组件对应，并保持最小覆盖。
- 实验工具只用于验证确实新增的组件，不用于包装已有训练/评估命令，也不为普通参数
  实验新增一次性脚本。新实验优先复用已有入口；确需新工具时才创建或复用 `scripts/`，
  并遵循 handoff skill 的最简创建规则。
- 不使用 callback 或脚本自动选取 “best checkpoint”。既有的 checkpoint 排名脚本
  （`select_critic_lr_candidate.py`）按此规则视为停用，不作为新实验入口。
- 多个 checkpoint 用于判断训练后期是否稳定，不用于从大量单点中挑选偶然最低值。
- 同一训练配置共用 `post-trained/<EXPERIMENT_ID>/`，模型 checkpoint 按
  `update<N>/` 分层保存；目录名中的 update 不补零，例如 `update29`、`update30`。
  不得覆盖既有 checkpoint，也不得覆盖 `pretrained/end2race.pth`。
- 若先完成 U30、后验证 U45 是否收敛，U45 必须从 canonical BC 重新完整运行45个
  updates。共享区间 U1--U30 的 actor 与 critic 必须逐 update、逐 tensor 完全一致，
  才能把 U31--U45 合并进同一个 `EXPERIMENT_ID`；共享区间只保留一份。任何一个
  checkpoint 不一致都表示两条不同轨迹，必须使用不同的实验ID，禁止合并。
- 当前不增加 resume checkpoint。计划跑到 U45 时应直接以 `num_updates=45` 启动，
  长任务依靠 tmux 防止终端断开；中断后按完整 fresh-start 合同重新运行。
- 长任务通过 tmux pane、进程状态、checkpoint、metrics 和原生结果文件持续监控；
  不以额外日志文件替代这些证据。必须区分 active、interrupted、training-only
  和 complete。

## 4. Evaluation panel

`PANEL_ID` 是一组固定评测场景和协议的短名称，不是新的模型或分析系统。
例如一张地图上的固定多车 600-episode 协议可以命名为 `austin600`。

一个 panel 的精确定义至少包括：

- 地图与评估模式；
- startpoint 集合或生成规则；
- ego/opponent raceline；
- opponent speed、interval、noise、laps 和 episode 时长；
- 预期 episode 数和场景身份集合。

这些细节记录在 `eval_manifest.json`，不全部编码进目录名。评估协议发生实质变化时，
必须使用新的 `PANEL_ID`，不能继续沿用旧名称。

### 4.1 Panel 的角色

每个 panel 在一次实验中承担哪种角色，必须在预注册时写清，事后不得更换：

| 角色 | 作用 | 判定权 |
|---|---|---|
| 主验收 panel | 目标任务分布上的整体表现 | 决定接受/否决 |
| 守门 KPI panel | 保护不能被牺牲的次要能力 | 单独否决权 |
| 诊断/特化 panel | 已按失败筛选出的困难子集，用于机制归因 | **无验收权** |
| 泛化 panel | 训练中未见的地图或起点 | 支持性证据 |

诊断/特化 panel 上的改善是特化信号，不是通过证据。已按失败筛选的子集天然会对
针对该失败的干预给出正向结果；只有在主验收和守门 panel 同时不劣化时，
该改善才有意义。用特化 panel 单独接受一个改动是本项目已反复出现的错误。

守门 panel 显著变差时，即使主验收 panel 改善也不判通过；两者必须同时报告。

**当前项目覆盖（2026-08-06用户最新决定）**：后续正式验收默认必须运行四张赛道：
`Austin`、`Hockenheim`、`MoscowRaceway`和`Nuerburgring`各600 episode。`Austin600`与
三张跨地图合计`crossmap1800`都具有正式验收权，同时必须报告三张跨地图各自结果；具体实验的
最低验收线固定为canonical BC：每张地图的ego collision不得高于BC、overtake不得低于BC，
opp-wall单列，并报告配对身份变化；更严格的改进目标和checkpoint band必须在运行前预注册。near400和hard子集只保留
机制/特化诊断意义，没有独立验收权。三张跨地图已经参与过走廊探索设计，因此可以作为当前
开发验收集，但不得再称为从未参与设计的最终泛化留出。

### 4.2 评估要求

- 正式评测固定使用 CUDA/GPU；不要为同一模型重复运行 CPU 对照，也不能将 CPU 与 CUDA
  结果配对。若 CUDA 不可用则停止并修复运行环境，不静默退回 CPU。
- 对照组和实验组使用完全相同的 panel。
- 保存每个 episode 的数值 trace。
- 结论以固定场景的配对身份变化为主，同时报告总量。
- **每个比较必须给出消除数、新造数和配对显著性检验**，不得只报净变化。
  总量相同不代表失败集合相同；净变化为零可能掩盖大量身份churn。
- 报告当前结果时完整给出四张正式赛道：Austin600、三张跨地图各600及crossmap1800合计；
  其他诊断panel只有在实验预注册包含它时才报告，且不得用诊断结果替代正式四图验收。
- 多个 checkpoint 用于判断后期是否稳定：报告区间或极差，不要只引用单点最优值。
- 区分 ego-opponent、ego-wall 和 opponent-wall collision。
- 使用尾部/甩尾等派生标签时必须写明采用哪个定义；不同 artifact 的口径可能不同，
  不得把两种口径合并成一个未命名的计数。
- 训练 metrics 只用于解释学习过程，不能代替确定性 evaluation。
- 正式 panel 必须检查预期 episode 数、唯一场景数、error 数、结果与 trace key
  一致性、数值有限性、collision marker 和 terminal row 合同。

### 4.3 多臂对照的 trace 命名

评估器按 actor 文件名 stem 组织 trace 输出。不同实验臂如果使用同名 checkpoint
（例如都叫 `actor_u0030.pth`），trace 文件会互相覆盖，而 outcome JSON 不受影响——
即结果看起来正常，trace 却已经串臂。

多臂对照必须为每个臂指定唯一的 actor alias，使各臂 trace 目录互不相交。

## 5. 实验命名

`EXPERIMENT_ID` 只描述实验主题和被改变的设置：

```text
<topic>_<setting>
```

名称应简短且能区分实验，例如：

```text
baseline_repro
global_temporal_speed_noise_hold50steps
actorlr3
```

不要在 ID 中重复写 seed、日期、初始化模型或 update 数；这些属于 `run_config.json`。
完全相同的重复实验使用明确后缀，例如 `exploration_k50_repeat2`，不得覆盖第一次运行。

Collision cache不是实验ID，名称必须直接表达分类actor、地图、池类型和场景数：

```text
post-trained/collision-cache/<ACTOR_ID>_<map>_<pool_type>_<scenario_count>/
```

例如`pretrained_end2race_austin_collision_pool_479`。Cache迁移只更新当前代码和cache自身
身份；历史run的`run_config.json`继续保留当时真实使用的路径，不得回写成新路径。

## 6. 产物路径

新实验采用以下目标结构。已确认属于production或同一训练轨迹的 checkpoint 按本节
归档；其余历史actor/eval结果不为追求目录整齐而批量改名。清理历史分析输出前，
仅将跨实验复用的固定panel输入集中到下述`post-trained/panels/`：

```text
post-trained/<EXPERIMENT_ID>/
├── run_config.json
├── trajectory_manifest.json
├── metrics.jsonl
├── episodes.jsonl
├── collision_scenarios.json
├── ordinary_scenarios.json
├── collision_cache_info.json
├── critic_warmup.pt
├── update1/
│   ├── actor.pth
│   └── critic.pt
├── ...
├── update30/
│   ├── actor.pth
│   └── critic.pt
└── update45/
    ├── actor.pth
    └── critic.pt

eval_results/<EXPERIMENT_ID>/update30/<MAP_NAME>/
├── multiagents/
│   ├── results_multi.json
│   ├── eval_manifest.json
│   └── traces/
├── singleagent/
│   ├── results_single.json
│   └── traces/
├── noise10/
├── noise20/
└── noise30/

post-trained/panels/<PANEL_SET_ID>/
├── README.md
├── *_scenarios.json
└── selection metadata
```

例如 production PPO 的稳定路径为：

```text
post-trained/ppo_privilege_gru_clip020/update30/actor.pth
post-trained/ppo_privilege_gru_clip020/update45/actor.pth

eval_results/ppo_privilege_gru_clip020/update30/Austin/
eval_results/ppo_privilege_gru_clip020/update45/Austin/
```

`EXPERIMENT_ID` 只表达训练配置；`update<N>` 表达同一条训练轨迹上的 checkpoint；
`MAP_NAME` 表达评估地图。noise 是地图目录下的协议子目录，不编码进实验ID或
checkpoint名。原始训练器内部使用的 `actor_u0030.pth` 等文件名可以保留作为运行
产物，但长期引用和评估统一使用上述 canonical 路径。canonical `actor.pth` 和
`critic.pt` 优先与原始训练权重建立硬链接：不复制大权重，清理任一文件名也不会使
另一入口断链。不要把长期canonical入口做成依赖待清理目录的软链接。原始运行目录
继续保存完整 metrics、episodes、run_config 和场景池记录，直到对应信息按清理合同
完成迁移。

同配置的短run和长run通过共享update逐tensor一致性检查后，以长run作为完整训练记录
来源；将其metrics、episodes、场景池、run_config和warmup critic迁入canonical根目录，
并把metrics中的checkpoint引用改为canonical `update<N>`路径。`trajectory_manifest.json`
记录来源run、共享区间验证和fresh-start延长方式。迁移验证通过后，原始短run和长run
只是重复容器，可以清理。

最小评估产物的职责：

- `results_multi.json`：保存总体结果以及每个 episode 的 outcome/metrics。
- `eval_manifest.json`：保存模型/checkpoint标识、panel定义、实际命令、代码来源，
  以及 episode/trace 完整性验证结果。
- `traces/`：保存与 episode key 一一对应的数值 NPZ。
- `post-trained/panels/`：只保存跨实验复用的固定ScenarioSpec、fixed pool和必要的
  selection provenance；不得写actor评估结果、日志或临时分析。

不默认生成：

- `trace_and_scenario_results.json`：与 `results_multi.json` 的 episode 数据重复。
- 独立的 `validation_receipt.json`：验证结果合并到 `eval_manifest.json`。
- `comparison_vs_<CONTROL_ID>.csv`：配对比较默认从两份 `results_multi.json`
  直接重算；确有长期保存需要时，才在对应 eval 目录增加一个简洁 comparison 文件。
- 额外的独立分析结果树。

必要的验证信息和可选比较结果都留在对应的 eval 路径中；运行过程不另建日志文件。

补充说明：

- 本节只约束**新实验**。历史分析产物不要求重命名；其中的核心内容已经迁入
  ANALYSIS，并把决策摘要写入HANDOFF，后续清理原文件不影响理解。
- 完整结论写进 ANALYSIS，不再单独建分析目录。配对比较从 `results_multi.json`
  重算；ANALYSIS记录比较口径、分母、增删身份、p值、机制和边界，HANDOFF只保留
  baseline、核心数字、判决和停止/重开规则。两者都不记录复算脚本或证据路径。
- 需要跨多个 panel 的机制诊断（例如按 regime 拆解失败迁移）时，允许在**主验收
  panel 的 eval 目录**下增加一个简洁的诊断文件，并在 ANALYSIS 中给出完整结论、
  在 HANDOFF 中给出行动摘要；
  不为此新建顶层分析目录。
- 不在HANDOFF或ANALYSIS维护JSON/CSV/report路径清单或文件摘要；这些运行产物允许在
  核心结果完成迁移后由用户另行清理。
- **模型 checkpoint 例外**：actor checkpoint 会长期保留且身份必须可核对，其 SHA-256
  集中登记在 HANDOFF 的模型身份登记节（canonical 初始化模型、production 模型、
  各臂被评 checkpoint），其他章节引用该节而不重复写摘要。同时登记等价集（同一份权重
  存在于多个 run 目录）和命名陷阱（未完成 run 与其 `_rerun`、同 run 不同 update）。
  不对分析/评估产物计算摘要——文件会被清理，摘要即成死重量。

## 7. 可复现性与记录

`run_config.json` 至少记录：

- `EXPERIMENT_ID`、seed、初始化actor路径及其 SHA-256；
- 实际训练参数、updates、代码 commit/工作树状态；
- 使用的数据、cache 或场景来源，以及 cache 的分类 actor 是否与训练起点一致；
- 任何显式放宽默认不变量的开关及其后果；
- 完整训练命令。

判定一次训练"完成"需要同时满足：`run_config.json` 声明的 update 区间、
`metrics.jsonl` 的 warmup/formal 行数与数值有限性、预期的最后 checkpoint 与
最终 actor 产物存在、没有进程仍在写该目录。行数本身是必要条件不是充分条件；
resume、延长和纯分析目录各有自己的合同。

`eval_manifest.json` 至少记录：

- `PANEL_ID` 和完整评估参数；
- actor checkpoint路径及其 SHA-256；
- 实际评估命令；
- 预期/实际 episode 与 trace 数、唯一 key 数、error 数和验证结论。

若评估中断或 panel 不完整，不得在原结果上宣称完成；保留错误证据，并在明确的新目录
中重新运行或恢复。

### 7.1 实验目录清理合同

当用户明确决定删除历史分析产物、实验工具或回归测试源码时，清理前必须完成：

1. 在 `HANDOFF.md` 固化production、最终判决、停止/重开规则；
2. 在 `ANALYSIS.md` 固化控制变量、分母、完整headline、removed/created、p值、关键分层、
   机制边界，以及仍会影响下一步的scenario身份或动作参数；
3. 在 `EXPERIMENTS.md` 固化脚本/test的CLI、关键常量、输入输出schema、算法不变量和断言；
4. 明确列出清理后失去的能力，不得使用“无损迁移”描述Markdown摘要；
5. 核对四份文档没有把中断run、旧口径、离线候选或诊断panel写成production证据。

清理后允许声称“核心判决与实现合同已保留”，但不得声称：

- 原始trace、逐scenario配对和任意新分层仍可恢复；
- 回归测试源码可逐行还原；
- Markdown中的三位小数或聚合表可替代原始浮点数据；
- 删除后的历史评估仍通过了当前代码的重新执行验证。

若未来需要发表级复核，应重新运行固定panel并生成新的机器可读结果，而不是从Markdown
反向构造原始artifact。

## 8. GUIDE、HANDOFF、ANALYSIS 与 EXPERIMENTS 的边界

`GUIDE.md` 只回答“实验以后应该怎样设计、运行和保存”。

`.agents/HANDOFF.md` 负责记录：

- 当前 production/default 配置；
- 当前基线与候选模型；
- reward、pool、exploration 等功能的production状态；
- 已完成实验的核心数字、判决和停止/重开规则；
- 正在运行、暂停或中断的任务。

`.agents/ANALYSIS.md` 负责记录：

- 实验问题、控制变量和配置；
- panel/cohort定义、样本量和比较口径；
- 完整结果、配对身份、p值、checkpoint band与分层；
- 机制证据、失败原因、未知项和证据边界。

`.agents/EXPERIMENTS.md` 负责记录实验工具的实现逻辑、适用范围与工具入口。

实验完成、终止、被否决或 production 默认发生变化时，按照
`.claude/skills/handoff-log/SKILL.md` 更新 ANALYSIS 和 HANDOFF；实现/测试变化由
EXPERIMENTS维护。不在 GUIDE 中追加实验流水账。
