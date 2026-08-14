# End2Race 实验工具与回归测试重建规范

本文件的唯一用途：**在历史实验工具和回归测试源码被删除后，让 coding agent
据此按需重新生成功能等价的实现。** 它不是使用说明，也不记录实验结论——判决与停止规则在
`.agents/HANDOFF.md`，完整实验数据与机制分析在 `.agents/ANALYSIS.md`，
实验方法规范在 `.agents/GUIDE.md`。

**当前代码边界（reward清理后）**：Post-pass生产模块、`train_ppo.py`中的Post-pass
开关以及L12的risk纵向override均已删除；production reward固定为四项并从YAML读取
纵向`0.6m`。本文中Post-pass/L12的接口、测试和参数只用于解释或按需重建历史实验，
不得当作当前测试要求，也不得自动恢复到production。

**2026-07-30训练分布与优化接口清理边界**：完成本文保真记录后，活动训练代码已经移除
ordinary起点`50→150`扩展、外部fixed collision pool入口、boundary-aware
hard-neighbor 805/比例采样、collision-cache actor-path mismatch逃生开关、speed-std
退火、旧版conditional-temporal臂和target-KL early stop。默认479 collision cache、
固定50个ordinary起点和collision/ordinary双角色队列必须保留；训练入口只读取固定cache，
不再包含cache miss时的自动分类、写盘或CPU回退。2026-08-14又移除ordinary异线高速重加权、
三路ordinary queue及其YAML配置；历史实验语义保存在§0.5.5，不得把它当作当前入口。

下面的§0.5是这些退役接口的重建合同；§2.8、§2.9、§2.11和§1.5仍保留历史测试的
逐项断言。hard-neighbor与outcome-aware的源码模块也已删除：它们没有当前调用方，
完整805池和20%臂已否决，10%臂主动终止，outcome-aware从未完成独立训练A/B。
文中出现旧模块名或flag只表示“历史/重建接口”，不表示当前源码或CLI仍支持它们。

记录时间：2026-07-30（清理前保真审计修订）。对应83个文件：18个回归侧Python文件
（16个`test_*.py` + 2个历史工具），47个实验工具Python文件 + 19个shell runner，
合计约26,700行Python。

---

## 0. 使用本文件前必读

### 0.0 按方法族定位实现

详细章节保留历史编号和依赖顺序，避免破坏脚本、测试及ANALYSIS中的引用；实现检索统一从下表
进入。同一组件若服务多个方法族，主归类按其改变的训练变量，其他用途在具体章节交叉说明。

| 方法族 | 实现与重建入口 |
|---|---|
| Reward与优化目标 | §1.8--§1.11、§2.1--§2.6、§3.3、§5.3、§20、§23--§24 |
| 训练pool与采样 | §0.5、§2.8--§2.11、§3.1、§4 |
| 探索与curriculum | §1.10、§2.7、§2.16、§15--§17、§21 |
| BC/参考策略正则 | §11--§12、§18、§22 |
| 反事实动作学习 | §8--§9、§13--§14、§25--§26、§32--§33 |
| 表征与辅助监督 | §14、§19 |
| Checkpoint组合 | §10 |
| 共享验证、记录与评测合同 | §0.1--§0.4、§1--§7 |

当前用户已冻结新tech，不新增实现。若未来用户明确解除冻结，新增实现仍必须使用描述性机制名称并
追加到对应方法族；Round/Gate编号只能作为历史定位信息，不能替代方法名称。

**2026-08-10实际清理边界**：历史专题合同已经固化后，`scripts/`中的23个已完成或关闭实验工具
已删除。下文出现这些文件名时表示“可按本节重建的历史接口”，不表示文件仍在工作树。当前只保留：

```text
scripts/screen_reward_candidate.py
scripts/test_screen_reward_candidate.py
scripts/test_first_action_preference.py
scripts/build_bc_first_action_preference.py
```

前两个承担reward候选合规筛查及其回归；后两个验证合并到`ppo/rollout.py`的固定first-action
preference数据/loss、canonical-BC固定数据构建、K10/K50探索与默认走廊门合同。普通四图
600评测统一使用根目录`evaluate.sh`，不再保存重复ScenarioSpec或保留第二个
显式panel评测入口。First-action preference的固定训练数据、训练期PPO模块和正式checkpoint仍保留，
但其一次性dataset builder与formal eval analyzer已经删除，重建合同见§25。

同日PPO主线继续删除了训练入口参数保护层、collision cache自动构建、scheduler恢复接口、
缺失rollout buffer时的静默policy路径，以及每轮102,400 transition的full-buffer ratio/dry-gradient
审计旁路。当前production固定使用CUDA和已有collision cache；worker异常传播、环境关闭、checkpoint
保存、输入cache身份核对及实际训练likelihood仍属于主线合同，不按“保护性代码”删除。

2026-08-10后续`ppo/`清理又从活动代码移除了§21和§24所述prefix-reset / prefix-local
joint-temporal实现：当前CLI、VecEnv、policy、rollout buffer和训练器均不再接受prefix panel或
joint-temporal专用状态。下文保留的是历史重建合同，不是待运行入口。同步删除的纯记录路径包括
episode reward分项/risk/最小净空累计、full-buffer value/critic/exploration/prefix统计、重复
update/timestep/config字段、preference逐minibatch梯度分解和全量校准样本键；reward、GAE、P20
输入、动作likelihood、optimizer、checkpoint与正式outcome记录未改变。当前活动训练侧保留production
PPO、数值hold步数的时序动作探索和固定dataset first-action preference；100 Hz序列loss抽样与
same-state branched PPO已经退役。

同日进一步把PPO内部参数从`env.py`、`policy.py`、`reward.py`、`rollout.py`和`scenarios.py`
集中到`ppo_config.yaml`；各模块沿用`latticeplanner.utils.load_config()`直接加载，不另建
`ppo/config.py`。CLI训练参数继续只在`train_ppo.py`。模块内仅保留
结构/schema常量。旧`FIRST_ACTION_PREFERENCE_DATASET_ID`及其名字相等校验已删除，因为schema、
gate verdict、manifest/gate SHA与sequence SHA已经完整约束数据合同。`run_config.json`继续在
`ppo_config`字段记录全部实际配置，不再重复写一套大写常量别名。

**2026-08-13汇报期代码收口**：在线collision-triggered临时preference、移除走廊lateral-offset门和
全局steering K10的
正式结果已经在`ANALYSIS.md` §59固化，活动CLI、环境旁路、collector、临时loss/metrics及专用回归
随之删除；policy不再保留temporal steering state。下文§31--§32.2只保留历史机制，不是当前实现合同。固定BC-native builder仍保留；三个lead和
12个single-step residual的YAML键统一为`first_action_preference_lead_steps`与
`first_action_preference_action_residuals`。用户同时冻结新tech，本文不得据此自动重建历史接口。

### 0.1 保真度分级（重要，不要误期待）

| 分级 | 覆盖对象 | 能做到什么 |
|---|---|---|
| **A 精确契约** | 16个`test_*.py`；`postpass_reward_calculation.py` | 断言、fixture、数值、容差逐条记录。重写后应当**通过同样的判定**，行为等价 |
| **B 功能规范** | 其余 45 个 `*.py` | 目的、CLI、输入、输出 schema、算法步骤、不变量。重写后**功能等价但不逐行相同**；输出字段名可能需要按下游消费者对齐 |
| **C 调用契约** | 19 个 shell runner | 调用顺序、参数、目录命名、前置校验 |

B 级是刻意的取舍，理由有三条，接手者应当接受而不是试图硬凑 A 级：

1. 这些分析脚本的**结论已经固化在 ANALYSIS §16-§21**，重建它们不是为了重新得到那些
   数字（数字已经有了），而是为了将来做**新的同类分析**时不必从零设计；
2. `.agents/GUIDE.md` §6 已规定新实验不再创建独立分析结果树，所以这些脚本的
   产物布局本身已经不是未来的目标形态；
3. 23,000 行一次性分析 plumbing 的逐行规范，其体积会超过代码本身，且大部分内容
   （CSV 写盘、路径拼接、argparse 样板）对重建毫无信息量。

**如果将来需要某个具体脚本达到 A 级**，正确做法是在删除前对那一个文件单独补写详细
规范，而不是指望本文件。第 5 节列出了最值得这样处理的 6 个文件。

用户已决定后续删除历史实验工具与回归测试源码。因此本文件在清理后是**重建规格**，不是源码
备份：A表示重建后应达到相同行为合同，B表示可恢复主要算法和schema，C只恢复运行编排。
原注释、异常文本、局部变量、浮点运算顺序和逐行执行细节不会被Markdown无损保留。

### 0.2 依赖图与重建顺序（关键）

部分回归测试**依赖实验工具函数**，不是相反。相关导入关系如下：

```text
test_postpass.py                        -> postpass_reward_calculation
test_compare_postpass_formulas.py       -> compare_postpass_formulas.replay_trace
test_postpass_episode_replay.py         -> validate_postpass_reward (3 个函数)
test_compare_risk_potential_variants.py -> compare_risk_potential_variants.py（importlib 按路径加载）
test_select_critic_lr_candidate.py      -> select_critic_lr_candidate (CANDIDATES, LATE_UPDATES, select)
test_build_heldout_hard_instrument.py   -> build_heldout_hard_instrument (7 个函数)
test_analyze_l12_heldout_hard_instrument.py -> analyze_l12_heldout_hard_instrument
```

曾经还有一个测试依赖独立影子模块 `shadow_contract.py` 提供的
`PostpassState` / `RewardConfig` / `postpass_reward_step`。该 oracle 已于
2026-07-30 迁入 **`scripts/screen_reward_candidate.py`**；独立目录可在完成本节所列
验证后删除，迁移边界与明确退役的能力见 §5.3。

因此重建顺序是：

```text
1. postpass_reward_calculation.py        （A 级，纯函数，无项目依赖除 ppo.geometry）
2. 独立 oracle                            （已在 scripts/screen_reward_candidate.py；
                                           若连它也丢失，按 §1.8/§1.9 + §5.3 重写）
3. 其余 5 个被测试 import 的实验工具        （B 级，但要保住被 import 的符号名和签名）
4. 回归测试集合                            （A 级）
5. 剩余实验工具与 shell runner              （按需）
```

**当前仓库里仍存在、不需要重建的三个脚本**（历史测试数量不作当前保证）：

```text
scripts/screen_reward_candidate.py        reward 候选的合规门禁 + 学习信号量化 + 独立 oracle
scripts/test_screen_reward_candidate.py   对应回归测试
scripts/test_first_action_preference.py       first-action preference数据/loss + online collision preference机械合同测试
scripts/build_bc_first_action_preference.py   canonical-BC固定偏好数据构建
```

**被测试 import 的符号名和函数签名属于契约的一部分**，重建时不得改名，否则测试无法
按本文件重写。

实验工具内部还有一层真实import DAG；必须按下列顺序恢复，不能把相同helper复制成多个
悄然漂移的版本：

```text
postpass_reward_calculation
  -> validate_postpass_reward
     -> analyze_postpass_geometry_cases

analyze_baseline_reward_seed42
  -> analyze_l12_deterministic_transfer
     -> analyze_l12_heldout_hard_instrument
     -> analyze_interval15_difficult_experiment
     -> analyze_speedstd050_experiment
  -> analyze_structured_speed_exploration
     -> analyze_temporal_late_checkpoint_stability
  -> analyze_speedstd_anneal_experiment

analyze_crossmap_bc_u30
  -> analyze_crossmap_b_t
  -> analyze_ctv2_checkpoint_eval
  -> analyze_ctv2_corridor_temporal

diagnose_crossmap_temporal_corridor
  -> validate_ctv2_corridor_gate
  -> diagnose_ctv2_preflight_panels
  -> diagnose_ctv2_ordinary_checkpoint_curve
  -> analyze_ctv2_loss_mechanism

run_u30_oracle_reachability
  -> evaluate_shared_oracle_library
     -> probe_shared_oracle_ranking
     -> scan_u30_braking_landscape
```

关键跨文件符号包括：

- `EXPECTED_TRACE_FIELDS`、`collision_features`、`relative_progress`、
  `load_panel`、`paired_stats`；
- `exact_mcnemar`、`recursively_finite`、`validate_trace`；
- `BatchProjector`、`gate_masks`、`temporal_active`、`evaluate_trace`；
- `SAFE_OUTCOMES`、`SEGMENTS`、`vector_bounds`、`flatten_result`、
  `result_score`、`scenario_record`、`load_collision_labels`；
- `DEFAULT_PANELS`、`first_crossing`、`load_reference_tail_labels`；
- `exact_mcnemar_p`、`validate_eval`、`ARM_DIRS`、`MAPS`、`load_arm`；
- `load_json`、`initialize_library_worker`、`make_library_task`、
  `run_library_worker`、`scenario_metadata`。

这些名字属于可复用接口。重建时若要改名，必须同时更新全部消费者与§2测试，不得只让
单个脚本“能跑”。

### 0.3 实验工具不依赖回归测试，但历史回归集合混有两个非测试文件

历史回归集合混有两个可执行工具（历史遗留，`unittest discover` 不会收集它们）：

- `probe_hard_neighbors.py` —— standalone hard-neighbor 采样探针
- `build_outcome_aware_cache.py` —— outcome-aware pool 构建驱动
- 以及一份笔记 `OUTCOME_AWARE_MERGE_NOTES.md`

规范见 §4。按 GUIDE §3，重建时应作为实验工具保存，不要再混入回归测试集合。

### 0.4 通用运行约束（所有文件适用）

- 解释器：`/home/haowei/miniconda3/envs/end2race/bin/python`；从 repo root 运行。
- 所有脚本以 `PROJECT_ROOT = Path(__file__).resolve().parents[1]` 定位仓库根。
- 分析脚本一律**只读**训练/评估产物，只向指定 output dir 写盘。
- 多进程一律 `forkserver`；worker 内把 OpenMP/MKL/OpenBLAS/Torch 线程数限为 1；
  CUDA 初始化必须在创建 subprocess 之后。
- 写盘用「临时文件 + 原子 rename」；多个脚本有 `atomic_write_json` / `atomic_write_csv`。
- 断言优先 fail closed：schema、数量、唯一性、有限性任一不符即抛异常，不静默跳过。

### 0.5 退役训练分布接口的功能重建合同

本节记录功能等价所需的最小实现，不要求恢复原注释、异常文本、局部变量名或JSON产品
哈希。四项均为训练期场景分布工具，不改变reward、actor输入、网络结构或确定性eval。

#### 0.5.1 Ordinary 50→150

历史入口为`--ordinary_startpoint_count {50,150}`，默认50。调用链：

```text
train_ppo.py
  -> ordinary_scenarios(map_name, startpoint_count)
  -> ordinary_startpoints(map_name, startpoint_count)
  -> CentralScheduleSubprocVecEnv
```

重建时必须保留以下语义：

1. `ordinary_startpoints(map_name, 50)`调用基础
   `generate_separated_startpoints(map_name, 50, minimum_distance)`，得到固定baseline；
2. 请求数小于50必须拒绝；等于50直接返回baseline；
3. 扩展时读取去掉重复闭合端点的ego raceline，并以当前
   `get_circular_startpoints(map_name, ego_raceline, 50, 0)`为eval起点；
4. 只允许与任一eval起点欧氏距离至少`ordinary_startpoint_min_distance=1.0m`的waypoint；
5. 先原样保留baseline 50个，再迭代选择“到已选起点的最小环形progress距离最大”的
   候选；并列时取allowed数组中最早者，直到达到请求数；
6. `ordinary_scenarios`按startpoint→opponent raceline→speed的稳定顺序展开，因此
   150起点的前600条ScenarioSpec必须逐条等于50起点的完整600条；
7. 训练记录里的effective config和ordinary场景清单必须写实际请求值。

测试合同见§2.8：固定50个Austin index必须逐项相等；150个起点唯一、前缀兼容，
1800场景前600条与baseline一致，所有扩展起点到当前eval起点距离
`>=1.0-1e-12m`。实验结果见ANALYSIS的ordinary150专题：Austin碰撞`14→35`、
near`28→55`、hard`54→38`。这只否决“150起点+固定rollout预算”的整体配置；
多样性增加与单场景重复覆盖降至约三分之一不可分离，不能写成多样性普遍无用。

#### 0.5.2 外部fixed collision pool

历史公共入口：

```python
FIXED_COLLISION_POOL_SCHEMA = 1
ALLOWED_SOURCE_LABELS = frozenset({"ego_collision", "near_miss"})
load_fixed_collision_pool(
    path: str | Path, *, map_name: str
) -> tuple[tuple[ScenarioSpec, ...], dict[str, Any]]
```

CLI`--fixed_collision_pool_file PATH`与reclassification、actor-mismatch、
hard-neighbor及hard fraction互斥；省略时必须逐字节走默认collision cache路径。
加载器要求顶层字段严格等于
`schema_version/purpose/source/selection/sampling/entries`：

- `selection`严格含`split="train"`、正整数`interval_idx`、正数
  `near_miss_clearance_m`、固定
  `include_outcomes=["ego_collision","overtake_or_follow_near_miss"]`；
- `source`严格含root和四个非空身份字符串；
- 每个entry严格含`source_label/source_outcome/min_obb_clearance_m/scenario`；
- ScenarioSpec必须匹配运行map、`pool=="collision"`和selection interval；
- collision标签必须对应`source_outcome=="ego_collision"`；
- near-miss只允许overtake/follow且clearance在`[0, threshold]`；
- scenario_id和完整物理键都必须唯一；
- sampling必须精确等于`uniform_cycle_over_combined_pool`、实际数量和标签计数。

成功返回有序ScenarioSpec元组和info；info至少包含mode、原文件、purpose、source、
selection、sampling、collision_count。历史实现还记录输入文件SHA，但重建不再要求
把产品哈希写入HANDOFF。§2.11记录四项回归：合法两条池、错误interval、越界near-miss、
重复物理场景。唯一正式A/B是interval-15 collision+near-miss池：U5/U10/U15
Austin分别`14→14 / 16→20 / 15→16`，均无改善；这否决该池，不证明通用loader无效。

#### 0.5.3 Boundary-aware hard-neighbor 805与比例采样

历史CLI：

```text
--hard_neighbors
--hard_neighbor_cache_dir <独立schema-2 cache>
--hard_neighbor_fraction <0,1内有理数，可省略>
```

核心公开结构和函数：

```python
BoundaryCandidatePlan(...)
BoundaryDiscovery(pair_records, candidates, generated_candidate_count)
discover_boundary_candidates(
    candidates, outcomes, *,
    interval_indices=(8,10,12,15),
    speed_scales=(0.45,...,0.85),
    max_candidates_per_family=24,
) -> BoundaryDiscovery
materialize_boundary_candidates(
    plans, base_candidates
) -> tuple[ScenarioSpec, ...]
load_hard_neighbor_cache(...) -> tuple[tuple[ScenarioSpec, ...], dict]
resolve_training_collision_scenarios(
    args, base_candidates, start_method
) -> tuple[tuple[ScenarioSpec, ...], dict]
```

边界生成算法：

1. 将speed scale以`SPEED_FIXED_POINT_SCALE=1000`转为整数，拒绝不能精确表示的值；
2. family键固定为map、ego raceline、startpoint ordinal/ego index、opponent raceline、
   horizon、timestep、integrator；
3. 在每个完整interval×speed lattice中只检查一轴相邻边；
4. 仅`ego_collision↔other`构成边界，任一端为invalid则排除；
5. speed边生成严格中点；interval边只在端点之间存在内部整数时生成；
6. 同一物理候选合并所有source pair id；按稳定排序和family上限确定性选取；
7. 物化时按opponent raceline重新映射waypoint，设置`pool="hard_neighbor"`，
   生成唯一id，并要求reset pose和初始速度全部finite；
8. 使用同一冻结BC重放候选，只把确认ego_collision合并到base479。

schema-2 cache必须包含classification config、base outcomes、boundary pairs、
boundary outcomes、最终collision scenarios、summary、build metadata和manifest。
身份绑定冻结actor、base cache语义文件、地图/raceline与planner资产、generator参数、
horizon/timestep/integrator；任何差异fail closed。构建先写同父目录临时目录，完整验证后
原子发布，禁止覆盖已有正式cache。正式构建结果是1042个边界pair、1183个唯一候选、
326 collision/857 other/0 invalid，最终479+326=805。

采样有两种历史语义：

- 只传`--hard_neighbors`：805条合并后用单一无放回queue，boundary自然占326/805；
- 再传fraction：base479与boundary326使用独立无放回queue，按确定性周期精确插入
  boundary reset；0.20周期前10个位置为`{2,7}`，0.10为`{5}`。

Scheduler state必须保存两队列、source cursor、fraction numerator/denominator，
恢复后后续序列逐项一致；默认479与ordinary队列的相对顺序不得改变。完整测试合同见§2.9。
结果：完整805的U25+平均碰撞`13.0→21.4`、U45`12→17`；20%在U35/U40/U45
为`14→27 / 11→20 / 12→19`并同时丢超车。10%训练到45 updates，但只有U1–U20
完整eval，因此从未判定有效或无效；本次删除是用户主动终止该未定案方向，不得改写成
“所有hard-neighbor比例均被证伪”。

#### 0.5.4 Collision-cache actor-path mismatch逃生开关

历史入口`--allow_collision_cache_actor_mismatch`只允许忽略
`classification_config.pretrained_model_path`，不允许忽略任何其他身份字段。算法是：

1. 分别复制cached/current config并弹出actor path；
2. 对剩余字段做排序JSON精确比较；
3. actor相同则返回false；actor不同且开关关闭则报错；
4. actor不同、开关打开且其他字段相同则返回true；
5. map、候选网格、horizon等任何其他字段不同仍必须报错。

它与`--reclassify_collisions`、hard-neighbor和fixed pool互斥，且要求四个base cache文件
完整存在。§1.5历史测试覆盖“只差actor路径允许/拒绝”和“map同时不同仍拒绝”。
该功能从未声称改善策略，只为早期Post-pass快速验证复用旧cache；删除后严格合同恢复为
完整classification config一致，否则使用新的空cache目录重新分类。

#### 0.5.5 退役项：ordinary异线高速重加权

2026-08-14按用户决定删除`ordinary_offline_fast_fraction`、
`ordinary_offline_fast_min_speed_scale`和三路ordinary调度代码。当前`ScenarioScheduler`只保留
collision/ordinary两个均匀无放回队列；历史checkpoint不受影响。

历史值`ordinary_offline_fast_fraction=0.6`把ordinary分为same-line、off-line-fast
（异线且speed scale≥0.7）和off-line-slow，三组份额从`1/3, 1/3, 1/3`改为
`5/15, 9/15, 1/15`。三组各自使用独立无放回随机队列，same-line沿用ordinary seed，
off-line-fast和off-line-slow分别使用`SeedSequence([seed, 0x4F464653])`与
`SeedSequence([seed, 0x4F464C57])`；15槽周期按组内均匀位置排布。该方法只改变采样权重，
不改变场景集合、reward或optimizer。它在跨地图得到54–57 collision/1146–1155 overtake，
对production 80/1142双轴占优，但near400碰撞恶化到63–77，故历史协议否决。若要复现，
只能按本段在独立实验分支重建，不得自动恢复到production。

#### 0.5.6 退役target-KL early stop

历史入口`--target_kl`把有限正数传给PPO actor phase。每个有效recurrent minibatch计算：

```text
log_ratio = new_log_prob - old_log_prob
approx_kl = mean(exp(log_ratio) - 1 - log_ratio)
if approx_kl > 1.5 * target_kl:
    跳过当前minibatch及本update剩余actor optimizer steps
    critic phase仍继续
```

这不是rollback：导致下一minibatch首次越界的前一个optimizer step已经保留。历史
`0.02/0.04`均频繁早停且没有优于target-KL关闭的同配置对照；`0.04`在U5--U20的
Austin旧面板平均碰撞为51.5。production winner始终使用`None`，因此当前代码删除CLI、
early-stop分支及专用telemetry；常规`approx_kl_mean`仍保留作为训练健康指标。

#### 0.5.7 退役speed-std退火和旧conditional-temporal臂

speed-std退火的历史实现按formal update把初值线性插值到终值，达到指定update后固定
终值；`.40→.15/10u`实验得到Austin44、near68、hard32，否决。旧
`conditional_temporal`使用escalating-required-deceleration门，门内std 0.25并保持
50步，实测门曝光仅约0.51%，Austin15/342、near40/295、hard46/17，未通过验收。

当前仅保留三个模式：逐步独立速度高斯噪声（`baseline`）、全局时间相关速度噪声
（`temporal_global`）和前向走廊门控时间相关速度噪声（`corridor_temporal`）。
2026-07-30 移除了条件门控高方差逐步独立速度噪声（`conditional_white`）及其专用的
`EscalatingRequiredDecelerationGate`（303行）、`FollowingDangerGateConfig`
和`_causal_rate`——C的门曝光仅约0.15%，属于没有机制证据的臂，却占用该文件近一半代码。
**仅剩的门是前向走廊门控时间相关速度噪声使用的front-corridor门**；
`_FrenetProjector`、`_wrap_angle`、`_Projection`是历史两类门共用的基础件，随门保留。
全局时间相关速度噪声不使用任何门。

前向走廊门控时间相关速度噪声的前距由YAML键
`front_corridor_gate_maximum_gap_m`定义，默认2.0m。1.0m训练臂失去
跨地图同线收益；1.5m没有训练A/B，不能写成已证伪，但不再为门宽单独保留CLI。

条件门控高方差逐步独立速度噪声的重建语义仍完整保留在本节与`ANALYSIS.md`：门参数
（corridor entry/exit 0.20/0.25m、
safe gap 0.50m、required relative deceleration 1.25m/s²、persistence 0.20s、
front gap 2.0m、closing time 1.5s）与`conditional_white`的std 0.50、无时间保持。
其U30 checkpoint仍在`post-trained/ppo_conditional_white_speed_noise_0p50/`，
**但当前代码已无法复现该训练**；若要重跑必须先按本节重建模式与门。

---

## 1. 共享原语（A 级，所有脚本共用，必须逐字一致）

这一节是全仓库分析结果的**定义层**。ANALYSIS里的对应数字由这些定义产生；重建时
任何偏差都会让新结果与历史结果不可比。

### 1.1 Outcome 分类与碰撞口径

评估器给每个 episode 一个 typed outcome，取值集合：

```text
"overtake" | "follow" | "ego-opp" | "ego-wall" | "opp-wall"
```

判定规则：

```python
COLLISION_OUTCOMES = {"ego-opp", "ego-wall"}          # ego collision
def is_collision(outcome):  return outcome in COLLISION_OUTCOMES
def is_overtake(outcome):   return outcome == "overtake"
```

**`opp-wall` 不计入 ego collision。** `results_multi.json` 的 `final` 块里
`collision_count` 包含 `opp-wall`，所以 ego collision 必须自己算：

```python
ego = final["ego_opp_collision_count"] + final["ego_wall_collision_count"]
```

历史数字用的就是这个口径（例：production U30 Austin600 `collision_count=14`，
`ego_opp=11`、`ego_wall=2`、`opp_wall=1` → ego collision = 13）。

### 1.2 同线 / 异线（same-line / off-line）

```python
EGO_RACELINE = "raceline1"                  # 来自 ppo/ppo_config.yaml
SAME_LINE  = lambda spec: spec.opp_raceline == EGO_RACELINE
OFF_LINE   = lambda spec: spec.opp_raceline != EGO_RACELINE
```

trace/episode key 场景 ID 形如 `ol{line}_e{ego_idx}_o{opp_idx}_s{speed}`，
`SAME_LINE_PREFIX = "ol1_"`（因为 ego raceline 是 raceline1）。多个脚本靠
episode key 前缀而不是 spec 字段来分组，两种方式必须给出同一结果。

**历史off-line fast定义**（原`ScenarioScheduler.is_offline_fast`已删除）：

```text
off-line（opp_raceline != raceline1）AND opp_speedscale >= 0.7
```

在 shipped 的 600 条 ordinary panel 上恰好命中 200 条。

### 1.3 甩尾（fishtail）与碰撞模式

```text
mode_counts:
  post_overtake_rear : ego 已完成超车后，车尾侧接触
  rear_end_opp       : ego 从后方撞上对手
  wall               : ego 撞墙

fishtail := post_overtake_rear AND (pre-impact 运动学 slip >= 8.0 deg
                                    OR 两车 heading 差 >= 20.0 deg)
slip 检测窗口：碰撞前 60 步，stride 5
```

派生标签有多个口径（`merge_tail`、`structural_tail`），**报告时必须写明用哪一个**，
不得合并成一个未命名的计数。

### 1.4 精确 McNemar（配对显著性）

所有配对比较统一用**双侧精确二项检验**，不是卡方近似：

```python
def mcnemar_exact(favorable: int, adverse: int) -> float:
    """favorable = 被 treatment 修好的数量，adverse = 新造的数量。"""
    n = favorable + adverse
    if n == 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(min(favorable, adverse), n, 0.5).pvalue)
```

碰撞类比较报 `removed / created`；超车类报 `lost / gained`。**只报净变化是禁止的**。

配套的功效工具（`analyze_l12_heldout_hard_instrument.exact_mcnemar_detectable_net`）：
给定 discordant 总数，返回能达到 `p < 0.05` 的**最小**净差；返回 `None` 表示该
discordant 数下任何净差都无法显著。已验证的阈值（测试直接钉住）：

```text
exact_mcnemar_detectable_net(16)  == 10
exact_mcnemar_detectable_net(41)  == 15
exact_mcnemar_detectable_net(129) == 25
```

其最小性由 `test_analyze_l12_heldout_hard_instrument.py` 反向验证：
`favorable-adverse == net` 且 `binomtest(adverse, discordant, 0.5).pvalue < 0.05`，
同时 `adverse+1` 时不再显著。

### 1.5 Quantile 约定

所有脚本自带 `quantile()` 助手，语义统一为：**在有限值上做线性插值分位数，
空输入返回 `None`（不是 0.0、不是 NaN）**。P50/P90/P95 一律按此计算。
重建时不要直接换成 `np.quantile` 而不处理空输入与非有限值。

### 1.6 Trace（NPZ）schema 与验证契约

每个 episode 存一个 NPZ，用 `allow_pickle=False` 打开。字段（`EXPECTED_TRACE_FIELDS`）
至少包含：

```text
time_s              (T,)      单调递增
ego_pose            (T, 3)    [x, y, heading]
opp_pose            (T, 3)
collisions          (T, 2)    bool，[ego, opponent]
action_applied      (T,)      bool
terminal_post_step  (T,)      bool
（另有 scan / speed / steering / desired_speed 等，共 16 个数组）
```

**终帧契约（必须验证）**：

```text
terminal_post_step 恰好只有最后一行为 True
action_applied     恰好只有最后一行为 False
最后一行的 action 数组是占位零值，不得当作实际动作解读
普通 8 秒 episode 通常 802 行 = 801 applied-action frame + 1 terminal frame
```

`validate_trace()` 的标准检查项：所有数组首维一致；全部数值有限；
terminal/action_applied 契约成立；trace 记录的碰撞状态与 episode outcome 一致；
trace 文件名与 `episode_key` 一致。

**版本边界**：0721 目录的旧 trace 没有 terminal 行，schema 不同；两种行语义不得混用。

### 1.7 Panel 定义

| PANEL_ID | 构成 | 角色（见 GUIDE §4.1） |
|---|---|---|
| `austin600` | Austin，50 个 eval 起点 × 3 opponent raceline × 4 speed scale，interval 15 | **当前唯一主验收** |
| `near400` | 近距面板，400 场景 | 历史守门；当前只作诊断 |
| `hard73` | 73 条已筛困难 interval-15 场景 | 诊断/特化，**无验收权** |
| `crossmap 1800` | Hockenheim / MoscowRaceway / Nuerburgring 各 600 | 历史泛化诊断；当前无验收权 |
| `ordinary600` | 50 个**训练**起点 × 3 raceline × 4 speed，interval 15 | 训练分布 |
| `collision candidates 10800` | 100 训练起点 × 3 raceline × 4 interval(8/10/12/15) × 9 speed(0.45..0.85) | cache 构建 |

跨地图 panel 构造（`build_crossmap600_panels.py`）：每张地图 50 个圆周起点、
offset 0、3 条 opponent raceline、speed scale `(0.5, 0.6, 0.7, 0.8)`、interval 15。
**训练起点与 eval 起点必须保持至少 1.0 m 间隔**，构造后要实测最小成对距离并写入产物。

### 1.8 Post-pass 几何（A 级，`postpass_reward_calculation.py`）

纯函数模块，唯一外部依赖 `ppo.geometry.rectangle_clearance`。整车尺寸
`length=0.58`、`width=0.31`。以下是完整规范，可直接据此重写：

```python
@dataclass(frozen=True)
class EgoInducedRearClosing:
    counterfactual_previous_clearance_m: float
    current_clearance_m: float
    closing_m: float
```

**输入校验**：pose 必须 reshape 成 `(3,)` 且全有限，否则
`ValueError("<name> must be one finite [x, y, heading] pose")`；
length/width 必须有限且 > 0，否则
`ValueError("Vehicle length and width must be finite and positive")`。

```python
oriented_rectangle_vertices(pose, length, width) -> (4,2)
    # 顺序固定：rear-left, rear-right, front-right, front-left
    longitudinal = (cos θ, sin θ);  lateral = (-sin θ, cos θ)
    rear  = center - 0.5*length*longitudinal
    front = center + 0.5*length*longitudinal
    返回 (rear+0.5w*lat, rear-0.5w*lat, front-0.5w*lat, front+0.5w*lat)

rear_half_vertices(pose, length, width)
    # 车尾一半作为独立 OBB：对平移、yaw、后角扫掠都敏感
    中心沿纵轴后移 0.25*length，然后用 (0.5*length, width) 建 OBB

signed_rear_longitudinal_gap(ego, opp, length, width) -> float
    # 在 ego 车体纵轴上投影：ego 车尾最小投影 − 对手最大投影
    ego_axis = (cos θ_ego, sin θ_ego)
    return min(ego_vertices @ ego_axis) - max(opp_vertices @ ego_axis)
    # 正 = ego 整条尾边在对手前投影之前；0 = 边缘对齐；负 = 纵向仍重叠

rear_half_clearance(ego, opp, length, width) -> float
    return rectangle_clearance(rear_half_vertices(ego), oriented_rectangle_vertices(opp))

ego_induced_rear_closing(prev_ego, cur_ego, cur_opp, length, width)
    # 关键：两次 clearance 都用【当前】对手 pose，所以对手运动不会误算到 ego 头上
    prev_c = rear_half_clearance(prev_ego, cur_opp)
    cur_c  = rear_half_clearance(cur_ego,  cur_opp)
    closing = max(0.0, prev_c - cur_c)          # 只取缩小，不取张开

rear_gap_unsafe_fraction(signed_gap, safe_gap) -> [0,1]
    # safe_gap <= 0 或非有限 -> ValueError
    return clip((safe_gap - signed_gap) / safe_gap, 0, 1)

postpass_penalty_basis(closing_m, signed_gap, safe_gap, *, active) -> float
    # closing_m 非有限或 < 0 -> ValueError
    if not active: return 0.0
    u = rear_gap_unsafe_fraction(signed_gap, safe_gap)
    return u * u * closing_m                     # 注意是 u²，与 q 幂次无关

bounded_postpass_reward(basis, weight, step_cap, used, episode_cap) -> float <= 0
    # 任一输入非有限或 < 0 -> ValueError
    # used > episode_cap + 1e-12 -> ValueError
    used = min(used, episode_cap)
    remaining = max(0.0, episode_cap - used)
    return -min(weight*basis, step_cap, remaining)
```

### 1.9 历史 Post-pass 门控固定参数（已从production删除）

```text
pass margin                        0.05 m
safe rear gap                      0.60 m
activation rear-half clearance     0.20 m
closing deadband                   0.10 m/s
maximum ego-induced closing time   0.75 s
weight                             0.25 /m
per-step cap                       0.005
per-episode cap                    0.05
proximity_power                    2（CLI 可选 1）
terminal_refund                    False
```

metadata 键名必须用 `maximum_ego_induced_closing_time_s`，**禁止出现 `maximum_ttc_s`**
（这不是完整两车物理 TTC，只提取 ego 引起的尾部接近）。

### 1.10 训练期探索门（`ppo/exploration.py`）

这些门只服务训练期探索臂，不进入当前reward。当前保留的前向走廊门控时间相关速度噪声使用
`FrontCorridorGate`：

```text
maximum_front_gap_m                          2.0   （当前YAML front_corridor_gate_maximum_gap_m；历史CLI已删除）
maximum_abs_opponent_lateral_d_m             0.25
require_positive_lateral_overlap             True
TEMPORAL_RESAMPLE_STEPS                      50    （0.5 s hold）
```

已退役的条件门控高方差逐步独立速度噪声使用`EscalatingRequiredDecelerationGate`：

```text
corridor_entry_abs_d_m / exit                0.20 / 0.25   （hysteresis）
safe_gap_m                                   0.50
required_relative_deceleration_mps2          1.25
required_deceleration_persistence_s          0.20
entry_closing_time_s / exit                  1.5 / 2.0
```

`EscalatingRequiredDecelerationGate`的派生量定义（因果，不许用未来样本）：

```text
front_gap        = Frenet 纵向中心距 − 两车各自在自身切线方向的 OBB 纵向支撑
closing_speed    = ego 减对手 unwrapped raceline progress 的因果后向窗口导数（正 = 接近）
closing_time     = front_gap / max(closing_speed, 0)   —— 局部线性 gap 耗尽时间，不是物理 TTC
required_relative_deceleration = closing_speed² / (2 * (front_gap − safe_gap))，仅作选择器
required_deceleration_growth   = 上式的因果后向窗口变化率
```

当前`SPEED_EXPLORATION_MODES`三种：`baseline`（默认）、
`temporal_global`、`corridor_temporal`。历史`conditional_temporal`已否决并删除，
本节只保留其重建语义。

### 1.11 Risk potential 变体（`compare_risk_potential_variants.py`）

生产公式（`ppo.reward.anisotropic_risk_potential`）：

```text
d = min( hypot(longitudinal/L, lateral/0.2), wall/0.2 )
Φ = -0.05 * max(0, 1-d)^2         # L 默认 0.6；L12 实验令 L=1.2
```

常量：`GAMMA=0.999`、`LATERAL_SAFE_M=0.2`、`WALL_SAFE_M=0.2`、
`MAXIMUM_MAGNITUDE=0.05`、`POWER=2`。`VARIANTS` 是 `RiskVariant` dataclass 列表，
必须包含：`VARIANTS[0]` = 生产 baseline（L=0.6，`min` 组合）、L12、L20，以及
`variant_id="SUM20"` 的 bounded-sum 变体。

Shaping 与望远镜（`shaping_rewards`）：

```text
r_t = gamma * Φ(s_{t+1}) - Φ(s_t)
真 terminal:  Φ(next) = 0
truncation :  保留最后一个 physical potential，由 PPO bootstrap
discounted_return  = -Φ[0] + gamma^(T-1) * Φ[-1]   （truncation 情形）
discounted_telescope_residual 必须 < 1e-15
```

`potential_without_wall` 用于检查 `min()` 是否屏蔽了墙项（生产 `min` 不屏蔽；
bounded-sum 在车辆项饱和时会屏蔽——这是它被否决的原因之一）。

---

## 2. 回归测试重建规范（A级，16个测试文件）

统一约定：`unittest`，从 repo root 对重建后的回归目录运行 discovery；目录名称由
重建任务决定，本文件不保留被删除位置。
两个文件用 pytest 风格裸函数（`test_analyze_l12_heldout_hard_instrument.py`、
`test_build_heldout_hard_instrument.py`，后者用 `tmp_path` fixture）。

清理前共有**97个`test_*`定义**：91个由`unittest discover`收集，另外6个只由pytest
收集。两套都必须重建并执行；不能用“unittest 91项通过”替代pytest 6项。2026-07-30
清理前复核这6项为**6 passed**。

### 2.1 `test_postpass.py`（157 行，8 个 test）

`class PostPassGeometryTests`。从 `postpass_reward_calculation` import
7 个符号。模块常量 `VEHICLE_LENGTH_M=0.58`、`VEHICLE_WIDTH_M=0.31`。

| test | 断言 |
|---|---|
| `test_axis_aligned_signed_rear_gap_has_physical_surface_semantics` | opp=(0,0,0), ego=(1.20,0,0) → gap == `1.20-0.58`，`places=12`；ego=(0.30,0.50,0) → gap < 0 |
| `test_vertices_are_rear_to_front_and_have_expected_dimensions` | pose=(1,2,0)：前 2 个顶点 x == `1-0.29`，后 2 个 == `1+0.29`；`ptp(y) == width` |
| `test_forward_motion_opens_rear_clearance_and_has_zero_closing` | opp=(0,0,0)、prev=(0.90,0.50,0)、cur=(0.95,0.50,0) → `current > counterfactual_previous` 且 `closing_m == 0.0` |
| `test_yaw_can_sweep_rear_half_toward_opponent_without_center_motion` | opp=(0.20,0,0)、prev=(0.70,0.40,0)、cur 仅 heading 改为 `+0.45` → `closing_m > 0`；heading `-0.45` → `closing_m == 0.0` |
| `test_opponent_motion_is_not_attributed_when_ego_pose_is_unchanged` | prev == cur == (0.90,0.55,0.10)，opp=(0.20,0.10,0) → `closing_m == 0.0` |
| `test_rear_half_clearance_is_zero_at_overlap` | ego=(0.30,0,0)、opp=(0,0,0) → clearance == 0.0 |
| `test_penalty_basis_is_sparse_and_quadratically_gated` | `unsafe_fraction(0.60,0.60)==0`、`(-0.10,0.60)==1`、`(0.30,0.60)≈0.5`；`basis(0.01,0.0,0.60,active=False)==0`；`basis(0.01,0.30,0.60,active=True)≈0.0025` |
| `test_bounded_reward_respects_step_and_episode_caps` | `(0.005,1.0,0.02,0.10,0.20)→-0.005`；`(0.050,…)→-0.02`（step cap）；`used=0.195→-0.005`（剩余额度）；`used=0.20→0.0` |

### 2.2 `test_postpass_reward_integration.py`（496 行，3 个 class）

**依赖一份独立实现的 oracle**：`PostpassState`、`RewardConfig`、`postpass_reward_step(...)`。
它原在独立影子模块 `shadow_contract.py`，现已迁入
**`scripts/screen_reward_candidate.py`**（见 §5.3），直接从那里 import 即可。
若连它也丢失，必须按 §1.8/§1.9 独立重写（**独立实现是这个测试的全部价值——
不能让 oracle 直接调用生产代码，否则交叉校验退化为恒等式**）。oracle 返回字段名与
生产略有差异，映射关系：`closing_speed_mps↔ego_induced_closing_speed_mps`、
`ttc_s↔ego_induced_closing_time_s`（生产用 `None` 表示无穷，oracle 用 `inf`）。

测试内 helper：

```python
class LinearProjector:      track_length=100.0; project(pos)= pos[0] % 100.0
class ConstantMapClearance: rectangle_clearance(_v) = 1.0
raw_observation(ego_x, opp_x) = {"poses_x":(ego_x,opp_x), "poses_y":zeros(2), "poses_theta":zeros(2)}
production_reward(enabled, power=2) = PPOTransitionReward(
    "Austin","raceline1", projector=LinearProjector(), gamma=0.999,
    vehicle_length=0.58, vehicle_width=0.31, map_clearance=ConstantMapClearance(),
    risk_longitudinal_clearance_m=0.60, risk_lateral_clearance_m=0.20,
    risk_wall_clearance_m=0.20, risk_potential_maximum=0.05,
    postpass_penalty_enabled=enabled, postpass_proximity_power=power,
    transition_dt_s=0.01)
```

`class FixedPostpassContractTests`（setUp: prev_ego=(0,0,0)、cur_ego=(0.4,0,0)、cur_opp=(0.3,0,0)）:

- `test_fixed_gate_matches_isolated_oracle` —— `reset(-0.30)` 后
  `step(prev_rel=-0.30, cur_rel=0.10, …, opponent_collision_latched=False)`；
  断言 `triggered` 为真，且 `entered`/`cleared` 与 oracle 相同；8 个数值字段
  （reward、signed_rear_gap_m、rear_half_clearance_m、ego_induced_closing_m、
  closing_speed、closing_time、penalty_basis_m、episode_penalty_used）
  `places=12`。
- `test_randomized_fixed_contract_matches_isolated_oracle` ——
  `rng = np.random.default_rng(20260725)`，**256 组**随机几何：
  `prev_ego ~ U((-2,-2,-π),(2,2,π))`、
  `cur_ego = prev + U((-0.05,-0.05,-0.10),(0.05,0.05,0.10))`、
  `cur_opp = cur_ego + U((-0.70,-0.70,-0.30),(0.70,0.70,0.30))`、
  `opponent_collision_latched = (rng.integers(0,5)==0)`。
  每组新建实例并 `reset(-0.30)`。断言 `phase_active`、`triggered` 相等，6 个数值字段
  `places=11`；closing_time 为 `None` 时断言 oracle 的 `ttc_s` 为 `inf`。
- `test_opponent_collision_suppresses_penalty` —— `latched=True` →
  `entered` 真、`phase_active` 假、`triggered` 假、`reward == 0.0`。
- `test_episode_cap_and_reset` —— 连续 40 步（首步 prev_rel=-0.30，其余 0.10）后
  `penalty_used == FIXED_POSTPASS_CONFIG.maximum_episode_penalty`（`places=12`）；
  `reset(-0.30)` 后 `entered`/`cleared` 假、`penalty_used == 0.0`。
- `test_metadata_uses_physical_closing_time_name` ——
  `fixed_postpass_config_metadata()` 含 `maximum_ego_induced_closing_time_s`、
  不含 `maximum_ttc_s`、`proximity_power == 2`、`terminal_refund` 为假。
- `test_linear_proximity_changes_only_penalty_magnitude` —— 固定几何
  `cur_ego=(0,0,-0.62846831)`、`prev = cur + 0.1*((0.02876983,0.05587781,-0.58513743)-cur)`、
  `opp=(-0.57454773,-0.29809255,0.61511948)`。两个实例 power=2 / power=1：
  `proximity_fraction` 相等（`places=15`）；
  `linear.basis * linear.proximity_fraction == squared.basis`（`places=15`）；
  `linear.reward < squared.reward`。

`class ProductionRewardIntegrationTests`（`_one_step(enabled, terminated=False)`：
prev=raw_observation(0.0,0.3)、cur=raw_observation(0.4,0.3)，`reset` 后 `step`）:

- `test_default_off_preserves_four_component_reward` —— off 臂
  `reward_postpass == 0.0`、`postpass_enabled` 假、
  `reward_total == progress+relative+collision+risk`（`places=15`）；on/off 两臂
  四个分量逐一相等（`places=15`）；on 臂 `reward_postpass < 0` 且
  `total == off_total + postpass`。
- `test_terminal_flag_does_not_refund_postpass_penalty` ——
  terminated 与否 `reward_postpass` 与 `postpass_episode_penalty_used` 完全相同，
  但 `reward_risk` **必须不同**（terminal 令 Φ(next)=0）。

`class PostpassExperimentInterfaceTests`:

- `test_cli_switch_is_explicit_and_off_by_default` —— `patch("sys.argv", …)` 四次：
  裸命令 → `postpass_penalty` 假、`allow_collision_cache_actor_mismatch` 假；
  `--postpass_penalty` → 真且 `proximity_power == 2`；
  加 `--postpass_proximity_power 1` → 1；`--allow_collision_cache_actor_mismatch` → 真。
- `test_collision_cache_override_ignores_only_actor_path` ——
  `validate_collision_cache_identity(cached, current, candidate_count=10_800, allow_pretrained_model_path_mismatch=…)`。
  cached/current 只差 `pretrained_model_path` 时：False → `RuntimeError`，True → 返回真值；
  再改 `map_name` 为 `"Spielberg"` 时即使 allow=True 也必须 `RuntimeError`。
- `test_episode_metrics_include_postpass_selectivity` —— 造一条含 20 个
  `episode_*` 字段的 record（`env_role="collision"`、`episode_outcome="ego_collision"`），
  调 `End2RaceRecurrentPPO._episode_metrics([record])`，断言
  `postpass_trigger_episode_count==1`、`postpass_trigger_episode_rate==1.0`、
  `mean_episode_reward_postpass==-0.02`、
  `mean_episode_postpass_first_trigger_lead_s==0.3`、
  `ego_collision_postpass_trigger_episode_rate==1.0`、
  `overtake_postpass_trigger_episode_rate is None`。

### 2.3 `test_postpass_episode_replay.py`（87 行，5 个 test）

`@unittest.skipUnless(TRACE_ROOT.is_dir(), "Austin600 terminal traces are unavailable")`
—— 重建时令 `TRACE_ROOT = eval_results/pretrained_end2race/Austin/multiagents/traces`。
**这 5 个就是
ANALYSIS §16.5 记的「5 项 skip」**：trace fixture不在预期路径时整类跳过，不是失败。

从 `validate_postpass_reward` import `TrackProjector`、
`replay_episode_geometry`、`setting_episode_result`、`validate_trace`。
`setUpClass` 建 `TrackProjector(f1tenth_racetracks/Austin/raceline1.csv)`。
helper `replay(episode_key, collision)`：`validate_trace(TRACE_ROOT/f"{key}.npz",
expected_collision=collision)` → `replay_episode_geometry` →
`setting_episode_result(geometry, trace["time_s"], pass_margin_m=0.05,
safe_rear_gap_m=0.60, closing_deadband_mps=0.10, clear_mode="latched")`。

| test | episode key | collision | 断言 |
|---|---|---|---|
| `test_primary_tail_collision_has_preterminal_signal` | `ol0_e1283_o1279_s0.7` | True | `pass_detected` 真、`preterminal_triggered` 真、`preterminal_trigger_steps > 1`、`first_trigger_lead_to_terminal_s > 0.10` |
| `test_successful_overtake_has_small_but_nonzero_wait_signal` | `ol0_e0_o15_s0.5` | False | `pass_detected` 真、`preterminal_triggered` 真、`0 < basis_sum_m < 0.02`、`-0.02 < proposed_reward_sum < 0` |
| `test_ordinary_follow_has_no_postpass_phase_or_penalty` | `ol0_e0_o15_s0.8` | False | `pass_detected` 假、`triggered` 假、`basis_sum_m == 0.0`、`proposed_reward_sum == 0.0` |
| `test_prepass_collision_is_not_misclassified_as_postpass` | `ol0_e727_o739_s0.5` | True | `pass_detected` 假、`triggered` 假、`basis_sum_m == 0.0` |
| `test_opponent_collision_before_pass_disables_treatment` | `ol0_e1368_o1370_s0.8` | False | `pass_detected` 真、`active_steps == 0`、`triggered` 假、`basis_sum_m == 0.0` |

### 2.4 `test_compare_postpass_formulas.py`（67 行，1 个 test）

从 `compare_postpass_formulas` import `replay_trace`。本地 `LinearProjector`
同 §2.2。造一条 2 帧 trace：

```python
cur_ego  = (0.0, 0.0, -0.62846831)
prev_ego = cur_ego + 0.1*((0.02876983, 0.05587781, -0.58513743) - cur_ego)
opp      = (-0.57454773, -0.29809255, 0.61511948)
trace = {"time_s": (0.0, 0.01), "ego_pose": (prev_ego, cur_ego),
         "opp_pose": (opp, opp), "collisions": zeros((2,2),bool),
         "action_applied": (True, False), "terminal_post_step": (False, True)}
```

唯一测试名为`test_q_power_changes_only_magnitude`。`replay_trace(trace, LinearProjector())`
返回按 formula 分组的 row 列表，
formula 名固定 `no_q` / `q_linear` / `q_squared`。断言：三式 `trigger_steps` 都为 1；
`q_squared.applied_penalty < q_linear < no_q`；且**幅度关系精确成立**（`places=12`）：

```text
no_q.applied_penalty     * q == q_linear.applied_penalty
q_linear.applied_penalty * q == q_squared.applied_penalty
其中 q = q_squared row 的 "trigger_q_mean"
```

### 2.5 `test_compare_risk_potential_variants.py`（137 行，5 个 test）

**按路径加载脚本**（不是包 import）：`importlib.util.spec_from_file_location`
→ `module_from_spec` → 注册进 `sys.modules` → `exec_module`。重建时保留这一形式，
因为脚本文件名带下划线且不保证在包路径上。

| test | 断言 |
|---|---|
| `test_baseline_matches_production_formula` | `rng=default_rng(17)`，longitudinal~U(0,2)、lateral~U(0,0.8)、wall~U(0,0.8) 各 64 个；`MODULE.potential_components(..., MODULE.VARIANTS[0])["potential"]` 与逐点 `ppo.reward.anisotropic_risk_potential(..., 0.6, 0.2, 0.2, 0.05)` 全等（`atol=1e-15, rtol=0`） |
| `test_bounded_sum_has_explicit_total_cap` | `SUM20` 变体，输入 `lon/lat/wall = [10,0,10,0]/[10,0,10,0]/[10,10,0,0]` → potential 恰为 `[0, -0.05, -0.05, -0.05]`；且全体 `<= 0` 且 `>= -0.05` |
| `test_discounted_telescope_uses_discounted_not_plain_sum` | `Φ=(0,-0.01,-0.03,-0.02)`、`gamma=0.999`、`terminated=True` → `abs(discounted_telescope_residual) < 1e-15`，且 `sum(reward)` **不**接近 `-Φ[0]`（证明用的是 discounted 而非朴素望远镜） |
| `test_truncation_carries_final_physical_potential` | `Φ=(-0.01,-0.02,-0.03)`、`terminated=False` → `discounted_return == -Φ[0] + 0.999² * Φ[-1]`（`places=15`）、`carried_terminal_potential == Φ[-1]` |
| `test_sum_cap_can_suppress_wall_marginal_when_vehicle_is_saturated` | `SUM20`，三项 clearance 全 0 → `potential[0] == -0.05` 且 `potential_without_wall[0] == -0.05`（即墙项边际被 cap 吞掉） |

### 2.6 历史 `test_risk_longitudinal_ab.py`（已退役；122 行，4 个 test）

| test | 断言 |
|---|---|
| `test_cli_default_and_l12_override_are_explicit` | 裸命令 `risk_longitudinal_clearance_m == float(PPO_CONFIG["risk_longitudinal_clearance_m"])`；`--risk_longitudinal_clearance_m 1.2` → 1.2 |
| `test_effective_config_records_the_override_without_mutating_file_config` | `effective_ppo_config(Namespace(risk_longitudinal_clearance_m=1.2))["risk_longitudinal_clearance_m"] == 1.2`，**同时 `PPO_CONFIG[...] 仍为 0.6`**（不得改全局 YAML 默认） |
| `test_environment_passes_default_and_override_to_reward` | 用 `SimpleNamespace` 造假 core（`params={"length":0.58,"width":0.31}`、`sim.agents=[agent,agent]`、`agent.scan_simulator` 带 `dt=ones((2,2))`/`map_resolution=0.05`/`origin=zeros(2)`、`core.unwrapped=core`）；patch 掉 `ppo.environment.OccupancyMapClearance` 与 `PPOTransitionReward`；`supplied=None→0.6`、`1.2→1.2` 传给 reward 的 kwargs |
| `test_run_config_records_seed_and_effective_reward_value` | 在 `POST_TRAINED_ROOT` 下建临时目录（**必须在该 root 内，因为 `train_ppo` 要求 output dir 位于 `post-trained/`**），`TrainingRecorder(dir, hidden_scale=4).write_run_config(args, effective_ppo_config(args), {})` 后 `run_config.json` 的 `args.seed==42`、`args.risk_longitudinal_clearance_m==1.2`、`ppo_config.risk_longitudinal_clearance_m==1.2` |

### 2.7 `test_speed_std_annealing.py`（89 行，5 个 test）

被测：`ppo.algorithm.linear_speed_std_for_update(initial, final, anneal_updates, update)`
与 `train_ppo.parse_arguments/validate_arguments`。

```text
固定默认：linear_speed_std_for_update(0.15, None, 0, 1) == 0.15；(…, 0, 30) == 0.15
10-update 线性表：update ∈ [1,10] 时 0.40 + (update-1)*(0.15-0.40)/9（places=15）
                  update 11 与 30 都恰为 0.15（退火结束后保持 final）
CLI 默认：speed_physical_std==0.15、speed_physical_std_final is None、
          speed_physical_std_anneal_updates==0
CLI 接受完整臂：--speed_physical_std 0.40 --speed_physical_std_final 0.15
                --speed_physical_std_anneal_updates 10 --num_updates 30 → validate 通过
CLI 拒绝不完整：只给 --speed_physical_std_anneal_updates 10 →
                ValueError 且消息匹配 "requires speed_physical_std_final"
```

五个测试名依次为`test_fixed_default_is_unchanged`、
`test_ten_update_schedule_hits_both_endpoints_and_stays_final`、
`test_cli_defaults_preserve_fixed_std_contract`、
`test_cli_accepts_the_single_annealing_arm`、
`test_cli_rejects_incomplete_schedule`。

### 2.8 `test_ordinary_startpoint_extension.py`（65 行，2 个 test）

模块常量 `BASELINE_AUSTIN_STARTPOINTS` 是 50 个整数的元组，**必须逐字保留**：

```text
21, 63, 105, 147, 189, 231, 272, 314, 356, 398,
440, 482, 524, 566, 608, 650, 692, 734, 776, 818,
861, 904, 947, 989, 1032, 1064, 1106, 1149, 1192, 1234,
1278, 1320, 1362, 1404, 1446, 1488, 1530, 1572, 1614, 1656,
1698, 1740, 1782, 1824, 1865, 1907, 1949, 1991, 2033, 2075
```

- `test_extension_preserves_baseline_prefix_and_scenarios` ——
  `ordinary_startpoints("Austin",50)` 等于上表；`(…,150)[:50]` 等于它；长度 150 且
  全部唯一。`ordinary_scenarios("Austin",50)` 长度 600、`(…,150)` 长度 1800，
  且**后者前 600 条与前者逐条相同**（扩展必须是前缀扩展）。
- `test_all_extended_starts_are_separated_from_current_eval` ——
  读 `raceline1.csv`（`delimiter=";"`、`comments="#"`，**丢掉最后一行** `[:-1]`），
  eval 起点用 `utils.get_circular_startpoints("Austin","raceline1.csv",50,0)`，
  150 个训练起点到 50 个 eval 起点的最小欧氏距离 `>= 1.0 - 1e-12`。

### 2.9 `test_hard_neighbors.py`（311 行，3 个 class）

三个 helper：
`make_lattice()` 造 4 条 collision ScenarioSpec（`interval_idx ∈ (8,10)` ×
`speed_scale ∈ (0.5,0.6)`，`ego_idx=10`、`opp_idx=10+interval`、`raceline1`、Austin，
id `toy-{0..3}`）；`make_outcomes(scenarios, values)` 生成
`{"candidate_index","scenario_id","outcome"}` 列表；
`make_scheduler_pool(pool, count)` 造 `{pool}-{i:03d}` 的 spec（`ego_idx=10+i`、
`opp_idx=18+i`、speed 0.5、interval 8）。

`class BoundaryDiscoveryTests`（被测 `ppo.hard_neighbors.discover_boundary_candidates` /
`materialize_boundary_candidates`，调用参数
`interval_indices=(8,10), speed_scales=(0.5,0.6), max_candidates_per_family=…`）:

- `test_discovers_only_one_axis_outcome_flips` —— outcomes
  `("ego_collision","other","other","other")` → `len(pair_records)==2`、
  `generated_candidate_count==2`、精化点集恰为 `{(8,550),(9,500)}`（speed 用**千分位整数**
  表示：0.55→550），**不含对角点 `(9,550)`**；`{record["axis"]}=={"speed","interval"}`。
- `test_invalid_endpoint_is_not_a_boundary` —— outcomes
  `("ego_collision","invalid","other","other")` → 只剩 1 个 pair、精化点 `{(9,500)}`。
- `test_family_cap_is_deterministic` —— `max_candidates_per_family=1` 连跑两次结果
  **完全相等**（`first == second`）；`generated_candidate_count==2` 但
  `len(candidates)==1`；所有 pair 的 `selected_scenario_ids` 总数为 1。
- `test_materialization_preserves_physics_and_finite_reset` —— 2 条物化场景 id 唯一、
  `pool == "hard_neighbor"`、`opp_idx == (ego_idx + interval_idx) % 2096`
  （2096 是 Austin raceline 点数）；`to_reset_spec("collision")` 的 `poses` 与
  `initial_speed_feature` 全有限。

`class TrainingSwitchTests`:

```text
裸命令                                  → hard_neighbors 假、hard_neighbor_fraction is None
--hard_neighbors                        → 真、fraction 仍 None（旧 805 uniform 语义）
--hard_neighbors --hard_neighbor_fraction 0.20
                                        → 真、0.20，且 hard_neighbor_cache_dir 默认
                                          "post-trained/collision-cache/pretrained_end2race_austin_boundary_aware_collision_pool_805"
```

对应测试名：`test_hard_neighbors_are_off_by_default_and_explicitly_enabled`。

`class HardNeighborSchedulingTests`（setUp：base 17 条、hard 11 条、ordinary 13 条）:

- `test_fraction_is_exact_over_each_quota_cycle` —— 连取 100 个 collision scenario：
  `fraction=0.20` → 恰 20 个 hard，且前 10 个中 hard 位置恰为 `{2,7}`；
  `fraction=0.10` → 恰 10 个，前 10 中位置恰为 `{5}`。（**确定性周期，不是随机**）
- `test_base_and_ordinary_relative_orders_match_legacy_scheduler` ——
  分层 scheduler 里过滤出 `pool=="collision"` 的序列，必须与 legacy
  `ScenarioScheduler(42, base, ordinary)` 的前 50 个逐一相同；ordinary 前 50 个也相同。
- `test_scheduler_state_round_trip_is_exact` —— 走 37 步后 `state_dict()`，
  新实例 `load_state_dict` 后续 100 组 `(collision_id, ordinary_id)` 与源完全一致。
- `test_stratified_mode_requires_both_collision_sources` —— 只给 base 就用
  `fraction=0.20` → `ValueError` 且消息匹配 `"non-empty base and hard-neighbor"`。

### 2.10 `test_ordinary_offline_fast_sampling.py`（189 行，2 个 class）

该测试对应的活动功能已于2026-08-14删除；以下只保留历史重建断言。

文件顶部有长 docstring 说明动机（CT-v2 的回退集中在 off-line fast regime），
并 `sys.path.insert(0, parents[1])`。`build_scenarios()` 返回
`(list(expanded_scenarios("Austin"))[:479], ordinary_scenarios("Austin"))`。

`class OfflineFastPredicateTests`:

- `test_predicate_matches_regime_definition` —— `raceline0 + 0.7` → 真；
  `(EGO_RACELINE, 0.8)`、`("raceline2", 0.6)`、`(EGO_RACELINE, 0.5)` 全假。
- `test_shipped_ordinary_panel_split` —— ordinary 600 条中 off-line-fast 恰 **200** 条。

`class StratifiedOrdinaryQueueTests`（helper `_shares(fraction, draws=30000)`
统计 same-line / offline-fast / offline-slow 三档实际份额）:

| test | 断言 |
|---|---|
| `test_default_path_is_unchanged_and_reproducible` | 同 seed 两实例前 500 个 ordinary id 完全相同；`sorted(state_dict()) == ["collision","ordinary"]`（**state key 集合不得变，否则已有 run 无法 resume**）；默认份额 ≈ `200/600`（places=4） |
| `test_same_line_share_is_preserved_exactly` | `fraction ∈ (0.5,0.6,0.55)`：same-line 份额恒 ≈ `200/600`、fast ≈ fraction、slow ≈ `1-200/600-fraction`（全 places=4）。**同线份额不可移动**是硬约束 |
| `test_rejects_fraction_that_starves_offline_slow` | `2/3`、`0.7`、`0.9` → `ValueError` |
| `test_realised_share_equals_requested_fraction` | `0.5/0.6/0.25` 实测份额 == 请求值（places=4） |
| `test_rejects_out_of_range_fractions` | `0.0`、`1.0`、`-0.5`、`1.5`、`nan` → `ValueError` |
| `test_state_dict_round_trip` | 走 137 步后 round-trip，后续 60 个 id 一致 |
| `test_state_dict_rejects_mismatched_fraction` | 0.6 的 state 载入 0.5 的 scheduler → `ValueError` |
| `test_stratified_state_dict_rejects_default_state` | 默认 state 载入分层 scheduler → `(ValueError, KeyError)` |
| `test_reweighting_does_not_change_the_scenario_set` | 30000 次抽取见到的 id 集合 **等于** 全部 ordinary id 集合（只改权重不改可达集） |

### 2.11 `test_fixed_collision_pool.py`（139 行，4 个 test）

被测 `ppo.fixed_collision_pool.load_fixed_collision_pool(path, map_name=…)`。
`_payload()` 造 schema_version=1 的 JSON：`source`（root + 4 个 sha 字段）、
`selection`（`split="train"`、`interval_idx=15`、`near_miss_clearance_m=0.1`、
`include_outcomes=["ego_collision","overtake_or_follow_near_miss"]`）、
`sampling`（`mode="uniform_cycle_over_combined_pool"`、`scenario_count=2`、
`source_label_counts={"ego_collision":1,"near_miss":1}`）、`entries`（每条含
`source_label`、`source_outcome`、`min_obb_clearance_m`、`scenario`=`asdict(ScenarioSpec)`）。
两条场景：`collision-a`（ego 10 / opp 25 / raceline1 / 0.5 / interval 15）与
`collision-b`（ego 20 / opp 35 / raceline2 / 0.7 / interval 15）。

```text
test_loads_valid_pool_and_records_hash
           → len(scenarios)==2；info["fixed_collision_pool_sampling"]["source_label_counts"]
             == {"ego_collision":1,"near_miss":1}；len(info["fixed_collision_pool_sha256"])==64
test_rejects_wrong_interval
           → interval 改 12，RuntimeError 匹配 "map/pool/interval"
test_rejects_near_miss_above_threshold
           → near-miss 的 min_obb_clearance_m 改 0.2（>0.1），RuntimeError 匹配 "near-miss label"
test_rejects_duplicate_physical_scenario
           → 把第二条的 ego_idx/opp_idx/raceline/speedscale 全改成与第一条相同
               → RuntimeError 匹配 "physical scenarios"（物理键去重）
```

### 2.12 `test_select_critic_lr_candidate.py`（127 行，2 个 test）

从 `select_critic_lr_candidate` import `CANDIDATES`、`LATE_UPDATES`、`select`。
`LATE_UPDATES` 是三个后期 update（测试里按 `[35, 40, 45]` 使用，`best_late_checkpoint`
断言 35）。`CANDIDATES` 是 4 个 `Candidate` dataclass，字段至少含
`key`、`run_name`、`hard_neighbors`、`hard_neighbor_fraction`；4 个 key 含 `"hard020"`。

测试在临时 root 下造完整目录树：`post-trained/<run>/run_config.json`（`{"args": {...}}`
含 critic/env_workers/batch_size/num_updates/epochs/三个 LR/std/clip/target_kl 及
候选自身的 hard 设置）、`post-trained/<run>/checkpoints/actor_u{update:04d}.pth`、
`eval_results/<run>_u{update:04d}_Austin/multiagents/results_multi.json`
（`final` 含 total_episodes=600、following/overtaking/collision/error_count、
avg_speed_mean=5.0；`episodes` 是 600 个 `episode-{i:03d}`）。

- `test_selects_aggregate_late_collision_winner` —— 四候选后期碰撞
  `[12,11,12] / [20,22,17] / [10,13,11] / [11,12,12]`（超车分别 350-357 / 365-366 /
  358-360 / 360-362）→ `selected_candidate.key == "hard020"`、
  `best_late_checkpoint.update == 35`、
  `critic_lr_experiment.hard_neighbor_fraction == 0.20`、
  `critic_lr_experiment.starts_from == "pretrained/end2race.pth"`。
- `test_rejects_incomplete_evaluation` —— 任一 panel `error_count=1` →
  `ValueError` 匹配 `"contains 1 errors"`（**fail closed，不许跳过残缺面板**）。

### 2.13 `test_build_heldout_hard_instrument.py`（120 行，pytest 风格，4 个函数）

从 `build_heldout_hard_instrument` import `MAP_NAME`、
`NEAR_MISS_MAX_PER_SPLIT`、`build_candidate_scenarios`、
`deterministic_near_miss_sample`、`existing_startpoints`、`load_completed_prefix`、
`persist_shard`、`select_instrument_startpoints`；从 `ppo.outcome_aware_hard`
import `CandidateLabel`。

- `test_instrument_startpoints_are_blocked_and_disjoint` ——
  `select_instrument_startpoints(MAP_NAME)` 返回 `(selected, audit)`：
  200 条；`split` 计数 `{"train":100,"heldout_eval":100}`；200 个 `ego_idx` 唯一；
  与 `existing_startpoints()` 全部并集**无交集**；
  `audit["minimum_distance_to_existing_start_m"] >= 0.5`；
  `audit["minimum_cross_split_physical_distance_m"] >= 3.5`；
  20 个 block 各 10 条，且**偶数 block 全为 train、奇数 block 全为 heldout_eval**。
- `test_instrument_candidate_grid_is_complete_and_unique` ——
  `build_candidate_scenarios(selected, MAP_NAME)` 长度 **21,600**、id 全唯一、
  pool 计数 `{"instrument_train":10_800, "instrument_heldout_eval":10_800}`。
- `test_near_miss_sampling_is_deterministic_and_excludes_collisions` ——
  取 `selected[:1]` 的场景，人工造 label（`outcome = "follow" if i%3 else "ego_collision"`、
  `min_obb_clearance_m = 0.05 if i%2 else 0.20`）；
  `deterministic_near_miss_sample` 连跑两次 id 序列相同（第二次传
  `[replace(l) for l in labels]` 证明不依赖对象身份）；数量 `<= NEAR_MISS_MAX_PER_SPLIT`；
  被选中的每条 `outcome != "ego_collision"` 且 `min_obb_clearance_m <= 0.10`。
- `test_completed_shards_resume_from_validated_prefix(tmp_path)` ——
  `persist_shard(tmp_path, scenarios, labels, 0, 50, actor_sha256="a"*64)` 后
  `load_completed_prefix(tmp_path, scenarios, 50, actor_sha256)` 返回
  `(saved, 50)`（**分片可恢复**，且校验 actor sha 与场景一致性）。

### 2.14 `test_analyze_l12_heldout_hard_instrument.py`（26 行，pytest 风格，2 个函数）

见 §1.4，全文即该节所述两个函数。依赖 `scipy.stats.binomtest`。测试名为
`test_exact_mcnemar_detectable_net_is_minimal`与`test_known_mcnemar_thresholds`。

### 2.15 `test_outcome_aware_hard.py`（386 行，5 个 class）

文件 docstring 说明分两档运行：默认快速套件（**不碰模拟器**），另有真实重放 smoke，
后者由环境变量 `END2RACE_RUN_SIM=1` 开启，否则 skip（skip 消息
`"set END2RACE_RUN_SIM=1 to run the tiny real-BC smoke"`）。
常量 `PROJECT_ROOT`、`BASE_CACHE_DIR`
（`post-trained/collision-cache/pretrained_end2race_austin_collision_pool_479`）、
`HARD_CACHE_DIR`
（`…/pretrained_end2race_austin_boundary_aware_collision_pool_805`）、
`MAP_NAME="Austin"`。
Helper：`_read_jsonl`、`_base_candidates_and_outcomes`、`_dummy_collision_label`、
`_noncollision_label`。

五个 class 的职责：

| class | 契约 |
|---|---|
| `RecordRoundTripTests` | `CandidateLabel` 等记录类型 JSONL 序列化/反序列化后逐字段相等（含 float 精度） |
| `FilterLogicTests` | outcome/clearance 过滤谓词：只有 `ego_collision` 进 collision pool；near-miss 需 `min_obb_clearance_m` 在阈值内；`invalid` 一律排除 |
| `BoundaryAwareReconstructionTests` | 由 base outcomes + generator config **确定性重建** boundary 候选，重建结果与 cache 内记录一致 |
| `CacheRoundTripTests` | 写入后再加载得到相同**有序**场景列表；manifest 校验；身份不一致时 fail closed |
| `RealReplaySmokeTests` | 小规模真实 reset/classification/load 全链路（需要模拟器，属慢速档） |

精确测试清单和不可删减断言：

- `test_label_record_round_trip`：`CandidateLabel.to_record/from_record` 后 dataclass 全等；
- `test_label_record_rejects_wrong_schema`：`label_schema=999` 必须 `RuntimeError`；
- `test_label_record_rejects_extra_field`：多出任意字段必须 `RuntimeError`；
- `test_all_mode_keeps_every_collision`：最终有序ID为
  `collision-a, hard-safe-fish, hard-follow-fish, hard-safe-rearend`，审计 kept=3；
- `test_safe_overtake_drops_follow_side`：保留两个 safe-side，丢
  `hard-follow-fish`，审计 kept=2；
- `test_fishtail_mode_keeps_only_safe_post_overtake_fishtail`：只剩
  `collision-a, hard-safe-fish`；
- `test_fishtail_rearend_adds_rear_end_quota`：保留 safe fishtail 和 safe rear-end，
  丢 follow-side；
- `test_safe_clearance_threshold_is_enforced`：把 safe-side clearance 降到0.05后必须丢弃；
- `test_require_all_pairs_safe_vs_any`：同一候选来自一安全一follow pair时，all-safe为空，
  any-safe保留`hard-multi`；
- `test_invalid_mode_rejected`：未知mode必须`ValueError`；
- `test_all_mode_reconstructs_boundary_aware_v1`：从冻结base outcomes重建出的最终有序
  ScenarioSpec记录必须与已发布cache逐字节相同，final count一致；
- `test_build_then_load_round_trips`：safe-overtake tiny cache build/load后只含
  `collision-a, hard-safe-fish`，summary含`filter_audit`；
- `test_refuses_to_overwrite_existing_cache`：同一位置二次发布必须`RuntimeError`；
- `test_tampered_pool_is_rejected_on_load`：篡改最终pool但不改manifest必须`RuntimeError`；
- `test_config_mismatch_is_rejected`：发布后改变任一config字段必须`RuntimeError`；
- `test_boundary_labels_match_shipped_binary_outcomes`：仅在
  `END2RACE_RUN_SIM=1`运行；取一collision与一other真实重放，前者仍为collision且
  collision mode/bearing存在，后者必须为overtake或follow。

### 2.16 `test_structured_speed_exploration.py`（373 行，2 个 class）

`build_policy(mode)` 构造真实 `End2RaceGRUPolicy`：observation space
`Box(-inf, inf, shape=(END2RACE_OBSERVATION_SIZE,))`（361）、action space
`Box(low=(-0.52,-inf), high=(0.52,inf))`、`lr_schedule=lambda _p: 1.0`、
`checkpoint_path=pretrained/end2race.pth`、`hidden_scale=4`、`critic_variant="mlp"`、
LR `3e-6/3e-5/3e-4`、`steering_latent_std=0.03`、`speed_physical_std=0.15`、
`speed_exploration_mode=mode`。**需要真实 BC checkpoint 存在。**

`class DistributionTests`:

- `test_per_sample_speed_std_replays_exact_log_probability` ——
  `EvaluatorCompatibleJointDistribution`，means `[[0.1,3.0],[-0.2,4.0],[0.0,5.0]]`、
  `base_log_std=log([0.03,0.15])`、`speed_log_std=log([0.15,0.50,0.25])`、
  `standard_noise=[-1.0,0.5,1.5]`、`torch.manual_seed(12)`。
  `sample_with_speed_standard_noise(noise)` 得 actions，然后
  `log_prob(actions)` 与重建分布的 `log_prob(actions)` **完全相等**（`rtol=0, atol=0`）；
  且 `(actions[:,1]-means[:,1]) / speed_std == standard_noise`（`atol=2e-6`）。

`class PolicyModeTests`:

- `test_baseline_forward_is_exact_legacy_sampling_path` —— 保存/恢复
  `torch.get_rng_state()`，比较 `policy.forward(...)` 与手工
  `_actor_forward → _distribution → get_actions → log_prob` 路径的 actions/log_prob/
  hidden **完全相等**（`rtol=0, atol=0`）；且
  `set(policy.actor_checkpoint_state_dict()) == set(torch.load(BC, weights_only=True))`。
- `test_conditional_white_changes_only_gated_speed_std` ——
  `prepare_rollout_exploration(gates=[F,T,F], starts=[T,T,T])` 后
  `_structured_rollout_parameters(3)` 返回 `(speed_log_std, noise, gates, temporal, block)`：
  `noise is None`；`exp(speed_log_std) == [0.15, 0.50, 0.15]`（`atol=1e-7`）；
  `gates.tolist()==[False,True,False]`；`temporal.any()` 为假。
- `test_global_temporal_noise_is_held_for_exactly_fifty_actions` ——
  连走 `TEMPORAL_RESAMPLE_STEPS+1 = 51` 步（只有 step 0 是 episode start）：
  前 50 步 noise **只有一个不同值**、`block_ids[:50] == [1]*50`；
  第 51 步 noise 与第 1 步不同、`block_ids[-1]==2`；全程 `active` 为真。
- `test_conditional_temporal_block_survives_gate_drop_and_reset_clears` ——
  只有 step 0 有 gate/start，其后 gate 掉落：50 步内 `active` 恒真、
  `exp(std)` 恒 `0.25`（`<1e-6`）、noise 只有一个值、block 恒 `{1}`。
  随后 `starts=[True]`、`gates=[False]` → `active` 假、`std==0.15`（places=6）、`block==0`。
- `test_corridor_temporal_holds_baseline_std_for_fifty_actions` —— 同上，但
  `exp(std)` 恒 **0.15**（corridor 臂不抬 std，只做时序保持）。
- `test_front_corridor_gate_matches_frozen_geometry` —— 读真实
  `f1tenth_racetracks/Austin/raceline1.csv`（`delimiter=";"`、`comments="#"`），
  ego 固定在 index 0，对手放在 index N 并可沿法向偏移 `lateral_d_m`：
  `gate.reset(observation(6))` → True；`gate.step(observation(18), 0.01)` → False（太远）；
  `gate.step(observation(6, lateral_d_m=0.40), 0.02)` → False（出走廊）。
- `test_recurrent_buffer_replays_each_structured_distribution_at_ratio_one` ——
  对 4 个非 baseline mode 各跑一次：`torch.manual_seed(20260729)`、`n_steps=8`、
  `n_envs=2`，建 `End2RaceRolloutBuffer(gamma=0.999, gae_lambda=0.995)`，
  逐步 `prepare_rollout_exploration(gates, starts)` → `policy.forward` → `buffer.add`，
  其中 `starts=[step==0, step in (0,4)]`、`gates=[step in (1,2,5), step in (0,4)]`。
  `compute_returns_and_advantage` 后遍历 `buffer.get(8, rng=default_rng(7))`，
  用 `mask > 1e-8` 过滤，比较 `evaluate_actor_actions` 与 `old_log_prob`：
  **普通重放误差 <= 3e-5；`collection_equivalent=True` 重放误差 <= 2e-6**；
  有效 transition 数恰为 `n_steps*n_envs`。

---

## 3. 实验工具重建规范（B 级，47 个 Python）

统一样板（所有脚本共有，重建时先写这层）：

```python
"""<一句话目的>"""
from __future__ import annotations
PROJECT_ROOT = Path(__file__).resolve().parents[1]
def parse_arguments() -> argparse.Namespace: ...   # 见每条的 CLI
def sha256_file(path) / atomic_write_json / atomic_write_csv / write_csv
def main() -> None: ...
if __name__ == "__main__": main()
```

所有分析脚本共有的验证前置（**不通过就抛异常，不写产物**）：panel episode 数与预期
一致、场景 ID 唯一、`error_count == 0`、结果与 trace key 集合相等、全部数值有限、
collision marker 与 outcome 一致、terminal 行契约成立。

### 3.1 Panel / pool 构建器（6 个）

| 文件 | 目的与算法 | CLI |
|---|---|---|
| `build_austin600_panel.py` (123) | 输出当前固定 Austin600 的 ScenarioSpec JSON 列表（50 eval 起点 × 3 raceline × 4 speed，interval 15） | `--output` |
| `build_crossmap600_panels.py` (208) | 为 Hockenheim/MoscowRaceway/Nuerburgring 各建 600 场景确定性 panel。常量 `OPPONENT_LINES`、`SPEED_SCALES=(0.5,0.6,0.7,0.8)`、`STARTPOINT_COUNT=50`、`INTERVAL_INDEX=15`。`minimum_pairwise_distance()` 实测起点间距并写入；`write_immutable_json` 拒绝覆盖已有文件 | 无（常量驱动） |
| `build_l12_validation_panels.py` (119) | 建 L12 迁移检查用的 held-out 近距 panel；常量 `INTERVALS`、`SPEED_SCALES` | `--map-name`、`--output`(必需) |
| `build_interval15_difficult_pool.py` (268) | 从 held-out 工具产物中筛出 interval-15 的 collision + near-miss，构建冻结训练池。`NEAR_MISS_CLEARANCE_M` 为 near-miss 阈值。输出 train 池与 held-out 池两份 | `--source-root`、`--output`、`--heldout-output` |
| `build_collision_cache.py` (221) | 不启动 PPO 而单独构建一个严格 collision cache；读 `ppo/ppo_config.yaml` 的 start method；写 `RECEIPT_NAME` 收据并与 `expected_receipt()` 比对 | `--pretrained-model-path`、`--cache-dir`、`--map-name`、`--hidden-scale`、`--env-workers` |
| `build_heldout_hard_instrument.py` (820) | **最复杂的构建器**，见下 | `--actor-path`、`--output-dir`、`--workers`、`--hidden-scale`、`--shard-size`、`--prepare-only` |

`build_heldout_hard_instrument.py` 关键契约（测试已钉住，见 §2.13）：

```text
SCHEMA_VERSION、MAP_NAME="Austin"
BLOCK_COUNT=20、STARTPOINTS_PER_BLOCK=10  → 200 起点
偶数 block → split "train"；奇数 block → "heldout_eval"（各 100）
BLOCK_EDGE_BUFFER_M=2.0、MIN_EXISTING_START_DISTANCE_M=0.5、
MIN_WITHIN_BLOCK_PROGRESS_M=0.8、cross-split 最小物理距离 >= 3.5 m
候选网格：200 起点 × 108 组合 = 21,600（train/heldout 各 10,800），pool 名
  "instrument_train" / "instrument_heldout_eval"
NEAR_MISS_CLEARANCE_M=0.1、NEAR_MISS_MAX_PER_SPLIT=400
EVAL_STARTPOINT_COUNT=50、DEFAULT_SHARD_SIZE=600
分类分片可恢复：persist_shard / load_completed_prefix / validate_shard，
  绑定 actor sha256；MAX_CONSECUTIVE_POOL_RESTARTS 限制重启次数
选择只用冻结 U30，**绝不读 Control-U15 / L12-U15**（防 treatment 泄漏）
```

### 3.2 历史显式ScenarioSpec评估执行器

> **历史状态与当前边界**：`scripts/evaluate_scenario_panel.py`曾于2026-07-30重建；2026-08-10确认
> 四图标准JSON与`evaluate.sh`生成身份逐条相同且没有其他当前消费者后，该脚本与重复JSON一起
> 删除。当前普通模型eval只能使用`evaluate.sh`。只有未来明确授权不规则显式ScenarioSpec面板时，
> 才按本节重建，不得为标准四图600恢复第二套入口。重建版与原版的差异：默认 `--collision-scope ego`（面板口径，
> 见 HANDOFF §4.3）；运行前用 `get_opponent_startpoint` 逐条校验面板 `opp_idx`，
> 不一致即 fail closed；2026-08-06修正为通过`evaluate_segment(..., trace_output_path=...)`
> 直接写`--output-dir/traces`，不得先写model-derived canonical根再搬移。旧搬移实现会掏空
> 已存在的canonical trace目录，已在跨地图fresh重评中实际触发，因此不得恢复。manifest记
> `status: fresh_evaluation`和实际`device`，以区别于既有臂的
> `complete_trace_reconstruction` 包。已用它精确复现 production 的
> near400 `28/325` 与 hard73 `54/12`。

`evaluate_scenario_panel.py`在旧的显式panel体系中曾是优先重建脚本；**当前标准四图评测不得重建它**。
与 `evaluate.sh` 的区别：不构造固定笛卡尔积，而是消费一份显式的 ScenarioSpec JSON，
因此可以评训练 collision cache、自建 panel 或跨地图 panel。

```text
CLI: --model-path --panel --output-dir --panel-id（必需）
     --map-name=Austin --workers=12 --hidden-scale=4 --sim-duration=8.0
     --device={auto,cpu,cuda} --collision-scope={legacy,ego} --save-traces
行为: load_panel → opp_idx/key校验 → forkserver worker池（worker_init限线程）
     → evaluate_one逐场景确定性评估 → episodes.partial.jsonl按episode key恢复
     → results_multi.json保存final+逐episode，eval_manifest.json保存协议和完整性
     → --save-traces时每个worker直接写output-dir/traces/<episode_key>.npz
陷阱: CPU/GPU确定性actor的微小数值差可在临界episode造成outcome翻转；manifest必须记录device，
     同一对照不能混用device。旧的“先写canonical再搬trace”逻辑禁止恢复
```

当前稳定输出合同：

```text
results_multi.json:
  final: total_episodes, follow/overtake/collision及子类、均速/距离、error_count
  episodes: {episode_key: 完整metrics+ScenarioSpec身份}

eval_manifest.json:
  schema_version=1, status="fresh_evaluation", complete, comparison_ready
  actor_path/actor_sha256, map/panel身份, scenario/result/trace count
  deterministic_actor, device, noise=0.0, sim_duration_s, collision_scope
  unique_episode_keys, trace_result_key_sets_equal, opp_idx校验结果
  command, git_commit, worktree_status（2026-08-08起由fresh runner直接记录）

episodes.partial.jsonl:
  每个完成episode一行完整metrics，按episode_key恢复；final成功后删除
```

重建时还必须保留：

- 接受bare list、`{"entries": [...]}`或`{"scenarios": [...]}`三种panel payload；
- 用`get_opponent_startpoint()`重算并核对`opp_idx`，再用`episode_key()`检查key唯一；
- partial按`episode_key`恢复，只运行缺失场景，完成数必须严格等于panel数；
- `--save-traces`时trace count必须等于episode count，且直接写目标目录；
- opponent-only wall事件在ego scope不终止episode，若保存trace则单独统计；
- panel为空、字段缺失、opp_idx不一致、episode key重复或完成数不等均fail closed。

2026-08-06回归：冻结production U30、Austin hard73 panel、CPU、8 workers、ego scope下，
目标目录直接生成73个episode result和73个NPZ，key集合完全一致；运行前后既有Austin
production `multiagents/near400/hard73` trace计数保持`600/400/73`。跨地图CUDA fresh评测
精确复现历史`80 collisions / 1142 overtakes`及same-line/off-line `66/14`；CPU为`78/1142`
及`64/14`，因此`device`是必须固定和记录的评测协议字段，不是可忽略的运行细节。

### 3.3 离线 reward 候选验证（5 个，A/B 级混合）

| 文件 | 内容 |
|---|---|
| `postpass_reward_calculation.py` (249) | **A 级，见 §1.8 全文规范** |
| `validate_postpass_reward.py` (1068) | Post-pass 宽门扫描。常量 `PASS_MARGINS_M`、`SAFE_REAR_GAPS_M`、`REAR_CLOSING_DEADBANDS_MPS`、`CLEAR_MODES=(latched, reactive)`、`PROPOSED_REWARD_WEIGHT_PER_M/MAXIMUM_STEP_PENALTY/MAXIMUM_EPISODE_PENALTY`。`class TrackProjector` 从 raceline CSV 建投影。核心函数 `replay_episode_geometry(trace, projector)` 与 `setting_episode_result(geometry, time_s, *, pass_margin_m, safe_rear_gap_m, closing_deadband_mps, clear_mode)`（**这两个 + `validate_trace` + `TrackProjector` 被 §2.3 测试直接 import，签名与返回键名是契约**）。返回键至少含 `pass_detected`、`triggered`、`preterminal_triggered`、`preterminal_trigger_steps`、`first_trigger_lead_to_terminal_s`、`active_steps`、`basis_sum_m`、`proposed_reward_sum`。历史规模：4 个 panel、2400 episode、1,903,873 transition、96 组设置 |
| `compare_postpass_formulas.py` (434) | 在保存的 trace 上对比 `no_q` / `q_linear` / `q_squared` 三式（`FORMULAS` 常量）。`replay_trace(trace, projector)` 返回按 formula 的 row 列表，键含 `formula`、`trigger_steps`、`applied_penalty`、`raw_penalty`、`trigger_q_mean`、`episode_cap_hit_count`、`step_cap_hit_steps`（**被 §2.4 测试 import**）。门控三式完全相同，只改已触发后的幅度 |
| `validate_following_response_reward.py` (1304) | following-response 候选的离线因果重放。`class ReplayConfig / TrackProjection / FrenetProjector / GateReplay`；三候选 `linear closing excess` / `required deceleration` / `escalating required deceleration`。因果助手 `_causal_window_rate`、`_smooth_l1`、`_unwrap_relative_progress`、`_frenet_footprint_geometry`。`_run_internal_checks()` 必须验证：赛道轴/Frenet 方向、closing sign、**未来样本不改变过去速率**、grace/persistence 因果重叠。输出预注册准入矩阵 + `ready_for_training_ab` 布尔 |
| `compare_risk_potential_variants.py` (1026) | **见 §1.11**。复用 `analyze_baseline_reward_seed42` 的 transition 重放，不改 `ppo/reward.py`。`formula_self_check()` 做望远镜残差自检 |

`validate_postpass_reward.py` 清理前固定矩阵：

```text
vehicle length/width = 0.58/0.31m
pass margins        = 0, 0.05, 0.10, 0.30m
safe rear gaps      = 0.30, 0.60m
closing deadbands   = 0, 0.05, 0.10, 0.15, 0.20, 0.30m/s
clear modes         = latched, reactive
reward weight       = 0.25/m
step/episode caps   = 0.005 / 0.05
```

共`4×2×6×2=96`组。候选选择先保留最高reference-tail capture，再按以下tuple从小到大：

```text
follow triggers, episode-cap hits, total penalty, trigger-step fraction,
overtake trigger rate, pass margin, -safe gap, -closing deadband, clear mode
```

`setting_episode_result()`完整关键返回键：
`pass_detected/pass_index/active_steps/trigger_steps/preterminal_trigger_steps/triggered/
preterminal_triggered/basis_sum_m/basis_max_step_m/proposed_reward_sum/
proposed_penalty_sum/proposed_penalty_nonzero_steps/proposed_step_cap_hit/
proposed_episode_cap_hit/minimum_signed_rear_gap_m/first_trigger_index/
last_trigger_index/first_trigger_time_s/first_trigger_lead_to_terminal_s`。

`compare_risk_potential_variants.py` 清理前的精确变体矩阵：

```text
共同参数:
  lateral_safe=0.2m, wall_safe=0.2m, maximum_magnitude=0.05,
  power=2, tight-follow longitudinal=0.8m, collision window=1.5s,
  numerical tolerance=1e-15
B0    longitudinal=0.6m, composition=min,     role=baseline
L12   longitudinal=1.2m, composition=min,     role=single_axis
L20   longitudinal=2.0m, composition=min,     role=single_axis
SUM06 longitudinal=0.6m, composition=sum_cap, role=single_axis
SUM12 longitudinal=1.2m, composition=sum_cap, role=interaction
SUM20 longitudinal=2.0m, composition=sum_cap, role=interaction
```

注意：`min`标签在源码中的实际basis组合是
`max(vehicle_basis, wall_basis)`，因为potential最终取负；`sum_cap`是
`min(1, vehicle_basis + wall_basis)`。不得按标签把前者误写为两个basis取min。

`validate_following_response_reward.py::ReplayConfig` 的清理前固定值：

```text
vehicle length/width                  0.58 / 0.31 m
closing window                       0.10 s
entry/exit gap                       2.00 / 2.20 m
entry/exit closing time              1.50 / 2.00 s
corridor entry/exit |d|              0.20 / 0.25 m
lateral overlap entry                0.02 m
warning grace / recovery hold        0.20 / 0.10 s
safe gap / response horizon          0.50 m / 1.00 s
linear excess persistence            0.10 s
required / escalating deceleration   1.50 / 1.25 m/s²
required-decel persistence/window    0.20 / 0.20 s
minimum deceleration growth          2.00 m/s³
lateral escape/opening window        0.40 / 0.10 s
smooth-L1 delta                      0.50 m/s
reward weight / step cap / ep cap    0.10/m / 0.005 / 0.05
required-decel report cap            100 m/s²
```

这些值只重建离线候选；不能因此把following-response写入production reward。

### 3.4 实验分析器（19 个）

共同结构：读一个或多个 eval panel 的 `results_multi.json`（+ 可选 traces）→ 验证 →
按场景身份配对 → 精确 McNemar → 写 `summary.json` + 若干 CSV。

| 文件 | 分析对象 |
|---|---|
| `analyze_baseline_reward_seed42.py` (1300) | 默认四项 reward 的逐 transition 重放审计。常量 `SEED`、`GAMMA`、`WINDOWS_S`、`SCENARIO_PATTERN`。核心：`replay_episode`、`collision_features`、`last_contiguous_true_onset`、`yaw_rate_deg_s`、`window_metrics`、`summarize_cohort`。产出碰撞 taxonomy、risk 激活提前量、最后 1 s 净 shaping |
| `analyze_structured_speed_exploration.py` (1059) | 逐步独立、条件白噪声、全局时间相关、条件时间相关四臂两阶段分析（先主面板配对，再建紧凑诊断 panel 做 OL1 机制重放）。常量 `ARMS`、`PRIMARY_UPDATES`、`PANELS`、`EXPECTED_BASELINE_SHA256`（钉住production U30身份） |
| `analyze_l12_deterministic_transfer.py` (637) | cache372 / held-out near600 配对，含 `relative_progress`、outcome flow 矩阵 |
| `analyze_l12_heldout_hard_instrument.py` (423) | hard334 / near-miss400；`stratum_rows()` 按 interval/speed/raceline/block 分层；`exact_mcnemar_detectable_net`（**§1.4，被测试 import**） |
| `analyze_interval15_difficult_experiment.py` (406) | interval-15 困难池 A/B（`CONTROL_RUN`/`TREATMENT_RUN`/`UPDATES` 常量） |
| `analyze_speedstd050_experiment.py` (487) | std 0.50 三面板 A/B，含 `read_formal_metrics`、`normalized_run_config`（配置差异声明） |
| `analyze_speedstd_anneal_experiment.py` (667) | std 退火实验，含 `clearance_rows`、`training_comparison` |
| `analyze_baseline_late_checkpoint_stability.py` (467) | B 的 U20-U30 曲线稳定性 |
| `analyze_temporal_late_checkpoint_stability.py` (217) | T 与 B 在同一后期 checkpoint 对比 |
| `analyze_crossmap_bc_u30.py` (343) | BC vs U30 三张 held-out 地图，逐地图 + pooled McNemar |
| `analyze_crossmap_b_t.py` (321) | B vs T 跨地图；`EXPECTED_MODEL_SHA256` 钉身份 |
| `analyze_ctv2_corridor_temporal.py` (351) | CT-v2 U30 预注册跨地图主结果 + Austin600/near400 次级 |
| `analyze_ctv2_checkpoint_eval.py` (343) | CT-v2 U10/15/20/25/30 曲线 |
| `analyze_ctv2_late_stability.py` (244) | CT-v2 Austin600 u26-u30 band，与 B u24-u30 band 并列 |
| `analyze_ctv2_crossmap_band.py` (220) | CT-v2 跨地图 u27-u30 band vs B、T |
| `analyze_ctv2_gap1_band.py` (250) | 三臂 band：B / CT-v2 gap2.0 / gap1.0 |
| `analyze_arm_bands.py` (215) | 全部 exploration 臂在五个 panel 上的 band 级比较（`ARMS`、`REFERENCE_*` 常量） |
| `analyze_ctv2_loss_mechanism.py` (632) | **失败机制诊断**：2,800 对匹配场景，`collision_classification`（same-line/off-line × 碰撞子型）、`flow_label`（outcome 流向）、`pair_metrics`（1.5 s 事件窗口内配对动作差）、`cohort_membership`。常量 `HOLD_STEPS`、`EVENT_WINDOW_S` |
| `analyze_singleagent_noise_eval.py` (290) | BC/U30 单车 10 圈 × 3 地图 × 3 masking 等级；`NOISE_LEVELS`、`MODELS` |

`analyze_ctv2_loss_mechanism.py` 的事件对齐不可简化成“各取最后1.5秒”：

```text
若 treatment 新造碰撞: event_end = treatment collision time
若 treatment 消除碰撞: event_end = baseline collision time
否则:                    event_end = 两臂共同可比结束时刻
event_start = max(0, event_end - 1.5s)
```

随后只在`action_applied`且time/action/speed有限的行比较command/measured speed。raw gate
固定使用`front_corridor_overlap_gap2`，active gate是raw gate经`HOLD_STEPS=50`扩展；
必须同时输出raw/active exposure和hold spill，避免把时间保持后的步误称为原始几何命中。

### 3.5 Gate / 预飞诊断（5 个）

| 文件 | 内容 |
|---|---|
| `validate_ctv2_corridor_gate.py` (164) | 生产 CT-v2 raw gate 与冻结离线公式逐步交叉校验，要求**零误差**；`EXPECTED_MODEL_SHA`、`EXPECTED_SCENARIOS_SHA` 钉身份 |
| `sweep_corridor_gate_width.py` (171) | 在真实训练池上扫 corridor 臂宽 `--widths`（默认含 2.0）；输出各宽度的 episode 触发率与 active step exposure，按 all/same_line/off_line 分层 |
| `diagnose_ctv2_preflight_panels.py` (485) | 第二阶段离线预飞 panel 校验；`GATES` 常量枚举候选门；`paired_outcome_summary` |
| `diagnose_ctv2_ordinary_checkpoint_curve.py` (940) | 在 600 条 ordinary 训练池上重放首选门（`PREFERRED_GATE`、`WINDOW_SECONDS`），并比较配对 B-uK/T-uK actor 的固定状态曲线；含真实 actor 加载与批量重放 |
| `diagnose_crossmap_temporal_corridor.py` (683) | corridor-gated temporal处理的离线预飞：复现逐步独立/全局时间相关速度探索的raceline×speed联合表、场景标签oracle混合；`class BatchProjector`；常量 `CORRIDOR_ABS_D_M`、`TEMPORAL_HOLD_STEPS` |

### 3.6 Oracle / 可达性探针（5 个）

这一组的共同边界：**actor 网络与 recurrent 更新完全不变**，干预以 wrapper 形式在
仿真步注入；结论只能作为"可达性见证"，**不可部署**（使用了未来碰撞时刻与场景特定
优化参数）。搜索失败也不等于物理不可能。

| 文件 | 内容 |
|---|---|
| `run_u30_oracle_reachability.py` (750) | 对 13 个 U30 Austin600 失败搜索特权动作干预。`class ScheduledActor` 包装冻结 actor，在场景特定开环窗口施加分段仿射动作。`DT_S`、`SEGMENT_DURATION_S`、`SEGMENTS`、`VECTOR_SIZE`、`SAFE_OUTCOMES`；CEM 式搜索（`--population 24`、`--iterations 6`、`--elite-count 6`）；`match_controls` 做配对对照 |
| `evaluate_shared_oracle_library.py` (687) | 冻结 13 条 schedule 的库在 334 条 held-out U30 失败上的**覆盖门**（不是学习到的选择器）。`SEGMENT_DURATION_S`；分片 checkpoint（`--checkpoint-scenarios 8`）；输出 `coverage_summary.csv`（含 `rescued_rate`、`overtake_available_rate`） |
| `probe_shared_oracle_ranking.py` (1022) | 检验冻结 U30 状态能否**排序**这个共享库。`class StateTransform/StandardTransform`；`ACTION_COLUMNS`、`L2_GRID`；按 scenario 分组切分（`grouped_split`）、成对差分学习（`pairwise_differences`/`fit_pairwise`）、`choose_l2`、`wilson_interval`、`exact_mcnemar` |
| `scan_u30_braking_landscape.py` (604) | 扫制动幅度 × 干预提前量（`BRAKING_DELTAS_MPS`、`REQUESTED_LEADS_S`）；只用库中 schedule 12（steering 不动，5 段 0.30 s 恒定目标速度差） |
| `diagnose_actor_observability.py` (899) | 冻结 U30 GRU 状态的**线性可观测性**探针。`class LinearProbe`；`TARGET_NAMES`、`SAFE_COHORTS`、`COLLISION_COHORTS`；先用保存的 observation 重放并核对动作（`maximum_absolute_raw_action_error` 容差 2e-4），再按 episode 切分训练线性探针（`--max-epochs 200`、`--patience 20`、`--sample-stride 2`、`--event-window-s 1.5`） |

### 3.7 Checkpoint 选择与笔记本（4 个）

| 文件 | 内容 |
|---|---|
| `select_critic_lr_candidate.py` (334) | 4 个固定候选（无 hard / 805 uniform / 20% / 10%）的后期 checkpoint 选择器。`LATE_UPDATES`、`EXPECTED_EPISODES`、`CANDIDATES`（`class Candidate`）；`validate_run_config` 用 `require_equal` 逐字段核对共同控制；`candidate_score` 以后期聚合碰撞为主。**按 GUIDE §3 此脚本已停用**，重建仅为满足 §2.12 测试 |
| `probe_ctv2_late_behavior.py` (139) | CT-v2 后期区间行为探针：把「outcome 计数摆动」与「策略是否真的在变」分开；`SAME_LINE_PREFIX` 分组 |
| `build_u30_observability_oracle_notebook.py` (244) | 项目环境无 nbformat/nbclient，故用 stdlib 在**同一命名空间**执行所有 code cell、捕获 stdout，写出合法 v4 notebook。helper `markdown_cell`/`code_cell` |
| `build_u30_oracle_library_notebook.py` (227) | 同上，针对冻结 oracle library 诊断 |

### 3.8 其他（1 个）

`analyze_postpass_geometry_cases.py` (750) —— 在保存 trace 上审计 post-pass 几何门，
含 `RELATIVE_RISK_CLEARANCE_GATES_M`、`RELATIVE_RISK_TTC_HORIZONS_S`、
`SELECTED_RELATIVE_RISK_CANDIDATES`、`CASE_KEYS`（个案如 sp21 的时序拆解）。

### 3.9 逐文件CLI与持久化输出合同

本节用于防止“算法说明还在，但入口和消费者字段丢了”。旧默认目录刻意不保留；
所有path参数在重建时必须显式传入，不能重新硬编码历史位置。`常量驱动`表示原工具
没有CLI，其输入集合由同文件顶层常量定义。下列输出是下游会消费的稳定basename，
不是完整临时文件清单。

#### 3.9.1 构建、评估与reward工具

| 文件 | CLI（默认只列非路径值） | 必须保留的主要输出/API |
|---|---|---|
| `build_austin600_panel.py` | `--output <path>` | `austin600_scenarios.json`型ScenarioSpec列表；50×3×4、interval 15 |
| `build_crossmap600_panels.py` | 常量驱动 | 每地图`*_600_scenarios.json`、`manifest.json`；`build_panel`、`minimum_pairwise_distance` |
| `build_l12_validation_panels.py` | `--map-name Austin --output <path>` | held-out近距ScenarioSpec JSON |
| `build_interval15_difficult_pool.py` | `--source-root --output --heldout-output` | `train_interval15_difficult_pool.json`、`heldout_eval_collision_scenarios.json`、`heldout_eval_interval15_collision_scenarios.json`；`build_payload` |
| `build_collision_cache.py` | `--pretrained-model-path --cache-dir --map-name Austin --hidden-scale 4 --env-workers 12` | `build_receipt.json`；`expected_receipt`必须逐字段复核 |
| `build_heldout_hard_instrument.py` | `--actor-path --output-dir --workers 12 --hidden-scale 4 --shard-size 600 [--prepare-only]` | `design_manifest.json`、`candidate_scenarios.json`、`candidate_labels.jsonl`、`classification_summary.json`及分split场景JSON；分片API见§3.1 |
| `evaluate_scenario_panel.py` | `--model-path --scenarios --output-dir --workers 12 --start-method forkserver --hidden-scale 4 --device auto [--save-traces]` | `manifest.json`、`episodes.partial.jsonl`、`episodes.jsonl`、`summary.json`；完整schema见§3.2 |
| `postpass_reward_calculation.py` | 无CLI，纯函数模块 | `EgoInducedRearClosing`及§1.8列出的8个几何/penalty函数 |
| `validate_postpass_reward.py` | `--root --panels ... --tail-labels --output-dir` | `episode_setting_results.csv`、`setting_sweep.csv`、`summary.json`；四个被测试import的符号见§3.3 |
| `compare_postpass_formulas.py` | `--root --panels ... --output-dir` | `episodes.csv`、`summary.csv`、`summary.json`；`FORMULAS`与`replay_trace` |
| `validate_following_response_reward.py` | `--panel-dir [--episodes-jsonl] --raceline --output-dir` | `episodes.csv`、`group_summary.csv`、`summary.json`；四项因果self-check和`ready_for_training_ab` |
| `compare_risk_potential_variants.py` | `--panel-dir --run-config --actor --output-dir` | `variant_episode_metrics.csv`、`variant_cohort_summary.csv`、`variant_comparison.csv`、`target_timing.csv`、`wall_marginal.csv`、`summary.json`；`RiskVariant`与`formula_self_check` |

#### 3.9.2 结果分析器

| 文件 | CLI | 必须保留的主要输出/API |
|---|---|---|
| `analyze_baseline_reward_seed42.py` | `--panel-dir --run-config --actor --output-dir` | `episodes.csv`、`collision_cases.csv`、`collision_windows.csv`、`target_timelines.csv`、`cohort_summary.csv`、`summary.json`；transition replay与taxonomy |
| `analyze_structured_speed_exploration.py` | `--result-root --analysis-dir --diagnostic-root [--require-diagnostic]` | `panel_results.csv`、`paired_vs_baseline.csv`、`outcome_transitions.csv`、`training_windows.csv`、`baseline_late_stability.csv`、`diagnostic_scenarios.json`、`diagnostic_episode_features.csv`、`diagnostic_aligned_ol1_targets.csv`、`diagnostic_paired.csv`、`diagnostic_summary.json`、`validation_receipt.json`、`summary.json` |
| `analyze_l12_deterministic_transfer.py` | `--root` | `paired_episodes.csv`、`stratified_comparison.csv`、`validation_receipt.json`、`summary.json` |
| `analyze_l12_heldout_hard_instrument.py` | `--root` | `paired_episodes.csv`、`stratified_comparison.csv`、`validation_receipt.json`；`exact_mcnemar_detectable_net`、`stratum_rows` |
| `analyze_interval15_difficult_experiment.py` | `--root` | `checkpoint_comparison.csv`、`paired_episodes.csv`、`stratified_comparison.csv`、`validation_receipt.json` |
| `analyze_speedstd050_experiment.py` | `--root` | `checkpoint_comparison.csv`、`paired_episodes.csv`、`stratified_comparison.csv`、`training_comparison.csv`、`validation_receipt.json`；`normalized_run_config` |
| `analyze_speedstd_anneal_experiment.py` | 常量驱动 | `panel_summary.csv`、`paired_comparison.csv`、`paired_comparison.json`、`clearance_mechanism.csv`、`training_comparison.csv`、`validation_receipt.json` |
| `analyze_baseline_late_checkpoint_stability.py` | 常量驱动 | `checkpoint_curve.csv`、`paired_checkpoint_transitions.csv`、`paired_checkpoint_transitions.json`、`validation_receipt.json` |
| `analyze_temporal_late_checkpoint_stability.py` | 常量驱动 | `checkpoint_summary.csv`、`paired_by_checkpoint.csv`、`outcome_transitions.csv`、`summary.json` |
| `analyze_crossmap_bc_u30.py` | 常量驱动 | `comparison.csv`、`outcome_transitions.csv`、`validation.csv`、`summary.json` |
| `analyze_crossmap_b_t.py` | 常量驱动 | `comparison.csv`、`comparison_vs_bc.csv`、`paired_episodes.csv`、`stratified_comparison.csv`、`outcome_transitions.csv`、`validation.csv`、`summary.json` |
| `analyze_ctv2_corridor_temporal.py` | 常量驱动 | `comparison.csv`、`paired_episodes.csv`、`stratified_comparison.csv`、`validation.csv`、`summary.json` |
| `analyze_ctv2_checkpoint_eval.py` | 常量驱动 | `checkpoint_summary.csv`、`corridor_summary.csv`、`paired_vs_u30.csv`、`validation.csv`、`receipt.json`、`summary.json` |
| `analyze_ctv2_late_stability.py` | `--updates ...` | `late_stability.csv`、`summary.json` |
| `analyze_ctv2_crossmap_band.py` | `--updates ...` | `crossmap_band.csv`、`summary.json` |
| `analyze_ctv2_gap1_band.py` | `--updates ...` | `three_arm_band.csv`、`summary.json` |
| `analyze_arm_bands.py` | `--arms ...` | `arm_bands.csv`、`summary.json`；band内先逐checkpoint聚合，不能挑单点winner |
| `analyze_ctv2_loss_mechanism.py` | 常量驱动 | `collision_modes.csv`、`outcome_flows.csv`、`per_episode.csv`、`strata.csv`、`cohort_action_gate.csv`、`receipt.json`；事件对齐见§3.4 |
| `analyze_singleagent_noise_eval.py` | 常量驱动 | `comparison.csv`、`summary.json`；必须先验证`results_single.json`与`lap10.npz` |
| `analyze_postpass_geometry_cases.py` | `--root --output-dir` | `collision_episode_kinematics.csv`、`surface_gate_sweep.csv`、`relative_risk_gate_sweep.csv`、`relative_risk_candidate_episodes.csv`、`case_snapshots.csv`、`summary.json` |

#### 3.9.3 Gate、oracle与辅助工具

| 文件 | CLI（默认只列非路径值） | 必须保留的主要输出/API |
|---|---|---|
| `validate_ctv2_corridor_gate.py` | `--panel-dir --output` | 单一validation JSON；冻结公式与生产逐步mask必须零误差 |
| `sweep_corridor_gate_width.py` | `--widths 2.0 1.5 1.0 0.75 --panels ...` | stdout/汇总表；`raw_observation`、`held_mask`、`replay_panel` |
| `diagnose_ctv2_preflight_panels.py` | `--root` | `panel_episode_gate_diagnostics.csv`、`panel_gate_summary.csv`、`panel_validation.csv`、`panel_summary.json` |
| `diagnose_ctv2_ordinary_checkpoint_curve.py` | `--root --b-run --t-run --ordinary-scenarios --batch-size 32 --device auto` | `ordinary_gate_episodes.csv`、`ordinary_gate_summary.csv`、`fixed_state_checkpoint_curve.csv`、`fixed_state_exact_target_curve.csv`、`fixed_state_replay_error_by_episode.csv`、`checkpoint_sha256.csv`、`diagnostic_summary.json` |
| `diagnose_crossmap_temporal_corridor.py` | `--root --output-dir` | `joint_raceline_speed.csv`、`episode_gate_diagnostics.csv`、`gate_summary.csv`、`oracle_mixtures.csv`、`summary.json` |
| `run_u30_oracle_reachability.py` | `--actor --episodes-csv --output-dir --workers 8 --population 24 --iterations 6 --elite-count 6 [--scenario-limit N] --seed 42 --device auto` | `best_interventions.csv`、`search_attempts.csv`、`matched_control_results.csv`、`validation_receipt.json`；`ScheduledActor`与CEM搜索 |
| `evaluate_shared_oracle_library.py` | `--actor --instrument-dir --library-csv --output-dir --workers 8 --device cpu --checkpoint-scenarios 8 [--scenario-limit N]` | `baseline_replay.csv`、`candidate_results.csv`、`scenario_coverage.csv`、`coverage_summary.csv`、`validation_receipt.json` |
| `probe_shared_oracle_ranking.py` | `--actor --instrument-dir --library-results-dir --output-dir --workers 12 --pca-components 16 --seed 42 [--force-replay]` | `decision_state_features.npz`、`scenario_splits.csv`、`validation_grid.csv`、`test_metrics.csv`、`test_selections.csv`、`paired_comparisons.csv`、`validation_receipt.json` |
| `scan_u30_braking_landscape.py` | `--actor --instrument-dir --output-dir --workers 12 --checkpoint-scenarios 4 [--scenario-limit N] [--braking-deltas ...] [--requested-leads ...]` | `scan_results.csv`、`landscape_by_collision_mode.csv`、`landscape_by_fishtail.csv`、`landscape_by_interval.csv`、`landscape_summary.csv`、`validation_receipt.json` |
| `diagnose_actor_observability.py` | `--panel-dir --episodes-csv --actor --output-dir --device auto --sample-stride 2 --event-window-s 1.5 --max-epochs 200 --patience 20 --seed 42` | `episode_replay_summary.csv`、`probe_metrics.csv`、`probe_predictions.csv`、`validation_receipt.json` |
| `select_critic_lr_candidate.py` | `--root --output` | 单一JSON；`Candidate`、`CANDIDATES`、`LATE_UPDATES`、`select`；已停用，不得恢复成自动选模入口 |
| `probe_ctv2_late_behavior.py` | `--updates ...` | `late_behaviour.csv`、`late_behaviour.json` |
| `build_u30_observability_oracle_notebook.py` | 常量驱动 | `u30_observability_oracle.ipynb`；stdlib同命名空间执行cell |
| `build_u30_oracle_library_notebook.py` | 常量驱动 | `u30_oracle_library_diagnostic.ipynb`；同上 |

### 3.10 功能等价的最低标准

“可重建”在本文中严格指以下五项同时成立：

1. 同一合法输入产生相同scenario集合、分类、配对关系和核心数值（容差以§2测试为准）；
2. 非法schema、重复scenario、缺失trace、非有限数、身份不一致均fail closed；
3. CLI flag、默认非路径参数、被测试import的符号名和持久化输出字段保持兼容；
4. resume/partial语义、原子写盘、场景顺序和随机种子保持确定性；
5. 重新实现后必须用§2的断言重建回归，不得仅凭脚本运行结束宣称等价。

不要求保留旧默认目录、注释文本、局部变量、打印格式或逐行实现；这些不属于功能合同。

---

## 4. 历史回归集合中的两个非测试工具

### 4.1 `probe_hard_neighbors.py`（364 行）

standalone hard-neighbor 采样探针，**从不进入 PPO scheduler**。

```text
CLI: --cache-dir(默认 default cache) --model(默认 canonical BC) --output-dir
流程:
  load_cache            读训练 collision cache（不读 eval 结果）
  local_collision_support  局部支持定义：同 ego_idx、同 opponent raceline、
                           |Δinterval| <= 2、|Δspeed| <= 0.05、排除自身
  select_seeds          每条 opponent raceline 选局部支持最高的 1 个种子（共 3 个）
  neighbor_pool         每种子生成 interval_delta ∈ {-1,+1} × speed_delta ∈ {-0.05,0,+0.05}
                        = 6 条，去掉自身重复 → 3 种子共 18 条唯一 neighbor
  minimum_evaluation_distance  校验训练/eval 起点至少 1 m
  rollout               用冻结 BC 完整跑 3 个原种子 + 18 个 neighbor
  write_outputs         episodes.csv / selected_scenarios.json / summary.json，
                        summary 中 "pipeline_integration": false
历史结果（作为重建后的自检参考）：11/18 = 61.1% 碰撞；base 有效候选碰撞率 4.45%；
描述性富集 13.73×。**这是有意富集后的探针数字，不是无偏估计。**
```

### 4.2 `build_outcome_aware_cache.py`（202 行）

outcome-aware hard pool 的构建驱动，**只读** base 与 `boundary-aware-v1` cache，
从不改动它们。两种模式：

```text
run_inspection  训练进行中也可安全运行：只统计、不写正式 cache
run_full_build  完整构建；_reindex 重排候选索引；_filter_spec 施加 outcome/clearance 过滤
CLI: --base-cache(默认 default) --model(默认 canonical BC) + 模式开关
```

配套笔记 `OUTCOME_AWARE_MERGE_NOTES.md` 记录合并边界，重建时可省略。

---

## 5. 清理前重点保真审计

按「被复用程度 × 重写难度 × 出错代价」排序。2026-07-30清理准备中已补入：
`evaluate_scenario_panel`精确输出/续跑合同、risk变体矩阵、held-out instrument固定值、
CT-v2事件对齐、following-response完整ReplayConfig和post-pass候选选择规则。仍然只达到
行为/算法合同，不是逐行源码备份：

1. **`evaluate_scenario_panel.py`（历史优先级）** —— 当时19个 shell runner依赖它；输出schema
   是历史分析脚本的输入契约。2026-08-10后标准四图600统一走`evaluate.sh`，当前不应重建。
2. **`validate_postpass_reward.py`** —— 被测试 import 4 个符号，返回键名是契约。
3. **`compare_risk_potential_variants.py`** —— 被测试按路径加载，`VARIANTS[0]` 必须
   与生产公式逐位相等（`atol=1e-15`）。
4. **`build_heldout_hard_instrument.py`** —— 21,600 候选网格、20 block 交替划分、
   分片恢复语义，测试钉得很死，凭描述容易写偏。
5. **`analyze_ctv2_loss_mechanism.py`** —— 唯一产出「same-line 降 / off-line 升」
   机制结论的脚本，其cohort与分类定义是ANALYSIS §18.4的来源。
6. **`validate_following_response_reward.py`** —— 1,304 行里最有价值的是四项因果
   自检；重写若丢掉它们，结论就不再可信。

删除后的直接影响也必须记录：

- `train_ppo.py + ppo/` production主路径不import这些实验工具或回归测试，基础训练不会仅因
  这两个目录消失而停止；
- 清理前的`run.sh`曾直接引用`compare_postpass_formulas.py`、`build_collision_cache.py`、
  `run_structured_speed_exploration.sh`、`build_interval15_difficult_pool.py`和
  `run_interval15_difficult_validation.sh`；2026-07-30这些历史段落已从执行入口移除，
  当前根目录无`run.sh`；
- 删除回归测试源码后不能再声称当前工作树“91 tests通过”，只能引用2026-07-30清理前的历史
  结果；任何源码再改动后都需要先重建测试或建立新的回归；
- 历史`outcome_aware_hard`源码删除后，其筛选、cache和测试合同只由本文件承担；若重建，
  不要恢复对已删除测试路径的注释引用。

### 5.1 历史产物清理：panel 与 fixed pool 的恢复边界（2026-07-30 实测校正）

**当前状态更新（2026-08-10）：** 下文记录的是2026-07-30当时的可恢复性核验，不是当前资产
清单。用户已把模型eval锁定为四图各600，并授权删除`heldout_hard_v1`、旧near/hard trace、
fixed-pool wrapper与候选标签。集合等式和重建规则仍是历史实现知识，但仓库不再保存对应输入，
不得把下文“保留/已保存”解释为当前路径存在，也不得未经新授权恢复这些旧面板。

清理前的第二次实物核验否定了“hard73、hard334、near400 没有存活 trace、必须重跑
21,600 候选才能恢复”的说法。三个 panel 的**物理场景身份**当时都能由保留的 eval trace
集合精确恢复：

```text
N400 = keys(T_ctv2_near_Austin)                                  # 400
H334 = keys(control_u15_Austin) - N400                           # 734 - 400 = 334
H73  = keys(interval15_control_u0015_Austin) - Austin600 - N400  # 1073 - 600 - 400 = 73
```

三次集合等式均与原始 ScenarioSpec 列表逐场景一致，且 Austin600、H73、H334、N400
在所用差集里互不重叠。这里的 key 是
`(map, opponent_raceline, ego_idx, opp_idx, opponent_speed_scale)`；trace 名提供
raceline、ego/opp index 和 speed，目录提供 map。恢复后可以生成新的稳定
`scenario_id`，继续做同物理场景的 paired eval。

**不能**用 `interval_idx = (opp_idx - ego_idx) mod 2096` 恢复跨 raceline 的 interval。
原 hard73 中只有 37/73 满足该算式；正确恢复方法是上面的已验证集合差，或保留显式
ScenarioSpec。原始 `scenario_id`、`pool` 和 `startpoint_ordinal` 不在 trace 文件名中，
但它们不是确定性仿真或按物理 key 配对所需的输入。

fixed interval-15 训练池也不是仅存一份：对应训练 run 内保留了**有序 229 条**
`collision_scenarios.json`，与原 pool wrapper 的 `entries[*].scenario` 完全一致；
`collision_cache_info.json`还保留 selection、sampling 和来源摘要。它足以恢复训练所用
场景及顺序，但不能替代每条 `source_label/source_outcome/min_clearance` 的原始证据，
历史命令也需要把 `--fixed_collision_pool_file` 重指向一个重建或抽出的合规 wrapper。

因此保留要求必须分级，而不是把约24 MiB都称为runtime必需：

| 等级 | 内容 | 实测大小 | 是否必要 |
|---|---|---:|---|
| 继续工程最小输入 | hard73、hard334、near400显式ScenarioSpec + fixed-pool wrapper | 约0.45 MiB | 2026-07-30曾保存；2026-08-10删除 |
| provenance摘要 | `design_manifest.json` + `classification_summary.json` | 约0.07 MiB | 2026-07-30曾保存；2026-08-10删除，结论已固化 |
| 完整选择审计 | 21,600条`candidate_scenarios.json` + `candidate_labels.jsonl` | 约21.16 MiB | 2026-07-30曾保存；2026-08-10删除，不能再独立复跑selection |
| 可重算产物 | 重复的partial/final episodes、logs、stdout、重复panel规格 | 其余大部分 | 汇总结论已固化且trace保留时可删 |

完整选择审计的价值是真实的：重新分类只能生成一个**新**panel，不能证明原panel当时没有
查看treatment；但这是发表/审计级provenance要求，不是以后运行同一物理panel的必要条件。
三张跨地图panel、Austin600和near600由确定性生成合同及存活trace覆盖，不需要再为它们
保留重复的scenario JSON副本。

同时提醒：`post-trained/collision-cache/`中的训练cache和独立post-pass oracle不在本次
清理范围内，不要顺手删除。

---

## 5.3 已退役的独立 Post-pass 影子探针（核心合同已迁出，重型运行器明确退役）

该独立目录是 2026-07-23 建立的 Post-pass reward **接入 PPO 之前的独立影子验证**。
2026-07-30 已完成清理迁移：可复用数学合同进入
`scripts/screen_reward_candidate.py`，配套回归测试位于
`scripts/test_screen_reward_candidate.py`，直接运行和 `unittest discover` 均为
**55 个测试通过**。saved-episode 与 live-simulator 的旧运行器没有逐行搬运，而是按
下文保存功能重建合同并明确退役；因此可以删除原目录，但不得称为源码的逐字无损备份。

### 5.3.1 它为什么存在

Post-pass 当时是拟加入 production reward 的第五项。在花掉一次训练之前，需要先独立回答：
公式实现对不对、这个 reward 增量经 GAE 会变成多大的学习信号、门控是否会误伤成功超车。
它刻意**既不 import `ppo` 也不 import 模拟器**——它是交叉校验的 oracle 一侧，
和被测实现共用代码就等于没有校验。

### 5.3.2 代码迁移结果（21/24 个符号）

| 原符号 | 现位置 |
|---|---|
| `RewardConfig`、`PostpassState`、`ClosingGeometry`、`RewardStep` | `screen_reward_candidate.py` Part 3 |
| `oriented_rectangle_vertices`、`rear_half_vertices`、`rectangle_clearance`（自带 SAT + 点-线段最近距离，不用 `ppo.geometry`）、`signed_rear_longitudinal_gap`、`rear_half_clearance`、`ego_induced_rear_closing` | 同上 |
| `_unit_shortfall`、`bounded_negative_reward`、`postpass_reward_step` | 同上 |
| `normalized_advantages`、`clipped_ppo_policy_loss`、`mean_squared_value_loss`、`fixed_prediction_value_loss_delta` | Part 2 |
| `gae_delta_from_reward_delta` | **未逐字迁移**：等价功能由 `gae_advantage_delta` 提供，改用 `episode_starts` 约定以对齐 production rollout buffer（原版用 `episode_end`），数学相同 |
| `_huber`、`masked_follow_teacher_huber_loss` | **刻意不迁移**，见 §5.3.6 |

新增（原目录没有、但属于同一思路的延伸）：`crosscheck_geometry()` 把 256 组随机几何
交叉校验**协议本身**固化成可调用函数，`wilson_interval()` 从 `validate_shadow.py` 迁入。

### 5.3.3 已通过的 22 项不变量（重建时的检查清单）

`outputs/unit_receipt.json` 记录 `status: passed`、`test_count: 22`：

```text
几何    known_axis_aligned_gap, overlap_clearance_zero, rigid_transform_invariance,
        ego_induced_closing_direction
reward  postpass_trigger_and_sign, prepass_zero, opponent_collision_suppression,
        safe_gap_latched_clear, step_and_episode_caps,
        sequential_vectorized_episode_equivalence
GAE     gae_impulse_closed_form, gae_no_cross_episode_leak,
        gae_multi_episode_explicit_convolution
统计    wilson_interval_boundary_cases,
        accepted_selectivity_and_rejected_fallback_order
数据    npz_crc_shape_dtype_schema
损失    clipped_policy_loss_manual_case, torch_sample_std_advantage_normalization,
        fixed_prediction_value_loss_delta
teacher follow_teacher_zero_and_empty_mask,
        follow_teacher_gradient_finite_difference        <- 见 §5.3.6
交叉    candidate_module_crosscheck
```

`scripts/test_screen_reward_candidate.py` 已覆盖除 `npz_crc_*` 和两项 teacher
之外的全部条目；`sequential_vectorized_episode_equivalence` 已通过
`test_sequential_state_machine_matches_vectorized_formula` 恢复。

### 5.3.4 交叉校验的实测结果（值得记住的数字）

对当时的 `postpass_reward_calculation.py`（已随实验工具清理，规范见 §1.8）跑 **256 组**随机几何，
**5 个量的最大绝对误差全部是精确 `0.0`**：`signed_gap`、`closing`、
`counterfactual_clearance`、`current_clearance`、`bounded_reward`。

原交叉校验采样协议已按源码恢复进 `crosscheck_geometry`：
`previous_ego ~ N(0,1)^3`；`current_ego = previous_ego +
N(0,diag(0.10,0.10,0.03)^2)`；`opponent ~ N(0,1)^3`；
`basis ~ U(0,0.10)`；`used ~ U(0,0.04)`；seed **`20260723`**。
必须同时报告 `signed_gap`、`counterfactual_clearance`、`current_clearance`、
`closing`、`bounded_reward` 五项；漏掉 cap 行为不能声称复现了原协议。

**重要边界**：`validation_receipt.json` 记录 `mode: "unit"`、`saved_episodes: null`
——只有 unit 档跑过。saved-episode 重放与 live 探针**从未留下收据**。

### 5.3.4b 预注册准入阈值与 41 点门宽网格（2026-07-30 补录）

这两项原先只存在于 `validate_shadow.py` 的常量里，首轮迁移**漏掉了**，现已补进
`screen_reward_candidate.py` 并加了 5 个测试钉住。

**预注册准入阈值**（命名常量，不得静默放宽——放宽即是另一个实验）：

```text
ACCEPT_MIN_TAIL_CAPTURE     = 0.90    目标尾部捕获率下限
ACCEPT_MAX_OVERTAKE_TRIGGER = 0.20    成功超车误触率上限
ACCEPT_MAX_FOLLOW_TRIGGER   = 0.01    正常跟车误触率上限（最严的一条）
```

**关键规则：准入必须在每个 panel 上分别成立，不是看合并后的聚合值。**
原实现取跨 panel 的最大值再比阈值（`attach_panel_selectivity_guards`），因为一个设置
完全可能聚合看起来选择性很好、却在某一个 panel 上超预算。已迁移为
`per_panel_acceptance()`，并有一个测试专门构造"均值 0.105 通过、最差 panel 0.19 且
follow 0.05 不通过"的情形。

**41 点门宽网格**（`postpass_gate_sweep_grid()`）——shipped 配置就是从这里选出来的：

```text
1 个无门基线：activation_clearance=None, closing_time=None, deadband 0.10
40 个组合   ：clearance (0.05, 0.10, 0.15, 0.20, 0.30)
            × ego-induced closing time (0.25, 0.50, 0.75, 1.00)
            × closing deadband (0.05, 0.10)
最终选中    ：clearance 0.20m / closing time 0.75s / deadband 0.10m/s
```

记录它的用途是**防止重复提案**：clearance 0.30、closing time 1.00 等点位已经量过并落选。
注意这与 EXPERIMENTS §3.3 里 `validate_postpass_reward.py` 的 96 点网格是**不同的轴集**
（那一组扫 pass margin / safe gap / deadband / clear mode），两者不可混引。

### 5.3.5 两个未迁移为代码、只保留设计的部分

这两项从未产生通过收据，也依赖已经退役的面板别名；继续工程不需要保留其源码。
下面记录的是**功能重建合同**，不是仍待执行的实验。

#### A. `validate_shadow.py` 的 saved-episode 重放

CLI：

```text
--mode {unit,saved-episodes,all}      默认 unit
--root PATH
--output-dir PATH
--panels PANEL [PANEL ...]
--tail-labels PATH
--candidate-module PATH              可选；纯函数候选模块
```

历史默认 panel 是 BC、production U30、普通 U45 和 hard U45 各600条；旧路径不再是合同，
重建时必须显式传入当前 panel。固定数值为 `gamma=0.999`、
`gae_lambda=0.995`、value-loss coefficient `0.5`；准入阈值为 primary-tail
preterminal capture `>=0.90`、successful-overtake trigger `<=0.20`、
follow trigger `<=0.01`，且后两条必须逐 panel 通过。

算法顺序：

1. fail closed 检查宿主进程并尝试 `nice(10)`；
2. 读取碰撞标签，以 `(panel_label, scenario_id)` 建唯一索引，拒绝缺失/重复；
3. 对每个600-episode panel 对齐聚合JSON、逐episode JSON、标签和NPZ key；
4. 直接读NPZ ZIP成员头，验证成员集合唯一、CRC、shape、dtype和Fortran标志，不一次性
   保留完整归档；
5. 用闭合raceline累计弧长投影两车progress，处理wraparound并定位首次pass crossing；
6. 固定当前opponent pose，用前后ego pose计算rear-half clearance loss；
7. 扫41个配置：一个无clearance/TTC gate的基线，加
   `clearance={0.05,0.10,0.15,0.20,0.30}m ×
   closing_time={0.25,0.50,0.75,1.00}s ×
   deadband={0.05,0.10}m/s`；
8. 对每个配置计算trigger、step/episode cap、GAE传播、capture和guardrail，先在通过集内
   按最低overtake trigger、再按最高capture选；无通过项时只返回明确标记的高capture fallback；
9. 重放前后核对受保护输入摘要以及trace size/mtime，任何变化都失败；
10. 原子写结果，不授权production接入。

严格trace成员合同：

```text
float64: time_s, ego_pose[N,3], opp_pose[N,3]
float32: ego_lidar_360[N,360], opp_lidar_360[N,360],
         ego_raw_action[N,2], ego_executed_action[N,2],
         opp_executed_action[N,2],
         ego_measured_speed_mps[N], opp_measured_speed_mps[N]
bool:    collisions[N,2], action_applied[N], terminal_post_step[N]
```

稳定输出 basename：

```text
unit_receipt.json
validation_receipt.json
setting_sweep.csv
panel_setting_sweep.csv
selected_episode_results.csv
input_trace_manifest.csv
saved_episode_summary.json
```

历史重档意图是逐member验证 **2,400 个trace**，但实际收据只有unit模式，
`saved_episodes=null`。删除源码后失去的是对旧panel的一键重放和旧CSV/JSON布局，不是
已经记录的22项unit结果或当前独立oracle。

#### B. `live_episode_probe.py` 的四episode CPU探针

CLI：

```text
--root PATH
--output-dir PATH
--saved-summary PATH
--selected-episodes PATH
--bc-results PATH
--tail-labels PATH
--model-path PATH
```

输入前提是 saved-episode summary 已通过。选择规则固定：

1. 从“primary tail且preterminal触发”的BC行按首次触发提前量排序，取最早和中位两个；
2. 每个target配一个相同opponent raceline、相同speed、未触发且成功超车的control；
3. control先最小化环形ego起点距离，再最小化scenario ordinal，最终必须得到4个唯一场景；
4. 跨raceline时把ego起点映射到opponent raceline，再由实际opponent index恢复interval。

运行固定为CPU、Torch单线程、无CUDA。每个场景重定向到独占新目录，调用8秒
multi-agent evaluator并生成 `live_trace.npz`、`live_metrics.json`；随后比较：

```text
live outcome/steps 与 saved episode
saved vector replay 与 live vector replay
live vector replay 与逐transition状态机
tail target 必须preterminal触发
safe-overtake control 必须不触发
```

稳定汇总输出为 `live_episode_results.csv` 和 `live_episode_summary.json`。运行前、每个
episode前后都重新检查宿主空闲；源文件、模型和输入trace前后摘要必须一致；已有输出目录
拒绝覆盖。该探针从未留下收据，所以删除源码后不得声称它曾通过，也不得把本节当作
production准入证据。

**它的隔离契约比 GUIDE §3 更严，重型诊断可照抄**：

```text
只读已保存的 NPZ；单 CPU 线程、不用 CUDA
发现 training / evaluation / SUMO / 其他 audit 进程可见时拒绝运行
/proc 被沙箱过滤、无法可信列举进程时 fail closed（重档必须先由外部确认宿主空闲）
live 仿真前后校验源码哈希；拒绝覆盖已有 live-probe 目录
只往本目录 outputs/ 写
最后一句：通过探针不等于授权把 treatment 接进 PPO 管线
```

### 5.3.6 `masked_follow_teacher_huber_loss`：已实现、已单测、**刻意未接入**

这是原目录里唯一**违反用户 PPO 要求**的部分，也是唯一没有迁移的实质功能。

| 项 | 状态 |
|---|---|
| 实现 | 完整：masked Huber imitation loss + 解析 student 梯度 |
| 验证 | 2 项单测通过：`follow_teacher_zero_and_empty_mask`、`follow_teacher_gradient_finite_difference`（有限差分核对解析梯度） |
| 固定参数 | `action_scales=(0.52, 10.0)`、`component_weights=(1.0, 1.0)`、`huber_delta=0.10`；对 active mask 与权重和归一化 |
| 是否接入 | **从未**。`ppo/`、`train_ppo.py`、`run.sh` 对该目录引用数为 0；无对应训练 run；HANDOFF 模型登记无该臂 |
| 违反哪条 | GUIDE §1「单阶段 PPO；不把 imitation/蒸馏/二次微调混入 PPO 单变量实验」 |
| 不违反哪条 | **不是 shield**：它是训练期损失，不改运行时动作，也不给 actor 特权/未来信息 |

**不要复活它，有两个独立理由：**

1. **BC 作为 teacher 自相矛盾。** 原 docstring 自己写了
   `"not evidence that the BC teacher is safe in the masked states"`。masked states 就是
   同线跟车，而 BC 恰恰在这里最弱（U30 的 13 个 ego 碰撞有 8 个在 OL1）。
   在 teacher 失败的状态里模仿 teacher 是自我否定的。ANALYSIS 也已说明 oracle
   可达性用了未来碰撞时刻，**不能充当该 teacher**——所以这条线缺的是可用 teacher，
   不只是授权。
2. **它想买的能力已被合规路径拿到过。** T 臂（`temporal_global`，只改探索噪声的时间
   相关性、单阶段、无 teacher）把 B 的 8 个 OL1 目标**全部化解**，指令速度真的降到
   对手速度以下。要继续这个目标，应该改进 T 的条件化，而不是引入第二个目标。

若将来仍要做，前提是：用户显式重新授权 + 先解决 teacher 来源 + 按 GUIDE 单变量执行。
`scripts/screen_reward_candidate.py` 的 `check_compliance` 会把带
`auxiliary_objectives=("teacher",)` 的候选**直接拒绝**，这是刻意的护栏。

---

## 6. 19 个 shell runner（C 级）

统一形态：`#!/usr/bin/env bash` + `set -euo pipefail`，顶部注释写清实验动机与
单一变量，内部**逐条显式**调用（不用动态数组/模型选择器，见 GUIDE §3），
每步前置校验目录不存在或为空后再写。

四个真正启动训练的runner共享同一fresh-start合同，除表中单变量外不得改变：

```text
initializer=pretrained/end2race.pth, critic=privilege_gru, map=Austin
n_envs=16, env_workers=12, seed=42, ordinary_startpoint_count=50
collision cache=default, n_steps=6400, batch_size=12800
actor_epochs=2, critic_epochs=5
gru/head/critic LR=3e-6/3e-5/3e-4
steering_latent_std=0.03, speed_physical_std=0.15
gamma=0.999, gae_lambda=0.995, clip_range=0.20
```

训练runner的唯一变量：

| runner | 单变量 |
|---|---|
| `run_structured_speed_exploration.sh` | 四个独立fresh run：`baseline`、`conditional_white`、`global_temporal`、`conditional_temporal`；其余参数完全相同 |
| `run_ctv2_corridor_temporal.sh` | `speed_exploration_mode=corridor_temporal`，30 updates |
| `run_ctv2_gap1_corridor_temporal.sh` | 在上一行基础上仅`corridor_gate_front_gap_m=1.0` |
| `run_ctv2_45u_fourmap.sh` | 与CT-v2相同但`num_updates=45` |
| `run_offline_fast_reweight_arm.sh` | 与CT-v2相同，额外`ordinary_offline_fast_fraction=0.6`（可由环境变量覆写） |

评估runner统一要求：每个actor/panel组合调用一次`evaluate_scenario_panel.py`，
明确传入model、scenario、独立output、workers和`--save-traces`；不得让不同actor共享
trace alias。checkpoint band必须按预注册update全部评完，分析器读取完整band而不是自动
选择最低碰撞点。

| runner | 调用链 |
|---|---|
| `run_structured_speed_exploration.sh` (189) | 逐步独立、条件白噪声、全局时间相关、条件时间相关四臂fresh训练；四臂评U10/U20/U30；另评逐步独立臂U24–U29；U30生成隔离actor alias后跑诊断panel；最后执行两阶段分析 |
| `run_ctv2_corridor_temporal.sh` (240) | 先做gate零误差校验，再训CT-v2 U30，评Austin600/near400/三张跨图并分析 |
| `run_ctv2_gap1_corridor_temporal.sh` (137) | 训练唯一差异为corridor front gap 1.0而非2.0 |
| `run_ctv2_45u_fourmap.sh` (343) | CT-v2从BC训45 updates；只评U45的Austin600和三张跨图 |
| `run_offline_fast_reweight_arm.sh` (217) | CT-v2加ordinary off-line-fast重权，默认fraction 0.6；仅训练和完整性校验 |
| `run_ctv2_checkpoint_curve.sh` (135) | CT-v2评U10/U15/U20/U25；与已有U30合并分析Austin600和near400曲线 |
| `run_ctv2_late_checkpoint_stability.sh` (131) | CT-v2评U26–U29 Austin600；与U30合成band，目的为稳定性而非选点 |
| `run_ctv2_crossmap_band.sh` (124) | CT-v2 U27/U28/U29 × 三张跨图 |
| `run_ctv2_gap1_band_eval.sh` (132) | gap1 U27–U30 × Austin600 + 三张跨图 |
| `run_ctv2_offline_preflight.sh` (66) | 逐步独立/全局时间相关速度探索分别在Austin、near和collision-pool诊断panel评估，再做gate预飞 |
| `run_arm_band_eval.sh` (159) | 参数为`RUN_DIR RESULT_ROOT ALIAS_PREFIX UPDATE...`；每个update评Austin600、near400及三张跨图，全部保存trace |
| `run_baseline_late_checkpoint_stability.sh` (36) | 逐步独立速度探索U21–U29逐点评Austin600+near400 |
| `run_temporal_late_checkpoint_stability.sh` (29) | 全局时间相关速度探索U24–U29逐点评Austin600+near400，再分析同update两臂 |
| `run_crossmap_bc_u30_eval.sh` (30) | 构建三张600 panel；BC/U30各评三图；统一BC-vs-U30分析 |
| `run_crossmap_t_eval.sh` (38) | 给全局时间相关速度探索U30独立alias，评三张跨图，再与production U30和BC配对分析 |
| `run_l12_heldout_hard_validation.sh` (60) | Control-U15/L12-U15各评held-out hard与near-miss，共4组；alias必须隔离trace |
| `run_interval15_difficult_validation.sh` (57) | 先构建pool与Austin panel；control/treatment各评U5/U10/U15 × Austin/hard/near，再分析 |
| `run_speedstd050_validation.sh` (46) | std0.50 treatment评U5/U10/U15 × Austin/hard/near；control读取已完成对应panel |
| `run_singleagent_noise_eval.sh` (35) | BC/U30 × 三张图 × masking 0.1/0.2/0.3，单次10圈，然后统一分析 |

---

## 7. 重建后的自检

重建后应使用正式解释器，在新建的回归目录执行`unittest discover`。目录名称由重建任务
决定；本文件不保留已删除目录的路径。

`unittest`预期（2026-07-30 实测）：**91 tests、85 passed、6 skipped**。6 个 skip 全部是**条件
跳过而非失败**，且分别由两个不同原因触发——重建后必须仍然是这 6 个，不能多也不能少：

```text
5 个  test_postpass_episode_replay.PostPassRealEpisodeReplayTests.*
      skip 理由 "Austin600 terminal traces are unavailable"
      —— eval_results/pretrained_end2race/Austin/multiagents/traces 不存在时整类跳过（§2.3）
1 个  test_outcome_aware_hard.RealReplaySmokeTests.test_boundary_labels_match_shipped_binary_outcomes
      skip 理由 "set END2RACE_RUN_SIM=1 to run the tiny real-BC smoke"
      —— 需要模拟器的慢速档，靠环境变量 END2RACE_RUN_SIM=1 开启（§2.15）
```

即历史测试集合里有**两种独立的跳过闸门**：trace fixture 存在性，以及
`END2RACE_RUN_SIM` 环境变量。重建时两个闸门都要保留，否则默认套件会试图跑模拟器。

随后还必须用pytest执行§2开头列出的两个文件，预期**6 passed**。因此完整静态回归合同
是97个测试定义，而不是只看unittest的91项。

分模块复核 reward 侧（这 6 个模块单独跑是 **34 tests、29 passed、5 skipped**，
与全套的 91 不是一回事，不要混引）：

```bash
$PY -m unittest test_postpass test_postpass_episode_replay \
  test_postpass_reward_integration test_risk_longitudinal_ab \
  test_compare_postpass_formulas test_compare_risk_potential_variants
```

另外三项只能靠人工核对：`ppo/` 生产模块未被改动；分析脚本仍然只读；
新脚本没有把 boundary/gate 逻辑散落进 `ppo/algorithm.py`、`reward.py`、
`environment.py` 或 eval 脚本（复杂度必须留在 cache builder / 离线脚本里）。

---

## 8. Regime可分性与梯度冲突无训练审计（2026-08-05）

### 8.1 边界与调用链

本轮六个一次性工具只冻结并重放production U30与前向走廊门控时间相关速度噪声U30，没有
更新参数、创建actor或写入任何`post-trained/`目录。它们在结果固化后已按用户要求删除；
以下是历史调用链和功能重建合同，不表示当前源码仍有这些入口：

```text
audit_regime_gradient_conflict.py
-> extract_regime_representations.py
-> run_regime_counterfactuals.py
-> analyze_regime_separability.py
-> analyze_regime_gradient_conflict.py
-> build_regime_gradient_audit_notebook.py
```

正式解释器统一为`/home/haowei/miniconda3/envs/end2race/bin/python`。六个脚本当时默认共享
`analysis_results/regime_gradient_conflict_audit_20260805`作为审计根；可通过各自CLI改根，
但一次审计内不能混用不同根。该审计根及其notebook、NPZ、JSON和partial续跑记录已删除。
结果判决已固化到`ANALYSIS.md` §23；本节只记录已删除实现的功能重建合同。

### 8.2 `audit_regime_gradient_conflict.py`：cohort与数据质量门

默认输入是production/treatment各自U30评测根与actor checkpoint，默认事件窗`1.5s`；
`--stage`目前只有`stage0`，这是刻意的fail-closed接口，不允许跳过预审直接运行后续阶段。

算法顺序：

1. 逐图、逐episode核对production/treatment trace key、scenario字段、finite动作行与结果；
2. 三张跨地图production若缺`results_multi.json`，从完整trace按当前ego collision scope重建；
3. 按既有paired alignment规则确定事件终点：新造碰撞用treatment碰撞时刻，修复碰撞用
   production碰撞时刻，其余用共同可比结束时刻，再向前取窗口；
4. 构造互斥S/O/N cohort，并检查当前固定分母必须为`54/69/59`；
5. 写`cohorts.json`、`cohorts.csv`、`data_quality.json`及必要的`legacy_rechecks/`凭据。

S是ordinary面板same-line production ego碰撞而treatment修复；O是ordinary面板off-line、
speed scale至少0.7、production成功而treatment新碰撞或丢超车；N是near400中production成功
而treatment新碰撞或丢超车。任何trace缺失、窗口为空、schema不一致、非finite动作或固定分母
漂移都必须失败，不能静默缩小样本。

### 8.3 `extract_regime_representations.py`：严格序列重放

输入`cohorts.json`与两份冻结actor；默认设备`auto`，`--maximum_replay_error=5e-5`。
每条episode从起点按当前actor调用顺序重建361D observation，并逐步更新GRU hidden；不能把
事件窗当成零hidden的新序列。输出为`representations/<cohort>_<map>_<episode>.npz`与
`replay_summary.json`，至少包含窗口内observation、production/treatment 1680D hidden、两臂
动作和anchor trace动作。raw与executed anchor动作任一最大误差超过阈值即失败。

### 8.4 `run_regime_counterfactuals.py`：闭环动作收益标签

默认`--workers=4 --device=cuda --maximum_tasks=0`；0表示执行全部任务。每个episode固定运行
五个闭环分支：`anchor`、`beneficial_full`、`beneficial_speed`、`beneficial_steering`、
`decelerate_0p15`。替换只发生在已对齐事件窗内；最后一项是在adverse speed上减`0.15m/s`。
碰撞口径固定为ego scope。

任务结果逐条追加到`counterfactuals/episodes.partial.jsonl`，以
`cohort|map|episode|variant`去重并支持中断续跑；重复task id、变体不全或结果不可解析均失败。
全部完成后原子写`counterfactuals/results.json`。只有outcome rank严格上升才标记动作有利：
`ego collision < follow < overtake`。若anchor重跑outcome不等于原评测，该episode设置
`counterfactual_valid=false`并从表征标签分析排除，不选择性重跑；speed标签还要求成功臂
窗口平均speed确实更低。`maximum_tasks`只用于受控烟测，未覆盖全部变体时不得生成正式结论。

### 8.5 `analyze_regime_separability.py`：A审计

默认参数为`--repeats=10 --maximum_epochs=600 --patience=60 --learning_rate=0.01
--weight_decay=0.01 --device=auto`，默认输出`separability_results.json`；敏感性运行通过
`--weight_decay`与`--output_name`生成独立文件，不能覆盖主结果。

特征取paired窗口最后一个可执行状态：361D observation或production 1680D hidden；
`cohort_only`是显式基线。标签是闭环counterfactual得到的“持续减速有利”“完整production
动作有利”“steering修正有利/必要”，不是same-line/off-line类别。每次按
`map + ego startpoint`做60/20/20 grouped split，训练集统计量标准化，validation负责early
stop和balanced-accuracy threshold，test只报告一次。最多尝试500个确定性group shuffle，
任一split缺一类或凑不齐全部repeat时该组合写入`skipped`，不能退化成逐episode随机切分。

### 8.6 `analyze_regime_gradient_conflict.py`：B审计

默认`--folds=5 --subset=all --device=auto`；`--subset=validated_full`只保留anchor稳定且
full-window动作替换确实提高outcome的episode。两次输出分别为
`gradient_conflict_results.json`与`gradient_conflict_results_validated_full.json`。

所有梯度都在同一个冻结production actor上计算。每个episode先从起点无梯度重放hidden，
进入事件窗后才启用BPTT；目标是提高successful-arm动作相对adverse-arm动作的Gaussian
log-probability margin。Steering使用当前squashed Gaussian合同（bound`0.52`、latent
std`0.03`），speed使用physical Gaussian std`0.15`；episode先各自均值，再对cohort等权。
脚本严格检查actor可训练参数名，只接受当前GRU与output-head合同；参数布局漂移即失败。

输出必须包含S/O/N动作方向、GRU与output-head cosine、最后steering/speed行cosine、逐参数层
梯度范数，以及按`map + ego startpoint`哈希的group-fold-delete敏感性。这里是当前冻结actor
上的原始诊断梯度，不是缺少optimizer state时不可恢复的历史Adam/PPO更新。

### 8.7 `build_regime_gradient_audit_notebook.py`：可复核汇总

默认读取上述JSON并生成`regime_gradient_conflict_audit.ipynb`。环境不要求`nbformat`、
Jupyter或IPython：builder使用标准库构造notebook，在单一共享namespace中顺序执行所有代码
cell，将stdout/异常写回cell output后再原子保存。任何代码cell异常都终止构建；最后一格必须
输出`VALIDATION_STATUS=PASS`和C的触发状态，不能只生成未执行notebook。

### 8.8 验证、删除后果与重建边界

2026-08-05实测验证：六个脚本全部通过`py_compile`；182份representation的最大anchor
raw/executed误差均为`1.91e-6`；counterfactual覆盖`182 * 5 = 910`个唯一任务；notebook五个
代码cell全部执行并以`VALIDATION_STATUS=PASS`结束。没有为这组一次性审计另建unit test；
上述确定性分母、严格replay误差、任务覆盖和notebook断言是当前端到端验证门。

2026-08-05按用户要求已删除`representations/`、counterfactual partial/result、各分析JSON、
notebook和六个一次性脚本，只保留一份Markdown结果记录并同步本组三份权威文档。删除导致
任意重新分层、重新画图、逐episode复核、续跑和逐字节复现能力不可恢复；本节只支持功能级
重建，不能保证得到相同split或文件字节。已经固化到`ANALYSIS.md` §23的判决不因产物删除而
失效。模型checkpoint、production评测trace和collision cache不属于这组产物，未被删除。

## 9. `diagnose_ppo_regime_gradients.py`：fresh first-step PPO无更新诊断

### 9.1 用途与调用关系

该脚本只用于从冻结actor/critic采集一条fresh production-contract rollout，并测量optimizer
执行前的advantage加权分层梯度；不训练、不保存新checkpoint。调用链为：

```text
diagnose_ppo_regime_gradients.py
-> train_ppo.build_model + CentralScheduleSubprocVecEnv
-> DiagnosticRolloutBuffer -> policy.evaluate_actor_actions
-> 分层PPO loss与torch.autograd.grad
```

必需CLI为`--actor-path --critic-path --scenario-root --output-dir --cell-id
--checkpoint-update --speed-exploration-mode`；非路径默认值为`n_envs=16`、`n_steps=6400`、
`batch_size=12800`、`seed=42`、`minimum_regime_transitions=256`、`device=cuda`。探索模式只接受
production baseline或前向走廊门控时间相关速度噪声。

### 9.2 核心算法

1. 读取保存的collision/ordinary ScenarioSpec队列，按原production env rank合同创建vector env；
2. 用指定actor构建`privilege_gru` PPO并严格载入对应critic，固定production超参数；
3. fresh采集102400个transition，stage regime/scenario/episode和已有探索遥测；
4. 计算GAE/return，并用collection-equivalent recurrent replay验证old/new log-prob；
5. 按production minibatch sequencer重放，优势只在有效slot上标准化；
6. same-line、offline-fast、offline-slow分组loss统一用`sum/N_total`，验证其梯度逐参数重构总loss；
7. 对GRU与output head分别报告cosine、各组/合成范数、cancellation ratio和虚拟对称投影；
8. 不调用optimizer，不修改任何模型tensor。

固定输出basename为`transitions.npz`、`results.json`、`manifest.json`。NPZ字段包括reward/value/
return/advantage、old/new log-prob、20D privileged state、action、episode start、三类identity与
outcome，以及完整探索遥测。结果JSON保存aggregate和逐minibatch有效样本/梯度指标；manifest
只记录模型身份、参数、场景数和完成状态。输出目录非空时fail closed；场景ID重复、字段非法、
ratio或梯度分解超容差、transition覆盖不全均失败。

### 9.3 当前验证

2026-08-06最小baseline重复烟测在相同checkpoint、seed和探索下18个NPZ array逐元素完全一致，
聚合梯度完全一致；走廊模式烟测记录danger gate `22.75%`、temporal active `26.51%`，同block
残差最大漂移`3.22e-6`，collection-equivalent log-ratio误差为0。

正式K=`1/10/20/30`三格网格共12个cell全部完成；每格102400 transition、8个minibatch，全部
通过样本门、ratio exact fallback和`2e-5`梯度分解门，没有optimizer step。结果判决和完整数字
见`ANALYSIS.md` §24：真实PPO冲突未跨checkpoint稳定，当前不实现梯度投影训练。一次性脚本和
NPZ在核心结果固化后删除；本节保留功能级重建合同，但任意重新按scenario/outcome/credit字段
分层和逐字节复现能力不可恢复。

### 9.4 Role-mix描述性复核

不需要新脚本：读取每个cell的`transitions.npz`，按逻辑env奇偶划分collision/ordinary role，
按`episode_id`聚合逐步reward，只纳入`episode_outcome_code>=0`的完整episode；分别计算两臂
role mean return差。break-even只在`Δcollision>0`且`Δordinary<0`时计算
`-Δordinary / (Δcollision - Δordinary)`。按每个env完成episode顺序对齐并比较
`scenario_index`，必须同时报告共同位置数和真正identity匹配数，禁止把相同候选池称为严格配对。

训练轨迹复核直接读取两臂`metrics.jsonl`中同update的`mean_collision_episode_return`和
`mean_ordinary_episode_return`，不得把不同update拼成重复样本。当前K1/K10/K20/K30 fresh结果
及U27--U30/U42--U45历史结果已固化到`ANALYSIS.md` §25；它们否定了稳定break-even，未授权
修改50/50 role合同。该复核只是episode-return描述，不替代transition advantage或GAE审计。

## 10. `average_actor_checkpoints.py`：固定四点actor等权平均

### 10.1 用途与调用关系

该脚本只构造兼容actor checkpoint的**固定四源等权平均**，不训练、不选择checkpoint、不读取
评估结果。Round Z0调用链为：

```text
average_actor_checkpoints.py
-> torch.load四个actor state dict
-> average_state_dicts
-> 原子torch.save
-> End2Race(hidden_scale=4).load_state_dict(strict=True)
-> model_manifest.json
```

CLI：`--source-paths`必须且恰好4个；`--output-path`必需；`--hidden-scale=4`；
`--evaluation-alias=CTv2_U42_U45_EQUAL_AVG`。输出目录必须尚不存在，脚本创建
`actor.pth`、同目录的`<evaluation-alias>.pth`硬链接和`model_manifest.json`；alias用于避免不同
actor同stem导致trace串臂，不能当作第二个模型。

### 10.2 核心算法与不变量

1. 四个source path必须唯一、存在，且每份state dict恰好12 keys；key名称和顺序完全相同；
2. 每个value必须是tensor；shape与dtype逐源一致；所有浮点source tensor必须finite；
3. 浮点tensor严格按CLI source顺序转float64累加、除以4，再cast回reference dtype；
4. 非浮点tensor不平均，四源必须`torch.equal`后clone，否则fail closed；
5. 输出使用同目录临时文件加`os.replace`原子落盘；随后建立hardlink alias；
6. 从落盘文件重新读取，检查12 keys、finite，并在fresh `End2Race(hidden_scale)`上strict-load；
7. manifest原子写入source相对路径/SHA/0.25权重、算法、输出与alias SHA、hardlink inode判断、
   各source相对L2距离、git commit和worktree status；
8. 任一异常删除本次output/alias/manifest，并只在目录为空时删除新目录；绝不覆盖或修改source。

`relative_l2_distance(left, right)`只累计浮点tensor，计算
`sqrt(sum((left-right)^2) / sum(right^2))`，reference零范数时失败。`sha256_file`按1 MiB分块。

### 10.3 测试与当前验证

`scripts/test_average_actor_checkpoints.py`包含3个unittest：

- `test_float_tensors_use_equal_average_and_preserve_dtype`：四个float32向量得到精确等权均值且
  dtype不变；
- `test_non_floating_tensor_must_match`：任一int64值不同必须raise；
- `test_key_order_shape_and_source_count_fail_closed`：source数不是4、key/order不同、shape不同
  分别fail closed。

2026-08-08实测：两个文件通过`py_compile`，3/3 unittest通过；真实U42--U45源构造后通过
12-key/finite/fresh strict-load，alias与canonical输出同inode且SHA一致。相对四源L2为
`1.0771e-4 / 6.1200e-5 / 6.1609e-5 / 1.1707e-4`。评估结论不属于本文件，见
`ANALYSIS.md` §33。

### 10.4 删除后果

删除脚本不会使已生成actor失效，但会失去按相同float64顺序、非浮点一致性、原子写盘和
strict-load合同重新构造其他固定四点平均的现成入口。功能重建必须保留§10.1--§10.3全部CLI、
运算顺序、失败条件、manifest字段和三项测试；不能用直接float32相加或`torch.optim.swa_utils`
近似替代后声称字节等价。

## 11. `analyze_bc_anchor_gate_a.py`：Gate A全量质量门与cohort冻结

### 11.1 用途与输入输出

该脚本不运行actor、不训练；默认只消费预先冻结的development panel/split manifest与
BC、U42--U45五个fresh eval package，验证Gate A证据后一次性写出机器报告和28条
共识cohort panel。Round Z3最小扩展允许对事先冻结且尚未打开的validation split运行
同一质量门，但只输出stable collision cohort。CLI为：

```text
--panel
--split-manifest
--evaluation-root
--report
--cohort-panel
--workers=12
--split-name={development,validation}        默认 development
--collision-only-validation                 默认 false；必须与validation同用
```

两个输出任一已存在即fail closed，不能根据第一次结果覆盖或改cohort。cohort panel保持原
development panel顺序；机器报告保存五个actor的路径/身份/aggregate、质量门、单点与共识
计数、全部scenario key、分层、六条准入判决和下一动作。

### 11.2 验证与筛选合同

1. development panel key唯一，完整key与起点集合必须逐项等于冻结split manifest；
2. development/validation起点零交集，但不打开validation actor结果；
3. 五个manifest必须分别绑定canonical BC、固定U42--U45、同一development panel，且明确
   fresh deterministic CUDA、ego scope、8秒、hidden scale 4、保存trace、0 error；
4. 每臂result、trace和panel key集合精确相等，episode七字段identity逐项等于panel；
5. 每个NPZ用`allow_pickle=False`打开，所有数组首维相等、全部有限，LiDAR/action/pose/
   collision shape符合schema，`len(trace)=steps+1`且`time_s`严格递增；
6. `collisions[:,0/1]`与三个typed marker逐行一致，typed ego marker与result outcome一致；
7. `terminal_post_step`仅末行true，`action_applied`仅末行false；
8. 先从BC筛`overtake + raceline0/2`，再定义U44单点回归；主cohort固定要求U44回归且
   U42--U45至少3/4回归，不得看结果后退回单点集合；
9. 按raceline、全部speed及U44 collision/lost-overtake分层，逐条计算§4.2六个Gate门；
10. 只有全部质量合同通过才原子写盘；科学Gate失败仍如实写fail报告，但不得启动Gate B。

collision-only validation模式保留相同`BC-safe overtake`和U42--U45至少3/4回归定义，
但cohort只保留U44 ego collision。样本门固定为至少4条、3个ego startpoint且
raceline0/2均非空；不满足时写inconclusive，不允许退回U44单点cohort。

### 11.3 当前端到端验证

2026-08-08脚本通过`py_compile`并在真实Gate A包上完成3,590条trace全量检查：五臂各718条，
0 error、无partial、所有key/identity/finite/marker/terminal合同通过。BC-safe候选308条，U44
单点46条，共识28条；21个起点、raceline0/2=`20/8`、collision/lost=`19/9`，六条Gate A门
全通过。当前未另建合成unit test；真实五臂端到端检查和输出拒绝覆盖是本节点验证。实验判决
与证据边界见`ANALYSIS.md` §34，当前接续状态见`HANDOFF.md` §9.15。

Round Z3模式后续在真实validation包上检查750条trace，得到7条、6起点的stable
collision cohort，raceline0/2=`6/1`，所有质量与样本门通过。该扩展未改变默认
development输出；用既有3,590条trace作回归检查时，默认cohort、分层和六条criteria
逐项与旧report相等。

## 12. `run_bc_anchor_gate_b.py`：持久化双actor反事实branch runner

### 12.1 用途、CLI与调用关系

该脚本实现预注册Gate B，不能用于训练。默认仍只允许development双分层；
Round Z3增加一个显式collision-only validation模式。CLI全部显式给出：

```text
--cohort-panel
--development-panel
--gate-a-report
--gate-a-evaluation-root
--bc-model-path
--u44-model-path
--output-dir
--workers=12
--hidden-scale=4
--sim-duration=8.0
--prepare-only
--collision-only-validation                 默认 false
```

调用链：

```text
build_plan
  -> intervention_window（首次collision或首次最小OBB clearance）
  -> select_controls（同raceline/speed、circular index距离、SHA tie-break、无放回）
worker_initializer
  -> 每个forkserver CUDA worker一次加载BC与U44
run_stage(branch0)
  -> validate_branch0硬门
  -> run_stage(full_bc / bc_steering / bc_speed；collision-only只full_bc)
  -> validate_branch_traces
  -> analyze_gate_b
```

`--prepare-only`必须先运行：它在任何branch结果产生前原子写`gate_b_plan.json`，冻结输入身份、
28条cohort窗口、28条controls及匹配关系；后续命令只加载并核对该plan，不重算。正式输出为每
branch的`results.json`、`traces/<episode_key>.npz`，以及`branch0_contract.json`和
`gate_b_report.json`。已有最终report时拒绝覆盖。
若同raceline/speed、无放回control池根本不存在完整匹配，`build_plan`在写plan前fail
closed，因此不会留下伪冻结plan或运行branch0。

### 12.2 窗口、hidden与动作合同

1. U44和BC各维护自己的GRU hidden；每一步都先在同一实际branch observation与previous
   measured speed上分别forward，再选执行动作；因此接管前二者读取同一U44 prefix，接管后都
   沿反事实branch递归，窗口后U44可从正确hidden恢复；
2. collision窗口为`[first_ego_collision_step-150, first_ego_collision_step)`；L/control先用
   `get_vertices(length=0.58,width=0.31)`和`rectangle_clearance`逐帧重算OBB距离，取第一次全局
   minimum，窗口为`[argmin-100,argmin+50)`；两端按episode动作行截断，terminal不执行；
3. 窗口少于50动作步的cohort明确列为excluded；任一C/L低于8直接停止。当前28条均eligible；
4. controls只从development中BC/U44 outcome均为overtake的262条候选选取；同raceline/speed，
   circular ego waypoint-index距离升序，`SHA256(scenario_key)`破平，无放回；
5. 每步先产生`u44_raw_action`和`bc_raw_action`。branch0执行U44；full执行BC二维；steering-only
   执行`BC steer + U44 speed`；speed-only相反。`action_source_code=-1/0/1/2/3`分别表示terminal、
   U44、BC-steering、BC-speed、full-BC；
6. 标准16字段外，trace增加两份raw action、`intervention_active`和`action_source_code`，足以
   审计hidden路径上的teacher/student mean与实际动作；
7. ego scope下只因ego collision终止，opponent-wall仍单列marker；环境、planner、progress与
   首帧speed合同逐项复制当前evaluator。

### 12.3 恢复、效率与失败关闭

四个stage均用`episodes.partial.jsonl`逐episode flush并按key恢复；result与trace key未齐时不写
最终文件。orphan trace只会被同一冻结task的确定性重跑替换。一个Pool跨四个stage复用：12个
worker各只加载一次两份actor，不像通用panel evaluator那样每个scenario重新加载；这只改变
执行拓扑，不改变224次episode语义、分支数或验收门。

branch0必须先完成全部56条并逐字段验证：ego raw/executed严格0误差；opponent action、pose、
speed、LiDAR为`rtol=0,atol=1e-6`；boolean marker、outcome、首次collision、长度与terminal严格
相同。失败则抛错且不调度干预。三组干预结束后，`validate_branch_traces`再验证全部224条的
key、finite、数组对齐、marker、terminal、冻结窗口、source code、selected raw和clip后executed
action；任何异常都不写Gate report。

### 12.4 当前端到端验证与删除后果

2026-08-08先用C/L/control各1条烟测，所有比较字段最大误差均为0；正式branch0随后56/56同样
全部为0。四branch共224条result/NPZ通过完整质量门；full/steering/speed中分别6/5/7条因干预
后较早collision而窗口未执行满，均保留为真实outcome。脚本通过`py_compile`；没有另建合成
unit test，三类烟测、56条精确replay和224条真实端到端合同是当前验证。

删除脚本不会改变已记录Gate B科学失败，但会失去已验证的双hidden branch engine、确定性
control匹配、断点恢复和逐动作审计入口。若未来独立预注册动作库或first-action preference，
可复用该engine；不能删除验证逻辑后从保存U44 trace直接伪造反事实branch。结果见
`ANALYSIS.md` §35，当前停止边界见`HANDOFF.md` §9.16。

Round Z3使用`--collision-only-validation --prepare-only`消费上节生成的7条cohort，在任何branch前
发现`r0/s0.60` source/control=`2/0`、`r0/s0.85=3/2`，因分层control support不足抛错。
没有生成plan、branch0、full-BC或report；这是预注册inconclusive分支，不是runner失败或
teacher outcome。详见`ANALYSIS.md` §37与`HANDOFF.md` §9.18。

## 13. `run_counterfactual_action_gate.py`：固定局部动作库branch与hidden排序Gate

### 13.1 用途、CLI与调用关系

该脚本只做Austin训练侧零更新机制筛选，不训练或修改actor。CLI为：

```text
--development-panel
--bc-results
--u44-results
--u44-trace-root
--u44-model-path
--output-dir
--workers=20
--hidden-scale=4
--sim-duration=8.0
--prepare-only
```

调用链：

```text
build_plan -> source_event -> action_gate_plan.json
worker_initializer（每个persistent CUDA worker一次加载U44）
run_tasks(noop) -> 全456条compare_replay硬门 + hidden snapshot
run_tasks(early/late * 12 residual actions)
validate_results -> existence_metrics
ActionScorer五折startpoint外推（仅existence通过的prefix）
-> action_gate_report.json
```

`--prepare-only`在任何branch前冻结输入hash、456条cohort、5-fold、两个prefix、12个动作和全部
准入线。正式执行若发现已有最终report则拒绝覆盖；partial JSONL逐条flush，重启按`result_key`
恢复，orphan trace由同一确定性任务覆盖。

### 13.2 Cohort、动作与trace合同

cohort从既有Gate A development的718条BC/U44配对结果按outcome定义，事件前不足150步剔除；
固定109 inherited collision、46 created collision、13 lost-overtake、63 inherited-follow诊断、
225 safe controls。collision事件取首次typed ego marker，其余取逐帧OBB clearance第一次全局
minimum；early/late分别执行`[event-150,event-100)`、`[event-100,event-50)`，碰撞可提前终止。

动作始终是当前反事实trajectory上U44 mean的residual：steering `+/-0.02,+/-0.04 rad`、speed
`+/-0.5,+/-1.0 m/s`及四个`+/-0.02`与`+/-0.5`组合；窗口外恢复U44，steering仍clip至
`[-0.52,0.52]`。每条no-op在两个prefix起点保存更新当前observation后的1680D hidden。

为降低10,944条candidate branch的I/O，候选NPZ只保存逐步time、U44/selected/executed action、
collision与typed marker、action/terminal/intervention标记；outcome、progress、speed和proximity
保存在result。no-op运行时在内存保留完整LiDAR/action/pose/speed/marker数组，与既有U44 source
逐字段比较后只写compact trace和hidden。这样不重复保存456份完整LiDAR，但精确重放证据不减少。

### 13.3 验证、排序与删除后果

no-op要求raw/executed action严格0误差，opponent action、pose/speed和双360D LiDAR为
`rtol=0,atol=1e-6`，boolean、outcome和terminal严格相同；任一失败不调度候选。候选验证
result/plan/trace key、finite、数组对齐、terminal、冻结窗口、residual raw action及clip后执行
动作。

oracle label只接受最终无ego collision的overtake；多解按归一化动作范数、final relative
progress和名字确定。`ActionScorer`读取hidden，经`Linear(1680,128)+ReLU`后与二维归一化动作
拼接，再用`Linear(130,64)+ReLU+Linear(64,1)`对13个动作打分；fold按ego startpoint SHA分组，
fold-local标准化，固定Adam、100 epochs，无模型选择。每折固定动作baseline只从train folds按
`target success - 5*control harm`选取。数值准入和prefix分流以预注册§23为唯一权威。

删除脚本会失去宽cohort动作库重放、compact branch审计、hidden snapshot和action-conditioned
五折排序能力；Gate B的BC结论不受影响，但不能只靠其56条BC branch替代本Gate。

### 13.4 当前端到端验证

2026-08-08先用1条no-op和1条candidate烟测，no-op最大误差0、candidate准确执行50步；正式
456条no-op的全部比较字段最大误差仍为0。10,944条candidate与456条no-op合计11,400条compact
NPZ、8,185,913行全部通过key、finite、对齐、terminal、窗口、residual和executed-action合同，
无partial文件。首次12-worker运行在完成部分任务后按result key恢复，并仅抑制重复RK4 warning；
随后按机器20个CPU核心改为20个persistent worker继续，已完成结果不重算，科学合同不变。

两个prefix的action existence均通过，但五折ActionScorer均未通过全部预注册门；最终报告原子
写入并在同一固定seed/硬件上重跑head，聚合与首次结果逐项相同。科学判决见`ANALYSIS.md` §36，
当前停止边界见`HANDOFF.md` §9.17。

## 14. `run_action_response_representation_gate.py`：50步actor-visible历史动作响应Gate

### 14.1 用途、CLI与依赖

该脚本是Round Z4-A的零actor-update离线probe。现有`run_counterfactual_action_gate.py`只能
读取单点frozen hidden并学习top-1 preference，不能表达“同一真实动作分支标签下增加可训练
历史编码器、与容量匹配hidden control比较”，因此本轮新建这一窄脚本；它不运行环境、branch
或PPO，也不保存可部署模型。

CLI全部显式：

```text
--plan                         Round Z2 action_gate_plan.json
--branch0-results              456条noop结果
--candidate-results            early/late各12动作，共10,944条结果
--hidden-root                  456个含early/late 1680D hidden的NPZ目录
--u44-trace-root               Round Z2 source U44完整trace目录
--u44-model-path               冻结U44 actor；路径和SHA必须与plan一致
--output-report                唯一输出JSON；必须事先不存在
--device=cuda                  固定；非CUDA或CUDA不可用直接失败
--nested-operating-point       默认false；运行Round Z5 5x4 nested calibration
--nested-outer-z4-seeds        默认false；仅与上一flag同用，outer model恢复Z4 seeds
```

调用/数据流：

```text
validate_inputs
  -> 456 task/branch0、10,944 candidate、13动作、5 folds、五层分母和U44身份硬校验
load_actor_features
  -> source trace最近50步ego LiDAR + previous measured speed
  -> frozen U44 sigmoid LiDAR transform + frozen speed_mlp
  -> history[456,50,420] + saved late hidden[456,1680]
build_labels
  -> noop与late 12-action真实终局 -> labels[456,13]
5-fold grouped OOF
  -> FrozenHiddenOutcomeModel
  -> HistoryOutcomeModel
  -> fold-train-only fixed action baseline
summarize -> criteria -> atomic output report

--nested-operating-point时：
outer 5-fold
  -> outer-train内部按独立hash生成4个inner folds
  -> inner-OOF概率穷举lambda与noop margin
  -> 在5/225双control率预算内冻结操作点
  -> outer-train重训，outer-test只应用一次
  -> nested summary + 诊断性outer-OOF Pareto
```

脚本只导入`model.End2Race`和`utils.atomic_write_json`，没有其他脚本依赖；也没有符号被别的
脚本导入。输出父目录可新建，但已有最终report严格拒绝覆盖。没有partial/resume：单次计算约
数秒，异常时不写最终JSON；成功时只原子写一个report，不写checkpoint或日志副本。

### 14.2 输入、feature和泄漏合同

1. plan必须恰有456个唯一task，分层固定为inherited collision 109、created collision 46、
   lost-overtake 13、inherited-follow 63、safe-control 225；
2. branch0 keys必须精确等于456个task且summary=`456/0 errors`；candidate keys必须精确等于
   `456 x early/late x 12`且summary=`10944/0 errors`，不能只给late子集；
3. fold逐task重算`SHA256("counterfactual-action-fold-v1|ego_idx") mod 5`，并检查任意两折
   ego startpoint集合不交；late start index必须至少50；
4. 每条history indices为`late_start-49 ... late_start`共50步；previous speed读取
   `ego_measured_speed_mps[indices-1]`。source trace只读取`ego_lidar_360`和该speed字段；
5. 加载plan登记的U44 12-key actor并核对SHA。LiDAR逐维按actor当前`k`计算
   `(-1/(1+exp(-k*x))+1)*2`，speed走冻结`speed_mlp(1->60+ReLU)`，拼成420D；
6. hidden从Round Z2 branch0保存NPZ的`late`键读取。feature必须`50x420`、hidden必须1680D，
   所有输入finite；
7. 标签把`ego-opp/ego-wall`合并为collision class 0，follow为1，overtake为2。它只进入loss，
   不进入history feature。noop动作编码`[0,0]`，其余按plan动作顺序编码
   `[steering_delta/0.02, speed_delta/0.5]`。

这套构造不读pose、opponent LiDAR/speed、stratum、距event时间或branch outcome作为feature。
报告`quality_validation`必须写出tensor shape、finite、fold startpoint数、字段白名单、空的
forbidden-fields列表、分层和三类标签数。

### 14.3 模型、训练和动作选择算法

每折train/test由plan fold直接决定。hidden逐维均值/标准差只在train scenarios计算；420D
feature统计量在train scenarios的所有50步上计算；std `<1e-6`置1。test只使用train统计量。

- `FrozenHiddenOutcomeModel`：hidden `1680->192+ReLU`，扩到13动作后拼2D action，再经
  `194->64+ReLU->3`；
- `HistoryOutcomeModel`：hidden `1680->128+ReLU`；history逐步`420->64+ReLU`后GRU64取末
  hidden；两路拼192D，再与2D action走完全相同`194->64+ReLU->3` head。

两者固定Adam、learning rate `1e-3`、weight decay `1e-4`、100 epochs、scenario batch 64。
fold初始化seed=`5200+fold`，shuffle CPU generator seed=`5300+fold`；CUDA deterministic打开、
benchmark关闭。每个scenario的13动作同时构成batch，三分类cross entropy权重只由训练折
全部scenario-action标签计算`N/(3*N_class)`；空类fail closed。无early stopping、validation
选择或超参入口。

测试动作分数固定`P(overtake)-5*P(collision)`，`argmax`天然按plan动作顺序破精确平局。
grouped fixed baseline也只用train fold：每个单一动作统计所有非safe-control场景的overtake
次数，减去5倍safe-control非overtake次数；同分按动作顺序。它不看test outcome。

行为summary逐task回查真实branch result，记录四个非control层success、三主层total、safe
control新collision/overtake loss、动作与终局映射、相对noop的removed/created、lost/gained及
双侧exact binomial p。报告同时保存全13动作三分类OOF accuracy、每折分母/类别权重/final
train loss/fixed action和执行命令/HEAD/worktree snapshot。

准入七个boolean固定为：inherited/created/lost至少`11/7/4`，safe-control新collision/loss至多
`4/11`，target total分别至少超过grouped fixed与frozen-hidden control 9。全部为true才写
`pass_to_independent_validation`，否则写`fail_close_tested_representation_instance`。

### 14.4 当前验证、结果与删除后果

脚本通过`py_compile`、CLI smoke和`git diff --check`。真实运行验证456条/70 startpoints、
5,928标签、22,800个history rows全部finite且折间无startpoint泄漏；每折三类均非空，OOF
预测完整。独立只读重算逐episode选择映射、真实branch outcome、四层success、control harm和
全部criteria一致。

当前treatment为`68/32/2`、target 102、controls `13/21`；frozen-hidden control为
`68/33/3`、104、`13/20`；fixed为`45/33/1`、79、`5/5`。因此只通过collision层和fixed margin，
失败lost、两个control门和frozen-hidden margin。科学判决与边界见`ANALYSIS.md` §38和
`HANDOFF.md` §9.19。

删除脚本不会改变已形成的失败判决，但会失去从Round Z2 source trace重建actor-visible
50步feature、容量匹配history/control五折训练和逐episode动作选择复核入口。由于没有保存模型，
删除后不能只凭report恢复各fold参数；不过已完成的数值、机制边界和停止规则已固化，不需要为
复现历史判决而重跑。它不是通用representation框架，不能扩展参数后继续扫描。

### 14.5 Round Z5 nested operating-point扩展

该模式复用14.1--14.3全部输入与模型，不调用环境。inner fold固定为
`SHA256("action-response-inner-v1|ego_idx) mod 4`，每个outer-train内部startpoint互斥。
`lambda=(0,0.25,0.5,1,2,3,5,8,12,20,32)`；每个lambda的tau候选为inner-OOF上
`best_nonnoop score - noop score`全部唯一值及infinity。分数为
`P(overtake)-lambda*P(collision)`，margin达到tau才干预。

inner可行性用整数交叉乘法要求两项safe-control harm率均不高于`5/225`；可行点依次按target
高、collision harm低、overtake loss低、干预数低、lambda高、tau高选择。inner model seed为
`6200+10*outer+inner`、shuffle为`6300+10*outer+inner`。默认outer seed为`6400+outer`与
`6500+outer`；显式`--nested-outer-z4-seeds`改为原Z4的`5200+outer / 5300+outer`，其他内容
不变。后一flag未配前一flag时立即抛错。

输出在原Z4字段之外增加：每个inner/outer分母、类别数、loss、inner选择的lambda/tau/metrics、
outer-test metrics、nested逐episode summary、基于outer-OOF概率的非判决Pareto、matched `5/5`
事后最优诊断、frozen-vs-fixed及history-vs-frozen配对exact统计。infinity阈值写字符串，正式
JSON不存在非有限常量。已有output仍拒绝覆盖，没有checkpoint或partial文件。

2026-08-08默认独立outer seeds得到frozen target 69、controls `1/3`，history 63、`6/8`；
exact-Z4-seed复核得到frozen 66、`3/4`，history 55、`7/7`；fixed两次严格复现79、`5/5`。
两份report均通过标准JSON解析和逐episode raw branch outcome独立重算。默认Z4非nested路径另用
新临时输出回归，frozen/history/fixed的success、harm与动作计数逐项等于既有104/102/79报告。

诊断outer-OOF全局选点在独立seed能读到`84 @ 3/5`，但nested只有69；exact Z4 seeds的诊断
matched-harm最高为`72 @ 4/5`。正式判决只使用nested，见`ANALYSIS.md` §39与`HANDOFF.md`
§9.20。删除脚本还会失去nested calibration、经验Pareto和exact-seed复核能力；不得从诊断
frontier反推一个新阈值用于训练。

## 15. Prefix-reset snapshot no-op工程门

### 15.1 `scripts/run_prefix_reset_snapshot_gate.py`

职责：检验当前F110、LatticePlanner opponent、PPO wrapper/reward与U44 actor/critic recurrent
state能否在冻结交互prefix处完整序列化和逐位恢复。它是prefix-reset的机械必要条件工具，不是
训练入口，不修改`ppo/`生产模块，也不实现current-network burn-in、rollout buffer或GAE。

CLI：

```text
--gate-b-plan PATH   默认Gate B冻结plan；只取role=cohort的28条任务
--actor-path PATH    默认U44 12-key actor
--critic-path PATH   默认同一U44 privilege-GRU critic
--output-dir PATH    默认prefix-reset snapshot Gate评测根
--workers INT        默认4；spawn进程，每worker只加载一次actor/critic
--hidden-scale INT   默认4
--prepare-only       只冻结plan，不创建snapshot/trace/report
```

没有别的脚本导入其符号。直接依赖`model.End2Race`、`ppo.env.make_environment`、
`ppo.policy.PrivilegeGRUCritic`、`ppo.scenarios.EpisodeResetSpec`以及`utils`的原子JSON/NPZ和起点
构造函数。输出目录已有不同plan、完整report或任一partial snapshot/trace目录时fail closed；
没有resume，也不覆盖结果。

### 15.2 Plan与固定控制

`snapshot_gate_plan.json`在任何后缀执行前原子写入。输入必须恰为Gate A由U42--U45至少3/4
共识得到、Gate B冻结的28条development cohort：collision 19、lost-overtake 9、21个ego
startpoint、opponent raceline0/2=`20/8`。Task按episode key稳定排序；plan记录输入模型身份、
任务/窗口、snapshot字段与零误差准入合同。已有plan只有逐JSON相等才复用。

固定环境为Austin、seed42、`privileged=True`、`corridor_temporal` gate；网络为U44 actor和
同update critic，CUDA float32、eval mode、初始actor/critic hidden全零、deterministic mean
action。每条在`window.start_index`、当前observation尚未被网络消费前保存。只有这一套固定
合同，没有CPU fallback、模型/seed/window选择或阈值参数。

### 15.3 Snapshot schema与恢复顺序

顶层pickle为`schema_version/environment/observation/actor_hidden/critic_hidden`。Environment
部分按以下顺序显式捕获和恢复：

1. 两台RaceCar：7D state、opponent poses、accel、steer-angle velocity、steering buffer、
   in-collision和各自scan RNG bit-generator state；
2. Simulator：agent poses、collisions、collision indices；F110Env：pose/collision镜像、
   near-start/toggle、lap/time、start pose/rotation、render/current observation和Gym
   `OrderEnforcing._has_reset`；
3. LatticePlannerOpponentController：trajectory、tracker count、speed scale；planner动态字段
   `best_traj/best_traj_ref_v/best_traj_idx/prev_traj_local/prev_opp_pose/goal_grid/state_i/state_t/
   step_all_cost/all_costs/last_s/step`，selection只允许`None`或`np.argmin`；PurePursuit保存
   `prev_error`与可选`nearest_dist`，rendered waypoints非空会fail closed；
4. PPOTransitionReward的previous ego/opponent progress、relative position、两种collision
   latch、scenario ID、previous risk potential和current clearances；
5. End2RaceGymnasiumEnv的raw observation、previous speed、elapsed/current spec/reset RNG、
   全部episode reward/return/clearance/risk累计量和corridor current gate；
6. 当前observation与消费前actor/critic hidden。

Austin scan map、raceline projector、planner配置/静态waypoints和网络参数在同worker内保持只读，
不复制。每份dict用`pickle.HIGHEST_PROTOCOL`往返后才恢复；snapshot bytes在成功任务结束时写入。

### 15.4 后缀执行与trace合同

原后缀先从snapshot点继续到真实terminal/truncation；之后在同一environment加载snapshot与
hidden，再跑恢复后缀。Actor和PrivilegeGRUCritic每步分别消费381D observation中的
`360 LiDAR + previous speed`和20D privileged尾部；steering按`[-0.52,0.52]`执行。Runner只为
审计临时包装opponent controller action，记录真实executed action，调用后立即删除实例包装。

每侧稳定写一个压缩NPZ，字段为：pre/post 381D observation、双360D LiDAR、两车7D state、
两车steering buffer、actor/critic前后hidden、actor raw/executed action、opponent action、critic
value、reward及四分量、collision、terminated/truncated、action-applied与terminal-post-step。
全部数组首维对齐且finite；最后恰有一条`terminal_post_step=true/action_applied=false`，此前
全部action applied。Episode summary另含后缀action数、absolute terminal step、首次ego collision
step、outcome、terminated/truncated与episode return。

比较要求两侧shape一致；bool用`np.array_equal`，所有数值转float64计算最大绝对误差，任何字段
非零即该task失败。28条全部完成且全部task/pass、全部trace字段零误差、全部episode summary相同、
28份snapshot与两侧各28份trace齐全，才写`pass_snapshot_mechanical_gate`；否则写
`fail_stop_prefix_reset_snapshot`。异常发生在最终report前时不产生伪完成JSON。

### 15.5 当前运行验证与删除后果

脚本通过`py_compile`、CLI smoke与`git diff --check`。真实4-worker CUDA运行完成28/28，独立
读取全部28份pickle与56份NPZ后再次逐字段`np.array_equal`；所有连续字段最大误差0、所有bool
逐元素相同，episode summary完全相等。Prefix 0--701步，26/28大于0、中位345.5；完整轨迹
16,385步中可跳过prefix 9,589步（58.5%）；后缀99--800步、中位154.5。该比例不是wall-clock
benchmark。

`collision/lost=19/9`是Gate A来源标签；本轮当前环境原后缀为14次ego collision、6次overtake、
8次follow，只用于恢复覆盖审计。Runner固定deterministic mean action，不采样训练期exploration；
因此snapshot schema尚未覆盖或裁决structured exploration RNG/residual block，必须由Z6-B另行固定
“恢复还是重采样”的语义并验证log-probability重建。

删除脚本会失去对当前第三方F110对象布局、LatticePlanner动态字段和wrapper/reward状态的可执行
精确恢复实现，也失去自动生成/比较快照证据的能力；Markdown足以保留判决，但不能逐行重建
pickle/trace。脚本通过只准入下一道current-network burn-in与GAE语义Gate，不代表prefix-reset
PPO有效。若第三方F110、planner字段或wrapper状态改变，现有schema应fail closed并重新审计，
不能静默忽略新可变字段。

## 16. Prefix-reset current-network与PPO语义Gate

### 16.1 `scripts/run_prefix_reset_semantics_gate.py`

该脚本实现Z6-B与后续Z6-BR，不训练模型。它读取Z6-A冻结plan/snapshot/original suffix，重新
收集prefix observation，使用真实相邻U45 actor/critic检查参数改变后的burn-in，并直接调用当前
policy/buffer代码验证snapshot boundary、GAE和likelihood。它不导入Z6-A脚本符号，也不修改环境
snapshot schema。

CLI：

```text
--snapshot-dir PATH                 默认Z6-A结果根
--source-actor-path PATH            默认U44 actor
--source-critic-path PATH           默认U44 privilege-GRU critic
--current-actor-path PATH           默认同轨迹U45 actor
--current-critic-path PATH          默认同轨迹U45 critic
--output-dir PATH                   默认Z6-B结果根
--workers INT                       默认4
--hidden-scale INT                  默认4
--prepare-only                      只冻结plan
--residual-adjudication-only        不跑simulator，只执行Z6-BR并写独立裁决
```

普通入口先验证Z6-A verdict、28 task/21 startpoint、prefix总数9,589、每条snapshot与source trace、
四个checkpoint身份；`semantics_gate_plan.json`必须在任何prefix重放前原子写入。已有不同plan、
完整report或partial `prefixes/`都fail closed，不resume、不覆盖。CUDA是硬要求。

### 16.2 Prefix收集与burn-in

4个spawn worker各加载U44/U45 actor与critic一次。每条按Z6-A相同Austin seed42、scenario与
window，从reset开始以U44 deterministic mean action运行；每个动作前保存381D observation，数组
长度必须严格等于prefix step。Window observation与U44逐步actor/critic hidden逐位比较Z6-A
snapshot。

`reference_burn_in`从零hidden逐step、batch-size-one消费prefix；`sequence_burn_in`把同一
`[P,381]`一次交给GRU。两路GRU实际只用前361D；P20只用于critic最后一个value late-fusion。
每条prefix NPZ固定保存：prefix/window observation、source/current两路hidden、current fast
hidden、current reference/fast action和value。全部float32、finite；`P=0`返回严格零hidden。

事前fast容差`5e-5`同时检查actor hidden、critic hidden、window mean action和value。U44 source
fast失败只能拒绝该快速路径；逐步reference精确时不关闭语义。U45是固定真实current-network
checkpoint，不做内存扰动或optimizer step。

### 16.3 `End2RaceRolloutBuffer.recurrent_resets`

`ppo/rollout.py`增加一个默认等价、opt-in的reset side channel：

```text
stage_recurrent_resets(bool[n_envs])
```

`reset()`分配`[buffer_size,n_envs]` bool数组和一次性stage槽；`add()`若没有stage就复制传入的
`episode_start`，因此普通production路径不变。若stage存在则保存独立mask并立即清空，shape不等
`(n_envs,)`时报错。`get()`仍用原`episode_starts`调用`create_sequencers`，但返回给actor/critic
replay的`RecurrentRolloutBufferSamples.episode_starts`改为pad后的`recurrent_resets`。GAE始终读取
原boundary mask。

合成buffer固定6行：boundary `[T,F,F,T,F,T]`、recurrent reset全false，三段初始actor hidden
`1/2/3`、critic `11/12/13`。Reward/value同时覆盖terminated、已加入`gamma*V_terminal`的timeout
和rollout-cut bootstrap；独立float32反向递推用`gamma=.999, lambda=.995`比较advantage/return。
第二个默认buffer完全不stage reset，要求两个mask逐元素相同。

### 16.4 Baseline/corridor likelihood

脚本直接构造当前`End2RaceGRUPolicy`并加载U45 critic。Baseline使用28条window首transition；
corridor使用28条各51步suffix observation，首步exploration start=true、之后false，所有actor/
critic recurrent reset始终false。Corridor首步强制正gate、以后false，用来覆盖50步hold与第51步
release，不把该合成gate解释为真实场景曝光率。

每步保存实际sampled action、old log-prob、speed log-std、active、block id和反算residual；按
sequence-major重排后调用当前`evaluate_actor_actions(..., collection_equivalent=True)`。准入要求
`max |log_ratio|`及`max |ratio-1| <= 5e-5`；普通batched replay也报告但不替代exact fallback。
探索start与recurrent reset分开传入，并将首步actor hidden与直接actor forward比较为0误差。

### 16.5 Z6-BR测量裁决与产物

首次Z6-B的strict `semantics_gate_report.json`不会被覆盖。它唯一失败量是
`standard_residual=(action_speed-mean_speed)/std`在相同block内不逐位相同；该字段只参与
`_exploration_statistics` telemetry，不参与distribution replay。

`--residual-adjudication-only`绕过plan重建，重新读取冻结plan和strict report，核对U45身份后只
重跑51步policy。扩展后的log-prob Gate同时记录内部`_temporal_speed_noise`；独立
`residual_measurement_adjudication.json`要求：strict report除该残差条件外全部通过、内部noise
前50步误差0、反算误差不高于原`5e-5`、likelihood不高于原`5e-5`、第51步release且revisit产生
新residual。它不运行environment。

正式运行结果：28/28 source exact，9,589 prefix rows；U45 fast最大误差`5.84e-6`，GAE/return
`1.49e-8/0`，baseline/corridor exact log-ratio均0。Strict report因反算residual`3.18885e-6 > 0`
给出fail；Z6-BR内部noise误差0，全部裁决条件通过。删除脚本会失去current-network prefix重建、
buffer mask/GAE fixture、两种exploration likelihood与telemetry measurement audit的可执行入口；
Markdown不能恢复逐task prefix数组或重新检验未来checkpoint。

## 17. Prefix-reset训练密度Gate、持久化panel与正式训练接入

### 17.1 `post-trained/panels/prefix_reset_consensus_v1`

这是训练输入而非可清理分析产物。Manifest固定schema 1、panel id、28个唯一episode key、来源
snapshot与prefix文件SHA。每个任务复制Z6-A已经验证的snapshot，并只保存finite float32
`prefix_observations[P,381]`与`window_observation[381]`；28项合计9,589行。Loader逐项校验hash、
shape、key集合、window与snapshot observation逐位相同，任一漂移立即拒绝。

### 17.2 `ppo/env.py`的opt-in prefix reset

`load_prefix_reset_panel()`执行上述持久化校验。`End2RaceGymnasiumEnv.restore_prefix_snapshot()`
恢复F110、scan、LatticePlanner/PurePursuit、reward/wrapper与scenario身份，并再次检查恢复后的
381D observation。Worker命令`reset_prefix`只在显式启用时存在。

`CentralScheduleSubprocVecEnv`新增默认关闭的`prefix_reset_inputs`与`prefix_reset_interval`。
启用后只计collision-role reset，每第N次从独立`SeedSequence([seed,0x50524658])`队列取一个
prefix；完整遍历28项后才重排。Prefix不消费479 collision queue；非prefix collision与ordinary
调度完全沿用production。Scheduler state同时保存prefix RNG、queue、位置和collision reset计数。
默认未启用时调用链与旧production相同。

### 17.3 `ppo/rollout.py`的current-network burn-in与硬门

Prefix boundary开始前，collector以当前内存actor/critic从零hidden逐步、batch-size-one消费
保存的actor-visible prefix；不保存旧U44 hidden，也不把prefix行写入rollout buffer。第一条后缀
transition使用`episode_starts=true`切GAE/sequence、`recurrent_resets=false`保留burn-in state，
exploration则按新episode重采样。

普通训练仍用SB3 collector；只有env显式启用prefix时使用自定义collector。Timeout bootstrap、
rollout-cut final value、真实terminal与GAE保持production语义。Buffer replay返回float32
`recurrent_resets`；actor warmup按真实episode boundary切序列，但GRU reset使用独立mask。
每个formal update记录prefix reset/key/transition/window/burn-in统计，并强制审计完整buffer：
collection-equivalent最大ratio误差须`<=5e-5`，普通batched最大`|ratio-1|`须`<0.02`，否则在
optimizer前fail closed。

### 17.4 `train_ppo.py` CLI

新增两个成对参数，默认值保持关闭：

```text
--prefix_reset_panel PATH
--prefix_reset_interval INT
```

必须同时给出非空panel与正interval；只给一个即拒绝。Run config保存两者。正式Z6-F固定interval
3，其他训练参数不因该入口改变。

### 17.5 `scripts/run_prefix_reset_density_gate.py`

脚本先从Z6-A/B输入构建持久化panel，再用隔离子进程收集baseline/treatment各102,400行，无
optimizer step。它完整核对role、pool membership、reset history、prefix覆盖/密度、GAE、exact与
batched likelihood、参数摘要和墙钟。Z6-CR adjudication模式重新收集full treatment，报告完整
batched误差分布，并对相同8个minibatch分别执行普通/exact dry actor backward，比较累计梯度；
两路都不step参数。

正式结果为：Z6-C原12项通过11项，唯一batched最大ratio偏差`0.010755 > 0.01`，保留machine
fail；Z6-CR的clip fraction 0、mean KL `9.05e-10`、gradient cosine `0.9999838`、相对L2差
`0.005706`，通过独立因果裁决。实现前两次preflight分别在构造器签名与真实replay bool dtype处
失败，均发生在科学运行前；后者另以1,680行真实buffer回归确认返回float32且GAE误差`<=1e-6`。

删除本节脚本会失去完整no-update集成与干梯度裁决入口；删除持久化panel会失去正式训练输入，
不能从Markdown无损重建snapshot bytes。删除env/rollout/train接入则普通production仍可运行，
但Z6-F无法按已验证语义执行。

### 17.6 Z6-F正式训练与评测运行验证

`run.sh`只含一条显式Z6-F训练命令。训练根保存原始`checkpoints/actor_uNNNN.pth`和
`critic_uNNNN.pt`；完成审计后为U1--U30建立规范`update<N>/actor.pth,critic.pt` hardlink，inode
相同，不复制权重。`trajectory_manifest.json`记录固定训练变量、完整性、评测数量与最终层级判决。

训练实测31行metrics、30对checkpoint、5,166条episode记录全部finite；每个actor严格12-key，
每update完成16/16 actor step。Pre-update batched ratio最大`0.019095 < 0.02`，exact全0；
prefix/window fraction最低`8.43%/4.26%`。训练后KL允许按production `target_kl=None`自然记录，
出现U14 mean `0.43747`等尖峰，未事后增加early stop。

U27--U30使用`evaluate_scenario_panel.py`在四图固定600 panel上串行运行，每个调用8 workers、CUDA、
deterministic、ego scope、8秒、保存traces。16个调用共9,600 result/trace，全部actor/panel身份匹配、
0 error、key集合相等；所有NPZ数组对齐finite、时间严格递增、末行唯一terminal且无action，ego-opp/
ego-wall marker与result outcome严格一致。Runner现有每task加载actor实现未改写；checkpoint级调用
串行，避免多评测进程争用GPU。

正式U30为四图`103/1522`，逐图BC线通过但`collision<40`失败；U27--U30只有U28/U30通过且不
连续。本工具链验证的是可执行的prefix采样与标准PPO训练，不包含部署时snapshot、prefix panel或
critic；部署actor仍为原12-key结构。删除run/panel会分别失去训练复现记录/训练输入，不能仅凭
Markdown无损恢复；评测产物清理前核心数字与证据边界已固化到`ANALYSIS.md` §43。

## 18. Collision-only BC overlap-supported独立panel与Gate

### 18.1 `scripts/build_collision_only_anchor_overlap_panel.py`

该脚本只构造Z7持久化ScenarioSpec输入，不读取任何actor outcome。CLI只有必需的
`--output-dir PATH`和固定默认`--startpoint-count 40`；后者不是可扫参数，非40立即拒绝。它读取
heldout candidate与标准Austin600，取二者ego startpoint并集作为exact exclusion set；在Austin
raceline1的2,096个唯一waypoint上按
`SHA256("collision-only-anchor-overlap-v2|Austin|" + offset)`排序搜索offset，取第一个能产生40个
唯一、零交集循环等progress起点的集合。

每个起点生成`raceline0/2 × interval 8/10/12/15 × speed 0.45:0.05:0.85`，`opp_idx`必须由
`get_opponent_startpoint`生成。输出固定为`full_scenarios.json`和`panel_manifest.json`；任一已存在
则拒绝覆盖。Manifest保存输入panel SHA、算法、offset、40起点、维度、计数和零交集检查。
正式Z7得到offset 1629、2,880个唯一key。删除脚本会失去从历史输入确定性重建panel的入口；
持久化panel本身仍是后续重放的直接输入。

### 18.2 `scripts/analyze_collision_only_anchor_overlap_gate.py`

CLI固定包含：

```text
--stage {candidates,final}
--panel PATH
--panel-manifest PATH
--full-evaluation-root PATH
--candidate-panel PATH
--candidate-manifest PATH
--candidate-evaluation-root PATH
--cohort-panel PATH
--report PATH
--workers INT                         默认12
```

`candidates`阶段要求BC/U44都用panel id `collision_only_anchor_overlap_v2_full`完成2,880条CUDA、
ego-scope、8秒、trace评估；逐result核对identity/finite/aggregate，逐NPZ核对array、shape、time、
terminal/action-applied与typed collision。候选唯一规则为`BC=overtake`且
`U44 in {ego-opp,ego-wall}`，在U42/U43/U45 outcome前原子冻结panel和manifest，已有输出拒绝覆盖。

`final`阶段用manifest SHA确认BC/U44未漂移，并要求U42/U43/U45在候选panel id
`collision_only_anchor_overlap_v2_candidate`上完成同一质量合同。稳定source为U42--U45至少3/4
非overtake；source/control窗口直接复用`run_bc_anchor_gate_b.intervention_window`的U44 first
collision前150步与minimum-OBB `[-100,+50)`，少于50步排除。

每个`(raceline,speed)`中source按
`SHA256("collision-only-anchor-overlap-v2|source|" + scenario_key)`排序，取前
`min(eligible source,eligible safe control)`条；control按循环ego index距离、key SHA破平，无放回。
V0要求source至少12、起点至少8、两条raceline各至少2、exact controls等数和所有质量检查。输出
固定cohort panel与V0 report，包含完整support table、source→control映射及下游所需的
`cohort_definition.consensus/collision_scenario_keys`；样本门失败写inconclusive且不得运行branch。

### 18.3 `scripts/run_bc_anchor_gate_b.py`的Z7最小扩展

原`--collision-only-validation`模式保持不变，但V0 experiment id白名单新增
`bc_collision_only_anchor_overlap_v2`，plan/report沿用输入experiment id。若V0带
`control_support.expected_source_to_control`，runner用原same-raceline/speed nearest无放回算法
重建后必须逐映射相等，否则在branch0前停止。旧Z3 report没有该字段，行为不变。

正式调用仍只有`branch0`与`full_bc`。branch0的raw/executed action要求严格0误差，其余连续字段
`<=1e-6`、boolean/outcome/steps/first collision/terminal完全相同；不通过不运行teacher。
full-BC在窗口同时替换steering/speed，后续恢复U44。判据固定rescue `>=ceil(.5N)`、rescued
overtake `>=ceil(.8R)`、至少2起点与每个有至少2 source的raceline救1条、controls新collision和
overtake loss各`<=floor(.05N)`。

Z7实测BC/U44全面板、三个59条候选actor及两个82条branch共6,101次fresh episode。V0为41
source/41 control通过；branch0最大误差全0；full-BC rescue 18/41、control collision/loss各4/41，
V1失败。删除本节工具会失去outcome-unseen panel构造、分阶段等价筛选、support-overlap匹配和
branch逐步合同的可执行入口；核心科学判决已固化到`ANALYSIS.md` §44。

## 19. `run_gru_action_response_aux_gate.py`：GRU-changing paired action-response auxiliary Gate

### 19.1 用途、CLI与数据流

该脚本零新仿真、无PPO，首次让action-response auxiliary loss反传进原actor GRU。它导入既有
Z4/Z5脚本的input validator、真实branch record选择、分层summary和paired exact函数，避免重写
456 task/13 action/五层分母语义。CLI为：

```text
--plan PATH
--branch0-results PATH
--candidate-results PATH
--hidden-root PATH
--u44-trace-root PATH
--u44-model-path PATH
--output-report PATH
--device cuda                         默认cuda且固定
--epochs 10                           默认10且非10拒绝
```

已有report拒绝覆盖；没有partial/resume或checkpoint。成功只原子写一个report，模型全部驻内存
并删除。固定seed bases为7100/8100。

### 19.2 Exact recurrent输入

每条task读取late start、U44完整source trace和保存hidden。先用场景初速`0.9×`初始化previous
speed，之后用前一trace row measured speed；从零hidden按episode、按step、batch-size-one运行U44
到`late_start-50`，保存detached initial hidden。最后50行LiDAR经冻结`k`变换、speed经冻结
`speed_mlp`形成`50×420`，再逐步送原GRU。重建late hidden和source raw action都要求最大误差
`<=1e-5`，否则训练前停止。

首次preflight用batched burn-in，误差`0.021586/0.009939`被该门拦截且未写report。正式修复改为
严格batch-size-one，456条两项最大误差均0。不得恢复批量近似作为fallback。

### 19.3 Target、模型与优化

动作顺序固定noop加plan的12 residual。每个state-action标签为：最终ego collision二值，以及
`final_relative_position_m(candidate)-final_relative_position_m(noop)`。Control模型只用保存
frozen hidden；treatment复制U44 GRU并设train mode。两者共享同构head：
`1680->192 ReLU`，拼2D归一化动作，`194->64 ReLU->2`。

Loss为`0.5×class-balanced BCE + 0.5×SmoothL1`；progress只用train split mean/std。GRU/head
LR=`3e-6/3e-4`，Adam weight decay `1e-4`，batch64，10 epochs，gradient norm clip1.0。`k`、
speed MLP和actor output layer不在optimizer。每fold报告GRU参数相对L2、test hidden相对L2及冻结
output head下steering/speed functional drift；执行门要求前两者分别`>=1e-7/>=1e-5`。

### 19.4 Rotating无泄漏操作点

对test fold f，calibration为`(f+1)%5`，其余三fold训练。每个startpoint一次test、一次calibration、
三次train；test labels不进入模型或操作点。Score为预测progress delta减`lambda*P(collision)`；
lambda网格`0,.25,.5,1,2,4,8,16`，tau为calibration全部best-nonnoop/noop margin加infinity。
Calibration两类safe-control harm各不得超过`floor(.05*N)`；tie-break依次target高、harm低、干预
少、lambda/tau大。冻结lambda/tau后只对test应用一次，不用test把harm顶到预算。

每seed最终要求GRU真正改变、controls两类harm各`<=5/225`、target `>=88`、相对frozen `>=+9`，
且I/C/L分别`>=11/7/4`；两个seed都过才准入独立validation。

### 19.5 正式结果与删除边界

两seed均真实改变GRU。Seed7100 frozen/treatment为`58 @ 12/13`与`50 @ 14/15`；target paired
`2/10,p=.0386`。Seed8100为`61 @ 19/20`与`58 @ 12/13`；target `11/14,p=.690`，control两类
harm均`1/8,p=.0391`，但绝对预算仍失败。Lost恢复为0/1。结果关闭当前具体paired
action-response auxiliary，不保存模型或
运行PPO；不外推所有改变GRU表征的辅助监督。

删除脚本会失去exact recurrent input构造、原GRU direct auxiliary update、rotating
train/calibration/test和两seed功能漂移审计入口；核心结果与适用边界已写入`ANALYSIS.md` §45。

## 20. Z9 collision-cost Constrained PPO

### 20.1 单一入口与目标去重

`scripts/run_constrained_ppo.py`包含prepare、preflight和formal三种固定阶段。prepare只写机器冻结
合同；preflight收集一个canonical BC完整rollout并在任何actor update前做机械与OOF Gate；只有
report全部通过，formal才允许创建一次新的30-update目录。

`ConstrainedEnd2RacePPO.collect_rollouts()`复用生产scheduler、环境、actor和reward critic。
每步从`info`并行保存`ego_collision`、`reward_collision`、done、ego waypoint与scenario identity；
rollout完成后强制验证`reward_collision == -2 * cost`，再以
`training_reward = environment_reward - reward_collision`重算reward GAE。这样first collision只由
constraint计价，不在reward和cost中重复出现。

### 20.2 Cost buffer、critic与dual

`ConstrainedRolloutBuffer`在原buffer之外保存cost、cost value、cost advantage和cost return，并在
原recurrent minibatch相同的padding/排列上暴露cost侧张量。Cost critic只读381D observation尾部
20个P20特权量，结构为20-120-30-1；warm-up与正式训练均使用MSE、LR3e-4、grad clip0.5。

Formal actor不新增第二套action或head：先计算`A_reward - lambda*A_cost`，然后由原PPO代码执行
一次标准化、ratio/clip和当前GRU/output-head optimizer。每个rollout之后以已完成episode的
collision率执行`lambda <- clip(lambda + .5*(rate-.10),0,20)`；每个update另存训练期cost critic，
actor checkpoint仍由原12-key验证器保存。

### 20.3 OOF可学习性与机械梯度门

Preflight只取rollout内完整episode。真实cost target是collision episode中
`0.999^(terminal_step-current_step)`，安全episode为0；尾部未完成episode不参加OOF。所有相同
`ego_idx`严格进入同fold，五折各重新初始化同构critic，10 epochs，无test调参。报告常数均值
baseline MSE、skill、episode-start AUROC和距terminal至少100步的early AUROC。

机械梯度门在actor参数不变时，对同一完整buffer分别累计reward-only与
`reward-lambda*cost` clipped PPO梯度，记录norm、差分相对L2和cosine。它只证明cost信号真正能进入
当前actor调用链，不证明更新方向会改善四图行为。Formal准入、四图验收与停止边界见统一预注册
§35；运行结果完成后必须在`ANALYSIS.md`新增独立章节。

### 20.4 执行结果与保留价值

最终有效preflight为102,400行、153个完整episode、57 collision、85个起点。Reward/cost唯一化、
cost GAE、cost warm-up、dual方向与actor梯度全部通过；起点五折OOF的MSE skill、episode-start
AUROC、early AUROC为`0.04038/0.42855/0.60703`，低于`0.05/0.65/0.65`，formal按合同未运行。

前两个目录分别记录tensor/numpy bootstrap类型错误和尾部未完成episode fold查表错误；都没有
actor更新或科学report，不得合并为三次seed。当前脚本已修复两处机械问题。保留该脚本可以重建
reward/cost唯一化、cost-side buffer、P20 critic、dual与OOF/梯度审计；但当前配置已关闭，禁止
直接改budget、dual或critic重跑。

## 21. Prefix-local joint temporal exploration

### 21.1 正式入口与固定CLI

正式训练仍使用`train_ppo.py`，只新增已有`--speed_exploration_mode`的取值
`prefix_joint_temporal`；没有独立trainer、第二种actor loss或resume入口。该mode要求同时启用
`--prefix_reset_panel`和正`--prefix_reset_interval`，并硬校验steering latent std `.03`、speed
physical std `.15`。正式命令固定canonical BC、Austin、seed42、privilege-GRU、16 env、6,400
steps、batch12,800、30 updates、actor/critic epoch `2/5`、GRU/head/critic LR
`3e-6/3e-5/3e-4`、gamma/GAE `.999/.995`和clip`.20`。

项目根`run.sh`只有这一条显式命令。输出目录必须在启动前不存在；训练器按既有逻辑写
`run_config.json`、warm-up与formal `metrics.jsonl`、`episodes.jsonl`、每update 12-key actor和
critic。不得从Gate E2 actor、Z6-F或任一PPO checkpoint初始化；Gate actor只用于回归测试。

### 21.2 Environment side channel与处理范围

`ppo.env.load_prefix_reset_panel(panel_dir)`除既有snapshot/prefix外强制读取每项`stratum`，仅允许
`collision/lost_overtake`且总数必须为`19/9`。`CentralScheduleSubprocVecEnv`把
`prefix_reset_stratum`写入reset info和reset history；scheduler、28项顺序、每3次collision-role
reset中1次prefix均未改。

`End2RaceGymnasiumEnv.step(action)`在info增加`executed_ego_action`，内容是环境实际收到的2D ego
动作。joint mode采集逐行硬校验raw PPO action、pre-env clipped action和该字段逐位相同；任何
clipping或wrapper/simulator差异立即抛错。该字段是训练期审计，不进入actor observation、reward、
critic或部署checkpoint。

### 21.3 满秩采样器与精确条件likelihood

`ppo.policy`新增公开常量：

```text
PREFIX_JOINT_TEMPORAL_EXPLORATION_MODE = "prefix_joint_temporal"
JOINT_TEMPORAL_RHO = .90
JOINT_TEMPORAL_BLOCK_STEPS = 50
JOINT_TEMPORAL_PREFIX_STEPS = 150
```

`End2RaceGRUPolicy.prepare_rollout_exploration(...)`接收与16个逻辑slot对齐的`prefix_active`、
`prefix_steps`和`prefix_collision_source`。`configure_joint_temporal_generators(seed,batch_size)`为每个
逻辑env创建独立torch generator，seed由`SeedSequence([seed,6,rank])`派生；common/innovation流
不使用全局baseline RNG。inactive路径仍先按原baseline固定形状采样，joint stream不创建或推进，
因此关闭处理时action/log-prob/RNG/telemetry逐位保留。

`_prefix_joint_temporal_actions(mean,distribution,baseline_actions)`只覆盖
`prefix_active & collision_source & prefix_step<150`的slot。每个50步block和动作维独立采样
`r_t=sqrt(.9)*epsilon_block+sqrt(.1)*eta_t`；steering在latent mean上加`.03*r_t`后执行
`.52*tanh`，speed在physical mean上加`.15*r_t`。每env的`block_count*batch_size+rank+1`形成正的
int64 UID；position必须严格按`prefix_step%50`连续。block、episode或inactive会清空common、残差
和UID状态；跨rollout但未跨block时状态继续。

下列函数是E0与replay共同使用的公开数学组件：

- `joint_temporal_standardized_residuals(mean_actions,actions) -> (residuals,jacobian)`；
- `joint_temporal_conditional_parameters(residual_sum,position,rho=.90) -> (mean,variance)`；
- `joint_temporal_conditional_log_prob(...) -> (log_prob,residual)`；
- `joint_temporal_sequence_log_prob(mean_actions,actions,rho=.90) -> (per_step_log_prob,residuals)`。

position `n>0`使用`c=rho/(1+(n-1)rho)`、条件mean `c*sum(previous residuals)`、variance
`1-rho^2*n/(1+(n-1)rho)`；position0为标准Normal。steering density包含`.03` scale及
`.52*(1-(a/.52)^2)` Jacobian，speed包含`.15` scale。candidate replay对**当前PPO rollout内部**的
历史必须从candidate actor mean重新计算residual并保留梯度；但上一rollout已经在旧policy下实现的
residual sum是本rollout的固定incoming探索状态，不能在actor update后用新mean重新解释。两者的
边界由buffer rollover决定，不等于条件于保存common latent的另一种surrogate。

### 21.4 Buffer、minibatch cut与跨rollout context

`End2RaceRolloutBuffer.stage_exploration(...)`新增并逐step存储：active bool、int64 block UID、
int64 position、prefix step、collision-source bool和2D standard residual。UID/position在
`swap_and_flatten`后仍显式保持int64；标量尾部singleton必须reshape为一维，避免NumPy广播。

buffer保留两代context：`joint_context_carry`是本轮开头replay消费的incoming，
`joint_context_next_carry`是本轮结束后为下一rollout生成的outgoing。`reset()`只在新buffer开始时把
上轮outgoing提升为本轮incoming并清空next；`finalize_joint_context_carry()`只写next，绝不能在
本轮optimizer/replay前覆盖incoming。第一次formal U2正因单字段实现被outgoing覆盖incoming而在
任何U2 optimizer step前停止，故这条生命周期是硬回归合同。

`finalize_joint_context_carry()`在rollout结束时为每env保存未完成block从position0到末行的
positions与standard residuals；不保存供新actor重演的旧observation/action/hidden。
`_joint_context_for_sequence()`在随机circular split或minibatch sequence从block中间开始时把上一
rollout所需行压成固定`fixed_residual_sum`，并只返回当前rollout内从边界或本轮block起点到sequence
start的observation/action/positions及该起点实际保存的actor hidden。context必须UID相同、position
严格连续；缺行、跨多于一个rollout、residual非finite或source/window不符立即失败。

`_joint_temporal_replay_log_prob()`先用原distribution计算全部inactive行，再按recurrent sequence逐行
覆盖active行。每条sequence以固定incoming residual sum初始化；当前rollout context从实际保存的
current-rollout hidden经candidate actor重放，其mean不detach并参与后续条件likelihood梯度，但
context行没有advantage/mask，不进入PPO loss、KL、clip fraction或valid count。block position0清空
残差；active非零position没有匹配UID/context时fail closed。第二个formal attempt曾因错误重放上一
rollout而在U2 exact得到`.342732/.290172`并停止，这个真实failure是该边界的回归来源。

### 21.5 Formal fail-closed与新增metrics

`End2RaceRecurrentPPO._assert_full_buffer_ratio_identity()`对joint mode每个formal update都测普通
batched和collection-equivalent exact全buffer ratio；exact的log-ratio或`|ratio-1|`任一超过
`5e-5`直接在actor optimizer前停止。batched两项均小于`.02`正常继续；达到`.02`且exact通过时，
`_adjudicate_batched_replay()`在正式actor step前以相同first-epoch 8 minibatch累计batched/exact
dry gradient，要求cosine`>=.999`、相对L2`<=.02`、两路clip fraction0、mean KL`<=1e-4`、valid
count和minibatch identity相同、参数不step；未通过立即停止。裁决RNG结束后恢复，正式actor仍使用
原预定minibatch序列。

每个formal metrics row除既有PPO/value/episode/prefix统计外新增：joint active count/fraction、
unique block count、treatment leak count、两维residual mean/std、聚合cross-correlation、steering
接近`.95/.99` bound比例、active speed min/max、action identity行数；以及batched/exact ratio是否
实测、各自最大误差和可选的完整batched裁决结构。`actor_optimizer_steps_planned/completed`必须仍为
16/16。聚合residual correlation的有效独立单位近似block而非transition，不承担E0总体协方差判决。

### 21.6 E0工具

`scripts/run_prefix_joint_temporal_e0.py`只有`--output`一个CLI，默认写Gate根下E0 JSON；固定seed
`20260809`，失败exit1，JSON用temporary+replace原子写。它执行：100,000-block均值/方差/50×50
相关与跨维相关；长度1--50的float64 conditional对直接MVN及autograd；FP32 reference；全部49个
cut；实际buffer context和人工跨rollout carry，包括20行fixed incoming、rollout-start零行current
context、step5五行current context及双代UID生命周期；rho0退化；inactive policy action/log-prob/global
RNG/telemetry bitwise；真实sampler150步position/UID/source泄漏；strict 12-key deterministic actor。
报告含每项阈值、逐cut误差、runtime版本及参与源文件身份。删除脚本会失去相关likelihood的独立
数值oracle和任意cut回归入口。

原独立预注册文档已经在2026-08-10归并后删除。E0/E1源码不再运行时读取该Markdown，而是用常量
`LEGACY_PREREGISTRATION_SHA256="ad935f5ded3e3170eeb2032c9f330f8bfd89b5c7e2be9db2fbda51927a8becc4"`
保留历史报告中的`source_sha256.preregistration`身份；完整方法、公式、Gate、实际修复和正式结果
以本节、`ANALYSIS.md` §48与`HANDOFF.md` §9.29为准。该常量只承担历史provenance连续性，不能被
误解为当前源码仍验证一个已删除文件。

### 21.7 E1工具

`scripts/run_prefix_joint_temporal_e1.py`支持`--arm orchestrate|baseline|treatment`，其余CLI固定
output/panel/cache/actor、16 env、6,400 steps、batch12,800、seed42、interval3、hidden-scale4。
orchestrate先验证passed E0并冻结plan，再用独立子进程按baseline→treatment顺序采集；两臂都启用
同一28-prefix scheduler，唯一变化是exploration mode。每臂零optimizer step，保存arm report、
episodes和空formal metrics；汇总要求102,400、51,200/51,200、28/28 prefix、treatment 19/19
collision source、active`>=2%`、零分类泄漏、连续block、GAE误差`<=1e-6`、exact`<=5e-5`、
batched`<.02`或通过冻结裁决、102,400 action identity、16/16 dry actor minibatch、参数不变和wall
ratio`<=1.35`。删除脚本会失去完整source/window泄漏分类和no-update集成复现入口。

### 21.8 E2工具

`scripts/run_prefix_joint_temporal_e2.py`CLI同E1固定输入但没有arm。它验证E1 source身份与pass报告，
按相同seed/scheduler重建一个disabled baseline buffer并确认reset history逐项复现；随后冻结这一个
buffer、actor/critic及Adam状态。A路设mode baseline，B路设mode prefix_joint_temporal但active全
false；两路用同一actor/critic RNG，各真实执行1 actor epoch和1 critic epoch。报告逐项比较step、
valid counts、loss、KL/clip、gradient norm、actor/critic tensor、两套optimizer state、12-key strict
reload和deterministic action/hidden。两个checkpoint只用于回归，不得评测或部署。删除脚本会失去
disabled新replay对真实optimizer step的bitwise语义证明入口。

### 21.9 跨actor-update生命周期实现门

E0--E2均不让joint-active buffer跨过真实actor update，因此无法覆盖第二个formal attempt暴露的
概率状态漂移。修复后用现有`train_ppo.py`直接执行一次实现门，不新增wrapper：正式CLI完全相同，
只把`num_updates`从30限制为2并写入独立新目录。它仍先完成同一12-epoch critic warm-up，再完整执行
U1和U2；用途只是在U1更新真实GRU/output head后，让U2 full-buffer exact Gate消费跨rollout block。

U2实测collection-equivalent `max|log ratio|=0`、`max|ratio-1|=0`，普通batched最大`.000997`；
102,400 action identity、3,685 active、75 blocks、零泄漏、16/16 actor steps并成功写出strict 12-key
U2。该run不是方法性能臂，U1/U2 checkpoint不得正式评测、选点或部署；它只证明修复后的实现已经
覆盖并通过原失败调用链。删除该目录会丢失真实跨update回归实物，但上述CLI与判据足以重建。

### 21.10 固定600正式评测分析器

`scripts/analyze_prefix_joint_temporal_formal_eval.py`只分析已完成的固定正式包，不启动仿真或选择
checkpoint。CLI为`--run-dir`、`--evaluation-root`、`--panel-root`、`--alias-prefix`、`--report`与
`--workers`；非路径默认分别对应post-failure exploratory run、`eval_results`、标准四图600 panel、
`PJTE_u`、run内`formal_eval_report.json`和8 workers。这些是已删除历史分析器当时的默认值，不是
当前入口。已有report拒绝覆盖，JSON经temporary replace
原子写。

分析器先要求训练恰好1行warm-up+30行formal、update连续1--30、全部finite、exact`<=5e-5`、
actor 16/16、泄漏0、action identity 102,400；再严格加载U27--U30的12-key actor。每个checkpoint
当时只接受Austin/Hockenheim/MoscowRaceway/Nuerburgring各600的`standard_multiagent_600_v1` manifest，
CUDA、ego scope、deterministic、8秒、600 result与600 trace、0 error且actor/panel SHA匹配。

9,600个NPZ逐个检查数值dtype/finite、所有数组首维对齐、LiDAR/action/pose/collision shape、时间严格
递增、末行唯一`terminal_post_step=true`和`action_applied=false`、typed collision与二维collision及
result outcome一致。随后在同一2,400场景身份上相对canonical BC、U44和对应update的Z6-F计算
collision removed/created、overtake lost/gained和双侧exact McNemar；逐图重算BC线，独立判定每个
U27--U30的`collision<40`、`overtake>1500`与四图BC gate，并按冻结定义计算U30 L2-M/L2-P。
报告固定记录`evidence_status=post_failure_exploratory_not_original_confirmatory`，避免把后验修复run
误写成原confirmatory通过。删除脚本会失去9,600 trace质量合同与所有配对/L2/L3的一次性可复算入口。

### 21.11 Exact-actor replay修订与第二条生命周期门

第一条post-failure run在U14任何optimizer step前触发§21.5 batched裁决并失败；U1--U13完整，
U14未写checkpoint。由于exception先于metrics落盘，裁决细项没有持久化，只能确认batched至少一项
达到`.02`、exact仍`<=5e-5`且dry-gradient verdict为fail。该目录不得resume或评测。

后续固定实现不放宽裁决线，而是在`End2RaceRecurrentPPO.train()`中令
`prefix_joint_temporal`的正式actor minibatch直接调用
`evaluate_actor_actions(..., collection_equivalent=True)`；其他探索模式仍传false。每个formal row
新增`actor_replay_mode`，新正式run必须30/30均为`collection_equivalent`。普通batched full-buffer
ratio仍作为诊断记录；若它达到`.02`而exact通过，记录
`not_applicable_exact_actor_replay`，不再让未被optimizer使用的近似路径阻断exact更新。exact任一项
超过`5e-5`仍在actor step前硬停止。

修订后E0全部通过。独立全规模lifecycle仍使用canonical BC、seed42、16×6,400和完整warm-up，只把
formal updates固定为2；U1/U2均exact 0、16/16 steps、leak 0、identity 102,400，实际replay mode
均为collection-equivalent，batched诊断最大`.007528`，两个actor均strict 12-key并正常结束。该目录
只承担实现准入，不选点、不评测。

§21.10分析器的当前默认run为
`post-trained/ppo_prefix_reset_joint_temporal_rho0p90_postfailure_exact_actor_exploratory`，并额外要求
30行formal的`actor_replay_mode`全部为collection-equivalent；evidence status固定为
`post_failure_exact_actor_exploratory_not_original_confirmatory`。评测规模仍为U27--U30四图各600，
不生成near400。

### 21.12 Exact-actor正式执行、600评测与分析器实测

最终训练入口`run.sh`只含一条显式命令，输出到全新目录
`post-trained/ppo_prefix_reset_joint_temporal_rho0p90_postfailure_exact_actor_exploratory`。它从
canonical BC启动，Austin、seed42、16×6,400、30 formal updates；与§21.11 lifecycle相同，实际
actor update固定走collection-equivalent replay。训练成功结束，metrics为31行，actor/critic各30个。

训练后按update 27→30、地图Austin→Hockenheim→MoscowRaceway→Nuerburgring串行执行
`scripts/evaluate_scenario_panel.py`；每次传入对应`checkpoints/actor_u00NN.pth`、
`standard_multiagent_600_v1/<map>_600_scenarios.json`、唯一`eval_results/PJTE_uNN/<map>/multiagents`
目录、8 workers、CUDA、ego scope与`--save-traces`。共16个包、9,600 episode；没有near400。

上述显式JSON和评测脚本已于2026-08-10删除；当前四图600由`evaluate.sh`直接生成。这里保留调用链
只为解释历史9,600条结果的来源，不构成恢复第二套标准评测入口的理由。

`scripts/analyze_prefix_joint_temporal_formal_eval.py`首次真实运行修复了两个入口假设，均发生在报告
生成前：脚本目录启动时需把project root加入`sys.path`才能导入根`utils.py`；actor路径必须使用
recorder真实布局`checkpoints/actor_u{update:04d}.pth`，不能假设`updateN/actor.pth`。修复没有改变
指标、panel或判据。当前分析器完整要求30行formal全为collection-equivalent actual replay，并对
16个manifest、9,600个NPZ及三套baseline作全量验证与配对。

机器报告写入run内`formal_eval_report.json`，evidence status为
`post_failure_exact_actor_exploratory_not_original_confirmatory`。质量合同全部true；训练exact最大0、
batched诊断最大`.032192`、active`2.8418%--3.8096%`。四点aggregate为
`132/1494、132/1508、121/1510、117/1513`，`formal_600_task_achieved_updates=[]`且总判决false。
U30三组配对与完整逐图值固化在HANDOFF §9.29和ANALYSIS §48.10；分析器JSON可用于复算，但文档
本身已包含行动所需数字与停止规则。

## 22. Collision-only BC functional regularization

### 22.1 Anchor dataset builder

`scripts/build_collision_bc_anchor_dataset.py`是窄用途冻结器，不启动仿真或训练。CLI只有
`--gate-dir`（默认Z7有效V1目录）和`--output-dir`（默认
`post-trained/panels/collision_bc_anchor_v1`）。输出目录存在且非空时拒绝覆盖；输入必须同时包含
Z7的V1 report、plan、branch0 results和full-BC results，且实验ID、Gate名和18条
`rescued_overtake_scenario_keys`必须精确匹配。

对每条source，builder要求role/stratum为`cohort/collision`、branch0为ego collision、full-BC最终
overtake、计划和实际窗口均为150 applied steps、intervention mask只覆盖冻结窗口且action source
code为3。它读取full-BC真实反事实trace，而不是把teacher动作贴到原U44失败后续；用场景初速
`0.9×`和之后前一行measured speed重建361D actor observation，再调用现有
`EvaluatorCompatibleJointDistribution`把teacher steering physical mean转换为latent mean。

每条`sequences/<episode_key>.npz`只含四个numeric数组：

```text
observations                    float32 [T,361]
teacher_latent_steering_mean    float32 [T]
teacher_physical_speed_mean     float32 [T]
anchor_mask                     bool    [T]
```

`manifest.json`固定`dataset_id=collision_bc_anchor_v1`、18 episode、每条150 anchor step、source
outcome、窗口、sequence相对文件名和SHA。所有数组finite；sequence文件与manifest都属于训练输入，
删除后不能在不重新读取Z7 full-BC trace的情况下复现实验。

### 22.2 历史Runtime loss与训练入口（当前已退役）

历史`ppo/collision_anchor.py`公开`CollisionBCAnchor(path, policy, device)`，以及`loss()`和
`maximum_action_error()`。构造时重新核验manifest、18个唯一key、每个NPZ哈希、shape、mask和
finite。`loss()`按episode从零hidden逐步运行当前student actor；prefix只改变hidden，mask外不计
loss。每条episode分别计算：

```text
steering = .5 * mean(((student_latent - teacher_latent) / .03)^2)
speed    = .5 * mean(((student_speed  - teacher_speed)  / .15)^2)
episode  = steering + speed
```

最终先对150步取均值，再对18条episode等权均值，返回
`(total_loss, steering_loss, speed_loss)`。`maximum_action_error()`用相同recurrent路径做无梯度机械
对齐检查。

历史`train_ppo.py`新增两个CLI：`--collision_bc_anchor_dataset`默认空字符串、
`--collision_bc_anchor_beta`默认0。Beta必须finite且非负；正beta必须同时提供dataset。关闭路径不
实例化anchor，保持原PPO调用。启用时`End2RaceRecurrentPPO`在每个actor minibatch对同一18条输入
计算anchor loss，执行：

```text
combined_loss = PPO policy loss + beta * anchor total loss
```

同一个actor optimizer更新GRU和output head；critic optimizer、rollout、reward和部署actor不读取
anchor。每个formal row除原PPO指标外记录beta、episode数、总/steering/speed loss、pre/post loss、
functional drift、PPO/anchor/combined gradient norm、学习率加权step-space norm和clip后norm。
Run config另写`COLLISION_BC_ANCHOR` enabled/dataset/beta/loss合同。

### 22.3 固定实现检查与正式执行

`scripts/run_collision_bc_anchor_training_gate.py`只承担dataset/梯度机械检查，CLI为
`--actor-path`、`--collision-cache-dir`、`--anchor-dataset`、`--gate-dir`、`--seed`、`--n-envs`、
`--n-steps`和`--batch-size`；它没有生成候选actor的权限。有效attempt固定beta为
`.006405998602049812`，证明canonical对齐、anchor gradient非零、strict 12-key和一条实际
optimizer路径可执行。正式训练仍直接调用`train_ppo.py`，固定canonical BC、Austin、seed42、
16×6,400、batch12,800、45 updates、actor/critic epoch`2/5`、GRU/head/critic LR
`3e-6/3e-5/3e-4`、corridor-temporal speed std`.15`/hold50/gap2和beta上述值。

Recorder真实布局同时有`checkpoints/actor_uNNNN.pth`与后来为正式评测建立的
`update<N>/actor.pth`硬链接；两者SHA必须相同，不能当作两个模型。正式评测用根`evaluate.sh`
逐次运行一个`MODEL_PATH × MAP_NAME`；它不是全流程orchestrator。为本轮增加的最小兼容项是环境
变量`PYTHON`与`COLLISION_SCOPE`，并在aggregate `error_count != 0`时保留worker临时目录、非零
退出；不得恢复“有worker error仍删临时目录并返回成功”的旧行为。

删除builder会失去从Z7 branch冻结18条反事实sequence的能力。2026-08-10按用户授权，历史
`ppo/collision_anchor.py`、两个train CLI、`End2RaceRecurrentPPO`接入和run-config telemetry已经
完整删除；本节保留功能重建合同，不表示当前源码仍能启用。正式效果与停止规则在`ANALYSIS.md`
§49；这些实现记录不授权改变beta、teacher、窗口或重跑。

## 23. Calibrated collision-cost Constrained PPO direct formal

### 23.1 新模式与固定合同

`scripts/run_constrained_ppo.py`保留旧`prepare/preflight/formal`，新增
`--mode direct_formal`。它不读取旧Z9 preflight verdict，也不运行新的研究Gate；唯一用途是执行
用户授权的独立`d=.19`正式实例。常量`DIRECT_COST_BUDGET=.19`来自记录的同协议U44训练分布
`ceil(100*26/141)/100`，不提供`--cost-budget`覆盖。

`run_direct_formal(args)`在创建环境前强制：actor必须是canonical BC真实路径；seed42、16 env、
6,400 steps、batch12,800和30 updates必须精确；输出目录若已存在且非空则拒绝覆盖。它随后配置
固定训练数值、479条canonical Austin collision pool和600条ordinary pool，调用同一个
`ConstrainedEnd2RacePPO.learn()`完成1个warm-up rollout加30个formal rollout。结束时要求U30 actor
与cost critic都存在，actor必须finite strict 12-key；`training_summary.json`固定写30 updates、
budget来源和评测band`[27,28,29,30]`。

### 23.2 Reward/cost、critic与dual调用链

`ConstrainedEnd2RacePPO.collect_rollouts()`通过每步info保存first-ego-collision cost、原
`reward_collision`、done、ego index和scenario ID，shape必须为`[6400,16]`。机械恒等式为
`reward_collision == -2 * cost`；随后以`adjusted_reward = original_reward - reward_collision`
从reward GAE精确移除collision尖峰，first collision只作为cost计价。Reward critic仍为
`privilege_gru`；独立训练期cost critic为P20 MLP `20→120 ReLU→30 ReLU→1`，LR`3e-4`、5 epochs、
grad clip`.5`。

Cost GAE固定`gamma=.999/lambda=.995`。Formal actor先构造
`A_reward - lambda_cost*A_cost`，再由原PPO对合成advantage做标准化与clipped surrogate；没有第二
actor/head。Dual从1开始，每个rollout按所有已完成episode的pooled ego collision rate更新：

```text
lambda <- clip(lambda + .5 * (pooled_rate - .19), 0, 20)
```

每个formal row记录reward去重误差、cost event、完成collision/episode数与pooled rate、budget、
lambda used/after、三类advantage均值/标准差、cost critic五轮loss/gradient、pre/post cost value
loss/EV/预测与return尺度，并由父类另存PPO actor/reward critic指标和actor/critic checkpoint。当前
实现的dual权威口径是pooled完成episode率；scenario IDs和done在rollout内用于机械审计，但metrics
没有另存collision/ordinary role分层率和尾部未完成episode数。因此最终解释不得声称episode层
严格50/50，也不得伪造role统计；该telemetry缺口不改变实际dual输入，但属于结果边界。

### 23.3 正式执行与后续唯一动作

执行期间根`run.sh`只包含一次`direct_formal`调用；第一次输出实验为
`ppo_constrained_collision_cost_calibrated_v1`，修复后fresh重跑只把输出改为
`ppo_constrained_collision_cost_calibrated_v1_rerun`，其余仍为canonical BC、固定479 collision
cache和上述全部参数。该命令只完成训练，不自动eval；任务结束后已从`run.sh`移除。训练正常完成
后，只加载U27--U30 actor，按checkpoint串行
运行Austin、Hockenheim、MoscowRaceway、Nuerburgring各600，CUDA deterministic、ego scope、
numeric trace；cost critic和dual不进入评测。

不得因中途collision rate、lambda或value fit修改budget/dual/critic/reward/updates，也不得把旧
`d=.10` Z9 preflight否决改写为通过。删除`direct_formal`会失去固定可达budget的无Gate正式入口；
旧三阶段入口和其`d=.10`历史语义仍独立存在。

### 23.4 第一次formal暴露的buffer继承漂移

第一次`direct_formal`完成双critic warm-up和第一个formal rollout后，在任何formal optimizer step
前触发`IndexError`。原因是`ConstrainedRolloutBuffer.get()`复制父类旧字段清单，未展平后来新增的
`joint_temporal_active/block_uid/block_position/prefix_step/collision_source/standard_residuals`；
父类telemetry用102,400展平索引访问仍为6,400行的数组。

修复后的override不再复制父清单：只先对四个cost数组调用`swap_and_flatten`，再
`yield from super().get(batch_size, rng=rng)`。这样父类拥有基础字段、joint字段、hidden swap、
rotation和`generator_ready`的唯一权威，cost子类只追加cost side tensor。最小回归构造4-step×2-env
buffer，要求3个minibatch消费8/8有效transition，base observation、joint active和cost advantage都
展平为8行，`current_cost_advantages/returns`与最后一个sequence mask对齐。

失败目录只含两行warm-up metrics、153+159 episode和两个warm-up critic，无actor checkpoint，
禁止resume/eval。修复后正式执行必须使用全新`_rerun`目录；相同CLI除output-dir外不得改变。这个
机械重启不构成第二个科学seed，也不能把失败目录warm-up与新目录formal拼成一条run。

### 23.5 本次自动评测编排的边界

本轮没有把训练、eval、最终统计和文档更新塞进新的长期orchestrator。执行时使用一个一次性shell
watcher完成以下固定顺序：等待`training_summary.json`；验证62行metrics、30组checkpoint、每轮
16/16 actor step、reward/cost唯一化与U27--U30 finite strict 12-key；随后按
`U27→U28→U29→U30`，每点按`Austin→Hockenheim→MoscowRaceway→Nuerburgring`串行调用
`scripts/evaluate_scenario_panel.py`。每次固定workers12、hidden scale4、8秒、CUDA、ego collision
scope、保存numeric trace，任一命令非零即停止。

因此本次从用户视角是启动后无需人工续命，但从代码合同看仍是“训练入口 + 一次性评测watcher +
独立最终审计”三段，不存在一个可复用的单命令全流程脚本。最终16包均完成；性能与停止规则只在
`ANALYSIS.md` §50和`HANDOFF.md`对应判决中维护，本节不重复作为结果权威。
`training_summary.json`中的`training_complete_evaluation_pending`只是训练完成后触发watcher的
单向sentinel，watcher不会回写该文件；不得因这个字段保留原值而把已经完成的eval误判为仍在运行。

## 24. `analyze_signed_interaction_phase_gate.py`：二值front potential与clearing/closing离线诊断

`scripts/analyze_signed_interaction_phase_gate.py`是只读诊断，不启动仿真step、模型推理、训练或
评测。CLI为：

- `--bc-root`：默认`eval_results/pretrained_end2race`；
- `--u44-root`：默认历史走廊时间相关U44四图根；
- `--output`：默认写V2 schema的`gate_report.json`；旧V1产品保留原machine fail但不是当前科学
  verdict。

输入固定读取Austin、Hockenheim、MoscowRaceway、Nuerburgring四张正式地图各600的
`results_multi.json`与numeric trace。每包必须600 episode、0 error；BC/U44 key集合必须相等，
pose finite，数组长度、`action_applied`和唯一末行`terminal_post_step`合同必须成立。脚本只创建
每张地图的F110环境以读取与训练一致的map distance field和resolution，随后立即关闭；不调用
`reset/step`，所以`new_simulation=false`。

核心算法固定为：

1. source取`BC无ego collision、U44为ego-opp collision`，必须精确23条；
2. control从U44安全overtake中按同地图、opponent raceline、speed分层，选初始ego XY最近者，
   无放回一对一匹配；
3. source事件取首次ego-opp marker，control事件取真实OBB全局最小surface clearance；
4. front定义为对手中心在ego车体纵向轴前方且两车OBB在ego横向轴投影严格重叠；车长/宽固定
   `.58/.31m`；
5. current potential为`-.05*max(vehicle_shortfall²,wall_shortfall²)`，signed候选只把vehicle项乘
   `front`；尺度固定纵`.6m`、横`.2m`、wall`.2m`、gamma`.999`；
6. 按真实collision terminal refund重放事件前150个transition，负shaping mass为
   `sum(max(-F,0))`，released为current减signed；
7. V1旧程序要求source front计数精确复现`4/3/3/1`，且control aggregate released严格大于source、
   matched `control-source`中位数为正；V2仍重算并保存这两项，但固定标记
   `legacy_absolute_release_criteria_scientifically_valid=false`，因为绝对量受基线risk mass混淆；
8. V2逐episode计算released/current-negative比例；分母为0时写`null`而不是0，只在pair两侧均定义
   时计算归一化paired差；
9. 方向窗口固定过去10步=`.1s`。未截断方向为
   `(normalized_OBB_distance[t-10]-distance[t])/.1s`，正值closing；active-risk方向为
   `(q_vehicle[t]-q_vehicle[t-10])/.1s`，用于区分“存在几何方向”与“当前potential实际有support”；
10. event/提前`.5/1.0/1.5s`分别报告连续速率和closing二值的source-outcome AUROC；AUROC用全
    source/control pairwise排名，95%区间按“地图+source ego startpoint”的21个cluster做10,000次
    bootstrap，matched control随source绑定；seed从`20260810`按metric和offset确定性派生。它固定为诊断，不是
    训练准入或方法类否决门；
11. 另在全部2,400条U44 trace统计front翻转，只在翻转相邻两帧调用真实OBB/wall potential，记录
    非零跳变、额外shaping及分位数；front翻转和`.02`不参加判决。

V2输出JSON固定`schema_version=2`、`verdict=diagnostic_complete_scientific_effect_inconclusive`，
另存`legacy_procedural_verdict`、`training_decision`、`scientific_falsification=false`、绝对尺度
混淆、source/control cohort、paired selectivity、归一化比例、两套方向AUROC/区间、匹配距离及
`<=10/20m`敏感性、全局flip诊断和已知风险。路径可清理，行动所需数字已经写入ANALYSIS §53和
HANDOFF对应判决。删除脚本会失去按相同OBB、wall、terminal和pair-bootstrap合同重算的便利；
重建时不得把control改为有放回、把事件改成pass crossing、把四个offset当独立样本，或用任一
AUROC关闭整个interaction-phase方法类。

## 25. Simulator-return-filtered first-action preference

### 25.1 调用链与新增接口

```text
scripts/build_first_action_preference_dataset.py
-> ppo/rollout.py: FirstActionPreferenceDataset
-> train_ppo.py --first_action_preference_dataset --first_action_preference_step_fraction
-> ppo/rollout.py: End2RaceRecurrentPPO._calibrate_first_action_preference()/train()
-> scripts/analyze_first_action_preference_formal_eval.py
```

`train_ppo.py`新增两个默认关闭的CLI：`--first_action_preference_dataset`默认空字符串，
`--first_action_preference_step_fraction`默认0。正fraction必须提供dataset，且不能与online
same-state branch同时启用。当前formal合同把fraction锁为`.10`，并锁定canonical BC、Austin、
`privilege_gru`、corridor-temporal、16 env、6,400 steps、12,800 batch、45 updates、actor/critic
epoch `2/5`；disabled path不构造dataset或改变原PPO loss。

### 25.2 数据构造器

`build_first_action_preference_dataset.py`必填`--plan`、`--branch0-root`、`--u44-model-path`、
`--output-dir`；`--workers=12`、`--hidden-scale=4`，`--prepare-only`只冻结plan。它固定从输入plan
选择inherited/created collision、lost-overtake与safe-control episode，在event前150/100/50步
构造state；所有新simulator/task seed固定42。动作库为4个steering、4个speed和4个coordinated
single-step residual，候选只在snapshot后的第一步替换动作，之后恢复同一冻结U44。

12个残差的物理值固定为steering `-0.04/-0.02/+0.02/+0.04 rad`，speed
`-1.0/-0.5/+0.5/+1.0 m/s`，以及`steering +/-0.02 rad`与`speed +/-0.5 m/s`的四个笛卡尔组合。
候选相对当前snapshot上的U44 deterministic raw mean构造；steering按正式actor边界clip到
`[-.52,.52]`，speed不增加额外项目级clip。clip后与noop或另一候选逐位相同的executed action必须
去重，不能靠重复候选提高state权重。

核心顺序：

1. `build_plan()`按episode key和lead生成确定性state/candidate identity；已有plan必须逐字段相等，
   非空新目录fail closed；
2. `run_snapshot_task()`从场景起点重放到decision index，保存raw F110、planner、projector、
   previous speed和actor hidden；pickle往返后noop suffix必须action/trace/result精确；提前终止state
   明确记unavailable；
3. `run_candidate_task()`从同一snapshot执行一个candidate raw action，然后冻结U44跑到terminal；
4. `preference_direction()`只比较`(no_ego_collision,overtake)`两维；严格Pareto优劣才生成pair，
   相等或互不可比均不训练；
5. `assemble_dataset()`按episode写361D observation sequence、episode start和带decision index的
   good/bad action；target/control按episode而非pair均衡；所有JSON采用atomic replace，partial JSONL
   按state/candidate id可恢复且冲突fail closed。

Snapshot必须覆盖F110两车完整动力学、scan/collision状态、opponent `LatticePlanner`内部状态、
reward/progress projector与elapsed-time状态、previous measured speed、当前observation、U44读取该
observation前的GRU hidden和随机状态，并通过pickle往返。Noop与candidate在决策步都先让U44对同一
observation forward；candidate只替换该步executed ego action，下一步起使用该次forward得到的next
hidden恢复同一U44闭环，opponent继续在反事实状态上真实规划。不得把source后续observation、opponent
action或U44 action硬贴到candidate branch。

历史future event只用于冻结Austin episode identity和event-relative decision index；正式seed42标签
只比较同一个当前snapshot的noop/candidate terminal `(no_ego_collision,overtake)`。PPO rollout、
四图eval和部署均不读future信息，但dataset设计与标签使用训练期hindsight，不能写成整个训练完全
没有未来信息。测试三图不得进入state、动作、beta、label或checkpoint选择。

稳定产品为`plan.json`、`snapshot_results.json`、`candidate_results.json`、`gate_report.json`、
`manifest.json`与`episodes/*.npz`。Manifest schema v1；每个episode NPZ必须且只含
`observations[T,361] float32`与`episode_starts[T] bool`，首行start为true且之后全false。Manifest
记录role、stratum、sequence hash、decision index、lead和每个pair的good/bad 2D action、family、
direction。删除该目录会删除formal训练输入，重建成本是1,173次snapshot/noop和14,076次candidate
闭环分支，不能把它当普通分析产品清理。

### 25.3 Actor preference loss与beta

`FirstActionPreferenceDataset(path, policy, device, seed)`验证dataset/gate schema、hash、唯一episode、
361D shape、finite数值及非空target/control。采样RNG由`SeedSequence([seed,0xF1A57])`独立派生；
每个optimizer minibatch无放回循环抽8个target episode与8个control episode。每条episode只做一次
从零hidden开始的因果GRU sequence forward，再在所有decision index读取当前student mean。

每个pair的margin为`log pi(a_good|h)-log pi(a_bad|h)`，loss为`softplus(-margin)`；先在state内
平均pair，再在episode内平均state，再分别平均target/control，最终各权`.5`。它直接反传到原actor
GRU与output head；没有额外部署head。

首个formal actor update前，`_calibrate_first_action_preference()`保存并恢复actor minibatch RNG与
preference sampler state，在与正式actor相同的16个minibatch上分别算PPO和preference的LR加权
step-space norm。固定：

```text
beta = target_step_fraction * median(PPO step norm) / median(preference step norm)
```

任一norm或beta非finite/非正即fail closed。beta只算一次并写calibration JSON；正式loss固定为
`policy_loss + beta*preference_loss`，之后沿用原actor max-grad-norm `.5`。Metrics逐轮保存role/family
margin、satisfied fraction、两类gradient/step norm、cosine、combined/clipped norm和固定pair计数。

### 25.4 正式结果审计脚本

`analyze_first_action_preference_formal_eval.py`参数为`--run-dir`、`--evaluation-root`、`--panel-root`、
`--report`和`--workers=8`。它固定验证U42--U45四图各600：训练必须seed42、46行finite metrics、
45个连续formal update、每轮16 actor step、固定beta及337/103 pair；checkpoint和canonical
`update<N>/actor.pth`必须hash相等且strict 12-key。

结果侧要求600 episode、0 error、aggregate/episode重算一致、panel/result/trace key相等。Numeric
trace必须字段齐备、首维对齐、finite、`len=steps+1`、time递增且终值与result一致；末行唯一
terminal且不应用动作，typed collision必须与二维marker及result outcome一致。随后对BC、旧U44、
RW30逐图和pooled计算collision removed/created、overtake lost/gained及双侧exact McNemar。已有
report拒绝覆盖。脚本不做checkpoint选择；主点固定U44。

本轮没有另建独立unit-test文件；数据构造器、dataset loader、formal argument validation和最终
分析器本身均为fail-closed执行合同，且已在完整14,076 branch、45 update、9,600 trace规模上走通。
2026-08-10模块合并后，当时的`scripts/test_first_action_preference.py`曾用保留的真实dataset断言
46/19 episode计数、8/8 batch和三项finite preference loss。2026-08-12当前回归已改为运行时生成
1 target/1 control的最小schema fixture，只验证hash/schema/sampler/loss，不再依赖旧U44 panel；正式
46/19数据与U44结果仍按本节前述完整产物审计，不受测试fixture替换影响。
源码最小静态自检为相关五个Python文件`py_compile`。删除分析器会失去一键重算完整trace合同与配对
统计的便利，但不会删除已在ANALYSIS和HANDOFF固化的科学结论。

## 26. Online same-state branched PPO

### 26.1 目的、边界与调用链

该实现用于消除§25固定preference数据来自上一代U44闭环的依赖。它不读取U44、RW、first-action
preference panel或任何旧PPO trace；唯一允许的持久训练输入仍是原PPO共有的canonical BC初始化、
canonical-BC Austin collision cache与ordinary scenario定义。新增分支状态、动作和return全部在
当前student的同一轮Austin on-policy collection内临时产生，rollout结束后不写成跨update数据集。

准确名称是`single-stage PPO with online same-state branched return augmentation`。它没有BC/模仿loss，
新增actor项仍是clipped PPO surrogate；但因为额外分支样本作为独立stratum并以固定`.10`加权，所以
不能把它称为“完全未改动的标准PPO”。部署actor输入、结构与strict 12-key checkpoint不变，snapshot、
branch simulator和临时buffer均只在训练时存在。

```text
train_ppo.py --online_same_state_branch_ppo
-> ppo/env.py: End2RaceGymnasiumEnv.capture_runtime_snapshot()/restore_runtime_snapshot()
-> ppo/rollout.py: End2RaceRecurrentPPO._collect_online_branch_rollouts()
-> ppo/rollout.py: OnlineBranchRollout
-> ppo/policy.py: forward_independent_collection()/evaluate_independent_actor_actions()
```

CLI只新增`--online_same_state_branch_ppo`布尔开关，默认关闭，不增加dataset/model/teacher路径。Formal
validation锁定canonical BC、Austin、`privilege_gru`、`speed_noise_hold_steps=1`且
`front_corridor_speed_noise_hold_steps=0`的逐步独立Gaussian exploration、
seed42、16 env、6,400 steps、12,800 batch、45 updates和actor/critic epoch `2/5`；它与prefix-reset、
fixed first-action preference互斥。disabled path不构造runtime snapshot或branch
buffer，也不改变原collector和actor loss。

### 26.2 固定在线分支算法

固定常量位于`ppo/ppo_config.yaml`，由`ppo/rollout.py`消费：

```text
ONLINE_BRANCH_STATES_PER_ROLLOUT = 16
ONLINE_BRANCH_ACTIONS_PER_STATE  = 4
ONLINE_BRANCH_HORIZON_STEPS      = 100
ONLINE_BRANCH_LOSS_COEFFICIENT   = .10
gamma                            = 主PPO gamma（formal为.999）
```

每个formal rollout把16个触发点等距放在6,400个vector timestep内；触发点不看未来collision、terminal、
geometry或正式eval身份。第`i`个触发点只取rank `i`，所以固定覆盖8个collision-role与8个ordinary-role
当前状态。每个状态依次执行：

1. 在主动作产生前保存该worker完整runtime snapshot、当前381D observation、actor/critic pre-observation
   hidden与episode-start；保存全局CPU/CUDA torch RNG；
2. 当前update内参数冻结的`pi_old`在同一observation/actor hidden上采4个第一动作，并保存各自精确
   old log-prob；第一动作若需要action-space clipping立即fail closed；
3. 每个候选前恢复相同snapshot；执行候选第一动作，之后最多99步继续使用同一冻结`pi_old`与各分支
   自己演化的actor/critic hidden；四个候选复用同一组continuation torch standard-noise流和snapshot
   中相同的scan RNG起点，以common random numbers降低第一动作比较方差，同时每条分支的边际动作
   分布仍是当前`pi_old`；continuation的physical action clipping与主collector一致；
4. terminal使用真实原reward return；timeout或100步horizon使用当前reward critic bootstrap。Horizon
   bootstrap直接复用最后一个continuation forward在末状态产生的value，不把同一末observation再次
   输入recurrent critic。分支不调用
   parent scheduler reset，不改变scenario queue；
5. K=4候选在同一状态内用leave-one-out baseline：
   `A_i = G_i - mean(G_j, j != i)`。因此每组advantage和为0，且对第i个采样动作的baseline不含其自身
   return；全部64条advantage再作全buffer标准化；
6. `finally`无条件恢复worker snapshot和torch RNG，再验证恢复observation逐位相等；之后才调用原主
   vector action，所以在线搜索既不替换主动作，也不推进主rollout或改变其随机动作流；
7. actor每个epoch把64条branch样本无放回分到原8个actor minibatch，每个minibatch优化
   `L_main_PPO + .10 * L_branch_PPO`。Branch项使用同一clip range、stored old log-prob和当前actor
   log-prob；critic只按原主rollout训练，不把branch return写入value-loss target。

Formal每轮额外模拟步上限为`16*4*100=6,400`，相对主`16*6,400=102,400`为6.25%；提前terminal会
减少实际值。Metrics保存state/action/simulator-step计数、return/advantage/length、四类outcome、branch
loss/KL/clip fraction与更新前最大absolute log-ratio。更新前branch log-ratio必须`<=5e-5`。

### 26.3 Runtime snapshot schema

`capture_runtime_snapshot()`返回schema v1，顶层只含`schema_version/environment/observation`。Environment
state覆盖：两台RaceCar的state、opponent pose、acceleration、steering velocity/buffer、collision与scan
RNG；simulator poses/collision arrays；F110 lap/time/render字段与class-level current observation；
LatticePlanner trajectory、tracker count、speed scale、12个planner字段、selection function、pure-pursuit
error/nearest distance；reward projector/relative/collision latch/risk/clearance状态；wrapper elapsed time、
previous measured speed、raw observation、current EpisodeResetSpec、episode reward/counter/min-clearance字段；
reset RNG与causal corridor gate当前值。恢复拒绝schema或字段集合漂移，并要求重建381D observation逐位
等于snapshot值。Prefix-reset仍复用同一底层restore helper，但保留原先reset、source标注和hidden burn-in
语义。

### 26.4 回归测试与已验证结果

本节以下是退役general-state branch实现当时的历史回归合同；当前替代回归为
`scripts/test_first_action_preference.py`，验证fixed preference与collision-triggered临时preference，
不再重建旧general-state branch likelihood。
固定断言：

1. 运行时最小fixed-preference fixture加载为1 target/1 control，按8/8循环采样且总/两role loss全部
   finite；测试不读取任何旧U44 dataset；
2. 真实单环境在第5步capture，restore后用同一动作得到381D next observation、reward、terminated/
   truncated及关键info逐位相同；
3. 真实16-worker/CUDA policy对16个当前on-policy状态各跑4条100步分支，共64 action、6,400 simulator
   step；每个rank结束后主observation逐位未变，role精确32/32；
4. leave-one-out advantage逐state和为0且全局finite/nonconstant；更新前最大absolute log-ratio为0；
5. `.10*L_branch_PPO`产生finite非零actor gradient，未执行optimizer时actor逐tensor不变；
6. 另走完整16-step生命周期：主rollout、在线分支、一次actor/critic update、checkpoint和metrics全部
   完成；actor确实改变、`_n_updates=1`、metrics记录64条branch action。

2026-08-10修正horizon末状态重复critic forward后重跑机械摘要：snapshot exact=true；CUDA 64条均走满
100步，branch return mean/std `.0349048/.3974554`、raw leave-one-out advantage std `.0424168`、
pre-update max abs log-ratio `0`；独立gradient norm `13.0399`且finite；
actor/critic update和临时checkpoint成功。短生命周期的critic explained variance为`-.805422`，因主
rollout只有16步，不是性能或critic准入证据。模块合并当时的真实preference smoke为46/19 episode、
8/8 batch，总/target/control loss为`2.995607/5.957267/.033947`且finite；当前自包含fixture只替代
机械回归的数据依赖，不改写该历史结果。

### 26.5 尚未证明与停止边界

2026-08-11固定合同已完成45个formal update和U45四图各600。训练累计720个state、2,880条branch、
266,795 branch simulator step；outcome为horizon 2,463、ego collision 197、overtake 156、follow 64，
每轮advantage std均值`.03638`。U45四图为`23/359、33/361、42/388、32/393`，合计`130/1501`；
相对BC碰撞无变化而超车增加，但被RW30与first-action preference U44计数支配。当前固定实例关闭，
不得扫描state数、K、horizon、loss coefficient、seed、探索std、trigger或update长度。训练branch不能
替代模型eval；完整执行和配对见§30与`ANALYSIS.md` §56。2026-08-10 `OnlineBranchRollout`与固定常量已经并入
`ppo/rollout.py`；删除其中的在线分支实现、四个接入点和对应测试会失去该在线训练臂及其
snapshot/ratio/lifecycle回归能力，但不会影响开关默认关闭时的production路径。

## 27. 数值接口的时间相关速度探索

### 27.1 CLI与默认等价路径

`train_ppo.py`不再暴露文字型探索模式，改为三个直接表达物理语义的整数接口：

```text
--steering_noise_hold_steps 1
--speed_noise_hold_steps 1
--front_corridor_speed_noise_hold_steps 0
```

默认`1/1/0`表示每个`.01s` simulator step都重新采样steering与speed standard Gaussian residual，且不构造
前向走廊gate；它继续走原逐步独立collector，不增加gate几何、temporal state或额外RNG draw。
`--speed_noise_hold_steps 10`实现全局K10：actor对当前361D observation和GRU hidden的mean仍以
100 Hz重算，只把同一个standard residual `z`连续用于10个mean，实际speed action始终为
`mean_t + .15*z`。

双频探索通过：

```text
--speed_noise_hold_steps 10 --front_corridor_speed_noise_hold_steps 50
```

走廊外每10步重采样；当前causal 2m front-corridor gate由false切true时立即采一个新residual并持有
50步，gate由true切false时立即采新residual并重新进入K10节奏。Episode start同样强制新块。Gate只读
当前状态，不读future collision/outcome；steering仍逐步独立，speed marginal std固定`.15`。

调用链为：

```text
train_ppo.py --speed_noise_hold_steps K [--front_corridor_speed_noise_hold_steps C]
-> CentralScheduleSubprocVecEnv(front corridor enabled iff C>0)
-> End2RaceGRUPolicy.prepare_rollout_exploration()
-> End2RaceGRUPolicy._structured_rollout_parameters()
-> End2RaceRolloutBuffer.stage_exploration()
```

每个transition保存并重放其边际Gaussian log-prob，actor/value、GAE和recurrent sequence仍按100 Hz。
准确边界是：这是“时间相关collection exploration + per-transition PPO likelihood”，不是把整段K步
residual当作一个联合随机变量后计算block joint likelihood。Deterministic eval直接取actor mean，三个
hold参数均不进入部署actor或12-key checkpoint。

2026-08-11新增`--steering_noise_hold_steps`。设为`10`时，steering latent standard residual连续保留
10个`.01s` step，actor steering mean仍每步重算，物理动作仍为
`0.52*tanh(latent_mean_t + .03*z)`；speed K10语义不变。新接口不改变边际`.03/.15`标准差，且
`front_corridor_speed_noise_hold_steps=0`时完全不构造走廊gate。当前正式处理只测试全局
steering K10与speed K10的组合，不测试steering走廊K50。

### 27.2 回归合同

`scripts/test_first_action_preference.py::exploration_test()`直接检查：speed K10及steering+speed K10的0--9、10--19、20--24
分别保持三个residual；K10/K50在第12步进入gate时重采样、12--61共50步逐位不变、第62步退出gate
再次重采样。2026-08-10实测通过。2026-08-11两臂正式30-update与四图各600均完成：全局K10为
`85/1488`，K10/K50为`74/1566`；后者相对前者collision `57/46,p=.3245`、overtake
`33/111,p=4.58e-11`。因此固定K10/K50配置通过为高超车前沿，未建立安全改善，见§30。

## 28. 100 Hz序列上的10 Hz直接PPO loss抽样

### 28.1 精确训练语义

历史实验CLI为`--ppo_loss_sample_stride`，默认`1`。设为`10`时环境仍以100 Hz执行并保存全部observation、action、
reward、old log-prob、actor/critic hidden与episode boundary；GAE/return也在完整100 Hz transition链上
计算。Recurrent actor和critic在minibatch内仍逐步消费全部有效行，因此第`t`个被选loss位置的hidden
包含此前每个`.01s` observation，不把GRU改成10 Hz。

每个formal rollout结束后，独立`SeedSequence([seed,7])` RNG对每个env内的每个episode segment随机
选择一个`0..min(9,length-1)`相位，再选`phase, phase+10, ...`。该mask在一个rollout内固定，actor的
所有epoch和critic的所有epoch使用同一批位置；padding只影响replay有效性，不会被选入loss。Formal
actor clipped surrogate、advantage normalization、KL/clip telemetry和value MSE只在mask为1的位置
计算。Critic warm-up不是PPO formal loss，继续使用完整100 Hz数据。

因此该历史接口准确描述为“100 Hz collection/GAE/recurrent replay + 10 Hz direct actor/value loss”，不能写成
10 Hz环境、10 Hz动作输出、10 HzGRU或丢弃90% rollout。它是有意改变优化估计量的实验轴，并不声称
与标准100 Hz PPO数学等价。2026-08-11负结果确认后，用户明确退役该方法；CLI与buffer mask、RNG、
metrics和专用测试均已从活动代码删除，以下只保留历史语义。

### 28.2 回归合同

`End2RaceRolloutBuffer.prepare_loss_sampling()`在flatten前生成固定mask；stride=1不调用RNG且mask全1。
测试用2个env、100步、每个env两段episode，实测200个collection位置选择20个loss位置，每段相邻
位置严格相差10。真实collision-prefix生命周期又用stride10完成一次actor/critic formal update，metrics
确认selected count严格位于0与total之间。2026-08-11正式U30和四图各600完成，结果`133/1452`；
相对BC collision `51/55,p=.771`、overtake `41/48,p=.525`，当前固定实例关闭，见§30。
随后`post-trained/ppo_loss_sample_stride10`与对应`eval_results`被删除；模型身份和统计保留在
`HANDOFF.md`与`ANALYSIS.md` §56。

## 29. 当前student碰撞前1秒定向分支PPO

### 29.1 接口、数据边界与恢复算法

CLI为`--collision_prefix_branch_ppo`，默认关闭。它与`--online_same_state_branch_ppo`互斥，并要求
逐步独立speed exploration；它不读取U44/RW、first-action preference dataset、旧trace、旧hidden或
旧PPO checkpoint，canonical BC与普通PPO collision cache仍是共有训练输入。

每个worker在episode reset后只保存一个完整起点runtime snapshot和之后的ego action数组，不逐步保存
100份大snapshot。当前student真实产生ego collision后，worker先保存terminal state，再恢复episode
起点并确定性重放`episode_steps-100`个已执行ego action，由LatticePlanner在重放状态上重新计算对手
动作；得到恰好碰撞时刻前`100*.01=1.00s`的完整prefix snapshot后恢复terminal。Vector env随后照常reset
到主rollout的下一scenario，旧prefix只通过一次`take_collision_prefix_snapshot()`交给parent。

Parent为每个rank保留最近100个主rollout pre-observation：381D observation、actor/critic hidden、
episode-start与rollout step。只有worker prefix observation与parent历史逐位相同时才生成branch；formal
rollout开始不足100步、短episode或没有collision时不生成数据，也不报错。每个有效prefix：

1. 当前冻结`pi_old`在同一observation/hidden采4个第一动作及精确old log-prob；
2. 每个候选恢复同一prefix，之后由同一`pi_old`闭环到terminal或100步horizon；候选共享continuation
   RNG，return使用原reward和critic bootstrap；
3. 同状态4个return形成leave-one-out advantage，再进入与§26相同的clipped branch PPO；branch loss
   系数固定`.10`，critic仍只学习主rollout；
4. 每组结束恢复vector auto-reset后的主environment snapshot与torch RNG，所以搜索不替换原主动作、
   不改变scenario scheduler或主rollout随机流；
5. 一个rollout若没有有效prefix，actor只执行原main PPO loss。Variable branch样本无放回分给现有actor
   minibatch，不要求人为凑满固定状态数。

固定内部量位于`ppo/ppo_config.yaml`：lookback 100步、4 actions/state、branch horizon 100步、loss
coefficient `.10`。部署actor结构、361D输入和12-key checkpoint不变。该方法使用collision hindsight
决定训练时回溯哪个状态，但future信息不会作为actor输入或部署gate；准确名称是
`single-stage PPO with collision-triggered one-second same-state branch return augmentation`。

### 29.2 机械验证与尚未证明

真实Austin collision-cache scenario在第740步发生ego collision；重建prefix与terminal elapsed-time差
`0.9999999999999787s`。完整2-worker CPU生命周期使用正式`.03/.15`探索，在一个有效prefix上得到4条
branch：3条ego collision、1条安全跑满100步，return mean/std `-1.36334/.695286`、raw leave-one-out
advantage std `.927048`、394个额外simulator step；pre-update log-ratio identity通过，并与stride10
main loss共同完成一次actor/critic optimizer update和临时checkpoint。该回归与§26的16-worker CUDA
64分支测试在同一脚本顺序通过。

这些数字只证明1秒恢复、状态/hidden对齐、动作结果差异、likelihood与optimizer链正确；它们不是
collision rescue率，更不是正式模型性能。没有30/45-update Austin训练，没有四图各600 eval，也没有
证据说明固定4动作和100步horizon足够。正式运行前不得把机械测试写成方法通过；默认关闭路径不保存
起点snapshot/action history，不改变production collection或训练数学。

## 30. 2026-08-11 PPO temporal exploration, loss sampling, and online branch execution

### 30.1 Executed contracts

Four fresh actors were trained from `pretrained/end2race.pth`, Austin only, seed42. All unchanged parameters used
the current `privilege_gru`, clip `.20`, 16 x 6,400 transitions/update, fixed collision cache and reward contract.

```bash
python train_ppo.py --pretrained_model_path pretrained/end2race.pth --output_dir post-trained/ppo_global_temporal_speed_noise_hold10steps --seed 42 --num_updates 30 --speed_noise_hold_steps 10
python train_ppo.py --pretrained_model_path pretrained/end2race.pth --output_dir post-trained/ppo_global_hold10_front_corridor_hold50_speed_noise --seed 42 --num_updates 30 --speed_noise_hold_steps 10 --front_corridor_speed_noise_hold_steps 50
python train_ppo.py --pretrained_model_path pretrained/end2race.pth --output_dir post-trained/ppo_loss_sample_stride10 --seed 42 --num_updates 30 --ppo_loss_sample_stride 10
python train_ppo.py --pretrained_model_path pretrained/end2race.pth --output_dir post-trained/ppo_online_same_state_branched_return --seed 42 --num_updates 45 --online_same_state_branch_ppo
```

The first three runs contain warm-up plus 30 complete formal metric rows and 30 actor/critic checkpoint pairs; the
online run contains warm-up plus 45 formal rows and 45 pairs. All final actors are finite strict 12-key checkpoints.

### 30.2 Evaluation and data contract

The frozen U30/U45 actor of each run was evaluated on Austin, Hockenheim, MoscowRaceway and Nuerburgring, 600
deterministic ego-scope episodes per map. All 16 packages have 600 unique JSON episodes, 600 exact-key NPZ traces,
zero errors, finite numeric arrays, common trace lengths, exactly one terminal post-step row, no terminal action and
exact `ego-opp`/`ego-wall` marker agreement.

Stride10 Moscow initially had one worker SIGSEGV. The exact scenario succeeded in an isolated deterministic rerun;
its metrics/trace were restored and the original batch was re-aggregated to 600 unique episodes and zero errors. This
repairs an intermittent worker failure; it does not hide a deterministic scenario failure.

### 30.3 Results and fixed decisions

| Actor | Austin | Hockenheim | Moscow | Nuerburgring | Total | Decision |
|---|---:|---:|---:|---:|---:|---|
| Global K10 speed residual U30 | `17/350` | `18/358` | `29/384` | `21/396` | `85/1488` | positive vs BC, superseded |
| Global K10 + front-corridor K50 U30 | `25/384` | `13/389` | `18/397` | `18/396` | `74/1566` | retain high-overtake frontier |
| Stride10 direct PPO loss U30 | `34/333` | `32/340` | `44/382` | `23/397` | `133/1452` | tested instance closed |
| Online same-state branch U45 | `23/359` | `33/361` | `42/388` | `32/393` | `130/1501` | tested instance closed |

K10 to K10/K50 paired collision is `57 removed / 46 created,p=.3245`; overtake is
`33 lost / 111 gained,p=4.58e-11`. Against RW30 `73/1516`, K10/K50 collision is `38/39,p=1` and overtake is
`32/82,p=3.14e-6`. The fixed dual-frequency configuration therefore contributes a confirmed progress gain at
statistically indistinguishable collision, not a confirmed safety gain.

Online branch telemetry totals 720 states, 2,880 branch actions and 266,795 simulator steps; 2,463/2,880 branches
end at the 100-step horizon. This high unresolved fraction and general-state sampling are the observed mechanism
limits for the fixed instance. Collision-triggered one-second branch PPO in §29 was not trained and is not judged by
this result.

Decision-grade package audit and reference comparisons are retained in `ANALYSIS.md` §56.

After the recorded audit, the stride10 method was explicitly retired: its model/evaluation directories and active
CLI, rollout-mask, telemetry and test implementation were removed. This does not alter the preserved result above.

## 31. Global K10 steering-and-speed temporal exploration execution

The completed treatment used:

```bash
python train_ppo.py --pretrained_model_path pretrained/end2race.pth --output_dir post-trained/ppo_global_temporal_steering_speed_noise_hold10steps --seed 42 --num_updates 30 --steering_noise_hold_steps 10 --speed_noise_hold_steps 10 --front_corridor_speed_noise_hold_steps 0
```

`End2RaceGRUPolicy` keeps independent per-slot counters and standard residuals for steering and speed. Steering uses
`0.52*tanh(latent_mean_t + .03*z_block)` and speed uses `mean_t + .15*z_block`; both block lengths are ten simulator
steps. Actor means, observations, GRU execution, reward, GAE and direct PPO losses remain 100 Hz. The environment
does not construct a front-corridor gate when the corridor hold is zero. Evaluation remains deterministic.

The regression in `scripts/test_first_action_preference.py::exploration_test()` checks three distinct K10
blocks for both dimensions and preserves the older speed-only K10 and speed K10/K50 contracts. The full real
F110/CUDA lifecycle also completed with zero pre-update log-ratio error and a successful optimizer update.

Formal U30 performance is Austin `12/311`, Hockenheim `10/322`, Moscow `17/378`, Nuerburgring `14/387`, total
`53/1398`. Relative to speed-only K10 it produces collision `62/30,p=.0011109` and overtake
`123/33,p=2.10e-13`; it is a conservative trade rather than a useful frontier extension. The fixed instance is
closed; the CLI and temporal-steering policy state were removed on 2026-08-13, while normal per-step steering
exploration remains unchanged.

## 32. Collision-triggered current-update and canonical-BC first-action preference

### 32.1 Removed implementation

The failed general-state online branch implementation is no longer callable. The deleted path sampled 16 arbitrary
on-policy states, drew four stochastic first actions, stopped after at most 100 steps, bootstrapped unresolved returns
with the critic, and optimized a second clipped-PPO surrogate. Its U45 `130/1501` result remains in §30 and
`ANALYSIS.md` §56.4, but the following interfaces and helpers were removed:

```text
--online_same_state_branch_ppo
--collision_prefix_branch_ppo
OnlineBranchRollout
forward_independent_collection
evaluate_independent_actor_actions
```

Historical §29 documents the retired one-second branch-PPO prototype; it is not a current run contract.

### 32.2 Online collision-triggered temporary preference（历史实现，已删除）

以下只记录2026-08-12正式实验使用过的机制。该实例U30为`85/1466`并关闭；2026-08-13已从
`train_ppo.py`、`ppo/env.py`、`ppo/rollout.py`、YAML和回归脚本删除，不能把下述CLI当作当前入口。

历史CLI是一个非负比例：

```text
--online_collision_preference_step_fraction
```

Zero preserves production PPO. A positive value is mutually exclusive with a fixed first-action preference dataset.
It supports both stepwise-independent exploration and the current temporally correlated speed exploration. In the
latter case the online collector stages the same causal front-corridor gate before every actor step and carries the
post-step/reset gate into the next transition; branch simulation itself remains deterministic. Each worker stores one
episode-start runtime snapshot and the applied ego actions. On current-policy ego collision it deterministically
reconstructs snapshots 150, 100 and 50 simulator steps before contact. The parent keeps the corresponding complete
actor observation/hidden history; episodes whose beginning predates the formal collector are not labeled.

At each aligned prefix, frozen current `pi_k` supplies its deterministic noop action. The fixed physical residual
library contains steering `[-.04,-.02,+.02,+.04]`, speed `[-1,-.5,+.5,+1]`, and four `steering +/- .02 x speed
+/- .5` combinations. Every candidate changes only the first action; noop and candidates then use the same frozen
actor deterministically on their own counterfactual observations until real terminated/truncated terminal. There is
no 100-step branch horizon, critic bootstrap, sampled branch action, return ranking or old-policy branch ratio.

Terminal outcomes map to `(no_ego_collision, overtake)`: collision `(0,0)`, follow `(1,0)`, overtake `(1,1)`.
Candidate/noop pairs are retained only under strict Pareto ordering; equal outcomes produce no label. A temporary
episode replays its actor-visible 361D sequence from zero hidden and applies
`softplus(-(log_pi(good)-log_pi(bad)))` at the labeled decision positions. Candidate-preferred and noop-preferred
losses receive equal weight when both exist. Before every update with labels, beta is calibrated from median
learning-rate-weighted actor step norms so the auxiliary step fraction equals the CLI request. The pairs are reset at
the next rollout and never written to a persistent dataset. An update without labels executes plain PPO.

删除前的mechanical regression使用K10 global speed noise plus K50 front-corridor
configuration and a deterministic current-policy collision only to make the terminal outcome reproducible. It
produced 39 true-terminal branches over three prefixes, 3 strict pairs over two states, 3,915 additional simulator
steps, a finite beta, and one completed actor/critic update. Snapshot next-transition identity and a self-contained
fixed-dataset schema/sampler/loss fixture are checked in the same script; no old U44 preference panel is required.
This is an execution test, not a performance result.

### 32.3 Canonical-BC fixed source

`scripts/build_bc_first_action_preference.py` creates the independent fixed-data alternative. The actor is hard-bound
to `pretrained/end2race.pth`, map Austin, seed42 and hidden scale4. It uses only the existing canonical-BC
classification of all 10,800 Austin collision candidates and the 479 collision subset. Current BC collisions form
target sources. Controls are cache-noncollision difficult candidates that current BC actually finishes as overtake,
ordered high-speed first; ordinary/eval panels and every PPO checkpoint/trace are excluded.

Each source is branched immediately after its baseline, preventing intervening resets from invalidating runtime
snapshot state. It uses the same three leads, fixed residuals, deterministic true-terminal continuation and strict
outcome ordering as the online route. Only labeled episodes are written as schema-1 361D NPZ sequences. Manifest
provenance records canonical actor/cache hashes and explicitly records no prior PPO/eval use. A dataset is published
atomically only when both target and control labeled episodes exist; the existing loader then verifies gate verdict,
gate hash and every sequence hash.

The minimal smoke build requested one target and two difficult controls. It generated one labeled target with 10
pairs and one labeled control with 1 pair; 117 branches reached terminal in 37,221 simulator steps. The published
temporary dataset loaded and produced a finite loss. Formal construction defaults to 64 target sources and 64 safe
control sources; actual labeled counts are reported by the gate and must be inspected before training.

### 32.4 Historical commands and evidence boundary

以下命令只用于解释已完成run，不再能由当前HEAD直接执行；当前没有活动命令。

```bash
python train_ppo.py --pretrained_model_path pretrained/end2race.pth --output_dir post-trained/ppo_online_collision_first_action_preference --seed 42 --num_updates 30 --speed_noise_hold_steps 10 --front_corridor_speed_noise_hold_steps 50 --online_collision_preference_step_fraction 0.10

python scripts/build_bc_first_action_preference.py --output_dir post-trained/panels/bc_first_action_preference_v1 --target_source_count 64 --control_source_count 64
python train_ppo.py --pretrained_model_path pretrained/end2race.pth --output_dir post-trained/ppo_bc_source_first_action_preference --seed 42 --num_updates 30 --speed_noise_hold_steps 10 --front_corridor_speed_noise_hold_steps 50 --first_action_preference_dataset post-trained/panels/bc_first_action_preference_v1 --first_action_preference_step_fraction 0.10
```

These were separate arms and were not combined. Both started the only student training from canonical BC and
produced the standard 12-key actor, but both added a preference loss and therefore were not pure PPO. Both later
completed formal training and four-map 600 evaluation; final evidence is in `ANALYSIS.md` §59. The online route was
materially expensive because branch simulation was synchronous with the vector rollout.

### 32.5 Completed controlled run matrix and reconstruction boundary

The completed matrix used one explicit pure-PPO control and five treatments. All arms started from canonical BC,
trained only Austin with seed42 for 30 intended formal updates, and retained the same reward, critic, collision cache,
PPO settings and actor schema. The control was global speed K10 plus front-corridor speed K50. The treatments were:

1. the same K10/K50 control plus canonical-BC fixed first-action preference;
2. the same K10/K50 control plus online collision-triggered temporary first-action preference;
3. pure PPO with only the corridor hold changed from K50 to K75;
4. pure PPO with only the corridor hold changed from K50 to K100;
5. pure PPO K10/K50 with only `abs(opponent lateral_d) < 0.25m` removed from the gate, while the positive lateral
   OBB overlap and `(0,2m)` front-body-gap conditions remain.

Control and both preference arms evaluated U27--U30 on four maps x600; K100 and the lateral-offset-gate ablation
evaluated U30 on four maps x600. K75 stopped after U20 and had no model eval. U30 four-map totals were control
`74/1566`, fixed BC preference `67/1436`, online collision preference `85/1466`, K100 `110/1425`, and no-lateral
gate `69/1369`; exact paired results and scientific boundaries are in `ANALYSIS.md` §59. All four completed
treatments were closed and their raw run/eval products were deleted after recording; K75 was deleted as interrupted,
not as a measured failure. A later user instruction briefly created a `run.sh` for the distinct §33 BC-native scale arm,
but it was never executed and was removed during reporting cleanup.

Current mechanical regression keeps K10/K50 block timing, the fixed `.25m` lateral-offset gate, runtime snapshot and
fixed-dataset loss. The removed-lateral and online preference lifecycle assertions were deleted with their activity
paths. The numeric hold CLI remains generic even though the tested K100 instance is closed. Reconstructing a deleted
historical arm would require explicit future authorization; this document alone does not authorize doing so.

## 33. 未执行的BC-native旧规模固定第一动作偏好合同

这是冻结前曾由用户明确要求并写入根目录`run.sh`、但从未执行的合同。`run.sh`已在汇报清理时移除，
因此本节只记录历史设计，不是当前待办或执行授权。它复用§32.3的
canonical-BC builder和现有`FirstActionPreferenceDataset` loss，只把source请求规模改为168个BC
collision target与225个BC safe-overtake control，并把训练合同恢复为45 formal updates。实际labeled
episode和pair数必须由构造后的`gate_report.json`给出；请求source数不是监督标签数。

活动配置键为：

```text
first_action_preference_target_episodes_per_batch: 8
first_action_preference_control_episodes_per_batch: 8
first_action_preference_lead_steps: [150, 100, 50]
first_action_preference_action_residuals: 12个固定single-step物理残差
```

原`run.sh`的固定顺序是：

1. `scripts/build_bc_first_action_preference.py`构造
   `post-trained/panels/bc_native_first_action_preference_168target_225control_v1`；
2. 从`pretrained/end2race.pth` fresh start，只训练Austin、seed42，使用全局speed K10和2m前向走廊
   speed K50、preference step fraction `.10`，输出到新的
   `post-trained/ppo_bc_native_full_scale_first_action_preference`；
3. 固定评测U42--U45，每个checkpoint依次运行Austin、Hockenheim、MoscowRaceway和Nuerburgring
   各600 deterministic episode并保存完整trace。

截至2026-08-13，dataset目录与run目录均不存在，且没有活动训练/评测进程，所以没有label数量、训练
遥测或性能结果。Lattice-reference、原`instrument_train`场景宇宙、周期性刷新或其他数据来源也都只是历史讨论。
