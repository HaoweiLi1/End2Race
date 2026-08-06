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
固定50个ordinary起点、collision/ordinary双角色队列和`--reclassify_collisions`必须
保留。ordinary异线高速重加权虽然未通过production验收，但作为已证明能改变跨地图
安全/超车前沿的研究工具继续保留，改由`ppo_config.yaml`定义并默认关闭。

下面的§0.5是这些退役接口的重建合同；§2.8、§2.9、§2.11和§1.5仍保留历史测试的
逐项断言。hard-neighbor与outcome-aware的源码模块也已删除：它们没有当前调用方，
完整805池和20%臂已否决，10%臂主动终止，outcome-aware从未完成独立训练A/B。
文中出现旧模块名或flag只表示“历史/重建接口”，不表示当前源码或CLI仍支持它们。

记录时间：2026-07-30（清理前保真审计修订）。对应83个文件：18个回归侧Python文件
（16个`test_*.py` + 2个历史工具），46个实验工具Python文件 + 19个shell runner，
合计约26,700行Python。

---

## 0. 使用本文件前必读

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

**当前仓库里已经存在、不需要重建的两个文件**（2026-07-30 新增，55 个测试通过）：

```text
scripts/evaluate_scenario_panel.py        显式 ScenarioSpec 面板的确定性评估器（2026-07-30 重建）
scripts/screen_reward_candidate.py        reward 候选的合规门禁 + 学习信号量化 + 独立 oracle
scripts/test_screen_reward_candidate.py   对应回归测试
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

#### 0.5.5 保留项：ordinary异线高速重加权

不要删除`ordinary_offline_fast_fraction`配置、三路ordinary queue或对应state
round-trip合同。配置位于`ppo/ppo_config.yaml`，`null`表示production均匀ordinary
队列；历史实验值`0.6`启用三路重加权。它将ordinary分为same-line、off-line-fast
（异线且speed scale≥0.7）和off-line-slow，锁定same-line自然份额，把指定fraction分给
off-line-fast，其余给slow；只改采样权重，不改可达场景集合。比例0.6虽被near400否决，
但跨地图得到54–57 collision/1146–1155 overtake，对production 80/1142双轴占优，
仍是有机制价值的默认关闭研究工具。

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
early-stop分支及专用telemetry；常规`approx_kl_mean/max`仍保留作为诊断指标。

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

历史数字用的就是这个口径（例：B/U30 Austin600 `collision_count=14`，
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

**off-line fast 定义**（`ScenarioScheduler.is_offline_fast`）：

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

## 3. 实验工具重建规范（B 级，46 个 Python）

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

### 3.2 通用评估执行器（1 个，其他所有 runner 都依赖它）

> **已重建（2026-07-30）**：`scripts/evaluate_scenario_panel.py` 现存在于仓库中，
> 不需要按本节从零重写。重建版与原版的差异：默认 `--collision-scope ego`（面板口径，
> 见 HANDOFF §4.3）；运行前用 `get_opponent_startpoint` 逐条校验面板 `opp_idx`，
> 不一致即 fail closed；2026-08-06修正为通过`evaluate_segment(..., trace_output_path=...)`
> 直接写`--output-dir/traces`，不得先写model-derived canonical根再搬移。旧搬移实现会掏空
> 已存在的canonical trace目录，已在跨地图fresh重评中实际触发，因此不得恢复。manifest记
> `status: fresh_evaluation`和实际`device`，以区别于既有臂的
> `complete_trace_reconstruction` 包。已用它精确复现 production 的
> near400 `28/325` 与 hard73 `54/12`。

`evaluate_scenario_panel.py`—— **最应优先重建的脚本**。
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
| `analyze_structured_speed_exploration.py` (1059) | B/C/T/CT 四臂两阶段分析（先主面板配对，再建紧凑诊断 panel 做 OL1 机制重放）。常量 `ARMS`、`PRIMARY_UPDATES`、`PANELS`、`EXPECTED_BASELINE_SHA256`（钉住 B/U30 身份） |
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
| `diagnose_crossmap_temporal_corridor.py` (683) | corridor-gated temporal 处理的离线预飞：复现 B/T 的 raceline×speed 联合表、场景标签 oracle 混合；`class BatchProjector`；常量 `CORRIDOR_ABS_D_M`、`TEMPORAL_HOLD_STEPS` |

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

1. **`evaluate_scenario_panel.py`** —— 19 个 shell runner 全依赖它；输出 schema
   是所有分析脚本的输入契约。写错整条链断。
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
  当前`run.sh`不再依赖它们；
- 删除回归测试源码后不能再声称当前工作树“91 tests通过”，只能引用2026-07-30清理前的历史
  结果；任何源码再改动后都需要先重建测试或建立新的回归；
- 历史`outcome_aware_hard`源码删除后，其筛选、cache和测试合同只由本文件承担；若重建，
  不要恢复对已删除测试路径的注释引用。

### 5.1 历史产物清理：panel 与 fixed pool 的恢复边界（2026-07-30 实测校正）

清理前的第二次实物核验否定了“hard73、hard334、near400 没有存活 trace、必须重跑
21,600 候选才能恢复”的说法。三个 panel 的**物理场景身份**都能由保留的 eval trace
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
| 继续工程最小输入 | hard73、hard334、near400显式ScenarioSpec + fixed-pool wrapper | 约0.45 MiB | 已保存到`post-trained/panels/heldout_hard_v1/` |
| provenance摘要 | `design_manifest.json` + `classification_summary.json` | 约0.07 MiB | 已随panel保存；记录U30-only选择、split不相交和排除既有起点 |
| 完整选择审计 | 21,600条`candidate_scenarios.json` + `candidate_labels.jsonl` | 约21.16 MiB | 已随panel保存；用于独立复核panel选择和防treatment泄漏 |
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
| `run_structured_speed_exploration.sh` (189) | B/C/T/CT fresh训练；四臂评U10/U20/U30；另评B的U24–U29；U30生成隔离actor alias后跑诊断panel；最后执行两阶段分析 |
| `run_ctv2_corridor_temporal.sh` (240) | 先做gate零误差校验，再训CT-v2 U30，评Austin600/near400/三张跨图并分析 |
| `run_ctv2_gap1_corridor_temporal.sh` (137) | 训练唯一差异为corridor front gap 1.0而非2.0 |
| `run_ctv2_45u_fourmap.sh` (343) | CT-v2从BC训45 updates；只评U45的Austin600和三张跨图 |
| `run_offline_fast_reweight_arm.sh` (217) | CT-v2加ordinary off-line-fast重权，默认fraction 0.6；仅训练和完整性校验 |
| `run_ctv2_checkpoint_curve.sh` (135) | CT-v2评U10/U15/U20/U25；与已有U30合并分析Austin600和near400曲线 |
| `run_ctv2_late_checkpoint_stability.sh` (131) | CT-v2评U26–U29 Austin600；与U30合成band，目的为稳定性而非选点 |
| `run_ctv2_crossmap_band.sh` (124) | CT-v2 U27/U28/U29 × 三张跨图 |
| `run_ctv2_gap1_band_eval.sh` (132) | gap1 U27–U30 × Austin600 + 三张跨图 |
| `run_ctv2_offline_preflight.sh` (66) | B/T分别在Austin、near和collision-pool诊断panel评估，再做gate预飞 |
| `run_arm_band_eval.sh` (159) | 参数为`RUN_DIR RESULT_ROOT ALIAS_PREFIX UPDATE...`；每个update评Austin600、near400及三张跨图，全部保存trace |
| `run_baseline_late_checkpoint_stability.sh` (36) | B的U21–U29逐点评Austin600+near400 |
| `run_temporal_late_checkpoint_stability.sh` (29) | T的U24–U29逐点评Austin600+near400，再分析同update B/T |
| `run_crossmap_bc_u30_eval.sh` (30) | 构建三张600 panel；BC/U30各评三图；统一BC-vs-U30分析 |
| `run_crossmap_t_eval.sh` (38) | 给T-U30独立alias，评三张跨图，再与既有B和BC配对分析 |
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
