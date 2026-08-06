# End2Race PPO 结构重构规格

## 0. 文档状态

本文原为已经确认的 PPO 代码结构重构计划。**本地 Phase 0--4、远端仓库镜像和两套
memory合并已于2026-07-31全部完成并通过验收。**
当前源码状态以 `HANDOFF.md` 为入口，完整等价性结果见 `ANALYSIS.md` §22；本文件继续
保留实施合同、验收边界和远端迁移步骤。

当前阶段状态：

| 阶段 | 状态 | 结论 |
|---|---|---|
| Phase 0 | 完成 | 冻结了工作树、运行状态、配置和重构前配对基线 |
| Phase 1 | 完成 | 活动代码只保留三种完整命名的探索模式 |
| Phase 2 | 完成 | PPO 已收口为五个 Python 模块和一个 YAML |
| Phase 3 | 完成 | 三模式短训逐记录、逐指标、逐 checkpoint tensor 等价 |
| Phase 4 | 完成 | 文档与本地工作树已完成最终交叉检查 |
| 远端镜像与 memory | 完成 | 本地最终复核后执行；仓库零差异，memory并集可读 |

实施时必须保持：

- actor 网络、361D actor observation 和部署接口不变；
- production reward 数学定义不变；
- scenario pool 内容、角色调度和无放回队列语义不变；
- PPO rollout、warm-up、actor/critic 分阶段更新语义不变；
- 逐步独立速度噪声、全局时间相关速度噪声和前向走廊门控时间相关速度噪声的行为不变；
- 已有模型权重仍可由 evaluator 严格加载；
- 以已经单独对齐production baseline U30的训练默认值为重构基线，不在模块搬迁中再次改值；

本次重构的目标是减少模块数量和重复配置，使职责边界清晰；不是新增算法、修改实验
变量或重新解释历史结果。

---

## 1. 目标文件结构

重构完成后，`ppo/` 只保留：

```text
ppo/
├── env.py
├── policy.py
├── reward.py
├── scenarios.py
└── rollout.py
```

通用训练记录功能精简后进入根目录 `utils.py`。`train_ppo.py` 保持为薄入口。

**`ppo/ppo_config.yaml` 保留，不计入"五个模块"。** 上面的文件树只列 Python 模块。
该 YAML 是 production 常量的单一来源，含 risk 四项尺度（`risk_longitudinal_clearance_m`
等）、`start_method`、`simulator_timestep`/`episode_horizon`、ordinary/collision 场景网格
和 `front_corridor_gate_maximum_gap_m`。**不得把它内联进 .py**：HANDOFF §3.7 的合同是
"不允许在CLI覆盖risk纵向尺度；纵向尺度固定从`ppo_config.yaml`读取0.6"，内联会破坏该
合同并让 production reward 常量失去单一可审计位置。重构后 `ppo/` = 5 个 .py + 1 个 yaml。

当前模块到目标模块的迁移关系：

| 当前模块 | 目标位置 | 说明 |
|---|---|---|
| `environment.py` | `env.py` | 单环境、opponent controller |
| `vec_env.py` | `env.py` | 一 env 一 worker、父进程场景调度 |
| `exploration.py` 的走廊几何 | `env.py` | `FrontCorridorGate` 和 Frenet 投影 |
| `policy.py` | `policy.py` | actor/critic、动作分布和探索采样 |
| `exploration.py` 的模式与时间噪声 | `policy.py` | 三种保留的速度探索模式 |
| `privileged.py` | `policy.py` | privileged feature 合同与 extractor |
| `reward.py` | `reward.py` | reward 四项和 potential shaping |
| `geometry.py` | `reward.py` | OBB、墙面和 clearance 几何 |
| `scenarios.py` | `scenarios.py` | 场景定义、生成、队列、调度 |
| `collision_classification.py` | `scenarios.py` | collision cache 和候选分类 |
| `algorithm.py` | `rollout.py` | rollout buffer、采样、PPO 更新 |
| `training_records.py` | `utils.py` | 精简后的训练记录与 checkpoint |

完成迁移并通过验证后，删除被取代的旧模块。不得先删旧文件再补调用链。

---

## 2. 五个 PPO 模块的职责

### 2.1 `env.py`

只负责运行环境和多进程环境执行：

- `LatticePlannerOpponentController`；
- `End2RaceGymnasiumEnv`；
- F110 reset/step/termination；
- actor observation 与 privileged observation 组装；
- reward 调用和 episode info 汇总；
- 前向走廊探索 gate；
- worker thread 限制；
- subprocess worker；
- `CentralScheduleSubprocVecEnv`；
- 父进程调用 `ScenarioScheduler.next(rank)`。

它不负责：

- 生成场景网格；
- collision cache 读写；
- PPO loss；
- actor/critic 网络定义；
- 速度噪声概率分布。

### 2.2 `policy.py`

这里的 “Policy” 使用 SB3 的广义含义，不只是纯神经网络结构。它负责：

- End2Race actor 的 PPO adapter；
- critic 模型结构；
- recurrent state 接口；
- actor/critic optimizer；
- steering squashed Gaussian；
- speed physical Gaussian；
- action sampling；
- action log-prob；
- `forward()` 和 `evaluate_actions()`；
- 三种保留的训练期速度探索；
- privileged feature schema、normalization 和 extractor。

根目录 `model.py` 仍保存原始 End2Race actor 网络结构。`ppo/policy.py` 负责把它接入 PPO。

### 2.3 `reward.py`

负责：

- progress reward；
- relative-progress reward；
- ego collision penalty；
- potential-based risk shaping；
- progress wraparound；
- OBB clearance；
- occupancy-map wall clearance；
- reward result/info 合同。

`geometry.py` 的实现并入这里，但不得改变数值公式、边界、单位或当前 production 参数。

### 2.4 `scenarios.py`

负责“下一局运行什么”：

- `EpisodeResetSpec`；
- `ScenarioSpec`；
- ordinary/collision 候选生成；
- startpoint 生成与隔离；
- `RoleScenarioQueue`；
- `ScenarioScheduler`；
- ordinary 异线高速重加权；
- collision cache 的严格加载、验证和写入；
- 使用初始化 actor 分类 collision candidates。

### 2.5 `rollout.py`

负责“如何收集 transition 并训练 PPO”：

- `End2RaceRolloutBuffer`；
- rollout collection；
- recurrent minibatch；
- returns/GAE；
- critic warm-up；
- actor PPO update；
- critic update；
- exploration transition telemetry；
- rollout/replay log-prob 一致性检查；
- metrics 汇总；
- checkpoint 调用。

虽然文件名是 `rollout.py`，它实际包含完整 PPO 训练算法，而不仅是数据采集。

---

## 3. `scenarios.py` 与 `rollout.py` 的边界

二者没有职责重叠：

```text
scenarios.py
    决定下一次reset使用哪个场景
        ↓
env.py
    执行环境transition
        ↓
rollout.py
    保存transition并更新PPO
```

`scenarios.py` 不保存 observation/action/reward，也不计算 advantage 或 loss。

`rollout.py` 不生成起点、对手 raceline、速度档或 collision pool，也不决定场景队列顺序。

两边共同出现 `scenario_id` 和 `env_role` 是跨层接口，不是重复逻辑：

- `scenarios.py` 创建这些字段；
- `env.py` 将其写入 episode info；
- `rollout.py` 只记录和分组统计。

---

## 4. Exploration 功能和迁移

### 4.1 当前保留的模式

当前活动训练代码应保留：

| 模式 | 语义 |
|---|---|
| `baseline` | 逐步独立速度高斯噪声 |
| `temporal_global` | 全局时间相关速度噪声：一个速度标准残差保持固定步数 |
| `corridor_temporal` | 前向走廊门控时间相关速度噪声：走廊触发后保持速度残差 |

历史条件门控高方差逐步独立速度噪声和旧所需减速度门控时间相关速度噪声已退役，
不重新加入 production 入口。

### 4.2 Baseline 高斯噪声

baseline 的速度动作由 Policy 概率分布产生：

```text
actor输出速度均值
    ↓
Normal(mean_speed, speed_physical_std)
    ↓
每一步独立rsample()
    ↓
保存action和log_prob
```

因此 baseline 高斯噪声必须进入 `policy.py`，不能放在 `env.py`、`reward.py` 或
`scenarios.py`。

### 4.3 全局时间相关速度噪声

全局时间相关速度噪声保持每步速度噪声的边际标准差不变，但同一个标准残差连续使用
固定步数：

```text
采样epsilon
→ 连续TEMPORAL_RESAMPLE_STEPS步使用
→ 到期后重新采样
```

噪声缓存、剩余步数和 block ID 属于 policy 的动作采样状态，进入 `policy.py`。

### 4.4 前向走廊门控时间相关速度噪声

前向走廊门控时间相关速度噪声分成两部分：

1. `env.py` 根据两车当前真值计算前向走廊 gate；
2. `policy.py` 在 gate 触发后启动或刷新时间相关速度噪声。

走廊判据保持当前语义：

- opponent 在 ego 前方；
- 正表面 gap 在规定范围内；
- opponent 相对 ego raceline 的横向偏移在阈值内；
- 两车横向 OBB 仍有正重叠。

`rollout.py` 继续保存：

- gate；
- temporal active；
- block ID；
- per-transition speed log-std；
- standard residual。

这些字段用于保证 rollout 时和更新时的 log-prob 合同一致。

### 4.5 为什么不移动到 `scripts/`

探索模块当前属于 production runtime library：

- `env` 计算 gate；
- `policy` 采样动作；
- `rollout` 记录并重算 log-prob。

让 production PPO import `scripts.exploration` 会把测试/诊断目录变成运行依赖，违背
`GUIDE.md` 的脚本规则。`scripts/` 只保存离线门宽扫描、暴露率分析、轨迹重放等工具。

为了满足五文件目标，删除原 `exploration.py` 文件，但不是删除探索能力，而是将其逻辑
分别并入 `env.py`、`policy.py` 和 `rollout.py`。

确定性 eval 始终使用 actor 均值，不受训练探索模式影响。

---

## 5. Collision classification 合并和 cache 简化

### 5.1 合并目标

`collision_classification.py` 完整合并到 `scenarios.py`，因为它的产物是训练场景池。

合并后的内部区段：

```text
场景数据结构
场景网格生成
场景队列和调度
cache schema/validation/load/write
候选场景分类worker
resolve_collision_scenarios
```

### 5.2 避免循环导入

`env.py` 需要在模块顶层导入 `ScenarioSpec` 和 `ScenarioScheduler`，因此
`scenarios.py` 顶层不得反向 import `ppo.env`。

collision classification worker 必须使用局部延迟导入：

```python
def _collision_worker_init(...):
    from ppo.env import make_environment, limit_worker_threads
    from ppo.policy import END2RACE_LIDAR_SIZE, STEERING_BOUND
```

这些导入只在实际重建 cache 的子进程初始化时发生。正常 cache hit 不加载分类执行依赖。

### 5.3 CLI 决策

保留：

```text
--collision_cache_dir
```

删除：

```text
--reclassify_collisions
```

两者原本不完全重复：

- `collision_cache_dir` 指定 cache 身份和路径；
- `reclassify_collisions` 强制覆盖重建。

但训练入口不需要提供“覆盖现有 cache”的第二种状态。

删除 `--reclassify_collisions` 必须覆盖完整调用链，不能只删 CLI 定义
（与 §7 删除 `detached_gru` 同样标准）：

- `train_ppo.py` 的 `add_argument`；
- `collision_classification.py` 中 `args.reclassify_collisions` 的两处使用；
- **两条 `RuntimeError` 文案中的 `use --reclassify_collisions`** ——
  identity 不匹配与 cache 不完整各一处。删除开关后这两句会指向不存在的参数，
  必须改写为"请指定一个新的空 `--collision_cache_dir`"。

### 5.4 简化后的 cache 生命周期

规则固定为：

```text
cache目录不存在或为空
→ 使用当前pretrained actor分类
→ 写入完整cache

cache完整且identity匹配
→ 严格加载

cache只存在部分文件
→ fail closed，不自动补写

cache存在但模型路径、地图或候选配置不匹配
→ fail closed，不覆盖
→ 要求使用新的collision_cache_dir
```

需要重新分类时，新命令指定一个新的空目录。不得原地覆盖已有 cache。

**已知限制：cache身份是"绝对路径 + 全JSON相等"，仓库移动到不同绝对路径后无法复用。**
当前 `classification_config.json` 存的是
`str(Path(args.pretrained_model_path).expanduser().resolve())`（实测为
`/home/haowei/Documents/End2Race/pretrained/end2race.pth`），而
`validate_collision_cache_identity` 用
`json.dumps(cached, sort_keys=True) != json.dumps(current, sort_keys=True)` 做**整体**比较。
本次远端迁移仍使用完全相同的
`/home/haowei/Documents/End2Race`，因此现有cache可以直接复用。**本次结构重构不得顺带
修改cache schema或把actor身份改成相对路径**：现有cache保存的是绝对路径，直接改变当前
配置会使完整JSON比较失败；只比较相对路径还会削弱actor身份。

如果以后确实需要支持不同安装路径，应作为独立迁移设计新schema、旧cache兼容规则和
actor内容身份，再单独验证。本次只在删除`--reclassify_collisions`后把错误提示改为
“指定新的空`--collision_cache_dir`”，保持现有cache身份语义不变。

**分类进程数直接使用 `n_envs`，不再存在独立的 `env_workers`。已确认这不会使现有 cache
失效**：`env_workers` 不在 `classification_config.json` 的身份字段内（实测该文件只含
schema、actor 路径、hidden_scale、map、racelines、collision 网格四项、timestep/horizon、
candidate_count 共 13 个键）。

历史 cache 的 `classification_summary.json` 仍保留名为 `env_workers` 的统计字段，
仅用于兼容既有 cache schema；其值现在等于 `n_envs`。它不是 CLI、不是独立执行拓扑，
也不参与 cache identity。不得为了满足字面搜索而破坏已冻结 cache 的读取合同。

cache 继续保存足以验证以下身份的信息：

- pretrained actor 路径；
- hidden scale；
- map；
- racelines；
- startpoints；
- interval；
- speed scales；
- timestep 和 horizon；
- candidate count。

不为本次重构新增 SHA-256。

---

## 6. 一 env 一 worker

### 6.1 删除的抽象

删除：

- `worker_count` 构造参数；
- `--env_workers`；
- `worker_env_indices`；
- `env_to_worker`；
- `_group_entries()`；
- 一个 worker 内维护多个 env 的 list/dict；
- 按 worker 聚合和拆分命令结果的代码。

### 6.2 新合同

```text
env rank 0 ↔ worker 0
env rank 1 ↔ worker 1
...
env rank N ↔ worker N
```

每个 worker：

- 持有一个 env；
- 持有一个固定 rank；
- 每次接收一个 action/reset 请求；
- 返回一个 env 的结果。

父进程仍持有唯一 `ScenarioScheduler`，并在 reset 时调用 `next(rank)`。不得把场景队列
移动到 worker，否则会改变全局无放回和角色调度语义。

### 6.3 必须保留的防护

- CUDA 初始化前创建 subprocess；
- worker thread 限制；
- worker exception 回传；
- broken pipe/EOF 处理；
- terminal observation；
- done 后父进程提供新场景并 reset；
- close/terminate/kill 清理；
- SB3 `VecEnv` 的 `get_attr`、`set_attr`、`env_method` 等必要接口。

一 env 一 worker 只改变执行拓扑，不改变：

```text
n_envs × n_steps
```

代表的逻辑 rollout transition 数。

**已有短运行支持拓扑本身不改变数值（2026-07-30）**：同一 seed、`n_envs=4`，分别用
`--env_workers 4 / 2 / 1` 各跑 1 个 formal update，五项指标逐位相同
（`policy_gradient_loss=0.014701`、`value_loss=0.267463`、`approx_kl=0.027928`、
`clip_fraction=0.282031`、`explained_variance_post=0.675083`）。

根因在于每个 env 的 RNG 由 **(base seed, rank)** 决定而与 worker 数无关：

```python
logical_seeds[rank] = np.random.SeedSequence([seed, 1, rank % 2, rank // 2]).generate_state(1)[0]
```

且 worker 结果由 `by_rank = {int(row[0]): row ...}` 按 rank 重组，与返回顺序无关。
这说明“一 env 一 worker”有很强的语义等价依据，但单次短运行不是所有平台和完整训练
长度上的数学证明。实施时仍按 §12.5 核对 transition 合同；若完整训练的最终模型性能
与历史运行不同，但静态、transition、首轮更新和数值有限性检查均通过，则按用户决定
**不以该性能差异阻断结构重构**，只在验收记录中如实说明。若差异来自 rank、场景顺序、
reward、done、recurrent state、log-prob 或 minibatch 数量改变，则属于实现错误，不能
用这条例外放行。

---

## 7. Policy 和 critic 清理

只移除：

```text
detached_gru
```

保留：

```text
mlp
independent_gru
priviledge_mlp
privilege_gru
```

删除 `detached_gru` 必须覆盖完整调用链：

- `DetachedGRUCritic`；
- `CRITIC_VARIANTS` 选项；
- `detached_gru_feature_size`；
- rollout buffer detached feature 数组；
- staged/current detached feature；
- policy forward/evaluate 分支；
- rollout critic input/statistics 分支。

不得借此删除或重写其他 critic，后续仍可能进行受控对比。

历史拼写 `priviledge_mlp` 暂时保留，避免破坏已有命令和 run config；拼写修正不是本次
结构重构的一部分。

---

## 8. Training recorder 和 critic checkpoint

### 8.1 Critic 的重要性

critic 对训练是核心组件，用于：

- value prediction；
- bootstrap；
- GAE/returns；
- advantage；
- critic loss。

但确定性 evaluator 和部署只加载 actor。当前 pipeline 也没有保存 optimizer、RNG、
环境队列和完整 recurrent runtime 状态，因此单独的 critic checkpoint 不能实现严格 resume。

过去评估很少使用 critic 权重，表示“逐 update 保存 critic 文件的利用率低”，不表示
critic 网络或训练指标不重要。

### 8.2 Recorder 保留内容

精简后保留：

- 输出目录非空保护；
- `run_config.json`；
- collision/ordinary scenario records；
- `episodes.jsonl`；
- `metrics.jsonl`；
- warm-up critic checkpoint；
- 每个 formal update 的 actor checkpoint；
- 每个 formal update 的 critic checkpoint。

计划删除：

- 非实验语义的开始时间；
- 与最后一个 update 重复的 `actor_final.pth`（**已核实无运行时消费者**：全仓库
  `.py`/`.sh` 中除 `training_records.py` 自身写入外无任何引用，canonical run 目录也只
  用 `update<N>/actor.pth`；仅遗留布局的 hard-neighbor 10% 目录存有一份。删除时必须
  同步更新README的run artifact结构和HANDOFF完成检查，不得留下过期合同）；
- 每次 actor 保存后都重新构造并加载一次模型；
- recorder 内递归 finite walker。

Tensor finite 检查留在 `rollout.py` 的优化边界；JSON 记录使用 `allow_nan=False`。

虽然 evaluator 不使用 critic checkpoint，现行 `GUIDE.md` 仍要求：当一次完整 U45
训练与既有 U30 训练合并为同一实验轨迹时，共享区间的 actor 和 critic 必须逐 update、
逐 tensor 一致。因此，“只保存最终 critic”会改变当前实验完整性合同，不能作为结构
重构的顺带修改。

如果后续明确决定不再使用 critic 权重进行轨迹一致性验证，可以另行讨论：

```text
每update保存actor
只保存final critic
```

但该存储策略变更必须先同步修改 `GUIDE.md`，并明确接受以后无法用 critic 权重验证
两次完整训练的共享区间。当前重构计划暂不采用它。

### 8.3 迁移到 `utils.py` 的依赖限制

`utils.py` 不得 import `ppo.scenarios`，因为 `scenarios.py` 已经 import 根目录工具，
否则形成循环。

Recorder 接受普通 dataclass、mapping 或预序列化 records，不在类型签名中依赖
`ScenarioSpec`。Torch/model 导入应限制在 checkpoint 方法内部或使用私有名称，避免
污染现有 `from utils import *` 的 evaluator。

当前 `ppo/training_records.py` 在模块顶层 import 了 `torch`、`model.End2Race` 和
`ppo.scenarios.ScenarioSpec` 三者，而 `eval_multiagent.py`、`eval_singleagent.py`
都用 `from utils import *`，且 `utils.py` **没有定义 `__all__`**。直接搬迁会同时造成
循环导入（utils ↔ ppo.scenarios）和 evaluator 启动时被迫加载 torch。
因此搬迁时必须：给 `utils.py` 增加 `__all__` 明确导出面，并把 torch/model 导入下沉到
checkpoint 方法内部。

---

## 9. `train_ppo.py` 精简

### 9.1 最终职责

`train_ppo.py` 只负责：

1. 解析运行参数；
2. 验证输入/输出路径；
3. 设置随机种子和 PyTorch 数值模式；
4. 调用 `scenarios.py` 准备场景；
5. 调用 `env.py` 构造 vector env；
6. 调用 `rollout.py` 构造 PPO；
7. 写入 run config；
8. 启动指定 formal updates；
9. 在 `finally` 中关闭环境。

删除 `from ppo.algorithm import *`，所有导入必须显式。

### 9.2 保留的 CLI

路径、实验身份和训练变量保留：

```text
--pretrained_model_path
--output_dir
--hidden_scale
--critic
--map_name
--n_envs
--seed
--collision_cache_dir
--n_steps
--batch_size
--num_updates
--actor_epochs
--critic_epochs
--gru_learning_rate
--head_learning_rate
--critic_learning_rate
--steering_latent_std
--speed_physical_std
--speed_exploration_mode
--gamma
--gae_lambda
--clip_range
```

每个 `parser.add_argument(...)` 保持为一条物理代码行，不为普通参数增加多行 help。
完整代码风格约定见 `.agents/STYLE.md`：新代码不写 `from __future__ import annotations`、
不写模块级 docstring、普通脚本函数不加类型注解、编排逻辑直接放在 `__main__` 块。
本次重构是搬迁，**风格调整与行为调整不得放在同一个 diff**。

计划形式：

```python
parser.add_argument("--pretrained_model_path", type=str, default="pretrained/end2race.pth")
parser.add_argument("--output_dir", type=str, default="post-trained/ppo")
parser.add_argument("--hidden_scale", type=int, default=4)
parser.add_argument("--critic", choices=CRITIC_VARIANTS, default="privilege_gru")
parser.add_argument("--map_name", type=str, default="Austin")
parser.add_argument("--n_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--collision_cache_dir", type=str, default="post-trained/collision-cache/pretrained_end2race_austin_collision_pool_479")
parser.add_argument("--n_steps", type=int, default=6400)
parser.add_argument("--batch_size", type=int, default=12800)
parser.add_argument("--num_updates", type=int, default=30)
parser.add_argument("--actor_epochs", type=int, default=2)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--gru_learning_rate", type=float, default=3.0e-6)
parser.add_argument("--head_learning_rate", type=float, default=3.0e-5)
parser.add_argument("--critic_learning_rate", type=float, default=3.0e-4)
parser.add_argument("--steering_latent_std", type=float, default=0.03)
parser.add_argument("--speed_physical_std", type=float, default=0.15)
parser.add_argument("--speed_exploration_mode", choices=SPEED_EXPLORATION_MODES, default=BASELINE_EXPLORATION_MODE)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--gae_lambda", type=float, default=0.995)
parser.add_argument("--clip_range", type=float, default=0.20)
```

### 9.3 删除的 CLI

```text
--env_workers
--reclassify_collisions
```

### 9.4 验证职责下沉

`train_ppo.py` 只验证：

- pretrained model 文件存在；
- output directory 合法且为空；
- collision cache directory 不与输出目录相同；
- map/path 字符串非空。

模块内部验证：

| 模块 | 验证内容 |
|---|---|
| `scenarios.py` | cache identity、场景唯一性、候选数量、队列合同 |
| `env.py` | `n_envs`、地图、环境空间、一 env 一 worker |
| `policy.py` | critic、exploration mode、std、privileged feature |
| `rollout.py` | steps、batch、epochs、LR、gamma、GAE、clip |

### 9.5 默认值已单独对齐production baseline U30

在结构重构开始前，训练入口的三个默认差异已经作为独立修改对齐：

```text
critic=privilege_gru
clip=0.20
num_updates=30
```

其余默认参数与production baseline U30已一致。后续模块搬迁必须保持这些默认值不变，不得再把
默认值调整与文件移动、职责合并或代码风格修改放在同一个行为变更中。

---

## 10. 重构后的调用链

```text
train_ppo.py
├── scenarios.prepare_training_scenarios()
│   ├── generate candidate grids
│   ├── load or build collision cache
│   └── return collision/ordinary pools
├── env.CentralScheduleSubprocVecEnv()
│   ├── ScenarioScheduler
│   └── one environment per worker
├── utils.TrainingRecorder()
└── rollout.End2RaceRecurrentPPO()
    ├── policy.End2RaceGRUPolicy
    ├── policy action distribution/exploration
    ├── env step/reset
    ├── rollout buffer
    └── actor/critic updates
```

建议的 import 方向：

```text
reward
  ↑
policy
  ↑
env ← scenarios
  ↑
rollout
```

`scenarios.py` 的 collision worker 只能局部延迟 import `env` 和 `policy`，不得形成
模块顶层环。

---

## 11. 实施顺序

按以下顺序实施，每一步都先保持行为等价：

1. 新建 `env.py`，合并单环境和一 env 一 worker vector env；
2. 将 collision classification/cache 合并进 `scenarios.py`；
3. 将 geometry 合并进 `reward.py`；
4. 将 privileged schema/extractor 合并进 `policy.py`；
5. 移动 exploration：
   - gate 到 `env.py`；
   - sampling/state 到 `policy.py`；
   - telemetry 到 `rollout.py`；
6. 将 `algorithm.py` 规范为 `rollout.py`；
7. 精简 recorder 并迁移到 `utils.py`；
8. 删除 `detached_gru` 全调用链；
9. 精简 `train_ppo.py` 和 CLI；
10. 全部验证通过后删除旧模块。

迁移期间不得同时保留两份可被调用的同名实现。每完成一项，应立即更新 import，再删除
旧入口，避免行为分叉。

---

## 12. 验收条件

### 12.1 静态检查

- 所有 Python 文件可编译；
- `train_ppo.py --help` 只包含确认保留的参数；
- 不再出现 `--env_workers`、`--reclassify_collisions`、`detached_gru` 活动入口；
- `env_workers` 只允许出现在 §5.4 所述历史 cache summary 兼容字段；
- 不再 import 已删除模块；
- `ppo/` 只剩五个目标 Python 模块加 `ppo_config.yaml`（见 §1）；
- 全仓库无 `use --reclassify_collisions` 之类指向已删除参数的文案；
- 没有 production 模块 import `scripts/`；
- `git diff --check` 通过。

### 12.2 Actor 和 evaluator

- canonical actor checkpoint 严格加载；
- actor state dict 合同不变；
- 固定 observation/hidden 输入上的 deterministic actor 输出不变；
- evaluator 无需加载 critic；
- 三种探索模式训练得到的 actor checkpoint 仍可确定性 eval。

### 12.3 Reward 和 environment

- 固定 transition 上 reward 各分量逐值一致；
- terminal/done/truncated 和 terminal observation 一致；
- privileged feature shape、名称、边界和数值一致；
- OBB/wall clearance 数值一致；
- opponent controller 行为一致。

### 12.4 Scenario 和 cache

- 相同 seed 下 collision/ordinary 场景序列一致；
- rank 奇偶角色语义一致；
- cache hit 加载的 scenario IDs 和顺序一致；
- 新空目录可以完成 classification 和原子化写入；
- partial cache、identity mismatch 必须 fail closed；
- 已有 cache 不被自动覆盖。

### 12.5 Rollout 和 PPO

在相同 seed、相同配置的短受控运行中核对：

- observations；
- actions；
- rewards；
- dones/episode starts；
- recurrent states；
- old log-prob；
- returns/advantages；
- actor/critic minibatch数量；
- 首个 formal update metrics；
- actor checkpoint tensors。

不能仅以 `n_envs × n_steps` 相同声称语义等价。

**对照仍应显式传production配置，避免以后默认值漂移。** 当前CLI默认已经按§9.5对齐为
`critic=privilege_gru / clip=0.20 / num_updates=30`；显式参数是为了钉住验收合同，不是
因为当前默认值不同。重构前后的命令不能机械共用：重构前仍有`--env_workers`，重构后
该参数已经删除。

重构前的短对照命令形式：

```bash
python train_ppo.py --output_dir post-trained/_tmp_check --critic privilege_gru --seed 42 \
  --n_envs 4 --env_workers 4 --n_steps 800 --batch_size 1600 --num_updates 1 \
  --actor_epochs 2 --critic_epochs 5 --clip_range 0.20
```

重构后使用同一命令但删除`--env_workers 4`。`n_steps=800`等于一个未提前终止episode的
8秒horizon，因此非碰撞env可以产生完整episode记录；碰撞env可能提前reset并在同一
rollout内开始后续episode，不能写成“每个env恰好一个episode”。该配置在重构前的
baseline模式基线为：

```text
policy_gradient_loss=0.014701  value_loss=0.267463  approx_kl=0.027928
clip_fraction=0.282031         explained_variance_post=0.675083
```

重构后baseline必须逐位复现这五个值。全局时间相关速度噪声和前向走廊门控时间相关
速度噪声也要各做一次重构前后配对短运行；它们与自己的重构前结果比较，不与上面的
baseline数值比较。验证完删除临时目录。

验收分两层：

1. **模块搬迁等价性**：保持原 worker 拓扑时，以上项目和 checkpoint tensor 必须一致；
2. **拓扑简化正确性**：切换为一 env 一 worker后，必须保证rank/RNG/场景顺序和全部
   transition合同一致。若只有长训练后的策略性能或最终tensor漂移，不单独阻断重构；
   若首轮transition或PPO数学合同已经不一致，则必须修复。

### 12.6 Structured exploration

分别验证逐步独立速度噪声、全局时间相关速度噪声和前向走廊门控时间相关速度噪声：

- baseline 每步独立速度残差；
- 全局时间相关速度噪声在同一 block 内保持残差；
- 前向走廊门控时间相关速度噪声只在 gate 后激活 block；
- episode reset 清除 temporal state；
- rollout 和 update 前的 log-prob 重算一致；
- deterministic eval 不使用探索噪声。

---

## 13. 明确不做的事项

本次重构不：

- 修改 production reward；
- 修改 collision/ordinary pool 内容或权重；
- 修改 actor 输入或网络；
- 新增 resume；
- 新增 shield、oracle 或 action 后处理；
- 恢复已退役的条件门控高方差逐步独立速度噪声或旧所需减速度门控时间相关速度噪声；
- 改变 temporal hold、走廊门宽或 speed std；
- 修改 production checkpoint；
- 自动选择最佳 checkpoint；
- 新建额外测试树、bash wrapper 或日志文件；
- 计算或记录新 SHA-256；
- 删除现有实验权重或 eval 结果；
- 把历史实验结论重写为新的算法结论。

只有完成以上等价性验证后，才能把该重构视为 production-safe。

---

## 14. 术语与退役探索模式清理

### 14.1 后续统一使用的名称

后续代码注释、CLI说明、HANDOFF和对用户汇报不得再用`C`、`T`、`CT`、`CT-v2`
这类单字母简称。统一使用：

| 稳定代码值 | 对外简洁标题 | 状态 |
|---|---|---|
| `baseline` | 逐步独立速度高斯噪声 | production默认 |
| `temporal_global` | 全局时间相关速度噪声 | 保留的实验能力 |
| `corridor_temporal` | 前向走廊门控时间相关速度噪声 | 保留的实验能力 |
| `conditional_white` | 条件门控高方差逐步独立速度噪声 | 已退役，活动代码移除 |
| `conditional_temporal` | 旧所需减速度门控时间相关速度噪声 | 已退役，活动代码移除 |

`temporal_global`、`corridor_temporal`是稳定的机器可读配置值，不需要为了文字风格重命名。
已有模型目录、eval目录、历史实验ID和历史脚本名同样不改名，否则会破坏结果与权重的
对应关系。历史表格首次出现时写完整标题，必要时在括号中保留原配置值；不得再用单字母
简称组织新的分析。

### 14.2 条件门控高方差逐步独立速度噪声的完整移除

> **状态：已于 2026-07-30 完成，不是待办项。** 复核结果：`ppo/` 与 `train_ppo.py` 中
> `conditional_white` / `EscalatingRequiredDecelerationGate` / `FollowingDangerGateConfig` /
> `_causal_rate` / `CONDITIONAL_WHITE_*` 的引用数为 **0**；`ppo_config.yaml` 无遗留无消费者键；
> `ppo/exploration.py` 由 659 行降至 302 行。`exploration_danger_gate_*` 遥测字段**保留**，
> 因为它由前向走廊门控共用，不是该模式专属。移除后 baseline 五项指标逐位不变，
> 全局时间相关与前向走廊门控两种模式均实测可正常训练。
> 下面的清单保留为**验收核对表**，不是重构期间的新增工作。

原清单（现作为核对项）：

- `conditional_white` mode；
- 专为该模式服务的gate、常量、配置字段和状态；
- policy sampling/evaluate-actions分支；
- rollout buffer和telemetry专用字段；
- `train_ppo.py` choices/help中的入口；
- production配置中的无消费者键；
- 只验证该活动入口的测试。

历史checkpoint、eval结果和结论不因代码入口移除而改名或改写。文档必须明确：
该方案已完成实验且因门控暴露过稀、未通过综合验收而退役；保留历史记录只为解释为何
不再采用，不表示它仍是可运行的production选项。

### 14.3 记录文档校正范围

实施时同步校正：

- `HANDOFF.md`：活动状态只写三种现存模式，不再写“保留baseline/C/T/CT-v2”；
- `ANALYSIS.md`：修正“当前代码仍保留四种模式”等过期表述，历史结果保留；
- `EXPERIMENTS.md`：历史脚本名和模式值保持原样，但叙述改用完整标题；
- `ppo_config.yaml`、`scenarios.py`、探索相关实现：把面向读者的`CT-v2`注释改为
  “前向走廊门控时间相关速度噪声”；
- `train_ppo.py`：移除“retained C/T/CT-v2 arms”这类help文字。

不得对模型权重目录或已冻结的实验结果做批量重命名。

---

## 15. 本地重构实施阶段

本计划由后续goal托管时按以下阶段执行。当前只写计划，不创建goal、不修改production代码、
不启动训练。

### Phase 0：冻结现状

1. 记录本地branch、Git工作区和现有未提交改动；
2. 记录production默认值、活动探索模式、PPO文件调用链；
3. 记录远端仓库和两套memory的只读清单；
4. 不清理、不reset、不覆盖用户已有修改。

### Phase 1：先完成退役入口和术语校正

1. 按§14删除条件门控高方差逐步独立速度噪声的活动代码；
2. 更新HANDOFF、ANALYSIS、EXPERIMENTS和代码注释；
3. 运行静态导入、CLI choices、配置消费者检查；
4. 确认逐步独立、全局时间相关、前向走廊门控时间相关三种模式仍可构造。

### Phase 2：按依赖顺序重构

严格按§11逐模块搬迁。每次只处理一个职责边界，先改import和调用链，再删除被取代
文件；行为搬迁与纯风格修改不放在同一个diff。

### Phase 3：本地验收

至少完成：

- Python编译、import和`git diff --check`；
- `train_ppo.py --help`和默认值检查；
- canonical actor严格加载与固定输入输出检查；
- reward、environment、scenario/cache固定样本对照；
- 三种活动探索模式的采样、reset和log-prob合同；
- 保持原拓扑的短受控等价运行；
- 一 env 一 worker的短受控正确性运行；
- evaluator smoke test；
- 文档与当前代码状态交叉检查。

一 env 一 worker若只在更长训练后的最终性能上出现差异，按§6.3的用户决定记录但不
阻断；任何transition语义或PPO数学合同差异仍然阻断。

### Phase 4：形成可迁移的本地快照

1. 确认重构和记录文档都处于同一工作树状态；
2. 保存最终Git状态和变更摘要；
3. 是否提交、推送或生成Git bundle以用户届时授权为准；
4. 不把未验证的中间状态同步为远端正式目录。

Phase 4是远端写入前的硬门槛。只有Phase 0–4全部完成、所有本地验收通过且本地代码冻结后，
才进入§16的远端同步收尾。远端同步后不得继续在本地或远端顺手修改代码；若远端复核暴露
真实代码问题，必须回到本地修复、重新完成受影响的本地验证，再重新执行完整镜像同步。

---

## 16. 最终收尾：远端End2Race完整同步计划

### 16.1 已完成的只读预检

目标主机：

```text
haowei@192.168.2.209
```

目标路径：

```text
/home/haowei/Documents/End2Race
```

已确认：

- SSH BatchMode可连接；
- 远端同一路径已存在且是Git仓库；
- 远端当前branch为`main`；
- 远端工作区有未提交的`run.sh`；
- 远端有本地仓库没有的`.remember/`；
- 远端具备`git`、`rsync`、`tmux`和`end2race` conda环境；
- 远端可用磁盘足以接收本地约120GB主体数据。

这些只是迁移可行性检查，不表示已经写入或同步过任何远端文件。

### 16.2 同步原则

- 这是全部本地重构、记录校正和验证结束后的最后收尾工作，不是实施过程中的中间步骤；
- Phase 0–4任一项未完成时，禁止对远端End2Race目录执行写操作；
- 本地代码在开始同步前冻结，后续只允许完成镜像、远端复核和memory迁移；
- 本地`/home/haowei/Documents/End2Race/`是仓库迁移的唯一权威源；
- 直接同步到远端同名正式目录，不创建incoming目录，不保留远端旧仓库副本；
- 使用镜像语义：同路径文件由本地覆盖，远端存在而本地不存在的文件删除；
- 远端未提交的`run.sh`和远端独有的`.remember/`不再单独保全；若本地不存在，镜像时删除；
- 包括`.git/`、`.agents/`、被Git忽略的文件、未跟踪文件、模型权重、eval结果和panel在内，
  完整复制本地End2Race目录；
- 不通过`git reset`拼装远端工作树，也不把远端改动合并回本地。

这是用户对远端仓库的明确覆盖授权，只适用于
`/home/haowei/Documents/End2Race/`。它不自动授权覆盖远端home下的Codex或Claude
memory；memory仍按§17的并集合并规则处理。

### 16.3 推荐执行流程

1. 确认Phase 0–4已经完成并冻结本地最终快照；
2. 确认没有本地训练或评估继续写End2Race；
3. 远端确认没有进程正在从目标目录训练、评估或写模型；
4. 先运行一次`rsync` dry-run，确认source/target方向正确，并检查预计删除项；
5. 在`tmux`中执行本地到远端的原位镜像同步，使用归档与删除语义，并启用大文件可续传；
6. 传输结束后再运行一次dry-run，必须没有待复制、待覆盖或待删除项目；
7. 核对Git状态、顶层结构、文件数量和主要目录大小；不为此新增SHA-256清单；
8. 直接在正式路径运行§18的完整远端验证；
9. 最后执行§17的Codex与Claude Code memory迁移和连续性验证。

同步命令必须保持明确的尾部斜杠语义：

```text
source: /home/haowei/Documents/End2Race/
target: haowei@192.168.2.209:/home/haowei/Documents/End2Race/
```

执行时可以使用`rsync -a --delete --partial`一类参数，但不得反转source与target。由于这是
约120GB的长传输，必须在`tmux`中运行；不额外创建bash wrapper或实验log文件。

### 16.4 Git的作用

Git只用于在同步前后检查仓库状态。实际迁移由文件镜像完成，因为只用`git pull`无法同步
未提交改动、被忽略文件、模型权重和eval结果。
仓库的`.agents/`当前被`.gitignore`忽略，本计划和HANDOFF类文件也必须由文件同步显式
带到远端，不能依赖Git提交自动出现。

本地工作树无论是否提交，都按文件级现状完整复制。是否向GitHub提交或push不是远端迁移
的必要条件，也不能在未获授权时擅自执行。

### 16.5 实际执行结果

2026-07-31已从本地正式路径向`.209`同名路径执行完整镜像。首轮dry-run确认删除只发生在
远端End2Race目录内；正式传输覆盖`.git`、`.agents`、模型、eval、panel和忽略文件。
完成后的第二次dry-run显示`PENDING_CHANGES=0`。

远端正式路径随后通过：

- 五个PPO Python模块加一个YAML的结构检查；
- 全部Python编译和`git diff --check`；
- `train_ppo.py --help`与三种活动探索模式检查；
- 独立导入不提前加载Torch的合同；
- `scripts/test_screen_reward_candidate.py`的60项unittest；
- production U30严格加载和真实Austin evaluator smoke。Evaluator返回状态码1
  （follow）；错误专用状态码4未出现。

远端未安装`rg`，因此远端文件枚举使用`find`，没有为验收安装新依赖。

---

## 17. Codex与Claude Code memory迁移

### 17.1 迁移边界

迁移目标是让远端agent延续项目判断和工作规范，不是复制登录态。只迁移：

- Codex：`~/.codex/memories/`；
- Claude Code：
  `~/.claude/projects/-home-haowei-Documents-End2Race/memory/`；
- 仓库内`.agents/`、`.claude/`、`.codex/`随End2Race一起同步。

不得复制：

- Codex或Claude的认证token；
- 账号配置、session cookie、API key；
- 与本项目无关的全局会话缓存；
- 运行中锁文件。

### 17.2 Codex memory

本地和远端`~/.codex/memories/`都已有独立内容，不能用单向覆盖：

1. 备份远端整个memory目录；
2. 建立memory暂存目录；
3. 以本地当前memory为主线复制到暂存目录；
4. 合并远端独有的rollout summaries、skills和extensions，遇到同名不同内容时保留两份
   待核对，不静默覆盖；
5. 校正`MEMORY.md`、`memory_summary.md`及其相对引用，使本地主线和远端独有条目都可被
   检索；
6. 不直接覆盖memory内部的`.git`；若两边都是独立历史，保留本地主历史并把远端独有内容
   作为文件级合并输入；
7. 校验索引引用存在、summary可读、没有指向已删除绝对路径的关键入口；
8. 通过后再用暂存目录替换远端memory，旧目录继续保留到用户确认。

### 17.3 Claude Code memory

本地与远端项目memory中的辅助Markdown文件不同，采用并集合并：

1. 备份远端项目memory目录；
2. 将两边非冲突的主题文件全部放入暂存目录；
3. 对双方都有的`MEMORY.md`人工去重合并，以本地当前项目状态为主，保留远端独有历史；
4. 修正过时路径、已删除模块和字母简称，确保与新的HANDOFF/ANALYSIS一致；
5. 验证远端Claude Code从同一项目路径能读取合并后的memory；
6. 不复制`~/.claude.json`、认证信息或其他全局用户配置。

因为本地和远端项目路径相同，Claude Code的项目编码目录
`-home-haowei-Documents-End2Race`无需改名。

### 17.4 实际合并结果

覆盖前备份位于：

```text
/home/haowei/memory_backups/end2race_migration_20260731/
├── codex_memories/
└── claude_memory/
```

Codex以本地最新索引为主，文件级并集后保留51份rollout summary；远端独有handoff skill
已校正为当前四份`.agents`权威文档、五模块PPO结构和`.209`主机。新增ad-hoc continuity
note记录本轮重构与迁移边界。

Claude两端主题文件全部保留，活动`MEMORY.md`已改为以当前`.agents/HANDOFF.md`为第一
入口，并明确旧PPO V1、12D critic、`.127`主机和旧远端运行政策只作历史材料。远端检查
得到10个memory文件，当前索引可读取`ppo/env.py`等新结构。

---

## 18. 远端验收和goal完成条件

### 18.1 远端代码验收

在新的正式远端目录中确认：

- Git branch、工作树和本地最终快照一致；
- `ppo/`为五个目标Python模块加一个YAML；
- 退役的条件门控高方差逐步独立速度噪声不再有活动入口；
- 文档和CLI不再用单字母简称描述活动方案；
- `end2race` conda环境下Python编译和import通过；
- `train_ppo.py --help`、production默认值和三种活动探索模式正确；
- canonical actor严格加载，固定输入的确定性输出符合本地验收；
- 最小训练smoke和evaluator smoke无异常；
- 模型、eval、panel和记录文档均在预定路径可见。

### 18.2 Memory连续性验收

- 远端Codex能检索当前HANDOFF/refactor约束和历史关键结论；
- 远端Claude Code能读取项目memory及仓库内handoff skill；
- 本地独有和远端独有的memory主题文件都没有丢失；
- memory中不含新增认证材料；
- 两套memory的项目路径均指向新的正式远端目录。

### 18.3 Goal终止规则

只有同时满足以下条件，后续goal才可标记完成：

1. 本地重构与验证完成；
2. 条件门控高方差逐步独立速度噪声的活动代码和过期记录已清理；
3. 本地最终快照已经冻结，远端同步前没有遗留代码修改；
4. 本地End2Race已作为最后收尾步骤原位镜像覆盖远端同名目录，第二次dry-run显示零差异；
5. 远端代码复核通过，期间没有直接在远端修补代码；
6. Codex和Claude Code memory完成并集合并和可读性检查；
7. 两套旧memory备份仍在；
8. 最终状态、memory备份位置和任何已知差异已向用户汇报。

遇到以下情况必须停止而不是强行继续：

- SSH身份变化或目标路径与预检不符；
- 远端仍有进程正在使用目标目录；
- dry-run显示source/target方向错误或删除范围超出目标End2Race目录；
- 磁盘空间不足或大文件传输不完整；
- 本地静态/transition/PPO合同验证失败；
- memory合并出现无法自动判定的同名冲突；
- 覆盖后远端验证失败。

远端End2Race旧内容没有回退副本，这是用户明确接受的迁移方式。Codex和Claude Code
memory仍保留覆盖前备份；这些memory备份只能在用户完成远端接管并明确确认后另行清理。
