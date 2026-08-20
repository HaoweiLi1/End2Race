# End2Race PPO 讨论笔记：我们怎么接到 SB3、试了什么、最后改了什么

更新时间：2026-08-14

这份文档主要用于组内讨论，不是另一份实验规范。这里想讲清楚的不是“哪个数字最好”，而是：

1. 原来的 End2Race actor 怎样放进 PPO；
2. SB3 原本能做什么，我们又补了哪些连接；
3. critic、reward、探索频率和二次数据分别解决什么问题；
4. 最后为什么必须修改 startpoint，以及这个改动怎样改变了结果口径。

当前状态仍以 `HANDOFF.md` 为准，详细实验统计在 `ANALYSIS.md`，已经退役的实现怎样重建在
`EXPERIMENTS.md`。本文只把这些内容按讨论顺序串起来。

## 1. PPO 的整体逻辑是什么

最核心的一点是：**我们没有把原 End2Race 换成 SB3 默认的 MLP/LSTM policy。** 原模型仍然是
部署 actor，PPO 只是接在它外面做继续训练。

```text
canonical BC checkpoint（原 End2Race，严格 12-key）
        ↓
360D LiDAR + 上一步实测 ego speed
        ↓
原 pressure/speed 前端 → 原 GRU → 原 2D action head
        ↓
steering / desired speed
        ↓
F110 双车环境，收集 reward、value、log-prob、terminal
        ↓
GAE + PPO clipped actor update
        ↓
独立 critic value update
        ↓
只导出原 12-key actor，评测器仍按原 End2Race 加载
```

所以部署侧没有 critic，也没有 20 维 privilege，更没有额外 selector。最终 actor 仍只输入 361 维，
输出 steering 和 desired speed。

训练上，一次 rollout 有：

```text
16 env × 6,400 step = 102,400 transitions
```

第一次 rollout 不更新 actor，只用来 warm-up critic。之后每个 formal update 都在同一份 old rollout
上先更新 actor，再更新 critic：

```text
fixed rollout / old log-prob / returns / advantages
    → actor phase：critic 冻结，做 2 epochs PPO
    → critic phase：actor 冻结，做 5 epochs value fitting
    → 保存 actor、critic 和本轮 metrics
```

这里没有在 actor 更新后重新计算本 rollout 的 return/advantage，也没有把下一轮数据混进当前更新，
仍然是标准 PPO 的 old-rollout 逻辑。

## 2. 在 SB3 / sb3-contrib 上具体接了哪些东西

我们使用的是 `sb3-contrib.RecurrentPPO` 的生命周期，但没有直接使用它默认的 recurrent policy。
SB3 主要提供 `learn()` 主循环、callback/logger、PPO schedule、VecEnv 接口和基础 buffer 约定；下面
这些是为了适配 End2Race 补的连接。

### 2.1 Policy 接口：把 End2Race GRU 接成 SB3 recurrent policy

原模型内部是 GRU，SB3 recurrent 接口传递的是 `(h, c)`。`GRUWithLSTMStateInterface` 做的事情很
简单：真实使用 `h`，`c` 始终是同形状的零张量。这样可以进入 SB3 的 recurrent 生命周期，又没有
假装原模型内部存在 LSTM cell。

Episode reset 时 GRU hidden 清零；普通 rollout 边界不清零，因为同一 episode 可能跨过 rollout
边界。Collection 时仍按 logical slot、batch-size-one 跑原 actor，避免批处理顺序改变原模型数值；
只有训练 replay 才把有效 recurrent sequence 组成 FP32 batch。

### 2.2 动作分布：要和原评测器真正执行的动作一致

SB3 默认连续动作分布不能直接表达当前 steering 语义，所以实现了
`EvaluatorCompatibleJointDistribution`：

```text
steering latent ~ Normal(mean, 0.03)
steering        = 0.52 × tanh(latent)

speed           ~ Normal(mean, 0.15 m/s)
```

Steering 的 log-prob 包含 tanh Jacobian；speed 在物理单位里计算概率。Buffer 保存的是同一 physical
action，训练时也用同一动作重算 log-prob，避免“概率对应一个动作、环境执行另一个动作”。两维
`log_std` 都冻结，训练不靠不断放大噪声获得表面 entropy。

### 2.3 Rollout buffer 和 collector

自定义 `End2RaceRolloutBuffer` 除了 SB3 的 observation/action/reward/value/log-prob，还保存：

- actor 的 GRU hidden；
- recurrent critic 的独立 hidden；
- 每一步实际使用的 speed exploration log-std；
- sequence padding 的有效 mask；
- episode start 和 env boundary。

自定义 collector 还处理了三件原生 collector 不知道的事：

1. 按 env slot 保持原 End2Race recurrent 执行顺序；
2. 保存 K10/K50 等时间相关 speed residual 的真实概率状态；
3. 对 8 秒 time-limit 使用 terminal observation 和 continuation critic hidden 做 bootstrap，而真实
   ego collision 不 bootstrap。

### 2.4 Actor、critic 分开优化

原 actor 不是一个 SB3 默认共享 backbone。我们明确分成两个 optimizer：

```text
actor optimizer
  GRU              lr = 3e-6
  output head      lr = 3e-5

critic optimizer   lr = 3e-4
```

Actor 的 pressure 参数 `k`、speed MLP、两维 log-std 冻结；只训练 GRU 和 output head。Critic 使用
自己的参数，不与 actor 共享 optimizer。Actor phase 冻结 critic，critic phase 冻结 actor。

### 2.5 场景调度和 VecEnv

16 个 logical env 对应 16 个 worker。偶数 rank 固定是 collision role，奇数 rank 固定是 ordinary
role，因此每个 rollout 的 transition 预算正好 50/50。

SB3 worker 内 auto-reset 无法表达两个“全 worker 共享、无放回消费”的场景队列，所以
`CentralScheduleSubprocVecEnv` 在父进程统一发 reset spec：collision role 消费冻结的 479 条
collision cache，ordinary role 消费 600 条普通场景；一个池完整走完后才重新 shuffle。

### 2.6 当前代码不是一层层 wrapper

现在只有一个很薄的 `End2RaceRecurrentPPO` 子类，覆盖 `_setup_model()`、`collect_rollouts()`、
`train()` 和日志入口。真正的 buffer、collector、warm-up、actor update、critic update 都是
`ppo/rollout.py` 中的普通函数。环境 reward、走廊 gate 和 privilege critic 共用一个
`TrackProjector`，PPO 与 evaluator 的 opponent 都调用同一个 `lattice_opponent_action()`。

环境侧随后又把 gate 几何、opponent planner 的 episode 状态和共享 lattice action 调用直接收进
`End2RaceGymnasiumEnv`，删除无消费者的 wrapper、render 和重复状态，`ppo/env.py` 最终由 849 行
收至 580 行。收口后再次用同一 2-env、1,800-step、K10/K50、U1 真实运行对照，9 条 episode、
metrics 和三个 checkpoint 都逐项相同。这个结果只说明重构没有改变训练语义，不是新的模型性能实验。

## 3. 一开始固定了哪些设置

所有主要单变量实验都从同一个基础合同出发，避免 critic、reward、pool、探索一起变化。

```text
初始化 actor              pretrained/end2race.pth（canonical BC）
训练地图                  Austin
seed                      42
仿真                      RK4，100 Hz，8 s
logical env / worker      16 / 16
collision / ordinary      8 / 8 env，transition 预算 50/50
collision pool            冻结 BC collision cache，479 条
ordinary pool             50 starts × 3 opponent lines × 4 speeds = 600
n_steps                   6400
transitions / rollout     102400
batch size                12800
formal updates            30（早期 critic 对照为 20）
actor / critic epochs     2 / 5
gamma / GAE lambda        0.999 / 0.995
clip range                0.20（早期 critic 对照为 0.15）
advantage normalization   on
entropy coefficient       0
max gradient norm         0.5
steering / speed std      0.03 / 0.15
target KL                 off
```

训练只在 Austin 上进行。固定评测后来统一为 Austin、Hockenheim、MoscowRaceway、Nuerburgring，
每图 600 个 episode。绝大多数正式训练只有 seed 42，所以这里能讨论的是固定合同下的强弱和机制，
不能说已经估计出跨 seed 方差。

## 4. Critic 为什么试了这些结构，结果是什么

### 4.1 先把“4 种 critic”这个历史说清楚

这件事有两个阶段，名字很容易混在一起。

最初代码评审的四种是：

| 结构 | 输入和做法 | 当时想回答的问题 |
|---|---|---|
| `mlp` | 当前 361D observation 经 BC pressure/speed 前端，再接 `420→60→1` | 单帧信息是否已经够估 value |
| `detached_gru` | 读取 rollout 时保存的 actor hidden，detach 后 `LayerNorm→420→1` | 能否直接复用 actor 的历史表示，又不让 value loss 改 actor |
| `independent_gru` | 独立复制 BC pressure/speed/GRU，`1680→420→1` | critic 是否需要自己的可训练历史 |
| `priviledge_mlp` | 只看当前 20D simulator/map privilege，`20→120→30→1` | 不靠 LiDAR 历史，只用真值几何是否更容易估 value |

后面发现 `priviledge_mlp` 有当前几何，但没有 LiDAR/历史；`independent_gru` 有历史，但不知道精确两车
几何，所以又加了第五个组合结构 `privilege_gru`。随后 `detached_gru` 退出活动实现。当前
`policy.py` 保留 `mlp`、`privileged_mlp`、`independent_gru` 三个注释块，实际启用
`privilege_gru`。

因此，**“早期评审过四种结构”和“最后同参数正式比较的三条 critic 训练臂”不是同一张表。**
现存记录没有给普通 MLP 或 detached GRU 一张与最终 P20/default 完全同合同的 Austin600 曲线，
这里不补造数字。

### 4.2 真正可横向比较的三条 20-update 结果

三条都从同一 BC、同一 seed、同一 reward/pool 开始，使用当时的旧 Austin600 startpoint 合同；
单元格是 `collision/overtake`。

| Critic | U1 | U5 | U10 | U15 | U20 | U5--U20碰撞均值 | 后段 EV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent GRU | `19/345` | `19/339` | `27/339` | `29/330` | `34/329` | `27.25` | `0.911` |
| Privileged MLP | `24/343` | `25/345` | `25/347` | `21/349` | `25/360` | `24.00` | `0.694` |
| **Privilege GRU** | **`16/350`** | `18/343` | `18/339` | **`14/349`** | **`14/349`** | **`16.00`** | **`0.912`** |

这张表给出的解释比较直接：

- Independent GRU 的 value EV 不差，但 actor 结果随训练明显恶化，说明“有历史”不等于 value 信号
  就能正确引导策略；
- Privileged MLP 有精确几何，但没有时序 LiDAR/运动历史，P20 本身没有补掉这个缺口；
- Privilege GRU 同时保留历史表示和当前几何，在这轮里碰撞最低、后段更稳定、EV 也高，所以选它。

但也不能说它已经稳定解决所有碰撞。U15 和 U20 都是 14 次碰撞，却只共享 8 个场景：U20 修复了
6 个，同时又新造 6 个，Jaccard 只有 0.40。稳定的是总数区间，不是失败身份。

### 4.3 最后选中的 privilege GRU 结构

```text
360D LiDAR ─→ copied BC pressure k ─┐
previous speed ─→ copied speed MLP ─┴→ independent copied GRU → hidden 1680D
                                                        ↓ Linear(1680,420)
20D current privilege ─────────────────────────────────→ Linear(20,420, no bias)
                                                        ↓ 相加 → ReLU → Linear(420,1)
```

20D projection 的权重初始化为零，所以刚初始化时 privilege 不会突然改变 value；训练再决定怎样使用
这些量。Actor 完全看不到后 20 维，最终 checkpoint 仍是原来的 12 keys。

### 4.4 20 个 privilege 分别是什么、为什么选

这些量都是**当前动作执行前**的 simulator/map 状态，不含 sampled action、下一状态、未来碰撞、
terminal outcome。

| 类别 | 字段 | 物理含义 |
|---|---|---|
| 两车相对位置 | `delta_s` | 环形赛道上 ego 相对 opponent 的纵向 progress；正值表示ego领先，负值表示落后 |
| 两车相对位置 | `relative_lateral` | 在 ego 车体坐标下，对手在左还是右、横向偏了多少 |
| 两车相对姿态 | `sin/cos_relative_heading` | 两车朝向差；用 sin/cos 避免角度在 ±π 跳变 |
| 两车相对速度 | `relative_long_velocity` | 对手相对 ego 的纵向速度，回答“距离在拉开还是闭合” |
| 两车相对速度 | `relative_lat_velocity` | 对手相对 ego 的横向速度，帮助识别并线/侧向接近 |
| 两车旋转 | `relative_yaw_rate` | 对手 yaw rate 减 ego yaw rate，表示相对旋转趋势 |
| Ego 动力学 | `ego_speed` | 当前实测 ego 速度，不只依赖 actor 输入中的上一时刻速度 |
| Ego 动力学 | `ego_yaw_rate` | 当前旋转速度，帮助识别甩尾或快速转向阶段 |
| Ego 动力学 | `ego_steering_angle` | 模拟器内部当前 steering 状态 |
| Ego 动力学 | `ego_slip_angle` | 车身朝向和速度方向的差，表示侧滑状态 |
| 两车净空 | `obb_longitudinal_clearance` | 两个车辆 OBB 表面在 ego 纵向轴上的净空 |
| 两车净空 | `obb_lateral_clearance` | 两个车辆 OBB 表面在 ego 横向轴上的净空 |
| 墙面净空 | `wall_clearance` | ego 车身外轮廓到地图墙面的最小距离 |
| 赛道余量 | `left_body_margin/right_body_margin` | 考虑车长、车宽和朝向投影后，车身左右各剩多少可用宽度 |
| 赛道姿态 | `sin/cos_track_heading_error` | ego 朝向相对当前赛道切线的误差 |
| 赛道几何 | `current_curvature` | 当前 raceline 曲率 |
| 赛道几何 | `lookahead_mean_curvature` | 前方 1 m 的平均曲率，给 value 一个短期弯道预告 |

相对位置、速度、yaw、steering、slip 和曲率都按固定物理尺度归一化。三项 clearance 不再硬截断，
而用：

```text
normalized = distance / (distance + half_response)
half_response = 0.6 m longitudinal / 0.2 m lateral / 0.2 m wall
```

这样近距离分辨率高，远距离仍保持单调，不会大量精确饱和到 1。注意这些 0.6/0.2/0.2 也是
reward risk 的物理尺度，但 critic 的 soft normalization 和 reward 公式是两件事。

## 5. Reward 是怎么设置的，改 reward 得到了什么

### 5.1 当前默认 reward

当前每一步只有四项：

```text
r = 0.01 × ego_progress_delta
  + 0.02 × (ego_progress_delta - opponent_progress_delta)
  - 2.0 × first_ego_collision
  + gamma × Phi(next) - Phi(current)
```

如果 opponent 已经碰墙，relative-progress 项停止累计，避免奖励 ego 继续超过一个已经失效的对手；
但 opponent-only collision 不会终止 ego episode。

风险 potential 使用车辆 OBB 净空和墙面净空：

```text
vehicle_distance = hypot(longitudinal_clearance / 0.6,
                         lateral_clearance / 0.2)
wall_distance    = wall_clearance / 0.2
distance         = min(vehicle_distance, wall_distance)
Phi              = -0.05 × max(0, 1 - distance)^2
```

它是 potential-based shaping，主要重新分配风险信用，不直接把某种动作写成规则。真实 terminal 令
`Phi(next)=0`；time-limit 保留物理 next potential 并做 value bootstrap。

### 5.2 Reward 实验结果

下面这些是 startpoint 修改前的历史 Austin/困难面板结果，适合解释 reward 机制，不应与后面的新
Austin600 数字逐场景混算。

| Reward 改动 | 关键结果 | 我们怎么理解 |
|---|---|---|
| 默认四项 | U30 Austin `14 collision / 366 overtake`；600/600 分量求和和一次性碰撞罚通过 | 公式实现没发现错误，但风险信号对甩尾通常偏晚 |
| Post-pass `q²`, LR1 | `14→20` collision，甩尾 `4→8`，overtake `366→360` | 车尾净空通常在 yaw/slip 已经起势后才触发，惩罚的是后果 |
| Post-pass `q`, LR1 | collision仍 `14`，甩尾 `4→2`，overtake `366→354` | 有局部方向改善，但靠降速/丢超车换来，不能接受 |
| Post-pass `q²`, LR3 | `15` collision、`368` overtake，甩尾 `4→7` | 加大学习率也没有稳定解决时机问题 |
| Risk L12 | 把纵向 risk 尺度 `0.6→1.2 m` | 训练碰撞率和最小净空明显改善，但最终目标分布没有稳定净收益 |
| Following-response | required-deceleration/escalation 离线门覆盖目标 `6/7` | clean同线follow误触 `21/192=10.94%`，且没有同走廊成功超车正对照，所以没有进入训练 |

L12 是最值得说明的负结果：它不是完全没作用。Updates 11--15 的 collision-role 训练碰撞率从
`56.92%` 降到 `38.31%`，平均最小净空从 `0.1143 m` 增到 `0.1821 m`。但在未见起点 hard
场景里，收益随 interval 从 8、10、12 到 15 逐渐消失；目标 interval-15 恰好是
`43→43`，消除/新造 `10/10`。在本来能贴身通过的 near-miss400 上，碰撞 `30→40`，超车
`330→304`。所以结论不是“risk shaping 无效”，而是**它确实让车更保守，但主要作用在错误的
regime，并干扰了原本成功的贴身通过。**

最后保留默认四项、纵向尺度 0.6；Post-pass 和 L12 的活动入口都已删除。

## 6. 前向走廊，以及 speed / steer 频率实验

### 6.1 为什么会想到保持 speed 噪声

默认训练每 0.01 s 都重新采样一次 speed noise。对于同线追车，真正需要的动作往往不是“某一步
突然减速”，而是连续约 0.5 s 都保持偏低的目标速度。逐步独立白噪声能采到单步减速，但很难形成
连续同方向序列，所以先测试了时间相关 speed residual。

历史 8 个同线碰撞目标上：

| 探索方式 | 8个目标的结果 | 指令速度曾低于对手 |
|---|---|---:|
| 默认逐步独立 speed noise | 7 ego-opp + 1 ego-wall | `0/8` |
| 条件白噪声放大 | 3 follow + 2 overtake + 3 collision | `3/8` |
| **全局 speed residual 保持50步** | **7 follow + 1 overtake，8个全部救回** | **`4/8`** |
| 条件保持50步 | 4 follow + 1 overtake + 3 collision | `1/8` |

这证明 0.5 s 相关块真的让 actor 学会了持续减速，不只是训练 metric 变好。

### 6.2 前向走廊 gate 是什么

我们不希望所有状态都变得保守，所以把 K50 speed residual 限在一个当前、因果的前向走廊：

```text
opponent 在 ego 前方
两车车身纵向表面 gap < 2.0 m
opponent 相对 ego raceline 横向偏移 < 0.25 m
两车横向投影有正重叠
```

门只决定训练时 speed residual 保持多久，不输入 actor，也不改变 deterministic eval。2 m 门在
same-line episode 上选择性很好；收窄到 1 m 后曝光从约 `17.53%` 降到 `4.43%`，同线收益也基本
消失，所以保留 2 m。

### 6.3 它学会了什么，又把问题转移到哪里

历史匹配 trace 的分解很一致：

| 面板 | Same-line collision | Off-line collision | 合计 |
|---|---:|---:|---:|
| 跨地图1800 | `66→22` | `14→24` | `80→46` |
| Austin600 | `8→5` | `6→15` | `14→20` |
| Near400 | `4→0` | `24→35` | `28→35` |

也就是说，同线追尾确实下降，但异线高速超车的新碰撞上升。走廊 gate 限制的是采样位置，actor
参数仍然是共享的，因此学到的持续减速倾向可以外溢到门外。这个解释与结果一致，但不能说已经唯一
证明了内部因果。

### 6.4 后来真正做的频率对照

Startpoint 修正后的四图 2,400 场景结果如下；这里 `K10=0.1 s`，`K50=0.5 s`：

| 训练期探索 | Steering 频率 | Speed 频率 | 四图 `collision/overtake` | 看到的现象 |
|---|---|---|---:|---|
| 默认 production | 每步独立 | 每步独立 | `94/1508` | 当前默认基线 |
| 全局 speed K10 | 每步独立 | 全局保持10步 | `85/1488` | 比BC好，但超车没有形成新前沿 |
| **走廊外K10、走廊内K50** | 每步独立 | 外10步、内50步 | **`74/1566`** | 相对全局K10显著增加超车，安全改善未确认 |
| 全局 steering+speed K10 | steering也保持10步 | speed保持10步 | `53/1398` | 碰撞少，但主要变成保守follow，超车损失很大 |
| 走廊K100 | 每步独立 | 门内保持100步 | `110/1425` | 保持太久，两轴都恶化 |
| 走廊K75 | 每步独立 | 门内保持75步 | 只到U20 | 没有U30四图结果，不能写成失败 |
| 去掉0.25 m lateral gate | 每步独立 | 更大范围进入K50 | `69/1369` | 扩大曝光没有得到安全超车，超车显著下降 |

K10/K50 相对全局 K10 的同场景配对是：collision removed/created `57/46, p=.324`，overtake
lost/gained `33/111, p=4.58e-11`。所以它明确提高了超车，不能写成已确认减少碰撞。

Steering K10 则是另一个很清楚的结果：相对 speed K10，碰撞显著下降，但超车从 `1488` 降到
`1398`，follow 增加 122。它更像整体降低动作变化和速度，而不是学会安全超车。当前代码因此恢复为
steering 每 0.01 s 独立采样，只保留 speed 的频率控制。

## 7. “二次数据”实验到底做了什么

这里把二次数据理解为：**先用已有 actor 跑出失败/安全轨迹，再从这些轨迹构造新的采样分布或监督
信号。** 它不是简单再加一张地图。

### 7.1 先改训练数据分布：hard pool 和 ordinary 重加权

默认 479 条 collision cache 是 canonical BC 在 10,800 个候选场景中真实碰撞的固定集合。
我们试过扩大和重排训练数据：

| 数据实验 | 结果 | 说明 |
|---|---|---|
| Full hard-neighbor 805 | Austin U45 `12→17`；U25以后平均 `13.0→21.4` | 困难覆盖增加，但没有给 actor 正确动作 |
| Hard-neighbor 20% | U35/U40/U45 `27/20/19`，同时损失超车 | 低比例混合仍没有通过 |
| Hard-neighbor 10% | 只完成到U20统一评测 | 用户停止，晚期性能未知 |
| Ordinary starts `50→150` | Austin `14→35`，near `28→55`，hard `54→38` | 对 hard 专门化，但主分布和near明显恶化；且覆盖次数变为原来的1/3 |
| Ordinary异线高速权重 `33.3%→60%` | 当前四图U30 `73/1516` | 收回超车，但历史near400碰撞 `63--77`，保守度被转移到贴身通过场景 |

这些实验说明：只增加“更难”或“更需要超车”的数据，能移动策略偏向，却不会自动告诉 actor 在困难
状态应该采取哪一个动作。

### 7.2 再从模拟器回报构造 first-action preference

这条实验不是加一个部署 selector，而是从旧 U44 的失败状态和匹配安全状态出发：同一个 snapshot
只替换第一步动作，后续恢复同一个冻结 actor，跑到 terminal。只有候选动作在
`(不发生ego碰撞, 是否超车)` 上严格优于原动作，才形成 good/bad pair。

最终数据是：

```text
target   46 episodes / 83 states / 337 pairs
control  19 episodes / 34 states / 103 pairs
total    65 episodes / 117 states / 440 pairs

candidate preferred 200
noop preferred      240
```

训练时在普通 PPO actor loss 上增加：

```text
softplus(-(log pi(good) - log pi(bad)))
```

Beta 只在第一次 formal update 前按 PPO/preference 的实际梯度量级标定一次。最终 actor 仍是原 361D、
12-key 模型，不携带 snapshot、future result 或辅助 head。

结果是 U42--U45 四个 checkpoint 都形成新前沿，主点 U44 为 `49 collision / 1530 overtake`。
它优于旧前向走廊 U44 的 `62/1478`，也优于 ordinary 重加权 U30 的 `73/1516`。这说明二次数据
最有用的地方不是“再放更多失败”，而是把**哪一个第一步动作在同一状态下更好**直接写进 actor。

但这条线也不应讲成已经解决：数据来自上一代 U44，存在闭环；只有 seed42、Austin 训练；49 个
残留碰撞中 47 个还是车辆碰撞，40 个是 side/rear 接触。当前活动源码已移除该训练分支，保留的是
成功 checkpoint、结果和历史实现合同。

另外两条二次监督对照没有成功：canonical-BC 固定来源偏好 U30 为 `67/1436`，同 update 在线
collision-triggered 偏好 U30 为 `85/1466`。Online same-state branched PPO 虽然真实生成了
720 states、2,880 branches，最终 U45 是 `130/1501`。因此不是“有模拟器反事实数据就一定有效”，
关键仍然是状态来源、pair 质量和这个信号是否直接更新最终 actor。

## 8. 最重要的改动：Startpoint 修改前后到底发生了什么

### 8.1 原 End2Race 的取点逻辑

每条 raceline CSV 的最后一行会重复第一行的物理 `(x,y)`，只是最后一行的累计进度 `s` 是完整
赛道长度。原评测取点是：

```python
max_waypoints = len(waypoints)
np.linspace(0, max_waypoints - 1, 50, dtype=int)
```

这会同时选中 index 0 和最后一个 closing index。名义是 50 个 index，实际只有 49 个物理位置。
四张地图都存在这个问题。

Austin `raceline1.csv` 有 2,097 行，最后一行是第一物理点的闭合副本；真正唯一 waypoint 只有
2,096 个，合法索引是 `0..2095`。旧 `ego_idx=2096` 在 12 个真实采集 job 中都触发过越界。

更麻烦的是，不同调用路径对末端 index 的处理不一致：有的先 modulo，有的先做 opponent mapping。
旧 same-line 公式在 `ego=0` 和 `ego=2096` 时曾得到 opponent index 15 和 14；对应初始 pose 约差
`0.202 m / 0.0525 rad`，LiDAR 首帧有 17 条 beam 不同，第一步 action 也不同。所以它不只是
“JSON key 重复”，而是能改变实际 episode。

### 8.2 当前统一逻辑

现在先明确 raceline 的环形语义：

1. 断言最后 `(x,y)` 和第一点相同；
2. 用最后一行的 `s` 作为赛道总长 `L`；
3. 实际 waypoint 永远使用半开集合 `data[:-1]`；
4. 在环形 progress 上均匀放目标，再选圆周距离最近的 waypoint。

```text
target_k = (offset_s + k × L / N) mod L
distance = min(|s_i - target_k|, L - |s_i - target_k|)
index_k  = argmin_i distance
```

评测直接取 50 个整格：

```text
eval = get_circular_startpoints(..., 50, offset=0)
```

训练使用半格交错：

```text
ordinary 50   = get_circular_startpoints(..., 100, 0)[1::2]
collision 100 = get_circular_startpoints(..., 200, 0)[1::2]
```

也就是 eval 在每个整格，train 在两个 eval 格之间。当前 exact index 没有重叠，但空间上仍然相近；
8 秒 episode 又会行驶约几十米，所以 Austin600 不能叫独立地图盲测。

Opponent 也统一处理：同 raceline 直接复用 ego index；不同 raceline 先按 `(x,y)` 找最近 waypoint；
最后才在对手的唯一 waypoint 数上加 interval 并 modulo。

### 8.3 四张地图修改前后的实际唯一性

| 地图 | CSV行数 | 唯一waypoint | 旧“50点”的物理唯一数 | 当前eval | 当前ordinary train | 当前collision candidates |
|---|---:|---:|---:|---:|---:|---:|
| Austin | 2097 | 2096 | **49** | **50** | 50 | 100 |
| Hockenheim | 1798 | 1797 | **49** | **50** | 50 | 100 |
| MoscowRaceway | 1610 | 1609 | **49** | **50** | 50 | 100 |
| Nuerburgring | 2229 | 2228 | **49** | **50** | 50 | 100 |

四图 50 eval starts × 3 opponent racelines 的 opponent index 已全部检查为合法。Austin ordinary
生成 600 个唯一 scenario ID；collision candidate 是 100×3×4×9=`10,800` 个唯一 ID。

### 8.4 这个改动怎样改变结果

最直接的例子是同一个 canonical BC 在 Austin600 上：

| Startpoint 合同 | Collision | Overtake | Follow | 说明 |
|---|---:|---:|---:|---|
| 旧 `linspace`/闭合端点合同 | `22` | `344` | `234` | 只有49个唯一物理起点；历史结果 |
| **当前 circular/半开合同** | **`33`** | **`339`** | **`228`** | 600唯一场景；28 ego-opp + 5 ego-wall |
| 数值变化 | **`+11`** | **`-5`** | **`-6`** | 这是评测分布变化，不是BC权重变化 |

当前 BC 还有 1 个 opponent-wall event，但该 episode 后来仍发生 ego-opponent collision，因此它是
事件标记，不额外增加 headline collision。

当前同一 startpoint 合同下，Production U30 的 Austin 是 `14/366/220`。所以现在可以说“当前 BC
到当前 U30 是 collision `33→14`、overtake `339→366`”；不能拿旧 BC 的 `22/344/234` 与当前 U30
作逐场景配对。即使某个模型修改前后总数碰巧一样，scenario identity 也已经改变，不能据此说
startpoint 对它没有影响。

这也解释了为什么前面 critic、reward、早期走廊实验会保留旧数字：它们仍然能说明当时固定面板内的
机制，但不能直接和当前四图结果拼成一条“持续提升曲线”。Startpoint 改动以后，最终保留模型都按
同一新合同重新看：

| 模型 | Austin | Hockenheim | Moscow | Nuerburgring | 四图合计 |
|---|---:|---:|---:|---:|---:|
| Canonical BC | `33/339` | `27/343` | `43/373` | `26/390` | `129/1445` |
| 当前 Production U30 | `14/366` | `26/356` | `32/385` | `22/401` | `94/1508` |
| 前向走廊 K50 U44 | `18/344` | `16/347` | `15/390` | `13/397` | `62/1478` |
| Ordinary异线高速重加权 U30 | `16/368` | `17/368` | `17/389` | `23/391` | `73/1516` |
| First-action preference U44 | `8/372` | `9/375` | `18/386` | `14/397` | `49/1530` |
| 全局 speed K10 U30 | `17/350` | `18/358` | `29/384` | `21/396` | `85/1488` |
| 走廊外K10、走廊内K50 U30 | `25/384` | `13/389` | `18/397` | `18/396` | `74/1566` |

因此 startpoint 修改带来的核心结果不是“碰撞统一变多或变少”，而是：

- 消除了首尾重复和越界；
- 统一了训练、评测和 opponent mapping 的物理语义；
- 重新定义了每个 episode 的身份；
- 让当前模型可以在同一新 panel 上公平配对；
- 同时切断了旧 panel 与新 panel 的逐场景可比性。

## 9. 目前怎么理解整个实验过程

如果把所有实验压成一条讨论结论，大概是：

1. 用 SB3 做 PPO 是可行的，但必须保留原 End2Race GRU、动作分布、hidden 和 evaluator 合同，不能
   直接套默认 policy。
2. Privilege GRU 比“只有历史”或“只有当前真值几何”更合适；privilege 帮 critic 估值，不替 actor
   做决策。
3. 默认 reward 没有明显实现错误。Post-pass 太晚，L12 有效但脱靶，简单加强惩罚不会自动解决
   侧后接触。
4. 时间相关 speed exploration 真能教会持续减速，但共享 actor 会把保守性迁移到异线高速超车。
5. 改 speed/steer 频率能移动安全--超车折中；K10/K50 提高超车，steer K10 则主要让整体更保守。
6. 只增加 hard 数据或重加权只能移动失败分布。二次数据真正有效的一次，是把同状态下第一步动作的
   simulator return 排序直接写进最终 actor。
7. Startpoint 修正不是外围清理，而是评测合同变化。旧结果仍可解释历史机制，但最终模型比较必须在
   当前 circular startpoint 下重新建立。

当前 production 仍是 U30 `94/1508`；first-action preference U44 是已有的最好候选 `49/1530`，
但没有自动替换 production，也还没有达到碰撞 `<40`、超车 `>1500` 的联合目标。

## 10. 代码从哪里看

| 内容 | 当前文件 |
|---|---|
| SB3入口和薄适配类 | `../train_ppo.py` |
| Rollout buffer、collector、warm-up、actor/critic update | `../ppo/rollout.py` |
| End2Race policy适配、动作分布、P20、critic | `../ppo/policy.py` |
| F110环境、reward调用、走廊gate、VecEnv | `../ppo/env.py` |
| 四项reward与净空计算 | `../ppo/reward.py` |
| Collision/ordinary场景调度 | `../ppo/scenarios.py` |
| 当前固定内部参数 | `../ppo/ppo_config.yaml` |
| Circular startpoint和opponent mapping | `../utils.py` |
| 固定评测入口 | `../evaluate.sh` |
