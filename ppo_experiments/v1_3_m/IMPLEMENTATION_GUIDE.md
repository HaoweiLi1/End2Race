# End2Race PPO V1.3-M：受控更新强度下的 Margin 信号边际价值验证

**状态：** `READY_FOR_LOCAL_CODEX_IMPLEMENTATION`（owner 放行后执行）
**制定：** 2026-07-17，Claude Code session（受 owner 委托设计 v1_3 探索；执行方为本地 Codex）
**正式配置名：** `v1_3_m`
**正式训练 seed：** `20260723, 20260724, 20260725, 20260726, 20260727`（与 `v1_3_b` 完全相同，用于逐 seed 配对）
**唯一正式判定 checkpoint：** 每个 seed 的 `U8`
**开发评价面板：** Austin canonical development 600，`EGO_IDX_OFFSET=0`
**姊妹实验：** `ppo_experiments/v1_3_b/IMPLEMENTATION_GUIDE.md`（commit `641299e`，n_epochs=4 强度对照臂，由另一 Codex 会话执行）

---

# 1. 唯一问题

> 在与 V1.3-B **完全相同**的受控更新强度下（n_epochs=4、target_kl=0.010、post-update KL guardrail=0.020、原始 LR、8 updates），把 reward 中加入已标定的 clearance margin 项，能否形成受控、非偶然、跨五个 seed 一致的 actor 改善——并且相对"只加强度、不加信号"的 V1.3-B 有逐 seed 的边际增益？

V1.3-M **不是**以下任务：

- 不是提高 LR、改 critic、改 hard pool/batch/rollout/exploration std；
- 不是消费任何 holdout（本轮纯机制验证；holdout33 已作废，holdout19 归已关闭的 supervised 线预注册，均不得触碰）；
- 不是部署确认；即使 5/5 通过也只是机制结论，后续需另行预注册新面板确认；
- 不是任何形式的示范/监督数据混合（owner 硬约束：multiexpert 实测证明多 expert 风格混合监督导致行为混乱，本计划严禁引入任何示范数据）；
- 不是在 U2/U4/U8 中挑最好 checkpoint。

---

# 2. 设计推理与证据链

每条判断附来源，执行者可只读复核：

1. **碰撞指示函数无 margin 梯度**：`reward_collision=-2.0` 只在碰撞步触发一次（`ppo/reward.py`），碰前梯度处处为零；BC 运行在 clearance 尾部（audit：oriented clearance p01=0.23m、p05=0.48m）。V1.x 的 46-checkpoint churn、新增碰撞集合逐 checkpoint 几乎不重叠（`CHECKPOINT_DISTRIBUTION_ANALYSIS.json`、HANDOFF_CC §4），与"边缘 case 被无方向微扰随机翻转"一致。
2. **margin 设计已通过 P0 验证并标定**（`ppo_experiments/signal_repair/SG_P0_RESULTS.json`，commit `bf0c1c5`）：H0 24 个碰撞中 **20/24 = 83.3% 为对手贴近型**（F0 门槛 60% ✓）；常数标定选定 **margin_weight=0.02、margin_threshold=0.5m**（安全超车罚分中位数 0.0015 ≪ 其 relative 收益 0.214 的 30%，碰撞/安全分布分离 ✓）。
3. **margin 从未在受控强度下被测过**：SG-full 死于 ×10 LR 的步长过冲（U1 KL 0.048–0.093，`SG_P1_RESULTS.json`），护栏按预注册终止。该结果只否定那个 LR 配置，不构成对 margin 项的证据。
4. **V1.3-B 恰好补上了缺失的对照臂**：它在原 LR 下用 n_epochs=4 提高更新强度、不改 reward。V1.3-M 与它共用全部配置与 seeds，唯一差异是 margin 项 → 两计划合起来构成 2×1 因子设计（强度 | 强度+信号），边际归因只需 5 个新 run。
5. **微 LR 族的 null 已示范**（ALS-final，HANDOFF_CC §7）：训练侧 hard-branch 碰撞率三 seed 三个方向（+0.08/−0.06/+0.01），dev U2=15/22/22。任何新臂必须显著优于该 null。

## 2.1 预注册预测（制定者记录在案，供事后对账）

- 若 V1.3-B 受控且方向跨 seed 不一致（其结局分类 `INCONSISTENT`），预测 V1.3-M 通过主门（margin 提供了缺失的方向信息）；
- 若 V1.3-B 自身已 5/5 通过，预测 V1.3-M 的逐 seed collision 不劣于 V1.3-B；若 M 无边际增益，margin 主张降级为"非必要"。

---

# 3. 与 V1.3-B 的非冲突合同（硬规则）

1. **命名空间完全隔离**：配置 `v1_3_m`；run 目录 `runs/ppo/v1_3_m_seed<seed>/`；records `ppo_experiments/v1_3_m/`。不写入任何 `v1_3_b` 路径。
2. **实现顺序**：M1（实现）仅在 v1_3_b 的实现 commit（含 per-config `n_epochs`、`update_kl_guardrail` 字段与 plumbing）落地后开始，**复用其字段与机制，不得重复实现**；对 `ppo/config.py` 只做追加（一个配置常量 + CONFIGS 注册），在 v1_3_b 改动之后 rebase，不并发编辑共享文件。
3. **资源顺序**：M3（正式训练）默认在 v1_3_b 五个正式 run 全部到终态后启动；owner 明确批准时才允许并发，且全机正式进程总数 ≤8。不停止、不修改、不重跑任何 v1_3_b 产物。
4. **跨计划只读**：对 v1_3_b 的 results/telemetry 只读引用；其数字缺失或延迟时，V1.3-M 的主门独立成立（见 §7），配对对比降级为不可用并如实记录。
5. **评估器零改动**：`eval_multiagent.py`、`evaluate.sh`、`utils.py`、地图/raceline 资产、`pretrained/end2race.pth` 一律不动；hash 与 v1_3_b 预注册记录一致才可评估。

---

# 4. 配置定义

`v1_3_m` = v1_3_b 配置逐字段相同，仅以下两项不同：

| 字段 | v1_3_b | v1_3_m | 来源 |
|---|---|---|---|
| margin_weight | 0.0 | **0.02** | SG P0 标定锁定值 |
| margin_threshold | 0.0 | **0.5** (m) | SG P0 标定锁定值 |

共同字段（照抄 v1_3_b，此处冻结）：n_envs=16、n_steps=1600、batch_size=1600、updates=8、checkpoint_updates=(2,4,8)、n_epochs=4、gamma=0.999、gae_lambda=0.995、clip_range=0.10、target_kl=0.010、update_kl_guardrail=0.020、GRU LR=1e-6、head LR=1e-5、critic LR=3e-4、critic=C0、hard pool=H0 p0.50 with_replacement、std=0.05/0.15、SIM_DURATION=8.0。

margin 语义（已在 commit `58e8728` 实现并测试）：`reward_margin = -margin_weight * max(0, margin_threshold - clearance)^2`，clearance 为 ego-opponent oriented rectangle clearance；**opponent collision latch 生效后此项恒为 0**（与 relative 项冻结同语义）；`reward_margin` 独立记录于 info 与 telemetry。

执行者必须从 `SG_PREREGISTRATION.json` / `SG_P0_RESULTS.json` 只读复核 0.02/0.5 为唯一锁定选择；如记录含糊或与本文不一致，**停止并上报**，不得自选常数。

---

# 5. 实现规格（预计新增 <40 行）

1. **前置核验（M0）**：确认当前 checkout（owner 批准的、含 v1_3_b 实现的 HEAD）中 `ppo/reward.py`/`ppo/config.py` 的 margin 字段、SG 单测（margin=0 逐位回归等价、latch 清零、数值正确性）存在且通过；确认 `update_kl_guardrail` 与 per-config `n_epochs` plumbing 存在（v1_3_b 实现）。缺任一项 → 停止上报，不得自行补实现超出本文授权。
2. **新增配置**：`ppo/config.py` 追加 `V1_3_M = replace(<v1_3_b 常量>, name="v1_3_m", margin_weight=0.02, margin_threshold=0.5)` 并注册 CONFIGS；`_validate` 无需改动。
3. **测试**：新增/扩展一个用例：`get_config("v1_3_m")` 解析成功且 resolved_config 记录 margin 字段；margin>0 时 `PPOTransitionReward` 在构造的近距位姿上产生预期罚分（若 SG 测试已覆盖则引用之）。
4. **聚合脚本**：复制 v1_3_b 的 `aggregate_results.py` 模式到 `ppo_experiments/v1_3_m/`，输出增加 §6 的 margin 遥测与 §7 的配对对比字段。

---

# 6. 遥测与护栏（每 run）

继承 v1_3_b 全部：per-update approx_kl 序列、实际 optimizer steps（target_kl 早停可见）、post-update KL > 0.020 → run 立即 FAIL 并保留证据、clip_fraction、EV、action 统计有限性。

新增（margin 特有）：

1. `reward_margin` 每 update 均值：**若恒为 0 → 配置未生效，立即中止全部 run（实现错误）**；其随 update 的趋势入报告（预期：非零，且随策略学到更宽 margin 而幅度收窄）。
2. 全分支 overtake 完成占比：晚窗(U5–8) ≥ 0.7 × 早窗(U1–4)，否则该 run 标记 `EXPLORATION_COLLAPSE` 记为失败（margin 项的已知风险是压制超车，此为训练侧提前止损；产品级由 overtake≥340 门兜底）。
3. hard-branch(bc_ego_collision) 完成 episode 碰撞率 早窗/晚窗（对照 ALS null：+0.08/−0.06/+0.01）——**只报告，不作门**（8 updates 窗口小）。
4. U8 权重组 RMS 位移（vs BC；及 vs v1_3_b 同 seed U8，若可读）——只读诊断，对照 ε-球基线（V1.x：GRU 0.0094%/head 0.11%）。

---

# 7. 预注册判据

## 7.1 有效性（每 seed）

U8 评估：600 unique scenarios、error=0、collision+follow+overtake=600；训练无 guardrail FAIL、无 EXPLORATION_COLLAPSE、无 non-finite。基础设施 INVALID 允许同 seed 一次受控重评/重训（证据保留）；性能差不是 INVALID。

## 7.2 主门（自足，逐字镜像 v1_3_b）

五个 seed 的 U8 **全部**满足：

```text
G = fixed_collision - new_collision >= 5
ego_collision <= 16
overtake >= 340
```

5/5 通过 → `V13M_PASS`；否则 `V13M_FAIL`。不做 4/5 豁免，不以中间 checkpoint 改写结论。

## 7.3 归因对比（次级，需 v1_3_b 五 run 全部终态且 valid）

逐 seed 配对 `Δcol(seed) = col_m(U8) - col_b(U8)`。**margin 边际价值确认** iff `V13M_PASS` 且（v1_3_b 未 5/5 通过 **或** [≥4/5 个 seed Δcol<0 且 median(col_m) < median(col_b)]）。v1_3_b 数据不可用时本节记 `UNAVAILABLE`，不影响 7.2。

## 7.4 结局矩阵（预注册解读，禁止事后改写）

| v1_3_b | v1_3_m | 结论与下一步 |
|---|---|---|
| PASS | PASS 且更优 | 强度+信号最优；为 m 预注册**新面板**确认轮（owner 决定） |
| PASS | 无边际增益 | 强度已足够，margin 非必要；后续归 v1_3_b 线处理 |
| FAIL（受控但方向不一致） | PASS | **A1 证实：margin 是缺失的方向信息**；为 m 预注册新面板确认轮 |
| FAIL | FAIL（均受控） | **PPO 线证据链闭合**（微 LR/强度/强度+信号全部证伪）；建议 owner 关闭 PPO 方向，不消费任何 holdout |
| 被 guardrail 终止（不受控） | —（M3 不启动） | 同强度必然同死；停止并回 owner，附 LR 校准阶梯提案（本计划不预授权） |

---

# 8. 执行流程

| Phase | 内容 | 门槛/止损 |
|---|---|---|
| M0 | 只读核验：checkout、margin 实现与测试、SG 常数锁定记录、v1_3_b 实现 commit 与运行状态、无冲突进程 | 任一缺失 → 停止上报 |
| M1 | 追加 `v1_3_m` 配置 + 测试 + 聚合脚本；commit（"Implement PPO v1_3_m margin arm"） | 测试不过 → 不得开跑 |
| M2 | 写 `V13M_PREREGISTRATION.json`（锁 HEAD、文件 hash、配置、seeds、判据、命令）；commit | 先于任何正式 run |
| M3 | 5 seeds fresh start 正式训练（默认 v1_3_b 训练全部终态后启动；并发需 owner 批准，全机 ≤8） | guardrail FAIL / reward_margin 恒 0 → 停 |
| M4 | 每 seed 仅评 U8 于 dev 面板（`EGO_IDX_OFFSET=0`，ego scope，8 workers）；训练全部终态前不得评任何 candidate | INVALID 一次受控重试 |
| M5 | `V13M_RESULTS.json` + 配对对比 + 结局矩阵判定 + `V13M_FINAL_REPORT.md`；commit | 判定后不得追加 run |

预算（v1_3_b 完成后）：训练 5 × 8 updates（n_epochs=4，约 8–12 min/update）≈ 2 seed 并发下 3–4 h；评估 5 × ~13 min ≈ 1 h；合计约半天。

---

# 9. Records 与状态机

```text
ppo_experiments/v1_3_m/
  IMPLEMENTATION_GUIDE.md          （本文）
  V13M_PREREGISTRATION.json
  V13M_RESULTS.json
  V13M_FINAL_REPORT.md
  V13M_STATUS.json
```

状态：`READY_FOR_IMPLEMENTATION → IMPLEMENTED → PREREGISTERED → TRAINING → TRAINING_COMPLETE → DEV_EVALUATED → V13M_PASS / V13M_FAIL / BLOCKED_<reason>`。只前进；失败不得静默重试。字段惯例沿用 v1_2_reduced（schema_version/device/source_commit/completed_at/输入输出 sha256/attempts/decision inputs）。每阶段完成即 commit。

---

# 10. 停止条件

1. v1_3_b 被其 guardrail 判定不受控（M3 前置失效）；
2. margin 实现/常数记录缺失或含糊；
3. `reward_margin` 遥测恒 0；
4. 需要修改共享 evaluator/模型/资产才能继续；
5. 与 v1_3_b 或其它正式进程发生输出路径/资源冲突；
6. NaN/Inf、frozen-key drift、checkpoint 非 12-key 兼容 schema；
7. owner 停止指令。

停止时写明状态、证据路径、已耗预算与唯一待 owner 决策项。

---

# 11. 给本地 Codex 的启动指令

```text
完整读取并严格执行：
/home/haowei/Documents/End2Race/ppo_experiments/v1_3_m/IMPLEMENTATION_GUIDE.md

它是 v1_3_m 的唯一行动权威，与 ppo_experiments/v1_3_b/IMPLEMENTATION_GUIDE.md 并行不冲突：
命名空间、run 目录、records 完全隔离；同 5 seeds 用于逐 seed 配对；实现在 v1_3_b
实现 commit 之后 rebase 追加；正式训练默认在 v1_3_b 五个 run 终态后启动。

先只读核验（M0）：HEAD 与 git status、margin 实现（ppo/reward.py / ppo/config.py，
commit 58e8728 血统）及其单测、SG_P0/SG_PREREGISTRATION 中 margin_weight=0.02 与
margin_threshold=0.5 的锁定记录、v1_3_b 的实现与运行状态、当前无输出路径冲突。
禁止自动 pull/merge/reset/clean，禁止触碰 v1_3_b 与任何运行中的进程。

随后按 M1→M5 顺序执行；judgment 只认每 seed 的 U8；判据以本文 §7 为准，训练开始后
不得更改。遇 §10 任一停止条件即停，写出状态与证据，等 owner。
```
