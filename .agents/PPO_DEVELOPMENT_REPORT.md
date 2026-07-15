# End2Race 从模仿学习模型到 B3 PPO 的开发汇报

状态：**B2 首次目标对齐 PPO 评估已完成并失败；B3 已实现、审阅 GO，尚未运行**

更新日期：2026-07-13

当前分支：`chore/commit-evidence-pipeline`

当前 B3 边界提交：`21085bc`

## 0. 一页结论

项目从已经会驾驶的 End2Race 模仿学习模型（下称 BC）出发，尝试用 PPO
降低碰撞，同时保持 BC 的超车能力。固定产品目标始终是词典序的：

1. corrected overtake 不低于 BC；
2. 在满足第 1 条的候选中，将 any-agent collision 降至相对 BC
   `RR <= 0.70`。

截至目前，PPO **不是没有效果**。历史 residual PPO 产生过局部改善，B2
更在当前 288 场景开发集上将部分候选的碰撞从 BC 的 24 次降到 8–9 次。
但这些安全改善伴随严重超车损失：B2 最安全的 C 臂把超车从 138 次降到
95/88 次。因此六个 B2 候选全部失败，没有选臂，也没有打开 fresh/final
pool。

B2 同时揭示一个关键实现缺陷：PPO 优化的是 raw-logit Bernoulli 采样策略，
产品评估却使用独立的 centered threshold。所有候选的 standard deterministic
intervention 都是 0，而 centered primary rule 产生了 24,379–29,388 次介入。
部署决策面并不是 PPO 所优化分布的 mode，训练期 dual 也没有观察到部署期
的大量介入和超车损失。

B3 因此只做一个主修：让 rollout sampling、old/new log-prob、checkpoint
replay 和 deterministic deployment 使用**同一个 effective-logit 分布**。
B3 不放宽超车门，不改 reward 权重，不恢复 warm-start，不打开 final pool。
实现与 CPU 合同测试已完成，独立边界审阅为 GO；下一步才是六个 40 轮
learner 和第一次 B3 目标评估。

## 1. 起点：End2Race BC 与历史 PPO 基线

### 1.1 BC 提供什么

BC 是主驾驶策略：它已经学会跟随、赛车线控制和超车。后续 residual PPO
没有从随机驾驶策略开始，而是在 BC 输出上添加有界的小幅转向/减速修正。
因此真正困难的不是“让车动起来”，而是：

- 只在危险且可恢复的交互中介入；
- 避免在原本安全的超车中误刹或乱转向；
- 避免 PPO 通过减少交互来伪装成安全改善。

不同阶段使用不同冻结数据面板，数字不可直接拼接：

| 证据面板 | BC collision | BC overtake | 用途 |
|---|---:|---:|---|
| D0.1 canonical primary，N=3036 | 170 | 1792 | 历史模型与诊断底座 |
| B1/B2 opened-development，N=288 | 24 | 138 | 当前策略开发和配对门 |

### 1.2 历史 PPO 已经存在

仓库里的 `train_ppo.py` 是完整、实际运行过的 PPO 循环。项目从来不是
“一行 PPO 都没有”；后来缺失的是把 B+ 的层级动作、执行动作 log-prob、
双目标 GAE、overtake dual 和配对 KPI 评估接起来的专用 runner。B2 已经
补齐这条接线。

## 2. 按时间梳理做过的 PPO 尝试和测试

### 2.1 早期全参数 PPO：critic、credit、gate、anchor（07-03 至 07-05）

主要尝试包括 privileged critic、碰撞 credit assignment、延长训练、
pre-overtake BC anchor 和 speed-anchor。代表性 OL1 结果：

| 方案 | collision | overtake | 结论 |
|---|---:|---:|---|
| BC OL1 baseline | 2.5% | 1.5% | 参考 |
| privileged critic PPO | 39.5% | 3.0% | 超车略升但碰撞灾难性增加 |
| PreBC10 | 22.5% | 3.0% | anchor 有稳定作用，但远差于 BC 安全性 |
| Anchor75 | 29.0% | 4.5% | 提高超车，但安全不可接受 |

训练延长到 Resume600 后 critic explained variance 仍较高，但确定性碰撞从
25.0% 恶化到 38.5%。这排除了“只要继续训练就会自动恢复安全”的解释。
事故分解还显示主要瓶颈是**超车前追尾/并行接触**，不是最初猜测的
post-overtake 稳定性。

### 2.2 冻结 BC 的 residual PPO：D1、旧 D2、D4（07-04 至 07-09）

为了保护 BC 驾驶能力，后续改为冻结绝大多数 BC 参数，只训练小残差头；
速度残差通常限制为 `[-1.0, 0.0] m/s`，即只允许减速，不允许正向加速。
尝试过：

- speed freeze / residual action；
- advantage normalization 与更大 rollout；
- canonical OL1 curriculum；
- hard-start、offset 和 speedscale 课程；
- lateral/progress gate 几何；
- clearance、TTC/closing 和 reward shaping；
- 不同 residual margin、seed 与 checkpoint iteration。

D2-c 将灾难性训练稳定下来，但三 seed pooled 仍是 79/1800 collision、
23/1800 overtake，相比 BC 的 72/1800、24/1800 没有可靠提升。更激进的
超车 reward/margin 能产生真实新超车，但也制造更多新碰撞，例如一个配对
切片得到 gained overtake 3、lost 1，同时 fixed collision 2、new collision 7。

D4/canonical 路线最终产出 cand040、cand120、cand160。P1 原始 holdout
评估曾显示 cand120 同时改善两个 KPI；随后 D0/D0.1 发现重复物理起点导致
原始显著性被高估，并统一到 canonical estimand。D0.1 primary N=3036：

| 模型 | collision | overtake | collision RR vs BC |
|---|---:|---:|---:|
| BC | 170 | 1792 | 1.000 |
| cand160 | 154 | 1799 | 0.906 |
| cand120 | 168 | 1797 | 0.988 |
| cand040 | 166 | 1787 | 0.976 |

cand160 仍是用户指定部署 baseline：它有较稳健的跨图趋势，但 `RR=0.906`
远未达到 0.70 产品目标，Austin 超车点估计也略低于 BC。旧 PPO 的合理结论
是“能产生局部改善，但没有稳定达到产品词典序目标”，不是“PPO 完全无效”。

### 2.3 A 轮诊断：D0.1、D2、D2.5、D2R（07-10 至 07-11）

这轮没有训练最终策略，而是在回答失败来自哪里。

#### D0.1：先把评价底座做对

- 对 16,800 occurrence 做完整性和 canonical 重算；
- primary scenario 数固定为 3036；
- 区分 ego/opp collision、phase、L4 cluster 和 corrected outcome；
- 撤回由重复 deterministic starts 导致的过强显著性结论。

结论：后续必须用物理场景去重、配对 transition 和 clustered 统计，不能只看
未去重的 pooled rate。

#### D2/D2R：危险信息能否从部署观测解码

五个家族都没有通过原始预注册门：D2 的 linear/static MLP/T1/T2，加上
D2R-G。所有家族的 TTC MAE 都远高于 0.30 s；部分家族还失败于 Brier skill
或 1s/2s false alarm。

重要正结果是 D2R-G 使用 LiDAR/速度历史达到 1s recall 0.868、false alarm
0.099、Brier skill 0.130。这证明部署观测中存在可监督解码的风险信息，但
不证明 PPO 一定能从稀疏奖励中学会使用它。T1→T2→D2R 的提升同时改变了
输入、架构和监督，不能包装成“冻结 BC 是唯一因果瓶颈”。

项目所有者后来前瞻性将 TTC 降级为 policy diagnostic。历史五个家族仍保持
原门下 FAIL，不能改写成 PASS；但 TTC 不再阻止 PPO 测试真实 KPI。

#### D2.5：有界残差动作是否有可恢复解

对 91 个 non-test BC ego-collision episode，用固定 90 分支上限的宏动作库
搜索。结果在 67 个案例找到 confirmed-safe-pass witness，全部无 clipping，
且无需正向速度残差。

这排除了“当前有界转向/刹车残差在所测案例中根本不可能救碰撞”的强说法。
但 67/91 只是固定库在该子集的已证实恢复集，不是理论天花板、全分布 RR，
也不保证反应式 PPO 能选中同样动作。

### 2.4 B1：监督 warm-start 与结构支架（07-11 至 07-12）

B1 保留 BC 作为主驾驶，用 D2.5 witness 监督一个小型介入/刹车/转向头，
希望给 PPO 一个避免稀疏探索的起点。它同时建立了后续仍有价值的组件：

- 10 Hz macro action 与 variable-length GAE；
- 层级 latent 和执行动作 ledger；
- 每个 100 Hz step 基于 BC headroom 的有界合成；
- collision/overtake 双 critic 与有上界 dual；
- A/B/C representation arms；
- 288 场景配对评估和四态 transition；
- sampled/executed/logged action 一致性检查。

但 warm-start 本身出现镜像失败。

第一版把约 34% 的人工 brake 配比烙进策略。Task 10 闭环结果：

| Arm | collision | fixed/new | gained/lost overtake |
|---|---:|---:|---:|
| A | 91 | 11/78 | 26/31 |
| B | 54 | 14/44 | 15/71 |
| C | 67 | 13/56 | 28/54 |
| BC | 24 | — | 138 total overtake |

三臂均制造远多于修复数量的新碰撞，并净损失超车。机制审计发现：刹车频率
接近训练配比；转向残差在 NO_OP 时仍常开；`BC steer + residual` 会被模拟器
外部 clipping，导致执行动作与策略 log-prob 不一致。

层级重写修复了常开转向和 clipping 后，第二版使用 0.574% 自然正例率，
又出现三臂 0/9 positive episodes、0/39 positive macros。1,024 个 batch 中
231 个完全没有正例。

因此排除的不是“所有监督 warm-start 数学上不可能”，而是：

- 现有 67-witness 蒸馏不应继续作为 PPO admission gate；
- 逐步 recall/loss 不能代替闭环 collision/overtake；
- 不能再通过第三轮 sampling/calibration 调参拖延 direct PPO。

### 2.5 B2：第一次 B+ 专用 direct PPO（07-13）

B2 从 canonical BC、同一 sidecar initialization 和 fresh residual heads
开始，不加载 warm-start action checkpoint。专用 runner 完成三臂×双 seed×
20 iterations，并在冻结的 288 场景上评估 BC+六候选，共 2,016 配对行。

完整性通过、外部 clipping 为 0。每 seed 产品结果：

| Candidate | collision / RR | overtake | fixed/new | gained/lost | 判定 |
|---|---:|---:|---:|---:|---|
| A seed0 | 26 / 1.083 | 124 | 11/13 | 7/21 | FAIL |
| A seed1 | 11 / 0.458 | 113 | 20/7 | 7/32 | FAIL |
| B seed0 | 17 / 0.708 | 118 | 14/7 | 6/26 | FAIL |
| B seed1 | 17 / 0.708 | 97 | 19/12 | 7/48 | FAIL |
| C seed0 | 8 / 0.333 | 95 | 21/5 | 1/44 | FAIL |
| C seed1 | 9 / 0.375 | 88 | 20/5 | 1/51 | FAIL |

B2 证明 PPO 能学习明显降低碰撞的行为，但六个候选全部净损失超车，甚至
没有通过开发期 1 percentage-point 容差。C 的碰撞改善主要通过把
collision 转为 safe-follow、减少 interaction/overtake，而不是恢复成安全通过；
两个 seed 的 `collision->confirmed_pass` 都是 0。

因此 B2 是**真实 KPI 失败**，不是代理门失败：

- `any_opened_dev_point_target_hit=false`；
- `arm_selection_performed=false`；
- `fresh_pool_opened=false`。

不得因为 C 的碰撞最低就忽略超车硬约束，也不得继续评估这六个 checkpoint。

## 3. B2 的核心实现问题与 B3 假设

B2 rollout 的 PPO distribution 和 deterministic deployment 不是同一个决策面：

- rollout/log-prob 使用 raw gate logits，fresh bias 为 `-6`；
- product evaluator 使用 `raw > -6` 的 centered threshold；
- pooled A/B/C 的 standard intervention decisions 全为 0；
- centered primary interventions 分别为 27,851 / 29,388 / 24,379。

所以 B2 数据只能说明 centered 部署规则的闭环结果，不能说明 PPO 梯度已经
直接优化了该 deterministic rule。训练期策略接近 BC，dual 只小幅变化；部署
时却大量介入并损失超车。这是 B3 要可证伪的主假设：

> 如果 sampling、stored/replayed log-prob 和 deterministic mode 使用同一个
> effective-logit policy，PPO 是否能同时学到选择性介入和超车约束？

B3 的 fresh effective priors 固定为：

```text
P(intervene) = 0.10
P(brake | intervene) = 0.50
P(joint brake) = 0.05
```

raw head 仍从 `-6` 开始；strict deterministic mode 在 fresh state 仍是 NO_OP。
不再有 centered mode 或外部 gate-offset schedule。训练、replay、checkpoint 和
评估全部消费同一个 `ell_I/ell_B`。

## 4. 已排除、已降级与仍未解决的设计/推理

### 4.1 有充分证据排除或停止作为主线

| 设计或推理 | 当前裁定 | 证据 |
|---|---|---|
| “旧 PPO 只需训练更久” | 排除 | Resume600 critic 仍可学，但确定性碰撞恶化 |
| “privileged critic 单独足够” | 排除 | EV 改善未阻止 policy 安全退化 |
| “加强 BC/speed anchor 单独足够” | 排除 | PreBC10/Anchor75 仍远差于 BC 安全性；anchor baseline pooled 失败 |
| “主要问题是 post-overtake 稳定” | 排除为主因 | 事故主体是 pre-overtake rear-end/alongside |
| “只看 aggregate rate 即可” | 排除 | D0 发现重复物理 starts；必须 canonical/paired/L4 accounting |
| “TTC MAE 必须先过才能 PPO” | 由 owner 前瞻性取消 | PPO 不消费精确 TTC；真实 KPI 更直接。旧 FAIL 保留 |
| “warning threshold 直接接刹车就安全” | 排除 | 约 9.9% safe false alarm 在 B1 闭环兑现为误干预和超车损失 |
| “现有 warm-start 是 PPO 必要准入门” | 停止作为主线 | 两次镜像失败；B2 无 warm-start 仍学到强碰撞改善 |
| “转向残差可以无 gate 常开” | 排除 | B1 有 20 个从未刹车但新增碰撞的 A episodes |
| “先相加、再让模拟器 clipping” | 排除 | 会破坏 executed action 与 logged log-prob 一致性 |
| “碰撞最低的模型就是最好” | 排除 | 词典序硬约束要求超车先非劣；B2 C 因超车崩溃失败 |
| “centered deployment 可代表 PPO 学到的策略” | B3 前瞻性排除 | B2 standard decisions=0，而 centered rule 大量介入 |
| “失败后继续打开 fresh pool 找好结果” | 禁止 | opened-development 门未过；fresh pool 保持封存 |

### 4.2 已得到方向性证据，但不能过度推断

| 观察 | 可以说 | 不能说 |
|---|---|---|
| D2R 风险分类较好 | 部署观测含风险信息 | 感知一定不是 PPO 瓶颈 |
| T1→T2→D2R 提升 | 重新编码 raw LiDAR 有价值 | 冻结 BC 是唯一因果变量 |
| D2.5 67/91 | 所测案例存在有界残差解 | 74% 理论天花板或全分布 RR |
| B2 C RR 很低 | PPO 能学到减少碰撞的行为 | PPO 已接近产品成功 |
| B1/B2 dual 变化小 | 训练期约束信号没有覆盖部署损失 | 仅提高 dual LR 一定能解决 |

### 4.3 仍未被排除的问题

- B3 的统一 policy 是否足以产生选择性介入；
- sidecar 在正确的 on-policy 闭环中是否有净增益；
- 40 iterations 是否足够满足两目标，但本轮禁止事后自动延长；
- 部分/全部解冻 BC 是否必要；
- 正向 speed residual 是否有助于恢复超车；
- 若 B3 仍失败，问题是在 reward/cost credit、dual signal、训练预算还是
  representation。必须由 B3 rollout/transition 数据再判断，不能预先编故事。

## 5. B3 下一步计划

### 5.1 当前已完成

- B3 unified policy、runner、evaluator、CLI 和两端控制面已实现；
- 九组核心 CPU 合同测试通过；历史兼容矩阵 20/21，通过项无 B3 regression；
- conditional brake 边界测试已钉住：顶层介入时 `ell_B==0` 不刹，
  `ell_B>0` 刹车；
- sampled policy gradient 在该边界是 `a-0.5 = +/-0.5`，不是依赖熵梯度；
- 固定提交：`19e83ae`、`c320e83`、`21085bc`；
- owner-relayed 独立审阅结论：GO；
- 没有创建 RunPlan，没有使用 GPU，没有新结果。

### 5.2 执行顺序

1. 从 clean committed source 创建唯一 `plan-b3` 并 `show` 审阅；
2. 两端 isolated staging；
3. topology BC baseline 必须精确复现 24 collision / 138 overtake；
4. 双端 preflight、四图 plumbing smoke、共享 `READY.json`；
5. A/B/C × seeds 0/1，全部 40 iterations；
6. 收集六个 iteration-40 checkpoint，不根据 seed0 或 iter20 提前筛选；
7. 冻结一个 288×7 EvalPlan；本地 shard0、远端 shards1–3；
8. 合并一次，先判 corrected-overtake，再判 collision RR 和 transitions。

预计无故障墙钟时间 7.5–8.5 h，含网络/恢复余量按 9–11 h。耗时瓶颈是本地
RTX 3080 的三个 seed1 learner，预计约 5.5–6 h；评估约 1 h 20 min。

### 5.3 B3 的终止与继续条件

| 结果 | 决策 |
|---|---|
| 六候选都不满足超车非劣 | B3 停止；不打开 fresh pool，不挑“最安全”模型 |
| 超车过线但 RR>0.70 | 记录方向改善；不能宣称达到产品目标，需前瞻性决定是否值得确认 |
| 至少一臂每 seed/pooled 均过方向门且 RR<=0.70 | 才能提出独立的 fresh/final confirmation 计划 |
| action/log-prob、clipping、baseline 或 completeness 失败 | 完整性失败，结果不得用于科学判定；修工程问题后使用新计划 |

## 6. 当前可以对外汇报的结论

1. End2Race BC 提供稳定驾驶底座，历史 PPO 可以改变安全/超车 frontier，
   但旧候选没有稳定达到 `RR<=0.70 且超车不降`。
2. D0.1 建立了可审计的 canonical 评价底座；早期未去重显著性已纠正。
3. 部署观测含风险信息，有界 residual 动作在大量所测碰撞案例中存在安全
   通过解；因此“完全看不见”或“动作空间完全无解”都不成立。
4. TTC 探针与 warm-start 提供了诊断价值，但不再是 PPO 准入门。
5. B2 是当前第一次完整 B+ PPO 直接 KPI 裁决：碰撞可显著下降，但六个
   候选全部以超车损失换安全，故实质失败。
6. B3 不是新模型家族 sweep，而是修复 PPO 最基本的 policy identity：让训练
   和部署优化同一个策略。B3 尚无数值结果，不能提前宣称会成功。

## 7. 主要证据入口

- 当前 authority：`.agents/HANDOFF.md`
- B2 计划：`.agents/B2_PPO_PLAN.md`
- B3 计划：`.agents/B3_PPO_PLAN.md`
- B3 实现记录：`.agents/B3_IMPLEMENTATION_RECORD.md`
- 完整实验总账：`docs/EXPERIMENT_RECORD.md`
- P1：`Experiments/A1_p1_validation/p1_final_report_20260710.md`
- D0.1：`Experiments/A0_project_registry/D01_EVIDENCE_REPORT_20260711.md`
- D2：`Experiments/A3_d2_representation/D2_EVIDENCE_REPORT_20260711.md`
- D2.5：`Experiments/A4_d25_counterfactual/D25_EVIDENCE_REPORT_20260711.md`
- D2R：`Experiments/A5_d2r_geometry/D2R_EVIDENCE_REPORT_20260711.md`
- B1 Task 10：
  `Experiments/B1_route_r2_scaffold/artifacts/task10_warmstart_20260712_105740/`
- B2 canonical summary：
  `Experiments/B2_ppo_pilot/evaluations/b2_eval_20260713_165800/merged/summary.json`

本报告用于项目汇报和下一次审阅。冻结 artifact、原始 spec 和历史 FAIL/PASS
仍是数值事实的最高证据；本报告不会重写它们。
