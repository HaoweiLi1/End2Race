# End2Race — Agent 合约

**任何 agent（Claude / Codex）开工前先读这一份。**

- 当前状态与下一步 → `.agents/HANDOFF.md`（唯一权威入口）
- B2 BC-direct PPO 执行计划（已批准）→ `.agents/B2_PPO_PLAN.md`
- B2 Opus 4.8/max 独立审计与裁定 → `.agents/B2_PPO_REVIEW.md`
- B2 pre-GPU 实现复核（GO_FOR_STAGING）→ `.agents/B2_IMPLEMENTATION_REVIEW.md`
- B3 统一训练/部署策略计划（`IMPLEMENTED, REVIEWED GO, PAUSED UNRUN`）→
  `.agents/B3_PPO_PLAN.md`
- B3 实现与待审清单 → `.agents/B3_IMPLEMENTATION_RECORD.md`
- **B4 已关闭计划**（唯一 seed1 与 3x4x50 product-grid 已完成，结果为
  `B4_SUBSTANTIVE_NEGATIVE`，未选择 candidate）→
  `.agents/B4_DIRECT_HEAD_PPO_PLAN.md`
- **B4 数值结果与外审证据** → `.agents/B4_DIRECT_HEAD_PPO_RESULT.md`
- **B4 substantive-negative 原因分析与下一假设排序** →
  `.agents/B4_SUBSTANTIVE_NEGATIVE_ANALYSIS.md`（exact content boundary
  `dd49ce00bc82095a1cdd832caa485bce01c1991f`）
- **B5-A safe-reference 单变量实验计划与结果** →
  `.agents/B5_SAFE_TRUST_REGION_PLAN.md`、
  `.agents/B5_SAFE_TRUST_REGION_RESULT.md`（exact result/evidence boundary
  `d57d6e9bc4c49fdd9e522f4b4e825277239b405d`）
- **B5-A 统计限定与 objective-alignment 机制审计** →
  `.agents/B5_POSTHOC_STATISTICS_AND_OBJECTIVE_AUDIT.md`（B5-B weighting
  learner `NO-GO, UNRUN`；exact content boundary
  `ba0d9c6400b75498f2e258ed2019982863cdd1c6`）
- B4 外部审计草案（已被 owner decision 取代，仅保留为审计证据）→
  `.agents/B4_DIRECT_HEAD_PPO_EXTERNAL_AUDIT_PLAN.md`
- 从 End2Race BC 到 B3 的 PPO 开发汇报 → `.agents/PPO_DEVELOPMENT_REPORT.md`
- 仓库结构 → `.agents/REPO_GUIDE.md`
- 实验编号与索引 → `Experiments/INDEX.md`
- 完整实验史 → `docs/EXPERIMENT_RECORD.md`

---

## 1. 当前产品目标（B4 前瞻性 authority）

1. **安全优先**：降低任意方碰撞率（产品目标 `RR ≤ 0.70` vs BC）；
2. **B4 硬约束**：最终 600-case BC-compatible grid 上，candidate terminal
   overtake 至少 `ceil(0.95 * BC_overtake)`；
3. 在满足上述条件后优先 collision 更低，再优先 overtake 更高。

该 5% 变更只向前适用于 B4。B2 仍按其运行时冻结的 strict/1pp 门保持 FAILED；
B3 不追溯改写，状态为已实现、已审阅 GO、暂停未运行。详见
`.agents/B4_DIRECT_HEAD_PPO_PLAN.md` §1。

2026-07-14 B4 已执行结束：BC 为 `24 collision / 342 overtake`；iter10 为
`24/332`，iter20 为 `36/294`，iter30 为 `39/296`。没有 snapshot 同时满足
collision strict improvement、`fixed>new` 和 overtake `>=325`，因此 selected
candidate 为 none。当前没有自动 B3/B5 或追加 seed 的 authority；详见 result record。

现有 product rows、replay 与 snapshots 的只读复算进一步确认：iter10/20/30
在相同 BC 历史上的平均 signed speed drift 约为 `-0.031/-0.095/-0.102 m/s`，
BC-relative equal-std KL 约为 `0.027/0.138/0.188`；100 Hz exploration 的
lag-1 correlation 约为零，训练 collision 比例是 product 的 `9.375x`。
这些证据优先支持“缺少 BC-preserving constraint 导致非选择性累计漂移”，但不证明
Residual 唯一正确，也不判定 frozen GRU representation 是否充分。详见 analysis record。

2026-07-14 B5-A 已完成唯一 seed1：safe-reference hard cap 全程满足，iter10
在相同 opened 600-case panel 得到 `22 collision / 347 overtake`，paired
`9 fixed / 7 new collision`，因此选择为 `OPENED_DEVELOPMENT_SURVIVOR`。
它未达到 collision `<=16` target，且 iter20/30 回升至 25/27 collisions。
该面板不是 fresh/final；外部结果审阅前不得运行 seed0 或打开 sealed pool。
后续 startpoint-cluster 复算确认 iter10 的 `9 fixed / 7 new` 仍在噪声底内；
历史 feasibility verdict 保留，但 checkpoint 不晋升。对 opened-Austin outcome
权重的远端函数空间/actor+Adam 审计未满足启动 B5-B 的机制条件，因此 B5-B 和
AR(1) 均未运行；详见 post-hoc audit。

**每设一道门，都要能回答一句话：「这道门挡住了，交付物会因此变好吗？」答不上来的门，不要设。**

这个项目已经因为违反这条规则烧掉两周（详见 §2）。

---

## 2. Claude 和 Codex 常犯的错误

以下每一条都在这个项目里**真实发生过**，不是假设。

### 2.1 编造有说服力的因果故事（Claude，累犯）

| 说过的话 | 实情 |
|---|---|
| 「TTC 门是后来随意添加、从未批准」 | **错**。它白纸黑字写在已批准 spec 的 `Pre-registered gate` 一节 |
| 「67/91 = 74% 天花板」 | **错**。那是固定分支库在子集上的可恢复率，不是天花板 |
| 「beta_bc=5.0 证明软锁防过头」 | **证据不足**。2 个 seed，无消融 |
| 「危险可感知，所以感知不是瓶颈」 | **过强**。可解码 ≠ PPO 能学到 |
| 「每轮约 10 次刹车尝试」 | **算错**。重算是约 1 次 |

最严重的是第一条：它不只是算错，而是**在没读那份 spec 的情况下编了一个很好听的叙事**（「代理指标僭越成目标」）。叙事之所以有说服力，恰恰因为是奔着有说服力去构造的。

**规律：流畅度和可靠度负相关。** 特别干净利落、特别有说服力的因果结论，八成是编的。真实的技术发现通常带限定词（「这三个家族同时改了四个变量，不是干净消融」）。

**防御：任何断言都要能答出「你读了哪个文件第几行 / 跑了哪条命令」。答不出就是在编。**

### 2.2 用"完整性通过"掩盖"实质失败"（Codex）

Task 6 第一次 warm-start：三臂 loss 都下降、结构测试全过、manifest 全绿，Codex 判定「不否定完整性通过，后续再查」。

实际上 `gate_recall = 0`、`gate_accuracy` 三臂 16 位小数完全相同——gate **从未产生过任何正预测**。根因是 `INITIAL_BRAKE_LOGIT = -6.0`（先验 0.25%）与训练数据（22.9% 正例）**差 93 倍**，数学上不可能成功。

**规律：integrity PASS 和 substance PASS 是两件事。** 产物完整、哈希正确、测试通过，完全不代表实验做对了。

**防御：看到"loss 下降但指标没动"，立刻怀疑初始化/标定，不要推迟到下一阶段。**

### 2.3 把中间产物的及格线当成交付物的及格线（双方）

三次弯路，形状一模一样：

| 弯路 | 中间产物 | 自设及格线 | 代价 |
|---|---|---|---|
| 1 | TTC 回归探针 | TTC MAE ≤ 0.30s | 5 个家族全败 |
| 2 | warm-start 蒸馏 | gate recall / loss 门 | 2 次失败，约 4,600 行 |

**共同特征：每个都带一套自己的及格线，而那些及格线没有一条出现在用户的目标里。** 每次失败都触发"换个架构再试一次"，而不是回头问"这道门是否服务于交付物"。

**防御：一道门连续挡住多个架构、而它测的东西又不是交付物需要的 → 质疑门，不要继续换架构。** 质疑的合法方式是**项目所有者前瞻性 override**（说明部署为什么不需要它），不是"没考过所以改考卷"。

### 2.4 恒真检查伪装成通过项（Codex）

`false_intervention_episodes_le_7_of_75` 这条 check **永远为 True**——阈值规则取
`nextafter(第8大负例,+inf)`，构造上保证 `<=7`；并列时可以少于7，本次 release
观测到恰好7。它出现在汇总表里像个成绩，实际不携带任何模型信息。

**防御：每条 check 问一句「它有可能失败吗？」不可能失败的，别列进通过项。**

### 2.5 只有互相顶嘴才抓得住（有效做法，保留）

Codex 用引用把 Claude 的五条全顶回来了；Claude 用原始数据抓出了 Codex 的 gate 全 NO_OP 和恒真检查。

**双 agent 有效，但前提是双方都真的去读文件、跑命令，而不是凭记忆推理。**

---

## 3. 远端 Ubuntu 使用

### 唯一连接方式

```bash
ssh haowei@192.168.2.127
```

**只有这一种。** 不要用 Tailscale 地址、不要用主机名、不要用其他 IP。

| | 本地 | 远端 |
|---|---|---|
| 主机 | 本机 | `haowei@192.168.2.127`（haowei-MSI） |
| GPU | RTX 3080 Laptop 16GB | RTX 4080 SUPER 16GB |
| 仓库 | `/home/haowei/Documents/End2Race` | `~/Documents/End2Race` |
| Python | `~/miniconda3/envs/end2race/bin/python` | 同左 |
| 评估需要 | — | `DISPLAY=:1` |

**每个模拟器任务必须有独立的绝对 `NUMBA_CACHE_DIR`**（共享缓存曾损坏过一次 Task-10 运行，runner 会强制检查）。

### 任务分流（2026-07-12 起）

**本地验证无误后，PPO 训练和 PPO 评估在本地与远端同时跑，加速实验。**

- 已关闭的 B4 只有一个 architecture 和一个 seed：seed1 learner 已在远端 RTX 4080
  完成；未启动 seed0，不存在 A/B/C arm queue。
- 下述 A/B/C 描述只适用于已冻结的 B2/B3 历史 topology，不得套到 B4。

- learner 以完整 seed queue 分配：seed1 的 A/B/C 本地串行，seed0 的 A/B/C
  远端串行；两台 host 并行，但每张 GPU 同时只允许一个 learner。
- B4 product evaluation 已按 5 个等量 startpoint shards 完成：**本地 1/5、远端
  4/5**。该分配是已关闭实验的 provenance，不是下一轮执行授权。
- 下述本地 1/4、远端 3/4 只适用于 B2/B3 历史 288-panel evaluation。
- 只有冻结 checkpoint 的 B2/B3 scenario evaluation 才按**本地 1/4、远端 3/4**分片；
  远端 shards 1–3 在同一 GPU lock 内串行。
- 分流的目的是**并行加速**，不是交叉验证。
- **不要**为了"验证两台设备结果一致"而在两边跑同一个任务——那是浪费算力。

### 交叉验证的唯一触发条件

**只在实验结果出现明显错误时**才做跨设备交叉验证。例如：

- 指标出现物理上不可能的值；
- 同一配置两次运行结果差异巨大；
- 结果与已知基线严重矛盾。

平时不做。

### 不要再做的事

- ❌ **不要为新产物设置哈希 manifest**。已有的历史哈希保留（它们是既成事实），但新工作不再建哈希链。
- ❌ **不要跨设备重复运行同一任务来验证 bit-identical**。
- ❌ 不要写散落的 ssh 单行命令（见 §4）。

---

## 4. 怎么跑实验

所有批量/托管运行都由 `Experiments/runner.py` 的不可变 RunPlan 控制，用仓库根
的 `run.sh` 执行。B2 runner 完成后使用：

```bash
./run.sh plan ...                 # 生成一次、不可覆盖的计划
./run.sh show <plan.json>         # 只打印冻结命令
./run.sh stage <plan.json> --all-hosts # 两端仓库外隔离部署
./run.sh baseline-preflight <plan.json> # 按最终拓扑重放：local shard0 + remote shards1–3，合并锁死 24/138
./run.sh preflight <plan.json> --all-hosts # source/input/env/GPU/CLI fail-closed
./run.sh plumbing-smoke <plan.json> # 四图×三臂各一次短更新，只验证接线
./run.sh execute <plan.json> --all-hosts # 只消费同一计划
./run.sh resume <plan.json> --host <local|remote> # 仅从完整 iteration 边界显式恢复
./run.sh status <plan.json> --all-hosts # 状态
./run.sh collect <plan.json>      # 本地/远端结果回收并重验 COMPLETE envelope
```

`plumbing-smoke` 会在三臂接线均通过后原子发布两端相同的 `READY.json`。
learner 只接受 READY 所绑定的 RunPlan/source/input/baseline/P3 哈希，并在
GPU lock 内重新哈希 staged tree；即使直接调用 `ppo-pilot` CLI，缺少 READY
也会 fail-closed。READY 不得手工创建或编辑。

**新增托管工作 → 在 `runner.py` 里加 Job，不要写成 shell 单行命令。** 一个 job 应该在烧 GPU 之前就能被审阅、被 dry-run。

双机容量与并发合同见 `.agents/COMPUTE_CAPACITY_AND_EXECUTION_GUIDE.md`。
实测推荐值是：独立 learner 远端默认 4、吞吐模式最多 6，本地默认 2、
吞吐模式最多 4；eval-only 优先 CPU，远端 12 workers、本地 6–8 workers。
单 learner 的 batch-1、100 Hz 串行 collector 天然不会拉满 GPU，禁止复制同一
seed/job 或绕过 runner 的 GPU lock 来制造利用率。未来并发必须先把 slot、CPU
affinity、thread limit 和独立输出/cache/RNG 固定进 RunPlan。

> **B2 首轮 pilot 与冻结评估已完成：** 训练 RunPlan 是
> `b2_direct_20260713_081422`，评估 EvalPlan 是
> `b2_eval_20260713_165800`。六个 candidate 全部 integrity PASS，但全部因
> corrected overtake 下降而未通过方向门；没有 arm selection，fresh pool 未打开。
> 不得继续运行同一候选、medium/final confirmation 或旧占位 job。下一次数值
> PPO 必须来自新的、前瞻性批准的目标/约束修订和唯一 RunPlan。

> **B3 当前完成实现与独立边界审阅，尚未运行：** B3 已把采样、PPO
> log-prob 与确定性部署统一为一个 effective-logit policy，并冻结
> 40-iteration 控制合同。九组核心 CPU 合同测试通过，边界补丁提交为
> `21085bc`，独立审阅结论为 GO；尚未创建 RunPlan、stage 或启动 GPU。
> 先读 `.agents/B3_PPO_PLAN.md` 与 `.agents/B3_IMPLEMENTATION_RECORD.md`。

---

## 5. 提交纪律

- 实验做完**记得提交 git**。
- 大产物（`Experiments/*` 的数据、`eval_results/`、`pretrained/` 的模型）不进 git，只提交代码、spec、索引和 runner。
- 主目录**只保留**原始 python 文件（`model.py` `train.py` `utils.py` `eval_*.py` `demonstration.py`）和 PPO 文件（`train_ppo.py` `ppo_utils.py`）。新脚本：临时的放 `tests/`，长期有效的合并进主目录已有文件。

---

## 6. 当前位置（2026-07-13）

- 历史 PPO/P1 与 B1 Task 10 记录过碰撞和超车；真正缺失的是：**尚无
  B+ v2.2 PPO 学习后的 candidate 接受当前词典序开发门或 fresh 产品门。**
- 五个感知探针保留其原始 TTC-gate FAIL，TTC 已被 owner 前瞻性降为
  policy diagnostic；旧 warm-start 闭环有害，hierarchical replacement
  Task 6 又出现 positive recall 0，因此 warm-start 不再建议作为 PPO 准入门。
- B+ v2.2 专用 PPO runner 已实现并完成首次三臂、双 seed、20-iteration
  managed pilot。六个 learner release 全部 integrity PASS；训练 RunPlan 是
  `b2_direct_20260713_081422`。
- topology-matched BC baseline、双端 preflight 与 P3 均通过。冻结评估
  `b2_eval_20260713_165800` 已完成 4 shards、288 场景、7 variants、2,016 行，
  `integrity_passed=true`，并独立重算一致。
- **六个候选全部未通过真实方向门**：没有任何 seed 同时满足 corrected
  overtake 非劣与 collision `RR <= 0.70`。最安全的 C 臂为 8/9 次碰撞
  (`RR=0.333/0.375`)，但超车从 BC 的 138 降到 95/88；所有候选都净损失超车。
- `any_opened_dev_point_target_hit=false`、`arm_selection_performed=false`、
  `fresh_pool_opened=false`。不得打开 fresh/final pool，也不得把碰撞下降单独写成
  成功。下一步只能基于训练/paired transition 证据前瞻性修订 PPO 目标或约束，
  不能返回 TTC 或 warm-start 代理门。

---

## 7. B5-A 已完成状态（2026-07-14）

- B4 已在 Austin `3 racelines x 4 speeds x 50 startpoints` 面板得到真实的
  `B4_SUBSTANTIVE_NEGATIVE`；该 600-case 面板现正式称为
  **opened-development regression panel**，fresh/final 仍封存。
- B5-A 是严格单变量实验：完整保留 B4 seed1 的模型、LR、clip、epochs、100 Hz
  iid exploration、reward、`6/6/4` 精确 curriculum 顺序、30 iterations 和
  `0/10/20/30` snapshots。
- 唯一新增机制是从 1,640 个 training L2 确定性选择的 64-episode canonical-BC
  safe reference，以及累计 `D_safe <= 0.01` hard cap。拒绝 retry 必须同时恢复
  output head 与完整 Adam state，并复用相同 minibatch order。
- 唯一合法 learner seed1 已在远端 RTX 4080 SUPER 完成 30/30；本地只承担
  correctness 与 product shard0，没有补跑 seed0 或第二 arm。
- reference audit、BC baseline、双端 preflight、production plumbing、原子训练
  release 与 1,800 个新 candidate episodes 均完成。复用的 600 个 BC rows 加上
  candidates 共 2,400 paired rows，完整性通过。
- 正式结果为：iter10 `22/347`、iter20 `25/349`、iter30 `27/343`
  （collision/overtake）。只有 iter10 满足 collision `<24`、overtake `>=325`、
  `fixed>new` 和 speed projection 0，故选择为 opened-development survivor。
- 详细冻结协议与结果：`.agents/B5_SAFE_TRUST_REGION_PLAN.md`、
  `.agents/B5_SAFE_TRUST_REGION_RESULT.md`。下一合法动作是外部结果审阅；不得自动
  运行 seed0、B5-B/AR(1)、修改 cap/LR、解冻 GRU 或打开 fresh/final。

---

## 8. B6 AR(1) phase-0 已完成（2026-07-14）

- 外审批准的下一变量先以 no-learning direct-outcome audit 执行；没有启动 PPO
  learner。simulator、GRU 与 canonical BC actor mean 始终为 100 Hz。
- 使用 Task-8 training-only 中 60 个同时含 collision/overtake/follow 的 L4，
  每个 outcome hash 选一个 L2，4 个固定 innovation seeds，iid/AR(1) 两臂，
  共 1,440 episodes / 720 pairs。Austin 600、seed0 与 sealed pool 未使用。
- 最终有效边界为 source `677ab3a75070f7ef5d685ad34e987f65c99893b3`、
  RunPlan SHA256 `4a3923dbe2cf87073aa0aadb0bc59d8d8222882c107cb3d31c5e50f275dbbe7f`。
- integrity PASS：iid lag-1 约 0，AR(1) 约 0.95；marginal std 相符；framewise
  ratio error 0；speed projection 0。
- direct result 为 NO-GO：repair `+8/240`（3.33pp，L4 p=0.262）；safe→collision
  harm `+48/480`（10pp，L4 p=4.19e-9）；lost overtake `+17/240`
  （7.08pp，L4 p=0.000473）。learner 保持 UNRUN。
- 结论只否定 `rho=.95 + std=(.03,.20)` 的无条件 full-action correlated
  exploration，不否定所有 temporal exploration 或 plain End2Race。
  但按 owner 停止规则，当前 frozen-feature direct-head 调参线关闭；下一步必须先
  外审 `.agents/B6_TEMPORAL_EXPLORATION_PHASE0_RESULT.md`，再前瞻性决定
  representation adaptation 或 bounded macro safety-control proposal。
- B6 code/evidence review boundary 是
  `081092987877619e9b84f108f80cbebe3bda847c`；其后的提交只记录可寻址边界。

---

## 9. B7 plain recurrent PPO 已获执行授权（2026-07-14）

- owner 已停止 Route-B gate/macro/wrapper 循环，并授权一个单配置的产品导向
  engineering run；完整冻结协议见 `.agents/B7_PLAIN_RECURRENT_PPO_PLAN.md`。
- deployment 仍为 canonical 12-key `End2Race.state_dict()`；冻结输入编码，低 LR
  训练原始 GRU (`1e-6`) 与 output head (`1e-5`)。
- 每轮 32 个完整 episode，只做一个 recurrent actor Adam step；collision `-2`
  分布到终局前 100 frames，critic 增加 normalized remaining time 并按 episode
  等权。
- sampler 固定为 16 map-balanced representative + 8 archived BC collision + 8
  previous-policy hard/preservation；只复用 scenario identity，不复用 transition。
- actor step 只有在 old-policy rollout mean-KL `<=.015` 且本轮 archived-BC-safe
  observation histories 上 canonical-BC mean-KL `<=.01` 时接受；否则恢复 actor+
  完整 Adam、下轮两个 LR 均减半，连续三次拒绝即停。
- 只运行远端 seed1、10 iterations、只评估 iter10。若 288 opened-development
  gate 不通过，plain recurrent PPO 主线关闭；通过后才可外审并条件运行 seed0。
- 新 authority 覆盖本文件更早的“远端 unattended authority revoked”历史描述，
  但只限这个 B7 RunPlan；不得并行复制 seed1、打开 Austin/sealed 或自动调参。

## 10. B7 已早停、无 candidate（2026-07-14）

- remediation 后的合法 source 为 `3e262e2bf00acd8ef9338122a82780e68a825981`，
  RunPlan SHA256 为 `3cd0f801f59609fcf6ab02a674851f49678de6b0fb04dc6a27201ff08c2672ad`。
- seed1 共完成 9 iteration、288 complete episodes、203,289 transitions；actor
  update 在 iter1/2/3/6 接受，在 iter4/5/7/8/9 回滚。
- iter7/8/9 连续三次 safe KL 超过 `0.01`，故按预注册规则早停；状态为
  `EARLY_STOP_NO_CANDIDATE`。只有 canonical BC 的 iter0 actor，不存在 iter10。
- 因没有 candidate，288 eval、seed0、Austin 600 与 sealed/final 均未启动。
  不得 post-hoc 选择 iter6、放宽 cap、续跑或调参。
- 该结果关闭 owner 指定的 plain recurrent PPO engineering line，但没有形成
  candidate KPI 结论。详见 `.agents/B7_PLAIN_RECURRENT_PPO_RESULT.md`。
