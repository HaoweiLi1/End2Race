# End2Race — Agent 合约

**任何 agent（Claude / Codex）开工前先读这一份。**

- 当前状态与下一步 → `.agents/HANDOFF.md`（唯一权威入口）
- B2 BC-direct PPO 执行计划（已批准）→ `.agents/B2_PPO_PLAN.md`
- B2 Opus 4.8/max 独立审计与裁定 → `.agents/B2_PPO_REVIEW.md`
- B2 pre-GPU 实现复核（GO_FOR_STAGING）→ `.agents/B2_IMPLEMENTATION_REVIEW.md`
- 仓库结构 → `.agents/REPO_GUIDE.md`
- 实验编号与索引 → `Experiments/INDEX.md`
- 完整实验史 → `docs/EXPERIMENT_RECORD.md`

---

## 1. 项目目标（词典序，不要偏离）

1. **硬约束**：超车率不低于 BC；
2. 在此前提下**降低任意方碰撞率**（目标 `RR ≤ 0.70` vs BC）；
3. 可选：之后再提升超车率。

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

- learner 以完整 seed queue 分配：seed1 的 A/B/C 本地串行，seed0 的 A/B/C
  远端串行；两台 host 并行，但每张 GPU 同时只允许一个 learner。
- 只有冻结 checkpoint 的 scenario evaluation 才按**本地 1/4、远端 3/4**分片；
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
./run.sh stage <plan.json>        # 两端仓库外隔离部署
./run.sh baseline-preflight <plan.json> # 本地先重放 BC 288 行并锁死 24/138
./run.sh preflight <plan.json>    # source/input/env/GPU/CLI fail-closed
./run.sh execute <plan.json>      # 只消费同一计划
./run.sh resume <plan.json>       # 仅从已验证的完整 iteration 边界显式恢复
./run.sh status <plan.json>       # 状态
./run.sh collect <plan.json>      # 本地/远端结果回收并重验 COMPLETE envelope
```

**新增托管工作 → 在 `runner.py` 里加 Job，不要写成 shell 单行命令。** 一个 job 应该在烧 GPU 之前就能被审阅、被 dry-run。

> **B2 当前处于实现/preflight 阶段：** 旧 `b2-exploration-sweep` 与
> `b2-ppo-pilot-seed0` 占位 job 不得执行。owner 已授权 managed B2，但只有
> `.agents/B2_PPO_PLAN.md` 的 Tasks 0–6、测试、隔离 staging 和 preflight
> 全部通过后才可启动冻结的20轮数值 pilot。远端旧 dirty repository 永不作为
> B2 执行目录。

---

## 5. 提交纪律

- 实验做完**记得提交 git**。
- 大产物（`Experiments/*` 的数据、`eval_results/`、`pretrained/` 的模型）不进 git，只提交代码、spec、索引和 runner。
- 主目录**只保留**原始 python 文件（`model.py` `train.py` `utils.py` `eval_*.py` `demonstration.py`）和 PPO 文件（`train_ppo.py` `ppo_utils.py`）。新脚本：临时的放 `tests/`，长期有效的合并进主目录已有文件。

---

## 6. 当前位置（2026-07-12）

- 历史 PPO/P1 与 B1 Task 10 记录过碰撞和超车；真正缺失的是：**尚无
  B+ v2.2 PPO 学习后的 candidate 接受当前词典序开发门或 fresh 产品门。**
- 五个感知探针保留其原始 TTC-gate FAIL，TTC 已被 owner 前瞻性降为
  policy diagnostic；旧 warm-start 闭环有害，hierarchical replacement
  Task 6 又出现 positive recall 0，因此 warm-start 不再建议作为 PPO 准入门。
- 仓库有历史 `train_ppo.py`；B+ v2.2 的层级 rollout、完整 replay、two-head
  constrained clipped PPO、双层探索、exact checkpoint/resume、直接 KPI evaluator
  与隔离 RunPlan runner 现已接通。两轮 Opus 实现审计已闭环为
  `GO_FOR_STAGING`；数值 pilot 尚未启动，仍须完成 clean commit、BC-only 288
  基准预检、两端 staging/preflight 和四图 plumbing smoke。
- 已批准下一步见 `.agents/B2_PPO_PLAN.md`：从 BC-identical fresh policy
  开始三臂 PPO，用完整记账的 top/conditional-brake 训练期探索，直接评估
  collision RR 与 corrected overtake。Claude Code `claude-opus-4-8` / max 的
  设计审计与实现复核分别记录在 `.agents/B2_PPO_REVIEW.md`、
  `.agents/B2_IMPLEMENTATION_REVIEW.md`。
