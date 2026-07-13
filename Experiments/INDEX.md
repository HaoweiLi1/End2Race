# Experiments 索引

实验按**轮次**编号。大写字母 = 一轮大实验，轮内用数字（`A1`），需要再拆分时用小写字母（`A1a`）。
每个实验目录**自包含**：日志、产物、模型都在里面。

所有移动都记录在 `PATH_MIGRATION.tsv`（旧路径 → 新路径）。

---

## A 轮 — 诊断期（已完成）

回答的问题是：**PPO 为什么改不动碰撞率？** 结论是感知与动作空间都不是死路，但没有一个探针通过它自己设的 TTC 门。

| 编号 | 原路径 | 内容 | 结论 |
|---|---|---|---|
| `A0_project_registry` | `logs/ppo_next_unattended_20260710_230212` | 追加式 registry（项目总账）+ D0.1 证据报告 | 基础设施，跨实验共用 |
| `A1_p1_validation` | `logs/p1_validation_20260710_121955` | cand160 vs BC 配对验证 + 3 个候选 checkpoint | **统计等价**，不得声称优越 |
| `A2_d0_canonical_audit` | `logs/d0_canonical_audit_20260710_121955` | 正典审计 | estimand 修正 N=3024 → **3036** |
| `A3_d2_representation` | `logs/d2_representation_20260711_174039` | 表示探针（4 家族）+ **封存的 test seal** | 4 家族**全部未过** TTC 门 |
| `A4_d25_counterfactual` | `logs/d25_counterfactual_20260711` | 反事实可恢复性 oracle | **67/91 可恢复**，Route-R2 动作空间可行 |
| `A5_d2r_geometry` | `logs/d2r_geometry_20260711` | 时空几何探针 | **未过** TTC + 2s 误报门 |

`A3` 里的 `artifacts/split_lock/test_seal.json`（SHA256 `cee71d81…a87e`）是**从未打开的分组测试集**的封条。不要动它。

## B 轮 — Route-R2 策略（已完成至 B4；下一轮未授权）

| 编号 | 原路径 | 内容 | 状态 |
|---|---|---|---|
| `B1_route_r2_scaffold` | `logs/bplus_v22_d3r2_20260711` | v2.2 支架：层级动作、identity 门、warm-start、闭环评估（Task 1–10） | Task 10 **FAILED**；warm-start 两次失败 |
| `B2_ppo_pilot` | — | B+ v2.2 PPO 接线 + 三臂 BC-direct pilot | **首轮训练与评估完成，产品方向门 FAILED**；六个候选均净损失超车，未选臂，fresh pool 未打开。训练 `b2_direct_20260713_081422`，评估 `b2_eval_20260713_165800` |
| `B3_ppo_unified` | — | 统一 stochastic rollout / PPO log-prob / deterministic deployment 的三臂 PPO | **实现、边界回归和独立审阅 GO，尚未创建 RunPlan 或运行 GPU**；固定 40 iterations，B2 结果不续训 |
| `B4_direct_head_ppo` | — | plain End2Race frozen-feature output-head-only PPO，唯一 seed1 | **30/30 训练与 2400-row 产品评估完成，B4_SUBSTANTIVE_NEGATIVE**；BC/iter10/20/30 collision-overtake=`24/342`,`24/332`,`36/294`,`39/296`，未选择 candidate，fresh/final pool 未打开 |

## C 轮 — 预留

## `_archive/` — 已废弃轨道

废弃的 PPO 调参期（D1 / 旧 D2 / D4a / D4c / anchor，07-04 至 07-08）。结论已固化在 `_archive/legacy_reports/`，原始产物按字节保留、未删除。

- `_archive/eval_results/` — **73 个评估输出目录，33 GB**
- `_archive/models/` — 187 个废弃轨道 checkpoint，约 9 GB
- `_archive/legacy_runs/`、`legacy_reports/`、`reviews/`、`superseded_artifacts/`

⚠️ `_archive/eval_results/` 和 `_archive/models/` 是**本地唯一副本**——远端没有备份，因为它们产生于
2026-07-08「实验只在远端跑」政策之前（全部是 `*_local_*` 命名的本机运行）。虽然都属废弃轨道、
不被任何活证据引用，**未经项目所有者批准不要删除**。

---

## 仓库其余目录

| 目录 | 性质 |
|---|---|
| `bplus_v22/` `d0/` `d2/` `d25/` `d2r/` | 证据模块源码 |
| `f1tenth_gym/` `f1tenth_racetracks/` `latticeplanner/` | 模拟器与规划器（运行时依赖） |
| `pretrained/` | **只放原始 BC 模型** `end2race.pth` |
| `eval_results/` | 评估输出**暂存区**（活代码硬编码写入）。跑完把输出移进所属实验目录 |
| `logs/` | 仅保留兼容符号链接——若干冻结 spec 按 `logs/…` 旧路径引用报告 |
| `scripts/` `tests/` `docs/` | 分析工具 / 测试 / 文档 |

---

## 已知破坏：旧 release 无法重新校验

2026-07-12 的这次迁移，把证据目录从 `logs/` 移到了 `Experiments/`。项目所有者**明确批准**了随之而来的代价：

**已完成 release 的 `config.json` / `pinned_inputs.json` 里冻结记录着旧的 `logs/…` 路径。这些 JSON 是不可变证据（它们自身的哈希被记在 manifest 里），不能改写。因此 `validate-source-preflight`、`validate-warmstart` 等按路径解析的校验，对旧 release 会失败。**

**没有丢失的东西**（已逐项核验）：

- 每个 release 内部的 `output_manifest.sha256` 仍然自校验通过——**产物内容逐字节完好**；
- test seal 哈希迁移前后一致：`cee71d818bc050b0ca0647ee32ed1b5655e471ea60b39133aed7b37fc9c1a87e`；
- 原始 BC 模型哈希一致：`b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4`。

失效的只有"按旧路径去仓库根下找文件"这一步。**证据内容可验，证据位置的自动解析不可验。**

新 release 使用新路径，校验正常。

---

## 怎么跑实验

不要手敲 ssh 命令。B2/B3/B4 由 `Experiments/runner.py` 生成不可变 RunPlan，再用
仓库根的 `run.sh` 分阶段执行：

```bash
./run.sh plan ...                 # 从 clean commit 生成冻结 plan/source/input bundles
./run.sh show <plan.json>         # 只打印，不执行
./run.sh stage <plan.json> --all-hosts --dry-run
./run.sh baseline-preflight <plan.json> --dry-run  # BC-only 288 rows, expect 24/138
./run.sh preflight <plan.json> --all-hosts --dry-run
./run.sh plumbing-smoke <plan.json> --dry-run
./run.sh execute <plan.json> --all-hosts
./run.sh resume <plan.json> --host <local|remote>  # only after an interrupted learner queue
./run.sh status <plan.json> --all-hosts
./run.sh collect <plan.json>
```

`plumbing-smoke` 通过后会生成并同步相同的 `READY.json`；`execute/resume`
在 GPU lock 内重验完整 staged source、runtime inputs 与 READY 所绑定的
BC/P3 marker 后才会启动 learner。不要手工创建或修改 READY。

旧 `run <job>` / `split <job>` 与 B2 占位 job 已删除。learner 按完整 seed queue
运行；只有冻结 checkpoint 的 evaluation 才能按 scenario shard。

B4 的唯一已授权执行现已关闭。训练 release 是
`B4_direct_head_ppo/runs/b4_seed1_20260714_003027`，最终 3x4x50 证据是
`B4_direct_head_ppo/product_evaluations/b4_product_seed1_20260714_003027/final`。
不要用上述命令自动创建 B3/B5 或续跑 B4；需新的前瞻性 owner decision。
