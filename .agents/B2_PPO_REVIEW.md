# B2 PPO 独立审计与裁定

状态：**APPROVED_WITH_BLOCKING_FIXES — 下列裁定必须先进入计划与测试，之后才可数值运行**

日期：2026-07-12（Asia/Singapore）

审阅基线：`chore/commit-evidence-pipeline` @ `c1a22b601b32a757da321e0761350c601cc0f794`

被审文件：`.agents/B2_PPO_PLAN.md`

## 1. 审阅来源

项目所有者要求在可用时通过 Claude Code 的 Opus 4.8/max 审计。Codex 以
只读工具运行：

```text
claude --model opus --effort max --tools Read,Grep,Glob ...
```

Claude 返回的实际 serving model 是 `claude-opus-4-8`，判定为
`APPROVE_WITH_BLOCKING_FIXES`。Claude 没有写文件或执行实验。

Codex 另行进行了三项只读交叉审计：

- 层级 action / exploration / replay / PPO 概率合同；
- 历史 `train_ppo.py`、canonical simulator 与 B2 环境接口；
- stale/dirty 远端上的隔离部署、GPU 独占与 eval shard 回收。

## 2. 总体判定

B2 是当前最小且目标对齐的路线：停止第三轮 warm-start 和第六个探针，直接
训练 PPO，并以 corrected overtake 与 any-agent collision 做 paired closed-loop
测量。

但原草案不能直接运行。若保持 `deterministic() == (logit > 0)`，fresh gate
从 `-6.0` 出发，在 20 iteration 的 PPO clip 预算内几乎必然仍低于 0；这会让
primary deterministic candidate 等于 BC，而不是对 PPO 能力的有效测量。原草案
还低估了 collision exposure、没有定义完整 episode rollout 边界，也没有把
双 offset、两 critic、RNG 与隔离远端部署落实成可执行合同。

以下裁定前瞻性修复这些问题；旧 B1/D2/D2R 结果不改写。

## 3. Blocking fixes 与最终裁定

### R1 — primary deterministic policy 使用冻结的 centered threshold

拒绝两种做法：

- 不修改 fresh `INITIAL_INTERVENTION_LOGIT=-6.0`；
- 不靠把 gate bias LR 提高到足以硬跨 `-6 -> 0`，因为 PPO clip 本身限制每轮
  behavior probability 的位移，高 LR 不能诚实解决可达性。

Primary iteration-20 deterministic contract 冻结为：

```text
top INTERVENE iff raw learned intervention logit > -6.0
conditional BRAKE iff raw learned conditional-brake logit > -6.0
continuous latents use learned means
all behavior exploration offsets are zero
comparison is strict >
```

因此 fresh zero-weight/equal-bias policy 仍严格 NO_OP、与 BC bit-identical；PPO
只需学出相对 fresh prior 的有符号状态依赖，而不是把稀有事件概率推到 50%。
该 threshold 在任何 outcome 前冻结，不从 development data 校准。标准 Bernoulli
mode（`logit > 0`）同时报告为 diagnostic，但不参与 primary selection。

### R2 — collision-enriched training curriculum，不改 eval panel

Task-8 training 1,640 rows 中，经已打开 D2 metadata 按 L2 join 后有 81 个
BC any-agent collision rows、61 个 collision L4。原 20×约13 episode 只会看到
约14个自然 collision events，不能支撑目标。

训练 scenario iterator 冻结为每 iteration 按 episode 数交替取：

- 50%：81 个 BC-collision rows 的无放回 keyed cycle；
- 50%：其余 1,559 rows 的无放回 keyed cycle。

每轮固定运行同一 seed 下预生成的16个完整 episodes（8 collision-bearing、
8 remaining），不能在 episode 中间切断；同 seed 的 A/B/C 因而看到完全相同
的 scenario list，不会因某臂早碰撞而改变后续训练分布。20轮固定暴露160个
BC-collision episodes，但不得
把“暴露”谎报成 candidate 实际 collision event。288 development panel 保持
完全不变，curriculum 标签不进入 actor/critic observation。

### R3 — exploration 期间冻结 dual

- iteration 1–9 只要 behavior exploration multiplier > 0，dual 固定为 `1.0`；
- iteration 10 起 offsets=0，每个完整 rollout 后只调用一次 dual update；
- floor 来自已打开 canonical BC outcome 按 L2 join 到同一冻结 50/50
  curriculum 的 rate `- 1 percentage point`，在训练前一次性冻结；P0 live
  simulator identity 负责核验 baseline parity；
- update 只使用该 iteration 已完成 episode 的 corrected overtake rate；
- dual 仍限制在 `[0,3]`，EMA/LR/32-episode规则保持不变。

这样避免随机探索本身造成的 overtake 损失把 dual 推到上界并压低 collision
权重。

### R4 — collision advantage 不得把空通道噪声标准化为单位信号

不用每 batch 的 collision standard deviation 强制归一。使用只由 event-bearing
rollout 更新的 running scale；在首个 event-bearing rollout 之前或 scale 无效时，
collision actor advantage 置零。performance channel 独立处理。scale、更新计数
和状态必须进入 checkpoint/resume；EMA decay 前瞻性锁为 `0.99`。

### R5 — keyed sampling 与完整 behavior replay

新增显式双 offset、非持久化 behavior context。每条 transition 保存 top/brake
offset、std scale 与 schedule id。`sample()` 必须支持按
`(seed,l2,repeat,macro,component)` 生成 top uniform、steer normal、brake uniform、
brake normal；即使分支 inactive 也固定生成全部四个 draw。PPO replay 用同一条
transition 的 context 重建 joint log-prob。不得复用会修改 policy state buffer 的
`apply_intervention_logit_offset()`。

### R6 — exploration 时降低随机 steering 干扰

训练 behavior 的 `steer_std_scale` 冻结为 `0.1`，`brake_std_scale=1.0`；所有
iterations 相同，不做 sweep。learned steer mean 仍可训练，动作空间不删转向。
Primary deterministic eval 不采样 std。

### R7 — 明确 terminal / rollout / minibatch 语义

- simulator 8 秒 horizon 是产品 task terminal：写入 corrected terminal outcome，
  两 critic bootstrap 都为 0；
- any-agent collision 是 task terminal，bootstrap 0；
- collector 每轮固定16个完整 episode，不按 macro 数截断；
- infrastructure abort 是 integrity failure，不是训练 truncation；
- 3 PPO epochs，minibatch size 固定 128，keyed deterministic shuffle，最后一个
  小 batch 不丢；
- optimizer 固定 Adam，`betas=(0.9,0.999)`, `eps=1e-8`；
- action core/adapter LR `3e-5`，gate/mean/std heads LR `3e-4`，C sidecar LR
  `3e-6`，两个 critics LR `5e-5`；每组独立 clip `0.5`。

### R8 — 新 B2 two-head schema，不污染 legacy 三通道

实现新的 `B2Critics`、`B2MacroRecord/Buffer` 和 two-channel critic loss；历史
`V22Critics`、`MacroRecord`、`separate_critic_losses` 保持 fail-closed 兼容。
B2 不实例化 legacy reward critic，也不把 dense reward/TTC/alarm 写进 actor。

### R9 — 安全远端执行是数值运行硬前置

不 checkout、pull、rsync 或删除远端 dirty worktree。代码先形成 clean local
commit，然后用该 commit 的 `git archive` 在两端创建仓库外隔离运行根：

```text
/home/haowei/end2race_runs/<run_id>/{repo,inputs,outputs,cache,control}
```

只显式 stage BC、完整 sidecar release 和 corrected Task-8 release。不可变
RunPlan 固定 source/input digests、环境、host、arm×seed queue、唯一输出和
checkpoint contract。两端 preflight 必须验证模块 `__file__` 落在 staged repo、
CLI capabilities、CUDA/GPU、remote `DISPLAY=:1`、独立 cache 和 GPU flock。
learner 不分 shard；seed0 A/B/C 在 remote 串行，seed1 A/B/C 在 local 串行。
冻结 checkpoint 后另建 EvalPlan，local shard0、remote 在一个 GPU lock 内依次
跑 shard1–3；回收后严格验证 288×7=2,016 paired rows 才能合并。

### R10 — implementation audit rulings before numerical execution

The second internal implementation audit found and prospectively resolved four
integrity gaps before any B2 numerical rollout:

- actor advantages are normalized/scaled once over the complete 16-episode
  rollout and then sliced unchanged into keyed minibatches; a singleton final
  minibatch is valid and must not be re-normalized;
- each iteration persists a full macro replay NPZ (deployable/privileged inputs,
  latent, behavior context, old log-prob/entropy, values, advantages, direct
  terminal signals and composition digests) plus top/conditional/joint action
  and per-head entropy diagnostics;
- resume is explicit, never implicit: it verifies the immutable plan/config,
  curriculum prefix, committed checkpoint/replay hashes, scenario-repeat cursor,
  RNG, dual, critics and all optimizers before continuing from the next complete
  iteration; an incomplete later iteration is quarantined and replayed from the
  previous committed boundary;
- EvalPlan checkpoints are bound to the parent training RunPlan.  Merge must
  fail closed unless the live BC baseline is exactly 24 collisions and 138
  corrected overtakes on all 288 opened-development rows, then report per-seed
  and pooled lexicographic verdicts.  A COMPLETE marker without the expected
  output envelope is not a completed job.

Dual timing is also frozen explicitly.  Iterations 1–9 neither update nor count
toward the zero-exploration dual warm-up.  Iteration 10 records the first 16
zero-exploration episodes but leaves lambda at 1; iteration 11 records the next
16, performs the first eligible post-rollout update, and iteration 12 is the
first actor update that can consume a changed lambda.  This lag is intentional:
the dual never uses outcomes from the rollout before that rollout's on-policy
PPO update, and exploration-contaminated outcomes do not ratchet the constraint.

## 4. Non-blocking claim boundaries

- A 与 B 不只是“有没有 sidecar”：A 的 BC adapter 是随机初始化，B/C sidecar
  是预训练的。因此 H_sidecar 只能解释成完整 representation arm comparison，
  不能声称是纯粹的冻结因果消融。
- 288 development panel 与 sidecar fit/warm-start L4 有重叠，只能做 opened-dev
  mechanism selection；六个比较不做 multiplicity inference，不能宣称泛化。
- `false_intervention <= 7/75` 是历史 calibration construction consistency，
  不进入 B2 模型成绩。
- historical `train_ppo.py` 提供 PPO 骨架；缺的是 B+ v2.2 的环境、两信号、
  层级 replay、checkpoint、paired evaluator 与 runner 接线，不能写成“PPO 一行
  都没有”。

## 5. 数值授权边界

项目所有者当前请求 Codex 托管执行 B2、审计和结果后的下一步优化，因此不再
需要为 Tasks 0–6 或已冻结的 20-iteration pilot 逐阶段请求输入。仍必须：

1. 先把 R1–R9 写入 live plan；
2. 完成单元/模拟器 plumbing tests；
3. Claude/Opus blocking review 结论在本文件留档；
4. source/input/run plan preflight 全通过；
5. 任一 integrity failure 停止受影响 queue 并修复，不能把不完整结果当 KPI；
6. B2 结果后只进行证据支持且仍直接优化 product KPI 的下一实验，不返回 TTC、
   第三轮 warm-start 或 outcome-tuned exploration sweep。
