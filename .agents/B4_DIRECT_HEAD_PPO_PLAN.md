# B4 Plain-End2Race Frozen-Feature Direct-Head PPO

状态：**IMPLEMENTED LOCALLY, BLOCKING TESTS PASS, AWAITING EXTERNAL IMPLEMENTATION REVIEW GO — NO RUNPLAN / NO GPU / NO NUMERICAL RESULT**

日期：2026-07-13（Asia/Singapore）

上游审计草案：`.agents/B4_DIRECT_HEAD_PPO_EXTERNAL_AUDIT_PLAN.md`（已被本文件取代）

本文件是 B4 唯一生效的前瞻性实验 authority。它批准实现、CPU 合同测试、
production-shaped 四地图 smoke 的实现准备和实现审阅；只有实现审阅返回 GO 后，
才允许创建一个不可变 B4 RunPlan。本文本身不授权 GPU 训练、288 开发面板评估或
fresh/final pool 访问。

---

## 1. Owner decisions

### D1-B：安全优先，corrected overtake 最多相对下降 5%

B4 的产品偏好为：优先降低 any-agent collision；corrected terminal overtake 可以
不变、提高，或相对 canonical BC 最多下降 5%。已打开的 288-scenario development
baseline 每 seed 为：

```text
collision = 24
overtake  = 138
```

因此 B4 的硬门是每 seed：

```text
overtake >= ceil(0.95 * 138) = 132
```

两 seed 都通过时 pooled 自然至少为 264；`pooled >=264` 只报告，不另设冗余硬门。
此变更只向前适用于 B4。B2 仍按其运行时冻结的 strict/1pp 目标保持 FAILED，B2/B3
历史文档和结论不得追溯改写。

### D2-B：暂停 B3 数值运行，优先 B4

plain `End2Race.state_dict()` 由原始 BC evaluator 严格加载已成为硬产品需求。因此：

```text
B3 = IMPLEMENTED, REVIEWED GO, PAUSED UNRUN
```

保留 B3 commits `19e83ae`、`21085bc`、计划、实现和测试证据；不创建 `plan-b3`，
不把 B3 写成 FAILED。B4 negative 后也不得自动运行 B3，必须另作前瞻性 owner 决定。

---

## 2. 单一可证伪问题

> 从 canonical BC 严格初始化、冻结原始 GRU feature provider、只微调原始
> `End2Race.output_layer` 的 plain PPO，能否在每 seed corrected overtake 至少
> 132/288 的条件下显著减少 any-agent collision？

准确实验名：

> **plain-End2Race frozen-feature direct-head PPO control experiment**

本实验不声称解决 side/rear representation 缺陷。head-only 与 residual head 都只
消费冻结的 1680D GRU feature；B4 检验的是更简单参数化和 strict deployment schema，
不是 representation sufficiency。

---

## 3. 模型和 checkpoint 合同

### 3.1 Deployment actor

`model.py` 中现有 `End2Race` 类保持不变。canonical actor 共有 11,301,482 参数，
原始 `output_layer` 有 706,862 参数（6.2546%）。初始化必须：

```python
actor = End2Race(mask_prob=0.0, hidden_scale=4)
actor.load_state_dict(bc_state, strict=True)
```

冻结：

```text
k
speed_mlp.*
dummy_embedding
gru.*
```

唯一可训练 actor 参数：

```text
output_layer.*
```

训练期另建独立 scalar critic：

```text
12 -> 128 -> SiLU -> 128 -> SiLU -> 1
```

最终 candidate 只保存 `actor.state_dict()`，且 key/shape 必须与 canonical BC 的 12
个 keys 完全相同。禁止写入 deployment checkpoint：

```text
critic.*
std / log_std
res_head.* / residual_budgets
sidecar.* / action_core.*
intervention_gate.* / brake_gate.*
optimizer / scheduler / RNG
```

full resume checkpoint 与 actor-only snapshot 分离。每次 candidate 保存前，将冻结
tensors 与 iteration-0 snapshot 做 exact equality；actor-only 文件必须通过一个不含
residual autodetection 的 plain `End2Race(...).load_state_dict(..., strict=True)` fixture。

### 3.2 Frozen feature replay

rollout 每个 100 Hz step 在 `torch.no_grad()` 下保存 1680D GRU feature。PPO update
只重放 `output_layer(feature)`，不重跑冻结 GRU，也不对 800-step 序列反向传播。
update 前必须证明保存动作在未变参数下 replay ratio 为 1；只有 `output_layer.*`
产生 gradient。

---

## 4. 动作概率合同

首轮不实现 truncated Gaussian。PPO 的策略动作变量是 raw latent：

```text
u_steer ~ Normal(mu_steer, 0.03)
u_speed ~ Normal(mu_speed, 0.20)
std fixed; entropy coefficient = 0
```

simulator 执行声明式、固定且无参数的 actuator projection：

```python
a_steer = clip(u_steer, -0.52, 0.52)
a_speed = clip(u_speed, 0.0, 20.0)
```

buffer 必须同时保存：

```text
raw latent u
old log_prob(u)
executed projection(u)
projection delta = executed - raw
frozen actor feature
12D privileged critic feature
episode identity, step index, terminal boundary
```

PPO ratio 只在同一个 raw latent 上定义：

```text
r = exp(new_log_prob(u) - old_log_prob(u))
```

多个 latent 投影为同一物理命令不会破坏该 latent-policy probability contract；越界
只形成探索浪费/saturation，必须以 projection frequency、每维 count、mean/max
absolute delta 完整报告。不得把 projection 隐藏，也不要求 stochastic projection
绝对为零。

deterministic product evaluator 使用 actor mean。steering 的 `[-0.52,0.52]` projection
与原 evaluator 一致；promotion 前必须证明 288 deterministic episodes 上 mean speed
全部在 `[0,20]`，即 deterministic speed projection count 为 0。

---

## 5. 数据、episode、reward 和 credit

使用已打开且与 288 development L2-disjoint 的 Task-8 training population：

```text
BC collision =   81
BC overtake  = 1001
BC follow    =  558
total        = 1640
```

每 seed 在运行前冻结完整 30-iteration order。每 iteration 16 个 complete episodes：

```text
6 BC-collision
6 BC-overtake
4 BC-follow
```

各 outcome group 内按 seed keyed order 无放回轮换；用尽后进入带 repeat id 的下一轮。
不加入 stage curriculum、hard-start、lateral offset 或 sampler sweep。

episode 语义：

```text
第一次 any-agent collision -> true terminal, bootstrap 0
8 s product horizon        -> true terminal, bootstrap 0
infrastructure failure     -> invalid; 不进入 update
```

episode 结束后使用 paired product evaluation 的 corrected classifier。唯一训练 reward
只写在最后 transition：

```text
R_terminal = -2 * collision_any + 1 * (corrected_outcome3 == "overtake")
```

paired diagnostic：

```text
DeltaR_pair = 2 * (C_BC - C_pi) + (O_pi - O_BC)
```

paired delta 不进入 actor loss。TTC、dense clearance/progress、dual、BC anchor、
warm-start loss 均不得进入首轮 B4 objective。标量 reward 不保证 5% 门；若最终不
满足 hard gate，B4 即 substantive negative。

GAE 固定：

```text
gamma      = 0.999
gae_lambda = 0.997
```

在 100 Hz 下 `(gamma * lambda)^100 ~= 0.67`。每个 episode 独立反向计算，terminal
zero bootstrap，严禁跨 episode 传播。

---

## 6. PPO 与 weighting 合同

冻结参数：

| 项目 | 值 |
|---|---:|
| actor LR | `3e-5` |
| critic LR | `3e-4` |
| PPO clip | `0.10` |
| target weighted KL | `0.015` |
| actor epochs | 最多 `3` |
| critic epochs | 固定 `3` |
| minibatch | `1024` transitions，保留 tail |
| gamma / lambda | `0.999 / 0.997` |
| entropy coefficient | `0` |
| mean bound coefficient | `0.01` |
| BC anchor / dual | `0 / none` |
| actor / critic grad norm | 分别 `0.5` |
| iterations / seeds | `30 / (0,1)` |
| snapshots | `0,10,20,30` |
| architecture arms | `1` |

actor 对 episode 等权。若 rollout 有 `N` transitions、`E=16` episodes、episode i
长度为 `T_i`，每个 transition 的权重：

```text
w_it = (N / E) / T_i
```

该权重用于：

- weighted advantage mean/variance normalization；
- clipped actor surrogate；
- actor weighted KL 和 weighted clip fraction；
- actor KL early stop（使用 weighted full-rollout KL）。

同时报告 actor KL/clip fraction 的 weighted 和 ordinary unweighted 版本。critic 只用
普通 unweighted per-state MSE，不复制 actor episode weighting。

update 必须拆分：

```text
actor loop:
  最多 3 epochs；每 epoch 后 full-rollout weighted KL > 0.015 时停止后续 actor epoch

critic loop:
  始终完成 3 epochs；不消费 actor early-stop flag
```

actor optimizer 只能持有 `output_layer.*`，critic optimizer 只能持有 critic 参数；
梯度分别裁剪，不允许 combined optimizer。

mean bound penalty 只作用于 actor mean（steer outside `[-.52,.52]`、speed outside
`[0,20]`）；它不被表述为 sample hard bound，sample 的执行边界来自固定 projection。

---

## 7. Blocking preflight

只允许以下会使结果无效的 blocking checks：

1. canonical BC strict load；iteration-0 actor state/action/hidden/trajectory/outcome identity；
2. raw latent、stored latent、old/new log-prob、固定 projection 和 projection ledger replay；
3. any-agent/horizon true terminal、terminal reward、zero-bootstrap、GAE 不跨 episode；
4. 两个不同长度 synthetic episodes 的 episode-weighted actor/advantage/KL 手算回归；
5. actor/critic optimizer 与 gradient isolation；frozen tensors exact equal；
6. actor-only checkpoint 由 plain strict loader 加载且只有 canonical keys；
7. production-shaped 四地图 smoke，验证 simulator wiring、完整 episode、atomic output。

不得添加 TTC/Brier、warm-start recall、sidecar representation、恒真 threshold 或
“stochastic projection 必须为 0”等代理门。

---

## 8. Snapshot、resume 与运行纪律

- iteration 0 actor snapshot 必须与 BC tensor-by-tensor exact equal；
- actor-only snapshots 只在 0/10/20/30；
- 每个完整 iteration 后原子保存 full resume checkpoint；
- resume 恢复 actor、critic、两个 optimizer、iteration、curriculum order/cursor、RNG；
- 不从半 episode 或半 PPO epoch 恢复；
- seed0 的中间结果不得改变 seed1 或后续训练；
- 实现 review GO 后只创建一个 immutable B4 RunPlan，运行两个 30-iteration seeds；
- implementation review GO 前不得创建 RunPlan 或启动 GPU。

---

## 9. Frozen 288x7 evaluation

唯一 EvalPlan variants：

```text
BC
seed0_iter10
seed0_iter20
seed0_iter30
seed1_iter10
seed1_iter20
seed1_iter30
```

共 `288 * 7 = 2016` paired rows。candidate selection 只能比较同一 iteration 的两
seed pair：iter10、iter20、iter30；禁止 seed0/seed1 选择不同 iteration。

完整性要求还包括所有 candidate 的 deterministic mean speed projection count 为 0。

每个 seed feasibility：

```text
corrected overtake >= 132
any-agent collision <= 24
```

pooled feasibility：

```text
fixed_collision > new_collision
两个 seed 的 collision 都相对 BC 同方向改善
```

产品 collision 目标：

```text
each seed <= 16
pooled    <= 33
```

选择顺序：

1. 丢弃完整性失败；
2. 丢弃任一 seed overtake `<132`；
3. 丢弃任一 seed collision `>24`；
4. 丢弃 pooled `fixed_collision <= new_collision`；
5. 丢弃任一 seed collision 未改善；
6. 选择 pooled collision 最低的同-iteration pair；
7. 并列时选 pooled overtake 更高；再并列选更早 iteration。

达到 feasibility 但未达到 collision product target 的结果只能称为
`opened-development directional survivor`，不能称为产品成功。

若没有合格 pair，B4 为 substantive negative 并停止。不得自动：

```text
运行 B3 fallback
解冻 GRU
修改 reward/std/LR/clip/KL
延长到 40/60 iterations
加入 second arm / sweep
打开 fresh/final pool
```

任何后续动作需要基于完整 B4 报告另写前瞻性 owner decision。

---

## 10. Evidence-preservation boundary

- 不删除或重写 legacy PPO、Residual、B2/B3 policy/loaders/tests；
- B2 继续保持其历史 FAILED；
- B3 继续保持 `IMPLEMENTED, REVIEWED GO, PAUSED UNRUN`；
- B4 使用独立 versioned direct-head implementation，复用已审计 scenario loader、
  12D critic feature、corrected classifier 和 paired simulator mechanics，但 B4 actor
  路径不得实例化 residual、sidecar、gate、action core 或 dual；
- implementation diff、CPU matrix、production smoke 结果和 review verdict 写入本文件
  的实施记录；implementation review GO 前不得创建 B4 RunPlan。

---

## 11. Implementation review checklist

外部 reviewer 必须从实际 diff 与测试证据回答：

1. raw latent 是否是 sampling、storage 和 PPO log-ratio 的同一变量；
2. fixed projection 是否声明、可重放且完整记账；
3. actor path 是否确实只有 plain `End2Race.output_layer` 可训练；
4. reward/GAE/terminal 是否逐 episode 且符合本文件；
5. actor weights、weighted normalization/KL 与 unweighted critic 是否实现一致；
6. actor early stop 是否不影响三个 critic epochs；
7. resume 和 actor-only snapshot 是否 fail closed；
8. paired evaluator 是否强制同-iteration两 seed selection、per-seed 132 门和 zero
   deterministic speed projection；
9. B2/B3 tests/evidence 是否未被改写；
10. 是否存在会使 KPI 无效的 blocker。不得用新增 TTC/warm-start/sidecar proxy gate
    代替产品结果。

---

## 12. Implementation record

### 12.1 Source boundary and status

实现基线是 local commit `4b06b7a`；B4 external-review boundary 是紧随其后的、
包含本计划、B4 modules、B4 tests 和 control-plane wiring 的本地 commit。该 commit
只是为了冻结外审证据，不代表 implementation review GO。B3 commits `19e83ae`、
`21085bc` 及其历史文件没有被删除或重写。

截至本记录：

```text
B4 code                       IMPLEMENTED LOCALLY
CPU blocking contracts       PASS
four-map simulator identity  PASS (CPU, no RunPlan)
external implementation review  PENDING
B4 RunPlan / staging         NOT CREATED
GPU learner                  NOT STARTED
288x7 EvalPlan               NOT CREATED
B4 numerical/KPI result      NONE
fresh/final pool             SEALED
```

### 12.2 File map

- `bplus_v22/b4_direct.py`：冻结配置、plain actor strict loader、fixed-std raw Normal、
  fixed projection、complete-episode GAE、episode-equivalent actor weights、分离的
  actor/critic update、curriculum、actor-only/full-resume checkpoint；
- `bplus_v22/b4_env.py`：100 Hz complete-episode simulator collector、any-agent/horizon
  true terminal、terminal-only `-2C+O`、raw/executed/projection ledger；
- `bplus_v22/b4_eval.py`：plain deterministic actor adapter、288x7 paired rows、同
  iteration 两-seed feasibility/selection、projection guardrail；
- `bplus_v22/b4_runner.py`：RunPlan validation、四地图 smoke、30-iteration learner、
  iteration-atomic replay/checkpoint/ledger 与 fail-closed resume；
- `bplus_v22/b4_cli.py`、`bplus_v22/cli.py`：B4 baseline/smoke/learner/eval/merge CLI；
- `Experiments/runner.py`：两 seed train topology、四 eval shards、immutable
  `plan-b4`/`plan-b4-eval` builders 和 strict collection envelopes；builder 已实现但
  本轮没有调用；
- `tests/test_b4_direct.py`：actor/action/GAE/weight/update/checkpoint/resume/curriculum；
- `tests/test_b4_control_plane.py`：frozen plan、marker、host/job topology；
- `tests/test_b4_eval.py`：完整 2016-row Cartesian product、paired diagnostic 和
  same-iteration selection；
- `tests/test_b4_simulator_smoke.py`：可复现的四地图 iteration-0 production-shaped
  trajectory/outcome identity。

`model.py` 的 `End2Race` 类未修改。B4 deployment snapshot 由
`save_actor_snapshot()` 只保存 `policy.actor.state_dict()`；strict fixture 拒绝 full
resume checkpoint 和任何非 canonical 12-key mapping。

### 12.3 Blocking test evidence

以下命令在 2026-07-13 当前 working tree 执行并通过：

```bash
PYTHONPATH=. python tests/test_b4_direct.py
PYTHONPATH=. python tests/test_b4_control_plane.py
PYTHONPATH=. python tests/test_b4_eval.py
NUMBA_CACHE_DIR=/tmp/end2race_b4_cpu_smoke_20260713_repro \
  PYTHONPATH=.:f1tenth_gym/gym python tests/test_b4_simulator_smoke.py
```

覆盖结果：

1. canonical 12 keys、11,301,482 total parameters、706,862 trainable head parameters；
2. raw Normal log-prob 与 `torch.distributions.Normal` 一致，projection 可 exact replay；
3. pre-update ratio-one tolerance、terminal-zero-bootstrap GAE、约 0.67 的 1 s
   propagation、跨 episode 零泄漏；
4. 每个 episode actor total weight 相同，weighted normalization 手算回归通过；
5. aggressive-LR synthetic update 触发 actor KL early stop，而 critic 仍完成 3 epochs；
6. 只有 output head/critic 分别变化，frozen actor exact equal；
7. actor-only strict roundtrip、full checkpoint restore、非-BC iteration-0 resume 拒绝；
8. pool `81/1001/558`、每轮 `6/6/4`、两 seed 30-iteration curriculum digest 固定；
9. 288x7 必须为 2016 unique paired rows；损坏 L2 或 paired transition diagnostic 会
   fail closed；只选择同 iteration seed pair，per-seed 132 门生效；
10. 四地图 production-shaped identity 如下，未比较 candidate KPI：

| map | steps | terminal | trajectory diff | outcome identity | speed projection | steer projection | replay log-prob delta |
|---|---:|---|---:|---|---:|---:|---:|
| Austin | 801 | product horizon | 0 | true | 0 | 0 | 0 |
| Hockenheim | 801 | product horizon | 0 | true | 0 | 0 | 0 |
| MoscowRaceway | 801 | product horizon | 0 | true | 0 | 0 | 0 |
| Nuerburgring | 801 | product horizon | 0 | true | 0 | 0 | 0 |

`py_compile` 对五个 B4 modules、两个修改入口也通过；`run.sh list`、
`plan-b4 --help`、`plan-b4-eval --help`、CLI capabilities 与 B4 pilot/eval help 均可加载。

### 12.4 Compatibility matrix

以下九个既有 B2/B3 CPU programs 在最终 B4 code diff 上全部通过：

```text
test_bplus_v22_exploration.py
test_bplus_v22_objective.py
test_bplus_v22_ppo.py
test_bplus_v22_ppo_buffer.py
test_bplus_v22_ppo_env.py
test_bplus_v22_remediated_model.py
test_bplus_v22_ppo_runner.py
test_bplus_v22_ppo_eval.py
test_experiment_runner.py
```

`test_experiment_runner.py` 的 terminal baseline-acceptance-failure 文本是其临时目录内
预期的 negative-path fixture；program 最终为 `ALL TESTS PASSED`。Gym 的 upstream
deprecation warning 和 RK4 warning 也不构成 test failure。

### 12.5 Problems found and fixes

1. **100 Hz horizon one-frame drift**：最初用带 tolerance 的 `lap_time` 比较，在浮点
   累积下于 800 steps 停止，而原 product loop 的字面 `lap_time < 8` 会执行 801
   steps。B4 改为与 product evaluator 相同的 literal boundary；四地图完整数组随后
   exact equal。概念上的 8 s product horizon 没有改变。
2. **Replay floating-point contract**：逐 step forward 与 batched GEMM 在 float32 下可有
   约 `1e-5` 的重放差异。实现将 blocker 明确为 `max |ratio-1| <= 1e-4`，而不是声称
   bitwise arithmetic identity；raw latent 和公式仍完全相同。当前四地图 CPU smoke 的
   observed log-prob delta 为 0。
3. **PyTorch 2.6 full-resume loading**：含 Python/NumPy RNG tuple 的私有 full checkpoint
   不能使用 state-dict-only `weights_only=True`。实现仅对 RunPlan/hash-bound full
   checkpoint 显式使用 `weights_only=False`，并强制 `map_location="cpu"`，避免 GPU
   resume 时把 `torch_cpu` RNG byte tensor 错映射到 CUDA。actor-only deployment loader
   始终 `weights_only=True`。
4. **Resume prefix gap**：resume 现在单独 strict-load `iter_0000.pth` 并核对 canonical BC
   tensor hash，同时要求 loaded checkpoint iteration 与 committed ledger 长度相同。
5. **Paired diagnostic trust**：merge 不再直接信任 shard 的 fixed/new/gained/lost 字段，
   而是从同一 row 的 frozen BC fields 重算并校验后才汇总。
6. **Retained actor tail**：actor tail 不按自身长度重新归一成一个完整 optimizer step；
   它沿用 full-minibatch denominator。critic 按 owner 决定保留普通 unweighted
   per-state MSE。

### 12.6 Known limits and review boundary

- 四地图 smoke 是当前 working tree 上直接运行的 CPU production-shaped fixture；由于
  RunPlan 尚未获准创建，它不是 staged/GPU marker。review GO 后必须在 immutable staged
  source 上重新运行同一 smoke，不能把本次 CPU 结果当成 staging authorization。
- 已测试 full-checkpoint CPU roundtrip、RNG payload contract 和 resume fail-closed
  invariants；尚未做真实 GPU interruption/resume，因为这会越过当前 authority。
- `b4_env.py` 调用 legacy `compute_shaped_reward()` 的返回值会被丢弃；该调用只更新已
  批准的 12D privileged critic feature 中三个 past-only phase fields。buffer actor reward
  仍只有 terminal `-2C+O`。reviewer 应从 replay/ledger 证实没有 dense term 进入 reward。
- 尚无 stochastic simulator rollout 的性能证据，也无 overtake/collision result。projection
  是否造成探索浪费只能在获准 pilot 后报告，不能从 deterministic smoke 推断。
- B4 仍是 frozen-feature control；任何 positive/negative result 都不能证明 GRU
  representation 充分或不足。

外部 reviewer 应按 §11 从 actual diff 回答 correctness 问题。GO 只授权下一步创建唯一
immutable B4 RunPlan；GO 本身不等于数值成功。若有 blocker，应保持当前无 RunPlan 状态
并在本节追加审计结论，而不是静默改参数。
