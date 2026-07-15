# SB3-Contrib GRU RecurrentPPO proof of concept（历史记录）

日期：2026-07-15（Asia/Singapore）

## 结论

```text
Historical recurrent-core POC: PARTIAL PASS
Real integration before repairs: FAIL
```

历史 POC 证明了：无需 fork SB3、无需修改 site-packages、无需把 End2Race GRU 改成 LSTM，即可使用 SB3-Contrib 2.7.1 的 stock `RecurrentPPO`、stock `RecurrentRolloutBuffer` 和 stock PPO update 正确重放 pre-action GRU hidden。

但严格审查后，该版本的真实 F1Tenth reset、speed feature 时序和 raw/executed action 合同均不合格，因此不是完整 integration PASS。修复后的实现与新验收数据见 `docs/SB3_GRU_REPAIR_REPORT.md`。下文的数据保留为 repair 前的历史记录。

本次只运行两个短合成 F1Tenth legacy-API env、一个 20-transition rollout 和一次 `learning_rate=0` 的 PPO train。没有执行正式 End2Race learner 训练，没有 reward shaping，没有运行大规模评估，也没有写入或覆盖 `pretrained/end2race.pth`。

## 1. 新增文件

| 文件 | 用途 |
|---|---|
| `rl/sb3_end2race_policy.py` | `End2RaceGRUPolicy`、GRU/LSTM-state adapter、BC-compatible actor export |
| `rl/end2race_gymnasium_env.py` | legacy F1Tenth 到 Gymnasium 的 observation/termination wrapper |
| `scripts/smoke_sb3_gru.py` | 两个并行 env、一个 rollout、一次零学习率 train 及全部测量 |
| `tests/test_sb3_gru_integration.py` | A-E acceptance tests 和 zero-lr smoke assertion；使用 Python 标准库 `unittest` |
| `docs/SB3_GRU_POC_REPORT.md` | 本报告 |

没有修改 `model.py`、`train.py`、`eval_multiagent.py`、`pretrained/end2race.pth`、现有 PPO 历史文件或任何第三方库文件。没有创建 Git commit。

## 2. 固定环境

```text
Python                  3.10.19
torch                   2.7.0+cu128
gymnasium               1.2.3
stable-baselines3       2.7.1
sb3-contrib             2.7.1
```

固定环境未安装 `pytest`，因此没有增加依赖；测试文件使用 Python 3.10 标准库 `unittest`，也仍可由兼容的外部 test runner 收集。

## 3. Policy 设计

### 原始 actor 与 checkpoint

`End2RaceGRUPolicy(RecurrentActorCriticPolicy)` 直接持有：

```python
self.end2race_actor = End2Race(mask_prob=0.0, hidden_scale=4)
self.end2race_actor.load_state_dict(bc_state_dict, strict=True)
```

actor 保留原始的 12-key schema：learnable LiDAR preprocessing `k`、`dummy_embedding`、speed MLP、单层 GRU 和 output layer。没有复制或重写这些计算。actor GRU 接口为：

```text
input_size  = 420
hidden_size = 1680
num_layers  = 1
batch_first = True
```

Policy 的 Gaussian mean 直接取 `End2Race.forward()` 的两维输出，不经过新的 actor MLP/action head。`policy.forward(..., deterministic=True)` 与原始 actor 做了逐 timestep 对比。可训练 `log_std` 只影响采样方差和 log probability；把 `log_std` 人为改变 `+1` 后 deterministic action 的变化仍为 `0.0`。

critic 是独立的 feed-forward value network，输入相同的 361D deployment observation。它不读取或修改 actor hidden，也不进入 deterministic actor mean。

### GRU dummy-cell transport

SB3 继续使用：

```text
RNNStates(
  pi=(actor_h, actor_dummy_c),
  vf=(zero_h_v, zero_c_v),
)
```

actor 的 `h` 是唯一真实 memory。每次 recurrent call 前，只对 `episode_starts=True` 的 sequence/env slot 清零 `h`；其他并行 env 保持原状态。输入 `c` 完全忽略，输出 `c` 始终为 `zeros_like(h)`。

`GRUWithLSTMStateInterface` 暴露 SB3 所需的：

```text
num_layers
hidden_size
input_size
forward(time_major_x, (h, c))
```

adapter 将 SB3 time-major input 转给原始 batch-first GRU，返回 `(next_h, zero_dummy_c)`。接口测试使用非零 random dummy cell，adapter output/hidden 与直接调用原始 GRU 的最大误差为 `0.0`，输出 cell 最大绝对值为 `0.0`。

`RecurrentPPO` 从 `policy.lstm_actor.num_layers/hidden_size` 建立 stock recurrent buffer；本 POC 没有覆盖 `RecurrentPPO.collect_rollouts()`、`RecurrentPPO.train()` 或 `RecurrentRolloutBuffer`。

### 严格 replay identity 的执行方式

同一 GRU 在 rollout 和 padded minibatch 中可能面对不同 `n_seq` batch packing，CPU BLAS 的浮点累加顺序会随 batch width 产生约 `1e-6` 级舍入差。没有放宽验收阈值；POC 的 actor replay 对每个 sequence slot 使用相同的 batch-size-1 GRU kernel，然后重新组合 sequence batch。这样：

- 每段内部仍保留可微 recurrent graph/BPTT；
- 不改变 stock buffer、padding、mask 或 PPO update；
- rollout 与 replay 的 kernel packing 一致，最终 logp identity 为精确 `0.0`；
- 代价是吞吐量较低。正式实现前需要评估 vectorized 优化，但不能删除 replay identity 门禁。

## 4. Gymnasium F1Tenth wrapper

`End2RaceGymnasiumEnv` 包装现有 legacy 或 Gymnasium-style F1Tenth env，不修改 F1Tenth 源码。

actor observation 是固定 `(361,) float32` array：

```text
[360D ego LiDAR, previous ego speed]
```

synthetic legacy core 特意提供了 ego/opponent pose、reference geometry 等 privileged fields；wrapper 输出仍只有上述 361 个 deployment values。policy 不接收 opponent pose、collision risk 或 reference geometry。

termination mapping：

| 原因 | terminated | truncated |
|---|---:|---:|
| collision | `True` | `False` |
| sim-duration timeout | `False` | `True` |

collision 优先于同时到达的 timeout。wrapper 原样返回 base reward，不进行 reward shaping。timeout terminal-value correction 仅由 stock `RecurrentPPO.collect_rollouts()` 完成。

本次动态测试通过一个 legacy-API synthetic F1Tenth core 执行 wrapper，以避免正式仿真训练或大规模评估；实际 F1Tenth map/vehicle dynamics 不属于本 POC 的验证范围。

## 5. 五项测试结果

### A. BC sequence identity — PASS

对 100 个连续 synthetic LiDAR/speed timestep，分别运行原始 `End2Race` 和 SB3 policy public deterministic forward path：

| 指标 | 结果 | 阈值 |
|---|---:|---:|
| max action absolute error | `0.0` | `<= 1e-6` |
| max hidden absolute error | `0.0` | `<= 1e-6` |
| adapter output/hidden max error | `0.0` | `<= 1e-6` |
| deterministic action error after changing `log_std` | `0.0` | `0` |

`log_std` 是 trainable parameter，并确实存在于 policy optimizer parameter groups。

### B. Episode reset identity — PASS

使用 2 个并行 state slot，env 0 在 step `0, 3` reset，env 1 在 step `0, 5` reset：

| 指标 | 结果 |
|---|---:|
| max action identity error | `0.0` |
| max hidden identity error | `0.0` |
| reset slot vs fresh zero-state max error | `0.0` |
| dummy cell max absolute value | `0.0` |
| 未结束 env 与错误 zero-reset 结果的最小差距 | `1.9730424881` |

最后一个非零 margin 证明一个 env reset 时，没有错误清零另一个 env 的 hidden。

### C. PPO replay identity — PASS

通过 stock `RecurrentPPO.collect_rollouts()` 收集 2 env × 10 step。任何 optimizer step 之前，迭代 stock recurrent rollout buffer，并调用 custom policy 的标准 `evaluate_actions()` API：

| 指标 | 结果 | 阈值 |
|---|---:|---:|
| valid timesteps | `20` | — |
| padding timesteps | `3` | `> 0` |
| episode-start flags | `5` | `>= 2` |
| max `abs(new_logp-old_logp)` | `0.0` | `<= 1e-6` |
| mean `abs(new_logp-old_logp)` | `0.0` | `<= 1e-7` |
| max `abs(ratio-1)` | `0.0` | `<= 1e-6` |

覆盖了普通连续 sequence、错位 episode boundary、timeout truncation、padding mask 和 2 个 parallel env。

### D. Timeout/collision bootstrap — PASS

rollout 内包含 2 次 collision termination 和 1 次 timeout truncation：

| 指标 | 结果 | 阈值 |
|---|---:|---:|
| collision reward vs raw reward error | `0.0` | `<= 1e-7` |
| timeout reward vs `raw + gamma * V(terminal_obs)` error | `2.5629997e-08` | `<= 1e-6` |
| terminal advantage vs `corrected_reward - V(current)` error | `0.0` | `<= 1e-6` |

因此 collision 使用 zero terminal bootstrap；timeout 使用 terminal observation value；terminal transition 的 advantage 没有跨入自动 reset 后的新 episode。

### E. Actor checkpoint compatibility — PASS

测试严格执行 actor-only export：

```python
torch.save(policy.end2race_actor.state_dict(), output_path)
model = End2Race(mask_prob=0.0, hidden_scale=4)
model.load_state_dict(torch.load(output_path), strict=True)
```

结果：12 个 actor keys 与 BC schema 完全相同，strict load 成功，round-trip max error `0.0`。`output_path` 位于自动删除的临时目录；没有覆盖任何已有 checkpoint。

## 6. Smoke train 结果

```text
parallel envs:                  2
最长 episode simulation time:  0.7 s
rollouts:                       1
PPO train calls:                1
learning rate:                  0.0
max parameter delta:            0.0
```

脚本要求的摘要输出：

```text
actor identity max error: 0
hidden identity max error: 0
replay logp max error: 0
ratio max deviation: 0
timeout bootstrap result: max error=2.56299972357e-08
strict checkpoint load result: True
```

## 7. 是否建议进入正式实现

**建议进入正式 End2Race PPO 的工程实现阶段，但本 POC 本身不应当作训练配置直接扩展。** 它已经证明 recurrent state transport、BC actor identity、stock buffer/update、episode reset、timeout 和 actor-only checkpoint schema 都可行。

正式训练前仍需由 owner 审阅并解决：

1. 在真实 F1Tenth env 上做短的 wrapper contract test，包括实际 reset/step API、timestep 和 terminal observation。
2. 明确连续 action 的物理 bounds、sample clipping 和部署时 steering/speed 语义；不能改变 deterministic BC mean。
3. 评估 batch-size-1 per-sequence GRU replay 的吞吐量，再做不破坏 identity test 的 vectorized 优化。
4. 另行设计 reward、opponent controller、checkpoint naming/retention；这些不属于本 POC，也未在此调整。
5. 保留 A-E tests 作为后续实现的 hard gate，不通过时报告精确失败点，不得放宽阈值。

## 8. 复现

```bash
/home/haowei/miniconda3/envs/end2race/bin/python scripts/smoke_sb3_gru.py
/home/haowei/miniconda3/envs/end2race/bin/python -m unittest discover -s tests -p 'test_sb3_gru_integration.py' -v
```

本 POC 未提交 Git commit，等待 owner 审阅。
