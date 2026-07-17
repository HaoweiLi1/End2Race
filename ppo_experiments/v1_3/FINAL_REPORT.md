# End2Race PPO V1.3 统一结果

**状态：** `CLOSED_NO_PRODUCT_VERDICT`  
**整理日期：** 2026-07-17（Asia/Singapore）  
**部署建议：** 仍为 canonical BC `pretrained/end2race.pth`。

## 1. “完整结果”的口径

这里把“完整”定义为：实验到达了预注册允许的合法终态，而不是机械地要求所有原计划 seed 都启动。

- A、B 命中 fail-fast KL guardrail 后，协议要求停止后续 seed 和评价；它们是完整的阴性机制结论。
- D 的 3 个 fresh seed 都完成固定 U8 训练并通过 process gate；BC preflight 随后命中协议不一致停止条件。它是完整的训练/机制结论，但没有产品性能结论。
- C 的 formal run 被宿主 SIGSEGV 中断，E 的 candidate CPU 评价被 owner 停止，M 从未启动；三者不是完整性能实验，原目录和 raw artifacts 已清理。

## 2. 完整终态

| 实验 | 唯一变化 | 实际执行 | 合法终态 | 结论 |
|---|---|---|---|---|
| V1.3-A | 1 epoch，actor LR 固定为原来的 3 倍 | seed 20260718 到 U5 | `FAIL_KL_UNSTABLE` | U5 KL=`0.096712` > `0.02`；“只提高 LR”不是受控设置 |
| V1.3-B | 原 LR，`n_epochs=4` | seed 20260723 到 U5 | `FAIL_KL_UNSTABLE` | U5 KL=`0.119444` > `0.02`；“只增加 rollout reuse”不是受控设置 |
| V1.3-D | physical Gaussian steering；关闭 autograd multithreading 以消除已复现的宿主崩溃 | seeds 20260735/36/37 均到 U8 | `STOP_PROTOCOL_DRIFT` | 三 seed 训练稳定；因 BC 评价设备合同未冻结而停止 candidate 评价 |

### V1.3-D 三 seed 训练证据

| Seed | U1-U8 完成 | max KL | `[0.002,0.010]` 内 update | U8 GRU relative RMS | U8 head relative RMS |
|---:|:---:|---:|---:|---:|---:|
| 20260735 | 是 | 0.007611 | 8/8 | 0.000156622 | 0.001633347 |
| 20260736 | 是 | 0.015772 | 6/8 | 0.000142277 | 0.001485071 |
| 20260737 | 是 | 0.008354 | 6/8 | 0.000137447 | 0.001493497 |

三条 run 的 frozen actor 参数和 `log_std` 最大变化均为 0，U8 checkpoint 均为严格 12-key actor。由此可以确认：一个 PPO epoch 并非天然“监督太弱到无法更新 actor”；在修正 steering likelihood 几何后，它能在 3 个 fresh seed 上产生受控、可测且量级相近的非零参数位移。

这仍不等于“跨 seed 行为同方向改善”。本轮没有完成任何 D candidate 的 600-case CPU 评价，也没有测量跨 seed 参数更新向量的方向一致性。

## 3. 已完成并保留的诊断

### 3.1 Steering likelihood 机制

同一 seed 20260718 的单因素 probe 只把旧 atanh-squashed steering likelihood 换成 physical Gaussian：8 个 update 全部完成，max KL 从 A 的失控值 `0.096712` 降为 `0.010797`，U8 GRU/head relative RMS 分别为 `0.000147441` / `0.001559292`。该 probe 是机制证据，不是产品评价；整理后保存在 `ppo_experiments/v1_3/MECHANISM_PROBE.json`。

结合 A、B、该 probe 与 D，当前最窄结论是：旧 steering likelihood 几何是 KL 失控的关键机制；固定提高 LR 或增加 epochs 都不能绕过它。不能据此声称 PPO 已带来 collision 改善。

### 3.2 Evaluation device contract

D 把历史 CPU canonical reference `21 collision / 233 follow / 346 overtake` 与一次 CUDA 评价 `22 / 234 / 344` 比较，导致 exact-match gate 停止。后续诊断确认两者场景集合相同，但有 4 个边界 case 改变 outcome。

在历史合同下重跑——persistent spawned CPU workers、每 worker `torch.set_num_threads(1)`、模型每 worker 只加载一次——600-case BC 精确回到 `21 / 233 / 346`，与历史逐场景 outcome 及 final relative position 均无差异。正式规则因此是：BC 与 candidate 必须在同一冻结 CPU evaluator contract 下配对，禁止再用 GPU 结果对比历史 CPU baseline。完整记录见 `ppo_experiments/v1_3/EVALUATION_PROTOCOL.json`；已验证实现保留为 `ppo_experiments/v1_3/evaluate_cpu_pool.py`。

## 4. 证明、证伪与未回答问题

本轮证明或强支持：

1. A：单独把 actor LR 提高 3 倍会触发预注册 KL 失控。
2. B：单独把 PPO epochs 提高到 4 会触发预注册 KL 失控。
3. physical Gaussian 修正后，1 epoch 可以在 3 个 fresh seed 上稳定地产生非零、量级相近的 actor 更新。
4. evaluator device 是实验身份的一部分；CPU/GPU 边界漂移足以改变 collision/follow/overtake 计数。

本轮没有证明：

1. D actor 的 collision 低于 BC；
2. 三个 seed 的行为改进方向一致；
3. reward 与目标行为对齐；
4. 任一 V1.3 actor 通过 dev、holdout 或 deployment gate。

因此 `selection_performed=false`、`holdout_performed=false`、`promotion_performed=false`，`performance_verdict=null`。V1.3 不能被表述为性能改善。

## 5. 保留证据

- A：`ppo_experiments/v1_3_a/RESULTS.json`（SHA-256 `311e3030f93efd14eb07e976a66e26715a2d2b2624d341618ab8862c92927b4f`）
- B：`ppo_experiments/v1_3_b/RESULTS.json`（SHA-256 `5653fe2e9930983574f26a20fe74b11fda2331feeeb9b13de5a21f0152985fcc`）
- D：`ppo_experiments/v1_3_d/RESULTS.json`（SHA-256 `9ed27a8681dfea1bcf462d2c81844e8debffd56f45d4ecd0c97294096a31e7d9`）
- D recovery probe：`ppo_experiments/v1_3_d/RECOVERY_PROBE.json`（SHA-256 `636f35282afcb177dcc27574c092ab0bcff1625a0dd3e1edd9eb3cb1851e7e1d`）
- 统一机器可读摘要：`ppo_experiments/v1_3/RESULTS.json`

完整 A/B/D raw runs 与 checkpoints 保留在 `runs/ppo/v1_3_{a,b,d}*`。不完整 C/E/M 的专属 records、runs、logs、candidate copies 和 evaluation outputs 已删除；其中已独立完成并验证的 mechanism probe、CPU evaluator contract 和 evaluator implementation 已合并保留。
