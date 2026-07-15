# SB3-Contrib GRU 极小非零学习率验证报告

## Verdict

**历史非零实验严格总 verdict：FAIL**

**数值与 optimizer 管线子结论：PASS**

**修正后 snapshot visual replay：PASS_VISUAL_REPLAY**

### Post-audit 勘误

原始 `training_rollout.mp4` 虽然有 100 个可解码帧，但未注册 evaluator 的跟车相机 callback。车辆坐标经 renderer 的 `x50` 缩放后位于默认原点相机范围之外，所以画面只有赛道，ego 和 opponent 不可见。此前只验证容器、帧数、FPS 和可解码性，错误地把它标记为交互视频 PASS。

最初验收要求把“training rollout 视频成功”列为总 PASS 的必要条件，因此不能用后续 replay 追认历史 training rollout：原始严格总 verdict 修正为 **FAIL**。原始 `results.json` 和 MP4 保留作为未通过语义检查的历史证据，没有事后改写。

修复内容：

- future training recorder 在每次 render 前把相机中心设置为 ego；
- overlay 显示 step、ego/opponent steering 与 speed、collision 和 timeout；
- 原始 RGB 帧与 MP4 解码帧都必须逐帧检测到中央黄色 ego 和红色 opponent；
- 仅“文件存在且可解码”不再能通过视频门禁。

修复后使用原实验的 pre-update policy/RNG snapshot 做了独立、零 optimizer 的 visual replay。它验证相机和动作可视化设置，但明确不是原始 training rollout。

本次实验仅验证真实 F1Tenth 交互、stock recurrent rollout、stock PPO update、梯度、参数更新、动作合同和 actor-only checkpoint 的训练管线。诊断 reward 不是正式 reward；本结论不表示 PPO 驾驶性能、避碰或超车能力有效，也不授权开始长时训练。

执行了且仅执行了：

- 1 次 `_setup_model()`；
- 1 次 `_setup_learn()`；
- 1 次 `RecurrentPPO.collect_rollouts()`；
- 1 次 `RecurrentPPO.train()`；
- 1 次 `optimizer.step()`。

没有调用 `learn()`，没有第二个 rollout、epoch、minibatch 或 optimizer update。

## 环境与 Git 基线

| 项目 | 值 |
|---|---|
| Python | 3.10.19 |
| PyTorch | 2.7.0+cu128 |
| Gymnasium | 1.2.3 |
| stable-baselines3 | 2.7.1 |
| sb3-contrib | 2.7.1 |
| device | CUDA，NVIDIA GeForce RTX 4080 SUPER |
| Git commit | `25818303af247be875e51febf1250bd071214e3c` |
| pretrained SHA256（前/后） | `b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4` / 相同 |

执行前工作树已包含 owner 尚未提交的 repair POC 修改。本实验未提交 commit，也未覆盖 `pretrained/end2race.pth`。

运行内对第三方 Python 源码树计算的前后摘要完全相同：

| package | Python 文件数 | SHA256 |
|---|---:|---|
| stable-baselines3 | 64 | `09807016889f42529bbe3e962213f03b1d1ed2dd2298ae2993ec94a313ae2b22` |
| sb3-contrib | 43 | `c36f336d40d9b09151f30be3ddb99395a443e93edd5217a47355327eb0b59a7a` |

结论：没有修改 site-packages、stock `RecurrentPPO` 或 stock `RecurrentRolloutBuffer`。

## 运行前门禁

在非零 update 前执行：

```text
python -m unittest tests.test_sb3_gru_integration -v
Ran 11 tests in 3.454s
OK
```

11/11 repair tests 通过后才创建真实 training rollout。CUDA 模型接口预检确认 optimizer 有 17 个唯一 tensor，actor-only schema 有 12 keys。Headless render 预检得到 `uint8`、`800 x 1000 x 3`，预检没有调用 simulator `step()`。

## 固定 Austin 场景

| 项目 | 值 |
|---|---|
| map | Austin |
| agents | 2 |
| timestep | 0.01 s |
| integrator | `Integrator.RK4` |
| ego index | 0 |
| ego raceline | raceline1 |
| opponent raceline | raceline1 |
| ego waypoint index | 0 |
| interval index | 15 |
| opponent waypoint index | 15 |
| opponent speed scale | 0.5 |
| seed | 20260715 |
| sim duration | 1.0 s |
| initial speed feature | 3.9039588 m/s |
| ego pose | `[130.006574, 51.339912, -0.391425]` |
| opponent pose | `[130.454777, 48.773169, 4.142401]` |

Opponent 使用每 episode fresh 的现有 Lattice Planner controller。其 action 只在 wrapper 内与 ego action 合并；buffer action 宽度为 2，仅包含 ego action。Lattice Planner 没有进入 optimizer、value、advantage 或 rollout buffer。

## PPO 配置

| 参数 | 值 |
|---|---:|
| algorithm | stock `sb3_contrib.RecurrentPPO` |
| buffer | stock `sb3_contrib.common.recurrent.buffers.RecurrentRolloutBuffer` |
| n_envs | 1 |
| n_steps | 100 |
| batch_size | 100 |
| n_epochs | 1 |
| learning_rate | 1e-6 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.10 |
| vf_coef | 0.5 |
| ent_coef | 1e-4 |
| max_grad_norm | 0.5 |
| target_kl | None |

一个 100-step rollout 形成一个 minibatch，因此 stock `train()` 只到达一次 optimizer step。Adam 所有已激活 state 的 step 均为 1，最大值没有超过 1。

## 临时 diagnostic smoke reward

仅在 `scripts/smoke_sb3_nonzero_update.py` 的 repo-local wrapper 中使用：

```text
r_smoke = raw_base_reward
        + 0.01 * tanh(next_ego_measured_speed / 10.0)
        - 0.001 * (executed_ego_steering / 0.52)^2
        - 1.0 * ego_collision
```

- reward 使用 wrapper 实际执行的 ego steering 和真实 post-step measured speed；
- opponent-only collision 不产生 collision penalty；
- 100/100 reward 均 finite；
- raw base reward 总和：`1.0000000000000007`；
- timeout bootstrap 前 smoke reward 总和：`1.2444540630099574`；
- SB3 timeout bootstrap 后 buffer reward 总和：`1.059393048286438`。

这只是为了确保一次 update 获得 action-sensitive signal，不是正式 reward 设计。

## 唯一 training rollout

| 验收项 | 结果 |
|---|---:|
| collect_rollouts 调用 | 1 |
| rollout start/end | 1 / 1 |
| simulator steps | 100 |
| valid transitions | 100 |
| buffer action shape | `[100, 1, 2]` |
| episode outcome | step 100 timeout |
| ego collision | false |
| opponent-only collision count | 0 |
| observations/actions/log_probs/values/rewards/advantages/returns finite | PASS |
| steering range | `[-0.519999683, 0.085740231]` |
| steering mean | `-0.304255068` |
| speed range | `[1.854863405, 7.773490906]` m/s |
| speed mean | `4.414156437` m/s |
| log probability range | `[-3.465355396, 13.921479225]` |
| value range | `[-1.037452936, 0.326875389]` |
| advantage mean/std | `0.213723093 / 0.320980340` |
| return mean/std | `-0.129317418 / 0.125798970` |

### Action trace

首个 transition 的完整物理动作路径：

| 位置 | steering | desired speed |
|---|---:|---:|
| raw End2Race mean | -0.235135406 | 4.000380993 |
| SB3 sampled action | -0.288338363 | 4.851915359 |
| SB3 clipped action | -0.288338363 | 4.851915359 |
| rollout buffer | -0.288338363 | 4.851915359 |
| Gymnasium wrapper ego action | -0.288338363 | 4.851915359 |
| real F1Tenth core ego action | -0.288338363 | 4.851915359 |
| real F1Tenth core opponent action | -0.149791524 | 2.895028114 |

结果：

- SB3 pre-env clipping count：`0`；
- buffer → core ego action max error：`0.0`；
- wrapper → core ego action max error：`0.0`；
- opponent action 不在 buffer 中。

## Update 前 recurrent replay identity

通过 stock `RecurrentRolloutBuffer.get(100)` 和 stock policy `evaluate_actions()` 重算：

| 指标 | 结果 | 阈值 |
|---|---:|---:|
| valid transitions | 100 | 100 |
| minibatches | 1 | 1 |
| padding timesteps | 38 | 已由 mask 排除 |
| max `|replayed_logp - old_logp|` | 0.0 | <= 1e-6 |
| mean error | 0.0 | <= 1e-7 |
| max `|ratio - 1|` | 0.0 | <= 1e-6 |

padding 来源于 recurrent buffer 对单个 cyclically split batch 的 sequence padding；误差统计包含全部 100 个有效 timestep。

## 唯一非零 PPO update

当前 optimizer 实例的 `step` 被临时包装计数并在结束后恢复；没有修改 optimizer class 或 SB3 全局对象。参数上的 post-accumulate gradient hooks 记录裁剪前梯度，optimizer step 入口记录裁剪后梯度。

| 指标 | 结果 |
|---|---:|
| optimizer.step count | 1 |
| Adam step min/max | 1 / 1 |
| gradient norm before clipping | 57.624153095 |
| gradient norm after clipping | 0.500000013 |
| gradients finite before/after | PASS / PASS |
| policy loss | -5.72204577e-08 |
| value loss | 0.148705930 |
| entropy loss | 2.461716175 |
| total loss | 0.074599080 |
| clip fraction | 0.0 |
| SB3 train-time pre-step approximate KL | 0.0 |

SB3 的 train-time KL 在 optimizer step 前由 old/new 相同参数计算，因此本次只有一个 minibatch时为 0。参数更新后对原 buffer 独立重算的 KL 见下一节。

## 参数更新

| 参数分类 | max absolute delta | 结果 |
|---|---:|---|
| actor `k` | 1.013278961e-06 | changed |
| actor `speed_mlp` | 9.983778000e-07 | changed |
| actor GRU | 1.013278961e-06 | changed |
| actor output layer | 1.000240445e-06 | changed |
| actor `dummy_embedding` | 0.0 | unchanged as required |
| independent critic | 9.983778000e-07 | changed |
| distribution `log_std` | 9.853860092e-07 | changed |
| full policy/global maximum | 1.013278961e-06 | `0 < delta < 1e-4` |

全部 delta finite。Optimizer 的 17 个 parameter identity 在 update 前后保持唯一且完全相同。Opponent trainable parameter count 和 optimizer overlap 均为 0，opponent parameter delta 为 0。

## Update 后数值验证

在不收集第二个 rollout 的情况下，对原来的 buffer 和固定 100-step observation sequence 验证：

| 指标 | 结果 |
|---|---:|
| post-update approximate KL | 0.000526221993 |
| KL threshold | `< 1e-3` PASS |
| ratio min/max | 0.933228970 / 1.267154932 |
| ratio max deviation | 0.267154932 |
| new log probability range | `[-3.481789589, 13.921426773]` |
| near-boundary steering log probabilities | `[-1135.0720, -2791.8591]`, finite |
| fixed sequence max action change | 0.000324249268 |
| fixed sequence final hidden max change | 0.000947102904 |
| action/hidden finite | PASS / PASS |
| deterministic steering range | `[-0.130524889, 0.254970312]` |
| dummy cell max absolute value | 0.0 |

Update 后 ratio 不再要求等于 1。虽然单个 timestep 的 ratio deviation 可达约 0.267，整体 approximate KL 仍为 `5.26e-4`，所有数值有限且通过附件规定的 KL 门槛。

## Actor-only checkpoint

输出：[actor_after_update.pth](../artifacts/sb3_nonzero_smoke/actor_after_update.pth)

| 验收项 | 结果 |
|---|---|
| key count | 12 |
| missing keys | 0 |
| unexpected keys | 0 |
| parameters finite | PASS |
| exported state vs policy actor max error | 0.0 |
| strict load into fresh `End2Race(mask_prob=0.0, hidden_scale=4)` | PASS |
| SHA256 | `7fcec2288494e82eeb49f113f9279774077f15d5cfeb0a42d90a693d7b214bcf` |

原 checkpoint 哈希在实验前后保持不变。

## Training rollout 视频

输出：[training_rollout.mp4](../artifacts/sb3_nonzero_smoke/training_rollout.mp4)

| 项目 | 值 |
|---|---:|
| source | 唯一 training rollout 的真实 Env.step |
| frames | 100 captured / 100 decoded |
| fps | 100 |
| resolution | 1000 x 800 |
| dtype | uint8 RGB |
| duration | 1.0 s |
| file size | 19,956 bytes |
| SHA256 | `7187df088b6c692cea3f0d40d34a6491a6a1aad45a7af5492a21b65937f678d6` |
| first/last frame readable | PASS / PASS |
| ego/opponent 可见 | **FAIL / FAIL** |
| interaction-video semantic gate | **FAIL** |

帧在 diagnostic wrapper 的 `step()` 内、真实 F1Tenth step 之后且 DummyVecEnv auto-reset 之前获取。因此第 100 帧是 timeout terminal frame，没有混入 reset 后的新 episode，也没有为了录像额外调用 `env.step()`。

但车辆不在默认相机 viewport 内。这个文件只能证明历史 capture/encoding 调用发生，不能作为双车交互的视觉证据。

## 修正后的 interaction visual replay

输出：[interaction_replay.mp4](../artifacts/sb3_nonzero_smoke/interaction_replay.mp4)

Provenance：

- 加载历史实验在 rollout 前保存的完整 policy 和 RNG snapshot；
- 固定相同 Austin scenario；
- 首个随机 action 与历史 training rollout 的误差为 `0.0`；
- `collect_rollouts()` 调用 0 次；
- `train()` / `learn()` / `optimizer.step()` 调用均为 0 次；
- policy 最大 parameter delta 为 `0.0`；
- 原始 `training_rollout.mp4` SHA256 前后相同，未被覆盖。

| 项目 | 值 |
|---|---:|
| verdict | `PASS_VISUAL_REPLAY` |
| real F1Tenth steps | 100 |
| outcome | step 100 timeout，无 ego collision |
| frames | 100 captured / 100 decoded |
| fps | 100 |
| resolution | 1000 x 800 |
| duration | 1.0 s |
| file size | 203,495 bytes |
| SHA256 | `755748bb5e66af801f689dee42214555e3e49c9da424f5c5312b48cc92434e42` |
| raw ego pixels/frame minimum | 124 |
| raw opponent pixels/frame minimum | 131 |
| decoded ego pixels/frame minimum | 116 |
| decoded opponent pixels/frame minimum | 101 |
| action → real core max error | 0.0 |

黄色矩形为 ego，红色矩形为 opponent。相机跟随 ego；两辆车在 100/100 个原始帧和解码帧中均可见。

## 产物与只读验收测试

Artifacts 均位于被 `.gitignore` 排除的 `artifacts/sb3_nonzero_smoke/`：

- `training_rollout.mp4`；
- `actor_after_update.pth`；
- `results.json`；
- `run.log`；
- `run_state.json`；
- `pre_update_snapshot.pth`，包含 rollout 前 policy、actor、critic、log_std、optimizer 和 RNG state。
- `interaction_replay.mp4`，相机修正后的零 update visual replay；
- `interaction_replay_results.json`，replay provenance 与逐帧车辆可见性证据。

非零实验完成后只读取 artifacts 运行：

```text
python -m unittest tests.test_sb3_nonzero_update -v
Ran 14 tests in 0.213s
OK
```

该 test module 不导入或执行 smoke runner，不会产生第二次 update。

视频修复后额外运行只读语义测试：

```text
python -m unittest tests.test_sb3_interaction_video tests.test_sb3_nonzero_update -v
Ran 21 tests in 0.296s
OK
```

## 验收结论

| 验收项 | 结果 |
|---|---|
| 原 11 项 repair tests | PASS |
| 真实 Austin 唯一 100-step rollout | PASS |
| 唯一 optimizer update | PASS |
| actor/critic/log_std 有限非零更新 | PASS |
| dummy embedding 不变 | PASS |
| 无 NaN/Inf | PASS |
| action contract | PASS |
| update 前 replay identity | PASS |
| update 后 KL finite 且 `< 1e-3` | PASS |
| 12-key actor-only checkpoint strict load | PASS |
| pretrained checkpoint 未改变 | PASS |
| historical training rollout MP4 容器/帧 | PASS |
| historical training rollout 双车可见性 | **FAIL** |
| future recorder 跟车相机与逐帧可见性门禁 | PASS |
| snapshot visual replay 双车可见性 | PASS |
| site-packages 未改变 | PASS |

历史非零实验因原 training MP4 的语义失败，严格总 verdict 为 **FAIL**；其 PPO replay、唯一 optimizer update、参数 delta、KL、checkpoint 等数值子项仍保持通过。当前 recorder 设置已经修正，并由零 update replay 独立验证为 **PASS_VISUAL_REPLAY**。如需重新取得满足全部原始条件的 `PASS_FOR_TRAINING_PIPELINE_IMPLEMENTATION`，必须以后明确授权一次新的非零 smoke；本次没有执行第二次 optimizer update，也没有开始正式 learner。
