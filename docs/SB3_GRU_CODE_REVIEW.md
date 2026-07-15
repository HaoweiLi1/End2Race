# SB3-Contrib GRU RecurrentPPO POC 源码审查

日期：2026-07-15（Asia/Singapore）

行号均指本次审查时的当前 worktree。除阅读源码外，审查只运行了合成短 rollout 和不调用 `optimizer.step()` 的 backward 诊断；没有运行真实 End2Race learner 训练。

## Verdict: FAIL

这个 verdict 是对“当前 POC 可直接连接真实 F1Tenth 并进入非零学习率训练”的否定，**不是对 SB3 路线或 GRU dummy-cell 方案的否定**。

已经确认正确的核心结论是：

- actor mean 确实直接来自原始 `End2Race`；
- stock `RecurrentRolloutBuffer` 保存 pre-action state，POC 会用每段真实初始 hidden 重放；
- dummy cell 没有进入 GRU 计算；
- actor/critic 梯度隔离正确，actor-only checkpoint schema 与 BC 兼容。

但当前 wrapper 对真实 `F110Env.reset(poses)` 无法完成首次及自动 reset，actor 的 speed 观测时序与原 evaluator/BC 数据都不同，并且当前高斯方差导致大量 buffer raw action 与环境实际执行 action 不同。因此现有 0-error 测试不足以支持 POC 报告中的总体 `PASS`。

### 审查范围

已审查：

- `rl/sb3_end2race_policy.py`
- `rl/end2race_gymnasium_env.py`
- `scripts/smoke_sb3_gru.py`
- `tests/test_sb3_gru_integration.py`
- `docs/SB3_GRU_POC_REPORT.md`
- 相关原始文件：`model.py`、`eval_multiagent.py`、`train.py`、`demonstration.py`、`f1tenth_gym/gym/f110_gym/envs/f110_env.py`、`base_classes.py`、`dynamic_models.py`
- 实际安装的 SB3/SB3-Contrib 2.7.1 源码。

按文件时间和引用关系自动扫描后，本次 POC 新增的 Python 文件只有上述两个 `rl/` 文件、smoke 脚本和 integration test。`tests/test_actor_identity.py`、`tests/test_replay_identity.py`、`tests/test_hidden_reset.py` 时间早于 POC 且均为 0 行；`scripts/rl_library_audit/*.py` 属于之前的库审计，不是此 POC 实现。

## Critical issues

### C1. wrapper 不能 reset 真实 F1Tenth env

**证据**

- `rl/end2race_gymnasium_env.py:98-107`, `End2RaceGymnasiumEnv._base_reset()`：只有 `options` 含 `poses` 时才向 base env 传 poses；普通 reset 最终调用 `self.f110_env.reset()`。
- `f1tenth_gym/gym/f110_gym/envs/f110_env.py:334-388`, `F110Env.reset(self, poses)`：`poses` 是必需位置参数，没有 default。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/stable_baselines3/common/vec_env/dummy_vec_env.py:68-72`, `DummyVecEnv.step_wait()`：episode 结束后自动调用 `env.reset()`，不传 poses/options。
- `scripts/smoke_sb3_gru.py:74-77`, `SyntheticLegacyF110Env.reset()` 接受任意 `**kwargs` 且不需 poses，所以合成测试屏蔽了真实 API 错误。
- `docs/SB3_GRU_POC_REPORT.md:113-115` 也明确说真实 F1Tenth reset/step 不在已验证范围，但报告在 `:7` 仍给出了总体 `PASS`。

**影响**

对原始 `F110Env` 直接使用此 wrapper 时，首次 `DummyVecEnv.reset()` 就会缺少 poses；即使用 VecEnv options 完成首次 reset，第一个异步 episode 结束后的自动 reset 仍会失败。当前 POC 并未证明能跑真实 End2Race env。

**建议修复**

在 repo 内的 wrapper 显式接受并管理 reset-pose provider（或可重复生成 poses 的 callback），保证首次 reset 和每个并行 env 的自动 reset 都向 legacy `F110Env.reset(poses)` 传入正确 shape 的 poses。增加一个 reset 必须传 poses 的 strict fake，并触发至少两个 env 错开的多次自动 reset。

### C2. PPO 优化的 raw action 与环境执行 action 大量不一致

**证据**

- `model.py:40-44,96-100`, `End2Race.forward()`：最后是没有边界 activation 的 `Linear(..., 2)`，输出是无界 raw mean。
- `train.py:41-43,83-95`：两维 BC target 是 physical `steer` 和 `desired_speed`，不是 normalized action。
- `rl/sb3_end2race_policy.py:197-211`, `End2RaceGRUPolicy.actor_mean()` / `_distribution()`：mean 直接传给 `DiagGaussianDistribution`，没有 action MLP、scale 或 squash。
- `rl/sb3_end2race_policy.py:232-238`, `forward()`：对无界 Normal 采样并立即对该 raw sample 计算 log probability。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/stable_baselines3/common/distributions.py:153-176`, `DiagGaussianDistribution.proba_distribution()` / `log_prob()`：分布是 `Normal(mean, exp(log_std))`，log probability 是 raw action 上的 Normal density。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/ppo_recurrent/ppo_recurrent.py:238-252,287-295`, `RecurrentPPO.collect_rollouts()`：传给 env 的是 `np.clip(actions, low, high)`，但 buffer 保存的是原始 `actions`，old log probability 也对原始 `actions` 计算。
- `rl/end2race_gymnasium_env.py:32-33,47-51`：Box 是 steering `[-0.52, 0.52]` rad、speed `[0, 20]` m/s。
- `rl/end2race_gymnasium_env.py:128-145,159-162`, `_joint_action()` / `step()`：wrapper 不再 clip，直接传递 SB3 已 clip 的物理 action。
- `f1tenth_gym/gym/f110_gym/envs/base_classes.py:546-564`, `Simulator.step()`：第一维是 desired steering angle，第二维是 desired velocity。`base_classes.py:254-302`, `RaceCar.update_pose()` 和 `dynamic_models.py:179-219`, `pid()` 再将目标值转成受限的 steering velocity/acceleration。
- 本次同配置合成 rollout 的只读诊断：20 个 transition 中 12 个在执行前被 clip；steering 10/20，speed 2/20。raw 范围为 steering `[-1.3145, 2.2706]`、speed `[-1.1398, 6.1850]`，最大 clip delta 分别为 `1.7506` 和 `1.1398`。原因是 `scripts/smoke_sb3_gru.py:113-118` 设 `log_std_init=0.0`，即两维的初始标准差都是 1 个物理单位。

**结论**

- 没有 steering 重复缩放，没有 speed 重复缩放，也没有 normalized/physical 数值的隐式转换；全链路都在物理单位。
- buffer action 和 old/new log probability 使用的是同一 raw action，所以 replay identity 会精确通过。
- 但 environment 经常执行另一个 clipped action。因此 replay identity 证明了“PPO 重放自己记录的 raw sample”，没有证明“PPO likelihood 对应实际环境 action”。
- 原 evaluator 在 `eval_multiagent.py:169-174` 对 steering 做一次 `[-0.52,0.52]` clip，但对 speed 不 clip；POC 多出 speed `[0,20]` clip。

**建议修复**

在非零学习率前由 owner 固定单一 action contract：明确 BC mean、采样分布、buffer action、log probability 和环境执行值各自是 physical 还是 normalized。如果要消除执行不一致，应在 repo-local policy 中使用支持边界及正确 Jacobian/log probability 的分布，或给出另一个经数学定义且测试的方案。仅调小 `log_std` 可以减少但不能从语义上消除这个问题。新测试必须同时记录 raw sample、buffer action、old logp、SB3 clipped action 和 core env 收到的 action，并报告每维 clipping rate。

### C3. POC 的 speed observation 时序不等于原 evaluator，也不等于 BC 训练数据

**证据**

- `rl/end2race_gymnasium_env.py:95-96`, `_actor_observation()` 将 `_previous_ego_speed` 与当前 LiDAR 组合。
- `rl/end2race_gymnasium_env.py:116-126`, `reset()` 在返回首个 observation 前把 `_previous_ego_speed` 设为 reset observation 的 `linear_vels_x`。
- `rl/end2race_gymnasium_env.py:159-167`, `step()` 在返回 `obs_(t+1)` 前把 `_previous_ego_speed` 更新为同一 `obs_(t+1)` 中的 speed。因此 policy 实际收到的是 **当前 observation 的 measured speed**，不是 previous speed。
- `eval_multiagent.py:117-120` 将首次 speed 设为 raceline `initial_speed * 0.9`。
- `eval_multiagent.py:164-174,200-203` 在当前 action 推理后、`env.step()` 前才将 `prev_speed = obs['linear_vels_x'][0]`。所以从第二次推理起，原 evaluator 对 `LiDAR_t` 使用的是上一次 decision observation 的 measured speed，比 POC 的 speed 落后一个 env step。
- `train.py:81-95`, `SequenceDataset._create_sequences()` 更明确：BC 训练对 `lidar[t]` 使用的是 `action_data[t-1].desired_speed`，即 **上一个 desired-speed command**，并非 measured vehicle speed。
- `scripts/smoke_sb3_gru.py:140-160`, `bc_sequence_identity()` 对两条路径传入同一个任意 random speed，因此不可能发现 observation 时序错误。`run_poc():455-463` 只测 shape，不测数值对齐。

**影响**

actor identity `0.0` 只证明“给定相同 361D tensor 时两个 actor 一致”，没有证明 wrapper 生成了 BC/evaluator 所需的 361D tensor。这是 distribution shift，也会改变 deterministic BC mean。

**建议修复**

首先固定唯一定义。若以 BC schema 为权威，wrapper 应保存上一个实际执行的 desired-speed command，并明确 reset 时的初值规则；若要以原 evaluator 为权威，则应精确复制其首值和一步 lag。不能在不重训 BC 的情况下默认改成 current measured speed。测试应使用可区分 command/current/previous measured speed 的数列，直接比较 wrapper observation 与独立 oracle。

## Major issues

### M1. base terminal 与 wrapper timeout 同步发生时会被误分类为 truncation

**证据**

- `rl/end2race_gymnasium_env.py:169-179`, `step()` 的优先级是 collision → wrapper timeout → base flags。如果 non-collision `base_terminated=True` 与 `timeout=True` 同步出现，base terminal 被丢弃并返回 `terminated=False, truncated=True`。
- `f1tenth_gym/gym/f110_gym/envs/f110_env.py:232-274`, `_check_done()`：真实 legacy env 的 done 来源是 ego collision 或所有 agent 完成 finish toggles。lap/finish 是 non-collision base terminal，可能在 sim-duration 边界与 timeout 同步。
- `scripts/smoke_sb3_gru.py:46-84` 的 synthetic core 只会产生 collision done，不能产生 non-collision base terminal，所以此分支未测。

**建议修复**

将真正 MDP terminal（collision 或 base terminated）置于 time-limit truncation 之前，并定义 base truncated 与 wrapper timeout 的合并规则。增加 base terminal 与 timeout 同一 step 的测试，确保不会错误 bootstrap terminal state。

### M2. replay test 没有证明它声称的全部 coverage

**证据**

- `scripts/smoke_sb3_gru.py:330-340`, `replay_identity()` 确实从 stock buffer sample 取 `old_log_prob`，并通过一次新的 `policy.evaluate_actions()` 计算 `new_logp`；没有直接复用同一 logp tensor。
- 但 `replay_identity():359-365` 的 `ordinary_continuous_sequence=True` 和 `timeout_truncation=True` 是硬编码布尔值，不是从 buffer/events 推导。
- `tests/test_sb3_gru_integration.py:45-55` 不 assert `valid_timesteps == n_steps * n_envs`，不 assert 每个原 transition 只统计一次，不 assert 存在 `episode_start=False` 且 initial hidden 非零的 minibatch sequence。
- 本次独立诊断确认当前 seed 实际产生了 20 valid + 3 padding、8 段 sequence，其中 3 段是 `episode_start=False` 且 initial hidden 非零的 continuation sequence；但这些关键性质并没有被 unittest 锁定。
- `scripts/smoke_sb3_gru.py:41-84` 的 synthetic env 完全忽略 action 对 observation/reward 的影响，所以 C2 中的 raw/executed action 错配不会导致任何测试失败。

**建议修复**

将 coverage 从 buffer 索引、mask、episode starts 和 wrapper terminal event 实际推导；assert valid 数精确为 20，assert 存在 padding，assert 存在非零 initial hidden 的 continuation sequence，并用独立逐步 oracle 检查该 initial hidden 等于对应 rollout pre-action hidden。同时增加 action-sensitive env 检查执行 action。

### M3. zero-learning-rate smoke 不是 optimizer/梯度正确性测试

**证据**

- `scripts/smoke_sb3_gru.py:99-120`, `build_model()` 将 `learning_rate=0.0`。
- `run_poc():476-478` 调用 stock `model.train()` 后只检查 parameter delta。即使 actor 没有梯度、optimizer 含错误参数或某些参数重复，`max_parameter_delta` 仍可以是 0。
- `tests/test_sb3_gru_integration.py:82-91` 只 assert lr 和 delta 都是 0。

**审查诊断**

为区分测试缺口和实现错误，本次对一个 stock padded minibatch 分别对 policy loss/value loss backward，但未调用 optimizer step。结果是 policy loss 对 12 个 actor tensor 中的 11 个产生非零有限梯度（`dummy_embedding` 因 `mask_prob=0` 预期无梯度），value loss 只对 4 个 critic tensor 产生梯度。因此当前梯度隔离实现正确，但 unittest 未证明它。

**建议修复**

保留 lr=0 smoke，另增加不 step optimizer 的 loss-specific gradient assertions：policy loss 必须到达 active actor + `log_std` 且不到 critic，value loss 必须只到 critic，unused/dummy 参数必须符合明确预期。

### M4. 当前 POC 报告的 `PASS` 表述过宽

`docs/SB3_GRU_POC_REPORT.md:7-9` 声称总体 `PASS`，但 `:113-115` 明确排除真实 F1Tenth，`:211-215` 又把真实 wrapper 和 action contract 留作未解决事项。加上 C1-C3，更准确的说法应是：“GRU state transport/replay 子证明 PASS；完整 End2Race POC FAIL，等待 wrapper/action/observation fixes。”

**建议修复**

在实现修复并重新审查后再更新 POC 报告；不应通过放宽 replay 阈值或仅保留 synthetic 0-error 来恢复 `PASS`。

## Minor issues

### m1. optimizer 包含永远不会使用的 inherited `action_net`

- `rl/sb3_end2race_policy.py:90-105` 先让 parent 创建 placeholder policy modules；`:141-147` 又用 `self.parameters()` 重建 optimizer。
- 自定义 `forward()` / `get_distribution()` / `evaluate_actions()` 分别在 `:225-272` 绕过 parent `action_net`。
- 诊断显示 inherited `action_net` 有 2 个 tensor/4 个 scalar，在 optimizer 中但 policy/value loss 梯度都为零。它不会改变 mean，但是无用 optimizer state 和令人误解的参数。

**建议修复**：重建 optimizer 时显式使用 actor + critic + `log_std` 的唯一参数集，并 assert 无重复/无未分类参数。

### m2. 两个“安全性”测试结果是硬编码声明

- `scripts/smoke_sb3_gru.py:455-463` 把 `privileged_simulator_fields_in_actor_observation=False` 直接写入结果，没有由比较输入数值推导。
- `run_poc():506-507` 直接设 `third_party_sources_modified=False` 和 `end2race_learner_training_performed=False`；`tests/test_sb3_gru_integration.py:90-91` 只 assert 这两个常量。

**建议修复**：privileged-field test 应用改变这些 raw fields 但保持 LiDAR/speed 不变的 metamorphic test。第三方完整性应在审计脚本/报告中用 distribution RECORD/hash 或干净环境比较，不应伪装成 unit assertion。

### m3. LiDAR/observation 没有 finite/range contract

`rl/end2race_gymnasium_env.py:42-46` 宣告无穷 Box，`:85-90` 不处理 NaN/Inf 或 sensor range；`eval_multiagent.py:152-166` 也没有这些处理。这不是 POC 相对 evaluator 的新偏差，但对非零学习率是未定义输入合同，NaN/Inf 会直接进入 `model.py:80-98` 的 exponential/GRU。

**建议修复**：根据 BC 数据生成路径定义且测试有限性、最大量程和 invalid-beam 处理，不能只为 PPO 临时增加与 BC/evaluator 不一致的 clip。

### m4. downsampling 与 BC 数据生成路径存在共享的历史偏差

POC `rl/end2race_gymnasium_env.py:85-90` 与 evaluator `eval_multiagent.py:152-155` 对大于 360 的 scan 都用 integer `np.linspace(0, n-1, 360)`，因此两者相互一致。但 BC 数据生成使用 `demonstration.py:282-286` 调用 `latticeplanner/utils.py:523-528`, `downsample_lidar()`，对 1440 beam 是固定 `scan[::4][:360]`。`linspace(..., dtype=int)` 的索引并不完全等于 `0,4,...,1436`。

**建议修复**：在正式实现前确认 checkpoint 实际训练数据的 beam count/index，将索引向量作为显式 contract 并用 monotonic beam-id scan 测试精确顺序。

## Confirmed-correct properties

### 1. End2Race actor path

- `rl/sb3_end2race_policy.py:107-113` 创建原始 `End2Race(mask_prob=0.0, hidden_scale=...)` 并 strict load BC state。
- `actor_mean():189-205` 将 360D LiDAR 和 1D speed 直接传给 `self.end2race_actor(...)`。
- 该调用使用 `model.py:80-98` 的 learnable `k` 处理、`speed_mlp`、GRU 和 `output_layer`。
- `actor_mean():207-211` 将原始两维输出直接作为 Gaussian mean。parent 的 action MLP/head 没有在 `forward()`、`get_distribution()` 或 `evaluate_actions()` 中被调用。
- `log_std` 只在 `:135-139,210-211` 设定采样方差，不改变 deterministic Normal mode。

因此“deterministic raw mean 与原 `End2Race` 给定同一 input/state 时一致”是真的。需与 C2 区分：`policy.predict()`/rollout 向 env 发送前仍会对越界 deterministic/stochastic action 做 Box clip，参见 `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/common/recurrent/policies.py:404-425`, `RecurrentActorCriticPolicy.predict()`。

### 2. BC checkpoint 加载、训练参数和导出

- strict load：`rl/sb3_end2race_policy.py:107-113`。
- actor-only export helper：`:274-277`；smoke 实际执行 `torch.save(policy.end2race_actor.state_dict(), ...)` 于 `scripts/smoke_sb3_gru.py:419-426`。
- 导出 key 精确为 12 个：`k`、`dummy_embedding`、speed MLP 2 个、GRU 4 个、output layer 4 个。strict round-trip test `:419-437` 是有效的。
- optimizer 包含这 12 个 actor tensors；active deterministic actor 的 11 个 tensor 可从 policy loss 获得梯度。`dummy_embedding` 虽在 12-key schema/optimizer 中，但 `mask_prob=0.0` 时 `model.py:87-91` 的分支不会执行，所以它不会被 PPO 更新。
- GRU 同时通过 `end2race_actor.gru` 和 `lstm_actor.gru` 注册为模块别名，但 PyTorch `parameters()` 去重；诊断显示 optimizer 19/19 个 parameter tensor 全部唯一，无重复 actor GRU 参数。

### 3. GRU dummy-cell adapter

- `rl/sb3_end2race_policy.py:38-47` 从真实 GRU 暴露 `input_size`、`hidden_size`、`num_layers`。
- `GRUWithLSTMStateInterface.forward():49-62` 验证 `(h,c)` shape，只将 `h` 传给 GRU，返回 `zeros_like(next_hidden)` 作为 next cell。`c` 不是 `Parameter`。
- 实际 PPO path 不通过 adapter `forward()`；`policy.lstm_actor` 主要用于 stock `RecurrentPPO._setup_model()` 读取 state shape，见 `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/ppo_recurrent/ppo_recurrent.py:139-185`。真实 actor 计算在 `actor_mean()` 中直接调用原 `End2Race`。
- `actor_mean():172-188` 按 `episode_starts` 仅清零对应 sequence/env slot 的 h，`:208` 返回同 shape/device/dtype 的 zero c。
- obs/state 都在 SB3 policy 所在 device；`zeros_like`、乘法和 `torch.cat` 保持 dtype/device/batch dimension。`RecurrentPPO._setup_model():152,161-174` 负责将 policy 移到 device 并创建 `[layers,n_envs,hidden]` 的 h/c。
- 手工 staggered reset test 能够发现“一个 env reset 误清另一个 env”，这一 policy-level 性质正确；但它不弥补 C1 的真实 env reset API 问题。

### 4. Stock recurrent replay 路径

- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/ppo_recurrent/ppo_recurrent.py:231-243` 用 `_last_lstm_states` 调用 actor，得到 next state。
- 同函数 `:287-299` 向 buffer 传入的是调用前的 `self._last_lstm_states`，然后才用 next state 更新 `_last_lstm_states`，所以保存的是 pre-action state。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/common/recurrent/buffers.py:136-145`, `RecurrentRolloutBuffer.add()` 将 h/c 复制到 buffer。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/common/recurrent/buffers.py:147-197` 保持每个 env 内的时序顺序；`:64-95,199-242` 根据 episode start 和 env change 切段，用每段在 buffer 中的真实 initial h/c，然后 padding 并生成 mask。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/ppo_recurrent/ppo_recurrent.py:336-364` 把该 initial state 传入 `evaluate_actions()`，并用 mask 排除 padding；`:371-395` 对 value/entropy loss 也使用 mask。
- POC `replay_identity():330-342` 使用 stock sample 的 actions/initial state/episode starts 新算 logp，与 old tensor 是两次独立计算。
- 独立诊断确认当前 rollout 有 15 个 `episode_start=False` 且 stored pre-action h 非零的 transition；minibatch 中有 3 个 continuation sequence 使用非零 initial h。因此 `0.0` replay error 不是因为所有段都被统一清零。

### 5. Timeout bootstrap 的 stock SB3 部分

- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/stable_baselines3/common/vec_env/dummy_vec_env.py:56-73` 在 auto-reset 前把 wrapper 返回的 final observation 保存为 `terminal_observation`，并仅对 `truncated and not terminated` 设 `TimeLimit.truncated=True`。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/ppo_recurrent/ppo_recurrent.py:268-285` 仅对该 truncation 用 `terminal_observation` 计算 `gamma * V(terminal_obs)`；不使用 reset 后 observation。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/stable_baselines3/common/buffers.py:425-435`, `RolloutBuffer.compute_returns_and_advantage()` 用下一 transition 的 `episode_starts` 将 GAE 在 episode boundary 处截断。
- 当 wrapper 正确返回 collision terminated/timeout truncated 时，collision 不 bootstrap，timeout 用 final observation bootstrap，advantage 不跨 episode。当前 synthetic D test 对这一 stock 路径是有意义的。C1/M1 是 wrapper 进入该路径前的另外问题。

### 6. 依赖和第三方源码

- 当前环境实测：Python `3.10.19`，PyTorch `2.7.0+cu128`，Gymnasium `1.2.3`，stable-baselines3 `2.7.1`，sb3-contrib `2.7.1`。
- `install.sh:8` 已精确固定 `stable-baselines3==2.7.1 sb3-contrib==2.7.1`。
- repo 内没有 monkey patch SB3 全局类，没有复制/私改 `RecurrentPPO` 或 `RecurrentRolloutBuffer`；POC 只子类化 policy 并导入 stock `RecurrentPPO`。
- 按安装 distribution `RECORD` 的 hash 做了只读校验：stable-baselines3 检查 73 个 hashed file，sb3-contrib 检查 51 个 hashed file，均为 0 mismatch。这比 `run_poc():506` 的硬编码 `False` 更能支持“未修改 site-packages”。

## Action-space trace

| 阶段 | 数值/范围 | 语义 | 源码证据 |
|---|---|---|---|
| `End2Race` output/mean | 无界 `float32[2]` | desired steering rad, desired speed m/s | `model.py:40-44,96-100`; `train.py:41-43` |
| POC actor mean | 不变 | 同上，无 actor MLP/head | `rl/sb3_end2race_policy.py:189-211` |
| Gaussian sample | 无界 Normal；POC 初始 std=`[1 rad, 1 m/s]` | physical raw action | policy `:210-235`; smoke `:113-118`; SB3 distributions `:153-176` |
| rollout old logp | 对 raw sample 计算 | raw action density | policy `:233-235`; stock collect `:238-243` |
| rollout buffer action | raw sample，不是 clipped copy | 与 old/new logp 一致 | stock collect `:244-252,287-295` |
| scale/unscale | 无 | 没有 normalized action | custom policy `squash_output=False`; stock collect 只 clip |
| SB3 env action | per-dim clip 到 `[-0.52,0.52]`, `[0,20]` | physical command | wrapper `:32-33,47-51`; stock collect `:246-252` |
| wrapper | 不再 clip/scale | 同上 | wrapper `:128-145,159-162` |
| F1 simulator | desired targets 进入 steering delay + PID + dynamics constraints | steering state 又受 default `[-0.4189,0.4189]`，velocity state受 `[-5,20]` 约束 | `base_classes.py:254-302,546-564`; `f110_env.py:139-158`; `dynamic_models.py:30-87,179-219` |
| original evaluator | steering 一次 clip `[-0.52,0.52]`；speed 不 clip | physical command | `eval_multiagent.py:169-174,200-203` |

最终判定：没有重复 scaling，但有高频的 **raw likelihood action → clipped executed action** 不同。wrapper 自身没有不透明 clip；F1 的延迟、PID 和 dynamics saturation 是另一层明确的 plant 动力学，不应与 SB3 前置 clip 混为一谈。

## Gradient/optimizer parameter table

诊断对象是 `scripts/smoke_sb3_gru.py:99-120` 的 policy；只 backward，不做 optimizer step。optimizer 由 `rl/sb3_end2race_policy.py:141-147` 创建，只有一个 Adam parameter group。

| optimizer group / 逻辑归属 | tensor 数 | scalar parameter 数 | policy loss | value loss | 是否导出 actor-only |
|---|---:|---:|---|---|---|
| group 0 总计 | 19 | 11,313,105 | — | — | 部分 |
| `end2race_actor` | 12 | 11,301,482 | 11/12 tensor 非零梯度 | 0/12 | 是，全部 12 key |
| independent `value_net` | 4 | 11,617 | 0/4 | 4/4 非零梯度 | 否 |
| `log_std` | 1 | 2 | 1/1 非零梯度 | 0/1 | 否 |
| inherited unused `action_net` | 2 | 4 | 0/2 | 0/2 | 否 |

说明：

- actor 的 12 个 tensor 全部在 optimizer；唯一没有 policy gradient 的是 `dummy_embedding`，因为 policy 固定 `mask_prob=0.0`。
- critic 是 `rl/sb3_end2race_policy.py:125-133` 的 feed-forward `361 -> 32 -> 1` network（smoke 设 `critic_hidden_size=32`）。`_critic_values():213-217` 不读 actor h/GRU，因此 value loss 不更新 End2Race actor。
- policy loss 更新 End2Race active actor 和 `log_std`；value loss 只更新 critic；critic 不能改变 deterministic actor mean。
- dummy h/c 都是 runtime tensors，不是 Parameter。optimizer 中没有 dummy state，也没有重复 parameter identity。

## Observation differences from original evaluator

| 项目 | 原 `eval_multiagent.py` | POC wrapper | 判定 |
|---|---|---|---|
| LiDAR agent | 固定 `scans[0]` (`:152`) | `scans[ego_index]` (`wrapper:85-86`) | `ego_index=0` 时相同；POC 可配置 |
| beam 顺序 | 保持 raw scan 顺序 | 保持 raw scan 顺序 | 相同 |
| `n>360` downsample | integer `linspace(0,n-1,360)` (`:153-155`) | 同样的 integer linspace (`wrapper:87-89`) | 相同 |
| `n<360` | 不 downsample，后续 model width 会不匹配 | 用 linspace 重复/上采样到 360 | 不同；POC 会静默制造 beam |
| dtype | selection 后构造 float32 tensor (`:164-166`) | 先 cast float32 再 selection (`wrapper:85-90`) | 实际有限 scan 上等价，但处理顺序不同 |
| NaN/Inf | 不处理 | 不处理 | 相同，均缺 contract |
| LiDAR range | 不 clip | 不 clip，Box 是 `[-inf,inf]` | 相同，均缺 contract |
| noise | 可选按比例将 beam 置 0 (`:157-162`)，default 0 | 无 noise path | default 时相同；非零 evaluator noise 时不同 |
| first speed | raceline initial desired speed `*0.9` (`:112,120`) | reset observation 的 measured speed (`wrapper:119`) | 不同 |
| later speed | 上一 decision observation 的 measured speed（`:166-174,202`） | 当前 LiDAR 同一 observation 的 measured speed (`wrapper:165-167`) | 不同，POC 少一步 lag |
| BC train speed（补充） | evaluator 本身已不完全一致 | previous desired-speed command (`train.py:83-95`) | POC 与 BC 也不同 |
| hidden reset | episode 开始显式 zero (`eval:117-120`) | `episode_starts` 按 env slot zero (`policy:185-208`) | policy 语义一致，POC 支持异步 env |
| privileged state | actor 只收 LiDAR + speed | wrapper output 只有 361D LiDAR + speed | 相同；pose/opponent/reference 不进 actor |

额外注意：两者的 linspace downsampling 都与 BC data-generation 的 `scan[::4]` 存在历史偏差，见 minor m4。

## Terminated/truncated assessment

| 情况 | 当前路径 | 判定 |
|---|---|---|
| ego collision | wrapper `:169-175` 返回 terminated=True/truncated=False | 正确 |
| wrapper sim-duration | wrapper `:171-175` 返回 terminated=False/truncated=True | 单独发生时正确 |
| legacy lap/finish done | `_step_result():80-82` 映射为 base terminated | 作为 MDP terminal 合理 |
| base terminal + timeout 同 step | timeout 覆盖 base terminal | **错误，M1** |
| final observation | DummyVecEnv 在 reset 前存 `terminal_observation` | 正确，前提是 reset API 已修复 |
| timeout bootstrap | stock collect 仅对 TimeLimit truncation 用 terminal obs value | 正确 |
| collision bootstrap | 无 timeout reward correction，GAE 在 done 截断 | 正确，bootstrap 0 |
| advantage 跨 episode | `episode_starts[t+1]` 使 `next_non_terminal=0` | 不跨 episode，正确 |
| auto reset poses | DummyVecEnv 不传 poses，wrapper 也没有 provider | **错误，C1** |

真实 F1Tenth legacy `done` 的其他来源已通过 `f110_env.py:232-274` 查明：除 ego collision 外只有所有 agent finish toggle。它不是 time-limit，应作为 terminated。

## Test-independence assessment

| unittest | 实际证明的性质 | 独立性/盲区 | 评价 |
|---|---|---|---|
| A. BC sequence identity (`tests:15-33`) | public custom policy deterministic raw output/h 与另一个 `End2Race` instance 在 100 步上一致；adapter 忽略 c | reference 和 policy 共用同一 `End2Race` class/checkpoint，因此能发现 wiring/extra head，不能发现 model 共享逻辑错误；不经 wrapper 或 action clip | 部分有效 |
| B. episode reset (`tests:35-43`) | `actor_mean()` 对手工 starts 的 slot-wise h reset，其他 slot 连续，c=0 | starts 是手工数组，不经 stock collector/VecEnv 自动 reset；不能发现 C1 | policy-level 有效，integration 不足 |
| C. replay identity (`tests:45-55`) | stock collect → stock buffer/get → custom `evaluate_actions()`；old/new tensor 独立，当前 seed 确有 padding/boundaries/nonzero initial h | coverage 的两个布尔量硬编码；未 assert valid=20/非零 initial h；env 忽略 action；只证 raw action likelihood，不证 executed action | recurrent 核心强，完整 PPO/env 结论过宽 |
| D. timeout/collision (`tests:57-72`) | synthetic wrapper + DummyVecEnv + stock timeout reward correction，使用了 terminal obs 而非 reset obs | expected terminal value 也由同一 policy critic 计算；critic 是 feed-forward，不压测 recurrent value state；synthetic reset/base done 过度简化 | stock timeout 子路径有效，真实 wrapper 不足 |
| E. checkpoint (`tests:74-80`) | 真实 actor state_dict 临时导出、新 `End2Race` strict load、12 key 顺序一致 | 不证明 PPO loss 能更新 actor，但这不是该测试的职责 | 有效 |
| zero-lr smoke (`tests:82-91`) | stock `train()` 能返回，lr=0 时 state_dict 不变 | lr=0 必然屏蔽 gradient/optimizer 问题；第三方/未训练是硬编码布尔值 | 仅是 API smoke，不是学习正确性证明 |

“测试与实现共享错误逻辑”的主要实例是：A 的两边使用同一 random speed；C 的 old/new logp 都对 raw action，合成 env 又忽略 executed action；D 的 expected value 来自同一 critic；smoke 用 lr=0。这些不会推翻已确认的 recurrent replay 子结论，但会推翻“六个测试证明完整实现正确”的结论。

## Required fixes before nonzero-learning-rate training

下列项目全部是 hard gate，不应通过放宽 `1e-6` replay/identity 阈值绕过：

1. 修复 C1：为真实 legacy `F110Env.reset(poses)` 实现可重复、支持每个并行 env 异步 auto-reset 的 poses provider，并用 strict-reset fake 和一次真实 env contract test 验证。
2. 修复 C3：由 owner 确定 speed feature 是 previous desired command、previous measured speed 还是 current measured speed；根据 BC checkpoint 合同实现唯一定义和 reset 初值，添加独立数列 oracle test。
3. 解决 C2：冻结 action 单位、bounds、分布和 transform/log-Jacobian 合同；证明 buffer/logp action 与环境实际执行 action 的关系，并将 clipping rate 设为明确门禁。
4. 修复 M1：terminal 优先于 time-limit，对 base terminated/base truncated/collision/timeout 的所有组合建立表驱动测试。
5. 加强 C test：从实际 buffer/events 推导 coverage，assert 20 个 valid transition 全部且仅一次进入统计，并独立核对至少一段 continuation 的非零 pre-action initial h。
6. 增加不 optimizer-step 的 gradient test，锁定 actor/critic/`log_std`/unused parameter 的梯度归属；不能用 lr=0 parameter delta 替代。
7. 将 optimizer 中的 inherited unused `action_net` 排除，并 assert parameter identities 唯一、全部已分类。
8. 固定 360D beam index/NaN/Inf/range contract，至少解释并测试 evaluator linspace 与 BC data `scan[::4]` 的选择。
9. 上述全部通过后，再将 `docs/SB3_GRU_POC_REPORT.md` 从子路径 PASS 更新为经证据支持的整体 verdict。

在这些 hard gate 完成前，**建议继续采用 SB3-Contrib 作为 recurrent PPO 技术路线，但不建议运行任何非零学习率的 End2Race learner 训练**。
