# Recurrent PPO correctness audit

审计日期：2026-07-15（Asia/Singapore）

## 结论摘要

| 实现 | `max abs(replay_logp - rollout_logp)` | 判定 | 结论 |
|---|---:|---|---|
| Tianshou 0.5.1 stock PPO + persistent GRU | `9.187382698059082` | **FAIL** | stock update 没有取得 rollout action 对应的 pre-action hidden state，不能保证 recurrent PPO ratio 正确 |
| SB3-Contrib 2.7.1 stock LSTM RecurrentPPO | `2.384185791015625e-07` | **PASS** | buffer 保存 pre-action state，minibatch 以每段初始 state 重放，episode/env boundary、padding 与 mask 均正确接通 |
| SB3-Contrib + audit-only GRU/dummy-cell policy | `0.0` | **PASS** | 不改 `RecurrentPPO` 和 `RecurrentRolloutBuffer` 可行，但必须提供 custom policy adapter |

验收阈值：PASS `<= 1e-6`；WARN `(1e-6, 1e-4]`；FAIL `> 1e-4`。

最终建议：**使用 SB3-Contrib RecurrentPPO**。它的 stock recurrent rollout/update 数据通路满足本审计要求；End2Race 的 GRU 可以通过 custom recurrent policy 和 dummy cell transport 接入。不要将 Tianshou 0.5.1 stock PPO 直接用于 persistent-hidden End2Race actor。

## 1. 范围与环境

本次只运行 constant-observation toy environments 和学习率为 0 的 toy PPO update；没有导入或训练 End2Race learner，没有修改 site-packages，也没有修改 `model.py`、`train.py`、`eval_multiagent.py` 或任何 PPO 实现。

初始环境中 SB3 已经是要求版本，所以没有执行安装，也没有更改 Gymnasium：

```text
Python 3.10.19
torch 2.7.0+cu128
gymnasium 1.2.3
tianshou 0.5.1

stable_baselines3 2.7.1
sb3_contrib 2.7.1
tianshou 0.5.1
package location: /home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages
python executable: /home/haowei/miniconda3/envs/end2race/bin/python
```

Tianshou import 时还打印了 legacy `gym` 的维护状态警告；这不改变本次实际记录的 Gymnasium 1.2.3，也未据此升级或降级任何包。

完整 `pip show` 字段在 `results.json`。`dump_source_evidence.py` 对题目指定的对象实际调用了 `inspect.getsource()`/`inspect.getsourcelines()`，完整输出在 `source_evidence.txt`；每个对象的绝对路径和范围也写入 `results.json`。

## 2. Tianshou 0.5.1 源码审计

### 2.1 Collector 给 actor 的 state

结论：本步 actor 输入是上一步 actor 返回并暂存在 `self.data.policy.hidden_state` 的 state；episode 第一步为 `None`/reset state。

证据：

- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/tianshou/data/collector.py`，`Collector.collect()`，L260-L287：L263 从 `self.data.policy` 弹出 `hidden_state` 为 `last_state`，L279/L281 调用 `self.policy(self.data, last_state)`，L285 取得本次返回的 next state。
- 同文件 `Collector._reset_state()`，L157-L166：episode done 后对 tensor/ndarray/Batch state 清零或清空；`Collector.collect()` L336-L350 在 done 时调用它。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/tianshou/policy/modelfree/pg.py`，`PGPolicy.forward()`，L100-L132：L120 将该 state 原样传给 `actor(batch.obs, state=state, ...)`，L132 把 actor 的 `hidden` 作为返回 `state`。

### 2.2 Buffer 保存 pre-action 还是 next state

结论：当前 transition 保存的是 actor 本次返回的 **next/post-observation state**，不是生成该 action 时 actor 的输入 state。

证据：

- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/tianshou/data/collector.py`，`Collector.collect()`，L282-L291：L285 读取 `result.state`，L287 写到 `policy.hidden_state`；L328-L331 随后把包含该 `policy` 的当前 transition 加入 buffer。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/tianshou/data/buffer/base.py`，`ReplayBuffer.add()`，L216-L273：L232-L237 复制输入 Batch 的全部键，L260/L272 将其写入 `_meta[ptr]`。这里没有把 hidden 向前移动一个 transition 的逻辑。

动态 probe 也显示 transition 0/1 存储的 hidden 分别为 `0.09078431`、`0.17074686`，正好是 GRU 消费相同步 observation 后连续返回的 `h_0`、`h_1`，不是 transition 0/1 所需的输入 `0`、`h_0`。

### 2.3 `PPOPolicy.process_fn()` 的 `logp_old`

结论：**没有**传入 rollout pre-action state，也没有使用 buffer 中保存的 post state。

证据：

- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/tianshou/policy/modelfree/ppo.py`，`PPOPolicy.process_fn()`，L86-L96：L95 直接执行 `self(batch).dist.log_prob(batch.act)`，没有 state 参数。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/tianshou/policy/modelfree/pg.py`，`PGPolicy.forward()`，L100-L120：state 默认值是 `None`，因此上述调用在 L120 以 `state=None` 调 actor。

所以 Tianshou 的 `logp_old` 并不是 rollout 时保存的 old log probability，而是 update 前用零/空 state 对每个 sampled observation 重新计算的值。对 memoryless actor 二者可能相同；对 persistent GRU 一般不相同。

### 2.4 `PPOPolicy.learn()` 的 new log probability

结论：**没有**把每段 sequence 的 rollout 初始 recurrent state 传给 actor。

证据：

- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/tianshou/policy/modelfree/ppo.py`，`PPOPolicy.learn()`，L98-L156：L105 以 `batch.split()` 生成 minibatch，L107 直接 `self(minibatch)`，仍无 state；L112-L114 用这个 distribution 与 `minibatch.logp_old` 形成 ratio。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/tianshou/policy/modelfree/pg.py`，`PGPolicy.forward()`，L100-L120：缺省 state 最终为 actor 的 `state=None`。

### 2.5 `Batch.split()` 是否保持 recurrent sequence/episode boundary

结论：**不保持**。默认会对所有 transition 做独立随机 permutation；它不知道 episode boundary，也不返回 sequence initial states 或 padding mask。

证据：

- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/tianshou/data/batch.py`，`Batch.split()`，L746-L770：`shuffle=True` 是默认值；L761-L764 对整个 batch 生成 permutation；L766-L770 仅按 size 切 index array。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/tianshou/policy/modelfree/ppo.py`，`PPOPolicy.learn()`，L105：调用未覆盖 `shuffle`，因此使用上述默认随机 permutation。

### 2.6 `stack_num` 的语义

结论：`stack_num` 首先是 **frame/observation history** 采样机制，不是 rollout 使用的真实 persistent hidden state。它也会按相同索引规则取 `policy` 字段，但 stock PPO 不把其中 hidden 作为 actor state 使用。

证据：

- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/tianshou/data/buffer/base.py`，`ReplayBuffer.__init__()`，L21-L28 和 L39-L61：参数明确称为 frame-stack sampling argument。
- 同文件 `ReplayBuffer.get()`，L316-L353：L325-L326 的例子返回 `[obs[t-3], ..., obs[t]]`；L347-L353 沿 `prev()` 收集并 stack 值。
- 同文件 `ReplayBuffer.__getitem__()`，L359-L388：L373/L375 通过 `get()` 取得 stacked `obs/obs_next`，L387 也取得 `policy`；但 `PPOPolicy.process_fn()` L95 和 `learn()` L107 都没有把 `batch.policy.hidden_state` 传给 policy。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/tianshou/utils/net/common.py`，`Recurrent.forward()`，L297-L328：训练输入预期为 `[batch, len, dim]`，state 为 `None` 时 L313-L314 从零状态重放这个 observation window。有限 observation window 不等于 rollout 时跨整个 episode 累积的 hidden。

### 2.7 Tianshou stock verdict

结论：对持续使用 persistent GRU hidden 的一般 actor，stock Tianshou 0.5.1 PPO **不能严格保证** `replay_logp == rollout_logp`。

源码上的充分原因是：Collector 把 post state 存进当前 transition；`process_fn()` 和 `learn()` 又完全不传 state；`Batch.split()` 还会打散 transition。动态测试得到最大误差 `9.187382698059082`，明确为 FAIL。

这个 verdict 不声称所有网络都会产生误差：memoryless actor、hidden 实际未被使用，或 observation stack 恰好完整重建 history 时可能偶然相等；stock API 对任意 persistent-hidden actor 没有严格保证。

## 3. SB3-Contrib 2.7.1 源码审计

### 3.1 Buffer 保存 pre-action 还是 post-action LSTM state

结论：保存 **pre-action** state。

证据：

- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/ppo_recurrent/ppo_recurrent.py`，`RecurrentPPO.collect_rollouts()`，L231-L242：先从 `_last_lstm_states` deepcopy 到 `lstm_states`，再由 policy forward 返回更新后的 `lstm_states`。
- 同函数 L287-L299：L294 写入 buffer 的却是尚未更新的 `self._last_lstm_states`；L299 才把 post-action `lstm_states` 赋回 `_last_lstm_states`。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/common/recurrent/buffers.py`，`RecurrentRolloutBuffer.add()`，L136-L145：把传入 pair 的 actor/critic hidden 和 cell 原样写入当前位置。

动态 buffer probe 的前三个 actor pre-action hidden 为 `[0.0, 0.61613768, 0.87155354]`，和上述顺序一致。

### 3.2 sequence 如何按 episode start/env boundary 切分

结论：sequence start 是 `episode_starts OR env_change`；minibatch 首元素也强制成为新 sequence。数据只做一次 circular rotation 来 shuffle，原 sequence 内相邻顺序不被 permutation 打乱。

证据：

- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/common/recurrent/buffers.py`，`create_sequencers()`，L64-L95：L82 对两类 boundary 做 logical-or，L84 强制首项为 start，L86-L89 得到每段 start/end。
- 同文件 `RecurrentRolloutBuffer.get()`，L147-L197：L157-L173 先按 env-major 顺序 flatten；L182-L186 只做 circular split/concatenate；L188-L191 标出每个 env 的首 timestep；L193-L197 再按 minibatch 切片。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/ppo_recurrent/ppo_recurrent.py`，`RecurrentPPO.collect_rollouts()`，L287-L299：当前 transition 保存 `_last_episode_starts`，env step 后以 `dones` 更新下一 transition 的 episode-start 标志。

### 3.3 minibatch padding

结论：每个 sequence 保持原长度，以 zero padding 补到该 minibatch 的最大 sequence length，然后 flatten 为 `n_seq * max_length`。

证据：

- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/common/recurrent/buffers.py`，`pad()`/`pad_and_flatten()`，L18-L61：L37 逐段切 tensor，L38 用 PyTorch `pad_sequence(..., batch_first=True, padding_value=0)`；L61 再 flatten scalar fields。
- 同文件 `RecurrentRolloutBuffer._get_samples()`，L199-L242：L210-L213 计算 `n_seq`/`max_length`/padded size；L231-L241 对 observation、action、old logp、advantage、return、episode-start 和 mask 使用同一 padding 方案。

本次 stock LSTM 动态测试确实产生 8 个 padding timestep：两个 minibatch 各为 10 valid + 4 padded，另外两个各为 10 valid + 0 padded。

### 3.4 每段 initial recurrent state 如何进入 `evaluate_actions()`

结论：buffer 从每段首 transition 的 **pre-action** state 构造 `(layers, n_seq, hidden)`，train 原样传给 `evaluate_actions()`，policy 再按 sequence 维度重放。

证据：

- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/common/recurrent/buffers.py`，`RecurrentRolloutBuffer._get_samples()`，L214-L229：L220-L226 以 `batch_inds[seq_start_indices]` 取 actor/critic 的 h/c，swap 成 `(layers, n_seq, dim)`；L239 放入返回的 `RNNStates`。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/ppo_recurrent/ppo_recurrent.py`，`RecurrentPPO.train()`，L336-L350：L345-L350 把 `rollout_data.lstm_states` 和 `episode_starts` 传给 `policy.evaluate_actions()`。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/common/recurrent/policies.py`，`RecurrentActorCriticPolicy._process_sequence()`，L162-L211：L182-L187 用 state 的 `n_seq` 恢复 time-major sequence；L191-L205 从传入 h/c 开始并在 episode-start 时清零。`evaluate_actions()` L310-L345 在 L331/L333 对 actor/critic 调用该函数。

### 3.5 padding timestep 是否通过 mask 排除

结论：是。padding 位置仍会产生网络输出和 ratio，但不会进入 advantage normalization、policy loss、clip fraction、value loss、entropy loss 或 KL。

证据：

- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/common/recurrent/buffers.py`，`RecurrentRolloutBuffer._get_samples()`，L231-L242：L241 对全 1 valid mask 使用同一 zero-padding，因而 padding 自动为 0。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/ppo_recurrent/ppo_recurrent.py`，`RecurrentPPO.train()`，L342-L404：L343 转 bool mask；L356、L364、L368、L382、L389/L391、L403 分别 mask normalization、policy、clip fraction、value、entropy 和 KL。

动态测试中 padding 上的最大 logp 差为 `11.5923061`，但 40 个 valid timestep 的最大差仅 `2.3841858e-07`。这直接验证了 verdict 必须且确实只基于 mask 内 timestep。

### 3.6 timeout value bootstrap

结论：当 VecEnv 返回 done、`terminal_observation` 且 `TimeLimit.truncated=True` 时，用 **post-action critic LSTM state** 对 terminal observation 估值，把 `gamma * terminal_value` 加入当前 reward；正常 termination 不走此分支。

证据：

- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/ppo_recurrent/ppo_recurrent.py`，`RecurrentPPO.collect_rollouts()`，L268-L285：L270-L275 检查 timeout 条件；L278-L281 从 forward 返回的 `lstm_states.vf` 取对应 env；L283 令 episode-start 为 false；L284-L285 计算 terminal value 并加到 reward。
- 同函数 L287-L306：修正后的 reward 随 transition 入 buffer，rollout 结束后再计算 return/advantage。

toy env 原始每步 reward 为 `1.0`：正常终止 buffer reward 保持 `1.0`；timeout buffer reward 为 `0.58067441`。后者小于 1 是因为该固定随机 critic 的 terminal value 为负，不代表没有 bootstrap。

### 3.7 GRU hidden + dummy LSTM cell 可行性

结论：**可行，但不是把 `nn.LSTM` 直接换成 `nn.GRU` 就结束。** `RecurrentPPO` 和 `RecurrentRolloutBuffer` 可以保持 stock；必须提供 custom `RecurrentActorCriticPolicy`，让 h transport 成为真实 GRU state、忽略 c 并返回同形 zero dummy cell，同时实现 GRU-aware sequence reset/unroll。

证据：

- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/ppo_recurrent/ppo_recurrent.py`，`RecurrentPPO._setup_model()`，L139-L185：L154-L174 只从 `policy.lstm_actor` 读取 `num_layers`/`hidden_size` 来建立两个同形 state 和 buffer shape；`nn.GRU` 也有这些属性。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/common/recurrent/buffers.py`，`RecurrentRolloutBuffer.reset()`/`add()`，L129-L145：buffer 只分配并复制 h/c arrays，不解释 cell 的数值语义，所以 c 可以是 dummy tensor。
- `/home/haowei/miniconda3/envs/end2race/lib/python3.10/site-packages/sb3_contrib/common/recurrent/policies.py`，`RecurrentActorCriticPolicy._process_sequence()`，L162-L211：stock 实现 L192/L199-L205 按 LSTM `(h, c)` 调用 module；GRU 只接受 h，因此必须在 custom policy 覆盖这段 adapter 行为。
- 同文件 `RecurrentActorCriticPolicy.__init__()`，L113-L148：parent 创建 LSTM，并在 L147-L148 建 optimizer。如果 adapter 在 `super().__init__()` 后替换 module，必须重建 optimizer；生产实现也可通过完整 subclass 在 optimizer 创建前组装 GRU。

审计目录内的 `GRUDummyCellPolicy` smoke test 未修改两个 library class：actor/critic buffer cell 全为 0，40 个 valid timestep 的 logp 最大误差为 `0.0`，一次 lr=0 train 后参数最大变化为 `0.0`。

## 4. 最小动态测试设计与结果

两个主测试都满足：observation 每步固定为 `[1.0]`；recurrent state 随 history 累积；同一 observation 的 deterministic action 至少出现两个类别；episode 长度 20；先 normal termination、再 timeout；总计 40 transitions；实际执行一次学习率为 0 的 PPO update。

Tianshou 使用真实 `nn.GRU` 和强 history-dependent categorical head。`LoggingOnlyPPOPolicy` 只把 rollout distribution 对已采样 action 的 log probability 放进 `result.policy.rollout_logp`；它没有改变 hidden、Collector、buffer、`process_fn()` 或 `learn()`。因此这是 telemetry subclass，不是 recurrent 修复。

SB3 主测试使用未 subclass 的 stock `MlpLstmPolicy`/`RecurrentActorCriticPolicy`；只把 LSTM/action-head 权重固定为容易观察 history 影响的值，没有改 policy、buffer 或 PPO 控制流。

| 检查项 | Tianshou | SB3-Contrib stock LSTM |
|---|---:|---:|
| transitions / episodes | 40 / 2 | 40 / 2 |
| end types | terminated, truncated | terminated, truncated |
| history-conditioned deterministic action 类别数 | 2 | 2 |
| valid replay-vs-rollout max error | `9.1873826981` | `2.3841857910e-07` |
| valid mean error | `5.4995374680` | `1.1920929133e-08` |
| padding timesteps | 不适用（stock 无 sequence padding） | 8 |
| optimizer lr | 0 | 0 |
| update 后参数 max delta | 0 | 0 |
| verdict | **FAIL** | **PASS** |

## 5. End2Race 集成改动量估计

以下是未来集成估计，不是本次已实施修改。当前 `model.py:30-37` 是 1-layer batch-first GRU，`model.py:66-100` 的 `End2Race.forward()` 已显式接收/返回 hidden；可通过 wrapper/composition 保持 `model.py` 不变。当前 checkout 中 `rl/env.py`、`rl/sb3_policy.py`、`rl/callbacks.py`、`rl/checkpoint.py` 是空 scaffold，根目录没有 `train_ppo.py`。

| 方案 | correctness-critical 文件估计 | 函数/方法估计 | 主要工作 |
|---|---:|---:|---|
| SB3-Contrib | 3 个 | 8-12 个 | `rl/sb3_policy.py` 中 End2Race distribution/value heads、GRU sequence adapter、`forward/evaluate_actions/predict_values`；`rl/env.py` 的 obs/action 与 terminated/truncated contract；一个新的 RL runner 做 RecurrentPPO wiring |
| Tianshou 0.5.1 | 5 个 | 12-18 个 | actor/critic adapter；保存 rollout logp 和 **pre-action** hidden 的 Collector 路径；episode-aware sequence sampler/padding；覆盖 PPO `process_fn/learn` 并传 sequence initial state；env/runner wiring |

若 Tianshou 仅把每个 transition 的 pre-action hidden 当作 detached input，可以减少 sequence sampler 工作并修正 logp identity，但不会提供跨 sequence 的 recurrent BPTT；因此不把这个较弱方案计作完整 recurrent PPO 集成。

SB3 仍需要 custom policy，因为 stock policy 是 LSTM，不会自动理解 End2Race 的 GRU、LiDAR/speed 双输入和既有 action head；优势是 rollout buffer、sequence minibatching、mask、timeout 以及 PPO update 本身无需重写。

## 6. 建议与复现

建议采用 **SB3-Contrib 2.7.1 RecurrentPPO + repo-local GRU/dummy-cell custom policy**。在真正接入 End2Race 前，应保留本审计的 replay identity、episode reset 和 actor identity 测试作为门禁，并额外验证连续 action distribution 的 squash/scale 与现有 steering/speed 语义。

复现命令：

```bash
/home/haowei/miniconda3/envs/end2race/bin/python scripts/rl_library_audit/run_audit.py
/home/haowei/miniconda3/envs/end2race/bin/python scripts/rl_library_audit/dump_source_evidence.py
```

产物：

- `scripts/rl_library_audit/results.json`：机器可读版本、源码位置、动态指标。
- `scripts/rl_library_audit/source_evidence.txt`：由 `inspect.getsource()` 读取的完整指定源码。
- `scripts/rl_library_audit/run_audit.py`：toy tests、telemetry subclass 和 GRU dummy-cell smoke test。

本审计没有进行任何 End2Race learner 训练，也没有创建 Git commit。
