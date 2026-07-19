# End2Race PPO 最终性能主线整合报告

最终结论：`READY_FOR_DEFAULT_PPO_TRAINING`

当前 HEAD 为 `e1c0d2b61e4ebc5c619f4c013dad330acf1fdfa0`。本次未 commit、
未 push；全部生产修改保留在工作树，等待 owner 检查。

## 唯一默认架构

正式 PPO 路径现为 Phase 1–4 + Phase 5B：16 个 logical env、6 个
`forkserver` simulator worker、父进程中央 scenario/reset 调度、planner
静态资产缓存、C0 actor-h-only rollout buffer，以及仅用于
`evaluate_actions()` 的逐 timestep active-slot FP32 GRU batching。

Collection 继续逐 logical env 使用原 batch-1 actor forward。没有集成 A、C、
A+B 或 A+C；生产代码中没有 backend registry、selector、auto selection、
packed replay 或 collection batching。

TF32/数据类型合同在任何 CUDA model build 前设置并在 B replay 运行时
fail closed：cuDNN TF32 off、CUDA matmul TF32 off、matmul precision
`highest`、cuDNN benchmark off；model/input/hidden/distribution/gradient 保持
FP32。未使用 Float64 正式训练、AMP、FP16、BF16、`torch.compile` 或强制
deterministic algorithms。

## 修改文件与整合位置

- `ppo/buffer.py`：C0 buffer reset 时清空 minibatch validity 元数据；仍只保存
  actor hidden h，并在取 batch 时惰性生成三个零 transport state。
- `ppo/policy.py`：collection `_actor_forward()` 保持 batch-1；新增并固定
  `_actor_replay_batched()` 为唯一 training replay；加入 dtype、device、TF32、
  padded shape fail-closed 合同；C0 轻量 buffer 通过显式状态语义检查选用。
- `ppo/vec_env.py`：保留父进程中央调度与 worker lifecycle；线程限制改为
  production fail-closed；删除 process/step/reset 的实验计时遥测和额外 IPC
  字段。
- `train_ppo.py`：统一设置 TF32-off FP32 合同；删除可选 Dummy production
  分支，`build_training_vector_env()` 只构造中央 6-worker 路径。
- `tests/test_ppo_batched_replay.py`：保留唯一最小 batch-1 reference helper，
  覆盖 active mapping、hidden gather/scatter、padding、episode reset、layout、
  masked gradient 和 fail-closed 合同。

Phase 1 的实现主体位于 `ppo/vec_env.py` 与 `train_ppo.py`；Phase 2 位于
`ppo/environment.py`；Phase 3 位于 `ppo/buffer.py` 和
`End2RaceRecurrentPPO._setup_model()`；Phase 4 的 invalid-padding skip 已并入
Phase 5B 的 `_actor_replay_batched()`。

## 数据流审计

Reset 流为父进程 sampler → `CentralEpisodeScheduler` → `EpisodeResetSpec` →
原 rank 所属 worker → reset observation → SB3 episode_start。worker 不持有
sampler/visit count，auto-reset 结果按 rank 写回。固定 action replay 与已提交的
Phase 1 artifact byte-identical，observation、executed action、reward、
terminated/truncated、outcome、reset order 和 visit counts 全部 PASS。

Collection 流未改变：raw observation → 361D actor observation（包含 previous
desired speed）→ batch-1 GRU → physical mean → 原 distribution → sampled action/
old logp → env.step。zero-LR capture 中 policy action 与 executed action 的完整
step hash 相同：`b34c892f…`。

Rollout buffer 仅保存 actor h；observation/action/value/logp/advantage/return/
episode_start 的 flatten 与 minibatch 顺序未改。正式 update 有 25,600 个有效
position、38,400 个 padded-flat position，padding ratio `1.50`，四个 minibatch
的 sequence slot 数为 `13/12/10/13`。

Training replay B 每 timestep 按原 slot 升序 gather active feature/hidden，执行
一次 batched FP32 GRU，再显式 scatter 回原 slot。invalid padding 不进入 actor、
不更新 hidden；最终恢复原 padded-flat layout。mask、advantage normalization
集合、critic C0、loss reduction 和 optimizer-step 边界均未改变。

## 参数与 checkpoint 流

初始 model/optimizer hash 分别为：

- `9b2feab66b34caa6c94f3038c18c0b99bcd7c9e7230ee2be259abf46fa79bbf8`
- `104e720666be943109e4a888dc8285a667998bb66c5e531c82b9a3bac60b2dee`

它们与 current-HEAD reference bundle 完全一致。optimizer group 顺序仍为
GRU/head/critic，LR 为 `1e-6/1e-5/3e-4`，weight decay 为 0，max grad norm
为 `0.5`。12 个 trainable parameter tensor 进入 optimizer；`k`、speed MLP、
dummy embedding 与 `log_std` 继续冻结。

zero-LR 完整 update 中 17 个 model parameter delta 全部严格为 0。正式 nonzero
update 完成预期 4 个 optimizer steps，active parameter 数 12，训练 finite；
冻结 actor 与 `log_std` 的 max delta 均为 0。已注册四-minibatch B 审计最差
gradient cosine `0.9999999995`、relative L2 `3.24e-5`；parameter-delta cosine
`0.99999999996`、relative L2 `9.37e-6`，全部 PASS。

`save_actor()` 未改。旧 BC checkpoint 和新 PPO actor checkpoint 都保持原
12-key schema 并 strict-load；新 checkpoint hash 为 `315739f2…`。evaluator
无需修改。

## 验证结果

| 验证 | 结果 |
|---|---|
| T1 compile/import、diff check、dtype/TF32、checkpoint | PASS |
| T2 reset/replay、worker lifecycle、planner cache、buffer/padding | PASS |
| T3 Phase 5B 专项测试 | 8/8 PASS |
| T4 全部现有测试 | 21/21 PASS |
| zero-LR 完整 update | PASS |
| T5 warm-up + 一次正式完整 update | PASS |

正式 lifecycle 检查覆盖正常 close 和 worker 异常传播/cleanup；最终没有
worker 或 zombie 残留。正式环境未安装 pytest，因此使用仓库原有 unittest
runner，未修改 site-packages。

## 一次正式性能结果

配置：N1-H1F-p50，seed `20260917`，16 env，6 workers，1600 steps，
batch 6400，1 epoch，25,600 transitions。

| Pipeline | rollout s | train s | total s | total trans/s |
|---|---:|---:|---:|---:|
| Phase 1–4 reference | 19.024 | 15.182 | 34.432 | 743.5 |
| Final Phase 1–4 + 5B | 18.952 | 3.473 | 22.628 | 1131.3 |

Train 改善 `77.13%`（要求 ≥65%），total update 改善 `34.28%`（要求
≥25%），总吞吐改善 `52.17%`；两个性能 gate 均 PASS。正式测量完成 35 个
episode，outcome 为 collision/follow/overtake `13/9/13`，与 Phase 1–4
reference 相同。peak allocated VRAM `797.5 MiB`，RSS median `8011.3 MiB`。

## 仓库状态

生产 diff 仅涉及 `ppo/buffer.py`、`ppo/policy.py`、`ppo/vec_env.py` 和
`train_ppo.py`；新增一个 untracked 专项测试。未修改 PPO 算法、reward、
critic、action distribution、GAE、训练配置、scenario distribution、终止语义、
RNG 语义或 evaluator。未修改 site-packages，未 stage、commit 或 push。

完整结构化证据见同目录 `DATA_FLOW_AUDIT.json`、
`PARAMETER_FLOW_AUDIT.json`、`VALIDATION_SUMMARY.json` 和
`PRODUCTION_DIFF.patch`。

## 最终结论

`READY_FOR_DEFAULT_PPO_TRAINING`
