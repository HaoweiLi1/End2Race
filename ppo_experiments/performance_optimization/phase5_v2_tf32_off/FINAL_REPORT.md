# End2Race PPO Phase 5-v2 TF32-Off Batched GRU Audit

## 最终 verdict

`PHASE5_COMBINED_FORWARD_AB_REQUIRES_PRODUCT_TEST`

- **BEST_LOW_ERROR：B**
- **BEST_BALANCED：A+B**
- **BEST_MAX_SPEED：A+C**
- **唯一下一步产品分布验证候选：A+B**

以上均为实验分支结论，不代表 merge/deployment ready。A 相关路径的真实
closed-loop 会因正常 FP32 差异分叉，必须经过更大产品分布验证；本任务没有
修改或自动合入正式 backend。

## 基线与不可变 provenance

- HEAD：`e1c0d2b61e4ebc5c619f4c013dad330acf1fdfa0`
- config：`N1-H1F-p50`，16 env，6 forkserver workers，n_steps=1600，
  batch_size=6400，n_epochs=1，25,600 transitions/update
- BC：`b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4`
- H1 manifest：`ad35a6d56dddfe7c5e0460877f3aeb41ecd428d83c61ca1d1dc82c2b8709b0b8`
- frozen rollout：`f64819004506e018ef972205ae16b6a80aef24b0ca6d42bedc47fefd26746f6a`
- initial model：`9b2feab66b34caa6c94f3038c18c0b99bcd7c9e7230ee2be259abf46fa79bbf8`
- initial optimizer：`104e720666be943109e4a888dc8285a667998bb66c5e531c82b9a3bac60b2dee`
- minibatch order：`d2155c247ecdc05367a26640526f40d98b622b76032efeeff4669a6c15bd295b`
- Python 3.10.19、NumPy 1.26.4、PyTorch 2.7.0+cu128、CUDA 12.8、
  cuDNN 90701、SB3/sb3-contrib 2.7.1、RTX 4080 SUPER

原 current-HEAD bundle、rollout、model/optimizer/RNG 和 minibatch 文件未重新
生成。旧 `ef8570bc...` evidence 也保持不变。此前 stale-reference 停止仍标为
`STOP_STALE_REFERENCE_ARTIFACTS / NO_MECHANISM_TEST_RUN`。

## Stage 0：TF32 因果与绝对 logp gate

CPU/GPU Float64 所有 A/B/C 中间层最大误差分别为 `1.60e-14` 和
`1.07e-14`，语义 gate PASS。TF32-off 使 core GRU/hidden/action/latent
误差缩小 `279x–939x`。

| Candidate | logp mean | p95 | p99 | max | `max<=1e-5` |
|---|---:|---:|---:|---:|---|
| A | 4.17e-7 | 1.31e-6 | 2.17e-6 | 2.38e-6 | PASS |
| B | 1.58e-7 | 7.15e-7 | 1.52e-6 | 3.34e-6 | PASS |
| C | 1.85e-7 | 9.54e-7 | 1.82e-6 | 2.38e-6 | PASS |

结论为 `TF32_DOMINANT_FOR_CORE_FORWARD_NUMERICS /
LOGP_ABSOLUTE_ERROR_PASS`。

## Stage 1：R0 与 R1

R0/R1 的 actor mean、sampled action、hidden、old logp、value、rollout、
loss、gradient、parameter delta、optimizer、固定 action 文件和 12-key
checkpoint 均 bitwise identical：`R1_BASELINE_EXACT`。

正式性能均先独立 warm-up，再测一次：

| Reference | rollout s | train s | total s | total trans/s |
|---|---:|---:|---:|---:|
| R0 default TF32 | 18.675 | 15.063 | 34.141 | 749.8 |
| R1 TF32 off | 18.401 | 14.999 | 33.800 | 757.4 |

R1 total 比 R0 快约 `1.0%`，属于单次测量波动；没有观察到关闭 TF32 的速度
成本。

## Stage 2：A collection batch16

A1 单步真实 rollout 输入：

- processed feature exact；
- hidden max `6.85e-7`；
- physical mean max `4.77e-7`；
- latent mean p99 `9.76e-8`；
- sampled logp p99/max `4.77e-7 / 4.77e-7`；
- 全部 A1 gate PASS。

A2 1,400-step teacher-forced：

- step 1400 hidden p99/max `3.87e-7 / 7.18e-6`；
- physical mean p99/max `9.54e-7 / 9.54e-7`；
- sampled logp p99/max `5.36e-6 / 5.72e-6`；
- 无指数发散，全部 gate PASS。

A3 使用独立新进程运行相同 seed 的 R1/A closed-loop。observation 从 step 3
分叉，跨闭环 action 首次超过 `1e-5/1e-4/1e-3` 的 step 为 `23/45/45`。
在同一 candidate trajectory、相同 observation 下，A 对 batch-1 的 action
max 仍仅 `2.86e-6`，说明后续大分叉来自闭环敏感性而非单步 numerical gate
失败。Quick run outcome 从 R1 的 collision/follow/overtake=`13/9/13` 变为
A 的 `16/6/13`。因此 A 只能标为
`DISTRIBUTIONAL_ONLY_REQUIRES_PRODUCT_TEST`。

## Stage 3：B/C 四 minibatch 数值审计

四个 logical minibatch 均从完全相同的 model、optimizer、training RNG、
frozen rollout 和 minibatch order 开始并按同一顺序连续更新。

| 最大/最差指标（4 minibatches） | B | C | Gate |
|---|---:|---:|---|
| reference-vs-candidate policy KL | 1.09e-10 | 5.89e-9 | <=1e-6 |
| gradient cosine min | 0.9999999995 | 0.9999996691 | >=0.99999 |
| gradient relative L2 max | 3.24e-5 | 8.14e-4 | <=1e-3 |
| delta cosine min | 0.9999999996 | 0.9999998141 | >=0.99999 |
| delta relative L2 max | 2.68e-5 | 6.12e-4 | <=1e-3 |
| policy-loss abs max | 2.71e-7 | 3.03e-6 | <=1e-5 |
| hidden p99/max | 4.17e-7 / 1.10e-5 | 1.13e-6 / 1.93e-5 | recorded |
| new-logp p99/max | 3.20e-5 / 4.27e-3 | 3.42e-4 / 1.35e-2 | recorded |

两者所有注册 gate、finite、optimizer step count 和 strict 12-key checkpoint
均 PASS。完整 logp 尾部明显大于小型 Stage 0 oracle，但 KL、loss、gradient 和
delta 仍在预注册训练 gate 内；未隐瞒或重新放宽阈值。B 的 gradient/delta
误差约比 C 低一个数量级以上，因此 C 不支配 B。

冻结训练 microbenchmark：R1/B/C 全四 actor forward 为
`3644.65/1003.63/116.35 ms`；完整 frozen train 为
`15.293/3.984/0.873 s`。

## Stage 4–5：正式性能与组合

所有点均独立 warm-up 一次、正式测量一次；只在完整计时区间边界同步，未在
每次 GRU call 前后强制同步。padding ratio 保持 `1.50`。

| Backend | rollout s | train s | total s | total trans/s | total改善 | actor speedup | peak VRAM MiB | RSS median MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 18.401 | 14.999 | 33.800 | 757.4 | — | 1.00x | 832.7 | 7925.9 |
| A | 15.519 | 14.993 | 30.907 | 828.3 | 8.6% | 3.67x collection | 832.7 | 7971.5 |
| B | 18.698 | 3.405 | 22.500 | 1137.8 | 33.4% | 3.44x training | 797.5 | 7854.3 |
| C | 18.337 | 0.475 | 19.220 | 1332.0 | 43.1% | 27.07x training | 685.2 | 7558.0 |
| A+B | 15.191 | 3.389 | 18.981 | 1348.7 | 43.8% | 3.69x collection | 797.4 | 7891.6 |
| A+C | 15.040 | 0.461 | 15.901 | 1609.9 | 53.0% | 3.81x collection | 685.2 | 7612.1 |

A、B、C 都通过各自性能 gate。C train 比 B 快约 `86%`，但其
gradient/delta error 显著更高，触发“保留 B 和 C、同时测试两种组合”的
条件。A+B/A+C 都完成一个 25,600-transition rollout、一个 PPO update、
正常 worker close，以及可 strict-load 的原 12-key actor checkpoint。

## Pareto 与选择

- Collection frontier：A batch16；因为默认 batch16 通过，无需运行 batch8/4。
- Training frontier：B、C；两者分别占低误差和最高训练速度端。
- Full-update frontier：R1、A、B、C、A+B、A+C 均因误差与速度维度互有取舍
  而保留。
- BEST_LOW_ERROR（排除 reference）：B。
- BEST_BALANCED：A+B。它保留 B 的低 training error，同时 total update
  改善 `43.8%`。
- BEST_MAX_SPEED：A+C，total update 改善 `53.0%`。
- 唯一产品分布验证候选：A+B；A+C 保留为 max-speed 对照，不应先于 A+B
  进入产品验证。

## 必答问题

1. 原大误差是否主要由 TF32 导致？**是，对 core forward numerics 有充分
   证据；logp 绝对误差 gate 也通过。**
2. R0/R1 是否等价、速度成本？**bitwise exact；未观察到成本，repeat1
   total 约快 1.0%。**
3. A 的误差和速度？**A1/A2 全部通过；closed-loop 分叉，因此仅
   distributional；rollout/total 改善 `15.7%/8.6%`。**
4. B 四 minibatch 与速度？**全部 numeric gate PASS；train/total 改善
   `77.3%/33.4%`。**
5. C 四 minibatch 与速度？**全部 numeric gate PASS；train/total 改善
   `96.8%/43.1%`。**
6. B/C 谁占优？**B 低误差，C 高速度；互不支配。**
7. A 是否值得产品分布验证？**值得，但只能作为包含 A 的 distributional
   candidate 验证。**
8. A+B/A+C 是否在 frontier？**两者都在。**
9. BEST_LOW_ERROR/BALANCED/MAX_SPEED？**B / A+B / A+C。**
10. 唯一下一步产品候选？**A+B。**
11. 是否没有候选值得继续？**否；A+B 值得进入正式产品分布验证。**

## 仓库完整性

所有实现、raw diagnostics、性能结果和 checkpoint 都仅位于本 untracked
实验目录。未修改 tracked production files、site-packages 或 evaluator；未
commit、未 push。最终封存检查已通过：7 个 production source、14 个
current-HEAD reference artifact、6 个 legacy artifact 和 3 个输入的 hash
一致；35 个 JSON 全部可解析；10 个 actor checkpoint 均为原 12-key schema
并可 strict-load；没有 worker/zombie 残留。
