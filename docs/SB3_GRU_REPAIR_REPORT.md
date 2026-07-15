# SB3-Contrib GRU RecurrentPPO real-integration repair

日期：2026-07-15（Asia/Singapore）

## Verdict: PASS_FOR_NONZERO_LR_SMOKE

所有本次定义的 hard gate 已通过。这个 verdict 只表示可以在 owner 审阅后进入“最多一次极小非零学习率 update smoke”，不表示已完成正式 PPO 训练、reward 设计或性能验证。

本次没有运行非零学习率 update，没有运行正式 learner，没有 reward sweep，没有修改 site-packages，没有覆盖 checkpoint，也没有创建 Git commit。

## Scope and preserved architecture

修改了：

- `rl/sb3_end2race_policy.py`
- `rl/end2race_gymnasium_env.py`
- `scripts/smoke_sb3_gru.py`
- `tests/test_sb3_gru_integration.py`
- `docs/SB3_GRU_POC_REPORT.md`
- 新增本报告。

保留不变：

- stock SB3-Contrib 2.7.1 `RecurrentPPO`；
- stock `RecurrentRolloutBuffer`；
- 原始 `End2Race` GRU actor 及 12-key actor-only checkpoint schema；
- GRU hidden + dummy zero cell transport；
- independent feed-forward critic；
- opponent 为固定 Lattice Planner，不属于 policy/optimizer/value/advantage。

环境版本：

```text
Python                  3.10.19
torch                   2.7.0+cu128
gymnasium               1.2.3
stable-baselines3       2.7.1
sb3-contrib             2.7.1
```

## 1. Reset and asynchronous auto-reset evidence

`EpisodeResetSpec` 现在包含：

```text
poses:                 (num_agents, 3)
initial_speed_feature: float
scenario:              dict
```

`End2RaceGymnasiumEnv.reset()` 在每次 reset 都先调用 env-local provider，然后始终调用：

```python
f110_env.reset(poses=spec.poses)
```

验收结果：

| 项目 | 结果 |
|---|---:|
| parallel envs | 2 |
| synthetic episode lengths | 4 / 7 steps |
| staggered auto-resets | 3 |
| reset counts（含首次） | 3 / 2 |
| 所有 reset pose shapes | 全部 `(2,3)` |
| strict core 收到 poses | PASS |
| provider/controller instances 独立 | PASS |
| parallel env RNG samples 独立 | PASS |
| 两 env planner identities 无交集 | PASS |
| 同 seed reset pose max error | `0.0` |
| options fixed-spec override | PASS |
| options override 时 provider 仍被调用 | PASS |
| scenario-only options override | PASS |
| strict core 缺少 poses 时立即拒绝 reset | PASS |

每次 reset 都重新创建 Lattice Planner。初始 snapshot 确认：

- tracker previous error = 0；
- cached trajectory = absent；
- tracker counter = 0；
- planner step = 0；
- previous opponent pose = 0；
- previous local trajectory = 0。

## 2. Speed-feature timestep trace

正式采用原 `eval_multiagent.py` deployment timing：

```text
BC offline training:          previous desired-speed command
original deployment evaluator: previous measured speed
PPO wrapper:                  exact evaluator timing
```

独立 oracle 使用可区分数列，结果：

| decision observation | wrapper speed feature | independent oracle |
|---|---:|---:|
| reset | 11 | 11 |
| after step 1 | 21 | 21 |
| after step 2 | 22 | 22 |

```text
speed feature max error: 0.0
previous desired commands 12/13 used: no
current measured speeds 22/23 used at same observation: no
```

reset 值来自 `EpisodeResetSpec.initial_speed_feature`。每次 step 在 core step 之前保存当前 raw observation 的 ego measured speed，并将它与 core step 返回的下一帧 LiDAR 组合。

## 3. Ego-only collision and terminal precedence

collision 只定义为：

```python
ego_collision = bool(collisions.size > ego_index and collisions[ego_index])
```

opponent collision 只记录为 diagnostic，不修改 reward、termination 或 PPO KPI。表驱动测试：

| case | terminated | truncated | result |
|---|---:|---:|---|
| ego collision only | true | false | PASS |
| opponent collision only | false | false | PASS |
| base terminal only | true | false | PASS |
| timeout only | false | true | PASS |
| base truncation only | false | true | PASS |
| ego collision + timeout | true | false | PASS |
| opponent collision + timeout | false | true | PASS |
| base terminal + timeout | true | false | PASS |
| base truncation + timeout | false | true | PASS |
| ego + opponent collision | true | false | PASS |

base reward 原样返回，10 个 case 的 max reward error 为 `0.0`。合成 stock rollout 中实际出现 5 个 opponent-only collision transition，它们全部 `terminated=False, truncated=False`，下一 transition 继续同一 ego episode。

## 4. Evaluator-compatible physical action distribution

不再使用两维普通 `DiagGaussianDistribution` + SB3 Box clip。repo-local joint distribution 为：

```text
z_steer ~ Normal(mu_z, steer_std)
steering = 0.52 * tanh(z_steer)

speed ~ Normal(raw_End2Race_speed, speed_std)
```

`mu_z` 由 clipped raw End2Race steering 的 inverse tanh 得到，使 deterministic physical mode 匹配原 evaluator：

```text
steering = clip(raw_End2Race_steering, -0.52, 0.52)
speed    = raw_End2Race_speed
```

steering log probability 包含 inverse tanh、base Normal density、`tanh` Jacobian 和 `0.52` scaling Jacobian。一个不复用 distribution helper 的独立 scalar oracle 结果：

```text
joint log-prob independent-oracle error: 1.11e-16
```

SB3 2.7.1 要求 Box 两端必须有限，因此 speed 使用 float32 最大有限范围 `[-finfo.max, finfo.max]` 作为明确 no-op bound，并用每步 clipping-count 测试保证 stock SB3 没有改变 speed sample。

### Ego action trace

第一个合成 transition 的 trace：

| stage | `[steering, speed]` |
|---|---|
| raw End2Race mean | `[0.43346083, 2.01734710]` |
| distribution physical sample | `[0.43356800, 1.34332871]` |
| rollout buffer action | `[0.43356800, 1.34332871]` |
| SB3 action passed to wrapper | `[0.43356800, 1.34332871]` |
| wrapper ego joint-action slot | `[0.43356800, 1.34332871]` |
| core env received ego action | `[0.43356800, 1.34332871]` |
| fixed opponent action | `[0.07000000, 1.14999998]` |
| old log probability | `1.77697551` |

全 rollout 结果：

```text
max |buffer ego action - core ego action|:   0.0
max |SB3 ego action - core ego action|:      0.0
max |wrapper ego action - core ego action|:  0.0
SB3 pre-env clipping count:                  0
SB3 pre-env max clip delta:                  0.0
steering out-of-bound count:                 0
fully traced ego transitions:                20
```

synthetic core 的 next observation 和 reward 都显式依赖 executed ego action，不再是 action-ignoring mock。

```text
action-sensitive observation oracle max error: 8.80e-09
PPO buffer action width:                         2
opponent action present in PPO buffer:           false
```

## 5. Opponent controller trace and scope

- ego action 唯一来自 SB3 PPO；
- opponent action 来自 fresh `setup_opp_planner()` Lattice Planner；
- trajectory 每 `planner.conf.tracker_steps` 重规划；
- Pure Pursuit 每 simulator step 调用；
- opponent steering 被裁剪到 `[-0.52,0.52]`；
- opponent speed 乘 scenario `opp_speedscale`；
- controller `actions()` 不接收 PPO ego action，policy 只能通过后续环境状态间接影响 opponent；
- 同 raw state/scenario 的两个 fresh controller 输出 max error `0.0`；
- opponent action 不进 PPO buffer，opponent state 不进 361D actor observation，opponent planner parameter 数在 optimizer 中为 0。

## 6. Recurrent replay identity and independent coverage

coverage 不再使用硬编码布尔值，而是从 stock buffer、minibatch masks、episode starts 和 wrapper events 推导。

| metric | result | gate |
|---|---:|---:|
| valid timesteps | 20 | `== 20` |
| each original transition counted once | true | true |
| padding timesteps | 3 | `> 0` |
| continuation sequences | 3 | `> 0` |
| continuation sequences with nonzero pre-action h | 3 | `> 0` |
| raw nonzero pre-action states | 15 | `> 0` |
| independent continuation-hidden oracle error | `0.0` | `<= 1e-6` |
| timeout events | 1 | `> 0` |
| ego true terminal events | 2 | `> 0` |
| opponent-only collision continuations | 5 | `> 0` |
| replay logp max / mean error | `0.0 / 0.0` | `<= 1e-6 / 1e-7` |
| ratio max deviation | `0.0` | `<= 1e-6` |

独立 hidden oracle 从对应 episode 起点以 zero h 逐步调用原 `End2Race`，重建 continuation sequence 起点的 pre-action h；没有调用 policy 的 sequence helper。

## 7. Bootstrap and advantage boundaries

```text
ego collision zero-bootstrap max error:       5.66e-08
timeout terminal-value bootstrap max error:   6.38e-08
terminal advantage boundary max error:        5.96e-08
```

true terminal 优先于同 step timeout，不 bootstrap。timeout 使用 DummyVecEnv 在 auto-reset 前保存的 `terminal_observation`，不使用 reset 后 observation。

## 8. Gradient ownership and optimizer classification

只做 backward，不调用 optimizer step：

| loss | active actor | critic | distribution `log_std` |
|---|---:|---:|---:|
| policy loss | 11/11 finite nonzero | 0/4 | 1/1 finite nonzero |
| value loss | 0/11 | 4/4 finite nonzero | 0/1 |

`dummy_embedding` 在 `mask_prob=0` 时按预期无梯度。

optimizer 只有一个 parameter group：

| classification | tensor count | scalar count |
|---|---:|---:|
| End2Race actor（含 dummy embedding） | 12 | 11,301,482 |
| independent critic | 4 | 11,617 |
| joint-distribution `log_std` | 1 | 2 |
| total | 17 | 11,313,101 |
| inherited unused `action_net` | 0 | 0 |
| opponent planner | 0 | 0 |

parameter identities 全部唯一，所有 optimizer parameter 都精确归类为 actor/critic/distribution，无未分类或重复项。

## 9. LiDAR contract

第一版严格保留 evaluator 索引：

```python
np.linspace(0, n - 1, 360, dtype=int)
```

monotonic 720-beam oracle 结果：

```text
first / last selected index: 0 / 719
all 360 selected beam max error: 0.0
```

fail-fast 结果：

```text
scan.size < 360: PASS (raises ValueError)
NaN:             PASS (raises ValueError)
Inf:             PASS (raises ValueError)
```

只改变 opponent pose/reference geometry，保持 LiDAR/speed 相同的 metamorphic test 得到 actor observation max error `0.0`，确认 privileged simulator fields 不进 actor input。

没有修改为 BC demonstration 的 `scan[::4]`，没有插值、重复 beam、新 range clip 或用 0 替换 invalid beam。

## 10. Real Austin F110 contract smoke

本地 Austin map 资源可用，已运行一次不含训练的真实 contract smoke：

```text
real legacy F110 reset(poses):      PASS
initial actor observation shape:    (1, 361)
one real simulator step:            PASS
fixed Lattice Planner opponent:     PASS
timeout terminal observation:       PASS
DummyVecEnv asynchronous auto-reset: PASS
auto-reset count:                   1
all real reset pose shapes:         (2, 3)
```

## 11. Reward scope

本次没有添加 reward shaping，wrapper 原样返回 base reward。opponent-only collision 不产生 penalty、不结束 ego episode、不进 collision KPI。未为 opponent 计算 policy log probability、value、advantage、imitation loss 或 optimizer update。

## 12. Test execution

```bash
/home/haowei/miniconda3/envs/end2race/bin/python -m unittest tests.test_sb3_gru_integration -v
```

```text
11 tests
Ran in about 3.4s
OK
```

zero-learning-rate stock train API smoke：

```text
parallel envs:        2
rollouts:             1
PPO train calls:      1
learning rate:        0.0
max parameter delta:  0.0
```

actor-only checkpoint 仍为 12 key，temporary export 后新 `End2Race(...).load_state_dict(..., strict=True)` 通过，round-trip max error `0.0`。

## Owner handoff

当前状态允许 owner 审阅后运行一次极小非零学习率 single-update gradient smoke。仍不允许由本报告推断 collision/overtake 性能，也不应在 reward、checkpoint retention 和正式训练配置完成独立审查前启动长时 learner。

未提交 Git commit，等待 owner 审阅。
