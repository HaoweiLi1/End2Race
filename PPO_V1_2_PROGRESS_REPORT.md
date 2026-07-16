# PPO V1.2 当前实验结果（停止快照）

快照时间：2026-07-16 09:01:11（Asia/Singapore）  
实验代码 HEAD：`f272f7301e6275bd87775fa3bdcee09ee7e6cc26`  
当前状态：**已按用户要求终止全部实验，不再续跑**。

## 1. 完成范围

| 项目 | 状态 | 结果 |
|---|---|---|
| Prompt 1：V1.2 实验基础设施 | 完成 | 125-arm 注册表、4 种 critic、runner、审计器、测试与报告已实现 |
| Prompt 2：hard-pool 构建 | 完成 | 7 个 pool 全部生成并通过完成门 |
| Prompt 3：正式 sweep | 部分完成后停止 | 5/125 arm 正式完成：C 阶段 4/4，H 阶段 1/48 |
| Prompt 4：只读审计 | 未执行 | 正式 sweep 未完成，因此未运行最终全局审计 |

停止时 manifest 中另有 `H-H0_CURRENT_DET-p25-bc` 标记为 RUNNING/attempt 2，但 attempt 1 与 attempt 2 都没有形成检查点评估结果，故不计入任何实验结论。停止标记见 `runs/ppo_v1_2/EXPERIMENT_STOPPED.json`。

## 2. Prompt 1：实现与验证

- 实现提交：
  - `fd9b2531484fbf0d230d18a54f92e320097931b8` — Simple PPO V1.2 实验基础设施。
  - `f272f7301e6275bd87775fa3bdcee09ee7e6cc26` — hard-pool 可恢复进度记录。
- 测试：41/41 通过（原有 27 项 + V1.2 新增项）。
- critic smoke：8/8 通过；mini preflight：10/10 通过。
- 关键报告：`IMPLEMENTATION_REPORT.md`。

## 3. Prompt 2：hard-pool 结果

- 候选 10,800；有效 10,764；无效 36（均为显式初始碰撞）。
- H1 outcome：overtake 6,113；follow 4,168；ego_collision 482；opponent_only_collision 58；error 1。
- H2 pass 1：43,056/43,056；H2 pass 2：3,696/3,696。
- 已知单场景异常：`v12-sp026-ego0551-raceline2-i08-v045`，`ZeroDivisionError: division by zero`；pass 1 的 4 个随机种子均记录为 error。该异常未被静默吞掉。

| pool | 数量 | SHA-256 |
|---|---:|---|
| H0_CURRENT_DET | 24 | `d022150627ff66af61e0b600699588181d8b06edf78a5723cf9ee503f89edcc9` |
| H1_EXPANDED_DET | 482 | `b760e0ed95bef1694db1955502fdef5b64555080223c6a768db12e4105541df2` |
| H2_STOCH_CORE | 685 | `97b20441984a2305ebbd5b0a2771786029d83b39474c94dbeef9db4d64110a8b` |
| H2_STOCH_BOUNDARY | 239 | `0e163dccfbd1df60f1717988f2628afe91d44fc9e293e10d86df90d6e879ff81` |
| H2_STOCH_ALL | 924 | `e3b137e5e23e6d100d8387b17c2566b74da07f713fc3dac82b4fb2c467d5ff00` |
| H3_UNION_CORE | 810 | `5705d1874930123555db53076bdc174a3f6e0b6cc597066b8096dd187f20398a` |
| H3_UNION_ALL | 1,029 | `594cc391b80751f35902c17d7f39ad4e24e98122cc5868a1ccd82bdf8cf3e807` |

完成门：`runs/ppo_v1_2/hard_pools/HARD_POOL_COMPLETION.json` 为 PASS。

## 4. Prompt 3：已完成正式结果

配对 BC 基线固定为：21 ego_collision / 233 follow / 346 overtake / 0 error（总数 600；另有 opponent_only_collision 3）。

### Stage C（4/4 完成）

所有 arm 均为 attempt 1，完成 8/8 updates、128/128 optimizer steps，600-case 检查点合法且 validation PASS。

| rank | arm | 选中 update | ego_collision | follow | overtake | error | actor SHA-256 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | C-C0_RAW_SINGLE_FRAME | 2 | 15 | 232 | 353 | 0 | `0aaba3ba7a58cea70afd2ac5a6428666fef2401f72d4f0e730a9e6115880ddfd` |
| 2 | C-C3_PRIVILEGED_PHYSICAL | 4 | 20 | 231 | 349 | 0 | `195cb6f362dfee25a1e46c8ff862eb90efe21417f871afe5a7f4682f25360526` |
| 3 | C-C2_DETACHED_ACTOR_HIDDEN | 2 | 21 | 232 | 347 | 0 | `d6350fe3fef323ce0521ffd5889887cae5fdee80dcbfbcccfd1a6f352a82adef` |
| 4 | C-C1_FROZEN_BC_FEATURE | 2 | 22 | 231 | 347 | 0 | `1d9c44c1e905db2d77b9eadbc486ce6328b2b4e2cd60951e1158e16b48812cba` |

C 阶段按预注册 selector 选出前 2 名：`C-C0_RAW_SINGLE_FRAME` 与 `C-C3_PRIVILEGED_PHYSICAL`。其中 C0 相对 BC 为碰撞 -6、超车 +7，是当前已完成结果中的最佳点。

### Stage H（1/48 完成）

| arm | attempt | 选中 update | ego_collision | follow | overtake | error | optimizer steps | actor SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| H-H0_CURRENT_DET-p25-wr | 1 | 2 | 18 | 237 | 345 | 0 | 128/128 | `f8343c2080eac4fd8da55e0698b838b1210f837d3fff670f46d0109de0681f6f` |

该 H arm 相对 BC 碰撞 -3、超车 -1；它是合法的阶段内候选，但 H 阶段未闭合，不能据此给出 H 阶段最终排名。

## 5. 结论边界

- 当前可确认的最佳已完成点是 C0 update 2：15/232/353/0。
- 不能宣称 Prompt 3、全局 125-arm sweep 或 PPO V1.2 最终实验已完成。
- H/B/R/K/E/G/W/X/S 的阶段屏障均未完成；不存在合法的全局选择结果。
- Prompt 4 最终审计未运行，不存在最终 `AUDIT_REPORT.md` 或全局 PASS 结论。
- 所有未完成 attempt 均不参与上述表格、排名或改进判断。

## 6. Git 审阅范围

`.gitignore` 已改为放出 V1.2 轻量证据：约 240 个文件、23.93 MiB。继续忽略：

- 所有 `runs/ppo_v1_2/**/checkpoints/` 及常见模型扩展名；
- 6 个单文件超过 1 MiB 的 hard-pool 原始/中间文件；
- stale `SWEEP.lock` 与空的 transient runner 日志。

轻量证据包含 resolved configs、训练指标、逐检查点评估、小型日志、stage 排名/选择、hard-pool manifests 与完成门。模型文件仍保留在本地，但不会进入 Git。
