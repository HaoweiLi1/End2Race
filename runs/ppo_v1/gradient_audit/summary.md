# PPO V1 gradient audit

## 审计边界

- Git HEAD：`fd22c962a4fd62681a9a4fb53e14f277ef8f3418`
- BC actor：`pretrained/end2race.pth`，SHA-256 `b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4`；fresh model 的 12 个 actor tensors 全部 bitwise equal。
- stock 版本：stable-baselines3 `2.7.1`，sb3-contrib `2.7.1`。
- 正式 rollout：16 env × 800 step = 12,800 transitions；只收集 1 个 rollout；optimizer.step = 0；未保存 checkpoint。
- stock minibatch：16 × 800 valid transitions；padding 总数 8,227。

## Verdict

1. **Actor/critic 是否存在错误 direct gradient leakage：否。** policy-only backward 的 critic norm 最大值为 `0`；`vf_coef * value_loss` backward 的 GRU/head norm 最大值分别为 `0` / `0`。
2. **Critic 是否显著主导 global gradient norm：否。** critic 的 squared global-norm fraction 为 0.000146413–0.112026（median 0.00799835）。
3. **Stock global clipping 是否显著压小 actor gradient：是；但 critic 是否是主要原因：否。** combined 理论 multiplier 为 0.00685217–0.0992229（median 0.0252965），即 actor norm 缩小 90.0777–99.3148（median 97.4704）%。与 actor-only counterfactual 相比，critic 额外造成的 actor norm 缩小仅为 0.00732092–5.76763（median 0.400729） 个百分点；GRU/head 使用同一个 global multiplier。
4. **Critic Tanh 是否明显饱和：是。** `abs(preactivation)>3` 比例为 `21.31%`；Tanh `abs(value)>0.99` 比例为 `26.96%`；preactivation `max_abs=11.2988`。
5. **batch_size=800 的实际 recurrent 几何：** 每 minibatch `n_seq` 为 2–3（median 2），涉及 unique episodes 为 2–3（median 2）；按 800-step formal horizon 折算均为 `1.000` episode-equivalent。详细 sequence 长度和 outcome 关联见 `minibatches.jsonl`。
6. **V1.1 前是否有阻断性问题：有。** critic 第一层 Tanh 明显饱和。本审计不据此修改参数、网络、reward 或 clipping，也未启动训练。

## Critic 数值证据

- raw LiDAR：mean `2.70388`，std `3.43542`，min/max `0.215397` / `30.0485`。
- previous speed：mean `5.47669`，std `1.54906`，min/max `0` / `7.67121`。
- first-layer preactivation：mean `0.107358`，std `2.4709`。
- critic prediction：mean `0.049307`，std `0.268654`；returns mean `0.0887235`，std `0.288516`。
- explained variance：`-0.314583`（stock rollout values：`-0.314583`）。

## Rollout episode evidence

- completed episodes：`16`；outcomes：`{"ego_collision": 2, "follow": 5, "overtake": 9}`。
- sampler branch transitions：`{"all_training": 9437, "bc_ego_collision": 3363}`。
- 每个 minibatch 的 collision/follow/overtake 只统计能通过本 rollout 的 info 与 episode key 关联到完整 outcome 的 episode；未完成 episode 单独计为 unclassified。

## 完整性证明

- 所有 parameter tensors bitwise unchanged：`True`。
- parameter SHA-256 before/after：`d392ba1bc5d7a7ed680ed75564423c37ada89bf5867b4d8353a643a83e30813d` / `d392ba1bc5d7a7ed680ed75564423c37ada89bf5867b4d8353a643a83e30813d`。
- 所有 gradients 在每次 backward 后清空：`True`。
- 所有 rollout transitions 恰好进入一个 stock minibatch：`True`。
