# Simple PPO V1.1 实验报告

## 结论

三阶段均从 canonical BC fresh start，未复用 V1 update 20，未 resume，未启动第二 seed。V1.1 pilot 的 selection 结果是 **update 2**：15 个 ego collision、353 个 overtake；相对同一 paired BC（21/346）为 -6 collision、+7 overtake，也优于 V1 best update 5（17/347）。

但主要实验假设只得到部分支持：V1.1 确实显著增加了训练 episode 和真实 collision events，overtake 在全部 checkpoint 保持；collision 却没有随训练下降，也没有显示 checkpoint 波动被实质压低。update 2 之后 collision 反弹，update 15 达到 29，update 20 回落到 19。

## Fresh-start 与验收

- canonical BC：`pretrained/end2race.pth`，SHA256 `b5a1360fee18c2875185a3d23ab21cbdd8a4cdb2e94639433a148f34809ac5e4`。
- 三个 run 的 `train_bc_outcomes.json` SHA256 均为 `3fc7e4fd5effb6e98c1cbbec1c339416c1f138cdb6557d51bc077aebe4b59aa2`，与 V1 paired BC 文件一致。
- V1.1 pilot resolved contract：16 envs、1,600 steps、batch 1,600、collision sampling 0.50、20 updates、25,600 transitions/update、16 minibatches/update、320 optimizer steps，总计 512,000 transitions。
- 评价点严格为 2、3、5、10、15、20；每点均 600 cases、0 error、actor-only 12 keys strict load 通过。
- update 20 完整 checkpoint CPU reload 通过，`num_timesteps=512000`，三组 LR 为 `1e-6 / 1e-5 / 3e-4`。
- 指定回归：27 tests，全部通过。

### 阶段 1：zero-LR smoke

`runs/ppo_v1/v1_1_zero_lr_smoke`

- 25,600 valid transitions，16 recurrent minibatches，16 次 optimizer step；三组 LR 全为 0。
- GRU/head/critic 及所有冻结组 parameter delta 均精确为 0。
- rollout observations/actions/rewards/advantages/returns/log_probs/values 全部 finite。
- recurrent replay：25,600 valid、15,130 padded、ratio 精确为 1、最大 log-prob replay error 为 0。
- 完整 checkpoint reload 与 12-key actor strict load 均通过。

### 阶段 2：two-update nonzero smoke

`runs/ppo_v1/v1_1_nonzero_smoke`

- 51,200 transitions、32 optimizer steps；每个 update 16 steps，stock train 内三组 LR 恒为 `1e-6 / 1e-5 / 3e-4`。
- 最大绝对 delta：GRU `1.72183e-5`、head `2.13881e-4`、critic `6.11298e-3`，均正且 finite。
- perception/k、speed_mlp、dummy_embedding、log_std delta 均精确为 0。
- ratio、KL、clip fraction、policy/value loss 全部 finite；完整 checkpoint reload 与 12-key strict load 通过。

## Paired deterministic evaluation

所有行使用同一 600-case development panel。overtake selection floor 为 `ceil(0.95 * 346) = 329`。

|Checkpoint|Collision|Follow|Overtake|Opponent-only|Fixed|New|Gained OT|Lost OT|Error|
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|Paired BC|21|233|346|3|0|0|0|0|0|
|V1 best U5|17|236|347|3|11|7|9|8|0|
|V1.1 U2 (selected)|15|232|353|3|10|4|10|3|0|
|V1.1 U3|23|234|343|3|6|8|7|10|0|
|V1.1 U5|23|231|346|3|5|7|7|7|0|
|V1.1 U10|23|231|346|3|9|11|10|10|0|
|V1.1 U15|29|227|344|3|5|13|9|11|0|
|V1.1 U20|19|225|356|3|10|8|16|6|0|

Selection 复算：全部 V1.1 checkpoint 均超过 overtake floor；最少 collision 是 update 2 的 15，因此选择 update 2。它相对 V1 best update 5 再减少 2 collision，并增加 6 overtake。

## 训练覆盖与 episode/reset 统计

V1.1 共 734 completed episodes、207 ego-collision episodes、231 follow、296 overtake；其中 `bc_ego_collision` branch 贡献 347 completed episodes 和 193 collision events。V1 原 pilot 为 326 completed episodes、35 collision events。因此 V1.1 在两倍 transitions 下获得约 2.25 倍 completed episodes 和 5.91 倍真实 collision events。

`all/bc` 分别表示 `all_training / bc_ego_collision`。

|U|Completed all/bc|Collision all/bc|Follow all/bc|Overtake all/bc|Resets all/bc|Unique|Partial in/out|
|-:|:-|:-|:-|:-|:-|-:|:-|
|1|18/15|0/7|9/4|9/4|23/26|34|0/5|
|2|11/25|2/13|4/4|5/8|19/17|34|5/13|
|3|20/13|0/3|7/6|13/4|17/16|40|13/14|
|4|18/19|2/11|9/3|7/5|18/19|41|14/16|
|5|15/23|0/11|3/7|12/5|16/22|41|16/16|
|6|14/23|0/15|7/2|7/6|12/25|39|16/16|
|7|19/14|0/4|8/5|11/5|19/14|43|16/16|
|8|17/18|0/12|10/4|7/2|17/18|39|16/16|
|9|22/14|1/6|11/4|10/4|24/12|46|16/16|
|10|27/12|2/10|14/1|11/1|25/14|46|16/16|
|11|19/21|0/15|10/5|9/1|24/16|44|16/16|
|12|27/9|2/4|14/3|11/2|25/11|46|16/16|
|13|21/16|0/11|5/4|16/1|17/20|42|16/16|
|14|16/21|2/10|10/4|4/7|20/17|46|16/16|
|15|20/14|0/7|7/1|13/6|17/17|44|16/16|
|16|21/17|1/10|4/4|16/3|24/14|44|16/16|
|17|23/15|0/9|10/2|13/4|18/20|46|16/16|
|18|14/23|0/10|6/5|8/8|17/20|39|16/16|
|19|20/18|0/10|2/2|18/6|24/14|49|16/16|
|20|25/17|2/15|10/1|13/1|21/21|47|16/16|

## 每 update 优化统计

下表 delta 均相对 fresh BC；冻结组每 update 仍精确为 0。每行三组 LR 均为 `GRU=1e-6, head=1e-5, critic=3e-4`，每行 16 optimizer steps。完整 transition-level branch counts、reset/episode branch maps 和 action min/max 位于 `training_metrics.jsonl`。

|U|Eps|Coll all/bc|PG loss|V loss|EV|KL|Clip|Speed mean±sd|Steer mean±sd|GRU Δ|Head Δ|Critic Δ|
|-:|-:|:-|-:|-:|-:|-:|-:|:-|:-|-:|-:|-:|
|1|33|0/7|0.00365142|0.179807|-0.3108|0.00248951|0.1276|5.368±1.320|-0.0122±0.0838|1.14e-5|1.21e-4|0.00378|
|2|36|2/13|8.83006e-5|0.230418|0.0036|0.000940959|0.0361|5.153±1.353|-0.0151±0.0871|1.72e-5|2.14e-4|0.00611|
|3|33|0/3|0.000728278|0.0702968|0.0852|0.000745554|0.0245|5.532±1.307|-0.0104±0.0851|2.29e-5|2.76e-4|0.00864|
|4|37|2/11|0.00233073|0.157347|0.0670|0.00214675|0.1201|5.346±1.231|-0.0099±0.0857|2.70e-5|2.56e-4|0.0104|
|5|38|0/11|0.000584163|0.125999|0.1231|0.00354220|0.0129|5.479±1.366|-0.0168±0.0921|2.99e-5|2.85e-4|0.0103|
|6|37|0/15|0.00137003|0.174977|0.1635|0.127059|0.0582|5.189±1.326|-0.0157±0.0913|3.24e-5|3.04e-4|0.0121|
|7|33|0/4|0.000501765|0.0572742|-0.1446|0.000731937|0.0211|5.311±1.293|-0.0081±0.0910|3.49e-5|3.25e-4|0.0141|
|8|35|0/12|0.000861338|0.123003|0.1039|0.000643847|0.0216|5.208±1.321|-0.0110±0.0857|3.76e-5|4.01e-4|0.0164|
|9|36|1/6|0.00263374|0.0762105|0.0807|0.00251164|0.1396|5.402±1.328|-0.0105±0.0798|4.15e-5|4.33e-4|0.0165|
|10|39|2/10|0.00125374|0.118796|0.1676|0.00558402|0.0685|5.223±1.316|-0.0088±0.0748|4.51e-5|4.47e-4|0.0167|
|11|40|0/15|0.000160889|0.142268|0.2911|0.000566964|0.0175|5.421±1.166|-0.0174±0.0823|4.80e-5|4.60e-4|0.0181|
|12|36|2/4|0.000140333|0.0811367|0.3329|0.000670696|0.0202|5.438±1.241|-0.0036±0.0758|5.01e-5|5.01e-4|0.0189|
|13|37|0/11|0.00105678|0.124367|0.2045|0.00217868|0.0294|5.333±1.435|-0.0151±0.0912|5.17e-5|5.01e-4|0.0199|
|14|37|2/10|0.000725058|0.117238|0.2786|0.000563718|0.0144|5.391±1.340|-0.0104±0.0833|5.76e-5|5.02e-4|0.0191|
|15|34|0/7|0.000820619|0.0891706|0.2376|0.000519997|0.0169|5.716±1.216|-0.0151±0.0858|5.87e-5|5.29e-4|0.0192|
|16|38|1/10|0.00109099|0.128004|0.3145|0.000750409|0.0245|5.492±1.353|-0.0083±0.0885|5.88e-5|5.38e-4|0.0196|
|17|38|0/9|0.000356786|0.100334|0.3214|0.000647263|0.0221|5.323±1.460|-0.0207±0.0854|5.77e-5|5.54e-4|0.0196|
|18|37|0/10|0.000447551|0.111687|0.2843|0.000735386|0.0248|5.412±1.429|-0.0055±0.0853|6.54e-5|6.27e-4|0.0197|
|19|38|0/10|0.000862403|0.115598|0.4098|0.00126904|0.0616|5.781±1.199|-0.0127±0.0823|6.69e-5|6.66e-4|0.0205|
|20|42|2/15|0.00100401|0.186986|0.2261|0.243824|0.0530|5.280±1.363|-0.0176±0.0887|7.04e-5|6.98e-4|0.0215|

所有 policy/value loss、EV、KL、clip fraction 和 action statistics 都是 finite。KL 中位数为 `8.45684e-4`，但 update 6 (`0.127059`) 和 update 20 (`0.243824`) 有明显尖峰；本实验按合同没有 target-KL gate，因此没有提前停止或更改 PPO。

## A–E verdict

**A. collision 是否随训练下降：否。** V1.1 checkpoint collision 序列为 `15, 23, 23, 23, 29, 19`（updates `2,3,5,10,15,20`），线性斜率约 `+0.189 collision/update`。final 虽优于 BC 的 21，但不优于早期 update 2 的 15。

**B. fixed 是否稳定大于 new：否。** fixed/new 依次为 `10/4, 6/8, 5/7, 9/11, 5/13, 10/8`；仅 update 2 和 20 为 fixed > new。

**C. overtake 是否保持：是。** 六个 checkpoint 全部超过 329 floor，最小 343（BC 的 99.1%），selected update 2 为 353，final update 20 为 356。

**D. critic explained variance 是否改善：是，但非单调。** update 1 为 `-0.3108`，update 20 为 `0.2261`；前 5 updates 均值 `-0.0064`，后 5 均值 `0.3112`，线性斜率 `+0.02295/update`。中间仍有 update 7 的 `-0.1446` 回落。

**E. 增加数据是否减少 checkpoint 波动：没有充分证据。** 在共同评价点 5/10/15/20 上，collision 的 V1/V1.1 population std 为 `3.90/3.57`，range 均为 10，mean absolute adjacent change 均为 `5.33`；因此 collision 波动没有实质降低。overtake 波动有所收窄：std `5.85 -> 4.69`、range `16 -> 12`、adjacent change `7.00 -> 4.67`。综合 verdict 为：overtake 更稳定，但主要 collision checkpoint 波动未改善。

## Evidence

- `resolved_config.json`：最终 V1.1 合同。
- `training_metrics.jsonl`：每 update optimizer、loss、EV、KL、action、LR、parameter delta 及完整 rollout episode/reset/branch 统计。
- `episodes.jsonl`：逐 completed episode 证据。
- `evaluations/update_*_rows.json`：逐 scenario paired deterministic 结果。
- `selection.json`：selection rule 输出。
- `experiment_acceptance.json`：机器可读验收与 V1/V1.1 对比摘要。

实验按 update 20 和全部评价完成后停止；未启动第二 seed，未修改 reward，未延长训练，未添加模型组件或 custom PPO loss。
