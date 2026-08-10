# Collision-only BC Functional Regularization 与 Constrained PPO 最终训练计划

状态：**已完成；方法A/B训练、固定四图各600与最终审计均结束，两条固定实例都未形成新前沿并已关闭**

日期：2026-08-10（Asia/Singapore）
适用仓库：`/home/haowei/Documents/End2Race`

## 0. 技术摘要

本计划只保留两条尚未得到正式端点模型、但作用机制不同的候选：

1. **方法 A：Single-stage PPO with collision-only hindsight-selected counterfactual BC functional regularization**；
2. **方法 B：Calibrated Lagrangian Constrained PPO with a separated collision cost**。

两条方法都从canonical BC fresh start，只在Austin训练，最终部署actor仍是361D输入、原End2Race
结构和strict 12-key checkpoint。训练期teacher、anchor dataset、cost critic和dual变量全部不进入
评测或部署。

执行顺序固定为A训练与评测完成后运行B，两个方法都不再设置研究性训练前Gate或再次授权暂停点。
只有会使训练产物无效的机械错误才停止；模型效果由固定四图各600直接判定。

今后所有新actor只在以下四张固定面板评测：

- Austin：600 episode；
- Hockenheim：600 episode；
- MoscowRaceway：600 episode；
- Nuerburgring：600 episode。

全部使用CUDA、deterministic mean action、ego collision scope并保存numeric trace。不运行near400、
hard73、noise、single-agent、其他startpoint、失败子集或额外地图eval。训练侧Gate、反事实branch和
rollout审计只属于机制/实现检查，不能写成模型性能。

## 1. 当前比较基线与“更好”的定义

四图指标统一写作`ego collision / overtake`。当前旧模型前沿为：

| 模型 | 四图结果 | 证据边界 | 当前角色 |
|---|---:|---|---|
| Canonical BC | `129 / 1445` | 正式CUDA四图 | 初始化与逐图最低线 |
| U44前向走廊时间相关速度探索 | `62 / 1478` | 正式CUDA四图配对 | 当前安全端基准 |
| ordinary异线高速重加权U30 | `73 / 1516` | 600-trace重建，CUDA metadata不完整 | 当前超车端候选，正式比较前需固定四图复核 |
| Production U30 | `94 / 1508` | Austin/Hockenheim已有正式结果；其余旧包不完整 | 当前部署模型，不作为放宽U44/RW前沿的理由 |

本计划不再把`collision < 40且overtake > 1500`作为唯一成败线。固定采用三层判决：

### 1.1 全面突破旧前沿

主checkpoint同时满足：

```text
collision < 62
overtake > 1516
```

这表示同时严格优于U44的安全端和重加权U30的超车端。

### 1.2 有效的新前沿模型

候选必须同时满足：

1. 每张地图collision不高于canonical BC、overtake不低于canonical BC；
2. 严格Pareto支配U44或正式复核后的重加权U30：
   - `collision <= baseline collision`；
   - `overtake >= baseline overtake`；
   - 两者至少一项严格改善；
3. 不被另一条旧前沿模型Pareto支配；
4. 主checkpoint、场景集合和评测协议均预先固定，不从band中事后挑模型。

计数承担产品候选判定；配对exact McNemar结果承担证据强度判定。严格改善轴`p < 0.05`时称为
“配对确认的新前沿”；计数通过但`p >= 0.05`时只能称为“方向性候选”，由用户决定是否切换，
不能自动替换production。

### 1.3 机制有变化但没有产生新前沿

若碰撞下降但超车下降，或超车增加但碰撞增加，必须记录为安全—超车交易。它可以帮助理解方法，
但不算本计划的成功。

## 2. 共同执行边界

两条正式训练共同固定：

- canonical BC：`pretrained/end2race.pth`；
- 训练地图：Austin；
- seed：42；
- actor输入：361D；
- critic输入：按各方法的训练合同；
- `n_envs=16`、`n_steps=6400`，每个rollout 102,400 transition；
- `batch_size=12800`；
- actor/critic epochs：`2/5`；
- GRU/head/reward-critic LR：`3e-6/3e-5/3e-4`；
- steering latent std / speed physical std：`0.03/0.15`；
- `gamma=0.999`、`GAE lambda=0.995`、`clip_range=0.20`；
- canonical BC Austin collision pool 479和ordinary pool 600；
- 8 collision-role + 8 ordinary-role逻辑环境；
- 不使用checkpoint热启动、二阶段repair、模型选择器或multi-map训练；
- 不扫描beta、cost budget、dual LR、窗口、teacher、学习率、更新数或评测panel；
- run目录必须全新，任何中断不得写回原目录或冒充resume。

评测按checkpoint串行执行；每次`evaluate.sh`仍由当前入口内部使用固定worker数，不并行启动多个
checkpoint级评测任务。

## 3. 执行前的一次性旧基线闭合

为使最终比较都来自同一固定四图600协议，正式产品判决前完成一次、且只完成一次：

1. 重加权U30在四张地图各600的fresh CUDA deterministic复核，共2,400 episode；
2. Production U30只补齐当前缺少规范包的MoscowRaceway600与Nuerburgring600，共1,200 episode；
3. U44与BC已有合格四图正式包，不重复运行。

合计3,600 episode。这些结果只闭合旧基线证据，不参与方法参数、anchor选择、cost参数或checkpoint
选择。若用户不授权这3,600次基线闭合，新模型仍可相对U44作正式判决，但与RW/production的比较
必须继续标注证据级别不完全一致，不能声称全面支配旧前沿。

## 4. 方法 A：Collision-only BC Functional Regularization

### 4.1 要检验的命题

U44已经在四图形成较安全的策略，但仍有一组相对canonical BC新造的collision。Z7说明canonical
BC在U44已经进入不利状态后，不是完全无用：在41条稳定collision source中，固定1.5秒闭环BC动作
救回18条，18条全部最终overtake；41条严格匹配safe controls中，同样的强BC接管新造4次collision
并损失4次overtake。

本方法不在部署时接管动作，而是在PPO训练期用18条已验证成功的反事实序列约束actor不要遗忘这些
BC安全动作。待检验命题是：**弱、选择性的functional regularization能保留BC对这些失败状态的
局部安全行为，同时由PPO继续承担自然状态上的动作选择。**

### 4.2 准确命名与方法边界

准确名称：

```text
Single-stage PPO with collision-only hindsight-selected
counterfactual BC functional regularization
```

它不是纯PPO，因为actor loss包含第二项模仿式functional loss。它仍满足：

- 只有一条fresh student RL轨迹；
- 从canonical BC初始化；
- 没有U30/U44热启动；
- 没有二阶段repair；
- 只用Austin训练侧数据；
- 最终actor结构和12-key checkpoint不变；
- teacher和anchor dataset部署时全部删除；
- formal rollout和部署不读取未来collision信息。

必须同时承认：训练anchor由历史U44未来结局与碰撞窗口离线筛选，所以整个训练设计使用了
historical hindsight，不能写成“训练完全不使用未来信息”。

### 4.3 “完整BC接管”在本方法中的作用

Z7的完整BC接管只是构造训练标签的反事实branch：

```text
U44运行到原碰撞前1.5秒
-> 最后150个动作步的steering和speed全部由canonical BC闭环输出
-> 若episode继续，窗口后恢复U44用于判定最终outcome
```

“完整”表示steering和speed同时由BC输出；BC每步读取干预后产生的新observation，不是把另一条
轨迹的动作硬贴到U44 observation。该branch本身不可部署，也不是最终模型eval。

### 4.4 固定anchor dataset

只使用Z7中满足以下全部条件的18条source：

1. branch0精确复现U44并以ego collision结束；
2. full-BC branch无ego collision；
3. full-BC branch最终overtake；
4. 150个计划干预步全部实际执行。

现有机器结果已核实：18/18窗口均为150步，覆盖11个ego startpoint，opponent raceline0/2为
`16/2`，speed scale `0.50--0.85`。这个覆盖明显偏向raceline0，必须作为泛化风险记录，不能通过
重采样、复制raceline2或加入未救回source来“平衡”。

每条训练sequence固定保存：

- scenario identity和episode-start；
- 从episode起点到干预结束的361D actor-visible observation；
- 干预开始位置和150步anchor mask；
- canonical BC在同一反事实sequence上的latent steering mean与physical speed mean；
- branch0/full-BC最终outcome和碰撞类型，仅用于审计；
- branch0精确重放合同。

前缀只用于从零hidden递归burn-in，不计anchor loss。干预窗口使用真实full-BC branch产生的后续
observation，禁止把BC动作监督到原U44碰撞轨迹的后续observation。student hidden每次更新时由
当前student重新递归，不能预先缓存U44 hidden。

这18条是训练输入，必须保存在独立、不可覆盖的panel目录。四图600结果、near/hard结果和Z7的
41条matched controls都不进入anchor dataset；controls只保留为方法风险证据。

### 4.5 Actor loss

正式actor loss固定为：

```text
L_actor = L_PPO + beta_anchor * L_collision_anchor
```

现有actor的steering使用latent Gaussian、speed使用physical Gaussian，std分别为`.03/.15`。
Anchor loss使用同方差Gaussian mean-KL：

```text
L_collision_anchor
  = mean_episode(
      mean_150_steps(
        0.5 * ((steer_latent_student - steer_latent_BC) / 0.03)^2
      + 0.5 * ((speed_student - speed_BC) / 0.15)^2
      )
    )
```

steering latent转换必须调用当前action distribution已有实现，不另写近似atanh。先在每条episode的
150步内求平均，再对18条episode求平均，避免序列长度或padding改变权重。

每个PPO actor minibatch固定配全部18条anchor sequence；不shuffle、不按当前loss采样、不读取训练
outcome。数据量很小、每update重复使用是本方法的真实限制，不能用随机增强或新teacher掩盖。

Anchor loss同时更新GRU与actor output head；critic、reward和exploration不读取该loss。

### 4.6 Beta只标定一次

canonical BC初始化时student与teacher相同，anchor loss接近零，不能在初始化点定beta。固定使用
U44 actor作为已漂移student代理，并使用U44同一训练轨迹的一个102,400-transition Austin rollout，
只计算第一个actor epoch的8个minibatch梯度，不执行optimizer step。

学习率加权step-space范数：

```text
||g||_eta = sqrt((3e-6)^2 * ||g_GRU||^2 + (3e-5)^2 * ||g_head||^2)

beta_anchor
  = 0.25 * median_i(
      ||g_PPO_i||_eta / max(||g_anchor_i||_eta, 1e-12)
    )
```

beta只计算一次并写入正式run config，不做候选网格、不根据四图结果改变。`0.25`表示在U44代理点，
anchor目标step-space贡献约为PPO actor step的25%，不是成功概率。

### 4.7 开跑前只保留一次beta标定

Z7行为结果已经固定为18 rescue / 4 harm，18条、每条150步的anchor dataset已经构建。正式训练前
唯一需要从数据生成的训练参数是`beta_anchor`。固定U44 rollout与§4.6公式得到：

```text
beta_anchor = 0.006405998602049812
```

该rollout不产生候选模型，也不承担方法有效性判决，只负责把PPO与anchor更新尺度固定下来。数据
finite、canonical BC动作对齐、anchor gradient非零和strict 12-key属于最小机械合同；shadow
minibatch方向、动作分量占比、`beta=0`位级对照等研究性检查不再作为训练前Gate，也不触发重跑或
参数调整。方法是否有效只由45-update训练后的固定四图600评测判定。

### 4.8 正式训练合同

基线是产生U44的前向走廊时间相关速度探索45-update轨迹；treatment相对它唯一新增
`beta_anchor * L_collision_anchor`。

| 项目 | 固定值 |
|---|---|
| 实验ID | `ppo_corridor_temporal_collision_bc_anchor_v1` |
| 初始化 | canonical BC |
| map / seed | Austin / 42 |
| formal updates | 45 |
| speed exploration | `corridor_temporal`，std `.15`，hold 50，2m前向走廊 |
| PPO参数 | §2共同合同 |
| 唯一treatment | fixed beta的collision-only anchor loss |
| actor checkpoints | 每个formal update保存；最终strict 12-key |
| critic | 原`privilege_gru`，部署删除 |

正式命令在beta固定后写入根目录`run.sh`，并且`run.sh`只保留这一条显式命令。计划中的
新CLI最小为`--collision_bc_anchor_dataset`与`--collision_bc_anchor_beta`；实际名称在实现后必须与
`--help`和`run_config.json`一致，不得同时增加通用teacher框架。

训练中每update记录：PPO actor loss、anchor总loss及两动作分量、PPO/anchor/combined gradient
norm、实际beta、clip前后norm、actor KL、anchor functional drift、16/16 actor step、critic loss、
episode结果和12-key checkpoint。不得用训练episode结果早停或选择checkpoint。

立即停止条件仅限：非finite、beta漂移、anchor数据身份变化、optimizer step数错误、checkpoint
不完整或关闭路径合同被破坏。训练中碰撞率高、anchor loss不降或dual不存在都不是提前改参数的理由。

### 4.9 固定四图600评测

预先评估U42、U43、U44、U45；U44是唯一主checkpoint，其他三点只报告稳定性，不参与模型选择。
每个checkpoint运行四张固定地图各600，共9,600 episode。不得运行其他eval。

主报告必须包含：

- 每图与四图合计collision/overtake；
- ego-opp、ego-wall、opponent-wall分列；
- 相对BC、U44、正式复核RW和production的removed/created、lost/gained、exact paired p；
- 每包600 unique、0 error、600 trace、result/trace/panel key一致、数值finite、terminal与collision
  marker合同；
- U42--U45完整表，不只报最好单点。

方法A按§1判决。它的直接四图问题射程是20条off-line、BC-overtake的U44 created collision；若只
修复这些且没有其他影响，静态参照点为`42/1498`。这只是“20条目标全部翻转、其余episode完全
不变”假设下的作用范围算术，不是训练actor的数学上限、模型预测或`<40`承诺。Functional
regularization可能通过共享参数影响更多episode，因此理论上可以高于或低于该点；现有证据只足以
把主要成功预期放在§1.2新前沿，不能把§1.1写成有数据支持的预期。

### 4.10 方法A停止规则

1. 只运行一条45-update formal，不扫beta或训练长度；
2. 不从U42/U43/U45改选主模型；
3. 不因四图某张图失败而用该图反向修改Austin anchor；
4. 不新增lost-overtake anchor、same-line teacher、运行时BC fallback或二阶段repair；
5. 若U44主点没有形成§1.2的新前沿，关闭当前18-anchor实例；
6. 无论结果如何，先完整记录再决定是否授权方法B。

## 5. 方法 B：校准后的Constrained PPO

### 5.1 要检验的命题

标准PPO让同一个reward critic同时拟合密集progress与一次性`-2.0`collision尖峰。校准后的
Constrained PPO把同一collision事件从reward目标中移出，只作为独立cost，并用dual变量按明确的
训练侧budget调节安全压力。本轮只检验两条尚未被正式训练触及的机制：

1. **目标分离**：移除稀疏collision尖峰后，reward critic对剩余
   `progress + relative progress + risk` return的标准化held-out拟合是否改善；
2. **自适应安全课程**：cost credit在真正决定接触的晚期窗口是否跨startpoint可用，且由已证明
   可达的budget驱动时，dual能否从BC的高碰撞区向U44级安全区调节，而不是单向放大全局保守度。

必须准确区分三个角色：真实cost return沿采样轨迹提供策略梯度；cost critic是state-dependent
baseline与bootstrap，主要影响方差和延迟credit；dual提供自适应标量约束。Cost critic不是状态条件
安全信号的唯一来源，因此Z9的OOF失败不能数学推出Constrained PPO等价于固定碰撞罚分，但足以
阻止在没有新证据时直接支付正式训练预算。

### 5.2 旧Z9结论保持有效

现有102,400-transition Z9 preflight中：

- 153个完整episode、57个collision、85个ego startpoint；
- cost event与collision episode严格都是57；
- reward collision去重最大误差0；
- cost advantage / return std为`.27325/.25761`；
- collision rate `57/153=37.25%`使dual从`1.0`升至`1.13627`；
- 合成actor gradient相对reward-only的差分L2为31.1%，cosine `.96687`；
- P20 startpoint-OOF的MSE skill / episode-start AUROC / early AUROC为
  `.04038/.42855/.60703`，均未过`.05/.65/.65`。

因此机械链路成立，但当前P20 baseline没有获得预注册要求的跨起点提前排序能力。旧machine
verdict继续是`fail_stop_exact_constrained_implementation`；旧report、目录和停止事实不得修改为
pass。新方案不是`endpoint_formal` override，而是一个使用新机制判据与一次性budget公式的独立
预注册实例。

现存Z9目录没有保存102,400条transition、逐行P20 observation、cost return或OOF prediction，只有
聚合report、episode摘要和critic checkpoint。因此晚期窗口与reward分离复核必须fresh收集训练侧
rollout；不能写成“零新仿真历史重算”。

### 5.3 Formal算法边界

同一个first ego collision事件不能同时在reward和cost重复计价。Formal在reward GAE前逐transition
精确移除现有`-2.0`collision分量，只定义：

```text
cost_t = 1[first ego collision transition]
```

Reward critic保持`privilege_gru`。独立cost critic固定为训练期P20 MLP：

```text
20 -> 120 ReLU -> 30 ReLU -> 1
```

Cost GAE固定为`gamma_cost=.999`、`lambda_cost_GAE=.995`。Actor使用：

```text
A_actor = A_reward - lambda_cost * A_cost
```

只对合成后的advantage执行当前minibatch PPO归一化，再进入标准clipped surrogate。Formal dual固定：

```text
lambda <- clip(
  lambda + 0.5 * (completed_episode_collision_rate - d_calibrated),
  0,
  20
)
```

`lambda_0=1`、dual LR `.5`、cost critic LR `3e-4`、每update 5个cost critic epoch和上下界`[0,20]`
保持Z9不变。budget固定为§5.4的`.19`，不是CLI候选网格，也不根据训练中途或四图结果反向选择；
不得换recurrent cost critic。

Dual实际使用的是**所有已完成episode合并后的collision率**。16个逻辑slot在transition层为8个
collision role和8个ordinary role，但两类episode长度不同，完成episode数不保证50/50。正式记录
必须同时报告pooled rate、两role各自分子/分母和未完成episode数；不得再把pooled rate写成episode
层严格50/50，也不得把它等同于四图collision/2400。

### 5.4 开跑前固定budget，不再运行B0研究Gate

取消BC加两个U44 no-update rollout、paired reward-critic、晚期OOF和fold-gradient检查。它们只能
改变“是否值得训练”的先验，不能替代正式模型结果，因而不再消耗训练前预算。

直接使用已经记录的同协议U44训练分布结果`26/141 = 18.4%`，按1个百分点向上取整固定：

```text
d = ceil(100 * 26/141) / 100 = 0.19
```

该值只固定当前实验实例，不宣称是Constrained PPO的普适最优budget，也不得在训练中或评测后修改。
正式入口仅做最小机械保护：collision不能在reward/cost重复计价、cost与advantage必须finite、dual必须
在`[0,20]`、optimizer step和checkpoint必须完整、最终actor必须strict 12-key。这些保护只防止无效
训练产物，不对模型效果设置开跑门槛。

### 5.5 正式训练合同

| 项目 | 固定值 |
|---|---|
| 实验ID | `ppo_constrained_collision_cost_calibrated_v1` |
| 初始化 | canonical BC |
| map / seed | Austin / 42 |
| formal updates | 30 |
| exploration | production baseline independent Gaussian |
| PPO参数 | §2共同合同 |
| reward critic | `privilege_gru`，拟合collision-removed reward return |
| cost critic | 同构P20 MLP，formal fresh初始化 |
| cost / dual | §5.3；固定`d=.19` |
| 部署产物 | 原361D、strict 12-key actor；删除reward critic、cost critic和dual |

Formal入口必须验证canonical actor、pool和固定训练合同，不提供`--cost-budget`自由覆盖。当前源码
每个warm-up/formal cost-critic minibatch各只调用一次
`clip_grad_norm_`；旧计划中的“连续重复调用”描述已核对为过时，不再安排相关代码修复。

每update必须记录：completed episode/collision计数、pooled和两role collision rate、未完成episode、
lambda used/after、reward/cost/combined advantage尺度、reward去重误差、reward/cost value loss与
explained variance、两个critic gradient、actor PPO指标、16/16 actor step和三类checkpoint。

立即停止条件只包括：非finite、reward/cost重复计价、cost event与collision不一致、lambda越界、
optimizer step数错误、checkpoint不完整或actor不再strict 12-key。不得根据
中途rate修改budget、dual LR、cost critic、reward或训练长度。

### 5.6 固定四图600评测

预先评估U27、U28、U29、U30；U30是唯一主checkpoint，其他三点只报告稳定性。每个checkpoint
运行四张固定地图各600，共9,600 episode。不得运行其他eval。

评测时只加载actor：不加载reward critic、cost critic、dual或任何cost feature。报告合同与方法A
§4.9相同，并关联训练期lambda、pooled/role collision rate和两个critic fit解释端点；训练telemetry
不能替代四图结果。

方法B按§1判决。若只减少collision但降低overtake，记为约束引起的安全—性能交易，不算新前沿。

### 5.7 方法B停止规则

1. 只运行一条`P20 × d=.19 × lambda0=1 × dual_lr=.5`的30-update formal；
2. 不扫budget、dual、cost critic、reward、探索、训练长度或checkpoint；
3. 不从U27--U29改选主模型，也不根据四图结果改变Austin cost定义；
4. U30未形成§1.2新前沿即关闭该固定实例；
5. 失败不外推为所有Constrained PPO无效，也不自动提出recurrent cost critic或“安全reward +
   overtake约束”重试；后者是目标方向不同的新方法。

## 6. 实现与验证工作清单

### 6.1 方法A最小代码范围

预计只需要：

1. 一个窄用途dataset builder，从现有Z7 branch产物冻结18条anchor训练输入；
2. `train_ppo.py`增加dataset与fixed beta两个显式参数；
3. PPO actor update中加入collision-only anchor loss；
4. 一个对应的最小回归脚本，验证dataset、recurrent重建、loss和strict 12-key；
5. `EXPERIMENTS.md`记录CLI、schema、公式、断言和删除后失去的能力。

不增加通用teacher注册表、runtime selector、第二actor、callback框架或自动beta调节器。

### 6.2 方法B最小代码范围

预计只需要：

1. 保留现有`scripts/run_constrained_ppo.py`算法主体；
2. 新增固定`d=.19`的`direct_formal`入口，不提供自由budget参数；
3. 复用现有reward去重、cost GAE、P20 cost critic与dual更新；
4. 保留非finite、重复计价、dual边界、optimizer/checkpoint和strict 12-key机械保护；
5. `EXPERIMENTS.md`记录新模式、固定budget来源和旧Z9证据边界。

不把当前实现迁入production默认，不新增recurrent cost critic或参数网格。

## 7. 串行执行顺序

```text
构造18条anchor dataset并一次标定beta
  -> A fresh 45-update训练
  -> A U42--U45固定四图各600
  -> 固定d=.19并启动B fresh 30-update训练
  -> B U27--U30固定四图各600
  -> 最终比较与收口
```

任何一步失败都不自动切换下一方法。任何训练或eval启动前，根目录`run.sh`只放当前唯一显式命令；
完成、停止或放弃后移除该命令。现有用户工作树禁止reset、checkout或clean。

## 8. 产物与记录

训练输入：

```text
post-trained/panels/collision_bc_anchor_v1/
```

正式run：

```text
post-trained/ppo_corridor_temporal_collision_bc_anchor_v1/
post-trained/ppo_constrained_collision_cost_calibrated_v1/
```

正式eval继续使用既有规范：

```text
eval_results/<EXPERIMENT_ID>/update<N>/<MAP_NAME>/multiagents/
```

每个方法完成、停止或中断后：

- `ANALYSIS.md`写完整设计、分母、逐图/合计、配对、训练band、机制结果、失败原因和未知项；
- `HANDOFF.md`写当前判决、是否允许下一步、停止/重开规则和持久actor SHA；
- `EXPERIMENTS.md`只写新增工具与接口的重建合同；
- 不在HANDOFF/ANALYSIS增加JSON/CSV/report路径清单或分析产物哈希。

## 9. 当前执行授权

用户已要求简化开跑前流程并直接完成A、B训练与固定四图600评测。A固定
`beta_anchor=0.006405998602049812`；B固定`d=.19`。两条训练不再由研究性probe阻止，最终只按§1与
四图600结果判决。

## 10. 执行结果与自动化边界

### 10.1 实际执行不是一个持久脚本包办全部环节

根`run.sh`在执行期只负责当前方法的训练，不包含eval、最终统计或文档更新。方法B另用一次性
watcher等待训练完成，先验证62行metrics、30组actor/reward/cost checkpoint、16/16 actor step、
reward/cost唯一化和U27--U30 strict 12-key，再串行运行16个固定CUDA 600包。该watcher使本次任务
从用户角度无需中途再次下命令，但它不是仓库中的长期“一键训练+评测+科研判决”入口；最终9600
trace审计、配对统计和文档收口仍独立执行。任务完成后`run.sh`中的已执行命令已按合同移除。

### 10.2 方法A最终结果

方法A完整完成45个Austin formal update与U42--U45固定四图各600。U44主点为
`58 collision / 1390 overtake`：相对历史U44 collision净少4但未检出，overtake显著净少88；
四张地图overtake均低于BC线。Regularizer确实进入GRU/output-head优化，但当前固定teacher、anchor
与beta把策略推向更少超车的区域，没有形成新前沿。固定实例关闭，完整结论见`ANALYSIS.md` §49。

### 10.3 方法B最终结果

第一次方法B目录在任何formal optimizer step前因cost-buffer继承漂移停止，没有科学模型结果。
最小修复后的fresh `_rerun`完整完成30个Austin formal update；cost critic、dual与combined actor
advantage都真实执行。30/30 pooled collision rate高于`d=.19`，dual从1经warm-up后单调升至
`3.0988`，没有达到约束平衡。

固定四图结果为：

| update | Austin | Hockenheim | MoscowRaceway | Nuerburgring | 四图合计 |
|---:|---:|---:|---:|---:|---:|
| U27 | `22/371` | `33/363` | `32/394` | `22/398` | `109/1526` |
| U28 | `16/374` | `27/371` | `37/394` | `26/397` | `106/1536` |
| U29 | `23/371` | `28/371` | `33/394` | `25/397` | `109/1533` |
| **U30主点** | **`22/372`** | **`37/365`** | **`35/394`** | **`25/397`** | **`119/1528`** |

U30相对U44显著新增57次collision并显著新增50次overtake；相对RW30显著新增46次collision，
overtake净多12次但未检出。U28虽双轴优于本方法其他三个late点，仍未Pareto支配U44/RW30，且
不得事后替代预注册U30。方法B因此只形成高超车/高碰撞交易点，没有完成§1.1或§1.2。固定
`P20 × d=.19 × lambda0=1 × dual_lr=.5`实例关闭，完整证据见`ANALYSIS.md` §50。

### 10.4 最终决策

两条方法的训练机制都确实执行，但都没有得到优于既有前沿的模型。Production保持原U30；安全端
仍看U44，高超车端仍看证据级别较低、待fresh CUDA确认的RW30。不得用A/B相邻checkpoint事后选点，
也不得扫描beta、budget、dual、critic或训练长度重试。A/B的方法类没有被数学上整体否决；合法
重开必须引入并先验证新的状态条件安全机制，而不是只改变剂量。
