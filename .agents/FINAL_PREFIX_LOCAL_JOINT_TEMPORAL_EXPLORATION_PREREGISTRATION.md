# End2Race 最后一轮 exploration：prefix-local joint temporal exploration 预注册

状态：**逻辑与执行合同已冻结；尚未实现、尚未运行、没有新 actor。**

2026-08-09 第二次实现前审计：纠正“无未来信息”“状态分布不变”和“严格单变量”的过强表述；
冻结 marginal autoregressive likelihood 的 minibatch/cross-rollout 上下文合同；把 batched 裁决移到
每个 optimizer step 之前；增加轻量 disabled-path optimizer 回归；拆分机制信号与产品资格；正式
验收只使用固定 600-episode panel，**不运行 near400**。这些修订不改变探索核、cohort、reset、
window、std、rho、H、PPO预算或四图正式评测场景。

2026-08-09 修订（在任何实现与运行之前）：§3.1 补入两条此前遗漏的已验证事实（§18时间相关
速度探索的正面先验；既有`corridor_temporal`实现存在同一秩亏近似）；§7.3/§8.2/§10 把
batched replay 越界从硬失败改为预注册裁决，exact 项仍不可裁决；§9.2 由单一"最终目标"判据
改为 L1/L2/L3 三层判决并冻结 L2 阈值。**探索核、cohort、reset 比例、window、std、rho、H、
PPO 预算与评测面板均未改动。**

本文是独立的新方法预注册，不改写 `GATES.md` 中 Z0--Z9 的历史设计或判决。用户本轮授权的是
审计与生成执行文档；代码实现、no-update Gate 和正式训练须在用户明确下达“开始执行”后进行。

## 1. 技术摘要

最后一轮只保留一个纯 RL 候选：

> **Single-stage recurrent PPO with prefix-local temporally correlated joint exploration**
>
> 单阶段 recurrent PPO + prefix 局部二维时间相关探索

它不增加 imitation loss、cost critic、runtime selector、部署期 gate，也不读取当前 student rollout
的未来collision、terminal outcome或任何部署期未来信息；最终 actor 仍为361D输入、12-key
checkpoint。但28项prefix及其`collision/lost_overtake`来源标签本来就是按历史U42--U45未来结局
离线筛选的，因此准确名称是：**actor objective仍为纯PPO，但训练curriculum使用冻结的historical
hindsight。**

相对已经完成的Z6-F，scenario pool、reset scheduler、28项prefix集及顺序、reset比例、reward、
critic、PPO budget与评测面板这些**外生合同**不变。处理是在 **19项 collision-source prefix**
恢复后的前150个动作步内，把原本逐步独立的steering/speed标准化残差改成固定50步块内相关的
二维高斯过程；9项lost-overtake prefix与所有其他状态继续使用baseline独立高斯。不同动作允许且
预期改变处理后的真实on-policy状态访问分布，这正是本方法的作用机制，不能写成“训练状态分布
不变”。

本方法检验的不是“增加 steering 幅度”，而是：**当前逐步独立探索能否产生足够持续、可由 PPO
正确计分的纵横向动作序列。** steering latent std 固定 `0.03`，speed physical std 固定
`0.15 m/s`，两维瞬时边际方差不变，steering-speed 瞬时交叉协方差仍为0；只改变时间相关性。

正式训练前必须通过三道门：

1. 数学与实现 Gate：相关动作过程必须具有满秩、可精确重建的条件 likelihood；
2. 102,400-transition no-update Gate：current-network burn-in、GAE、完整buffer likelihood、
   exposure、block边界和参数不变合同全部通过；
3. disabled-path optimizer Gate：在冻结buffer克隆上实际执行actor/critic optimizer，确认新代码
   关闭时不改变baseline更新语义。

如果实现只能“重复同一个噪声50步，再把每步当独立高斯计算 log-prob”，立即停止。那种实现只能
证明 collection/replay 对同一近似自洽，不能证明 PPO 使用了正确的相关轨迹 likelihood。

## 2. 对 Claude 本轮分析的裁决

| Claude 主张 | 裁决 | 写入本方案的边界 |
|---|---|---|
| Z6-F `103/1522` 被重加权 U30 `73/1516` 双轴支配 | **错误** | 重加权少30次碰撞，Z6-F多6次超车；两点互不支配 |
| Z6-F “缺持续协调探索”是解释 | **同意改为待检验假说** | Z6-F只隔离了prefix-reset，不能从结果反推出探索根因 |
| 原 §6.2 同时改prefix cohort、联合残差、时间相关性 | **正确的高风险批评** | 本方案不改28项cohort和reset比例，只改一个探索核 |
| `z≈0` 时 steering physical std约`0.0156 rad` | **正确的局部近似** | `0.52 × 0.03 = 0.0156`；离开零点后受`tanh`导数影响，只会更小 |
| 单步`|Δsteer|≥0.02`约20%，50步同号持续约`1e-50` | **正确，但只限局部独立采样近似** | 用于说明“幅度已有、持续性缺失”的量级，不外推为全状态精确概率 |
| 因此提高 steering std 是错误旋钮 | **过强** | steering-std A/B从未运行；只能说当前证据优先支持先测相关性，不构成std科学否决 |
| collision-only后 exposure约5.7% | **只对prefix来源后缀的粗算成立** | `19/28×8.43%=5.72%`不是前150步有效处理率；按既有window数据估计约`2.89%--3.60%`，必须实测 |
| preference独立样本很少，beta是关键 | **正确，需细化** | Z2非noop oracle独立prefix为early 165、late 166；pair数可更多，但独立状态不超过该量级 |
| preference是同一家族第四个变体，先验低 | **正确** | Z2/Z4/Z5/Z8没有否决新loss，但已显示同一跨startpoint泛化难题 |
| Constrained PPO若重开可反转目标/约束方向 | **合理的新设计提示，不是已验证结论** | 可研究“安全为reward、超车保有量为constraint”，但它不是本轮exploration臂 |

## 3. 已验证事实、假说和未知项

### 3.1 已验证事实

1. Z2在future-event定位的两个50步窗口中，collision两层的oracle rescue为early `137/155`、
   late `141/155`；动作库中广泛存在局部闭环解。
2. Z2最强统一固定动作包含`steer +0.02 rad / speed +0.5 m/s`，但这个结果不能隔离steering、
   speed或两者组合的因果贡献。
3. 334条历史困难失败上的持续制动扫描表明，持续`-0.15 m/s`已经产生非零且可观的救援率；
   因此小幅持续动作并非必然落在无梯度平台。
4. 当前 stochastic actor 对 steering 使用 latent Normal 后经`tanh×0.52`，latent std `0.03`；
   speed 使用physical Normal，std `0.15 m/s`。baseline下每一步重新采样steering与speed。
5. Z6-A--Z6-C已经验证snapshot精确恢复、current-network burn-in、prefix GAE与现有likelihood
   重建；Z6-F正式训练证明prefix-reset可以稳定运行并改变最终actor。
6. Z6-F U30为`103/1522`；重加权 U30为`73/1516`但CUDA provenance待确认。两点互不支配，
   当前没有已测部署actor达到四图`collision < 40 / overtake > 1500`。
7. **持续性在本项目已有一次正面证据，本文此前遗漏。** §18的前向走廊门控**时间相关速度**
   探索把跨地图same-line collision从`66`降到`22`，这正是"在门控窗口内持续同向速度残差"
   的效果，也是U44安全端的直接来源。它与Z2的动作存在性是相互独立的两类先验：Z2说明
   *存在*可行的持续动作，§18说明*on-policy持续探索*确实曾经改变过策略。
8. **同一秩亏问题存在于既有实现，本文的标准因此高于历史臂。** 现有
   `corridor_temporal`在`_structured_rollout_parameters`中把同一标准化速度残差保持
   `TEMPORAL_RESAMPLE_STEPS=50`步，而PPO按逐步边际Normal计算log-prob——正是§6.1所拒绝的
   奇异构造。因此U44、重加权臂以及§18全部时间相关臂的训练似然都是近似的。这不影响它们
   **确定性评测**结果（评测无噪声），但意味着：(a) 本文的精确似然要求严格高于产出项目
   最佳安全点那条臂的标准；(b) "持续探索有效"这一先验是在**近似似然**下取得的，本文把它
   升级为精确似然，属于同时提高严谨度而非只换机制。该事实应在最终报告中如实记录。

### 3.2 待检验假说

在Z6-F已经提高关键状态访问密度后，当前逐步独立高斯仍很难连续50步产生同方向纵横向残差。
如果把**相同边际幅度**变成可精确计分的时间相关过程，PPO可能获得更多具有连续动作因果结构的
on-policy样本，并学到比Z6-F更安全、同时保留进度的动作映射。

这不是Z6-F结果已经识别出的根因。Z6-F同时可能失败于prefix选择、reward/advantage排序、单seed
优化方差、Austin到三图泛化或共享参数干扰。正式结果只能判定本文固定实例，不能把成败归因到
所有“持续协调动作”方法。

### 3.3 仍然未知

- 相关探索是否会把actor推向新的安全点，还是再次形成安全--超车trade-off；
- Austin collision-source prefix上学到的行为能否跨地图泛化；
- 同时相关steering与speed是否优于只相关其中一维；本文不做component消融；
- steering std变化是否有价值；本文冻结std，故不回答该问题；
- 有限可部署点是否真的位于同一条Pareto曲线；当前数据不足以证明。

## 4. 为什么优先检验“持续性”而不是“幅度”

在deterministic steering latent mean `z=0`附近：

```text
physical action = 0.52 * tanh(z)
d action / dz   = 0.52
local physical std ≈ 0.52 * 0.03 = 0.0156 rad
```

对独立`N(0,0.03²)` latent noise，忽略动态mean变化并在`z=0`计算：

```text
P(|Δsteer| >= 0.02 rad) = 0.1996028
P(|Δsteer| >= 0.04 rad) = 0.0101940
P(连续50步都 >= +0.02 rad) = 9.05e-51
P(连续50步同一方向且每步|Δ|>=0.02) = 1.81e-50
```

前两项说明Z2的`0.02 rad`幅度已在单步常规支持内；后两项说明逐步独立采样几乎不可能自然形成
50步同号动作。动态latent mean、`tanh`饱和、车辆steering delay和闭环状态变化会改变实际概率，
所以这些不是训练数据上的精确频率；它们只承担**量级论证**。

因此本轮冻结`0.03/0.15`边际std，不扫幅度。若本轮失败，不能反向证明“加大std一定无效”，
但在没有新机制证据时也不再开启std sweep。

## 5. 单一方法级复合 treatment 定义

### 5.1 对照与处理

科学对照是已经完成的 Z6-F `ppo_prefix_reset_consensus1of3`，不是普通production PPO。

| 项目 | Z6-F control | 本轮 treatment |
|---|---|---|
| 初始化 | canonical BC | 相同 |
| 训练地图 | Austin | 相同 |
| seed / env / rollout | 42 / 16 / 6,400 | 相同 |
| critic / PPO | privilege-GRU / production PPO | 相同 |
| reward / cache / role | 固定四项 / 479 / 50:50 | 相同 |
| prefix集合 | 28项，collision/lost=`19/9` | **相同28项** |
| prefix reset比例 | collision role每3次reset中1次 | 相同 |
| prefix窗口 | 恢复后前150动作步 | 相同 |
| baseline边际std | steering latent`.03`、speed physical`.15` | 相同 |
| 唯一方法级处理 | 窗口内仍逐步独立 | 19项collision-source窗口内使用固定相关探索核 |

不得把prefix集合改为19项collision-only，不得删除9项lost-overtake，不得改变prefix抽样顺序、
每3次reset比例或150步window。本文“一次只改一个轴”只表示引入一个预注册的**复合exploration
mode**；该mode同时包含historical collision-source gate、steering时间相关性和speed时间相关性。
正式成败只可归因于完整组合，不能进一步拆成“source gate贡献”“steering贡献”或“speed贡献”。

### 5.2 处理范围

只有同时满足以下条件的transition进入相关探索：

1. 当前episode由Z6 snapshot恢复；
2. snapshot在冻结元数据中属于19项`collision` source，而不是9项`lost_overtake`；
3. 当前是恢复后的第0--149个`action_applied=true`动作步；
4. episode尚未terminal。

150步分成3个互不相关的50步block。episode reset、prefix reset、terminal和每个50步block边界
都清空相关状态。其余transition走现有baseline分布，不能消耗相关采样器的随机数并改变baseline
RNG序列。

### 5.3 Exposure的正确口径

Z6-F正式训练的prefix suffix transition最低占比为`8.43%`，前150步window最低为`4.26%`；
Z6-C no-update Gate的对应window为`5.31%`。如果28项近似均匀且长度差异忽略：

```text
collision-source suffix粗估 = 19/28 * 8.43% = 5.72%
collision-source 150-step window粗估 = 19/28 * 4.26% = 2.89%
Z6-C口径粗估 = 19/28 * 5.31% = 3.60%
```

因此`5.7%`不能直接称为相关exploration有效处理率。正式使用机器实测的
`joint_temporal_active_fraction`，no-update Gate最低线固定为`>=2.0%`，并要求19项collision
source全部至少出现一次。这个2%沿用Z6-C既有prefix-window最低密度门，不由本轮结果反推。

## 6. 满秩时间相关探索与精确 likelihood

### 6.1 不允许使用精确hold的原因

若在一个block只采样一个残差并精确重复50步，则整个50维动作序列只由一个随机变量生成，联合
分布在50维空间中是奇异的。actor mean更新后，旧动作序列通常不再位于新策略的同一低维流形，
普通逐步高斯ratio不是该联合分布的精确likelihood ratio。

所以本轮不用“完全相同残差hold50”。它改用满秩的等相关高斯：保留强公共分量，同时给每步
非零innovation，使所有动作序列都有有限密度。

### 6.2 固定探索核

每个50步block、每个动作维度分别生成标准化残差：

```text
r_t = sqrt(rho) * epsilon_block + sqrt(1-rho) * eta_t
epsilon_block ~ N(0,1)
eta_t        ~ N(0,1), iid
rho          = 0.90
H            = 50
```

steering和speed使用独立的`epsilon_block/eta_t`，故同一步两维交叉协方差为0；它们共享block
起止时刻，但不强行规定“左转必须加速”之类的符号关系。

固定值：

```text
steering latent std = 0.03
speed physical std  = 0.15 m/s
rho                 = 0.90
block length H      = 50 action steps
eligible window     = 150 action steps
```

不扫`rho`、H、std或维间协方差。`rho=.90`使边际方差保持1、块内任意两步相关系数为.90，同时
保留`1-rho=.10`的innovation方差，避免奇异hold。

### 6.3 逐步条件密度

设当前block已经有`n=t-1`个标准化残差。等相关高斯的精确条件分布为：

```text
c_n = rho / (1 + (n-1)*rho)
mu_t = c_n * sum(r_1 ... r_n)
v_t  = 1 - rho^2 * n / (1 + (n-1)*rho)
r_t | r_<t ~ N(mu_t, v_t)
```

`t=1`时固定`mu_1=0, v_1=1`。

steering物理动作`a_s`先反变换：

```text
x_t = atanh(clamp(a_s / 0.52))
m_t = atanh(clamp(actor_mean_steer / 0.52))
r_t = (x_t - m_t) / 0.03
```

其log-prob必须包含`-log(0.03)`与现有`tanh×0.52` Jacobian；speed用
`r_t=(a_v-mean_v)/0.15`且无`tanh` Jacobian。两维条件log-prob相加。

PPO replay必须按完整block顺序重建candidate-policy下的历史残差；不得保存old residual后在新策略
下直接当常量。temporal block边界可以切分likelihood sequence，但不能错误清零actor GRU hidden。
只有该条件因子分解通过数值参考验证，actor loss才可继续使用现有standard clipped PPO surrogate。

### 6.4 minibatch、block与跨rollout实现合同

本文冻结的是§6.3的**边缘化后autoregressive conditional likelihood**。不得为了实现方便静默改成
“保存公共`epsilon_block`并条件于该latent”的增广策略likelihood；后者是另一种合法但不同的PPO
surrogate，若未来采用必须另行改写方法和Gate，本轮不允许。

当前rollout buffer会在全部flattened transition中随机选择`split_index`，所以actor minibatch可能从
episode、prefix window或50步block中间开始；prefix episode和block也可能跨越两个6,400-step
rollout边界。新实现不能在minibatch或rollout边界把`n/sum_r`静默清零。每个处理transition至少要
保存或可无歧义恢复：

```text
joint_temporal_active
block_uid              int64，训练路径中不得转float32
block_position         0..49
prefix_step            0..149，审计字段
source_stratum         collision，审计与处理泄漏字段
```

`env_rank`若已编码进全局唯一`block_uid`则不强制重复保存。actor sequence从block中间开始时，必须
提供最多49步的likelihood context：从真实block start和该点保存的actor hidden开始，重放此前
observation与physical action，在candidate actor下重新计算每一步mean和residual累积量。context行：

- 不进入PPO loss、advantage normalization、clip fraction、KL或valid-transition计数；
- 仍参与candidate conditional likelihood的计算图，过去candidate mean不得detach；
- 遵守现有recurrent PPO在sequence起点使用保存hidden的截断语义，不把block边界误当episode reset。

若block跨rollout边界，下一rollout必须携带block-start context或等价的无损恢复索引；不得因buffer
reset改变`block_uid/block_position`或探索状态。唯一另一种可接受实现是在预注册中明确把rollout
边界定义成相关block边界，但那会改变当前`prefix_step//50`的固定处理核，故本轮禁止。

### 6.5 方法命名边界

满足§6.3时，本方法可以称为“single-stage PPO with prefix-local correlated exploration”：相关
采样器只是训练期随机策略的内部状态，actor loss仍是PPO，部署actor不变。

若实际实现沿用现有“相关残差 + 每步边际Normal log-prob”的近似，只能称为带相关行为噪声的
近似PPO，**不满足本文准入条件，不得启动正式训练。**

## 7. 执行顺序

### 7.1 P0：先补齐零训练产品基线

在消耗最后一次训练前，优先完成现有重加权U30的固定CUDA四图确认，以及production缺失的
MoscowRaceway/Nuerburgring规范包，共3,600 episode。它不决定本文机制Gate，但决定最后的新actor
能否与RW/production做同证据级别比较。

如果P0暂不执行，本文仍可相对正式CUDA的BC、U44和Z6-F判决，但不得宣称“正式超过RW或
production”。不能把重建包`73/1516`直接升级为正式CUDA baseline。

### 7.2 Gate E0：零仿真的数学与实现门

只实现最小相关采样/likelihood组件，不接optimizer。固定seed进行以下检查：

1. treatment disabled时，现有baseline action、log-prob、RNG消费、checkpoint与telemetry fast path
   逐位不变；`rho=0`测试模式只要求分布、log-prob、ratio和gradient与baseline数值等价，除非实现
   显式转入baseline fast path，否则不要求随机样本逐位相同；
2. 100,000个synthetic block中，每维均值绝对值`<=0.01`、方差在`[0.98,1.02]`；每一维50×50
   经验相关矩阵的diag最大误差`<=0.01`、全部off-diagonal与`.90`最大误差`<=0.01`；steering-speed
   交叉相关使用`RMS <=0.005`且最大绝对值`<=0.016`，不采用会因2,500重比较而高概率误杀正确
   sampler的“全部<=0.01”；
3. 随机长度1--50的float64 conditional实现与直接multivariate Normal Cholesky联合log-prob误差
   `<=1e-6`；FP32逐步条件log-prob与float64 reference误差`<=5e-5`；
4. 随机变化candidate mean sequence上，conditional实现对全部mean的autograd与float64直接MVN
   joint log-likelihood gradient逐维最大误差`<=1e-5`；steering物理空间连同`tanh×0.52` Jacobian
   再独立通过一次；
5. 同一个完整50步block在全部49个cut位置构造minibatch，cut-context的总candidate log-prob、
   每步有效log-prob和对candidate means的gradient与未切分reference一致：float64 log-prob误差
   `<=1e-6`、gradient最大误差`<=1e-5`；
6. 人工让block跨越rollout边界，恢复后的block identity、每步log-prob和gradient满足同一reference
   线，且上一rollout context不进入loss/advantage/metrics；
7. steering逆`tanh`、Jacobian、动作边界、terminal、episode reset和block reset全部finite；
8. deterministic evaluation与strict 12-key actor加载完全不经过相关采样器。

任一失败即停在实现层。允许修复确定的实现bug后用同一合同重跑E0；不得借修bug改变`rho/H/std`
或处理范围。

### 7.3 Gate E1：102,400-transition no-update集成门

使用canonical BC、fresh privilege-GRU critic、seed42、Austin、16×6,400和Z6-F同一prefix合同，
分别收集一个disabled baseline与一个treatment rollout，不执行optimizer step。

全部条件必须同时满足：

| 条件 | 准入线 |
|---|---:|
| 每臂transition | `102,400`，finite |
| collision/ordinary role | `51,200 / 51,200` |
| baseline disabled回归 | 与现有Z6-C/Z6-F同合同；参数与输出语义不变 |
| prefix queue | 28项全部覆盖；不得换集合/顺序 |
| treatment source | 19项collision source全部覆盖 |
| 相关active fraction | `>=2.0%`，并报告实测值与block数 |
| 处理泄漏 | lost-prefix、full-start、ordinary、150步外均为0 |
| current-network burn-in | 沿用Z6-B逐步exact语义；不得复用U44旧hidden |
| GAE advantage/return | reference最大误差各`<=1e-6` |
| exact collection/replay | `max |log-ratio| <=5e-5`且`max |ratio-1| <=5e-5` |
| batched replay | 两项均`<0.02`；越界时按下段裁决条款处理，不直接判科学失败 |
| action执行一致性 | pre-env clipping count=0；stored PPO action、wrapper action和simulator executed ego action逐行相同；inverse-tanh finite |
| RNG隔离 | baseline sampler保持Z6-F调用顺序；joint sampler使用独立固定stream；disabled时joint stream不创建、不推进 |
| actor/critic参数 | Gate前后逐tensor不变 |
| dry actor gradient | finite；16/16 planned minibatch均可计算，不step |
| 工程wall-clock | treatment/baseline `<=1.35x`；仅为资源门，不是方法科学必要条件 |

E1只证明工程与likelihood合同成立，不证明正式训练会改善。任何科学门失败即停止；纯性能问题可在
不改变数学和样本语义的前提下优化实现一次，再完整重跑E1。

RNG合同不要求treatment开始后所有未直接处理env的物理轨迹长期逐位等于baseline，因为处理可能
改变terminal时刻并经共享scheduler间接改变后续reset。它要求的是：原baseline steering/speed
draw仍按Z6-F固定形状和顺序消费，eligible slot再由专用joint draw覆盖；joint stream的消费次数与
common block/innovation计数一致，不能因另一env提前结束而让随机数身份移位。可用固定形状的单一
专用generator完成，不强制每env、common与innovation各建一套子流。

**batched replay 门的裁决条款（预注册，不得事后添加）。** `0.02`这个阈值卡在对照勉强通过的
位置：Z6-F正式训练实测最大batched ratio为`0.019095`，余量仅`4.7%`；Z6-C更曾以`0.010755`对
`0.01`刀锋失败，最终由独立预注册的Z6-CR裁决收口。因此本文预先固定：

- **exact collection/replay 两项仍通过**（`max|log-ratio| <=5e-5` 且 `max|ratio-1| <=5e-5`）
  而**仅batched越界**时，不判E1科学失败，而是在本update任何actor optimizer step之前记录
  实测值并按Z6-CR同一口径裁决：相同8个first-epoch minibatch上比较普通batched与exact dry
  actor，要求累计gradient cosine`>=0.999`、相对L2差`<=0.02`、clip fraction均为0、mean
  approximate KL`<=1e-4`、valid transition/mask逐项相同，两条路径全部finite且参数不step；
- 裁决通过（梯度方向一致、policy-loss差可忽略）则E1记为
  `pass_after_batched_replay_adjudication`，**原batched失败值必须与裁决结果并列保留**，
  不得简写为"E1通过"；
- **exact两项中任一项越界，直接判科学失败，不进入裁决。** exact是似然合同本身，batched
  只是执行路径的数值特性；两条路径必须使用同一个§6.3精确相关likelihood，裁决不得放行逐步
  独立边际Normal或minibatch起点清空相关历史。

### 7.4 Gate E2：disabled-path轻量optimizer回归

E0/E1通过后、正式训练前，在E1 disabled baseline的同一冻结buffer上克隆两份完全相同的
actor、critic和optimizer状态，固定同一组actor/critic minibatch indices：

1. A路径显式走现有baseline legacy fast path；B路径启用新代码但`treatment disabled`；
2. 两路各执行一个完整actor epoch和一个完整critic epoch，原E1模型本身不step；
3. optimizer step数、minibatch顺序、loss、clip/KL、gradient finite合同一致；
4. 更新后actor/critic逐tensor bitwise相同，optimizer tensor状态逐项相同；
5. 两路保存并strict reload的最终actor仍为相同12-key，deterministic action逐位相同。

E2直接覆盖“新buffer/likelihood代码关闭时是否改变optimizer调用”的风险，不额外运行一条完整
warm-up+U1，也不要求工作树clean或先创建新的正式训练目录。E2失败只说明disabled实现存在回归；
允许修复该实现bug后重跑受影响的E0/E1/E2，不构成相关探索方法的科学否决。

### 7.5 为什么不再增加一个大规模branch Gate

Z2已经用11,400条闭环分支证明50步持续动作在当前失败窗口广泛存在；历史334条持续制动扫描又
证明1-sigma speed动作有非零效果。再做动作库oracle主要重复“有没有动作”，而本轮新问题是
“on-policy相关探索能否让PPO采到并正确计分”。为提高效率，不再用future-event oracle做新的
动作选择器训练；E0/E1之后只完成轻量E2，三门全部通过才进入唯一正式run。

### 7.6 最小provenance合同

正式run前生成紧凑`gate_plan.json`，记录预注册文档和实际参与训练的源码/config内容身份、当前
worktree status、canonical BC、28-prefix manifest与collision cache身份、`rho/H/std/window/reset`
合同、Python/PyTorch/SB3/CUDA/GPU版本、seed derivation及E0/E1/E2报告身份。用户已有未提交改动
必须保留，**不以worktree clean或创建Git commit作为准入条件**。

E1通过后若采样、likelihood、buffer、optimizer或训练语义源码改变，必须重跑受影响的Gate；仅
文档、注释或不参与判决的展示字段变化不机械要求全门重跑。正式结果的配置、样本量、配对身份
变化、不确定性、机制结果、产品判决与停止边界必须写入`HANDOFF.md`本身，不能只留报告路径。

## 8. 唯一正式训练

### 8.1 固定配置

```text
EXPERIMENT_ID            ppo_prefix_reset_joint_temporal_rho0p90
initial actor            pretrained/end2race.pth
training map             Austin only
seed                     42
critic                   privilege_gru
n_envs / n_steps         16 / 6400
formal updates           30
batch                    12800
actor / critic epochs    2 / 5
GRU / head / critic LR   3e-6 / 3e-5 / 3e-4
clip                     0.20
steer / speed std        0.03 latent / 0.15 physical
prefix set               Z6-F原28项，19 collision + 9 lost
prefix schedule          collision role每3次reset中1次
joint temporal kernel    rho=.90, H=50，仅19项collision-source前150步
```

除上表最后一行外，必须与Z6-F一致。不得加入BC loss、preference loss、cost critic、reward变更、
collision-only cohort、std扫描、额外地图训练或checkpoint continuation。

### 8.2 运行规则

1. E0/E1/E2通过后才创建根目录`run.sh`，只放一条显式训练命令；
2. 使用`/home/haowei/miniconda3/envs/end2race/bin/python`与tmux；
3. 输出必须是全新目录，禁止写入Z6-F或任何已有`post-trained/`目录；
4. warm-up + 30 formal updates，不能从U30延长或中断resume；
5. 每update记录相关active fraction、block数、两维residual统计、exact/batched ratio、GAE与完整
   PPO telemetry；
6. 每update **exact ratio**任一项`>5e-5`时，在actor step前fail closed（似然合同失败，不可裁决）；
   **batched任一项`>=0.02`但exact仍通过时**，必须在本update第一个actor optimizer step前完成
   §7.3的冻结dry-gradient裁决；通过才继续该update，失败则立即停止。不得先执行actor update、
   再在训练结束后解释此前梯度。若**任一update的exact越界**，仍立即停止且不重跑；
7. 任何update actor optimizer steps不是16/16、数值非finite或12-key失败时停止；
8. 不并行启动第二个训练臂，不进行seed重复。

## 9. 正式评测与判据

### 9.1 固定评测

训练完成后只评U27、U28、U29、U30；每个checkpoint在Austin、Hockenheim、MoscowRaceway、
Nuerburgring各600 episode，CUDA、ego collision scope、deterministic mean action、保存numeric
traces。四个checkpoint共9,600 episode，不做best-checkpoint选择。

每个checkpoint使用唯一eval alias：`PJTE_u27/PJTE_u28/PJTE_u29/PJTE_u30`，不得让同名
`actor.pth`覆盖trace。**最终任务验收只认上述固定600-episode正式panel；不运行near400，也不以
near400、hard73或其他困难子集增加、替代或推翻正式判决。**

每图必须满足600 unique、0 error、finite、result/trace key一致、collision marker和terminal row
合同。逐图和四图都报告collision/overtake；U30相对BC、U44、Z6-F U30报告collision
removed/created、overtake lost/gained与exact paired p。如果P0已补齐，再以同口径报告相对RW30与
production；否则明确标为不可作同级配对。

U30仍是预注册主判决点。若U27--U29中任何一个**预先固定、完整评测的checkpoint**在相同四图
各600合同上独立达到§9.2的L3数值线，也记录为`formal_600_task_achieved_at_Uxx=true`并视为本次
“找到满足600正式评测的actor”任务完成；必须同时保留U30主判决，不得把该规则扩展到额外
checkpoint、追加训练或事后新panel。是否切换production仍需单独用户授权和稳定性判读。

### 9.2 分层判决

U30仍是预注册主点，但用户明确规定：最终任务评测口径是固定600-episode正式panel，不是
near400；若U27--U29中存在完整四图600结果达到同一产品目标，也算找到满足任务的actor。判读
因此同时报告U30主判决与有限预注册checkpoint集合中的`formal_600_task_achieved`，不得新增
checkpoint或panel寻找最好点。

| 层级 | 条件 | 含义 |
|---|---|---|
| L1 实现通过 | E0、E1、E2全部通过；batched只能在optimizer前按§7.3裁决 | 实现可信，允许正式训练 |
| L2-M 固定seed机制信号 | U30满足下方三条 | 当前seed42实例产生了与假说一致的安全改善信号，不等于多seed复现 |
| L2-P 最低产品资格 | U30四张正式600 panel逐图通过BC线 | U30具备最低产品候选资格 |
| L3-600 最终任务目标 | 任一预注册U27--U30满足下方三条 | 在正式600评测上找到达到用户目标的actor，任务完成 |

**L2-M 固定seed机制信号**必须同时满足：

1. treatment U30四图collision`<=91`，即比Z6-F U30的`103`至少减少12次，并且同一2400场景
   配对exact McNemar双侧`p<0.05`；
2. treatment U30四图overtake`>=1507`，等价于相对Z6-F U30的paired
   `lost_overtake - gained_overtake <=15`。双侧exact p继续报告，但`p>=0.05`不再冒充非劣证据；
3. 相邻点方向一致：treatment U29 collision严格低于Z6-F U29的`101`，treatment U30严格低于
   Z6-F U30的`103`。不得把treatment U29拿去和Z6-F U30比较。

这里12次是四图2400 episode的0.5个百分点；15次约为Z6-F U30 `1522` overtakes的0.99%。
L2-M通过只写“固定seed机制一致信号”，因为episode-level配对p不覆盖训练seed不确定性。L2-M
未通过表示本固定实例未建立预注册正面信号；除非结果直接推翻方法必要条件，否则不能写成对所有
时间相关探索的方法类科学否决。

**L2-P 最低产品资格**要求U30四张地图分别满足：

```text
collision <= 该图canonical BC
overtake  >= 该图canonical BC
```

L2-M通过而L2-P失败，准确记录为“机制信号成立，但地图级产品守门失败”；不得归并为机制无效。

**L3-600最终任务目标**对每个预注册checkpoint `C in {U27,U28,U29,U30}`独立判定，必须同时满足：

1. C四图合计`collision <40`；
2. C四图合计`overtake >1500`；
3. C每张地图collision不高于该图canonical BC、overtake不低于BC。

任一C通过即设置`formal_600_task_achieved_at_C=true`并算本轮任务完成；U30是否通过仍单列，不能
把U27--U29通过改写成“U30通过”。U29与U30均逐图通过BC线时记为late-band产品稳定性支持；
U29与U30均达到`<40/>1500`且逐图通过BC线时记为强稳定成功。稳定性决定是否建议切换production，
但不再否定某个完整600正式checkpoint已经达到任务目标这一事实。

### 9.3 次级判读

以下只用于描述，不构成 L1、L2-M、L2-P或L3-600任何一层：

- 是否严格Pareto优于U44 `62/1478`；
- U27--U30 aggregate范围和碰撞身份churn；
- same-line/off-line、ego-opp/ego-wall分层；
- 19项collision-source窗口内的训练return/advantage与探索block统计；
- 相对重加权U30 `73/1516` 的关系（**只有 §7.1 的 P0 已完成**才可作同证据级别比较；
  否则只能标注为重建包对正式包，不得写成"超过 RW"）。

即使次级指标改善，只要L2-M与L3-600均失败，本固定实例仍关闭，不调参重试。

## 10. 停止规则

1. E0、E1或E2任一必要实现合同失败：不正式训练。E1的batched replay项按§7.3在optimizer前
   裁决，exact项越界则不可裁决；实现Gate失败不写成方法科学否决；
2. likelihood只能通过逐步边际近似：不正式训练；
3. 正式run一旦发生actor update，不因中期结果差而调参数、换prefix或重启同配置；
4. 完成30 updates后无论结果如何关闭本实例；L2-M通过而L3-600失败时记为“固定seed机制信号
   成立、未达产品目标”，同样关闭本实例，不据此授权参数扫描或后续臂；
5. 不扫`rho`、H、steering/speed std、prefix比例、19/9构成、window、LR、reward、seed或updates；
6. 不延长到U45，不增加U27--U30之外的checkpoint；§9.1允许预注册四点中任何完整600结果达到
   L3-600时计为任务完成，但U30主判决必须保留，不得借此事后改写训练假说；
7. 不因collision改善而忽略逐图BC overtake守门线；
8. 只有实现/基础设施在**任何actor update前**确定性失败，才可修复后在新目录按完全相同合同重跑。
   **例外**：若训练在actor update之后因§8.2规则6以外的确定性基础设施故障（机器崩溃、磁盘、
   驱动）中断，允许在新目录按完全相同合同重跑一次并如实记录中断原因；不得因中期指标不佳
   借此重启；
9. 结束、失败或中断后按handoff-log合同更新`ANALYSIS.md`、`HANDOFF.md`与必要的
   `EXPERIMENTS.md`实现记录。
10. 不运行near400；任何near400历史结果或其他困难子集都不能替代、增加或推翻四图600正式判决。

## 11. 未选择的两条方向

### 11.1 Simulator-return-filtered first-action preference

该方向有真实机制差异：它直接改变最终actor对候选动作的log-prob，而不是训练外置selector。
但当前不作为最后正式臂，理由是：

1. Z2的label来自50步动作序列；序列终局更好不识别第一步动作本身的Q优势。同prefix只消除`t`
   时刻state mismatch，不能消除后续49步共同干预；
2. Z2全456条中，非noop oracle独立prefix仅early165、late166，严格安全/progress过滤后只会更少；
   pair数可以扩增，但独立状态数不会因此增加；
3. Z5在168 target、225 controls的无泄漏nested合同下未建立frozen-hidden selector严格优于fixed；
   preference仍面对同一跨startpoint泛化问题；
4. 若未来重开，必须先构造“仅第一步不同、后续恢复同一策略”的一阶干预，或准确改名为
   macro-action/sequence preference；
5. `beta`不能拍定。应在训练fold内按Gate C的LR加权step-space范数和clip-aware虚拟step确定唯一
   beta，再冻结到held-out fold；还必须重新burn-in已改变的GRU，并用新起点闭环验证actor副本。

因此它是低先验、非纯PPO的后备研究方向，不与本轮并行。

### 11.2 反转约束方向的Constrained PPO

Z9只关闭`P20 MLP + collision cost constraint + d=.10 + dual_lr=.5`实例。若未来重开，较贴近
当前失败模式的设计可能是：优化安全目标，同时把overtake/progress保有量作为约束，直接阻止
“拿超车换安全”。但这会改变reward/constraint定义，并引入稀疏、延迟的overtake cost、budget
估计与dual稳定性问题；它没有提供关键状态应怎样转向/调速的信息。

这只是新研究命题，不是Z9数据已经支持的解，也不进入最后一轮exploration执行队列。

## 12. 最终证据边界

即使本方案成功，也只能说明：在Austin-only、seed42、固定28项hindsight prefix和本文相关探索
核下，一条fresh recurrent PPO轨迹产生了更好的四图deterministic actor。它不能证明相关探索
普遍优于独立探索、不能证明steering或speed哪一维起主导作用，也不能证明其他seed、地图合同或
更一般驾驶任务同样有效。

若失败，严谨关闭的是：

```text
Z6-F fixed 28-prefix curriculum
+ collision-source first-150-step
+ rho=.90, H=50 full-rank joint temporal exploration
+ canonical-BC fresh Austin PPO, seed42, 30 updates
```

失败不构成“所有时间相关steering探索”“所有prefix-reset”或“PPO理论上无法达到目标”的证明；
但按本项目的最后预算规则，不再以剂量、窗口、std或相关系数扫描继续该阶段。
