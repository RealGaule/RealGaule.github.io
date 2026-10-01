---
title: "沿噪声尺度追踪模态：分叉检测、温度与自适应带宽"
date: 2026-10-01
description: "在 Gibbs 目标的高斯平滑族上，分叉检测、温度调节、sample 与 refine 两阶段的带宽自适应分别是什么问题；用一维双盆地给出可算的例子；把相关文献按问题分类，列出做法、结论与局限，最后整理尚未解决的问题。"
summary: "四个决策都发生在同一族平滑分布 $p_{\\sigma,\\lambda}$ 上，都能从一批加权 rollout 的后验均值与协方差读出信号。文献已讲清分叉何时发生、温度如何与噪声耦合；面向黑箱目标的在线分叉检测、按统计量选温度与带宽仍是空白。"
tags: ["分叉", "温度", "扩散", "Tweedie", "Mean Shift", "SMC", "S2R", "模态枚举"]
---

本章自成一体。记号沿用[《MPPI 迭代、Mean Shift 与 EM》](/posts/mppi-meanshift-em/)和[《SMC 与 CEM》](/posts/smc-cem/)：控制节点 $V$，代价 $C(V)$，温度 $\lambda$，带宽（噪声标准差）$\sigma$，后验 $r_z^*$，后验均值 $m(z)$。S2R 代码里的带宽 `h` 就是这里的 $\sigma$。

S2R（Sample-and-Refine）是本项目的规划器，分两个阶段。sample 阶段做扩散式采样：沿递减的带宽把粒子按质量分到各个模态。refine 阶段在每个模态内用小带宽的 mean shift 把粒子推到峰上。MBD（Pan 等 2024）和 DIAL-MPC（Xue 等 2025）同样沿手工日程退火噪声。文中的 PushT 指平面推物任务（$D=22$）；Dubins 指绕单个障碍物的 Dubins 小车任务，障碍两侧各有一个模态。

第 2 节讲背景，并给出一个可以手算的例子；第 3 节陈述四个问题；第 4 节把相关文献分类，逐篇说明做法、结论与局限；第 5 节列出尚未解决的问题。

## 1. 结论

> [!conclusion] 结论
> **四个问题。** S2R 的 sample 与 refine 阶段、MBD、DIAL-MPC 这类采样式规划器，都沿带宽 $\sigma$ 从大到小处理平滑族 $p_{\sigma,\lambda}=\pi_\lambda*\mathcal N(0,\sigma^2I)$。其中四个决策至今靠手工设定：
> 1. **分叉检测**：降低 $\sigma$ 时，新的峰何时、在哪里、沿哪个方向出现；
> 2. **温度调节**：$\lambda$ 取多大，是否随 $\sigma$ 变，改了之后要不要修正权重；
> 3. **sample 阶段的带宽**：从哪个 $\sigma$ 开始，下一级取多少，何时转入 refine；
> 4. **refine 阶段的带宽**：追踪峰时用多大的 $\sigma$，如何缩小，何时停。
>
> **共同信号。** 每一步的加权 rollout 都给出后验均值 $m(z)$ 与后验协方差。协方差与提议方差之比 $M=\mathrm{Cov}[V\mid z]/\sigma^2$ 有三个身份：mean-shift 映射的 Jacobian、平滑对数密度的曲率（$M-I=\sigma^2\nabla^2\log p_\sigma$）、盆地内的分辨率。四个决策都能从 $(m,M,\mathrm{ESS})$ 读出。但 $\sigma$ 与 $\lambda$ 不是同一个旋钮：盆地内二者只通过 $\sigma^2A/\lambda$ 起作用，盆地之间 $\lambda$ 还决定质量比。
>
> **文献现状。**
>
> | 问题 | 已有做法 | 已经解决 | 尚未解决 |
> |---|---|---|---|
> | 分叉检测 | 确定性退火的临界温度；扩散模型的对称破缺与物种分化理论；尺度空间奇点理论；基于训练好去噪器的检测器 | 分叉何时发生（对称情形有闭式）；一般情形是折叠而非叉形 | 只用 rollout 的在线检测；折叠的检测；蒙特卡洛噪声下的可靠性 |
> | 温度 | $\sigma^2\propto\lambda$ 的耦合；分叉后的局部降温（TSR）；改温度时的精确 SMC 权重；扩散路径保质量、温度路径扭曲质量 | 两类耦合规则与精确修正 | 从统计量自适应选 $\lambda$；分叉窗口里用什么温度 |
> | sample 带宽 | 固定日程（EDM 等）；按 ESS 自适应选温度；在分叉窗口加密的理论建议 | 起点与日程形状的经验规则 | 按后验统计量在线选下一级 $\sigma$ |
> | refine 带宽 | 退火 mean shift；可变带宽 mean shift；单循环同伦；按峰估计采样方差 | 局部尺度的经验选取 | 与分叉检测衔接、有收敛保证的自适应规则 |
>
> **两个绕不开的事实。** 一般情形的分叉是折叠：新峰在远处与一个鞍点成对出现，正在追踪的峰对它毫无反应（第 2.4 节）。$M$ 的最大特征值在高维、小 ESS 时被系统性抬高，阈值检验几乎处处触发（第 3.1 节）。

## 2. 背景

### 2.1 目标与平滑族

控制节点 $V\in B=[-1,1]^D$，$D$ 等于节点数乘以执行器数。代价 $C(V)$ 由一次仿真 rollout 得到，没有梯度。目标是 Gibbs 分布

$$
\pi_\lambda(V)=\frac{1}{Z_\lambda}\,e^{-C(V)/\lambda}\,\mathbb 1_B(V).
$$

它的峰就是 $C$ 在 $B$ 内的局部极小，位置与 $\lambda$ 无关；$\lambda$ 决定各峰的质量之比。一个峰的**质量**指 $\pi_\lambda$ 落在该峰吸引盆内的概率。绕障碍物从左走还是从右走、推物体时从哪一面接触，都是不同的峰。MPPI 对整个分布取加权平均，两个峰之间若是高代价区，平均值就落进去，这就是 mode averaging。

把 $\pi_\lambda$ 与各向同性高斯卷积，得到平滑族

$$
p_{\sigma,\lambda}\coloneqq\pi_\lambda*\mathcal N(0,\sigma^2I),\qquad \sigma\ge0 .
$$

$\lambda$ 固定时简写为 $p_\sigma$。高斯卷积满足半群性质，$\{p_\sigma\}$ 是一条从 $p_0=\pi_\lambda$ 出发、$\sigma\to\infty$ 时趋于高斯的路径。$\sigma$ 是**分辨率**：比 $\sigma$ 小的结构被抹平。

两端的行为是确定的：

- 若 $\pi_\lambda$ 的支撑落在半径为 $r_B$ 的球内，$\sigma>r_B$ 时 $p_\sigma$ 对数凹，只有一个峰。把 $\pi_\lambda$ 看成许多点质量之和，这是 You（2026）定理 7.2 的直接推论。对盒子 $B$ 有 $r_B=\sqrt D$。
- $\sigma\to0$ 时 $p_\sigma$ 的峰趋于 $C$ 的局部极小。

在两端之间降低 $\sigma$，峰一个个出现。这就是本章说的**分叉**。

### 2.2 一批加权 rollout 给出的三个量

在当前点 $z$ 附近撒 $K$ 个提议 $V_k\sim\mathcal N(z,\sigma^2I)$（截断到 $B$），按 $w_k\propto e^{-C(V_k)/\lambda}$ 加权并归一化。它们近似后验

$$
r_z^*(V)\propto\pi_\lambda(V)\,\mathcal N(V;z,\sigma^2I),
$$

即"一个干净点 $V\sim\pi_\lambda$ 被加噪成 $z$ 之后，$V$ 的后验"。$\pi_\lambda$ 在 $B$ 外为零，所以截断到 $B$ 只改变一个常数，不改变后验，还避免了把样本浪费在 $B$ 外。由这批样本得到三个量：

$$
m\approx\sum_kw_kV_k,\qquad \mathrm{Cov}\approx\sum_kw_k\,(V_k-m)(V_k-m)^\top,\qquad \mathrm{ESS}=\Big(\sum_kw_k^2\Big)^{-1}.
$$

> [!theorem] 定理 1（一阶与二阶 Tweedie）
> 记 $m(z)=\mathbb E_{r_z^*}[V]$，$\mathrm{Cov}(z)=\mathrm{Cov}_{r_z^*}[V]$，$M(z)\coloneqq\mathrm{Cov}(z)/\sigma^2$。则
> $$
> m(z)=z+\sigma^2\nabla\log p_\sigma(z),\qquad M(z)=\nabla m(z)=I+\sigma^2\nabla^2\log p_\sigma(z).
> $$

**证明。** $p_\sigma(z)=\int\pi_\lambda(V)\,\mathcal N(V;z,\sigma^2I)\,dV$。由 $\nabla_z\mathcal N(V;z,\sigma^2I)=\mathcal N(V;z,\sigma^2I)\,(V-z)/\sigma^2$，

$$
\nabla\log p_\sigma(z)=\frac{1}{p_\sigma(z)}\int\pi_\lambda(V)\,\mathcal N(V;z,\sigma^2I)\,\frac{V-z}{\sigma^2}\,dV=\frac{m(z)-z}{\sigma^2},
$$

这是第一式。再对 $m(z)=\int V\,r_z^*(V)\,dV$ 求导。由 $r_z^*=\pi_\lambda\mathcal N/p_\sigma$ 和第一式，

$$
\nabla_zr_z^*(V)=r_z^*(V)\Big[\frac{V-z}{\sigma^2}-\frac{m(z)-z}{\sigma^2}\Big]=r_z^*(V)\,\frac{V-m(z)}{\sigma^2},
$$

所以

$$
\nabla m(z)=\frac{1}{\sigma^2}\int V\,(V-m(z))^\top\,r_z^*(V)\,dV=\frac{\mathrm{Cov}(z)}{\sigma^2}.
$$

另一方面，对第一式求导得 $\nabla m=I+\sigma^2\nabla^2\log p_\sigma$。两者相等，即第二式。$\square$

$\nabla\log p_\sigma$ 称为 **score**；由定理 1，它等于 $(m(z)-z)/\sigma^2$，所以一批加权 rollout 就给出 score 的估计。下文 $\lambda_{\max}(\cdot)$ 表示最大特征值，与温度 $\lambda$ 无关。

$M$ 有三种读法，后文反复用到。

1. **Jacobian。** $M$ 是 mean-shift 映射 $z\mapsto m(z)$ 的 Jacobian。在不动点附近，每迭代一步，误差约乘以 $\lambda_{\max}(M)$：小于 1 时线性收敛，越接近 1 越慢（Carreira-Perpiñán 2007）。
2. **曲率。** $M-I=\sigma^2\nabla^2\log p_\sigma$。在不动点处，$\lambda_{\max}(M)<1$ 是峰，$>1$ 是鞍点或密度的极小点，$=1$ 是退化点。叉形分叉就是某个不动点的 $\lambda_{\max}(M)$ 穿过 1。下文把不是峰的临界点统称为鞍点（一维里它其实是 $p_\sigma$ 的极小点）。
3. **盆地内的分辨率。** 设盆地内 $C(V)\approx C_0+\tfrac12(V-\mu)^\top A(V-\mu)$，则 $\pi_\lambda$ 近似 $\mathcal N(\mu,\lambda A^{-1})$，后验是两个高斯之积，$\mathrm{Cov}=(A/\lambda+I/\sigma^2)^{-1}$，于是
$$
M=(I+R)^{-1},\qquad R\coloneqq\sigma^2A/\lambda .
$$
$R$ 是提议宽度 $\sigma$ 与盆地宽度 $\sqrt{\lambda/A}$ 之比的平方（矩阵形式）：
   - $R\gg1$ 时提议比盆地宽，$M\approx R^{-1}$，收缩很快；
   - $R\ll1$ 时 $M\approx I$。

   所以 **$\sigma\to0$ 时任何峰的 $M$ 都趋于 $I$**。用"$M$ 是否接近 1"判断分叉，只在 $\sigma$ 与盆地宽度相当时有意义。在小 $\sigma$ 处要改看曲率 $(M-I)/\sigma^2$，而蒙特卡洛误差会被放大 $1/\sigma^2$ 倍。

   $R$ 含有未知的曲率 $A$，不能直接观测。但在二次盆地里 $M^{-1}-I=R\propto\sigma^2$，所以可以用上一级的 $M$ 外推下一级的基准：
$$
M_{\mathrm{pred}}(\sigma')=\Big(I+\frac{\sigma'^2}{\sigma^2}\big(M(\sigma)^{-1}-I\big)\Big)^{-1}.
$$
   分叉的信号是实测的 $M(\sigma')$ 明显偏离 $M_{\mathrm{pred}}(\sigma')$，而不是超过 1。

ESS 也只依赖 $R$。在峰 $z=\mu$ 处取提议 $q=\mathcal N(\mu,\sigma^2I)$，未归一化权重 $\tilde w=e^{-(C-C_0)/\lambda}$。沿 $A$ 的第 $i$ 个特征方向（对应 $R$ 的特征值 $R_i$），有 $\mathbb E_q[\tilde w]=(1+R_i)^{-1/2}$，$\mathbb E_q[\tilde w^2]=(1+2R_i)^{-1/2}$，各方向相乘，得

$$
\frac{\mathrm{ESS}}{K}\approx\frac{(\mathbb E_q\tilde w)^2}{\mathbb E_q\tilde w^2}=\prod_i\frac{\sqrt{1+2R_i}}{1+R_i}.
$$

单个方向的因子在 $R_i=1$ 时为 0.87，$R_i=4$ 时为 0.60。若 $D=22$ 个方向都取 $R_i=4$，ESS 只剩 $K$ 的约 $10^{-5}$。ESS 由"有多少个方向上提议比盆地宽得多"决定。ESS 的定义与含义见[《SMC 与 CEM》](/posts/smc-cem/)第 2.2 节。

### 2.3 两种用法：搬运与追踪

同一批统计量有两种用法，详见[《MPPI 迭代、Mean Shift 与 EM》](/posts/mppi-meanshift-em/)第 6 节。

- **搬运（diffusion，S2R 的 sample 阶段）。** 让一群粒子在每一级都服从 $p_\sigma$，一级一级降到 $\sigma\approx0$：
$$
z'=z+\kappa\,\big(m(z)-z\big)+\sigma'\sqrt\kappa\,\xi,\qquad \kappa=1-\sigma'^2/\sigma^2,\quad \xi\sim\mathcal N(0,I).
$$
在 score 精确、日程足够细的极限下，粒子落入各峰的比例等于峰的质量。
- **追踪（mean shift / EM，S2R 的 refine 阶段）。** 固定 $\sigma$ 反复做 $z\leftarrow m(z)$，收敛到 $p_\sigma$ 的峰；再缩小 $\sigma$，得到峰的路径。这就是延拓（continuation）；它与确定性退火思路相近，但确定性退火使用的后验不同（第 4.1 节）。

S2R 当前的设定全部是手工给的（PushT 配置，$D=22$）：

| 环节 | 当前做法 |
|---|---|
| 起点 | 盒子中心的 $\mathcal N(0,\sigma_0^2I)$，$\sigma_0=2\sqrt D\approx9.38$ |
| sample 日程 | EDM 日程（$\rho=7$），32 级从 9.38 降到 0.15，每级一步 DDPM |
| 切换 | 降到 0.15 后转入 refine |
| refine | 四级起始带宽 0.12、0.07、0.04、0.02；每步走完整的 mean-shift 步并做接受检查；位移连续两次小于 $0.05\sigma$，或连续三次被拒绝，即停；自洽带宽：逐坐标取 $\sigma_j'^2=\min(c\,\mathrm{Var}_j,\sigma_j^2)$，$\mathrm{Var}_j$ 为加权样本在第 $j$ 个坐标上的方差，$c=2$（见第 3.4 节） |
| 温度 | 全程固定 |

本章的四个问题，就是把这张表的每一行从"手工给定"改成"由统计量决定"时遇到的问题。

### 2.4 一个可算的例子：一维双盆地

后文的结论都能在这个例子上算出来。

**模型。** 为了得到闭式，本例取 $V\in\mathbb R$，不加盒约束。两个曲率相同的二次盆地，右深左浅：

$$
C(V)=\min\Big\{c_1+\tfrac A2(V-a)^2,\ c_2+\tfrac A2(V+a)^2\Big\},\qquad \Delta C\coloneqq c_2-c_1\ge0 .
$$

当盆地宽度 $s\coloneqq\sqrt{\lambda/A}$ 远小于间距 $a$ 时，可以把 $e^{-C/\lambda}$ 中的 $\max$ 换成和，近似为两个高斯之和：

$$
\pi_\lambda\approx \omega\,\mathcal N(a,s^2)+(1-\omega)\,\mathcal N(-a,s^2),\qquad \log\frac{\omega}{1-\omega}=\frac{\Delta C}{\lambda},\qquad s^2=\frac{\lambda}{A}.
$$

卷积之后仍是两个高斯之和，方差变为 $v=s^2+\sigma^2$：

$$
p_{\sigma,\lambda}=\omega\,\mathcal N(a,v)+(1-\omega)\,\mathcal N(-a,v).
$$

记 $\gamma(z)$ 为点 $z$ 属于右盆地的后验概率（责任度）。由 $\gamma'(z)=2a\,\gamma(1-\gamma)/v$，

$$
\frac{d}{dz}\log p=\frac{(2\gamma-1)\,a-z}{v},\qquad M(z)=1+\sigma^2\frac{d^2}{dz^2}\log p=\frac{s^2}{v}+\frac{4a^2\sigma^2\,\gamma(1-\gamma)}{v^2}.
$$

> [!theorem] 定理 2（Robertson–Fryer 1969）
> 两个等方差高斯的混合，权重为 $\omega$ 与 $1-\omega$，记 $d=a/\sqrt v$（半间距除以标准差）。$d\le1$ 时对任何权重都单峰。$d>1$ 时，混合双峰当且仅当
> $$
> \Big\lvert\log\frac{\omega}{1-\omega}\Big\rvert<g(d)\coloneqq2d\sqrt{d^2-1}-2\,\mathrm{arccosh}\,d .
> $$

代入本例：$d^2=a^2/(\sigma^2+\lambda/A)$，$\lvert\log\frac{\omega}{1-\omega}\rvert=\Delta C/\lambda$。我在 400 组随机参数上用数值求峰核对了这个判据，结果全部一致（脚本见图 1 说明）。

**对称情形 $\Delta C=0$：叉形分叉。** 对所有 $d>1$ 都有 $g(d)>0$，所以双峰当且仅当 $d>1$，即

$$
\sigma^2<\sigma_c^2\coloneqq a^2-\lambda/A .
$$

$\sigma>\sigma_c$ 时唯一的峰在 $z=0$。在 $z=0$ 处 $\gamma=\tfrac12$，$M(0)=1+\sigma^2(a^2-v)/v^2$，恰在 $\sigma=\sigma_c$ 处穿过 1。这时正在追踪的点自己变成鞍点，两侧各长出一个峰，分裂方向就是 $M$ 的主特征向量。"$\lambda_{\max}(M)\ge1$ 即分叉"描述的正是这种情形，它与[《MPPI 迭代、Mean Shift 与 EM》](/posts/mppi-meanshift-em/)第 5.4 节的临界带宽一致。

**非对称情形 $\Delta C>0$：折叠分叉。** 此时双峰需要 $g(d)>\Delta C/\lambda$。$g$ 从 $g(1)=0$ 开始单调增，所以浅盆地的峰要到更小的带宽 $\sigma_f<\sigma_c$ 才出现。$\sigma_f$ 由 $g(d_f)=\Delta C/\lambda$ 和 $\sigma_f^2=a^2/d_f^2-\lambda/A$ 决定。新峰出现的方式是：**一个峰和一个鞍点在远离深盆地的地方成对诞生**。

取 $a=1$、$A=20$、$\Delta C=2$、$\lambda=1$，得 $\sigma_c=0.975$，$\sigma_f=0.576$。在 $\sigma_f$ 附近，深盆地那个峰的 $M$ 只有约 0.13，对新峰的诞生毫无反应。新峰自己在诞生时 $M=1$（退化），随后迅速下降（图 1b）。

**温度起两个作用。**

- 决定盆地宽度 $s^2=\lambda/A$，使 $\sigma_c$ 略降。
- 决定非对称度 $\Delta C/\lambda$。升温让两峰质量更接近，折叠点向叉形点靠拢：$\lambda=0.5,1,2$ 时 $\sigma_f=0.49,0.58,0.64$，而 $\sigma_c\approx0.99,0.97,0.95$。

**极限 $\Delta C/\lambda\gg1$。** 由 $g(d)=2d^2-1-2\log(2d)+o(1)$，主导阶为

$$
\sigma_f^2\approx\lambda\Big(\frac{2a^2}{\Delta C}-\frac1A\Big).
$$

修正项随 $\Delta C/\lambda$ 只按对数衰减，并不小：$\Delta C/\lambda=20$ 时，实际的 $a^2/d_f^2$ 只有主导阶的约 80%，$\Delta C/\lambda=100$ 时约 94%。所以只能说：在这个极限下，浅峰出现时的带宽近似按 $\sqrt\lambda$ 缩放；沿 $\sigma^2\propto\lambda$ 的射线，浅峰是否可见近似与 $\lambda$ 无关。这与 Lopez Amado 等（2026）的两原子律一致：他们记两原子间距为 $\delta$、轻原子质量为 $\theta$，得到轻原子吸引子消失的尺度 $\sigma^\star\approx\delta/\sqrt{2\log(1/\theta)}$；取 $\delta=2a$、$\theta\approx e^{-\Delta C/\lambda}$，得 $\sigma^{\star2}\approx2a^2\lambda/\Delta C$，与上式相同。本节的数值例子 $\Delta C/\lambda\le4$ 远不在这个极限里：$\sigma_f$ 随 $\lambda$ 的增长远慢于 $\sqrt\lambda$，主导阶公式在 $\lambda=1$ 时给出 0.97，实际是 0.58。在这个范围内沿 $\sigma^2\propto\lambda$ 降温，浅峰反而更早可见。

{{< figure src="two-basin.svg" alt="一维双盆地的分叉图、沿分支的 M、以及 (λ, σ) 平面上的双峰区" caption=`一维双盆地，$a=1$，$A=20$。(a) $p_{\sigma,\lambda}$ 的临界点随带宽 $\sigma$ 的变化（$\lambda=1$）。灰色：$\Delta C=0$，在 $\sigma_c$ 处叉形分叉。彩色：$\Delta C=2$，深盆地的峰一路平滑延续，浅盆地的峰与一个鞍点在 $\sigma_f$ 处成对诞生（折叠）。(b) 沿各分支的 $M=\mathrm{Cov}[V\mid z]/\sigma^2$。对称情形下，追踪点的 $M$ 在 $\sigma_c$ 处穿过 1。非对称情形下，深盆地的 $M$ 在 $\sigma_f$ 附近只有约 0.13，只有浅峰自己在诞生时 $M=1$。$\sigma\to0$ 时任何峰的 $M$ 都趋于 1。(c) $\Delta C=2$ 时双峰区（阴影）在 $(\lambda,\sigma)$ 平面上的范围：升温使浅峰在更大的 $\sigma$ 处出现，降温使它推迟到更小的 $\sigma$。图由 tools/render_adaptive_mode_tracking.py 生成，脚本同时核对定理 2。` >}}

> [!note] 从例子得到的三点
> 1. $\sigma$ 是**分裂参数**，$\lambda$ 通过 $\Delta C/\lambda$ 充当**非对称参数**，同时改变盆地宽度。尖点突变（cusp）的控制平面是 $(\sigma,\Delta C/\lambda)$；固定 $\Delta C>0$ 时，$(\sigma,\lambda)$ 平面只覆盖其中一侧，只含一条折叠线，尖点本身要到 $\lambda\to\infty$ 才达到。这也只是两峰附近的局部描述：$\lambda$ 大到与势垒相当时两盆地会融成一个；温度也只改变 $e^{-\Delta C/\lambda}$ 这一部分，盆地宽窄不同带来的质量差（拉普拉斯近似下的 $\det A$ 之比）不随温度消失。
> 2. 只要两峰不对称，分叉就是折叠。正在追踪的峰看不到它，只有从新峰自己的分支（或从新峰附近出发）才能看到 $M=1$ 的信号。
> 3. 任何峰在 $\sigma\to0$ 时都有 $M\to I$，所以 $M$ 必须对照盆地内的基准（由上一级外推的 $M_{\mathrm{pred}}$，第 2.2 节）来读，而不能直接和 1 比。

## 3. 四个问题

### 3.1 分叉检测

**问题。** 已在带宽 $\sigma$ 上追踪若干个峰，把带宽降到 $\sigma'<\sigma$ 时，要判断三件事：有没有新峰出现，出现在哪里，沿哪个方向把粒子分开。反过来还要判断有没有峰消失：$D\ge2$ 时，降低 $\sigma$ 也可能让峰湮灭，峰的谱系不一定是一棵树。

**两类事件。**

| | 叉形（pitchfork） | 折叠（fold，saddle-node） |
|---|---|---|
| 何时出现 | 存在对称性（两峰质量、形状相同） | 一般情形 |
| 追踪点上的信号 | $\lambda_{\max}(M)$ 穿过 1，主特征向量给出分裂方向 | 没有信号；新峰与鞍点在远处成对诞生 |
| 新峰诞生时 | 与追踪点重合 | 自身 $M=1$，随后下降 |

Damon（1995）证明，高斯模糊下的一般奇点只有折叠一种，叉形需要对称性。机器人问题里两侧完全对称的情形（例如障碍物正对起点）恰恰是例外。

**难点。**

1. **折叠在局部看不见**，见第 2.4 节。
2. **蒙特卡洛噪声。** 用 $n$ 个有效样本估计 $D$ 维协方差时，即使真实协方差是 $\sigma^2I$，样本协方差的最大特征值也会被抬高到约 $(1+\sqrt{D/n})^2$ 倍（Marchenko–Pastur）。只有当真实的特征值"尖峰"超过约 $1+\sqrt{D/n}$ 时，样本协方差才能把它显现出来（Baik–Ben Arous–Péché 2005）。$D=50$、$n=100$ 时抬高约 2.9 倍，阈值 1 几乎处处触发；$D=22$、$n=128$ 时约 2.0 倍。重要性权重是重尾的，情况更糟。
3. **小 $\sigma$ 时问题病态**，见第 2.2 节第 3 点。
4. **高维的峰谱系不是树。** $D\ge2$ 时，高斯平滑本身也能制造新峰（Carreira-Perpiñán–Williams 2003），峰的路径还可能成环（Kuijper–Florack 2004）。

**对 S2R 的影响。** 若在分叉完成之前就从 sample 转入 refine，refine 会按初始噪声而不是按质量选边。S2R 的自洽带宽逐坐标取 $\sigma_j'^2=\min(c\,\mathrm{Var}_j,\sigma_j^2)=\min(c\,M_{jj},1)\,\sigma_j^2$，第 $j$ 个坐标只在 $M_{jj}<1/c$ 时缩小。分叉前 $M$ 接近 1，规则本身会让带宽变大，被上限截住后就停在原处不动。

### 3.2 温度调节

**$\lambda$ 的三个角色。**

1. **定义目标。** 峰的位置与 $\lambda$ 无关。记峰 $i$ 处的代价为 $c_i$、Hessian 为 $A_i$，在拉普拉斯近似下，峰 $i$ 与峰 $j$ 的质量比为 $e^{-(c_i-c_j)/\lambda}\sqrt{\det A_j/\det A_i}$。
2. **决定盆地宽度** $\sqrt{\lambda/A}$，从而移动分叉带宽（第 2.4 节）。
3. **决定 ESS 与收缩率。** 盆地内一切只依赖 $R=\sigma^2A/\lambda$（第 2.2 节）。

**问题。** sample、分叉、refine、计算质量这四个环节，各自该用什么温度？温度要不要随 $\sigma$ 变？温度改了之后，已有粒子的权重要不要修正？

**一个容易说错的点。** 取 $\lambda\propto\sigma^2$ 时 $R$ 不变，所以盆地内的 ESS 和收缩率不变；只有在 $\Delta C/\lambda\gg1$ 的极限下，浅峰是否可见才近似不变（第 2.4 节）。它**会改变峰之间的质量比**。分支分开之后，粒子不再在分支之间转移，所以此后改变温度不会改变各分支的粒子数；但这些粒子数反映的是分开时那个温度下的质量比，而不是目标温度下的。因此各峰的质量需要在一个固定的参考温度下单独重新估计。

### 3.3 sample 阶段的带宽

**问题。** 起点 $\sigma_{\max}$ 取多大，下一级 $\sigma'$ 取多少，每步注入多少新噪声，何时转入 refine。

**约束。**

- **起点**：$\sigma_{\max}$ 超过支撑半径（盒子为 $\sqrt D$）就保证单峰（第 2.1 节），这足以让追踪从任意点出发找到唯一的峰。搬运则不够：从 $\mathcal N(0,\sigma^2I)$ 起步还要求 $p_\sigma$ 接近这个高斯，即 $\sigma$ 远大于 $\pi_\lambda$ 的均值偏移与散布；否则起点的偏差会带进质量分配。另一种做法是像 Raya–Ambrogioni（2023）那样，从拟合的高斯（后验均值与协方差）起步。
- **步长**：$\sigma'$ 越接近 $\sigma$，每步越准，但步数越多。分叉只发生在很窄的带宽窗口里，固定日程在窗口外浪费步数，在窗口内步数又不够。
- **质量**：中等带宽处各峰互相重叠，质量信息就在这里传递（Dennehy 等 2026）。这一段走得太快，粒子按质量分配就会失真。
- **终点**：所有关心的峰都已分开之后才能转入 refine（第 3.1 节）。在 Dubins 单障碍任务的初步实验中，若 sample 在分叉之前（$\sigma=0.6$）就转入 refine，两种障碍偏移下左右两侧的选择比例分别为 58:42 与 11:89，比例由初始噪声决定，而不是由两侧的质量决定。

### 3.4 refine 阶段的带宽

**问题。** 追踪峰时用多大的 $\sigma$，如何缩小，何时停，以及何时可以开始。

**两个误差。**

- $\sigma$ 大：$p_\sigma$ 的峰偏离 $C$ 的局部极小，偏差是 $O(\sigma^2)$。盆地不对称时，宽的一侧会把峰拉过去，见[《MPPI 迭代、Mean Shift 与 EM》](/posts/mppi-meanshift-em/)第 5.5 节。
- $\sigma$ 小：$R\ll1$，$M\approx I$，收缩变慢，每步的位移被蒙特卡洛噪声淹没。

**自洽带宽。** 在一维二次盆地里，规则 $\sigma'^2=c\,\mathrm{Cov}$ 给出 $\sigma'^2=c\,\sigma^2/(1+R)$。它在 $R>c-1$ 时缩小，不动点是

$$
\sigma_*^2=(c-1)\,\lambda/A .
$$

- $c=1$ 时 $\sigma_*=0$，带宽一路塌缩，这就是 CEM 的协方差塌缩（[《SMC 与 CEM》](/posts/smc-cem/)第 3.4 节）。
- $c>1$ 时带宽停在与盆地宽度成比例的尺度。

剩下三个问题：$c$ 怎么选；停在 $\sigma_*$ 是否足够精确；以及这个规则必须在分叉完成后才能启动（第 3.1 节）。

**停止。** 位移 $\lVert m-z\rVert<\varepsilon\sigma$ 是 mean shift 的经典收敛准则。再加一道接受检查：候选点的代价不能比当前点差太多，防止一次坏的蒙特卡洛估计把点推进障碍物。

### 3.5 四个问题共用一组统计量

| 决策 | 读什么 | 规则示例 |
|---|---|---|
| 往哪走 | $m-z$ | mean shift / 降噪步 |
| 是否收敛 | $\lVert m-z\rVert/\sigma$ | 位移停止 |
| 是否分叉 | $M$ 对照由上一级外推的 $M_{\mathrm{pred}}$；新峰自身的 $M$ | 叉形用偏离检验；折叠需另找办法 |
| 分裂方向 | $M$ 的主特征向量 | 确定性退火式的分裂 |
| 下一级带宽 | $\mathrm{Cov}$、ESS | 自洽带宽；按 ESS 选步长 |
| 温度 | $R=\sigma^2A/\lambda$；峰之间的质量比 | $\lambda\propto\sigma^2$；分叉时升温 |
| 样本数 | ESS | ESS 过低则加倍 |

这张表说明四个决策能从同一组量读出，但它们不是"一个旋钮"：$\sigma$ 决定看多细，$\lambda$ 决定峰之间怎么分质量。

## 4. 相关文献：分类与分析

文献按它回答的问题分为六类：分叉何时发生（4.1）、在哪里发生（4.2）、发生之后怎么管理分支（4.3）、温度（4.4）、sample 带宽（4.5）、refine 带宽（4.6）。每篇写做法、结论或解决了什么，最后一段是这一类的分析。

### 4.1 分叉何时发生：理论刻画

**确定性退火（Rose、Gurewitz、Fox 1990；Rose 1998）。**
- **做法**：聚类时用 Gibbs 软分配 $p(y\mid x)\propto e^{-\lVert x-y\rVert^2/T}$，从高温逐步降温；聚类中心是加权平均的不动点。
- **结论**：温度降到 $T_c=2\lambda_{\max}(\Sigma_{x\mid y})$ 时中心失稳，沿主特征向量一分为二，其中 $\Sigma_{x\mid y}$ 是分给该中心的点的后验协方差。反复分裂就得到层级。它解决了聚类陷入坏局部极小的问题，并自然给出层级。
- **与我们的关系**：判据的结构与 $\lambda_{\max}(M)\ge1$ 相同（$T=2\sigma^2$），但后验不同。退火用的是簇内的软划分分布，mean shift 用的是以 $z$ 为中心的核加权分布，所以临界值不同：双高斯例子里，退火在 $T_c/2=a^2+s^2$ 处分裂，mean shift 在 $\sigma_c^2=a^2-s^2$ 处分裂。退火里叉形之所以是一般情形，是因为重合的中心被复制成多份，人为造出了对称性；黑箱目标没有这种对称性。

**扩散模型中的对称破缺与物种分化。**
- **Raya–Ambrogioni（NeurIPS 2023）**：把反向扩散写成随时间变化的势能中的带噪梯度流。中心不动点在临界噪声处失稳，以叉形分出通向各数据点的分支。实践做法叫 Gaussian Late Start：分叉之前的分布基本就是一个高斯，于是直接从拟合的高斯起步，跳过这段步数。少步数时 FID 明显下降（如 CelebA32 DDIM-10 从 11.37 降到 7.27），多样性也提高。分叉时间靠扫描 FID 找到，不是在线检测。
- **Biroli、Bonnaire、de Bortoli、Mézard（Nat. Commun. 2024）**：在维数与样本数都很大的极限下，把反向过程分成三个阶段：纯噪声、按类别分化（speciation）、塌到训练点（collapse）。VP 过程的分化时间是 $t_S=\tfrac12\log\Lambda$，$\Lambda$ 为数据协方差的最大特征值。它只描述第一次分裂。
- **Li–Chen（ICML 2024）；Sclocchi–Favero–Wyart（PNAS 2025）**：类别这样的特征只在很窄的时间窗口里被决定。前向—反向实验显示，类别在某个临界噪声处发生相变；数据的层级结构对应先后发生的多次分化。
- **Stančević–Ambrogioni（arXiv 2508.19897）**：给出三个等价的诊断量：条件熵产生率的峰、不动点处 Jacobian 特征值变号、路径后验熵方差的峰。对 VP 过程证明了方差峰在 $t_S=\tfrac12\log(\text{数据维数})$ 处变尖，与 Biroli 等的 $\tfrac12\log\Lambda$ 同类；VE 和 EDM 参数化下没有这个尖峰。
- **Sarkar（Score Shocks，arXiv 2604.07404）**：VE 过程的 score $s$ 经 $u=-2s$ 变换后满足黏性 Burgers 方程。该文的扩散时间 $\tau$ 与本章带宽的关系是 $\sigma^2=2\tau$。对称双高斯（分量方差 $s^2$）的分化时刻为 $\tau^*=(a^2-s^2)/2$，正是 $\sigma_c^2=a^2-s^2$；分界面宽度为 $v/a$（$v=s^2+\sigma^2$）。它还给出一个一般的局部判据：把密度拆成 $p=p^{(1)}+p^{(2)}$，令 $\varphi=\log(p^{(1)}/p^{(2)})$，score 写成两部分 score 的平均项 $\bar s$ 加上沿 $\nabla\varphi$ 的修正；记 $n$ 为分界面 $\{\varphi=0\}$ 的法向，若 $\partial_n\bar s_n+\lvert\nabla\varphi\rvert^2/4>0$，就出现双峰。作者建议在 $\tau^*$ 附近加密步长，但没有做实验。

**尺度空间的奇点理论。**
- **Damon（1995）**：给出热方程（高斯模糊）一般奇点的分类：只有折叠，即峰与鞍点成对湮灭或成对产生；叉形不是一般情形。这里的方向是 $\sigma$ 增大（模糊）：湮灭在任何维数都会出现，产生只在 $D\ge2$ 时出现。换成本章 $\sigma$ 减小的方向，就是新峰与鞍点成对诞生是常态（第 2.4 节的一维例子即是），而峰成对消失只在 $D\ge2$ 时出现。
- **Kuijper–Florack（IJCV 2004）**：用突变理论（包括尖点展开）分析几乎同时发生的产生与湮灭，指出峰的路径在尺度空间里可能成环，有的峰只在一段中间尺度上存在，因此无论从粗到细还是从细到粗，单向追踪都可能漏掉它们。
- **Carreira-Perpiñán–Williams（2003）**：高斯混合的峰数可以多于分量数；$D\ge2$ 时峰数不随带宽单调变化。
- **You（Density Evolution，arXiv 2606.00233）**：把 KDE 尺度空间、mode tree、SiZer、持久同调统一成"随参数演化的密度"。给出峰的速度公式 $\dot m=-[\nabla^2f]^{-1}\nabla\partial_tf$：Hessian 非奇异时峰光滑移动，分支事件只在 $\det\nabla^2f=0$ 处发生。还给出共享协方差高斯混合的对数凹判据（定理 7.2），由它可推出：支撑半径为 $r_B$ 时，$\sigma>r_B$ 必然对数凹。全文只有理论，没有实验。

**分析。** "分叉何时发生"在理论上已经很完整。分叉等价于平滑对数密度的 Hessian 在某处退化；对称情形有闭式的临界带宽。确定性退火、Biroli、Score Shocks 都把临界点与某种协方差的最大特征值联系起来，判据结构相同，但精确临界值不同：确定性退火与 Biroli 用的是整体数据协方差（双高斯为 $a^2+s^2$），mean shift 与 Score Shocks 的精确值是 $a^2-s^2$，只在 $a\gg s$ 时近似一致。缺口有两个：
- 扩散模型的主流理论（Raya–Ambrogioni、Biroli、Stančević–Ambrogioni）讨论的是对称情形或大维极限下的叉形。Lopez Amado 等（2026）已在扩散模型中指出吸引子以折叠方式消失，但只针对单个样本的临界尺度，没有与尺度空间的一般奇点分类系统地接上。
- 这些都是对数据分布或训练好的 score 做的分析，不是在线算法。

### 4.2 分叉在哪里：检测器

**From Modes to Memories（Lopez Amado、Fumero、Locatello，arXiv 2609.39648）。**
- **做法**：固定 $\sigma$，反复做 $x\leftarrow\hat x_\sigma(x)$，$\hat x_\sigma$ 为去噪器。精确去噪器下这就是 mean shift，$\hat x_\sigma(x)=x+\sigma^2\nabla\log p_\sigma(x)$。若前 16 步迭代中每一步与 $x$ 的距离都不超过 $0.25\lVert x\rVert$，就说 $x$ 在 $\sigma$ 处"被保留"。在 $\log\sigma$ 上二分，得到 $x$ 的临界尺度 $\sigma^\star(x)$。
- **结论**：在 $\sigma^\star$ 处吸引子以 saddle-node（折叠）方式消失；质量越大，$\sigma^\star$ 越大；两原子定理给出 $\sigma^\star\approx\delta/\sqrt{2\log(1/\theta)}$（$\delta$ 为间距，$\theta$ 为轻原子质量）。用它检测扩散模型的记忆化：CIFAR-10 上区分重复次数 $\ge8$ 的样本与只出现一次的样本，AUC 为 1.000。
- **分析**：这是扩散模型文献中少数**从峰自己的分支出发**、因而能看见折叠的检测器；经典尺度空间的由细到粗追踪、Silverman 的临界带宽也属于这一视角，deflation 则从另一侧去找不连通的解（第 4.3 节）。它需要先有候选点，依赖学到的去噪器，按样本逐个二分；没有温度，也不建谱系。

**Projection Caustics（Sakamoto–Sakamoto，arXiv 2606.13191）。**
- **做法**：把采样中突然的"选定某个峰"解释为最近点投影不再唯一的焦散面，提出检测器 $\mathrm{CBD}=\lVert\nabla_xu_t\rVert_F$，$u_t$ 为归一化的去噪方向，用有限差分估计。
- **结论**：CBD 与扰动敏感度的相关系数为 0.928。只在 CBD 标出的约 4% 的步上施加分类引导，分类准确率就达到 96%。
- **分析**：它定位的是"降噪方向转得最快"的窗口，相当于检测分叉窗口。它需要训练好的模型，并且是在基线采样之后才用，只能算准在线。

**Entropic Signature（Handke 等，arXiv 2602.09651）。**
- **做法**：跟踪类别后验的条件熵 $H(Z\mid X_t)$。它对时间的导数（熵产生率）在分化时刻出峰。类别后验由条件与无条件去噪器的重建误差递推估计。
- **分析**：信号清楚，但需要事先知道类别标签，而模态枚举恰恰不知道有哪些类。

**EigenScore（Shoushtari 等，ICLR 2026）。**
- **做法**：利用 $\mathrm{Cov}[x\mid x_t]=\sigma_t^2\nabla\hat x(x_t)$（$\hat x$ 为去噪器），用有限差分 JVP 加子空间迭代求后验协方差的前几个特征值，只需前向调用；用于分布外（OOD）检测。
- **分析**：对我们是技术工具。$D$ 很大时不必构造整个 $M$，用子空间迭代求 $\lambda_{\max}(M)$ 即可。

**Silverman（1981）；SiZer（Chaudhuri–Marron 1999）。**
- **做法**：Silverman 用临界带宽加平滑 bootstrap 检验峰的个数；SiZer 在所有尺度上检验导数过零点的显著性（同时置信区间）。
- **分析**：这是统计学里现成的思路，用来回答"噪声下看到的峰是不是真的"，可以借来校准 $\lambda_{\max}(M)$ 检验。

**分析。** 现有检测器要么依赖训练好的 score 或去噪器，要么（Silverman、SiZer）作用于数据的核密度估计；都在数据分布上工作，都没有处理"每一步只有几百个加权 rollout"时的估计噪声。能看见折叠的主要有两种视角：从候选峰出发、增大尺度看它在哪里消失；或者像 deflation 那样，排斥已知的解，去找不连通的新解。

### 4.3 分叉之后：分支管理与多解枚举

- **Mode tree 与尺度空间聚类（Minnotte–Scott 1993；Leung–Zhang–Xu 2000）**：把 KDE 的峰随带宽的变化画成树，按峰在多宽的尺度范围内存活来判断它是否可信。持久性聚类 ToMATo（Chazal 等 2013）按峰的显著度（与势垒高度相当）合并峰。分析：输出形式正是我们想要的，但处理的是数据样本，而且主要在低维。
- **Deflation（Farrell、Birkisson、Funke 2015）**：把残差乘上一个在已找到的解处趋于无穷的因子，Newton 法就回不到旧解，只能去找新解。它能找到与已知分支不连通的解，并能与延拓结合画出分叉图。分析：这正是处理"折叠诞生、与追踪分支不连通"的工具，但需要 Jacobian。把它改成作用在 $m(z)-z$ 上的无导数版本，目前没有人做过。
- **P-HO（Pardis、Chignoli、Kim，IROS 2024）**：在问题参数（质量、力限等）空间上做 RRT 式的树搜索，每个参数保留多个局部解，以此绕过同伦路径上的折叠、分叉与不连通。cart-pole 上成功率约 80%，线性同伦约 10%（从图上读出的近似值）。分析：说明机器人轨迹优化里确实存在分支结构，但它处理分叉的办法是"多存几支"，并不检测分叉。
- **黑箱 niching（HillVallEA，Maree 等 2018）**：在两个解的连线上取点，检查中间有没有势垒，以此把种群分到不同盆地，再在每个盆地里局部优化。分析：这个势垒检验可以直接拿来合并重复的峰。
- **机器人里的 rollout 聚类。**
  - Clustered MPPI（Patrick–Bakolas，ACC 2024）对每条 rollout 的（扰动, 代价）向量做 DBSCAN，在每簇内做 MPPI，执行代价最低的簇；Dubins 任务 1000 张地图上碰撞 6 次，MPPI 为 11 次。
  - CE-MPPI（Liu、Chang、Chen，arXiv 2607.06499）用"终点相对于碰撞 rollout 平均终点的方向"作特征来聚类，真机 UR5e 上耗时比 MPPI 少 48%。
  - BiC-MPPI（Jung–Kim 2024）保留所有簇并行精修。
  - OT-MPC（Pacelli、Ratheesh、Theodorou，arXiv 2605.02147）用 Sinkhorn 最优传输把每个粒子移到附近低代价提议的重心，相当于带核的 mean shift；Push-T 成功率 76%，MPPI 为 4%。作者指出核宽 $\varepsilon$ 的自适应调度尚未解决。

  分析：这些方法都在单一尺度上工作（前三者一次性聚类，OT-MPC 以固定核宽做粒子精修），没有带宽延拓，没有分叉检测，也不给质量。OT-MPC 是 PushT 上必须比较的基线。

### 4.4 温度

**Probabilistic Gaussian Homotopy（Gal、Wu Fung、Haber，arXiv 2603.13546）。**
- **做法**：平滑 Boltzmann 密度而不是平滑目标函数，$p_t(x)\propto\mathbb E_z[e^{-f(\alpha x+\beta z)/\lambda}]$。标准耦合取 $\beta=\sqrt\lambda$，即 $\sigma^2=\lambda$。用自归一化权重的蒙特卡洛梯度做梯度下降，$\lambda$ 与 $\sigma$ 按线性日程一起降到 0。
- **结论**：此时 $-\lambda\log p_t$ 是有限温度的 Moreau 包络，并且 $\mathbb E[y\mid x]=x+\lambda\nabla\log p_t(x)$，这正是 $\sigma^2=\lambda$ 时的 Tweedie。在 10 维 Ackley、Griewank、Alpine1 上，达到成功阈值所需的次数（论文报告的计数）比 CMA-ES 少；在 Levy 上更差。
- **分析**：它给出了 $\sigma^2\propto\lambda$ 这个耦合和理由，但只走单条路径，需要梯度，日程固定，不枚举峰。

**TSR（Xu 等，ICML 2026，arXiv 2510.01184）。**
- **做法**：在每个噪声级把 score 乘以 $r(\sigma)=(s^2+\sigma^2)/(s^2/k+\sigma^2)$，其中 $s$ 为假定的峰宽。大 $\sigma$ 时 $r\approx1$，峰之间的分配不变；小 $\sigma$ 时 $r\approx k$，每个峰的方差缩小 $k$ 倍（相当于对该峰取 $p^k$）。
- **结论**：不需训练就能做局部降温，并保持各峰的均值与权重。（我们的推论：$k\to\infty$ 时，对高斯峰一步去噪直接跳到峰中心，相当于 mean shift。）
- **分析**：权重之所以保得住，是因为只在峰已经分开之后才降温，与"分叉之后再降温"是同一个思路。但 $s$ 和日程都是固定的超参数，不由统计量决定。

**FKC（Skreta 等，ICML 2025）与 PITA（Akhound-Sadegh 等，NeurIPS 2025）。**
- **做法**：把"沿噪声路径同时改温度"写成 Feynman–Kac 偏微分方程，用带权 SDE 加 SMC 重采样，精确采样降温后的分布。PITA 先在高温下训练扩散模型，推理时沿固定的温度梯子逐级降温，再蒸馏。
- **结论**：FKC 在 LJ-13 上明显优于不修正的版本。作者称 PITA 是第一个扩展到 Cartesian 坐标肽链的扩散采样器。
- **分析**：它们给出了改温度时权重的精确增量，可以拿来检验我们的做法。但温度梯子是固定的；而且权重退化会损失多样性（FKC 的图像实验里最终样本几乎相同），这对保留多个峰不利。

**MAD-Path（Chen、Liu、Yang，arXiv 2607.11631）。**
- **做法**：用扩散（OU）路径代替温度路径做 MCMC，在路径空间加 Metropolis 校正。
- **结论**：扩散路径精确保持混合权重，温度路径会扭曲不对称峰之间的权重。在 20 维不等方差双高斯上，只有它恢复出了正确权重。
- **分析**：这支持"以 $\sigma$ 为主延拓变量、$\lambda$ 慎动"。它需要先找到各个峰来构造参考混合，而这一步正是模态枚举。

**Dennehy 等（arXiv 2607.15485）。**
- **结论**：峰之间的质量信息存在于中等噪声处。在 MNIST 潜空间的 1/8 两类混合（真实权重 0.4）上，标准日程的权重误差约 1%；把日程的时间支撑压缩后，相对误差升到 31%，而样本在视觉上相似。
- **分析**：sample 日程不能跳过中等带宽；质量估计也应在峰已分开、但仍有重叠的尺度上进行。

**分析。** 温度方面已有两类耦合规则（全局的 $\sigma^2\propto\lambda$；分叉后的局部降温），有改温度时的精确修正，也已知温度路径会扭曲质量。还没有人从统计量自适应地选 $\lambda$。第 2.4 节的例子提示了一个可做的问题：在分叉窗口里升温，能让浅峰更早出现、两峰更接近对称；但升温也会加宽盆地、推迟分叉。最优的追踪温度取决于这两者的折中。

### 4.5 sample 阶段的带宽

- **固定日程。** Song–Ermon（2020）建议 $\sigma_{\max}$ 取数据点两两距离的最大值，对盒子是 $2\sqrt D$。EDM（Karras 等 2022）在 $\sigma^{1/\rho}$ 上等距取点，$\rho=7$，使步数集中在小 $\sigma$。MBD（Pan 等 2024）、DIAL-MPC（Xue 等 2025）在机器人上沿手工日程退火。分析：由第 2.1 节的对数凹条件，$\sqrt D$ 已足以保证单峰，但搬运还要求起点分布接近 $p_{\sigma_{\max}}$（第 3.3 节）；日程的形状与分叉窗口的位置无关。
- **在分叉前直接起步（Raya–Ambrogioni 2023）。** 见第 4.1 节：分叉之前的阶段可以用一个拟合的高斯代替。分析：这对应"用后验均值与协方差拟合一个高斯，从它起步"的做法，但他们的分叉时间是离线扫出来的。
- **按 ESS 自适应选温度（Jasra 等 2011；Del Moral、Doucet、Jasra 2012；Zhou、Johansen、Aston 2016；Goshtasbpour 等 2023）。**
  - 做法：用二分法选下一个温度（Del Moral 等 2012 中是 ABC 的容差），使增量权重的 ESS（或条件 ESS）等于目标比例，或使它以恒定速率下降。
  - 结论：日程自动在难走的地方变密。
  - 分析：这个思路可以搬到带宽轴上：用二分法选 $\sigma'$，使某个统计量变化一个固定量，比如新提议的 ESS、$M$ 的变化、责任度的熵。但在扩散式的搬运里，相邻两级带宽之间没有现成的增量权重，统计量需要重新设计。
- **在分叉窗口加密（Score Shocks；CBD；Li–Chen）。** 理论建议在 $\tau^*$ 附近加密步长；CBD 显示只有少数步真正重要。分析：都是建议或事后诊断，没有在线选择带宽的规则。

**分析。** 关于 sample 带宽，有两件事已经明确：起点要超过支撑半径，且起点分布要接近 $p_{\sigma_{\max}}$；步数应集中在分叉窗口和中等带宽。还没解决的是在线判断"现在是否在窗口里"，这又回到第 4.2 节的检测问题。

### 4.6 refine 阶段的带宽

- **退火 mean shift（Shen、Brooks、van den Hengel 2005/2007）**：用一串递减的带宽做 mean shift，把带宽当温度，经验上常能到达全局峰（没有保证）。日程固定，只追一个峰。
- **可变带宽 mean shift（Comaniciu、Ramesh、Meer 2001；Comaniciu 2003）**：每个峰的带宽由它的局部协方差决定，取跨尺度最稳定的那个估计。分析：与自洽带宽同源，都利用"后验协方差由提议方差与盆地方差组合而成"这一高斯恒等式；但它处理的是数据 KDE。
- **延拓式找峰（Pulkkinen、Mäkelä、Karmitsa 2013）**：把峰随尺度变化的路径写成 ODE，用信赖域预测—校正法追踪，只追一支。第 4.1 节的峰速度公式就是它的预测步。
- **单循环 Gaussian homotopy（Iwakiri 等，NeurIPS 2022）**：在一个循环里同时更新迭代点和平滑尺度，也有零阶版本。
- **SVG-MPPI（Honda 等，ICRA 2024）**：用少量 SVGD 粒子把分布推向一个峰，对推送路径拟合高斯来估计这个峰的方差，再用它作 MPPI 的采样协方差。分析：这是机器人里最接近"按峰的后验方差设采样宽度"的做法。
- **CEM 的协方差更新**：用精英样本的协方差作为下一轮的协方差，相当于 $c=1$ 的自洽带宽，因此会塌缩（[《SMC 与 CEM》](/posts/smc-cem/)第 3.4 节）。

**分析。** refine 带宽的各种规则在别的领域都出现过。自洽带宽取 $c>1$ 时，不动点与盆地宽度成比例（第 3.4 节）。还没解决两点：何时可以开始（要求分叉已经完成）；以及停在 $\sigma_*$ 与所需精度之间是什么关系。

## 5. 尚未解决的问题

1. **只用 rollout 检测折叠。** 文献中能看见折叠的视角主要是"从候选峰出发、增大尺度看它何时消失"和 deflation，二者都没有黑箱、只用 rollout 的版本。可能的做法：
   - $M$ 偏离外推基准 $M_{\mathrm{pred}}$ 时，沿主特征向量 $v_1$ 主动探测 $z\pm\alpha\sigma v_1$；
   - 用 sample 阶段的粒子群作为候选峰的来源；
   - 把 deflation 改成作用在 $m(z)-z$ 上的无导数版本；
   - 监测权重分布是否双峰。
2. **有限样本下的分裂检验。** 用交叉拟合（一半样本估主方向，另一半估 Rayleigh 商）和 bootstrap 置信界，并给出所需的 ESS 下界，大致是 $n\gtrsim D/(\text{尖峰}-1)^2$ 的形式。
3. **$(\sigma,\lambda)$ 平面上的检测图。** 对非对称双盆地给出叉形线、折叠线，以及追踪点上的检验能看到分裂的区域（第 2.4 节已有一维闭式），由此得到分叉窗口里追踪温度的选法。
4. **按统计量在线选 sample 带宽。** 用二分法选 $\sigma'$，使统计量变化固定量；在分叉窗口与中等带宽处自动加密。
5. **refine 何时开始、何时停。** 与分叉检测衔接：每个分支确认已经分开后，再进入自洽收缩；停止准则与所需精度挂钩。
6. **盒边界上的峰。** 饱和控制会让峰落在 $B$ 的边界上。这些是 KKT 点而不是光滑极大，曲率检验和拉普拉斯质量都要按截断高斯修正。
7. **评价方式。** 指标用"质量不小于 $\varepsilon$ 的峰的召回率随 rollout 数的曲线"以及质量误差；对比对象是同等预算下的多起点加聚类、OT-MPC 和 Stein 类方法。

## 6. 文献

1. K. You. Density evolution: a multiscale view of density estimation. arXiv:2606.00233, 2026.
2. M. Á. Carreira-Perpiñán. Gaussian mean-shift is an EM algorithm. *IEEE TPAMI*, 2007.
3. C. A. Robertson, J. G. Fryer. Some descriptive properties of normal mixtures. *Skandinavisk Aktuarietidskrift*, 1969.
4. C. Lopez Amado, M. Fumero, F. Locatello. From modes to memories: characterizing the scale-space dynamics of diffusion models. arXiv:2609.39648, 2026.
5. J. Damon. Local Morse theory for solutions to the heat equation and Gaussian blurring. *J. Differential Equations*, 1995.
6. V. A. Marčenko, L. A. Pastur. Distribution of eigenvalues for some sets of random matrices. *Math. USSR-Sbornik*, 1967.
7. J. Baik, G. Ben Arous, S. Péché. Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. *Ann. Probab.*, 2005.
8. M. Á. Carreira-Perpiñán, C. K. I. Williams. On the number of modes of a Gaussian mixture. *Scale-Space*, 2003.
9. A. Kuijper, L. M. J. Florack. The relevance of non-generic events in scale space models. *IJCV*, 2004.
10. A. Dennehy, R. Muthukumar, R. Willett, N. Chandramoorthy. Diffusion models recover accurate mixture weights despite score function insensitivity. arXiv:2607.15485, 2026.
11. K. Rose, E. Gurewitz, G. C. Fox. Statistical mechanics and phase transitions in clustering. *Phys. Rev. Lett.*, 1990.
12. K. Rose. Deterministic annealing for clustering, compression, classification, regression, and related optimization problems. *Proc. IEEE*, 1998.
13. G. Raya, L. Ambrogioni. Spontaneous symmetry breaking in generative diffusion models. *NeurIPS*, 2023.
14. G. Biroli, T. Bonnaire, V. de Bortoli, M. Mézard. Dynamical regimes of diffusion models. *Nat. Commun.*, 2024.
15. M. Li, S. Chen. Critical windows: non-asymptotic theory for feature emergence in diffusion models. *ICML*, 2024.
16. A. Sclocchi, A. Favero, M. Wyart. A phase transition in diffusion models reveals the hierarchical nature of data. *PNAS*, 2025.
17. D. Stančević, L. Ambrogioni. The information dynamics of generative diffusion. arXiv:2508.19897, 2025.
18. K. Sarkar. Score shocks: the Burgers equation structure of diffusion generative models. arXiv:2604.07404, 2026.
19. R. Sakamoto, K. Sakamoto. The geometry of phase transitions in generative dynamics via projection caustics. arXiv:2606.13191, 2026.
20. F. Handke, D. Stančević, F. Koulischer, T. Demeester, L. Ambrogioni. The entropic signature of class speciation in diffusion models. *ICML*, 2026 (arXiv:2602.09651).
21. S. Shoushtari, Y. Wang, X. Shi, M. S. Asif, U. S. Kamilov. EigenScore: OOD detection using posterior covariance in diffusion models. *ICLR*, 2026.
22. B. W. Silverman. Using kernel density estimates to investigate multimodality. *JRSS-B*, 1981.
23. P. Chaudhuri, J. S. Marron. SiZer for exploration of structures in curves. *JASA*, 1999.
24. M. C. Minnotte, D. W. Scott. The mode tree: a tool for visualization of nonparametric density features. *JCGS*, 1993.
25. Y. Leung, J.-S. Zhang, Z.-B. Xu. Clustering by scale-space filtering. *IEEE TPAMI*, 2000.
26. F. Chazal, L. J. Guibas, S. Y. Oudot, P. Skraba. Persistence-based clustering in Riemannian manifolds. *J. ACM*, 2013.
27. P. E. Farrell, Á. Birkisson, S. W. Funke. Deflation techniques for finding distinct solutions of nonlinear partial differential equations. *SIAM J. Sci. Comput.*, 2015.
28. S. Pardis, M. Chignoli, S. Kim. Probabilistic homotopy optimization for dynamic motion planning. *IROS*, 2024.
29. S. C. Maree, T. Alderliesten, D. Thierens, P. A. N. Bosman. Real-valued evolutionary multi-modal optimization driven by hill-valley clustering. *GECCO*, 2018.
30. S. Patrick, E. Bakolas. Path integral control with rollout clustering and dynamic obstacles. *ACC*, 2024.
31. Z. Liu, K. Chang, X. Chen. Clustering-embedded model predictive path integral control: avoiding averaging-induced failure. arXiv:2607.06499, 2026.
32. M. Jung, K.-K. K. Kim. BiC-MPPI: goal-pursuing, sampling-based bidirectional rollout clustering path integral. arXiv:2410.06493, 2024.
33. V. Pacelli, A. Ratheesh, E. A. Theodorou. Sampling-based control via entropy-regularized optimal transport. arXiv:2605.02147, 2026.
34. E. Gal, S. Wu Fung, E. Haber. Probabilistic Gaussian homotopy: a probability-space continuation framework for nonconvex optimization. arXiv:2603.13546, 2026.
35. Y. Xu, Y. Wu, S. Park, Z. Zhou, S. Tulsiani. Temporal score rescaling for temperature sampling in diffusion and flow models. *ICML*, 2026 (arXiv:2510.01184).
36. M. Skreta et al. Feynman–Kac correctors in diffusion: annealing, guidance, and product of experts. *ICML*, 2025.
37. T. Akhound-Sadegh et al. Progressive inference-time annealing of diffusion models for sampling from Boltzmann densities. *NeurIPS*, 2025.
38. H. Chen, S. Liu, J. Yang. Markov chain Monte Carlo with diffusion paths. arXiv:2607.11631, 2026.
39. Y. Song, S. Ermon. Improved techniques for training score-based generative models. *NeurIPS*, 2020.
40. T. Karras, M. Aittala, T. Aila, S. Laine. Elucidating the design space of diffusion-based generative models. *NeurIPS*, 2022.
41. C. Pan, Z. Yi, G. Shi, G. Qu. Model-based diffusion for trajectory optimization. *NeurIPS*, 2024.
42. H. Xue, C. Pan, Z. Yi, G. Qu, G. Shi. Full-order sampling-based MPC for torque-level locomotion control via diffusion-style annealing. *ICRA*, 2025.
43. A. Jasra, D. A. Stephens, A. Doucet, T. Tsagaris. Inference for Lévy-driven stochastic volatility models via adaptive sequential Monte Carlo. *Scand. J. Stat.*, 2011.
44. P. Del Moral, A. Doucet, A. Jasra. An adaptive sequential Monte Carlo method for approximate Bayesian computation. *Stat. Comput.*, 2012.
45. Y. Zhou, A. M. Johansen, J. A. D. Aston. Toward automatic model comparison: an adaptive sequential Monte Carlo approach. *JCGS*, 2016.
46. S. Goshtasbpour, V. Cohen, F. Perez-Cruz. Adaptive annealed importance sampling with constant rate progress. *ICML*, 2023.
47. C. Shen, M. J. Brooks, A. van den Hengel. Fast global kernel density mode seeking with application to localisation and tracking. *ICCV*, 2005; 期刊版 Fast global kernel density mode seeking: applications to localization and tracking. *IEEE TIP*, 2007.
48. D. Comaniciu, V. Ramesh, P. Meer. The variable bandwidth mean shift and data-driven scale selection. *ICCV*, 2001.
49. D. Comaniciu. An algorithm for data-driven bandwidth selection. *IEEE TPAMI*, 2003.
50. S. Pulkkinen, M. M. Mäkelä, N. Karmitsa. A continuation approach to mode-finding of multivariate Gaussian mixtures and kernel density estimates. *J. Global Optim.*, 2013.
51. H. Iwakiri, Y. Wang, S. Ito, A. Takeda. Single loop Gaussian homotopy method for non-convex optimization. *NeurIPS*, 2022.
52. K. Honda et al. Stein variational guided model predictive path integral control: proposal and experiments with fast maneuvering vehicles. *ICRA*, 2024.
