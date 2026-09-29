---
title: "SMC 与 CEM：从自适应重要性采样看两类群体算法"
date: 2026-09-29
description: "序贯蒙特卡洛（SMC）与交叉熵方法（CEM）各自面对的问题、设定与求解过程，含全部推导和手算例子；以及二者在自适应重要性采样框架下的对比与统一。"
summary: "SMC 与 CEM 各自的问题、设定、算法与推导；二者都是“采样—加权—更新提议”的迭代，差别在群体如何被总结（粒子 vs 参数族）和目标序列走向哪里。"
tags: ["SMC", "CEM", "重要性采样", "ESS", "MPPI"]
---

本章自成一体。记号沿用[《MPPI 迭代、Mean Shift 与 EM》](/posts/mppi-meanshift-em/)：代价 $C(V)$，温度 $\lambda$，Gibbs 因子 $\gamma(V)=e^{-C(V)/\lambda}$，Gibbs 分布 $\pi=\gamma/Z$，高斯 $\mathcal N(V;U,\Sigma)$。第 2 节讲 SMC，第 3 节讲 CEM，第 4 节对比与统一。

## 1. 结论

> [!conclusion] 结论
> SMC 与 CEM 每一轮做的是同一件事：**从当前提议采样 → 按适合度重加权 → 用重加权后的群体构造下一个提议**。二者都属于自适应重要性采样，差别只在两个轴上：
>
> | | SMC | CEM |
> |---|---|---|
> | 重加权后的群体怎么总结 | 原样保留为带权粒子，重采样 + MCMC 移动 | 压缩成参数族（高斯）的均值、协方差，其余信息丢弃 |
> | 权重的形状 | 增量幂 $\gamma^{\Delta\beta}$（温度桥） | 指示函数 $\mathbb{1}\{C\le\gamma_t\}$（阈值桥） |
> | 自适应机制 | 二分选温度使 ESS $=N/2$ | 固定 elite 比例 $\rho$，阈值取分位数 |
> | 目标序列的终点 | 固定目标 $\pi$（采样、积分、归一化常数） | 退化为最优点上的 Dirac（优化） |
>
> 两种自适应本质相同：**目标只能变尖到群体仍能代表它的程度**。SMC 用粒子携带多模态，能按质量分配；CEM 用单峰族总结群体，只能塌缩到一侧——这正是 CEM 与 MPPI 共有的 mode averaging。

## 2. SMC：序贯蒙特卡洛

### 2.1 它面对的问题

**起源。** 粒子滤波（Gordon 等 1993）：状态 $x_t$ 随时间演化，观测 $y_t$ 逐个到来，要在线估计后验 $p(x_t\mid y_{1:t})$。后验没有闭式，又要随每个新观测更新。

**推广。** Del Moral–Doucet–Jasra (2006) 把它推广为通用采样器：只要能构造一串从简单到复杂的分布 $\pi_0\to\pi_1\to\dots\to\pi_T$，就能用同一套机制从最终的 $\pi_T$ 采样。本章讨论这个静态版本。

**设定。** 目标 $\pi_T(x)\propto\gamma_T(x)$，只能评估未归一化密度 $\gamma_T$（例如 $e^{-C(x)/\lambda}$，评估一次等于一次 rollout），要得到它的样本、期望 $\mathbb E_{\pi_T}[f]$，以及归一化常数 $Z_T=\int\gamma_T$。

### 2.2 直接重要性采样为什么失败：权重退化与 ESS

从简单提议 $q$ 采 $N$ 个样本，权重 $w_i\propto\gamma_T(x_i)/q(x_i)$，估计 $\mathbb E_{\pi_T}[f]\approx\sum_iw_if(x_i)$。若 $q$ 与 $\pi_T$ 相差很远（高维下几乎必然），权重会极端不均：几乎全部质量落在一两个样本上。衡量这件事的仪表是**有效样本数**。

> [!definition] 定义（ESS）
> 对归一化权重 $w_1,\dots,w_N$（$w_i\ge0$，$\sum_iw_i=1$），
> $$
> \mathrm{ESS}=\frac{1}{\sum_{i=1}^Nw_i^2},\qquad 1\le\mathrm{ESS}\le N .
> $$

**为什么它叫"有效样本数"。** 把权重视为常数，$f(x_i)$ 独立同分布、方差 $\sigma^2$，则

$$
\mathrm{Var}\Big(\sum_iw_if(x_i)\Big)=\sum_iw_i^2\,\mathrm{Var}\big(f(x_i)\big)=\sigma^2\sum_iw_i^2 ,
$$

交叉项因独立而为零。$n$ 个等权样本的均值方差是 $\sigma^2/n$，令两者相等得 $n=1/\sum_iw_i^2=\mathrm{ESS}$：这组带权样本的精度相当于 ESS 个等权样本。（权重实际依赖样本，这是 Kish 的近似，作为诊断足够。）

**两端的界。** 由 $0\le w_i\le1$ 有 $w_i^2\le w_i$，求和得 $\sum w_i^2\le1$，即 $\mathrm{ESS}\ge1$，取等当且仅当一个权重为 1；由 Cauchy–Schwarz，$1=(\sum_iw_i\cdot1)^2\le(\sum_iw_i^2)(\sum_i1^2)=N\sum_iw_i^2$，即 $\mathrm{ESS}\le N$，取等当且仅当权重全等于 $1/N$。

**ESS 的两端各有含义。** ESS $\approx1$：只信一个样本，估计噪声极大；ESS $\approx N$：权重几乎均匀，加权均值 $\approx$ 提议均值，权重没有提供信息。健康区间在中间。

### 2.3 造桥：幂（温度）序列

SMC 的想法是**不一步跳过去，而是走过去**：插入一串中间目标，相邻两个足够近，使每一步的权重都温和。最常用的桥是幂：

$$
\pi_\beta\propto\gamma_T^{\beta},\qquad 0=\beta_0<\beta_1<\dots<\beta_T=1 ,
$$

或带起点分布的版本 $\pi_\beta\propto\pi_0^{1-\beta}\gamma_T^{\beta}$。

**幂就是温度。** $\gamma_T^\beta=e^{-\beta C/\lambda}=e^{-C/(\lambda/\beta)}$，所以 $\pi_\beta$ 是温度 $\lambda/\beta$ 的 Gibbs 分布：$\beta=0$ 是均匀分布（温度无穷大），$\beta$ 从 0 升到 1 就是温度从 $\infty$ 降到 $\lambda$，即退火。任何正密度都可写成 $e^{-E}$，取幂即给能量乘系数，因此这个桥对一般目标同样适用。

**为什么用幂。** 一是增量权重简单：从 $\pi_{\beta_t}$ 到 $\pi_{\beta_{t+1}}$，

$$
\frac{\gamma_T(x)^{\beta_{t+1}}}{\gamma_T(x)^{\beta_t}}=\gamma_T(x)^{\beta_{t+1}-\beta_t}=e^{-(\beta_{t+1}-\beta_t)\,C(x)/\lambda},
$$

只需评估一次 $C(x)$，且 $\Delta\beta$ 越小权重越温和，这是自适应选 $\Delta\beta$ 的基础。二是对数空间是直线，$\log\pi_\beta=\beta\log\gamma_T+\text{const}$，模态位置不动，只是峰逐渐变尖。三是 $\beta=0$ 端可以直接采样。

### 2.4 算法：重加权、重采样、移动

维护 $N$ 个带权粒子 $\{x^{(i)},w^{(i)}\}$，从 $\pi_0$ 采样起步，权重 $1/N$。每一步 $\pi_{\beta_t}\to\pi_{\beta_{t+1}}$：

**（i）重加权。** 粒子不动，$w^{(i)}\leftarrow w^{(i)}\,\gamma_T(x^{(i)})^{\beta_{t+1}-\beta_t}$，再归一化。理由：若 $\{x^{(i)},w^{(i)}\}$ 代表 $\pi_{\beta_t}$，则对任意 $f$，
$$
\mathbb E_{\pi_{\beta_{t+1}}}[f]=\frac{\int f\,\gamma_T^{\beta_{t+1}}}{\int\gamma_T^{\beta_{t+1}}}
=\frac{\int f\,\gamma_T^{\Delta\beta}\,\gamma_T^{\beta_t}}{\int\gamma_T^{\Delta\beta}\,\gamma_T^{\beta_t}}
=\frac{\mathbb E_{\pi_{\beta_t}}\big[f\,\gamma_T^{\Delta\beta}\big]}{\mathbb E_{\pi_{\beta_t}}\big[\gamma_T^{\Delta\beta}\big]}
\approx\frac{\sum_iw^{(i)}\gamma_T(x^{(i)})^{\Delta\beta}f(x^{(i)})}{\sum_iw^{(i)}\gamma_T(x^{(i)})^{\Delta\beta}} .
$$

**（ii）重采样（ESS 低于阈值时，常取 $N/2$）。** 按权重有放回地抽 $N$ 个索引，得到新粒子集，权重重置为 $1/N$：高权重粒子被复制多份，低权重粒子被淘汰。设第 $i$ 个粒子被复制 $n_i$ 次，任何合法的重采样方案都满足 $\mathbb E[n_i]=Nw^{(i)}$，于是
$$
\mathbb E\Big[\frac1N\sum_in_if(x^{(i)})\Big]=\sum_iw^{(i)}f(x^{(i)}) :
$$
重采样后的等权平均，其期望正是重采样前的加权平均。**它不改变分布，只把"质量集中在少数粒子"换成"多数粒子集中在高质量区域"**；代价是一次额外的抽样噪声，所以 ESS 高时不做。

**系统重采样**把这份噪声压到最小：把权重首尾相接排成长度为 1 的线段（累积权重是各段右端点），抽一个 $u\sim U(0,1/N)$，把 $N$ 根等间距的"梳齿"插在 $u,u+\tfrac1N,\dots,u+\tfrac{N-1}{N}$ 处，梳齿落在哪一段就复制哪个粒子。因为齿距固定为 $1/N$，长度 $w_i$ 的段里最少落 $\lfloor Nw_i\rfloor$ 根、最多 $\lceil Nw_i\rceil$ 根，随机性只剩取整方向，且 $\mathbb E[n_i]=Nw_i$ 仍成立。

**（iii）移动。** 重采样后有许多完全相同的复制品，群体多样性下降。对每个粒子做一步以 $\pi_{\beta_{t+1}}$ 为不变分布的 MCMC（随机游走 Metropolis：提议 $x'=x+\tau\xi$，以概率 $\min\{1,\pi_{\beta_{t+1}}(x')/\pi_{\beta_{t+1}}(x)\}$ 接受）。MCMC 核保 $\pi_{\beta_{t+1}}$ 不变，所以这一步不改变分布，只恢复多样性。**移动必须是保分布的 MCMC，不能是往低代价推的优化步，否则引入偏差。**

**自适应温度。** $\beta_{t+1}$ 不预设，而是用二分法解"增量权重的 ESS 等于 $N/2$"：目标能变多尖，取决于粒子还能不能代表它。严格说这使温度序列依赖粒子，破坏了精确无偏性；已证明随 $N\to\infty$ 相合（Beskos 等 2016），实践中普遍采用。

**副产品：归一化常数。** 由（i）的分母，$Z_{\beta_{t+1}}/Z_{\beta_t}=\mathbb E_{\pi_{\beta_t}}[\gamma_T^{\Delta\beta}]\approx\sum_iw^{(i)}\gamma_T(x^{(i)})^{\Delta\beta}$，连乘得到 $Z_T/Z_0$ 的估计（固定温度序列时无偏）。这是 MCMC 做不到的。

### 2.5 手算例子

一维双阱代价 $C(x)=(x^2-1)^2$，极小点 $\pm1$，中间势垒高 1；$\lambda=0.05$，目标很尖。$N=8$ 个粒子从 $\pi_0=U[-2,2]$ 抽出，权重各 $1/8$：

| 粒子 $i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| $x_i$ | −1.9 | −1.3 | −0.9 | −0.4 | 0.2 | 0.7 | 1.1 | 1.6 |
| $C(x_i)$ | 6.81 | 0.48 | 0.04 | 0.71 | 0.92 | 0.26 | 0.04 | 2.43 |

从 $\beta=0$ 走到 $\Delta\beta$，权重 $w_i\propto e^{-\Delta\beta\,C(x_i)/\lambda}$。

**ESS 高（$\Delta\beta=0.02$，温度 2.5）**：$w=(0.012,0.148,0.176,0.135,0.124,0.161,0.176,0.068)$，$\mathrm{ESS}=6.74$。8 个粒子几乎都在起作用，不重采样。

**ESS 低（$\Delta\beta=1$，一步跳到目标）**：$w=(0,0,0.537,0,0,0.006,0.457,0)$，$\mathrm{ESS}=2.01$。只有离 $\pm1$ 最近的两个粒子有分量，其余白算——这就是直接重要性采样退化的样子。

**自适应选 $\Delta\beta$（目标 ESS $=4$）**，二分法：

| 尝试 $\Delta\beta$ | 0.500 | 0.250 | 0.125 | 0.1875 | 0.156 | 0.141 | 0.133 |
|---|---|---|---|---|---|---|---|
| ESS | 2.24 | 2.93 | 4.15 | 3.38 | 3.72 | 3.92 | 4.03 |
| 判断 | 太尖 | 太尖 | 可以 | 太尖 | 太尖 | 太尖 | 可以 |

收敛到 $\Delta\beta\approx0.135$，$w=(0.000,0.099,0.324,0.053,0.030,0.177,0.317,0.001)$，$\mathrm{ESS}=4.00$：这一步只把目标变尖到"8 个粒子里还有 4 个能代表它"的程度。

**系统重采样**：累积权重把 8 个粒子排成线段 $[0,0.000)\,[0.000,0.099)\,[0.099,0.423)\,[0.423,0.476)\,[0.476,0.506)\,[0.506,0.683)\,[0.683,1.000)\,[1.000,1.001)$；抽到 $u=0.031$，梳齿在 $0.031,0.156,0.281,0.406,0.531,0.656,0.781,0.906$，分别落入第 $2,3,3,3,6,6,7,7$ 段：

| 粒子 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 期望复制数 $Nw_i$ | 0.00 | 0.79 | 2.59 | 0.43 | 0.24 | 1.42 | 2.54 | 0.00 |
| 实际复制数 | 0 | 1 | 3 | 0 | 0 | 2 | 2 | 0 |

新粒子集 $\{-1.3,-0.9,-0.9,-0.9,0.7,0.7,1.1,1.1\}$，权重各 $1/8$。实际复制数是期望复制数取整的随机版本，等权平均的期望等于原来的加权平均。

**移动**（$\pi_\beta$，$\beta=0.135$，随机游走标准差 0.3）：粒子 $0.7\to0.95$，密度比 1.97，必接受；粒子 $1.1\to1.6$，密度比 0.002，几乎必拒绝。三个 $-0.9$ 各自提议不同的点，接受后分散开。

例子里能看到 SMC 的两个特征：勉强能代表目标就停下来（ESS 仪表）；群体按目标质量在两个阱之间自动分配（重采样后 4 个在左、4 个在右，对应两个等深的阱）。

### 2.6 直观理解

一群人过河，不是一个人过河：MCMC 是一条链慢慢走，SMC 是一群粒子同时走，用群体的分布代表目标。重采样像自然选择，与当前目标不合的粒子被淘汰、合的被复制；MCMC 移动像变异，让复制出的后代不完全一样。每一步只走"看得清"的距离：ESS 掉到一半就停下来重采样，或缩小步子。

## 3. CEM：交叉熵方法

### 3.1 原始问题：稀有事件概率

CEM（Rubinstein 1997, 1999）最初不是优化算法，而是稀有事件概率的重要性采样。设 $X\sim f$，要估计

$$
\ell=P\big(S(X)\ge\gamma\big)=\mathbb E_f\big[\mathbb{1}\{S(X)\ge\gamma\}\big],
$$

而这个事件很稀有（$\ell\sim10^{-9}$），直接采样几乎采不到。

### 3.2 理论：零方差提议与交叉熵投影

重要性采样估计 $\hat\ell=\frac1N\sum_i\mathbb{1}\{S(X_i)\ge\gamma\}\,f(X_i)/g(X_i)$，$X_i\sim g$。**零方差**的提议是

$$
g^*(x)=\frac{\mathbb{1}\{S(x)\ge\gamma\}\,f(x)}{\ell},
$$

因为此时每个样本的被加数都恒等于 $\ell$。它不能直接用：归一化常数就是要求的 $\ell$。于是退一步，在参数族 $\{f(\cdot;v)\}$ 里找离 $g^*$ 最近的成员，距离用 KL 散度（交叉熵）：

$$
v^*=\arg\min_vD_{\mathrm{KL}}\big(g^*\,\|\,f(\cdot;v)\big)
=\arg\min_v\Big\{\mathbb E_{g^*}[\log g^*]-\mathbb E_{g^*}[\log f(X;v)]\Big\}
=\arg\max_v\ \mathbb E_{g^*}\big[\log f(X;v)\big].
$$

第一项与 $v$ 无关。这是**前向 KL**，等价于在 $g^*$ 下做极大似然。把 $g^*$ 的期望换回 $f$ 下的期望再用样本近似：

$$
\mathbb E_{g^*}\big[\log f(X;v)\big]=\frac1\ell\,\mathbb E_f\big[\mathbb{1}\{S(X)\ge\gamma\}\log f(X;v)\big]
\ \propto\ \sum_{i:\,S(X_i)\ge\gamma}\log f(X_i;v),
$$

即**只用落在事件里的样本（elite）做极大似然**。若样本来自 $f(\cdot;u)$ 而不是 $f$，则每项还要乘似然比 $f(X_i)/f(X_i;u)$；优化版本把这一项省略（每轮的目标相对当前采样器定义），与 MPPI 省略重要性采样修正项是同一件事。

**高斯族的闭式解。** $f(\cdot;v)=\mathcal N(U,\Sigma)$，elite 集合 $E$，$|E|=k$：

$$
\sum_{i\in E}\log\mathcal N(X_i;U,\Sigma)=-\frac k2\log\det\Sigma-\frac12\sum_{i\in E}(X_i-U)^\top\Sigma^{-1}(X_i-U)+\text{const}.
$$

对 $U$ 求导：$\Sigma^{-1}\sum_{i\in E}(X_i-U)=0\Rightarrow U=\frac1k\sum_{i\in E}X_i$。代回后对 $\Sigma$ 求导（用 $\partial\log\det\Sigma=\mathrm{tr}(\Sigma^{-1}\partial\Sigma)$ 与 $\partial\,\mathrm{tr}(\Sigma^{-1}A)=-\mathrm{tr}(\Sigma^{-1}\partial\Sigma\,\Sigma^{-1}A)$）得 $\Sigma=\frac1k\sum_{i\in E}(X_i-U)(X_i-U)^\top$。所以 CE 投影就是 elite 的样本均值与样本协方差——**矩匹配**。

这一步与 MPPI 把 $r_U^*\propto\gamma\cdot\mathcal N(V;U,\Sigma)$ 投影到高斯族是同一个 M-projection，只是权重从 Boltzmann 因子 $e^{-C/\lambda}$ 换成了指示函数 $\mathbb{1}\{C\le\gamma\}$，并且同时更新了协方差。

### 3.3 多层技巧：分位数自适应阈值

稀有事件一上来采不到 elite。于是把阈值也做成序列：每轮从当前 $f(\cdot;v_t)$ 采 $N$ 个样本，取 $S$ 的 $(1-\rho)$ 分位数作为 $\gamma_t$（elite 比例 $\rho$ 固定，常取 $1\%\sim10\%$），在这 $\rho N$ 个 elite 上做 MLE 得 $v_{t+1}$。$\gamma_t$ 逐轮上升，直到达到目标 $\gamma$，最后一轮用 $f(\cdot;v_T)$ 做带似然比的重要性采样估计 $\ell$。

固定 elite 比例意味着**每轮的"有效样本数"恒等于 $\rho N$**：无论 $S$ 的尺度如何，阈值都自动调到让恰好这么多样本入选。这与 SMC 用 ESS 二分选温度是同一种自适应，只是仪表从 ESS 换成了分位数。

### 3.4 用于优化

把"事件 $S\ge\gamma$"里的 $\gamma$ 不断抬高（对最小化问题即 $C\le\gamma_t$ 不断降低），事件集合缩小到最优点附近，$f(\cdot;v_t)$ 退化成最优点上的 Dirac。算法：

1. 从 $\mathcal N(U_t,\Sigma_t)$ 采 $N$ 个 $V_i$，rollout 得 $C(V_i)$；
2. 取代价最小的 $k=\rho N$ 个为 elite；
3. $U_{t+1}=$ elite 均值，$\Sigma_{t+1}=$ elite 协方差（常加下限 $\sigma_{\min}$ 或平滑 $\Sigma_{t+1}\leftarrow\alpha\Sigma_{\text{elite}}+(1-\alpha)\Sigma_t$ 防止过早塌缩）。

`hydrax` 里 `cem.py` 正是这个循环，附加一小部分始终按初始方差采样的探索样本。收敛性质：有限空间下以概率 1 收敛到最优（Costa–Jones–Kroese 2007）；连续空间下在 MRAS 框架（Hu–Fu–Marcus 2007）的条件下收敛到局部最优，CEM 是 MRAS 取截断参考分布时的特例。

### 3.5 在 MPPI 框架里的位置

按前一章的记号，CEM 一轮是对截断分布

$$
r^{\mathrm{CEM}}_U(V)\propto\mathbb{1}\{C(V)\le\gamma_t\}\,\mathcal N(V;U,\Sigma)
$$

做均值与协方差匹配；MPPI 是对 $r_U^*\propto e^{-C(V)/\lambda}\mathcal N(V;U,\Sigma)$ 做均值匹配。两者都是单峰族上的 M-projection：若 elite 同时落在两个模态，均值同样落在中间。CEM 之所以在双峰情形能更快锁定一侧，是因为协方差逐轮收缩：一旦随机性让 elite 偏向一侧，$\Sigma$ 变小，下一轮几乎只在那一侧采样。这是 $\Sigma$ 自适应的效果，不是 elite 过滤本身的效果。

## 4. 对比与统一视角

### 4.1 同一个循环

两者每一轮都是

$$
\text{从当前提议采样}\ \to\ \text{按适合度重加权}\ \to\ \text{用重加权后的群体构造下一个提议}.
$$

这就是**自适应重要性采样**（Bugallo 等 2017 的综述把 CEM、PMC、SMC 采样器放在同一框架下）。MPPI 也在其中：它的"构造下一个提议"是只更新均值。

### 4.2 轴一：群体怎么被总结

| | CEM | SMC |
|---|---|---|
| 重加权后的群体 | 压缩成参数族的充分统计量（高斯：均值、协方差） | 原样保留为带权粒子，重采样 + MCMC 移动 |
| 丢掉了什么 | 均值、协方差以外的一切，尤其是多模态结构 | 什么都不丢 |
| 后果 | 单峰族 → 多模态被平均或塌缩到一侧 | 多模态按质量保留 |
| 每轮成本 | 拟合参数，便宜 | 重采样 + $N$ 步 MCMC，较贵 |

介于两者之间的是 PMC（Cappé 等 2004）：用加权样本拟合**混合分布**作为下一个提议，保留多模态又有参数形式。

### 4.3 轴二：目标序列走向哪里

| | 稀有事件 CEM | 优化 CEM | SMC 采样器 | SMC 用于优化 |
|---|---|---|---|---|
| 目标序列 | $\mathbb{1}\{S\ge\gamma_t\}f$，$\gamma_t\to\gamma$ | $\gamma_t\to\max S$，目标退化为 Dirac | $\gamma_T^{\beta_t}$，$\beta_t\to1$ | $\beta_t\to\infty$（温度 $\to0$），退化为 Dirac |
| 求的是 | 概率 $\ell$ | 最优点 | 样本 / 积分 / 归一化常数 | 最优点 |

- **稀有事件 CEM 与 SMC 处理同一个问题。** SMC 那边对应的方法叫 subset simulation / multilevel splitting（Au–Beck 2001；Cérou–Del Moral–Furon–Guyader 2012）：同样用分位数自适应设阈值序列，但粒子用 MCMC 移动而不是重新拟合高斯。
- **优化 CEM 与 SMC 采样器处理的问题不同**（找点 vs 采样），但把 SMC 的温度一直退火到 0 就成了群体式模拟退火，又回到同一个问题。

### 4.4 自适应机制的等价

| | SMC | CEM |
|---|---|---|
| 仪表 | ESS $=1/\sum w_i^2$ | elite 数 $=\rho N$ |
| 调节量 | 温度增量 $\Delta\beta$（二分） | 阈值 $\gamma_t$（分位数） |
| 含义 | 目标只能尖到 $N/2$ 个粒子仍能代表它 | 目标只能尖到 $\rho N$ 个样本仍入选 |

指示函数权重下 ESS 恰好等于 elite 数（$k$ 个权重各 $1/k$，$\mathrm{ESS}=k$），所以 CEM 的分位数规则就是"ESS 固定为 $\rho N$"的特例。

### 4.5 一张图

| 算法 | 权重形状 | 群体总结 | 协方差 | 目标终点 |
|---|---|---|---|---|
| MPPI（一轮） | $e^{-C/\lambda}$ | 高斯均值 | 固定 | 平滑自由能的极小点 |
| CEM | $\mathbb{1}\{C\le\gamma_t\}$ | 高斯均值 + 协方差 | 自适应收缩 | 最优点（Dirac） |
| PMC | $\gamma/q$ | 混合分布 | 自适应 | 固定目标 |
| SMC 采样器 | $\gamma^{\Delta\beta}$ | 粒子 + 重采样 + MCMC | 无参数 | 固定目标 $\pi$ |
| S2R（多链） | $e^{-C/\lambda}$ | 每链一个点，链间独立 | 带宽按日程 | 每链到 $\pi*p$ 的一个峰 |

### 4.6 对 S2R 的含义

S2R 的多条链各自独立，等于"只有移动步、从不重加权也不重采样"：链之间不交换信息，落进差模态的链一直待在那里。SMC 式的改造是：在每个噪声级用 $e^{-C/\lambda}$ 给各链重加权，ESS 低时重采样（复制好的链、淘汰差的链），再各自做去噪步作为移动。这会让群体按质量分配模态，代价是链不再独立、BRHP 的逐链 mode 选择需要重新定义；这是算法上的实质改动，不是调参。CEM 给 S2R 的启发则是协方差自适应：refine 阶段的带宽可以取当前后验协方差的倍数，而不是固定阶梯。

## 5. 文献

1. N. J. Gordon, D. J. Salmond, A. F. M. Smith. Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proc. F*, 1993.
2. P. Del Moral, A. Doucet, A. Jasra. Sequential Monte Carlo samplers. *JRSS-B*, 2006.
3. A. Jasra, D. A. Stephens, A. Doucet, T. Tsagaris. Inference for Lévy-driven stochastic volatility models via adaptive sequential Monte Carlo. *Scand. J. Stat.*, 2011.
4. A. Beskos, A. Jasra, N. Kantas, A. Thiery. On the convergence of adaptive sequential Monte Carlo methods. *Ann. Appl. Probab.*, 2016.
5. L. Kish. *Survey Sampling*. Wiley, 1965.
6. R. Y. Rubinstein. Optimization of computer simulation models with rare events. *EJOR*, 1997.
7. R. Y. Rubinstein, D. P. Kroese. *The Cross-Entropy Method*. Springer, 2004.
8. A. Costa, O. D. Jones, D. Kroese. Convergence properties of the cross-entropy method for discrete optimization. *Oper. Res. Lett.*, 2007.
9. J. Hu, M. C. Fu, S. I. Marcus. A model reference adaptive search method for global optimization. *Oper. Res.*, 2007.
10. S.-K. Au, J. L. Beck. Estimation of small failure probabilities in high dimensions by subset simulation. *Prob. Eng. Mech.*, 2001.
11. F. Cérou, P. Del Moral, T. Furon, A. Guyader. Sequential Monte Carlo for rare event estimation. *Stat. Comput.*, 2012.
12. O. Cappé, A. Guillin, J.-M. Marin, C. P. Robert. Population Monte Carlo. *JCGS*, 2004.
13. M. F. Bugallo, V. Elvira, L. Martino, D. Luengo, J. Míguez, P. M. Djurić. Adaptive importance sampling: the past, the present, and the future. *IEEE Signal Process. Mag.*, 2017.
14. G. Williams et al. Information-theoretic model predictive control. *IEEE T-RO*, 2018.
