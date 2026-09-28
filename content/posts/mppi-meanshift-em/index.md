---
title: "MPPI 迭代、Mean Shift 与 EM：同一个迭代的三种读法"
date: 2026-09-27
description: "MPPI 的反复迭代、高斯核 mean shift 和 EM 算法各自解决什么问题、各自是什么，以及为什么它们是在同一条平滑密度上做同一个不动点迭代。每一步推导都完整给出。"
summary: "MPPI 迭代、mean shift 与 EM 各自的问题、定义与统一：三者都是 $U\\leftarrow\\mathbb E_{r_U^*}[V]$，都在爬升卷积密度 $\\pi*p$，不动点是它的众数；含全部推导。"
tags: ["MPPI", "mean shift", "EM", "自由能", "mode seeking"]
---

本章自成一体：第 2 节给出全部记号和三个反复使用的基本事实并证明；第 3 节分别说明三种算法各自解决什么问题、算法是什么，并推导各自的更新式；第 4 节证明三者是同一个迭代并给出收敛性质；第 5 节用可以手算的例子解释直观。记号与[《高斯卷积、自由能与相对熵》](/posts/gibbs-convolution/)一致，但不依赖那一章。

## 1. 结论

> [!conclusion] 结论
> MPPI 的反复迭代、高斯核 mean shift 和（高斯观测模型下的）EM 是**同一个不动点迭代**
>
> $$
> \boxed{\;U^{+}=T(U):=\mathbb E_{r_U^*}[V]=\frac{\int V\,\pi(V)\,\mathcal N(V;U,\Sigma)\,dV}{\int \pi(V)\,\mathcal N(V;U,\Sigma)\,dV}\;}
> $$
>
> 它们只在 $\pi$ 是什么、$\Sigma$ 叫什么、期望怎么算这三点上不同：
>
> | | $\pi$ 是什么 | $\Sigma$ 叫什么 | 期望怎么算 | 想求什么 |
> |---|---|---|---|---|
> | MPPI 迭代 | 代价的 Gibbs 分布 $e^{-C/\lambda}/Z_C$ | 探索噪声协方差 | 采样 + rollout，蒙特卡洛 | 低代价的控制序列 |
> | Mean shift | 数据点的经验分布 $\frac1N\sum_n\delta_{V_n}$ | 核带宽 | 对数据点精确求和 | 密度的众数（聚类） |
> | EM | 隐变量的先验 | 观测噪声协方差 | 精确或蒙特卡洛 | 极大似然的参数 $U$ |
>
> 三者共同爬升的函数都是卷积密度 $(\pi*p)(U)$，等价地下降自由能 $F(U)=-\lambda\log(\pi*p)(U)+\text{const}$。因此：
>
> 1. **不动点**是 $\pi*p$ 的驻点，稳定的不动点是它的**众数**，不是代价 $C$ 的极小点；
> 2. 每一步 $(\pi*p)(U)$ **单调不减**，从几乎所有初值收敛到某个众数；
> 3. 收敛是**线性**的，速率 $r=\lambda_{\max}\!\big(\Sigma^{-1}\,\mathrm{Cov}_{r_{U^*}^*}[V]\big)\in[0,1)$，在两个众数刚好合并的 $\Sigma$ 处 $r\to1$，收敛极慢。

## 2. 记号与三个基本事实

### 2.1 记号

- $V\in\mathbb R^d$：一整段控制序列（把 $N$ 步、每步 $n_u$ 维的控制拉直，$d=Nn_u$）。$U\in\mathbb R^d$：当前的名义控制 / 迭代点。
- $C(V)$：轨迹代价，下有界。$\lambda>0$：温度。
- $\pi(V)=e^{-C(V)/\lambda}/Z_C$，$Z_C=\int e^{-C(V)/\lambda}\,dV<\infty$：代价诱导的 **Gibbs 分布**。
- $p=\mathcal N(0,\Sigma)$，$\Sigma\succ0$：高斯核。$\mathcal N(V;U,\Sigma)=(2\pi)^{-d/2}\det(\Sigma)^{-1/2}\exp\!\big(-\tfrac12(V-U)^\top\Sigma^{-1}(V-U)\big)$。
- 卷积：$(\pi*p)(U)=\int\pi(V)\,p(U-V)\,dV$。
- 以 $U$ 为中心的**最优分布 / 后验**：$r_U^*(V)=\dfrac{\pi(V)\,\mathcal N(V;U,\Sigma)}{(\pi*p)(U)}$。
- 自由能：$F(U)=-\lambda\log(\pi*p)(U)-\lambda\log Z_C$。
- $\mathbb E_{r}[\cdot]$、$\mathrm{Cov}_r[\cdot]$：在分布 $r$ 下的期望与协方差。

### 2.2 事实 A：高斯核关于两个参数对称

由定义，$\mathcal N(V;U,\Sigma)$ 只通过 $(V-U)^\top\Sigma^{-1}(V-U)$ 依赖 $V,U$，而 $(V-U)^\top\Sigma^{-1}(V-U)=(U-V)^\top\Sigma^{-1}(U-V)$，故

$$
\mathcal N(V;U,\Sigma)=\mathcal N(U;V,\Sigma)=p(U-V)=p(V-U).
$$

于是 $(\pi*p)(U)=\int\pi(V)\,\mathcal N(V;U,\Sigma)\,dV$，这正是 $r_U^*$ 的归一化常数；$r_U^*$ 确实是概率密度。

> [!note] 注（对称性带来的对偶）
> 同一个卷积有两种读法：
> $$
> (\pi*p)(U)=\mathbb E_{V\sim\pi}\big[\mathcal N(U;V,\Sigma)\big]=\mathbb E_{V\sim\mathcal N(U,\Sigma)}\big[\pi(V)\big]\cdot\text{const},
> $$
> 左边"在每个 $V$ 上放核、在 $U$ 处求值"（KDE），右边"在 $U$ 上放窗口、对 $V$ 加权平均"（局部平均）。由此 KDE 对偶于：边缘似然（3.3 节，mean shift = EM）、Nadaraya–Watson 核回归（3.2 节的 $T(U)$）、以及重要性采样的提议分布（3.1 节：前向加噪核 $q(U\mid V)$ 与围绕 $U$ 的提议 $\mathcal N(V;U,\Sigma)$ 是同一个函数）。$\pi$ 只能评估、不能采样，MPPI 与 S2R 用的都是右边这一侧。

### 2.3 事实 B：高斯核关于中心的梯度

对指数求导：$\nabla_U\big[-\tfrac12(V-U)^\top\Sigma^{-1}(V-U)\big]=\Sigma^{-1}(V-U)$，所以

$$
\nabla_U\,\mathcal N(V;U,\Sigma)=\Sigma^{-1}(V-U)\,\mathcal N(V;U,\Sigma).
$$

### 2.4 事实 C：Gibbs 变分原理与自由能

> [!theorem] 事实 C
> 设 $p$ 为概率分布，$c$ 下有界，$Z=\mathbb E_p[e^{-c/\lambda}]<\infty$，$r^*=p\,e^{-c/\lambda}/Z$。则对任意概率分布 $r$，
>
> $$
> \mathbb E_r[c]+\lambda D_{\mathrm{KL}}(r\|p)=-\lambda\log Z+\lambda D_{\mathrm{KL}}(r\|r^*).
> $$
>
> 因此左边的最小值是 $-\lambda\log Z$，在 $r=r^*$ 处唯一取到。

**证明。** 由 $r^*$ 的定义，$\log r^*=\log p-c/\lambda-\log Z$，即 $c=\lambda\log p-\lambda\log r^*-\lambda\log Z$。两边对 $r$ 取期望：

$$
\mathbb E_r[c]=\lambda\,\mathbb E_r[\log p]-\lambda\,\mathbb E_r[\log r^*]-\lambda\log Z .
$$

加上 $\lambda D_{\mathrm{KL}}(r\|p)=\lambda\mathbb E_r[\log r]-\lambda\mathbb E_r[\log p]$，$\log p$ 项抵消：

$$
\mathbb E_r[c]+\lambda D_{\mathrm{KL}}(r\|p)=\lambda\,\mathbb E_r[\log r]-\lambda\,\mathbb E_r[\log r^*]-\lambda\log Z=\lambda D_{\mathrm{KL}}(r\|r^*)-\lambda\log Z .
$$

KL 非负且仅在 $r=r^*$ 时为零，故得最小值与唯一最小点。$\square$

**应用到控制。** 取 $p=\mathcal N(0,\Sigma)$、$c(\epsilon)=C(U+\epsilon)$，则

$$
Z(U)=\mathbb E_{\epsilon\sim p}\big[e^{-C(U+\epsilon)/\lambda}\big]=\int p(\epsilon)\,e^{-C(U+\epsilon)/\lambda}\,d\epsilon
\overset{V=U+\epsilon}{=}\int p(V-U)\,e^{-C(V)/\lambda}\,dV
=Z_C\int\pi(V)\,p(V-U)\,dV=Z_C\,(\pi*p)(U),
$$

其中换元 $V=U+\epsilon$ 的雅可比为 1，最后一步用了事实 A。于是自由能

$$
F(U):=\min_r\Big\{\mathbb E_{\epsilon\sim r}[C(U+\epsilon)]+\lambda D_{\mathrm{KL}}(r\|p)\Big\}=-\lambda\log Z(U)=-\lambda\log(\pi*p)(U)-\lambda\log Z_C ,
$$

最优分布在扰动坐标下是 $p(\epsilon)e^{-C(U+\epsilon)/\lambda}/Z(U)$，换回 $V=U+\epsilon$ 就是 $r_U^*(V)=\pi(V)\mathcal N(V;U,\Sigma)/(\pi*p)(U)$。**结论：求 $\arg\min_U F(U)$ 等价于求 $\arg\max_U(\pi*p)(U)$。**

## 3. 三者分别解决什么问题、分别是什么

### 3.1 MPPI 迭代：带 KL 正则的随机最优控制

**问题。** 名义控制 $U$ 执行时叠加扰动 $\epsilon\sim\mathcal N(0,\Sigma)$。允许把扰动分布换成任意 $r$，但按 $\lambda D_{\mathrm{KL}}(r\|p)$ 付出代价，于是每个 $U$ 对应一个自由能 $F(U)$（事实 C）。要找的是 $\arg\min_U F(U)=\arg\max_U(\pi*p)(U)$。

**一轮更新的推导。** 对固定的 $U$，最优扰动分布是 $r_U^*$。它一般不是高斯，无法直接执行；MPPI 取它的**均值**作为新的名义控制：

$$
U^{+}=\mathbb E_{r_U^*}[V]=\frac{\int V\,\pi(V)\,\mathcal N(V;U,\Sigma)\,dV}{\int\pi(V)\,\mathcal N(V;U,\Sigma)\,dV}.
$$

（这也是把 $r_U^*$ 投影到均值为 $U^{+}$、协方差 $\Sigma$ 的高斯族上：$\arg\min_{U'}D_{\mathrm{KL}}(r_U^*\|\mathcal N(U',\Sigma))$，因为 $D_{\mathrm{KL}}(r\|\mathcal N(U',\Sigma))=\tfrac12\mathbb E_r[(V-U')^\top\Sigma^{-1}(V-U')]+\text{const}$，对 $U'$ 求导得 $\Sigma^{-1}(\mathbb E_r[V]-U')=0$。）

这个积分算不出来，用重要性采样。从提议分布 $\mathcal N(U,\Sigma)$ 采 $V_k=U+\epsilon_k$，$\epsilon_k\sim\mathcal N(0,\Sigma)$，$k=1,\dots,K$。目标密度与提议密度之比

$$
\frac{r_U^*(V)}{\mathcal N(V;U,\Sigma)}=\frac{\pi(V)\,\mathcal N(V;U,\Sigma)}{(\pi*p)(U)\,\mathcal N(V;U,\Sigma)}=\frac{\pi(V)}{(\pi*p)(U)}\ \propto\ e^{-C(V)/\lambda},
$$

高斯密度约掉，只剩 Gibbs 因子；与 $V$ 无关的常数在自归一化时消失。因此

$$
w_k=\frac{e^{-C(V_k)/\lambda}}{\sum_{j=1}^K e^{-C(V_j)/\lambda}},\qquad
U^{+}\approx\sum_{k=1}^K w_k V_k ,
$$

这就是 MPPI 的更新式。$K\to\infty$ 时右边依概率收敛到 $\mathbb E_{r_U^*}[V]$（自归一化重要性采样的相合性）。MPC 里通常只做一轮就执行；本章关心从固定初值**反复做**会到哪里。

> [!note] 注（带控制代价项的版本）
> 信息论 MPPI 的原始推导把参考分布固定为 $\mathcal N(0,\Sigma)$ 而不是 $\mathcal N(U,\Sigma)$，此时密度比多出因子 $\mathcal N(V;0,\Sigma)/\mathcal N(V;U,\Sigma)=\exp(-U^\top\Sigma^{-1}\epsilon-\tfrac12U^\top\Sigma^{-1}U)$，权重指数里多一项 $U^\top\Sigma^{-1}\epsilon$。本章讨论的是不带这一项的版本，它对应把 $U$ 视作自由能 $F(U)$ 的自变量。

### 3.2 Mean shift：从样本找密度的众数

**问题。** 给定数据点 $V_1,\dots,V_N\in\mathbb R^d$，用高斯核做密度估计

$$
\hat\rho(U)=\frac1N\sum_{n=1}^N\mathcal N(U;V_n,\Sigma),
$$

求 $\hat\rho$ 的众数（局部极大值点）。众数用来做聚类、跟踪、图像分割。

**算法与推导。** 对 $\hat\rho$ 求梯度，逐项用事实 B（把 $\mathcal N(U;V_n,\Sigma)$ 看成中心为 $V_n$、自变量为 $U$，梯度为 $\Sigma^{-1}(V_n-U)\mathcal N(U;V_n,\Sigma)$）：

$$
\nabla\hat\rho(U)=\frac1N\sum_{n=1}^N\Sigma^{-1}(V_n-U)\,\mathcal N(U;V_n,\Sigma)
=\Sigma^{-1}\Big[\frac1N\sum_n V_n\,\mathcal N(U;V_n,\Sigma)-U\,\hat\rho(U)\Big].
$$

两边除以 $\hat\rho(U)$ 并左乘 $\Sigma$：

$$
\Sigma\,\nabla\log\hat\rho(U)=\underbrace{\sum_{n=1}^N\frac{\mathcal N(U;V_n,\Sigma)}{\sum_m\mathcal N(U;V_m,\Sigma)}\,V_n}_{=:T(U)}-U .
$$

右边第一项是"以 $U$ 为中心放一个高斯窗口，窗口内数据点的加权重心"：由事实 A，$\mathcal N(U;V_n,\Sigma)=\mathcal N(V_n;U,\Sigma)$，所以权重可以读作"数据点 $V_n$ 落在以 $U$ 为中心的高斯窗口中的权重"。Mean shift 就是反复令 $U\leftarrow T(U)$；$T(U)-U$ 称为 mean-shift 向量，它等于 $\Sigma\nabla\log\hat\rho(U)$，所以每一步都沿密度的梯度方向、以 $\Sigma$ 为预条件走一步（Comaniciu–Meer）。

**与统一记号的对应。** 令 $\pi_N=\frac1N\sum_n\delta_{V_n}$（经验分布），则 $\hat\rho=\pi_N*p$，且

$$
T(U)=\sum_n\frac{\pi_N(V_n)\,\mathcal N(V_n;U,\Sigma)}{\sum_m\pi_N(V_m)\,\mathcal N(V_m;U,\Sigma)}\,V_n=\mathbb E_{r_U^*}[V],\qquad r_U^*(V_n)\propto\pi_N(V_n)\,\mathcal N(V_n;U,\Sigma).
$$

所以 mean shift 就是把 $\pi$ 取成经验分布后的第 1 节迭代。

### 3.3 EM：含隐变量的极大似然

**问题。** 概率模型：隐变量 $V\sim\pi$，观测 $U=V+\epsilon$，$\epsilon\sim\mathcal N(0,\Sigma)$ 与 $V$ 独立，即 $U\mid V\sim\mathcal N(U;V,\Sigma)$。把 $U$ 当成待估参数（观测模型的位置参数），要极大化边缘似然

$$
L(U)=\int\pi(V)\,\mathcal N(U;V,\Sigma)\,dV\overset{\text{事实 A}}{=}\int\pi(V)\,\mathcal N(V;U,\Sigma)\,dV=(\pi*p)(U).
$$

$\pi$ 是先验，$\mathcal N(U;V,\Sigma)$ 是似然；由事实 A，也可以把 $\mathcal N(V;U,\Sigma)$ 当先验、$\pi\propto e^{-C/\lambda}$ 当"最优性"似然（Wang–Sharifi–Fazlyab 的写法），边缘完全相同。

**算法与推导。** EM 交替两步。

*E 步*：由贝叶斯公式算隐变量的后验

$$
r_U^*(V)=\frac{\pi(V)\,\mathcal N(U;V,\Sigma)}{\int\pi(V')\,\mathcal N(U;V',\Sigma)\,dV'}=\frac{\pi(V)\,\mathcal N(V;U,\Sigma)}{(\pi*p)(U)} .
$$

*M 步*：极大化完全数据对数似然在后验下的期望

$$
Q(U'\mid U)=\mathbb E_{r_U^*}\big[\log\pi(V)+\log\mathcal N(V;U',\Sigma)\big].
$$

第一项与 $U'$ 无关。第二项展开：$\log\mathcal N(V;U',\Sigma)=-\tfrac12(V-U')^\top\Sigma^{-1}(V-U')-\tfrac12\log\det(2\pi\Sigma)$，所以

$$
Q(U'\mid U)=-\tfrac12\,\mathbb E_{r_U^*}\big[(V-U')^\top\Sigma^{-1}(V-U')\big]+\text{const}.
$$

对 $U'$ 求梯度：$\nabla_{U'}Q=\mathbb E_{r_U^*}[\Sigma^{-1}(V-U')]=\Sigma^{-1}\big(\mathbb E_{r_U^*}[V]-U'\big)$；Hessian 为 $-\Sigma^{-1}\prec0$，故 $Q$ 严格凹，唯一极大点是

$$
U'=\mathbb E_{r_U^*}[V]=T(U).
$$

所以 EM 的一步也是第 1 节的迭代。EM 的一般理论保证 $L(U^{+})\ge L(U)$，第 4 节给出完整证明。

> [!tip]+ 一个直观例子：从混合身高数据估计男女平均身高
> **问题。** 观测到 $N$ 个人的身高 $x_1,\dots,x_N$，性别 $z_i\in\{\text{男},\text{女}\}$ 没有记录。模型：先验 $P(z_i=\text{男})=\alpha$；似然 $x_i\mid z_i\sim\mathcal N(\mu_{z_i},\sigma^2)$（$\sigma$ 已知）；参数 $\theta=(\alpha,\mu_{\text{男}},\mu_{\text{女}})$。若性别已知，估计就是分组求平均；性别未知时只能极大化边缘似然
> $$
> \log p(x;\theta)=\sum_i\log\Big[\alpha\,\mathcal N(x_i;\mu_{\text{男}},\sigma^2)+(1-\alpha)\,\mathcal N(x_i;\mu_{\text{女}},\sigma^2)\Big],
> $$
> 对数里套着求和，没有闭式解。这就是 EM 要解决的问题：**数据里缺失的那部分（性别）恰好是让估计变简单的部分。**
>
> **E 步。** 用当前参数算每个人性别的后验（责任度）：
> $$
> w_i=P(z_i=\text{男}\mid x_i;\theta)=\frac{\alpha\,\mathcal N(x_i;\mu_{\text{男}},\sigma^2)}{\alpha\,\mathcal N(x_i;\mu_{\text{男}},\sigma^2)+(1-\alpha)\,\mathcal N(x_i;\mu_{\text{女}},\sigma^2)} .
> $$
>
> **M 步。** 把责任度当权重，按"性别已知"的公式更新，即极大化完全数据对数似然在后验下的期望 $Q(\theta\mid\theta^{(t)})=\sum_i\big[w_i\log\big(\alpha\,\mathcal N(x_i;\mu_{\text{男}},\sigma^2)\big)+(1-w_i)\log\big((1-\alpha)\,\mathcal N(x_i;\mu_{\text{女}},\sigma^2)\big)\big]$，得
> $$
> \mu_{\text{男}}=\frac{\sum_i w_ix_i}{\sum_i w_i},\qquad
> \mu_{\text{女}}=\frac{\sum_i(1-w_i)x_i}{\sum_i(1-w_i)},\qquad
> \alpha=\frac1N\sum_i w_i .
> $$
> 交替进行，每一轮 $\log p(x;\theta)$ 不减。
>
> **与本章的对应。**
>
> | 身高例子 | 本章（MPPI / mean shift） |
> |---|---|
> | 缺失变量：性别 $z_i$ | $V$：哪条控制序列是最优的 |
> | 先验 $P(z)$ | $\mathcal N(V;U,\Sigma)$ |
> | 似然 $\mathcal N(x_i;\mu_z,\sigma^2)$ | $e^{-C(V)/\lambda}$ |
> | 责任度 $w_i$ | 权重 $w_k\propto e^{-C(V_k)/\lambda}$ |
> | M 步的加权平均 | $U^{+}=\sum_k w_kV_k$ |
> | 边缘似然 $\log p(x;\theta)$ | $\log(\pi*p)(U)$ |
>
> **为什么本章的权重里没有先验。** 身高例子把 $z_i$ 的两个取值都枚举出来，先验 $\alpha$ 只能作为因子写进权重。本章的 $V$ 连续且高维，无法枚举，改为从先验 $\mathcal N(U,\Sigma)$ 采样 $V_k$ 再按似然加权：
> $$
> \mathbb E_{r_U^*}[V]=\frac{\mathbb E_{V\sim\mathcal N(U,\Sigma)}\big[V\,e^{-C(V)/\lambda}\big]}{\mathbb E_{V\sim\mathcal N(U,\Sigma)}\big[e^{-C(V)/\lambda}\big]}\approx\frac{\sum_kV_k\,e^{-C(V_k)/\lambda}}{\sum_k e^{-C(V_k)/\lambda}},
> $$
> 先验通过样本的分布进入估计（先验大的地方样本多），权重里只剩似然。这是提议分布等于先验的重要性采样；若样本来自别的提议分布 $g$，先验就会以 $\mathcal N(V_k;U,\Sigma)/g(V_k)$ 的形式回到权重中（3.1 节注里的控制代价项正是这种情形）。

## 4. 关联：同一迭代、单调性、不动点与收敛速率

> [!theorem] 定理 1（三者是同一迭代）
> 设 $\pi$ 是 $\mathbb R^d$ 上的概率测度（Gibbs 密度或经验分布均可），$p=\mathcal N(0,\Sigma)$，$\Sigma\succ0$，$(\pi*p)(U)<\infty$。则 MPPI 的精确期望更新、高斯核 mean shift 与 3.3 节 EM 的一步都等于
>
> $$
> T(U)=\mathbb E_{r_U^*}[V]=U+\Sigma\,\nabla\log(\pi*p)(U)=U-\frac{\Sigma}{\lambda}\nabla F(U),
> $$
>
> 并且：
>
> 1. **单调性**：$(\pi*p)(T(U))\ge(\pi*p)(U)$，等号当且仅当 $T(U)=U$；
> 2. **不动点**：$T(U)=U\iff\nabla(\pi*p)(U)=0$；
> 3. **局部收敛速率**：在不动点 $U^*$ 处 $\partial T(U^*)=\mathrm{Cov}_{r_{U^*}^*}[V]\,\Sigma^{-1}$，其特征值实且非负；$U^*$ 是 $\pi*p$ 的严格局部极大值当且仅当这些特征值都小于 1，此时迭代局部线性收敛，速率 $r=\lambda_{\max}\big(\Sigma^{-1}\mathrm{Cov}_{r_{U^*}^*}[V]\big)$。

### 4.1 更新式等于 score 上升

在 $(\pi*p)(U)=\int\pi(V)\mathcal N(V;U,\Sigma)dV$ 中只有核依赖 $U$，在积分号下求导并用事实 B：

$$
\nabla(\pi*p)(U)=\int\pi(V)\,\Sigma^{-1}(V-U)\,\mathcal N(V;U,\Sigma)\,dV
=\Sigma^{-1}\Big[\int V\,\pi(V)\mathcal N(V;U,\Sigma)\,dV-U\int\pi(V)\mathcal N(V;U,\Sigma)\,dV\Big].
$$

方括号里第二个积分是 $(\pi*p)(U)$，第一个积分是 $(\pi*p)(U)\,\mathbb E_{r_U^*}[V]$。于是

$$
\nabla(\pi*p)(U)=\Sigma^{-1}(\pi*p)(U)\big(\mathbb E_{r_U^*}[V]-U\big)
\ \Longrightarrow\
\mathbb E_{r_U^*}[V]-U=\Sigma\,\nabla\log(\pi*p)(U).
$$

再由 $F=-\lambda\log(\pi*p)+\text{const}$ 得 $\nabla\log(\pi*p)=-\nabla F/\lambda$。第 3 节已证三种算法的更新式都是 $\mathbb E_{r_U^*}[V]$，故三者相同，且都等于 $U+\Sigma\nabla\log(\pi*p)(U)$。这个恒等式就是 Tweedie 公式：后验均值 = 观测 + 噪声协方差 × 边缘密度的 score。

### 4.2 单调性与不动点（EM 的下界论证）

任取 $U'$。把 $(\pi*p)(U')$ 写成关于 $r_U^*$ 的期望再用 Jensen 不等式（$\log$ 凹）：

$$
\log(\pi*p)(U')=\log\int r_U^*(V)\,\frac{\pi(V)\,\mathcal N(V;U',\Sigma)}{r_U^*(V)}\,dV
\ \ge\ \int r_U^*(V)\log\frac{\pi(V)\,\mathcal N(V;U',\Sigma)}{r_U^*(V)}\,dV=:\mathcal L(U'\mid U).
$$

Jensen 取等当且仅当被积的比值关于 $V$ 几乎处处为常数。由 $r_U^*(V)=\pi(V)\mathcal N(V;U,\Sigma)/(\pi*p)(U)$，比值等于 $(\pi*p)(U)\,\mathcal N(V;U',\Sigma)/\mathcal N(V;U,\Sigma)$，它为常数当且仅当 $U'=U$。特别地 $\mathcal L(U\mid U)=\log(\pi*p)(U)$。

展开 $\mathcal L$：

$$
\mathcal L(U'\mid U)=\mathbb E_{r_U^*}[\log\pi(V)]+\mathbb E_{r_U^*}[\log\mathcal N(V;U',\Sigma)]-\mathbb E_{r_U^*}[\log r_U^*(V)],
$$

只有中间一项依赖 $U'$，它就是 3.3 节的 $Q(U'\mid U)$，严格凹，唯一极大点 $U'=T(U)$。于是

$$
\log(\pi*p)(T(U))\ \overset{\text{Jensen}}{\ge}\ \mathcal L(T(U)\mid U)\ \overset{\text{M 步}}{\ge}\ \mathcal L(U\mid U)=\log(\pi*p)(U).
$$

这证明了单调性。等号成立当且仅当两个不等号同时取等：第二个取等要求 $U$ 已经是 $Q(\cdot\mid U)$ 的极大点，即 $T(U)=U$；反之 $T(U)=U$ 时两边显然相等。结合 4.1，$T(U)=U\iff\Sigma\nabla\log(\pi*p)(U)=0\iff\nabla(\pi*p)(U)=0$，不动点恰是驻点。$(\pi*p)(U_t)$ 单调有界必收敛；若 $\pi*p$ 的驻点孤立，则 $U_t$ 收敛到某个驻点，且鞍点、极小点不稳定（见 4.3），故从几乎所有初值收敛到局部极大值。Fashing–Tomasi 从 bound optimization 的角度、Carreira-Perpiñán 从 EM 的角度给出的正是这一论证。

### 4.3 收敛速率

对 $T(U)=\int V\,r_U^*(V)\,dV$ 求关于 $U$ 的雅可比。先算 $\nabla_U\log r_U^*(V)$：

$$
\log r_U^*(V)=\log\pi(V)+\log\mathcal N(V;U,\Sigma)-\log(\pi*p)(U)
\ \Longrightarrow\
\nabla_U\log r_U^*(V)=\Sigma^{-1}(V-U)-\Sigma^{-1}\big(T(U)-U\big)=\Sigma^{-1}\big(V-T(U)\big),
$$

其中第二项用了 4.1 的 $\nabla\log(\pi*p)(U)=\Sigma^{-1}(T(U)-U)$。于是 $\nabla_U r_U^*(V)=r_U^*(V)\,\Sigma^{-1}(V-T(U))$，

$$
\partial T(U)=\int V\,\big[\nabla_U r_U^*(V)\big]^\top dV=\int V\,(V-T(U))^\top r_U^*(V)\,dV\ \Sigma^{-1}.
$$

因为 $\int(V-T(U))^\top r_U^*(V)dV=0$，可以把被积的 $V$ 换成 $V-T(U)$：

$$
\partial T(U)=\int(V-T(U))(V-T(U))^\top r_U^*(V)\,dV\ \Sigma^{-1}=\mathrm{Cov}_{r_U^*}[V]\,\Sigma^{-1}.
$$

记 $M=\Sigma^{-1/2}\mathrm{Cov}_{r_U^*}[V]\Sigma^{-1/2}$，它对称半正定。$\partial T=\Sigma^{1/2}M\Sigma^{-1/2}$ 与 $M$ 相似，特征值实且 $\ge0$。

再看 $\log(\pi*p)$ 的 Hessian。对 4.1 的 $\nabla\log(\pi*p)(U)=\Sigma^{-1}(T(U)-U)$ 求导：

$$
\nabla^2\log(\pi*p)(U)=\Sigma^{-1}\big(\partial T(U)-I\big)=\Sigma^{-1/2}\big(M-I\big)\Sigma^{-1/2}.
$$

在不动点 $U^*$，$U^*$ 是严格局部极大值 $\iff$ Hessian 负定 $\iff M\prec I\iff$ $\partial T(U^*)$ 的特征值全在 $[0,1)$。此时 $U_{t+1}-U^*=\partial T(U^*)(U_t-U^*)+o(\|U_t-U^*\|)$，误差按谱半径 $r=\lambda_{\max}(M)=\lambda_{\max}(\Sigma^{-1}\mathrm{Cov}_{r_{U^*}^*}[V])<1$ 线性收缩。若 $U^*$ 是鞍点或极小点，$M$ 有特征值 $\ge1$，沿对应方向扰动被放大，不稳定。$\square$

这正是 Carreira-Perpiñán 的速率 $\lambda_{\max}(\sigma^{-2}\,\mathrm{Cov})$（各向同性核 $\Sigma=\sigma^2I$），也是 Wang 等人定理 2 中 $-(\nabla^2_{11}Q)^{-1}\nabla^2_{12}Q$ 在高斯固定协方差时的取值（$\nabla^2_{11}Q=-\Sigma^{-1}$，$\nabla^2_{12}Q=\mathrm{Cov}_{r_U^*}[\Sigma^{-1}(V-U)]\cdot$ 等价形式）。

## 5. 直观解释与可手算的例子

### 5.1 三个角度，同一个点

三种算法各自从一个角度看待同一个结果：

- **优化（MPPI）**：结果是一个优化问题的极小点——但不是原代价 $C$ 的极小点，而是自由能 $F(U)=\mathbb E_\epsilon[C(U+\epsilon)]+\text{KL 正则}$ 的极小点，即"在噪声 $\Sigma$ 下平均代价最低、且不偏离参考分布太远"的控制。它先在分布层面求出最优分布 $r_U^*$，再投影回高斯族只保留均值。
- **几何（mean shift）**：把分布看成一堆点或一张密度图，问"点最密集的地方在哪"。结果的刻画是：一个点，恰好是自己邻域内点的加权重心。迭代就是"站在哪里，就往周围点的重心挪"。
- **统计（EM）**：把结果看成一个生成模型的极大似然参数。问法是："假设先从 $\mathcal N(U,\Sigma)$ 抽出控制 $V$，再以概率 $e^{-C(V)/\lambda}$ 被判为最优；观测到了'最优'这个事件却没看到 $V$，哪个 $U$ 最能解释这个观测？"迭代是"猜隐变量（E 步），再重新拟合参数（M 步）"的自洽循环，停在"由 $U$ 推出的 $V$ 反过来又推荐同一个 $U$"的不动点。

三者落在同一个点上，是因为它们盯着同一个函数，只是各自起了名字：

$$
\log(\pi*p)(U)\;=\;\underbrace{-F(U)/\lambda+\text{const}}_{\text{优化：自由能}}\;=\;\underbrace{\log\hat\rho(U)}_{\text{几何：平滑密度}}\;=\;\underbrace{\log L(U)}_{\text{统计：边缘似然}} .
$$

而三者的迭代都落到同一步 $U\leftarrow\mathbb E_{r_U^*}[V]$，原因是 4.1 节的 Tweedie 恒等式：后验均值减去当前点等于平滑密度的 score 乘以 $\Sigma$。这条恒等式把统计对象（后验均值）和分析对象（梯度）粘在一起，所以"往后验均值走"（EM / MPPI）与"沿密度梯度爬"（mean shift）是同一件事。

| 角度 | $U$ 是什么 | 结果是什么 | 一步迭代的含义 |
|---|---|---|---|
| 优化（MPPI） | 名义控制 | 平滑自由能的极小点 | 沿 $-\nabla F$ 走一步预条件梯度 |
| 几何（mean shift） | 密度图上的位置 | 平滑密度的峰 | 挪到邻域点的加权重心 |
| 统计（EM） | 生成模型的参数 | 边缘似然的极大点 | 猜隐变量，再重新拟合 |

### 5.2 地形固定，窗口在动

$\pi*p$ 是一张固定的"模糊地形"：把 $e^{-C/\lambda}$ 这张图用宽度 $\Sigma$ 的高斯核模糊一遍。每一轮迭代变的只是站的位置 $U$ 和以 $U$ 为中心的观察窗口 $r_U^*$；窗口的归一化常数恰是地形在 $U$ 处的高度 $(\pi*p)(U)$。迭代把窗口中心挪到窗口内亮度的重心上，重心与中心重合时停下，那里就是模糊地形的峰。

### 5.3 二次代价：一步一步收缩到峰

取 $C(V)=\tfrac12(V-m)^\top A(V-m)$，$A\succ0$。则 $\pi=\mathcal N(m,\lambda A^{-1})$（因为 $e^{-C/\lambda}\propto\exp(-\tfrac12(V-m)^\top\tfrac{A}{\lambda}(V-m))$）。

*后验。* 两个高斯相乘，指数相加：

$$
-\tfrac12(V-m)^\top\tfrac{A}{\lambda}(V-m)-\tfrac12(V-U)^\top\Sigma^{-1}(V-U)
=-\tfrac12 V^\top\big(\tfrac{A}{\lambda}+\Sigma^{-1}\big)V+V^\top\big(\tfrac{A}{\lambda}m+\Sigma^{-1}U\big)+\text{const},
$$

配方得 $r_U^*=\mathcal N\big(\mu_U,P\big)$，$P=\big(\tfrac{A}{\lambda}+\Sigma^{-1}\big)^{-1}$，$\mu_U=P\big(\tfrac{A}{\lambda}m+\Sigma^{-1}U\big)$。

*迭代。* $T(U)=\mu_U$，故

$$
T(U)-m=P\Sigma^{-1}(U-m)\qquad\big(\text{因为 }P\tfrac{A}{\lambda}m+P\Sigma^{-1}m=P(\tfrac A\lambda+\Sigma^{-1})m=m\big).
$$

每轮误差乘以 $P\Sigma^{-1}=(I+\Sigma A/\lambda)^{-1}$，特征值在 $(0,1)$ 内，不动点 $U^*=m$。这里 $\mathrm{Cov}_{r_U^*}[V]=P$，所以 $\partial T=P\Sigma^{-1}$，与定理 1 一致。

*卷积。* $\pi*p=\mathcal N(m,\lambda A^{-1}+\Sigma)$，峰在 $m$，与 $C$ 的极小点重合：对称二次代价下卷积不移动峰。

*小噪声极限。* $\Sigma\to0$ 时 $(I+\Sigma A/\lambda)^{-1}\approx I-\Sigma A/\lambda$，故 $T(U)-U\approx-\tfrac{\Sigma}{\lambda}A(U-m)=-\tfrac{\Sigma}{\lambda}\nabla C(U)$：迭代退化为步长 $\Sigma/\lambda$ 的梯度下降。一般代价下同样成立，因为 $\Sigma\to0$ 时 $\pi*p\to\pi$、$\nabla\log\pi=-\nabla C/\lambda$。

### 5.4 双峰代价：临界带宽与众数合并

一维，$\pi=\tfrac12\mathcal N(a,s^2)+\tfrac12\mathcal N(-a,s^2)$（两条等价路线）。高斯与高斯卷积仍是高斯，方差相加：

$$
(\pi*p)(U)=\tfrac12\mathcal N(U;a,v)+\tfrac12\mathcal N(U;-a,v),\qquad v=s^2+\sigma^2 .
$$

对称性给出 $(\pi*p)'(0)=0$，所以 $U=0$（两条路线的正中间，即穿过障碍物的那条）总是不动点。它是峰还是谷，看二阶导。对单个高斯 $g(U)=\mathcal N(U;\mu,v)$，$g'=-\tfrac{U-\mu}{v}g$，$g''=\big(\tfrac{(U-\mu)^2}{v^2}-\tfrac1v\big)g$。在 $U=0$ 两项相同，

$$
(\pi*p)''(0)=\Big(\frac{a^2}{v^2}-\frac1v\Big)\mathcal N(0;a,v)=\frac{a^2-v}{v^2}\,\mathcal N(0;a,v).
$$

- $v<a^2$，即 $\sigma^2<a^2-s^2$：二阶导为正，$U=0$ 是两峰之间的**谷**，不稳定；迭代从任何非零初值出发都会滑向 $\pm a$ 附近的峰（蒙特卡洛噪声会把恰好在 0 的初值推开）。
- $v>a^2$：二阶导为负，两峰**合并**成 $U=0$ 处的一个峰，它是稳定不动点：采样噪声大到这个程度时，穿过障碍物的中间轨迹就是迭代 MPPI 的收敛点，而不是暂态。
- $v=a^2$：二阶导为 0，$\pi*p$ 在 0 附近几乎是平的。由 4.3，此时 $\mathrm{Cov}_{r_0^*}[V]=\sigma^2$（Hessian 为零等价于 $M=I$），速率 $r=1$，收敛退化为次线性——这就是 Carreira-Perpiñán 指出的"众数合并处极慢"。

### 5.5 卷积偏向宽盆地

设 $C$ 有两个盆地，底部代价 $C_i$、曲率 $A_i$。Laplace 近似下第 $i$ 个盆地在 $\pi$ 中的质量

$$
m_i\propto e^{-C_i/\lambda}\det\!\big(\lambda A_i^{-1}\big)^{1/2},
$$

宽盆地（$A_i$ 小）质量大。卷积后每个盆地近似为 $m_i\,\mathcal N(\cdot;\,V_i,\ \lambda A_i^{-1}+\Sigma)$，峰高 $\propto m_i\det(\lambda A_i^{-1}+\Sigma)^{-1/2}$。当 $\Sigma\gg\lambda A_i^{-1}$ 时峰高 $\approx m_i\det(\Sigma)^{-1/2}$，只由质量决定：**深而窄的盆地可能输给浅而宽的盆地**。这是迭代 MPPI 偏好鲁棒解、而不一定是最低代价解的根源；也说明 $\pi*p$ 的众数一般不是 $C$ 的极小点。

### 5.6 从这个框架看 CEM 与 S2R

- **CEM**：把权重 $e^{-C/\lambda}$ 换成"前 $k$ 名"的指示函数，并同时按 elite 的样本协方差更新 $\Sigma$。它仍是对某个平滑密度做均值（与协方差）匹配，只是 $\pi$ 被截断、$\Sigma$ 逐轮收缩，因而在双峰情形能更快锁定一侧。
- **S2R diffusion 的 sample 阶段**：每步同样计算 $\mathbb E_{r_U^*}[V]$（E 步），但只走部分步长并注入高斯噪声，同时让 $\Sigma$ 按日程递减：这是带退火的随机 EM（第 6 节）。带宽降到 5.4 节的临界值以下时，两峰分开，注入的噪声决定链落入哪个峰。
- **S2R 的 refine 阶段**：固定小带宽、不加噪声，就是本章的 mean shift/EM，收敛到 $\pi*p$ 在小 $\Sigma$ 下的峰，即接近 $C$ 的局部极小点（5.3 节的小噪声极限）。

## 6. 第四个角度：Diffusion 是输运，不是找峰

前三个角度的对象都是**一个点**，问题都是"在固定的模糊地形 $\pi*p$ 上找峰"。Diffusion 的对象是**一个分布**（一群粒子），问题是"把一个简单分布搬运成 $\pi$"。它用的仍是同一族地形和同一条 Tweedie 恒等式，但用法不同。本节把噪声取为各向同性 $\Sigma=\sigma^2I$，以便让 $\sigma$ 作为退火变量。

### 6.1 正向过程就是那一族模糊地形

记 $p_\sigma\coloneqq\pi*\mathcal N(0,\sigma^2I)$。高斯卷积有半群性质 $\mathcal N(0,\sigma_1^2I)*\mathcal N(0,\sigma_2^2I)=\mathcal N(0,(\sigma_1^2+\sigma_2^2)I)$，所以"多步加噪"与"一步加到总方差"是同一件事：正向过程给出的是**单参数族** $\{p_\sigma\}_{\sigma\ge0}$，从 $p_0=\pi$ 连续变到 $\sigma\to\infty$ 时的近似高斯。前三个角度里固定 $\Sigma$ 的地形，就是这一族中的一张。

在固定的 $\sigma$ 上反复做 MPPI、mean shift 或 EM，收敛到 $p_\sigma$ 的众数（定理 1）。若让 $\sigma$ 逐级下降、每级都收敛到众数，得到的是一条**众数路径**；这叫确定性退火或延拓法，S2R 的 refine 阶段就是它。Diffusion 不是这样：它**不在任何一级收敛到众数**，而是让粒子在每一级都服从 $p_\sigma$，从 $p_{\sigma_{\max}}$ 一路搬到 $p_0=\pi$。输出是 $\pi$ 的样本，落入各模态的概率等于模态的质量。

### 6.2 有了 score 之后怎么降噪

由 4.1 节（取 $\Sigma=\sigma^2I$），在带噪点 $z$ 处

$$
m(z)\coloneqq\mathbb E_{r_z^*}[V]=z+\sigma^2\nabla\log p_\sigma(z),
$$

即 score 指向"所有可能被加噪成 $z$ 的干净点的后验平均"。一步降噪要从 $z=V+\sigma\epsilon$ 得到同一条噪声路径上噪声更少的点 $z'=V+\sigma'\epsilon'$，$\sigma'<\sigma$。

**推导。** 正向加噪逐步累加，所以老噪声等于新噪声加一段独立增量：

$$
\sigma\epsilon=\sigma'\epsilon'+\sqrt{\sigma^2-\sigma'^2}\,\eta,\qquad \epsilon',\eta\overset{\text{i.i.d.}}{\sim}\mathcal N(0,I).
$$

已知总和 $\sigma\epsilon=z-V$，求分量 $\sigma'\epsilon'$，这是"两个独立高斯之和已知、推其中一个"的高斯回归。$\mathrm{Cov}(\sigma'\epsilon',\sigma\epsilon)=\sigma'^2I$，$\mathrm{Var}(\sigma\epsilon)=\sigma^2I$，故

$$
\mathbb E[\sigma'\epsilon'\mid\sigma\epsilon]=\frac{\sigma'^2}{\sigma^2}(z-V),\qquad
\mathrm{Var}[\sigma'\epsilon'\mid\sigma\epsilon]=\sigma'^2-\frac{\sigma'^4}{\sigma^2}=\sigma'^2\Big(1-\frac{\sigma'^2}{\sigma^2}\Big).
$$

记 $\kappa\coloneqq1-\sigma'^2/\sigma^2\in(0,1]$，则 $z'=V+(1-\kappa)(z-V)+\sigma'\sqrt\kappa\,\xi=z+\kappa(V-z)+\sigma'\sqrt\kappa\,\xi$。真正的 $V$ 未知，用后验均值 $m(z)$ 代替，得到反向一步

$$
\boxed{\;z'=z+\kappa\,\big(m(z)-z\big)+\sigma'\sqrt\kappa\,\xi,\qquad \kappa=1-\frac{\sigma'^2}{\sigma^2}\;}
$$

**两个系数的含义。** $\kappa$ 是"朝后验均值走多少"：$\sigma'$ 接近 $\sigma$ 时 $\kappa\approx0$，几乎不动；$\sigma'=0$ 时 $\kappa=1$，直接跳到 $m(z)$，**这就是 MPPI 的一步**——MBD 指出的"单步去噪等于 MPPI"即此。噪声项 $\sigma'\sqrt\kappa\,\xi$ 是"对剩余不确定性保持诚实"：用方差记账核对，

| 成分 | 方差 |
|---|---|
| 保留的老偏差 $(1-\kappa)(z-V)$ | $(\sigma'^2/\sigma^2)^2\sigma^2=\sigma'^4/\sigma^2$ |
| 新补的噪声 $\sigma'\sqrt\kappa\,\xi$ | $\sigma'^2-\sigma'^4/\sigma^2$ |
| 合计 | $\sigma'^2$ |

朝后验均值收缩后老偏差只剩 $\sigma'^4/\sigma^2$，少于目标 $\sigma'^2$，缺的部分由新噪声补齐；补少了分布逐步塌到众数（回到 mean shift），补多了分布过宽。例：$\sigma=1$，$\sigma'=0.6$，$\kappa=0.64$，保留方差 $0.1296$，补 $0.2304$，合计 $0.36=\sigma'^2$。

**精确与近似。** 若 $V$ 已知，上式是精确的反向转移（布朗桥）。用 $m(z)$ 代替 $V$ 等于把后验近似为集中在均值的点：噪声大时后验接近高斯、近似好；噪声小时靠步子小（$\kappa$ 小）控制误差，所以日程在低噪声处要密。

### 6.3 反向方差不是唯一的

被固定的只是"$z'$ 围绕 $V$ 的总方差等于 $\sigma'^2$"；这 $\sigma'^2$ 里多少来自保留的老噪声方向、多少来自新鲜噪声，是自由的。把新噪声拆成

$$
\sigma'\epsilon'=\sqrt{\sigma'^2-\tau^2}\;\epsilon+\tau\,\xi,\qquad 0\le\tau\le\sigma',
$$

对任何 $\tau$ 两项方差之和都是 $\sigma'^2$。老噪声方向用 $(z-m)/\sigma$ 估计，得到一族更新

$$
z'=m(z)+\sqrt{\sigma'^2-\tau^2}\,\frac{z-m(z)}{\sigma}+\tau\,\xi .
$$

| 选择 | $\tau^2$ | 名称 |
|---|---|---|
| $\sigma'^2\big(1-\sigma'^2/\sigma^2\big)$ | 布朗桥，与马尔可夫前向过程一致 | DDPM（$\eta=1$），Ho 等人的 $\tilde\beta_t$ |
| 介于两者之间 | 与某个非马尔可夫、边缘相同的前向过程一致 | DDIM，$0<\eta<1$ |
| $0$ | 完全复用老噪声方向 | DDIM 确定性（$\eta=0$），概率流 ODE |

Song 等人（DDIM）证明同一族边缘 $\{p_\sigma\}$ 可由无穷多个前向过程生成，每个对应一种反向方差。另一个方向的修正：若给定 $z$ 时 $V$ 仍有明显方差，精确反向核是对后验的混合，其方差为 $\sigma'^2\kappa+\kappa^2\,\mathrm{Cov}[V\mid z]$；Ho 等人的 $\tilde\beta_t$ 与 $\beta_t$ 分别对应后验为点质量与后验方差等于数据方差两个极端，Analytic-DPM 给出在 $\tilde\beta_t$ 上加估计的 $\kappa^2\mathrm{Cov}[V\mid z]$ 作为最优方差。新鲜噪声多，能纠正 score 估计误差、有利于按质量分配模态；新鲜噪声少，路径平滑、可复现，但对 score 误差更敏感。

### 6.4 分叉处的行为

5.4 节算过：$p_\sigma$ 在临界带宽 $\sigma^2=a^2-s^2$（$a$、$s$ 为 5.4 节的峰间距与峰宽）处从一个峰分裂成两个。延拓法在分叉点停在中间的鞍点上，必须靠数值扰动随机选一支；Diffusion 在分叉点由注入的噪声把粒子分到两支，在 score 精确、日程足够细的理想极限下，进入每一支的概率等于该支的质量。这是"搬运分布"与"跟踪众数"的本质区别。

### 6.5 四个角度

| 角度 | 对象 | 在哪张地形上 | 迭代做什么 | 输出 |
|---|---|---|---|---|
| 优化（MPPI） | 一个点 | 固定 $\sigma$ 的 $p_\sigma$ | 走到后验均值 | $p_\sigma$ 的众数 |
| 几何（mean shift） | 一个点 | 同上 | 挪到邻域重心 | 同上 |
| 统计（EM） | 一个参数 | 同上 | 猜隐变量、再拟合 | 同上 |
| 输运（diffusion） | 一个样本 / 一群粒子 | 整族 $\{p_\sigma\}$，$\sigma$ 由大到小 | 朝后验均值走 $\kappa$ 的比例，补噪声 | $\pi$ 的样本，按模态质量分配 |

前三者的核心恒等式是"后验均值 $-$ 当前点 $=\sigma^2\times$ score"，步长为 1、不加噪声；Diffusion 用同一恒等式，把步长改成 $\kappa<1$ 并补回噪声，于是"找峰"变成"搬运分布"。S2R 把两者接起来：先用 diffusion 按质量把粒子分到各模态（sample），再用固定小带宽的 mean shift 把每个粒子推到所在模态的峰（refine）。

## 7. 文献

1. G. Williams, A. Aldrich, E. A. Theodorou. Model predictive path integral control: from theory to parallel computation. *JGCD*, 2017.
2. G. Williams et al. Information-theoretic model predictive control: theory and applications to autonomous driving. *IEEE T-RO*, 2018.
3. Y. Cheng. Mean shift, mode seeking, and clustering. *IEEE TPAMI*, 1995.
4. D. Comaniciu, P. Meer. Mean shift: a robust approach toward feature space analysis. *IEEE TPAMI*, 2002.
5. M. Fashing, C. Tomasi. Mean shift is a bound optimization. *IEEE TPAMI*, 2005.
6. M. Á. Carreira-Perpiñán. Gaussian mean-shift is an EM algorithm. *IEEE TPAMI*, 2007.
7. A. P. Dempster, N. M. Laird, D. B. Rubin. Maximum likelihood from incomplete data via the EM algorithm. *JRSS-B*, 1977.
8. J. Wang, S. Sharifi, M. Fazlyab. Generalized model predictive path integral control as expectation–maximization. arXiv:2606.00317, 2026.
9. M. Fazlyab, S. Sharifi, J. Wang. Model predictive path integral control as preconditioned gradient descent. arXiv:2603.24489, 2026.
10. M. Okada, T. Taniguchi. Variational inference MPC for Bayesian model-based reinforcement learning. *CoRL*, 2019.
11. B. Efron. Tweedie's formula and selection bias. *JASA*, 2011.
12. J. Ho, A. Jain, P. Abbeel. Denoising diffusion probabilistic models. *NeurIPS*, 2020.
13. J. Song, C. Meng, S. Ermon. Denoising diffusion implicit models. *ICLR*, 2021.
14. Y. Song et al. Score-based generative modeling through stochastic differential equations. *ICLR*, 2021.
15. F. Bao, C. Li, J. Zhu, B. Zhang. Analytic-DPM: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. *ICLR*, 2022.
16. C. Pan, Z. Yi, G. Shi, G. Qu. Model-based diffusion for trajectory optimization. *NeurIPS*, 2024.
