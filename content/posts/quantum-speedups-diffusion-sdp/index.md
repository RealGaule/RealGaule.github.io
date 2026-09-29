---
title: "量子计算能加速什么：优化与采样、扩散模型、半定规划"
date: 2026-09-29
description: "量子算法对优化与采样的已证明加速及其隐藏成本；扩散模型去噪为何是退火却没有可加速的瓶颈；量子 SDP 求解器的复杂度与机器人学 SDP 的结构为何不匹配，以及可能的算法方向。"
summary: "三部分：量子计算对优化与采样算法的已有加速（哪些有证明、哪些是启发式、代价藏在哪里）；量子计算与扩散模型（去噪是退火吗，量子能加速哪一步）；量子计算与 SDP（量子 SDP 求解器的复杂度、机器人 SDP 的结构、为何不匹配、可能的新算法）。"
tags: ["量子计算", "扩散模型", "半定规划", "模拟退火", "采样", "机器人"]
---

本章分三部分。第 2 节说明量子加速的来源和隐藏成本；第 3 节整理量子计算对优化与采样算法的已有加速；第 4 节讨论扩散模型的去噪过程与退火的关系，以及量子计算能否加速它；第 5 节讨论量子 SDP 求解器与机器人学中 SDP 的匹配问题。每一部分先给结论，再给依据。所有引用的文献都在第 6 节，均已核对存在。

## 1. 结论

> [!conclusion] 结论
> 1. **优化与采样。** 量子计算有严格证明的加速几乎全是平方根级：Grover 搜索把 $O(N)$ 变成 $O(\sqrt N)$，量子随机游走把马尔可夫链的混合代价从 $O(1/\delta)$ 变成 $O(1/\sqrt\delta)$，振幅估计把蒙特卡洛的 $O(1/\epsilon^2)$ 变成 $O(1/\epsilon)$。指数加速只出现在结构特殊的问题上（整数分解、量子系统模拟、少数线性代数问题），且依赖苛刻的输入输出假设。QAOA、量子退火等启发式方法没有理论保证，也尚无实用规模的优势证据。
> 2. **扩散模型。** 去噪过程在数学上是带学习势能的退火 Langevin 动力学，噪声水平扮演温度，这一类比已被形式化。但扩散模型的设计目的恰恰是绕过退火的混合瓶颈：给定准确 score，反向过程步数对维度近线性。量子采样算法加速的正是那个被绕过的瓶颈，所以对数据驱动的扩散模型帮助很小；只在扩散被用作显式能量函数的采样器或优化器时，才有平方根级的理论空间。
> 3. **半定规划。** 量子 SDP 求解器的运行时间对 $\gamma = Rr/\epsilon$ 是高次多项式，其中 $R$、$r$ 是原对偶解的迹界，$\epsilon$ 是精度；下界证明这种依赖不可避免。机器人学中的 SDP 迹界随规模线性增长、需要 $10^{-6}$ 以上的认证精度、且最优解原始退化，三点都击中量子算法的弱点，因此没有渐近加速。有价值的是算法思想迁移：低精度矩阵指数迭代加舍入加对偶证书的流水线，以及针对秩约束的量子启发层级。

## 2. 预备：量子加速从哪里来，代价藏在哪里

### 2.1 四个加速来源

量子算法的加速几乎都能归结为下面四个原语之一。

**振幅放大（Grover）。** 若一个随机过程以概率 $p$ 成功，经典期望需要 $O(1/p)$ 次重复，量子只需 $O(1/\sqrt p)$ 次。在 $N$ 个候选中找一个满足条件的元素，就是 $p = 1/N$ 的特例，得到 $O(\sqrt N)$。这个平方根已被证明是最优的。

**量子随机游走（Szegedy）。** 对可逆马尔可夫链 $P$，令 $\delta$ 为谱间隙。经典混合到平稳分布需要 $O(1/\delta)$ 步，Szegedy 构造的量子游走把相关任务的代价降到 $O(1/\sqrt\delta)$。这是量子模拟退火和量子配分函数估计的基础。

**振幅估计。** 估计一个有界随机变量的期望到加性误差 $\epsilon$，经典蒙特卡洛需要 $O(1/\epsilon^2)$ 个样本，量子振幅估计需要 $O(1/\epsilon)$ 次调用。

**量子线性代数（HHL 与 QSVT）。** 给定矩阵 $A$ 的 block encoding，量子奇异值变换可以对其奇异值施加任意有界多项式，代价对维度 $N$ 是 $\mathrm{poly}\log N$，对条件数 $\kappa$ 是多项式。解线性方程、模拟哈密顿量、制备 Gibbs 态都由此得到。

### 2.2 四个隐藏成本

> [!note] 读任何量子加速主张时要检查的四件事
> 1. **输入。** 把 $N$ 维经典数据装进量子态或 QRAM 一般需要 $O(N)$ 时间，这会抹掉维度上的对数加速。
> 2. **输出。** 量子算法的结果是一个量子态。读出一个 $n$ 维向量的全部分量需要 $O(n/\epsilon^2)$ 次采样（层析），读出一个标量期望需要 $O(1/\epsilon)$。
> 3. **精度。** 很多量子界对 $1/\epsilon$ 是多项式依赖，经典内点法或牛顿法是 $\log(1/\epsilon)$。需要高精度的任务天然不利于量子算法。
> 4. **硬件。** 上述所有原语都要求容错量子计算机。当前含噪声的中等规模设备只能跑没有理论保证的变分算法。

**去量子化。** Tang 及 Chia 等人证明，若允许经典算法拥有对输入的采样查询访问（与量子算法的输入假设对等），则许多基于 QSVT 的量子机器学习算法都有只差多项式因子的经典对应。Le Gall 进一步证明这类去量子化对近似采样访问是鲁棒的。任何依赖 QRAM 输入的指数加速主张都要先过这一关。

## 3. 量子计算对优化与采样算法的已有加速

### 3.1 有严格证明的加速

| 问题 | 经典代价 | 量子代价 | 说明与文献 |
|---|---|---|---|
| 无结构搜索、约束满足 | $O(N)$ | $O(\sqrt N)$ | Grover；平方根最优 |
| 最小值查找 | $O(N)$ | $O(\sqrt N)$ | Dürr 与 Høyer |
| 蒙特卡洛均值估计 | $O(1/\epsilon^2)$ | $\tilde O(1/\epsilon)$ | Montanaro 2015 [^m15]；多层蒙特卡洛见 An 等 2021 |
| 可逆马尔可夫链混合 | $O(1/\delta)$ | $O(1/\sqrt\delta)$ | Szegedy 2004；一般情形带 $\sqrt N$ 因子，缓变链序列可去掉（Orsucci 等 2018） |
| 模拟退火的量子模拟 | 每个温度 $O(1/\delta)$ | 每个温度 $O(1/\sqrt\delta)$ | Somma 等 2007/2008；Wocjan 与 Abeyesinghe 2008 |
| 配分函数估计 | 谱间隙与精度均为一次方 | 二者均为平方根 | Wocjan 等 2009；Harrow 与 Wei 2020（自适应调度）；Arunachalam 等 2022；Cornelissen 与 Hamoudi 2023（对 $\log\lvert\Omega\rvert$ 次线性） |
| 马尔可夫链有限时间演化 | $O(t)$ | $O(\sqrt t)$ | Apers 与 Sarlette 2019 快进 |
| 不可逆链 | 无一般加速 | 有条件平方根 | Claudon 等 2025 |
| 对数凹分布采样与归一化常数 | 依赖维度 $d$ 与条件数 | 在精度与条件数上平方根改进 | Childs 等 2022 |
| 非对数凹分布采样 | 沿温度路径的退火 | 量子模拟退火版本 | Ozgul 等 2024；随机 oracle 下的加速见 Ozgul 等 2025 |
| Langevin 类连续动力学 | 混合时间 $\tau$ | 算符级加速 | Leng 等 2026 |
| 一般凸优化（成员 oracle） | $\tilde O(n^2)$ 次查询 | $\tilde O(n)$ 次查询 | Chakrabarti 等 2020 |
| 线性规划 | 内点法 $\tilde O(\sqrt n\,\mathrm{poly}(d))$ 行查询 | 同阶但输出经典解 | Apers 与 Gribling 2026 |
| 半定规划 | 见第 5 节 | 见第 5 节 | |
| 线性方程组 | $O(N\kappa)$ 或更好 | $O(\mathrm{poly}\log N\cdot\kappa/\epsilon)$ | HHL；QSVT（Gilyén 等 2019）；注意输入输出成本 |

上表有两个共同点。第一，除线性代数一行之外，全部是平方根级加速。第二，凡是涉及“沿温度或时间路径走一遍”的算法（退火、配分函数、缓变链），量子版本的加速都作用在每一步的混合代价上，而不是路径长度上。

### 3.2 启发式方法

**量子退火与 QAOA。** Kadowaki 与 Nishimori 1998 年提出用横场强度代替温度做退火，Santoro 等 2002 年在二维自旋玻璃上用路径积分蒙特卡洛观察到比热退火更低的残余能量，Morita 与 Nishimori 2007 年给出实时 Schrödinger 演化收敛的充分条件。Crosson 与 Harrow 2016 年证明模拟量子退火（一种经典 MCMC）在特定构造的实例上比模拟退火指数快，但这是对特定实例的结果。QAOA 及其热启动变体（Egger 等 2021；Tate 等 2023 用 Burer–Monteiro 低秩 SDP 解初始化）、量子松弛舍入（Dupont 等 2024）都没有一般性的理论优势。

**量子增强 MCMC。** Layden 等 2023 年在超导处理器上用量子演化生成 Metropolis 提议，对小规模 Ising 模型观察到多项式级的混合改进。Orfi 等 2024 年分析指出，这种改进需要精细调节量子淬火参数，否则谱间隙没有优势。

**连续优化的量子动力学。** 量子哈密顿下降（Leng 等 2023）、量子 Langevin 动力学（Chen 等 2025）和量子隧穿游走（Liu 等 2023，对局部极小近似全局的景观有证明）利用量子隧穿穿越经典梯度法无法穿越的壁垒。Herman 等 2025 年给出 Schrödinger 算符谱性质与经典 Langevin 混合时间之间的严格对应，作为这类方法可能有优势的机制。

### 3.3 已知的限制

- **NP 难问题没有多项式量子算法。** 组合优化中不存在任何 NP 完全问题被证明可由量子计算机多项式时间求解，普遍观点是不可能。
- **量子 Gibbs 采样器同样慢混合。** Gamarnik 等 2024 年把经典的瓶颈引理推广到量子 Gibbs 采样器，证明低温下存在无条件的混合时间下界。自旋玻璃的 overlap gap property 同时阻碍经典算法与低深度量子算法。
- **加速对象是谱间隙。** 若一个采样问题的经典算法已经不受混合时间限制，量子随机游走就没有可加速的对象。这一点是第 4 节的核心。

## 4. 量子计算与扩散模型

### 4.1 去噪过程是模拟退火吗

**相似之处。** 前向过程的边缘分布是数据分布与高斯核的卷积

$$
p_t(x) = \int p_0(x_0)\,\mathcal N\!\big(x;\,\alpha_t x_0,\ \sigma_t^2 I\big)\,dx_0 ,
$$

$\sigma_t$ 大时 $p_t$ 接近平坦高斯，$\sigma_t \to 0$ 时回到多峰的数据分布。反向时间 SDE

$$
dx = \big[f(x,t) - g(t)^2\,\nabla_x \log p_t(x)\big]\,dt + g(t)\,d\bar w
$$

每一步都是“沿 score 上升加噪声”，即 Langevin 动力学；沿 $\sigma_t$ 递减走完整条路径，就是带调度的退火 Langevin。Song 与 Ermon 2019 年的 NCSN 直接命名为 annealed Langevin dynamics，在每个噪声水平上跑若干步 Langevin 再降噪声，与模拟退火在每个温度上跑 Metropolis 再降温同构。Sohl-Dickstein 等 2015 年的原始论文本身就以非平衡统计物理为动机。

这一类比近年被严格化。Cordero-Encinar 等 2025 年给出“扩散路径上的退火 Langevin Monte Carlo”的非渐近收敛分析；Guo 等 2025 年证明退火 Langevin 对非对数凹目标的可证明收益；Chen 等 2026 年把扩散路径当作 MCMC 的插值路径，与几何调温路径对比。统计物理侧的 Biroli 等 2024、Ambrogioni 2025、Raya 与 Ambrogioni 2023 指出反向动力学存在对应对称性破缺的相变（speciation），Ghio 等 2024 年从自旋玻璃角度证明扩散采样器在经典退火失效的相变点上同样失效。Montanari 2023 年把扩散采样等同于 Eldan 的随机局部化，El Alaoui 等 2022、2023 年及 Huang 等 2024 年用它给出自旋玻璃 Gibbs 测度的多项式时间采样算法，同时用弱 Poincaré 不等式把随机局部化与模拟退火放进同一分析框架。

**关键差别。**

> [!formula] 两条平滑路径
> 模拟退火用幂次调温，扩散用高斯卷积：
> $$
> p_\beta(x) \propto e^{-\beta E(x)}, \qquad p_\sigma(x) = (p_0 * \mathcal N(0,\sigma^2 I))(x).
> $$
> 前者保持每个局部极小的位置，只改变陡峭程度；后者会把相邻的峰合并，在高噪声段直接消除一部分局部极小。Chehab 等 2024 年分别给出几何调温路径的上下界以及“膨胀路径”等实用扩散路径。

| | 模拟退火 | 扩散模型去噪 |
|---|---|---|
| 目标 | $T\to0$ 时的全局最小值（优化） | $\sigma\to0$ 时忠实采样整个分布 |
| 中间分布 | $e^{-\beta E}$ | $p_0 * \mathcal N(0,\sigma^2 I)$ |
| 梯度来源 | 解析的 $\nabla E$ | 学得的 $\nabla\log p_t$，由 Tweedie 公式等于 MMSE 去噪器 |
| 每个温度上的工作 | 等待马尔可夫链混合 | 一步或几步，不等待混合 |
| 代价的主导项 | 谱间隙 $1/\delta$ | score 网络前向传播次数 |

最后一行是决定性的。Chen 等 2023 年证明，给定 $L^2$ 精确的 score，反向过程在多项式时间内从几乎任何有界二阶矩的分布中采样；Benton 等 2024 年把步数收紧到对维度近线性的 $\tilde O(d\log^2(1/\delta)/\epsilon^2)$。扩散模型的“退火”不需要在每个温度上混合，因为穿越壁垒的信息已经在训练时编码进了 score。

**扩散作为优化器。** 当扩散被用于组合优化（DIFUSCO、T2T、DiffUCO 及其可扩展版本）或显式玻尔兹曼分布的采样（DDS、iDEM、PITA、扩散 Gibbs 采样、序贯控制 Langevin 扩散），目标变成了 $e^{-E/T}$ 的低温极限，此时它才真正是退火，也就重新面对退火的全部困难。Grenioux 等 2026 年对扩散退火玻尔兹曼生成器的元分析表明，它在模式权重估计上有系统性问题。

### 4.2 量子计算能加速哪一步

把去噪过程拆成三个部件，量子加速只能落在其中之一。

**（a）采样子程序。** 若扩散在做显式能量函数的采样或优化，那么每个噪声水平上的 Langevin 步可以换成量子随机游走或量子模拟退火，得到 $1/\delta \to 1/\sqrt\delta$ 的改进（Somma 等 2008；Ozgul 等 2024；Leng 等 2026）。这是第 3.1 节的结果直接搬过来，收益是平方根级，需要容错硬件，且在 overlap gap property 出现的硬实例上同样无效（Gamarnik 等 2024）。对数据驱动的扩散模型，这一步不存在，因为它本来就不等待混合。

**（b）模拟反向动力学。** Fokker–Planck 方程经变换 $p = \psi\sqrt{p_{\mathrm{eq}}}$ 变成虚时间 Schrödinger 方程，量子退火与模拟退火正是实时间与虚时间演化的对偶。三篇近期工作沿这条线走：Halmos 等 2026 年证明 score 采样与一族“score 哈密顿量”基态的绝热输运精确对应；Layden 等 2025 年证明流模型与概率流 ODE 的输运动力学对应一个特殊的连续变量 Schrödinger 方程，可以高效量子模拟；Wang 等 2025 年用 Carleman 线性化把概率流 ODE 变成线性系统再用量子线性系统求解器求解。Jin 等 2024 年的 Schrödingerization 可以把 Fokker–Planck 方程编码进酉演化，Kharazi 等 2026 年据此给出高维 Fokker–Planck 反应速率估计的可证明加速，An 等 2026 年给出线性耗散 ODE 的时间快进。

这些结果的共同障碍在第 2.2 节：漂移项包含学得的 score 网络，把它做成 block encoding 的成本极高；最终得到的是幅值编码的密度量子态，读出样本或统计量需要额外的测量开销。它们目前都是理论构造。

**（c）score 网络本身。** 量子神经网络推理（Kerenidis 等 2019 年的量子卷积网络；Guo 等 2024 年的量子 Transformer）依赖 QRAM 与 block encoding 输入，受第 2.2 节的去量子化结果约束。此外，学习高斯混合的 score 与连续 LWE 问题相关，Bruna 等 2021 年及 Gupte 等 2022 年的归约本身就是量子的，说明 score 学习的困难同样约束量子计算机。

### 4.3 “量子扩散模型”是另一个问题

Zhang 等 2024 年的 QuDDPM、Parigi 等 2024 年的量子噪声驱动扩散、Cacioppo 等 2023 年、Kwun 等 2024 年的混合态版本、Chen 与 Zhao 2024 年的全量子生成扩散、Liu 等 2025 年的测量式量子扩散，做的都是在量子态空间上定义扩散来生成量子数据，不宣称加速经典图像或文本生成。Kölle 等 2024 年用变分量子线路替换去噪网络，是启发式的小规模实验。Fayad 2026 年证明经典的“固定噪声改漂移即可反演”原理对量子通道一般不成立，Gabbassov 2026 年与 Bompais 等 2026 年为连续测量的量子轨迹推出反向随机 Schrödinger 方程。Cao 等 2026 年证明 QuDDPM 存在贫瘠高原。Huang 等 2025 年给出经典难以模拟却可高效训练的生成量子模型族，但针对的是生成分布本身的经典不可模拟性，而非扩散模型的加速。

**一个经典竞争者。** Coles 等 2023 年的热力学 AI 框架把扩散、退火和 Langevin 统一为可由物理噪声直接实现的算法；Jelinčič 等 2025 年和 Singh 等 2026 年用概率比特硬件直接运行类扩散的去噪链，Niazi 等 2024 年用 FPGA 上的稀疏 Ising 机训练深度玻尔兹曼网络。任何量子加速主张都要先赢过这类专用经典硬件。

### 4.4 小结

> [!conclusion] 扩散模型部分的结论
> 去噪过程在数学形式上是带学习势能的退火 Langevin，但它的设计目的正是绕过退火的混合瓶颈。量子采样算法能加速的恰恰是那个被绕过的瓶颈，所以对主流数据驱动的扩散模型帮助很小。理论上成立的两条路，量子模拟反向动力学和量子加速采样子程序，一条受输入输出成本约束，一条只在扩散被当作显式能量函数的采样器或优化器时才有对象，收益是平方根级。

## 5. 量子计算与半定规划

### 5.1 标准形式与决定复杂度的参数

$$
\min_{X\succeq0}\ \mathrm{Tr}(CX)\quad\text{s.t.}\quad \mathrm{Tr}(A_iX)=b_i,\ i=1,\dots,m,
$$

其中 $X\in\mathbb S^n$。量子 SDP 文献中反复出现的参数有：矩阵维度 $n$，约束数 $m$，每行非零元数 $s$，原始解的迹界 $R\ge\mathrm{Tr}\,X^\star$，对偶解的范数界 $r$，精度 $\epsilon$，以及（内点法类）牛顿系统的条件数 $\kappa$。记

$$
\gamma = \frac{Rr}{\epsilon}.
$$

### 5.2 量子 SDP 求解器

| 工作 | 复杂度 | 输入与输出 |
|---|---|---|
| Brandão 与 Svore 2017 | $\sqrt{nm}\,s^2\,\mathrm{poly}(\log n,\log m,R,r,1/\epsilon)$ | 稀疏 oracle；输出 Gibbs 态 |
| van Apeldoorn 等 2020 | $\tilde O\big(\sqrt{nm}\,s^2\gamma^8\big)$；下界 $\Omega(\sqrt m+\sqrt n)$ | 证明对 $\gamma$ 的多项式依赖不可避免 |
| van Apeldoorn 与 Gilyén 2019 | $\tilde O\big((\sqrt m+\sqrt n\,\gamma)\,\alpha\,\gamma^4\big)$ | block encoding 输入，$\alpha=s$ 对应稀疏访问 |
| Brandão 等 2019 | $\tilde O\big(s^2(\sqrt m\,\epsilon^{-10}+\sqrt n\,\epsilon^{-12})\big)$ | 归一化 SDP；量子态输入模型下 $\tilde O(\sqrt m\,\mathrm{poly}(\log n,r,R,1/\epsilon))$ |
| Brandão、França、Kueng 2022（Hamiltonian Updates） | $\tilde O\big(n^{1.5}(\sqrt s)^{1+o(1)}\mathrm{poly}(1/\epsilon)\big)$ | 仅对角约束的 MaxQP，误差 $\epsilon n\lVert A\rVert$ |
| Kerenidis 与 Prakash 2020（QIPM） | $\tilde O\big(n^{2.5}\xi^{-2}\mu\kappa^3\log(1/\epsilon)\big)$ | QRAM；层析读出经典解 |
| Augustino 等 2023（QIPM） | 同类，$m=O(n^2)$ 稠密设定 | 输出经典原对偶三元组 |
| Huang 等 2022（鲁棒 IPM 量子版） | $(mn^{1.5}+n^3)\,\mathrm{poly}(\kappa,\log(mn/\epsilon))$ | 输出经典解 |
| Mohammadisiahroudi 等 2026 | 低精度量子解上做迭代精化 | 缓解量子解精度不足 |
| Stilck França 等 2025 | 阶数 $k$ 的 Lasserre 松弛：$O(n^k\epsilon^{-4}+n^{k/2}\epsilon^{-5})$ | 假设松弛精确且最优点在 $\ell_1$ 球内 |

**NISQ 与量子启发。** 变分 SDP（Bharti 等 2022；Patel、Coles 与 Wilde 2024；QSlack）把原变量参数化为量子线路制备的态，只对特定输入模型有保证。割平面法把特征值 oracle 换成 VQE（Marecek 等 2025；Ozbaygin 等 2026），迭代复杂度不变，实验规模到 $n=150$。量子启发的经典算法：Chia 等 2020 年对约束矩阵与代价矩阵都低秩的 SDP 给出次线性算法；Yu 等 2022 年用可分量子态刻画秩约束 SDP，得到多项式规模的经典层级；Nana Liu 等 2025 年与 Minervini 等 2026 年把 SDP 重写为 Jaynes 意义下的自由能极小化，梯度由 Gibbs 态的期望给出。

### 5.3 下界、资源估计与批评

- van Apeldoorn 等证明任何量子 SDP 求解器需要 $\Omega(\sqrt m+\sqrt n)$ 次查询，且对 $\gamma$ 的多项式依赖无法避免；他们同时指出，凡是 $Rr/\epsilon$ 随 $n$ 增长的“组合型”SDP，量子求解器都不比经典快。
- Dalzell 等 2023 年对 QIPM 做端到端资源估计：约 $800n^2$ 个逻辑量子比特，T 深度约 $10^{10}\kappa_F n^{1.5}\xi^{-2}$ 量级，结论是即便容错也难以胜过经典内点法。
- Augustino 等 2023 年“矩阵乘法时间内解 QUBO 松弛、量子更快”的主张已由作者标注为不成立。
- Henze 等 2025 年对 Hamiltonian Updates 做非渐近分析，$n=1024$ 的实例中量子成本由估计对角元主导；Ostermann 等 2025 年在真实物流 QUBO 上比较量子与经典 SDP 松弛。
- Aaronson 2015 年的评论概括了 HHL 类算法的四个前提：输入可快速装载、矩阵稀疏或可 block encode、条件数小、只需要解的某个统计量而非全部分量。

### 5.4 机器人学中的 SDP 及其结构

| 问题类别 | 代表工作 | SDP 结构 | 求解方式 | 在线还是离线 |
|---|---|---|---|---|
| 位姿图优化、旋转平均 | SE-Sync（Rosen 等 2019）；Shonan（Dellaert 等 2020）；Eriksson 等 2018 | $n=dN$，块对角约束 $X_{ii}=I_d$，$m=N\,d(d+1)/2$，代价为稀疏连接拉普拉斯，解秩 $d$，$\mathrm{Tr}\,X=n$ | Burer–Monteiro 与 Riemannian staircase | 在线，秒级 |
| 含外点的鲁棒估计 | QUASAR（Yang 与 Carlone 2019）；TEASER 2020；STRIDE（Yang 等 2022、2023） | 四元数升维 $4(N+1)$；TLS 稀疏二阶松弛的最大块 $(1+d)(1+N)$，$m=O(n_1^2)$，解秩 1 且原始退化 | MOSEK 或 STRIDE，分钟级 | 半在线 |
| 相对位姿、标定、形状 | Briales 等 2018；Garcia-Salguero 等 2021；Tirado-Garín 等 2024；Giamou 等 2019；Wise 等 2026；Shi 等 2021 | 十到几十维的小 QCQP 松弛，秩 1 | 内点法，毫秒到秒 | 在线 |
| 距离与地标定位 | CORA（Papalia 等 2024，维度约 16000）；Holmes 等 2023；AutoTight（Dümbgen 等 2024）；Barfoot 等 2025；Korotkine 等 2025 | 需要冗余约束才紧；MOSEK 容差 $10^{-10}$ | Riemannian staircase 或 MOSEK | 在线到半在线 |
| 大规模与工程化 | Building Rome（Han 等 2025，$X\in\mathbb S^{30465}$）；弦稀疏（Dümbgen 等 2024；Subramanian 等 2026）；Xu 等 2026；CP-Cert（Holmes 等 2026）；SDPRLayers | 弦稀疏图，Bayes 树团即最大团 | 分解后的 ADMM 或 BM | 在线 |
| SOS 控制综合与验证 | 漏斗库（Majumdar 与 Tedrake 2017）；占用测度（Majumdar 等 2014；Zhao 等 2020）；CLF 与 CBF（Dai 等 2023）；DSOS（Ahmadi 与 Majumdar 2019） | Gram 矩阵多块，约束数随多项式次数组合增长 | 内点法 | 离线 |
| 无碰撞认证与规划 | C-IRIS（Dai 等 2024）；Amice 等 2024；Li 等 2024；Graesdal 等 2026；Morozov 等 2024 | 大量独立的小 SOS 程序 | 并行内点法 | 离线 |
| 轨迹优化与接触 | STROM（Kang 等 2024）；Kang 等 2025；CARDAL（Li 等 2026） | 链状稀疏 Lasserre 二阶松弛，每阶段一个 $190\times190$ 矩块 | GPU ADMM，秒级 | 在线边缘 |
| 在线控制 | TinySDP（Mahajan 等 2026，微控制器上 25 Hz）；协方差引导（Rapakoulias 等 2023）；KernelSOS（Groudiev 等 2025） | 状态维乘时域，规模小 | ADMM 或内点法 | 在线 |
| 神经网络控制器验证 | Reach-SDP（Hu 等 2020）；DeepSDP；LipSDP（Fazlyab 等） | 维度等于神经元总数，只需要界 | 内点法，扩展性差 | 离线 |
| 逆运动学与多机器人 | CIDGIK（Giamou 等 2022）；IKSPARK；DC2-PGO（Tian 等 2021）；DCORA（Thoms 等 2025）；Wu 等 2023；Wang 等 2024 | 秩 $d$ 约束的 Gram 矩阵；分布式 BM | 凸迭代或分布式 Riemannian 块坐标下降 | 在线 |

共同特点有四条：解低秩；约束块对角、代价稀疏；$\mathrm{Tr}\,X^\star$ 随规模线性增长；要得到可认证的低秩解并检验对偶证书 $S=C-\Lambda^\star\succeq0$，需要 $10^{-6}$ 到 $10^{-10}$ 的精度，而紧的松弛在最优点必然原始退化。

### 5.5 参数代入：为什么不匹配

以 SE-Sync 为例。$n=dN$，$R=\mathrm{Tr}\,X^\star=n$，因此 $\gamma\ge n/\epsilon$。代入 van Apeldoorn 与 Gilyén 的界，量子运行时间至少含

$$
\sqrt n\,\gamma^5 \;\gtrsim\; n^{5.5}\,\epsilon^{-5}
$$

量级的因子；代入 Brandão 等 2019 年归一化 SDP 的界，$\epsilon^{-12}$ 的精度依赖在 $\epsilon\sim10^{-6}$ 时是天文数字。经典 Riemannian staircase 每步代价 $O(\mathrm{nnz}(C)\,p)$，实践中几乎总是一步认证。这不是常数因子的差距，而是渐近上的反向差距，且 van Apeldoorn 等的下界说明这是模型层面的限制。

内点法类量子算法输出经典解，但依赖 $\kappa$。机器人 SDP 恰好是原始退化的，STRIDE 与 CP-Cert 两篇工作专门处理这一点，趋近最优时 $\kappa$ 爆炸。Huang 等 2022 年的 $(mn^{1.5}+n^3)\mathrm{poly}(\kappa)$ 在 $m\sim n$ 时是 $n^{2.5}\mathrm{poly}(\kappa)$，对比经典 BM 的近线性代价没有优势。

再加两条结构性障碍。机器人需要的是秩 $d$ 因子 $Y\in\mathbb R^{n\times d}$ 本身，从量子态读出它至少需要 $O(nd/\epsilon^2)$ 次采样；在线问题的延迟预算是毫秒到秒，而任何有保证的量子 SDP 算法都要求容错硬件和 QRAM。与此同时经典侧进展极快：弦稀疏自动分解进入 GTSAM，多 GPU 求解器在秒级解决二阶矩松弛，微控制器上能跑 SDP 型 MPC，Shaikewitz 等 2026 年甚至把类别级位姿形状估计从 SDP 简化为 $4\times4$ 非线性特征问题在毫秒内求解。

> [!note] 已有的量子与机器人视觉交叉工作走的是另一条路
> QuantumSync（Birdal 等 2021）、Q-Match（2021）、QuMoSeg（2022）、QuAnt（2023）、CCuantuMM（2023）、QuMF（2023）、Doan 等 2022 年带证书的混合鲁棒拟合、Q-FW（Yurtsever 等 2022）、量子多重旋转平均 IQARS（Wang 等 2026，15 个旋转约需 2500 个物理比特）都绕开 SDP，把同步、匹配和旋转平均改写为 QUBO 交给量子退火机，规模停留在几十到几百个逻辑比特，多数没有认证环节。量子 LM 束调整（Bernecker 等 2022）与 QSVT 版 LQG（Dehaghani 等 2025）、大规模线性控制的量子策略梯度（Clayton 等 2024）属于线性代数加速而非 SDP。综述见 Meli 等 2025。

### 5.6 可能的算法方向

**唯一有理论文献支撑的切入点是 Lasserre 层级。** Stilck França 等 2025 年针对阶数 $k$ 的矩松弛给出 $O(n^k\epsilon^{-4}+n^{k/2}\epsilon^{-5})$，而经典方法处理 $\binom{n+k}{k}\sim n^k$ 维矩阵至少需要 $n^{2k}$ 量级。这正对应 STRIDE、STROM 和接触规划所用的二阶松弛。前提是松弛精确（机器人问题通常靠冗余约束才紧）且只能给低精度解，所以它适合的位置是下面的流水线，而不是直接替换求解器。

**量子启发的经典低精度求解器。** 把 SDP 的角色从“求精确解”降为“给出粗初值”：用矩阵指数乘性权重迭代

$$
X_{t+1}\ \propto\ \exp\!\Big(-\eta\sum_{k\le t}\big(C+\Lambda_k\big)\Big)
$$

配合 Chebyshev 或 Lanczos 近似，每步 $O(\mathrm{nnz})$；取前 $d$ 个特征向量舍入到 $\mathrm{SO}(d)^N$，做黎曼局部精化，最后用廉价的对偶证书认证。Henze 等已指出舍入后精度要求可以放松，Nana Liu 等的“SDP 等于自由能极小化”给出理论框架。风险在于 Burer–Monteiro 已经很快，该方案只在 staircase 失效的多局部极小场景下有意义。

**秩约束的量子启发层级。** Yu 等 2022 年用可分量子态刻画秩约束 SDP。CIDGIK 与 IKSPARK 中的“秩 $d$”约束目前靠凸迭代启发式处理，这个层级可能给出更紧的凸松弛。

**认证估计作为基态制备。** 松弛紧时 $X^\star/\mathrm{Tr}\,X^\star$ 是秩 $d$ 密度矩阵，恰是对偶证书矩阵 $S(\Lambda^\star)=C-\Lambda^\star$ 的零能基态。于是位姿图优化等价于对 $m$ 个块对角对偶变量做外层搜索、内层制备稀疏哈密顿量的基态，代价由谱间隙 $\Delta$ 决定，$\Delta$ 与测量图的代数连通度和噪声水平相关。这个视角与 Halmos 等的 score 哈密顿量以及 Nana Liu 等的热力学框架一致，理论上干净，但读出成本仍是 $O(n/\epsilon^2)$。

**混合“量子提议、经典认证”。** Doan 等 2022 年和 Q-FW 已经在视觉中把量子退火嵌入带证书的框架。移到 TLS 鲁棒感知上，让退火机处理外点二值变量，让 SDP 对偶证书做最后验证。只在退火机优于 GNC 等经典启发式时才有意义，目前无此证据。

**离线 SOS 设计问题。** C-IRIS、漏斗库和接触规划的高阶松弛离线求解，延迟不敏感，约束数随多项式次数组合爆炸，是 Stilck França 等算法唯一可能有用的场景，仍需容错硬件。

### 5.7 小结

> [!conclusion] SDP 部分的结论
> 以现有量子 SDP 算法，机器人 SDP 得不到渐近加速，原因是迹界随规模增长、认证精度要求高以及原始退化导致条件数恶化，三者都直接击中量子算法的弱点，而 van Apeldoorn 等的下界说明这是模型层面的限制。可行的产出是算法思想迁移：低精度矩阵指数迭代加舍入加对偶证书的流水线，针对秩约束的量子启发层级，以及把认证估计看作基态制备的统一视角。

## 6. 文献

### 6.1 量子加速的原语与限制

1. Szegedy, Quantum speed-up of Markov chain based algorithms, FOCS 2004. <https://doi.org/10.1109/FOCS.2004.53>
2. Somma, Boixo, Barnum, Knill, Quantum simulations of classical annealing processes, PRL 2008. <https://arxiv.org/abs/0804.1571>；早期版本 Quantum simulated annealing, <https://arxiv.org/abs/0712.1008>
3. Wocjan, Abeyesinghe, Speed-up via quantum sampling, PRA 2008. <https://arxiv.org/abs/0804.4259>
4. Wocjan, Chiang, Nagaj, Abeyesinghe, Quantum speed-up for approximating partition functions, PRA 2009. <https://arxiv.org/abs/0811.0596>
5. Boixo, Knill, Somma, Quantum state preparation by phase randomization, QIC 2009. <https://arxiv.org/abs/0903.1652>
6. Montanaro, Quantum speedup of Monte Carlo methods, Proc. R. Soc. A 2015. <https://arxiv.org/abs/1504.06987>
7. Orsucci, Briegel, Dunjko, Faster quantum mixing for slowly evolving sequences of Markov chains, Quantum 2018. <https://arxiv.org/abs/1503.01334>
8. Apers, Sarlette, Quantum fast-forwarding: Markov chains and graph property testing, 2019. <https://arxiv.org/abs/1804.02321>
9. Gilyén, Su, Low, Wiebe, Quantum singular value transformation and beyond, STOC 2019. <https://arxiv.org/abs/1806.01838>
10. Harrow, Wei, Adaptive quantum simulated annealing for Bayesian inference and estimating partition functions, SODA 2020. <https://arxiv.org/abs/1907.09965>
11. Arunachalam, Havlicek, Nannicini, Temme, Wocjan, Simpler (classical) and faster (quantum) algorithms for Gibbs partition functions, Quantum 2022. <https://arxiv.org/abs/2009.11270>
12. Cornelissen, Hamoudi, A sublinear-time quantum algorithm for approximating partition functions, SODA 2023. <https://arxiv.org/abs/2207.08643>
13. Chakrabarti, Childs, Li, Wu, Quantum algorithms and lower bounds for convex optimization, Quantum 2020. <https://arxiv.org/abs/1809.01731>
14. Childs, Li, Liu, Wang, Zhang, Quantum algorithms for sampling log-concave distributions and estimating normalizing constants, NeurIPS 2022. <https://arxiv.org/abs/2210.06539>
15. Ozgul, Li, Mahdavi, Wang, Stochastic quantum sampling for non-logconcave distributions and estimating partition functions, ICML 2024. <https://arxiv.org/abs/2310.11445>
16. Ozgul, Li, Mahdavi, Wang, Quantum speedups for sampling and non-convex optimization with stochastic oracles, 2025. <https://arxiv.org/abs/2504.03626>
17. Leng, Ding, Chen, Lin, Operator-level quantum acceleration of non-logconcave sampling, PNAS 2026. <https://arxiv.org/abs/2505.05301>
18. Claudon 等, Quantum speedup for nonreversible Markov chains, Nature Communications 2025. <https://arxiv.org/abs/2501.05868>
19. An, Linden, Liu, Montanaro, Shao, Wang, Quantum-accelerated multilevel Monte Carlo methods for stochastic differential equations in mathematical finance, Quantum 2021. <https://arxiv.org/abs/2012.06283>
20. Apers, Gribling, Quantum speedups for linear programming via interior point methods, SIAM J. Comput. 2026. <https://arxiv.org/abs/2311.03215>
21. Kadowaki, Nishimori, Quantum annealing in the transverse Ising model, PRE 1998. <https://arxiv.org/abs/cond-mat/9804280>
22. Santoro, Martoňák, Tosatti, Car, Theory of quantum annealing of an Ising spin glass, Science 2002. <https://arxiv.org/abs/cond-mat/0205280>
23. Morita, Nishimori, Convergence of quantum annealing with real-time Schrödinger dynamics, JPSJ 2007. <https://arxiv.org/abs/quant-ph/0702252>
24. Crosson, Harrow, Simulated quantum annealing can be exponentially faster than classical simulated annealing, FOCS 2016. <https://arxiv.org/abs/1601.03030>
25. Layden 等, Quantum-enhanced Markov chain Monte Carlo, Nature 2023. <https://arxiv.org/abs/2203.12497>
26. Orfi, Sels, Quantum enhanced Markov chains require fine-tuned quenches, 2024. <https://arxiv.org/abs/2408.07881>
27. Leng, Hickman, Li, Wu, Quantum Hamiltonian descent, 2023. <https://arxiv.org/abs/2303.01471>
28. Chen, Lu, Wang, Liu, Li, Quantum Langevin dynamics for optimization, CMP 2025. <https://arxiv.org/abs/2311.15587>
29. Liu, Su, Li, On quantum speedups for nonconvex optimization via quantum tunneling walks, Quantum 2023. <https://arxiv.org/abs/2209.14501>
30. Herman 等, Mechanisms for quantum advantage in global optimization of nonconvex functions, 2025. <https://arxiv.org/abs/2510.03385>
31. Gamarnik, Kiani, Zlokapa 等, Slow mixing of quantum Gibbs samplers, 2024. <https://arxiv.org/abs/2411.04300>
32. Egger, Mareček, Woerner, Warm-starting quantum optimization, Quantum 2021. <https://arxiv.org/abs/2009.10095>
33. Tate, Farhadi, Herold, Mohler, Gupta, Bridging classical and quantum with SDP initialized warm-starts for QAOA, ACM TQC 2023. <https://arxiv.org/abs/2010.14021>
34. Dupont, Sundar, Extending relax-and-round combinatorial optimization solvers with quantum correlations, PRA 2024. <https://arxiv.org/abs/2307.05821>
35. Aaronson, Read the fine print, Nature Physics 2015. <https://www.nature.com/articles/nphys3272>
36. Chia, Gilyén, Li, Lin, Tang, Wang, Sampling-based sublinear low-rank matrix arithmetic framework for dequantizing quantum machine learning, STOC 2020. <https://arxiv.org/abs/1910.06151>
37. Le Gall, Robust dequantization of the quantum singular value transformation and quantum machine learning algorithms, Computational Complexity 2025. <https://arxiv.org/abs/2304.04932>

### 6.2 扩散模型与退火

38. Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli, Deep unsupervised learning using nonequilibrium thermodynamics, ICML 2015. <https://arxiv.org/abs/1503.03585>
39. Song, Ermon, Generative modeling by estimating gradients of the data distribution, NeurIPS 2019. <https://arxiv.org/abs/1907.05600>
40. Chen, Chewi, Li, Li, Salim, Zhang, Sampling is as easy as learning the score, ICLR 2023. <https://arxiv.org/abs/2209.11215>
41. Benton, De Bortoli, Doucet, Deligiannidis, Nearly $d$-linear convergence bounds for diffusion models via stochastic localization, ICLR 2024. <https://arxiv.org/abs/2308.03686>
42. Cordero-Encinar, Akyildiz, Duncan, Non-asymptotic analysis of diffusion annealed Langevin Monte Carlo for generative modelling, 2025. <https://arxiv.org/abs/2502.09306>
43. Guo, Tao, Chen, Provable benefit of annealed Langevin Monte Carlo for non-log-concave sampling, ICLR 2025. <https://arxiv.org/abs/2407.16936>
44. Chen 等, Markov chain Monte Carlo with diffusion paths, 2026. <https://arxiv.org/abs/2607.11631>
45. Chehab 等, Provable convergence and limitations of geometric tempering for Langevin dynamics, 2024. <https://arxiv.org/abs/2410.09697>
46. Chehab, Korba, A practical diffusion path for sampling, 2024. <https://arxiv.org/abs/2406.14040>
47. Biroli, Bonnaire, De Bortoli, Mézard, Dynamical regimes of diffusion models, Nature Communications 2024. <https://arxiv.org/abs/2402.18491>
48. Ambrogioni, The statistical thermodynamics of generative diffusion models, Entropy 2025. <https://arxiv.org/abs/2310.17467>
49. Raya, Ambrogioni, Spontaneous symmetry breaking in generative diffusion models, NeurIPS 2023. <https://arxiv.org/abs/2305.19693>
50. Ghio, Dandi, Krzakala, Zdeborová, Sampling with flows, diffusion and autoregressive neural networks: a spin-glass perspective, PNAS 2024. <https://arxiv.org/abs/2308.14085>
51. Montanari, Sampling, diffusions, and stochastic localization, 2023. <https://arxiv.org/abs/2305.10690>
52. El Alaoui, Montanari, Sellke, Sampling from the Sherrington–Kirkpatrick Gibbs measure via algorithmic stochastic localization, 2022. <https://arxiv.org/abs/2203.05093>
53. El Alaoui, Montanari, Sellke, Sampling from mean-field Gibbs measures via diffusion processes, 2023. <https://arxiv.org/abs/2310.08912>
54. Huang, Montanari, Pham, Sampling from spherical spin glasses in total variation via algorithmic stochastic localization, 2024. <https://arxiv.org/abs/2404.15651>
55. Huang 等, Weak Poincaré inequalities, simulated annealing, and sampling from spherical spin glasses, 2024. <https://arxiv.org/abs/2411.09075>
56. Cotler, Rezchikov, Renormalizing diffusion models, 2023. <https://arxiv.org/abs/2308.12355>
57. Doucet, Grathwohl, Matthews, Strathmann, Score-based diffusion meets annealed importance sampling, NeurIPS 2022. <https://arxiv.org/abs/2208.07698>
58. Vargas, Grathwohl, Doucet, Denoising diffusion samplers, ICLR 2023. <https://arxiv.org/abs/2302.13834>
59. Akhound-Sadegh 等, Iterated denoising energy matching for sampling from Boltzmann densities, ICML 2024. <https://arxiv.org/abs/2402.06121>
60. Akhound-Sadegh 等, Progressive inference-time annealing of diffusion models for sampling from Boltzmann densities, NeurIPS 2025. <https://arxiv.org/abs/2506.16471>
61. Chen, Zhang, Hernández-Lobato, Diffusive Gibbs sampling, ICML 2024. <https://arxiv.org/abs/2402.03008>
62. Chen 等, Sequential controlled Langevin diffusions, ICLR 2025. <https://arxiv.org/abs/2412.07081>
63. Grenioux 等, Diffusion-based annealed Boltzmann generators: benefits, pitfalls and hopes, TMLR 2026. <https://arxiv.org/abs/2601.21026>
64. Sun, Yang, DIFUSCO: graph-based diffusion solvers for combinatorial optimization, NeurIPS 2023. <https://arxiv.org/abs/2302.08224>
65. Sanokowski, Hochreiter, Lehner, A diffusion model framework for unsupervised neural combinatorial optimization, ICML 2024. <https://arxiv.org/abs/2406.01661>
66. Sanokowski 等, Scalable discrete diffusion samplers: combinatorial optimization and statistical physics, ICLR 2025. <https://arxiv.org/abs/2502.08696>
67. Bruna, Regev, Song, Tang, Continuous LWE, STOC 2021. <https://arxiv.org/abs/2005.09595>
68. Gupte, Vafa, Vaikuntanathan, Continuous LWE is as hard as LWE and applications to learning Gaussian mixtures, FOCS 2022. <https://arxiv.org/abs/2204.02550>

### 6.3 量子计算与扩散模型

69. Halmos 等, The Score Hamiltonian: mapping diffusion models to adiabatic transport, 2026. <https://arxiv.org/abs/2606.05217>
70. Layden 等, Wavefunction flows: efficient quantum simulation of continuous flow models, 2025. <https://arxiv.org/abs/2510.08462>
71. Wang 等, Towards efficient quantum algorithms for diffusion probabilistic models, 2025. <https://arxiv.org/abs/2502.14252>
72. Jin, Liu, Yu, Quantum simulation of partial differential equations via Schrödingerisation, PRL 2024. <https://arxiv.org/abs/2212.13969>
73. Jin, Liu, Yu, Quantum simulation of the Fokker–Planck equation via Schrödingerization, 2024. <https://arxiv.org/abs/2404.13585>
74. Kharazi 等, Provable quantum speedups for reaction-rate estimation in high-dimensional Fokker–Planck dynamics, 2026. <https://arxiv.org/abs/2601.15523>
75. An, Childs, Lin, Fast-forwarding quantum algorithms for linear dissipative differential equations, Quantum 2026. <https://arxiv.org/abs/2410.13189>
76. Krovi, Improved quantum algorithms for linear and nonlinear differential equations, Quantum 2023. <https://arxiv.org/abs/2202.01054>
77. Kerenidis, Landman, Prakash, Quantum algorithms for deep convolutional neural networks, ICLR 2020. <https://arxiv.org/abs/1911.01117>
78. Guo 等, Quantum Transformer: accelerating model inference via quantum linear algebra, 2024. <https://arxiv.org/abs/2402.16714>
79. Zhang, Xu, Chen, Zhuang, Generative quantum machine learning via denoising diffusion probabilistic models, PRL 2024. <https://arxiv.org/abs/2310.05866>
80. Parigi, Martina, Caruso, Quantum-noise-driven generative diffusion models, Adv. Quantum Technol. 2024. <https://arxiv.org/abs/2308.12013>
81. Cacioppo, Colantonio, Bordoni, Giagu, Quantum diffusion models, 2023. <https://arxiv.org/abs/2311.15444>
82. Kwun, Zhang, Zhuang, Mixed-state quantum denoising diffusion probabilistic model, 2024. <https://arxiv.org/abs/2411.17608>
83. Chen, Zhao, Quantum generative diffusion model, 2024. <https://arxiv.org/abs/2401.07039>
84. Kölle 等, Quantum denoising diffusion models, 2024. <https://arxiv.org/abs/2401.07049>
85. Liu 等, Measurement-based quantum diffusion models, 2025. <https://arxiv.org/abs/2508.08799>
86. Cao 等, Mitigating barren plateaus in quantum denoising diffusion probabilistic model, PRA 2026. <https://arxiv.org/abs/2512.06695>
87. Fayad, Score reversal is not free for quantum diffusion models, 2026. <https://arxiv.org/abs/2603.06488>
88. Gabbassov, Stochastic Schrödinger equations for quantum reverse diffusion, PRR 2026. <https://arxiv.org/abs/2511.15919>
89. Bompais 等, Generating quantum ensembles via reverse-time quantum diffusions, 2026. <https://arxiv.org/abs/2606.03848>
90. Huang 等, Generative quantum advantage for classical and quantum problems, 2025. <https://arxiv.org/abs/2509.09033>
91. Coles 等, Thermodynamic AI and the fluctuation frontier, 2023. <https://arxiv.org/abs/2302.06584>
92. Jelinčič 等, An efficient probabilistic hardware architecture for diffusion-like models, 2025. <https://arxiv.org/abs/2510.23972>
93. Singh 等, From independent to correlated diffusion: generalized generative modeling with probabilistic computers, 2026. <https://arxiv.org/abs/2603.27996>
94. Niazi 等, Training deep Boltzmann networks with sparse Ising machines, Nature Electronics 2024. <https://arxiv.org/abs/2303.10728>

### 6.4 量子 SDP

95. Brandão, Svore, Quantum speed-ups for solving semidefinite programs, FOCS 2017. <https://arxiv.org/abs/1609.05537>
96. van Apeldoorn, Gilyén, Gribling, de Wolf, Quantum SDP-solvers: better upper and lower bounds, Quantum 2020. <https://arxiv.org/abs/1705.01843>
97. van Apeldoorn, Gilyén, Improvements in quantum SDP-solving with applications, ICALP 2019. <https://arxiv.org/abs/1804.05058>
98. Brandão, Kalev, Li, Lin, Svore, Wu, Quantum SDP solvers: large speed-ups, optimality, and applications to quantum learning, ICALP 2019. <https://arxiv.org/abs/1710.02581>
99. Brandão, França, Kueng, Faster quantum and classical SDP approximations for quadratic binary optimization, Quantum 2022. <https://arxiv.org/abs/1909.04613>
100. Kerenidis, Prakash, A quantum interior point method for LPs and SDPs, ACM TQC 2020. <https://arxiv.org/abs/1808.09266>
101. Augustino, Nannicini, Terlaky, Zuluaga, Quantum interior point methods for semidefinite optimization, Quantum 2023. <https://arxiv.org/abs/2112.06025>
102. Huang, Jiang, Song, Tao, Zhang, A faster quantum algorithm for semidefinite programming via robust IPM framework, 2022. <https://arxiv.org/abs/2207.11154>
103. Mohammadisiahroudi 等, Quantum computing inspired iterative refinement for semidefinite optimization, Math. Prog. 2026. <https://arxiv.org/abs/2312.11253>
104. Mohammadisiahroudi 等, Quantum interior point methods: a review of developments and an optimally scaling framework, 2025. <https://arxiv.org/abs/2512.06224>
105. Augustino 等, A quantum central path algorithm for linear optimization, 2023. <https://arxiv.org/abs/2311.03977>
106. Stilck França 等, Quantum speed-ups for solving semidefinite relaxations of polynomial optimization, 2025. <https://arxiv.org/abs/2511.14389>
107. Dalzell 等, End-to-end resource analysis for quantum interior point methods and portfolio optimization, PRX Quantum 2023. <https://arxiv.org/abs/2211.12489>
108. Augustino 等, Solving the semidefinite relaxation of QUBOs in matrix multiplication time, and faster with a quantum computer, 2023（量子加速主张已由作者撤回）. <https://arxiv.org/abs/2301.04237>
109. Henze 等, Solving quadratic binary optimization problems using quantum SDP methods: non-asymptotic running time analysis, 2025. <https://arxiv.org/abs/2502.15426>
110. Ostermann 等, Benchmarking of quantum and classical SDP relaxations for QUBO formulations of real-world logistics problems, 2025. <https://arxiv.org/abs/2503.10801>
111. Bharti, Haug, Vedral, Kwek, Noisy intermediate-scale quantum algorithm for semidefinite programming, PRA 2022. <https://arxiv.org/abs/2106.03891>
112. Patel, Coles, Wilde, Variational quantum algorithms for semidefinite programming, Quantum 2024. <https://arxiv.org/abs/2112.08859>
113. Chen 等, QSlack: a slack-variable approach for variational quantum semi-definite programming, PRA 2025. <https://arxiv.org/abs/2312.03830>
114. Mareček 等, A cutting-plane method for semidefinite programming with potential applications on noisy quantum devices, Allerton 2025. <https://arxiv.org/abs/2110.03400>
115. Ozbaygin 等, A variational quantum eigensolver-based cutting plane framework for semidefinite programming problems, 2026. <https://arxiv.org/abs/2609.02139>
116. Chia, Li, Lin, Wang, Quantum-inspired sublinear algorithm for solving low-rank semidefinite programming, MFCS 2020. <https://arxiv.org/abs/1901.03254>
117. Yu, Simnacher, Wyderka, Gühne 等, Quantum-inspired hierarchy for rank-constrained optimization, PRX Quantum 2022. <https://arxiv.org/abs/2012.00554>
118. Liu 等, Quantum thermodynamics and semi-definite optimization, 2025. <https://arxiv.org/abs/2505.04514>
119. Minervini 等, Quantum thermodynamics and semidefinite optimization: Boltzmann, Fermi–Dirac, and Bose–Einstein frameworks, 2026. <https://arxiv.org/abs/2608.21123>

### 6.5 机器人学中的 SDP

120. Rosen, Carlone, Bandeira, Leonard, SE-Sync: a certifiably correct algorithm for synchronization over the special Euclidean group, IJRR 2019. <https://arxiv.org/abs/1612.07386>
121. Dellaert 等, Shonan rotation averaging: global optimality by surfing $SO(p)^n$, ECCV 2020. <https://arxiv.org/abs/2008.02737>
122. Eriksson, Olsson, Kahl, Chin, Rotation averaging and strong duality, CVPR 2018. <https://arxiv.org/abs/1705.01362>
123. Fan, Wang, Murphey, CPL-SLAM, T-RO 2020. <https://arxiv.org/abs/2007.06708>
124. Yang, Carlone, A quaternion-based certifiably optimal solution to the Wahba problem with outliers, ICCV 2019. <https://arxiv.org/abs/1905.12536>
125. Yang, Shi, Carlone, TEASER: fast and certifiable point cloud registration, T-RO 2020. <https://arxiv.org/abs/2001.07715>
126. Yang, Carlone, Certifiably optimal outlier-robust geometric perception: semidefinite relaxations and scalable global optimization, TPAMI 2022. <https://arxiv.org/abs/2109.03349>
127. Yang, Liang, Carlone, Toh, An inexact projected gradient method with rounding and lifting by nonlinear programming for solving rank-one semidefinite relaxation of polynomial optimization, Math. Prog. 2023. <https://arxiv.org/abs/2105.14033>
128. Peng, Fazlyab, Vidal, Towards understanding the semidefinite relaxations of truncated least-squares in robust rotation search, 2022. <https://arxiv.org/abs/2207.08350>
129. Briales, Kneip, Gonzalez-Jimenez, A certifiably globally optimal solution to the non-minimal relative pose problem, CVPR 2018.
130. Garcia-Salguero, Briales, Gonzalez-Jimenez, Fast and robust certifiable estimation of the relative pose between two calibrated cameras, JMIV 2021. <https://arxiv.org/abs/2101.08524>
131. Tirado-Garín, Civera, From correspondences to pose: non-minimal certifiably optimal relative pose without disambiguation, CVPR 2024. <https://arxiv.org/abs/2312.05995>
132. Härenstam-Nielsen, Zeller, Cremers, Semidefinite relaxations for robust multiview triangulation, CVPR 2023. <https://arxiv.org/abs/2301.11431>
133. Giamou 等, Certifiably globally optimal extrinsic calibration from per-sensor egomotion, RA-L 2019. <https://arxiv.org/abs/1809.03554>
134. Wise 等, A certifiably correct algorithm for generalized robot-world and hand-eye calibration, IJRR 2026. <https://arxiv.org/abs/2507.23045>
135. Shi, Yang, Carlone, Optimal pose and shape estimation for category-level 3D object perception, RSS 2021. <https://arxiv.org/abs/2104.08383>
136. Yang, Carlone, In perfect shape: certifiably optimal 3D shape reconstruction from 2D landmarks, CVPR 2020. <https://arxiv.org/abs/1911.11924>
137. Shaikewitz 等, Category-level object shape and pose estimation in less than a millisecond, ICRA 2026. <https://arxiv.org/abs/2509.18979>
138. Papalia, Fishberg, O'Neill, How, Rosen, Leonard, Certifiably correct range-aided SLAM, T-RO 2024. <https://arxiv.org/abs/2302.11614>
139. Thoms 等, Distributed certifiably correct range-aided SLAM, ICRA 2025. <https://arxiv.org/abs/2503.03192>
140. Holmes, Barfoot, An efficient global optimality certificate for landmark-based SLAM, RA-L 2023. <https://arxiv.org/abs/2206.12961>
141. Dümbgen, Holmes, Agro, Barfoot, Toward globally optimal state estimation using automatically tightened semidefinite relaxations, T-RO 2024. <https://arxiv.org/abs/2308.05783>
142. Holmes, Dümbgen, Barfoot, On semidefinite relaxations for matrix-weighted state-estimation problems in robotics, T-RO 2024. <https://arxiv.org/abs/2308.07275>
143. Barfoot, Holmes, Dümbgen, Certifiably optimal rotation and pose estimation based on the Cayley map, IJRR 2025. <https://arxiv.org/abs/2308.12418>
144. Korotkine 等, Globally optimal data-association-free landmark-based localization using semidefinite relaxations, RA-L 2025. <https://arxiv.org/abs/2504.08547>
145. Han 等, Building Rome with convex optimization, RSS 2025. <https://arxiv.org/abs/2502.04640>
146. Dümbgen, Holmes, Barfoot, Exploiting chordal sparsity for fast global optimality with application to localization, WAFR 2024. <https://arxiv.org/abs/2406.02365>
147. Subramanian 等, Exploiting chordal sparsity for globally optimal estimation with factor graphs, ICRA 2026 workshop. <https://arxiv.org/abs/2605.30617>
148. Xu 等, Certifiable factor graph optimization, 2026. <https://arxiv.org/abs/2603.01267>
149. Holmes 等, Following a unique path: a fast certifier applied to outlier-robust pose registration, 2026. <https://arxiv.org/abs/2609.03222>
150. Holmes, Dümbgen, Barfoot, SDPRLayers: certifiable backpropagation through polynomial optimization problems in robotics, 2024. <https://arxiv.org/abs/2405.19309>
151. Papalia 等, An overview of the Burer–Monteiro method for certifiable robot perception, RSS 2024. <https://arxiv.org/abs/2410.00117>
152. Zhao 等, Advances in global solvers for 3D vision, 2026. <https://arxiv.org/abs/2602.14662>
153. Majumdar, Tedrake, Funnel libraries for real-time robust feedback motion planning, IJRR 2017. <https://arxiv.org/abs/1601.04037>
154. Majumdar, Vasudevan, Tobenkin, Tedrake, Convex optimization of nonlinear feedback controllers via occupation measures, IJRR 2014. <https://arxiv.org/abs/1305.7484>
155. Zhao, Mohan, Vasudevan, Optimal control of polynomial hybrid systems via convex relaxations, T-AC 2020. <https://arxiv.org/abs/1702.04310>
156. Dai, Permenter, Convex synthesis and verification of control-Lyapunov and barrier functions with input constraints, ACC 2023. <https://arxiv.org/abs/2210.00629>
157. Ahmadi, Majumdar, DSOS and SDSOS optimization, SIAM J. Appl. Algebra Geom. 2019. <https://arxiv.org/abs/1706.02586>
158. Dai, Amice, Werner, Zhang, Tedrake, Certified polyhedral decompositions of collision-free configuration space, IJRR 2024. <https://arxiv.org/abs/2302.12219>
159. Amice 等, Certifying bimanual RRT motion plans in a second, ICRA 2024. <https://arxiv.org/abs/2310.16603>
160. Li 等, Collision-free trajectory optimization in cluttered environments using sums-of-squares programming, RA-L 2024. <https://arxiv.org/abs/2404.05242>
161. Graesdal 等, Semidefinite relaxations for collision-free motion planning, 2026. <https://arxiv.org/abs/2606.14063>
162. Morozov 等, Multi-query shortest-path problem in graphs of convex sets, WAFR 2024. <https://arxiv.org/abs/2409.19543>
163. Kang, Xu, Sarva, Liang, Yang, Fast and certifiable trajectory optimization, WAFR 2024. <https://arxiv.org/abs/2406.05846>
164. Kang 等, Global contact-rich planning with sparsity-rich semidefinite relaxations, RSS 2025. <https://arxiv.org/abs/2502.02829>
165. Li 等, A curvature-aware rank-adaptive distributed augmented-Lagrangian solver for large-scale SDPs, 2026. <https://arxiv.org/abs/2607.17933>
166. Mahajan 等, TinySDP: real time semidefinite optimization for certifiable and agile edge robotics, RSS 2026. <https://arxiv.org/abs/2605.13748>
167. Rapakoulias, Tsiotras, Discrete-time optimal covariance steering via semidefinite programming, CDC 2023. <https://arxiv.org/abs/2302.14296>
168. Groudiev 等, Sampling-based global optimal control and estimation via semidefinite programming, 2025. <https://arxiv.org/abs/2507.17572>
169. Hu, Fazlyab, Morari, Pappas, Reach-SDP, CDC 2020. <https://arxiv.org/abs/2004.07876>
170. Fazlyab, Morari, Pappas, Safety verification and robustness analysis of neural networks via quadratic constraints and semidefinite programming, T-AC 2022. <https://arxiv.org/abs/1903.01287>
171. Fazlyab 等, Efficient and accurate estimation of Lipschitz constants for deep neural networks, NeurIPS 2019. <https://arxiv.org/abs/1906.04893>
172. Giamou 等, Convex iteration for distance-geometric inverse kinematics, RA-L 2022. <https://arxiv.org/abs/2109.03374>
173. Wu 等, IKSPARK: obstacle-aware inverse kinematics via convex optimization, 2024. <https://arxiv.org/abs/2403.12235>
174. Tian, Khosoussi, Rosen, How, Distributed certifiably correct pose-graph optimization, T-RO 2021. <https://arxiv.org/abs/1911.03721>
175. Wu 等, Distributed optimization in sensor network for scalable multi-robot relative state estimation, 2023. <https://arxiv.org/abs/2303.01242>
176. Wang 等, Certifiable mutual localization and trajectory planning for bearing-based robot swarm, 2024. <https://arxiv.org/abs/2401.07784>
177. Zheng, Fantuzzi, Papachristodoulou, Goulart, Wynn, Chordal decomposition in operator-splitting methods for sparse semidefinite programs, Math. Prog. 2020. <https://arxiv.org/abs/1707.05058>

### 6.6 量子计算与机器人视觉

178. Birdal, Golyanik, Theobalt, Guibas, Quantum permutation synchronization, CVPR 2021. <https://arxiv.org/abs/2101.07755>
179. Seelbach Benkner 等, Q-Match: iterative shape matching via quantum annealing, ICCV 2021. <https://arxiv.org/abs/2105.02878>
180. Arrigoni 等, Quantum motion segmentation, ECCV 2022. <https://arxiv.org/abs/2203.13185>
181. Seelbach Benkner 等, QuAnt: quantum annealing with learnt couplings, ICLR 2023. <https://arxiv.org/abs/2210.08114>
182. Bhatia 等, CCuantuMM: cycle-consistent quantum-hybrid matching of multiple shapes, CVPR 2023. <https://arxiv.org/abs/2303.16202>
183. Farina 等, Quantum multi-model fitting, CVPR 2023. <https://arxiv.org/abs/2303.15444>
184. Doan, Sasdelli, Suter, Chin, A hybrid quantum-classical algorithm for robust fitting, CVPR 2022. <https://arxiv.org/abs/2201.10110>
185. Yurtsever, Birdal, Golyanik, Q-FW: a hybrid classical-quantum Frank–Wolfe for quadratic binary optimization, ECCV 2022. <https://arxiv.org/abs/2203.12633>
186. Wang 等, Quantum multiple rotation averaging, 3DV 2026. <https://arxiv.org/abs/2602.10115>
187. Bernecker 等, Quantum Levenberg–Marquardt algorithm for optimization in bundle adjustment, 2022. <https://arxiv.org/abs/2203.02311>
188. Dehaghani 等, Quantum solution framework for finite-horizon LQG control via block encodings and QSVT, 2025. <https://arxiv.org/abs/2507.09841>
189. Clayton 等, Differentiable quantum computing for large-scale linear control, 2024. <https://arxiv.org/abs/2411.01391>
190. Kuete Meli 等, Quantum-enhanced computer vision: going beyond classical algorithms, 2025. <https://arxiv.org/abs/2510.07317>

[^m15]: Montanaro 的结果对任意方差有界的随机子程序成立，因此也适用于把量子子程序当作黑箱的情形。
