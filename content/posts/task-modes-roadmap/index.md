---
title: "机器人决策中的结构化多模态：七个研究问题的调研与分析"
date: 2026-09-30
description: "围绕博士论文主题《Structured Multimodality in Robot Decision Making: Discovering, Learning, and Exploiting Task Modes》，对七个研究问题逐一给出问题陈述、相关研究、存在的问题与可能的思路，并说明它们如何以“模态结构作为先验”组织成一条主线。"
summary: "以“任务模态的发现、学习与利用”为骨架，逐一分析七个研究问题：离线模态枚举、无数据学习 score、一步推理、从传感器到仿真器、从示教提取模态与约束、RL 与枚举的联系、定义模态的代价设计。"
tags: ["研究规划", "模态枚举", "扩散", "MPPI", "强化学习", "real2sim", "示教学习"]
---

本章是一份研究路线图。主题是博士论文的拟定题目

> **Structured Multimodality in Robot Decision Making: Discovering, Learning, and Exploiting Task Modes**
> 《机器人决策中的结构化多模态：任务模态的发现、学习与利用》

先说明用词：这里的**多模态（multimodality）指目标分布有多个峰**，即同一任务存在多种彼此不同的合理做法（从障碍物左侧还是右侧绕过、用手撑还是用腿摆起身），与"多传感器模态"（视觉、语言、触觉）无关。**任务模态（task mode）**指这些做法在轨迹空间中对应的峰及其吸引盆地。

七个问题按副标题的三个动词组织：**发现**（第 3 节：问题一、五、七）、**学习**（第 4 节：问题二、六）、**利用**（第 5 节：问题三、四）。每个问题固定五段：问题陈述、相关研究、存在的问题、可能的思路、与模态枚举的接口及可行性。第 2 节给出共同的设定与记号，第 6 节给出问题之间的依赖与优先级。记号与[《MPPI 迭代、Mean Shift 与 EM》](/posts/mppi-meanshift-em/)、[《SMC 与 CEM》](/posts/smc-cem/)一致。

## 1. 总览

> [!conclusion] 一条主线
> 任务（代价 + 仿真器）诱导出轨迹空间上的未归一化分布 $\pi\propto e^{-C/\lambda}$。这个分布的**模态结构**——有几个峰、各在哪、各占多少质量、彼此之间的势垒多高——是一个可以被显式计算出来的对象。七个问题的共同思想是：**先把这个结构枚举出来，再把它作为后续采样、学习、推理与部署的先验。** 这样每一步的贡献都落在"结构先验"上，而不与神经采样器、一致性模型、real2sim 等通用工具正面竞争。

| 问题 | 阶段 | 输入 | 输出 | 一句话 |
|---|---|---|---|---|
| 一 | 发现 | 代价 + 仿真器 | 持久模态集（位置、盆地、质量、势垒） | 离线枚举全部有意义的做法 |
| 五 | 发现 | 示教数据 | 示教的模态与低熵片段（任务约束） | 有数据版本的枚举 |
| 七 | 发现 | 设计者意图 | 使模态树与意图一致的代价 | 什么代价让模态有意义 |
| 二 | 学习 | $\pi$ 的评估 + 枚举结果 | 可条件化的 score/采样器 | 无数据摊销，用枚举保证覆盖 |
| 六 | 学习 | 代价 + 仿真器 | 与 RL 的接口与诊断 | 两种摊销的联系 |
| 三 | 利用 | 枚举结果或学到的 score | $O(1)$ 串行深度的在线推理 | 离散选峰 + 峰内一步 |
| 四 | 利用 | 传感器数据 | 可 rollout 的仿真器与其不确定性 | 模态对模型误差的鲁棒性 |

## 2. 共同设定与记号

- **任务**：运行代价 $\ell(x,u)$、终端代价 $\phi(x_N)$、只能 rollout 的仿真器 $x_{t+1}=f(x_t,u_t)$（无梯度，可能有域随机化）。决策变量是盒约束的控制节点序列 $V\in\mathcal B\subset\mathbb R^D$，经插值展开为逐步控制。$C(V)$ 是一条完整 rollout 的总代价。
- **目标分布**：$\pi_\lambda(V)\propto e^{-C(V)/\lambda}$，$\lambda$ 为温度。它的局部极大值就是 $C$ 的局部极小值；每个极大值 $\mu_m$ 有吸引盆地 $B_m$、质量 $w_m(\lambda)=\int_{B_m}\pi_\lambda$、与相邻盆地之间的势垒高度（持久性）$\Delta_m$。
- **尺度族**：$p_h=\pi_\lambda*\mathcal N(0,h^2I)$。训练自由扩散（MBD、DIAL-MPC、本项目的 S2R）用 rollout 做蒙特卡洛估计 Tweedie 后验均值 $m_h(z)=\mathbb E[V\mid z]=z+h^2\nabla\log p_h(z)$，据此去噪。
- **两个旋钮**：本章统一称为**温度 $\lambda$ 与尺度 $h$**。$\lambda$ 决定各模态的质量与模态内宽度，但不改变 $h=0$ 时模态的位置；$h$ 使相邻模态合并。二者合起来是"温度—尺度双参数退火"。
- **已有结论**（前两章）：固定 $(\lambda,h)$ 下 MPPI 迭代、mean shift、EM 是同一个不动点迭代，收敛到 $p_h$ 的众数；局部线性收敛率 $r=\lambda_{\max}(\Sigma^{-1}\mathrm{Cov}_{r^*}[V])$；$r$ 穿过 1 即模态分叉，与 Rose 确定性退火的临界温度、扩散模型的对称破缺是同一条件。扩散是把粒子群沿 $\{p_h\}$ 搬运，分叉处按模态质量分流。TSR 的 $r(h)$ 缩放在分叉后才生效、$k\to\infty$ 极限是 mean shift。

## 3. 发现：问题一、五、七

### 3.1 问题一：离线模态枚举作为规划方法

**问题陈述。** 给定代价、仿真器、目标温度 $\lambda$，输出 $\pi_\lambda$ 的持久模态目录：每个模态的位置 $\mu_m$、盆地 $B_m$、质量 $w_m$、持久性 $\Delta_m$，以及它们在 $h$ 方向的合并树。用途是离线规划：得到可解释的备选方案集合，并对局部最优免疫。难点：(i) score 只能由蒙特卡洛估计，小 $h$、高维下权重退化，不动点和用于分叉检测的雅可比 $\Sigma^{-1}\mathrm{Cov}$ 都有噪声；(ii) 高斯尺度空间只在一维保证"随 $h$ 增大模态只合并不新生"，$D\ge2$ 时平滑可以制造模态（Carreira-Perpiñán & Williams 2003），自顶向下的延拓可能漏掉分支；(iii) 质量不等的两个模态不是对称 pitchfork 分叉，小模态以鞍结（fold）方式在别处出现，沿临界特征向量的分支切换找不到它；(iv) 接触类代价分段光滑，节点空间里有指数多的浅局部极小（时间平移、冗余节点、对称），"全部模态"必须限定为"持久性超过阈值、并按行为等价合并后"；(v) 质量需要每个盆地的归一化常数；(vi) 盒约束产生边界模态；(vii) 除二维解析函数外没有带真值模态集的基准。

**相关研究。**
- 训练自由扩散：MBD（Pan et al. 2024）用 rollout 估计平滑目标的 score 并反向去噪，单链或独立多链穿过 $\{p_h\}$，不跟踪不动点、不检测分叉、不报告分支与质量；DIAL-MPC（Xue et al. 2025）是其在线腿式版本；Reverse Diffusion Monte Carlo（Huang et al. 2024）给出这一机制的采样理论。
- 扩散模型的相变：Raya & Ambrogioni (2023) 证明反向过程在临界噪声处发生对称破缺（中心不动点失稳），并对对称高斯混合给出临界值——这是本章分叉判据的学习版；Biroli et al. (2024) 给出"speciation"（提交到某一类）与"collapse"两个转变的标度律；Li & Chen (2024) 的 critical windows 证明对分得开的 log-concave 混合，反向过程在一个窄窗口内提交到子混合，提交概率等于其权重——这是"分叉处按质量分流"最严格的表述，但只对精确 score 的随机输运成立。
- 尺度空间与模态树：Silverman (1981) 的临界带宽、Minnotte & Scott (1993) 的 mode tree、SiZer（Chaudhuri & Marron 1999）的显著性检验、Carreira-Perpiñán (2007) 的 mean shift = EM。
- 持久性：ToMATo（Chazal et al. 2013）用 0 维持久性合并模态并有稳定性定理；Wales 的能量地形（basin hopping、disconnectivity graph，Wales & Doye 1997）是化学界的模态枚举，依赖梯度与鞍点搜索。
- 多解优化：niching 综述（Li et al. 2017，CEC 基准、NEA2）；Quality-Diversity（CVT-MAP-Elites，Vassiliades et al. 2018）以人工行为描述子为格子；机器人侧的 rollout 聚类 MPPI（Patrick & Bakolas 2024）、M3P2I、SV-MPC（Lambert et al. 2020）、Osa (2020) 的多模态轨迹优化（混合表示、固定模态数）。

**存在的问题。** 没有一个算法同时做到：沿 $h$ 延拓 Tweedie 不动点、用噪声感知的雅可比同时检测 pitchfork 与 fold 型分支事件、按显式持久性剪枝、在目标温度下报告质量，并给出完备性的条件（log-concave 分离分量、无模态新生、ESS 下界）。具体空白：$D\ge2$ 的尺度空间因果性条件；fold 型分支的检测；蒙特卡洛噪声下的分叉显著性检验（SiZer 的对应物）；三种持久性（尺度持久性、势垒持久性、质量持久性）哪一种对应"有意义的备选"；行为等价的商空间；只用 rollout 的质量估计；组合爆炸下的 anytime 枚举与截断误差；截断核下的 Tweedie 恒等式；随机仿真器下 $\exp(-\mathbb E C/\lambda)$ 与 $\mathbb E\exp(-C/\lambda)$ 的模态差异；评估基准。

**可能的思路。**
1. 把 Tweedie 延拓写成伪弧长延拓问题 $F(x,h)=x-\mathbb E_h[V\mid x]=0$：雅可比 $J=I-\Sigma^{-1}\mathrm{Cov}_{\text{post}}$ 由同一批加权 rollout 免费得到，按 $J$ 的最小奇异值控制 $h$ 的步长，由 $\det J$ 变号或 $\lambda_{\max}$ 穿过 1 检测分支点（数值分岔的标准做法，Keller；Allgower & Georg）。
2. 带统计保护的分叉检验：ESS $\ll D$ 时用收缩估计（Ledoit–Wolf）估协方差，对加权 rollout 做 bootstrap 得 $\lambda_{\max}$ 的置信区间，只有下界超过 1 才宣布分叉；未被认证稳定时不降 $h$，把预算花在当前 $h$ 上加样本。
3. 混合分支切换：认证的 pitchfork 处沿 $\pm v_{\text{crit}}$ 生成子分支；为抓 fold 型分支，在 $h-\mathrm dh$ 处额外跑一小步随机输运（DDPM 噪声而非 mean shift）并对粒子做 mean shift 聚类，不落在已有盆地的簇开新分支。
4. 持久性剪枝的无梯度实现：相邻模态之间沿直线或 nudged elastic band 用 rollout 估势垒，prominence 低于 $\tau\lambda$ 的并入父支；同时报告尺度持久性。
5. 质量估计三路：Laplace（延拓已跟踪的协方差，接触任务上标记为不可靠）、沿分支自身 $h$ 路径的退火重要性采样/热力学积分、随机输运的粒子比例（Li–Chen 条件下无偏）。
6. 双温度枚举：在大 $\lambda_{\text{run}}$ 下枚举（后验更宽、ESS 更高、浅伪模态更少），再在目标 $\lambda$ 下按 $\exp(-C(1/\lambda-1/\lambda_{\text{run}}))$ 重加权与重剪枝；TSR 可视为它的开环特例。
7. 行为商空间：从已有 rollout 计算描述子（接触序列、末端路径、关键关节符号），合并描述子相同的节点空间模态；保留 CVT-MAP-Elites 式档案作为召回率基线。
8. 评估：NEA2、basin hopping（以 MPPI 为局部优化器）、SV-MPC、rollout 聚类 MPPI、Osa 式混合优化，先在 Griewank/Rastrigin/GMM（已知局部极小）和 pendulum（两个摆起方向）上比，再上接触任务。

**接口与可行性。** 这是主线本身。hydrax 已有 mean shift 校正器（MCSA）、群体输运（DDPM/TwoStage）、加权协方差工具、已知极小的数值任务与 pendulum/humanoid 流程；离线设定允许每个 $h$ 步数千次 GPU rollout。最小的第一篇：延拓 + 认证的 pitchfork 检测 + Laplace 质量，在 Griewank/GMM/pendulum 上以 NEA2、CVT-MAP-Elites 为召回率基线。风险：$D\ge2$ 的非因果性使完备性只能是条件性的；fold 型分支恰是机器人里常见的"少用的策略"；小 $h$ 处 ESS 崩溃使树只能分辨到某个 $h_{\min}$；接触非光滑让持久性阈值与行为商空间承担大部分实际工作；评估基准本身是研究的一部分。

### 3.2 问题五：从示教中提取模态与任务约束

**问题陈述。** 输入 $N$（10–300）条人类示教 $\tau_i=(o_t,s_t,a_t)$，长度不齐、可能来自不同策略、只含成功样本，另有只能 rollout 的仿真器与可能只部分已知的代价。输出：(a) 示教的持久模态（策略簇）及占比；(b) 每个模态内的结构化约束——关键帧/子目标、物体相对的关键点关系、以及**低熵片段**（示教分布在某些方向上方差急剧收缩的时间窗，对应插孔等精度关键时刻）；(c) 把它们作为采样式 MPC / 训练自由扩散的分段终端代价、硬约束或引导，让示教学习策略只做提议、规划器负责精度与可行性。动机来自示教学习策略的三个弱点：所有时刻同等对待、没有最优性与避障保证、以及"捷径学习"——策略偏向低维本体状态而压制高维图像（causal confusion，de Haan et al. 2019；Lu et al. 2026 定量地把失败定位在需要视觉定位的运动过渡阶段）。难点：示教稀少而轨迹维度极高，必须先做相位对齐与任务相关的低维嵌入，否则时序方差伪装成空间方差；"低方差 = 重要"是 ProMP/TP-GMM 的经典启发式，但低方差也可能只是操作者习惯，示教只给可行集内部的样本（Chou et al. 2018 指出需要反例）；约束在哪个坐标系下塌缩是待搜索对象；接触约束在纯运动学示教中不可观测；约束以软代价进入 $C$ 会改变模态结构，以硬约束进入则要在盒约束节点空间上投影，都与 $(\lambda,h)$ 的分时调度耦合。

**相关研究。** 关键帧与航点：AWE（Shi et al. 2023）按线性可重建性对单条示教提取最少航点；Keyframe-Focused IL（Wen et al. 2021）以动作变点为关键帧加权训练；Q-attention/ARM（James & Davison 2022）用夹爪状态变化与近零速度定义关键帧。约束表示：ReKep（Huang et al. 2024）用 VLM 生成关键点关系代价、分阶段优化；kPAM（Manuelli et al. 2019）是其几何前身；Ureche et al. (2015) 从示教提取连续任务约束（哪些变量在何时重要）；Niekum et al. (2015) 把无结构示教切成技能并自动推断坐标系。约束可辨识性：Chou et al. (2018) 的逆约束学习用 hit-and-run 造反例。多模态示教：BeT（Shafiullah et al. 2022）、IBC（Florence et al. 2021）、Diffusion Policy（Chi et al. 2023）；MimicGen（Mandlekar et al. 2023）说明"示教 = 物体相对坐标下的分段片段"这一结构假设有效。持久性聚类先例：Merkt et al. (2021) 用持久同调对最优控制解集聚类以避免平均掉不连续的多解。约束进入扩散规划：SafeDiffuser（Xiao et al. 2025）、MPD（Carvalho et al. 2023）。

**存在的问题。** 轨迹空间的模态树在 $N\approx50$ 下没有意义，需要相位对齐与嵌入，而嵌入选择改变模态结构；习惯与约束的可辨识性需要代价意义上的判据（违反后 $C$ 上升多少）；条件分布 $p(a_{t:t+k}\mid o_t)$ 的模态要同时沿 $h$ 与沿相位 $t$ 延拓，二者一致性无理论；示教模态的占比反映操作者偏好而非 $\pi_\lambda$ 的质量，两者如何融合缺少原则（乘积分布的模态不是并集）；接触约束不可观测；分时的 $\lambda_t/h_t$ 调度下 mean shift 的收敛率与分叉判据需重推；从示教提取的约束不给最优性，保证只能来自规划器的 rollout 验证；若约束提取只用本体状态，规划层会重演 modality collapse。

**可能的思路。**
1. 示教即样本的模态树：相位对齐后在物体相对的关键点/末端位姿空间嵌入，跑带分叉检测与持久性剪枝的 mean shift 延拓，得到策略簇及占比。
2. 分时持久性/低熵谱：对每个相位计算模态内示教协方差 $\Sigma_t$，以最小特征值方向定义精度关键片段；它同时给出关键帧处的 Mahalanobis 航点代价，以及 S2R 的各向异性分时 $h_t/\Sigma_t$ 调度（示教协方差白化后的 mean shift，在低方差方向上把收敛率压小、自由方向放松）。
3. 仿真干预式约束验证：对每个候选低方差方向做受控扰动 rollout，以代价上升量作为该约束的势垒；只保留代价上持久的约束，用代价持久性替代密度持久性区分习惯与约束（把 Chou 的反例与 de Haan 的干预搬进仿真器）。
4. 分叉触发的规划：Diffusion Policy 的 $\epsilon$ 网络就是学到的 $\nabla\log p_h(a\mid o)$，可以不用 rollout 直接对它做 Tweedie 延拓，得到当前观测下动作分布的模态树；只在出现临界特征值或低熵片段时调用带 rollout 的 MPC 精修——"关键时刻"由策略分布的相变定义，而非人工标注。
5. 已知组合：AWE/ARM 关键帧 + 跨示教对齐 + 分阶段终端代价；从示教拟合 ReKep 式关系约束（按跨示教方差排序、持久性截断）；软约束入 $C$ / 硬约束投影 / 策略作为提议分布 $q$、规划器按 $e^{-C/\lambda}$ 重加权，对乘积分布做枚举。

**接口与可行性。** 示教是操作者隐式目标 $\pi_{\text{demo}}$ 的样本，与规划目标 $\pi$ 共享"模态 = 策略、质量 = 占比、$h$ 合并策略、$\lambda$ 控制宽度"的几何；模态枚举的三件工具（延拓与分叉检测、持久性剪枝、收敛率特征值）在这里分别对应策略簇树、习惯/约束辨识、关键时刻检测。可行性中高：robomimic 的多人示教（Square 含插入精度段）或 Diffusion Policy 的 PushT 数据可直接用；风险是示教太少时模态树不稳定（须报告对 $h$ 的敏感性与持久性图）、相位对齐失败产生伪低熵段、以及被视为 ProMP 启发式的重新包装——贡献必须落在代价持久性辨识、与训练自由扩散 MPC 的分时 $(h,\lambda)$ 耦合、分叉触发规划三点上。

### 3.3 问题七：定义模态的代价设计

**问题陈述。** 模态由代价定义。问题是逆向的：怎样的 $C$ 使枚举出的模态与设计者心中的备选（左/右绕行、哪种接触序列、哪个目标）一致，而不是参数化、仿真器或惩罚形状的产物。难点：(1) "有意义的备选"是行为层面的陈述（同伦类、接触序列、离散决策），而模态在节点空间，两者之间隔着非线性、接触不连续的仿真器；(2) 地形只能通过带噪 rollout 访问，持久性要统计估计，接触抖动、盒边界饱和、样条节点冗余产生的低持久性极小与真实备选难以区分；(3) 对最优点无害的代价变换对临界点结构未必无害：势函数整形在开环轨迹空间上退化为终端项，单调变换 $g(C)$ 保持模态但重加权质量，惩罚权重改变势垒不移动模态，标量化权重让全局最优在盆地间跳变；(4) $\lambda$ 与 $h$ 作用不同，持久性诊断必须说明在哪个 $(\lambda,h)$ 下计算；(5) "选代价参数使持久性树等于意图树"是欠定且分段光滑的逆问题，且除二维导航（同伦类给出真值）外没有基准。

**相关研究。** 奖励整形理论（Ng et al. 1999）：势函数整形是保持最优策略的唯一变换——但在开环有限时域轨迹空间上它恰是终端代价的改变；MaxEnt IRL（Ziebart et al. 2008）拟合的是特征期望而非模态集，其梯度机制可复用于质量对参数的导数；LLM 奖励设计（Eureka，Ma et al. 2024；Text2Reward）只优化训练后的任务成功率，没有地形视角；奖励误设计检查表（Knox et al. 2023）的"一次碰撞值多少时间"正是撞与不撞两个盆地之间的势垒；奖励黑客的形式化（Skalse et al. 2022）说明保持策略排序的变换几乎只有仿射，暗示"保持模态树"的变换类更窄。采样式 MPC 的代价实践：信息论 MPPI 的指示型碰撞惩罚（Williams et al. 2018），STORM 的多项加权代价（Bhardwaj et al. 2021）。拓扑定义的备选：同伦类约束搜索（Bhattacharya et al. 2010）、T-MPC++ 按同伦类并行优化（de Groot et al. 2024）。持久性与能量地形：ToMATo 的稳定性定理；disconnectivity graph（Becker & Karplus 1997）显式给出温度依赖的盆地与温度无关的鞍点树；Das & Dennis (1997) 说明加权和只能到达 Pareto 前沿的凸部分，权重变化时全局最优在不连通的"瓣"之间跳变——轨迹空间里这些瓣就是不同盆地；Kappen (2005) 是路径积分控制里"噪声/温度变化引起对称破缺"的最早表述。

**存在的问题。** 二维同伦类之外没有"有意义备选"的可操作定义；没有哪类代价编辑保持模态集、持久性排序或质量的刻画；指示型障碍惩罚产生平台而非有限势垒，尺度空间分叉理论在平台边界上不直接适用；伪模态（饱和、冗余节点、接触抖动、域随机化）缺少噪声模型来判断持久性是否超过蒙特卡洛噪声；逆问题的势垒项在最低鞍点切换时不连续；诊断该在哪个 $(\lambda,h)$ 下报告无约定；开环模态与滚动时域闭环行为的关系未定；无基准。

**可能的思路。**
1. 持久性树作为代价设计诊断：在目标 $\lambda$ 下枚举，用节点空间的一维路径（线性插值或几步 nudged elastic band，每点一次 rollout）估相邻模态间势垒，输出以代价为单位（$\lambda$ 不变）的 disconnectivity graph 与 $\lambda$ 下的质量；对照设计者列出的意图备选，"缺失的意图备选"与"未预期的高持久性模态"是两个失败标志。
2. 刻画开环轨迹优化中保持模态的变换类：$g(C)$ 单调保持模态集与持久性排序、只重加权质量；势函数整形退化为 $\Phi(x_T)-\Phi(x_0)$；单个惩罚权重 $w$ 单调移动势垒而不移动可行域内的模态；标量化权重通过鞍结事件创造/消灭模态。把后两者写成以 $w$ 为延拓变量的延拓问题，复用 Tweedie 延拓与特征值穿越检测。
3. 硬/软障碍惩罚：检验猜想——指示型惩罚下二维导航的持久模态恰是可行同伦类；软惩罚 $w\,d$ 下当势垒低于阈值或 $h$ 超过障碍在节点空间的厚度时类合并。由此得到校准规则：选 $w$ 使每个意图类的势垒同时超过蒙特卡洛噪声与 $\lambda\log(\text{质量比})$。Dubins/点质量基准可立即测试。
4. 终端/运行归因：沿连接路径分解势垒来自 $C_{\text{run}}$ 还是 $C_{\text{term}}$；假设运行代价产生"走哪条路"型模态、终端代价产生"到哪个终态"型模态。
5. 行为描述子商空间上的持久性（QD 的思想新用）。
6. 逆持久性校准：$C_\theta=\sum_i\theta_iC_i$，在持久性树上定义损失，用 CMA-ES/贝叶斯优化在重复枚举上优化，质量项用 MaxEnt-IRL 式梯度 $\partial\log w_m/\partial\theta_j=-(\mathbb E_{B_m}[C_j]-\mathbb E_\pi[C_j])/\lambda$。
7. 把持久性树摘要加入 Eureka 式的 LLM 反思提示。
8. 噪声感知持久性：对模态与鞍点的代价做 bootstrap，剪掉不显著的模态，输出 $(\lambda,h)$ 上的显著性图（SiZer 的对应物）——这同时从数据里给出枚举的剪枝阈值。

**接口与可行性。** 问题七是枚举输出的设计侧用途：同一套延拓与分岔检测机器，把延拓变量从 $h$ 换成代价参数，就得到意图备选诞生或消亡的临界权重。持久性阈值与"意图备选"是同一个数的两面，可以闭环：枚举 → 对照意图 → 调代价 → 再枚举。可行：数值任务给持久性估计的真值，Dubins/点质量给同伦类真值，pendulum/humanoid 给接触情形；势垒估计的 rollout 数 $O(\#\text{模态}^2\times\text{路径点})$ 远小于枚举本身。风险：伪模态主导树（须先做噪声感知剪枝）；线性插值势垒是上界；逆校准可能无解（行为差异只在时域后段）；势函数整形"无效"的结论只在开环有限时域、$\Phi(x_T)=0$ 下成立。

## 4. 学习：问题二、六

### 4.1 问题二：无数据地学习未归一化分布的 score

**问题陈述。** 只有 $C$ 的黑箱评估（每次评估一条 rollout，可批量并行），没有梯度、没有来自 $\pi$ 的样本——与拟合示教分布的 Diffusion Policy 根本不同。输出一个可条件化（任务、初始状态 $x_0$、温度）的 score 网络 $s_\theta(z,h,x_0)$ 或采样器，使去噪能覆盖所有持久模态并按质量分配，且在线调用比每步做 MC 估计便宜。难点：现有神经采样器几乎都依赖 $\nabla E$ 或 Langevin 预条件，而这里只有梯度自由的 Tweedie 估计 $(\sum_iw_iV_i-z)/h^2$，大 $h$ 小 $\lambda$ 时 ESS 崩溃；训练点 $(z,h)$ 放在哪是探索问题，自举外循环只在模型已有质量的区域精化，自我强化 mode dropping；无真值样本，覆盖与质量无法直接评估；每个训练点要 $N$ 条 rollout，摊销收益要算清；接触代价不连续，小 $h$ 下有大量伪模态；条件化到 $x_0$ 后模态结构随 $x_0$ 分岔，网络在分岔面附近必然不光滑。

**相关研究。** 神经采样器：PIS（Zhang & Chen 2022）把采样写成随机最优控制（与 MPPI 同源的 Girsanov 变分）；DDS（Vargas et al. 2023）把 VP 扩散反向 drift 学成网络——与 S2R 完全同构，区别只在 score 从哪来；Berner et al. (2024) 与 Richter & Berner (2024) 的统一 SOC 框架与 log-variance 损失；iDEM（Akhound-Sadegh et al. 2024）用 $K$ 个噪声样本的自归一化 IS 估计噪声级 score 并回归，外循环自举——但其估计式含 $\nabla E$；BNEM（OuYang et al. 2026）回归噪声化能量的 log-mean-exp 估计再自动微分，只需能量值，且由小 $h$ 向大 $h$ 自举引导（与尺度空间自底向上构造方向一致）；Adjoint Sampling（Havens et al. 2025）需终端 $\nabla E$；Boltzmann generators（Noé et al. 2019）最早指出 training-by-energy 会 mode collapse；FAB（Midgley et al. 2023）用 mass-covering 的 $\alpha=2$ 散度 + 退火链发现新模态；off-policy 训练与基准（Sendera et al. 2024）；大规模评估与 mode-collapse 指标（Blessing et al. 2024，其覆盖指标需要已知模态）。**最重要的警告**：He et al. (2025) 系统检验发现几乎所有成功的神经采样器都依赖 Langevin 预条件，去掉后连简单目标都覆盖不了，并提出并行回火 + 生成模型作为强基线。控制侧：学习 MPC 采样分布（Sacks & Boots 2022，学到的分布只作提议、在线仍 rollout 重加权）；D-MPC（Zhou et al. 2024，需离线数据）；MBD/DIAL 为训练自由基线，也是 Q2 的"教师"。

**存在的问题。** 梯度自由 score 学习的方差随 $(h,\lambda,D,N)$ 的标度未刻画；无 Langevin 预条件时能否避免 mode collapse；训练点的独立于模型的覆盖机制；无真值下的验收标准；条件化到 $x_0$ 的分岔面表示；仿真预算与摊销收益的临界点；截断/clip 对去噪器的影响；温度 $\lambda$ 是否作为输入；"蒸馏 MC 估计 + 从模型重采样"循环的不动点是 $\pi$ 还是丢了模态的 $\tilde\pi$。

**可能的思路。**
1. 梯度自由的去噪/能量匹配：对 $(z,h)$ 回归 $x_0$ 预测 $D_\theta\leftarrow\sum_iw_iV_i$（截断正态 IS，输出天然在盒内、数值比 score 稳定），或按 BNEM 回归噪声化能量再微分；用 BNEM 型自举缓解高噪声方差。这是 iDEM/BNEM 在 rollout-only 下的直接移植，尚无人在控制任务上做。
2. **模态分层的训练数据**：用枚举得到的 $\{\mu_m(h),\Sigma_m(h),w_m,B_m\}$ 构造训练点分布 $z\sim\sum_mw_m\mathcal N(\mu_m(h),\Sigma_m(h)+h^2I)$，保证每个持久模态出现在训练集中，绕开自举外循环的 mode dropping；同时用混合/防御式 IS 提议提高大 $h$ 下的 ESS。
3. 模态树门控的混合专家去噪器：$s_\theta=\sum_jg_js_j$，门控索引为尺度 $h$ 上存活的树节点，超过合并尺度后子模态的专家自然合并；门控先验 = 目标温度下的质量；改温度只需重加权门控 + TSR 式收缩组内方差。
4. 从在线 MBD/S2R 免费蒸馏：每个去噪步产生 $(z,h,x_0,\hat D_{\text{MC}})$，离线回归；在线用作 Sacks–Boots 式提议或 S2R 的前几步替代，只在小 $h$ 用 rollout 精化。
5. 学到的去噪器作为在线 MC 估计的控制变量：$\hat s=s_\theta+[\hat D^{(N')}_{\text{MC}}-D_\theta]/h^2$，少量 rollout 估残差（与仓库 smc_dpm 的控制变量思路同构）。
6. 评估协议：以枚举模态为参考报告每个持久模态的命中率与质量误差、路径 IS 权重的 $\log Z$ 与 ESS、闭环 MPC 代价；并行回火/退火 MCMC（mean shift 核）+ 生成模型必须作为基线。
7. 反哺枚举：用 $s_\theta$ 做便宜的不动点延拓与分叉检测（$\partial D_\theta/\partial z$ 的最大特征值过 1 可自动微分得到），再用 rollout 验证发现的分支，形成"枚举 ↔ 学习"的交替精化。

**接口与可行性。** 双向依赖：枚举解决 Q2 最致命的两个问题——训练点覆盖与无真值评估；反过来学到的光滑 $D_\theta$ 让延拓、雅可比特征值与分叉检测可用自动微分廉价完成。合理顺序：固定 $x_0$ 完成枚举 → 以枚举为骨架训练条件采样器 → 研究模态树随 $x_0$ 的二参数分岔以支持在线 MPC。可行性：核心构件在仓库中已具备，GMM/Griewank/pendulum 上可在天级别验证"梯度自由能量匹配 + 模态分层训练"是否优于自举外循环；仿真成本每训练点 $N\approx10^3$ 条 rollout、$10^4$–$10^5$ 个训练点，小系统可接受，humanoid 需实测。风险：rollout-only 是未被验证的 regime；摊销收益不确定（S2R 在 GPU 上已经很快），应事先定义"每步 rollout 数减少 $k$ 倍且闭环代价不变"的验收标准；接触伪模态污染分层训练集，依赖 Q1 的剪枝可靠性。

### 4.2 问题六：强化学习与模态枚举

**问题陈述。** RL 与枚举有相同的输入（代价/奖励 + 仿真器）。control-as-inference 下 RL 的最优轨迹分布 $p(\tau\mid O)\propto\exp(R/\alpha)$ 与 $\pi\propto e^{-C/\lambda}$ 在确定性动力学下完全一致（$\alpha\leftrightarrow\lambda$）；差别在摊销的载体：RL 把在线计算摊销成跨状态的参数化策略（通常单峰高斯），枚举摊销成固定初始状态下的开环模态集。四个接口问题：(a) 枚举结果能否作为 RL 的多模态课程/初始化；(b) RL 策略能否作为枚举的 warm start；(c) 枚举能否作为 RL mode collapse 的可测诊断；(d) 多模态计划集在模型误差下的鲁棒性——分叉不只发生在 $h$ 方向，也发生在物理参数 $\theta$ 方向。难点：开环节点空间的模态与闭环状态分布上的策略空间不匹配（两个开环模态在反馈下可能等价，反之可能分裂）；RL 用高斯策略对多峰目标做 M-投影，天然平均或塌缩且不知道丢了哪些模态；质量度量的是开环盆地体积，不等于闭环价值——窄但可镇定的模态可能是 RL 的最优选择。真机方面，RL 的 sim2real 与枚举面临同样的"仿真器从哪来"，且 RL 已有一套实践。

**相关研究。** MPC 即 RL：PI²（Theodorou et al. 2010）在策略参数空间做与 MPPI 同构的加权平均——"RL = 参数空间的 mode seeking"；信息论 MPPI（Williams et al. 2017）明确定位为 model-based RL；Levine (2018) 的 control-as-inference 教程；Kappen (2005) 的路径积分与对称破缺；soft Q-learning（Haarnoja et al. 2017）是 RL 中最早明确保留多峰的方法。两级摊销：POLO（Lowrey et al. 2019）用学到的 value 做终端代价 + 在线 MPPI；TD-MPC/TD-MPC2（Hansen et al. 2022, 2024）在隐空间用 MPPI，policy prior 采样 + 上一步解 warm start——但 prior 是单峰高斯。多样性：MAP-Elites（Mouret & Clune 2015）与 Cully et al. (2015) 的"离线行为地图 → 在线适应"；DIAYN（Eysenbach et al. 2019）由判别器定义多样性；DvD（Parker-Holder et al. 2020）用行为嵌入核矩阵行列式度量种群多样性；"One Solution is Not All You Need"（Kumar et al. 2020）训练隐变量条件策略覆盖同一 MaxEnt 目标的多个解，以便真机上至少一个模态可迁移。规划器蒸馏：GPS（Levine & Koltun 2013）、OPT-Mimic（Fuchioka et al. 2023）都只蒸馏单一解。扩散策略的 RL 微调：DPPO（Ren et al. 2025）零样本 sim2real 且强调结构化探索，但未测量是否保留多峰。sim2real 实践：域随机化（Tobin et al. 2017；Peng et al. 2018）、RMA（Kumar et al. 2021）、残差 RL（Johannink et al. 2019）、PETS（Chua et al. 2018）、SimOpt（Chebotar et al. 2019）——都把模型误差处理为对 $\theta$ 取期望或自适应，隐含单一策略/单一链。

**存在的问题。** 反馈下的模态等价没有从代价地形内生的定义；用哪些模态训练缺判据；mode collapse 缺真值度量，跨状态枚举的代价指数级；policy prior 作为 warm start 会把延拓起点集中在已塌缩的分支；$\theta$ 方向的分叉（域随机化对 $\theta$ 取期望会抹掉"部分 $\theta$ 下不存在"的模态）没有工作沿 $\theta$ 做延拓；模态沿状态轨迹消失时的在线切换规则；理论上高斯策略的 soft policy iteration 是否也是某个平滑密度上的 EM、其塌缩是否对应 $\lambda_{\max}(\Sigma^{-1}\mathrm{Cov})$ 穿过 1；"多模态计划集更鲁棒"尚无隔离机制的实验。

**可能的思路。**
1. 逐模态模仿课程：对每个枚举模态 rollout 得参考 $(s,a)$ 序列，用参考状态初始化训练 mode-conditioned 策略 $\pi(a\mid s,k)$；以质量作采样权重、以持久性（势垒）作难度排序。
2. 枚举作为 mode collapse 诊断：在诊断状态上得 $\{V_k,w_k\}$；策略加噪 rollout 的节点序列作初值跑小 $h$ 的 Tweedie 迭代，收敛到哪个不动点即归属哪个盆地；报告覆盖 $c_k$、$\mathrm{KL}(w\|c)$、缺失模态数、以及收敛到枚举之外不动点的"伪模态"（提示枚举遗漏）。可检验 DvD/DIAYN 度量是否与之一致。
3. 相对于 policy prior 的发现：以 RL 策略样本为大 $h$ 处起点沿 $h$ 下行延拓，只追踪 prior 未覆盖的分支——同时是诊断与补全。
4. $\theta$ 延拓与 robust mass：把延拓变量换成物理参数 $\theta$（或 $(h,\theta)$ 二维），对每个模态沿 $\theta\sim p(\theta)$ 追踪不动点是否 fold 消失；robust mass $=\mathbb E_\theta[w_k(\theta)\mathbb 1\{k\text{ 在 }\theta\text{ 下存在}\}]$。名义质量高但 $\theta$ 脆弱的模态是预期的 sim2real 失败点。
5. 两级摊销：离线枚举 $K$ 个模态，在线 $K$ 条并行 MPPI/DIAL 链各初始化于一个模态，终端代价用 RL 学的 value，链间按质量 × 当前代价分配资源——把 TD-MPC2 的单峰 prior 换成模态集 prior。
6. 用枚举模态集生成多模态演示训练 Diffusion Policy，再 DPPO 微调，用思路 2 在前后测覆盖；若塌缩则加盆地覆盖辅助奖励。
7. 残差 RL/RMA 的模态化：以每个开环模态为名义，学残差策略；适应模块估计 $\hat\theta$ 落出某模态的 fold 区域时按 robust mass 表切换模态（Cully 的代价地形版本）。
8. 理论：把高斯策略的 soft policy iteration 写成对 $\exp(Q/\alpha)$ 的 M-投影迭代，检验其收敛率是否为 $\lambda_{\max}(\Sigma^{-1}\mathrm{Cov})$、$\alpha$ 退火的临界温度是否与 Rose 判据一致；成立则可在 RL 训练中在线计算该特征值作塌缩预警。

**接口与可行性。** RL 的 mode collapse 与 MPPI 的单峰塌缩是同一算子在不同空间的同一现象，枚举中的分叉检测、分支切换、质量分配正是 RL 缺少的部分；反之枚举不能跨状态泛化，这是 RL 的强项。仓库已有的离线枚举流程、MPPI/DIAL 链与 Tweedie 迭代直接支撑诊断、prior 相对发现与两级摊销；$\theta$ 延拓只需把延拓变量换成 MJX 模型参数（`domain_randomize_model` 已存在）；RL 侧需接入 MJX/Brax PPO，是工程量。风险：开环模态与闭环行为对应可能很弱（先用诊断验证再投入训练）；盆地归属需多次 rollout，高维上昂贵；$\theta$ 延拓在接触任务里被非光滑干扰；无硬件时 sim2real 主张只能用 hold-out $\theta$ 代理；概念上可能被视为"多起点 + 模仿 + 域随机化"的组合，必须靠诊断量与理论联系区分。

## 5. 利用：问题三、四

### 5.1 问题三：一步或低串行深度的在线推理

**问题陈述。** 去噪需要很多串行步，所以只能离线；一步去噪不准，因为一步 Tweedie/MPPI 从 $z$ 跳到的是高噪声级的后验均值，当后验横跨多个盆地时它是各盆地均值的质量加权平均，落在任何盆地之外。目标：给定状态与每个控制周期 $N$ 条并行 rollout 的预算，以 $O(1)$ 串行深度——理想是一次批量 rollout 加一次归约——得到 $\pi_\lambda$ 的近模态解。难点：(i) 峰内一步收缩需要 $h^2\gg\lambda/\mu_{\min}(H_k)$（大 $h$），而盆地分离需要 $h\ll\Delta$，两者同时成立要求地形"信噪比" $\Delta^2\mu_{\min}(H_k)/\lambda\gg1$；(ii) 精确的一步映射 $z\mapsto V^*(z)$（概率流的 flow map，或 TSR $k\to\infty$ 恢复的 mean shift 极限）在盆地边界不连续，任何光滑的摊销近似（一致性模型、flow map、MeanFlow、shortcut）在那里 Lipschitz 常数爆炸，标准误差界失效；(iii) MC score 在大 $h$ 时 ESS 崩溃，单个"大步"恰是统计上最差的一步；(iv) 在线 MPC 用上一步解的平移做 warm start 掩盖了问题——这是隐式的时间上的模态选择，在盆地切换（接触模式改变、新障碍）时失效，而那正是需要枚举的时刻。

**相关研究。** 一致性模型（Song et al. 2023）与改进的一致性训练（Song & Dhariwal 2024，无需教师）；flow map matching（Boffi et al. 2024）把一致性模型、CTM（Kim et al. 2024）、渐进蒸馏（Salimans & Ho 2022）统一为两时刻 flow map 的近似；MeanFlow（Geng et al. 2025）与 shortcut models（Frans et al. 2024）从零训练一步模型、只需瞬时速度；一致性模型的收敛保证（Lyu et al. 2023）——误差界依赖 Lipschitz 常数，未利用多模态。机器人一步策略：Consistency Policy（Prasad et al. 2024）、OneDP（Wang et al. 2024）沿链做 KL 蒸馏——mode-seeking，会在每个状态塌到一个模态。能量目标的一步采样器：Jutras-Dubé et al. (2026) 的自蒸馏 + 确定性流 IS 权重、Single-Step Consistent Diffusion Samplers、Zhang et al. (2024) 用 IS 重加权保持无偏。并行时间：ParaDiGMS（Shih et al. 2023）把 $T$ 步串行去噪写成整条轨迹上的 Picard 不动点迭代，一次并行评估所有 drift。在线 MPC：DIAL-MPC 用平移解 warm start 的少步退火达到 50 Hz；MPC 作教师的策略蒸馏（MPC-Net，Carius et al. 2020，为四足显式用混合专家；PLATO；DAgger）；Sacks & Boots (2022) 的学习提议分布。理论支撑：Koehler & Vuong (2023) 证明用数据初始化的短 Langevin 链能正确采样 log-concave 混合——初始化携带模态权重，短链只做局部工作。输出头结构：MultiPath（Chai et al. 2019）一次前向输出锚点上的分类 + 每锚点偏移；VQ-BeT（Lee et al. 2024）的离散码 + 残差。

**存在的问题。** 没有一步方法处理 flow map 在盆地边界的不连续；"一步可行性"条件 $\Delta^2\mu_{\min}/\lambda\gg1$ 只是从 mean shift 收敛率猜出的，非高斯或接触非光滑盆地下未建立；ESS 受限的 score 误差如何通过单个大步传播；对状态 $x$ 的摊销——模态集、质量、盆地随 $x$ 变化，跨状态的模态身份如何保持一致；蒸馏目标的 mode collapse；Picard 并行在分叉附近无收敛保证；在线盆地切换的 $O(1)$ 检测与重选没有原则性处理；接触下局部 Hessian 可能不存在。

**可能的思路。**
1. **模态并行、盆地受限的 mean shift**（$O(1)$ 串行深度）：给定 $K$ 个枚举模态 $\{\mu_k,\Sigma_k,w_k\}$，每个周期每模态采 $N/K$ 个样本 $V\sim\mathcal N(\mu_k(x),h_k^2)$，$h_k$ 取得使峰内收缩充分而与其他盆地的 Bhattacharyya 重叠很小；权重 $e^{-C/\lambda}\mathbb 1[V\in B_k]$（按 $\Sigma_k$ 度量下最近模态判归属）；每模态一步 Tweedie；门控按更新后的加权代价或质量。一次批量 rollout 加归约。误差界：$\mathbb E\|V_k^+-\mu_k^*\|\le r_k\|\mu_k-\mu_k^*\|+O(\sqrt{\mathrm{tr}\,\mathrm{Cov}_{\text{post}}/\mathrm{ESS}_k})$ 加门控误判项。
2. 盆地内的 TSR：门控已处理模态权重，TSR"保持权重"的性质不再需要，可在每个盆地内取大 $k$ 而无开环分叉问题。
3. 带 rollout 教师的一致性训练：用 MC Tweedie 估计作为一致性训练的无偏 score；给 $f_\theta$ 一个混合专家头，每个枚举模态一个专家，门控由尺度空间追踪的盆地标签监督——把不连续移出每个专家，使 Lipschitz 型误差界逐专家成立。
4. 以 S2R 规划器为教师的策略蒸馏（MPC-Net/PLATO/DAgger 循环）：离线从多个状态跑完整退火，记录全部模态与质量，训练 MultiPath 式头（模态分类 + 每模态偏移），在线策略门控与规划器不一致处做 DAgger 重标注。
5. Picard 并行 S2R：把 $T$ 步链写成整条轨迹的不动点，一次批量 MJX 调用评估所有 MC score，用尺度空间延拓路径 $h\mapsto\mu_k(h)$ 作初值，使迭代 $O(1)$ 轮收敛。
6. Warm start 即门控 + 切换检测（DIAL-MPC 扩展）：主链在当前盆地内少步循环，同时维护 $K$ 条影子链各每周期一步，某影子链代价领先超过持久性裕度（模态树的势垒）时切换。
7. 需要模态质量的随机策略：按 $w_k$ 抽 $k$、盆地内一步、以 $e^{-C/\lambda}/q(V)$ 重加权校正门控与专家误差。
8. 精确陈述"一步可行性"条件：以高斯混合替代 $p_h$，一步无限制 Tweedie 在 $\varepsilon$ 内到达某模态当且仅当主导模态的后验责任度超过 $1-\varepsilon'$ 且 $r_k\le\varepsilon$；导出可行的 $h$ 区间，证明其非空当且仅当 $\Delta_{kj}^2\mu_{\min}(H_k)/\lambda$ 超过常数乘 $\log(w_j/w_k)+d$——把"一步 MPPI 为何失败"变成可从枚举模态计算的地形统计量。

**接口与可行性。** 一步失败的根源是模态结构而非步数；枚举把不连续的部分移进离散门控（对 $K$ 个模态并行），每个专家只剩光滑、收缩的一步映射，误差由收敛率 $r_k$ 控制。模态树提供锚点与门控目标；势垒给在线切换裕度；延拓路径是 Picard 并行的理想初值；Koehler–Vuong 定理说明初始化带对质量后只需短局部链。可行：模态并行受限 mean shift 是对 two_stage/mcsa 的小改动，一次批量 rollout；先在 Griewank/GMM/pendulum 验证，再上已聚类模态的 humanoid。风险：接触任务的地形信噪比可能不满足一步条件——此时 2–4 步（如 DIAL-MPC）是诚实答案，贡献变为"何时一步足够"的判据；最近模态判归属对非凸盆地不对；跨状态模态身份漂移；mode-seeking 蒸馏目标会塌缩，除非模态变量显式。

### 5.2 问题四：从原始传感器到可用的仿真器

**问题陈述。** 真机只有本体感知、相机、LiDAR 与已知的机器人 URDF；输出一个 MJX 可加载、可被采样式优化器每个 MPC 步调用数千次的仿真器：(a) 碰撞环境（网格/高度场/基元）；(b) 被操作物体的物理功能（质量、惯量、摩擦、关节类型/轴/限位/阻尼，如抽屉、门）；以及 $\theta$ 的不确定性描述。难点：**表示不匹配**——感知给的是辐射场、高斯泼溅、TSDF、语义点云，接触求解器要的是水密凸块、铰接运动树、接触参数；现有"GS + 物理"系统要么在泼溅上直接跑连续介质求解器（材料手工设定），要么悄悄退回标准引擎里的网格、GS 只管渲染。**可观测性**——对接触模态最关键的参数（摩擦、恢复系数、铰链阻尼、质量分布）在交互前不可见，且最好的仿真器在真实碰撞上也不一致（Acosta et al. 2022）。**误差放大**——优化器取 $N\times H\times K$ 条 rollout 的 argmax，偏好利用仿真器不准的控制序列；枚举刻意探索每个盆地，包括只在错误模型里存在的盆地。因此问题不只是"怎么建孪生"，而是"$\pi_\theta$ 的哪些模态在 $\theta_{\text{sim}}\to\theta_{\text{real}}$ 下幸存"，以及怎样把孪生的不确定性在模态层面消费。

**相关研究。** GS + 物理：PhysGaussian（Xie et al. 2024，MPM 状态附在高斯核上，材料手工设定）；RoboGSim（Li et al. 2024）、SplatSim、VR-Robo——GS 管外观、物理仍来自网格；Real-is-Sim（Abou-Chakra et al. 2025，基于 PEGS）在 30–60 Hz 用图像残差的"视觉力"同步物理孪生，策略只在孪生里动作；PhysTwin（Jiang et al. 2025）从交互视频拟合弹簧-质量模型并用于 MPC；PAC-NeRF（Li et al. 2023）联合几何与物理参数辨识。铰接物体：Ditto（Jiang et al. 2022）从交互前后点云预测关节；PARIS、ArtGS；Real2Code（Mandi et al. 2024）用 LLM 生成关节代码；URDFormer（Chen et al. 2024）从单张图生成整场景 URDF（动力学为模板默认值）。数字表亲（Dai et al. 2024）：不求精确孪生，求分布上的鲁棒性。系统辨识：ASID（Memmel et al. 2024）主动探索最大化 Fisher 信息；BayesSim（Ramos et al. 2019）似然自由推断 $p(\theta\mid\text{real})$；SimOpt（Chebotar et al. 2019）；RialTo（Torne et al. 2024）real-to-sim-to-real 流程。仿真器保真度：Acosta et al. (2022) 对 Drake/MuJoCo/Bullet 在真实碰撞上的验证——非弹性碰撞好、弹性碰撞差。学习的世界模型：DINO-WM（Zhou et al. 2024）在冻结特征空间学动力学并用 CEM 规划；Cosmos、Genie 的视频世界模型（目前太慢、动作条件弱）。

**存在的问题。** 模态层面的 sim-to-real 差距无人测量（基准只看轨迹或像素误差）；表示不匹配未解决；接触参数在交互前不可观测且 MuJoCo 有保真度下限；单视角铰接病态；优化器利用仿真伪影（风险策略只在盆地内缓解，不消除伪盆地）；重建（秒—分钟）与 MPC（毫秒）的时间尺度不匹配，何时重同步无原则；持久性鲁棒性需要模型误差的一致范数界，随机版本未推导；学习的世界模型缺少尺度空间解释。

**可能的思路。**
1. 双轨孪生（工程基线）：物理来自 LiDAR/RGB-D → TSDF → 网格 → 凸分解 → MJX 几何体（地形用高度场），外观只在代价需要图像时用 GS；明确 GS 对 rollout 物理无贡献。
2. 探后重建的铰接：Real2Code/URDFormer 给先验 URDF，一次短推/拉，Ditto/PARIS 两状态重建修正关节，再用 BayesSim 式推断辨识铰链摩擦/阻尼，输出 URDF + $p(\theta\mid\text{data})$。
3. 在 rollout 内消费 $p(\theta\mid\text{data})$：每样本抽 $\theta_i$（`domain_randomize_model`）配风险策略，目标变为 $\bar\pi(V)=\int e^{-C(V;\theta)/\lambda}p(\theta)\mathrm d\theta$——注意它平均的是 $e^{-C}$ 而非 $V$，合并或消灭模态的方式与 $h$ 平滑不同，应先在二维基准上刻画。
4. **持久性证书**：由持久性图的稳定性定理（Cohen-Steiner, Edelsbrunner & Harer 2007），若 $\sup_V|C_{\text{sim}}-C_{\text{real}}|\le\varepsilon$，则持久性大于 $2\varepsilon$ 的每个模态在真实地形中都有对应；枚举已算持久性，只需从 hold-out 真实 rollout 或 $\theta$ 后验的散布估计 $\varepsilon$，剪掉持久性低于 $2\varepsilon$ 的模态——把"持久的模态更鲁棒"从口号变成可检验判据。
5. $(h,\theta)$ 二参数延拓：在 $\theta_{\text{MAP}}$ 枚举后沿后验样本或主方向延拓每个不动点，分支在可信区域内无 fold 即"鲁棒"；局部灵敏度 $\mathrm dV^*/\mathrm d\theta=-H^{-1}\partial^2C/\partial V\partial\theta$ 可用 Tweedie 后验协方差作 $H^{-1}$ 由 rollout 估计。
6. 模态判别的主动辨识：不是 ASID 的 Fisher 信息，而是选短探测动作最大化各枚举模态预测观测之间的分歧（对离散变量"哪个模态是真的"做贝叶斯实验设计）；只需辨识改变模态集的 $\theta$ 方向。
7. 几何尺度空间——障碍膨胀：以膨胀半径 $r$ 为延拓变量，在 $r$ 小于重建误差（体素尺寸）时消失的模态（窄缝穿越）剪掉，得到每模态的"间隙持久性"。
8. 同步孪生内的事件触发重枚举：采用 Real-is-Sim 架构，残差幅度是模型误差信号，超过当前执行模态的持久性裕度时才重枚举，否则在盆地内廉价精化。
9. 隐空间世界模型作 rollout 引擎：在 DINO-WM 式隐动力学里跑 S2R，检验其枚举模态是否与 MuJoCo 孪生一致。

**接口与可行性。** 模型误差是对 $C$ 的扰动，作用在尺度空间分析的同一对象上：持久性 = 鲁棒性（精确到 $2\varepsilon$）；延拓可沿 $\theta$ 做；模态集本身可作为辨识目标。可行：地形高度场与基元/凸包已可接 MJX；每样本 $\theta$ 随机化与风险策略已有；持久性剪枝与 $\theta$ 延拓复用模态追踪器，可先在数值任务上以扰动目标验证。铰接辨识需要交互与仓库外的重建栈。风险：MJX 只支持凸网格；弹性碰撞保真度下限使冲击型模态不可验证；一致范数界难得、后验界可能过于保守；模态判别探测可能不安全或不可逆；学习世界模型无模态对应保证。**战略上**：real2sim 是独立的研究领域，本论文的贡献应停留在接口——模型不确定性下的模态鲁棒性——配最小的孪生，而非新的重建方法。

## 6. 依赖、优先级与共同的风险

**依赖关系。** 问题一是所有其他问题的输入：它产出的模态集、质量、势垒分别被 Q2（训练分层与评估）、Q3（锚点、门控、切换裕度）、Q6（课程、诊断、robust mass）、Q4（持久性证书、$\theta$ 延拓）消费；Q7 决定问题一的输出是否有意义（持久性阈值与意图备选是同一个数的两面）；Q5 是问题一的"有数据版本"，方法复用。

**优先级。** 一 → 二 → 三 是方法论文的主干（枚举 → 摊销 → $O(1)$ 推理），每篇的对比基线都是"没有模态结构时的做法"，而不是各领域的 SOTA。七是主干的地基，应尽早把持久性诊断做出来。五、六是有独立贡献的支线：五可行且切中示教学习的软肋，六的 QD 联系与 $\theta$ 延拓值得写进相关工作并做对比。四以消费者身份参与：用最小的孪生验证"持久模态在模型误差下幸存"。

**共同的假设与风险。** 所有问题都建立在"模态 = 有意义的任务备选"这一假设上，而它取决于持久性阈值、行为商空间与代价设计。三个反复出现的技术风险：高维下高斯尺度空间可以制造模态（枚举完备性只能是条件性的）；不等质量模态以 fold 而非 pitchfork 出现（沿特征向量的分支切换会漏掉小模态）；蒙特卡洛 score 在大 $h$ 高维下 ESS 崩溃（分叉检测与质量估计都需要统计保护）。这三点应在问题一的第一篇里正面处理，其余问题才有可靠的输入。

## 7. 文献

1. C. Pan, Z. Yi, G. Shi, G. Qu. Model-based diffusion for trajectory optimization. *NeurIPS*, 2024. arXiv:2407.01573.
2. H. Xue, C. Pan, Z. Yi, G. Qu, G. Shi. Full-order sampling-based MPC for torque-level locomotion control via diffusion-style annealing (DIAL-MPC). *ICRA*, 2025. arXiv:2409.15610.
3. X. Huang, H. Dong, Y. Hao, Y. Ma, T. Zhang. Reverse diffusion Monte Carlo. *ICLR*, 2024. arXiv:2307.02037.
4. G. Raya, L. Ambrogioni. Spontaneous symmetry breaking in generative diffusion models. *NeurIPS*, 2023. arXiv:2305.19693.
5. G. Biroli, T. Bonnaire, V. de Bortoli, M. Mézard. Dynamical regimes of diffusion models. *Nature Communications* 15:9957, 2024. arXiv:2402.18491.
6. M. Li, S. Chen. Critical windows: non-asymptotic theory for feature emergence in diffusion models. *ICML*, 2024. arXiv:2403.01633.
7. B. W. Silverman. Using kernel density estimates to investigate multimodality. *JRSS-B* 43(1):97–99, 1981.
8. M. C. Minnotte, D. W. Scott. The mode tree: a tool for visualization of nonparametric density features. *JCGS* 2(1):51–68, 1993.
9. P. Chaudhuri, J. S. Marron. SiZer for exploration of structures in curves. *JASA* 94:807–823, 1999.
10. M. Á. Carreira-Perpiñán. Gaussian mean-shift is an EM algorithm. *IEEE TPAMI* 29(5):767–776, 2007.
11. M. Á. Carreira-Perpiñán, C. K. I. Williams. On the number of modes of a Gaussian mixture. *Scale-Space*, LNCS 2695, 2003.
12. F. Chazal, L. J. Guibas, S. Y. Oudot, P. Skraba. Persistence-based clustering in Riemannian manifolds. *J. ACM* 60(6), 2013.
13. D. Cohen-Steiner, H. Edelsbrunner, J. Harer. Stability of persistence diagrams. *Discrete Comput. Geom.* 37:103–120, 2007.
14. D. J. Wales, J. P. K. Doye. Global optimization by basin-hopping and the lowest energy structures of Lennard-Jones clusters. *J. Phys. Chem. A* 101(28):5111–5116, 1997.
15. O. M. Becker, M. Karplus. The topology of multidimensional potential energy surfaces. *J. Chem. Phys.* 106(4):1495–1517, 1997.
16. X. Li, M. G. Epitropakis, K. Deb, A. Engelbrecht. Seeking multiple solutions: an updated survey on niching methods and their applications. *IEEE TEVC* 21(4):518–538, 2017.
17. V. Vassiliades, K. Chatzilygeroudis, J.-B. Mouret. Using centroidal Voronoi tessellations to scale up MAP-Elites. *IEEE TEVC*, 2018. arXiv:1610.05729.
18. S. Patrick, E. Bakolas. Path integral control with rollout clustering and dynamic obstacles. *ACC*, 2024. arXiv:2403.18066.
19. A. Lambert, A. Fishman, D. Fox, B. Boots, F. Ramos. Stein variational model predictive control. *CoRL*, 2020. arXiv:2011.07641.
20. T. Osa. Multimodal trajectory optimization for motion planning. *IJRR* 39(8):983–1001, 2020.
21. Q. Zhang, H. Chen. Path integral sampler: a stochastic control approach for sampling. *ICLR*, 2022. arXiv:2111.15141.
22. F. Vargas, W. Grathwohl, A. Doucet. Denoising diffusion samplers. *ICLR*, 2023. arXiv:2302.13834.
23. J. Berner, L. Richter, K. Ullrich. An optimal control perspective on diffusion-based generative modeling. *TMLR*, 2024. arXiv:2211.01364.
24. L. Richter, J. Berner. Improved sampling via learned diffusions. *ICLR*, 2024. arXiv:2307.01198.
25. T. Akhound-Sadegh et al. Iterated denoising energy matching for sampling from Boltzmann densities. *ICML*, 2024. arXiv:2402.06121.
26. R. OuYang, B. Qiang, J. M. Hernández-Lobato. BNEM: a Boltzmann sampler based on bootstrapped noised energy matching. *TMLR*, 2026. arXiv:2409.09787.
27. A. Havens et al. Adjoint sampling: highly scalable diffusion samplers via adjoint matching. arXiv:2504.11713, 2025.
28. F. Noé, S. Olsson, J. Köhler, H. Wu. Boltzmann generators. *Science*, 2019. arXiv:1812.01729.
29. L. I. Midgley et al. Flow annealed importance sampling bootstrap. *ICLR*, 2023. arXiv:2208.01893.
30. J. He et al. No trick, no treat: pursuits and challenges towards simulation-free training of neural samplers. arXiv:2502.06685, 2025.
31. D. Blessing et al. Beyond ELBOs: a large-scale evaluation of variational methods for sampling. *ICML*, 2024. arXiv:2406.07423.
32. M. Sendera et al. Improved off-policy training of diffusion samplers. *NeurIPS*, 2024. arXiv:2402.05098.
33. J. Sacks, B. Boots. Learning sampling distributions for model predictive control. *CoRL*, 2022. arXiv:2212.02587.
34. G. Zhou et al. Diffusion model predictive control. *TMLR*, 2024. arXiv:2410.05364.
35. Y. Song, P. Dhariwal, M. Chen, I. Sutskever. Consistency models. *ICML*, 2023. arXiv:2303.01469.
36. Y. Song, P. Dhariwal. Improved techniques for training consistency models. *ICLR*, 2024. arXiv:2310.14189.
37. N. M. Boffi, M. S. Albergo, E. Vanden-Eijnden. Flow map matching with stochastic interpolants. arXiv:2406.07507, 2024.
38. D. Kim et al. Consistency trajectory models. *ICLR*, 2024. arXiv:2310.02279.
39. T. Salimans, J. Ho. Progressive distillation for fast sampling of diffusion models. *ICLR*, 2022. arXiv:2202.00512.
40. Z. Geng et al. Mean flows for one-step generative modeling. arXiv:2505.13447, 2025.
41. K. Frans et al. One step diffusion via shortcut models. arXiv:2410.12557, 2024.
42. J. Lyu, Z. Chen, S. Feng. Convergence guarantee for consistency models. arXiv:2308.11449, 2023.
43. A. Prasad et al. Consistency policy: accelerated visuomotor policies via consistency distillation. *RSS*, 2024. arXiv:2405.07503.
44. Z. Wang et al. One-step diffusion policy: fast visuomotor policies via diffusion distillation. arXiv:2410.21257, 2024.
45. P. Jutras-Dubé et al. One-step diffusion samplers via self-distillation and deterministic flow. *AISTATS*, 2026. arXiv:2512.05251.
46. F. Zhang et al. Efficient and unbiased sampling of Boltzmann distributions via consistency models. arXiv:2409.07323, 2024.
47. A. Shih et al. Parallel sampling of diffusion models (ParaDiGMS). *NeurIPS*, 2023. arXiv:2305.16317.
48. J. Carius, F. Farshidian, M. Hutter. MPC-Net: a first principles guided policy search. *IEEE RA-L*, 2020. arXiv:1909.05197.
49. G. Kahn et al. PLATO: policy learning using adaptive trajectory optimization. *ICRA*, 2017. arXiv:1603.00622.
50. S. Ross, G. Gordon, D. Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning (DAgger). *AISTATS*, 2011.
51. F. Koehler, T. Vuong. Sampling multimodal distributions with the vanilla score: benefits of data-based initialization. arXiv:2310.01762, 2023.
52. Y. Chai et al. MultiPath: multiple probabilistic anchor trajectory hypotheses for behavior prediction. *CoRL*, 2019. arXiv:1910.05449.
53. S. Lee et al. Behavior generation with latent actions (VQ-BeT). *ICML*, 2024. arXiv:2403.03181.
54. T. Xie et al. PhysGaussian: physics-integrated 3D Gaussians for generative dynamics. *CVPR*, 2024. arXiv:2311.12198.
55. X. Li et al. RoboGSim: a real2sim2real robotic Gaussian splatting simulator. arXiv:2411.11839, 2024.
56. J. Abou-Chakra et al. Real-is-Sim: bridging the sim-to-real gap with a dynamic digital twin. arXiv:2504.03597, 2025.
57. H. Jiang et al. PhysTwin: physics-informed reconstruction and simulation of deformable objects from videos. arXiv:2503.17973, 2025.
58. X. Li et al. PAC-NeRF: physics augmented continuum neural radiance fields for geometry-agnostic system identification. *ICLR*, 2023. arXiv:2303.05512.
59. Z. Jiang, C.-C. Hsu, Y. Zhu. Ditto: building digital twins of articulated objects from interaction. *CVPR*, 2022. arXiv:2202.08227.
60. Z. Mandi et al. Real2Code: reconstruct articulated objects via code generation. arXiv:2406.08474, 2024.
61. Z. Chen et al. URDFormer: a pipeline for constructing articulated simulation environments from real-world images. *RSS*, 2024. arXiv:2405.11656.
62. T. Dai et al. Automated creation of digital cousins for robust policy learning. *CoRL*, 2024. arXiv:2410.07408.
63. M. Memmel et al. ASID: active exploration for system identification in robotic manipulation. *ICLR*, 2024. arXiv:2404.12308.
64. F. Ramos, R. C. Possas, D. Fox. BayesSim: adaptive domain randomization via probabilistic inference for robotics simulators. *RSS*, 2019. arXiv:1906.01728.
65. Y. Chebotar et al. Closing the sim-to-real loop: adapting simulation randomization with real world experience. *ICRA*, 2019. arXiv:1810.05687.
66. M. Torne et al. Reconciling reality through simulation: a real-to-sim-to-real approach for robust manipulation (RialTo). *RSS*, 2024. arXiv:2403.03949.
67. B. Acosta, W. Yang, M. Posa. Validating robotics simulators on real-world impacts. *IEEE RA-L*, 2022. arXiv:2110.00541.
68. G. Zhou et al. DINO-WM: world models on pre-trained visual features enable zero-shot planning. arXiv:2411.04983, 2024.
69. L. X. Shi et al. Waypoint-based imitation learning for robotic manipulation (AWE). *CoRL*, 2023. arXiv:2307.14326.
70. C. Wen et al. Keyframe-focused visual imitation learning. *ICML*, 2021. arXiv:2106.06452.
71. S. James, A. J. Davison. Q-attention: enabling efficient learning for vision-based robotic manipulation. *IEEE RA-L*, 2022. arXiv:2105.14829.
72. W. Huang et al. ReKep: spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. *CoRL*, 2024. arXiv:2409.01652.
73. L. Manuelli et al. kPAM: keypoint affordances for category-level robotic manipulation. *ISRR*, 2019. arXiv:1903.06684.
74. A. L. P. Ureche et al. Task parameterization using continuous constraints extracted from human demonstrations. *IEEE T-RO* 31(6):1458–1471, 2015.
75. S. Niekum et al. Learning grounded finite-state representations from unstructured demonstrations. *IJRR* 34(2):131–157, 2015.
76. G. Chou, D. Berenson, N. Ozay. Learning constraints from demonstrations. *WAFR*, 2018. arXiv:1812.07084.
77. A. Mandlekar et al. MimicGen. *CoRL*, 2023. arXiv:2310.17596.
78. N. M. Shafiullah et al. Behavior transformers: cloning k modes with one stone. *NeurIPS*, 2022. arXiv:2206.11251.
79. P. Florence et al. Implicit behavioral cloning. *CoRL*, 2021. arXiv:2109.00137.
80. C. Chi et al. Diffusion policy: visuomotor policy learning via action diffusion. *RSS*, 2023. arXiv:2303.04137.
81. P. de Haan, D. Jayaraman, S. Levine. Causal confusion in imitation learning. *NeurIPS*, 2019. arXiv:1905.11979.
82. Lu et al. When would vision-proprioception policies fail in robotic manipulation? *ICLR*, 2026. arXiv:2602.12032.
83. A. Paraschos, C. Daniel, J. Peters, G. Neumann. Probabilistic movement primitives. *NeurIPS*, 2013.
84. W. Merkt et al. Memory clustering using persistent homology for multimodality- and discontinuity-sensitive learning of optimal control warm-starts. *IEEE T-RO*, 2021. arXiv:2010.01024.
85. W. Xiao et al. SafeDiffuser: safe planning with diffusion probabilistic models. *ICLR*, 2025. arXiv:2306.00148.
86. J. Carvalho et al. Motion planning diffusion. *IROS*, 2023. arXiv:2308.01557.
87. E. Theodorou, J. Buchli, S. Schaal. A generalized path integral control approach to reinforcement learning. *JMLR* 11, 2010.
88. G. Williams et al. Information theoretic MPC for model-based reinforcement learning. *ICRA*, 2017.
89. G. Williams et al. Information-theoretic model predictive control: theory and applications to autonomous driving. *IEEE T-RO*, 2018. arXiv:1707.02342.
90. S. Levine. Reinforcement learning and control as probabilistic inference: tutorial and review. arXiv:1805.00909, 2018.
91. H. J. Kappen. Path integrals and symmetry breaking for optimal control theory. *J. Stat. Mech.*, 2005. arXiv:physics/0505066.
92. T. Haarnoja, H. Tang, P. Abbeel, S. Levine. Reinforcement learning with deep energy-based policies. *ICML*, 2017. arXiv:1702.08165.
93. K. Lowrey et al. Plan online, learn offline (POLO). *ICLR*, 2019. arXiv:1811.01848.
94. N. Hansen, X. Wang, H. Su. Temporal difference learning for model predictive control (TD-MPC). *ICML*, 2022. arXiv:2203.04955.
95. N. Hansen, H. Su, X. Wang. TD-MPC2. *ICLR*, 2024. arXiv:2310.16828.
96. J.-B. Mouret, J. Clune. Illuminating search spaces by mapping elites. arXiv:1504.04909, 2015.
97. A. Cully, J. Clune, D. Tarapore, J.-B. Mouret. Robots that can adapt like animals. *Nature*, 2015. arXiv:1407.3501.
98. B. Eysenbach, A. Gupta, J. Ibarz, S. Levine. Diversity is all you need (DIAYN). *ICLR*, 2019. arXiv:1802.06070.
99. J. Parker-Holder et al. Effective diversity in population based reinforcement learning (DvD). *NeurIPS*, 2020. arXiv:2002.00632.
100. S. Kumar, A. Kumar, S. Levine, C. Finn. One solution is not all you need: few-shot extrapolation via structured MaxEnt RL. *NeurIPS*, 2020. arXiv:2010.14484.
101. S. Levine, V. Koltun. Guided policy search. *ICML*, 2013.
102. Y. Fuchioka, Z. Xie, M. van de Panne. OPT-Mimic. *ICRA*, 2023. arXiv:2210.01247.
103. A. Z. Ren et al. Diffusion policy policy optimization (DPPO). *ICLR*, 2025. arXiv:2409.00588.
104. J. Tobin et al. Domain randomization for transferring deep neural networks from simulation to the real world. *IROS*, 2017. arXiv:1703.06907.
105. X. B. Peng et al. Sim-to-real transfer of robotic control with dynamics randomization. *ICRA*, 2018. arXiv:1710.06537.
106. A. Kumar, Z. Fu, D. Pathak, J. Malik. RMA: rapid motor adaptation for legged robots. *RSS*, 2021. arXiv:2107.04034.
107. T. Johannink et al. Residual reinforcement learning for robot control. *ICRA*, 2019. arXiv:1812.03201.
108. K. Chua et al. Deep reinforcement learning in a handful of trials using probabilistic dynamics models (PETS). *NeurIPS*, 2018. arXiv:1805.12114.
109. A. Y. Ng, D. Harada, S. Russell. Policy invariance under reward transformations. *ICML*, 1999.
110. B. D. Ziebart et al. Maximum entropy inverse reinforcement learning. *AAAI*, 2008.
111. Y. J. Ma et al. Eureka: human-level reward design via coding large language models. *ICLR*, 2024. arXiv:2310.12931.
112. W. B. Knox et al. Reward (mis)design for autonomous driving. *Artificial Intelligence* 316, 2023. arXiv:2104.13906.
113. J. Skalse et al. Defining and characterizing reward hacking. *NeurIPS*, 2022. arXiv:2209.13085.
114. M. Bhardwaj et al. STORM: an integrated framework for fast joint-space model-predictive control for reactive manipulation. *CoRL*, 2021. arXiv:2104.13542.
115. S. Bhattacharya, V. Kumar, M. Likhachev. Search-based path planning with homotopy class constraints. *AAAI*, 2010.
116. O. de Groot et al. Topology-driven parallel trajectory optimization in dynamic environments. *IEEE T-RO* 41:110–126, 2025. arXiv:2401.06021.
117. I. Das, J. E. Dennis. A closer look at drawbacks of minimizing weighted sums of objectives for Pareto set generation. *Structural Optimization* 14:63–69, 1997.
118. Y. Xu et al. Temporal score rescaling for temperature sampling in diffusion and flow models. arXiv:2510.01184, 2025.
119. K. Rose. Deterministic annealing for clustering, compression, classification, regression, and related optimization problems. *Proc. IEEE* 86(11), 1998.
