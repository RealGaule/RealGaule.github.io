# 写作速查

## 图片

图片放在 `docs/assets/images/`，用相对路径引用。点击图片可以放大查看。

![示例图片](../assets/images/example.svg){ width="480" }

## 公式

行内公式：$E = mc^2$，以及 $\nabla \cdot \mathbf{E} = \rho / \varepsilon_0$。

独立公式：

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

多行对齐：

$$
\begin{aligned}
\mathcal{L}(\theta) &= \mathbb{E}_{x \sim p_{\text{data}}}\left[\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2\right] \\
x_{t-1} &= \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t z
\end{aligned}
$$

## 代码

```python
import numpy as np

x = np.linspace(0, 1, 100)
print(x.mean())
```

## 提示框

!!! note "笔记"
    这是一个提示框。

??? tip "可折叠的内容"
    点击标题展开。

## 表格与任务列表

| 符号 | 含义 |
|---|---|
| $\mu$ | 均值 |
| $\sigma$ | 标准差 |

- [x] 搭好网站
- [ ] 写第一篇笔记

## 脚注

这是一句需要注释的话[^1]。

[^1]: 这是脚注内容。
