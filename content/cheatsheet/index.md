---
title: "写作速查"
date: 2026-09-25
description: "图片、公式、代码、提示框的写法示例。"
summary: "图片、公式、代码、提示框、表格、链接与脚注的 Markdown 写法和渲染效果。"
aliases: ["/notes/example/cheatsheet/", "/notes/cheatsheet/"]   # old /notes/ URLs
hiddenInRss: true                   # a standalone page, not a post
---

每一节先给出 Markdown 写法，再给出渲染效果。

## 图片

图片和文章放在同一个文件夹（页面包，例如 `content/posts/<slug>/`），用 `figure` shortcode 引用。点击图片会单独打开原图文件（在手机上可以放大细看）。标题里可以写公式，编号（“图 1：”）自动生成，不要写进标题。

```markdown
{{</* figure src="example.svg" alt="示例图片" width="480" caption=`示例图片，标题里可以写公式 $e^{i\pi}+1=0$` */>}}
```

{{< figure src="example.svg" alt="示例图片" width="480" caption=`示例图片，标题里可以写公式 $e^{i\pi}+1=0$` >}}

标题用反引号括起来，里面的反斜杠会原样保留。

## 公式

行内公式写在 `$...$` 里，独立公式写在 `$$...$$` 里（`$$` 各占一行）。

```markdown
行内公式：$E = mc^2$，以及 $\nabla \cdot \mathbf{E} = \rho / \varepsilon_0$。

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

$$
\begin{aligned}
a &= b + c \\
d &= e
\end{aligned}
$$
```

渲染效果：

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

代码块用三个反引号围起来，后面写语言名（`python`、`bash`、`cpp` 等）以获得语法高亮：

````markdown
```python
import numpy as np
print(np.pi)
```
````

渲染效果：

```python
import numpy as np

x = np.linspace(0, 1, 100)
print(x.mean())
```

## 提示框

写成引用块，第一行是 `[!类型] 标题`，后面每一行都以 `> ` 开头。类型有 `theorem`（定理）、`definition`（定义）、`conclusion`（结论）、`formula`（公式）、`note`（注）、`tip`（提示）；不写标题时显示括号里的默认标题。在 `]` 后面加 `-` 变成默认折叠，加 `+` 变成可折叠但默认展开。两个提示框之间要空一行。提示框里的独立公式可以写在同一行（`> $$ … $$`），也可以让 `$$` 各占一行，这时公式的每一行同样以 `> ` 开头（见下面的定理示例）。

```markdown
> [!note] 笔记
> 这是一个提示框。

> [!tip]- 可折叠的内容
> 点击标题展开。

> [!theorem] 定理 1（标题里可以写公式 $F=-\lambda\log Z$）
> 正文可以有公式、列表和表格：
>
> $$
> F(U)=\min_r \mathcal A_U[r]
> $$
>
> - 条件：$\|B\| < \infty$
>
> | 符号 | 含义 |
> |---|---|
> | $\lambda$ | 温度 |

> [!note]- 证明（默认折叠）
> 折叠的内容。
```

> [!note] 笔记
> 这是一个提示框。

> [!tip]- 可折叠的内容
> 点击标题展开。

> [!theorem] 定理 1（标题里可以写公式 $F=-\lambda\log Z$）
> 正文可以有公式、列表和表格：
>
> $$
> F(U)=\min_r \mathcal A_U[r]
> $$
>
> - 条件：$\|B\| < \infty$
>
> | 符号 | 含义 |
> |---|---|
> | $\lambda$ | 温度 |

> [!note]- 证明（默认折叠）
> 折叠的内容。

> [!definition]
> 不写标题时显示默认标题“定义”。

> [!conclusion] 结论
> 结论框。

> [!formula] 公式
> $$e^{i\pi}+1=0$$

## 表格与任务列表

```markdown
| 符号 | 含义 |
|---|---|
| $\mu$ | 均值 |
| $\sigma$ | 标准差 |

- [x] 搭好网站
- [ ] 写第一篇笔记
```

渲染效果：

| 符号 | 含义 |
|---|---|
| $\mu$ | 均值 |
| $\sigma$ | 标准差 |

表格里的公式不能出现单独的 `|`（会把单元格切开）：绝对值写 `\lvert x \rvert`，条件竖线写 `\mid`，`\|` 可以直接用。

- [x] 搭好网站
- [ ] 写第一篇笔记

## 链接

链接到站内其他页面用 `relref`（生成站内路径，本地预览和线上都能用；`ref` 生成完整网址），例如 [高斯卷积、自由能与相对熵]({{< relref "/posts/gibbs-convolution" >}})：

```markdown
[高斯卷积、自由能与相对熵]({{</* relref "/posts/gibbs-convolution" */>}})
```

## 脚注

```markdown
这是一句需要注释的话[^1]。

[^1]: 这是脚注内容。
```

渲染效果：

这是一句需要注释的话[^1]。

[^1]: 这是脚注内容。
