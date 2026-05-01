# Day 04：规模法则 —— 越大越强的幂律

> 📚 第十三周 · 预训练与规模法则 · 第 4 天

2020年，OpenAI发表了一项惊人的发现：语言模型的性能与模型规模之间遵循**精确的幂律关系**。参数量增加10倍，loss就按固定比例下降。这不是经验观察，而是可以用数学公式描述的规律。这就是**规模法则(Scaling Laws)**。

**今天的任务**：
1. 理解规模法则的核心发现：L(N) = a · N^(-b) + c
2. 认识参数量、数据量、计算量三者的幂律关系
3. 建立对"越大越好"的数学直觉

---

## 1. 历史剧场：2020，规模法则的发现

在GPT-2(15亿参数)和GPT-3(1750亿参数)之间，OpenAI做了一项系统性的实验：训练从1亿到150亿参数不等的多个模型，记录每个模型的最终loss。

结果令人震惊：**loss与参数量之间呈现完美的幂律关系**。

Kaplan et al. (2020) "Scaling Laws for Neural Language Models" 的核心发现：
- 模型参数量 $N$ 增大 → loss 按 $L(N) \approx a \cdot N^{-b}$ 下降
- 训练数据量 $D$ 增大 → loss 按 $L(D) \approx c \cdot D^{-d}$ 下降
- 计算量 $C$ 增大 → loss 按 $L(C) \approx e \cdot C^{-f}$ 下降

**这意味着**：只要持续增大模型和数据，性能就会持续提升——没有饱和的迹象！

---

## 2. 生活隐喻：规模法则的直觉

- **规模法则** = 城市规模效应：城市越大，人均创新产出越高（超线性缩放）
- **幂律下降** = 练习效应：练习越多进步越快，但边际收益递减
- **没有饱和** = 知识无上限：学得越多，能学到的还越多

关键洞察：
- 模型规模每增加10倍，loss下降一个固定比例
- 这意味着**投资更大的模型是值得的**——回报是可预测的
- 也意味着**小模型有根本性的上限**——再怎么优化也追不上大模型

---

## 3. 数学直觉：规模法则公式

### 3.1 参数量缩放

$$L(N) = \frac{a}{N^{\alpha}} + L_{\infty}$$

其中：
- $N$ = 模型参数量
- $L$ = 交叉熵loss
- $a, \alpha$ = 拟合常数（Kaplan发现 $\alpha \approx 0.076$）
- $L_{\infty}$ = 不可约loss（数据本身的熵，无法消除）

### 3.2 数据量缩放

$$L(D) = \frac{c}{D^{\beta}} + L_{\infty}$$

- $D$ = 训练token数
- $\beta \approx 0.095$

### 3.3 最优计算分配

给定计算预算 $C$，最优的参数量和数据量满足：

$$N \propto C^{0.73}, \quad D \propto C^{0.27}$$

**结论**：计算量增加时，应该把更多预算花在增大模型上，而非增加数据。

---

## 4. 代码实验室：规模法则示意图

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

### 4.1 用模拟数据展示幂律关系

```python
# 模拟不同规模模型的loss
# 真实实验中需要训练多个不同大小的模型
# 这里用幂律公式直接生成模拟数据

# 规模法则参数（模拟Kaplan等人的发现）
a = 2.0       # 缩放系数
alpha = 0.076  # 幂律指数
L_inf = 1.5   # 不可约loss（数据本身的熵）

# 模型参数量（从1M到100B）
params = np.logspace(6, 11, 50)  # 1M到100B

# 理论幂律曲线
loss_theory = a * params**(-alpha) + L_inf

# 模拟实测点（加噪声）
np.random.seed(42)
params_measured = np.logspace(6, 11, 15)
loss_measured = a * params_measured**(-alpha) + L_inf
loss_measured += np.random.normal(0, 0.02, len(params_measured))

# 可视化
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(params_measured, loss_measured, color='#e74c3c',
           s=60, zorder=5, label='实测点')
ax.plot(params, loss_theory, color='#3498db', linewidth=2,
        alpha=0.8, label=f'幂律拟合: L = {a}·N^(-{alpha}) + {L_inf}')
ax.set_xscale('log')
ax.set_xlabel('模型参数量 N', fontsize=12)
ax.set_ylabel('交叉熵 Loss', fontsize=12)
ax.set_title('规模法则：模型越大，Loss越低', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 标注关键模型
key_models = {'GPT-1\n(117M)': 117e6, 'GPT-2\n(1.5B)': 1.5e9,
              'GPT-3\n(175B)': 175e9}
for name, p in key_models.items():
    loss_p = a * p**(-alpha) + L_inf
    ax.annotate(name, xy=(p, loss_p), fontsize=9,
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.show()
```

### 4.2 三种缩放的关系

```python
# 参数量、数据量、计算量三者的缩放关系
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 参数量缩放
N = np.logspace(6, 11, 50)
L_N = 2.0 * N**(-0.076) + 1.5
axes[0].plot(N, L_N, color='#3498db', linewidth=2)
axes[0].set_xscale('log')
axes[0].set_xlabel('参数量 N', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('L(N) = a·N^(-α) + L∞', fontsize=13)
axes[0].grid(True, alpha=0.3)

# 数据量缩放
D = np.logspace(8, 13, 50)  # 100M到10T tokens
L_D = 5.0 * D**(-0.095) + 1.5
axes[1].plot(D, L_D, color='#27ae60', linewidth=2)
axes[1].set_xscale('log')
axes[1].set_xlabel('数据量 D (tokens)', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('L(D) = c·D^(-β) + L∞', fontsize=13)
axes[1].grid(True, alpha=0.3)

# 计算量缩放
C = np.logspace(15, 22, 50)  # FLOPs
L_C = 8.0 * C**(-0.05) + 1.5
axes[2].plot(C, L_C, color='#9b59b6', linewidth=2)
axes[2].set_xscale('log')
axes[2].set_xlabel('计算量 C (FLOPs)', fontsize=12)
axes[2].set_ylabel('Loss', fontsize=12)
axes[2].set_title('L(C) = e·C^(-γ) + L∞', fontsize=13)
axes[2].grid(True, alpha=0.3)

plt.suptitle('规模法则：三种缩放维度', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
```

---

## 今日结语

规模法则揭示了深度学习最深刻的规律之一：模型性能与规模之间遵循精确的幂律关系。参数量、数据量、计算量每增加一个数量级，loss就按固定比例下降。这意味着投资更大的模型是可预测回报的，也解释了为什么GPT-3、GPT-4越来越大。

明天，我们将探索规模法则的一个惊人推论：涌现能力——当模型大到某个阈值时，突然出现小模型完全没有的新能力！

---

### 翻译词典

| 生活中的直觉 | 深度学习术语 |
|---|---|
| 城市越大越创新 | 规模法则：模型越大越强 |
| 练习越多进步越快但递减 | 幂律下降：边际收益递减 |
| 知识无上限 | 不可约loss以下无法突破 |
| 投资大模型回报可预测 | 缩放律指导资源分配 |
| 把预算花在增模型而非增数据 | N∝C^0.73, D∝C^0.27 |
