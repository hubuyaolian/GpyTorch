# Day 07：规模法则实验 —— 亲手拟合"越大越强"的幂律

> 📊 第十四周 · 预训练与规模法则 · 第 7 天

规模法则说 $L(N) = a \cdot N^{-b} + c$，但这是OpenAI用成千上万块GPU跑出来的。我们没有那么多算力，但我们可以**用小模型复现同样的规律**——5个不同d_model的MiniGPT，5条训练曲线，亲手拟合幂律。你会发现：即使模型小到只有几百个参数，幂律依然成立。

**今天的任务**：
1. 训练5个不同d_model（16/32/64/128/256）的MiniGPT
2. 记录每个规模的最终loss
3. 手写拟合幂律 $L(N)=a \cdot N^{-b}+c$，可视化log轴散点+拟合曲线

---

## 1. 历史剧场：2020，用钱烧出来的幂律

2020年，Kaplan et al. 在"Scaling Laws for Neural Language Models"中做了一项前所未有的实验：

他们训练了**从1亿到150亿参数**不等的多个GPT模型，每个模型用不同量的数据训练到收敛。总共消耗了数千GPU天的算力。

结果：loss与参数量之间呈现**近乎完美的幂律关系**。在log-log图上，数据点几乎排成一条直线。

更惊人的发现：
- 幂律指数 $\alpha \approx 0.076$——参数量增加10倍，loss下降约15%
- **没有饱和迹象**——至少在GPT-3的规模（175B参数）上，幂律依然成立
- 这意味着：**继续增大模型，性能还会继续提升**

这就是为什么GPT-4比GPT-3大，GPT-5会比GPT-4更大——规模法则给了可预测的回报。

---

## 2. 生活隐喻：规模效应无处不在

- **幂律** = 城市规模效应：城市人口每增加1倍，人均创新产出增加约1.15倍（Geoffrey West的发现）
- **边际递减** = 练习效应：从0分到60分容易，从90分到95分难，但进步永远在
- **没有天花板** = 知识无上限：学得越多，能学到的还越多，只是越来越慢

关键洞察：
- 幂律的指数 $b$ 决定了"增大模型的性价比"
- $b$ 越大，增大模型收益越高
- $b$ 越小，增大模型收益越低——但**永远不会到零**

---

## 3. 数学直觉：幂律拟合

### 3.1 规模法则公式

$$L(N) = a \cdot N^{-b} + c$$

其中：
- $N$ = 模型参数量
- $a$ = 缩放系数
- $b$ = 幂律指数（Kaplan发现 $b \approx 0.076$）
- $c$ = 不可约loss（数据本身的熵，无法消除）

### 3.2 对数空间中的线性关系

两边取对数（忽略 $c$）：

$$\log(L - c) = \log(a) - b \cdot \log(N)$$

在log-log空间中，幂律变成**线性关系**！斜率就是 $-b$。

### 3.3 拟合方法

给定数据点 $(N_i, L_i)$，用最小二乘法拟合 $a, b, c$：

$$\min_{a,b,c} \sum_i (L_i - a \cdot N_i^{-b} - c)^2$$

这是一个非线性优化问题，可以用 `scipy.optimize.curve_fit` 求解。

---

## 4. 代码实验室：规模法则实验

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

### 4.1 复用MiniGPT架构（精简版）

```python
class MiniGPT(nn.Module):
    """精简版MiniGPT，用于规模法则实验.
    
    通过d_model控制模型规模，其余超参数按比例缩放。
    """
    def __init__(self, vocab_size=50, d_model=32, n_heads=4,
                 n_layers=2, max_seq_len=16):
        super().__init__()
        d_ff = d_model * 4  # FFN维度 = 4倍d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        # 简化：用单头注意力代替多头（d_model小时n_heads=1）
        actual_heads = min(n_heads, d_model // 4) if d_model >= 8 else 1
        self.blocks = nn.ModuleList([
            self._make_block(d_model, actual_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _make_block(self, d_model, n_heads, d_ff):
        """构建Transformer块（内联定义，减少代码量）."""
        return nn.ModuleDict({
            'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
            'ln1': nn.LayerNorm(d_model),
            'ff': nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                               nn.Linear(d_ff, d_model)),
            'ln2': nn.LayerNorm(d_model)
        })

    def forward(self, x):
        """前向传播."""
        B, S = x.shape
        h = self.tok_emb(x) + self.pos_emb(torch.arange(S, device=x.device))
        for block in self.blocks:
            # 自注意力 + 残差
            h_norm = block['ln1'](h)
            attn_out, _ = block['attn'](h_norm, h_norm, h_norm,
                                        is_causal=True)
            h = h + attn_out
            h = h + block['ff'](block['ln2'](h))  # FFN + 残差
        return self.lm_head(self.ln_f(h))
```

### 4.2 生成训练语料

```python
def make_data(vocab_size=50, seq_len=16, n=3000, seed=42):
    """生成带规律的语料：token[i+1] = (token[i]*2 + 1) % vocab_size."""
    torch.manual_seed(seed)
    seqs = []
    for _ in range(n):
        start = torch.randint(0, vocab_size, (1,)).item()
        seq = [start]
        for _ in range(seq_len - 1):
            seq.append((seq[-1] * 2 + 1) % vocab_size)
        seqs.append(seq)
    return torch.tensor(seqs, dtype=torch.long)


def count_params(model):
    """计算模型可训练参数量."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

### 4.3 训练不同规模的模型

```python
def train_model(model, data, epochs=40, lr=1e-3, batch_size=64):
    """训练模型并返回每轮loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    N = data.shape[0]
    for epoch in range(epochs):
        total_loss, n_b = 0.0, 0
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            batch = data[perm[i:i+batch_size]]
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
        losses.append(total_loss / n_b)
    return losses


# 5个不同d_model的规模
d_models = [16, 32, 64, 128, 256]
VOCAB = 50
SEQ_LEN = 16
data = make_data(VOCAB, SEQ_LEN, n=3000)

results = []  # 存储 (参数量, 最终loss, loss曲线)
for d_model in d_models:
    torch.manual_seed(42)
    model = MiniGPT(vocab_size=VOCAB, d_model=d_model,
                    n_layers=2, max_seq_len=SEQ_LEN)
    n_params = count_params(model)
    losses = train_model(model, data, epochs=50, lr=1e-3)
    final_loss = losses[-1]
    results.append((n_params, final_loss, losses))
    print(f"d_model={d_model:3d} | 参数量={n_params:>6d} | 最终loss={final_loss:.4f}")
```

### 4.4 手写幂律拟合

```python
def power_law(N, a, b, c):
    """幂律函数: L(N) = a * N^(-b) + c."""
    return a * np.power(N, -b) + c

# 提取数据点
params_arr = np.array([r[0] for r in results], dtype=float)
loss_arr = np.array([r[1] for r in results])

# 用scipy拟合幂律参数
popt, pcov = curve_fit(power_law, params_arr, loss_arr,
                       p0=[10.0, 0.1, 1.0],  # 初始猜测
                       maxfev=10000)
a_fit, b_fit, c_fit = popt
print(f"\n拟合结果: L(N) = {a_fit:.4f} · N^(-{b_fit:.4f}) + {c_fit:.4f}")
print(f"幂律指数 b = {b_fit:.4f}")
print(f"不可约loss c = {c_fit:.4f}")
```

### 4.5 可视化：log轴散点 + 幂律拟合曲线

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：训练曲线对比
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#27ae60', '#3498db']
for i, (n_p, f_l, losses) in enumerate(results):
    axes[0].plot(losses, color=colors[i], linewidth=2,
                label=f'd_model={d_models[i]} ({n_p}参数)', alpha=0.8)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('不同规模模型的训练曲线', fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# 右图：规模法则——log轴散点 + 拟合曲线
N_smooth = np.logspace(np.log10(params_arr.min() * 0.5),
                       np.log10(params_arr.max() * 2), 100)
L_smooth = power_law(N_smooth, a_fit, b_fit, c_fit)

axes[1].scatter(params_arr, loss_arr, color='#e74c3c', s=100,
               zorder=5, label='实测点', edgecolors='black', linewidth=1)
axes[1].plot(N_smooth, L_smooth, color='#3498db', linewidth=2,
           alpha=0.8, label=f'幂律拟合: L={a_fit:.2f}·N$^{{-{b_fit:.3f}}}$+{c_fit:.2f}')
axes[1].set_xscale('log')
axes[1].set_xlabel('参数量 N', fontsize=12)
axes[1].set_ylabel('最终 Loss', fontsize=12)
axes[1].set_title('规模法则：模型越大，Loss越低', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle('规模法则实验：亲手验证幂律', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
```

---

## 今日结语

今天我们亲手验证了规模法则：训练了5个从d_model=16到d_model=256的MiniGPT，记录了每个规模的最终loss，并用 `curve_fit` 拟合出幂律 $L(N) = a \cdot N^{-b} + c$。即使在几百到几万参数的小模型上，**幂律依然成立**——模型越大，loss越低，而且下降趋势可预测。

这就是OpenAI持续增大模型的数学依据：规模法则保证了投资回报的可预测性。

明天，我们将探索规模法则的惊人推论：涌现能力——当模型大到某个阈值，突然出现小模型完全没有的新能力！

---

### 翻译词典

| 生活中的直觉 | 深度学习术语 |
|---|---|
| 城市越大越创新 | 规模法则：模型越大loss越低 |
| 练习越多进步越慢但不停 | 幂律下降：边际递减但无上限 |
| log-log图上的直线 | 幂律在对数空间中是线性的 |
| 不可消除的噪声 | 不可约loss c（数据本身的熵） |
| 投资大模型回报可预测 | 幂律指数b决定性价比 |
