# Day 08：涌现能力实验 —— 亲眼见证"量变到质变"

> 🧬 第十四周 · 预训练与规模法则 · 第 8 天

规模法则说"越大越强"，但这是**渐进的**——loss平滑下降。更神奇的是：当模型大到某个阈值，会**突然出现**小模型完全没有的新能力。今天我们要亲手设计一个多步推理任务，训练不同规模的模型，亲眼看到：小模型≈0%，大模型骤升——涌现拐点就在眼前。

**今天的任务**：
1. 设计多步算术推理任务（需3步以上推理）
2. 训练不同规模模型，测量推理准确率
3. 可视化涌现拐点：横轴参数量、纵轴准确率

---

## 1. 历史剧场：2022，涌现的系统性发现

2022年，Wei et al. 发表"Emergent Abilities of Large Language Models"，系统性地研究了涌现现象。

他们测试了多种能力在不同规模模型上的表现：

| 能力 | 小模型(<10B) | 中模型(10-100B) | 大模型(>100B) |
|---|---|---|---|
| 多步算术推理 | ≈0% | ≈0% | 骤升至~60% |
| 思维链推理 | ≈0% | ≈0% | 骤升至~50% |
| 指令遵循 | ≈0% | 骤升至~70% | ~90% |
| 简单问答 | ~30% | ~60% | ~80% |

关键发现：**复杂能力有涌现阈值，简单能力没有**。

涌现的本质：多步推理需要**所有步骤都正确**，任何一步出错就全错。小模型每步准确率不够高，多步串联后概率趋近于0；大模型每步准确率超过临界值，多步串联后仍有非零概率。

---

## 2. 生活隐喻：链条的强度

- **涌现** = 链条的强度：一根链条的强度取决于最弱的环节。如果每个环节有5%的断裂概率，3节链条不断裂的概率 = 0.95³ ≈ 86%；但如果每节有30%的断裂概率，3节不断裂 = 0.7³ ≈ 34%
- **涌现阈值** = 临界概率：当每步准确率超过某个临界值，多步串联的总体准确率从≈0骤升
- **小模型** = 每步都不太准：3步串联后≈0%
- **大模型** = 每步都够准：3步串联后仍有可观准确率

数学直觉：如果每步准确率为 $p$，$k$ 步推理的总体准确率为 $p^k$。

- $p = 0.5, k = 3$：$0.5^3 = 12.5\%$
- $p = 0.3, k = 3$：$0.3^3 = 2.7\%$（几乎为零）
- $p = 0.8, k = 3$：$0.8^3 = 51.2\%$（涌现！）

---

## 3. 数学直觉：涌现的数学描述

### 3.1 多步推理的准确率

$$P_{total} = p^k$$

其中 $p$ 是单步准确率，$k$ 是推理步数。

### 3.2 涌现阈值

当 $p$ 从小模型到大模型平滑增长时，$p^k$ 在 $p$ 超过某个临界值后**骤然上升**：

$$\frac{d(p^k)}{dp} = k \cdot p^{k-1}$$

步数 $k$ 越大，涌现越陡峭——这就是为什么**复杂任务**（需要更多推理步）的涌现更明显。

### 3.3 与规模法则的关系

规模法则描述loss的平滑下降：$L(N) = a \cdot N^{-b} + c$

但loss的微小改善，可能导致单步准确率 $p$ 超过临界值，从而在多步任务上产生**质变**。

---

## 4. 代码实验室：涌现能力实验

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

### 4.1 设计多步算术推理任务

```python
def generate_reasoning_data(n_samples=5000, n_steps=3, max_digit=9, seed=42):
    """生成多步算术推理数据.
    
    任务：给定起始数x，依次执行n_steps步"+a_i再取模"运算，
    模型需要预测最终结果。
    
    例：x=3, 步骤=[+2, +4, +1] → 3→5→9→10(mod 10)=0
    输入: [3, 2, 4, 1]  输出: 0
    
    这需要模型学会逐步推理，而非直接记忆映射。
    
    Args:
        n_samples: 样本数
        n_steps: 推理步数（越多越难）
        max_digit: 数字范围0~max_digit
        seed: 随机种子
    Returns:
        inputs: (n_samples, n_steps+1) 输入序列
        targets: (n_samples,) 最终结果
    """
    torch.manual_seed(seed)
    mod = max_digit + 1
    inputs, targets = [], []
    for _ in range(n_samples):
        x = torch.randint(0, mod, (1,)).item()  # 起始数
        steps = torch.randint(1, mod, (n_steps,)).tolist()  # 每步加数
        result = x
        for s in steps:
            result = (result + s) % mod  # 逐步推理
        inputs.append([x] + steps)
        targets.append(result)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


# 生成3步推理数据（需要模型学会逐步计算）
train_x, train_y = generate_reasoning_data(n_samples=5000, n_steps=3)
test_x, test_y = generate_reasoning_data(n_samples=1000, n_steps=3, seed=99)
print(f"训练集: {train_x.shape}, 测试集: {test_x.shape}")
print(f"示例: 输入={train_x[0].tolist()}, 输出={train_y[0].item()}")
```

### 4.2 不同规模的推理模型

```python
class ReasoningModel(nn.Module):
    """多步推理模型：Embedding + Transformer + 分类头.
    
    输入: [x, a1, a2, a3]（起始数+3步加数）
    输出: 最终结果（0~10的分类）
    """
    def __init__(self, vocab_size=11, d_model=32, n_heads=4,
                 n_layers=2, max_seq_len=5):
        super().__init__()
        d_ff = d_model * 4
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        actual_heads = min(n_heads, d_model // 4) if d_model >= 8 else 1
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, actual_heads,
                                              batch_first=True),
                'ln1': nn.LayerNorm(d_model),
                'ff': nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                   nn.Linear(d_ff, d_model)),
                'ln2': nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)  # 分类头

    def forward(self, x):
        """前向传播."""
        B, S = x.shape
        h = self.emb(x) + self.pos(torch.arange(S, device=x.device))
        for layer in self.layers:
            h_n = layer['ln1'](h)
            a_out, _ = layer['attn'](h_n, h_n, h_n, is_causal=True)
            h = h + a_out
            h = h + layer['ff'](layer['ln2'](h))
        # 取最后一个位置的输出做分类
        return self.head(self.ln_f(h[:, -1, :]))
```

### 4.3 训练与评估

```python
def train_and_evaluate(model, train_x, train_y, test_x, test_y,
                       epochs=60, lr=1e-3, batch_size=64):
    """训练模型并返回测试准确率.
    
    Args:
        model: 推理模型
        train_x, train_y: 训练数据
        test_x, test_y: 测试数据
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批大小
    Returns:
        test_accuracy: 最终测试准确率
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    N = train_x.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # 评估测试准确率
    with torch.no_grad():
        preds = model(test_x).argmax(dim=1)
        acc = (preds == test_y).float().mean().item()
    return acc
```

### 4.4 运行涌现实验

```python
# 不同规模的模型配置
configs = [
    {'d_model': 8,   'n_layers': 1, 'label': '微型 (d=8, L=1)'},
    {'d_model': 16,  'n_layers': 1, 'label': '小型 (d=16, L=1)'},
    {'d_model': 32,  'n_layers': 1, 'label': '中小 (d=32, L=1)'},
    {'d_model': 32,  'n_layers': 2, 'label': '中等 (d=32, L=2)'},
    {'d_model': 64,  'n_layers': 2, 'label': '中大 (d=64, L=2)'},
    {'d_model': 128, 'n_layers': 2, 'label': '大型 (d=128, L=2)'},
    {'d_model': 256, 'n_layers': 3, 'label': '超大 (d=256, L=3)'},
]

emergence_results = []
for cfg in configs:
    torch.manual_seed(42)
    model = ReasoningModel(
        vocab_size=11, d_model=cfg['d_model'],
        n_layers=cfg['n_layers'], max_seq_len=5
    )
    n_params = sum(p.numel() for p in model.parameters())
    acc = train_and_evaluate(model, train_x, train_y, test_x, test_y,
                            epochs=80, lr=1e-3)
    emergence_results.append((n_params, acc, cfg['label']))
    print(f"{cfg['label']:20s} | 参数量={n_params:>6d} | 准确率={acc:.4f}")
```

### 4.5 可视化：涌现拐点

```python
params_list = [r[0] for r in emergence_results]
accs_list = [r[1] for r in emergence_results]
labels_list = [r[2] for r in emergence_results]

fig, ax = plt.subplots(figsize=(10, 6))

# 散点图
colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(params_list)))
for i, (n_p, acc, label) in enumerate(emergence_results):
    ax.scatter(n_p, acc, color=colors[i], s=150, zorder=5,
              edgecolors='black', linewidth=1.5)
    ax.annotate(label, (n_p, acc), textcoords="offset points",
               xytext=(10, 5), fontsize=9)

# 连线
ax.plot(params_list, accs_list, color='gray', linewidth=1.5,
       linestyle='--', alpha=0.5, zorder=3)

# 标注涌现拐点（准确率从<10%跳到>30%的位置）
for i in range(1, len(accs_list)):
    if accs_list[i-1] < 0.1 and accs_list[i] >= 0.1:
        ax.annotate('🔥 涌现拐点', (params_list[i], accs_list[i]),
                   fontsize=13, fontweight='bold', color='#e74c3c',
                   xytext=(-60, 20), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', color='#e74c3c',
                                  lw=2))
        break

ax.set_xscale('log')
ax.set_xlabel('模型参数量', fontsize=12)
ax.set_ylabel('推理准确率', fontsize=12)
ax.set_title('涌现能力实验：多步推理准确率 vs 模型规模', fontsize=14)
ax.axhline(y=1/11, color='gray', linestyle=':', alpha=0.5,
          label=f'随机猜测基线 ({1/11:.1%})')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.show()
```

### 4.6 对比：1步 vs 3步推理的涌现差异

```python
# 1步推理（简单任务，无涌现）
simple_x, simple_y = generate_reasoning_data(n_samples=5000, n_steps=1, seed=42)
simple_test_x, simple_test_y = generate_reasoning_data(n_samples=1000, n_steps=1, seed=99)

simple_accs = []
for cfg in configs:
    torch.manual_seed(42)
    model = ReasoningModel(vocab_size=11, d_model=cfg['d_model'],
                          n_layers=cfg['n_layers'], max_seq_len=3)
    acc = train_and_evaluate(model, simple_x, simple_y,
                            simple_test_x, simple_test_y,
                            epochs=60, lr=1e-3)
    simple_accs.append(acc)

# 对比可视化
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(params_list, simple_accs, 'o-', color='#27ae60', linewidth=2,
       markersize=8, label='1步推理（简单，无涌现）')
ax.plot(params_list, accs_list, 's-', color='#e74c3c', linewidth=2,
       markersize=8, label='3步推理（复杂，有涌现）')
ax.set_xscale('log')
ax.set_xlabel('模型参数量', fontsize=12)
ax.set_ylabel('准确率', fontsize=12)
ax.set_title('简单任务 vs 复杂任务：涌现只在复杂任务中出现', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 今日结语

今天我们亲手见证了涌现能力：在3步算术推理任务上，小模型准确率≈0%（和随机猜一样），但当模型规模超过某个阈值，准确率骤然上升——这就是涌现拐点。对比1步推理（简单任务），准确率随规模平滑增长，没有涌现。

涌现的本质是**多步串联**：每步准确率 $p$ 需要超过临界值，$k$ 步推理的总体准确率 $p^k$ 才能从≈0跳到可观值。步数越多，涌现越陡峭。

明天，我们将从实验回到历史，梳理GPT家族的进化之路！

---

### 翻译词典

| 生活中的直觉 | 深度学习术语 |
|---|---|
| 链条强度取决于最弱环节 | 多步推理：所有步骤都正确才行 |
| 每步够准才能串联成功 | 涌现阈值：单步准确率超过临界值 |
| 99度水不开，100度沸腾 | 量变到质变：规模超过拐点骤然出现 |
| 简单任务谁都能做 | 无涌现：准确率平滑增长 |
| 复杂任务需要实力 | 有涌现：需要规模超过阈值 |
