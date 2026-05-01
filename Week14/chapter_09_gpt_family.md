# Day 09：GPT家族进化史 —— 从117M到1.8T的五年征途

> 🏛️ 第十四周 · 预训练与规模法则 · 第 9 天

从GPT-1的1.17亿参数到GPT-4的1.8万亿参数，五年间模型规模增长了15000倍。但GPT家族的进化不只是"堆参数"——每一代都有**核心突破**。今天我们梳理这条进化之路，理解"为什么GPT越来越强"不只是因为大，更因为每一代都解决了上一代的关键问题。

**今天的任务**：
1. 梳理GPT-1→GPT-2→GPT-3→InstructGPT→GPT-4的进化脉络
2. 对比各代参数量、训练数据、核心突破
3. 用表格+时间线图展示进化历程

---

## 1. 历史剧场：GPT家族的五代传奇

### GPT-1（2018年6月）—— "预训练+微调"的开创者

OpenAI发布"Improving Language Understanding by Generative Pre-Training"。

- **参数量**：1.17亿（117M）
- **训练数据**：BookCorpus（约5GB，7000本书）
- **架构**：12层Transformer Decoder，768维
- **核心突破**：证明了"预训练+微调"范式的有效性——在12项NLP任务中9项达到SOTA
- **局限**：每个下游任务都需要单独微调，不够通用

### GPT-2（2019年2月）—— "零样本学习"的先驱

OpenAI发布"Language Models are Unsupervised Multilingual Learners"。

- **参数量**：15亿（1.5B）
- **训练数据**：WebText（约40GB，800万网页）
- **架构**：48层Transformer Decoder，1600维
- **核心突破**：**不做微调**，直接用预训练模型做零样本（zero-shot）推理——"语言模型足够大，就能直接理解任务"
- **局限**：零样本效果还不够好，经常"答非所问"

### GPT-3（2020年6月）—— "少样本学习"的里程碑

OpenAI发布"Language Models are Few-Shot Learners"。

- **参数量**：1750亿（175B）
- **训练数据**：Common Crawl等混合语料（约570GB，3000亿token）
- **架构**：96层Transformer Decoder，12288维
- **核心突破**：**少样本（few-shot）学习**——只需在prompt中给几个例子，模型就能学会新任务，无需微调
- **局限**：可能生成有害内容、不符合人类意图

### InstructGPT（2022年3月）—— "对齐人类意图"的转折点

OpenAI发布"Training language models to follow instructions with human feedback"。

- **参数量**：175B（与GPT-3同规模）
- **核心突破**：**RLHF**（基于人类反馈的强化学习）——让模型学会"听指令"和"符合人类价值观"
- **三步流程**：SFT（监督微调）→ RM（训练奖励模型）→ PPO（强化学习优化）
- **意义**：同样175B参数，InstructGPT的1.3B小模型在人类评估中都能胜过GPT-3 175B——**对齐比规模更重要**

### GPT-4（2023年3月）—— "多模态+对齐"的集大成者

OpenAI发布GPT-4 Technical Report。

- **参数量**：约1.8万亿（1.8T，MoE架构，8个专家各约220B）
- **训练数据**：约13万亿token（多模态：文本+图像）
- **核心突破**：**多模态理解**（能看图）+ **更强的对齐**（更安全、更听话）+ **涌现推理能力**（通过BAR考试、医学考试等）
- **意义**：GPT-4不只是更大，而是**更对齐、更多模态、更会推理**

---

## 2. 生活隐喻：GPT家族的进化就像人类文明

- **GPT-1** = 文艺复兴：发现了"预训练"这个新范式，但每个任务还要单独微调（每个城邦各自为政）
- **GPT-2** = 启蒙运动：提出"零样本"——模型应该能直接理解任务，不需要特殊训练（理性万能论）
- **GPT-3** = 工业革命：规模爆发，少样本学习让模型成为"通用工具"（大规模生产）
- **InstructGPT** = 法治社会：光强大不够，还要"听话"和"安全"——RLHF就是给AI立规矩
- **GPT-4** = 现代文明：多模态（能看能读）、对齐（安全可控）、推理（深度思考）——全面进化

---

## 3. 数学直觉：GPT进化的核心公式

### 3.1 GPT-1/2/3：规模法则驱动

$$\text{Performance} \propto N^{\alpha} \cdot D^{\beta} \cdot C^{\gamma}$$

GPT-1→2→3的进化主要是**增大 $N, D, C$**，沿着规模法则曲线前进。

### 3.2 InstructGPT：对齐效率

$$\text{Usefulness} = \text{Capability} \times \text{Alignment}$$

InstructGPT的发现：**对齐(Alignment)可以弥补规模的不足**。1.3B的InstructGPT在人类评估中胜过175B的GPT-3。

### 3.3 GPT-4：规模+对齐+多模态

$$\text{GPT-4} = \text{Scale}(1.8\text{T}) + \text{RLHF} + \text{Multimodal} + \text{MoE}$$

GPT-4的进化不是单一维度的，而是**规模、对齐、多模态、架构**四维同时提升。

---

## 4. 代码实验室：GPT家族进化可视化

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

### 4.1 GPT家族对比表格

```python
# GPT家族数据
gpt_family = [
    {'name': 'GPT-1', 'year': 2018, 'params': 0.117e9,
     'data': '5GB', 'tokens': '5B', 'breakthrough': '预训练+微调范式',
     'color': '#3498db'},
    {'name': 'GPT-2', 'year': 2019, 'params': 1.5e9,
     'data': '40GB', 'tokens': '40B', 'breakthrough': '零样本学习',
     'color': '#27ae60'},
    {'name': 'GPT-3', 'year': 2020, 'params': 175e9,
     'data': '570GB', 'tokens': '300B', 'breakthrough': '少样本学习',
     'color': '#f39c12'},
    {'name': 'InstructGPT', 'year': 2022, 'params': 175e9,
     'data': '570GB', 'tokens': '300B', 'breakthrough': 'RLHF对齐',
     'color': '#e74c3c'},
    {'name': 'GPT-4', 'year': 2023, 'params': 1.8e12,
     'data': '13T tokens', 'tokens': '13T', 'breakthrough': '多模态+MoE',
     'color': '#9b59b6'},
]

# 打印对比表
print(f"{'模型':<14} {'年份':<6} {'参数量':<14} {'训练数据':<12} {'核心突破'}")
print("-" * 70)
for g in gpt_family:
    p = g['params']
    if p >= 1e12:
        p_str = f"{p/1e12:.1f}T"
    elif p >= 1e9:
        p_str = f"{p/1e9:.1f}B"
    else:
        p_str = f"{p/1e6:.0f}M"
    print(f"{g['name']:<14} {g['year']:<6} {p_str:<14} {g['data']:<12} {g['breakthrough']}")
```

### 4.2 时间线图：GPT家族进化

```python
fig, ax = plt.subplots(figsize=(14, 6))

years = [g['year'] for g in gpt_family]
params = [g['params'] for g in gpt_family]
names = [g['name'] for g in gpt_family]
colors = [g['color'] for g in gpt_family]
breakthroughs = [g['breakthrough'] for g in gpt_family]

# 绘制时间线
ax.plot(years, [1]*len(years), color='gray', linewidth=3, alpha=0.3, zorder=1)

# 绘制每个节点
for i, g in enumerate(gpt_family):
    # 圆点大小与参数量的log成正比
    size = 200 + 100 * np.log10(g['params'] / 1e6)
    ax.scatter(g['year'], 1, color=g['color'], s=size, zorder=5,
              edgecolors='black', linewidth=1.5)
    # 模型名称
    ax.annotate(g['name'], (g['year'], 1),
               xytext=(0, -35), textcoords='offset points',
               fontsize=12, fontweight='bold', ha='center',
               color=g['color'])
    # 参数量
    p = g['params']
    p_str = f"{p/1e12:.1f}T" if p >= 1e12 else (f"{p/1e9:.1f}B" if p >= 1e9 else f"{p/1e6:.0f}M")
    ax.annotate(p_str, (g['year'], 1),
               xytext=(0, -55), textcoords='offset points',
               fontsize=10, ha='center', color='gray')
    # 核心突破
    ax.annotate(g['breakthrough'], (g['year'], 1),
               xytext=(0, 25), textcoords='offset points',
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=g['color'],
                        alpha=0.2, edgecolor=g['color']))

ax.set_xlim(2017.5, 2023.5)
ax.set_ylim(0.5, 1.5)
ax.set_xlabel('年份', fontsize=12)
ax.set_title('GPT家族进化史：从117M到1.8T的五年征途', fontsize=15)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.show()
```

### 4.3 参数量增长可视化

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：参数量增长（线性轴，GPT-4一骑绝尘）
for i, g in enumerate(gpt_family):
    axes[0].bar(i, g['params'], color=g['color'], edgecolor='black',
               linewidth=1, alpha=0.8)
    p = g['params']
    p_str = f"{p/1e12:.1f}T" if p >= 1e12 else (f"{p/1e9:.1f}B" if p >= 1e9 else f"{p/1e6:.0f}M")
    axes[0].text(i, g['params'], p_str, ha='center', va='bottom',
                fontsize=10, fontweight='bold')
axes[0].set_xticks(range(len(gpt_family)))
axes[0].set_xticklabels([g['name'] for g in gpt_family], fontsize=10)
axes[0].set_ylabel('参数量', fontsize=12)
axes[0].set_title('参数量增长（线性轴）', fontsize=13)

# 右图：参数量增长（log轴，看清每代增长）
log_params = [np.log10(g['params']) for g in gpt_family]
for i, g in enumerate(gpt_family):
    axes[1].bar(i, log_params[i], color=g['color'], edgecolor='black',
               linewidth=1, alpha=0.8)
    axes[1].text(i, log_params[i], f"10^{log_params[i]:.1f}", ha='center',
                va='bottom', fontsize=10, fontweight='bold')
axes[1].set_xticks(range(len(gpt_family)))
axes[1].set_xticklabels([g['name'] for g in gpt_family], fontsize=10)
axes[1].set_ylabel('log₁₀(参数量)', fontsize=12)
axes[1].set_title('参数量增长（对数轴）', fontsize=13)

plt.suptitle('GPT家族：参数量的指数增长', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
```

### 4.4 核心突破的演进逻辑

```python
# 用少量代码展示模型规模差异对推理的影响
import torch
import torch.nn as nn

def show_capacity_gap(small_dim=16, large_dim=256):
    """展示不同规模模型的容量差异."""
    small = nn.Linear(small_dim, small_dim)
    large = nn.Linear(large_dim, large_dim)
    n_small = sum(p.numel() for p in small.parameters())
    n_large = sum(p.numel() for p in large.parameters())
    print(f"小模型 (d={small_dim}): {n_small} 参数")
    print(f"大模型 (d={large_dim}): {n_large} 参数")
    print(f"规模比: {n_large/n_small:.0f}x")
    print(f"这就是GPT-1→GPT-4规模增长的缩影：")
    print(f"  GPT-1: 117M → GPT-4: 1.8T = {1.8e12/0.117e9:.0f}x")

show_capacity_gap()
```

---

## 今日结语

GPT家族的五年进化史，是一部"规模+创新"的双轮驱动史：

- **GPT-1**开创了预训练-微调范式
- **GPT-2**提出了零样本学习的愿景
- **GPT-3**用175B参数实现了少样本学习的突破
- **InstructGPT**发现对齐比规模更重要——RLHF让1.3B模型胜过175B
- **GPT-4**集大成：多模态+MoE+对齐，参数量1.8T

进化的核心逻辑：**每一代都解决了上一代的关键问题**。GPT-1的微调麻烦→GPT-2的零样本；GPT-2效果不够→GPT-3的少样本；GPT-3不听话→InstructGPT的RLHF；InstructGPT只能看文本→GPT-4的多模态。

明天，我们将总结这一阶段的核心概念，并引出下一个关键问题：预训练模型虽然强大，但如何确保它安全、可控？

---

### 翻译词典

| 生活中的直觉 | 深度学习术语 |
|---|---|
| 文艺复兴发现新范式 | GPT-1：预训练+微调 |
| 启蒙运动的理性万能 | GPT-2：零样本学习 |
| 工业革命的大规模生产 | GPT-3：少样本+175B参数 |
| 法治社会给权力立规矩 | InstructGPT：RLHF对齐 |
| 现代文明全面发展 | GPT-4：多模态+MoE+对齐 |
| 对齐比规模更重要 | 1.3B InstructGPT > 175B GPT-3 |
