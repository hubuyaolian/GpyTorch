# Day 08：提示工程 —— 设计高效Prompt的艺术

> 🧠 第十八周 · 思维链与工具使用 · 第8天

无论是CoT、工具使用还是ReAct，都需要精心设计的**提示（Prompt）**来触发。同样的模型，不同的prompt可能产生天差地别的效果。提示工程（Prompt Engineering）就是研究如何设计高效prompt的系统性方法。

**今天的任务**：
1. 掌握提示工程的核心原则：清晰、具体、结构化
2. 学习高级技巧：少样本、思维链触发、角色设定
3. 对比不同prompt策略的效果

---

## 1. 历史剧场：提示工程的演进

### 🎭 从"零样本"到"少样本"到"CoT"

```
2020 GPT-3: 零样本/少样本提示
  → 证明大模型可以通过prompt直接执行任务

2022 CoT: 思维链提示
  → "Let's think step by step"触发推理

2023 ReAct: 推理-行动提示
  → 结构化的Thought/Action/Observation格式

2023 System Prompt: 系统提示
  → 定义模型的角色、能力边界、输出格式
```

### 🎭 提示工程的六大原则

```
原则1: 清晰明确
  ❌ "帮我处理一下这个"
  ✅ "请将以下文本翻译成英文，保持专业术语不变"

原则2: 提供上下文
  ❌ "写一封邮件"
  ✅ "你是一位客户经理，请写一封道歉邮件给投诉的客户"

原则3: 结构化输出
  ❌ "分析一下优缺点"
  ✅ "请按以下格式输出：\n优点：\n- ...\n缺点：\n- ..."

原则4: 分步指导
  ❌ "完成这个任务"
  ✅ "步骤1: ... 步骤2: ... 步骤3: ..."

原则5: 示例驱动
  ❌ 纯文字描述
  ✅ 提供2-3个输入输出示例

原则6: 约束边界
  ❌ 无限制
  ✅ "回答不超过200字，使用中文"
```

---

## 2. 生活隐喻：给助手下指令

### 📋 好指令 vs 坏指令

```
坏指令: "帮我弄一下那个东西"
  → 助手完全不知道你要什么

好指令: "请把桌上那份10页的报告，提取前3页的要点，
        每个要点不超过50字，用bullet point格式，
        30分钟内完成"
  → 任务明确、格式清晰、有约束
```

### 🔑 提示工程的本质

提示工程的本质是**降低模型输出空间的不确定性**：

```
没有prompt约束:
  输出空间 = 所有可能的文本序列 → 天文数字

有prompt约束:
  输出空间 = 符合格式、内容、长度要求的文本 → 大幅缩小

约束越多 → 输出越确定 → 质量越可控
```

---

## 3. 数学直觉：Prompt的信息论视角

### 📐 Prompt作为条件

$$P(\text{output} | \text{prompt}) \neq P(\text{output})$$

好的prompt使 $P(\text{output} | \text{prompt})$ 集中在高质量输出上。

### 📐 互信息最大化

最优prompt最大化输出与目标之间的互信息：

$$\text{prompt}^* = \arg\max_{\text{prompt}} I(\text{output}; \text{target} | \text{prompt})$$

### 📐 少样本的信息量

$k$个示例提供的额外信息：

$$I(\text{examples}) = \sum_{i=1}^{k} \log \frac{P(\text{output}_i | \text{input}_i, \text{prompt})}{P(\text{output}_i | \text{prompt})}$$

---

## 4. 代码实验室：提示工程技巧对比

```python
"""
提示工程：不同策略的效果对比
==========================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 模拟不同prompt策略
# ============================================================
class PromptStrategy:
    """提示策略模拟器"""

    def __init__(self):
        self.strategies = {
            '零样本': {
                'quality': 0.40,
                'consistency': 0.30,
                'format_adherence': 0.20,
                'description': '直接提问，无示例'
            },
            '少样本(2例)': {
                'quality': 0.60,
                'consistency': 0.55,
                'format_adherence': 0.50,
                'description': '提供2个输入输出示例'
            },
            '少样本(5例)': {
                'quality': 0.70,
                'consistency': 0.65,
                'format_adherence': 0.70,
                'description': '提供5个输入输出示例'
            },
            'CoT触发': {
                'quality': 0.75,
                'consistency': 0.60,
                'format_adherence': 0.45,
                'description': '"Let\'s think step by step"'
            },
            '角色+格式': {
                'quality': 0.65,
                'consistency': 0.75,
                'format_adherence': 0.85,
                'description': '设定角色+指定输出格式'
            },
            'CoT+角色+格式': {
                'quality': 0.85,
                'consistency': 0.80,
                'format_adherence': 0.80,
                'description': '组合所有技巧'
            },
        }

    def evaluate(self, strategy_name: str, n_tasks: int = 100):
        """评估策略效果"""
        s = self.strategies[strategy_name]
        results = {
            'quality': s['quality'] + np.random.randn(n_tasks) * 0.1,
            'consistency': s['consistency'] + np.random.randn(n_tasks) * 0.1,
            'format': s['format_adherence'] + np.random.randn(n_tasks) * 0.1,
        }
        return results

# ============================================================
# 2. 运行对比
# ============================================================
print("=" * 60)
print("📝 提示工程：策略效果对比")
print("=" * 60)

evaluator = PromptStrategy()

print(f"\n{'策略':<18} {'质量':>8} {'一致性':>8} {'格式遵循':>8}")
print("-" * 46)

strategy_names = list(evaluator.strategies.keys())
quality_means = []
consistency_means = []
format_means = []

for name in strategy_names:
    results = evaluator.evaluate(name)
    q = np.clip(results['quality'].mean(), 0, 1)
    c = np.clip(results['consistency'].mean(), 0, 1)
    f = np.clip(results['format'].mean(), 0, 1)
    quality_means.append(q)
    consistency_means.append(c)
    format_means.append(f)
    print(f"{name:<18} {q:>8.1%} {c:>8.1%} {f:>8.1%}")

# ============================================================
# 3. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 3.1 三维评分
x = np.arange(len(strategy_names))
width = 0.25
axes[0].bar(x - width, quality_means, width, label='质量', color='#4ecdc4', alpha=0.8)
axes[0].bar(x, consistency_means, width, label='一致性', color='#45b7d1', alpha=0.8)
axes[0].bar(x + width, format_means, width, label='格式遵循', color='#9b59b6', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels([s[:6] for s in strategy_names], fontsize=8, rotation=20)
axes[0].set_ylabel('评分')
axes[0].set_title('提示策略三维评分')
axes[0].legend(fontsize=8)

# 3.2 Prompt长度 vs 效果
prompt_lengths = [20, 80, 200, 100, 150, 300]
overall_scores = [
    (q + c + f) / 3
    for q, c, f in zip(quality_means, consistency_means, format_means)
]
colors_p = ['#95a5a6', '#f39c12', '#ff6b6b', '#45b7d1', '#9b59b6', '#4ecdc4']
for i, (length, score) in enumerate(zip(prompt_lengths, overall_scores)):
    axes[1].scatter(length, score, s=200, color=colors_p[i], alpha=0.8, zorder=5)
    axes[1].annotate(strategy_names[i][:8], (length, score),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
axes[1].set_xlabel('Prompt长度(tokens)')
axes[1].set_ylabel('综合评分')
axes[1].set_title('Prompt长度 vs 效果')
axes[1].grid(True, alpha=0.3)

# 3.3 组合效应
combos = ['基础', '+少样本', '+CoT', '+角色', '+格式', '全组合']
combo_scores = [0.40, 0.55, 0.65, 0.58, 0.60, 0.82]
axes[2].bar(combos, combo_scores, color=colors_p, alpha=0.8)
axes[2].set_ylabel('综合评分')
axes[2].set_title('技巧组合的叠加效应')
for i, s in enumerate(combo_scores):
    axes[2].text(i, s + 0.02, f'{s:.0%}', ha='center', fontsize=9)

plt.suptitle('提示工程：设计高效Prompt的艺术', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('prompt_engineering.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 提示工程可视化已保存")
```

---

## 今日结语

提示工程的核心原则：**清晰、具体、结构化**。好的prompt通过约束输出空间来提升质量——约束越多，输出越确定。最有效的策略是组合多种技巧：少样本示例提供模式，CoT触发推理，角色设定提供上下文，格式约束确保输出可用。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Prompt Engineering | 提示工程 | 设计高效prompt的系统性方法 |
| Zero-shot | 零样本 | 不提供示例，直接提问 |
| Few-shot | 少样本 | 提供少量输入输出示例 |
| System Prompt | 系统提示 | 定义模型角色和约束 |
| Format Adherence | 格式遵循 | 输出符合指定格式的程度 |
| Output Space | 输出空间 | 所有可能输出的集合 |
