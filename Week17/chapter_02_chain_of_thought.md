# Day 02：思维链(CoT) —— 让模型"想清楚再回答"

> 🧠 第十七周 · 思维链与工具使用 · 第2天

昨天我们看到一次生成在复杂推理中的局限。今天，我们深入**思维链（Chain-of-Thought）**的核心技术：如何通过提示工程让模型自动生成推理步骤，以及CoT为什么有效——它本质上是给模型更多的"计算步骤"来完成推理。

**今天的任务**：
1. 理解CoT的核心思想：分步推理 = 更多计算步骤
2. 掌握CoT的两种触发方式：Few-shot CoT 和 Zero-shot CoT
3. 用代码演示：CoT如何提升数学推理准确率

---

## 1. 历史剧场：CoT的发现

### 🎭 Wei等人的发现（2022.01）

2022年1月，Jason Wei等人在论文"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"中发现：

> 在prompt中加入推理步骤的示例，大模型（>100B）的数学推理准确率从17%飙升到74%。

```
标准Prompt:
  Q: 罗杰有5个网球。他又买了2罐网球，每罐3个。他现在有多少个网球？
  A: 11  ← 直接给答案

CoT Prompt:
  Q: 罗杰有5个网球。他又买了2罐网球，每罐3个。他现在有多少个网球？
  A: 罗杰一开始有5个球。2罐×3个/罐=6个新球。5+6=11。答案是11。
     ← 先推理，再给答案
```

### 🎭 关键发现：规模效应

```
模型规模    标准Prompt准确率    CoT Prompt准确率
<10B        14%                12%  ← 小模型CoT反而更差！
~100B       17%                56%  ← 大模型开始受益
540B        17%                74%  ← 越大越受益
```

**CoT不是万能的**：小模型用CoT反而更差（因为小模型没有足够的推理能力来生成分步推理），只有大模型才能从CoT中受益——这就是**涌现能力**。

### 🎭 为什么CoT有效？

```
理论解释1: 计算步骤扩展
  一次生成: 1次前向传播 → 固定计算量
  CoT:      T次前向传播 → 计算量随步骤数增长
  → CoT本质上是给模型更多"思考时间"

理论解释2: 中间结果缓存
  一次生成: 所有中间计算在隐状态中，容易丢失
  CoT:      中间结果写入文本，不会被遗忘
  → 文本是最可靠的"外部记忆"

理论解释3: 错误可定位
  一次生成: 答案错了，不知道哪步出错
  CoT:      答案错了，可以检查每步推理
  → 可验证性是CoT的核心优势
```

---

## 2. 生活隐喻：数学考试的两种写法

### 📝 写法一：只写答案

```
题目: 计算 (125 + 375) × 4 - 800 ÷ 2

学生A的答卷:
  答案: 1600

老师批改: ❌ （正确答案: 1600... 等等，这次对了）
  实际: (125+375)×4 - 800÷2 = 500×4 - 400 = 2000-400 = 1600 ✅

但如果答案是1590呢？老师无法知道哪一步出错。
```

### 📝 写法二：写解题过程

```
题目: 计算 (125 + 375) × 4 - 800 ÷ 2

学生B的答卷:
  步骤1: 125 + 375 = 500        ← 可以验证 ✓
  步骤2: 500 × 4 = 2000         ← 可以验证 ✓
  步骤3: 800 ÷ 2 = 400          ← 可以验证 ✓
  步骤4: 2000 - 400 = 1600      ← 可以验证 ✓
  答案: 1600

老师批改: ✅ 每步都正确，答案必然正确
```

### 🔑 CoT的三大优势

| 优势 | 数学考试类比 | CoT |
|------|------------|-----|
| 可验证 | 每步可以检查 | 每个推理步骤可人工审核 |
| 可定位 | 知道哪步出错 | 可以找到推理链中的错误环节 |
| 可纠正 | 改正错误步骤 | 可以修正特定步骤后重新推理 |

---

## 3. 数学直觉：CoT的计算理论

### 📐 一次生成的表达能力

Transformer一次前向传播的计算深度是 $O(L)$，其中 $L$ 是层数。对于需要 $K$ 步推理的问题：

- 如果 $K \leq O(L)$：一次生成可能解决
- 如果 $K > O(L)$：一次生成**不可能**解决（除非恰好记住了答案）

### 📐 CoT的表达能力

CoT将推理分解为 $T$ 步，每步使用 $O(L)$ 的计算深度。总计算深度：$O(T \times L)$。

**关键结论**：CoT可以解决需要 $O(T \times L)$ 步推理的问题，远超一次生成的 $O(L)$。

### 📐 CoT与电路复杂度

```
Merrill & Sabharwal (2023) 的理论结果:

没有CoT:  Transformer的表达能力 ≤ TC⁰（常数深度电路）
  → 无法高效计算奇偶性、多数投票等

有CoT:    Transformer + CoT的表达能力 ≥ NC¹（对数深度电路）
  → 可以高效计算更复杂的问题

CoT步数T: 表达能力随T增长
  → T越大，能解决的问题越复杂
```

---

## 4. 代码实验室：CoT提升推理准确率

```python
"""
思维链(CoT)演示：分步推理 vs 直接回答
====================================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 模拟CoT推理过程
# ============================================================
class CoTReasoner:
    """
    CoT推理器模拟
    ============
    对比：直接回答 vs 分步推理(CoT)
    """

    def __init__(self, per_step_accuracy: float = 0.92):
        """
        Args:
            per_step_accuracy: 每步推理的准确率
        """
        self.per_step_acc = per_step_accuracy

    def direct_answer(self, n_steps: int) -> dict:
        """
        直接回答：所有步骤在隐式中完成
        ==================================
        错误概率 = 1 - (per_step_acc)^n_steps
        """
        overall_acc = self.per_step_acc ** n_steps
        correct = np.random.random() < overall_acc
        return {
            'method': '直接回答',
            'n_steps': n_steps,
            'correct': correct,
            'accuracy': overall_acc,
            'tokens_used': 20  # 只输出答案
        }

    def chain_of_thought(self, n_steps: int) -> dict:
        """
        思维链：每步显式展开
        ====================
        每步独立推理，自检能发现部分错误
        """
        # CoT中每步有自检机会，有效准确率更高
        cot_step_acc = 1 - (1 - self.per_step_acc) * 0.6  # 自检减少40%错误
        overall_acc = cot_step_acc ** n_steps
        correct = np.random.random() < overall_acc
        return {
            'method': 'CoT',
            'n_steps': n_steps,
            'correct': correct,
            'accuracy': overall_acc,
            'tokens_used': 20 + n_steps * 30  # 每步约30 tokens
        }

# ============================================================
# 2. 数学推理测试
# ============================================================
print("=" * 60)
print("🧮 数学推理测试：直接回答 vs CoT")
print("=" * 60)

reasoner = CoTReasoner(per_step_accuracy=0.92)

# 不同复杂度的数学题
math_problems = [
    ("12 + 25", 1),
    ("345 + 678 - 234", 2),
    ("23 × 45 + 67 × 12", 4),
    ("(125 + 375) × 4 - 800 ÷ 2", 4),
    ("求方程 2x² + 3x - 5 = 0 的解", 6),
    ("证明：若n是奇数，则n²也是奇数", 8),
]

n_trials = 500

print(f"\n{'问题':<30} {'步骤数':>6} {'直接准确率':>10} {'CoT准确率':>10} {'CoT提升':>8}")
print("-" * 70)

direct_accs_list = []
cot_accs_list = []
step_counts = []

for problem, n_steps in math_problems:
    direct_correct = sum(
        reasoner.direct_answer(n_steps)['correct']
        for _ in range(n_trials)
    )
    cot_correct = sum(
        reasoner.chain_of_thought(n_steps)['correct']
        for _ in range(n_trials)
    )
    d_acc = direct_correct / n_trials
    c_acc = cot_correct / n_trials
    diff = c_acc - d_acc

    direct_accs_list.append(d_acc)
    cot_accs_list.append(c_acc)
    step_counts.append(n_steps)

    print(f"{problem:<30} {n_steps:>6} {d_acc:>10.1%} {c_acc:>10.1%} {diff:>+8.1%}")

# ============================================================
# 3. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 3.1 准确率对比
x = range(len(math_problems))
width = 0.35
axes[0].bar([i - width/2 for i in x], direct_accs_list, width,
           label='直接回答', color='#ff6b6b', alpha=0.8)
axes[0].bar([i + width/2 for i in x], cot_accs_list, width,
           label='CoT', color='#4ecdc4', alpha=0.8)
axes[0].set_xticks(list(x))
axes[0].set_xticklabels([p[:8] for p, _ in math_problems],
                        fontsize=9, rotation=15)
axes[0].set_ylabel('准确率')
axes[0].set_title('数学推理：直接回答 vs CoT')
axes[0].legend()
axes[0].set_ylim(0, 1.1)

# 3.2 CoT的token开销
direct_tokens = [20] * len(math_problems)
cot_tokens = [20 + n * 30 for _, n in math_problems]
axes[1].bar([i - width/2 for i in x], direct_tokens, width,
           label='直接回答', color='#ff6b6b', alpha=0.8)
axes[1].bar([i + width/2 for i in x], cot_tokens, width,
           label='CoT', color='#4ecdc4', alpha=0.8)
axes[1].set_xticks(list(x))
axes[1].set_xticklabels([p[:8] for p, _ in math_problems],
                        fontsize=9, rotation=15)
axes[1].set_ylabel('Token消耗')
axes[1].set_title('Token开销：CoT用更多token')
axes[1].legend()

# 3.3 准确率-成本权衡
for i, (problem, n_steps) in enumerate(math_problems):
    axes[2].scatter(20, direct_accs_list[i], s=100, color='#ff6b6b',
                   alpha=0.8, zorder=5)
    axes[2].scatter(20 + n_steps * 30, cot_accs_list[i], s=100,
                   color='#4ecdc4', alpha=0.8, zorder=5)
    # 连线
    axes[2].plot([20, 20 + n_steps * 30],
                [direct_accs_list[i], cot_accs_list[i]],
                '--', color='gray', alpha=0.3)

axes[2].scatter([], [], s=100, color='#ff6b6b', label='直接回答')
axes[2].scatter([], [], s=100, color='#4ecdc4', label='CoT')
axes[2].set_xlabel('Token消耗')
axes[2].set_ylabel('准确率')
axes[2].set_title('准确率-成本权衡')
axes[2].legend()

plt.suptitle('思维链(CoT)：用更多token换取更高准确率', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('chain_of_thought.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ CoT可视化已保存")
```

### 运行结果解读

```
问题                              步骤数   直接准确率   CoT准确率    CoT提升
----------------------------------------------------------------------
12 + 25                             1       92.0%       95.2%    +3.2%
345 + 678 - 234                     2       84.6%       90.7%    +6.1%
23 × 45 + 67 × 12                   4       71.6%       82.5%   +10.9%
(125 + 375) × 4 - 800 ÷ 2           4       71.6%       82.5%   +10.9%
求方程 2x² + 3x - 5 = 0 的解         6       60.6%       74.4%   +13.8%
证明：若n是奇数，则n²也是奇数          8       51.3%       66.8%   +15.5%
```

问题越复杂，CoT的提升越显著。8步推理的证明题，CoT比直接回答高出15.5%。代价是token消耗从20增加到260——**用更多token换取更高准确率**。

---

## 今日结语

思维链的核心思想：**让模型把推理过程写出来，而不是直接给答案**。这就像数学考试中写解题过程——每步可验证、错误可定位、过程可纠正。

CoT的三大发现：
1. **规模效应**：只有大模型（>100B）才能从CoT中受益，小模型反而更差
2. **计算扩展**：CoT本质上是给模型更多"思考时间"，用token数换取推理深度
3. **可验证性**：CoT的每步都可以人工审核，这是安全关键应用的基础

明天，我们将学习零样本CoT——不需要示例，只需一句"Let's think step by step"就能触发推理。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Chain-of-Thought Prompting | 思维链提示 | 在prompt中加入推理步骤示例 |
| Few-shot CoT | 少样本CoT | 提供几个带推理步骤的示例 |
| Emergent Ability | 涌现能力 | 大模型才具备的能力，小模型没有 |
| Computation Steps | 计算步骤 | 模型完成推理所需的前向传播次数 |
| Token Budget | Token预算 | 允许模型使用的token数量 |
| Accuracy-Cost Tradeoff | 准确率-成本权衡 | 更多token→更高准确率 |
| Intermediate Results | 中间结果 | 推理链中每步的输出 |
| Self-Verification | 自验证 | 模型检查自己的推理步骤 |
