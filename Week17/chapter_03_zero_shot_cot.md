# Day 03：零样本CoT —— "Let's think step by step"的魔法

> 🧠 第十七周 · 思维链与工具使用 · 第3天

昨天的Few-shot CoT需要在prompt中提供推理步骤的示例，这既费时又需要领域知识。2022年，Kojima等人发现了一个令人惊讶的事实：**只需在问题后加上"Let's think step by step"，大模型就能自动生成推理步骤**——无需任何示例！这就是零样本CoT（Zero-shot CoT）。

**今天的任务**：
1. 理解零样本CoT的发现和原理
2. 对比Few-shot CoT和Zero-shot CoT的效果
3. 分析"Let's think step by step"为什么有效
4. 用代码演示：不同触发短语的效果对比

---

## 1. 历史剧场：一句提示词的魔法

### 🎭 Kojima等人的发现（2022.10）

2022年10月，Kojima等人发表"Large Language Models are Zero-Shot Reasoners"，核心发现：

```
标准Prompt:
  Q: 停车场有3辆汽车，又来了2辆，开走了1辆，现在有多少辆？
  A: 4  ← 直接回答，可能出错

Zero-shot CoT Prompt:
  Q: 停车场有3辆汽车，又来了2辆，开走了1辆，现在有多少辆？
  A: Let's think step by step.
     首先，停车场有3辆汽车。
     然后，又来了2辆，所以3+2=5辆。
     接着，开走了1辆，所以5-1=4辆。
     因此，现在有4辆汽车。
  ← 自动生成了推理步骤！
```

**不需要任何示例，一句"Let's think step by step"就够了！**

### 🎭 效果对比

| 方法 | MultiArith准确率 | GSM8K准确率 |
|------|-----------------|-------------|
| 直接回答 | 20.7% | 10.4% |
| Zero-shot CoT | 40.7% | 20.6% |
| Few-shot CoT | 54.2% | 35.5% |

Zero-shot CoT比直接回答翻倍，虽然不如Few-shot CoT，但**零成本**——不需要编写示例。

### 🎭 为什么"Let's think step by step"有效？

```
假设1: 格式触发
  → "step by step"触发了模型内部的"分步推理"模式
  → 模型在预训练数据中见过大量分步推理的文本
  → 这句话激活了相关的"推理回路"

假设2: 注意力重分配
  → 没有CoT时，模型的注意力集中在"生成答案"
  → 有CoT时，注意力分配到"生成推理步骤"
  → 改变了token生成的优先级

假设3: 上下文窗口利用
  → 直接回答：模型只用输入来生成输出
  → CoT：模型用自己生成的中间步骤作为额外上下文
  → 自生成的上下文是最相关的"工作记忆"
```

---

## 2. 生活隐喻：一句话改变思维方式

### 💬 场景：问朋友数学题

```
你: "345 × 789 等于多少？"
朋友: "272205"  ← 直接回答，可能心算出错

你: "345 × 789 等于多少？你一步步算"
朋友: "好，345 × 789...
       先算 345 × 700 = 241500
       再算 345 × 80 = 27600
       再算 345 × 9 = 3105
       加起来 241500 + 27600 + 3105 = 272205"
  ← "一步步算"触发了分步推理模式
```

**"一步步算"这四个字，改变了朋友的思维方式**——从心算（快速但易错）变成笔算（慢但准确）。"Let's think step by step"对大模型的作用完全一样。

### 🔑 不同触发短语的效果

| 触发短语 | 效果 | 原因 |
|---------|------|------|
| "Let's think step by step" | 最好 | 最直接地触发分步推理 |
| "Let's think slowly" | 中等 | 触发"慢思考"但不强调分步 |
| "Let's solve this" | 较弱 | 触发"解题"但不强调步骤 |
| 无触发 | 最弱 | 默认模式：直接回答 |

---

## 3. 数学直觉：Zero-shot CoT的理论分析

### 📐 条件概率视角

直接回答：

$$P(\text{answer} | \text{question})$$

Zero-shot CoT：

$$P(\text{answer} | \text{question}, \text{CoT trigger}) = \sum_{\text{steps}} P(\text{answer} | \text{steps}) \cdot P(\text{steps} | \text{question}, \text{CoT trigger})$$

CoT触发词改变了推理步骤的分布 $P(\text{steps} | \cdot)$，使其更倾向于**分步、有序、可验证**的推理。

### 📐 信息论视角

```
直接回答的信息瓶颈:
  I(question; answer) ≤ I(question; hidden_state)
  → 答案的信息受限于单次前向传播的隐状态

CoT的信息扩展:
  I(question; answer) ≤ I(question; steps) + I(steps; answer)
  → 答案的信息可以分步累积
  → 每步生成的文本成为下一步的额外输入
  → 信息容量随步骤数线性增长
```

### 📐 两阶段生成

Zero-shot CoT实际上是两阶段生成：

```
阶段1: 生成推理步骤
  steps = Generate(question + "Let's think step by step")

阶段2: 提取最终答案
  answer = Extract(steps)
  → 通常用"Therefore, the answer is"触发
```

---

## 4. 代码实验室：Zero-shot CoT效果对比

```python
"""
Zero-shot CoT演示：不同触发短语的效果
====================================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 模拟不同触发方式
# ============================================================
class CoTTriggerSimulator:
    """
    CoT触发模拟器
    ============
    模拟不同触发短语对推理准确率的影响
    """

    def __init__(self):
        # 不同触发方式的效果参数
        self.triggers = {
            '直接回答': {
                'step_acc_boost': 0.0,    # 无提升
                'step_generation_prob': 0.0,  # 不会生成步骤
                'description': '默认模式，直接给答案'
            },
            'Let\'s solve this': {
                'step_acc_boost': 0.03,
                'step_generation_prob': 0.3,
                'description': '触发解题模式，但不强调步骤'
            },
            'Let\'s think slowly': {
                'step_acc_boost': 0.05,
                'step_generation_prob': 0.5,
                'description': '触发慢思考，部分分步'
            },
            'Let\'s think step by step': {
                'step_acc_boost': 0.08,
                'step_generation_prob': 0.9,
                'description': '最强触发，完整分步推理'
            },
        }

    def evaluate(self, trigger_name: str, n_steps: int,
                 n_trials: int = 500) -> float:
        """
        评估某种触发方式的准确率

        Args:
            trigger_name: 触发方式名称
            n_steps: 问题需要的推理步骤数
            n_trials: 试验次数
        Returns:
            accuracy: 准确率
        """
        trigger = self.triggers[trigger_name]
        base_step_acc = 0.90  # 基础每步准确率
        boosted_acc = base_step_acc + trigger['step_acc_boost']
        step_gen_prob = trigger['step_generation_prob']

        correct_count = 0
        for _ in range(n_trials):
            # 决定是否生成推理步骤
            if np.random.random() < step_gen_prob:
                # 生成了步骤：用提升后的准确率
                overall_acc = boosted_acc ** n_steps
            else:
                # 没生成步骤：用基础准确率
                overall_acc = base_step_acc ** n_steps

            if np.random.random() < overall_acc:
                correct_count += 1

        return correct_count / n_trials

# ============================================================
# 2. 运行对比
# ============================================================
print("=" * 60)
print("🧪 Zero-shot CoT：不同触发短语效果对比")
print("=" * 60)

simulator = CoTTriggerSimulator()

# 不同复杂度的问题
complexities = [1, 3, 5, 8, 10, 15]
trigger_names = list(simulator.triggers.keys())
colors = ['#95a5a6', '#f39c12', '#45b7d1', '#4ecdc4']

results = {}
for trigger in trigger_names:
    results[trigger] = []
    for n_steps in complexities:
        acc = simulator.evaluate(trigger, n_steps)
        results[trigger].append(acc)

# 打印结果
print(f"\n{'触发方式':<30} ", end="")
for n in complexities:
    print(f"{'%d步' % n:>8}", end="")
print()
print("-" * 78)

for trigger in trigger_names:
    print(f"{trigger:<30} ", end="")
    for acc in results[trigger]:
        print(f"{acc:>8.1%}", end="")
    print()

# ============================================================
# 3. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 3.1 准确率曲线
for i, trigger in enumerate(trigger_names):
    axes[0].plot(complexities, results[trigger], 'o-', color=colors[i],
                linewidth=2, markersize=6, label=trigger, alpha=0.8)
axes[0].set_xlabel('推理步骤数')
axes[0].set_ylabel('准确率')
axes[0].set_title('不同触发方式：准确率 vs 复杂度')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# 3.2 触发效果雷达图
categories = ['步骤生成\n概率', '准确率\n提升', '简单题\n效果', '复杂题\n效果']
trigger_scores = {
    '直接回答': [0, 0, 0.9, 0.3],
    'Let\'s solve this': [0.3, 0.3, 0.85, 0.45],
    'Let\'s think slowly': [0.5, 0.5, 0.88, 0.55],
    'Let\'s think step by step': [0.9, 0.8, 0.92, 0.7],
}

x = np.arange(len(categories))
width = 0.2
for i, trigger in enumerate(trigger_names):
    offset = (i - 1.5) * width
    axes[1].bar(x + offset, trigger_scores[trigger], width,
               color=colors[i], alpha=0.8, label=trigger[:15])
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories, fontsize=9)
axes[1].set_ylabel('评分')
axes[1].set_title('触发方式综合评分')
axes[1].legend(fontsize=7)

# 3.3 Few-shot vs Zero-shot CoT对比
methods = ['直接回答', 'Zero-shot\nCoT', 'Few-shot\nCoT']
math_acc = [20.7, 40.7, 54.2]
gsm8k_acc = [10.4, 20.6, 35.5]

x = np.arange(len(methods))
width = 0.35
axes[2].bar(x - width/2, math_acc, width, label='MultiArith',
           color='#4ecdc4', alpha=0.8)
axes[2].bar(x + width/2, gsm8k_acc, width, label='GSM8K',
           color='#45b7d1', alpha=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(methods)
axes[2].set_ylabel('准确率 (%)')
axes[2].set_title('Few-shot vs Zero-shot CoT')
axes[2].legend()

plt.suptitle('零样本CoT："Let\'s think step by step"的魔法', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('zero_shot_cot.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Zero-shot CoT可视化已保存")
```

### 运行结果解读

"Let's think step by step"在所有复杂度上都优于其他触发方式，且优势随问题复杂度增大而增大。Zero-shot CoT虽然不如Few-shot CoT，但**零成本**——不需要编写任何示例。

---

## 今日结语

零样本CoT的发现揭示了大模型的一个深层特性：**推理能力已经隐含在预训练权重中，只需要正确的"触发词"来激活**。"Let's think step by step"就像一把钥匙，打开了模型内部的推理回路。

关键洞察：
1. **零成本**：不需要编写示例，一句触发词即可
2. **规模依赖**：和Few-shot CoT一样，只有大模型才能受益
3. **不是万能**：Zero-shot CoT的效果不如Few-shot CoT，但性价比极高
4. **触发机制**：触发词激活了预训练数据中的"分步推理"模式

明天，我们将学习工具使用——让模型不仅能"思考"，还能"动手"：调用计算器、搜索引擎、代码执行器等外部工具。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Zero-shot CoT | 零样本思维链 | 不需要示例，只用触发词 |
| Trigger Phrase | 触发短语 | 激活推理模式的提示词 |
| Few-shot CoT | 少样本思维链 | 需要提供推理步骤示例 |
| Two-stage Generation | 两阶段生成 | 先生成步骤，再提取答案 |
| Extraction | 提取 | 从推理步骤中提取最终答案 |
| Information Bottleneck | 信息瓶颈 | 单次前向传播的信息容量限制 |
| Reasoning Mode | 推理模式 | 模型内部的分步推理"回路" |
