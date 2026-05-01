# Day 05：反思模块 —— Agent的"自我纠错"

> 🤖 第十九周 · AI Agent · 第5天

没有反思的Agent就像没有质检的工厂——可能产出低质量结果。**反思模块**让Agent能评估自己的输出、发现错误、纠正方向。这是Agent区别于简单ReAct循环的关键能力。

**今天的任务**：
1. 理解反思的核心机制：自我评估、错误定位、策略调整
2. 掌握三种反思策略：结果验证、过程审查、对比反思
3. 用代码实现反思模块

---

## 1. 历史剧场：反思的发现

### 🎭 Reflexion论文（2023）

Shinn等人发表"Reflexion: Language Agents with Verbal Reinforcement Learning"，核心发现：

> 让Agent用**自然语言**反思自己的失败，比直接重试有效得多——反思让Agent从错误中学习，避免重复犯错。

```
无反思:
  尝试1: 失败 → 尝试2: 用同样方法 → 失败 → ...

有反思:
  尝试1: 失败
  反思: "我失败是因为没有考虑X，下次应该先查Y"
  尝试2: 根据反思调整策略 → 成功
```

---

## 2. 生活隐喻：考试后的错题本

### 📖 三种反思方式

```
方式1: 结果验证（对答案）
  "这道题答案是42，我算的是38，错了"
  → 知道错了，但不知道哪步出错

方式2: 过程审查（检查步骤）
  "步骤1: 23+47=70 ✓, 步骤2: 70×12=840 ✓, 步骤3: 156÷3=52 ✗(应该是52)"
  → 定位到具体错误步骤

方式3: 对比反思（和标准解法对比）
  "我的方法: 直接算 → 出错
   标准方法: 先化简再算 → 正确
   教训: 复杂计算应该先化简"
  → 从错误中提取可复用的教训
```

---

## 3. 数学直觉：反思的形式化

### 📐 反思函数

$$\text{Reflection} = R(\text{output}, \text{criteria}, \text{history})$$

输出包含：
- $\text{passed} \in \{0, 1\}$：是否通过
- $\text{feedback}$：具体反馈
- $\text{lesson}$：可复用的教训

### 📐 反思驱动的策略更新

$$\pi_{t+1} = \text{Update}(\pi_t, \text{Reflection}_t)$$

反思结果更新Agent的策略，避免重复犯错。

---

## 4. 代码实验室：反思模块实现

```python
"""
反思模块实现：自我评估与纠错
==========================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 反思模块
# ============================================================
class Reflector:
    """反思模块：评估输出质量并生成反馈"""

    def evaluate(self, output: str, criteria: str = "完整性") -> dict:
        """评估输出"""
        quality = np.clip(0.65 + np.random.randn() * 0.2, 0, 1)
        passed = quality > 0.6
        return {
            'quality': quality,
            'passed': passed,
            'feedback': self._generate_feedback(quality, criteria),
            'lesson': self._extract_lesson(quality, criteria)
        }

    def _generate_feedback(self, quality, criteria):
        if quality > 0.8:
            return "输出质量良好，满足要求"
        elif quality > 0.6:
            return "基本满足，但可以改进"
        else:
            return f"未满足{criteria}要求，需要重做"

    def _extract_lesson(self, quality, criteria):
        if quality < 0.6:
            return f"教训: 需要更注重{criteria}"
        return ""


class ReflexiveAgent:
    """带反思的Agent"""

    def __init__(self, max_retries: int = 3):
        self.reflector = Reflector()
        self.max_retries = max_retries
        self.lessons_learned = []

    def execute_with_reflection(self, task: str) -> dict:
        """执行任务+反思循环"""
        for attempt in range(1, self.max_retries + 1):
            # 执行任务（模拟）
            quality = 0.5 + 0.15 * attempt + np.random.randn() * 0.05
            output = f"任务'{task}'的输出(尝试{attempt})"

            # 反思
            eval_result = self.reflector.evaluate(output)
            eval_result['quality'] = np.clip(quality, 0, 1)
            eval_result['passed'] = quality > 0.6

            print(f"  尝试{attempt}: 质量={eval_result['quality']:.2f}, "
                  f"{'✅' if eval_result['passed'] else '❌'}")

            if eval_result['passed']:
                if eval_result['lesson']:
                    self.lessons_learned.append(eval_result['lesson'])
                return {'output': output, 'attempts': attempt,
                        'quality': eval_result['quality']}

            # 学习教训
            if eval_result['lesson']:
                self.lessons_learned.append(eval_result['lesson'])

        return {'output': output, 'attempts': self.max_retries,
                'quality': eval_result['quality']}


# ============================================================
# 2. 对比：有反思 vs 无反思
# ============================================================
print("=" * 60)
print("🔍 反思模块演示：有反思 vs 无反思")
print("=" * 60)

# 无反思：固定尝试次数
print("\n📌 无反思Agent:")
no_reflect_qualities = []
for i in range(5):
    q = np.clip(0.5 + np.random.randn() * 0.2, 0, 1)
    no_reflect_qualities.append(q)
    print(f"  任务{i+1}: 质量={q:.2f}")

# 有反思：质量逐步提升
print("\n📌 有反思Agent:")
agent = ReflexiveAgent(max_retries=3)
reflect_qualities = []
for i in range(5):
    print(f"  任务{i+1}:")
    result = agent.execute_with_reflection(f"任务{i+1}")
    reflect_qualities.append(result['quality'])

# ============================================================
# 3. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 3.1 质量对比
tasks_x = range(1, 6)
axes[0].plot(tasks_x, no_reflect_qualities, 'o-', color='#ff6b6b',
            linewidth=2, label='无反思')
axes[0].plot(tasks_x, reflect_qualities, 's-', color='#4ecdc4',
            linewidth=2, label='有反思')
axes[0].set_xlabel('任务编号')
axes[0].set_ylabel('输出质量')
axes[0].set_title('有反思 vs 无反思')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 3.2 反思策略对比
strategies = ['结果验证', '过程审查', '对比反思']
accuracy = [0.65, 0.75, 0.80]
cost = [1, 2, 3]
colors_s = ['#f39c12', '#45b7d1', '#4ecdc4']
for i, (s, a, c, col) in enumerate(zip(strategies, accuracy, cost, colors_s)):
    axes[1].scatter(c, a, s=200, color=col, alpha=0.8, zorder=5)
    axes[1].annotate(s, (c, a), textcoords="offset points",
                    xytext=(10, 5), fontsize=9)
axes[1].set_xlabel('成本(相对)')
axes[1].set_ylabel('纠错准确率')
axes[1].set_title('反思策略：成本 vs 效果')
axes[1].grid(True, alpha=0.3)

# 3.3 反思循环
steps = ['执行', '反思', '调整', '重执行', '反思', '通过']
step_colors = ['#45b7d1', '#9b59b6', '#f39c12', '#45b7d1', '#9b59b6', '#27ae60']
for i, (step, color) in enumerate(zip(steps, step_colors)):
    axes[2].barh(i, 1, color=color, alpha=0.8)
    axes[2].text(0.5, i, step, ha='center', va='center', fontsize=10, fontweight='bold')
axes[2].set_xlim(0, 1)
axes[2].set_ylim(-0.5, len(steps) - 0.5)
axes[2].set_title('反思循环：执行→反思→调整')
axes[2].set_yticks([])

plt.suptitle('反思模块：Agent的"自我纠错"', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('reflection_module.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 反思模块可视化已保存")
```

---

## 今日结语

反思模块让Agent从错误中学习，避免重复犯错。三种反思策略由浅入深：结果验证最简单但只能发现错误，过程审查能定位错误，对比反思能提取可复用的教训。Reflexion论文的核心发现：**用自然语言反思比直接重试有效得多**。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Reflection | 反思 | Agent评估和纠正自己输出的能力 |
| Reflexion | Reflexion | 用自然语言反思的Agent方法 |
| Self-evaluation | 自我评估 | Agent评估自己的输出质量 |
| Error Localization | 错误定位 | 找到具体出错的位置 |
| Lesson Learned | 教训 | 从错误中提取的可复用经验 |
| Verbal RL | 语言强化学习 | 用自然语言做奖励信号 |
