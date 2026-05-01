# Day 03：规划模块 —— 将"大目标"变成"小任务"

> 🤖 第十九周 · AI Agent · 第3天

Agent最核心的能力是**规划**：将一个复杂的大目标分解为可执行的子任务序列。好的规划让Agent高效完成目标，坏的规划让Agent在无关任务上浪费时间。今天我们深入规划模块的设计：分解策略、依赖管理、动态调整。

**今天的任务**：
1. 理解规划的核心挑战：分解粒度、依赖关系、动态调整
2. 掌握三种规划策略：自顶向下、逐步细化、自适应
3. 用代码实现一个规划器

---

## 1. 历史剧场：规划的经典方法

### 🎭 三种规划策略

```
策略1: 自顶向下（Top-down）
  目标 → [任务1, 任务2, 任务3, 任务4, 任务5]
  一次性分解所有子任务，然后顺序执行
  优点: 全局视角  缺点: 不够灵活

策略2: 逐步细化（Iterative）
  目标 → 任务1 → 执行 → 任务2 → 执行 → ...
  每完成一个任务再规划下一个
  优点: 灵活  缺点: 可能走偏

策略3: 自适应（Adaptive）
  目标 → 粗略计划 → 执行+反思 → 调整计划 → 继续
  先粗略规划，执行中根据反馈调整
  优点: 兼顾全局和灵活  缺点: 实现复杂
```

---

## 2. 生活隐喻：旅行规划

### ✈️ 三种旅行规划方式

```
自顶向下: 出发前制定完整行程
  Day1: 北京→巴黎, Day2: 卢浮宫, Day3: 凡尔赛...
  → 计划周全但遇到突发情况难以调整

逐步细化: 走一步看一步
  先到巴黎 → 看看有什么好玩的 → 决定明天去哪
  → 灵活但可能错过好景点

自适应: 粗略框架+灵活调整
  大方向: 巴黎3天→罗马2天
  每天: 根据天气和心情调整具体行程
  → 兼顾全局和灵活
```

---

## 3. 数学直觉：规划的形式化

### 📐 任务依赖图

子任务之间可能有依赖关系，形成**有向无环图（DAG）**：

$$G = (V, E), \quad V = \{t_1, t_2, \ldots, t_n\}, \quad E \subseteq V \times V$$

其中 $(t_i, t_j) \in E$ 表示 $t_j$ 依赖 $t_i$ 的输出。

### 📐 规划的优化目标

$$\text{Plan}^* = \arg\min_{\text{Plan}} \mathbb{E}[\text{Cost}(\text{Plan})] \quad \text{s.t.} \quad \text{Goal完成}$$

---

## 4. 代码实验室：规划器实现

```python
"""
规划模块实现：三种策略对比
========================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 三种规划策略
# ============================================================
class TopDownPlanner:
    """自顶向下规划：一次性分解"""

    def plan(self, goal: str) -> list:
        return [
            {'task': '搜索信息', 'deps': []},
            {'task': '整理数据', 'deps': ['搜索信息']},
            {'task': '分析发现', 'deps': ['整理数据']},
            {'task': '撰写报告', 'deps': ['分析发现']},
            {'task': '审阅修改', 'deps': ['撰写报告']},
        ]

class IterativePlanner:
    """逐步细化规划：每步规划下一个"""

    def __init__(self):
        self.completed = []

    def next_task(self, goal: str) -> dict:
        if not self.completed:
            task = {'task': '搜索信息', 'deps': []}
        elif len(self.completed) == 1:
            task = {'task': '整理数据', 'deps': ['搜索信息']}
        elif len(self.completed) == 2:
            task = {'task': '分析发现', 'deps': ['整理数据']}
        elif len(self.completed) == 3:
            task = {'task': '撰写报告', 'deps': ['分析发现']}
        else:
            task = {'task': '审阅修改', 'deps': ['撰写报告']}
        self.completed.append(task)
        return task

class AdaptivePlanner:
    """自适应规划：粗略框架+动态调整"""

    def plan(self, goal: str) -> list:
        # 粗略框架
        return [
            {'task': '调研阶段', 'subtasks': ['搜索', '整理'], 'flexible': True},
            {'task': '分析阶段', 'subtasks': ['分析', '验证'], 'flexible': True},
            {'task': '输出阶段', 'subtasks': ['撰写', '审阅'], 'flexible': True},
        ]

    def adjust(self, plan: list, feedback: str) -> list:
        """根据反馈调整计划"""
        if "数据不足" in feedback:
            plan.insert(1, {'task': '补充调研', 'subtasks': ['补充搜索'], 'flexible': True})
        return plan

# ============================================================
# 2. 对比运行
# ============================================================
print("=" * 60)
print("📋 规划策略对比")
print("=" * 60)

# 自顶向下
td = TopDownPlanner()
td_plan = td.plan("写市场分析报告")
print(f"\n📌 自顶向下: {len(td_plan)}个任务")
for t in td_plan:
    print(f"   {t['task']} (依赖: {t['deps']})")

# 自适应
ap = AdaptivePlanner()
ap_plan = ap.plan("写市场分析报告")
print(f"\n📌 自适应: {len(ap_plan)}个阶段")
adjusted = ap.adjust(ap_plan, "数据不足，需要补充")
print(f"   调整后: {len(adjusted)}个阶段")

# ============================================================
# 3. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 3.1 三种策略对比
strategies = ['自顶向下', '逐步细化', '自适应']
flexibility = [0.3, 0.8, 0.7]
efficiency = [0.8, 0.5, 0.7]
quality = [0.6, 0.5, 0.8]

x = np.arange(len(strategies))
width = 0.25
axes[0].bar(x - width, flexibility, width, label='灵活性', color='#4ecdc4', alpha=0.8)
axes[0].bar(x, efficiency, width, label='效率', color='#45b7d1', alpha=0.8)
axes[0].bar(x + width, quality, width, label='质量', color='#9b59b6', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(strategies)
axes[0].set_ylabel('评分')
axes[0].set_title('规划策略三维对比')
axes[0].legend(fontsize=8)

# 3.2 任务依赖图
tasks = ['搜索', '整理', '分析', '撰写', '审阅']
for i, task in enumerate(tasks):
    axes[1].scatter(i, 0, s=300, color='#4ecdc4', alpha=0.8, zorder=5)
    axes[1].text(i, 0, task, ha='center', va='center', fontsize=9, fontweight='bold')
    if i > 0:
        axes[1].annotate('', xy=(i-0.1, 0), xytext=(i-1+0.1, 0),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2))
axes[1].set_xlim(-0.5, len(tasks)-0.5)
axes[1].set_title('任务依赖图（DAG）')
axes[1].set_yticks([])

# 3.3 规划深度 vs 效果
depths = [1, 2, 3, 4, 5, 6]
td_quality = [0.4, 0.55, 0.65, 0.70, 0.72, 0.72]
ad_quality = [0.4, 0.50, 0.60, 0.72, 0.80, 0.82]

axes[2].plot(depths, td_quality, 'o-', color='#ff6b6b', linewidth=2, label='自顶向下')
axes[2].plot(depths, ad_quality, 's-', color='#4ecdc4', linewidth=2, label='自适应')
axes[2].set_xlabel('规划深度')
axes[2].set_ylabel('目标完成质量')
axes[2].set_title('规划深度 vs 质量')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('规划模块：将"大目标"变成"小任务"', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('planning_module.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 规划模块可视化已保存")
```

---

## 今日结语

规划是Agent最核心的能力。三种策略各有优劣：自顶向下效率高但不够灵活，逐步细化灵活但可能走偏，自适应兼顾全局和灵活。实际应用中，**自适应规划**是最常用的策略——先粗略规划，执行中根据反馈调整。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Top-down Planning | 自顶向下规划 | 一次性分解所有子任务 |
| Iterative Planning | 逐步细化规划 | 每步规划下一个任务 |
| Adaptive Planning | 自适应规划 | 粗略框架+动态调整 |
| Task Dependency | 任务依赖 | 子任务之间的先后关系 |
| DAG | 有向无环图 | 任务依赖关系的数学表示 |
| Replanning | 重新规划 | 根据反馈调整计划 |
