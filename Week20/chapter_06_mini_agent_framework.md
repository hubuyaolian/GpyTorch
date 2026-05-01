# Day 06：迷你Agent框架 —— 从零实现一个Agent

> 🤖 第二十周 · AI Agent · 第6天

前五天我们分别学习了Agent的四大模块。今天，我们把它们组合成一个**完整可运行的迷你Agent框架**——虽然简化，但包含规划、记忆、反思、行动的完整闭环。

**今天的任务**：
1. 实现MiniAgent类：组合四大模块
2. 运行一个完整的Agent任务
3. 分析Agent的执行轨迹和性能

---

## 1. 历史剧场：Agent框架的设计哲学

### 🎭 简单即美

```
LangChain: 功能全面但复杂
  → 数百个类，学习曲线陡峭

我们的MiniAgent: 核心功能最小化
  → 4个模块，1个循环，清晰易懂
  → 教学优先，工程其次
```

---

## 2. 代码实验室：MiniAgent完整实现

```python
"""
MiniAgent：完整可运行的迷你Agent框架
==================================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 四大模块（精简版）
# ============================================================
class MiniPlanner:
    """规划器"""
    def plan(self, goal):
        if "报告" in goal:
            return ["搜索信息", "整理数据", "分析发现", "撰写报告", "审阅修改"]
        elif "分析" in goal:
            return ["收集数据", "分析数据", "总结发现"]
        return ["执行任务"]

class MiniMemory:
    """记忆库"""
    def __init__(self):
        self.store = []
    def save(self, key, value):
        self.store.append({'key': key, 'value': value})
    def retrieve(self, query):
        return [m for m in self.store if query in m['key'] or m['key'] in query]
    def __len__(self):
        return len(self.store)

class MiniReflector:
    """反思器"""
    def evaluate(self, output):
        quality = np.clip(0.6 + 0.1 * len(output) / 10 + np.random.randn() * 0.1, 0, 1)
        return {'quality': quality, 'passed': quality > 0.55,
                'feedback': "通过" if quality > 0.55 else "需改进"}

class MiniExecutor:
    """执行器"""
    def execute(self, task, memory):
        relevant = memory.retrieve(task)
        ctx = f"(参考{len(relevant)}条记忆)" if relevant else ""
        return f"完成: {task} {ctx}"

# ============================================================
# 2. MiniAgent
# ============================================================
class MiniAgent:
    """
    迷你Agent框架
    ============
    规划→执行→反思→记忆 的完整闭环
    """

    def __init__(self, max_retries=2):
        self.planner = MiniPlanner()
        self.memory = MiniMemory()
        self.reflector = MiniReflector()
        self.executor = MiniExecutor()
        self.max_retries = max_retries

    def run(self, goal):
        """运行Agent"""
        print(f"🎯 目标: {goal}")

        # 规划
        plan = self.planner.plan(goal)
        print(f"📋 计划: {plan}")

        results = []
        for i, task in enumerate(plan):
            # 执行
            output = self.executor.execute(task, self.memory)

            # 反思
            for attempt in range(1, self.max_retries + 1):
                eval_result = self.reflector.evaluate(output)
                if eval_result['passed']:
                    break
                output = self.executor.execute(task, self.memory)

            # 记忆
            self.memory.save(task, output)
            results.append({
                'task': task, 'output': output,
                'quality': eval_result['quality'],
                'attempts': attempt
            })
            status = '✅' if eval_result['passed'] else '⚠️'
            print(f"  {status} {task}: 质量={eval_result['quality']:.2f}")

        avg_q = np.mean([r['quality'] for r in results])
        print(f"\n📊 完成: {len(results)}个任务, 平均质量={avg_q:.2f}, 记忆={len(self.memory)}条")
        return {'goal': goal, 'results': results, 'avg_quality': avg_q}

# ============================================================
# 3. 运行
# ============================================================
print("=" * 60)
print("🤖 MiniAgent演示")
print("=" * 60)

agent = MiniAgent()
result = agent.run("写一份市场分析报告")

# ============================================================
# 4. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

tasks = [r['task'] for r in result['results']]
qualities = [r['quality'] for r in result['results']]
colors_q = ['#4ecdc4' if q > 0.55 else '#ff6b6b' for q in qualities]

axes[0].bar(tasks, qualities, color=colors_q, alpha=0.8)
axes[0].set_ylabel('质量')
axes[0].set_title('各子任务质量')
axes[0].set_ylim(0, 1)

# Agent执行流程
flow = ['目标', '规划', '执行1', '反思1', '记忆1', '执行2', '反思2', '记忆2', '...', '完成']
for i, step in enumerate(flow):
    color = '#4ecdc4' if i % 3 == 0 else ('#45b7d1' if i % 3 == 1 else '#9b59b6')
    axes[1].barh(i, 1, color=color, alpha=0.7)
    axes[1].text(0.5, i, step, ha='center', va='center', fontsize=9)
axes[1].set_xlim(0, 1)
axes[1].set_title('Agent执行流程')
axes[1].set_yticks([])

plt.suptitle('MiniAgent：完整可运行的迷你框架', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('mini_agent_framework.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ MiniAgent可视化已保存")
```

---

## 今日结语

MiniAgent虽然简化，但包含了Agent的核心闭环：规划→执行→反思→记忆。这个框架可以扩展为更复杂的系统，但核心循环不变。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| MiniAgent | 迷你Agent | 教学用的简化Agent框架 |
| Agent Loop | Agent循环 | 规划→执行→反思→记忆的闭环 |
