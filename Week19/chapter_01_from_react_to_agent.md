# Day 01：从ReAct到Agent —— 从"执行任务"到"完成目标"

> 🤖 第十九周 · AI Agent · 第1天

ReAct让模型能"思考-行动-观察"循环，但它只能执行**单个任务**——你问它一个问题，它回答完就结束了。现实中的复杂目标需要**多步规划、跨任务记忆、自我反思**：比如"帮我调研竞品并写分析报告"，这需要搜索、整理、分析、写作多个步骤，且后续步骤依赖前面的结果。**Agent就是能自主完成复杂目标的AI系统**。

**今天的任务**：
1. 理解Agent与ReAct的本质区别：从单任务到多步骤目标
2. 分析Agent的四大核心能力：规划、记忆、反思、行动
3. 回顾Agent的历史：从AutoGPT到现代框架

---

## 1. 历史剧场：2023，Agent元年

### 🎭 AutoGPT的爆发（2023.03）

2023年3月，AutoGPT在GitHub上星标暴涨，它是第一个让GPT-4**自主完成复杂目标**的系统：

```
目标: "调研量子计算领域的最新进展，写一份5页的分析报告"

AutoGPT的执行过程:
  Task 1: 搜索"量子计算 2024 最新进展"
  Task 2: 整理搜索结果，提取关键信息
  Task 3: 搜索"量子计算 主要公司"
  Task 4: 整理公司信息
  Task 5: 分析技术趋势
  Task 6: 撰写报告大纲
  Task 7: 根据大纲写报告
  Task 8: 审阅报告，修改不足
  → 产出: 5页分析报告
```

**AutoGPT vs ReAct的核心区别**：AutoGPT能将大目标分解为多个子任务，并在子任务间传递信息。

### 🎭 Agent的四大核心能力

```
1. 规划(Planning): 将大目标分解为子任务序列
   "写分析报告" → [搜索, 整理, 分析, 写作, 审阅]

2. 记忆(Memory): 跨任务存储和检索信息
   Task 2需要用到Task 1的搜索结果

3. 反思(Reflection): 评估自己的输出质量
   "这份报告是否完整？是否需要补充？"

4. 行动(Action): 调用工具执行具体操作
   搜索、计算、写文件、发邮件...
```

### 🎭 Agent框架的演进

```
2023.03  AutoGPT: 第一个自主Agent（但经常跑偏）
2023.04  BabyAGI: 更简洁的任务管理Agent
2023.10  LangGraph: LangChain的Agent图框架
2024.01  CrewAI: 多Agent协作框架
2024.03  AutoGen: 微软的多Agent对话框架
```

---

## 2. 生活隐喻：实习生 vs 项目经理

### 👨‍💼 ReAct = 实习生

```
你: "帮我查一下蒙特利尔的人口"
实习生: [查资料] "约420万"
  → 完成一个任务就结束了

你: "帮我写一份加拿大市场分析报告"
实习生: ??? （不知道怎么分解任务）
```

### 👩‍💼 Agent = 项目经理

```
你: "帮我写一份加拿大市场分析报告"
项目经理:
  1. [规划] 分解为: 市场规模→竞争格局→进入壁垒→报告撰写
  2. [行动] 搜索市场规模数据
  3. [记忆] 存储搜索结果
  4. [行动] 搜索竞争格局
  5. [记忆] 存储竞争数据
  6. [反思] 数据是否充分？需要补充吗？
  7. [行动] 撰写报告
  8. [反思] 报告质量如何？需要修改吗？
  9. [行动] 修改完善
  → 产出完整报告
```

### 🔑 Agent vs ReAct

| | ReAct | Agent |
|---|---|---|
| 目标类型 | 单个问题 | 复杂目标 |
| 任务数 | 1个 | 多个（自动分解） |
| 记忆 | 无（每轮独立） | 有（跨任务存储） |
| 反思 | 无 | 有（自我评估） |
| 规划 | 无（逐步执行） | 有（提前规划） |
| 类比 | 实习生 | 项目经理 |

---

## 3. 数学直觉：Agent的形式化

### 📐 Agent的MDP扩展

ReAct是一个单轮MDP，Agent是一个**分层MDP（Hierarchical MDP）**：

```
高层MDP: 目标 → 子目标序列 → 最终结果
  状态: 当前进度（哪些子目标已完成）
  动作: 选择下一个子目标
  奖励: 最终目标是否完成

低层MDP: 子目标 → ReAct循环 → 子目标结果
  状态: 当前推理-行动轨迹
  动作: Thought/Action
  奖励: 子目标是否完成
```

### 📐 规划的复杂度

```
无规划(ReAct): 每步贪心选择 → 可能走入死胡同
有规划(Agent): 提前分解任务 → 全局最优

规划的价值:
  V(plan) = E[完成目标] - E[不规划完成目标]
  → 对于复杂目标，V(plan) >> 0
  → 对于简单问题，V(plan) ≈ 0
```

---

## 4. 代码实验室：ReAct vs Agent对比

```python
"""
ReAct vs Agent对比：单任务 vs 多步骤目标
========================================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 模拟ReAct和Agent的执行过程
# ============================================================
class ReActSimulator:
    """ReAct模拟：只能执行单任务"""

    def execute(self, task: str) -> dict:
        """执行单个任务"""
        return {
            'method': 'ReAct',
            'tasks_completed': 1,
            'quality': 0.7 + np.random.randn() * 0.1,
            'steps': np.random.randint(2, 5)
        }

class AgentSimulator:
    """Agent模拟：能分解和执行多步骤目标"""

    def execute(self, goal: str) -> dict:
        """执行复杂目标"""
        # 分解为子任务
        n_subtasks = np.random.randint(3, 8)
        subtask_results = []
        memory = []

        for i in range(n_subtasks):
            # 执行子任务（利用记忆中的信息）
            quality = 0.75 + 0.02 * len(memory) + np.random.randn() * 0.05
            result = {'subtask': i, 'quality': quality}
            subtask_results.append(result)
            memory.append(result)

            # 反思：检查是否需要补充
            if quality < 0.7 and i < n_subtasks - 1:
                # 补充一个子任务
                extra = {'subtask': f'补充{i}', 'quality': 0.8}
                subtask_results.append(extra)
                memory.append(extra)

        return {
            'method': 'Agent',
            'tasks_completed': n_subtasks,
            'quality': np.mean([r['quality'] for r in subtask_results]),
            'steps': len(subtask_results),
            'memory_used': len(memory)
        }

# ============================================================
# 2. 运行对比
# ============================================================
print("=" * 60)
print("🤖 ReAct vs Agent：单任务 vs 多步骤目标")
print("=" * 60)

react = ReActSimulator()
agent = AgentSimulator()

# 简单问题
print("\n📌 简单问题: '蒙特利尔人口多少？'")
r1 = react.execute("查人口")
print(f"   ReAct: 完成{r1['tasks_completed']}个任务, 质量={r1['quality']:.2f}")

# 复杂目标
print("\n📌 复杂目标: '写加拿大市场分析报告'")
a1 = agent.execute("写报告")
print(f"   Agent: 完成{a1['tasks_completed']}个任务, 质量={a1['quality']:.2f}, "
      f"记忆项={a1['memory_used']}")

# ============================================================
# 3. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 3.1 能力对比
capabilities = ['单任务', '多步骤', '规划', '记忆', '反思', '行动']
react_scores = [0.8, 0.2, 0.1, 0.1, 0.1, 0.8]
agent_scores = [0.7, 0.8, 0.7, 0.7, 0.6, 0.7]

x = np.arange(len(capabilities))
width = 0.35
axes[0].bar(x - width/2, react_scores, width, label='ReAct', color='#ff6b6b', alpha=0.8)
axes[0].bar(x + width/2, agent_scores, width, label='Agent', color='#4ecdc4', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(capabilities, fontsize=9)
axes[0].set_ylabel('能力评分')
axes[0].set_title('ReAct vs Agent：能力对比')
axes[0].legend()

# 3.2 目标复杂度 vs 完成率
complexities = [1, 2, 3, 5, 8, 10]
react_completion = [0.85, 0.60, 0.40, 0.20, 0.10, 0.05]
agent_completion = [0.80, 0.75, 0.70, 0.60, 0.50, 0.40]

axes[1].plot(complexities, react_completion, 'o-', color='#ff6b6b',
            linewidth=2, label='ReAct')
axes[1].plot(complexities, agent_completion, 's-', color='#4ecdc4',
            linewidth=2, label='Agent')
axes[1].fill_between(complexities, react_completion, agent_completion,
                     alpha=0.2, color='#4ecdc4')
axes[1].set_xlabel('目标复杂度（子任务数）')
axes[1].set_ylabel('完成率')
axes[1].set_title('复杂度 vs 完成率')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3.3 Agent的四大核心能力
cores = ['规划\n(Planning)', '记忆\n(Memory)', '反思\n(Reflection)', '行动\n(Action)']
importance = [0.85, 0.75, 0.70, 0.90]
difficulty = [0.80, 0.65, 0.75, 0.50]

axes[2].bar(x, importance, 0.35, label='重要性', color='#4ecdc4', alpha=0.8)
axes[2].bar(x + 0.35, difficulty, 0.35, label='实现难度', color='#ff6b6b', alpha=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(cores, fontsize=9)
axes[2].set_ylabel('评分')
axes[2].set_title('Agent四大核心能力')
axes[2].legend(fontsize=8)

plt.suptitle('从ReAct到Agent：从"执行任务"到"完成目标"', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('from_react_to_agent.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 对比可视化已保存")
```

---

## 今日结语

Agent与ReAct的本质区别：**ReAct执行单个任务，Agent完成复杂目标**。Agent通过规划（分解目标）、记忆（跨任务存储）、反思（自我评估）、行动（调用工具）四大能力，将大目标分解为子任务序列并自主执行。

从ReAct到Agent的进化，就像从实习生到项目经理——不只是能力更强，而是**工作方式根本不同**：从"你问我答"变成"你给目标，我自主完成"。

明天，我们将深入Agent的架构设计。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Agent | 智能体 | 能自主完成复杂目标的AI系统 |
| Planning | 规划 | 将大目标分解为子任务序列 |
| Memory | 记忆 | 跨任务存储和检索信息 |
| Reflection | 反思 | 评估自己的输出质量 |
| Hierarchical MDP | 分层MDP | Agent的理论基础 |
| AutoGPT | AutoGPT | 第一个自主Agent系统 |
| Subtask | 子任务 | 大目标分解出的子步骤 |
