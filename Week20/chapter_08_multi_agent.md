# Day 08：多Agent协作 —— 团队作战

> 🤖 第二十周 · AI Agent · 第8天

单个Agent的能力有限，但**多个Agent协作**可以完成更复杂的任务——就像公司里不同部门协作完成项目。今天我们学习多Agent系统的设计：角色分工、通信机制、冲突解决。

**今天的任务**：
1. 理解多Agent协作的核心模式：分工、讨论、投票
2. 掌握三种协作架构：顺序、并行、层次
3. 用代码模拟多Agent协作

---

## 1. 历史剧场：多Agent框架

### 🎭 三种协作模式

```
模式1: 顺序协作（流水线）
  AgentA(搜索) → AgentB(分析) → AgentC(写作)
  → 简单但串行，效率低

模式2: 并行协作（分工）
  AgentA(搜索市场) ─┐
  AgentB(搜索技术) ─├→ AgentC(整合)
  AgentC(搜索竞争) ─┘
  → 并行但需要整合

模式3: 讨论协作（辩论）
  AgentA: "应该进入市场"
  AgentB: "不应该，风险太大"
  AgentC(裁判): 综合双方观点做决策
  → 质量高但成本高
```

### 🎭 代表性框架

```
CrewAI: 角色分工+顺序/并行执行
AutoGen: 多Agent对话+代码执行
LangGraph: 图结构+条件分支
ChatDev: 软件开发的多Agent模拟
```

---

## 2. 生活隐喻：公司团队

### 🏢 多Agent = 公司部门

| Agent角色 | 公司部门 | 职责 |
|----------|---------|------|
| 研究员 | 市场部 | 搜索信息、调研 |
| 分析师 | 分析部 | 数据分析、趋势判断 |
| 写作者 | 内容部 | 撰写报告、文档 |
| 审阅者 | 质检部 | 审查质量、提出修改 |

---

## 3. 代码实验室：多Agent协作

```python
"""
多Agent协作模拟
==============
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 多Agent系统
# ============================================================
class WorkerAgent:
    """工作Agent"""
    def __init__(self, name, role):
        self.name = name
        self.role = role

    def work(self, task, inputs=None):
        quality = 0.7 + np.random.randn() * 0.1
        return {'agent': self.name, 'role': self.role,
                'output': f"{self.role}完成{task}", 'quality': np.clip(quality, 0, 1)}

class CoordinatorAgent:
    """协调Agent"""
    def coordinate(self, agents, goal):
        results = []
        for agent in agents:
            result = agent.work(goal)
            results.append(result)
        return results

# ============================================================
# 2. 运行多Agent
# ============================================================
print("=" * 60)
print("👥 多Agent协作演示")
print("=" * 60)

agents = [
    WorkerAgent("研究员", "搜索调研"),
    WorkerAgent("分析师", "数据分析"),
    WorkerAgent("写作者", "内容撰写"),
    WorkerAgent("审阅者", "质量审查"),
]

coordinator = CoordinatorAgent()
results = coordinator.coordinate(agents, "市场分析报告")

for r in results:
    print(f"  {r['agent']}({r['role']}): 质量={r['quality']:.2f}")

# ============================================================
# 3. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 3.1 协作模式对比
modes = ['单Agent', '顺序协作', '并行协作', '讨论协作']
quality = [0.60, 0.72, 0.75, 0.82]
cost = [100, 300, 250, 500]
colors_m = ['#95a5a6', '#f39c12', '#45b7d1', '#4ecdc4']

for i, (m, q, c, col) in enumerate(zip(modes, quality, cost, colors_m)):
    axes[0].scatter(c, q, s=200, color=col, alpha=0.8, zorder=5)
    axes[0].annotate(m, (c, q), textcoords="offset points", xytext=(10, 5), fontsize=9)
axes[0].set_xlabel('Token成本')
axes[0].set_ylabel('输出质量')
axes[0].set_title('协作模式：成本 vs 质量')
axes[0].grid(True, alpha=0.3)

# 3.2 Agent角色分工
roles = ['研究员', '分析师', '写作者', '审阅者']
contribution = [0.25, 0.30, 0.28, 0.17]
axes[1].pie(contribution, labels=roles, colors=['#4ecdc4', '#45b7d1', '#9b59b6', '#f39c12'],
           autopct='%1.0f%%', startangle=90)
axes[1].set_title('各角色贡献占比')

# 3.3 协作效率
n_agents = [1, 2, 3, 4, 5, 6]
efficiency = [0.6, 0.72, 0.78, 0.82, 0.80, 0.75]  # 4个最优，多了反而下降
axes[2].plot(n_agents, efficiency, 'o-', color='#4ecdc4', linewidth=2, markersize=8)
axes[2].axvline(x=4, color='gray', linestyle='--', alpha=0.5)
axes[2].annotate('最优团队规模', xy=(4, 0.82), xytext=(5, 0.85),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))
axes[2].set_xlabel('Agent数量')
axes[2].set_ylabel('协作效率')
axes[2].set_title('团队规模 vs 效率（边际递减）')
axes[2].grid(True, alpha=0.3)

plt.suptitle('多Agent协作：团队作战', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('multi_agent.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 多Agent协作可视化已保存")
```

---

## 今日结语

多Agent协作通过角色分工提升整体质量。三种模式各有优劣：顺序简单但慢，并行快但需整合，讨论质量高但成本高。关键发现：**团队规模有最优值**——4-5个Agent通常最优，更多反而因通信开销降低效率。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Multi-Agent | 多Agent | 多个Agent协作的系统 |
| Coordinator | 协调者 | 管理其他Agent的协调Agent |
| Sequential | 顺序协作 | Agent按顺序执行 |
| Parallel | 并行协作 | Agent同时执行 |
| Debate | 讨论协作 | Agent通过辩论达成共识 |
