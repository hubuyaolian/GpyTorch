# Day 07：Agent实战 —— 用Agent完成真实任务

> 🤖 第二十周 · AI Agent · 第7天

昨天我们实现了MiniAgent框架。今天，我们用它完成几个**模拟的真实任务**，展示Agent在不同场景下的表现，并分析执行轨迹。

**今天的任务**：
1. 用Agent完成三种不同类型的任务
2. 分析Agent的执行轨迹和效率
3. 讨论Agent的局限和改进方向

---

## 1. 历史剧场：Agent的实际应用

### 🎭 Agent的三大应用场景

```
场景1: 研究助手
  目标: "调研量子计算最新进展"
  → 搜索论文、整理要点、写摘要

场景2: 编程助手
  目标: "开发一个Web爬虫"
  → 设计架构、写代码、测试、调试

场景3: 数据分析
  目标: "分析销售数据趋势"
  → 读取数据、统计分析、生成图表、写报告
```

---

## 2. 代码实验室：Agent实战

```python
"""
Agent实战：三种任务场景
====================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 模拟三种任务场景
# ============================================================
class TaskSimulator:
    """任务模拟器"""

    def research_task(self):
        """研究助手任务"""
        steps = ['搜索论文', '提取要点', '整理文献', '写摘要', '审阅']
        qualities = [0.7, 0.75, 0.8, 0.72, 0.85]
        return steps, qualities

    def coding_task(self):
        """编程助手任务"""
        steps = ['设计架构', '写核心代码', '写测试', '运行测试', '修复bug']
        qualities = [0.8, 0.65, 0.7, 0.6, 0.85]
        return steps, qualities

    def analysis_task(self):
        """数据分析任务"""
        steps = ['读取数据', '清洗数据', '统计分析', '生成图表', '写报告']
        qualities = [0.9, 0.85, 0.75, 0.8, 0.7]
        return steps, qualities

# ============================================================
# 2. 运行三种任务
# ============================================================
sim = TaskSimulator()

tasks = {
    '研究助手': sim.research_task(),
    '编程助手': sim.coding_task(),
    '数据分析': sim.analysis_task(),
}

print("=" * 60)
print("🤖 Agent实战：三种任务场景")
print("=" * 60)

for name, (steps, qualities) in tasks.items():
    print(f"\n📌 {name}:")
    for step, q in zip(steps, qualities):
        status = '✅' if q > 0.65 else '⚠️'
        print(f"  {status} {step}: 质量={q:.2f}")
    print(f"  平均质量: {np.mean(qualities):.2f}")

# ============================================================
# 3. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 3.1 三种任务的质量对比
task_names = list(tasks.keys())
avg_qualities = [np.mean(q) for _, q in tasks.values()]
axes[0].bar(task_names, avg_qualities,
           color=['#4ecdc4', '#45b7d1', '#9b59b6'], alpha=0.8)
axes[0].set_ylabel('平均质量')
axes[0].set_title('三种任务场景：平均质量')
axes[0].set_ylim(0, 1)

# 3.2 各步骤质量
for i, (name, (steps, qualities)) in enumerate(tasks.items()):
    color = ['#4ecdc4', '#45b7d1', '#9b59b6'][i]
    axes[1].plot(range(len(steps)), qualities, 'o-', color=color,
                linewidth=2, label=name, alpha=0.8)
axes[1].set_xlabel('步骤')
axes[1].set_ylabel('质量')
axes[1].set_title('各步骤质量变化')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# 3.3 Agent局限
limitations = ['幻觉\n(编造信息)', '循环\n(重复犯错)', '成本\n(token消耗)', '安全\n(越权操作)']
severity = [0.7, 0.5, 0.6, 0.8]
axes[2].bar(limitations, severity, color='#ff6b6b', alpha=0.8)
axes[2].set_ylabel('严重程度')
axes[2].set_title('Agent的四大局限')

plt.suptitle('Agent实战：用Agent完成真实任务', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('agent_in_action.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Agent实战可视化已保存")
```

---

## 今日结语

Agent在三种场景下表现各异：研究助手质量最稳定，编程助手在测试环节容易出问题，数据分析在报告撰写环节质量下降。Agent的四大局限——幻觉、循环、成本、安全——是当前研究的重点方向。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Research Agent | 研究助手Agent | 帮助调研和写摘要 |
| Coding Agent | 编程助手Agent | 帮助开发和调试代码 |
| Analysis Agent | 数据分析Agent | 帮助分析数据和生成报告 |
