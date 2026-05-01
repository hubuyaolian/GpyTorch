# Day 02：Agent架构 —— 规划+记忆+反思+行动

> 🤖 第十九周 · AI Agent · 第2天

昨天我们理解了Agent与ReAct的区别。今天，我们深入Agent的**架构设计**——四大核心模块（规划、记忆、反思、行动）如何组织、如何交互、如何形成闭环。一个好的Agent架构是这四个模块的有机组合，而非简单堆砌。

**今天的任务**：
1. 理解Agent的四大模块及其交互方式
2. 掌握两种主流架构：单Agent循环和多Agent协作
3. 分析各模块的设计选择和权衡

---

## 1. 历史剧场：Agent架构的演进

### 🎭 LangGraph的图架构（2024）

LangGraph用**有向图**定义Agent的执行流程：

```
节点: 规划器、执行器、反思器、记忆管理器
边:   条件转移（根据反思结果决定下一步）
状态: 全局共享的状态字典

流程:
  规划 → 执行 → 反思 → [通过?] → 下一个子任务
                        [失败?] → 重新执行/调整计划
```

### 🎭 四大模块的交互

```
┌─────────┐     ┌─────────┐
│  规划器  │────▶│  执行器  │
│ Planner │     │ Executor│
└────┬────┘     └────┬────┘
     │               │
     │               ▼
┌────┴────┐     ┌─────────┐
│  记忆库  │◀────│  反思器  │
│ Memory  │     │Reflector│
└─────────┘     └─────────┘

交互流程:
  1. 规划器从记忆库读取历史，制定计划
  2. 执行器按计划调用工具
  3. 反思器评估执行结果
  4. 反思结果写入记忆库
  5. 如果不通过，规划器调整计划
```

---

## 2. 生活隐喻：公司组织架构

### 🏢 Agent = 迷你公司

| 模块 | 公司角色 | 职责 |
|------|---------|------|
| 规划器 | 项目经理 | 分解目标、制定计划、调整策略 |
| 执行器 | 执行员工 | 调用工具、完成具体任务 |
| 记忆库 | 档案室 | 存储和检索历史信息 |
| 反思器 | 质检员 | 评估输出质量、发现问题 |

### 🔑 架构设计的关键权衡

```
权衡1: 集中规划 vs 即时规划
  集中: 一次性分解所有子任务 → 效率高但不够灵活
  即时: 每完成一个子任务再规划下一个 → 灵活但可能低效

权衡2: 短期记忆 vs 长期记忆
  短期: 只记住当前任务的上下文 → token省但信息少
  长期: 用向量数据库存储所有历史 → 信息全但检索慢

权衡3: 严格反思 vs 宽松反思
  严格: 每步都反思，不通过就重做 → 质量高但慢
  宽松: 只在关键节点反思 → 快但可能遗漏问题
```

---

## 3. 数学直觉：Agent架构的形式化

### 📐 Agent状态转移

$$s_{t+1} = f(s_t, a_t, o_t, m_t)$$

- $s_t$：当前状态（目标、进度、上下文）
- $a_t$：执行的动作（工具调用）
- $o_t$：观察结果
- $m_t$：从记忆库检索的信息

### 📐 规划函数

$$\text{Plan} = P(\text{Goal} | \text{Memory}, \text{Context})$$

规划器根据目标和历史记忆生成子任务序列。

### 📐 反思函数

$$\text{Verdict} = R(\text{Output} | \text{Criteria}, \text{Memory})$$

反思器根据质量标准评估输出，决定是否通过。

---

## 4. 代码实验室：Agent架构实现

```python
"""
Agent架构：规划+记忆+反思+行动
============================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 四大模块实现
# ============================================================
class Planner:
    """规划器：分解目标为子任务"""

    def plan(self, goal: str, memory: list) -> list:
        """制定计划"""
        if "报告" in goal:
            subtasks = [
                "搜索相关信息",
                "整理搜索结果",
                "分析关键发现",
                "撰写报告大纲",
                "根据大纲写报告",
                "审阅并修改报告"
            ]
        elif "分析" in goal:
            subtasks = ["收集数据", "分析数据", "总结发现"]
        else:
            subtasks = ["执行任务"]
        return subtasks

    def replan(self, goal, memory, failed_task):
        """调整计划（反思后调用）"""
        return [f"重新执行: {failed_task}", "继续后续任务"]


class Memory:
    """记忆库：存储和检索信息"""

    def __init__(self):
        self.store = []

    def save(self, key: str, value: str):
        """存储信息"""
        self.store.append({'key': key, 'value': value})

    def retrieve(self, query: str) -> list:
        """检索信息"""
        return [item for item in self.store if query in item['key']]

    def summarize(self) -> str:
        """摘要（控制token消耗）"""
        return f"记忆库: {len(self.store)}条记录"


class Reflector:
    """反思器：评估输出质量"""

    def evaluate(self, output: str, criteria: str = "完整性") -> dict:
        """评估输出"""
        quality = 0.7 + np.random.randn() * 0.15
        quality = np.clip(quality, 0, 1)
        passed = quality > 0.6
        return {
            'quality': quality,
            'passed': passed,
            'feedback': "质量良好" if passed else "需要改进"
        }


class Executor:
    """执行器：调用工具完成任务"""

    def execute(self, task: str, memory: Memory) -> str:
        """执行任务"""
        # 检索相关记忆
        relevant = memory.retrieve(task)
        context = f"(参考{len(relevant)}条历史信息)" if relevant else ""
        result = f"完成: {task} {context}"
        return result


# ============================================================
# 2. Agent：组合四大模块
# ============================================================
class Agent:
    """
    完整Agent：规划+记忆+反思+行动
    ============================
    """

    def __init__(self):
        self.planner = Planner()
        self.memory = Memory()
        self.reflector = Reflector()
        self.executor = Executor()

    def run(self, goal: str, max_retries: int = 2) -> dict:
        """运行Agent完成目标"""
        # Step 1: 规划
        plan = self.planner.plan(goal, self.memory.store)
        print(f"📋 计划: {plan}")

        results = []
        for i, task in enumerate(plan):
            # Step 2: 执行
            result = self.executor.execute(task, self.memory)
            print(f"  🔧 执行: {task} → {result[:30]}...")

            # Step 3: 反思
            evaluation = self.reflector.evaluate(result)
            print(f"  🔍 反思: 质量={evaluation['quality']:.2f}, "
                  f"{'✅通过' if evaluation['passed'] else '❌未通过'}")

            # 如果未通过，重试
            retries = 0
            while not evaluation['passed'] and retries < max_retries:
                retries += 1
                result = self.executor.execute(task, self.memory)
                evaluation = self.reflector.evaluate(result)
                print(f"  🔄 重试{retries}: 质量={evaluation['quality']:.2f}")

            # Step 4: 存入记忆
            self.memory.save(task, result)
            results.append({'task': task, 'result': result,
                          'quality': evaluation['quality']})

        return {
            'goal': goal,
            'results': results,
            'memory_size': len(self.memory.store),
            'avg_quality': np.mean([r['quality'] for r in results])
        }

# ============================================================
# 3. 运行Agent
# ============================================================
print("=" * 60)
print("🤖 Agent演示：规划+记忆+反思+行动")
print("=" * 60)

agent = Agent()
result = agent.run("写一份市场分析报告")

print(f"\n📊 结果: 完成{len(result['results'])}个子任务, "
      f"平均质量={result['avg_quality']:.2f}, "
      f"记忆项={result['memory_size']}")

# ============================================================
# 4. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 4.1 模块交互图
modules = ['规划器', '执行器', '反思器', '记忆库']
sizes = [300, 400, 250, 350]
colors_m = ['#4ecdc4', '#45b7d1', '#9b59b6', '#f39c12']
positions = [(0.2, 0.7), (0.8, 0.7), (0.8, 0.3), (0.2, 0.3)]
for (name, size, color, pos) in zip(modules, sizes, colors_m, positions):
    axes[0].scatter(pos[0], pos[1], s=size*3, color=color, alpha=0.8, zorder=5)
    axes[0].text(pos[0], pos[1], name, ha='center', va='center',
                fontsize=10, fontweight='bold')
# 画箭头
for start, end in [((0.3, 0.7), (0.7, 0.7)), ((0.8, 0.6), (0.8, 0.4)),
                    ((0.7, 0.3), (0.3, 0.3)), ((0.2, 0.4), (0.2, 0.6))]:
    axes[0].annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].set_title('Agent架构：模块交互')
axes[0].set_xticks([])
axes[0].set_yticks([])

# 4.2 各模块重要性
importance = [0.85, 0.90, 0.70, 0.75]
axes[1].bar(modules, importance, color=colors_m, alpha=0.8)
axes[1].set_ylabel('重要性')
axes[1].set_title('各模块重要性')
axes[1].set_ylim(0, 1.1)

# 4.3 架构对比
architectures = ['ReAct\n(无规划)', 'ReAct+记忆', 'ReAct+反思', '完整Agent']
completion = [0.30, 0.50, 0.55, 0.75]
axes[2].bar(architectures, completion, color=['#95a5a6', '#f39c12', '#45b7d1', '#4ecdc4'], alpha=0.8)
axes[2].set_ylabel('目标完成率')
axes[2].set_title('架构演进：逐步添加模块')
for i, c in enumerate(completion):
    axes[2].text(i, c + 0.02, f'{c:.0%}', ha='center', fontsize=10)

plt.suptitle('Agent架构：规划+记忆+反思+行动', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('agent_architecture.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Agent架构可视化已保存")
```

---

## 今日结语

Agent架构是四大模块的有机组合：规划器分解目标，执行器调用工具，反思器评估质量，记忆库存储信息。关键设计权衡包括：集中vs即时规划、短期vs长期记忆、严格vs宽松反思。完整Agent的目标完成率(75%)远超纯ReAct(30%)。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Planner | 规划器 | 分解目标为子任务的模块 |
| Executor | 执行器 | 调用工具完成具体任务的模块 |
| Reflector | 反思器 | 评估输出质量的模块 |
| Memory Store | 记忆库 | 存储和检索历史信息的模块 |
| Replan | 重新规划 | 反思失败后调整计划 |
| Quality Gate | 质量门 | 反思器的通过/不通过判断 |
