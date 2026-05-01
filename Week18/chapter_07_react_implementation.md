# Day 07：ReAct实现 —— 把"侦探破案"写成代码

> 🧠 第十八周 · 思维链与工具使用 · 第7天

昨天我们理解了ReAct的框架。今天，我们实现一个完整的ReAct Agent，包括工具注册、提示模板、循环控制、错误处理和早停机制。这个实现将是一个可运行的迷你ReAct框架。

**今天的任务**：
1. 实现完整的ReAct Agent类
2. 添加错误处理和重试机制
3. 实现多轮对话的ReAct
4. 可视化：ReAct轨迹和性能分析

---

## 1. 历史剧场：ReAct在产品中的应用

### 🎭 LangChain的ReAct实现

LangChain是最流行的ReAct实现框架，核心组件：

```
Agent: 决定下一步做什么（Thought/Action）
Tool:  执行具体操作（search/calculate/lookup）
Executor: 管理循环（调用Agent→执行Tool→返回Observation）
```

### 🎭 ReAct的工程挑战

```
挑战1: 提示模板设计
  → 如何让模型稳定输出Thought/Action/Answer格式
  → 不同模型对格式的遵循度不同

挑战2: 错误处理
  → 工具调用可能失败
  → 模型可能生成无效的Action
  → 需要优雅的降级策略

挑战3: 循环控制
  → 何时停止循环？
  → 最大步数限制
  → 检测到"我有足够信息"时提前终止

挑战4: 上下文管理
  → 每轮的Thought/Action/Observation都消耗token
  → 长轨迹可能超出上下文窗口
  → 需要摘要或截断策略
```

---

## 2. 生活隐喻：项目经理的工作方式

### 📋 ReAct = 项目经理的"调研-执行"循环

```
项目: "评估是否应该进入日本市场"

轮1:
  Thought: "先了解日本市场规模"
  Action:  查询市场报告
  Observation: 日本市场规模约1.2万亿美元

轮2:
  Thought: "了解竞争格局"
  Action:  查询竞争对手数据
  Observation: 主要竞争对手: A公司(30%), B公司(20%)

轮3:
  Thought: "评估进入壁垒"
  Action:  查询法规要求
  Observation: 需要本地化认证，周期约6个月

轮4:
  Thought: "综合分析：市场大但竞争激烈，壁垒中等"
  Answer:  建议进入，但需做好6个月认证准备
```

---

## 3. 数学直觉：ReAct的收敛性

### 📐 信息增益视角

每轮ReAct的信息增益：

$$I_t = H(\text{Answer} | \text{context}_{t-1}) - H(\text{Answer} | \text{context}_t)$$

当 $I_t < \epsilon$（信息增益很小）时，应该停止循环。

### 📐 最优停止问题

ReAct的停止时机是一个**最优停止问题**：

$$T^* = \arg\min_T \mathbb{E}[\text{Error}(T)] + \lambda \cdot T$$

- $\text{Error}(T)$：$T$步后的答案误差（随$T$减小）
- $\lambda \cdot T$：每步的成本（token消耗、延迟）
- $T^*$：平衡准确率和成本的最优步数

---

## 4. 代码实验室：完整ReAct实现

```python
"""
完整ReAct Agent实现
==================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import re
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 工具注册系统
# ============================================================
class ToolRegistry:
    """工具注册表：管理所有可用工具"""

    def __init__(self):
        self.tools = {}

    def register(self, name: str, description: str, func):
        """注册工具"""
        self.tools[name] = {
            'name': name,
            'description': description,
            'func': func
        }

    def execute(self, name: str, **kwargs):
        """执行工具"""
        if name not in self.tools:
            return f"错误: 工具'{name}'不存在"
        try:
            result = self.tools[name]['func'](**kwargs)
            return result
        except Exception as e:
            return f"工具执行错误: {str(e)}"

    def get_descriptions(self) -> str:
        """获取所有工具的描述（用于prompt）"""
        desc = ""
        for name, tool in self.tools.items():
            desc += f"- {name}: {tool['description']}\n"
        return desc


# 注册工具
registry = ToolRegistry()
registry.register("search", "搜索信息，参数: query(str)",
                  lambda query: {
                      "蒙特利尔人口": "约420万",
                      "蒙特利尔法语比例": "约60%",
                      "地球到月球距离": "约384,400公里",
                  }.get(query, f"未找到'{query}'的信息"))
registry.register("calculate", "精确计算，参数: expression(str)",
                  lambda expression: f"={eval(expression)}" if expression else "空表达式")
registry.register("lookup", "查字典，参数: word(str)",
                  lambda word: f"'{word}'的定义: ...")

# ============================================================
# 2. ReAct Agent完整实现
# ============================================================
class ReActAgent:
    """
    完整ReAct Agent
    ==============
    支持：多轮循环、错误处理、早停、轨迹记录
    """

    def __init__(self, registry: ToolRegistry, max_steps: int = 8,
                 verbose: bool = True):
        self.registry = registry
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, question: str) -> dict:
        """
        运行ReAct循环
        =============
        """
        trajectory = []
        total_tokens = 0

        for step in range(1, self.max_steps + 1):
            # Step 1: 生成Thought
            thought = self._think(question, trajectory, step)
            trajectory.append({
                'step': step, 'type': 'thought', 'content': thought
            })
            total_tokens += len(thought) // 4  # 估算token数

            if self.verbose:
                print(f"  💭 Step {step} Thought: {thought}")

            # 检查是否可以给出答案
            if self._should_answer(thought):
                answer = self._extract_answer(thought, question, trajectory)
                trajectory.append({
                    'step': step, 'type': 'answer', 'content': answer
                })
                if self.verbose:
                    print(f"  ✅ Answer: {answer}")
                break

            # Step 2: 生成并执行Action
            action_name, action_args = self._act(thought, question, trajectory)
            trajectory.append({
                'step': step, 'type': 'action',
                'content': f"{action_name}({action_args})"
            })

            if self.verbose:
                print(f"  🔧 Action: {action_name}({action_args})")

            # Step 3: 执行并获取Observation
            observation = self.registry.execute(action_name, **action_args)
            trajectory.append({
                'step': step, 'type': 'observation',
                'content': str(observation)
            })
            total_tokens += len(str(observation)) // 4

            if self.verbose:
                print(f"  👁️ Observation: {observation}")

        return {
            'question': question,
            'trajectory': trajectory,
            'total_steps': step,
            'total_tokens': total_tokens,
            'answer': next(
                (t['content'] for t in trajectory if t['type'] == 'answer'),
                "未能得出答案"
            )
        }

    def _think(self, question, trajectory, step):
        """生成Thought"""
        observations = [t for t in trajectory if t['type'] == 'observation']

        if step == 1:
            if "人口" in question:
                return "我需要查人口数据来回答这个问题"
            elif "距离" in question:
                return "我需要查距离数据"
            elif "计算" in question:
                return "我需要用计算器精确计算"
            return "我需要搜索相关信息"

        if observations:
            last_obs = observations[-1]['content']
            if "420万" in last_obs and "法语" in question:
                return "查到人口420万，现在查法语比例，然后计算"
            elif "60%" in last_obs or "60" in last_obs:
                return "查到法语比例60%，计算420×0.6=252万。我有足够信息：约252万人"
            elif "384" in last_obs:
                return f"查到距离{last_obs}。我有足够信息给出答案"
            elif "=" in last_obs:
                return f"计算完成。我有足够信息给出答案"

        return "需要更多信息继续推理"

    def _should_answer(self, thought):
        """判断是否应该给出答案"""
        return "我有足够信息" in thought

    def _act(self, thought, question, trajectory):
        """生成Action"""
        if "查人口" in thought:
            return "search", {"query": "蒙特利尔人口"}
        elif "法语比例" in thought:
            return "search", {"query": "蒙特利尔法语比例"}
        elif "距离" in thought:
            return "search", {"query": "地球到月球距离"}
        elif "计算" in thought and "420" in thought:
            return "calculate", {"expression": "420 * 0.6"}
        elif "计算" in thought:
            return "calculate", {"expression": "1+1"}
        return "search", {"query": question}

    def _extract_answer(self, thought, question, trajectory):
        """从Thought和轨迹中提取答案"""
        observations = [t for t in trajectory if t['type'] == 'observation']
        if "252" in thought:
            return "蒙特利尔说法语的人口约252万（420万×60%）"
        for obs in reversed(observations):
            if obs['content'] and obs['content'] != "未找到信息":
                return f"根据查到的信息: {obs['content']}"
        return "无法确定答案"


# ============================================================
# 3. 运行演示
# ============================================================
print("=" * 60)
print("🔄 ReAct Agent 完整演示")
print("=" * 60)

agent = ReActAgent(registry, max_steps=8, verbose=True)

questions = [
    "蒙特利尔说法语的人口大约多少？",
    "地球到月球的距离是多少？",
]

results = []
for q in questions:
    print(f"\n📌 问题: {q}")
    print("-" * 40)
    result = agent.run(q)
    results.append(result)
    print()

# ============================================================
# 4. 性能分析可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 4.1 ReAct轨迹可视化
for i, result in enumerate(results):
    steps = [t for t in result['trajectory'] if t['type'] != 'answer']
    types = [t['type'] for t in steps]
    type_colors = {'thought': '#4ecdc4', 'action': '#45b7d1', 'observation': '#f39c12'}
    for j, (step, stype) in enumerate(zip(steps, types)):
        axes[0].barh(i * 10 + j, 1, color=type_colors[stype], alpha=0.8)
        label = stype[:3]
        axes[0].text(0.5, i * 10 + j, label, ha='center', va='center', fontsize=8)

axes[0].set_xlim(0, 1)
axes[0].set_title('ReAct轨迹')
axes[0].set_yticks([])

# 4.2 步骤数分布
step_counts = [r['total_steps'] for r in results]
axes[1].hist(step_counts, bins=range(1, 10), color='#4ecdc4', alpha=0.8, edgecolor='black')
axes[1].set_xlabel('步骤数')
axes[1].set_ylabel('频次')
axes[1].set_title('ReAct步骤数分布')

# 4.3 方法对比
methods = ['直接回答', '纯CoT', '纯行动', 'ReAct']
accuracy = [0.30, 0.45, 0.50, 0.75]
token_cost = [50, 150, 200, 300]

colors_m = ['#95a5a6', '#ff6b6b', '#f39c12', '#4ecdc4']
for i, (m, a, c, col) in enumerate(zip(methods, accuracy, token_cost, colors_m)):
    axes[2].scatter(c, a, s=200, color=col, alpha=0.8, zorder=5)
    axes[2].annotate(m, (c, a), textcoords="offset points",
                    xytext=(10, 5), fontsize=9)

axes[2].set_xlabel('Token消耗')
axes[2].set_ylabel('准确率')
axes[2].set_title('准确率 vs 成本')
axes[2].grid(True, alpha=0.3)

plt.suptitle('ReAct实现：完整的推理-行动循环', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('react_implementation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ ReAct实现可视化已保存")
```

---

## 今日结语

完整的ReAct实现包含五个核心组件：工具注册、Thought生成、Action解析与执行、Observation处理、循环控制与早停。工程上的关键挑战是提示模板设计（让模型稳定输出结构化格式）和错误处理（工具调用失败时的优雅降级）。

明天，我们将学习提示工程——如何设计高效的prompt来触发CoT、工具使用和ReAct。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Tool Registry | 工具注册表 | 管理所有可用工具的注册中心 |
| Early Stopping | 早停 | 检测到足够信息时提前终止循环 |
| Error Handling | 错误处理 | 工具调用失败时的降级策略 |
| Trajectory | 轨迹 | 完整的推理-行动历史记录 |
| Token Budget | Token预算 | 允许消耗的最大token数 |
| Prompt Template | 提示模板 | 触发模型输出特定格式的模板 |
