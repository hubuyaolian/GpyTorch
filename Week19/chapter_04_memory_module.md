# Day 04：记忆模块 —— Agent的"长期记忆"

> 🤖 第十九周 · AI Agent · 第4天

ReAct每轮推理都是独立的，没有跨任务记忆。但Agent需要**记住之前做过什么、发现了什么、犯了什么错**，才能在后续任务中利用这些信息。记忆模块就是Agent的"长期记忆"系统。

**今天的任务**：
1. 理解记忆的三个层次：工作记忆、短期记忆、长期记忆
2. 掌握记忆的存储和检索机制
3. 分析记忆管理的挑战：容量、检索、遗忘

---

## 1. 历史剧场：记忆系统的演进

### 🎭 三个记忆层次

```
工作记忆(Working Memory):
  → 当前对话的上下文窗口
  → 容量: 4K-128K tokens
  → 特点: 速度快，容量有限

短期记忆(Short-term Memory):
  → 当前任务执行过程中的中间结果
  → 容量: 数十条记录
  → 特点: 任务结束后可清除

长期记忆(Long-term Memory):
  → 向量数据库存储的历史经验
  → 容量: 无限
  → 特点: 需要检索，有延迟
```

---

## 2. 生活隐喻：人脑的记忆系统

### 🧠 三层记忆

| 记忆层次 | 人脑 | Agent |
|---------|------|-------|
| 工作记忆 | 正在想的事情（7±2项） | 当前上下文窗口 |
| 短期记忆 | 今天做过的事 | 当前任务的中间结果 |
| 长期记忆 | 多年的经验和知识 | 向量数据库 |

### 🔑 记忆的挑战

```
挑战1: 容量限制
  → 上下文窗口有限，不能塞入所有历史
  → 解决: 摘要、截断、检索

挑战2: 检索效率
  → 长期记忆量大，如何快速找到相关信息？
  → 解决: 向量相似度检索（RAG）

挑战3: 遗忘与混淆
  → 旧记忆可能过时，相似记忆可能混淆
  → 解决: 时间衰减、置信度加权
```

---

## 3. 数学直觉：记忆检索

### 📐 向量检索

长期记忆用向量数据库存储，检索时计算相似度：

$$\!m^* = \arg\max_{m \in \mathcal{M}} \text{sim}(\text{Embed}(q), \text{Embed}(m!))$$

### 📐 记忆衰减

旧记忆的权重随时间衰减：

$$w(m, t) = w_0 \cdot e^{-\lambda (t - t_{\text{created}})}$$

---

## 4. 代码实验室：记忆模块实现

```python
"""
记忆模块实现：三层记忆系统
========================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 三层记忆系统
# ============================================================
class WorkingMemory:
    """工作记忆：当前上下文"""

    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.content = []

    def add(self, text: str):
        self.content.append(text)
        # 超出容量时截断
        total = sum(len(c) for c in self.content)
        while total > self.max_tokens and self.content:
            self.content.pop(0)
            total = sum(len(c) for c in self.content)

    def get(self) -> str:
        return "\n".join(self.content)


class ShortTermMemory:
    """短期记忆：当前任务的中间结果"""

    def __init__(self):
        self.results = {}

    def save(self, task: str, result: str):
        self.results[task] = result

    def load(self, task: str) -> str:
        return self.results.get(task, "")

    def clear(self):
        self.results.clear()


class LongTermMemory:
    """长期记忆：向量数据库（简化模拟）"""

    def __init__(self, decay_rate: float = 0.01):
        self.memories = []
        self.decay_rate = decay_rate

    def save(self, key: str, value: str, timestamp: int = 0):
        self.memories.append({
            'key': key, 'value': value,
            'timestamp': timestamp, 'embed': np.random.randn(64)
        })

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """检索最相关的记忆"""
        query_embed = np.random.randn(64)
        scored = []
        for m in self.memories:
            sim = np.dot(query_embed, m['embed']) / (
                np.linalg.norm(query_embed) * np.linalg.norm(m['embed']) + 1e-8
            )
            # 时间衰减
            decay = np.exp(-self.decay_rate * m['timestamp'])
            scored.append((m, sim * decay))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, s in scored[:top_k]]


# ============================================================
# 2. 完整记忆系统
# ============================================================
class MemorySystem:
    """三层记忆系统"""

    def __init__(self):
        self.working = WorkingMemory()
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    def save_task_result(self, task: str, result: str, step: int = 0):
        self.short_term.save(task, result)
        self.long_term.save(task, result, timestamp=step)
        self.working.add(f"[{task}] {result[:100]}")

    def get_relevant(self, query: str) -> list:
        return self.long_term.retrieve(query)


# ============================================================
# 3. 演示
# ============================================================
print("=" * 60)
print("🧠 记忆模块演示")
print("=" * 60)

memory = MemorySystem()

# 模拟Agent执行过程中的记忆使用
tasks = [
    ("搜索市场数据", "市场规模: 1.2万亿美元, 增长率: 15%"),
    ("搜索竞争格局", "主要玩家: A(30%), B(20%), C(15%)"),
    ("分析趋势", "趋势: AI驱动增长, 移动化加速"),
    ("撰写报告", "报告初稿完成, 3页"),
]

for i, (task, result) in enumerate(tasks):
    memory.save_task_result(task, result, step=i)
    print(f"  📝 保存: {task} → {result[:30]}...")

# 检索相关记忆
print(f"\n🔍 检索'市场': 找到{len(memory.get_relevant('市场'))}条相关记忆")
print(f"   工作记忆: {len(memory.working.content)}条")
print(f"   短期记忆: {len(memory.short_term.results)}条")
print(f"   长期记忆: {len(memory.long_term.memories)}条")

# ============================================================
# 4. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 4.1 三层记忆容量
layers = ['工作记忆', '短期记忆', '长期记忆']
capacity = [4096, 50, 10000]
speed = [0.01, 0.05, 0.2]

axes[0].bar(layers, [np.log10(c) for c in capacity],
           color=['#4ecdc4', '#45b7d1', '#9b59b6'], alpha=0.8)
axes[0].set_ylabel('容量(log10 tokens)')
axes[0].set_title('三层记忆容量')

# 4.2 检索延迟 vs 容量
axes[1].scatter(capacity, speed, s=200, color=['#4ecdc4', '#45b7d1', '#9b59b6'], alpha=0.8, zorder=5)
for i, layer in enumerate(layers):
    axes[1].annotate(layer, (capacity[i], speed[i]),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)
axes[1].set_xlabel('容量')
axes[1].set_ylabel('检索延迟(秒)')
axes[1].set_title('容量 vs 延迟权衡')
axes[1].set_xscale('log')

# 4.3 记忆衰减
time = np.arange(0, 100)
for decay, label in [(0.01, 'λ=0.01(慢衰减)'), (0.05, 'λ=0.05(中衰减)'), (0.1, 'λ=0.1(快衰减)')]:
    axes[2].plot(time, np.exp(-decay * time), linewidth=2, label=label)
axes[2].set_xlabel('时间步')
axes[2].set_ylabel('记忆权重')
axes[2].set_title('记忆时间衰减')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.suptitle('记忆模块：Agent的"长期记忆"', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('memory_module.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 记忆模块可视化已保存")
```

---

## 今日结语

记忆模块让Agent能跨任务积累和利用信息。三层记忆各有分工：工作记忆处理当前上下文，短期记忆存储任务中间结果，长期记忆用向量数据库存储历史经验。关键挑战是容量限制和检索效率——RAG技术是解决长期记忆检索的主流方案。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Working Memory | 工作记忆 | 当前上下文窗口 |
| Short-term Memory | 短期记忆 | 当前任务的中间结果 |
| Long-term Memory | 长期记忆 | 向量数据库存储的历史 |
| Vector Database | 向量数据库 | 存储嵌入向量的数据库 |
| Memory Decay | 记忆衰减 | 旧记忆权重随时间降低 |
| RAG | 检索增强生成 | 从长期记忆检索相关信息 |
