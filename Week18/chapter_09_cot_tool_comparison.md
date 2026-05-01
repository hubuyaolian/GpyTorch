# Day 09：CoT与工具使用对比 —— 何时思考，何时动手

> 🧠 第十八周 · 思维链与工具使用 · 第9天

我们学习了CoT、工具使用和ReAct三种方法。但实际应用中，**不是所有问题都需要CoT，也不是所有问题都需要工具**。今天我们系统对比这三种方法的适用场景，建立"何时思考、何时动手"的决策框架。

**今天的任务**：
1. 建立问题分类框架：按知识依赖和推理复杂度分类
2. 对比CoT、工具使用、ReAct在不同问题类型上的效果
3. 制定方法选择的决策树

---

## 1. 历史剧场：不同方法的不同赛道

### 🎭 问题类型与方法匹配

```
类型1: 纯推理（不需要外部知识）
  例: "证明√2是无理数"
  最佳方法: CoT（纯推理，不需要工具）

类型2: 纯检索（不需要推理）
  例: "2024年奥斯卡最佳影片是什么？"
  最佳方法: 工具使用（查一下就行，不需要推理）

类型3: 推理+检索（需要两者）
  例: "蒙特利尔说法语的人口约多少？"
  最佳方法: ReAct（先查后算）

类型4: 创意生成（不需要精确性）
  例: "写一首关于春天的诗"
  最佳方法: 直接生成（CoT和工具都不需要）
```

---

## 2. 生活隐喻：不同任务用不同工具

### 🔧 工具箱类比

```
任务: 修一把椅子

简单拧螺丝: 只需螺丝刀 → 工具使用
复杂木工: 需要量尺寸+锯+打磨 → ReAct
设计新椅子: 需要创意 → 直接生成
证明椅子能承重: 需要数学推理 → CoT
```

### 🔑 决策树

```
问题是否需要精确事实？
├── 是 → 事实是否可能过时/幻觉？
│   ├── 是 → 使用工具（search/lookup）
│   └── 否 → 直接回答
└── 否 → 问题是否需要多步推理？
    ├── 是 → 推理是否需要外部信息？
    │   ├── 是 → ReAct（推理+工具）
    │   └── 否 → CoT（纯推理）
    └── 否 → 直接生成
```

---

## 3. 数学直觉：方法选择的形式化

### 📐 问题空间

每个问题可以用两个维度刻画：

- $k$：知识依赖度（0=不需要外部知识，1=完全依赖外部知识）
- $r$：推理复杂度（0=不需要推理，1=极复杂推理）

### 📐 方法选择规则

$$\text{Method}(k, r) = \begin{cases} \text{直接生成} & k < 0.3, r < 0.3 \\ \text{CoT} & k < 0.3, r \geq 0.3 \\ \text{工具使用} & k \geq 0.3, r < 0.3 \\ \text{ReAct} & k \geq 0.3, r \geq 0.3 \end{cases}$$

---

## 4. 代码实验室：方法选择决策框架

```python
"""
CoT vs 工具使用 vs ReAct：方法选择框架
=====================================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 问题分类器
# ============================================================
class MethodSelector:
    """方法选择器：根据问题特征推荐最佳方法"""

    def select(self, question: str) -> dict:
        """
        选择最佳方法
        ===========
        基于知识依赖度和推理复杂度
        """
        k = self._knowledge_dependency(question)
        r = self._reasoning_complexity(question)

        if k < 0.3 and r < 0.3:
            method = "直接生成"
        elif k < 0.3 and r >= 0.3:
            method = "CoT"
        elif k >= 0.3 and r < 0.3:
            method = "工具使用"
        else:
            method = "ReAct"

        return {
            'question': question,
            'knowledge_dep': k,
            'reasoning_comp': r,
            'method': method
        }

    def _knowledge_dependency(self, q: str) -> float:
        """评估知识依赖度"""
        high_k = ["天气", "新闻", "最新", "当前", "今天", "搜索", "查"]
        med_k = ["人口", "首都", "历史", "谁", "哪年"]
        if any(kw in q for kw in high_k):
            return 0.9
        elif any(kw in q for kw in med_k):
            return 0.5
        return 0.1

    def _reasoning_complexity(self, q: str) -> float:
        """评估推理复杂度"""
        high_r = ["证明", "推导", "为什么", "如何", "分析"]
        med_r = ["计算", "多少", "比较", "排序"]
        if any(kw in q for kw in high_r):
            return 0.8
        elif any(kw in q for kw in med_r):
            return 0.5
        return 0.1

# ============================================================
# 2. 运行分类
# ============================================================
print("=" * 60)
print("📊 方法选择决策框架")
print("=" * 60)

selector = MethodSelector()

test_questions = [
    "写一首关于春天的诗",
    "证明√2是无理数",
    "今天北京天气怎么样？",
    "蒙特利尔说法语的人口约多少？",
    "2 + 3等于多少？",
    "分析莎士比亚《哈姆雷特》的主题",
    "2024年诺贝尔物理学奖得主是谁？",
    "计算(125+375)×4-800÷2",
]

results = []
for q in test_questions:
    result = selector.select(q)
    results.append(result)
    print(f"\n📌 {q}")
    print(f"   知识依赖: {result['knowledge_dep']:.1f}, "
          f"推理复杂: {result['reasoning_comp']:.1f}")
    print(f"   → 推荐方法: {result['method']}")

# ============================================================
# 3. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 3.1 问题空间分布
method_colors = {
    '直接生成': '#95a5a6', 'CoT': '#ff6b6b',
    '工具使用': '#45b7d1', 'ReAct': '#4ecdc4'
}
for r in results:
    color = method_colors[r['method']]
    axes[0].scatter(r['knowledge_dep'], r['reasoning_comp'],
                   s=150, color=color, alpha=0.8, zorder=5)
    axes[0].annotate(r['question'][:6],
                    (r['knowledge_dep'], r['reasoning_comp']),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

# 画分界线
axes[0].axvline(x=0.3, color='gray', linestyle='--', alpha=0.5)
axes[0].axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('知识依赖度')
axes[0].set_ylabel('推理复杂度')
axes[0].set_title('问题空间分布')

# 添加方法标签
for method, color in method_colors.items():
    axes[0].scatter([], [], s=100, color=color, label=method)
axes[0].legend(fontsize=8, loc='upper left')

# 3.2 各方法准确率
methods = ['直接生成', 'CoT', '工具使用', 'ReAct']
task_types = ['创意', '推理', '检索', '混合']
acc_matrix = np.array([
    [0.80, 0.30, 0.20, 0.25],  # 直接生成
    [0.70, 0.75, 0.35, 0.50],  # CoT
    [0.40, 0.30, 0.90, 0.55],  # 工具使用
    [0.60, 0.70, 0.85, 0.80],  # ReAct
])

x = np.arange(len(task_types))
width = 0.2
for i, (method, color) in enumerate(zip(methods, method_colors.values())):
    axes[1].bar(x + i * width - 0.3, acc_matrix[i], width,
               label=method, color=color, alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(task_types)
axes[1].set_ylabel('准确率')
axes[1].set_title('各方法在不同任务上的准确率')
axes[1].legend(fontsize=7)

# 3.3 成本-效果权衡
costs = [50, 200, 150, 350]
best_acc = [0.80, 0.75, 0.90, 0.80]
for i, (method, color) in enumerate(zip(methods, method_colors.values())):
    axes[2].scatter(costs[i], best_acc[i], s=200, color=color,
                   alpha=0.8, zorder=5, label=method)
axes[2].set_xlabel('Token成本')
axes[2].set_ylabel('最佳准确率')
axes[2].set_title('成本-效果权衡')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.suptitle('CoT vs 工具使用 vs ReAct：何时思考，何时动手', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('cot_tool_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 方法对比可视化已保存")
```

---

## 今日结语

方法选择的核心原则：**匹配问题特征**。纯推理用CoT，纯检索用工具，推理+检索用ReAct，创意生成直接输出。没有"万能方法"，只有"最适方法"。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Knowledge Dependency | 知识依赖度 | 问题对外部知识的需求程度 |
| Reasoning Complexity | 推理复杂度 | 问题所需的推理步骤数 |
| Decision Tree | 决策树 | 方法选择的判断流程 |
| Problem Space | 问题空间 | 所有问题构成的二维空间 |
