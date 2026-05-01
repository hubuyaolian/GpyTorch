# Day 04：工具使用概念 —— 让模型"动手"而不只是"动嘴"

> 🧠 第十七周 · 思维链与工具使用 · 第4天

思维链让模型学会了"想清楚再回答"，但它仍然只能生成文本——算不了大数、查不了资料、跑不了代码。你让它算"√2的前100位小数"，它只能编；你让它查"2024年诺贝尔物理学奖得主"，它只能猜。**工具使用（Tool Use）**的核心思想：**让模型调用外部工具来弥补自身能力的不足**。

**今天的任务**：
1. 理解工具使用的核心动机：模型能力的边界与扩展
2. 掌握工具使用的三种范式：API调用、代码执行、检索增强
3. 分析工具使用与思维链的互补关系

---

## 1. 历史剧场：从"只会说话"到"会动手"

### 🎭 大模型的能力边界

```
模型擅长的:
  ✅ 文本生成、翻译、摘要
  ✅ 知识问答（训练数据覆盖的）
  ✅ 简单推理（思维链辅助）
  ✅ 代码生成（但不执行）

模型不擅长的:
  ❌ 精确计算（345.678 × 912.345 = ?）
  ❌ 实时信息（今天天气、最新新闻）
  ❌ 代码执行（写了代码但跑不了）
  ❌ 数据库查询（无法访问外部数据）
  ❌ 图像处理（纯文本模型）
```

### 🎭 Toolformer的突破（2023.02）

2023年2月，Schick等人发表"Toolformer: A Language Model Can Teach Itself to Use Tools"，核心发现：

> 让模型学会在合适的位置插入API调用，自动决定**何时**调用工具、**调用哪个**工具、**如何解析**返回结果。

```
普通模型:
  用户: "2023年奥斯卡最佳影片是什么？"
  模型: "《瞬息全宇宙》"  ← 可能正确，也可能幻觉

Toolformer:
  用户: "2023年奥斯卡最佳影片是什么？"
  模型: [Wikipedia("2023 Academy Award Best Picture")] → "《瞬息全宇宙》"
  ← 调用Wikipedia API获取准确信息
```

### 🎭 工具使用的三大范式

```
范式1: API调用（Toolformer, ChatGPT Plugins）
  模型 → 生成API调用 → 执行 → 返回结果 → 继续生成

范式2: 代码执行（Code Interpreter, PAL）
  模型 → 生成代码 → 执行代码 → 返回输出 → 继续生成

范式3: 检索增强（RAG, RETRO）
  模型 → 生成查询 → 检索文档 → 拼接上下文 → 继续生成
```

---

## 2. 生活隐喻：学者 vs 工匠

### 🎓 纯语言模型 = 书斋学者

```
学者: "根据我的知识，√2 ≈ 1.41421356..."
  → 只能凭记忆，精度有限

学者: "2024年的诺贝尔奖...我记得好像是..."
  → 记忆可能过时或错误

学者: "这段代码的运行结果应该是..."
  → 只能推演，不能实际运行
```

### 🔧 工具增强模型 = 带工具箱的工匠

```
工匠: "让我用计算器算一下 √2"
  → [Calculator] √2 = 1.414213562373095048801688724209...
  → 精确到任意位数

工匠: "让我查一下最新资料"
  → [Search] 2024 Nobel Physics → ...
  → 获取实时准确信息

工匠: "让我跑一下这段代码"
  → [Python] execute(code) → output
  → 获得真实运行结果
```

### 🔑 关键洞察

| | 学者（纯模型） | 工匠（工具增强） |
|---|---|---|
| 知识来源 | 训练数据（可能过时） | 训练数据 + 实时工具 |
| 计算能力 | 近似（可能出错） | 精确（工具保证） |
| 信息时效 | 截止训练时间 | 实时 |
| 可信度 | 不确定（幻觉风险） | 高（工具验证） |
| 灵活性 | 高（任何话题） | 中（受限于可用工具） |

---

## 3. 数学直觉：工具使用的形式化

### 📐 纯语言模型的生成

$$y = f_\theta(x)$$

模型只能基于输入 $x$ 和参数 $\theta$ 生成输出 $y$。

### 📐 工具增强的生成

$$y = f_\theta(x, r_1, r_2, \ldots, r_K)$$

其中 $r_k = \text{Tool}_k(q_k)$ 是第 $k$ 次工具调用的结果，$q_k = g_\theta(x, r_1, \ldots, r_{k-1})$ 是模型生成的查询。

### 📐 工具调用的决策

模型需要学习三个决策：

```
决策1: 何时调用工具
  P(call_tool | context) → 是否需要工具辅助

决策2: 调用哪个工具
  P(tool_id | context) → 选择最合适的工具

决策3: 如何构造查询
  q = g_θ(context) → 生成有效的工具查询
```

### 📐 工具使用的误差分析

```
纯模型误差:
  ε_model = |f_θ(x) - y_true|
  → 对于计算类问题，ε可能很大

工具增强误差:
  ε_tool = |f_θ(x, Tool(q)) - y_true|
  → 如果Tool精确，ε_tool ≈ |f_θ(x, r_exact) - y_true|
  → 误差主要来自"如何使用工具结果"，而非"工具本身"

关键: 工具将误差从"计算不准确"转移到"使用不恰当"
  → 后者更容易通过训练改善
```

---

## 4. 代码实验室：工具使用概念演示

```python
"""
工具使用概念演示：纯模型 vs 工具增强
==================================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 模拟：纯模型 vs 工具增强
# ============================================================
class PureModel:
    """纯语言模型：只能凭"记忆"回答"""

    def calculate(self, expression: str) -> dict:
        """模拟模型的计算能力（不精确）"""
        # 模型对简单计算较准，复杂计算误差大
        try:
            true_value = eval(expression)
        except Exception:
            true_value = 0

        # 模拟误差：随表达式复杂度增长
        complexity = len(expression)
        relative_error = 0.01 * complexity  # 1% × 复杂度
        noise = np.random.normal(0, relative_error * abs(true_value))
        model_value = true_value + noise

        return {
            'method': '纯模型',
            'value': model_value,
            'true_value': true_value,
            'error': abs(model_value - true_value),
            'relative_error': abs(model_value - true_value) / (abs(true_value) + 1e-10)
        }

    def lookup(self, query: str) -> dict:
        """模拟模型的知识检索（可能幻觉）"""
        # 模型有50%概率给出正确信息
        correct = np.random.random() < 0.5
        return {
            'method': '纯模型',
            'correct': correct,
            'confidence': 0.8  # 模型总是很自信
        }


class ToolAugmentedModel:
    """工具增强模型：可以调用外部工具"""

    def calculate(self, expression: str) -> dict:
        """调用计算器工具（精确）"""
        try:
            true_value = eval(expression)
        except Exception:
            true_value = 0

        return {
            'method': '工具增强',
            'value': true_value,
            'true_value': true_value,
            'error': 0.0,
            'relative_error': 0.0
        }

    def lookup(self, query: str) -> dict:
        """调用搜索工具（准确）"""
        return {
            'method': '工具增强',
            'correct': True,
            'confidence': 0.99
        }

    def execute_code(self, code: str) -> dict:
        """调用代码执行器"""
        try:
            # 安全起见，只模拟执行
            return {
                'method': '工具增强',
                'success': True,
                'output': '代码执行结果'
            }
        except Exception as e:
            return {
                'method': '工具增强',
                'success': False,
                'output': str(e)
            }


# ============================================================
# 2. 批量测试
# ============================================================
print("=" * 60)
print("🔧 纯模型 vs 工具增强：计算准确率对比")
print("=" * 60)

np.random.seed(42)
pure = PureModel()
tool = ToolAugmentedModel()

# 不同复杂度的计算题
expressions = [
    ("2 + 3", "2+3"),
    ("12 × 15", "12*15"),
    ("345 × 789", "345*789"),
    ("√2 × π", "math.sqrt(2)*math.pi"),
    ("2^20 + 3^15", "2**20+3**15"),
    ("sin(π/4) × √2", "math.sin(math.pi/4)*math.sqrt(2)"),
]

print(f"\n{'表达式':<20} {'纯模型误差':>12} {'工具增强误差':>14}")
print("-" * 50)

pure_errors = []
tool_errors = []

for name, expr in expressions:
    p_result = pure.calculate(expr)
    t_result = tool.calculate(expr)
    pure_errors.append(p_result['relative_error'])
    tool_errors.append(t_result['relative_error'])
    print(f"{name:<20} {p_result['relative_error']:>12.4f} {t_result['relative_error']:>14.4f}")

# ============================================================
# 3. 知识检索对比
# ============================================================
print("\n" + "=" * 60)
print("🔍 知识检索：准确率对比")
print("=" * 60)

n_queries = 500
pure_correct = sum(1 for _ in range(n_queries) if pure.lookup("query")['correct'])
tool_correct = sum(1 for _ in range(n_queries) if tool.lookup("query")['correct'])

print(f"\n纯模型准确率:   {pure_correct/n_queries:.1%}")
print(f"工具增强准确率: {tool_correct/n_queries:.1%}")

# ============================================================
# 4. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 4.1 计算误差对比
x = range(len(expressions))
width = 0.35
axes[0].bar([i - width/2 for i in x], pure_errors, width,
           label='纯模型', color='#ff6b6b', alpha=0.8)
axes[0].bar([i + width/2 for i in x], tool_errors, width,
           label='工具增强', color='#4ecdc4', alpha=0.8)
axes[0].set_xticks(list(x))
axes[0].set_xticklabels([n for n, _ in expressions], fontsize=9, rotation=15)
axes[0].set_ylabel('相对误差')
axes[0].set_title('计算误差：纯模型 vs 工具增强')
axes[0].legend()

# 4.2 能力对比雷达图
categories = ['精确计算', '实时信息', '代码执行', '文本生成', '推理能力']
pure_scores = [0.3, 0.2, 0.0, 0.9, 0.7]
tool_scores = [0.95, 0.9, 0.85, 0.9, 0.7]

x_radar = np.arange(len(categories))
axes[1].bar(x_radar - 0.2, pure_scores, 0.35,
           label='纯模型', color='#ff6b6b', alpha=0.8)
axes[1].bar(x_radar + 0.2, tool_scores, 0.35,
           label='工具增强', color='#4ecdc4', alpha=0.8)
axes[1].set_xticks(x_radar)
axes[1].set_xticklabels(categories, fontsize=9)
axes[1].set_ylabel('能力评分')
axes[1].set_title('能力对比：工具弥补短板')
axes[1].legend(fontsize=8)
axes[1].set_ylim(0, 1.1)

# 4.3 三种工具范式
paradigms = ['API调用\n(Toolformer)', '代码执行\n(Code Interpreter)', '检索增强\n(RAG)']
accuracy = [0.85, 0.95, 0.80]
flexibility = [0.7, 0.9, 0.6]
safety = [0.8, 0.5, 0.9]

x_p = np.arange(len(paradigms))
width = 0.25
axes[2].bar(x_p - width, accuracy, width, label='准确率', color='#4ecdc4', alpha=0.8)
axes[2].bar(x_p, flexibility, width, label='灵活性', color='#45b7d1', alpha=0.8)
axes[2].bar(x_p + width, safety, width, label='安全性', color='#9b59b6', alpha=0.8)
axes[2].set_xticks(x_p)
axes[2].set_xticklabels(paradigms, fontsize=9)
axes[2].set_ylabel('评分')
axes[2].set_title('三种工具范式对比')
axes[2].legend(fontsize=8)

plt.suptitle('工具使用：让模型"动手"而不只是"动嘴"', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('tool_use_concept.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 工具使用概念可视化已保存")
```

### 运行结果解读

工具增强模型在精确计算上误差为零（计算器保证），在实时信息检索上准确率从50%提升到99%。三种工具范式中，代码执行最灵活但安全性最低，检索增强最安全但灵活性有限，API调用是折中方案。

---

## 今日结语

工具使用的核心思想：**让模型调用外部工具来弥补自身能力的不足**。纯语言模型是"书斋学者"——知识渊博但手无缚鸡之力；工具增强模型是"带工具箱的工匠"——既能思考又能动手。

三种工具范式各有优劣：
1. **API调用**：适合结构化信息获取（天气、股票、地图）
2. **代码执行**：适合精确计算和数据处理（数学、统计、可视化）
3. **检索增强**：适合知识密集型任务（问答、摘要、写作）

明天，我们将深入函数调用（Function Calling）的具体实现——ChatGPT和开源模型如何让模型学会生成结构化的工具调用。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Tool Use | 工具使用 | 模型调用外部工具扩展能力 |
| Toolformer | Toolformer | 让模型自学使用工具的方法 |
| Function Calling | 函数调用 | 模型生成结构化的API调用 |
| Code Interpreter | 代码解释器 | 执行模型生成的代码 |
| RAG | 检索增强生成 | Retrieval-Augmented Generation |
| API | 应用编程接口 | Application Programming Interface |
| Tool Augmented | 工具增强 | 配备外部工具的模型 |
| Hallucination | 幻觉 | 模型生成虚假信息 |
