---
name: course-writing
description: AI进化史课程编写规范 — Jupyter Notebook 教学内容的结构、代码风格、验证标准。用于编写新课程或审查已有课程。
origin: project
---

# AI 进化史课程编写规范

本规范从 Week01-Week02 的教学实践中提炼，用于统一后续课程的编写质量。

## 何时激活

- 编写新的每周 Notebook 课程
- 审查已有课程内容
- 从 Markdown 源文件转换 Notebook
- 修复课程中的 Bug

## 章节结构（固定模板）

每个 Notebook / Markdown 章节必须遵循以下结构：

```
1. 历史背景（Markdown）— 这个时代遇到了什么痛点
2. 生活隐喻（Markdown）— 用类比解释数学概念（公司/山脉/水管/攀岩）
3. 数学直觉（Markdown）— LaTeX 公式 + 直觉解释 + 图示
4. 代码实现（Code）— 从零手写，验证前面的解释
5. 验证实验（Code）— 用代码证明概念
6. 翻译词典（Markdown）— 生活直觉 → 深度学习术语映射表
7. 总结预告（Markdown）— 承上启下
```

**每个 section 可以有多个 Markdown cell**，展开解释概念，不是每节只许一个 cell。

### 结构要求

- **先讲后写**：每个概念先用充分的 Markdown 解释（类比、公式、原理），再用代码验证。Markdown 解释是主体，代码是辅助验证。代码演示必须始终存在，但读者可自主选择是否跳过。
- **代码紧跟概念**：讲完概念 A 后，如果 A 可以用代码演示，代码直接放在 A 后面。不要把所有代码堆到"代码实现"section。"代码实现"section 放完整的类定义（带详细注释），"验证实验"section 放最终的完整验证。
- **渐进式构建**：同一周内使用继承链逐层叠加能力（如 `Perceptron → PerceptronWithMemory`）
- **验证实验必须通过**：所有验证代码的输出必须显示正确结果，不得出现错误标记

## 代码规范

### 语言一致性（最高优先级）

**所有注释、print 输出、图表标题、轴标签必须使用中文。**

```python
# 正确
print("XOR 问题的四个点：")
plt.title('梯度下降：寻找最小值')
plt.xlabel('迭代次数')
plt.ylabel('损失')
status = "✓ 正确" if correct else "✗ 错误"

# 错误 — 中英文混杂
print("Step 1: Calculate hidden layer raw scores")
plt.title('Gradient Descent: Finding Minimum')
status = "✓ Correct" if correct else "✗ Wrong"
```

**禁止出现的英文模式**：
- `Step 1/2/3/4` → `第 1/2/3/4 步`
- `True/Prediction/Result` → `真实标签/预测值/结果`
- `Advantages/Disadvantages` → `优点/缺点`
- `Note:` → `注意：`
- `Effect of` → `的影响`
- `Validation` → `验证`
- `Different` (作形容词) → `不同`
- `Complete Training Pipeline` → `完整的训练流程`

### PyTorch 代码风格

```python
# 正确 — 底层张量运算，揭示数据流
z1 = torch.matmul(X, self.W1.T) + self.b1.T
h = (z1 > 0).float()

# 避免 — 高级封装掩盖底层细节（早期课程）
h = torch.nn.functional.relu(z1)
```

- 优先使用 `torch.matmul`、`torch.sigmoid` 等底层运算
- 早期课程（Week01-06）刻意避免 `nn.Linear`、`nn.Module` 等高级 API
- 后期课程（Week07+）可引入高级 API，但需先展示底层实现

### 逐行注释（必须）

代码块必须有**逐行中文注释**，解释"为什么"而非仅"是什么"：

```python
# 正确 — 注释解释意图和原因
self.W1 = torch.randn(input_size, hidden_size)  # 隐藏层权重，随机初始化
z1 = torch.matmul(x, self.W1) + self.b1  # 线性变换：输入 × 权重 + 偏置
a1 = sigmoid(z1)  # Sigmoid 激活，引入非线性

# 错误 — 只说"是什么"，不解释为什么
self.W1 = torch.randn(input_size, hidden_size)  # 初始化权重
z1 = torch.matmul(x, self.W1) + self.b1  # 计算 z1
a1 = sigmoid(z1)  # 激活
```

注释要求：
- **类和函数**：必须有 docstring，说明输入输出是什么（形状、类型、含义）
- **每一行**：都要有注释，解释这一步在做什么、为什么这样做
- **注释语言**：全部中文

### Cell 独立性（必须）

每个代码 cell 必须**独立可运行**，不能依赖上方 cell 的变量、函数或类：

```python
# 正确 — 每个 cell 自包含
import torch  # 每个 cell 自己 import

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

X_xor = torch.tensor([[0.0, 0.0], ...])  # 每个 cell 自己定义数据
y_pred = sigmoid(torch.matmul(X_xor, W) + b)

# 错误 — 借用上方 cell 的变量
y_pred = sigmoid(torch.matmul(X_xor, W) + b)  # X_xor、W、sigmoid 都没定义！
```

为什么必须独立：
- 用户在 Jupyter 中可能跳着运行 cell
- 用户可能单独查看某个 cell 的效果
- 依赖上方 cell 会导致单独运行时报错，影响学习体验

可以重复（imports、函数定义、数据定义），但不能借用。

### 代码块长度

- 单个代码块不超过 **80 行**
- 代码 cell 聚焦于验证概念，核心逻辑突出，不需要面面俱到
- 超长代码拆分为多个 cell，每个 cell 配一个 Markdown 解释

### 中文字体配置

每个 Notebook 的第一个代码 cell 必须包含：

```python
import matplotlib.pyplot as plt
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Helvetica Neue', 'Heiti SC']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass
```

## 教学模式

### 生活隐喻（必须）

每个核心概念必须有一个可感知的类比。类比必须在独立的 Markdown cell 中展开解释，不能只塞在代码注释里：

| 概念 | 好的隐喻 | 差的隐喻 |
|------|----------|----------|
| 权重与偏置 | 篮球迷 vs 社交蝴蝶 | 权重是参数 |
| 梯度下降 | 在山上找最陡的下坡路 | 沿负梯度方向更新 |
| 隐藏层 | 地球投影到平面 | 增加一层网络 |
| 损失函数 | GPS 说 3 公里，实际 5 公里 | 预测与真实的差距 |

### 验证实验（必须）

每个关键概念必须有代码验证：

```python
# XOR 验证 — 必须展示四个输入的预测结果
for i in range(4):
    x_str = f"({X[i,0].item()}, {X[i,1].item()})"
    true_label = int(y[i].item())
    pred_label = int(y_pred[i].item())
    status = "✓ 正确" if true_label == pred_label else "✗ 错误"
    print(f"{x_str}: 真实={true_label}, 预测={pred_label} {status}")
```

### 可视化（鼓励）

- 使用 `matplotlib` 可视化决策边界、损失曲线、梯度流
- 图表标题和轴标签必须中文
- 保存图片：`plt.savefig('xxx.png', dpi=150, bbox_inches='tight')`

## 常见 Bug 清单

### P0 — XOR 标签顺序

XOR 真值表，输入顺序 `[0,0], [1,1], [0,1], [1,0]` 时：

```python
# 正确 XOR
y_xor = torch.tensor([0.0, 0.0, 1.0, 1.0])  # (0,0)->0, (1,1)->0, (0,1)->1, (1,0)->1

# 错误 — 这是 XNOR
y_xor = torch.tensor([1.0, 1.0, 0.0, 0.0])  # (0,0)->1, (1,1)->1, (0,1)->0, (1,0)->0
```

**验证方法**：运行验证 cell，四个输出必须全部显示正确。

### P0 — 训练不收敛

训练 XOR 时的推荐配置：

```python
# 推荐配置
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = torch.nn.BCELoss()
epochs = 1000

# 如果仍不收敛，检查：
# 1. 标签是否正确（XOR vs XNOR）
# 2. 学习率是否合适（0.01 ~ 0.3）
# 3. 隐藏层维度是否足够（至少 4）
# 4. 激活函数是否正确（隐藏层 ReLU，输出层 Sigmoid）
```

### P1 — 注释语言混杂

编写完成后全局搜索以下英文模式并替换：

| 搜索 | 替换为 |
|------|--------|
| `Step \d` | `第 N 步` |
| `True` (标签上下文) | `真实标签` |
| `Prediction` | `预测值` |
| `Result` | `结果` |
| `Correct` | `正确` |
| `Wrong` | `错误` |
| `Advantages` | `优点` |
| `Disadvantages` | `缺点` |
| `Note:` | `注意：` |
| `Different` (形容词) | `不同` |

## 发布前检查清单

每个 Notebook 发布前必须逐项检查：

- [ ] **所有 cell 可执行**：从头到尾运行无报错
- [ ] **验证实验通过**：所有验证输出全部正确
- [ ] **训练收敛**：损失下降到合理值（XOR 应 < 0.05）
- [ ] **语言统一**：无中英文混杂的注释、print、图表标题
- [ ] **图片正常**：所有 matplotlib 图表正确显示
- [ ] **Markdown 渲染**：LaTeX 公式正确显示
- [ ] **承上启下**：开头回顾昨天内容，结尾预告明天内容
- [ ] **翻译词典**：核心术语有生活直觉到术语的映射

## 每周检测 Notebook

每周末包含一个 `WeekXX_检测_*.ipynb`，用于检验该周学习成果。新增课程时必须同步更新检测 Notebook。

## MD 到 Notebook 转换（Week07+）

Week07-Week12 的 Notebook 通过 `convert_md_to_ipynb.py` 从 Markdown 源文件自动转换：

1. Markdown 源文件遵循 `_chapter_template.md` 模板
2. 转换脚本自动注入 matplotlib 可视化图片
3. 转换后仍需运行检查清单验证

```bash
python convert_md_to_ipynb.py
python validate_imgs.py
```
