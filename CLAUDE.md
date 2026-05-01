# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

《AI进化史：从感知机到Transformer的觉醒之路》— 一个为期 14 周的深度学习教学项目，通过 Jupyter Notebook 演示从感知机到 Transformer 再到 Agent 的核心架构演进。核心理念是"逻辑复现"：用精简的 PyTorch 代码还原伟大架构的本质，而非深陷底层数学推导。完整课程计划见 `00_AI进化史_总体大纲.md`。

另有独立子课程 `LLM_应用落地_求职冲刺班/`，涵盖 RAG、Agent、工程调优等 LLM 应用落地内容。

## 技术栈

- **语言**: Python
- **核心依赖**: PyTorch（张量运算 + Autograd 自动求导）
- **载体**: Jupyter Notebook (.ipynb)，Markdown 源文件 (.md)
- **风格**: 教学导向，中文注释，渐进式构建（先建空壳，再逐步添加权重、激活函数、反向传播）
- **目标平台**: Apple Silicon (MPS) 优化，同时兼容其他平台

## 行为准则

> 偏谨慎而非求快。简单任务自行判断，不必死板执行。

### 先想后写

- 明确说出你的假设，不确定时先问再动手。
- 存在多种解读时，列出让用户选，不要默默选一个。
- 有更简单的方案就主动提出，该推回时推回。
- 搞不清楚的地方停下来，说清楚哪里不懂，再问。

### 最简优先

- 最少代码解决问题，不做任何"以防万一"的扩展。
- 不为单次使用的代码搞抽象。
- 不加没被要求的灵活性/可配置性。
- 不为不可能的场景写错误处理。
- 200 行能 50 行搞定就重写。自问："高级工程师会觉得这过度设计吗？"是就简化。

### 最小化改动

- 只改必须改的，不要"顺手"重构邻近代码、注释或格式。
- 保持已有代码风格，即使你会写得不一样。
- 你的改动引入了未使用的导入/变量/函数时自行清理；发现已有的死代码只提不删。
- 判断标准：每一行改动都能直接追溯到用户的需求。

### 目标驱动

- 把任务转化为可验证的目标：如"添加验证" → "先写失败的测试，再实现通过"。
- 多步任务先列出简要计划，每步附验证方式：
  1. [步骤] → 验证：[检查项]
  2. [步骤] → 验证：[检查项]

## 内容生产流水线

部分 Notebook 从 Markdown 源文件自动转换（**不是手动编辑**）：

1. **Markdown 源文件** — `chapter_XX_*.md` 遵循 `_chapter_template.md` 模板结构（历史剧场 → 生活隐喻 → 数学直觉 → 代码实验室 → 今日结语 → 翻译词典）
2. **转换脚本** — `convert_md_to_ipynb.py` 将 MD 解析为 Notebook cell，并通过 `add_visual_diagrams()` 在特定章节锚点（如 `## 2. 生活隐喻`）自动注入 matplotlib 生成的 base64 可视化图片
3. **图片验证** — `validate_imgs.py` 扫描 Notebook 中内嵌 base64 图片的数量和大小

周次 1-3（感知机→CNN）和 6-11（ResNet→预训练）的 Notebook 为手动编辑，直接在 Jupyter 中创建。周次 4-5 待创建。周次 12-14 为 Markdown 源文件。

## 开发命令

```bash
# 从 Markdown 源文件批量生成 Notebook（Week07-Week12）
python convert_md_to_ipynb.py

# 验证 Notebook 中内嵌图片（当前只扫描 Week07-08）
python validate_imgs.py

# 启动 Jupyter Notebook（在项目根目录下运行）
jupyter notebook

# 运行单个 notebook（命令行方式）
jupyter nbconvert --to notebook --execute Week01/Day01_张量_深度学习的通用货币.ipynb

# 安装依赖（仅需 PyTorch + matplotlib + numpy）
pip install torch matplotlib numpy
```

## 课程结构

项目覆盖 **14 周** + 额外求职课程，完整计划见 `00_AI进化史_总体大纲.md`：

| 阶段 | 周次 | 形式 | 主题 |
|------|------|------|------|
| ❄️ 初春与寒冬 | 1 | Notebook（手动） | 感知机, XOR 危机 |
| ☀️ 破局与复兴 | 2 | Notebook（手动） | 隐藏层, 激活函数, 反向传播, MLP, 损失函数, 梯度下降 |
| 👁️ 视觉征服 | 3 | Notebook（手动） | 卷积, 卷积层, 池化, LeNet-5, 梯度消失 |
| 🔧 训练秘籍 | 4 | **待创建** | 优化器进阶, Dropout, BatchNorm, 学习率调度 |
| 🏗️ CNN进化论 | 5 | **待创建** | VGG, GoogLeNet, 1x1卷积, 全局平均池化 |
| 🌉 深度鸿沟 | 6 | Notebook（手动） | 退化问题, 残差连接 F(x)+x, ResNet |
| ⏳ 记忆诞生 | 7 | Notebook（手动） | RNN, 隐藏状态, LSTM, 门控机制 |
| 📝 序列到序列 | 8 | Notebook（手动） | 词嵌入, GRU, 文本生成, Seq2Seq |
| 🚀 注意力时代 | 9 | Notebook（手动） | Self-Attention Q/K/V, 多头注意力, 位置编码, Encoder |
| 🏛️ 完整Transformer | 10 | Notebook（手动） | Decoder, 完整 Encoder-Decoder, 翻译实战 |
| 📚 预训练革命 | 11 | Notebook（手动） | 预训练-微调, BERT, GPT, Scaling Laws, 涌现能力 |
| 🎯 对齐之路 | 12 | Markdown | 对齐问题, 指令微调, 奖励模型, RLHF, PPO, DPO |
| 🧠 思维与工具 | 13 | Markdown | 提示工程, CoT, 工具使用, ReAct, 推理模型 |
| 🤖 Agent时代 | 14 | Markdown | Agent 架构, 规划, 记忆, 反思, 多Agent |

周次 1-3 和 6-11 的 Notebook 为手动编辑；周次 4-5 待创建；周次 12-14 目前只有 Markdown 源文件。

## 编码规范

- **解释优先**: 每个概念先用充分的 Markdown 讲解（类比、公式、原理），代码作为验证辅助。理解原理后可跳过代码。
- **代码紧跟概念**: 讲完概念 A 后，如果 A 可以用代码演示，代码直接放在 A 后面。不要把所有代码堆到"代码实现"section。
- **逐行中文注释**: 代码块每一行都要有中文注释，解释"为什么这样做"。类和函数必须有 docstring，说明输入输出（形状、类型、含义）。
- **Cell 独立性**: 每个代码 cell 必须独立可运行——自己 import、自己定义函数/类/数据。可以重复，但不能借用上方 cell 的变量。
- **渐进式构建**: 同一周内 Notebook 中的类使用继承链逐层叠加能力（如 `Perceptron → PerceptronWithMemory → FullyFunctionalPerceptron`）
- **避免高级封装**: 优先使用 `torch.matmul`、`torch.sigmoid` 等底层张量运算，刻意避免 `nn.Linear`、`nn.Module` 等高级 API，以揭示底层数据流
- **梯度流程完整展示**: 前向传播 → Loss 计算 → `loss.backward()` → `torch.no_grad()` 下手动更新 → `.grad.zero_()` 清零
- **中文注释**: 所有注释和讲解使用中文，行内注释解释"为什么"而非"是什么"
- **禁止复杂语法**: 不使用三元运算符和复杂单行推导式，保证最高可读性
- **代码格式**: 符合 Flake8 规范，Google 风格 Docstring
- **中文字体配置**: matplotlib 使用 `plt.rcParams['font.sans-serif'] = ['STHeiti', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']`

## 教学设计模式

每个 Notebook / Markdown 章节遵循固定结构（每个 section 可以有多个 Markdown cell 展开解释）：
1. **历史背景** — 这个时代遇到了什么痛点
2. **生活隐喻** — 用类比（公司/山脉/水管/攀岩）解释数学概念
3. **数学直觉** — LaTeX 公式 + 直觉解释
4. **代码实现** — 从零手写，逐行中文注释，单个代码块不超过 80 行
5. **验证实验** — 用代码证明概念（如 XOR 不可解 → XOR 可解）
6. **翻译词典** — 生活直觉 → 深度学习术语的映射表
7. **总结预告** — 承上启下

**代码位置原则**：讲完概念 A 后，如果 A 可以用代码演示，代码直接紧跟 A 后面。"代码实现"section 放完整类定义，"验证实验"section 放最终验证。不要把所有代码堆到最后。

## 每周检测 Notebook

每周末包含一个 `WeekXX_检测_*.ipynb`，用于检验该周学习成果。新增 Notebook 时需要同步维护。
