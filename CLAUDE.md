# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

《AI进化史：从感知机到Transformer的觉醒之路》— 一个为期 20 周的深度学习教学项目，通过 Jupyter Notebook 演示从感知机到 Transformer 再到 Agent 的核心架构演进。核心理念是"逻辑复现"：用精简的 PyTorch 代码还原伟大架构的本质，而非深陷底层数学推导。

另有独立子课程 `LLM_应用落地_求职冲刺班/`，涵盖 RAG、Agent、工程调优等 LLM 应用落地内容。

## 技术栈

- **语言**: Python
- **核心依赖**: PyTorch（张量运算 + Autograd 自动求导）
- **载体**: Jupyter Notebook (.ipynb)，Markdown 源文件 (.md)
- **风格**: 教学导向，中文注释，渐进式构建（先建空壳，再逐步添加权重、激活函数、反向传播）
- **目标平台**: Apple Silicon (MPS) 优化，同时兼容其他平台

## 内容生产流水线

Week07-Week12 的 Notebook 并非手动编辑，而是通过 `convert_md_to_ipynb.py` 从 Markdown 源文件自动转换：

1. **Markdown 源文件** — `chapter_XX_*.md` 遵循 `_chapter_template.md` 模板结构（历史剧场 → 生活隐喻 → 数学直觉 → 代码实验室 → 今日结语 → 翻译词典）
2. **转换脚本** — `convert_md_to_ipynb.py` 将 MD 解析为 Notebook cell，并通过 `add_visual_diagrams()` 在特定章节锚点（如 `## 2. 生活隐喻`）自动注入 matplotlib 生成的 base64 可视化图片
3. **图片验证** — `validate_imgs.py` 扫描 Notebook 中内嵌 base64 图片的数量和大小

Week01-Week03 的 Notebook 为手动编辑，直接在 Jupyter 中创建。

```bash
# 从 Markdown 源文件批量生成 Notebook（Week07-Week12）
python convert_md_to_ipynb.py

# 验证 Notebook 中内嵌图片
python validate_imgs.py
```

## 课程范围与结构

项目实际覆盖 **20 周** + 额外求职课程：

| 阶段 | 周次 | 形式 | 主题 |
|------|------|------|------|
| 初春与寒冬 | 1-2 | Notebook（手动） | 感知机, XOR 危机, 线性分类 |
| 破局与复兴 | 3-4 | Notebook（手动） | MLP, 反向传播, 梯度下降 |
| 视觉征服 | 5-6 | **Week05-06 尚未创建** | CNN, 卷积核, Pooling, 梯度消失 |
| 深度鸿沟 | 7-8 | MD→Notebook（自动） | 残差连接 F(x)+x, 退化问题 |
| 记忆诞生 | 9-10 | MD→Notebook（自动） | RNN, LSTM, GRU, Seq2Seq |
| 注意力时代 | 11-12 | MD→Notebook（自动） | Self-Attention Q/K/V, Transformer |
| 预训练革命 | 13-14 | Markdown only | 预训练, 微调, Scaling Laws, GPT 家族 |
| 对齐之路 | 15-16 | Markdown only | RLHF, 奖励模型, PPO |
| 思维与工具 | 17-18 | Markdown only | CoT, Tool Use, ReAct, Prompt Engineering |
| Agent 时代 | 19-20 | Markdown only | Agent 架构, 记忆, 规划, 多智能体 |

Week13-Week20 目前只有 Markdown 源文件，尚未转换为 Notebook。

## 开发命令

```bash
# 启动 Jupyter Notebook（在项目根目录下运行）
jupyter notebook

# 运行单个 notebook（命令行方式）
jupyter nbconvert --to notebook --execute Week01/Day01_张量_深度学习的通用货币.ipynb

# 从 Markdown 源文件批量生成 Notebook
python convert_md_to_ipynb.py

# 安装依赖（仅需 PyTorch + matplotlib + numpy）
pip install torch matplotlib numpy
```

## 编码规范

- **教学优先**: 每个概念先用 Markdown 单元格讲解，紧接着用代码单元格具象化
- **渐进式构建**: 同一周内 Notebook 中的类使用继承链逐层叠加能力（如 `Perceptron → PerceptronWithMemory → FullyFunctionalPerceptron`）
- **避免高级封装**: 优先使用 `torch.matmul`、`torch.sigmoid` 等底层张量运算，刻意避免 `nn.Linear`、`nn.Module` 等高级 API，以揭示底层数据流
- **梯度流程完整展示**: 前向传播 → Loss 计算 → `loss.backward()` → `torch.no_grad()` 下手动更新 → `.grad.zero_()` 清零
- **中文注释**: 所有注释和讲解使用中文，行内注释解释"为什么"而非"是什么"
- **禁止复杂语法**: 不使用三元运算符和复杂单行推导式，保证最高可读性
- **代码格式**: 符合 Flake8 规范，Google 风格 Docstring
- **中文字体配置**: matplotlib 使用 `plt.rcParams['font.sans-serif'] = ['STHeiti', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']`

## 教学设计模式

每个 Notebook / Markdown 章节遵循固定结构：
1. **历史背景** — 这个时代遇到了什么痛点
2. **生活隐喻** — 用类比（公司/山脉/水管/攀岩）解释数学概念
3. **数学直觉** — LaTeX 公式 + 直觉解释
4. **代码实现** — 从零手写，逐行中文注释，单个代码块不超过 80 行
5. **验证实验** — 用代码证明概念（如 XOR 不可解 → XOR 可解）
6. **翻译词典** — 生活直觉 → 深度学习术语的映射表
7. **总结预告** — 承上启下

## 每周检测 Notebook

每周末包含一个 `WeekXX_检测_*.ipynb`，用于检验该周学习成果。这是新增 Notebook 时需要同步维护的部分。
