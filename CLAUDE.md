# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

《AI进化史：从感知机到Transformer的觉醒之路》— 一个为期 12 周的深度学习教学项目，通过 Jupyter Notebook 演示从感知机到 Transformer 的核心架构演进。核心理念是"逻辑复现"：用精简的 PyTorch 代码还原伟大架构的本质，而非深陷底层数学推导。

## 技术栈

- **语言**: Python
- **核心依赖**: PyTorch (张量运算 + Autograd 自动求导)
- **载体**: Jupyter Notebook (.ipynb)
- **风格**: 教学导向，中文注释，渐进式构建（先建空壳，再逐步添加权重、激活函数、反向传播）

## 项目结构

```
├── 00_AI进化史_总体大纲.md    # 12周课程总体路线图
├── Week01/                    # ❄️ 感知机与 XOR 危机
│   ├── Day01_张量_深度学习的通用货币.ipynb
│   ├── Day02_生物神经元与感知机的诞生.ipynb
│   ├── Day03_权重与偏置_赋予神经元灵魂.ipynb
│   ├── Day04_线性分类边界_感知机的几何直觉.ipynb
│   ├── Day05_XOR危机_AI第一次寒冬的代码现场.ipynb
│   └── images/                # Week01 可视化图片
├── Week02/                    # ☀️ 隐藏层与反向传播
│   ├── Day01_隐藏层_打破线性边界的魔法.ipynb
│   ├── Day02_非线性激活函数_Sigmoid与ReLU.ipynb
│   ├── Day03_反向传播的直觉_链式法则的代码实现.ipynb
│   ├── Day04_手写MLP_用代码解决XOR问题.ipynb
│   ├── Day05_损失函数与梯度下降_模型的吐槽大会.ipynb
│   └── images/                # Week02 可视化图片
├── Week03/                    # 👁️ CNN 与视觉征服
│   ├── Day01_卷积的诞生_让机器学会看图.ipynb
│   ├── Day02_卷积层_让机器自己学习特征.ipynb
│   ├── Day03_池化_信息的浓缩与抽象.ipynb
│   ├── Day04_构建CNN_手写LeNet5.ipynb
│   ├── Day05_梯度消失_深度网络的老年痴呆.ipynb
│   ├── images/                # Week03 可视化图片
│   └── data/                  # MNIST 数据集
└── Week04-Week12/             # 后续课程（ResNet、RNN/LSTM、Transformer）
```

## 课程大纲与进度

| 阶段 | 周次 | 主题 | 核心概念 |
|------|------|------|----------|
| 初春与寒冬 | 1-2 | 感知机 | Perceptron, Weight, Bias, 线性分类, XOR 危机 |
| 破局与复兴 | 3-4 | MLP + 反向传播 | 隐藏层, Sigmoid/ReLU, Backprop, 梯度下降 |
| 视觉征服 | 5-6 | CNN | 卷积核, 权值共享, Pooling, 梯度消失 |
| 深度鸿沟 | 7-8 | ResNet | 残差连接 F(x)+x, 退化问题 |
| 记忆诞生 | 9-10 | RNN/LSTM | 隐藏状态, 门控机制, 长距离依赖 |
| 注意力时代 | 11-12 | Transformer | Self-Attention Q/K/V, 并行计算 |

## 开发命令

```bash
# 启动 Jupyter Notebook（在项目根目录下运行）
jupyter notebook

# 运行单个 notebook（命令行方式）
jupyter nbconvert --to notebook --execute Week01/01_感知机与XOR危机.ipynb

# 安装依赖（仅需 PyTorch）
pip install torch
```

## 编码规范

- **教学优先**: 每个概念先用 Markdown 单元格讲解，紧接着用代码单元格具象化
- **渐进式构建**: 同一周内 Notebook 中的类使用继承链逐层叠加能力（如 `Perceptron → PerceptronWithMemory → FullyFunctionalPerceptron`）
- **避免高级封装**: 优先使用 `torch.matmul`、`torch.sigmoid` 等底层张量运算，刻意避免 `nn.Linear`、`nn.Module` 等高级 API，以揭示底层数据流
- **梯度流程完整展示**: 前向传播 → Loss 计算 → `loss.backward()` → `torch.no_grad()` 下手动更新 → `.grad.zero_()` 清零
- **中文注释**: 所有注释和讲解使用中文，行内注释解释"为什么"而非"是什么"

## 教学设计模式

每个 Notebook 遵循固定结构：
1. **历史背景** — 这个时代遇到了什么痛点
2. **核心概念** — 用类比（公司/山脉/水管）解释数学概念
3. **代码实现** — 从零手写，逐行注释
4. **验证实验** — 用代码证明概念（如 XOR 不可解 → XOR 可解）
5. **总结预告** — 承上启下
