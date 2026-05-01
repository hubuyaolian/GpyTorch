# Day 01：从Transformer到预训练 —— 为什么"先上学再找工作"？

> 📚 第十三周 · 预训练与规模法则 · 第 1 天

我们在Week 12搭建了完整的Transformer，它能做序列反转、简单翻译。但如果你只用几百条数据从零训练它，效果会很差——它需要**海量数据**才能学会语言的规律。这就是Transformer的局限：从零训练太慢、太费数据。

**今天的任务**：
1. 理解Transformer从零训练的核心局限
2. 认识"预训练-微调"范式的诞生背景
3. 建立从"小数据从零训练"到"大数据预训练+小数据微调"的直觉

---

## 1. 历史剧场：2018，预训练范式的诞生

2017年Transformer论文发表后，人们很快发现了一个尴尬的事实：Transformer虽然架构强大，但从零训练需要**海量数据和算力**。普通研究者根本玩不起。

2018年，两篇论文同时改变了局面：

- **GPT-1**（OpenAI，2018年6月）：先用Transformer Decoder在大量无标注文本上做**预训练**（学习语言的通用规律），再用少量标注数据做**微调**（适应特定任务）。在12项NLP任务中9项达到SOTA。
- **BERT**（Google，2018年10月）：用Transformer Encoder做双向预训练（完形填空），再微调下游任务。横扫11项NLP基准。

**核心洞察**：与其每个任务都从零训练，不如先让模型"广泛阅读"（预训练），再"专业培训"（微调）。

---

## 2. 生活隐喻：先上大学再找工作

- **从零训练** = 让一个人从文盲开始，只看几本专业书就想成为专家 → 效果差，因为缺乏基础知识
- **预训练** = 上大学，广泛学习各科知识 → 建立通用的知识基础
- **微调** = 毕业后入职培训，学习岗位特定技能 → 在已有基础上快速适应

关键区别：
- 从零训练：数据少 → 欠拟合，数据多 → 训练太慢
- 预训练+微调：预训练用大数据学通用知识，微调用小数据学特定技能 → **又快又好**

---

## 3. 数学直觉：预训练-微调范式

### 3.1 从零训练的损失

$$L_{from\_scratch} = L_{task}(\theta_{random})$$

从随机初始化的参数 $\theta_{random}$ 开始，只用下游任务数据优化。数据少时，模型很难学到语言的通用规律。

### 3.2 预训练-微调的损失

$$L_{pretrain} = L_{LM}(\theta_{random}) \rightarrow \theta_{pretrained}$$
$$L_{finetune} = L_{task}(\theta_{pretrained}) \rightarrow \theta_{finetuned}$$

预训练阶段：在大规模无标注语料上做语言模型训练，得到 $\theta_{pretrained}$。
微调阶段：从 $\theta_{pretrained}$ 出发，用少量下游数据继续优化。

**核心优势**：$\theta_{pretrained}$ 已经编码了语言的通用知识，微调只需在此基础上做小幅调整。

---

## 4. 代码实验室：从零训练 vs 预训练的直觉

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

### 4.1 用简单实验展示"数据量"对训练效果的影响

```python
# 用一个简单的2层网络做分类任务
# 对比：少量数据 vs 大量数据的训练效果

class SimpleNet(nn.Module):
    """简单的2层分类网络."""
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """前向传播."""
        h = F.relu(self.fc1(x))  # 隐藏层+ReLU激活
        return self.fc2(h)       # 输出层

# 生成模拟数据
torch.manual_seed(42)
n_features = 10

# 大量数据（模拟"预训练语料"的规模）
X_large = torch.randn(2000, n_features)
y_large = (X_large[:, 0] + X_large[:, 1] > 0).long()

# 少量数据（模拟"下游任务"的规模）
X_small = torch.randn(100, n_features)
y_small = (X_small[:, 0] + X_small[:, 1] > 0).long()

# 测试数据
X_test = torch.randn(500, n_features)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).long()
```

### 4.2 对比：少量数据训练 vs 先用大数据预训练再微调

```python
def train_and_eval(model, X, y, X_test, y_test, epochs, lr=0.01):
    """训练模型并返回每轮测试准确率."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    accs = []
    for epoch in range(epochs):
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 测试准确率
        with torch.no_grad():
            pred = model(X_test).argmax(dim=1)
            acc = (pred == y_test).float().mean().item()
        accs.append(acc)
    return accs

# 方式1：只用少量数据从零训练
model_scratch = SimpleNet()
accs_scratch = train_and_eval(model_scratch, X_small, y_small,
                               X_test, y_test, epochs=100)

# 方式2：先用大数据"预训练"，再用小数据"微调"
model_pretrain = SimpleNet()
# 预训练阶段：用大数据训练
train_and_eval(model_pretrain, X_large, y_large,
               X_test, y_test, epochs=50)
# 微调阶段：用小数据继续训练（小学习率）
accs_finetune = train_and_eval(model_pretrain, X_small, y_small,
                                X_test, y_test, epochs=100, lr=0.001)

# 可视化对比
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(accs_scratch, color='#e74c3c', label='从零训练(少量数据)',
        alpha=0.8, linewidth=2)
ax.plot(accs_finetune, color='#3498db', label='预训练+微调',
        alpha=0.8, linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('测试准确率', fontsize=12)
ax.set_title('从零训练 vs 预训练+微调', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 今日结语

Transformer虽然架构强大，但从零训练需要海量数据。2018年GPT-1和BERT开创了"预训练-微调"范式：先在大数据上预训练学通用知识，再在小数据上微调学特定技能。这就像"先上大学再找工作"——有了通用基础，适应新任务就快得多。

明天，我们将深入预训练的直觉：语言模型到底在"学"什么？

---

### 翻译词典

| 生活中的直觉 | 深度学习术语 |
|---|---|
| 先上大学再找工作 | 预训练-微调范式 (Pretrain-Finetune) |
| 广泛阅读各科知识 | 预训练：学习语言通用表示 |
| 毕业后入职培训 | 微调：适应下游任务 |
| 从文盲开始只看专业书 | 从零训练：数据少效果差 |
| 通用知识基础 | 预训练权重 (Pretrained Weights) |
