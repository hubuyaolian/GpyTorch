# Day 03：微调与迁移学习 —— 从"通才"到"专才"

> 📚 第十三周 · 预训练与规模法则 · 第 3 天

预训练让模型成了"通才"——懂语言但不擅长任何具体任务。微调就是让这个通才变成"专才"：用少量任务数据继续训练，快速适应特定任务。这就是迁移学习在NLP中的成功应用。

**今天的任务**：
1. 理解微调(Fine-tuning)与特征提取(Feature Extraction)两种迁移方式
2. 用代码演示"冻结部分层"与"全量微调"的差异
3. 体会预训练权重作为"先验知识"的价值

---

## 1. 历史剧场：迁移学习的NLP革命

在计算机视觉领域，迁移学习早已是标配——用ImageNet预训练的ResNet微调下游任务。但NLP领域长期缺乏有效的预训练模型。

2018年，这一切改变了：
- **ULMFiT**（Howard & Ruder, 2018）：首次证明在NLP中预训练-微调范式可行
- **GPT-1**：将ULMFiT的思想与Transformer结合，在12项任务上取得突破
- **BERT**：双向预训练+微调，进一步提升了迁移学习的效果

**关键发现**：NLP的迁移学习甚至比CV更有效——因为语言的知识更具通用性。

---

## 2. 生活隐喻：医学本科与专科规培

- **预训练** = 医学本科5年：学习解剖、生理、病理等通用医学知识
- **全量微调** = 专科规培：在通用基础上，所有知识都可以根据专科需要调整
- **冻结底层微调顶层** = 规培时保留基础医学知识，只调整临床技能
- **特征提取（冻结全部）** = 本科毕业后直接上岗，不再学习，只用已有知识

冻结底层的原因：底层学到了最通用的特征（如词向量、语法结构），微调时不应破坏。

---

## 3. 数学直觉：微调的两种策略

### 3.1 全量微调

$$\theta_{finetuned} = \theta_{pretrained} - \eta \nabla L_{task}(\theta_{pretrained})$$

所有参数都参与更新，学习率 $\eta$ 通常比预训练时小（1/10到1/100）。

### 3.2 部分微调（冻结底层）

$$\theta_{frozen} = \theta_{pretrained}^{[1:k]} \quad \text{(前k层冻结)}$$
$$\theta_{tunable} = \theta_{pretrained}^{[k+1:]} - \eta \nabla L_{task}(\theta_{pretrained}^{[k+1:]})$$

底层参数保持预训练值不变，只更新顶层参数。

---

## 4. 代码实验室：微调策略对比

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

### 4.1 构建可冻结层的模型

```python
class PretrainModel(nn.Module):
    """可冻结底层的预训练模型."""
    def __init__(self, input_dim=10, hidden1=64, hidden2=32, output_dim=4):
        super().__init__()
        # 底层（通用特征提取）
        self.layer1 = nn.Linear(input_dim, hidden1)
        # 中层（任务相关特征）
        self.layer2 = nn.Linear(hidden1, hidden2)
        # 顶层（任务输出）
        self.output = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        """前向传播."""
        h1 = F.relu(self.layer1(x))   # 底层特征
        h2 = F.relu(self.layer2(h1))  # 中层特征
        return self.output(h2)        # 输出

    def freeze_bottom(self, n_layers=1):
        """冻结底层参数."""
        if n_layers >= 1:
            for param in self.layer1.parameters():
                param.requires_grad = False  # 冻结第1层
        if n_layers >= 2:
            for param in self.layer2.parameters():
                param.requires_grad = False  # 冻结第2层

    def unfreeze_all(self):
        """解冻所有参数."""
        for param in self.parameters():
            param.requires_grad = True
```

### 4.2 模拟预训练→微调流程

```python
torch.manual_seed(42)

# 预训练数据（大量，4分类任务）
X_pretrain = torch.randn(2000, 10)
y_pretrain = (X_pretrain[:, 0] > 0).long() + 2 * (X_pretrain[:, 1] > 0).long()

# 下游任务数据（少量，4分类任务，但分布略有不同）
X_downstream = torch.randn(150, 10) + 0.5  # 分布偏移
y_downstream = (X_downstream[:, 0] > 0.3).long() + \
               2 * (X_downstream[:, 1] > 0.3).long()

# 测试数据
X_test = torch.randn(500, 10) + 0.5
y_test = (X_test[:, 0] > 0.3).long() + 2 * (X_test[:, 1] > 0.3).long()

# Step 1: 预训练
model_pretrained = PretrainModel()
optimizer = torch.optim.Adam(model_pretrained.parameters(), lr=0.01)
for epoch in range(100):
    logits = model_pretrained(X_pretrain)
    loss = F.cross_entropy(logits, y_pretrain)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("预训练完成!")
```

### 4.3 对比三种微调策略

```python
import copy

def finetune_and_eval(model, X, y, X_test, y_test, epochs=80, lr=0.001):
    """微调模型并返回准确率曲线."""
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    accs = []
    for epoch in range(epochs):
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = model(X_test).argmax(dim=1)
            acc = (pred == y_test).float().mean().item()
        accs.append(acc)
    return accs

# 策略1: 从零训练（对照组）
model_scratch = PretrainModel()
accs_scratch = finetune_and_eval(model_scratch, X_downstream, y_downstream,
                                  X_test, y_test, lr=0.01)

# 策略2: 全量微调
model_full = copy.deepcopy(model_pretrained)
model_full.unfreeze_all()
accs_full = finetune_and_eval(model_full, X_downstream, y_downstream,
                               X_test, y_test, lr=0.001)

# 策略3: 冻结底层，只微调顶层
model_frozen = copy.deepcopy(model_pretrained)
model_frozen.freeze_bottom(n_layers=1)  # 冻结第1层
accs_frozen = finetune_and_eval(model_frozen, X_downstream, y_downstream,
                                 X_test, y_test, lr=0.001)

# 可视化
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(accs_scratch, color='#e74c3c', label='从零训练', alpha=0.8)
ax.plot(accs_full, color='#3498db', label='全量微调', alpha=0.8)
ax.plot(accs_frozen, color='#27ae60', label='冻结底层微调', alpha=0.8)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('测试准确率', fontsize=12)
ax.set_title('三种微调策略对比', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 今日结语

微调是让预训练模型从"通才"变成"专才"的关键步骤。我们对比了三种策略：从零训练效果最差，全量微调和冻结底层微调都显著优于从零训练。冻结底层的好处是保留通用特征、防止过拟合，但全量微调在小学习率下通常效果更好。

明天，我们将探索一个惊人的发现：规模法则——模型越大，效果越好，而且遵循精确的幂律！

---

### 翻译词典

| 生活中的直觉 | 深度学习术语 |
|---|---|
| 医学本科→专科规培 | 预训练→微调 |
| 保留基础医学，调整临床技能 | 冻结底层，微调顶层 |
| 不再学习直接上岗 | 特征提取（冻结全部） |
| 通才变专才 | 迁移学习 (Transfer Learning) |
| 小学习率微调 | 避免破坏预训练知识 |
