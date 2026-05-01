# Day 02：预训练的直觉 —— 语言模型如何"读懂"语言？

> 📚 第十三周 · 预训练与规模法则 · 第 2 天

预训练到底在做什么？简单说，就是让模型做**"预测下一个词"**的任务。通过反复猜测下一个词是什么，模型被迫学会语法、语义、甚至一些常识推理。这就是语言模型预训练的核心直觉。

**今天的任务**：
1. 理解因果语言模型（Causal LM）的预训练目标
2. 用代码演示简化版语言模型预训练
3. 观察预训练过程中模型"学到"了什么

---

## 1. 历史剧场：语言模型的进化

语言模型的思想由来已久：

- **2003年**：Bengio的神经概率语言模型——用神经网络建模 P(w_t | w_{t-1}, ..., w_{t-n})
- **2018年**：GPT-1将语言模型推向预训练范式——用Transformer Decoder做大规模因果语言模型
- **2020年**：GPT-2/3证明：只要模型够大、数据够多，"预测下一个词"就能涌现出惊人的能力

**核心洞察**：预测下一个词看似简单，但要做到好，模型必须理解语法、语义、逻辑、甚至世界知识。

---

## 2. 生活隐喻：完形填空与续写

- **因果语言模型** = 续写故事：给你"从前有座山，山里有座"，你要猜下一个词是"庙"
- **掩码语言模型** = 完形填空：给你"从前有座[MASK]，山里有座庙"，你要猜[MASK]是"山"

续写（GPT路线）只能看前面的词，完形填空（BERT路线）可以看前后所有词。

**预训练的本质**：让模型做海量的"续写"或"填空"练习，通过大量练习学会语言的规律。

---

## 3. 数学直觉：因果语言模型

### 3.1 自回归分解

任何一段文本 $x_1, x_2, ..., x_T$ 的概率可以分解为：

$$P(x_1, x_2, ..., x_T) = \prod_{t=1}^{T} P(x_t | x_1, ..., x_{t-1})$$

因果语言模型就是学习这个条件概率链——每次根据前面所有词，预测下一个词。

### 3.2 预训练损失

$$L_{pretrain} = -\sum_{t=1}^{T} \log P(x_t | x_1, ..., x_{t-1}; \theta)$$

最小化这个损失，就是让模型尽可能准确地预测每个位置的下一个词。

---

## 4. 代码实验室：简化版语言模型预训练

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

### 4.1 构建简化版GPT语言模型

```python
class PositionalEncoding(nn.Module):
    """位置编码."""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        PE = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                       -(np.log(10000.0) / d_model))
        PE[:, 0::2] = torch.sin(pos * div)  # 偶数位用sin
        PE[:, 1::2] = torch.cos(pos * div)  # 奇数位用cos
        self.register_buffer('PE', PE.unsqueeze(0))

    def forward(self, x):
        """加入位置信息."""
        return x + self.PE[:, :x.size(1)]


class MiniGPTLM(nn.Module):
    """简化版GPT语言模型（用于预训练）."""
    def __init__(self, vocab_size, d_model=64, n_heads=4,
                 d_ff=256, n_layers=2, max_len=50):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        # Transformer Decoder层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_ff, batch_first=True, dropout=0.1)
        self.decoder = nn.ModuleList([decoder_layer] * n_layers)
        # 语言模型头：预测下一个词
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """前向传播：输入序列 → 每个位置的下一个词概率."""
        emb = self.pos_enc(self.embedding(x) * np.sqrt(self.d_model))
        S = x.size(1)
        # 因果掩码：每个词只能看到自己和之前的词
        mask = nn.Transformer.generate_square_subsequent_mask(S)
        out = emb
        enc_dummy = emb  # Decoder-only: encoder输出用自身代替
        for layer in self.decoder:
            out = layer(out, enc_dummy, tgt_mask=mask)
        return self.lm_head(out)  # (batch, seq_len, vocab_size)
```

### 4.2 在小语料上做预训练

```python
# 构建模拟语料
vocab = {'<PAD>': 0, '深': 1, '度': 2, '学': 3, '习': 4,
         '是': 5, 'AI': 6, '的': 7, '未': 8, '来': 9,
         '机': 10, '器': 11, '神': 12, '经': 13, '网': 14,
         '络': 15, '很': 16, '强': 17}
vocab_size = len(vocab)

# 模拟训练语料（多条句子）
corpus = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9],      # 深度学习是AI的未来
    [10, 11, 12, 13, 14, 15, 16, 17],  # 机器神经网络很强
    [1, 2, 3, 4, 16, 17],              # 深度学习很强
    [6, 7, 8, 9, 5, 1, 2, 3, 4],       # AI的未来是深度学习
    [12, 13, 14, 15, 5, 6, 7, 10, 11], # 神经网络是AI的机器
]

# 预训练
model = MiniGPTLM(vocab_size, d_model=32, n_heads=4, d_ff=128, n_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

losses = []
for epoch in range(200):
    total_loss = 0
    for sent in corpus:
        sent_tensor = torch.tensor([sent])
        input_ids = sent_tensor[:, :-1]    # 输入：去掉最后一个词
        target_ids = sent_tensor[:, 1:]    # 目标：去掉第一个词（预测下一个词）
        logits = model(input_ids)
        loss = loss_fn(logits.reshape(-1, vocab_size),
                       target_ids.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss / len(corpus))

print(f"预训练完成! 最终Loss: {losses[-1]:.4f}")
```

### 4.3 观察预训练后模型的"续写"能力

```python
# 用预训练模型做续写
model.eval()
with torch.no_grad():
    # 给定开头"深度"，看模型续写什么
    prompt = torch.tensor([[1]])  # "深"
    generated = [1]
    for _ in range(6):  # 续写6个词
        logits = model(prompt)
        next_id = logits[0, -1, :].argmax().item()
        generated.append(next_id)
        prompt = torch.tensor([generated])

# 解码
id_to_word = {v: k for k, v in vocab.items()}
result = ''.join([id_to_word.get(i, '?') for i in generated])
print(f"续写结果: {result}")

# 可视化训练损失
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(losses, color='#3498db', alpha=0.8, linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('语言模型预训练损失', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 今日结语

预训练的核心就是让模型做"预测下一个词"的任务。通过海量练习，模型学会了语法、语义和世界知识。GPT路线用因果语言模型（续写），BERT路线用掩码语言模型（填空）。今天我们用简化代码演示了GPT风格的预训练过程。

明天，我们将学习微调与迁移学习——如何让预训练模型快速适应新任务。

---

### 翻译词典

| 生活中的直觉 | 深度学习术语 |
|---|---|
| 续写故事 | 因果语言模型 (Causal LM) |
| 完形填空 | 掩码语言模型 (Masked LM) |
| 大量练习学会规律 | 预训练：学习语言通用表示 |
| 预测下一个词 | 自回归 (Autoregressive) |
| 只能看前面的词 | 因果掩码 (Causal Mask) |
