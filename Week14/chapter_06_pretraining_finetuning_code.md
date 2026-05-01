# Day 06：预训练+微调完整代码实战 —— 从"裸奔"到"站在巨人肩上"

> 🔧 第十四周 · 预训练与规模法则 · 第 6 天

昨天我们建立了预训练-微调的直觉，但直觉不能当饭吃——今天我们要**亲手写代码**，把"先上学再找工作"的范式跑通。你会亲眼看到：同样的下游任务，从零训练磕磕绊绊，预训练+微调一路高歌。

**今天的任务**：
1. 实现MiniGPTForPretraining类（复用Decoder-only Transformer架构）
2. 跑通完整的预训练→微调流程
3. 对比"从零训练"与"预训练+微调"的训练曲线，用可视化说话

---

## 1. 历史剧场：2018，第一行预训练代码

2018年6月，OpenAI发布GPT-1的论文。核心代码逻辑其实非常简洁：

1. **预训练阶段**：把大量无标注文本喂给Transformer Decoder，做下一个token预测（语言模型任务）
2. **微调阶段**：保留预训练权重，换一个任务头（如分类器），用少量标注数据继续训练

Radford et al. 在论文中写道："We demonstrate that language understanding can be acquired by generative pre-training."——生成式预训练就能学会语言理解。

这个发现太重要了：**不需要标注数据**，只要让模型"读"足够多的文本，它就能学会语言的通用规律。微调只是锦上添花。

---

## 2. 生活隐喻：先通识再专精

- **从零训练** = 让一个婴儿直接学做手术 → 缺乏基础知识，怎么教都学不好
- **预训练** = 先上医学院学解剖、生理、药理 → 建立医学通识
- **微调** = 毕业后选外科方向做住院医 → 在通识基础上专精化

关键区别：
- 从零训练：参数随机初始化，下游数据少 → 欠拟合
- 预训练+微调：参数已有语言知识，微调只需小幅调整 → **收敛快、效果好**

---

## 3. 数学直觉：预训练-微调的优化视角

### 3.1 从零训练的优化起点

$$\theta_{scratch} \sim \mathcal{N}(0, \sigma^2)$$

从随机参数出发，在下游任务 $\mathcal{L}_{task}$ 上优化。数据少时，优化空间巨大，容易陷入局部最优。

### 3.2 预训练-微调的优化起点

$$\theta_{pretrained} = \arg\min_\theta \mathcal{L}_{LM}(\theta; \mathcal{D}_{large})$$

预训练把参数移到了一个**更好的起点**——已经编码了语言的统计规律。微调只需：

$$\theta_{finetuned} = \theta_{pretrained} - \eta \nabla \mathcal{L}_{task}(\theta_{pretrained})$$

因为起点好，微调用更少的数据和更少的步数就能收敛。

### 3.3 为什么预训练有效？

预训练学到的表示 $\theta_{pretrained}$ 位于参数空间中一个**对NLP任务普遍有利的区域**。微调只是在这个好区域里做小幅调整，而不是从随机位置艰难搜索。

---

## 4. 代码实验室：预训练+微调完整实战

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

### 4.1 MiniGPT核心组件：Decoder-only Transformer

```python
class MultiHeadAttention(nn.Module):
    """多头自注意力机制（Decoder-only，带因果掩码）."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        self.d_k = d_model // n_heads  # 每个头的维度
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)  # 查询投影
        self.W_k = nn.Linear(d_model, d_model)  # 键投影
        self.W_v = nn.Linear(d_model, d_model)  # 值投影
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影

    def forward(self, x):
        """前向传播：带因果掩码的多头注意力.
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            out: (batch, seq_len, d_model)
        """
        B, S, D = x.shape
        # 投影并分头: (B, S, D) -> (B, n_heads, S, d_k)
        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        # 注意力分数: (B, n_heads, S, S)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        # 因果掩码：下三角矩阵，遮住未来位置
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)  # 注意力权重
        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    """Transformer解码器块：注意力 + FFN + 残差 + LayerNorm."""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)  # 注意力前的LayerNorm
        self.ln2 = nn.LayerNorm(d_model)  # FFN前的LayerNorm
        self.ff = nn.Sequential(           # 前馈网络
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        """前向传播：Pre-LN Transformer块."""
        x = x + self.attn(self.ln1(x))  # 残差连接 + 注意力
        x = x + self.ff(self.ln2(x))    # 残差连接 + FFN
        return x
```

### 4.2 MiniGPTForPretraining：预训练语言模型

```python
class MiniGPTForPretraining(nn.Module):
    """MiniGPT预训练模型：Decoder-only Transformer + LM头.
    
    架构：Token Embedding + Position Embedding + N层Transformer + LM头
    预训练目标：下一个token预测（因果语言模型）
    """
    def __init__(self, vocab_size=100, d_model=64, n_heads=4,
                 n_layers=2, d_ff=256, max_seq_len=32):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)   # token嵌入
        self.pos_emb = nn.Embedding(max_seq_len, d_model)   # 位置嵌入
        self.blocks = nn.ModuleList([                        # Transformer块
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)                   # 最终LayerNorm
        self.lm_head = nn.Linear(d_model, vocab_size)       # 语言模型头

    def forward(self, input_ids):
        """前向传播：输入token序列，输出logits.
        
        Args:
            input_ids: (batch, seq_len) token ID序列
        Returns:
            logits: (batch, seq_len, vocab_size) 每个位置的词表分布
        """
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)  # 嵌入
        for block in self.blocks:
            x = block(x)          # 逐层通过Transformer
        x = self.ln_f(x)         # 最终归一化
        return self.lm_head(x)   # 映射到词表大小
```

### 4.3 模拟语料生成

```python
def generate_corpus(vocab_size=100, seq_len=16, n_samples=2000, seed=42):
    """生成模拟语料：带简单规律的token序列.
    
    规律设计：token[i+1] = (token[i] + 3) % vocab_size
    模型需要学会这个"加3取模"的规律才能做好预测。
    
    Args:
        vocab_size: 词表大小
        seq_len: 序列长度
        n_samples: 样本数量
        seed: 随机种子
    Returns:
        sequences: (n_samples, seq_len) 的token ID张量
    """
    torch.manual_seed(seed)
    sequences = []
    for _ in range(n_samples):
        start = torch.randint(0, vocab_size, (1,)).item()  # 随机起始token
        seq = [(start + 3 * i) % vocab_size for i in range(seq_len)]
        sequences.append(seq)
    return torch.tensor(sequences, dtype=torch.long)


def generate_downstream_data(vocab_size=100, seq_len=16, n_samples=200, seed=99):
    """生成下游任务数据：规律略有不同.
    
    下游规律：token[i+1] = (token[i] + 5) % vocab_size
    微调需要从"加3"的知识迁移到"加5"的规律。
    """
    torch.manual_seed(seed)
    sequences = []
    for _ in range(n_samples):
        start = torch.randint(0, vocab_size, (1,)).item()
        seq = [(start + 5 * i) % vocab_size for i in range(seq_len)]
        sequences.append(seq)
    return torch.tensor(sequences, dtype=torch.long)
```

### 4.4 预训练与微调循环

```python
def pretrain(model, data, epochs=30, lr=1e-3, batch_size=64):
    """预训练循环：下一个token预测.
    
    Args:
        model: MiniGPT模型
        data: (N, seq_len) 语料数据
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批大小
    Returns:
        losses: 每轮平均loss列表
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    N = data.shape[0]
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        perm = torch.randperm(N)  # 随机打乱
        for i in range(0, N, batch_size):
            batch = data[perm[i:i+batch_size]]
            input_ids = batch[:, :-1]   # 输入：前seq_len-1个token
            targets = batch[:, 1:]      # 目标：后seq_len-1个token（错一位）
            logits = model(input_ids)   # (B, S-1, vocab_size)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        losses.append(total_loss / n_batches)
    return losses


def finetune(model, data, epochs=30, lr=1e-4, batch_size=64):
    """微调循环：在预训练权重基础上继续训练.
    
    关键区别：学习率更小（1e-4 vs 1e-3），因为只需小幅调整。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    N = data.shape[0]
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            batch = data[perm[i:i+batch_size]]
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        losses.append(total_loss / n_batches)
    return losses
```

### 4.5 对比实验：从零训练 vs 预训练+微调

```python
# 超参数
VOCAB_SIZE = 100
SEQ_LEN = 16
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 256

# 生成数据
pretrain_data = generate_corpus(VOCAB_SIZE, SEQ_LEN, n_samples=2000)
downstream_data = generate_downstream_data(VOCAB_SIZE, SEQ_LEN, n_samples=200)

# ===== 方式1：从零训练（只用下游数据）=====
torch.manual_seed(42)
model_scratch = MiniGPTForPretraining(
    vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=SEQ_LEN
)
losses_scratch = finetune(model_scratch, downstream_data, epochs=50, lr=1e-3)

# ===== 方式2：预训练 + 微调 =====
torch.manual_seed(42)
model_pt = MiniGPTForPretraining(
    vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=SEQ_LEN
)
# 阶段1：在大语料上预训练
pretrain_losses = pretrain(model_pt, pretrain_data, epochs=30, lr=1e-3)
# 阶段2：在下游数据上微调（小学习率）
finetune_losses = finetune(model_pt, downstream_data, epochs=50, lr=1e-4)
```

### 4.6 可视化：预训练优势一目了然

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：预训练阶段的loss曲线
axes[0].plot(pretrain_losses, color='#3498db', linewidth=2, label='预训练loss')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('阶段1：预训练（大语料）', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 右图：微调 vs 从零训练的对比
axes[1].plot(losses_scratch, color='#e74c3c', linewidth=2,
           label='从零训练（仅下游数据）', alpha=0.8)
axes[1].plot(finetune_losses, color='#3498db', linewidth=2,
           label='微调（预训练→下游数据）', alpha=0.8)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('阶段2：从零训练 vs 微调', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle('预训练+微调 vs 从零训练', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()

# 打印最终loss对比
print(f"从零训练最终loss: {losses_scratch[-1]:.4f}")
print(f"预训练+微调最终loss: {finetune_losses[-1]:.4f}")
print(f"预训练带来改善: {losses_scratch[-1] - finetune_losses[-1]:.4f}")
```

---

## 今日结语

今天我们亲手实现了完整的预训练→微调流程：MiniGPT先在大语料上学会"加3取模"的规律，再用小学习率在下游数据上微调到"加5取模"。对比实验清楚地表明：**预训练+微调的loss收敛更快、最终更低**，而从零训练在少量下游数据上磕磕绊绊。

这就是2018年GPT-1和BERT的核心洞见：**先让模型广泛阅读，再让它专业培训**。这个范式至今仍是NLP的基石。

明天，我们将用代码验证规模法则：训练5个不同大小的MiniGPT，亲手拟合幂律曲线！

---

### 翻译词典

| 生活中的直觉 | 深度学习术语 |
|---|---|
| 先上医学院再选外科 | 预训练→微调 (Pretrain→Finetune) |
| 医学通识知识 | 预训练权重 (Pretrained Weights) |
| 住院医专精化 | 微调：小学习率小幅调整 |
| 婴儿直接学手术 | 从零训练：缺基础效果差 |
| 大语料学通用规律 | 语言模型预训练 (LM Pretraining) |
| 下游任务小数据 | 微调数据 (Fine-tuning Data) |
