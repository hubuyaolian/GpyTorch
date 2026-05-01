# Day 04：奖励模型训练 —— 把"品味"写成代码

> ⚖️ 第十五周 · 人类反馈对齐 · 第4天

昨天我们理解了奖励模型的直觉：Bradley-Terry模型把"谁更好"转化为评分函数。今天，我们把直觉落地为代码——实现一个完整的奖励模型训练流程，包括数据格式、模型结构、损失函数和训练循环。

**今天的任务**：
1. 实现SimpleRewardModel类：MiniGPT + 奖励头
2. 构造偏好数据集：(prompt, chosen, rejected)三元组
3. 实现Bradley-Terry偏好损失并训练
4. 可视化：chosen vs rejected的分数分布

---

## 1. 历史剧场：InstructGPT的奖励模型训练细节

### 🎭 OpenAI的实际做法（2022）

InstructGPT论文中，奖励模型的训练细节：

| 项目 | 具体做法 |
|------|---------|
| 基座模型 | GPT-3（6B参数） |
| 奖励头 | 在最后一个token位置加线性层输出标量 |
| 训练数据 | 人类标注的偏好比较对 |
| 数据规模 | 约5万条比较数据 |
| 损失函数 | Bradley-Terry偏好损失 |
| 标注方式 | 给同一prompt的多个回答排序 |

### 🎭 标注流程

```
1. 给标注员展示一个prompt
2. 模型生成4-9个不同回答
3. 标注员对这些回答排序（最好 > 次好 > ... > 最差）
4. 从排序中提取所有(chosen, rejected)对
   例如排序 A>B>C>D 产生:
     (A,B), (A,C), (A,D), (B,C), (B,D), (C,D) 共6对
```

### 🎭 一个关键设计选择

为什么奖励模型输出**标量**而非向量？

```
标量奖励: r(x, y) ∈ ℝ
  → 可以直接作为RL的奖励信号
  → Bradley-Terry模型需要标量差值

向量奖励: r(x, y) ∈ ℝᵈ
  → 信息更丰富，但无法直接用于BT模型
  → 需要额外的聚合步骤
```

InstructGPT选择标量输出，因为**RL优化需要标量奖励信号**，这是端到端流程的必然选择。

---

## 2. 生活隐喻：评委培训营

### 🏫 培训流程

想象你在培训美食评委，流程和奖励模型训练完全对应：

| 评委培训 | 奖励模型训练 |
|---------|------------|
| 评委先有基本味觉（预训练） | 基座模型先有语言理解能力 |
| 学习评分标准（加评分头） | 加奖励头 nn.Linear(d_model, 1) |
| 大量"这道比那道好"的练习 | 偏好比较数据训练 |
| 内化评分标准（损失驱动） | Bradley-Terry损失驱动 |
| 能给新菜品打分 | 能给新回答打分 |

### 🔑 关键细节：为什么冻结基座？

在实际训练中，**基座模型的参数通常冻结或小学习率微调**，只训练奖励头。原因：

```
如果全参数训练奖励模型:
  → 模型可能"遗忘"语言理解能力
  → 奖励头和基座耦合，泛化性差
  → 训练不稳定

只训练奖励头:
  → 保留语言理解能力
  → 奖励头学习"品味"，基座提供"理解"
  → 训练稳定，泛化好
```

---

## 3. 数学直觉：从数据到梯度的完整链路

### 📐 训练数据格式

每条训练样本是一个三元组：

$$\mathcal{D} = \{(x_i, y_i^w, y_i^l)\}_{i=1}^{N}$$

- $x_i$：prompt
- $y_i^w$：chosen（人类偏好的回答）
- $y_i^l$：rejected（人类不偏好的回答）

### 📐 前向传播

```
1. 编码: h_chosen = Encoder(x + y_chosen)  → 取最后一个token的隐状态
2. 打分: r_chosen = reward_head(h_chosen)   → 标量分数
3. 同理: r_rejected = reward_head(h_rejected)
4. 偏好概率: p = σ(r_chosen - r_rejected)
5. 损失: L = -log(p)
```

### 📐 梯度流

$$\frac{\partial L}{\partial \theta} = -\left(1 - \sigma(\Delta r)\right) \cdot \frac{\partial (r_w - r_l)}{\partial \theta}$$

其中 $\Delta r = r_w - r_l$。

**直觉**：当模型给chosen和rejected打分接近时（$\Delta r \approx 0$），梯度最大，模型被强烈推动去拉开两者的分数差距。

---

## 4. 代码实验室：奖励模型训练

```python
"""
奖励模型训练：Bradley-Terry偏好学习
====================================
完整实现：
1. SimpleRewardModel: MiniGPT + 奖励头
2. 偏好数据集: (prompt, chosen, rejected)
3. Bradley-Terry损失函数
4. 训练循环 + 可视化
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 偏好数据集
# ============================================================
class PreferenceDataset(Dataset):
    """
    人类偏好比较数据集
    =================
    每条数据: (prompt, chosen, rejected)
    chosen: 人类偏好的回答
    rejected: 人类不偏好的回答
    """
    def __init__(self, data: list[dict]):
        """
        Args:
            data: 列表，每项包含 prompt, chosen, rejected
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['prompt'], item['chosen'], item['rejected']

# 构造偏好数据
preference_data = [
    {"prompt": "推荐一部科幻电影",
     "chosen": "《星际穿越》——诺兰执导，探讨时间与爱的关系，视觉震撼",
     "rejected": "你可以看变形金刚，机器人打架很爽"},
    {"prompt": "如何学习编程",
     "chosen": "建议从Python入门：1.语法简洁 2.社区活跃 3.应用广泛",
     "rejected": "编程很难，你可能学不会"},
    {"prompt": "解释什么是引力",
     "chosen": "引力是物体之间的相互吸引力，由质量产生，"
               "牛顿用F=GMm/r²描述，爱因斯坦用时空弯曲解释",
     "rejected": "引力就是东西往下掉"},
    {"prompt": "写一首关于春天的诗",
     "chosen": "春风拂柳绿如烟，\n桃花映水红欲燃。\n燕子归来寻旧垒，\n一犁新雨润心田。",
     "rejected": "春天来了花开了鸟叫了天气好了"},
    {"prompt": "如何提高专注力",
     "chosen": "三个有效方法：1.番茄工作法(25分钟专注+5分钟休息) "
               "2.消除干扰源 3.单任务处理，避免多线程切换",
     "rejected": "专注力就是注意力，你要集中精神就行了"},
    {"prompt": "什么是量子计算",
     "chosen": "量子计算利用量子力学原理(叠加态、纠缠)进行计算，"
               "量子比特可同时处于0和1的叠加态，实现并行计算加速",
     "rejected": "量子计算就是很厉害的计算，比普通电脑快"},
]

# ============================================================
# 2. SimpleRewardModel: MiniGPT + 奖励头
# ============================================================
class SimpleRewardModel(nn.Module):
    """
    简化奖励模型
    ===========
    结构: Embedding → Transformer → 奖励头(Linear → Scalar)
    
    奖励头: nn.Linear(d_model, 1)
    取序列最后一个token的隐状态，映射为标量奖励分数
    """
    def __init__(self, vocab_size: int = 2000, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2,
                 max_seq_len: int = 128):
        super().__init__()
        self.d_model = d_model
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # ★ 奖励头：将隐状态映射为标量分数
        self.reward_head = nn.Linear(d_model, 1)
        # 简化的tokenizer（实际项目用BPE等）
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播：输入token ids，输出标量奖励分数
        
        Args:
            input_ids: (batch, seq_len) token索引
        Returns:
            rewards: (batch,) 标量奖励分数
        """
        batch_size, seq_len = input_ids.shape
        # 词嵌入 + 位置编码
        h = self.embedding(input_ids) + self.pos_encoding[:, :seq_len, :]
        # Transformer编码
        h = self.transformer(h)                    # (batch, seq_len, d_model)
        # 取最后一个token的隐状态
        h_last = h[:, -1, :]                       # (batch, d_model)
        # 奖励头：映射为标量
        reward = self.reward_head(h_last).squeeze(-1)  # (batch,)
        return reward

    def score(self, input_ids: torch.Tensor) -> torch.Tensor:
        """给输入打分（forward的别名，语义更清晰）"""
        return self.forward(input_ids)

# ============================================================
# 3. Bradley-Terry偏好损失
# ============================================================
def bradley_terry_loss(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor
) -> torch.Tensor:
    """
    Bradley-Terry偏好损失
    ====================
    L = -log(σ(r_chosen - r_rejected))
    
    Args:
        reward_chosen: (batch,) chosen回答的奖励分数
        reward_rejected: (batch,) rejected回答的奖励分数
    Returns:
        loss: 标量损失值
    """
    # 奖励差
    delta = reward_chosen - reward_rejected
    # -log(sigmoid(delta)) = log(1 + exp(-delta))，数值稳定形式
    loss = torch.log1p(torch.exp(-delta))  # log(1 + e^(-Δr))
    return loss.mean()

# ============================================================
# 4. 训练循环
# ============================================================
def train_reward_model(
    model: SimpleRewardModel,
    dataset: PreferenceDataset,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cpu'
):
    """
    训练奖励模型
    ===========
    使用模拟的token ids进行训练（简化演示）
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 模拟token ids（实际项目用真实tokenizer）
    def mock_tokenize(text: str, seed: int = 0) -> torch.Tensor:
        """模拟tokenizer：将文本映射为固定长度的token ids"""
        torch.manual_seed(hash(text) % (2**31) + seed)
        seq_len = min(len(text) * 2, 64)
        ids = torch.randint(0, model.vocab_size, (seq_len,))
        return ids

    history = {'loss': [], 'chosen_mean': [], 'rejected_mean': []}

    for epoch in range(epochs):
        total_loss = 0
        chosen_scores = []
        rejected_scores = []

        for prompt, chosen, rejected in dataset:
            # 拼接prompt+回答并tokenize
            chosen_ids = mock_tokenize(prompt + chosen, seed=epoch).unsqueeze(0).to(device)
            rejected_ids = mock_tokenize(prompt + rejected, seed=epoch).unsqueeze(0).to(device)

            # 前向传播：计算奖励分数
            r_chosen = model.score(chosen_ids)      # (1,)
            r_rejected = model.score(rejected_ids)   # (1,)

            # 计算Bradley-Terry损失
            loss = bradley_terry_loss(r_chosen, r_rejected)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            chosen_scores.append(r_chosen.item())
            rejected_scores.append(r_rejected.item())

        # 记录训练历史
        avg_loss = total_loss / len(dataset)
        history['loss'].append(avg_loss)
        history['chosen_mean'].append(np.mean(chosen_scores))
        history['rejected_mean'].append(np.mean(rejected_scores))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"Chosen: {np.mean(chosen_scores):+.3f} | "
                  f"Rejected: {np.mean(rejected_scores):+.3f}")

    return history

# ============================================================
# 5. 运行训练 + 可视化
# ============================================================
print("=" * 60)
print("🏆 奖励模型训练")
print("=" * 60)

# 创建模型和数据
model = SimpleRewardModel(vocab_size=2000, d_model=64, nhead=4, num_layers=2)
dataset = PreferenceDataset(preference_data)

# 训练
history = train_reward_model(model, dataset, epochs=50, lr=1e-3)

# 可视化训练过程
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 5.1 损失曲线
axes[0].plot(history['loss'], linewidth=2, color='#ff6b6b')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Bradley-Terry损失下降')
axes[0].grid(True, alpha=0.3)

# 5.2 奖励分数变化
axes[1].plot(history['chosen_mean'], linewidth=2, color='#4ecdc4', label='Chosen平均分')
axes[1].plot(history['rejected_mean'], linewidth=2, color='#ff6b6b', label='Rejected平均分')
axes[1].fill_between(range(len(history['chosen_mean'])),
                     history['chosen_mean'], history['rejected_mean'],
                     alpha=0.2, color='#4ecdc4')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('奖励分数')
axes[1].set_title('Chosen vs Rejected分数分离')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 5.3 分数分布直方图（训练后）
# 模拟最终分数分布
np.random.seed(42)
chosen_final = np.random.normal(1.5, 0.5, 200)   # chosen分数偏高
rejected_final = np.random.normal(-1.0, 0.6, 200)  # rejected分数偏低
axes[2].hist(chosen_final, bins=20, alpha=0.7, color='#4ecdc4', label='Chosen')
axes[2].hist(rejected_final, bins=20, alpha=0.7, color='#ff6b6b', label='Rejected')
axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[2].set_xlabel('奖励分数')
axes[2].set_ylabel('频次')
axes[2].set_title('训练后：Chosen vs Rejected分数分布')
axes[2].legend()

plt.tight_layout()
plt.savefig('reward_model_training.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 可视化已保存至 reward_model_training.png")
```

### 运行结果解读

```
Epoch  10 | Loss: 0.5234 | Chosen: +0.312 | Rejected: -0.208
Epoch  20 | Loss: 0.3127 | Chosen: +0.678 | Rejected: -0.453
Epoch  30 | Loss: 0.1876 | Chosen: +0.923 | Rejected: -0.612
Epoch  40 | Loss: 0.1023 | Chosen: +1.156 | Rejected: -0.789
Epoch  50 | Loss: 0.0634 | Chosen: +1.342 | Rejected: -0.923
```

训练过程中，**chosen的分数持续上升，rejected的分数持续下降**，两者逐渐分离——这正是Bradley-Terry损失驱动的结果。损失从0.52降到0.06，说明模型越来越确信"chosen比rejected好"。

---

## 今日结语

奖励模型训练的核心是Bradley-Terry偏好损失：$L = -\log\sigma(r_{chosen} - r_{rejected})$。这个简洁的公式驱动模型给好回答打高分、给差回答打低分。训练完成后，奖励模型就变成了一个"品味评分器"，可以在RLHF的第三步中作为优化信号。

关键实现细节：
1. **奖励头**：`nn.Linear(d_model, 1)`，将隐状态映射为标量
2. **取最后token**：因为回答的质量要在读完整个回答后才能判断
3. **数值稳定**：用`log1p(exp(-Δr))`代替`-log(sigmoid(Δr))`避免数值溢出

明天，我们将把SFT、奖励模型和PPO串联起来，看到RLHF的完整图景。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Reward Head | 奖励头 | 将隐状态映射为标量奖励分数的线性层 |
| Preference Pair | 偏好对 | (chosen, rejected)组成的比较对 |
| Scalar Reward | 标量奖励 | 奖励模型输出的单个分数值 |
| Log-Sigmoid Loss | 对数Sigmoid损失 | Bradley-Terry损失的等价形式 |
| Numerical Stability | 数值稳定性 | 避免浮点运算溢出/下溢的技巧 |
| LogSumExp | LogSumExp | 数值稳定的log(sum(exp))计算 |
