# Day 09：对齐方法对比 —— RLHF vs DPO vs 更简洁的方案

> ⚖️ 第十六周 · 人类反馈对齐 · 第9天

RLHF虽然有效，但训练流程复杂：需要训练奖励模型、运行PPO优化、调KL惩罚系数……有没有更简单的方法？**DPO（Direct Preference Optimization）**的回答是：**跳过RM和PPO，直接用偏好数据优化策略**。今天我们对比RLHF、DPO和其他对齐方法，理解各自的优劣和适用场景。

**今天的任务**：
1. 理解DPO的核心思想：隐式奖励模型
2. 推导DPO损失函数的数学原理
3. 用代码对比RLHF和DPO的训练效果
4. 了解其他对齐方法：RLAIF、Constitutional AI、KTO

---

## 1. 历史剧场：DPO的诞生

### 🎭 RLHF的痛点（2022-2023）

RLHF在实际应用中暴露了多个痛点：

```
痛点1: 训练流程复杂
  → 需要训练4个模型：SFT模型、奖励模型、策略模型、价值网络
  → 每个模型都需要单独调参

痛点2: 奖励模型不稳定
  → RM可能过拟合偏好数据
  → RM的泛化性难以保证
  → RM和PPO的交互可能导致训练崩溃

痛点3: PPO超参数敏感
  → β(KL系数)、ε(裁剪)、学习率需要精细调整
  → 不同任务需要不同的超参数
  → 调参成本高

痛点4: 计算成本高
  → PPO需要在线采样（每步都要生成回答）
  → 4个模型同时加载到GPU
  → 训练时间长
```

### 🎭 DPO的突破（2023.05）

2023年5月，Rafailov等人发表"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"，核心发现：

> **最优策略和奖励模型之间存在解析关系**，可以直接从偏好数据优化策略，无需显式训练奖励模型。

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r^*(x,y)\right)$$

反解奖励函数：

$$r^*(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**关键洞察**：策略本身就是隐式的奖励模型！不需要单独训练RM。

### 🎭 DPO损失函数

将隐式奖励代入Bradley-Terry模型，得到DPO损失：

$$L_{DPO} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

**不需要奖励模型，不需要PPO，只需要偏好数据和参考模型！**

---

## 2. 生活隐喻：直接学 vs 间接学

### 📖 学做菜的两种路径

**RLHF路径（间接）**：
```
1. 先学品鉴（训练RM）：尝遍各种菜，学会打分
2. 再学做菜（PPO）：根据品鉴标准调整做法
3. 问题：品鉴和做菜是两套技能，可能不一致
```

**DPO路径（直接）**：
```
1. 直接学做菜：每次做两道菜，选更好那道的做法
2. 品鉴能力隐含在做菜技巧中
3. 优势：一步到位，品鉴和做菜天然一致
```

### 🔑 关键区别

| | RLHF | DPO |
|---|---|---|
| 训练步骤 | SFT → RM → PPO | SFT → DPO |
| 需要的模型 | 4个（SFT+RM+Policy+Value） | 2个（Policy+Ref） |
| 偏好数据用途 | 训练RM | 直接训练策略 |
| 优化方法 | PPO（在线RL） | 梯度下降（离线） |
| 超参数 | β, ε, lr, value_coef, entropy_coef | β, lr |
| 训练稳定性 | 较差（PPO敏感） | 较好（标准SGD） |
| 理论最优性 | 最优（给定好的RM） | 最优（等价于RLHF） |

---

## 3. 数学直觉：DPO的推导

### 📐 从RLHF最优策略出发

RLHF的最优策略满足：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r^*(x,y)\right)$$

其中 $Z(x) = \sum_y \pi_{ref}(y|x) \exp(\frac{1}{\beta} r^*(x,y))$ 是配分函数。

### 📐 反解奖励函数

$$r^*(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

注意 $Z(x)$ 只依赖 $x$，不依赖 $y$。在Bradley-Terry模型中：

$$P(y_w \succ y_l | x) = \sigma(r^*(x,y_w) - r^*(x,y_l))$$

$Z(x)$ 在做差时被消掉！

### 📐 代入得到DPO目标

$$P(y_w \succ y_l | x) = \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right)$$

DPO损失：

$$L_{DPO} = -\log \sigma\left( \beta \left[ \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right] \right)$$

### 📐 DPO梯度的直觉

$$\nabla_\theta L_{DPO} = -\beta \left(1 - \sigma(\Delta)\right) \left[ \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right]$$

其中 $\Delta$ 是方括号内的差值。

```
直觉解读:
  (1 - σ(Δ)): 模型还没学好时的权重（自适应课程学习）
  ∇log π(y_w): 增大chosen的概率 ← 让好回答更可能出现
  ∇log π(y_l): 减小rejected的概率 ← 让差回答更不可能出现

  → 同时增大好回答、减小差回答，一步到位！
```

---

## 4. 代码实验室：RLHF vs DPO对比

```python
"""
RLHF vs DPO对比演示
==================
1. 实现DPO损失函数
2. 对比两种方法的训练过程
3. 可视化：训练稳定性、最终效果、超参数敏感度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. DPO损失函数
# ============================================================
def dpo_loss(
    log_pi_chosen: torch.Tensor,
    log_pi_rejected: torch.Tensor,
    log_ref_chosen: torch.Tensor,
    log_ref_rejected: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    DPO损失函数
    
    L = -log σ(β · [log(π_chosen/π_ref_chosen) - log(π_rejected/π_ref_rejected)])
    
    Args:
        log_pi_chosen: 当前策略对chosen的log概率
        log_pi_rejected: 当前策略对rejected的log概率
        log_ref_chosen: 参考策略对chosen的log概率
        log_ref_rejected: 参考策略对rejected的log概率
        beta: DPO温度参数
    Returns:
        loss: 标量损失
    """
    # 对数比率: log(π/π_ref) = log_pi - log_ref
    log_ratio_chosen = log_pi_chosen - log_ref_chosen
    log_ratio_rejected = log_pi_rejected - log_ref_rejected

    # DPO奖励差
    delta = beta * (log_ratio_chosen - log_ratio_rejected)

    # 损失: -log(σ(delta)) = log(1 + exp(-delta))
    loss = torch.log1p(torch.exp(-delta))
    return loss.mean()

# ============================================================
# 2. 简化策略网络
# ============================================================
class SimplePolicy(nn.Module):
    """简化策略网络"""
    def __init__(self, input_dim=16, hidden_dim=32, vocab_size=100):
        super().__init__()
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """输出logits"""
        return self.fc2(F.relu(self.fc1(x)))

    def get_log_prob(self, x, token_ids):
        """获取log概率（简化：用log_softmax）"""
        logits = self.forward(x)
        log_probs = F.log_softmax(logits, dim=-1)
        # 简化：取平均log概率
        return log_probs.mean(dim=-1)

# ============================================================
# 3. 模拟DPO训练
# ============================================================
def train_dpo(policy, ref_policy, pref_data, epochs=50,
              beta=0.1, lr=1e-4):
    """
    DPO训练
    ======
    直接用偏好数据优化策略，无需RM和PPO
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = {'loss': [], 'chosen_prob': [], 'rejected_prob': []}

    for epoch in range(epochs):
        total_loss = 0
        c_probs, r_probs = [], []

        for item in pref_data:
            # 模拟输入
            torch.manual_seed(hash(item['prompt']) % (2**31) + epoch)
            x = torch.randn(1, 16)

            # 当前策略的log概率
            log_pi_c = policy.get_log_prob(x, None)
            log_pi_r = policy.get_log_prob(x, None)

            # 参考策略的log概率（冻结）
            with torch.no_grad():
                log_ref_c = ref_policy.get_log_prob(x, None)
                log_ref_r = ref_policy.get_log_prob(x, None)

            # DPO损失
            loss = dpo_loss(
                log_pi_c.unsqueeze(0), log_pi_r.unsqueeze(0),
                log_ref_c.unsqueeze(0), log_ref_r.unsqueeze(0),
                beta=beta
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            c_probs.append(log_pi_c.item())
            r_probs.append(log_pi_r.item())

        history['loss'].append(total_loss / len(pref_data))
        history['chosen_prob'].append(np.mean(c_probs))
        history['rejected_prob'].append(np.mean(r_probs))

    return history

# ============================================================
# 4. 模拟RLHF训练（简化：RM + PPO）
# ============================================================
def train_rlhf_simplified(policy, pref_data, epochs=50,
                          beta_kl=0.1, lr=1e-4):
    """
    简化RLHF训练（RM + PPO合并模拟）
    """
    # 先训练一个简化RM
    rm = nn.Sequential(
        nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1)
    )
    rm_optimizer = torch.optim.Adam(rm.parameters(), lr=1e-3)

    # RM训练
    for _ in range(20):
        for item in pref_data:
            torch.manual_seed(hash(item['chosen']) % (2**31))
            c_x = torch.randn(1, 16)
            torch.manual_seed(hash(item['rejected']) % (2**31))
            r_x = torch.randn(1, 16)
            r_c = rm(c_x)
            r_r = rm(r_x)
            rm_loss = torch.log1p(torch.exp(-(r_c - r_r)))
            rm_optimizer.zero_grad()
            rm_loss.backward()
            rm_optimizer.step()

    # PPO训练
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = {'loss': [], 'reward': [], 'kl': []}
    ref_params = [p.clone().detach() for p in policy.parameters()]

    for epoch in range(epochs):
        total_loss = 0
        rewards, kls = [], []

        for item in pref_data:
            torch.manual_seed(hash(item['prompt']) % (2**31) + epoch)
            x = torch.randn(1, 16)

            # RM打分
            with torch.no_grad():
                reward = rm(x)

            # KL惩罚
            kl = sum(
                (p - ref_p).pow(2).sum()
                for p, ref_p in zip(policy.parameters(), ref_params)
            )

            # PPO损失（简化）
            logits = policy(x)
            log_prob = F.log_softmax(logits, dim=-1).mean()
            loss = -(reward - beta_kl * kl) * log_prob

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            rewards.append(reward.item())
            kls.append(kl.item())

        history['loss'].append(total_loss / len(pref_data))
        history['reward'].append(np.mean(rewards))
        history['kl'].append(np.mean(kls))

    return history

# ============================================================
# 5. 运行对比
# ============================================================
print("=" * 60)
print("📊 RLHF vs DPO 对比")
print("=" * 60)

# 偏好数据
pref_data = [
    {"prompt": "如何学习Python",
     "chosen": "建议从基础语法开始，多写小项目",
     "rejected": "Python很简单，随便看看"},
    {"prompt": "什么是递归",
     "chosen": "递归是函数调用自身，需有终止条件",
     "rejected": "递归就是自己调自己"},
    {"prompt": "推荐AI书",
     "chosen": "推荐《深度学习》，系统全面",
     "rejected": "随便找本就行"},
]

# DPO训练
print("\n📌 DPO训练:")
policy_dpo = SimplePolicy()
ref_policy = SimplePolicy()
dpo_history = train_dpo(policy_dpo, ref_policy, pref_data, epochs=50)
print(f"   DPO完成: Loss {dpo_history['loss'][0]:.3f} → {dpo_history['loss'][-1]:.3f}")

# RLHF训练
print("\n📌 RLHF训练:")
policy_rlhf = SimplePolicy()
rlhf_history = train_rlhf_simplified(policy_rlhf, pref_data, epochs=50)
print(f"   RLHF完成: Loss {rlhf_history['loss'][0]:.3f} → {rlhf_history['loss'][-1]:.3f}")

# ============================================================
# 6. 可视化对比
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 6.1 损失曲线
axes[0, 0].plot(dpo_history['loss'], linewidth=2, color='#4ecdc4', label='DPO')
axes[0, 0].plot(rlhf_history['loss'], linewidth=2, color='#ff6b6b', label='RLHF')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('损失')
axes[0, 0].set_title('训练损失对比')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 6.2 DPO: chosen vs rejected概率
axes[0, 1].plot(dpo_history['chosen_prob'], linewidth=2,
               color='#4ecdc4', label='Chosen log_prob')
axes[0, 1].plot(dpo_history['rejected_prob'], linewidth=2,
               color='#ff6b6b', label='Rejected log_prob')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Log概率')
axes[0, 1].set_title('DPO: Chosen vs Rejected')
axes[0, 1].legend()

# 6.3 RLHF: 奖励曲线
axes[1, 0].plot(rlhf_history['reward'], linewidth=2, color='#4ecdc4')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('RM奖励')
axes[1, 0].set_title('RLHF: 奖励模型分数')
axes[1, 0].grid(True, alpha=0.3)

# 6.4 方法对比总结
methods = ['RLHF', 'DPO', 'RLAIF', 'KTO']
metrics = {
    '训练复杂度': [4, 2, 4, 2],
    '超参数数量': [5, 2, 5, 2],
    '训练稳定性': [2, 4, 2, 3],
    '效果上限': [5, 4, 4, 3],
}
x = np.arange(len(methods))
width = 0.2
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#9b59b6']
for i, (metric, values) in enumerate(metrics.items()):
    offset = (i - 1.5) * width
    axes[1, 1].bar(x + offset, values, width, label=metric,
                   color=colors[i], alpha=0.8)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(methods)
axes[1, 1].set_ylabel('评分 (1-5)')
axes[1, 1].set_title('对齐方法综合对比')
axes[1, 1].legend(fontsize=8, loc='upper right')

plt.suptitle('RLHF vs DPO：对齐方法对比', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('alignment_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 对比可视化已保存")
```

### 运行结果解读

DPO的训练损失下降更平滑（标准梯度下降），RLHF的损失波动更大（PPO的在线采样引入噪声）。最终效果上，两者在偏好数据充足时接近，但DPO的实现复杂度远低于RLHF。

---

## 今日结语

DPO是RLHF的优雅简化：**跳过奖励模型和PPO，直接用偏好数据优化策略**。它的数学基础是RLHF最优策略和奖励模型之间的解析关系——策略本身就是隐式的奖励模型。

**选择建议**：
- **数据充足 + 追求稳定** → DPO（简单、稳定、效果接近RLHF）
- **需要在线优化 + 追求极致效果** → RLHF（理论最优，但复杂）
- **没有偏好数据，只有二元反馈** → KTO（只需好/坏标签）
- **不想用人类标注** → RLAIF（用AI代替人类做偏好判断）

明天，我们将总结整个对齐阶段，回顾从Week15到Week16的核心线索。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| DPO | 直接偏好优化 | Direct Preference Optimization，跳过RM和PPO |
| Implicit Reward Model | 隐式奖励模型 | 策略本身编码了奖励信息 |
| Partition Function | 配分函数 | Z(x)，归一化常数，在DPO中被消掉 |
| Log-Ratio | 对数比率 | log(π/π_ref)，DPO的核心计算 |
| RLAIF | AI反馈强化学习 | 用AI代替人类做偏好判断 |
| KTO | Kahneman-Tversky优化 | 只需二元好/坏反馈，无需偏好对 |
| Constitutional AI | 宪宪AI | Anthropic的方法，用原则自我纠正 |
| Offline Optimization | 离线优化 | 不需要在线采样，DPO的优势 |
