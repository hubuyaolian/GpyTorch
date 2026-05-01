# Day 05：RLHF全流程概览 —— 三步炼成对齐模型

> ⚖️ 第十五周 · 人类反馈对齐 · 第5天

前四天我们分别理解了对齐问题、指令微调、奖励模型直觉和训练。今天，是时候把所有拼图拼在一起了——**RLHF（Reinforcement Learning from Human Feedback）**是一个三步流水线：SFT → 训练奖励模型 → PPO优化。每一步都建立在前一步的基础上，最终炼成一个"既聪明又听话"的对齐模型。

**今天的任务**：
1. 掌握RLHF三步流程的完整架构
2. 理解每步的输入/输出/目的
3. 理解PPO在RLHF中的角色
4. 预告Week16的实现计划

---

## 1. 历史剧场：RLHF的诞生之路

### 🎭 三个关键突破的汇聚

RLHF不是一夜之间发明的，它是三条研究线的汇聚：

```
研究线1: 从人类偏好学习
  Christiano et al. (2017) "Deep RL from Human Preferences"
  → 用人类比较数据训练奖励模型，用于Atari游戏
  → 核心贡献：比较数据 → 奖励模型 → RL优化

研究线2: 语言模型的对齐
  Ziegler et al. (2019) "Fine-Tuning Language Models from Human Preferences"
  → 将偏好学习引入语言模型
  → 核心贡献：证明RLHF在文本生成上可行

研究线3: 大规模验证
  Ouyang et al. (2022) "Training language models to follow instructions"
  → InstructGPT: SFT + RM + PPO 在GPT-3上大规模验证
  → 核心贡献：RLHF在工业级模型上有效
```

### 🎭 ChatGPT的诞生时刻

```
2022年11月30日，OpenAI发布ChatGPT

背后的技术栈:
  GPT-3.5 (预训练)
    → SFT (指令微调，人类撰写示范数据)
      → RM (奖励模型，人类标注偏好数据)
        → PPO (强化学习优化，用RM作为奖励信号)

结果: 5天100万用户，2个月1亿用户
证明: RLHF是对齐的有效方案
```

---

## 2. 生活隐喻：三步炼丹术

### 🔮 将普通人培养成金牌管家

想象你要把一个普通人培养成金牌管家，过程和RLHF完全对应：

| 步骤 | 管家培养 | RLHF |
|------|---------|------|
| **Step 1** | 基础培训：学会基本礼仪和服务规范 | **SFT**：指令微调，学会听指令 |
| **Step 2** | 培养品味：学会判断什么服务是好的 | **训练RM**：学会给回答打分 |
| **Step 3** | 反复练习：根据品味反馈不断改进 | **PPO**：用RM信号优化策略 |

### 🔑 Step 3的关键：为什么需要PPO？

```
问：有了SFT模型和RM，为什么不直接用RM选最好的回答？

答：因为RM只能"评判"不能"生成"——
   你需要从所有可能回答中搜索RM分数最高的那个，
   但回答空间是指数级的，无法穷举。

   PPO的作用：在RM的引导下，逐步调整生成策略，
   让模型"学会"生成高分回答，而不是每次都搜索。
```

就像管家不能每次服务都翻手册查"什么服务最好"，而是要通过反复练习**内化**品味标准——PPO就是内化的过程。

---

## 3. 数学直觉：RLHF的优化目标

### 📐 三步流程的数学表达

**Step 1: SFT（监督微调）**

$$\theta_{sft} = \arg\max_\theta \mathbb{E}_{(x,y) \sim D_{sft}} [\log P_\theta(y|x)]$$

目标：在人类示范数据上最大化似然。

**Step 2: 训练奖励模型**

$$\phi^* = \arg\min_\phi \mathbb{E}_{(x,y_w,y_l) \sim D_{pref}} \left[-\log \sigma(r_\phi(x,y_w) - r_\phi(x,y_l))\right]$$

目标：学习人类偏好评分函数。

**Step 3: PPO优化**

$$\theta^* = \arg\max_\theta \mathbb{E}_{x \sim D, y \sim \pi_\theta} \left[ r_{\phi^*}(x,y) - \beta \cdot KL(\pi_\theta \| \pi_{sft}) \right]$$

目标：最大化奖励，同时不要偏离SFT模型太远。

### 📐 KL惩罚项的直觉

$$-\beta \cdot KL(\pi_\theta \| \pi_{sft})$$

为什么需要KL惩罚？

```
没有KL惩罚:
  → 模型可能"奖励黑客"：生成RM给高分但实际无意义的回答
  → 模型可能退化：只输出几种RM给高分的模式，丧失多样性
  → 模型可能遗忘：偏离预训练学到的语言能力

有KL惩罚:
  → 约束模型不要偏离SFT模型太远
  → β控制"创新vs保守"的权衡
  → β大→保守（接近SFT），β小→激进（追求高分）
```

### 📐 PPO的更新规则（简化）

```
1. 从当前策略π_θ采样回答: y ~ π_θ(·|x)
2. 用RM计算奖励: r = r_φ(x, y)
3. 计算优势函数: A = r - V(x)  （V是价值函数的估计）
4. 更新策略: θ ← θ + η · A · ∇_θ log π_θ(y|x)
5. KL约束: 如果 KL(π_θ || π_sft) > δ, 停止更新
```

---

## 4. 代码实验室：RLHF三步流程图

```python
"""
RLHF全流程概览：三步流水线可视化
=================================
用ASCII原理图 + 流程可视化展示RLHF的完整架构
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. RLHF三步流程 ASCII原理图
# ============================================================
rlhf_diagram = """
╔══════════════════════════════════════════════════════════════╗
║                    RLHF 三步流水线                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      ║
║  │  Step 1:    │    │  Step 2:    │    │  Step 3:    │      ║
║  │  SFT        │───▶│  训练RM     │───▶│  PPO优化    │      ║
║  │  指令微调   │    │  奖励模型   │    │  策略优化   │      ║
║  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘      ║
║         │                  │                  │              ║
║  输入:  │           输入:  │           输入:  │              ║
║  预训练 │           SFT模型│           SFT模型│              ║
║  模型   │           +偏好  │           +RM    │              ║
║  +示范  │           数据   │                  │              ║
║  数据   │                  │           输出:  │              ║
║         │           输出:  │           对齐后  │              ║
║  输出:  │           奖励   │           模型    │              ║
║  SFT模型│           模型   │           π_rlhf  │              ║
║         │                  │                  │              ║
║  目的:  │           目的:  │           目的:  │              ║
║  学会听 │           学会品 │           学会生成│              ║
║  指令   │           味打分 │           高分回答│              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
print(rlhf_diagram)

# ============================================================
# 2. 三步流程详细对比
# ============================================================
print("=" * 70)
print("📋 RLHF三步流程详细对比")
print("=" * 70)

steps = [
    {
        "name": "Step 1: SFT (监督微调)",
        "input": "预训练模型 + 人类示范数据",
        "output": "SFT模型 (能听指令)",
        "loss": "L = -Σ log P(y|x)  (交叉熵)",
        "data": "~10K条人类撰写的(instruction, output)对",
        "time": "几小时 (全参数微调)",
        "purpose": "从续写模式切换到问答模式"
    },
    {
        "name": "Step 2: 训练RM (奖励模型)",
        "input": "SFT模型 + 人类偏好数据",
        "output": "奖励模型 (能给回答打分)",
        "loss": "L = -E[log σ(r_w - r_l)]  (Bradley-Terry)",
        "data": "~50K条人类标注的(chosen, rejected)对",
        "time": "几小时 (训练奖励头)",
        "purpose": "学习人类偏好，构建评分函数"
    },
    {
        "name": "Step 3: PPO (策略优化)",
        "input": "SFT模型 + 奖励模型",
        "output": "RLHF模型 (生成高分回答)",
        "loss": "max E[r(x,y)] - β·KL(π||π_sft)",
        "data": "用RM在线评分，无需额外人类数据",
        "time": "几天 (迭代优化)",
        "purpose": "在RM引导下优化生成策略"
    }
]

for step in steps:
    print(f"\n🔹 {step['name']}")
    print(f"   输入:   {step['input']}")
    print(f"   输出:   {step['output']}")
    print(f"   损失:   {step['loss']}")
    print(f"   数据:   {step['data']}")
    print(f"   时间:   {step['time']}")
    print(f"   目的:   {step['purpose']}")

# ============================================================
# 3. 可视化：RLHF三步流程
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 3.1 Step 1: SFT - 指令遵循率提升
epochs_sft = np.arange(0, 100)
follow_rate = 1 - np.exp(-epochs_sft / 30)
axes[0].plot(epochs_sft, follow_rate, linewidth=2.5, color='#4ecdc4')
axes[0].axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('SFT训练步数')
axes[0].set_ylabel('指令遵循率')
axes[0].set_title('Step 1: SFT — 学会听指令')
axes[0].annotate('SFT模型\n(能听指令)', xy=(80, 0.93),
                 fontsize=10, ha='center',
                 bbox=dict(boxstyle='round', facecolor='#4ecdc4', alpha=0.3))
axes[0].set_ylim(0, 1.05)

# 3.2 Step 2: RM - 偏好区分度提升
epochs_rm = np.arange(0, 50)
chosen_score = 1.5 * (1 - np.exp(-epochs_rm / 15))
rejected_score = -1.0 * (1 - np.exp(-epochs_rm / 20))
axes[1].plot(epochs_rm, chosen_score, linewidth=2.5, color='#4ecdc4', label='Chosen分数')
axes[1].plot(epochs_rm, rejected_score, linewidth=2.5, color='#ff6b6b', label='Rejected分数')
axes[1].fill_between(epochs_rm, chosen_score, rejected_score, alpha=0.15, color='#4ecdc4')
axes[1].set_xlabel('RM训练步数')
axes[1].set_ylabel('奖励分数')
axes[1].set_title('Step 2: 训练RM — 学会品味')
axes[1].legend()
axes[1].annotate('奖励模型\n(能给回答打分)', xy=(40, 0.5),
                 fontsize=10, ha='center',
                 bbox=dict(boxstyle='round', facecolor='#45b7d1', alpha=0.3))

# 3.3 Step 3: PPO - 人类偏好提升
epochs_ppo = np.arange(0, 200)
# 人类偏好得分随PPO训练提升
human_pref = 0.6 + 0.3 * (1 - np.exp(-epochs_ppo / 60))
# KL散度随训练增大（偏离SFT模型）
kl_div = 0.5 * (1 - np.exp(-epochs_ppo / 80))
# 奖励黑客风险
hacking_risk = 0.1 * np.exp(-((epochs_ppo - 150) / 40) ** 2)

ax3 = axes[2]
ax3.plot(epochs_ppo, human_pref, linewidth=2.5, color='#4ecdc4', label='人类偏好得分')
ax3.plot(epochs_ppo, kl_div, linewidth=2, color='#ff6b6b', linestyle='--', label='KL散度(偏离度)')
ax3.fill_between(epochs_ppo, 0, hacking_risk * 5, alpha=0.2, color='red', label='奖励黑客风险区')
ax3.set_xlabel('PPO训练步数')
ax3.set_ylabel('得分 / 散度')
ax3.set_title('Step 3: PPO — 学会生成高分回答')
ax3.legend(loc='right', fontsize=8)
ax3.annotate('RLHF模型\n(对齐完成)', xy=(170, 0.85),
             fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='#4ecdc4', alpha=0.3))
ax3.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('rlhf_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 可视化已保存至 rlhf_overview.png")

# ============================================================
# 4. RLHF vs 其他对齐方法对比
# ============================================================
print("\n" + "=" * 70)
print("📊 RLHF vs 其他对齐方法")
print("=" * 70)

methods = [
    ("仅SFT", "指令微调", "简单直接", "无法区分好与更好", "★★★"),
    ("SFT+RM", "+奖励模型", "有评分标准", "RM只评判不生成", "★★★★"),
    ("RLHF", "+PPO优化", "端到端对齐", "训练复杂、成本高", "★★★★★"),
    ("DPO", "直接偏好优化", "无需RM和RL", "隐式RM，灵活性低", "★★★★"),
]

print(f"\n{'方法':<12} {'技术':<14} {'优势':<14} {'局限':<16} {'效果'}")
print("-" * 70)
for method, tech, adv, limit, score in methods:
    print(f"{method:<12} {tech:<14} {adv:<14} {limit:<16} {score}")

print("\n💡 DPO (Direct Preference Optimization) 是RLHF的简化版，")
print("   跳过RM训练和PPO，直接用偏好数据优化策略。")
print("   Week16将详细对比RLHF和DPO。")
```

### 运行结果解读

ASCII原理图清晰展示了RLHF的三步流水线：SFT→RM→PPO，每一步的输入、输出、目的一目了然。三张子图分别展示了每步的训练动态：

- **Step 1**：指令遵循率从0快速上升到85%+
- **Step 2**：chosen和rejected的分数逐渐分离
- **Step 3**：人类偏好得分提升，同时KL散度增大（需要注意奖励黑客风险）

---

## 今日结语

RLHF是一个精妙的三步流水线：

1. **SFT**：让模型从"续写机器"变成"问答助手"——学会听指令
2. **训练RM**：让模型从"只会回答"变成"有品味"——学会评判好坏
3. **PPO**：让模型从"有品味但不一定做到"变成"既懂品味又能做到"——学会生成高分回答

三步缺一不可：没有SFT，模型连指令都听不懂；没有RM，PPO没有优化方向；没有PPO，RM的品味无法内化到生成策略中。

**下周预告**：Week16我们将逐步实现PPO算法和完整的RLHF训练流程，并对比DPO这一更简洁的替代方案。从概念到代码，从理解到实现——对齐之路，我们继续前行。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| RLHF | 基于人类反馈的强化学习 | 对齐的核心训练范式 |
| PPO | 近端策略优化 | Step 3使用的RL算法 |
| SFT | 监督微调 | Step 1的指令微调 |
| KL Penalty | KL惩罚 | 防止策略偏离参考模型太远 |
| Reward Hacking | 奖励黑客 | 模型钻RM漏洞获得高分但不真正对齐 |
| Reference Model | 参考模型 | KL惩罚中的参考，通常是SFT模型 |
| Advantage Function | 优势函数 | 衡量某个动作比平均水平好多少 |
| Value Function | 价值函数 | 从某状态出发的期望总回报 |
| DPO | 直接偏好优化 | RLHF的简化替代，跳过RM和PPO |
| Alignment Pipeline | 对齐流水线 | SFT→RM→PPO的完整训练流程 |
