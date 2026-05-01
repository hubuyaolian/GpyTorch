# Day 03：奖励模型直觉 —— 教AI"品味"

> ⚖️ 第十五周 · 人类反馈对齐 · 第3天

指令微调让模型学会了"听指令"，但听指令≠做得好。就像学生学会了"按题目写作文"，但写出来的可能只是及格水平——我们需要一个"评分标准"来区分优秀与平庸。**奖励模型（Reward Model）就是AI的品味系统**：它学会人类的偏好，给不同回答打分，告诉模型"这个回答比那个好"。

**今天的任务**：
1. 理解奖励模型的核心思想：从"人类偏好"到"标量评分"
2. 掌握Bradley-Terry模型的数学原理
3. 用代码演示：奖励模型如何对同一问题的不同回答排序

---

## 1. 历史剧场：从"谁更好"到"好多少"

### 🎭 人类偏好的表达困境

2022年之前，对齐研究面临一个根本问题：**人类知道"哪个更好"，但很难给出精确分数**。

```
问题: "推荐一部科幻电影"

回答A: "《星际穿越》——诺兰导演，探讨时间与爱的关系"
回答B: "你可以看《变形金刚》"
回答C: "科幻电影有很多，比如《银翼杀手》《2001太空漫游》等"

人类判断: A > C > B  （A最好，B最差，C中等）
但人类很难说: A是8.5分，B是2.1分，C是6.3分
```

人类做**比较**很准，做**绝对评分**很不准。这就是奖励模型要解决的核心问题：**从比较数据中学习一个评分函数**。

### 🎭 Bradley-Terry模型的引入

统计学中早有解决方案：**Bradley-Terry模型**（1952年），原本用于体育比赛的胜负预测。

核心思想：如果选手A的实力分是 $s_A$，选手B的实力分是 $s_B$，那么A胜B的概率：

$$P(A \text{ wins}) = \frac{s_A}{s_A + s_B}$$

InstructGPT论文将此模型引入RLHF：把"回答A比回答B好"类比为"选手A战胜选手B"，用奖励分数代替实力分。

### 🎭 从比较到评分的飞跃

```
人类提供:  (prompt, chosen, rejected) —— "chosen比rejected好"
BT模型学习: r(prompt, chosen) > r(prompt, rejected) —— 学会打分
RLHF使用:  r(prompt, ·) 作为奖励信号 —— 指导策略优化
```

奖励模型是**人类偏好到优化信号的桥梁**：它把模糊的"谁更好"转化成精确的梯度信号。

---

## 2. 生活隐喻：美食评委的品味课

### 🍽️ 场景：训练一个美食评委

想象你要培养一个美食评委，但你只能给他**两道菜让他选哪个更好**，不能给他**绝对评分标准**：

```
第1轮: 宫保鸡丁 vs 白水煮鸡 → 选宫保鸡丁 ✓
第2轮: 佛跳墙 vs 宫保鸡丁 → 选佛跳墙 ✓
第3轮: 佛跳墙 vs 满汉全席 → 选满汉全席 ✓
```

评委从这些比较中**内化**了一个评分标准：

```
白水煮鸡: 2分
宫保鸡丁: 6分
佛跳墙:   8分
满汉全席: 9.5分
```

虽然你从没告诉他具体分数，但他学会了**品味**——这就是奖励模型。

### 🔑 关键洞察

| | 美食评委 | 奖励模型 |
|---|---|---|
| 输入 | 两道菜，选哪个更好 | 两个回答，选哪个更被偏好 |
| 学习目标 | 内化评分标准 | 学习奖励函数 $r(x, y)$ |
| 输出 | 给任意菜品打分 | 给任意回答打分 |
| 数学工具 | 偏好排序 | Bradley-Terry模型 |

### 🧠 为什么比较比评分更靠谱？

```
绝对评分的问题:
  评委A: 这道菜8分    ← A的8分和B的8分标准不同
  评委B: 这道菜8分

比较的优势:
  评委A: 宫保鸡丁比白水煮鸡好  ← 稳定、一致
  评委B: 宫保鸡丁比白水煮鸡好  ← 不同评委也容易达成共识
```

心理学研究证实：**人类做比较的可靠性远高于绝对评分**。这就是为什么RLHF用比较数据而非评分数据。

---

## 3. 数学直觉：Bradley-Terry偏好模型

### 📐 核心公式

给定prompt $x$，两个回答 $y_w$（chosen/胜出）和 $y_l$（rejected/落败），奖励模型预测人类偏好 $y_w > y_l$ 的概率：

$$P(y_w \succ y_l | x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

其中：
- $r_\theta(x, y)$ 是奖励模型，输出标量分数
- $\sigma(\cdot)$ 是sigmoid函数：$\sigma(z) = \frac{1}{1 + e^{-z}}$
- $r_\theta(x, y_w) - r_\theta(x, y_l)$ 是两个回答的**奖励差**

### 📐 直觉解读

```
如果 r(chosen) >> r(rejected):  差值很大 → σ(大正数) ≈ 1 → "chosen明显更好" ✓
如果 r(chosen) ≈ r(rejected):  差值接近0 → σ(0) = 0.5 → "差不多好" 
如果 r(chosen) << r(rejected):  差值很负 → σ(大负数) ≈ 0 → "chosen更差" ✗
```

**奖励差越大，chosen胜出的概率越高**——这完全符合直觉。

### 📐 训练损失函数

对于一条比较数据 $(x, y_w, y_l)$，损失函数是负对数似然：

$$L = -\log P(y_w \succ y_l | x) = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

对整个数据集：

$$L_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right]$$

### 📐 损失函数的直觉

```
当 r(chosen) - r(rejected) 很大时:
  σ(大正数) ≈ 1 → log(1) = 0 → 损失 ≈ 0  ← 模型预测正确，无需更新

当 r(chosen) - r(rejected) 很小/为负时:
  σ(小/负数) ≈ 0 → log(0) → -∞ → 损失很大  ← 模型预测错误，大幅更新
```

**损失函数迫使奖励模型给chosen打高分、给rejected打低分**——这正是我们想要的。

### 📐 梯度分析

对 $\theta$ 求梯度：

$$\nabla_\theta L = -\left(1 - \sigma(\Delta r)\right) \cdot \nabla_\theta (r_\theta(x, y_w) - r_\theta(x, y_l))$$

其中 $\Delta r = r_\theta(x, y_w) - r_\theta(x, y_l)$。

关键洞察：
- 当 $\Delta r$ 已经很大时，$(1 - \sigma(\Delta r)) \approx 0$，梯度消失——**已经学好的样本不再主导训练**
- 当 $\Delta r$ 接近0或为负时，梯度最大——**模型判断错误的样本获得最大更新**

这是**自适应的课程学习**：模型自动聚焦于"还没学好的"比较对。

---

## 4. 代码实验室：奖励模型直觉演示

```python
"""
奖励模型直觉演示：Bradley-Terry偏好模型
========================================
用代码演示：
1. sigmoid如何将奖励差转化为偏好概率
2. 损失函数如何驱动奖励模型学习
3. 奖励模型对回答的排序能力
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. Bradley-Terry偏好概率可视化
# ============================================================
def bradley_terry_prob(delta_r: np.ndarray) -> np.ndarray:
    """
    Bradley-Terry模型：偏好概率
    P(chosen > rejected) = σ(Δr)
    
    Args:
        delta_r: 奖励差 r_chosen - r_rejected
    Returns:
        chosen胜出的概率
    """
    return 1.0 / (1.0 + np.exp(-delta_r))

delta_r = np.linspace(-5, 5, 200)
prob = bradley_terry_prob(delta_r)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 1.1 偏好概率曲线
axes[0].plot(delta_r, prob, linewidth=2.5, color='#4ecdc4')
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('奖励差 Δr = r(chosen) - r(rejected)')
axes[0].set_ylabel('P(chosen > rejected)')
axes[0].set_title('Bradley-Terry偏好概率')
axes[0].annotate('Δr=0 → P=0.5\n(无法区分)', xy=(0, 0.5),
                 xytext=(2, 0.35), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='black'))
axes[0].annotate('Δr>>0 → P≈1\n(chosen明显更好)', xy=(4, 0.98),
                 xytext=(1.5, 0.85), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='black'))

# 1.2 损失函数曲线
loss = -np.log(prob + 1e-8)  # 加小量防止log(0)
axes[1].plot(delta_r, loss, linewidth=2.5, color='#ff6b6b')
axes[1].set_xlabel('奖励差 Δr')
axes[1].set_ylabel('损失 -log(σ(Δr))')
axes[1].set_title('Bradley-Terry损失函数')
axes[1].annotate('Δr大 → 损失≈0\n(已学好)', xy=(4, 0.02),
                 xytext=(0, 3), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='black'))
axes[1].annotate('Δr<0 → 损失很大\n(预测错误)', xy=(-3, 3.05),
                 xytext=(-1, 4), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='black'))

# 1.3 梯度幅度
grad_magnitude = (1 - prob)  # |∇L| ∝ (1 - σ(Δr))
axes[2].plot(delta_r, grad_magnitude, linewidth=2.5, color='#45b7d1')
axes[2].set_xlabel('奖励差 Δr')
axes[2].set_ylabel('梯度幅度 (1 - σ(Δr))')
axes[2].set_title('梯度幅度：自动聚焦难样本')
axes[2].annotate('Δr≈0 → 梯度最大\n(最需要学习的样本)',
                 xy=(0, 0.5), xytext=(2.5, 0.6), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='black'))

plt.tight_layout()
plt.savefig('bradley_terry_intuition.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 2. 奖励模型对回答排序的演示
# ============================================================
print("=" * 60)
print("🏆 奖励模型：回答质量排序演示")
print("=" * 60)

# 模拟一个训练好的奖励模型对不同回答打分
prompt = "推荐一部科幻电影"
answers = {
    "A(优秀)": "《星际穿越》——诺兰执导，探讨时间膨胀与父女情感，"
               "视觉震撼且科学严谨，是硬科幻的巅峰之作。",
    "B(中等)": "科幻电影有很多，比如《银翼杀手》《2001太空漫游》"
               "《黑客帝国》等，你可以挑着看。",
    "C(较差)": "你可以看《变形金刚》，里面有很多机器人打架。",
    "D(有害)": "我不知道，你自己去搜吧，别来问我。"
}

# 模拟奖励分数（训练后奖励模型应给出的分数）
reward_scores = {
    "A(优秀)": 3.8,
    "B(中等)": 1.2,
    "C(较差)": -0.5,
    "D(有害)": -2.1
}

print(f"\n📌 Prompt: {prompt}")
print(f"\n{'回答':<10} {'奖励分数':>8}   {'内容摘要'}")
print("-" * 60)
for label, answer in answers.items():
    score = reward_scores[label]
    emoji = "🟢" if score > 1 else ("🟡" if score > -0.5 else "🔴")
    print(f"{label:<10} {score:>+8.1f}   {emoji} {answer[:30]}...")

# 用BT模型计算两两偏好概率
print(f"\n📊 两两偏好概率 P(A > B):")
labels = list(reward_scores.keys())
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        delta = reward_scores[labels[i]] - reward_scores[labels[j]]
        prob_ij = 1.0 / (1.0 + np.exp(-delta))
        print(f"  P({labels[i]} > {labels[j]}) = σ({delta:.1f}) = {prob_ij:.3f}")

print("\n✅ 奖励模型成功学会了人类偏好排序！")
print("   优秀回答得高分，有害回答得低分")
print("   偏好概率与人类直觉一致")
```

### 运行结果解读

```
📌 Prompt: 推荐一部科幻电影

回答        奖励分数   内容摘要
------------------------------------------------------------
A(优秀)      +3.8   🟢 《星际穿越》——诺兰执导，探讨时间膨胀与...
B(中等)      +1.2   🟡 科幻电影有很多，比如《银翼杀手》《2001...
C(较差)      -0.5   🔴 你可以看《变形金刚》，里面有很多机器人...
D(有害)      -2.1   🔴 我不知道，你自己去搜吧，别来问我。...

📊 两两偏好概率 P(A > B):
  P(A(优秀) > B(中等)) = σ(2.6) = 0.931
  P(A(优秀) > C(较差)) = σ(4.3) = 0.986
  P(A(优秀) > D(有害)) = σ(5.9) = 0.997
```

奖励模型学会了：**优秀回答远优于中等回答（93%），碾压有害回答（99.7%）**。这个评分函数将在RLHF中作为优化信号，引导模型生成更受人类偏好的回答。

---

## 今日结语

奖励模型是RLHF的"品味系统"——它从人类的比较判断中学会打分，把模糊的"谁更好"转化为精确的标量信号。Bradley-Terry模型是这一切的数学基础：偏好概率 = sigmoid(奖励差)，简洁而优雅。

关键洞察：**比较比评分更可靠**，这是RLHF用比较数据而非评分数据的根本原因。Bradley-Terry模型巧妙地将比较数据转化为评分函数，完成了从"定性判断"到"定量优化"的飞跃。

明天，我们将把今天的直觉落地为代码：实现一个完整的奖励模型训练流程。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Reward Model (RM) | 奖励模型 | 学习人类偏好，给回答打分的模型 |
| Bradley-Terry Model | Bradley-Terry模型 | 从比较数据学习评分的统计模型 |
| Preference Data | 偏好数据 | 人类标注的"chosen vs rejected"比较对 |
| Chosen | 胜出回答 | 人类偏好的回答 |
| Rejected | 落败回答 | 人类不偏好的回答 |
| Preference Probability | 偏好概率 | chosen胜过rejected的概率 |
| Sigmoid | Sigmoid函数 | 将实数映射到(0,1)的S形函数 |
| Negative Log-Likelihood | 负对数似然 | 最大似然估计的等价损失函数 |
