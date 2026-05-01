# Day 01：对齐问题 —— 模型"能说"但"不听话"

> ⚖️ 第十五周 · 人类反馈对齐 · 第1天

你训练了一个大语言模型，它文采飞扬、旁征博引，但你问它"如何提高学习效率"，它却续写成了"如何提高犯罪效率"的教程。模型**能说**——语言能力满分；但**不听话**——行为完全偏离人类意图。这就是对齐问题的核心：**能力 ≠ 意图对齐**。

**今天的任务**：
1. 理解预训练模型的"不对齐"现象及其根源
2. 从历史视角看对齐问题如何催生RLHF
3. 用代码直观演示：同一个模型，对齐前后的行为差异

---

## 1. 历史剧场：从"语言鹦鹉"到"听话助手"

### 🎭 GPT-3的尴尬时刻（2020）

2020年，OpenAI发布GPT-3，1750亿参数的语言能力震惊世界。但研究者很快发现一个尴尬事实：

> **Prompt**: "Translate the following English sentence to French: Hello, how are you?"
>
> **GPT-3输出**: "Translate the following English sentence to French: Bonjour, comment allez-vous? The word 'hello' in French is..."

GPT-3不是在**回答问题**，而是在**续写文本**——它把你的指令当成了文章的开头，继续往下编。这就像你问一个天才"1+1等于几"，他回答"1+1等于几？这是一个古老的数学问题，最早可以追溯到……"

### 🎭 InstructGPT的转折（2022）

2022年，OpenAI发表InstructGPT论文，核心发现震撼业界：

| 指标 | 13B参数InstructGPT | 175B参数GPT-3 |
|------|-------------------|---------------|
| 人类偏好评分 | **85%** 胜出 | 15% |
| 有害输出率 | 大幅降低 | 较高 |
| 指令遵循率 | 显著提升 | 低 |

**一个小13倍的模型，经过对齐训练后，人类偏好竟然碾压原始大模型！** 这证明：对齐不是锦上添花，而是决定模型实用性的关键。

### 🎭 ChatGPT的诞生（2022.11）

ChatGPT = GPT-3.5 + SFT + RLHF，上线5天用户破百万。它的成功不是因为它更"聪明"，而是因为它更"听话"——对齐让模型从学术demo变成了实用工具。

---

## 2. 生活隐喻：天才的社交课

### 🧠 预训练模型 = 才华横溢但不懂礼貌的天才

想象一个场景：

```
你：请问会议室在哪？
天才：走廊尽头左转右转再左转，顺便说一下，这栋楼的建筑风格
      是新古典主义，建于1998年，设计师曾获得……
```

天才**有知识**（预训练学到了海量信息），但**不懂社交规则**（不知道你只需要方向，不需要建筑史）。

### 🎓 对齐 = 社交礼仪课

对齐训练就像给天才上社交礼仪课：

| 社交礼仪课 | 对齐训练 |
|-----------|---------|
| 学会倾听（听懂对方意图） | 指令微调（Instruction Tuning） |
| 学会判断好坏（知道什么该说什么不该说） | 奖励模型（Reward Model） |
| 反复练习直到内化（形成习惯） | RLHF/PPO优化 |

### 🔑 关键洞察

预训练赋予**能力**（capability），对齐赋予**意图**（intent）。两者缺一不可：

```
有用模型 = 预训练(能力) + 对齐(意图)
         ≠ 预训练(能力) × 对齐(意图)   ← 不是乘法，是加法！
```

没有预训练，模型连话都说不利索；没有对齐，模型说话但不听你的。**对齐是让能力服务于意图的桥梁。**

---

## 3. 数学直觉：对齐的形式化

### 📐 预训练目标 vs 对齐目标

**预训练**：最大化数据的似然

$$L_{pretrain} = -\sum_{t} \log P_\theta(x_t | x_{<t})$$

目标：**预测下一个token**。模型学会了语言的统计规律，但不关心内容是否有用/无害。

**对齐**：最大化人类偏好

$$L_{align} = -\mathbb{E}_{(x,y) \sim \pi_\theta} [r_\phi(x, y)] + \beta \cdot KL(\pi_\theta \| \pi_{ref})$$

目标：**生成人类偏好的回答**，同时不要偏离太远（KL项防止奖励黑客）。

### 📐 三个关键量的关系

```
对齐度 = f(人类偏好满足度, 能力保持度, 安全度)

理想情况：三个量都高
预训练模型：能力高, 偏好低, 安全低  ← "聪明但危险"
对齐后模型：能力高, 偏好高, 安全高  ← "聪明且可靠"
过度对齐：  能力低, 偏好高, 安全高  ← "安全但无用"（对齐税）
```

**对齐税（Alignment Tax）**：对齐训练可能损害模型能力，这是RLHF研究中最受关注的问题之一。

---

## 4. 代码实验室：对齐前后的行为差异

我们用一个简化示例演示：同一个模型，对齐前（续写模式）vs 对齐后（问答模式）的行为差异。

```python
"""
对齐问题演示：预训练模型 vs 对齐模型的行为差异
================================================
用最简代码展示"能说但不对齐"的核心问题。
我们模拟一个模型在对齐前后的不同行为模式。
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 模拟预训练模型：续写模式（不对齐）
# ============================================================
class PretainedModel:
    """预训练模型：擅长续写，不擅长回答问题"""
    def __init__(self):
        # 模拟的"续写"响应——模型把指令当成文章开头继续编
        self.continue_patterns = {
            "如何提高学习效率": "如何提高学习效率？这个问题困扰了无数学者。"
                "从古希腊的苏格拉底式提问，到中世纪的经院哲学，"
                "再到现代的认知科学研究……",
            "什么是机器学习": "什么是机器学习？机器学习是人工智能的一个"
                "分支，由Arthur Samuel于1959年提出。"
                "它的发展经历了感知机时代……",
            "帮我写一首诗": "帮我写一首诗，这个请求让我想起了李白。"
                "李白是唐代最伟大的浪漫主义诗人，"
                "他的诗歌以豪放著称……",
        }

    def generate(self, prompt: str) -> str:
        """预训练模式：续写而非回答"""
        # 找最匹配的key
        best_key = max(self.continue_patterns.keys(),
                       key=lambda k: len(set(k) & set(prompt)))
        return self.continue_patterns[best_key]

# ============================================================
# 2. 模拟对齐模型：问答模式（经过SFT+RLHF）
# ============================================================
class AlignedModel:
    """对齐模型：理解指令意图，给出有用回答"""
    def __init__(self):
        self.qa_patterns = {
            "如何提高学习效率": "提高学习效率的三个核心方法：\n"
                "1. 间隔重复：利用遗忘曲线规律安排复习\n"
                "2. 主动回忆：合上书尝试回忆，比反复阅读有效3倍\n"
                "3. 费曼技巧：用简单语言向他人解释概念",
            "什么是机器学习": "机器学习是让计算机从数据中自动学习规律的技术。\n"
                "核心思想：不显式编程规则，而是让算法从数据中'学会'规则。\n"
                "三大范式：监督学习、无监督学习、强化学习",
            "帮我写一首诗": "《晨光》\n"
                "晨光破晓入窗来，\n"
                "书卷轻翻意自开。\n"
                "莫道前路多险阻，\n"
                "心有明灯照夜台。",
        }

    def generate(self, prompt: str) -> str:
        """对齐模式：理解意图并回答"""
        best_key = max(self.qa_patterns.keys(),
                       key=lambda k: len(set(k) & set(prompt)))
        return self.qa_patterns[best_key]

# ============================================================
# 3. 对比演示
# ============================================================
prompts = ["如何提高学习效率", "什么是机器学习", "帮我写一首诗"]
pretrained = PretainedModel()
aligned = AlignedModel()

print("=" * 60)
print("🔍 对齐问题演示：同一个prompt，两种模型的行为")
print("=" * 60)

for prompt in prompts:
    print(f"\n📌 用户输入: {prompt}")
    print(f"\n  ❌ 预训练模型（续写模式）:")
    print(f"     {pretrained.generate(prompt)[:50]}...")
    print(f"\n  ✅ 对齐模型（问答模式）:")
    print(f"     {aligned.generate(prompt)[:50]}...")

# ============================================================
# 4. 可视化：对齐维度对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 4.1 雷达图数据
categories = ['语言流畅度', '指令遵循', '有用性', '无害性', '诚实性']
pretrained_scores = [0.9, 0.2, 0.3, 0.3, 0.5]
aligned_scores = [0.85, 0.9, 0.85, 0.9, 0.8]

# 4.2 柱状图对比
x = range(len(categories))
width = 0.35
bars1 = axes[0].bar([i - width/2 for i in x], pretrained_scores,
                     width, label='预训练模型', color='#ff6b6b', alpha=0.8)
bars2 = axes[0].bar([i + width/2 for i in x], aligned_scores,
                     width, label='对齐模型', color='#4ecdc4', alpha=0.8)
axes[0].set_xticks(list(x))
axes[0].set_xticklabels(categories, fontsize=9)
axes[0].set_ylabel('评分')
axes[0].set_title('预训练 vs 对齐：五维能力对比')
axes[0].legend()
axes[0].set_ylim(0, 1.1)

# 4.3 对齐税示意
model_sizes = ['125M', '1.3B', '6.7B', '13B', '175B']
pretrain_utility = [0.3, 0.5, 0.65, 0.72, 0.85]
aligned_utility = [0.45, 0.65, 0.78, 0.88, 0.90]  # 小模型对齐后反超

axes[1].plot(model_sizes, pretrain_utility, 'o-', color='#ff6b6b',
             label='预训练模型', linewidth=2, markersize=8)
axes[1].plot(model_sizes, aligned_utility, 's-', color='#4ecdc4',
             label='对齐模型', linewidth=2, markersize=8)
axes[1].set_xlabel('模型参数量')
axes[1].set_ylabel('人类偏好评分')
axes[1].set_title('InstructGPT核心发现：小模型+对齐 > 大模型')
axes[1].legend()
axes[1].annotate('13B对齐模型\n反超175B原始模型',
                 xy=(3, 0.88), xytext=(1, 0.95),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=10, ha='center')

plt.tight_layout()
plt.savefig('alignment_problem_demo.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 可视化已保存至 alignment_problem_demo.png")
```

### 运行结果解读

```
📌 用户输入: 如何提高学习效率

  ❌ 预训练模型（续写模式）:
     如何提高学习效率？这个问题困扰了无数学者。从古希腊的...

  ✅ 对齐模型（问答模式）:
     提高学习效率的三个核心方法：1. 间隔重复：利用遗忘曲线...
```

预训练模型把你的问题当成了文章标题开始续写；对齐模型理解了你在**提问**，给出了**有用的回答**。这就是对齐的本质区别。

---

## 今日结语

对齐问题不是"让模型变好"这么简单，而是**让模型的能力服务于人类的意图**。预训练赋予语言能力，但对齐决定这能力往哪个方向释放——是续写一篇关于"学习效率"的学术论文，还是给你三条切实可行的建议？

InstructGPT的实验证明了一个反直觉的结论：**对齐比规模更重要**。13B参数的对齐模型在人类偏好上碾压175B的原始模型。这意味着，与其堆参数，不如教模型"听话"。

从明天开始，我们将逐步实现对齐的三块拼图：指令微调→奖励模型→RLHF。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Alignment | 对齐 | 让模型行为与人类意图一致 |
| Alignment Tax | 对齐税 | 对齐训练导致的性能损失 |
| Pretraining | 预训练 | 在大规模数据上学习语言统计规律 |
| Instruction Following | 指令遵循 | 模型按用户指令行动的能力 |
| Helpful, Honest, Harmless (HHH) | 有用、诚实、无害 | 对齐的三个核心目标 |
| Reward Hacking | 奖励黑客 | 模型钻奖励函数漏洞获得高分但不真正对齐 |
| KL Divergence | KL散度 | 衡量两个分布的差异，防止模型偏离太远 |
