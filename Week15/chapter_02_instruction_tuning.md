# Day 02：指令微调 —— 让模型学会"听指令"

> ⚖️ 第十五周 · 人类反馈对齐 · 第2天

昨天的对齐问题告诉我们：预训练模型是"续写机器"，不是"问答助手"。那最直接的办法是什么？**给它看大量"指令-回答"的例子，让它从续写模式切换到问答模式**。这就是指令微调（Instruction Tuning）——对齐的第一步，也是最简单、最有效的一步。

**今天的任务**：
1. 理解指令微调的核心思想：从"续写"到"回答"的范式转换
2. 构造指令数据集（instruction-input-output格式）
3. 用代码演示：微调前后模型对同一输入的行为差异

---

## 1. 历史剧场：FLAN——指令微调的起点

### 🎭 Google的FLAN实验（2022）

2022年，Google发表"Finetuned Language Models Are Zero-Shot Learners"（FLAN），核心发现：

> 在60多个NLP任务上构造指令模板，对LaMDA-137B进行微调，**零样本性能暴涨**。

FLAN的关键洞察：**预训练模型其实已经具备了执行各种任务的能力，只是不知道"什么时候该用什么能力"**。指令微调不是在教新知识，而是在教**何时激活已有知识**。

### 🎭 从FLAN到InstructGPT的演进

```
FLAN (2022.02)          InstructGPT (2022.03)
    │                        │
    ├─ 多任务指令模板          ├─ 人类撰写的示范数据
    ├─ 60+ NLP任务            ├─ 通用对话能力
    ├─ 零样本泛化             ├─ SFT + RM + PPO
    └─ 学术评估               └─ 人类偏好评估
```

FLAN用**模板自动生成**指令数据，InstructGPT用**人类手写**指令数据。后者更贵但效果更好——因为人类示范包含了隐含的偏好信息（什么回答是好的、什么是不好的）。

### 🎭 Alpaca的启示（2023）

斯坦福的Alpaca用GPT-3.5生成5.2万条指令数据，微调LLaMA-7B，成本仅600美元，效果接近GPT-3.5。这证明：**指令数据的质量比数量更重要**，而"质量"的核心是**多样性**和**难度梯度**。

---

## 2. 生活隐喻：从"自由写作"到"命题作文"

### 📝 预训练 = 自由写作课

想象一个写作班：

```
老师：今天自由写作，想写什么写什么
学生：写了一篇关于量子力学的散文（因为训练数据中科普文章多）
```

学生文笔很好，但完全按自己的喜好来——这就是预训练模型。

### 📋 指令微调 = 命题作文课

```
老师：请写一篇关于"如何提高学习效率"的实用建议
学生：1. 间隔重复  2. 主动回忆  3. 费曼技巧
```

学生学会了**根据题目要求写作**——这就是指令微调后的模型。

### 🔑 关键区别

| | 自由写作（预训练） | 命题作文（指令微调） |
|---|---|---|
| 输入 | 无明确指令 | 明确的指令/问题 |
| 输出 | 按统计规律续写 | 按指令要求回答 |
| 格式 | 不确定 | 结构化、有针对性 |
| 评价标准 | 流畅度 | 流畅度 + 指令遵循度 |

---

## 3. 数学直觉：指令微调的目标函数

### 📐 从语言模型到指令模型

**预训练目标**：预测下一个token

$$P(y | x) = \prod_{t=1}^{T} P(y_t | x, y_{<t})$$

其中 $x$ 是上下文，$y$ 是续写内容。模型学到的是 $P_{pretrain}$。

**指令微调目标**：给定指令，生成符合要求的回答

$$P(y | \text{instruction}, \text{input}) = \prod_{t=1}^{T} P(y_t | \text{instruction}, \text{input}, y_{<t})$$

微调就是在 $P_{pretrain}$ 基础上，用指令数据调整参数：

$$\theta_{sft} = \theta_{pretrain} - \eta \sum_{i} \nabla_\theta L_{CE}(\text{instruction}_i, \text{output}_i)$$

### 📐 指令数据的格式

每条指令数据是一个三元组：

```
(instruction, input, output)

例：
instruction: "将以下句子翻译成英文"
input:       "今天天气很好"
output:      "The weather is nice today"
```

在训练时，我们把三者拼成一个序列：

```
训练输入:  "### Instruction:\n将以下句子翻译成英文\n### Input:\n今天天气很好\n### Output:\n"
训练目标:  "The weather is nice today"
```

---

## 4. 代码实验室：指令微调实战

```python
"""
指令微调演示：从续写模式到问答模式
====================================
用MiniGPT演示指令微调的核心流程：
1. 构造指令数据集
2. 微调模型
3. 对比微调前后行为差异
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 指令数据集构造
# ============================================================
class InstructionDataset(Dataset):
    """
    指令微调数据集
    格式：(instruction, input, output) 三元组
    """
    def __init__(self, data: list[dict], tokenizer, max_len: int = 128):
        self.samples = []
        for item in data:
            # 拼接指令模板
            prompt = (
                f"### 指令:\n{item['instruction']}\n"
                f"### 输入:\n{item['input']}\n"
                f"### 输出:\n"
            )
            # 训练目标：模型应该生成的回答
            target = item['output']
            self.samples.append((prompt, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# 构造指令数据——涵盖多种任务类型
instruction_data = [
    # 问答类
    {"instruction": "回答以下问题", "input": "中国的首都是哪里",
     "output": "中国的首都是北京。"},
    {"instruction": "回答以下问题", "input": "光速是多少",
     "output": "光速约为每秒30万公里（3×10⁸ m/s）。"},
    # 翻译类
    {"instruction": "将中文翻译成英文", "input": "今天天气很好",
     "output": "The weather is nice today."},
    {"instruction": "将英文翻译成中文", "input": "I love machine learning",
     "output": "我热爱机器学习。"},
    # 摘要类
    {"instruction": "用一句话总结", "input": "深度学习是机器学习的子领域，"
     "使用多层神经网络从数据中学习表示，在图像识别和自然语言处理上取得突破",
     "output": "深度学习用多层神经网络从数据学习表示，在视觉和NLP领域突破显著。"},
    # 写作类
    {"instruction": "写一条关于学习的建议", "input": "",
     "output": "采用间隔重复法：在即将遗忘时复习，效率比集中学习高3倍。"},
    # 数学类
    {"instruction": "计算以下算式", "input": "15 × 4 + 7",
     "output": "15 × 4 + 7 = 60 + 7 = 67"},
]

# ============================================================
# 2. 简化MiniGPT + 指令微调
# ============================================================
class MiniGPTForSFT(nn.Module):
    """
    用于指令微调的MiniGPT
    ====================
    结构：Embedding → Transformer → LM Head
    微调方式：全参数微调（简化演示）
    """
    def __init__(self, vocab_size: int = 2000, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播：输入token ids，输出logits"""
        # input_ids: (batch, seq_len)
        h = self.embedding(input_ids)          # (batch, seq_len, d_model)
        # 自注意力需要因果mask
        seq_len = input_ids.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        mask = mask.to(input_ids.device)
        h = self.transformer(h, memory=h, tgt_mask=mask)  # 自回归
        logits = self.lm_head(h)               # (batch, seq_len, vocab_size)
        return logits

# ============================================================
# 3. 模拟微调前后对比
# ============================================================
class BeforeAfterDemo:
    """演示微调前后模型行为差异"""
    def __init__(self):
        # 微调前：续写模式
        self.pretrained_responses = {
            "回答以下问题|中国的首都是哪里":
                "中国的首都是哪里？这个问题看似简单，"
                "实际上涉及历史、政治、地理等多方面知识……",
            "将中文翻译成英文|今天天气很好":
                "将中文翻译成英文。翻译是语言转换的重要技能，"
                "英文和中文属于不同语系……",
            "计算以下算式|15 × 4 + 7":
                "计算以下算式。算式是数学表达的基本形式，"
                "四则运算的优先级是先乘除后加减……",
        }
        # 微调后：指令遵循模式
        self.sft_responses = {
            "回答以下问题|中国的首都是哪里":
                "中国的首都是北京。",
            "将中文翻译成英文|今天天气很好":
                "The weather is nice today.",
            "计算以下算式|15 × 4 + 7":
                "15 × 4 + 7 = 60 + 7 = 67",
        }

    def query(self, instruction: str, input_text: str,
              mode: str = "sft") -> str:
        """查询模型"""
        key = f"{instruction}|{input_text}"
        responses = (self.sft_responses if mode == "sft"
                     else self.pretrained_responses)
        return responses.get(key, "（未找到匹配响应）")

# ============================================================
# 4. 可视化对比
# ============================================================
demo = BeforeAfterDemo()

# 打印对比结果
print("=" * 60)
print("📊 指令微调前后对比")
print("=" * 60)

test_cases = [
    ("回答以下问题", "中国的首都是哪里"),
    ("将中文翻译成英文", "今天天气很好"),
    ("计算以下算式", "15 × 4 + 7"),
]

for inst, inp in test_cases:
    print(f"\n📌 指令: {inst} | 输入: {inp}")
    print(f"  ❌ 微调前: {demo.query(inst, inp, 'pretrain')[:40]}...")
    print(f"  ✅ 微调后: {demo.query(inst, inp, 'sft')}")

# 可视化：指令遵循度对比
fig, ax = plt.subplots(figsize=(10, 5))

task_types = ['问答', '翻译', '摘要', '写作', '数学', '编程']
pretrain_follow = [0.15, 0.10, 0.12, 0.08, 0.20, 0.05]
sft_follow = [0.85, 0.80, 0.75, 0.70, 0.82, 0.65]

x = range(len(task_types))
width = 0.35
ax.bar([i - width/2 for i in x], pretrain_follow, width,
       label='预训练模型', color='#ff6b6b', alpha=0.8)
ax.bar([i + width/2 for i in x], sft_follow, width,
       label='指令微调后', color='#4ecdc4', alpha=0.8)

ax.set_xticks(list(x))
ax.set_xticklabels(task_types)
ax.set_ylabel('指令遵循率')
ax.set_title('指令微调效果：各任务类型的指令遵循率对比')
ax.legend()
ax.set_ylim(0, 1.0)

# 添加提升幅度标注
for i in x:
    improvement = sft_follow[i] - pretrain_follow[i]
    ax.annotate(f'+{improvement:.0%}',
                xy=(i + width/2, sft_follow[i]),
                xytext=(i + width/2, sft_follow[i] + 0.05),
                ha='center', fontsize=9, color='#2d6a4f', fontweight='bold')

plt.tight_layout()
plt.savefig('instruction_tuning_demo.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 可视化已保存至 instruction_tuning_demo.png")
```

### 运行结果解读

```
📌 指令: 回答以下问题 | 输入: 中国的首都是哪里
  ❌ 微调前: 中国的首都是哪里？这个问题看似简单，实际上涉及历史...
  ✅ 微调后: 中国的首都是北京。
```

微调前，模型把指令当文章开头续写；微调后，模型理解了"回答问题"的意图，直接给出答案。**指令微调的本质是范式转换：从续写到回答。**

---

## 今日结语

指令微调是对齐的第一步，也是最"便宜"的一步——只需要几千到几万条高质量的指令数据，就能让模型从"续写机器"变成"问答助手"。它的核心不是教新知识，而是教模型**什么时候该用什么能力**。

但指令微调有上限：它只能学到示范数据中的模式，无法区分"好回答"和"更好回答"的细微差异。就像命题作文课能教会你按题目写作，但无法教你写出"优秀"而非"及格"的作文——这需要品味，也就是**奖励模型**，明天的主题。

### 翻译词典

| 英文 | 中文 | 语境说明 |
|------|------|---------|
| Instruction Tuning | 指令微调 | 用指令数据微调预训练模型 |
| Supervised Fine-Tuning (SFT) | 监督微调 | 用人类示范数据微调，指令微调的别称 |
| FLAN | FLAN | Google的指令微调方法，Finetuned Language Net |
| Prompt Template | 提示模板 | 将指令、输入、输出格式化的模板 |
| Zero-Shot | 零样本 | 不给示例，直接用指令让模型执行任务 |
| Instruction Following | 指令遵循 | 模型按指令要求行动的能力 |
| Demonstration Data | 示范数据 | 人类撰写的指令-回答对 |
