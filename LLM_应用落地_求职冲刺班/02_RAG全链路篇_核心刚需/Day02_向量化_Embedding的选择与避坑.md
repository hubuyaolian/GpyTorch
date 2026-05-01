# 🌌 Day02: 向量化 (Embedding) 的选择与降维

**【真实工作痛点】**
“我用 OpenAI 的 API 做知识库，结果遇到两个大坑：第一，搜索‘请假’，它把‘报销’也搜出来了，在老外模型眼里它们都是‘行政规定’，中文细分极差。第二，公司财务数据根本不许传给外网 API，合规直接把项目卡死了。”

**Embedding 就是把“文字”变成数学上的“坐标（浮点数数组）”**。我们需要本地私有化部署一个专门针对中文优化的开源模型（如 `bge-small-zh`）。下面我们把 PyTorch 转化为向量的过程，拆成 4 步一点点看。

---

## 步骤 1：加载分词器 (Tokenizer) 和模型 (Model)

计算机是不认识汉字的，它只认识数字。所以我们需要一个“字典”（Tokenizer）把汉字变成 ID，再把 ID 喂给模型。

```python
import torch
from transformers import AutoTokenizer, AutoModel

# 1. 告诉代码我们要用 HuggingFace 上的哪个开源模型
model_name = "BAAI/bge-small-zh-v1.5"

# 2. 加载分词器字典
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. 加载模型本体（神经网络权重）
model = AutoModel.from_pretrained(model_name)

# 4. 极其关键的一步：将模型设置为“评估模式” (Evaluation Mode)
# 这告诉 PyTorch：我们只是用来推理，不要去计算和保存用于训练的梯度，从而省下巨量内存。
model.eval()
```

---

## 步骤 2：把人类文字变成张量 (Tensor)

我们拿到了一句话 `"年假五天"`，现在要用刚才的字典把它转译成模型认识的数字张量。

```python
text = "公司规定，员工每年有五天带薪年假。"

# 翻译文字为 Tensor
encoded_input = tokenizer(
    text, 
    padding=True,      # 如果同时传好几句话，短的句子用 0 补齐长度
    truncation=True,   # 如果句子太长超出了模型限制（通常 512 token），直接砍掉尾巴
    return_tensors="pt" # "pt" 代表返回 PyTorch 格式的 Tensor 数据
)
# 此时 encoded_input 里面包含 input_ids（字的序号）和 attention_mask（区分字和补齐的0）
```

---

## 步骤 3：进行前向传播计算 (Forward Pass)

数据准备好了，扔进神经网络里进行疯狂的矩阵乘法，提炼出语义特征。

```python
# torch.no_grad() 再次强调：不要算梯度！纯为了提速和省显存。
with torch.no_grad():
    # 将字典里翻译好的输入，全部解包 (**) 喂给模型
    model_output = model(**encoded_input)
    
# 模型吐出的结果是一个极复杂的元组，里面包含了每一层的状态。
```

---

## 步骤 4：提取核心向量并做 L2 归一化

模型吐出了一大堆数据（每个字都有一个特征向量），但我们只需要**能代表整句话意思**的那个向量。
在 BERT 系列模型中，通常放在句首的特殊的 `[CLS]` 标记的特征，就被设计用来代表全局语义。

```python
# 从输出中提取特征：
# 0: 取模型输出的第一项 (last_hidden_state)
# 第一个 0: 取 batch 里的第一句话
# 第二个 0: 取这句话的第一个 token（也就是 [CLS] 标记）
# : : 取这个 token 所有的维度数据 (例如 512 个维度的浮点数)
sentence_embedding = model_output[0][0, 0, :]

# 【高阶工程技巧】：L2 归一化
# 向量的长短会影响计算速度。把向量“拉伸或缩短”到长度为 1（在一个单位球面上）。
# 这样做的好处是：后续我们算“余弦相似度”时，只需算极其简单的“点积”即可！大大加速。
normalized_embedding = torch.nn.functional.normalize(
    sentence_embedding, 
    p=2, 
    dim=0
)

# 把张量转换回普通的 Python 列表，准备存入数据库
final_vector = normalized_embedding.tolist()
```

**💡 面试杀手锏话术：**
“在向量化阶段，我们不仅选用了 BGE 中文模型解决了数据出域的合规问题，我还特别在模型输出端做了 **L2 归一化处理**。这使得我们在后续向量数据库中，可以直接使用底层的**内积 (Inner Product) 运算**来代替复杂的**余弦相似度 (Cosine) 公式**，让上游的检索耗时降低了 30%。”
