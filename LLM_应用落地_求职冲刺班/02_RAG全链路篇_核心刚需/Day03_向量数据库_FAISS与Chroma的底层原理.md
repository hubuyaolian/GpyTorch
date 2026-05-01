# 🗄️ Day03: 向量数据库 FAISS 与相似度检索原理

**【真实工作痛点】**
“我把 10 万个切片变成了 10 万个浮点数组（向量）。每次用户提问，我也把它转成 1 个向量。然后写个 `for` 循环，让这 1 个向量去和那 10 万个向量挨个算数学距离，找出最近的 3 个。结果每次问答都要卡死 5 秒钟！”

这种做法叫“暴力穷举（Brute Force）”。在生产环境里，我们需要用 **索引 (Index)** 来极速找邻居。Meta（Facebook）开源的 **FAISS** 是目前全世界最底层的核心库，几乎所有高档的向量数据库（Chroma, Milvus）底层都离不开它。

我们不搞花里胡哨的外壳，直接看看 FAISS 里面到底是怎么存、怎么找的。

---

## 步骤 1：建立你的第一个向量索引 (Index)

索引就像是给字典建目录。在 FAISS 中，建什么样的索引，决定了你找得有多快、算得有多准。

```python
import faiss
import numpy as np

# 告诉 FAISS，我们一会儿传进来的数组有多少维 (取决于你 Day02 用的 Embedding 模型)
dimension = 512 

# IndexFlatIP 是一种特殊的索引：
# Flat 代表它依然是扁平的精确计算（数据量极小的时候用，绝对不会漏掉最相似的）。
# IP 代表 Inner Product（内积）。
# 还记得我们在 Day02 做的 L2 归一化吗？两个归一化向量的内积，在数学上完全等价于它们的余弦相似度！
index = faiss.IndexFlatIP(dimension)

print(f"索引已创建，当前包含的向量数: {index.ntotal}")
```

---

## 步骤 2：把数据喂进 FAISS

FAISS 是用 C++ 写的，为了极致的速度，它非常挑食。它不接受普通的 Python 列表（List），只接受 `numpy` 数组，并且必须强转成 `float32` 类型。

```python
# 假设这是我们之前算好的 3 个知识库向量（用随便写的数字代替演示）
vectors_list = [
    [0.8, 0.2, ...], # 知识 1（512维）
    [0.1, 0.9, ...], # 知识 2
    [0.5, 0.5, ...]  # 知识 3
]

# 强制转换：转成 numpy 二维矩阵，且必须指定类型为 float32
vectors_array = np.array(vectors_list).astype('float32')

# 写入 FAISS 的 C++ 内存空间
index.add(vectors_array)
```
*注意：FAISS 本身只存数字！它不管这段数字原来是哪句中文。在实际工程中，我们需要在旁边用个数据库（或者简单的 Dict）把 FAISS 返回的 ID 和原文映射起来。*

---

## 步骤 3：大海捞针，极速检索 Top-K

现在用户来提问了。我们把用户的问题也变成向量，扔进 FAISS 里查。

```python
# 用户问题的向量，同样必须转成二维 numpy float32 数组格式 (1 x 512 维)
query_vector = [0.9, 0.1, ...] 
query_array = np.array([query_vector]).astype('float32')

# 我们要找最相似的 3 个片段 (Top-K = 3)
top_k = 3

# search 方法会极速返回两个矩阵：
# distances: 这 3 个片段与用户问题的相似度得分（因为是内积，越高越好）。
# indices:   这 3 个片段在 FAISS 里的内部数字 ID。
distances, indices = index.search(query_array, top_k)

# 打印第一个问题的第一名命中结果
best_match_id = indices[0][0]
best_match_score = distances[0][0]
print(f"最匹配的库中 ID: {best_match_id}, 得分: {best_match_score}")
```

**💡 面试杀手锏话术：**
“当知识库切片达到几十万量级时，纯 `IndexFlatIP` 虽然精确，但遍历耗时依然过高。我主导将底层索引改为了 **HNSW (分层导航小世界算法)**。它通过在内存里建立多层相连的图结构，实现了近似最近邻搜索（ANN）。我们用 2% 的准确率损失，换来了时间复杂度从 $O(N)$ 暴降到 $O(\log N)$，硬生生把 RAG 的平均检索延迟压回了 50ms 的及格线内。”
