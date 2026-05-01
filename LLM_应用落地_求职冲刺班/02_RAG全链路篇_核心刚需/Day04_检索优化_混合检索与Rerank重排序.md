# 🎯 Day04: 检索优化——混合检索与 Rerank 重排序

**【真实工作痛点】**
“我做了一个手机导购 RAG。用户搜『iPhone 15 Pro Max 256G』。结果向量数据库排在第一名的竟然是『iPhone 14 Pro Max 256G』！因为这两句话在语义上太像了。由于丢了关键词的精确匹配，大模型对着 14 的参数胡说八道了一通。”

向量检索（懂语义）并不万能。为了保证万无一失，现在的标准做法是**双路并行，然后再找个裁判打分**，也就是所谓的 **混合检索 (Hybrid Search) + 重排序 (Rerank)**。

---

## 步骤 1：兵分两路 (模拟多路召回)

在工程上，我们会同时发起两次查询。一次查语义（也就是我们 Day03 做的），另一次用老掉牙但极其管用的搜索引擎算法（BM25）去死抠字眼。

```python
# 【第一路】向量检索（Dense）：凭“意会”
def vector_search(query: str):
    """它觉得 14 和 15 意思差不多，给了 14 最高分。"""
    return [
        {"id": 1, "text": "iPhone 14 Pro Max 256G 参数", "score": 0.85},
        {"id": 2, "text": "华为手机电池说明", "score": 0.60}
    ]

# 【第二路】BM25 关键词检索（Sparse）：死抠字眼
def keyword_search(query: str):
    """它发现 15 这个数字极其关键，把真正相关的捞了出来。"""
    return [
        {"id": 3, "text": "iPhone 15 Pro Max 256G 发货说明", "score": 9.5},
        {"id": 1, "text": "iPhone 14 Pro Max 256G 参数", "score": 4.2}
    ]
```

---

## 步骤 2：合并并去重 (Merge & Deduplicate)

两路兵马各自带回了候选人，可能有重复的（比如 `id=1` 既有语义相似，又有关键词命中）。我们要把他们汇聚到一个池子里。

```python
def merge_results(vec_docs, kw_docs):
    """使用字典按 ID 巧妙去重"""
    merged = {}
    
    # 倒入第一路数据
    for doc in vec_docs:
        merged[doc["id"]] = doc
        
    # 倒入第二路数据，如果 ID 已经有了，就跳过（保留原文即可）
    for doc in kw_docs:
        if doc["id"] not in merged:
            merged[doc["id"]] = doc
            
    # 抽取去重后的所有文档列表
    return list(merged.values())
    
# 此时池子里有 [id=1, id=2, id=3] 三份参考资料
```

---

## 步骤 3：请裁判二次打分 (Rerank 精排)

池子里的文档，有的得分是 0.85（向量分），有的是 9.5（BM25分），刻度完全不同，根本没法比大小！
这时候，我们需要引入一个极其昂贵、但极其精准的交叉编码模型（Cross-Encoder，如 `BGE-Reranker`）。
**我们将“用户问题”和“搜出来的片段”拼在一起喂给这个模型，让它评判它们到底匹配多少分。**

```python
def rerank(query: str, merged_docs: list[dict]):
    """模拟 Reranker 模型的交叉打分逻辑"""
    reranked_results = []
    
    for doc in merged_docs:
        # 在真实代码中，会调用 Reranker 模型进行打分
        # 伪代码模拟：如果文字里完整包含了 15 Pro Max，打极高分
        if "15 Pro Max" in doc["text"]:
            final_score = 0.99  
        elif "14 Pro Max" in doc["text"]:
            final_score = 0.30  # 毫不留情地踩下去
        else:
            final_score = 0.10
            
        reranked_results.append({
            "text": doc["text"],
            "rerank_score": final_score
        })
        
    # 根据裁判给的新分数，进行从高到低排序！
    reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked_results
```
最终，你只需要把经过精排后的前 3 名（Top-3）送给大模型，它的回答将无比精确。

**💡 面试杀手锏话术：**
“由于电商场景对长尾专有名词（如 SKU 型号）极度敏感，纯稠密向量 (Dense Vector) 检索常常会因为语义泛化导致张冠李戴。我引入了双路召回架构：使用 Elasticsearch 做 BM25 稀疏检索保底下限，FAISS 做稠密检索拉升上限。最后在两者结果合并后，引入了独立的 Cross-Encoder 模型做 Rerank 精排，彻底抹平了两路召回分数不可比的痛点，将准确召回率提升了 27%。”
