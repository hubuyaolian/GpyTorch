# 🔪 Day01: 文档解析与切分 (Chunking) 的艺术

**【真实工作痛点】**
“老板给了我一个 100 页的 PDF 公司规章制度，让我做一个问答机器人。我直接把全文丢给大模型，结果超出了 Token 限制。后来我按每 500 个字生硬地切断，结果问‘年假怎么算’时，前一半内容在 Chunk A，后一半在 Chunk B，大模型什么也回答不出来。”

在 RAG 链路中，**切分（Chunking）是决定成败的第一步**。切得不好，后文的检索和生成全盘崩溃。为了让你彻底搞懂，我们不调用任何 LangChain 封装好的黑盒，直接把底层的三种切分逻辑**掰开揉碎了写一遍**。

---

## 1. 最粗暴的切法：按固定长度切分 (Naive Split)

这是最反人类直觉的切法，也是很多无经验新手的默认做法。我们硬生生地按照 `chunk_size` 去截断字符串。

```python
def naive_split_by_length(text: str, chunk_size: int) -> list[str]:
    """策略一：最粗暴的按固定字数切分（反面教材）。"""
    chunks = []
    current_index = 0
    text_length = len(text)
    
    # 只要没切完，就一直切
    while current_index < text_length:
        # 算出这一刀切到哪里（当前位置 + 切片长度）
        end_index = current_index + chunk_size
        # 用 Python 切片语法截取这段文字
        chunk = text[current_index:end_index]
        chunks.append(chunk)
        
        # 把刀挪到下一个起点
        current_index = end_index
        
    return chunks
```

**🔍 致命缺陷分析：**
如果你传入 `"今天天气非常好。我们决定去公园野餐。"`，并设置 `chunk_size = 10`。
切出来的第一块是：`"今天天气非常好。我们决"`。
第二块是：`"定去公园野餐。"`。
**一句话被腰斩了！语义彻底断裂。**

---

## 2. 稍微聪明的切法：按标点符号切分

既然固定长度会切断句子，那我们遇到“句号”再切，总行了吧？

```python
def split_by_punctuation(text: str) -> list[str]:
    """策略二：按标点符号切分，保证句子的完整性。"""
    chunks = []
    current_chunk = ""
    
    for char in text:
        current_chunk += char
        # 一旦遇到中文句号，说明这句话说完了
        if char == "。":
            chunks.append(current_chunk)
            # 清空缓存，准备装载下一句话
            current_chunk = ""
            
    # 兜底：如果最后一段话没有句号结尾，也要加进去
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks
```

**🔍 致命缺陷分析：**
这确实保住了单句话的完整性。但如果作者写了一段长达 1000 字、没有句号的长难句（或者英文文献），这个切块还是会远远超出后续大模型和向量库的处理极限。

---

## 3. 工业界标配：滑动窗口切分 (Sliding Window with Overlap)

为了既控制最大长度，又防止上下文（尤其是代词如“他”、“这件事”）断裂，我们引入了**重叠区 (Overlap)**。
切第一刀的时候，我们切下 `[0:20]`。
切第二刀的时候，我们**往回退几步**，切下 `[15:35]`。这样它们中间就有 5 个字的重叠（缓冲地带）。

```python
def sliding_window_split(text: str, chunk_size: int, overlap_size: int) -> list[str]:
    """策略三：滑动窗口切分（带 Overlap 重叠），防止上下文断裂。"""
    chunks = []
    current_index = 0
    text_length = len(text)
    
    # 核心数学逻辑：每次往前走多少步？
    # 步长 = 切片总长度 - 想要的重叠长度
    step = chunk_size - overlap_size
    
    # 防呆设计：重叠区不能比切片还大，否则就死循环倒退了
    if step <= 0:
        raise ValueError("重叠大小不能大于或等于切片大小！")
        
    while current_index < text_length:
        end_index = current_index + chunk_size
        chunk = text[current_index:end_index]
        chunks.append(chunk)
        
        # 关键点：指针按照 step 前进，而不是直接跳到 end_index
        current_index += step
        
    return chunks
```

**💡 面试杀手锏话术：**
“在我们的业务中，由于涉及到法律条文，跨段落的指代非常频繁。我放弃了简单的长度切分，而是基于双换行符（段落）做一级切分，然后在段落内部使用带 **15% Overlap 的滑动窗口**进行二级切分。这样既没有破坏条文结构，又保证了召回片段的语义连贯性。”
