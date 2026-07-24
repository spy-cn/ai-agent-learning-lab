# TF-IDF：词频-逆文档频率

词袋模型简单地用"词出现次数"表示文本，但这带来一个问题：**高频词（如"的""是""在"）主导了向量，而真正有区分度的词被淹没**。TF-IDF（Term Frequency-Inverse Document Frequency）正是为解决这一问题而生，它通过衡量词的**稀有程度**来重新加权，是信息检索和文本分类中最经典的权重方案。

---

## 一、核心思想

> **一个词的重要性 = 它在当前文档中出现的频率 × 它在整个语料库中的稀有程度**

```
┌──────────────────────────────────────────────────────────┐
│                  TF-IDF 直觉                             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  文档 D: "自然语言处理 是 AI 的 重要 方向，              │
│           自然语言处理 很 有趣"                          │
│                                                          │
│  词频排序:                                               │
│    自然语言处理: 2次  ←  TF高                            │
│    的: 1次                                               │
│    是: 1次                                               │
│    AI: 1次                                               │
│    ...                                                   │
│                                                          │
│  问题："自然语言处理"和"的"TF可能差不多，                │
│        但"自然语言处理"更能代表这篇文档！                │
│                                                          │
│  解决：IDF 降低"所有文档都出现的常见词"的权重            │
│                                                          │
│  ┌─────────────────────────────────────────────┐         │
│  │             TF × IDF                        │         │
│  │  "的"       高TF × 低IDF = 低权重 ✓         │         │
│  │  "自然语言处理" 高TF × 高IDF = 高权重 ✓      │         │
│  └─────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────┘
```

---

## 二、公式推导

### 2.1 TF（Term Frequency，词频）

TF 衡量词 t 在文档 d 中出现的频繁程度：

```
                词 t 在文档 d 中出现的次数
    TF(t, d) = ────────────────────────────────
                文档 d 中所有词的总数
```

**最简单的形式**（原始计数归一化）：

$$TF(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

其中 $f_{t,d}$ 是词 t 在文档 d 中的原始出现次数。

### 2.2 IDF（Inverse Document Frequency，逆文档频率）

IDF 衡量词 t 在整个语料库中的稀有程度：

```
                    N（文档总数）
    IDF(t) = log ────────────────────────────
                 包含词 t 的文档数 + 1
```

**完整形式**（加 1 平滑，避免除零）：

$$IDF(t) = \log\frac{N + 1}{DF(t) + 1} + 1$$

其中：
- $N$ 是语料库文档总数
- $DF(t)$ 是包含词 t 的文档数
- 分子分母都 +1 是平滑技巧（sklearn 默认实现）
- 加 1 保证 IDF 非负

### 2.3 TF-IDF = TF × IDF

$$TF\text{-}IDF(t, d) = TF(t, d) \times IDF(t)$$

**直觉理解**：

```
    TF-IDF 高 ← TF 高（文档中频繁出现）
              AND
              IDF 高（语料库中稀有）

    TF-IDF 低 ← TF 低（文档中罕见）
              OR
              IDF 低（语料库中常见，如停用词）
```

### 2.4 计算示例

```
语料库（3篇文档）：
    D1: "自然 语言 处理"        → 包含"自然"
    D2: "自然 语言 很 有趣"      → 包含"自然"
    D3: "深度 学习 很 有趣"      → 不含"自然"

计算词"自然"在 D1 中的 TF-IDF：

    N = 3（共3篇文档）
    DF("自然") = 2（D1, D2 包含"自然"）

    TF("自然", D1) = 1/3 ≈ 0.333

    IDF("自然") = log(3/2) + 1
               = log(1.5) + 1
               ≈ 0.405 + 1
               = 1.405

    TF-IDF("自然", D1) = 0.333 × 1.405 ≈ 0.468

对比词"处理"（只在 D1 出现）：
    DF("处理") = 1
    IDF("处理") = log(3/1) + 1 = log(3) + 1 ≈ 2.099
    TF-IDF("处理", D1) = (1/3) × 2.099 ≈ 0.700

→ "处理" 的 TF-IDF > "自然" 的 TF-IDF
  因为 "处理" 更稀有，更能代表 D1
```

---

## 三、手动实现 TF-IDF

理解原理的最佳方式是自己实现一遍。

```python
import math
from collections import Counter

class SimpleTFIDF:
    """A from-scratch TF-IDF implementation for learning purposes."""

    def __init__(self):
        self.vocabulary_ = {}   # word -> index
        self.idf_ = {}          # word -> idf value

    def fit(self, documents):
        """Build vocabulary and compute IDF for each word."""
        # documents: list of word lists (already tokenized)
        N = len(documents)
        df = Counter()  # document frequency

        for doc in documents:
            unique_words = set(doc)
            for word in unique_words:
                df[word] += 1

        # Build vocabulary (sorted for reproducibility)
        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(df))}

        # Compute IDF with smoothing: idf = log((N+1)/(df+1)) + 1
        for word, freq in df.items():
            self.idf_[word] = math.log((N + 1) / (freq + 1)) + 1

        return self

    def transform(self, documents):
        """Convert documents to TF-IDF vectors (unnormalized)."""
        import numpy as np
        V = len(self.vocabulary_)
        matrix = np.zeros((len(documents), V))

        for i, doc in enumerate(documents):
            total_words = len(doc)
            word_counts = Counter(doc)
            for word, count in word_counts.items():
                if word in self.vocabulary_:
                    idx = self.vocabulary_[word]
                    tf = count / total_words            # Term frequency
                    idf = self.idf_[word]               # Inverse doc frequency
                    matrix[i, idx] = tf * idf           # TF-IDF

        # L2 normalization (like sklearn's TfidfVectorizer default)
        norms = np.sqrt((matrix ** 2).sum(axis=1, keepdims=True))
        norms[norms == 0] = 1  # avoid division by zero
        matrix = matrix / norms
        return matrix

    def fit_transform(self, documents):
        return self.fit(documents).transform(documents)


# ---------- Test the implementation ----------
docs = [
    ["自然", "语言", "处理"],
    ["自然", "语言", "很", "有趣"],
    ["深度", "学习", "很", "有趣"]
]

tfidf = SimpleTFIDF()
matrix = tfidf.fit_transform(docs)

print("=== 手动实现的 TF-IDF ===")
print(f"词表: {list(tfidf.vocabulary_.keys())}")
print(f"词表大小: {len(tfidf.vocabulary_)}")
print(f"\nIDF 值:")
for word, idf_val in sorted(tfidf.idf_.items(), key=lambda x: -x[1]):
    print(f"  {word}: {idf_val:.4f}")

print(f"\nTF-IDF 矩阵形状: {matrix.shape}")
print(f"矩阵:\n{matrix.round(4)}")
```

---

## 四、sklearn TfidfVectorizer 用法

实际工作中，直接使用 sklearn 优化过的 `TfidfVectorizer`。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import pandas as pd

# ---------- Prepare corpus ----------
corpus = [
    "自然语言处理是人工智能的重要分支",
    "深度学习和机器学习是AI的核心技术",
    "自然语言处理与计算机视觉是两大方向",
    "深度学习推动了自然语言处理的突破",
    "计算机视觉专注于图像和视频的理解"
]

def chinese_tokenizer(text):
    return list(jieba.cut(text))

# ---------- Configure TfidfVectorizer ----------
vectorizer = TfidfVectorizer(
    tokenizer=chinese_tokenizer,
    token_pattern=None,
    ngram_range=(1, 2),        # unigrams + bigrams
    max_df=0.9,                # ignore terms in >90% documents
    min_df=1,                  # keep terms appearing in at least 1 doc
    max_features=100,
    norm='l2',                 # L2 normalization (default)
    smooth_idf=True,           # IDF smoothing (default)
    sublinear_tf=False         # if True, TF = 1 + log(count)
)

# ---------- Fit and transform ----------
tfidf_matrix = vectorizer.fit_transform(corpus)

# ---------- Inspect ----------
print("=== sklearn TfidfVectorizer ===")
print(f"词表大小: {len(vectorizer.vocabulary_)}")
print(f"矩阵形状: {tfidf_matrix.shape}")

# View IDF values
feature_names = vectorizer.get_feature_names_out()
idf_values = vectorizer.idf_
idf_df = pd.DataFrame({'word': feature_names, 'idf': idf_values})
idf_df = idf_df.sort_values('idf', ascending=False)

print("\nIDF 最高的10个词（最稀有）:")
print(idf_df.head(10).to_string(index=False))

print("\nIDF 最低的10个词（最常见）:")
print(idf_df.tail(10).to_string(index=False))

# ---------- View TF-IDF matrix as DataFrame ----------
df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names,
    index=[f"Doc{i+1}" for i in range(len(corpus))]
)
# Show top features for each document
print("\n各文档 TF-IDF 最高的5个特征:")
for i in range(len(corpus)):
    top5 = df.iloc[i].sort_values(ascending=False).head(5)
    print(f"\n  Doc{i+1}: {corpus[i][:20]}...")
    for word, score in top5.items():
        print(f"    {word}: {score:.4f}")
```

### TfidfVectorizer 关键参数

| 参数 | 说明 | 常用值 |
|------|------|--------|
| `norm` | 归一化方式 | `'l2'`（默认）, `'l1'`, None |
| `smooth_idf` | IDF 平滑 | True（默认） |
| `sublinear_tf` | 亚线性 TF | True/False |
| `max_df` | 最大文档频率 | 0.85-0.95 |
| `min_df` | 最小文档频率 | 1-5 |
| `ngram_range` | n-gram 范围 | (1,1), (1,2) |

---

## 五、TF-IDF 变体

标准 TF-IDF 有多种改进变体，适应不同场景。

```
┌──────────────────────────────────────────────────────────────┐
│                    TF-IDF 变体总览                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ① 对数 TF（Logarithmic TF）                                 │
│     TF = 1 + log(count)     if count > 0                     │
│     TF = 0                  if count = 0                     │
│     作用：抑制极高词频的主导效应                              │
│     场景：sklearn sublinear_tf=True                          │
│                                                              │
│  ② 归一化 IDF（Normalized IDF）                              │
│     IDF = log(N / df)        （标准）                        │
│     IDF = log((N+1)/(df+1)) + 1  （平滑版，sklearn默认）     │
│     作用：避免 df=0 时除零，防止负值                         │
│                                                              │
│  ③ 亚线性 TF（Sublinear TF，同①）                            │
│     对数缩放，让"出现10次"和"出现1次"差异不那么大            │
│                                                              │
│  ④ 最大 TF 归一化（Max TF Normalization）                    │
│     TF = count / max_count_in_doc                            │
│     作用：消除长短文档的差异                                  │
│                                                              │
│  ⑤ 旋转 IDF（Probabilistic IDF）                             │
│     IDF = log((N - df) / df)                                 │
│     作用：更严格的稀有度惩罚（可能产生负值）                  │
└──────────────────────────────────────────────────────────────┘
```

### 变体效果对比

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

corpus = [
    "AI AI AI AI AI 深度 学习 自然 语言 处理",
    "深度 学习 改变 世界 AI 时代",
    "自然 语言 处理 是 AI 的 分支"
]

tokenized = [" ".join(jieba.cut(doc)) for doc in corpus]

# Variant 1: Standard TF-IDF
v1 = TfidfVectorizer(token_pattern=r'\S+', sublinear_tf=False, norm='l2')
m1 = v1.fit_transform(tokenized)

# Variant 2: Sublinear TF
v2 = TfidfVectorizer(token_pattern=r'\S+', sublinear_tf=True, norm='l2')
m2 = v2.fit_transform(tokenized)

print("=== 标准TF vs 亚线性TF 对词'AI'的影响 ===")
words = v1.get_feature_names_out()
ai_idx = list(words).index('AI')
print(f"\n标准TF-IDF 'AI' 权重:    {m1.toarray()[:, ai_idx].round(3)}")
print(f"亚线性TF-IDF 'AI' 权重:  {m2.toarray()[:, ai_idx].round(3)}")
print(f"\n说明: Doc1 中'AI'出现5次，亚线性TF缓解了高频主导效应")
```

---

## 六、应用场景

### 6.1 文档检索

```
┌────────────────────────────────────────────────────────────┐
│                 TF-IDF 用于文档检索                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  查询: "自然语言处理 深度学习"                              │
│                                                            │
│  Step 1: 将查询转为 TF-IDF 向量                            │
│     q_vec = [0, 0.3, 0.5, 0, 0.4, ...]                    │
│                                                            │
│  Step 2: 计算查询与每篇文档的余弦相似度                    │
│     sim(q, D_i) = cos(q_vec, d_i_vec)                     │
│                                                            │
│  Step 3: 按相似度排序返回文档                              │
│     Doc3: 0.85  ←  最相关                                  │
│     Doc7: 0.62                                             │
│     Doc1: 0.41                                             │
│                                                            │
│  搜索引擎（早期）和向量数据库的基石                         │
└────────────────────────────────────────────────────────────┘
```

```python
from sklearn.metrics.pairwise import cosine_similarity

# Query similarity example
query = ["自然语言处理 深度学习"]
query_vec = vectorizer.transform([" ".join(jieba.cut(q)) for q in query])

similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

print("=== 文档检索结果 ===")
print(f"查询: {query[0]}\n")
ranked_indices = similarities.argsort()[::-1]
for rank, idx in enumerate(ranked_indices):
    print(f"  排名{rank+1} (相似度={similarities[idx]:.4f}): {corpus[idx]}")
```

### 6.2 关键词提取

```python
def extract_keywords_tfidf(text, vectorizer, top_k=5):
    """Extract top-K keywords from text using TF-IDF."""
    # Transform text to TF-IDF vector
    tokenized = [" ".join(jieba.cut(text))]
    vec = vectorizer.transform(tokenized)

    # Get feature names and scores
    feature_names = vectorizer.get_feature_names_out()
    scores = vec.toarray().flatten()

    # Sort by score descending
    top_indices = scores.argsort()[::-1][:top_k]
    keywords = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]

    return keywords


# Extract keywords
sample_text = "深度学习极大地推动了自然语言处理领域的发展"
keywords = extract_keywords_tfidf(sample_text, vectorizer, top_k=5)

print(f"文本: {sample_text}")
print("关键词提取结果:")
for word, score in keywords:
    print(f"  {word}: {score:.4f}")
```

### 6.3 文本分类

```python
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Build a TF-IDF + SVM text classifier
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(
        tokenizer=chinese_tokenizer,
        token_pattern=None,
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=2,
        sublinear_tf=True
    )),
    ('clf', LinearSVC(C=1.0))
])

# Toy data
texts = [
    "这部电影太精彩了 强烈推荐",
    "剧情无聊 浪费时间",
    "非常好看 值得二刷",
    "什么烂片 失望透顶",
    "经典之作 百看不厌",
    "难看 故事老套"
] * 10
labels = [1, 0, 1, 0, 1, 0] * 10

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

pipe.fit(X_train, y_train)
print("=== TF-IDF + SVM 分类报告 ===")
print(classification_report(y_test, pipe.predict(X_test)))
```

---

## 七、词袋 vs TF-IDF 对比

```
┌────────────────────────────────────────────────────────────┐
│           CountVectorizer  vs  TfidfVectorizer             │
├────────────────────────┬───────────────────────────────────┤
│     词袋模型 (Count)   │       TF-IDF                      │
├────────────────────────┼───────────────────────────────────┤
│ 值 = 原始词频计数       │ 值 = TF×IDF（加权后）             │
│ 高频词主导             │ 常见词被降低权重                   │
│ 不考虑稀有度           │ 稀有词权重提升                    │
│ 对长短文档敏感          │ 归一化后更公平                    │
│ 可解释性更强           │ 语义区分度更好                    │
│ 信息检索中效果一般     │ 信息检索的经典方法                │
│ 适合: 朴素贝叶斯       │ 适合: SVM、逻辑回归等             │
└────────────────────────┴───────────────────────────────────┘
```

| 对比维度 | 词袋模型（Count） | TF-IDF |
|----------|-------------------|--------|
| 权重计算 | 原始词频 | TF × IDF |
| 常见词影响 | 主导向量 | 权重降低 |
| 稀有词影响 | 被高频词淹没 | 权重提升 |
| 归一化 | 通常不做 | 默认 L2 归一化 |
| 区分能力 | 弱 | 强 |
| 信息检索 | 一般 | 优秀 |
| 文本分类 | 配合朴素贝叶斯好 | 配合 SVM/逻辑回归好 |
| 解释性 | 直观 | 需理解 IDF |

---

## 小结

| 要点 | 内容 |
|------|------|
| 核心思想 | 词的重要性 = TF（局部频率）× IDF（全局稀有度） |
| TF 公式 | 词在文档中的出现次数 / 文档总词数 |
| IDF 公式 | log(N / DF)，DF 越大 IDF 越小 |
| 关键变体 | 对数 TF（sublinear）、归一化 IDF、L2 归一化 |
| 工具 | sklearn `TfidfVectorizer` |
| 主要优点 | 比纯词袋更能突出关键词、无需训练数据 |
| 主要缺点 | 仍然稀疏、无语义、无词序 |
| 应用场景 | 文档检索、关键词提取、文本分类特征 |
| 与词袋关系 | TF-IDF 是词袋的改进版（加权方案） |
| 后续发展 | 为词嵌入（Word2Vec）等语义方法奠定基础 |

---

| [← 回到目录](../README.md) | [上一篇：词袋模型](01-词袋模型.md) | [下一篇：Word2Vec词嵌入](03-Word2Vec词嵌入.md) |
|---|---|---|
