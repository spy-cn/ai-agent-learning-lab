# GloVe 与 FastText

Word2Vec 开创了神经词嵌入的时代，但它仍有不足：**仅利用局部上下文窗口、无法处理未登录词**。GloVe（Pennington et al., 2014）通过**全局共现矩阵**改进了 Word2Vec 的局部性；FastText（Bojanowski et al., 2017）通过**子词嵌入**解决了未登录词问题。三者共同构成了经典静态词嵌入的三驾马车。

---

## 一、GloVe：全局向量表示

### 1.1 核心思想

Word2Vec 只利用了**局部上下文窗口**（滑动窗口内的词），而 GloVe（Global Vectors for Word Representation）利用了**全局词共现统计信息**。

```
┌────────────────────────────────────────────────────────────────┐
│           Word2Vec vs GloVe：信息利用方式                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Word2Vec（局部）:                                             │
│                                                                │
│    窗口逐句滑动:  [the cat] sits [on the]                      │
│                   ^^^^^         ^^^^^^                         │
│                   局部上下文                                    │
│                                                                │
│    → 只看到窗口内的词关系                                      │
│    → 遍历整个语料库间接学习全局信息                            │
│                                                                │
│  GloVe（全局）:                                                │
│                                                                │
│    先统计全局共现矩阵:                                         │
│           ice   steam   water   ...                            │
│    ice   [ 0      2     1000   ... ]                           │
│    steam [ 2      0     1200   ... ]                           │
│    water [1000   1200    0     ... ]                           │
│                                                                │
│    直接利用全局统计:                                           │
│    ratio = P(ice|water) / P(steam|water)                       │
│    这个比值比单看频率更有区分力                                │
│                                                                │
│    → 直接基于全局共现矩阵训练                                  │
│    → 统计信息利用更充分                                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 共现矩阵构建

**共现矩阵**（Co-occurrence Matrix）记录了每对词在窗口内共同出现的次数。

```python
import numpy as np
from collections import defaultdict

def build_cooccurrence_matrix(corpus, window_size=2):
    """
    Build word-word co-occurrence matrix.
    
    Args:
        corpus: list of tokenized sentences
        window_size: context window size
    
    Returns:
        cooccurrence dict: {(word_i, word_j): count}
        vocabulary: {word: index}
    """
    # Build vocabulary
    vocab = set()
    for sentence in corpus:
        vocab.update(sentence)
    vocab = sorted(vocab)
    word2idx = {w: i for i, w in enumerate(vocab)}
    
    # Count co-occurrences with distance weighting
    cooc_counts = defaultdict(float)
    
    for sentence in corpus:
        for i, center_word in enumerate(sentence):
            center_idx = word2idx[center_word]
            # Look at context window
            for j in range(max(0, i - window_size), 
                           min(len(sentence), i + window_size + 1)):
                if i != j:
                    context_word = sentence[j]
                    context_idx = word2idx[context_word]
                    # Weight by inverse distance (GloVe style)
                    distance = abs(i - j)
                    cooc_counts[(center_idx, context_idx)] += 1.0 / distance
    
    # Convert to dense matrix (for small vocabularies)
    V = len(vocab)
    matrix = np.zeros((V, V))
    for (i, j), count in cooc_counts.items():
        matrix[i, j] = count
    
    return matrix, word2idx, vocab


# ---------- Build co-occurrence matrix ----------
corpus = [
    ["自然", "语言", "处理", "很", "有趣"],
    ["自然", "语言", "处理", "是", "AI", "分支"],
    ["深度", "学习", "改变", "自然", "语言", "处理"],
    ["深度", "学习", "和", "自然", "语言", "处理", "紧密", "相关"]
]

cooc_matrix, word2idx, vocab = build_cooccurrence_matrix(corpus, window_size=2)

print("=== 共现矩阵 ===")
print(f"词表: {vocab}")
print(f"矩阵形状: {cooc_matrix.shape}")
print("矩阵（整数化）:")
header = "        " + "  ".join(f"{w:>4}" for w in vocab)
print(header)
for i, word in enumerate(vocab):
    row = "  ".join(f"{int(cooc_matrix[i,j]):>4}" for j in range(len(vocab)))
    print(f"{word:>6}  {row}")
```

### 1.3 共现矩阵示例

```
         自然  语言  处理  很  有趣  是  AI  分支  深度  学习  改变
  自然 [  0    3    3    0   0    0   0    0    0    0    0  ]
  语言 [  3    0    3    1   0    1   0    0    0    0    0  ]
  处理 [  3    3    0    1   1    0   1    0    1    0    1  ]
    很 [  0    1    1    0   1    0   0    0    0    0    0  ]
  有趣 [  0    0    1    1   0    0   0    0    0    0    0  ]
    ...

矩阵中 X[i][j] = 词 i 和词 j 在窗口内共现的加权次数
```

### 1.4 GloVe 目标函数

GloVe 的核心思想：**词向量的点积应该能预测共现概率的比值**。

```
┌──────────────────────────────────────────────────────────────┐
│                    GloVe 目标函数推导                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  起点: 词向量的点积应与共现概率相关                          │
│                                                              │
│     w_i · w_j ≈ log(P(j|i))                                 │
│                                                              │
│  其中 P(j|i) = X_ij / X_i  (词j在词i上下文中的比例)         │
│                                                              │
│  关键洞察: 概率比值比单一概率更有区分力                      │
│                                                              │
│     P(k|i)     P(ice|water)                                 │
│     ─────── ≈ ──────────────                                │
│     P(k|j)     P(steam|water)                                │
│                                                              │
│  对于 "water" 的上下文:                                      │
│     ice 出现多 → 比值 > 1                                   │
│     steam 出现多 → 比值 < 1                                 │
│     无关词(如 fashion) → 比值 ≈ 1                          │
│                                                              │
│  推导: w_i · (w_k - w_j) ≈ log(P(k|i) / P(k|j))            │
│                                                              │
│  最终目标函数（加权最小二乘）:                               │
│                                                              │
│     J = Σ_{i,j} f(X_ij) · (w_i^T w_j + b_i + b_j - log X_ij)²│
│                                                              │
│  其中:                                                       │
│     w_i, w_j: 词向量（待学习参数）                           │
│     b_i, b_j: 偏置项                                        │
│     X_ij: 共现次数                                          │
│     f(X_ij): 加权函数（控制高频共现的影响）                 │
│                                                              │
│  加权函数 f(x) 的设计:                                       │
│     • x < x_max 时: f(x) = (x/x_max)^α                     │
│     • x ≥ x_max 时: f(x) = 1                                │
│     典型值: x_max=100, α=0.75                               │
│                                                              │
│  作用:                                                       │
│     ✓ 高频共现不过度主导                                    │
│     ✓ 低频共现（多为噪声）权重低                            │
│     ✓ 中频共现贡献最大                                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 1.5 GloVe vs Word2Vec 对比

| 维度 | Word2Vec | GloVe |
|------|----------|-------|
| 信息来源 | 局部上下文窗口 | 全局共现矩阵 |
| 学习方式 | 在线（逐窗口） | 离线（矩阵分解） |
| 目标函数 | 预测概率（似然） | 加权最小二乘 |
| 训练数据 | 直接用原始文本 | 需先统计共现矩阵 |
| 计算效率 | 高（流式处理） | 中（需存储矩阵） |
| 低频词 | Skip-gram较好 | 一般 |
| 语义类比 | 好 | 略优 |
| 适用场景 | 在线学习、增量更新 | 离线训练、高质量嵌入 |

---

## 二、FastText：子词嵌入

### 2.1 核心改进

FastText（Facebook AI Research）在 Word2Vec 的基础上做了一个简单但 powerful 的改进：**把每个词拆分为字符 n-gram**。

```
┌──────────────────────────────────────────────────────────────┐
│               FastText 子词拆分示意                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  原始词: "where"                                             │
│                                                              │
│  添加边界标记: "<where>"                                    │
│                                                              │
│  拆分为 3-gram:                                              │
│     <wh, whe, her, ere, re>                                 │
│                                                              │
│  词向量 = 所有子词向量的求和（或平均）:                     │
│                                                              │
│     vec("where") = vec("<wh")                               │
│                  + vec("whe")                                │
│                  + vec("her")                                │
│                  + vec("ere")                                │
│                  + vec("re>")                                │
│                  + vec("<where>")  ← 完整词本身              │
│                                                              │
│  中文示例: "自然语言"                                        │
│  (汉字级别) 3-gram:                                          │
│     <自, 自然, 然语, 语言, 言>                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 为什么子词嵌入有效？

```
┌──────────────────────────────────────────────────────────────┐
│              FastText 解决的核心问题                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  问题1: 未登录词（OOV）                                      │
│  ─────────────────────────                                   │
│  Word2Vec: "unhappiness" 不在词表 → 无法表示               │
│  FastText: "unhappiness" = un + happi + ness + ...          │
│            即使没见过整词，子词也能组合出向量 ✓             │
│                                                              │
│  问题2: 形态学关系                                           │
│  ─────────────────                                           │
│  Word2Vec: "eat" 和 "eaten" 没有任何关联                    │
│  FastText: "eat" 和 "eaten" 共享子词 "eat"                  │
│            → 它们的向量自然接近 ✓                           │
│                                                              │
│  问题3: 拼写错误                                             │
│  ─────────────────                                           │
│  Word2Vec: "teh"（拼写错误）无法表示                        │
│  FastText: "teh" 和 "the" 共享子词                          │
│            → 向量仍然接近 ✓                                 │
│                                                              │
│  子词嵌入在以下语言中优势特别明显:                          │
│     • 形态丰富的语言（土耳其语、芬兰语、阿拉伯语）          │
│     • 复合词多的语言（德语、荷兰语）                        │
│     • 词形变化多的语言（俄语、拉丁语）                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.3 FastText 训练代码

```python
from gensim.models import FastText
import jieba

# ---------- Prepare corpus ----------
raw_texts = [
    "自然语言处理是人工智能的重要分支",
    "深度学习改变了自然语言处理领域",
    "机器学习和深度学习是AI核心技术",
    "自然语言处理包括分词词性标注命名实体识别",
    "计算机视觉专注于图像和视频理解",
    "语音识别将语音转换为文本",
    "深度学习推动了语音识别的突破",
    "自然语言处理和计算机视觉是两大方向",
    "人工智能正在改变我们的生活方式",
    "机器学习算法包括监督学习和无监督学习"
]

sentences = [list(jieba.cut(text)) for text in raw_texts]

# ---------- Train FastText ----------
model_ft = FastText(
    sentences=sentences,
    vector_size=100,        # Vector dimension
    window=5,               # Context window
    min_count=1,            # Min word frequency
    workers=4,
    sg=1,                   # 1=Skip-gram, 0=CBOW
    min_n=3,                # Min char n-gram length
    max_n=6,                # Max char n-gram length
    epochs=50,
    seed=42
)

print("=== FastText 模型 ===")
print(f"词表大小: {len(model_ft.wv.key_to_index)}")
print(f"向量维度: {model_ft.wv.vector_size}")

# ---------- Key advantage: handle OOV words ----------
oov_word = "自然语言处理技术"  # This exact word may not be in vocabulary
print(f"\n测试未登录词: '{oov_word}'")
print(f"在词表中? {oov_word in model_ft.wv}")

# FastText can STILL generate a vector for OOV words!
if oov_word not in model_ft.wv:
    vec = model_ft.wv[oov_word]  # This works! Computed from subwords
    print(f"OOV词向量（前10维）: {vec[:10].round(4)}")
    print("FastText 通过子词组合成功生成了向量！")

# ---------- Compare with similar words ----------
print("\n与 '自然语言处理' 最相似的词:")
for word, score in model_ft.wv.most_similar("自然语言处理", topn=5):
    print(f"  {word}: {score:.4f}")
```

---

## 三、加载预训练模型

### 3.1 加载预训练 GloVe

```python
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# ---------- Load pre-trained GloVe ----------
# Download from: https://nlp.stanford.edu/projects/glove/
# Files: glove.6B.zip (50d, 100d, 200d, 300d)

# Step 1: Convert GloVe format to Word2Vec format
# glove_input_file = 'glove.6B.100d.txt'
# word2vec_output_file = 'glove.6B.100d.w2v.txt'
# glove2word2vec(glove_input_file, word2vec_output_file)

# Step 2: Load with gensim
# glove_model = KeyedVectors.load_word2vec_format(
#     word2vec_output_file,
#     binary=False
# )

# print(f"GloVe 词表大小: {len(glove_model.key_to_index)}")  # ~400K words
# print(f"向量维度: {glove_model.vector_size}")  # 100

# ---------- GloVe analogy examples ----------
# Classic analogy: king - man + woman = queen
# result = glove_model.most_similar(
#     positive=['woman', 'king'],
#     negative=['man']
# )
# print(f"king - man + woman ≈ {result[0]}")
# # Output: ('queen', 0.7699)

# Capital analogy: Paris - France + Italy = Rome
# result = glove_model.most_similar(
#     positive=['paris', 'italy'],
#     negative=['france']
# )
# print(f"paris - france + italy ≈ {result[0]}")
# # Output: ('rome', 0.7823)
```

### 3.2 加载预训练 FastText

```python
# ---------- Load pre-trained FastText ----------
# Download from: https://fasttext.cc/
# Files: crawl-300d-2M.vec (2M words, 300d, multilingual)
#        cc.zh.300.vec (Chinese, 300d)

# fasttext_model = KeyedVectors.load_word2vec_format(
#     'cc.zh.300.vec',
#     binary=False
# )

# print(f"FastText 词表大小: {len(fasttext_model.key_to_index)}")
# print(f"向量维度: {fasttext_model.vector_size}")

# ---------- FastText OOV capability with pre-trained ----------
# oov_word = "区块链人工智能"  # Combined neologism
# vec = fasttext_model[oov_word]  # Works even if not in vocabulary!
# print(f"OOV词 '{oov_word}' 的向量: {vec[:5]}")

# ---------- Chinese word similarity ----------
# print("与'自然语言'最相似的词:")
# for w, s in fasttext_model.most_similar("自然语言", topn=5):
#     print(f"  {w}: {s:.4f}")
```

### 3.3 统一加载接口

```python
def load_pretrained_embeddings(model_type='fasttext', lang='zh'):
    """
    Load pre-trained word embeddings.
    
    Args:
        model_type: 'word2vec', 'glove', or 'fasttext'
        lang: 'zh' (Chinese) or 'en' (English)
    
    Returns:
        KeyedVectors object
    """
    paths = {
        ('word2vec', 'en'): 'GoogleNews-vectors-negative300.bin',
        ('glove', 'en'): 'glove.6B.300d.w2v.txt',
        ('fasttext', 'zh'): 'cc.zh.300.vec',
        ('fasttext', 'en'): 'crawl-300d-2M.vec',
    }
    
    path = paths.get((model_type, lang))
    if path is None:
        raise ValueError(f"Unsupported combination: {model_type}/{lang}")
    
    is_binary = path.endswith('.bin')
    model = KeyedVectors.load_word2vec_format(path, binary=is_binary)
    
    print(f"已加载 {model_type} ({lang}) 预训练模型")
    print(f"  词表大小: {len(model.key_to_index):,}")
    print(f"  向量维度: {model.vector_size}")
    
    return model


# Uncomment to load (requires downloaded model files)
# model = load_pretrained_embeddings('fasttext', 'zh')
```

---

## 四、三种嵌入方法综合对比

```
┌────────────────────────────────────────────────────────────────────┐
│            Word2Vec vs GloVe vs FastText 对比                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Word2Vec                                                          │
│  ┌──────────────────────────────────────┐                          │
│  │ • 滑动窗口学习局部模式               │                          │
│  │ • Skip-gram / CBOW                   │                          │
│  │ • 整词作为最小单元                   │                          │
│  │ • 无法处理 OOV                       │                          │
│  │ • 训练速度快                         │                          │
│  │ • 适合: 通用场景、在线学习           │                          │
│  └──────────────────────────────────────┘                          │
│                                                                    │
│  GloVe                                                             │
│  ┌──────────────────────────────────────┐                          │
│  │ • 全局共现矩阵 + 矩阵分解            │                          │
│  │ • 利用统计信息更充分                 │                          │
│  │ • 整词作为最小单元                   │                          │
│  │ • 无法处理 OOV                       │                          │
│  │ • 训练需较大内存（存储矩阵）         │                          │
│  │ • 适合: 高质量嵌入、离线训练         │                          │
│  └──────────────────────────────────────┘                          │
│                                                                    │
│  FastText                                                          │
│  ┌──────────────────────────────────────┐                          │
│  │ • 基于 Word2Vec + 子词分解           │                          │
│  │ • 字符 n-gram 作为最小单元           │                          │
│  │ • 能处理 OOV ✓                       │                          │
│  │ • 能捕捉形态学关系                   │                          │
│  │ • 训练速度略慢（更多子词）           │                          │
│  │ • 适合: 形态丰富语言、OOV多的场景    │                          │
│  └──────────────────────────────────────┘                          │
│                                                                    │
└─────────────────────────────────────────────────────────��──────────┘
```

### 详细对比表

| 特性 | Word2Vec | GloVe | FastText |
|------|----------|-------|----------|
| **信息来源** | 局部上下文窗口 | 全局共现矩阵 | 局部上下文 + 子词 |
| **最小单元** | 整词 | 整词 | 字符 n-gram |
| **模型架构** | 神经网络（浅层） | 矩阵分解 | 神经网络（浅层） |
| **目标函数** | 条件概率（似然） | 加权最小二乘 | 条件概率（似然） |
| **训练方式** | 在线/流式 | 离线（需预计算） | 在线/流式 |
| **未登录词(OOV)** | 不支持 | 不支持 | **支持** ✓ |
| **形态学关系** | 不捕捉 | 不捕捉 | **捕捉** ✓ |
| **训练速度** | 快 | 中 | 略慢 |
| **推理速度** | 快 | 快 | 较慢（子词求和） |
| **内存占用** | 小 | 大（矩阵） | 中 |
| **类比推理** | 好 | 略优 | 略差 |
| **语义相似度** | 好 | 好 | 好 |
| **低频词效果** | Skip-gram较好 | 一般 | **最好** ✓ |
| **作者** | Google (Mikolov) | Stanford (Pennington) | Facebook (Bojanowski) |
| **发布年份** | 2013 | 2014 | 2016 |
| **中文预训练** | Tencent词向量 | 少 | **cc.zh.300** ✓ |

### 选择建议

```
┌──────────────────────────────────────────────┐
│              方法选择决策树                  │
├──────────────────────────────────────────────┤
│                                              │
│  需要处理未登录词?                           │
│     ├── 是 → FastText                        │
│     └── 否 ↓                                 │
│                                              │
│  追求最高嵌入质量?                           │
│     ├── 是 → GloVe (大数据预训练)            │
│     └── 否 ↓                                 │
│                                              │
│  需要在线/增量训练?                          │
│     ├── 是 → Word2Vec                        │
│     └── 否 → GloVe 或 FastText              │
│                                              │
│  处理形态丰富的语言?                         │
│     └── 是 → FastText                        │
│                                              │
│  通用场景快速上手?                           │
│     └── Word2Vec 或 GloVe (预训练)          │
│                                              │
└──────────────────────────────────────────────┘
```

---

## 五、性能评测与对比实验

```python
import time
from gensim.models import Word2Vec, FastText

# ---------- Prepare larger corpus ----------
large_corpus = [
    "自然语言处理是人工智能的重要分支",
    "深度学习改变了自然语言处理领域",
    "机器学习和深度学习是AI核心技术",
    "自然语言处理包括分词词性标注命名实体识别",
    "计算机视觉专注于图像和视频理解",
    "语音识别将语音转换为文本",
    "深度学习推动了语音识别的突破",
    "自然语言处理和计算机视觉是两大方向",
    "人工智能正在改变我们的生活方式",
    "机器学习算法包括监督学习和无监督学习",
    "神经网络是深度学习的基础",
    "词向量是自然语言处理的基础技术",
    "BERT是预训练语言模型的代表",
    "GPT系列模型展现了强大的生成能力",
    "Transformer架构是现代NLP的基石"
]
sentences = [list(jieba.cut(text)) for text in large_corpus]

# ---------- Compare training time ----------
print("=== 训练时间对比 ===\n")

# Word2Vec (Skip-gram)
start = time.time()
model_w2v = Word2Vec(sentences, vector_size=100, window=5, 
                     min_count=1, sg=1, epochs=50, seed=42)
t_w2v = time.time() - start
print(f"Word2Vec (Skip-gram): {t_w2v:.3f}秒")

# FastText
start = time.time()
model_ft = FastText(sentences, vector_size=100, window=5,
                    min_count=1, sg=1, min_n=3, max_n=6, 
                    epochs=50, seed=42)
t_ft = time.time() - start
print(f"FastText:            {t_ft:.3f}秒")

# ---------- Compare OOV handling ----------
print("\n=== 未登录词处理对比 ===")
oov_words = ["自然语言", "深度学习技术", "人工智能模型"]

for word in oov_words:
    in_w2v = word in model_w2v.wv
    in_ft = word in model_ft.wv
    print(f"\n'{word}':")
    print(f"  Word2Vec 在词表? {in_w2v}  {'✗ 无法表示' if not in_w2v else '✓'}")
    print(f"  FastText 在词表? {in_ft}  {'✓ 可用子词生成' if not in_ft else '✓'}")
    
    if not in_ft:
        vec = model_ft.wv[word]
        print(f"  FastText OOV向量前5维: {vec[:5].round(3)}")

# ---------- Compare similarity quality ----------
print("\n=== 相似度质量对比 ===")
test_word = "自然语言处理"
if test_word in model_w2v.wv:
    print(f"\n与 '{test_word}' 最相似的词:")
    
    print("  Word2Vec:")
    for w, s in model_w2v.wv.most_similar(test_word, topn=3):
        print(f"    {w}: {s:.4f}")
    
    print("  FastText:")
    for w, s in model_ft.wv.most_similar(test_word, topn=3):
        print(f"    {w}: {s:.4f}")
```

---

## 小结

| 要点 | 内容 |
|------|------|
| **GloVe 核心** | 基于全局共现矩阵的词向量，最小化加权最小二乘 |
| **GloVe 公式** | J = Σ f(X_ij) · (w_i·w_j + b_i + b_j - log X_ij)² |
| **GloVe 优势** | 充分利用全局统计信息，语义类比效果略优 |
| **FastText 核心** | 把词拆分为字符 n-gram，词向量 = 子词向量之和 |
| **FastText 优势** | 能处理未登录词(OOV)、捕捉形态学关系 |
| **共现矩阵** | 记录词对在窗口内共同出现的加权次数 |
| **加权函数** | 高频不过度主导，低频（噪声）权重低 |
| **OOV 处理** | FastText 独有优势，通过子词组合生成向量 |
| **预训练模型** | GloVe (Stanford)、FastText (Facebook cc.zh.300) |
| **选择建议** | OOV多→FastText，质量优先→GloVe，通用→Word2Vec |
| **共同局限** | 都是静态表示，无法处理一词多义 → 需 ELMo/BERT |

---

| [← 回到目录](../README.md) | [上一篇：Word2Vec词嵌入](03-Word2Vec词嵌入.md) | [下一篇：分布式表示进阶](05-分布式表示进阶.md) |
|---|---|---|
