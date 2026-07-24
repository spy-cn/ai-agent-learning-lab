# Word2Vec 词嵌入

词袋模型和 TF-IDF 存在一个根本缺陷：**它们把每个词当作孤立的符号，词与词之间没有任何语义关联**。"猫"和"狗"是完全不同的两个维度，"开心"和"快乐"也毫无联系。Word2Vec（Mikolov et al., 2013）革命性地提出：**用一个低维稠密向量来表示每个词，让语义相近的词在向量空间中距离相近**，开启了神经词嵌入的时代。

---

## 一、分布式假设

Word2Vec 的理论基础是**分布式假设**（Distributional Hypothesis）：

> *"You shall know a word by the company it keeps."* — J.R. Firth, 1957

**含义**：语义相似的词往往出现在相似的上下文中。

```
┌──────────────────────────────────────────────────────────────┐
│                 分布式假设的直觉                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  观察以下句子：                                              │
│    "我今天吃了一个  苹果  "                                   │
│    "我今天吃了一个  橘子  "                                   │
│    "我买了一个    苹果  "                                     │
│    "我买了一个    橘子  "                                     │
│                                                              │
│  "苹果" 和 "橘子" 出现在几乎完全相同的上下文中               │
│  → 它们的语义应该很接近                                      │
│  → 它们的词向量应该距离很近                                  │
│                                                              │
│  而相比之下：                                                │
│    "苹果" 和 "汽车" 的上下文完全不同                         │
│    → 它们的词向量距离应该很远                                │
│                                                              │
│  核心思想：词的语义 = 它的上下文分布                         │
└──────────────────────────────────────────────────────────────┘
```

### 离散表示 vs 分布式表示

```
    离散表示（One-Hot）            分布式表示（Word2Vec）

    "猫" → [1,0,0,0,...,0]        "猫" → [0.2, -0.4, 0.8, ...]
    "狗" → [0,1,0,0,...,0]        "狗" → [0.3, -0.3, 0.7, ...]
    "车" → [0,0,1,0,...,0]        "车" → [-0.5, 0.9, -0.2, ...]

    维度: V（词表大小，数万）       维度: d（通常 100-300）
    稀疏: 99.99%+ 是 0              稠密: 所有维度都有值
    语义: 任意两词正交（无关）      语义: 相似词向量接近
```

---

## 二、Word2Vec 的两种架构

Word2Vec 有两种模型架构：**CBOW** 和 **Skip-gram**。

```
┌─────────────────────────────────────────────────────────────────┐
│                    CBOW vs Skip-gram                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  句子: "The cat sits on the mat"                               │
│  窗口大小 = 2                                                  │
│                                                                 │
│  ══════════════════ CBOW ══════════════════                    │
│                                                                 │
│         上下文词           预测           上下文词               │
│       ┌──────────┐                     ┌──────────┐            │
│       │  "The"   │                     │  "sits"  │            │
│       └────┬─────┘                     └────┬─────┘            │
│            │    ┌───────────┐              │                   │
│            └──→ │ 求和/平均  │ ←───────────┘                   │
│                 │  → 预测   │                                │
│            ┌──→ │ 中心词    │ ←───────────┐                   │
│            │    └─────┬─────┘             │                   │
│       ┌────┴─────┐   │             ┌─────┴────┐               │
│       │  "cat"   │   │             │   "on"   │               │
│       └──────────┘   ▼             └──────────┘               │
│                  目标词                                         │
│                  "sits"                                        │
│                                                                 │
│  规则: 用上下文词（窗口内）预测中心词                          │
│  速度: 快（一次求和平均）                                      │
│  适合: 高频词、小数据集                                         │
│                                                                 │
│  ══════════════════ Skip-gram ══════════════════               │
│                                                                 │
│                    ┌─────────┐                                 │
│                    │ "sits"  │  ← 中心词（输入）              │
│                    └────┬────┘                                 │
│                         │                                      │
│           ┌─────────────┼─────────────┐                       │
│           │             │             │                        │
│           ▼             ▼             ▼                        │
│       ┌──────┐     ┌──────┐     ┌──────┐                     │
│       │"The" │     │"cat" │     │ "on" │  ← 上下文（目标）    │
│       └──────┘     └──────┘     └──────┘                     │
│                                                                 │
│  规则: 用中心词预测窗口内的上下文词                            │
│  速度: 慢（每个上下文词单独训练）                              │
│  适合: 低频词、大数据集                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### CBOW vs Skip-gram 对比

| 特性 | CBOW | Skip-gram |
|------|------|-----------|
| 预测方向 | 上下文 → 中心词 | 中心词 → 上下文 |
| 训练速度 | 快 | 慢（约3-5倍） |
| 低频词效果 | 较差 | 较好 |
| 高频词效果 | 较好 | 一般 |
| 数据需求 | 小数据即可 | 需要大数据 |
| 内存占用 | 较小 | 较大 |
| 适用场景 | 通用、资源有限 | 追求质量、低频词重要 |

> **经验法则**：默认用 Skip-gram（gensim `sg=1`），除非训练数据很少或计算资源紧张。

---

## 三、训练优化：负采样与层次 Softmax

Word2Vec 原始版本中，输出层需要计算所有词的概率（Softmax），当词表很大时（10万+）计算极其昂贵。两种优化方案：

```
┌──────────────────────────────────────────────────────────────────┐
│              训练优化方法对比                                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ① 负采样（Negative Sampling）                                   │
│                                                                  │
│     原始 Softmax: 计算所��� V 个词的概率                          │
│     负采样:       只区分"目标词"和"少量噪声词"                   │
│                                                                  │
│     训练样本: (中心词, 真实上下文词)  → 正样本 y=1               │
│               (中心词, 随机抽取的词)  → 负样本 y=0               │
│                                                                  │
│     抽样数 k 通常为 5（大数据可减到2-3）                         │
│     计算量从 O(V) 降到 O(k)                                      │
│                                                                  │
│  ② 层次 Softmax（Hierarchical Softmax）                         │
│                                                                  │
│     将词表组织成一棵哈夫曼树:                                    │
│                                                                  │
│              [根]                                                │
│             /    \                                               │
│           ○      ○         每个词 = 从根到叶子的路径            │
│          / \    / \              预测 = 路径上每个节点的二分类   │
│         ○   ○  ○   ○                                            │
│        / \ / \/ \ / \                                           │
│       w1 w2 ...... wV                                           │
│                                                                  │
│     高频词路径短（接近根），低频词路径长                         │
│     计算量从 O(V) 降到 O(log V)                                  │
│                                                                  │
│  对比:                                                           │
│     负采样:  实现简单、通用性好、默认选择                        │
│     层次HS:  对低频词略好、实现复杂、较少使用                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 四、gensim Word2Vec 完整训练代码

```python
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import jieba
import numpy as np

# ---------- 1. Prepare training corpus ----------
# In practice, load a large corpus (Wikipedia, news articles, etc.)
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
    "机器学习算法包括监督学习和无监督学习",
    "神经网络是深度学习的基础",
    "Transformer架构 revolutionized自然语言处理",
    "BERT是预训练语言模型的代表",
    "GPT系列模型展现了强大的生成能力",
    "词向量是自然语言处理的基础技术"
]

# ---------- 2. Tokenize ----------
sentences = [list(jieba.cut(text)) for text in raw_texts]
print(f"训练句子数: {len(sentences)}")
print(f"示例分词: {sentences[0]}")

# ---------- 3. Train Word2Vec (Skip-gram) ----------
model = Word2Vec(
    sentences=sentences,
    vector_size=100,      # Dimension of word vectors
    window=5,             # Max distance between current and predicted word
    min_count=1,          # Ignore words with frequency < min_count
    workers=4,            # Number of threads
    sg=1,                 # 1 for Skip-gram, 0 for CBOW
    hs=0,                 # 0 for negative sampling, 1 for hierarchical softmax
    negative=5,           # Number of negative samples
    epochs=50,            # Number of training epochs
    seed=42
)

print(f"\n词表大小: {len(model.wv.key_to_index)}")
print(f"词向量维度: {model.wv.vector_size}")

# ---------- 4. Save and load ----------
model.save("word2vec_model.bin")
loaded_model = Word2Vec.load("word2vec_model.bin")

# ---------- 5. Get word vector ----------
word = "自然语言处理"
if word in model.wv:
    vec = model.wv[word]
    print(f"\n'{word}' 的词向量（前10维）: {vec[:10].round(4)}")
```

### 词向量训练：超参数调优版

```python
# For better quality with more data:
model_tuned = Word2Vec(
    sentences=sentences,
    vector_size=200,      # Larger dimensions for richer representation
    window=8,             # Larger window captures broader context
    min_count=2,          # Filter rare words
    workers=4,
    sg=1,                 # Skip-gram
    negative=10,          # More negative samples for better quality
    alpha=0.025,          # Initial learning rate
    min_alpha=0.0001,     # Final learning rate (decays)
    epochs=100,
    seed=42
)
```

---

## 五、词向量特性

### 5.1 语义相似度

```python
# Find most similar words
def print_similar(word, model, topn=5):
    print(f"\n与 '{word}' 最相似的词:")
    if word in model.wv:
        for w, score in model.wv.most_similar(word, topn=topn):
            print(f"  {w}: {score:.4f}")
    else:
        print(f"  '{word}' 不在词表中")

print_similar("自然语言处理", model)
print_similar("深度学习", model)
print_similar("人工智能", model)
```

### 5.2 类比推理（King - Man + Woman ≈ Queen）

Word2Vec 最令人惊叹的特性是**向量算术**能捕捉语义关系：

```
┌──────────────────────────────────────────────────────────────┐
│                   词向量类比推理                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  经典案例（英文）:                                           │
│                                                              │
│     king - man + woman ≈ queen                               │
│                                                              │
│     vec("king")                                          │
│       - vec("man")                                         │
│       + vec("woman")                                      │
│       ≈ vec("queen")                                       │
│                                                              │
│  向量空间中的几何解释:                                      │
│                                                              │
│      · king                         · queen                 │
│         \                            /                      │
│          \    "royalty"           /                         │
│           \   方向              /                            │
│            \                  /                             │
│      · man ───────────→ · woman                           │
│                "gender" 方向                               │
│                                                              │
│  类比: vec(king) - vec(man) 提取了 "royalty" 维度           │
│        加到 woman 上，得到 queen                            │
│                                                              │
│  其他经典案例:                                               │
│     Paris:France :: Tokyo:Japan                             │
│     walking:walked :: swimming:swam                         │
│     big:bigger :: great:greater                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

```python
# Analogy reasoning (works best with large pre-trained models)
def analogy(model, word_a, word_b, word_c, topn=3):
    """
    Solve: a is to b as c is to ?
    Computes: vec(b) - vec(a) + vec(c)
    """
    try:
        result = model.wv.most_similar(
            positive=[word_b, word_c],
            negative=[word_a],
            topn=topn
        )
        print(f"{word_a} : {word_b} :: {word_c} : ?")
        for word, score in result:
            print(f"  → {word} (score={score:.4f})")
    except KeyError as e:
        print(f"词不在词表中: {e}")

# Try analogies (need larger corpus for good results)
analogy(model, "机器学习", "深度学习", "自然语言处理")
```

### 5.3 不相似词检测

```python
# Find the odd one out
def find_odd_one(model, words):
    """Find the word that doesn't belong."""
    odd = model.wv.doesnt_match(words)
    print(f"词组: {words}")
    print(f"不属于该组的词: {odd}")
    return odd

find_odd_one(model, ["自然语言处理", "深度学习", "计算机视觉", "早餐"])
```

---

## 六、词向量可视化（t-SNE 降维）

将高维词向量降维到 2D 进行可视化。

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# ---------- Collect word vectors ----------
words = list(model.wv.key_to_index.keys())
word_vectors = np.array([model.wv[w] for w in words])

print(f"待可视化词数: {len(words)}")
print(f"原始向量维度: {word_vectors.shape}")

# ---------- t-SNE dimensionality reduction ----------
tsne = TSNE(
    n_components=2,
    perplexity=min(5, len(words) - 1),  # perplexity must be < n_samples
    random_state=42,
    n_iter=1000
)
vectors_2d = tsne.fit_transform(word_vectors)

# ---------- Plot ----------
plt.figure(figsize=(12, 8))
plt.rcParams['font.sans-serif'] = ['SimHei']  # Chinese font
plt.rcParams['axes.unicode_minus'] = False

scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='steelblue', alpha=0.7)

# Annotate each point with its word
for i, word in enumerate(words):
    plt.annotate(
        word,
        (vectors_2d[i, 0], vectors_2d[i, 1]),
        fontsize=9,
        alpha=0.8
    )

plt.title("Word2Vec 词向量 t-SNE 可视化", fontsize=16)
plt.xlabel("t-SNE 维度 1")
plt.ylabel("t-SNE 维度 2")
plt.tight_layout()
plt.savefig("word2vec_tsne.png", dpi=150, bbox_inches='tight')
plt.show()

print("可视化图已保存: word2vec_tsne.png")
```

```
┌─────────────────────────────────────────────────────┐
│           t-SNE 可视化示意（概念图）                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│     深度学习 ·        · 机器学习                     │
│        神经网络 ·         · 监督学习                │
│                                                     │
│              · 自然语言处理                         │
│        · 分词    · 词性标注                         │
│                                                     │
│    · 计算机视觉    · 图像理解                       │
│                                                     │
│  · 语音识别    · 语音转换                           │
│                                                     │
│  语义相关的词聚类在一起                            │
│  语义无关的词距离较远                               │
└─────────────────────────────────────────────────────┘
```

---

## 七、超参数选择指南

```
┌────────────────────────────────────────────────────────────────┐
│                  Word2Vec 超参数选择指南                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  vector_size（词向量维度）                                     │
│  ├── 50-100:  小数据集/简单任务                               │
│  ├── 200-300: 中等数据集（常用默认）                          │
│  └── 500+:    大数据集/复杂语义                               │
│  经验: 数据量越大，可用更高维度                                │
│        太低 → 欠拟合（丢失信息）                               │
│        太高 → 过拟合 + 计算成本                               │
│                                                                │
│  window（上下文窗口大小）                                      │
│  ├── 2-5:  注重句法关系（主谓、修饰）                         │
│  ├── 5-10: 注重语义关系（主题、话题）（推荐）                 │
│  └── 10+:  广泛主题                                            │
│  注意: Skip-gram 中 window 是最大值，实际随机采样 [1, window]  │
│                                                                │
│  min_count（最小词频阈值）                                     │
│  ├── 1-2:  小数据集（保留所有词）                             │
│  ├── 5-10: 中等数据集                                         │
│  └── 20+:  大数据集（过滤噪声）                               │
│  过滤低频词能提升整体质量                                      │
│                                                                │
│  sg（模型架构）                                                │
│  ├── sg=0: CBOW（快，适合高频词）                              │
│  └── sg=1: Skip-gram（慢，低频词效果好，质量更高）            │
│                                                                │
│  negative（负采样数）                                          │
│  ├── 5-20:  默认范围（5是常用值）                             │
│  └── 小数据集用大值，大数据集用小值                           │
│                                                                │
│  epochs（训练轮数）                                            │
│  ├── 5-10:   大数据集（百万级句子）                           │
│  ├── 10-50:  中等数据集                                       │
│  └── 50-100: 小数据集                                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 超参数推荐组合

| 场景 | vector_size | window | min_count | sg | epochs |
|------|-------------|--------|-----------|-----|--------|
| 小数据（<1万句） | 100 | 5 | 1 | 1 | 50-100 |
| 中等数据（10万句） | 200 | 5-8 | 3-5 | 1 | 10-20 |
| 大数据（百万句） | 300 | 5-10 | 5-10 | 1 | 5-10 |
| 资源受限 | 100 | 3-5 | 5 | 0 (CBOW) | 5-10 |
| 最高质量 | 300 | 8-10 | 2-5 | 1 | 15-30 |

---

## 八、加载预训练词向量

训练自己的词向量需要大量数据，实际中常使用预训练模型。

```python
from gensim.models import KeyedVectors

# ---------- Option 1: Load pre-trained Word2Vec (Google News) ----------
# Download: https://code.google.com/archive/p/word2vec/
# File: GoogleNews-vectors-negative300.bin (about 3.5 GB)

# google_model = KeyedVectors.load_word2vec_format(
#     'GoogleNews-vectors-negative300.bin',
#     binary=True
# )
# print(f"词表大小: {len(google_model.key_to_index)}")  # ~3 billion words
# print(f"维度: {google_model.vector_size}")  # 300

# ---------- Option 2: Load Chinese pre-trained vectors ----------
# Common sources: Tencent AI Lab, Facebook fastText, etc.

# tencent_model = KeyedVectors.load_word2vec_format(
#     'Tencent_AILab_ChineseEmbedding.txt',
#     binary=False
# )

# ---------- Using pre-trained model for analogy ----------
# Classic example with Google News model:
# result = google_model.most_similar(
#     positive=['woman', 'king'],
#     negative=['man']
# )
# print(result[0])  # ('queen', 0.7118)

# ---------- Fine-tuning: continue training on domain data ----------
# model = Word2Vec.load("word2vec_model.bin")
# more_sentences = [list(jieba.cut(t)) for t in domain_texts]
# model.build_vocab(more_sentences, update=True)  # Add new words
# model.train(more_sentences, total_examples=len(more_sentences),
#             epochs=model.epochs)  # Continue training
```

---

## 小结

| 要点 | 内容 |
|------|------|
| 核心思想 | 基于分布式假设，用低维稠密向量表示词的语义 |
| 理论基础 | "You shall know a word by the company it keeps" |
| 两种架构 | CBOW（上下文→中心词）、Skip-gram（中心词→上下文） |
| 训练优化 | 负采样（常用）、层次 Softmax |
| 关键特性 | 语义相似度、类比推理、聚类分析 |
| 推荐架构 | Skip-gram（质量更好，尤其低频词） |
| 超参数 | vector_size(100-300)、window(5-10)、min_count(2-5) |
| 主要优点 | 低维稠密、语义丰富、可迁移 |
| 主要缺点 | 静态表示（无法处理一词多义）、需要大数据训练 |
| 工具 | gensim `Word2Vec`、预训练 Tencent/Google 模型 |
| 后续演进 | GloVe（全局统计）、FastText（子词）、ELMo/BERT（动态） |

---

| [← 回到目录](../README.md) | [上一篇：TF-IDF](02-TF-IDF.md) | [下一篇：GloVe与FastText](04-GloVe与FastText.md) |
|---|---|---|
