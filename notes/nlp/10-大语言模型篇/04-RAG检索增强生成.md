# RAG检索增强生成

大语言模型的知识存储在其参数中，而参数是在训练截止日之前冻结的。当用户提问超出模型知识范围、涉及最新信息或特定领域专业内容时，LLM容易产生幻觉（Hallucination）——自信地编造看似合理但完全错误的内容。RAG（Retrieval-Augmented Generation，检索增强生成）将一个外部知识库接入LLM，让模型先"查阅资料"再回答问题，从根本上缓解了知识截止和幻觉问题。

## RAG的动机：LLM的三大局限

```
┌──────────────────────────────────────────────────────────────────┐
│                  LLM 的三大原生局限                                │
├────────────────────┬─────────────────────────────────────────────┤
│      局限           │                    表现                      │
├────────────────────┼─────────────────────────────────────────────┤
│  ① 知识截止日期      │ 模型训练数据有截止日期，无法获取最新信息        │
│  (Knowledge Cutoff) │ 例: 问2024年后的事件→无法回答或编造           │
├────────────────────┼─────────────────────────────────────────────┤
│  ② 幻觉             │ 模型会生成看似合理但完全虚构的内容               │
│  (Hallucination)    │ 例: 编造不存在的论文、错误的统计数据            │
├────────────────────┼─────────────────────────────────────────────┤
│  ③ 缺乏领域知识      │ 通用模型对垂直领域(医疗/法律/金融)知识不足       │
│  (Domain Gap)       │ 例: 无法准确回答特定企业的内部流程              │
└────────────────────┴─────────────────────────────────────────────┘
```

RAG的解决方案：**让LLM回答问题时"开卷考试"**——先从外部知识库检索相关文档，再基于检索到的内容生成回答。本质上是将检索系统的"精确性"与生成模型的"流畅性"结合。

## RAG架构核心

RAG系统由两个核心组件构成：

```
┌──────────────────────────────────────────────────────────────────┐
│                    RAG 完整工作流程                                │
│                                                                   │
│  用户查询                                                          │
│  ┌──────────┐                                                      │
│  │  "2024年  │                                                    │
│  │  Nobel物理 │                                                    │
│  │  奖得主?"  │                                                    │
│  └─────┬────┘                                                      │
│        │                                                           │
│        ▼                                                           │
│  ┌──────────────────────────┐                                      │
│  │  Embedding Model          │  ← 将查询向量化                      │
│  │  (text-embedding-3-large) │                                     │
│  └─────┬────────────────────┘                                      │
│        │                                                           │
│        │ query vector (1536-dim)                                    │
│        ▼                                                           │
│  ┌──────────────────────────────────────────┐                      │
│  │          Vector Database                   │                     │
│  │  ┌──────────────────────────────────┐     │                     │
│  │  │ 文档块向量索引                      │     │                     │
│  │  │ doc_1: [0.12, -0.34, ...]        │     │                     │
│  │  │ doc_2: [0.45, 0.21, ...]         │  ← 相似度检索Top-K         │
│  │  │ doc_3: [-0.15, 0.87, ...]        │     │                     │
│  │  │ ...                              │     │                     │
│  │  └──────────────────────────────────┘     │                     │
│  └─────┬────────────────────────────────────┘                      │
│        │                                                           │
│        │ 返回Top-K相关文档片段                                       │
│        ▼                                                           │
│  ┌──────────────────────────────────────┐                          │
│  │         上下文拼接                     │                          │
│  │  ┌──────────────────────────────┐    │                          │
│  │  │ 基于以下参考资料回答问题:      │    │                          │
│  │  │                              │    │                          │
│  │  │ [文档1] 2024年诺贝尔物理学奖   │    │                          │
│  │  │ 授予John Hopfield和Geoffrey   │    │                          │
│  │  │ Hinton...                    │    │                          │
│  │  │                              │    │                          │
│  │  │ [文档2] 瑞典皇家科学院宣布...  │    │                          │
│  │  │                              │    │                          │
│  │  │ 问题: 2024年Nobel物理奖得主?  │    │                          │
│  │  └──────────────────────────────┘    │                          │
│  └─────┬────────────────────────────────┘                          │
│        │                                                           │
│        ▼                                                           │
│  ┌──────────────────────────┐                                      │
│  │    LLM 生成器             │                                      │
│  │    (GPT-4 / Llama等)      │                                      │
│  └─────┬────────────────────┘                                      │
│        │                                                           │
│        ▼                                                           │
│  ┌──────────────────────────────────────────┐                      │
│  │  "2024年诺贝尔物理学奖授予了John          │                      │
│  │   Hopfield和Geoffrey Hinton，以表彰       │                      │
│  │   他们在人工神经网络方面的基础性           │                      │
│  │   发现和发明..."                           │                      │
│  └──────────────────────────────────────────┘                      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### RAG与传统方法的对比

```
┌───────────────┬───────────────┬───────────────┬───────────────┐
│      特征       │   传统搜索     │   纯LLM生成    │     RAG       │
├───────────────┼───────────────┼───────────────┼───────────────┤
│ 知识来源        │ 检索结果列表   │ 模型参数       │ 外部知识库     │
│ 准确度          │ 高(但需人工筛选)│ 不稳定         │ 高            │
│ 可溯源           │ ✓            │ ✗             │ ✓             │
│ 更新成本         │ 低            │ 需重新训练     │ 低(更新文档库) │
│ 流畅表达         │ ✗             │ ✓             │ ✓             │
│ 私有数据支持     │ ✓             │ ✗             │ ✓             │
└───────────────┴───────────────┴───────────────┴───────────────┘
```

## 向量数据库选型

向量数据库是RAG系统的存储和检索核心。以下是主流选型对比：

| 向量数据库 | 特点 | 适用场景 | 部署方式 |
|-----------|------|---------|---------|
| **FAISS** (Meta) | 纯C++/Python库，极轻量，支持GPU加速 | 小规模原型、嵌入式计算环境 | 本地库 |
| **Chroma** | 简单易用API，Python优先 | 快速原型、小型应用 | 本地/嵌入式 |
| **Milvus** | 分布式架构，支持十亿级向量 | 大规模生产环境 | 独立服务 |
| **Pinecone** | 全托管云服务，零运维 | 不想管理基础设施的团队 | SaaS |
| **Qdrant** | Rust编写，高性能，支持过滤 | 性能敏感场景 | 独立服务/云 |
| **Weaviate** | 内置向量化模块，GraphQL接口 | 需要多模态检索的场景 | 独立服务/云 |

**选型建议**：原型阶段用Chroma，本地部署用FAISS+GPU，生产环境用Milvus或Qdrant，不想运维用Pinecone。

## Embedding模型选型

嵌入模型将文本转换为向量表示，直接影响检索质量。

| 模型 | 维度 | 语言 | 特点 |
|------|------|------|------|
| text-embedding-3-large (OpenAI) | 3072 | 多语言 | 性能最强，需付费API |
| text-embedding-3-small (OpenAI) | 1536 | 多语言 | 性价比高 |
| bge-large-zh-v1.5 (BAAI) | 1024 | 中文 | 中文Embedding标杆，开源 |
| bge-m3 (BAAI) | 1024 | 多语言 | 支持稠密+稀疏混合检索 |
| m3e-base (Moka) | 768 | 中文 | 轻量级中文嵌入，适合本地部署 |
| stella-base-zh-v3-1792d | 1792 | 中文 | 高维中文嵌入，检索精度高 |
| jina-embeddings-v3 | 1024 | 多语言 | 支持长文本(8192 tokens) |

```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# ============================================================
# Embedding模型加载与向量化演示
# ============================================================

def load_embedding_model(model_name="BAAI/bge-large-zh-v1.5"):
    """加载中文Embedding模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def encode_text(texts, tokenizer, model, max_length=512):
    """
    将文本列表转换为向量表示
    
    参数:
        texts: 文本字符串列表
    返回:
        embeddings: numpy数组, shape (len(texts), hidden_dim)
    """
    # BGE系列模型需要使用特殊指令前缀
    if "bge" in model.config._name_or_path.lower():
        texts = [f"为这个句子生成表示以用于检索相关文章：{t}" for t in texts]
    
    inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        # 取[CLS] token的输出或均值池化
        attention_mask = inputs["attention_mask"]
        hidden_states = outputs.last_hidden_state
        
        # 均值池化（考虑attention mask）
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
    return embeddings.numpy()

# 演示文本向量化
print("=" * 60)
print("Embedding向量化演示")
print("=" * 60)

# 使用较小的模型演示（实际生产建议用bge-large-zh-v1.5）
sample_texts = [
    "机器学习是人工智能的一个分支",
    "深度学习使用多层神经网络进行特征提取",
    "今天天气真好，适合出去散步",
]

print(f"输入文本数量: {len(sample_texts)}")
print(f"向量维度(以bge-large-zh为例): 1024")
print()

# 演示向量相似度计算
emb1 = np.random.randn(1024).astype(np.float32)
emb2 = np.random.randn(1024).astype(np.float32)
emb3 = np.random.randn(1024).astype(np.float32)

# 使emb1和emb2更相似（模拟相关文本）
emb2 = emb1 * 0.8 + emb2 * 0.2

# 余弦相似度
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"相似度(相关文本):  {cosine_sim(emb1, emb2):.4f}")
print(f"相似度(无关文本):  {cosine_sim(emb1, emb3):.4f}")
print("(相关文本的向量相似度显著高于无关文本)")
```

## 文档分块策略（Chunking）

文档分块直接影响检索质量——块太大则检索精度下降，块太小则上下文碎片化。

```
┌──────────────────┬──────────────────┬──────────────────┬──────────┐
│     策略          │     原理          │      优点          │   缺点     │
├──────────────────┼──────────────────┼──────────────────┼──────────┤
│ 固定大小分块      │ 按固定token/字符数 │ 实现简单          │ 可能截断   │
│ (Fixed-size)     │ 切分, 可加overlap  │ 通用性好          │ 语义单元   │
├──────────────────┼──────────────────┼──────────────────┼──────────┤
│ 递归分块          │ 按分隔符层级       │ 保持段落完整性     │ 块大小     │
│ (Recursive)      │ (\n\n→\n→。→.)    │ 对自然文本效果好   │ 不均匀     │
│                  │ 递归切分           │                   │           │
├──────────────────┼──────────────────┼──────────────────┼──────────┤
│ 语义分块          │ 用模型判断语义     │ 块内语义连贯      │ 计算成本   │
│ (Semantic)       │ 边界(sentence-    │ 检索精度最高      │ 较高       │
│                  │ transformer)      │                   │           │
├──────────────────┼──────────────────┼──────────────────┼──────────┤
│ Small-to-Big     │ 检索时用小块，     │ 兼顾检索精度和     │ 实现复杂   │
│                  │ 生成时用大块上下文  │ 生成质量          │           │
│                  │ (父文档+子文档)    │                   │           │
└──────────────────┴──────────────────┴──────────────────┴──────────┘
```

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# 演示文本（模拟技术文档）
demo_text = """# RAG系统架构

RAG由检索器和生成器两部分组成。

## 检索器

检索器负责从知识库中查找与用户查询最相关的文档。常见的检索方式包括：
- BM25稀疏检索：基于词频-逆文档频率的经典信息检索方法
- 稠密向量检索：使用Embedding模型将文本映射到向量空间，基于余弦相似度检索
- 混合检索：融合BM25和稠密检索的结果，取长补短

## 生成器

生成器基于检索到的文档上下文生成回答。通常使用大语言模型作为生成器,
将检索结果与用户查询拼接后送入模型。
"""

# 递归分块器（推荐默认方案）
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,        # 每块最多200字符
    chunk_overlap=40,      # 块间重叠40字符
    separators=["\n\n", "\n", "。", ".", " ", ""],  # 分隔符优先级
    length_function=len,
)

chunks = recursive_splitter.split_text(demo_text)
print("=" * 60)
print("文档分块演示 (RecursiveCharacterTextSplitter)")
print("=" * 60)
for i, chunk in enumerate(chunks):
    print(f"\n[块 {i+1}] (长度: {len(chunk)}字符):")
    print(f"  {chunk[:100]}...")
    if len(chunk) > 100:
        print(f"  ...{chunk[-50:]}")

print(f"\n总块数: {len(chunks)}")
```

## 检索策略

### 稀疏检索：BM25

BM25是一类经典的词频-逆文档频率检索算法，基于关键词精确匹配：

```python
import math
import numpy as np
from collections import Counter

class BM25:
    """从零实现BM25检索算法"""
    
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        参数:
            corpus: 文档列表，每个元素是一段文本
            k1: 词频饱和参数，控制词频对得分的影响
            b:  文档长度归一化参数
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)  # 文档总数
        
        # 分词和统计
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        
        # 计算平均文档长度
        self.avgdl = np.mean([len(doc) for doc in self.tokenized_corpus])
        
        # 计算IDF
        self.idf = self._compute_idf()
        
    def _tokenize(self, text):
        """简单分词（实际应用应使用jieba等分词器）"""
        return text.lower().split()
    
    def _compute_idf(self):
        """计算每个词的IDF"""
        df = Counter()  # 文档频率
        for doc in self.tokenized_corpus:
            df.update(set(doc))
        
        idf = {}
        for term, freq in df.items():
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
        return idf
    
    def score(self, query, doc_idx):
        """计算查询与文档的BM25得分"""
        query_tokens = self._tokenize(query)
        doc_tokens = self.tokenized_corpus[doc_idx]
        doc_len = len(doc_tokens)
        doc_counter = Counter(doc_tokens)
        
        score = 0.0
        for token in query_tokens:
            if token not in self.idf:
                continue
            
            tf = doc_counter.get(token, 0)
            if tf == 0:
                continue
            
            idf = self.idf[token]
            
            # BM25核心公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
            
        return score
    
    def search(self, query, top_k=3):
        """检索Top-K文档"""
        scores = [self.score(query, i) for i in range(self.N)]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(i, scores[i]) for i in top_indices if scores[i] > 0]

# BM25演示
corpus = [
    "深度学习使用多层神经网络进行特征提取和模式识别",
    "机器学习是人工智能的核心分支包含监督学习和无监督学习",
    "自然语言处理研究计算机与人类语言的交互",
    "今天天气很好适合户外活动散步和野餐",
    "深度神经网络在图像识别和自然语言处理中取得了突破性进展",
]

query = "深度学习在自然语言处理中的应用"
bm25 = BM25(corpus)
results = bm25.search(query, top_k=3)

print("\n" + "=" * 60)
print("BM25 检索演示")
print("=" * 60)
print(f"查询: {query}\n")
for i, (idx, score) in enumerate(results):
    print(f"  Top-{i+1} 得分 {score:.4f}: {corpus[idx]}")
```

### 稠密检索（Dense Retrieval）

基于Embedding向量的语义相似度检索：

```python
class DenseRetriever:
    """稠密向量检索器"""
    
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.documents = []
        self.embeddings = None  # 形状 (N, dim)
        
    def add_documents(self, documents, embeddings_function):
        """添加文档并计算向量"""
        self.documents = documents
        # 实际应用中使用真实的Embedding模型
        self.embeddings = embeddings_function(documents)
        
    def search(self, query_embedding, top_k=3):
        """基于余弦相似度检索Top-K文档"""
        # 归一化后点积 = 余弦相似度
        doc_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        scores = np.dot(doc_norm, query_norm)  # (N,)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(i, scores[i]) for i in top_indices]
```

### 混合检索（Hybrid Retrieval）与RRF融合

将BM25和稠密检索结果融合的最常用方法是**RRF（Reciprocal Rank Fusion）**：

```
RRF_score(d) = Σ_{r ∈ R} 1 / (k + rank_r(d))

其中:
- R: 多个检索器的排名列表集合
- k: 平滑参数(通常设为60)
- rank_r(d): 文档d在检索器r中的排名
```

```python
def reciprocal_rank_fusion(rankings_list, k=60):
    """
    RRF融合多个检索器的排名结果
    
    参数:
        rankings_list: 排名列表的列表
            每个ranking是文档索引列表，按相关性降序
        k: RRF平滑参数
    
    返回:
        fused_scores: 字典 {doc_idx: RRF_score}
    """
    fused_scores = {}
    
    for ranking in rankings_list:
        for rank, doc_idx in enumerate(ranking):
            rrf_score = 1.0 / (k + rank + 1)  # rank从1开始
            fused_scores[doc_idx] = fused_scores.get(doc_idx, 0) + rrf_score
    
    # 按得分降序排列
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs

# 混合检索演示
print("\n" + "=" * 60)
print("混合检索 + RRF融合演示")
print("=" * 60)

# 模拟BM25和Dense检索排名
bm25_ranking = [3, 1, 0, 4, 2]    # BM25认为doc_3最相关
dense_ranking = [0, 4, 1, 3, 2]   # Dense认为doc_0最相关

fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=60)
print(f"BM25排名:     {bm25_ranking}")
print(f"Dense排名:    {dense_ranking}")
print(f"RRF融合结果:  {[(f'Doc_{d}', f'{s:.4f}') for d, s in fused[:5]]}")
```

## 重排序（Reranker）

初次检索返回Top-K候选项后，使用**Cross-encoder**模型进行精细重排序可以显著提升结果质量：

```
┌─────────────────────────────────────────────────┐
│              检索流程 (含Reranker)               │
│                                                  │
│  Embedding检索 → 召回Top-K候选 → Reranker精排     │
│   (高召回)        (K=20~100)       (返回Top-N)    │
│                                                  │
│  Bi-encoder:                    Cross-encoder:   │
│  查询和文档独立编码                查询和文档联合编码  │
│  速度快，可预先索引                精度高但速度慢     │
│  适用: 海量文档初筛              适用: 少量候选精排  │
└─────────────────────────────────────────────────┘
```

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class CrossEncoderReranker:
    """Cross-encoder重排序器"""
    
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
    def compute_scores(self, query, documents):
        """
        为每对(query, document)计算相关性分数
        
        参数:
            query: 查询字符串
            documents: 候选文档列表
        返回:
            相关性分数列表
        """
        pairs = [[query, doc] for doc in documents]
        
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            )
            scores = self.model(**inputs).logits.squeeze(-1)
            
        return scores.numpy()
    
    def rerank(self, query, documents, top_n=3):
        """重排序并返回Top-N文档"""
        scores = self.compute_scores(query, documents)
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(i, scores[i]) for i in top_indices]

# 演示Reranker概念
print("\n" + "=" * 60)
print("Reranker 重排序演示")
print("=" * 60)
candidates = [
    "神经网络在图像分类中表现优异",
    "卷积神经网络是CNN的全称",
    "图像分割是计算机视觉的重要任务",
    "循环神经网络适合处理序列数据",
]
query = "CNN在图像识别中的应用"

# 模拟初始检索得分和重排序得分
initial_scores = [0.85, 0.78, 0.76, 0.45]
rerank_scores = [0.92, 0.81, 0.65, 0.30]

initial_rank = np.argsort(initial_scores)[::-1]
rerank_rank = np.argsort(rerank_scores)[::-1]

print(f"查询: {query}\n")
print("初始检索排名 → 重排序后排名:")
for i in range(len(candidates)):
    print(f"  初始#{np.where(initial_rank==i)[0][0]+1} → "
          f"重排#{np.where(rerank_rank==i)[0][0]+1}: {candidates[i][:50]}...")
```

## LangChain实现完整RAG

LangChain提供了构建RAG系统的一站式工具链。以下是完整实现：

```python
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline

# ============================================================
# LangChain RAG 完整流程
# ============================================================

def build_langchain_rag(documents_dir="./docs/", persist_dir="./faiss_index/"):
    """构建完整的LangChain RAG系统"""
    
    # === 第1步: 文档加载 ===
    print("[1/5] 加载文档...")
    # loader = DirectoryLoader(documents_dir, glob="**/*.txt")
    # documents = loader.load()
    # print(f"    加载了 {len(documents)} 个文档")
    
    # 模拟文档
    documents_texts = [
        "RAG是检索增强生成的缩写，它结合了信息检索和文本生成技术。",
        "向量数据库如FAISS和Milvus可以高效存储和检索高维向量。",
        "Embedding模型将文本转换为固定维度的向量表示。",
        "LangChain是一个构建LLM应用的框架，提供了文档加载、分块、向量化等功能。",
    ]
    print(f"    模拟加载 {len(documents_texts)} 个文档片段")
    
    # === 第2步: 文档分块 ===
    print("[2/5] 文档分块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    # chunks = text_splitter.split_documents(documents)
    chunks = text_splitter.create_documents(documents_texts)
    print(f"    生成了 {len(chunks)} 个文档块")
    
    # === 第3步: 向量化与存储 ===
    print("[3/5] 向量化和索引构建...")
    # 使用本地Embedding模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",  # 轻量级中文embedding
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    # 构建FAISS索引
    vectorstore = FAISS.from_documents(chunks, embeddings)
    # 或使用Chroma: vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    print(f"    FAISS索引已构建，共 {vectorstore.index.ntotal} 个向量")
    
    # === 第4步: 配置检索器 ===
    print("[4/5] 配置检索器...")
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # 或 "mmr"(最大边际相关性)
        search_kwargs={"k": 3},    # 返回Top-3
    )
    
    # === 第5步: 构建RAG链 ===
    print("[5/5] 构建RAG问答链...")
    
    # 自定义Prompt模板
    rag_prompt = PromptTemplate(
        template="""基于以下参考资料回答用户问题。如果参考资料中没有相关信息，请明确说明不知道。

参考资料：
{context}

用户问题: {question}

请用中文给出准确、简洁的回答，并引用参考资料的来源:""",
        input_variables=["context", "question"],
    )
    
    # 创建RAG链（此处仅为演示流程，实际使用需配置LLM）
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",  # stuff / map_reduce / refine / map_rerank
    #     retriever=retriever,
    #     chain_type_kwargs={"prompt": rag_prompt},
    #     return_source_documents=True,
    # )
    
    # === 测试检索 ===
    test_query = "什么是RAG？它有什么特点？"
    retrieved_docs = retriever.get_relevant_documents(test_query)
    
    print(f"\n测试查询: {test_query}")
    print(f"检索到 {len(retrieved_docs)} 个相关文档:")
    for i, doc in enumerate(retrieved_docs):
        print(f"  [{i+1}] {doc.page_content[:80]}...")
    
    return vectorstore, retriever

# 执行构建
vectorstore, retriever = build_langchain_rag()

# 演示相似度搜索
print("\n" + "=" * 60)
print("相似度搜索测试")
print("=" * 60)
results = vectorstore.similarity_search_with_score("向量数据库的选择", k=2)
for doc, score in results:
    print(f"  得分 {score:.4f}: {doc.page_content[:80]}...")
```

## LlamaIndex框架实现RAG

LlamaIndex（原GPT Index）是另一个流行的RAG框架，在数据连接和索引结构方面有独特优势：

```python
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# ============================================================
# LlamaIndex RAG 实现
# ============================================================

def build_llamaindex_rag():
    """使用LlamaIndex构��RAG系统"""
    
    # === 配置全局设置 ===
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=50,
    )
    
    # === 创建文档 ===
    documents = [
        Document(text="RAG通过检索外部知识库来增强大语言模型的回答准确性，减少幻觉问题。"),
        Document(text="LlamaIndex支持多种索引结构，包括向量索引、树索引、关键字索引等。"),
        Document(text="FAISS是Meta开发的高效向量相似度搜索库，支持GPU加速和多种索引类型。"),
        Document(text="混合检索结合BM25关键词匹配和稠密向量语义检索，使用RRF融合结果。"),
    ]
    
    # === 构建索引 ===
    print("构建LlamaIndex向量索引...")
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=False,
    )
    
    # === 创建查询引擎（配置检索参数） ===
    query_engine = index.as_query_engine(
        similarity_top_k=3,         # 检索Top-3
        response_mode="compact",    # 紧凑模式：先压缩上下文再生成
        # 可选: "refine" "tree_summarize" "simple_summarize" "no_text"
    )
    
    # === 测试查询 ===
    test_queries = [
        "RAG如何减少大模型的幻觉问题？",
        "LlamaIndex支持哪些索引类型？",
        "FAISS和混合检索有什么区别？",
    ]
    
    print("=" * 60)
    print("LlamaIndex RAG 查询演示（模拟）")
    print("=" * 60)
    
    # 模拟检索结果（实际运行会调用LLM生成回答）
    for query in test_queries:
        print(f"\n查询: {query}")
        
        # 获取检索到的节点（source nodes）
        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(query)
        
        print(f"  检索到 {len(nodes)} 个相关节点:")
        for j, node in enumerate(nodes):
            print(f"    [{j+1}] (得分 {node.score:.4f}): {node.text[:60]}...")
    
    return index, query_engine

# 构建和测试
index, query_engine = build_llamaindex_rag()

# LlamaIndex高级特性：将检索器转换为查询引擎
print("\n" + "=" * 60)
print("LlamaIndex 高级功能概览")
print("=" * 60)
print("""
│  功能               │  说明                                          │
├─────────────────────┼───────────────────────────────────────────────│
│  SubQuestionQuery    │  将复杂问题分解为子问题，分别检索再综合回答      │
│  RouterQueryEngine   │  根据问题类型路由到不同的索引                   │
│  RecursiveRetriever  │  支持多级索引的递归检索                        │
│  KnowledgeGraphIndex │  支持知识图谱增强的RAG                         │
│  Chat Engine         │  支持多轮对话的RAG                             │
└─────────────────────┴───────────────────────────────────────────────┘
""")
```

## Advanced RAG技术

基础RAG已经能解决很多问题，但在复杂场景下需要更强的检索和推理能力：

### 1. 查询改写（Query Rewriting）

用户的问题往往口语化、不完整，直接检索效果差。查询改写将用户问题转换为更适合检索的形式：

```python
QUERY_REWRITE_EXAMPLE = """
原始查询: "那个能做图的AI叫啥来着？"
改写后:   "可以生成图像的人工智能工具和模型有哪些？"

原始查询: "上次说的那个方法"
改写后:   "之前讨论过的[基于上下文补充]方法"

改写策略:
  ├── 消歧: 补充代词指代、明确模糊术语
  ├── 扩展: 添加同义词、相关概念
  └── 分解: 将复合问题拆分为多个独立查询
"""
```

### 2. 多跳检索（Multi-hop Retrieval）

复杂问题需要多步推理和多轮检索：

```
用户问题: "OpenAI最新模型的参数量与GPT-3相比增长了多少倍？"

第1跳检索: "OpenAI最新模型的参数量是多少？"
  → 找到: GPT-4有约1.8万亿参数

第2跳检索: "GPT-3的参数量是多少？"
  → 找到: GPT-3有1750亿参数

第3步推理: 1.8万亿 / 1750亿 ≈ 10.3倍
  生成回答: "GPT-4的参数量约为GPT-3的10倍左右"
```

### 3. Self-RAG：自我反思的检索增强

Self-RAG让模型在生成过程中自我评估是否需要检索，以及检索到的内容是否相关：

```
┌──────────────────────────────────────────┐
│           Self-RAG 决策流程               │
├──────────────────────────────────────────┤
│                                          │
│  对于每个生成段落:                        │
│  ┌────────────────────────┐             │
│  │ ① 判断: 是否需要检索？   │              │
│  │   → 需要 → 执行检索     │              │
│  │   → 不需要 → 用自身知识   │              │
│  └────────┬───────────────┘              │
│           ▼                              │
│  ┌────────────────────────┐             │
│  │ ② 判断: 检索结果是否相关？│              │
│  │   → 相关 → 使用检索内容  │              │
│  │   → 不相关 → 重新检索   │              │
│  └────────┬───────────────┘              │
│           ▼                              │
│  ┌────────────────────────┐             │
│  │ ③ 判断: 生成内容是否受支持？│            │
│  │   → 完全支持 / 部分支持 / 不支持│        │
│  └────────────────────────┘              │
│                                          │
└──────────────────────────────────────────┘
```

### 4. HyDE（Hypothetical Document Embeddings）

HyDE的核心思想别出心裁：**先让LLM生成一个假设性回答，再用这个假设回答去检索**：

```
传统RAG:    查询文本 → Embedding → 向量检索 → 相关文档
HyDE:       查询文本 → LLM生成假设文档 → Embedding → 向量检索 → 相关文档

原理: 假设文档包含了与真实相关文档更相似的语言模式，
      能更好地弥合"简短查询"与"详细文档"之间的语义鸿沟。
```

```python
def hyde_retrieval_example(query):
    """HyDE检索示例"""
    # 步骤1: 用LLM生成假设文档
    hyde_prompt = f"""请根据以下问题，写一段假设性的回答段落。
这段回答不需要完全准确，目的是用于后续检索。
问题: {query}

假设回答:"""
    
    # hypothetical_doc = llm.generate(hyde_prompt)  # 实际调用LLM
    hypothetical_doc = f"关于{query}，相关的技术方案包括..."  # 模拟
    
    # 步骤2: 用假设文档（而非原始查询）进行向量检索
    # hyde_embedding = embedding_model.encode(hypothetical_doc)
    # results = vectorstore.similarity_search_by_vector(hyde_embedding)
    
    return hypothetical_doc  # 实际应用返回检索结果
```

## RAG评估

评估RAG系统需要同时关注检索质量和生成质量两个维度：

| 评估指标 | 英文名 | 定义 | 评估方法 |
|---------|--------|------|---------|
| 忠实度 | Faithfulness | 生成回答中的事实是否在检索文档中找到支撑 | 分解为原子声明，逐一验证 |
| 答案相关性 | Answer Relevance | 回答是否直接响应了用户问题 | 生成反向问题，与原始问题比对 |
| 上下文相关性 | Context Relevance | 检索文档是否与问题相关 | 判断每段文档的必要性 |
| 上下文召回率 | Context Recall | 检索文档覆盖了回答中多少必要信息 | 对比回答所需信息与检索内容 |
| 上下文精确率 | Context Precision | 检索结果中相关文档的比例 | 标注检索结果的相关性 |

```python
# 简单评估示例
def evaluate_rag_response(question, answer, retrieved_docs, ground_truth=None):
    """
    评估RAG回答质量
    
    返回评估结果字典
    """
    evaluation = {
        "question": question,
        "answer_length": len(answer),
        "num_retrieved_docs": len(retrieved_docs),
        "avg_doc_length": np.mean([len(d) for d in retrieved_docs]) if retrieved_docs else 0,
    }
    
    # 简单的上下文利用率评估（模拟）
    # 实际应用中使用RAGAS: https://github.com/explodinggradients/ragas
    print(f"RAG回答评估 (模拟)")
    print(f"  问题: {question[:50]}...")
    print(f"  回答长度: {len(answer)} 字符")
    print(f"  检索文档数: {len(retrieved_docs)}")
    print(f"  平均文档长度: {evaluation['avg_doc_length']:.0f} 字符")
    
    return evaluation

# 测试评估
evaluate_rag_response(
    question="什么是RAG？",
    answer="RAG是检索增强生成...",
    retrieved_docs=["RAG技术文档..."],
)
```

## RAG vs 长上下文LLM：如何选择？

随着模型上下文窗口的不断增长（GPT-4 Turbo 128K, Claude 200K, Gemini 1.5 Pro 1M+），一个自然的问题是：**还需要RAG吗？**

```
┌──────────────┬─────────────────────┬─────────────────────┐
│      维度      │        RAG          │   长上下文LLM        │
├──────────────┼─────────────────────┼─────────────────────┤
│ 适用知识量     │ 数十万到数十亿文档    │ 一次性放几百页文档    │
│ 推理延迟       │ 检索(ms) + 生成(s)   │ 长上下文处理慢       │
│ 成本           │ 按文档量+API调用计费  │ 按token数计费        │
│ 知识更新       │ 更新文档库即可        │ 需要重新训练         │
│ 可解释性       │ 高，可追溯到源文档     │ 低，黑盒推理         │
│ 精确引用       │ 强，天然支持          │ 弱，容易张冠李戴      │
└──────────────┼─────────────────────┼─────────────────────┘
```

**选择建议**：
- 知识库庞大（>10万文档）且频繁更新 → **RAG**
- 需要精确溯源和引用 → **RAG**
- 单篇长文档深度分析（如整本书、长论文） → **长上下文LLM**
- 最佳实践：**RAG + 长上下文结合**，用RAG检索相关片段，再用长上下文LLM进行深度推理

```python
# RAG vs 长上下文的实用判断函数
def recommend_approach(total_docs, avg_doc_length, update_frequency, need_citation):
    """
    基于场景特征推荐使用RAG还是长上下文LLM
    
    返回: "rag" / "long_context" / "hybrid"
    """
    score_rag = 0
    score_lc = 0
    
    # 文档量判断
    if total_docs > 10000:
        score_rag += 3
    elif total_docs > 100:
        score_rag += 1
    else:
        score_lc += 2
    
    # 总token量判断（粗略估计）
    total_tokens = total_docs * avg_doc_length * 1.3  # 1.3是token/字符比例
    if total_tokens > 200000:  # 超过大多数模型的上下文窗口
        score_rag += 2
    else:
        score_lc += 1
    
    # 更新频率
    if update_frequency == "daily":
        score_rag += 2
    elif update_frequency == "weekly":
        score_rag += 1
    
    # 引用需求
    if need_citation:
        score_rag += 2
    
    if score_rag > score_lc:
        return "RAG"
    elif score_lc > score_rag:
        return "长上下文LLM"
    else:
        return "RAG + 长上下文 混合方案"

# 场景测试
print("=" * 60)
print("RAG vs 长上下文LLM 选择建议")
print("=" * 60)
scenarios = [
    (50000, 1000, "daily", True, "企业知识库问答"),
    (5, 5000, "monthly", False, "单篇论文分析"),
    (200, 2000, "weekly", True, "客服系统"),
]

for total_docs, avg_len, update_freq, need_cite, desc in scenarios:
    recommendation = recommend_approach(total_docs, avg_len, update_freq, need_cite)
    print(f"  场景: {desc}")
    print(f"    文��量={total_docs}, 更新={update_freq}, 需引用={need_cite}")
    print(f"    推荐: {recommendation}\n")
```

## 小结

| 要点 | 内容 |
|------|------|
| RAG动机 | 解决LLM三大局限：知识截止日期、幻觉、缺乏领域专业知识 |
| RAG架构 | 检索器(Retriever) + 生成器(Generator)，先查资料再回答 |
| 向量数据库 | FAISS(本地)/Milvus(分布式)/Chroma(简易)/Pinecone(SaaS) |
| Embedding模型 | bge-large-zh(中文标杆)、text-embedding-3(多语言)、m3e-base(轻量) |
| 文档分块 | 固定大小/递归分块/语义分块/Small-to-Big，推荐递归分块为默认方案 |
| 检索策略 | BM25(稀疏)+Dense(稠密)→RRF混合融合，配合Cross-encoder��排序 |
| LangChain RAG | 文档加载→分块→向量化→检索→生成，完整的工具链 |
| LlamaIndex | 强大的索引结构和数据连接能力，适合复杂数据源场景 |
| 高级RAG | 查询改写、多跳检索、Self-RAG、HyDE等方法进一步提升效果 |
| RAG评估 | 忠实度、答案相关性、上下文相关性、上下文召回率/精��率 |
| RAG vs 长上下文 | 知识库大/需溯源→RAG；单文档深度分析→长上下文；最优→混合方案 |

---

| [← 回到目录](../README.md) | [上一篇：03-RLHF与对齐训练](03-RLHF与对齐训练.md) | [下一篇：../11-高级应用篇/01-多模态NLP](../11-高级应用篇/01-多模态NLP.md) |
|---|---|---|
