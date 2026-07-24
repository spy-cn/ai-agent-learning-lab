# PyTorch与NLP

前面的章节我们使用 sklearn、gensim 等传统机器学习工具处理文本，它们在中小规模任务上表现优秀，但面对**大规模语料**、**复杂语义建模**和**端到端学习**的需求时显得力不从心。深度学习框架（尤其是 PyTorch）凭借**自动求导**、**GPU 加速**和**动态计算图**成为现代 NLP 的标配。本篇将系统介绍如何用 PyTorch 处理文本数据，并从零构建一个文本分类模型，为后续的词嵌入、RNN、CNN、Attention 等章节打下基础。

---

## 一、PyTorch 基础回顾

### 1.1 Tensor 创建与操作

Tensor（张量）是 PyTorch 的核心数据结构，类似 NumPy 的 ndarray，但支持 GPU 计算和自动求导。

```python
import torch

# ===== 创建 Tensor 的常见方式 =====
# 从列表创建
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(f"a:\n{a}\nshape={a.shape}, dtype={a.dtype}")

# 常用初始化
zeros = torch.zeros(3, 4)        # 全零
ones = torch.ones(2, 3)          # 全一
rand = torch.rand(2, 3)          # 均匀分布随机
randn = torch.randn(2, 3)        # 标准正态分布
arange = torch.arange(0, 10, 2)  # 序列 [0, 2, 4, 6, 8]

# ===== 形状操作 =====
x = torch.randn(2, 3, 4)
print(f"\n原始形状: {x.shape}")
print(f"reshape: {x.reshape(2, 12).shape}")    # (2, 12)
print(f"permute: {x.permute(2, 0, 1).shape}")  # (4, 2, 3) 维度重排
print(f"unsqueeze: {x.unsqueeze(0).shape}")    # (1, 2, 3, 4) 增加维度

# ===== 运算 =====
b = torch.tensor([[5., 6.], [7., 8.]])
print(f"\n逐元素加: \n{a + b}")
print(f"矩阵乘法: \n{a @ b}")          # 等价于 torch.matmul
print(f"逐元素乘: \n{a * b}")          # Hadamard 积
print(f"求和: {a.sum()}, 均值: {a.mean()}")
print(f"按维度求和: {a.sum(dim=0)}")    # 按列求和
```

### 1.2 自动求导（Autograd）

autograd 是 PyTorch 的灵魂，它自动追踪张量上的运算并构建计算图，反向传播时自动计算梯度。

```python
import torch

# ===== 标量求导示例 =====
# y = x^2 + 2x + 1, dy/dx = 2x + 2
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1
y.backward()  # 反向传播
print(f"x={x.item()}, y={y.item()}, dy/dx={x.grad.item()}")  # 2*3+2=8

# ===== 矩阵求导示例：简单线性回归 =====
torch.manual_seed(42)
# 真实参数 w=2, b=1
X = torch.randn(100, 1)
y_true = 2 * X + 1 + 0.1 * torch.randn(100, 1)

# 可训练参数
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()                 # 清空梯度
    y_pred = X @ w + b                    # 前向传播
    loss = ((y_pred - y_true) ** 2).mean()  # MSE
    loss.backward()                       # 反向传播，自动计算梯度
    optimizer.step()                      # 更新参数
    if epoch % 20 == 0:
        print(f"epoch {epoch}: loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

print(f"\n最终参数: w={w.item():.4f} (真值2.0), b={b.item():.4f} (真值1.0)")
```

### 1.3 GPU 使用

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 将 Tensor 移到 GPU
x = torch.randn(3, 3).to(device)

# 注意：参与运算的 Tensor 必须在同一设备
# y = torch.randn(3, 3)          # CPU
# z = x + y                       # ❌ 报错：设备不一致
y = torch.randn(3, 3).to(device)
z = x + y                         # ✅
```

---

## 二、nn.Embedding 详解

`nn.Embedding` 是 NLP 中最重要的层之一，它将**离散的词索引**映射为**稠密的连续向量**。

### 2.1 为什么需要 Embedding

```
传统 One-Hot 表示的问题:
  词汇表大小 V = 50000
  每个词 = 一个 50000 维的稀疏向量（只有1个位置是1）

  "猫"   = [0, 0, ..., 1, ..., 0]   ← 50000维，只有一个1
  "狗"   = [0, 0, ..., 0, 1, .., 0] ← 50000维
  "汽车" = [0, 1, 0, ..., 0]        ← 50000维

  问题：维度高、稀疏、无法表达词之间相似度

Embedding 表示:
  词汇表大小 V = 50000, 嵌入维度 d = 300
  每个词 = 一个 300 维的稠密向量（通过训练学习）

  "猫"   → [0.21, -0.43, 0.85, ...]  ← 300维
  "狗"   → [0.19, -0.40, 0.80, ...]  ← 300维（与"猫"接近）
  "汽车" → [-0.55, 0.32, -0.11, ...] ← 300维（与"猫狗"差异大）

  优势：低维、稠密、能捕捉语义相似性
```

### 2.2 nn.Embedding 的使用

```python
import torch
import torch.nn as nn

# ===== 基本 API =====
# num_embeddings: 词表大小; embedding_dim: 嵌入维度
embedding = nn.Embedding(num_embeddings=10, embedding_dim=4)
print(f"权重形状: {embedding.weight.shape}")  # (10, 4)

# 查表：输入索引，返回对应向量
idx = torch.tensor([0, 3, 7])  # 三个词的索引
vecs = embedding(idx)
print(f"查表结果: \n{vecs}")
print(f"结果形状: {vecs.shape}")  # (3, 4)

# 批量查表（用于批处理）
batch_idx = torch.tensor([
    [0, 1, 2],    # 句子1: 词索引 [0, 1, 2]
    [3, 4, 5],    # 句子2: 词索引 [3, 4, 5]
])
batch_vecs = embedding(batch_idx)
print(f"\n批量结果形状: {batch_vecs.shape}")  # (2, 3, 4) → (batch, seq_len, emb_dim)
```

### 2.3 Embedding 本质是查表

```
nn.Embedding 内部维护一个权重矩阵 W (V × d):

     词表     ←  嵌入维度 d →
  ┌───────────────────────────┐
  │ 0  │ 0.21  -0.43   0.85  0.12 │  ← "我"    W[0]
  │ 1  │ 0.55   0.31  -0.22  0.78 │  ← "爱"    W[1]
  │ 2  │-0.11   0.44   0.67 -0.33 │  ← "编"    W[2]
  │ 3  │ 0.88  -0.55   0.01  0.42 │  ← "程"    W[3]
  │ ...│         ...              │
  │ V-1│-0.34   0.72   0.15 -0.91 │  ← "<pad>" W[V-1]
  └───────────────────────────┘

输入索引 [1, 2, 3]  →  查表取第1、2、3行  →  拼成 (3, d) 的矩阵
                      ↓
              [0.55, 0.31, -0.22, 0.78]
              [-0.11, 0.44, 0.67, -0.33]
              [0.88, -0.55, 0.01, 0.42]

梯度更新时：只有被查过的行（词）的梯度非零，其余保持不变。
```

---

## 三、PyTorch 处理文本数据流程

### 3.1 完整流程概览

```
原始文本
  │
  ├── 1. 分词（Tokenization）
  │      "我爱自然语言处理" → ["我", "爱", "自然", "语言", "处理"]
  │
  ├── 2. 构建词表（Vocabulary）
  │      为每个词分配唯一索引: {"我":0, "爱":1, "自然":2, ...}
  │
  ├── 3. 文本转索引序列（Numericalization）
  │      ["我", "爱", "自然", "语言", "处理"] → [0, 1, 2, 3, 4]
  │
  ├── 4. 统一长度（Padding / Truncation）
  │      不同句子长度不同 → 补齐到固定长度
  │      [0,1,2]      → [0,1,2,0,0]  (补<pad>, 索引通常为0)
  │      [5,6,7,8,9] → [5,6,7,8,9]   (已满足长度)
  │
  ├── 5. 构建 Dataset 和 DataLoader
  │      封装成可迭代的批次数据
  │
  └── 6. 喂入模型
         batch → Embedding → RNN/CNN/Attention → 输出
```

### 3.2 代码实现

```python
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ===== 1. 原始数据 =====
texts = [
    "这部电影 太 好看 了",
    "剧情 真是 烂透 了",
    "演员 演技 很 棒",
    "浪费 时间 别 看",
    "画面 精美 值得 推荐",
    "无聊 透顶 不 推荐",
]
labels = [1, 0, 1, 0, 1, 0]  # 1=正面, 0=负面

# ===== 2. 构建词表 =====
def build_vocab(text_list):
    # 分词（这里假设已经按空格分好）
    all_words = []
    for text in text_list:
        all_words.extend(text.split())
    # 统计词频，按频率降序分配索引
    counter = Counter(all_words)
    # 0: <pad>, 1: <unk>
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, _ in counter.most_common():
        vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(texts)
print(f"词表大小: {len(vocab)}")
print(f"词表: {vocab}")

# ===== 3. 文本转索引序列 =====
def text_to_indices(text, vocab, max_len=8):
    indices = [vocab.get(word, vocab['<unk>']) for word in text.split()]
    # 截断
    indices = indices[:max_len]
    # 补齐
    indices = indices + [0] * (max_len - len(indices))
    return indices

X = [text_to_indices(t, vocab) for t in texts]
print(f"\n文本转索引示例:")
for text, x in zip(texts[:2], X[:2]):
    print(f"  '{text}' → {x}")

# ===== 4. 构建 Dataset =====
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=8):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 转索引 + padding
        indices = text_to_indices(self.texts[idx], self.vocab, self.max_len)
        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

dataset = TextDataset(texts, labels, vocab)
print(f"\nDataset 长度: {len(dataset)}")
print(f"第0个样本: {dataset[0]}")

# ===== 5. 构建 DataLoader =====
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
print(f"\nDataLoader 批次数: {len(dataloader)}")
for batch_x, batch_y in dataloader:
    print(f"  batch_x 形状: {batch_x.shape}, batch_y: {batch_y}")
    break
```

---

## 四、文本批处理：Padding 和 Masking

### 4.1 为什么需要 Padding

```
批处理时，同一批的序列长度必须一致:

句子1: "我 爱 深度 学习"           → [3, 1, 5, 8, 2]   (长度5)
句子2: "这 部 电影 太 好看 了"    → [6, 7, 4, 9, 10, 3] (长度6)
句子3: "烂透 了"                  → [11, 3]            (长度2)

不补齐 → 无法堆叠成矩阵 → 无法批处理

Padding 后（max_len=6）:
句子1: [3, 1, 5, 8, 2, 0]   ← 补一个 <pad>
句子2: [6, 7, 4, 9, 10, 3]  ← 已是6
句子3: [11, 3, 0, 0, 0, 0]  ← 补四个 <pad>

可堆叠成 (3, 6) 的矩阵 → 可批处理
```

### 4.2 Masking 机制

Padding 引入的 `<pad>` 位置是**无意义的**，在计算损失或聚合时应当忽略，这就是 Masking。

```python
import torch
import torch.nn.functional as F

# 示例 batch
batch = torch.tensor([
    [3, 1, 5, 8, 2, 0],   # 第1句：有效长度5，最后一个<pad>
    [6, 7, 4, 9, 10, 3],  # 第2句：有效长度6，无<pad>
    [11, 3, 0, 0, 0, 0],  # 第3句：有效长度2，后4个<pad>
])

# ===== 构造 mask =====
# 1=有效位置, 0=padding位置
mask = (batch != 0).float()  # (batch_size, seq_len)
print(f"Mask:\n{mask}")

# ===== Mask 在池化中的应用 =====
# 场景：对序列向量取平均时，应忽略 padding 位置
# 假设经过 Embedding 后的向量
embed = torch.nn.Embedding(20, 4)
vec = embed(batch)  # (3, 6, 4)
print(f"\nEmbedding 输出形状: {vec.shape}")

# 错误做法：直接 mean，padding 位置会稀释结果
wrong_mean = vec.mean(dim=1)

# 正确做法：masked mean，只对有效位置取平均
mask_expanded = mask.unsqueeze(-1)  # (3, 6, 1) → 广播
sum_vec = (vec * mask_expanded).sum(dim=1)  # 只累加有效位置
valid_len = mask.sum(dim=1, keepdim=True)   # 每句的有效长度
correct_mean = sum_vec / valid_len.clamp(min=1)
print(f"正确池化结果形状: {correct_mean.shape}")
```

---

## 五、从零构建文本分类模型

### 5.1 模型架构

我们将用**词袋平均（Bag-of-Vectors）**作为基线模型：词嵌入 → 取平均 → 全连接 → 分类。

```
输入: [3, 1, 5, 8, 2, 0]   ← 词索引序列 (batch_size=1, seq_len=6)
        │
        ▼
  ┌─────────────┐
  │  Embedding   │  查表得到稠密向量
  │  (V × d)     │
  └──────┬──────┘
         │
         ▼   (batch, seq_len, d)
  ┌─────────────┐
  │  Mean Pool   │  沿序列维度取平均
  │  (带 mask)   │
  └──────┬──────┘
         │
         ▼   (batch, d)
  ┌─────────────┐
  │  Linear      │  全连接映射到类别数
  │  (d → C)     │
  └──────┬──────┘
         │
         ▼
  logits (batch, C)  → softmax → 分类概率
```

### 5.2 完整训练代码

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# ===== 1. 准备数据 =====
texts = [
    "这部 电影 太 好看 了", "演员 演技 很 棒", "画面 精美 值得 推荐",
    "剧情 紧凑 引人入胜", "配乐 动人 优秀",
    "剧情 真是 烂透 了", "浪费 时间 别 看", "无聊 透顶 不 推荐",
    "演员 演技 差", "画面 粗糙 垃圾",
] * 10  # 重复扩充数据
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 10

# ===== 2. 构建词表 =====
counter = Counter()
for t in texts:
    counter.update(t.split())
vocab = {'<pad>': 0, '<unk>': 1}
for w, _ in counter.most_common():
    vocab[w] = len(vocab)
vocab_size = len(vocab)

def text_to_seq(text, max_len=8):
    idx = [vocab.get(w, 1) for w in text.split()][:max_len]
    return idx + [0] * (max_len - len(idx))

# ===== 3. Dataset / DataLoader =====
class TextClsDataset(Dataset):
    def __init__(self, texts, labels):
        self.x = torch.tensor([text_to_seq(t) for t in texts], dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

dataset = TextClsDataset(texts, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ===== 4. 模型定义 =====
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        mask = (x != 0).float().unsqueeze(-1)  # (batch, seq_len, 1)
        emb = self.embedding(x) * mask          # (batch, seq_len, embed_dim)
        # masked mean pooling
        pooled = emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (batch, embed_dim)
        logits = self.fc(pooled)               # (batch, num_classes)
        return logits

model = TextClassifier(vocab_size, embed_dim=16, num_classes=2)
print(model)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

# ===== 5. 训练 =====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(30):
    total_loss = 0
    correct = 0
    total = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_y)
        correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total += len(batch_y)
    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d} | Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f}")

# ===== 6. 推理测试 =====
test_texts = ["电影 好看 推荐", "剧情 无聊 垃圾"]
model.eval()
with torch.no_grad():
    for t in test_texts:
        x = torch.tensor([text_to_seq(t)], dtype=torch.long)
        prob = torch.softmax(model(x), dim=-1)
        pred = prob.argmax(dim=-1).item()
        label = '正面' if pred == 1 else '负面'
        print(f"'{t}' → {label} (置信度: {prob[0][pred]:.2%})")
```

---

## 六、模型保存与加载

### 6.1 两种保存方式

```python
import torch

# ===== 方式1：只保存模型参数（推荐） =====
torch.save(model.state_dict(), 'text_classifier.pt')
# state_dict 是一个字典：{层名: 参数Tensor}

# 加载
model_loaded = TextClassifier(vocab_size, embed_dim=16, num_classes=2)
model_loaded.load_state_dict(torch.load('text_classifier.pt'))
model_loaded.eval()

# ===== 方式2：保存整个模型（含架构，不推荐） =====
# torch.save(model, 'full_model.pt')
# model = torch.load('full_model.pt')  # 依赖类定义的位置

# ===== 保存训练检查点（含优化器状态，用于恢复训练） =====
checkpoint = {
    'epoch': 30,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'loss': total_loss / total,
}
torch.save(checkpoint, 'checkpoint.pt')

# 恢复训练
ckpt = torch.load('checkpoint.pt')
model.load_state_dict(ckpt['model_state'])
optimizer.load_state_dict(ckpt['optimizer_state'])
```

### 6.2 同时保存词表

```python
import json

# 训练完成后，保存模型 + 词表（推理时必需）
save_package = {
    'model_state': model.state_dict(),
    'config': {
        'vocab_size': vocab_size,
        'embed_dim': 16,
        'num_classes': 2,
        'max_len': 8,
    },
    'vocab': vocab,
}
torch.save(save_package, 'nlp_model_package.pt')

# 加载推理
pkg = torch.load('nlp_model_package.pt')
m = TextClassifier(**pkg['config'])
m.load_state_dict(pkg['model_state'])
m.eval()
v = pkg['vocab']
```

---

## 七、GPU 使用要点

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 把模型和数据移到 GPU
model = model.to(device)

for batch_x, batch_y in dataloader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    logits = model(batch_x)  # 在 GPU 上计算
    break
```

```
GPU 使用注意事项:
  ┌──────────────────────────────────────────────┐
  │ 1. 模型和输入数据必须在同一设备              │
  │ 2. 只有 Tensor 类型可移到 GPU（Python 数值不行）│
  │ 3. 损失结果 item() 会自动转 CPU 标量         │
  │ 4. 推理时用 torch.no_grad() 省显存           │
  │ 5. DataLoader 用 num_workers>0 加速数据加载   │
  └──────────────────────────────────────────────┘
```

---

## 小结

| 要点 | 内容 |
|------|------|
| Tensor | PyTorch 核心数据结构，支持 GPU 加速和自动求导 |
| Autograd | `requires_grad=True` + `backward()` 自动计算梯度 |
| nn.Embedding | 将词索引映射为稠密向量，本质是可学习的查表 |
| 文本处理流程 | 分词 → 构词表 → 转索引 → padding → Dataset/DataLoader |
| Padding | 用索引0补齐到固定长度，使不同长度句子可批处理 |
| Masking | 在聚合（如平均池化）时忽略 padding 位置 |
| 训练循环 | forward → loss → backward → step → zero_grad |
| 模型保存 | 推荐 `state_dict()`，配合词表一起保存 |
| GPU | `.to(device)` 迁移模型和数据，注意设备一致性 |

---

| [← 回到目录](../README.md) | [上一篇：文本聚类与主题模型](../05-传统机器学习篇/05-文本聚类与主题模型.md) | [下一篇：词嵌入实践](02-词嵌入实践.md) |
|---|---|---|
