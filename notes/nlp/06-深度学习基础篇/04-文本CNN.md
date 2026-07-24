# 文本CNN

上一篇我们用 RNN/LSTM 处理序列，它们擅长捕捉长距离依赖，但有一个天然短板：**必须按时间步顺序计算**，无法并行。**卷积神经网络（CNN）** 提供了一种截然不同的视角——用**一维卷积**在文本上滑动，像 n-gram 一样提取局部特征。TextCNN 凭借**训练速度快**和**局部特征建模能力强**，成为文本分类的经典基线。

---

## 一、CNN 用于文本：一维卷积

### 1.1 从图像卷积到文本卷积

```
图像 CNN: 二维卷积 (2D Conv)
  输入: (H, W, C) 高×宽×通道（如 224×224×3）
  卷积核: (kH, kW, C) 在 H、W 两个方向滑动

文本 CNN: 一维卷积 (1D Conv)
  输入: (seq_len, embed_dim) 序列长度×嵌入维度
  卷积核: (kernel_size, embed_dim) 只在序列方向滑动

  ┌──────────────────────────────────────┐
  │  文本卷积示意（embed_dim 作为"宽度"）  │
  │                                       │
  │  词嵌入矩阵:                           │
  │    ┌───┬───┬───┬───┬───┬───┐         │
  │    │我 │爱 │自 │然 │语 │言 │  seq_len=6
  │    ├───┼───┼───┼───┼───┼───┤         │
  │    │v₁ │v₁ │v₁ │v₁ │v₁ │v₁ │  embed_dim
  │    │v₂ │v₂ │v₂ │v₂ │v₂ │v₂ │         │
  │    │.. │.. │.. │.. │.. │.. │         │
  │    └───┴───┴───┴───┴───┴───┘         │
  │                                       │
  │  卷积核 (kernel_size=3, embed_dim):    │
  │    ┌───┬───┬───┐                      │
  │    │ w │ w │ w │  覆盖连续3个词        │
  │    └───┴───┴───┘                      │
  │         ↓ 滑动方向 →                  │
  │  输出: [c₁, c₂, c₃, c₄] 4个特征值     │
  └──────────────────────────────────────┘

  关键: 卷积核的"宽度"必须等于 embed_dim
        只在 seq_len 方向滑动（提取 n-gram 特征）
```

### 1.2 一维卷积计算过程

```
输入: (batch, seq_len, embed_dim)，假设 batch=1, seq_len=6, embed_dim=5

       t=1  t=2  t=3  t=4  t=5  t=6
     ┌────┬────┬────┬────┬────┬────┐
     │ e₁₁│ e₂₁│ e₃₁│ e₄₁│ e₅₁│ e₆₁│   embed_dim=5
     │ e₁₂│ e₂₂│ e₃₂│ e₄₂│ e₅₂│ e₆₂│
     │ e₁₃│ e₂₃│ e₃₃│ e₄₃│ e₅₃│ e₆₃│
     │ e₁₄│ e₂₄│ e₃₄│ e₄₄│ e₅₄│ e₆₄│
     │ e₁₅│ e₂₅│ e₃₅│ e₄₅│ e₅₅│ e₆₅│
     └────┴────┴────┴────┴────┴────┘

卷积核: kernel_size=3, 输出通道=out_channels=1

  窗口1: 覆盖 t=1,2,3 → 点积求和 → c₁
  窗口2: 覆盖 t=2,3,4 → 点积求和 → c₂
  窗口3: 覆盖 t=3,4,5 → 点积求和 → c₃
  窗口4: 覆盖 t=4,5,6 → 点积求和 → c₄

  c_i = σ(W · [e_{i,1..5}, e_{i+1,1..5}, e_{i+2,1..5}] + b)

输出: [c₁, c₂, c₃, c₄]，长度 = seq_len - kernel_size + 1 = 6-3+1 = 4
```

### 1.3 PyTorch 一维卷积 API

```python
import torch
import torch.nn as nn

# Conv1d 输入形状: (batch, in_channels, seq_len)
# 注意: embed_dim 作为 in_channels（通道数）

conv = nn.Conv1d(
    in_channels=5,       # embed_dim
    out_channels=10,     # 卷积核数量（输出特征图数）
    kernel_size=3,       # 卷积核覆盖的词数
)

# 模拟输入
batch_size, seq_len, embed_dim = 2, 6, 5
x = torch.randn(batch_size, seq_len, embed_dim)
# Conv1d 要求 (batch, channels, seq_len) → 需要转置
x = x.transpose(1, 2)  # (2, 5, 6) ← embed_dim 作为通道
print(f"转置后输入: {x.shape}")

out = conv(x)
print(f"卷积输出: {out.shape}")  # (2, 10, 4) ← seq_len - kernel_size + 1 = 4
```

---

## 二、TextCNN 架构

### 2.1 整体架构

TextCNN（Kim 2014）的核心创新：**多尺度卷积核**并行提取不同粒度的 n-gram 特征，再通过池化获取最重要的特征。

```
TextCNN 完整架构:

输入层:
  词索引: [3, 1, 5, 8, 2, 7, 4]   seq_len=7
    │
    ▼
Embedding:
  ┌────┬────┬────┬────┬────┬────┬────┐
  │ e₁ │ e₂ │ e₃ │ e₄ │ e₅ │ e₆ │ e₇ │   每个词 → embed_dim=5维
  └────┴────┴────┴────┴────┴────┴────┘
    │
    ▼ 转置为 (embed_dim, seq_len)
多尺度并行卷积（多个不同 kernel_size 的卷积核）:
  ┌─────────────┐ ┌───────────┐ ┌─────────┐
  │ kernel=2    │ │ kernel=3  │ │ kernel=4│
  │ out_ch=6    │ │ out_ch=6  │ │ out_ch=6│
  └──────┬──────┘ └─────┬─────┘ └────┬────┘
         │              │            │
         ▼              ▼            ▼
  [c₁..c₆]       [c₁..c₅]     [c₁..c₄]     ← 各卷积输出序列
  (6, seq-1)     (6, seq-2)   (6, seq-3)
         │              │            │
         ▼              ▼            ▼
全局最大池化（每个通道取最大值）:
  [m₁..m₆]       [m₁..m₆]     [m₁..m₆]     ← 每个卷积核只保留1个最强特征
  (6,)           (6,)         (6,)
         │              │            │
         └──────────────┼────────────┘
                        ▼
              拼接 (concat): 6+6+6 = 18维
                        │
                        ▼
                 Dropout (防过拟合)
                        │
                        ▼
              全连接 (Linear): 18 → num_classes
                        │
                        ▼
                   logits / softmax
```

### 2.2 为什么用多个不同尺寸的卷积核

```
不同 kernel_size 捕捉不同粒度的 n-gram:

  kernel_size=2 → 关注 bigram（2个连续词）
    "不 好" / "很 棒" / "浪费 时间"

  kernel_size=3 → 关注 trigram（3个连续词）
    "太 好看 了" / "不 推荐 太"

  kernel_size=4 → 关注 4-gram
    "这部 电影 太 好看" / "剧情 真是 烂透 了"

多尺度并行的好处:
  ┌────────────────────────────────────────┐
  │ 1. 不同长度的短语特征都能被捕捉         │
  │ 2. 模型自动学习哪些 n-gram 最有判别力   │
  │ 3. 并行计算，速度远快于 RNN             │
  └────────────────────────────────────────┘
```

### 2.3 全局最大池化的作用

```
每个卷积核在序列上滑动产生多个特征值，全局最大池化只保留最重要的一个:

  卷积输出: [0.2, 1.5, -0.3, 0.8, 2.1, -1.0]
                    ↑               ↑
                    局部最大        全局最大
  池化后: 2.1  ← 只保留最强的那个特征

意义:
  - 不管这个关键 n-gram 出现在句子哪个位置，都能被提取出来
  - 把变长的卷积输出转成定长表示（利于后续全连接）
  - 类比：扫描一段话，只记住最"醒目"的那个点
```

---

## 三、TextCNN vs LSTM 对比

```
┌──────────────────┬─────────────────────┬─────────────────────┐
│       维度       │       TextCNN       │        LSTM         │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 序列处理方式     │ 局部滑动窗口        │ 逐步递进            │
│ 并行化           │ ✅ 完全可并行        │ ❌ 必须按顺序计算    │
│ 训练速度         │ 快                  │ 慢                  │
│ 长距离依赖       │ 弱（受kernel限制）  │ 强（隐状态传递）    │
│ 位置敏感性       │ 不敏感（最大池化）  │ 敏感（按序处理）    │
│ 局部特征         │ 强（n-gram）        │ 较弱                │
│ 适合任务         │ 分类（短文本）      │ 序列标注、生成      │
│ 超参数           │ kernel_sizes组合    │ hidden_dim, layers  │
│ 内存占用         │ 中                  │ 大（存所有时间步）  │
│ 典型应用         │ 情感分析、主题分类  │ 翻译、NER、语言模型│
└──────────────────┴─────────────��───────┴─────────────────────┘

实践建议:
  - 文本分类首选 TextCNN（快且效果好）
  - 需要生成或序列标注时用 LSTM/GRU
  - 两者都可作为基线，效果接近时选 CNN（更快）
```

---

## 四、PyTorch TextCNN 完整实现

### 4.1 模型定义

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 kernel_sizes=[2, 3, 4], num_filters=10, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 多尺度一维卷积（用 ModuleList 管理多个 Conv1d）
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        # 全连接：num_filters × 卷积核种类数 → num_classes
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)          # (batch, seq_len, embed_dim)
        emb = emb.transpose(1, 2)        # (batch, embed_dim, seq_len) ← Conv1d 格式

        # 对每个卷积核：Conv → ReLU → 全局最大池化
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(emb))         # (batch, num_filters, seq_len - k + 1)
            # 全局最大池化：取每个通道在整个序列上的最大值
            pooled = F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2)
            # pooled: (batch, num_filters)
            conv_outs.append(pooled)

        # 拼接所有卷积结果
        cat = torch.cat(conv_outs, dim=1)  # (batch, num_filters * len(kernel_sizes))
        cat = self.dropout(cat)
        logits = self.fc(cat)             # (batch, num_classes)
        return logits

# 测试模型结构
model = TextCNN(
    vocab_size=1000, embed_dim=50, num_classes=2,
    kernel_sizes=[2, 3, 4], num_filters=8,
)
print(model)
print(f"参数量: {sum(p.numel() for p in model.parameters())}")

# 前向传播测试
x = torch.randint(0, 1000, (4, 10))  # batch=4, seq_len=10
out = model(x)
print(f"\n输入: {x.shape} → 输出: {out.shape}")  # (4, 2)
```

### 4.2 可视化各层输出形状

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 详细打印每一层的形状变化
print("=== TextCNN 各层形状追踪 ===\n")
batch_size, seq_len, embed_dim = 2, 10, 50
vocab_size = 1000

x = torch.randint(0, vocab_size, (batch_size, seq_len))
print(f"输入: {x.shape}")

emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
e = emb(x)
print(f"Embedding: {e.shape}")  # (2, 10, 50)

e_t = e.transpose(1, 2)
print(f"转置: {e_t.shape}")  # (2, 50, 10)

for k in [2, 3, 4]:
    conv = nn.Conv1d(embed_dim, out_channels=8, kernel_size=k)
    c = F.relu(conv(e_t))
    print(f"Conv1d(k={k}): {c.shape}")  # (2, 8, 10-k+1)
    p = F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2)
    print(f"  MaxPool: {p.shape}")  # (2, 8)
```

---

## 五、TextCNN 用于情感分析

### 5.1 完整训练代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# ===== 1. 数据 =====
texts = [
    "这部 电影 真的 太 好看 了", "剧情 紧凑 引人入胜",
    "演员 演技 精湛 出色", "画面 精美 值得 推荐",
    "配乐 动人 故事 感人", "这部电影 太 棒 了",
    "这部 电影 真的 太 烂 了", "剧情 烂 演技 差",
    "浪费 时间 无聊 透顶", "画面 粗糙 不 推荐",
    "配乐 难听 故事 无趣", "这部电影 太 差 了",
] * 15
labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0] * 15

# ===== 2. 词表 =====
counter = Counter()
for t in texts:
    counter.update(t.split())
vocab = {'<pad>': 0, '<unk>': 1}
for w, _ in counter.most_common():
    vocab[w] = len(vocab)

max_len = 12
def text_to_seq(text):
    idx = [vocab.get(w, 1) for w in text.split()][:max_len]
    return idx + [0] * (max_len - len(idx))

# ===== 3. Dataset =====
class SentiDataset(Dataset):
    def __init__(self, texts, labels):
        self.x = torch.tensor([text_to_seq(t) for t in texts])
        self.y = torch.tensor(labels)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

loader = DataLoader(SentiDataset(texts, labels), batch_size=16, shuffle=True)

# ===== 4. 模型 =====
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 kernel_sizes, num_filters, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        outs = [F.max_pool1d(F.relu(c(emb)), kernel_size=c(emb).shape[2]).squeeze(2)
                for c in self.convs]
        cat = self.dropout(torch.cat(outs, dim=1))
        return self.fc(cat)

model = TextCNN(
    vocab_size=len(vocab), embed_dim=32, num_classes=2,
    kernel_sizes=[2, 3, 4], num_filters=16,
)
print("=== TextCNN 模型结构 ===")
print(model)
print(f"参数量: {sum(p.numel() for p in model.parameters())}")

# ===== 5. 训练 =====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

print("\n=== 训练过程 ===")
for epoch in range(30):
    total_loss, correct, total = 0, 0, 0
    for bx, by in loader:
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(by)
        correct += (logits.argmax(1) == by).sum().item()
        total += len(by)
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:3d} | Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f}")

# ===== 6. 推理测试 =====
model.eval()
test_texts = [
    "电影 好看 推荐",
    "剧情 无聊 浪费 时间",
    "演员 演技 精彩",
    "画面 粗糙 差",
]
print("\n=== 推理测试 ===")
with torch.no_grad():
    for t in test_texts:
        x = torch.tensor([text_to_seq(t)])
        prob = torch.softmax(model(x), dim=-1)
        pred = prob.argmax(-1).item()
        label = '正面 😊' if pred == 1 else '负面 😞'
        print(f"  '{t}' → {label} ({prob[0][pred]:.2%})")
```

### 5.2 超参数调优建议

```
TextCNN 关键超参数:

┌───────────────┬──────────────────────────────────────┐
│    超参数     │              建议                    │
├───────────────┼──────────────────────────────────────┤
│ embed_dim     │ 50 / 100 / 200 / 300                │
│ kernel_sizes  │ [2,3,4] / [3,4,5] / [2,3,4,5]       │
│ num_filters   │ 50~200（每种尺寸）                  │
│ dropout       │ 0.3 ~ 0.5                           │
│ 学习率        │ 0.001 ~ 0.01（Adam）                │
│ max_len       │ 略大于训练集平均句长                 │
│ 激活函数      │ ReLU（默认）                         │
└───────────────┴──────────────────────────────────────┘

经验:
  - kernel_sizes 覆盖常见 n-gram 长度（2~5）
  - num_filters 不宜过大（易过拟合）
  - 文本短 → 小 kernel；文本长 → 可加大 kernel
```

---

## 六、进阶：CNN 与 RNN 的结合

TextCNN 擅长局部特征，LSTM 擅长长距离依赖，二者可以组合使用。

```
组合架构示例（C-LSTM）:

  输入
    │
    ▼
  Embedding
    │
    ▼
  多尺度 CNN（提取局部 n-gram）
    │
    ▼
  LSTM（在 CNN 输出序列上建模长距离）
    │
    ▼
  分类层

适用场景: 句子较长、既有局部关键短语又有跨子句依赖的复杂任务
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    """CNN 提取局部特征 → LSTM 建模序列依赖"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 kernel_sizes=[2, 3], num_filters=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        # LSTM 输入维度 = num_filters × len(kernel_sizes)
        self.lstm = nn.LSTM(
            input_size=num_filters * len(kernel_sizes),
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)  # (B, E, L)
        # 多尺度卷积后拼接，得到 (B, num_filters*K, L')
        conv_outs = [F.relu(c(emb)) for c in self.convs]
        merged = torch.cat(conv_outs, dim=1)  # (B, num_filters*K, L')
        # 转成 (B, L', num_filters*K) 喂给 LSTM
        seq = merged.transpose(1, 2)
        _, (h_n, _) = self.lstm(seq)
        last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(last)

# 简单验证
model = CNN_LSTM(1000, 50, 64, 2)
x = torch.randint(0, 1000, (4, 10))
print(f"CNN-LSTM 输入 {x.shape} → 输出 {model(x).shape}")
```

---

## 小结

| 要点 | 内容 |
|------|------|
| 文本卷积 | 一维卷积（Conv1d），在 seq_len 方向滑动，核宽=embed_dim |
| TextCNN 核心 | 多尺度卷积核 + 全局最大池化，提取不同 n-gram |
| 多尺度核 | kernel_sizes=[2,3,4] 覆盖 bigram/trigram/4-gram |
| 全局最大池化 | 每个通道取最大值，获取最重要特征并转成定长 |
| vs LSTM | CNN 并行快、局部强；RNN 序列强、长距离优 |
| 适用任务 | 文本分类（情感、主题）、短文本匹配 |
| 超参数 | embed_dim, kernel_sizes, num_filters, dropout |
| 典型优势 | 训练速度比 LSTM 快数倍，分类性能接近或更优 |
| 进阶方向 | C-LSTM（CNN+LSTM 组合）、多层 CNN、空洞卷积 |

---

| [← 回到目录](../README.md) | [上一篇：RNN-LSTM-GRU](03-RNN-LSTM-GRU.md) | [下一篇：注意力机制](05-注意力机制.md) |
|---|---|---|
