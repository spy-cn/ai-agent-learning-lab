# RNN-LSTM-GRU

前面的章节我们用"词袋平均"做文本分类，但这种做法**完全忽略了词序**——"狗咬人"和"人咬狗"在词袋模型里是等价的。自然语言本质上是**有序序列**，理解一个词的含义往往需要看它的上下文。**循环神经网络（RNN）** 正是为处理序列数据而生，它通过"记忆"将历史信息传递到当前时刻。本篇将系统讲解 RNN、LSTM、GRU 的原理与实现。

---

## 一、为什么需要处理序列

### 1.1 词序的重要性

```
句子1: "我 爱 你"      → 正面情感
句子2: "你 爱 我"      → 正面情感
句子3: "我 不 爱 你"   → 负面情感
句子4: "你 不 爱 我"   → 负面情感

词袋模型（Bag-of-Words）视角:
  所有句子都由 {我, 爱, 你, 不} 构成
  句子1 = 句子2（顺序被丢弃）
  句子3 = 句子4（顺序被丢弃）
  → 无法区分"我爱你"和"你不爱我"

序列模型（RNN）视角:
  按顺序读入每个词，根据"前面读过什么"来理解当前词
  - "我 不 爱 你" 中，"不" 改变了后续"爱"的含义
  - 序列位置是核心信息
```

### 1.2 序列数据的多样性

```
NLP 中的序列任务:

  ┌────────────────┬──────────────────────────────┐
  │     任务       │          示例                │
  ├────────────────┼──────────────────────────────┤
  │ 情感分类       │ [这, 电影, 好, 看] → 正面     │
  │ 词性标注       │ [我, 爱, NLP] → [代, 动, 名]  │
  │ 机器翻译       │ [I, love, NLP] → [我, 爱,NLP]│
  │ 语言模型       │ [今天, 天气] → 预测下一个词   │
  │ 命名实体识别   │ [张三, 在, 北京] → [人, 0, 地]│
  └────────────────┴──────────────────────────────┘

共同特点：当前输出依赖于之前看过的内容 → 需要"记忆"
```

---

## 二、RNN 原理

### 2.1 RNN 的核心思想

RNN 通过一个**隐状态（hidden state）** 在时间步之间传递信息，相当于网络的"记忆"。

```
RNN 展开图（按时间步）:

  时刻 t=1      t=2        t=3        t=4
                ┌───┐ h₁  ┌───┐ h₂  ┌───┐ h₃
                │RNN│◄────│RNN│◄────│RNN│◄──── ...
            ┌───┴───┴──┐ ┌┴───┴───┐ ┌┴───┴───┐
   输入 x₁ ─┤  concat  │ │ concat │ │ concat │  ← 当前输入 x_t
            └──────────┘ └────────┘ └────────┘
                │            │           │
                ▼            ▼           ▼
               o₁           o₂          o₃        ← 各时刻输出

  展开理解：同一个 RNN 单元在不同时刻被复用（权重共享）
  - h_t = f(W_xh · x_t + W_hh · h_{t-1} + b)
  - o_t = g(W_ho · h_t + b_o)
```

### 2.2 RNN 计算公式

```
对于时刻 t:
  ┌─────────────────────────────────────────────┐
  │  h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b) │  隐状态更新
  │  o_t = softmax(W_ho · h_t + b_o)              │  输出
  └─────────────────────────────────────────────┘

  其中:
  - x_t: 时刻 t 的输入向量 (embed_dim,)
  - h_{t-1}: 上一时刻的隐状态 (hidden_dim,)
  - h_t: 当前隐状态 (hidden_dim,)
  - W_xh: 输入到隐状态的权重 (hidden_dim, embed_dim)
  - W_hh: 隐状态到隐状态的权重 (hidden_dim, hidden_dim)
  - tanh: 双曲正切激活函数，输出范围 [-1, 1]

初始隐状态 h_0 通常为零向量
```

### 2.3 RNN 的梯度问题

```
反向传播通过时间（BPTT）时，梯度需要连乘 W_hh 多次:

  ∂L/∂h_0 = ∂L/∂h_T · W_hh^T · W_hh^T · ... · W_hh^T  (连乘 T 次)

  ┌──────────────────────────────────────────────────────┐
  │  问题1: 梯度消失 (Vanishing Gradient)                │
  │  - 当 W_hh 的特征值 < 1 时，连乘导致梯度趋近于 0     │
  │  - 后果：网络无法学习长距离依赖                       │
  │  - 类比：传话游戏，信息经过多人传递后失真             │
  ├──────────────────────────────────────────────────────┤
  │  问题2: 梯度爆炸 (Exploding Gradient)                │
  │  - 当 W_hh 的特征值 > 1 时，连乘导致梯度发散到无穷   │
  │  - 后果：参数更新过大，训练不稳定                     │
  └──────────────────────────────────────────────────────┘

  梯度随时间步的变化（示意）:
  梯度
   │ ─── ─── ─── ─── ─── ───  (爆炸)
   │
   │ ────────────────────     (理想)
   │
   │ ●●●                     (消失，趋近0)
   └─────────────────────────→ 时间步 T
```

### 2.4 简单 RNN 代码实现

```python
import torch
import torch.nn as nn

class SimpleRNNCell(nn.Module):
    """手动实现一个 RNN 单元，帮助理解原理"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_xh = nn.Linear(input_dim, hidden_dim)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, h_prev):
        # x: (batch, input_dim)
        # h_prev: (batch, hidden_dim)
        h = torch.tanh(self.W_xh(x) + self.W_hh(h_prev))
        return h

# 手动按时间步循环
batch_size, input_dim, hidden_dim, seq_len = 2, 4, 8, 5
rnn_cell = SimpleRNNCell(input_dim, hidden_dim)

# 模拟输入序列 (batch, seq_len, input_dim)
x_seq = torch.randn(batch_size, seq_len, input_dim)
h = torch.zeros(batch_size, hidden_dim)  # 初始隐状态

print("=== 手动 RNN 前向传播 ===")
for t in range(seq_len):
    h = rnn_cell(x_seq[:, t, :], h)  # 取第 t 个时间步
    print(f"  t={t}: 隐状态均值={h.mean():.4f}, 标准差={h.std():.4f}")
```

---

## 三、LSTM 门控机制详解

为了解决 RNN 的梯度问题，**长短期记忆网络（LSTM）** 引入了**门控机制**，能选择性地记忆、遗忘信息。

### 3.1 LSTM 结构

```
LSTM 单元内部结构:

  ┌─────────────────────────────────────────────────────┐
  │                    LSTM Cell                         │
  │                                                      │
  │   h_{t-1} ──┐    ┌─── 遗忘门 f ──┐                 │
  │             │    │                │                 │
  │   x_t ──────┼────┼─── 输入门 i ──┼──┐              │
  │             │    │                │  │              │
  │             │    └─── 候选值 g ──┘  │              │
  │             │                       ▼               │
  │   C_{t-1} ──┼─────×──── + ──── ● ──┬──→ C_t         │
  │  (旧记忆)   │     ↑          ↑     │  (新记忆)      │
  │             │     │          │     │                 │
  │             │     └── ── ── ┘     │                 │
  │             │           ↓          │                 │
  │             │     ┌─── 输出门 o ──┐│                 │
  │             │     │      ↓        ││                 │
  │             │     │      × ───────┘└──→ h_t          │
  │             │     │      ↓                           │
  │             │     └─ tanh(C_t)                       │
  └─────────────┴───────────────────────────────────────┘

  两条核心信息流:
  - C_t (Cell State): 贯穿全程的"长期记忆"主线
  - h_t (Hidden State): 当前时刻的"短期记忆"输出
```

### 3.2 三个门的作用

```
┌─────────────────────────────────────────────────────────┐
│  遗忘门 (Forget Gate) f_t = σ(W_f · [h_{t-1}, x_t])    │
│  ────────────────────────��────────────────────         │
│  作用: 决定从旧记忆 C_{t-1} 中丢弃什么                  │
│  输出: 0~1 之间的值，0=完全遗忘，1=完全保留             │
│  类比: 删除不再相关的旧信息                             │
├─────────────────────────────────────────────────────────┤
│  输入门 (Input Gate) i_t = σ(W_i · [h_{t-1}, x_t])     │
│  ─────────────────────────────────────────────         │
│  作用: 决定哪些新信息需要写入记忆                        │
│  输出: 0~1 之间的值，控制新信息的写入量                  │
│  类比: 选择性地记录重要新信息                            │
├─────────────────────────────────────────────────────────┤
│  候选值 (Candidate) g_t = tanh(W_g · [h_{t-1}, x_t])   │
│  ─────────────────────────────────────────────         │
│  作用: 生成新的候选记忆内容                              │
│  输出: -1~1 之间的值                                    │
├─────────────────────────────────────────────────────────┤
│  输出门 (Output Gate) o_t = σ(W_o · [h_{t-1}, x_t])    │
│  ─────────────────────────────────────────────         │
│  作用: 决定从记忆 C_t 中输出什么作为 h_t                │
│  输出: 0~1 之间的值，控制记忆的输出量                    │
│  类比: 决定当前时刻"说"什么                             │
└─────────────────────────────────────────────────────────┘

记忆更新公式:
  C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t   (旧记忆过滤 + 新记忆加入)
  h_t = o_t ⊙ tanh(C_t)              (输出门控制)

σ 为 sigmoid 函数，⊙ 为逐元素乘
```

### 3.3 LSTM 缓解梯度消失的原因

```
LSTM 的细胞状态 C_t 的更新是"加法"而非"乘法":

  C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t
         ↑                ↑
         乘法（但 f_t 可学习）  加法（不连乘）

梯度反向传播时:
  ∂C_t/∂C_{t-1} = f_t  (一个标量门，而非矩阵连乘)

当 f_t ≈ 1 时，梯度可以无损传递 → 缓解梯度消失
LSTM 可以通过学习让遗忘门在长距离依赖时保持接近 1
```

---

## 四、GRU 简化版

**门控循环单元（GRU）** 是 LSTM 的简化版本，参数更少、训练更快，效果通常接近。

### 4.1 GRU 结构

```
GRU 单元内部结构:

  ┌────────────────────────────────────┐
  │            GRU Cell                 │
  │                                     │
  │   h_{t-1} ┌─── 重置门 r ───┐      │
  │           │                 │      │
  │   x_t ────┼─── 更新门 z ───┤      │
  │           │                 │      │
  │           │     r ⊙ h_{t-1} │      │
  │           │          ↓      │      │
  │           │   候选 h̃_t      │      │
  │           │          ↓      │      │
  │           └── (1-z) ⊙ h̃ + z ⊙ h_{t-1} ──→ h_t
  │                                     │
  └─────────────────────────────────────┘

  比 LSTM 少了:
  - 独立的细胞状态 C_t（只用 h_t）
  - 输出门（更新门同时承担输入/遗忘功能）
```

### 4.2 GRU 的两个门

```
重置门 (Reset Gate) r_t = σ(W_r · [h_{t-1}, x_t])
  作用: 控制如何将新输入与历史记忆结合
  r_t≈0: 完全忽略历史，相当于重置记忆
  r_t≈1: 保留历史信息

更新门 (Update Gate) z_t = σ(W_z · [h_{t-1}, x_t])
  作用: 控制历史信息的保留比例（兼并了LSTM的遗忘门和输入门）
  z_t≈1: 保留旧状态 h_{t-1}（类似LSTM的遗忘门=1）
  z_t≈0: 采用新候选 h̃_t

候选隐状态:
  h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])

最终更新:
  h_t = (1 - z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}
```

### 4.3 LSTM vs GRU 对比

```
┌──────────────┬─────────────────────┬─────────────────────┐
│     维度     │       LSTM          │        GRU          │
├──────────────┼─────────────────────┼─────────────────────┤
│ 门数量       │ 3个（遗忘、输入、输出）│ 2个（重置、更新）  │
│ 状态         │ h_t 和 C_t          │ 只有 h_t            │
│ 参数量       │ 多（4组权重）       │ 少（3组权重）       │
│ 训练速度     │ 较慢                │ 较快                │
│ 长距离依赖   │ 强                  │ 略弱但接近          │
│ 适用场景     │ 复杂序列、大语料    │ 中小数据、实时任务  │
│ 调参难度     │ 选择多              │ 选择少，易上手      │
└──────────────┴─────────────────────┴─────────────────────┘

经验法则：先用 GRU（快），效果不够再换 LSTM
```

---

## 五、PyTorch 实现 nn.LSTM / nn.GRU

### 5.1 nn.LSTM 基本 API

```python
import torch
import torch.nn as nn

# ===== 创建 LSTM =====
lstm = nn.LSTM(
    input_size=10,     # 输入特征维度（通常是 embed_dim）
    hidden_size=20,    # 隐状态维度
    num_layers=1,      # RNN 层数
    batch_first=True,  # 输入形状为 (batch, seq, feature)
    bidirectional=False,
)

# ===== 前向传播 =====
batch_size, seq_len, input_size = 3, 5, 10
x = torch.randn(batch_size, seq_len, input_size)

# 初始隐状态和细胞状态（可选，默认为零）
h0 = torch.zeros(1, batch_size, 20)  # (num_layers, batch, hidden)
c0 = torch.zeros(1, batch_size, 20)

# 输出: output 所有时刻的隐状态, (h_n, c_n) 最后时刻的隐状态和细胞状态
output, (h_n, c_n) = lstm(x, (h0, c0))

print(f"输入形状: {x.shape}")
print(f"output 形状: {output.shape}")  # (3, 5, 20) 所有时刻
print(f"h_n 形状: {h_n.shape}")       # (1, 3, 20) 最后时刻隐状态
print(f"c_n 形状: {c_n.shape}")       # (1, 3, 20) 最后时刻细胞状态

# 重要：output[:, -1, :] 等于 h_n[-1]（单向时）
print(f"\noutput最后时刻 == h_n: {torch.allclose(output[:, -1, :], h_n[-1])}")
```

### 5.2 nn.GRU 使用

```python
import torch
import torch.nn as nn

gru = nn.GRU(
    input_size=10,
    hidden_size=20,
    num_layers=2,      # 2层 GRU
    batch_first=True,
    bidirectional=True, # 双向
)

x = torch.randn(3, 5, 10)
output, h_n = gru(x)  # GRU 没有 c_n

print(f"2层双向 GRU:")
print(f"  输入: {x.shape}")
print(f"  output: {output.shape}")  # (3, 5, 40) ← hidden_size*2（双向）
print(f"  h_n: {h_n.shape}")       # (4, 3, 20) ← num_layers*2（双向）
```

### 5.3 处理变长序列（pack_padded_sequence）

```python
import torch
import torch.nn as nn

# 对于 padding 过的序列，用 packing 提高效率并避免 padding 影响
lstm = nn.LSTM(input_size=4, hidden_size=8, batch_first=True)

# 假设 batch 内序列真实长度不同
seqs = torch.tensor([
    [1, 2, 3, 4, 5],   # 长度5
    [1, 2, 3, 0, 0],   # 长度3（后两个是 padding）
    [1, 2, 0, 0, 0],   # 长度2
], dtype=torch.float).unsqueeze(-1).repeat(1, 1, 4)  # (3, 5, 4)
lengths = torch.tensor([5, 3, 2])

# ===== Packing =====
packed = nn.utils.rnn.pack_padded_sequence(
    seqs, lengths, batch_first=True, enforce_sorted=False
)
print(f"Packed 对象: {type(packed)}")

# LSTM 处理 packed 数据
packed_output, (h_n, c_n) = lstm(packed)
print(f"h_n 形状: {h_n.shape}")  # (1, 3, 8)

# ===== Unpacking =====
output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
print(f"output 形状: {output.shape}")  # (3, 5, 8)
```

---

## 六、双向 RNN 和多层 RNN

### 6.1 双向 RNN（BiRNN）

```
单向 RNN 只能利用"过去"的上下文，双向 RNN 同时利用"过去"和"未来":

  正向 →   →   →   →   →
  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
  │ → │ │ → │ │ → │ │ → │ │ → │
  └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
    │     │     │     │     │
    ●─────●─────●─────●─────●   ← 正向隐状态 h→
    ●─────●─────●─────●─────●   ← 反向隐状态 h←
    │     │     │     │     │
  └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
  │ ← │ │ ← │ │ ← │ │ ← │ │ ← │
  └───┘ └───┘ └───┘ └───┘ └───┘
  反向 ←   ←   ←   ←   ←

  每个时刻的最终表示 = [h→_t ; h←_t]  (正向与反向拼接)

  优势: 上下文信息更完整（例如理解"苹果"需要看后文是"公司"还是"水果"）
  注意: 不能用于实时流式任务（需要未来信息）
```

### 6.2 多层 RNN（Stacked RNN）

```
多层 RNN（这里以2层为例）:

  Layer 2:  h₂₁ ← h₂₂ ← h₂₃ ← h₂₄   (高层特征)
              ↑      ↑      ↑      ↑
  Layer 1:  h₁₁ ← h₁₂ ← h₁₃ ← h₁₄   (低层特征)
              ↑      ↑      ↑      ↑
   输入:     x₁    x₂    x₂    x₄

  - 第1层接收原始词嵌入，输出隐状态序列
  - 第2层接收第1层的隐状态序列作为输入
  - 层数越多，表达能力越强，但也更难训练（通常1~3层）

经验: num_layers 一般取 1、2 或 3，超过4层效果往往不再提升
```

### 6.3 代码示例

```python
import torch
import torch.nn as nn

# 双向 + 多层 LSTM
lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,         # 2层
    bidirectional=True,   # 双向
    batch_first=True,
)

x = torch.randn(3, 5, 10)
output, (h_n, c_n) = lstm(x)
print(f"双向2层 LSTM:")
print(f"  output 形状: {output.shape}")  # (3, 5, 40) ← hidden_size*2
print(f"  h_n 形状: {h_n.shape}")       # (4, 3, 20) ← num_layers*2

# 提取最后一层的正向和反向隐状态做分类
# h_n 形状 (4, batch, hidden): [layer1_fwd, layer1_bwd, layer2_fwd, layer2_bwd]
last_layer = torch.cat([h_n[-2], h_n[-1]], dim=1)  # 拼接正向和反向
print(f"  最后层拼接: {last_layer.shape}")  # (3, 40)
```

---

## 七、文本分类 LSTM 完整代码

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# ===== 1. 数据准备 =====
texts = [
    "这部 电影 真的 太 好看 了", "演员 演技 棒 剧情 紧凑",
    "画面 精美 值得 推荐", "配乐 动人 故事 感人",
    "这部 电影 真的 太 烂 了", "剧情 差 演员 演技 烂",
    "浪费 时间 无聊 透顶", "不 推荐 太 差 了",
] * 15
labels = [1, 1, 1, 1, 0, 0, 0, 0] * 15

# ===== 2. 词表 =====
counter = Counter()
for t in texts:
    counter.update(t.split())
vocab = {'<pad>': 0, '<unk>': 1}
for w, _ in counter.most_common():
    vocab[w] = len(vocab)

max_len = 10
def text_to_seq(text):
    idx = [vocab.get(w, 1) for w in text.split()][:max_len]
    return idx + [0] * (max_len - len(idx))

# ===== 3. Dataset =====
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.x = torch.tensor([text_to_seq(t) for t in texts])
        self.y = torch.tensor(labels)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

loader = DataLoader(TextDataset(texts, labels), batch_size=16, shuffle=True)

# ===== 4. LSTM 文本分类模型 =====
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * num_directions, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)               # (batch, seq_len, embed_dim)
        output, (h_n, c_n) = self.lstm(emb)   # output: (batch, seq_len, hidden*dirs)
        # 取最后一个时刻的输出（或用 h_n）
        if self.lstm.bidirectional:
            # 拼接最后层正向和反向的最终隐状态
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)         # (batch, num_classes)
        return logits

# ===== 5. 训练 =====
model = LSTMClassifier(
    vocab_size=len(vocab), embed_dim=32, hidden_dim=64,
    num_classes=2, num_layers=2, bidirectional=True,
)
print(model)
print(f"参数量: {sum(p.numel() for p in model.parameters())}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

print("\n=== 训练过程 ===")
for epoch in range(25):
    total_loss, correct, total = 0, 0, 0
    for bx, by in loader:
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(by)
        correct += (logits.argmax(1) == by).sum().item()
        total += len(by)
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:3d} | Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f}")

# ===== 6. 推理 =====
model.eval()
test_texts = ["电影 好看 推荐", "剧情 无聊 烂"]
print("\n=== 推理测试 ===")
with torch.no_grad():
    for t in test_texts:
        x = torch.tensor([text_to_seq(t)])
        prob = torch.softmax(model(x), dim=-1)
        pred = prob.argmax(-1).item()
        label = '正面' if pred == 1 else '负面'
        print(f"  '{t}' → {label} ({prob[0][pred]:.2%})")
```

---

## 小结

| 要点 | 内容 |
|------|------|
| 序列处理动机 | 文本有序，词的含义依赖上下文，需要"记忆"机制 |
| RNN 核心 | 隐状态在时间步间传递，权重共享 |
| RNN 缺陷 | 梯度消失（无法学长距离依赖）/ 梯度爆炸 |
| LSTM 创新 | 细胞状态 + 三个门（遗忘、输入、输出），缓解梯度问题 |
| GRU 简化 | 两个门（重置、更新），参数少、速度快 |
| LSTM vs GRU | 大数据/复杂任务选 LSTM；小数据/快速实验选 GRU |
| 双向 RNN | 同时利用前后文，适合非流式任务 |
| 多层 RNN | 层层堆叠提升表达力，通常 1-3 层 |
| PyTorch API | `nn.LSTM` / `nn.GRU`，注意 `batch_first=True` |
| 梯度裁剪 | `clip_grad_norm_` 防止梯度爆炸 |
| 变长序列 | `pack_padded_sequence` 提高效率并避免 padding 影响 |

---

| [← 回到目录](../README.md) | [上一篇：词嵌入实践](02-词嵌入实践.md) | [下一篇：文本CNN](04-文本CNN.md) |
|---|---|---|
