# Transformer 架构

2017 年，Google 团队在论文 **"Attention is All You Need"** 中提出了 Transformer，这是一种完全基于注意力机制的序列模型，**彻底摒弃了 RNN 和 CNN**。Transformer 不仅在机器翻译任务上取得了 SOTA（State-of-the-Art）性能，更重要的是，它奠定了后续 BERT、GPT、Llama 等所有现代大语言模型的基础架构。可以说，Transformer 是现代深度学习最重要的里程碑之一。

---

## 一、论文核心思想

### 1.1 为什么提出 Transformer

```
RNN 的两大痛点：

  痛点1: 无法并行
  ─────────────────────────────────
  h₁ → h₂ → h₃ → ... → hₙ
  必须先算 h₁ 才能算 h₂，GPU 并行能力无法发挥

  痛点2: 长距离依赖衰减
  ─────────────────────────────────
  句子: "The animal didn't cross the street because it was too tired"
                                                    ↑
                                            "it" 指代谁?
  RNN 从 "animal" 到 "it" 路径太长，信息逐步丢失

Transformer 的解决方案:
  用 Self-Attention 替代循环结构，任意两词之间直接交互，距离为 O(1)
```

### 1.2 核心创新点

```
Transformer 的四大创新:

  ① 完全基于注意力机制 —— 没有任何循环 / 卷积结构
  ② Multi-Head Attention —— 多头并行，从不同子空间关注不同关系
  ③ Positional Encoding   —— 用正余弦编码注入位置信息
  ④ Encoder-Decoder 架构  —— 编码器理解源语言，解码器生成目标语言
```

---

## 二、整体架构

### 2.1 完整架构图

```
                    Transformer 完整架构

   ┌─────────────────────────┐    ┌──────────────────────────────┐
   │        ENCODER          │    │          DECODER             │
   │       (编码器 × N)      │    │         (解码器 × N)         │
   │                         │    │                              │
   │   输入嵌入               │    │   输出嵌入（已生成的词）       │
   │   + 位置编码             │    │   + 位置编码                  │
   │        ↓                │    │        ↓                     │
   │   ┌────────────┐        │    │   ┌────────────┐             │
   │   │  Multi-Head│        │    │   │  Masked    │ ← 只看左侧   │
   │   │ Attention  │        │    │   │  Multi-Head│   已生成词   │
   │   └─────┬──────┘        │    │   │  Attention │             │
   │         │    + ←残差    │    │   └─────┬──────┘             │
   │       LayerNorm         │    │         │    + ←残差          │
   │         ↓               │    │       LayerNorm               │
   │   ┌────────────┐        │    │         ↓                     │
   │   │ Feed-Forward│       │    │   ┌──────────────────┐        │
   │   │ Network    │        │    │   │ Cross Attention  │        │
   │   └─────┬──────┘        │    │   │ (Q来自解码器,    │        │
   │         │    + ←残差    │    │   │  K,V来自编码器)  │←───────┼─ 编码器
   │       LayerNorm         │    │   └─────┬────────────┘        │  输出
   │         ↓               │    │         │    + ←残差          │
   │   编码器输出 ────────────┼────┼─→→→→→→ LayerNorm            │
   │                         │    │         ↓                     │
   │                         │    │   ┌────────────┐              │
   │                         │    │   │ Feed-Forward│             │
   │                         │    │   │ Network    │              │
   │                         │    │   └─────┬──────┘              │
   │                         │    │         │    + ←残差          │
   │                         │    │       LayerNorm               │
   │                         │    │         ↓                     │
   │                         │    │    Linear → Softmax           │
   │                         │    │         ↓                     │
   │                         │    │    输出概率分布                │
   └─────────────────────────┘    └──────────────────────────────┘
```

### 2.2 数据流向

```
以英译中为例: "I love NLP" → "我 爱 自然语言处理"

  源语言输入:  [I, love, NLP]
       ↓
  [Encoder × 6]  理解英文句子语义
       ↓
  编码器输出:  含有全局语义的向量序列
       ↓
  Decoder 输入: [<START>, 我, 爱]  (teacher forcing 训练)
       ↓
  [Decoder × 6]  结合已生成词 + 编码器输出，预测下一个词
       ↓
  输出: 自然 → 语言处理 → <END>
```

---

## 三、编码器组件详解

### 3.1 Multi-Head Attention

```
Multi-Head Attention（多头注意力）:

  将 d_model=512 维向量拆成 h=8 个头，每个头 d_k=64 维

  输入:  X  (seq_len, 512)
         ↓
    ┌────────────────────────────────────────────┐
    │  分成 8 个头 (每个 64 维)                  │
    │                                            │
    │   head₁    head₂        head₇    head₈   │
    │   ┌───┐    ┌───┐        ┌───┐    ┌───┐   │
    │   │Q₁K₁V₁│  │Q₂K₂V₂│ ...│Q₇K₇V₇│ │Q₈K₈V₈││
    │   │att₁│    │att₂│      │att₇│   │att₈│   │
    │   └─┬─┘    └─┬─┘        └─┬─┘    └─┬─┘   │
    │     └────────┴────Concat──┴────────┘      │
    │                    ↓                      │
    │              Linear (W^O)                 │
    │                    ↓                      │
    └────────────────────────────────────────────┘
                    输出 (seq_len, 512)

  直觉: 不同头关注不同关系
    - 头1: 语法关系 (主谓)
    - 头2: 语义关系 (同义/反义)
    - 头3: 指代消解 (it → animal)
    - 头4: 长距离依赖
    - ...
```

### 3.2 Feed-Forward Network (FFN)

```
Position-wise Feed-Forward Network:

  对每个位置独立施加两层全连接 + ReLU 激活:

    FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

  维度变化:
              W₁              W₂
    (512) ──────→ (2048) ──────→ (512)
     d_model     4×d_model     d_model

  特点:
    - 每个位置独立计算（Position-wise）
    - 中间层扩展 4 倍，增加非线性表达能力
    - 不同位置共享相同参数
```

### 3.3 残差连接 + LayerNorm

```
残差连接 + 层归一化（Add & Norm）:

  每个子层都经过:
    output = LayerNorm(x + Sublayer(x))

  作用:
  ┌────────────────────────────────────────────────┐
  │                                                │
  │   x ──────┬──────────────┐                     │
  │           │              │                     │
  │           ↓              │                     │
  │      [ Sublayer ]        │  残差连接            │
  │           │              │  避免梯度消失        │
  │           ↓              │  允许训练深层网络    │
  │         output          │                     │
  │           │              │                     │
  │           └────── + ←───┘                     │
  │                  ↓                            │
  │             LayerNorm                         │
  │             (对每个样本的特征维度归一化)        │
  │                  ↓                            │
  │               最终输出                         │
  │                                                │
  └────────────────────────────────────────────────┘
```

---

## ��、解码器额外组件

### 4.1 Masked Multi-Head Attention

```
Masked Attention（掩码注意力）:

  解码器生成第 t 个词时，只能看到前 t-1 个词，不能"偷看"未来

  ┌────────────────────────────────────┐
  │     t=1   t=2   t=3   t=4         │
  │                                    │
  │ t=1  ✓                             │
  │ t=2  ✓    ✓                        │
  │ t=3  ✓    ✓     ✓                  │
  │ t=4  ✓    ✓     ✓     ✓            │
  │                                    │
  │ 上三角部分被掩码为 -∞              │
  │ softmax 后变为 0                   │
  └────────────────────────────────────┘

  示例: 生成 "我 爱 自然 语言"
    生成 "自然" 时，只能看到 "我 爱"，看不到 "语言"
```

### 4.2 Cross Attention（交叉注意力）

```
Encoder-Decoder Cross Attention:

  ┌──────────────────────────────────────────────┐
  │                                              │
  │  Query  来自解码器上一层的输出    "要生成什么"│
  │  Key    来自编码器的输出          "源语言信息"│
  │  Value  来自编码器的输出          "源语言内容"│
  │                                              │
  │  作用: 解码器"查看"编码器输出的哪些部分       │
  │                                              │
  │  示例: 翻译 "I love NLP" → "我爱NLP"         │
  │    生成"爱"时，decoder 对 encoder 的 "love"   │
  │    给予最高注意力权重                         │
  │                                              │
  └──────────────────────────────────────────────┘
```

---

## 五、为什么 Transformer 取代 RNN

### 5.1 并行化

```
并行化对比:

  RNN: 必须串行处理
    h₁ → h₂ → h₃ → ... → hₙ
    总时间: O(n) 串行步骤

  Transformer: 可完全并行
    h₁
    h₂    ← 同时计算
    h₃
    ...
    hₙ
    总时间: O(1) 串行步骤（矩阵乘法）

  在 GPU 上，Transformer 训练速度比 RNN 快数倍到数十倍
```

### 5.2 长距离依赖

```
长距离依赖对比:

  句子: "The cat, which already ate fish, was full."

  RNN 路径:
    cat ──→ (which) ──→ (ate) ──→ (fish) ──→ (was) ──→ full
    距离 O(n)，信息逐步衰减

  Transformer 路径:
    cat ←──────direct attention──────→ full
    距离 O(1)，信息无损传递

  这也是 Transformer 在长文本任务上表现远超 RNN 的核心原因
```

### 5.3 综合对比

```
┌──────────────┬─────────────────┬──────────────────┐
│     特性      │      RNN        │   Transformer    │
├──────────────┼─────────────────┼──────────────────┤
│   并行化      │     差           │     好           │
│  长距离依赖   │    衰减          │    无损 O(1)     │
│   计算复杂度  │  O(n·d²)        │  O(n²·d)        │
│   显存占用    │     低           │     较高         │
│   可解释性    │     低           │     中           │
└──────────────┴─────────────────┴──────────────────┘

  注意: Transformer 复杂度 O(n²·d)，序列长时显存压力大
        这是后来 Flash Attention 等优化的动机
```

---

## 六、PyTorch 实现

### 6.1 完整 Transformer 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力层"""
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V 的线性映射
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性映射 + 分头: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头: (batch, heads, seq, d_k) -> (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        return self.W_o(attn_output)


class FeedForwardNetwork(nn.Module):
    """前馈网络"""
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 子层1: Multi-Head Self-Attention + Add & Norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        # 子层2: FFN + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 子层1: Masked Self-Attention
        attn1 = self.masked_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        # 子层2: Cross Attention（Q来自解码器，K/V来自编码器）
        attn2 = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        # 子层3: FFN
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x


class Transformer(nn.Module):
    """简化版完整 Transformer"""
    def __init__(self, src_vocab_size=10000, tgt_vocab_size=10000,
                 d_model=512, num_heads=8, num_layers=6, d_ff=2048,
                 max_len=512, dropout=0.1):
        super().__init__()
        # 词嵌入 + 位置编码
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # 编码器和解码器堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    @staticmethod
    def create_positional_encoding(max_len, d_model):
        """正弦余弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def encode(self, src, src_mask=None):
        """编码器前向传播"""
        x = self.src_embedding(src) + self.positional_encoding[:, :src.size(1)]
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        """解码器前向传播"""
        x = self.tgt_embedding(tgt) + self.positional_encoding[:, :tgt.size(1)]
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.fc_out(dec_output)


# ========== 测试运行 ==========
if __name__ == "__main__":
    # 超参数
    batch_size = 2
    src_len = 10
    tgt_len = 8
    src_vocab = 1000
    tgt_vocab = 1000

    # 创建模型
    model = Transformer(src_vocab_size=src_vocab, tgt_vocab_size=tgt_vocab,
                        d_model=512, num_heads=8, num_layers=6)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟输入
    src = torch.randint(0, src_vocab, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))

    # 前向传播
    output = model(src, tgt)
    print(f"输入形状:   src={src.shape}, tgt={tgt.shape}")
    print(f"输出形状:   {output.shape}")  # (batch, tgt_len, tgt_vocab)
    print(f"前向传播成功!")
```

### 6.2 生成掩码

```python
def create_masks(src, tgt, pad_idx=0):
    """创建源语言和目标语言的掩码"""
    # 源语言 padding mask
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # 目标语言 padding mask
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)

    # 目标语言因果 mask（下三角）
    tgt_len = tgt.size(1)
    tgt_causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()
    tgt_mask = tgt_pad_mask & tgt_causal_mask

    return src_mask, tgt_mask


# 测试 mask 生成
if __name__ == "__main__":
    src = torch.randint(1, 1000, (2, 6))
    tgt = torch.randint(1, 1000, (2, 5))
    src_mask, tgt_mask = create_masks(src, tgt)
    print(f"源 mask 形状: {src_mask.shape}")
    print(f"目标 mask 形状: {tgt_mask.shape}")
    print(f"因果 mask (下三角):\n{tgt_mask[0, 0, :, :]}")
```

---

## 七、使用 PyTorch 内置 Transformer

```python
import torch
import torch.nn as nn

# PyTorch 内置 Transformer
transformer = nn.Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True  # PyTorch 1.9+ 支持 batch 优先
)

# 模拟输入
batch_size = 4
src_len = 20
tgt_len = 15
d_model = 512

src = torch.randn(batch_size, src_len, d_model)
tgt = torch.randn(batch_size, tgt_len, d_model)

# 生成因果 mask
tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len)

# 前向传播
output = transformer(src, tgt, tgt_mask=tgt_mask)
print(f"输出形状: {output.shape}")  # (4, 15, 512)
```

---

## 小结

| 要点 | 内容 |
|------|------|
| 论文 | "Attention is All You Need" (2017, Google) |
| 核心创新 | 完全基于注意力机制，摒弃 RNN/CNN |
| 架构 | Encoder-Decoder 结构，各 N=6 层 |
| 编码器组件 | Multi-Head Attention + FFN + Add & Norm |
| 解码器额外组件 | Masked Attention（防偷看）+ Cross Attention（看编码器） |
| 并行化 | Self-Attention 矩阵运算，完全可并行 |
| 长距离依赖 | 任意两词交互距离 O(1) |
| 缩放因子 | 1/√d_k 防止 softmax 进入饱和区 |
| Multi-Head | 多头并行，不同子空间关注不同关系 |
| 位置编码 | 弥补 Self-Attention 无位置感知的缺陷 |
| 复杂度 | O(n²·d)，长序列显存压力大 |
| 影响 | BERT、GPT、T5、Llama 等所有现代大模型的基础 |

---

| [← 回到目录](../README.md) | [上一篇：注意力机制](../06-深度学习基础篇/05-注意力机制.md) | [下一篇：位置编码](02-位置编码.md) |
|---|---|---|
