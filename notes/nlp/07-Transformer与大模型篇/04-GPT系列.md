# GPT 系列

如果说 BERT 开创了"理解"语言的预训练范式，那么 **GPT（Generative Pre-trained Transformer）** 则定义了"生成"语言的新方向。GPT 采用 Transformer 的**解码器**架构，通过**自回归**方式预测下一个词。从 GPT-1 的 117M 参数到 GPT-3 的 1750 亿参数，再到 GPT-4 的多模态能力，GPT 系列不仅展示了**Scaling Law**（规模法则）的威力，更揭示了当模型足够大时会涌现出惊人的**少样本学习**和**推理**能力，直接引爆了当今的大模型时代。

---

## 一、GPT 是什么

### 1.1 核心概念

```
GPT = Generative Pre-trained Transformer

  G - Generative      生成式: 预测下一个词（自回归）
  P - Pre-trained     预训练: 在大规模文本上无监督学习
  T - Transformer     基于 Transformer 解码器架构

  核心思想: 自回归语言建模

  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │  给定已生成的词，预测下一个词                    │
  │                                                 │
  │  "自然 语言 处理 是 人工智能 的 重要 [?]"        │
  │                                          ↓      │
  │                                      预测: "分支"│
  │                                                 │
  │  P(分支|自然语言处理是人工智能的重要)            │
  │                                                 │
  │  生成过程: 一步步预测，每次加一个词              │
  │  "自然" → "自然语言" → "自然语言处理" → ...      │
  │                                                 │
  └─────────────────────────────────────────────────┘
```

### 1.2 自回归语言模型

```
自回归 (Autoregressive) 语言模型:

  训练目标: 最大化序列的似然概率

    P(x₁, x₂, ..., xₙ) = ∏ P(xₜ | x₁, ..., xₜ₋₁)

  即: 把序列概率分解为条件概率的连乘

  示例:
    P("我 爱 自然 语言") = P(我) × P(爱|我) × P(自然|我爱)
                         × P(语言|我爱自然)

  ┌────────────────────────────────────────────────┐
  │  时间步 t=1: 输入 [<START>]          → 预测 "我"│
  │  时间步 t=2: 输入 [<START>, 我]      → 预测 "爱"│
  │  时间步 t=3: 输入 [<START>, 我, 爱]  → 预测 "自然"│
  │  ...                                           │
  └────────────────────────────────────────────────┘

  训练时用 Teacher Forcing:
    每一步输入真实的前序词，不是模型自己生成的
```

---

## 二、GPT vs BERT

### 2.1 架构对比

```
GPT vs BERT 架构对比:

  BERT (Encoder-only):
  ┌─────────────────────┐
  │  双向 Self-Attention │  ← 每个词能看到所有词
  │  (左右上下文)        │
  └─────────────────────┘
  适合: 理解任务 (分类、问答、NER)

  GPT (Decoder-only):
  ┌─────────────────────┐
  │  单向 Causal Atten  │  ← 每个词只能看左侧
  │  (只看左侧已生成词)  │
  └─────────────────────┘
  适合: 生成任务 (对话、写作、翻译)

  详细对比:
  ┌──────────────┬──────────────────┬──────────────────┐
  │              │      BERT        │       GPT        │
  ├──────────────┼──────────────────┼──────────────────┤
  │  架构        │  Encoder-only    │  Decoder-only    │
  │  注意力方向  │  双向            │  单向(从左到右)  │
  │  预训练任务  │  MLM + NSP       │  自回归 LM       │
  │  目标        │  理解            │  生成            │
  │  输出        │  表示向量        │  下一个词概率    │
  │  擅长任务    │  分类、抽取      │  生成、对话      │
  │  [MASK] 标记 │  需要            │  不需要          │
  └──────────────┴──────────────────┴──────────────────┘
```

### 2.2 因果注意力图解

```
GPT 的因果注意力 (Causal Attention):

  句子: "我 爱 自然 语言 处理"

  注意力矩阵 (下三角):
         我   爱   自然  语言  处理
  我    [ ✓    ✗    ✗    ✗    ✗ ]   ← "我" 只能看到自己
  爱    [ ✓    ✓    ✗    ✗    ✗ ]   ← "爱" 能看到 "我"
  自然  [ ✓    ✓    ✓    ✗    ✗ ]   ← "自然" 能看到 "我 爱"
  语言  [ ✓    ✓    ✓    ✓    ✗ ]
  处理  [ ✓    ✓    ✓    ✓    ✓ ]   ← "处理" 能看到所有左侧词

  实现方式: 上三角掩码为 -∞
    [  0   -∞   -∞   -∞   -∞ ]
    [  0    0   -∞   -∞   -∞ ]
    [  0    0    0   -∞   -∞ ]
    [  0    0    0    0   -∞ ]
    [  0    0    0    0    0  ]
  softmax 后上三角变为 0

  BERT 的双向注意力矩阵 (全 True):
         我   爱   自然  语言  处理
  我    [ ✓    ✓    ✓    ✓    ✓ ]   ← 每个词都能看到所有词
  爱    [ ✓    ✓    ✓    ✓    ✓ ]
  ...
```

---

## 三、GPT 发展史

### 3.1 发展时间线

```
GPT 发展历程:

  参数量 →
  10⁸    10⁹     10¹⁰     10¹¹     10¹²

  GPT-1  GPT-2   GPT-3    GPT-4
  117M   1.5B    175B     ~1.7T(估)
   │      │       │        │
  2018   2019   2020     2023
   │      │       │        │
   ↓      ↓       ↓        ↓
  证明   展现   涌现     多模态
  可行   潜力   能力     通用AI

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  GPT-1 (2018.06) "Improving Language Understanding│
  │  by Generative Pre-Training"                     │
  │  - 参数: 117M                                    │
  │  - 层数: 12                                      │
  │  - 证明 "预训练+微调" 范式可行                   │
  │  - 效果不如 BERT                                 │
  │                                                  │
  │  GPT-2 (2019.02) "Language Models are Unsupervised│
  │  Multitask Learners"                             │
  │  - 参数: 1.5B (增大 13 倍)                       │
  │  - 层数: 48                                      │
  │  - Zero-shot 能力初现                            │
  │  - 当时认为太危险，分阶段发布                     │
  │                                                  │
  │  GPT-3 (2020.05) "Language Models are Few-Shot    │
  │  Learners"                                       │
  │  - 参数: 175B (增大 100 倍)                      │
  │  - Few-shot 学习能力                             │
  │  - 展现涌现能力                                  │
  │  - In-context learning                           │
  │                                                  │
  │  GPT-4 (2023.03) 多模态大模型                    │
  │  - 参数: 未公开（估计 ~1.7T）                    │
  │  - 支持图像输入                                  │
  │  - 推理能力大幅提升                              │
  │  - 通过人类反馈强化学习(RLHF)对齐                │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

### 3.2 规模增长

```
GPT 参数量指数增长:

参数量
  │
  │                                    GPT-4 (~1.7T)
  │                              ╱
  │                        ╱
  │                  ╱  GPT-3 (175B)
  │            ╱
  │      ╱  GPT-2 (1.5B)
  │  ╱
  │ GPT-1 (117M)
  └──────────────────────────────────────
      2018    2019    2020    2021    2022    2023

  每代增长约 10-100 倍
  这就是 "Scaling Law" 的直观体现
```

---

## 四、Scaling Law

### 4.1 核心发现

```
Scaling Law (Kaplan et al., 2020):

  模型性能(Loss)与三个因素呈幂律关系:

  ┌────────────────────────────────────────────────┐
  │                                                │
  │  ① 模型参数量 (N)                              │
  │  ② 数据量 (D)                                  │
  │  ③ 计算量 (C)                                  │
  │                                                │
  │  Loss ≈ A / N^α + B / D^β + C_offset          │
  │                                                │
  │  其中 α ≈ 0.076, β ≈ 0.095                    │
  │                                                │
  │  核心洞察:                                     │
  │  - 模型越大，数据越多，性能越好                 │
  │  - 性能提升是可预测的（幂律）                   │
  │  - 投入更多资源 = 更好的模型（无饱和迹象）      │
  │                                                │
  └────────────────────────────────────────────────┘

  Loss 与模型大小的关系 (对数图):

  Loss
   4.0 │ ●
       │   ●  GPT-1
   3.5 │     ●
       │       ●  GPT-2
   3.0 │         ●
       │           ●
   2.5 │             ●  GPT-3
       │               ●
   2.0 │                 ● (预测)
       └──────────────────────────────
        10⁸  10⁹  10¹⁰  10¹¹  10¹²
                 参数量 N
```

### 4.2 计算最优分配

```
Chinchilla 发现 (DeepMind, 2022):

  之前的做法: 模型太大，数据太少 (计算次优)

  Chinchilla 最优:
  ┌────────────────────────────────────────────────┐
  │                                                │
  │  给定计算预算 C，最优分配为:                    │
  │                                                │
  │  参数量 N ≈ ∝ C^0.5                            │
  │  数据量 D ≈ ∝ C^0.5                            │
  │                                                │
  │  即: 模型大小和数据量应该同步增长               │
  │                                                │
  │  GPT-3 的教训:                                 │
  │  - 参数 175B，但只训练了 300B tokens           │
  │  - 数据量严重不足 (应该 ~3.5T tokens)          │
  │  - Chinchilla 70B + 1.4T tokens 反超 GPT-3     │
  │                                                │
  │  这就是为什么 Llama 等新模型用更多数据训练      │
  │                                                │
  └────────────────────────────────────────────────┘
```

---

## 五、涌现能力

### 5.1 什么是涌现

```
涌现能力 (Emergent Abilities):

  定义: 模型规模达到某个阈值后，突然出现的新能力

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  性能                                             │
  │   100% │                        ╱━━━━━━━━        │
  │    80% │                    ╱━━                  │
  │    60% │                ╱━━                      │
  │    40% │            ╱━━                          │
  │    20% │        ╱━━                              │
  │     0% │━━━━━━━                                  │
  │        └──────────────────────────────────────    │
  │         10⁸  10⁹  10¹⁰  10¹¹  10¹²  10¹³         │
  │                    参数量                         │
  │                                                  │
  │  特点: 相位突变 (phase transition)               │
  │  - 小模型完全做不到 (接近 0% 准确率)             │
  │  - 大到某规模突然能做好                          │
  │  - 不是渐进提升，而是"突变"                      │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

### 5.2 典型的涌现能力

```
已观察到的涌现能力:

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  能力1: 少样本学习 (Few-shot Learning)           │
  │  ──────────────────────────────────────────────  │
  │  GPT-3 (175B) 可以通过几个例子学会新任务         │
  │  小模型不行                                      │
  │                                                  │
  │  能力2: 思维链推理 (Chain-of-Thought)            │
  │  ──────────────────────────────────────────────  │
  │  大模型可以通过"一步步思考"解决复杂数学/推理     │
  │  通常在 >60B 参数时出现                          │
  │                                                  │
  │  能力3: 指令遵循 (Instruction Following)         │
  │  ──────────────────────────────────────────────  │
  │  理解并执行复杂自然语言指令                      │
  │                                                  │
  │  能力4: 代码生成与执行                            │
  │  ──────────────────────────────────────────────  │
  │  编写、理解、调试代码                            │
  │                                                  │
  │  能力5: 多语言能力                                │
  │  ──────────────────────────────────────────────  │
  │  跨语言翻译、理解、生成                          │
  │                                                  │
  │  能力6: 知识问答                                  │
  │  ──────────────────────────────────────────────  │
  │  回答事实性问题，体现"世界知识"                  │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

---

## 六、少样本学习

### 6.1 三种学习方式

```
大模型的学习方式 (按提示复杂度):

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  ① Zero-shot (零样本)                            │
  │  ──────────────────────────────────────────────  │
  │  不给任何示例，直接描述任务                       │
  │                                                  │
  │  Prompt: "翻译成英文: 我爱自然语言处理"          │
  │  Output: "I love natural language processing"    │
  │                                                  │
  │  ② One-shot (单样本)                             │
  │  ──────────────────────────────────────────────  │
  │  给 1 个示例                                     │
  │                                                  │
  │  Prompt: "猫→cat, 狗→"                          │
  │  Output: "dog"                                   │
  │                                                  │
  │  ③ Few-shot (少样本)                             │
  │  ──────────────────────────────────────────────  │
  │  给少量示例 (通常 3-10 个)                       │
  │                                                  │
  │  Prompt:                                         │
  │    "猫→cat                                       │
  │     狗→dog                                       │
  │     鸟→bird                                      │
  │     鱼→"                                         │
  │  Output: "fish"                                  │
  │                                                  │
  │  关键: 不更新参数! 只通过上下文学习               │
  │  这叫 In-Context Learning (ICL)                  │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

### 6.2 与微调的对比

```
Few-shot vs Fine-tuning:

  ┌──────────────┬──────────────────┬──────────────────┐
  │              │   Fine-tuning    │    Few-shot      │
  ├──────────────┼──────────────────┼──────────────────┤
  │  更新参数    │  是              │  否              │
  │  需要训练    │  是              │  否              │
  │  数据需求    │  大 (千~万)      │  小 (几个)       │
  │  任务切换    │  需重新训练      │  换 prompt 即可  │
  │  效果        │  通常更好        │  依赖模型大小    │
  │  部署成本    │  每任务一个模型  │  一个模型通吃    │
  └──────────────┴──────────────────┴──────────────────┘

  GPT-3 的贡献: 证明了 Few-shot 可以接近甚至达到 Fine-tuning 效果
  这是大模型实用化的关键 —— 不需要为每个任务训练
```

---

## 七、HuggingFace 使用 GPT

### 7.1 文本生成

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 加载 GPT-2
model_name = "gpt2"  # 可选: gpt2-medium, gpt2-large, gpt2-xl
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"词表大小: {tokenizer.vocab_size}")
print(f"层数: {model.config.n_layer}")
print(f"隐藏维度: {model.config.n_embd}")

# 文本生成
def generate_text(prompt, max_length=100, temperature=0.7, top_k=50):
    """GPT-2 文本生成"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# 测试生成
prompt = "Natural language processing is"
result = generate_text(prompt, max_length=50)
print(f"\nPrompt: {prompt}")
print(f"Generated: {result}")
```

### 7.2 查看模型结构

```python
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")

# 打印模型结构
print("GPT-2 模型结构:")
print(model)

# 关键组件:
# - wte: 词嵌入 (Token Embedding)
# - wpe: 位置嵌入 (Position Embedding, 可学习)
# - h: 12 个 Transformer 解码器层
#     - ln_1: LayerNorm
#     - attn: 因果 Self-Attention
#     - ln_2: LayerNorm
#     - mlp: 前馈网络
# - ln_f: 最终 LayerNorm
# - lm_head: 语言模型输出头

# 验证因果注意力掩码
def check_causal_mask():
    """检查 GPT-2 的因果掩码"""
    import torch.nn.functional as F

    seq_len = 5
    # 创建因果掩码 (下三角)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    print(f"\n因果掩码 ({seq_len}x{seq_len}):")
    print(mask)

check_causal_mask()
```

### 7.3 简化版 GPT 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GPTBlock(nn.Module):
    """GPT 解码器层 (简化版)"""
    def __init__(self, d_model=768, num_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        # LayerNorm 在 Attention 之前 (Pre-LN, GPT-2 风格)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Pre-LN 自注意力
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + attn_out
        # Pre-LN FFN
        h = self.ln2(x)
        ffn_out = self.ffn(h)
        x = x + ffn_out
        return x


class SimpleGPT(nn.Module):
    """简化版 GPT 模型"""
    def __init__(self, vocab_size=50257, d_model=768, num_heads=12,
                 num_layers=12, d_ff=3072, max_len=1024):
        super().__init__()
        # 词嵌入 + 位置嵌入 (都是可学习的)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # 堆叠的 Transformer 解码器层
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享 (嵌入和输出层)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()

        # 位置索引
        positions = torch.arange(seq_len, device=input_ids.device)

        # 嵌入 = 词嵌入 + 位置嵌入
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # 因果掩码 (下三角)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'), diagonal=1)

        # 通过所有层
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (batch, seq, vocab)
        return logits


# 测试
if __name__ == "__main__":
    model = SimpleGPT(
        vocab_size=50257, d_model=768, num_heads=12,
        num_layers=12, d_ff=3072, max_len=1024
    )
    print(f"简化版 GPT 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟前向传播
    input_ids = torch.randint(0, 50257, (2, 20))
    logits = model(input_ids)
    print(f"输入: {input_ids.shape}")
    print(f"输出 logits: {logits.shape}")  # (2, 20, 50257)
    print(f"可以预测每个位置的下一个词")

    # 生成下一个词的概率分布
    next_token_logits = logits[0, -1, :]  # 取最后一个位置
    next_token_prob = F.softmax(next_token_logits, dim=-1)
    next_token_id = torch.argmax(next_token_prob).item()
    print(f"预测的下一个 token ID: {next_token_id}")
```

---

## 八、GPT 系列影响

```
GPT 系列的深远影响:

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  技术层面:                                       │
  │  - 证明了 Decoder-only 架构的强大                 │
  │  - 确立了 Scaling Law 的指导作用                  │
  │  - 展示了涌现能力的存在                          │
  │  - 推动了 Few-shot / Zero-shot 学习              │
  │                                                  │
  │  应用层面:                                       │
  │  - ChatGPT 引爆 AI 普及                          │
  │  - 代码助手 (GitHub Copilot, CodeBuddy)          │
  │  - 自动写作、翻译、摘要                          │
  │  - 对话系统、客服机器人                          │
  │                                                  │
  │  生态层面:                                       │
  │  - 开源大模型涌现 (Llama, Mistral, Qwen...)     │
  │  - AI 应用开发框架 (LangChain, LlamaIndex...)    │
  │  - RAG (检索增强生成) 技术                       │
  │  - AI Agent 智能体                               │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

---

## 小结

| 要点 | 内容 |
|------|------|
| 全称 | Generative Pre-trained Transformer |
| 架构 | Decoder-only（只用 Transformer 解码器） |
| 注意力 | 因果注意力（Causal Attention），只看左侧 |
| 预训练 | 自回归语言模型（预测下一个词） |
| GPT-1 | 117M，证明预训练+微调可行 |
| GPT-2 | 1.5B，展现 Zero-shot 潜力 |
| GPT-3 | 175B，Few-shot 学习，涌现能力 |
| GPT-4 | 多模态，更强推理，RLHF 对齐 |
| Scaling Law | 参数×数据×计算 同步增长，性能幂律提升 |
| Chinchilla | 模型和数据应同步增大（各 C^0.5） |
| 涌现能力 | 规模超过阈值后突变出现的新能力 |
| Few-shot | 不更新参数，通过上下文示例学习 |
| vs BERT | GPT 擅长生成，BERT 擅长理解 |
| 影响 | 开创了大模型时代和通用 AI 浪潮 |

---

| [← 回到目录](../README.md) | [上一篇：BERT预训练模型](03-BERT预训练模型.md) | [下一篇：T5与编码器-解码器](05-T5与编码器-解码器.md) |
|---|---|---|
