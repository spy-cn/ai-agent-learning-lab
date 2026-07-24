# BERT 预训练模型

2018 年，Google 发表的 **BERT（Bidirectional Encoder Representations from Transformers）** 彻底改变了 NLP 的范式。BERT 利用 Transformer 的编码器，通过在大规模无标注文本上进行**双向预训练**，学习到丰富的语言表示，随后只需在下游任务上微调即可达到 SOTA 表现。BERT 的出现标志着 NLP 正式进入**预训练 + 微调**时代。

---

## 一、BERT 是什么

### 1.1 名称含义

```
BERT = Bidirectional Encoder Representations from Transformers

  B - Bidirectional     双向: 同时从左→右和右→左理解上下文
  E - Encoder           使用 Transformer 的编码器
  R - Representations   学习语言表示
  T - Transformers      基于 Transformer 架构

  核心创新: 真正的双向理解

  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │  传统单向 (GPT/ELMo):                           │
  │    "我 [?] 吃 苹果"  只能从左→右看               │
  │    或只能从右→左看                               │
  │                                                 │
  │  BERT 双向:                                     │
  │    "我 [?] 吃 苹果"  ← 同时看左边和右边          │
  │    左侧上下文 + 右侧上下文 → 更准确预测           │
  │                                                 │
  └─────────────────────────────────────────────────┘
```

### 1.2 核心思想

```
BERT 的核心思想:

  ① 用 Transformer 编码器作为骨干
  ─────────────────────────────────
    只用 Encoder，不要 Decoder
    因为 BERT 的目标是"理解"语言，不是"生成"语言

  ② 设计两个预训练任务
  ─────────────────────────────────
    MLM (Masked Language Model): 随机遮挡词，让模型预测
    NSP (Next Sentence Prediction): 判断两句话是否相邻

  ③ 预训练 + 微调范式
  ─────────────────────────────────
    Step 1: 在大规模无标注文本上预训练（学习通用语言表示）
    Step 2: 在下游任务上微调（只需少量标注数据）

  ┌──────────────┐     ┌──────────────────┐
  │ 大规模无标注  │ ──→ │   通用语言表示    │
  │   文本语料    │ 预练│  (Pre-trained BERT)│
  └──────────────┘     └────────┬─────────┘
                                │ 微调
                    ┌───────────┼───────────┐
                    ↓           ↓           ↓
              ┌──────────┐ ┌────────┐ ┌─────────┐
              │ 文本分类  │ │ 问答   │ │ NER     │
              │ 情感分析  │ │ QA     │ │ 命名实体│
              └──────────┘ └────────┘ └─────────┘
```

---

## 二、两个预训练任务

### 2.1 MLM（Masked Language Model）

```
Masked Language Model (掩码语言模型):

  随机遮挡输入中 15% 的 token，让模型预测被遮挡的词

  示例:
    原始: "自然 语言 处理 是 人工智能 的 重要 分支"
    遮挡: "自然 [MASK] 处理 是 [MASK] 的 重要 分支"
    目标: 预测 [MASK] → "语言", "人工智能"

  ┌─────────────────────────────────────────────────┐
  │  15% 被选中的 token 中:                         │
  │                                                 │
  │  80% → 替换为 [MASK]                            │
  │        "语言" → "[MASK]"                        │
  │                                                 │
  │  10% → 替换为随机词                             │
  │        "语言" → "苹果" (随机)                   │
  │                                                 │
  │  10% → 保持不变                                 │
  │        "语言" → "语言" (不变)                   │
  │                                                 │
  │  为什么这样设计?                                │
  │  - 80% [MASK]: 标准掩码预测                     │
  │  - 10% 随机词: 防止模型认为只有[MASK]需要预测   │
  │  - 10% 不变:   保持输入分布的一致性             │
  └─────────────────────────────────────────────────┘

  对比 GPT 的单向语言模型:
    GPT: "我 爱 [?]"        ← 只能看左侧
    BERT: "我 [?] NLP"      ← 左右都能看
```

### 2.2 NSP（Next Sentence Prediction）

```
Next Sentence Prediction (下一句预测):

  判断两个句子是否是原文中的相邻句子

  ┌────────────────────────────────────────────────┐
  │                                                │
  │  输入 A (正例):                                │
  │    句子A: "今天天气很好"                        │
  │    句子B: "我们去公园散步"   ← IsNext: Yes     │
  │                                                │
  │  输入 B (负例):                                │
  │    句子A: "今天天气很好"                        │
  │    句子B: "Python是编程语言"  ← IsNotNext: No  │
  │                                                │
  │  目标: 让模型理解句子间的关系                   │
  │  用途: 问答、自然语言推断等需要句对理解的任务   │
  │                                                │
  └────────────────────────────────────────────────┘

  注意: RoBERTa 研究发现 NSP 任务效果有限，后来去掉了
```

---

## 三、BERT 架构详解

### 3.1 架构图

```
                    BERT 架构

   ┌─────────────────────────────────────┐
   │           输入表示                  │
   │  Token + Segment + Position Embedding │
   │           ↓                         │
   │   ┌─────────────────────────────┐   │
   │   │    Transformer Encoder      │   │
   │   │           × 12              │   │  ← BERT-Base
   │   │         (或 × 24)           │   │  ← BERT-Large
   │   └──────────┬──────────────────┘   │
   │              ↓                      │
   │   输出: 每个位置的隐藏表示          │
   │   [CLS] h₁   h₂   h₃  ...  hₙ     │
   └─────────────────────────────────────┘

   预训练输出:
     [CLS] → IsNext/NotNext (NSP 任务)
     [MASK] 位置 → 预测原词 (MLM 任务)

   微调时:
     [CLS] → 分类标签 (文本分类)
     所有位置 → 序列标注 (NER/POS)
```

### 3.2 输入表示

```
BERT 输入 = Token Embedding + Segment Embedding + Position Embedding

  示例: 句对 "今天天气好" / "去公园吗"

  输入序列:
  [CLS] 今天 天气 好 [SEP] 去 公园 吗 [SEP]
    ↑    ── 句子A ──  ↑   ── 句子B ──  ↑
   分类     0 0 0     分隔   1 1 1     分隔

  ┌──────┬──────┬──────┬───────┬──────┬──────┬──────┬───────┐
  │      │Token │Segmt │Pos    │      │Token │Segmt │Pos    │
  ├──────┼──────┼──────┼───────┼──────┼──────┼──────┼───────┤
  │[CLS] │ e_cls│  0   │   0   │  ... │      │      │       │
  │今天  │ e_今 │  0   │   1   │      │      │      │       │
  │天气  │ e_天 │  0   │   2   │      │      │      │       │
  │好    │ e_好 │  0   │   3   │      │      │      │       │
  │[SEP] │ e_sep│  0   │   4   │      │      │      │       │
  │去    │ e_去 │  1   │   5   │      │      │      │       │
  │��园  │ e_园 │  1   │   6   │      │      │      │       │
  │吗    │ e_吗 │  1   │   7   │      │      │      │       │
  │[SEP] │ e_sep│  1   │   8   │      │      │      │       │
  └──────┴──────┴──────┴───────┴──────┴──────┴──────┴───────┘

  三种 Embedding 相加 → 最终输入

  [CLS]: 特殊 token，其最终隐藏状态用于分类任务
  [SEP]: 句子分隔符
```

### 3.3 模型规模

```
BERT 的两种规格:

  ┌──────────────┬──────────┬──────────┬──────────┐
  │              │   层数   │  隐藏维度│   头数    │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ BERT-Base    │    12    │   768    │    12    │
  │ BERT-Large   │    24    │   1024   │    16    │
  └──────────────┴──────────┴──────────┴──────────┘

  BERT-Base:  约 110M 参数
  BERT-Large: 约 340M 参数

  对比:
    GPT-1:    117M (与 BERT-Base 相近)
    GPT-3:    175,000M (BERT 的 500 倍)
```

---

## 四、BERT 变体家族

### 4.1 家族概览

```
BERT 变体家族:

                    ┌── RoBERTa (优化训练)
                    │
                    ├── ALBERT (参数共享)
  BERT (2018) ──────┼── DistilBERT (模型蒸馏)
                    │
                    ├── ELECTRA (替换检测)
                    │
                    └── SpanBERT (片段预测)

  时间线:
    2018.10  BERT      原始版本
    2019.07  XLNet     排列语言模型
    2019.07  RoBERTa   更好的训练策略
    2019.09  ALBERT    参数减少
    2019.10  DistilBERT 轻量化
    2020.03  ELECTRA   替换检测预训练
    2020+    各领域 BERT (中文BERT, SciBERT, BioBERT...)
```

### 4.2 RoBERTa

```
RoBERTa (Robustly Optimized BERT Pretraining):

  核心改进: 不是改架构，而是优化训练策略

  ┌────────────────────────────────────────────────┐
  │  改进1: 去掉 NSP 任务                          │
  │  ──────────────────────────────────────────────│
  │  研究发现 NSP 对效果提升有限甚至有害            │
  │                                                │
  │  改进2: 更大的批量 (batch_size 8K)             │
  │  ──────────────────────────────────────────────│
  │  大 batch + 更多数据 = 更好的表示               │
  │                                                │
  │  改进3: 更多训练数据 (16GB → 160GB)            │
  │  ──────────────────────────────────────────────│
  │  CC-News, OpenWebText, Stories 等新增语料      │
  │                                                │
  │  改进4: 动态掩码 (Dynamic Masking)             │
  │  ──────────────────────────────────────────────│
  │  BERT 在预处理时一次性确定掩码                  │
  │  RoBERTa 在每个 epoch 重新随机掩码              │
  │                                                │
  │  改进5: 更长的训练 (更多 epochs)                │
  │  ──────────────────────────────────────────────│
  │  从 40K 步增加到 500K 步                       │
  └────────────────────────────────────────────────┘

  结果: 同样架构，效果大幅超过原始 BERT
```

### 4.3 ALBERT

```
ALBERT (A Lite BERT):

  目标: 减少参数量，保持或提升性能

  ┌────────────────────────────────────────────────┐
  │  技术1: 因式分解词嵌入                          │
  │  ──────────────────────────────────────────────│
  │  BERT:  Vocab × d_model = 30K × 768 = 23M     │
  │  ALBERT: Vocab × d_emb + d_emb × d_model       │
  │         = 30K × 128 + 128 × 768 = 4.7M        │
  │                                                │
  │  技术2: 跨层参数共享                            │
  │  ──────────────────────────────────────────────│
  │  BERT:  12 层各有独立参数                       │
  │  ALBERT: 12 层共享同一组参数                    │
  │                                                │
  │  效果: 参数量减少 90%，性能接近                 │
  │  ALBERT-xxlarge: 性能超过 BERT-Large            │
  └────────────────────────────────────────────────┘
```

### 4.4 DistilBERT

```
DistilBERT (蒸馏 BERT):

  用知识蒸馏技术压缩模型

  ┌────────────────────────────────────────────────┐
  │                                                │
  │   Teacher (BERT)     Student (DistilBERT)      │
  │   ┌──────────┐       ┌──────────┐              │
  │   │ 12 层    │ ───→  │ 6 层     │              │
  │   │ 110M参数 │ 蒸馏  │ 66M参数  │              │
  │   └──────────┘       └──────────┘              │
  │                        ↑                       │
  │                   学习 Teacher 的              │
  │                   输出分布和隐藏层              │
  │                                                │
  │  结果: 参数减少 40%，速度快 60%，               │
  │        性能保留约 97%                           │
  │                                                │
  └────────────────────────────────────────────────┘
```

### 4.5 ELECTRA

```
ELECTRA (替换检测):

  创新的预训练任务: 不是预测 [MASK]，而是检测哪个词被替换了

  ┌────────────────────────────────────────────────┐
  │                                                │
  │  步骤1: 生成器替换部分词                        │
  │  ──────────────────────────────────────────────│
  │  原始: "自然 语言 处理"                         │
  │  替换: "自然 编程 处理"  ← "语言"被替换为"编程" │
  │                                                │
  │  步骤2: 判别器检测                              │
  │  ──────────────────────────────────────────────│
  │  "自然" → original (正确)                      │
  │  "编程" → replaced (检测到替换!)                │
  │  "处理" → original (正确)                      │
  │                                                │
  │  优势: 所有 token 都参与训练 (不是只有 15%)    │
  │  效率: 比 MLM 训练效率更高                      │
  │                                                │
  └────────────────────────────────────────────────┘
```

---

## 五、BERT 下游任务应用

### 5.1 四大应用场景

```
BERT 微调的四种典型任务:

  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  任务1: 句子对分类 (Sentence Pair Classification)      │
  │  ──────────────────────────────────────────────────    │
  │  [CLS] 句子A [SEP] 句子B [SEP]                         │
  │   ↓                                                    │
  │  取 [CLS] 的输出 → 分类器 → 类别                       │
  │  应用: 自然语言推断(NLI)、语义相似度                    │
  │                                                        │
  │  任务2: 单句分类 (Single Sentence Classification)      │
  │  ──────────────────────────────────────────────────    │
  │  [CLS] 句子 [SEP]                                      │
  │   ↓                                                    │
  │  取 [CLS] 的输出 → 分类器 → 类别                       │
  │  应用: 情感分析、主题分类                               │
  │                                                        │
  │  任务3: 问答 (Question Answering)                      │
  │  ──────────────────────────────────────────────────    │
  │  [CLS] 问题 [SEP] 文章 [SEP]                           │
  │   ↓                                                    │
  │  每个位置 → 分类器 → 答案起始/结束位置                 │
  │  应用: 抽取式问答 (SQuAD)                               │
  │                                                        │
  │  任务4: 序列标注 (Sequence Labeling)                   │
  │  ──────────────────────────────────────────────────    │
  │  [CLS] 词1 词2 词3 ... [SEP]                           │
  │   ↓   ↓   ↓   ↓                                        │
  │  每个位置 → 分类器 → 标签                              │
  │  应用: 命名实体识别(NER)、词性标注(POS)                │
  │                                                        │
  └────────────────────────────────────────────────────────┘
```

### 5.2 微调策略

```
微调 (Fine-tuning) 策略:

  ┌────────────────────────────────────────────┐
  │                                            │
  │  冻结策略选择:                             │
  │                                            │
  │  策略A: 全部微调 (推荐)                    │
  │  ─────────────────────────────             │
  │  解冻所有 BERT 层 + 新分类头               │
  │  用较小学习率 (2e-5 ~ 5e-5)                │
  │  效果最好，但需要保存完整模型               │
  │                                            │
  │  策略B: 冻结底层，只训练顶层                │
  │  ─────────────────────────────             │
  │  冻结 BERT 前 N 层，只微调后几层           │
  │  适合数据量小的场景                         │
  │                                            │
  │  策略C: 只训练分类头 (特征提取)            │
  │  ─────────────────────────────             │
  │  完全冻结 BERT，只训练分类器               │
  │  速度最快，但效果一般                      │
  │                                            │
  └────────────────────────────────────────────┘
```

---

## 六、HuggingFace 使用 BERT

### 6.1 基础使用

```python
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch

# 加载预训练 tokenizer 和 model
model_name = "bert-base-chinese"  # 中文 BERT
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"词表大小: {tokenizer.vocab_size}")
print(f"隐藏维度: {model.config.hidden_size}")
print(f"层数: {model.config.num_hidden_layers}")

# 文本编码
text = "自然语言处理是人工智能的重要分支"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
print(f"\n输入文本: {text}")
print(f"Token IDs: {inputs['input_ids']}")
print(f"解码: {tokenizer.decode(inputs['input_ids'][0])}")

# 获取词嵌入
with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state: (batch, seq_len, hidden_size)
# outputs.pooler_output: (batch, hidden_size) - [CLS] 的池化输出
print(f"\n最后隐藏层形状: {outputs.last_hidden_state.shape}")
print(f"池化输出形状: {outputs.pooler_output.shape}")
```

### 6.2 文本分类微调

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch


class TextClassificationDataset(Dataset):
    """文本分类数据集"""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 模拟情感分析数据
train_texts = [
    "这部电影太好看了，强烈推荐!",
    "垃圾产品，千万别买",
    "质量不错，物美价廉",
    "太差了，完全不值这个价",
    "服务态度很好，下次还来",
    "糟糕的体验，再也不来了"
]
train_labels = [1, 0, 1, 0, 1, 0]  # 1=正面, 0=负面

# 创建数据集
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)

# 创建分类模型
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2  # 二分类
)

# 训练配置
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 训练 (取消注释来实际运行)
# trainer.train()

print("BERT 文本分类微调代码已准备就绪!")
print(f"训练样本数: {len(train_dataset)}")
print(f"分类类别: {model.config.num_labels}")
```

### 6.3 提取词向量

```python
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")
model.eval()

def get_sentence_embedding(text, method='cls'):
    """获取句子向量表示"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    if method == 'cls':
        # 方法1: 用 [CLS] 的输出
        embedding = outputs.last_hidden_state[:, 0, :]
    elif method == 'mean':
        # 方法2: 所有 token 的平均池化
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embedding = torch.sum(token_embeddings * mask, 1) / torch.sum(mask, 1)

    return embedding.squeeze(0)


# 计算句子相似度
sent1 = "我喜欢这个产品"
sent2 = "这个产品很好用"
sent3 = "这个产品太差了"

emb1 = get_sentence_embedding(sent1, 'mean')
emb2 = get_sentence_embedding(sent2, 'mean')
emb3 = get_sentence_embedding(sent3, 'mean')

# 余弦相似度
sim_12 = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
sim_13 = F.cosine_similarity(emb1.unsqueeze(0), emb3.unsqueeze(0)).item()

print(f"'{sent1}' vs '{sent2}' 相似度: {sim_12:.4f}")
print(f"'{sent1}' vs '{sent3}' 相似度: {sim_13:.4f}")
print(f"(语义相近的句子相似度应更高)")
```

---

## 七、BERT 变体对比

```
BERT 变体对比表:

┌──────────────┬──────────┬──────────┬──────────┬────────────────┐
│    模型      │  参数量  │  改进点  │  预训练  │     特点       │
├──────────────┼──────────┼──────────┼──────────┼────────────────┤
│ BERT         │ 110M/340M│ 双向编码 │ MLM+NSP  │ 开创性         │
│ RoBERTa     │ 125M/355M│ 优化训练 │ MLM      │ 效果更好       │
│ ALBERT      │ 12M/235M │ 参数共享 │ MLM+SOP  │ 参数少         │
│ DistilBERT  │ 66M      │ 知识蒸馏 │ MLM      │ 速度快         │
│ ELECTRA     │ 110M     │ 替换检测 │ RTD      │ 训练高效       │
│ SpanBERT    │ 110M     │ 片段掩码 │ Span MLM │ 适合抽取任务   │
└──────────────┴──────────┴──────────┴──────────┴────────────────┘
```

---

## 小结

| 要点 | 内容 |
|------|------|
| 全称 | Bidirectional Encoder Representations from Transformers |
| 架构 | 只用 Transformer 编码器（Encoder-only） |
| 核心创新 | 真正的双向语言理解 |
| 预训练任务 | MLM（掩码语言模型）+ NSP（下一句预测） |
| 特殊标记 | [CLS] 用于分类，[SEP] 分隔句子 |
| 输入表示 | Token + Segment + Position Embedding |
| 预训练规模 | BERT-Base 110M，BERT-Large 340M |
| 微调方式 | 在 [CLS] 上接分类器，全部参数微调 |
| RoBERTa | 去掉 NSP，更多数据，更好训练 |
| ALBERT | 词嵌入分解 + 跨层参数共享，减少参数 |
| DistilBERT | 知识蒸馏，6 层，速度快 60% |
| ELECTRA | 替换检测任务，训练效率更高 |
| 应用场景 | 分类、问答、NER、语义相似度等理解任务 |
| 局限 | 不擅长生成任务（需要 GPT/T5） |

---

| [← 回到目录](../README.md) | [上一篇：位置编码](02-位置编码.md) | [下一篇：GPT系列](04-GPT系列.md) |
|---|---|---|
