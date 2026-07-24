# NLP 发展史

从 1950 年代到 2020 年代，NLP 经历了四次范式转变——每一步都让机器更接近人类语言能力。

## 四个时代总览

```
┌──────────────────────────────────────────────────────────────────┐
│                    NLP 70年发展史                                 │
├───────────┬──────────────┬───────────────┬───────────────────────┤
│ 规则时代   │  统计时代    │  深度学习时代   │    大模型时代          │
│ 1950-1980s│  1990-2010  │  2013-2018    │   2018-至今           │
├───────────┼──────────────┼───────────────┼───────────────────────┤
│ 人工编写   │  数学统计    │  神经网络      │   预训练+微调          │
│ 语法规则   │  概率模型    │  端到端学习    │   涌现能力             │
│           │              │               │                       │
│ 代表:     │  代表:       │  代表:         │   代表:               │
│ ELIZA     │  IBM Model   │  Word2Vec     │   BERT                │
│ SHRDLU    │  HMM         ���  Seq2Seq      │   GPT-3/4             │
│           │  CRF         │  Transformer  │   ChatGPT             │
└───────────┴──────────────┴───────────────┴───────────────────────┘
```

## 第一时代：规则时代 (1950s-1980s)

### 关键事件

| 年份 | 事件 | 意义 |
|------|------|------|
| 1950 | **图灵测试** | 提出"机器能否思考"的问题 |
| 1954 | **乔治城-IBM实验** | 俄英机器翻译，首个NLP系统 |
| 1956 | **ELIZA** | 第一个聊天机器人（模式匹配） |
| 1968 | **SHRDLU** | 积木世界自然语言理解 |
| 1971 | **LUNAR** | 月球岩石数据的自然语言查询 |
| 1978 | **XCALIBUR** | 专家系统自然语言接口 |

### 规则方法的特点

```
规则时代的 NLP:
┌───────────────────────────────────────────┐
│                                           │
│  方法: 手工编写语法规则                    │
│                                           │
│  if 输入包含("你好"):                      │
│      回复("你好！有什么可以帮您？")        │
│                                           │
│  规则模板:                                │
│  S → NP VP                               │
│  NP → Det N | Det Adj N                   │
│  VP → V | V NP | V NP PP                  │
│                                           │
│  优点:                                    │
│  ✓ 精确（规则覆盖的情况完美）              │
│  ✓ 可解释                                │
│  ✓ 不需要数据                             │
│                                           │
│  缺点:                                    │
│  ✗ 规则爆炸（语言现象太多）               │
│  ✗ 无法处理新语言现象                     │
│  ✗ 无法处理歧义                          │
│  ✗ 跨语言需要重写                        │
│                                           │
└───────────────────────────────────────────┘
```

### ELIZA 示例

```python
# ELIZA 的核心思想：模式匹配 + 替换

eliza_rules = {
    r"I need (.*)": [
        "Why do you need {0}?",
        "Would it really help you to get {0}?"
    ],
    r"Why don'?t you (.*)": [
        "Do you really think I don't {0}?",
        "Perhaps eventually I will {0}."
    ],
    r"(.*) my (.*)": [
        "Your {1}? Tell me more about your {1}.",
        "Why do you say your {1}?"
    ],
}

def eliza_respond(user_input, rules):
    """ELIZA 风格的回复"""
    import re
    import random
    
    for pattern, responses in rules.items():
        match = re.match(pattern, user_input, re.IGNORECASE)
        if match:
            response = random.choice(responses)
            return response.format(*match.groups())
    
    return "Tell me more about that."

# 测试
print(eliza_respond("I need some help", eliza_rules))
# → "Why do you need some help?"
print(eliza_respond("I am worried about my job", eliza_rules))
# → "Your job? Tell me more about your job."
```

## 第二时代：统计时代 (1990s-2010)

### 核心思想转变

```
规则 → 统计: 让数据说话

规则时代:   人写规则 → 应用规则
统计时代:   收集数据 → 统计规律 → 应用规律

关键洞察:
  "语言有统计规律，不需要完美规则"
  
  P("the cat sat") > P("the sat cat")
  
  → 用概率模型选择最可能的解释
```

### 关键技术

| 技术 | 时间 | 核心思想 |
|------|------|----------|
| **n-gram 语言模型** | 1980s | 用前n-1个词预测第n个词 |
| **IBM翻译模型** | 1993 | 统计机器翻译 |
| **HMM** | 1990s | 隐马尔可夫模型做词性标注 |
| **CRF** | 2001 | 条件随机场做序列标注 |
| **最大熵** | 1990s | 最大化熵的原则建模 |
| **感知器** | 2002 | 结构化感知器 |
| **WordNet** | 1995 | 语义网络 |

### n-gram 语言模型

```python
import nltk
from nltk import bigrams, trigrams
from collections import Counter, defaultdict

# 构建简单的 bigram 语言模型
text = """
the cat sat on the mat
the dog sat on the log
the cat chased the dog
the dog chased the cat
""".strip()

tokens = text.split()

# 统计 bigram 频率
bigram_counts = Counter(bigrams(tokens))
unigram_counts = Counter(tokens)

# 计算 P(word | prev_word)
def bigram_probability(prev_word, word):
    """P(word | prev_word) = Count(prev_word, word) / Count(prev_word)"""
    pair_count = bigram_counts[(prev_word, word)]
    prev_count = unigram_counts[prev_word]
    return pair_count / prev_count if prev_count > 0 else 0

# 预测下一个词
def predict_next(prev_word):
    candidates = [(w, bigram_probability(prev_word, w)) 
                  for w in set(tokens)]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:3]

print(predict_next("the"))
# [('cat', 0.4), ('dog', 0.4), ('mat', 0.2)]
print(predict_next("sat"))
# [('on', 1.0)]

# 语言模型困惑度 (简化)
def sentence_probability(sentence):
    tokens = sentence.split()
    prob = unigram_counts[tokens[0]] / sum(unigram_counts.values())
    for i in range(1, len(tokens)):
        prob *= bigram_probability(tokens[i-1], tokens[i])
    return prob

print(f"P('the cat sat on the mat') = {sentence_probability('the cat sat on the mat'):.6f}")
print(f"P('the mat sat on the cat') = {sentence_probability('the mat sat on the cat'):.6f}")
```

### HMM 词性标注

```python
# 简化的 HMM 词性标注

# 观测序列: 词序列
# 隐状态: 词性序列

# 转移概率 P(POS_i | POS_{i-1})
transition = {
    'START': {'N': 0.4, 'V': 0.3, 'Det': 0.3},
    'Det':   {'N': 0.8, 'Adj': 0.2},
    'N':     {'V': 0.5, 'N': 0.3, '.': 0.2},
    'V':     {'Det': 0.4, 'N': 0.5, '.': 0.1},
    'Adj':   {'N': 0.9, '.': 0.1},
}

# 发射概率 P(word | POS)
emission = {
    'Det': {'the': 0.6, 'a': 0.3, 'an': 0.1},
    'N':   {'cat': 0.3, 'dog': 0.3, 'mat': 0.2, 'fish': 0.2},
    'V':   {'sat': 0.3, 'chased': 0.3, 'ran': 0.4},
    'Adj': {'big': 0.4, 'small': 0.3, 'red': 0.3},
}

# 维特比算法
def viterbi(words, transition, emission):
    """维特比算法寻找最优词性序列"""
    states = ['Det', 'N', 'V', 'Adj']
    
    V = [{}]  # V[t][state] = (概率, 路径)
    
    # 初始化
    for s in states:
        trans_p = transition['START'].get(s, 0)
        emit_p = emission.get(s, {}).get(words[0], 0)
        V[0][s] = (trans_p * emit_p, [s])
    
    # 递推
    for t in range(1, len(words)):
        V.append({})
        for s in states:
            emit_p = emission.get(s, {}).get(words[t], 0)
            if emit_p == 0:
                V[t][s] = (0, [])
                continue
            
            best = max(
                (V[t-1][s0][0] * transition[s0].get(s, 0) * emit_p, 
                 V[t-1][s0][1] + [s])
                for s0 in states
                if transition[s0].get(s, 0) > 0
            )
            V[t][s] = best
    
    # 回溯最优路径
    best_last = max(V[-1][s] for s in states)
    return best_last[1]

words = ['the', 'cat', 'sat']
result = viterbi(words, transition, emission)
print(f"句子: {' '.join(words)}")
print(f"词性: {' '.join(result)}")
# 词性: Det N V
```

## 第三时代：深度学习时代 (2013-2018)

### 范式转变

```
统计方法 → 神经网络

统计时代:  特征工程 (人工设计特征) → 分类器
深度学习:  原始文本 → 神经网络自动学习特征

关键论文时间线:
  2003 - 神经网络语言模型 (Bengio)
  2013 - Word2Vec (Mikolov)     → 词嵌入革命
  2014 - Seq2Seq (Sutskever)    → 编码器-解码器
  2014 - GRU (Cho)              → 门控RNN
  2015 - Attention (Bahdanau)   → 注意力机制
  2015 - Skip-thought           → 句向量
  2018 - ELMo (Peters)          → 上下文词向量
```

### Word2Vec: 分布式表示

```python
# Word2Vec 的核心思想: 词的含义由其上下文决定

# "You shall know a word by the company it keeps"
#                          — J.R. Firth (1957)

from gensim.models import Word2Vec

# 准备语料
sentences = [
    ['the', 'cat', 'sat', 'on', 'the', 'mat'],
    ['the', 'dog', 'sat', 'on', 'the', 'log'],
    ['cats', 'and', 'dogs', 'are', 'pets'],
    ['the', 'cat', 'chased', 'the', 'mouse'],
    ['dogs', 'like', 'to', 'play', 'with', 'balls'],
]

# 训练 Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, 
                 min_count=1, workers=4)

# 词向量
cat_vec = model.wv['cat']
print(f"'cat' 向量维度: {cat_vec.shape}")

# 词相似度
similarity = model.wv.similarity('cat', 'dog')
print(f"cat-dog 相似度: {similarity:.4f}")

# 类比: king - man + woman ≈ queen
# (需要更大的语料库才能得到好结果)
```

### RNN/LSTM/Transformer 演进

```
2014 Seq2Seq: 编码器-解码器架构
┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
│ ENC │ ──→ │ ENC │ ──→ │ ENC │ ──→ │ CTX │ ──→ 解码器
└─────┘     └─────┘     └─────┘     └─────┘
  I          am          here

问题: 固定长度的上下文向量是瓶颈

     ↓ 2015 Attention

带注意力的 Seq2Seq:
┌─────┐     ┌─────┐     ┌─────┐
│ ENC │     │ ENC │     │ ENC │
└──┬──┘     └──┬──┘     └──┬──┘
   │           │           │     ← 每步都关注所有位置
   └─────┬─────┘───────────┘
         │ Attention
         ↓
      Decoder

     ↓ 2017 Transformer (只保留注意力!)

Transformer:
┌──────────────────────────────────────────┐
│              Multi-Head Attention        │
│    ┌───┐ ┌───┐ ┌───┐ ┌───┐               │
│    │H1 │ │H2 │ │H3 │ │H4 │  ...          │
│    └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘               │
│      └─────┴──┬──┴─────┘                 │
│         Concat + Linear                  │
└──────────────────────────────────────────┘
      完全不需要 RNN，全靠 Attention
```

## 第四时代：大模型时代 (2018-至今)

### 预训练 + 微调范式

```
传统: 每个任务从零训练
  分类任务 → 从零训练一个分类器
  NER任务  → 从零训练一个序列标注器

大模型: 先预训练，再微调
  大规模语料 → 预训练 → 通用模型 → 微调 → 特定任务

              预训练 (无监督)
         ┌──────────────────────┐
         │  海量文本 (TB级)      │
         │  学习语言通用知识     │
         └──────────┬───────────┘
                    │
         ┌──────────┴───────────┐
         │     通用语言模型       │
         │   (BERT / GPT / T5)  │
         └──────────┬───────────┘
                    │ 微调
    ┌───────────────┼───────────────┐
    │               │               │
  分类微调       NER微调         生成微调
    │               │               │
  情感分析       实体识别         机器翻译
```

### 关键里程碑

| 时间 | 模型 | 参数量 | 突破 |
|------|------|--------|------|
| 2018 | **BERT** | 340M | 刷新11项NLP纪录 |
| 2018 | **GPT** | 117M | 生成式预训练 |
| 2019 | **GPT-2** | 1.5B | 零样本生成 |
| 2019 | **T5** | 11B | 统一所有任务 |
| 2020 | **GPT-3** | 175B | 少样本学习 |
| 2022 | **ChatGPT** | ~175B | RLHF对齐 |
| 2023 | **GPT-4** | ~1.7T | 多模态推理 |
| 2023 | **Llama 2** | 70B | 开源SOTA |
| 2024 | **Llama 3** | 405B | 开源前沿 |

### Scaling Law (缩放定律)

```
性能 ← 模型规模

LLM 的三大 Scaling:
  1. 模型参数量 (Parameters)
  2. 训练数据量 (Tokens)
  3. 计算量 (FLOPs)

核心发现:
  ┌─────────────────────────────────────┐
  │                                     │
  │  性能 ∝ 参数量^α                    │
  │                                     │
  │  Loss = A / N^α + B / D^β + C      │
  │  (N=参数, D=数据)                    │
  │                                     │
  │  关键: 按"最优比例"同时增大N和D     │
  │  Chinchilla: ~20 tokens/parameter   │
  │                                     │
  └─────────────────────────────────────┘
```

### 涌现能力 (Emergent Abilities)

```
模型达到一定规模后突然出现的能力:

参数量 →   1B      10B      50B     100B+
           │        │        │        │
基础能力:  ████████ ████████ ████████ ████████
           翻译/分类/抽取
           
涌现能力:                     ████████ ████████
                              少样本学习

                                        ████████
                                        思维链推理

                                        ████████
                                        指令跟随
```

## NLP 技术栈演化

```
2010年的 NLP 工具栈:
  分词: Stanford Tokenizer
  词性: Stanford POS Tagger  
  句法: Stanford Parser
  NER:  Stanford NER
  → 每个任务一个独立工具

2015年的 NLP 工具栈:
  词向量: Word2Vec / GloVe
  框架:  Keras / TensorFlow
  工具:  NLTK / spaCy

2020年的 NLP 工具栈:
  预训练: HuggingFace Transformers
  数据集: HuggingFace Datasets
  分词:   HuggingFace Tokenizers
  → 一个框架解决所有任务

2024年的 NLP 工具栈:
  基础:   PyTorch / JAX
  模型:   Llama / Qwen / Mistral
  应用:   LangChain / LlamaIndex
  部署:   vLLM / TGI / TensorRT-LLM
  向量库: FAISS / Milvus / Chroma
```

## 中文 NLP 特殊挑战

```
中文 NLP 的独特问题:

1. 无空格分词
   英文: "I love NLP" → ["I", "love", "NLP"]
   中文: "我爱自然语言处理" → ["我", "爱", "自然语言", "处理"]
                                    ↑ 分词边界不确定

2. 歧义分词
   "结婚的和尚未结婚的" → ?
   → "结婚 / 的 / �� / 尚未 / 结婚 / 的" (正确)
   → "结婚 / 的 / 和尚 / 未 / 结婚 / 的" (错误)

3. 没有词形变化
   英文: run/running/ran → 有规律
   中文: 跑 → 没有变化

4. 字符集大
   英文: 26个字母
   中文: ~70000常用字

5. 方言差异
   中文NLP需要考虑简体/繁体/方言
```

## 小结

| 时代 | 核心方法 | 代表 | 局限 |
|------|----------|------|------|
| **规则** | 人工规则 | ELIZA | 规则爆炸 |
| **统计** | 概率模型 | HMM/CRF | 需特征工程 |
| **深度学习** | 神经网络 | BERT/GPT | 需大量标注 |
| **大模型** | 预训练+对齐 | GPT-4 | 计算成本高 |

> **核心趋势**：从"人工设计特征"到"机器自动学习"，从"每个任务独立训练"到"预训练通用模型"。

---

| [← 回到目录](../README.md) | [上一篇：什么是自然语言处理](01-什么是自然语言处理.md) | [下一篇：Python环境搭建](03-Python环境搭建.md) |
|---|---|---|
