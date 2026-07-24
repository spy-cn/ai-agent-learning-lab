# HuggingFace Transformers：预训练模型生态系统

HuggingFace 是当今 NLP 领域最重要的开源社区，其 Transformers 库汇聚了数以万计的预训练模型，让开发者只需几行代码就能使用最前沿的 NLP 能力。本章将系统介绍 HuggingFace 生态系统的核心组件，并通过丰富的代码示例帮助你快速上手。

## 一、HuggingFace 生态全景

HuggingFace 生态由四大核心库构成，它们协同工作，覆盖了从数据处理到模型部署的完整流程。

```
┌─────────────────────────────────────────────────────────────┐
│                   HuggingFace 生态系统                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│   │ Transformers │   │   Datasets   │   │  Tokenizers  │   │
│   │  模型库/架构  │   │  数据集仓库   │   │  分词器(Rust) │   │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   │
│          │                  │                  │            │
│          └──────────┬───────┴──────────┬───────┘            │
│                     │                  │                     │
│              ┌──────┴──────┐   ┌──────┴──────┐              │
│              │  Accelerate │   │     Hub     │              │
│              │ 分布式训练加速│   │ 模型/数据托管│              │
│              └─────────────┘   └─────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 各库职责对比

| 库名 | 主要功能 | 典型用途 |
|------|---------|---------|
| `transformers` | 模型架构加载与推理 | BERT/GPT/T5 等模型使用 |
| `datasets` | 数据集加载与预处理 | 加载/处理 NLP 数据集 |
| `tokenizers` | 高速分词 | BPE/WordPiece 分词 |
| `accelerate` | 训练加速 | 多卡训练/混合精度 |

```python
# 安装核心库
# pip install transformers datasets tokenizers accelerate

# 验证安装
import transformers
import datasets

print(f"Transformers 版本: {transformers.__version__}")
print(f"Datasets 版本: {datasets.__version__}")
```

## 二、pipeline()：一行代码完成 NLP 任务

`pipeline()` 是 Transformers 提供的最高层 API，封装了「分词→模型推理→后处理」的完整流程。

```
输入文本 → [Tokenizer] → [Model] → [Post-process] → 结果
           自动加载      自动加载     自动解码
```

### 2.1 情感分析

```python
from transformers import pipeline

# 一行代码创建情感分析 pipeline
classifier = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-chinanews-chinese")

# 单条推理
result = classifier("这部电影拍得太好了，演员演技在线！")
print(result)
# [{'label': 'positive', 'score': 0.9987}]

# 批量推理
results = classifier([
    "今天天气真糟糕，一直在下雨。",
    "我终于通过了考试，太开心了！",
    "这家餐厅的服务态度很差。"
])
for text, result in zip(["下雨天", "通过考试", "服务差"], results):
    print(f"{text} → {result['label']} ({result['score']:.4f})")
```

### 2.2 命名实体识别（NER）

```python
from transformers import pipeline

# 实体识别 pipeline
ner = pipeline("ner", model="ckiplab/bert-base-chinese-ner", aggregation_strategy="simple")

text = "张三在北京清华大学攻读计算机科学，导师是李四教授。"
entities = ner(text)

print("识别到的实体：")
for ent in entities:
    print(f"  {ent['word']:10s} → {ent['entity_group']:8s} (置信度: {ent['score']:.4f})")

# 输出示例：
# 张三        → PERSON  (置信度: 0.9981)
# 北京清华大学 → ORG     (置信度: 0.9823)
# 李四        → PERSON  (置信度: 0.9965)
```

### 2.3 问答系统

```python
from transformers import pipeline

# 阅读理解式问答
qa = pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")

context = """
华为技术有限公司成立于1987年，总部位于广东省深圳市。
创始人任正非当时43岁，以2.1万元人民币起步。
目前华为是全球领先的信息与通信技术解决方案供应商。
"""

question = "华为的创始人是谁？"
answer = qa(question=question, context=context)
print(f"问题: {question}")
print(f"答案: {answer['answer']} (置信度: {answer['score']:.4f})")
# 答案: 任正非
```

### 2.4 文本生成

```python
from transformers import pipeline

# 中文文本生成
generator = pipeline("text-generation", model="uer/gpt2-chinese-cluecorpussmall")

prompt = "春天来了，"
outputs = generator(
    prompt,
    max_length=100,
    num_return_sequences=3,
    temperature=0.8,
    do_sample=True,
    pad_token_id=generator.tokenizer.eos_token_id
)

print("生成的续写：")
for i, output in enumerate(outputs, 1):
    print(f"  版本{i}: {output['generated_text']}")
```

### 2.5 机器翻译

```python
from transformers import pipeline

# 英译中
translator_en2zh = pipeline(
    "translation_en_to_zh",
    model="Helsinki-NLP/opus-mt-en-zh"
)

text = "Artificial intelligence is transforming the world."
result = translator_en2zh(text)
print(f"原文: {text}")
print(f"译文: {result[0]['translation_text']}")
# 译文: 人工智能正在改变世界。
```

### pipeline 支持的任务一览

| 任务类型 | pipeline 名称 | 说明 |
|---------|--------------|------|
| 情感分析 | `sentiment-analysis` | 文本正负面判断 |
| 命名实体识别 | `ner` | 识别文本中的实体 |
| 问答 | `question-answering` | 基于上下文回答 |
| 文本生成 | `text-generation` | 续写文本 |
| 文本摘要 | `summarization` | 长文变短文 |
| 翻译 | `translation_xx_to_yy` | 语言转换 |
| 零样本分类 | `zero-shot-classification` | 无需训练的分类 |
| 填空 | `fill-mask` | 预测被遮盖的词 |

## 三、AutoModel 与 AutoTokenizer

`Auto*` 系列类可以自动根据模型名称推断出正确的类，极大简化了模型加载过程。

```
模型名称 (如 bert-base-chinese)
         │
         ▼
┌─────────────────┐     自动查询配置
│ AutoConfig      │ ←── 从 Hub 拉取 config.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐     自动匹配分词器
│ AutoTokenizer   │ ←── BertTokenizer / T5Tokenizer ...
└────────┬────────┘
         │
         ▼
┌─────────────────┐     自动匹配模型架构
│ AutoModel       │ ←── BertModel / T5Model ...
└─────────────────┘
```

### 3.1 自动加载模型

```python
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-chinese"

# 自动加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print(f"模型架构:\n{model.config.model_type}")
print(f"隐藏层维度: {model.config.hidden_size}")
print(f"层数: {model.config.num_hidden_layers}")
print(f"词表大小: {model.config.vocab_size}")

# 获取文本的词向量
text = "自然语言处理很有趣"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# outputs.last_hidden_state 形状: [batch_size, seq_len, hidden_size]
print(f"\n输入: {text}")
print(f"词嵌入形状: {outputs.last_hidden_state.shape}")
# torch.Size([1, 12, 768])
```

### 3.2 任务特定模型类

```python
from transformers import (
    AutoModelForSequenceClassification,  # 文本分类
    AutoModelForTokenClassification,     # 序列标注/NER
    AutoModelForQuestionAnswering,       # 问答
    AutoModelForCausalLM,                # 语言模型
    TFAutoModelForSequenceClassification  # TensorFlow 版
)

# 加载带分类头的 BERT（顶部多了线性层）
classifier = AutoModelForSequenceClassification.from_pretrained(
    "uer/roberta-base-finetuned-chinanews-chinese"
)

print("分类头配置:")
print(f"  类别数: {classifier.config.num_labels}")
print(f"  标签映射: {classifier.config.id2label}")
```

## 四、HuggingFace Hub 模型上传与分享

HuggingFace Hub 是模型和数据集的中央仓库，任何人都可以上传自己训练的模型。

### 4.1 上传模型

```python
# 步骤1：在 https://huggingface.co 注册账号
# 步骤2：获取 Access Token (Settings → Access Tokens)

from huggingface_hub import HfApi, create_repo

# 登录（在终端运行 huggingface-cli login，或使用 token）
api = HfApi(token="your_token_here")

# 创建模型仓库
repo_url = api.create_repo(repo_id="your-username/my-bert-model", exist_ok=True)

# 上传模型文件
api.upload_folder(
    folder_path="./my_model_dir",       # 本地模型目录
    repo_id="your-username/my-bert-model",
    repo_type="model"
)

print(f"模型已上传: {repo_url}")
```

### 4.2 从 Hub 加载

```python
from transformers import AutoModel, AutoTokenizer

# 加载别人分享的模型（完全相同的代码）
model = AutoModel.from_pretrained("your-username/my-bert-model")
tokenizer = AutoTokenizer.from_pretrained("your-username/my-bert-model")

# 查看模型下载位置（默认缓存）
import os
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
print(f"缓存目录: {cache_dir}")
```

## 五、中文预训练模型推荐

| 模型名称 | 类型 | 特点 | 适用场景 |
|---------|------|------|---------|
| `bert-base-chinese` | 编码器 | Google 官方中文 BERT | 基础中文 NLP 任务 |
| `hfl/chinese-roberta-wwm-ext` | 编码器 | 全词遮盖，效果优于 BERT | 分类、NER、QA |
| `hfl/chinese-macbert-base` | 编码器 | MacBERT 改进版 | 通用中文理解 |
| `uer/roberta-base-finetuned-*` | 编码器 | 预训练 + 微调 | 即开即用 |
| `uer/gpt2-chinese-cluecorpussmall` | 解码器 | 中文 GPT-2 | 文本生成 |
| `THUDM/chatglm3-6b` | 编解码器 | 对话大模型 | 聊天/问答 |
| `Qwen/Qwen1.5-7B-Chat` | 解码器 | 通义千问 | 通用大模型 |
| `meta-llama/Llama-2-7b-chat-hf` | 解码器 | Llama 2（需授权） | 英文为主多语言 |

```python
# 加载中文 RoBERTa（推荐用于理解类任务）
from transformers import AutoTokenizer, AutoModel

model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 对比 BERT 和 RoBERTa 的分词结果
text = "自然语言处理"
print(f"原始: {text}")

bert_tokens = AutoTokenizer.from_pretrained("bert-base-chinese").tokenize(text)
roberta_tokens = tokenizer.tokenize(text)
print(f"BERT分词:    {berta_tokens}")
print(f"RoBERTa分词: {roberta_tokens}")
```

## 六、Tokenizer 详解

Tokenizer 是连接文本与模型的桥梁，负责将字符串转换为数字 ID。

```
"我喜欢NLP"
    │
    ▼
┌────────────┐
│ Tokenizer  │  ── 词表映射
└─────┬──────┘
      │
      ├─→ input_ids:        [101, 2769, 1599, 3614, 100243, 102]
      ├─→ token_type_ids:   [0,   0,    0,    0,    0,      0]
      └─→ attention_mask:   [1,   1,    1,    1,    1,      1]
```

### 6.1 encode 与 decode

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

text = "深度学习改变世界"

# === 编码：文本 → ID ===
# 方式1：分步操作
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"文本:   {text}")
print(f"分词:   {tokens}")
print(f"ID:     {ids}")

# 方式2：一步到位（会自动加 [CLS] 和 [SEP]）
encoded = tokenizer.encode(text)
print(f"encode: {encoded}")   # 包含特殊 token
print(f"解码:   {tokenizer.decode(encoded)}")
print(f"decode(ids): {tokenizer.decode(ids)}")  # 不含特殊 token

# 方式3：调用 __call__（推荐，返回字典）
outputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
print(f"\n__call__ 输出:")
for key, value in outputs.items():
    print(f"  {key}: {value}")
```

### 6.2 特殊 Token

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# BERT 特殊 token
print("特殊 Token：")
print(f"  [CLS] (开头): {tokenizer.cls_token} → ID: {tokenizer.cls_token_id}")
print(f"  [SEP] (结尾): {tokenizer.sep_token} → ID: {tokenizer.sep_token_id}")
print(f"  [PAD] (填充): {tokenizer.pad_token} → ID: {tokenizer.pad_token_id}")
print(f"  [UNK] (未知): {tokenizer.unk_token} → ID: {tokenizer.unk_token_id}")
print(f"  [MASK](掩码): {tokenizer.mask_token} → ID: {tokenizer.mask_token_id}")

# 句子对编码（用于 NLI 等任务）
sentence_a = "今天天气很好"
sentence_b = "适合出去玩"
pair_output = tokenizer(sentence_a, sentence_b)
print(f"\n句子对编码:")
print(f"  input_ids:      {pair_output['input_ids']}")
print(f"  token_type_ids: {pair_output['token_type_ids']}")  # 0表示句子A，1表示句子B

# 添加自定义 token
num_added = tokenizer.add_tokens(["深度学习", "神经网络"])
print(f"\n添加了 {num_added} 个新 token")
print(f"新词表大小: {len(tokenizer)}")

# 模型需要 resize embedding
# model.resize_token_embeddings(len(tokenizer))
```

### 6.3 批量处理与对齐

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

sentences = [
    "短句。",
    "这是一个中等长度的句子，包含更多信息。",
    "这句话很长很长很长很长很长很长很长很长很长。"
]

# 自动 padding 到同一长度
batch = tokenizer(
    sentences,
    padding=True,          # 自动补齐
    truncation=True,       # 超长截断
    max_length=20,         # 最大长度
    return_tensors="pt"    # 返回 PyTorch 张量
)

print("批量编码结果：")
print(f"input_ids 形状: {batch['input_ids'].shape}")
for i, sent in enumerate(sentences):
    ids = batch['input_ids'][i]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"  句子{i+1}: {tokens}")
```

## 七、完整 Pipeline 使用示例

以下是一个完整示例：使用 HuggingFace 构建一个中文情感分析系统。

```python
"""
完整示例：中文情感分析 pipeline
演示：模型加载 → 批量推理 → 结果可视化
"""
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# === 步骤1：指定模型 ===
model_name = "uer/roberta-base-finetuned-chinanews-chinese"

# === 步骤2：加载分词器和模型 ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print("=== 模型信息 ===")
print(f"模型: {model_name}")
print(f"类别: {model.config.id2label}")

# === 步骤3：创建 pipeline ===
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=0  # 使用 GPU（设为 -1 用 CPU）
)

# === 步骤4：批量推理 ===
texts = [
    "这家店的服务态度非常好，下次还会来。",
    "质量太差了，用���一���就坏了。",
    "总体还可以，性价比一般。",
    "强烈推荐！超出预期！",
    "物流太慢，等了一周才到。"
]

print("\n=== 情感分析结果 ===")
results = sentiment_pipe(texts)

for text, result in zip(texts, results):
    label = "正面" if result['label'] == 'positive' else "负面"
    bar = "█" * int(result['score'] * 20)
    print(f"  [{label}] {result['score']:.4f} {bar}")
    print(f"        \"{text}\"")

# === 步骤5：保存模型到本地 ===
save_path = "./sentiment_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n模型已保存到: {save_path}")
```

## 八、HuggingFace Hub 搜索与选择

```python
from huggingface_hub import HfApi

api = HfApi()

# 搜索中文情感分析模型
models = api.list_models(
    filter="chinese",
    sort="downloads",
    direction=-1,
    limit=10
)

print("热门中文模型 Top 10：")
for i, m in enumerate(models, 1):
    print(f"  {i:2d}. {m.modelId:45s} 下载量: {m.downloads:>8}")
```

## 小结

| 要点 | 内容 |
|------|------|
| 生态系统 | Transformers（模型）+ Datasets（数据）+ Tokenizers（分词）+ Accelerate（加速）|
| pipeline() | 最高层 API，一行代码完成 NLP 任务 |
| AutoModel/AutoTokenizer | 自动匹配模型架构，无需手动指定类名 |
| Hub 平台 | 模型/数据集中央仓库，支持上传分享 |
| 中文模型 | bert-base-chinese（基础）、chinese-roberta-wwm-ext（推荐）、ChatGLM（对话）|
| Tokenizer | 文本与模型的桥梁：encode/decode/特殊 token |
| 批量处理 | padding + truncation + return_tensors |

```
┌─────────────────────────────────────────────────────────────┐
│                   快速选择指南                                │
├──────────────────┬──────────────────────────────────────────┤
│  零基础快速上手   │ pipeline() 一行代码                      │
│  需要自定义控制   │ AutoTokenizer + AutoModel               │
│  需要微调训练     │ AutoModelForXxx + Trainer               │
│  需要分享模型     │ HfApi.upload_folder()                   │
│  理解类任务       │ BERT / RoBERTa（编码器架构）             │
│  生成类任务       │ GPT / LLaMA（解码器架构）                │
│  翻译/摘要        │ T5 / BART（编码器-解码器架构）           │
└──────────────────┴──────────────────────────────────────────┘
```

---

| [← 回到目录](../README.md) | [上一篇：T5与编码器-解码器](../07-Transformer与大模型篇/05-T5与编码器-解码器.md) | [下一篇：模型微调](02-模型微调.md) |
|---|---|---|
