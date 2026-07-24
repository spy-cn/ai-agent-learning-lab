# 03 - 常见问题FAQ

本章汇总了NLP学习与实践中最常见的25个问题，覆盖模型选型、训练技巧、推理优化、工程实践等多个方面，每个问题给出简洁解答并附有详细章节链接。

---

## 基础概念

### Q1: 词向量和BERT的区别是什么？

**答**：词向量（Word2Vec/GloVe）是**静态**的，一个词只有一个固定的向量表示，无法区分多义词。BERT是**动态**的上下文相关表示，同一个词在不同上下文中会得到不同的向量。

```
静态词向量:                    动态表示(BERT):
"苹果" → [0.23, -0.15, ...]    "吃苹果" → [0.31, 0.12, ...]  (水果)
"苹果手机" → [0.23, -0.15, ...] "苹果手机" → [-0.21, 0.45, ...] (品牌)
    ↑ 完全相同的向量               ↑ 不同上下文产生不同向量
```

详见：[02-词向量与文本表示](../03-基础篇/02-词向量与文本表示.md)

---

### Q2: BERT和ChatGPT的核心区别是什么？

- **架构**：BERT是Encoder-only（双向注意力），ChatGPT是Decoder-only（自回归）
- **训练目标**：BERT是MLM（掩码预测），ChatGPT是LM（下一个token预测）
- **擅长任务**：BERT擅长**理解**（分类/NER），ChatGPT擅长**生成**（对话/写作）
- **使用方式**：BERT需要微调，ChatGPT可以prompt直接使用

详见：[01-模型速查表](01-模型速查表.md)

---

### Q3: 什么时候用Encoder-Decoder架构（如T5/BART）？

**答**：当任务需要"输入→输出"的映射，且输入和输出是不同的序列时，使用Encoder-Decoder架构效果最好。

| 适用场景 | 示例 |
|----------|------|
| 机器翻译 | 中文 → 英文 |
| 文本摘要 | 长文本 → 短摘要 |
| 问答生成 | 问题+上下文 → 答案 |
| 文本改写 | 原文 → 改写后文本 |

详见：[01-模型速查表](01-模型速查表.md) Encoder-Decoder部分

---

## 模型训练

### Q4: 微调时应该冻结哪些层？

**答**：根据数据量和任务相似度决定：

```python
# 策略选择
if 数据量 < 1000 and 任务相似:
    # 冻结所有BERT层，只训练分类头
    for param in model.bert.parameters():
        param.requires_grad = False

elif 数据量 < 5000:
    # 冻结底层（embedding + 前6层），微调上层
    for name, param in model.bert.named_parameters():
        if "encoder.layer" in name:
            layer_num = int(name.split(".")[3])
            if layer_num < 6:  # 冻结前6层
                param.requires_grad = False

else:
    # 全量微调（推荐）
    # 默认全部可训练，无需额外操作
    pass
```

| 策略 | 适用场景 | 训练速度 | 效果 |
|------|----------|----------|------|
| 冻结全部BERT | < 1000条数据 | 最快 | 较差 |
| 冻结底层 | 1000-5000条 | 快 | 尚可 |
| 全量微调 | > 5000条 | 慢 | **最好** |

详见：[06-预训练模型微调](../05-进阶篇/06-预训练模型微调.md)

---

### Q5: 怎么防止灾难性遗忘（Catastrophic Forgetting）？

**答**：微调过程中模型逐渐遗忘预训练知识的现象。

| 方法 | 实现 | 适用 |
|------|------|------|
| 小学习率 | lr=2e-5 或更小 | 所有场景 |
| 渐进解冻 | 逐层解冻训练 | 大模型 |
| 混合数据 | 预训练数据+微调数据混合 | 有预训练数据时 |
| LoRA/Adapter | 只训练少量参数 | 大模型首选 |
| 正则化 | 增加KL散度约束 | 学术研究 |

详见：[06-预训练模型微调](../05-进阶篇/06-预训练模型微调.md)

---

### Q6: LoRA应该设置多大rank？

```python
from peft import LoraConfig

# 经验法则
lora_config = LoraConfig(
    r=8,              # rank: 默认8，一般4-16足够
    lora_alpha=16,    # alpha: 通常设为 r×2
    target_modules=["q_proj", "v_proj"],  # 应用的目标模块
    lora_dropout=0.1,
)
```

| Rank (r) | 可训练参数 | 适用场景 |
|----------|-----------|----------|
| r=4 | 最少 | 简单任务（分类），快速实验 |
| r=8 | 适中 | **通用推荐**，大多数任务够用 |
| r=16 | 较多 | 复杂任务（生成/指令微调） |
| r=32+ | 最多 | 接近全量微调效果，性价比下降 |

详见：[09-高效微调](../05-进阶篇/09-高效微调.md)

---

### Q7: 多GPU训练怎么做？

```python
# 方法1: HuggingFace Trainer（最简单）
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    # 自动检测多GPU，无需额外配置
)

# 方法2: PyTorch DDP（灵活）
# 启动命令
# torchrun --nproc_per_node=4 train.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])

# 方法3: DeepSpeed（大规模）
# deepspeed train.py --deepspeed ds_config.json
```

详见：[10-大规模训练与部署](../05-进阶篇/10-大规模训练与部署.md)

---

## 推理优化

### Q8: 推理太慢怎么办？（量化/ONNX/vLLM）

**答**：按需求选择优化方案：

```python
# 方案1: 量化 (INT8) - BERT等小模型
from optimum.onnxruntime import ORTModelForSequenceClassification
model = ORTModelForSequenceClassification.from_pretrained(
    "model_path", export=True, provider="CUDAExecutionProvider"
)

# 方案2: ONNX导出 - 通用加速
# python -m transformers.onnx --model=model_path output/

# 方案3: vLLM - 大模型推理
# python -m vllm.entrypoints.api_server --model Qwen/Qwen2-7B
```

| 方法 | 加速比 | 精度损失 | 适用 |
|------|--------|----------|------|
| INT8量化 | 2x | < 1% | BERT等 |
| ONNX | 2-3x | 无 | 通用 |
| TensorRT | 3-5x | < 1% | NVIDIA GPU |
| vLLM | 10-30x(吞吐) | 无 | 大模型 |

详见：[10-大规模训练与部署](../05-进阶篇/10-大规模训练与部署.md)

---

### Q9: 显存不够怎么办？（OOM问题）

**答**：按紧急程度排序：

| 方法 | 显存节省 | 代码改动 | 副作用 |
|------|----------|----------|--------|
| 减小batch_size | 50% | 1行 | 训练可能不稳定 |
| 梯度累积 | 等效增大batch | 2行 | 无 |
| FP16混合精度 | 50% | 1行 | 几乎无 |
| 梯度检查点 | 60% | 1行 | 训练变慢20% |
| INT8量化 | 75% | 少量 | 可能轻微精度损失 |
| LoRA | 90%+ | 少量 | 适配器训练 |
| 换小模型 | 按比例 | 1行 | 效果可能下降 |

```python
# 组合使用（最有效）
training_args = TrainingArguments(
    per_device_train_batch_size=1,       # 极小batch
    gradient_accumulation_steps=32,       # 累积到等效batch=32
    fp16=True,                           # 混合精度
    gradient_checkpointing=True,          # 梯度检查点
)
```

详见：[10-大规模训练与部署](../05-进阶篇/10-大规模训练与部署.md)

---

### Q10: 如何处理长文本（超过512/4096 tokens）？

**答**：

| 策略 | 方法 | 代码 |
|------|------|------|
| 截断 | 取前512 tokens | `tokenizer(text, truncation=True, max_length=512)` |
| 滑动窗口 | 分段处理，每段独立 | 循环切片文本 |
| 拼接策略 | 取头+尾 | `text[:256] + text[-256:]` |
| 长上下文模型 | Longformer/BigBird | 替换模型即可 |
| 层次化池化 | 分块Embedding后聚合 | 自定义Pooling |

```python
# 滑动窗口处理长文本
def process_long_text(text, model, tokenizer, window=512, stride=256):
    tokens = tokenizer.encode(text)
    results = []
    for start in range(0, len(tokens), stride):
        window_tokens = tokens[start:start+window]
        output = model(torch.tensor([window_tokens]))
        results.append(output)
    return torch.mean(torch.stack(results), dim=0)  # 平均池化
```

详见：[06-预训练模型微调](../05-进阶篇/06-预训练模型微调.md) 长文本处理部分

---

## 模型选型

### Q11: 中文NLP用什么预训练模型？

**答**：按任务推荐：

- **分类/NER**：`hfl/chinese-roberta-wwm-ext`（全词遮掩，效果最好）
- **生成**：`Qwen/Qwen2-7B` 或 `THUDM/chatglm3-6b`（中文生成质量高）
- **Embedding**：`BAAI/bge-large-zh-v1.5`（中文语义检索SOTA）
- **轻量级**：`hfl/chinese-bert-wwm-ext`（均衡选择）

详见：[01-模型速查表](01-模型速查表.md)

---

### Q12: 如何选择分词器（Tokenizer）？

**答**：

| 类型 | 代表 | 优点 | 缺点 |
|------|------|------|------|
| BPE | GPT-2/LLaMA | 无OOV问题 | 中文效率低 |
| WordPiece | BERT | 粒度适中 | 需特殊处理中文 |
| SentencePiece | T5/LLaMA | 语言无关 | 模型稍大 |
| 词级别 | jieba分词 | 中文友好 | OOV问题 |

**关键原则：分词器必须和预训练模型匹配！**

```python
# 正确做法：使用与模型匹配的分词器
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# 不要混用不同模型的分词器
```

详见：[03-Transformer详解](../04-核心篇/03-Transformer详解.md)

---

### Q13: 如何选择Embedding模型？

| 场景 | 推荐模型 | 维度 | 原因 |
|------|----------|------|------|
| 中文语义搜索 | bge-large-zh-v1.5 | 1024 | 中文SOTA |
| 多语言场景 | bge-m3 | 1024 | 支持100+语言 |
| 轻量中文 | m3e-base | 768 | 速度快，效果可接受 |
| 英文为主 | text-embedding-3-large | 3072 | OpenAI最强 |
| 快速原型 | all-MiniLM-L6-v2 | 384 | 超轻量 |

详见：[01-模型速查表](01-模型速查表.md) Embedding部分

---

## RAG与生成

### Q14: RAG和微调有什么区别？什么时候用哪个？

```
RAG (检索增强生成):
  优点: 知识实时更新，可溯源，幻觉少
  缺点: 需要维护知识库，延迟较高
  适用: 知识密集型问答、客服、企业搜索

微调 (Fine-tuning):
  优点: 延迟低，专精某任务，知识内化
  缺点: 知识固定，需要重新训练更新
  适用: 特定风格生成、专业领域分类、NER
```

**选择指南**：
- 需要外部知识 + 时效性 → RAG
- 需要特定风格 + 低延迟 → 微调
- 两者结合 → **RAFT**（微调模型理解检索结果）

详见：[10-RAG技术](../11-高级应用篇/10-RAG技术.md)

---

### Q15: LLM幻觉（Hallucination）怎么解决？

**答**：

| 方法 | 效果 | 实现难度 |
|------|------|----------|
| RAG检索增强 | ★★★★★ | 中 |
| Prompt约束（"不知道就说不知道"） | ★★★ | 低 |
| CoT思维链推理 | ★★★ | 低 |
| Self-Consistency（多次采样投票） | ★★★★ | 中 |
| 事实校验模块 | ★★★★ | 高 |
| RLHF对齐训练 | ★★★★★ | 极高 |

```python
# 简单的防幻觉Prompt
ANTI_HALLUCINATION_PROMPT = """请基于以下知识回答问题。
如果知识库中没有相关信息，请明确回答"根据现有资料，我无法回答这个问题"，
不要编造或臆测信息。

知识库: {context}
问题: {question}
"""
```

详见：[10-RAG技术](../11-高级应用篇/10-RAG技术.md)

---

### Q16: Prompt Engineering vs Fine-tuning 如何选择？

| 维度 | Prompt Engineering | Fine-tuning |
|------|-------------------|-------------|
| 成本 | 低（无需训练） | 高（需要数据和GPU） |
| 灵活性 | 高（随时调整） | 低（需重新训练） |
| 一致性 | 中（受prompt影响） | 高（模型固定） |
| 数据需求 | 0-50条 | 500-10000条 |
| 延迟 | 高（长prompt增加推理） | 低（短输入即可） |
| 适用 | 探索/快速验证 | 生产环境 |

**实践建议**：先用Prompt验证可行性，确定方案后用Fine-tuning固化。

详见：[08-Prompt工程](../11-高级应用篇/08-Prompt工程.md)

---

## 评估与监控

### Q17: 怎么评估生成质量？

**答**：

| 指标 | 含义 | 代码 | 适用任务 |
|------|------|------|----------|
| BLEU | n-gram精确匹配 | `from sacrebleu import corpus_bleu` | 翻译 |
| ROUGE-L | 最长公共子序列 | `from rouge import Rouge` | 摘要 |
| BERTScore | 语义相似度 | `from bert_score import score` | 通用生成 |
| BLEURT | 学习型评估 | `from bleurt import score` | 通用生成 |
| Perplexity | 语言模型困惑度 | `torch.exp(loss)` | 语言模型 |

```python
# BERTScore评估（推荐）
from bert_score import score

P, R, F1 = score(
    [generated_text],
    [reference_text],
    lang="zh",
    model_type="bert-base-chinese"
)
print(f"BERTScore F1: {F1.mean():.4f}")
```

详见：[05-评估与基准](../07-实践篇/05-评估与基准.md)

---

### Q18: 怎么监控模型在线上表现？

**答**：

```python
# 关键监控维度
monitoring_metrics = {
    "模型指标": {
        "推理延迟(p50/p95/p99)": "prometheus + grafana",
        "吞吐量(QPS)": "日志统计",
        "预测分布漂移": "KL散度 vs 训练集分布",
        "置信度分布": "概率分布直方图",
    },
    "业务指标": {
        "用户点击率": "埋点统计",
        "纠错率(用户修改回答)": "对话日志分析",
        "负反馈率": "用户举报/差评",
    }
}
```

详见：[05-端到端NLP项目模板](../12-实战篇/05-端到端NLP项目模板.md)

---

## 数据与工程

### Q19: 数据增强有用吗？有哪些方法？

**答**：**非常有用**，尤其是在数据量 < 2000条时。

| 方法 | 效果 | 实现难度 | 中文支持 |
|------|------|----------|----------|
| EDA（同义词替换） | ★★★ | 低 | 一般 |
| 回译（中→英→中） | ★★★★ | 低 | **好** |
| 随机插入/删除/交换 | ★★ | 低 | 一般 |
| 对抗训练 | ★★★★ | 高 | 好 |
| LLM生成 | ★★★★★ | 中 | **好** |

```python
# 回译数据增强
from transformers import pipeline

# 中 -> 英 -> 中
translator_en = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
translator_zh = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")

def back_translate(text: str) -> str:
    en = translator_en(text)[0]["translation_text"]
    zh = translator_zh(en)[0]["translation_text"]
    return zh

# 使用
augmented = back_translate("这家餐厅的服务非常好")
print(augmented)  # "这家餐厅的服务很棒" (语义保持，表达变化)
```

详见：[03-数据处理与增强](../05-进阶篇/03-数据处理与增强.md)

---

### Q20: 怎么处理领域术语/OOV问题？

**答**：

| 方法 | 适用场景 | 实现 |
|------|----------|------|
| 添加特殊token | OOV术语较少 | `tokenizer.add_tokens(["BLOCKCHAIN", "NFT"])` |
| 词汇扩展预训练 | 大量OOV术语 | 领域自适应预训练 |
| 字级别模型 | 中文通用 | 使用字级别tokenizer |
| 提示注入 | LLM场景 | Prompt中提供术语定义 |

```python
# 添加领域术语
tokenizer.add_tokens(["区块链", "NFT", "智能合约", "DeFi"])
model.resize_token_embeddings(len(tokenizer))

# 对新token的embedding进行初始化
with torch.no_grad():
    model.embeddings.word_embeddings.weight[-4:] = \
        model.embeddings.word_embeddings.weight[:4].clone()
```

详见：[06-预训练模型微调](../05-进阶篇/06-预训练模型微调.md)

---

### Q21: 训练时如何设置batch_size和学习率？

**答**：

```python
# 经验法则
batch_size_rules = {
    "BERT分类": (16, 32),       # batch_size: 16-32
    "BERT NER": (8, 16),        # NER需要更大显存
    "GPT微调": (4, 8),          # 大模型batch宜小
    "LLM全量": (1, 4),          # 配合梯度累积
}

lr_rules = {
    "BERT微调": (2e-5, 5e-5),   # 推荐 2e-5
    "GPT微调": (1e-5, 3e-5),
    "从头训练": (1e-4, 5e-4),
    "LoRA": (1e-4, 5e-4),       # LoRA可用更大学习率
}
```

详见：[06-预训练模型微调](../05-进阶篇/06-预训练模型微调.md)

---

### Q22: 如何做模型蒸馏（Knowledge Distillation）？

**答**：将大模型（Teacher）的知识迁移到小模型（Student）。

```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels,
                      temperature=4.0, alpha=0.5):
    """
    alpha: 蒸馏损失权重 (0-1)
    temperature: 软化概率分布
    """
    # 硬标签损失（真实标签）
    hard_loss = F.cross_entropy(student_logits, labels)

    # 软标签损失（教师输出）
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
    soft_loss *= temperature ** 2  # 缩放

    return alpha * hard_loss + (1 - alpha) * soft_loss
```

| Teacher | Student | 压缩比 | 精度保留 |
|---------|---------|--------|----------|
| BERT-large | BERT-base | 3x | ~99% |
| BERT-base | DistilBERT | 2x | ~97% |
| Llama-70B | Llama-7B | 10x | ~95% |

详见：[09-高效微调](../05-进阶篇/09-高效微调.md)

---

## 实战部署

### Q23: 多轮对话的上下文如何管理？

**答**：

```python
class ContextManager:
    def __init__(self, max_tokens=4096):
        self.max_tokens = max_tokens

    def trim_context(self, messages: list) -> list:
        """智能裁剪对话历史"""
        total = 0
        trimmed = []
        # 从最新消息开始保留
        for msg in reversed(messages):
            tokens = len(tokenizer.encode(msg["content"]))
            if total + tokens > self.max_tokens:
                break
            trimmed.insert(0, msg)
            total += tokens
        return trimmed

    def summarize_old_messages(self, messages: list, llm) -> list:
        """将旧对话压缩为摘要"""
        if len(messages) <= 6:
            return messages

        old = messages[:-4]  # 前4轮之前的消息
        recent = messages[-4:]

        summary_prompt = f"请简洁总结以下对话：{str(old)}"
        summary = llm.invoke(summary_prompt).content

        return [{"role": "system", "content": f"对话摘要：{summary}"}] + recent
```

详见：[02-智能客服机器人](../12-实战篇/02-智能客服机器人.md)

---

### Q24: Docker中如何管理模型版本？

**答**：

```yaml
# docker-compose.yml 多模型服务编排
version: '3.8'
services:
  sentiment-api:
    image: nlp-sentiment:v2.1
    environment:
      - MODEL_VERSION=v2.1
      - MODEL_PATH=/models/sentiment_v2.1
    volumes:
      - ./models:/models:ro
    ports:
      - "8001:8000"

  search-api:
    image: nlp-search:v1.0
    volumes:
      - ./models/embedding:/models:ro
      - ./faiss_index:/index:ro
    ports:
      - "8002:8000"
```

详见：[05-端到端NLP项目模板](../12-实战篇/05-端到端NLP项目模板.md)

---

### Q25: 如何做A/B测试来评估模型上线效果？

**答**：

```python
import random
import hashlib

class ABTestRouter:
    """A/B测试流量路由"""

    def __init__(self):
        self.experiments = {
            "sentiment_v2": {
                "control": "model_v1",     # 对照组
                "treatment": "model_v2",   # 实验组
                "traffic_split": 0.5,      # 50%流量到实验组
            }
        }

    def route(self, user_id: str, experiment: str) -> str:
        """根据用户ID确定性路由"""
        exp = self.experiments[experiment]
        # 哈希分桶确保同一用户始终在同一组
        hash_val = int(hashlib.md5(
            f"{user_id}_{experiment}".encode()
        ).hexdigest(), 16) % 100

        if hash_val < exp["traffic_split"] * 100:
            return exp["treatment"]
        return exp["control"]

    def analyze(self, control_metrics, treatment_metrics):
        """分析A/B测试结果"""
        from scipy import stats

        t_stat, p_value = stats.ttest_ind(
            control_metrics, treatment_metrics
        )

        print(f"对照组均值: {np.mean(control_metrics):.4f}")
        print(f"实验组均值: {np.mean(treatment_metrics):.4f}")
        print(f"P-value: {p_value:.4f}")

        if p_value < 0.05:
            effect = (np.mean(treatment_metrics) - np.mean(control_metrics))
            effect_pct = effect / np.mean(control_metrics) * 100
            print(f"显著提升: {effect_pct:+.1f}%")
```

详见：[05-端到端NLP项目模板](../12-实战篇/05-端到端NLP项目模板.md)

---

### 其他常见问题速览

| # | 问题 | 简短回答 | 详见 |
|---|------|----------|------|
| 26 | 如何安装HuggingFace模型到离线环境？ | `model.save_pretrained("./local")` 保存后拷贝 | [模型速查表](01-模型速查表.md) |
| 27 | 训练loss不下降怎么办？ | 检查数据、降低学习率、检查梯度 | [预训练微调](../05-进阶篇/06-预训练模型微调.md) |
| 28 | 如何提升小模型效果？ | 蒸馏、数据增强、模型集成 | [高效微调](../05-进阶篇/09-高效微调.md) |
| 29 | 训练多久算过拟合？ | 验证集loss开始上升时 | [评估与基准](../07-实践篇/05-评估与基准.md) |
| 30 | 文本分类类别不均衡怎么办？ | Focal Loss/重采样/类别加权 | [情感分析系统](../12-实战篇/01-情感分析系统.md) |

---

## 小结

本章汇总了NLP学习与实践中最高频的25+个问题及其解决方案。

| 要点 | 内容 |
|------|------|
| **模型选型** | 理解用BERT系，生成用GPT系，中文选专用模型 |
| **训练技巧** | 小数据冻结，大学习率用LoRA，防遗忘用小lr |
| **推理优化** | 量化+ONNX+vLLM三件套，batch_size/梯度累积解OOM |
| **RAG vs 微调** | 需要外部知识用RAG，需要风格固化用微调 |
| **幻觉问题** | RAG最有效，Prompt约束最简单 |
| **评估监控** | BERTScore评生成，Prometheus+Grafana监控 |
| **工程实践** | 分层配置+实验管理+CI/CD是生产级项目标配 |

---

| [← 回到目录](../README.md) | [上一篇：02-任务选择指南](02-任务选择指南.md) | [下一篇：04-推荐资源](04-推荐资源.md) |
|---|---|---|
