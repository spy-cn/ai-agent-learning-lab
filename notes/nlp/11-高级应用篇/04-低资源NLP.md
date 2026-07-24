# 低资源NLP

低资源NLP（Low-resource NLP）研究在标注数据极其稀缺的场景下建立有效的NLP模型。这一方向对中文方言、少数民族语言、垂直领域（医疗、法律）的NLP应用至关重要。本章系统介绍数据增强、迁移学习、Few-shot/Zero-shot Learning、主动学习和跨语言迁移等核心方法。

## 低资源NLP的挑战

```
┌─────────────────────────────────────────────────────────────────────┐
│                     低资源NLP面临的核心挑战                           │
├───────────────────┬─────────────────────────────────────────────────┤
│  数据稀缺           │  标注样本极少（几十到几百条），甚至零标注        │
│  类别不平衡         │  少量正例 vs 大量负例，长尾分布严重             │
│  领域偏差           │  通用语料训练的模型在垂直领域泛化差              │
│  语言多样性         │  方言、古汉语、少数民族语言缺乏标注资源          │
│  过拟合风险         │  小数据集上大模型极易过拟合                     │
│  评估困难           │  测试集太小导致评估方差大                       │
└───────────────────┴─────────────────────────────────────────────────┘
```

**典型低资源场景**：

| 场景 | 标注量级 | 示例 |
|------|----------|------|
| 极少样本（Few-shot） | 每类5-50条 | 新情感类别分类 |
| 零样本（Zero-shot） | 0条 | 未见过的实体类型 |
| 弱监督（Weakly-supervised） | 知识库自动标注 | 远程监督关系抽取 |
| 跨领域迁移 | 目标域少量/无标注 | 通用→医疗领域适配 |
| 跨语言迁移 | 目标语言无标注 | 英文训练→中文推理 |

## 数据增强方法

在标注数据有限时，数据增强是扩大训练集的最直接手段。

```
┌─────────────────────────────────────────────────────────────────────┐
│                      数据增强方法体系                                 │
├───────────────┬─────────────────────────────────────────────────────┤
│  EDA           │  同义词替换 / 随机插入 / 随机交换 / 随机删除          │
│  (Easy Data    │  简单高效，适合文本分类，但可能改变语义               │
│   Augmentation)│                                                      │
├───────────────┼─────────────────────────────────────────────────────┤
│  回译          │  原文 → 中间语言（如英语）→ 回译到原文语言            │
│  (Back-        │  生成语义相似但表达不同的文本                         │
│   Translation) │  依赖翻译模型质量                                    │
├───────────────┼─────────────────────────────────────────────────────┤
│  上下文替换    │  用语言模型根据上下文生成替换词                        │
│  (Contextual   │  比同义词替换更灵活，保持上下文一致性                  │
│   Augment)    │  如使用BERT做masked token预测                        │
├───────────────┼─────────────────────────────────────────────────────┤
│  LLM生成       │  用GPT等模型根据少量标注样本批量生成相似样本           │
│  (LLM-based)  │  可生成高质量多样文本，但成本较高                      │
├───────────────┼─────────────────────────────────────────────────────┤
│  MixUp/        │  在嵌入空间插值：x̃ = λx_i + (1-λ)x_j               │
│  CutMix        │  文本领域较难直接应用，NLP中多用Manifold Mixup       │
└───────────────┴─────────────────────────────────────────────────────┘
```

### EDA数据增强实现

```python
import random
import jieba
from typing import List
from collections import Counter

class EDAAugmenter:
    """EDA (Easy Data Augmentation) 中文文本数据增强"""
    
    def __init__(self, synonym_dict=None, alpha_sr=0.1, alpha_ri=0.1, 
                 alpha_rs=0.1, alpha_rd=0.1, num_aug=4):
        """
        Args:
            synonym_dict: 同义词词典 {word: [synonyms]}
            alpha_sr: 同义词替换比例
            alpha_ri: 随机插入比例
            alpha_rs: 随机交换比例
            alpha_rd: 随机删除比例
            num_aug: 每条文本生成的增强样本数
        """
        self.synonym_dict = synonym_dict or self._default_synonyms()
        self.alphas = {
            "sr": alpha_sr,  # synonym replacement
            "ri": alpha_ri,  # random insertion
            "rs": alpha_rs,  # random swap
            "rd": alpha_rd,  # random deletion
        }
        self.num_aug = num_aug
    
    def _default_synonyms(self):
        """内置简单同义词词典"""
        return {
            "好": ["棒", "优秀", "不错", "出色"],
            "坏": ["差", "糟糕", "恶劣", "不好"],
            "喜欢": ["喜爱", "钟爱", "偏好"],
            "开心": ["高兴", "快乐", "愉悦", "欣喜"],
            "难过": ["悲伤", "伤心", "痛苦", "忧愁"],
        }
    
    def augment(self, text: str) -> List[str]:
        """生成num_aug个增强文本"""
        words = list(jieba.cut(text))
        augmented = []
        
        for _ in range(self.num_aug):
            op = random.choice(["sr", "ri", "rs", "rd"])
            
            if op == "sr":
                aug = self._synonym_replacement(words.copy())
            elif op == "ri":
                aug = self._random_insertion(words.copy())
            elif op == "rs":
                aug = self._random_swap(words.copy())
            else:
                aug = self._random_deletion(words.copy())
            
            augmented.append("".join(aug))
        
        return augmented
    
    def _synonym_replacement(self, words: List[str]) -> List[str]:
        """同义词替换"""
        n = max(1, int(len(words) * self.alphas["sr"]))
        replaceable = [i for i, w in enumerate(words) if w in self.synonym_dict]
        
        if not replaceable:
            return words
        
        for idx in random.sample(replaceable, min(n, len(replaceable))):
            synonyms = self.synonym_dict[words[idx]]
            words[idx] = random.choice(synonyms)
        
        return words
    
    def _random_insertion(self, words: List[str]) -> List[str]:
        """随机插入同义词"""
        n = max(1, int(len(words) * self.alphas["ri"]))
        
        for _ in range(n):
            insert_pos = random.randint(0, len(words) - 1)
            # 找一个可替换的词的同义词来插入
            candidates = [w for w in words if w in self.synonym_dict]
            if candidates:
                syn_word = random.choice(candidates)
                synonym = random.choice(self.synonym_dict[syn_word])
                words.insert(insert_pos, synonym)
            else:
                words.insert(insert_pos, words[random.randint(0, len(words)-1)])
        
        return words
    
    def _random_swap(self, words: List[str]) -> List[str]:
        """随机交换相邻词的位置"""
        n = max(1, int(len(words) * self.alphas["rs"]))
        
        for _ in range(n):
            if len(words) >= 2:
                idx = random.randint(0, len(words) - 2)
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
        
        return words
    
    def _random_deletion(self, words: List[str]) -> List[str]:
        """随机删除词"""
        if len(words) <= 1:
            return words
        
        n = max(1, int(len(words) * self.alphas["rd"]))
        for _ in range(n):
            if len(words) > 1:
                del words[random.randint(0, len(words) - 1)]
        
        return words


# 测试EDA
augmenter = EDAAugmenter(num_aug=4)
sample = "这部电影真的很好看我很喜欢"
augmented_texts = augmenter.augment(sample)

print(f"原文: {sample}")
print(f"\n增强文本 ({len(augmented_texts)} 条):")
for i, aug in enumerate(augmented_texts, 1):
    print(f"  {i}. {aug}")
```

### 回译数据增强

```python
"""
回译数据增强
需要安装: pip install transformers sentencepiece
"""

from transformers import MarianMTModel, MarianTokenizer

class BackTranslator:
    """使用MarianMT进行回译数据增强"""
    
    def __init__(self, src_lang="zh", pivot_lang="en"):
        """
        Args:
            src_lang: 源语言
            pivot_lang: 中间语言（回译的桥梁）
        """
        # 中文 → 英文模型
        self.zh2en_model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{pivot_lang}"
        self.zh2en_tokenizer = MarianTokenizer.from_pretrained(self.zh2en_model_name)
        self.zh2en_model = MarianMTModel.from_pretrained(self.zh2en_model_name)
        
        # 英文 → 中文模型
        self.en2zh_model_name = f"Helsinki-NLP/opus-mt-{pivot_lang}-{src_lang}"
        self.en2zh_tokenizer = MarianTokenizer.from_pretrained(self.en2zh_model_name)
        self.en2zh_model = MarianMTModel.from_pretrained(self.en2zh_model_name)
    
    def translate(self, text, tokenizer, model, max_length=128):
        """翻译单条文本"""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs, max_length=max_length)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    
    def back_translate(self, text):
        """回译：中文 → 英文 → 中文"""
        # 前向翻译
        en_text = self.translate(text, self.zh2en_tokenizer, self.zh2en_model)
        # 反向翻译
        back_zh = self.translate(en_text, self.en2zh_tokenizer, self.en2zh_model)
        return back_zh
    
    def augment(self, texts, num_variants=1):
        """对多条文本进行回译增强"""
        augmented = []
        for text in texts:
            for _ in range(num_variants):
                aug = self.back_translate(text)
                augmented.append(aug)
        return augmented


# 使用示例（需要下载模型）
print("回译数据增强器已定义")
print("模型下载可能需要几分钟时间")
print("安装: pip install transformers sentencepiece")
```

## 迁移学习策略

```
┌─────────────────────────────────────────────────────────────────────┐
│                      低资源迁移学习策略                               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────┐            │
│  │  策略1: 预训练 + 微调 (Pre-train + Fine-tune)       │            │
│  │                                                      │            │
│  │  大规模预训练模型 ──→ 少量标注数据微调 ──→ 目标模型    │            │
│  │  (BERT/GPT/Llama)     (几百-几千条)                  │            │
│  │  - 适用: 目标领域与预训练语料有一定相似度               │            │
│  │  - 技巧: 低学习率、早停、逐层解冻                      │            │
│  └─────────────────────────────────────────────────────┘            │
│                           │                                          │
│                           ▼                                          │
│  ┌─────────────────────────────────────────────────────┐            │
│  │  策略2: 领域自适应 (Domain Adaptation)               │            │
│  │                                                      │            │
│  │  源域 (丰富标注) ──→ 适配 ──→ 目标域 (少量/无标注)    │            │
│  │  如: 新闻文本 → 医疗文本                               │            │
│  │  - 继续预训练(continued pre-training): 在目标域无标注  │            │
│  │    文本上继续MLM训练                                  │            │
│  │  - 对抗训练: 域分类器引导学习域不变特征                 │            │
│  │  - 数据选择: 从源域中选择与目标域最相似的子集           │            │
│  └─────────────────────────────────────────────────────┘            │
│                           │                                          │
│                           ▼                                          │
│  ┌─────────────────────────────────────────────────────┐            │
│  │  策略3: 多任务学习 (Multi-task Learning)             │            │
│  │  同时学习多个相关任务，共享底层表示                     │            │
│  │  例: 情感分析 + 主题分类 + 讽刺检测一起训练            │            │
│  └─────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

### 低资源微调技巧

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset

class LowResourceFineTuningTips:
    """低资源场景下的微调技巧总结"""
    
    @staticmethod
    def create_low_resource_dataset(texts, labels, tokenizer, max_len=128):
        """创建小样本数据集"""
        
        class SmallDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_len,
                    return_tensors="pt"
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": torch.tensor(self.labels[idx], dtype=torch.long)
                }
        
        return SmallDataset(texts, labels, tokenizer, max_len)
    
    @staticmethod
    def get_training_args(output_dir="./results", num_epochs=20):
        """获取适合低资源的训练参数"""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,   # 小batch防止过拟合
            per_device_eval_batch_size=8,
            learning_rate=2e-5,               # 低学习率
            warmup_ratio=0.1,
            weight_decay=0.01,                # L2正则化
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # 低资源关键配置
            gradient_accumulation_steps=4,    # 梯度累积模拟大batch
            fp16=torch.cuda.is_available(),
        )


# 低资源微调最佳实践
tips = {
    "学习率": "使用较小的学习率（1e-5 ~ 5e-5），避免破坏预训练权重",
    "Batch Size": "小batch（4-8）+ 梯度累积，增加更新频率",
    "Epoch数": "允许更多epoch（15-30），但使用早停",
    "正则化": "增加weight_decay(0.01-0.1) + dropout(0.3-0.5)",
    "逐层解冻": "先冻结底层，只训练分类头 → 逐层解冻微调",
    "数据增强": "每个样本做2-4倍EDA/回译增强",
    "评估策略": "使用交叉验证（如5-fold），减少单次划分偏差",
    "混合精度": "使用FP16减少显存占用，允许更大模型",
}

print("低资源微调最佳实践:")
for k, v in tips.items():
    print(f"  [{k}]: {v}")
```

## Few-shot Learning

### Prototypical Networks

```
┌─────────────── Prototypical Networks ───────────────┐
│                                                       │
│  核心思想: 为每个类别学习一个原型向量（prototype），     │
│            新样本通过到最近原型的距离分类                │
│                                                       │
│  训练阶段 (Episode-based):                             │
│  ┌─────────────────────────────────────────┐          │
│  │  Support Set (N way × K shot)           │          │
│  │  Class A: [x_a1, x_a2, ..., x_ak]      │          │
│  │  Class B: [x_b1, x_b2, ..., x_bk]      │          │
│  │  Class C: [x_c1, x_c2, ..., x_ck]      │          │
│  └─────────────────────────────────────────┘          │
│                │                                       │
│                ▼                                       │
│  计算原型: c_k = 1/K Σ f_φ(x_ki)  (均值)               │
│                │                                       │
│                ▼                                       │
│  ┌─────────────────────────────────────────┐          │
│  │  Query Set: 未标注样本                   │          │
│  │  对每个query: p(y=k|x) ∝ -||f_φ(x)-c_k||│          │
│  └─────────────────────────────────────────┘          │
└───────────────────────────────────────────────────────┘
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    """原型网络实现"""
    def __init__(self, encoder, embedding_dim=128):
        super().__init__()
        self.encoder = encoder  # 编码器（如BERT/CNN）
        self.embedding_dim = embedding_dim
    
    def compute_prototypes(self, support_embeddings, support_labels, n_way):
        """计算每个类别的原型向量（均值）"""
        prototypes = []
        
        for c in range(n_way):
            class_mask = (support_labels == c)
            class_embeds = support_embeddings[class_mask]  # (K, D)
            prototype = class_embeds.mean(dim=0)           # (D,)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)  # (n_way, D)
    
    def forward(self, support_set, support_labels, query_set, n_way, k_shot):
        """
        Args:
            support_set: (n_way * k_shot, *) 支持集
            support_labels: (n_way * k_shot,)
            query_set: (n_query, *) 查询集
        Returns:
            query_logits: (n_query, n_way)
        """
        # 编码
        support_embeds = self.encoder(support_set)  # (N_s, D)
        query_embeds = self.encoder(query_set)      # (N_q, D)
        
        # 计算原型
        prototypes = self.compute_prototypes(support_embeds, support_labels, n_way)
        
        # 计算query到各原型的欧氏距离
        # dist[i,j] = ||q_i - p_j||^2
        dists = torch.cdist(query_embeds, prototypes, p=2)  # (N_q, n_way)
        
        # 转换为logits（负距离→高相似度→高logit）
        logits = -dists
        
        return logits
    
    def few_shot_predict(self, support_set, support_labels, query_set, n_way):
        """少样本预测"""
        with torch.no_grad():
            support_embeds = self.encoder(support_set)
            query_embeds = self.encoder(query_set)
            
            prototypes = self.compute_prototypes(support_embeds, support_labels, n_way)
            dists = torch.cdist(query_embeds, prototypes, p=2)
            
            predictions = dists.argmin(dim=-1)  # 最小距离对应的类别
        
        return predictions


# 简化的CNN编码器（用于文本的原型网络）
class TextCNNEncoder(nn.Module):
    """文本CNN编码器（用于原型网络）"""
    def __init__(self, vocab_size, embedding_dim=128, num_filters=64, 
                 filter_sizes=[2,3,4], output_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs) 
            for fs in filter_sizes
        ])
        self.projection = nn.Linear(num_filters * len(filter_sizes), output_dim)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x: (B, L)
        embeds = self.embedding(x).transpose(1, 2)  # (B, D, L)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embeds))      # (B, F, L')
            pooled = F.max_pool1d(conv_out, conv_out.size(-1)).squeeze(-1)  # (B, F)
            conv_outputs.append(pooled)
        
        combined = torch.cat(conv_outputs, dim=1)  # (B, 3*F)
        output = self.projection(self.dropout(combined))  # (B, D_out)
        
        return output

print("原型网络和文本CNN编码器已定义")
```

## SetFit：Sentence Transformers + 少样本分类

SetFit由Hugging Face提出，是当前最流行的少样本文本分类框架。它不需要prompt engineering，仅需少量标注样本。

```
┌───────────────── SetFit 框架 ─────────────────────┐
│                                                     │
│  Step 1: 使用少量标注样本生成训练对                    │
│  ┌────────────────────────────────────────────┐    │
│  │  正例对: 同类别样本两两配对                     │    │
│  │  负例对: 不同类别样本两两配对                   │    │
│  │  (x_i, x_j), y=1 if same_class else 0      │    │
│  └────────────────────────────────────────────┘    │
│                     │                               │
│                     ▼                               │
│  Step 2: 对比微调Sentence Transformer               │
│  ┌────────────────────────────────────────────┐    │
│  │  CosineSimilarityLoss                       │    │
│  │  同类样本嵌入更近，异类样本嵌入更远             │    │
│  └────────────────────────────────────────────┘    │
│                     │                               │
│                     ▼                               │
│  Step 3: 训练分类器                                 │
│  ┌────────────────────────────────────────────┐    │
│  │  用微调后的encoder提取训练集嵌入               │    │
│  │  训练轻量分类器（Logistic Regression）        │    │
│  └────────────────────────────────────────────┘    │
│                     │                               │
│                     ▼                               │
│  Step 4: 推理                                      │
│  ┌────────────────────────────────────────────┐    │
│  │  新样本 → encoder → 嵌入 → 分类器 → 预测标签   │    │
│  └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### 使用SetFit做少样本文本分类

```python
"""
SetFit 少样本文本分类完整示例
安装: pip install setfit datasets
"""

from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset

def setfit_demo():
    """SetFit少样本分类演示"""
    
    # 准备训练数据（模拟小样本场景）
    train_texts = [
        # 正面 (每类仅8个样本)
        "这款手机性能非常出色",
        "物流很快，包装也很精美",
        "性价比很高，推荐购买",
        "用了几天很满意",
        "质量超出预期",
        "客服态度非常好",
        "外观设计很漂亮",
        "功能齐全，操作流畅",
        # 负面
        "非常失望的一次购物",
        "质量太差了",
        "用了三天就坏了",
        "客服根本不回复",
        "和描述完全不一样",
        "快递太慢，包装破损",
        "价格贵还不实用",
        "买了就后悔了",
    ]
    
    train_labels = [1]*8 + [0]*8  # 1=正面, 0=负面
    
    test_texts = [
        "这个产品真的很好用",
        "垃圾产品，别买",
        "还不错，值得入手",
        "太差了，退货",
    ]
    test_labels = [1, 0, 1, 0]
    
    # 创建HuggingFace Dataset
    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "label": train_labels
    })
    
    eval_dataset = Dataset.from_dict({
        "text": test_texts,
        "label": test_labels
    })
    
    # 加载SetFit模型（使用小模型加快演示）
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 配置训练参数
    args = TrainingArguments(
        output_dir="./setfit_results",
        batch_size=4,
        num_epochs=1,           # SetFit一般1-3个epoch就够了
        num_iterations=10,      # 生成训练对的迭代数
        body_learning_rate=2e-5,
        learning_rate=2e-5,
        l2_weight=0.01,
    )
    
    # 训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()
    
    # 评估
    metrics = trainer.evaluate()
    print(f"评估指标: {metrics}")
    
    # 预测
    predictions = model.predict(["这个产品质量很棒", "非常不好用"])
    print(f"预测结果: {predictions}")  # [1, 0] (正面, 负面)
    
    return model, metrics

if __name__ == "__main__":
    print("SetFit少样本文本分类代码就绪")
    print("运行 setfit_demo() 开始训练")
    # model, metrics = setfit_demo()
```

## Zero-shot Learning

零样本学习利用大规模NLI（自然语言推理）预训练模型，将分类任务转化为推理任务。

```
┌───────────── Zero-shot 分类原理 ─────────────────┐
│                                                    │
│  传统分类: "这部电影很好看" → 分类器 → 正面/负面    │
│                                                    │
│  Zero-shot分类 (NLI方式):                           │
│  ┌─────────────────────────────────────────┐       │
│  │  前提: "这部电影很好看"                     │       │
│  │  假设1: "这段文本表达正面情感"              │       │
│  │  假设2: "这段文本表达负面情感"              │       │
│  │                                          │       │
│  │  NLI模型 → 蕴含概率:                      │       │
│  │    假设1: 0.92  ← 最高                     │       │
│  │    假设2: 0.03                             │       │
│  └─────────────────────────────────────────┘       │
│                                                    │
│  预测结果: 正面情感                                 │
└────────────────────────────────────────────────────┘
```

```python
from transformers import pipeline

def zero_shot_classification_demo():
    """零样本文本分类演示"""
    
    # 加载零样本分类pipeline
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1  # CPU
    )
    
    # 定义候选标签
    candidate_labels = [
        "正面评价", "负面评价", "中性评价",
        "产品功能", "物流服务", "价格问题", "客服体验"
    ]
    
    # 测试文本
    texts = [
        "快递两天就到了，包装完好，给个好评",
        "手机用了不到一周屏幕就出问题了，质量太差",
        "客服态度很好，耐心解答了我的所有问题"
    ]
    
    for text in texts:
        result = classifier(text, candidate_labels)
        
        print(f"\n文本: {text}")
        print(f"最佳标签: {result['labels'][0]} (置信度: {result['scores'][0]:.3f})")
        print(f"Top-3 标签:")
        for label, score in zip(result["labels"][:3], result["scores"][:3]):
            print(f"  {label}: {score:.3f}")

if __name__ == "__main__":
    print("零样本文本分类代码就绪")
    # zero_shot_classification_demo()
```

## 主动学习（Active Learning）

主动学习通过人机协作的迭代标注策略，用最少的标注量达到最佳模型性能。

```
┌─────────────── 主动学习循环 ──────────────────────┐
│                                                     │
│  ┌─────────────────────────────────────┐           │
│  │         未标注数据池 (Pool)            │           │
│  │  [x1, x2, x3, ..., xn]              │           │
│  └──────────────┬──────────────────────┘           │
│                 │                                   │
│                 ▼                                   │
│  ┌─────────────────────────────────────┐           │
│  │  采样策略:                              │           │
│  │  - 不确定性采样: 选模型最不确定的样本     │           │
│  │  - 多样性采样:  选与已标注差异最大的样本  │           │
│  │  - 混合策略:    不确定+多样              │           │
│  └──────────────┬──────────────────────┘           │
│                 │                                   │
│                 ▼                                   │
│  ┌─────────────────────────────────────┐           │
│  │  人工标注: 选出K个样本送人工标注       │           │
│  └──────────────┬──────────────────────┘           │
│                 │                                   │
│                 ▼                                   │
│  ┌─────────────────────────────────────┐           │
│  │  模型更新: 用新标注数据更新模型         │           │
│  └──────────────┬──────────────────────┘           │
│                 │                                   │
│                 ▼                                   │
│  满足停止条件? → 否 → 回到采样步骤                   │
│          ↓                                         │
│         是                                          │
│         输出最终模型                                  │
└─────────────────────────────────────────────────────┘
```

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

class ActiveLearner:
    """主动学习器"""
    
    def __init__(self, model=None, strategy="uncertainty", random_state=42):
        self.model = model or LogisticRegression(random_state=random_state)
        self.strategy = strategy
        self.labeled_indices = set()
        self.rng = np.random.RandomState(random_state)
    
    def query_samples(self, X_pool, X_labeled, y_labeled, n_queries=5):
        """
        从未标注池中选择最具信息量的样本
        
        Args:
            X_pool: 未标注数据池
            X_labeled: 已标注数据
            y_labeled: 已标注标签
            n_queries: 每轮选择的样本数
        """
        if self.strategy == "uncertainty":
            return self._uncertainty_sampling(X_pool, n_queries)
        elif self.strategy == "diversity":
            return self._diversity_sampling(X_pool, X_labeled, n_queries)
        elif self.strategy == "hybrid":
            return self._hybrid_sampling(X_pool, X_labeled, n_queries)
        else:
            # 随机采样作为baseline
            return self.rng.choice(len(X_pool), n_queries, replace=False)
    
    def _uncertainty_sampling(self, X_pool, n_queries):
        """不确定性采样：选预测概率最接近0.5的样本"""
        probas = self.model.predict_proba(X_pool)
        # 置信度 = max(probas)，不确定性 = 1 - 置信度
        uncertainty = 1 - np.max(probas, axis=1)
        return np.argsort(uncertainty)[-n_queries:]
    
    def _diversity_sampling(self, X_pool, X_labeled, n_queries):
        """多样性采样：选与已标注集最不相似的样本"""
        if len(X_labeled) == 0:
            return self.rng.choice(len(X_pool), n_queries, replace=False)
        
        # 计算到已标注集的平均距离
        distances = cdist(X_pool, X_labeled, metric="cosine").min(axis=1)
        return np.argsort(distances)[-n_queries:]
    
    def _hybrid_sampling(self, X_pool, X_labeled, n_queries, alpha=0.5):
        """混合采样：结合不确定性和多样性"""
        # 不确定性分数
        probas = self.model.predict_proba(X_pool)
        uncertainty = 1 - np.max(probas, axis=1)
        
        # 多样性分数
        if len(X_labeled) > 0:
            diversity = cdist(X_pool, X_labeled, metric="cosine").min(axis=1)
        else:
            diversity = np.ones(len(X_pool))
        
        # 归一化后加权组合
        u_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
        d_norm = (diversity - diversity.min()) / (diversity.max() - diversity.min() + 1e-8)
        
        combined = alpha * u_norm + (1 - alpha) * d_norm
        return np.argsort(combined)[-n_queries:]
    
    def active_learning_loop(self, X_pool, y_pool, n_init=5, n_queries=5, max_queries=50):
        """
        主动学习主循环
        
        Returns:
            history: 每轮的性能记录
        """
        # 随机初始化标注集
        init_indices = self.rng.choice(len(X_pool), n_init, replace=False)
        X_labeled = X_pool[init_indices].copy()
        y_labeled = y_pool[init_indices].copy()
        
        available = np.setdiff1d(np.arange(len(X_pool)), init_indices)
        
        history = []
        n_rounds = 0
        
        while n_rounds * n_queries < max_queries:
            n_rounds += 1
            
            # 训练模型
            self.model.fit(X_labeled, y_labeled)
            
            # 记录当前性能
            if len(available) > 0:
                X_remaining = X_pool[available]
                y_remaining = y_pool[available]
                y_pred = self.model.predict(X_remaining)
                acc = accuracy_score(y_remaining, y_pred)
                history.append({
                    "round": n_rounds,
                    "labeled": len(y_labeled),
                    "accuracy": acc
                })
            
            # 选择下一批标注样本
            query_indices = self.query_samples(
                X_pool[available], X_labeled, y_labeled, n_queries
            )
            
            # 更新标注集
            selected = available[query_indices]
            X_labeled = np.vstack([X_labeled, X_pool[selected]])
            y_labeled = np.hstack([y_labeled, y_pool[selected]])
            
            available = np.setdiff1d(available, selected)
            
            if len(available) == 0:
                break
        
        return history


# 模拟主动学习
np.random.seed(42)
from sklearn.datasets import make_classification

# 生成模拟数据
X, y = make_classification(n_samples=500, n_features=20, n_classes=2, 
                           n_informative=10, random_state=42)

learner = ActiveLearner(strategy="uncertainty")
history = learner.active_learning_loop(X, y, n_init=10, n_queries=5, max_queries=100)

print("主动学习训练曲线:")
for record in history[::5]:  # 每5轮打印一次
    print(f"  Round {record['round']:2d} | 标注: {record['labeled']:3d} | 准确率: {record['accuracy']:.4f}")
```

## 跨语言迁移

```
┌─────────────────────────────────────────────────────────────────────┐
│                      跨语言迁移范式                                  │
├─────────────────────┬───────────────────────────────────────────────┤
│  多语言预训练模型     │  mBERT / XLM-RoBERTa / mT5 / mGPT           │
│  (Multi-lingual PLM) │  1个模型覆盖100+语言，共享subword词表         │
│                      │  在English上微调 → 直接推理其他语言           │
├─────────────────────┼───────────────────────────────────────────────┤
│  零样本跨语言迁移     │  源语言(英语)训练 → 目标语言(中文/法语)测试    │
│  (Zero-shot Cross-   │  依赖多语言模型的对齐能力                     │
│   lingual Transfer)  │  简单但效果不如目标语言微调                    │
├─────────────────────┼───────────────────────────────────────────────┤
│  翻译增强            │  训练集: 英语标注 → 机器翻译为目标语言          │
│  (Translate-Train)   │  在翻译后的目标语言数据上训练                   │
│                      │  需要翻译API，可能有翻译误差传播               │
├─────────────────────┼───────────────────────────────────────────────┤
│  测试时翻译           │  训练: 英语标注数据                           │
│  (Translate-Test)    │  推理: 目标语言输入 → 翻译为英语 → 预测        │
│                      │  推理速度慢（需翻译）                          │
└─────────────────────┴───────────────────────────────────────────────┘
```

```python
"""
多语言模型跨语言迁移示例
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def cross_lingual_transfer_demo():
    """多语言模型零样本跨语言迁移"""
    
    # 使用XLM-RoBERTa（覆盖100+语言）
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )
    
    # 场景：在英语情感分析数据上训练，零样本迁移到中文/法语
    print("场景: 英语训练 → 零样本迁移到中文/法语/德语")
    
    # 英文训练数据（模拟）
    train_texts_en = [
        "I love this product, it's amazing!",     # positive
        "This is terrible, waste of money",        # negative  
        "It's okay, nothing special",               # neutral
    ]
    train_labels = [2, 0, 1]  # 0=neg, 1=neu, 2=pos
    
    # 零样本跨语言推理
    test_texts = {
        "中文": [
            "这个产品太棒了，我非常喜欢！",
            "质量太差了，完全不值这个价钱",
        ],
        "法语": [
            "J'aime beaucoup ce produit!",
            "C'est vraiment terrible",
        ],
        "德语": [
            "Ich liebe dieses Produkt!",
            "Das ist wirklich schrecklich",
        ]
    }
    
    label_map = {0: "负面", 1: "中性", 2: "正面"}
    
    print("\n=== 多语言零样本推理 ===")
    for lang, texts in test_texts.items():
        print(f"\n{lang}:")
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                logits = model(**inputs).logits
                pred = torch.argmax(logits, dim=-1).item()
                probs = torch.softmax(logits, dim=-1)[0]
                print(f"  文本: {text}")
                print(f"  预测: {label_map[pred]} (正面:{probs[2]:.2f} 中性:{probs[1]:.2f} 负面:{probs[0]:.2f})")

if __name__ == "__main__":
    print("跨语言迁移代码就绪")
    # cross_lingual_transfer_demo()
```

## 远程监督与数据编程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    弱监督标注方法                                     │
├─────────────────┬───────────────────────────────────────────────────┤
│  远程监督          │  利用已有知识库（如Freebase/Wikidata）自动标注     │
│  (Distant         │  假设: 如果句子中同时出现两个实体且知识库中存在关系 │
│   Supervision)    │  则标注为对应的关系类型                            │
│                   │  问题: 噪声标注（不是所有共现都表示该关系）           │
├─────────────────┼───────────────────────────────────────────────────┤
│  数据编程          │  用专家编写的标注函数（LF）批量标注数据              │
│  (Data            │  LF可以是: 规则/关键词/正则/外部模型输出            │
│   Programming)    │  多个LF的冲突通过标签模型自动解决                    │
│                   │  主流工具: Snorkel / Skweak                        │
└─────────────────┴───────────────────────────────────────────────────┘
```

```python
"""
远程监督关系抽取示例
"""

class DistantSupervisionExtractor:
    """远程监督关系抽取"""
    
    def __init__(self, kg_triplets):
        """
        Args:
            kg_triplets: 知识库三元组列表 [(head, relation, tail)]
        """
        self.kg = self._index_kg(kg_triplets)
    
    def _index_kg(self, triplets):
        """索引知识库以便快速查找"""
        index = {}
        for h, r, t in triplets:
            key = (h.lower(), t.lower())
            if key not in index:
                index[key] = []
            index[key].append(r)
        return index
    
    def label_sentence(self, sentence, entities):
        """
        用远程监督标注单个句子
        
        Args:
            sentence: 原始文本
            entities: 句子中出现的实体列表
        Returns:
            候选关系标注
        """
        labels = []
        
        for e1 in entities:
            for e2 in entities:
                if e1 == e2:
                    continue
                key = (e1.lower(), e2.lower())
                if key in self.kg:
                    for relation in self.kg[key]:
                        labels.append({
                            "sentence": sentence,
                            "head": e1,
                            "relation": relation,
                            "tail": e2,
                            "confidence": 1.0
                        })
        
        return labels


# 模拟知识库
kg_triplets = [
    ("乔布斯", "创始人", "苹果公司"),
    ("库克", "CEO", "苹果公司"),
    ("北京", "首都", "中国"),
    ("马云", "CEO", "阿里巴巴"),
]

extractor = DistantSupervisionExtractor(kg_triplets)

# 远程监督标注
sentences = [
    ("乔布斯在车库里创立了苹果公司", ["乔布斯", "苹果公司"]),
    ("库克担任苹果公司的CEO", ["库克", "苹果公司"]),
    ("北京是中国的首都", ["北京", "中国"]),
]

print("=== 远程监督标注结果 ===")
for sentence, entities in sentences:
    labels = extractor.label_sentence(sentence, entities)
    print(f"\n句子: {sentence}")
    if labels:
        for l in labels:
            print(f"  → ({l['head']}, {l['relation']}, {l['tail']})")
    else:
        print("  无匹配关系")
```

## 小结

| 要点 | 内容 |
|------|------|
| 低资源NLP挑战 | 数据稀缺、类别不平衡、领域偏差、过拟合风险、评估困难 |
| EDA数据增强 | 同义词替换/随机插入/交换/删除，简单高效的中文增强 |
| 回译增强 | 中文→英文→中文，生成语义相似但表达多样的文本 |
| 预训练+微调 | 大模型 + 小数据 = 低学习率 + 强正则化 + 早停 |
| 领域自适应 | 目标域继续预训练 + 对抗训练 + 逐层解冻微调 |
| Prototypical Networks | Episode训练，按类内均值计算原型，最小距离分类 |
| SetFit | 对别微调Sentence Transformer + 轻量分类器，少样本SOTA |
| Zero-shot分类 | 将分类转为NLI任务，利用蕴含概率做无样本分类 |
| 主动学习 | 不确定性采样/多样性采样/混合策略，迭代标注最小化标注量 |
| 跨语言迁移 | mBERT/XLM-R，零样本迁移 + Translate-Train/Test |
| 远程监督 | 利用知识库自动标注关系数据，需处理噪声标签 |

---

| [← 回到目录](../README.md) | [上一篇：信息抽取与知识图谱](03-信息抽取与知识图谱.md) | [下一篇：情感分析系统](../12-实战篇/01-情感分析系统.md) |
|---|---|---|
