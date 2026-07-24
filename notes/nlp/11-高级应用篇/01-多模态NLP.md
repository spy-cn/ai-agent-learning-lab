# 多模态NLP

多模态NLP（Multimodal NLP）是指将文本与图像、音频、视频等多种模态信息进行联合建模与理解的技术方向。传统NLP只处理纯文本输入，而多模态NLP打破了模态壁垒，让模型能够像人类一样"看"图像、"听"声音，并结合语言进行综合推理。本章将系统介绍多模态NLP的核心模型、关键任务和应用场景。

## 多模态NLP的定义与挑战

```
┌─────────────────────────────────────────────────────────────────────┐
│                       多模态NLP全景                                  │
├─────────────┬───────────────────────────────────────────────────────┤
│  文本+图像   │  图文匹配、图像描述(Image Captioning)、视觉问答(VQA)   │
│  文本+视频   │  视频理解、视频问答、视频摘要                          │
│  文本+音频   │  语音识别(ASR)、语音翻译、多模态情感分析               │
│  文本+多模态 │  跨模态检索、多模态对话、具身智能                       │
└─────────────┴───────────────────────────────────────────────────────┘
```

**核心挑战**：
1. **模态对齐**：不同模态的表示空间不同，需要将图像像素、音频波形与文本token映射到统一语义空间
2. **模态融合**：如何有效融合异构信息，而不是简单拼接
3. **跨模态推理**：需要模型在不同模态之间进行逻辑推理（如"图中人物的情绪是什么？"）
4. **数据稀缺**：大规模高质量的多模态配对数据获取成本高
5. **计算开销**：处理高维图像/视频数据需要大量计算资源

## CLIP：对比语言-图像预训练

CLIP（Contrastive Language-Image Pre-training）由OpenAI于2021年提出，是多模态领域的里程碑工作。它通过在4亿图文对上训练，学习将图像和文本映射到统一的语义空间。

### CLIP双塔结构

```
┌────────────────────── CLIP 双塔架构 ──────────────────────┐
│                                                            │
│   图像编码器                    文本编码器                   │
│   ┌─────────────┐             ┌─────────────┐              │
│   │    ViT /    │             │ Transformer │              │
│   │  ResNet     │             │  (BERT/     │              │
│   │             │             │  GPT-2)     │              │
│   └──────┬──────┘             └──────┬──────┘              │
│          │                           │                     │
│          ▼                           ▼                     │
│   ┌─────────────┐             ┌─────────────┐              │
│   │  图像嵌入    │             │  文本嵌入    │              │
│   │  d=512/d=768│             │  d=512/d=768│              │
│   └──────┬──────┘             └──────┬──────┘              │
│          │                           │                     │
│          └───────────┬───────────────┘                     │
│                      │                                     │
│                      ▼                                     │
│          ┌─────────────────────┐                           │
│          │  Cosine Similarity  │                           │
│          │  Matrix (B × B)     │                           │
│          └─────────────────────┘                           │
│                      │                                     │
│                      ▼                                     │
│          ┌─────────────────────┐                           │
│          │  InfoNCE Loss       │                           │
│          │  匹配对拉近/不匹配拉远│                           │
│          └─────────────────────┘                           │
└───────────────────────────────────────────────────────────┘
```

**对比学习目标**：给定一个batch中的B个图文对，CLIP最大化对角线上匹配图文对的余弦相似度，同时最小化非匹配对的相似度。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    """CLIP对比学习损失的核心实现"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature  # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(0.07).reciprocal()))
    
    def forward(self, image_embeds, text_embeds):
        """
        Args:
            image_embeds: (B, D) 图像嵌入
            text_embeds:  (B, D) 文本嵌入
        Returns:
            loss: 对称InfoNCE损失
        """
        # L2归一化
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # 计算余弦相似度矩阵 (B × B)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        
        # 标签：对角线位置为匹配对
        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=image_embeds.device)
        
        # 对称交叉熵损失
        loss_i = F.cross_entropy(logits_per_image, labels)  # 图像→文本
        loss_t = F.cross_entropy(logits_per_text, labels)   # 文本→图像
        loss = (loss_i + loss_t) / 2
        
        return loss

# 模拟演示
B, D = 4, 512
image_embeds = torch.randn(B, D)
text_embeds = torch.randn(B, D)

clip_loss_fn = CLIPLoss()
loss = clip_loss_fn(image_embeds, text_embeds)
print(f"CLIP对比损失: {loss.item():.4f}")
```

### 零样本图像分类原理

CLIP最惊艳的能力是**零样本分类（Zero-shot Classification）**——无需在目标类别的标注图像上微调，直接利用文本描述进行分类。

```
┌──────────── 零样本分类流程 ────────────┐
│                                         │
│  输入: 一张猫的图片                       │
│                                         │
│  候选标签 → 文本模板                      │
│  ┌─────────────────────────┐            │
│  │ "dog"   → "a photo of a dog"        │
│  │ "cat"   → "a photo of a cat"        │
│  │ "bird"  → "a photo of a bird"       │
│  │ "car"   → "a photo of a car"        │
│  └─────────────────────────┘            │
│               │                         │
│               ▼                         │
│  计算图像与每个文本描述的相似度            │
│  Cat: 0.92  ★ 最高                      │
│  Dog: 0.35                               │
│  Bird: 0.12                              │
│  Car: 0.03                               │
│               │                         │
│               ▼                         │
│  预测结果: "cat"                          │
└─────────────────────────────────────────┘
```

### HuggingFace CLIP 使用代码

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# 加载CLIP模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def clip_zero_shot_classification(image_path, candidate_labels, custom_templates=None):
    """
    使用CLIP进行零样本图像分类
    
    Args:
        image_path: 图像路径
        candidate_labels: 候选标签列表，如 ["cat", "dog", "bird"]
        custom_templates: 可选的文本模板，如 "a picture of a {}"
    Returns:
        排序后的分类结果
    """
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    
    # 构造文本描述：默认使用OpenAI的prompt engineering模板
    if custom_templates is None:
        templates = [
            "a photo of a {}.",
            "a blurry photo of a {}.",
            "a good photo of a {}.",
            "a photo of many {}.",
            "a photo of the large {}.",
        ]
    else:
        templates = [custom_templates]
    
    # 对每个标签生成多个模板描述
    texts = []
    for label in candidate_labels:
        for template in templates:
            texts.append(template.format(label))
    
    # 推理
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # (1, num_texts)
        
        # 对同一类别的多个模板取平均
        probs = logits_per_image.softmax(dim=-1)
        num_labels = len(candidate_labels)
        num_templates = len(templates)
        
        # 聚合同一标签的多个模板概率
        label_probs = []
        for i in range(0, len(texts), num_templates):
            label_probs.append(probs[:, i:i+num_templates].mean().item())
        
        # 排序输出
        results = sorted(
            zip(candidate_labels, label_probs),
            key=lambda x: x[1], reverse=True
        )
    
    return results

# 使用示例（需要实际图片）
# results = clip_zero_shot_classification("cat.jpg", ["cat", "dog", "bird", "car", "tree"])
# for label, prob in results:
#     print(f"{label}: {prob:.4f}")

print("CLIP零样本分类代码就绪")
print("使用方法: results = clip_zero_shot_classification('image.jpg', ['cat', 'dog'])")
```

## BLIP-2：Q-Former桥接视觉与语言

BLIP-2（Bootstrapping Language-Image Pre-training）由Salesforce提出，通过 **Q-Former（Querying Transformer）** 优雅地连接冻结的图像编码器和冻结的大语言模型。

### BLIP-2 架构

```
┌────────────────────── BLIP-2 架构 ────────────────────────────┐
│                                                                │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │  冻结的图像   │     │   Q-Former   │     │  冻结的 LLM  │    │
│  │  编码器       │────▶│  (可训练)     │────▶│  (如OPT/     │    │
│  │  (ViT-g/14)  │     │              │     │  FlanT5)     │    │
│  └──────────────┘     └──────────────┘     └──────────────┘   │
│                              ▲                                │
│                              │                                │
│                     ┌────────┴────────┐                       │
│                     │  可学习的Query   │                       │
│                     │  Tokens (32个)   │                      │
│                     └─────────────────┘                       │
│                                                                │
│  Q-Former内部结构:                                              │
│  ┌─────────────────────────────────────────────┐               │
│  │  Self-Attention: Query tokens 相互交互        │              │
│  │  Cross-Attention: Query tokens 与图像特征交互  │              │
│  │  Feed-Forward: 非线性变换                     │              │
│  └─────────────────────────────────────────────┘               │
│                                                                │
│  训练目标:                                                       │
│  1. 图文对比损失 (ITC): 对齐Query表示与文本表示                     │
│  2. 图文匹配损失 (ITM): 二分类判断图文是否匹配                     │
│  3. 语言建模损失 (LM): 基于图像生成文本描述                        │
└────────────────────────────────────────────────────────────────┘
```

**核心设计思想**：
- **冻结编码器**：图像编码器（ViT）和LLM都不参与训练，大幅降低训练成本
- **Q-Former瓶颈**：仅训练Q-Former（约188M参数），作为图像特征到文本空间的"翻译器"
- **可学习的Query Tokens**：32个可学习的token，通过交叉注意力从冻结图像特征中提取与文本语义最相关的信息

### BLIP-2 图像描述和VQA代码

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

# 加载BLIP-2模型
def load_blip2():
    """加载BLIP-2模型用于图像描述和VQA"""
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return processor, model

def image_captioning(processor, model, image_path):
    """使用BLIP-2生成图像描述"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device, torch.float16)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=5,
            temperature=0.7
        )
    
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption.strip()

def visual_question_answering(processor, model, image_path, question):
    """使用BLIP-2进行视觉问答(VQA)"""
    image = Image.open(image_path).convert("RGB")
    
    # 构造VQA提示
    prompt = f"Question: {question} Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(
        model.device, torch.float16
    )
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=3,
            temperature=0.5
        )
    
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return answer.strip()

# 使用示例（需要实际图片）
# processor, model = load_blip2()
# caption = image_captioning(processor, model, "scene.jpg")
# print(f"图像描述: {caption}")
# answer = visual_question_answering(processor, model, "scene.jpg", "How many people are in the image?")
# print(f"VQA答案: {answer}")

print("BLIP-2图像描述和VQA代码就绪")
```

## 视觉问答（VQA）

视觉问答（Visual Question Answering）是结合计算机视觉和自然语言理解的典型多模态任务：给定一张图像和一个关于该图像的自然语言问题，模型需要生成正确的答案。

```
┌────────────────────────────────────────────────┐
│                 VQA 任务示意                    │
│                                                │
│   输入图像: [一张公园场景的照片]                  │
│                                                │
│   问题: "How many dogs are in the image?"      │
│          "图中几只狗？"                          │
│                                                │
│   ┌──────────┐     ┌──────────┐                │
│   │ 图像编码器 │     │ 问题编码器 │               │
│   └─────┬────┘     └─────┬────┘                │
│         │                │                     │
│         └───────┬────────┘                     │
│                 ▼                              │
│         ┌──────────────┐                       │
│         │  多模态融合    │                       │
│         │  (Cross-Atten)│                      │
│         └──────┬───────┘                       │
│                ▼                               │
│         ┌──────────────┐                       │
│         │  答案解码器    │                       │
│         └──────┬───────┘                       │
│                ▼                               │
│         答案: "2"                               │
└────────────────────────────────────────────────┘
```

**VQA类型分类**：

| 类型 | 描述 | 示例 |
|------|------|------|
| 是/否问题 | 回答yes/no | "Is there a cat?" |
| 计数问题 | 统计数量 | "How many people?" |
| 物体识别 | 识别物体 | "What is on the table?" |
| 颜色问题 | 询问颜色 | "What color is the car?" |
| 空间关系 | 空间推理 | "What is to the left of?" |
| 动作识别 | 识别动作 | "What is the person doing?" |
| OCR-VQA | 图中文字理解 | "What does the sign say?" |

```python
import torch
import torch.nn as nn

class SimpleVQAModel(nn.Module):
    """简化的VQA模型结构演示"""
    def __init__(self, vision_dim=2048, text_dim=768, hidden_dim=1024, num_answers=3129):
        super().__init__()
        # 图像特征投影
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        # 文本特��投影
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        # 多模态融合（逐元素乘，也可用注意力融合）
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_answers)  # 分类到预定义答案集
        )
    
    def forward(self, vision_features, text_features):
        """
        Args:
            vision_features: (B, 2048) 图像特征（如ResNet输出）
            text_features:   (B, 768)  文本特征（如BERT[CLS]输出）
        Returns:
            logits: (B, num_answers) 答案分类logits
        """
        v = self.vision_proj(vision_features)  # (B, hidden_dim)
        t = self.text_proj(text_features)      # (B, hidden_dim)
        
        # 多模态融合：乘性交互
        fused = v * t  # 逐元素相乘
        logits = self.fusion(fused)
        
        return logits

# 模拟前向传播
B = 4
model = SimpleVQAModel()
vision_feat = torch.randn(B, 2048)
text_feat = torch.randn(B, 768)
logits = model(vision_feat, text_feat)
print(f"VQA模型输出: {logits.shape}")  # (4, 3129)
pred_answer_idx = logits.argmax(dim=-1)
print(f"预测答案索引: {pred_answer_idx}")
```

## 多模态大模型全景

当前多模态大模型（MLLM）呈现百花齐放态势，以下是主流模型概览：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    多模态大模型全景对比                                │
├──────────┬────────────┬───────────────┬──────────────┬──────────────┤
│  模型     │  开发者     │  架构/视觉编码  │  语言模型     │  能力特点     │
├──────────┼────────────┼───────────────┼──────────────┼──────────────┤
│ GPT-4V   │ OpenAI     │ 闭源          │ GPT-4        │ 全能, 最强    │
│ Gemini   │ Google     │ 闭源          │ Gemini       │ 多模态原生    │
│ Qwen-VL  │ 阿里       │ ViT-G         │ Qwen         │ 中文多模态    │
│ Llava    │ Wisconsin  │ CLIP-ViT      │ Llama/Vicuna │ 开源先驱     │
│ InternVL │ 上海AI Lab │ InternViT-6B  │ InternLM     │ 最强开源之一  │
│ CogVLM   │ 清华       │ ViT-EVA02     │ Llama        │ 视觉专家深融  │
│ MiniCPM-V│ 清华/面壁  │ SigLIP-400M   │ MiniCPM      │ 端侧多模态   │
└──────────┴────────────┴───────────────┴──────────────┴──────────────┘
```

### 多模态大模型通用架构

```python
class GenericMLLM(nn.Module):
    """通用多模态大模型架构示意"""
    def __init__(self):
        super().__init__()
        # 1. 视觉编码器：将图像编码为视觉token序列
        self.vision_encoder = None  # ViT / SigLIP
        
        # 2. 视觉-语言投影层：将视觉特征映射到LLM空间
        self.vision_projector = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.GELU(),
            nn.Linear(4096, 4096),  # 匹配LLM隐藏维度
        )
        
        # 3. 大语言模型：接收图像+文本token序列，自回归生成
        self.language_model = None  # Llama / Qwen / InternLM
    
    def forward(self, images, text_input_ids):
        """
        Args:
            images: (B, 3, H, W) 输入图像
            text_input_ids: (B, L) 文本token序列
        """
        # 提取视觉特征
        vision_features = self.vision_encoder(images)  # (B, N_vis, D_vis)
        
        # 投影到LLM空间
        vision_embeds = self.vision_projector(vision_features)  # (B, N_vis, D_llm)
        
        # 获取文本嵌入
        text_embeds = self.language_model.get_input_embeddings()(text_input_ids)
        
        # 拼接：视觉token + 文本token → 送入LLM
        # 在文本嵌入前插入视觉token
        combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        
        # LLM自回归生成
        outputs = self.language_model(inputs_embeds=combined_embeds)
        
        return outputs

print("多模态大模型通用架构已定义")
```

## 多模态NLP应用场景

```
┌─────────────────────────────────────────────────────────────────────┐
│                      多模态NLP典型应用                               │
├───────────────────┬─────────────────────────────────────────────────┤
│  OCR + 文档理解    │  扫描文档识别 → 关键信息提取 → 结构化输出        │
│  图表问答          │  财报图表 → 趋势分析 → 自然语言问答              │
│  视频理解          │  视频帧序列 → 动作识别 → 事件描述                │
│  多模态搜索        │  文字搜图 / 以图搜图 / 跨模态检索                │
│  内容审核          │  图文联合审核 → 识别违规内容                     │
│  具身智能          │  视觉感知 + 语言指令 → 机器人执行               │
│  医疗影像分析      │  X光/CT图像 + 报告文本 → 辅助诊断               │
└───────────────────┴─────────────────────────────────────────────────┘
```

### 完整代码：使用 BLIP-2 做图像描述和 VQA

```python
"""
BLIP-2 图像描述和视觉问答完整示例
运行前请安装: pip install transformers pillow torch
"""

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def main():
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型（使用较小版本以便CPU运行）
    model_name = "Salesforce/blip2-opt-2.7b"
    print(f"加载模型: {model_name}")
    
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    if device == "cpu":
        model = model.to(device)
    
    # 图像描述函数
    def generate_caption(image_path):
        """生成图像的文字描述"""
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        if device == "cuda":
            inputs = {k: v.half() for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=5,
                temperature=0.7,
                do_sample=False
            )
        
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.strip()
    
    # 视觉问答函数
    def answer_question(image_path, question):
        """对图像进行问答"""
        image = Image.open(image_path).convert("RGB")
        prompt = f"Question: {question} Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        if device == "cuda":
            inputs = {k: v.half() if k == "pixel_values" else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=30,
                num_beams=3,
                temperature=0.3,
                do_sample=False
            )
        
        answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer.strip()
    
    # 多问题批量VQA
    def batch_vqa(image_path, questions):
        """批量进行VQA"""
        image = Image.open(image_path).convert("RGB")
        answers = []
        for q in questions:
            prompt = f"Question: {q} Answer:"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            if device == "cuda":
                inputs = {k: v.half() if k == "pixel_values" else v for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=20, num_beams=3
                )
            
            ans = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            answers.append(ans.strip())
        return answers
    
    print("\n模型加载完成!")
    print("=" * 50)
    print("可用函数:")
    print("  generate_caption(image_path)  - 生成图像描述")
    print("  answer_question(image_path, question) - 视觉问答")
    print("  batch_vqa(image_path, questions) - 批量问答")
    print("=" * 50)
    
    # 示例调用（需要替换为实际图片路径）
    # caption = generate_caption("path/to/your/image.jpg")
    # print(f"图像描述: {caption}")
    #
    # answer = answer_question("path/to/your/image.jpg", "What is in this image?")
    # print(f"VQA答案: {answer}")
    #
    # questions = ["How many people?", "What is the weather like?", "Is it daytime?"]
    # answers = batch_vqa("path/to/your/image.jpg", questions)
    # for q, a in zip(questions, answers):
    #     print(f"Q: {q}")
    #     print(f"A: {a}\n")
    
    return model, processor, generate_caption, answer_question, batch_vqa

if __name__ == "__main__":
    model, processor, cap_fn, vqa_fn, batch_vqa_fn = main()
```

## 小结

| 要点 | 内容 |
|------|------|
| 多模态NLP定义 | 文本+图像+音频+视频的联合理解与生成 |
| CLIP核心技术 | 双塔架构（ViT + Transformer）+ 对比学习（InfoNCE Loss）|
| CLIP零样本分类 | 图像嵌入 vs 模板文本嵌入 → 取最大相似度的标签 |
| BLIP-2创新 | Q-Former桥接冻结的图像编码器和LLM，仅训练188M参数 |
| Q-Former三目标 | ITC（对比损失）+ ITM（匹配损失）+ LM（生成损失）|
| VQA任务 | 图像+自然语言问题 → 答案生成，含��数/颜色/空间推理等子任务 |
| MLLM主流模型 | GPT-4V / Gemini / Qwen-VL / Llava / InternVL / CogVLM |
| MLLM通用模式 | 视觉编码器 → 投影层 → LLM（视觉token前置插入文本序列）|
| 典型应用 | OCR文档理解、图表问答、视频理解、多模态搜索、医疗影像分析 |

---

| [← 回到目录](../README.md) | [上一篇：RAG检索增强生成](../10-大语言模型篇/04-RAG检索增强生成.md) | [下一篇：对话系统](02-对话系统.md) |
|---|---|---|
