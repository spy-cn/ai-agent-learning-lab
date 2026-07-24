# LLM原理与缩放律

大语言模型（Large Language Model, LLM）是当代NLP的核心范式。从GPT-3到Llama、Qwen，LLM以惊人的速度重塑了自然语言处理的技术格局。本章将深入LLM的核心原理、缩放律理论、涌现能力，以及主流开源/闭源模型的全景对比。

## LLM的核心能力

大语言模型具备三大核心能力，这使它从根本上区别于传统的判别式预训练模型（如BERT）：

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM 三大核心能力                              │
├─────────────┬───────────────────────────────────────────────────┤
│  理解能力    │  阅读理解、语义分析、情感识别、意图理解            │
│  (Understand)│  输入 → 深层语义表示 → 输出判断                   │
├─────────────┼───────────────────────────────────────────────────┤
│  生成能力    │  文本续写、摘要生成、翻译、代码生成、创���写作      │
│  (Generate)  │  输入 → 逐token自回归生成 → 流畅文本              │
├─────────────┼───────────────────────────────────────────────────┤
│  推理能力    │  数学推理、逻辑推理、常识推理、多步推理            │
│  (Reason)    │  输入 → 分解问题 → 中间推理步骤 → 最终结论        │
└─────────────┴───────────────────────────────────────────────────┘
```

LLM采用 **Decoder-only** 架构（即因果Transformer），通过自回归方式逐token生成文本。其训练目标是简单的"下一个token预测"（Next Token Prediction），但在海量文本上训练后，展现出了极其丰富的能力。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 演示LLM的自回归生成原理
model_name = "Qwen/Qwen2.5-0.5B"  # 使用小模型便于演示
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    device_map="auto"
)

def understand_generate_reason(prompt, max_new_tokens=100):
    """展示LLM三大能力：理解、生成、推理"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 自回归生成：每次预测下一个token
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 理解能力示例
print("=== 理解能力 ===")
print(understand_generate_reason("分析这句话的情感：'今天天气真好，心情不错！'"))

# 生成能力示例
print("\n=== 生成能力 ===")
print(understand_generate_reason("写一首关于春天的四行诗"))

# 推理能力示例
print("\n=== 推理能力 ===")
print(understand_generate_reason("小明有5个苹果，给了小红2个，又买了3个，现在有几个？"))
```

## Scaling Law (缩放律)

缩放律是理解LLM最重要的理论之一。2020年，Kaplan等人（OpenAI）发现：**模型的测试损失随模型参数量(N)、数据量(D)、计算量(C)呈幂律下降**。

### 核心公式

```
┌──────────────────────────────────────────────────────────────┐
│                    Kaplan Scaling Law                        │
│                                                              │
│   L(N) = (Nc / N)^α_N        模型参数缩放律                  │
│   L(D) = (Dc / D)^α_D        数据量缩放律                    │
│   L(C) = (Cc / C)^α_C        计算量缩放律                    │
│                                                              │
│   其中 α_N ≈ 0.076, α_D ≈ 0.095, α_C ≈ 0.050               │
│   L 为交叉熵损失（自然对数，不含可预测性）                    │
│                                                              │
│   关键结论：增大模型参数收益最大，其次数据，最后计算          │
└──────────────────────────────────────────────────────────────┘
```

### Kaplan vs Chinchilla 缩放律对比

DeepMind在2022年提出的 **Chinchilla缩放律** 对Kaplan的结论做了重要修正：

```
┌───────────────────┬──────────────────────┬──────────────────────┐
│      维度         │   Kaplan (OpenAI)    │  Chinchilla (DeepMind)│
├───────────────────┼──────────────────────┼──────────────────────┤
│  发表时间         │  2020年              │  2022年              │
│  最优策略         │  大模型 + 小数据      │  模型与数据等比缩放   │
│  tokens/parameter │  ~1.7 (偏小)         │  ~20 (最优)          │
│  70B模型最优数据  │  ~175B tokens        │  ~1.4T tokens        │
│  核心结论         │  参数越大越好        │  数据同等重要甚至更   │
│                   │                      │  重要，很多模型      │
│                   │                      │  "训练不足"          │
│  典型代表         │  GPT-3 (175B/300B)   │  Chinchilla (70B/1.4T)│
│                   │  训练数据不足        │  用更少参数打败GPT-3  │
└───────────────────┴──────────────────────┴──────────────────────┘
```

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Kaplan 缩放律：损失随参数量的幂律下降
def kaplan_scaling(N, Nc=8.8e13, alpha=0.076):
    """Kaplan scaling law: L(N) = (Nc/N)^alpha"""
    return (Nc / N) ** alpha

# Chinchilla 缩放律：考虑模型和数据的联合优化
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    """Chinchilla: L(N,D) = E + A/N^alpha + B/D^beta"""
    return E + A / (N ** alpha) + B / (D ** beta)

# 绘制缩放律曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：Kaplan - 损失 vs 参数量
params = np.logspace(7, 11, 100)  # 10M到100B
losses = kaplan_scaling(params)
axes[0].loglog(params, losses, 'b-', linewidth=2)
axes[0].set_xlabel('Parameters (N)')
axes[0].set_ylabel('Loss')
axes[0].set_title('Kaplan Scaling Law: Loss vs Parameters')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=1.75e11, color='r', linestyle='--', alpha=0.5, label='GPT-3 (175B)')
axes[0].legend()

# 右图：Chinchilla - 最优tokens/parameter
compute_budgets = np.logspace(18, 21, 50)  # FLOPs
# Chinchilla: N_opt ∝ C^0.5, D_opt ∝ C^0.5 (近似等比)
N_opt = 0.6 * compute_budgets ** 0.45  # 近似
D_opt = 30 * N_opt  # 最优tokens ≈ 20 * parameters
ratio = D_opt / N_opt

axes[1].semilogx(compute_budgets, ratio, 'g-', linewidth=2)
axes[1].axhline(y=20, color='r', linestyle='--', alpha=0.7, label='最优 ratio ≈ 20')
axes[1].set_xlabel('Compute Budget (FLOPs)')
axes[1].set_ylabel('Optimal Tokens / Parameter')
axes[1].set_title('Chinchilla: Optimal Data-to-Parameter Ratio')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../assets/scaling_law.png', dpi=150, bbox_inches='tight')
plt.show()
print("缩放律图已保存")
print(f"\nChinchilla核心结论：对于{70e9:.0f}B参数的模型，")
print(f"最优训练数据量 ≈ 20 × 70B = 1.4T tokens")
```

## 涌现能力 (Emergent Abilities)

涌现能力是指**模型规模达到某个阈值后，某些能力突然、非线性地出现**的现象。这是LLM最令人惊奇的特性之一。

```
        能力
         │                    ┌─── 突然出现
         │                   /
         │                  / ← 涌现点
         │                 /
         │                /
         │    ───────────/
         │   平稳期
         └─────────────────────────── 模型规模
              10^9   10^10  10^11  10^12
              1B     10B    100B   1T

  ┌──────────────────────────────────────────────────┐
  │           典型涌现能力及出现规模                   │
  ├──────────────────────┬──────────────────────────┤
  │  能力                │  涌现规模(参数)          │
  ├──────────────────────┼──────────────────────────┤
  │  少样本学习          │  ~10B                    │
  │  思维链推理(CoT)     │  ~60B                    │
  │  多语言翻译          │  ~10B                    │
  │  指令遵循            │  ~10B (经指令微调)       │
  │  数学推理(GSM8K)     │  ~60B                    │
  │  代码生成            │  ~10B                    │
  └──────────────────────┴──────────────────────────┘
```

```python
# 演示涌现能力：不同规模模型在不同任务上的表现差异
import matplotlib.pyplot as plt

model_sizes = [0.5, 1.5, 7, 13, 30, 70, 175, 540]  # B参数

# 模拟各任务的表现曲线（体现"涌现"特性）
tasks = {
    '基础问答':     [40, 50, 62, 68, 73, 78, 82, 85],   # 平缓增长
    '少样本翻译':   [5, 8, 15, 25, 45, 65, 75, 82],     # ~30B涌现
    '思维链推理':   [2, 3, 5, 8, 15, 45, 60, 70],       # ~60B涌现
    '数学GSM8K':    [1, 2, 3, 5, 8, 30, 50, 65],        # ~60B强涌现
}

fig, ax = plt.subplots(figsize=(10, 6))
markers = ['o', 's', '^', 'D']
for (name, scores), m in zip(tasks.items(), markers):
    ax.plot(model_sizes, scores, marker=m, linewidth=2, markersize=8, label=name)

ax.set_xlabel('Model Parameters (Billions)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Emergent Abilities: Performance vs Model Scale', fontsize=14)
ax.set_xscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(model_sizes)
ax.set_xticklabels([f'{s}B' for s in model_sizes])
plt.tight_layout()
plt.savefig('../../assets/emergent_abilities.png', dpi=150, bbox_inches='tight')
plt.show()
print("涌现能力：小模型表现接近随机，跨过阈值后突飞猛进")
```

## 主流开源LLM全景

2023-2024年是开源大模型爆发的一年，多个模型已接近甚至超越GPT-3.5水平。

```
┌──────────────────────────────────────────────────────────────────────┐
│                      主流开源LLM时间线                                │
│                                                                      │
│  2023.02 ── Llama (7B/13B/33B/65B)       Meta                      │
│  2023.07 ── Llama 2 (7B/13B/70B)         Meta                      │
│  2023.09 ── Mistral-7B                   Mistral AI                │
│  2023.09 ── Qwen (7B/14B/72B)            阿里                      │
│  2023.10 ── ChatGLM3 (6B)                智谱                      │
│  2023.12 ── Mixtral 8x7B (MoE)           Mistral AI                │
│  2024.02 ── Gemma (2B/7B)                Google                    │
│  2024.02 ── DeepSeek-MoE                 深度求索                   │
│  2024.03 ── Qwen1.5 (0.5B~110B)          阿里                      │
│  2024.04 ── Llama 3 (8B/70B)             Meta                      │
│  2024.05 ── DeepSeek-V2 (23B/MoE)        深度求索                   │
│  2024.06 ── Qwen2 (0.5B~72B)             阿里                      │
│  2024.07 ── Llama 3.1 (8B/70B/405B)      Meta                      │
│  2024.09 ── Qwen2.5 (0.5B~72B)           阿里                      │
│  2024.12 ── DeepSeek-V3 (671B/MoE)       深度求索                   │
│  2025.01 ── DeepSeek-R1 (推理模型)        深度求索                   │
└──────────────────────────────────────────────────────────────────────┘
```

### 闭源 vs 开源对比

| 维度 | 闭源模型 (GPT-4/Claude/Gemini) | 开源模型 (Llama/Qwen/DeepSeek) |
|------|------|------|
| 性能上限 | 最强（GPT-4/Claude 3.5） | 接近GPT-3.5，部分超越 |
| 可定制性 | 有限（仅API） | 完全可控（权重/训练） |
| 数据隐私 | 数据上传到云端 | 可本地部署，数据不出域 |
| 部署成本 | 按调用量付费，长期成本高 | 需GPU硬件，边际成本低 |
| 微调能力 | 不支持或受限 | 支持全参数/LoRA微调 |
| 上下文长度 | 128K~2M | 32K~128K（部分更长） |
| 多语言能力 | 强（100+语言） | 中文以Qwen/ChatGLM为佳 |
| 适用场景 | 快速验证、生产部署 | 研究、私有化、定制需求 |

```python
# 主流开源LLM加载与推理对比
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import time

def benchmark_model(model_name, prompt="请解释什么是机器学习", device="auto"):
    """加载开源LLM并测试生成速度"""
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params / 1e9:.2f}B")
    print(f"显存占用: {total_params * 2 / 1e9:.2f} GB (FP16)")
    
    # 推理测试
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start_time = time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=100, 
            temperature=0.7, do_sample=True
        )
    elapsed = time() - start_time
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
    
    print(f"生成 {tokens_generated} tokens, 耗时 {elapsed:.2f}s")
    print(f"生成速度: {tokens_generated / elapsed:.1f} tokens/s")
    print(f"回复: {response[:200]}...")
    
    del model
    torch.cuda.empty_cache()
    return response

# 主流开源模型列表（按规模排序）
opensource_llms = [
    ("Qwen/Qwen2.5-0.5B",    "阿里 Qwen2.5 0.5B"),
    ("Qwen/Qwen2.5-1.5B",    "阿里 Qwen2.5 1.5B"),
    ("meta-llama/Llama-3.2-1B", "Meta Llama 3.2 1B"),
]

for model_id, model_desc in opensource_llms:
    try:
        benchmark_model(model_id)
    except Exception as e:
        print(f"  {model_desc} 加载失败: {e}")
```

## 模型规模与能力关系

不同规模的模型适用于不同场景，选择合适的模型至关重要：

```
┌─────────────────────────────────────────────────────────────────┐
│               模型规模选型指南                                    │
├──────────────┬──────────────┬───────────────────────────────────┤
│  规模        │  代表模型    │  适用场景                         │
├──────────────┼──────────────┼───────────────────────────────────┤
│  0.5-3B      │  Qwen2.5-1.5B│  端侧部署、简单任务、原型验证    │
│  (小型)      │  Gemma-2B    │  手机/边缘设备                    │
│              │              │  延迟 < 1s, 内存 < 4GB           │
├──────────────┼──────────────┼───────────────────────────────────┤
│  7-14B       │  Llama3-8B   │  通用任务、中等复杂度推理         │
│  (中型)      │  Qwen2.5-7B  │  单卡部署 (16-24GB显存)          │
│              │  Mistral-7B  │  对话/摘要/翻译/代码生成          │
├──────────────┼──────────────┼───────────────────────────────────┤
│  30-70B      │  Llama3-70B  │  复杂推理、高质量生成             │
│  (大型)      │  Qwen2.5-72B │  多卡部署 (2-4×A100)             │
│              │  DeepSeek-V2 │  接近GPT-3.5/Claude水平          │
├──────────────┼──────────────┼───────────────────────────────────┤
│  100B+       │  Llama3.1-   │  最强开源能力                     │
│  (超大型)    │    405B      │  集群部署 (8+×A100)              │
│              │  DeepSeek-V3 │  接近GPT-4水平                   │
│  MoE         │  Mixtral 8x7B│  MoE: 总参数大但激活参数小       │
│              │              │  推理速度接近小模型               │
└──────────────┴──────────────┴───────────────────────────────────┘
```

```python
# MoE (Mixture of Experts) 模型演示
# MoE是当前LLM的重要架构创新：参数总量大但单次推理激活的参数少
# 例如 Mixtral 8x7B: 总参数 ~47B, 但每次推理只激活 ~13B

code_explain = """
# MoE架构核心思想
┌──────────────────────────────────────────────────────┐
│                  MoE Layer Structure                 │
│                                                      │
│        Input Token                                   │
│             │                                        │
│             ▼                                        │
│      ┌─────────────┐                                 │
│      │   Router    │ ← 门控网络选择Top-K专家         │
│      │  (Gate)     │                                 │
│      └──────┬──────┘                                 │
│             │                                        │
│    ┌────────┼────────┐                               │
│    ▼        ▼        ▼                               │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐                     │
│  │Exp1│  │Exp2│  │Exp3│  │Exp4│  ... Exp8            │
│  │FFN │  │FFN │  │FFN │  │FFN │                     │
│  └──┬─┘  └──┬─┘  └────┘  └──┬─┘                     │
│     │       │        ✗       │    ✗  ← 未激活       │
│     └───────┼───────────────┘                        │
│             ▼                                        │
│      Weighted Sum (输出)                             │
│                                                      │
│  优势: 总参数 8×7B=47B，单次推理仅激活 2×7B=14B     │
└──────────────────────────────────────────────────────┘
"""
print(code_explain)

# 模拟MoE路由机制
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    """简化版MoE层：演示路由机制"""
    def __init__(self, d_model=512, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # 路由门控
        self.gate = nn.Linear(d_model, num_experts)
        # 专家网络 (每个是一个FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        # 路由打分
        gate_logits = self.gate(x)  # [B, S, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # 选择Top-K个专家
        topk_probs, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        # 归一化概率
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # 执行专家计算
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = topk_indices[..., i]  # [B, S]
            prob = topk_probs[..., i:i+1]  # [B, S, 1]
            for b in range(batch_size):
                for s in range(seq_len):
                    eidx = expert_idx[b, s].item()
                    output[b, s] += prob[b, s] * self.experts[eidx](x[b, s])
        
        # 记录负载均衡（辅助损失）
        self.aux_loss = self._load_balancing_loss(gate_probs, topk_indices)
        return output
    
    def _load_balancing_loss(self, gate_probs, topk_indices):
        """负载均衡损失：防止只使用少数专家"""
        num_tokens = gate_probs.shape[0] * gate_probs.shape[1]
        # 每个专家被选中的频率
        expert_mask = F.one_hot(topk_indices, self.num_experts).float()
        expert_freq = expert_mask.sum(dim=-2).mean(dim=(0,1)) / self.top_k
        # 每个专家的平均概率
        expert_prob = gate_probs.mean(dim=(0,1))
        # 负载均衡损失
        return self.num_experts * (expert_freq * expert_prob).sum()

# 测试MoE层
moe = MoELayer(d_model=512, num_experts=8, top_k=2)
x = torch.randn(2, 10, 512)
output = moe(x)
print(f"MoE输入: {x.shape}")
print(f"MoE输出: {output.shape}")
print(f"总参数: {sum(p.numel() for p in moe.parameters())/1e6:.1f}M")
print(f"每次推理激活参数: ~{sum(p.numel() for e in moe.experts[:2] for p in e.parameters())/1e6:.1f}M (2/8专家)")
print(f"负载均衡损失: {moe.aux_loss.item():.4f}")
```

## 小结

| 要点 | 内容 |
|------|------|
| LLM三大能力 | 理解（Understand）+ 生成（Generate）+ 推理（Reason） |
| 架构基础 | Decoder-only Transformer，自回归生成，Next Token Prediction |
| Kaplan缩放律 | 损失 ∝ N^-0.076，参数增大收益最大 |
| Chinchilla缩放律 | 最优tokens/parameter ≈ 20，模型与数据等比缩放 |
| 涌现能力 | 规模达阈值后能力非线性跃升（CoT推理需~60B） |
| MoE架构 | 总参数大、激活参数小，兼顾性能和速度 |
| 主流开源 | Llama、Qwen、Mistral、DeepSeek、ChatGLM |
| 选型原则 | 端侧0.5-3B / 通用7-14B / 复杂70B+ / MoE兼顾性价比 |
| Chinchilla启示 | 很多模型"训练不足"，数据量应 ≈ 20×参数量 |

---

| [← 回到目录](../README.md) | [上一篇：文本生成与摘要](../09-NLP核心任务篇/05-文本生成与摘要.md) | [下一篇：上下文学习与思维链](02-上下文学习与思维链.md) |
|---|---|
