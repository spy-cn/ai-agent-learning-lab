# LoRA 与 PEFT：参数高效微调

当模型参数量达到数十亿甚至上百亿时，全参数微调的成本变得令人望而却步。PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）技术应运而生：它只更新模型中极小一部分参数（通常 < 1%），就能达到接近甚至媲美全参数微调的效果。本章将深入讲解 PEFT 的核心原理与实战。

## 一、PEFT 的动机

### 1.1 大模型微调的困境

```
┌─────────────────────────────────────────────────────────────┐
│                全参数微调的成本墙                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  模型大小        显存需求(训练)     GPU 需求                 │
│  ───────────────────────────────────────────────            │
│  BERT-base      ~2 GB            1× T4 (16GB)              │
│  BERT-large     ~6 GB            1× V100 (32GB)            │
│  LLaMA-7B       ~60 GB           4× A100 (40GB)            │
│  LLaMA-13B      ~100 GB          8× A100 (40GB)            │
│  LLaMA-70B      ~500 GB          32× A100 (40GB)           │
│                                                             │
│  问题1：显存不够 → 需要昂贵的多卡集群                       │
│  问题2：存储爆炸 → 每个任务一份完整模型副本                  │
│  问题3：切换慢   → 加载/切换大模型很慢                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 PEFT 的核心思想

```
全参数微调:                     PEFT (参数高效微调):
┌────────────────────┐          ┌────────────────────┐
│                    │          │  ❄❄❄❄ 冻结 99.9%   │
│  原模型参数(全部更新)│          │  ┌──────────────┐ │
│  ████████████████  │          │  │  原模型(冻结) │ │
│  ████████████████  │          │  │  ████████████ │ │
│                    │          │  └──────┬───────┘ │
│  每个任务存一份     │          │    ┌────┴────┐    │
│  7B → 14GB/任务     │          │    │ 适配器   │    │ (训练)
│                    │          │    │ (新增)   │    │
└────────────────────┘          │    └─────────┘    │
                                │  每个任务只存适配器 │
                                │  7B → 8MB/任务     │
                                └────────────────────┘
```

| 对比维度 | 全参数微调 | PEFT |
|---------|-----------|------|
| 训练参数量 | 100% | 0.01% ~ 1% |
| 显存占用 | 模型权重 + 梯度 + 优化器状态 | 仅适配器参数 |
| 存储成本 | 每任务 ~14GB（7B模型） | 每任务 ~8MB |
| 任务切换 | 重新加载整个模型 | 切换适配器（秒级）|
| 多任务 | 需要多份模型 | 多个适配器共存 |
| 效果 | 基准 | 接近全参（差距 < 1%）|

### 1.3 PEFT 方法分类

```
                    PEFT 方法大全
                         │
          ┌──────────────┼──────────────┐
          │              │              │
     ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
     │ 增量式   │   │ 提示式  │   │ 重组式   │
     │(Add)    │   │(Prompt)│   │(Reparam)│
     └────┬────┘   └────┬────┘   └─────────┘
          │              │
     ┌────┼────┐    ┌────┼────┐
     │    │    │    │         │
   LoRA Adapter IA3  Prefix  Prompt
                    Tuning  Tuning
```

## 二、LoRA（Low-Rank Adaptation）

### 2.1 LoRA 原理

LoRA 的核心思想：**模型权重的更新矩阵是低秩的**，可以用两个小矩阵的乘积来近似。

```
┌─────────────────────────────────────────────────────────────┐
│                    LoRA 原理图解                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  原始全参数微调：                                            │
│         W (d×d)                                              │
│  h = W·x        W 被更新，d 可能=4096                        │
│  参数量: d×d = 16M (d=4096)                                 │
│                                                             │
│  LoRA 低秩分解：                                             │
│                                                             │
│       x (输入)                                               │
│        │                                                    │
│        ├────────────────────────┐                           │
│        │                        │                           │
│        ▼                        ▼                           │
│    ┌───────┐  ❄ 冻结     ┌───────────┐  ← r 维 (如 r=8)    │
│    │ W₀    │             │ ↓ B        │  d×r               │
│    │ d×d   │             │ r×d        │                    │
│    │(预训练)│             │ (可训练)   │                    │
│    └───┬───┘             └─────┬─────┘                     │
│        │                       │ ↑ A                       │
│        │                       │ d×r                       │
│        │                       │ (可训练)                   │
│        │                       │                           │
│        ▼                       ▼                           │
│      W₀·x       +           B·A·x                           │
│        │                       │                           │
│        └───────┬───────────────┘                           │
│                │                                            │
│                ▼                                            │
│              h = W₀x + ΔW·x = W₀x + BAx                    │
│                                                             │
│  参数量: 2×d×r = 2×4096×8 = 65K  (减少 256 倍！)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

数学表达：

```
全参数微调:    W' = W₀ + ΔW        (ΔW 是 d×d 矩阵)

LoRA:         W' = W₀ + B·A        (B: d×r, A: r×d)

其中:
  W₀ ∈ ℝ^{d×d}   预训练权重（冻结，不更新）
  B  ∈ ℝ^{d×r}   降维矩阵（初始化为零）
  A  ∈ ℝ^{r×d}   升维矩阵（正态分布初始化）
  r  << d          秩（如 r=8, d=4096）

训练时只更新 A 和 B，参数量从 d² 降到 2dr
```

### 2.2 为什么 LoRA 有效

```python
# LoRA 的核心假设：模型微调时的权重变化是"低秩"的

# 实证：分析微调后的 ΔW 的奇异值分解
import torch

# 假设这是微调前后的权重差
delta_W = torch.randn(4096, 4096) * 0.02  # 模拟 ΔW

# SVD 分解
U, S, V = torch.linalg.svd(delta_W)

# 看看前几个奇异值占了多少能量
total_energy = (S**2).sum()
for top_k in [1, 4, 8, 16, 64, 256]:
    energy = (S[:top_k]**2).sum() / total_energy
    print(f"  前 {top_k:3d} 个奇异值占总能量的 {energy*100:.1f}%")

# 典型结果：
#   前   1 个奇异值占总能量的  8.3%
#   前   4 个奇异值占总能量的 25.7%
#   前   8 个奇异值占总能量的 42.1%  ← r=8 就能捕获近一半
#   前  16 个奇异值占总能量的 61.3%
#   前  64 个奇异值占总能量的 89.5%
#   前 256 个奇异值占总能量的 99.9%
```

### 2.3 LoRA 超参数

| 超参数 | 含义 | 推荐值 | 说明 |
|--------|------|--------|------|
| `r` (rank) | 低秩矩阵的秩 | 8, 16, 64 | 越大表达能力越强，但参数越多 |
| `alpha` | 缩放因子 | 通常 = 2×r | 控制 LoRA 更新的强度 |
| `target_modules` | 应用到哪些层 | q_proj, v_proj | 默认只加到 Attention 的 Q/V |
| `dropout` | LoRA 层 dropout | 0.05 | 防过拟合 |
| `bias` | 是否训练 bias | "none" | 通常不需要 |

```python
# rank r 的影响
r_values = [1, 2, 4, 8, 16, 32, 64, 128]
d = 4096  # 模型隐藏维度

print("LoRA 参数量 vs rank：")
for r in r_values:
    # 每个应用点有 A(d×r) + B(r×d) = 2dr 参数
    params_per_module = 2 * d * r
    # 通常应用在 Q, K, V, O 等 Attention 矩阵
    num_modules = 4
    total = params_per_module * num_modules
    ratio = total / (d * d * num_modules) * 100
    print(f"  r={r:3d}: 每层 {total/1e6:.2f}M 参数 ({ratio:.2f}% of original)")
```

### 2.4 alpha 的作用

```
实际更新公式：h = W₀x + (α/r) · BAx

  α (alpha) ──→ 缩放系数
  r (rank)  ──→ 秩

  作用：当 r 变大时，BA 的值可能变大，用 α/r 保持稳定的更新幅度

  经验法则：
    alpha = 2 × rank   (最常用)
    alpha = rank       (保守更新)
    alpha = 16 (固定)   (与 rank 无关)
```

## 三、HuggingFace PEFT 库实战

### 3.1 安装与基本使用

```python
# 安装
# pip install peft

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# === 步骤1：加载基础模型 ===
model_name = "bigscience/bloom-560m"  # 示例用小模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# === 步骤2：配置 LoRA ===
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # 任务类型
    r=8,                             # 秩
    lora_alpha=32,                   # 缩放因子 (alpha = 4×r 或 2×r)
    lora_dropout=0.05,               # dropout
    target_modules=[                 # 应用 LoRA 的层
        "query_key_value"            # BLOOM 的注意力层名
    ],
)

# === 步骤3：用 PEFT 包装模型 ===
peft_model = get_peft_model(model, lora_config)

# 查看可训练参数
peft_model.print_trainable_parameters()
# 输出: trainable params: 1,572,864 || all params: 559,407,360 || trainable%: 0.281%
```

### 3.2 target_modules 查找

```python
# 不同模型的层名不同，需要对应
model_target_modules = {
    # === Transformer Attention 层 ===
    "LLaMA / Mistral": ["q_proj", "v_proj"],           # 推荐默认
    "LLaMA (全部)":     ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
    "BLOOM":           ["query_key_value"],
    "BART / mBART":    ["q_proj", "v_proj"],
    "GPT-2":           ["c_attn"],
    "ChatGLM":         ["query_key_value"],
    
    # === 也可以加到 FFN 层 ===
    "LLaMA FFN":       ["gate_proj", "up_proj", "down_proj"],
}

# 自动发现可用模块名
def find_linear_layers(model):
    """找出模型中所有线性层"""
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 取最后一层名
            layer_name = name.split(".")[-1]
            if layer_name not in linear_layers:
                linear_layers.append(layer_name)
    return linear_layers

# 使用方法
layers = find_linear_layers(model)
print(f"可用的线性层: {layers}")
```

### 3.3 完整 LoRA 微调示例

```python
"""
完整示例：用 LoRA 微调 LLaMA 做中文对话
"""
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# === 配置 ===
model_name = "meta-llama/Llama-2-7b-hf"  # 或中文模型
output_dir = "./lora-output"

# === 加载模型 ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,     # 半精度加载
    device_map="auto",             # 自动分配到多卡
    load_in_4bit=True,             # QLoRA：4bit量化加载（见下文）
)

# === LoRA 配置 ===
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 20,090,880 || all params: 6,755,295,232 || trainable%: 0.297%

# === 数据处理 ===
def format_instruction(instruction: str, output: str) -> str:
    """格式化为指令微调格式"""
    return f"""### 指令：
{instruction}

### 回答：
{output}{tokenizer.eos_token}"""

dataset = load_dataset("your_dataset")  # 替换为你的数据集

def preprocess(examples):
    texts = [format_instruction(instr, out) 
             for instr, out in zip(examples["instruction"], examples["output"])]
    return tokenizer(texts, truncation=True, max_length=512)

dataset = dataset.map(preprocess, batched=True)

# === 训练 ===
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # 等效 batch=16
    learning_rate=3e-4,               # LoRA 可以用更大的学习率
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    fp16=True,
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# === 保存（只有适配器，非常小）===
model.save_pretrained("./lora-adapter")
# 只保存了 ~80MB 的适配器权重，而非 13GB 的完整模型
```

### 3.4 加载 LoRA 模型

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 步骤1：加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 步骤2：加载 LoRA 适配器
lora_model = PeftModel.from_pretrained(
    base_model,
    "./lora-adapter"
)

# 现在可以使用微调后的模型了
# 推理效果等同于全参微调，但加载极快

# === 多适配器切换 ===
# 场景：同一个基础模型，不同的 LoRA 适配器
adapter_names = ["lora-coding", "lora-chat", "lora-medical"]

# 加载多个适配器
for name in adapter_names:
    lora_model.load_adapter(f"./adapters/{name}", adapter_name=name)

# 切换使用
lora_model.set_adapter("lora-coding")  # 切到编程适配器
# ... 推理编程问题 ...

lora_model.set_adapter("lora-medical")  # 切到医疗适配器
# ... 推理医疗问题 ...
```

### 3.5 合并 LoRA 到基础模型

```python
# 当需要部署时，可以把 LoRA 权重合并到基础模型中
# 好处：推理时没有额外的计算开销

# 合并并卸载 LoRA 模块
merged_model = lora_model.merge_and_unload()

# 保存合并后的完整模型
merged_model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")

# 合并后的模型可以独立使用，不依赖 peft
```

## 四、Adapter Tuning

### 4.1 Adapter 原理

```
┌─────────────────────────────────────────────────────────────┐
│                    Adapter Tuning 结构                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   原 Transformer 层：                                        │
│   ┌────────────────────────────────┐                        │
│   │   x → [Attention] → [FFN] → y  │                        │
│   └────────────────────────────────┘                        │
│                                                             │
│   加入 Adapter 后：                                           │
│   ┌─────────────────────────────────────────────┐           │
│   │                                             │           │
│   │   x → [Attention] → + → [FFN] → + → y       │           │
│   │                    ↑             ↑          │           │
│   │              ┌─────┴────┐  ┌────┴────┐     │           │
│   │              │ Adapter  │  │ Adapter │     │           │
│   │              │ ┌──────┐ │  │ ┌──────┐│     │           │
│   │              │ │Down  │ │  │ │Down  ││     │ (冻结原模型)│
│   │              │ │d→r   │ │  │ │d→r   ││     │           │
│   │              │ ├──────┤ │  │ ├──────┤│     │           │
│   │              │ │ReLU  │ │  │ │ReLU  ││     │           │
│   │              │ ├──────┤ │  │ ├──────┤│     │           │
│   │              │ │  Up  │ │  │ │  Up  ││     │           │
│   │              │ │r→d   │ │  │ │r→d   ││     │           │
│   │              │ └──────┘ │  │ └──────┘│     │           │
│   │              └──────────┘  └─────────┘     │           │
│   │   ❄ 冻结原层  ✦ 只训练 Adapter              │           │
│   └─────────────────────────────────────────────┘           │
│                                                             │
│   特点：在原层之后插入瓶颈结构（bottleneck）                 │
│   参数量：2×d×r (r远小于d)                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Adapter vs LoRA 对比

| 特性 | Adapter | LoRA |
|------|---------|------|
| 插入位置 | 层内部新增模块 | 原权重的旁路 |
| 推理延迟 | 增加（多了计算层）| 可消除（合并后）|
| 参数量 | ~1-5% | ~0.1-1% |
| 效果 | 好 | 接近全参 |
| 推理速度 | 变慢 | 不变（合并后）|

```python
from transformers import AutoModelForSequenceClassification
from peft import AdapterConfig, get_peft_model

# Adapter 配置
adapter_config = AdapterConfig(
    task_type="SEQ_CLS",
    r=8,                    # 瓶颈维度
    alpha=16,
    dropout=0.1,
)

# peft 库也支持 Adapter（但 LoRA 更主流）
# 更多时候直接用 LoRA
```

## 五、Prefix Tuning / Prompt Tuning

### 5.1 Prefix Tuning

```
┌─────────────────────────────────────────────────────────────┐
│                   Prefix Tuning 原理                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  原始输入：                                                  │
│  [token1] [token2] [token3] ... [tokenN]                    │
│                                                             │
│  Prefix Tuning：                                            │
│  [P1][P2]...[P_k] [token1] [token2] ... [tokenN]            │
│   ↑ 可训练的前缀   ↑ 实际输入（冻结）                         │
│   (虚拟 token)                                               │
│                                                             │
│  ┌──────────────────────────────────────────────┐           │
│  │  [P_1][P_2]...[P_k]  这 k 个"虚拟 token"      │           │
│  │  的 Key/Value 是可训练参数                      │           │
│  │                                               │           │
│  │  它们不对应任何实际词，而是直接优化 embedding    │           │
│  │  来引导模型行为                                │           │
│  └──────────────────────────────────────────────┘           │
│                                                             │
│  在每层 Attention 的 KV cache 前面加上可训练的前缀            │
│                                                             │
│         Attention Layer                                      │
│         ┌────────────────────────────────────┐              │
│  Query  │ [Q₁][Q₂]...[Qₙ]                    │              │
│         │                                     │              │
│  Key    │ [K_p₁][K_p₂]...[K_pk][K₁][K₂]...[Kₙ]│  ← 前缀可训练│
│         │                                     │              │
│  Value  │ [V_p₁][V_p₂]...[V_pk][V₁][V₂]...[Vₙ]│  ← 前缀可训练│
│         └────────────────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Prompt Tuning

```python
"""
Prompt Tuning vs Prefix Tuning vs LoRA 的区别
"""

comparison = """
┌──────────────┬────────────���─┬──────────────┬──────────────┐
│     方法      │  可训练参数   │   插入位置    │   推理开销    │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Prompt Tuning│  仅输入层     │  输入端       │  几乎无       │
│              │  k×d 参数    │  (Embedding)  │              │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Prefix Tuning│  每层 KV     │  所有层的     │  小          │
│              │  L×k×2×d    │  Attention    │              │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ LoRA         │  Q/V 权重旁路 │  权重矩阵     │  可消除       │
│              │  2×r×d      │  旁路         │  (合并后)     │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 全参微调      │  所有参数     │  全部         │  无          │
│              │  100%        │              │              │
└──────────────┴──────────────┴──────────────┴──────────────┘
"""

# Prompt Tuning 代码示例
from peft import PromptTuningConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

prompt_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,       # 虚拟 token 数量
    token_dim=model.config.hidden_size,
    num_transformer_submodules=1,
    num_attention_layers=model.config.num_hidden_layers,
    num_attention_heads=model.config.num_attention_heads,
)

prompt_model = get_peft_model(model, prompt_config)
prompt_model.print_trainable_parameters()
# 参数量极少：~20×1024 = 20K 参数
```

### 5.3 Prefix Tuning 实现

```python
from peft import PrefixTuningConfig, get_peft_model

prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,       # 前缀长度
    token_dim=768,               # 与模型隐藏维度一致
    num_transformer_submodules=1,
    num_attention_layers=12,
    num_attention_heads=12,
    encoder_hidden_size=768,
)

prefix_model = get_peft_model(model, prefix_config)
prefix_model.print_trainable_parameters()
```

## 六、QLoRA（量化 LoRA）

### 6.1 QLoRA 概念

QLoRA = Quantization + LoRA，先将基础模型量化到 4-bit，然后在量化模型上应用 LoRA。

```
┌─────────────────────────────────────────────────────────────┐
│                      QLoRA 流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  步骤1：加载预训练模型（量化到 4-bit）                        │
│  ┌─────────────────────────────────────────────┐            │
│  │  原模型 FP16 (13GB for 7B)                   │            │
│  │       ↓ 量化 (NF4)                           │            │
│  │  4-bit 模型 (~3.5GB for 7B)                  │            │
│  └──────────────────────┬──────────────────────┘            │
│                         │                                   │
│  步骤2：添加 LoRA 适配器（FP16 精度）                        │
│  ┌──────────────────────▼──────────────────────┐            │
│  │  4-bit 冻结权重  +  FP16 LoRA 适配器          │            │
│  │  ████████████████    ████ (可训练, 很小)      │            │
│  └──────────────────────┬──────────────────────┘            │
│                         │                                   │
│  步骤3：训练（只更新 LoRA 参数）                              │
│  ┌──────────────────────▼──────────────────────┐            │
│  │  前向传播时：4-bit → FP16 → 计算              │            │
│  │  反向传播时：只更新 LoRA 的 FP16 参数          │            │
│  │  显存只需 ~6GB (7B模型)                       │            │
│  └─────────────────────────────────────────────┘            │
│                                                             │
│  奇迹：在一张 24GB 消费级 GPU 上微调 70B 模型！              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 QLoRA 代码实现

```python
"""
QLoRA：在单卡 24GB GPU 上微调 LLaMA-7B
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# === 步骤1：4-bit 量化配置 ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 4-bit 量化加载
    bnb_4bit_quant_type="nf4",           # Normal Float 4 量化类型
    bnb_4bit_compute_dtype=torch.float16, # 计算时反量化为 FP16
    bnb_4bit_use_double_quant=True,      # 双重量化（进一步压缩）
)

# === 步骤2：加载量化模型 ===
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# 为量化训练做准备
model = prepare_model_for_kbit_training(model)

# === 步骤3：配置 LoRA ===
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # QLoRA 通常用稍大的 r
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 40M || all params: 3,540M || trainable%: 1.13%

# 显存使用对比
print("""
显存对比（LLaMA-7B）：
  全参数微调 (FP16):   ~60 GB   需要 4× A100(40GB)
  LoRA (FP16):         ~16 GB   需要 1× A100(40GB)
  QLoRA (4-bit):       ~6 GB    只需 1× RTX 3090(24GB)  ← 消费级！
""")
```

### 6.3 NF4 量化原理

```
┌─────────────────────────────────────────────────────────────┐
│              NF4 (Normal Float 4) 量化                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  理论基础：预训练权重服从正态分布                              │
│                                                             │
│  普通 INT4（均匀量化）：                                      │
│  -128 ────┬───┬───┬───┬───┬───┬───┬───┬──── 127              │
│           │   │   │   │   │   │   │   │                      │
│  问题：权重集中在 0 附近，大量量化等级被浪费                  │
│                                                             │
│  NF4（正态分布量化）：                                       │
│  -1.0 ────┬───┬───┬───┬──┬─┬──┬───┬───┬──── 1.0             │
│           │   │   │   │  │ │  │   │   │                      │
│           ←─ 密 ─→  ←密→ ←密→ ← ─ 疏 ──→                   │
│  量化等级按正态分布排列，0 附近密集，两端稀疏                 │
│                                                             │
│  结果：同样 4-bit，NF4 的量化误差更小                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 七、LoRA 超参数选择指南

### 7.1 rank r 的选择

```
┌─────────────────────────────────────────────────────────────┐
│                  rank r 选择指南                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  r=4~8                                                     │
│  ├── 通用任务（分类、简单生成）                               │
│  ├── 数据量少（< 10K 样本）                                  │
│  └── 模型大（> 13B，本身能力强）                              │
│                                                             │
│  r=16~64                                                   │
│  ├── 复杂任务（代码生成、推理）                               │
│  ├── 数据量中等（10K~100K）                                 │
│  └── 模型中等（7B~13B）                                     │
│                                                             │
│  r=128+                                                    │
│  ├── 领域差异大（医疗→代码）                                 │
│  ├── 数据量大（> 100K）                                     │
│  └── 追求最佳效果（不在乎参数量）                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
# 经验性推荐
lora_recommendations = {
    # 任务 → (r, alpha, target_modules)
    "文本分类":    (8, 16, ["q_proj", "v_proj"]),
    "命名实体识别": (16, 32, ["q_proj", "v_proj"]),
    "文本生成":    (16, 32, ["q_proj", "k_proj", "v_proj", "o_proj"]),
    "代码生成":    (32, 64, ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]),
    "数学推理":    (64, 128, ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"]),
    "多轮对话":    (32, 64, ["q_proj", "k_proj", "v_proj", "o_proj"]),
}
```

### 7.2 学习率选择

```python
# LoRA 学习率通常比全参微调大
lr_guide = """
全参数微调学习率:    1e-5 ~ 2e-5
LoRA 学习率:        1e-4 ~ 5e-4   (大约 10 倍)
Prompt Tuning:      1e-3 ~ 5e-3   (更大)

原因：
  全参微调更新所有参数，小学习率避免破坏预训练知识
  LoRA 只更新少量新增参数，大学习率加速收敛
  
QLoRA 额外建议：
  配合 cosine 调度器效果更好
  warmup 建议 3% ~ 10% 总步数
"""
```

## 八、PEFT 方法全面对比

```
┌─────────────────────────────────────────────────────────────┐
│                PEFT 方法对比总表                              │
├──────────┬─────────┬──────────┬─────────┬─────────┬────────┤
│  方法     │参数量(%) │ 推理开销  │ 效果    │ 训练速度│ 复杂度 │
├──────────┼─────────┼──────────┼─────────┼─────────┼────────┤
│ 全参微调  │ 100%    │ 无       │ ★★★★★  │ 慢      │ 低     │
│ LoRA     │ 0.1-1%  │ 可消除   │ ★★★★☆  │ 快      │ 低     │
│ QLoRA    │ 0.1-1%  │ 可消除   │ ★★★★☆  │ 快      │ 中     │
│ Adapter  │ 1-5%    │ 有       │ ★★★★☆  │ 中      │ 中     │
│ Prefix   │ 0.1-1%  │ 小       │ ★★★☆☆  │ 快      │ 高     │
│ Prompt   │ <0.1%   │ 极小     │ ★★★☆☆  │ 极快    │ 低     │
└──────────┴─────────┴──────────┴─────────┴─────────┴────────┘
```

| 使用场景 | 推荐方法 | 原因 |
|---------|---------|------|
| 消费级 GPU 微调大模型 | QLoRA | 显存需求最低 |
| 追求最佳效果 | LoRA (大 r) | 效果接近全参 |
| 多任务切换 | LoRA | 适配器切换快 |
| 极致参数效率 | Prompt Tuning | 参数最少 |
| 生产部署 | LoRA (合并后) | 推理无额外开销 |

## 小结

| 要点 | 内容 |
|------|------|
| PEFT 动机 | 大模型全参微调太贵，PEFT 只训练 0.1%~1% 参数 |
| LoRA 原理 | W = W₀ + BA，低秩分解近似权重更新 |
| LoRA 超参 | r=8/16/64，alpha=2r，target=Q/V 或全部线性层 |
| QLoRA | 4-bit 量化 + LoRA，单卡微调 7B 模型 |
| Adapter | 在层内插入瓶颈模块，推理有额外开销 |
| Prefix/Prompt Tuning | 优化虚拟前缀 token，参数极少 |
| 学习率 | PEFT 可用更大学习率（1e-4 ~ 5e-4）|
| 部署 | LoRA 可合并到基础模型，消除推理开销 |

```
┌─────────────────────────────────────────────────────────────┐
│                   PEFT 选择决策树                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  显存 < 24GB？                                               │
│    ├─ 是 → QLoRA (4-bit + LoRA)                             │
│    └─ 否 → 显存 < 80GB？                                     │
│            ├─ 是 → LoRA (r=8~16)                            │
│            └─ 否 → 效果要求极高？                             │
│                    ├─ 是 → 全参微调                          │
│                    └─ 否 → LoRA (r=32~64)                   │
│                                                             │
│  需要多任务？                                                │
│    └─ 是 → LoRA（多个适配器，随时切换）                      │
│                                                             │
│  需要部署到生产？                                            │
│    └─ 是 → LoRA → merge_and_unload()（合并消除开销）        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

| [← 回到目录](../README.md) | [上一篇：Prompt Engineering](03-Prompt-Engineering.md) | [下一篇：模型量化与推理优化](05-模型量化与推理优化.md) |
|---|---|---|
