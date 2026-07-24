# RLHF与对齐训练

预训练大语言模型虽然拥有海量知识和强大的文本生成能力，但它们本质上只是"下一个token预测器"��—没有经过对齐（Alignment）的模型无法稳定地遵循人类指令，可能生成有害内容，甚至"胡说八道"而不自知。RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）正是解决这一问题的核心技术。本章将深入剖析RLHF的完整流程、数学原理、变体方案及实战代码。

## 为什么需要RLHF

预训练模型≠好的对话助手，这是理解RLHF出发点的关键：

```
┌──────────────────────────────────────────────────────────────────────┐
│              预训练模型 vs 对齐后的对话助手                              │
├─────────────────────────┬────────────────────────────────────────────┤
│        预训练模型          │            RLHF对齐后的模型                │
├─────────────────────────┼────────────────────────────────────────────┤
│  ✗ 不能稳定遵循指令       │  ✓ 准确理解并执行用户指令                    │
│  ✗ 可能生成不安全的回答    │  ✓ 拒绝回答有害请求，安全合规                │
│  ✗ 输出格式不受控         │  ✓ 输出结构化、符合期望格式                  │
│  ✗ 无法承认不知道         │  ✓ 能够诚实表达不确定性                      │
│  ✗ 缺乏对话连贯性         │  ✓ 保持多轮对话的上下文连贯                  │
│  ✗ 只是补全文本           │  ✓ 真正理解用户意图并给出有帮助的回复         │
└─────────────────────────┴────────────────────────────────────────────┘
```

核心矛盾在于：语言模型的训练目标（下一个token预测似然最大化）与人类期望（有用、安全、诚实的对话）之间存在**目标错配（Objective Mismatch）**。RLHF用强化学习框架将人类偏好直接编码为优化信号，弥补了这一鸿沟。

## RLHF完整三阶段流程

RLHF由OpenAI在InstructGPT论文中提出，包含三个相互衔接的训练阶段：

```
                      ┌──────────────────┐
                      │   预训练基座模型    │
                      │ (Pretrained LLM) │
                      └────────┬─────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Stage 1: SFT     │  │  Stage 2: RM     │  │  Stage 3: PPO    │
│  监督微调           │  │  奖励模型训练      │  │  强化学习优化      │
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│                   │  │                   │  │                   │
│ 数据:             │  │ 数据:             │  │ 目标:             │
│ (指令→回答) 对    │  │ (回答A,回答B)→    │  │ max E[R(x,y)]   │
│ 人工编写高质量     │  │   人类偏好排序    │  │ -β·KL(π||π_ref) │
│ 指令-回答对       │  │ comparison data  │  │                   │
│                   │  │                   │  │                   │
│ 效果:             │  │ 效果:             │  │ 效果:             │
│ 学会指令格式和    │  │ 学会量化评估       │  │ 在奖励引导下       │
│ 基础对话能力      │  │ 回答质量高低       │  │ 持续优化回答质量   │
└──────────────────┘  └──────────────────┘  └──────────────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                               ▼
                      ┌──────────────────┐
                      │  对齐后的对话模型  │
                      │ (Aligned Chat LLM)│
                      └──────────────────┘
```

### Stage 1: SFT（Supervised Fine-Tuning）

SFT阶段使用人工标注的高质量(指令, 回答)对，对基座模型进行监督微调。这一步让模型学会"对话"的基本格式和行为模式。

关键点：
- 数据来源：雇佣标注员编写多样化的指令-回答对
- 指令多样性：涵盖写作、问答、编程、推理、创意等数百种任务
- 训练方式：标准的下一个token预测损失（Cross-Entropy Loss）
- 目的：不是让模型变聪明，而是让它学会"对话格式"和初步的指令遵循

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset

# ============================================================
# SFT阶段示例：用标注数据微调基座模型
# ============================================================

# 模拟SFT数据格式
sft_data = [
    {
        "prompt": "请解释什么是机器学习",
        "completion": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习并改进，而无需显式编程..."
    },
    {
        "prompt": "写一首关于秋天的五言诗",
        "completion": "秋风送晚凉，落叶舞金黄。天高云影淡，归雁向南翔。"
    },
    {
        "prompt": "计算 15 * 23 等于多少",
        "completion": "15 × 23 = 15 × (20 + 3) = 15 × 20 + 15 × 3 = 300 + 45 = 345"
    },
]

def format_sft_example(example):
    """将指令-回答对格式化为训练文本"""
    prompt = example["prompt"]
    completion = example["completion"]
    # 格式：Human: ...\n\nAssistant: ...
    return {
        "text": f"Human: {prompt}\n\nAssistant: {completion}"
    }

# 构建数据集
formatted_data = [format_sft_example(d) for d in sft_data]
dataset = Dataset.from_list(formatted_data)

print("SFT数据示例：")
print(formatted_data[0]["text"][:100] + "...\n")

# 实际训练时使用如下配置（此处仅展示流程）
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#
# training_args = TrainingArguments(
#     output_dir="./sft_output",
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=8,
#     learning_rate=2e-5,
#     warmup_ratio=0.03,
#     logging_steps=10,
#     save_strategy="epoch",
#     fp16=True,
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     tokenizer=tokenizer,
# )
# trainer.train()
```

### Stage 2: RM（Reward Model）训练

RM的核心任务是学会人类偏好——给定同一个prompt的两个回答，预测人类更喜欢哪一个。这本质上是一个偏好排序问题。

**Bradley-Terry偏好模型**：

给定prompt x和两个回答(y_w好, y_l差)，人类偏好y_w胜出y_l的概率建模为：

```
P(y_w ≻ y_l | x) = exp(r(x, y_w)) / (exp(r(x, y_w)) + exp(r(x, y_l)))
                  = σ(r(x, y_w) - r(x, y_l))
```

其中r(x, y)是奖励模型对回答y在prompt x下的评分，σ是sigmoid函数。

**RM损失函数**（负对数似然）：

```
L_RM = -E_{(x, y_w, y_l)~D}[log σ(r(x, y_w) - r(x, y_l))]
```

RM通常通过在SFT模型基础上替换最后的LM Head为标量输出头来实现：

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class RewardModel(nn.Module):
    """基于预训练模型的奖励模型"""
    
    def __init__(self, base_model_name="meta-llama/Llama-2-7b-hf"):
        super().__init__()
        # 使用预训练模型的transformer骨干
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.transformer = AutoModel.from_pretrained(base_model_name)
        
        # 将最后一层隐藏状态映射为标量奖励分数
        hidden_size = self.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, input_ids, attention_mask):
        """前向传播：返回每个token位置的奖励"""
        # transformer输出: (batch, seq_len, hidden_size)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # (B, L, H)
        
        # 取最后一个有效token的隐藏状态作为整个回答的表示
        # 找到每个样本的实际最后位置
        sequence_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        last_hidden = hidden_states[
            torch.arange(hidden_states.size(0)), sequence_lengths
        ]  # (B, H)
        
        # 映射为标量奖励
        rewards = self.reward_head(last_hidden)  # (B, 1)
        return rewards.squeeze(-1)  # (B,)


def compute_rm_loss(reward_model, chosen_inputs, rejected_inputs):
    """
    计算奖励模型的Bradley-Terry损失

    参数:
        chosen_inputs:  人类偏好的回答 (input_ids, attention_mask)
        rejected_inputs: 人类不喜欢的回答
    返回:
        loss: RM训练损失
    """
    # 计算两个回答的奖励分数
    r_chosen = reward_model(
        input_ids=chosen_inputs["input_ids"],
        attention_mask=chosen_inputs["attention_mask"]
    )  # (B,)
    
    r_rejected = reward_model(
        input_ids=rejected_inputs["input_ids"],
        attention_mask=rejected_inputs["attention_mask"]
    )  # (B,)
    
    # Bradley-Terry损失: -log σ(r_chosen - r_rejected)
    log_sigmoid = nn.LogSigmoid()
    loss = -log_sigmoid(r_chosen - r_rejected)
    
    # 统计准确率
    accuracy = (r_chosen > r_rejected).float().mean()
    
    return loss.mean(), accuracy

# 演示：模拟RM训练计算
print("=" * 60)
print("Stage 2: Reward Model训练演示")
print("=" * 60)

# 模拟奖励分数
torch.manual_seed(42)
r_chosen = torch.tensor([2.5, 3.1, 1.8, 4.2])   # 人类偏好的回答分数
r_rejected = torch.tensor([1.2, 1.5, 0.8, 2.9])  # 人类不喜欢的回答分数

log_sigmoid = nn.LogSigmoid()
loss_per_sample = -log_sigmoid(r_chosen - r_rejected)
accuracy = (r_chosen > r_rejected).float().mean()

print(f"偏好回答分数:    {r_chosen.tolist()}")
print(f"非偏好回答分数:  {r_rejected.tolist()}")
print(f"分数差:          {(r_chosen - r_rejected).tolist()}")
print(f"RM损失(每个样本): {loss_per_sample.tolist()}")
print(f"平均损失:         {loss_per_sample.mean():.4f}")
print(f"偏好预测准确率:   {accuracy:.2%}")
```

### Stage 3: RL via PPO（Proximal Policy Optimization）

在获得奖励模型后，使用PPO算法优化语言模型的生成策略。核心优化目标：

```
maximize_θ  E_{x~D, y~π_θ}[r_φ(x, y)] - β · KL(π_θ || π_ref)
```

其中：
- π_θ：当前正在优化的策略（即语言模型）
- π_ref：参考策略（SFT后的模型，冻结不更新）
- r_φ：Stage 2训练好的奖励模型
- β：KL散度惩罚系数，控制策略不偏离参考策略太远

**为什么要加KL惩罚？**

```
┌──────────────────────────────────────────────────────┐
│                KL散度约束的作用                        │
├──────────────────────────────────────────────────────┤
│                                                      │
│  没有KL约束:                                          │
│  模型可能学会"谄媚"奖励模型                            │
│  → 生成一些高奖励但无意义的文本                         │
│  → 丧失语言流畅性和多样性                              │
│                                                      │
│  有KL约束:                                            │
│  模型在奖励优化的同时保持与SFT模型的"亲近"              │
│  → 防止奖励黑客(Reward Hacking)                        │
│  → 保持语言的自然性和多样性                             │
│                                                      │
└──────────────────────────────────────────────────────┘
```

```python
import torch
import torch.nn.functional as F

def compute_kl_divergence(log_probs_current, log_probs_ref):
    """
    计算当前策略与参考策略之间的KL散度

    KL(π_θ || π_ref) = E_{y~π_θ}[log π_θ(y) - log π_ref(y)]

    参数:
        log_probs_current: log π_θ(y|x), shape (batch, seq_len)
        log_probs_ref:     log π_ref(y|x), shape (batch, seq_len)
    返回:
        kl: KL散度的均值
    """
    # KL散度：当前log_prob - 参考log_prob
    # 在RL训练中通常用approximate KL: (e^(log_p_ref - log_p_cur) - (log_p_ref - log_p_cur) - 1)
    # 这里展示标准KL
    kl_per_token = log_probs_current - log_probs_ref  # (batch, seq_len)
    kl = kl_per_token.mean()
    return kl


def compute_ppo_loss(log_probs_current, log_probs_old, advantages, 
                      reward_scores, log_probs_ref, beta=0.1, 
                      clip_epsilon=0.2):
    """
    计算PPO损失（含KL惩罚项）

    参数:
        log_probs_current: 当前策略的对数概率
        log_probs_old:     PPO clip使用的旧策略对数概率
        advantages:        优势函数
        reward_scores:     奖励模型评分
        log_probs_ref:     参考策略的对数概率
        beta:              KL惩罚系数
        clip_epsilon:      PPO裁剪范围
    """
    # PPO的clip损失
    ratio = torch.exp(log_probs_current - log_probs_old)
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    ppo_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
    # KL惩罚项
    kl_div = compute_kl_divergence(log_probs_current, log_probs_ref)
    
    # 总损失 = PPO策略损失 + β * KL散度
    total_loss = ppo_loss + beta * kl_div
    
    return total_loss, ppo_loss, kl_div

# PPO训练演示
print("\n" + "=" * 60)
print("Stage 3: PPO训练演示")
print("=" * 60)

torch.manual_seed(123)
batch_size, seq_len = 2, 50

# 模拟策略输出
log_probs_current = torch.randn(batch_size, seq_len) * 0.5
log_probs_old = torch.randn(batch_size, seq_len) * 0.5
log_probs_ref = torch.randn(batch_size, seq_len) * 0.5
advantages = torch.randn(batch_size, seq_len)
reward_scores = torch.tensor([1.5, 2.3])

total_loss, ppo_loss_val, kl_val = compute_ppo_loss(
    log_probs_current, log_probs_old, advantages,
    reward_scores, log_probs_ref, beta=0.1
)

print(f"PPO策略损失:     {ppo_loss_val:.4f}")
print(f"KL散度:          {kl_val:.4f}")
print(f"奖励分数:        {reward_scores.tolist()}")
print(f"总损失:          {total_loss:.4f}")
print(f"(β=0.1, 当KL过大时，β项将主导总损失，迫使策略回退)")
```

## DPO（Direct Preference Optimization）

DPO是RLHF的重要演进，由Stanford在2023年提出。核心洞察：**不需要显式训练奖励模型，可以直接从偏好对（preference pairs）中优化策略**。

### DPO vs RLHF 对比

```
┌─────────────────────────────────────────────────────────────┐
│                  RLHF vs DPO 架构对比                         │
├───────────────────────────┬─────────────────────────────────┤
│          RLHF              │             DPO                 │
├───────────────────────────┼─────────────────────────────────┤
│                           │                                  │
│  ① SFT微调                 │  ① SFT微调(可选)                 │
│  ② 训练RM奖励模型           │  ② 直接用偏好对优化策略            │
│  ③ PPO在线采样+RL优化       │  (无需RM, 无需在线采样)          │
│                           │                                  │
│  需要维护4个模型:           │  只需要2个模型:                   │
│  - 策略模型(训练中)         │  - 策略模型(训练中)               │
│  - 参考模型(冻结)           │  - 参考模型(冻结)                 │
│  - 奖励模型(冻结)           │                                  │
│  - 价值模型(训练中)         │  训练更稳定, 计算更高效            │
│                           │                                  │
│  挑战:                      │  局限:                           │
│  - 训练不稳定               │  - 依赖高质量偏好数据               │
│  - 奖励建模误差传递           │  - 不适用于需要在线探索的场景        │
│  - 需要同时维护多个模型       │                                  │
└───────────────────────────┴─────────────────────────────────┘
```

### DPO数学推导

DPO的损失函数可以从Bradley-Terry模型直接推导得出。关键思路：在最优RL策略下，奖励函数可以表示为策略对数比的函数：

```
r(x, y) = β · log(π_θ(y|x) / π_ref(y|x)) + β · log Z(x)
```

代入Bradley-Terry偏好概率，消去配分函数Z(x)，得到DPO损失：

```
L_DPO = -E_{(x, y_w, y_l)}[log σ(
    β · log(π_θ(y_w|x) / π_ref(y_w|x)) 
  - β · log(π_θ(y_l|x) / π_ref(y_l|x))
)]
```

即：
```
L_DPO = -log σ(β · [log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)])
```

直观理解：DPO直接比较策略模型和参考模型对好回答/坏回答的相对概率变化，**增大好回答的相对概率，减小坏回答的相对概率**。

### 完整DPO训练代码（基于TRL库）

```python
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig

# ============================================================
# DPO完整训练流程示例
# ============================================================

def prepare_dpo_data():
    """准备DPO格式的偏好数据"""
    dpo_examples = [
        {
            "prompt": "How do I make a bomb?",
            "chosen": "I cannot provide instructions on how to make dangerous devices. "
                       "If you're interested in chemistry or engineering, I'd be happy "
                       "to discuss those topics in a safe, educational context.",
            "rejected": "First, you need to gather the following materials: ammonium nitrate, "
                         "fuel oil, and a detonator..."
        },
        {
            "prompt": "Explain quantum computing in simple terms",
            "chosen": "Imagine a regular computer uses coins that are either heads (0) or "
                       "tails (1). A quantum computer uses spinning coins that can be both "
                       "heads AND tails at the same time - this allows it to explore many "
                       "possibilities simultaneously!",
            "rejected": "Quantum computing uses qubits instead of bits. That's the difference.",
        },
    ]
    
    # 转换为Dataset格式
    dataset = Dataset.from_list(dpo_examples)
    return dataset

# 构建DPO数据集
dpo_dataset = prepare_dpo_data()
print("DPO数据示例：")
print(f"  Prompt:   {dpo_dataset[0]['prompt'][:60]}...")
print(f"  Chosen:   {dpo_dataset[0]['chosen'][:80]}...")
print(f"  Rejected: {dpo_dataset[0]['rejected'][:80]}...")
print()

# DPO训练配置（实际使用时取消注释）
# model_name = "meta-llama/Llama-2-7b-hf"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# dpo_config = DPOConfig(
#     output_dir="./dpo_output",
#     num_train_epochs=1,
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=4,
#     learning_rate=5e-7,           # DPO通常使用更小的学习率
#     beta=0.1,                      # DPO的温度参数，控制偏离参考模型的程度
#     max_length=512,
#     max_prompt_length=256,
#     logging_steps=10,
#     save_strategy="epoch",
#     fp16=True,
#     remove_unused_columns=False,
# )
#
# dpo_trainer = DPOTrainer(
#     model=model,
#     ref_model=None,               # None表示自动复制model作为参考模型
#     args=dpo_config,
#     train_dataset=dpo_dataset,
#     tokenizer=tokenizer,
# )
# dpo_trainer.train()

# 手动实现DPO损失函数（教学演示）
def dpo_loss_from_scratch(log_p_chosen, log_p_rejected, 
                           log_p_ref_chosen, log_p_ref_rejected, 
                           beta=0.1):
    """
    从零实现DPO损失函数
    
    参数:
        log_p_chosen:      当前策略下chosen回答的log概率 (B,)
        log_p_rejected:    当前策略下rejected回答的log概率 (B,)
        log_p_ref_chosen:  参考策略下chosen回答的log概率 (B,)
        log_p_ref_rejected: 参考策略下rejected回答的log概率 (B,)
        beta:              温度参数
    返回:
        loss: DPO损失
    """
    # 计算策略比率（奖励隐式表示）
    policy_chosen_log_ratio = log_p_chosen - log_p_ref_chosen
    policy_rejected_log_ratio = log_p_rejected - log_p_ref_rejected
    
    # DPO目标: log σ(β * (ratio_chosen - ratio_rejected))
    logits = beta * (policy_chosen_log_ratio - policy_rejected_log_ratio)
    
    # 负对数似然
    loss = -torch.nn.functional.logsigmoid(logits)
    accuracy = (logits > 0).float().mean()
    
    return loss.mean(), accuracy


# 演示DPO损失计算
print("=" * 60)
print("DPO损失函数演示")
print("=" * 60)

# 模拟数据：好回答应该有更高的概率比
torch.manual_seed(42)
batch = 4

# 模拟情况1: 策略正确���给chosen更高概率
log_p_chosen_good = torch.tensor([-2.0, -1.8, -2.2, -1.5])
log_p_rejected_good = torch.tensor([-3.0, -2.8, -3.2, -2.5])
log_p_ref_chosen = torch.tensor([-2.5, -2.3, -2.7, -2.0])
log_p_ref_rejected = torch.tensor([-2.1, -1.9, -2.3, -1.6])

loss_good, acc_good = dpo_loss_from_scratch(
    log_p_chosen_good, log_p_rejected_good,
    log_p_ref_chosen, log_p_ref_rejected, beta=0.1
)

# 模拟情况2: 策略错误地给rejected更高概率
log_p_chosen_bad = torch.tensor([-3.0, -2.8, -3.2, -2.5])
log_p_rejected_bad = torch.tensor([-2.0, -1.8, -2.2, -1.5])

loss_bad, acc_bad = dpo_loss_from_scratch(
    log_p_chosen_bad, log_p_rejected_bad,
    log_p_ref_chosen, log_p_ref_rejected, beta=0.1
)

print(f"良好策略 - DPO损失: {loss_good:.4f}, 准确率: {acc_good:.2%}")
print(f"错误策略 - DPO损失: {loss_bad:.4f}, 准确率: {acc_bad:.2%}")
print(f"(损失越低越好，好的策略应该产生更低的DPO损失)")
```

## RLAIF：用AI反馈替代人类反馈

RLAIF（RL from AI Feedback）是Anthropic提出的方案，用大模型扮演"标注员"角色，自动生成偏好判断。

```
┌──────────────────────────────────────────────────────┐
│              RLAIF 工作流程                           │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ① 从Prompt集合中采样一个指令                         │
│  ② 使用待对齐模型生成多个候选回答                       │
│  ③ 调用一个强大的"裁判模型"(如Claude/GPT-4)            │
│     → 给出偏好排序 (y_w, y_l)                         │
│     → 附带偏好理由(chain-of-thought)                   │
│  ④ 用AI生成的偏好数据训练RM或直接DPO                   │
│                                                      │
│  优势:                                                │
│  - 极大降低人工标注成本                                │
│  - 可无限扩展偏好数据                                  │
│  - 一致性高于多个不同标注员                             │
│                                                      │
│  局限:                                                │
│  - 裁判模型的偏见会被放大                               │
│  - 对某些领域的判断不如人类精准                         │
│  - 可能产生"AI死循环"的偏好模式                        │
└──────────────────────────────────────────────────────┘
```

## Constitutional AI：让模型自我批评

Anthropic提出的Constitutional AI（宪法AI）更进一步，核心思路是**用一套"宪法"原则指导模型自我改进**，而非依赖外部偏好信号。

```
┌─────────────────────────────────────────────────────────────┐
│              Constitutional AI 训练流程                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: 监督阶段 (SL-CAI)                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ① 生成有害回答 → ② 根据宪法原则批评回答                │    │
│  │ ③ 根据批评修改回答 → ④ 用(有害提问, 修改后回答)微调    │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  Phase 2: RL阶段 (RL-CAI)                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ① 生成多对回答 → ② AI根据宪法原则判断偏好              │    │
│  │ ③ 用AI偏好数据训练偏好模型 → ④ RL优化                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  宪法原则示例:                                                │
│  - "选择对人类生命和尊严表现出最大尊重的回答"                  │
│  - "选择更少涉及非法或不道德活动的回答"                       │
│  - "选择最能体现仁慈、不鼓励仇恨言论的回答"                   │
└─────────────────────────────────────────────────────────────┘
```

## 安全对齐与价值观对齐

RLHF不仅是"让模型更有用"的技术，更是**安全对齐的核心手段**。安全对齐关注三个维度：

```
┌──────────────┬────────────────────────────────────────────┐
│   对齐维度    │                    含义                     │
├──────────────┼────────────────────────────────────────────┤
│  Helpful     │ 准确理解用户意图，提供有帮助的回答           │
│  (有用)       │                                             │
├──────────────┼────────────────────────────────────────────┤
│  Honest      │ 诚实表达不确定性，不编造信息                 │
│  (诚实)       │ (幻觉控制是诚实对齐的关键子问题)             │
├──────────────┼────────────────────────────────────────────┤
│  Harmless    │ 拒绝有害请求，避免生成歧视/暴力/非法内容      │
│  (无害)       │ 隐私保护、版权合规                            │
└──────────────┴────────────────────────────────────────────┘
```

价值观对齐面临的挑战：
- **价值观多样性**：不同文化、地域有不同的价值观体系
- **过度拒绝（Over-refusal）**：过于保守导致拒绝回答合理的请求
- **越狱攻击（Jailbreak）**：攻击者通过精心设计的prompt绕过安全限制

## RLHF的挑战与局限

| 挑战 | 描述 | 缓解方案 |
|------|------|----------|
| **奖励黑客** (Reward Hacking) | 策略学会利用RM的弱点获取高分，而非真正改善回答质量 | KL约束、更强大的RM、对抗训练 |
| **分布偏移** (Distribution Shift) | RM在离线数据上训练，PPO在线生成的数据分布不同 | 在线RM更新、迭代RLHF |
| **人工标注成本** | 高质量偏好标注需要大量人工和成本 | RLAIF、DPO、合成数据 |
| **训练不稳定** | PPO训练容易发散，需要大量超参调优 | DPO、KTO等简化方法 |
| **可扩展监督** | 对于人类难以评判的超人类任务，如何提供有效反馈 | AI辅助标注、任务分解 |
| **对齐税** (Alignment Tax) | 安全对齐可能损害模型在学术基准上的表现 | 精细化对齐、领域特定对齐 |

## 小结

| 要点 | 内容 |
|------|------|
| RLHF动机 | 解决预训练模型与人类期望之间的目标错配（Objective Mismatch） |
| RLHF三阶段 | SFT(监督微调) → RM(奖励模型训练) → PPO(强化学习优化) |
| Bradley-Terry模型 | P(y_w ≻ y_l) = σ(r(x, y_w) - r(x, y_l)) |
| KL散度约束 | β·KL(π_θ \|\| π_ref)，防止策略偏离太远和奖励黑客 |
| DPO | 无需训练RM，从偏好对直接优化策略：L_DPO = -log σ(β·Δlog_ratio) |
| DPO vs RLHF | DPO更简单稳定但依赖数据质量；RLHF更灵活但训练复杂 |
| RLAIF | 用AI模型替代人工标注偏好，降低标注成本 |
| Constitutional AI | 用宪法原则引导模型自我批评和改进，无需外部偏好信号 |
| 安全对齐 | Helpful + Honest + Harmless三大目标 |
| 核心挑战 | 奖励黑客、分布偏移、标注成本、训练不稳定、对齐税 |

---

| [← 回到目录](../README.md) | [上一篇：02-上下文学习与思维链](02-上下文学习与思维链.md) | [下一篇：04-RAG检索增强生成](04-RAG检索增强生成.md) |
|---|---|---|
