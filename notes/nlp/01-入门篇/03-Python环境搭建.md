# Python 环境搭建

工欲善其事，必先利其器。本章搭建完整的 NLP 开发环境。

## 环境总览

```
NLP 开发环境全景:

┌──────────────────────────────────────────────────┐
│                   应用层                         │
│  LangChain │ LlamaIndex │ Streamlit │ Gradio    │
├──────────────────────────────────────────────────┤
│                  模型框架层                      │
│  HuggingFace Transformers │ vLLM │ DeepSpeed    │
├──────────────────────────────────────────────────┤
│                   NLP 工具层                     │
│  spaCy │ NLTK │ Jieba │ HanLP │ stanza          │
├──────────────────────────────────────────────────┤
│                  深度学习层                      │
│  PyTorch │ TensorFlow │ JAX                     │
├──────────────────────────────────────────────────┤
│                   基础层                         │
│  Python 3.10+ │ Conda/UV │ pip                  │
└──────────────────────────────────────────────────┘
```

## 1. Python 环境

### Conda (推荐)

```bash
# === 安装 Conda (Miniconda 推荐) ===
# Windows: 从 https://docs.conda.io/en/latest/miniconda.html 下载
# 或使用 winget:
winget install Miniconda3

# macOS / Linux:
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh

# === 创建 NLP 专用环境 ===
conda create -n nlp python=3.11 -y
conda activate nlp

# 验证
python --version  # Python 3.11.x
```

### UV (现代替代方案)

```bash
# UV 是极速的 Python 包管理器
# 安装 UV
pip install uv

# 创建虚拟环境
uv venv nlp-env --python 3.11

# 激活
source nlp-env/bin/activate    # Linux/Mac
nlp-env\Scripts\activate       # Windows

# UV 安装包 (比 pip 快 10-100倍)
uv pip install torch transformers
```

## 2. 核心库安装

### 基础科学计算

```bash
# 数据处理基础
pip install numpy pandas scipy

# 可视化
pip install matplotlib seaborn plotly

# 机器学习
pip install scikit-learn xgboost lightgbm

# Jupyter
pip install jupyterlab ipywidgets
```

### 深度学习框架

```bash
# === PyTorch (NLP 首选) ===
# CPU 版本
pip install torch torchvision torchaudio

# GPU 版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU 版本 (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### NLP 专用库

```bash
# === 传统 NLP 工具 ===

# NLTK: 经典教学用库
pip install nltk

# spaCy: 工业级NLP库
pip install spacy

# Jieba: 中文分词
pip install jieba

# HanLP: 中文NLP综合库
pip install hanlp

# stanza: Stanford NLP 的 Python 接口
pip install stanza

# === 深度学习 NLP ===

# HuggingFace 全家桶 (最重要!)
pip install transformers    # 预训练模型
pip install datasets        # 数据集
pip install tokenizers      # 分词器
pip install accelerate      # 分布式训练
pip install peft            # 参数高效微调
pip install trl             # RLHF训练
pip install evaluate        # 评估指标

# 一键安装 HuggingFace 全家桶
pip install transformers[torch] datasets tokenizers accelerate
```

### 下载语言模型

```python
# === spaCy 模型 ===
# 命令行执行
# python -m spacy download en_core_web_sm    # 英文小型
# python -m spacy download en_core_web_md    # 英文中型
# python -m spacy download en_core_web_lg    # 英文大型
# python -m spacy download zh_core_web_sm    # 中文小型

# === NLTK 数据 ===
import nltk
nltk.download('all', quiet=True)  # 下载所有数据

# === stanza 模型 ===
import stanza
stanza.download('en')  # 英文
stanza.download('zh')  # 中文
```

## 3. 验证安装

```python
# install_check.py — 一键验证所有安装

import sys
print(f"Python: {sys.version}")

# === 基础库 ===
import numpy as np
import pandas as pd
import matplotlib
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")

# === ML ===
import sklearn
print(f"scikit-learn: {sklearn.__version__}")

# === 深度学习 ===
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# === NLP 工具 ===
import nltk
print(f"NLTK: {nltk.__version__}")

import spacy
print(f"spaCy: {spacy.__version__}")

import jieba
print(f"Jieba: {jieba.__version__}")

import transformers
print(f"Transformers: {transformers.__version__}")

import datasets
print(f"Datasets: {datasets.__version__}")

print("\n✅ 所有核心库安装成功!" if torch.cuda.is_available() 
      else "\n⚠️  CUDA 不可用，将使用 CPU 模式")
```

## 4. IDE 配置

### VS Code 推荐扩展

```
必装扩展:
├── Python (Microsoft)           # Python 基础支持
├── Pylance (Microsoft)          # 类型检查/智能提示
├── Jupyter (Microsoft)          # Notebook 支持
├── Jupyter Keymap               # 快捷键
├── Python Indent                # 自动缩进
├── autoDocstring                # 自动文档字符串
└── GitLens                      # Git 增强

NLP 辅助:
├── Chinese Word Segmentation    # 中文分词高亮
└── Markdown All in One          # 文档编写
```

### PyCharm 配置

```
PyCharm NLP 开发配置:
1. Settings → Project → Interpreter → 选择 conda nlp 环境
2. Settings → Editor → File Types → 添加 .ipynb 支持
3. 安装插件:
   - Python Jupyter Notebook
   - Markdown Navigator
```

## 5. 项目结构模板

```
nlp_project/
├── README.md
├── requirements.txt          # pip freeze > requirements.txt
├── environment.yml           # conda env export > environment.yml
├── setup.py                  # 包安装配置
├── .gitignore
│
├── data/
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后数据
│   └── external/             # 外部数据
│
├── src/
│   ├── __init__.py
│   ├── data/                 # 数据处理
│   │   ├── dataset.py        # 数据集类
│   │   └── preprocess.py     # 预处理
│   ├── models/               # 模型定义
│   │   ├── classifier.py
│   │   └── ner_model.py
│   ├── training/             # 训练逻辑
│   │   ├── trainer.py
│   │   └── metrics.py
│   └── utils/                # 工具
│       ├── config.py
│       └── logger.py
│
├── notebooks/                # Jupyter 探索
│   ├── 01_eda.ipynb          # 数据探索
│   └── 02_baseline.ipynb     # 基线模型
│
├── scripts/                  # 脚本
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── configs/                  # 配置文件
│   └── default.yaml
│
├── tests/                    # 测试
└── logs/                     # 日志
```

### requirements.txt 模板

```txt
# requirements.txt — NLP 项目依赖

# === 基础 ===
numpy>=1.24
pandas>=2.0
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
tqdm>=4.65

# === 机器学习 ===
scikit-learn>=1.3
xgboost>=2.0

# === 深度学习 ===
torch>=2.0
torchvision>=0.15
torchaudio>=2.0

# === NLP 工具 ===
nltk>=3.8
spacy>=3.6
jieba>=0.42
hanlp>=2.2

# === HuggingFace ===
transformers>=4.35
datasets>=2.14
tokenizers>=0.14
accelerate>=0.24
peft>=0.7
evaluate>=0.4

# === 应用层 ===
langchain>=0.1
sentence-transformers>=2.2
faiss-cpu>=1.7

# === 开发工具 ===
jupyterlab>=4.0
ipywidgets>=8.0
pytest>=7.4
```

### .gitignore 模板

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
dist/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# 数据
data/raw/
data/external/
*.csv
*.tsv
*.json.gz

# 模型
*.bin
*.pt
*.pth
*.ckpt
checkpoints/
wandb/

# 环境
.env
*.env
```

## 6. GPU 配置

### 检查 GPU 环境

```python
import torch

def check_gpu():
    """全面检查 GPU 环境"""
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 编译版本: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"\n✅ GPU 可用!")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  总显存: {props.total_mem / 1024**3:.1f} GB")
            print(f"  多处理器: {props.multi_processor_count}")
            print(f"  计算能力: {props.major}.{props.minor}")
        
        # 当前 GPU
        print(f"\n当前设备: cuda:{torch.cuda.current_device()}")
    else:
        print("\n❌ GPU 不可用，使用 CPU 模式")
        print("提示: 安装 CUDA 版 PyTorch 以使用 GPU")
    
    # MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"\n🍎 Apple MPS 可用!")

check_gpu()
```

### GPU 内存管理

```python
# 常见 GPU 操作
import torch

# 指定 GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 张量移到 GPU
tensor = torch.randn(100, 100).to(device)

# 模型移到 GPU
model = MyModel().to(device)

# 清空缓存
torch.cuda.empty_cache()

# 查看 GPU 内存使用
print(f"已分配: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
print(f"已缓存: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

# 混合精度训练 (省显存、加速)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 7. 常用工具配置

### TensorBoard

```python
# 训练可视化
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs/experiment_1')

# 记录 loss
for epoch in range(epochs):
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)

# 记录词嵌入可视化
word_embeddings = model.embedding.weight.data
writer.add_embedding(word_embeddings, metadata=vocab, 
                     tag='word_embeddings')

writer.close()

# 启动 TensorBoard
# tensorboard --logdir logs/
```

### Weights & Biases (W&B)

```python
# pip install wandb
import wandb

# 初始化
wandb.init(
    project="nlp-text-classification",
    config={
        "learning_rate": 2e-5,
        "architecture": "BERT",
        "dataset": "IMDB",
        "epochs": 5,
    }
)

# 训练循环
for epoch in range(epochs):
    # ... 训练代码 ...
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "accuracy": acc,
    })

wandb.finish()
```

## 小结

| 工具 | 用途 | 安装命令 |
|------|------|----------|
| **Conda** | 环境管理 | `conda create -n nlp python=3.11` |
| **PyTorch** | 深度学习 | `pip install torch` |
| **Transformers** | 预训练模型 | `pip install transformers` |
| **spaCy** | 工业级NLP | `pip install spacy` |
| **Jieba** | 中文分词 | `pip install jieba` |
| **NLTK** | 教学/经典 | `pip install nltk` |

---

| [← 回到目录](../README.md) | [上一篇：NLP发展史](02-NLP发展史.md) | [下一篇：第一个NLP程序](04-第一个NLP程序.md) |
|---|---|---|
