# 05 - 端到端NLP项目模板

一个规范的NLP项目不仅需要好的模型代码，更需要合理的工程结构、配置管理和实验追踪体系。本章提供一个生产级的NLP项目模板，涵盖从开发到部署的全生命周期。

---

## 1. 项目结构总览

```
nlp-project/
├── configs/                    # 配置文件目录
│   ├── config.yaml            # 主配置
│   ├── model/                 # 模型配置
│   │   ├── bert_base.yaml
│   │   └── roberta_large.yaml
│   ├── data/                  # 数据配置
│   │   └── dataset.yaml
│   └── training/              # 训练配置
│       └── default.yaml
├── src/                       # 源代码
│   ├── __init__.py
│   ├── data/                  # 数据处理模块
│   │   ├── __init__.py
│   │   ├── dataset.py         # Dataset定义
│   │   ├── preprocessing.py   # 数据预处理
│   │   └── augment.py         # 数据增强
│   ├── models/                # 模型定义
│   │   ├── __init__.py
│   │   ├── base_model.py      # 基类
│   │   ├── classifier.py      # 分类模型
│   │   └── components/        # 模型组件
│   │       └── attention.py
│   ├── training/              # 训练逻辑
│   │   ├── __init__.py
│   │   ├── trainer.py         # 训练器
│   │   ├── optimizer.py       # 优化器配置
│   │   └── scheduler.py       # 学习率调度
│   ├── evaluation/            # 评估
│   │   ├── __init__.py
│   │   ├── metrics.py         # 评估指标
│   │   └── evaluator.py       # 评估器
│   ├── inference/             # 推理
│   │   ├── __init__.py
│   │   ├── predictor.py       # 预测器
│   │   └── api.py             # FastAPI服务
│   └── utils/                 # 工具函数
│       ├── __init__.py
│       ├── logger.py          # 日志配置
│       ├── registry.py        # 注册机制
│       └── helpers.py         # 辅助函数
├── tests/                     # 单元测试
│   ├── test_data.py
│   ├── test_model.py
│   └── test_training.py
├── scripts/                   # 运行脚本
│   ├── train.py               # 训练入口
│   ├── evaluate.py            # 评估入口
│   └── deploy.sh              # 部署脚本
├── notebooks/                 # Jupyter Notebooks
│   └── exploration.ipynb
├── outputs/                   # 输出目录（模型/日志）
│   ├── checkpoints/
│   ├── logs/
│   └── predictions/
├── requirements.txt           # Python依赖
├── Dockerfile                 # Docker配置
├── docker-compose.yml         # 多服务编排
├── .env.example               # 环境变量示例
├── Makefile                   # 快捷命令
├── .pre-commit-config.yaml   # Pre-commit hooks
├── .github/
│   └── workflows/
│       ├── ci.yaml           # CI流水线
│       └── deploy.yaml       # 部署流水线
└── README.md                  # 项目文档
```

---

## 2. 配置管理

### 2.1 Hydra/OmegaConf配置

```python
# configs/config.yaml
# 使用Hydra进行分层配置管理

defaults:
  - model: bert_base
  - data: dataset
  - training: default
  - _self_

experiment:
  name: nlp_experiment
  seed: 42
  output_dir: ./outputs/${experiment.name}

wandb:
  project: nlp-project
  entity: your_team
  tags: [baseline]
  log_model: true

mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: ${experiment.name}
```

```yaml
# configs/model/bert_base.yaml
# @package _group_
model_name: bert-base-chinese
num_labels: 3
max_length: 128
dropout: 0.1
freeze_backbone: false
```

```yaml
# configs/training/default.yaml
# @package _group_
epochs: 10
batch_size: 32
learning_rate: 2e-5
warmup_steps: 500
weight_decay: 0.01
gradient_accumulation_steps: 1
max_grad_norm: 1.0
fp16: true
early_stopping_patience: 3
```

### 2.2 配置加载代码

```python
# src/utils/config.py
from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import OmegaConf
import hydra

@dataclass
class ModelConfig:
    model_name: str = "bert-base-chinese"
    num_labels: int = 3
    max_length: int = 128
    dropout: float = 0.1
    freeze_backbone: bool = False

@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    fp16: bool = True
    early_stopping_patience: int = 3

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = "nlp_base"
    seed: int = 42
    output_dir: str = "./outputs"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        cfg = OmegaConf.load(path)
        schema = OmegaConf.structured(cls)
        merged = OmegaConf.merge(schema, cfg)
        return OmegaConf.to_object(merged)

# 使用
# cfg = Config.from_yaml("configs/config.yaml")
# print(cfg.model.model_name)  # bert-base-chinese
```

---

## 3. 数据处理模块

```python
# src/data/dataset.py
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch
from typing import Dict, Any

class NLPDataset(Dataset):
    """通用NLP数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load_data(self, path: str) -> list:
        """支持json/jsonl/csv"""
        import json
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                return [json.loads(line) for line in f]
            else:
                return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }

def create_dataloaders(config) -> tuple:
    """创建训练/验证/测试DataLoader"""
    train_dataset = NLPDataset(
        config.data.train_path,
        config.tokenizer,
        config.model.max_length
    )
    val_dataset = NLPDataset(
        config.data.val_path,
        config.tokenizer,
        config.model.max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size * 2,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader
```

---

## 4. 训练逻辑

```python
# src/training/trainer.py
import torch
import wandb
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import numpy as np

class NLPtrainer:
    """通用NLP训练器"""

    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.criterion = torch.nn.CrossEntropyLoss()

        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # 初始化WandB
        wandb.init(
            project=config.wandb.project,
            config=OmegaConf.to_container(self.config, resolve=True)
        )

    def _build_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )

    def _build_scheduler(self):
        from transformers import get_linear_schedule_with_warmup
        total_steps = len(self.train_loader) * self.config.training.epochs
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=total_steps
        )

    def train(self):
        for epoch in range(self.config.training.epochs):
            # Training
            train_loss = self._train_epoch(epoch)
            # Validation
            val_loss, metrics = self._validate(epoch)
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.config.training.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {metrics['accuracy']:.4f}"
            )
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": metrics["accuracy"],
                "learning_rate": self.scheduler.get_last_lr()[0],
            })
            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss, metrics)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            self.scheduler.step()

        wandb.finish()

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for batch in progress:
            inputs = {k: v.to(self.device) for k, v in batch.items()
                      if k != "labels"}
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = self.criterion(outputs.logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def _validate(self, epoch: int):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()
                          if k != "labels"}
                labels = batch["labels"].to(self.device)

                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        return total_loss / len(self.val_loader), {"accuracy": accuracy}

    def _save_checkpoint(self, epoch, val_loss, metrics):
        path = Path(self.config.output_dir) / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "metrics": metrics,
        }, path / "best_model.pt")
        logger.info(f"Saved checkpoint to {path / 'best_model.pt'}")
```

---

## 5. 实验管理

### 5.1 MLflow集成

```python
# src/utils/tracking.py
import mlflow
import mlflow.pytorch
from pathlib import Path

class ExperimentTracker:
    """统一实验追踪：MLflow + WandB"""

    def __init__(self, config):
        self.config = config
        self.use_mlflow = getattr(config, "mlflow", None) is not None
        self.use_wandb = getattr(config, "wandb", None) is not None

    def start(self):
        if self.use_mlflow:
            mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
            mlflow.set_experiment(self.config.mlflow.experiment_name)
            self.mlflow_run = mlflow.start_run()
            mlflow.log_params(OmegaConf.to_container(self.config, resolve=True))

    def log_metrics(self, metrics: dict, step: int = None):
        """批量记录指标"""
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_artifact(self, path: str):
        """记录产物（模型、图表等）"""
        if self.use_mlflow:
            mlflow.log_artifact(path)
        if self.use_wandb:
            wandb.save(path)

    def log_model(self, model, artifact_path: str = "model"):
        """记录模型"""
        if self.use_mlflow:
            mlflow.pytorch.log_model(model, artifact_path)

    def end(self):
        if self.use_mlflow:
            mlflow.end_run()
```

### 5.2 超参搜索

```python
# scripts/hparam_search.py
import optuna
from src.training.trainer import NLPtrainer

def objective(trial):
    """Optuna超参优化目标函数"""
    # 定义搜索空间
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.3)

    # 更新配置
    config.training.learning_rate = lr
    config.training.batch_size = batch_size
    config.model.dropout = dropout
    config.training.warmup_steps = int(warmup_ratio * total_steps)

    # 训练并返回验证指标
    trainer = NLPtrainer(model, config, train_loader, val_loader)
    val_metrics = trainer.validate_only()  # 只验证不训练完整epoch

    return val_metrics["accuracy"]

# 运行超参搜索
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(),
    study_name="nlp_hparam_search"
)
study.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
print(f"Best accuracy: {study.best_value:.4f}")
```

---

## 6. 日志系统

```python
# src/utils/logger.py
from loguru import logger
import sys
from pathlib import Path

def setup_logger(output_dir: str = "./outputs/logs"):
    """配置loguru日志系统"""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 移除默认handler
    logger.remove()

    # 控制台输出（彩色）
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # 所有日志文件
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="00:00",  # 每天轮转
        retention="30 days",
        compression="zip",
        encoding="utf-8",
    )

    # 错误日志单独文件
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        level="ERROR",
        rotation="00:00",
        retention="90 days",
    )

    return logger

# 使用示例
# logger = setup_logger("./outputs/logs")
# logger.info("训练开始")
# logger.debug(f"模型参数: {sum(p.numel() for p in model.parameters())}")
# logger.error("训练中断: CUDA out of memory")
```

---

## 7. 代码质量工具

### 7.1 Pre-commit配置

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.10
        args: [--line-length=100]

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
        additional_dependencies: [types-PyYAML]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: detect-private-key
```

### 7.2 Makefile快捷命令

```makefile
# Makefile
.PHONY: install test lint format train evaluate clean

install:
	pip install -r requirements.txt
	pre-commit install

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=html

train:
	python scripts/train.py --config configs/config.yaml

evaluate:
	python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt

clean:
	rm -rf outputs/checkpoints/*
	rm -rf outputs/logs/*
	rm -rf .pytest_cache
	rm -rf __pycache__ */__pycache__

docker-build:
	docker build -t nlp-project:latest .

docker-run:
	docker run -p 8000:8000 --gpus all nlp-project:latest

all: lint test train
```

---

## 8. CI/CD与部署

### 8.1 GitHub Actions CI

```yaml
# .github/workflows/ci.yaml
name: NLP Project CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Lint
      run: |
        pip install flake8 black isort mypy
        make lint

    - name: Test
      run: |
        pytest tests/ --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
```

### 8.2 Docker部署

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# 下载预训练模型（可选，减少启动时间）
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-chinese')"

EXPOSE 8000

CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### 8.3 项目模板快速生成脚本

```python
# scripts/create_project.py
"""快速生成NLP项目模板"""
import os

PROJECT_STRUCTURE = {
    "configs": ["config.yaml"],
    "configs/model": ["bert_base.yaml", "roberta_large.yaml"],
    "configs/data": ["dataset.yaml"],
    "configs/training": ["default.yaml"],
    "src": ["__init__.py"],
    "src/data": ["__init__.py", "dataset.py", "preprocessing.py", "augment.py"],
    "src/models": ["__init__.py", "base_model.py", "classifier.py"],
    "src/models/components": ["__init__.py", "attention.py"],
    "src/training": ["__init__.py", "trainer.py", "optimizer.py", "scheduler.py"],
    "src/evaluation": ["__init__.py", "metrics.py", "evaluator.py"],
    "src/inference": ["__init__.py", "predictor.py", "api.py"],
    "src/utils": ["__init__.py", "logger.py", "registry.py", "helpers.py"],
    "tests": ["test_data.py", "test_model.py", "test_training.py"],
    "scripts": ["train.py", "evaluate.py", "deploy.sh"],
    "notebooks": ["exploration.ipynb"],
    "outputs/checkpoints": [],
    "outputs/logs": [],
    "outputs/predictions": [],
    ".github/workflows": ["ci.yaml", "deploy.yaml"],
}

def create_project(root_dir: str = "."):
    """创建项目骨架"""
    for dir_path, files in PROJECT_STRUCTURE.items():
        full_dir = os.path.join(root_dir, dir_path)
        os.makedirs(full_dir, exist_ok=True)

        for file in files:
            file_path = os.path.join(full_dir, file)
            if not os.path.exists(file_path):
                if file == "__init__.py":
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write('"""{} module"""\n'.format(dir_path))
                else:
                    Path(file_path).touch()

    print(f"项目模板已创建在: {os.path.abspath(root_dir)}")

if __name__ == "__main__":
    create_project("nlp-project")
```

---

## 小结

本章提供了一个完整的NLP项目工程模板，覆盖从开发到部署的完整流程。

| 要点 | 内容 |
|------|------|
| **项目结构** | 模块化组织：configs/ src/ tests/ scripts/ |
| **配置管理** | Hydra + OmegaConf 分层配置 |
| **训练框架** | 自定义Trainer + Early Stopping + Checkpoint |
| **实验管理** | MLflow (追踪) + WandB (可视化) + Optuna (超参搜索) |
| **日志系统** | loguru (分级/轮转/压缩) |
| **代码质量** | pre-commit + black + flake8 + mypy |
| **CI/CD** | GitHub Actions (Lint/Test/Coverage) |
| **部署** | Docker + FastAPI + docker-compose |

---

| [← 回到目录](../README.md) | [上一篇：04-文档问答系统](04-文档问答系统.md) | [下一篇：01-模型速查表](../13-附录/01-模型速查表.md) |
|---|---|---|
