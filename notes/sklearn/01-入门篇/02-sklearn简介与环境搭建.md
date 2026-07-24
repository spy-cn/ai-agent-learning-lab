# 02 - sklearn 简介与环境搭建

## scikit-learn 是什么？

scikit-learn（简称 sklearn）是一个基于 Python 的**开源机器学习库**，构建在 NumPy、SciPy、Matplotlib 之上。

### 核心特点

| 特点 | 说明 |
|------|------|
| **统一的 API** | 所有模型都遵循 `fit() → predict()` 模式，换算法只改一行代码 |
| **丰富的算法** | 分类、回归、聚类、降维、预处理等几十种算法 |
| **优质文档** | 官方文档堪称教科书级别，每个算法都有示例 |
| **与生态融合** | 完美配合 pandas、numpy、matplotlib |
| **BSD 开源** | 商用友好，无限制 |

### 架构层级

```
            ┌─────────────────────────┐
            │      scikit-learn        │
            │  (算法、模型选择、预处理)  │
            └────────────┬────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │  NumPy  │    │  SciPy  │    │Matplotlib│
    │(数组计算)│    │(科学计算)│    │ (绘图)   │
    └─────────┘    └─────────┘    └─────────┘
```

---

## 安装指南

### 方式一：pip 安装（推荐新手）

```bash
# 创建虚拟环境（可选但推荐）
python -m venv ml-env

# 激活环境
# Windows:
ml-env\Scripts\activate
# macOS/Linux:
source ml-env/bin/activate

# 安装 sklearn 及数据科学全家桶
pip install scikit-learn numpy pandas matplotlib seaborn jupyter
```

### 方式二：conda 安装（推荐数据科学）

```bash
# 使用 conda 创建独立环境
conda create -n ml-env python=3.10
conda activate ml-env

# 安装
conda install scikit-learn pandas numpy matplotlib seaborn jupyter
```

### 验证安装

```python
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

import numpy as np
print(f"numpy version: {np.__version__}")

import pandas as pd
print(f"pandas version: {pd.__version__}")

import matplotlib
print(f"matplotlib version: {matplotlib.__version__}")
```

预期输出：
```
scikit-learn version: 1.3+        # >= 1.0 即可
numpy version: 1.24+
pandas version: 2.0+
matplotlib version: 3.7+
```

---

## 核心模块导览

sklearn 的模块组织非常清晰，按功能分类：

```python
sklearn.datasets       # 内置数据集
sklearn.preprocessing  # 数据预处理（标准化、编码等）
sklearn.impute         # 缺失值填充
sklearn.feature_extraction    # 特征提取（文本、图像）
sklearn.feature_selection     # 特征选择
sklearn.decomposition  # 降维（PCA、NMF）
sklearn.linear_model   # 线性模型（回归、逻辑回归、Ridge、Lasso）
sklearn.neighbors      # K近邻
sklearn.tree           # 决策树
sklearn.svm            # 支持向量机
sklearn.ensemble       # 集成学习（随机森林、GBDT）
sklearn.naive_bayes    # 朴素贝叶斯
sklearn.cluster        # 聚类（KMeans、DBSCAN）
sklearn.model_selection  # 模型选择（交叉验证、超参搜索）
sklearn.metrics        # 评估指标（准确率、F1、MSE等）
sklearn.pipeline       # 流水线
sklearn.manifold       # 流形学习（t-SNE等）
```

---

## sklearn 的统一 API 设计

这是 sklearn 最优雅的设计——**所有估计器（estimator）遵循同一套接口**：

### 核心 API 三件套

```python
from sklearn import SomeEstimator

model = SomeEstimator(...)  # 1. 实例化，传入超参数
model.fit(X, y)             # 2. 训练：从数据中学习
model.predict(X_new)        # 3. 预测：对新数据做推断
```

| 方法 | 用途 | 适用对象 |
|------|------|----------|
| `fit(X, y)` | 训练模型 | 所有监督/无监督模型 |
| `predict(X)` | 预测标签/值 | 监督模型、聚类 |
| `transform(X)` | 转换数据 | 预处理器、降维器 |
| `fit_transform(X)` | 训练 + 转换（等价于先fit后transform） | 预处理器 |
| `score(X, y)` | 评估（返回准确率/R²） | 所有监督模型 |

### 示例：换算法只改一行

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 三个完全不同的算法，用法一模一样！
models = [
    LogisticRegression(),
    RandomForestClassifier(),
    SVC()
]

for model in models:
    model.fit(X_train, y_train)          # 训练
    accuracy = model.score(X_test, y_test)  # 评估
    print(f"{model.__class__.__name__}: {accuracy:.3f}")
```

---

## 常用约定

sklearn 社区有一些约定俗成的规则：

### 命名约定
- **特征矩阵**用大写 `X`（因为它是矩阵/二维）
- **目标向量**用小写 `y`（因为它是向量/一维）
- **预测值**用 `y_pred`

### 数据格式要求
```python
# X 必须是二维数组：(样本数, 特征数)
X = np.array([
    [5.1, 3.5, 1.4, 0.2],   # 样本1的4个特征
    [4.9, 3.0, 1.4, 0.2],   # 样本2
    [4.7, 3.2, 1.3, 0.2],   # 样本3
])

# y 必须是一维数组：(样本数,)
y = np.array([0, 0, 1])  # 对应3个样本的标签
```

> ⚠️ 常见错误：传入一维数组给需要二维的接口。用 `X.reshape(-1, 1)` 或 `X[:, np.newaxis]` 升维。

---

## 版本说明

```python
# 查看当前版本
import sklearn
print(sklearn.__version__)

# 不同版本的主要变化
# v1.0 (2021): 新增 FeatureHasher、Histogram boosting
# v1.1 (2022): QuantileRegressor、改进的 API
# v1.2 (2023): FreshHoldStrategy、pd.DataFrame 支持
# v1.3 (2023): TunedThresholdClassifier、元数据路由
```

> 💡 建议：**使用最新稳定版**。本教程基于 1.3+ 编写，大部分代码在 1.0+ 均可运行。

---

## 开发环境推荐

### Jupyter Notebook / JupyterLab（强烈推荐）

交互式编写，逐段运行代码，适合学习和实验：

```bash
pip install jupyterlab
jupyter lab
```

### VS Code

安装 Python 扩展和 Jupyter 扩展，可直接在编辑器中运行 `.ipynb` 文件。

### Google Colab（免安装）

浏览器中直接运行，预装 sklearn，适合快速实验。

---

## 小结

| 要点 | 内容 |
|------|------|
| 安装 | `pip install scikit-learn` |
| 核心 API | `fit()` → `predict()` → `score()` |
| 特征矩阵 | 二维数组 `X`，形状 `(n_samples, n_features)` |
| 目标向量 | 一维数组 `y`，形状 `(n_samples,)` |
| 统一接口 | 换算法只需改模型类名 |

---

## 下一节

➡️ [03-数据加载与探索](./03-数据加载与探索.md)
