# 04 - 梯度提升树（GBDT）

## 核心思想

> GBDT 是 Boosting 家族中最强大的算法。每一棵新树去拟合**之前所有树的残差**（预测值与真实值的差距）。

### 残差拟合的直觉

```
真实值：        y = [10, 20, 30]

第1棵树预测：   ŷ₁ = [8, 18, 28]
残差：         y - ŷ₁ = [2, 2, 2]       ← 第2棵树的目标！

第2棵树拟合残差： [2, 2, 2] → 预测 [1.5, 1.5, 1.5]
新残差：       [0.5, 0.5, 0.5]           ← 第3棵树的目标

第3棵树拟合残差： → 预测 [0.5, 0.5, 0.5]

最终预测 = ŷ₁ + ŷ₂ + ŷ₃
         = [8+1.5+0.5, 18+1.5+0.5, 28+1.5+0.5]
         = [10, 20, 30]  ✓ 逐渐逼近真实值
```

---

## 与梯度下降的联系

GBDT 名字中"梯度"的含义：

```
对于平方损失：
  损失 L = ½(y - ŷ)²
  梯度 ∂L/∂ŷ = -(y - ŷ) = ŷ - y = -残差

→ 拟合残差 = 拟合负梯度
```

所以 GBDT 实际上是在**函数空间中做梯度下降**：

```
参数空间梯度下降（如线性回归）：
  w_new = w_old - lr × ∂L/∂w

函数空间梯度下降（GBDT）：
  F_new(x) = F_old(x) - lr × ∂L/∂F(x)
           = F_old(x) + lr × h_t(x)     ← h_t 拟合负梯度
```

> 💡 这个视角的意义：GBDT 可以使用**任何可微的损失函数**，而 AdaBoost 本质上只能用指数损失。

---

## sklearn 实现

### GradientBoostingClassifier

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42)

gbdt = GradientBoostingClassifier(
    n_estimators=100,      # 树的数量
    learning_rate=0.1,     # 学习率
    max_depth=3,           # 每棵树的深度
    random_state=42
)
gbdt.fit(X_train, y_train)

print(f"训练集: {gbdt.score(X_train, y_train):.4f}")
print(f"测试集: {gbdt.score(X_test, y_test):.4f}")
```

### GradientBoostingRegressor

```python
from sklearn.ensemble import GradientBoostingRegressor

gbdt_reg = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    loss='squared_error',  # 损失函数
    random_state=42
)
gbdt_reg.fit(X_train, y_train)
print(f"R²: {gbdt_reg.score(X_test, y_test):.4f}")
```

---

## 关键超参数

```python
GradientBoostingClassifier(
    n_estimators=100,         # 树的数量（阶段数）
    learning_rate=0.1,        # 学习率（每棵树的贡献）
    max_depth=3,              # 每棵树的深度
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=1.0,            # 每棵树使用的样本比例（<1.0 → 随机GBDT）
    max_features=None,        # 每棵树分裂考虑的特征数
    random_state=42
)
```

### 最关键的三个参数

#### 1. n_estimators + learning_rate（联动调节）

```python
# 这两个参数互相影响，通常一起调：
# 学习率越小，需要更多的树

# 组合1：快速
GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

# 组合2：更精确（推荐）
GradientBoostingClassifier(n_estimators=500, learning_rate=0.01)

# 组合3：极致（比赛用）
GradientBoostingClassifier(n_estimators=1000, learning_rate=0.005)
```

> 💡 经验法则：**学习率越小、树越多，通常效果越好**（但训练更慢，且最终会过拟合）。

#### 2. max_depth（控制每棵树复杂度）

```python
# GBDT 通常用浅树（3-8层）
# 与随机森林不同，GBDT 的树不需要很深
GradientBoostingClassifier(max_depth=3)   # 默认，常用
GradientBoostingClassifier(max_depth=5)   # 更复杂
```

#### 3. subsample（随机 GBDT）

```python
# 每棵树只用一部分样本 → 引入随机性 → 降低方差
GradientBoostingClassifier(subsample=0.8)  # 用80%样本
```

---

## 早停（Early Stopping）

```python
# 训练太多树会过拟合，用验证集自动确定最佳树数
import numpy as np
import matplotlib.pyplot as plt

gbdt = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.1,
    validation_fraction=0.2,    # 留20%做验证
    n_iter_no_change=10,        # 连续10轮没提升就停止
    tol=1e-4,                   # 提升阈值
    random_state=42
)
gbdt.fit(X_train, y_train)

print(f"实际使用的树数: {gbdt.n_estimators_}")

# 查看每轮的验证集得分
train_scores = gbdt.train_score_
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(train_scores)+1), train_scores, 'b.-')
plt.xlabel('迭代轮次')
plt.ylabel('训练集得分')
plt.title(f'GBDT 训练曲线（在 {gbdt.n_estimators_} 轮停止）')
plt.grid(True)
plt.show()
```

---

## 特征重要性

```python
importances = gbdt.feature_importances_

for name, imp in sorted(zip(data.feature_names, importances),
                         key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {name:25s}: {imp:.4f}")
```

---

## HistGradientBoosting（推荐！）

sklearn 0.24+ 引入的**高性能版本**，灵感来自 LightGBM：

```python
from sklearn.ensemble import HistGradientBoostingClassifier

hist_gb = HistGradientBoostingClassifier(
    max_iter=100,               # 等价于 n_estimators
    learning_rate=0.1,
    max_depth=None,             # 叶节点数代替深度
    max_leaf_nodes=31,          # 最大叶节点数
    early_stopping=True,        # 自动早停（默认开启）
    random_state=42
)
hist_gb.fit(X_train, y_train)
print(f"测试集: {hist_gb.score(X_test, y_test):.4f}")
```

### HistGradientBoosting 的优势

| 特性 | GradientBoosting | HistGradientBoosting |
|------|-----------------|---------------------|
| 大数据支持 | ❌ 慢 | ✅ 快10倍+ |
| 缺失值 | ❌ 需预处理 | ✅ 原生支持 |
| 类别特征 | ❌ 需编码 | ✅ 原生支持 |
| 早停 | 需手动 | ✅ 默认开启 |
| 样本权重 | ✅ | ✅ |
| 预测速度 | 中 | 快 |

```python
# 原生支持缺失值！不需要 SimpleImputer
hist_gb.fit(X_with_nan, y)  # 直接训练

# 原生支持类别特征
from sklearn.datasets import fetch_openml
hist_gb = HistGradientBoostingClassifier(
    categorical_features=['city', 'gender']  # 指定类别列
)
```

> 💡 **生产环境推荐 HistGradientBoosting**，效果接近 LightGBM，且是纯 sklearn 实现。

---

## 完整调参示例

```python
from sklearn.model_selection import GridSearchCV

# 分两步调参，避免搜索空间太大

# 第1步：先调树的结构参数
param_grid_1 = {
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [1, 5, 10],
}
grid1 = GridSearchCV(
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    param_grid_1, cv=5
)
grid1.fit(X_train, y_train)
print(f"最佳结构参数: {grid1.best_params_}")

# 第2步：调学习率和树数
best_depth = grid1.best_params_['max_depth']
best_leaf = grid1.best_params_['min_samples_leaf']

param_grid_2 = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
}
grid2 = GridSearchCV(
    GradientBoostingClassifier(max_depth=best_depth,
                               min_samples_leaf=best_leaf,
                               random_state=42),
    param_grid_2, cv=5
)
grid2.fit(X_train, y_train)
print(f"最佳参数: {grid2.best_params_}")
print(f"最佳准确率: {grid2.best_score_:.4f}")
```

---

## GBDT vs 随机森林

| 特性 | 随机森林 | GBDT |
|------|---------|------|
| 训练方式 | 并行 | 串行 |
| 树深度 | 深（完全长） | 浅（3-8层） |
| 主要降低 | 方差 | 偏差 |
| 过拟合 | 不容易 | 可能（树太多） |
| 参数敏感度 | 低（默认就好） | 高（需调参） |
| 准确率 | 好 | 通常更好 |
| 缺失值 | ❌ | ❌（Hist版✅） |

---

## 优缺点

### 优点
- ✅ **准确率极高**（最强经典算法之一）
- ✅ 能处理非线性关系
- ✅ 能输出特征重要性
- ✅ 灵活的损失函数
- ✅ Hist 版本支持缺失值和类别特征

### 缺点
- ❌ 参数调优复杂
- ❌ 串行训练（不能并行）
- ❌ 容易过拟合（需早停）
- ❌ 原生版不支持缺失值
- ❌ 不如 LightGBM/XGBoost 快

---

## 小结

| 要点 | 内容 |
|------|------|
| 核心思想 | 拟合残差（负梯度） |
| 基学习器 | 浅决策树 |
| 关键参数 | n_estimators + learning_rate（联动） |
| max_depth | 浅树（3-8） |
| 早停 | 防止过拟合 |
| 推荐 | 生产用 HistGradientBoosting |

---

## 下一节

➡️ [05-Voting与Stacking.md](./05-Voting与Stacking.md)
