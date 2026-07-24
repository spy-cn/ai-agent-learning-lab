# 05 - Voting 与 Stacking

## 核心思想

> Bagging 和 Boosting 都是用**同类型**的模型集成。Voting 和 Stacking 则是组合**不同类型**的模型，利用它们的多样性。

```
不同模型擅长不同领域：

模型A（决策树）：  擅长非线性、特征交互
模型B（SVM）：    擅长高维间隔
模型C（逻辑回归）：擅长线性关系、概率

组合后：          综合各自优势 → 更强的整体
```

---

## 一、Voting（投票法）

### 1. Hard Voting（硬投票）

> 少数服从多数。

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)

# 定义基模型
clf1 = LogisticRegression(max_iter=200)
clf2 = DecisionTreeClassifier(random_state=42)
clf3 = SVC(random_state=42)

# 硬投票
voting_clf = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('dt', clf2),
        ('svm', clf3)
    ],
    voting='hard'       # 硬投票：多数表决
)

# 对比单个模型 vs 投票
for name, clf in [('LR', clf1), ('DT', clf2), ('SVM', clf3), ('Voting', voting_clf)]:
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"{name:8s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 2. Soft Voting（软投票）⭐ 推荐

> 对各模型的**概率预测取平均**，选概率最大的类别。

```python
voting_clf = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('dt', clf2),
        ('svm', SVC(probability=True, random_state=42))  # SVM 需开 probability
    ],
    voting='soft'       # 软投票：概率平均
)
```

**Hard vs Soft 对比**：

```
模型A预测：  [0.8, 0.1, 0.1]  → 类别0
模型B预测：  [0.6, 0.3, 0.1]  → 类别0
模型C预测：  [0.1, 0.8, 0.1]  → 类别1

Hard Voting：  2票类别0 vs 1票类别1 → 类别0

Soft Voting：  平均概率 = [(0.8+0.6+0.1)/3, (0.1+0.3+0.8)/3, ...]
             = [0.5, 0.4, 0.1]  → 类别0
```

> 💡 Soft Voting 通常更好，因为它利用了概率信息而不是简单的投票。

### 加权投票

```python
# 给更强的模型更大权重
voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('svm', clf3)],
    voting='soft',
    weights=[1, 1, 2]      # SVM 权重更大
)
```

---

## 二、Stacking（堆叠法）

### 核心思想

> 用多个基模型预测，把预测结果作为**新特征**，再训练一个**元模型**来组合它们。

```
                原始数据
                   │
          ┌────────┼────────┐
          │        │        │
       模型A    模型B    模型C     ← 第0层（基模型）
          │        │        │
          ▼        ▼        ▼
       预测A     预测B    预测C
          │        │        │
          └────────┼────────┘
                   │
              [预测A, 预测B, 预测C]  ← 新特征
                   │
              元模型（如逻辑回归）   ← 第1层
                   │
                 最终预测
```

### sklearn 实现

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# 基模型
base_models = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt',  DecisionTreeClassifier(random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# 元模型
meta_model = LogisticRegression()

# Stacking
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,                   # 用5折交叉验证生成元特征
    n_jobs=-1
)

stacking_clf.fit(X_train, y_train)
print(f"Stacking 准确率: {stacking_clf.score(X_test, y_test):.4f}")
```

### Stacking 的工作原理

```
关键问题：基模型的预测作为元特征，怎么避免数据泄露？

答案：用交叉验证生成元特征

第1折：用 [2,3,4,5] 折训练基模型 → 预测第1折 → 元特征[1]
第2折：用 [1,3,4,5] 折训练基模型 → 预测第2折 → 元特征[2]
...
第5折：用 [1,2,3,4] 折训练基模型 → 预测第5折 → 元特征[5]

拼接 → 完整的元特征矩阵 → 训练元模型
```

### 使用概率作为元特征

```python
# 默认用 predict 的结果（类别标签）作为元特征
# 也可以用概率（更丰富）

stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    stack_method='predict_proba',   # 用概率作为元特征
    cv=5
)
```

| stack_method | 说明 |
|--------------|------|
| `'auto'` | 自动选择（优先 predict_proba） |
| `'predict_proba'` | 用概率（每类一列） |
| `'decision_function'` | 用决策函数值 |
| `'predict'` | 用预测标签 |

---

## Voting vs Stacking 对比

| 特性 | Voting | Stacking |
|------|--------|---------|
| 复杂度 | 简单 | 较复杂 |
| 组合方式 | 平均/投票 | 学习如何组合 |
| 元模型 | 无 | 有 |
| 训练速度 | 快 | 慢（交叉验证） |
| 效果 | 好 | 通常更好 |
| 调参 | 权重 | 元模型也要调 |

---

## 实战：三种集成对比

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import (
    VotingClassifier, StackingClassifier,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 定义4种不同的基模型
base_estimators = [
    ('lr',   LogisticRegression(max_iter=500)),
    ('rf',   RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm',  SVC(probability=True, random_state=42)),
    ('gbdt', GradientBoostingClassifier(random_state=42)),
]

# Voting（软投票）
voting = VotingClassifier(base_estimators, voting='soft')

# Stacking
stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

# 对比
models = dict(base_estimators + [('Voting', voting), ('Stacking', stacking)])

print(f"{'模型':<12} {'准确率':<15}")
print("-" * 35)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name:<12} {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 使用建议

### 什么时候用 Voting？

- 基模型差异大（如 SVM + 树 + 线性）
- 不想增加复杂度
- 快速尝试集成效果

### 什么时候用 Stacking？

- 追求极致准确率
- 有足够的计算资源
- 基模型在不同样本上有优势

### 组合模型的原则

```
✅ 好的组合：
- 决策树 + SVM + 逻辑回归（差异大）
- 随机森林 + GBDT + KNN（不同机制）

❌ 差的组合：
- 5棵相同的决策树（没有多样性）
- 5个完全相关的模型（集成没有意义）
```

---

## 小结

| 方法 | 核心思想 | 适用 |
|------|---------|------|
| Hard Voting | 多数表决 | 简单快速 |
| Soft Voting | 概率平均 | 推荐，利用概率信息 |
| Stacking | 学习如何组合 | 追求最佳效果 |

**关键原则**：组合的模型要有**多样性**——犯不同的错误才能互补。

---

## 集成学习篇总结

恭喜！你已学完集成学习：

| 方法 | 策略 | 代表算法 |
|------|------|---------|
| Bagging | 并行 + 降方差 | 随机森林 |
| Boosting | 串行 + 降偏差 | AdaBoost、GBDT |
| Voting | 投票/概率平均 | 软投票 |
| Stacking | 元模型组合 | StackingClassifier |

> 💡 **实用建议**：如果只选一个算法，选 **HistGradientBoosting**；如果参加比赛，用 **Stacking + GBDT**。

接下来学习如何评估和优化模型：

➡️ [../06-模型优化篇/01-交叉验证.md](../06-模型优化篇/01-交叉验证.md)
