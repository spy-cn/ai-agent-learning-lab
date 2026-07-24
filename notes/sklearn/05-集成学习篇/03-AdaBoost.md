# 03 - AdaBoost

## 核心思想

> 串行训练一系列**弱学习器**（通常是浅决策树），每个新模型重点关注**前一个模型分错的样本**。

```
第1轮：  所有样本等权重
        ●  ●  ●  ★  ●        ← ★ 被分错了
                          │
第2轮：  提高 ★ 的权重        ← 重点关注难样本
        ●  ●  ●  ★★ ●        ← 这次 ★ 分对了，但别的可能错
                          │
第3轮：  继续调整权重
        ●  ●  ★  ●  ★        ← 又关注新的错误
                          │
最终：   加权组合所有弱学习器
```

---

## 算法步骤

### 训练过程

```
1. 初始化：所有样本权重相等 w_i = 1/N

2. For t = 1 to T（训练T个弱学习器）:
   a. 用当前权重训练弱学习器 h_t
   b. 计算加权错误率 ε_t
   c. 计算学习器权重 α_t = 0.5 × log((1-ε)/ε)
      → 错误率越低，权重越大（话语权越大）
   d. 更新样本权重：
      - 分错的样本 → 增加权重（下次更关注）
      - 分对的样本 → 减少权重

3. 最终模型 = 所有弱学习器的加权投票
   H(x) = sign(Σ α_t × h_t(x))
```

### 直觉

```
第1棵树：  分错了 20 个样本         权重 α=0.8
第2棵树：  重点关注那20个，分错10个   权重 α=0.6
第3棵树：  重点关注那10个，分错5个    权重 α=0.9
...
最终：     综合所有树的加权投票
```

---

## sklearn 实现

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42)

# AdaBoost 默认用深度为1的决策树（树桩）作为弱学习器
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # 弱学习器
    n_estimators=50,             # 弱学习器数量
    learning_rate=1.0,           # 学习率
    random_state=42
)
ada.fit(X_train, y_train)

print(f"训练集准确率: {ada.score(X_train, y_train):.4f}")
print(f"测试集准确率: {ada.score(X_test, y_test):.4f}")
```

### AdaBoostRegressor

```python
from sklearn.ensemble import AdaBoostRegressor

ada_reg = AdaBoostRegressor(
    n_estimators=50,
    learning_rate=0.5,
    loss='linear',        # 损失函数: linear/square/exponential
    random_state=42
)
ada_reg.fit(X_train, y_train)
```

---

## 关键超参数

### n_estimators（弱学习器数量）

```python
# 更多弱学习器 → 更强的模型，但可能过拟合
import matplotlib.pyplot as plt

scores = []
for n in range(1, 200, 5):
    ada = AdaBoostClassifier(n_estimators=n, random_state=42)
    ada.fit(X_train, y_train)
    scores.append(ada.score(X_test, y_test))

plt.plot(range(1, 200, 5), scores)
plt.xlabel('n_estimators')
plt.ylabel('准确率')
plt.title('AdaBoost: 弱学习器数量 vs 准确率')
plt.grid(True)
plt.show()
```

### learning_rate（学习率）

```python
# 学习率控制每个弱学习器的贡献
AdaBoostClassifier(learning_rate=0.1)   # 慢学习，更保守
AdaBoostClassifier(learning_rate=1.0)   # 默认
AdaBoostClassifier(learning_rate=2.0)   # 激进，容易过拟合
```

> 💡 **n_estimators 和 learning_rate 的权衡**：
> - 学习率小 + 弱学习器多 → 慢但可能更好（类似深度学习）
> - 学习率大 + 弱学习器少 → 快但可能过拟合
> - 经验：`learning_rate=0.1, n_estimators=200~500` 常比默认好

### estimator（基学习器）

```python
# 默认：深度1的决策树（树桩）
AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1))

# 可以用更深的树（但要注意不要过强）
AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3))
```

> ⚠️ AdaBoost 的基学习器应该是**弱学习器**（比随机猜好一点就行）。太强的基学习器反而不好。

---

## AdaBoost 的特点

### 与随机森林对比

| 特性 | 随机森林 | AdaBoost |
|------|---------|---------|
| 训练方式 | 并行 | 串行 |
| 基学习器 | 深决策树 | 浅决策树（树桩） |
| 样本权重 | 等权重 | 动态调整（关注难样本） |
| 组合方式 | 等权投票 | 加权投票 |
| 主要降低 | 方差 | 偏差 |
| 过拟合 | 不容易 | 可能（弱学习器太多时） |

### 训练误差分析

```python
# 记录每轮的训练误差
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)

# estimator_errors_ 记录了每轮的错误率
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(ada.estimator_errors_)+1),
         ada.estimator_errors_, 'b.-')
plt.xlabel('迭代轮次')
plt.ylabel('加权错误率')
plt.title('AdaBoost 每轮的错误率')
plt.grid(True)
plt.show()
```

---

## 优缺点

### 优点
- ✅ 简单，参数少
- ✅ 不需要特征缩放
- ✅ 能自动聚焦于难样本
- ✅ 有特征重要性

### 缺点
- ❌ **对异常值和噪声非常敏感**（异常值会被反复提高权重）
- ❌ 串行训练，不能并行
- ❌ 弱学习器选择有限
- ❌ 通常不如 GBDT 效果好

> 💡 在大多数场景下，**GBDT / HistGradientBoosting 比 AdaBoost 效果更好**。AdaBoost 主要用于学习 Boosting 的基本概念。

---

## 实战对比

```python
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.model_selection import cross_val_score

models = {
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'GBDT':     GradientBoostingClassifier(n_estimators=100, random_state=42),
}

print(f"{'模型':<12} {'准确率':<12}")
print("-" * 30)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name:<12} {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 小结

| 要点 | 内容 |
|------|------|
| 核心思想 | 串行训练，聚焦难样本 |
| 基学习器 | 浅决策树（树桩） |
| 关键参数 | n_estimators, learning_rate |
| 样本权重 | 动态调整（分错的加权） |
| 优点 | 简单、能聚焦难样本 |
| 缺点 | 对噪声敏感、通常不如GBDT |

---

## 下一节

➡️ [04-梯度提升树GBDT.md](./04-梯度提升树GBDT.md)
