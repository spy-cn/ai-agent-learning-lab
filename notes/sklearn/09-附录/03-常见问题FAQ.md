# 常见问题 FAQ

## Q1: ValueError: Found array with dim 3. Estimator expected <= 2.

**原因**：sklearn 只接受二维数组 `(n_samples, n_features)`。

```python
# ❌ 错误：图像数据是3维 (samples, height, width)
X.shape  # (100, 28, 28)

# ✅ 解决：展平为2维
X = X.reshape(100, -1)  # (100, 784)
```

---

## Q2: 模型在训练集上准确率100%？

**过拟合了。** 解决方法：

```python
# 决策树/随机森林：限制深度
DecisionTreeClassifier(max_depth=5)

# 增加正则化
LogisticRegression(C=0.1)

# 交叉验证验证
cross_val_score(model, X, y, cv=5)
```

---

## Q3: SVC 好慢？

SVC 的时间复杂度是 O(n²)~O(n³)，样本超过1万就很慢。

```python
# 替代方案
from sklearn.svm import LinearSVC  # 线性核，快得多
from sklearn.ensemble import HistGradientBoostingClassifier  # 树模型
```

---

## Q4: OneHotEncoder 报 "unknown category"？

```python
# 测试集出现了训练集没有的新类别
# 解决：设 handle_unknown
OneHotEncoder(handle_unknown='ignore')
```

---

## Q5: GridSearchCV 怎么调 Pipeline 里的参数？

```python
# 参数名格式：步骤名__参数名
param_grid = {
    'model__C': [0.1, 1, 10],
    'preprocessor__num__scaler__with_mean': [True, False]
}
```

---

## Q6: predict 和 predict_proba 的区别？

```python
model.predict(X)          # 返回类别标签
model.predict_proba(X)    # 返回概率（每类一列）

# predict 内部就是 predict_proba 后取最大概率的类别
```

---

## Q7: cross_val_score 的 scoring 怎么用？

```python
cross_val_score(model, X, y, scoring='accuracy')      # 默认
cross_val_score(model, X, y, scoring='f1')            # F1
cross_val_score(model, X, y, scoring='roc_auc')       # AUC
cross_val_score(model, X, y, scoring='neg_mean_squared_error')  # 回归

# 注意回归指标前有 neg_（因为 sklearn 统一"越大越好"）
```

---

## Q8: 怎么处理类别不平衡？

```python
# 方法1：class_weight
LogisticRegression(class_weight='balanced')

# 方法2：SMOTE
from imblearn.over_sampling import SMOTE

# 方法3：用 F1/AUC 而非 accuracy 评估
```
