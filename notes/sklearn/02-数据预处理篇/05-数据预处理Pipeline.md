# 05 - 数据预处理 Pipeline

## 为什么需要 Pipeline？

### 痛点：重复而凌乱的预处理代码

没有 Pipeline 时，你的代码可能长这样：

```python
# ❌ 痛苦的写法
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test  = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

encoder = OneHotEncoder(...)
X_train = encoder.fit_transform(X_train)
X_test  = encoder.transform(X_test)

model = SomeModel()
model.fit(X_train, y_train)
model.score(X_test, y_test)

# 问题：
# 1. 代码重复、容易漏步骤
# 2. 误用 fit_transform(X_test) 导致数据泄露
# 3. 没法配合 GridSearchCV 调参
# 4. 部署时需要重复所有预处理步骤
```

### 解决方案：Pipeline

```python
# ✅ 优雅的写法
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   SomeModel())
])

pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
# 一行搞定！
```

---

## Pipeline 基础

### 创建和使用

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 创建 Pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # 步骤1：填充缺失
    ('scaler',  StandardScaler()),                   # 步骤2：标准化
    ('clf',     LogisticRegression())                # 步骤3：分类器
])

# 使用——和普通模型完全一样
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
score  = pipe.score(X_test, y_test)
```

### Pipeline 的工作原理

```
原始数据 X
    │
    ▼
┌──────────┐     ┌──────────┐     ┌──────────┐
│ imputer  │────►│  scaler  │────►│   clf    │
│ fit +    │     │ fit +    │     │ fit      │
│ transform│     │ transform│     │          │
└──────────┘     └──────────┘     └──────────┘
                                          │
                                          ▼
                                     预测结果
```

- `fit()` 时：每个步骤依次 `fit_transform`，最后一步只 `fit`
- `predict()` 时：每个步骤依次 `transform`，最后一步 `predict`

### 简写：make_pipeline

不需要手写步骤名时：

```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    LogisticRegression()
)
# 步骤名自动生成为类名的小写：'simpleimputer', 'standardscaler', 'logisticregression'
```

### 访问 Pipeline 中的步骤

```python
# 通过名称访问
scaler = pipe.named_steps['scaler']
print(scaler.mean_)  # 查看标准化参数

# 通过索引访问
first_step = pipe[0]
last_step  = pipe[-1]
```

---

## ColumnTransformer：处理混合类型列

真实数据通常既有数值列又有文本列，需要**不同**的预处理。

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 示例数据
df = pd.DataFrame({
    '年龄':   [25, 30, np.nan, 45],
    '收入':   [5000, 8000, 6000, np.nan],
    '城市':   ['北京', '上海', '广州', '北京'],
    '学历':   ['本科', '硕士', '高中', '博士'],
})
y = [0, 1, 0, 1]

# 区分列类型
numeric_features = ['年龄', '收入']
categorical_features = ['城市', '学历']

# 数值列的预处理流水线
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 类别列的预处理流水线
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 用 ColumnTransformer 组合
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 完整 Pipeline：预处理 + 模型
full_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# 使用
full_pipe.fit(df, y)
predictions = full_pipe.predict(df)
```

### 架构图

```
                原始数据
            ┌──────┴──────┐
            │             │
      数值列 Pipeline   类别列 Pipeline
      ┌─────┴─────┐   ┌─────┴─────┐
      │ imputer   │   │ imputer   │
      │ scaler    │   │ onehot    │
      └─────┬─────┘   └─────┬─────┘
            │               │
            └───────┬───────┘
                    │
              拼接后的特征
                    │
              ┌─────┴─────┐
              │  分类器   │
              └───────────┘
```

---

## Pipeline + GridSearchCV（最强大的组合）

Pipeline 的最大优势：**可以网格搜索预处理参数 + 模型参数**。

```python
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

# 参数名格式：步骤名__参数名
param_grid = {
    'scaler__with_mean': [True, False],          # 预处理参数
    'clf__C': [0.01, 0.1, 1, 10, 100],           # 模型参数
    'clf__penalty': ['l1', 'l2'],                 # 模型参数
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"最佳参数: {grid.best_params_}")
print(f"最佳准确率: {grid.best_score_:.4f}")
```

### 搜索不同的模型

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC())  # 占位
])

# 用一个 Pipeline 搜索多种模型！
param_grid = [
    {
        'clf': [SVC()],
        'clf__C': [0.1, 1, 10],
        'clf__kernel': ['rbf', 'linear']
    },
    {
        'clf': [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [None, 10]
    },
]

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"最佳模型: {grid.best_estimator_.named_steps['clf']}")
```

---

## FeatureUnion：并行特征提取

当你需要从同一份数据中**同时提取多种特征**：

```python
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# 并行：PCA 降维 + SelectKBest 选择，结果拼接
combined_features = FeatureUnion([
    ('pca', PCA(n_components=3)),
    ('best', SelectKBest(k=5))
])

pipe = Pipeline([
    ('features', combined_features),  # 输出 3+5=8 个特征
    ('clf', LogisticRegression())
])
```

---

## 缓存中间结果（warm_start / memory）

当 Pipeline 中某些步骤计算量很大（如特征提取），可以用缓存：

```python
from tempfile import mkdtemp
from shutil import rmtree

# 创建临时缓存目录
cachedir = mkdtemp()

pipe = Pipeline([
    ('pca', PCA()),
    ('clf', SVC())
], memory=cachedir)

# ... 使用 ...

# 清理缓存
rmtree(cachedir)
```

---

## 完整实战示例

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# ===== 1. 加载泰坦尼克数据 =====
X, y = fetch_openml('titanic', version=1, as_frame=True, return_X_y=True)
X = X[['pclass', 'sex', 'age', 'fare', 'embarked']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ===== 2. 定义预处理 =====
numeric_features = ['age', 'fare']
categorical_features = ['pclass', 'sex', 'embarked']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])

# ===== 3. 完整 Pipeline =====
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# ===== 4. 训练评估 =====
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

# ===== 5. 网格搜索调参 =====
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5],
}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"最佳参数: {grid.best_params_}")
print(f"最佳分数: {grid.best_score_:.4f}")
```

---

## 小结

| 工具 | 用途 |
|------|------|
| `Pipeline` | 串联多个步骤（预处理→模型） |
| `make_pipeline` | 自动命名步骤的快捷写法 |
| `ColumnTransformer` | 对不同列应用不同预处理 |
| `FeatureUnion` | 并行提取多种特征后拼接 |
| `GridSearchCV` + Pipeline | 同时调预处理参数和模型参数 |

**核心原则**：
- **永远用 Pipeline**，从机制上防止数据泄露
- `fit` 只在训练集上调用，测试集永远只 `transform`
- Pipeline 让代码更简洁、可复现、可部署

---

## 数据预处理篇总结

恭喜！你已完成数据预处理篇的学习：

| 章节 | 核心工具 |
|------|----------|
| 数据清洗 | `SimpleImputer` |
| 特征缩放 | `StandardScaler` / `MinMaxScaler` |
| 类别编码 | `OneHotEncoder` / `OrdinalEncoder` |
| 特征选择 | `SelectKBest` / `SelectFromModel` |
| Pipeline | `Pipeline` / `ColumnTransformer` |

下一步，进入算法的核心——监督学习：

➡️ [../03-监督学习篇/01-线性回归.md](../03-监督学习篇/01-线性回归.md)
