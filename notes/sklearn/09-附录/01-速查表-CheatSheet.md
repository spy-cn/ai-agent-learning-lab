# sklearn 速查表（CheatSheet）

## 核心工作流

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SomeModel()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

---

## 分类算法速查

| 算法 | 导入 | 一句话 |
|------|------|--------|
| 逻辑回归 | `from sklearn.linear_model import LogisticRegression` | 线性分类基线 |
| KNN | `from sklearn.neighbors import KNeighborsClassifier` | 距离投票 |
| 决策树 | `from sklearn.tree import DecisionTreeClassifier` | 可解释 |
| 随机森林 | `from sklearn.ensemble import RandomForestClassifier` | 开箱即用 |
| GBDT | `from sklearn.ensemble import GradientBoostingClassifier` | 最强内置 |
| SVM | `from sklearn.svm import SVC` | 高维小数据 |
| 朴素贝叶斯 | `from sklearn.naive_bayes import GaussianNB` | 文本/快速 |

---

## 回归算法速查

| 算法 | 导入 |
|------|------|
| 线性回归 | `from sklearn.linear_model import LinearRegression` |
| Ridge | `from sklearn.linear_model import Ridge` |
| Lasso | `from sklearn.linear_model import Lasso` |
| 随机森林 | `from sklearn.ensemble import RandomForestRegressor` |
| GBDT | `from sklearn.ensemble import GradientBoostingRegressor` |

---

## 聚类算法速查

| 算法 | 导入 | 需要K? |
|------|------|--------|
| KMeans | `from sklearn.cluster import KMeans` | ✅ |
| DBSCAN | `from sklearn.cluster import DBSCAN` | ❌ |
| 层次聚类 | `from sklearn.cluster import AgglomerativeClustering` | ✅ |

---

## 预处理速查

```python
# 缩放
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 编码
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

# 缺失值
from sklearn.impute import SimpleImputer

# 特征选择
from sklearn.feature_selection import SelectKBest, SelectFromModel

# 降维
from sklearn.decomposition import PCA

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
```

---

## 模型选择速查

```python
# 交叉验证
from sklearn.model_selection import cross_val_score

# 调参
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# 评估
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)
```

---

## 模型保存

```python
import joblib
joblib.dump(model, 'model.joblib')
model = joblib.load('model.joblib')
```

---

## 常用参数默认值

```python
RandomForestClassifier(n_estimators=100, max_depth=None)
GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
SVC(C=1.0, kernel='rbf', gamma='scale')
KNeighborsClassifier(n_neighbors=5)
LogisticRegression(C=1.0, penalty='l2')
```
