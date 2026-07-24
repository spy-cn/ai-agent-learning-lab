# 04 - PCA 降维

## 为什么需要降维？

### 维度灾难

```
1维（线段）：     100个点铺满1米线段 → 很密
2维（正方形）：   100个点铺在1m²面积 → 还行
3维（立方体）：   100个点铺在1m³体积 → 开始稀疏
100维：           100个点在100维空间 → 极度稀疏
                  → 距离都差不多远，模型失效
```

特征越多：
1. 需要的样本量指数级增长
2. 计算量增大
3. 过拟合风险增加
4. 很多特征可能相关或冗余

---

## PCA 核心思想

> **主成分分析**：找到数据方差最大（信息最多）的几个方向，把数据投影到这些方向上，用更少的维度表示大部分信息。

### 直觉

```
原始2D数据：              PCA 后1D：
    •                           降维到1维
  •   •     •           →      保留了最多的方差（信息）
    • •   •                   丢弃了方差最小的方向
      •
  特征1 vs 特征2               只保留主成分1
  （高度相关）                 （原来两个特征的线性组合）
```

### 数学过程（简化版）

```
1. 标准化数据（PCA 对尺度敏感）
2. 计算协方差矩阵
3. 求协方差矩阵的特征值和特征向量
4. 按特征值从大到小排序
5. 选前 K 个特征向量作为新坐标轴
6. 将数据投影到新坐标轴上
```

---

## sklearn 实现

### 基本用法

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

# PCA 前必须标准化！
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 降到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"原始形状: {X_scaled.shape}")  # (150, 4)
print(f"降维后形状: {X_pca.shape}")    # (150, 2)

# 可视化
plt.figure(figsize=(8, 6))
for target, color, name in zip([0, 1, 2], ['red', 'green', 'blue'], iris.target_names):
    plt.scatter(X_pca[y==target, 0], X_pca[y==target, 1],
                c=color, label=name, edgecolors='black', s=60)
plt.xlabel(f'第一主成分 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'第二主成分 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('PCA: 鸢尾花数据降到2维')
plt.legend()
plt.show()
```

### explained_variance_ratio（方差解释比）

```python
print(f"各主成分解释的方差比例:")
print(pca.explained_variance_ratio_)
# [0.7296, 0.2285]  → 第一主成分解释72.96%，第二主成分22.85%
print(f"总共保留: {pca.explained_variance_ratio_.sum():.1%}")
# 95.8%  → 2维保留了原来4维95.8%的信息
```

---

## 选择保留多少维度？

### 方法一：指定保留的方差比例

```python
# 自动选择维度，保留 95% 的方差
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"保留95%方差需要 {pca.n_components_} 个维度")
```

### 方法二：碎石图（Scree Plot）

```python
pca = PCA().fit(X_scaled)  # 保留所有维度

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
         pca.explained_variance_ratio_, 'bo-')
plt.xlabel('主成分编号')
plt.ylabel('方差解释比例')
plt.title('碎石图：选择主成分个数')
plt.grid(True)

# 累计方差
cumulative = np.cumsum(pca.explained_variance_ratio_)
plt.twinx()
plt.plot(range(1, len(cumulative)+1), cumulative, 'ro-')
plt.ylabel('累计方差比例', color='red')
plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.5)
plt.show()
```

---

## PCA 的应用场景

### 1. 数据可视化（降至2D/3D）

```python
# 高维数据无法直接可视化，PCA降到2维
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_high_dim)  # 比如从64维降到2维
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
```

### 2. 加速训练

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# 先降维再分类 → 训练更快
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # 保留95%信息
    ('svm', SVC())
])
```

### 3. 降噪

```python
# 通过丢弃方差小的成分（往往是噪声）来降噪
pca = PCA(n_components=10)  # 只保留前10个主成分
X_denoised = pca.inverse_transform(pca.fit_transform(X_noisy))
```

---

## PCA 实战：手写数字降维

```python
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data      # 1797个样本，每个64维（8x8像素）
y = digits.target

print(f"原始维度: {X.shape}")  # (1797, 64)

# 标准化 + PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=15, alpha=0.6)
plt.colorbar(scatter, label='数字')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('手写数字 PCA 降维（64D → 2D）')
plt.show()

print(f"2维保留了 {pca.explained_variance_ratio_.sum():.1%} 的信息")
```

---

## PCA 的变体

### IncrementalPCA（增量 PCA）

大数据无法一次性装入内存时：

```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=2, batch_size=100)
# 可以分批 fit
for batch in np.array_split(X_large, n_batches):
    ipca.partial_fit(batch)
X_transformed = ipca.transform(X_large)
```

### SparsePCA（稀疏 PCA）

结果更可解释——每个主成分只依赖少数原始特征：

```python
from sklearn.decomposition import SparsePCA

spca = SparsePCA(n_components=2, alpha=1)
X_spca = spca.fit_transform(X_scaled)
```

### KernelPCA（核 PCA）

能处理非线性降维：

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X_scaled)
```

---

## PCA 的优缺点

### 优点
- ✅ **降低维度**，减少计算量
- ✅ **降噪**（丢弃低方差成分）
- ✅ **可视化**高维数据
- ✅ **消除共线性**（主成分相互正交）
- ✅ 无监督，不需要标签

### 缺点
- ❌ **可解释性差**（主成分是原始特征的线性组合，难以命名）
- ❌ **线性方法**（只捕捉线性关系，非线性需用 t-SNE/UMAP）
- ❌ **方差大 ≠ 信息多**（有时方差小的方向才是区分类的关键）
- ❌ **对尺度敏感**（必须标准化）
- ❌ 可能丢失对任务有用的信息

---

## PCA vs 特征选择

| 对比项 | PCA（降维） | 特征选择 |
|--------|-----------|---------|
| 结果 | 新特征（原特征的组合） | 原始特征子集 |
| 可解释性 | 差 | 好 |
| 信息保留 | 按方差保留 | 按相关性/重要性 |
| 适合 | 压缩、可视化、降噪 | 需要解释、精简模型 |

---

## 可视化工具推荐

PCA 适合线性降维和快速可视化。对于更复杂的可视化：

```python
# t-SNE：非线性降维，可视化效果更好（但只能用于可视化，不能用于特征提取）
from sklearn.manifold import TSNE
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
```

> ⚠️ t-SNE 只能 `fit_transform`，没有 `transform` 方法，不能用于新数据。且计算量大。

---

## 小结

| 要点 | 内容 |
|------|------|
| 核心思想 | 保留方差最大的方向 |
| 必须标准化 | ✅ |
| 选择维度 | 指定数量 / 方差比例(0.95) / 碎石图 |
| 主要应用 | 可视化、加速、降噪 |
| 可解释性 | 差（主成分是组合） |
| 线性方法 | 非线性用 KernelPCA/t-SNE |

---

## 下一节

➡️ [05-异常检测.md](./05-异常检测.md)
