# 02 - DBSCAN 密度聚类

## 为什么需要 DBSCAN？

KMeans 的局限：

```
KMeans 处理不了的情况：
 1. 非球形簇（如两个月牙形）
 2. 不知道应该分几类
 3. 有噪声/异常值
```

DBSCAN（Density-Based Spatial Clustering of Applications with Noise）基于**密度**聚类，完美解决以上问题。

---

## 核心概念

### 1. 两个关键参数

| 参数 | 含义 |
|------|------|
| `eps` (ε) | 邻域半径：以某点为圆心，半径 eps 的范围 |
| `min_samples` | 成为核心点所需的最少邻居数 |

### 2. 三种点类型

```
核心点（Core）：     邻居数 ≥ min_samples
                    → 密度高，是簇的"骨架"

边界点（Border）：   邻居数 < min_samples，但在某个核心点的 eps 范围内
                    → 簇的边缘

噪声点（Noise）：    既不是核心点，也不在任何核心点附近
                    → 离群点/异常值
```

```
              eps 圆
            ┌─────────┐
            │   ● ← 核心点（圆内≥4个点）
            │  ● ●    │
            │ ●  ● ●  │
            │○   ● ●  │ ← ○ 边界点（在核心点eps内，但自身不够密）
            └─────────┘

     ×              ×     ← 噪声点（远离所有核心点）
```

---

## 算法步骤

```
1. 对每个点，计算 eps 范围内的邻居数
2. 标记核心点（邻居 ≥ min_samples）
3. 从一个核心点出发，把所有密度相连的核心点 + 边界点归为一个簇
4. 不属于任何簇的点标记为噪声（label = -1）
```

### 密度相连

```
核心点A → eps范围有核心点B → B的eps范围有核心点C → ...
  │                                        │
  └── 一条"密度链"连起来，都属于同一个簇 ──┘
```

---

## sklearn 实现

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 生成月牙形数据（KMeans 处理不了）
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)
X = StandardScaler().fit_transform(X)

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

print(f"发现的簇数: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"噪声点数: {list(labels).count(-1)}")

# 可视化
plt.figure(figsize=(8, 6))
unique_labels = set(labels)
colors = ['purple', 'gold', 'teal', 'coral']

for label in unique_labels:
    if label == -1:
        color = 'black'
        marker = 'x'
        name = '噪声'
    else:
        color = colors[label % len(colors)]
        marker = 'o'
        name = f'簇 {label}'

    mask = labels == label
    plt.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
                s=50, label=name, edgecolors='white')

plt.title(f'DBSCAN: {len(unique_labels)-1} 个簇')
plt.legend()
plt.show()
```

---

## 与 KMeans 对比

```python
from sklearn.cluster import KMeans

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# KMeans
km = KMeans(n_clusters=2, random_state=42, n_init=10)
km_labels = km.fit_predict(X)
axes[0].scatter(X[:, 0], X[:, 1], c=km_labels, cmap='viridis', s=50)
axes[0].set_title('KMeans: 错误地切成了两半')

# DBSCAN
db = DBSCAN(eps=0.3, min_samples=5)
db_labels = db.fit_predict(X)
axes[1].scatter(X[:, 0], X[:, 1], c=db_labels, cmap='viridis', s=50)
axes[1].set_title('DBSCAN: 正确分开了两个月牙')

plt.tight_layout()
plt.show()
```

---

## 参数选择

### eps 的选择：K-距离图

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# 计算每个点到第 min_samples 近邻的距离
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

# 取第5近邻的距离，排序
k_distances = np.sort(distances[:, 4])

plt.figure(figsize=(8, 4))
plt.plot(k_distances)
plt.xlabel('样本（排序后）')
plt.ylabel(f'第5近邻距离')
plt.title('K-距离图：选择 eps')
plt.grid(True)
# 选择曲线"拐点"处的 y 值作为 eps
plt.axhline(y=0.3, color='red', linestyle='--', label='eps ≈ 0.3')
plt.legend()
plt.show()
```

**经验法则**：
- `min_samples` ≥ 维度 + 1，通常取 5
- `eps` 从 K-距离图的拐点选择
- 如果太多点被标记为噪声 → 减小 eps 或减小 min_samples
- 如果所有点变成一个大簇 → 增大 eps 或增大 min_samples

---

## DBSCAN 的优缺点

### 优点
- ✅ **不需要预设簇数** K
- ✅ 能发现**任意形状**的簇（月牙、环、不规则）
- ✅ 自动识别**噪声/异常值**
- ✅ 对异常值鲁棒

### 缺点
- ❌ 对 eps 和 min_samples 敏感
- ❌ 不同密度簇难以同时处理（一组参数无法兼顾密度差异大的簇）
- ❌ 高维数据效果差（维度灾难）
- ❌ 数据量大时内存消耗高

---

## 何时用 DBSCAN vs KMeans？

| 场景 | 推荐 |
|------|------|
| 簇大致球形 | KMeans |
| 簇形状不规则 | DBSCAN |
| 不知道几类 | DBSCAN |
| 需要检测异常值 | DBSCAN |
| 数据量大，需要快 | KMeans（或 MiniBatchKMeans） |
| 簇密度差异大 | HDBSCAN（sklearn 外） |

---

## 小结

| 要点 | 内容 |
|------|------|
| 核心思想 | 基于密度连通性 |
| 关键参数 | eps（半径）、min_samples（最少邻居） |
| 不需指定 K | ✅ 自动发现簇数 |
| 异常检测 | 噪声标记为 -1 |
| 适合 | 任意形状簇、异常检测 |
| 不适合 | 密度差异大、高维 |

---

## 下一节

➡️ [03-层次聚类.md](./03-层次聚类.md)
