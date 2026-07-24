# 01 - KMeans 聚类

## 什么是聚类？

> **物以类聚，人以群分。** 把相似的样本自动分到一组，不预先告诉算法应该分几类。

```
聚类前：                        聚类后：
  •          •                     • ■        ■ ■
      •         •               • • ■ ■      ■ ■ ■
         •             →            ■ ■        ■
    •         •                  • ■ ■        ■ ■
       •                                ■
                                    （自动发现了3个群）
```

与分类的区别：
- **分类**（监督）：预先知道有哪些类别，学习如何分类
- **聚类**（无监督）：不知道有什么类别，自动发现群组

---

## KMeans 算法原理

### 核心思想

给定 K 个簇，KMeans 通过迭代寻找每个簇的**中心点（质心）**，让样本到其所属簇中心的距离之和最小。

### 算法步骤

```
1. 随机选择 K 个点作为初始质心
2. 将每个样本分配到最近的质心 → 形成 K 个簇
3. 重新计算每个簇的质心（所有点的平均位置）
4. 重复步骤 2-3，直到质心不再变化（收敛）
```

### 可视化过程

```
迭代1：          迭代2：          迭代3：          收敛：
  ★    •   •       ★ ■   ■          ■ ■ ★ ■        ■ ■ ★ ■
•  •     •    →   •  • ■   ■   →    ■ ■ ■   →      ■ ■ ■
    •     ★          ★ •          •  ★ ■          ■ ★ ■
  •     •            ■   ■          ■ ■ ■          ■ ■ ■
     ★              ★               ★ ■            ★ ■

★=质心  ■=属于某簇  •=未分配
```

---

## sklearn 实现

### 基本用法

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成模拟数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# KMeans 聚类
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)

# 获取结果
labels = kmeans.labels_           # 每个样本的簇标签
centers = kmeans.cluster_centers_  # 簇中心坐标
print(f"簇中心:\n{centers}")

# 预测新样本
new_point = [[0, 0]]
cluster = kmeans.predict(new_point)
print(f"新样本属于簇: {cluster[0]}")
```

### 可视化

```python
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='质心')
plt.title(f'KMeans 聚类 (K=4)')
plt.legend()
plt.colorbar(label='簇编号')
plt.show()
```

---

## 如何选择 K 值？

### 方法一：肘部法则（Elbow Method）

尝试不同的 K，画出**惯性（inertia）**随 K 变化的曲线，找"肘部"。

```python
k_range = range(1, 11)
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('K (簇数)')
plt.ylabel('Inertia (簇内距离平方和)')
plt.title('肘部法则选择 K')
plt.axvline(x=4, color='red', linestyle='--', label='肘部 K=4')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**肘部法则的直觉**：
```
inertia
  │
高│  ╲
  │   ╲
  │    ╲
  │     ╲____          ← 这里有个"肘部"（拐点）
  │          \___
低│              \___
  └──────────────────► K
   1  2  3  4  5  6  7
```

K 增大时 inertia 总是减小（极端情况 K=样本数，inertia=0）。我们要找的是**下降速率明显变缓**的点。

### 方法二：轮廓系数（Silhouette Score）

```python
from sklearn.metrics import silhouette_score

k_range = range(2, 11)
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores.append(score)

best_k = k_range[np.argmax(sil_scores)]
print(f"最佳 K（轮廓系数）: {best_k}")

plt.plot(k_range, sil_scores, 'ro-')
plt.xlabel('K')
plt.ylabel('轮廓系数')
plt.title('轮廓系数法（越大越好）')
plt.grid(True)
plt.show()
```

**轮廓系数含义**：
- 范围 [-1, 1]
- 接近 1：样本离自己簇很近，离其他簇很远（好）
- 接近 0：样本在簇边界（模糊）
- 接近 -1：样本可能被分错簇了（差）

---

## KMeans 的关键参数

```python
KMeans(
    n_clusters=8,         # K 值
    init='k-means++',     # 初始化方法
    n_init=10,            # 不同初始化运行次数
    max_iter=300,         # 最大迭代次数
    random_state=42       # 随机种子
)
```

### init 参数

| 值 | 说明 |
|----|------|
| `'k-means++'` | 智能选择初始点（默认，推荐） |
| `'random'` | 随机选初始点 |
| 数组 | 手动指定初始质心 |

> 💡 `k-means++` 通过让初始质心互相远离，显著减少了陷入差解的概率。

### n_init 参数

KMeans 对初始化敏感，sklearn 默认运行 10 次（`n_init=10`），取最好的结果。

---

## KMeans 的局限

### 1. 需要预先指定 K

实际中你通常不知道数据应该分几类。需要用肘部法则或轮廓系数辅助判断。

### 2. 假设簇是球形的

```
适合 KMeans：              不适合 KMeans：
  • • ★ ■ ■                 • • •  ■ ■
 • ★★ ■■■ ★              • • ★ ★  ■ ■ ★
  • • ■ ★ ■                • • •  ■ ■
 • • • ★ ■                ← 两个月牙形，KMeans 分不开
（球形簇）
```

### 3. 对异常值敏感

一个远离的异常点可能独自形成一个簇，或扭曲质心。

### 4. 对初始值敏感

不同的初始化可能得到不同的结果（`n_init` 缓解此问题）。

---

## 数据缩放的重要性

```python
from sklearn.preprocessing import StandardScaler

# KMeans 基于距离，必须缩放！
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans.fit(X_scaled)
```

如果两个特征量级差异大（如年龄 vs 收入），不缩放会让大量级特征完全主导距离计算。

---

## 实战：客户分群

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 模拟客户数据：消费频率、平均消费金额
np.random.seed(42)
data = {
    '消费频率': np.concatenate([
        np.random.normal(5, 1, 100),    # 低频客户
        np.random.normal(15, 2, 100),   # 中频
        np.random.normal(30, 3, 100),   # 高频
    ]),
    '平均消费': np.concatenate([
        np.random.normal(100, 20, 100),
        np.random.normal(500, 50, 100),
        np.random.normal(1500, 100, 100),
    ])
}
df = pd.DataFrame(data)

# 缩放
scaler = StandardScaler()
X = scaler.fit_transform(df)

# 用肘部法则选 K
inertias = []
for k in range(1, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

# 聚类
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

# 可视化
plt.figure(figsize=(8, 6))
for c in range(3):
    subset = df[df['cluster'] == c]
    plt.scatter(subset['消费频率'], subset['平均消费'], label=f'群组{c}', s=50)
plt.xlabel('消费频率')
plt.ylabel('平均消费')
plt.title('客户分群结果')
plt.legend()
plt.show()

# 各群组特征
print(df.groupby('cluster').mean())
```

---

## 小结

| 要点 | 内容 |
|------|------|
| 核心思想 | 最小化簇内距离 |
| 算法步骤 | 选中心 → 分配 → 更新中心 → 重复 |
| 选择 K | 肘部法则 / 轮廓系数 |
| 必须缩放 | ✅ 基于距离 |
| 优点 | 简单、快速、可扩展 |
| 缺点 | 需指定K、只适合球形簇、异常值敏感 |

---

## 下一节

➡️ [02-DBSCAN密度聚类.md](./02-DBSCAN密度聚类.md)
