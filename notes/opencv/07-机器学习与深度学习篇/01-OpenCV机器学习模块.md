# 01 - OpenCV 机器学习模块

## 一、OpenCV ml 模块概述

OpenCV 内置了 `ml`（Machine Learning）模块，提供传统机器学习算法，无需额外依赖。

```
OpenCV ml 模块包含：
├── SVM (支持向量机)
├── KNN (K近邻)
├── Decision Tree (决策树)
├── Random Trees (随机森林)
├── Boosting (提升)
├── Logistic Regression (逻辑回归)
├── ANN_MLP (人工神经网络)
├── Naive Bayes (朴素贝叶斯)
└── EM (期望最大化)
```

## 二、SVM 支持向量机

```python
import cv2
import numpy as np

# 准备数据
# 特征：[hue, saturation]
train_data = np.array([
    [20, 200], [30, 220], [25, 180],  # 黄色
    [120, 200], [130, 220], [125, 180] # 蓝色
], dtype=np.float32)

labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

# 创建 SVM
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)  # LINEAR / RBF / POLY / SIGMOID
svm.setC(2.67)     # 惩罚参数
svm.setGamma(5.383) # RBF 核参数

# 训练
svm.train(train_data, cv2.ml.ROW_SAMPLE, labels)

# 预测
test = np.array([[28, 210]], dtype=np.float32)
_, result = svm.predict(test)
print(f"预测类别: {result[0][0]}")  # 0=黄, 1=蓝

# 保存/加载
svm.save("svm_model.xml")
svm_loaded = cv2.ml.SVM_load("svm_model.xml")
```

## 三、KNN K近邻

```python
# 创建 KNN
knn = cv2.ml.KNearest_create()

# 训练
knn.train(train_data, cv2.ml.ROW_SAMPLE, labels)

# 预测
test = np.array([[28, 210]], dtype=np.float32)
k = 3  # 最近邻数
_, results, neighbors, dist = knn.findNearest(test, k)
print(f"预测: {results[0][0]}")
print(f"邻居标签: {neighbors[0]}")
```

## 四、决策树与随机森林

```python
# ─── 决策树 ───
dtree = cv2.ml.DTrees_create()
dtree.setMaxDepth(10)
dtree.setMinSampleCount(2)
dtree.setRegressionAccuracy(0.01)
dtree.setUse1SERules(True)
dtree.setTruncatePrunedTree(True)
dtree.train(train_data, cv2.ml.ROW_SAMPLE, labels)

_, result = dtree.predict(test)

# ─── 随机森林 ───
rf = cv2.ml.RTrees_create()
rf.setMaxDepth(10)
rf.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1))
rf.train(train_data, cv2.ml.ROW_SAMPLE, labels)

_, result = rf.predict(test)
```

## 五、人工神经网络（ANN_MLP）

```python
# 创建 ANN
ann = cv2.ml.ANN_MLP_create()

# 层结构：输入层 → 隐藏层 → 输出层
layer_sizes = np.array([2, 10, 2], dtype=np.int32)  # 2输入, 10隐藏, 2输出
ann.setLayerSizes(layer_sizes)
ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0)
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
ann.setBackpropMomentumScale(0.1)
ann.setBackpropWeightScale(0.1)

# 需要将标签转为 one-hot 编码
labels_onehot = np.zeros((len(labels), 2), dtype=np.float32)
for i, label in enumerate(labels):
    labels_onehot[i][label] = 1.0

# 训练
ann.train(train_data, cv2.ml.ROW_SAMPLE, labels_onehot)

# 预测
_, result = ann.predict(test)
print(f"预测概率: {result}")
```

## 六、逻辑回归

```python
lr = cv2.ml.LogisticRegression_create()
lr.setLearningRate(0.001)
lr.setIterations(1000)
lr.setRegularization(cv2.ml.LogisticRegression_REG_L2)
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(10)

lr.train(train_data, cv2.ml.ROW_SAMPLE, labels)
_, result = lr.predict(test)
```

## 七、训练数据准备

```python
def prepare_training_data(images, labels):
    """将图像列表转为训练数据矩阵"""
    # 特征提取（如 HOG、颜色直方图等）
    features = []
    for img in images:
        # 示例：使用缩小的灰度图像作为特征
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        resized = cv2.resize(gray, (32, 32))
        features.append(resized.flatten().astype(np.float32))

    train_data = np.array(features, dtype=np.float32)
    train_labels = np.array(labels, dtype=np.int32)

    return train_data, train_labels

# 使用 HOG 特征
def extract_hog_features(img):
    """提取 HOG 特征"""
    hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(img, (64, 128))
    features = hog.compute(resized)
    return features.flatten()
```

## 八、模型保存与加载

```python
# 所有 ml 模型都支持 XML/YAML 格式保存
svm.save("model_svm.xml")
knn.save("model_knn.xml")
ann.save("model_ann.xml")

# 加载
svm = cv2.ml.SVM_load("model_svm.xml")
knn = cv2.ml.KNearest_create()
knn.read(cv2.FileStorage("model_knn.xml", cv2.FILE_STORAGE_READ).getFirstTopLevelNode())
```

## 九、算法对比

| 算法 | 速度 | 精度 | 适用 |
|------|------|------|------|
| KNN | 训练快，预测慢 | 中 | 小数据 |
| SVM | 中 | 高 | 中小数据 |
| 决策树 | 快 | 中 | 可解释性 |
| 随机森林 | 中 | 高 | 通用 |
| ANN | 慢 | 高 | 复杂模式 |

## 小结

OpenCV 的 ml 模块适合：
- 轻量级应用（无需额外库）
- 传统视觉分类任务
- 嵌入式部署

对于复杂任务（图像分类、目标检测），推荐使用深度学习框架（PyTorch/TensorFlow）+ OpenCV 的 dnn 模块做推理。

---

[下一节：DNN 模块入门 →](02-DNN模块入门.md)
