# 02 - DNN 模块入门

## 一、OpenCV DNN 模块概述

OpenCV 的 `dnn`（Deep Neural Network）模块可以加载预训练的深度学习模型进行推理。

```
特点：
✓ 不需要安装 PyTorch/TensorFlow
✓ 支持多种模型格式（ONNX、Caffe、TensorFlow）
✓ 支持 CUDA 加速
✓ 适合部署（轻量、跨平台）

限制：
✗ 不支持训练
✗ 算子覆盖不如完整框架
✗ 推理速度不如专用框架
```

## 二、加载模型

```python
import cv2
import numpy as np

# ─── 方式1：加载 Caffe 模型 ───
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "model.caffemodel")

# ─── 方式2：加载 TensorFlow 模型 ───
net = cv2.dnn.readNetFromTensorflow("frozen.pb", "config.pbtxt")

# ─── 方式3：加载 ONNX 模型（推荐）───
net = cv2.dnn.readNetFromONNX("model.onnx")

# ─── 方式4：加载 Darknet 模型（YOLO）───
net = cv2.dnn.readNetFromDarknet("yolov.cfg", "yolov.weights")

# ─── 方式5：通用加载（自动识别）───
net = cv2.dnn.readNet("model.onnx")
```

## 三、配置后端

```python
# 使用 CPU（默认）
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 使用 CUDA（需要 OpenCV 编译时启用 CUDA）
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# 使用 Intel OpenVINO
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
```

## 四、输入预处理

```python
img = cv2.imread("image.jpg")

# blobFromImage：图像 → 神经网络输入格式
blob = cv2.dnn.blobFromImage(
    img,
    scalefactor=1.0/255.0,   # 归一化因子
    size=(224, 224),          # 目标尺寸
    mean=(0, 0, 0),           # 减去的均值 (BGR)
    swapRB=True,              # BGR → RGB
    crop=False                # 是否裁剪
)

# blob 形状: (1, 3, 224, 224) → (batch, channels, height, width)

# ─── 批量处理 ───
images = [img1, img2, img3]
blob = cv2.dnn.blobFromImages(images, 1.0/255.0, (224, 224), swapRB=True)
```

### 常见归一化参数

| 模型 | scalefactor | size | mean | swapRB |
|------|------------|------|------|--------|
| ResNet | 1.0 | 224×224 | (104, 177, 123) | False |
| MobileNet | 1/127.5 | 224×224 | (127.5, 127.5, 127.5) | True |
| YOLO | 1/255.0 | 416×416 | (0, 0, 0) | True |
| SSD | 1.0 | 300×300 | (104, 177, 123) | False |

## 五、推理流程

```python
# 1. 加载模型
net = cv2.dnn.readNetFromONNX("resnet50.onnx")

# 2. 预处理
blob = cv2.dnn.blobFromImage(img, 1.0/255.0, (224, 224), swapRB=True)

# 3. 设置输入
net.setInput(blob)

# 4. 前向传播
output = net.forward()

# output 形状取决于模型，通常是 (1, num_classes)
print(f"输出形状: {output.shape}")
print(f"预测类别: {np.argmax(output[0])}")
```

## 六、获取模型层信息

```python
# 获取所有层名
layer_names = net.getLayerNames()
print(f"总层数: {len(layer_names)}")

# 获取输出层
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print(f"输出层: {output_layers}")

# 获取多个层的输出（用于中间特征提取）
layer_names = ["conv1", "pool1", "fc6"]
outputs = net.forward(layer_names)
```

## 七、图像分类示例

```python
def classify_image(img_path, model_path, labels_path):
    """使用 DNN 进行图像分类"""
    # 加载标签
    with open(labels_path) as f:
        labels = [line.strip() for line in f.readlines()]

    # 加载模型
    net = cv2.dnn.readNetFromONNX(model_path)

    img = cv2.imread(img_path)
    blob = cv2.dnn.blobFromImage(img, 1.0/255.0, (224, 224), swapRB=True)

    net.setInput(blob)
    output = net.forward()

    # Softmax 获取概率
    probabilities = cv2.softmax(output[0])

    # Top-5 预测
    top5 = np.argsort(probabilities)[::-1][:5]

    for idx in top5:
        print(f"{labels[idx]}: {probabilities[idx]*100:.2f}%")

classify_image("cat.jpg", "resnet50.onnx", "imagenet_labels.txt")
```

## 八、性能测量

```python
import time

# 推理计时
start = time.time()
output = net.forward()
elapsed = time.time() - start

print(f"推理时间: {elapsed*1000:.1f}ms")
print(f"FPS: {1/elapsed:.1f}")

# 使用 OpenCV 计时
t, _ = net.getPerfProfile()
print(f"推理时间: {t / cv2.getTickFrequency() * 1000:.1f}ms")
```

## 九、模型格式转换

### PyTorch → ONNX

```python
import torch

# 假设 model 是 PyTorch 模型
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)
```

### TensorFlow → ONNX

```bash
# 使用 tf2onnx
python -m tf2onnx.convert --input model.pb --output model.onnx
```

## 十、常见预训练模型下载

| 模型 | 用途 | 下载来源 |
|------|------|----------|
| YOLOv4/v5 | 目标检测 | pjreddie.com / github |
| SSD MobileNet | 目标检测 | TensorFlow Model Zoo |
| ResNet | 图像分类 | ONNX Model Zoo |
| OpenCV Face Detector | 人脸检测 | OpenCV repo |
| East Text Detector | 文字检测 | GitHub |

## 小结

| 操作 | 函数 |
|------|------|
| 加载模型 | `cv2.dnn.readNet()` |
| 预处理 | `cv2.dnn.blobFromImage()` |
| 推理 | `net.forward()` |
| 获取层 | `net.getLayerNames()` |
| 性能 | `net.getPerfProfile()` |

---

[下一节：YOLO 目标检测 →](03-YOLO目标检测.md)
