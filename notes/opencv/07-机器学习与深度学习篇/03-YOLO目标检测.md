# 03 - YOLO 目标检测

## 一、YOLO 概述

YOLO（You Only Look Once）是最流行的实时目标检测算法。

```
YOLO 工作原理：

输入图像 → 网格划分 → 每个网格预测
                         ↓
                    边界框 + 置信度 + 类别
                         ↓
                    NMS 去重 → 最终结果

┌─────┬─────┬─────┐
│     │     │  ●  │  ← 每个网格预测多个框
├─────┼─────┼─────┤
│     │ □   │     │
├─────┼─────┼─────┤
│     │     │     │
└─────┴─────┴─────┘
```

## 二、YOLO 版本对照

| 版本 | 特点 | OpenCV 支持 |
|------|------|------------|
| YOLOv3 | 经典稳定 | ✓ |
| YOLOv4 | 性能提升 | ✓ |
| YOLOv5 | PyTorch 原生 | 需转 ONNX |
| YOLOv7/v8 | 最新 | 需转 ONNX |

## 三、加载 YOLO 模型

```python
import cv2
import numpy as np

# 加载模型
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# 设置后端
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 加载类别标签
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 获取输出层名
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
```

## 四、单图检测

```python
def detect_objects(img_path, conf_threshold=0.5, nms_threshold=0.4):
    """YOLO 目标检测"""
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # 预处理
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # 推理
    outputs = net.forward(output_layers)

    # 解析结果
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                # YOLO 输出的是相对坐标 (0-1)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 转为左上角坐标
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 非极大值抑制 (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # 绘制结果
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = colors[class_ids[i]]

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

result = detect_objects("street.jpg")
cv2.imshow("YOLO Detection", result)
cv2.waitKey(0)
```

## 五、视频实时检测

```python
def detect_video(video_path=0):
    """视频/摄像头实时检测"""
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 缩小帧尺寸以提高速度
        frame = cv2.resize(frame, (640, 480))

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        # 解析（同上）
        boxes, confidences, class_ids = [], [], []
        h, w = frame.shape[:2]

        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if conf > 0.5:
                    cx, cy = int(det[0]*w), int(det[1]*h)
                    bw, bh = int(det[2]*w), int(det[3]*h)
                    boxes.append([cx-bw//2, cy-bh//2, bw, bh])
                    confidences.append(float(conf))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
                cv2.rectangle(frame, (x,y), (x+bw, y+bh), (0,255,0), 2)
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # FPS
        t, _ = net.getPerfProfile()
        fps = cv2.getTickFrequency() / t
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("YOLO Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

## 六、ONNX 格式 YOLO

```python
# YOLOv5/v8 需要先导出为 ONNX
# 导出命令（在 YOLOv5 目录下）：
# python export.py --weights yolov5s.pt --include onnx

# 加载 ONNX YOLO
net = cv2.dnn.readNetFromONNX("yolov5s.onnx")

# YOLOv5 ONNX 的输出格式略有不同
# 需要调整后处理代码
```

## 七、性能优化建议

```python
# 1. 降低输入分辨率（牺牲精度换速度）
input_size = (320, 320)  # 默认 416，可降到 320

# 2. 使用 CUDA
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# 3. 跳帧检测（每 N 帧检测一次）
detect_every_n = 3
frame_count = 0

while True:
    ret, frame = cap.read()
    if frame_count % detect_every_n == 0:
        detections = detect_objects(frame)
    else:
        # 使用上一帧结果
        pass
    frame_count += 1
```

## 小结

| 操作 | 函数 |
|------|------|
| 加载 | `cv2.dnn.readNet()` |
| 推理 | `net.forward()` |
| NMS | `cv2.dnn.NMSBoxes()` |
| FPS | `net.getPerfProfile()` |

---

[下一节：人脸检测与识别 →](04-人脸检测与识别.md)
