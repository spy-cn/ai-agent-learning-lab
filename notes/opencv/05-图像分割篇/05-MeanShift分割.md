# 05 - Mean-Shift 分割

## 一、Mean-Shift 原理

Mean-Shift 是一种基于密度的非参数聚类方法，用于图像分割和滤波。

```
Mean-Shift 流程：
1. 对每个像素，在其邻域内计算密度梯度
2. 沿梯度方向移动到密度最大值
3. 相似像素被分配到同一区域

效果：颜色相近的像素被合并为区域
```

## 二、pyrMeanShiftFiltering

OpenCV 中最常用的 Mean-Shift 函数是 `pyrMeanShiftFiltering`。

```python
import cv2

img = cv2.imread("image.jpg")

# Mean-Shift 滤波
result = cv2.pyrMeanShiftFiltering(
    img,
    sp=20,       # 空间窗口半径
    sr=30,       # 颜色窗口半径
    maxLevel=1   # 金字塔层数
)

cv2.imshow("Mean-Shift Filtered", result)
cv2.waitKey(0)
```

### 参数详解

| 参数 | 含义 | 效果 |
|------|------|------|
| sp | 空间窗口半径 | 越大区域越大 |
| sr | 颜色窗口半径 | 越大颜色合并越多 |
| maxLevel | 金字塔层数 | 越大计算越快但精度低 |

```
sp=10, sr=10：  保留细节，区域小
sp=40, sr=40：  强烈合并，区域大

sp 控制空间平滑（类似滤波器大小）
sr 控制颜色平滑（类似颜色量化级别）
```

## 三、Mean-Shift 分割流程

```python
import cv2
import numpy as np

img = cv2.imread("image.jpg")

# Step 1: Mean-Shift 滤波
filtered = cv2.pyrMeanShiftFiltering(img, sp=20, sr=30)

# Step 2: 转灰度
gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

# Step 3: 阈值化（或 Canny）
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 4: 查找轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 5: 填充每个区域为随机颜色
result = np.zeros_like(img)
for i, cnt in enumerate(contours):
    color = [int(c) for c in np.random.randint(0, 255, 3)]
    cv2.drawContours(result, [cnt], -1, color, -1)

cv2.imshow("Mean-Shift Segmentation", result)
cv2.waitKey(0)
```

## 四、Mean-Shift 跟踪

Mean-Shift 也用于目标跟踪——在连续帧中追踪目标。

```python
# 设置初始跟踪窗口
r, h, c, w = 250, 80, 400, 100  # y, height, x, width
track_window = (c, r, w, h)

cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read()

# 设置 ROI
roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# 计算直方图
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 设置终止条件
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Mean-Shift 跟踪
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    # 绘制跟踪框
    x, y, w, h = track_window
    result = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Mean-Shift Tracking", result)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 五、CamShift（连续自适应 Mean-Shift）

CamShift 是 Mean-Shift 的改进版，能自动调整窗口大小。

```python
while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # CamShift（会返回旋转矩形）
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # 绘制旋转矩形
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    result = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    cv2.imshow("CamShift", result)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
```

## 六、分割方法对比

| 方法 | 速度 | 效果 | 适用场景 |
|------|------|------|----------|
| 阈值 | 极快 | 简单 | 二值分割 |
| Mean-Shift | 中 | 好 | 颜色分割 |
| 分水岭 | 中 | 好 | 分离粘连 |
| GrabCut | 慢 | 很好 | 交互式抠图 |
| 超像素 | 中 | 好 | 中间表示 |

## 小结

| 功能 | 函数 |
|------|------|
| Mean-Shift 滤波 | `cv2.pyrMeanShiftFiltering()` |
| Mean-Shift 跟踪 | `cv2.meanShift()` |
| CamShift 跟踪 | `cv2.CamShift()` |

---

[→ 进入计算机视觉应用篇](../06-计算机视觉应用篇/01-模板匹配.md)
