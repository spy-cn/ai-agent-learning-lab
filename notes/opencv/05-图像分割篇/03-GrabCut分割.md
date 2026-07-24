# 03 - GrabCut 交互式分割

## 一、GrabCut 原理

GrabCut 是一种基于图割（Graph Cut）的交互式前景提取算法。用户指定一个矩形区域，算法迭代优化分割边界。

```
用户操作：
┌───────────────────────────┐
│ ┌─────────────────────┐   │
│ │                     │   │  ← 矩形框包含前景
│ │    ██████████       │   │
│ │    ██████████       │   │
│ │                     │   │
│ └─────────────────────┘   │
└───────────────────────────┘

GrabCut 迭代：
1. 用高斯混合模型(GMM)建模前景和背景
2. 优化能量函数，更新分割
3. 用户可手动标记修正
4. 重复 2-3 直到收敛
```

## 二、基本用法

```python
import cv2
import numpy as np

img = cv2.imread("person.jpg")

# 定义包含前景的矩形 (x, y, width, height)
rect = (50, 50, 400, 500)

# 初始化 mask
mask = np.zeros(img.shape[:2], np.uint8)

# 初始化背景和前景模型
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# GrabCut
cv2.grabCut(img, mask, rect, bgd_model, fgd_model,
    iterCount=5,  # 迭代次数
    mode=cv2.GC_INIT_WITH_RECT)

# mask 值含义：
# 0 (GC_BGD):    确定背景
# 1 (GC_FGD):    确定前景
# 2 (GC_PR_BGD): 可能背景
# 3 (GC_PR_FGD): 可能前景

# 提取前景：将 0 和 2 设为背景
mask_result = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
result = img * mask_result[:, :, np.newaxis]

cv2.imshow("GrabCut Result", result)
cv2.waitKey(0)
```

## 三、带 Mask 的 GrabCut（手动修正）

```python
# 初始矩形
cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# 手动标记修正
# 用户在 mask 上画白色（确定前景）和黑色（确定背景）
mask[certain_fg_y1:certain_fg_y2, certain_fg_x1:certain_fg_x2] = 1  # GC_FGD
mask[certain_bg_y1:certain_bg_y2, certain_bg_x1:certain_bg_x2] = 0  # GC_BGD

# 用 mask 模式继续迭代
cv2.grabCut(img, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

# 提取结果
mask_result = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
result = img * mask_result[:, :, np.newaxis]
```

## 四、交互式 GrabCut

```python
import cv2
import numpy as np

class GrabCutApp:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.mask = np.zeros(self.img.shape[:2], np.uint8)
        self.bgd_model = np.zeros((1, 65), np.float64)
        self.fgd_model = np.zeros((1, 65), np.float64)
        self.rect = None
        self.drawing = False
        self.mode = None  # 'rect' or 'brush'
        self.brush_size = 5

    def mouse_callback(self, event, x, y, flags, param):
        if self.mode == 'rect':
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.temp_rect = (self.start[0], self.start[1], x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.rect = (min(self.start[0], x), min(self.start[1], y),
                            abs(x - self.start[0]), abs(y - self.start[1]))

    def run(self):
        cv2.namedWindow("GrabCut")
        cv2.setMouseCallback("GrabCut", self.mouse_callback)

        print("操作：r=画矩形 | s=执行分割 | q=退出")

        while True:
            display = self.img.copy()
            if hasattr(self, 'temp_rect') and self.drawing:
                cv2.rectangle(display, (self.temp_rect[0], self.temp_rect[1]),
                             (self.temp_rect[2], self.temp_rect[3]), (0, 255, 0), 2)

            cv2.imshow("GrabCut", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and self.rect:
                cv2.grabCut(self.img, self.mask, self.rect,
                           self.bgd_model, self.fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
                result = self.img * mask2[:, :, np.newaxis]
                cv2.imshow("Result", result)
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

# 运行
app = GrabCutApp("person.jpg")
app.run()
```

## 五、GrabCut 应用场景

| 场景 | 效果 |
|------|------|
| 人像抠图 | 优秀 |
| 产品去背景 | 优秀 |
| 复杂背景分割 | 中等 |
| 多目标分割 | 需多次操作 |

## 小结

| 参数 | 说明 |
|------|------|
| rect | 前景大致区域 |
| iterCount | 迭代次数（5-10） |
| GC_INIT_WITH_RECT | 矩形初始化 |
| GC_INIT_WITH_MASK | Mask 修正 |

---

[下一节：超像素分割 →](04-超像素分割.md)
