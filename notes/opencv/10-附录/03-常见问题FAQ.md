# 常见问题 FAQ

## 安装问题

### Q1: `ModuleNotFoundError: No module named 'cv2'`

```bash
# 解决：安装 opencv
pip install opencv-contrib-python
```

### Q2: `libGL.so.1: cannot open shared object file`

```bash
# Linux 服务器无 GUI 依赖
# 方案1: 安装 headless 版
pip install opencv-python-headless

# 方案2: 安装系统依赖
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

### Q3: 多个 opencv 包冲突

```bash
# 彻底清理后重装
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-contrib-python
```

### Q4: `cv2.xfeatures2d` 无法使用

```bash
# 需要安装 contrib 版本
pip install opencv-contrib-python
# SIFT 已在 OpenCV 4.4+ 免费使用
# SURF 仍需要 contrib（某些版本）
```

## 运行问题

### Q5: `cv2.imshow()` 窗口不响应

```python
# 原因：缺少 waitKey
cv2.imshow("img", img)
cv2.waitKey(0)  # 必须有这一行！
cv2.destroyAllWindows()
```

### Q6: 视频播放速度异常

```python
# 原因：waitKey 延迟不对
fps = 30
delay = int(1000 / fps)  # 毫秒
cv2.waitKey(delay)
```

### Q7: 摄像头延迟严重

```python
# 方案1: 减小缓冲区
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 方案2: 使用线程读取
import threading
class CameraThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def read(self):
        return self.ret, self.frame
```

### Q8: 中文路径无法读取图像

```python
# cv2.imread 不支持中文路径
import numpy as np

# 解决方案
def imread_cn(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

def imwrite_cn(path, img):
    cv2.imencode('.jpg', img)[1].tofile(path)
```

### Q9: 图像颜色不对（红蓝反转）

```python
# OpenCV 使用 BGR，matplotlib 使用 RGB
# 显示前转换
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
```

### Q10: SIFT/SURF 不可用

```python
# OpenCV 4.4+ SIFT 已免费
sift = cv2.SIFT_create()  # 直接使用

# 确保安装的是 4.4+
print(cv2.__version__)
```

## 性能问题

### Q11: 处理速度太慢

```
优化清单：
1. 使用向量化操作（避免 for 循环）
2. 降低图像分辨率
3. 只处理 ROI
4. 使用 UMat（OpenCL 加速）
5. 使用多线程/多进程
6. 使用更快算法（ORB > SIFT）
```

### Q12: 内存占用过高

```python
# 及时释放变量
del large_array
import gc
gc.collect()

# 使用生成器处理大文件
def frame_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()
```

### Q13: CUDA 不可用

```python
# 检查 OpenCV 是否编译了 CUDA
print(cv2.cuda.getCudaEnabledDeviceCount())
# 返回 0 → 需要从源码编译 CUDA 版本
```

## 兼容性问题

### Q14: `cv2.face` 找不到

```bash
# face 模块在 contrib 中
pip install opencv-contrib-python
```

### Q15: `cv2.TrackerXXX_create` 找不到

```python
# OpenCV 4.5+ 追踪器移到了主模块
# 旧: cv2.TrackerCSRT_create() (contrib)
# 新: cv2.TrackerCSRT_create() (main)

# 确保版本
print(cv2.__version__)
```

### Q16: VideoWriter 输出文件为 0KB

```python
# 检查：
# 1. fourcc 编码是否被系统支持
# 2. 分辨率是否与帧一致
# 3. 是否调用了 release()

out = cv2.VideoWriter("out.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    30, (640, 480))  # ← 必须与帧尺寸一致

# 写入...
out.release()  # ← 必须调用！
```

## 常见错误

### Q17: `error: (-215:Assertion failed) ...`

```
原因：函数参数无效（尺寸/类型不匹配）

检查：
- 图像是否为 None（路径错误？）
- 图像通道数是否正确
- 数组形状是否符合要求
- 数据类型是否正确（uint8 vs float32）
```

### Q18: 轮廓检测为空

```python
# 轮廓检测需要二值图像
# 确保先做了 threshold 或 Canny
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### Q19: `TypeError: expected integer argument`

```python
# OpenCV 的坐标参数需要是整数
# 错误：cv2.circle(img, (100.5, 200.3), ...)
# 正确：
cv2.circle(img, (int(x), int(y)), int(r), ...)
```

---

[下一节：推荐资源 →](04-推荐资源.md)
