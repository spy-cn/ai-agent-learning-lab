# 04 - 第一个 OpenCV 程序

## 一、Hello OpenCV — 图像读写

最基础的 OpenCV 操作：读取、显示、保存图像。

```python
import cv2

# 1. 读取图像
img = cv2.imread("image.jpg")

# 检查是否读取成功
if img is None:
    print("错误：无法加载图像！")
    exit()

print(f"图像尺寸: {img.shape}")    # (高, 宽, 通道数)
print(f"数据类型: {img.dtype}")     # uint8
print(f"总像素数: {img.size}")      # 高×宽×通道数

# 2. 显示图像
cv2.imshow("Original", img)

# 3. 等待按键（0 表示无限等待）
key = cv2.waitKey(0)
print(f"按键码: {key}")  # 可用于实现交互

# 4. 保存图像
cv2.imwrite("output.jpg", img)

# 5. 关闭所有窗口
cv2.destroyAllWindows()
```

## 二、图像基本操作

```python
import cv2
import numpy as np

img = cv2.imread("image.jpg")

# ─── 像素操作 ───
# 获取像素值 (B, G, R)
pixel = img[100, 200]
print(f"BGR at (100,200): {pixel}")

# 修改像素
img[100, 200] = [255, 0, 0]

# ─── ROI 操作 ───
# 提取感兴趣区域 [y_start:y_end, x_start:x_end]
face_region = img[100:300, 200:400]

# 将 ROI 复制到其他位置
img[50:250, 50:250] = face_region

# ─── 通道操作 ───
# 分离通道
b, g, r = cv2.split(img)

# 只保留红色通道
zeros = np.zeros(img.shape[:2], dtype=np.uint8)
red_only = cv2.merge([zeros, zeros, r])

# ─── 图像缩放 ───
# 按指定尺寸
resized = cv2.resize(img, (320, 240))

# 按比例缩放
h, w = img.shape[:2]
scaled = cv2.resize(img, None, fx=0.5, fy=0.5,
                    interpolation=cv2.INTER_AREA)
```

## 三、图像处理流水线

一个完整的图像处理示例：

```python
import cv2
import numpy as np

def process_image(image_path):
    """图像处理流水线"""
    # Step 1: 读取
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取: {image_path}")

    # Step 2: 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 4: Canny 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # Step 5: 轮廓检测
    contours, _ = cv2.findContours(edges,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 6: 在原图上绘制轮廓
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # 输出结果
    print(f"检测到 {len(contours)} 个轮廓")

    # 显示结果
    images = {
        "Original": img,
        "Gray": gray,
        "Edges": edges,
        "Result": result
    }

    return images

# 运行
results = process_image("image.jpg")

for name, img in results.items():
    cv2.imshow(name, img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 四、摄像头/视频操作

```python
import cv2

# 打开摄像头（0 = 默认摄像头）
cap = cv2.VideoCapture(0)

# 检查是否打开成功
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        break

    # 处理帧
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 120)

    # 显示
    cv2.imshow("Camera", frame)
    cv2.imshow("Edges", edges)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
```

## 五、实用工具函数

建议在项目中维护以下工具函数：

```python
import cv2
import numpy as np

def display_images(images_dict, window_name="Display"):
    """在网格中显示多张图像"""
    # images_dict: {"Title": img, ...}
    images = list(images_dict.values())
    titles = list(images_dict.keys())

    # 统一尺寸
    h = min(img.shape[0] for img in images)
    w = min(img.shape[1] for img in images)
    images = [cv2.resize(img, (w, h)) for img in images]

    # 水平拼接
    combined = np.hstack(images)

    cv2.imshow(window_name, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def put_text_cn(img, text, org, color=(0, 255, 0), size=0.7):
    """在图像上绘制中文文字（使用 Pillow）"""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("simhei.ttf", int(size * 20))
    draw.text(org, text, fill=color[::-1], font=font)  # RGB -> BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def add_timestamp(img):
    """添加时间戳"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, timestamp, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img
```

## 六、常见错误排查

| 错误信息 | 原因 | 解决方法 |
|----------|------|----------|
| `img is None` | 文件路径错误 | 检查路径，使用绝对路径 |
| `error: (-215) ...` | 参数无效 | 检查输入图像格式和尺寸 |
| `cv2.error: OpenCV(4.x)` | 版本不兼容 | 检查 API 变更 |
| `imshow 不显示` | headless 版本 | 安装非 headless 版本 |

## 小结

| 操作 | 函数 |
|------|------|
| 读取图像 | `cv2.imread()` |
| 显示图像 | `cv2.imshow()` + `cv2.waitKey()` |
| 保存图像 | `cv2.imwrite()` |
| 读取视频 | `cv2.VideoCapture()` |
| 像素访问 | `img[y, x]` |
| ROI | `img[y1:y2, x1:x2]` |

恭喜！你已经完成了 OpenCV 的第一个程序。接下来进入图像基础篇深入学习。

---

[→ 进入图像基础篇](../02-图像基础篇/01-数字图像基础.md)
