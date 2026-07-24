# 02 - Mat 与 NumPy 数组操作

## 一、OpenCV 中的图像就是 NumPy 数组

在 Python 中，OpenCV 的图像（`cv2.Mat`）就是 **NumPy ndarray**。这意味着你可以用所有 NumPy 操作来处理图像。

```python
import cv2
import numpy as np

img = cv2.imread("image.jpg")

print(type(img))    # <class 'numpy.ndarray'>
print(img.shape)    # (H, W, C) 例如 (480, 640, 3)
print(img.dtype)    # uint8
print(img.ndim)     # 3 (维度数)
print(img.size)     # H × W × C (总元素数)
```

## 二、属性详解

```python
h, w, c = img.shape
print(f"高度: {h} 像素")
print(f"宽度: {w} 像素")
print(f"通道数: {c}")  # 灰度=1, 彩色=3, RGBA=4

# 图像的数据类型
print(f"每个像素占用: {img.itemsize} 字节")
print(f"总内存: {img.nbytes / 1024:.1f} KB")
```

## 三、像素访问方式

```python
# ─── 方式 1：直接索引（最直观）───
pixel = img[100, 50]         # 返回 [B, G, R]
blue = img[100, 50, 0]      # 单通道
img[100, 50] = [255, 0, 0]  # 修改

# ─── 方式 2：item / itemset（更快）───
blue = img.item(100, 50, 0)
img.itemset(100, 50, 0, 255)

# ─── 方式 3：NumPy 花式索引 ───
# 获取某个矩形区域所有红色通道
red_channel_block = img[50:150, 50:150, 2]
```

## 四、ROI（Region of Interest）

ROI 是图像中的一个矩形区域，是日常操作中最常用的技巧。

```python
# ─── 提取 ROI ───
# 语法：img[y_start:y_end, x_start:x_end]
face = img[100:300, 200:400]

# ─── 复制 ROI ───
# 将一个区域复制到另一个位置
region = img[100:200, 100:200].copy()
img[200:300, 300:400] = region

# ─── ROI 用于处理 ───
# 只对某个区域做模糊
img[100:200, 100:200] = cv2.GaussianBlur(img[100:200, 100:200], (15, 15), 0)
```

## 五、通道操作

```python
# ─── 分离通道 ───
# 方法 1：cv2.split（返回三个独立数组）
b, g, r = cv2.split(img)

# 方法 2：NumPy 索引（更高效）
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]

# ─── 合并通道 ───
merged = cv2.merge([b, g, r])

# ─── 替换单个通道 ───
img[:, :, 2] = 0  # 将红色通道设为 0

# ─── 只保留某一通道 ───
zeros = np.zeros_like(b)
red_only = cv2.merge([zeros, zeros, r])
green_only = cv2.merge([zeros, g, zeros])
blue_only = cv2.merge([b, zeros, zeros])

# ─── 通道交换（BGR → RGB）───
rgb = img[:, :, ::-1]  # NumPy 切片反转
# 或
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

## 六、图像拼接

```python
# ─── 水平拼接 ───
# hstack 要求所有图像高度相同
result = np.hstack([img1, img2, img3])

# ─── 垂直拼接 ───
# vstack 要求所有图像宽度相同
result = np.vstack([img1, img2])

# ─── 智能拼接（尺寸不同时）───
def hconcat_resize(img_list, interpolation=cv2.INTER_LINEAR):
    h_min = min(img.shape[0] for img in img_list)
    im_list_resize = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min),
                                 interpolation=interpolation) for img in img_list]
    return cv2.hconcat(im_list_resize)

def vconcat_resize(img_list, interpolation=cv2.INTER_LINEAR):
    w_min = min(img.shape[1] for img in img_list)
    im_list_resize = [cv2.resize(img, (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                 interpolation=interpolation) for img in img_list]
    return cv2.vconcat(im_list_resize)

# ─── 网格拼接 ───
def grid_concat(img_list, cols=3):
    rows = (len(img_list) + cols - 1) // cols
    # 补齐
    while len(img_list) < rows * cols:
        img_list.append(np.zeros_like(img_list[0]))
    # 分行拼接
    row_imgs = [hconcat_resize(img_list[i:i+cols]) for i in range(0, len(img_list), cols)]
    return vconcat_resize(row_imgs)
```

## 七、图像填充

```python
# ─── 边界填充 ───
# cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
padded = cv2.copyMakeBorder(img,
    top=20, bottom=20, left=20, right=20,
    borderType=cv2.BORDER_CONSTANT,
    value=[255, 0, 0])  # 蓝色边框

# 边界类型
border_types = {
    cv2.BORDER_CONSTANT:    "固定值填充",
    cv2.BORDER_REPLICATE:   "复制边缘像素",
    cv2.BORDER_REFLECT:     "镜像反射 fedcba|abcdef|fedcba",
    cv2.BORDER_WRAP:        "环绕 cdefgh|abcdefgh|abcdefg",
    cv2.BORDER_REFLECT_101: "镜像（不含边缘）fedcb|abcdef|edcba"
}

# ─── NumPy 填充 ───
padded_np = np.pad(img, ((10, 10), (10, 10), (0, 0)),
                   mode='constant', constant_values=0)
```

## 八、数据类型转换

```python
# uint8 → float32（用于计算）
img_f = img.astype(np.float32)
img_f_norm = img_f / 255.0  # 归一化到 [0, 1]

# float32 → uint8（用于显示/保存）
# 必须先 clip 到 [0, 255]，再转换
img_u8 = np.clip(img_f, 0, 255).astype(np.uint8)

# 归一化
img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
```

## 九、性能技巧

```python
import time

img = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)

# ─── 慢：逐像素遍历 ───
start = time.time()
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        img[y, x] = [255 - img[y, x, 0],
                     255 - img[y, x, 1],
                     255 - img[y, x, 2]]
print(f"逐像素: {time.time() - start:.3f}s")

# ─── 快：NumPy 向量化 ───
start = time.time()
img_inv = 255 - img
print(f"向量化: {time.time() - start:.3f}s")

# ─── 最快：OpenCV 内置函数 ───
start = time.time()
img_inv = cv2.bitwise_not(img)
print(f"OpenCV: {time.time() - start:.3f}s")
```

**性能排序**：OpenCV 内置 > NumPy 向量化 > 逐像素循环

## 小结

| 操作 | 方法 |
|------|------|
| 像素访问 | `img[y, x]` 或 `img.item(y, x, c)` |
| ROI | `img[y1:y2, x1:x2]` |
| 通道操作 | `cv2.split()` / `cv2.merge()` |
| 拼接 | `np.hstack()` / `np.vstack()` |
| 类型转换 | `.astype()` + `np.clip()` |
| 性能优化 | 优先使用向量化操作 |

---

[下一节：色彩空间转换 →](03-色彩空间转换.md)
