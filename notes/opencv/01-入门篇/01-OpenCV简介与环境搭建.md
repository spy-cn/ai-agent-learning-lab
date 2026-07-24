# 01 - OpenCV 简介与环境搭建

## 一、OpenCV 是什么？

**OpenCV**（Open Source Computer Vision Library）是一个开源的计算机视觉和机器学习软件库。它由 Intel 于 1999 年发起，现在由非营利组织 OpenCV.org 维护。

### 核心特点

| 特点 | 说明 |
|------|------|
| **开源免费** | BSD 许可证，商用无需开源 |
| **跨平台** | Windows / Linux / macOS / Android / iOS |
| **多语言** | C++ / Python / Java / MATLAB |
| **丰富的算法** | 超过 2500+ 个优化算法 |
| **活跃社区** | GitHub Star 70k+，下载量超千万 |

### 应用领域

```
┌─────────────────────────────────────────────────┐
│                  OpenCV 应用领域                  │
├──────────────┬──────────────┬───────────────────┤
│  医学影像     │  自动驾驶     │  工业检测          │
│  安防监控     │  人脸识别     │  增强现实(AR)      │
│  文档扫描     │  机器人视觉   │  运动分析          │
│  全景拼接     │  图像搜索引擎 │  手势识别          │
└──────────────┴──────────────┴───────────────────┘
```

## 二、模块架构

OpenCV 由多个模块组成：

```
OpenCV
├── core       → 核心数据结构（Mat、Point、Rect 等）
├── imgproc    → 图像处理（滤波、几何变换、特征）
├── highgui    → GUI 功能（窗口、滑动条、鼠标）
├── video      → 视频分析（光流、跟踪、背景建模）
├── calib3d    → 相机标定与 3D 重建
├── features2d → 特征检测与描述（SIFT、ORB）
├── objdetect  → 目标检测（Haar 级联、HOG）
├── dnn        → 深度学习推理（YOLO、SSD 等）
├── ml         → 传统机器学习（SVM、KNN、决策树）
├── flann      → 快速近似最近邻搜索
├── photo      → 计算摄影（去噪、HDR）
├── stitching  → 图像拼接
├── xfeatures2d → 额外特征（SIFT 扩展）
├── ximgproc   → 扩展图像处理
├── aruco      → ArUco 标记检测
└── tracking   → 扩展跟踪算法
```

> **注意**：标有 `x` 前缀的模块（如 `xfeatures2d`）属于 `opencv_contrib`，需要安装 `opencv-contrib-python` 才能使用。

## 三、Python 环境安装

### 方式一：pip 安装（推荐）

```bash
# 基础版（仅包含主模块）
pip install opencv-python

# 完整版（包含 contrib 扩展模块，推荐）
pip install opencv-contrib-python

# 验证安装
python -c "import cv2; print(cv2.__version__)"
```

### 方式二：conda 安装

```bash
conda install -c conda-forge opencv
```

### 方式三：从源码编译（需要 C++ 扩展模块时）

```bash
# 适用于需要 CUDA 加速或自定义编译选项的场景
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# 使用 CMake 配置和编译
mkdir build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ..
make -j8
make install
```

## 四、推荐 IDE 配置

### VS Code 配置

```json
// settings.json 推荐配置
{
    "python.languageServer": "Pylance",
    "python.analysis.typeCheckingMode": "basic"
}
```

安装扩展：
- Python（Microsoft）
- Pylance（Microsoft）
- OpenCV Snippets（可选）

### PyCharm 配置

PyCharm 对 OpenCV 的自动补全支持良好，安装 `opencv-python` 后即可使用。

## 五、包的区别说明

| 包名 | 说明 |
|------|------|
| `opencv-python` | 主模块，适合大多数用户 |
| `opencv-contrib-python` | 主模块 + 扩展模块，**推荐安装** |
| `opencv-python-headless` | 无 GUI 功能，适合服务器环境 |
| `opencv-contrib-python-headless` | 扩展 + 无 GUI |

> **重要**：不要同时安装多个 OpenCV 包，会导致冲突。使用前先卸载：
> ```bash
> pip uninstall opencv-python opencv-contrib-python
> pip install opencv-contrib-python
> ```

## 六、验证安装

```python
import cv2
import numpy as np

# 打印版本
print(f"OpenCV 版本: {cv2.__version__}")

# 检查构建信息
build_info = cv2.getBuildInformation()
print("CUDA 支持:", "CUDA YES" in build_info)
print("FFmpeg 支持:", "FFMPEG YES" in build_info)

# 创建测试图像
img = np.zeros((200, 400, 3), dtype=np.uint8)
cv2.putText(img, "OpenCV Installed!", (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 小结

| 内容 | 要点 |
|------|------|
| OpenCV 定位 | 开源计算机视觉库，2500+ 算法 |
| 核心模块 | core / imgproc / highgui / dnn |
| Python 安装 | `pip install opencv-contrib-python` |
| 版本选择 | headless 适合服务器，contrib 适合开发 |

---

[下一节：环境搭建详解 →](02-环境搭建与安装.md)
