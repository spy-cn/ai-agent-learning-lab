# 05 - OpenCV.js 与 Web 应用

## 一、OpenCV.js 概述

OpenCV.js 是 OpenCV 的 WebAssembly 版本，可在浏览器中运行。

```
特点：
✓ 无需服务器，完全在客户端运行
✓ 使用 WebAssembly，性能接近原生
✓ JavaScript API，与 Python/C++ 类似
✓ 支持 SIMD 加速
```

## 二、在 HTML 中加载

```html
<!DOCTYPE html>
<html>
<head>
    <title>OpenCV.js Demo</title>
</head>
<body>
    <h1>OpenCV.js Demo</h1>
    <input type="file" id="fileInput" accept="image/*">
    <canvas id="canvas"></canvas>

    <!-- 加载 OpenCV.js -->
    <script async src="https://docs.opencv.org/4.x/opencv.js"
            onload="onOpenCvReady()" type="text/javascript"></script>

    <script>
        function onOpenCvReady() {
            cv.onRuntimeInitialized = () => {
                console.log("OpenCV.js is ready");
                document.getElementById('fileInput').disabled = false;
            };
        }

        document.getElementById('fileInput').addEventListener('change', (e) => {
            const file = e.target.files[0];
            const img = new Image();
            img.onload = () => {
                processImage(img);
            };
            img.src = URL.createObjectURL(file);
        });

        function processImage(img) {
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');

            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            // OpenCV.js 处理
            const src = cv.imread(canvas);
            const dst = new cv.Mat();

            // 转灰度
            cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
            // Canny 边缘
            cv.Canny(dst, dst, 50, 100);

            cv.imshow(canvas, dst);

            src.delete();
            dst.delete();
        }
    </script>
</body>
</html>
```

## 三、JavaScript API 与 Python 对照

| Python | JavaScript |
|--------|-----------|
| `cv2.imread()` | `cv.imread(canvas)` |
| `cv2.imshow()` | `cv.imshow(canvas, mat)` |
| `cv2.cvtColor()` | `cv.cvtColor()` |
| `cv2.Mat()` | `cv.Mat()` |
| `img.shape` | `mat.rows`, `mat.cols()` |
| `del img` | `mat.delete()` |

## 四、内存管理

```javascript
// OpenCV.js 需要手动释放内存！
function processImage(src) {
    let dst = new cv.Mat();
    cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);

    // 使用 dst...

    // 用完后必须释放
    dst.delete();
}

// 批量释放
function safeProcess() {
    let mat1 = new cv.Mat();
    let mat2 = new cv.Mat();
    let mat3 = new cv.Mat();

    try {
        // 处理...
    } finally {
        mat1.delete();
        mat2.delete();
        mat3.delete();
    }
}
```

## 五、实时摄像头处理

```html
<script>
async function startCamera() {
    const video = document.getElementById('video');
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.play();

    processVideo();
}

function processVideo() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');

    const src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    const dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);

    const cap = new cv.VideoCapture(video);

    function process() {
        cap.read(src);
        cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
        cv.Canny(dst, dst, 50, 100);
        cv.imshow(canvas, dst);

        requestAnimationFrame(process);
    }

    process();
}
</script>
```

## 六、Python 后端 + 前端 OpenCV

对于复杂场景，可以使用 Python 后端处理，前端展示。

```python
# Flask 后端
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # 编码为 JPEG
        ret, buffer = cv2.imencode('.jpg', edges_colored)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

```html
<!-- index.html -->
<img src="{{ url_for('video_feed') }}">
```

## 七、WebAssembly 性能优化

```javascript
// 1. 使用 SIMD 版本（更快）
// <script src="opencv-simd.js">

// 2. 使用 Web Worker（不阻塞 UI）
const worker = new Worker('opencv-worker.js');

worker.postMessage({ image: canvasData });
worker.onmessage = (e) => {
    const result = e.data.result;
    // 更新 UI
};

// opencv-worker.js
self.onmessage = (e) => {
    // 在 Worker 中处理
    const src = cv.imread(e.data.canvas);
    // ...
    self.postMessage({ result: resultCanvas });
};
```

## 八、部署建议

```
方案对比：

1. 纯前端（OpenCV.js）
   ✓ 无服务器成本
   ✓ 隐私好（数据不上传）
   ✗ 首次加载慢（OpenCV.js 约 8-10MB）
   ✗ 性能不如原生

2. Python 后端 + 前端
   ✓ 性能好
   ✓ 可用 GPU
   ✗ 服务器成本
   ✗ 网络延迟

3. 混合方案
   ✓ 前端做简单处理（裁剪、滤镜）
   ✓ 后端做复杂处理（AI 推理）
```

## 小结

| 方案 | 适用 |
|------|------|
| OpenCV.js | 轻量级、隐私敏感 |
| Flask + OpenCV | 服务端处理 |
| Web Worker | 不阻塞 UI |
| SIMD | 性能优化 |

---

[→ 进入实战篇](../09-实战篇/01-文档扫描仪.md)
