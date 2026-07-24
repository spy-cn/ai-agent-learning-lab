# OpenCV 速查表 (Cheat Sheet)

## 一、图像读写

```python
cv2.imread(path, flag)           # 读取: 0=灰度, 1=彩色, -1=含alpha
cv2.imwrite(path, img)           # 保存
cv2.imshow(winname, img)         # 显示
cv2.waitKey(delay)               # 等待按键 (0=无限)
cv2.destroyAllWindows()          # 关闭所有窗口
```

## 二、基本操作

```python
img.shape    # (H, W, C)
img[y, x]    # 像素值
img[y1:y2, x1:x2]  # ROI
cv2.split(img)      # 分离通道
cv2.merge([b,g,r])  # 合并通道
cv2.copyMakeBorder(img, t, b, l, r, type, value)  # 边界填充
```

## 三、色彩空间

```python
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 灰度
cv2.cvtColor(img, cv2.COLOR_BGR2HSV)     # HSV
cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # RGB
cv2.cvtColor(img, cv2.COLOR_BGR2LAB)     # LAB
```

## 四、绘图

```python
cv2.line(img, pt1, pt2, color, thickness)         # 直线
cv2.rectangle(img, pt1, pt2, color, thickness)     # 矩形 (-1=填充)
cv2.circle(img, center, radius, color, thickness)  # 圆
cv2.ellipse(img, center, axes, angle, start, end, color, thick)  # 椭圆
cv2.polylines(img, [pts], isClosed, color, thick)  # 多边形
cv2.fillPoly(img, [pts], color)                    # 填充多边形
cv2.putText(img, text, org, font, scale, color, thick)  # 文字
```

## 五、几何变换

```python
cv2.resize(img, dsize, fx, fy, interpolation)      # 缩放
cv2.flip(img, flipCode)                             # 翻转 (1=H, 0=V)
cv2.warpAffine(img, M, dsize)                      # 仿射变换
cv2.warpPerspective(img, M, dsize)                  # 透视变换
cv2.getRotationMatrix2D(center, angle, scale)       # 旋转矩阵
cv2.getAffineTransform(pts1, pts2)                  # 仿射矩阵
cv2.getPerspectiveTransform(pts1, pts2)             # 透视矩阵
```

## 六、滤波

```python
cv2.blur(img, ksize)                       # 均值滤波
cv2.GaussianBlur(img, ksize, sigmaX)       # 高斯滤波
cv2.medianBlur(img, ksize)                 # 中值滤波
cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)  # 双边滤波
cv2.filter2D(img, ddepth, kernel)          # 自定义卷积
```

## 七、形态学

```python
cv2.erode(img, kernel, iterations)           # 腐蚀
cv2.dilate(img, kernel, iterations)          # 膨胀
cv2.morphologyEx(img, op, kernel)            # 高级形态学
# op: MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT
cv2.getStructuringElement(shape, ksize)      # 结构元素
```

## 八、阈值

```python
cv2.threshold(img, thresh, maxval, type)             # 全局阈值
cv2.adaptiveThreshold(img, maxval, method, type, blockSize, C)  # 自适应
# type: THRESH_BINARY, THRESH_BINARY_INV, THRESH_OTSU, THRESH_TRIANGLE
```

## 九、边缘检测

```python
cv2.Canny(img, threshold1, threshold2)     # Canny
cv2.Sobel(img, ddepth, dx, dy, ksize)      # Sobel
cv2.Laplacian(img, ddepth, ksize)           # Laplacian
cv2.Scharr(img, ddepth, dx, dy)            # Scharr
```

## 十、轮廓

```python
cv2.findContours(img, mode, method)        # 查找轮廓
cv2.drawContours(img, contours, idx, color, thickness)  # 绘制
cv2.contourArea(cnt)                        # 面积
cv2.arcLength(cnt, closed)                  # 周长
cv2.boundingRect(cnt)                       # 外接矩形
cv2.minAreaRect(cnt)                        # 最小外接矩形
cv2.minEnclosingCircle(cnt)                 # 最小外接圆
cv2.convexHull(cnt)                         # 凸包
cv2.approxPolyDP(cnt, epsilon, closed)      # 多边形近似
cv2.moments(cnt)                            # 矩（质心等）
cv2.matchShapes(cnt1, cnt2, method, param)  # 形状匹配
```

## 十一、特征检测

```python
# 检测器
cv2.cornerHarris(img, blockSize, ksize, k)
cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance)
cv2.FastFeatureDetector_create()
cv2.SIFT_create()
cv2.ORB_create()
cv2.AKAZE_create()

# 检测+描述
kp, des = detector.detectAndCompute(img, mask)

# 绘制
cv2.drawKeypoints(img, keypoints, outImg, color, flags)

# 匹配
cv2.BFMatcher(normType, crossCheck)
cv2.FlannBasedMatcher(index_params, search_params)
matcher.match(des1, des2)
matcher.knnMatch(des1, des2, k)

# 单应矩阵
cv2.findHomography(src, dst, method, threshold)
```

## 十二、霍夫变换

```python
cv2.HoughLines(img, rho, theta, threshold)           # 标准直线
cv2.HoughLinesP(img, rho, theta, threshold, minLineLen, maxLineGap)  # 概率直线
cv2.HoughCircles(img, method, dp, minDist, param1, param2, minRadius, maxRadius)  # 圆
```

## 十三、视频

```python
cap = cv2.VideoCapture(source)    # 打开
cap.read()                        # 读帧 → (ret, frame)
cap.get(propId)                   # 获取属性
cap.set(propId, value)            # 设置属性
cap.release()                     # 释放

out = cv2.VideoWriter(path, fourcc, fps, size)  # 创建写入器
out.write(frame)                  # 写帧
out.release()                     # 释放
```

## 十四、分割

```python
cv2.watershed(img, markers)                    # 分水岭
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode)  # GrabCut
cv2.pyrMeanShiftFiltering(img, sp, sr)         # Mean-Shift
cv2.distanceTransform(img, distanceType, maskSize)  # 距离变换
cv2.connectedComponents(img)                    # 连通区域
```

## 十五、直方图

```python
cv2.calcHist(images, channels, mask, histSize, ranges)
cv2.equalizeHist(img)                    # 均衡化
cv2.createCLAHE(clipLimit, tileGridSize)  # CLAHE
cv2.compareHist(hist1, hist2, method)     # 比较
cv2.calcBackProject(images, channels, hist, ranges, scale)  # 反向投影
```

## 十六、DNN

```python
cv2.dnn.readNet(model, config)           # 加载模型
cv2.dnn.readNetFromONNX(model)           # ONNX
cv2.dnn.readNetFromCaffe(proto, model)   # Caffe
cv2.dnn.readNetFromDarknet(cfg, weights) # Darknet
cv2.dnn.blobFromImage(img, scale, size, mean, swapRB)  # 预处理
net.setInput(blob)                       # 设置输入
net.forward()                            # 推理
net.forward(output_names)                # 多输出
cv2.dnn.NMSBoxes(boxes, scores, conf_thres, nms_thres)  # NMS
```

## 十七、常用属性

```python
# VideoCapture 属性
cv2.CAP_PROP_FRAME_WIDTH
cv2.CAP_PROP_FRAME_HEIGHT
cv2.CAP_PROP_FPS
cv2.CAP_PROP_FRAME_COUNT
cv2.CAP_PROP_POS_FRAMES

# imread 标志
cv2.IMREAD_COLOR        # 1
cv2.IMREAD_GRAYSCALE    # 0
cv2.IMREAD_UNCHANGED    # -1

# 插值方法
cv2.INTER_NEAREST       # 最近邻
cv2.INTER_LINEAR        # 双线性（默认）
cv2.INTER_CUBIC         # 三次
cv2.INTER_AREA          # 区域（缩小推荐）
cv2.INTER_LANCZOS4      # Lanczos（高质量）
```

---

[下一节：算法选择指南 →](02-算法选择指南.md)
