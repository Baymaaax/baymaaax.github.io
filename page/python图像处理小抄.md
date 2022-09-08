###### [返回主页](../README.md)

------



# python图像处理

## 1. 图像读写

### 1.1 Matplotlib

```python
import matplotlib.pyplot as plt

# 读取
im=plt.imread("01.png") 
# type(im) -> numpy.ndarray
# 像素值被缩放至[0，1]
# im.shape -> (M, N) for grayscale images.
#			     -> (M, N, 3) for RGB images.
#          -> (M, N, 4) for RGBA images.

# 显示
plt.imshow(im)
plt.show()

# 保存
plt.savefig("02.png") # 在plt.show()之前保存，否则为空白

```

### 1.2 Pillow

```python
from PIL import Image

# 读取
im=Image.open("01.png")
# type(im) -> PIL.PngImagePlugin.PngImageFile

# 转化为矩阵
im_arr=np.asarray(im)
# or im_arr=np.array(im)
# type(im_arr) -> numpy.ndarray

# 矩阵转化为图像
im_=Image.fromarray(im_arr)
# type(im_) -> PIL.Image.Image

# 显示
im.show()

# 保存
im.save("02.png")
```

### 1.3 OpenCV

```python
import cv2

# 读取
im=cv2.imread("01.png",cv2.IMREAD_COLOR)
# type(im) -> numpy.ndarray
#					 -> NoneType 图片不存在时不会报错 
# 不同于其他方法，opencv读取出的图像为bgr，而非rgb
# 彩色图：cv2.IMREAD_COLOR
# 灰度图：cv2.IMREAD_GRAYSCALE

# 颜色通道转换
b,g,r = cv2.split(im)
im_rgb = cv2.merge([r,g,b])
# type(im_rgb) -> numpy.ndarray

# 显示
cv2.imshow('01', im)
cv2.waitKey(0)

# 保存
cv2.imwrite("02.png",im)
```

## 2. OpenCV 基础

### 2.1 绘制矩形框

```python
cv2.rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None)
"""
img:图像
pt1:矩形的一个顶点
pt2:矩形的另一个顶点
color:颜色（0,0,255） bgr
thickness:线条粗细，负值时，绘制实心矩形
lineType:线条绘制类型,8 (or0) - 8-connected line（8邻接)连接线。
              		4 -4-connected line(4邻接)连接线。
									CV_AA- antialiased 线条。
shift:坐标点的小数点位数
"""
```

 ### 2.2 添加文字

```python
putText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None)
"""
img:图像
text：文字
org：文本框左下角坐标
fontFace：字体
fontScale：字号
color：颜色
thickness：线条粗细
lineType：线条绘制类型
bottomLeftOrigin：为True时，文字上下颠倒
"""
```

