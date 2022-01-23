import numpy as np
import cv2

# 读取图像
image = cv2.imread('./test2.jpg')
# 转化为灰度图，进行边缘检测
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用Scharr算子进行边缘检测
gradX = cv2.Sobel(gray, ddepth = cv2.cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
# 从X中减去Y，得到包含高水平梯度和低竖直梯度的图像区域
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# 使用3*3的内核对梯度图进行平均模糊，有助于平滑梯度表征的图形中的高频噪声
blurred = cv2.blur(gradient, (3, 3))   # old_(9,9)
# 将模糊化后的图形进行二值化
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

# 构造一个长方形内核，内核的宽度大于长度，消除条形码中垂直条之间的缝隙
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
# 使用上面构造的内核进行形态学操作，消除竖杠间的缝隙，闭运算
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 首先进行4次腐蚀，然后进行4次膨胀，消除小斑点，增强条形码区域
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)

# 找到图像中的最大轮廓，如果我们正确完成了图像处理步骤，这里应该对应于条形码区域
(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

# 为最大轮廓确定最小边框
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))

# 显示检测到的条形码
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.imshow("Image", image)
cv2.waitKey(0)
