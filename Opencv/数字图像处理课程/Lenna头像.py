# -*- coding: utf-8 -*-
# @Time    : 2021/8/31 22:47
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Lenna头像.py
# @Software: PyCharm
import cv2 as cv
import numpy as np
'''
    cv的方法 By Ronchy
'''
img = cv.imread('Lenna.png')
cv.imshow("image",img)
h,w,c = img.shape
print("彩色图像:",h,w,c) #高 宽 通道数


img_grey = cv.imread('Lenna.png',0)#cv_grey
#cv.namedWindow('image', cv.WINDOW_NORMAL)#可以调整窗口大小
cv.imshow("image_grey",img_grey)
h,w = img_grey.shape
print("灰度图像:",h,w) #高 宽 通道数

img_binary = cv.cvtColor(img,cv.COLOR_RGB2GRAY) #灰度化处理为单通道
ret, binary = cv.threshold(img_binary,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imshow("img_binary",binary)#显示二值化图像

cv.waitKey(0)
cv.destroyAllWindows()#简单的销毁我们创建的所有窗口。如果你想销毁任意指定窗口，应该使用函数

