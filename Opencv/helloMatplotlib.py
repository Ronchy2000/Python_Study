# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 10:30
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : helloMatplotlib.py
# @Software: PyCharm
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('.\helloCV_OIP.jpg')
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()
plt.imshow(img,cmap='gray',interpolation='bicubic')
#plt.xticks([]),plt.yticks([]) #隐藏x,y轴上的刻度值
plt.show()
'''
    Matplotlib 函数绘制出的图像是RGB模式（无法显示正确彩色图像）
    OpenCv加载彩色图像处于BGR模式（原彩）
'''