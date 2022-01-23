# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 11:04
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 图像基本操作.py
# @Software: PyCharm
import numpy as np
import cv2 as cv

img = cv.imread('.\helloCV_OIP.jpg')
# #访问图像属性
# print(img.shape)
# #>>> (250, 251, 3)  宽高 通道数3
# print(img.size)
# #>>> 188250 =  250*251*3  像素总数
# print(img.dtype )
# #>>> uint8  图像数据类型
# #访问像素值
# pix = img[100,100]
# print(pix)

# #用像素点创建图像
# img[100,100]= [255,255,255]
# print(img[100,100])
# cv.imshow("image",img)

#图像ROI：
# pic = img[100:150,100:150]
# img[0:50,0:50] = pic
# cv.imshow("image",img)

#图像拆分通道
#B G R  置为0
# #B
# img[:,:,0] = 0
# #G
# img[:,:,1] = 0
# #R
# img[:,:,2] = 0


##返回图像 BGR 三个矩阵
# b,g,r = cv.split(img)
# print("b",b)
# print("g:",g)
# print("r:",r)

lenna = 'E:\Developer\Python\Myworkshop\Python_Study\Opencv\数字图像处理课程\Lenna.png'


cv.imshow("image:",img)


cv.waitKey(0)
cv.destroyAllWindows()