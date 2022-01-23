# -*- coding: utf-8 -*-
# @Time    : 2021/9/1 19:25
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Lenna_numpy修改.py
# @Software: PyCharm
import numpy as np
import matplotlib
import cv2 as cv
img = cv.imread('Lenna.png')

'''访问单个像素'''
#px = img[0,0] #img是一个三维的列表,px是一维，包括B G R三个值
#print(px)
#print("Blue",px[2])
'''''''''''''''''''''''''' ''''''''''''''''''''''''''''''''
#访问RED
print(img.item(0,0,2))

img.itemset((0,0,2),100)
print(img.item(0,0,2))

shp = img.shape
print(shp)
'''方法二'''
#创建一个画布
grey_img = np.zeros((shp[0],shp[1],3),np.uint8)
binary_img = np.zeros((shp[0],shp[1],3),np.uint8)
####灰度图
for i in range(shp[0]):
    for j in range(shp[1]):
        grey_img[i,j] = 0.3*img[i,j,0] + 0.59*img[i,j,1] + 0.11*img[i,j,2]
###二值化
for i in range(shp[0]):
    for j in range(shp[1]):
        if grey_img[i,j,1] >127 :
            binary_img[i,j] = [255,255,255]
        else:
            binary_img[i,j] = [0,0,0]
'''        
for i in range(shp[0]):
    for j in range(shp[1]):
        #img.item(i,j,0) = img.item(i,j,0) * 0.114
        #img.item(i,j,1) = img.item(i, j, 1) * 0.578
        #img.item(i,j,2) = img.item(i, j, 2) * 0.299
        img.itemset((i, j, 0), img.item(i,j,0) * 0.3+img.item(i, j, 1) * 0.59 + img.item(i, j, 2) * 0.11) #Blue
        img.itemset((i, j, 1), img.item(i,j,0) * 0.3+img.item(i, j, 1) * 0.59 + img.item(i, j, 2) * 0.11) #Green
        img.itemset((i, j, 2), img.item(i,j,0) * 0.3+img.item(i, j, 1) * 0.59 + img.item(i, j, 2) * 0.11) #Red
'''
print('灰度属性',grey_img.shape)
cv.imshow("灰度图",grey_img)
print('二值图属性',grey_img.shape)
cv.imshow("二值化图",binary_img)
#cv.imshow("RGB_img",img)

cv.waitKey(0)
cv.destroyAllWindows()
