# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 10:53
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : hello Plot.py
# @Software: PyCharm
#基础绘图操作

import numpy as np
import cv2 as cv
#创建黑色的图像
img = np.zeros((512,512,3),np.uint8)
#绘制一条厚度为5的蓝色对角线
cv.line(img,(0,0),(511,511),(255,0,0),5)
cv.line(img,(0,511),(511,0),(0,255,0),5)
#rectangle
cv.rectangle(img,(384,0),(510,128),(0,0,255),3)
#circle
cv.circle(img,(447,63), 63, (0,0,255), -1)
#font
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500),font,1,(255,255,255),2,cv.LINE_AA)

cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()