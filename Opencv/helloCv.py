# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 9:52
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : helloCv.py
# @Software: PyCharm
import cv2 as cv
import numpy as np
# print(cv.__version__)
img = ".\helloCV_OIP.jpg"
img_org = cv.imread(img) #origin Image
img_cv = cv.imread(img,0) #灰度图

cv.imshow('imag_org',img_org)
cv.imshow('imag_cv',img_cv)

'''如果使用的是64位计算机，则必须 k = cv.waitKey(0) 按如下所示修改行： k = cv.waitKey(0) & 0xFF'''
key = cv.waitKey(0)&0xFF
if key == 27:          #push down 'ESC' to quit out
    cv.destroyAllWindows()
elif key == ord('s'):  #push 's' to save
    cv.imwrite('OIP灰度图.png',img_cv)
    cv.destroyAllWindows()

