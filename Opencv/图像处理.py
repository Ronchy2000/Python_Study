# -*- coding: utf-8 -*-
# @Time    : 2022/3/1 12:40
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 图像处理.py
# @Software: PyCharm
import numpy as np
import cv2 as cv

img1 = cv.imread('./袁艺-DSP分类.png')
cv.imshow("image",img1)



cv.waitKey(0)
cv.destroyAllWindows()