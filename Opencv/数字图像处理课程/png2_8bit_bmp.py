# -*- coding: utf-8 -*-
# @Time    : 2022/5/8 10:08
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : png2_8bit_bmp.py
# @Software: PyCharm
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = 'flower256.bmp'
# path1 = 'Lenna.png'
img = cv.imread(path)

#获取图片的宽和高
width,height = img.shape[:2][::-1]
img_gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)

#图像FFT
dft = cv.dft(np.float32(img_gray), flags = cv.DFT_COMPLEX_OUTPUT)
fshift = np.fft.fftshift(dft)

#截止频率
i = 10

#设置低通滤波器
rows, cols = img_gray.shape
crow,ccol = int(rows/2), int(cols/2) #中心位置
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-i :crow+i , ccol-i :ccol+i ] = 1

#掩膜图像和频谱图像乘积
f = fshift * mask
print(f.shape, fshift.shape, mask.shape)
#傅里叶逆变换
ishift = np.fft.ifftshift(f)
iimg = cv.idft(ishift)
res = cv.magnitude(iimg[:,:,0], iimg[:,:,1])

plt.subplot(122), plt.imshow(res, 'gray'), plt.title('Result Image')
plt.axis('off')
plt.show()

###显示图像频谱图
# result = 20 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
# plt.subplot(122),plt.imshow(result,cmap='gray')
# plt.title('original'), plt.axis('off')
# plt.show()

# cv.imshow('pinpu',result)
# cv.imshow('img',img_gray)
# cv.imwrite("flower_gray.bmp",img_gray)
cv.waitKey(0)