# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 19:41
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : png2dat.py
# @Software: PyCharm

import os
import sys
from PIL import Image
import numpy as np

Bytes = 4  # Every line have Bytes bytes
DataAddr = 512

file_name = r'E:\Developer\Python\Myworkshop\Python_Study\Opencv\数字图像处理课程\Lenna.png'

img = Image.open(file_name)
img = img.resize((56, 56))
r, g, b = img.split()
r_array = np.array(r).reshape(-1)
g_array = np.array(g).reshape(-1)
b_array = np.array(b).reshape(-1)
merge_array = np.concatenate((r_array, g_array, b_array))
print(merge_array.shape)
# img=np.array(r)
# data=img.reshape(-1)
# data=np.clip(data,0,1)
# print(data)
# print(data.shape,type(data))
file_name1 = "image_all.dat"
# print(file_name1)
file1 = open(file_name1, 'w')

num = 0
for m, j in enumerate(merge_array):
    file1.write('@')
    print(m, j)
    # print(num+=1)
    # print(hex(m)[2:].zfill(4))
    print(hex(m)[2:].zfill(4))
    file1.write(hex(m)[2:].zfill(4))
    file1.write('\n')
    file1.write(hex(j)[2:].zfill(8))
    file1.write('\n')
file1.close()
