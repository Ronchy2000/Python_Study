# -*-coding:utf-8 -*-
# -on my iMac-

# @File      : 向量化的好处.py
# @Time      : 2022/2/3 下午6:57
# @Author    : Ronchy
import numpy as np
import time
#两个1000000维的随机向量用于矩阵运算
v1 = np.random.rand(1000000)
v2 = np.random.rand(1000000)
v = 0

tic = time.time()
for i in range(1000000):
    v+=v1[i]*v2[i]
toc = time.time()
print("非向量化的计算时间：",str((toc-tic)*1000)+"ms\n")

#>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<
tic2 = time.time()
v = np.dot(v1,v2)
toc2 = time.time()
print("向量化的计算时间",str((toc2-tic2)*1000)+"ms\n")

'''
运行结果：
非向量化的计算时间： 286.19980812072754ms
向量化的计算时间 1.178741455078125ms
'''