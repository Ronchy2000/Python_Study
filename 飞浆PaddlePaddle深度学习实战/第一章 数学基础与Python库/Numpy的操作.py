# -*-coding:utf-8 -*-
# -on my iMac-

# @File      : Numpy的操作.py
# @Time      : 2022/2/3 下午6:43
# @Author    : Ronchy
'''
基本操作
'''
import numpy as np
a =[1,2,3,4]
b = np.array(a)
print(type(b)) #<class 'numpy.ndarray'>

print(b.shape) #(4,) array的大小
print(b.argmax()) #array中最大值的索引  print(b.argmin())
print(b.max()) #array中的最大值
print(b.mean())#array中的平均值

#>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<#
x = np.array(np.arange(12).reshape((3,4)))
print("x=\n",x)
t = x.transpose() #矩阵的转置
print('t=\n',t)

