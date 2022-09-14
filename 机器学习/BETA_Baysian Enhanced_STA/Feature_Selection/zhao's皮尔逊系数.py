# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 15:14
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : zhao's皮尔逊系数.py
# @Software: PyCharm


import numpy as np
import pandas as pd

df1 = pd.read_csv("..\\Benchmark\\timing1500x14.csv")
df2 = pd.read_csv("..\\Benchmark\\timing3700x14.csv")
df3 = pd.read_csv("..\\Benchmark\\timing9500x14.csv")
df4 = pd.read_csv("..\\Benchmark\\timing20000x14.csv")
df5 = pd.read_csv("..\\Benchmark\\timing50000x14.csv")
df6 = pd.read_csv("..\\Benchmark\\timing100000x14.csv")
df_data1 = np.array(df1.values[:, 1:])
df_data2 = np.array(df2.values[:1500, 1:])
df_data3 = np.array(df3.values[:1500, 1:])
df_data4 = np.array(df4.values[:1500, 1:])
df_data5 = np.array(df5.values[:1500, 1:])
df_data6 = np.array(df6.values[:1500, 1:])

### Calculate the pearson correleation .
# calc demo
np_data = df_data6
x = np_data[:,0]  #第一列
y = np_data[:,12]  #第二列

pccs = np.corrcoef( x , y )   #皮尔逊相关系数

print(pccs)    #对角线——变量与自身的相关系数为1


#======================================================
# r_xy = []
# corr_coeffi = []
# for i in range(0,np_data.shape[1] - 1): #按列
#     x = np_data[:, i]
#     for j in range(1,np_data.shape[1]):
#         if i == j:
#             continue
#         tmp = np_data[:,j]
#         pccs = np.corrcoef(x,tmp)   #皮尔逊相关系数
#         corr_coeffi.append(pccs[0][1])
#     r_xy.append( sum(np.array(corr_coeffi)) )
#     corr_coeffi = []
# # print(sum( np.array(corr_coeffi ) ))
# print(r_xy)
# #查找最大数 Rx
# Rx = r_xy[0]
# for i in r_xy:
#     if i > Rx:
#         Rx = i
# print("--------------------------------------------------")
# print("皮尔逊相关系数计算完成，计算结果如下！")
# print("Corner{}.".format(r_xy.index(Rx) +1  ),end='\t')
# print("Rx is {}.".format(Rx))







