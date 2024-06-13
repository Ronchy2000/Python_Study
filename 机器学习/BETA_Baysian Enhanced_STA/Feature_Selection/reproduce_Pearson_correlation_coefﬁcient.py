# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 15:14
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : reproduce_Pearson_correlation_coefﬁcient.py
# @Software: PyCharm
'''
皮尔逊 - 结果
data2   Corner2
data3   Corner14
data4   Corner1
data5   Corner5
data6   Corner4
'''
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

## iterative
# data2
r_xy = []
corr_coeffi = []
for i in range(0,df_data2.shape[1]): #按列
    x = df_data2[:, i]
    for j in range(1,df_data2.shape[1]):
        if i == j:
            continue
        tmp = df_data2[:,j]
        pccs = np.corrcoef(x,tmp)   #计算—皮尔逊相关系数
        corr_coeffi.append(pccs[0][1])
    r_xy.append( sum(np.array(corr_coeffi)) )
    corr_coeffi = []
# print(sum( np.array(corr_coeffi ) ))
print(len(r_xy),r_xy)
#查找最大数 Rx
Rx = r_xy[0]
for i in r_xy:
    if i > Rx:
        Rx = i
print("--------------------------------------------------")
print("皮尔逊相关系数计算完成，计算结果如下！")
print("Corner{}.".format(r_xy.index(Rx) +1  ),end='\t')
print("Rx is {}.".format(Rx))
#===================================================================================###
# data3
r_xy = []
corr_coeffi = []
for i in range(0,df_data3.shape[1]): #按列
    x = df_data3[:, i]
    for j in range(1,df_data3.shape[1]):
        if i == j:
            continue
        tmp = df_data3[:,j]
        pccs = np.corrcoef(x,tmp)   #计算—皮尔逊相关系数
        corr_coeffi.append(pccs[0][1])
    r_xy.append( sum(np.array(corr_coeffi)) )
    corr_coeffi = []
# print(sum( np.array(corr_coeffi ) ))
print(len(r_xy),r_xy)
#查找最大数 Rx
Rx = r_xy[0]
for i in r_xy:
    if i > Rx:
        Rx = i
print("--------------------------------------------------")
print("皮尔逊相关系数计算完成，计算结果如下！")
print("Corner{}.".format(r_xy.index(Rx) +1  ),end='\t')
print("Rx is {}.".format(Rx))
#===================================================================================###
# data4
r_xy = []
corr_coeffi = []
for i in range(0,df_data4.shape[1]): #按列
    x = df_data4[:, i]
    for j in range(1,df_data4.shape[1]):
        if i == j:
            continue
        tmp = df_data4[:,j]
        pccs = np.corrcoef(x,tmp)   #计算—皮尔逊相关系数
        corr_coeffi.append(pccs[0][1])
    r_xy.append( sum(np.array(corr_coeffi)) )
    corr_coeffi = []
# print(sum( np.array(corr_coeffi ) ))
print(len(r_xy),r_xy)
#查找最大数 Rx
Rx = r_xy[0]
for i in r_xy:
    if i > Rx:
        Rx = i
print("--------------------------------------------------")
print("皮尔逊相关系数计算完成，计算结果如下！")
print("Corner{}.".format(r_xy.index(Rx) +1  ),end='\t')
print("Rx is {}.".format(Rx))
#===================================================================================###
# data5
r_xy = []
corr_coeffi = []
for i in range(0,df_data5.shape[1]): #按列
    x = df_data5[:, i]
    for j in range(1,df_data5.shape[1]):
        if i == j:
            continue
        tmp = df_data5[:,j]
        pccs = np.corrcoef(x,tmp)   #计算—皮尔逊相关系数
        corr_coeffi.append(pccs[0][1])
    r_xy.append( sum(np.array(corr_coeffi)) )
    corr_coeffi = []
# print(sum( np.array(corr_coeffi ) ))
print(len(r_xy),r_xy)
#查找最大数 Rx
Rx = r_xy[0]
for i in r_xy:
    if i > Rx:
        Rx = i
print("--------------------------------------------------")
print("皮尔逊相关系数计算完成，计算结果如下！")
print("Corner{}.".format(r_xy.index(Rx) +1  ),end='\t')
print("Rx is {}.".format(Rx))
#===================================================================================###
# data6
r_xy = []
corr_coeffi = []
for i in range(0,df_data6.shape[1]): #按列
    x = df_data6[:, i]
    for j in range(1,df_data6.shape[1]):
        if i == j:
            continue
        tmp = df_data6[:,j]
        pccs = np.corrcoef(x,tmp)   #计算—皮尔逊相关系数
        corr_coeffi.append(pccs[0][1])
    r_xy.append( sum(np.array(corr_coeffi)) )
    corr_coeffi = []
# print(sum( np.array(corr_coeffi ) ))
print(len(r_xy),r_xy)
#查找最大数 Rx
Rx = r_xy[0]
for i in r_xy:
    if i > Rx:
        Rx = i
print("--------------------------------------------------")
print("皮尔逊相关系数计算完成，计算结果如下！")
print("Corner{}.".format(r_xy.index(Rx) + 1),end='\t')
print("Rx is {}.".format(Rx))

