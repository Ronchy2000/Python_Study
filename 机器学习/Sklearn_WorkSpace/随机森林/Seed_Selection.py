# -*- coding: utf-8 -*-
# @Time    : 2022/7/21 12:32
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Seed_Selection.py
# @Software: PyCharm
# This is the Code to found the seed.

import numpy as np
import pandas as pd

df = pd.read_csv("mydata2_corner1-corner14.csv")
df = pd.read_csv("timing1500x14_delete_first_col.csv")

print(df.shape)    #csv的shape
np_data = np.empty(df.shape,dtype=float) #create a new empty matrix


header = list(df.columns.values)
feature_name = header[:]
print("feature_name:\n\n",feature_name)

for i in range(len(feature_name)):
    np_data[:,i] =  ( df[ feature_name[i] ].values.tolist() )
    # print(np_data[:,i])
print("-----------------------------------")
print(np_data)
print("数据导入numpy！")


### Calculate the pearson correleation .

#calc demo
# x = np_data[:,0]  #第一列
# y = np_data[:,1]  #第二列
#
# pccs = np.corrcoef( x , y )   #皮尔逊相关系数
#
# print(pccs)    #对角线——变量与自身的相关系数为1

r_xy = []
corr_coeffi = []
for i in range(0,np_data.shape[1] - 1): #按列
    x = np_data[:, i]
    for j in range(1,np_data.shape[1]):
        if i == j:
            continue
        tmp = np_data[:,j]
        pccs = np.corrcoef(x,tmp)   #皮尔逊相关系数
        corr_coeffi.append(pccs[0][1])
    r_xy.append( sum(np.array(corr_coeffi)) )
    corr_coeffi = []
# print(sum( np.array(corr_coeffi ) ))
print(r_xy)
#查找最大数 Rx
Rx = r_xy[0]
for i in r_xy:
    if i > Rx:
        Rx = i
print("--------------------------------------------------")
print("皮尔逊相关系数计算完成，计算结果如下！")
print("Corner{}.".format(r_xy.index(Rx) +1  ),end='\t')
print("Rx is {}.".format(Rx))

'''
结果保存
--------------------------------------------------
皮尔逊相关系数计算完成，计算结果如下！
Corner1.	Rx is 0.19247381984258294.
'''


