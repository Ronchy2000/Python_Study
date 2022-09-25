# -*- coding: utf-8 -*-
# @Time    : 2022/9/20 21:10
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 计算概率估计.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from 分布概率 import prob_not_violation
##使用累积分布函数

df1 = pd.read_csv("../15nm实验/BETA_oneCorner/b17_15/b17_15nm_v1_prediction.csv")
df2 = pd.read_csv("../15nm实验/BETA_oneCorner/b17_15/b17_15nm_v1_covariance.csv")
df3 = pd.read_csv("../15nm实验/BETA_oneCorner/b17_15/b17_15nm_v1_real.csv")

#OK
df_data1 = np.array(df1.values[:, 1:])
df_data2 = np.array(df2.values[:, 1:])
df_data3 = np.array(df3.values[:, 1:])


###计算相对误差
# relative_error = (np.abs(df_data1 - df_data3))/df_data3
# print(relative_error)



np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# print( prob_not_violation(1200, df_data1, np.sqrt( df_data2) ) )

##Corner1
# for i in prob_not_violation(1200, df_data1, np.sqrt( df_data2) )[:,0]:
#     print('%.2f'%i,end='\n')

##Corner2
# for i in prob_not_violation(1200, df_data1, np.sqrt( df_data2) )[:,1]:
#     print('%.2f' % i, end='\n')

##Corner3
# for i in prob_not_violation(1200, df_data1, np.sqrt(df_data2))[:, 2]:
#     print('%.2f' % i, end='\n')

##Corner4
# for i in prob_not_violation(1200, df_data1, np.sqrt(df_data2))[:, 3]:
#     print('%.2f' % i, end='\n')


