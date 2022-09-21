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
df1 = pd.read_csv("b17_VTL1_covariance.csv")


df_data1 = np.array(df1.values[:, 1:])
mean = np.tile(77.2, (df_data1.shape[0],1))

print( prob_not_violation(0, mean, np.sqrt( df_data1) ))




