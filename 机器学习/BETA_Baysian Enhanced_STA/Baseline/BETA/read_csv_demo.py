# -*- coding: utf-8 -*-
# @Time    : 2022/9/18 23:52
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : read_csv_demo.py
# @Software: PyCharm

import numpy as np
import pandas as pd

df1 = pd.read_csv("timing1500x14.csv")
df2 = pd.read_csv("timing1500x14.csv")
df3 = pd.read_csv("timing1500x14.csv")

df_data1 = np.array(df1.values[:, 1:]) #所有行  + 除第一列不要
df_data2 = np.array(df2.values[:, 1:])
df_data3 = np.array(df3.values[:, 1:])
print("df_data1:",df_data1)

