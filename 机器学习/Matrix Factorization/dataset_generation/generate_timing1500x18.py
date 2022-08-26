# -*- coding: utf-8 -*-
# @Time    : 2022/8/12 20:32
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : timing1500x18.py
# @Software: PyCharm
import pandas as pd
import numpy as np
'''
数据生成1000-1500之间;
'''
# df = pd.read_csv('timing1500x18.csv')

path = (np.random.rand(1500)*(157-93)+92).reshape(1500,1)
corner = (np.random.rand(18)*(91-104)+93).reshape(1,18)
print(path.shape)
print(corner.shape)

output = (np.matmul(path,corner))/10
print(output)

col_name = [('Corner' + str(i)) for i in range(1,19)]
df = pd.DataFrame(output,columns=col_name)
df.to_csv("timing1500x18.csv")




