# -*- coding: utf-8 -*-
# @Time    : 2022/8/21 10:13
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : test.py
# @Software: PyCharm

import numpy as np
import pandas as pd

# np.random.seed(seed=7)
# state = np.random.get_state()
# users = np.random.permutation([1,2,3,4,5,6,7,8,9])
# print(users)

df = pd.read_csv("timing1500x14_flattern.csv")

# print(df.columns.values)
# print(df['row'])
# for i in df['row']:
#     print(i)

# df = pd.read_csv("timing1500x14_flattern.csv")
#取corner0
group = df.groupby(['col'])

#按组划分
df1 = group.get_group(0)
df2 = group.get_group(1)
out = pd.concat([df1,df2],axis=0)
print(out)
# print( list(df.groupby(['col'])) )