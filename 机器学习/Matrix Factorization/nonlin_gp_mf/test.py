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

df = pd.read_csv("timing_flattern_3colx13.csv")

# print(df.columns.values)
print(df['row'])
# for i in df['row']:
#     print(i)

