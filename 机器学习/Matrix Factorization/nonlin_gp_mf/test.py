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

#按组划分  就这样来
# id = [0,1]
# out = pd.DataFrame(columns=['row','col','value'])
# for i in id:
#     df_tmp = group.get_group(i)
#     out = pd.concat([out, df_tmp], axis=0)
# print(out)
# print( list(df.groupby(['col'])) )


#------------------------------------------
complementart_set_id = [i for i in range(14)]
print(complementart_set_id)