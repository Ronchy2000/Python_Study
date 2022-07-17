# -*- coding: utf-8 -*-
# @Time    : 2022/7/15 18:38
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 数据生成.py
# @Software: PyCharm

import pandas as pd
import numpy as np

path = []
corner = []
for i in range(15):
    corner.append( str("corner"+str(i+1)) )
for i in range(1405):
    path.append(str("path"+str(i+1)))

df = pd.read_csv("工作簿1.csv",index_col=None)
print(df)
# data = np.random.randint(0,2,1405*15).reshape(1405,15)
# df = pd.DataFrame(data,columns=corner,index=path)

# df.to_csv('工作簿1.csv', sep=',', encoding='utf-8')



