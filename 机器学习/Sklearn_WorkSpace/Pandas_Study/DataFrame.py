# -*- coding: utf-8 -*-
# @Time    : 2022/7/18 18:12
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : DataFrame.py
# @Software: PyCharm

import pandas as pd
data = {
    "calories":[420,380,390],
    "duration":[50,40,45]
}
myvar = pd.DataFrame(data)
print(myvar)
print('--------------------------------')
print("myvar.loc[0]:\n",myvar.loc[0])
print('--------------------------------')

#Add a list of names to give each row a name:
df = pd.DataFrame(data, index = ["day1", "day2", "day3"])







