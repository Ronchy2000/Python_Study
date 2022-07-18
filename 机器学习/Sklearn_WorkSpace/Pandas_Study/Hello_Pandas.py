# -*- coding: utf-8 -*-
# @Time    : 2022/7/18 16:28
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Hello_Pandas.py
# @Software: PyCharm

import pandas as pd

pd.options.display.max_rows = 9999 #大于9999 ，就缩略显示 ，否则就全部显示

df = pd.read_csv('data.csv')
# print(df.to_string())  #Tip: use to_string() to print the entire DataFrame.
# print(df)

print(df.head(10))  #打印前  10  行
print("---------------------------------------")
print(df.tail(10))  #打印后  10  行








