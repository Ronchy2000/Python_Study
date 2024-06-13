# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 13:47
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : test.py
# @Software: PyCharm

import pandas as pd
import torch

df = pd.read_csv("timing1500x14.csv")

xte = [i for i in range(1,1500,2)]

yte = df.loc[xte]['Corner1']
yte = df['Corner1']  #索引列
xte = torch.Tensor(xte).view(-1,1)

print(yte)