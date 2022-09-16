# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 15:23
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : pyHSICLasso_feature_select.py
# @Software: PyCharm
# @Github_link: https://github.com/riken-aip/pyHSICLasso
'''
        皮尔逊 - 结果       HSIC
data2    Corner2
data3    Corner14
data4    Corner1
data5    Corner5
data6    Corner4
'''
from pyHSICLasso import HSICLasso
import pandas as pd
import numpy as np

hsic_lasso = HSICLasso()

df = pd.read_csv("")


hsic_lasso.input("timing1500x14.csv")
hsic_lasso.regression(5)
hsic_lasso.plot()
hsic_lasso.dump()

hsic_lasso.get_index()
hsic_lasso.get_index_score()
hsic_lasso.get_index_score()
hsic_lasso.get_features()

