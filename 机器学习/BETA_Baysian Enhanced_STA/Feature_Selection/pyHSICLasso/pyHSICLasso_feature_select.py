# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 15:23
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : pyHSICLasso_feature_select.py
# @Software: PyCharm
# @Github_link: https://github.com/riken-aip/pyHSICLasso

from pyHSICLasso import HSICLasso
hsic_lasso = HSICLasso()

hsic_lasso.input("timing1500x14.csv")
hsic_lasso.regression(5)
hsic_lasso.plot()
hsic_lasso.dump()

hsic_lasso.get_index()
hsic_lasso.get_index_score()
hsic_lasso.get_index_score()
hsic_lasso.get_features()

hsic_lasso.save_param() #Save selected features and its neighbors