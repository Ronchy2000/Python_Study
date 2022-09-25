# -*- coding: utf-8 -*-
# @Time    : 2022/9/20 14:31
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 分布概率.py
# @Software: PyCharm

import math
import scipy.stats
import numpy as np
#使用-累积分布函数

def prob_not_violation(X, mean, covariance):
    # X = 0 # x<X  的概率
    # mean = np.array([0,1,2,3,4,5])
    # covariance = np.array([0.5, 1, 1.5, 2, 1.5, 2])

    # prob of  x > X
    # prob = 1 - scipy.stats.norm(mean, covariance).cdf(X)
    # prob of  x < X
    prob = scipy.stats.norm(mean, covariance).cdf(X)
    # print(prob)
    return prob
