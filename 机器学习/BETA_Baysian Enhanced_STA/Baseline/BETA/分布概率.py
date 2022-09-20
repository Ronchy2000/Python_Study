# -*- coding: utf-8 -*-
# @Time    : 2022/9/20 14:31
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 分布概率.py
# @Software: PyCharm

import scipy.stats
import numpy as np

#均值
mean = 23
#标准差
std = 8

#计算概率
prob = scipy.stats.norm(mean, std).pdf(0)
print("prob.", prob)