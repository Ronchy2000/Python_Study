# -*- coding: utf-8 -*-
# @Time    : 2022/9/10 11:38
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : test_np.py
# @Software: PyCharm

import numpy as np

x = [[1,2,3,4,5,5,1,-1],
[1,2,3,4,5,5,1,-1],[1,2,3,4,5,5,1,-1] ]

x = np.array(x).reshape(-1,1)
x = np.maximum(x, -x)  # numpy取绝对值方法
print(x)

x_morethan_3 = x[x>3]
print(len(x_morethan_3))