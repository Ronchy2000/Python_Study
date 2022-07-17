# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 16:08
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : plot图像.py
# @Software: PyCharm
import matplotlib.pyplot as plt
x = [i for i in range(1,8) ]
y = [2.275,2.125,1.895,1.762,1.589,1.59,1.569]
plt.plot(x,y,ls="-",lw=2,label="测试集")
plt.legend()
plt.show()