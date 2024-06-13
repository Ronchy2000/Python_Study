# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 17:43
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : plot_test.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np

# x = np.random.uniform(-1, 1, 4)
# y = np.random.uniform(-1, 1, 4)
# p1, = plt.plot([1, 2, 3])
# p2, = plt.plot([3, 2, 1])
# l1 = plt.legend([p2, p1], ["line 2", "line 1"], loc='upper left')
#
# p3 = plt.scatter(x[0:2], y[0:2], marker='D', color='r')
# p4 = plt.scatter(x[2:], y[2:], marker='D', color='g')
# # This removes l1 from the axes.
# plt.legend([p3, p4], ['label', 'label1'], loc='lower right', scatterpoints=1)
# # Add l1 as a separate artist to the axes
# plt.gca().add_artist(l1)
# plt.show()

# %%
x=np.linspace(1,10,20)
dy=np.random.rand(20)
y=np.sin(x)*3
print(x.shape,dy.shape,y.shape)
plt.errorbar(x,y,yerr=dy,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
#fmt :   'o' ',' '.' 'x' '+' 'v' '^' '<' '>' 's' 'd' 'p'
plt.show()
