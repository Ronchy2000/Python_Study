# -*- coding: utf-8 -*-
# @Time    : 2022/2/27 14:30
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 打印尺子.py
# @Software: PyCharm

'''
1
1 2 1
1 2 1  3  1 2 1
1 2 1 3 1 2 1  4  1 2 1 3 1 2 1

'''
def ruler(n):
    if n ==1: #基准情形
        return 1
    else:
        tmp = ruler(n-1)
        return str(tmp) +' '+ str(n)+' ' + str(tmp)
print(ruler(3),end='\t')
print(ruler(4))