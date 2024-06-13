# -*- coding: utf-8 -*-
# @Time    : 2022/4/1 18:35
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : LIS问题.py
# @Software: PyCharm
'''
最长不降子序列问题
长度为n的序列
找到最长递增子序列的长度（以及序列值
'''
def bag(n, c, w, v):
    """
    测试数据：
    n = 6  物品的数量，
    c = 10 书包能承受的重量，
    w = [2, 2, 3, 1, 5, 2] 每个物品的重量，
    v = [2, 3, 1, 5, 4, 3] 每个物品的价值
    """
