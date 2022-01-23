# -*- coding: utf-8 -*-
# @Time    : 2021/4/17 15:07
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 寻找2020.py
# @Software: PyCharm
with open('data2020.txt','r',encoding='utf-8') as f:
    # print(f.read())
    li = []
    coloum = []
    j = 0
    for line in f:
        li.append(list(line))
    for i in li:
        for j in li:
            coloum.append(i[i][j])
    print(coloum)
