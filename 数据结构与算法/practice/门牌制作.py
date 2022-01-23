# -*- coding: utf-8 -*-
# @Time    : 2021/4/17 14:10
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 门牌制作.py
# @Software: PyCharm

list = list(range(1,2021))
# list.append(2222)
print(list)
def count(list):
    count = 0
    for i in list:
        for j in str(i):
            if j == '2':
                count += 1
    return count

print(count(list))