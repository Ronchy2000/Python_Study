# -*- coding: utf-8 -*-
# @Time    : 2021/4/10 20:54
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 二维数据写入处理.py
# @Software: PyCharm
ls = ['China',"America","Russia","Japan","Korea"]
with open('test3.txt', 'w', encoding='utf-8') as f:
    for item in ls:
        f.write(','.join(item) + '\n')  #以csv格式写入

        f.write(''.join(item) + '\n')