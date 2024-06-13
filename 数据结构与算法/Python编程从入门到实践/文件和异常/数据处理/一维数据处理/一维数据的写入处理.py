# -*- coding: utf-8 -*-
# @Time    : 2021/4/10 20:23
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 一维数据的写入处理.py
# @Software: PyCharm
ls = ['China','America','Japan']
with open('test.txt', 'w') as f:
    f.write(' '.join(ls))  #空格方式写入
    f.write('\n')
    f.write(','.join(ls))  #逗号方式写入
    f.write('\n')
    f.write('\\n')
    f.write('$'.join(ls))