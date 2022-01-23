# -*- coding: utf-8 -*-
# @Time    : 2021/3/8 19:44
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 输出100到999之间的水仙花数.py
# @Software: PyCharm
'''输出100到999之间的水仙花数.py
    举例：
    153=3*3*3+5*5*5+1*1*1
'''
for i in range(100,1000):   #注意range()函数[0,stop)不包括最后一个数
    unit_digit = i%10
    tens_digit = (i//10)%10
    hundreds_digit = i//100   #注意：整除！ //
    #print(hundreds_digit,tens_digit,unit_digit)
    if i == unit_digit**3+tens_digit**3+hundreds_digit**3:
        print(i)