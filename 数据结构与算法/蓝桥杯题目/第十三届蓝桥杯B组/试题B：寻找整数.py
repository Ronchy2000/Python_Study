# -*- coding: utf-8 -*-
# @Time    : 2022/4/10 14:15
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 试题B：寻找整数.py
# @Software: PyCharm
'''
注意到是11和17的倍数，很快很简单
'''
ls = [11*17*i for i in range(1,100000,2)]

print(ls)
ans = []
for i in ls:
    if i % 3 == 2 and i % 5 == 4and i % 6 == 5 and i % 7 == 4 and i % 8 == 1 and i % 9 == 2 and i % 10 == 9 and i%12 == 5 and i%13 ==10:
         print(i)
         ans.append(i)
         continue
print("Finished1!")
for i in ans:
    if i%49 == 46 and i%48 == 41 and i%47 == 5 and i%46==15:
        print(i)
print("Finished2!")

'''

ans = 2443529

'''