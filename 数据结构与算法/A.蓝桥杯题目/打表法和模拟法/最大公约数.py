# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 15:21
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 最大公约数.py
# @Software: PyCharm

#递归求最大公约数  ,辗转相除法
def GCD(a,b):
    if a % b == 0:
        return b
    else:
        return GCD(b,a%b)

# print(GCD(10,20))
ans = 0
for a in range(1,2021):
    for b in range(1,2021):
        if GCD(a,b) == 1:
            ans += 1
print(ans)

#print(2481215)