# -*- coding: utf-8 -*-
# @Time    : 2022/3/22 13:19
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 模拟打表法.py
# @Software: PyCharm
#找约数--暴力枚举法！
def yueshu(a):
    ans = 0
    for i in range(1,a+1):
        if a % i ==0:
            ans += 1
    return ans

#print(yueshu(5))
for ii in range(1,400000):
    print(yueshu(ii),ii)
    if yueshu(ii) == 100:
        print(('ii:',ii))
        break
print('end!')
