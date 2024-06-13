# -*- coding: utf-8 -*-
# @Time    : 2022/4/19 14:21
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : tmp.py
# @Software: PyCharm
max_value = 0
cnt_ = 0
cnt_max = 0

a = [6,7,10,16,12,5,11,10,7,9]
b= [ 5,9, 7, 6, 5,9, 9, 7,9,8]

for i in range(0,len(a)):
    if a[i]>b[i]:
        max_value = (a[i]-b[i]) if (a[i]-b[i])>max_value else max_value
        cnt_ +=1
    else:
        cnt_ = 0
    if cnt_ > cnt_max:
        cnt_max = cnt_
print('cnt_max:',cnt_max,'max_value:',max_value)
