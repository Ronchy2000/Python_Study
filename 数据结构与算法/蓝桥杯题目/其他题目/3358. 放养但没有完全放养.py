# -*- coding: utf-8 -*-
# @Time    : 2022/3/15 15:40
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 3358. 放养但没有完全放养.py
# @Software: PyCharm
from collections import deque
#alpbelt = 'abcdefdhijklmnopqrstuvwxyz'
alpbelt_list = deque(alpbelt)
# print(alpbelt_list)#强制类型转换
cnt = 0


if __name__ == '__main__':
    a = input()
    a_list = deque(a)
    tmp = a_list.popleft()
    while a_list:
        for i in alpbelt_list:
            if i == tmp:
                tmp = a_list.popleft()
                print('right')
                cnt +=1

    print(cnt)




