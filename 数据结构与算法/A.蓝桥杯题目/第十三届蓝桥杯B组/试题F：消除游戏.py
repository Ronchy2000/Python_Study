# -*- coding: utf-8 -*-
# @Time    : 2022/4/10 16:15
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 试题F：消除游戏.py
# @Software: PyCharm

from collections import deque

s = input()
ls = list(s)
def judge(str):
    ans_index = []
    for i in range(1,len(str)-1):
        if str[i-1] !=str[i]:
            if str[i] == str[i+1]:
                #record i-1 ,i
                ans_index.append(i-1)
                ans_index.append(i)
            elif str[i] != str[i+1]:
                #record i-1,i,i+1
                ans_index.append(i - 1)
                ans_index.append(i)
                ans_index.append(i + 1)
        elif str[i-1] ==str[i]:
            if str[i] != str[i + 1]:
                #record i,i+1
                ans_index.append(i)
                ans_index.append(i + 1)
            elif str[i] == str[i + 1]:
                pass
    return ans_index


while len(ls) != 1:
    index = judge(ls)
    # print('index:',index)
    t = [i for i in range(0,len(ls))]
    the_real_answer_index = set(index)^set(t) #求全集的补集

    if len(the_real_answer_index) == 0:
        print("EMPTY")
        break
    else:
        ls.clear()
        for i in the_real_answer_index:
            ls.append(s[i])
            # print(s[i], end='')
        # print(ls)
if len(ls) == 1:
    print(ls[0])





