# -*- coding: utf-8 -*-
# @Time    : 2022/4/10 14:43
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 试题C：纸张尺寸.py
# @Software: PyCharm
ope = input()
A0 = [1189,841]
ans = A0
if ope == 'A0':
    for i in A0:
        print(i)
elif ope == 'A1':
    # ans = list(map(lambda x:x//2,A0))
    ans = sorted(ans,reverse=True)
    ans[0] = ans[0]//2
    ans.sort(reverse=True)
    for i in ans:
        print(i)
elif ope == 'A2':
    for i in range(0,2):
        ans = sorted(ans, reverse=True)
        ans[0] = ans[0] // 2
        ans.sort(reverse=True)

    for i in ans:
        print(i)

elif ope == 'A3':
    for i in range(0,3):
        ans = sorted(ans, reverse=True)
        ans[0] = ans[0] // 2
        ans.sort(reverse=True)

    for i in ans:
        print(i)
elif ope == 'A4':
    for i in range(0,4):
        ans = sorted(ans, reverse=True)
        ans[0] = ans[0] // 2
        ans.sort(reverse=True)
    for i in ans:
        print(i)
elif ope == 'A5':
    for i in range(0,5):
        ans = sorted(ans, reverse=True)
        ans[0] = ans[0] // 2
        ans.sort(reverse=True)
    for i in ans:
        print(i)
elif ope == 'A6':
    for i in range(0,6):
        ans = sorted(ans, reverse=True)
        ans[0] = ans[0] // 2
        ans.sort(reverse=True)
    for i in ans:
        print(i)
elif ope == 'A7':
    for i in range(0,7):
        ans = sorted(ans, reverse=True)
        ans[0] = ans[0] // 2
        ans.sort(reverse=True)
    for i in ans:
        print(i)
elif ope == 'A8':
    for i in range(0,8):
        ans = sorted(ans, reverse=True)
        ans[0] = ans[0] // 2
        ans.sort(reverse=True)
    for i in ans:
        print(i)
elif ope == 'A9':
    for i in range(0,9):
        ans = sorted(ans, reverse=True)
        ans[0] = ans[0] // 2
        ans.sort(reverse=True)
    for i in ans:
        print(i)

