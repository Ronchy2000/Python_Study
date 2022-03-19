# -*- coding: utf-8 -*-
# @Time    : 2022/3/16 16:15
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 万万没想到之聪明的编辑.py
# @Software: PyCharm
# https://www.nowcoder.com/test/question/42852fd7045c442192fa89404ab42e92?pid=16516564&tid=53689888

'''
我叫王大锤，是一家出版社的编辑。我负责校对投稿来的英文稿件，这份工作非常烦人，因为每天都要去修正无数的拼写错误。但是，优秀的人总能在平凡的工作中发现真理。我发现一个发现拼写错误的捷径：

1. 三个同样的字母连在一起，一定是拼写错误，去掉一个的就好啦：比如 helllo -> hello
2. 两对一样的字母（AABB型）连在一起，一定是拼写错误，去掉第二对的一个字母就好啦：比如 helloo -> hello
3. 上面的规则优先“从左到右”匹配，即如果是AABBCC，虽然AABB和BBCC都是错误拼写，应该优先考虑修复AABB，结果为AABCC

我特喵是个天才！我在蓝翔学过挖掘机和程序设计，按照这个原理写了一个自动校对器，工作效率从此起飞。用不了多久，我就会出任CEO，当上董事长，迎娶白富美，走上人生巅峰，想想都有点小激动呢！
……
万万没想到，我被开除了，临走时老板对我说： “做人做事要兢兢业业、勤勤恳恳、本本分分，人要是行，干一行行一行。一行行行行行；要是不行，干一行不行一行，一行不行行行不行。” 我现在整个人红红火火恍恍惚惚的……

请听题：请实现大锤的自动校对程序
'''
from collections import deque
cnt = 1
s= 'llooa'
str = deque(s) #离散
truth = []

# for i in str:
#     print(i)

def find_3conti(tmp):
    global cnt,truth
    while len(tmp) != 1:
        a = tmp.popleft()
        if a == tmp[0]:
            cnt += 1
            truth.append(tmp[0])
        elif a != tmp[0]:
            truth.append(a)
        if cnt == 3:
            cnt = 1
            truth.pop()
            print('yes')
    truth.append(str[-1])
    print(truth)

def judge_repeat(tmp):
    global cnt, truth
    while len(tmp) != 4:
        a = tmp.popleft()
        truth.append(a)
        if a == tmp[0]:
            truth.append(tmp.popleft())
            b = tmp.popleft()
            #判断第三个元素
            if b == tmp[0]:
                print("no")

    print(truth)





if __name__ == "__main__":
    M = int(input())
    while M:
        M -= 1
        #str = input()
        #str1 = find_3conti(str)
        print(str)
        judge_repeat(str)



