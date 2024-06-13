# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 23:01
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 二叉树结点问题.py
# @Software: PyCharm
'''
　一棵二叉树有2021个结点。该树满足任意结点的左子树结点个数和右子树的结点个数之差最多为1。
　　定义根结点的深度为0，子结点的深度比父结点深度多1。
　　请问，树中深度最大的结点的深度最大可能是多少？
答案提交
　　这是一道结果填空的题，你只需要算出结果后提交即可。本题的结果为一个整数，在提交答案时只填写这个整数，填写多余的内容将无法得分。


'''
###
#该题类似于平衡二叉树
#
cnt = 0
num = 2
sum = 0
for i in range(0,15):

    if sum > 2021:
        break
    cnt += 1
    sum += (num**i)
    print('每层个数',num**i)
print(sum)
#从0开始，所以要减1
print('深度',cnt-1)