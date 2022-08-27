# -*- coding: utf-8 -*-
# @Time    : 2021/4/10 20:36
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 二维数据读入处理.py
# @Software: PyCharm
#一般使用csv格式，即中间用逗号分隔
ls = []
with open('test2.txt', 'r') as f:
    for line in f:   #注意：这是二维列表，列表套列表
        line = line.replace("\n","")
        ls.append(line.split(","))
print(ls)
for l in ls: #print 列表
    print(l)

for l in ls:    #print 列表中的内容
    for content in l:
        print(content)