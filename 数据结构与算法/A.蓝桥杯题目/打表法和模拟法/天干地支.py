# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 15:43
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 天干地支.py
# @Software: PyCharm
#把余数，对应好
tg = ['geng','xin','ren','gui','jia','yi','bing','ding','wu','ji']
dz = ['shen','you','xu','hai','zi','chou','yin','mao','chen','si','wu','wei']
'''
天干是10个一轮回
year % 10 余数就是对应的天干

地支是12个一轮回
year % 12 余数就是对应的地支

'''
year = int(input())

print(tg[year%10],dz[year%12],sep = '')