# -*- coding: utf-8 -*-
# @Time    : 2021/3/13 14:23
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 读文件.py
# @Software: PyCharm
file_path = 'f:\Readme.txt'
'''
#读整个文件
with open(file_path,encoding='utf-8') as rm:
    contents = rm.read()
    print(contents)
'''
#逐行读取文件
with open(file_path,'r',encoding='utf-8') as rm:
    for line in rm:
        print(line)