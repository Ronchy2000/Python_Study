# -*- coding: utf-8 -*-
# @Time    : 2021/1/15 14:18
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : dict.py
# @Software: PyCharm

dict = {'apple1':'apple1','apple2':'apple2','apple3':'apple3','apple4':4}
print(dict['apple1'],'\n')
print('apple3:',dict['apple3'])
#修改
dict['apple3']='Hello World!'
print('apple3:','dict[apple3]','\n')
#三种查找方式
print('apple5是否能查到？','apple5' in dict)  #不存在apple5，查不到，返回false
print(dict.get('apple5'))  #查不到,默认返回None
print(dict.get('apple5',-1)) #查不到,返回指定的值: -1(此处)
