# -*- coding: utf-8 -*-
# @Time    : 2021/3/7 17:15
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 返回字典.py
# @Software: PyCharm
def build_person(first_name,last_name):
    person = {'first':first_name,'last':last_name}
    return person

musician = build_person('jimi','hendrix')
print(musician)