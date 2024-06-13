# -*- coding: utf-8 -*-
# @Time    : 2021/3/8 21:16
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 第一个类.py
# @Software: PyCharm

class Student:
    "This is a demo for Python class"
    print("Hello Demo Class")

print(Student.__doc__)

stud1 = Student()
#python中一切皆对象,已经开辟了内存空间！
print(id(Student))   #打印内存
print(type(Student)) #类对象的类型：type类型


print(Student)
#实例对象的类型:生成的类的类型,即Student类型
print(type(stud1))

