# -*- coding: utf-8 -*-
# @Time    : 2021/4/11 19:28
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 类的析构函数.py
# @Software: PyCharm
class Student:
    def __init__(self,name):
        self.name = name

    def __del__(self):
        print("格式化",self.name)

stud1 = Student("老王")
print(isinstance(stud1,Student))
print(issubclass(Student,object))
del stud1
