# -*- coding: utf-8 -*-
# @Time    : 2021/4/11 19:03
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 自由方法_普通函数.py
# @Software: PyCharm
class Student:
    "This is a Student Class"

    count = 0
    def __init__(self,name):
        self.name = name
        Student.count += 1
#define 自由方法
    def foo():
        Student.count *= 100
        return Student.count
def foo2():
    Student.count /= 10
    return Student.count
print(Student.__doc__)
stud1 = Student("Bob")
#作用域名. 函数名
print(Student.foo())
print(foo2())