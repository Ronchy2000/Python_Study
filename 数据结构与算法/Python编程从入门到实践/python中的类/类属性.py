# -*- coding: utf-8 -*-
# @Time    : 2021/3/21 18:13
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 类属性.py
# @Software: PyCharm
class Student:
    name = "ronchy"
    age = 0
    def setting(self,name1,age1):
        self.name = name1
        self.age = age1
print(Student.name)
print(Student.age)

stud1= Student()
stud1.setting("Libai",21)

print(Student.name)
print(Student.age)