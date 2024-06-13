# -*- coding: utf-8 -*-
# @Time    : 2021/4/11 18:47
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 类方法2.py
# @Software: PyCharm
class Student:
    "This is a Student class"
    count = 0
    def __init__(self,name,id):
        self.name = name
        self.id = id
        Student.count += 1
    @classmethod
    def getcount(cls):
        s = "零一二三四五六七八九多"
        return s[Student.count]

print(Student.__doc__)
stud1 = Student("ronchy",2019070727)

print("学生数量：")
print(stud1.getcount())
print(Student.getcount())
