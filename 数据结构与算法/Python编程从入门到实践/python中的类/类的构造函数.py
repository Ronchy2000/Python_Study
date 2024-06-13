# -*- coding: utf-8 -*-
# @Time    : 2021/4/11 18:00
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 类的构造函数.py
# @Software: PyCharm
class Student:
    count = 0
    def __init__(self,name,id):
        self.name = name
        self.id = id
        # print(name)
        # print(id)
        Student.count += 1

#如果类对象只包含一个函数，则Run时没有执行信息

#2个实例对象使用init函数
stud1 = Student("Ronchy",2019070727)
stud2 = Student("Lu",2019070727)

'''类属性，实例属性'''
print("总数:",Student.count)
print(stud1.name,stud1.id)
print(stud2.name,stud2.id)

