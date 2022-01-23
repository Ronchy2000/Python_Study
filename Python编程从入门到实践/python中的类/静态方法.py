# -*- coding: utf-8 -*-
# @Time    : 2021/4/11 19:20
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 静态方法.py
# @Software: PyCharm
class Student:
    "This is a Student Class"
    count = 0
    def __init__(self,name,id):
        self.name = name
        self.id = id
        Student.count += 1

    @staticmethod
    def output():
        print("我可以输出信息")
        print("学生数量:",Student.count)


print(Student.__doc__)
stud1 = Student("Lu",2020)
stud2 = Student("Rone",2019)
stud3 = Student("Kitty",2018)
stud1.output()
print("\n")
Student.output()
