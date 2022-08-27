# -*- coding: utf-8 -*-
# @Time    : 2021/4/11 18:27
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 实例方法.py
# @Software: PyCharm
class Student:
    "This is a Student Class"
    count = 0
    def __init__(self,name,sex):
        self.name = name
        self.sex = sex
        Student.count += 1
    def output(self):
        print("Name:",self.name)
        print("Sex:",self.sex)


print(Student.__doc__)

stud1 = Student("Ronchy","Male")
print("I'm student" + str(Student.count))
stud1.output()

stud2 = Student("LU","Female")
print("I'm student" + str(Student.count))
stud2.output()
