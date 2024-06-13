# -*- coding: utf-8 -*-
# @Time    : 2021/3/8 21:25
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 定义Python中的类.py
# @Software: PyCharm
class Student:
#类属性：
    name = ""
    sex = ""
    num = ""
#在类之外定义的称为函数,类内定义的称为方法
#初始化方法:
    def __init__(self,name,sex,num):
        self.name = name
        self.sex = sex
        self.num = num
#实例方法：
    def out1(self):
        print(name,sex,num)
#静态方法
    @staticmethod
    def out2():
        print("我使用了staticmethod进行修饰,我是静态方法")
#类方法
    @classmethod
    def out3(cls):
        print("我是类方法，我使用了classmethod进行修饰")

stu1 = Student('Ronchy','male','2019070727')
print(id(stu1))  #利用计算器计算;将id(十进制)改为十六进制,等于stud1的地址
print(type(stu1))
print(stu1)  #输出内存地址
print('----------------------------------')

print(id(Student))   #打印内存
print(type(Student))
print(Student