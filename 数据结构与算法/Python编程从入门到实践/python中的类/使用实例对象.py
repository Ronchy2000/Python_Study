# -*- coding: utf-8 -*-
# @Time    : 2021/3/8 21:52
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 使用实例对象.py
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
        print(self.name,self.sex,self.num)
#静态方法
    @staticmethod
    def out2():
        print("我使用了staticmethod进行修饰,我是静态方法")
#类方法
    @classmethod
    def out3(cls):
        print("我是类方法，我使用了classmethod进行修饰")

stu1 = Student('Ronchy','male','2019070727')
stu1.out1()
Student.eat(stu1) #与上一行一样
print("我直接访问姓名",stu1.name)
stu1.out2()
stu1.out3()
