# -*- coding: utf-8 -*-
# @Time    : 2021/3/21 19:08
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 类方法.py
# @Software: PyCharm
class Student:
    name = "ronchy"     #类属性
    age = 0

    #初始化方法
    def __init__(self,name,age):
        self.name = name      #self.name称为实体属性,将局部变量name的值赋给实体属性name
        self.age = age

    # 实例方法
    def output(self):
        print(self.name)
        print(self.age)
    #静态方法
    @staticmethod
    def method():
        print("我使用了Staticmethod")
    #类方法
    @classmethod
    def cm(cls):
        print("我使用了classmethod")