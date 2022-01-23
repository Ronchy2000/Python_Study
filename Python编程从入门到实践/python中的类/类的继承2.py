# -*- coding: utf-8 -*-
# @Time    : 2021/3/13 20:54
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 类的继承2.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time    : 2021/3/13 20:34
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : test.py
# @Software: PyCharm
class Apple():
    def __init__(self,product1='ipod1',product2='ipod2'):
        self.ipod1= product1
        self.ipod2= product2
    def output(self):
        print('product1=',self.ipod1)
        print('product2=',self.ipod2)

# ipod = Apple()
# ipod.output()
class Iphone(Apple):
    def __init__(self,product1='ipod1',product2='ipod2',price=0):
        super().__init__(product1='ipod1',product2='ipod2')
        self.price = price
    def output2(self):
        self.output()
        print(self.price)
iphone = Iphone()
iphone.output2()
