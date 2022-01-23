
#long long ago

# from math import sqrt  #导入函数
#
#
# print("Hello Python World!")
# print(10//3)   ##取整
# print("4的平方根:",sqrt(4))
# #注意满足条件及格式
# if 1==2:print('One equals two')
# if 1==1:print('One equals one')
#
# #幂运算
# print(2**4)
# print("Function :",pow(2,4)) #效果相同
#
# #导入模块
# import math
# print("floor函数 floor(32.996):",math.floor(32.996))  #以module.function来使用模块中的函数
# print(int(32.996))
# #2020.10.6
a=['name','phone_num','QQ_num']
print("a",a)                             #  -1引用最后一个
a.append("Ronchy")                   #append()添加到末尾
# del a[1]
print("a=",a)

popped_a=a.pop()        #pop()弹出最后个元素，即栈顶元素，将其与原栈分离，但仍可使用
print("a=", a)          #可指定弹出某一项->  a.pop(1)
print("popped_a=",popped_a)

# print("a=",a)
# a.remove("name")        #remove("")删除指定的元素
# print("a=",a)

a.insert(-1,"name!")       #插入元素,剩余的元素右移
#类比 a.append("name!")
print("a=",a)
