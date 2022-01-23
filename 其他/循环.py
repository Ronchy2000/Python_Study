#遍历列表
'''
a=['a.我','b.真','c.得','d.帅']
for a1 in a: #a1为临时变量
    print(a1.title()+",陆荣琦")
    print("我循环了")   #循环缩进
print("我没循环")
'''
#while 循环
active = True #使用标志
current_number = 1
while current_number <= 5:
    print(current_number)
    current_number += 1
    if current_number == 7:
        active = False