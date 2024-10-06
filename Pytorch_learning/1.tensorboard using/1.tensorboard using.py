'''
打开tensorboard的方式：
1. cd到相应目录下
2. 输入命令tensorboard --logdir=1.tensorboard using
3. 其他：默认端口为http://localhost:6006/ ，更改指定具体端口：tensorboard --logdir=1.tensorboard using --ports=6007

# writer.add_image() #添加图像
# writer.add_scalar() # 添加标量记录形式


Link: 7.3.6连续变量可视化
https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%B8%83%E7%AB%A0/7.3%20%E4%BD%BF%E7%94%A8TensorBoard%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B.html
'''

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./logs")

for i in range(100):
    x = i
    y = x ** 2
    writer.add_scalar("x = i", x, i)
    writer.add_scalar("y = x**2", y, x)
print("write done!")
writer.close()




