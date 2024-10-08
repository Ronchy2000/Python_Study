'''
打开tensorboard的方式：
1. cd到相应目录下
2. 输入命令tensorboard --logdir=logs
3. 其他：默认端口为http://localhost:6006/ ，更改指定具体端口：tensorboard --logdir=1.tensorboard using --ports=6007

# writer.add_image() #添加图像
# writer.add_scalar() # 添加标量记录形式


Link: 7.3.6连续变量可视化
https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%B8%83%E7%AB%A0/7.3%20%E4%BD%BF%E7%94%A8TensorBoard%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B.html
'''

from torch.utils.tensorboard import SummaryWriter


def record_in_2image():
    writer = SummaryWriter("./logs")

    for i in range(100):
        x = i
        y = x ** 2
        writer.add_scalar("x = i", x, i)
        writer.add_scalar("y = x**2", y, x)
        writer.add_scalar("y = x**2", y**2, x)  #不要在同一个writer中往一个标签里面写多个值，这里就是明显的问题，可以看效果
    print("write done!")
    writer.close()



#--------------------------------------
'''
如果想在同一张图中显示多个曲线，
则需要分别建立存放子路径（使用SummaryWriter指定路径即可自动创建，
但需要在tensorboard运行目录下），同时在add_scalar中修改曲线的标签使其一致即可：
'''
def record_in_1image():
    writer1 = SummaryWriter('./logs/x')
    writer2 = SummaryWriter('./logs/y')
    for i in range(500):
        x = i
        y = x**2
        writer1.add_scalar("same", x, i) #日志中记录x在第step i 的值
        writer2.add_scalar("same", y, i) #日志中记录y在第step i 的值
    print("write done!")
    writer1.close()
    writer2.close()

if __name__ == "__main__":
    # record_in_2image()
    record_in_1image()


