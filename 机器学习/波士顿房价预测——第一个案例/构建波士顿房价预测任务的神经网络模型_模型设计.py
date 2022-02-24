import numpy as np
import json
#使用matplotlib将两个变量和对应的Loss作3D图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from 构建波士顿房价预测任务的神经网络模型_数据预处理 import load_data
# 获取数据
training_data, test_data = load_data()
x = training_data[:, :-1] #不包括最后一个
y = training_data[:, -1:] #只取最后一个
# # # 查看数据
# # print(training_data[0])
# # print('x[0]:', x[0])
# # print('y[0]:', y[0])
# w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]
# w = np.array(w).reshape([13, 1]) #转置矩阵
# # print(w)
# x1 = x[0]
# # print(x1)
# t= np.dot(x1,w) #dot()返回的是两个数组的点积
# # print(t)
#
# '''
#
# 完整的线性回归公式，还需要初始化偏移量b，同样随意赋初值-0.2
# 那么，线性回归模型的完整输出是z=t+b，
# 这个从特征和参数计算输出值的过程称为“前向计算”。
# '''
# b = -0.2
# z = t+b
# print('z:',z)

'''
将上述计算预测输出的过程以“类和对象”的方式来描述，
类成员变量有参数w和b。通过写一个forward函数（代表“前向计算”）完成上述从特征和参数到输出预测值的计算过程
代码如下所示。
'''
class Network():
    def __init__(self,num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights,1)
        self.b = 0.

    def forward(self,x):
        z = np.dot(x,self.w)+ self.b
        return z

#模型设计完成后，需要通过训练配置寻找模型的最优值
    #通过损失函数来衡量模型的好坏——方差
    def loss(self, z, y):
        #z为预测值矩阵，y为真实值矩阵。  z-y =error
        error = z - y
        cost = error **2
        cost = np.mean(cost)  #得到整体的loss
        return cost

net = Network(13)
# 此处可以一次性计算多个样本的预测值和损失函数


x1 = x[0:3]
y1 = y[0:3]
print('---------------利用类来描述---------------------')
z = net.forward(x1)
print('predict: ', z)
loss = net.loss(z, y1)
print('loss:', loss)

losses = []
#只画出参数w5和w9在区间[-160, 160]的曲线部分，以及包含损失函数的极值
w5 = np.arange(-160.0, 160.0, 1.0)
w9 = np.arange(-160.0, 160.0, 1.0)
losses = np.zeros([len(w5), len(w9)])

#计算设定区域内每个参数取值所对应的Loss
for i in range(len(w5)):
    for j in range(len(w9)): 
        net.w[5] = w5[i]
        net.w[9] = w9[j]
        z = net.forward(x)
        loss = net.loss(z, y)
        losses[i, j] = loss

##画图
# fig = plt.figure()
# ax = Axes3D(fig)
#
# w5, w9 = np.meshgrid(w5, w9)
#
# ax.plot_surface(w5, w9, losses, rstride=1, cstride=1, cmap='rainbow')
# plt.show()


x1 = x[0]
y1 = y[0]
z1 = net.forward(x1)
# print('x1{},shape{}'.format(x1,x1.shape))
# print('y1{},shape{}'.format(y1,y1.shape))
# print('z1{},shape{}'.format(z1,z1.shape))

#计算梯度
# gradient_w0 = (z1-y1)*x1[0]
# print('gradient_w0{}'.format(gradient_w0))

# gradient_w1 = (z1-y1)*x1[1]
# print('gradient_w1{}'.format(gradient_w1))
#不需要两层for循环
#num 广播机制！扩展参数的维度
# gradient_w = (z1-y1)*x1
# print('gradient_w{0},gradient.shape{1}'.format(gradient_w,gradient_w.shape))
# #扩展样本的维度
# x3samples = x[0:]
# y3samples = y[0:]
# z3samples = net.forward(x3samples)
#
# gradient_w = (z3samples-y3samples)*x3samples
# print('gradient_w{0},gradient{1}'.format(gradient_w,gradient_w.shape))


#广播机制可以把矩阵直接运算！
z = net.forward(x)
gradient_w = (z-y)*x
print('gradiient_w:',gradient_w)
print('未执行mean操作 gradient shape{0}'.format(gradient_w.shape))
print('w',net.w.shape)
#404个样本，对同一个参数的作用合并。
print('--------------执行mean操作--------------')
gradient_w = np.mean(gradient_w,axis = 0)
print('已执行mean操作 gradient shape{0}'.format(gradient_w.shape))
print('w',net.w.shape)
print(gradient_w)
print(net.w)
#发现 上矩阵 互为转置
gradient_w = gradient_w[:,np.newaxis]
print('newaxis',gradient_w.shape)
gradient_b = z-y
gradient_b = np.mean(gradient_b)
