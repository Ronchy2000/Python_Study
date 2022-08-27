# -*- coding: utf-8 -*-
# @Time    : 2022/8/27 16:08
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Hello_MLP.py
# @Software: PyCharm
# 利用api实现 Multi-layer Perceptron (MLP) 多层感知机

import torch
from torch import nn
from d2l import torch as d2l

# 模型
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ =="__main__":
    net.apply(init_weights)
    # 训练过程
    batch_size, lr, num_epochs = 256, 0.1, 5
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
