import torch
import torchvision
from model import *
from torch import nn
from  torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor()
                                         , download=True)
test_data = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor()
                                         , download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：", train_data_size)
print("测试数据集的长度为：", test_data_size)


train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#  创建网络模型
model = Seq()

#  损失函数
loss_fn = nn.CrossEntropyLoss()