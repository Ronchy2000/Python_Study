import torch
import torchvision
from model import *
from torch import nn
from  torch.utils.data import DataLoader
import time
#  定义训练的设备
device = torch.device("cpu")  #  使用第一块显卡训练
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #  优先使用GPU训练


train_data = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor()
                                         , download=True)
test_data = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor()
                                         , download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：", train_data_size)
print("测试数据集的长度为：", test_data_size)


train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#  创建网络模型
model = Seq()
model= model.to(device)  #  也可以直接model.to(device)

#  损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

#  优化器
learning_rate = 1e-2    #表示：0.01   这种记法不用写一堆0了
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#  设置训练网络的一些参数
total_train_step = 0 #  记录训练的次数
total_test_step = 0  #  记录测试的次数
epoch = 10 #  训练的轮数

start_time = time.time()
for i in range(epoch):
    print("------第{}轮训练开始------".format(i+1))
    start_epoch_time = time.time()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = model(imgs)
        loss =  loss_fn(outputs, targets)
        #  使用优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()))  #loss 与 loss.item()  的区别：item是索引到tensor中的值。
    end_epoch_time = time.time()
    print("本轮训练结束：{}".format(end_epoch_time - start_epoch_time))

end_time = time.time()
print("训练结束，总耗时：{}".format(end_time - start_time))

#  保存模型
torch.save(model.state_dict(), "./models/model.pth")
print("模型已保存")