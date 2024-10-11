import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, Flatten, MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# dataset =torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
# 
# dataloader = DataLoader(dataset, batch_size=1)

class Seq(nn.Module):
    def __init__(self):
        super(Seq, self).__init__()
        self.model1 = nn.Sequential(  #  上面的顺序模型，写入Sequential中，减少代码量。 在forward函数中直接调用model1即可
            Conv2d(3, 32, 5, padding = 2),
            MaxPool2d(kernel_size = 2, stride=2),
            Conv2d(32, 32, 5, padding = 2),
            MaxPool2d(kernel_size = 2, stride=2),
            Conv2d(32, 64, 5, padding = 2),
            MaxPool2d(kernel_size = 2, stride=2),
            Flatten(start_dim=1,end_dim=-1),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

seq = Seq()
loss_crossentropu = nn.CrossEntropyLoss()
optim = torch.optim.SGD(seq.parameters(), lr=0.01)

if __name__ == '__main__':
    # print("seq_module:", seq)
    for epoch in range(10):
        running_loss = 0.0
        for data in dataloader:
            imgs, targets = data
            outputs = seq(imgs)
            result_loss = loss_crossentropu(outputs, targets)
            #  -----------
            optim.zero_grad() #  先清零梯度
            result_loss.backward()  # 对loss进行反向传播
            optim.step()
            # print("output:", outputs)
            # print("target:", targets)
            running_loss = running_loss + result_loss
        print("result_loss:", result_loss)

