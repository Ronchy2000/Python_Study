import torch
from torch import nn
from torch.nn import Conv2d, Flatten, MaxPool2d
from torch.utils.tensorboard import SummaryWriter


class Seq(nn.Module):
    def __init__(self):
        super(Seq, self).__init__()
        # self.conv1 = Conv2d(3, 32, 5, padding = 2)  #卷积层，注意通过输入和输出的大小来确定padding的大小。（与通道数区分开）
        # self.maxpool1 = MaxPool2d(kernel_size = 2, stride=2)  #池化层,步长默认是卷积核的尺寸
        # self.conv2 = Conv2d(32, 32, 5, padding = 2)
        # self.maxpool2 = MaxPool2d(kernel_size = 2, stride=2)
        # self.conv3 = Conv2d(32, 64, 5, padding = 2)
        # self.maxpool3 = MaxPool2d(kernel_size = 2, stride=2)
        #
        # self.flatten = Flatten(start_dim=1,end_dim=-1)
        # self.linear1 = nn.Linear(1024, 64)
        # self.linear2 = nn.Linear(64, 10)


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
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # print("x.shape:", x.shape)
        # x = self.linear1(x)
        # x = self.linear2(x)

        x = self.model1(x)

        return x


seq = Seq()

if __name__ == '__main__':
    # print("seq_module:", seq)
    input = torch.ones((64,3,32,32))
    # print("input:", input)
    output = seq(input)
    print("output:", output.shape)

    writer = SummaryWriter("./logs/logs_seq")
    writer.add_graph(seq, input)
    writer.close()