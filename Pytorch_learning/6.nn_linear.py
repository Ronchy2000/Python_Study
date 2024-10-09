import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(196608, 10)  #输入的shape为196608，输出为10.降低了维度。
    def forward(self, input):
        return self.linear1(input)

net = Net()

if __name__ == '__main__':
    for data in dataloader:
        imgs, targets = data
        print(imgs.shape)
        input = torch.flatten(imgs)
        print("input.shape", input.shape)

        output = net(input)
        print("output.shape:",output.shape)
