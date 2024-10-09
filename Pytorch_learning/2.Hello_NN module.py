import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


if __name__ == '__main__':
    Hello_nn = Net()
    x = torch.tensor(1.0)
    output = Hello_nn(x)
    print("output",output)