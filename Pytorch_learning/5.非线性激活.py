import torch.nn as nn
from torch.nn import ReLU  #  非线性激活函数
import torch
import torch.nn.functional as F

'''
Torch文档可以查看非线性激活函数的细节：
https://pytorch.org/docs/stable/nn.html#pooling-layers

ReLU非线性激活函数：https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
y = 0, x < 0
y = x, x > 0
param: inplace: True，返回值将替代input，原input值丢失
              False，返回值为ouput，不会改变input的值

Sigmoid: https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid

Softmax: 
Applies the Softmax function to an n-dimensional input Tensor.
Rescales them so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.
'''

input = torch.tensor([[1, -0.5],
                      [-1, 3]], dtype = torch.float32)

input = torch.reshape(input,(-1, 1, 2, 2))
print(input.shape)

class Net_ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = ReLU(inplace = False)
    def forward(self, input):
        output = self.relu1(input)
        return output

if __name__ == '__main__':
    net = Net_ReLU()
    output = net(input)
    print("output", output)
