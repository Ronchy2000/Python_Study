import torch
import torch.nn as nn
import torch.nn.functional as F
'''
池化的目的是缩小数据量
默认的池化 stride = kernel size
将5x5的矩阵，变成2x2
'''

class Max_pooling(nn.Module):
    def __init__(self):
        super(Max_pooling, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size = 3, ceil_mode= True) #默认的池化 stride = kernel_size

    def forward(self, input):
        output = self.maxpooling(input)
        return output

if __name__ == "__main__":
    # input matrix
    input = torch.tensor([[1, 2, 0, 3, 1],
                          [0, 1, 2, 3, 1],
                          [1, 2, 1, 0, 0],
                          [5, 2, 3, 1, 1],
                          [2, 1, 0, 1, 1]], dtype= torch.float32)

    input = torch.reshape(input, (1, 1, 5, 5))
    print("input.shape:{}\n".format(input.shape))
    maxpooling = Max_pooling()
    output = maxpooling(input)
    print("output:", output)

