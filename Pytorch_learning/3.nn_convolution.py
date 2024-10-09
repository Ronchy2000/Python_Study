import torch
import torch.nn as nn
import torch.nn.functional as F

#input matrix
input = torch.tensor([[1,2,0,3,1],
                     [0,1,2,3,1],
                     [1,2,1,0,0],
                     [5,2,3,1,1],
                     [2,1,0,1,1]])
#Kernal
kernal = torch.tensor([[1,2,1],
                 [0,1,0],
                 [2,1,0]])

input = torch.reshape( input, (1,1,5,5))
kernal = torch.reshape(kernal,(1,1,3,3))

# print("input.shape:", input.shape)
# print("kernal.shape:", kernal.shape)
output = F.conv2d(input, kernal, stride = 1)
print("output:", output)

output2 = F.conv2d(input, kernal, stride = 2)  # 卷积步长为2
print("output2:", output2)

output3 = F.conv2d(input, kernal, stride = 1, padding= 1) #  四周多了一圈空白值
print("output3:", output3)