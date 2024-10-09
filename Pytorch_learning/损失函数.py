import torch
import torch.nn as nn
'''
Loss function:
1.计算实际输出和目标之间的差距
2. 为我们更新输出（反向传播）提供一定的依据
'''

inputs = torch.tensor([1,2,3], dtype=torch.float32)
targets = torch.tensor([1,2,5], dtype=torch.float32)

# inputs = torch.reshape(inputs,(1,1,1,3))
# targets = torch.reshape(targets,(1,1,1,3))

loss_mae = nn.L1Loss()  #  绝对误差
loss_mse = nn.MSELoss() #  均方误差
loss_crossentropy = nn.CrossEntropyLoss()  #交叉熵， 常用于 分类classification 问题

result_mae = loss_mae(inputs, targets)
result_mse = loss_mse(inputs, targets)
result_crossentropy = loss_crossentropy(inputs, targets)

print("result_mae:", result_mae)
print("result_mse:", result_mse)
print("result_crossentropy:", result_crossentropy)