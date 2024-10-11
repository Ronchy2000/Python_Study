import torch
from torchvision.models import vgg16

vgg16_model = vgg16(pretrained=False)

# 保存方式1: 模型结构 + 模型参数
torch.save(vgg16_model, 'vgg16_method1.pth')
print('save success')

# 保存方式2：模型参数（官方推荐）
torch.save(vgg16_model.state_dict(), "vgg16_method2.pth")