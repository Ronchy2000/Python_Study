import torch
import torchvision

#  保存方式1，加载模型
# 注意，使用方式1，一定要 import NN相对应的类，就不会报错了。
# model1 = torch.load("vgg16_method1.pth")
# print("model", model1)

#  保存方式2，加载模型
# #  这两行可以查看保存的模型参数
# model2_param = torch.load("vgg16_method2.pth")
# print("model", model2_param)


vgg16 = torchvision.models.vgg16(pretrained = False) #  先创建模型
vgg16.load_state_dict(torch.load("vgg16_method2.pth")) # 再加载参数
print("vgg16", vgg16)

