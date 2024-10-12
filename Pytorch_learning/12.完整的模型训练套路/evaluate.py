import torchvision
import torch
import torch.nn  as nn
import torch.nn.functional as F1
from PIL import Image
from model import *


image_path = "../data/golden-retriever-tongue-out.jpg"

image = Image.open(image_path)
print("image",image)
image = image.convert("RGB")

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32))
                                               ,torchvision.transforms.ToTensor()])

image = transform(image)
print("image.shape",image.shape)


#  加载已训练好的模型参数
model = Seq()  #  创建模型
model.load_state_dict (torch.load("models/model.pth")) #  加载已保存的参数
# print("model:", model)

image = torch.reshape(image, (1,3,32,32))
model.eval()

with torch.no_grad():
    outputs = model(image)
    print("outputs:",outputs)
    print("outputs.argmax:",outputs.argmax(1))

##  分类5 就是狗。 正确。