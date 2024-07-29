import torchvision
import torch
from torch import nn
from model_save import *

#方式1：保存方式1，加载模型
model1 = torch.load('vgg16_method1.pth')
print(model1)

#方式2:加载模型，字典格式
# 以模型架构方式显示
# vgg16 = torchvision.models.vgg16(prettrained=False)
# vgg16.load_state_dict(torch.load('vgg16_method2.pth'))
# 以字典格式显示
model2 = torch.load('vgg16_method2.pth')
print(model2)

#陷阱
# 方法一这样会报错
# model = torch.load('net_method1.pth')

#必须把网络重新创建一次，或者import过来
# class Net(nn .Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.conv1 = nn.Conv2d(3,64,3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

model = torch.load('net_method1.pth')
print(model)