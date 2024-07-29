import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./data',train=False,download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


#为了满足maxpool输入的要求，N,C,H,W
# input = torch.reshape(input,(-1,1,5,5))

class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

net = Net()

step = 0
writer = SummaryWriter('logs_maxpool')
for data in dataloader:
    imgs,targets = data
    writer.add_images('input', imgs, step)
    output = net(imgs)#最大池化不会改变通道数，原来有3维，池化后还是3维
    writer.add_images('output',output,step)
    step += 1
writer.close()