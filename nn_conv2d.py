import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=64)

class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = Conv2d(3,6,3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

net = Net()

step = 0
writer = SummaryWriter('logs')
for data in dataloader:
    imgs,targets = data
    output = net(imgs)
    writer.add_images('input',imgs,step)
    # 直接输出会报错，设置的6个channel，用reshape改为3个通道，-1让它自动设置bach_size批次
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images('output',output,step)
    step += 1