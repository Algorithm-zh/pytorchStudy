import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



dataset = torchvision.datasets.CIFAR10('./data',train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=1)

class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model1 = Sequential(#用于组合
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self,x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
net = Net()
for data in dataloader:
    imgs, targets = data
    outputs = net(imgs)
    result_loss = loss(outputs,targets)
    result_loss.backward()#反向传播可以计算出各个节点的参数
    print(result_loss)