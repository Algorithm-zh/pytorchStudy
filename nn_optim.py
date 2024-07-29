import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



dataset = torchvision.datasets.CIFAR10('./data',train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=64)

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
#优化器的使用
#1.定义优化器
optim = torch.optim.SGD(net.parameters(), lr=0.01,)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = net(imgs)
        result_loss = loss(outputs,targets)
        #2.对网络中每个参数的梯度清零
        optim.zero_grad()
        #3.调用损失函数的反向传播函数求出每个节点的梯度
        result_loss.backward()#反向传播可以计算出各个节点的参数对应梯度
        #4.调用step进行调优，使loss不断减少
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)