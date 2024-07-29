import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10('./data',train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=64)

class Net(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = Linear(196608,10)
    def forward(self, input):
        output = self.linear1(input)
        return output

net = Net()

for data in dataloader:
    imgs, target = data
    output = torch.flatten(imgs)#直接摊平
    # output = torch.reshape(imgs,(1,1,1,-1))
    print(output.shape)
    output = net(output)
    print(output.shape)
