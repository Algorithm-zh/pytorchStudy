import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    net = Net()
    #测试网络正确性，创建一个输入的尺寸，看输出的尺寸是否是我们想要的
    input = torch.ones((64,3,32,32))
    output = net(input)
    print(output.shape)