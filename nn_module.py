from torch import nn
import torch

# 自己搭建一个神经网络，使用nn包
class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = input + 1
        return output

net = Net()
x = torch.tensor(1.0)
output = net(x)
print(output)