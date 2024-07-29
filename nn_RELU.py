import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,-0.5],
                      [-1,3]])
input = torch.reshape(input,(-1,1,2,2))

dataset = torchvision.datasets.CIFAR10('./data',train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)
class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # inplace=True的时候，input的值会改变，false时不会改变
        self.relu1 = ReLU()
        self.sigmod1 = Sigmoid()#非线性变换，如果在神经网络中使用线性激活函数，那么整个网络将只能表示线性变换，线性变换的组合仍然是线性的，这将限制网络的表示能力，非线性变换的主要目的是在网络中引入非线性特征，提高模型的泛化能力

    def forward(self,input):
        output = self.sigmod1(input)
        return output

net = Net()

writer = SummaryWriter('logs_relu')
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images('inputs',imgs,global_step=step)
    output = net(imgs)
    writer.add_images('output',output,global_step=step)
    step += 1
writer.close()

