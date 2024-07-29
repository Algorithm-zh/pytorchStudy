import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
dataset_transorm = transforms.Compose([
    transforms.ToTensor()
])
# root保存位置 train，true为训练集，false为测试集,download为true表示下载
# 点进去可以直接查看下载地址，可以使用迅雷下载之后放到对应位置
train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transorm, download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transorm, download=True)

writer = SummaryWriter("CIFAR10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_set',img,i)

writer.close()