import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

#准备数据集
train_data = torchvision.datasets.CIFAR10('../data',train=True,transform=torchvision.transforms.ToTensor(),download=True)

test_data = torchvision.datasets.CIFAR10('../data',train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print(f'训练数据集的长度为:{train_data_size}')
print(f'测试数据集的长度为:{test_data_size}')

# dataloader加载数据集
train_dataloader = DataLoader(dataset=train_data,batch_size=64)
test_dataloader = DataLoader(dataset=test_data,batch_size=64)


# 网络模型 损失函数 数据加cuda


#创建网络模型
net = Net()
if torch.cuda.is_available():
    net = net.cuda()
#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.cuda()

#优化器 ,随机梯度下降法
learning_rate = 1e-2
optimizer = torch.optim.SGD(net.parameters(),learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

#添加tensorboard
writer = SummaryWriter('../logs_train')

for i in range(epoch):
    print(f'第{i + 1}轮训练开始')

    #训练步骤开始
    net.train()#只有网络中有某些层才需要调用
    for data in train_dataloader:
        imgs,targets = data

        imgs = imgs.cuda()
        targets = targets.cuda()

        outputs = net(imgs)
        loss = loss_fn(outputs, targets)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f'训练次数：{total_train_step},Loss：{loss.item()}')
            writer.add_scalar('train_loss',loss.item(),total_train_step)

    #测试步骤开始
    net.eval()#只有网络中有某些层才需要调用
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():#测试不需要梯度不需要优化
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            # 1表示横向对比，找出最大值的下标
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print(f'整体测试集上的Loss：{total_test_loss}')
    print(f'整体测试集上的正确率：{total_accuracy/test_data_size}')
    writer.add_scalar('test_loss',total_test_loss,total_test_step)
    writer.add_scalar('test_accuracy',total_accuracy/test_data_size,total_test_step)
    total_test_step += 1

    torch.save(net,f'net_{i}.pth')
    #保存方式二：torch.save(net.state_dict(),f'net_{i}')
    print('模型已保存')
writer.close()
