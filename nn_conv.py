import torch
import torch.nn.functional as F
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

kernal = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

# conv2d输入需要4个参数，batch_size(每次划分多少数据给神经网络)，通道为1(默认灰度图像)，高度5，宽度5
input = torch.reshape(input, (1,1,5,5))
kernal = torch.reshape(kernal, (1,1,3,3))

output1 = F.conv2d(input, kernal, stride=1)
print(output1)

output2 = F.conv2d(input, kernal, stride=1, padding=1)
print(output2)