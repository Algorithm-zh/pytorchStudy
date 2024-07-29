import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备测试数据集
test_Data = torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor())
# 设置dataset设置数据集来源，设置每次从dataset中取的数据数量，设置是否打乱，num_workers=0一般不会报错，drop_last=True ：DataLoader 中的此设置会删除不完整的最后一批（如果它小于指定的批量大小）。这确保了训练期间处理的每个批次包含相同数量的样本。
test_loader = DataLoader(dataset=test_Data,batch_size=64,shuffle=False,num_workers=0,drop_last=True)

#target对应图片标签的索引
img, target = test_Data[0]
print(img)
print(target)

writer = SummaryWriter('dataloader')

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)#torch.Size([4, 3, 32, 32]) 4个图片，3个通道，32×32
        # print(targets)#tensor([0, 9, 1, 5]),将4个图片进行打包 4个图片的target
        writer.add_images('Epoch:{}'.format(epoch), imgs, step)
        step += 1

writer.close()