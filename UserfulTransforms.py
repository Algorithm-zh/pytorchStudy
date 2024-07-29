from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('logs')
img = Image.open('data/train/ants_image/0013035.jpg')
# totensor使用，将图片转换为tensor类型
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)#往tensorboard中写图片

# 归一化
# 1.将一列数据变化到某个固定区间(范围)中，通常，这个区间是[0, 1] 或者（-1,1）之间的小数。
# 主要是为了数据处理方便提出来的，把数据映射到0～1范围之内处理，更加便捷快速
# 2.把有量纲表达式变成无量纲表达式，便于不同单位或量级的指标能够进行比较和加权。归一化是一种简化计算的方式，
# 即将有量纲的表达式，经过变换，化为无量纲的表达式，成为纯量。
# 图片是rgb三个通道，6个参数表示三个通道的平均值和标准差
# mean均值，std标准差
# output[channel] = (input[channel] - mean[channel]) / std[channel]
print(img_tensor[0][0][0])# tensor(0.3137),这是输入值
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0]) # tensor(-0.3725)，这是输出值
writer.add_image("Normalize", img_norm)


#等比列缩放Resize
#输入PILimg or Tensor 输出PILimg or Tensor
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img_tensor)
print(img_resize)
writer.add_image("Resize", img_resize)


#Compose，组合，可以将多种操作以参数形式传入组合在一起
#Compose()中的参数是一个列表，列表形式为[1,2,3]
#Compose需要的数据是transforms类型，所以参数为[transforms1,transforms2...]
#将图片短边缩放至512，长宽比保持不变，如果高度>宽度，则图像将被重新缩放为(size*高度/宽度，size)
trans_resize_2 = transforms.Resize(512)
#使用compose，首先第一个参数就是将图片进行缩放，然后第二个参数将图片转换为tensor类型
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)


#RandomCrop 随机裁剪
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('RandomCrop',img_crop,i)


writer.close()