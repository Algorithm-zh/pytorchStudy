import imageio.plugins.opencv
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

#tensor
#通过transforms.ToTensor去解决两个问题
#1.transforms该如何使用(python)
#2.为什么需要Tensor数据类型
#包装了神经网络所需要的理论基础参数

img_path = 'data/train/ants_image/0013035.jpg'
img = Image.open(img_path)
writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()#返回的是ToTensor对象,相当于一个工具
tensor_img = tensor_trans(img)#传入图片返回tensor类型的图片
writer.add_image("Tensor_img",tensor_img)
writer.close()