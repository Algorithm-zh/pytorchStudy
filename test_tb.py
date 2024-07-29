from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import  Image
#会将事件写到事件文件logs里
writer = SummaryWriter("logs")

image_path = "data/train/ants_image/0013035.jpg"
image_PIL = Image.open(image_path)
image_Array = np.array(image_PIL)

writer.add_image("test",image_Array,dataformats='HWC')
#img读取需要的数据类型为img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
#正常读出来是<class 'PIL.JpegImagePlugin.JpegImageFile'>类型，不符合要求
#常用的方式是
# 1.使用opencv读取numpy类型
# 2.使用numpy的np.array()方法转化为numpy类型
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close()
# tensorboard --logdir=事件文件名 (事件文件名不加引号)--port=端口号 使用该命令可以打开tensorboard