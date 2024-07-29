import torch
import torchvision
from PIL import Image
from model import *
image_path = '../imgs/dog.png'
image = Image.open(image_path)

# png格式图片有四个通道，rgb+透明通道，所以用转换变为三个通道
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)

model = torch.load('net_29.pth')
print(model)
image = torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))

