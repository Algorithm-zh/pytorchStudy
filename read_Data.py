from torch.utils.data import Dataset
from PIL import Image
import os
class MyData(Dataset):
    def __init__(self,root_dir,label_dir):#用self相当于变成全局变量，就可以在其他函数访问
        self.root_dir = root_dir #root地址，一般到train
        self.label_dir = label_dir #标签名
        self.path = os.path.join(self.root_dir,self.label_dir)#将地址相连，这种方式能避免出错
        self.img_path = os.listdir(self.path)#以列表形式返回图片的名字的合集
        print(self.path)


    def __getitem__(self, idx):
        img_name = self.img_path[idx]#图片名字
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)#将地址和标签和图片名相连得出相对地址
        img = Image.open(img_item_path)#打开图片，是PIL格式
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)

train_dataset = ants_dataset + bees_dataset#这种情况可能用于数据集不够的情况，将仿造的数据集和真实的数据集结合


